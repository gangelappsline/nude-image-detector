"""Application factory for the Nude Image Detector API.

``create_app()`` is the single entry point used by the dev server, gunicorn,
the Docker image and the test-suite, so every environment gets exactly the same
middleware, error contract and engine wiring.
"""

from __future__ import annotations

import logging
import time
import uuid

from flask import Flask, g, jsonify, request
from werkzeug.middleware.proxy_fix import ProxyFix

from .api.errors import register_error_handlers
from .api.routes import api_bp
from .config import Settings
from .core.engine import build_engine
from .logging_conf import setup_logging
from .security import RateLimiter, enforce_request_limits, request_cost, require_auth
from .service import AnalysisService
from .ui import ui_bp

logger = logging.getLogger(__name__)

#: Endpoints that must stay reachable without credentials (probes, docs).
PUBLIC_ENDPOINTS = frozenset(
    {
        "api.health",
        "api.ready",
        "api.info",
        "api.labels",
        "api.openapi",
        "ui.index",
        "ui.docs",
        "static",
    }
)

DEFAULT_CSP = (
    "default-src 'self'; "
    "script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "form-action 'self'; "
    "frame-ancestors *; "
    "base-uri 'self'"
)


def create_app(settings: Settings | None = None, *, testing: bool = False) -> Flask:
    """Build and configure the Flask application."""
    settings = settings or Settings.from_env()
    setup_logging(settings.log_level, json_output=settings.log_json)

    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config["NID_SETTINGS"] = settings
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length
    app.config["PROPAGATE_EXCEPTIONS"] = True
    app.config["TESTING"] = testing
    app.json.sort_keys = False
    app.json.ensure_ascii = False  # keep Spanish text readable in payloads
    app.url_map.strict_slashes = False

    # --- engine + service ------------------------------------------------ #
    engine = build_engine(settings)
    service = AnalysisService(settings, engine)
    limiter = RateLimiter(settings)

    app.extensions["nid_settings"] = settings
    app.extensions["nid_engine"] = engine
    app.extensions["nid_service"] = service
    app.extensions["nid_limiter"] = limiter

    if settings.engine != "disabled" and settings.load_model_at_startup:
        try:
            engine.load()
        except Exception:  # pragma: no cover - startup must not die here
            logger.exception(
                "No se pudo cargar el modelo al arrancar; /ready devolverá 503 "
                "hasta que la carga tenga éxito."
            )
    if settings.warmup and engine.is_ready:
        try:
            engine.warmup()
        except Exception:  # pragma: no cover
            logger.exception("warmup_fallido")

    # --- blueprints ------------------------------------------------------ #
    app.register_blueprint(api_bp)
    app.register_blueprint(ui_bp)
    register_error_handlers(app, settings)

    if settings.expose_openapi:
        from .openapi import build_openapi_spec

        spec = build_openapi_spec(settings)

        @app.get("/openapi.json", endpoint="openapi")
        def openapi_spec():  # pragma: no cover - trivial
            return jsonify(spec)

        PUBLIC_ENDPOINTS_LOCAL = PUBLIC_ENDPOINTS | {"openapi"}
    else:
        PUBLIC_ENDPOINTS_LOCAL = PUBLIC_ENDPOINTS

    # --- middleware ------------------------------------------------------ #
    @app.before_request
    def _before() -> None:
        g.request_id = _request_id()
        g.started_at = time.monotonic()
        g.json_payload = None

        endpoint = request.endpoint or ""
        if endpoint in PUBLIC_ENDPOINTS_LOCAL:
            return

        require_auth(request, settings)
        if settings.rate_limit_enabled:
            decision = enforce_request_limits(
                request, settings, limiter, cost=request_cost(request)
            )
            g.rate_limit = decision

    @app.after_request
    def _after(response):
        request_id = getattr(g, "request_id", None)
        if request_id:
            response.headers.setdefault("X-Request-ID", request_id)

        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Content-Security-Policy", DEFAULT_CSP)
        if settings.cors_origin:
            response.headers.setdefault("Access-Control-Allow-Origin", settings.cors_origin)
            response.headers.setdefault("Vary", "Origin")
            response.headers.setdefault(
                "Access-Control-Allow-Headers", "Content-Type, X-API-Key, Authorization, X-Request-ID"
            )
            response.headers.setdefault("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")

        endpoint = request.endpoint or ""
        if endpoint.startswith("api.") and endpoint not in {"api.health", "api.ready"}:
            # Analysis responses can embed a base64 image: never cache them.
            response.headers.setdefault("Cache-Control", "no-store")
        elif endpoint in {"api.health", "api.ready"}:
            response.headers.setdefault("Cache-Control", "no-store")

        decision = getattr(g, "rate_limit", None)
        if decision is not None:
            for header, value in decision.headers().items():
                response.headers.setdefault(header, value)

        started = getattr(g, "started_at", None)
        if started is not None and endpoint not in {"api.health", "api.ready", "static"}:
            logger.info(
                "http_request method=%s path=%s status=%d ms=%.1f",
                request.method,
                request.path,
                response.status_code,
                (time.monotonic() - started) * 1000,
                extra={
                    "method": request.method,
                    "path": request.path,
                    "status": response.status_code,
                    "duration_ms": round((time.monotonic() - started) * 1000, 1),
                    "endpoint": endpoint or None,
                    "user_agent": (request.headers.get("User-Agent") or "")[:200],
                },
            )
        return response

    if settings.trusted_proxy_count > 0:
        app.wsgi_app = ProxyFix(  # type: ignore[method-assign]
            app.wsgi_app,
            x_for=settings.trusted_proxy_count,
            x_proto=settings.trusted_proxy_count,
            x_host=0,
        )

    logger.info(
        "app_iniciada service=%s version=%s engine=%s strictness=%s auth=%s url_fetch=%s",
        settings.service_name,
        settings.version,
        settings.engine,
        settings.strictness,
        settings.auth_enabled,
        settings.fetch_enabled,
        extra={
            "engine": settings.engine,
            "strictness": settings.strictness,
            "auth_enabled": settings.auth_enabled,
            "fetch_enabled": settings.fetch_enabled,
        },
    )
    return app


def _request_id() -> str:
    """Reuse a caller-supplied id (validated) or mint a short one."""
    supplied = request.headers.get("X-Request-ID", "")
    if supplied and 6 <= len(supplied) <= 128 and all(
        ch.isalnum() or ch in "-_." for ch in supplied
    ):
        return supplied
    return uuid.uuid4().hex[:16]


__all__ = ["DEFAULT_CSP", "PUBLIC_ENDPOINTS", "create_app"]
