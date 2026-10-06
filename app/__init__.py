"""Application factory for the Nude Image Detector API.

``create_app()`` is the single entry point used by the dev server and by
gunicorn, so every environment gets exactly the same middleware, error contract
and engine wiring.
"""

from __future__ import annotations

import logging
import time
import uuid

from flask import Flask, g, request
from werkzeug.middleware.proxy_fix import ProxyFix

from .api.errors import register_error_handlers
from .api.routes import api_bp
from .config import Settings
from .core.engine import build_engine
from .logging_conf import setup_logging
from .security import RateLimiter, enforce_request_limits, request_cost, require_auth
from .service import AnalysisService

logger = logging.getLogger(__name__)

#: Endpoints reachable without credentials: orchestrator probes and discovery.
PUBLIC_ENDPOINTS = frozenset({"api.health", "api.ready", "api.info", "api.labels"})


def create_app(settings: Settings | None = None) -> Flask:
    """Build and configure the Flask application."""
    settings = settings or Settings.from_env()
    setup_logging(settings.log_level, json_output=settings.log_json)

    # static_folder=None: this service exposes JSON endpoints only, so Flask's
    # automatic /static route is removed instead of serving an empty directory.
    app = Flask(__name__, static_folder=None)
    app.config["NID_SETTINGS"] = settings
    app.config["MAX_CONTENT_LENGTH"] = settings.max_content_length
    app.config["PROPAGATE_EXCEPTIONS"] = True
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
    register_error_handlers(app, settings)

    # --- middleware ------------------------------------------------------ #
    @app.before_request
    def _before() -> None:
        g.request_id = _request_id()
        g.started_at = time.monotonic()
        g.json_payload = None

        if (request.endpoint or "") in PUBLIC_ENDPOINTS:
            return

        require_auth(request, settings)
        if settings.rate_limit_enabled:
            g.rate_limit = enforce_request_limits(
                request, settings, limiter, cost=request_cost(request)
            )

    @app.after_request
    def _after(response):
        request_id = getattr(g, "request_id", None)
        if request_id:
            response.headers.setdefault("X-Request-ID", request_id)

        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        # Analysis responses can embed a base64 image: never cache them.
        response.headers.setdefault("Cache-Control", "no-store")
        if settings.cors_origin:
            response.headers.setdefault("Access-Control-Allow-Origin", settings.cors_origin)
            response.headers.setdefault("Vary", "Origin")
            response.headers.setdefault(
                "Access-Control-Allow-Headers",
                "Content-Type, X-API-Key, Authorization, X-Request-ID",
            )
            response.headers.setdefault(
                "Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"
            )

        decision = getattr(g, "rate_limit", None)
        if decision is not None:
            for header, value in decision.headers().items():
                response.headers.setdefault(header, value)

        started = getattr(g, "started_at", None)
        if started is not None and (request.endpoint or "") not in {"api.health", "api.ready"}:
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
                    "endpoint": request.endpoint,
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


__all__ = ["create_app", "PUBLIC_ENDPOINTS"]
