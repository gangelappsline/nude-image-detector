"""HTTP layer: the REST endpoints of the moderation API.

Conventions
-----------
* A successful *analysis* always answers ``200 OK``, even when the verdict is
  ``block``: the API did its job.  Use ``?reject_on_block=true`` if your gateway
  prefers an HTTP error for rejected uploads.
* Every response carries ``request_id`` (also in the ``X-Request-ID`` header) for
  support tracing.
* Images are never written to disk and never logged.
"""

from __future__ import annotations

import base64
import binascii
import logging
import time
from typing import Any

from flask import Blueprint, Response, current_app, g, jsonify, request

from ..config import PROFILES, Settings
from ..core.labels import catalogue
from ..errors import ApiError, BadRequest, MissingImage, PayloadTooLarge, TooManyItems
from ..logging_conf import redact_url
from ..service import (
    AnalysisOutcome,
    AnalysisService,
    PolicyOverrides,
    describe_source_raw,
    describe_source_upload,
)

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__)

#: Accepted JSON keys carrying base64-encoded image bytes.
_BASE64_KEYS = ("image_base64", "image", "base64", "data")
_MAX_BASE64_CHARS = 64 * 1024 * 1024


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _settings() -> Settings:
    return current_app.config["NID_SETTINGS"]


def _service() -> AnalysisService:
    return current_app.extensions["nid_service"]


def _bool_param(*names: str, default: bool = False) -> bool:
    for name in names:
        value = request.args.get(name)
        if value is None and request.form:
            value = request.form.get(name)
        if value is None:
            payload = getattr(g, "json_payload", None)
            if isinstance(payload, dict):
                value = payload.get(name)
        if value is None:
            continue
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


def _raw_param(*names: str) -> Any:
    for name in names:
        value = request.args.get(name)
        if value is None and request.form:
            value = request.form.get(name)
        if value is None:
            payload = getattr(g, "json_payload", None)
            if isinstance(payload, dict):
                value = payload.get(name)
        if value is not None and value != "":
            return value
    return None


def _policy_overrides() -> PolicyOverrides:
    strictness = _raw_param("strictness", "profile")
    return PolicyOverrides(
        strictness=str(strictness).lower() if strictness else None,
        block_threshold=_raw_param("block_threshold"),
        review_threshold=_raw_param("review_threshold"),
        suggestive_factor=_raw_param("suggestive_factor"),
        min_confidence=_raw_param("min_confidence"),
    )


def _decode_base64(value: Any) -> bytes:
    if not isinstance(value, str):
        raise BadRequest("El campo con la imagen en base64 debe ser una cadena de texto.")
    candidate = value.strip()
    if not candidate:
        raise BadRequest("El campo con la imagen en base64 está vacío.")
    if len(candidate) > _MAX_BASE64_CHARS:
        raise PayloadTooLarge("La imagen en base64 es demasiado grande.")
    if candidate.lower().startswith("data:"):
        # Tolerate data URLs: "data:image/png;base64,...."
        if "," not in candidate:
            raise BadRequest("El data URL está mal formado (falta la coma separadora).")
        candidate = candidate.split(",", 1)[1]
    candidate = "".join(candidate.split())
    try:
        return base64.b64decode(candidate, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise BadRequest(f"El contenido base64 no es válido: {exc}") from exc


def _first_upload():
    """Return the first file part of a multipart request, if any."""
    if not request.files:
        return None
    if "file" in request.files:
        return request.files["file"]
    for name in ("image", "images", "files"):
        if name in request.files:
            return request.files[name]
    return next(iter(request.files.values()), None)


def _json_payload() -> dict[str, Any] | None:
    if getattr(g, "json_payload", None) is not None:
        return g.json_payload

    payload: dict[str, Any] | None = None
    content_type = (request.content_type or "").split(";")[0].strip().lower()
    if content_type == "application/json" or request.method in {"GET", "DELETE"}:
        parsed = request.get_json(silent=True)
        if parsed is None and content_type == "application/json" and request.data:
            raise BadRequest("El cuerpo JSON está mal formado.")
        if isinstance(parsed, dict):
            payload = parsed
        elif parsed is not None:
            raise BadRequest("El cuerpo JSON debe ser un objeto.")
    g.json_payload = payload
    return payload


def _model_info() -> dict[str, Any]:
    engine = current_app.extensions["nid_engine"]
    return engine.describe()


def _request_options() -> dict[str, Any]:
    return {
        "overrides": _policy_overrides(),
        "include_censored": _bool_param("censor", "blur", "censored"),
        "include_detections": not _bool_param("no_detections", default=False),
        "reject_on_block": _bool_param("reject_on_block", "reject_blocked"),
    }


# --------------------------------------------------------------------------- #
# Service status
# --------------------------------------------------------------------------- #
@api_bp.get("/health")
def health() -> Response:
    """Liveness probe: the process is up and answering."""
    service = _service()
    body = service.health()
    body.update(
        {
            "service": _settings().service_name,
            "version": _settings().version,
            "request_id": getattr(g, "request_id", None),
        }
    )
    return jsonify(body)


@api_bp.get("/ready")
def ready() -> tuple[Response, int]:
    """Readiness probe: 200 only when the model can actually score images."""
    engine = current_app.extensions["nid_engine"]
    settings = _settings()
    is_ready = engine.is_ready
    payload = {
        "ready": is_ready,
        "engine": engine.name,
        "model": engine.model,
        "version": engine.version,
        "auth_required": settings.auth_enabled,
        "request_id": getattr(g, "request_id", None),
    }
    if not is_ready and settings.engine != "disabled":
        try:
            engine.load()
            payload["ready"] = engine.is_ready
            is_ready = engine.is_ready
        except Exception:  # pragma: no cover - surfaced through the status code
            logger.exception("carga_modelo_fallida")
    return jsonify(payload), 200 if is_ready else 503


@api_bp.get("/v1/info")
def info() -> Response:
    """Discovery: what this deployment can do and how it will judge images."""
    settings = _settings()
    return jsonify(
        {
            "service": settings.service_name,
            "version": settings.version,
            "model": _model_info(),
            "configuration": settings.public_dict(),
            "profiles": {
                name: {
                    "block_threshold": profile.block_threshold,
                    "review_threshold": profile.review_threshold,
                    "suggestive_factor": profile.suggestive_factor,
                    "description_es": profile.description_es,
                }
                for name, profile in PROFILES.items()
            },
            "verdicts": {
                "block": "Rechaza la subida: desnudez detectada con confianza suficiente.",
                "review": "Ambiguo: debería pasar por revisión humana.",
                "allow": "Sin hallazgos relevantes.",
            },
            "request_id": getattr(g, "request_id", None),
        }
    )


@api_bp.get("/v1/labels")
def labels() -> Response:
    """Label catalogue with severity and effective weight."""
    settings = _settings()
    return jsonify(
        {
            "engine": current_app.extensions["nid_engine"].name,
            "label_count": len(catalogue(settings.label_weights)),
            "labels": catalogue(settings.label_weights),
            "request_id": getattr(g, "request_id", None),
        }
    )


# --------------------------------------------------------------------------- #
# Single analysis
# --------------------------------------------------------------------------- #
@api_bp.post("/v1/analyze")
def analyze() -> Response:
    """Analyse one image: multipart upload, JSON URL, JSON base64 or raw body."""
    _json_payload()  # validates/normalises JSON early
    settings = _settings()
    options = _request_options()
    started = time.monotonic()

    payload, status = _run_single(options)
    response = jsonify(payload)
    response.status_code = status
    response.headers["X-Content-Type-Options"] = "nosniff"
    logger.info(
        "analisis_finalizado verdict=%s risk=%.3f status=%d ms=%.1f",
        payload.get("verdict"),
        payload.get("risk_score", 0.0),
        status,
        (time.monotonic() - started) * 1000,
        extra={
            "verdict": payload.get("verdict"),
            "risk_score": payload.get("risk_score"),
            "http_status": status,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
            "engine": settings.engine,
        },
    )
    return response


def _run_single(options: dict[str, Any]) -> tuple[dict[str, Any], int]:
    settings = _settings()
    service = _service()
    include_detections: bool = options["include_detections"]

    outcome = _dispatch_single(service, settings, options)
    payload = outcome.to_dict(
        request_id=getattr(g, "request_id", None),
        include_detections=include_detections,
        model_info=_model_info(),
    )
    status = 200
    if options["reject_on_block"] and outcome.result.verdict == "block":
        status = 422
    _log_outcome(outcome)
    return payload, status


def _dispatch_single(
    service: AnalysisService, settings: Settings, options: dict[str, Any]
) -> AnalysisOutcome:
    overrides: PolicyOverrides = options["overrides"]
    include_censored: bool = options["include_censored"]
    include_detections: bool = options["include_detections"]

    # 1) multipart file upload
    upload = _first_upload()
    if upload is not None:
        filename = upload.filename or None
        data = upload.read()
        if not data:
            raise BadRequest("El archivo enviado está vacío.")
        if len(data) > settings.max_content_length:
            raise PayloadTooLarge(
                "La imagen supera el tamaño máximo permitido.",
                details={"max_bytes": settings.max_content_length, "received_bytes": len(data)},
            )
        return service.analyze_bytes(
            data,
            overrides=overrides,
            source=describe_source_upload(filename, len(data)),
            declared_mime=upload.mimetype,
            filename=filename,
            include_censored=include_censored,
            include_detections=include_detections,
        )

    # 2) JSON body: url, base64, or a list of items (single element)
    payload = _json_payload()
    if isinstance(payload, dict):
        url = payload.get("url") or (request.form.get("url") if request.form else None)
        if url:
            if not isinstance(url, str):
                raise BadRequest("El campo 'url' debe ser una cadena de texto.")
            return service.analyze_url(
                url,
                overrides=overrides,
                include_censored=include_censored,
                include_detections=include_detections,
            )

        for key in _BASE64_KEYS:
            if payload.get(key):
                data = _decode_base64(payload[key])
                return service.analyze_bytes(
                    data,
                    overrides=overrides,
                    source={"type": "base64", "bytes": len(data)},
                    declared_mime=payload.get("mime_type") or payload.get("content_type"),
                    filename=payload.get("filename"),
                    include_censored=include_censored,
                    include_detections=include_detections,
                )

        if request.form and request.form.get("url"):  # pragma: no cover - defensive
            return service.analyze_url(request.form["url"], overrides=overrides)

    # 3) form-encoded URL without JSON
    if request.form and request.form.get("url"):
        return service.analyze_url(
            request.form["url"],
            overrides=overrides,
            include_censored=include_censored,
            include_detections=include_detections,
        )

    # 4) raw binary body
    content_type = (request.content_type or "").split(";")[0].strip().lower()
    if content_type.startswith("image/") or content_type == "application/octet-stream":
        data = request.get_data(cache=False)
        if not data:
            raise MissingImage()
        return service.analyze_bytes(
            data,
            overrides=overrides,
            source=describe_source_raw(len(data), content_type or None),
            declared_mime=content_type or None,
            filename=None,
            include_censored=include_censored,
            include_detections=include_detections,
        )

    raise MissingImage()


def _log_outcome(outcome: AnalysisOutcome) -> None:
    source_url = outcome.source.get("url") or outcome.source.get("final_url")
    logger.info(
        "analisis_completado verdict=%s risk=%.3f source=%s cached=%s ms=%.1f",
        outcome.result.verdict,
        outcome.result.risk_score,
        outcome.source.get("type"),
        outcome.cached,
        outcome.elapsed_ms,
        extra={
            "verdict": outcome.result.verdict,
            "risk_score": round(outcome.result.risk_score, 4),
            "source_type": outcome.source.get("type"),
            "source_url": redact_url(source_url) if source_url else None,
            "cached": outcome.cached,
            "elapsed_ms": round(outcome.elapsed_ms, 1),
            "sha256": outcome.image.sha256,
            "labels": outcome.result.labels_found,
        },
    )


# --------------------------------------------------------------------------- #
# Batch analysis
# --------------------------------------------------------------------------- #
@api_bp.post("/v1/analyze/batch")
def analyze_batch() -> Response:
    """Analyse up to ``NID_BATCH_MAX_ITEMS`` images in one round trip.

    Partial failures are reported per item instead of failing the whole request:
    one broken URL should not discard the other nineteen analyses.
    """
    _json_payload()
    settings = _settings()
    service = _service()
    options = _request_options()
    overrides: PolicyOverrides = options["overrides"]
    started = time.monotonic()

    items = _collect_batch_items(settings)
    if not items:
        raise MissingImage()
    if len(items) > settings.batch_max_items:
        raise TooManyItems(
            f"El lote tiene {len(items)} elementos y el máximo es {settings.batch_max_items}.",
            details={"received": len(items), "max_items": settings.batch_max_items},
        )

    results: list[dict[str, Any]] = []
    verdict_counts = {"allow": 0, "review": 0, "block": 0}
    highest_risk = 0.0

    for index, item in enumerate(items):
        entry: dict[str, Any] = {"index": index, "source": item["source"]}
        item_started = time.monotonic()
        try:
            if item["kind"] == "url":
                outcome = service.analyze_url(
                    item["value"],
                    overrides=overrides,
                    include_censored=False,
                    include_detections=options["include_detections"],
                )
            else:
                outcome = service.analyze_bytes(
                    item["value"],
                    overrides=overrides,
                    source=item["source"],
                    declared_mime=item.get("declared_mime"),
                    filename=item.get("filename"),
                    include_censored=False,
                    include_detections=options["include_detections"],
                )
            payload = outcome.to_dict(
                request_id=getattr(g, "request_id", None),
                include_detections=options["include_detections"],
                model_info=_model_info(),
            )
            payload.pop("request_id", None)
            payload["elapsed_ms"] = round((time.monotonic() - item_started) * 1000, 1)
            entry.update({"ok": True, "result": payload})
            verdict_counts[outcome.result.verdict] = verdict_counts.get(outcome.result.verdict, 0) + 1
            highest_risk = max(highest_risk, outcome.result.risk_score)
            _log_outcome(outcome)
        except Exception as exc:
            code = exc.code if isinstance(exc, ApiError) else "internal_error"
            message = exc.message if isinstance(exc, ApiError) else "Error interno al analizar el elemento."
            status = exc.status if isinstance(exc, ApiError) else 500
            if status >= 500 and not isinstance(exc, ApiError):
                logger.exception("error_lote_item index=%d", index)
            else:
                logger.info("error_lote_item index=%d code=%s", index, code)
            entry.update(
                {
                    "ok": False,
                    "error": {"code": code, "message": message, "status": status},
                }
            )
            verdict_counts["error"] = verdict_counts.get("error", 0) + 1
        results.append(entry)

    body = {
        "request_id": getattr(g, "request_id", None),
        "count": len(results),
        "summary": {
            "total": len(results),
            "succeeded": sum(1 for r in results if r.get("ok")),
            "failed": sum(1 for r in results if not r.get("ok")),
            "verdicts": verdict_counts,
            "highest_risk_score": round(highest_risk, 4),
            "any_blocked": verdict_counts.get("block", 0) > 0,
            "policy": overrides.resolve(settings).to_dict(),
        },
        "results": results,
        "elapsed_ms": round((time.monotonic() - started) * 1000, 1),
    }
    response = jsonify(body)
    if options["reject_on_block"] and verdict_counts.get("block", 0):
        response.status_code = 422
    return response


def _collect_batch_items(settings: Settings) -> list[dict[str, Any]]:
    """Normalise every accepted batch shape into ``[{kind, value, source}, ...]``."""
    items: list[dict[str, Any]] = []

    # multipart: several file parts ("files" or any number of repeated fields)
    if request.files:
        uploads = request.files.getlist("files")
        if not uploads:
            uploads = [part for key in request.files for part in request.files.getlist(key)]
        for upload in uploads:
            if upload is None:
                continue
            data = upload.read()
            if not data:
                raise BadRequest(f"El archivo '{upload.filename}' está vacío.")
            if len(data) > settings.max_content_length:
                raise PayloadTooLarge(
                    f"El archivo '{upload.filename}' supera el tamaño máximo permitido.",
                    details={"max_bytes": settings.max_content_length},
                )
            items.append(
                {
                    "kind": "bytes",
                    "value": data,
                    "filename": upload.filename,
                    "declared_mime": upload.mimetype,
                    "source": describe_source_upload(upload.filename, len(data)),
                }
            )
        if items:
            return items

    payload = _json_payload()
    if not isinstance(payload, dict):
        raise MissingImage()

    urls = payload.get("urls")
    if isinstance(urls, str):
        urls = [urls]
    if isinstance(urls, list):
        for url in urls:
            if not isinstance(url, str) or not url.strip():
                raise BadRequest("Cada URL del lote debe ser una cadena no vacía.")
            items.append(
                {
                    "kind": "url",
                    "value": url,
                    "source": {"type": "url", "url": redact_url(url)},
                }
            )
        return items

    raw_items = payload.get("items")
    if isinstance(raw_items, list):
        for position, item in enumerate(raw_items):
            if not isinstance(item, dict):
                raise BadRequest(f"El elemento {position} del lote debe ser un objeto JSON.")
            if item.get("url"):
                if not isinstance(item["url"], str):
                    raise BadRequest(f"El campo 'url' del elemento {position} debe ser una cadena.")
                items.append(
                    {
                        "kind": "url",
                        "value": item["url"],
                        "source": {"type": "url", "url": redact_url(item["url"])},
                    }
                )
                continue
            encoded = next((item[key] for key in _BASE64_KEYS if item.get(key)), None)
            if encoded:
                data = _decode_base64(encoded)
                items.append(
                    {
                        "kind": "bytes",
                        "value": data,
                        "filename": item.get("filename"),
                        "declared_mime": item.get("mime_type") or item.get("content_type"),
                        "source": {
                            "type": "base64",
                            "filename": item.get("filename"),
                            "bytes": len(data),
                        },
                    }
                )
                continue
            raise BadRequest(
                f"El elemento {position} del lote necesita 'url' o una imagen en base64."
            )
        return items

    # Tolerate a single-object batch: {"url": "..."} or {"image_base64": "..."}
    if payload.get("url"):
        items.append(
            {
                "kind": "url",
                "value": payload["url"],
                "source": {"type": "url", "url": redact_url(payload["url"])},
            }
        )
        return items
    encoded = next((payload[key] for key in _BASE64_KEYS if payload.get(key)), None)
    if encoded:
        data = _decode_base64(encoded)
        items.append(
            {
                "kind": "bytes",
                "value": data,
                "filename": payload.get("filename"),
                "declared_mime": payload.get("mime_type"),
                "source": {"type": "base64", "bytes": len(data)},
            }
        )
        return items

    raise MissingImage()


# --------------------------------------------------------------------------- #
# Administration
# --------------------------------------------------------------------------- #
@api_bp.delete("/v1/cache")
def clear_cache() -> Response:
    """Flush the analysis cache. Requires authentication to be configured."""
    settings = _settings()
    if not settings.auth_enabled:
        # Without API keys this endpoint would be a free DoS lever for anyone.
        raise BadRequest(
            "Configura NID_API_KEYS para poder usar los endpoints de administración.",
            details={"code": "auth_required_for_admin"},
        )
    service = _service()
    before = service.cache.stats().size
    service.cache.clear()
    return jsonify(
        {
            "cleared": True,
            "entries_before": before,
            "entries_after": service.cache.stats().size,
            "request_id": getattr(g, "request_id", None),
        }
    )
