"""Error handlers that keep the API's JSON contract stable.

Whatever goes wrong - a validation error, a Flask 413, an unexpected exception -
the client always receives the same envelope::

    {"request_id": "...", "error": {"code": "...", "message": "...", "details": {}}}
"""

from __future__ import annotations

import logging

from flask import Blueprint, g, jsonify, request
from werkzeug.exceptions import (
    HTTPException,
    MethodNotAllowed,
    NotFound,
    RequestEntityTooLarge,
)
from werkzeug.exceptions import (
    UnsupportedMediaType as WerkzeugUnsupportedMediaType,
)

from ..config import Settings
from ..errors import ApiError, PayloadTooLarge, UnsupportedMediaType

logger = logging.getLogger(__name__)

errors_bp = Blueprint("errors", __name__)

#: Maps Werkzeug/HTTP status codes onto our stable error codes.
_STATUS_CODES = {
    400: ("bad_request", "La solicitud no es válida."),
    401: ("unauthorized", "Autenticación requerida."),
    403: ("forbidden", "Operación no permitida."),
    404: ("not_found", "El recurso no existe."),
    405: ("method_not_allowed", "El método HTTP no está permitido para esta ruta."),
    413: ("payload_too_large", "El cuerpo de la solicitud es demasiado grande."),
    415: ("unsupported_media_type", "El tipo de contenido no está soportado."),
    422: ("unprocessable_image", "No se pudo procesar la imagen."),
    429: ("rate_limited", "Demasiadas solicitudes."),
    500: ("internal_error", "Error interno del servicio."),
    503: ("model_unavailable", "El servicio no está listo para analizar imágenes."),
}


def error_envelope(code: str, message: str, status: int, details: dict | None = None) -> tuple:
    payload: dict = {
        "request_id": getattr(g, "request_id", None),
        "error": {"code": code, "message": message, "status": status},
    }
    if details:
        payload["error"]["details"] = details
    return jsonify(payload), status


def register_error_handlers(app, settings: Settings) -> None:
    """Attach the handlers to an application instance."""

    @app.errorhandler(ApiError)
    def _handle_api_error(exc: ApiError):
        if exc.status >= 500:
            logger.error(
                "error_servicio code=%s status=%d path=%s",
                exc.code,
                exc.status,
                request.path,
                extra={"error_code": exc.code, "details": exc.details},
            )
        else:
            logger.info(
                "error_cliente code=%s status=%d path=%s",
                exc.code,
                exc.status,
                request.path,
                extra={"error_code": exc.code},
            )
        response, status = error_envelope(exc.code, exc.message, exc.status, exc.details)
        for header, value in exc.headers.items():
            response.headers[header] = value
        return response, status

    @app.errorhandler(RequestEntityTooLarge)
    def _handle_too_large(exc: RequestEntityTooLarge):
        mapped = PayloadTooLarge(
            "El cuerpo de la solicitud supera el límite del servidor.",
            details={"max_content_length": settings.max_content_length},
        )
        return error_envelope(mapped.code, mapped.message, mapped.status, mapped.details)

    @app.errorhandler(WerkzeugUnsupportedMediaType)
    def _handle_media_type(exc: WerkzeugUnsupportedMediaType):
        mapped = UnsupportedMediaType(
            "El tipo de contenido no está soportado.",
            details={
                "received": request.content_type,
                "accepted": [
                    "multipart/form-data (campo 'file')",
                    "application/json (clave 'url' o 'image_base64')",
                    "image/* (cuerpo binario)",
                ],
            },
        )
        return error_envelope(mapped.code, mapped.message, mapped.status, mapped.details)

    @app.errorhandler(HTTPException)
    def _handle_http_exception(exc: HTTPException):
        code, message = _STATUS_CODES.get(
            exc.code or 500, (f"http_{exc.code}", exc.description or "Error.")
        )
        if isinstance(exc, (NotFound, MethodNotAllowed)):
            message = exc.description or message
        return error_envelope(code, message, exc.code or 500)

    @app.errorhandler(Exception)
    def _handle_unexpected(exc: Exception):
        logger.exception("error_no_controlado path=%s", request.path)
        return error_envelope(
            "internal_error",
            "Error interno del servicio.",
            500,
            {"type": type(exc).__name__} if settings.debug else None,
        )
