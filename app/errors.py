"""Typed exceptions shared by the whole application.

Every error is mapped to a stable machine-readable ``code``, an HTTP status and
a human message, so clients can branch on ``error.code`` instead of parsing
strings.  Messages are in Spanish (the product language) and never leak
internals such as stack traces or absolute filesystem paths.
"""

from __future__ import annotations

from typing import Any


class ApiError(Exception):
    """Base class for all expected, client-facing failures."""

    #: Stable machine-readable identifier.
    code: str = "internal_error"
    #: HTTP status returned to the client.
    status: int = 500
    #: Default human-readable message.
    message: str = "Error interno del servicio."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message or self.message)
        self.message = message or self.message
        self.details = details or {}
        self.headers = headers or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


# --------------------------------------------------------------------------- #
# 4xx - client problems
# --------------------------------------------------------------------------- #
class BadRequest(ApiError):
    code = "bad_request"
    status = 400
    message = "La solicitud no es válida."


class MissingImage(BadRequest):
    code = "missing_image"
    status = 400
    message = (
        "No se recibió ninguna imagen. Envía un archivo multipart con el campo "
        "'file', un JSON con la clave 'url', o el binario de la imagen como cuerpo."
    )


class InvalidUrl(BadRequest):
    code = "invalid_url"
    status = 400
    message = "La URL de la imagen no es válida."


class TooManyItems(BadRequest):
    code = "too_many_items"
    status = 400
    message = "El lote supera el número máximo de elementos permitidos."


class Unauthorized(ApiError):
    code = "unauthorized"
    status = 401
    message = "Falta o es inválida la clave de API."


class Forbidden(ApiError):
    code = "forbidden"
    status = 403
    message = "Operación no permitida."


class RateLimited(ApiError):
    code = "rate_limited"
    status = 429
    message = "Demasiadas solicitudes. Intenta de nuevo más tarde."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        retry_after: int | None = None,
    ) -> None:
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        super().__init__(message, details=details, headers=headers or {})


class PayloadTooLarge(ApiError):
    code = "payload_too_large"
    status = 413
    message = "La imagen supera el tamaño máximo permitido."


class UnsupportedMediaType(ApiError):
    code = "unsupported_media_type"
    status = 415
    message = "El tipo de archivo no está soportado."


class UnprocessableImage(ApiError):
    code = "unprocessable_image"
    status = 422
    message = "El archivo no pudo decodificarse como imagen."


class ImageTooLarge(UnprocessableImage):
    code = "image_too_large"
    status = 422
    message = "La imagen tiene demasiados píxeles."


# --------------------------------------------------------------------------- #
# 5xx - upstream / service problems
# --------------------------------------------------------------------------- #
class UpstreamError(ApiError):
    code = "upstream_error"
    status = 502
    message = "No se pudo obtener la imagen remota."


class UpstreamTimeout(UpstreamError):
    code = "upstream_timeout"
    status = 504
    message = "El servidor remoto tardó demasiado en responder."


class BlockedDestination(Forbidden):
    """Raised when a URL points at a forbidden network (SSRF protection)."""

    code = "blocked_destination"
    status = 403
    message = "La URL apunta a una dirección de red no permitida."


class ModelUnavailable(ApiError):
    code = "model_unavailable"
    status = 503
    message = "El modelo de detección no está disponible."


class InferenceError(ApiError):
    code = "inference_error"
    status = 500
    message = "Falló la inferencia del modelo."
