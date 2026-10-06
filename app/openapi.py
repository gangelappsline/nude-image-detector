"""Hand-written OpenAPI 3.0 specification.

Keeping the spec next to the code (instead of generating it from decorators)
means the contract is explicit and reviewable - which matters for an API whose
job is to gate user uploads.
"""

from __future__ import annotations

from typing import Any

from .config import PROFILES, Settings

ANALYSIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["verdict", "nsfw", "risk_score", "image", "source", "policy"],
    "properties": {
        "request_id": {"type": "string", "description": "Id de correlación; también va en la cabecera X-Request-ID."},
        "verdict": {
            "type": "string",
            "enum": ["block", "review", "allow"],
            "description": "Decisión de moderación lista para usar.",
        },
        "verdict_description_es": {"type": "string"},
        "nsfw": {"type": "boolean", "description": "true cuando verdict == 'block'."},
        "risk_score": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Riesgo agregado. Compara con tus propios umbrales si prefieres decidir tú.",
        },
        "scores": {
            "type": "object",
            "properties": {
                "explicit": {"type": "number"},
                "suggestive": {"type": "number"},
            },
        },
        "reasons": {"type": "array", "items": {"type": "string"}, "description": "Explicación legible de los hallazgos que más pesaron."},
        "flags": {"type": "array", "items": {"type": "string"}},
        "explicit_labels": {"type": "array", "items": {"type": "string"}},
        "labels_found": {"type": "array", "items": {"type": "string"}},
        "severity_counts": {"type": "object", "additionalProperties": {"type": "integer"}},
        "top_label": {"type": ["string", "null"]},
        "max_confidence": {"type": "number"},
        "frames_with_findings": {"type": "integer"},
        "detections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                    "severity": {"type": "string", "enum": ["explicit", "suggestive", "neutral"]},
                    "weight": {"type": "number"},
                    "box": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4, "description": "[x1, y1, x2, y2] en píxeles del fotograma analizado."},
                    "area_ratio": {"type": "number"},
                    "frame_index": {"type": "integer"},
                    "counted": {"type": "boolean", "description": "false si la confianza quedó bajo min_confidence."},
                    "description_es": {"type": "string"},
                },
            },
        },
        "image": {
            "type": "object",
            "properties": {
                "format": {"type": "string"},
                "mime_type": {"type": "string"},
                "width": {"type": "integer"},
                "height": {"type": "integer"},
                "bytes": {"type": "integer"},
                "sha256": {"type": "string"},
                "animated": {"type": "boolean"},
                "frames_total": {"type": "integer"},
                "frames_analysed": {"type": "integer"},
                "downscaled": {"type": "boolean"},
            },
        },
        "source": {"type": "object", "description": "De dónde vino la imagen (upload, url, base64 o raw_body)."},
        "policy": {"type": "object", "description": "Umbrales efectivos aplicados."},
        "model": {"type": "object"},
        "cached": {"type": "boolean"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "elapsed_ms": {"type": "number"},
        "censored_image": {
            "type": "object",
            "description": "Solo presente si se pidió censor=true y la imagen no pasó el filtro.",
            "properties": {
                "format": {"type": "string"},
                "bytes": {"type": "integer"},
                "data_url": {"type": "string"},
            },
        },
    },
}

ERROR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["error"],
    "properties": {
        "request_id": {"type": ["string", "null"]},
        "error": {
            "type": "object",
            "required": ["code", "message", "status"],
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Código estable para branching del cliente.",
                    "examples": [
                        "missing_image",
                        "invalid_url",
                        "blocked_destination",
                        "payload_too_large",
                        "unsupported_media_type",
                        "unprocessable_image",
                        "image_too_large",
                        "upstream_error",
                        "upstream_timeout",
                        "rate_limited",
                        "unauthorized",
                        "model_unavailable",
                    ],
                },
                "message": {"type": "string"},
                "status": {"type": "integer"},
                "details": {"type": "object"},
            },
        },
    },
}

_POLICY_PARAMS = [
    {
        "name": "strictness",
        "in": "query",
        "required": False,
        "schema": {"type": "string", "enum": sorted(PROFILES), "default": "balanced"},
        "description": "Perfil de estrictez. Se ignora si NID_ALLOW_REQUEST_OVERRIDES=false.",
    },
    {
        "name": "block_threshold",
        "in": "query",
        "required": False,
        "schema": {"type": "number", "minimum": 0, "maximum": 1},
        "description": "Sobrescribe el umbral de bloqueo.",
    },
    {
        "name": "review_threshold",
        "in": "query",
        "required": False,
        "schema": {"type": "number", "minimum": 0, "maximum": 1},
        "description": "Sobrescribe el umbral de revisión.",
    },
    {
        "name": "min_confidence",
        "in": "query",
        "required": False,
        "schema": {"type": "number", "minimum": 0, "maximum": 1},
        "description": "Confianza mínima para que una detección cuente en la puntuación.",
    },
    {
        "name": "censor",
        "in": "query",
        "required": False,
        "schema": {"type": "boolean", "default": False},
        "description": "Devuelve además una copia pixelada/difuminada de la imagen (base64) si no pasa el filtro.",
    },
    {
        "name": "no_detections",
        "in": "query",
        "required": False,
        "schema": {"type": "boolean", "default": False},
        "description": "Omite la lista de detecciones para respuestas más ligeras.",
    },
    {
        "name": "reject_on_block",
        "in": "query",
        "required": False,
        "schema": {"type": "boolean", "default": False},
        "description": "Devuelve HTTP 422 en lugar de 200 cuando el veredicto es 'block'.",
    },
]


def build_openapi_spec(settings: Settings) -> dict[str, Any]:
    """Return the OpenAPI document for this deployment."""
    security_schemes: dict[str, Any] = {}
    security: list[dict[str, Any]] = []
    if settings.auth_enabled:
        security_schemes["ApiKeyAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
        }
        security_schemes["BearerAuth"] = {"type": "http", "scheme": "bearer"}
        security = [{"ApiKeyAuth": []}, {"BearerAuth": []}]

    return {
        "openapi": "3.0.3",
        "info": {
            "title": "Nude Image Detector API",
            "version": settings.version,
            "description": (
                "API de moderación de imágenes: recibe una subida multipart, una URL o un "
                "base64 y devuelve un veredicto (`block` / `review` / `allow`) con el riesgo "
                "de desnudez detectado. Pensada para bloquear contenido explícito antes de "
                "que llegue a tu almacenamiento."
            ),
            "license": {"name": "MIT"},
        },
        "servers": [{"url": "/", "description": "Despliegue actual"}],
        "security": security,
        "components": {
            "securitySchemes": security_schemes,
            "schemas": {
                "Analysis": ANALYSIS_SCHEMA,
                "Error": ERROR_SCHEMA,
                "BatchResult": {
                    "type": "object",
                    "properties": {
                        "request_id": {"type": "string"},
                        "count": {"type": "integer"},
                        "summary": {
                            "type": "object",
                            "properties": {
                                "total": {"type": "integer"},
                                "succeeded": {"type": "integer"},
                                "failed": {"type": "integer"},
                                "verdicts": {"type": "object", "additionalProperties": {"type": "integer"}},
                                "highest_risk_score": {"type": "number"},
                                "any_blocked": {"type": "boolean"},
                            },
                        },
                        "results": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "index": {"type": "integer"},
                                    "ok": {"type": "boolean"},
                                    "source": {"type": "object"},
                                    "result": {"$ref": "#/components/schemas/Analysis"},
                                    "error": {"type": "object"},
                                },
                            },
                        },
                        "elapsed_ms": {"type": "number"},
                    },
                },
            },
            "responses": {
                "BadRequest": {"description": "Solicitud inválida.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                "Unauthorized": {"description": "Clave de API ausente o inválida.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                "Forbidden": {"description": "Destino bloqueado por la política SSRF o endpoint no permitido.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                "PayloadTooLarge": {"description": "Imagen demasiado grande.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                "UnsupportedMediaType": {"description": "Formato no soportado.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                "UnprocessableImage": {"description": "El archivo no es una imagen decodificable.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                "RateLimited": {"description": "Límite de peticiones superado.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                "UpstreamError": {"description": "No se pudo descargar la imagen remota.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
                "ServiceUnavailable": {"description": "El modelo no está listo.", "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}}},
            },
        },
        "paths": {
            "/v1/analyze": {
                "post": {
                    "operationId": "analyzeImage",
                    "summary": "Analiza una imagen (upload, URL, base64 o cuerpo binario)",
                    "tags": ["moderation"],
                    "parameters": _POLICY_PARAMS,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "file": {"type": "string", "format": "binary"},
                                        "strictness": {"type": "string", "enum": sorted(PROFILES)},
                                    },
                                }
                            },
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "url": {"type": "string", "format": "uri", "example": "https://example.com/foto.jpg"},
                                        "image_base64": {"type": "string", "description": "Imagen codificada en base64 (admite data URLs)."},
                                        "strictness": {"type": "string", "enum": sorted(PROFILES)},
                                        "block_threshold": {"type": "number"},
                                        "review_threshold": {"type": "number"},
                                    },
                                }
                            },
                            "image/*": {
                                "schema": {"type": "string", "format": "binary"},
                                "example": "bytes de la imagen",
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Análisis completado (incluso si el veredicto es 'block').",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Analysis"}}},
                        },
                        "400": {"$ref": "#/components/responses/BadRequest"},
                        "401": {"$ref": "#/components/responses/Unauthorized"},
                        "403": {"$ref": "#/components/responses/Forbidden"},
                        "413": {"$ref": "#/components/responses/PayloadTooLarge"},
                        "415": {"$ref": "#/components/responses/UnsupportedMediaType"},
                        "422": {"$ref": "#/components/responses/UnprocessableImage"},
                        "429": {"$ref": "#/components/responses/RateLimited"},
                        "502": {"$ref": "#/components/responses/UpstreamError"},
                        "503": {"$ref": "#/components/responses/ServiceUnavailable"},
                    },
                }
            },
            "/v1/analyze/batch": {
                "post": {
                    "operationId": "analyzeBatch",
                    "summary": f"Analiza hasta {settings.batch_max_items} imágenes en una sola llamada",
                    "tags": ["moderation"],
                    "parameters": _POLICY_PARAMS,
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "urls": {
                                            "type": "array",
                                            "items": {"type": "string", "format": "uri"},
                                            "maxItems": settings.batch_max_items,
                                        },
                                        "items": {
                                            "type": "array",
                                            "maxItems": settings.batch_max_items,
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "url": {"type": "string", "format": "uri"},
                                                    "image_base64": {"type": "string"},
                                                    "filename": {"type": "string"},
                                                },
                                            },
                                        },
                                    },
                                }
                            },
                            "multipart/form-data": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "files": {
                                            "type": "array",
                                            "items": {"type": "string", "format": "binary"},
                                        }
                                    },
                                }
                            },
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Resultados por elemento; los fallos individuales no abortan el lote.",
                            "content": {"application/json": {"schema": {"$ref": "#/components/schemas/BatchResult"}}},
                        },
                        "400": {"$ref": "#/components/responses/BadRequest"},
                        "401": {"$ref": "#/components/responses/Unauthorized"},
                        "413": {"$ref": "#/components/responses/PayloadTooLarge"},
                        "429": {"$ref": "#/components/responses/RateLimited"},
                    },
                }
            },
            "/v1/info": {
                "get": {
                    "operationId": "getInfo",
                    "summary": "Modelo, perfiles de estrictez y límites del despliegue",
                    "tags": ["service"],
                    "responses": {"200": {"description": "Información del servicio."}},
                }
            },
            "/v1/labels": {
                "get": {
                    "operationId": "getLabels",
                    "summary": "Catálogo de etiquetas con severidad y peso efectivo",
                    "tags": ["service"],
                    "responses": {"200": {"description": "Catálogo de etiquetas."}},
                }
            },
            "/health": {
                "get": {
                    "operationId": "health",
                    "summary": "Liveness probe",
                    "tags": ["service"],
                    "responses": {"200": {"description": "El proceso responde."}},
                }
            },
            "/ready": {
                "get": {
                    "operationId": "ready",
                    "summary": "Readiness probe (503 si el modelo no está cargado)",
                    "tags": ["service"],
                    "responses": {
                        "200": {"description": "Listo para analizar."},
                        "503": {"$ref": "#/components/responses/ServiceUnavailable"},
                    },
                }
            },
            "/v1/cache": {
                "delete": {
                    "operationId": "clearCache",
                    "summary": "Vacía la caché de análisis (requiere NID_API_KEYS)",
                    "tags": ["admin"],
                    "responses": {
                        "200": {"description": "Caché vaciada."},
                        "400": {"$ref": "#/components/responses/BadRequest"},
                        "401": {"$ref": "#/components/responses/Unauthorized"},
                    },
                }
            },
        },
    }
