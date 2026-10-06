"""Documentation UI.

A single self-contained page (no CDNs, no build step) that doubles as:

* **API reference** - endpoints, payloads, error codes, configuration table.
* **Playground** - upload a file or paste a URL and see the verdict, the risk
  breakdown and the detection boxes drawn over the image.

It ships with the service so anyone evaluating the API can try it immediately,
and it is safe to expose publicly because it only calls the same public API.
"""

from __future__ import annotations

from flask import Blueprint, current_app, jsonify, render_template

from .config import PROFILES, Settings
from .core.labels import catalogue

ui_bp = Blueprint("ui", __name__)


def _context() -> dict[str, object]:
    settings: Settings = current_app.config["NID_SETTINGS"]
    engine = current_app.extensions["nid_engine"]
    return {
        "settings": settings,
        "engine": engine.describe(),
        "profiles": [
            {
                "name": name,
                "block_threshold": profile.block_threshold,
                "review_threshold": profile.review_threshold,
                "suggestive_factor": profile.suggestive_factor,
                "description_es": profile.description_es,
                "active": name == settings.strictness,
            }
            for name, profile in PROFILES.items()
        ],
        "labels": catalogue(settings.label_weights),
        "info": settings.public_dict(),
        "endpoints": [
            {
                "method": "POST",
                "path": "/v1/analyze",
                "summary": "Analiza una imagen: multipart, JSON con url, JSON con base64 o cuerpo binario.",
            },
            {
                "method": "POST",
                "path": "/v1/analyze/batch",
                "summary": f"Analiza hasta {settings.batch_max_items} imágenes; los fallos individuales no abortan el lote.",
            },
            {"method": "GET", "path": "/v1/info", "summary": "Modelo, perfiles y límites del despliegue."},
            {"method": "GET", "path": "/v1/labels", "summary": "Catálogo de etiquetas, severidad y peso."},
            {"method": "GET", "path": "/health", "summary": "Liveness probe."},
            {"method": "GET", "path": "/ready", "summary": "Readiness probe (503 si el modelo no cargó)."},
            {"method": "GET", "path": "/openapi.json", "summary": "Especificación OpenAPI 3.0."},
            {"method": "DELETE", "path": "/v1/cache", "summary": "Vacía la caché (requiere NID_API_KEYS)."},
        ],
        "error_codes": [
            ("missing_image", 400, "No llegó ninguna imagen en la petición."),
            ("bad_request", 400, "Parámetros o JSON inválidos."),
            ("invalid_url", 400, "La URL no es válida (esquema, host, caracteres)."),
            ("too_many_items", 400, "El lote supera NID_BATCH_MAX_ITEMS."),
            ("unauthorized", 401, "Falta o es inválida la cabecera X-API-Key."),
            ("forbidden", 403, "Operación no permitida (por ejemplo, URL fetching desactivado)."),
            ("blocked_destination", 403, "La URL apunta a una red privada/protegida (anti-SSRF)."),
            ("payload_too_large", 413, "Se superó NID_MAX_CONTENT_LENGTH o NID_FETCH_MAX_BYTES."),
            ("unsupported_media_type", 415, "Formato no permitido (por ejemplo HEIC/AVIF)."),
            ("unprocessable_image", 422, "El archivo no se pudo decodificar como imagen."),
            ("image_too_large", 422, "Demasiados píxeles (posible bomba de descompresión)."),
            ("rate_limited", 429, "Se superó el límite de peticiones por ventana."),
            ("upstream_error", 502, "El servidor remoto falló o devolvió un estado inesperado."),
            ("upstream_timeout", 504, "El servidor remoto tardó demasiado."),
            ("model_unavailable", 503, "El motor de detección no está cargado."),
            ("inference_error", 500, "El modelo falló al procesar la imagen."),
        ],
        "config_vars": [
            ("NID_ENGINE", settings.engine, "nudenet | mock | disabled"),
            ("NID_STRICTNESS", settings.strictness, "Perfil por defecto: strict | balanced | lenient"),
            ("NID_BLOCK_THRESHOLD", settings.block_threshold, "Sobrescribe el umbral de bloqueo del perfil"),
            ("NID_REVIEW_THRESHOLD", settings.review_threshold, "Sobrescribe el umbral de revisión"),
            ("NID_MIN_CONFIDENCE", settings.min_confidence, "Confianza mínima para contar una detección"),
            ("NID_LABEL_WEIGHTS", settings.label_weights or "{}", "JSON: peso por etiqueta (0-1)"),
            ("NID_MAX_CONTENT_LENGTH", settings.max_content_length, "Bytes máximos por imagen"),
            ("NID_MAX_IMAGE_PIXELS", settings.max_image_pixels, "Píxeles máximos (anti bomba de descompresión)"),
            ("NID_MAX_FRAMES", settings.max_frames, "Fotogramas muestreados en imágenes animadas"),
            ("NID_API_KEYS", "***" if settings.api_keys else "(vacío)", "Lista separada por comas; activa la autenticación"),
            ("NID_RATE_LIMIT_ENABLED", settings.rate_limit_enabled, "Activa el límite de peticiones por proceso"),
            ("NID_FETCH_ENABLED", settings.fetch_enabled, "Permite el análisis por URL"),
            ("NID_FETCH_ALLOW_PRIVATE_NETWORKS", settings.fetch_allow_private_networks, "Solo para pruebas: desactiva la protección SSRF"),
            ("NID_FETCH_ALLOWED_HOSTS", settings.fetch_allowed_hosts or "(todos los públicos)", "Allowlist de hosts"),
            ("NID_FETCH_TIMEOUT_SECONDS", settings.fetch_timeout_seconds, "Timeout por salto de redirección"),
            ("NID_BATCH_MAX_ITEMS", settings.batch_max_items, "Máximo de imágenes por lote"),
            ("NID_CACHE_ENABLED", settings.cache_enabled, "Caché de detecciones por SHA-256"),
            ("NID_MAX_CONCURRENT_INFERENCES", settings.max_concurrent_inferences, "Semáforo de inferencias simultáneas"),
        ],
    }


@ui_bp.get("/")
def index():
    """Render the docs + playground page."""
    return render_template("index.html", **_context())


@ui_bp.get("/docs")
def docs():
    """Alias of the documentation page."""
    return render_template("index.html", **_context())


@ui_bp.get("/ui/config")
def ui_config():
    """JSON subset used by the playground JavaScript (no template coupling)."""
    settings: Settings = current_app.config["NID_SETTINGS"]
    engine = current_app.extensions["nid_engine"]
    return jsonify(
        {
            "engine": engine.describe(),
            "info": settings.public_dict(),
            "auth_required": settings.auth_enabled,
            "url_fetching": settings.fetch_enabled,
            "batch_max_items": settings.batch_max_items,
            "profiles": {
                name: {
                    "block_threshold": profile.block_threshold,
                    "review_threshold": profile.review_threshold,
                    "suggestive_factor": profile.suggestive_factor,
                }
                for name, profile in PROFILES.items()
            },
        }
    )
