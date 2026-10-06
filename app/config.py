"""Application settings, read from environment variables (``NID_`` prefix).

Design rules:

* Every knob that a deployment may need to tune is an environment variable, so
  the same image runs in dev, staging and production unchanged.
* A ``.env`` file is honoured when present (12-factor friendly), but real
  environment variables always win.
* Values are validated at start-up: a typo fails fast instead of silently
  weakening moderation.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

try:  # optional in production images that bake the env in
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

ENV_PREFIX = "NID_"

#: Accepted image container formats (validated by magic bytes, not by extension).
DEFAULT_ALLOWED_MIME_TYPES: tuple[str, ...] = (
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
)


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #
class ConfigError(RuntimeError):
    """Raised when an environment variable cannot be parsed."""


def _raw(key: str, default: str | None = None) -> str | None:
    value = os.environ.get(ENV_PREFIX + key)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _str(key: str, default: str) -> str:
    return _raw(key, default) or default


def _optional_str(key: str) -> str | None:
    return _raw(key, None)


def _bool(key: str, default: bool) -> bool:
    value = _raw(key, None)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ConfigError(f"{ENV_PREFIX}{key} debe ser un booleano (true/false), se recibió '{value}'")


def _int(key: str, default: int, *, minimum: int | None = None, maximum: int | None = None) -> int:
    value = _raw(key, None)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{key} debe ser un entero, se recibió '{value}'") from exc
    _check_range(key, parsed, minimum, maximum)
    return parsed


def _float(
    key: str, default: float, *, minimum: float | None = None, maximum: float | None = None
) -> float:
    value = _raw(key, None)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{key} debe ser un número, se recibió '{value}'") from exc
    _check_range(key, parsed, minimum, maximum)
    return parsed


def _check_range(key: str, value: Any, minimum: Any, maximum: Any) -> None:
    if minimum is not None and value < minimum:
        raise ConfigError(f"{ENV_PREFIX}{key} debe ser >= {minimum}, se recibió {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{ENV_PREFIX}{key} debe ser <= {maximum}, se recibió {value}")


def _list(key: str, default: Iterable[str]) -> tuple[str, ...]:
    value = _raw(key, None)
    if value is None:
        return tuple(default)
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    return items or tuple(default)


def _json_map(key: str, default: Mapping[str, Any] | None = None) -> dict[str, Any]:
    value = _raw(key, None)
    if value is None:
        return dict(default or {})
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{ENV_PREFIX}{key} debe ser JSON válido: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ConfigError(f"{ENV_PREFIX}{key} debe ser un objeto JSON")
    return parsed


# --------------------------------------------------------------------------- #
# Strictness profiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Profile:
    """A named preset for the moderation policy."""

    name: str
    block_threshold: float
    review_threshold: float
    suggestive_factor: float
    description_es: str


#: Ready-to-use profiles.  ``suggestive_factor`` scales how much "covered /
#: bikini / shirtless" evidence alone can push the risk score.
PROFILES: Mapping[str, Profile] = {
    p.name: p
    for p in (
        Profile(
            name="strict",
            block_threshold=0.30,
            review_threshold=0.12,
            suggestive_factor=1.00,
            description_es=(
                "Bloquea incluso contenido sugerente (ropa interior, bikinis, torsos "
                "descubiertos). Recomendado para apps infantiles o escolares."
            ),
        ),
        Profile(
            name="balanced",
            block_threshold=0.55,
            review_threshold=0.28,
            suggestive_factor=0.75,
            description_es=(
                "Bloquea desnudos explícitos y manda a revisión el contenido ambiguo. "
                "Acepta playa/deporte evidente. Valor por defecto."
            ),
        ),
        Profile(
            name="lenient",
            block_threshold=0.80,
            review_threshold=0.50,
            suggestive_factor=0.45,
            description_es=(
                "Solo bloquea desnudos inequívocos y de alta confianza. Útil en "
                "contextos artísticos o médicos donde los falsos positivos cuestan caro."
            ),
        ),
    )
}

DEFAULT_PROFILE = "balanced"


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration."""

    # --- HTTP server -------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"
    log_json: bool = True
    trusted_proxy_count: int = 0

    # --- Authentication / abuse -------------------------------------------
    api_keys: tuple[str, ...] = ()
    rate_limit_enabled: bool = False
    rate_limit_requests: int = 120
    rate_limit_window_seconds: int = 60

    # --- Upload limits -----------------------------------------------------
    max_content_length: int = 10 * 1024 * 1024  # 10 MiB
    max_image_pixels: int = 40_000_000  # ~6324x6324, PIL decompression-bomb guard
    max_analysis_pixels: int = 4_000_000  # downscale above this before inference
    allowed_mime_types: tuple[str, ...] = DEFAULT_ALLOWED_MIME_TYPES

    # --- Detection engine --------------------------------------------------
    engine: str = "nudenet"  # nudenet | mock | disabled
    model_path: str | None = None  # optional custom .onnx (defaults to the bundled one)
    inference_resolution: int = 320
    min_confidence: float = 0.30
    max_concurrent_inferences: int = 2
    load_model_at_startup: bool = True
    warmup: bool = True
    max_frames: int = 3  # frames sampled from animated images (GIF/APNG/animated WebP)
    blur_strength: int = 51

    # --- Policy ------------------------------------------------------------
    strictness: str = DEFAULT_PROFILE
    block_threshold: float | None = None
    review_threshold: float | None = None
    suggestive_factor: float | None = None
    label_weights: dict[str, float] = field(default_factory=dict)
    allow_request_overrides: bool = True

    # --- Remote URL fetching (SSRF surface) --------------------------------
    fetch_enabled: bool = True
    fetch_timeout_seconds: float = 8.0
    fetch_max_bytes: int = 10 * 1024 * 1024
    fetch_max_redirects: int = 3
    fetch_allow_private_networks: bool = False
    fetch_allowed_hosts: tuple[str, ...] = ()  # empty = allow every public host
    fetch_blocked_hosts: tuple[str, ...] = ()
    fetch_blocked_cidrs: tuple[str, ...] = (
        "169.254.169.254/32",  # cloud metadata (AWS/GCP/Azure)
        "100.64.0.0/10",  # CGNAT
        "192.0.0.0/24",  # IETF protocol assignments
        "198.18.0.0/15",  # benchmarking
    )
    fetch_allowed_ports: tuple[int, ...] = (80, 443)
    fetch_user_agent: str = "nude-image-detector/1.0 (+https://github.com/gangelappsline/nude-image-detector)"

    # --- Cache -------------------------------------------------------------
    cache_enabled: bool = True
    cache_maxsize: int = 512
    cache_ttl_seconds: int = 3600

    # --- Batch -------------------------------------------------------------
    batch_max_items: int = 10

    # --- Misc --------------------------------------------------------------
    expose_openapi: bool = True
    cors_origin: str = "*"
    service_name: str = "nude-image-detector"
    version: str = "1.0.0"

    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, *, load_dotenv_file: bool = True) -> Settings:
        """Build settings from the environment, validating every value."""
        if load_dotenv_file and load_dotenv is not None:
            load_dotenv(override=False)

        engine = _str("ENGINE", "nudenet").lower()
        if engine not in {"nudenet", "mock", "disabled"}:
            raise ConfigError(
                f"{ENV_PREFIX}ENGINE debe ser 'nudenet', 'mock' o 'disabled', se recibió '{engine}'"
            )

        strictness = _str("STRICTNESS", DEFAULT_PROFILE).lower()
        if strictness not in PROFILES:
            raise ConfigError(
                f"{ENV_PREFIX}STRICTNESS debe ser uno de {sorted(PROFILES)}, se recibió '{strictness}'"
            )

        weights = _json_map("LABEL_WEIGHTS")
        for name, value in weights.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ConfigError(
                    f"{ENV_PREFIX}LABEL_WEIGHTS['{name}'] debe ser un número entre 0 y 1"
                )
            _check_range(f"LABEL_WEIGHTS['{name}']", float(value), 0.0, 1.0)
            weights[name] = float(value)

        log_level = _str("LOG_LEVEL", "INFO").upper()
        if log_level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError(f"{ENV_PREFIX}LOG_LEVEL inválido: '{log_level}'")

        block = _float("BLOCK_THRESHOLD", -1.0, minimum=-1.0, maximum=1.0)
        review = _float("REVIEW_THRESHOLD", -1.0, minimum=-1.0, maximum=1.0)
        suggestive = _float("SUGGESTIVE_FACTOR", -1.0, minimum=-1.0, maximum=1.0)

        settings = cls(
            host=_str("HOST", "0.0.0.0"),
            port=_int("PORT", 8000, minimum=1, maximum=65535),
            debug=_bool("DEBUG", False),
            log_level=log_level,
            log_json=_bool("LOG_JSON", True),
            trusted_proxy_count=_int("TRUSTED_PROXY_COUNT", 0, minimum=0, maximum=10),
            api_keys=_list("API_KEYS", ()),
            rate_limit_enabled=_bool("RATE_LIMIT_ENABLED", False),
            rate_limit_requests=_int("RATE_LIMIT_REQUESTS", 120, minimum=1),
            rate_limit_window_seconds=_int("RATE_LIMIT_WINDOW_SECONDS", 60, minimum=1),
            max_content_length=_int("MAX_CONTENT_LENGTH", 10 * 1024 * 1024, minimum=1024),
            max_image_pixels=_int("MAX_IMAGE_PIXELS", 40_000_000, minimum=10_000),
            max_analysis_pixels=_int("MAX_ANALYSIS_PIXELS", 4_000_000, minimum=10_000),
            allowed_mime_types=tuple(
                m.lower() for m in _list("ALLOWED_MIME_TYPES", DEFAULT_ALLOWED_MIME_TYPES)
            ),
            engine=engine,
            model_path=_optional_str("MODEL_PATH"),
            inference_resolution=_int("INFERENCE_RESOLUTION", 320, minimum=64, maximum=1280),
            min_confidence=_float("MIN_CONFIDENCE", 0.30, minimum=0.0, maximum=1.0),
            max_concurrent_inferences=_int("MAX_CONCURRENT_INFERENCES", 2, minimum=1, maximum=64),
            load_model_at_startup=_bool("LOAD_MODEL_AT_STARTUP", True),
            warmup=_bool("WARMUP", True),
            max_frames=_int("MAX_FRAMES", 3, minimum=1, maximum=32),
            blur_strength=_int("BLUR_STRENGTH", 51, minimum=1, maximum=201),
            strictness=strictness,
            block_threshold=None if block < 0 else block,
            review_threshold=None if review < 0 else review,
            suggestive_factor=None if suggestive < 0 else suggestive,
            label_weights=weights,
            allow_request_overrides=_bool("ALLOW_REQUEST_OVERRIDES", True),
            fetch_enabled=_bool("FETCH_ENABLED", True),
            fetch_timeout_seconds=_float("FETCH_TIMEOUT_SECONDS", 8.0, minimum=0.5, maximum=120.0),
            fetch_max_bytes=_int("FETCH_MAX_BYTES", 10 * 1024 * 1024, minimum=1024),
            fetch_max_redirects=_int("FETCH_MAX_REDIRECTS", 3, minimum=0, maximum=10),
            fetch_allow_private_networks=_bool("FETCH_ALLOW_PRIVATE_NETWORKS", False),
            fetch_allowed_hosts=tuple(h.lower() for h in _list("FETCH_ALLOWED_HOSTS", ())),
            fetch_blocked_hosts=tuple(h.lower() for h in _list("FETCH_BLOCKED_HOSTS", ())),
            fetch_blocked_cidrs=_list("FETCH_BLOCKED_CIDRS", cls.fetch_blocked_cidrs),
            fetch_allowed_ports=tuple(
                _int_port(p) for p in _list("FETCH_ALLOWED_PORTS", ("80", "443"))
            ),
            fetch_user_agent=_str("FETCH_USER_AGENT", cls.fetch_user_agent),
            cache_enabled=_bool("CACHE_ENABLED", True),
            cache_maxsize=_int("CACHE_MAXSIZE", 512, minimum=0, maximum=100_000),
            cache_ttl_seconds=_int("CACHE_TTL_SECONDS", 3600, minimum=1),
            batch_max_items=_int("BATCH_MAX_ITEMS", 10, minimum=1, maximum=100),
            expose_openapi=_bool("EXPOSE_OPENAPI", True),
            cors_origin=_str("CORS_ORIGIN", "*"),
            service_name=_str("SERVICE_NAME", "nude-image-detector"),
            version=_str("VERSION", "1.0.0"),
        )
        settings.validate()
        return settings

    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Cross-field consistency checks."""
        profile = self.profile
        block = self.block_threshold if self.block_threshold is not None else profile.block_threshold
        review = (
            self.review_threshold if self.review_threshold is not None else profile.review_threshold
        )
        if review > block:
            raise ConfigError(
                f"El umbral de revisión ({review}) no puede ser mayor que el de bloqueo ({block})"
            )
        if self.max_analysis_pixels > self.max_image_pixels:
            raise ConfigError(
                f"{ENV_PREFIX}MAX_ANALYSIS_PIXELS no puede superar a {ENV_PREFIX}MAX_IMAGE_PIXELS"
            )

    @property
    def profile(self) -> Profile:
        return PROFILES[self.strictness]

    @property
    def effective_block_threshold(self) -> float:
        return self.block_threshold if self.block_threshold is not None else self.profile.block_threshold

    @property
    def effective_review_threshold(self) -> float:
        return (
            self.review_threshold if self.review_threshold is not None else self.profile.review_threshold
        )

    @property
    def effective_suggestive_factor(self) -> float:
        return (
            self.suggestive_factor
            if self.suggestive_factor is not None
            else self.profile.suggestive_factor
        )

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)

    def public_dict(self) -> dict[str, Any]:
        """Non-sensitive view of the configuration, exposed by ``GET /v1/info``."""
        return {
            "engine": self.engine,
            "strictness": self.strictness,
            "profile": {
                "name": self.profile.name,
                "description_es": self.profile.description_es,
            },
            "thresholds": {
                "block": round(self.effective_block_threshold, 4),
                "review": round(self.effective_review_threshold, 4),
                "suggestive_factor": round(self.effective_suggestive_factor, 4),
            },
            "min_confidence": self.min_confidence,
            "limits": {
                "max_content_length_bytes": self.max_content_length,
                "max_image_pixels": self.max_image_pixels,
                "max_frames": self.max_frames,
                "batch_max_items": self.batch_max_items,
            },
            "features": {
                "url_fetching": self.fetch_enabled,
                "auth_required": self.auth_enabled,
                "rate_limit_enabled": self.rate_limit_enabled,
                "cache_enabled": self.cache_enabled,
                "request_overrides": self.allow_request_overrides,
            },
            "label_weight_overrides": self.label_weights,
        }


def _int_port(value: str) -> int:
    try:
        port = int(value)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}FETCH_ALLOWED_PORTS contiene un puerto inválido: '{value}'") from exc
    if not 1 <= port <= 65535:
        raise ConfigError(f"{ENV_PREFIX}FETCH_ALLOWED_PORTS contiene un puerto fuera de rango: {port}")
    return port
