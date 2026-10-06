"""Analysis service: the single orchestration point of the API.

Flow for one image::

    bytes ──> SafeImage ──> engine.analyze ──> detections ──> policy.evaluate ──> verdict
              (validate)     (ONNX, cached)                    (thresholds)

The cache stores **detections, not verdicts**: two clients may ask for the same
photo with different strictness profiles, and re-scoring a cached detection list
is free while re-running the model is not.
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from flask import g, has_request_context

from .config import Settings
from .core.cache import TTLCache
from .core.engine import Detection, DetectionEngine
from .core.fetcher import FetchResult, fetch_image_url
from .core.imaging import SafeImage, load_image, render_censored
from .core.policy import VERDICT_ALLOW, AnalysisResult, PolicyConfig, evaluate
from .errors import BadRequest
from .logging_conf import redact_url

logger = logging.getLogger(__name__)


@dataclass
class PolicyOverrides:
    """Per-request policy tweaks (only honoured when the deployment allows it)."""

    strictness: str | None = None
    block_threshold: float | None = None
    review_threshold: float | None = None
    suggestive_factor: float | None = None
    min_confidence: str | None = None

    def resolve(self, settings: Settings) -> PolicyConfig:
        if not settings.allow_request_overrides:
            return PolicyConfig.from_settings(settings)

        def _as_float(value: Any, name: str) -> float | None:
            if value is None or value == "":
                return None
            try:
                parsed = float(value)
            except (TypeError, ValueError) as exc:
                raise BadRequest(f"El parámetro '{name}' debe ser un número entre 0 y 1.") from exc
            if not 0.0 <= parsed <= 1.0:
                raise BadRequest(f"El parámetro '{name}' debe estar entre 0 y 1.")
            return parsed

        strictness = self.strictness
        if strictness is not None:
            from .config import PROFILES

            strictness = str(strictness).lower().strip()
            if strictness not in PROFILES:
                raise BadRequest(
                    f"Perfil de estrictez desconocido: '{strictness}'.",
                    details={"accepted": sorted(PROFILES)},
                )

        return PolicyConfig.from_settings(
            settings,
            profile=strictness,
            block_threshold=_as_float(self.block_threshold, "block_threshold"),
            review_threshold=_as_float(self.review_threshold, "review_threshold"),
            suggestive_factor=_as_float(self.suggestive_factor, "suggestive_factor"),
            min_confidence=_as_float(self.min_confidence, "min_confidence"),
        )


@dataclass
class AnalysisOutcome:
    """Everything the API needs to serialise one analysis."""

    image: SafeImage
    result: AnalysisResult
    policy: PolicyConfig
    source: dict[str, Any]
    cached: bool
    elapsed_ms: float
    censored: bytes | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(
        self,
        *,
        request_id: str | None = None,
        include_detections: bool = True,
        model_info: dict[str, Any] | None = None,
        censored_format: str = "png",
    ) -> dict[str, Any]:
        payload = self.result.to_dict(include_detections=include_detections)
        payload.update(
            {
                "request_id": request_id or _current_request_id(),
                "image": self.image.describe(),
                "source": self.source,
                "model": model_info or {},
                "cached": self.cached,
                "elapsed_ms": round(self.elapsed_ms, 1),
            }
        )
        if self.warnings:
            payload["warnings"] = self.warnings
        if self.censored:
            encoded = base64.b64encode(self.censored).decode("ascii")
            payload["censored_image"] = {
                "format": censored_format,
                "bytes": len(self.censored),
                "data_url": f"data:image/{censored_format};base64,{encoded}",
            }
        return payload


def _current_request_id() -> str | None:
    if has_request_context():
        return getattr(g, "request_id", None)
    return None


class AnalysisService:
    """Stateless orchestration over a long-lived engine and cache."""

    def __init__(
        self,
        settings: Settings,
        engine: DetectionEngine,
        cache: TTLCache | None = None,
    ) -> None:
        self.settings = settings
        self.engine = engine
        self.cache = cache if cache is not None else TTLCache(
            maxsize=settings.cache_maxsize,
            ttl=settings.cache_ttl_seconds,
            enabled=settings.cache_enabled,
        )

    # ------------------------------------------------------------------ #
    def analyze_bytes(
        self,
        data: bytes,
        *,
        overrides: PolicyOverrides | None = None,
        source: dict[str, Any] | None = None,
        declared_mime: str | None = None,
        filename: str | None = None,
        include_censored: bool = False,
        include_detections: bool = True,
    ) -> AnalysisOutcome:
        started = time.monotonic()
        policy = (overrides or PolicyOverrides()).resolve(self.settings)

        image = load_image(
            data, self.settings, declared_mime=declared_mime, filename=filename
        )

        warnings: list[str] = []
        if image.mime_mismatch:
            warnings.append(
                f"El Content-Type declarado ({image.declared_mime}) no coincide con el "
                f"formato real ({image.detected_mime})."
            )
        if image.is_animated and image.frames_total > len(image.frames):
            warnings.append(
                f"Imagen animada con {image.frames_total} fotogramas; se analizaron "
                f"{len(image.frames)} (NID_MAX_FRAMES)."
            )
        if image.downscaled:
            warnings.append(
                f"La imagen se redujo a {image.width}x{image.height} antes de la inferencia "
                f"(original {image.source_width}x{image.source_height})."
            )

        detections, cached, inference_ms = self._detect(image)
        result = evaluate(detections, policy)
        result.inference_ms = inference_ms
        result.analysis_ms = (time.monotonic() - started) * 1000

        censored = None
        if include_censored and result.verdict != VERDICT_ALLOW:
            boxes = result.sensitive_boxes
            if boxes:
                censored = render_censored(
                    image, boxes, strength=self.settings.blur_strength
                )

        return AnalysisOutcome(
            image=image,
            result=result,
            policy=policy,
            source=source or {"type": "upload", "filename": filename},
            cached=cached,
            elapsed_ms=(time.monotonic() - started) * 1000,
            censored=censored,
            warnings=warnings,
        )

    def analyze_url(
        self,
        url: str,
        *,
        overrides: PolicyOverrides | None = None,
        include_censored: bool = False,
        include_detections: bool = True,
    ) -> AnalysisOutcome:
        fetch: FetchResult = fetch_image_url(url, self.settings)
        source = fetch.describe()
        outcome = self.analyze_bytes(
            fetch.data,
            overrides=overrides,
            source=source,
            declared_mime=fetch.content_type,
            filename=None,
            include_censored=include_censored,
            include_detections=include_detections,
        )
        return outcome

    # ------------------------------------------------------------------ #
    def _detect(self, image: SafeImage) -> tuple[list[list[Detection]], bool, float]:
        """Run the engine, using and populating the SHA-256 keyed cache.

        Returns ``(detections_per_frame, came_from_cache, inference_ms)``.
        """
        key = f"{self.engine.name}:{self.engine.version}:{image.sha256}"
        cached_payload = self.cache.get(key)
        if cached_payload is not None:
            return (
                [[Detection.from_dict(item) for item in frame] for frame in cached_payload],
                True,
                0.0,
            )

        started = time.monotonic()
        detections = self.engine.analyze(image)
        elapsed = (time.monotonic() - started) * 1000
        logger.info(
            "inferencia_completada frames=%d detections=%d ms=%.1f",
            len(detections),
            sum(len(frame) for frame in detections),
            elapsed,
            extra={"engine": self.engine.name, "inference_ms": round(elapsed, 1)},
        )
        self.cache.set(key, [[d.to_cache_dict() for d in frame] for frame in detections])
        return detections, False, elapsed

    # ------------------------------------------------------------------ #
    def health(self) -> dict[str, Any]:
        stats = self.cache.stats().to_dict()
        return {
            "status": "ok" if self.engine.is_ready else "degraded",
            "engine": self.engine.name,
            "model_ready": self.engine.is_ready,
            "cache": stats,
        }


def describe_source_upload(filename: str | None, size: int) -> dict[str, Any]:
    """Source metadata for a multipart upload (never the content itself)."""
    return {
        "type": "upload",
        "filename": filename,
        "bytes": size,
    }


def describe_source_raw(size: int, content_type: str | None) -> dict[str, Any]:
    """Source metadata for a request whose body *is* the image."""
    return {"type": "raw_body", "bytes": size, "content_type": content_type}


def describe_source_url(url: str) -> dict[str, Any]:
    """Source metadata for URL mode, with credentials/query strings stripped."""
    return {"type": "url", "url": redact_url(url)}


__all__ = [
    "AnalysisOutcome",
    "AnalysisService",
    "PolicyOverrides",
    "describe_source_raw",
    "describe_source_upload",
    "describe_source_url",
]
