"""Detection engines.

The API talks to an engine, never directly to a model library.  That keeps the
HTTP layer testable without ONNX, and lets you swap the model (a bigger
checkpoint, a cloud provider, a fine-tuned classifier) by changing one class.

Contract:

    engine.analyze(image: SafeImage) -> list[list[Detection]]

one list of detections per analysed frame (images are usually single-frame;
animated ones contribute several).
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import Settings
from ..errors import InferenceError, ModelUnavailable
from .imaging import SafeImage
from .labels import MODEL_LABELS, Severity, label_info

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One model finding, normalised to the API's own vocabulary."""

    label: str
    score: float
    box: tuple[int, int, int, int]  # (x1, y1, x2, y2) in frame pixels
    frame_index: int
    area_ratio: float
    severity: Severity
    weight: float

    def to_dict(self, *, counted: bool = True) -> dict[str, Any]:
        info = label_info(self.label)
        x1, y1, x2, y2 = self.box
        return {
            "label": self.label,
            "confidence": round(self.score, 4),
            "severity": self.severity.value,
            "weight": round(self.weight, 4),
            "box": [x1, y1, x2, y2],
            "box_xywh": [x1, y1, x2 - x1, y2 - y1],
            "area_ratio": round(self.area_ratio, 6),
            "frame_index": self.frame_index,
            "counted": counted,
            "description_es": info.description_es,
        }

    def to_cache_dict(self) -> dict[str, Any]:
        """Full-precision representation used by the result cache."""
        x1, y1, x2, y2 = self.box
        return {
            "label": self.label,
            "confidence": self.score,
            "box": [x1, y1, x2, y2],
            "frame_index": self.frame_index,
            "area_ratio": self.area_ratio,
            "severity": self.severity.value,
            "weight": self.weight,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Detection:
        """Rebuild a detection from a cached/serialised representation."""
        x1, y1, x2, y2 = payload["box"]
        return cls(
            label=str(payload["label"]),
            score=float(payload["confidence"]),
            box=(int(x1), int(y1), int(x2), int(y2)),
            frame_index=int(payload.get("frame_index", 0)),
            area_ratio=float(payload.get("area_ratio", 0.0)),
            severity=Severity(str(payload.get("severity", "suggestive"))),
            weight=float(payload.get("weight", 0.0)),
        )


class DetectionEngine(ABC):
    """Interface every backend must implement."""

    #: Short identifier echoed in API responses.
    name: str = "abstract"
    #: Human readable model identifier.
    model: str = "unknown"
    #: Version of the backend library / weights.
    version: str = "0.0.0"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._loaded_at: float | None = None

    # ------------------------------------------------------------------ #
    def load(self) -> None:  # noqa: B027 - optional hook, not every engine needs it
        """Idempotent, thread-safe initialisation."""

    def warmup(self) -> None:  # noqa: B027 - optional hook, not every engine needs it
        """Run a dummy inference so the first real request is fast."""

    @property
    def is_ready(self) -> bool:
        return self._loaded_at is not None

    def describe(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "model": self.model,
            "version": self.version,
            "labels": list(MODEL_LABELS),
            "ready": self.is_ready,
            "loaded_at_epoch": self._loaded_at,
            "inference_resolution": self.settings.inference_resolution,
            "min_confidence": self.settings.min_confidence,
        }

    @abstractmethod
    def analyze(self, image: SafeImage) -> list[list[Detection]]:
        """Return detections per analysed frame."""


class NudeNetEngine(DetectionEngine):
    """YOLOv8-nano nudity detector shipped inside the ``nudenet`` wheel.

    The ONNX session is created once and shared: ``onnxruntime`` is thread-safe
    for ``run()``, and a semaphore caps how many inferences execute at once so a
    burst of uploads cannot exhaust the CPU.
    """

    name = "nudenet"
    model = "320n.onnx"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._detector: Any = None
        self._init_lock = threading.Lock()
        self._inference_semaphore = threading.Semaphore(settings.max_concurrent_inferences)
        self.version = self._nudenet_version()

    @staticmethod
    def _nudenet_version() -> str:
        try:
            from importlib.metadata import PackageNotFoundError, version

            return version("nudenet")
        except Exception:  # pragma: no cover - metadata is best effort
            try:
                from importlib.metadata import PackageNotFoundError  # noqa: F401

                return "unknown"
            except Exception:
                return "unknown"

    # ------------------------------------------------------------------ #
    def load(self) -> None:
        if self._detector is not None:
            return
        with self._init_lock:
            if self._detector is not None:
                return
            started = time.monotonic()
            try:
                from nudenet import NudeDetector

                self._detector = NudeDetector(
                    model_path=self.settings.model_path,
                    inference_resolution=self.settings.inference_resolution,
                )
            except ImportError as exc:
                raise ModelUnavailable(
                    "El paquete 'nudenet' no está instalado.",
                    details={"hint": "pip install -r requirements.txt"},
                ) from exc
            except Exception as exc:
                raise ModelUnavailable(
                    "No se pudo cargar el modelo de detección.",
                    details={"reason": type(exc).__name__},
                ) from exc

            self.model = (
                "custom.onnx"
                if self.settings.model_path
                else f"320n@{self.settings.inference_resolution}"
            )
            self._loaded_at = time.time()
            logger.info(
                "modelo_cargado engine=%s model=%s ms=%.1f",
                self.name,
                self.model,
                (time.monotonic() - started) * 1000,
            )

    def warmup(self) -> None:
        self.load()
        dummy = np.zeros((64, 64, 4), dtype=np.uint8)
        dummy[..., 3] = 255
        self._run_detector(dummy)

    # ------------------------------------------------------------------ #
    def _run_detector(self, rgba: np.ndarray) -> list[dict[str, Any]]:
        assert self._detector is not None
        with self._inference_semaphore:
            try:
                return self._detector.detect(rgba)
            except Exception as exc:  # pragma: no cover - depends on model internals
                logger.exception("fallo_inferencia")
                raise InferenceError(
                    "El modelo no pudo procesar la imagen.",
                    details={"reason": type(exc).__name__},
                ) from exc

    def analyze(self, image: SafeImage) -> list[list[Detection]]:
        self.load()
        results: list[list[Detection]] = []
        for frame_index, frame in enumerate(image.frames):
            frame_height, frame_width = frame.shape[:2]
            total_area = float(frame_width * frame_height) or 1.0
            raw = self._run_detector(frame)
            detections: list[Detection] = []
            for item in raw:
                label = str(item.get("class", "")).upper()
                score = float(item.get("score", 0.0))
                x, y, w, h = (float(v) for v in item.get("box", (0, 0, 0, 0)))
                x1 = int(max(0, min(x, frame_width)))
                y1 = int(max(0, min(y, frame_height)))
                x2 = int(max(0, min(x + w, frame_width)))
                y2 = int(max(0, min(y + h, frame_height)))
                info = label_info(label)
                detections.append(
                    Detection(
                        label=label,
                        score=score,
                        box=(x1, y1, x2, y2),
                        frame_index=frame_index,
                        area_ratio=max(0.0, ((x2 - x1) * (y2 - y1)) / total_area),
                        severity=info.severity,
                        weight=self._weight_for(label),
                    )
                )
            detections.sort(key=lambda d: (d.weight * d.score), reverse=True)
            results.append(detections)
        return results

    def _weight_for(self, label: str) -> float:
        overrides = self.settings.label_weights
        if label in overrides:
            return float(overrides[label])
        return label_info(label).weight


class MockEngine(DetectionEngine):
    """Engine that always reports "nothing found".

    Intended **only** for wiring, load and integration tests in environments
    without ONNX.  It is deliberately loud: the name appears in every response
    and in ``/health`` so it can never be mistaken for a working moderator.
    """

    name = "mock"
    model = "mock-none"
    version = "0.0.0"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        logger.warning(
            "El motor 'mock' está activo: NINGUNA imagen será marcada como NSFW. "
            "No uses NID_ENGINE=mock en producción."
        )

    def load(self) -> None:
        if self._loaded_at is not None:
            return
        self._loaded_at = time.time()

    def warmup(self) -> None:
        self.load()

    def analyze(self, image: SafeImage) -> list[list[Detection]]:
        self.load()
        return [[] for _ in image.frames]


class DisabledEngine(DetectionEngine):
    """Placeholder used when detection is switched off (e.g. static docs serving)."""

    name = "disabled"
    model = "none"
    version = "0.0.0"

    def analyze(self, image: SafeImage) -> list[list[Detection]]:
        raise ModelUnavailable(
            "El motor de detección está deshabilitado en este despliegue.",
            details={"hint": "Establece NID_ENGINE=nudenet"},
        )


def build_engine(settings: Settings) -> DetectionEngine:
    """Instantiate the engine selected by configuration."""
    engines: dict[str, type[DetectionEngine]] = {
        "nudenet": NudeNetEngine,
        "mock": MockEngine,
        "disabled": DisabledEngine,
    }
    engine_cls = engines.get(settings.engine, NudeNetEngine)
    return engine_cls(settings)


def flatten(detections_per_frame: Sequence[Sequence[Detection]]) -> list[Detection]:
    """Flatten per-frame detections into a single list."""
    return [detection for frame in detections_per_frame for detection in frame]
