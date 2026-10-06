"""Core domain: image handling, detection engine and moderation policy."""

from .engine import Detection, DetectionEngine, MockEngine, NudeNetEngine, build_engine
from .imaging import SafeImage, load_image, render_censored, sniff_mime
from .labels import MODEL_LABELS, Severity, catalogue, label_info
from .policy import (
    VERDICT_ALLOW,
    VERDICT_BLOCK,
    VERDICT_REVIEW,
    AnalysisResult,
    PolicyConfig,
    evaluate,
    verdict_for,
)

__all__ = [
    "MODEL_LABELS",
    "VERDICT_ALLOW",
    "VERDICT_BLOCK",
    "VERDICT_REVIEW",
    "AnalysisResult",
    "Detection",
    "DetectionEngine",
    "MockEngine",
    "NudeNetEngine",
    "PolicyConfig",
    "SafeImage",
    "Severity",
    "build_engine",
    "catalogue",
    "evaluate",
    "label_info",
    "load_image",
    "render_censored",
    "sniff_mime",
    "verdict_for",
]
