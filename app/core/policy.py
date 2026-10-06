"""From raw detections to a moderation decision.

A detector only answers *"what body parts are visible, with what confidence"*.
The policy layer answers the product question: **should this upload be allowed?**

Scoring model
-------------
Each detection contributes a probability-like value

    p = weight(label) * confidence * area_factor(box)

* ``weight`` encodes how serious the body part is (see :mod:`app.core.labels`).
* ``confidence`` is the model's score.
* ``area_factor`` shrinks the contribution of tiny boxes (a speck in the
  background matters less than a close-up), saturating at
  ``area_saturation`` of the frame.

Contributions are combined with a **noisy-OR** (``1 - Π(1 - p)`), which is the
right aggregation for "any one of these findings is enough": it saturates near 1
instead of blowing past it like a sum would.

Two scores come out: ``explicit`` and ``suggestive``.  The final risk is

    risk = max(explicit, suggestive_factor * suggestive)

so swimwear/bikini evidence can reach the review queue but only reaches "block"
when the profile allows it.  A single strong explicit detection also gets a
floor (``explicit_floor_factor``) - "FEMALE_GENITALIA_EXPOSED at 0.8" should not
be rescued by the area heuristic.

Everything is deterministic and explainable: the response lists the exact
detections that drove the verdict, so a moderator can audit a false positive.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from .engine import Detection
from .labels import Severity

#: Verdicts, ordered from most to least restrictive.
VERDICT_BLOCK = "block"
VERDICT_REVIEW = "review"
VERDICT_ALLOW = "allow"

VERDICTS: tuple[str, str, str] = (VERDICT_BLOCK, VERDICT_REVIEW, VERDICT_ALLOW)

VERDICT_DESCRIPTIONS_ES: dict[str, str] = {
    VERDICT_BLOCK: "Rechaza la subida: se detectó desnudez con confianza suficiente.",
    VERDICT_REVIEW: "No bloquea por sí solo, pero debería pasar por revisión humana.",
    VERDICT_ALLOW: "No se detectó contenido problemático.",
}


@dataclass(frozen=True)
class PolicyConfig:
    """Everything needed to turn detections into a verdict."""

    block_threshold: float
    review_threshold: float
    suggestive_factor: float
    min_confidence: float
    profile_name: str = "balanced"
    #: Boxes covering at least this fraction of the frame count at full strength.
    area_saturation: float = 0.01
    #: Multiplier applied to boxes of negligible size.
    area_floor: float = 0.75
    #: Floor for a single strong explicit detection.
    explicit_floor_factor: float = 0.90
    #: Operator/request overrides on top of the catalogue's default weights.
    label_weights: Mapping[str, float] = field(default_factory=dict)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        profile: str | None = None,
        block_threshold: float | None = None,
        review_threshold: float | None = None,
        suggestive_factor: float | None = None,
        min_confidence: float | None = None,
        label_weights: Mapping[str, float] | None = None,
    ) -> PolicyConfig:
        """Resolve effective policy: profile defaults <- deployment env <- request."""
        from ..config import PROFILES

        profile_name = (profile or settings.strictness or "balanced").lower()
        preset = PROFILES.get(profile_name) or settings.profile
        profile_name = preset.name

        block = _first_defined(block_threshold, settings.block_threshold, preset.block_threshold)
        review = _first_defined(review_threshold, settings.review_threshold, preset.review_threshold)
        suggestive = _first_defined(
            suggestive_factor, settings.suggestive_factor, preset.suggestive_factor
        )
        confidence = _first_defined(min_confidence, settings.min_confidence)

        if review > block:
            # Keep the invariant even when a client sends nonsense overrides.
            review = block

        weights: dict[str, float] = dict(settings.label_weights)
        if label_weights:
            weights.update({str(k): float(v) for k, v in label_weights.items()})

        return cls(
            block_threshold=float(block),
            review_threshold=float(review),
            suggestive_factor=float(suggestive),
            min_confidence=float(confidence),
            profile_name=profile_name,
            label_weights=weights,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile_name,
            "block_threshold": round(self.block_threshold, 4),
            "review_threshold": round(self.review_threshold, 4),
            "suggestive_factor": round(self.suggestive_factor, 4),
            "min_confidence": round(self.min_confidence, 4),
            "area_saturation": self.area_saturation,
            "area_floor": self.area_floor,
            "explicit_floor_factor": self.explicit_floor_factor,
            "label_weight_overrides": dict(self.label_weights),
        }


def _first_defined(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    raise ValueError("ningún valor definido para la política")


def noisy_or(values: Iterable[float]) -> float:
    """Probabilistic OR: ``1 - Π(1 - p_i)`` with every ``p_i`` clamped to [0, 1)."""
    complement = 1.0
    for value in values:
        clamped = min(max(float(value), 0.0), 0.999999)
        complement *= 1.0 - clamped
    return 1.0 - complement


def area_factor(area_ratio: float, policy: PolicyConfig) -> float:
    """Scale a detection by how much of the frame it occupies."""
    ratio = max(0.0, min(1.0, float(area_ratio) / policy.area_saturation))
    return policy.area_floor + (1.0 - policy.area_floor) * ratio


def contribution(detection: Detection, policy: PolicyConfig) -> float:
    """Policy-weighted probability contributed by one detection."""
    return (
        detection.weight
        * detection.score
        * area_factor(detection.area_ratio, policy)
    )


def verdict_for(risk_score: float, policy: PolicyConfig) -> str:
    if risk_score >= policy.block_threshold:
        return VERDICT_BLOCK
    if risk_score >= policy.review_threshold:
        return VERDICT_REVIEW
    return VERDICT_ALLOW


@dataclass
class AnalysisResult:
    """The moderated outcome for one image."""

    risk_score: float
    explicit_score: float
    suggestive_score: float
    verdict: str
    nsfw: bool
    reasons: list[str]
    flags: list[str]
    detections: list[Detection]
    counted_detections: list[Detection]
    severity_counts: dict[str, int]
    labels_found: list[str]
    max_confidence: float
    top_label: str | None
    frames_with_findings: int
    policy: PolicyConfig
    inference_ms: float = 0.0
    analysis_ms: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    @property
    def explicit_labels(self) -> list[str]:
        return sorted(
            {d.label for d in self.counted_detections if d.severity is Severity.EXPLICIT}
        )

    @property
    def sensitive_boxes(self) -> list[tuple[int, int, int, int]]:
        """Boxes worth censoring (explicit + strong suggestive findings)."""
        return [
            d.box
            for d in self.counted_detections
            if d.severity is Severity.EXPLICIT
            or (d.severity is Severity.SUGGESTIVE and d.score >= 0.5)
        ]

    def to_dict(self, *, include_detections: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "verdict": self.verdict,
            "verdict_description_es": VERDICT_DESCRIPTIONS_ES[self.verdict],
            "nsfw": self.nsfw,
            "risk_score": round(self.risk_score, 4),
            "scores": {
                "explicit": round(self.explicit_score, 4),
                "suggestive": round(self.suggestive_score, 4),
            },
            "reasons": self.reasons,
            "flags": self.flags,
            "explicit_labels": self.explicit_labels,
            "labels_found": self.labels_found,
            "severity_counts": self.severity_counts,
            "top_label": self.top_label,
            "max_confidence": round(self.max_confidence, 4),
            "frames_with_findings": self.frames_with_findings,
            "policy": self.policy.to_dict(),
            "timing_ms": {
                "inference": round(self.inference_ms, 1),
                "analysis": round(self.analysis_ms, 1),
            },
        }
        if include_detections:
            counted = {id(d) for d in self.counted_detections}
            payload["detections"] = [
                d.to_dict(counted=id(d) in counted) for d in self.detections
            ]
        return payload


def evaluate(
    detections_per_frame: Sequence[Sequence[Detection]],
    policy: PolicyConfig,
) -> AnalysisResult:
    """Apply ``policy`` to the detections of every analysed frame."""
    if policy.label_weights:
        detections_per_frame = [
            [
                Detection(
                    label=d.label,
                    score=d.score,
                    box=d.box,
                    frame_index=d.frame_index,
                    area_ratio=d.area_ratio,
                    severity=d.severity,
                    weight=float(policy.label_weights.get(d.label, d.weight)),
                )
                for d in frame
            ]
            for frame in detections_per_frame
        ]

    all_detections: list[Detection] = [d for frame in detections_per_frame for d in frame]
    counted = [d for d in all_detections if d.score >= policy.min_confidence]

    explicit_values = [
        contribution(d, policy) for d in counted if d.severity is Severity.EXPLICIT
    ]
    suggestive_values = [
        contribution(d, policy) for d in counted if d.severity is Severity.SUGGESTIVE
    ]

    explicit_or = noisy_or(explicit_values)
    explicit_peak = max(
        (d.weight * d.score for d in counted if d.severity is Severity.EXPLICIT),
        default=0.0,
    )
    explicit_score = max(explicit_or, explicit_peak * policy.explicit_floor_factor)
    suggestive_score = noisy_or(suggestive_values)

    risk = max(explicit_score, policy.suggestive_factor * suggestive_score)
    risk = min(max(risk, 0.0), 1.0)
    verdict = verdict_for(risk, policy)

    severity_counts: Counter[str] = Counter()
    for detection in counted:
        severity_counts[detection.severity.value] += 1
    for severity in Severity:
        severity_counts.setdefault(severity.value, 0)

    best_by_label: dict[str, Detection] = {}
    for detection in counted:
        current = best_by_label.get(detection.label)
        if current is None or detection.score > current.score:
            best_by_label[detection.label] = detection

    ranked = sorted(
        best_by_label.values(), key=lambda d: (d.weight * d.score, d.area_ratio), reverse=True
    )
    reasons = [_reason_for(d, policy) for d in ranked[:5]]

    flags: list[str] = []
    if any(d.severity is Severity.EXPLICIT for d in counted):
        flags.append("explicit_nudity")
    if any(d.severity is Severity.SUGGESTIVE for d in counted):
        flags.append("suggestive_content")
    if sum(1 for d in counted if d.severity is Severity.EXPLICIT) > 1:
        flags.append("multiple_explicit_findings")
    frames_with_findings = len({d.frame_index for d in counted if d.weight > 0})
    if frames_with_findings > 1:
        flags.append("findings_in_multiple_frames")
    if not counted and all_detections:
        flags.append("low_confidence_findings_only")

    top = ranked[0] if ranked else None

    return AnalysisResult(
        risk_score=risk,
        explicit_score=explicit_score,
        suggestive_score=suggestive_score,
        verdict=verdict,
        nsfw=verdict == VERDICT_BLOCK,
        reasons=reasons,
        flags=flags,
        detections=all_detections,
        counted_detections=counted,
        severity_counts=dict(severity_counts),
        labels_found=sorted({d.label for d in counted}),
        max_confidence=max((d.score for d in all_detections), default=0.0),
        top_label=top.label if top else None,
        frames_with_findings=frames_with_findings,
        policy=policy,
    )


def _reason_for(detection: Detection, policy: PolicyConfig) -> str:
    from .labels import label_info

    info = label_info(detection.label)
    return (
        f"{detection.label}: {info.description_es} "
        f"(confianza {detection.score:.2f}, área {detection.area_ratio * 100:.1f}%, "
        f"peso {detection.weight:.2f})"
    )


def empty_result(policy: PolicyConfig) -> AnalysisResult:
    """Result for an image with no findings at all."""
    return evaluate([[]], policy)
