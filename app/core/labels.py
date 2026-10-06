"""Label catalogue produced by the detection model.

The model (NudeNet ``320n.onnx``, a YOLOv8-nano trained on ~160k images) emits
17 body-part labels.  Raw labels are *not* a moderation decision on their own:
"MALE_BREAST_EXPOSED" (a shirtless man at the beach) and
"FEMALE_GENITALIA_EXPOSED" are very different things for a product.

Every label is therefore mapped to a :class:`Severity` tier and a default
weight in ``[0, 1]``.  Weights are the single knob you need to tune the policy
for your community (see ``NID_LABEL_WEIGHTS`` in ``.env.example``).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType


class Severity(str, Enum):
    """How damaging a detected body part is for a moderation policy."""

    EXPLICIT = "explicit"
    SUGGESTIVE = "suggestive"
    NEUTRAL = "neutral"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.value


@dataclass(frozen=True)
class LabelInfo:
    """Static metadata about one model label."""

    name: str
    severity: Severity
    weight: float
    description_es: str


def _info(name: str, severity: Severity, weight: float, description_es: str) -> LabelInfo:
    return LabelInfo(name=name, severity=severity, weight=weight, description_es=description_es)


_E = Severity.EXPLICIT
_S = Severity.SUGGESTIVE
_N = Severity.NEUTRAL

#: Full catalogue emitted by the bundled model, with default policy weights.
LABELS: Mapping[str, LabelInfo] = MappingProxyType(
    {
        info.name: info
        for info in (
            # --- Explicit nudity: the thing you almost always want to block ---
            _info("FEMALE_GENITALIA_EXPOSED", _E, 1.00, "Genitales femeninos expuestos"),
            _info("MALE_GENITALIA_EXPOSED", _E, 1.00, "Genitales masculinos expuestos"),
            _info("ANUS_EXPOSED", _E, 1.00, "Ano expuesto"),
            _info("BUTTOCKS_EXPOSED", _E, 0.90, "Glúteos expuestos (desnudos)"),
            _info("FEMALE_BREAST_EXPOSED", _E, 0.85, "Pecho femenino expuesto (pezón/areola)"),
            # --- Suggestive: covered genitals/buttocks, swimwear-ish context ---
            _info("FEMALE_GENITALIA_COVERED", _S, 0.55, "Zona genital femenina cubierta (ropa interior/bikini)"),
            _info("MALE_GENITALIA_COVERED", _S, 0.45, "Zona genital masculina cubierta"),
            _info("ANUS_COVERED", _S, 0.50, "Ano cubierto por ropa"),
            _info("BUTTOCKS_COVERED", _S, 0.45, "Glúteos cubiertos por ropa"),
            _info("FEMALE_BREAST_COVERED", _S, 0.35, "Pecho femenino cubierto (top/bikini)"),
            _info("MALE_BREAST_EXPOSED", _S, 0.15, "Torso masculino descubierto (normal en playa/deporte)"),
            _info("BELLY_EXPOSED", _S, 0.10, "Vientre/abdomen descubierto"),
            _info("ARMPITS_EXPOSED", _S, 0.05, "Axilas visibles"),
            # --- Neutral: present in nearly every portrait, must not drive verdicts ---
            _info("FACE_FEMALE", _N, 0.00, "Rostro femenino"),
            _info("FACE_MALE", _N, 0.00, "Rostro masculino"),
            _info("FEET_EXPOSED", _N, 0.00, "Pies descubiertos"),
            _info("FEET_COVERED", _N, 0.00, "Pies cubiertos (calzado)"),
            _info("BELLY_COVERED", _N, 0.00, "Vientre cubierto por ropa"),
            _info("ARMPITS_COVERED", _N, 0.00, "Axilas cubiertas por ropa"),
            _info("HANDS", _N, 0.00, "Manos"),
            _info("LIPS", _N, 0.00, "Labios"),
            _info("NECK", _N, 0.00, "Cuello"),
            _info("LEGS", _N, 0.00, "Piernas"),
            _info("ARMS", _N, 0.00, "Brazos"),
        )
    }
)

#: Labels the bundled model can actually emit (others exist for model swaps).
MODEL_LABELS: tuple[str, ...] = (
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
)

#: Explicit labels only - handy for "hard block, no nuance" integrations.
EXPLICIT_LABELS: frozenset[str] = frozenset(
    name for name, info in LABELS.items() if info.severity is Severity.EXPLICIT
)

SUGGESTIVE_LABELS: frozenset[str] = frozenset(
    name for name, info in LABELS.items() if info.severity is Severity.SUGGESTIVE
)


def label_info(name: str) -> LabelInfo:
    """Return metadata for ``name``.

    Unknown labels (e.g. after swapping the model for a newer checkpoint) are
    treated as *suggestive* with a conservative weight instead of crashing: a
    new label from an NSFW model is far more likely to be risky than benign.
    """
    info = LABELS.get(name)
    if info is not None:
        return info
    return LabelInfo(
        name=name,
        severity=Severity.SUGGESTIVE,
        weight=0.5,
        description_es="Etiqueta desconocida (tratada como sugerente por seguridad)",
    )


def severity_of(name: str) -> Severity:
    return label_info(name).severity


def weight_of(name: str, overrides: Mapping[str, float] | None = None) -> float:
    """Weight of ``name``, honouring operator-provided overrides."""
    if overrides and name in overrides:
        return float(overrides[name])
    return label_info(name).weight


def catalogue(overrides: Mapping[str, float] | None = None) -> list[dict[str, object]]:
    """Serialisable catalogue, used by ``GET /v1/labels`` and the docs page."""
    out: list[dict[str, object]] = []
    for name in MODEL_LABELS:
        info = label_info(name)
        out.append(
            {
                "label": name,
                "severity": info.severity.value,
                "weight": round(weight_of(name, overrides), 4),
                "default_weight": info.weight,
                "description_es": info.description_es,
                "overridden": bool(overrides and name in overrides),
            }
        )
    return out
