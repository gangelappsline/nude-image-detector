"""Policy engine: detections in, verdict out."""

from __future__ import annotations

import pytest

from app.config import PROFILES, Settings
from app.core.policy import (
    VERDICT_ALLOW,
    VERDICT_BLOCK,
    VERDICT_REVIEW,
    PolicyConfig,
    area_factor,
    contribution,
    evaluate,
    noisy_or,
    verdict_for,
)

from .conftest import base_settings, detection


# --------------------------------------------------------------------------- #
# Maths
# --------------------------------------------------------------------------- #
def test_noisy_or_of_nothing_is_zero() -> None:
    assert noisy_or([]) == 0.0


def test_noisy_or_of_single_value_is_that_value() -> None:
    assert noisy_or([0.4]) == pytest.approx(0.4)


def test_noisy_or_is_monotonic_and_bounded() -> None:
    one = noisy_or([0.6])
    two = noisy_or([0.6, 0.6])
    three = noisy_or([0.6, 0.6, 0.6])
    assert one < two < three <= 1.0


def test_noisy_or_clamps_out_of_range_inputs() -> None:
    assert 0.0 <= noisy_or([-1.0, 2.0]) <= 1.0


def test_area_factor_is_bounded_by_floor_and_one(policy: PolicyConfig) -> None:
    assert area_factor(0.0, policy) == pytest.approx(policy.area_floor)
    assert area_factor(policy.area_saturation, policy) == pytest.approx(1.0)
    assert area_factor(10.0, policy) == pytest.approx(1.0)


def test_bigger_boxes_contribute_more(policy: PolicyConfig) -> None:
    small = detection("FEMALE_BREAST_EXPOSED", 0.9, area_ratio=0.0001)
    large = detection("FEMALE_BREAST_EXPOSED", 0.9, area_ratio=0.20)
    assert contribution(large, policy) > contribution(small, policy)


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
def test_clean_image_is_allowed(policy: PolicyConfig) -> None:
    result = evaluate([[]], policy)
    assert result.verdict == VERDICT_ALLOW
    assert result.nsfw is False
    assert result.risk_score == 0.0
    assert result.reasons == []
    assert result.explicit_labels == []


def test_neutral_labels_never_trigger(policy: PolicyConfig) -> None:
    frame = [
        detection("FACE_FEMALE", 0.99),
        detection("FACE_MALE", 0.99),
        detection("FEET_EXPOSED", 0.9),
        detection("BELLY_COVERED", 0.8),
    ]
    result = evaluate([frame], policy)
    assert result.verdict == VERDICT_ALLOW
    assert result.risk_score == 0.0


def test_explicit_nudity_is_blocked(policy: PolicyConfig) -> None:
    result = evaluate([[detection("FEMALE_GENITALIA_EXPOSED", 0.92, area_ratio=0.05)]], policy)
    assert result.verdict == VERDICT_BLOCK
    assert result.nsfw is True
    assert result.explicit_score > policy.block_threshold
    assert result.explicit_labels == ["FEMALE_GENITALIA_EXPOSED"]
    assert "explicit_nudity" in result.flags


def test_exposed_breast_is_blocked_under_balanced(policy: PolicyConfig) -> None:
    result = evaluate([[detection("FEMALE_BREAST_EXPOSED", 0.9, area_ratio=0.04)]], policy)
    assert result.verdict == VERDICT_BLOCK


def test_weak_explicit_signal_lands_in_review(policy: PolicyConfig) -> None:
    result = evaluate([[detection("FEMALE_BREAST_EXPOSED", 0.35, area_ratio=0.0005)]], policy)
    assert result.verdict in {VERDICT_REVIEW, VERDICT_ALLOW}
    assert result.verdict != VERDICT_BLOCK


def test_suggestive_only_is_not_blocked_under_balanced(policy: PolicyConfig) -> None:
    frame = [
        detection("FEMALE_GENITALIA_COVERED", 0.8, area_ratio=0.03),
        detection("BELLY_EXPOSED", 0.7, area_ratio=0.05),
    ]
    result = evaluate([frame], policy)
    assert result.suggestive_score > 0
    assert result.verdict == VERDICT_REVIEW


def test_strict_profile_blocks_suggestive_content(strict_policy: PolicyConfig) -> None:
    frame = [detection("FEMALE_GENITALIA_COVERED", 0.85, area_ratio=0.04)]
    result = evaluate([frame], strict_policy)
    assert result.verdict == VERDICT_BLOCK


def test_lenient_profile_allows_suggestive_content(lenient_policy: PolicyConfig) -> None:
    frame = [detection("BELLY_EXPOSED", 0.9, area_ratio=0.1)]
    result = evaluate([frame], lenient_policy)
    assert result.verdict == VERDICT_ALLOW


def test_lenient_profile_still_blocks_clear_nudity(lenient_policy: PolicyConfig) -> None:
    result = evaluate([[detection("MALE_GENITALIA_EXPOSED", 0.97, area_ratio=0.08)]], lenient_policy)
    assert result.verdict == VERDICT_BLOCK


def test_shirtless_man_is_not_blocked_by_default(policy: PolicyConfig) -> None:
    """The most common false positive of naive filters must not block."""
    frame = [
        detection("MALE_BREAST_EXPOSED", 0.95, area_ratio=0.2),
        detection("BELLY_EXPOSED", 0.9, area_ratio=0.15),
        detection("FACE_MALE", 0.99, area_ratio=0.02),
    ]
    result = evaluate([frame], policy)
    assert result.verdict != VERDICT_BLOCK


def test_below_min_confidence_is_ignored(policy: PolicyConfig) -> None:
    frame = [detection("FEMALE_GENITALIA_EXPOSED", policy.min_confidence - 0.01, area_ratio=0.2)]
    result = evaluate([frame], policy)
    assert result.counted_detections == []
    assert result.verdict == VERDICT_ALLOW
    assert "low_confidence_findings_only" in result.flags
    # It is still reported so a moderator can see why nothing was counted.
    assert len(result.detections) == 1


def test_label_weight_override_can_neutralise_a_label() -> None:
    settings = base_settings(label_weights={"FEMALE_BREAST_EXPOSED": 0.0})
    policy = PolicyConfig.from_settings(settings)
    result = evaluate([[detection("FEMALE_BREAST_EXPOSED", 0.99, area_ratio=0.3)]], policy)
    assert result.verdict == VERDICT_ALLOW


def test_label_weight_override_can_escalate_a_label() -> None:
    settings = base_settings(label_weights={"MALE_BREAST_EXPOSED": 1.0})
    policy = PolicyConfig.from_settings(settings)
    result = evaluate([[detection("MALE_BREAST_EXPOSED", 0.95, area_ratio=0.2)]], policy)
    assert result.verdict == VERDICT_BLOCK


def test_multiple_findings_saturate_towards_one(policy: PolicyConfig) -> None:
    frame = [
        detection("FEMALE_GENITALIA_EXPOSED", 0.9, area_ratio=0.05),
        detection("FEMALE_BREAST_EXPOSED", 0.9, area_ratio=0.05),
        detection("BUTTOCKS_EXPOSED", 0.9, area_ratio=0.05),
    ]
    result = evaluate([frame], policy)
    assert result.risk_score > 0.95
    assert "multiple_explicit_findings" in result.flags


def test_animated_frames_are_aggregated_by_worst_frame(policy: PolicyConfig) -> None:
    frames = [
        [],
        [],
        [detection("FEMALE_BREAST_EXPOSED", 0.93, area_ratio=0.06, frame_index=2)],
    ]
    result = evaluate(frames, policy)
    assert result.verdict == VERDICT_BLOCK
    assert result.frames_with_findings == 1
    assert result.counted_detections[0].frame_index == 2


def test_verdict_thresholds_are_inclusive(policy: PolicyConfig) -> None:
    assert verdict_for(policy.block_threshold, policy) == VERDICT_BLOCK
    assert verdict_for(policy.review_threshold, policy) == VERDICT_REVIEW
    assert verdict_for(0.0, policy) == VERDICT_ALLOW


# --------------------------------------------------------------------------- #
# Policy resolution
# --------------------------------------------------------------------------- #
def test_policy_from_profile_defaults() -> None:
    settings = base_settings(strictness="strict")
    policy = PolicyConfig.from_settings(settings)
    assert policy.profile_name == "strict"
    assert policy.block_threshold == PROFILES["strict"].block_threshold


def test_request_overrides_win_over_environment() -> None:
    settings = base_settings(strictness="balanced", block_threshold=0.9)
    policy = PolicyConfig.from_settings(settings, block_threshold=0.2)
    assert policy.block_threshold == 0.2


def test_environment_overrides_win_over_profile() -> None:
    settings = base_settings(strictness="balanced", review_threshold=0.05)
    policy = PolicyConfig.from_settings(settings)
    assert policy.review_threshold == 0.05


def test_inverted_thresholds_are_repaired() -> None:
    settings = base_settings()
    policy = PolicyConfig.from_settings(settings, block_threshold=0.2, review_threshold=0.9)
    assert policy.review_threshold <= policy.block_threshold


def test_unknown_profile_falls_back_to_deployment_default() -> None:
    settings = base_settings(strictness="lenient")
    policy = PolicyConfig.from_settings(settings, profile="not-a-profile")
    assert policy.profile_name == "lenient"


def test_policy_to_dict_is_serialisable(policy: PolicyConfig) -> None:
    payload = policy.to_dict()
    assert payload["profile"] == "balanced"
    assert payload["block_threshold"] == pytest.approx(PROFILES["balanced"].block_threshold)
    import json

    json.dumps(payload)


# --------------------------------------------------------------------------- #
# Response shape
# --------------------------------------------------------------------------- #
def test_result_to_dict_contract(policy: PolicyConfig) -> None:
    result = evaluate([[detection("ANUS_EXPOSED", 0.88, area_ratio=0.04)]], policy)
    payload = result.to_dict()

    for key in (
        "verdict",
        "nsfw",
        "risk_score",
        "scores",
        "reasons",
        "flags",
        "detections",
        "policy",
        "severity_counts",
    ):
        assert key in payload

    assert payload["scores"]["explicit"] > 0
    assert payload["detections"][0]["label"] == "ANUS_EXPOSED"
    assert payload["detections"][0]["counted"] is True
    assert set(payload["detections"][0]["box"]) >= set()
    assert payload["severity_counts"]["explicit"] == 1


def test_result_to_dict_can_hide_detections(policy: PolicyConfig) -> None:
    result = evaluate([[detection("ANUS_EXPOSED", 0.88)]], policy)
    payload = result.to_dict(include_detections=False)
    assert "detections" not in payload
    assert payload["verdict"] == VERDICT_BLOCK


def test_reasons_are_human_readable_and_spanish(policy: PolicyConfig) -> None:
    result = evaluate([[detection("FEMALE_BREAST_EXPOSED", 0.91, area_ratio=0.03)]], policy)
    assert result.reasons
    reason = result.reasons[0]
    assert "FEMALE_BREAST_EXPOSED" in reason
    assert "confianza" in reason
    assert "área" in reason


def test_sensitive_boxes_only_include_serious_findings(policy: PolicyConfig) -> None:
    frame = [
        detection("FEMALE_GENITALIA_EXPOSED", 0.9, box=(1, 1, 10, 10)),
        detection("FACE_FEMALE", 0.99, box=(20, 20, 40, 40)),
        detection("BELLY_EXPOSED", 0.2, box=(50, 50, 60, 60)),
    ]
    result = evaluate([frame], policy)
    assert (1, 1, 10, 10) in result.sensitive_boxes
    assert (20, 20, 40, 40) not in result.sensitive_boxes


def test_settings_validate_rejects_analysis_pixels_above_limit() -> None:
    from app.config import ConfigError

    with pytest.raises(ConfigError):
        Settings(engine="mock", max_image_pixels=100, max_analysis_pixels=10_000).validate()
