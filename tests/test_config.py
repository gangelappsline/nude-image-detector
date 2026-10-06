"""Configuration parsing and validation."""

from __future__ import annotations

import os

import pytest

from app.config import PROFILES, ConfigError, Settings


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove every NID_ variable so tests start from a known baseline."""
    for key in list(os.environ):
        if key.startswith("NID_"):
            monkeypatch.delenv(key, raising=False)


def test_defaults_are_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    settings = Settings.from_env(load_dotenv_file=False)

    assert settings.engine == "nudenet"
    assert settings.strictness == "balanced"
    assert settings.fetch_enabled is True
    # The SSRF guard must be ON by default - that is the whole point.
    assert settings.fetch_allow_private_networks is False
    assert settings.auth_enabled is False
    assert settings.max_content_length == 10 * 1024 * 1024
    assert "169.254.169.254/32" in settings.fetch_blocked_cidrs


def test_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("NID_ENGINE", "mock")
    monkeypatch.setenv("NID_STRICTNESS", "strict")
    monkeypatch.setenv("NID_BLOCK_THRESHOLD", "0.42")
    monkeypatch.setenv("NID_MIN_CONFIDENCE", "0.5")
    monkeypatch.setenv("NID_API_KEYS", "k1, k2 ,k3")
    monkeypatch.setenv("NID_LABEL_WEIGHTS", '{"MALE_BREAST_EXPOSED": 0.0}')
    monkeypatch.setenv("NID_FETCH_ALLOWED_PORTS", "80,8080")

    settings = Settings.from_env(load_dotenv_file=False)

    assert settings.engine == "mock"
    assert settings.strictness == "strict"
    assert settings.block_threshold == 0.42
    assert settings.min_confidence == 0.5
    assert settings.api_keys == ("k1", "k2", "k3")
    assert settings.auth_enabled is True
    assert settings.label_weights == {"MALE_BREAST_EXPOSED": 0.0}
    assert settings.fetch_allowed_ports == (80, 8080)


@pytest.mark.parametrize("value", ["true", "1", "yes", "ON", "True"])
def test_booleans_accept_common_spellings(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("NID_CACHE_ENABLED", value)
    assert Settings.from_env(load_dotenv_file=False).cache_enabled is True


@pytest.mark.parametrize("value", ["maybe", "2", ""])
def test_invalid_boolean_fails_fast(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("NID_CACHE_ENABLED", value)
    if value == "":
        # An empty variable falls back to the default instead of erroring.
        assert Settings.from_env(load_dotenv_file=False).cache_enabled is True
    else:
        with pytest.raises(ConfigError):
            Settings.from_env(load_dotenv_file=False)


def test_invalid_engine_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("NID_ENGINE", "tensorflow-magic")
    with pytest.raises(ConfigError, match="ENGINE"):
        Settings.from_env(load_dotenv_file=False)


def test_invalid_strictness_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("NID_STRICTNESS", "extreme")
    with pytest.raises(ConfigError, match="STRICTNESS"):
        Settings.from_env(load_dotenv_file=False)


def test_review_threshold_cannot_exceed_block(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("NID_BLOCK_THRESHOLD", "0.3")
    monkeypatch.setenv("NID_REVIEW_THRESHOLD", "0.9")
    with pytest.raises(ConfigError, match="revisión"):
        Settings.from_env(load_dotenv_file=False)


def test_invalid_label_weights_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("NID_LABEL_WEIGHTS", '{"FACE_MALE": "mucho"}')
    with pytest.raises(ConfigError):
        Settings.from_env(load_dotenv_file=False)

    monkeypatch.setenv("NID_LABEL_WEIGHTS", "[1, 2]")
    with pytest.raises(ConfigError):
        Settings.from_env(load_dotenv_file=False)

    monkeypatch.setenv("NID_LABEL_WEIGHTS", "not json")
    with pytest.raises(ConfigError):
        Settings.from_env(load_dotenv_file=False)


def test_out_of_range_numbers_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("NID_PORT", "99999")
    with pytest.raises(ConfigError):
        Settings.from_env(load_dotenv_file=False)

    monkeypatch.delenv("NID_PORT")
    monkeypatch.setenv("NID_MIN_CONFIDENCE", "1.8")
    with pytest.raises(ConfigError):
        Settings.from_env(load_dotenv_file=False)


def test_invalid_port_list_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("NID_FETCH_ALLOWED_PORTS", "80,notaport")
    with pytest.raises(ConfigError, match="puerto"):
        Settings.from_env(load_dotenv_file=False)


def test_profiles_are_ordered_from_strict_to_lenient() -> None:
    assert PROFILES["strict"].block_threshold < PROFILES["balanced"].block_threshold
    assert PROFILES["balanced"].block_threshold < PROFILES["lenient"].block_threshold
    for profile in PROFILES.values():
        assert profile.review_threshold <= profile.block_threshold
        assert 0.0 <= profile.suggestive_factor <= 1.0


def test_effective_thresholds_prefer_explicit_overrides() -> None:
    settings = Settings(engine="mock", strictness="balanced", block_threshold=0.11)
    assert settings.effective_block_threshold == 0.11
    assert settings.effective_review_threshold == PROFILES["balanced"].review_threshold


def test_public_dict_never_leaks_secrets() -> None:
    settings = Settings(engine="nudenet", api_keys=("super-secret",))
    rendered = repr(settings.public_dict())
    assert "super-secret" not in rendered
    assert settings.public_dict()["features"]["auth_required"] is True
