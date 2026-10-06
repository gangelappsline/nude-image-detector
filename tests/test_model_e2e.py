"""End-to-end tests with the real ONNX model.

Marked ``slow`` (``pytest -m "not slow"`` skips them).  They use synthetic
images only: the repository never stores, downloads or generates explicit
content, and the assertions are written so a false positive in random noise
cannot make them flaky.
"""

from __future__ import annotations

import io
import time

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.config import Settings
from app.core.engine import NudeNetEngine
from app.core.imaging import load_image
from app.core.labels import MODEL_LABELS, label_info
from app.core.policy import PolicyConfig, evaluate

from .conftest import base_settings, noise_image, skin_tone_image, solid_image

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def real_engine() -> NudeNetEngine:
    settings = base_settings(engine="nudenet", cache_enabled=False)
    engine = NudeNetEngine(settings)
    engine.load()
    engine.warmup()
    return engine


def _detections_for(engine: NudeNetEngine, data: bytes, settings: Settings):
    image = load_image(data, settings)
    return image, engine.analyze(image)


def test_model_ships_inside_the_wheel_and_loads_offline(real_engine: NudeNetEngine) -> None:
    assert real_engine.is_ready is True
    assert real_engine.version  # resolved from package metadata
    described = real_engine.describe()
    assert described["engine"] == "nudenet"
    assert described["ready"] is True


def test_label_catalogue_covers_every_model_label() -> None:
    """A model label missing from the catalogue would silently weaken moderation."""
    for label in MODEL_LABELS:
        info = label_info(label)
        assert info.name == label
        assert 0.0 <= info.weight <= 1.0
        assert info.description_es


def test_plain_colours_produce_no_detections(real_engine: NudeNetEngine) -> None:
    settings = base_settings(engine="nudenet")
    for color in ((255, 255, 255), (0, 0, 0), (30, 144, 255), (34, 139, 34)):
        _image, frames = _detections_for(real_engine, solid_image(128, 128, color), settings)
        assert frames == [[]], f"un color plano no debería detectar nada ({color})"
        assert evaluate(frames, PolicyConfig.from_settings(settings)).verdict == "allow"


def test_geometric_drawing_produces_no_detections(real_engine: NudeNetEngine) -> None:
    settings = base_settings(engine="nudenet")
    canvas = Image.new("RGB", (240, 180), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([20, 20, 100, 160], fill=(200, 30, 30))
    draw.ellipse([120, 30, 220, 130], fill=(30, 90, 200))
    draw.line([0, 0, 240, 180], fill=(20, 20, 20), width=4)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")

    _, frames = _detections_for(real_engine, buffer.getvalue(), settings)
    assert frames == [[]]


def test_flat_skin_tone_square_is_not_flagged(real_engine: NudeNetEngine) -> None:
    """The classic false positive of naive "skin pixel" filters must not happen."""
    settings = base_settings(engine="nudenet")
    image, frames = _detections_for(real_engine, skin_tone_image(160, 160), settings)
    explicit = [d for frame in frames for d in frame if d.severity.value == "explicit"]
    assert explicit == []
    result = evaluate(frames, PolicyConfig.from_settings(settings))
    assert result.verdict == "allow"
    assert image.detected_mime == "image/jpeg"


def test_random_noise_is_never_blocked(real_engine: NudeNetEngine) -> None:
    settings = base_settings(engine="nudenet")
    for seed in (1, 2, 3):
        _, frames = _detections_for(real_engine, noise_image(128, 128, seed=seed), settings)
        result = evaluate(frames, PolicyConfig.from_settings(settings))
        assert result.verdict != "block", f"ruido aleatorio (seed={seed}) no debería bloquearse"


def test_inference_is_fast_on_cpu(real_engine: NudeNetEngine) -> None:
    settings = base_settings(engine="nudenet")
    image = load_image(solid_image(320, 320, fmt="JPEG"), settings)
    started = time.monotonic()
    real_engine.analyze(image)
    elapsed = time.monotonic() - started
    # Generous bound: the model takes ~30 ms on 2 vCPU; CI machines vary a lot.
    assert elapsed < 3.0, f"la inferencia tardó {elapsed:.2f}s"


def test_large_images_are_downscaled_and_still_analysed(real_engine: NudeNetEngine) -> None:
    settings = base_settings(engine="nudenet", max_analysis_pixels=100_000)
    image = load_image(solid_image(2000, 1500, fmt="JPEG"), settings)
    assert image.downscaled is True
    assert image.width * image.height <= 100_000
    frames = real_engine.analyze(image)
    assert len(frames) == 1


def test_api_end_to_end_with_the_real_model() -> None:
    from app import create_app

    settings = base_settings(
        engine="nudenet", load_model_at_startup=True, warmup=True, cache_enabled=True
    )
    app = create_app(settings, testing=True)
    client = app.test_client()

    assert client.get("/ready").status_code == 200

    response = client.post(
        "/v1/analyze",
        data={"file": (io.BytesIO(solid_image(200, 200, (120, 200, 90))), "verde.png", "image/png")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["verdict"] == "allow"
    assert body["model"]["engine"] == "nudenet"
    assert body["cached"] is False

    # The very same bytes must now come from the cache.
    again = client.post(
        "/v1/analyze",
        data={"file": (io.BytesIO(solid_image(200, 200, (120, 200, 90))), "verde.png", "image/png")},
        content_type="multipart/form-data",
    ).get_json()
    assert again["cached"] is True
    assert again["image"]["sha256"] == body["image"]["sha256"]


def test_batch_end_to_end_with_the_real_model() -> None:
    import base64

    from app import create_app

    app = create_app(base_settings(engine="nudenet", cache_enabled=False), testing=True)
    client = app.test_client()
    raw = base64.b64encode(noise_image(96, 96, seed=42)).decode()

    response = client.post(
        "/v1/analyze/batch",
        json={"items": [{"image_base64": raw}, {"image_base64": raw}]},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["summary"]["total"] == 2
    assert body["summary"]["succeeded"] == 2
    assert body["summary"]["any_blocked"] is False


def test_gradient_image_round_trip_keeps_rgba_contract(real_engine: NudeNetEngine) -> None:
    """The model expects 4-channel RGBA; verify our pipeline always delivers it."""
    settings = base_settings(engine="nudenet")
    array = np.zeros((80, 120, 3), dtype=np.uint8)
    array[:, :, 0] = np.linspace(0, 255, 120, dtype=np.uint8)
    array[:, :, 2] = np.linspace(0, 255, 80, dtype=np.uint8)[:, None]
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="PNG")

    image = load_image(buffer.getvalue(), settings)
    assert image.primary_frame.shape[2] == 4
    real_engine.analyze(image)
