"""Engine unit tests.

The real ONNX model is exercised in ``test_model_e2e.py``; here we stub the
detector so we can assert on *normalisation* (box conversion, clipping,
ordering, per-frame results) and on failure handling deterministically.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np
import pytest

from app.config import Settings
from app.core.engine import (
    Detection,
    DisabledEngine,
    MockEngine,
    NudeNetEngine,
    build_engine,
    flatten,
)
from app.core.labels import MODEL_LABELS, label_info
from app.errors import InferenceError, ModelUnavailable

from .conftest import base_settings, safe_image


class FakeDetector:
    """Mimics ``nudenet.NudeDetector.detect`` (boxes are ``[x, y, w, h]``)."""

    def __init__(self, results=None, *, delay: float = 0.0, error: Exception | None = None) -> None:
        self.results = results if results is not None else []
        self.calls = 0
        self.delay = delay
        self.error = error
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()
        self.seen_shapes: list[tuple[int, ...]] = []

    def detect(self, rgba: np.ndarray):
        with self._lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            self.calls += 1
            self.seen_shapes.append(tuple(rgba.shape))
            if self.delay:
                time.sleep(self.delay)
            if self.error is not None:
                raise self.error
            if callable(self.results):
                return self.results(rgba)
            return list(self.results)
        finally:
            with self._lock:
                self.concurrent -= 1


def _engine_with(fake: FakeDetector, settings: Settings | None = None) -> NudeNetEngine:
    """Build a NudeNetEngine whose detector is already 'loaded' (no ONNX)."""
    engine = NudeNetEngine(settings or base_settings(engine="nudenet"))
    engine._detector = fake
    engine._loaded_at = time.time()
    return engine


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_build_engine_selects_the_configured_backend() -> None:
    assert isinstance(build_engine(base_settings(engine="nudenet")), NudeNetEngine)
    assert isinstance(build_engine(base_settings(engine="mock")), MockEngine)
    assert isinstance(build_engine(base_settings(engine="disabled")), DisabledEngine)


def test_mock_engine_never_reports_anything() -> None:
    engine = build_engine(base_settings(engine="mock"))
    image = safe_image(frames=2)
    assert engine.analyze(image) == [[], []]
    assert engine.is_ready is True


def test_disabled_engine_refuses_to_work() -> None:
    engine = build_engine(base_settings(engine="disabled"))
    with pytest.raises(ModelUnavailable):
        engine.analyze(safe_image())
    assert engine.is_ready is False


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def test_xywh_boxes_are_converted_to_xyxy() -> None:
    fake = FakeDetector([{"class": "FEMALE_BREAST_EXPOSED", "score": 0.9, "box": [10, 20, 30, 40]}])
    engine = _engine_with(fake)
    image = safe_image(width=200, height=100)

    (detections,) = engine.analyze(image)
    assert len(detections) == 1
    detection = detections[0]
    assert detection.box == (10, 20, 40, 60)  # x1, y1, x2, y2
    assert detection.area_ratio == pytest.approx((30 * 40) / (200 * 100))
    assert detection.severity.value == "explicit"
    assert detection.weight == pytest.approx(label_info("FEMALE_BREAST_EXPOSED").weight)


def test_boxes_are_clipped_to_the_frame() -> None:
    fake = FakeDetector([{"class": "BUTTOCKS_EXPOSED", "score": 0.8, "box": [-50, -20, 400, 400]}])
    engine = _engine_with(fake)
    image = safe_image(width=64, height=64)

    (detections,) = engine.analyze(image)
    x1, y1, x2, y2 = detections[0].box
    assert (x1, y1) == (0, 0)
    assert (x2, y2) == (64, 64)
    assert 0.0 <= detections[0].area_ratio <= 1.0


def test_frames_receive_rgba_arrays_of_the_expected_shape() -> None:
    fake = FakeDetector([])
    engine = _engine_with(fake)
    engine.analyze(safe_image(width=48, height=32, frames=3))
    assert fake.calls == 3
    assert fake.seen_shapes == [(32, 48, 4)] * 3


def test_frame_index_is_propagated() -> None:
    payload = [{"class": "FACE_FEMALE", "score": 0.9, "box": [0, 0, 10, 10]}]
    fake = FakeDetector(lambda rgba: list(payload))
    engine = _engine_with(fake)
    frames = engine.analyze(safe_image(frames=3))
    assert [d.frame_index for frame in frames for d in frame] == [0, 1, 2]


def test_detections_are_sorted_by_policy_weight_then_score() -> None:
    fake = FakeDetector(
        [
            {"class": "FACE_FEMALE", "score": 0.99, "box": [0, 0, 10, 10]},
            {"class": "FEMALE_GENITALIA_EXPOSED", "score": 0.6, "box": [0, 0, 10, 10]},
            {"class": "BELLY_EXPOSED", "score": 0.9, "box": [0, 0, 10, 10]},
        ]
    )
    engine = _engine_with(fake)
    (detections,) = engine.analyze(safe_image())
    assert detections[0].label == "FEMALE_GENITALIA_EXPOSED"
    assert detections[-1].label == "FACE_FEMALE"


def test_unknown_labels_are_treated_as_suggestive_not_crash() -> None:
    fake = FakeDetector([{"class": "BRAND_NEW_LABEL", "score": 0.7, "box": [1, 1, 5, 5]}])
    engine = _engine_with(fake)
    (detections,) = engine.analyze(safe_image())
    assert detections[0].severity.value == "suggestive"
    assert detections[0].weight == 0.5


def test_label_weight_overrides_are_applied_by_the_engine() -> None:
    settings = base_settings(engine="nudenet", label_weights={"FACE_FEMALE": 0.42})
    fake = FakeDetector([{"class": "FACE_FEMALE", "score": 0.9, "box": [0, 0, 10, 10]}])
    engine = _engine_with(fake, settings)
    (detections,) = engine.analyze(safe_image())
    assert detections[0].weight == 0.42


# --------------------------------------------------------------------------- #
# Failure handling
# --------------------------------------------------------------------------- #
def test_inference_errors_are_wrapped() -> None:
    fake = FakeDetector(error=RuntimeError("segmentation fault in onnx"))
    engine = _engine_with(fake)
    with pytest.raises(InferenceError) as excinfo:
        engine.analyze(safe_image())
    # Internals must not leak to the client.
    assert "segmentation fault" not in str(excinfo.value.message)


def test_missing_nudenet_package_is_reported_as_model_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "nudenet", None)
    engine = NudeNetEngine(base_settings(engine="nudenet"))
    with pytest.raises(ModelUnavailable):
        engine.load()


def test_broken_model_path_is_reported_as_model_unavailable() -> None:
    engine = NudeNetEngine(base_settings(engine="nudenet", model_path="/does/not/exist.onnx"))
    with pytest.raises(ModelUnavailable):
        engine.load()


# --------------------------------------------------------------------------- #
# Concurrency
# --------------------------------------------------------------------------- #
def test_semaphore_caps_concurrent_inferences() -> None:
    settings = base_settings(engine="nudenet", max_concurrent_inferences=1)
    fake = FakeDetector([], delay=0.05)
    engine = _engine_with(fake, settings)

    threads = [
        threading.Thread(target=lambda: engine.analyze(safe_image())) for _ in range(4)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert fake.calls == 4
    assert fake.max_concurrent == 1


def test_concurrent_requests_do_not_corrupt_results() -> None:
    settings = base_settings(engine="nudenet", max_concurrent_inferences=4)
    payload = [{"class": "ANUS_EXPOSED", "score": 0.77, "box": [2, 2, 8, 8]}]
    fake = FakeDetector(lambda rgba: list(payload), delay=0.01)
    engine = _engine_with(fake, settings)

    collected: list[list[Detection]] = []
    lock = threading.Lock()

    def run() -> None:
        (detections,) = engine.analyze(safe_image())
        with lock:
            collected.append(detections)

    threads = [threading.Thread(target=run) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert len(collected) == 8
    assert all(len(d) == 1 and d[0].label == "ANUS_EXPOSED" for d in collected)


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
def test_describe_exposes_the_label_catalogue() -> None:
    engine = build_engine(base_settings(engine="mock"))
    engine.load()
    described = engine.describe()
    assert described["engine"] == "mock"
    assert described["ready"] is True
    assert set(described["labels"]) == set(MODEL_LABELS)


def test_warmup_marks_the_engine_ready() -> None:
    engine = MockEngine(base_settings(engine="mock"))
    assert engine.is_ready is False
    engine.warmup()
    assert engine.is_ready is True


def test_load_is_idempotent() -> None:
    engine = MockEngine(base_settings(engine="mock"))
    engine.load()
    first = engine._loaded_at
    engine.load()
    assert engine._loaded_at == first


def test_flatten_helper() -> None:
    a = Detection("FACE_FEMALE", 0.9, (0, 0, 1, 1), 0, 0.01, label_info("FACE_FEMALE").severity, 0.0)
    b = Detection("ANUS_EXPOSED", 0.9, (0, 0, 1, 1), 1, 0.01, label_info("ANUS_EXPOSED").severity, 1.0)
    assert flatten([[a], [b]]) == [a, b]
