"""HTTP contract tests for the API (mock engine: fast and deterministic)."""

from __future__ import annotations

import base64
import io
import json

import pytest

from app.config import Settings

from .conftest import base_settings, detection, noise_image, solid_image


def _upload(client, data: bytes, *, name: str = "file", filename: str = "foto.png", mime: str = "image/png", **query):
    return client.post(
        "/v1/analyze" + (f"?{'&'.join(f'{k}={v}' for k, v in query.items())}" if query else ""),
        data={name: (io.BytesIO(data), filename, mime)},
        content_type="multipart/form-data",
    )


@pytest.fixture
def blocked_app(settings: Settings, mocker: pytest.MockerFixture):
    """App whose engine always reports explicit nudity."""
    from app import create_app

    app = create_app(settings, testing=True)
    engine = app.extensions["nid_engine"]
    mocker.patch.object(
        engine,
        "analyze",
        return_value=[[detection("FEMALE_GENITALIA_EXPOSED", 0.96, area_ratio=0.07)]],
    )
    return app


# --------------------------------------------------------------------------- #
# Service endpoints
# --------------------------------------------------------------------------- #
def test_health_is_public_and_reports_engine(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] in {"ok", "degraded"}
    assert body["engine"] == "mock"
    assert "cache" in body


def test_ready_reports_model_state(client) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.get_json()["ready"] is True


def test_info_exposes_policy_without_secrets() -> None:
    settings = base_settings(api_keys=("top-secret-key",))
    from app import create_app

    app = create_app(settings, testing=True)
    body = app.test_client().get("/v1/info").get_json()

    assert "top-secret-key" not in json.dumps(body)
    assert body["configuration"]["features"]["auth_required"] is True
    assert set(body["profiles"]) == {"strict", "balanced", "lenient"}
    assert body["configuration"]["thresholds"]["block"] == pytest.approx(0.55)


def test_labels_catalogue_is_complete(client) -> None:
    body = client.get("/v1/labels").get_json()
    assert body["label_count"] >= 17
    labels = {item["label"]: item for item in body["labels"]}
    assert labels["FEMALE_GENITALIA_EXPOSED"]["severity"] == "explicit"
    assert labels["FACE_FEMALE"]["severity"] == "neutral"
    assert labels["FACE_FEMALE"]["weight"] == 0.0


def test_openapi_document_is_valid_json(client) -> None:
    body = client.get("/openapi.json").get_json()
    assert body["openapi"].startswith("3.")
    assert "/v1/analyze" in body["paths"]
    assert "/v1/analyze/batch" in body["paths"]


def test_docs_page_renders(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Nude Image Detector" in response.get_data(as_text=True)


def test_unknown_route_returns_json_envelope(client) -> None:
    response = client.get("/v1/nope")
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "not_found"


def test_wrong_method_returns_json_envelope(client) -> None:
    response = client.get("/v1/analyze")
    assert response.status_code == 405
    assert response.get_json()["error"]["code"] == "method_not_allowed"


# --------------------------------------------------------------------------- #
# Analyze: input modes
# --------------------------------------------------------------------------- #
def test_analyze_multipart_upload(client) -> None:
    response = _upload(client, solid_image(64, 48))
    assert response.status_code == 200
    body = response.get_json()
    assert body["verdict"] == "allow"
    assert body["nsfw"] is False
    assert body["risk_score"] == 0.0
    assert body["source"]["type"] == "upload"
    assert body["source"]["filename"] == "foto.png"
    assert body["image"]["width"] == 64
    assert body["image"]["sha256"]
    assert body["request_id"]
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_analyze_accepts_any_file_field_name(client) -> None:
    for name in ("image", "files", "whatever"):
        response = _upload(client, solid_image(32, 32), name=name)
        assert response.status_code == 200, name


def test_analyze_json_base64(client) -> None:
    payload = {"image_base64": base64.b64encode(solid_image(40, 40)).decode()}
    response = client.post("/v1/analyze", json=payload)
    assert response.status_code == 200
    body = response.get_json()
    assert body["source"]["type"] == "base64"
    assert body["image"]["width"] == 40


def test_analyze_json_base64_accepts_data_url(client) -> None:
    raw = base64.b64encode(solid_image(24, 24)).decode()
    response = client.post("/v1/analyze", json={"image_base64": f"data:image/png;base64,{raw}"})
    assert response.status_code == 200


def test_analyze_raw_binary_body(client) -> None:
    response = client.post(
        "/v1/analyze", data=solid_image(30, 20), content_type="image/png"
    )
    assert response.status_code == 200
    assert response.get_json()["source"]["type"] == "raw_body"


def test_analyze_url_mode(url_client, local_base_url: str) -> None:
    response = url_client.post("/v1/analyze", json={"url": f"{local_base_url}/image"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["source"]["type"] == "url"
    assert body["source"]["host"] == "127.0.0.1"
    assert body["verdict"] == "allow"


def test_analyze_url_mode_rejects_private_targets(client) -> None:
    response = client.post("/v1/analyze", json={"url": "http://169.254.169.254/latest/meta-data/"})
    assert response.status_code == 403
    assert response.get_json()["error"]["code"] == "blocked_destination"


def test_analyze_url_disabled() -> None:
    from app import create_app

    app = create_app(base_settings(fetch_enabled=False), testing=True)
    response = app.test_client().post("/v1/analyze", json={"url": "https://example.com/a.jpg"})
    assert response.status_code == 403
    assert response.get_json()["error"]["code"] == "forbidden"


def test_analyze_without_image_is_400(client) -> None:
    response = client.post("/v1/analyze", json={})
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "missing_image"


def test_analyze_malformed_json_is_400(client) -> None:
    response = client.post("/v1/analyze", data="{not json", content_type="application/json")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "bad_request"


def test_analyze_invalid_base64_is_400(client) -> None:
    response = client.post("/v1/analyze", json={"image_base64": "!!!not-base64!!!"})
    assert response.status_code == 400
    assert "base64" in response.get_json()["error"]["message"].lower()


def test_analyze_non_image_upload_is_422(client) -> None:
    response = _upload(client, b"this is not an image at all", filename="nota.png")
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "unprocessable_image"


def test_analyze_empty_file_is_400(client) -> None:
    response = _upload(client, b"")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "bad_request"


def test_analyze_rejects_oversized_body() -> None:
    from app import create_app

    app = create_app(base_settings(max_content_length=2048), testing=True)
    big = noise_image(200, 200, seed=2)
    assert len(big) > 2048
    response = app.test_client().post(
        "/v1/analyze", data={"file": (io.BytesIO(big), "big.jpg", "image/jpeg")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "payload_too_large"


def test_analyze_rejects_pixel_bomb() -> None:
    from app import create_app

    app = create_app(
        base_settings(max_image_pixels=1000, max_analysis_pixels=1000, max_content_length=5_000_000),
        testing=True,
    )
    response = _upload(app.test_client(), solid_image(200, 200))
    assert response.status_code == 422
    assert response.get_json()["error"]["code"] == "image_too_large"


# --------------------------------------------------------------------------- #
# Verdict behaviour
# --------------------------------------------------------------------------- #
def test_blocked_image_returns_200_with_block_verdict(blocked_app) -> None:
    response = _upload(blocked_app.test_client(), solid_image(64, 64))
    assert response.status_code == 200
    body = response.get_json()
    assert body["verdict"] == "block"
    assert body["nsfw"] is True
    assert body["risk_score"] > 0.5
    assert body["explicit_labels"] == ["FEMALE_GENITALIA_EXPOSED"]
    assert body["reasons"]


def test_reject_on_block_returns_422(blocked_app) -> None:
    response = _upload(blocked_app.test_client(), solid_image(64, 64), reject_on_block="true")
    assert response.status_code == 422
    assert response.get_json()["verdict"] == "block"


def test_censor_returns_a_data_url(blocked_app) -> None:
    response = _upload(blocked_app.test_client(), noise_image(96, 96, seed=9), censor="true")
    body = response.get_json()
    assert body["verdict"] == "block"
    censored = body["censored_image"]
    assert censored["format"] == "png"
    assert censored["data_url"].startswith("data:image/png;base64,")
    decoded = base64.b64decode(censored["data_url"].split(",", 1)[1])
    assert decoded[:8] == b"\x89PNG\r\n\x1a\n"


def test_censor_is_skipped_when_allowed(client) -> None:
    response = _upload(client, solid_image(64, 64), censor="true")
    assert "censored_image" not in response.get_json()


def test_no_detections_hides_the_list(client) -> None:
    with_detections = _upload(client, solid_image(64, 64)).get_json()
    assert "detections" in with_detections
    without = _upload(client, solid_image(64, 64), no_detections="true").get_json()
    assert "detections" not in without


def test_strictness_override_changes_the_policy(client, mocker: pytest.MockerFixture) -> None:
    """A suggestive-only image: allowed under lenient, blocked under strict."""
    app = client.application
    engine = app.extensions["nid_engine"]
    mocker.patch.object(
        engine,
        "analyze",
        return_value=[[detection("FEMALE_GENITALIA_COVERED", 0.85, area_ratio=0.05)]],
    )

    lenient = _upload(client, solid_image(64, 64), strictness="lenient").get_json()
    strict = _upload(client, solid_image(64, 64), strictness="strict").get_json()

    assert lenient["policy"]["profile"] == "lenient"
    assert strict["policy"]["profile"] == "strict"
    assert lenient["verdict"] != "block"
    assert strict["verdict"] == "block"


def test_threshold_overrides_are_applied(client) -> None:
    body = _upload(client, solid_image(32, 32), block_threshold="0.9", review_threshold="0.8").get_json()
    assert body["policy"]["block_threshold"] == 0.9
    assert body["policy"]["review_threshold"] == 0.8


def test_invalid_strictness_is_400(client) -> None:
    response = _upload(client, solid_image(32, 32), strictness="extremo")
    assert response.status_code == 400
    assert response.get_json()["error"]["details"]["accepted"]


def test_invalid_threshold_is_400(client) -> None:
    response = _upload(client, solid_image(32, 32), block_threshold="mucho")
    assert response.status_code == 400

    response = _upload(client, solid_image(32, 32), block_threshold="1.5")
    assert response.status_code == 400


def test_request_overrides_can_be_disabled() -> None:
    from app import create_app

    app = create_app(base_settings(allow_request_overrides=False, strictness="balanced"), testing=True)
    body = _upload(app.test_client(), solid_image(32, 32), strictness="strict").get_json()
    assert body["policy"]["profile"] == "balanced"


def test_detection_below_min_confidence_is_reported_but_not_counted(
    client, mocker: pytest.MockerFixture
) -> None:
    app = client.application
    engine = app.extensions["nid_engine"]
    mocker.patch.object(
        engine, "analyze", return_value=[[detection("FEMALE_BREAST_EXPOSED", 0.22, area_ratio=0.2)]]
    )
    body = _upload(client, solid_image(64, 64)).get_json()
    assert body["verdict"] == "allow"
    assert body["detections"][0]["counted"] is False


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #
def test_repeated_image_is_served_from_cache() -> None:
    from app import create_app

    app = create_app(base_settings(cache_enabled=True, cache_maxsize=8, cache_ttl_seconds=60), testing=True)
    client = app.test_client()
    payload = solid_image(50, 50, (10, 200, 120))

    first = _upload(client, payload).get_json()
    second = _upload(client, payload).get_json()

    assert first["cached"] is False
    assert second["cached"] is True
    assert first["image"]["sha256"] == second["image"]["sha256"]
    assert app.extensions["nid_service"].cache.stats().hits == 1


def test_cache_is_keyed_by_content_not_filename() -> None:
    from app import create_app

    app = create_app(base_settings(cache_enabled=True), testing=True)
    client = app.test_client()
    payload = solid_image(30, 30)

    _upload(client, payload, filename="a.png")
    body = _upload(client, payload, filename="b.png").get_json()
    assert body["cached"] is True


def test_cache_can_be_flushed() -> None:
    from app import create_app

    settings = base_settings(cache_enabled=True, api_keys=("admin-key",))
    app = create_app(settings, testing=True)
    client = app.test_client()
    headers = {"X-API-Key": "admin-key"}

    _upload(client, solid_image(30, 30))
    response = client.delete("/v1/cache", headers=headers)
    assert response.status_code == 200
    assert response.get_json()["cleared"] is True


def test_cache_flush_requires_auth(client) -> None:
    response = client.delete("/v1/cache")
    assert response.status_code == 400


# --------------------------------------------------------------------------- #
# Authentication and rate limiting
# --------------------------------------------------------------------------- #
@pytest.fixture
def auth_app() -> Settings:
    return base_settings(api_keys=("clave-uno", "clave-dos"))


def test_auth_is_enforced_when_keys_are_configured(auth_app: Settings) -> None:
    from app import create_app

    client = create_app(auth_app, testing=True).test_client()

    assert client.post("/v1/analyze", json={}).status_code == 401
    assert _upload(client, solid_image(32, 32)).status_code == 401

    ok = client.post(
        "/v1/analyze",
        data={"file": (io.BytesIO(solid_image(32, 32)), "a.png", "image/png")},
        content_type="multipart/form-data",
        headers={"X-API-Key": "clave-uno"},
    )
    assert ok.status_code == 200

    bearer = client.post(
        "/v1/analyze",
        data={"file": (io.BytesIO(solid_image(32, 32)), "a.png", "image/png")},
        content_type="multipart/form-data",
        headers={"Authorization": "Bearer clave-dos"},
    )
    assert bearer.status_code == 200

    wrong = client.post(
        "/v1/analyze",
        data={"file": (io.BytesIO(solid_image(32, 32)), "a.png", "image/png")},
        content_type="multipart/form-data",
        headers={"X-API-Key": "clave-falsa"},
    )
    assert wrong.status_code == 401


def test_public_endpoints_do_not_require_auth(auth_app: Settings) -> None:
    from app import create_app

    client = create_app(auth_app, testing=True).test_client()
    for path in ("/health", "/ready", "/v1/info", "/v1/labels", "/", "/openapi.json"):
        assert client.get(path).status_code == 200, path


def test_rate_limit_returns_429_with_retry_after() -> None:
    from app import create_app

    settings = base_settings(rate_limit_enabled=True, rate_limit_requests=2, rate_limit_window_seconds=60)
    client = create_app(settings, testing=True).test_client()

    first = _upload(client, solid_image(20, 20))
    second = _upload(client, solid_image(20, 20))
    third = _upload(client, solid_image(20, 20))

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.headers["X-RateLimit-Limit"] == "2"
    assert third.status_code == 429
    assert third.get_json()["error"]["code"] == "rate_limited"
    assert int(third.headers["Retry-After"]) > 0


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
def test_batch_of_urls(url_client, local_base_url: str) -> None:
    response = url_client.post(
        "/v1/analyze/batch",
        json={"urls": [f"{local_base_url}/image", f"{local_base_url}/png"]},
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["count"] == 2
    assert body["summary"]["succeeded"] == 2
    assert body["summary"]["verdicts"]["allow"] == 2
    assert all(item["ok"] for item in body["results"])


def test_batch_isolates_failures(url_client, local_base_url: str) -> None:
    response = url_client.post(
        "/v1/analyze/batch",
        json={
            "urls": [
                f"{local_base_url}/image",
                f"{local_base_url}/missing",
                "ftp://127.0.0.1/x.jpg",  # esquema no permitido
            ]
        },
    )
    body = response.get_json()
    assert body["summary"]["succeeded"] == 1
    assert body["summary"]["failed"] == 2

    codes = [item["error"]["code"] for item in body["results"] if not item["ok"]]
    assert "upstream_error" in codes
    assert "invalid_url" in codes
    assert body["results"][0]["ok"] is True


def test_batch_blocks_private_urls_when_the_guard_is_on(client) -> None:
    """Same batch endpoint, but with the SSRF guard enabled (default config)."""
    response = client.post(
        "/v1/analyze/batch",
        json={"urls": ["http://169.254.169.254/latest/meta-data/", "http://10.0.0.1/a.jpg"]},
    )
    body = response.get_json()
    assert body["summary"]["succeeded"] == 0
    assert {item["error"]["code"] for item in body["results"]} == {"blocked_destination"}


def test_batch_of_base64_items(client) -> None:
    raw = base64.b64encode(solid_image(24, 24)).decode()
    response = client.post(
        "/v1/analyze/batch",
        json={"items": [{"image_base64": raw, "filename": "a.png"}, {"image_base64": raw}]},
    )
    body = response.get_json()
    assert body["summary"]["total"] == 2
    assert body["results"][0]["result"]["source"]["type"] == "base64"


def test_batch_multipart_files(client) -> None:
    response = client.post(
        "/v1/analyze/batch",
        data={
            "files": [
                (io.BytesIO(solid_image(20, 20)), "a.png", "image/png"),
                (io.BytesIO(solid_image(20, 20, (5, 5, 5))), "b.png", "image/png"),
            ]
        },
        content_type="multipart/form-data",
    )
    body = response.get_json()
    assert body["summary"]["total"] == 2
    assert body["summary"]["succeeded"] == 2


def test_batch_rejects_too_many_items() -> None:
    from app import create_app

    app = create_app(base_settings(batch_max_items=2), testing=True)
    response = app.test_client().post(
        "/v1/analyze/batch", json={"urls": ["https://a.test/1.jpg"] * 3}
    )
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "too_many_items"


def test_batch_without_items_is_400(client) -> None:
    response = client.post("/v1/analyze/batch", json={})
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "missing_image"


def test_batch_with_invalid_item_is_400(client) -> None:
    response = client.post("/v1/analyze/batch", json={"items": [{"nothing": 1}]})
    assert response.status_code == 400


def test_batch_single_object_is_tolerated(url_client, local_base_url: str) -> None:
    response = url_client.post("/v1/analyze/batch", json={"url": f"{local_base_url}/image"})
    assert response.status_code == 200
    assert response.get_json()["summary"]["total"] == 1


def test_batch_reject_on_block(blocked_app, mocker: pytest.MockerFixture) -> None:
    client = blocked_app.test_client()
    raw = base64.b64encode(solid_image(20, 20)).decode()
    response = client.post(
        "/v1/analyze/batch?reject_on_block=true", json={"items": [{"image_base64": raw}]}
    )
    assert response.status_code == 422
    assert response.get_json()["summary"]["any_blocked"] is True


# --------------------------------------------------------------------------- #
# Headers and tracing
# --------------------------------------------------------------------------- #
def test_request_id_is_echoed_when_valid(client) -> None:
    custom = client.post(
        "/v1/analyze",
        data={"file": (io.BytesIO(solid_image(16, 16)), "a.png", "image/png")},
        content_type="multipart/form-data",
        headers={"X-Request-ID": "req-12345678"},
    )
    assert custom.headers["X-Request-ID"] == "req-12345678"
    assert custom.get_json()["request_id"] == "req-12345678"


def test_invalid_request_id_is_replaced(client) -> None:
    response = client.post(
        "/v1/analyze",
        data={"file": (io.BytesIO(solid_image(16, 16)), "a.png", "image/png")},
        content_type="multipart/form-data",
        headers={"X-Request-ID": "ab"},
    )
    assert response.headers["X-Request-ID"] != "ab"


def test_security_headers_are_present(client) -> None:
    response = _upload(client, solid_image(16, 16))
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert "Content-Security-Policy" in response.headers
    assert response.headers["Cache-Control"] == "no-store"
    assert response.headers["Access-Control-Allow-Origin"] == "*"


def test_cors_origin_is_configurable() -> None:
    from app import create_app

    app = create_app(base_settings(cors_origin="https://miapp.example"), testing=True)
    response = app.test_client().get("/health")
    assert response.headers["Access-Control-Allow-Origin"] == "https://miapp.example"


# --------------------------------------------------------------------------- #
# Engine failures must not leak internals
# --------------------------------------------------------------------------- #
def test_inference_failure_is_reported_cleanly(client, mocker: pytest.MockerFixture) -> None:
    app = client.application
    engine = app.extensions["nid_engine"]
    mocker.patch.object(engine, "analyze", side_effect=RuntimeError("boom: /secret/path.onnx"))

    response = _upload(client, solid_image(32, 32))
    assert response.status_code == 500
    body = response.get_json()
    assert body["error"]["code"] == "internal_error"
    assert "/secret/path.onnx" not in json.dumps(body)


def test_disabled_engine_returns_503() -> None:
    from app import create_app

    app = create_app(base_settings(engine="disabled"), testing=True)
    client = app.test_client()
    assert client.get("/ready").status_code == 503

    response = _upload(client, solid_image(32, 32))
    assert response.status_code == 503
    assert response.get_json()["error"]["code"] == "model_unavailable"
