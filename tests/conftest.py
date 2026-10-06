"""Shared fixtures: synthetic images, an app with a mock engine, and a local
HTTP server to exercise URL mode without touching the internet.

No test in this suite requires network access or a real photograph.
"""

from __future__ import annotations

import io
import socket
import threading
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.core.engine import Detection
from app.core.imaging import SafeImage
from app.core.labels import label_info
from app.core.policy import PolicyConfig


# --------------------------------------------------------------------------- #
# Image builders
# --------------------------------------------------------------------------- #
def solid_image(
    width: int = 64,
    height: int = 48,
    color: tuple[int, int, int] = (90, 120, 160),
    fmt: str = "PNG",
) -> bytes:
    """Return the bytes of a plain-colour image."""
    image = Image.new("RGB", (width, height), color)
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    return buffer.getvalue()


def noise_image(width: int = 96, height: int = 96, seed: int = 7, fmt: str = "JPEG") -> bytes:
    """Return a deterministic pseudo-random image (never matches a real body)."""
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format=fmt, quality=90)
    return buffer.getvalue()


def skin_tone_image(width: int = 120, height: int = 120, fmt: str = "JPEG") -> bytes:
    """A flat skin-coloured square: the classic false positive of naive filters."""
    return solid_image(width, height, (224, 172, 140), fmt)


def animated_gif(frames: int = 4, size: int = 40) -> bytes:
    """Build a multi-frame GIF (used to test frame sampling)."""
    images = [
        Image.new("RGB", (size, size), (i * 40 % 255, 30, 200 - i * 30 % 200))
        for i in range(frames)
    ]
    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=50,
        loop=0,
    )
    return buffer.getvalue()


def rgba_png(width: int = 48, height: int = 48) -> bytes:
    """PNG with a real alpha channel (transparency must survive normalisation)."""
    array = np.zeros((height, width, 4), dtype=np.uint8)
    array[..., 0] = 200
    array[..., 3] = 128
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def detection(
    label: str,
    score: float,
    *,
    box: tuple[int, int, int, int] = (10, 10, 60, 60),
    area_ratio: float | None = None,
    frame_index: int = 0,
) -> Detection:
    """Build a Detection with the label's catalogue weight already applied."""
    info = label_info(label)
    if area_ratio is None:
        x1, y1, x2, y2 = box
        area_ratio = ((x2 - x1) * (y2 - y1)) / 640_000.0
    return Detection(
        label=label,
        score=score,
        box=box,
        frame_index=frame_index,
        area_ratio=area_ratio,
        severity=info.severity,
        weight=info.weight,
    )


def safe_image(width: int = 64, height: int = 64, frames: int = 1) -> SafeImage:
    """A minimal SafeImage carrying ``frames`` synthetic RGBA frames."""
    data = solid_image(width, height)
    arrays = [np.zeros((height, width, 4), dtype=np.uint8) for _ in range(frames)]
    for array in arrays:
        array[..., 3] = 255
    return SafeImage(
        data=data,
        sha256="0" * 64,
        size_bytes=len(data),
        detected_mime="image/png",
        declared_mime="image/png",
        pil_format="PNG",
        width=width,
        height=height,
        source_width=width,
        source_height=height,
        frames=arrays,
        frame_indices=tuple(range(frames)),
        frames_total=frames,
        is_animated=frames > 1,
    )


# --------------------------------------------------------------------------- #
# Settings / app fixtures
# --------------------------------------------------------------------------- #
def base_settings(**overrides: object) -> Settings:
    """Test settings: mock engine, no cache surprises, SSRF guard enabled."""
    defaults: dict[str, object] = {
        "engine": "mock",
        "cache_enabled": False,
        "load_model_at_startup": False,
        "warmup": False,
        "log_json": False,
        "debug": True,
        "max_frames": 3,
    }
    defaults.update(overrides)
    return Settings(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def settings() -> Settings:
    return base_settings()


@pytest.fixture
def app(settings: Settings):
    from app import create_app

    application = create_app(settings, testing=True)
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def policy(settings: Settings) -> PolicyConfig:
    return PolicyConfig.from_settings(settings)


@pytest.fixture
def strict_policy(settings: Settings) -> PolicyConfig:
    return PolicyConfig.from_settings(replace(settings, strictness="strict"))


@pytest.fixture
def lenient_policy(settings: Settings) -> PolicyConfig:
    return PolicyConfig.from_settings(replace(settings, strictness="lenient"))


# --------------------------------------------------------------------------- #
# Local HTTP server (for URL mode)
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    """Serves generated images, redirects and a few hostile responses."""

    def log_message(self, *args: object) -> None:  # silence stderr noise
        return None

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path
        if path.startswith("/image"):
            self._send(200, noise_image(64, 64, seed=3), "image/jpeg")
        elif path.startswith("/png"):
            self._send(200, solid_image(32, 32, fmt="PNG"), "image/png")
        elif path.startswith("/hop"):
            self.send_response(302)
            self.send_header("Location", "/image")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path.startswith("/relative"):
            self.send_response(302)
            self.send_header("Location", "image")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path.startswith("/loop"):
            self.send_response(302)
            self.send_header("Location", "/loop")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif path.startswith("/html"):
            self._send(200, b"<html>not an image</html>", "text/html")
        elif path.startswith("/huge"):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(50 * 1024 * 1024))
            self.end_headers()
        elif path.startswith("/slow"):
            import time

            time.sleep(3)
            self._send(200, noise_image(32, 32), "image/jpeg")
        elif path.startswith("/missing"):
            self._send(404, b"nope", "text/plain")
        elif path.startswith("/empty"):
            self._send(200, b"", "image/png")
        else:
            self._send(404, b"unknown route", "text/plain")


@pytest.fixture(scope="session")
def local_server() -> Iterator[tuple[str, int]]:
    """Start a throwaway HTTP server on an ephemeral loopback port."""
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[0], httpd.server_address[1]
    try:
        yield host, port
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


@pytest.fixture
def local_base_url(local_server: tuple[str, int]) -> str:
    host, port = local_server
    return f"http://{host}:{port}"


@pytest.fixture
def url_settings(local_server: tuple[str, int]) -> Settings:
    """Settings that allow loopback so the local test server is reachable."""
    _, port = local_server
    return base_settings(
        fetch_enabled=True,
        fetch_allow_private_networks=True,
        fetch_allowed_ports=(port, 80, 443),
        fetch_timeout_seconds=3.0,
    )


@pytest.fixture
def url_app(url_settings: Settings):
    from app import create_app

    application = create_app(url_settings, testing=True)
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def url_client(url_app):
    return url_app.test_client()


def free_port() -> int:
    """Return an unused TCP port (helper for ad-hoc servers in tests)."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
