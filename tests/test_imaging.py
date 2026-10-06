"""Image validation and normalisation."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.core.imaging import load_image, render_censored, sniff_mime
from app.errors import (
    ImageTooLarge,
    PayloadTooLarge,
    UnprocessableImage,
    UnsupportedMediaType,
)

from .conftest import animated_gif, base_settings, noise_image, rgba_png, solid_image


# --------------------------------------------------------------------------- #
# Magic-byte sniffing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fmt,mime",
    [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("GIF", "image/gif"), ("WEBP", "image/webp"), ("BMP", "image/bmp")],
)
def test_sniff_supported_formats(fmt: str, mime: str) -> None:
    assert sniff_mime(solid_image(fmt=fmt)) == mime


def test_sniff_rejects_non_images() -> None:
    assert sniff_mime(b"PK\x03\x04 not really an image") is None
    assert sniff_mime(b"<html><body>hola</body></html>") is None
    assert sniff_mime(b"") is None
    assert sniff_mime(b"short") is None


def test_sniff_detects_heic_family() -> None:
    heic = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 16
    assert sniff_mime(heic) == "image/x-heic"
    avif = b"\x00\x00\x00\x1cftypavif" + b"\x00" * 16
    assert sniff_mime(avif) == "image/x-avif"


# --------------------------------------------------------------------------- #
# Happy paths
# --------------------------------------------------------------------------- #
def test_load_png(settings: Settings) -> None:
    image = load_image(solid_image(80, 60, fmt="PNG"), settings)
    assert image.width == 80 and image.height == 60
    assert image.detected_mime == "image/png"
    assert image.frames and image.frames[0].shape == (60, 80, 4)
    assert image.frames[0].dtype == np.uint8
    assert len(image.sha256) == 64
    assert image.is_animated is False


def test_load_jpeg_noise(settings: Settings) -> None:
    image = load_image(noise_image(), settings)
    assert image.detected_mime == "image/jpeg"
    assert image.frames[0].shape[2] == 4


def test_declared_mime_mismatch_is_reported_not_fatal(settings: Settings) -> None:
    image = load_image(solid_image(fmt="PNG"), settings, declared_mime="image/gif")
    assert image.mime_mismatch is True
    assert image.detected_mime == "image/png"
    assert image.describe()["mime_mismatch"] is True


def test_rgba_transparency_is_preserved(settings: Settings) -> None:
    image = load_image(rgba_png(), settings)
    assert image.frames[0].shape[2] == 4
    assert image.frames[0][..., 3].max() == 128


def test_greyscale_and_palette_modes_are_normalised(settings: Settings) -> None:
    grey = io.BytesIO()
    Image.new("L", (20, 20), 128).save(grey, format="PNG")
    image = load_image(grey.getvalue(), settings)
    assert image.frames[0].shape == (20, 20, 4)

    palette = io.BytesIO()
    Image.new("RGB", (20, 20), (10, 200, 90)).convert("P").save(palette, format="PNG")
    image = load_image(palette.getvalue(), settings)
    assert image.frames[0].shape[2] == 4


def test_cmyk_jpeg_is_normalised(settings: Settings) -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (24, 24), (200, 30, 30)).convert("CMYK").save(buffer, format="JPEG")
    image = load_image(buffer.getvalue(), settings)
    assert image.frames[0].shape == (24, 24, 4)


def test_16bit_png_is_scaled_not_clipped(settings: Settings) -> None:
    array = (np.linspace(0, 65535, 32 * 32).reshape(32, 32)).astype(np.uint16)
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    image = load_image(buffer.getvalue(), settings)
    frame = image.frames[0]
    assert frame.shape == (32, 32, 4)
    # A plain convert("L") would saturate to white; scaling keeps the gradient.
    assert frame[..., 0].min() < 32
    assert frame[..., 0].max() > 200


def test_exif_orientation_is_applied() -> None:
    settings = base_settings()
    source = Image.new("RGB", (100, 40), (210, 180, 140))
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 CW
    buffer = io.BytesIO()
    source.save(buffer, format="JPEG", exif=exif.tobytes())

    image = load_image(buffer.getvalue(), settings)
    assert image.orientation_applied is True
    # Width and height are swapped after honouring EXIF.
    assert image.width == 40 and image.height == 100


def test_orientation_flag_is_false_when_there_is_no_exif(settings: Settings) -> None:
    """Pillow's exif_transpose copies the image even without a tag: don't lie."""
    image = load_image(solid_image(60, 40, fmt="PNG"), settings)
    assert image.orientation_applied is False
    assert (image.width, image.height) == (60, 40)

    plain_jpeg = load_image(noise_image(60, 40, seed=21), settings)
    assert plain_jpeg.orientation_applied is False


def test_large_images_are_downscaled_before_inference() -> None:
    settings = base_settings(max_analysis_pixels=10_000, max_image_pixels=10_000_000)
    image = load_image(solid_image(400, 300, fmt="PNG"), settings)
    assert image.downscaled is True
    assert image.width * image.height <= 10_000
    # Aspect ratio is preserved.
    assert abs((image.width / image.height) - (400 / 300)) < 0.1
    assert image.source_width == 400 and image.source_height == 300


def test_animated_gif_samples_several_frames() -> None:
    settings = base_settings(max_frames=3)
    image = load_image(animated_gif(frames=6), settings)
    assert image.is_animated is True
    assert image.frames_total == 6
    assert len(image.frames) == 3
    assert image.frame_indices[0] == 0 and image.frame_indices[-1] == 5


def test_animated_gif_respects_max_frames_one() -> None:
    settings = base_settings(max_frames=1)
    image = load_image(animated_gif(frames=5), settings)
    assert len(image.frames) == 1


# --------------------------------------------------------------------------- #
# Rejections
# --------------------------------------------------------------------------- #
def test_empty_payload_is_rejected(settings: Settings) -> None:
    with pytest.raises(UnprocessableImage):
        load_image(b"", settings)


def test_non_image_payload_is_rejected(settings: Settings) -> None:
    with pytest.raises(UnprocessableImage, match="no parece ser una imagen"):
        load_image(b"PK\x03\x04this-is-a-zip-file-not-an-image", settings)


def test_heic_is_rejected_with_actionable_message(settings: Settings) -> None:
    heic = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 64
    with pytest.raises(UnsupportedMediaType, match="HEIC"):
        load_image(heic, settings)


def test_disallowed_format_is_rejected() -> None:
    settings = base_settings(allowed_mime_types=("image/png",))
    with pytest.raises(UnsupportedMediaType):
        load_image(noise_image(fmt="JPEG"), settings)


def test_oversized_payload_is_rejected() -> None:
    settings = base_settings(max_content_length=1024)
    payload = noise_image(200, 200, seed=11)
    assert len(payload) > 1024, "el fixture debe superar el límite para que la prueba tenga sentido"
    with pytest.raises(PayloadTooLarge) as excinfo:
        load_image(payload, settings)
    assert excinfo.value.details["max_bytes"] == 1024


def test_pixel_bomb_is_rejected() -> None:
    settings = base_settings(max_image_pixels=1000, max_analysis_pixels=1000)
    with pytest.raises(ImageTooLarge) as excinfo:
        load_image(solid_image(200, 200, fmt="PNG"), settings)
    assert excinfo.value.details["max_pixels"] == 1000


def test_truncated_jpeg_is_rejected(settings: Settings) -> None:
    data = noise_image(160, 160)
    with pytest.raises(UnprocessableImage):
        load_image(data[: len(data) // 2], settings)


def test_corrupt_png_header_is_rejected(settings: Settings) -> None:
    broken = b"\x89PNG\r\n\x1a\n" + b"garbage-garbage-garbage"
    with pytest.raises(UnprocessableImage):
        load_image(broken, settings)


# --------------------------------------------------------------------------- #
# Censoring
# --------------------------------------------------------------------------- #
def test_render_censored_produces_a_valid_image(settings: Settings) -> None:
    # A noisy image: censoring a flat colour would be a no-op and prove nothing.
    image = load_image(noise_image(120, 120, seed=5, fmt="PNG"), settings)
    censored = render_censored(image, [(20, 20, 80, 80)], strength=21)
    assert censored[:8] == b"\x89PNG\r\n\x1a\n"

    reloaded = Image.open(io.BytesIO(censored))
    assert reloaded.size == (image.width, image.height)

    before = image.primary_frame[40:70, 40:70, :3]
    after = np.asarray(reloaded.convert("RGB"))[40:70, 40:70]
    # The censored region must actually change.
    assert not np.array_equal(before, after)
    # Regions outside the box must stay untouched.
    assert np.array_equal(image.primary_frame[0:10, 0:10, :3], np.asarray(reloaded.convert("RGB"))[0:10, 0:10])


def test_render_censored_ignores_degenerate_boxes(settings: Settings) -> None:
    image = load_image(solid_image(40, 40, fmt="PNG"), settings)
    censored = render_censored(image, [(10, 10, 10, 10), (-5, -5, -1, -1)])
    assert censored[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_censored_clips_out_of_bounds_boxes(settings: Settings) -> None:
    image = load_image(solid_image(30, 30, fmt="PNG"), settings)
    censored = render_censored(image, [(0, 0, 10_000, 10_000)])
    assert len(censored) > 0
