"""Safe image decoding and normalisation.

Everything that can go wrong with an untrusted image is handled here, before a
single pixel reaches the model:

* **Lying content types** - the real format is sniffed from magic bytes, never
  trusted from the ``Content-Type`` header or the file extension.
* **Decompression bombs** - both a byte cap and a *pixel* cap are enforced.
* **EXIF orientation** - a rotated photo is analysed the way a human sees it.
* **Animated containers** - GIF/APNG/animated WebP can hide explicit frames
  behind a benign first frame, so up to ``max_frames`` frames are sampled.
* **Exotic colour modes** - palette, greyscale, CMYK and 16-bit images are
  normalised to the RGBA ``uint8`` arrays the model expects.
* **Oversized pictures** - they are downscaled before inference to bound memory
  (relative geometry, and therefore the policy, is unaffected).
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from ..config import Settings
from ..errors import (
    ImageTooLarge,
    PayloadTooLarge,
    UnprocessableImage,
    UnsupportedMediaType,
)

# Reject truncated files instead of silently analysing half an image.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Signatures -> MIME type.
_SIGNATURES: tuple[tuple[bytes, int, str], ...] = (
    (b"\xff\xd8\xff", 0, "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", 0, "image/png"),
    (b"GIF87a", 0, "image/gif"),
    (b"GIF89a", 0, "image/gif"),
    (b"BM", 0, "image/bmp"),
    (b"II*\x00", 0, "image/tiff"),
    (b"MM\x00*", 0, "image/tiff"),
    (b"\x00\x00\x01\x00", 0, "image/x-icon"),
)

#: Container formats we can identify but deliberately do not analyse: neither
#: OpenCV nor the bundled model decode them reliably, and pretending otherwise
#: would let an explicit photo through as "safe".
_UNSUPPORTED_BRANDS: dict[str, str] = {
    "heic": "HEIC/HEIF",
    "heix": "HEIC/HEIF",
    "hevc": "HEIC/HEIF",
    "heim": "HEIC/HEIF",
    "mif1": "HEIF",
    "msf1": "HEIF",
    "avif": "AVIF",
    "avis": "AVIF animado",
}


def sniff_mime(data: bytes) -> str | None:
    """Detect the container format from magic bytes.

    Returns a MIME type, ``"image/x-<brand>"`` for known-but-unsupported
    containers (HEIC/AVIF), or ``None`` when the payload is not an image at all.
    """
    if len(data) < 12:
        return None

    for signature, offset, mime in _SIGNATURES:
        if data[offset : offset + len(signature)] == signature:
            return mime

    # RIFF .... WEBP
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"

    # ISO base media file format: ....ftyp<brand>
    if data[4:8] == b"ftyp":
        brand = data[8:12].decode("latin-1", "replace").strip("\x00 ").lower()
        if brand in _UNSUPPORTED_BRANDS:
            return f"image/x-{brand}"
        return None

    return None


@dataclass
class SafeImage:
    """A validated, normalised image ready for inference."""

    data: bytes
    sha256: str
    size_bytes: int
    detected_mime: str
    declared_mime: str | None
    pil_format: str | None
    width: int
    height: int
    source_width: int
    source_height: int
    frames: list[np.ndarray] = field(default_factory=list)
    frame_indices: tuple[int, ...] = (0,)
    frames_total: int = 1
    is_animated: bool = False
    downscaled: bool = False
    orientation_applied: bool = False
    mime_mismatch: bool = False

    @property
    def is_supported(self) -> bool:
        return not self.detected_mime.startswith("image/x-")

    @property
    def primary_frame(self) -> np.ndarray:
        return self.frames[0]

    def describe(self) -> dict[str, object]:
        """Serialisable metadata returned to API clients (never the pixels)."""
        return {
            "format": (self.pil_format or self.detected_mime.split("/")[-1]).lower(),
            "mime_type": self.detected_mime,
            "declared_mime_type": self.declared_mime,
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "bytes": self.size_bytes,
            "sha256": self.sha256,
            "animated": self.is_animated,
            "frames_total": self.frames_total,
            "frames_analysed": len(self.frames),
            "downscaled": self.downscaled,
            "orientation_applied": self.orientation_applied,
            "mime_mismatch": self.mime_mismatch,
        }


def _exif_orientation(im: Image.Image) -> int:
    """Read the EXIF orientation tag (1 = no transform needed).

    Reading the tag ourselves instead of trusting the identity of the object
    returned by ``ImageOps.exif_transpose`` matters: Pillow returns a *copy* even
    when there is nothing to rotate, which would report a transform that never
    happened.
    """
    try:
        exif = im.getexif()
        orientation = int(exif.get(274, 1) or 1)  # 274 == ExifTags.Base.Orientation
    except Exception:  # pragma: no cover - malformed EXIF must not break analysis
        return 1
    return orientation if orientation in {1, 2, 3, 4, 5, 6, 7, 8} else 1


def _to_rgba(im: Image.Image) -> Image.Image:
    """Normalise any PIL mode to RGBA (4 channels, uint8)."""
    mode = im.mode
    if mode == "RGBA":
        return im
    if mode in ("RGB", "RGBX", "BGR", "YCbCr", "HSV"):
        return im.convert("RGBA")
    if mode in ("P", "PA"):
        # Palette images may carry transparency; convert() expands it.
        return im.convert("RGBA")
    if mode in ("L", "LA", "1", "I", "F"):
        if mode in ("I", "F"):
            # 16-bit / float greyscale: scale down to 8 bits preserving absolute
            # intensity (a plain convert("L") would clip everything to white).
            arr = np.asarray(im, dtype=np.float64)
            if arr.size == 0:
                return im.convert("L").convert("RGBA")
            peak = float(np.nanmax(arr)) if np.isfinite(arr).any() else 0.0
            divisor = 256.0 if peak > 255.0 else 1.0
            arr = np.clip(arr / divisor, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="L").convert("RGBA")
        return im.convert("RGBA")
    if mode == "CMYK":
        return im.convert("RGB").convert("RGBA")
    return im.convert("RGBA")


def _frame_indices(total: int, max_frames: int) -> tuple[int, ...]:
    """Evenly sample up to ``max_frames`` indices out of ``total`` frames."""
    if total <= 1 or max_frames <= 1:
        return (0,)
    if total <= max_frames:
        return tuple(range(total))
    step = (total - 1) / (max_frames - 1)
    return tuple(sorted({round(i * step) for i in range(max_frames)}))


def _target_size(width: int, height: int, max_pixels: int) -> tuple[int, int]:
    """Downscale dimensions preserving aspect ratio so that w*h <= max_pixels."""
    pixels = width * height
    if pixels <= max_pixels:
        return width, height
    scale = (max_pixels / float(pixels)) ** 0.5
    return max(1, int(width * scale)), max(1, int(height * scale))


def load_image(
    data: bytes,
    settings: Settings,
    *,
    declared_mime: str | None = None,
    filename: str | None = None,
) -> SafeImage:
    """Validate ``data`` and return a :class:`SafeImage`.

    Raises an :class:`~app.errors.ApiError` subclass for every rejection reason,
    with a stable ``code`` the client can branch on.
    """
    if not data:
        raise UnprocessableImage("El archivo está vacío.")

    if len(data) > settings.max_content_length:
        raise PayloadTooLarge(
            "La imagen supera el tamaño máximo permitido.",
            details={
                "max_bytes": settings.max_content_length,
                "received_bytes": len(data),
                "filename": filename,
            },
        )

    detected = sniff_mime(data)
    if detected is None:
        raise UnprocessableImage(
            "El contenido no parece ser una imagen soportada.",
            details={
                "declared_mime_type": declared_mime,
                "accepted": sorted(settings.allowed_mime_types),
                "hint": "Se validan los bytes reales del archivo, no la extensión ni el Content-Type.",
            },
        )

    if detected.startswith("image/x-"):
        brand = _UNSUPPORTED_BRANDS.get(detected.split("image/x-")[-1], "desconocido")
        raise UnsupportedMediaType(
            f"El formato {brand} no está soportado; convierte la imagen a JPEG, PNG o WebP.",
            details={"detected": detected, "accepted": sorted(settings.allowed_mime_types)},
        )

    if detected not in settings.allowed_mime_types:
        raise UnsupportedMediaType(
            f"El tipo de imagen '{detected}' no está permitido.",
            details={"detected": detected, "accepted": sorted(settings.allowed_mime_types)},
        )

    if declared_mime and declared_mime.split(";")[0].strip().lower() != detected:
        # Not fatal - servers and browsers disagree constantly - but worth noting.
        mismatch = True
    else:
        mismatch = False

    previous_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    try:
        try:
            with Image.open(io.BytesIO(data)) as opened:
                pil_format = (opened.format or "").upper() or None
                source_width, source_height = opened.size
                is_animated = bool(getattr(opened, "is_animated", False))
                frames_total = int(getattr(opened, "n_frames", 1) or 1)

                if source_width <= 0 or source_height <= 0:
                    raise UnprocessableImage("La imagen tiene dimensiones inválidas.")
                if source_width * source_height > settings.max_image_pixels:
                    raise ImageTooLarge(
                        "La imagen tiene más píxeles de los permitidos.",
                        details={
                            "max_pixels": settings.max_image_pixels,
                            "width": source_width,
                            "height": source_height,
                        },
                    )

                indices = _frame_indices(frames_total, settings.max_frames)
                frames: list[np.ndarray] = []
                orientation_applied = False
                downscaled = False

                for index in indices:
                    if is_animated and index < frames_total:
                        try:
                            opened.seek(index)
                        except (EOFError, OSError):
                            continue
                    frame = opened.copy()

                    if _exif_orientation(frame) != 1:
                        oriented = ImageOps.exif_transpose(frame)
                        if oriented is not frame:
                            orientation_applied = True
                            frame.close()
                            frame = oriented

                    target_w, target_h = _target_size(
                        frame.width, frame.height, settings.max_analysis_pixels
                    )
                    if (target_w, target_h) != (frame.width, frame.height):
                        frame.thumbnail((target_w, target_h), Image.LANCZOS)
                        downscaled = True

                    rgba = _to_rgba(frame)
                    frames.append(np.asarray(rgba, dtype=np.uint8))
                    if rgba is not frame:
                        rgba.close()
                    frame.close()
        except Image.DecompressionBombError as exc:
            raise ImageTooLarge(
                "La imagen tiene más píxeles de los permitidos (posible bomba de descompresión).",
                details={"max_pixels": settings.max_image_pixels},
            ) from exc
        except (UnidentifiedImageError, SyntaxError) as exc:
            raise UnprocessableImage(
                "El archivo está corrupto o no es una imagen legible.",
                details={"detected_mime": detected},
            ) from exc
        except (OSError, ValueError) as exc:
            if isinstance(exc, (UnsupportedMediaType, UnprocessableImage, ImageTooLarge)):
                raise
            raise UnprocessableImage(
                "No se pudo decodificar la imagen.", details={"reason": type(exc).__name__}
            ) from exc
    finally:
        Image.MAX_IMAGE_PIXELS = previous_limit

    if not frames:
        raise UnprocessableImage("No se pudo extraer ningún fotograma de la imagen.")

    height, width = frames[0].shape[:2]

    return SafeImage(
        data=data,
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        detected_mime=detected,
        declared_mime=(declared_mime.split(";")[0].strip().lower() if declared_mime else None),
        pil_format=pil_format,
        width=width,
        height=height,
        source_width=source_width,
        source_height=source_height,
        frames=frames,
        frame_indices=indices,
        frames_total=frames_total,
        is_animated=is_animated,
        downscaled=downscaled,
        orientation_applied=orientation_applied,
        mime_mismatch=mismatch,
    )


def render_censored(
    image: SafeImage,
    boxes: Sequence[Sequence[float]],
    *,
    strength: int = 51,
    output_format: str = "PNG",
) -> bytes:
    """Return a copy of the image with ``boxes`` pixelated and blurred.

    ``boxes`` are ``(x1, y1, x2, y2)`` in the coordinate space of
    ``image.frames[0]``.  Pixelation first, then a Gaussian pass: blurring alone
    can leave enough signal for a determined viewer to reconstruct the region.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - opencv is a hard dependency
        raise RuntimeError("OpenCV es necesario para censurar imágenes") from exc

    frame = cv2.cvtColor(image.primary_frame, cv2.COLOR_RGBA2RGB)
    height, width = frame.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = (int(v) for v in box)
        # Grow the region slightly: detection boxes are often tighter than the
        # actual sensitive area.
        pad_x = int((x2 - x1) * 0.12)
        pad_y = int((y2 - y1) * 0.12)
        x1 = max(0, min(width - 1, x1 - pad_x))
        x2 = max(1, min(width, x2 + pad_x))
        y1 = max(0, min(height - 1, y1 - pad_y))
        y2 = max(1, min(height, y2 + pad_y))
        if x2 <= x1 or y2 <= y1:
            continue

        region = frame[y1:y2, x1:x2]
        rh, rw = region.shape[:2]
        small = cv2.resize(region, (max(1, rw // 16), max(1, rh // 16)), interpolation=cv2.INTER_AREA)
        pixelated = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        kernel = max(3, strength if strength % 2 == 1 else strength + 1)
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(pixelated, (kernel, kernel), 0)

    rgb = Image.fromarray(frame, mode="RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format=output_format)
    return buffer.getvalue()
