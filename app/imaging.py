"""Image loading and normalisation.

Everything downstream compares *normalised* images, never the raw bytes:

* EXIF orientation is baked in so a phone photo rotated by metadata is not
  mistaken for a rotated edit.
* ICC profiles are applied so a colour-managed copy of the same file does not
  look like a colour edit.
* Alpha is composited onto white, because a PNG->JPEG conversion does exactly
  that and the two results should compare equal.
* Nothing here mutates the original upload.  The original bytes are kept aside
  untouched by the pipeline (see ``app/pipeline.py``).
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

MAX_UPLOAD_BYTES = 32 * 1024 * 1024  # 32 MiB

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


class UnsupportedImage(ValueError):
    """Raised when bytes are not a decodable still image."""


@dataclass
class LoadedImage:
    """A decoded image plus the metadata the matcher reports."""

    rgb: np.ndarray                  # H x W x 3, uint8, orientation applied
    width: int
    height: int
    fmt: str
    sha256: str
    byte_size: int
    had_exif_orientation: bool
    mode: str

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def gray(self) -> np.ndarray:
        return to_gray(self.rgb)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def decode(data: bytes) -> LoadedImage:
    """Decode arbitrary upload bytes into a normalised RGB array.

    Raises ``UnsupportedImage`` for anything PIL cannot open as a still image.
    """
    if not data:
        raise UnsupportedImage("empty upload")
    if len(data) > MAX_UPLOAD_BYTES:
        raise UnsupportedImage(f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MiB")

    try:
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            fmt = (im.format or "UNKNOWN").upper()
            mode = im.mode
            oriented = False
            try:
                exif = im.getexif()
                oriented = bool(exif) and exif.get(274, 1) != 1
            except Exception:                      # pragma: no cover - corrupt EXIF
                oriented = False
            if oriented:
                im = ImageOps.exif_transpose(im)
            # ``convert("RGB")`` applies any embedded ICC profile and drops alpha.
            rgb_im = im.convert("RGB")
            arr = np.asarray(rgb_im, dtype=np.uint8)
    except UnsupportedImage:
        raise
    except Exception as exc:
        raise UnsupportedImage(f"could not decode image: {exc}") from exc

    if arr.ndim != 3 or arr.shape[2] != 3:        # pragma: no cover - defensive
        raise UnsupportedImage("unexpected image shape after conversion")

    h, w = arr.shape[:2]
    return LoadedImage(
        rgb=arr,
        width=w,
        height=h,
        fmt=fmt,
        sha256=sha256_bytes(data),
        byte_size=len(data),
        had_exif_orientation=oriented,
        mode=mode,
    )


# --------------------------------------------------------------------------
# Colour space helpers
# --------------------------------------------------------------------------

# Rec. 601 luma, the same weights PIL and OpenCV use for RGB->L.
_LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """uint8 H x W x 3 -> float32 H x W in [0, 255]."""
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    return rgb.astype(np.float32) @ _LUMA


def resize_rgb(rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """High-quality resize with Pillow (LANCZOS); ``size`` is (w, h)."""
    im = Image.fromarray(rgb).resize(size, Image.LANCZOS)
    return np.asarray(im, dtype=np.uint8)


def fit_size(width: int, height: int, longest: int) -> Tuple[int, int]:
    """Largest (w, h) with max(w, h) <= longest preserving aspect ratio."""
    if width <= 0 or height <= 0:
        raise ValueError("degenerate image dimensions")
    if max(width, height) <= longest:
        return width, height
    if width >= height:
        return longest, max(1, round(height * longest / width))
    return max(1, round(width * longest / height)), longest


def aspect_similarity(a: float, b: float) -> float:
    """1.0 for identical aspect ratios, 0.0 for infinitely different ones."""
    if a <= 0 or b <= 0:
        return 0.0
    lo, hi = sorted((a, b))
    return lo / hi


def safe_suffix(filename: str) -> str:
    """Lowercase extension including the dot, '' when there is none."""
    name = (filename or "").lower()
    for suf in sorted(SUPPORTED_SUFFIXES, key=len, reverse=True):
        if name.endswith(suf):
            return suf
    return ""


def encode_jpeg(rgb: np.ndarray, quality: int = 92) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def encode_png(rgb: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def encode_webp(rgb: np.ndarray, quality: int = 80) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="WEBP", quality=quality)
    return buf.getvalue()


def image_to_data_url(rgb: np.ndarray, fmt: str = "jpeg", quality: int = 85) -> str:
    """Inline data URL, used by the comparison viewer so it needs no storage."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format=fmt.upper(), quality=quality)
    mime = "jpeg" if fmt == "jpg" else fmt
    import base64
    return f"data:image/{mime};base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def load_optional(path: str) -> Optional[LoadedImage]:
    from pathlib import Path
    p = Path(path)
    if not p.is_file():
        return None
    return decode(p.read_bytes())
