"""Deterministic synthetic photo fixtures for the 1:1 matcher test suite.

The scenes are *structured* -- gradients, polygons, edges, fine texture -- not
random noise.  That matters: white noise loses almost all of its energy under
resampling, so a noise image would fail an SSIM resize test for the wrong
reason.  Structured scenes behave like real photographs under resize,
recompression and mild edits, which is what the thresholds are calibrated on.

Everything is seeded, so a failing test reproduces exactly.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

SCENE_W, SCENE_H = 640, 480


# --------------------------------------------------------------------------
# Scene synthesis
# --------------------------------------------------------------------------

def _scene(seed: int, variant_detail: int = 0) -> np.ndarray:
    """Draw a deterministic landscape-ish scene.

    ``seed`` fixes the whole composition.  ``variant_detail`` perturbs only the
    *fine* detail (foliage, texture, pebbles) while leaving the composition
    recognisably the same -- that is how "a different photograph of the same
    subject" is produced.
    """
    rng = np.random.default_rng(seed)
    detail_rng = np.random.default_rng(seed * 7919 + variant_detail)

    img = Image.new("RGB", (SCENE_W, SCENE_H), (0, 0, 0))
    px = np.asarray(img).astype(np.float32).copy()

    # --- sky gradient --------------------------------------------------------
    top = rng.integers(60, 130, 3)
    bottom = rng.integers(170, 235, 3)
    for y in range(SCENE_H):
        t = y / (SCENE_H - 1)
        px[y, :, :] = top * (1 - t) + bottom * t

    img = Image.fromarray(np.clip(px, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)

    # --- sun ------------------------------------------------------------------
    sun_x = int(rng.integers(80, SCENE_W - 80))
    sun_y = int(rng.integers(50, 150))
    sun_c = tuple(int(v) for v in rng.integers(230, 256, 3))
    draw.ellipse([sun_x - 34, sun_y - 34, sun_x + 34, sun_y + 34], fill=sun_c)

    # --- mountains --------------------------------------------------------------
    horizon = int(SCENE_H * 0.55)
    for layer, shade in enumerate((90, 120, 150)):
        base = horizon - layer * 26
        pts = [(0, base + 40)]
        x = 0
        while x < SCENE_W:
            step = int(rng.integers(40, 110))
            pts.append((x, base - int(rng.integers(10, 90))))
            x += step
        pts += [(SCENE_W, base + 40), (SCENE_W, SCENE_H), (0, SCENE_H)]
        c = tuple(int(np.clip(shade + rng.integers(-25, 25), 0, 255)) for _ in range(3))
        draw.polygon(pts, fill=c)

    # --- ground ------------------------------------------------------------------
    ground = tuple(int(v) for v in rng.integers(60, 110, 3))
    draw.rectangle([0, horizon + 20, SCENE_W, SCENE_H], fill=ground)

    # --- path ---------------------------------------------------------------------
    path_c = tuple(int(v) for v in rng.integers(140, 190, 3))
    draw.polygon(
        [(SCENE_W // 2 - 30, horizon + 20), (SCENE_W // 2 + 30, horizon + 20),
         (SCENE_W - 60, SCENE_H), (60, SCENE_H)],
        fill=path_c,
    )

    # --- house -----------------------------------------------------------------------
    hx, hy = int(rng.integers(60, 200)), int(horizon + 30)
    hw, hh = int(rng.integers(90, 140)), int(rng.integers(70, 100))
    wall = tuple(int(v) for v in rng.integers(170, 220, 3))
    draw.rectangle([hx, hy, hx + hw, hy + hh], fill=wall)
    draw.polygon([(hx - 12, hy), (hx + hw + 12, hy), (hx + hw // 2, hy - 46)],
                 fill=tuple(int(v) for v in rng.integers(90, 140, 3)))
    for wx in range(hx + 12, hx + hw - 20, 30):
        draw.rectangle([wx, hy + 16, wx + 18, hy + 36],
                       fill=tuple(int(v) for v in rng.integers(30, 70, 3)))
    draw.rectangle([hx + hw // 2 - 12, hy + hh - 40, hx + hw // 2 + 12, hy + hh],
                   fill=tuple(int(v) for v in rng.integers(70, 110, 3)))

    # --- trees (fine detail -- these move with variant_detail) -----------------------
    for _ in range(6):
        tx = int(detail_rng.integers(20, SCENE_W - 20))
        ty = int(detail_rng.integers(horizon + 10, SCENE_H - 60))
        th = int(detail_rng.integers(45, 85))
        trunk = tuple(int(v) for v in detail_rng.integers(70, 100, 3))
        draw.rectangle([tx - 4, ty, tx + 4, ty + th], fill=trunk)
        leaf = tuple(int(v) for v in detail_rng.integers(40, 90, 3))
        for r in range(3):
            rr = 26 - r * 6
            draw.ellipse([tx - rr, ty - rr * 2 + r * 8, tx + rr, ty + rr * 0.4 + r * 8], fill=leaf)

    # --- fine texture ---------------------------------------------------------------
    arr = np.asarray(img).astype(np.float32)
    noise = detail_rng.normal(0, 3.5, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return arr


def base_scene(seed: int = 42) -> np.ndarray:
    return _scene(seed, 0)


def alternate_photo(seed: int = 42) -> np.ndarray:
    """Same composition, different shot: new fine detail + slight viewpoint.

    This is the case the matcher must NOT call a 1:1 match.
    """
    arr = _scene(seed, variant_detail=1)
    # A small perspective shift, as if taken half a step to the side.
    import cv2
    h, w = arr.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.05, h * 0.02], [w, 0], [w * 0.97, h], [0, h * 0.98]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(arr, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    zoom = warped[int(h * 0.04):int(h * 0.96), int(w * 0.04):int(w * 0.96)]
    return np.asarray(Image.fromarray(zoom).resize((w, h), Image.LANCZOS))


def unrelated_scene(seed: int = 999) -> np.ndarray:
    """A completely different image: different subject, not another landscape.

    An earlier version of this fixture was a second landscape with a different
    seed.  That was the wrong control -- two random landscapes share a sky
    gradient on top and ground below, so they score ~0.78 SSIM and the case
    proved nothing.  This one is a genuinely unrelated composition: flat
    background, unrelated objects, no shared layout with the base scene.
    """
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (SCENE_W, SCENE_H), tuple(int(v) for v in rng.integers(20, 45, 3)))
    draw = ImageDraw.Draw(img)

    # unrelated objects: rings, bars, a grid, scattered discs
    for _ in range(14):
        cx = int(rng.integers(40, SCENE_W - 40))
        cy = int(rng.integers(40, SCENE_H - 40))
        r = int(rng.integers(18, 70))
        colour = tuple(int(v) for v in rng.integers(90, 255, 3))
        kind = int(rng.integers(0, 3))
        if kind == 0:
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=5)
        elif kind == 1:
            draw.rectangle([cx - r, cy - r // 3, cx + r, cy + r // 3], fill=colour)
        else:
            draw.polygon([(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)], fill=colour)

    step = int(rng.integers(45, 80))
    line_colour = tuple(int(v) for v in rng.integers(60, 120, 3))
    for x in range(0, SCENE_W, step):
        draw.line([(x, 0), (x, SCENE_H)], fill=line_colour, width=2)
    for y in range(0, SCENE_H, step):
        draw.line([(0, y), (SCENE_W, y)], fill=line_colour, width=2)

    arr = np.asarray(img).astype(np.float32)
    arr += rng.normal(0, 3.0, arr.shape).astype(np.float32)
    return np.clip(arr, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Transform helpers
# --------------------------------------------------------------------------

def _pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def resized(arr: np.ndarray, factor: float) -> np.ndarray:
    h, w = arr.shape[:2]
    return np.asarray(_pil(arr).resize((max(8, int(w * factor)), max(8, int(h * factor))), Image.LANCZOS))


def crop_region(arr: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    h, w = arr.shape[:2]
    return arr[int(h * y0):int(h * y1), int(w * x0):int(w * x1)].copy()


def brightness(arr: np.ndarray, delta: int) -> np.ndarray:
    return np.clip(arr.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def contrast(arr: np.ndarray, gain: float) -> np.ndarray:
    mean = arr.mean()
    return np.clip((arr.astype(np.float32) - mean) * gain + mean, 0, 255).astype(np.uint8)


def colour_shift(arr: np.ndarray, dr: int, dg: int, db: int) -> np.ndarray:
    out = arr.astype(np.int16)
    out[:, :, 0] += dr
    out[:, :, 1] += dg
    out[:, :, 2] += db
    return np.clip(out, 0, 255).astype(np.uint8)


def watermark(arr: np.ndarray, text: str = "SAMPLE") -> np.ndarray:
    img = _pil(arr).convert("RGB")
    draw = ImageDraw.Draw(img)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.text((img.width - 150, img.height - 46), text, fill=(255, 255, 255, 235))
    od.rectangle([img.width - 158, img.height - 52, img.width - 12, img.height - 14],
                 outline=(255, 255, 255, 235), width=2)
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    return np.asarray(img)


def border(arr: np.ndarray, frac: float = 0.06) -> np.ndarray:
    h, w = arr.shape[:2]
    b = int(min(h, w) * frac)
    out = np.full((h + 2 * b, w + 2 * b, 3), 248, dtype=np.uint8)
    out[b:b + h, b:b + w] = arr
    return out


def screenshot(arr: np.ndarray) -> np.ndarray:
    """A browser screenshot: chrome bar top and bottom, slightly rescaled."""
    h, w = arr.shape[:2]
    bar = int(h * 0.09)
    out = np.zeros((h + 2 * bar, w, 3), dtype=np.uint8)
    out[:bar] = np.array([52, 54, 60], dtype=np.uint8)
    out[bar:bar + h] = arr
    out[bar + h:] = np.array([38, 38, 42], dtype=np.uint8)
    return resized(out, 0.85)


def blur(arr: np.ndarray, radius: float = 1.6) -> np.ndarray:
    return np.asarray(_pil(arr).filter(ImageFilter.GaussianBlur(radius)))


def sharpen(arr: np.ndarray) -> np.ndarray:
    return np.asarray(_pil(arr).filter(ImageFilter.SHARPEN))


def rotate(arr: np.ndarray, deg: int) -> np.ndarray:
    return np.asarray(_pil(arr).rotate(-deg, expand=True))


def mirror(arr: np.ndarray) -> np.ndarray:
    return np.asarray(_pil(arr).transpose(Image.FLIP_LEFT_RIGHT))


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------

def to_bytes(arr: np.ndarray, fmt: str = "PNG", quality: int = 90, exif: Optional[bytes] = None) -> bytes:
    buf = io.BytesIO()
    kwargs: Dict[str, object] = {}
    if fmt.upper() == "JPEG":
        kwargs["quality"] = quality
    if fmt.upper() == "WEBP":
        kwargs["quality"] = quality
    if exif is not None:
        kwargs["exif"] = exif
    _pil(arr).save(buf, format=fmt.upper(), **kwargs)
    return buf.getvalue()


def _exif_with_orientation() -> bytes:
    """EXIF blob carrying Orientation=6 ("rotate 90 CW to display")."""
    from PIL.ExifTags import Base as ExifBase
    img = Image.new("RGB", (8, 8))
    exif = img.getexif()
    exif[ExifBase.Orientation] = 6
    exif[ExifBase.Make] = "FixtureCam"
    return exif.tobytes()


def exif_rotated_bytes(arr: np.ndarray) -> bytes:
    """The same picture stored the way a phone camera stores it.

    Pixels are rotated 90 degrees counter-clockwise and Orientation=6 is set,
    so that any EXIF-aware viewer displays it upright.  The earlier version of
    this fixture stamped the tag onto *unrotated* pixels, which genuinely does
    render rotated -- so the matcher was right to call it a rotation and the
    fixture was wrong.
    """
    img = Image.fromarray(arr).transpose(Image.ROTATE_90)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95, exif=_exif_with_orientation())
    return buf.getvalue()


# --------------------------------------------------------------------------
# The labelled case matrix
# --------------------------------------------------------------------------

@dataclass
class Case:
    """One labelled test case: bytes plus the verdict the spec expects."""

    name: str
    data: bytes
    expected: str          # Label value
    group: str             # expected UI group
    note: str = ""


def build_cases(scene_seed: int = 42) -> List[Case]:
    """Every edge case listed in the specification, with its expected verdict."""
    base = base_scene(scene_seed)

    png_exact = to_bytes(base, "PNG")
    jpeg_q90 = to_bytes(base, "JPEG", 90)

    cases = [
        # --- identical files ------------------------------------------------
        Case("identical_bytes", png_exact, "EXACT_1TO1", "exact",
             "the same bytes uploaded twice"),
        Case("renamed_identical", to_bytes(base, "PNG"), "EXACT_1TO1", "exact",
             "identical content, different filename"),
        Case("metadata_stripped", jpeg_q90, "EXACT_1TO1", "exact",
             "EXIF stripped by the server"),
        Case("metadata_added", exif_rotated_bytes(base), "EXACT_1TO1", "exact",
             "camera-style JPEG: rotated pixels + EXIF orientation tag"),

        # --- resize / recompress / format ------------------------------------
        Case("resized_down_half", to_bytes(resized(base, 0.5), "PNG"), "EXACT_1TO1", "exact",
             "half resolution"),
        Case("resized_up_double", to_bytes(resized(base, 2.0), "JPEG", 92), "EXACT_1TO1", "exact",
             "double resolution"),
        Case("jpeg_recompressed", to_bytes(base, "JPEG", 82), "EXACT_1TO1", "exact",
             "lossy re-encode at q=82"),
        Case("png_to_jpeg", to_bytes(base, "JPEG", 88), "EXACT_1TO1", "exact",
             "PNG -> JPEG conversion"),
        Case("jpeg_to_webp", to_bytes(base, "WEBP", 85), "EXACT_1TO1", "exact",
             "PNG -> WebP conversion"),
        Case("heavily_compressed", to_bytes(base, "JPEG", 18), "EXACT_1TO1", "exact",
             "very lossy q=18 re-encode"),

        # --- minor edits -----------------------------------------------------
        Case("brightness_up", to_bytes(brightness(base, 26), "JPEG", 90), "EDITED_1TO1", "edited_resized",
             "global +26 brightness"),
        Case("brightness_down", to_bytes(brightness(base, -26), "JPEG", 90), "EDITED_1TO1", "edited_resized",
             "global -26 brightness"),
        Case("contrast_up", to_bytes(contrast(base, 1.28), "JPEG", 90), "EDITED_1TO1", "edited_resized",
             "contrast gain 1.28"),
        Case("colour_shift", to_bytes(colour_shift(base, 30, -18, 12), "JPEG", 90), "EDITED_1TO1", "edited_resized",
             "per-channel colour shift"),
        Case("watermarked", to_bytes(watermark(base), "JPEG", 90), "EDITED_1TO1", "edited_resized",
             "text overlay in the corner"),
        Case("bordered", to_bytes(border(base, 0.06), "JPEG", 90), "EDITED_1TO1", "edited_resized",
             "uniform frame around the content"),
        Case("screenshot", to_bytes(screenshot(base), "PNG"), "EDITED_1TO1", "edited_resized",
             "browser screenshot with chrome bars"),
        Case("blurred", to_bytes(blur(base), "JPEG", 90), "EDITED_1TO1", "edited_resized",
             "Gaussian blur filter"),

        # --- geometry ---------------------------------------------------------
        Case("crop_centre", to_bytes(crop_region(base, 0.25, 0.25, 0.75, 0.75), "JPEG", 90),
             "CROP", "crops", "centre 50% crop"),
        Case("crop_corner", to_bytes(crop_region(base, 0.0, 0.0, 0.62, 0.66), "JPEG", 90),
             "CROP", "crops", "top-left crop"),
        Case("rotated_90", to_bytes(rotate(base, 90), "PNG"), "EDITED_1TO1", "edited_resized",
             "rotated 90 degrees"),
        Case("rotated_180", to_bytes(rotate(base, 180), "PNG"), "EDITED_1TO1", "edited_resized",
             "rotated 180 degrees"),
        Case("mirrored", to_bytes(mirror(base), "PNG"), "EDITED_1TO1", "edited_resized",
             "horizontally mirrored"),

        # --- must NOT be a 1:1 match -------------------------------------------
        Case("different_photo_same_subject", to_bytes(alternate_photo(scene_seed), "JPEG", 90),
             "VISUALLY_SIMILAR", "similar",
             "same composition, different shot -- the key false-positive case"),
        Case("unrelated_image", to_bytes(unrelated_scene(777), "JPEG", 90),
             "UNRELATED", "unrelated", "a completely different scene"),
    ]
    return cases


def case_map(seed: int = 42) -> Dict[str, Case]:
    return {c.name: c for c in build_cases(seed)}
