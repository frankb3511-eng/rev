"""Offline demo source.

When no search engine is reachable, the pipeline still needs candidates to
verify.  This module *synthesises* a plausible set of web results from the
uploaded image -- resized copies, re-encodes, a watermark, crops, a lookalike
shot and unrelated images -- and hands them to the verification pipeline as
``data:`` URLs.

Only the "search" step is synthetic.  Downloading, hashing, filtering,
alignment, feature matching, classification, deduplication and merging are all
the real pipeline running on real pixels, so a demo run is a faithful
demonstration of the matcher rather than a mock of it.
"""

from __future__ import annotations

import base64
from typing import List

import numpy as np
from PIL import Image

from .imaging import LoadedImage, decode, encode_jpeg, encode_png, encode_webp
from .pipeline import RawResult

ENGINE_NAME = "demo"
ENGINE_DISPLAY = "Demo results (synthesised)"


def _data_url(payload: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def _variant(rgb: np.ndarray, kind: str) -> bytes:
    """Apply one transformation; returns encoded bytes."""
    img = Image.fromarray(rgb)
    h, w = rgb.shape[:2]

    if kind == "resized_small":
        return encode_jpeg(np.asarray(img.resize((w // 2, h // 2), Image.LANCZOS)), 88)
    if kind == "resized_large":
        return encode_png(np.asarray(img.resize((int(w * 1.6), int(h * 1.6)), Image.LANCZOS)))
    if kind == "recompressed":
        return encode_jpeg(rgb, 70)
    if kind == "heavily_compressed":
        return encode_jpeg(rgb, 22)
    if kind == "webp":
        return encode_webp(rgb, 80)
    if kind == "brighter":
        return encode_jpeg(np.clip(rgb.astype(np.int16) + 24, 0, 255).astype(np.uint8), 90)
    if kind == "warmer":
        out = rgb.astype(np.int16)
        out[:, :, 0] += 26
        out[:, :, 2] -= 20
        return encode_jpeg(np.clip(out, 0, 255).astype(np.uint8), 90)
    if kind == "watermarked":
        from PIL import ImageDraw
        im = img.convert("RGBA")
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.text((w - 140, h - 44), "STOCKPHOTO", fill=(255, 255, 255, 230))
        merged = Image.alpha_composite(im, overlay).convert("RGB")
        return encode_jpeg(np.asarray(merged), 90)
    if kind == "bordered":
        b = int(min(h, w) * 0.05)
        out = np.full((h + 2 * b, w + 2 * b, 3), 250, dtype=np.uint8)
        out[b:b + h, b:b + w] = rgb
        return encode_jpeg(out, 90)
    if kind == "crop_centre":
        return encode_jpeg(rgb[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)], 90)
    if kind == "crop_detail":
        return encode_jpeg(rgb[int(h * 0.35):int(h * 0.72), int(w * 0.45):int(w * 0.9)], 90)
    if kind == "rotated":
        return encode_png(np.asarray(img.rotate(90, expand=True)))
    raise ValueError(f"unknown demo variant {kind!r}")


def _lookalike(rgb: np.ndarray) -> bytes:
    """A different photograph of the same subject.

    Small perspective shift, different crop and a fresh noise realisation: close
    enough that a naive similarity score would call it a match, different
    enough that the block-level test rejects it.  This is the case the whole
    design is built around.
    """
    import cv2

    h, w = rgb.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[w * 0.06, h * 0.03], [w, 0], [w * 0.96, h], [0, h * 0.97]])
    warped = cv2.warpPerspective(
        rgb, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REFLECT
    )
    zoom = warped[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)]
    zoom = np.asarray(Image.fromarray(zoom).resize((w, h), Image.LANCZOS))
    rng = np.random.default_rng(7)
    zoom = np.clip(zoom.astype(np.float32) + rng.normal(0, 7, zoom.shape), 0, 255)
    return encode_jpeg(zoom.astype(np.uint8), 88)


def _unrelated(rng: np.random.Generator, size) -> bytes:
    w, h = size
    img = Image.new("RGB", (w, h), tuple(int(v) for v in rng.integers(15, 40, 3)))
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    for _ in range(12):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(20, 80))
        colour = tuple(int(v) for v in rng.integers(80, 255, 3))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=6)
    return encode_jpeg(np.asarray(img), 85)


#: (variant, the label a search engine would have shown, fake host)
VARIANTS = [
    ("recompressed", "Visually similar", "cdn.example-photos.com"),
    ("resized_small", "Exact match", "images.example-photos.com"),
    ("watermarked", "Visually similar", "stock.example.org"),
    ("crop_centre", "Partial match", "forum.example.net"),
    ("webp", "Exact match", "assets.example-cdn.io"),
    ("brighter", "Visually similar", "blog.example.com"),
    ("resized_large", "Exact match", "printshop.example.co"),
    ("bordered", "Visually similar", "gallery.example.dev"),
    ("crop_detail", "Partial match", "forum.example.net"),
    ("heavily_compressed", "Visually similar", "cache.example-old.com"),
    ("warmer", "Visually similar", "shop.example.store"),
    ("rotated", "Exact match", "mirror.example.ru"),
]


def synthesize_results(image: LoadedImage, include_unrelated: int = 3) -> List[RawResult]:
    """Build a synthetic result set for one uploaded image."""
    rng = np.random.default_rng(1234)
    rgb = image.rgb
    results: List[RawResult] = []

    for i, (kind, engine_label, host) in enumerate(VARIANTS):
        payload = _variant(rgb, kind)
        results.append(RawResult(
            url=_data_url(payload, "image/jpeg" if payload[:3] != b"\x89PN" else "image/png"),
            engine=ENGINE_NAME,
            engine_label=engine_label,
            page_url=f"https://{host}/photo/{i}",
            title=f"{kind.replace('_', ' ')} on {host}",
        ))

    # The same content returned by a second engine: exercises URL dedup + merge.
    results.append(RawResult(
        url=results[1].url,
        engine="demo_second",
        engine_label="Exact match",
        page_url="https://mirror2.example.com/same-file",
        title="same bytes, different engine",
    ))
    # And a byte-identical copy under a different URL: exercises content-hash merge.
    results.append(RawResult(
        url=_data_url(_variant(rgb, "recompressed"), "image/jpeg"),
        engine="demo_third",
        engine_label="Visually similar",
        page_url="https://archive.example.info/copy",
        title="independent re-encode",
    ))

    results.append(RawResult(
        url=_data_url(_lookalike(rgb), "image/jpeg"),
        engine=ENGINE_NAME,
        engine_label="Visually similar",
        page_url="https://lookalike.example.com/another-shot",
        title="a different photograph of the same subject",
    ))

    for i in range(include_unrelated):
        results.append(RawResult(
            url=_data_url(_unrelated(rng, (image.width, image.height)), "image/jpeg"),
            engine=ENGINE_NAME,
            engine_label="Visually similar",
            page_url=f"https://noise.example{i}.com/img",
            title="unrelated image",
        ))
    return results


def display_name() -> str:
    return ENGINE_DISPLAY
