"""Visual descriptor embedding (stage 5's optional embedding signal).

The default backend is a **classical fixed-length descriptor** built locally,
with no model download and no network access:

* per-cell colour moments (mean / std / skewness) on a 4x4 grid, 3 channels
* per-cell gradient-orientation histograms (HOG-style), 4x4 grid x 9 bins
* a global 64-bin RGB histogram
* edge-density per cell

The vector is L2-normalised and compared by cosine similarity.  It is far more
robust to colour and brightness edits than raw pixel error, while -- unlike a
semantic CNN embedding -- it stays sensitive to *layout*, which is exactly the
property needed to keep "different photo of the same subject" out of the 1:1
buckets.

A ``torch``/``open_clip`` backend is available and used automatically only when
``allow_clip`` is enabled **and** the packages import.  The pipeline never
depends on it; the backend actually used is reported in every result.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import config
from .imaging import resize_rgb, to_gray

log = logging.getLogger(__name__)

_CLIP_CACHE: dict = {}


@dataclass
class Embedding:
    vector: np.ndarray
    backend: str
    dim: int

    def to_dict(self) -> dict:
        return {"backend": self.backend, "dim": self.dim}


# --------------------------------------------------------------------------
# Classical descriptor
# --------------------------------------------------------------------------

def classical_embedding(rgb: np.ndarray, size: int = 128) -> Embedding:
    """Fixed-length classical visual descriptor."""
    small = resize_rgb(rgb, (size, size)).astype(np.float32)
    gray = to_gray(small)

    grid = config.EMBEDDING.grid
    cell = size // grid
    parts = []

    # --- colour moments per cell -------------------------------------------
    for gy in range(grid):
        for gx in range(grid):
            patch = small[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell]
            for c in range(3):
                chan = patch[:, :, c].ravel()
                mean = chan.mean()
                std = chan.std()
                skew = float(np.cbrt(np.mean((chan - mean) ** 3)) / (std + 1e-6))
                parts.append([mean / 255.0, std / 255.0, skew * 0.25])

    # --- gradient orientation histograms per cell ---------------------------
    gx_ = np.gradient(gray, axis=1)
    gy_ = np.gradient(gray, axis=0)
    mag = np.sqrt(gx_ ** 2 + gy_ ** 2)
    ang = (np.arctan2(gy_, gx_) + np.pi) / (2 * np.pi)      # [0, 1)
    hog_bins = config.EMBEDDING.hog_bins
    for gy in range(grid):
        for gx in range(grid):
            m = mag[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell].ravel()
            a = ang[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell].ravel()
            hist = np.zeros(hog_bins, dtype=np.float32)
            if m.size:
                idx = np.clip((a * hog_bins).astype(np.int32), 0, hog_bins - 1)
                np.add.at(hist, idx, m)
            parts.append((hist / (hist.sum() + 1e-6)).tolist())

    # --- global colour histogram --------------------------------------------
    from .hashing import color_histogram
    parts.append(color_histogram(rgb, config.EMBEDDING.colour_bins).tolist())

    # --- edge density per cell ----------------------------------------------
    edges = (mag > 24.0).astype(np.float32)
    for gy in range(grid):
        for gx in range(grid):
            patch = edges[gy * cell:(gy + 1) * cell, gx * cell:(gx + 1) * cell]
            parts.append([float(patch.mean())])

    vec = np.asarray([v for chunk in parts for v in chunk], dtype=np.float32)
    vec = _l2(vec)
    return Embedding(vector=vec, backend="classical-descriptor", dim=int(vec.size))


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


# --------------------------------------------------------------------------
# Optional CLIP backend
# --------------------------------------------------------------------------

def _load_clip():
    if not config.EMBEDDING.allow_clip:
        return None
    if "model" in _CLIP_CACHE:
        return _CLIP_CACHE["model"] or None
    try:
        import clip                                    # type: ignore
        import torch                                   # type: ignore
        from PIL import Image                           # noqa: F401
        model, preprocess = clip.load(config.EMBEDDING.clip_model, device="cpu")
        _CLIP_CACHE["model"] = (model, preprocess, clip, torch)
        return _CLIP_CACHE["model"]
    except Exception as exc:                            # pragma: no cover
        log.info("CLIP backend unavailable (%s); using classical descriptor", exc)
        _CLIP_CACHE["model"] = None
        return None


def clip_embedding(rgb: np.ndarray) -> Optional[Embedding]:
    """Semantic embedding via openai/clip.  ``None`` when unavailable."""
    loaded = _load_clip()
    if not loaded:
        return None
    from PIL import Image
    model, preprocess, clip, torch = loaded              # pragma: no cover
    with torch.no_grad():                                # pragma: no cover
        feat = model.encode_image(preprocess(Image.fromarray(rgb)).unsqueeze(0))
    vec = feat.squeeze(0).cpu().numpy().astype(np.float32)   # pragma: no cover
    return Embedding(vector=_l2(vec), backend=f"clip:{config.EMBEDDING.clip_model}", dim=int(vec.size))


def embed(rgb: np.ndarray) -> Embedding:
    """Public entry point: CLIP when enabled and importable, else classical."""
    emb = clip_embedding(rgb)
    return emb if emb is not None else classical_embedding(rgb)


def compare(a: Embedding, b: Embedding) -> Tuple[float, str]:
    """Cosine similarity, plus which backend produced it."""
    if a.backend != b.backend:
        # Never compare vectors from different backends -- fall back to the
        # classical descriptor so the signal stays meaningful.
        return 0.0, f"mismatch({a.backend} vs {b.backend})"
    return max(0.0, cosine(a.vector, b.vector)), a.backend
