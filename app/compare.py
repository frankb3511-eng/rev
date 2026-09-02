"""Aligned image pairs for the side-by-side comparison viewer.

The viewer needs more than the two raw files: comparing a crop or a bordered
copy pixel-by-pixel is meaningless until they are registered, so this module
re-runs the *same* alignment the matcher used and returns:

``original``  the original at the comparison size
``candidate`` the candidate, aligned into the original's frame
``difference`` |original - candidate| amplified, so residual edits are visible
``heatmap``   the same difference, colourised

Everything is produced from the untouched original upload; nothing is modified
in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config
from .features import locate, match_features
from .imaging import LoadedImage, encode_jpeg, encode_png, fit_size, resize_rgb, to_gray
from .matcher import align_pair, resolve_orientation
import app.hashing as hashing


@dataclass
class ComparisonImages:
    original: np.ndarray
    candidate: np.ndarray
    difference: np.ndarray
    heatmap: np.ndarray
    align_method: str
    align_valid_fraction: float
    coverage_orig: float
    coverage_cand: float
    size: Tuple[int, int]

    def original_bytes(self) -> bytes:
        return encode_png(self.original)

    def candidate_bytes(self) -> bytes:
        return encode_png(self.candidate)

    def difference_bytes(self) -> bytes:
        return encode_png(self.difference)

    def heatmap_bytes(self) -> bytes:
        return encode_jpeg(self.heatmap, quality=92)


def _amplify(delta: np.ndarray, gain: float = 4.0) -> np.ndarray:
    """Stretch a small residual so it is actually visible."""
    out = np.clip(delta.astype(np.float32) * gain, 0, 255)
    return out.astype(np.uint8)


def build_comparison(
    original: LoadedImage,
    candidate: LoadedImage,
    size: int = 720,
) -> ComparisonImages:
    """Register the candidate against the original and render the views."""
    work = config.CANDIDATE_FILTER.work_size

    orig_hashes = hashing.compute(original)
    cand_hashes = hashing.compute(candidate)

    _orientation, _hashes, cand_rgb, _sim = resolve_orientation(
        orig_hashes, original.rgb, candidate.rgb, work
    )

    ow, oh = fit_size(original.width, original.height, work)
    cw, ch = fit_size(cand_rgb.shape[1], cand_rgb.shape[0], work)
    orig_work = resize_rgb(original.rgb, (ow, oh))
    cand_work = resize_rgb(cand_rgb, (cw, ch))

    feat = match_features(to_gray(orig_work), to_gray(cand_work))
    tpl = locate(orig_work, cand_work, work=256)

    a, b, method, valid = align_pair(orig_work, cand_work, feat, tpl, size)

    delta = np.abs(a.astype(np.int16) - b.astype(np.int16))
    difference = _amplify(delta)

    gray_delta = to_gray(delta).astype(np.uint8)
    heatmap = cv2.applyColorMap(
        np.clip(gray_delta * 6, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    cov_o = cov_c = 0.0
    if feat.trusted and feat.coverage is not None:
        cov_o, cov_c = feat.coverage.original, feat.coverage.candidate
    elif tpl.confident and tpl.coverage is not None:
        cov_o, cov_c = tpl.coverage.original, tpl.coverage.candidate

    return ComparisonImages(
        original=a,
        candidate=b,
        difference=difference,
        heatmap=heatmap,
        align_method=method,
        align_valid_fraction=valid,
        coverage_orig=cov_o,
        coverage_cand=cov_c,
        size=(a.shape[1], a.shape[0]),
    )
