"""Edit attribution: *why* two otherwise-identical frames differ.

The classifier needs to tell "same image, touched up" apart from "different
photograph of the same subject".  Both produce non-zero pixel differences, so
this module decomposes the difference into interpretable causes:

``brightness``   global additive offset   (cand ~= orig + b)
``contrast``     global multiplicative gain (cand ~= a * orig, a != 1)
``colour``       per-channel mean shift with luminance structure intact
``blur``/``sharpen``  Laplacian-variance ratio moved substantially
``watermark``    a small, localised cluster of changed blocks
``border``       uniform frame added around the content
``recompress``   high-frequency residual only, structure and stats intact

The gain/offset fit is a plain least-squares solve on the paired pixel values,
which is what makes a brightness/contrast edit detectable *without* being
confused with a genuinely different image: a real second photograph does not
fit a global linear model well, so its residual stays high.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import cv2
import numpy as np

from . import config
from .imaging import to_gray


@dataclass
class EditReport:
    """Attributed causes of difference between an original and a candidate."""

    #: Least-squares fit ``cand ~= gain * orig + offset`` on 0-255 luma.
    gain: float = 1.0
    offset: float = 0.0
    #: Fraction of variance in the candidate explained by that linear fit.
    linear_r2: float = 1.0
    #: Residual mean |delta| (0-255) after the fit is removed.
    residual_mad: float = 0.0

    brightness: bool = False
    contrast: bool = False
    colour: bool = False
    blur: bool = False
    sharpen: bool = False
    watermark: bool = False
    border: bool = False
    recompress: bool = False

    #: Fraction of blocks that differ (post gain/offset normalisation).
    changed_fraction: float = 0.0
    #: Fraction of blocks that differ a *lot* -- the localised-edit cluster.
    localised_fraction: float = 0.0
    #: Fraction of blocks with a large *peak* difference: thin high-contrast
    #: strokes (text, logos, rules) that the block mean averages away.
    peak_fraction: float = 0.0
    #: Bounding box of the localised cluster in normalised [0,1] coords.
    localised_box: Optional[Tuple[float, float, float, float]] = None
    border_frac: float = 0.0
    channel_shift: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    sharpness_ratio: float = 1.0
    #: Set when the pair is not an edited version of anything (a crop, a
    #: lookalike, an unrelated image).  "Brightness +26" is only a meaningful
    #: statement about an image that *is* otherwise this image; on a crop or an
    #: unrelated photo the same numbers are just measurement noise, and
    #: surfacing them as edit claims was misleading.
    suppress_labels: bool = False

    @property
    def labels(self) -> List[str]:
        if self.suppress_labels:
            return []
        out = []
        if self.border:
            out.append("border")
        if self.watermark:
            out.append("watermark/text overlay")
        if self.brightness:
            out.append("brightness")
        if self.contrast:
            out.append("contrast")
        if self.colour:
            out.append("colour")
        if self.blur:
            out.append("blur")
        if self.sharpen:
            out.append("sharpen")
        if self.recompress and not out:
            out.append("recompressed")
        return out

    @property
    def any(self) -> bool:
        return bool(self.labels)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["labels"] = self.labels
        d["gain"] = round(self.gain, 4)
        d["offset"] = round(self.offset, 3)
        d["linear_r2"] = round(self.linear_r2, 4)
        d["residual_mad"] = round(self.residual_mad, 3)
        d["changed_fraction"] = round(self.changed_fraction, 4)
        d["localised_fraction"] = round(self.localised_fraction, 4)
        d["peak_fraction"] = round(self.peak_fraction, 4)
        d["channel_shift"] = [round(v, 3) for v in self.channel_shift]
        d["sharpness_ratio"] = round(self.sharpness_ratio, 4)
        d["border_frac"] = round(self.border_frac, 4)
        return d


def _fit_gain_offset(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares ``b ~= gain*a + offset``; returns (gain, offset, r2)."""
    x = a.ravel().astype(np.float64)
    y = b.ravel().astype(np.float64)
    if x.size < 16:
        return 1.0, 0.0, 0.0
    sx = x.std()
    if sx < 1e-6:
        return 1.0, float(y.mean() - x.mean()), 0.0
    gain = float(np.cov(x, y)[0, 1] / (sx ** 2))
    offset = float(y.mean() - gain * x.mean())
    pred = gain * x + offset
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 1.0
    return gain, offset, float(np.clip(r2, 0.0, 1.0))


def _laplacian_variance(gray: np.ndarray) -> float:
    u = np.clip(gray, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(u, cv2.CV_32F)
    return float(lap.var())


def _uniform_band(band: np.ndarray) -> bool:
    """A border band is flat: low internal variance."""
    return float(band.std()) <= config.EDITS.border_uniform_std


def detect_border(orig_rgb: np.ndarray, cand_rgb: np.ndarray) -> Tuple[bool, float]:
    """Border/letterbox detection on aspect-preserving copies.

    This must NOT run on the non-uniformly stretched comparison canvases: the
    stretch is exactly what erases the aspect-ratio evidence the detector reads.
    Measured before this fix, the bordered fixture was reported as having no
    border at all, because both canvases were squares and the "inner aspect
    moves towards the original aspect" test could never fire.
    """
    from .imaging import fit_size, resize_rgb

    oh, ow = orig_rgb.shape[:2]
    ch, cw = cand_rgb.shape[:2]
    w, h = fit_size(cw, ch, 512)
    cand_small = resize_rgb(cand_rgb, (w, h))
    return _detect_border(cand_small, ow / max(1, oh))


def _detect_border(
    cand_rgb: np.ndarray,
    orig_aspect: float,
) -> Tuple[bool, float]:
    """Detect uniform letterbox/pillarbox bands around the content.

    A frame is not always on all four sides -- a browser screenshot adds bars
    top and bottom only -- so horizontal and vertical thicknesses are searched
    *jointly*.  Testing them independently was wrong: the aspect-ratio check
    then removed only one pair of bands, so the inner rectangle never matched
    the original's aspect and a genuine border was reported as none.

    A candidate band must be flat (uniform colour, not picture content), and
    stripping the bands must bring the remaining rectangle's aspect ratio
    towards the original's.  Returns ``(detected, border_area_fraction)``.
    """
    E = config.EDITS
    h, w = cand_rgb.shape[:2]
    gray = to_gray(cand_rgb).astype(np.float32)
    small_side = float(min(h, w))
    if small_side <= 16:
        return False, 0.0

    max_t = int(max(2, round(small_side * 0.12)))

    # Precompute flatness per thickness: a band of thickness t is flat only when
    # t does not exceed the real border, so these sets are contiguous from 2 up.
    h_ok = [0]
    v_ok = [0]
    for t in range(2, max_t + 1):
        if _uniform_band(gray[:t, :]) and _uniform_band(gray[-t:, :]):
            h_ok.append(t)
        if _uniform_band(gray[:, :t]) and _uniform_band(gray[:, -t:]):
            v_ok.append(t)

    best_gap = _aspect_gap(w / max(1.0, h), orig_aspect)
    best = (0, 0, best_gap)
    for tb in h_ok:
        for lr in v_ok:
            iw, ih = w - 2 * lr, h - 2 * tb
            if iw <= 8 or ih <= 8:
                continue
            gap = _aspect_gap(iw / ih, orig_aspect)
            if gap < best[2] - 1e-9 or (abs(gap - best[2]) <= 1e-9 and tb + lr > best[0] + best[1]):
                best = (tb, lr, gap)

    tb, lr, gap = best
    if tb == 0 and lr == 0:
        return False, 0.0
    band_area = 2 * tb * w + 2 * lr * (w - 2 * tb)
    frac = float(band_area) / float(w * h)
    detected = gap <= 0.03 and frac >= E.border_min_frac
    return detected, frac


def _aspect_gap(a: float, b: float) -> float:
    """0.0 for equal aspect ratios, approaching 1.0 as they diverge."""
    if a <= 0 or b <= 0:
        return 1.0
    lo, hi = sorted((float(a), float(b)))
    return 1.0 - lo / hi


def analyze(
    orig_rgb: np.ndarray,
    cand_rgb: np.ndarray,
    border: Optional[Tuple[bool, float]] = None,
) -> EditReport:
    """Attribute the difference between two same-aspect comparison canvases.

    Both arrays must already be normalised to the same width/height by the
    caller (see ``app/matcher.py`` -- it resizes to ``compare_size``).

    ``border`` is supplied by the caller from :func:`detect_border`, which needs
    aspect-preserving copies rather than these stretched canvases.
    """
    E = config.EDITS
    rep = EditReport()

    orig_gray = to_gray(orig_rgb)
    cand_gray = to_gray(cand_rgb)

    # --- global linear fit --------------------------------------------------
    gain, offset, r2 = _fit_gain_offset(orig_gray, cand_gray)
    rep.gain, rep.offset, rep.linear_r2 = gain, offset, r2

    pred = np.clip(gain * orig_gray + offset, 0, 255)
    residual = np.abs(cand_gray - pred)
    rep.residual_mad = float(residual.mean())

    # --- per-channel colour shift -------------------------------------------
    shift = tuple(
        float(cand_rgb[:, :, c].mean() - orig_rgb[:, :, c].mean()) for c in range(3)
    )
    rep.channel_shift = shift  # type: ignore[assignment]
    max_shift = max(abs(v) for v in shift)
    if max_shift >= E.colour_channel_shift and r2 >= 0.90:
        rep.colour = True

    if offset >= E.brightness_offset and r2 >= 0.90:
        rep.brightness = True
    if not (E.contrast_gain_min <= gain <= E.contrast_gain_max) and r2 >= 0.90:
        rep.contrast = True

    # --- sharpness -----------------------------------------------------------
    lv_o, lv_c = _laplacian_variance(orig_gray), _laplacian_variance(cand_gray)
    ratio = lv_c / lv_o if lv_o > 1e-6 else 1.0
    rep.sharpness_ratio = ratio
    # Comparing the Laplacian variance of two *different* photographs is
    # meaningless -- unrelated images routinely differ by 2x and were being
    # labelled "sharpen".  Only claim a sharpness edit when the frames genuinely
    # correspond, which is the same r2 gate the gain/offset tests use.
    if r2 >= 0.90:
        if ratio <= E.sharpness_ratio:
            rep.blur = True
        elif ratio >= 1.0 / E.sharpness_ratio:
            rep.sharpen = True

    # --- block-level difference map ------------------------------------------
    grid = config.PIXEL.block_grid
    h, w = residual.shape
    bh, bw = max(1, h // grid), max(1, w // grid)
    rows = h // bh
    cols = w // bw
    blocks = residual[: rows * bh, : cols * bw].reshape(rows, bh, cols, bw).mean(axis=(1, 3))

    eps = config.PIXEL.block_diff_epsilon
    changed = blocks > eps
    rep.changed_fraction = float(changed.mean()) if blocks.size else 0.0

    median = float(np.median(blocks))
    strong = blocks > max(eps, median * E.localised_multiplier)

    # Peak statistic: the 95th percentile of |delta| within each block.  Thin
    # strokes move the tail, not the mean, so a faint text watermark that the
    # mean statistic misses still shows up here.
    flat = residual[: rows * bh, : cols * bw].reshape(rows, bh, cols, bw)
    flat = flat.transpose(0, 2, 1, 3).reshape(-1, bh * bw)
    peak = np.percentile(flat, 95, axis=1).reshape(rows, cols)
    peaked = peak > config.PIXEL.peak_diff_epsilon
    rep.peak_fraction = float(peaked.mean()) if peak.size else 0.0

    cluster = strong | peaked
    rep.localised_fraction = float(cluster.mean()) if blocks.size else 0.0
    if E.localised_min_frac <= rep.localised_fraction <= E.localised_max_frac:
        ys, xs = np.nonzero(cluster)
        if ys.size:
            rep.localised_box = (
                float(xs.min() / max(1, cols - 1)),
                float(ys.min() / max(1, rows - 1)),
                float(xs.max() / max(1, cols - 1)),
                float(ys.max() / max(1, rows - 1)),
            )
        # An overlay is only claimed when the *rest of the frame is essentially
        # pixel-identical*.  The looser 20% gate that preceded this let the peak
        # statistic fire on crops and unrelated images, where the peaks are just
        # misregistration rather than an edit.
        if (
            rep.changed_fraction <= config.PIXEL.local_diff_exact
            and rep.localised_fraction >= E.localised_min_frac
        ):
            rep.watermark = True

    # --- border ---------------------------------------------------------------
    if border is not None:
        rep.border, rep.border_frac = border
    else:
        rep.border, rep.border_frac = _detect_border(
            cand_rgb, orig_rgb.shape[1] / max(1, orig_rgb.shape[0])
        )

    # --- pure recompression ----------------------------------------------------
    # Structure and statistics are intact but pixels differ by a small amount
    # everywhere: the signature of a lossy re-encode.
    if (
        not (rep.brightness or rep.contrast or rep.colour or rep.blur
             or rep.sharpen or rep.watermark or rep.border)
        and rep.residual_mad > 0.4
        and r2 >= 0.985
        and rep.changed_fraction <= config.PIXEL.local_diff_near
    ):
        rep.recompress = True

    return rep
