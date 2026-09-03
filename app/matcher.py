"""The 1:1 matcher: independent signal computation, fusion, classification.

This module never looks at anything a search engine said.  It is given two
images -- the original upload and a candidate -- and returns a verdict derived
entirely from measurements taken on those pixels.

Pipeline
--------
1. ``SignalSet`` -- every independent measurement, computed at the caller's
   chosen resolution.
2. ``fuse()`` -- the documented weighted blend into a single ``identity``
   score in [0, 1].
3. ``classify()`` -- a decision tree over ``identity``, the two coverage
   numbers and the attributed edits.
4. ``confidence()`` -- a logistic over the *margin* by which the chosen branch
   cleared its own boundary, so the percentage is derived from the evidence
   rather than asserted.

The classes exposed publicly are ``Verdict`` (one pair) and the helpers
``compare_images()`` / ``match_pair()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.metrics import (
    normalized_root_mse,
    peak_signal_noise_ratio,
    structural_similarity,
)

from . import config
from .descriptor import Embedding, compare as compare_embeddings, embed
from .editdetect import EditReport, analyze as analyze_edits, detect_border
from .features import FeatureMatch, TemplateMatch, locate, match_features, merge_coverage
from .geometry import Coverage, apply_orientation
from .hashing import Hashes, HashComparison, compare as compare_hashes
from .imaging import aspect_similarity, fit_size, resize_rgb, to_gray


class Label(str, Enum):
    """The seven verdicts required by the specification."""

    EXACT_1TO1 = "EXACT_1TO1"
    NEAR_1TO1 = "NEAR_1TO1"
    EDITED_1TO1 = "EDITED_1TO1"
    CROP = "CROP"
    VISUALLY_SIMILAR = "VISUALLY_SIMILAR"
    UNRELATED = "UNRELATED"
    ERROR = "ERROR"

    @property
    def is_one_to_one(self) -> bool:
        return self in (Label.EXACT_1TO1, Label.NEAR_1TO1, Label.EDITED_1TO1)

    @property
    def group(self) -> str:
        """UI grouping key, in display order."""
        return {
            Label.EXACT_1TO1: "exact",
            Label.NEAR_1TO1: "edited_resized",
            Label.EDITED_1TO1: "edited_resized",
            Label.CROP: "crops",
            Label.VISUALLY_SIMILAR: "similar",
            Label.UNRELATED: "unrelated",
            Label.ERROR: "unrelated",
        }[self]

    @property
    def display(self) -> str:
        return {
            Label.EXACT_1TO1: "1:1 MATCH",
            Label.NEAR_1TO1: "NEAR 1:1 MATCH",
            Label.EDITED_1TO1: "EDITED 1:1 MATCH",
            Label.CROP: "CROP OF ORIGINAL",
            Label.VISUALLY_SIMILAR: "SIMILAR IMAGE",
            Label.UNRELATED: "UNRELATED",
            Label.ERROR: "COULD NOT VERIFY",
        }[self]


GROUP_ORDER = ["exact", "edited_resized", "crops", "similar", "unrelated"]

GROUP_TITLES = {
    "exact": "EXACT 1:1 MATCHES",
    "edited_resized": "EDITED / RESIZED 1:1 MATCHES",
    "crops": "CROPS",
    "similar": "SIMILAR IMAGES",
    "unrelated": "UNRELATED / LOW CONFIDENCE",
}


# --------------------------------------------------------------------------
# Signals
# --------------------------------------------------------------------------

@dataclass
class SignalSet:
    """Every independent measurement taken on one (original, candidate) pair."""

    sha_identical: bool = False
    hash_sim: float = 0.0
    #: Global (whole-frame) hash similarity, kept for the report.
    hash_sim_global: float = 0.0
    #: Hash similarity measured on the aligned region.
    hash_sim_aligned: float = 0.0
    hash_sim_unoriented: float = 0.0
    hash_best: float = 0.0
    hash_worst: float = 0.0
    hash_best_name: str = ""
    aspect_sim: float = 1.0
    ssim: float = 0.0
    ms_ssim: float = 0.0
    nrmse_sim: float = 0.0
    psnr: float = 0.0
    grad_corr: float = 0.0
    hist_corr: float = 0.0
    emb_cos: float = 0.0
    emb_backend: str = ""
    feat_inlier_ratio: float = 0.0
    feat_inliers: int = 0
    feat_good: int = 0
    feat_trusted: bool = False
    coverage_orig: float = 0.0
    coverage_cand: float = 0.0
    coverage_source: str = "none"
    template_score: float = 0.0
    template_direction: str = "none"
    aligned_ssim: Optional[float] = None
    align_method: str = "none"
    align_valid_fraction: float = 1.0
    ssim_raw: float = 0.0
    changed_fraction: float = 1.0
    localised_fraction: float = 0.0
    orientation: str = "identity"
    size_orig: Tuple[int, int] = (0, 0)
    size_cand: Tuple[int, int] = (0, 0)

    @property
    def feat_sim(self) -> float:
        """Feature evidence mapped to [0, 1]; an untrusted homography is 0.

        Scales with the *number* of inliers as well as their ratio.  Ratio alone
        was too generous: 12 inliers out of 12 good matches on two unrelated
        images scored a perfect 1.0, because the denominator shrank with the
        evidence.  Requiring a real inlier population fixes that.
        """
        if not self.feat_trusted:
            return 0.0
        F = config.FEATURES
        count_term = min(1.0, self.feat_inliers / F.inliers_saturating)
        ratio_term = min(1.0, self.feat_inlier_ratio / F.inlier_ratio_strong)
        return float(count_term * ratio_term)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["size_orig"] = list(self.size_orig)
        d["size_cand"] = list(self.size_cand)
        d["feat_sim"] = round(self.feat_sim, 4)
        for k in ("hash_sim", "hash_sim_global", "hash_sim_aligned", "hash_sim_unoriented",
                  "hash_best", "hash_worst", "ssim",
                  "ssim_raw", "ms_ssim", "nrmse_sim", "grad_corr", "hist_corr", "emb_cos",
                  "feat_inlier_ratio", "coverage_orig", "coverage_cand", "template_score",
                  "changed_fraction", "localised_fraction", "aspect_sim",
                  "align_valid_fraction"):
            d[k] = round(float(d[k]), 4)
        d["aligned_ssim"] = None if d["aligned_ssim"] is None else round(float(d["aligned_ssim"]), 4)
        d["psnr"] = round(float(d["psnr"]), 2)
        return d


# --------------------------------------------------------------------------
# Individual metric helpers
# --------------------------------------------------------------------------

def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    ga, gb = to_gray(a).astype(np.uint8), to_gray(b).astype(np.uint8)
    win = 7 if min(ga.shape) >= 7 else 3
    return float(structural_similarity(ga, gb, data_range=255, win_size=win))


def _ms_ssim(a: np.ndarray, b: np.ndarray, levels: int = 3) -> float:
    """Multiscale SSIM: average SSIM across a Gaussian pyramid."""
    from PIL import Image
    ga = to_gray(a).astype(np.uint8)
    gb = to_gray(b).astype(np.uint8)
    scores = []
    for _ in range(levels):
        if min(ga.shape) < 16:
            break
        win = 7 if min(ga.shape) >= 7 else 3
        scores.append(float(structural_similarity(ga, gb, data_range=255, win_size=win)))
        ga = np.asarray(Image.fromarray(ga).resize(
            (max(8, ga.shape[1] // 2), max(8, ga.shape[0] // 2)), Image.BILINEAR))
        gb = np.asarray(Image.fromarray(gb).resize(
            (max(8, gb.shape[1] // 2), max(8, gb.shape[0] // 2)), Image.BILINEAR))
    return float(np.mean(scores)) if scores else 0.0


def _nrmse_sim(a: np.ndarray, b: np.ndarray) -> float:
    u8a, u8b = to_gray(a).astype(np.uint8), to_gray(b).astype(np.uint8)
    return float(np.clip(1.0 - normalized_root_mse(u8a, u8b, normalization="euclidean"), 0.0, 1.0))


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Peak signal-to-noise ratio in dB; 100 dB stands in for "bit-identical"."""
    u8a, u8b = to_gray(a).astype(np.uint8), to_gray(b).astype(np.uint8)
    if np.array_equal(u8a, u8b):
        return 100.0
    try:
        val = peak_signal_noise_ratio(u8a, u8b, data_range=255)
    except Exception:                              # pragma: no cover
        return 0.0
    return 100.0 if math.isinf(val) else float(val)


def _grad_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of Sobel gradient magnitudes.

    Invariant to global brightness offset and (up to scale) to contrast gain,
    which is why it carries real weight: it keeps an edited copy scoring high
    while a different photograph -- different edges -- scores low.
    """
    ga = cv2.Sobel(to_gray(a).astype(np.uint8), cv2.CV_32F, 1, 1, ksize=3).ravel()
    gb = cv2.Sobel(to_gray(b).astype(np.uint8), cv2.CV_32F, 1, 1, ksize=3).ravel()
    if ga.size < 16:
        return 0.0
    sa, sb = ga.std(), gb.std()
    if sa < 1e-6 or sb < 1e-6:
        return 0.0
    return float(np.clip(np.mean((ga - ga.mean()) * (gb - gb.mean())) / (sa * sb), 0.0, 1.0))


def _hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    ha = cv2.calcHist([a], [0, 1], None, [32, 32], [0, 256, 0, 256])
    hb = cv2.calcHist([b], [0, 1], None, [32, 32], [0, 256, 0, 256])
    cv2.normalize(ha, ha, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hb, hb, 0, 1, cv2.NORM_MINMAX)
    val = cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)
    return float(np.clip(val, 0.0, 1.0))


# --------------------------------------------------------------------------
# Orientation selection (cheap, hash driven)
# --------------------------------------------------------------------------

def best_orientation(
    orig_hashes: Hashes,
    cand_rgb: np.ndarray,
) -> Tuple[str, float, Hashes]:
    """Cheapest orientation search: compare hashes under all 8 orientations.

    Rotations and mirrors destroy global perceptual hashes, so a rotated copy
    would otherwise be thrown out at stage 3.  Testing the eight rigid
    orientations against the original's hash costs milliseconds and recovers
    them.  The winning orientation is used for every expensive metric and is
    reported, so the UI can say "rotated 90 deg" instead of hiding it.

    A non-identity orientation is only accepted when it beats identity by a
    clear margin.  Without that guard a crop -- whose hash is weak in *every*
    orientation -- can latch onto a spurious mirror and then be compared
    upside down.

    Returns ``(orientation_name, fused_similarity, hashes_in_that_orientation)``.
    """
    from .hashing import compute as compute_hashes
    from .imaging import LoadedImage

    results = []
    for name in geometry_orientation_names():
        rgb = apply_orientation(cand_rgb, name)
        h = compute_hashes(_as_loaded(rgb))
        results.append((name, compare_hashes(orig_hashes, h).fused, h, rgb))

    identity_row = next(r for r in results if r[0] == "identity")
    best = max(results, key=lambda r: r[1])
    if best[0] != "identity" and best[1] - identity_row[1] < ORIENTATION_MARGIN:
        best = identity_row
    return best[0], best[1], best[2]


#: How much better a rotation/mirror must score than identity to be believed.
ORIENTATION_MARGIN = 0.08


def _as_loaded(rgb: np.ndarray) -> "LoadedImage":
    from .imaging import LoadedImage
    return LoadedImage(
        rgb=rgb, width=rgb.shape[1], height=rgb.shape[0], fmt="MEM",
        sha256="", byte_size=0, had_exif_orientation=False, mode="RGB",
    )


#: How much better identity must sweep than the hash-chosen rotation before the
#: rotation is discarded.
ORIENTATION_GEOMETRY_MARGIN = 0.08


def resolve_orientation(
    orig_hashes: Hashes,
    orig_rgb: np.ndarray,
    cand_rgb_raw: np.ndarray,
    work_size: int,
) -> Tuple[str, Hashes, np.ndarray, float]:
    """Pick the candidate orientation, then *verify* it geometrically.

    The hash vote alone is not trustworthy for crops.  Measured on the test
    fixtures: for a centre crop of a near-symmetric scene the mirrored copy
    hashed *closer* to the original than the unmirrored one did (0.653 vs
    0.573), because the scene is roughly left-right symmetric -- and the mirror
    then made the crop impossible to localise (template score 0.664 and a
    nonsense 2% coverage, against 0.963 and a correct 26% in identity).

    So when the hash vote proposes a rotation or mirror, the template sweep is
    run in both orientations and identity wins unless the rotation genuinely
    localises better.  This costs one extra sweep and only on the rare
    candidate where a rotation was proposed at all.
    """
    from .features import _sweep
    from .hashing import compute as compute_hashes

    orientation, orient_sim, hashes = best_orientation(orig_hashes, cand_rgb_raw)
    if orientation == "identity":
        return "identity", hashes, cand_rgb_raw, orient_sim

    ow, oh = fit_size(orig_rgb.shape[1], orig_rgb.shape[0], work_size)
    orig_work = resize_rgb(orig_rgb, (ow, oh))

    def sweep_score(rgb: np.ndarray) -> float:
        cw, ch = fit_size(rgb.shape[1], rgb.shape[0], work_size)
        c_work = resize_rgb(rgb, (cw, ch))
        sa, _ = _sweep(orig_work, c_work, 256)
        sb, _ = _sweep(c_work, orig_work, 256)
        return max(sa, sb)

    rotated = apply_orientation(cand_rgb_raw, orientation)
    if sweep_score(cand_rgb_raw) - sweep_score(rotated) >= ORIENTATION_GEOMETRY_MARGIN:
        ident_hashes = compute_hashes(_as_loaded(cand_rgb_raw))
        ident_sim = compare_hashes(orig_hashes, ident_hashes).fused
        return "identity", ident_hashes, cand_rgb_raw, ident_sim
    return orientation, hashes, rotated, orient_sim


def geometry_orientation_names() -> List[str]:
    from .geometry import orientation_names
    return orientation_names()



# --------------------------------------------------------------------------
# Signal computation
# --------------------------------------------------------------------------

def warp_into_original(
    orig_size: Tuple[int, int],
    cand_rgb: np.ndarray,
    homography: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Express the candidate in the ORIGINAL's coordinate frame.

    Returns ``(warped_candidate, valid_mask)``.  This is what makes one code
    path handle both containment directions:

    * candidate is the original plus a border -> the warp covers the whole
      original frame, so the valid mask is essentially full;
    * candidate is a crop of the original -> the warp covers only the cropped
      part, and the valid mask marks exactly which part.
    """
    ow, oh = orig_size
    try:
        h_inv = np.linalg.inv(homography)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(h_inv)):
        return None
    warped = cv2.warpPerspective(
        cand_rgb, h_inv, (ow, oh),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )
    ones = np.full(cand_rgb.shape[:2], 255, dtype=np.uint8)
    mask = cv2.warpPerspective(
        ones, h_inv, (ow, oh),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    return warped, mask > 127


def _bbox_of(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def align_pair(
    orig_work: np.ndarray,
    cand_work: np.ndarray,
    feat: FeatureMatch,
    tpl: TemplateMatch,
    compare_size: int,
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    """Line the two images up before measuring them.

    Returns ``(a, b, method, valid_fraction)`` at a common ``compare_size``
    square.  Priority:

    1. already coextensive (both coverages ~1) -- no resampling at all, so an
       exact copy is not penalised by an unnecessary warp;
    2. a trusted homography -- the candidate is warped into the original's
       frame and both are cropped to the valid region;
    3. a confident template sweep -- the localised box is cut out and resized;
    4. nothing trustworthy -- plain full-frame stretch, which is the honest
       fallback and is why ``valid_fraction`` is reported.
    """
    def pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return (
            resize_rgb(a, (compare_size, compare_size)),
            resize_rgb(b, (compare_size, compare_size)),
        )

    cov = feat.coverage if feat.trusted else None
    if cov is not None and cov.original >= 0.98 and cov.candidate >= 0.98:
        a, b = pair(orig_work, cand_work)
        return a, b, "coextensive", 1.0

    if feat.trusted and feat.homography is not None:
        warped = warp_into_original(
            (orig_work.shape[1], orig_work.shape[0]), cand_work, feat.homography
        )
        if warped is not None:
            w_cand, mask = warped
            box = _bbox_of(mask)
            if box is not None:
                x0, y0, x1, y1 = box
                if (x1 - x0) >= 16 and (y1 - y0) >= 16:
                    a, b = pair(orig_work[y0:y1, x0:x1], w_cand[y0:y1, x0:x1])
                    return a, b, "homography", float(mask.mean())

    if tpl.confident and tpl.aligned_region is not None:
        if tpl.direction == "original-in-candidate":
            a, b = pair(orig_work, tpl.aligned_region)
        else:
            a, b = pair(tpl.aligned_region, cand_work)
        return a, b, "template", 1.0

    a, b = pair(orig_work, cand_work)
    return a, b, "none", 1.0


def compute_signals(
    orig_hashes: Hashes,
    orig_rgb: np.ndarray,
    cand_hashes: Hashes,
    cand_rgb_raw: np.ndarray,
    *,
    work_size: int = None,
    compare_size: int = None,
    run_features: bool = True,
) -> Tuple["SignalSet", EditReport, FeatureMatch, TemplateMatch]:
    """Take every independent measurement on one (original, candidate) pair.

    Every pixel metric is taken on the *aligned* pair when a trustworthy
    alignment exists.  Measuring a bordered or cropped copy against the
    stretched full frame is what made those cases score as merely "similar":
    the content was identical, it was just not where the metric was looking.
    """
    cf = config.CANDIDATE_FILTER
    work_size = work_size or cf.work_size
    compare_size = compare_size or cf.compare_size

    sig = SignalSet()
    sig.sha_identical = bool(orig_hashes.sha256 and orig_hashes.sha256 == cand_hashes.sha256)
    sig.size_orig = (orig_rgb.shape[1], orig_rgb.shape[0])
    sig.size_cand = (cand_rgb_raw.shape[1], cand_rgb_raw.shape[0])

    # --- orientation (hashes are recomputed in the winning orientation) --------
    orientation, cand_hashes_oriented, cand_rgb, orient_sim = resolve_orientation(
        orig_hashes, orig_rgb, cand_rgb_raw, work_size
    )
    sig.orientation = orientation
    sig.size_cand = (cand_rgb.shape[1], cand_rgb.shape[0])
    sig.aspect_sim = aspect_similarity(
        sig.size_orig[0] / max(1, sig.size_orig[1]),
        sig.size_cand[0] / max(1, sig.size_cand[1]),
    )

    # Hash similarity must compare like with like, i.e. in the same orientation.
    hc = compare_hashes(orig_hashes, cand_hashes_oriented)
    sig.hash_sim_global = hc.fused
    sig.hash_sim, sig.hash_best, sig.hash_worst = hc.fused, hc.best, hc.worst
    sig.hash_best_name = hc.best_name
    sig.hash_sim_unoriented = orient_sim

    from .hashing import histogram_cosine
    sig.hist_corr = histogram_cosine(orig_hashes.color_hist, cand_hashes_oriented.color_hist)

    # --- working-resolution copies ----------------------------------------------
    ow, oh = fit_size(*sig.size_orig, work_size)
    cw, ch = fit_size(*sig.size_cand, work_size)
    orig_work = resize_rgb(orig_rgb, (ow, oh))
    cand_work = resize_rgb(cand_rgb, (cw, ch))

    # --- embedding ---------------------------------------------------------------
    emb_a, emb_b = embed(orig_work), embed(cand_work)
    sig.emb_cos, sig.emb_backend = compare_embeddings(emb_a, emb_b)

    # --- geometry ------------------------------------------------------------------
    feat = _empty_features()
    tpl = TemplateMatch(-1.0, None, "none", 0.0, None)
    if run_features:
        feat = match_features(to_gray(orig_work), to_gray(cand_work))
        tpl = locate(orig_work, cand_work, work=min(256, work_size))
    sig.feat_inliers, sig.feat_good = feat.inliers, feat.good_matches
    sig.feat_inlier_ratio, sig.feat_trusted = feat.inlier_ratio, feat.trusted
    sig.template_score, sig.template_direction = tpl.score, tpl.direction

    coverage, source = merge_coverage(feat, tpl)
    sig.coverage_source = source
    if coverage is not None:
        sig.coverage_orig, sig.coverage_cand = coverage.original, coverage.candidate

    # --- aligned pixel metrics ---------------------------------------------------------
    a, b, method, valid_frac = align_pair(orig_work, cand_work, feat, tpl, compare_size)
    sig.align_method = method
    sig.align_valid_fraction = valid_frac
    sig.ssim = _ssim(a, b)
    sig.ms_ssim = _ms_ssim(a, b)
    sig.nrmse_sim = _nrmse_sim(a, b)
    sig.psnr = _psnr(a, b)
    sig.grad_corr = _grad_corr(a, b)

    edits = analyze_edits(a, b)
    sig.changed_fraction = edits.changed_fraction
    sig.localised_fraction = edits.localised_fraction

    # Unaligned reference values, kept for the report so a reader can see how
    # much the alignment mattered.
    ra = resize_rgb(orig_rgb, (compare_size, compare_size))
    rb = resize_rgb(cand_rgb, (compare_size, compare_size))
    sig.ssim_raw = _ssim(ra, rb)

    # Borders and letterbox bars only exist *before* alignment -- registration
    # crops them away, so a border detected on the aligned pair is always
    # invisible.  Detect them on aspect-preserving copies of the real images,
    # which is also the only place the aspect-ratio evidence survives.
    border_found, border_frac = detect_border(orig_rgb, cand_rgb)
    if border_found and not edits.border:
        edits.border = True
        edits.border_frac = border_frac

    # Global hashes cannot survive a crop or a border, which unfairly penalised
    # otherwise-perfect matches: a bordered copy scored 0.69 hash similarity and
    # dropped to 88.7% confidence despite 0.1% of its blocks differing.  When a
    # trustworthy alignment exists, measure the hashes on the aligned content --
    # the region that actually corresponds -- and use whichever is stronger.
    if method in ("homography", "template"):
        from .hashing import compute as compute_hashes
        aligned_cmp = compare_hashes(
            compute_hashes(_as_loaded(a)), compute_hashes(_as_loaded(b))
        )
        sig.hash_sim_aligned = aligned_cmp.fused
        if aligned_cmp.fused > sig.hash_sim:
            sig.hash_sim = aligned_cmp.fused
            sig.hash_best = max(sig.hash_best, aligned_cmp.best)
            sig.hash_best_name = f"{aligned_cmp.best_name} (aligned)"

    return sig, edits, feat, tpl


def fuse(sig: SignalSet) -> float:
    """Weighted blend of the independent signals into ``identity`` in [0, 1].

    Weights come from ``config.CLASSIFIER.weights`` (see docs/MATCHING.md for
    the reasoning).  Signals that survive recompression and resizing dominate
    raw pixel error, which is the one metric a lossy re-encode destroys.
    """
    w = config.normalised_identity_weights()
    values = {
        "hash_sim": sig.hash_sim,
        "ssim": sig.ssim,
        "nrmse_sim": sig.nrmse_sim,
        "grad_corr": sig.grad_corr,
        "emb_cos": sig.emb_cos,
        "feat_sim": sig.feat_sim,
        "hist_corr": sig.hist_corr,
    }
    return float(sum(w[k] * float(np.clip(values[k], 0.0, 1.0)) for k in w))


def frame_agreement(sig: SignalSet) -> float:
    """How strongly the geometry says the two frames cover the same content.

    Both coverages must be high.  When no geometric evidence was obtained the
    aspect ratio is used as a weak stand-in, which can only ever reach ~1.0 for
    a matching aspect -- never enough on its own to assert a 1:1 match, since
    ``identity`` has to clear its own threshold too.
    """
    if sig.coverage_source == "none":
        return float(np.clip(sig.aspect_sim * 0.9, 0.0, 0.9))
    return float(np.clip(sig.coverage_orig * sig.coverage_cand, 0.0, 1.0))


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

MATERIAL_EDITS = ("brightness", "contrast", "colour", "blur", "sharpen", "watermark", "border")


def _empty_features() -> FeatureMatch:
    """Neutral placeholder; ``match_pair`` swaps in the real measurement."""
    return FeatureMatch(0, 0, 0, 0, 0, 0.0, None, None, "none")


@dataclass
class Verdict:
    """One (original, candidate) determination."""

    label: Label
    confidence: float
    identity: float
    frame_agreement: float
    signals: SignalSet
    edits: EditReport
    features: FeatureMatch
    template: TemplateMatch
    reasons: List[str] = field(default_factory=list)
    overlap: Optional[float] = None

    @property
    def group(self) -> str:
        return self.label.group

    def to_dict(self) -> dict:
        return {
            "label": self.label.value,
            "display": self.label.display,
            "is_one_to_one": self.label.is_one_to_one,
            "group": self.group,
            "confidence": round(self.confidence, 1),
            "identity": round(self.identity, 4),
            "frame_agreement": round(self.frame_agreement, 4),
            "overlap": None if self.overlap is None else round(self.overlap, 4),
            "reasons": self.reasons,
            "signals": self.signals.to_dict(),
            "edits": self.edits.to_dict(),
            "features": self.features.to_dict(),
            "template": self.template.to_dict(),
        }


def _confidence(margin: float) -> float:
    """Logistic over a normalised decision margin.

    ``margin`` is how far the binding evidence sits past its own threshold, in
    units of the distance from that threshold to 1.0.  The steepness constant
    lives in ``config.CLASSIFIER.confidence_steepness`` and is calibrated by
    ``scripts/benchmark.py``.  Nothing here is a hand-picked percentage.
    """
    z = config.CLASSIFIER.confidence_steepness * float(np.clip(margin, -1.5, 1.5))
    return float(100.0 / (1.0 + math.exp(-z)))


def classify(sig: SignalSet, edits: EditReport) -> Verdict:
    """Decision tree over the measured signals.  Order matters.

    The two questions are kept strictly separate:

    * **Geometry** -- do the two frames cover the same content?  Answered by
      ``coverage_orig`` / ``coverage_cand``, never by a similarity score.
    * **Fidelity** -- given that they do, are the pixels the same?  Answered by
      ``identity`` and by how many blocks still differ *after alignment*.

    A second photograph of the same subject passes the geometry question and
    fails the fidelity one.  That is exactly the false positive the
    specification calls out, so ``changed_fraction`` gates every 1:1 bucket.
    """
    C = config.CLASSIFIER
    P = config.PIXEL
    F = config.FEATURES

    identity = fuse(sig)
    agreement = frame_agreement(sig)
    reasons: List[str] = []

    material_edits = [e for e in edits.labels if any(m in e for m in MATERIAL_EDITS)]
    if sig.orientation != "identity":
        material_edits = [_orientation_label(sig.orientation)] + material_edits

    # Geometry, with a deliberately weak stand-in when nothing was measurable.
    has_geometry = sig.coverage_source != "none"
    if has_geometry:
        cov_o, cov_c = sig.coverage_orig, sig.coverage_cand
    else:
        cov_o = cov_c = float(np.clip(sig.aspect_sim, 0.0, 0.95))
        reasons.append("no geometric alignment obtained; falling back to aspect ratio")

    clean = sig.changed_fraction <= P.local_diff_near

    def mk(label: Label, margin: float, overlap=None) -> Verdict:
        if not label.is_one_to_one:
            edits.suppress_labels = True
        return Verdict(
            label=label,
            confidence=_confidence(margin),
            identity=identity,
            frame_agreement=agreement,
            signals=sig,
            edits=edits,
            features=_empty_features(),
            template=TemplateMatch(sig.template_score, None, sig.template_direction, 0.0, None),
            reasons=reasons,
            overlap=overlap,
        )

    # 0. byte-identical files ---------------------------------------------------
    if sig.sha_identical:
        reasons.append("SHA-256 of the two files is identical")
        v = mk(Label.EXACT_1TO1, 1.0)
        v.confidence = 100.0
        return v

    # 1. crop: only part of the original is present -------------------------------
    if cov_o < F.coverage_crop_max and cov_c >= F.coverage_crop_min_cand:
        # The aligned region must still be the same picture.  A different
        # photograph of the same subject also produces a plausible homography
        # with high coverage -- what gives it away is that its aligned pixels
        # differ almost everywhere.
        if sig.ssim >= P.ssim_near and clean:
            overlap = cov_o
            margin = 0.5 * min(1.0, sig.ssim / P.ssim_exact) + 0.5 * cov_c
            reasons.append(
                f"only {overlap * 100:.0f}% of the original is present "
                f"(candidate is {cov_c * 100:.0f}% original content)"
            )
            reasons.append(
                f"aligned SSIM {sig.ssim:.3f}, {sig.changed_fraction * 100:.1f}% of blocks differ "
                f"(alignment: {sig.align_method})"
            )
            return mk(Label.CROP, margin, overlap=overlap)
        reasons.append(
            f"partial coverage ({cov_o * 100:.0f}%) but the aligned pixels differ in "
            f"{sig.changed_fraction * 100:.0f}% of blocks -- not a crop of this image"
        )

    # 2. full original present ------------------------------------------------------
    if cov_o >= F.coverage_full and identity >= C.identity_near and clean:
        extra_content = cov_c < F.coverage_crop_max
        if material_edits or extra_content:
            margin = _identity_margin(identity, C.identity_near)
            if extra_content and "border" not in " ".join(material_edits):
                reasons.append(
                    f"the original is fully present but fills only {cov_c * 100:.0f}% "
                    f"of the candidate -- extra content around it"
                )
            if material_edits:
                reasons.append("edits detected: " + ", ".join(material_edits))
            reasons.append(
                f"identity {identity:.3f}, {sig.changed_fraction * 100:.1f}% of blocks differ"
            )
            return mk(Label.EDITED_1TO1, margin)

        if identity >= C.identity_exact and sig.changed_fraction <= P.local_diff_exact:
            margin = _identity_margin(identity, C.identity_exact) * 1.2
            reasons.append(
                f"identity {identity:.3f} above the {C.identity_exact:.2f} exact threshold, "
                f"{sig.changed_fraction * 100:.2f}% of blocks differ"
            )
            if edits.labels:
                reasons.append("benign transform only: " + ", ".join(edits.labels))
            reasons.append(f"alignment: {sig.align_method}")
            return mk(Label.EXACT_1TO1, margin)

        margin = _identity_margin(identity, C.identity_near)
        reasons.append(
            f"identity {identity:.3f} in the near band "
            f"[{C.identity_near:.2f}, {C.identity_exact:.2f})"
        )
        if edits.labels:
            reasons.append("benign transform only: " + ", ".join(edits.labels))
        return mk(Label.NEAR_1TO1, margin)

    # 3. why the 1:1 branch was not taken ---------------------------------------------
    if cov_o < F.coverage_full:
        reasons.append(f"only {cov_o * 100:.0f}% of the original is present")
    if not clean:
        reasons.append(
            f"{sig.changed_fraction * 100:.1f}% of blocks differ -- above the "
            f"{P.local_diff_near * 100:.0f}% limit for a 1:1 claim"
        )
    if identity < C.identity_near:
        reasons.append(
            f"identity {identity:.3f} below the {C.identity_near:.2f} 1:1 threshold"
        )

    # 4. visually similar, or unrelated --------------------------------------------------
    if identity >= C.identity_similar and sig.ssim >= P.ssim_similar:
        margin = (identity - C.identity_similar) / max(1e-6, 1.0 - C.identity_similar)
        reasons.append(
            f"identity {identity:.3f} and SSIM {sig.ssim:.3f} clear the similar thresholds "
            f"({C.identity_similar:.2f} / {P.ssim_similar:.2f}) but nothing higher"
        )
        return mk(Label.VISUALLY_SIMILAR, margin)

    reasons.append(
        f"identity {identity:.3f} / SSIM {sig.ssim:.3f} below the similar thresholds"
    )
    return mk(Label.UNRELATED, (identity - C.identity_similar) / max(1e-6, 1.0 - C.identity_similar))


_ORIENTATION_LABELS = {
    "rot90": "rotated 90 degrees",
    "rot180": "rotated 180 degrees",
    "rot270": "rotated 270 degrees",
    "mirror-h": "mirrored horizontally",
    "mirror-h-rot90": "mirrored and rotated 90 degrees",
    "mirror-h-rot180": "mirrored and rotated 180 degrees",
    "mirror-h-rot270": "mirrored and rotated 270 degrees",
}


def _orientation_label(name: str) -> str:
    return _ORIENTATION_LABELS.get(name, name)


def _identity_margin(identity: float, threshold: float) -> float:
    """How far ``identity`` is past ``threshold``, normalised to the remaining headroom."""
    return (identity - threshold) / max(1e-6, 1.0 - threshold)


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

@dataclass
class MatchInput:
    hashes: Hashes
    rgb: np.ndarray


def match_pair(
    original: MatchInput,
    candidate: MatchInput,
    *,
    work_size: int = None,
    compare_size: int = None,
    run_features: bool = True,
) -> Verdict:
    """Full stage-4/5 comparison of one candidate against the original."""
    sig, edits, feat, tpl = compute_signals(
        original.hashes, original.rgb, candidate.hashes, candidate.rgb,
        work_size=work_size, compare_size=compare_size, run_features=run_features,
    )
    verdict = classify(sig, edits)
    # classify() builds placeholder containers; attach the real ones so the
    # report carries the actual inlier counts and template box.
    verdict.features = feat
    verdict.template = tpl
    return verdict


@dataclass
class QuickPair:
    """Cheap candidate-vs-candidate comparison, used only for merge decisions."""

    identity: float
    ssim: float
    changed_fraction: float
    hash_sim: float
    #: Material edits separating the two, e.g. ``["brightness"]``.
    edit_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k in ("identity", "ssim", "changed_fraction", "hash_sim"):
            d[k] = round(float(d[k]), 4)
        return d


def quick_identity(
    a_hashes: Hashes,
    a_rgb: np.ndarray,
    b_hashes: Hashes,
    b_rgb: np.ndarray,
    size: int = 112,
) -> QuickPair:
    """Compare two *candidates* cheaply, to decide whether they are one image.

    Deliberately omits the 8-orientation search, ORB/RANSAC and the template
    sweep.  Using the full :func:`match_pair` here was both slow (candidate-vs-
    candidate pairs grow quadratically, and 66 pairs cost ~7s) and wrong: it
    applied orientation correction, so a rotated copy and a brightened copy
    looked like the same image and were merged into a single card.  Two results
    should only be merged when they are the same *rendered* image.
    """
    from .hashing import histogram_cosine

    hc = compare_hashes(a_hashes, b_hashes)
    a = resize_rgb(a_rgb, (size, size))
    b = resize_rgb(b_rgb, (size, size))
    ssim = _ssim(a, b)
    grad = _grad_corr(a, b)
    nrmse = _nrmse_sim(a, b)
    hist = histogram_cosine(a_hashes.color_hist, b_hashes.color_hist)
    edits = analyze_edits(a, b)
    # ``changed_fraction`` is measured *after* the gain/offset fit, so a pure
    # brightness or contrast edit leaves it near zero.  Merge decisions need the
    # attributed causes as well, or a brightened copy merges into the plain one.
    border_found, border_frac = detect_border(a_rgb, b_rgb)
    if border_found:
        edits.border = True
        edits.border_frac = border_frac

    w = config.normalised_identity_weights()
    identity = (
        w["hash_sim"] * hc.fused
        + w["ssim"] * ssim
        + w["nrmse_sim"] * nrmse
        + w["grad_corr"] * grad
        + w["emb_cos"] * 0.0            # embedding skipped: cost outweighs value here
        + w["feat_sim"] * 0.0           # geometry skipped for the same reason
        + w["hist_corr"] * hist
    )
    # Renormalise over the signals actually computed so the value stays
    # comparable with the full ``identity`` scale.
    used = w["hash_sim"] + w["ssim"] + w["nrmse_sim"] + w["grad_corr"] + w["hist_corr"]
    return QuickPair(
        identity=float(identity / used),
        ssim=float(ssim),
        changed_fraction=float(edits.changed_fraction),
        hash_sim=float(hc.fused),
        edit_labels=[e for e in edits.labels if any(m in e for m in MATERIAL_EDITS)],
    )


def error_verdict(reason: str) -> Verdict:
    sig = SignalSet()
    return Verdict(
        label=Label.ERROR,
        confidence=0.0,
        identity=0.0,
        frame_agreement=0.0,
        signals=sig,
        edits=EditReport(),
        features=_empty_features(),
        template=TemplateMatch(-1.0, None, "none", 0.0, None),
        reasons=[reason],
    )
