"""Single source of truth for every threshold and weight in the 1:1 matcher.

Nothing in the matcher hard-codes a magic number.  Every decision boundary the
pipeline uses is declared here with the reason it was chosen and the evidence
behind it.  ``docs/MATCHING.md`` narrates the same values; ``scripts/benchmark.py``
re-measures them against the labelled fixture matrix so the numbers can be
re-derived instead of trusted.

Rationale summary (see docs/MATCHING.md for the long form)
---------------------------------------------------------
Perceptual hashing
    The comparative study in Electronics 15(7):1493 measured aHash/dHash/pHash/
    wHash on UKBench + ABO under identical preprocessing.  Their headline
    findings, which this config follows:
      * pHash (block-DCT) is the best all-round hash -- near-perfect exact-
        duplicate metrics AND the highest robustness of the four to JPEG
        recompression, scaling and blur.  It gets the largest weight.
      * dHash is the strongest cheap signal under brightness/gamma change.
      * aHash has the best raw perceptual robustness but the weakest
        discrimination, so it is down-weighted rather than dropped.
      * wHash (Haar) survives JPEG best but collapses under gamma correction.
    The same paper reports that a normalised Hamming similarity of
    ``S_hash >= 0.93`` identifies near-duplicates at high precision on a 64-bit
    hash, i.e. ~4.5 bits of difference.  That is why ``HASH_NEAR_SIM = 0.93``.

Feature matching
    ORB + Lowe ratio 0.75 + RANSAC homography at 4 px reprojection error is the
    standard OpenCV recipe for locating one image inside another.  It is what
    gives us the crop/coverage numbers rather than a global similarity guess.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict


# --------------------------------------------------------------------------
# Perceptual hash configuration
# --------------------------------------------------------------------------

#: Number of bits per perceptual hash.  64 is the standard size; the published
#: thresholds quoted above are calibrated for exactly this size.
HASH_BITS = 64

#: Working resolution used to compute the hashes.  32x32 is the conventional
#: pHash pre-DCT size and is what makes the hashes resize-invariant.
HASH_WORK_SIZE = 32

#: Relative importance of each hash in the fused ``hash_sim`` signal.
#: Weights are normalised at import time.
HASH_WEIGHTS: Dict[str, float] = {
    "phash": 0.45,   # block-DCT: best overall (robust to JPEG, scale, blur)
    "dhash": 0.25,   # gradient: best under brightness / gamma change
    "whash": 0.20,   # Haar DWT: best under heavy JPEG, weak under gamma
    "ahash": 0.10,   # mean threshold: robust but low discrimination
}

#: Normalised Hamming similarity (1 - dist/64) at which a hash pair is
#: considered a near-duplicate.  Directly from the 0.93 threshold in
#: Electronics 15(7):1493 sec. 4.3.
HASH_NEAR_SIM = 0.93

#: Below this a hash pair contributes nothing to the "same frame" hypothesis.
HASH_UNRELATED_SIM = 0.68


# --------------------------------------------------------------------------
# Stage 3 -- candidate filtering (the cheap gate before expensive work)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateFilter:
    """Which candidates are worth running the expensive comparison on."""

    #: Keep any candidate whose fused hash similarity reaches this.
    keep_hash_sim: float = 0.72
    #: Always keep at least this many candidates ranked by hash similarity, so
    #: that crops (which perturb global hashes badly) are not discarded.
    keep_top_n: int = 24
    #: A crop keeps most of the original colour palette even when its global
    #: hash drifts.  Aspect-tolerant colour similarity can rescue such a
    #: candidate; both conditions must hold.
    rescue_color_sim: float = 0.80
    rescue_hash_sim: float = 0.55
    #: Never spend stage-4 time on this many candidates no matter what.
    hard_cap: int = 60
    #: Longest side (px) used for the thumbnail pass.
    thumb_size: int = 96
    #: Longest side (px) used for the detailed pass.
    work_size: int = 512
    #: Longest side (px) used for SSIM / pixel metrics.
    compare_size: int = 256


CANDIDATE_FILTER = CandidateFilter()


# --------------------------------------------------------------------------
# Stage 4 -- pixel and structural metrics
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class PixelThresholds:
    """Boundaries on the stage-4 pixel metrics."""

    #: SSIM at which two frames are pixel-for-pixel the same rendered image.
    #: Identical frames resized + JPEG-recompressed typically stay above 0.95;
    #: two different photographs of one subject rarely clear 0.75.
    ssim_exact: float = 0.94
    ssim_near: float = 0.80
    ssim_similar: float = 0.45

    #: Fraction of 8x8 blocks (after gain/offset normalisation) that may differ
    #: before the pair stops being "the same frame".  This is the single most
    #: important guard against calling a different photo of the same subject a
    #: 1:1 match: a second shot differs in *most* blocks, a watermark or a text
    #: overlay differs in *a few*.
    local_diff_exact: float = 0.04
    local_diff_near: float = 0.12
    local_diff_similar: float = 0.40

    #: Block difference is counted when its mean |delta| exceeds this (0-255).
    block_diff_epsilon: float = 6.0
    #: Peak threshold on the 95th percentile of |delta| inside a block.
    #: A thin watermark stroke changes few pixels, so the block *mean* stays
    #: small while its upper tail stays large.  Measured on the fixtures: a
    #: text-only watermark peaks at 69.7 while resized, heavily-recompressed,
    #: brightened and recoloured copies peak at 10.4 or below -- so this
    #: separates them with no false positives, where the mean statistic missed
    #: the watermark by a hair (0.39% of blocks vs a 0.40% floor).
    peak_diff_epsilon: float = 18.0
    #: Grid used for the block analysis.
    block_grid: int = 32

    #: Gradient (Sobel) correlation.  Structure survives brightness/contrast
    #: edits, so a high value with low pixel agreement means "edited".
    grad_corr_high: float = 0.90
    grad_corr_similar: float = 0.55

    #: Colour-histogram correlation below which the palette clearly changed.
    hist_corr_similar: float = 0.80


PIXEL = PixelThresholds()


# --------------------------------------------------------------------------
# Stage 5 -- local feature correspondence and coverage
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureThresholds:
    """ORB / RANSAC settings and coverage geometry."""

    n_features: int = 1200
    #: Lowe's ratio test.
    ratio: float = 0.75
    #: RANSAC reprojection tolerance in pixels.
    ransac_reproj: float = 4.0
    #: Fewest inliers before the homography is trusted at all.
    min_inliers: int = 15
    #: Inlier count at which the feature signal saturates.  Below this the
    #: signal is scaled down, which is what stops "12 matches out of 12"
    #: between two unrelated images from scoring like a verified copy.
    inliers_saturating: int = 40

    #: ``cov_orig``  = share of the ORIGINAL that is present in the candidate.
    #: ``cov_cand``  = share of the CANDIDATE that is original content.
    #: A true full-frame 1:1 needs both near 1.
    coverage_full: float = 0.90
    #: Below this ``cov_orig`` the candidate is not a full-frame match and the
    #: crop branch takes over.
    coverage_crop_max: float = 0.96
    #: A crop must still explain most of itself as original content.
    coverage_crop_min_cand: float = 0.75

    #: Inlier ratio (inliers / min(matches, 40)) counted as strong.
    inlier_ratio_strong: float = 0.45
    inlier_ratio_weak: float = 0.15

    #: Multi-scale template-matching correlation counted as a confident crop
    #: localisation.
    template_score_confident: float = 0.60
    template_scales: tuple = (0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0)


FEATURES = FeatureThresholds()


# --------------------------------------------------------------------------
# Edit detection
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EditThresholds:
    """When a difference counts as an *edit* rather than a different image."""

    #: Global gain from the least-squares fit cand ~= a*orig + b.
    contrast_gain_min: float = 0.90
    contrast_gain_max: float = 1.11
    #: Global offset, on the 0-255 scale.
    brightness_offset: float = 5.0
    #: Per-channel mean shift (0-255) counted as a colour edit.
    colour_channel_shift: float = 4.0
    #: Blur/sharpen: ratio of Laplacian variances that counts as a filter.
    sharpness_ratio: float = 0.72
    #: Localised-edit detection: a "changed" block is one whose mean |delta|
    #: exceeds this multiple of the global median block difference.
    localised_multiplier: float = 6.0
    #: A changed fraction between these bounds with the rest of the frame clean
    #: reads as a watermark / text overlay / logo.
    localised_min_frac: float = 0.004
    localised_max_frac: float = 0.30
    #: A uniform frame around the image counts as a border above this thickness
    #: share of the smaller dimension.
    border_min_frac: float = 0.015
    border_uniform_std: float = 6.0


EDITS = EditThresholds()


# --------------------------------------------------------------------------
# Descriptor embedding (stage 5, classical -- no model download required)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingConfig:
    """Fixed-length classical visual descriptor used as the embedding signal."""

    grid: int = 4                 # spatial grid for colour moments + gradients
    colour_bins: int = 4          # bins per RGB channel in the global histogram
    hog_bins: int = 9             # gradient orientation bins per cell
    #: ``torch``/``open_clip`` are used only if importable; the pipeline never
    #: depends on them and reports which backend produced the embedding.
    allow_clip: bool = False
    clip_model: str = "ViT-B-32"


EMBEDDING = EmbeddingConfig()


# --------------------------------------------------------------------------
# Classification + confidence
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Classifier:
    """Fusion weights and decision boundaries.

    ``identity`` is a weighted blend of the independent signals below.  The
    weights are deliberately not uniform: the signals that survive
    recompression and resizing (hashes, gradients, features) dominate the ones
    that do not (raw pixel error).
    """

    #: identity = w . s / sum(w)
    weights: Dict[str, float] = field(default_factory=lambda: {
        "hash_sim":   0.22,
        "ssim":       0.22,
        "nrmse_sim":  0.08,
        "grad_corr":  0.14,
        "emb_cos":    0.14,
        "feat_sim":   0.12,
        "hist_corr":  0.08,
    })

    #: identity must clear this for any 1:1 bucket.
    identity_exact: float = 0.92
    identity_near: float = 0.84
    identity_similar: float = 0.55

    #: Both coverages must clear this for a full-frame match.
    full_frame: float = 0.86

    #: Steepness of the logistic that turns a decision margin into a
    #: percentage.  Calibrated by scripts/benchmark.py -- see docs/MATCHING.md
    #: "Confidence calibration".
    confidence_steepness: float = 7.0


CLASSIFIER = Classifier()


# --------------------------------------------------------------------------
# Public helpers
# --------------------------------------------------------------------------

def normalised_hash_weights() -> Dict[str, float]:
    total = sum(HASH_WEIGHTS.values())
    return {k: v / total for k, v in HASH_WEIGHTS.items()}


def normalised_identity_weights() -> Dict[str, float]:
    total = sum(CLASSIFIER.weights.values())
    return {k: v / total for k, v in CLASSIFIER.weights.items()}


def as_dict() -> dict:
    """Everything, for the /api/thresholds debug endpoint and the docs build."""
    return {
        "hash_bits": HASH_BITS,
        "hash_work_size": HASH_WORK_SIZE,
        "hash_weights": normalised_hash_weights(),
        "hash_near_sim": HASH_NEAR_SIM,
        "hash_unrelated_sim": HASH_UNRELATED_SIM,
        "candidate_filter": asdict(CANDIDATE_FILTER),
        "pixel": asdict(PIXEL),
        "features": asdict(FEATURES),
        "edits": asdict(EDITS),
        "embedding": asdict(EMBEDDING),
        "classifier": {
            "weights": normalised_identity_weights(),
            "identity_exact": CLASSIFIER.identity_exact,
            "identity_near": CLASSIFIER.identity_near,
            "identity_similar": CLASSIFIER.identity_similar,
            "full_frame": CLASSIFIER.full_frame,
            "confidence_steepness": CLASSIFIER.confidence_steepness,
        },
    }
