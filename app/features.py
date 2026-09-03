"""Local feature correspondence (stage 5) and multi-scale crop localisation.

Two independent geometric estimators back the crop / coverage numbers:

1. **ORB + Lowe ratio test + RANSAC homography.**  Rotation, scale and mild
   perspective tolerant.  Gives an inlier ratio (how trustworthy the
   correspondence is) and a homography (where the original actually sits).
2. **Multi-scale edge template matching.**  Rotation intolerant but immune to
   descriptor noise, and it localises an axis-aligned crop precisely.  It is
   the fallback -- and the cross-check -- when feature matching is weak, which
   is common for low-texture images where ORB produces few keypoints.

Both report coverage in the same terms:

``original``  -- share of the ORIGINAL present in the candidate
``candidate`` -- share of the CANDIDATE explained by original content
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import cv2
import numpy as np

from . import config
from .geometry import (
    Coverage,
    coverage_from_box,
    coverage_from_homography,
    coverage_of_crop,
)
from .imaging import fit_size, resize_rgb, to_gray


@dataclass
class FeatureMatch:
    """Result of the ORB/RANSAC pass."""

    keypoints_orig: int
    keypoints_cand: int
    raw_matches: int
    good_matches: int
    inliers: int
    inlier_ratio: float
    homography: Optional[np.ndarray]
    coverage: Optional[Coverage]
    #: ``'similarity'`` (4-DOF, the normal case) or ``'homography'`` (8-DOF fallback).
    model: str = "none"

    @property
    def trusted(self) -> bool:
        return self.homography is not None and self.inliers >= config.FEATURES.min_inliers

    def to_dict(self) -> dict:
        return {
            "keypoints_orig": self.keypoints_orig,
            "keypoints_cand": self.keypoints_cand,
            "raw_matches": self.raw_matches,
            "good_matches": self.good_matches,
            "inliers": self.inliers,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "trusted": self.trusted,
            "model": self.model,
            "coverage": self.coverage.to_dict() if self.coverage else None,
        }


def _estimate_similarity(src: np.ndarray, dst: np.ndarray):
    """RANSAC similarity transform (rotation + uniform scale + translation).

    Returns ``(H_3x3, inlier_mask, model_name)`` or ``(None, None, 'none')``.
    """
    try:
        M, mask = cv2.estimateAffinePartial2D(
            src, dst,
            method=cv2.RANSAC,
            ransacReprojThreshold=config.FEATURES.ransac_reproj,
            maxIters=4000,
            confidence=0.999,
        )
    except cv2.error:
        return None, None, "none"
    if M is None:
        return None, None, "none"
    H = np.array([[M[0, 0], M[0, 1], M[0, 2]],
                  [M[1, 0], M[1, 1], M[1, 2]],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return H, mask, "similarity"


def _orb():
    return cv2.ORB_create(
        nfeatures=config.FEATURES.n_features,
        scaleFactor=1.2,
        nlevels=8,
        fastThreshold=8,
    )


def match_features(
    orig_gray: np.ndarray,
    cand_gray: np.ndarray,
) -> FeatureMatch:
    """ORB -> knn -> Lowe ratio -> RANSAC homography -> coverage."""
    detector = _orb()
    kp1, des1 = detector.detectAndCompute(_u8(orig_gray), None)
    kp2, des2 = detector.detectAndCompute(_u8(cand_gray), None)

    empty = FeatureMatch(
        keypoints_orig=len(kp1 or []),
        keypoints_cand=len(kp2 or []),
        raw_matches=0,
        good_matches=0,
        inliers=0,
        inlier_ratio=0.0,
        homography=None,
        coverage=None,
        model="none",
    )
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return empty

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE_HAMMING)
    raw = matcher.knnMatch(des1, des2, k=2)
    good = [m[0] for m in raw if len(m) == 2 and m[0].distance < config.FEATURES.ratio * m[1].distance]
    if len(good) < 8:
        empty.raw_matches = int(len(raw))
        empty.good_matches = int(len(good))
        return empty

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # A resized / recompressed / cropped copy of an image differs from the
    # original by a *similarity* transform (rotation + uniform scale + shift),
    # not by a general projective one.  Fitting the full 8-DOF homography was
    # measurably wrong here: on a low-texture scene with 36 inliers it
    # recovered perspective terms of 2.1e-4, i.e. a 22% warp across the frame,
    # which smeared the aligned comparison and turned a clean crop into "76% of
    # blocks differ".  estimateAffinePartial2D constrains the model to 4 DOF and
    # needs only 2 point pairs, so it is both more robust and the right prior.
    H, mask, model = _estimate_similarity(src, dst)
    if H is None:
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, config.FEATURES.ransac_reproj)
        model = "homography"
    inliers = int(mask.sum()) if mask is not None and H is not None else 0

    coverage = None
    if H is not None and inliers >= config.FEATURES.min_inliers:
        coverage = coverage_from_homography(
            H,
            (orig_gray.shape[1], orig_gray.shape[0]),
            (cand_gray.shape[1], cand_gray.shape[0]),
        )

    return FeatureMatch(
        keypoints_orig=len(kp1),
        keypoints_cand=len(kp2),
        raw_matches=int(len(raw)),
        good_matches=int(len(good)),
        inliers=inliers,
        inlier_ratio=float(min(1.0, inliers / max(1, min(len(good), 40)))),
        homography=H if inliers >= config.FEATURES.min_inliers else None,
        coverage=coverage,
        model=model if inliers >= config.FEATURES.min_inliers else "none",
    )


def _u8(gray: np.ndarray) -> np.ndarray:
    if gray.dtype == np.uint8:
        return gray
    return np.clip(gray, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Multi-scale template matching
# --------------------------------------------------------------------------

@dataclass
class TemplateMatch:
    """Best axis-aligned placement found by the template sweep."""

    score: float
    #: (x0, y0, x1, y1) in SCENE pixels.
    box: Optional[Tuple[int, int, int, int]]
    #: ``'original-in-candidate'`` or ``'candidate-in-original'``.
    direction: str
    scale: float
    coverage: Optional[Coverage]
    #: Region of the source image that aligns with the candidate, so the caller
    #: can run a pixel-level comparison on properly aligned content.
    aligned_region: Optional[np.ndarray] = None

    @property
    def confident(self) -> bool:
        return self.score >= config.FEATURES.template_score_confident

    def to_dict(self) -> dict:
        return {
            "score": round(self.score, 4),
            "box": list(self.box) if self.box else None,
            "direction": self.direction,
            "scale": round(self.scale, 4),
            "confident": self.confident,
            "coverage": self.coverage.to_dict() if self.coverage else None,
        }


def _edges(gray: np.ndarray) -> np.ndarray:
    u = _u8(gray)
    u = cv2.GaussianBlur(u, (3, 3), 0)
    return cv2.Canny(u, 50, 200)


def _sweep(
    template_rgb: np.ndarray,
    scene_rgb: np.ndarray,
    work: int,
):
    """Multi-scale template match of ``template`` inside ``scene``.

    Both are downscaled to ``work`` px on the longest side and matched with
    ``TM_CCOEFF_NORMED`` on *mean/std-normalised grayscale*, not on Canny edges.
    Canny was the obvious choice but measured badly: a one-pixel misalignment
    between two binary edge maps collapses the correlation, and the sweep
    topped out near 0.43 even for an exact copy.  Normalised grayscale is
    shift-tolerant enough to peak sharply at the correct scale.

    Scale 1.0 is included (a template exactly the size of the scene yields a
    single response value), which is what an unmodified full-frame copy needs.
    """
    tw0, th0 = template_rgb.shape[1], template_rgb.shape[0]
    sw, sh = scene_rgb.shape[1], scene_rgb.shape[0]

    scene_w, scene_h = fit_size(sw, sh, work)
    scene_small = _norm_gray(resize_rgb(scene_rgb, (scene_w, scene_h)))

    tpl_w, tpl_h = fit_size(tw0, th0, work)
    tpl_small = _norm_gray(resize_rgb(template_rgb, (tpl_w, tpl_h)))

    best_score, best_box = -1.0, None
    for frac in _scales():
        tw = max(8, int(round(tpl_w * frac)))
        th = max(8, int(round(tpl_h * frac)))
        if tw > scene_w or th > scene_h:
            continue
        tpl = cv2.resize(tpl_small, (tw, th), interpolation=cv2.INTER_AREA)
        tpl = _norm_gray_arr(tpl)
        try:
            res = cv2.matchTemplate(scene_small, tpl, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            continue
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_score:
            sx, sy = sw / scene_w, sh / scene_h
            best_score = float(max_val)
            best_box = (
                max_loc[0] * sx,
                max_loc[1] * sy,
                (max_loc[0] + tw) * sx,
                (max_loc[1] + th) * sy,
            )
    return best_score, best_box


def _scales() -> List[float]:
    """Geometric scale ladder from 0.15 to 1.0 inclusive."""
    lo, hi = config.FEATURES.template_scales[0], config.FEATURES.template_scales[-1]
    n = len(config.FEATURES.template_scales) * 2
    return [float(lo * (hi / lo) ** (i / (n - 1))) for i in range(n)]


def _norm_gray(rgb: np.ndarray) -> np.ndarray:
    return _norm_gray_arr(to_gray(rgb))


def _norm_gray_arr(gray: np.ndarray) -> np.ndarray:
    """Zero-mean unit-variance, so TM_CCOEFF_NORMED compares structure only."""
    g = gray.astype(np.float32)
    std = float(g.std())
    if std < 1e-6:
        return np.zeros_like(g, dtype=np.float32)
    return ((g - g.mean()) / std).astype(np.float32)


def locate(
    orig_rgb: np.ndarray,
    cand_rgb: np.ndarray,
    work: int = 256,
) -> TemplateMatch:
    """Bidirectional template sweep.

    Both containment directions are searched and the higher-scoring one wins:

    * *original-in-candidate* -- the candidate contains the whole original
      (borders, collages, a page around the picture).
    * *candidate-in-original* -- the candidate is a crop of the original.

    Only a confident sweep is reported; an unconfident sweep is still returned
    (with its score) so the caller can see how weak the evidence was.
    """
    from .imaging import resize_rgb

    ow, oh = orig_rgb.shape[1], orig_rgb.shape[0]
    cw, ch = cand_rgb.shape[1], cand_rgb.shape[0]

    empty = TemplateMatch(score=-1.0, box=None, direction="none", scale=0.0, coverage=None)

    score_a, box_a = _sweep(orig_rgb, cand_rgb, work)   # original inside candidate
    score_b, box_b = _sweep(cand_rgb, orig_rgb, work)   # candidate inside original

    if box_a is None and box_b is None:
        return empty

    if score_a >= score_b and box_a is not None:
        x0, y0, x1, y1 = box_a
        aligned = _extract(cand_rgb, x0, y0, x1, y1, (ow, oh))
        return TemplateMatch(
            score=score_a,
            box=(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))),
            direction="original-in-candidate",
            scale=float(np.sqrt(((x1 - x0) * (y1 - y0)) / max(1.0, float(ow * oh)))),
            coverage=coverage_from_box(box_a, (ow, oh), (cw, ch)),
            aligned_region=aligned,
        )
    x0, y0, x1, y1 = box_b
    aligned = _extract(orig_rgb, x0, y0, x1, y1, (cw, ch))
    return TemplateMatch(
        score=score_b,
        box=(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))),
        direction="candidate-in-original",
        scale=float(np.sqrt((float(ow * oh)) / max(1.0, (x1 - x0) * (y1 - y0)))),
        coverage=coverage_of_crop(box_b, (ow, oh), (cw, ch)),
        aligned_region=aligned,
    )


def _extract(rgb: np.ndarray, x0: float, y0: float, x1: float, y1: float, size) -> Optional[np.ndarray]:
    h, w = rgb.shape[:2]
    ix0, iy0 = int(max(0, min(w - 2, round(x0)))), int(max(0, min(h - 2, round(y0))))
    ix1, iy1 = int(max(ix0 + 2, min(w, round(x1)))), int(max(iy0 + 2, min(h, round(y1))))
    crop = rgb[iy0:iy1, ix0:ix1]
    if crop.size == 0:
        return None
    return resize_rgb(crop, (int(size[0]), int(size[1])))


def merge_coverage(
    feat: FeatureMatch,
    tpl: TemplateMatch,
) -> Tuple[Optional[Coverage], str]:
    """Pick the coverage estimate to trust, preferring the stronger evidence.

    A trusted homography wins when it has real inliers behind it; otherwise the
    template sweep is used.  When both exist and disagree badly the more
    conservative (lower ``original`` coverage) one is kept, because that is the
    direction that protects against false 1:1 claims.
    """
    fc = feat.coverage if feat.trusted else None
    tc = tpl.coverage if tpl.confident else None
    if fc is not None and tc is not None:
        if abs(fc.original - tc.original) > 0.15:
            return (fc if fc.original <= tc.original else tc), "conservative-agreement"
        return fc, "homography"
    if fc is not None:
        return fc, "homography"
    if tc is not None:
        return tc, "template"
    return None, "none"
