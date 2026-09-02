"""Geometric helpers: orientation search and convex-polygon coverage maths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: The eight rigid orientations we search over.  ``identity`` is tried first so
#: that an unmodified copy costs only one hash comparison.
ORIENTATIONS: Tuple[Tuple[str, int, bool], ...] = (
    ("identity", 0, False),
    ("rot90", 90, False),
    ("rot180", 180, False),
    ("rot270", 270, False),
    ("mirror-h", 0, True),
    ("mirror-h-rot90", 90, True),
    ("mirror-h-rot180", 180, True),
    ("mirror-h-rot270", 270, True),
)


def apply_orientation(rgb: np.ndarray, name: str) -> np.ndarray:
    """Return ``rgb`` rotated/mirrored according to an ``ORIENTATIONS`` name."""
    if name == "identity":
        return rgb
    rot = {"rot90": cv2.ROTATE_90_CLOCKWISE,
           "rot180": cv2.ROTATE_180,
           "rot270": cv2.ROTATE_90_COUNTERCLOCKWISE}
    out = rgb
    for part in name.split("-"):
        if part in rot:
            out = cv2.rotate(out, rot[part])
        elif part.startswith("mirror"):
            out = cv2.flip(out, 1)
    return out


def orientation_names() -> List[str]:
    return [name for name, _, _ in ORIENTATIONS]


# --------------------------------------------------------------------------
# Polygon coverage
# --------------------------------------------------------------------------

@dataclass
class Coverage:
    """How much of each image the other one explains."""

    #: Share of the ORIGINAL that is present inside the candidate frame.
    original: float
    #: Share of the CANDIDATE that is explained by original content.
    candidate: float
    #: Bounding box of the original's footprint in candidate pixels.
    box: Optional[Tuple[int, int, int, int]]
    #: Scale factor applied to the original by the homography.
    scale: float

    def to_dict(self) -> dict:
        return {
            "original": round(self.original, 4),
            "candidate": round(self.candidate, 4),
            "box": list(self.box) if self.box else None,
            "scale": round(self.scale, 4),
        }


def _poly_area(poly: np.ndarray) -> float:
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _intersect_area(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """Area of intersection of two convex polygons (shoelace fallback)."""
    pa = np.ascontiguousarray(a.reshape(-1, 1, 2), dtype=np.float32)
    pb = np.ascontiguousarray(b.reshape(-1, 1, 2), dtype=np.float32)
    try:
        area, _ = cv2.intersectConvexConvex(pa, pb, handleNested=False)
        if area is not None and float(area) > 0:
            return float(area)
    except Exception:                              # pragma: no cover - old OpenCV
        pass
    return _sutherland_hodgman_area(pa.reshape(-1, 2), pb.reshape(-1, 2))


def _sutherland_hodgman_area(subject: np.ndarray, clip: np.ndarray) -> Optional[float]:
    """Clip ``subject`` by convex ``clip``; returns the resulting area."""
    output = [tuple(p) for p in subject]
    if len(output) < 3 or len(clip) < 3:
        return None
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        if not output:
            break
        inside = lambda p: (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0
        result: List[Tuple[float, float]] = []
        for j in range(len(output)):
            cur = output[j]
            prev = output[j - 1]
            cur_in, prev_in = inside(cur), inside(prev)
            if cur_in:
                if not prev_in:
                    result.append(_intersect_point(prev, cur, a, b))
                result.append(cur)
            elif prev_in:
                result.append(_intersect_point(prev, cur, a, b))
        output = result
    if len(output) < 3:
        return None
    arr = np.asarray(output, dtype=np.float64)
    return _poly_area(arr)


def _intersect_point(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-12:
        return (float(x1), float(y1))
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def coverage_from_homography(
    homography: np.ndarray,
    orig_size: Tuple[int, int],
    cand_size: Tuple[int, int],
) -> Optional[Coverage]:
    """Coverage of original vs candidate implied by a homography.

    ``homography`` maps ORIGINAL pixel coordinates to CANDIDATE pixel
    coordinates.  The original's four corners are pushed through it; the
    resulting quadrilateral is intersected with the candidate's frame.
    """
    ow, oh = orig_size
    cw, ch = cand_size
    corners = np.float32([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)

    if not np.all(np.isfinite(warped)):
        return None

    warped_area = _poly_area(warped)
    cand_rect = np.float32([[0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]])
    cand_area = float((cw - 1) * (ch - 1))
    orig_area = float((ow - 1) * (oh - 1))
    if warped_area <= 0 or cand_area <= 0 or orig_area <= 0:
        return None

    overlap = _intersect_area(warped, cand_rect)
    if overlap is None:
        return None
    overlap = float(max(0.0, min(overlap, min(warped_area, cand_area))))

    xs, ys = warped[:, 0], warped[:, 1]
    box = (
        int(max(0, round(xs.min()))),
        int(max(0, round(ys.min()))),
        int(min(cw - 1, round(xs.max()))),
        int(min(ch - 1, round(ys.max()))),
    )
    scale = float(np.sqrt(warped_area / orig_area))
    return Coverage(
        original=float(np.clip(overlap / warped_area, 0.0, 1.0)),
        candidate=float(np.clip(overlap / cand_area, 0.0, 1.0)),
        box=box,
        scale=scale,
    )


def coverage_from_box(
    box: Sequence[float],
    orig_size: Tuple[int, int],
    cand_size: Tuple[int, int],
) -> Coverage:
    """Coverage when the ORIGINAL was located inside the CANDIDATE.

    ``box`` is the placement of the (scaled) original in candidate pixels --
    i.e. the "candidate contains the original" direction, which is how borders,
    collages and "original is a crop of candidate" cases present themselves.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    cw, ch = cand_size
    ow, oh = orig_size
    ix0, iy0 = max(0.0, x0), max(0.0, y0)
    ix1, iy1 = min(float(cw), x1), min(float(ch), y1)
    inside = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    box_area = max(1e-6, (x1 - x0) * (y1 - y0))
    cand_area = max(1e-6, float(cw * ch))
    return Coverage(
        original=float(np.clip(inside / box_area, 0.0, 1.0)),
        candidate=float(np.clip(inside / cand_area, 0.0, 1.0)),
        box=(int(round(ix0)), int(round(iy0)), int(round(ix1)), int(round(iy1))),
        scale=float(np.sqrt(box_area / max(1e-6, float(ow * oh)))),
    )


def coverage_of_crop(
    box: Sequence[float],
    orig_size: Tuple[int, int],
    cand_size: Tuple[int, int],
) -> Coverage:
    """Coverage when the CANDIDATE was located inside the ORIGINAL.

    ``box`` is the placement of the (scaled) candidate in ORIGINAL pixels --
    the "candidate is a crop of the original" direction.  By construction the
    whole candidate is explained by original content, so
    ``Coverage.candidate`` is 1.0 and ``Coverage.original`` is the share of the
    original the crop shows -- the "Estimated overlap" the UI reports.
    """
    x0, y0, x1, y1 = (float(v) for v in box)
    ow, oh = orig_size
    cw, ch = cand_size
    orig_area = max(1e-6, float(ow * oh))
    box_area = max(1e-6, (x1 - x0) * (y1 - y0))
    return Coverage(
        original=float(np.clip(box_area / orig_area, 0.0, 1.0)),
        candidate=1.0,
        box=(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))),
        scale=float(np.sqrt(orig_area / box_area)),
    )
