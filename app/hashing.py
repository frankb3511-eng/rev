"""Perceptual and cryptographic hashing (stages 1 and 2).

Four complementary 64-bit perceptual hashes are computed on a 32x32 grayscale
downsample of each image, plus a 64-bin colour histogram used as a cheap
crop-tolerant signal:

===========  ================================  ==========================
hash         transform                         strong against
===========  ================================  ==========================
``ahash``    mean threshold                    broad perturbations, weak
                                               discrimination
``dhash``    horizontal gradient sign          brightness / gamma change
``phash``    32x32 DCT -> 8x8 low-freq median  JPEG, scaling, blur (best
                                               overall)
``whash``    Haar DWT level-1 approximation    heavy JPEG; weak under gamma
===========  ================================  ==========================

Distances are Hamming distances converted to normalised similarity
``1 - dist/64``, which is the ``S_hash`` formulation used by the comparative
study this weighting comes from (see ``app/config.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
from scipy.fft import dct  # shipped with scikit-image; already a dependency

from . import config
from .imaging import LoadedImage, resize_rgb, to_gray


@dataclass
class Hashes:
    """All cheap fingerprints of one image, computed once and reused."""

    sha256: str
    width: int
    height: int
    ahash: np.ndarray          # bool (64,)
    dhash: np.ndarray
    phash: np.ndarray
    whash: np.ndarray
    color_hist: np.ndarray     # float32 (64,), L2-normalised
    luma_mean: float
    luma_std: float

    def to_dict(self) -> dict:
        return {
            "sha256": self.sha256,
            "width": self.width,
            "height": self.height,
            "ahash": _bits_to_hex(self.ahash),
            "dhash": _bits_to_hex(self.dhash),
            "phash": _bits_to_hex(self.phash),
            "whash": _bits_to_hex(self.whash),
            "luma_mean": round(self.luma_mean, 3),
            "luma_std": round(self.luma_std, 3),
        }


# --------------------------------------------------------------------------
# Bit helpers
# --------------------------------------------------------------------------

def _bits_to_hex(bits: np.ndarray) -> str:
    packed = np.packbits(bits.astype(np.uint8))
    return packed.tobytes().hex()


def _hex_to_bits(text: str) -> np.ndarray:
    raw = np.frombuffer(bytes.fromhex(text), dtype=np.uint8)
    return np.unpackbits(raw).astype(bool)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised Hamming similarity in [0, 1]; 1.0 == identical hashes."""
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    return 1.0 - hamming(a, b) / n


# --------------------------------------------------------------------------
# Individual hashes
# --------------------------------------------------------------------------

def _prepare(rgb: np.ndarray) -> np.ndarray:
    """Resize to HASH_WORK_SIZE x HASH_WORK_SIZE and return float32 grayscale."""
    side = config.HASH_WORK_SIZE
    small = resize_rgb(rgb, (side, side))
    return to_gray(small)


def average_hash(gray_small: np.ndarray) -> np.ndarray:
    """aHash: 1 where the pixel is at or above the mean, on an 8x8 thumbnail.

    The reduction to 8x8 is what makes this a 64-bit hash like the other three.
    Thresholding the 32x32 pre-hash thumbnail instead produced a 1024-bit hash,
    which silently changed the meaning of every 64-bit threshold in
    ``app/config.py`` (a 0.93 similarity is ~4.5 bits of 64, not ~72 bits of
    1024) and made aHash far more discriminative than the weighting assumed.
    """
    tiny = _reduce(gray_small, 8, 8)
    return (tiny >= tiny.mean()).ravel()


def difference_hash(gray_small: np.ndarray) -> np.ndarray:
    """dHash: 1 where the pixel is brighter than its left neighbour.

    Operates on a 9x8 window of the 32x32 downsample so the output is 64 bits
    while still using a real gradient rather than a subsample.
    """
    w = _reduce(gray_small, 9, 8)
    return (w[:, 1:] > w[:, :-1]).ravel()


def dct_hash(gray_small: np.ndarray) -> np.ndarray:
    """pHash: sign of the low-frequency DCT block against its own median.

    The DC coefficient is excluded from the median, exactly as the ``phash_org``
    correction does -- including it biases the threshold because the DC term is
    an order of magnitude larger than the rest.
    """
    block = _dct2(gray_small)[:8, :8].copy()
    ac = block.ravel()[1:]
    median = np.median(ac)
    return (block.ravel() > median)


def wavelet_hash(gray_small: np.ndarray) -> np.ndarray:
    """wHash: Haar level-1 approximation coefficients, median-thresholded."""
    a = gray_small
    for _ in range(5):                       # 32 -> 16 -> 8 -> 4 -> 2 -> 1
        even_r, odd_r = a[0::2, :], a[1::2, :]
        low_r = (even_r + odd_r) * 0.5
        even_c, odd_c = low_r[:, 0::2], low_r[:, 1::2]
        a = (even_c + odd_c) * 0.5
        if a.shape[0] <= 8 and a.shape[1] <= 8:
            break
    flat = a.ravel()
    if flat.size < 64:
        flat = np.resize(flat, 64)
    else:
        flat = flat[:64]
    return flat > np.median(flat)


def _reduce(gray: np.ndarray, w: int, h: int) -> np.ndarray:
    from PIL import Image
    im = Image.fromarray(gray.astype(np.float32), mode="F").resize((w, h), Image.LANCZOS)
    return np.asarray(im, dtype=np.float32)


def _dct2(gray: np.ndarray) -> np.ndarray:
    return dct(dct(gray, axis=0, norm="ortho"), axis=1, norm="ortho")


# --------------------------------------------------------------------------
# Colour histogram (cheap, crop tolerant)
# --------------------------------------------------------------------------

def color_histogram(rgb: np.ndarray, bins: int = 4) -> np.ndarray:
    """L2-normalised bins-per-channel RGB histogram (4^3 = 64 bins)."""
    small = resize_rgb(rgb, (32, 32)).astype(np.int32)
    idx = (
        (small[:, :, 0] >> (8 - 2)).astype(np.int32) * (bins * bins)
        + (small[:, :, 1] >> (8 - 2)).astype(np.int32) * bins
        + (small[:, :, 2] >> (8 - 2)).astype(np.int32)
    ).ravel()
    hist = np.bincount(idx, minlength=bins ** 3).astype(np.float32)
    norm = float(np.linalg.norm(hist))
    return hist / norm if norm > 0 else hist


def histogram_cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), 0.0, 1.0))


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------

def compute(image: LoadedImage) -> Hashes:
    gray_small = _prepare(image.rgb)
    hist = color_histogram(image.rgb, config.EMBEDDING.colour_bins)
    return Hashes(
        sha256=image.sha256,
        width=image.width,
        height=image.height,
        ahash=average_hash(gray_small),
        dhash=difference_hash(gray_small),
        phash=dct_hash(gray_small),
        whash=wavelet_hash(gray_small),
        color_hist=hist,
        luma_mean=float(to_gray(image.rgb).mean()),
        luma_std=float(to_gray(image.rgb).std()),
    )


@dataclass
class HashComparison:
    """Per-hash similarity plus the fused value used by the classifier."""

    per_hash: Dict[str, float]
    fused: float
    best: float
    worst: float
    best_name: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["per_hash"] = {k: round(v, 4) for k, v in self.per_hash.items()}
        d["fused"] = round(self.fused, 4)
        d["best"] = round(self.best, 4)
        d["worst"] = round(self.worst, 4)
        return d


def compare(a: Hashes, b: Hashes) -> HashComparison:
    """Fuse the four hash similarities with the configured weights."""
    weights = config.normalised_hash_weights()
    per: Dict[str, float] = {
        "ahash": similarity(a.ahash, b.ahash),
        "dhash": similarity(a.dhash, b.dhash),
        "phash": similarity(a.phash, b.phash),
        "whash": similarity(a.whash, b.whash),
    }
    fused = sum(per[k] * weights[k] for k in weights)
    best_name = max(per, key=lambda k: per[k])
    return HashComparison(
        per_hash=per,
        fused=float(fused),
        best=float(per[best_name]),
        worst=float(min(per.values())),
        best_name=best_name,
    )


def hashes_from_dict(d: dict) -> Optional[Hashes]:
    """Rebuild a ``Hashes`` from ``to_dict()`` output (used by cached results)."""
    try:
        return Hashes(
            sha256=d["sha256"],
            width=d["width"],
            height=d["height"],
            ahash=_hex_to_bits(d["ahash"]),
            dhash=_hex_to_bits(d["dhash"]),
            phash=_hex_to_bits(d["phash"]),
            whash=_hex_to_bits(d["whash"]),
            color_hist=np.zeros(64, dtype=np.float32),
            luma_mean=d.get("luma_mean", 0.0),
            luma_std=d.get("luma_std", 0.0),
        )
    except Exception:
        return None
