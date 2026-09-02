"""Unit tests for the hashing, geometry and edit-attribution primitives."""

from __future__ import annotations

import numpy as np
import pytest

from app import hashing
from app.editdetect import detect_border
from app.geometry import (
    apply_orientation,
    coverage_from_box,
    coverage_from_homography,
    coverage_of_crop,
)
from app.imaging import UnsupportedImage, aspect_similarity, decode, fit_size, to_gray
from tests.fixtures import (
    base_scene,
    border as add_border,
    brightness,
    colour_shift,
    contrast,
    crop_region,
    mirror,
    resized,
    rotate,
    to_bytes,
)


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------

def test_hash_is_64_bits():
    h = hashing.compute(decode(to_bytes(base_scene(42), "PNG")))
    for arr in (h.ahash, h.dhash, h.phash, h.whash):
        assert arr.shape == (64,)
        assert arr.dtype == bool


def test_identical_images_hash_identically():
    data = to_bytes(base_scene(42), "PNG")
    a = hashing.compute(decode(data))
    b = hashing.compute(decode(data))
    assert hashing.hamming(a.phash, b.phash) == 0
    assert hashing.similarity(a.phash, b.phash) == 1.0


def test_resize_barely_moves_the_hash():
    """Resize invariance is the property the whole design rests on."""
    a = hashing.compute(decode(to_bytes(base_scene(42), "PNG")))
    b = hashing.compute(decode(to_bytes(resized(base_scene(42), 0.4), "JPEG", 85)))
    for name in ("ahash", "dhash", "phash", "whash"):
        assert hashing.similarity(getattr(a, name), getattr(b, name)) >= 0.90, name


def test_unrelated_images_hash_far_apart():
    from tests.fixtures import unrelated_scene

    a = hashing.compute(decode(to_bytes(base_scene(42), "PNG")))
    b = hashing.compute(decode(to_bytes(unrelated_scene(31337), "PNG")))
    cmp = hashing.compare(a, b)
    assert cmp.fused < config_unrelated_threshold()


def config_unrelated_threshold() -> float:
    from app import config
    return config.HASH_NEAR_SIM - 0.15


def test_similarity_is_symmetric_and_bounded():
    a = hashing.compute(decode(to_bytes(base_scene(42), "PNG")))
    b = hashing.compute(decode(to_bytes(brightness(base_scene(42), 40), "JPEG", 88)))
    assert hashing.similarity(a.phash, b.phash) == hashing.similarity(b.phash, a.phash)
    assert 0.0 <= hashing.similarity(a.phash, b.phash) <= 1.0


def test_hash_roundtrips_through_hex():
    a = hashing.compute(decode(to_bytes(base_scene(42), "PNG")))
    restored = hashing.hashes_from_dict(a.to_dict())
    assert np.array_equal(restored.phash, a.phash)
    assert np.array_equal(restored.dhash, a.dhash)


# --------------------------------------------------------------------------
# Imaging helpers
# --------------------------------------------------------------------------

def test_fit_size_preserves_aspect():
    assert fit_size(1920, 1080, 512) == (512, 288)
    assert fit_size(1080, 1920, 512) == (288, 512)
    assert fit_size(300, 200, 512) == (300, 200)   # never upscales


def test_aspect_similarity():
    assert aspect_similarity(1.5, 1.5) == pytest.approx(1.0)
    assert aspect_similarity(1.0, 2.0) == pytest.approx(0.5)
    assert aspect_similarity(0.0, 1.0) == 0.0


def test_to_gray_uses_rec601_weights():
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255                                  # pure red
    assert to_gray(rgb).mean() == pytest.approx(0.299 * 255, abs=0.5)


def test_decode_rejects_garbage():
    with pytest.raises(UnsupportedImage):
        decode(b"not an image at all")
    with pytest.raises(UnsupportedImage):
        decode(b"")


def test_decode_applies_exif_orientation():
    from tests.fixtures import exif_rotated_bytes

    img = decode(exif_rotated_bytes(base_scene(42)))
    assert img.had_exif_orientation is True
    # Display size matches the upright original, not the rotated storage.
    assert (img.width, img.height) == (640, 480)


# --------------------------------------------------------------------------
# Orientation
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["identity", "rot90", "rot180", "rot270", "mirror-h"])
def test_apply_orientation_preserves_pixel_count(name):
    arr = base_scene(42)
    out = apply_orientation(arr, name)
    assert out.shape[2] == 3
    assert out.sum() != 0
    if name in ("identity", "rot180", "mirror-h"):
        # A 180 rotation and a mirror both preserve the frame dimensions.
        assert out.shape[:2] == arr.shape[:2]
    else:
        assert out.shape[:2] == (arr.shape[1], arr.shape[0])


def test_rotate_then_inverse_returns_the_original():
    arr = base_scene(42)
    assert np.array_equal(apply_orientation(rotate(arr, 90), "rot270"), rotate(rotate(arr, 90), 270))


def test_mirror_is_self_inverse():
    arr = base_scene(42)
    assert np.array_equal(apply_orientation(mirror(arr), "mirror-h"), arr)


# --------------------------------------------------------------------------
# Coverage geometry
# --------------------------------------------------------------------------

def test_coverage_of_crop_reports_the_overlap_of_the_original():
    # A 320x240 crop taken from a 640x480 original: 25% of the original.
    cov = coverage_of_crop((160, 120, 480, 360), (640, 480), (320, 240))
    assert cov.original == pytest.approx(0.25, abs=0.01)
    assert cov.candidate == 1.0


def test_coverage_from_box_handles_a_border():
    # A 640x480 original sitting inside a 740x580 candidate.
    cov = coverage_from_box((50, 50, 690, 530), (640, 480), (740, 580))
    assert cov.original == pytest.approx(1.0, abs=0.01)
    assert cov.candidate < 0.9


def test_coverage_from_identity_homography_is_total():
    H = np.eye(3)
    cov = coverage_from_homography(H, (640, 480), (640, 480))
    assert cov.original == pytest.approx(1.0, abs=0.01)
    assert cov.candidate == pytest.approx(1.0, abs=0.01)


def test_coverage_from_scaling_homography_is_total():
    # Pure 2x upscale: still a full-frame match.
    H = np.array([[2.0, 0, 0], [0, 2.0, 0], [0, 0, 1]])
    cov = coverage_from_homography(H, (640, 480), (1280, 960))
    assert cov.original == pytest.approx(1.0, abs=0.02)
    assert cov.candidate == pytest.approx(1.0, abs=0.02)


def test_coverage_from_partial_homography_reports_the_fraction():
    # The original mapped to the left half of a same-sized candidate.
    H = np.array([[0.5, 0, 0], [0, 0.5, 0], [0, 0, 1]])
    cov = coverage_from_homography(H, (640, 480), (640, 480))
    assert cov.candidate == pytest.approx(0.25, abs=0.02)


# --------------------------------------------------------------------------
# Border detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("frac,expected", [(0.06, True), (0.03, True), (0.0, False)])
def test_border_detection(frac, expected):
    orig = base_scene(42)
    cand = add_border(orig, frac) if frac else orig
    found, area = detect_border(orig, cand)
    assert found is expected
    if expected:
        assert area > 0.01


def test_screenshot_bars_are_detected_as_a_border():
    from tests.fixtures import screenshot

    orig = base_scene(42)
    found, area = detect_border(orig, screenshot(orig))
    assert found is True
    assert area > 0.1


def test_plain_image_has_no_border():
    orig = base_scene(42)
    found, area = detect_border(orig, resized(orig, 0.8))
    assert found is False
    assert area == 0.0
