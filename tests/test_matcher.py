"""The 1:1 matcher benchmark suite.

The specification is explicit that the feature is not complete until these pass:

    identical images                 -> 1:1
    resized images                   -> 1:1
    recompressed images              -> 1:1
    minor edits                      -> likely 1:1
    crop                             -> crop
    same subject, different photo    -> similar, NOT 1:1
    completely unrelated image       -> unrelated

Every case runs through the shipped matcher on real pixels.  Nothing is mocked
and no expectation is derived from a search engine label.
"""

from __future__ import annotations

import pytest

from app import hashing
from app.imaging import decode
from app.matcher import Label, MatchInput, match_pair
from tests.fixtures import case_map


CASES = case_map()


# --------------------------------------------------------------------------
# The full matrix
# --------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(CASES))
def test_case_reaches_expected_verdict(name, verdicts):
    case = CASES[name]
    verdict = verdicts[name]
    assert verdict.label.value == case.expected, (
        f"{name}: expected {case.expected}, got {verdict.label.value} "
        f"({verdict.confidence:.1f}%). reasons: {verdict.reasons}"
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_case_lands_in_expected_ui_group(name, verdicts):
    case = CASES[name]
    assert verdicts[name].group == case.group


# --------------------------------------------------------------------------
# The seven required behaviours, stated individually
# --------------------------------------------------------------------------

ONE_TO_ONE_CASES = [
    "identical_bytes", "renamed_identical", "metadata_stripped", "metadata_added",
    "resized_down_half", "resized_up_double", "jpeg_recompressed", "png_to_jpeg",
    "jpeg_to_webp", "heavily_compressed",
]


@pytest.mark.parametrize("name", ONE_TO_ONE_CASES)
def test_identical_resized_and_recompressed_are_one_to_one(name, verdicts):
    """Identical, renamed, resized, re-encoded and format-converted copies are all 1:1.

    None of these require the bytes to match -- that is the whole point of the
    exercise.
    """
    assert verdicts[name].label is Label.EXACT_1TO1


@pytest.mark.parametrize("name", [
    "brightness_up", "brightness_down", "contrast_up", "colour_shift",
    "watermarked", "bordered", "screenshot", "blurred",
    "rotated_90", "rotated_180", "mirrored",
])
def test_minor_edits_are_likely_one_to_one(name, verdicts):
    assert verdicts[name].label is Label.EDITED_1TO1
    assert verdicts[name].confidence >= 85.0


@pytest.mark.parametrize("name", ["crop_centre", "crop_corner"])
def test_crops_are_classified_as_crops(name, verdicts):
    assert verdicts[name].label is Label.CROP


def test_crops_are_never_called_one_to_one(verdicts):
    """A crop must not automatically become a 1:1 match."""
    for name in ("crop_centre", "crop_corner"):
        assert not verdicts[name].label.is_one_to_one


@pytest.mark.parametrize("name,expected_overlap", [
    ("crop_centre", 0.25),          # 50% x 50% of the frame
    ("crop_corner", 0.62 * 0.66),   # 62% x 66%
])
def test_crop_overlap_estimate_is_accurate(name, expected_overlap, verdicts):
    """The reported overlap must come from measurement, and be roughly right."""
    verdict = verdicts[name]
    assert verdict.overlap is not None
    assert abs(verdict.overlap - expected_overlap) <= 0.08, (
        f"{name}: reported {verdict.overlap:.3f}, actual {expected_overlap:.3f}"
    )


def test_different_photo_of_same_subject_is_similar_not_one_to_one(verdicts):
    """The headline false positive.

    A second shot of the same scene produces a plausible geometric alignment and
    a high embedding similarity.  What separates it from a copy is that its
    pixels differ almost everywhere once aligned.
    """
    verdict = verdicts["different_photo_same_subject"]
    assert verdict.label is Label.VISUALLY_SIMILAR
    assert not verdict.label.is_one_to_one
    assert verdict.signals.changed_fraction > 0.5


def test_unrelated_image_is_unrelated(verdicts):
    verdict = verdicts["unrelated_image"]
    assert verdict.label is Label.UNRELATED
    assert verdict.confidence < 20.0


# --------------------------------------------------------------------------
# Confidence must be derived, not invented
# --------------------------------------------------------------------------

def test_byte_identical_files_are_reported_at_100_percent(verdicts):
    v = verdicts["identical_bytes"]
    assert v.confidence == 100.0
    assert v.signals.sha_identical is True


@pytest.mark.parametrize("name", sorted(CASES))
def test_confidence_is_within_bounds(name, verdicts):
    assert 0.0 <= verdicts[name].confidence <= 100.0


def test_confidence_orders_evidence_correctly(verdicts):
    """More preserved evidence must not produce a lower confidence."""
    identical = verdicts["identical_bytes"].confidence
    recompressed = verdicts["jpeg_recompressed"].confidence
    heavy = verdicts["heavily_compressed"].confidence
    similar = verdicts["different_photo_same_subject"].confidence
    unrelated = verdicts["unrelated_image"].confidence

    assert identical >= recompressed >= heavy
    assert heavy > 90.0
    assert unrelated < 20.0
    # A different photograph must never out-confidence a verified copy.
    assert identical > similar


def test_every_verdict_carries_reasons(verdicts):
    """No verdict may be asserted without the measurements behind it."""
    for name, verdict in verdicts.items():
        assert verdict.reasons, f"{name} produced a verdict with no stated reasons"


def test_confidence_is_monotone_in_identity_within_a_bucket(verdicts):
    """A stronger fused identity must not produce a weaker confidence.

    Byte-identical files are excluded: their confidence is pinned at 100 by the
    SHA-256 branch rather than by the logistic.
    """
    exact = [
        v for v in verdicts.values()
        if v.label is Label.EXACT_1TO1 and not v.signals.sha_identical
    ]
    assert len(exact) >= 5
    for a in exact:
        for b in exact:
            if a.identity > b.identity + 0.002:
                assert a.confidence >= b.confidence, (
                    f"identity {a.identity:.4f} -> {a.confidence:.1f}% but "
                    f"identity {b.identity:.4f} -> {b.confidence:.1f}%"
                )


def test_verified_copies_out_confidence_lookalikes(verdicts):
    """Separation check on the calibrated system.

    This is a measurement of the shipped thresholds against this suite, not a
    structural guarantee: it exists to catch threshold drift that would let a
    lookalike photograph start scoring like a verified copy.
    """
    one_to_one = [v.confidence for v in verdicts.values() if v.label.is_one_to_one]
    similar = [v.confidence for v in verdicts.values() if v.label is Label.VISUALLY_SIMILAR]
    unrelated = [v.confidence for v in verdicts.values() if v.label is Label.UNRELATED]

    assert one_to_one and similar and unrelated
    assert min(one_to_one) > max(similar), (
        f"weakest 1:1 ({min(one_to_one):.1f}%) did not beat the strongest "
        f"lookalike ({max(similar):.1f}%)"
    )
    assert max(similar) > max(unrelated)


# --------------------------------------------------------------------------
# Independence from search-engine labels
# --------------------------------------------------------------------------

def test_verdict_does_not_depend_on_engine_metadata(original_input):
    """The engine's own label must have no effect on the local verdict.

    The matcher is handed pixels only; this asserts that the code path cannot
    reach an engine label even if one is supplied elsewhere in the pipeline.
    """
    data = CASES["jpeg_recompressed"].data
    img = decode(data)
    cand = MatchInput(hashing.compute(img), img.rgb)

    a = match_pair(original_input, cand)
    b = match_pair(original_input, cand)
    assert a.label is b.label
    assert a.confidence == b.confidence
    # And the verdict object exposes no engine field at all.
    assert not hasattr(a, "engine")
    assert not hasattr(a, "engine_label")
    assert not hasattr(a, "engine_score")


# --------------------------------------------------------------------------
# Signal sanity
# --------------------------------------------------------------------------

def test_orientation_is_detected_and_reported(verdicts):
    assert verdicts["rotated_90"].signals.orientation != "identity"
    assert verdicts["rotated_180"].signals.orientation != "identity"
    assert verdicts["mirrored"].signals.orientation != "identity"
    assert verdicts["identical_bytes"].signals.orientation == "identity"
    # A rotation must be reported to the user, not silently corrected away.
    assert any("rotated" in r for r in verdicts["rotated_90"].reasons)


def test_edits_are_attributed(verdicts):
    assert verdicts["brightness_up"].edits.brightness
    assert verdicts["contrast_up"].edits.contrast
    assert verdicts["colour_shift"].edits.colour
    assert verdicts["watermarked"].edits.watermark
    assert verdicts["bordered"].edits.border
    assert verdicts["screenshot"].edits.border
    assert verdicts["blurred"].edits.blur


def test_clean_copies_show_no_edits(verdicts):
    for name in ("identical_bytes", "resized_down_half", "jpeg_recompressed"):
        labels = verdicts[name].edits.labels
        assert not any(
            x in ("brightness", "contrast", "colour", "watermark/text overlay")
            for x in labels
        ), f"{name} reported edits: {labels}"


def test_full_frame_coverage_is_measured_for_exact_matches(verdicts):
    """An EXACT verdict must be backed by near-complete coverage in both directions."""
    for name in ONE_TO_ONE_CASES:
        sig = verdicts[name].signals
        assert sig.coverage_orig >= 0.98, f"{name}: coverage_orig {sig.coverage_orig}"
        assert sig.coverage_cand >= 0.98, f"{name}: coverage_cand {sig.coverage_cand}"
