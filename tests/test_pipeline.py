"""Tests for the staged pipeline: filtering, staging, and cross-engine merge."""

from __future__ import annotations

import base64

import pytest

from app import demo, hashing
from app.imaging import decode, encode_jpeg, resize_rgb
from app.matcher import GROUP_ORDER, Label
from app.pipeline import (
    RawResult,
    VerificationPipeline,
    canonical_url,
    stage1_dedupe,
    stage3_filter,
)
from app.search import make_downloader
from tests.fixtures import base_scene, crop_region, unrelated_scene, to_bytes


# --------------------------------------------------------------------------
# A small, fast fixture set
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def small_scene():
    return resize_rgb(base_scene(42), (320, 240))


def _data_url(payload: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(payload).decode("ascii")


def _pipeline(image, results, progress=None):
    return VerificationPipeline(
        original=image,
        raw_results=results,
        downloader=make_downloader(),
        progress=progress,
        max_workers=4,
    )


# --------------------------------------------------------------------------
# Stage 1
# --------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,same", [
    ("https://x.com/a.jpg", "https://x.com/a.jpg", True),
    ("https://x.com/a.jpg", "https://x.com/a.jpg?utm_source=twitter", True),
    ("https://x.com/a.jpg", "https://x.com/a.jpg#fragment", True),
    ("HTTPS://X.COM/a.jpg", "https://x.com/a.jpg", True),
    ("https://x.com/a.jpg", "https://x.com/b.jpg", False),
    ("https://x.com/a.jpg", "https://y.com/a.jpg", False),
])
def test_canonical_url(a, b, same):
    assert (canonical_url(a) == canonical_url(b)) is same


def test_stage1_collapses_duplicate_urls():
    results = [
        RawResult(url="https://x.com/a.jpg?ref=1", engine="google_lens"),
        RawResult(url="https://x.com/a.jpg?ref=2", engine="bing"),
        RawResult(url="https://y.com/b.jpg", engine="yandex"),
    ]
    uniques, aliases = stage1_dedupe(results)
    assert len(uniques) == 2
    assert len(aliases) == 1
    assert aliases[0].engine == "bing"


# --------------------------------------------------------------------------
# Stage 3
# --------------------------------------------------------------------------

def _stub(cid, rgb):
    class C:
        def __init__(self):
            self.cid = cid
            self.error = None
            self.image = decode(to_bytes(rgb, "JPEG", 90))
            self.hashes = hashing.compute(self.image)

        @property
        def ok(self):
            return True
    return C()


def test_stage3_hash_gate_separates_copies_from_junk(small_scene):
    """The cheap gate must rank a copy above unrelated content.

    The top-N safety net promotes everything when there are few candidates --
    that is deliberate, it is what stops crops being discarded -- so this
    asserts the gate's own measurement rather than the final keep flag.
    """
    from app import config

    original = decode(to_bytes(small_scene, "PNG"))
    stub = type("O", (), {"hashes": hashing.compute(original)})()

    _kept, decisions = stage3_filter(
        stub, [_stub("copy", small_scene), _stub("junk", unrelated_scene(555))]
    )
    by_cid = {d.cid: d for d in decisions}
    assert by_cid["copy"].hash_sim >= config.CANDIDATE_FILTER.keep_hash_sim
    assert by_cid["junk"].hash_sim < config.CANDIDATE_FILTER.keep_hash_sim
    assert by_cid["copy"].hash_sim > by_cid["junk"].hash_sim


def test_stage3_drops_junk_once_the_safety_net_is_full(small_scene):
    """With more candidates than keep_top_n, the gate's rejections are real."""
    from app import config

    original = decode(to_bytes(small_scene, "PNG"))
    stub = type("O", (), {"hashes": hashing.compute(original)})()
    n = config.CANDIDATE_FILTER.keep_top_n + 6

    cands = [_stub("copy", small_scene)]
    cands += [_stub(f"junk{i}", unrelated_scene(900 + i)) for i in range(n)]
    kept, _decisions = stage3_filter(stub, cands)

    kept_ids = {c.cid for c in kept}
    assert "copy" in kept_ids
    assert len(kept) < len(cands), "the gate rejected nothing at all"
    assert len(kept) <= config.CANDIDATE_FILTER.keep_top_n + 1


# --------------------------------------------------------------------------
# Full pipeline on the demo result set
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipeline_run(small_scene):
    image = decode(to_bytes(small_scene, "PNG"))
    results = demo.synthesize_results(image, include_unrelated=2)
    seen = []
    report, by_cid = _pipeline(image, results, progress=lambda s, d: seen.append(s)).run()
    return report, by_cid, seen


def test_pipeline_completes_and_reports_counts(pipeline_run):
    report, by_cid, seen = pipeline_run
    c = report.counts
    assert c["raw_results"] > 0
    assert c["downloaded"] > 0
    assert c["compared"] > 0
    assert c["merged_cards"] > 0
    assert c["one_to_one"] >= 1
    assert c["crops"] >= 1
    assert c["similar"] >= 1
    assert "original" in by_cid


def test_progress_statuses_are_emitted_in_order(pipeline_run):
    _report, _by_cid, seen = pipeline_run
    assert seen[0].startswith("Deduplicating")
    assert any(s.startswith("Checking") for s in seen)
    assert any("potential matches" in s for s in seen)
    assert any(s == "Verifying..." for s in seen)
    assert seen[-1] == "Complete"


def test_one_to_one_matches_are_grouped_first(pipeline_run):
    report, _by_cid, _seen = pipeline_run
    order = [g["group"] for g in report.groups]
    assert order == sorted(order, key=GROUP_ORDER.index), f"groups out of order: {order}"
    assert order[0] == "exact"


def test_group_titles_match_the_specification(pipeline_run):
    report, _by_cid, _seen = pipeline_run
    titles = {g["group"]: g["title"] for g in report.groups}
    assert titles.get("exact") == "EXACT 1:1 MATCHES"
    assert titles.get("edited_resized") == "EDITED / RESIZED 1:1 MATCHES"
    if "crops" in titles:
        assert titles["crops"] == "CROPS"
    if "similar" in titles:
        assert titles["similar"] == "SIMILAR IMAGES"
    if "unrelated" in titles:
        assert titles["unrelated"] == "UNRELATED / LOW CONFIDENCE"


def test_every_result_carries_confidence_and_provenance(pipeline_run):
    report, _by_cid, _seen = pipeline_run
    for r in report.results:
        assert 0.0 <= r.confidence <= 100.0
        assert r.sources, f"{r.key} has no sources"
        assert r.engines, f"{r.key} has no engines"
        assert r.representative_cid


def test_lookalike_is_not_promoted_to_one_to_one(pipeline_run):
    """The demo set contains a deliberately different photograph."""
    report, _by_cid, _seen = pipeline_run
    for r in report.results:
        titles = " ".join(s.title for s in r.sources)
        if "different photograph" in titles:
            assert not r.label.is_one_to_one, (
                f"lookalike was classified {r.label.value} at {r.confidence:.1f}%"
            )


# --------------------------------------------------------------------------
# Stage 6 -- merging
# --------------------------------------------------------------------------

def test_same_content_from_two_engines_becomes_one_card(small_scene):
    """Two engines returning the same bytes must produce a single card."""
    image = decode(to_bytes(small_scene, "PNG"))
    payload = encode_jpeg(small_scene, 90)
    results = [
        RawResult(url=_data_url(payload), engine="google_lens", engine_label="Exact match",
                  page_url="https://a.example/1"),
        RawResult(url=_data_url(payload), engine="bing", engine_label="Visually similar",
                  page_url="https://b.example/1"),
        RawResult(url=_data_url(payload), engine="yandex", engine_label="Same image",
                  page_url="https://c.example/1"),
    ]
    report, _by_cid = _pipeline(image, results).run()

    assert report.counts["merged_cards"] == 1, "identical content produced duplicate cards"
    card = report.results[0]
    assert set(card.engines) == {"google_lens", "bing", "yandex"}
    assert len(card.pages) == 3
    assert card.label is Label.EXACT_1TO1


def test_duplicate_urls_do_not_create_duplicate_cards(small_scene):
    image = decode(to_bytes(small_scene, "PNG"))
    payload = encode_jpeg(small_scene, 88)
    url = _data_url(payload)
    results = [RawResult(url=url, engine="google_lens", page_url="https://a.example/1")]
    results += [RawResult(url=url, engine="tineye", page_url=f"https://b.example/{i}")
                for i in range(4)]
    report, _by_cid = _pipeline(image, results).run()
    assert report.counts["unique_urls"] == 1
    assert report.counts["merged_cards"] == 1
    assert len(report.results[0].pages) == 5


def test_crops_report_their_overlap(small_scene):
    image = decode(to_bytes(small_scene, "PNG"))
    crop = crop_region(small_scene, 0.2, 0.2, 0.8, 0.8)
    results = [RawResult(url=_data_url(encode_jpeg(crop, 90)), engine="google_lens",
                         engine_label="Partial match", page_url="https://a.example/crop")]
    report, _by_cid = _pipeline(image, results).run()
    crops = [r for r in report.results if r.label is Label.CROP]
    assert crops, f"crop was not detected: {[r.label.value for r in report.results]}"
    assert 0.20 <= crops[0].overlap <= 0.52


# --------------------------------------------------------------------------
# Staging: unrelated candidates must not be deep-verified
# --------------------------------------------------------------------------

def test_unrelated_candidates_are_not_promoted_to_stage_5(small_scene):
    image = decode(to_bytes(small_scene, "PNG"))
    results = [
        RawResult(url=_data_url(encode_jpeg(unrelated_scene(seed), 85)),
                  engine="google_lens", page_url=f"https://n.example/{seed}")
        for seed in range(100, 108)
    ]
    report, by_cid = _pipeline(image, results).run()
    deep = [c for c in by_cid.values() if c.stage_reached >= 5]
    assert not deep, f"{len(deep)} unrelated images were deep-verified"
    assert report.counts["one_to_one"] == 0
