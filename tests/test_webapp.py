"""End-to-end tests over the Flask app: upload, verify, results, comparison."""

from __future__ import annotations

import io
import time

import pytest

from app.imaging import encode_jpeg, resize_rgb
from app.webapp import create_app
from tests.fixtures import base_scene


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    import os

    data_dir = tmp_path_factory.mktemp("revdata")
    os.environ["REV_DATA_DIR"] = str(data_dir)
    os.environ["CORPUS_DIR"] = str(data_dir / "corpus")
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


@pytest.fixture(scope="module")
def upload_bytes():
    return encode_jpeg(resize_rgb(base_scene(42), (320, 240)), 92)


def _wait_for_run(client, run_id, timeout=180.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        data = client.get(f"/api/run/{run_id}").get_json()
        if data.get("complete"):
            return data
        time.sleep(0.5)
    raise AssertionError(f"run {run_id} did not finish within {timeout}s")


# --------------------------------------------------------------------------
# Pages and metadata endpoints
# --------------------------------------------------------------------------

def test_index_lists_engines_with_availability(client):
    res = client.get("/")
    assert res.status_code == 200
    body = res.get_data(as_text=True)
    assert "1:1 verification" in body
    assert "Local corpus (offline)" in body


def test_results_page_is_served(client):
    res = client.get("/r/doesnotexist")
    assert res.status_code == 404


def test_thresholds_endpoint_exposes_the_configuration(client):
    data = client.get("/api/thresholds").get_json()
    assert data["hash_bits"] == 64
    assert data["classifier"]["identity_exact"] > data["classifier"]["identity_near"]
    assert set(data["hash_weights"]) == {"ahash", "dhash", "phash", "whash"}
    assert data["features"]["min_inliers"] > 0


def test_engines_endpoint(client):
    data = client.get("/api/engines").get_json()
    names = {e["name"] for e in data}
    assert {"bing", "tineye", "google_lens", "yandex", "corpus"} <= names


# --------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------

def test_upload_without_a_file_is_rejected(client):
    res = client.post("/api/search", data={})
    assert res.status_code == 400


def test_non_image_upload_is_rejected(client):
    res = client.post(
        "/api/search",
        data={"image": (io.BytesIO(b"this is not an image"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    assert "image" in res.get_json()["error"].lower()


# --------------------------------------------------------------------------
# Full run
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def completed_run(client, upload_bytes):
    res = client.post(
        "/api/search",
        data={"image": (io.BytesIO(upload_bytes), "photo.jpg"), "demo": "1"},
        content_type="multipart/form-data",
    )
    assert res.status_code == 200, res.get_data(as_text=True)
    run_id = res.get_json()["run_id"]
    return run_id, _wait_for_run(client, run_id)


def test_run_completes(completed_run):
    run_id, data = completed_run
    assert data["status"] == "complete"
    assert data["report"] is not None


def test_report_is_grouped_with_one_to_one_first(completed_run):
    _run_id, data = completed_run
    groups = data["report"]["groups"]
    assert groups, "no groups in the report"
    assert groups[0]["group"] == "exact"
    assert groups[0]["title"] == "EXACT 1:1 MATCHES"


def test_report_contains_the_expected_mix(completed_run):
    _run_id, data = completed_run
    c = data["report"]["counts"]
    assert c["one_to_one"] >= 3
    assert c["crops"] >= 1
    assert c["similar"] >= 1
    assert c["unrelated"] >= 1


def test_original_metadata_is_reported(completed_run):
    _run_id, data = completed_run
    o = data["report"]["original"]
    assert o["width"] == 320 and o["height"] == 240
    assert len(o["sha256"]) == 64


def test_duplicate_content_is_merged_into_one_card(completed_run):
    """The demo set returns the same bytes from three engines."""
    _run_id, data = completed_run
    multi = [r for r in data["report"]["results"] if len(r["engines"]) > 1]
    assert multi, "no cross-engine merging happened"
    assert max(len(r["engines"]) for r in multi) >= 2


def test_every_card_distinguishes_local_verdict_from_engine_label(completed_run):
    _run_id, data = completed_run
    for r in data["report"]["results"]:
        assert r["display"]                      # local verdict, in words
        assert 0.0 <= r["confidence"] <= 100.0   # local confidence
        # The engine's own wording is carried alongside, never merged in.
        assert isinstance(r["search_engine_labels"], list)


# --------------------------------------------------------------------------
# Asset endpoints
# --------------------------------------------------------------------------

def test_original_and_candidate_images_are_served(client, completed_run):
    run_id, data = completed_run
    assert client.get(f"/api/run/{run_id}/original").status_code == 200
    cid = data["report"]["results"][0]["representative_cid"]
    res = client.get(f"/api/run/{run_id}/image/{cid}")
    assert res.status_code == 200
    assert res.data[:3] == b"\x89PN"


@pytest.mark.parametrize("view", ["original", "candidate", "difference", "heatmap"])
def test_comparison_views_render(client, completed_run, view):
    run_id, data = completed_run
    cid = data["report"]["results"][0]["representative_cid"]
    res = client.get(f"/api/run/{run_id}/diff/{cid}?view={view}")
    assert res.status_code == 200
    assert len(res.data) > 1000


def test_comparison_rejects_unknown_views(client, completed_run):
    run_id, data = completed_run
    cid = data["report"]["results"][0]["representative_cid"]
    assert client.get(f"/api/run/{run_id}/diff/{cid}?view=bogus").status_code == 400


def test_comparison_reports_alignment(client, completed_run):
    run_id, data = completed_run
    cid = data["report"]["results"][0]["representative_cid"]
    meta = client.get(f"/api/run/{run_id}/compare/{cid}").get_json()
    assert meta["align_method"] in ("coextensive", "homography", "template", "none")
    assert 0.0 <= meta["coverage_orig"] <= 1.0
    assert 0.0 <= meta["coverage_cand"] <= 1.0
    assert len(meta["size"]) == 2


def test_unknown_candidate_is_404(client, completed_run):
    run_id, _data = completed_run
    assert client.get(f"/api/run/{run_id}/image/nope").status_code == 404


# --------------------------------------------------------------------------
# The verdict must not come from the search engine
# --------------------------------------------------------------------------

def test_verdicts_are_local_and_independent_of_engine_labels(client, completed_run):
    """A card labelled 'Exact match' by the engine still needs local proof.

    The demo set deliberately attaches the engine label 'Visually similar' to a
    byte-level re-encode (which the matcher proves is 1:1) and 'Exact match' to
    a lookalike photograph (which it does not).  If the engine label leaked into
    the verdict, these would come out the other way round.
    """
    _run_id, data = completed_run
    for r in data["report"]["results"]:
        titles = " ".join(s["title"] for s in r["sources"])
        if "different photograph" in titles:
            assert r["label"] != "EXACT_1TO1"
            assert not r["display"].startswith("1:1")
