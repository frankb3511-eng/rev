"""Flask application: upload, search, staged verification, results, comparison.

A search runs in a background thread so the browser can poll progress and show
the live status line ("Checking 183 images... 12 potential matches found...").
The original upload is written to disk once and never modified; every derived
image is stored separately.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, abort, jsonify, render_template, request, send_file

from . import config, demo, search as search_mod
from .compare import build_comparison
from .imaging import LoadedImage, UnsupportedImage, decode, encode_jpeg, encode_png
from .pipeline import Candidate, RawResult, VerificationPipeline

log = logging.getLogger(__name__)

def _data_dir() -> Path:
    """Absolute data directory, read when the app is created.

    These must not be module-level constants: reading the environment at import
    time meant an overridden ``REV_DATA_DIR`` was ignored, and a relative path
    made ``send_file`` resolve against Flask's ``root_path`` (``app/``) while
    ``Path.write_bytes`` had written relative to the process working directory.
    """
    return Path(os.environ.get("REV_DATA_DIR", "data")).resolve()


def _corpus_dir() -> Path:
    return Path(os.environ.get("CORPUS_DIR", str(_data_dir() / "corpus"))).resolve()


# --------------------------------------------------------------------------
# Run store
# --------------------------------------------------------------------------

@dataclass
class Run:
    run_id: str
    status: str = "queued"                 # queued | searching | verifying | complete | error
    status_text: str = "Queued"
    progress: dict = field(default_factory=dict)
    engines: List[dict] = field(default_factory=list)
    report: Optional[dict] = None
    error: Optional[str] = None
    started: float = field(default_factory=time.time)
    original_path: Path = field(default_factory=lambda: Path("."))
    candidates: Dict[str, Candidate] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def public(self) -> dict:
        with self.lock:
            return {
                "run_id": self.run_id,
                "status": self.status,
                "status_text": self.status_text,
                "progress": self.progress,
                "engines": self.engines,
                "error": self.error,
                "elapsed_s": round(time.time() - self.started, 2),
                "complete": self.status in ("complete", "error"),
                "has_report": self.report is not None,
            }


_RUNS: Dict[str, Run] = {}
_RUNS_LOCK = threading.Lock()


def _get_run(run_id: str) -> Run:
    run = _RUNS.get(run_id)
    if run is None:
        abort(404, description="unknown run")
    return run


# --------------------------------------------------------------------------
# App factory
# --------------------------------------------------------------------------

def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
    app.config["RUNS_DIR"] = _data_dir() / "runs"
    app.config["CORPUS_DIR"] = _corpus_dir()
    app.config["RUNS_DIR"].mkdir(parents=True, exist_ok=True)

    # -- pages ---------------------------------------------------------------
    @app.get("/")
    def index():
        from .engines import all_engines
        return render_template(
            "index.html",
            engines=[{
                "name": e.name,
                "display_name": e.display_name,
                "description": e.description,
                "available": e.available(),
                "note": e.availability_note(),
            } for e in all_engines()],
        )

    @app.get("/r/<run_id>")
    def results_page(run_id: str):
        run = _get_run(run_id)
        return render_template("results.html", run_id=run_id, state=run.public())

    # -- api ------------------------------------------------------------------
    @app.get("/api/engines")
    def api_engines():
        from .engines import all_engines
        return jsonify([{
            "name": e.name, "display_name": e.display_name,
            "available": e.available(), "note": e.availability_note(),
        } for e in all_engines()])

    @app.get("/api/thresholds")
    def api_thresholds():
        """Every threshold and weight the matcher uses, as configured."""
        return jsonify(config.as_dict())

    @app.post("/api/search")
    def api_search():
        file = request.files.get("image")
        if file is None or not file.filename:
            return jsonify({"error": "no image uploaded"}), 400
        data = file.read()
        try:
            original = decode(data)
        except UnsupportedImage as exc:
            return jsonify({"error": str(exc)}), 400

        engines = request.form.getlist("engines") or None
        use_demo = request.form.get("demo") in ("1", "true", "on")

        run_id = uuid.uuid4().hex[:12]
        run_dir = app.config["RUNS_DIR"] / run_id
        (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
        ext = f".{original.fmt.lower()}" if original.fmt != "UNKNOWN" else ".bin"
        original_path = run_dir / f"original{ext}"
        original_path.write_bytes(data)

        run = Run(run_id=run_id, original_path=original_path)
        with _RUNS_LOCK:
            _RUNS[run_id] = run

        thread = threading.Thread(
            target=_execute_run,
            args=(run_id, original, data, file.filename, engines, use_demo,
                  app.config["RUNS_DIR"], app.config["CORPUS_DIR"]),
            daemon=True,
        )
        thread.start()
        return jsonify({"run_id": run_id, "results_url": f"/r/{run_id}"})

    @app.get("/api/run/<run_id>")
    def api_run(run_id: str):
        run = _get_run(run_id)
        payload = run.public()
        with run.lock:
            payload["report"] = run.report
        return jsonify(payload)

    @app.get("/api/run/<run_id>/original")
    def api_original(run_id: str):
        run = _get_run(run_id)
        if not run.original_path.is_file():
            abort(404)
        return send_file(run.original_path, mimetype="image/png")

    @app.get("/api/run/<run_id>/image/<cid>")
    def api_candidate_image(run_id: str, cid: str):
        run = _get_run(run_id)
        cand = run.candidates.get(cid)
        if cand is None or cand.image is None:
            abort(404, description="unknown candidate")
        path = _candidate_path(run, cid, cand)
        return send_file(path, mimetype="image/png")

    @app.get("/api/run/<run_id>/diff/<cid>")
    def api_diff(run_id: str, cid: str):
        """Aligned comparison renders: ``?view=original|candidate|difference|heatmap``."""
        run = _get_run(run_id)
        cand = run.candidates.get(cid)
        if cand is None or cand.image is None:
            abort(404, description="unknown candidate")
        view = request.args.get("view", "difference")
        comparison = build_comparison(_load_original(run), cand.image)
        mapping = {
            "original": comparison.original_bytes,
            "candidate": comparison.candidate_bytes,
            "difference": comparison.difference_bytes,
            "heatmap": comparison.heatmap_bytes,
        }
        if view not in mapping:
            abort(400, description=f"view must be one of {sorted(mapping)}")
        import io
        return send_file(
            io.BytesIO(mapping[view]()),
            mimetype="image/png" if view != "heatmap" else "image/jpeg",
        )

    @app.get("/api/run/<run_id>/compare/<cid>")
    def api_compare_meta(run_id: str, cid: str):
        """Numbers behind the comparison view (alignment method, coverage)."""
        run = _get_run(run_id)
        cand = run.candidates.get(cid)
        if cand is None or cand.image is None:
            abort(404, description="unknown candidate")
        comparison = build_comparison(_load_original(run), cand.image)
        return jsonify({
            "align_method": comparison.align_method,
            "align_valid_fraction": round(comparison.align_valid_fraction, 4),
            "coverage_orig": round(comparison.coverage_orig, 4),
            "coverage_cand": round(comparison.coverage_cand, 4),
            "size": list(comparison.size),
        })

    return app


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _load_original(run: Run) -> LoadedImage:
    return decode(run.original_path.read_bytes())


def _candidate_path(run: Run, cid: str, cand: Candidate) -> Path:
    """Persist a candidate's pixels on first request, then serve from disk."""
    from flask import current_app
    path = current_app.config["RUNS_DIR"] / run.run_id / "candidates" / f"{cid}.png"
    if not path.is_file() and cand.image is not None:
        path.write_bytes(encode_png(cand.image.rgb))
    return path


def _execute_run(
    run_id: str,
    original: LoadedImage,
    original_bytes: bytes,
    filename: str,
    engine_names: Optional[List[str]],
    use_demo: bool,
    runs_dir: Path,
    corpus_dir: Path,
) -> None:
    """Background worker: search, then run the staged verification pipeline."""
    run = _get_run(run_id)

    def on_progress(status: str, detail: dict) -> None:
        with run.lock:
            run.status_text = status
            run.progress = {"status": status, **detail}

    try:
        with run.lock:
            run.status = "searching"
            run.status_text = "Searching reverse image engines..."

        if use_demo:
            raw = demo.synthesize_results(original)
            with run.lock:
                run.engines = [{
                    "engine": demo.ENGINE_NAME,
                    "display_name": demo.display_name(),
                    "count": len(raw),
                    "error": None,
                    "elapsed_ms": 0,
                }]
        else:
            outcome = search_mod.run_search(original_bytes, filename, engine_names)
            raw = outcome.results
            with run.lock:
                run.engines = [e.to_dict() for e in outcome.engines]

        if not raw:
            with run.lock:
                run.status = "error"
                run.status_text = "No results returned"
                run.error = "No search engine returned any results."
                run.report = {
                    "original_cid": "original",
                    "counts": {"raw_results": 0},
                    "groups": [], "results": [], "timings_ms": {}, "stages": [],
                    "thresholds": config.as_dict(),
                    "engines": run.engines,
                }
            return

        with run.lock:
            run.status = "verifying"

        pipeline = VerificationPipeline(
            original=original,
            raw_results=raw,
            downloader=search_mod.make_downloader(str(corpus_dir)),
            progress=on_progress,
        )
        report, by_cid = pipeline.run()

        # Store the original's own hashes so the UI can show them.
        payload = report.to_dict()
        payload["engines"] = run.engines
        payload["original"] = {
            "width": original.width,
            "height": original.height,
            "format": original.fmt,
            "bytes": original.byte_size,
            "sha256": original.sha256,
        }
        with run.lock:
            run.candidates = by_cid
            run.report = payload
            run.status = "complete"
            run.status_text = "Complete"
    except Exception as exc:                        # pragma: no cover - defensive
        log.exception("run %s failed", run_id)
        with run.lock:
            run.status = "error"
            run.status_text = "Failed"
            run.error = f"{type(exc).__name__}: {exc}"


app = create_app()


if __name__ == "__main__":                       # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), debug=False)
