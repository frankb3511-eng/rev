"""Search orchestration: query every selected engine, then download the hits.

The downloader understands three URL schemes so the same pipeline serves live
engines, the local corpus, and tests:

``http``/``https``  fetched with a browser-like UA, size cap and timeout
``corpus://``       resolved against the offline corpus directory
``data:``           decoded inline (used by the demo fixtures)
"""

from __future__ import annotations

import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional
from urllib.parse import unquote, urlsplit

from .engines import REQUEST_TIMEOUT, USER_AGENT, Engine, EngineError, select
from .imaging import MAX_UPLOAD_BYTES
from .pipeline import RawResult

log = logging.getLogger(__name__)

MAX_RESULT_BYTES = 24 * 1024 * 1024


@dataclass
class EngineOutcome:
    """What one engine contributed, including why it contributed nothing."""

    engine: str
    display_name: str
    count: int = 0
    error: Optional[str] = None
    elapsed_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "display_name": self.display_name,
            "count": self.count,
            "error": self.error,
            "elapsed_ms": self.elapsed_ms,
        }


@dataclass
class SearchOutcome:
    results: List[RawResult] = field(default_factory=list)
    engines: List[EngineOutcome] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_results": len(self.results),
            "engines": [e.to_dict() for e in self.engines],
        }


def run_search(
    image_bytes: bytes,
    filename: str,
    engine_names: Optional[List[str]] = None,
    max_workers: int = 4,
) -> SearchOutcome:
    """Query every selected engine concurrently.

    A failing engine never aborts the run: its error is recorded and surfaced in
    the UI next to the engines that did respond.
    """
    import time

    engines = select(engine_names)
    outcome = SearchOutcome()

    def work(engine: Engine) -> EngineOutcome:
        t0 = time.perf_counter()
        try:
            hits = engine.search(image_bytes, filename)
            return EngineOutcome(
                engine=engine.name, display_name=engine.display_name,
                count=len(hits), elapsed_ms=int((time.perf_counter() - t0) * 1000),
            ), hits
        except EngineError as exc:
            return EngineOutcome(engine.name, engine.display_name, 0, str(exc),
                                 int((time.perf_counter() - t0) * 1000)), []
        except Exception as exc:                       # network / parse errors
            log.warning("engine %s failed: %s", engine.name, exc)
            return EngineOutcome(engine.name, engine.display_name, 0, f"{type(exc).__name__}: {exc}",
                                 int((time.perf_counter() - t0) * 1000)), []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(work, e) for e in engines]
        for fut in as_completed(futures):
            res = fut.result()
            if isinstance(res, tuple):
                eo, hits = res
                outcome.engines.append(eo)
                outcome.results.extend(hits)
            else:                                      # pragma: no cover - defensive
                outcome.engines.append(res)

    outcome.engines.sort(key=lambda e: e.engine)
    return outcome


# --------------------------------------------------------------------------
# Downloader
# --------------------------------------------------------------------------

def make_downloader(corpus_dir: Optional[str] = None) -> Callable[[RawResult], bytes]:
    """Build the downloader used by the verification pipeline."""
    root = Path(corpus_dir or os.environ.get("CORPUS_DIR") or "data/corpus")

    def download(result: RawResult) -> bytes:
        url = result.url
        scheme = urlsplit(url).scheme.lower()

        if scheme == "data":
            payload = url.split(",", 1)[1] if "," in url else ""
            return base64.b64decode(payload)

        if scheme == "corpus" or url.startswith("/corpus/"):
            rel = url.split("/corpus/", 1)[-1] if "/corpus/" in url else urlsplit(url).path
            path = (root / unquote(rel)).resolve()
            if not str(path).startswith(str(root.resolve())):
                raise ValueError("corpus path escapes the corpus directory")
            return path.read_bytes()

        import requests

        headers = {"User-Agent": USER_AGENT, "Referer": result.page_url or ""}
        with requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True) as resp:
            resp.raise_for_status()
            ctype = resp.headers.get("Content-Type", "")
            if ctype and not ctype.startswith(("image/", "application/octet-stream", "binary/")):
                raise ValueError(f"not an image (Content-Type: {ctype})")
            chunks: List[bytes] = []
            size = 0
            for chunk in resp.iter_content(chunk_size=65536):
                size += len(chunk)
                if size > MAX_RESULT_BYTES:
                    raise ValueError(f"result image exceeds {MAX_RESULT_BYTES // (1024 * 1024)} MiB")
                chunks.append(chunk)
            return b"".join(chunks)

    return download
