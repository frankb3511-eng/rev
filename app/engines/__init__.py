"""Reverse-image-search engine adapters.

Every engine implements one method and returns plain ``RawResult`` records.
Nothing an engine returns is trusted: the ``engine_label`` and ``engine_score``
fields are carried through to the UI verbatim and displayed *next to* the local
verdict, never folded into it.

Two kinds of adapter live here:

* **API adapters** (Bing, TinEye) -- official, documented REST endpoints that
  need a key.  They are skipped silently when the key is absent.
* **Scrape adapters** (Google Lens, Yandex) -- best-effort HTML flows that need
  no key but break whenever the site changes, and need to be reachable from the
  server.
* **Offline corpus** -- a local folder of images served as if a search engine
  had returned them.  Always available; it is what makes the whole pipeline
  runnable and testable with no network access.

``available_engines()`` reports what is actually usable in this environment so
the UI can say so instead of silently returning nothing.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

from ..pipeline import RawResult

log = logging.getLogger(__name__)

REQUEST_TIMEOUT = 20
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


class EngineError(RuntimeError):
    """Raised when an engine cannot complete a search."""


class Engine(ABC):
    name: str = "engine"
    display_name: str = "Engine"
    #: Short human description shown in the engine picker.
    description: str = ""
    #: True when the adapter can run in this environment.
    always_available: bool = False

    def available(self) -> bool:
        return self.always_available

    def availability_note(self) -> str:
        return "ready" if self.available() else "unavailable"

    @abstractmethod
    def search(self, image_bytes: bytes, filename: str) -> List[RawResult]:
        """Return the hits this engine produced for one uploaded image."""

    # -- helpers -------------------------------------------------------------
    def _result(self, **kw) -> RawResult:
        kw.setdefault("engine", self.name)
        return RawResult(**kw)


# --------------------------------------------------------------------------
# Bing Visual Search (official API)
# --------------------------------------------------------------------------

class BingVisualSearch(Engine):
    name = "bing"
    display_name = "Bing Visual Search"
    description = "Official Bing Visual Search v7 API. Needs BING_SEARCH_API_KEY."
    endpoint = "https://api.bing.microsoft.com/v7.0/images/visualsearch"

    def available(self) -> bool:
        return bool(os.environ.get("BING_SEARCH_API_KEY"))

    def availability_note(self) -> str:
        return "ready" if self.available() else "set BING_SEARCH_API_KEY"

    def search(self, image_bytes: bytes, filename: str) -> List[RawResult]:
        import requests

        key = os.environ.get("BING_SEARCH_API_KEY")
        if not key:
            raise EngineError("BING_SEARCH_API_KEY is not set")
        files = {"image": (filename or "upload.jpg", image_bytes, "application/octet-stream")}
        resp = requests.post(
            self.endpoint, headers={"Ocp-Apim-Subscription-Key": key},
            files=files, timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        out: List[RawResult] = []
        for tag in data.get("tags", []):
            for action in tag.get("actions", []):
                for value in action.get("values", []) or []:
                    url = value.get("contentUrl") or value.get("hostPageUrl")
                    if not url:
                        continue
                    out.append(self._result(
                        url=url,
                        engine_label=value.get("name") or action.get("displayName") or "",
                        page_url=value.get("hostPageUrl", ""),
                        title=value.get("name", ""),
                        width=(value.get("thumbnail") or {}).get("width"),
                        height=(value.get("thumbnail") or {}).get("height"),
                    ))
        return out


# --------------------------------------------------------------------------
# TinEye (commercial API)
# --------------------------------------------------------------------------

class TinEye(Engine):
    name = "tineye"
    display_name = "TinEye"
    description = "TinEye commercial match API. Needs TINEYE_API_KEY."
    endpoint = "https://api.tineye.com/rest/match"

    def available(self) -> bool:
        return bool(os.environ.get("TINEYE_API_KEY"))

    def availability_note(self) -> str:
        return "ready" if self.available() else "set TINEYE_API_KEY"

    def search(self, image_bytes: bytes, filename: str) -> List[RawResult]:
        import requests

        key = os.environ.get("TINEYE_API_KEY")
        if not key:
            raise EngineError("TINEYE_API_KEY is not set")
        resp = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {key}"},
            files={"file": (filename or "upload.jpg", image_bytes, "application/octet-stream")},
            data={"sort": "score", "offset": 0, "limit": 100},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        out: List[RawResult] = []
        for m in (data.get("result") or {}).get("matches", []) or []:
            url = m.get("image_url")
            if not url:
                continue
            out.append(self._result(
                url=url,
                engine_label=f"TinEye match {m.get('score', '')}".strip(),
                page_url=m.get("backlinks", [{}])[0].get("backlink", "") if m.get("backlinks") else "",
                title=(m.get("backlinks") or [{}])[0].get("backlink_title", ""),
                width=m.get("width"),
                height=m.get("height"),
                engine_score=m.get("score"),
            ))
        return out


# --------------------------------------------------------------------------
# Google Lens (best effort, no key)
# --------------------------------------------------------------------------

class GoogleLens(Engine):
    name = "google_lens"
    display_name = "Google Lens"
    description = "Best-effort Lens upload flow. No key; breaks when Google changes the page."
    upload_url = "https://lens.google.com/v3/upload"

    def available(self) -> bool:
        return True

    def availability_note(self) -> str:
        return "best effort (no API key)"

    def search(self, image_bytes: bytes, filename: str) -> List[RawResult]:
        import re

        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        resp = session.post(
            self.upload_url,
            params={"st": "17", "ep": "gsris"},
            files={"encoded_image": (filename or "upload.jpg", image_bytes, "image/jpeg")},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

        # The upload response carries an encoded image id used to build the
        # results URL; the results page itself is JSON embedded in the HTML.
        ids = re.findall(r"/upload/([A-Za-z0-9_\-]{20,})", resp.text)
        if not ids:
            raise EngineError("could not obtain a Google Lens upload id")
        results_url = f"https://lens.google.com/lens?ep=gsris&rt=SEARCH&rct=1&p=/upload/{ids[0]}"
        page = session.get(results_url, timeout=REQUEST_TIMEOUT)
        page.raise_for_status()

        out: List[RawResult] = []
        seen = set()
        for m in re.finditer(
            r'"(https?://[^"]+\.(?:jpg|jpeg|png|webp|gif))"', page.text, re.IGNORECASE
        ):
            url = m.group(1).replace("\\u003d", "=").replace("\\/", "/")
            if url in seen or "gstatic.com" in url:
                continue
            seen.add(url)
            out.append(self._result(url=url, engine_label="Google Lens result"))
        if not out:
            raise EngineError("Google Lens returned no parseable results")
        return out


# --------------------------------------------------------------------------
# Yandex (best effort, no key)
# --------------------------------------------------------------------------

class YandexImages(Engine):
    name = "yandex"
    display_name = "Yandex Images"
    description = "Best-effort Yandex reverse-image flow. No key; may require a cookie jar."
    upload_url = "https://yandex.com/images/upload"
    cbcr_url = "https://yandex.com/images-appcontent-ajax/realtime/get-cbcr"

    def available(self) -> bool:
        return True

    def availability_note(self) -> str:
        return "best effort (no API key)"

    def search(self, image_bytes: bytes, filename: str) -> List[RawResult]:
        import json
        import re

        import requests

        session = requests.Session()
        session.headers.update({"User-Agent": USER_AGENT})
        # Yandex expects a warm cookie jar before it accepts an upload.
        try:
            session.get("https://yandex.com/images/", timeout=REQUEST_TIMEOUT)
        except requests.RequestException:
            pass
        resp = session.post(
            self.upload_url,
            files={"upfile": (filename or "upload.jpg", image_bytes, "image/jpeg")},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        try:
            uploaded = resp.json()
        except ValueError as exc:
            raise EngineError(f"Yandex upload did not return JSON: {exc}") from exc

        upstr = uploaded.get("uploaded-image") or uploaded.get("uploadedImage")
        if not upstr:
            raise EngineError("Yandex did not return an upload token")
        page = session.get(
            "https://yandex.com/images/search",
            params={"rpt": "imageview", "upfile": upstr, "cbircmd": "cbcr"},
            timeout=REQUEST_TIMEOUT,
        )
        page.raise_for_status()

        out: List[RawResult] = []
        seen = set()
        for m in re.finditer(
            r'"(https?://[^"]+\.(?:jpg|jpeg|png|webp))"', page.text, re.IGNORECASE
        ):
            url = m.group(1).replace("\\/", "/")
            if url in seen or "yastatic" in url or "yandex" in url:
                continue
            seen.add(url)
            out.append(self._result(url=url, engine_label="Yandex visually similar"))
        if not out:
            raise EngineError("Yandex returned no parseable results")
        return out


# --------------------------------------------------------------------------
# Offline corpus -- always available, fully deterministic
# --------------------------------------------------------------------------

class OfflineCorpus(Engine):
    """Serves images from a local folder as if an engine had returned them.

    This is not a stub for the real engines: it exercises the entire
    verification pipeline, including download, deduplication, merging and
    classification, with no network access.  It is also what the demo mode uses
    to produce a results page full of genuine 1:1 matches, crops and near
    misses.
    """

    name = "corpus"
    display_name = "Local corpus (offline)"
    description = "Searches a local folder of images. Always available."
    always_available = True

    def __init__(self, root: Optional[str] = None, base_url: str = "/corpus") -> None:
        self.root = root or os.environ.get("CORPUS_DIR") or "data/corpus"
        self.base_url = base_url.rstrip("/")

    def search(self, image_bytes: bytes, filename: str) -> List[RawResult]:
        from pathlib import Path

        from ..imaging import SUPPORTED_SUFFIXES

        root = Path(self.root)
        if not root.is_dir():
            raise EngineError(f"corpus directory {root} does not exist")
        out: List[RawResult] = []
        for path in sorted(root.rglob("*")):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            rel = path.relative_to(root).as_posix()
            out.append(self._result(
                url=f"{self.base_url}/{rel}",
                engine_label="local corpus hit",
                page_url=f"file://{path}",
                title=path.stem,
            ))
        if not out:
            raise EngineError(f"corpus directory {root} contains no images")
        return out


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

_ENGINES: Dict[str, Engine] = {}


def register(engine: Engine) -> Engine:
    _ENGINES[engine.name] = engine
    return engine


for _e in (BingVisualSearch(), TinEye(), GoogleLens(), YandexImages(), OfflineCorpus()):
    register(_e)


def get(name: str) -> Optional[Engine]:
    return _ENGINES.get(name)


def all_engines() -> List[Engine]:
    return list(_ENGINES.values())


def available_engines() -> List[Engine]:
    return [e for e in _ENGINES.values() if e.available()]


def select(names: Optional[Sequence[str]]) -> List[Engine]:
    """Resolve a user's engine selection; falls back to every available engine."""
    if not names:
        return available_engines()
    chosen = []
    for n in names:
        e = _ENGINES.get(n)
        if e is not None and e.available():
            chosen.append(e)
    return chosen or available_engines()
