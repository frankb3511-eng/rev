"""The staged 1:1 verification pipeline.

    STAGE 1  URL / content-hash deduplication      (no pixel work)
    STAGE 2  fast perceptual hashing               (32x32 downsample)
    STAGE 3  candidate filtering                   (cheap gate)
    STAGE 4  detailed pixel + structural comparison
    STAGE 5  feature / embedding verification      (promising candidates only)
    STAGE 6  cross-engine merge of duplicate results

Stage 4 produces a *provisional* verdict.  Anything that could conceivably be a
1:1 match is promoted to stage 5, where ORB/RANSAC geometry, template
localisation and the descriptor embedding decide the final label.  Candidates
stage 4 rates as unrelated never reach stage 5 -- that is what keeps the cost
flat when an engine returns thousands of results.

Nothing in this module reads a search engine's own similarity label.  Engine
labels are carried through to the UI verbatim and shown *next to* the local
verdict, never merged into it.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import config, hashing
from .imaging import LoadedImage, UnsupportedImage, decode
from .matcher import (
    GROUP_ORDER,
    GROUP_TITLES,
    Label,
    MatchInput,
    Verdict,
    classify,
    compute_signals,
    error_verdict,
    fuse,
    match_pair,
    quick_identity,
)

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, dict], None]


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class RawResult:
    """One hit returned by one engine, before any local verification."""

    url: str
    engine: str
    engine_label: str = ""
    page_url: str = ""
    title: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    engine_score: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Source:
    """Where a merged result was found."""

    engine: str
    engine_label: str
    url: str
    page_url: str = ""
    title: str = ""
    engine_score: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Candidate:
    """A downloaded result image plus everything measured about it."""

    cid: str
    url: str
    engine: str
    engine_label: str
    page_url: str
    title: str
    image: Optional[LoadedImage] = None
    hashes: Optional[hashing.Hashes] = None
    verdict: Optional[Verdict] = None
    stage_reached: int = 1
    error: Optional[str] = None
    download_ms: int = 0

    @property
    def ok(self) -> bool:
        return self.image is not None and self.hashes is not None


@dataclass
class MergedResult:
    """One result card: a cluster of locally-verified duplicates."""

    key: str
    label: Label
    display: str
    group: str
    confidence: float
    overlap: Optional[float]
    sources: List[Source] = field(default_factory=list)
    pages: List[str] = field(default_factory=list)
    engines: List[str] = field(default_factory=list)
    representative_cid: str = ""
    best_url: str = ""
    best_width: int = 0
    best_height: int = 0
    verdict: Optional[Verdict] = None
    member_cids: List[str] = field(default_factory=list)
    search_engine_labels: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label.value,
            "display": self.display,
            "group": self.group,
            "confidence": round(self.confidence, 1),
            "overlap": None if self.overlap is None else round(self.overlap, 4),
            "sources": [s.to_dict() for s in self.sources],
            "pages": self.pages,
            "engines": self.engines,
            "search_engine_labels": self.search_engine_labels,
            "representative_cid": self.representative_cid,
            "best_url": self.best_url,
            "best_width": self.best_width,
            "best_height": self.best_height,
            "member_cids": self.member_cids,
            "verdict": self.verdict.to_dict() if self.verdict else None,
        }


@dataclass
class PipelineReport:
    original_cid: str
    counts: Dict[str, int]
    groups: List[dict]
    results: List[MergedResult]
    timings_ms: Dict[str, int]
    stages: List[dict]
    thresholds: dict

    def to_dict(self) -> dict:
        return {
            "original_cid": self.original_cid,
            "counts": self.counts,
            "groups": self.groups,
            "results": [r.to_dict() for r in self.results],
            "timings_ms": self.timings_ms,
            "stages": self.stages,
            "thresholds": self.thresholds,
        }


# --------------------------------------------------------------------------
# Stage 1 -- URL deduplication
# --------------------------------------------------------------------------

def canonical_url(url: str) -> str:
    """Drop the fragments and tracking parameters that make one image look like two."""
    from urllib.parse import urlsplit, urlunsplit
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip().lower()
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", "")).lower()


def stage1_dedupe(results: Sequence[RawResult]) -> Tuple[List[RawResult], List[Source]]:
    """Collapse identical URLs into one representative plus a source list."""
    reps: Dict[str, RawResult] = {}
    extras: List[Source] = []
    for r in results:
        key = canonical_url(r.url)
        if key not in reps:
            reps[key] = r
        else:
            extras.append(Source(
                engine=r.engine, engine_label=r.engine_label, url=r.url,
                page_url=r.page_url, title=r.title, engine_score=r.engine_score,
            ))
    return list(reps.values()), extras


# --------------------------------------------------------------------------
# Stage 3 -- candidate filtering
# --------------------------------------------------------------------------

@dataclass
class FilterDecision:
    cid: str
    kept: bool
    reason: str
    hash_sim: float
    color_sim: float


def stage3_filter(
    original: Candidate,
    candidates: Sequence[Candidate],
) -> Tuple[List[Candidate], List[FilterDecision]]:
    """Cheap gate: hash similarity, colour similarity, then a top-N safety net."""
    cf = config.CANDIDATE_FILTER
    assert original.hashes is not None
    scored: List[Tuple[float, float, Candidate]] = []
    decisions: List[FilterDecision] = []

    for c in candidates:
        if not c.ok:
            decisions.append(FilterDecision(c.cid, False, c.error or "undecodable", 0.0, 0.0))
            continue
        hc = hashing.compare(original.hashes, c.hashes)
        color = hashing.histogram_cosine(original.hashes.color_hist, c.hashes.color_hist)
        keep = hc.fused >= cf.keep_hash_sim
        reason = "hash similarity above threshold"
        if not keep and color >= cf.rescue_color_sim and hc.fused >= cf.rescue_hash_sim:
            # Crops keep the palette but lose the global hash -- rescue them.
            keep, reason = True, "colour-palette rescue (possible crop)"
        if not keep:
            reason = f"hash similarity {hc.fused:.3f} below {cf.keep_hash_sim}"
        decisions.append(FilterDecision(c.cid, keep, reason, hc.fused, color))
        scored.append((hc.fused, color, c))

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    kept: List[Candidate] = []
    seen = set()
    for fused, _color, c in scored:
        d = next(x for x in decisions if x.cid == c.cid)
        if d.kept and len(kept) < cf.hard_cap:
            kept.append(c)
            seen.add(c.cid)

    # Top-N safety net: never discard the best-ranked candidates outright.
    for fused, _color, c in scored[: cf.keep_top_n]:
        if c.cid in seen or len(kept) >= cf.hard_cap:
            continue
        kept.append(c)
        seen.add(c.cid)
        d = next(x for x in decisions if x.cid == c.cid)
        if not d.kept:
            d.kept, d.reason = True, f"kept by top-{cf.keep_top_n} safety net"

    return kept, decisions


# --------------------------------------------------------------------------
# Stage 6 -- cross-engine merge
# --------------------------------------------------------------------------

def _union_find_merge(
    cands: Sequence[Candidate],
    is_same: Callable[[Candidate, Candidate], bool],
) -> List[List[Candidate]]:
    parent = {c.cid: c.cid for c in cands}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            if a.hashes and b.hashes and a.hashes.sha256 == b.hashes.sha256:
                union(a.cid, b.cid)
                continue
            if is_same(a, b):
                union(a.cid, b.cid)

    clusters: Dict[str, List[Candidate]] = {}
    for c in cands:
        clusters.setdefault(find(c.cid), []).append(c)
    return list(clusters.values())


def _same_rendered_image(a: Candidate, b: Candidate, cache: Dict[Tuple[str, str], bool]) -> bool:
    """Are two candidates the *same rendered image*?

    This is deliberately stricter than "both are 1:1 matches against the
    original".  A rotated copy and a brightened copy are both 1:1 against the
    original, but they are different images on different sites, and collapsing
    them into one card would hide sources.  Merging is for the case in the
    specification: one image returned by several engines.
    """
    key = tuple(sorted((a.cid, b.cid)))
    if key in cache:
        return cache[key]
    C = config.CLASSIFIER
    P = config.PIXEL
    ok = False
    if a.hashes and b.hashes and a.image and b.image:
        q = quick_identity(a.hashes, a.image.rgb, b.hashes, b.image.rgb)
        ok = (
            q.identity >= C.identity_exact
            and q.changed_fraction <= P.local_diff_exact
            # A brightness or watermark difference is a real difference between
            # two files, even though ``changed_fraction`` is blind to it (it is
            # measured after the gain/offset fit).  Only benign re-encodings
            # merge.
            and not q.edit_labels
        )
    cache[key] = ok
    return ok


def _union_find_merge(
    cands: Sequence[Candidate],
    is_same: Callable[[Candidate, Candidate], bool],
) -> List[List[Candidate]]:
    parent = {c.cid: c.cid for c in cands}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            if a.hashes and b.hashes and a.hashes.sha256 == b.hashes.sha256:
                union(a.cid, b.cid)
                continue
            if is_same(a, b):
                union(a.cid, b.cid)

    clusters: Dict[str, List[Candidate]] = {}
    for c in cands:
        clusters.setdefault(find(c.cid), []).append(c)
    return list(clusters.values())


def _crops_match(a: Candidate, b: Candidate, cache: Dict[Tuple[str, str], bool]) -> bool:
    """Two crops merge only if they cut the same region to within a tolerance."""
    key = tuple(sorted((a.cid, b.cid)))
    if key in cache:
        return cache[key]
    C = config.CLASSIFIER
    ok = False
    if a.verdict and b.verdict and a.verdict.label is Label.CROP and b.verdict.label is Label.CROP:
        if abs((a.verdict.overlap or 0) - (b.verdict.overlap or 0)) <= 0.05:
            q = quick_identity(a.hashes, a.image.rgb, b.hashes, b.image.rgb)
            ok = (
                q.identity >= C.identity_exact
                and q.changed_fraction <= config.PIXEL.local_diff_exact
                and not q.edit_labels
            )
    cache[key] = ok
    return ok


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

class VerificationPipeline:
    """Runs the whole staged verification for one uploaded image."""

    def __init__(
        self,
        original: LoadedImage,
        raw_results: Sequence[RawResult],
        downloader: Callable[[RawResult], bytes],
        progress: Optional[ProgressFn] = None,
        max_workers: int = 8,
        max_candidates: int = 400,
    ) -> None:
        self.original = original
        self.raw_results = list(raw_results)
        self.downloader = downloader
        self.progress = progress or (lambda _s, _d: None)
        self.max_workers = max_workers
        self.max_candidates = max_candidates
        self.timings: Dict[str, int] = {}
        self.stage_log: List[dict] = []

    # -- helpers -------------------------------------------------------------
    def _emit(self, status: str, **detail) -> None:
        self.progress(status, detail)

    def _log_stage(self, stage: str, **detail) -> None:
        self.stage_log.append({"stage": stage, **detail})

    # -- stages --------------------------------------------------------------
    def run(self) -> Tuple[PipelineReport, Dict[str, Candidate]]:
        total_t0 = time.perf_counter()
        C = config.CLASSIFIER
        cf = config.CANDIDATE_FILTER

        # ---------------- stage 1 -----------------------------------------
        t0 = time.perf_counter()
        self._emit("Deduplicating URLs...", total=len(self.raw_results))
        uniques, alias_sources = stage1_dedupe(self.raw_results)
        uniques = uniques[: self.max_candidates]
        self.timings["stage1_dedupe_ms"] = int((time.perf_counter() - t0) * 1000)
        self._log_stage(
            "1_url_dedup",
            incoming=len(self.raw_results),
            unique_urls=len(uniques),
            collapsed=len(self.raw_results) - len(uniques),
        )

        original_hashes = hashing.compute(self.original)
        original = Candidate(
            cid="original", url="", engine="", engine_label="", page_url="", title="",
            image=self.original, hashes=original_hashes,
        )

        # ---------------- stage 2 -----------------------------------------
        t0 = time.perf_counter()
        self._emit(f"Checking {len(uniques)} images...", total=len(uniques))
        candidates = self._download_and_hash(uniques)
        self.timings["stage2_hash_ms"] = int((time.perf_counter() - t0) * 1000)

        # content-hash deduplication: identical bytes are one candidate
        by_sha: Dict[str, List[Candidate]] = {}
        for c in candidates:
            if c.ok:
                by_sha.setdefault(c.hashes.sha256, []).append(c)
        content_unique = [v[0] for v in by_sha.values()]
        self._log_stage(
            "2_hash",
            downloaded=len(candidates),
            failed=sum(1 for c in candidates if not c.ok),
            identical_byte_groups=sum(1 for v in by_sha.values() if len(v) > 1),
            content_unique=len(content_unique),
        )

        # ---------------- stage 3 -----------------------------------------
        t0 = time.perf_counter()
        kept, decisions = stage3_filter(original, content_unique)
        self.timings["stage3_filter_ms"] = int((time.perf_counter() - t0) * 1000)
        self._log_stage(
            "3_filter",
            considered=len(content_unique),
            promoted=len(kept),
            rejected=len(content_unique) - len(kept),
        )

        # ---------------- stage 4 -----------------------------------------
        t0 = time.perf_counter()
        self._emit(f"Comparing {len(kept)} images...", checked=len(kept))
        provisional: Dict[str, Verdict] = {}
        for i, c in enumerate(kept, 1):
            sig, edits, _feat, _tpl = compute_signals(
                original.hashes, self.original.rgb, c.hashes, c.image.rgb,
                work_size=cf.thumb_size * 2,
                compare_size=cf.compare_size,
                run_features=False,
            )
            c.verdict = classify(sig, edits)
            c.stage_reached = 4
            provisional[c.cid] = c.verdict
            if i % 10 == 0:
                self._emit(f"Comparing {len(kept)} images...", checked=i, total=len(kept))
        self.timings["stage4_compare_ms"] = int((time.perf_counter() - t0) * 1000)

        promising = [
            c for c in kept
            if c.verdict and c.verdict.identity >= C.identity_similar * 0.85
        ]
        self._log_stage(
            "4_compare",
            compared=len(kept),
            promising=len(promising),
        )

        # ---------------- stage 5 -----------------------------------------
        t0 = time.perf_counter()
        self._emit(f"{len(promising)} potential matches found...", potential=len(promising))
        self._emit("Verifying...", verifying=len(promising))
        for c in promising:
            c.verdict = match_pair(
                MatchInput(original.hashes, self.original.rgb),
                MatchInput(c.hashes, c.image.rgb),
                work_size=cf.work_size,
                compare_size=cf.compare_size,
            )
            c.stage_reached = 5
        self.timings["stage5_verify_ms"] = int((time.perf_counter() - t0) * 1000)
        self._log_stage("5_verify", verified=len(promising))

        # ---------------- stage 6 -----------------------------------------
        t0 = time.perf_counter()
        merged = self._merge(candidates, by_sha, alias_sources)
        self.timings["stage6_merge_ms"] = int((time.perf_counter() - t0) * 1000)

        counts = {
            "raw_results": len(self.raw_results),
            "unique_urls": len(uniques),
            "downloaded": sum(1 for c in candidates if c.ok),
            "failed_downloads": sum(1 for c in candidates if not c.ok),
            "compared": len(kept),
            "verified": len(promising),
            "merged_cards": len(merged),
            "one_to_one": sum(1 for r in merged if r.label.is_one_to_one),
            "crops": sum(1 for r in merged if r.label is Label.CROP),
            "similar": sum(1 for r in merged if r.label is Label.VISUALLY_SIMILAR),
            "unrelated": sum(1 for r in merged if r.label in (Label.UNRELATED, Label.ERROR)),
        }
        self._log_stage("6_merge", cards=len(merged), one_to_one=counts["one_to_one"])

        groups = []
        for g in GROUP_ORDER:
            items = [r for r in merged if r.group == g]
            if items:
                groups.append({
                    "group": g,
                    "title": GROUP_TITLES[g],
                    "count": len(items),
                    "results": [r.to_dict() for r in items],
                })

        self.timings["total_ms"] = int((time.perf_counter() - total_t0) * 1000)
        self._emit("Complete", **counts)

        report = PipelineReport(
            original_cid="original",
            counts=counts,
            groups=groups,
            results=merged,
            timings_ms=self.timings,
            stages=self.stage_log,
            thresholds=config.as_dict(),
        )
        by_cid = {"original": original}
        by_cid.update({c.cid: c for c in candidates})
        return report, by_cid

    # -- download ------------------------------------------------------------
    def _download_and_hash(self, uniques: Sequence[RawResult]) -> List[Candidate]:
        out: List[Candidate] = []

        def work(idx_r: Tuple[int, RawResult]) -> Candidate:
            idx, r = idx_r
            cid = f"c{idx:05d}"
            c = Candidate(
                cid=cid, url=r.url, engine=r.engine, engine_label=r.engine_label,
                page_url=r.page_url, title=r.title,
            )
            t0 = time.perf_counter()
            try:
                data = self.downloader(r)
                c.download_ms = int((time.perf_counter() - t0) * 1000)
                c.image = decode(data)
                c.hashes = hashing.compute(c.image)
                c.stage_reached = 2
            except UnsupportedImage as exc:
                c.error = f"not an image: {exc}"
            except Exception as exc:                      # network / timeout
                c.error = f"download failed: {exc}"
            return c

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for c in pool.map(work, list(enumerate(uniques))):
                out.append(c)
        return out

    # -- merge ---------------------------------------------------------------
    def _merge(
        self,
        candidates: Sequence[Candidate],
        by_sha: Dict[str, List[Candidate]],
        alias_sources: Sequence[Source],
    ) -> List[MergedResult]:
        verified = [c for c in candidates if c.verdict is not None]
        # Representatives of each byte-identical group carry the group's sources.
        rep_of: Dict[str, str] = {}
        for _sha, group in by_sha.items():
            for c in group:
                rep_of[c.cid] = group[0].cid

        ones = [c for c in verified if c.verdict.label.is_one_to_one]
        crops = [c for c in verified if c.verdict.label is Label.CROP]
        rest = [c for c in verified
                if c.verdict.label in (Label.VISUALLY_SIMILAR, Label.UNRELATED, Label.ERROR)]

        cache: Dict[Tuple[str, str], bool] = {}

        # Cluster only within a label.  The matcher already decided that a
        # watermarked copy is EDITED while a clean re-encode is EXACT; merging
        # across that boundary would discard a distinction it had just proven.
        # This also guards the merge against the limits of its own resolution --
        # a faint watermark on a large image is not reliably re-detected at the
        # reduced size the pairwise merge runs at.
        by_label: Dict[Label, List[Candidate]] = {}
        for c in ones:
            by_label.setdefault(c.verdict.label, []).append(c)

        clusters: List[List[Candidate]] = []
        for _label, group in by_label.items():
            clusters += _union_find_merge(group, lambda a, b: _same_rendered_image(a, b, cache))
        clusters += _union_find_merge(crops, lambda a, b: _crops_match(a, b, cache))
        clusters += [[c] for c in rest]

        merged: List[MergedResult] = []
        for cluster in clusters:
            if not cluster:
                continue
            rep = max(cluster, key=lambda c: (c.image.width * c.image.height if c.image else 0))
            verdict = rep.verdict
            sources: List[Source] = []
            pages: List[str] = []
            engines: List[str] = []
            engine_labels: List[str] = []
            for c in cluster:
                for member in by_sha.get(rep.hashes.sha256, [rep]) if c.cid == rep.cid else [c]:
                    sources.append(Source(
                        engine=member.engine, engine_label=member.engine_label,
                        url=member.url, page_url=member.page_url, title=member.title,
                    ))
                    if member.page_url and member.page_url not in pages:
                        pages.append(member.page_url)
                    if member.engine and member.engine not in engines:
                        engines.append(member.engine)
                    if member.engine_label and member.engine_label not in engine_labels:
                        engine_labels.append(member.engine_label)
            merged.append(MergedResult(
                key=f"{verdict.label.value}:{rep.cid}",
                label=verdict.label,
                display=verdict.label.display,
                group=verdict.label.group,
                confidence=verdict.confidence,
                overlap=verdict.overlap,
                sources=sources,
                pages=pages,
                engines=engines,
                representative_cid=rep.cid,
                best_url=rep.url,
                best_width=rep.image.width if rep.image else 0,
                best_height=rep.image.height if rep.image else 0,
                verdict=verdict,
                member_cids=[c.cid for c in cluster],
                search_engine_labels=engine_labels,
            ))

        # Alias sources (URL-level duplicates) attach to whichever card owns the URL.
        url_to_card = {}
        for m in merged:
            for s in m.sources:
                url_to_card[canonical_url(s.url)] = m
        for s in alias_sources:
            card = url_to_card.get(canonical_url(s.url))
            if card is None:
                continue
            card.sources.append(s)
            if s.page_url and s.page_url not in card.pages:
                card.pages.append(s.page_url)
            if s.engine and s.engine not in card.engines:
                card.engines.append(s.engine)

        order = {g: i for i, g in enumerate(GROUP_ORDER)}
        merged.sort(key=lambda m: (order.get(m.group, 99), -m.confidence))
        return merged
