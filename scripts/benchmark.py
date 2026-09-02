"""Run the whole labelled case matrix and print the measured score table.

This is the calibration instrument: it shows every signal the matcher computed
for every edge case, the verdict it reached, and whether that verdict matches
the specification's expectation.  The thresholds in ``app/config.py`` were set
by reading this table.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import config                                    # noqa: E402
from app.imaging import decode                            # noqa: E402
from app.matcher import Label, MatchInput, match_pair     # noqa: E402
from tests.fixtures import base_scene, build_cases        # noqa: E402
import app.hashing as hashing                             # noqa: E402


def main() -> int:
    cases = build_cases()
    orig_img = decode(build_cases()[0].data if False else _original_bytes())
    orig_hashes = hashing.compute(orig_img)
    original = MatchInput(orig_hashes, orig_img.rgb)

    print(f"{'case':<32} {'expected':<18} {'got':<18} {'conf':>6} {'ident':>6} "
          f"{'frame':>6} {'covO':>5} {'covC':>5} {'chg%':>6} {'ssim':>5} {'hash':>5} {'ok':>3}")
    print("-" * 130)

    failures = []
    rows = []
    for case in cases:
        t0 = time.perf_counter()
        cand_img = decode(case.data)
        cand = MatchInput(hashing.compute(cand_img), cand_img.rgb)
        verdict = match_pair(original, cand)
        dt = (time.perf_counter() - t0) * 1000
        s = verdict.signals
        ok = verdict.label.value == case.expected
        if not ok:
            failures.append((case, verdict))
        rows.append((case, verdict, dt))
        print(f"{case.name:<32} {case.expected:<18} {verdict.label.value:<18} "
              f"{verdict.confidence:6.1f} {verdict.identity:6.3f} {verdict.frame_agreement:6.3f} "
              f"{s.coverage_orig:5.2f} {s.coverage_cand:5.2f} {s.changed_fraction*100:6.1f} "
              f"{s.ssim:5.2f} {s.hash_sim:5.2f} {'OK' if ok else 'FAIL':>4}")

    print("-" * 130)
    print(f"{len(cases) - len(failures)}/{len(cases)} cases match the expected verdict")
    total_ms = sum(dt for _, _, dt in rows)
    print(f"mean per-case verification time: {total_ms / max(1, len(rows)):.0f} ms")

    if failures:
        print("\nFAILURES")
        for case, verdict in failures:
            print(f"  {case.name}: expected {case.expected}, got {verdict.label.value} "
                  f"({verdict.confidence:.1f}%)")
            print(f"     reasons: {'; '.join(verdict.reasons)}")
            print(f"     signals: {verdict.signals.to_dict()}")
            print(f"     edits: {verdict.edits.to_dict()}")
    return 1 if failures else 0


def _original_bytes() -> bytes:
    from tests.fixtures import to_bytes
    return to_bytes(base_scene(42), "PNG")


if __name__ == "__main__":
    raise SystemExit(main())
