"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import hashing                                    # noqa: E402
from app.imaging import decode                             # noqa: E402
from app.matcher import MatchInput                         # noqa: E402
from tests.fixtures import base_scene, build_cases, to_bytes  # noqa: E402


@pytest.fixture(scope="session")
def cases():
    """The full labelled edge-case matrix."""
    return build_cases()


@pytest.fixture(scope="session")
def original_bytes():
    return to_bytes(base_scene(42), "PNG")


@pytest.fixture(scope="session")
def original(original_bytes):
    return decode(original_bytes)


@pytest.fixture(scope="session")
def original_input(original):
    return MatchInput(hashing.compute(original), original.rgb)


@pytest.fixture(scope="session")
def verdicts(original_input, cases):
    """Every case run through the real matcher, once per session.

    The matcher is deterministic, so caching the verdicts keeps the suite fast
    without weakening any assertion -- each test still reads the verdict the
    shipped code produced.
    """
    from app.matcher import match_pair

    out = {}
    for case in cases:
        cand = decode(case.data)
        out[case.name] = match_pair(original_input, MatchInput(hashing.compute(cand), cand.rgb))
    return out
