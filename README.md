# rev — reverse image search with independent 1:1 verification

Upload an image, search it across multiple reverse-image engines, then run a
**local computer-vision pipeline** over every returned result to decide what is
actually a 1:1 match.

The point of the second half is that a search engine's "visually similar" label
is not evidence. This matcher never reads one. It downloads the result, aligns
it against your original, and measures the pixels.

```
UPLOAD IMAGE
  → SEARCH MULTIPLE REVERSE IMAGE ENGINES
  → COLLECT ALL RESULTS
  → AUTOMATIC 1:1 VERIFICATION        ← local, independent
  → AUTOMATIC DEDUPLICATION
  → CLASSIFY:  1:1 / EDITED 1:1 / CROP / SIMILAR / UNRELATED
  → MERGE DUPLICATE RESULTS
  → SHOW ALL SOURCES
  → DISPLAY RESULTS ON ONE PAGE
```

---

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m app.webapp          # http://localhost:8000
```

Then:

```bash
.venv/bin/python scripts/benchmark.py   # the 25-case labelled matrix
.venv/bin/python -m pytest -q           # 181 tests
```

## Search engines

Set a key to enable the API adapters; they are skipped silently when absent.

| engine | how it runs |
|---|---|
| Bing Visual Search | official v7 API — `BING_SEARCH_API_KEY` |
| TinEye | commercial match API — `TINEYE_API_KEY` |
| Google Lens | best-effort upload scrape, no key |
| Yandex Images | best-effort upload scrape, no key |
| Local corpus | folder of images, always available — `CORPUS_DIR` |

> **Heads up on the scrape adapters.** The Google Lens and Yandex flows are
> written against their current undocumented upload endpoints and could **not be
> exercised in the environment this was built in** — outbound access to
> `lens.google.com`, `yandex.com`, `bing.com` and `tineye.com` is blocked there
> (only `pypi.org` resolves). Treat those two as unverified against the live
> sites; a failing engine is recorded and shown in the UI rather than aborting
> the run. The Bing and TinEye adapters target documented REST endpoints but
> were likewise not run against the real APIs without keys.

### Offline demo mode

Tick **Offline demo mode** on the upload page and the app synthesises a result
set from your upload — resizes, re-encodes, a WebP copy, a watermark, a border,
crops, a rotation, a *different photograph of the same subject*, and unrelated
images — then runs the real pipeline over it.

Only the "search" step is synthetic. Downloading, hashing, filtering, alignment,
feature matching, classification, deduplication and merging are all the real
code running on real pixels, so a demo run demonstrates the matcher rather than
mocking it.

---

## How the 1:1 verification works

Six stages, so cost stays flat when an engine returns thousands of hits:

```
STAGE 1  URL + content-hash deduplication      no pixel work
STAGE 2  fast perceptual hashing               aHash / dHash / pHash / wHash, 64-bit
STAGE 3  candidate filtering                   cheap gate + top-N safety net
STAGE 4  pixel + structural comparison         provisional verdict
STAGE 5  feature / embedding verification      promising candidates only
STAGE 6  cross-engine merge
```

Independent signals are fused into one `identity` score:

`hash_sim` (0.22) · `ssim` (0.22) · `grad_corr` (0.14) · `emb_cos` (0.14) ·
`feat_sim` (0.12) · `nrmse_sim` (0.08) · `hist_corr` (0.08)

Geometry — ORB + RANSAC under a **similarity transform**, backed by a
bidirectional multi-scale template sweep — produces two coverage numbers, and
those decide crop vs. full-frame, never a similarity score.

**📄 [docs/MATCHING.md](docs/MATCHING.md)** documents every algorithm, every
threshold, the evidence behind each, and the bugs each one caught. Read it before
changing a number.

### The case the design is built around

A *different photograph of the same subject* produces a plausible geometric
alignment (`coverage_orig = 0.94`) and high embedding similarity. What gives it
away is that **73.9% of its blocks differ** once aligned. That single measurement
gates every 1:1 bucket, and it is why the matcher will not call a lookalike a
match.

---

## Results

One page, grouped, 1:1 first:

```
1:1 MATCH CHECK
FOUND 5 TRUE 1:1 MATCHES

EXACT 1:1 MATCHES
─────────────────
[IMAGE]  1:1 MATCH — 100.0% confidence
Found by:  ✓ Google Lens  ✓ Bing  ✓ Yandex
Found on:  example.com  example2.com  example3.com
[Compare]

EDITED / RESIZED 1:1 MATCHES
CROPS                      ← with "Estimated overlap: 36%"
SIMILAR IMAGES
UNRELATED / LOW CONFIDENCE
```

Every card shows the local verdict **next to** what the engine said:

```
Local verification     1:1 MATCH — 99.7%
Search engine said     "Visually similar"
```

**Compare** opens a viewer with side-by-side, overlay (with a blend slider),
difference, and a colourised difference map, plus zoom and fit-to-screen. Both
frames are registered before comparison, so a resize, crop or border does not
show up as a false difference.

Results are grouped under the headings the specification asks for, and identical
content found by several engines collapses into one card listing every engine
and every page it appeared on.

---

## Layout

```
app/
  config.py        every threshold and weight, with its justification
  imaging.py       decode, EXIF orientation, ICC, normalisation
  hashing.py       aHash / dHash / pHash / wHash + colour histogram
  geometry.py      orientation search, convex-polygon coverage
  features.py      ORB + RANSAC similarity transform, template sweep
  descriptor.py    classical descriptor embedding (+ optional CLIP)
  editdetect.py    brightness / contrast / colour / watermark / border / blur
  matcher.py       signals → fusion → classification → confidence
  pipeline.py      the six stages
  compare.py       aligned renders for the comparison viewer
  demo.py          offline result synthesis
  search.py        engine fan-out + downloader
  webapp.py        Flask app
scripts/benchmark.py   the labelled matrix and calibration instrument
docs/MATCHING.md       algorithm and thresholds
tests/                 181 tests
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

| file | covers |
|---|---|
| `test_matcher.py` | the required edge-case matrix, confidence derivation, engine-independence |
| `test_pipeline.py` | staging, filtering, cross-engine merge |
| `test_primitives.py` | hashing, orientation, coverage geometry, border detection |
| `test_webapp.py` | upload → verify → results → comparison, over HTTP |

Fixtures are procedurally generated and seeded. They are *structured scenes*, not
random noise — white noise loses almost all its energy under resampling, so a
noise image would fail a resize test for the wrong reason.

## Environment

```
REV_DATA_DIR   where runs are stored           (default: ./data)
CORPUS_DIR     offline corpus folder           (default: ./data/corpus)
PORT           server port                     (default: 8000)
```
