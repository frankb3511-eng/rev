# The 1:1 matching algorithm

This document describes what the matcher actually computes, why each technique was
chosen, and the exact thresholds it uses. Every number here is declared in
[`app/config.py`](../app/config.py) and re-measured by
[`scripts/benchmark.py`](../scripts/benchmark.py).

The design goal is asymmetric on purpose: **missing a 1:1 match is cheap, falsely
claiming one is not.** A user who sees "1:1 MATCH — 99.8%" will act on it. So
every threshold errs towards requiring more evidence, and the single hardest case
in the suite — a *different photograph of the same subject* — is the one the
architecture is built around.

---

## 1. Independence from the search engine

Nothing in `app/matcher.py` reads an engine label, engine score, or result
ordering. `Verdict` exposes no `engine` field at all, and
`tests/test_matcher.py::test_verdict_does_not_depend_on_engine_metadata` asserts
that.

Engine wording is carried through to the UI and displayed *next to* the local
verdict:

```
Local verification     1:1 MATCH — 99.7%
Search engine said     "Visually similar"
```

The demo corpus deliberately attaches `"Visually similar"` to a byte-level
re-encode the matcher proves is 1:1, and `"Exact match"` to a lookalike
photograph the matcher rejects. If engine wording leaked into the verdict those
would come out the wrong way round; `test_webapp.py` checks they do not.

---

## 2. Perceptual hashing

Four 64-bit hashes are computed on a 32×32 grayscale downsample:

| hash | transform | strong against | weight |
|---|---|---|---|
| `phash` | 32×32 DCT → 8×8 low-frequency block, median-thresholded, DC excluded from the median | JPEG, scaling, blur — best overall | 0.45 |
| `dhash` | horizontal gradient sign on a 9×8 window | brightness / gamma change | 0.25 |
| `whash` | Haar DWT approximation coefficients | heavy JPEG; weak under gamma | 0.20 |
| `ahash` | mean threshold on an 8×8 thumbnail | broad perturbations, weakest discrimination | 0.10 |

The weighting follows the controlled comparison in *Electronics* 15(7):1493, which
measured aHash/dHash/pHash/wHash on UKBench and Amazon Berkeley Objects under
identical preprocessing and found pHash the best all-round hash — near-perfect
exact-duplicate metrics *and* the highest robustness of the four to JPEG
recompression, scaling and blur. The same paper reports that a normalised Hamming
similarity `S_hash ≥ 0.93` identifies near-duplicates at high precision on a
64-bit hash, which is where `HASH_NEAR_SIM = 0.93` comes from.

Similarity is normalised Hamming: `s = 1 − hamming/64`.

**Two bugs this caught during development**, both worth recording because they
changed what the thresholds meant:

* aHash was being computed on the 32×32 pre-hash thumbnail, producing a
  **1024-bit** hash. Every 64-bit threshold in the config silently meant
  something else (0.93 of 1024 bits is ~72 bits of slack, not ~4.5).
* Hash similarity was computed on the candidate in its *stored* orientation, so
  an EXIF-rotated copy scored 0.53 on hashes while every pixel metric said the
  frames were identical. Hashes are now recomputed in the orientation the
  matcher selects.

---

## 3. Orientation

Rotations and mirrors destroy global hashes, so a rotated copy would be discarded
before it ever reached the expensive stages. The matcher therefore compares the
original's hash against all **eight rigid orientations** of the candidate and
takes the winner.

A non-identity orientation is only accepted on two conditions:

1. it beats identity by at least `ORIENTATION_MARGIN = 0.08` in hash similarity;
2. the multi-scale template sweep localises *better* in that orientation than in
   identity, by at least `ORIENTATION_GEOMETRY_MARGIN = 0.08`.

Condition 2 exists because condition 1 alone is not enough. Measured on the
fixtures: for a centre crop of a near-symmetric scene, the **mirrored** crop
hashed closer to the original than the unmirrored one did (0.653 vs 0.573) — the
scene is roughly left–right symmetric, so the mirror is a plausible-looking
answer. Accepting it made the crop impossible to localise:

| orientation | template score | reported overlap |
|---|---|---|
| `mirror-h` (hash winner) | 0.664 | 2% (nonsense) |
| `identity` | 0.963 | 26% (true value: 25%) |

A rotation or mirror that survives both tests is reported to the user
(`"rotated 90 degrees"`) and classified as **EDITED 1:1**, not hidden inside
EXACT.

---

## 4. Geometry: coverage, not similarity

Whether two frames cover the same content is answered by geometry, never by a
similarity score. Two numbers are produced:

* `coverage_orig` — share of the **original** present in the candidate
* `coverage_cand` — share of the **candidate** explained by original content

| situation | `coverage_orig` | `coverage_cand` |
|---|---|---|
| resized / re-encoded copy | ≈ 1.0 | ≈ 1.0 |
| candidate is a crop of the original | < 1.0 | ≈ 1.0 |
| candidate is the original plus a border | ≈ 1.0 | < 1.0 |

`coverage_orig` is the number reported to the user as **Estimated overlap** for a
crop. Measured against ground truth on the fixtures: 0.251 vs a true 0.250, and
0.407 vs a true 0.409.

### 4.1 Similarity transform, not homography

The geometric model is `cv2.estimateAffinePartial2D` — rotation, uniform scale
and translation, 4 degrees of freedom — fitted with RANSAC at a 4 px reprojection
tolerance, falling back to a full 8-DOF `findHomography` only if that fails.

A copy of an image on the web differs from the original by a similarity
transform, not a projective one, so the constrained model is the correct prior
and needs only two point pairs. The unconstrained fit was measurably wrong: on a
low-texture scene with 36 inliers it recovered perspective terms of `2.1e-4`, a
**22% warp across the frame**, which smeared the aligned comparison and turned a
clean crop into "76% of blocks differ".

### 4.2 Template sweep

A bidirectional multi-scale template sweep backs the feature pass, because ORB
produces few keypoints on smooth images:

* *original-in-candidate* — the candidate contains the whole original (borders, collages)
* *candidate-in-original* — the candidate is a crop

Matching runs on **mean/std-normalised grayscale**, not Canny edges. Canny was the
obvious choice and measured badly: a one-pixel misalignment between two binary
edge maps collapses the correlation, and the sweep topped out near **0.43 even for
an exact copy**. Normalised grayscale peaks sharply at the correct scale
(**0.963** for the centre crop, **0.9999** for an identical frame).

---

## 5. Alignment before measurement

Every pixel metric is taken on the **aligned** pair. This was the single largest
correctness fix in the project. Measuring a bordered or cropped copy against the
stretched full frame is why those cases originally scored as merely "similar" —
the content was identical, it was just not where the metric was looking.

`align_pair()` chooses, in order:

1. **coextensive** — both coverages ≥ 0.98, so no resampling at all (an exact
   copy must not be penalised by an unnecessary warp);
2. **homography** — the candidate is warped into the original's frame and both
   are cropped to the valid region;
3. **template** — the localised box is cut out and resized;
4. **none** — plain full-frame stretch, with `align_valid_fraction` reported so
   the weakness is visible.

Effect on the fixtures:

| case | SSIM unaligned | SSIM aligned | verdict before | verdict after |
|---|---|---|---|---|
| bordered | 0.70 | 0.99 | SIMILAR (82.7%) | EDITED 1:1 (99.3%) |
| screenshot | 0.74 | 0.99 | SIMILAR (93.8%) | EDITED 1:1 (99.8%) |

---

## 6. The signals

| signal | what it measures | weight |
|---|---|---|
| `hash_sim` | fused perceptual hash similarity, measured on the aligned region when one exists | 0.22 |
| `ssim` | structural similarity on aligned grayscale | 0.22 |
| `grad_corr` | Pearson correlation of Sobel gradient magnitudes — invariant to brightness offset and, up to scale, contrast gain | 0.14 |
| `emb_cos` | cosine similarity of a fixed-length classical descriptor | 0.14 |
| `feat_sim` | ORB + RANSAC evidence | 0.12 |
| `nrmse_sim` | `1 − normalised RMSE` | 0.08 |
| `hist_corr` | colour-histogram correlation | 0.08 |

`identity = Σ wᵢ·sᵢ`.

Signals that survive recompression and resizing dominate raw pixel error, which
is the one metric a lossy re-encode destroys.

### 6.1 Why `feat_sim` scales with inlier *count*

Ratio alone was too generous: **12 inliers out of 12 good matches** between two
unrelated images scored a perfect 1.0, because the denominator shrank with the
evidence. `feat_sim` is now `min(1, inliers/40) × min(1, ratio/0.45)`, and
`min_inliers` is 15.

### 6.2 Why `hash_sim` uses the aligned region

Global hashes cannot survive a crop or a border. A bordered copy scored **0.69**
hash similarity and dropped to **88.7%** confidence despite only 0.1% of its
blocks differing. When a trustworthy alignment exists, the hashes are recomputed
on the aligned content and whichever is stronger is used. The bordered case went
to **99.3%**; the lookalike photograph was unaffected at 0.714 identity.

### 6.3 The embedding

The default embedding is a classical fixed-length descriptor built locally, with
no model download and no network access: per-cell colour moments (mean/std/skew)
on a 4×4 grid, per-cell gradient-orientation histograms (HOG-style, 9 bins), a
64-bin global RGB histogram, and per-cell edge density — L2-normalised, compared
by cosine.

It is deliberately *layout-sensitive*. A semantic CNN embedding would score two
different photographs of one subject very highly, which is precisely the false
positive to avoid. A `torch`/`open_clip` backend exists and is used automatically
only if `allow_clip` is enabled and the packages import; the backend actually
used is reported in every result.

---

## 7. Edit attribution

`app/editdetect.py` decomposes the residual difference into causes:

| cause | test |
|---|---|
| `brightness` | least-squares `cand ≈ a·orig + b`, offset ≥ 5.0 with R² ≥ 0.90 |
| `contrast` | gain outside [0.90, 1.11] with R² ≥ 0.90 |
| `colour` | per-channel mean shift ≥ 4.0 with R² ≥ 0.90 |
| `blur` / `sharpen` | Laplacian-variance ratio outside [0.72, 1/0.72], gated on R² ≥ 0.90 |
| `watermark` | a small localised cluster of changed blocks on an otherwise pixel-identical frame |
| `border` | flat bands whose removal brings the aspect ratio back to the original's |
| `recompress` | high-frequency residual only, R² ≥ 0.985, everything else clean |

Three of these were wrong in ways worth recording:

* **Border detection ran on the non-uniformly stretched comparison canvases.**
  The stretch is exactly what erases the aspect-ratio evidence the detector
  reads: both canvases were squares, so the "inner aspect moves towards the
  original" test could never fire, and a genuine border was reported as none. It
  now runs on aspect-preserving copies.
* **Bands were searched independently.** The aspect test then removed only one
  pair of bands, so the inner rectangle never matched. Horizontal and vertical
  thicknesses are now searched jointly.
* **Thin watermarks need a peak statistic, not a mean.** A text-only watermark
  changes few pixels, so the block *mean* stays small. Using the 95th percentile
  of |delta| within each block separates it cleanly — measured peak **69.7** for
  the watermark against **≤ 10.4** for resized, heavily-recompressed, brightened
  and recoloured copies, with zero false blocks. It is gated on
  `changed_fraction ≤ 4%`, because without that gate the peaks are just
  misregistration on crops and unrelated images.

Edit labels are only reported for 1:1 verdicts. "Brightness +26" is a meaningful
statement about an image that *is* otherwise this image; on a crop or an
unrelated photo the same numbers are measurement noise.

---

## 8. Classification

```
sha256 identical                                    -> EXACT 1:1, confidence 100

coverage_orig < 0.96  and  coverage_cand >= 0.75
    and aligned SSIM >= 0.80  and  changed_fraction <= 12%   -> CROP
                                                              overlap = coverage_orig

coverage_orig >= 0.90  and  identity >= 0.84  and  changed_fraction <= 12%
    edits present, or coverage_cand < 0.96        -> EDITED 1:1
    identity >= 0.92 and changed_fraction <= 4%   -> EXACT 1:1
    otherwise                                     -> NEAR 1:1

identity >= 0.55 and SSIM >= 0.45                 -> VISUALLY SIMILAR
otherwise                                         -> UNRELATED
```

### The false-positive guard

`changed_fraction` gates **every** 1:1 bucket. A second photograph of the same
subject passes the geometry question — it produces a plausible alignment with
`coverage_orig = 0.94` — and fails the fidelity one: **73.9% of its blocks
differ** after alignment. That single number is what keeps it out of the 1:1
buckets.

---

## 9. Confidence

Confidence is a logistic over the **margin** by which the chosen branch cleared
its own boundary:

```
margin     = (identity − threshold) / (1 − threshold)
confidence = 100 / (1 + exp(−7.0 · margin))
```

The steepness constant is `CLASSIFIER.confidence_steepness`. Nothing is a
hand-picked percentage; `test_confidence_is_monotone_in_identity_within_a_bucket`
asserts that a stronger fused identity never yields a weaker confidence.

Byte-identical files are pinned at exactly 100.0 by the SHA-256 branch.

Confidence is **confidence in the assigned label**, not a global ranking. A
`SIMILAR IMAGE — 92.8%` and an `EDITED 1:1 — 88.7%` are not directly comparable;
they answer different questions and are shown in different sections.
`test_verified_copies_out_confidence_lookalikes` is a *measurement* of the
calibrated system on the fixture suite, written to catch threshold drift — not a
structural guarantee.

---

## 10. Staged pipeline

```
STAGE 1  URL + content-hash deduplication        no pixel work
STAGE 2  fast perceptual hashing                 32x32 downsample
STAGE 3  candidate filtering                     cheap gate
STAGE 4  pixel + structural comparison           provisional verdict
STAGE 5  feature / embedding verification        promising candidates only
STAGE 6  cross-engine merge
```

Stage 4 emits a provisional verdict. Anything with
`identity ≥ identity_similar × 0.85` is promoted to stage 5; everything else is
finalised as unrelated and never pays for ORB or the template sweep.
`test_unrelated_candidates_are_not_promoted_to_stage_5` asserts this on eight
unrelated images.

Measured on a 16-candidate run:

| stage | ms |
|---|---|
| 1 URL dedup | 5 |
| 2 download + hash | 118 |
| 3 filter | < 1 |
| 4 compare (16 candidates) | 2237 |
| 5 verify (13 candidates) | 2762 |
| 6 merge | 443 |
| **total** | **5578** |

Full deep verification of one candidate costs roughly **245 ms**.

---

## 11. Merging duplicate results

Two results become one card only when **all** of these hold:

1. the matcher gave them the **same label**;
2. their pairwise `identity ≥ 0.92`;
3. their pairwise `changed_fraction ≤ 4%`;
4. no material edit separates them.

Pairwise comparison uses `quick_identity()` — hashes plus SSIM, gradients and
edit attribution at 112 px, with no orientation search, no ORB and no template
sweep.

Both restrictions were earned:

* Using the full matcher here cost **7016 ms** for 66 pairs (58% of the whole
  run) and was *wrong*: it applied orientation correction, so a rotated copy and
  a brightened copy looked identical and merged into one card. With
  `quick_identity` the same stage costs **443 ms** — a 15.8× improvement — and
  keeps distinct renderings apart.
* Condition 3 alone was not enough, because `changed_fraction` is measured
  *after* the gain/offset fit, so a pure brightness edit leaves it near zero.
  Condition 4 (attributed causes) is what separates a brightened copy from a
  clean one.

Condition 1 also covers the merge's own resolution limits: a faint text-only
watermark on a 640×480 image is not reliably re-detected at 112 px, but the
matcher has already labelled that candidate EDITED while labelling a clean
re-encode EXACT, so the two cannot merge.

**Known limitation.** At 640×480 that same faint watermark is faint enough that
the *matcher* also classifies it as EXACT rather than EDITED — the fixture
watermark (text **plus** a box outline) is detected at 2.1% changed blocks, while
the demo's text-only overlay is not. Both are correctly reported as 1:1 matches,
which is the part that matters; only the edited/exact distinction between them is
lost.

---

## 12. Measured results

`scripts/benchmark.py` output on the 25-case labelled matrix:

| case | expected | verdict | conf | identity | cov O | cov C | chg% |
|---|---|---|---|---|---|---|---|
| identical_bytes | EXACT | EXACT_1TO1 | 100.0 | 1.000 | 1.00 | 1.00 | 0.0 |
| renamed_identical | EXACT | EXACT_1TO1 | 100.0 | 1.000 | 1.00 | 1.00 | 0.0 |
| metadata_stripped | EXACT | EXACT_1TO1 | 100.0 | 0.994 | 1.00 | 1.00 | 0.0 |
| metadata_added | EXACT | EXACT_1TO1 | 100.0 | 0.995 | 1.00 | 1.00 | 0.0 |
| resized_down_half | EXACT | EXACT_1TO1 | 100.0 | 0.994 | 1.00 | 1.00 | 0.0 |
| resized_up_double | EXACT | EXACT_1TO1 | 100.0 | 0.996 | 1.00 | 1.00 | 0.0 |
| jpeg_recompressed | EXACT | EXACT_1TO1 | 100.0 | 0.994 | 1.00 | 1.00 | 0.0 |
| png_to_jpeg | EXACT | EXACT_1TO1 | 100.0 | 0.993 | 1.00 | 1.00 | 0.0 |
| jpeg_to_webp | EXACT | EXACT_1TO1 | 99.9 | 0.989 | 1.00 | 1.00 | 0.0 |
| heavily_compressed | EXACT | EXACT_1TO1 | 99.7 | 0.974 | 1.00 | 1.00 | 0.0 |
| brightness_up | EDITED | EDITED_1TO1 | 99.5 | 0.960 | 1.00 | 1.00 | 0.0 |
| brightness_down | EDITED | EDITED_1TO1 | 98.9 | 0.943 | 1.00 | 1.00 | 0.0 |
| contrast_up | EDITED | EDITED_1TO1 | 99.7 | 0.972 | 1.00 | 1.00 | 0.0 |
| colour_shift | EDITED | EDITED_1TO1 | 99.8 | 0.978 | 1.00 | 1.00 | 0.0 |
| watermarked | EDITED | EDITED_1TO1 | 99.8 | 0.983 | 1.00 | 1.00 | 2.1 |
| bordered | EDITED | EDITED_1TO1 | 99.3 | 0.955 | 1.00 | 0.82 | 0.1 |
| screenshot | EDITED | EDITED_1TO1 | 99.8 | 0.979 | 1.00 | 0.85 | 0.0 |
| blurred | EDITED | EDITED_1TO1 | 99.7 | 0.972 | 1.00 | 1.00 | 3.4 |
| crop_centre | CROP | CROP | 99.9 | 0.901 | **0.25** | 1.00 | 10.9 |
| crop_corner | CROP | CROP | 99.9 | 0.896 | **0.41** | 1.00 | 7.6 |
| rotated_90 | EDITED | EDITED_1TO1 | 99.9 | 1.000 | 1.00 | 1.00 | 0.0 |
| rotated_180 | EDITED | EDITED_1TO1 | 99.9 | 1.000 | 1.00 | 1.00 | 0.0 |
| mirrored | EDITED | EDITED_1TO1 | 99.9 | 1.000 | 1.00 | 1.00 | 0.0 |
| different_photo_same_subject | SIMILAR | VISUALLY_SIMILAR | 92.7 | 0.714 | 0.94 | 1.00 | **73.9** |
| unrelated_image | UNRELATED | UNRELATED | 1.3 | 0.274 | 0.00 | 0.00 | 100.0 |

**25/25.** Reported crop overlaps against ground truth: 0.251 vs 0.250, and
0.407 vs 0.409.

Reproduce with:

```bash
.venv/bin/python scripts/benchmark.py
.venv/bin/python -m pytest -q          # 181 tests
```
