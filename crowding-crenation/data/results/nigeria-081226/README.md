# Nigeria 081226: combined v2.2 on 8 FOVs — right buckets at the dense end, no ranking; the sparse end needed a gate

8 FOVs from `data/raw/nigeria-081226`, two slides (`HP231668` ×5, `HP245487 R` ×3), scored
with the calibrated **v2.2** params (`density_overlap_v2.2_params.json`, fit on 661 Tanzania
+ initial-dataset FOVs). No recalibration. Produced by `scripts/nigeria_081226.py` and
`scripts/nigeria_081226_sparse_diagnosis.py`.

Manual labels now exist for all 8 FOVs (`data/labels/nigeria-081226/`), so accuracy below is
measured, not inferred. The `_sparse` filename suffix is still not parsed anywhere — see the
note in that directory.

## Headline

Against the manual labels:

| | density | Rouleaux | both axes |
|---|---|---|---|
| v2.2 composite alone | 6/8 | 6/8 | **6/8** |
| v2.2 + empty-field gate | 8/8 | 8/8 | **8/8** |

6/8 (~75%) is in line with v2.2's own cross-validated exact-match of 69.4% / 67.6%, so the
earlier framing of this set — "the scores are not usable" — was wrong, written before the
labels existed. Two things are true at once, and they need separating:

- **The 6 dense FOVs land in the correct bucket but carry no ranking information.** All six
  are manually labeled Very Dense / Heavy Rouleaux, and all six score there. But they score
  there because every feature is pinned at its ceiling, not because the composite resolved
  them: density 0.862–0.956 on a scale whose top bucket starts at 0.711. Being right for the
  wrong reason still means the model cannot tell these six apart, and would not notice one of
  them becoming less dense.
- **The 2 near-empty FOVs failed by 3–4 buckets** — Dense (0.676) and Very Dense (0.736) with
  Heavy Rouleaux, on fields with almost no cells. That is the dangerous mode, and it is what
  the empty-field gate addresses.

`scripts/combined/README.md`'s "Known limitation" predicted the out-of-distribution part of
this ("every candidate feature is a raw pixel/intensity statistic, sensitive to staining
protocol, scanner, and illumination… spot-check before trusting it on a new slide or stain").
This run is that spot-check. The buckets survive it; the *scores* do not.

## Out-of-distribution check

Each FOV's features vs. the 2nd–98th percentile band v2.2 normalizes against (the range
`normalize_matrix` clips to — outside it, a feature is pinned at 0.0 or 1.0 by construction):

| feature | clipped | detail |
|---|---|---|
| `glcm_contrast` | 8/8 | 6 above p98 (max 466.8 = **5.36× the edge**), 2 below p2 (min 5.8 = 0.22×) |
| `edge_density_unmasked` | 8/8 | 6 above p98 (1.15×), 2 below p2 (0.06×) |
| `otsu_separability` | 8/8 | 6 above p98 (1.12×), 2 below p2 (0.84×) |
| `coverage` | 7/8 | 7 above p98 (1.21×) |
| `lbp_entropy` | 7/8 | 5 above p98 (1.01×), 2 below p2 (0.86×) |
| `tile_glcm_cv` | 5/8 | 5 above p98 (1.97×) |
| `saturation_score` | 2/8 | 2 above p98 (1.54×) |
| `tile_glcm_patchiness` | 2/8 | 2 above p98 (2.99×) |

7/8 FOVs are clipped on `coverage`, `glcm_contrast` and `edge_density_unmasked`
simultaneously. `glcm_contrast` at 5.4× the calibration ceiling is the single biggest
departure: these images carry far more high-frequency texture than any Tanzania dpc FOV in
the pool (calibration median 63.7; here 323–467 on the six textured FOVs). **This is why the
6 dense FOVs have no ranking information**: they are all past the ceiling on most features,
so the composite has nothing left to separate them with.

Note the split direction — **two distinct failure modes in one dataset**:

- **6 textured FOVs**: every feature pinned *high* → top bucket on both axes. Correct bucket,
  zero resolution.
- **2 near-empty FOVs**: the texture features fall *below* p2 while the Otsu-derived features
  run high → confidently wrong.

## Why the two visually-empty FOVs scored Dense / Very Dense

`fov-thumbnails-v2.2.png` shows `HP231668_..._3_sparse` and `..._4_sparse` as essentially
blank fields with a few specks. The raw composite gives them **0.676 (Dense)** and **0.736
(Very Dense)**, plus Heavy Rouleaux. Decomposed (`sparse-decomposition-gated.png`,
`sparse-score-decomposition-gated.csv`):

| feature | raw (FOV 3) | normalized | weight | contribution |
|---|---|---|---|---|
| `saturation_score` | 0.176 | 0.988 | 0.247 | **0.244** (36%) |
| `coverage` | 0.351 | 0.761 | 0.236 | **0.179** (27%) |
| `tile_glcm_patchiness` | 2.795 | 1.000 (clipped) | 0.171 | **0.171** (25%) |
| `tile_glcm_cv` | 0.612 | 1.000 (clipped) | 0.082 | **0.082** (12%) |
| `otsu_separability` | 0.498 | 0.000 (clipped low) | 0.014 | 0.000 |
| `lbp_entropy` | 2.747 | 0.000 (clipped low) | 0.065 | 0.000 |
| `glcm_contrast` | 6.883 | 0.000 (clipped low) | 0.087 | 0.000 |
| `edge_density_unmasked` | 0.005 | 0.000 (clipped low) | 0.098 | 0.000 |

Four compounding causes:

1. **Otsu has nothing to separate, so it segments noise.** On an empty field the grayscale
   histogram is unimodal; Otsu splits background noise near its middle and reports
   `coverage` = 0.351 / 0.484 — above the calibration p98 of 0.439, on images with almost no
   cells. The red overlay in `sparse-decomposition-gated.png` shows this directly. A genuinely
   sparse FOV in the calibration pool has median `coverage` = **0.096**.
2. **`saturation_score` fires on empty fields, not just saturated ones.** It is
   `coverage × (1 − otsu_separability)`, and `otsu_separability` is low both when a FOV is too
   dense to separate *and* when there is nothing to separate. It is the single largest
   contributor here (0.244, 36%) and its weight (0.247) is the largest on the density axis.
   Its own calibration partial correlation with Rouleaux is 0.005.
3. **The composite cannot express negative evidence.** `normalize()` clamps to `[0, 1]` and
   `weighted_composite` sums non-negative terms, so the four features that *do* correctly
   read the field as empty — `glcm_contrast` at 0.22× the band floor, `edge_density` at
   0.06×, plus `lbp_entropy` and `otsu_separability` — each contribute **exactly 0.0**,
   indistinguishable from a feature sitting at its calibration minimum. That silences 26.4% of
   the density weight, while the four features that misfire on an empty field carry the
   remaining 73.6%.
4. **`tile_glcm_patchiness` explodes on empty fields.** It is `(max − median) / median` of
   per-tile GLCM contrast; on a near-empty field the median tile contrast is ≈0, so the ratio
   blows up to 2.80 / 2.20 against a band ceiling of 0.934.

Compare the median calibration Sparser FOV (n=50) against these two:

| feature | Sparser median | FOV 3 | FOV 4 |
|---|---|---|---|
| `coverage` | 0.096 | 0.351 | 0.484 |
| `glcm_contrast` | 30.1 | 6.9 | 5.8 |
| `edge_density_unmasked` | 0.069 | 0.005 | 0.003 |
| `tile_glcm_patchiness` | 0.476 | 2.795 | 2.196 |

These FOVs are not merely mis-bucketed — they are *further* from the calibration Sparser
profile than a dense FOV is, in opposite directions on different features.

## The empty-field gate

Cause 3 is not a weighting problem, so reweighting cannot fix it: with no cells in the field,
`coverage` and `saturation_score` are not merely uninformative, they *actively assert* density.
The composite is answering the wrong question, so the fix discards its answer rather than
adjusting it.

**Rule** (`empty_field_override` in `density_overlap_v2.2_params.json`, applied by
`apply_empty_field_override` in `scripts/combined/_v2_common.py`): when **all four** of
`otsu_separability`, `lbp_entropy`, `glcm_contrast`, `edge_density_unmasked` sit strictly
below their calibration p2 floor, return the bottom bucket on both axes — `sparser` /
`no rouleaux` — bypassing the composite.

**It is a measured no-op on in-distribution data.** Over the 661-FOV v2.2 calibration set the
gate fires on 3 FOVs — `dpc-136`, `dpc-137`, `dpc-160` from KTR-72502948 — all three manually
labeled `sparser` + `no rouleaux`, and all three already predicted as such. Exact-match is
unchanged to four decimals: **0.6974 density / 0.6838 Rouleaux, gated and ungated**. Re-run
`python scripts/combined/check_empty_field_gate.py` to reproduce; it fails loudly if a gated
FOV is ever not a truly sparse one.

**The strictness is load-bearing, in both directions:**

- `≥3 of 4` below floor catches 13 calibration FOVs, one of which (`dpc-154`) is a genuine
  `monolayer` — a real false positive. 4-of-4 catches only true sparse fields.
- Adding a safety margin (`0.9 × p2`) stops catching Nigeria FOV 3, whose `otsu_separability`
  is 0.96× the floor — the tightest of the four margins by far.

So: exactly four of four, at exactly p2. The check script prints the near-miss list (FOVs one
feature short of firing) so that margin stays visible rather than implicit.

**Known boundary:** the gate cannot help a *partially* empty field, where enough cells remain
to hold one of the four features above its floor while Otsu still misreads the background.
Nothing in this dataset exercises that case.

## Throughput

Per-FOV cost at 2800×2800, single-threaded (measured, `HP231668_..._1.bmp`):

| step | time | share |
|---|---|---|
| `lbp_entropy` | 5.61 s | **81.6%** |
| `correct_illumination` (301px blur) | 0.72 s | 10.5% |
| `glcm_contrast` (whole image) | 0.13 s | 1.9% |
| `load_image` | 0.14 s | 2.0% |
| `tile_statistics` (7×7 GLCM) | 0.10 s | 1.5% |
| `edge_density_unmasked` | 0.09 s | 1.3% |
| `to_grayscale` / `otsu_segment` / `otsu_separability` / `cell_coverage` | 0.08 s | 1.2% |
| **total** | **6.87 s** | |

`lbp_entropy` dominates because `local_binary_pattern(gray, n_points=24, radius=3)` does ~24
bilinear samples per pixel over 7.84M pixels ≈ 188M interpolated reads, single-threaded in
skimage. Measured across three FOVs it ranges 5.5–7.4 s.

LBP must run at native resolution (downsampling the image changes the texture scale the
operator sees), so the ways to cut it are fewer sample points or fewer pixels via cropping —
both measured against the exact full-image value:

| variant | time | Δ entropy vs. full | speedup |
|---|---|---|---|
| `radius=3, n=24`, full image (current) | 5.5–7.4 s | — | 1× |
| central 1400×1400 crop (25% px) | 1.3–1.9 s | +0.005 / +0.004 / −0.017 | ~4× |
| **central 700×700 crop (6% px)** | **0.33–0.45 s** | **+0.004 / +0.007 / −0.007** | **~16×** |
| 4 windows of 700×700 (25% px) | 1.4–2.1 s | −0.003 / +0.005 / +0.017 | ~4× |
| `radius=1, n=8`, full image | 1.9–2.7 s | not comparable (different scale/bins) | ~3× |

A 700×700 central crop reproduces the full-image entropy to within ±0.007 — against an
`lbp_entropy` normalization span of 1.078 (`[3.118, 4.196]`) and a density weight of 0.065,
that is a worst-case composite shift of ~0.0004, far below the nearest bucket threshold gap.
It would take per-FOV cost from **6.9 s to ~1.7 s (4× overall)**. Not applied here — changing
a shipped feature warrants verifying the delta across all 661 calibration FOVs first, not 3.
Note `lbp_entropy` is one of the four gate features, so any such change must be re-checked
against the gate, not just the composite.

Caveat measured on this machine (16 cores): `cv2.getNumThreads()` is 16, so N worker
processes each spawn 16 OpenCV threads. Worker counts near the core count oversubscribe
badly; `cv2.setNumThreads(1)` inside workers is worth testing alongside any scaling change.

## Files

The `-gated` artifacts are the current ones; the plain `v2.2` files are the earlier ungated
run, kept for comparison. Nothing was recalibrated — the suffix marks a label change, not a
refit.

- `features-v2.2-gated.csv` — full feature vector per FOV, composite scores, pre- and
  post-gate buckets (`raw_density_label` / `density_label`), a `gated` flag, and the manual
  labels alongside.
- `ood-report-v2.2-gated.csv` — per-FOV × per-feature position within the calibration band,
  with a `clipped` flag and `x_over_p98` ratio. Identical to the ungated version: the gate
  changes labels, not features.
- `sparse-score-decomposition-gated.csv` — per-feature raw / normalized / weight /
  contribution for the two near-empty FOVs and one textured FOV, with the calibration Sparser
  median, plus `gate_feature` / `gate_below_floor` / `gate_fired` / `composite_label` /
  `final_label` so the decomposition explains the label that actually ships.
- `quality-grid-v2.2-gated.png` — manual annotation vs. model on the density × Rouleaux grid,
  two point groups in the form `plot_bucket_comparison_v2.py` uses, with connectors on
  disagreements (there are none once gated).
- `feature-ood-v2.2-gated.png` — each of the 8 features: calibration pool distribution, its
  normalization band, and where the Nigeria FOVs fall.
- `fov-thumbnails-v2.2-gated.png` — the 8 FOVs with predicted buckets; gated FOVs show both
  the gate's label and the composite score it overrode.
- `sparse-decomposition-gated.png` — contribution bars for the two near-empty FOVs + the Otsu
  mask overlay that starts the failure, annotated with the gate outcome.
- `scored-v2.2.csv`, `features-v2.2.csv`, `ood-report-v2.2.csv`,
  `sparse-score-decomposition.csv`, `quality-grid-v2.2.png`, `feature-ood-v2.2.png`,
  `fov-thumbnails-v2.2.png`, `sparse-decomposition.png` — the original ungated run.

## Open items

1. **Recalibration is deferred.** Pooling these 8 FOVs into a 669-FOV v2.3 refit would widen
   the bands and could restore some ranking resolution at the dense end — the gate does
   nothing for that, since those 6 FOVs are correctly bucketed and simply unresolvable. The
   refit must report CV restricted to the 661 in-distribution subset, not just the pooled
   average, or degradation from the 6 extreme outliers will hide behind it. Measured in
   advance: pooling moves the four gate floors by at most −0.031 (`lbp_entropy`) and −0.0016
   on the binding one (`otsu_separability`), so the gate survives the refit — but re-run
   `check_empty_field_gate.py` afterwards, and note the mild circularity that the gate's
   thresholds would then be fitted on data containing the FOVs it must catch.
2. **Decide whether these images are even the same modality.** `glcm_contrast` 5.4× the
   calibration ceiling and the fine-grained speckle texture in the thumbnails suggest a
   different imaging path (magnification, phase-contrast reconstruction, or sharpening) from
   the Tanzania dpc PNGs, not just a different stain. Worth confirming before any refit —
   if so, no reweighting of these features will transfer.
3. **Consider the OOD guard as a hard gate**, not just a report: refuse to emit a bucket when
   a FOV is clipped on N+ features, rather than returning a confident-looking "Very Dense".
   The empty-field gate is one narrow instance of this idea; the general form would also cover
   the 6 dense FOVs, which are currently right but unresolvable.
4. **8 FOVs across 2 slides is a spot-check set, not a calibration set.** The labels exist now,
   but they carry little weight in a 669-row ridge fit — expect them to move percentile bands,
   not weights.
