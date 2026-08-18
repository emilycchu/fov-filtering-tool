# Combined v2 density/Rouleaux pipeline

This directory holds the v2 calibrated pipeline (`merge_labels_v2.py` ->
`extract_features_v2.py` -> `calibrate_v2.py`/`calibrate_v2.1.py`/`calibrate_v2.2.py` ->
`score_fov_v2.py`) — fits density and Rouleaux composite scores against a growing pool of
manually-labeled FOVs, with fixed thresholds that generalize to new slides without
recalibration. This is the main subject of this README.

It's named "combined" because both the density and Rouleaux axes are fit from one shared
feature vector (`_v2_common.compute_features`) rather than separate pipelines — see
"Feature vector" below.

A sibling tool, `scripts/ai-first/score_new_slide.py` + `scripts/ai-first/label_new_slide.py`,
is an earlier, from-scratch classical-CV (watershed) pipeline that never touches a manual
label; it derives density/crowding labels from *that slide's own* quintiles. Superseded by
the v2 pipeline here for anything that needs to generalize across slides, but kept because
it needs no calibration set at all. See its own docstrings for how it works — not covered
further here.

## Overview: what v2 does

Given a raw FOV image, `score_fov_v2.py`:

1. Computes a fixed vector of classical CV features from the image (`_v2_common.compute_features`).
2. Combines a per-axis subset of those features into two independent `[0, 1]` composite
   scores — one for **density**, one for **Rouleaux** (cell overlap/stacking) — via a
   fitted weighted average (`src/composite_v2.py::weighted_composite`).
3. Buckets each composite score into one of 5 ordinal labels using fixed thresholds derived
   by calibration (`src/composite_v2.py::bucket`).

The weights, per-feature normalization ranges, and bucket thresholds are not hand-tuned —
they come from `calibrate_v2*.py` regressing those same features against a pool of manually
labeled FOVs, and are saved to a `density_overlap_v2*_params.json` file that `score_fov_v2.py`
loads at inference time. This is the key difference from the classical `src/composite.py`
pipeline in the repo root README: there, weights are hand-adjusted from correlation analysis;
here, they're fit.

```bash
python scripts/combined/score_fov_v2.py data/raw/some-dataset --params data/results/density-rouleaux-v2/density_overlap_v2.2_params.json --out-csv out.csv
```

## Feature vector (`_v2_common.compute_features`)

Every FOV image (regardless of density or Rouleaux) is reduced to this fixed set of
candidate features — the same function is imported by both calibration
(`extract_features_v2.py`) and inference (`score_fov_v2.py`), so the two can never compute
features differently:

| Feature | What it measures |
|---|---|
| `coverage` | Fraction of Otsu-masked foreground pixels (`src/segmentation.py::cell_coverage`) |
| `otsu_separability` | How cleanly Otsu's threshold splits the grayscale histogram into two populations (`src/features/otsu_separability.py`) — recomputes Otsu's own between-class/total-variance ratio (eta); eta -> 0 means a FOV so densely packed there's no real background left to separate from |
| `saturation_score` | `coverage * (1 - otsu_separability)`, clipped to [0, 1] — high when the image is both mostly-foreground *and* has a poorly-separated histogram, i.e. plausibly saturated/overcrowded past the point Otsu can even describe it. Currently informational only (see `apply_saturation_override` below) |
| `lbp_entropy` | Shannon entropy of the local-binary-pattern histogram (`src/features/lbp_entropy.py`) |
| `glcm_contrast` | Mean GLCM contrast across 4 angles, whole image (`src/features/glcm_contrast.py`) |
| `edge_density_unmasked` | Fraction of pixels on a Canny edge, unmasked (`src/features/edge_density.py`) |
| `tile_glcm_cv` | Coefficient of variation of per-tile GLCM contrast across a 7x7 grid (`src/features/tile_heterogeneity.py`) — a single whole-image contrast scalar can't tell a uniformly dense FOV from a patchy one; tiling exposes that variation. Computed on illumination-corrected grayscale (`correct_illumination`, 301px blur) so large-scale brightness gradients don't get read as tile-to-tile "heterogeneity" |
| `tile_glcm_patchiness` | `(max - median) / median` of the same per-tile GLCM contrast array — catches a small number of outlier tiles (e.g. one localized Rouleaux cluster) that barely move the mean/CV |

`otsu_threshold` is also recorded (for diagnostics) but is not a candidate feature.

Note the density/Rouleaux axes are **not** computed from distinct pipelines — both draw
from this one shared feature vector; what differs between axes is *which* features get
weighted in, and by how much (see below).

## Calibration data

`merge_labels_v2.py` pools every available manually-labeled dataset into one CSV
(`data/results/density-rouleaux-v2/merged-labels.csv`), each row a FOV with `density_label`
and `overlap_label` (displayed to the user as "Rouleaux", but named `overlap` internally to
match the source CSV columns and repo convention) on a shared 5-level ordinal scale:

- Density: `sparser`, `monolayer`, `slightly dense`, `dense`, `very dense`
- Rouleaux: `no rouleaux`, `slight rouleaux`, `some rouleaux`, `rouleaux`, `heavy rouleaux`

Sources merged (661 FOVs total, as of v2.2):

- `initial-dataset-071626` — 13 FOVs, clean `density`/`overlap` label columns.
- `tanzania-073026` (KTR-72502948) — 324 FOVs, free-text `tags` column parsed by
  `parse_tanzania_tags`.
- `tanzania-080526` (KTR-72502946) — 324 FOVs, same free-text format, images streamed
  directly from `gs://tanzania_02032026/TZ2025-Box5/KTR-72502946/` and never downloaded
  locally (see `data/results/tanzania-080526/README.md`).

`extract_features_v2.py` then runs `compute_features()` over every row's image (parallelized
via `multiprocessing.Pool`) and joins the result onto the label columns, producing
`features.csv` (or `features-v2.2.csv` for the 661-row pool) — the input to calibration.

## Calibration approach (`calibrate_v2.py` / `.1` / `.2`)

All three scripts share the same fitting machinery in `calibrate_v2.py`; `.1` and `.2` are
successive recalibrations (see "Version history" below) that reuse it wholesale and only
change *what* feeds in.

### 1. Feature selection (marginal + partial correlation)

Density and Rouleaux severity are themselves correlated in the manual labels (denser slides
tend to show more Rouleaux) — a plain marginal correlation of a feature against, say,
density can't tell whether that feature genuinely tracks density or is just riding the
density/Rouleaux confound. `correlation_table()` computes each candidate feature's Spearman
rho against both axes, plus its **partial** correlation with each axis controlling for the
other (`tanzania_comparison.partial_spearman`). `calibrate_v2.py` (the original v2) then
assigns each feature to whichever axis has the higher partial correlation, provided it
clears `MIN_PARTIAL_RHO = 0.05`; otherwise the feature is excluded from both composites.

This axis-exclusive selection turned out to be too strict for Rouleaux (see "v2.1" below),
so `calibrate_v2.1.py`/`calibrate_v2.2.py` skip the exclusive assignment and fit each axis
against the **full 8-feature candidate pool** instead, letting the regression step's own
sign-instability dropping (next section) do the real selection.

### 2. Ridge regression weight fitting

For a given axis and its candidate feature set, `fit_weights_stable`:

1. Percentile-normalizes each feature to its 2nd-98th percentile range across the
   calibration set (`percentile_ranges` / `normalize_matrix`) — robust to outliers, unlike a
   plain min-max.
2. Fits ridge regression (`fit_ridge`, closed-form, `alpha=10.0`) of the normalized feature
   matrix against the axis's ordinal label (0-4).
3. If any fitted coefficient comes out negative — despite the feature having been
   pre-selected for *positive* partial correlation with this axis — that's treated as a
   multicollinearity artifact (e.g. `tile_glcm_cv`/`tile_glcm_patchiness` are correlated at
   rho=0.63, both derived from the same per-tile array, and can flip each other's sign at
   low regularization), not a real inverse relationship. That feature is dropped and the fit
   re-run on the remaining features, iterating until all coefficients are non-negative.
4. The surviving coefficients are normalized to sum to 1 (`|coef| / sum(|coef|)`), becoming
   the composite's weights — so `weighted_composite()` at inference time is a convex
   combination of normalized features, always in `[0, 1]`.

`RIDGE_ALPHA = 10.0` was chosen empirically against this calibration set as the smallest
value that keeps correlated feature pairs' coefficients stably non-negative, so they
contribute jointly rather than one getting dropped by the sign-instability loop above.

### 3. Cross-validation

`cross_validate()` runs 5-fold CV (`N_FOLDS = 5`), stratified by ordinal label
(`stratified_folds` — shuffles within each label level before splitting, so every fold sees
every bucket) so a bucket with few examples isn't accidentally absent from a fold. Each fold
refits weights and PAVA thresholds (below) on the training rows only, then scores the held-out
rows — producing out-of-fold predictions used for the exact-match rate, off-by-one rate, and
confusion matrix reported for each axis. The final weights/thresholds shipped in the params
JSON are refit once more on the **full** calibration set (CV is for reporting expected
generalization, not for selecting the deployed fit).

### 4. PAVA-monotonic bucket thresholds

A raw composite score is continuous; `derive_thresholds()` turns it into 5 ordinal buckets:

1. Compute the median raw score within each manually-labeled level (`medians`).
2. These medians should increase monotonically with severity level, but with limited data
   per level they sometimes don't (e.g. "Some Rouleaux" scoring higher than "Rouleaux" on a
   composite that happens to conflate two different visual patterns). **PAVA**
   (pool-adjacent-violators algorithm, `_pava_merge`) enforces monotonicity by merging any
   adjacent violating levels into one weighted-average block, repeating until the whole
   sequence is non-decreasing.
3. Any level with too few (or zero) FOVs to have a stable median inherits its nearest
   neighbor's corrected value.
4. The final per-level bucket threshold is the midpoint between each pair of adjacent
   corrected medians — these are the cut points `score_fov_v2.py`'s `bucket()` applies to a
   new image's raw score.

When PAVA has to merge levels, that's reported as a `merged_bucket_groups` finding in
`calibration-report.md` — an honest signal that those buckets aren't yet cleanly separable
by the current features/sample size, not a fitting bug.

`bootstrap_median_ci` additionally bootstraps a 90% CI on each bucket's median raw score
(1000 resamples) so the report shows how much sampling noise is in each bucket's centroid.

### 5. Axis-separation check

Since density and Rouleaux are correlated in the labels, a sanity check confirms the two
*fitted* composites aren't just measuring the same thing twice: among FOVs where manual
density-rank and Rouleaux-rank disagree by at least `min_delta` levels,
`axis_separation_check()` tests whether the out-of-fold predicted score deltas diverge in
the same direction, at a better-than-chance rate (one-sided binomial test vs. 0.5, plus
Spearman rho between predicted and manual deltas).

### Output artifacts

- `density_overlap_v2*_params.json` — the deployed params (`feature_names`, `weights`,
  per-feature normalization `min`/`max`, `bucket_thresholds`, `bucket_labels`) that
  `score_fov_v2.py` loads.
- `calibration-report.md` — correlation tables, fitted weights, CV numbers, confusion
  matrices, bucket thresholds/CIs, PAVA merge notes, axis-separation results, and known
  cross-slide/cross-stain generalization caveats. `calibrate_v2.1.py`/`calibrate_v2.2.py`
  *append* a new dated section rather than overwriting it, so the report is a running
  history of every recalibration.
- `plots/` (via `plot_results_v2.py`, `plot_bucket_comparison_v2.py`) — jittered
  density/Rouleaux scatter plots and a manual-vs-model bucket-comparison grid, generated
  directly from `features.csv` + the params JSON (same scoring functions as
  `score_fov_v2.py`, so the plots always match what the tool would actually output).

## Version history

- **v2** (`calibrate_v2.py`) — first fit, 337 FOVs (13 initial + 324 tanzania-073026),
  axis-exclusive partial-correlation feature selection. Rouleaux ended up with only 2
  features (`tile_glcm_cv`, `tile_glcm_patchiness`), which had an inverted-U relationship
  with true Rouleaux severity — a severe, confluent Rouleaux sheet reads as *more*
  homogeneous than a moderate patchy case — forcing PAVA to merge the top 3 Rouleaux
  buckets.
- **v2.1** (`calibrate_v2.1.py`) — same 337 FOVs, but fits each axis against the full
  8-feature candidate pool instead of an axis-exclusive subset, letting `glcm_contrast` and
  `edge_density_unmasked` (excluded from Rouleaux in v2 on partial-correlation grounds) back
  into the Rouleaux composite, where they're cleanly monotonic across the previously-merged
  range. Trade-off: the two composites become more correlated with each other than the true
  manual density-vs-Rouleaux label correlation, since they now share more features.
- **v2.2** (`calibrate_v2.2.py`) — same full-feature-pool fitting as v2.1, pooling in a
  second Tanzania slide (`tanzania-080526`/KTR-72502946, 324 more FOVs) to grow the
  calibration set to 661, primarily to stress-test the Sparser/Monolayer boundary (thin in
  the original single-slide set: 44 Sparser examples against 241 Monolayer). Both axes'
  cross-validated rho improved after pooling (density 0.705->0.783, Rouleaux 0.620->0.737).
  See `data/results/tanzania-080526/README.md` for the full held-out-vs-pooled comparison.

**Known limitation (all versions):** every candidate feature is a raw pixel/intensity
statistic, sensitive to staining protocol, scanner, and illumination — not just true cell
density. Calibration is validated mainly on Tanzania-stain slides (only 13/661 FOVs are
non-Tanzania); spot-check `score_fov_v2.py` output against a handful of manual labels before
trusting it on a new slide or stain, and expect to refit if there's a systematic offset.
`saturation_score` is computed but not yet wired into a hard override
(`apply_saturation_override` is a documented no-op pending a fitted cutoff) — the project
decision so far is to keep it data-driven rather than a hard rule.

**Empty-field gate (v2.2 onward):** the one place a hard override *is* enabled. When all four
of `otsu_separability`, `lbp_entropy`, `glcm_contrast` and `edge_density_unmasked` fall below
their calibration p2 floor, `apply_empty_field_override` returns the bottom bucket on both
axes and the composite is discarded. This is not a tuning choice: on a field with no cells,
Otsu has no bimodal histogram to split, so `coverage` and `saturation_score` read background
noise as dense tissue while the four features that know better are clipped to 0 by
`normalize()` — no reweighting can fix a composite that is answering the wrong question.

It is a measured no-op in-distribution: over the 661-FOV v2.2 set it fires on 3 FOVs, all
manually labeled `sparser` + `no rouleaux` and already predicted as such, leaving exact-match
unchanged at 0.6974 / 0.6838. Run `check_empty_field_gate.py` to reproduce that, and re-run it
after any recalibration — the floors move with the fit. Requiring all four is load-bearing:
3-of-4 picks up a genuinely-`monolayer` FOV. Configured per-params-file as
`empty_field_override`; `write_params_json` emits it automatically, disabled if feature
selection dropped one of the four (as the original 5-feature v2 fit did). See
`data/results/nigeria-081226/README.md` for the failure that motivated it.

## Repository layout (this directory)

```
_v2_common.py            shared constants, label parsing, IO, compute_features() (the
                          single source of truth for the feature vector)
merge_labels_v2.py        pool manual label sources -> merged-labels.csv
extract_features_v2.py    compute_features() over the merged set -> features.csv
calibrate_v2.py            v2: axis-exclusive partial-correlation selection + ridge + PAVA
calibrate_v2.1.py          v2.1: full-feature-pool refit, same 337 FOVs
calibrate_v2.2.py          v2.2: full-feature-pool refit, pooled to 661 FOVs
calibrate_v2.2-optimized.py  v2.2 refit on stride-16 LBP entropy + 4x-downsampled
                           illumination background (6.3x faster per FOV, ~12x vs skimage LBP)
check_empty_field_gate.py  assert the empty-field gate is a no-op on the calibration set
score_fov_v2.py            inference: score a new image/directory with a params JSON
plot_results_v2.py         density/Rouleaux/density-vs-Rouleaux scatter plots
plot_bucket_comparison_v2.py  manual-vs-model bucket-grid comparison plot

```

The LBP runtime study lives in its own subdirectory, since none of it runs as part of
scoring — it is the evidence behind the stride, not a step in the pipeline:

```
lbp-optimization/bench_lbp.py               assert the fast kernel is bit-identical to
                                            skimage; time it (--full-set checks all 661)
lbp-optimization/extract_lbp_variants.py    entropy at every candidate stride, one pass
lbp-optimization/build_variant_features.py  patch only the lbp_entropy column per variant
lbp-optimization/compare_lbp_variants.py    fixed-params + refit comparison vs. v2.2
lbp-optimization/plot_lbp_variants.py       the runtime/accuracy tradeoff figure
```

`calibrate_v2.2-optimized.py` deliberately stays above, next to the other
`calibrate_v2*.py` scripts: it is a shipped fit rather than part of the study.

## LBP runtime (`bench_lbp.py` and friends)

`lbp_entropy` was 82% of `compute_features`' cost. `src/features/lbp_entropy.py` now computes
it as tiled, threaded numpy instead of skimage's per-pixel Cython loop: **2.6x faster and
bit-identical**, which `bench_lbp.py` asserts (add `--full-set` to check all 661 calibration
rows against `features-v2.2.csv`). Nothing about the fit or the feature vector changed.

The same kernel takes a `step` argument that subsamples the *centre* grid — 71x at stride-16,
with zero label changes across all 661 FOVs. Dropping the feature entirely also costs nothing
in-distribution, but disables the empty-field gate (Nigeria 8/8 → 6/8), so it was **not**
adopted: `EMPTY_FIELD_FEATURES` still lists all four features. The measurements, the
per-variant params JSONs, and the reasoning live in `data/results/lbp-runtime/README.md`.

### The two runtime knobs, and v2.2-optimized

`compute_features(image, lbp_step=1, blur_downsample=1)` takes both, and **`lbp_step=16` plus
`blur_downsample=4` is calibrated as `v2.2-optimized`** — same 661 FOVs, same procedure:

| | v2.2 | v2.2-optimized |
|---|---|---|
| `compute_features` per FOV | 3.11 s | **0.49 s** (6.3x) |
| — against the original v2.2 (skimage LBP) | ~5.9 s | **~12x** |
| density OOF exact / off-by-one | 69.4% / 98.0% | 69.4% / 98.0% |
| Rouleaux OOF exact / off-by-one | 67.6% / 93.8% | 67.6% / 93.8% |
| density CV mean rho | 0.783 | 0.783 |
| composite independence rho | 0.972 | 0.972 |
| label differences vs v2.2 | — | 1 of 661 density, 0 Rouleaux |
| empty-field gate | 3 FOVs, no-op | 3 FOVs, no-op |

Each knob moves exactly one thing, which is what makes them auditable:

| knob | what it changes | what it does not |
|---|---|---|
| `lbp_step` | `lbp_entropy` only | everything else, bit-for-bit |
| `blur_downsample` | `tile_glcm_cv`, `tile_glcm_patchiness` only | the other seven, bit-for-bit |

Both were verified that way on all 661 FOVs: re-extracting with `--lbp-step 16` left the other
eight columns byte-identical, and adding `--blur-downsample 4` left seven of nine untouched.

**Neither is a free knob.** They are the only parameters that can break this module's
"calibration and inference can never differ" guarantee, because a fit made on subsampled
features has to be *scored* on subsampled features. So the fit records them, and every
inference path reads them back with `lbp_step_from_params()` /
`blur_downsample_from_params()` — never hardcoded at a call site. Both default to 1, so
v2/v2.1/v2.2 params, which have neither key, score at full resolution exactly as before.
`extract_features_v2.py`'s `--lbp-step` / `--blur-downsample` default to 1 for the same reason,
so re-running it still reproduces the older feature CSVs.

#### The illumination blur, in detail

`correct_illumination(gray, blur_ksize=301, downsample=N)` estimates the illumination background
with a 301-px Gaussian and subtracts it, and only `tile_glcm_cv` / `tile_glcm_patchiness` consume
the result. A 301-px Gaussian is a low-pass filter by construction, so estimating it on a shrunken
copy loses almost nothing — the same reasoning that justifies striding the LBP centre grid.

**It became the bottleneck because the LBP work removed the previous one.** Once `lbp_entropy`
dropped from 4.80 s to 0.07 s, the 301-px blur was 0.65–0.75 s of a ~1.1 s feature vector, i.e.
**55–71% of everything that was left.**

The 661-FOV sweep (`sweep_blur_downsample.py`), scored with the v2.2-optimized params unchanged so
any label change is attributable to the blur alone:

| `blur_downsample` | `correct_illumination` | stage speedup | max drift, patchiness | density flips | Rouleaux flips | gate check |
|---|---|---|---|---|---|---|
| 1 | 0.773 s | 1.0x | 0 | 0 | 0 | pass |
| **2** | 0.198 s | 3.9x | **0.0616** | 0 | **4** | **FAIL** |
| **4 (adopted)** | **0.121 s** | **6.4x** | 0.0044 | **0** | **0** | **pass** |
| 8 | 0.123 s | 6.3x | 0.0070 | 0 | 0 | pass |
| 16 | 0.131 s | 5.9x | 0.0176 | 0 | 0 | pass |

**Why 6.4x and not 38x.** The blur *itself* goes 0.531 s → 0.014 s. But `correct_illumination` also
does `gray.astype(float32) - background + mean`, then clips and casts back — about **0.11 s of fixed
cost over 7.84M pixels regardless of downsample**. That tail is the floor for this stage, and it is
why 8 and 16 are no faster while drifting more. Reducing it would mean changing the arithmetic
(int16, or `cv2.subtract` with saturation), which changes output.

**A second benefit, easy to miss: it makes per-FOV cost nearly thread-independent.** The full-resolution
blur is OpenCV-multithreaded, so its cost swung 2.9x with `cv2.getNumThreads()` — and a batch pass
that fans out over processes has to pin OpenCV to one thread or oversubscribe. Measured on one
2800x2800 FOV at `lbp_step=16`:

| `cv2` threads | blur, ds=1 | blur, ds=4 | `compute_features`, ds=1 | `compute_features`, ds=4 |
|---|---|---|---|---|
| 1 | 2.404 s | 0.159 s | 2.790 s | **0.573 s** |
| 4 | 1.615 s | 0.119 s | 2.184 s | 0.536 s |
| 16 | 0.820 s | 0.107 s | 1.110 s | **0.491 s** |

So at `ds=1` the answer to "how long does a FOV take" depended **2.5x** on pool shape, which is
exactly what misleads batch sizing; at `ds=4` the spread is **1.17x**. The stage speedup is also
larger than the 6.4x headline when threads are pinned — 2.404 s → 0.159 s is **15x** — because the
headline was measured at default threading, where the full-res blur was already getting 16 threads.

#### So how much time did the blur actually save?

Per FOV, holding `lbp_step=16` fixed and changing only `blur_downsample` 1 → 4:

| | `compute_features`, ds=1 | ds=4 | **saved per FOV** |
|---|---|---|---|
| OpenCV free to use 16 threads | 1.110 s | 0.491 s | **0.619 s** (2.3x) |
| OpenCV pinned to 1 thread | 2.790 s | 0.573 s | **2.217 s** (4.9x) |

Two numbers because the full-resolution blur is thread-hungry and the optimized one is not, so the
saving depends on how much OpenCV threading the pass can actually get — and a batch pass that fans
out over processes gets much less than 16 threads each.

Over the 88,123 FOVs of the Tanzania cohort run, that is **15 core-hours saved at best case and
54 core-hours at worst**. For scale, the crowding pass as it actually ran cost about **17
core-hours** in total (88,123 x ~0.68 core-s of fetch + decode + features) — so without this one
change the pass would have cost roughly **2x to 4x the compute it did**.

**In wall clock on the run that actually happened:** the crowding pass took **46 minutes** at
31.9 FOV/s (8 processes x 8 threads on 32 vCPUs, ~22-way effective parallelism). At
`blur_downsample=1` the same pass would have taken roughly **1.5 to 3.5 hours** — the range is
wide for the reason above: with 8 processes contending for 32 cores, none of them gets the 16
threads that make the full-resolution blur look cheap in a single-FOV benchmark. So the honest
statement is "it turned a multi-hour pass into a 46-minute one", not a single ratio.

Worth separating from the LBP work when attributing credit: `lbp_entropy` went 4.80 s → 0.07 s
(the larger absolute saving, ~4.7 s/FOV), and the blur went 0.65–0.75 s → 0.12 s on top of that.
The blur is the smaller of the two in isolation, but it is the one that removed the *thread
sensitivity* — which is what made the per-FOV cost predictable enough to size a batch run against.

**Confirmed at scale:** the 271-slide Tanzania cohort run scored **88,123 FOVs at 31.9 FOV/s**
aggregate (8 processes x 8 threads, 46 min) — see
`data/results/tanzania-complete-081426/README.md`.

**`blur_downsample=2` is disqualified, and not for the reason you would guess.** It is the only
factor in the sweep that changes any label — 4 Rouleaux flips, all of them correct predictions
lost (`check_empty_field_gate.py` reports Rouleaux exact-match 0.6838 → 0.6808) — and it drifts
5–14x more than 4 despite being the gentler downsample. It is not the sigma mismatch from
integer-dividing the kernel size: OpenCV infers sigma from ksize, giving each factor a slightly
different effective sigma (45.5 → 46.0 → 46.4 → 47.2 → 51.2), and a sigma-matched variant passing
`sigmaX` explicitly measured identical (0.0361 vs 0.0389 mean error at `ds=4`). So it is the
`INTER_AREA` down / `INTER_LINEAR` up round-trip, and **the mechanism is unexplained** — recorded
as such rather than guessed at, since nothing rests on it: `ds=4` beats `ds=2` on both speed and
accuracy. Full working in `data/results/pipeline-runtime/README.md`.

The single in-sample difference is `dpc-176-KTR-72502946.png` (manual: dense), and it is a
threshold artifact rather than a feature one: its composite score moved by −0.00006 while the
dense/very-dense threshold moved down 0.00135, leaving it 0.0001 above instead of 0.0012
below. Under v2.2's *own* params, stride-16 features flip nothing at all.

The sibling from-scratch watershed pipeline lives in `scripts/ai-first/` instead:

```
scripts/ai-first/score_new_slide.py   from-scratch watershed pipeline (see "Overview" above)
scripts/ai-first/label_new_slide.py   slide-relative quintile labels for score_new_slide.py's output
```

<details>
<summary><h2>Opus 5 review feedback (2026-08-07)</h2></summary>

Review of the v2.2 pipeline against its own calibration numbers, given the stated purpose
(score density and Rouleaux). Not yet acted on — recorded here for reference.

### Headline diagnosis

**Density is in decent shape. Rouleaux is not, and v2.1/v2.2 masked the problem rather than
solving it.**

Metrics the calibration report doesn't currently surface, computed from its own v2.2
out-of-fold confusion matrices:

| | exact-match | majority baseline | balanced acc. | macro-F1 | quad. kappa |
|---|---|---|---|---|---|
| Density | 69.4% | 60.8% | 70.6% | 63.7% | 0.833 |
| Rouleaux | 67.6% | **63.5%** | **51.9%** | **49.7%** | 0.839 |

Rouleaux's headline 67.6% exact-match is 4.1 points above always guessing "No Rouleaux."
Per-level, the three middle buckets are barely functional:

| Rouleaux level | n | recall | precision | F1 |
|---|---|---|---|---|
| No Rouleaux | 420 | 81.2% | 91.2% | 85.9% |
| Slight | 97 | 30.9% | 28.0% | **29.4%** |
| Some | 46 | 43.5% | 29.9% | **35.4%** |
| Rouleaux | 34 | 35.3% | 24.0% | **28.6%** |
| Heavy | 64 | 68.8% | 69.8% | 69.3% |

To be fair to the model: quadratic kappa is 0.839, which is genuinely good. That
combination — high kappa, low macro-F1 — is the signature of "the ordering is right, but
the middle of the scale is unresolvable." The composite reliably separates
none-from-heavy and fails in between.

**Why:** the v2.1/v2.2 full-pool refit let the Rouleaux composite solve its problem by
proxying density. Cross-referencing the v2 partial-correlation table against the v2.2
Rouleaux weights:

| feature | partial rho w/ Rouleaux | v2.2 Rouleaux weight |
|---|---|---|
| `coverage` | **0.032** | 0.263 |
| `saturation_score` | **0.005** | 0.269 |
| `tile_glcm_patchiness` | 0.334 | 0.239 |
| `tile_glcm_cv` | 0.231 | 0.088 |

The two features with essentially zero Rouleaux-specific signal carry 53% of the weight;
the two with real signal carry 33%. That's why composite rho is 0.972 against a true label
rho of 0.823 — the Rouleaux score is mostly the density score, cashing in on the label
confound. It improves CV rho because the confound is real, but it isn't measuring
rouleaux. The axis-separation check corroborates: at |delta|>=2 the predicted-vs-manual
delta Spearman is -0.076, i.e. the sign test passes on a marginal-skew offset while
carrying no per-FOV discrimination.

### Suggestions, by payoff

1. **Add instance-level structural features (~half a day; highest-leverage change here).**
   Every candidate feature is a global or tile texture statistic; none measures the
   defining structure of rouleaux — cells stacked in linear chains. `score_new_slide.py`
   in this same directory already computes exactly that: watershed instance segmentation
   -> `touching_pairs` neighbor graph -> the inline-cosine test
   (`INLINE_COS_THRESHOLD = -0.7`) -> `rouleaux_fraction`. It is not in the v2 candidate
   pool. Add `rouleaux_fraction`, plus `n_cells` (cells per unit area — more
   stain-robust than `coverage`), `median_area`, `area_cv`, neighbor-degree distribution
   stats. Caveat: watershed is slower than the texture features; `rouleaux_fraction`
   doesn't need the `solidity` computation that dominates that script's runtime, but
   budget a timing check before committing.

2. **Reconsider whether the 5-level scale should be the product (~an hour to evaluate).**
   The stated purpose is pre-model FOV triage, i.e. "is this FOV usable," not "which of 5
   rouleaux levels." Collapsing the existing v2.2 predictions to a binary decision, with
   no retraining:
   - Rouleaux >= Some: 88.5% accuracy, 86.1% recall, 68.9% precision
   - Density >= Dense: 93.8% accuracy, 86.7% recall, 77.1% precision

   Far more defensible than 5 buckets where three have ~30% F1. Worth deciding
   deliberately rather than inheriting the annotation vocabulary as the product spec.

3. **Fix the validation protocol (~2 hours; changes what you believe, not the model).**
   `stratified_folds` stratifies by label but randomizes across slides, so every fold
   contains FOVs from all three slides. Same-slide FOVs share staining/illumination/
   scanner characteristics — the model can learn slide-specific offsets, and the CV
   number is optimistic. Evidence already in-repo: truly held-out KTR-72502946 scored
   65.7%/60.2%, versus pooled CV's 69.4%/67.6%. Make leave-one-slide-out the headline
   metric, and report balanced accuracy + QWK alongside exact-match.

4. **Estimate the label-noise ceiling (~1 hour of annotation; calibrates everything
   else).** All 661 labels come from one annotator; 5-point ordinal severity has
   substantial self-disagreement. Blind-relabel 50 random FOVs and compute self-agreement.
   If self-agreement on rouleaux is ~70%, 67.6% is at ceiling and items 1/5/6 here are
   wasted effort; if ~90%, there's real headroom. Cheapest experiment on this list —
   do it first or in parallel with (1).

5. **Replace midpoint-of-PAVA'd-medians threshold derivation (~half a day.)**
   `derive_thresholds` reduces each bucket to a single median, then cuts at midpoints —
   discarding within-bucket spread and ignoring class priors. This directly produces the
   density failure mode: Monolayer is 61% of the data, so the Sparser/Monolayer midpoint
   sits too high and 48 Monolayer FOVs get called Sparser (Sparser precision 47.9% — fewer
   than half of predicted-Sparser FOVs actually are). The report's "90.0% Sparser recall"
   headline hides this, and for a triage tool the error is costly in the wrong
   direction — Monolayer is the good state, so these are usable FOVs being rejected. Fix:
   fit cut points to directly optimize QWK (or misclassification cost) on training folds,
   or move to ordinal logistic regression / proportional-odds, fitting weights and
   thresholds jointly instead of bolting a threshold rule onto a least-squares fit.

6. **Let the model be non-linear (~half a day.)** The inverted-U that motivated v2.1
   (confluent rouleaux sheets read as more homogeneous than moderate patchy ones) is real
   physics, not a fitting artifact, and a single weighted sum can never represent it. v2.1
   diluted it with density features rather than modeling it. With 661 samples and
   ~8-14 features, gradient-boosted trees or an ordinal RF is reasonable, or keep the
   linear model and add a spline/quadratic basis on `tile_glcm_patchiness`. Compare
   honestly against the linear baseline under LOSO.

7. **Smaller items, in rough priority order:**
   - Percentile clipping bites exactly where it hurts: `normalize_matrix` clips to the
     2nd-98th percentile, so ~4% of FOVs pile up at 0.0 or 1.0 by construction — precisely
     the Very Dense / Heavy Rouleaux cases. Consider a rank transform or a soft
     (log/sigmoid) squash instead.
   - Only GLCM contrast is tiled today. `coverage` is the most interpretable density
     measure and isn't tiled, yet spatial variation in coverage is arguably a better
     patchiness signal than variation in GLCM contrast — `tile_coverage_cv` /
     `tile_coverage_patchiness` are nearly free given `tile_statistics` already exists.
     Multi-scale grids (3x3 / 7x7 / 15x15) would also capture different cluster sizes;
     7x7 is arbitrary.
   - Add an out-of-distribution guard. The known cross-stain limitation (Liberia
     `dpc-051`: coverage 0.79 vs. Tanzania's ~0.18-0.21) is currently handled by a README
     warning. A runtime check that flags "this FOV's features fall outside the
     calibration percentile range" would convert that documentation caveat into an actual
     safeguard.
   - `weighted_composite` silently renormalizes: a missing feature is skipped and the
     score divides by a smaller `total_weight`, yielding a plausible-looking number
     instead of an error. Given weights already sum to 1 from `fit_weights_stable`, make
     a missing feature raise.
   - Replace the greedy negative-coefficient drop loop in `fit_weights_stable` with
     non-negative least squares (`scipy.optimize.nnls`) or a positivity-constrained
     elastic net — same intent, solved in one shot instead of by iterative deletion that
     can discard a feature that would have been fine under better conditioning.

### Recommended next step

Run (3) leave-one-slide-out and (4) the label-noise ceiling first — together about half a
day, and they determine whether the Rouleaux axis is fixable or already at its ceiling. If
there's headroom, (1) instance-level rouleaux features is the change most likely to
actually move it, since it's the only suggestion that gives the Rouleaux axis signal that
isn't density in disguise.

</details>

