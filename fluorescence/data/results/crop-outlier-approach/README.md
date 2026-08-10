# Crop-outlier approach to filtering fluorescent-spot false positives (2026-08-10)

An exploration of a completely different, cheaper signal than `src/overexposure.py`'s
pixel-level blue-channel/contrast-ratio detector: does a genuine overexposure-halo artifact
show up as an abnormal *number* of detected crops on its own slide? The hypothesis is that the
halo spuriously inflates crop counts -- a targeted side effect worth filtering out -- the same
way it inflates `contrast_ratio`. This uses data already computed and sitting in GCS; no image
is ever read.

## Method

**Ground truth and dataset.** The existing 76-row diverse label set,
`data/labels/overexposure-diverse-080726.csv` (44 `spot_truth=yes`, 32 `spot_truth=no`, spanning
Liberia/Tanzania/Uganda) -- see `data/results/overexposed-diverse-080726/README.md` for how this
set was built. `spot_truth` is ground truth on whether the overexposure-halo artifact itself is
genuinely present.

**Metric: `n_spots_detected`.** The raw count of candidate fluorescent-spot crops *before* ML
filtering, chosen over the alternative (`n_rbcs`, red-blood-cell crops) for two reasons: it's
available uniformly across all three countries (see "Data source" below), and it's
mechanistically the right metric -- these crops come from thresholding local maxima in the same
blue-channel intensity map the halo lives in, so a bright halo plausibly explodes this count
with spurious detections. `n_rbcs` comes from Cellpose segmentation on a different channel and
isn't per-FOV available for Uganda without loading multi-GB precomputed arrays.

**Data source, confirmed by browsing the buckets directly** (`gcloud storage ls`/`cat`; not
documented anywhere else in this repo, since `src/gcs_fov.py`/`src/gcs_fov_multi.py`
deliberately avoid `detection_results/` on purpose -- see their docstrings):

- Liberia (`gs://liberia-2025`) and Tanzania (`gs://tanzania_02032026`) both have
  `detection_results/<...>/<model_version>/<slide_folder>/fov_summary.csv`, one row per
  `fov_id`, with an `n_spots_detected` column.
  - Liberia's slide folder is the same `_Blue` folder `src.gcs_fov.find_slide_blue_folder`
    already resolves, with the `_Blue` suffix stripped.
  - Tanzania's slide folder is `sample_id` directly (no box-probing needed here).
  - `fov_id` in `fov_summary.csv` uses the exact same raster addressing as the labels CSV
    (verified against LB25-D10's sparse-gap pattern, which matches `gcs_fov.py`'s documented
    stride-18/ColumnCount=13 special case exactly) -- joined directly, no row/col decoding.
  - `n_spots_detected` is identical across every model-version run folder for a slide (the
    upstream spot-finding step is shared, only the classifier differs) -- standardized on
    `v8_hardneg_single_t0.995`.
- Uganda (`gs://malaria-annotation-web`) has no per-FOV summary file, but
  `samples/<sample_id>/spots.csv` has the identical per-spot schema as Liberia/Tanzania's own
  `spots.csv` -- grouping by `fov_id` and counting rows reproduces `n_spots_detected` per FOV.

**Baseline: whole-slide leave-one-out median/MAD.** For each target FOV, the baseline is built
from every *other* FOV with detection results on that same slide (not a ± N `fov_id` window).
The center/spread statistic is **median and MAD** (median absolute deviation, scaled by 1.4826),
not mean/std: this label set only covers 76 hand-picked FOVs, so there's no ground truth on the
other ~95%+ of FOVs per slide, and an unknown number of *unlabeled* overexposed/artifact FOVs
could already be sitting in a slide's baseline population. Mean/std have zero breakdown
resistance -- one or two such FOVs shift them arbitrarily and mute the exact effect being
tested -- while median/MAD tolerate up to ~50% contamination before breaking. Mean/std are still
computed and reported (`baseline_mean`/`baseline_std` in `results.csv`), specifically so a large
mean-vs-median divergence is itself visible as the `mean_median_divergence` flag below.

`ratio_to_median = target_n_spots / baseline_median`, `robust_zscore = (target_n_spots -
baseline_median) / baseline_mad`.

**Code:** `scripts/crop-outlier-approach/crop_counts.py` (GCS resolution + per-sample cache +
baseline stats), `scripts/crop-outlier-approach/analyze_crop_outliers.py` (main pipeline,
produces `results.csv`), `scripts/crop-outlier-approach/report_tables.py` (the tables below).

## Results: positive vs. negative ground truth

**Positive summary** (n=44, 3 excluded -- see "Missing data" below)

- median `robust_zscore` **52.59** (mean 73.60, range 4.94-387.61)
- median `ratio_to_median` **19.80** (mean 30.81, range 2.44-216.50)
- 41/41 (**100.0%**) at or above 2 MAD above baseline
- 40/41 (97.6%) at or above 5 MAD above baseline
- 37/41 (90.2%) at or above 10 MAD above baseline
- 31/41 (75.6%) at or above 20 MAD above baseline

**Negative summary** (n=32, 0 excluded)

- median `robust_zscore` **2.25** (mean 6.54, range -0.67-118.67)
- median `ratio_to_median` **1.77** (mean 3.97, range 0.43-58.41)
- 17/32 (53.1%) at or above 2 MAD above baseline
- 7/32 (21.9%) at or above 5 MAD above baseline
- 4/32 (12.5%) at or above 10 MAD above baseline
- 1/32 (3.1%) at or above 20 MAD above baseline

**Every single positive row's crop count is above its own slide's median** (min ratio 2.44x,
i.e. even the weakest positive has nearly 2.5x its slide's typical crop count) -- a much cleaner
separation than the pixel-based detector's own known false-negative population (faint/diffuse
halos below `RATIO_THRESHOLD`, see `fluorescence/README.md`'s "Diffuse-halo signal" section).
None of the 5 `diffuse`-tagged rows or 5 `double`-tagged rows (the pixel detector's weakest
subsets) drop below `robust_zscore=11` here.

**Full positive-group table:**

| sample_id | fov_id | notes | target_n_spots | baseline_median | baseline_mad | ratio_to_median | robust_zscore | flags |
|---|---|---|---|---|---|---|---|---|
| LB-D10-2025-12-29-150312-0171084-VFPCHC-2-4 | 153 |  | 1818 | 46 | 19.274 | 39.522 | 91.938 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D10-2025-12-29-150312-0171084-VFPCHC-2-4 | 154 |  | 2866 | 46 | 19.274 | 62.304 | 146.313 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D10-2025-12-30-083614-0250901VFPCHC-2-1 | 210 |  | 4005 | 160 | 74.13 | 25.031 | 51.868 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D10-2025-12-30-083614-0250901VFPCHC-2-1 | 227 |  | 3168 | 160 | 74.13 | 19.8 | 40.577 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D10-2025-12-30-084453-0250071VFPCHC-2-2 | 200 |  | 5072 | 97 | 48.926 | 52.289 | 101.685 | mean_median_divergence;high_outlier |
| LB-D11-2025-12-19-111309-0211715-VFPCHC-3-1 | 277 | background | 6003 | 1828 | 610.831 | 3.284 | 6.835 | high_outlier |
| LB-D11-2025-12-19-131014-0241591-VFPCHC-3-2 | 278 | background | 7231 | 2364 | 944.416 | 3.059 | 5.153 | high_outlier |
| LB-D3-2025-09-02-141940-25087110-D-Only-1-2 | 42 | background | 1710 | 414 | 229.803 | 4.13 | 5.64 | high_outlier |
| LB-D3-2025-09-09-093425-250917463-D-Only-1-1 | 166 |  | 1998 | 168 | 41.513 | 11.893 | 44.083 | high_outlier |
| LB-D3-2025-09-27-121918-17217958-D-thin-4-4 | 262 |  | 9960 | 362 | 96.369 | 27.514 | 99.596 | mean_median_divergence;high_outlier |
| LB-D3-2025-10-03-104211-250917371-D-thin-2-3 | 4 |  | 8239 | 145 | 53.374 | 56.821 | 151.648 | mean_median_divergence;high_outlier |
| LB-D3-2025-10-03-104643-250917465-D-thin-3-4 | 185 | green | 2119 | 190 | 59.304 | 11.153 | 32.527 | high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 114 |  | 4571 | 219 | 63.752 | 20.872 | 68.265 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 125 |  | 534 | 219 | 63.752 | 2.438 | 4.941 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-03-125352-2402169466D-thin-2-1 | 3 | diffuse | 2656 | 252 | 42.995 | 10.54 | 55.913 | high_outlier |
| LB-D3-2025-10-03-130859-250916865-D-thin-1-4 | 236 |  | 2907 | 345 | 106.747 | 8.426 | 24.001 | high_outlier |
| LB-D3-2025-10-22-131729-250917745-D-thin-2-3 | 134 | double | 9946 | 271 | 106.747 | 36.701 | 90.635 | mean_median_divergence;high_outlier |
| LB-D3-2025-10-22-132316-2411189646-D-thin-1-4 | 135 | diffuse | 2858 | 398 | 154.19 | 7.181 | 15.954 | mean_median_divergence;high_outlier |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | 122 | double | 9030 | 204 | 118.608 | 44.265 | 74.413 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | 238 |  | 1524 | 204 | 118.608 | 7.471 | 11.129 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-10-24-132012-25046898-D-thin-1-4 | 3 | diffuse | 2797 | 509 | 114.16 | 5.495 | 20.042 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-24-132012-25046898-D-thin-1-4 | 305 |  | 3431 | 509 | 114.16 | 6.741 | 25.596 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-24-162727-230918080-D-thin-1-4 | 8 | diffuse | 2929 | 1074 | 167.534 | 2.727 | 11.072 | high_outlier |
| LB-D3-2025-10-25-105806-180951467-D-thin-1-1 | 270 |  | 2832 | 173 | 84.508 | 16.37 | 31.464 | high_outlier |
| LB-D3-2025-10-25-150947-250917467-D-thin-3-2 | 235 |  | 2501 | 345 | 163.086 | 7.249 | 13.22 | high_outlier |
| LB-D3-2025-10-27-123159-251123404-D-thin-4-1 | 48 |  | 5105 | 158 | 75.613 | 32.31 | 65.426 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-27-123159-251123404-D-thin-4-1 | 49 |  | 2566 | 158 | 75.613 | 16.241 | 31.847 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-10-27-124239-250916732-D-thin-1-3 | 301 |  | 9410 | 455 | 111.195 | 20.681 | 80.534 | high_outlier |
| LB-D3-2025-10-27-134711-250917368-D-thin-1-3 | 52 |  | 6375 | 108 | 32.617 | 59.028 | 192.138 | high_outlier |
| LB-D3-2025-10-27-154305-250917412-D-thin-1-4 | 119 | diffuse | 1037 | 177 | 57.821 | 5.859 | 14.873 | high_outlier |
| LB-D3-2025-10-27-173317-250917493-D-thin-2-4 | 82 |  | 4501 | 127 | 56.339 | 35.441 | 77.637 | mean_median_divergence;high_outlier |
| KIT-62500763 | 200 | green | -- | -- | -- | -- | -- | **no_data** |
| KIT-62501035 | 67 |  | 3656 | 149 | 44.478 | 24.537 | 78.848 | high_outlier |
| KIT-62501081 | 141 | double | 2335 | 74 | 42.995 | 31.554 | 52.587 | high_outlier |
| KIT-62501087 | 271 |  | 8660 | 40 | 22.239 | 216.5 | 387.607 | mean_median_divergence;high_outlier |
| KTR-72502946 | 54 |  | -- | -- | -- | -- | -- | **no_data** |
| KTR-72502946 | 198 |  | -- | -- | -- | -- | -- | **no_data** |
| NKR-72502319 | 293 |  | 5938 | 74 | 29.652 | 80.243 | 197.761 | same_slide_contamination;mean_median_divergence;high_outlier |
| NKR-72502319 | 311 |  | 5734 | 74 | 29.652 | 77.486 | 190.881 | same_slide_contamination;mean_median_divergence;high_outlier |
| RUB-62501332 | 133 |  | 2797 | 213 | 47.443 | 13.131 | 54.465 | high_outlier |
| RUB-62501389 | 284 | double | 9453 | 89 | 42.995 | 106.213 | 217.791 | mean_median_divergence;high_outlier |
| RUB-72501756 | 315 |  | 2125 | 171 | 63.752 | 12.427 | 30.65 | high_outlier |
| PAT-070-3 | 34 | double | 945 | 195 | 47.443 | 4.846 | 15.808 | high_outlier |
| PBC-608-KH-1 | 171 |  | 1913 | 57.0 | 17.791 | 33.561 | 104.321 | high_outlier |

**Full negative-group table:**

| sample_id | fov_id | notes | target_n_spots | baseline_median | baseline_mad | ratio_to_median | robust_zscore | flags |
|---|---|---|---|---|---|---|---|---|
| LB-D11-2025-12-17-115859-0250319D-thin-4-1 | 29 | background | 3473 | 1599 | 661.24 | 2.172 | 2.834 | high_outlier |
| LB-D11-2025-12-19-134126-025073-VFPCHC-3-1 | 1 | background | 3552 | 1871 | 1177.184 | 1.898 | 1.428 |  |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 257 | background | 2477 | 559 | 373.615 | 4.431 | 5.134 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 269 | background | 1385 | 559 | 373.615 | 2.478 | 2.211 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 274 | background | 1864 | 559 | 373.615 | 3.335 | 3.493 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 279 | background | 1468 | 559 | 373.615 | 2.626 | 2.433 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 289 | background | 2433 | 559 | 373.615 | 4.352 | 5.016 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 1 | background | 261 | 219 | 63.752 | 1.192 | 0.659 | same_slide_contamination |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 16 | background | 365 | 219 | 63.752 | 1.667 | 2.29 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 17 | background | 385 | 219 | 63.752 | 1.758 | 2.604 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 18 | background | 390 | 219 | 63.752 | 1.781 | 2.682 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 19 | background | 190 | 221 | 66.717 | 0.86 | -0.465 | same_slide_contamination |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 53 | background | 484 | 219 | 63.752 | 2.21 | 4.157 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 126 | artifact | 1181 | 219 | 63.752 | 5.393 | 15.09 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-24-113736-250918214-D-thin-2-3 | 96 | relabeled by Emily 2026-08-07 (was yes) | 957 | 271 | 62.269 | 3.531 | 11.017 | high_outlier |
| LB-D3-2025-10-27-144635-250918691-D-thin-2-2 | 57 | artifact | 492 | 371 | 131.951 | 1.326 | 0.917 | same_slide_contamination;mean_median_divergence |
| LB-D3-2025-10-27-144635-250918691-D-thin-2-2 | 243 | background | 558 | 371 | 131.951 | 1.504 | 1.417 | same_slide_contamination;mean_median_divergence |
| LB-D3-2025-10-27-145205-250917002-D-thin-3-3 | 310 | artifact | 342 | 181 | 53.374 | 1.89 | 3.016 | high_outlier |
| LB-D3-2025-10-27-155920-250713919-D-thin-3-3 | 169 | background | 256 | 287 | 80.06 | 0.892 | -0.387 |  |
| LB-D5-2026-01-27-112616-0240052-VFPCHC-2-2 | 40 | background | 5023 | 1556 | 1025.959 | 3.228 | 3.379 | mean_median_divergence;high_outlier |
| KIT-62501062 | 83 | artifact | 187 | 129 | 77.095 | 1.45 | 0.752 |  |
| NKR-72502319 | 119 | background | 92 | 74 | 29.652 | 1.243 | 0.607 | same_slide_contamination;mean_median_divergence |
| RUB-62501518 | 315 | background | 5549 | 95 | 45.961 | 58.411 | 118.667 | mean_median_divergence;high_outlier |
| RUB-62501529 | 87 | background | 333 | 228 | 80.06 | 1.461 | 1.312 |  |
| PAT-072-1 | 14 | artifact | 324 | 50 | 22.239 | 6.48 | 12.321 | same_slide_contamination;high_outlier |
| PAT-072-1 | 94 | artifact | 170 | 50 | 22.239 | 3.4 | 5.396 | same_slide_contamination;high_outlier |
| PAT-154-1 | 478 | background | 2606 | 3547.0 | 3017.091 | 0.735 | -0.312 |  |
| PBC-225_AM-1 | 30 | background | 2852 | 2930.5 | 720.544 | 0.973 | -0.109 |  |
| PBC-800-1 | 128 | background | 102 | 100.5 | 48.184 | 1.015 | 0.031 | same_slide_contamination;mean_median_divergence |
| PBC-800-1 | 732 | background | 176 | 100.5 | 48.184 | 1.751 | 1.567 | same_slide_contamination;mean_median_divergence |
| PAT-103-2 | 441 | background | 577 | 1326 | 1119.363 | 0.435 | -0.669 |  |
| PAT-112-2 | 124 | background | 147 | 121.0 | 34.1 | 1.215 | 0.762 |  |

## Discussion: does the signal separate ground truth?

**Yes, cleanly at the group-median level, but no single threshold perfectly separates the two
groups.** The medians are ~23x apart (52.59 vs. 2.25 `robust_zscore`; 19.80x vs. 1.77x ratio),
and every positive row clears at least 2.4x its own slide's baseline. But the two distributions
overlap in the `robust_zscore` ~3-16 range: 17 of 32 negatives (53.1%) already clear the
`high_outlier` threshold (`robust_zscore >= 2`) used elsewhere in this report, and the weakest
positive (`robust_zscore=4.94`) sits below several of the strongest false-alarm-prone negatives.

**The overlap is concentrated almost entirely in two `notes` categories: `background` and
`artifact`.** Every negative with `robust_zscore >= 5` is tagged `background` or `artifact`
(elevated illumination from many puncta, or debris/hair) *except one* -- `RUB-62501518`
(`robust_zscore=118.67`, ratio 58.4x), discussed below. This makes sense: `background`-tagged
negatives are, by construction, FOVs with a lot of real signal (many puncta) without an actual
halo -- exactly the population most likely to also have an elevated crop count for a completely
different, legitimate reason. This mirrors the pixel-based detector's own known false-positive
failure mode (`fluorescence/README.md`'s Calibration section: "a raw brightness-elevation
threshold ... produced a false positive on one negative-control FOV that was simply a 'busier'
frame") -- the crop-count signal inherits the same confound, not a new one.

**At a very high threshold (`robust_zscore >= 20`), recall drops to 75.6% but false-positive
rate falls to 3.1%** (31/41 positives vs. 1/32 negatives) -- a plausible operating point if this
were ever used as a coarse pre-filter, but not as a standalone replacement for
`src/overexposure.py`'s pixel-level detector, and not validated beyond this 76-row set.

**One negative deserves a named callout: `RUB-62501518` (Tanzania, fov 315, `background`).**
`robust_zscore=118.67` is higher than all but 4 of the 41 positive rows -- an outlier among
negatives, not a borderline case. 5,549 detected crops vs. a slide median of 95 is a large,
specific anomaly for a FOV the annotator tagged as ordinary background elevation, not an
overexposure halo. Worth a manual look at the raw image before trusting this row's `spot=no`
label at face value -- it's exactly the kind of case this whole-slide-median approach is
designed to surface for human review, not resolve unilaterally.

## Flagged rows

Every row with any of the flags below is listed explicitly, per Emily's request not to bury
unusual cases in prose. `results.csv`'s `flags` column is semicolon-joined; a row can carry
multiple flags.

- **`no_data`** (3 rows) -- sample has no `detection_results` under any model version at all.
  `KIT-62500763` and `KTR-72502946` (both Tanzania) -- confirmed to exist as raw samples
  (`KTR-72502946` is in `gs://tanzania_02032026/TZ2025-Box5/`) but were never run through
  detection. Excluded from all summary stats above.
- **`same_slide_contamination`** (33 rows) -- another row in this same 76-row label set shares
  the `sample_id` and therefore sits inside the baseline population. Notably includes
  `LB-D3-2025-10-27-123159-251123404-D-thin-4-1` fov 48 & 49 (both `spot_truth=yes`) -- each
  one's "baseline" partly includes the other's inflated count. With 323 other FOVs on that
  slide, one contaminating value moves the median/MAD negligibly in this specific case, but the
  flag is set regardless since the *direction* of the risk (baseline pulled toward, not away
  from, the target) is real and worth tracking per-row rather than assuming it's always small.
- **`mean_median_divergence`** (29 rows) -- `baseline_mean` and `baseline_median` differ by more
  than 25% of the median, a direct signal that a slide's own FOV population probably isn't a
  clean background sample (consistent with the reasoning that motivated median/MAD over
  mean/std in the first place -- see "Method").
- **`zero_or_undefined_baseline`** -- did not fire on any row in this label set.
- **`high_outlier`** (58 rows: 41 positive, 17 negative) / **`low_outlier`** (0 rows) --
  `robust_zscore >= 2` (or `<= -2`). See "Discussion" above for why this threshold alone doesn't
  cleanly separate ground truth.

## Caveats

- **Small, hand-picked label set.** 76 rows is not enough to fit or validate a hard decision
  threshold, and this set was deliberately curated to be "visually diverse," not a random or
  representative sample of any slide.
- **No ground truth on ~95%+ of each slide's other FOVs.** The whole-slide median/MAD baseline
  is a best-effort defense against unlabeled contamination (see "Method"), not a guarantee --
  `mean_median_divergence` is a proxy signal for this, not a fix.
- **This is a triage/exploration pass, not a replacement for `src/overexposure.py`.** No code
  outside `scripts/crop-outlier-approach/` was touched; nothing here changes production
  behavior.
- **Two samples have no detection results at all** (`KIT-62500763`, `KTR-72502946`) and are
  excluded from every statistic above, not imputed or approximated.
- **`v8_hardneg_single_t0.995` was used for every Liberia/Tanzania row**, chosen because
  `n_spots_detected` was verified identical across all available model-version run folders for
  the slides checked (the upstream spot-finding step is shared across models) -- not
  re-verified for every single slide in this label set.

## Files

- `results.csv` -- full per-row output (target/baseline stats, flags, `no_data_reason`)
- `../../scripts/crop-outlier-approach/crop_counts.py` -- GCS resolution + per-sample cache +
  baseline stats
- `../../scripts/crop-outlier-approach/analyze_crop_outliers.py` -- pipeline that produced
  `results.csv`
- `../../scripts/crop-outlier-approach/report_tables.py` -- generates the tables in this doc
