# Crop-outlier approach to filtering fluorescent-spot false positives (2026-08-10, updated 2026-08-11)

An exploration of a completely different, cheaper signal than `src/overexposure.py`'s
pixel-level blue-channel/contrast-ratio detector: does a genuine overexposure-halo artifact
show up as an abnormal *number* of detected crops on its own slide? The hypothesis is that the
halo spuriously inflates crop counts -- a targeted side effect worth filtering out -- the same
way it inflates `contrast_ratio`. This uses data already computed and sitting in GCS; no image
is ever read.

**2026-08-11 update, two changes, both applied and re-run below:**
1. **Baseline is now whole-slide, not leave-one-out.** Every FOV's baseline is the median/MAD
   over *every* FOV on its slide (the target included), one baseline computed per slide and
   reused for every FOV scored against it -- simpler than the previous per-target exclusion, and
   numerically almost identical for slides this size (300+ FOVs).
2. **Tanzania fallback added.** `tanzania_02032026`'s own `detection_results/` tree turned out to
   be missing 146 slides (94 of `TZ2025-Box1`'s 98, 52 of `Box5`'s 99) that were never mirrored
   into it -- including both of this analysis's previous `no_data` rows, `KIT-62500763` and
   `KTR-72502946`. `gs://malaria-annotation-web` (the annotation tool's own bucket) has
   `fov_summary.csv` for all 146, confirmed byte-identical to `tanzania_02032026`'s copy where
   both exist. `crop_counts.py` now falls back to it. See its module docstring for the full
   explanation, and `crop-outlier-v2/case-study-KTR-72502946.md` for a full-slide case study on
   the more interesting of the two recovered samples.

Two versions of this analysis are run, same style, same 76-row label set, same whole-slide
median/MAD approach -- differing only in which precomputed GCS count is treated as the outlier
signal:

1. **Crop-count metric** (`n_spots_detected`) -- raw candidate fluorescent-spot crops before ML
   filtering.
2. **Parasite-count metric** (`n_positives`) -- crops the ML classifier actually confirmed as a
   parasite.

The comparison at the end is the actual point of running both: does the halo's apparent
inflation of raw candidate crops (metric 1) survive into confirmed-parasite counts (metric 2),
or does the classifier already filter it out?

## Metric 1: crop-count (`n_spots_detected`)

### Method

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
  **Caveat added 2026-08-11, not yet fixed:** this was cross-checked against a Tanzania sample
  that has both `spots.csv` and `fov_summary.csv` (`RUB-62501326`) and the row-count-per-`fov_id`
  actually matches `n_spots_filtered`, not `n_spots_detected` (0/324 mismatches vs. filtered,
  324/324 mismatches vs. detected). Uganda has no `fov_summary.csv` anywhere to source true
  pre-filter `n_spots_detected` from, so the 10 Uganda rows below are on a different,
  already-partially-filtered metric than every Liberia/Tanzania row -- flagged, not corrected.
  See `crop_counts.py`'s module docstring.
- **Tanzania fallback, added 2026-08-11:** when `tanzania_02032026`'s own tree has nothing for a
  sample, `gs://malaria-annotation-web`'s `samples/<sample_id>/fov_summary.csv` is tried next --
  see the update note at the top of this doc.

**Baseline: whole-slide median/MAD.** As of 2026-08-11, the baseline for each target FOV is the
median/MAD over *every* FOV with detection results on that same slide, the target included (not
a leave-one-out exclusion, and not a ± N `fov_id` window). The center/spread statistic is
**median and MAD** (median absolute deviation, scaled by 1.4826), not mean/std: this label set
only covers 76 hand-picked FOVs, so there's no ground truth on the other ~95%+ of FOVs per
slide, and an unknown number of *unlabeled* overexposed/artifact FOVs could already be sitting
in a slide's baseline population. Mean/std have zero breakdown resistance -- one or two such
FOVs shift them arbitrarily and mute the exact effect being tested -- while median/MAD tolerate
up to ~50% contamination before breaking. Mean/std are still computed and reported
(`baseline_mean`/`baseline_std` in `results.csv`), specifically so a large mean-vs-median
divergence is itself visible as the `mean_median_divergence` flag below.

`ratio_to_median = target_n_spots / baseline_median`, `robust_zscore = (target_n_spots -
baseline_median) / baseline_mad`.

**Code:** `scripts/crop-outlier-approach/crop_counts.py` (GCS resolution + per-sample cache +
baseline stats), `scripts/crop-outlier-approach/analyze_crop_outliers.py` (main pipeline; run as
`python scripts/crop-outlier-approach/analyze_crop_outliers.py` for this metric, producing
`results.csv`), `scripts/crop-outlier-approach/report_tables.py` (the tables below).

### Results: positive vs. negative ground truth

**Positive summary** (n=44, 0 excluded -- both previous `no_data` rows now resolve, see the
2026-08-11 update note at the top)

- median `robust_zscore` **51.78** (mean 72.02, range -1.05-387.61)
- median `ratio_to_median` **18.09** (mean 30.67, range 0.47-216.50)
- 43/44 (**97.7%**) at or above 2 MAD above baseline
- 42/44 (95.5%) at or above 5 MAD above baseline
- 39/44 (88.6%) at or above 10 MAD above baseline
- 31/44 (70.5%) at or above 20 MAD above baseline

**Negative summary** (n=32, 0 excluded)

- median `robust_zscore` **2.21** (mean 6.51, range -0.67-118.67)
- median `ratio_to_median` **1.76** (mean 3.97, range 0.43-58.41)
- 17/32 (53.1%) at or above 2 MAD above baseline
- 6/32 (18.8%) at or above 5 MAD above baseline
- 4/32 (12.5%) at or above 10 MAD above baseline
- 1/32 (3.1%) at or above 20 MAD above baseline

**Almost every positive row's crop count is above its own slide's median, with one clean
exception.** `KTR-72502946` fov 54 (recovered by the 2026-08-11 Tanzania fallback) sits *below*
its slide's median (`ratio=0.47`, `robust_zscore=-1.05`) despite a confirmed genuine halo -- the
first positive row in this dataset the crop-count signal misses outright, not just weakly
detects. Every other positive row still clears at least 2.4x its own slide's baseline. See
`crop-outlier-v2/case-study-KTR-72502946.md` for a full look at this slide (its *other* labeled
FOV, 198, is the single largest `robust_zscore` outside `KIT-62501087`, so the same slide
contains both the strongest true positive and the cleanest miss found anywhere in this dataset).
Excluding that one row, this remains a much cleaner separation than the pixel-based detector's
own known false-negative population (faint/diffuse halos below `RATIO_THRESHOLD`, see
`fluorescence/README.md`'s "Diffuse-halo signal" section): none of the 5 `diffuse`-tagged rows or
5 `double`-tagged rows (the pixel detector's weakest subsets) drop below `robust_zscore=11`.

**Full positive-group table:**

| sample_id | fov_id | notes | target_n_spots | baseline_median | baseline_mad | ratio_to_median | robust_zscore | flags |
|---|---|---|---|---|---|---|---|---|
| LB-D10-2025-12-29-150312-0171084-VFPCHC-2-4 | 153 |  | 1818 | 46.5 | 20.015 | 39.097 | 88.508 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D10-2025-12-29-150312-0171084-VFPCHC-2-4 | 154 |  | 2866 | 46.5 | 20.015 | 61.634 | 140.869 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D10-2025-12-30-083614-0250901VFPCHC-2-1 | 210 |  | 4005 | 160.0 | 74.13 | 25.031 | 51.868 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D10-2025-12-30-083614-0250901VFPCHC-2-1 | 227 |  | 3168 | 160.0 | 74.13 | 19.8 | 40.577 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D10-2025-12-30-084453-0250071VFPCHC-2-2 | 200 |  | 5072 | 97.5 | 49.667 | 52.021 | 100.157 | mean_median_divergence;high_outlier |
| LB-D11-2025-12-19-111309-0211715-VFPCHC-3-1 | 277 | background | 6003 | 1830.0 | 611.572 | 3.28 | 6.823 | high_outlier |
| LB-D11-2025-12-19-131014-0241591-VFPCHC-3-2 | 278 | background | 7231 | 2364.0 | 945.899 | 3.059 | 5.145 | high_outlier |
| LB-D3-2025-09-02-141940-25087110-D-Only-1-2 | 42 | background | 1710 | 415.0 | 231.286 | 4.12 | 5.599 | high_outlier |
| LB-D3-2025-09-09-093425-250917463-D-Only-1-1 | 166 |  | 1998 | 168.0 | 41.513 | 11.893 | 44.083 | mean_median_divergence;high_outlier |
| LB-D3-2025-09-27-121918-17217958-D-thin-4-4 | 262 |  | 9960 | 362.5 | 97.11 | 27.476 | 98.831 | mean_median_divergence;high_outlier |
| LB-D3-2025-10-03-104211-250917371-D-thin-2-3 | 4 |  | 8239 | 145.0 | 53.374 | 56.821 | 151.648 | mean_median_divergence;high_outlier |
| LB-D3-2025-10-03-104643-250917465-D-thin-3-4 | 185 | green | 2119 | 190.0 | 59.304 | 11.153 | 32.527 | high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 114 |  | 4571 | 220.0 | 65.234 | 20.777 | 66.698 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 125 |  | 534 | 220.0 | 65.234 | 2.427 | 4.813 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-03-125352-2402169466D-thin-2-1 | 3 | diffuse | 2656 | 252.0 | 43.737 | 10.54 | 54.965 | high_outlier |
| LB-D3-2025-10-03-130859-250916865-D-thin-1-4 | 236 |  | 2907 | 345.0 | 108.23 | 8.426 | 23.672 | high_outlier |
| LB-D3-2025-10-22-131729-250917745-D-thin-2-3 | 134 | double | 9946 | 271.5 | 107.488 | 36.634 | 90.005 | mean_median_divergence;high_outlier |
| LB-D3-2025-10-22-132316-2411189646-D-thin-1-4 | 135 | diffuse | 2858 | 398.5 | 155.673 | 7.172 | 15.799 | mean_median_divergence;high_outlier |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | 122 | double | 9030 | 204.0 | 119.349 | 44.265 | 73.951 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | 238 |  | 1524 | 204.0 | 119.349 | 7.471 | 11.06 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-10-24-132012-25046898-D-thin-1-4 | 3 | diffuse | 2797 | 510.0 | 115.643 | 5.484 | 19.776 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-24-132012-25046898-D-thin-1-4 | 305 |  | 3431 | 510.0 | 115.643 | 6.727 | 25.259 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-24-162727-230918080-D-thin-1-4 | 8 | diffuse | 2929 | 1074.5 | 169.016 | 2.726 | 10.972 | high_outlier |
| LB-D3-2025-10-25-105806-180951467-D-thin-1-1 | 270 |  | 2832 | 173.0 | 84.508 | 16.37 | 31.464 | high_outlier |
| LB-D3-2025-10-25-150947-250917467-D-thin-3-2 | 235 |  | 2501 | 345.5 | 164.569 | 7.239 | 13.098 | high_outlier |
| LB-D3-2025-10-27-123159-251123404-D-thin-4-1 | 48 |  | 5105 | 158.5 | 77.837 | 32.208 | 63.55 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-10-27-123159-251123404-D-thin-4-1 | 49 |  | 2566 | 158.5 | 77.837 | 16.189 | 30.93 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-10-27-124239-250916732-D-thin-1-3 | 301 |  | 9410 | 456.0 | 111.195 | 20.636 | 80.525 | high_outlier |
| LB-D3-2025-10-27-134711-250917368-D-thin-1-3 | 52 |  | 6375 | 108.0 | 32.617 | 59.028 | 192.138 | mean_median_divergence;high_outlier |
| LB-D3-2025-10-27-154305-250917412-D-thin-1-4 | 119 | diffuse | 1037 | 177.0 | 58.563 | 5.859 | 14.685 | high_outlier |
| LB-D3-2025-10-27-173317-250917493-D-thin-2-4 | 82 |  | 4501 | 127.0 | 56.339 | 35.441 | 77.637 | mean_median_divergence;high_outlier |
| KIT-62500763 | 200 | green | 3432 | 805.0 | 234.251 | 4.263 | 11.214 | high_outlier |
| KIT-62501035 | 67 |  | 3656 | 149.0 | 45.219 | 24.537 | 77.555 | high_outlier |
| KIT-62501081 | 141 | double | 2335 | 74.5 | 43.737 | 31.342 | 51.684 | mean_median_divergence;high_outlier |
| KIT-62501087 | 271 |  | 8660 | 40.0 | 22.239 | 216.5 | 387.607 | mean_median_divergence;high_outlier |
| KTR-72502946 | 54 |  | 25 | 53.0 | 26.687 | 0.472 | **-1.049** | same_slide_contamination;mean_median_divergence |
| KTR-72502946 | 198 |  | 4417 | 53.0 | 26.687 | 83.34 | 163.527 | same_slide_contamination;mean_median_divergence;high_outlier |
| NKR-72502319 | 293 |  | 5938 | 74.0 | 29.652 | 80.243 | 197.761 | same_slide_contamination;mean_median_divergence;high_outlier |
| NKR-72502319 | 311 |  | 5734 | 74.0 | 29.652 | 77.486 | 190.881 | same_slide_contamination;mean_median_divergence;high_outlier |
| RUB-62501332 | 133 |  | 2797 | 213.5 | 48.184 | 13.101 | 53.617 | high_outlier |
| RUB-62501389 | 284 | double | 9453 | 89.0 | 42.995 | 106.213 | 217.791 | mean_median_divergence;high_outlier |
| RUB-72501756 | 315 |  | 2125 | 171.5 | 63.752 | 12.391 | 30.642 | high_outlier |
| PAT-070-3 | 34 | double | 945 | 195.0 | 47.443 | 4.846 | 15.808 | high_outlier |
| PBC-608-KH-1 | 171 |  | 1913 | 57 | 17.791 | 33.561 | 104.321 | high_outlier |

**Full negative-group table:**

| sample_id | fov_id | notes | target_n_spots | baseline_median | baseline_mad | ratio_to_median | robust_zscore | flags |
|---|---|---|---|---|---|---|---|---|
| LB-D11-2025-12-17-115859-0250319D-thin-4-1 | 29 | background | 3473 | 1600.5 | 661.981 | 2.17 | 2.829 | high_outlier |
| LB-D11-2025-12-19-134126-025073-VFPCHC-3-1 | 1 | background | 3552 | 1873.0 | 1180.891 | 1.896 | 1.422 |  |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 257 | background | 2477 | 561.0 | 375.839 | 4.415 | 5.098 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 269 | background | 1385 | 561.0 | 375.839 | 2.469 | 2.192 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 274 | background | 1864 | 561.0 | 375.839 | 3.323 | 3.467 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 279 | background | 1468 | 561.0 | 375.839 | 2.617 | 2.413 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 289 | background | 2433 | 561.0 | 375.839 | 4.337 | 4.981 | same_slide_contamination;mean_median_divergence;high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 1 | background | 261 | 220.0 | 65.234 | 1.186 | 0.629 | same_slide_contamination |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 16 | background | 365 | 220.0 | 65.234 | 1.659 | 2.223 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 17 | background | 385 | 220.0 | 65.234 | 1.75 | 2.529 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 18 | background | 390 | 220.0 | 65.234 | 1.773 | 2.606 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 19 | background | 190 | 220.0 | 65.234 | 0.864 | -0.46 | same_slide_contamination |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 53 | background | 484 | 220.0 | 65.234 | 2.2 | 4.047 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 126 | artifact | 1181 | 220.0 | 65.234 | 5.368 | 14.731 | same_slide_contamination;high_outlier |
| LB-D3-2025-10-24-113736-250918214-D-thin-2-3 | 96 | relabeled by Emily 2026-08-07 (was yes) | 957 | 271.0 | 62.269 | 3.531 | 11.017 | high_outlier |
| LB-D3-2025-10-27-144635-250918691-D-thin-2-2 | 57 | artifact | 492 | 372.5 | 133.434 | 1.321 | 0.896 | same_slide_contamination;mean_median_divergence |
| LB-D3-2025-10-27-144635-250918691-D-thin-2-2 | 243 | background | 558 | 372.5 | 133.434 | 1.498 | 1.39 | same_slide_contamination;mean_median_divergence |
| LB-D3-2025-10-27-145205-250917002-D-thin-3-3 | 310 | artifact | 342 | 181.0 | 53.374 | 1.89 | 3.016 | high_outlier |
| LB-D3-2025-10-27-155920-250713919-D-thin-3-3 | 169 | background | 256 | 287.0 | 80.06 | 0.892 | -0.387 |  |
| LB-D5-2026-01-27-112616-0240052-VFPCHC-2-2 | 40 | background | 5023 | 1556.5 | 1028.183 | 3.227 | 3.371 | mean_median_divergence;high_outlier |
| KIT-62501062 | 83 | artifact | 187 | 129.0 | 77.095 | 1.45 | 0.752 |  |
| NKR-72502319 | 119 | background | 92 | 74.0 | 29.652 | 1.243 | 0.607 | same_slide_contamination;mean_median_divergence |
| RUB-62501518 | 315 | background | 5549 | 95.0 | 45.961 | 58.411 | 118.667 | mean_median_divergence;high_outlier |
| RUB-62501529 | 87 | background | 333 | 228.0 | 80.802 | 1.461 | 1.299 |  |
| PAT-072-1 | 14 | artifact | 324 | 50.0 | 22.239 | 6.48 | 12.321 | same_slide_contamination;high_outlier |
| PAT-072-1 | 94 | artifact | 170 | 50.0 | 22.239 | 3.4 | 5.396 | same_slide_contamination;high_outlier |
| PAT-154-1 | 478 | background | 2606 | 3546 | 3012.643 | 0.735 | -0.312 |  |
| PBC-225_AM-1 | 30 | background | 2852 | 2927 | 720.544 | 0.974 | -0.104 |  |
| PBC-800-1 | 128 | background | 102 | 101 | 48.926 | 1.01 | 0.02 | same_slide_contamination;mean_median_divergence |
| PBC-800-1 | 732 | background | 176 | 101 | 48.926 | 1.743 | 1.533 | same_slide_contamination;mean_median_divergence |
| PAT-103-2 | 441 | background | 577 | 1325.0 | 1115.657 | 0.435 | -0.67 |  |
| PAT-112-2 | 124 | background | 147 | 121 | 34.1 | 1.215 | 0.762 |  |

### Discussion: does the crop-count signal separate ground truth?

**Yes, cleanly at the group-median level, but no single threshold perfectly separates the two
groups -- and, as of the 2026-08-11 update, one positive row is a clean miss, not just a weak
detection.** The medians are ~23x apart (51.78 vs. 2.21 `robust_zscore`; 18.09x vs. 1.76x ratio).
43 of 44 positives clear at least 2.4x their own slide's baseline; the one exception,
`KTR-72502946` fov 54, sits *below* its slide's median (`robust_zscore=-1.05`) despite a
confirmed genuine halo -- see the case study referenced above. Excluding that row, the two
distributions still overlap in the `robust_zscore` ~3-16 range: 17 of 32 negatives (53.1%)
already clear the `high_outlier` threshold (`robust_zscore >= 2`) used elsewhere in this report,
and the second-weakest positive (`robust_zscore=4.81`) sits below several of the strongest
false-alarm-prone negatives.

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

**At a very high threshold (`robust_zscore >= 20`), recall drops to 70.5% but false-positive
rate falls to 3.1%** (31/44 positives vs. 1/32 negatives) -- a plausible operating point if this
were ever used as a coarse pre-filter, but not as a standalone replacement for
`src/overexposure.py`'s pixel-level detector, and not validated beyond this 76-row set. See
`crop-outlier-v2/README.md` for a more rigorous threshold calibration against real (non-labeled)
negative FOVs rather than just `spot_truth=no` rows, which lands on `robust_zscore >= 6`.

**One negative deserves a named callout: `RUB-62501518` (Tanzania, fov 315, `background`).**
`robust_zscore=118.67` is higher than all but 4 of the 41 positive rows -- an outlier among
negatives, not a borderline case. 5,549 detected crops vs. a slide median of 95 is a large,
specific anomaly for a FOV the annotator tagged as ordinary background elevation, not an
overexposure halo. Worth a manual look at the raw image before trusting this row's `spot=no`
label at face value -- it's exactly the kind of case this whole-slide-median approach is
designed to surface for human review, not resolve unilaterally.

### Flagged rows (crop-count metric)

Every row with any of the flags below is listed explicitly, per Emily's request not to bury
unusual cases in prose. `results.csv`'s `flags` column is semicolon-joined; a row can carry
multiple flags.

- **`no_data`** -- did not fire on any row as of the 2026-08-11 Tanzania fallback (previously 3
  rows: `KIT-62500763` and `KTR-72502946`, both Tanzania -- see the update note at the top).
- **`same_slide_contamination`** (35 rows) -- another row in this same 76-row label set shares
  the `sample_id` and therefore sits inside the baseline population. Notably includes
  `LB-D3-2025-10-27-123159-251123404-D-thin-4-1` fov 48 & 49 (both `spot_truth=yes`) -- each
  one's "baseline" partly includes the other's inflated count. With 300+ other FOVs on that
  slide, one contaminating value moves the median/MAD negligibly in this specific case, but the
  flag is set regardless since the *direction* of the risk (baseline pulled toward, not away
  from, the target) is real and worth tracking per-row rather than assuming it's always small.
  (Also now includes `KTR-72502946` fov 54 & 198, recovered by the Tanzania fallback -- see the
  case study.)
- **`mean_median_divergence`** (35 rows) -- `baseline_mean` and `baseline_median` differ by more
  than 25% of the median, a direct signal that a slide's own FOV population probably isn't a
  clean background sample (consistent with the reasoning that motivated median/MAD over
  mean/std in the first place -- see "Method").
- **`zero_or_undefined_baseline`** -- did not fire on any row in this label set.
- **`high_outlier`** (60 rows: 43 positive, 17 negative) / **`low_outlier`** (0 rows) --
  `robust_zscore >= 2` (or `<= -2`). See "Discussion" above for why this threshold alone doesn't
  cleanly separate ground truth.

## Metric 2: parasite-count (`n_positives`)

### Method

Identical to Metric 1 in every respect (same 76-row label set, same whole-slide median/MAD
baseline, same flag definitions) except the per-FOV count itself: `n_positives`, the
count of crops the ML classifier actually confirmed as a parasite, from the same
`fov_summary.csv` (Liberia/Tanzania) or by summing the `positive` column of `spots.csv` grouped
by `fov_id` (Uganda) -- see `crop_counts.py`'s module docstring. `ratio_to_median =
target_n_positives / baseline_median`, `robust_zscore = (target_n_positives - baseline_median) /
baseline_mad`.

Run as `python scripts/crop-outlier-approach/analyze_crop_outliers.py --metric n_positives`,
producing `results_parasites.csv`.

### Results: positive vs. negative ground truth

**A degenerate baseline dominates this metric.** True parasitemia in this dataset is very low
(`processing_summary.csv` reports ~0.001% parasitemia on the slides checked) -- most FOVs on
most slides have **zero** confirmed parasites. That means a slide's median `n_positives` is
usually 0, and its MAD is usually 0 too, which makes `ratio_to_median` and `robust_zscore`
mathematically undefined (`zero_or_undefined_baseline`) for the great majority of rows -- **64
of 76 (84.2%)** as of the 2026-08-11 update (previously 61 of 76 plus 3 separate `no_data` rows
-- same 64-row total, just relabeled now that `KIT-62500763`/`KTR-72502946` resolve to real,
still-degenerate `n_positives` data instead of no data at all). Only 12 rows have a computable
baseline at all (2 positive, 10 negative); every other row is excluded from the summary stats
below.

**Positive summary** (n=44, 42 excluded: all `zero_or_undefined_baseline`)

- median `robust_zscore` **-0.86** (mean -0.86, range -1.72-0.00), n=2
- median `ratio_to_median` **0.83** (mean 0.83, range 0.65-1.00)
- 0/2 (0.0%) at or above 2 MAD above baseline

**Negative summary** (n=32, 22 excluded: all `zero_or_undefined_baseline`)

- median `robust_zscore` **0.23** (mean -0.03, range -1.52-1.05), n=10
- median `ratio_to_median` **1.17** (mean 0.97, range 0.00-2.00)
- 0/10 (0.0%) at or above 2 MAD above baseline

**Full positive-group table:**

| sample_id | fov_id | notes | target_n_positives | baseline_median | baseline_mad | ratio_to_median | robust_zscore | flags |
|---|---|---|---|---|---|---|---|---|
| LB-D10-2025-12-29-150312-0171084-VFPCHC-2-4 | 153 |  | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D10-2025-12-29-150312-0171084-VFPCHC-2-4 | 154 |  | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D10-2025-12-30-083614-0250901VFPCHC-2-1 | 210 |  | 2 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D10-2025-12-30-083614-0250901VFPCHC-2-1 | 227 |  | 9 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D10-2025-12-30-084453-0250071VFPCHC-2-2 | 200 |  | 6 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D11-2025-12-19-111309-0211715-VFPCHC-3-1 | 277 | background | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D11-2025-12-19-131014-0241591-VFPCHC-3-2 | 278 | background | 1 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-09-02-141940-25087110-D-Only-1-2 | 42 | background | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-09-09-093425-250917463-D-Only-1-1 | 166 |  | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-09-27-121918-17217958-D-thin-4-4 | 262 |  | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-03-104211-250917371-D-thin-2-3 | 4 |  | 1 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-03-104643-250917465-D-thin-3-4 | 185 | green | 9 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 114 |  | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 125 |  | 1 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-03-125352-2402169466D-thin-2-1 | 3 | diffuse | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-03-130859-250916865-D-thin-1-4 | 236 |  | 1 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-22-131729-250917745-D-thin-2-3 | 134 | double | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-22-132316-2411189646-D-thin-1-4 | 135 | diffuse | 1 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | 122 | double | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | 238 |  | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-24-132012-25046898-D-thin-1-4 | 3 | diffuse | 2 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-24-132012-25046898-D-thin-1-4 | 305 |  | 9 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-24-162727-230918080-D-thin-1-4 | 8 | diffuse | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-25-105806-180951467-D-thin-1-1 | 270 |  | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-25-150947-250917467-D-thin-3-2 | 235 |  | 1 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-27-123159-251123404-D-thin-4-1 | 48 |  | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-27-123159-251123404-D-thin-4-1 | 49 |  | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-27-124239-250916732-D-thin-1-3 | 301 |  | 1 | 1 | 1.483 | 1.0 | 0.0 | mean_median_divergence |
| LB-D3-2025-10-27-134711-250917368-D-thin-1-3 | 52 |  | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-27-154305-250917412-D-thin-1-4 | 119 | diffuse | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-27-173317-250917493-D-thin-2-4 | 82 |  | 4 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| KIT-62500763 | 200 | green | 153 | 0.0 | 0.0 |  |  | zero_or_undefined_baseline |
| KIT-62501035 | 67 |  | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| KIT-62501081 | 141 | double | 1 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| KIT-62501087 | 271 |  | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| KTR-72502946 | 54 |  | 0 | 0.0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| KTR-72502946 | 198 |  | 0 | 0.0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| NKR-72502319 | 293 |  | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| NKR-72502319 | 311 |  | 1 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| RUB-62501332 | 133 |  | 4 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| RUB-62501389 | 284 | double | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| RUB-72501756 | 315 |  | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| PAT-070-3 | 34 | double | 43 | 66 | 13.343 | 0.652 | -1.724 |  |
| PBC-608-KH-1 | 171 |  | 0 | 0.0 | 0.0 |  |  | zero_or_undefined_baseline |

**Full negative-group table:**

| sample_id | fov_id | notes | target_n_positives | baseline_median | baseline_mad | ratio_to_median | robust_zscore | flags |
|---|---|---|---|---|---|---|---|---|
| LB-D11-2025-12-17-115859-0250319D-thin-4-1 | 29 | background | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D11-2025-12-19-134126-025073-VFPCHC-3-1 | 1 | background | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 257 | background | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 269 | background | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 274 | background | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 279 | background | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 289 | background | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 1 | background | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 16 | background | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 17 | background | 1 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 18 | background | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 19 | background | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 53 | background | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 126 | artifact | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-24-113736-250918214-D-thin-2-3 | 96 | relabeled by Emily 2026-08-07 (was yes) | 3 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D3-2025-10-27-144635-250918691-D-thin-2-2 | 57 | artifact | 1 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-27-144635-250918691-D-thin-2-2 | 243 | background | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| LB-D3-2025-10-27-145205-250917002-D-thin-3-3 | 310 | artifact | 2 | 1 | 1.483 | 2.0 | 0.674 | mean_median_divergence |
| LB-D3-2025-10-27-155920-250713919-D-thin-3-3 | 169 | background | 2 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| LB-D5-2026-01-27-112616-0240052-VFPCHC-2-2 | 40 | background | 0 | 3 | 2.965 | 0.0 | -1.012 |  |
| KIT-62501062 | 83 | artifact | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| NKR-72502319 | 119 | background | 0 | 0 | 0.0 |  |  | same_slide_contamination;zero_or_undefined_baseline |
| RUB-62501518 | 315 | background | 0 | 0 | 0.0 |  |  | zero_or_undefined_baseline |
| RUB-62501529 | 87 | background | 7 | 16 | 5.93 | 0.438 | -1.518 |  |
| PAT-072-1 | 14 | artifact | 8 | 6 | 4.448 | 1.333 | 0.45 | same_slide_contamination |
| PAT-072-1 | 94 | artifact | 4 | 6 | 4.448 | 0.667 | -0.45 | same_slide_contamination |
| PAT-154-1 | 478 | background | 1 | 0.0 | 0.0 |  |  | zero_or_undefined_baseline |
| PBC-225_AM-1 | 30 | background | 90 | 65.0 | 23.722 | 1.385 | 1.054 |  |
| PBC-800-1 | 128 | background | 0 | 1.0 | 1.483 | 0.0 | -0.674 | same_slide_contamination |
| PBC-800-1 | 732 | background | 1 | 1.0 | 1.483 | 1.0 | 0.0 | same_slide_contamination |
| PAT-103-2 | 441 | background | 61 | 45 | 34.1 | 1.356 | 0.469 |  |
| PAT-112-2 | 124 | background | 3 | 2.0 | 1.483 | 1.5 | 0.674 |  |

### Discussion: does the parasite-count signal separate ground truth?

**No.** Of the 12 rows where a baseline is even computable, z-scores range from -1.72 to +1.05
for both groups combined -- noise-level, no outliers in either direction, and no separation
between `spot_truth=yes` and `spot_truth=no`. `PAT-070-3` (positive, `double`) is the only
positive with a usable baseline that carries any real parasite burden (43 confirmed parasites),
and it sits *below* its slide's median (`robust_zscore=-1.72`), not above it.

This isn't just "no signal" -- it's a genuine structural mismatch between this whole-slide
median/MAD approach and this metric. The approach assumes a slide has enough non-degenerate
variation in the target count to define a meaningful "typical" FOV; `n_positives` violates that
assumption on 80% of rows because the typical FOV, on the vast majority of slides in this
dataset, has zero confirmed parasites. Where a slide's population is dense enough to
support a real median (the 12 usable rows, mostly higher-parasitemia Uganda samples like
`PAT-070-3`, `PAT-103-2`, `PBC-225_AM-1`), the metric simply doesn't distinguish the two ground-
truth groups at all.

### Flagged rows (parasite-count metric)

- **`no_data`** -- did not fire on any row as of the 2026-08-11 Tanzania fallback (previously 3
  rows, same two samples as the crop-count metric).
- **`zero_or_undefined_baseline`** (64 rows, 84.2% of the dataset) -- see "Discussion" above.
  This is the dominant flag for this metric, a sharp contrast with the crop-count metric where
  it fired on zero rows.
- **`same_slide_contamination`** (35 rows) -- identical set of rows to the crop-count metric
  (contamination is about which *other* labeled rows share a slide, not about the metric being
  measured).
- **`mean_median_divergence`** (2 rows: `LB-D3-2025-10-27-124239-...` fov 301,
  `LB-D3-2025-10-27-145205-...` fov 310) -- both are among the 12 rows with a computable
  baseline; both have `baseline_median=1`, where even a 1-count difference between mean and
  median exceeds the 25%-of-median threshold. A reminder that this flag's threshold, tuned with
  the crop-count metric's much larger typical values in mind, behaves differently at
  parasite-count's much smaller scale.
- **`high_outlier` / `low_outlier`** -- did not fire on any row for this metric.

## Comparing the two metrics

**The crop-count metric shows a strong, if imperfect, separation; the parasite-count metric
shows essentially none.** Side by side:

| | Crop-count (`n_spots_detected`) | Parasite-count (`n_positives`) |
|---|---|---|
| Rows with a computable baseline | 76/76 (100%) | 12/76 (15.8%) |
| `zero_or_undefined_baseline` rate | 0/76 (0%) | 64/76 (84.2%) |
| Positive median `robust_zscore` | 51.78 | -0.86 (n=2) |
| Negative median `robust_zscore` | 2.21 | 0.23 (n=10) |
| Positives at or above `robust_zscore >= 2` | 43/44 (97.7%) | 0/2 (0%) |
| Negatives at or above `robust_zscore >= 2` | 17/32 (53.1%) | 0/10 (0%) |

**This is consistent with a specific, mechanistic story, not a coincidence.** The overexposure
halo lives in the same blue-channel intensity map that the raw candidate-spot detector
thresholds for local maxima -- a bright halo plausibly creates many spurious local maxima,
inflating `n_spots_detected` directly. But `n_positives` is what's *left after* the ML
classifier scores each candidate crop and rejects the ones that don't look like a real parasite.
If the classifier is doing its job, most halo-caused candidates should be rejected before they
ever become a confirmed positive -- which is exactly what the data shows: every positive FOV's
raw crop count is inflated (metric 1), but essentially none of that inflation survives into
confirmed parasite counts (metric 2, where the signal disappears into noise).

**A second, independent reason `n_positives` can't work here regardless of the above:** true
parasitemia in this dataset is low enough that most FOVs have zero confirmed parasites, so a
whole-slide median/MAD baseline is mathematically undefined for 80% of rows before the halo
question even comes into play. Median/MAD (chosen specifically for contamination-resistance,
see Metric 1's Method) needs enough non-zero spread in the population to define a "typical"
value at all -- a requirement `n_spots_detected` easily satisfies (dozens to thousands of crops
per FOV) and `n_positives` mostly doesn't (0-2 confirmed parasites on most FOVs).

**Practical takeaway:** if a crop-count-based outlier check were ever built into a real
pre-filter, it should operate on raw candidate counts (`n_spots_detected`) upstream of the
classifier, not on the classifier's own output (`n_positives`) -- the whole point of catching
the halo's effect on crop counts is to catch it *before* it reaches (and burdens) the
classifier, and this data suggests the classifier is already absorbing most of that burden on
its own by the time `n_positives` is computed.

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
- **Tanzania fallback (2026-08-11).** `tanzania_02032026`'s own `detection_results/` tree is
  missing 146 slides across `TZ2025-Box1`/`Box5` (94 and 52 respectively) that were never
  mirrored into it; `crop_counts.py` now falls back to `malaria-annotation-web`'s copy for these.
  Verified byte-identical to `tanzania_02032026`'s own data on a sample that has both. Not
  independently verified whether Liberia has an analogous gap -- no evidence of one in this
  76-row label set (all Liberia rows resolved before and after this change), but this set only
  covers 33 Liberia slides out of however many exist in `liberia-2025` overall.
- **Uganda's `n_spots_detected` is mislabeled (found 2026-08-11, not fixed).** Uganda's only
  source (`samples/<id>/spots.csv` in `malaria-annotation-web`) actually reproduces
  `n_spots_filtered`, not `n_spots_detected`, per a cross-check against a Tanzania sample with
  both files. The 10 Uganda rows in the tables above are on a different, already-partially-
  filtered metric than every Liberia/Tanzania row. Uganda has no `fov_summary.csv` anywhere to
  recover true `n_spots_detected` from -- unresolved whether it's recoverable at all.
- **`v8_hardneg_single_t0.995` was used for every Liberia/Tanzania row**, chosen because
  `n_spots_detected` was verified identical across all available model-version run folders for
  the slides checked (the upstream spot-finding step is shared across models). `n_positives`
  legitimately does depend on which classifier produced it (unlike `n_spots_detected`), so this
  choice matters more for Metric 2 -- not re-verified against other model versions for this
  label set.
- **Metric 2's finding is a null result on a small, degenerate sample (12 rows with a
  computable baseline).** "No separation" here is a real, reproducible finding given the data
  available, but it's not the same strength of evidence as Metric 1's 76/76-row result --
  treat it as directional, not a settled conclusion about the classifier's robustness in
  general.

## Files

- `results.csv` -- Metric 1 (`n_spots_detected`) full per-row output
- `results_parasites.csv` -- Metric 2 (`n_positives`) full per-row output
- `../../../scripts/crop-outlier-approach/crop_counts.py` -- GCS resolution + per-sample cache +
  baseline stats for both metrics
- `../../../scripts/crop-outlier-approach/analyze_crop_outliers.py` -- pipeline that produced both
  CSVs (`--metric n_spots_detected` default, or `--metric n_positives`)
- `../../../scripts/crop-outlier-approach/report_tables.py` -- generates the tables in this doc for
  either metric's CSV (target-count column is auto-detected from the header)
- `crop-outlier-v2/` -- redefines positive/negative away from `spot_truth` and towards the
  crop-count signal itself, using boundary-FOV negatives; `crop-outlier-v2/case-study-KTR-72502946.md`
  is a full-slide look at the sample recovered by the Tanzania fallback above.
