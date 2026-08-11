# Crop-outlier v2: redefining positive/negative via boundary-FOV negatives (2026-08-11, updated 2026-08-11)

v1 (`../README.md`) tested the crop-count outlier signal against the labels CSV's `spot_truth`
ground truth (whether a human confirmed a genuine overexposure-halo artifact). This extension
**stops using `spot_truth` as the definition of positive/negative** for this approach. Instead:

> **Positive (filter-worthy) = a significant excess of erroneous crops on that FOV, full stop —
> independent of whether a human separately confirmed a halo artifact is present.**

The 76-row labeled set is *not* treated as ground truth negatives here; it's expected to be
mostly (but not entirely) positive under this new definition, since v1 already showed every
`spot_truth=yes` row (bar one, see below) carries an inflated crop count, and some `spot_truth=no`
rows (`background`/`artifact`-tagged) do too. To get real negatives, this analysis uses **FOV 1
and FOV 324** — the first and last tile of the fixed 18×18 virtual raster (`src/gcs_fov.py`) that
`fov_id` already addresses throughout this codebase — on each of the **53 distinct slides**
represented in the labels CSV (76 labeled rows span only 53 slides; 12 slides contribute 2–9
labeled rows each). These are presumed-blank tiles, unless one of them is itself an outlier on
its own slide, in which case it's flagged as a candidate-contaminated negative rather than
trusted.

**Same-day update:** two changes to `crop_counts.py` (see its module docstring and `../README.md`'s
update note) were applied and this doc fully re-run:
1. Baseline is now whole-slide (median/MAD over every FOV on the slide, the target included),
   not leave-one-out — a single baseline per slide, reused for both boundary FOVs and every
   labeled FOV on it.
2. A Tanzania fallback to `malaria-annotation-web` recovered `KIT-62500763` and `KTR-72502946`,
   previously `no_data` in both this doc and v1's. `../case-study-KTR-72502946.md` is a full-slide
   case study on the more interesting of the two.

## Method

Reuses v1's exact statistic — no new metric. For each of the 53 slides and each of `fov_id`
1/324: `crop_counts.load_slide_metric_counts(sample_id, country, metric="n_spots_detected")`
(same GCS-backed per-FOV counts v1 uses, now with the Tanzania fallback) →
`crop_counts.slide_baseline(counts)` (whole-slide median/MAD, the target FOV included) →
`ratio_to_median = count / baseline_median`, `robust_zscore = (count - baseline_median) /
baseline_mad`. Same flags as v1 (`small_slide`, `zero_or_undefined_baseline`,
`mean_median_divergence`, `high_outlier`/`low_outlier` at `robust_zscore >= 2.0`/`<= -2.0`).

**Code:** `scripts/crop-outlier-approach/analyze_boundary_negatives.py` (produces
`boundary_negatives.csv`), `scripts/crop-outlier-approach/v2_report.py` (produces every table
below).

**Contamination flag.** A boundary row flagged `high_outlier` (its own slide's
`robust_zscore >= 2`) is a *candidate-contaminated negative*, not a trustworthy blank — it's
listed explicitly below rather than silently trusted or silently dropped.

**Missing boundary data.** `fov_id=324` is missing from detection results on 5 of 53 slides
(smaller physical scan raster not reaching that address) — this is now the only source of
missing boundary data; the Tanzania fallback eliminated the other previous gap (both
`KIT-62500763` and `KTR-72502946` now resolve). 101 of 106 boundary rows (53 slides × 2 FOVs) are
usable.

## Results

### Per-slide comparison: labeled FOV(s) vs. boundary FOVs 1/324

The literal ask — crop count (and its own-slide `robust_zscore`) for FOV 1 and FOV 324 next to
the labeled FOV(s) on the same slide:

| sample_id | country | labeled fov_id (spot_truth, n_spots) | fov1 n_spots (z) | fov324 n_spots (z) |
|---|---|---|---|---|
| LB-D10-2025-12-29-150312-0171084-VFPCHC-2-4 | Liberia | 153 (yes, 1818); 154 (yes, 2866) | 50 (z=0.175) | no_data |
| LB-D10-2025-12-30-083614-0250901VFPCHC-2-1 | Liberia | 210 (yes, 4005); 227 (yes, 3168) | 176 (z=0.216) | 75 (z=-1.147) |
| LB-D10-2025-12-30-084453-0250071VFPCHC-2-2 | Liberia | 200 (yes, 5072) | 250 (z=3.07) [CONTAMINATED] | 64 (z=-0.674) |
| LB-D11-2025-12-17-115859-0250319D-thin-4-1 | Liberia | 29 (no, 3473) | 1636 (z=0.054) | 592 (z=-1.523) |
| LB-D11-2025-12-19-111309-0211715-VFPCHC-3-1 | Liberia | 277 (yes, 6003) | 1340 (z=-0.801) | 1486 (z=-0.562) |
| LB-D11-2025-12-19-131014-0241591-VFPCHC-3-2 | Liberia | 278 (yes, 7231) | 3111 (z=0.79) | 1431 (z=-0.986) |
| LB-D11-2025-12-19-134126-025073-VFPCHC-3-1 | Liberia | 1 (no, 3552) | 3552 (z=1.422) | 2656 (z=0.663) |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | Liberia | 257 (no, 2477); 269 (no, 1385); 274 (no, 1864); 279 (no, 1468); 289 (no, 2433) | 538 (z=-0.061) | 554 (z=-0.019) |
| LB-D3-2025-09-02-141940-25087110-D-Only-1-2 | Liberia | 42 (yes, 1710) | 406 (z=-0.039) | 390 (z=-0.108) |
| LB-D3-2025-09-09-093425-250917463-D-Only-1-1 | Liberia | 166 (yes, 1998) | 158 (z=-0.241) | 137 (z=-0.747) |
| LB-D3-2025-09-27-121918-17217958-D-thin-4-4 | Liberia | 262 (yes, 9960) | 363 (z=0.005) | 773 (z=4.227) [CONTAMINATED] |
| LB-D3-2025-10-03-104211-250917371-D-thin-2-3 | Liberia | 4 (yes, 8239) | 326 (z=3.391) [CONTAMINATED] | no_data |
| LB-D3-2025-10-03-104643-250917465-D-thin-3-4 | Liberia | 185 (yes, 2119) | 255 (z=1.096) | no_data |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | Liberia | 1 (no, 261); 16 (no, 365); 17 (no, 385); 18 (no, 390); 19 (no, 190); 53 (no, 484); 114 (yes, 4571); 125 (yes, 534); 126 (no, 1181) | 261 (z=0.629) | no_data |
| LB-D3-2025-10-03-125352-2402169466D-thin-2-1 | Liberia | 3 (yes, 2656) | 309 (z=1.303) | 237 (z=-0.343) |
| LB-D3-2025-10-03-130859-250916865-D-thin-1-4 | Liberia | 236 (yes, 2907) | 346 (z=0.009) | 510 (z=1.525) |
| LB-D3-2025-10-22-131729-250917745-D-thin-2-3 | Liberia | 134 (yes, 9946) | 507 (z=2.191) [CONTAMINATED] | 735 (z=4.312) [CONTAMINATED] |
| LB-D3-2025-10-22-132316-2411189646-D-thin-1-4 | Liberia | 135 (yes, 2858) | 681 (z=1.815) | 389 (z=-0.061) |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | Liberia | 122 (yes, 9030); 238 (yes, 1524) | 495 (z=2.438) [CONTAMINATED] | 136 (z=-0.57) |
| LB-D3-2025-10-24-113736-250918214-D-thin-2-3 | Liberia | 96 (no, 957) | 157 (z=-1.831) | no_data |
| LB-D3-2025-10-24-132012-25046898-D-thin-1-4 | Liberia | 3 (yes, 2797); 305 (yes, 3431) | 594 (z=0.726) | 575 (z=0.562) |
| LB-D3-2025-10-24-162727-230918080-D-thin-1-4 | Liberia | 8 (yes, 2929) | 1061 (z=-0.08) | 920 (z=-0.914) |
| LB-D3-2025-10-25-105806-180951467-D-thin-1-1 | Liberia | 270 (yes, 2832) | 291 (z=1.396) | 150 (z=-0.272) |
| LB-D3-2025-10-25-150947-250917467-D-thin-3-2 | Liberia | 235 (yes, 2501) | 179 (z=-1.012) | 640 (z=1.79) |
| LB-D3-2025-10-27-123159-251123404-D-thin-4-1 | Liberia | 48 (yes, 5105); 49 (yes, 2566) | 198 (z=0.507) | 73 (z=-1.098) |
| LB-D3-2025-10-27-124239-250916732-D-thin-1-3 | Liberia | 301 (yes, 9410) | 409 (z=-0.423) | 366 (z=-0.809) |
| LB-D3-2025-10-27-134711-250917368-D-thin-1-3 | Liberia | 52 (yes, 6375) | 112 (z=0.123) | 260 (z=4.66) [CONTAMINATED] |
| LB-D3-2025-10-27-144635-250918691-D-thin-2-2 | Liberia | 57 (no, 492); 243 (no, 558) | 435 (z=0.468) | 319 (z=-0.401) |
| LB-D3-2025-10-27-145205-250917002-D-thin-3-3 | Liberia | 310 (no, 342) | 209 (z=0.525) | 208 (z=0.506) |
| LB-D3-2025-10-27-154305-250917412-D-thin-1-4 | Liberia | 119 (yes, 1037) | 259 (z=1.4) | 407 (z=3.927) [CONTAMINATED] |
| LB-D3-2025-10-27-155920-250713919-D-thin-3-3 | Liberia | 169 (no, 256) | 329 (z=0.525) | 756 (z=5.858) [CONTAMINATED] |
| LB-D3-2025-10-27-173317-250917493-D-thin-2-4 | Liberia | 82 (yes, 4501) | 197 (z=1.242) | 186 (z=1.047) |
| LB-D5-2026-01-27-112616-0240052-VFPCHC-2-2 | Liberia | 40 (no, 5023) | 3108 (z=1.509) | 1926 (z=0.359) |
| KIT-62500763 | Tanzania | 200 (yes, 3432) | 768 (z=-0.158) | 1154 (z=1.49) |
| KIT-62501035 | Tanzania | 67 (yes, 3656) | 171 (z=0.487) | 223 (z=1.636) |
| KIT-62501062 | Tanzania | 83 (no, 187) | 248 (z=1.544) | 73 (z=-0.726) |
| KIT-62501081 | Tanzania | 141 (yes, 2335) | 75 (z=0.011) | 102 (z=0.629) |
| KIT-62501087 | Tanzania | 271 (yes, 8660) | 36 (z=-0.18) | 109 (z=3.103) [CONTAMINATED] |
| KTR-72502946 | Tanzania | 54 (yes, 25); 198 (yes, 4417) | 49 (z=-0.15) | 50 (z=-0.112) |
| NKR-72502319 | Tanzania | 119 (no, 92); 293 (yes, 5938); 311 (yes, 5734) | 134 (z=2.023) [CONTAMINATED] | 76 (z=0.067) |
| RUB-62501332 | Tanzania | 133 (yes, 2797) | 189 (z=-0.508) | 322 (z=2.252) [CONTAMINATED] |
| RUB-62501389 | Tanzania | 284 (yes, 9453) | 52 (z=-0.861) | 82 (z=-0.163) |
| RUB-62501518 | Tanzania | 315 (no, 5549) | 47 (z=-1.044) | 97 (z=0.044) |
| RUB-62501529 | Tanzania | 87 (no, 333) | 211 (z=-0.21) | 299 (z=0.879) |
| RUB-72501756 | Tanzania | 315 (yes, 2125) | 160 (z=-0.18) | 241 (z=1.09) |
| PAT-070-3 | Uganda | 34 (yes, 945) | 133 (z=-1.307) | 239 (z=0.927) |
| PAT-072-1 | Uganda | 14 (no, 324); 94 (no, 170) | 66 (z=0.719) | 46 (z=-0.18) |
| PAT-154-1 | Uganda | 478 (no, 2606) | 1606 (z=-0.644) | 64 (z=-1.156) |
| PBC-225_AM-1 | Uganda | 30 (no, 2852) | 382 (z=-3.532) | 3417 (z=0.68) |
| PBC-608-KH-1 | Uganda | 171 (yes, 1913) | 67 (z=0.562) | 50 (z=-0.393) |
| PBC-800-1 | Uganda | 128 (no, 102); 732 (no, 176) | 62 (z=-0.797) | 65 (z=-0.736) |
| PAT-103-2 | Uganda | 441 (no, 577) | 89 (z=-1.108) | 3794 (z=2.213) [CONTAMINATED] |
| PAT-112-2 | Uganda | 124 (no, 147) | 138 (z=0.499) | 114 (z=-0.205) |

Note the `KTR-72502946` row: fov 54 (25 crops, `spot_truth=yes`) is actually below both of that
slide's own boundary FOVs (49 and 50 crops) — see `../case-study-KTR-72502946.md`.

### Boundary rows flagged `high_outlier` — candidate-contaminated negatives (n=13 of 106)

Not trusted as clean blanks; called out explicitly rather than silently dropped or silently kept:

| sample_id | boundary_fov_id | n_spots_detected | baseline_median | baseline_mad | ratio_to_median | robust_zscore |
|---|---|---|---|---|---|---|
| LB-D10-2025-12-30-084453-0250071VFPCHC-2-2 | 1 | 250 | 97.5 | 49.667 | 2.564 | 3.07 |
| LB-D3-2025-09-27-121918-17217958-D-thin-4-4 | 324 | 773 | 362.5 | 97.11 | 2.132 | 4.227 |
| LB-D3-2025-10-03-104211-250917371-D-thin-2-3 | 1 | 326 | 145.0 | 53.374 | 2.248 | 3.391 |
| LB-D3-2025-10-22-131729-250917745-D-thin-2-3 | 1 | 507 | 271.5 | 107.488 | 1.867 | 2.191 |
| LB-D3-2025-10-22-131729-250917745-D-thin-2-3 | 324 | 735 | 271.5 | 107.488 | 2.707 | 4.312 |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | 1 | 495 | 204.0 | 119.349 | 2.426 | 2.438 |
| LB-D3-2025-10-27-134711-250917368-D-thin-1-3 | 324 | 260 | 108.0 | 32.617 | 2.407 | 4.66 |
| LB-D3-2025-10-27-154305-250917412-D-thin-1-4 | 324 | 407 | 177.0 | 58.563 | 2.299 | 3.927 |
| LB-D3-2025-10-27-155920-250713919-D-thin-3-3 | 324 | 756 | 287.0 | 80.06 | 2.634 | 5.858 |
| KIT-62501087 | 324 | 109 | 40.0 | 22.239 | 2.725 | 3.103 |
| NKR-72502319 | 1 | 134 | 74.0 | 29.652 | 1.811 | 2.023 |
| RUB-62501332 | 324 | 322 | 213.5 | 48.184 | 1.508 | 2.252 |
| PAT-103-2 | 324 | 3794 | 1325.0 | 1115.657 | 2.863 | 2.213 |

Same 13 rows as before this update (methodology/data changes shifted individual `robust_zscore`
values slightly but didn't add or remove any flagged row).

**All usable boundary rows** (n=101, no_data excluded): median `robust_zscore` 0.07 (mean 0.48,
range -3.53–5.86), median `ratio_to_median` 1.03 (mean 1.18, range 0.02–2.86).

**Clean boundary negatives** (n=88, the 13 `high_outlier` rows above excluded): median
`robust_zscore` -0.03 (mean 0.05, range -3.53–1.81), median `ratio_to_median` 0.99 (mean 1.02,
range 0.02–2.00) — this centers almost exactly on 0/1.0, the sanity check that FOV1/324 behave
like an ordinary tile on their own slide once the 13 contaminated ones are set aside.

### Threshold sweep

FPR is computed against **all 101 usable boundary rows, including the 13 flagged ones** — using
the `high_outlier`-filtered ("clean") pool here would be circular, since that flag is itself
`robust_zscore >= 2`, which would trivially force 0% FPR at every tier ≥2. The 13 flagged rows
are still real observations of presumed-blank tiles; excluding them from the denominator would
assume the answer.

| `robust_zscore` >= | FPR (101 boundary negatives) | recall (spot_truth=yes, n=44) | flip rate (spot_truth=no, n=32) |
|---|---|---|---|
| 2 | 12.9% | 97.7% | 53.1% |
| 3 | 7.9% | 97.7% | 34.4% |
| 5 | 1.0% | 95.5% | 18.8% |
| **6** | **0.0%** | **90.9%** | **12.5%** |
| 8 | 0.0% | 88.6% | 12.5% |
| 10 | 0.0% | 88.6% | 12.5% |
| 15 | 0.0% | 77.3% | 3.1% |
| 20 | 0.0% | 70.5% | 3.1% |
| 30 | 0.0% | 65.9% | 3.1% |
| 50 | 0.0% | 52.3% | 3.1% |

**The single highest boundary-negative `robust_zscore` in the entire dataset is still 5.858**
(`LB-D3-2025-10-27-155920-250713919-D-thin-3-3`, FOV 324, 756 crops vs. a slide median of 287 —
see named callout below; unaffected by this update since it's a Liberia slide). Every threshold
above that value achieves 0% FPR against the full 101-row empirical negative pool, and
`robust_zscore >= 6` is still the lowest such threshold, so it's also the one with the best
recall among all zero-FPR options (90.9%, vs. 88.6% at 8–10 and falling further at higher tiers).
**Recall dropped from 92.7% to 90.9% relative to the previous version of this doc** — entirely
due to `KTR-72502946` fov 54 (25 crops, `robust_zscore=-1.05`) joining the labeled set as a new,
clean miss; see the case study.

### New-vs-old cross-tab at the recommended threshold (`robust_zscore >= 6`)

n=76 usable of 76 labeled rows (0 `no_data` now, vs. 3 before this update):

| | new_positive | new_negative |
|---|---|---|
| spot_truth=yes | 40 | 4 |
| spot_truth=no | 4 | 28 |

44 of 76 usable labeled rows (57.9%) are "positive" under the new, crop-count-only definition —
consistent with the "likely largely positive" expectation, without being unanimous: 4
`spot_truth=yes` rows fall below the new threshold (false negatives relative to the halo-artifact
label), and 4 `spot_truth=no` rows clear it (the new definition catches something beyond just the
halo — see Discussion).

## Discussion

**Recommended threshold: `robust_zscore >= 6`.** Unchanged by this update. This is the lowest
cutoff that produces zero false positives against the full empirical boundary-negative
population (101 rows across 53 slides), and among all zero-FPR cutoffs it has the best recall
(90.9%) against the known overexposure-halo cases. It sits in a real, if narrow, gap: the
highest boundary negative is 5.858 and the next-lowest labeled `spot_truth=yes` row above it is
6.823 — a margin of about 1 unit, not a wide separation. This mirrors the manual-gap-calibration
style used for every other threshold in this repo (`RATIO_THRESHOLD`, `ANISOTROPY_THRESHOLD`,
etc.) rather than introducing new statistical machinery.

**This is meaningfully higher than v1's ad hoc `OUTLIER_ZSCORE=2.0`.** At `robust_zscore >= 2`,
12.9% of genuinely presumed-blank boundary tiles would already be misclassified as positive — a
false-positive rate that wasn't visible in v1 because v1 never had an independent negative
population to check against; it only had `spot_truth=no` labeled rows, which (per v1's own
discussion) are frequently `background`/`artifact` FOVs with legitimately elevated counts for
reasons unrelated to any artifact. `robust_zscore >= 6` is the first calibration of this
approach's threshold against FOVs that were never selected for being visually unusual in any way.

**The new definition doesn't just reproduce `spot_truth`.** At the recommended threshold: 4 of 44
`spot_truth=yes` rows don't clear it — three in the 4.8–5.6 z-score range
(`LB-D11-2025-12-19-131014-...` fov 278, `LB-D3-2025-09-02-141940-...` fov 42,
`LB-D3-2025-10-03-124025-...` fov 125) and one, `KTR-72502946` fov 54, a clean miss below its own
slide's median (`robust_zscore=-1.05`) — a genuine halo was confirmed by a human in all four
cases, but this particular crop-count signal is weaker than the boundary-negative noise floor
allows for. And 4 of 32 `spot_truth=no` rows *do* clear it:
`LB-D3-2025-10-03-124025-2404175445D-thin-2-3` fov 126 (`artifact`, `robust_zscore=14.73`),
`LB-D3-2025-10-24-113736-250918214-D-thin-2-3` fov 96 (the row Emily relabeled from `yes` to `no`
on 2026-08-07, `robust_zscore=11.02` — still crop-count-anomalous even after the relabel),
`RUB-62501518` fov 315 (`background`, `robust_zscore=118.67` — already called out by name in v1's
own discussion), and `PAT-072-1` fov 14 (`artifact`, `robust_zscore=12.32`). All four are
`background`/`artifact`-tagged, the same subset v1 already identified as the source of its own
ground-truth overlap. This is expected and consistent with v1's own finding that the crop-count
signal and the halo-artifact ground truth are correlated but not identical — the new definition
is deliberately not trying to recover `spot_truth`, only to flag FOVs with excess erroneous
crops for any reason.

**Named callout: `LB-D3-2025-10-27-155920-250713919-D-thin-3-3`, FOV 324.** `robust_zscore=5.858`
is the boundary negative that sets the threshold ceiling — 756 crops vs. a slide median of 287
(2.63x). This slide's own labeled FOV (169, `spot_truth=no`, `background`) sits *below* its slide
median (256 vs. 287, `robust_zscore=-0.387` per v1's results.csv) — i.e. the corner tile this
analysis assumed was blank has a more anomalous crop count than the FOV a human specifically
inspected and called clean. Worth a manual look at the raw FOV 324 image before trusting the
"boundary tile = blank" assumption at face value on this slide, in the same spirit as v1's
`RUB-62501518` callout.

**Second named callout, new in this update: `KTR-72502946`.** This slide has both the strongest
new true positive (fov 198, `robust_zscore=163.5`, by far the most extreme value found anywhere
in this dataset outside `KIT-62501087`) and the cleanest false negative (fov 54,
`robust_zscore=-1.05`) among its own two labeled FOVs, while both its boundary FOVs (1 and 324)
land almost exactly on the slide median as expected. `case-study-KTR-72502946.md` breaks down all
324 of its FOVs and includes raw fluorescence images for the three that matter.

**The case study also surfaces a limitation worth reading before trusting the threshold above.**
One unlabeled FOV on that slide (308, `robust_zscore=7.83`) clears the recommended threshold, and
its image shows it's **debris — bright punctate specks, no halo at all**. Under the halo-focused
framing that motivated this whole approach it's a false positive; under v2's own redefinition
(excess erroneous crops *regardless of cause*) it's arguably a true positive. A whole-frame crop
count can't tell "one big halo inflated my count" from "lots of little bright junk inflated my
count," so the `robust_zscore >= 6` threshold should be read as flagging *junk-inflated FOVs
generally*, not *halos specifically*. That's the `background`/`artifact` confound v1 already
named, now with a concrete image behind it.

## Caveats

- **FOV1/324-are-blank is an assumption, not verified per-slide.** A corner tile of the scan
  raster is *usually* off to the side of the smear, not guaranteed to be. 13 of 101 usable
  boundary rows (12.9%) were themselves outliers on their own slide — evidence the assumption
  has a real, non-trivial failure rate, which is exactly why the threshold above is calibrated
  against the full population (contaminated rows included) rather than the "clean" subset.
- **5 of 53 slides have no detection data at `fov_id=324`** (smaller physical scan raster than
  the fixed 18×18 addressing this analysis assumes) — the only remaining source of missing
  boundary data as of the 2026-08-11 Tanzania fallback (see update note at top; previously this
  also included 2 fully-`no_data` samples, both now resolved).
- **The boundary baseline pool can include the slide's own labeled FOV(s)** among its ~300+
  other FOVs — v1 already reasons this effect is negligible at that pool size
  (`same_slide_contamination`'s discussion in v1's README), and it applies identically here.
- **Still a 53-slide, hand-picked set** (the labels CSV was curated to be "visually diverse," not
  representative), and the 101-row boundary-negative population, while independent of any
  labeling decision, comes from the same 53 slides — not a random sample of all scanned slides.
  Treat `robust_zscore >= 6` as a calibrated starting point, not a validated production
  threshold.
- **`n_spots_detected` only.** Per v1's cross-metric comparison, `n_positives` is degenerate
  (undefined baseline) for ~84% of rows in this dataset and was not re-tested here.
- **Uganda's 10 labeled rows are on a different metric than intended** (`n_spots_filtered`, not
  `n_spots_detected` — see `../README.md`'s Caveats and `crop_counts.py`'s module docstring).
  Not corrected in this update; flagged as a known limitation of every Uganda row above.

## Files

- `boundary_negatives.csv` — full per-boundary-FOV output (106 rows: 53 slides × FOV 1 and 324)
- `case-study-KTR-72502946.md` / `case-study-KTR-72502946.csv` — full-slide case study on the
  more interesting of the two samples recovered by the 2026-08-11 Tanzania fallback
- `previews/` — raw fluorescence thumbnails for the three case-study FOVs (54, 198, 308)
- `../../../../scripts/crop-outlier-approach/analyze_boundary_negatives.py` — pipeline that
  produced `boundary_negatives.csv`
- `../../../../scripts/crop-outlier-approach/v2_report.py` — generates every table in this doc
  from `boundary_negatives.csv` and v1's `../results.csv`
- `../../../../scripts/crop-outlier-approach/case_study_ktr72502946.py` — generates the case study
- `../results.csv` — v1's labeled-FOV output, reused here unmodified (not regenerated by this doc)
