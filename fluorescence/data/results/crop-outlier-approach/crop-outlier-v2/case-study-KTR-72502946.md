# Case study: KTR-72502946 (Tanzania), full slide (2026-08-11)

This slide was previously invisible to this whole pipeline. It's one of the two samples
(`KIT-62500763`, `KTR-72502946`) that showed up as `no_data` in both v1's `results.csv` and v2's
`boundary_negatives.csv` -- `tanzania_02032026`'s own `detection_results/` tree never got it
mirrored in. It has full detection data all along, in `gs://malaria-annotation-web`
(`samples/KTR-72502946/fov_summary.csv`) -- see `crop_counts.py`'s module docstring for the
fallback that now finds it (and `../README.md`'s Caveats section, updated the same day, for the
broader Tanzania gap this slide is one instance of: 146 slides across `TZ2025-Box1`/`Box5` never
mirrored into `tanzania_02032026`, all recoverable the same way).

## Method

Every one of this slide's 324 FOVs, scored against **one whole-slide baseline** (median and MAD
of `n_spots_detected` over all 324 FOVs, the target included -- `crop_counts.slide_baseline()`,
computed once per slide as of today's pipeline change, not per-target leave-one-out).
`ratio_to_median = n_spots_detected / baseline_median`, `robust_zscore = (n_spots_detected -
baseline_median) / baseline_mad`.

**Baseline for this slide:** median 53, MAD 26.687, mean 76.265, std 244.572 (a large
mean/median divergence -- expected, and explained below: one FOV is such an extreme outlier it
pulls the mean far from the median while barely moving it).

**Code:** `scripts/crop-outlier-approach/case_study_ktr72502946.py`, reusing
`crop_counts.load_slide_metric_counts`/`slide_baseline` unmodified. Full per-FOV output:
`case-study-KTR-72502946.csv` (324 rows). Run with `--previews` to also render the raw
fluorescence thumbnails below (streamed from GCS via `src.gcs_fov_multi.load_fov_image`, same
700px-downscale + caption convention as `scripts/render_fn_fp_previews.py`).

## Results

**Distribution across all 324 FOVs:**

| `robust_zscore` >= | FOVs | % of slide |
|---|---|---|
| 2 | 33 | 10.2% |
| 5 | 3 | 0.9% |
| 6 (v2's recommended threshold) | 2 | 0.6% |
| 10 | 1 | 0.3% |
| 20 | 1 | 0.3% |
| 50 | 1 | 0.3% |
| 100 | 1 | 0.3% |

Range: -1.461 to 163.527. A single FOV dominates everything above `z=10` -- see below.

**The two labeled FOVs (`spot_truth=yes` in the 76-row label set) and the two v2 boundary FOVs
(1, 324), side by side:**

| fov_id | role | n_spots_detected | ratio_to_median | robust_zscore |
|---|---|---|---|---|
| 1 | boundary | 49 | 0.925 | -0.150 |
| 54 | labeled (spot_truth=yes) | 25 | 0.472 | -1.049 |
| 198 | labeled (spot_truth=yes) | 4417 | 83.34 | **163.527** |
| 324 | boundary | 50 | 0.943 | -0.112 |

**Top 15 FOVs by `robust_zscore`:**

| fov_id | role | n_spots_detected | ratio_to_median | robust_zscore |
|---|---|---|---|---|
| 198 | labeled | 4417 | 83.34 | 163.527 |
| 308 | - | 262 | 4.943 | 7.832 |
| 95 | - | 194 | 3.66 | 5.284 |
| 77 | - | 186 | 3.509 | 4.984 |
| 294 | - | 186 | 3.509 | 4.984 |
| 94 | - | 184 | 3.472 | 4.909 |
| 276 | - | 181 | 3.415 | 4.796 |
| 167 | - | 171 | 3.226 | 4.422 |
| 136 | - | 169 | 3.189 | 4.347 |
| 149 | - | 163 | 3.075 | 4.122 |
| 131 | - | 155 | 2.925 | 3.822 |
| 113 | - | 152 | 2.868 | 3.71 |
| 227 | - | 151 | 2.849 | 3.672 |
| 312 | - | 151 | 2.849 | 3.672 |
| 240 | - | 142 | 2.679 | 3.335 |

## The three FOVs that matter, with images

Raw fluorescence, downscaled to 700px, captioned with this analysis's own numbers. Red caption =
`high_outlier` (`robust_zscore >= 2`), green = not.

### FOV 198 -- labeled `spot_truth=yes`, overwhelming true positive (z=163.5)

![KTR-72502946 fov 198](previews/KTR-72502946__fov198__preview.png)

A textbook overexposure halo: a single huge, bright, diffuse blue glow covering most of the
frame. 4417 crops vs. a slide median of 53. The visual confirms the mechanism this whole approach
assumes -- a bright halo saturating the blue channel generates thousands of spurious local
maxima for the spot detector to find.

### FOV 54 -- labeled `spot_truth=yes`, clean miss (z=-1.05)

![KTR-72502946 fov 54](previews/KTR-72502946__fov54__preview.png)

Also a confirmed halo, but a *far* fainter and dimmer one -- a soft, low-contrast glow in the
left/center of an otherwise dark frame, nothing like fov 198's saturating blaze. Only 25 crops,
below the slide's median of 53. **The image explains the miss:** this halo is dim enough that it
never pushes local-maxima counts above baseline, so a crop-count-based detector has nothing to
detect. This is the faint/diffuse failure mode v1's README names in the abstract, now visible
concretely.

### FOV 308 -- never labeled, flagged by the signal (z=7.83)

![KTR-72502946 fov 308](previews/KTR-72502946__fov308__preview.png)

**Not a halo at all.** There's no diffuse glow anywhere in this frame -- instead it's covered in
bright punctate specks, including several large saturated blobs and a distinctive diagonal
streak of puncta across the lower-left. This is debris/artifact contamination, and the elevated
crop count (262 vs. median 53) is the spot detector faithfully finding all of it.

## Discussion

**This one slide reproduces both the strength and the weakness of the crop-count signal at
once.** Its two labeled FOVs are both confirmed genuine overexposure-halo artifacts
(`spot_truth=yes`), but the crop-count signal treats them completely differently:

- **FOV 198 is an overwhelming true positive** -- 4417 crops vs. a slide median of 53
  (83x), `robust_zscore=163.5`. This is on the same scale as the largest positives found anywhere
  in v1's 76-row analysis (max there: `robust_zscore=387.6`, `KIT-62501087` fov 271).
- **FOV 54 is a clean miss** -- 25 crops, *below* the slide's own median (`ratio=0.47`,
  `robust_zscore=-1.05`). A human confirmed a genuine halo here; the crop-count signal shows
  nothing unusual at all, let alone an excess. The images above show why: fov 198's halo
  saturates the frame, fov 54's is a faint dim glow.

**Both boundary FOVs behave exactly as the "presumed blank" assumption predicts.** FOV 1
(`z=-0.15`) and FOV 324 (`z=-0.11`) both sit almost exactly on the slide median -- on this
particular slide, at least, the corner-tile-is-blank assumption holds up well.

**FOV 308, the one unlabeled FOV clearing the recommended threshold, turns out to be a false
positive of this approach -- and the image is what settles it.** Before pulling the image, this
FOV read as a promising lead ("nobody has looked at it; exactly the kind of case this approach
surfaces for review"). The image shows it's debris, not a halo: bright punctate specks and a
diagonal streak, with no diffuse glow anywhere. So under the *original* halo-focused framing this
is a straightforward false positive, and it's the `background`/`artifact` confound v1 already
identified as the dominant source of overlap between the groups -- a whole-frame crop count can't
distinguish "one big halo inflated my count" from "lots of little bright junk inflated my count."

Worth noting this cuts differently under v2's redefinition (positive = significant excess of
erroneous crops, *regardless* of cause): by that definition fov 308 is arguably a true positive,
since 262 debris-driven crops are genuinely erroneous crops the classifier has to process. Which
reading is right depends on what the downstream filter is for -- suppressing halo artifacts
specifically, or suppressing junk-inflated FOVs generally. This single FOV is a clean illustration
that those two goals are not the same thing, and v2's redefinition quietly commits to the
second one.

## Caveats

- One slide, two labeled FOVs -- not a basis for re-estimating recall on its own. It's useful as
  a concrete illustration of the aggregate finding, not a replacement for it.
- The fov 308 read above is from a single downscaled preview, not a full-resolution review or a
  second annotator -- confident enough to say "not a diffuse halo," but it hasn't been through
  the labeling process the 76-row set went through.
- The remaining 31 FOVs on this slide with `robust_zscore >= 2` (but below 6) were not inspected;
  given fov 308, some fraction of those are likely debris rather than halos too.

## Files

- `case-study-KTR-72502946.csv` -- full per-FOV output (324 rows)
- `previews/KTR-72502946__fov{54,198,308}__preview.png` -- the three thumbnails above
- `../../../../scripts/crop-outlier-approach/case_study_ktr72502946.py` -- script that produced
  both (`--previews` for the thumbnails)
