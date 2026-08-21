# Why rouleaux_fraction doesn't track manual overlap: a segmentation failure, not a threshold problem

Follow-up to `data/results/tanzania-073026/tanzania-comparison/README.md`, which found
`rouleaux_fraction` (from `scripts/ai-first/score_new_slide.py`) has no positive
relationship with manual overlap labels (marginal rho=-0.19, partial rho=-0.10).
Produced by `scripts/diagnose_rouleaux.py`.

The segmentation collapse below is one cause. A second, independent one -- the
per-FOV reference area against which "merged" is defined -- was identified later
and is recorded in "A second cause: the reference area is per-FOV" at the end of
this file.

## Method

Ran `score_new_slide.py`'s own `segment()`/`touching_pairs()` code (unmodified) on
4 FOVs manually tagged "heavy rouleaux", each paired with a "no rouleaux" FOV
matched on `coverage_fraction` (so density isn't a confound in the comparison).
Colored every surviving watershed instance by what the algorithm did with it:
blue = ordinary cell, red = flagged "inline" (counts toward `rouleaux_fraction`),
orange = oversized/merged blob (`area > MERGED_AREA_RATIO * reference_area`).

## Finding

![heavy rouleaux vs control](heavy-rouleaux-vs-control.png)

At the coverage levels where rouleaux/heavy-rouleaux tags actually occur on this
slide (coverage_fraction 0.55-0.75 — recall density and overlap are correlated,
rho=0.75, so overlap essentially only shows up in the denser FOVs), the
segmentation has already broken down: **15-20% of "cells" are one of a handful
of enormous merged blobs** covering most of the frame (orange dominates every
panel above), not individual cells. The "inline" pairs the algorithm counts as
rouleaux (red) are visibly thin cracks/slivers running between two of these
giant blobs — a watershed boundary artifact that trivially satisfies "exactly 2
neighbors positioned opposite each other," with no relationship to actual
cell-chain morphology.

For contrast, at low coverage (FOV 136, coverage=0.21):

![low coverage control](low-coverage-control.png)

Individual cells are cleanly separated (mostly blue), merged blobs are a small
minority (132/1988 = 6.6%, vs. 15-20% in the crowded pairs above), and the red
"inline" hits are visibly real adjacent-cell pairs, not slivers. The
segmentation works as intended here.

**Quantitative confirmation** (`diagnostic-summary.csv`): every heavy-rouleaux /
no-rouleaux pair, matched on coverage_fraction, has nearly identical
`rouleaux_fraction` — in 2 of 4 pairs the *no-rouleaux* FOV actually scores
higher:

| pair | heavy-rouleaux FOV | rouleaux_fraction | no-rouleaux FOV (matched coverage) | rouleaux_fraction |
|---|---|---|---|---|
| 0 | 205 | 0.119 | 61  | 0.105 |
| 1 | 206 | 0.117 | 242 | **0.127** |
| 2 | 210 | 0.131 | 236 | **0.139** |
| 3 | 215 | 0.118 | 248 | 0.088 |

## Answer to "would tuning `INLINE_COS_THRESHOLD` (or similar) help?"

No. The angle/degree geometry test never gets a fair look at real cell-chain
shapes here, because its input -- the watershed instance labels -- is already
wrong at these coverage levels: most of the true cell boundaries were never
recovered in the first place. Adjusting the -0.7 cosine cutoff, or
`MIN_DISTANCE`, or the merged/fragment area ratios, only reshuffles which
slivers and micro-fragments of the *already-collapsed* segmentation count as
"inline" -- it cannot recover the individual-cell boundaries that watershed
failed to find inside those giant orange blobs. This is a segmentation
capability problem, not a decision-boundary placement problem, and it is
concentrated exactly in the coverage range where overlap/rouleaux occurs on
this slide.

**Implication**: a working overlap/rouleaux measure on this slide needs
either (a) a segmentation step that doesn't collapse under this much crowding
(the current gradient-energy + distance-transform watershed is the bottleneck
-- a trained instance-segmentation model would be the standard fix), or (b) a
feature that doesn't depend on resolving individual cell instances at all --
which is exactly why the four-step pipeline's texture-based features (GLCM
contrast, unmasked edge density) showed a small positive partial correlation
with overlap in `tanzania-comparison/`: they measure local pixel statistics
directly and never need to correctly split touching cells apart.

## Caveats

- 4 pairs, one slide -- illustrative, not exhaustive. But the merged-blob
  fraction and the visual sliver pattern are consistent across all 4 pairs
  and starkly different from the low-coverage control, so this isn't cherry-picked.
- `n_merged` counts instances above `MERGED_AREA_RATIO`, which undercounts the
  true failure -- several of the orange regions above are single watershed
  labels spanning a large fraction of the entire frame, not "slightly
  oversized" cells.

---

# A second cause: the reference area is per-FOV

The finding above attributes `rouleaux_fraction`'s failure to watershed collapse. That
holds, but it is not the only mechanism. `score_new_slide.py:104` sets the yardstick for
"merged" from the FOV's own area distribution:

```python
reference_area = np.percentile(raw_props["area"], 75)
fragment_floor = FRAGMENT_AREA_RATIO * reference_area   # 0.35
merged_ceiling = MERGED_AREA_RATIO * reference_area     # 1.6
```

In an overlapped field the area distribution is itself inflated by the merging, so
`reference_area` rises with the thing it is used to detect, and `merged_ceiling` rises
with it. Two consequences:

- **The bias attenuates the signal where it matters most.** A higher ceiling flags *fewer*
  blobs as merged, and it is highest in the most overlapped fields. `n_merged` is not just
  an undercount (already noted in the caveats above) -- it is an undercount whose severity
  scales with overlap.
- **Values are not on a common scale across FOVs.** Each FOV is measured against its own
  yardstick, so `n_merged` and `rouleaux_fraction` are not comparable between FOVs even
  when the segmentation works.

This matters beyond `rouleaux_fraction`: any future feature keyed to `merged_ceiling`
inherits the same blindness, so the reference needs fixing before such a feature is worth
building.

## What the existing pairs show

Recomputed from `diagnostic-summary.csv` alone (frame = 2800x2800 = 7,840,000 px; no new
segmentation run). Both quantities below are normalized by **foreground**, not by frame
area, which is what makes them comparable at matched coverage:

| pair | FOV | tag | coverage | mean fg px / cell | n_merged / n_cells | rouleaux_fraction |
|---|---|---|---|---|---|---|
| 0 | 205 | heavy rouleaux | 0.555 | 915 | 0.160 | 0.119 |
| 0 | 61 | no rouleaux | 0.555 | 1241 | 0.192 | 0.105 |
| 1 | 206 | heavy rouleaux | 0.620 | 1189 | 0.172 | 0.117 |
| 1 | 242 | no rouleaux | 0.619 | 1095 | 0.153 | 0.127 |
| 2 | 210 | heavy rouleaux | 0.663 | 1786 | 0.192 | 0.131 |
| 2 | 236 | no rouleaux | 0.664 | 1175 | 0.125 | 0.139 |
| 3 | 215 | heavy rouleaux | 0.750 | 2935 | 0.176 | 0.118 |
| 3 | 248 | no rouleaux | 0.724 | 1887 | 0.134 | 0.088 |

Heavy-rouleaux / control ratio per pair:

| pair | mean fg px / cell | n_merged / n_cells | rouleaux_fraction |
|---|---|---|---|
| 0 | 0.74 | 0.83 | 1.14 |
| 1 | 1.09 | 1.13 | 0.92 |
| 2 | 1.52 | 1.54 | 0.94 |
| 3 | 1.56 | 1.31 | 1.35 |

**Both foreground-normalized quantities rank the heavy-rouleaux FOV above its
coverage-matched control in 3 of 4 pairs; `rouleaux_fraction` manages 2 of 4, which is
chance.** Pair 0 is the exception on both new measures -- and is one of the two pairs
`rouleaux_fraction` happens to get right, so the measures disagree on the same pair rather
than one strictly dominating.

## What this does not show

- **Not a validation.** 4 pairs on one slide, and these are proxies assembled from summary
  columns that already exist, not the fixed-reference feature itself. 3/4 on n=4 is one
  coin flip away from 2/4.
- **The drift is not a clean trend.** Isolating the four `no rouleaux` controls, so coverage
  varies and overlap does not, apparent mean fg px/cell runs 1241 (cov 0.555) -> 1095
  (0.619) -> 1175 (0.664) -> 1887 (0.724): roughly flat, then rising sharply at the top.
  So the contamination is real but concentrated at high coverage rather than proportional
  to it, and a single ratio across the whole span would overstate it.
- No refit or params change was made.

## Fix direction

Estimate the single-cell reference so that it is *not* a function of the overlap level:
from the modal area, or from a low percentile among blobs that are convex and unmerged
(isolated singles being the cleanest single-cell exemplars available), and hold it **fixed
per slide** rather than per FOV, so an overlapped FOV is measured against the same yardstick
as a clean one. Emit it alongside the original so the change stays attributable.
