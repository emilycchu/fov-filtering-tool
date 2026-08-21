# combined-v3: cohort completion and the frozen train/test split

Status: **splits frozen, annotation not started.**

v3 retrains the density/overlap model from scratch with slide-grouped splits, keeping the v2.2
machinery unchanged (feature extraction -> ridge -> PAVA) so the comparison is controlled. This
directory holds the two steps that had to happen before any FOV could be annotated: completing the
cohort, and choosing the slides.

## Headline: the Tanzania cohort is 497 slides, and the missing half is denser

Every cohort statistic this project had ever quoted came from the 271 slides in the annotatability
workbook. That workbook lists positives only. `gs://malaria-annotation-web/catalog.json` (schema
v2) records **497 Tanzania samples: 271 positive and 226 PCR-negative**, all with images in
`gs://tanzania_02032026`. The 226 negatives had never been scored.

They are **materially denser than the positives**:

| group | n | min | p25 | median | p75 | max | mean |
|---|---|---|---|---|---|---|---|
| positive | 271 | 0.015 | 0.161 | 0.247 | 0.370 | 0.707 | 0.268 |
| negative | 226 | 0.012 | 0.207 | **0.320** | 0.435 | 0.715 | 0.328 |
| all 497 | 497 | 0.012 | 0.177 | 0.286 | 0.399 | 0.715 | 0.295 |

Mann-Whitney z = -4.56, p = 5.2e-06; rank-biserial -0.237; two-sample KS 0.214.

**The gap is not the site confound.** Negatives are NKR-heavy and positives KIT-heavy, and site is
the strongest structural variable in the cohort -- but the negatives are denser *within every
site*: KIT 0.318 vs 0.263, KTR 0.315 vs 0.291, NKR 0.379 vs 0.282, RUB 0.215 vs 0.187. A plausible
mechanism is malarial anaemia thinning the positives' smears, which would make this a real
biological effect rather than a preparation artifact. Not tested here; recorded as the obvious
hypothesis.

Two published claims are now known to be positives-only artifacts:

- *"61% of slides land in the bottom density bucket."* Across 497 it is 256/497 (52%), and the
  negatives are 91/226 (40%) sparser against the positives' 165/271 (61%).
- *"None in the top bucket."* One negative slide (`RUB-72501745`, density 0.715) is `very dense`,
  and negatives supply 11 of the 17 `dense` slides.

`KTR-72502946` -- one of the two v2.2 calibration slides, carrying 324 of the 661 labelled FOVs --
**is PCR-negative** (and microscopy- and expert-microscopy-negative). `KTR-72502948` is positive on
all three. So the existing label investment already spans both classes. This also reframes an
existing result: the two leave-one-slide-out folds that "disagree sharply" (held-out density rho
+0.831 vs +0.454) are one negative and one positive slide, which may be the explanation rather than
slide-level noise.

## The frozen split

`slide-splits.csv`. Train is selected for **range coverage** (one slide per cohort density
quintile, maximising within-slide spread) and test for **distributional match** (minimising KS
against the cohort's pooled per-FOV CDF). Those are different objectives; conflating them is what
centred the v2.2 fit on the 77th-93rd percentile of its own cohort.

**Train/val -- 7 slide groups.** All 4 sites, 4 boxes, 4 negative / 3 positive.

| slide | truth | density | within-slide std | bucket | box | Q | source |
|---|---|---|---|---|---|---|---|
| KTR-72502904 | negative | 0.157 | 0.137 | sparser | Box5 | Q0 | to annotate |
| KIT-62500909 | negative | 0.190 | 0.185 | sparser | Box1 | Q1 | to annotate |
| RUB-72501818 | positive | 0.260 | 0.238 | sparser | Box3 | Q2 | to annotate |
| NKR-72502156 | negative | 0.326 | 0.190 | monolayer | Box4 | Q3 | to annotate |
| KTR-72502948 | positive | 0.372 | 0.145 | monolayer | Box5 | Q3 | **already annotated (324)** |
| KIT-62500670 | positive | 0.491 | 0.277 | slightly dense | Box1 | Q4 | to annotate |
| KTR-72502946 | negative | 0.492 | 0.146 | slightly dense | Box5 | Q4 | **already annotated (324)** |

**Test -- 3 slides, scored once after the fit is frozen.** 3 sites, 3 boxes, 2 positive / 1
negative (the cohort is 55:45).

| slide | truth | density | bucket | box |
|---|---|---|---|---|
| KTR-72502749 | positive | 0.165 | sparser | Box5 |
| RUB-72501749 | positive | 0.325 | monolayer | Box3 |
| NKR-72502143 | negative | 0.399 | monolayer | Box4 |

Pooled test median 0.287 against a cohort median of 0.279, **KS = 0.0173** over 2,425,365 candidate
slates. Liberia stays held out entirely -- which also means the 9 Liberia FOVs currently inside the
v2.2 calibration pool must come out of the v3 training set, along with the 4 single-FOV Tanzania
slides (one of which, `KIT-62501048`, is a catalog `test` slide).

Constraint respected in one direction only: **no catalog `test` slide enters v3 train.** The
workbook's 23/22 split is a parasite-annotation split -- it excludes all 119 hazard slides and
contains only ANNOTATABLE rank-3/4 slides -- so it is unsuitable as a crowding split, but keeping
its test slides out of our training set costs nothing.

## Resolved: the density axis gains two rungs, and the gate stops overriding labels

Four of the eight new slides are heavily gated by v2.2's empty-field rule -- `RUB-72501818`
95/324 (29.3%), `KTR-72502904` 74/324, `KIT-62500670` 64/324, `KIT-62500909` 56/324, and
`RUB-72501749` 28/324 in test. 12.2% of the eight slides' FOVs overall, so roughly 80 of the 648
sampled FOVs land here. That is far too many to leave to a rule nobody has checked.

**The gate conflates two visually distinct fields.** Both of these are gated, and both are forced
to `sparser` + `no rouleaux`:

| FOV | coverage | what it actually is |
|---|---|---|
| `KIT-62500670` fov198 | 0.9995 | genuinely blank -- flat grey, a few debris specks, nothing to judge |
| `RUB-72501818` fov107 | 0.9656 | hundreds of countable, well-separated cells; perfectly judgeable |

So labelling *whatever the gate flags* as empty would import that conflation straight into the
ground truth, and -- because you would only ever inspect what the gate selected -- its recall
would stay unmeasurable. That is the same circularity `check_empty_field_gate.py` already has.

**Coverage cannot substitute for it.** On a flat field Otsu's threshold is arbitrary, so a blank
FOV lands at either end of the coverage range depending on which side of the sensor noise the
threshold falls. Across the eight slides the gated FOVs' coverage is bimodal with *nothing*
between 0.10 and 0.20: 59 below 0.02, 71 in 0.02-0.05, 102 in 0.05-0.10, then 85 above 0.95.
Whatever separates the bottom of the scale has to be textural -- which is the one part of the old
gate's design that was right, and which the new bottom rungs inherit.

**Decision.** The density axis becomes **7 levels**:

    no cells < few cells < sparser < monolayer < slightly dense < dense < very dense

with the operational test tied to what the tool is for -- *can you judge how packed this field
is?* Nothing there at all -> `no cells`. Countable one by one -> `few cells`. Thin but genuine
monolayer -> `sparser`.

- The gate no longer overrides labels. Its flag is still recorded per FOV (already computed, so
  free) and is re-evaluated *afterwards* against the new labels: keep it only if it beats the
  model's own bottom-rung prediction. The labels are the irreversible part; the gate is twenty
  lines.
- `no cells` is excluded from the ridge fit and the PAVA medians (a blank field's composite
  answers a different question), but still scored at evaluation as the gate's replacement
  correctness check. `few cells` is fitted normally.
- An overlap tag on `no cells` raises rather than defaulting -- a field with no cells has no
  packing to describe. `few cells` *can* carry one: cells that are few can still touch, which is
  exactly the density-independent overlap signal v3 is trying to learn.
- **The 646 existing annotations need no relabelling.** Levels are looked up by name, never by
  index, so an old `sparser` stays `sparser` and simply takes ordinal 2 instead of 0.

Implemented in `scripts/combined/combined-v3/_v3_common.py`, which re-exports the unchanged v2.2
machinery so there is one import site for v3 code.

## Blind re-labels (independent of the split -- can start now)

`blind-relabels.zip` (203 MB): 50 FOVs drawn from the 646 already annotated, renamed `fov-01.png`
.. `fov-50.png` under a seeded shuffle with both slides interleaved, plus an `annotations.csv`
template and the tag vocabulary. The key is written **outside** the zip as
`blind-relabels-KEY.csv`.

A second pass through the annotation tool would not be blind -- it shows FOVs in slide order with
the sample id visible and the prior labels one click away, so it would measure recall of the first
pass rather than independent judgement.

Sampling is proportional to the real label distribution with a floor of 2 per level, so the pooled
self-agreement rate stays a near-unbiased ceiling while every bucket appears: density 30 monolayer
/ 7 slightly dense / 5 sparser / 4 dense / 4 very dense; overlap 31 none / 7 slight / 5 some / 3
rouleaux / 4 heavy. Each key row carries a `sampling_weight` so a weighted mean recovers the
unbiased estimate. The draw is 33 FOVs from KTR-72502946 and 17 from KTR-72502948 -- proportional
allocation landing unevenly, which tilts the ceiling toward the denser (negative) slide.

**The set ships with the 7-level vocabulary**, which means it measures vocabulary drift *plus*
annotator noise. Both are recoverable: score self-agreement after
`_v3_common.collapse_to_v2_density` maps the two new rungs back to `sparser` -- a clean noise
measurement, and nearly lossless since only 3 of the 661 existing FOVs fire the gate -- then
report how the old `sparser` FOVs redistribute across `no cells` / `few cells` / `sparser` as a
separate result. That redistribution is itself the first evidence on where the new boundaries
actually fall.

**Done, 2026-08-21.** Full write-up, with per-level ceilings and kappas, in
`scripts/combined/combined-v3/annotator-agreement.md`. The filled-in worksheet is
`../../labels/blind-relabels-082126/blind-relabels-annotations.txt` -- annotation input, so it
sits with the other label sets rather than here. Collapsed to the 5-level vocabulary,
self-agreement is 41/50 on density (82%, 82% weighted) and 41/47 on overlap (87%, 92%
weighted). **v2.2's 55% density exact-match is therefore well short of the ceiling, and further
feature work is justified.** The overlap denominator is 47 because the annotator had lost track
of the `rouleaux` rung by pass 2, so its 3 FOVs could not have agreed -- an artifact of the
worksheet, which needs the vocabulary repeated inline before the 648-FOV worklist reuses it.
The one real cluster is `slightly dense`, which lost 6 of its 7 FOVs and looks like a boundary
rather than a regime. Neither new bottom rung was used, so the redistribution question is still
open -- see that set's README.

## The run

226 negative slides, **73,213 DPC FOVs, 0 errors**, on the `crowding-tz-081426` n2-standard-32 Spot
VM in us-central1-a, **39 min 55 s at 30.6 FOV/s**, 8 processes x 8 threads (the bench re-chose the
same config as the 271-slide run). All 8 shards reported `total_errors=0`. Scored with the deployed `v2.2-optimized` params, gate check asserted first so both
halves of the cohort are scored by an identical fit.

Local streaming was measured at **0.25 FOV/s** -- 73,213 FOVs is ~285 GB, i.e. ~81 hours on a home
connection -- so the VM was necessary rather than merely faster. The VM was stopped manually
afterwards; its service account cannot self-stop.

Per-FOV CSVs land in `../tanzania-complete-081426/fov/crowding/`, the same directory as the
positives, so downstream reads one 497-slide result set. The positives' `slides.csv` /
`slide-index.json` were **not** modified -- `build_negatives_index.py` writes a parallel
`slides-negatives.csv` / `slide-index-negatives.json`, so the completed 271-slide run's audit trail
and `verify_regression.py` stay valid.

## Files

| file | what |
|---|---|
| `slide-summary-497.csv` | one row per slide, all 497. Same schema as the positives' summary minus the fluorescence columns (those passes are positives-only) |
| `cohort-497-comparison.md` | the positive-vs-negative comparison and the per-site breakdown |
| `slide-splits.csv` | the frozen roster: `slide_id, role, truth, site, box, catalog_split, n_fovs, density_mean, quintile, source` |
| `blind-relabels.zip` | 50 anonymised FOVs + annotation template. **Gitignored** -- 203 MB, past GitHub's per-file limit; the seed reproduces it |
| `blind-relabels-KEY.csv` | the un-blinding key. Do not open before annotating |
| `../../labels/blind-relabels-082126/` | the completed second pass, plus what it says about the ceiling |

## Reproducing

```bash
# 1. roster + FOV verification for the 226 negatives (~40 s, 231 GCS listings)
python scripts/tanzania-complete-081426/build_negatives_index.py

# 2. the pass itself -- on a VM; see scripts/tanzania-complete-081426/vm-startup-negatives.sh
python scripts/tanzania-complete-081426/run_crowding_pass.py \
    --slides-csv data/results/tanzania-complete-081426/slides-negatives.csv \
    --index      data/results/tanzania-complete-081426/slide-index-negatives.json \
    --procs 8 --threads 8 --mirror-gcs

# 3. unified summary + the positive/negative comparison (local, seconds)
python scripts/combined/combined-v3/summarize_cohort_497.py

# 4. the split (local, ~1 min; exhaustive over 2.4M test slates)
python scripts/combined/combined-v3/select_splits.py

# 5. the blind re-label package (downloads 33 FOVs from GCS)
python scripts/combined/combined-v3/build_blind_relabels.py
```

`summarize_cohort_497.py` imports `summarize_crowding` from the existing aggregator rather than
reimplementing it, so the gated-FOV convention and the bucket-of-mean rule are identical to the
published positives numbers by construction. Verified: it reproduces `slide-summary.csv`'s
`density_mean` bit-for-bit on spot-checked slides.

## Next

1. ~~Settle the empty-field vocabulary~~ -- done, see above: 7-level density ordinal,
   `_v3_common.py`. Worklist is no longer blocked.
2. Generate the 648-FOV worklist (8 slides x 81, stratified within each slide's own score range)
   from `slide-splits.csv`.
3. Annotate. ~~The 50 blind re-labels can be done in parallel, and ideally first~~ -- done,
   2026-08-21; the 648 remain.
4. Build `combined-v3/` proper: `extract_features_v3.py` (adds `hole_density`, measured at partial
   rho 0.547 with overlap and -0.086 with density, ~0.029 s/FOV at 2x mask downsampling),
   `calibrate_v3.py` (nested leave-one-slide-out, slide-balanced sample weights), `evaluate_v3.py`.
