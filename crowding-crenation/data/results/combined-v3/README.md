# combined-v3: cohort completion and the frozen train/test split

Status: **training labels ingested, v3 fitted.** The 4 test slides remain unannotated and unscored.

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

**Test -- 4 slides, scored once after the fit is frozen.** All 4 sites, 4 boxes, 2 positive / 2
negative (the cohort is 55:45). **Refrozen 2026-08-24** -- see below.

| slide | truth | density | bucket | site | box |
|---|---|---|---|---|---|
| KIT-62500735 | positive | 0.176 | sparser | KIT | Box1 |
| RUB-72501769 | negative | 0.276 | sparser | RUB | Box3 |
| NKR-72502165 | positive | 0.278 | sparser | NKR | Box4 |
| KTR-72502961 | negative | 0.455 | slightly dense | KTR | Box5 |

Pooled test median 0.286 against a cohort median of 0.279, **KS = 0.0138** over 51,513,782
candidate slates. Liberia stays held out entirely -- which also means the 9 Liberia FOVs currently
inside the v2.2 calibration pool must come out of the v3 training set, along with the 4 single-FOV
Tanzania slides (one of which, `KIT-62501048`, is a catalog `test` slide).

### Why the test slate was refrozen at 4 slides (2026-08-24)

The original slate was 3 slides under a `>=3 sites` rule, which meant it could not span all four
sites -- and the site it dropped was **KIT**, the worst one to lose: KIT is the largest site among
positives (104 of 271) and v3 *trains* on two KIT slides, so the model saw KIT and was never
tested on it.

Going to 4 slides, one per site, is better on every criterion at once:

| slate | slides | sites | boxes | truth | KS |
|---|---|---|---|---|---|
| original (3 slides) | 3 | 3, no KIT | 3 | 2p+1n | 0.0173 |
| 3 slides, KIT forced in | 3 | 3, no KTR | 3 | 2p+1n | 0.0195 |
| **refrozen (4 slides)** | 4 | **all 4** | **4** | 2p+2n | **0.0138** |
| 4 slides, 3p+1n | 4 | all 4 | 4 | 3p+1n | 0.0152 |
| 4 slides, 1p+3n | 4 | all 4 | 4 | 1p+3n | 0.0157 |

Requiring 4 distinct boxes is free -- the unconstrained search returns the identical slate. 2p+2n
is both the best-KS truth split and the closest to the cohort's 55:45.

**One caveat, stated rather than implied:** part of the KS gain is simply that 4 slides have more
freedom to match the cohort CDF than 3. Whether the all-sites constraint is *itself* free at n=4
was not established -- the unconstrained 4-slide search is an 843M-combination space that does not
factorise the way the one-per-site case does. Read 0.0138 as "4 slides covering all sites beats 3
slides covering three," not as "the site constraint costs nothing."

Two mechanics worth recording:

- **Train slides are pinned now.** `select_splits.py --pin-train` (on by default) reuses the train
  roster instead of re-deriving it. The five new train slides are annotated, so they are data
  rather than a search result; re-searching them after the test pool changed could silently pick
  different slides and orphan 412 labelled FOVs.
- **The all-sites search is factorised.** One slide per site over 4 sites is 51.5M slates x a
  201-point grid, which brute force cannot do. But KS depends on the slate only through the pooled
  histogram and the pooled FOV count, both plain sums, so the search splits into (site0, site1)
  pairs x (site2, site3) pairs -- ~4k iterations over a ~7k x 201 array. `select_test_all_sites`.

Because the test slides changed, `cohort-ranges.json` was regenerated (it excludes test slides) and
the fit re-run. The ranges now cover **493 slides / 159,716 FOVs**. Metrics barely moved: overlap
QWK 0.707 -> 0.708, everything else unchanged to the reported precision.

Constraint respected in one direction only: **no catalog `test` slide enters v3 train.** The
workbook's 23/22 split is a parasite-annotation split -- it excludes all 119 hazard slides and
contains only ANNOTATABLE rank-3/4 slides -- so it is unsuitable as a crowding split, but keeping
its test slides out of our training set costs nothing.

## Resolved: the density axis gains two rungs, and the gate stops overriding labels

Four of the eight new slides are heavily gated by v2.2's empty-field rule -- `RUB-72501818`
95/324 (29.3%), `KTR-72502904` 74/324, `KIT-62500670` 64/324, `KIT-62500909` 56/324, and
`RUB-72501749` 28/324 in test. 12.2% of the eight slides' FOVs overall, so roughly 80 of the 648
sampled FOVs land here. That is far too many to leave to a rule nobody has checked.

> These gating rates were measured against the **original** 8-slide split, before the test slate
> was refrozen at 4 slides on 2026-08-24. `RUB-72501749` is no longer a test slide. The four
> heavily-gated slides are all train slides and all still in the split, so the decision this
> section records is unaffected; the numbers are left as measured rather than restated.

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

## The training fit (2026-08-24)

`density_overlap_v3_params.json`, `calibration-report-v3.md`, `loso-folds-v3.csv`,
`oof-predictions-v3.csv`. **1060 labelled FOVs across 7 slide groups**, up from v2.2's 661
across 2, spanning all 4 sites and 4 boxes. The 4 test slides remain unannotated and unscored.

### The realised worklist

The plan was every 4th FOV: `{1} u {4,8,...,324}` = **82 per slide**, not the 81 estimated above.
Five slides were annotated (the 4 test slides stay untouched), 435 rows written, **412 usable**.

| slide | rows | usable | off-grid | grid FOVs absent |
|---|---|---|---|---|
| KIT-62500670 | 97 | 83 | 23 | 8 |
| KIT-62500909 | 83 | 82 | 4 | 3 |
| KTR-72502904 | 84 | 82 | 2 | 0 |
| NKR-72502156 | 89 | 83 | 8 | 1 |
| RUB-72501818 | 82 | 82 | 0 | 0 |

Three deviations from the grid, all recorded rather than smoothed over:

- **All 23 `Overexposed`-only rows are off-grid.** On hitting an overexposed field the annotator
  recorded its neighbours; those rows carry no density or overlap level, so they are excluded from
  training and written to `excluded-quality-only.csv` as a ready-made cohort for the deferred
  overexposure study.
- **14 off-grid rows carry a real label**, substituting for 12 grid FOVs that were absent
  (`KIT-62500670` fov53 for fov52, `NKR-72502156` fov69 for fov72, and so on). Net +2 are
  deliberate extras, including `NKR-72502156` fov213, added because the 1-in-4 grid under-sampled
  that slide's rouleaux.
- **`NKR-72502156` fov284 was corrected in the CSV**, from `"Slight Rouleaux"` to
  `"Monolayer, Slight Rouleaux"`. It was the only row in 435 with an overlap tag and no density
  tag, and the parser's raise on a missing density tag is the only thing that caught it -- so it
  was fixed in the data, not by giving the parser a default.

### What the new labels contain that the old ones do not

The 7-level vocabulary is exercised for the first time: **13 `no cells`, 24 `few cells`**, neither
of which appears in the legacy 648. Also new: **67 double-density rows** (`Sparser`+`Monolayer` and
`Few cells`+`Sparser`) and **4 double-overlap rows** (`KIT-62500670` fovs 28/64/152/188). A double
tag is read as *the more severe level, at its low end*. The legacy 648 contain zero doubles, so
that rule cannot perturb the existing pool -- `verify_v3_labels.py` asserts it.

Pooled label distribution (1060):

| axis | levels |
|---|---|
| density | `no cells` 13, `few cells` 24, `sparser` 188, `monolayer` 565, `slightly dense` 141, `dense` 79, `very dense` 50 |
| overlap | `no rouleaux` 743, `slight rouleaux` 129, `some rouleaux` 66, `rouleaux` 56, `heavy rouleaux` 66 |

Three parser defects were fixed on the way in, all in `_v3_common.py`: the tag lookup was
**case-sensitive** (the files write `Few cells`, the table said `Few Cells`), which also explains
why `blind-relabels-annotations.txt` parsed 0/50; and resolution was **last-tag-wins per axis**,
which silently returned `sparser` for `"Sparser, Few cells"` and the *milder* overlap for
`"Dense, Rouleaux, Some Rouleaux"`. Case-folding alone repairs the blind re-labels to 50/50 and
reproduces the published 82% / 87% / 74% ceiling exactly, which is now a checked number rather than
a hand-computed one.

### Results

Nested leave-one-slide-out, 7 outer folds x 6 inner. Shipped config: `alpha=10`,
`sample_weights=slide`, `sub_delta=0`, `ranges_scope=fold`, PAVA skips `no cells`.

| metric | density | vs baseline | overlap | vs baseline |
|---|---|---|---|---|
| exact (pooled OOF) | **62.3%** | 53.3% | **58.4%** | 70.1% |
| exact (slide-weighted) | 59.9% | 46.4% | 69.1% | 74.7% |
| off-by-one | 94.6% | -- | 91.7% | -- |
| balanced accuracy | 42.4% | 14.3% | 40.5% | 20.0% |
| macro F1 | 0.394 | -- | 0.371 | -- |
| QWK | 0.786 | -- | 0.708 | -- |
| annotator ceiling | 82% | -- | 87% | -- |
| folds scored | 7/7 | -- | 5/7 | -- |

**v3 scores lower than v2.2's 69.4% / 67.6% and that is the measurement, not a regression.**
v2.2's numbers are FOV-stratified 5-fold over 2 slides of one box with training-set-derived
ranges; v3's are leave-one-slide-out over 7 slides with cohort-derived ranges. Beating v2.2's
exact-match was explicitly not an acceptance criterion.

**The density axis passes every gate.** It beats its majority baseline by 9.0 points unweighted
and 13.5 slide-weighted, QWK 0.786, and **PAVA merged nothing** -- `no cells`, `few cells` and
`sparser` each kept a distinct cut point, so the 7-level extension held rather than collapsing.
That is the headline result: v2.2's empty-field pre-filter is now a prediction.

**The overlap axis fails the exact-match gate.** 58.4% against a 70.1% majority baseline, and
69.1% against 74.7% slide-weighted. It clears every other gate -- balanced accuracy 40.5% against
a majority classifier's 20.0%, QWK 0.708, 8 features kept in all 7 folds -- which is the signature
of a model that spreads predictions across the ordinal where the baseline collapses to the single
class holding 70% of the pool. For a QC tool that trade is arguably the right one, but it is a gate
failure as written and is reported as one rather than tuned away. Two of the 7 folds hold a test
set that is 100% `no rouleaux`, so only 5 are scored, and PAVA merged
`some rouleaux`+`rouleaux` in the full fit.

### Normalization is fold-local

The p2/p98 ranges come from 159,716 cohort FOVs rather than the 1060 labelled rows, because
v2.2's floors were set by 2 slides at the 77th-93rd cohort percentile and clipped 10-13% of the
cohort's sparse end to exactly 0 -- precisely the population the new bottom rungs exist to
separate (v2.2's `coverage` floor is 0.0735 against a cohort p2 of 0.0317).

**The model still trains on ~1000 rows.** The cohort supplies 18 numbers -- 9 features x p2/p98 --
and nothing else; those FOVs have no labels. But since the 493 range slides include the 7
training slides, each held-out slide originally contributed to its own feature scaling. That is
now closed: `--ranges-scope fold` (the default) derives each fold's ranges from the cohort minus
the slides held out at that level, outer and inner. The 4 test slides were excluded from the
start, so there was never any test leakage.

Measured, the residue was negligible but real:

| held-out slide | max span shift | worst feature |
|---|---|---|
| RUB-72501818 | 1.13% | `otsu_separability` |
| KIT-62500909 | 0.66% | `tile_glcm_patchiness` |
| KIT-62500670 | 0.40% | `lbp_entropy` |
| KTR-72502948 | 0.22% | `tile_glcm_patchiness` |
| KTR-72502904 | 0.15% | `otsu_separability` |
| KTR-72502946 | 0.12% | `coverage` |
| NKR-72502156 | 0.11% | `glcm_contrast` |

Any one slide is 0.203% of the 159,716 FOVs. End to end, closing the leak changed **1 of 1060**
density predictions and **2 of 1060** overlap predictions; max raw-score shift 2.0e-3, density QWK
0.784 -> 0.786, exact-match unchanged to four decimals. `--ranges-scope fixed` reproduces the un-corrected arm.

This does not reintroduce the scale-mixing that made threshold cross-fitting fail. There,
thresholds derived from *pooled* out-of-fold scores were applied to a single-scale score; here
each fold's thresholds come from its own fold's scale and only the ranges move between folds, by
~1%. Fold-local costs ~0.5 s: the 493 slides' columns are read once (~12 MB) and each of the 49
range sets is then a percentile call, rather than 49 re-reads at ~8 s each.

### Two A/Bs, both decided by the numbers

- **Slide-balanced weights win decisively.** Unweighted, density OOF exact falls to 48.4% --
  *below* its own baseline -- and overlap to 53.5%. Weighting is what makes the fit work, which is
  the direct measurement of defect (2): pooled naively, the two KTR/Box5 slides supply 61% of the
  rows. The cost is that weighting *tightens* the axis confound (composite rho 0.987 weighted
  against 0.937 unweighted), so the two effects trade against each other.
- **The double-label nudge carries no extractable signal at n=67.** `sub_delta=0` beats 0.33 by
  0.3 points on density (3 FOVs of 1060) and 0.1 on overlap (1 FOV). Under the rule set in
  advance -- ship 0 if 0.33 wins by less than one FOV's worth -- **`delta=0` ships**, and the
  honest conclusion is that 67 hedged rows are too few to learn from.

### Axis separation is unchanged, by construction

| basis | composite rho | manual label rho |
|---|---|---|
| v2.2 published | 0.972 | 0.823 |
| v3, full fit | 0.987 | 0.744 |
| v3, out-of-fold | 0.986 | 0.744 |

The gap did not close; it widened slightly. **This is the expected result, not a failed attempt.**
v3 changed the splits, the vocabulary and the weighting, and carries no mechanism aimed at the
confound: the two the design doc proposed -- the `hole_density` feature and restoring the
partial-rho gate on the overlap axis -- are both deferred to v3.1 (see "Next"). The manual label
rho itself moved from 0.823 to 0.744 because the vocabulary and the pool both changed, so those
two rows are not directly comparable either.

### Verification

| script | result |
|---|---|
| `verify_v3_labels.py --assert-counts` | green -- 648/648 parser equivalence, 648 rows joined to `merged-labels-v2.2.csv`, every count, blind re-labels 50/50 reproducing 41/50, 41/47, 35/47 |
| `verify_v3_fit.py` | green -- 69 numeric fields reproduce at max relative delta 0.000e+00, plus the structural invariants |
| `verify_regression.py` | green -- deployed v2.2 scorer unmoved, max delta 0.000e+00 |
| `calibrate_v2.2-optimized.py` re-run | reproduces the committed params bit-identically after the two additive `calibrate_v2.py` edits |
| `check_empty_field_gate.py --params <v3>` | "absent or disabled -- nothing to check", exit 0. **That is the correct outcome** and is recorded rather than skipped: v3 replaces the gate with the bottom rung |

Two facts found during the ingest, both pre-existing and unrelated to v3:

- **`merged-labels.csv` is stale.** It holds 337 rows (13 initial-dataset plus `KTR-72502948`
  only), written before `KTR-72502946` was annotated, and `_v2_common.MERGED_LABELS_CSV` still
  points at it. The live 661-row file is `merged-labels-v2.2.csv`, which is what
  `merge_labels_v2.py` reproduces byte-for-byte and what v3 joins against. The v2.2 chain passes
  `--labels-csv` explicitly, so nothing is broken today, but the default is a footgun.
- **Positives-only normalization barely mattered.** `cohort_ranges.py --slides positive` gives 269
  slides / 87,151 FOVs whose p2/p98 span differs from the all-slides span by at most **6.7%** on
  any feature. The negatives being denser at the slide level does not translate into a materially
  different pooled per-FOV percentile. `all` remains the default because it is the honest
  population, not because the alternative measured badly.

## Files

| file | what |
|---|---|
| `slide-summary-497.csv` | one row per slide, all 497. Same schema as the positives' summary minus the fluorescence columns (those passes are positives-only) |
| `cohort-497-comparison.md` | the positive-vs-negative comparison and the per-site breakdown |
| `slide-splits.csv` | the frozen roster: `slide_id, role, truth, site, box, catalog_split, n_fovs, density_mean, quintile, source` |
| `blind-relabels.zip` | 50 anonymised FOVs + annotation template. **Gitignored** -- 203 MB, past GitHub's per-file limit; the seed reproduces it |
| `blind-relabels-KEY.csv` | the un-blinding key. Do not open before annotating |
| `../../labels/blind-relabels-082126/` | the completed second pass, plus what it says about the ceiling |
| `merged-labels-v3.csv` | 1060 labelled FOVs: v2's 8 columns then `slide, fov_id, split, density_sub, overlap_sub, quality_tags` |
| `excluded-quality-only.csv` | the 23 `Overexposed`-only rows, the deferred overexposure study's cohort |
| `cohort-ranges.json` | p2/p98 for all 9 features over 493 non-test slides / 159,716 FOVs, with the knobs they were scored at |
| `features-v3.csv` | `merged-labels-v3.csv` joined to the cohort per-FOV features, plus v2.2's own predictions under a `cohort_` prefix |
| `density_overlap_v3_params.json` | the shipped v3 fit. `empty_field_override.enabled: false` -- the gate is now the bottom rung |
| `calibration-report-v3.md` | the full write-up: train/val/test roster with weights, correlations, per-axis fits, v3-vs-v2.2 weights and head-to-head, metrics, confusions, gate replacement, ablation |
| `loso-folds-v3.csv` | one row per axis x held-out slide |
| `oof-predictions-v3.csv` | per-FOV out-of-fold predictions, so plots read a file instead of re-running the fit |

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

# --- the training fit (all local except step 8's 20-FOV audit sample) ---

# 6. merge the labels (seconds)
python scripts/combined/combined-v3/merge_labels_v3.py
python scripts/combined/combined-v3/verify_v3_labels.py --assert-counts

# 7. cohort normalization ranges (~8 s over 493 slide CSVs)
python scripts/combined/combined-v3/cohort_ranges.py

# 8. join labels to the already-computed cohort features -- NOT a re-extraction
python scripts/combined/combined-v3/build_features_v3.py

# 9. fit (~11 s: 2 axes x 7 outer x 6 inner, plus the delta ablation)
python scripts/combined/combined-v3/calibrate_v3.py --sub-delta 0
python scripts/combined/combined-v3/verify_v3_fit.py

# 10. the gate must now report itself disabled, and exit 0
python scripts/combined/check_empty_field_gate.py     --params data/results/combined-v3/density_overlap_v3_params.json
```

`summarize_cohort_497.py` imports `summarize_crowding` from the existing aggregator rather than
reimplementing it, so the gated-FOV convention and the bucket-of-mean rule are identical to the
published positives numbers by construction. Verified: it reproduces `slide-summary.csv`'s
`density_mean` bit-for-bit on spot-checked slides.

## Worklist sampling: every 4th FOV

**Decided 2026-08-21: the worklist is every 4th FOV of each slide's 324, not a score-stratified
draw.** All ten split slides are exactly 324 FOVs, so this is 81 per slide and 648 across the eight
that are not already annotated -- the same size the stratified plan called for. Only the selection
rule changed.

The reason is that stratifying within a slide's score range means **choosing v3's training FOVs
with v2.2's own predictions** -- the model v3 exists to replace. Every-4th needs no scores, so it
cannot inherit their errors. That is the same circularity the empty-field gate has, and which the
7-level ordinal above was adopted to break; it would be odd to remove it from the labels and leave
it in the sampling. Systematic 1-in-4 is also self-weighting, so each slide's annotated subset
reproduces that slide's own distribution and needs no `sampling_weight` correction of the kind
`blind-relabels-KEY.csv` carries. It is what makes the estimate above -- ~80 gated FOVs, from a
12.2% population rate -- exact rather than approximate.

**It does not need to supply the dense end, because that is already annotated.** The two
pre-annotated slides hold 648 exhaustively labelled FOVs including 51 `dense`, 47 `very dense`, 31
`rouleaux` and 61 `heavy rouleaux`. The five remaining train slides are Q0-Q3 and are there to
extend the *low* end of the range, where a self-weighting sample is the right instrument. So the
floor-per-bucket logic that `build_blind_relabels.py` needed does not apply here.

Two caveats that do survive:

- **The dense end is single-site.** Both pre-annotated slides are KTR / Box5, and 78 of the 98
  dense-or-denser FOVs and 74 of the 92 top-two-overlap FOVs come from `KTR-72502946` alone. The
  multi-site breadth the split was chosen for is therefore concentrated at the sparse end, and a
  Box5 imaging characteristic could be learned as a proxy for density. `KIT-62500670` (Q4,
  `slightly dense`, Box1) is the densest unannotated train slide and so the main non-KTR
  contribution to the dense end -- its 81 do the most structural work of any slide in the
  worklist.
- **Check for raster aliasing before starting.** 324 = 18^2, so if acquisition rasters an 18x18
  grid, stepping by 4 across rows of 18 (18 mod 4 = 2) alternates between two even-column phases
  and never lands on an odd column -- half the stage columns unsampled. The per-FOV CSVs carry
  only `fov_id`, so this cannot be checked from this repo. If it holds, rotate the start offset
  per row, or step by a value coprime to the row length.

## Next

1. ~~Settle the empty-field vocabulary~~ -- done: 7-level density ordinal, `_v3_common.py`.
2. ~~Generate the worklist~~ -- done, every 4th FOV, 82/slide.
3. ~~Annotate~~ -- done for the 5 train slides (412 usable FOVs); blind re-labels done 2026-08-21.
4. ~~Build `combined-v3/` proper~~ -- done: `merge_labels_v3.py`, `cohort_ranges.py`,
   `build_features_v3.py`, `_v3_fit.py`, `calibrate_v3.py`, `_v3_report.py`, and the two verify
   scripts. `calibrate_v2.py` took exactly two additive, default-preserving edits
   (`fit_ridge(sample_weight=)` and `axis_separation_check(density_levels=, overlap_levels=)`),
   both proven not to move v2.2.
5. **Score the 4 test slides.** By construction this happens once, after the fit is frozen, in its
   own script. The fit is now frozen: `verify_v3_fit.py` reproduces it at 0.000e+00.
6. **v3.1: the overlap axis.** This is where the two deferred levers go, and the overlap gate
   failure above is the reason to spend the run:
   - `hole_density` / `hole_frac` / `close_delta` behind a `hole_downsample` knob. Measured at
     partial rho 0.547 with overlap against -0.086 with density on 120 FOVs of one slide, so it
     still needs to reproduce on a second slide. Adding it requires a re-scoring pass over the
     159,716 cohort FOVs, because its p2/p98 must come from the cohort like the other eight --
     taking it from the 1060 training rows would reintroduce the training-centred normalization v3
     just removed.
   - Restore the partial-rho gate (`MIN_PARTIAL_RHO`, `select_axis_features`) on the overlap axis
     and report a gated fit beside the unconstrained one. The design doc's argument was that the
     gate becomes affordable *because* `hole_density` survives it, so this is downstream of the
     feature work rather than independent of it.
7. **Deployment pass** over the 487 non-training slides once the test scores are in, with the
   positive-vs-negative separation as the confound check: a crowding score that separates PCR
   classes is measuring something it should not.
8. Smaller, still open: the raster-aliasing question (324 = 18^2, 18 mod 4 = 2); weighted bootstrap
   CIs (`bootstrap_median_ci` is unweighted, seed 42, currently labelled approximate); and the
   overexposure study on the 23 excluded rows.
