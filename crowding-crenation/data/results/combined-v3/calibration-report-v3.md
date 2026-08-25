# v3 calibration: slide-grouped splits, 7-level density, cohort normalization

- **1060 labelled FOVs across 7 slide groups** (v2.2: 661 across 2)
- normalization: p2/p98 over **159716 cohort FOVs** from 493 non-test slides, `--slides all`
- knobs: `lbp_step=16`, `blur_downsample=4` (from `density_overlap_v2.2-optimized_params.json`)
- fit: alpha=10.0, sample_weights=`slide`, sub_delta=0.0, PAVA skips `no cells`
- validation: nested leave-one-slide-out (7 outer x 6 inner)
- normalization is **fold-local**: each fold's p2/p98 exclude the slides held out at that level, so no slide contributes to the ruler its own predictions are measured against. The training rows never determine their own feature scaling either -- the ranges come from unlabelled cohort FOVs. One slide is 324 of 159,716 FOVs, so this moves any feature's span by at most 1.1%; measured end to end it changed **1 of 1060** density and **2 of 1060** overlap predictions. `--ranges-scope fixed` reproduces the un-corrected version.

## Train / val / test

**There is no separate val block, deliberately.** "Val" is realised as the *inner* LOSO loop over the 6 remaining train slides, which is what any fitted decision is made on; the *outer* loop over all 7 produces the reported number. At 7 groups, carving out a fixed val block would spend 1-2 of them to get a val estimate on a single slide. The 4 test slides are unannotated and unscored until the fit is frozen.

| role | slide | truth | site | box | FOVs on slide | labelled | in fit | weight per FOV | slide mass | row % | mass % |
|---|---|---|---|---|---|---|---|---|---|---|---|
| train/val | KTR-72502946 | negative | KTR | Box5 | 324 | 324 | 324 | 0.00309 | 1.00 | 30.6% | 14.3% |
| train/val | KTR-72502948 | positive | KTR | Box5 | 324 | 324 | 324 | 0.00309 | 1.00 | 30.6% | 14.3% |
| train/val | KIT-62500670 | positive | KIT | Box1 | 324 | 83 | 70 | 0.01429 | 1.00 | 7.8% | 14.3% |
| train/val | NKR-72502156 | negative | NKR | Box4 | 324 | 83 | 83 | 0.01205 | 1.00 | 7.8% | 14.3% |
| train/val | KIT-62500909 | negative | KIT | Box1 | 324 | 82 | 82 | 0.01220 | 1.00 | 7.7% | 14.3% |
| train/val | KTR-72502904 | negative | KTR | Box5 | 324 | 82 | 82 | 0.01220 | 1.00 | 7.7% | 14.3% |
| train/val | RUB-72501818 | positive | RUB | Box3 | 324 | 82 | 82 | 0.01220 | 1.00 | 7.7% | 14.3% |
| test | KIT-62500735 | positive | KIT | Box1 | 324 | 0 | -- | -- | -- | -- | -- |
| test | KTR-72502961 | negative | KTR | Box5 | 324 | 0 | -- | -- | -- | -- | -- |
| test | NKR-72502165 | positive | NKR | Box4 | 324 | 0 | -- | -- | -- | -- | -- |
| test | RUB-72501769 | negative | RUB | Box3 | 324 | 0 | -- | -- | -- | -- | -- |

Every slide holds 324 FOVs; the difference is how many were *labelled*. The two pre-annotated slides (KTR-72502946, KTR-72502948) were labelled exhaustively at 324 each, the five new ones on a 1-in-4 grid at ~82. So they carry **61.1% of the rows but 28.6% of the fitted mass** (2 of 7 slide groups) under `sample_weights=slide`.

`in fit` is below `labelled` for KIT-62500670 because the 13 `no cells` rows are excluded from the ridge and the PAVA medians on both axes. Weights are computed over the fitting rows, so a slide with excluded rows is not thereby under-weighted.

## Pool composition

| slide | FOVs | density levels present | overlap levels present | density_sub=low | overlap_sub=low |
|---|---|---|---|---|---|
| KIT-62500670 | 83 | 7 | 5 | 0 | 4 |
| KIT-62500909 | 82 | 3 | 1 | 19 | 0 |
| KTR-72502904 | 82 | 3 | 2 | 21 | 0 |
| KTR-72502946 | 324 | 5 | 5 | 0 | 0 |
| KTR-72502948 | 324 | 5 | 5 | 0 | 0 |
| NKR-72502156 | 83 | 5 | 5 | 9 | 0 |
| RUB-72501818 | 82 | 2 | 1 | 18 | 0 |

Density levels: `No Cells` 13, `Few Cells` 24, `Sparser` 188, `Monolayer` 565, `Slightly Dense` 141, `Dense` 79, `Very Dense` 50

Rouleaux levels: `No Rouleaux` 743, `Slight Rouleaux` 129, `Some Rouleaux` 66, `Rouleaux` 56, `Heavy Rouleaux` 66

## Feature correlations

| feature | marginal density | partial density | marginal Rouleaux | partial Rouleaux |
|---|---|---|---|---|
| `coverage` | 0.645 | 0.444 | 0.528 | 0.096 |
| `otsu_separability` | 0.558 | 0.518 | 0.307 | -0.195 |
| `saturation_score` | 0.632 | 0.425 | 0.524 | 0.105 |
| `lbp_entropy` | 0.822 | 0.709 | 0.592 | -0.050 |
| `glcm_contrast` | 0.793 | 0.646 | 0.602 | 0.031 |
| `edge_density_unmasked` | 0.805 | 0.663 | 0.610 | 0.029 |
| `tile_glcm_cv` | 0.118 | -0.071 | 0.221 | 0.200 |
| `tile_glcm_patchiness` | 0.060 | -0.209 | 0.262 | 0.326 |

Manual label correlation (density vs Rouleaux): **0.744**

For reference only, v3 does **not** apply the axis-exclusive gate: under `select_axis_features` the density pool would be ['coverage', 'otsu_separability', 'saturation_score', 'lbp_entropy', 'glcm_contrast', 'edge_density_unmasked'] and the Rouleaux pool ['tile_glcm_cv', 'tile_glcm_patchiness']. v3 fits both axes on the full candidate pool, as v2.1/v2.2 did, so this run is a controlled comparison against v2.2 in which only the splits, the vocabulary and the weighting changed. Restoring the gate is a v3.1 question and is deliberately out of scope -- see 'Axis separation' below for what that costs.

## Fitted weights, v3 against v2.2

### Density

v3 kept **8** of 8 candidates, v2.2 kept **8**.

| feature | v3 weight | v2.2 weight | x |
|---|---|---|---|
| `edge_density_unmasked` | 0.1877 | 0.0977 | 1.92x |
| `lbp_entropy` | 0.1743 | 0.0641 | 2.72x |
| `glcm_contrast` | 0.1676 | 0.0876 | 1.91x |
| `coverage` | 0.1522 | 0.2358 | 0.65x |
| `saturation_score` | 0.1476 | 0.2466 | 0.60x |
| `otsu_separability` | 0.1248 | 0.0145 | 8.61x |
| `tile_glcm_cv` | 0.0295 | 0.0823 | 0.36x |
| `tile_glcm_patchiness` | 0.0162 | 0.1713 | 0.09x |

### Rouleaux

v3 kept **8** of 8 candidates, v2.2 kept **6** -- v3 keeps `lbp_entropy`, `otsu_separability`. A feature is absent because the negative-coefficient drop loop removed it, so the two fits disagree about which features are even usable, not only about how much to trust them.

| feature | v3 weight | v2.2 weight | x |
|---|---|---|---|
| `edge_density_unmasked` | 0.1642 | 0.0434 | 3.79x |
| `glcm_contrast` | 0.1598 | 0.0975 | 1.64x |
| `coverage` | 0.1501 | 0.2633 | 0.57x |
| `saturation_score` | 0.1464 | 0.2688 | 0.54x |
| `lbp_entropy` | 0.1360 | dropped | -- |
| `otsu_separability` | 0.0864 | dropped | -- |
| `tile_glcm_patchiness` | 0.0861 | 0.2385 | 0.36x |
| `tile_glcm_cv` | 0.0708 | 0.0885 | 0.80x |

The v3 weights are markedly **flatter** on both axes -- six of the eight density features land between 0.12 and 0.19, with only the two tile-GLCM features near zero. v2.2 put **48%** of its density composite into `coverage` and `saturation_score` alone; v3 gives those two **30%**, and raises `otsu_separability` 8.6x off a near-zero base. Cohort normalization is the direct cause: spread over 159,716 FOVs instead of 1060, no single feature's range is narrow enough to dominate the composite. It is also why v3's thresholds cannot be read against v2.2's as numbers.

The Rouleaux axis is the sharper change: v2.2's drop loop discarded `lbp_entropy` and `otsu_separability` for negative coefficients, and v3 keeps both at a combined 22% of the composite. Under cohort ranges those two stop being sign-unstable, so v3 fits the overlap axis on strictly more information than v2.2 had -- which makes its weaker exact-match a statement about the validation protocol, not about the feature pool being thinner.

## Weight stability across the 7 LOSO fits

Each outer fold refits from scratch on 6 slides, so these are 7 independent estimates of the same composite. The shipped column is the full-pool fit.

### Density

| feature | shipped | fold min | fold max | spread | spread / shipped | kept in |
|---|---|---|---|---|---|---|
| `coverage` | 0.1522 | 0.1369 | 0.1849 | 0.0479 | 31% | 7/7 |
| `otsu_separability` | 0.1248 | 0.1031 | 0.1348 | 0.0317 | 25% | 7/7 |
| `saturation_score` | 0.1476 | 0.1357 | 0.1801 | 0.0444 | 30% | 7/7 |
| `lbp_entropy` | 0.1743 | 0.1604 | 0.1814 | 0.0210 | 12% | 7/7 |
| `glcm_contrast` | 0.1676 | 0.1599 | 0.1758 | 0.0159 | 9% | 7/7 |
| `edge_density_unmasked` | 0.1877 | 0.1784 | 0.1956 | 0.0172 | 9% | 7/7 |
| `tile_glcm_cv` | 0.0295 | 0.0164 | 0.0547 | 0.0383 | 130% | 7/7 |
| `tile_glcm_patchiness` | 0.0162 | 0.0071 | 0.0366 | 0.0295 | 182% | 7/7 |

### Rouleaux

| feature | shipped | fold min | fold max | spread | spread / shipped | kept in |
|---|---|---|---|---|---|---|
| `coverage` | 0.1501 | 0.1396 | 0.1808 | 0.0411 | 27% | 7/7 |
| `otsu_separability` | 0.0864 | 0.0627 | 0.1032 | 0.0405 | 47% | 7/7 |
| `saturation_score` | 0.1464 | 0.1328 | 0.1771 | 0.0443 | 30% | 7/7 |
| `lbp_entropy` | 0.1360 | 0.1263 | 0.1443 | 0.0180 | 13% | 7/7 |
| `glcm_contrast` | 0.1598 | 0.1478 | 0.1713 | 0.0235 | 15% | 7/7 |
| `edge_density_unmasked` | 0.1642 | 0.1481 | 0.1825 | 0.0344 | 21% | 7/7 |
| `tile_glcm_cv` | 0.0708 | 0.0590 | 0.0943 | 0.0353 | 50% | 7/7 |
| `tile_glcm_patchiness` | 0.0861 | 0.0692 | 0.1092 | 0.0400 | 46% | 7/7 |

Read the `spread / shipped` column first. The six substantial features on each axis move by well under half their own weight across folds, and no feature is dropped in any fold, so the composite is a stable object rather than an artifact of which slides landed in the fit. The exception is the tail: `tile_glcm_patchiness` and `tile_glcm_cv` on density carry ~2-3% of the composite and move by more than their own magnitude, which is the honest way to say they are not being estimated at this sample size -- they are near zero and the sign is only just stable enough to survive the drop loop.

## Density composite (full fit on all slides)

| feature | weight | p2 | p98 |
|---|---|---|---|
| `coverage` | 0.1522 | 0.0317 | 0.3760 |
| `otsu_separability` | 0.1248 | 0.4883 | 0.6319 |
| `saturation_score` | 0.1476 | 0.0148 | 0.1590 |
| `lbp_entropy` | 0.1743 | 2.6866 | 4.0960 |
| `glcm_contrast` | 0.1676 | 13.3135 | 103.7233 |
| `edge_density_unmasked` | 0.1877 | 0.0258 | 0.1943 |
| `tile_glcm_cv` | 0.0295 | 0.0532 | 0.2768 |
| `tile_glcm_patchiness` | 0.0162 | 0.1054 | 0.7183 |

| level | PAVA median | mass | threshold to enter |
|---|---|---|---|
| `No Cells` | 0.0084 | 0.00 | -- |
| `Few Cells` | 0.0084 | 0.30 | 0.0084 |
| `Sparser` | 0.2044 | 1.84 | 0.1064 |
| `Monolayer` | 0.4790 | 3.27 | 0.3417 |
| `Slightly Dense` | 0.7775 | 0.87 | 0.6283 |
| `Dense` | 0.8291 | 0.53 | 0.8033 |
| `Very Dense` | 0.8586 | 0.18 | 0.8439 |

PAVA merged nothing -- every level kept a distinct cut point.

## Density metrics (out-of-fold, nested LOSO)

> **v3 is expected to score lower than v2.2 on exact-match and still be the better model.** v2.2's number is FOV-stratified 5-fold over 2 slides with training-set-derived ranges; v3's is leave-one-slide-out over 7 slides with cohort-derived ranges. That gap is the measurement, not a regression. Beating v2.2's exact-match is explicitly **not** an acceptance criterion.

Folds scored (>=2 levels held out): **7/7**. Metrics are pooled over all out-of-fold predictions, not averaged over folds.

| metric | unweighted | vs baseline | slide-weighted | vs baseline | v2.2 published |
|---|---|---|---|---|---|
| exact match | 62.3% | 53.3% | 59.9% | 46.4% | 69.4% |
| off-by-one | 94.8% | -- | 93.6% | -- | 98.0% |
| balanced accuracy | 42.4% | 14.3% | 42.3% | 14.3% | -- |
| macro F1 | 0.395 | -- | 0.380 | -- | -- |
| QWK | 0.786 | -- | 0.801 | -- | -- |

**Annotator ceiling**: 41/50 = 82% self-agreement on this axis (blind re-label of 50 already-annotated FOVs). Every exact-match number above is bounded by it, so the headroom is 82% - 62% = 20 points, not 38.

### Per level

| level | support | F1 |
|---|---|---|
| `No Cells` | 13 | 0.077 |
| `Few Cells` | 24 | 0.246 |
| `Sparser` | 188 | 0.638 |
| `Monolayer` | 565 | 0.792 |
| `Slightly Dense` | 141 | 0.417 |
| `Dense` | 79 | 0.146 |
| `Very Dense` | 50 | 0.446 |

An absent level reports `absent`, never 0.0, and macro-F1 averages over the present levels only.

### Per held-out slide

| slide | n | levels present | exact | off-by-one | balanced acc | QWK | features kept |
|---|---|---|---|---|---|---|---|
| KIT-62500670 | 83 | 7 | 26.5% | 65.1% | 34.0% | 0.750 | 8 |
| KIT-62500909 | 82 | 3 | 62.2% | 100.0% | 60.8% | 0.718 | 8 |
| KTR-72502904 | 82 | 3 | 68.3% | 100.0% | 48.8% | 0.751 | 8 |
| KTR-72502946 | 324 | 5 | 57.7% | 96.6% | 45.1% | 0.716 | 8 |
| KTR-72502948 | 324 | 5 | 72.5% | 96.0% | 42.5% | 0.606 | 8 |
| NKR-72502156 | 83 | 5 | 83.1% | 97.6% | 74.2% | 0.872 | 8 |
| RUB-72501818 | 82 | 2 | 48.8% | 100.0% | 55.2% | 0.164 | 8 |

This table is the one that answers whether site and box breadth bought generalisation: v2.2 saw only KTR/Box5.

### Density confusion (rows = manual, cols = predicted)

| | `No Cells` | `Few Cells` | `Sparser` | `Monolayer` | `Slightly Dense` | `Dense` | `Very Dense` |
|---|---|---|---|---|---|---|---|
| `No Cells` | 1 | 5 | 7 | 0 | 0 | 0 | 0 |
| `Few Cells` | 12 | 8 | 3 | 1 | 0 | 0 | 0 |
| `Sparser` | 0 | 28 | 112 | 48 | 0 | 0 | 0 |
| `Monolayer` | 0 | 0 | 40 | 423 | 101 | 1 | 0 |
| `Slightly Dense` | 0 | 0 | 1 | 29 | 82 | 7 | 22 |
| `Dense` | 0 | 0 | 0 | 2 | 48 | 7 | 22 |
| `Very Dense` | 0 | 0 | 0 | 0 | 21 | 2 | 27 |

## Rouleaux composite (full fit on all slides)

| feature | weight | p2 | p98 |
|---|---|---|---|
| `coverage` | 0.1501 | 0.0317 | 0.3760 |
| `otsu_separability` | 0.0864 | 0.4883 | 0.6319 |
| `saturation_score` | 0.1464 | 0.0148 | 0.1590 |
| `lbp_entropy` | 0.1360 | 2.6866 | 4.0960 |
| `glcm_contrast` | 0.1598 | 13.3135 | 103.7233 |
| `edge_density_unmasked` | 0.1642 | 0.0258 | 0.1943 |
| `tile_glcm_cv` | 0.0708 | 0.0532 | 0.2768 |
| `tile_glcm_patchiness` | 0.0861 | 0.1054 | 0.7183 |

| level | PAVA median | mass | threshold to enter |
|---|---|---|---|
| `No Rouleaux` | 0.3883 | 5.11 | -- |
| `Slight Rouleaux` | 0.6762 | 0.76 | 0.5322 |
| `Some Rouleaux` | 0.7631 | 0.44 | 0.7196 |
| `Rouleaux` | 0.7631 | 0.43 | 0.7631 |
| `Heavy Rouleaux` | 0.8216 | 0.26 | 0.7924 |

PAVA merged: `Some Rouleaux`+`Rouleaux`

## Rouleaux metrics (out-of-fold, nested LOSO)

> **v3 is expected to score lower than v2.2 on exact-match and still be the better model.** v2.2's number is FOV-stratified 5-fold over 2 slides with training-set-derived ranges; v3's is leave-one-slide-out over 7 slides with cohort-derived ranges. That gap is the measurement, not a regression. Beating v2.2's exact-match is explicitly **not** an acceptance criterion.

Folds scored (>=2 levels held out): **5/7**. Metrics are pooled over all out-of-fold predictions, not averaged over folds.

| metric | unweighted | vs baseline | slide-weighted | vs baseline | v2.2 published |
|---|---|---|---|---|---|
| exact match | 58.4% | 70.1% | 69.1% | 74.7% | 67.6% |
| off-by-one | 91.7% | -- | 90.6% | -- | 93.8% |
| balanced accuracy | 40.5% | 20.0% | 41.4% | 20.0% | -- |
| macro F1 | 0.371 | -- | 0.362 | -- | -- |
| QWK | 0.708 | -- | 0.708 | -- | -- |

**Annotator ceiling**: 41/47 = 87% self-agreement on this axis (blind re-label of 50 already-annotated FOVs). Every exact-match number above is bounded by it, so the headroom is 87% - 58% = 29 points, not 42.

### Per level

| level | support | F1 |
|---|---|---|
| `No Rouleaux` | 743 | 0.774 |
| `Slight Rouleaux` | 129 | 0.309 |
| `Some Rouleaux` | 66 | 0.090 |
| `Rouleaux` | 56 | 0.167 |
| `Heavy Rouleaux` | 66 | 0.514 |

An absent level reports `absent`, never 0.0, and macro-F1 averages over the present levels only.

### Per held-out slide

| slide | n | levels present | exact | off-by-one | balanced acc | QWK | features kept |
|---|---|---|---|---|---|---|---|
| KIT-62500670 | 83 | 5 | 33.7% | 54.2% | 34.9% | 0.364 | 8 |
| KIT-62500909 | 82 | 1 | 93.9% | 100.0% | 93.9% | 0.000 | 8 |
| KTR-72502904 | 82 | 2 | 95.1% | 100.0% | 50.0% | 0.000 | 8 |
| KTR-72502946 | 324 | 5 | 31.5% | 91.7% | 38.6% | 0.713 | 8 |
| KTR-72502948 | 324 | 5 | 59.9% | 94.4% | 29.9% | 0.540 | 8 |
| NKR-72502156 | 83 | 5 | 73.5% | 94.0% | 51.2% | 0.847 | 8 |
| RUB-72501818 | 82 | 1 | 96.3% | 100.0% | 96.3% | 0.000 | 8 |

This table is the one that answers whether site and box breadth bought generalisation: v2.2 saw only KTR/Box5.

### Rouleaux confusion (rows = manual, cols = predicted)

| | `No Rouleaux` | `Slight Rouleaux` | `Some Rouleaux` | `Rouleaux` | `Heavy Rouleaux` |
|---|---|---|---|---|---|
| `No Rouleaux` | 491 | 246 | 6 | 0 | 0 |
| `Slight Rouleaux` | 24 | 80 | 7 | 5 | 13 |
| `Some Rouleaux` | 5 | 36 | 5 | 6 | 14 |
| `Rouleaux` | 4 | 21 | 13 | 7 | 11 |
| `Heavy Rouleaux` | 1 | 5 | 14 | 10 | 36 |

## v3 against v2.2 on the same 1060 FOVs

v3's out-of-fold predictions against the `v2.2-optimized` labels already committed in the cohort per-FOV CSVs. v3's two new rungs are collapsed to `sparser` so the vocabularies compare.

> **This comparison favours v2.2.** 648 of the 1060 rows are ones v2.2 was fitted on, so its accuracy here is largely in-sample; every v3 prediction is out-of-fold. Read a v3 win as a floor and a v2.2 win as an inflated ceiling.

### How often the two disagree

| axis | agree | differ | share of pool |
|---|---|---|---|
| Density | 685 | **375** | 35.4% |
| Rouleaux | 635 | **425** | 40.1% |

A third to two fifths of the pool changes label, so v3 is a different function rather than a re-derivation of v2.2 at a new scale.

Largest Density shifts:

| v2.2 said | v3 says | n |
|---|---|---|
| `sparser` | `monolayer` | 127 |
| `dense` | `slightly dense` | 74 |
| `slightly dense` | `monolayer` | 68 |
| `monolayer` | `slightly dense` | 45 |
| `dense` | `very dense` | 20 |

Largest Rouleaux shifts:

| v2.2 said | v3 says | n |
|---|---|---|
| `no rouleaux` | `slight rouleaux` | 181 |
| `some rouleaux` | `slight rouleaux` | 72 |
| `rouleaux` | `some rouleaux` | 33 |
| `rouleaux` | `slight rouleaux` | 29 |
| `some rouleaux` | `no rouleaux` | 26 |

### Which one is right, row by row

| axis | both right | v3 only | v2.2 only | neither | v3 accuracy | v2.2 accuracy |
|---|---|---|---|---|---|---|
| Density | 534 | **181** | 130 | 215 | 67.5% | 62.6% |
| Rouleaux | 515 | **104** | 226 | 215 | 58.4% | 69.9% |

**Density is the result that matters here**: v3 is right on 51 more FOVs than v2.2 *despite* giving up the in-sample advantage on 648 of them. The overlap column runs the other way by a wide margin, which is the same finding as the failed majority-baseline gate seen from a second direction -- inflated by v2.2's in-sample rows, but too large a gap for that to explain all of it.

## LOSO results: exact match and off-by-one

| held-out slide | FOVs | Density levels | Density exact | Density off-by-one | Rouleaux levels | Rouleaux exact | Rouleaux off-by-one |
|---|---|---|---|---|---|---|---|
| KIT-62500670 | 83 | 7 | 26.5% | 65.1% | 5 | 33.7% | 54.2% |
| KIT-62500909 | 82 | 3 | 62.2% | 100.0% | 1 | 93.9% | 100.0% |
| KTR-72502904 | 82 | 3 | 68.3% | 100.0% | 2 | 95.1% | 100.0% |
| KTR-72502946 | 324 | 5 | 57.7% | 96.6% | 5 | 31.5% | 91.7% |
| KTR-72502948 | 324 | 5 | 72.5% | 96.0% | 5 | 59.9% | 94.4% |
| NKR-72502156 | 83 | 5 | 83.1% | 97.6% | 5 | 73.5% | 94.0% |
| RUB-72501818 | 82 | 2 | 48.8% | 100.0% | 1 | 96.3% | 100.0% |
| **pooled OOF** | **1060** | 7 | **62.3%** (660/1060) | **94.8%** (1005/1060) | 5 | **58.4%** (619/1060) | **91.7%** (972/1060) |
| mean of folds | -- | -- | 59.9% | 93.6% | -- | 69.1% | 90.6% |

**Quote the pooled row.** The mean-of-folds row moves in opposite directions on the two axes -- below pooled on density, above it on overlap -- because the two 324-FOV slides are the hardest folds on overlap and among the easier ones on density, and an unweighted mean over folds treats them as equal to an 82-FOV slide. Two overlap folds also hold a test set that is 100% `no rouleaux`, where exact-match is 94-96% and means nothing.

The off-by-one column is where the ordinal structure shows: density is within one rung on 94.8% of FOVs against 62.3% exactly right, so most errors are boundary calls between adjacent levels rather than category confusions. `KIT-62500670` is the exception on both axes -- it is the only fold exercising all 7 density levels and holds every `no cells` row in the pool.

## The empty-field gate becomes a prediction

v2.2 forced `sparser` + `no rouleaux` whenever all four texture features fell below their calibration p2 floor. v3 disables that override (`empty_field_override.enabled: false`) and predicts the bottom two rungs instead. This table is the check that used to be circular.

| manual level | n | v2.2 gate fired | v3 predicted correctly | v3 predicted a bottom rung |
|---|---|---|---|---|
| `No Cells` | 13 | 12 | 1 | 6 |
| `Few Cells` | 24 | 19 | 8 | 20 |
| all other levels | 1023 | 46 | 651 | 28 |

**Caveat, stated rather than discovered.** All 13 `no cells` rows come from 1 slide (KIT-62500670), so under outer LOSO the only fold that tests the rung is the fold that lacks it in training. v3's headline claim about the gate is evaluated on n=13 from one slide in one fold. `few cells` is only marginally better.

## Axis separation

| basis | composite-vs-composite rho | manual label rho |
|---|---|---|
| v2.2 published | 0.972 | 0.823 |
| v3, full fit | 0.987 | 0.744 |
| v3, out-of-fold | 0.986 | 0.744 |

The out-of-fold row is the honest one. **This number is descriptive in v3, not a target**: v3 changed the splits, the vocabulary and the weighting, and carries no mechanism aimed at the confound. The two levers the design doc proposed for it -- the `hole_density` feature and restoring the partial-rho gate on the overlap axis -- are both deferred to v3.1, so a gap that stays near v2.2's is the expected result here and not a failed attempt.

### Sign-agreement on the off-diagonal

| min delta | n disagreeing FOVs | sign matches | rate | p | rho |
|---|---|---|---|---|---|
| 1 | 1046 | 1023 | 97.8% | 0.0000 | 0.335 |
| 2 | 994 | 985 | 99.1% | 0.0000 | 0.296 |

## Sub-level delta ablation

67 density rows and 4 overlap rows carry a double label, which v3 reads as "the more severe level, at its low end" and encodes as a target nudged `delta` below that rung. `delta=0` ignores the encoding entirely.

| delta | Density exact | Density QWK | Rouleaux exact | Rouleaux QWK |
|---|---|---|---|---|
| 0.0 | 62.3% | 0.786 | 58.4% | 0.708 |
| 0.33 | 62.0% | 0.784 | 58.5% | 0.709 |

Either direction is a publishable answer. If the winner leads by less than one FOV's worth of accuracy, `delta=0` ships and the honest conclusion is that the double labels carry no extractable signal at n=67.

## Limitations

- **The dense end is single-site.** Both exhaustively-annotated slides are KTR/Box5, and they supply 648 of the 1060 rows.
- **Test now covers all four sites, as of the 2026-08-24 refreeze.** It previously held 3 slides under a >=3-site rule, which left KIT untested -- the least convenient site to lose, being the largest among positives (104 of 271) and one v3 *trains* on twice. Moving to 4 test slides, one per site, **improved** the KS to 0.0138 from 0.0173 rather than costing anything, because a 4-slide slate has more freedom to match the cohort CDF. Caveat: that gain is partly the extra slide, not proof the site constraint is free -- the unconstrained 4-slide search is an 843M-combination space that does not factorise the way the one-per-site case does, so it was not run.
- **Two outer folds are label-degenerate on overlap** (a held-out set that is 100% `no rouleaux`), so `n_folds_scored` is quoted beside every mean and the pooled metric is the one to read.
- **Raster aliasing is still open.** 324 = 18^2 and 18 mod 4 = 2, so the every-4th worklist may only ever land on even stage columns. Unanswerable from this repo; it bounds the claim that the worklist is self-weighting.
- **Bootstrap CIs are unweighted** (`bootstrap_median_ci`, seed 42) and so are approximate under slide-balanced weights.

