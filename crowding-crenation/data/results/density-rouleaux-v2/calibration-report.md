# Density + Rouleaux v2 calibration report

n = 337 FOVs (13 from `initial-dataset-071626` + 324 from `tanzania-073026`). Manual density-vs-Rouleaux confound: Spearman rho = 0.671.

## Marginal + partial correlation (all candidate features)

| feature | marginal density | partial density | marginal Rouleaux | partial Rouleaux |
|---|---|---|---|---|
| coverage | 0.640 | 0.512 | 0.447 | 0.032 |
| otsu_separability | 0.394 | 0.285 | 0.284 | 0.030 |
| saturation_score | 0.627 | 0.510 | 0.423 | 0.005 |
| lbp_entropy | 0.607 | 0.539 | 0.346 | -0.104 |
| glcm_contrast | 0.691 | 0.534 | 0.529 | 0.122 |
| edge_density_unmasked | 0.684 | 0.537 | 0.507 | 0.090 |
| tile_glcm_cv | -0.067 | -0.207 | 0.126 | 0.231 |
| tile_glcm_patchiness | -0.042 | -0.262 | 0.219 | 0.334 |

Feature-selection rule: assign a feature to whichever axis has the higher *partial* correlation, provided it exceeds 0.05; otherwise excluded from both composites.

- Density candidate features: coverage, otsu_separability, saturation_score, lbp_entropy, glcm_contrast, edge_density_unmasked
- Rouleaux candidate features: tile_glcm_cv, tile_glcm_patchiness

## Density composite

Dropped for sign instability (negative ridge coefficient despite positive partial correlation -- a multicollinearity artifact, not a real inverse relationship): otsu_separability

Fitted weights (ridge regression on percentile-normalized features, refit on the full calibration set after cross-validation below):

| feature | weight | range (2nd-98th pct) |
|---|---|---|
| coverage | 0.226 | [0.06671, 0.4952] |
| saturation_score | 0.257 | [0.03137, 0.2016] |
| lbp_entropy | 0.062 | [3.048, 4.049] |
| glcm_contrast | 0.241 | [24.23, 94.2] |
| edge_density_unmasked | 0.215 | [0.05428, 0.1813] |

**Cross-validation** (5-fold, stratified by bucket): per-fold raw-score Spearman rho = [0.759, 0.707, 0.708, 0.616, 0.582], mean=0.674 (std=0.065). Out-of-fold overall rho=0.670.

Out-of-fold exact-match rate: 55.2%. Off-by-one rate: 92.6%.

Out-of-fold confusion matrix (rows=manual, cols=predicted):

| manual \ predicted | Sparser | Monolayer | Slightly Dense | Dense | Very Dense |
|---|---|---|---|---|---|
| Sparser | 44 | 0 | 0 | 0 | 0 |
| Monolayer | 45 | 127 | 44 | 13 | 3 |
| Slightly Dense | 1 | 14 | 6 | 8 | 5 |
| Dense | 0 | 1 | 2 | 3 | 7 |
| Very Dense | 0 | 0 | 2 | 6 | 6 |

**Bucket thresholds** (median-per-bucket, PAVA-corrected for monotonicity, midpoint cut points), with bucket counts and bootstrap 90% CI on each bucket's median raw score:

| bucket | n | median raw score | 90% CI |
|---|---|---|---|
| Sparser | 44 | 0.075 | [0.056, 0.114] |
| Monolayer | 232 | 0.418 | [0.395, 0.435] |
| Slightly Dense | 34 | 0.546 | [0.449, 0.622] |
| Dense | 13 | 0.686 | [0.626, 0.830] |
| Very Dense | 14 | 0.686 | [0.646, 0.726] |

Thresholds: [0.247, 0.482, 0.616, 0.686]

**Note:** PAVA merged the following adjacent buckets because their median raw scores were not monotonic at n=337 -- an honest finding that these buckets aren't cleanly separable by the current features/sample size, not a fitting bug: [['Dense', 'Very Dense']]

## Rouleaux composite

Fitted weights (ridge regression on percentile-normalized features, refit on the full calibration set after cross-validation below):

| feature | weight | range (2nd-98th pct) |
|---|---|---|
| tile_glcm_cv | 0.183 | [0.1179, 0.3257] |
| tile_glcm_patchiness | 0.817 | [0.1812, 1.02] |

**Cross-validation** (5-fold, stratified by bucket): per-fold raw-score Spearman rho = [0.207, 0.057, 0.361, 0.211, 0.23], mean=0.213 (std=0.096). Out-of-fold overall rho=0.208.

Out-of-fold exact-match rate: 44.8%. Off-by-one rate: 79.5%.

Out-of-fold confusion matrix (rows=manual, cols=predicted):

| manual \ predicted | No Rouleaux | Slight Rouleaux | Some Rouleaux | Rouleaux | Heavy Rouleaux |
|---|---|---|---|---|---|
| No Rouleaux | 121 | 75 | 27 | 0 | 19 |
| Slight Rouleaux | 28 | 22 | 3 | 0 | 3 |
| Some Rouleaux | 1 | 4 | 2 | 0 | 8 |
| Rouleaux | 2 | 2 | 3 | 0 | 4 |
| Heavy Rouleaux | 3 | 2 | 2 | 0 | 6 |

**Bucket thresholds** (median-per-bucket, PAVA-corrected for monotonicity, midpoint cut points), with bucket counts and bootstrap 90% CI on each bucket's median raw score:

| bucket | n | median raw score | 90% CI |
|---|---|---|---|
| No Rouleaux | 242 | 0.213 | [0.191, 0.241] |
| Slight Rouleaux | 56 | 0.216 | [0.184, 0.257] |
| Some Rouleaux | 15 | 0.603 | [0.375, 0.995] |
| Rouleaux | 11 | 0.603 | [0.277, 0.696] |
| Heavy Rouleaux | 13 | 0.603 | [0.259, 0.656] |

Thresholds: [0.214, 0.41, 0.603, 0.603]

**Note:** PAVA merged the following adjacent buckets because their median raw scores were not monotonic at n=337 -- an honest finding that these buckets aren't cleanly separable by the current features/sample size, not a fitting bug: [['Some Rouleaux', 'Rouleaux', 'Heavy Rouleaux']]

## Axis-separation check

Among FOVs where manual density-rank and Rouleaux-rank disagree by at least `min_delta` levels (the dense-but-not-Rouleauxed / Rouleauxed-but-not-dense cases this tool is meant to separate): do the out-of-fold predicted scores diverge in the same direction, at a better-than-chance rate?

| min |delta| | n | sign matches | match rate | binomial p (one-sided, vs. 0.5) | Spearman rho (predicted vs. manual delta) |
|---|---|---|---|---|---|
| 1 | 227 | 139 | 61.2% | 0.0004326 | 0.199 |
| 2 | 3 | 3 | 100.0% | 0.125 | 0.500 |

Qualitative spot-check (manual vs. out-of-fold predicted, |delta|>=2 subset):

| FOV | manual density | manual Rouleaux | predicted density | predicted Rouleaux |
|---|---|---|---|---|
| tanzania-073026/dpc-034-KTR-72502948.png | Slightly Dense | No Rouleaux | Dense | No Rouleaux |
| tanzania-073026/dpc-133-KTR-72502948.png | Sparser | Some Rouleaux | Sparser | Heavy Rouleaux |
| tanzania-073026/dpc-203-KTR-72502948.png | Very Dense | No Rouleaux | Dense | Some Rouleaux |

## Known limitations

**Cross-slide / cross-stain generalization risk.** All candidate features are raw pixel/intensity statistics (Otsu coverage, GLCM contrast, edge density, LBP entropy), which are sensitive to staining protocol, scanner, and illumination -- not just true cell density. Median feature values differ substantially between the two source datasets:

| dataset | n | coverage | otsu_separability | saturation_score | lbp_entropy | glcm_contrast | edge_density_unmasked | tile_glcm_cv | tile_glcm_patchiness |
|---|---|---|---|---|---|---|---|---|---|
| initial-071626 | 13 | 0.496 | 0.558 | 0.206 | 3.97 | 98.7 | 0.182 | 0.151 | 0.242 |
| tanzania-073026 | 324 | 0.179 | 0.559 | 0.0774 | 3.79 | 55.6 | 0.129 | 0.197 | 0.351 |

Concretely: `dpc-051-LB-D3-...` (initial-071626, Liberia slide, manually labeled *monolayer*) has Otsu coverage 0.79 -- roughly 4x a typical Tanzania monolayer FOV (~0.18-0.21) -- and the density composite accordingly (mis)scores it as very dense. Only 13 of 337 calibration FOVs are non-Tanzania, so this pipeline is validated mainly for the Tanzania KTR-72502948 slide/stain; **spot-check `score_fov_v2.py` output against a handful of manual labels before trusting it on a new slide or stain, and expect to refit if there's a systematic offset.**

**Rouleaux composite is markedly weaker than density** (CV mean rho 0.21 vs. 0.67) and relies on only two features (both new tile-heterogeneity measures) -- expected, since prior work in this repo already established Rouleaux is intrinsically harder to capture with instance-free texture statistics, but it means Rouleaux predictions should be trusted less than density predictions, especially at the upper end where PAVA had to merge three buckets together.


---

# v2.1 recalibration: full-feature-pool fitting

The v2 calibration above assigned each candidate feature to exactly one axis (whichever had the higher *partial* correlation). That worked for density but left the Rouleaux composite with only `tile_glcm_cv`/`tile_glcm_patchiness`, whose per-level medians turned out to be a genuine **inverted-U** with Rouleaux severity (Some Rouleaux scored *higher* patchiness than Rouleaux or Heavy Rouleaux) -- a real reflection of the user's own predicted exception: a severe, confluent Rouleaux sheet reads as smoother/more homogeneous than a moderate patchy case. No monotonic threshold cut on those two features alone could rank-order the top three levels, so PAVA merged them. `glcm_contrast` and `edge_density_unmasked` -- excluded from Rouleaux in v2 because their partial correlation favored density -- are each cleanly monotonic across exactly that range.

v2.1 fits both composites directly against their own ordinal label using the full 8-feature candidate pool (`coverage, otsu_separability, saturation_score, lbp_entropy, glcm_contrast, edge_density_unmasked, tile_glcm_cv, tile_glcm_patchiness`), instead of the axis-exclusive subset -- same `fit_weights_stable` ridge-plus-sign-drop fitting machinery, just a larger candidate pool per axis.

## Density composite (v2.1)

| feature | weight | range (2nd-98th pct) |
|---|---|---|
| coverage | 0.149 | [0.06671, 0.4952] |
| otsu_separability | 0.022 | [0.5116, 0.5941] |
| saturation_score | 0.162 | [0.03137, 0.2016] |
| lbp_entropy | 0.077 | [3.048, 4.049] |
| glcm_contrast | 0.177 | [24.23, 94.2] |
| edge_density_unmasked | 0.175 | [0.05428, 0.1813] |
| tile_glcm_cv | 0.050 | [0.1179, 0.3257] |
| tile_glcm_patchiness | 0.187 | [0.1812, 1.02] |

**Cross-validation** (5-fold): per-fold rho = [0.782, 0.754, 0.687, 0.663, 0.641], mean=0.705. Out-of-fold exact-match=62.0%, off-by-one=98.5%.

Out-of-fold confusion matrix (rows=manual, cols=predicted):

| manual \ predicted | Sparser | Monolayer | Slightly Dense | Dense | Very Dense |
|---|---|---|---|---|---|
| Sparser | 42 | 2 | 0 | 0 | 0 |
| Monolayer | 41 | 143 | 46 | 2 | 0 |
| Slightly Dense | 0 | 12 | 14 | 7 | 1 |
| Dense | 0 | 0 | 2 | 5 | 6 |
| Very Dense | 0 | 0 | 2 | 7 | 5 |

Thresholds: [0.281, 0.454, 0.602, 0.675] -- **PAVA still merged**: [['Dense', 'Very Dense']]

## Rouleaux composite (v2.1)

Dropped for sign instability: lbp_entropy

| feature | weight | range (2nd-98th pct) |
|---|---|---|
| coverage | 0.161 | [0.06671, 0.4952] |
| otsu_separability | 0.016 | [0.5116, 0.5941] |
| saturation_score | 0.164 | [0.03137, 0.2016] |
| glcm_contrast | 0.178 | [24.23, 94.2] |
| edge_density_unmasked | 0.124 | [0.05428, 0.1813] |
| tile_glcm_cv | 0.077 | [0.1179, 0.3257] |
| tile_glcm_patchiness | 0.279 | [0.1812, 1.02] |

**Cross-validation** (5-fold): per-fold rho = [0.512, 0.652, 0.706, 0.625, 0.603], mean=0.620. Out-of-fold exact-match=64.1%, off-by-one=92.6%.

Out-of-fold confusion matrix (rows=manual, cols=predicted):

| manual \ predicted | No Rouleaux | Slight Rouleaux | Some Rouleaux | Rouleaux | Heavy Rouleaux |
|---|---|---|---|---|---|
| No Rouleaux | 191 | 40 | 8 | 3 | 0 |
| Slight Rouleaux | 16 | 20 | 17 | 3 | 0 |
| Some Rouleaux | 2 | 6 | 1 | 2 | 4 |
| Rouleaux | 0 | 0 | 6 | 2 | 3 |
| Heavy Rouleaux | 0 | 0 | 5 | 6 | 2 |

Thresholds: [0.357, 0.439, 0.529, 0.617] -- **no PAVA merges** (all 5 buckets monotonically separable).

## Composite independence (v2.1)

Spearman rho between the two fitted composite scores: **0.952**, vs. the true manual density-vs-Rouleaux label correlation of **0.671**. This is the real trade-off of full-pool fitting: the composites are now noticeably more alike than the two severity axes actually are, because several of the best-fitting features (coverage, glcm_contrast, edge_density_unmasked) are legitimately shared between both axes rather than exclusive to one. Reported here rather than hidden.

## Axis-separation check (v2.1)

| min |delta| | n | sign matches | match rate | binomial p | Spearman rho |
|---|---|---|---|---|---|
| 1 | 227 | 141 | 62.1% | 0.0001591 | 0.098 |
| 2 | 3 | 1 | 33.3% | 0.875 | 0.000 |

**Takeaway**: despite the composites becoming much more correlated with each other, the axis-separation match rate barely moves relative to v2 (~61% vs ~62% at `|delta|>=1`, both significant) -- so the extra shared signal does not appear to cost the tool its ability to separate the specific dense-but-not-Rouleauxed / Rouleauxed-but-not-dense cases it's meant to catch, even though it does trade away composite independence as a diagnostic property.


---

# v2.2 recalibration: pooling in tanzania-080526 (KTR-72502946)

Same full-feature-pool fitting as v2.1, refit on 661 FOVs (the original 337 plus 324 more from a second Tanzania slide, KTR-72502946 -- streamed from GCS, never downloaded locally). This tests whether v2.1's thresholds, fit on a single slide's label distribution, generalize to a second slide once that slide's own labels are pooled in rather than held out.

## Density composite (v2.2)

| feature | weight | range (2nd-98th pct) |
|---|---|---|
| coverage | 0.236 | [0.07347, 0.4386] |
| otsu_separability | 0.014 | [0.5166, 0.6124] |
| saturation_score | 0.247 | [0.03368, 0.1781] |
| lbp_entropy | 0.065 | [3.118, 4.196] |
| glcm_contrast | 0.087 | [26.59, 87.11] |
| edge_density_unmasked | 0.098 | [0.05926, 0.1811] |
| tile_glcm_cv | 0.082 | [0.06242, 0.3111] |
| tile_glcm_patchiness | 0.171 | [0.1245, 0.9339] |

**Cross-validation** (5-fold): per-fold rho = [0.79, 0.794, 0.789, 0.76, 0.782], mean=0.783. Out-of-fold exact-match=69.4%, off-by-one=98.0%.

Out-of-fold confusion matrix (rows=manual, cols=predicted):

| manual \ predicted | Sparser | Monolayer | Slightly Dense | Dense | Very Dense |
|---|---|---|---|---|---|
| Sparser | 45 | 5 | 0 | 0 | 0 |
| Monolayer | 48 | 281 | 63 | 9 | 1 |
| Slightly Dense | 1 | 21 | 65 | 16 | 1 |
| Dense | 0 | 0 | 13 | 30 | 12 |
| Very Dense | 0 | 0 | 1 | 11 | 38 |

Thresholds: [0.292, 0.443, 0.585, 0.711] -- **no PAVA merges** (all 5 buckets monotonically separable).

## Rouleaux composite (v2.2)

Dropped for sign instability: otsu_separability, lbp_entropy

| feature | weight | range (2nd-98th pct) |
|---|---|---|
| coverage | 0.263 | [0.07347, 0.4386] |
| saturation_score | 0.269 | [0.03368, 0.1781] |
| glcm_contrast | 0.098 | [26.59, 87.11] |
| edge_density_unmasked | 0.043 | [0.05926, 0.1811] |
| tile_glcm_cv | 0.088 | [0.06242, 0.3111] |
| tile_glcm_patchiness | 0.239 | [0.1245, 0.9339] |

**Cross-validation** (5-fold): per-fold rho = [0.711, 0.72, 0.745, 0.772, 0.739], mean=0.737. Out-of-fold exact-match=67.6%, off-by-one=93.8%.

Out-of-fold confusion matrix (rows=manual, cols=predicted):

| manual \ predicted | No Rouleaux | Slight Rouleaux | Some Rouleaux | Rouleaux | Heavy Rouleaux |
|---|---|---|---|---|---|
| No Rouleaux | 341 | 59 | 14 | 5 | 1 |
| Slight Rouleaux | 31 | 30 | 24 | 10 | 2 |
| Some Rouleaux | 2 | 16 | 20 | 5 | 3 |
| Rouleaux | 0 | 2 | 7 | 12 | 13 |
| Heavy Rouleaux | 0 | 0 | 2 | 18 | 44 |

Thresholds: [0.374, 0.451, 0.553, 0.667] -- **no PAVA merges** (all 5 buckets monotonically separable).

## Sparser-bucket focus (v2.2)

Sparser-density FOV counts by source dataset:

| dataset | sparser | total |
|---|---|---|
| initial-071626 | 0 | 13 |
| tanzania-073026 | 44 | 324 |
| tanzania-080526 | 6 | 324 |

Out-of-fold, v2.2 calls Sparser correctly on 45/50 (90.0%) of manually-labeled Sparser FOVs; 5/50 are mistaken for Monolayer, its only neighbor on the scale.

Sparser/Monolayer raw-score threshold: **0.292** (v2.2, n=50 Sparser FOVs) vs. **0.281** (v2.1, n=44 Sparser FOVs, single-slide) -- moved by +0.011 after pooling in the second slide.

No PAVA merges anywhere on the density axis at this pool size.

## Composite independence (v2.2)

Spearman rho between the two fitted composite scores: **0.972**, vs. the true manual density-vs-Rouleaux label correlation of **0.823**.

## Axis-separation check (v2.2)

| min |delta| | n | sign matches | match rate | binomial p | Spearman rho |
|---|---|---|---|---|---|
| 1 | 456 | 318 | 69.7% | 9.901e-18 | 0.307 |
| 2 | 16 | 13 | 81.2% | 0.01064 | -0.076 |

---

# v2.2-lb-optimized: the v2.2 fit on stride-16 LBP entropy

Refit on the same 661 FOVs as v2.2, from `features-v2.2-lb-optimized.csv`, with `lbp_entropy` computed on a stride-16 centre grid. This is a runtime change, not a modelling one: `compute_features` drops from 5.85s to ~1.08s per FOV. The stride was validated first (`data/results/lbp-runtime/README.md`) -- across all 661 FOVs it changes none of the 1322 bucket assignments under v2.2's own params.

## Density composite (v2.2-lb-optimized)

| feature | weight | range (2nd-98th pct) |
|---|---|---|
| coverage | 0.236 | [0.07347, 0.4386] |
| otsu_separability | 0.015 | [0.5166, 0.6124] |
| saturation_score | 0.247 | [0.03368, 0.1781] |
| lbp_entropy | 0.064 | [3.127, 4.203] |
| glcm_contrast | 0.088 | [26.59, 87.11] |
| edge_density_unmasked | 0.098 | [0.05926, 0.1811] |
| tile_glcm_cv | 0.082 | [0.06242, 0.3111] |
| tile_glcm_patchiness | 0.171 | [0.1245, 0.9339] |

**Cross-validation** (5-fold): per-fold rho = [0.789, 0.794, 0.789, 0.761, 0.782], mean=0.783. Out-of-fold exact-match=69.4%, off-by-one=98.0%.

Out-of-fold confusion matrix (rows=manual, cols=predicted):

| manual \ predicted | Sparser | Monolayer | Slightly Dense | Dense | Very Dense |
|---|---|---|---|---|---|
| Sparser | 45 | 5 | 0 | 0 | 0 |
| Monolayer | 48 | 281 | 63 | 9 | 1 |
| Slightly Dense | 1 | 21 | 65 | 16 | 1 |
| Dense | 0 | 0 | 13 | 30 | 12 |
| Very Dense | 0 | 0 | 1 | 11 | 38 |

Thresholds: [0.292, 0.442, 0.584, 0.71] -- **no PAVA merges** (all 5 buckets monotonically separable).

## Rouleaux composite (v2.2-lb-optimized)

Dropped for sign instability: otsu_separability, lbp_entropy

| feature | weight | range (2nd-98th pct) |
|---|---|---|
| coverage | 0.263 | [0.07347, 0.4386] |
| saturation_score | 0.269 | [0.03368, 0.1781] |
| glcm_contrast | 0.098 | [26.59, 87.11] |
| edge_density_unmasked | 0.043 | [0.05926, 0.1811] |
| tile_glcm_cv | 0.088 | [0.06242, 0.3111] |
| tile_glcm_patchiness | 0.239 | [0.1245, 0.9339] |

**Cross-validation** (5-fold): per-fold rho = [0.711, 0.72, 0.745, 0.772, 0.739], mean=0.737. Out-of-fold exact-match=67.6%, off-by-one=93.8%.

Out-of-fold confusion matrix (rows=manual, cols=predicted):

| manual \ predicted | No Rouleaux | Slight Rouleaux | Some Rouleaux | Rouleaux | Heavy Rouleaux |
|---|---|---|---|---|---|
| No Rouleaux | 341 | 59 | 14 | 5 | 1 |
| Slight Rouleaux | 31 | 30 | 24 | 10 | 2 |
| Some Rouleaux | 2 | 16 | 20 | 5 | 3 |
| Rouleaux | 0 | 2 | 7 | 12 | 13 |
| Heavy Rouleaux | 0 | 0 | 2 | 18 | 44 |

Thresholds: [0.374, 0.451, 0.553, 0.667] -- **no PAVA merges** (all 5 buckets monotonically separable).

## What moved vs. v2.2

LBP entropy is now computed on a stride-16 centre grid (0.07s/FOV vs. 4.80s). Everything else is identical: same 661 FOVs, same candidate pool, same ridge/PAVA procedure.

**Density**

| feature | weight (v2.2) | weight (lb-optimized) | delta |
|---|---|---|---|
| coverage | 0.2356 | 0.2358 | +0.00021 |
| otsu_separability | 0.0140 | 0.0145 | +0.00052 |
| saturation_score | 0.2465 | 0.2466 | +0.00006 |
| lbp_entropy | 0.0654 | 0.0641 | -0.00127 |
| glcm_contrast | 0.0874 | 0.0877 | +0.00027 |
| edge_density_unmasked | 0.0976 | 0.0978 | +0.00020 |
| tile_glcm_cv | 0.0822 | 0.0822 | +0.00002 |
| tile_glcm_patchiness | 0.1714 | 0.1714 | -0.00000 |

Thresholds: [0.2916, 0.4422, 0.5844, 0.7096] vs. v2.2 [0.2917, 0.4427, 0.5853, 0.7109] (max shift 0.00135).

**Rouleaux**

| feature | weight (v2.2) | weight (lb-optimized) | delta |
|---|---|---|---|
| coverage | 0.2633 | 0.2633 | +0.00000 |
| saturation_score | 0.2688 | 0.2688 | +0.00000 |
| glcm_contrast | 0.0976 | 0.0976 | +0.00000 |
| edge_density_unmasked | 0.0434 | 0.0434 | +0.00000 |
| tile_glcm_cv | 0.0884 | 0.0884 | +0.00000 |
| tile_glcm_patchiness | 0.2385 | 0.2385 | +0.00000 |

Thresholds: [0.3738, 0.4508, 0.5533, 0.6672] vs. v2.2 [0.3738, 0.4508, 0.5533, 0.6672] (max shift 0.00000).

## Composite independence (v2.2-lb-optimized)

Spearman rho between the two fitted composite scores: **0.972**, vs. the true manual density-vs-Rouleaux label correlation of **0.823**.

## Axis-separation check (v2.2-lb-optimized)

| min |delta| | n | sign matches | match rate | binomial p | Spearman rho |
|---|---|---|---|---|---|
| 1 | 456 | 318 | 69.7% | 9.901e-18 | 0.307 |
| 2 | 16 | 13 | 81.2% | 0.01064 | -0.076 |
