# ziba-test — v2.2 density/Rouleaux scores

24 FOVs from `data/raw/ziba-test/`, scored with the calibrated v2.2 fit
(`density_overlap_v2.2_params.json`, the full-resolution 661-FOV fit). Inference only —
nothing was recalibrated, and there are no manual labels for this set, so every number here
is the model's own output.

```bash
python scripts/ziba_test.py --workers 4
```

| file | what it is |
|---|---|
| `quality-grid-v2.2.png` | the 24 FOVs on the density × Rouleaux grid, coloured by test group |
| `features-v2.2.csv` | raw feature vector + both composite scores + combined score + buckets, per FOV |
| `combined-score-by-group-v2.2.png` | combined severity, one box per test group, the 12 +/- DAPI pairs connected |
| `ood-report-v2.2.csv` | per-FOV × per-feature position within the calibration normalization band |

## The dataset

A 6 × 4 protocol sweep, not a slide cohort: six methanol volumes (100 / 150 / 200 / 250 /
500 µL, plus a Coplin jar dip) × four test groups (RT vs. 37+Hum incubation, each imaged with
and without DAPI). One FOV per protocol cell. All 24 are 2800×2800 DPC-style monochrome PNGs.

**The design is paired, not four independent groups.** The DAPI and non-DAPI file at a given
volume + incubation are *the same slide* — two fields of one smear. So this is 12 slides each
measured twice, and the four groups describe two sets of six slides rather than four samples of
six. That is what turns the ± DAPI contrast into a reliability test rather than a comparison,
and it is the single most informative thing in this dataset (see the next section).

Both figures encode the 2×2 structure the same way rather than treating the four groups as
unrelated categories: hue = incubation (blue = RT, red = 37+Hum), lightness + marker shape =
DAPI, and each point is labelled with its methanol volume so a single FOV can be traced back to
its protocol cell.

## Headline: v2.2 does not agree with itself on the same slide

Because ± DAPI is one slide measured twice, the true density is identical within each pair, so
any difference between the two is measurement error. It is large:

| | density | Rouleaux |
|---|---|---|
| mean signed difference (DAPI − no DAPI) | **+0.185** | **+0.238** |
| mean absolute difference | **0.217** | **0.263** |
| pairs where DAPI scored higher | 10 / 12 | 10 / 12 |
| Wilcoxon signed-rank | p = 0.012 | p = 0.009 |
| range of paired differences | −0.114 … +0.545 | −0.086 … +0.610 |

On a [0, 1] scale whose five buckets are ~0.15–0.20 wide, a mean absolute disagreement of 0.217
is **more than one bucket of noise on the same smear**, and the worst pair (37+Hum 500 µL:
0.412 → 0.956 on density) spans over half the scale — Monolayer to Very Dense on one slide. The
disagreement is also *biased*, not merely noisy: 10 of 12 pairs move the same direction, which
field-to-field variation alone would not produce.

Two components are mixed here and this dataset cannot separate them — a systematic DAPI-channel
effect (bounded by the +0.185 signed mean) and genuine field-to-field variation within a smear
(one FOV per field, so unmeasured). For deciding whether to trust a score, the total is what
matters.

**The direct consequence:** the RT vs. 37+Hum difference below (+0.119) is *smaller than the
same-slide disagreement* (0.217), so it cannot be separated from measurement error and should
not be read as an effect of incubation condition.

## Results

The 24 FOVs land in 6 of the 25 grid cells, spanning Monolayer → Very Dense on density and
No Rouleaux → Heavy Rouleaux on Rouleaux. Nothing is called Sparser.

| bucket | density | Rouleaux |
|---|---|---|
| Sparser / No Rouleaux | 0 | 5 |
| Monolayer / Slight Rouleaux | 8 | 5 |
| Slightly Dense / Some Rouleaux | 7 | 5 |
| Dense / Rouleaux | 4 | 4 |
| Very Dense / Heavy Rouleaux | 5 | 5 |

Mean composite score by test group (n = 6 each):

| test group | density | Rouleaux | density range |
|---|---|---|---|
| 37+Hum | 0.426 | 0.389 | 0.372 – 0.500 |
| RT | 0.505 | 0.477 | 0.387 – 0.739 |
| 37+Hum DAPI | 0.571 | 0.568 | 0.404 – 0.956 |
| RT DAPI | 0.730 | 0.775 | 0.610 – 0.927 |

Two things separate the groups, and only the second is a candidate for a real effect:

- **DAPI vs. no DAPI:** mean density 0.651 vs. 0.466, a gap of **+0.185**. This is *not* a
  group difference — it is the same slides measured twice, so it is the measurement error
  quantified in the section above. Every one of the five Very Dense FOVs and all four Dense
  FOVs is from a DAPI group, which is a statement about the imaging condition, not the smears.
- **RT vs. 37+Hum:** mean density 0.618 vs. 0.499, a gap of **+0.119** (Mann–Whitney p = 0.046,
  and these *are* different slides). But +0.119 sits below the 0.217 same-slide disagreement, so
  the nominal significance is not meaningful — the noise floor is above the effect. 37+Hum is
  also by far the tightest group — its six FOVs span 0.372–0.500, versus 0.404–0.956 for
  37+Hum DAPI, which is itself a symptom of the same instability.

**Methanol volume shows no monotone effect.** Pooled across groups, density score vs. volume
is ρ = +0.23 (p = 0.34, n = 20, Coplin excluded as a dip rather than a pipetted volume). Only
37+Hum DAPI is monotone in volume (ρ = +1.00 across its five volumes); RT DAPI and 37+Hum
actually trend *negative* (ρ = −0.50, −0.60, both n.s.). The 500 µL column has the highest
mean (0.683 vs. 0.499–0.559 for the others), but that is carried by two FOVs — 37+Hum DAPI
500 µL (0.956) and RT 500 µL (0.739) — while 37+Hum 500 µL scores 0.412, near the bottom of
the whole set. At one FOV per protocol cell there is no way to separate a volume effect from
FOV-to-FOV variation within a slide.

### Combined severity by group

`combined-score-by-group-v2.2.png` collapses the two axes into one number per FOV — the mean of
the density and Rouleaux composites — and boxes it per test group. That collapse is only
legitimate because of the correlation noted below (r = 0.994): the two axes are close to the
same reading on this set, so their mean discards almost nothing.

| test group | mean | median | range | IQR width |
|---|---|---|---|---|
| 37+Hum | 0.408 | 0.396 | 0.346 – 0.481 | 0.09 |
| RT | 0.491 | 0.453 | 0.366 – 0.730 | 0.11 |
| 37+Hum DAPI | 0.570 | 0.511 | 0.384 – 0.977 | 0.16 |
| RT DAPI | 0.752 | 0.701 | 0.620 – 0.959 | 0.24 |

All six FOVs are plotted on top of each box, because at n = 6 the quartiles rest on one or two
observations each and the box alone would overstate how well-determined the spread is. The
points also carry the **connectors joining the 12 ± DAPI pairs** — each line links the two
measurements of one slide, so a steep line is v2.2 contradicting itself rather than a difference
between groups. Those connectors are the most load-bearing marks in the figure: the RT → RT DAPI
lines and the 37+Hum 500 µL line climb further than any gap between the boxes.

Three things are visible in the points that the boxes hide:

- **The boxes are not four independent samples.** The two blue boxes are the same six slides as
  each other, and so are the two red. Comparing box 1 against box 2 is a repeatability check;
  only blue-vs-red is a group comparison.

- **The spreads are very unequal.** 37+Hum is tight (0.346–0.481, everything Monolayer or
  Slightly Dense); 37+Hum DAPI spans nearly the whole scale (0.384–0.977) on the same six
  volumes. A single group summary is much more meaningful for 37+Hum than for either DAPI group.
- **Each of the three higher groups has one runaway FOV** — RT 500 µL (0.730), 37+Hum DAPI
  500 µL (0.977), RT DAPI Coplin (0.959) — sitting well clear of its group's box. With n = 6 a
  single outlier moves the group mean by ~0.05–0.08, which is a large fraction of the
  between-group gaps being compared.

**The two axes are not independent here.** Density score and Rouleaux score correlate at
Pearson r = 0.994 / Spearman ρ = 0.987 across the 24 FOVs — which is why every point sits on
the grid diagonal and the off-diagonal cells are empty. The two axes share `coverage`,
`saturation_score`, `glcm_contrast`, `edge_density_unmasked`, `tile_glcm_cv` and
`tile_glcm_patchiness` (all six of the Rouleaux axis's features are also density features), so
on a set this uniform they are close to reporting the same thing twice. Read the grid position
as one severity reading, not two.

## Caveats — read these before using the numbers

**1. Two features are clipped on all 24 FOVs, so ~10% of each score is a constant.**
This is the out-of-distribution check `scripts/combined/README.md` asks for on a new
preparation, and it does not come back clean:

| feature | calibration band (p2–p98) | observed on ziba-test | clipped |
|---|---|---|---|
| `otsu_separability` | 0.5166 – 0.6124 | 0.4453 – 0.5135 | **24/24 below p2** |
| `glcm_contrast` | 26.59 – 87.11 | 102.6 – 194.6 | **24/24 above p98** (max 2.23× the edge) |
| `saturation_score` | 0.0337 – 0.1781 | 0.0723 – 0.4318 | 4/24 above p98 |
| `tile_glcm_cv` | 0.0624 – 0.3111 | 0.1284 – 0.5253 | 4/24 above p98 |
| `coverage` | 0.0735 – 0.4386 | 0.1437 – 0.8334 | 3/24 above p98 |
| `tile_glcm_patchiness` | 0.1245 – 0.9339 | 0.1606 – 1.3162 | 3/24 above p98 |
| `lbp_entropy` | 3.1180 – 4.1961 | 3.2161 – 3.9903 | 0/24 |
| `edge_density_unmasked` | 0.0593 – 0.1811 | 0.1168 – 0.1772 | 0/24 |

`otsu_separability` and `glcm_contrast` are pinned at 0.0 and 1.0 respectively on *every* FOV,
so they contribute a fixed offset instead of information: 10.1% of the density weight and 9.8%
of the Rouleaux weight (`glcm_contrast` alone) does no discriminating work here. The
`glcm_contrast` pin is an inflating constant — it adds a flat +0.087 (density) / +0.098
(Rouleaux) to all 24 scores. Refitting the composite with those two features dropped and the
remaining weights renormalized moves **4/24 density labels and 7/24 Rouleaux labels** (e.g. RT
150 µL Slightly Dense → Monolayer, RT DAPI 200 µL Dense → Slightly Dense). So the bucket
assignments near a threshold are sensitive to a term that carries no signal on this dataset.

Every FOV being *uniformly* out of band on those two features points at a whole-dataset
imaging/preparation difference from the calibration pool (all Tanzania/initial-dataset FOVs),
not at 24 individually unusual smears.

**Should those two features be dropped and the weights renormalized? No — not at scoring
time.** Because both are pinned to a *constant* on all 24 FOVs, removing them and
renormalizing is an affine transform of the score: `dropped = (full − 0.0874) / 0.8986` on the
density axis. Confirmed empirically — Spearman *and* Pearson correlation between the full and
dropped scores is exactly 1.000000 on both axes. So it cannot improve discrimination by any
amount; the FOV ranking is bit-identical. All it does is slide every score down against bucket
thresholds that were themselves fit *with* those features included, which is where the 4/24 and
7/24 label changes come from. Those changes are a scale mismatch, not a correction — the
thresholds and the weights are one calibrated object and cannot be varied independently. It is
the same invariant `scripts/combined/_v2_common.py` documents for `lbp_step`: a composite fit on
one feature set has to be scored on that feature set.

The drop-and-renormalize was run here only as a *sensitivity probe* — to measure how much of the
bucket assignment rests on a term carrying no signal — and the answer (4 density, 7 Rouleaux
labels sit within that offset of a threshold) is a reason to distrust those buckets, not a
recipe for fixing them. The real fixes are to refit (see below) or to report the scores with
this flag attached.

**2. The empty-field gate is inert on this set.** It fires only when all four of
`otsu_separability`, `lbp_entropy`, `glcm_contrast`, `edge_density_unmasked` fall below their
p2 floors. Here `glcm_contrast` sits 4–7× *above* its 26.59 floor on every FOV, so the gate
cannot fire regardless of what the images contain. It reported 0/24, and that 0 carries no
information.

**3. The scores are uncalibrated for this preparation and unvalidated against a human.** With
no `data/labels/ziba-test/`, there is no accuracy number to quote — the figures show where v2.2
*puts* these FOVs, not whether it is right. Note that the reliability finding needs no labels:
a scorer that contradicts itself by a bucket on one slide is failing whatever the ground truth
turns out to be. The four-group ordering (37+Hum < RT < 37+Hum DAPI < RT DAPI) should not be
read as a severity ordering at all, since two of its three steps are the DAPI artifact.

**4. What the ± DAPI pairs do and don't establish.** They are the same slide, so within-pair
differences are measurement error — that part is settled. What they cannot separate is *why*:
the pair is two different fields (pixel correlation ≈ 0.03, not the same field re-imaged), and
all 24 images are DPC-style monochrome regardless of the suffix, so the shift could be the
fluorescence channel altering the image statistics v2.2 reads, the DAPI staining step altering
the smear itself, or ordinary within-slide field variation. The 10/12 directional bias argues
against field variation being the whole story, but bounding each component needs several fields
per slide (see next steps).

## Suggested next steps

**Do not use these scores for the protocol comparison as they stand.** The question the sweep was
built to answer — whether methanol volume or incubation condition changes smear quality — needs
measurement noise below the effect size, and here the noise (0.217 same-slide) is above the
largest candidate effect (0.119 RT vs. 37+Hum). Volume shows nothing at all.

Given that dropping the clipped features is a no-op on the ranking (caveat 1), the two things
that could actually change the answer are:

1. **Diagnose the DAPI shift feature by feature (~30 min, no labels needed).** The 12 pairs are a
   free test set with a known-zero true difference, so per-feature paired differences from
   `features-v2.2.csv` would localize the instability. Caveat 1 makes `glcm_contrast` and
   `otsu_separability` the first suspects: both are already 24/24 outside their calibration bands.
   This is the highest-value next step because it is cheap and could show the problem is a
   normalization-band mismatch for this preparation rather than anything about the smears.
2. **Manual labels on a subset (~20 min).** Even 8 of the 24, chosen to span the score range,
   turns caveat 3 into an accuracy number and shows whether the *ranking* is sound while only the
   thresholds are off — a much smaller fix than a refit. Reuses `scripts/nigeria_081226.py`'s
   manual-vs-model grid unchanged.

Then, if the sweep is to be re-run: **several FOVs per slide**, which is the only way to split the
DAPI-channel effect from within-slide field variation, and which would also give the volume
comparison enough resolution to detect an effect if one exists.

A second, independent check: `scripts/ai-first/score_new_slide.py::score_fov` (classical
watershed) produces a direct `n_cells` count from raw pixels, sharing no features with v2.2 and
needing no calibration, so correlating it against `density_score` would test the ranking without
any labels at all. Measured at 26 s/FOV on this image size, so ~3 minutes for all 24 at
`--workers 4`. Started but not completed here — no numbers from it are quoted above.
