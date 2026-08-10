# Diverse overexposed FOVs test (2026-08-07)

Full run of the overexposure-detection pipeline (`src/overexposure.py`) against
`data/labels/overexposure-diverse-080726.csv` -- 76 rows, hand-picked to be visually diverse
"Overexposed"-tagged FOVs spanning all three labeled countries (51 Liberia, 15 Tanzania, 10
Uganda), each additionally labeled with ground truth on whether a genuine fluorescent spot is
present (independent of the overexposure look). Every FOV was streamed directly from GCS (no
local disk cache) via a new `src/gcs_fov_multi.py` resolver that extends the existing
Liberia-only `src/gcs_fov.py` to Tanzania (`gs://tanzania_02032026`) and Uganda
(`gs://malaria-annotation-web`).

## Input labels

| sample_id | fov_id | annotator | country | tags | spot | notes |
|---|---|---|---|---|---|---|
| LB-D10-2025-12-29-150312-0171084-VFPCHC-2-4 | 153 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D10-2025-12-29-150312-0171084-VFPCHC-2-4 | 154 | A. Chen | Liberia | Slightly Dense, Rouleaux, Overexposed | yes |  |
| LB-D10-2025-12-30-083614-0250901VFPCHC-2-1 | 210 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D10-2025-12-30-083614-0250901VFPCHC-2-1 | 227 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D10-2025-12-30-084453-0250071VFPCHC-2-2 | 200 | A. Chen | Liberia | Rouleaux, Overexposed | yes |  |
| LB-D11-2025-12-17-115859-0250319D-thin-4-1 | 29 | A. Chen | Liberia | Overexposed | no | background |
| LB-D11-2025-12-19-111309-0211715-VFPCHC-3-1 | 277 | A. Chen | Liberia | Overexposed | yes | background |
| LB-D11-2025-12-19-131014-0241591-VFPCHC-3-2 | 278 | A. Chen | Liberia | Overexposed | yes | background |
| LB-D11-2025-12-19-134126-025073-VFPCHC-3-1 | 1 | A. Chen | Liberia | Overexposed | no | background |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 257 | A. Chen | Liberia | Sparser, Crenated, Overexposed, Unfocused | no | background |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 269 | A. Chen | Liberia | Sparser, Overexposed, Unfocused | no | background |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 274 | A. Chen | Liberia | Sparser, Crenated, Overexposed | no | background |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 279 | A. Chen | Liberia | Sparser, Crenated, Unfocused, Overexposed | no | background |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 289 | A. Chen | Liberia | Sparser, Crenated, Overexposed, Unfocused | no | background |
| LB-D3-2025-09-02-141940-25087110-D-Only-1-2 | 42 | A. Chen | Liberia | Overexposed | yes | background |
| LB-D3-2025-09-09-093425-250917463-D-Only-1-1 | 166 | A. Chen | Liberia | Crenated, Overexposed | yes |  |
| LB-D3-2025-09-27-121918-17217958-D-thin-4-4 | 262 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D3-2025-10-03-104211-250917371-D-thin-2-3 | 4 | A. Chen | Liberia | Overexposed, Unfocused | yes |  |
| LB-D3-2025-10-03-104643-250917465-D-thin-3-4 | 185 | A. Chen | Liberia | Overexposed | yes | green |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 1 | A. Chen | Liberia | Crenated, Overexposed | no | background |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 16 | A. Chen | Liberia | Crenated, Overexposed | no | background |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 17 | A. Chen | Liberia | Crenated, Overexposed | no | background |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 18 | A. Chen | Liberia | Crenated, Overexposed | no | background |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 19 | A. Chen | Liberia | Crenated, Overexposed | no | background |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 53 | A. Chen | Liberia | Crenated, Overexposed | no | background |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 114 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 125 | A. Chen | Liberia | Crenated, Overexposed | yes |  |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 126 | A. Chen | Liberia | Overexposed, Artifact | no | artifact |
| LB-D3-2025-10-03-125352-2402169466D-thin-2-1 | 3 | A. Chen | Liberia | Overexposed | yes | diffuse |
| LB-D3-2025-10-03-130859-250916865-D-thin-1-4 | 236 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D3-2025-10-22-131729-250917745-D-thin-2-3 | 134 | A. Chen | Liberia | Overexposed | yes | double |
| LB-D3-2025-10-22-132316-2411189646-D-thin-1-4 | 135 | A. Chen | Liberia | Overexposed | yes | diffuse |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | 122 | A. Chen | Liberia | Overexposed | yes | double |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | 238 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D3-2025-10-24-113736-250918214-D-thin-2-3 | 96 | A. Chen | Liberia | Overexposed | no* | relabeled by Emily 2026-08-07 (was yes) |
| LB-D3-2025-10-24-132012-25046898-D-thin-1-4 | 3 | A. Chen | Liberia | Overexposed | yes | diffuse |
| LB-D3-2025-10-24-132012-25046898-D-thin-1-4 | 305 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D3-2025-10-24-162727-230918080-D-thin-1-4 | 8 | A. Chen | Liberia | Overexposed | yes | diffuse |
| LB-D3-2025-10-25-105806-180951467-D-thin-1-1 | 270 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D3-2025-10-25-150947-250917467-D-thin-3-2 | 235 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D3-2025-10-27-123159-251123404-D-thin-4-1 | 48 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D3-2025-10-27-123159-251123404-D-thin-4-1 | 49 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D3-2025-10-27-124239-250916732-D-thin-1-3 | 301 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D3-2025-10-27-134711-250917368-D-thin-1-3 | 52 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D3-2025-10-27-144635-250918691-D-thin-2-2 | 57 | A. Chen | Liberia | Overexposed, Large | no | artifact |
| LB-D3-2025-10-27-144635-250918691-D-thin-2-2 | 243 | A. Chen | Liberia | Unfocused, Overexposed | no | background |
| LB-D3-2025-10-27-145205-250917002-D-thin-3-3 | 310 | A. Chen | Liberia | Overexposed, Artifact | no | artifact |
| LB-D3-2025-10-27-154305-250917412-D-thin-1-4 | 119 | A. Chen | Liberia | Overexposed | yes | diffuse |
| LB-D3-2025-10-27-155920-250713919-D-thin-3-3 | 169 | A. Chen | Liberia | Overexposed | no | background |
| LB-D3-2025-10-27-173317-250917493-D-thin-2-4 | 82 | A. Chen | Liberia | Overexposed | yes |  |
| LB-D5-2026-01-27-112616-0240052-VFPCHC-2-2 | 40 | A. Chen | Liberia | Overexposed | no | background |
| KIT-62500763 | 200 | A. Chen | Tanzania | Overexposed | yes | green |
| KIT-62501035 | 67 | A. Chen | Tanzania | Overexposed | yes |  |
| KIT-62501062 | 83 | A. Chen | Tanzania | Overexposed | no | artifact |
| KIT-62501081 | 141 | A. Chen | Tanzania | Overexposed | yes | double |
| KIT-62501087 | 271 | A. Chen | Tanzania | Overexposed | yes |  |
| KTR-72502946 | 54 | E. Chu | Tanzania | Slightly Dense, Some Rouleaux, Overexposed | yes |  |
| KTR-72502946 | 198 | E. Chu | Tanzania | Dense, Some Rouleaux, Overexposed | yes |  |
| NKR-72502319 | 119 | A. Chen | Tanzania | Overexposed | no | background |
| NKR-72502319 | 293 | A. Chen | Tanzania | Overexposed | yes |  |
| NKR-72502319 | 311 | A. Chen | Tanzania | Overexposed | yes |  |
| RUB-62501332 | 133 | A. Chen | Tanzania | Overexposed | yes |  |
| RUB-62501389 | 284 | Z. Ahamad | Tanzania | Overexposed | yes | double |
| RUB-62501518 | 315 | A. Chen | Tanzania | Overexposed | no | background |
| RUB-62501529 | 87 | A. Ma | Tanzania | Overexposed | no | background |
| RUB-72501756 | 315 | A. Chen | Tanzania | Overexposed | yes |  |
| PAT-070-3 | 34 | A. Chen | Uganda | Overexposed | yes | double |
| PAT-072-1 | 14 | A. Chen | Uganda | Overexposed, Artifact | no | artifact |
| PAT-072-1 | 94 | A. Chen | Uganda | Sparser, Some Rouleaux, Overexposed, Artifact | no | artifact |
| PAT-154-1 | 478 | A. Chen | Uganda | Overexposed | no | background |
| PBC-225_AM-1 | 30 | A. Ma | Uganda | Sparser, Overexposed | no | background |
| PBC-608-KH-1 | 171 | A. Chen | Uganda | Overexposed | yes |  |
| PBC-800-1 | 128 | A. Ma | Uganda | Sparser, Deep Dimples, Overexposed, Medium | no | background |
| PBC-800-1 | 732 | A. Ma | Uganda | Deep Dimples, Overexposed | no | background |
| PAT-103-2 | 441 | E. Chu | Uganda | Overexposed | no | background |
| PAT-112-2 | 124 | E. Chu | Uganda | Overexposed | no | background |

`notes` is only partially filled in (background/artifact/diffuse/double/green on some rows,
blank on the rest) -- per Emily, the remaining rows still need categorizing. The subset
breakdowns below use only the rows currently tagged `background`/`diffuse`/`double`, so they
under-cover each category and will change as `notes` gets filled in further.

`*` fov=96 (`LB-D3-2025-10-24-113736-250918214-D-thin-2-3`) was originally labeled
`spot=yes`; Emily flagged it as mislabeled on 2026-08-07 after reviewing its preview (see
"FN/FP examples" -- it had shown up there as a false negative missed by both variants, but
looking at the actual image the ground truth itself was wrong, not the detector). Corrected to
`spot=no` in both `data/labels/overexposure-diverse-080726.csv` and `results.csv`; all
confusion matrices, rates, and the FN/FP example set below reflect the corrected label.

## Method and what "predicted" means here

Full per-row detection was run via `scripts/run_overexposed_diverse_test.py`, which streams
each FOV from GCS and calls the same production code path as `scripts/score_labels.py`:
`detect_overexposure()` (ratio gate -> anisotropy-fft fiber-debris demotion), plus the
advisory-only diffuse-fov step (`diffuse_candidate` -> neighbor-trend check ->
`diffuse_halo_flag`, fetching the 2 preceding fov_ids for context, same as
`scripts/scan_diffuse_candidates.py`).

**Update (2026-08-07, same day): `diffuse_candidate()` was changed** after this doc's first
pass found the diffuse-fov fold-in created 7 new false positives (see "Discussion" below for
the original numbers). It now requires `DIFFUSE_RATIO_MIN <= contrast_ratio < RATIO_THRESHOLD`
(previously just `not present`, any reason) -- see `src/overexposure.py`'s module docstring,
"Ratio floor for diffuse candidates", for the calibration. All numbers in this doc reflect the
fixed version; the "Which FOVs flip" and "Discussion" sections below narrate both the original
finding and the fix.

**Second update (2026-08-07, later the same day): the anisotropy fiber-debris filter itself was
fixed.** `KIT-62501087` (a real halo wrongly demoted by the anisotropy check because it's
clipped into a quarter-circle by the frame corner) was previously being rescued only by
accident, through the diffuse-fov fold-in -- and the `DIFFUSE_RATIO_MIN` fix above deliberately
excluded that accidental rescue. Emily asked for the underlying anisotropy misfire to be fixed
directly. A brainstorm (run through the Opus model, validated against real GCS-streamed images
rather than left as theory) found that naive corner-clip detection can't work -- `KIT-62501087`
is statistically indistinguishable from labeled fiber/debris cases on every corner-contact
metric -- but a rescue-only second opinion using two new signals (`radial_rho`, `r2_over_r1`)
does. See `src/overexposure.py`'s module docstring, "Corner-clipping rescue," and
`scripts/validate_corner_clip_fix.py` for the full validation. `KIT-62501087` is now a direct
`present_base=True` -- it no longer needs (or gets) any diffuse-fov involvement at all.

**Important framing note (corrected 2026-08-07 -- an earlier draft of this doc had this
backwards).** In this dataset, "the fluorescent spot" and "the overexposed halo artifact" are
the same thing -- `spot` is ground truth on whether the overexposure artifact itself is
genuinely present, not a separate real-signal-vs-artifact distinction. So the predicted label
is `present` directly, with no inversion:

```
predicted_spot_present = present
```

A false negative is a real halo the ratio/anisotropy gate missed entirely (most importantly, a
faint/diffuse halo below `RATIO_THRESHOLD` -- exactly the case the diffuse-fov step was built
to catch). A false positive is an ordinary FOV -- elevated background from many puncta,
debris/hair, etc. -- that tripped the ratio gate without an actual halo present.

**"Folded in" vs. not.** `present_base` is exactly what production code returns today (diffuse
fields computed but never gating). `present_folded = present_base OR diffuse_halo_flag` --
i.e., what `present` (and therefore predicted spot) would become if the diffuse-fov step's
flag *were* wired into the decision, which it currently is not (see
`fluorescence/README.md`'s "Diffuse-halo signal" section for why).

## Per-FOV runtime

Streamed live from GCS, one FOV at a time, no disk cache. `neighbor_fetch_s` is only nonzero
for rows that pass `diffuse_candidate()` (`DIFFUSE_RATIO_MIN <= contrast_ratio <
RATIO_THRESHOLD` and `diffuse_radius >= DIFFUSE_RADIUS_MIN`); it covers fetching + detecting
on up to 2 preceding fov_ids for the neighbor-trend check alone. `KIT-62501087` no longer
appears here at all -- with the corner-clipping rescue in place it's a direct present_base=True
and never reaches `diffuse_candidate()`.

| sample_id | fov_id | country | gcs_fetch_s | initial_test_s | anisotropy_s | diffuse_fov_s | neighbor_fetch_s |
|---|---|---|---|---|---|---|---|
| LB-D10-2025-12-29-150312-0171084-VFPCHC-2-4 | 153 | Liberia | 13.1255 | 0.065 | 0.0239 | 0.0029 | 0.0 |
| LB-D10-2025-12-29-150312-0171084-VFPCHC-2-4 | 154 | Liberia | 0.8479 | 0.0581 | 0.0268 | 0.0023 | 0.0 |
| LB-D10-2025-12-30-083614-0250901VFPCHC-2-1 | 210 | Liberia | 0.973 | 0.0618 | 0.0256 | 0.0029 | 0.0 |
| LB-D10-2025-12-30-083614-0250901VFPCHC-2-1 | 227 | Liberia | 1.049 | 0.0662 | 0.028 | 0.0024 | 0.0 |
| LB-D10-2025-12-30-084453-0250071VFPCHC-2-2 | 200 | Liberia | 1.113 | 0.0546 | 0.036 | 0.0025 | 0.0 |
| LB-D11-2025-12-17-115859-0250319D-thin-4-1 | 29 | Liberia | 1.3171 | 0.0501 | 0.0 | 0.0023 | 0.0 |
| LB-D11-2025-12-19-111309-0211715-VFPCHC-3-1 | 277 | Liberia | 1.356 | 0.0474 | 0.0 | 0.0021 | 2.3441 |
| LB-D11-2025-12-19-131014-0241591-VFPCHC-3-2 | 278 | Liberia | 1.0767 | 0.0577 | 0.0 | 0.002 | 2.1173 |
| LB-D11-2025-12-19-134126-025073-VFPCHC-3-1 | 1 | Liberia | 1.1154 | 0.0539 | 0.0 | 0.0024 | 0.0 |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 257 | Liberia | 0.8706 | 0.0589 | 0.0213 | 0.0024 | 0.0 |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 269 | Liberia | 0.9642 | 0.0532 | 0.0 | 0.0019 | 0.0 |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 274 | Liberia | 0.7573 | 0.0633 | 0.0086 | 0.0022 | 0.0 |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 279 | Liberia | 0.8522 | 0.0559 | 0.0 | 0.0017 | 2.3384 |
| LB-D3-2025-08-30-103102-250876706-D-thin-4 | 289 | Liberia | 0.9469 | 0.0568 | 0.0503 | 0.0027 | 0.0 |
| LB-D3-2025-09-02-141940-25087110-D-Only-1-2 | 42 | Liberia | 1.2035 | 0.0688 | 0.0252 | 0.0027 | 0.0 |
| LB-D3-2025-09-09-093425-250917463-D-Only-1-1 | 166 | Liberia | 0.9554 | 0.0751 | 0.0356 | 0.0033 | 0.0 |
| LB-D3-2025-09-27-121918-17217958-D-thin-4-4 | 262 | Liberia | 1.1603 | 0.0684 | 0.0445 | 0.0033 | 0.0 |
| LB-D3-2025-10-03-104211-250917371-D-thin-2-3 | 4 | Liberia | 1.1348 | 0.0645 | 0.0429 | 0.0032 | 0.0 |
| LB-D3-2025-10-03-104643-250917465-D-thin-3-4 | 185 | Liberia | 0.9834 | 0.0602 | 0.0175 | 0.0018 | 0.0 |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 1 | Liberia | 0.9949 | 0.058 | 0.0 | 0.0018 | 0.0 |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 16 | Liberia | 0.9921 | 0.0701 | 0.0 | 0.0025 | 0.0 |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 17 | Liberia | 1.0786 | 0.0562 | 0.0 | 0.0018 | 0.0 |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 18 | Liberia | 0.938 | 0.0569 | 0.0 | 0.0017 | 0.0 |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 19 | Liberia | 0.9886 | 0.0499 | 0.0 | 0.0018 | 0.0 |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 53 | Liberia | 1.0883 | 0.0612 | 0.0 | 0.0019 | 0.0 |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 114 | Liberia | 1.0496 | 0.0722 | 0.0586 | 0.0043 | 0.0 |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 125 | Liberia | 4.6512 | 0.0607 | 0.0083 | 0.002 | 0.0 |
| LB-D3-2025-10-03-124025-2404175445D-thin-2-3 | 126 | Liberia | 1.205 | 0.0626 | 0.0123 | 0.0028 | 0.0 |
| LB-D3-2025-10-03-125352-2402169466D-thin-2-1 | 3 | Liberia | 1.2026 | 0.0664 | 0.0181 | 0.0022 | 0.0 |
| LB-D3-2025-10-03-130859-250916865-D-thin-1-4 | 236 | Liberia | 1.207 | 0.0665 | 0.0339 | 0.003 | 0.0 |
| LB-D3-2025-10-22-131729-250917745-D-thin-2-3 | 134 | Liberia | 1.2211 | 0.0767 | 0.0748 | 0.0033 | 0.0 |
| LB-D3-2025-10-22-132316-2411189646-D-thin-1-4 | 135 | Liberia | 1.2628 | 0.0706 | 0.0 | 0.0024 | 2.2786 |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | 122 | Liberia | 0.9773 | 0.0583 | 0.0 | 0.0023 | 2.5157 |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | 238 | Liberia | 1.0603 | 0.0587 | 0.0225 | 0.0022 | 0.0 |
| LB-D3-2025-10-24-113736-250918214-D-thin-2-3 | 96 | Liberia | 1.392 | 0.0499 | 0.0 | 0.0011 | 0.0 |
| LB-D3-2025-10-24-132012-25046898-D-thin-1-4 | 3 | Liberia | 1.2375 | 0.0556 | 0.0 | 0.002 | 2.2518 |
| LB-D3-2025-10-24-132012-25046898-D-thin-1-4 | 305 | Liberia | 1.215 | 0.0511 | 0.0205 | 0.0023 | 0.0 |
| LB-D3-2025-10-24-162727-230918080-D-thin-1-4 | 8 | Liberia | 1.1332 | 0.134 | 0.0 | 0.0018 | 0.0 |
| LB-D3-2025-10-25-105806-180951467-D-thin-1-1 | 270 | Liberia | 1.3857 | 0.0551 | 0.0166 | 0.0032 | 0.0 |
| LB-D3-2025-10-25-150947-250917467-D-thin-3-2 | 235 | Liberia | 1.291 | 0.0584 | 0.0277 | 0.0027 | 0.0 |
| LB-D3-2025-10-27-123159-251123404-D-thin-4-1 | 48 | Liberia | 1.2194 | 0.0535 | 0.0203 | 0.0024 | 0.0 |
| LB-D3-2025-10-27-123159-251123404-D-thin-4-1 | 49 | Liberia | 1.2125 | 0.0547 | 0.0099 | 0.0022 | 0.0 |
| LB-D3-2025-10-27-124239-250916732-D-thin-1-3 | 301 | Liberia | 1.3246 | 0.0513 | 0.0426 | 0.0025 | 0.0 |
| LB-D3-2025-10-27-134711-250917368-D-thin-1-3 | 52 | Liberia | 1.1712 | 0.058 | 0.0341 | 0.0026 | 0.0 |
| LB-D3-2025-10-27-144635-250918691-D-thin-2-2 | 57 | Liberia | 1.0856 | 0.0523 | 0.0074 | 0.002 | 0.0 |
| LB-D3-2025-10-27-144635-250918691-D-thin-2-2 | 243 | Liberia | 1.2289 | 0.057 | 0.0 | 0.0012 | 0.0 |
| LB-D3-2025-10-27-145205-250917002-D-thin-3-3 | 310 | Liberia | 1.1561 | 0.0538 | 0.0033 | 0.0016 | 0.0 |
| LB-D3-2025-10-27-154305-250917412-D-thin-1-4 | 119 | Liberia | 1.1436 | 0.0537 | 0.0 | 0.0018 | 2.3977 |
| LB-D3-2025-10-27-155920-250713919-D-thin-3-3 | 169 | Liberia | 1.0672 | 0.0539 | 0.0 | 0.0013 | 0.0 |
| LB-D3-2025-10-27-173317-250917493-D-thin-2-4 | 82 | Liberia | 1.1778 | 0.0572 | 0.0134 | 0.0025 | 0.0 |
| LB-D5-2026-01-27-112616-0240052-VFPCHC-2-2 | 40 | Liberia | 1.3291 | 0.0536 | 0.0 | 0.002 | 0.0 |
| KIT-62500763 | 200 | Tanzania | 25.9773 | 0.0528 | 0.0349 | 0.0029 | 0.0 |
| KIT-62501035 | 67 | Tanzania | 0.8394 | 0.0519 | 0.0556 | 0.0033 | 0.0 |
| KIT-62501062 | 83 | Tanzania | 0.7877 | 0.0574 | 0.0 | 0.0018 | 0.0 |
| KIT-62501081 | 141 | Tanzania | 0.7537 | 0.0519 | 0.0225 | 0.003 | 0.0 |
| KIT-62501087 | 271 | Tanzania | 0.7146 | 0.0462 | 0.0304 | 0.0027 | 0.0 |
| KTR-72502946 | 54 | Tanzania | 0.9242 | 0.0339 | 0.0412 | 0.0028 | 0.0 |
| KTR-72502946 | 198 | Tanzania | 1.0375 | 0.0423 | 0.0552 | 0.0039 | 0.0 |
| NKR-72502319 | 119 | Tanzania | 0.8494 | 0.0323 | 0.0 | 0.0011 | 0.0 |
| NKR-72502319 | 293 | Tanzania | 0.9312 | 0.0413 | 0.0241 | 0.0042 | 0.0 |
| NKR-72502319 | 311 | Tanzania | 0.9676 | 0.0439 | 0.0184 | 0.0027 | 0.0 |
| RUB-62501332 | 133 | Tanzania | 0.8842 | 0.0423 | 0.0276 | 0.0022 | 0.0 |
| RUB-62501389 | 284 | Tanzania | 1.0072 | 0.0377 | 0.0262 | 0.0025 | 0.0 |
| RUB-62501518 | 315 | Tanzania | 0.9176 | 0.0343 | 0.0 | 0.0009 | 0.0 |
| RUB-62501529 | 87 | Tanzania | 0.861 | 0.043 | 0.0 | 0.0032 | 0.0 |
| RUB-72501756 | 315 | Tanzania | 0.8095 | 0.0423 | 0.0213 | 0.0021 | 0.0 |
| PAT-070-3 | 34 | Uganda | 0.9823 | 0.0388 | 0.0498 | 0.0027 | 0.0 |
| PAT-072-1 | 14 | Uganda | 0.8232 | 0.0411 | 0.0095 | 0.0022 | 0.0 |
| PAT-072-1 | 94 | Uganda | 0.8968 | 0.0363 | 0.0357 | 0.0028 | 0.0 |
| PAT-154-1 | 478 | Uganda | 1.0209 | 0.0415 | 0.0 | 0.001 | 0.0 |
| PBC-225_AM-1 | 30 | Uganda | 0.9194 | 0.0431 | 0.0 | 0.0009 | 0.0 |
| PBC-608-KH-1 | 171 | Uganda | 0.8432 | 0.0402 | 0.0363 | 0.0027 | 0.0 |
| PBC-800-1 | 128 | Uganda | 0.8606 | 0.0385 | 0.0 | 0.002 | 0.0 |
| PBC-800-1 | 732 | Uganda | 0.9539 | 0.0564 | 0.0 | 0.0054 | 0.0 |
| PAT-103-2 | 441 | Uganda | 1.0324 | 0.046 | 0.0 | 0.0012 | 0.0 |
| PAT-112-2 | 124 | Uganda | 0.8994 | 0.0414 | 0.0 | 0.0015 | 0.0 |

**Summary:**

| stage | min (s) | median (s) | max (s) | total (s) |
|---|---|---|---|---|
| gcs_fetch_s | 0.7146 | 1.0433 | 25.9773 | 119.9842 |
| time_initial_test_s | 0.0323 | 0.0547 | 0.1340 | 4.1648 |
| time_anisotropy_s | 0.0000 | 0.0129 | 0.0748 | 1.2992 |
| time_diffuse_fov_s | 0.0009 | 0.0023 | 0.0054 | 0.1799 |
| neighbor_fetch_s | 0.0000 | 0.0000 | 2.5157 | 16.2436 |

| country | n | median gcs_fetch_s | mean gcs_fetch_s |
|---|---|---|---|
| Liberia | 51 | 1.1348 | 1.4214 |
| Tanzania | 15 | 0.8842 | 2.5508 |
| Uganda | 10 | 0.9094 | 0.9232 |

Excluding the very first row of the run (a one-time GCS client cold-start cost), `gcs_fetch_s` ranges 0.71-25.98s with a median of 1.04s. **Neither `gcs_fov.py` (Liberia) nor `gcs_fov_multi.py` (Tanzania/Uganda) caches slide/box lookups across calls** -- every single fetch re-lists the Liberia `_Blue` folder and re-reads its `Scan.txt`, or (for Tanzania) re-checks up to 5 `TZ2025-Box<N>` prefixes, even for FOVs from the same scan/sample already resolved earlier in this same run. This isn't specific to this test -- it's how `scripts/score_labels.py` and `scripts/fetch_reference_images.py` already behave -- but it means per-FOV GCS time here is *not* representative of what a full-scan batch run would cost per FOV if slide/box resolution were cached once per sample_id; that's a real optimization opportunity if this pipeline is ever run at full-slide scale (see Recommendations). `time_initial_test_s`/`time_anisotropy_s`/`time_diffuse_fov_s` are all local CPU work (downsample/blur/threshold/FFT, now including the radial_rho/r2_over_r1 rescue check when anisotropy triggers) and dominated entirely by network time.

## Results: confusion matrices

Ground truth = `spot` column (is the halo artifact genuinely present). Predicted = `present`
directly (see "Method" above). All 4 matrices below are repeated for both variants.

### Diffuse-fov step NOT folded in (production behavior today, with the corner-clipping rescue fix)

**all** (n=76)

| | Predicted: spot | Predicted: no spot |
|---|---|---|
| Truth: spot | TP=37 | FN=7 |
| Truth: no spot | FP=5 | TN=27 |

FN rate: 7/44 (15.9%) -- FP rate: 5/32 (15.6%) -- **`KIT-62501087` moved from FN to TP here, directly, with no diffuse-fov involvement**

**background** (n=28)

| | Predicted: spot | Predicted: no spot |
|---|---|---|
| Truth: spot | TP=1 | FN=2 |
| Truth: no spot | FP=3 | TN=22 |

FN rate: 2/3 (66.7%) -- FP rate: 3/25 (12.0%)

**diffuse** (n=5)

| | Predicted: spot | Predicted: no spot |
|---|---|---|
| Truth: spot | TP=1 | FN=4 |
| Truth: no spot | FP=0 | TN=0 |

FN rate: 4/5 (80.0%) -- FP rate: n/a (no spot-negative rows in this subset)

**double** (n=5)

| | Predicted: spot | Predicted: no spot |
|---|---|---|
| Truth: spot | TP=4 | FN=1 |
| Truth: no spot | FP=0 | TN=0 |

FN rate: 1/5 (20.0%) -- FP rate: n/a (no spot-negative rows in this subset)

### Diffuse-fov step folded in, with the DIFFUSE_RATIO_MIN fix (`present_folded = present_base OR diffuse_halo_flag`)

**all** (n=76)

| | Predicted: spot | Predicted: no spot |
|---|---|---|
| Truth: spot | TP=43 | FN=1 |
| Truth: no spot | FP=5 | TN=27 |

FN rate: 1/44 (2.3%) -- FP rate: 5/32 (15.6%) -- **identical to the baseline FP rate: zero net new false positives**

**background** (n=28)

| | Predicted: spot | Predicted: no spot |
|---|---|---|
| Truth: spot | TP=3 | FN=0 |
| Truth: no spot | FP=3 | TN=22 |

FN rate: 0/3 (0.0%) -- FP rate: 3/25 (12.0%) -- **identical to baseline**

**diffuse** (n=5)

| | Predicted: spot | Predicted: no spot |
|---|---|---|
| Truth: spot | TP=4 | FN=1 |
| Truth: no spot | FP=0 | TN=0 |

FN rate: 1/5 (20.0%) -- FP rate: n/a (no spot-negative rows in this subset)

**double** (n=5)

| | Predicted: spot | Predicted: no spot |
|---|---|---|
| Truth: spot | TP=5 | FN=0 |
| Truth: no spot | FP=0 | TN=0 |

FN rate: 0/5 (0.0%) -- FP rate: n/a (no spot-negative rows in this subset)

## Discussion

**This section originally reported the diffuse-fov fold-in as a recall/specificity tradeoff
that traded a lot of false positives for a little more recall. Emily asked for a fix; the fix
is now in `src/overexposure.py` and the numbers below are what shipped, not the original
finding.** The "before" story, kept for context: 17 of 76 rows had `present_base=False` with a
large enough diffuse footprint to trigger the neighbor-trend check; 14 flipped from
`present=False` to `present=True` under fold-in -- 7 correct rescues of real halos the ratio
gate missed, but 7 new false positives, 6 of them `background`-tagged (ordinary elevated
illumination from puncta/staining, no real halo) and 1 (`fov126`) a fiber/debris artifact the
anisotropy filter had already correctly demoted. Folding in dropped the overall FN rate from
18.2% to 2.3% but more than doubled the FP rate, 15.6% to 37.5% -- and the `background` subset
alone went from FP 12.0% to 36.0%.

**The fix:** `diffuse_candidate()` now also requires `contrast_ratio >= DIFFUSE_RATIO_MIN
(2.30)` and, separately, requires a genuine ratio-gate miss (`contrast_ratio < RATIO_THRESHOLD`)
rather than accepting `present=False` for any reason. See `src/overexposure.py`'s module
docstring, "Ratio floor for diffuse candidates", for the full calibration writeup. Two design
alternatives were considered and rejected before this one: a new patch-grid illumination-
uniformity metric (tested against the real images and found to correlate at Spearman rho=0.95
with `contrast_ratio` -- a noisier restatement of an existing field, not a new signal), and a
ratio floor without the ratio-gate-miss requirement (keeps the `KIT-62501087` rescue below, but
leaves `fov126` unfixed). Emily chose the stricter version.

**Result with the `DIFFUSE_RATIO_MIN` fix alone (superseded below): zero net new false
positives, most of the recall gain kept.** FP stayed at 5/32 (15.6%) -- byte-identical to
baseline, because the two anisotropy-mechanism rows (`fov126`, `KIT-62501087`) and all 6
fake-`background` rows were excluded from `diffuse_candidate()` entirely (ratios 1.36-2.21, all
below `DIFFUSE_RATIO_MIN`, or 13-14, both above `RATIO_THRESHOLD` and reached `present=False`
only via anisotropy). FN improved from 18.2% to 4.5% (8 missed real halos -> 2) -- slightly
less improvement than the original, unfixed fold-in's 2.3%, because this fix also excluded
`KIT-62501087`'s rescue (an accident of the anisotropy filter misfiring on a real halo, not the
diffuse-fov step catching a faint one). All 5 non-`KIT-62501087` real-halo rescues survived:
`diffuse` subset recall 1/5 -> 4/5, `double` 4/5 -> 5/5.

**`fov279` is the one residual risk this fix doesn't structurally close.** It's `background`-
tagged, no real halo, `contrast_ratio=2.632` -- inside the `[2.30, 3.0)` candidate band -- and
is only excluded because `matches_neighbor_trend` happens to catch it (see
`src/overexposure.py`'s docstring). The ratio floor narrows how often that check has to do the
work; it doesn't replace it. Worth watching if `notes` labeling surfaces more cases like it.

**Third pass: the anisotropy filter's corner-clipping misfire was fixed directly, and
`KIT-62501087` no longer needs the diffuse-fov step at all.** Emily asked for the underlying
mechanism to be fixed rather than left as an accepted tradeoff. A brainstorm run through the
Opus model, validated against real GCS-streamed images (see
`scripts/validate_corner_clip_fix.py` and `src/overexposure.py`'s "Corner-clipping rescue"
docstring section), found that detecting corner-clipping directly can't work --
`KIT-62501087` is statistically indistinguishable from labeled fiber/debris cases on every
corner-contact metric tried -- but a rescue-only second opinion does: when anisotropy triggers
a demotion, check whether the whole frame's illumination correlates with distance from the
candidate's centroid (`radial_rho`, high for a halo's global radial field, low for a fiber's
local one) *and* whether the FFT's angular energy is a broad plateau rather than a narrow spike
(`r2_over_r1`, low for a clipped arc, high for a fiber). Measured 0 wrong rescues across all 6
labeled fiber/artifact cases plus their synthetic corner-clipped crops; rescues
`KIT-62501087` directly. `present_base` is now `True` for it -- it doesn't touch
`diffuse_candidate()` at all anymore.

**Current numbers, with both fixes in place:** baseline FN 18.2% -> **15.9%** (8 -> 7 missed
real halos; `KIT-62501087` moved to a direct true positive), FP unchanged at 15.6%. Folded-in
FN drops further to **2.3%** (1 missed -- `fov8`, still below any workable ratio floor at
1.98), FP still 15.6%, identical to baseline. Every other subset (`background`/`diffuse`/
`double`) is unchanged from the `DIFFUSE_RATIO_MIN`-only numbers above, since `KIT-62501087`
carries no `notes` tag.

**Baseline performance is already reasonably good** (FN 15.9%, FP 15.6% overall) for a
detector whose ratio/anisotropy design predates this specific labeled test set. `diffuse` is
the weakest baseline subset (FN 80.0%) precisely because it's defined as the faint/sub-ratio
population the diffuse-fov step targets -- which is exactly why folding it in (with the fix)
helps there so much, at no false-positive cost.

## Which FOVs flip if diffuse-fov is folded in

With the `DIFFUSE_RATIO_MIN` fix in place, only **6** rows have `present_base != present_folded`
-- down from the original 14 -- and every one of them is a correct rescue of a real halo the
ratio gate missed. Zero false positives are introduced by folding in.

| sample_id | fov_id | country | spot_truth | notes | contrast_ratio | diffuse_radius | outcome | preview |
|---|---|---|---|---|---|---|---|---|
| LB-D11-2025-12-19-111309-0211715-VFPCHC-3-1 | 277 | Liberia | yes | background | 2.75 | 176.2 | rescued (was FN) | [link](previews/LB-D11-2025-12-19-111309-0211715-VFPCHC-3-1__fov277__preview.png) |
| LB-D11-2025-12-19-131014-0241591-VFPCHC-3-2 | 278 | Liberia | yes | background | 2.43 | 161.9 | rescued (was FN) | [link](previews/LB-D11-2025-12-19-131014-0241591-VFPCHC-3-2__fov278__preview.png) |
| LB-D3-2025-10-22-132316-2411189646-D-thin-1-4 | 135 | Liberia | yes | diffuse | 2.59 | 155.7 | rescued (was FN) | [link](previews/LB-D3-2025-10-22-132316-2411189646-D-thin-1-4__fov135__preview.png) |
| LB-D3-2025-10-22-140622-250917738-D-thin-1-1 | 122 | Liberia | yes | double | 2.75 | 205.7 | rescued (was FN) | [link](previews/LB-D3-2025-10-22-140622-250917738-D-thin-1-1__fov122__preview.png) |
| LB-D3-2025-10-24-132012-25046898-D-thin-1-4 | 3 | Liberia | yes | diffuse | 2.45 | 129.4 | rescued (was FN) | [link](previews/LB-D3-2025-10-24-132012-25046898-D-thin-1-4__fov3__preview.png) |
| LB-D3-2025-10-27-154305-250917412-D-thin-1-4 | 119 | Liberia | yes | diffuse | 2.91 | 84.9 | rescued (was FN) | [link](previews/LB-D3-2025-10-27-154305-250917412-D-thin-1-4__fov119__preview.png) |

### The 8 rows that used to flip, and why they're excluded now

For the audit trail: before the fix, these 8 rows also flipped `present_base=False` ->
`present_folded=True`. All 8 are now correctly excluded from `diffuse_candidate()` before ever
reaching the neighbor-trend check.

**6 fake-`background` false positives, excluded by the new `DIFFUSE_RATIO_MIN=2.30` floor**
(all `contrast_ratio` 1.36-2.21, below the floor): `LB-D11-...-134126...` fov=1 (1.36),
`LB-D3-...-thin-4` fov=269 (2.15), `LB-D3-...-2404175445-2-3` fov=1 (1.92)/fov=19 (2.21)/fov=53
(2.05), `PBC-800-1` fov=732 (2.13). These are ordinary elevated-background FOVs, no real halo
-- the population this fix targets.

**`fov126` (`LB-D3-...-2404175445-2-3`), excluded by the ratio-gate-miss requirement, not the
floor.** Its `contrast_ratio=13.39` clears `RATIO_THRESHOLD=3.0` easily -- it was never a
ratio-gate miss. `results.csv` shows `anisotropy=0.5588`, above `ANISOTROPY_THRESHOLD=0.35`, so
this candidate was correctly demoted to `present=False` by the fiber/hair-debris check
(consistent with its `artifact` note). `diffuse_candidate()`'s old gate (`not present`, any
reason) let it through anyway; requiring `contrast_ratio < RATIO_THRESHOLD` now excludes it
structurally, since it's mechanically impossible to reach `present=False` via anisotropy while
also having `contrast_ratio < RATIO_THRESHOLD` (anisotropy is only evaluated when the ratio
gate already said yes).

**`KIT-62501087` fov=271 -- initially excluded the same way, now fixed directly (see
Discussion's "Third pass").** This was the mirror case: a real halo (`spot_truth=yes`)
mis-demoted by the anisotropy filter (`contrast_ratio=14.27`, `anisotropy=0.4602`), not a faint
halo the ratio gate missed. Its preview is a clean, sharply-defined, corner-clipped circular
halo -- the corner-clipping was confirmed (not just plausible) to bias the FFT-based anisotropy
measurement, via the validation in `scripts/validate_corner_clip_fix.py`. Rather than leave
this as an accepted tradeoff, the anisotropy filter itself now carries a rescue-only second
opinion (`radial_rho`/`r2_over_r1`, see `src/overexposure.py`'s "Corner-clipping rescue"
docstring section) that catches this exact case without re-admitting any labeled fiber/debris
example. `KIT-62501087` is now `present_base=True` directly -- it no longer reaches
`diffuse_candidate()` at all, so it's no longer part of "which FOVs flip" in any sense.

## FN/FP examples

Annotated previews for all 12 rows that are a false negative or false positive in either
variant, with both fixes applied, grouped into the same 4 buckets used throughout this doc. Red
outline = `present` (this variant's detector call fired); green = did not fire. Caption lines
show truth/notes, both variants' `present`, contrast ratio, and diffuse radius. (Down from 20
before the `DIFFUSE_RATIO_MIN` fix -- bucket D, "new false positive introduced by folding in,"
is now empty; `KIT-62501087` no longer appears anywhere in this section at all, since it's now
a direct true positive. See "Which FOVs flip" above for the full history.)

### A -- rescued by folding in (spot_truth=yes, missed at baseline, caught after fold-in) (n=6)

![LB-D11-2025-12-19-111309-0211715-VFPCHC-3-1 fov=277 (Liberia) -- truth=yes, notes=background, ratio=2.75](previews/LB-D11-2025-12-19-111309-0211715-VFPCHC-3-1__fov277__preview.png)
*LB-D11-2025-12-19-111309-0211715-VFPCHC-3-1 fov=277 (Liberia) -- truth=yes, notes=background, ratio=2.75*

![LB-D11-2025-12-19-131014-0241591-VFPCHC-3-2 fov=278 (Liberia) -- truth=yes, notes=background, ratio=2.43](previews/LB-D11-2025-12-19-131014-0241591-VFPCHC-3-2__fov278__preview.png)
*LB-D11-2025-12-19-131014-0241591-VFPCHC-3-2 fov=278 (Liberia) -- truth=yes, notes=background, ratio=2.43*

![LB-D3-2025-10-22-132316-2411189646-D-thin-1-4 fov=135 (Liberia) -- truth=yes, notes=diffuse, ratio=2.59](previews/LB-D3-2025-10-22-132316-2411189646-D-thin-1-4__fov135__preview.png)
*LB-D3-2025-10-22-132316-2411189646-D-thin-1-4 fov=135 (Liberia) -- truth=yes, notes=diffuse, ratio=2.59*

![LB-D3-2025-10-22-140622-250917738-D-thin-1-1 fov=122 (Liberia) -- truth=yes, notes=double, ratio=2.75](previews/LB-D3-2025-10-22-140622-250917738-D-thin-1-1__fov122__preview.png)
*LB-D3-2025-10-22-140622-250917738-D-thin-1-1 fov=122 (Liberia) -- truth=yes, notes=double, ratio=2.75*

![LB-D3-2025-10-24-132012-25046898-D-thin-1-4 fov=3 (Liberia) -- truth=yes, notes=diffuse, ratio=2.45](previews/LB-D3-2025-10-24-132012-25046898-D-thin-1-4__fov3__preview.png)
*LB-D3-2025-10-24-132012-25046898-D-thin-1-4 fov=3 (Liberia) -- truth=yes, notes=diffuse, ratio=2.45*

![LB-D3-2025-10-27-154305-250917412-D-thin-1-4 fov=119 (Liberia) -- truth=yes, notes=diffuse, ratio=2.91](previews/LB-D3-2025-10-27-154305-250917412-D-thin-1-4__fov119__preview.png)
*LB-D3-2025-10-27-154305-250917412-D-thin-1-4 fov=119 (Liberia) -- truth=yes, notes=diffuse, ratio=2.91*

### B -- still missed after folding in (spot_truth=yes, missed by both variants) (n=1)

![LB-D3-2025-10-24-162727-230918080-D-thin-1-4 fov=8 (Liberia) -- truth=yes, notes=diffuse, ratio=1.98](previews/LB-D3-2025-10-24-162727-230918080-D-thin-1-4__fov8__preview.png)
*LB-D3-2025-10-24-162727-230918080-D-thin-1-4 fov=8 (Liberia) -- truth=yes, notes=diffuse, ratio=1.98*

(`KIT-62501087` fov=271 briefly sat in this bucket after the `DIFFUSE_RATIO_MIN` fix, then moved
back out entirely once the anisotropy corner-clipping rescue was implemented -- it's now a
direct `present_base=True` true positive, not an error case in either variant. fov=96,
`LB-D3-2025-10-24-113736-250918214-D-thin-2-3`, was also originally in this bucket but has been
removed after Emily flagged its `spot=yes` label as incorrect on 2026-08-07 -- see the `*`
footnote under "Input labels". With the corrected `spot=no` label it's a true negative in both
variants.)

### C -- false positive already at baseline (spot_truth=no, present_base=True; folding in can't fix these, since it only ever turns present False->True) (n=5)

![LB-D3-2025-08-30-103102-250876706-D-thin-4 fov=257 (Liberia) -- truth=no, notes=background, ratio=3.56](previews/LB-D3-2025-08-30-103102-250876706-D-thin-4__fov257__preview.png)
*LB-D3-2025-08-30-103102-250876706-D-thin-4 fov=257 (Liberia) -- truth=no, notes=background, ratio=3.56*

![LB-D3-2025-08-30-103102-250876706-D-thin-4 fov=274 (Liberia) -- truth=no, notes=background, ratio=6.49](previews/LB-D3-2025-08-30-103102-250876706-D-thin-4__fov274__preview.png)
*LB-D3-2025-08-30-103102-250876706-D-thin-4 fov=274 (Liberia) -- truth=no, notes=background, ratio=6.49*

![LB-D3-2025-08-30-103102-250876706-D-thin-4 fov=289 (Liberia) -- truth=no, notes=background, ratio=5.12](previews/LB-D3-2025-08-30-103102-250876706-D-thin-4__fov289__preview.png)
*LB-D3-2025-08-30-103102-250876706-D-thin-4 fov=289 (Liberia) -- truth=no, notes=background, ratio=5.12*

![PAT-072-1 fov=14 (Uganda) -- truth=no, notes=artifact, ratio=3.15](previews/PAT-072-1__fov14__preview.png)
*PAT-072-1 fov=14 (Uganda) -- truth=no, notes=artifact, ratio=3.15*

![PAT-072-1 fov=94 (Uganda) -- truth=no, notes=artifact, ratio=5.91](previews/PAT-072-1__fov94__preview.png)
*PAT-072-1 fov=94 (Uganda) -- truth=no, notes=artifact, ratio=5.91*

### D -- new false positive introduced by folding in (n=0)

Empty. Before the fix, this bucket held the 6 fake-`background` false positives plus `fov126`
(the anisotropy-demoted debris artifact) -- see "Which FOVs flip" above for the full list and
why each is now excluded.

## Caveats

- **Subset sizes are small.** `diffuse` and `double` are 5 rows each; a single-row flip shifts
  their FN/FP rate by 20 points. Treat the subset numbers as directional, not statistically
  robust, until `notes` is filled in further ([[project-fluorescence-diffuse-halo-investigation]]
  already flags the underlying diffuse-halo signal as calibrated on n=1 per class).
- **`notes` is incomplete.** Many rows (mostly straightforward, unambiguous cases) have no
  `notes` value and are excluded from every subset but `all`.
- **GCS per-FOV timings include repeated, uncached slide/box lookups** (see "Per-FOV runtime"
  above) -- don't use these numbers directly to estimate full-slide batch throughput.
- **This doc originally had the predicted/ground-truth polarity inverted** (comparing `NOT
  present` against `spot`, on the mistaken assumption that `spot` meant "genuine parasite
  signal independent of the overexposure artifact"). Corrected 2026-08-07 after Emily clarified
  that `spot` ground truth *is* ground truth on the halo artifact's presence.
- **`DIFFUSE_RATIO_MIN` is a fit on the same 12 labeled rows it was validated against** (6 fake-
  `background` FPs, 6 real sub-ratio halos), cross-checked against 3 historical FOVs from the
  earlier fov62 investigation. It's a much cheaper, wider-margin fit than the patch-uniformity
  metric that was tried and rejected, using a field (`contrast_ratio`) the file already trusts
  -- but it's still calibrated on a small population. `fov279` (see "Discussion") is a known,
  unresolved near-miss inside the new candidate band.
- **`KIT-62501087`'s rescue was initially deliberately traded away** to fix `fov126`'s false
  positive (Variant D, chosen by Emily over Variant C -- see "Which FOVs flip"), then recovered
  by fixing the anisotropy filter's corner-clipping misfire directly (see Discussion's "Third
  pass"). Net result: both `fov126` (fixed) and `KIT-62501087` (rescued) are resolved, without
  the tradeoff.
- **The corner-clipping rescue (`radial_rho`/`r2_over_r1`) is calibrated on 6 labeled fiber/
  artifact cases plus their synthetic corner-clipped crops, and exactly 1 confirmed real
  corner-clipped halo** (`KIT-62501087`). Both thresholds (`RADIAL_RHO_MIN=0.90`,
  `R2_OVER_R1_MAX=0.44`) are provisional. It's rescue-only by construction (can only turn an
  anisotropy-triggered demotion back to `present=True`, never demote an already-passing halo),
  so it can't introduce a new false positive among currently-passing halos -- but it also isn't
  complete: a synthetic corner-clip of an unrelated real halo (`fov210`, cropped toward its own
  corner) narrowly fails the rescue (`radial_rho=0.852`, just under the 0.90 cutoff). That's not
  a regression (that FOV isn't actually corner-clipped in the real labeled set), but a reminder
  this doesn't generalize to every possible corner-clip geometry yet. See
  `scripts/validate_corner_clip_fix.py` for the full validation.

## Files

- `README.md` -- this file
- `results.csv` -- full per-row output: detection fields (`present_base`/`present_folded`/
  `diffuse_halo_flag`/`contrast_ratio`/`anisotropy`/`diffuse_radius`/etc.), both predicted-spot
  variants, and all runtime columns
- `previews/` -- annotated preview thumbnails for all 12 FN/FP rows (see "FN/FP examples"),
  plus `previews/manifest.csv` mapping each file back to its full result row and bucket
- `../../src/gcs_fov_multi.py` -- new LB/TZ/UG FOV resolver (streams, no disk cache)
- `../../scripts/run_overexposed_diverse_test.py` -- pipeline runner used for this test
- `../../scripts/analyze_overexposed_diverse.py` -- confusion-matrix/FN-FP tally generator
- `../../scripts/render_fn_fp_previews.py` -- renders the `previews/` thumbnails
- `../../scripts/validate_corner_clip_fix.py` -- validation harness for the anisotropy
  corner-clipping rescue (`radial_rho`/`r2_over_r1`), incl. synthetic corner-clipped crops

## Recommendations

1. **Cache slide/box resolution per sample_id within a run** if this pipeline is ever run at
   full-slide or multi-scan batch scale -- `gcs_fov.py`/`gcs_fov_multi.py` currently re-resolve
   on every single fetch, which is fine for a 76-row spot-check but would add up across a full
   slide (hundreds of FOVs, often the same handful of samples repeated).
2. **Finish categorizing `notes`** so the background/diffuse/double (and any new categories)
   breakdowns cover the full dataset, not just the ~35 rows currently tagged.
3. **Done, 2026-08-07: `diffuse_candidate()` now requires `DIFFUSE_RATIO_MIN <= contrast_ratio
   < RATIO_THRESHOLD`** (see `src/overexposure.py`'s docstring). This was originally going to
   be "recalibrate `DIFFUSE_ABS_DELTA`," but the actual fix turned out to be scoping candidacy
   with a field already computed (`contrast_ratio`), not a new threshold on the absolute-delta
   footprint. Folding the diffuse-fov step into `present` is now net-positive on this labeled
   set: zero new false positives, FN rate 18.2% -> 4.5%.
4. **Done, 2026-08-07: the anisotropy filter's corner-clipping misfire is fixed directly.**
   `_region_anisotropy`/`detect_overexposure` now carry a rescue-only second opinion
   (`radial_rho`/`r2_over_r1`, see `src/overexposure.py`'s "Corner-clipping rescue" docstring
   section and `scripts/validate_corner_clip_fix.py`) that catches `KIT-62501087` without
   re-admitting any labeled fiber/debris case. Baseline FN improves further, 18.2% -> 15.9%,
   with `KIT-62501087` now a direct true positive rather than a diffuse-fov-dependent one.
5. **Recalibrate `DIFFUSE_RATIO_MIN` (and re-check `fov279`) once more `background`-tagged
   labels exist**, particularly outside Liberia -- the current calibration set is 12 rows, 11
   of them Liberia.
6. **Recalibrate `RADIAL_RHO_MIN`/`R2_OVER_R1_MAX` once more corner-clipped examples exist**
   (both real halos and fibers/debris, across all 3 countries) -- currently 1 confirmed real
   corner-clipped halo and 6 labeled fiber/artifact cases plus synthetic crops. The brainstorm
   that produced this fix suggested deliberately sampling ~15-20 more corner-clipped real halos
   and ~10 corner-clipped fibers by walking FOVs adjacent to known positives/fibers (a halo big
   enough to be corner-clipped in one tile is visible in its neighbors) -- a labeling task, not
   a code change, and a good candidate for the next round of `notes` categorization work
   (Recommendation 2).
