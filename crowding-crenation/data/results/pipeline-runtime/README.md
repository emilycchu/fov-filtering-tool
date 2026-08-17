# Pipeline runtime after LBP: the illumination blur is 60% of what's left

The LBP work took `compute_features` from 5.85 s to ~1.1 s per 2800x2800 FOV, which moved the
bottleneck rather than removing it. This is the profile of what is expensive now, a 661-FOV
sweep of the one large remaining opportunity, and a record of what turned out not to work.

**Adopted**, after the sweep below established the factor: `correct_illumination` takes a
`downsample`, `compute_features` takes a `blur_downsample`, and it is recorded in the params
JSON so inference can only score at the factor its fit used — the same contract as `lbp_step`.
The calibrated fit is **`v2.2-optimized`**
(`scripts/combined/calibrate_v2.2-optimized.py`), which now carries both `lbp_step: 16` and
`blur_downsample: 4`. Defaults are 1 everywhere, so v2/v2.1/v2.2 params are untouched.

Every number below is reproducible from `scripts/combined/pipeline-runtime/`.

## Headline

As adopted, measured through the params round-trip (`compute_features` with the knobs each
params file actually records):

| | v2.2 params | v2.2-optimized params |
|---|---|---|
| `compute_features` per FOV | 3.11 s | **0.49 s** (6.3x) |
| — against the original v2.2 (skimage LBP) | ~5.9 s | **~12x** |
| features bit-identical vs the LBP-only fit | — | **7 of 9** |
| label changes vs v2.2 on 661 FOVs | — | **1 density, 0 Rouleaux** |
| OOF exact / off-by-one, both axes | 69.4/98.0, 67.6/93.8 | **unchanged** |
| empty-field gate | 3 FOVs, no-op | 3 FOVs, no-op |

The single label difference is `dpc-176-KTR-72502946.png`, and it is **not caused by the blur**
— it is the same knife-edge FOV the LBP refit already moved (score shifted 0.00006 while the
dense/very-dense threshold shifted 0.00135). Adding `blur_downsample=4` introduced **no new**
label differences at all.

The chosen factor is **4**, and the interesting part is *why not 2*: `ds=2` is the only factor
in the whole sweep that changes any label.

## Profile (per FOV, LBP already at stride-16)

Shares of the measured whole, across 3 FOVs, min of 5 each:

| stage | min | share | reducible? |
|---|---|---|---|
| `correct_illumination` (301-px GaussianBlur) | 0.65-0.75 s | **55-71%** | yes, 6.4x |
| decode: `imread` colour | 0.16-0.18 s | 13-15% | yes, exactly |
| decode: `imread` grayscale | 0.12-0.14 s | 10-12% | (the replacement) |
| `tile_glcm` 7x7 grid | 0.11-0.13 s | 9-11% | no route found |
| `glcm_contrast` whole image | 0.09-0.11 s | 7-10% | no — already optimal |
| `lbp_entropy` (stride-16) | 0.09-0.12 s | 7-11% | done |
| `edge_density` (Canny) | 0.03 s | 3% | not worth it |
| `otsu_segment` + `cell_coverage` + `otsu_separability` | 0.04 s | 4% | marginal |

**These are not an additive budget**, and `profile_compute_features.py` says so in its own
output. The blur's spread across 5 consecutive calls reached **0.334 s** — larger than every
stage below it. Read the share column, not the sum; an earlier hand-rolled version of this
profile produced a sum of parts that exceeded its own measured total purely from taking
min-of-N on each piece independently.

Absolute figures here were taken with 322-402 MB of 14 GB free (Chrome, not this work), so they
run high; the blur measured 0.53 s on a quiet machine and 0.77 s on a loaded one. **Ratios and
shares are the robust part.**

## Opportunity 1 — exact: decode grayscale, stop re-converting

Every source image in all three datasets is **already monochrome** — all three channels equal,
in the Tanzania PNGs, Liberia PNGs and Nigeria BMPs alike. So the pipeline decodes three
identical channels through libpng and then averages them back to grey, and does that conversion
six more times because `compute_features` computes `gray` and then passes the *colour* image to
every feature function, each of which opens with
`gray = image if image.ndim == 2 else cv2.cvtColor(...)`.

- `cv2.imread(path, IMREAD_GRAYSCALE)` and `cv2.imdecode(buf, IMREAD_GRAYSCALE)` are
  **bit-identical** to decode-then-`cvtColor`, verified on 8 local FOVs across all three
  datasets, on GCS-streamed bytes, and — as a by-product of this sweep's baseline check — on
  **all 661 calibration FOVs**.
- Saves ~0.04 s on decode plus ~0.018 s of redundant conversions.

Small on its own. It is also the change that makes the path honestly grayscale end to end,
which is what this pipeline actually consumes. The only colour consumer is
`nigeria_081226.py`'s thumbnail sheet (`COLOR_BGR2RGB`, cosmetic).

## Opportunity 2 — the blur, and the surprise at ds=2

A 301-px Gaussian is a low-pass filter by construction, so estimating it on a downsampled copy
loses almost nothing. Same reasoning as striding the LBP centre grid.

Sweep over all 661 FOVs. Only `tile_glcm_cv` and `tile_glcm_patchiness` consume `corrected`, so
only those two columns were recomputed and patched into `features-v2.2-optimized.csv`,
leaving the other seven bit-identical — any label change is attributable to the blur alone.
Scored with `density_overlap_v2.2-optimized_params.json` **unchanged**.

| ds | `correct_illumination` | stage speedup | cv max drift | patchiness max drift | density flips | Rouleaux flips | gate check |
|---|---|---|---|---|---|---|---|
| 1 | 0.773 s | 1.0x | 0 | 0 | 0 | 0 | pass |
| **2** | 0.198 s | 3.9x | **0.0170** | **0.0616** | 0 | **4** | **FAIL** |
| **4** | **0.121 s** | **6.4x** | 0.0033 | 0.0044 | **0** | **0** | **pass** |
| 8 | 0.123 s | 6.3x | 0.0034 | 0.0070 | 0 | 0 | pass |
| 16 | 0.131 s | 5.9x | 0.0036 | 0.0176 | 0 | 0 | pass |

Two things to take from this.

**`ds=2` is the worst factor, not the safest.** It drifts 5-14x more than `ds=4` and is the only
factor that changes any prediction: 4 Rouleaux flips, and those are 4 *correct* predictions
lost (`check_empty_field_gate.py` reports Rouleaux exact-match 0.6838 -> 0.6808). Every other
factor is clean on both axes. This is reproducible, consistent across FOVs, and shows up in the
raw background error too (0.13 grey levels at `ds=2` vs 0.04 at `ds=4`).

I ruled out the obvious cause. OpenCV infers sigma from ksize, and integer-dividing 301 gives
each factor a slightly different effective sigma (45.5 -> 46.0 -> 46.4 -> 47.2 -> 51.2), so I
tested a sigma-matched variant passing `sigmaX` explicitly: **no difference** (0.0361 vs 0.0389
mean error at `ds=4`). So it is the `INTER_AREA` down / `INTER_LINEAR` up round-trip, not the
blur — but **the mechanism is unexplained**, and I would rather record that than offer a
plausible-sounding guess. Nothing rests on it: `ds=4` beats `ds=2` on both speed and accuracy,
so the anomaly is a curiosity, not a tradeoff.

**The stage saturates at ds=4, and 6.4x is the ceiling — not 38x.** The blur *itself* goes from
0.531 s to 0.014 s (38x), but `correct_illumination` also does
`gray.astype(float32) - background.astype(float32) + mean`, then clip, then back to uint8 —
about **0.11 s of fixed cost over 7.84M pixels regardless of downsample**. That tail is why
`ds=8` and `ds=16` are no faster than `ds=4`, and it is the new floor for this stage. Reducing
it would mean changing the arithmetic (int16, or `cv2.subtract` with saturation), which changes
output; out of scope here.

## Measured dead ends — do not retry these

- **GLCM via `bincount` instead of `graycomatrix`.** Reformulating contrast as
  `sum (i-j)^2 P(i,j)` over a `bincount` joint histogram reproduces skimage's value *exactly*
  (67.017756 both) and runs **4x slower** (0.490 s vs 0.117 s). skimage's Cython is already
  efficient; this is not where the time is.
- **Sigma-matching the downsampled blur.** Principled, and worth nothing (above).
- **Downsampling past 4.** No faster, because of the float32 tail; strictly more drift.
- **Smaller LBP tiles to re-enable threading at stride-16.** ~0.05 s, and it adds a knob to a
  stage that is no longer the bottleneck.
- **Sharing Otsu between `otsu_segment` and `otsu_separability`.** Real duplication — both
  convert and threshold — but 0.04 s total for two signature changes.

## Opportunity 3 — the pool was the wrong kind

Once compute dropped to ~0.5 s, `multiprocessing.Pool` stopped making sense. Measured on 40
FOVs, same work, only the pool type changed:

| | local FOVs | GCS-streamed FOVs |
|---|---|---|
| `Pool(2)` processes | 0.493 s/FOV | 0.952 s/FOV |
| `ThreadPool(2)` | 0.365 s/FOV | 0.878 s/FOV |
| `ThreadPool(4)` | 0.267 s/FOV | **0.715 s/FOV** |
| `ThreadPool(8)` | **0.206 s/FOV** | 0.856 s/FOV |

**Threads win on both, by up to 2.4x.** The work releases the GIL almost throughout — blob
downloads are network IO, the compute is cv2/numpy — so threads actually parallelize it. And
because they share one address space, 8 of them fit on a machine where `Pool(4)` ran out of
memory outright: each *process* worker transiently holds two float32 copies of a 7.84M-pixel
image, and 4 of those exceeded the ~400 MB free.

GCS prefers 4 threads and local prefers 8: past 4 concurrent streams the per-thread
`storage.Client` connections stop paying for themselves. `_gcs_client()` already keys clients
by thread (`threading.local`), so a thread pool was the anticipated usage.

End to end, extracting all 661 features went from **~7.9 min to 3.4 min** (0.312 s/FOV wall,
6 threads) — and the output CSV is **byte-identical** to the process-pool version.

**One hazard this exposed.** `lbp_entropy._auto_workers()` returned 1 inside a *process* pool
(daemonic workers) but not inside a *thread* pool, whose workers are ordinary threads of the
non-daemonic main process. Every FOV would then have spawned its own 8 threads on top of the
pool's. It happened to be invisible at `lbp_step=16`, where the single-tile guard takes the
serial branch anyway — so it would only have bitten at small strides, which is exactly when it
costs most. `_auto_workers` now checks `threading.current_thread() is not main_thread()` too.

## IO, which the per-FOV numbers hide

GCS steady-state download is **0.33 s/FOV at ~12 MB/s**; the first call in a process costs
**4.3 s** (connection + auth setup, once). For the 324 streamed FOVs that is ~35% of per-FOV
cost once compute drops to 0.58 s, so a calibration run becomes noticeably IO-bound on its GCS
half. Threaded prefetch would hide it. Deliberately **not** pursued here: it changes no feature
values, and bundling it with work that does would muddy the evidence.

Also worth recording for anyone re-running this: `Pool(4)` **OOM'd outright** on a machine with
~400 MB free, because each worker transiently holds two float32 copies of a 7.84M-pixel image.
The sweep runs in-process at `--workers 1` for that reason (26 min for 661 FOVs).

## What was adopted, and what was not

**Adopted:**

1. **The grey passthrough.** `compute_features` now hands the `gray` it already computed to
   every feature function instead of the colour image, deleting six redundant `cvtColor` calls.
   Bit-identical by construction — each function's first line is the same conversion — and
   checked anyway.
2. **`blur_downsample=4`**, plumbed exactly like `lbp_step`: `correct_illumination(gray,
   blur_ksize, downsample)`, `compute_features(..., blur_downsample=...)`,
   `blur_downsample_from_params()`, recorded by `calibrate_v2.write_params_json`, read back by
   `score_fov_v2.py` and `nigeria_081226.py`. Defaults are 1, so v2/v2.1/v2.2 params score
   exactly as before.
3. **`v2.2-optimized`** — the refit carrying both knobs.

4. **Grayscale decode**, as `load_image(path, grayscale=False)` — opt-in, so callers that
   render images keep colour. `extract_features_v2.py` and `score_fov_v2.py` pass
   `grayscale=True`; `nigeria_081226.py` does not, because its thumbnail sheet needs colour.
5. **`ThreadPool` instead of `Pool`** in both batch paths, with `--pool process` as an escape
   hatch, plus the `_auto_workers` nesting fix that switch required.

**Not adopted:**

6. **`downsample=2`** — disqualified, see above.
7. **Anything past 4.** The float32 tail means there is no speed left, only drift.
8. **A dedicated GCS prefetcher.** The thread pool already overlaps downloads with compute, so
   a separate prefetch stage would add a queue and a lifecycle for a fraction of what switching
   pool types already delivered. Revisit only if GCS-half throughput becomes the binding
   constraint again.

**Next bottleneck** is the `tile_glcm` grid plus whole-image GLCM (~0.2 s combined), and no
cheap exact route was found for either.

## Reproducing

```bash
# stage profile (run it on a quiet machine -- the blur's spread reaches 0.33s)
python scripts/combined/pipeline-runtime/profile_compute_features.py --repeat 5 --fovs 3

# the 661-FOV blur sweep: slow half, then analysis
python scripts/combined/pipeline-runtime/sweep_blur_downsample.py --extract --workers 1
python scripts/combined/pipeline-runtime/sweep_blur_downsample.py --analyze

# gate check per factor
for ds in 1 2 4 8 16; do
  python scripts/combined/check_empty_field_gate.py \
    --features data/results/pipeline-runtime/features-blur-ds${ds}.csv \
    --params data/results/density-rouleaux-v2/density_overlap_v2.2-optimized_params.json \
    --expect-exact 0.6959 0.6838
done
```

## Files

| file | what |
|---|---|
| `stage-profile.csv` | per-stage min / median / spread / share, per FOV |
| `blur-variants.csv` | the two blur-derived features at every factor, all 661 FOVs |
| `blur-comparison.csv` | drift, label flips and refit metrics per factor |
| `features-blur-ds<N>.csv` | `features-v2.2-optimized.csv` with only those two columns swapped |
