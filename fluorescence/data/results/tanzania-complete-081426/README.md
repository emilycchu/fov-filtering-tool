# Tanzania complete: both fluorescence flagging methods across all 271 catalog slides

The fluorescence half of the slide-level Tanzania sweep. The crowding half, the frozen slide
list, and the analysis rationale live in
`../../../crowding-crenation/data/results/tanzania-complete-081426/README.md` -- read that first.

Two methods, over the same 271 slides, so their per-slide flag rates are directly comparable:

| | method | reads | cost |
|---|---|---|---|
| **1** | `run_overexposure_pass.py` -- the pixel-level halo detector (`src/overexposure.py`) | 87,801 fluorescence FOVs, streamed | the expensive one |
| **2** | `run_crop_outlier_pass.py` -- per-FOV spot count vs. the slide's own median/MAD | precomputed `fov_summary.csv`, no images | 271 slides in **48 s** |

## Status: both complete

Run on the `n2-standard-32` Spot VM, 2026-08-17. Method 1: 271 slides, **87,801 FOVs, 0 errors**,
74 min at 19.8 FOV/s. Method 2: 271 slides in **3.3 s** on the VM (48 s locally — it is bound by
271 small CSV fetches, so it tracks network latency rather than compute).

## Method 2 results (complete)

```
271 slides, metric=n_spots_detected, 16 threads
sources: {'tz_detection': 187, 'annotation_bucket': 84}
slide flags: {'mean_median_divergence': 32}
elapsed: 3.3s   (VM, in-region)
```

- **Source split is exactly 187 / 84**, matching what was predicted from the bucket layout: 187
  slides have results under `v8_hardneg_single_t0.995`, and 84 exist only in the annotation
  bucket, produced by a different (older) detector. The z-score is computed *within* a slide, so
  that difference largely cancels -- but `source` is a column on every slide so the claim stays
  checkable rather than assumed. `run_crop_outlier_pass.py --assert-sources` fails if the split
  moves.
- **32 slides carry `mean_median_divergence`** (|mean − median| > 25% of median), the existing
  `analyze_crop_outliers.py` signal for a slide whose count distribution is not clean. Worth
  looking at before trusting those slides' flag counts.
- **The pass stores `robust_zscore`, never a boolean.** The threshold is applied by the crowding
  side's `aggregate_slides.py --crop-outlier-z` (default 6.0), so re-sweeping costs a local
  re-run and the existing v1/v2 scripts keep their own `OUTLIER_ZSCORE = 2.0` untouched. Measured
  across all 271 slides:

  | threshold | FOVs flagged |
  |---|---|
  | z ≥ 2 | 6,165 |
  | **z ≥ 6** (adopted) | **641** |
  | z ≥ 10 | 289 |

## Method 1 was the slow half, and why

The 271-slide run took **74 minutes** for the overexposure pass against **46 minutes** for the
crowding pass over a comparable FOV count — 19.8 FOV/s vs 31.9 FOV/s — despite doing far less
compute per FOV (a 400px downsample and a blur, versus the whole 9-feature vector).

The cause was not missing threading: a `ThreadPoolExecutor` over each slide's FOVs was here from the
start. It was that the pass ran as **one process**, and threads saturate on the GIL well before the
machine does. numpy and cv2 release it for the blob download and the heavy array work, but
per-operation dispatch holds it — the same ceiling the crowding pass hit at 8 threads / 14.35 FOV/s
before its process fan-out took it to 29.75.

`--procs` now fans out over processes, each with its own GIL and thread pool, on a disjoint shard of
slides — sharded *after* the checkpoint skip so a resumed run spreads only the remaining work. On
the VM config used for the run (`--procs 8 --threads 8`), this pass should land near the crowding
pass's throughput rather than at 62% of it, cutting ~74 min to roughly 30.

Not re-run: the existing results are complete and correct, and re-streaming 474 GB to produce
identical CSVs faster would be spending an hour to save one. The flag applies to the next sweep.

## Method 1 design notes

Three things it does differently from `../../../fluorescence/scripts/run_overexposed_diverse_test.py`,
which scores isolated labeled FOVs:

- **Box resolution comes from `slide-index.json`, not `find_tz_box`.** That helper probes up to 5
  `TZ2025-Box<N>/` prefixes per call and caches nothing -- up to 439,005 list requests over this
  sweep. The blob name is constructed directly and `_download_color` reused.
- **`diffuse_halo_flag` costs zero extra fetches.** It needs the two preceding FOVs' results to
  test whether a candidate matches a neighbouring illumination trend; the diverse test re-downloads
  them because it only ever holds one FOV, whereas a full-slide pass already has all 324 in memory.
- **`result.mask` is dropped immediately** (~160 KB × 324 ≈ 52 MB per slide, never written out).
  The advisory-flag step only needs the scalar diffuse fields, which survive.

`present` is the counted flag -- what production returns today. `present_folded`
(`present or diffuse_halo_flag`) ships alongside as advisory, using the same column name the
diverse test uses so the two results files stay comparable.

`contrast_ratio` and `anisotropy` are stored per FOV so `RATIO_THRESHOLD` can be re-swept offline
without re-downloading 474 GB. The anisotropy *demotion* branch cannot be fully replayed that way,
because `radial_rho` is only computed when it fires.

## Verification already run

- Detector output matches the committed values on the regression slide: `present` and
  `contrast_ratio` exact on 12/12 KTR-72502946 FOVs vs.
  `../../../crowding-crenation/data/results/tanzania-080526/merged-results.csv`.
- `crop_counts.robust_zscore` and `ratio_to_median` reproduce all 324 values in
  `../crop-outlier-approach/crop-outlier-v2/case-study-KTR-72502946.csv` exactly.
- `load_slide_metric_counts` still returns a bare dict, so the three existing crop-outlier scripts
  are untouched by the `_with_source` refactor.

KTR-72502946 is the regression slide and is **not** one of the 271; it is scored only via
`--include-non-catalog`.

## Files

| file | what |
|---|---|
| `fov/overexposure/<slide>.csv` | per-FOV `present`, `present_folded`, and every `OverexposureResult` scalar |
| `fov/crop-outlier/<slide>.csv` | per-FOV `n_detected`, `ratio_to_median`, `robust_zscore` |
| `crop-outlier-slides.csv` | per slide: source, model version, baseline mean/std/median/MAD, flags |
| `logs/` | append-only errors + one progress line per completed slide |

## Running it

```bash
python scripts/tanzania-complete-081426/run_crop_outlier_pass.py --assert-sources
python scripts/tanzania-complete-081426/run_overexposure_pass.py --procs 8 --threads 8
```

Both are resumable per slide: a slide whose CSV exists with the expected row count is skipped, and
CSVs are renamed into place only after fsync, so a file that exists is complete.
