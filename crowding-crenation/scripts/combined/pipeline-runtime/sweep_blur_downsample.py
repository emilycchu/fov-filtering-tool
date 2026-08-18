"""Measure what a cheap illumination blur costs in accuracy, across the full 661-FOV set.

`correct_illumination`'s 301-px GaussianBlur is ~45% of per-FOV feature time now that LBP is
at stride-16. A 301-px Gaussian is by construction a low-pass filter, so estimating it on a
downsampled copy loses almost nothing -- the same reasoning that made striding the LBP centre
grid safe. On one FOV that is 0.531s -> 0.014s at downsample=4, for a background differing by
at most 1 grey level. This script establishes whether it is safe across the whole calibration
set, or only on the handful of FOVs the pilot looked at.

**Analysis only.** Nothing here modifies `src/segmentation.py`, `compute_features`, or any
params file. It writes to `data/results/pipeline-runtime/` and reads everything else.

`corrected` feeds exactly two features -- `tile_glcm_cv` and `tile_glcm_patchiness` -- so the
other seven columns cannot move, and there is no need to re-extract them. Each factor's feature
CSV is `features-v2.2-optimized.csv` with those two columns swapped, which makes any label
change attributable to the blur alone. Same trick as
`scripts/combined/lbp-optimization/build_variant_features.py`.

downsample=1 calls the real `correct_illumination`, so it must reproduce
`features-v2.2-optimized.csv` bit-for-bit. That is the gate on this whole study: if the
baseline does not reproduce, no other number here can be trusted. It doubles as a 661-FOV check
that decoding grayscale directly is identical to decode-then-convert, since this script loads
with `IMREAD_GRAYSCALE` where `compute_features` converts from colour.

Usage:
    python scripts/combined/pipeline-runtime/sweep_blur_downsample.py            # both stages
    python scripts/combined/pipeline-runtime/sweep_blur_downsample.py --extract  # slow half
    python scripts/combined/pipeline-runtime/sweep_blur_downsample.py --analyze  # fast half
"""
import argparse
import csv
import json
import sys
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    DENSITY_LEVELS,
    OVERLAP_LEVELS,
    RESULTS_DIR,
    TILE_GLCM_LEVELS,
    TILE_GRID_SIZE,
    apply_label_overrides,
    read_csv_dicts,
    write_csv_dicts,
)
from calibrate_v2 import CANDIDATE_FEATURES, calibrate_axis, load_features  # noqa: E402
from check_empty_field_gate import score_axis  # noqa: E402
from src.features.glcm_contrast import glcm_contrast  # noqa: E402
from src.features.tile_heterogeneity import (  # noqa: E402
    coefficient_of_variation,
    patchiness,
    tile_statistics,
)
from src.pipeline import GCSPath, is_gcs_path, to_dir  # noqa: E402
from src.segmentation import correct_illumination  # noqa: E402

OUT_DIR = ROOT / "data" / "results" / "pipeline-runtime"
VARIANTS_CSV = OUT_DIR / "blur-variants.csv"
COMPARISON_CSV = OUT_DIR / "blur-comparison.csv"
BASE_FEATURES_CSV = RESULTS_DIR / "features-v2.2-optimized.csv"
BASE_PARAMS_JSON = RESULTS_DIR / "density_overlap_v2.2-optimized_params.json"
MERGED_LABELS_CSV = RESULTS_DIR / "merged-labels-v2.2.csv"

DEFAULT_FACTORS = (1, 2, 4, 8, 16)
BLUR_KSIZE = 301
BLUR_FEATURES = ("tile_glcm_cv", "tile_glcm_patchiness")
AXES = [("density", "density_label", DENSITY_LEVELS), ("overlap", "overlap_label", OVERLAP_LEVELS)]


def column(feature, factor):
    return f"{feature}_ds{factor}"


def load_gray(path):
    """Grayscale straight from the decoder -- bit-identical to decode-then-cvtColor on every
    FOV in this repo (all three datasets store monochrome content), and it skips decoding three
    identical channels. Verified across all 661 rows by this script's downsample=1 baseline."""
    if isinstance(path, GCSPath) or is_gcs_path(str(path)):
        gcs = path if isinstance(path, GCSPath) else to_dir(str(path))
        from src.pipeline import _gcs_client

        data = _gcs_client().bucket(gcs.bucket).blob(gcs.blob_name).download_as_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
    else:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def corrected_at(gray, downsample, blur_ksize=BLUR_KSIZE):
    """`correct_illumination`, with the background estimated on a downsampled copy.

    downsample=1 delegates to the real function rather than reimplementing it at scale 1, so
    the baseline is the production code path and not a lookalike.
    """
    if downsample == 1:
        return correct_illumination(gray, blur_ksize=blur_ksize)
    ksize = blur_ksize // downsample
    ksize += 1 - ksize % 2  # GaussianBlur requires odd
    small = cv2.resize(gray, None, fx=1 / downsample, fy=1 / downsample,
                       interpolation=cv2.INTER_AREA)
    background = cv2.resize(cv2.GaussianBlur(small, (ksize, ksize), 0),
                            (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    corrected = gray.astype(np.float32) - background.astype(np.float32) + float(background.mean())
    return np.clip(corrected, 0, 255).astype(np.uint8)


def _tile_glcm_contrast(tile):
    quantized = (tile.astype(np.uint16) * TILE_GLCM_LEVELS // 256).astype(np.uint8)
    return glcm_contrast(quantized, levels=TILE_GLCM_LEVELS)


def _measure_one(row, factors=DEFAULT_FACTORS):
    gray = load_gray(row["image_path"])
    out = {"fov_key": row["fov_key"], "dataset": row["dataset"], "filename": row["filename"]}
    for factor in factors:
        stats = tile_statistics(corrected_at(gray, factor), grid_size=TILE_GRID_SIZE,
                                stat_fn=_tile_glcm_contrast)
        out[column("tile_glcm_cv", factor)] = repr(coefficient_of_variation(stats))
        out[column("tile_glcm_patchiness", factor)] = repr(patchiness(stats))
    return out


def extract(factors, workers, limit):
    rows = read_csv_dicts(MERGED_LABELS_CSV)
    if limit:
        rows = rows[:limit]
    print(f"{len(rows)} FOVs, downsample factors {list(factors)}, {workers} workers")
    start = time.perf_counter()
    measure = partial(_measure_one, factors=tuple(factors))

    def progress(i, result, results):
        results.append(result)
        if i % 50 == 0 or i == len(rows):
            rate = (time.perf_counter() - start) / i
            print(f"  {i}/{len(rows)}  {rate:.2f}s/FOV  "
                  f"eta {(len(rows) - i) * rate / 60:.1f} min", flush=True)

    results = []
    if workers <= 1:
        # In-process rather than Pool(1): each worker transiently holds two float32 copies of a
        # 7.8M-pixel image, and a second interpreter's numpy/cv2 is pure overhead. This machine
        # has been down to ~400 MB free, where Pool(4) OOMs outright.
        for i, row in enumerate(rows, start=1):
            progress(i, measure(row), results)
    else:
        with Pool(workers) as pool:
            for i, result in enumerate(pool.imap(measure, rows, chunksize=4), start=1):
                progress(i, result, results)
    fieldnames = ["fov_key", "dataset", "filename"] + [
        column(f, ds) for ds in factors for f in BLUR_FEATURES]
    write_csv_dicts(VARIANTS_CSV, fieldnames, results)
    print(f"wrote {len(results)} rows to {VARIANTS_CSV} in {(time.perf_counter()-start)/60:.1f} min")


def check_baseline(base_rows, variants):
    """The gate: downsample=1 must reproduce the shipped columns bit-for-bit."""
    mismatched = []
    for row in base_rows:
        v = variants[row["fov_key"]]
        for feature in BLUR_FEATURES:
            if row[feature] != v[column(feature, 1)]:
                mismatched.append((row["fov_key"], feature, row[feature], v[column(feature, 1)]))
    print(f"baseline check: {len(base_rows) - len({m[0] for m in mismatched})}/{len(base_rows)} "
          f"rows reproduce {BASE_FEATURES_CSV.name} bit-for-bit")
    for key, feature, want, got in mismatched[:5]:
        print(f"  MISMATCH {key} {feature}: shipped {want} != recomputed {got}")
    return not mismatched


def predict(rows, params):
    out = []
    for row in rows:
        density = score_axis(row, params["density"])
        overlap = score_axis(row, params["overlap"])
        out.append(apply_label_overrides(density, overlap, row, params))
    return out


def patched_rows(base_rows, variants, factor):
    rows = []
    for row in base_rows:
        patched = dict(row)
        for feature in BLUR_FEATURES:
            patched[feature] = variants[row["fov_key"]][column(feature, factor)]
        rows.append(patched)
    return rows


def analyze(factors):
    base_rows = read_csv_dicts(BASE_FEATURES_CSV)
    variants = {r["fov_key"]: r for r in read_csv_dicts(VARIANTS_CSV)}
    params = json.loads(BASE_PARAMS_JSON.read_text())
    with open(BASE_FEATURES_CSV, newline="") as f:
        fieldnames = f.readline().strip().split(",")

    # --limit produces a partial variants file; analyze what was measured rather than dying,
    # but say so, because refit metrics on a subset are not comparable to the shipped ones.
    if len(variants) < len(base_rows):
        base_rows = [r for r in base_rows if r["fov_key"] in variants]
        print(f"NOTE: variants cover only {len(base_rows)} of the calibration set -- this is a "
              "smoke run. Refit numbers below are not comparable to the 661-FOV fit.\n")

    if not check_baseline(base_rows, variants):
        print("\nFAILED: the downsample=1 baseline does not reproduce the shipped columns, so "
              "every other number in this sweep is suspect. Stopping.")
        return 1

    untouched = [c for c in fieldnames if c not in BLUR_FEATURES]
    baseline_pred = predict(base_rows, params)
    out_rows = []
    print(f"\n{'ds':>4} {'cv drift(max)':>14} {'patch drift(max)':>17} "
          f"{'flips D':>8} {'flips R':>8} {'refit exact D':>14} {'refit off1 D':>13}")
    for factor in factors:
        rows = patched_rows(base_rows, variants, factor)
        # attributability: every other column must be byte-identical
        assert all(a[c] == b[c] for a, b in zip(base_rows, rows) for c in untouched), \
            "a non-blur column moved; the patch is wrong"

        drift = {f: max(abs(float(r[f]) - float(b[f])) for r, b in zip(rows, base_rows))
                 for f in BLUR_FEATURES}
        pred = predict(rows, params)
        flips = [sum(1 for p, q in zip(pred, baseline_pred) if p[i] != q[i]) for i in (0, 1)]

        tmp = OUT_DIR / f"features-blur-ds{factor}.csv"
        write_csv_dicts(tmp, fieldnames, rows)
        feats = load_features(tmp)
        refit = {axis: calibrate_axis(feats, axis, f"{axis}_ord", key, levels, CANDIDATE_FEATURES)
                 for axis, key, levels in AXES}

        print(f"{factor:>4} {drift['tile_glcm_cv']:>14.6f} {drift['tile_glcm_patchiness']:>17.6f} "
              f"{flips[0]:>8} {flips[1]:>8} "
              f"{refit['density']['oof_exact_match_rate']:>14.4f} "
              f"{refit['density']['oof_off_by_one_rate']:>13.4f}")
        out_rows.append({
            "downsample": factor,
            "tile_glcm_cv_max_drift": f"{drift['tile_glcm_cv']:.8f}",
            "tile_glcm_patchiness_max_drift": f"{drift['tile_glcm_patchiness']:.8f}",
            "density_flips": flips[0], "overlap_flips": flips[1],
            "features_csv": tmp.name,
            **{f"refit_{axis}_{metric}": f"{refit[axis][f'oof_{metric}_rate']:.6f}"
               for axis in ("density", "overlap") for metric in ("exact_match", "off_by_one")},
        })

    write_csv_dicts(COMPARISON_CSV, list(out_rows[0]), out_rows)
    print(f"\nwrote {COMPARISON_CSV}")
    print("per-factor feature CSVs written for check_empty_field_gate.py --features")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Sweep illumination-blur downsample factors.")
    parser.add_argument("--factors", type=int, nargs="+", default=list(DEFAULT_FACTORS))
    parser.add_argument("--workers", type=int, default=4,
                        help="4 beats 8 here -- the work is memory-bandwidth-bound")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--extract", action="store_true", help="only the slow measurement pass")
    parser.add_argument("--analyze", action="store_true", help="only the analysis of an existing CSV")
    args = parser.parse_args()

    do_extract = args.extract or not args.analyze
    do_analyze = args.analyze or not args.extract
    if do_extract:
        extract(args.factors, args.workers, args.limit)
    if do_analyze:
        return analyze(args.factors)
    return 0


if __name__ == "__main__":
    sys.exit(main())
