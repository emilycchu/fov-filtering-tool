"""Prove the new pass reproduces already-validated per-FOV scores bit-for-bit.

`data/results/tanzania-080526/merged-results.csv` holds 324 verified per-FOV rows for
KTR-72502946, produced by `scripts/tanzania_080526.py` with the **v2.1** fit. That makes it the
one available golden-value target for this harness, and it exercises every part that could
silently go wrong:

  - blob path construction from (box, slide_id, fov_id)
  - the retry-wrapped `grayscale=True` download (bit-identical to the colour decode it replaces)
  - `compute_features` taking its knobs from the params file
  - the `score_features_v2` extraction from `score_image_v2`

Running against v2.1 also checks the knob defaults from the other direction: v2.1 predates both
`lbp_step` and `blur_downsample`, so a passing run proves they default to 1 rather than silently
inheriting the deployed fit's 16 and 4.

KTR-72502946 is not one of the 271 catalog slides, so it is only in the slide index because
`build_slide_index.py` adds it via REGRESSION_SLIDES.

Usage:
    python scripts/tanzania-complete-081426/verify_regression.py                  # all 324
    python scripts/tanzania-complete-081426/verify_regression.py --limit-fovs 24  # quick
    python scripts/tanzania-complete-081426/verify_regression.py --params-check   # also v2.2-optimized
"""
import argparse
import csv
import json
import sys
from multiprocessing.pool import ThreadPool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _slide_common import ROOT, dpc_gcs_path, with_retry  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    DEFAULT_SCORING_PARAMS,
    blur_downsample_from_params,
    compute_features,
    lbp_step_from_params,
    load_image,
)
from score_fov_v2 import score_features_v2  # noqa: E402

GOLDEN_CSV = ROOT / "data" / "results" / "tanzania-080526" / "merged-results.csv"
GOLDEN_SLIDE = "KTR-72502946"
GOLDEN_BOX = "TZ2025-Box5"
GOLDEN_PARAMS = ROOT / "data" / "results" / "density-rouleaux-v2" / "density_overlap_v2.1_params.json"

TOLERANCE = 1e-9

PARAMS_DIR = ROOT / "data" / "results" / "density-rouleaux-v2"


def load_golden():
    with open(GOLDEN_CSV, newline="", encoding="utf-8") as f:
        return {int(r["fov_id"]): r for r in csv.DictReader(f)}


def score(fov_ids, params, threads):
    def work(fov_id):
        image = with_retry(load_image, dpc_gcs_path(GOLDEN_BOX, GOLDEN_SLIDE, fov_id),
                           grayscale=True)
        features = compute_features(image, lbp_step=lbp_step_from_params(params),
                                    blur_downsample=blur_downsample_from_params(params))
        return fov_id, score_features_v2(features, params)

    with ThreadPool(threads) as pool:
        return dict(pool.map(work, fov_ids))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit-fovs", type=int, default=None,
                        help="check only the first N FOVs (default: all 324)")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--params-check", action="store_true",
                        help="also score with the deployed fit and report how much it moves")
    args = parser.parse_args()

    golden = load_golden()
    fov_ids = sorted(golden)
    if args.limit_fovs:
        fov_ids = fov_ids[:args.limit_fovs]

    with open(GOLDEN_PARAMS, encoding="utf-8") as f:
        v21 = json.load(f)
    print(f"golden: {GOLDEN_CSV.name} ({len(golden)} rows), params {v21.get('version')} "
          f"lbp_step={lbp_step_from_params(v21)} blur_downsample={blur_downsample_from_params(v21)}")
    print(f"scoring {len(fov_ids)} FOVs of {GOLDEN_SLIDE} ...", flush=True)

    scored = score(fov_ids, v21, args.threads)

    worst_density = worst_overlap = 0.0
    label_mismatches = []
    for fov_id in fov_ids:
        ref, got = golden[fov_id], scored[fov_id]
        worst_density = max(worst_density,
                            abs(got["density_score"] - float(ref["model_density_score"])))
        worst_overlap = max(worst_overlap,
                            abs(got["overlap_score"] - float(ref["model_overlap_score"])))
        if (got["density_label"] != ref["model_density_label"]
                or got["overlap_label"] != ref["model_overlap_label"]):
            label_mismatches.append(fov_id)

    print(f"\nmax |density_score - golden|: {worst_density:.3e}")
    print(f"max |overlap_score - golden|: {worst_overlap:.3e}")
    print(f"label mismatches: {len(label_mismatches)}"
          + (f" {label_mismatches[:10]}" if label_mismatches else ""))

    if args.params_check:
        with open(DEFAULT_SCORING_PARAMS, encoding="utf-8") as f:
            deployed = json.load(f)
        opt = score(fov_ids, deployed, args.threads)
        d_shift = max(abs(opt[i]["density_score"] - scored[i]["density_score"]) for i in fov_ids)
        o_shift = max(abs(opt[i]["overlap_score"] - scored[i]["overlap_score"]) for i in fov_ids)
        gated = sum(1 for i in fov_ids if opt[i]["empty_field_gated"])
        print(f"\n{deployed.get('version')} vs v2.1 on the same FOVs: "
              f"max density shift {d_shift:.4f}, max overlap shift {o_shift:.4f}, "
              f"{gated} gated")

    ok = worst_density <= TOLERANCE and worst_overlap <= TOLERANCE and not label_mismatches
    if not ok:
        print(f"\nFAILED: scores differ from the committed v2.1 values by more than {TOLERANCE}")
        return 1
    print(f"\nOK: {len(fov_ids)}/{len(fov_ids)} FOVs reproduce the committed v2.1 scores to "
          f"within {TOLERANCE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
