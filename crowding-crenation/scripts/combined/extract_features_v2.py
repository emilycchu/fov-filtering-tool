"""Compute all candidate density/Rouleaux features for every FOV in the merged calibration
set (see merge_labels_v2.py), joined against its manual labels.

Usage:
    python scripts/combined/extract_features_v2.py [--labels-csv PATH] [--out PATH]
        [--workers N] [--limit N] [--lbp-step N]

--lbp-step defaults to 1 (full resolution), so re-running this reproduces the v2/v2.1/v2.2
feature CSVs exactly. Pass --lbp-step 16 to build the LBP-optimized set; whichever value is
used must then be recorded in the params JSON by the calibration script, because
score_fov_v2.py reads it back to score with the same stride.
"""
import argparse
from functools import partial
from multiprocessing import Pool

from _v2_common import (
    DEFAULT_BLUR_DOWNSAMPLE,
    DEFAULT_LBP_STEP,
    FEATURES_CSV,
    MERGED_LABELS_CSV,
    compute_features,
    load_image,
    read_csv_dicts,
    write_csv_dicts,
)

FEATURE_NAMES = [
    "coverage", "otsu_threshold", "otsu_separability", "saturation_score",
    "lbp_entropy", "glcm_contrast", "edge_density_unmasked",
    "tile_glcm_cv", "tile_glcm_patchiness",
]
LABEL_FIELDNAMES = ["fov_key", "dataset", "filename", "image_path", "density_label", "overlap_label",
                     "density_ord", "overlap_ord"]
FIELDNAMES = LABEL_FIELDNAMES + FEATURE_NAMES


def _score_one(row, lbp_step=DEFAULT_LBP_STEP, blur_downsample=DEFAULT_BLUR_DOWNSAMPLE):
    image = load_image(row["image_path"])
    features = compute_features(image, lbp_step=lbp_step, blur_downsample=blur_downsample)
    return {**row, **features}


def main():
    parser = argparse.ArgumentParser(description="Extract candidate features for the merged calibration set.")
    parser.add_argument("--labels-csv", default=str(MERGED_LABELS_CSV))
    parser.add_argument("--out", default=str(FEATURES_CSV))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--lbp-step", type=int, default=DEFAULT_LBP_STEP,
                        help="LBP centre-grid stride; 1 (default) reproduces v2/v2.1/v2.2")
    parser.add_argument("--blur-downsample", type=int, default=DEFAULT_BLUR_DOWNSAMPLE,
                        help="illumination-background downsample; 1 (default) reproduces "
                             "v2/v2.1/v2.2. Use 4; 2 is measurably worse than 4")
    args = parser.parse_args()

    rows = read_csv_dicts(args.labels_csv)
    if args.limit:
        rows = rows[: args.limit]

    with Pool(args.workers) as pool:
        results = pool.map(partial(_score_one, lbp_step=args.lbp_step,
                                          blur_downsample=args.blur_downsample), rows)

    write_csv_dicts(args.out, FIELDNAMES, results)
    print(f"wrote {len(results)} rows to {args.out} "
          f"(lbp_step={args.lbp_step}, blur_downsample={args.blur_downsample})")


if __name__ == "__main__":
    main()
