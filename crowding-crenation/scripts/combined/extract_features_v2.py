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
from multiprocessing.pool import ThreadPool

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
    # grayscale=True is bit-identical here and skips decoding three identical channels --
    # every FOV in this repo is already monochrome. See src/pipeline.py::load_image.
    image = load_image(row["image_path"], grayscale=True)
    features = compute_features(image, lbp_step=lbp_step, blur_downsample=blur_downsample)
    return {**row, **features}


def main():
    parser = argparse.ArgumentParser(description="Extract candidate features for the merged calibration set.")
    parser.add_argument("--labels-csv", default=str(MERGED_LABELS_CSV))
    parser.add_argument("--out", default=str(FEATURES_CSV))
    parser.add_argument("--workers", type=int, default=8,
                        help="8 suits local images; 4 is faster for GCS-streamed ones")
    parser.add_argument("--pool", choices=("thread", "process"), default="thread",
                        help="thread (default) is ~2.4x faster and far lighter on memory")
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

# ---- OPTIMIZATION: threads, not processes -------------------------------------------------
# Measured on 40 FOVs (data/results/pipeline-runtime/README.md):
#
#                     local FOVs      GCS-streamed FOVs
#   Pool(2) processes   0.493 s/FOV      0.952 s/FOV
#   ThreadPool(4)       0.267 s/FOV      0.715 s/FOV   <- best for GCS
#   ThreadPool(8)       0.206 s/FOV      0.856 s/FOV   <- best for local
#
# Threads win on both counts because the work releases the GIL almost throughout: blob
# downloads are network IO, and the compute is cv2/numpy calls. They also share one address
# space, which is why 8 of them fit on a machine where Pool(4) ran out of memory outright --
# each *process* worker transiently holds two float32 copies of a 7.84M-pixel image.
#
# GCS prefers 4 and local prefers 8: past 4 concurrent streams the per-thread storage.Client
# connections stop paying for themselves. `_gcs_client()` already keys clients by thread
# (threading.local) precisely so a thread pool is safe here.
# ------------------------------------------------------------------------------------------
    pool_cls = ThreadPool if args.pool == "thread" else Pool
    with pool_cls(args.workers) as pool:
        results = pool.map(partial(_score_one, lbp_step=args.lbp_step,
                                   blur_downsample=args.blur_downsample), rows)

    write_csv_dicts(args.out, FIELDNAMES, results)
    print(f"wrote {len(results)} rows to {args.out} "
          f"(lbp_step={args.lbp_step}, blur_downsample={args.blur_downsample}, "
          f"{args.workers} {args.pool}s)")


if __name__ == "__main__":
    main()
