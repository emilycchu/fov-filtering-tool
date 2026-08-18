"""The runnable v2 tool: score a new FOV image or directory of images for density and
Rouleaux (overlap) severity, using the weights/thresholds fit by calibration. Unlike
scripts/ai-first/label_new_slide.py's slide-relative quintiles, these thresholds are fixed
ahead of time, so this generalizes to a new slide without re-deriving anything per slide.

Usage:
    python scripts/combined/score_fov_v2.py <input> [--params PATH] [--out-csv PATH]
        [--workers N] [--pool thread|process]

<input> is a single image file or a directory of images (local path or gs:// URI).
Without --out-csv, results are printed as JSON to stdout (matching src/pipeline.py's CLI).

--params defaults to the **v2.2-optimized** fit (661 FOVs). It used to default to the original
v2 fit (337 FOVs, superseded twice over), which meant a bare invocation silently scored with
the weakest available calibration -- masked only by every documented example passing --params
explicitly.

The default carrying `lbp_step`/`blur_downsample` is safe rather than risky: those live in the
params file and every inference path reads them back (`lbp_step_from_params`,
`blur_downsample_from_params`), so the features are always computed the way the fit they are
being scored against was built. Pass density_overlap_v2.2_params.json for the full-resolution
v2.2 fit.
"""
import argparse
import json
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
from pathlib import Path

from _v2_common import (
    DEFAULT_SCORING_PARAMS,
    apply_label_overrides,
    blur_downsample_from_params,
    compute_features,
    lbp_step_from_params,
    list_image_paths,
    load_image,
)
from src.composite_v2 import bucket, weighted_composite


def load_params(path):
    with open(path) as f:
        return json.load(f)


def _score_axis(features, axis_params):
    weights = axis_params["weights"]
    ranges = {n: (v["min"], v["max"]) for n, v in axis_params["normalization"].items()}
    score = weighted_composite(features, weights, ranges)
    label = bucket(score, axis_params["bucket_thresholds"], axis_params["bucket_labels"])
    return score, label


def score_image_v2(path, params):
    # grayscale=True is bit-identical and skips decoding three identical channels --
    # every FOV in this repo is already monochrome. See src/pipeline.py::load_image.
    image = load_image(path, grayscale=True)
    # Both runtime knobs come from the params file, never from a local default -- a fit made
    # on subsampled LBP entropy or a downsampled illumination background has to be scored the
    # same way, and only the fit knows which it was.
    features = compute_features(image, lbp_step=lbp_step_from_params(params),
                                blur_downsample=blur_downsample_from_params(params))

    density_score, density_label = _score_axis(features, params["density"])
    overlap_score, overlap_label = _score_axis(features, params["overlap"])

    density_label, overlap_label = apply_label_overrides(density_label, overlap_label, features, params)

    return {
        "filename": Path(str(path)).name,
        "path": str(path),
        "density_score": density_score,
        "density_label": density_label,
        "overlap_score": overlap_score,
        "overlap_label": overlap_label,
        "saturation_score": features.get("saturation_score"),
    }


def _score_one(args):
    path, params = args
    try:
        return score_image_v2(path, params)
    except Exception as e:  # corrupt/unreadable image -- a real boundary case, not speculative
        return {"filename": Path(str(path)).name, "path": str(path), "error": str(e)}


def score_path(input_path, params, workers=1, pool="thread"):
    from src.pipeline import IMAGE_EXTENSIONS, is_gcs_path, to_dir

    if is_gcs_path(input_path):
        from pathlib import PurePosixPath
        is_single_file = PurePosixPath(input_path).suffix.lower() in IMAGE_EXTENSIONS
        paths = [to_dir(input_path)] if is_single_file else list_image_paths(input_path)
    else:
        p = Path(input_path)
        paths = [p] if p.is_file() else list_image_paths(p)

    if workers > 1:
        # ---- OPTIMIZATION: threads, not processes -----------------------------------------
        # The work releases the GIL almost throughout -- blob downloads are network IO and the
        # compute is cv2/numpy -- so threads parallelize it while sharing one address space.
        # Measured 2.4x faster than a process pool on local FOVs and 1.3x on GCS-streamed
        # ones, and light enough on memory to run 8 where Pool(4) ran out. See
        # data/results/pipeline-runtime/README.md.
        # -----------------------------------------------------------------------------------
        pool_cls = ThreadPool if pool == "thread" else Pool
        with pool_cls(workers) as p:
            return p.map(_score_one, [(path, params) for path in paths])
    return [_score_one((path, params)) for path in paths]


def write_csv(rows, out_csv):
    import csv

    fieldnames = ["filename", "path", "density_score", "density_label", "overlap_score",
                  "overlap_label", "saturation_score", "error"]
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main():
    parser = argparse.ArgumentParser(description="Score FOV images for density and Rouleaux severity.")
    parser.add_argument("input", help="Image file or directory of images (local path or gs:// URI).")
    parser.add_argument("--params", default=str(DEFAULT_SCORING_PARAMS),
                        help="calibrated params JSON; defaults to the v2.2-optimized fit "
                             "(661 FOVs, ~12x faster per FOV, accuracy unchanged). Pass "
                             "density_overlap_v2.2_params.json for the full-resolution v2.2 fit")
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--workers", type=int, default=1,
                        help="8 suits local images; 4 is faster for GCS-streamed ones")
    parser.add_argument("--pool", choices=("thread", "process"), default="thread",
                        help="thread (default) is ~2.4x faster and far lighter on memory")
    args = parser.parse_args()

    params = load_params(args.params)
    results = score_path(args.input, params, workers=args.workers, pool=args.pool)

    if args.out_csv:
        write_csv(results, args.out_csv)
        print(f"wrote {len(results)} rows to {args.out_csv}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
