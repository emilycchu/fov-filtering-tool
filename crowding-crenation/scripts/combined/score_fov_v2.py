"""The runnable v2 tool: score a new FOV image or directory of images for density and
Rouleaux (overlap) severity, using the weights/thresholds calibrate_v2.py fit against the
337-FOV manual annotation set (density_overlap_v2_params.json). Unlike
scripts/ai-first/label_new_slide.py's slide-relative quintiles, these thresholds are fixed
ahead of time, so this generalizes to a new slide without re-deriving anything per slide.

Usage:
    python scripts/combined/score_fov_v2.py <input> [--params PATH] [--out-csv PATH] [--workers N]

<input> is a single image file or a directory of images (local path or gs:// URI).
Without --out-csv, results are printed as JSON to stdout (matching src/pipeline.py's CLI).
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from _v2_common import PARAMS_JSON, apply_label_overrides, compute_features, list_image_paths, load_image
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
    image = load_image(path)
    features = compute_features(image)

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


def score_path(input_path, params, workers=1):
    from src.pipeline import IMAGE_EXTENSIONS, is_gcs_path, to_dir

    if is_gcs_path(input_path):
        from pathlib import PurePosixPath
        is_single_file = PurePosixPath(input_path).suffix.lower() in IMAGE_EXTENSIONS
        paths = [to_dir(input_path)] if is_single_file else list_image_paths(input_path)
    else:
        p = Path(input_path)
        paths = [p] if p.is_file() else list_image_paths(p)

    if workers > 1:
        with Pool(workers) as pool:
            return pool.map(_score_one, [(p, params) for p in paths])
    return [_score_one((p, params)) for p in paths]


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
    parser.add_argument("--params", default=str(PARAMS_JSON))
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    params = load_params(args.params)
    results = score_path(args.input, params, workers=args.workers)

    if args.out_csv:
        write_csv(results, args.out_csv)
        print(f"wrote {len(results)} rows to {args.out_csv}")
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
