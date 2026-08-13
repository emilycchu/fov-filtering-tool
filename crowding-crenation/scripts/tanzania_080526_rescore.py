"""Rescore slide KTR-72502946 (tanzania-080526) with the v2.2-recalibrated params, reusing
the feature vectors already computed for it as part of the v2.2 calibration set
(data/results/density-rouleaux-v2/features-v2.2.csv) instead of re-fetching images from GCS
-- v2.2 pooled this slide's 324 FOVs into the calibration set, so its features are already
sitting in that file.

Fluorescent-spot results are unchanged from the original run (same images, same detector),
so they're carried over from data/results/tanzania-080526/merged-results.csv rather than
rerun.

Usage:
    python scripts/tanzania_080526_rescore.py
"""
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import apply_label_overrides  # noqa: E402
from src.composite_v2 import bucket, weighted_composite  # noqa: E402
from tanzania_080526 import OUT_DIR, plot_jitter_grid  # noqa: E402

FEATURES_CSV = ROOT / "data" / "results" / "density-rouleaux-v2" / "features-v2.2.csv"
PARAMS_JSON = ROOT / "data" / "results" / "density-rouleaux-v2" / "density_overlap_v2.2_params.json"
ORIGINAL_MERGED_CSV = OUT_DIR / "merged-results.csv"

FOV_RE = re.compile(r"dpc-(\d+)-KTR-72502946\.png")


def _axis_score_and_label(features, axis_params):
    weights = axis_params["weights"]
    ranges = {n: (v["min"], v["max"]) for n, v in axis_params["normalization"].items()}
    score = weighted_composite(features, weights, ranges)
    label = bucket(score, axis_params["bucket_thresholds"], axis_params["bucket_labels"])
    return score, label


def score_row(row, params):
    density_score, density_label = _axis_score_and_label(row, params["density"])
    overlap_score, overlap_label = _axis_score_and_label(row, params["overlap"])
    density_label, overlap_label = apply_label_overrides(density_label, overlap_label, row, params)
    return density_score, density_label, overlap_score, overlap_label


def load_cached_features(path):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for r in rows:
        if r["dataset"] != "tanzania-080526":
            continue
        m = FOV_RE.search(r["filename"])
        fov_id = int(m.group(1))
        for k in ("coverage", "otsu_threshold", "otsu_separability", "saturation_score", "lbp_entropy",
                  "glcm_contrast", "edge_density_unmasked", "tile_glcm_cv", "tile_glcm_patchiness"):
            r[k] = float(r[k])
        out[fov_id] = r
    return out


def load_fluorescence(path):
    rows = list(csv.DictReader(open(path)))
    return {int(r["fov_id"]): r for r in rows}


def write_merged_csv(results, out_path):
    fieldnames = [
        "fov_id", "manual_density", "manual_overlap",
        "model_density_label", "model_overlap_label", "model_density_score", "model_overlap_score",
        "fluorescent_present", "fluorescent_confidence", "fluorescent_contrast_ratio",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fov_id in sorted(results):
            writer.writerow({k: results[fov_id][k] for k in fieldnames})


def main():
    import json

    params = json.loads(PARAMS_JSON.read_text())
    features = load_cached_features(FEATURES_CSV)
    fluorescence = load_fluorescence(ORIGINAL_MERGED_CSV)

    common = sorted(set(features) & set(fluorescence))
    if len(common) != 324:
        print(f"warning: expected 324 FOVs, got {len(common)} in common "
              f"(features={len(features)}, fluorescence={len(fluorescence)})")

    results = {}
    for fov_id in common:
        row = features[fov_id]
        density_score, density_label, overlap_score, overlap_label = score_row(row, params)
        fl = fluorescence[fov_id]
        results[fov_id] = {
            "fov_id": fov_id,
            "manual_density": row["density_label"],
            "manual_overlap": row["overlap_label"],
            "model_density_label": density_label,
            "model_overlap_label": overlap_label,
            "model_density_score": density_score,
            "model_overlap_score": overlap_score,
            "fluorescent_present": fl["fluorescent_present"] == "True",
            "fluorescent_confidence": fl["fluorescent_confidence"],
            "fluorescent_contrast_ratio": fl["fluorescent_contrast_ratio"],
        }

    merged_csv = OUT_DIR / "merged-results-calibrated.csv"
    write_merged_csv(results, merged_csv)
    print(f"wrote {merged_csv}")

    records = list(results.values())
    plot_path = OUT_DIR / "jitter-bucket-comparison-calibrated.png"
    plot_jitter_grid(records, plot_path, model_label="score_fov_v2 (v2.2 recalibrated)")
    print(f"wrote {plot_path}")

    from _v2_common import density_ordinal, overlap_ordinal

    n = len(records)
    d_exact = sum(1 for r in records if r["manual_density"] == r["model_density_label"]) / n
    d_offbyone = sum(1 for r in records if abs(density_ordinal(r["manual_density"]) - density_ordinal(r["model_density_label"])) <= 1) / n
    o_exact = sum(1 for r in records if r["manual_overlap"] == r["model_overlap_label"]) / n
    o_offbyone = sum(1 for r in records if abs(overlap_ordinal(r["manual_overlap"]) - overlap_ordinal(r["model_overlap_label"])) <= 1) / n
    print(f"n={n}, density exact-match={d_exact:.1%}, off-by-one={d_offbyone:.1%}")
    print(f"Rouleaux exact-match={o_exact:.1%}, off-by-one={o_offbyone:.1%}")


if __name__ == "__main__":
    main()
