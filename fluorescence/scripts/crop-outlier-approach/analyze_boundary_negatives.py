"""Crop-outlier v2: for every distinct slide in data/labels/overexposure-diverse-080726.csv,
compute the same whole-slide `n_spots_detected` baseline stats analyze_crop_outliers.py already
computes for labeled FOVs, but for `fov_id` 1 and 324 instead --
the first and last tile of the fixed 18x18 virtual raster (src/gcs_fov.py) -- to use as
presumed-blank boundary negatives.

This redefines "positive"/"negative" for the crop-outlier approach away from the labels CSV's
`spot` ground truth (whether the overexposure-halo artifact is genuinely present) and towards the
crop-count signal itself: a FOV is filter-worthy if it carries a significant excess of erroneous
crops, independent of what a human tagged it as. FOV 1/324 stand in for "no artifact, blank
tile" -- unless one of them is itself an outlier on its own slide, which is exactly what the
`high_outlier`/`low_outlier` flags below (reused unmodified from analyze_crop_outliers.py) catch:
a flagged boundary row is a candidate-contaminated negative, not a trustworthy one.

Only `n_spots_detected` is used (not `n_positives`) -- see
data/results/crop-outlier-approach/README.md's cross-metric comparison for why: n_positives is
mathematically degenerate (undefined baseline) for 80% of rows in this dataset.

Usage:
    python scripts/crop-outlier-approach/analyze_boundary_negatives.py
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir, for crop_counts (hyphenated package)

from crop_counts import load_slide_metric_counts, slide_baseline  # noqa: E402  (needs sys.path first)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "results" / "crop-outlier-approach" / "crop-outlier-v2"
LABELS_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "labels" / "overexposure-diverse-080726.csv"
OUT_CSV = RESULTS_DIR / "boundary_negatives.csv"

METRIC = "n_spots_detected"
BOUNDARY_FOVS = (1, 324)

# same thresholds as analyze_crop_outliers.py, reused unmodified for comparability
SMALL_SLIDE_THRESHOLD = 20
MEAN_MEDIAN_DIVERGENCE_FRAC = 0.25
OUTLIER_ZSCORE = 2.0

FIELDNAMES = [
    "sample_id", "country", "boundary_fov_id", "n_spots_detected", "baseline_mean",
    "baseline_std", "baseline_median", "baseline_mad", "n_fovs_on_slide",
    "ratio_to_median", "robust_zscore", "mean_zscore", "flags", "no_data_reason",
]


def distinct_slides(labels_csv):
    """First-occurrence (sample_id -> country), preserving label-CSV order."""
    slides = {}
    for row in csv.DictReader(open(labels_csv)):
        slides.setdefault(row["sample_id"], row["country"])
    return list(slides.items())


def _missing_row(sample_id, country, fov_id, reason):
    return {
        "sample_id": sample_id, "country": country, "boundary_fov_id": fov_id,
        "n_spots_detected": "", "baseline_mean": "", "baseline_std": "", "baseline_median": "",
        "baseline_mad": "", "n_fovs_on_slide": "", "ratio_to_median": "",
        "robust_zscore": "", "mean_zscore": "", "flags": "no_data", "no_data_reason": reason,
    }


def process_boundary_fov(sample_id, country, fov_id, counts):
    if counts is None:
        return _missing_row(sample_id, country, fov_id, "no detection_results found for this sample")
    if fov_id not in counts:
        return _missing_row(sample_id, country, fov_id, "fov_id missing from detection results")

    target = counts[fov_id]
    stats = slide_baseline(counts)
    mean, std, median, mad, n_fovs = (
        stats["mean"], stats["std"], stats["median"], stats["mad"], stats["n_fovs"],
    )

    ratio_to_median = target / median if median else None
    robust_zscore = (target - median) / mad if mad else None
    mean_zscore = (target - mean) / std if std else None

    flags = []
    if n_fovs < SMALL_SLIDE_THRESHOLD:
        flags.append("small_slide")
    if median == 0 or mad == 0:
        flags.append("zero_or_undefined_baseline")
    if median and abs(mean - median) > MEAN_MEDIAN_DIVERGENCE_FRAC * median:
        flags.append("mean_median_divergence")
    if robust_zscore is not None and robust_zscore >= OUTLIER_ZSCORE:
        flags.append("high_outlier")
    elif robust_zscore is not None and robust_zscore <= -OUTLIER_ZSCORE:
        flags.append("low_outlier")

    return {
        "sample_id": sample_id,
        "country": country,
        "boundary_fov_id": fov_id,
        "n_spots_detected": target,
        "baseline_mean": round(mean, 3),
        "baseline_std": round(std, 3),
        "baseline_median": median,
        "baseline_mad": round(mad, 3),
        "n_fovs_on_slide": n_fovs,
        "ratio_to_median": round(ratio_to_median, 3) if ratio_to_median is not None else "",
        "robust_zscore": round(robust_zscore, 3) if robust_zscore is not None else "",
        "mean_zscore": round(mean_zscore, 3) if mean_zscore is not None else "",
        "flags": ";".join(flags),
        "no_data_reason": "",
    }


def main():
    slides = distinct_slides(LABELS_CSV)
    print(f"{len(slides)} distinct slides in {LABELS_CSV.name}")

    out_rows = []
    for i, (sample_id, country) in enumerate(slides):
        print(f"[{i + 1}/{len(slides)}] {sample_id} ({country})", flush=True)
        try:
            counts = load_slide_metric_counts(sample_id, country, metric=METRIC)
        except Exception as exc:
            print(f"  [ERROR] {exc}", flush=True)
            for fov_id in BOUNDARY_FOVS:
                out_rows.append(_missing_row(sample_id, country, fov_id, f"error: {exc}"))
            continue
        for fov_id in BOUNDARY_FOVS:
            out_rows.append(process_boundary_fov(sample_id, country, fov_id, counts))

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(out_rows)

    n_no_data = sum(1 for r in out_rows if "no_data" in r["flags"])
    n_flagged_high = sum(1 for r in out_rows if "high_outlier" in r["flags"])
    print(f"\nWrote {len(out_rows)} rows to {OUT_CSV} "
          f"({n_no_data} no_data, {n_flagged_high} high_outlier candidate-contaminated negatives)")


if __name__ == "__main__":
    main()
