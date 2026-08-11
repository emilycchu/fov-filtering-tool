"""Crop/parasite-outlier approach: for every FOV in data/labels/overexposure-diverse-080726.csv,
compare its per-FOV count for one of two metrics (`--metric`, see crop_counts.py's module
docstring) against a leave-one-out baseline built from every *other* FOV on the same slide,
reading only precomputed GCS detection output -- no image analysis at all.

- `n_spots_detected` (default) -- raw candidate fluorescent-spot crops before ML filtering.
- `n_positives` -- crops the ML classifier actually confirmed as a parasite.

Ground truth is the labels CSV's `spot` column (whether the overexposure-halo artifact is
genuinely present). The question this explores: do spot_truth=yes FOVs carry an abnormal count
relative to their own slide's baseline (the hypothesis being that the halo spuriously inflates
this count, the same targeted side effect `src/overexposure.py` catches via pixel signal
instead) -- and how do spot_truth=no FOVs compare on the same measure? Running both metrics
tests whether that inflation survives ML filtering or is specific to raw candidate crops -- see
`data/results/crop-outlier-approach/README.md`'s comparison section for the result.

Baseline center/spread is **median and MAD** (median absolute deviation), not mean/std. This
label set only covers 76 hand-picked FOVs -- there's no ground truth on the other ~95%+ of FOVs
per slide, so an unknown number of *unlabeled* overexposed/artifact FOVs could already be sitting
in a slide's baseline population. Mean/std have zero breakdown resistance (one or two such FOVs
shift them arbitrarily and mute the exact effect being tested); median/MAD tolerate up to ~50%
contamination before breaking. Mean/std are still computed and reported alongside, specifically
so a large mean-vs-median divergence is itself visible as a `mean_median_divergence` flag -- a
direct, computable signal that a slide's own population probably isn't clean.

Every row also gets an explicit `flags` column (semicolon-joined) rather than leaving unusual
cases to be discovered by hand -- see FLAG constants below for what each one means.

Usage:
    python scripts/crop-outlier-approach/analyze_crop_outliers.py
    python scripts/crop-outlier-approach/analyze_crop_outliers.py --metric n_positives
    python scripts/crop-outlier-approach/analyze_crop_outliers.py --limit 5
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir, for crop_counts (hyphenated package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root, for src.*

from crop_counts import load_slide_metric_counts, slide_baseline  # noqa: E402  (needs sys.path first)

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "results" / "crop-outlier-approach"
LABELS_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "labels" / "overexposure-diverse-080726.csv"

# per-metric output path + the CSV column name used for the target count (kept distinct so the
# two metrics' results.csv files are self-describing and don't collide with each other)
METRIC_CONFIG = {
    "n_spots_detected": {"out_csv": RESULTS_DIR / "results.csv", "target_field": "target_n_spots"},
    "n_positives": {"out_csv": RESULTS_DIR / "results_parasites.csv", "target_field": "target_n_positives"},
}

SMALL_SLIDE_THRESHOLD = 20        # fewer than this many *other* FOVs on the slide -> thin baseline
MEAN_MEDIAN_DIVERGENCE_FRAC = 0.25  # |mean - median| > this fraction of median -> likely-unclean slide
OUTLIER_ZSCORE = 2.0              # |robust_zscore| >= this -> high_outlier / low_outlier


def fieldnames_for(metric):
    return [
        "sample_id", "fov_id", "country", "spot_truth", "notes",
        METRIC_CONFIG[metric]["target_field"], "baseline_mean", "baseline_std", "baseline_median",
        "baseline_mad", "n_other_fovs_on_slide", "ratio_to_median", "robust_zscore", "mean_zscore",
        "flags", "no_data_reason",
    ]


def _missing_row(base, reason, metric):
    base.update({
        METRIC_CONFIG[metric]["target_field"]: "", "baseline_mean": "", "baseline_std": "",
        "baseline_median": "", "baseline_mad": "", "n_other_fovs_on_slide": "",
        "ratio_to_median": "", "robust_zscore": "", "mean_zscore": "",
        "flags": "no_data", "no_data_reason": reason,
    })
    return base


def process_row(row, sample_id_counts, metric):
    sample_id, fov_id, country = row["sample_id"], int(row["fov_id"]), row["country"]
    target_field = METRIC_CONFIG[metric]["target_field"]
    base = {
        "sample_id": sample_id,
        "fov_id": fov_id,
        "country": country,
        "spot_truth": row["spot"].strip().lower(),
        "notes": (row.get("notes") or "").strip().lower(),
    }

    counts = load_slide_metric_counts(sample_id, country, metric=metric)
    if counts is None:
        return _missing_row(base, "no detection_results found for this sample", metric)
    if fov_id not in counts:
        return _missing_row(base, "sample has detection_results, but this fov_id is missing from it", metric)

    target = counts[fov_id]
    stats = slide_baseline(counts, fov_id)
    mean, std, median, mad, n_other = (
        stats["mean"], stats["std"], stats["median"], stats["mad"], stats["n_other_fovs"],
    )

    ratio_to_median = target / median if median else None
    robust_zscore = (target - median) / mad if mad else None
    mean_zscore = (target - mean) / std if std else None

    flags = []
    if n_other < SMALL_SLIDE_THRESHOLD:
        flags.append("small_slide")
    if sample_id_counts[sample_id] > 1:
        flags.append("same_slide_contamination")
    if median == 0 or mad == 0:
        flags.append("zero_or_undefined_baseline")
    if median and abs(mean - median) > MEAN_MEDIAN_DIVERGENCE_FRAC * median:
        flags.append("mean_median_divergence")
    if robust_zscore is not None and robust_zscore >= OUTLIER_ZSCORE:
        flags.append("high_outlier")
    elif robust_zscore is not None and robust_zscore <= -OUTLIER_ZSCORE:
        flags.append("low_outlier")

    base.update({
        target_field: target,
        "baseline_mean": round(mean, 3),
        "baseline_std": round(std, 3),
        "baseline_median": median,
        "baseline_mad": round(mad, 3),
        "n_other_fovs_on_slide": n_other,
        "ratio_to_median": round(ratio_to_median, 3) if ratio_to_median is not None else "",
        "robust_zscore": round(robust_zscore, 3) if robust_zscore is not None else "",
        "mean_zscore": round(mean_zscore, 3) if mean_zscore is not None else "",
        "flags": ";".join(flags),
        "no_data_reason": "",
    })
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metric", choices=sorted(METRIC_CONFIG), default="n_spots_detected")
    parser.add_argument("--labels-csv", type=Path, default=LABELS_CSV)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    out_csv = args.out_csv or METRIC_CONFIG[args.metric]["out_csv"]

    rows = list(csv.DictReader(open(args.labels_csv)))
    if args.limit:
        rows = rows[: args.limit]
    sample_id_counts = Counter(r["sample_id"] for r in rows)

    out_rows = []
    for i, row in enumerate(rows):
        print(f"[{i + 1}/{len(rows)}] {row['sample_id']} fov={row['fov_id']} ({row['country']})", flush=True)
        try:
            out_rows.append(process_row(row, sample_id_counts, args.metric))
        except Exception as exc:
            print(f"  [ERROR] {exc}", flush=True)
            out_rows.append(_missing_row({
                "sample_id": row["sample_id"], "fov_id": row["fov_id"], "country": row["country"],
                "spot_truth": row["spot"].strip().lower(), "notes": (row.get("notes") or "").strip().lower(),
            }, f"error: {exc}", args.metric))

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_for(args.metric))
        writer.writeheader()
        writer.writerows(out_rows)

    n_no_data = sum(1 for r in out_rows if "no_data" in r["flags"])
    n_flagged = sum(1 for r in out_rows if r["flags"])
    print(f"\nWrote {len(out_rows)} rows to {out_csv} "
          f"({n_no_data} no_data, {n_flagged} with at least one flag)")


if __name__ == "__main__":
    main()
