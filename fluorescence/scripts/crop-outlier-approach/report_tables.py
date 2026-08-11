"""Build markdown tables from analyze_crop_outliers.py's results CSV (either metric): one table
+ summary stats for spot_truth=yes rows, one for spot_truth=no rows, and a dedicated
flagged-rows table. Prints markdown to stdout (same pattern as
scripts/analyze_overexposed_diverse.py) -- no new stats-library dependency.

Works on either metric's output unmodified -- the target-count column name (`target_n_spots` or
`target_n_positives`) is detected from the CSV header rather than hardcoded.

Usage:
    python scripts/crop-outlier-approach/report_tables.py
    python scripts/crop-outlier-approach/report_tables.py data/results/crop-outlier-approach/results.csv
    python scripts/crop-outlier-approach/report_tables.py data/results/crop-outlier-approach/results_parasites.csv
"""
import argparse
import csv
import statistics
from pathlib import Path

DEFAULT_RESULTS_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "results" / "crop-outlier-approach" / "results.csv"

OUTLIER_ZSCORE_TIERS = [2, 5, 10, 20]


def has_data(row):
    return "no_data" not in row["flags"].split(";")


def group_table(rows, label, target_field):
    lines = [f"### {label} (n={len(rows)})", "",
              f"| sample_id | fov_id | notes | {target_field} | baseline_median | baseline_mad | "
              "ratio_to_median | robust_zscore | flags |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['sample_id']} | {r['fov_id']} | {r['notes']} | {r[target_field]} | "
            f"{r['baseline_median']} | {r['baseline_mad']} | {r['ratio_to_median']} | "
            f"{r['robust_zscore']} | {r['flags']} |"
        )
    return "\n".join(lines) + "\n"


def group_summary(rows, label):
    with_data = [r for r in rows if has_data(r) and r["robust_zscore"]]
    n_no_data = len(rows) - len(with_data)
    lines = [f"**{label} summary** (n={len(rows)}, {n_no_data} no_data excluded from stats)", ""]
    if not with_data:
        lines.append("No rows with usable data.\n")
        return "\n".join(lines)

    zscores = [float(r["robust_zscore"]) for r in with_data]
    ratios = [float(r["ratio_to_median"]) for r in with_data if r["ratio_to_median"]]
    lines.append(f"- median robust_zscore: {statistics.median(zscores):.2f} "
                 f"(mean {statistics.mean(zscores):.2f}, range {min(zscores):.2f}-{max(zscores):.2f})")
    if ratios:
        lines.append(f"- median ratio_to_median: {statistics.median(ratios):.2f} "
                     f"(mean {statistics.mean(ratios):.2f}, range {min(ratios):.2f}-{max(ratios):.2f})")
    for tier in OUTLIER_ZSCORE_TIERS:
        n_above = sum(1 for z in zscores if z >= tier)
        lines.append(f"- {n_above}/{len(with_data)} ({n_above / len(with_data):.1%}) at or above {tier} MAD above baseline")
    return "\n".join(lines) + "\n"


def flagged_table(rows, target_field):
    flagged = [r for r in rows if r["flags"]]
    lines = [f"### Flagged rows (n={len(flagged)} of {len(rows)} total)", "",
              f"| sample_id | fov_id | spot_truth | notes | {target_field} | baseline_median | "
              "robust_zscore | flags | no_data_reason |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in flagged:
        lines.append(
            f"| {r['sample_id']} | {r['fov_id']} | {r['spot_truth']} | {r['notes']} | "
            f"{r[target_field]} | {r['baseline_median']} | {r['robust_zscore']} | "
            f"{r['flags']} | {r['no_data_reason']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path, nargs="?", default=DEFAULT_RESULTS_CSV)
    args = parser.parse_args()

    with open(args.results_csv) as f:
        reader = csv.DictReader(f)
        target_field = next(fn for fn in reader.fieldnames if fn.startswith("target_"))
        rows = list(reader)
    positives = [r for r in rows if r["spot_truth"] == "yes"]
    negatives = [r for r in rows if r["spot_truth"] == "no"]

    print(f"\n## Ground truth: positive (spot_truth=yes)\n")
    print(group_summary(positives, "Positive"))
    print(group_table(positives, "Positive rows", target_field))

    print(f"\n## Ground truth: negative (spot_truth=no)\n")
    print(group_summary(negatives, "Negative"))
    print(group_table(negatives, "Negative rows", target_field))

    print("\n## Flagged rows\n")
    print(flagged_table(rows, target_field))


if __name__ == "__main__":
    main()
