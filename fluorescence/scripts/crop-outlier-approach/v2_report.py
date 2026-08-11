"""Crop-outlier v2 reporting: join v1's results.csv (labeled-FOV crop counts +
whole-slide-baseline stats) with boundary_negatives.csv (the same stats for FOV 1/324 on each
slide) and print markdown tables to stdout -- same pattern as report_tables.py, no new stats
dependency.

Redefinition being tested here: "positive" (filter-worthy) = a significant excess of erroneous
crops (`robust_zscore` over some threshold), independent of the labels CSV's `spot_truth`
ground truth. FOV 1/324 stand in for confirmed-blank negatives *unless* one of them is itself a
`high_outlier` on its own slide (a candidate-contaminated negative, excluded from the clean
reference pool used to pick a threshold).

Usage:
    python scripts/crop-outlier-approach/v2_report.py
"""
import csv
import statistics
from pathlib import Path

RESULTS_V1_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "results" / "crop-outlier-approach" / "results.csv"
BOUNDARY_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "results" / "crop-outlier-approach" / "crop-outlier-v2" / "boundary_negatives.csv"

SWEEP_TIERS = [2, 3, 5, 6, 8, 10, 15, 20, 30, 50]


def load_rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def has_flag(row, flag):
    return flag in row["flags"].split(";")


def labeled_by_slide(v1_rows):
    by_slide = {}
    for r in v1_rows:
        by_slide.setdefault(r["sample_id"], []).append(r)
    return by_slide


def boundary_by_slide(boundary_rows):
    by_slide = {}
    for r in boundary_rows:
        by_slide.setdefault(r["sample_id"], {})[int(r["boundary_fov_id"])] = r
    return by_slide


def fmt_boundary_cell(row):
    if row is None:
        return "?"
    if has_flag(row, "no_data"):
        return "no_data"
    tag = " [CONTAMINATED]" if has_flag(row, "high_outlier") else ""
    return f"{row['n_spots_detected']} (z={row['robust_zscore']}){tag}"


def per_slide_comparison_table(labeled_by_slide_map, boundary_by_slide_map):
    lines = ["### Per-slide comparison: labeled FOV(s) vs. boundary FOVs 1/324", "",
             "| sample_id | country | labeled fov_id (spot_truth, n_spots) | fov1 n_spots (z) | fov324 n_spots (z) |",
             "|---|---|---|---|---|"]
    for sample_id, boundary in boundary_by_slide_map.items():
        labeled = labeled_by_slide_map.get(sample_id, [])
        country = labeled[0]["country"] if labeled else next(iter(boundary.values()))["country"]
        labeled_desc = "; ".join(
            f"{r['fov_id']} ({r['spot_truth']}, {r['target_n_spots'] or 'no_data'})" for r in labeled
        ) or "(none)"
        lines.append(
            f"| {sample_id} | {country} | {labeled_desc} | "
            f"{fmt_boundary_cell(boundary.get(1))} | {fmt_boundary_cell(boundary.get(324))} |"
        )
    return "\n".join(lines) + "\n"


def boundary_flagged_table(boundary_rows):
    flagged = [r for r in boundary_rows if has_flag(r, "high_outlier")]
    lines = [f"### Boundary rows flagged `high_outlier` -- candidate-contaminated negatives (n={len(flagged)} of {len(boundary_rows)})", "",
             "| sample_id | boundary_fov_id | n_spots_detected | baseline_median | baseline_mad | ratio_to_median | robust_zscore |",
             "|---|---|---|---|---|---|---|"]
    for r in flagged:
        lines.append(
            f"| {r['sample_id']} | {r['boundary_fov_id']} | {r['n_spots_detected']} | "
            f"{r['baseline_median']} | {r['baseline_mad']} | {r['ratio_to_median']} | {r['robust_zscore']} |"
        )
    return "\n".join(lines) + "\n"


def usable_boundary_rows(boundary_rows):
    """Rows with a computable robust_zscore, excluding no_data / zero_or_undefined_baseline."""
    return [r for r in boundary_rows if r["robust_zscore"] not in ("", None)]


def clean_negative_rows(boundary_rows):
    return [r for r in usable_boundary_rows(boundary_rows) if not has_flag(r, "high_outlier")]


def zscore_summary(rows, label):
    zscores = [float(r["robust_zscore"]) for r in rows]
    lines = [f"**{label}** (n={len(rows)})", ""]
    if not zscores:
        lines.append("No usable rows.\n")
        return "\n".join(lines)
    lines.append(f"- median robust_zscore: {statistics.median(zscores):.2f} "
                 f"(mean {statistics.mean(zscores):.2f}, range {min(zscores):.2f}-{max(zscores):.2f})")
    ratios = [float(r["ratio_to_median"]) for r in rows if r["ratio_to_median"]]
    if ratios:
        lines.append(f"- median ratio_to_median: {statistics.median(ratios):.2f} "
                     f"(mean {statistics.mean(ratios):.2f}, range {min(ratios):.2f}-{max(ratios):.2f})")
    return "\n".join(lines) + "\n"


def threshold_sweep_table(boundary_neg, labeled_yes, labeled_no):
    """FPR is computed against *all* usable boundary rows, not just the ones not already flagged
    `high_outlier` -- using the `high_outlier`-filtered ("clean") pool here would be circular,
    since that flag is itself `robust_zscore >= 2`, which would trivially force 0% FPR at every
    tier in this sweep. The 13 flagged boundary rows are still real observations of the same
    presumed-blank tiles; excluding them from the FPR denominator would be assuming the answer.
    """
    neg_z = [float(r["robust_zscore"]) for r in boundary_neg]
    yes_z = [float(r["robust_zscore"]) for r in labeled_yes if r["robust_zscore"]]
    no_z = [float(r["robust_zscore"]) for r in labeled_no if r["robust_zscore"]]

    lines = ["### Threshold sweep", "",
             f"Boundary negatives (all usable, including the 13 flagged) n={len(neg_z)}, "
             f"labeled spot_truth=yes n={len(yes_z)}, labeled spot_truth=no n={len(no_z)}.", "",
             "| robust_zscore >= | FPR (all boundary negatives) | recall (spot_truth=yes) | flip rate (spot_truth=no) |",
             "|---|---|---|---|"]
    for tier in SWEEP_TIERS:
        fpr = sum(1 for z in neg_z if z >= tier) / len(neg_z) if neg_z else float("nan")
        recall = sum(1 for z in yes_z if z >= tier) / len(yes_z) if yes_z else float("nan")
        flip = sum(1 for z in no_z if z >= tier) / len(no_z) if no_z else float("nan")
        lines.append(f"| {tier} | {fpr:.1%} | {recall:.1%} | {flip:.1%} |")
    return "\n".join(lines) + "\n"


def cross_tab(labeled_rows, threshold):
    """2x2: new-definition positive/negative vs. spot_truth yes/no, at a given z-score threshold."""
    usable = [r for r in labeled_rows if r["robust_zscore"]]
    tp = sum(1 for r in usable if r["spot_truth"] == "yes" and float(r["robust_zscore"]) >= threshold)
    fn = sum(1 for r in usable if r["spot_truth"] == "yes" and float(r["robust_zscore"]) < threshold)
    fp = sum(1 for r in usable if r["spot_truth"] == "no" and float(r["robust_zscore"]) >= threshold)
    tn = sum(1 for r in usable if r["spot_truth"] == "no" and float(r["robust_zscore"]) < threshold)
    lines = [f"### New-vs-old cross-tab at robust_zscore >= {threshold} (n={len(usable)} usable of {len(labeled_rows)})", "",
             "| | new_positive | new_negative |",
             "|---|---|---|",
             f"| spot_truth=yes | {tp} | {fn} |",
             f"| spot_truth=no | {fp} | {tn} |"]
    return "\n".join(lines) + "\n"


def main():
    v1_rows = load_rows(RESULTS_V1_CSV)
    boundary_rows = load_rows(BOUNDARY_CSV)

    labeled_map = labeled_by_slide(v1_rows)
    boundary_map = boundary_by_slide(boundary_rows)

    print(per_slide_comparison_table(labeled_map, boundary_map))
    print(boundary_flagged_table(boundary_rows))

    usable = usable_boundary_rows(boundary_rows)
    clean_neg = clean_negative_rows(boundary_rows)
    print(zscore_summary(usable, "All usable boundary rows (fov 1 & 324, no_data excluded)"))
    print(zscore_summary(clean_neg, "Clean boundary negatives (high_outlier excluded)"))

    labeled_yes = [r for r in v1_rows if r["spot_truth"] == "yes"]
    labeled_no = [r for r in v1_rows if r["spot_truth"] == "no"]
    print(threshold_sweep_table(usable, labeled_yes, labeled_no))

    for threshold in (2, 5, 6, 10):
        print(cross_tab(v1_rows, threshold))


if __name__ == "__main__":
    main()
