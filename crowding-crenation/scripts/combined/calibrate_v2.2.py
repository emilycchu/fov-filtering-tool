"""v2.2 recalibration: pool in the second Tanzania slide (tanzania-080526, KTR-72502946,
324 FOVs) alongside the original 337-FOV set (13 initial-071626 + 324 tanzania-073026),
growing the calibration set to 661 FOVs. Same full-feature-pool fitting as v2.1
(fit_weights_stable against the full 8-feature candidate pool per axis, not the
axis-exclusive subset v2 used) -- this script only changes *what data* goes in, not how
it's fit.

Motivation: KTR-72502946 adds 6 more Sparser-density FOVs (44 -> 50 across the pool) and a
second slide's worth of every other bucket, which is the more direct test of whether v2.1's
thresholds -- derived from a single slide's label distribution -- actually generalize,
particularly at the Sparser/Monolayer boundary where the original set had comparatively
few examples relative to Monolayer's dominance (241/337).

Usage:
    python scripts/combined/calibrate_v2.2.py [--features-csv PATH] [--params-out PATH]
        [--report-out PATH] [--prev-params PATH]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from scipy.stats import spearmanr

from _v2_common import DENSITY_LEVELS, OVERLAP_LEVELS, RESULTS_DIR, display_level
from calibrate_v2 import (
    CANDIDATE_FEATURES,
    N_FOLDS,
    _fmt_conf,
    axis_separation_check,
    calibrate_axis,
    composite_independence,
    correlation_table,
    load_features,
    write_params_json,
)

FEATURES_CSV_V2_2 = RESULTS_DIR / "features-v2.2.csv"
PARAMS_JSON_V2_2 = RESULTS_DIR / "density_overlap_v2.2_params.json"
PARAMS_JSON_V2_1 = RESULTS_DIR / "density_overlap_v2.1_params.json"
REPORT_MD = RESULTS_DIR / "calibration-report.md"


def dataset_density_breakdown(rows):
    counts = {}
    for r in rows:
        counts.setdefault(r["dataset"], Counter())[r["density_label"]] += 1
    return counts


def sparser_bucket_section(rows, density_result, prev_params):
    lines = ["## Sparser-bucket focus (v2.2)\n\n"]
    breakdown = dataset_density_breakdown(rows)
    lines.append("Sparser-density FOV counts by source dataset:\n\n")
    lines.append("| dataset | sparser | total |\n|---|---|---|\n")
    for dataset, counts in breakdown.items():
        total = sum(counts.values())
        lines.append(f"| {dataset} | {counts.get('sparser', 0)} | {total} |\n")

    levels = DENSITY_LEVELS
    sparser_i, monolayer_i = levels.index("sparser"), levels.index("monolayer")
    conf = density_result["oof_confusion_matrix"]
    sparser_row = conf[sparser_i]
    n_sparser = sum(sparser_row)
    lines.append(
        f"\nOut-of-fold, v2.2 calls Sparser correctly on {sparser_row[sparser_i]}/{n_sparser} "
        f"({sparser_row[sparser_i] / n_sparser:.1%}) of manually-labeled Sparser FOVs; "
        f"{sparser_row[monolayer_i]}/{n_sparser} are mistaken for Monolayer, its only neighbor "
        "on the scale.\n\n"
    )

    new_threshold = density_result["bucket_thresholds"][sparser_i]
    if prev_params is not None:
        prev_threshold = prev_params["density"]["bucket_thresholds"][sparser_i]
        lines.append(
            f"Sparser/Monolayer raw-score threshold: **{new_threshold:.3f}** (v2.2, n={n_sparser} Sparser "
            f"FOVs) vs. **{prev_threshold:.3f}** (v2.1, n=44 Sparser FOVs, single-slide) -- "
            f"{'moved' if abs(new_threshold - prev_threshold) > 1e-3 else 'unchanged'} by "
            f"{new_threshold - prev_threshold:+.3f} after pooling in the second slide.\n\n"
        )
    else:
        lines.append(f"Sparser/Monolayer raw-score threshold: **{new_threshold:.3f}** (n={n_sparser} Sparser FOVs).\n\n")

    if density_result["merged_bucket_groups"]:
        merged_levels = {i for g in density_result["merged_bucket_groups"] for i in g}
        if sparser_i in merged_levels:
            lines.append("**Note:** Sparser is still PAVA-merged with an adjacent bucket at this pool size -- "
                          "not yet cleanly separable.\n\n")
        else:
            lines.append("Sparser is not PAVA-merged with any adjacent bucket -- cleanly separable at this pool size.\n\n")
    else:
        lines.append("No PAVA merges anywhere on the density axis at this pool size.\n\n")

    return lines


def append_report_section(report_path, rows, density_result, overlap_result, independence, sep_checks, prev_params):
    lines = []
    lines.append("\n---\n\n")
    lines.append("# v2.2 recalibration: pooling in tanzania-080526 (KTR-72502946)\n\n")
    lines.append(
        f"Same full-feature-pool fitting as v2.1, refit on {len(rows)} FOVs (the original 337 plus 324 more "
        "from a second Tanzania slide, KTR-72502946 -- streamed from GCS, never downloaded locally). "
        "This tests whether v2.1's thresholds, fit on a single slide's label distribution, generalize "
        "to a second slide once that slide's own labels are pooled in rather than held out.\n\n"
    )

    for name, result, levels in [("Density", density_result, DENSITY_LEVELS), ("Rouleaux", overlap_result, OVERLAP_LEVELS)]:
        lines.append(f"## {name} composite (v2.2)\n\n")
        if result["dropped_sign_unstable_features"]:
            lines.append(f"Dropped for sign instability: {', '.join(result['dropped_sign_unstable_features'])}\n\n")
        lines.append("| feature | weight | range (2nd-98th pct) |\n|---|---|---|\n")
        for n in result["feature_names"]:
            lo, hi = result["normalization"][n]
            lines.append(f"| {n} | {result['weights'][n]:.3f} | [{lo:.4g}, {hi:.4g}] |\n")

        lines.append(f"\n**Cross-validation** ({N_FOLDS}-fold): per-fold rho = {[round(r, 3) for r in result['cv_fold_rho']]}, "
                     f"mean={result['cv_mean_rho']:.3f}. Out-of-fold exact-match={result['oof_exact_match_rate']:.1%}, "
                     f"off-by-one={result['oof_off_by_one_rate']:.1%}.\n\n")
        lines.append("Out-of-fold confusion matrix (rows=manual, cols=predicted):\n\n")
        lines.append(_fmt_conf(result["oof_confusion_matrix"], levels))
        lines.append(f"\nThresholds: {[round(t, 3) for t in result['bucket_thresholds']]}")
        if result["merged_bucket_groups"]:
            groups = [[display_level(levels[i]) for i in g] for g in result["merged_bucket_groups"]]
            lines.append(f" -- **PAVA still merged**: {groups}\n\n")
        else:
            lines.append(" -- **no PAVA merges** (all 5 buckets monotonically separable).\n\n")

    lines.extend(sparser_bucket_section(rows, density_result, prev_params))

    lines.append("## Composite independence (v2.2)\n\n")
    lines.append(
        f"Spearman rho between the two fitted composite scores: **{independence['composite_rho']:.3f}**, vs. the true "
        f"manual density-vs-Rouleaux label correlation of **{independence['manual_label_rho']:.3f}**.\n\n"
    )

    lines.append("## Axis-separation check (v2.2)\n\n")
    lines.append("| min |delta| | n | sign matches | match rate | binomial p | Spearman rho |\n|---|---|---|---|---|---|\n")
    for sep_check in sep_checks:
        sign_match_str = f"{sep_check['sign_match_rate']:.1%}" if sep_check["sign_match_rate"] is not None else "n/a"
        lines.append(f"| {sep_check['min_delta']} | {sep_check['n_disagreement']} | {sep_check['matches']} | "
                     f"{sign_match_str} | {sep_check['p_value']:.4g} | {sep_check['spearman_rho']:.3f} |\n")

    with open(report_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="v2.2 recalibration: pool in tanzania-080526 and refit both axes.")
    parser.add_argument("--features-csv", default=str(FEATURES_CSV_V2_2))
    parser.add_argument("--params-out", default=str(PARAMS_JSON_V2_2))
    parser.add_argument("--report-out", default=str(REPORT_MD))
    parser.add_argument("--prev-params", default=str(PARAMS_JSON_V2_1))
    args = parser.parse_args()

    rows = load_features(args.features_csv)
    _, rho_do = correlation_table(rows)

    density_result = calibrate_axis(rows, "density", "density_ord", "density_label", DENSITY_LEVELS, CANDIDATE_FEATURES)
    overlap_result = calibrate_axis(rows, "overlap", "overlap_ord", "overlap_label", OVERLAP_LEVELS, CANDIDATE_FEATURES)

    sep_checks = [
        axis_separation_check(rows, density_result["oof_pred_idx"], overlap_result["oof_pred_idx"], min_delta=1),
        axis_separation_check(rows, density_result["oof_pred_idx"], overlap_result["oof_pred_idx"], min_delta=2),
    ]
    independence = composite_independence(density_result, overlap_result, rho_do)

    prev_params = None
    if args.prev_params and Path(args.prev_params).exists():
        prev_params = json.loads(Path(args.prev_params).read_text())

    params_path = Path(args.params_out)
    write_params_json(params_path, density_result, overlap_result, len(rows))
    data = json.loads(params_path.read_text())
    data["version"] = "v2.2"
    data["generated_from"] = args.features_csv
    params_path.write_text(json.dumps(data, indent=2))

    append_report_section(Path(args.report_out), rows, density_result, overlap_result, independence, sep_checks, prev_params)

    print(f"density: features={density_result['feature_names']}, cv_mean_rho={density_result['cv_mean_rho']:.3f}, "
         f"exact_match={density_result['oof_exact_match_rate']:.1%}, off_by_one={density_result['oof_off_by_one_rate']:.1%}")
    print(f"overlap: features={overlap_result['feature_names']}, cv_mean_rho={overlap_result['cv_mean_rho']:.3f}, "
         f"exact_match={overlap_result['oof_exact_match_rate']:.1%}, off_by_one={overlap_result['oof_off_by_one_rate']:.1%}")
    print(f"composite independence rho={independence['composite_rho']:.3f} (manual label rho={independence['manual_label_rho']:.3f})")
    for sep_check in sep_checks:
        print(f"axis-separation (|delta|>={sep_check['min_delta']}): {sep_check['matches']}/{sep_check['n_disagreement']} "
              f"sign matches, p={sep_check['p_value']:.4g}")
    print(f"wrote {params_path}")
    print(f"appended to {args.report_out}")


if __name__ == "__main__":
    main()
