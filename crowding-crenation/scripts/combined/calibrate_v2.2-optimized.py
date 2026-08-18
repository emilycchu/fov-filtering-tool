"""v2.2-optimized: the v2.2 fit, recalibrated on the two runtime-optimized features.

Same 661 FOVs and the same full-feature-pool fitting as calibrate_v2.2.py. Two things change,
and each moves exactly one part of the feature vector:

- `lbp_entropy` is computed on a **stride-16** grid of centre pixels rather than every pixel:
  0.07s per FOV instead of 4.80s.
- the illumination background is estimated on a **4x downsampled** copy:
  `correct_illumination` drops from 0.77s to 0.12s, moving only `tile_glcm_cv` and
  `tile_glcm_patchiness`.

Together with converting to grey once instead of six times, `compute_features` goes from 5.85s
to ~0.58s per FOV -- about 10x.

Why refit at all, when the drift changes no labels? Because the fit should describe the
features the pipeline actually computes. Scoring optimized features against v2.2's
full-resolution params happens to give identical buckets on all 661 calibration FOVs, but that
is a measured property of this dataset, not a guarantee -- the fitted weights and the p2/p98
normalization bands shift in the 4th decimal, and the empty-field gate's floors are literally
percentiles of lbp_entropy and three other features. Refitting makes the params
self-consistent instead of relying on the drift staying small on data nobody has seen yet.

The params JSON records `lbp_step: 16` and `blur_downsample: 4`, and score_fov_v2.py /
nigeria_081226.py read both back (`_v2_common.lbp_step_from_params`,
`blur_downsample_from_params`), so this fit can only be scored against matching features.
Older params files have neither key and score at full resolution as before.

Evidence, both established before adoption and both sweeps run over all 661 FOVs:

- stride: data/results/lbp-runtime/README.md -- 71x faster, zero label changes, off-by-one
  unchanged, gate still firing on exactly its 3 FOVs.
- blur: data/results/pipeline-runtime/README.md -- 6.4x, zero label changes at downsample 4.
  Note downsample **2** is the one factor that does flip labels (4 correct Rouleaux
  predictions lost), so the chosen factor is not simply "the most conservative one".

Inputs are built by:
    python scripts/combined/extract_features_v2.py --labels-csv <merged-labels-v2.2.csv> \
        --out <features-v2.2-optimized.csv> --lbp-step 16 --workers 4

Usage:
    python scripts/combined/calibrate_v2.2-optimized.py [--features-csv PATH]
        [--params-out PATH] [--report-out PATH] [--prev-params PATH] [--lbp-step N]
        [--blur-downsample N]
"""
import argparse
import json
from pathlib import Path

from _v2_common import (
    BLUR_OPTIMIZED_DOWNSAMPLE,
    DENSITY_LEVELS,
    LBP_OPTIMIZED_STEP,
    OVERLAP_LEVELS,
    RESULTS_DIR,
    display_level,
)
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

FEATURES_CSV_LB = RESULTS_DIR / "features-v2.2-optimized.csv"
PARAMS_JSON_LB = RESULTS_DIR / "density_overlap_v2.2-optimized_params.json"
PARAMS_JSON_V2_2 = RESULTS_DIR / "density_overlap_v2.2_params.json"
REPORT_MD = RESULTS_DIR / "calibration-report.md"


def drift_section(density_result, overlap_result, prev_params, lbp_step):
    """What moved relative to v2.2 -- the point of the whole exercise is that it is nearly
    nothing, so state it in numbers rather than asserting it."""
    lines = ["## What moved vs. v2.2\n\n"]
    if prev_params is None:
        lines.append("No v2.2 params available to compare against.\n\n")
        return lines

    lines.append(f"LBP entropy is now computed on a stride-{lbp_step} centre grid "
                 "(0.07s/FOV vs. 4.80s). Everything else is identical: same 661 FOVs, same "
                 "candidate pool, same ridge/PAVA procedure.\n\n")
    for name, result, axis_key in [("Density", density_result, "density"),
                                   ("Rouleaux", overlap_result, "overlap")]:
        prev = prev_params[axis_key]
        lines.append(f"**{name}**\n\n")
        lines.append("| feature | weight (v2.2) | weight (optimized) | delta |\n|---|---|---|---|\n")
        for feature in result["feature_names"]:
            new = result["weights"][feature]
            old = prev["weights"].get(feature)
            old_str = f"{old:.4f}" if old is not None else "n/a"
            delta = f"{new - old:+.5f}" if old is not None else "new"
            lines.append(f"| {feature} | {old_str} | {new:.4f} | {delta} |\n")
        new_t = result["bucket_thresholds"]
        old_t = prev["bucket_thresholds"]
        lines.append(f"\nThresholds: {[round(t, 4) for t in new_t]} vs. v2.2 "
                     f"{[round(t, 4) for t in old_t]} "
                     f"(max shift {max(abs(a - b) for a, b in zip(new_t, old_t)):.5f}).\n\n")
    return lines


def append_report_section(report_path, rows, density_result, overlap_result, independence,
                          sep_checks, prev_params, lbp_step, features_csv):
    lines = ["\n---\n\n", "# v2.2-optimized: the v2.2 fit on stride-16 LBP entropy\n\n"]
    lines.append(
        f"Refit on the same {len(rows)} FOVs as v2.2, from `{Path(features_csv).name}`, with "
        f"`lbp_entropy` computed on a stride-{lbp_step} centre grid. This is a runtime change, "
        "not a modelling one: `compute_features` drops from 5.85s to ~1.08s per FOV. The stride "
        "was validated first (`data/results/lbp-runtime/README.md`) -- across all 661 FOVs it "
        "changes none of the 1322 bucket assignments under v2.2's own params.\n\n"
    )

    for name, result, levels in [("Density", density_result, DENSITY_LEVELS),
                                 ("Rouleaux", overlap_result, OVERLAP_LEVELS)]:
        lines.append(f"## {name} composite (v2.2-optimized)\n\n")
        if result["dropped_sign_unstable_features"]:
            lines.append(f"Dropped for sign instability: {', '.join(result['dropped_sign_unstable_features'])}\n\n")
        lines.append("| feature | weight | range (2nd-98th pct) |\n|---|---|---|\n")
        for feature in result["feature_names"]:
            lo, hi = result["normalization"][feature]
            lines.append(f"| {feature} | {result['weights'][feature]:.3f} | [{lo:.4g}, {hi:.4g}] |\n")
        lines.append(f"\n**Cross-validation** ({N_FOLDS}-fold): per-fold rho = "
                     f"{[round(r, 3) for r in result['cv_fold_rho']]}, mean={result['cv_mean_rho']:.3f}. "
                     f"Out-of-fold exact-match={result['oof_exact_match_rate']:.1%}, "
                     f"off-by-one={result['oof_off_by_one_rate']:.1%}.\n\n")
        lines.append("Out-of-fold confusion matrix (rows=manual, cols=predicted):\n\n")
        lines.append(_fmt_conf(result["oof_confusion_matrix"], levels))
        lines.append(f"\nThresholds: {[round(t, 3) for t in result['bucket_thresholds']]}")
        if result["merged_bucket_groups"]:
            groups = [[display_level(levels[i]) for i in g] for g in result["merged_bucket_groups"]]
            lines.append(f" -- **PAVA still merged**: {groups}\n\n")
        else:
            lines.append(" -- **no PAVA merges** (all 5 buckets monotonically separable).\n\n")

    lines.extend(drift_section(density_result, overlap_result, prev_params, lbp_step))

    lines.append("## Composite independence (v2.2-optimized)\n\n")
    lines.append(f"Spearman rho between the two fitted composite scores: "
                 f"**{independence['composite_rho']:.3f}**, vs. the true manual "
                 f"density-vs-Rouleaux label correlation of **{independence['manual_label_rho']:.3f}**.\n\n")

    lines.append("## Axis-separation check (v2.2-optimized)\n\n")
    lines.append("| min |delta| | n | sign matches | match rate | binomial p | Spearman rho |\n|---|---|---|---|---|---|\n")
    for sep_check in sep_checks:
        rate = f"{sep_check['sign_match_rate']:.1%}" if sep_check["sign_match_rate"] is not None else "n/a"
        lines.append(f"| {sep_check['min_delta']} | {sep_check['n_disagreement']} | {sep_check['matches']} | "
                     f"{rate} | {sep_check['p_value']:.4g} | {sep_check['spearman_rho']:.3f} |\n")

    with open(report_path, "a", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="v2.2 refit on stride-16 LBP entropy.")
    parser.add_argument("--features-csv", default=str(FEATURES_CSV_LB))
    parser.add_argument("--params-out", default=str(PARAMS_JSON_LB))
    parser.add_argument("--report-out", default=str(REPORT_MD),
                        help="appended to, matching calibrate_v2.2.py -- re-running duplicates the section")
    parser.add_argument("--prev-params", default=str(PARAMS_JSON_V2_2))
    parser.add_argument("--lbp-step", type=int, default=LBP_OPTIMIZED_STEP,
                        help="must match the stride --features-csv was extracted with")
    parser.add_argument("--blur-downsample", type=int, default=BLUR_OPTIMIZED_DOWNSAMPLE,
                        help="must match the downsample --features-csv was extracted with")
    args = parser.parse_args()

    rows = load_features(args.features_csv)
    _, rho_do = correlation_table(rows)

    density_result = calibrate_axis(rows, "density", "density_ord", "density_label",
                                    DENSITY_LEVELS, CANDIDATE_FEATURES)
    overlap_result = calibrate_axis(rows, "overlap", "overlap_ord", "overlap_label",
                                    OVERLAP_LEVELS, CANDIDATE_FEATURES)

    sep_checks = [
        axis_separation_check(rows, density_result["oof_pred_idx"], overlap_result["oof_pred_idx"], min_delta=1),
        axis_separation_check(rows, density_result["oof_pred_idx"], overlap_result["oof_pred_idx"], min_delta=2),
    ]
    independence = composite_independence(density_result, overlap_result, rho_do)

    prev_params = None
    if args.prev_params and Path(args.prev_params).exists():
        prev_params = json.loads(Path(args.prev_params).read_text())

    params_path = Path(args.params_out)
    write_params_json(params_path, density_result, overlap_result, len(rows),
                      lbp_step=args.lbp_step, blur_downsample=args.blur_downsample)
    data = json.loads(params_path.read_text())
    data["version"] = "v2.2-optimized"
    data["generated_from"] = args.features_csv
    params_path.write_text(json.dumps(data, indent=2))

    append_report_section(Path(args.report_out), rows, density_result, overlap_result,
                          independence, sep_checks, prev_params, args.lbp_step, args.features_csv)

    for name, result in [("density", density_result), ("overlap", overlap_result)]:
        print(f"{name}: features={result['feature_names']}, cv_mean_rho={result['cv_mean_rho']:.3f}, "
              f"exact_match={result['oof_exact_match_rate']:.1%}, "
              f"off_by_one={result['oof_off_by_one_rate']:.1%}")
    print(f"composite independence rho={independence['composite_rho']:.3f} "
          f"(manual label rho={independence['manual_label_rho']:.3f})")
    gate = json.loads(params_path.read_text())["empty_field_override"]
    print(f"empty-field gate: {'enabled' if gate['enabled'] else 'DISABLED'} "
          f"over {len(gate['thresholds'])} features")
    print(f"lbp_step={args.lbp_step}, blur_downsample={args.blur_downsample} "
          f"recorded in {params_path.name}")
    print(f"appended to {args.report_out}")


if __name__ == "__main__":
    main()
