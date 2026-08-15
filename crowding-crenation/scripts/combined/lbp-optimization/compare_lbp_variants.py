"""Compare every LBP runtime variant against the v2.2 fit on the full 661-FOV calibration set.

Two independent arms, because they answer different questions and can disagree:

1. **Fixed-params inference.** Score all 661 FOVs with `density_overlap_v2.2_params.json`
   untouched, swapping only the `lbp_entropy` value. This is what would happen if a faster LBP
   were dropped into the deployed pipeline tomorrow without recalibrating, so it is where the
   "zero label changes" bar is judged. Reuses `check_empty_field_gate.score_axis` so scoring is
   identical to the gate check, and `apply_label_overrides` so the empty-field gate is live.
   Not applicable to `nolbp`: v2.2's density composite weights `lbp_entropy`, so there is no
   way to score without it.

2. **Refit.** Rerun the v2.2 fit on the variant's feature CSV and compare fitted weights,
   normalization ranges, bucket thresholds, PAVA merges, and out-of-fold exact-match /
   off-by-one. Imports the fitting functions from `calibrate_v2.py` rather than restating them,
   so a variant cannot silently be fit differently from how v2.2 was.

`calibrate_v2.2.py` is deliberately not invoked directly: its `CANDIDATE_FEATURES` is
module-level, so the `nolbp` arm could not drop the feature, and its report writer opens
`calibration-report.md` in append mode -- one stray default would pollute the canonical v2.2
record. Everything here writes under `data/results/lbp-runtime/`.

Usage:
    python scripts/combined/lbp-optimization/compare_lbp_variants.py [--steps 2 4 6 8] [--out-dir PATH]
"""
import argparse
import json
import sys
from pathlib import Path

# one deeper than the other scripts/combined scripts, hence the extra .parent
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    DENSITY_LEVELS,
    LBP_COMPARISON_CSV,
    LBP_RUNTIME_DIR,
    OVERLAP_LEVELS,
    RESULTS_DIR,
    apply_label_overrides,
    read_csv_dicts,
    write_csv_dicts,
)
from calibrate_v2 import (  # noqa: E402
    CANDIDATE_FEATURES,
    calibrate_axis,
    load_features,
    write_params_json,
)
from check_empty_field_gate import score_axis  # noqa: E402

BASE_FEATURES_CSV = RESULTS_DIR / "features-v2.2.csv"
BASE_PARAMS_JSON = RESULTS_DIR / "density_overlap_v2.2_params.json"
AXES = [("density", "density_label", DENSITY_LEVELS), ("overlap", "overlap_label", OVERLAP_LEVELS)]


def variant_specs(steps):
    """(name, features_csv, candidate_features, scorable_with_v2_2_params) per variant.

    The exact variant reuses features-v2.2.csv directly: the fast kernel reproduces that
    column bit-for-bit (asserted by bench_lbp.py and re-checked over all 661 rows), so
    building a copy of it would only invite the two files to drift.
    """
    specs = [("exact", BASE_FEATURES_CSV, CANDIDATE_FEATURES, True)]
    for step in steps:
        specs.append((f"step{step}", LBP_RUNTIME_DIR / f"features-v2.2-step{step}.csv",
                      CANDIDATE_FEATURES, True))
    specs.append(("nolbp", BASE_FEATURES_CSV,
                  [f for f in CANDIDATE_FEATURES if f != "lbp_entropy"], False))
    return specs


def predict_fixed_params(rows, params):
    """Predicted (density, overlap) labels per FOV under fixed params, gate applied."""
    predictions = []
    for row in rows:
        density = score_axis(row, params["density"])
        overlap = score_axis(row, params["overlap"])
        predictions.append(apply_label_overrides(density, overlap, row, params))
    return predictions


def accuracy(rows, predictions, axis_index, label_key, levels):
    exact = off_by_one = 0
    for row, prediction in zip(rows, predictions):
        true_index = levels.index(row[label_key])
        predicted_index = levels.index(prediction[axis_index])
        exact += true_index == predicted_index
        off_by_one += abs(true_index - predicted_index) <= 1
    return exact / len(rows), off_by_one / len(rows)


def fixed_params_arm(specs, params):
    """Arm 1: label flips and accuracy under untouched v2.2 params."""
    baseline = None
    results = {}
    for name, features_csv, _, scorable in specs:
        if not scorable:
            continue
        rows = read_csv_dicts(features_csv)
        predictions = predict_fixed_params(rows, params)
        if baseline is None:
            baseline = predictions
        entry = {"n": len(rows)}
        for axis_index, (axis, label_key, levels) in enumerate(AXES):
            exact, off_by_one = accuracy(rows, predictions, axis_index, label_key, levels)
            flips = sum(1 for p, b in zip(predictions, baseline) if p[axis_index] != b[axis_index])
            entry[axis] = {"exact_match": exact, "off_by_one": off_by_one, "flips_vs_v2_2": flips}
        results[name] = entry
        print(f"  {name:<8} " + "  ".join(
            f"{axis}: exact={entry[axis]['exact_match']:.4f} off1={entry[axis]['off_by_one']:.4f} "
            f"flips={entry[axis]['flips_vs_v2_2']}" for axis, _, _ in AXES))
    return results


def write_gate3_params(nolbp_params_path):
    """A second no-LBP params file with the empty-field gate force-enabled on 3 features.

    Dropping `lbp_entropy` makes `calibrate_v2.empty_field_block` ship the gate disabled, which
    is the safe default but also silently gives up whatever the gate was buying (Nigeria 081226
    goes 6/8 -> 8/8 with it). The alternative is a 3-of-3 rule over the surviving features, and
    `empty_field_block`'s own docstring warns that a 3-of-4 rule picks up a genuinely-monolayer
    FOV. Rather than take that on faith, emit the config so `check_empty_field_gate.py` and
    `nigeria_081226.py` can measure it both ways.
    """
    data = json.loads(nolbp_params_path.read_text())
    data["empty_field_override"]["enabled"] = True
    data["empty_field_override"]["note"] = (
        "3-of-3 gate, force-enabled for measurement only. lbp_entropy is absent, so this is the "
        "weaker rule empty_field_block deliberately refuses to ship by default."
    )
    data["version"] += "-gate3"
    out_path = nolbp_params_path.with_name(nolbp_params_path.name.replace("nolbp", "nolbp-gate3"))
    out_path.write_text(json.dumps(data, indent=2))
    return out_path


def refit_arm(specs, out_dir):
    """Arm 2: refit each variant with the v2.2 procedure and record what moved."""
    results = {}
    for name, features_csv, candidates, _ in specs:
        rows = load_features(features_csv)
        axis_results = {}
        for axis, label_key, levels in AXES:
            axis_results[axis] = calibrate_axis(rows, axis, f"{axis}_ord", label_key, levels, candidates)

        params_path = out_dir / f"density_overlap_v2.2-{name}_params.json"
        write_params_json(params_path, axis_results["density"], axis_results["overlap"], len(rows))
        data = json.loads(params_path.read_text())
        data["version"] = f"v2.2-lbp-{name}"
        data["generated_from"] = str(features_csv)
        data["lbp_variant"] = name
        params_path.write_text(json.dumps(data, indent=2))

        entry = {"n": len(rows), "params": str(params_path)}
        for axis, _, levels in AXES:
            result = axis_results[axis]
            entry[axis] = {
                "cv_mean_rho": result["cv_mean_rho"],
                "oof_exact_match": result["oof_exact_match_rate"],
                "oof_off_by_one": result["oof_off_by_one_rate"],
                "thresholds": result["bucket_thresholds"],
                "lbp_weight": result["weights"].get("lbp_entropy"),
                "dropped": result["dropped_sign_unstable_features"],
                "pava_merges": [[levels[i] for i in group] for group in result["merged_bucket_groups"]],
            }
        entry["gate_enabled"] = json.loads(params_path.read_text())["empty_field_override"]["enabled"]
        results[name] = entry
        if name == "nolbp":
            entry["gate3_params"] = str(write_gate3_params(params_path))
        print(f"  {name:<8} " + "  ".join(
            f"{axis}: rho={entry[axis]['cv_mean_rho']:.3f} exact={entry[axis]['oof_exact_match']:.4f} "
            f"off1={entry[axis]['oof_off_by_one']:.4f}" for axis, _, _ in AXES)
            + f"  gate={'on' if entry['gate_enabled'] else 'OFF'}")
    return results


def write_comparison_csv(path, fixed, refit):
    fieldnames = ["variant", "axis", "fixed_exact_match", "fixed_off_by_one", "fixed_flips_vs_v2_2",
                  "refit_cv_mean_rho", "refit_oof_exact_match", "refit_oof_off_by_one",
                  "refit_lbp_weight", "refit_dropped", "refit_pava_merges", "refit_gate_enabled"]
    rows = []
    for variant, entry in refit.items():
        for axis, _, _ in AXES:
            fixed_entry = fixed.get(variant, {}).get(axis)
            rows.append({
                "variant": variant,
                "axis": axis,
                "fixed_exact_match": f"{fixed_entry['exact_match']:.6f}" if fixed_entry else "",
                "fixed_off_by_one": f"{fixed_entry['off_by_one']:.6f}" if fixed_entry else "",
                "fixed_flips_vs_v2_2": fixed_entry["flips_vs_v2_2"] if fixed_entry else "",
                "refit_cv_mean_rho": f"{entry[axis]['cv_mean_rho']:.6f}",
                "refit_oof_exact_match": f"{entry[axis]['oof_exact_match']:.6f}",
                "refit_oof_off_by_one": f"{entry[axis]['oof_off_by_one']:.6f}",
                "refit_lbp_weight": (f"{entry[axis]['lbp_weight']:.6f}"
                                     if entry[axis]["lbp_weight"] is not None else ""),
                "refit_dropped": ";".join(entry[axis]["dropped"]),
                "refit_pava_merges": ";".join("+".join(g) for g in entry[axis]["pava_merges"]),
                "refit_gate_enabled": entry["gate_enabled"],
            })
    write_csv_dicts(path, fieldnames, rows)


def main():
    parser = argparse.ArgumentParser(description="Compare LBP runtime variants against the v2.2 fit.")
    parser.add_argument("--steps", type=int, nargs="*", default=[2, 4, 6, 8],
                        help="pass no values to compare only the exact and nolbp variants, "
                             "which need no extraction pass")
    parser.add_argument("--out-dir", default=str(LBP_RUNTIME_DIR))
    parser.add_argument("--comparison-csv", default=str(LBP_COMPARISON_CSV))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = variant_specs(args.steps)
    params = json.loads(BASE_PARAMS_JSON.read_text())

    print("arm 1 -- fixed v2.2 params, only lbp_entropy swapped:")
    fixed = fixed_params_arm(specs, params)
    print("\narm 2 -- refit with the v2.2 procedure:")
    refit = refit_arm(specs, out_dir)

    write_comparison_csv(args.comparison_csv, fixed, refit)
    (out_dir / "variant-comparison.json").write_text(
        json.dumps({"fixed_params": fixed, "refit": refit}, indent=2))
    print(f"\nwrote {args.comparison_csv}")
    print(f"wrote {out_dir / 'variant-comparison.json'}")


if __name__ == "__main__":
    main()
