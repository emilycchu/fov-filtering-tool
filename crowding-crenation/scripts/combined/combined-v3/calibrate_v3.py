"""Fit the v3 crowding model: slide-grouped nested LOSO, 7-level density, cohort normalization.

Three things change from v2.2 and nothing else does. The feature extraction, the ridge, the
negative-coefficient drop loop and the PAVA threshold derivation are all v2.2's, so this is a
controlled comparison rather than a rewrite:

1. **The splits.** Nested leave-one-slide-out over 7 slide groups spanning 4 sites and 4 boxes,
   replacing FOV-stratified 5-fold over 2 slides from one box.
2. **The vocabulary.** Density gains `no cells` and `few cells` below `sparser`, so v2.2's
   empty-field pre-filter becomes a prediction instead of an assertion.
3. **The weighting.** Per-FOV weight `1 / n_fovs_in_slide`, so the two exhaustively-annotated
   slides contribute 2 of 7 units of mass rather than 648 of 1060 rows.

Plus the normalization ranges now come from the 159,716-FOV cohort rather than from the 1060
labelled rows -- see `cohort_ranges.py` for why, and for why that makes v3's raw scores and
thresholds incomparable to v2.2's as numbers.

## What is deliberately NOT here

The `hole_density` feature block and the restored partial-rho gate on the overlap axis. Both
were proposed by the design doc as the two levers on the density/overlap confound, and both are
deferred to a v3.1 round: adding a feature means the 493 cohort CSVs carry no column for it, so
its p2/p98 would have to come from the 1060 training rows -- reintroducing exactly the
training-centred normalization this version removes -- or from a re-scoring pass over 159,716
FOVs. Shipping either alongside the split, vocabulary and weighting changes would make both
uninterpretable. The report says so where it reports the confound.

Usage:
    python scripts/combined/combined-v3/calibrate_v3.py
    python scripts/combined/combined-v3/calibrate_v3.py --sample-weights none --sub-delta 0
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent
ROOT = V2_DIR.parent.parent
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(ROOT / "scripts" / "tanzania-complete-081426"))

from calibrate_v2 import (  # noqa: E402
    CANDIDATE_FEATURES,
    RIDGE_ALPHA,
    axis_separation_check,
    composite_independence,
    correlation_table,
    load_features,
    select_axis_features,
)
from _slide_common import write_csv_atomic  # noqa: E402
from _v3_common import (  # noqa: E402
    DENSITY_LEVELS,
    NON_FITTING_DENSITY_LEVELS,
    OVERLAP_LEVELS,
    collapse_to_v2_density,
)
from cohort_ranges import (  # noqa: E402
    COHORT_RANGES_JSON,
    load_cohort_columns,
    load_cohort_ranges,
)
from build_features_v3 import FEATURES_V3_CSV  # noqa: E402
from merge_labels_v3 import RESULTS_DIR, load_splits  # noqa: E402
import _v3_fit as f3  # noqa: E402
import _v3_report as rep  # noqa: E402

PARAMS_V3_JSON = RESULTS_DIR / "density_overlap_v3_params.json"
REPORT_V3_MD = RESULTS_DIR / "calibration-report-v3.md"
FOLD_DETAIL_CSV = RESULTS_DIR / "loso-folds-v3.csv"
OOF_CSV = RESULTS_DIR / "oof-predictions-v3.csv"
V22_PARAMS_JSON = (ROOT / "data" / "results" / "density-rouleaux-v2"
                   / "density_overlap_v2.2-optimized_params.json")

AXES = ("density", "overlap")
AXIS_SPEC = {
    "density": {"levels": DENSITY_LEVELS, "ord_key": "density_ord", "sub_key": "density_sub"},
    "overlap": {"levels": OVERLAP_LEVELS, "ord_key": "overlap_ord", "sub_key": "overlap_sub"},
}

FOLD_FIELDNAMES = ["axis", "held_out_slide", "n", "n_levels_present", "exact", "off_by_one",
                   "balanced_accuracy", "macro_f1", "qwk", "majority_baseline",
                   "n_features_kept", "features_kept", "dropped", "merged_groups"]
OOF_FIELDNAMES = ["fov_key", "slide", "fov_id", "density_label", "density_ord", "density_sub",
                  "density_pred_ord", "density_pred_label", "density_raw",
                  "overlap_label", "overlap_ord", "overlap_sub",
                  "overlap_pred_ord", "overlap_pred_label", "overlap_raw",
                  "cohort_empty_field_gated"]


def skip_levels_for(axis, include_no_cells):
    """Which level indices PAVA ignores. Only the density axis has a non-fitting rung."""
    if axis != "density" or include_no_cells:
        return ()
    return tuple(i for i, lv in enumerate(DENSITY_LEVELS) if lv in NON_FITTING_DENSITY_LEVELS)


def run_axis(rows, axis, oof_ranges, full_ranges, args, delta):
    """`oof_ranges` may be a callable of the excluded-slide set (fold-local normalization);
    `full_ranges` is always the all-non-test-slide ranges, because the shipped fit has no
    held-out slide to exclude."""
    spec = AXIS_SPEC[axis]
    kwargs = dict(ord_key=spec["ord_key"], sub_key=spec["sub_key"], delta=delta,
                  alpha=args.alpha, weighted=(args.sample_weights == "slide"),
                  skip_levels=skip_levels_for(axis, args.pava_include_no_cells))
    result = f3.nested_loso(rows, CANDIDATE_FEATURES, oof_ranges, len(spec["levels"]), **kwargs)
    result["full_fit"] = f3.fit_axis(rows, CANDIDATE_FEATURES, full_ranges, len(spec["levels"]),
                                    **kwargs)
    result["full_pred"], result["full_raw_score"] = f3.predict_axis(result["full_fit"], rows)
    return result


def axis_params_block(axis, fit, levels):
    return {
        "feature_names": list(fit["feature_names"]),
        "weights": {n: float(w) for n, w in zip(fit["feature_names"], fit["weights"])},
        "normalization": {n: {"min": fit["ranges"][n][0], "max": fit["ranges"][n][1]}
                          for n in fit["feature_names"]},
        "bucket_thresholds": [float(t) for t in fit["thresholds"]],
        "bucket_labels": list(levels),
        "pava_merged_groups": [[int(i) for i in g] for g in fit["merged_groups"]],
        "dropped_negative_coefficient": list(fit["dropped"]),
        "n_fit_rows": fit["n_fit_rows"],
    }


def write_params(path, rows, results, ranges_meta, args):
    slides = sorted({r["slide"] for r in rows})
    blob = {
        "version": "v3",
        "generated_from": "scripts/combined/combined-v3/calibrate_v3.py",
        "n_fovs": len(rows),
        "train_slides": slides,
        "alpha": args.alpha,
        "sample_weights": args.sample_weights,
        "sub_delta": args.sub_delta,
        "pava_include_no_cells": bool(args.pava_include_no_cells),
        "non_fitting_density_levels": sorted(NON_FITTING_DENSITY_LEVELS),
        "lbp_step": ranges_meta["lbp_step"],
        "blur_downsample": ranges_meta["blur_downsample"],
        "ranges_source": ranges_meta["generated_from"],
        "ranges_n_fovs": ranges_meta["n_fovs"],
        "ranges_n_slides": ranges_meta["n_slides"],
        "ranges_selector": ranges_meta["slides_selector"],
        "ranges_scope": args.ranges_scope,
        "density": axis_params_block("density", results["density"]["full_fit"], DENSITY_LEVELS),
        "overlap": axis_params_block("overlap", results["overlap"]["full_fit"], OVERLAP_LEVELS),
        "saturation_override": {"enabled": False},
        # v3 replaces the gate with the bottom rung of the ordinal. Recorded explicitly rather
        # than omitted so `check_empty_field_gate.py` reports "absent or disabled" deliberately.
        "empty_field_override": {"enabled": False,
                                 "replaced_by": "density bucket_labels[0:2]"},
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    return blob


def write_fold_detail(path, results):
    out = []
    for axis in AXES:
        for f in results[axis]["outer_folds"]:
            m = f["metrics"]
            out.append({
                "axis": axis, "held_out_slide": f["held_out_slide"], "n": f["n"],
                "n_levels_present": f["n_levels_present"],
                "exact": f"{m['exact']:.6f}", "off_by_one": f"{m['off_by_one']:.6f}",
                "balanced_accuracy": f"{m['balanced_accuracy']:.6f}",
                "macro_f1": f"{m['macro_f1']:.6f}", "qwk": f"{m['qwk']:.6f}",
                "majority_baseline": f"{m['majority_baseline']:.6f}",
                "n_features_kept": len(f["feature_names"]),
                "features_kept": ";".join(f["feature_names"]),
                "dropped": ";".join(f["dropped"]),
                "merged_groups": ";".join("+".join(str(i) for i in g)
                                          for g in f["merged_groups"]),
            })
    write_csv_atomic(path, FOLD_FIELDNAMES, out)


def write_oof(path, rows, results):
    out = []
    for i, r in enumerate(rows):
        row = {"fov_key": r["fov_key"], "slide": r["slide"], "fov_id": r["fov_id"],
               "cohort_empty_field_gated": r.get("cohort_empty_field_gated", "")}
        for axis in AXES:
            levels = AXIS_SPEC[axis]["levels"]
            res = results[axis]
            pred = int(res["outer_pred"][i])
            row[f"{axis}_label"] = r[f"{axis}_label"]
            row[f"{axis}_ord"] = r[f"{axis}_ord"]
            row[f"{axis}_sub"] = r[f"{axis}_sub"]
            row[f"{axis}_pred_ord"] = pred
            row[f"{axis}_pred_label"] = levels[pred]
            row[f"{axis}_raw"] = f"{res['outer_raw'][i]:.10g}"
        out.append(row)
    write_csv_atomic(path, OOF_FIELDNAMES, out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features-csv", default=str(FEATURES_V3_CSV))
    ap.add_argument("--ranges-json", default=str(COHORT_RANGES_JSON))
    ap.add_argument("--params-out", default=str(PARAMS_V3_JSON))
    ap.add_argument("--report-out", default=str(REPORT_V3_MD))
    ap.add_argument("--fold-detail-out", default=str(FOLD_DETAIL_CSV))
    ap.add_argument("--oof-out", default=str(OOF_CSV))
    ap.add_argument("--sub-delta", type=float, default=0.33,
                    help="how far below its rung a double-labelled row's target sits")
    ap.add_argument("--ablation-deltas", default="0,0.33",
                    help="comma-separated deltas to report side by side")
    ap.add_argument("--pava-include-no-cells", action="store_true",
                    help="let `no cells` contribute a PAVA median; off by default because those "
                         "rows are excluded from the fit")
    ap.add_argument("--alpha", type=float, default=RIDGE_ALPHA)
    ap.add_argument("--sample-weights", choices=("slide", "none"), default="slide")
    ap.add_argument("--ranges-scope", choices=("fold", "fixed"), default="fold",
                    help="`fold` derives each LOSO fold's p2/p98 from the cohort minus the "
                         "slides held out at that level; `fixed` uses the committed all-slide "
                         "ranges everywhere, which lets the leak be measured rather than assumed")
    ap.add_argument("--compare-params", default=str(V22_PARAMS_JSON))
    args = ap.parse_args()

    rows = load_features(args.features_csv)
    ranges, ranges_meta = load_cohort_ranges(args.ranges_json)
    splits = load_splits()

    # The ranges file and the params it was derived under must agree with the features on disk.
    # build_features_v3.py already proved the features round-trip at these knobs; this is the
    # cheap restatement so calibrate_v3 cannot be pointed at a mismatched pair.
    v22 = json.loads(Path(args.compare_params).read_text(encoding="utf-8"))
    for knob in ("lbp_step", "blur_downsample"):
        if ranges_meta[knob] != v22.get(knob):
            raise ValueError(f"{knob}: ranges JSON says {ranges_meta[knob]}, "
                             f"{Path(args.compare_params).name} says {v22.get(knob)}")

    # Fold-local normalization: each fold's p2/p98 exclude the slides held out at that level,
    # so a held-out slide never contributes to the ruler its own predictions are scored against.
    # Columns are loaded once (~12 MB) because nested LOSO needs 49 different range sets and
    # re-reading 493 CSVs for each would cost ~6.5 minutes instead of ~0.5 seconds.
    if args.ranges_scope == "fold":
        cohort = load_cohort_columns(splits, ranges_meta["slides_selector"])
        all_slide_ranges = cohort.ranges(())
        drift = max(abs(all_slide_ranges[f][i] - ranges[f][i])
                    for f in ranges for i in (0, 1))
        if drift > 1e-9:
            raise ValueError(f"recomputed all-slide ranges differ from {args.ranges_json} "
                             f"by {drift:.3e}; the two are supposed to be the same computation")
        oof_ranges = cohort.ranges
    else:
        cohort, oof_ranges = None, ranges

    results = {axis: run_axis(rows, axis, oof_ranges, ranges, args, args.sub_delta)
               for axis in AXES}

    deltas = sorted({float(d) for d in args.ablation_deltas.split(",") if d.strip() != ""}
                    | {args.sub_delta})
    ablation = {}
    for delta in deltas:
        ablation[delta] = (results if delta == args.sub_delta
                           else {axis: run_axis(rows, axis, oof_ranges, ranges, args, delta)
                                 for axis in AXES})

    correlation_rows, rho_do = correlation_table(rows)
    selections = select_axis_features(correlation_rows)
    full_indep = composite_independence(results["density"], results["overlap"], rho_do)
    oof_rho, _ = spearmanr(results["density"]["outer_raw"], results["overlap"]["outer_raw"])
    oof_indep = {"composite_rho": float(oof_rho), "manual_label_rho": float(rho_do)}
    sep_checks = [axis_separation_check(rows, results["density"]["outer_pred"],
                                       results["overlap"]["outer_pred"], min_delta=d,
                                       density_levels=DENSITY_LEVELS,
                                       overlap_levels=OVERLAP_LEVELS)
                  for d in (1, 2)]

    blob = write_params(args.params_out, rows, results, ranges_meta, args)
    write_fold_detail(args.fold_detail_out, results)
    write_oof(args.oof_out, rows, results)

    levels_by_axis = {a: AXIS_SPEC[a]["levels"] for a in AXES}
    fit_rows = f3.fitting_rows(rows)
    weighted = args.sample_weights == "slide"
    fit_weights = (f3.slide_balanced_weights(fit_rows) if weighted
                   else f3.unit_weights(fit_rows))

    lines = []
    rep.header(lines, len(rows), sorted({r["slide"] for r in rows}), ranges_meta, args)
    rep.split_table(lines, rows, splits, fit_rows, fit_weights, weighted)
    rep.pool_composition(lines, rows, levels_by_axis)
    rep.correlation_section(lines, correlation_rows, rho_do, selections)
    rep.weights_comparison(lines, blob, v22, AXES)
    rep.weight_stability(lines, results, blob, AXES)
    for axis in AXES:
        levels = AXIS_SPEC[axis]["levels"]
        rep.axis_fit_section(lines, axis, levels, results[axis]["full_fit"])
        rep.metrics_section(lines, axis, levels, results[axis])
        rep.confusion_section(lines, axis, levels,
                             [r[AXIS_SPEC[axis]["ord_key"]] for r in rows],
                             results[axis]["outer_pred"])
    rep.head_to_head(lines, rows, results, levels_by_axis, collapse_to_v2_density)
    rep.loso_summary(lines, rows, results, levels_by_axis)
    rep.gate_replacement_section(lines, rows, DENSITY_LEVELS, results["density"]["outer_pred"])
    rep.independence_section(lines, full_indep, oof_indep)
    rep.separation_check_section(lines, sep_checks)
    rep.ablation_section(lines, ablation, AXES)
    rep.footer(lines)
    Path(args.report_out).write_text("".join(lines), encoding="utf-8")

    for axis in AXES:
        p, w = results[axis]["pooled"], results[axis]["pooled_slide_weighted"]
        print(f"{axis}: OOF exact {p['exact']:.4f} (baseline {p['majority_baseline']:.4f}), "
              f"slide-weighted {w['exact']:.4f} (baseline {w['majority_baseline']:.4f}), "
              f"bal_acc {p['balanced_accuracy']:.4f}, QWK {p['qwk']:.4f}, "
              f"folds_scored {results[axis]['n_folds_scored']}/{results[axis]['n_folds']}")
    print(f"composite rho: full {full_indep['composite_rho']:.3f}, "
          f"OOF {oof_indep['composite_rho']:.3f} (manual labels {rho_do:.3f})")
    print(f"density PAVA merges (full fit): "
          f"{results['density']['full_fit']['merged_groups'] or 'none'}")
    print(f"wrote {args.params_out}\nwrote {args.report_out}\n"
          f"wrote {args.fold_detail_out}\nwrote {args.oof_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
