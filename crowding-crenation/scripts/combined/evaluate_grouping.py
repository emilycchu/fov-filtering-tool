"""How much of the reported accuracy is leakage between FOVs of the same slide?

`cross_validate` splits **FOVs**, stratified by label. 648 of the 661 calibration FOVs come from
two slides, so for almost every held-out FOV roughly 250 FOVs of the *same slide* -- same patient,
stain, scanner, session -- are in the training fold. Measured on the 271-slide cohort, within-slide
score spread (median std 0.076) is smaller than between-slide spread (0.140), so those are near
duplicates. The published figures therefore estimate "another FOV of a slide we have already seen",
while the tool is deployed **per slide**, where the question is "a slide we have never seen".

This script runs the identical fitting procedure under both splits and prints them side by side, so
the gap between them is the leakage. Nothing is refit for production and no params file is written.

Grouping is by slide, parsed from the filename -- `dataset` is not the slide (`initial-071626`
pools 13 FOVs from 9 slides). Leave-one-slide-out over the 11 groups; the two 324-FOV folds
dominate, and holding out `KTR-72502946` happens to train on exactly the 337-FOV set v2/v2.1 used.

Usage:
    python scripts/combined/evaluate_grouping.py
    python scripts/combined/evaluate_grouping.py --features data/results/.../features-v2.2.csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from _v2_common import (
    DENSITY_LEVELS,
    OVERLAP_LEVELS,
    RESULTS_DIR,
    read_csv_dicts,
    write_csv_dicts,
)
from calibrate_v2 import (
    KFOLD_SEED,
    N_FOLDS,
    RIDGE_ALPHA,
    cross_validate,
    grouped_folds,
    slide_of,
    stratified_folds,
)

DEFAULT_FEATURES = RESULTS_DIR / "features-v2.2-optimized.csv"
DEFAULT_PARAMS = RESULTS_DIR / "density_overlap_v2.2-optimized_params.json"

AXES = [("density", "density_ord", DENSITY_LEVELS),
        ("overlap", "overlap_ord", OVERLAP_LEVELS)]

OUT_CSV = RESULTS_DIR / "grouping-comparison.csv"


def numeric_rows(rows, feature_names, ord_keys):
    out = []
    for row in rows:
        try:
            clean = {name: float(row[name]) for name in feature_names}
            for key in ord_keys:
                clean[key] = int(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        clean["filename"] = row.get("filename", "")
        clean["dataset"] = row.get("dataset", "")
        out.append(clean)
    return out


def score_split(rows, feature_names, ord_key, levels, fold_of):
    pred_idx, raw, fold_rhos = cross_validate(rows, feature_names, ord_key, levels,
                                              k=N_FOLDS, seed=KFOLD_SEED, alpha=RIDGE_ALPHA,
                                              fold_of=fold_of)
    ord_values = np.array([r[ord_key] for r in rows])
    delta = np.abs(pred_idx - ord_values)
    return {
        "exact": float(np.mean(delta == 0)),
        "off_by_one": float(np.mean(delta <= 1)),
        "mean_abs_err": float(np.mean(delta)),
        "fold_rhos": fold_rhos,
        "cv_mean_rho": float(np.nanmean(fold_rhos)),
        "pred_idx": pred_idx,
        "ord_values": ord_values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--params", default=str(DEFAULT_PARAMS),
                        help="only read for its per-axis feature_names, so the comparison uses "
                             "the same features the deployed fit selected")
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)
    raw_rows = read_csv_dicts(args.features)

    all_features = sorted({n for axis, _k, _l in AXES
                           for n in params[axis]["feature_names"]})
    rows = numeric_rows(raw_rows, all_features, [k for _a, k, _l in AXES])
    print(f"features: {Path(args.features).name}  params: {params.get('version')}")
    print(f"n = {len(rows)} FOVs")

    group_of, group_names = grouped_folds(rows, slide_of)
    sizes = [int((group_of == i).sum()) for i in range(len(group_names))]
    print(f"slides: {len(group_names)}  (sizes: "
          f"{', '.join(str(s) for s in sorted(sizes, reverse=True))})\n")

    out_rows = []
    for axis, ord_key, levels in AXES:
        names = params[axis]["feature_names"]
        # Stratify on this axis's own labels, matching what calibrate_v2 does internally -- using
        # one axis's folds for both would not reproduce the published figures.
        strat_of = stratified_folds(np.array([r[ord_key] for r in rows]), N_FOLDS, KFOLD_SEED)
        strat = score_split(rows, names, ord_key, levels, strat_of)
        grouped = score_split(rows, names, ord_key, levels, group_of)

        label = "density" if axis == "density" else "Rouleaux"
        print(f"=== {label} ===")
        print(f"{'split':28s} {'exact':>8s} {'off-by-1':>9s} {'mean|err|':>10s} {'CV rho':>8s}")
        print(f"{'FOV-stratified (published)':28s} {strat['exact']:8.4f} "
              f"{strat['off_by_one']:9.4f} {strat['mean_abs_err']:10.3f} "
              f"{strat['cv_mean_rho']:8.3f}")
        print(f"{'leave-one-slide-out':28s} {grouped['exact']:8.4f} "
              f"{grouped['off_by_one']:9.4f} {grouped['mean_abs_err']:10.3f} "
              f"{grouped['cv_mean_rho']:8.3f}")
        print(f"{'gap (leakage)':28s} {grouped['exact'] - strat['exact']:+8.4f} "
              f"{grouped['off_by_one'] - strat['off_by_one']:+9.4f} "
              f"{grouped['mean_abs_err'] - strat['mean_abs_err']:+10.3f} "
              f"{grouped['cv_mean_rho'] - strat['cv_mean_rho']:+8.3f}")

        # Per-held-out-slide, since the two 324-FOV folds carry 648 of the 661 FOVs. The
        # leave-one-slide-out CV rho above is a mean over 11 folds of which 9 hold 1-4 FOVs, so
        # it is not a statistic to lean on; the two big folds are.
        print(f"\n  per held-out slide ({label}):")
        big = []
        for i, name in enumerate(group_names):
            mask = group_of == i
            n = int(mask.sum())
            delta = np.abs(grouped["pred_idx"][mask] - grouped["ord_values"][mask])
            exact = float(np.mean(delta == 0))
            off1 = float(np.mean(delta <= 1))
            rho = grouped["fold_rhos"][i]
            rho_txt = "   n/a" if np.isnan(rho) else f"{rho:+.3f}"
            print(f"    {name:52s} n={n:4d}  exact {exact:.3f}  off-by-1 {off1:.3f}  "
                  f"rho {rho_txt}")
            out_rows.append({"axis": axis, "held_out_slide": name, "n_fovs": n,
                             "grouped_exact": round(exact, 4),
                             "grouped_off_by_one": round(off1, 4),
                             "fold_rho": "" if np.isnan(rho) else round(rho, 4)})
            if n >= 100:
                big.append((exact, off1, rho))
        if big:
            print(f"    {'-- mean of the two 324-FOV folds':52s}        "
                  f"exact {np.mean([b[0] for b in big]):.3f}  "
                  f"off-by-1 {np.mean([b[1] for b in big]):.3f}  "
                  f"rho {np.nanmean([b[2] for b in big]):+.3f}")
        print()

        out_rows.append({"axis": axis, "held_out_slide": "ALL (FOV-stratified)",
                         "n_fovs": len(rows),
                         "grouped_exact": round(strat["exact"], 4),
                         "grouped_off_by_one": round(strat["off_by_one"], 4),
                         "fold_rho": round(strat["cv_mean_rho"], 4)})
        out_rows.append({"axis": axis, "held_out_slide": "ALL (leave-one-slide-out)",
                         "n_fovs": len(rows),
                         "grouped_exact": round(grouped["exact"], 4),
                         "grouped_off_by_one": round(grouped["off_by_one"], 4),
                         "fold_rho": round(grouped["cv_mean_rho"], 4)})

    write_csv_dicts(args.out_csv,
                    ["axis", "held_out_slide", "n_fovs", "grouped_exact", "grouped_off_by_one",
                     "fold_rho"],
                    out_rows)
    print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
