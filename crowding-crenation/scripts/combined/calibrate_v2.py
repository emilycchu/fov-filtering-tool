"""Calibrate the v2 density/Rouleaux composites against the merged 337-FOV manual
annotation set: marginal + partial correlation analysis (which features track which axis,
controlling for the other), ridge-regression weight fitting, stratified cross-validation,
PAVA-monotonic bucket-threshold derivation, and a full calibration report.

Usage:
    python scripts/combined/calibrate_v2.py [--features-csv PATH] [--params-out PATH]
        [--report-out PATH]
"""
import argparse
import json
import sys

import numpy as np
from scipy.stats import binomtest, spearmanr

from _v2_common import (
    AXIS_DISPLAY_NAMES,
    DEFAULT_BLUR_DOWNSAMPLE,
    DEFAULT_LBP_STEP,
    DENSITY_LEVELS,
    EMPTY_FIELD_FEATURES,
    FEATURES_CSV,
    OVERLAP_LEVELS,
    PARAMS_JSON,
    REPORT_MD,
    ROOT,
    display_level,
    read_csv_dicts,
)

sys.path.insert(0, str(ROOT / "scripts"))
from tanzania_comparison import partial_spearman  # noqa: E402

CANDIDATE_FEATURES = [
    "coverage", "otsu_separability", "saturation_score", "lbp_entropy",
    "glcm_contrast", "edge_density_unmasked", "tile_glcm_cv", "tile_glcm_patchiness",
]
MIN_PARTIAL_RHO = 0.05
# tile_glcm_cv and tile_glcm_patchiness are correlated (rho=0.63, both derived from the same
# per-tile GLCM array) and, at low regularization, this collinearity flips one of their ridge
# coefficients negative even though both have positive partial correlation with overlap --
# alpha=10 was the smallest value (checked empirically against this calibration set) that
# keeps both stably positive, letting them contribute jointly instead of one being dropped.
RIDGE_ALPHA = 10.0
N_FOLDS = 5
BOOTSTRAP_B = 1000
CI_SEED = 42
KFOLD_SEED = 7

AXES = [
    ("density", "density_ord", "density_label", DENSITY_LEVELS),
    ("overlap", "overlap_ord", "overlap_label", OVERLAP_LEVELS),
]


# --- data loading ---

def load_features(path):
    rows = read_csv_dicts(path)
    for r in rows:
        r["density_ord"] = int(r["density_ord"])
        r["overlap_ord"] = int(r["overlap_ord"])
        for f in CANDIDATE_FEATURES:
            r[f] = float(r[f])
    return rows


# --- correlation analysis ---

def correlation_table(rows):
    density = np.array([r["density_ord"] for r in rows], dtype=float)
    overlap = np.array([r["overlap_ord"] for r in rows], dtype=float)
    rho_do, _ = spearmanr(density, overlap)

    results = []
    for name in CANDIDATE_FEATURES:
        values = np.array([r[name] for r in rows], dtype=float)
        rho_fd, _ = spearmanr(values, density)
        rho_fo, _ = spearmanr(values, overlap)
        results.append({
            "feature": name,
            "marginal_density": rho_fd, "partial_density": partial_spearman(rho_fd, rho_fo, rho_do),
            "marginal_overlap": rho_fo, "partial_overlap": partial_spearman(rho_fo, rho_fd, rho_do),
        })
    return results, rho_do


def select_axis_features(correlation_rows, min_partial=MIN_PARTIAL_RHO):
    density_features, overlap_features = [], []
    for r in correlation_rows:
        pd, po = r["partial_density"], r["partial_overlap"]
        if pd > po and pd > min_partial:
            density_features.append(r["feature"])
        elif po > pd and po > min_partial:
            overlap_features.append(r["feature"])
    return {"density": density_features, "overlap": overlap_features}


# --- normalization + fitting ---

def percentile_ranges(rows, feature_names, lo=2, hi=98):
    ranges = {}
    for name in feature_names:
        values = np.array([r[name] for r in rows], dtype=float)
        lo_v, hi_v = np.percentile(values, [lo, hi])
        if hi_v <= lo_v:
            hi_v = lo_v + 1e-6
        ranges[name] = (float(lo_v), float(hi_v))
    return ranges


def normalize_matrix(rows, feature_names, ranges):
    X = np.zeros((len(rows), len(feature_names)))
    for j, name in enumerate(feature_names):
        lo_v, hi_v = ranges[name]
        raw = np.array([r[name] for r in rows], dtype=float)
        X[:, j] = np.clip((raw - lo_v) / (hi_v - lo_v), 0.0, 1.0)
    return X


def fit_ridge(X, y, alpha=RIDGE_ALPHA):
    n, k = X.shape
    A = np.column_stack([np.ones(n), X])
    reg = np.eye(k + 1) * alpha
    reg[0, 0] = 0.0
    beta = np.linalg.solve(A.T @ A + reg, A.T @ y)
    return beta[1:]


def fit_weights_stable(rows, feature_names, ord_key, alpha=RIDGE_ALPHA):
    """Fit ridge weights, iteratively dropping any feature whose coefficient comes out
    negative -- an instability artifact from correlated predictors (e.g. GLCM contrast and
    unmasked edge density are known near-duplicates, r^2=0.91, per pairwise_analysis.py),
    not a real inverse relationship, since candidates were pre-selected for positive partial
    correlation with this axis. Returns (feature_names, weights, ranges, dropped)."""
    names = list(feature_names)
    dropped = []
    coef, ranges = None, None
    while names:
        ranges = percentile_ranges(rows, names)
        X = normalize_matrix(rows, names, ranges)
        y = np.array([r[ord_key] for r in rows], dtype=float)
        coef = fit_ridge(X, y, alpha)
        neg = [n for n, c in zip(names, coef) if c < 0]
        if not neg:
            break
        dropped.extend(neg)
        names = [n for n in names if n not in neg]
    total = float(np.sum(np.abs(coef))) if len(names) else 0.0
    weights = coef / total if total > 1e-9 else np.full(len(names), 1.0 / max(len(names), 1))
    return names, weights, ranges, dropped


def raw_score(rows, feature_names, weights, ranges):
    X = normalize_matrix(rows, feature_names, ranges)
    return X @ weights


# --- cross-validation ---

def stratified_folds(ord_values, k, seed):
    rng = np.random.default_rng(seed)
    fold_of = np.full(len(ord_values), -1, dtype=int)
    for level in sorted(set(ord_values)):
        idx = np.where(np.array(ord_values) == level)[0]
        rng.shuffle(idx)
        for fold_i, chunk in enumerate(np.array_split(idx, k)):
            fold_of[chunk] = fold_i
    return fold_of


def cross_validate(rows, feature_names, ord_key, levels, k=N_FOLDS, seed=KFOLD_SEED, alpha=RIDGE_ALPHA):
    ord_values = np.array([r[ord_key] for r in rows])
    fold_of = stratified_folds(ord_values, k, seed)

    oof_pred_idx = np.full(len(rows), -1, dtype=int)
    oof_raw = np.full(len(rows), np.nan)
    fold_rhos = []

    for fold in range(k):
        test_mask = fold_of == fold
        train_mask = ~test_mask
        train_rows = [r for r, m in zip(rows, train_mask) if m]

        ranges = percentile_ranges(train_rows, feature_names)
        X_train = normalize_matrix(train_rows, feature_names, ranges)
        y_train = ord_values[train_mask].astype(float)
        coef = fit_ridge(X_train, y_train, alpha)
        total = float(np.sum(np.abs(coef)))
        weights = coef / total if total > 1e-9 else np.full(len(coef), 1.0 / len(coef))

        X_all = normalize_matrix(rows, feature_names, ranges)
        raw_all = X_all @ weights

        thresholds, _, _, _ = derive_thresholds(raw_all[train_mask], ord_values[train_mask], len(levels))
        pred_idx = np.array([sum(1 for t in thresholds if s >= t) for s in raw_all])

        oof_pred_idx[test_mask] = pred_idx[test_mask]
        oof_raw[test_mask] = raw_all[test_mask]

        rho, _ = spearmanr(raw_all[test_mask], ord_values[test_mask])
        fold_rhos.append(float(rho))

    return oof_pred_idx, oof_raw, fold_rhos


# --- bucket threshold derivation (PAVA) ---

def _pava_merge(values, weights):
    blocks = [[v, w, [i]] for i, (v, w) in enumerate(zip(values, weights))]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0] + 1e-12:
            v1, w1, idx1 = blocks[i]
            v2, w2, idx2 = blocks[i + 1]
            blocks[i:i + 2] = [[(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2, idx1 + idx2]]
            if i > 0:
                i -= 1
        else:
            i += 1
    return blocks


def derive_thresholds(raw_scores, ord_values, n_levels):
    medians, counts = [], []
    for level in range(n_levels):
        vals = raw_scores[ord_values == level]
        medians.append(float(np.median(vals)) if len(vals) else None)
        counts.append(int(len(vals)))

    present = [i for i in range(n_levels) if counts[i] > 0]
    blocks = _pava_merge([medians[i] for i in present], [counts[i] for i in present])

    corrected = [None] * n_levels
    for value, _, member_positions in blocks:
        for pos in member_positions:
            corrected[present[pos]] = value
    for i in range(n_levels):
        if corrected[i] is None:
            left = next((corrected[j] for j in range(i - 1, -1, -1) if corrected[j] is not None), None)
            right = next((corrected[j] for j in range(i + 1, n_levels) if corrected[j] is not None), None)
            corrected[i] = left if left is not None else right

    thresholds = [(corrected[i] + corrected[i + 1]) / 2 for i in range(n_levels - 1)]
    merged_groups = [[present[p] for p in member_positions] for _, _, member_positions in blocks if len(member_positions) > 1]
    return thresholds, corrected, counts, merged_groups


def bootstrap_median_ci(raw_scores, ord_values, n_levels, seed=CI_SEED, b=BOOTSTRAP_B, ci=0.90):
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (1 - ci) / 2 * 100, (1 - (1 - ci) / 2) * 100
    out = []
    for level in range(n_levels):
        vals = raw_scores[ord_values == level]
        if len(vals) == 0:
            out.append((None, None))
            continue
        boot = np.array([np.median(rng.choice(vals, size=len(vals), replace=True)) for _ in range(b)])
        out.append((float(np.percentile(boot, lo_q)), float(np.percentile(boot, hi_q))))
    return out


def bucket_index(score, thresholds):
    return sum(1 for t in thresholds if score >= t)


def confusion_matrix(true_idx, pred_idx, n_levels):
    m = np.zeros((n_levels, n_levels), dtype=int)
    for t, p in zip(true_idx, pred_idx):
        m[int(t), int(p)] += 1
    return m


# --- axis-separation check ---

def axis_separation_check(rows, oof_density_idx, oof_overlap_idx, min_delta=2):
    density_rank = np.array([r["density_ord"] for r in rows])
    overlap_rank = np.array([r["overlap_ord"] for r in rows])
    delta = density_rank - overlap_rank
    disagree_mask = np.abs(delta) >= min_delta

    pred_delta = oof_density_idx - oof_overlap_idx
    d, pd_ = delta[disagree_mask], pred_delta[disagree_mask]
    n = int(len(d))
    matches = int(np.sum(np.sign(d) == np.sign(pd_)))
    p_value = binomtest(matches, n, p=0.5, alternative="greater").pvalue if n else float("nan")
    rho = float(spearmanr(d, pd_)[0]) if n > 1 else float("nan")

    qualitative = []
    for i, r in enumerate(rows):
        if disagree_mask[i]:
            qualitative.append({
                "fov_key": r["fov_key"],
                "manual_density": r["density_label"], "manual_overlap": r["overlap_label"],
                "predicted_density": DENSITY_LEVELS[oof_density_idx[i]],
                "predicted_overlap": OVERLAP_LEVELS[oof_overlap_idx[i]],
            })
    return {"min_delta": min_delta, "n_disagreement": n, "matches": matches,
            "sign_match_rate": matches / n if n else None,
            "p_value": p_value, "spearman_rho": rho, "qualitative_rows": qualitative}


def composite_independence(density_result, overlap_result, rho_do):
    """How correlated the two fitted composites are, against how correlated the manual labels
    actually are. Lives here rather than in a calibrate_v2.<N>.py because those filenames are
    not importable (the dot makes them invalid module names), so every refit script that wants
    it would otherwise have to restate it."""
    rho, _ = spearmanr(density_result["full_raw_score"], overlap_result["full_raw_score"])
    return {"composite_rho": float(rho), "manual_label_rho": float(rho_do)}


def dataset_feature_summary(rows, feature_names):
    """Median of each feature by source dataset -- a coarse cross-slide/cross-stain check.
    Raw pixel/intensity features (coverage, GLCM contrast, edge density) are sensitive to
    staining protocol and scanner, so a large gap here between datasets is a real
    generalization risk worth surfacing, not just a label-mix difference."""
    summary = {}
    for dataset in sorted({r["dataset"] for r in rows}):
        sub = [r for r in rows if r["dataset"] == dataset]
        summary[dataset] = {
            "n": len(sub),
            **{name: float(np.median([r[name] for r in sub])) for name in feature_names},
        }
    return summary


# --- orchestration ---

def calibrate_axis(rows, axis, ord_key, label_key, levels, candidate_features):
    names, weights, ranges, dropped = fit_weights_stable(rows, candidate_features, ord_key)

    ord_values = np.array([r[ord_key] for r in rows])
    full_raw = raw_score(rows, names, weights, ranges)
    thresholds, centroids, bucket_counts, merged_groups = derive_thresholds(full_raw, ord_values, len(levels))
    cis = bootstrap_median_ci(full_raw, ord_values, len(levels))

    oof_pred_idx, oof_raw, fold_rhos = cross_validate(rows, names, ord_key, levels)
    confusion = confusion_matrix(ord_values, oof_pred_idx, len(levels))
    exact_match = float(np.trace(confusion) / confusion.sum())
    off_by_one = float(np.mean(np.abs(ord_values - oof_pred_idx) <= 1))
    oof_rho = float(spearmanr(oof_raw, ord_values)[0])

    return {
        "axis": axis,
        "feature_names": names,
        "weights": {n: float(w) for n, w in zip(names, weights)},
        "normalization": {n: list(ranges[n]) for n in names},
        "bucket_thresholds": [float(t) for t in thresholds],
        "bucket_labels": levels,
        "dropped_sign_unstable_features": dropped,
        "bucket_centroids": centroids,
        "bucket_counts": bucket_counts,
        "bucket_centroid_ci90": cis,
        "merged_bucket_groups": merged_groups,
        "cv_fold_rho": fold_rhos,
        "cv_mean_rho": float(np.mean(fold_rhos)),
        "cv_std_rho": float(np.std(fold_rhos)),
        "oof_overall_rho": oof_rho,
        "oof_exact_match_rate": exact_match,
        "oof_off_by_one_rate": off_by_one,
        "oof_confusion_matrix": confusion.tolist(),
        "oof_pred_idx": oof_pred_idx,
        "full_raw_score": full_raw,
    }


def empty_field_block(density_result):
    """The empty-field gate config, with its floors taken from this fit's density p2 values so
    a refit retunes the gate automatically (see apply_empty_field_override in _v2_common.py).

    All four features are required. A 3-of-4 gate is not a milder version of the same rule --
    measured on the v2.2 set it picks up a genuinely-monolayer FOV, where 4-of-4 picks up only
    true sparser fields. So if feature selection drops one (the original 5-feature v2 fit has
    no otsu_separability, for instance), the gate ships *disabled* with the floors it could
    derive, rather than silently enforcing a weaker rule. Enabling it is then a deliberate
    decision, taken after re-running scripts/combined/check_empty_field_gate.py.
    """
    available = [n for n in EMPTY_FIELD_FEATURES if n in density_result["normalization"]]
    missing = [n for n in EMPTY_FIELD_FEATURES if n not in density_result["normalization"]]
    if missing:
        print(f"warning: empty-field gate needs all of {list(EMPTY_FIELD_FEATURES)}, but feature "
              f"selection dropped {missing} from the density axis -- writing the gate DISABLED. "
              "A 3-of-4 rule is measurably worse, so do not just flip `enabled` without re-checking.")
    return {
        "enabled": not missing,
        "rule": "all_below",
        "thresholds": {n: density_result["normalization"][n][0] for n in available},
        "density_label": DENSITY_LEVELS[0],
        "overlap_label": OVERLAP_LEVELS[0],
    }


def write_params_json(out_path, density_result, overlap_result, n_fovs,
                      lbp_step=DEFAULT_LBP_STEP, blur_downsample=DEFAULT_BLUR_DOWNSAMPLE):
    """Both runtime knobs are recorded, not just used: score_fov_v2.py reads them back, so a
    fit made on subsampled LBP entropy or a downsampled illumination background can only ever
    be scored the same way. Omitting them would let calibration and inference silently differ."""
    def axis_block(res, levels):
        return {
            "feature_names": res["feature_names"],
            "weights": res["weights"],
            "normalization": {n: {"min": r[0], "max": r[1]} for n, r in res["normalization"].items()},
            "bucket_thresholds": res["bucket_thresholds"],
            "bucket_labels": levels,
        }

    params = {
        "version": "v2",
        "generated_from": "data/results/density-rouleaux-v2/features.csv",
        "n_fovs": n_fovs,
        "lbp_step": lbp_step,
        "blur_downsample": blur_downsample,
        "density": axis_block(density_result, DENSITY_LEVELS),
        "overlap": axis_block(overlap_result, OVERLAP_LEVELS),
        "saturation_override": {
            "density": {"enabled": False, "feature": "saturation_score", "threshold": None, "max_label": "very dense"},
            "overlap": {"enabled": False, "feature": "saturation_score", "threshold": None, "max_label": "heavy rouleaux"},
        },
        "empty_field_override": empty_field_block(density_result),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(params, f, indent=2)


def _fmt_conf(matrix, levels):
    header = "| manual \\ predicted | " + " | ".join(display_level(l) for l in levels) + " |\n"
    header += "|---" * (len(levels) + 1) + "|\n"
    lines = [header]
    for i, level in enumerate(levels):
        lines.append("| " + display_level(level) + " | " + " | ".join(str(x) for x in matrix[i]) + " |\n")
    return "".join(lines)


def write_report(out_path, correlation_rows, rho_do, selections, density_result, overlap_result, sep_checks,
                  dataset_summary, n_fovs):
    lines = []
    lines.append("# Density + Rouleaux v2 calibration report\n\n")
    lines.append(f"n = {n_fovs} FOVs (13 from `initial-dataset-071626` + 324 from `tanzania-073026`). "
                 f"Manual density-vs-Rouleaux confound: Spearman rho = {rho_do:.3f}.\n\n")

    lines.append("## Marginal + partial correlation (all candidate features)\n\n")
    lines.append("| feature | marginal density | partial density | marginal Rouleaux | partial Rouleaux |\n")
    lines.append("|---|---|---|---|---|\n")
    for r in correlation_rows:
        lines.append(f"| {r['feature']} | {r['marginal_density']:.3f} | {r['partial_density']:.3f} | "
                     f"{r['marginal_overlap']:.3f} | {r['partial_overlap']:.3f} |\n")
    lines.append(f"\nFeature-selection rule: assign a feature to whichever axis has the higher *partial* "
                 f"correlation, provided it exceeds {MIN_PARTIAL_RHO}; otherwise excluded from both composites.\n\n")
    lines.append(f"- Density candidate features: {', '.join(selections['density']) or '(none)'}\n")
    lines.append(f"- Rouleaux candidate features: {', '.join(selections['overlap']) or '(none)'}\n\n")

    for name, result, levels in [("Density", density_result, DENSITY_LEVELS), ("Rouleaux", overlap_result, OVERLAP_LEVELS)]:
        lines.append(f"## {name} composite\n\n")
        if result["dropped_sign_unstable_features"]:
            lines.append(f"Dropped for sign instability (negative ridge coefficient despite positive partial "
                         f"correlation -- a multicollinearity artifact, not a real inverse relationship): "
                         f"{', '.join(result['dropped_sign_unstable_features'])}\n\n")
        lines.append("Fitted weights (ridge regression on percentile-normalized features, refit on the full "
                     "calibration set after cross-validation below):\n\n")
        lines.append("| feature | weight | range (2nd-98th pct) |\n|---|---|---|\n")
        for n in result["feature_names"]:
            lo, hi = result["normalization"][n]
            lines.append(f"| {n} | {result['weights'][n]:.3f} | [{lo:.4g}, {hi:.4g}] |\n")

        lines.append(f"\n**Cross-validation** ({N_FOLDS}-fold, stratified by bucket): "
                     f"per-fold raw-score Spearman rho = {[round(r, 3) for r in result['cv_fold_rho']]}, "
                     f"mean={result['cv_mean_rho']:.3f} (std={result['cv_std_rho']:.3f}). "
                     f"Out-of-fold overall rho={result['oof_overall_rho']:.3f}.\n\n")
        lines.append(f"Out-of-fold exact-match rate: {result['oof_exact_match_rate']:.1%}. "
                     f"Off-by-one rate: {result['oof_off_by_one_rate']:.1%}.\n\n")
        lines.append("Out-of-fold confusion matrix (rows=manual, cols=predicted):\n\n")
        lines.append(_fmt_conf(result["oof_confusion_matrix"], levels))

        lines.append(f"\n**Bucket thresholds** (median-per-bucket, PAVA-corrected for monotonicity, "
                     f"midpoint cut points), with bucket counts and bootstrap 90% CI on each bucket's "
                     f"median raw score:\n\n")
        lines.append("| bucket | n | median raw score | 90% CI |\n|---|---|---|---|\n")
        for i, level in enumerate(levels):
            ci = result["bucket_centroid_ci90"][i]
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "n/a"
            lines.append(f"| {display_level(level)} | {result['bucket_counts'][i]} | {result['bucket_centroids'][i]:.3f} | {ci_str} |\n")
        lines.append(f"\nThresholds: {[round(t, 3) for t in result['bucket_thresholds']]}\n\n")
        if result["merged_bucket_groups"]:
            groups = [[display_level(levels[i]) for i in g] for g in result["merged_bucket_groups"]]
            lines.append(f"**Note:** PAVA merged the following adjacent buckets because their median raw "
                         f"scores were not monotonic at n={n_fovs} -- an honest finding that these buckets "
                         f"aren't cleanly separable by the current features/sample size, not a fitting bug: "
                         f"{groups}\n\n")

    lines.append("## Axis-separation check\n\n")
    lines.append("Among FOVs where manual density-rank and Rouleaux-rank disagree by at least `min_delta` levels "
                 "(the dense-but-not-Rouleauxed / Rouleauxed-but-not-dense cases this tool is meant to separate): "
                 "do the out-of-fold predicted scores diverge in the same direction, at a better-than-chance rate?\n\n")
    lines.append("| min |delta| | n | sign matches | match rate | binomial p (one-sided, vs. 0.5) | Spearman rho (predicted vs. manual delta) |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for sep_check in sep_checks:
        sign_match_str = f"{sep_check['sign_match_rate']:.1%}" if sep_check["sign_match_rate"] is not None else "n/a"
        lines.append(f"| {sep_check['min_delta']} | {sep_check['n_disagreement']} | {sep_check['matches']} | "
                     f"{sign_match_str} | {sep_check['p_value']:.4g} | {sep_check['spearman_rho']:.3f} |\n")

    strict = sep_checks[-1]
    lines.append(f"\nQualitative spot-check (manual vs. out-of-fold predicted, |delta|>={strict['min_delta']} subset):\n\n")
    lines.append("| FOV | manual density | manual Rouleaux | predicted density | predicted Rouleaux |\n|---|---|---|---|---|\n")
    for row in strict["qualitative_rows"]:
        lines.append(f"| {row['fov_key']} | {display_level(row['manual_density'])} | {display_level(row['manual_overlap'])} | "
                     f"{display_level(row['predicted_density'])} | {display_level(row['predicted_overlap'])} |\n")

    lines.append("\n## Known limitations\n\n")
    lines.append("**Cross-slide / cross-stain generalization risk.** All candidate features are raw pixel/intensity "
                 "statistics (Otsu coverage, GLCM contrast, edge density, LBP entropy), which are sensitive to "
                 "staining protocol, scanner, and illumination -- not just true cell density. Median feature values "
                 "differ substantially between the two source datasets:\n\n")
    lines.append("| dataset | n | " + " | ".join(CANDIDATE_FEATURES) + " |\n")
    lines.append("|---" * (len(CANDIDATE_FEATURES) + 2) + "|\n")
    for dataset, vals in dataset_summary.items():
        lines.append(f"| {dataset} | {vals['n']} | " + " | ".join(f"{vals[f]:.3g}" for f in CANDIDATE_FEATURES) + " |\n")
    lines.append("\nConcretely: `dpc-051-LB-D3-...` (initial-071626, Liberia slide, manually labeled *monolayer*) "
                 "has Otsu coverage 0.79 -- roughly 4x a typical Tanzania monolayer FOV (~0.18-0.21) -- and the "
                 "density composite accordingly (mis)scores it as very dense. Only 13 of 337 calibration FOVs are "
                 "non-Tanzania, so this pipeline is validated mainly for the Tanzania KTR-72502948 slide/stain; "
                 "**spot-check `score_fov_v2.py` output against a handful of manual labels before trusting it on "
                 "a new slide or stain, and expect to refit if there's a systematic offset.**\n\n")
    lines.append("**Rouleaux composite is markedly weaker than density** (CV mean rho "
                 f"{overlap_result['cv_mean_rho']:.2f} vs. {density_result['cv_mean_rho']:.2f}) and relies on only "
                 "two features (both new tile-heterogeneity measures) -- expected, since prior work in this repo "
                 "already established Rouleaux is intrinsically harder to capture with instance-free texture "
                 "statistics, but it means Rouleaux predictions should be trusted less than density predictions, "
                 "especially at the upper end where PAVA had to merge three buckets together.\n\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def main():
    parser = argparse.ArgumentParser(description="Calibrate the v2 density/Rouleaux composites.")
    parser.add_argument("--features-csv", default=str(FEATURES_CSV))
    parser.add_argument("--params-out", default=str(PARAMS_JSON))
    parser.add_argument("--report-out", default=str(REPORT_MD))
    args = parser.parse_args()

    rows = load_features(args.features_csv)
    correlation_rows, rho_do = correlation_table(rows)
    selections = select_axis_features(correlation_rows)

    density_result = calibrate_axis(rows, "density", "density_ord", "density_label", DENSITY_LEVELS, selections["density"])
    overlap_result = calibrate_axis(rows, "overlap", "overlap_ord", "overlap_label", OVERLAP_LEVELS, selections["overlap"])

    sep_checks = [
        axis_separation_check(rows, density_result["oof_pred_idx"], overlap_result["oof_pred_idx"], min_delta=1),
        axis_separation_check(rows, density_result["oof_pred_idx"], overlap_result["oof_pred_idx"], min_delta=2),
    ]
    dataset_summary = dataset_feature_summary(rows, CANDIDATE_FEATURES)

    from pathlib import Path
    write_params_json(Path(args.params_out), density_result, overlap_result, len(rows))
    write_report(Path(args.report_out), correlation_rows, rho_do, selections, density_result, overlap_result,
                 sep_checks, dataset_summary, len(rows))

    print(f"density: features={density_result['feature_names']}, cv_mean_rho={density_result['cv_mean_rho']:.3f}, "
         f"exact_match={density_result['oof_exact_match_rate']:.1%}")
    print(f"overlap: features={overlap_result['feature_names']}, cv_mean_rho={overlap_result['cv_mean_rho']:.3f}, "
         f"exact_match={overlap_result['oof_exact_match_rate']:.1%}")
    for sep_check in sep_checks:
        print(f"axis-separation (|delta|>={sep_check['min_delta']}): {sep_check['matches']}/{sep_check['n_disagreement']} "
              f"sign matches, p={sep_check['p_value']:.4g}")
    print(f"wrote {args.params_out}")
    print(f"wrote {args.report_out}")


if __name__ == "__main__":
    main()
