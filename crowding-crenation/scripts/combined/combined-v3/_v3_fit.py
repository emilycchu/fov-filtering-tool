"""v3's fitting layer: slide-balanced weights, weighted PAVA, ordinal metrics, nested LOSO.

Everything here is a weighted or vocabulary-generalised variant of something in
`calibrate_v2.py`. What is reused from there verbatim, and must stay that way: `_pava_merge`
(its second argument is already block mass, so v3 passes summed sample weights where v2 passes
counts), `normalize_matrix`, `fit_ridge` (now with `sample_weight=`), `grouped_folds`,
`bucket_index`, `confusion_matrix`, `correlation_table`, `select_axis_features`,
`composite_independence`, `load_features`.

## Why the weights

Pooled naively, the two exhaustively-annotated slides supply 648 of 1060 training rows (61%),
and their median FOV density sits at the 77th-93rd percentile of the cohort they score. Fitting
on that pool re-centres v3 exactly where v2.2 was mis-centred. Per-FOV weight
`1 / n_fovs_in_slide` makes each of the 7 slide groups contribute one unit of mass instead of
one unit per FOV, which is the whole point of grouping by slide in the first place.

## Why ridge and PAVA share rows

Measured, not assumed. Cross-fitting the thresholds -- deriving them from out-of-fold scores
rather than in-sample ones -- collapses density exact-match from 0.5522 to 0.2436 and overlap
from 0.6324 to 0.5023, because the composite renormalises its weights and (in v2.2) its ranges
per fold, so out-of-fold scores are a *mixture* of per-fold score scales and thresholds derived
from that mixture do not apply to the single-scale score they are used on. The honesty that
separation is meant to buy is already provided by the outer LOSO loop, which never lets the
thresholds see the held-out slide. See the v3 design doc for the trigger that would revisit this
(a disjoint calibration slice holding >=40 FOVs in all levels on both axes; v3 lands at 10
slides, the trigger needs ~15).

## Where the normalization comes from

Not from `rows`. The p2/p98 ranges are derived from the 159,716-FOV cohort and passed in, either
as a fixed dict or as a callable of the excluded-slide set. The callable form is the default and
makes the normalization **fold-local**: an outer fold's ranges exclude its held-out slide, an
inner fold's exclude both. The training rows never determine their own feature scaling, and no
slide contributes to the ruler its own predictions are measured against.

## Two places v3 raises where v2 shrugs

- `fit_weights_stable_w` raises if the negative-coefficient drop loop empties the feature set.
  v2 returns a zero-feature composite (`calibrate_v2.py:151-152` hands back a uniform weight
  vector over an empty name list), which would give PAVA a column of zeros and produce
  thresholds that look derived but are not.
- `ordinal_metrics` reports per-class F1 for an absent class as `None`, never `0.0`, and
  macro-F1 averages over present classes only. Two of v3's outer folds hold a test set that is
  100% `no rouleaux`, so 4-5 overlap classes are absent; scoring those as zero would drag the
  fold's macro-F1 down by 1/5 per absent class and make the number a count of missing classes
  rather than a measure of the model.
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent
sys.path.insert(0, str(V2_DIR))

from calibrate_v2 import (  # noqa: E402
    RIDGE_ALPHA,
    _pava_merge,
    bucket_index,
    fit_ridge,
    normalize_matrix,
)
from _v3_common import NON_FITTING_DENSITY_LEVELS, ordinal_target  # noqa: E402

SLIDE_KEY = "slide"


# --- sample weights ---------------------------------------------------------------------

def slide_balanced_weights(rows, slide_key=SLIDE_KEY):
    """`1 / n_fovs_in_slide`, counted over exactly the rows passed in.

    Counted over the *fitting* rows, not the whole pool: if `no cells` rows are excluded from
    the fit, a slide's weight must reflect how many rows it actually contributes, or the slides
    with excluded rows end up over-weighted.
    """
    counts = Counter(r[slide_key] for r in rows)
    return np.array([1.0 / counts[r[slide_key]] for r in rows], dtype=float)


def unit_weights(rows):
    return np.ones(len(rows), dtype=float)


# --- weighted order statistics ----------------------------------------------------------

def weighted_percentile(values, weights, q):
    """The q-th percentile (q in [0, 100]) of `values` under `weights`.

    Linear interpolation on the weighted CDF evaluated at the midpoint of each observation's
    mass. At **q=50 with equal weights this reproduces `np.median` exactly** (verified over 2000
    random arrays, odd and even n), which is the property the whole v2-equivalence argument rests
    on: `derive_thresholds_w` with unit weights reproduces `calibrate_v2.derive_thresholds`.

    At other quantiles it does *not* match `np.percentile` even with equal weights, because numpy
    places the i-th of n sorted values at position i/(n-1) while this places it at the midpoint
    of its own mass. Neither is more correct -- there is no single percentile convention -- but do
    not treat this as a drop-in `np.percentile`. Nothing in v3 needs it to be: the only caller is
    `weighted_median`, and the p2/p98 normalization ranges come from `cohort-ranges.json`, not
    from here.

    Note it returns an interpolated value rather than a data point, so a heavily up-weighted
    outlier pulls the result toward itself instead of snapping to it.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError("weighted_percentile of an empty array")
    weights = np.ones_like(values) if weights is None else np.asarray(weights, dtype=float)
    if np.any(weights < 0):
        raise ValueError("negative sample weight")
    order = np.argsort(values, kind="mergesort")
    v, w = values[order], weights[order]
    total = w.sum()
    if total <= 0:
        raise ValueError("sample weights sum to zero")
    cum = np.cumsum(w) - 0.5 * w
    return float(np.interp(q / 100.0 * total, cum, v))


def weighted_median(values, weights=None):
    return weighted_percentile(values, weights, 50.0)


# --- the fit ----------------------------------------------------------------------------

def fit_weights_stable_w(rows, feature_names, targets, ranges, alpha=RIDGE_ALPHA,
                         sample_weight=None):
    """v2's iterative negative-coefficient drop loop, with three v3 changes.

    - `ranges` are passed in (cohort-derived) rather than refit from `rows`. v2 refit them from
      each fold's training labels, which rescaled the composite fold to fold; here they come
      from 159,716 unlabelled cohort FOVs, so the score scale is stable across folds to within
      ~1% and a single set of PAVA thresholds is meaningful. See `nested_loso` for why that ~1%
      remains rather than being exactly zero.
    - `targets` is an explicit array, which is where `ordinal_target`'s sub-level nudge enters.
    - Raises rather than returning an empty composite. See the module docstring.

    -> (surviving_names, weights_summing_to_one, dropped)
    """
    names = list(feature_names)
    if not names:
        raise ValueError("fit_weights_stable_w called with no candidate features")
    y = np.asarray(targets, dtype=float)
    dropped, coef = [], None
    while names:
        X = normalize_matrix(rows, names, ranges)
        coef = fit_ridge(X, y, alpha, sample_weight)
        neg = [n for n, c in zip(names, coef) if c < 0]
        if not neg:
            break
        dropped.extend(neg)
        names = [n for n in names if n not in neg]
    if not names:
        raise ValueError(f"the negative-coefficient drop loop emptied the feature set; "
                         f"dropped in order: {dropped}")
    total = float(np.sum(np.abs(coef)))
    if total <= 1e-9:
        raise ValueError(f"fitted coefficients are all ~0 over {names}; the composite would be "
                         f"a column of zeros and its thresholds meaningless")
    return names, coef / total, dropped


def raw_score_w(rows, feature_names, weights, ranges):
    return normalize_matrix(rows, feature_names, ranges) @ weights


def derive_thresholds_w(raw_scores, ord_values, n_levels, weights=None, skip_levels=()):
    """Weighted per-level medians -> PAVA -> midpoint thresholds.

    Mirrors `calibrate_v2.derive_thresholds` and reuses `_pava_merge` unchanged; the only
    differences are that the per-level central value is a weighted median and the per-level mass
    handed to PAVA is summed weight rather than a count.

    `skip_levels` drops a level from the medians and from PAVA entirely. v3 uses it for
    `no cells`, which is excluded from the fit and scored only as the gate-replacement check.
    A skipped level is handled by the same path as a genuinely absent one -- its corrected value
    is interpolated from its neighbours -- so the returned threshold list still has
    `n_levels - 1` entries and `bucket_index` still spans the full vocabulary.

    -> (thresholds, corrected_medians, mass_per_level, merged_groups)
    """
    raw_scores = np.asarray(raw_scores, dtype=float)
    ord_values = np.asarray(ord_values)
    weights = np.ones(len(raw_scores)) if weights is None else np.asarray(weights, dtype=float)
    skip = set(skip_levels)

    medians, mass = [], []
    for level in range(n_levels):
        sel = (ord_values == level) & np.isfinite(raw_scores)
        if level in skip or not np.any(sel):
            medians.append(None)
            mass.append(0.0)
            continue
        medians.append(weighted_median(raw_scores[sel], weights[sel]))
        mass.append(float(weights[sel].sum()))

    present = [i for i in range(n_levels) if mass[i] > 0]
    if not present:
        raise ValueError("no level has any mass; cannot derive thresholds")
    blocks = _pava_merge([medians[i] for i in present], [mass[i] for i in present])

    corrected = [None] * n_levels
    for value, _, member_positions in blocks:
        for pos in member_positions:
            corrected[present[pos]] = value
    for i in range(n_levels):
        if corrected[i] is None:
            left = next((corrected[j] for j in range(i - 1, -1, -1)
                         if corrected[j] is not None), None)
            right = next((corrected[j] for j in range(i + 1, n_levels)
                          if corrected[j] is not None), None)
            corrected[i] = left if left is not None else right

    thresholds = [(corrected[i] + corrected[i + 1]) / 2 for i in range(n_levels - 1)]
    merged_groups = [[present[p] for p in member_positions]
                     for _, _, member_positions in blocks if len(member_positions) > 1]
    return thresholds, corrected, mass, merged_groups


# --- metrics ----------------------------------------------------------------------------

def ordinal_metrics(true_idx, pred_idx, n_levels, weights=None):
    """Exact, off-by-one, balanced accuracy, macro-F1, per-class F1, QWK, majority baseline.

    `weights` makes every metric slide-weighted; pass None for the unweighted arm. Reported side
    by side because they answer different questions: unweighted is "of the FOVs we labelled",
    slide-weighted is "of the slides we would deploy on", and the pool is 61% two slides.

    Per-class F1 is `None` for a class with no true instances, and macro-F1 averages over the
    present classes only -- see the module docstring for why a zero would be misleading here.
    """
    true_idx = np.asarray(true_idx, dtype=int)
    pred_idx = np.asarray(pred_idx, dtype=int)
    w = np.ones(len(true_idx)) if weights is None else np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        raise ValueError("metric weights sum to zero")

    exact = float(w[true_idx == pred_idx].sum() / total)
    off_by_one = float(w[np.abs(true_idx - pred_idx) <= 1].sum() / total)

    # Weighted confusion, built directly rather than via `confusion_matrix`, which counts rows.
    obs = np.zeros((n_levels, n_levels), dtype=float)
    for t, p, wi in zip(true_idx, pred_idx, w):
        obs[t, p] += wi

    support = obs.sum(axis=1)
    predicted = obs.sum(axis=0)
    present = [i for i in range(n_levels) if support[i] > 0]

    per_class_f1, recalls = [], []
    for i in range(n_levels):
        if support[i] <= 0:
            per_class_f1.append(None)
            continue
        tp = obs[i, i]
        recall = tp / support[i]
        precision = tp / predicted[i] if predicted[i] > 0 else 0.0
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
        per_class_f1.append(float(f1))
        recalls.append(float(recall))

    balanced_accuracy = float(np.mean(recalls)) if recalls else float("nan")
    macro_f1 = float(np.mean([per_class_f1[i] for i in present])) if present else float("nan")

    # Quadratic weighted kappa over the full vocabulary: the penalty matrix is defined by the
    # ordinal distance, so absent classes contribute nothing to either observed or expected and
    # do not need special-casing the way F1 does.
    pen = np.array([[(i - j) ** 2 for j in range(n_levels)] for i in range(n_levels)],
                   dtype=float) / ((n_levels - 1) ** 2)
    exp = np.outer(support, predicted) / total
    denom = float(np.sum(pen * exp))
    qwk = float("nan") if denom <= 0 else float(1.0 - np.sum(pen * obs) / denom)

    majority_level = int(np.argmax(support))
    majority_baseline = float(support[majority_level] / total)

    return {
        "n": int(len(true_idx)),
        "weighted": weights is not None,
        "exact": exact,
        "off_by_one": off_by_one,
        "balanced_accuracy": balanced_accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "qwk": qwk,
        "majority_baseline": majority_baseline,
        "majority_level": majority_level,
        "support": [float(s) for s in support],
        "n_levels_present": len(present),
    }


# --- LOSO -------------------------------------------------------------------------------

def fitting_rows(rows, non_fitting_levels=NON_FITTING_DENSITY_LEVELS):
    """The rows a composite may be fitted on: everything except the excluded density levels.

    Applied to **both** axes, not just density. A `no cells` row's overlap label is a forced
    default rather than an observation (`parse_tanzania_tags_v3` refuses to accept an overlap
    tag on `no cells`), so including those rows would train the overlap axis on 13 fabricated
    `no rouleaux` labels.
    """
    return [r for r in rows if r["density_label"] not in non_fitting_levels]


def _targets(rows, ord_key, sub_key, delta):
    return np.array([ordinal_target(int(r[ord_key]), r.get(sub_key) == "low", delta)
                     for r in rows], dtype=float)


def fit_axis(train_rows, feature_names, ranges, n_levels, *, ord_key, sub_key, delta,
             alpha=RIDGE_ALPHA, weighted=True, skip_levels=()):
    """One ridge + PAVA fit on `train_rows`. -> dict of everything needed to score new rows."""
    fit_rows = fitting_rows(train_rows)
    if not fit_rows:
        raise ValueError("no fitting rows left after excluding the non-fitting density levels")
    w = slide_balanced_weights(fit_rows) if weighted else None

    names, weights, dropped = fit_weights_stable_w(
        fit_rows, feature_names, _targets(fit_rows, ord_key, sub_key, delta), ranges,
        alpha=alpha, sample_weight=w)

    raw = raw_score_w(fit_rows, names, weights, ranges)
    ords = np.array([int(r[ord_key]) for r in fit_rows])
    thresholds, corrected, mass, merged = derive_thresholds_w(
        raw, ords, n_levels, weights=w, skip_levels=skip_levels)

    return {"feature_names": names, "weights": weights, "ranges": ranges,
            "thresholds": thresholds, "corrected_medians": corrected, "level_mass": mass,
            "merged_groups": merged, "dropped": dropped, "n_fit_rows": len(fit_rows)}


def predict_axis(fit, rows):
    """-> (predicted level indices, raw scores) for `rows` under a frozen `fit`."""
    raw = raw_score_w(rows, fit["feature_names"], fit["weights"], fit["ranges"])
    return np.array([bucket_index(s, fit["thresholds"]) for s in raw]), raw


def resolve_ranges(ranges, excluded):
    """`ranges` may be a concrete dict or a callable of the excluded-slide set.

    The callable form is what makes the normalization fold-local: each fold derives its p2/p98
    from the cohort *minus* the slides held out at that level, so a held-out slide never
    contributes to the ruler its own predictions are measured against. See `nested_loso`.
    """
    return ranges(frozenset(excluded)) if callable(ranges) else ranges


def loso(rows, feature_names, ranges, n_levels, *, ord_key, sub_key, delta,
         alpha=RIDGE_ALPHA, weighted=True, skip_levels=(), slide_key=SLIDE_KEY,
         base_excluded=frozenset()):
    """Leave-one-slide-out over `rows`. -> (oof_pred_idx, oof_raw, per_fold list).

    One fold per slide, folds in `sorted(set(slides))` order, so the whole thing is
    deterministic without an RNG -- which is what lets `verify_v3_fit.py` diff a re-run against
    the committed params rather than merely checking it is in range.

    `base_excluded` is the set of slides already held out by an enclosing loop; each fold's
    ranges are derived excluding `base_excluded | {held-out slide}`.
    """
    slides = sorted({r[slide_key] for r in rows})
    oof_pred = np.full(len(rows), -1, dtype=int)
    oof_raw = np.full(len(rows), np.nan)
    per_fold = []

    for slide in slides:
        test_mask = np.array([r[slide_key] == slide for r in rows])
        train_rows = [r for r, m in zip(rows, test_mask) if not m]
        test_rows = [r for r, m in zip(rows, test_mask) if m]
        fold_ranges = resolve_ranges(ranges, set(base_excluded) | {slide})
        fit = fit_axis(train_rows, feature_names, fold_ranges, n_levels, ord_key=ord_key,
                       sub_key=sub_key, delta=delta, alpha=alpha, weighted=weighted,
                       skip_levels=skip_levels)
        pred, raw = predict_axis(fit, test_rows)
        oof_pred[test_mask] = pred
        oof_raw[test_mask] = raw

        true = np.array([int(r[ord_key]) for r in test_rows])
        per_fold.append({
            "held_out_slide": slide,
            "n": len(test_rows),
            "n_levels_present": len(set(true.tolist())),
            "feature_names": fit["feature_names"],
            "dropped": fit["dropped"],
            "merged_groups": fit["merged_groups"],
            "metrics": ordinal_metrics(true, pred, n_levels),
        })
    return oof_pred, oof_raw, per_fold


def nested_loso(rows, feature_names, ranges, n_levels, *, ord_key, sub_key, delta,
                alpha=RIDGE_ALPHA, weighted=True, skip_levels=(), slide_key=SLIDE_KEY):
    """Outer LOSO for reporting; inner LOSO inside each outer fold for the fitted decisions.

    7 outer folds x 6 inner folds = 42 inner fits per config. Affordable because each fit is a
    <=9x9 `np.linalg.solve` over <=1060 rows, so the whole thing is dominated by reading the CSV.

    The outer loop's OOF predictions are the reported generalisation number. The inner loop
    never touches the outer held-out slide, so its estimate is what any per-config choice must
    be made on -- reporting `inner` and then picking on `outer` would be selecting on the test
    set one slide at a time.

    **The normalization is fold-local too.** Pass `ranges` as a callable and each fold derives
    its p2/p98 from the cohort minus the slides held out at that level: `{outer}` for an outer
    fold, `{outer, inner}` for an inner one. Without this the held-out slide contributes to the
    ruler its own predictions are measured against -- only 324 of 159,716 FOVs, and measured at
    at most a 1.2% shift in any feature's span, but v3's whole premise is that v2.2's validation
    leaked, so the residue is worth removing rather than bounding.

    This does not reintroduce the scale-mixing that made cross-fitting fail. There, thresholds
    derived from *pooled out-of-fold scores* were applied to a single-scale score. Here each
    fold's thresholds come from its own fold's scale, and only the ranges move between folds, by
    ~1%. The pooled OOF *labels* are therefore directly comparable; the pooled OOF *raw scores*
    (used only for the composite-vs-composite rho) mix scales by that same ~1%.

    -> {"outer_pred", "outer_raw", "outer_folds", "inner_by_outer_fold", "pooled"}
    """
    slides = sorted({r[slide_key] for r in rows})
    outer_pred = np.full(len(rows), -1, dtype=int)
    outer_raw = np.full(len(rows), np.nan)
    outer_folds, inner_by_fold = [], []

    for slide in slides:
        test_mask = np.array([r[slide_key] == slide for r in rows])
        train_rows = [r for r, m in zip(rows, test_mask) if not m]
        test_rows = [r for r, m in zip(rows, test_mask) if m]
        fold_ranges = resolve_ranges(ranges, {slide})

        _, _, inner_folds = loso(train_rows, feature_names, ranges, n_levels, ord_key=ord_key,
                                 sub_key=sub_key, delta=delta, alpha=alpha, weighted=weighted,
                                 skip_levels=skip_levels, slide_key=slide_key,
                                 base_excluded=frozenset({slide}))

        fit = fit_axis(train_rows, feature_names, fold_ranges, n_levels, ord_key=ord_key,
                       sub_key=sub_key, delta=delta, alpha=alpha, weighted=weighted,
                       skip_levels=skip_levels)
        pred, raw = predict_axis(fit, test_rows)
        outer_pred[test_mask] = pred
        outer_raw[test_mask] = raw

        true = np.array([int(r[ord_key]) for r in test_rows])
        outer_folds.append({
            "held_out_slide": slide,
            "n": len(test_rows),
            "n_levels_present": len(set(true.tolist())),
            "feature_names": fit["feature_names"],
            "weights": fit["weights"],
            "dropped": fit["dropped"],
            "merged_groups": fit["merged_groups"],
            "thresholds": fit["thresholds"],
            "ranges": fold_ranges,
            "metrics": ordinal_metrics(true, pred, n_levels),
        })
        inner_by_fold.append({"held_out_slide": slide, "inner_folds": inner_folds})

    true_all = np.array([int(r[ord_key]) for r in rows])
    slide_w = slide_balanced_weights(rows, slide_key)
    return {
        "outer_pred": outer_pred,
        "outer_raw": outer_raw,
        "outer_folds": outer_folds,
        "inner_by_outer_fold": inner_by_fold,
        # Pooled over all OOF predictions, not a mean of fold metrics: two folds hold a test set
        # that is 100% one class, so their per-fold macro-F1 and QWK are degenerate and a mean
        # over folds would weight those equally with a fold that exercises five levels.
        "pooled": ordinal_metrics(true_all, outer_pred, n_levels),
        "pooled_slide_weighted": ordinal_metrics(true_all, outer_pred, n_levels,
                                                 weights=slide_w),
        "n_folds_scored": sum(1 for f in outer_folds if f["n_levels_present"] >= 2),
        "n_folds": len(outer_folds),
    }
