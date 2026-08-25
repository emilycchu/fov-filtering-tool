"""Derive the p2/p98 feature normalization ranges from the scored cohort, not the label pool.

v2.2 normalized each feature against percentiles of its own ~1,000 labelled rows, and those
rows came from 2 slides sitting at the 77th and 93rd percentile of the 271-slide cohort they
went on to score. That is defect (2) of the v3 design doc: the fit is centred on the wrong part
of the distribution, and the cohort README measures 10-13% bottom-end clipping as a result.
The fix is to take the ranges from every FOV that has been scored and leave the label pool to
do nothing but supply labels.

## All 493 non-test slides, not the positives-only 87,799

The design doc quotes 87,799 FOVs because that is all that had ever been scored when it was
written -- the cohort pass was built from the annotatability spreadsheet, which is positives-only
by construction. The negatives have since been scored, so `all` is now available and is the
correct population: it is what the deployed model actually runs on.

**Measured, the choice barely moves the normalization.** `--slides positive` gives 269 slides /
87,151 FOVs, and its p2/p98 span differs from the all-slides span by at most **6.7%** on any of
the 9 features (`otsu_separability`; every other feature is within 3.1%). So the negatives being
denser at the slide-mean level does not translate into a materially different pooled per-FOV
percentile -- the design doc's expectation that positives-only would "reintroduce the bias" is
right in principle but small in this particular effect. `all` stays the default because it is
the honest population and costs nothing, not because the alternative was measured to be bad.

The 4 test slides are excluded. Including them would be transductive leakage: their feature
distribution would inform the normalization that the frozen model is later scored under.

## Two consequences for everything downstream

1. **The ranges are near-constant across all 49 fits, and fold-local by default.** v2.2 refit its
   percentile ranges inside every fold from that fold's *training labels*
   (`calibrate_v2.py:223`), which means a raw score of 0.43 in one fold is not the same quantity
   as 0.43 in another. Cohort ranges remove almost all of that: `calibrate_v3.py --ranges-scope
   fold` still derives each fold's p2/p98 from the cohort *minus* the slides held out at that
   level, so no slide contributes to the ruler its own predictions are measured against, but
   since one slide is 324 of 159,716 FOVs the ranges move by at most **1.1%** of a feature's
   span between folds (measured; `otsu_separability` on `RUB-72501818` is the worst case).
   Measured end to end, closing that residue moved **1 of 1060** density predictions and 2 of
   1060 overlap ones. `--ranges-scope fixed` uses one range set everywhere so the difference
   stays measurable rather than assumed.
2. **v3 raw scores and thresholds are not comparable to v2.2's.** The correction is a shift, not
   a widening -- as measured, every cohort p2 sits *below* v2.2's min (`coverage` 0.032 vs 0.073,
   `lbp_entropy` 2.69 vs 3.13), which is the 10-13% bottom-end clipping, while the width moves
   in both directions: `otsu_separability` and `glcm_contrast` widen by ~1.5x but
   `tile_glcm_patchiness` narrows to 0.76x and `coverage` to 0.95x. Either way each feature
   contributes a different normalized span than it did, so the fitted coefficients rescale.
   Comparing a v3 threshold to a v2.2 threshold as a number is meaningless; only the resulting
   labels and metrics compare.

Usage:
    python scripts/combined/combined-v3/cohort_ranges.py
    python scripts/combined/combined-v3/cohort_ranges.py --slides positive --out /tmp/pos.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent
ROOT = V2_DIR.parent.parent
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(ROOT / "scripts" / "tanzania-complete-081426"))

from _v2_common import blur_downsample_from_params, lbp_step_from_params  # noqa: E402
from _slide_common import CROWDING_FOV_DIR  # noqa: E402
from merge_labels_v3 import SPLITS_CSV, load_splits  # noqa: E402

# The 9 raw features `compute_features` returns, in its own key order. Every one is a column in
# the cohort per-FOV CSVs, which is what makes this a read rather than a re-extraction pass.
FEATURE_NAMES = ("coverage", "otsu_threshold", "otsu_separability", "saturation_score",
                 "lbp_entropy", "glcm_contrast", "edge_density_unmasked", "tile_glcm_cv",
                 "tile_glcm_patchiness")

RESULTS_DIR = ROOT / "data" / "results" / "combined-v3"
COHORT_RANGES_JSON = RESULTS_DIR / "cohort-ranges.json"
SLIDE_SUMMARY_497_CSV = RESULTS_DIR / "slide-summary-497.csv"
V22_PARAMS_JSON = (ROOT / "data" / "results" / "density-rouleaux-v2"
                   / "density_overlap_v2.2-optimized_params.json")

PCT_LO, PCT_HI = 2, 98

# Mirrors the degenerate-range guard in `calibrate_v2.percentile_ranges`: a feature whose p2 and
# p98 coincide would make `normalize_matrix` divide by zero. That function is not reused here
# because it takes a list of row dicts and this pass streams 160k rows column-wise, and because
# v3 replaces it outright -- the ranges now come from this file rather than from the fit.
DEGENERATE_EPS = 1e-6


def _read_feature_columns(path):
    """-> (dict of feature -> np.ndarray, n_rows_kept, n_rows_error) for one slide CSV.

    Rows with a non-empty `error` are dropped: the pass writes the row so the FOV is accounted
    for, but its feature values are absent or partial, and a blank parsed as a float would be a
    silent zero in a percentile.
    """
    import csv
    kept = {name: [] for name in FEATURE_NAMES}
    n_error = 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("error") or "").strip():
                n_error += 1
                continue
            for name in FEATURE_NAMES:
                kept[name].append(float(r[name]))
    return ({name: np.asarray(v, dtype=float) for name, v in kept.items()},
            len(kept[FEATURE_NAMES[0]]), n_error)


def select_slides(selector, splits, summary_csv=SLIDE_SUMMARY_497_CSV,
                  fov_dir=CROWDING_FOV_DIR):
    """-> (sorted slide ids to use, sorted test slide ids excluded).

    Membership comes from the scored CSVs actually on disk, intersected with the 497-slide
    summary, so a slide that was never scored cannot silently contribute an empty column.
    """
    from _v2_common import read_csv_dicts
    truth_of = {r["slide_id"]: r["truth"] for r in read_csv_dicts(summary_csv)}
    on_disk = {p.stem for p in fov_dir.glob("*.csv")}
    missing = sorted(set(truth_of) - on_disk)
    if missing:
        raise FileNotFoundError(f"{len(missing)} summary slides have no crowding CSV: "
                                f"{missing[:5]}")

    test_slides = sorted(s for s, r in splits.items() if r["role"] == "test")
    chosen = sorted(s for s in truth_of
                    if s not in set(test_slides)
                    and (selector == "all" or truth_of[s] == selector))
    return chosen, test_slides


def compute_ranges(slides, fov_dir=CROWDING_FOV_DIR):
    """-> (ranges dict, n_fovs, n_errors). Streams slide by slide, concatenates once."""
    chunks = {name: [] for name in FEATURE_NAMES}
    n_fovs = n_errors = 0
    for slide in slides:
        cols, kept, n_error = _read_feature_columns(fov_dir / f"{slide}.csv")
        for name in FEATURE_NAMES:
            chunks[name].append(cols[name])
        n_fovs += kept
        n_errors += n_error

    ranges = {}
    for name in FEATURE_NAMES:
        values = np.concatenate(chunks[name])
        lo, hi = (float(v) for v in np.percentile(values, [PCT_LO, PCT_HI]))
        if hi <= lo:
            hi = lo + DEGENERATE_EPS
        ranges[name] = {"min": lo, "max": hi}
    return ranges, n_fovs, n_errors


class CohortColumns:
    """Every non-test slide's feature columns, held in memory so that ranges excluding an
    arbitrary subset of slides cost a percentile call instead of a re-read of 493 CSVs.

    Nested LOSO needs 49 different range sets -- 7 outer folds excluding one slide each, and 42
    inner folds excluding two. Recomputing from disk would be 49 x ~8 s; from here it is ~10 ms
    each. The whole cohort is 159,716 x 9 float64 = ~12 MB, so holding it is cheaper than being
    clever about it.
    """

    def __init__(self, slides, fov_dir=CROWDING_FOV_DIR):
        self.slides = list(slides)
        self._cols = {}
        self.n_errors = 0
        for slide in self.slides:
            cols, _kept, n_error = _read_feature_columns(Path(fov_dir) / f"{slide}.csv")
            self._cols[slide] = cols
            self.n_errors += n_error
        self._cache = {}

    def n_fovs(self, exclude=()):
        exclude = set(exclude)
        return sum(len(self._cols[s][FEATURE_NAMES[0]])
                   for s in self.slides if s not in exclude)

    def ranges(self, exclude=()):
        """-> {feature: (lo, hi)}, the tuple form `normalize_matrix` expects. Memoised."""
        key = frozenset(exclude)
        if key in self._cache:
            return self._cache[key]
        missing = key - set(self.slides)
        if missing:
            raise KeyError(f"cannot exclude slides that are not loaded: {sorted(missing)}")
        kept = [s for s in self.slides if s not in key]
        if not kept:
            raise ValueError("every slide excluded; no rows left to take percentiles over")
        out = {}
        for name in FEATURE_NAMES:
            values = np.concatenate([self._cols[s][name] for s in kept])
            lo, hi = (float(v) for v in np.percentile(values, [PCT_LO, PCT_HI]))
            if hi <= lo:
                hi = lo + DEGENERATE_EPS
            out[name] = (lo, hi)
        self._cache[key] = out
        return out


def load_cohort_columns(splits, selector="all", summary_csv=SLIDE_SUMMARY_497_CSV,
                        fov_dir=CROWDING_FOV_DIR):
    """The cohort columns for every non-test slide, ready for per-fold range derivation."""
    slides, _test = select_slides(selector, splits, summary_csv, fov_dir)
    return CohortColumns(slides, fov_dir)


def load_cohort_ranges(path=COHORT_RANGES_JSON):
    """-> (ranges as {feature: (lo, hi)}, metadata dict).

    The tuple form is what `calibrate_v2.normalize_matrix` expects; the JSON stores
    `{"min":, "max":}` to match the params-file convention so `calibrate_v3.py` can copy the
    block straight into its own `normalization` section without reshaping it.
    """
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    ranges = {k: (v["min"], v["max"]) for k, v in blob["ranges"].items()}
    return ranges, blob


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--slides", choices=("all", "positive", "negative"), default="all",
                    help="which truth class to draw ranges from; 'all' is the only correct "
                         "default, the others exist so the alternative can be measured")
    ap.add_argument("--params-json", default=str(V22_PARAMS_JSON),
                    help="the params file these cohort CSVs were scored with; its knobs are "
                         "recorded so calibrate_v3.py can hard-fail on a mismatch")
    ap.add_argument("--out", default=str(COHORT_RANGES_JSON))
    args = ap.parse_args()

    params = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
    splits = load_splits()
    slides, test_slides = select_slides(args.slides, splits)
    ranges, n_fovs, n_errors = compute_ranges(slides)

    blob = {
        "generated_from": "data/results/tanzania-complete-081426/fov/crowding/*.csv",
        "slides_selector": args.slides,
        "n_slides": len(slides),
        "n_fovs": n_fovs,
        "n_rows_dropped_error": n_errors,
        "excluded_test_slides": test_slides,
        "splits_csv": str(SPLITS_CSV.relative_to(ROOT)).replace("\\", "/"),
        "params_source": str(Path(args.params_json).relative_to(ROOT)).replace("\\", "/"),
        "params_version": params.get("version"),
        "lbp_step": lbp_step_from_params(params),
        "blur_downsample": blur_downsample_from_params(params),
        "percentiles": [PCT_LO, PCT_HI],
        "feature_names": list(FEATURE_NAMES),
        "ranges": ranges,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")

    print(f"slides={args.slides}: {len(slides)} slides, {n_fovs} FOVs "
          f"({n_errors} rows dropped for a non-empty error), "
          f"{len(test_slides)} test slides excluded: {test_slides}")
    print(f"knobs from {Path(args.params_json).name}: "
          f"lbp_step={blob['lbp_step']}, blur_downsample={blob['blur_downsample']}")

    # How much wider these are than the fit they replace, for the two axes' shared features.
    v22 = params.get("density", {}).get("normalization", {})
    print(f"\n{'feature':<24}{'cohort p2':>12}{'cohort p98':>12}{'v2.2 min':>12}"
          f"{'v2.2 max':>12}{'width x':>9}")
    for name in FEATURE_NAMES:
        lo, hi = ranges[name]["min"], ranges[name]["max"]
        old = v22.get(name)
        if old is None:
            print(f"{name:<24}{lo:>12.4f}{hi:>12.4f}{'--':>12}{'--':>12}{'--':>9}")
            continue
        ratio = (hi - lo) / (old["max"] - old["min"])
        print(f"{name:<24}{lo:>12.4f}{hi:>12.4f}{old['min']:>12.4f}{old['max']:>12.4f}"
              f"{ratio:>9.2f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
