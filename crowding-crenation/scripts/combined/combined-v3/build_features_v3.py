"""Build the v3 fitting matrix by joining labels to already-computed features. Not an extraction.

All 7 v3 training slides were scored in the 497-slide cohort pass, and those per-FOV CSVs keep
the full 9-feature vector, not just the scores. So every feature the fit needs already exists on
disk, computed at exactly the knobs the fit will use. This script is a join plus three
consistency proofs; it downloads no images except for the audit sample and recomputes nothing.

## Why a join rather than re-running the extractor

`extract_features_v2.py` defaults to `--lbp-step 1 --blur-downsample 1`, while the cohort was
scored at 16 and 4. Running it at its defaults would produce a feature matrix normalized against
stride-16 cohort percentiles -- plausible weights at a silently wrong scale, with nothing
downstream able to notice. Sourcing the features from the cohort CSVs makes that
unrepresentable: there is no knob to get wrong, because no feature is computed here.

## The three proofs

1. **The join is total.** Every one of the 1060 label rows must find a `(slide, fov_id)` row in
   the cohort CSVs. A miss is a hard failure, never a dropped row.
2. **The join is row-aligned and the knobs match.** Feeding the joined feature vector back
   through `score_features_v2` with the same params must reproduce the `density_score` and
   `overlap_score` already committed in that same CSV, to absolute 1e-9. One check that catches
   a misaligned join and a knob mismatch at once: if the features came from a different FOV, or
   were computed at a different stride, the composite cannot land on the stored value.
3. **The cohort columns are what a fresh extraction would produce.** A sample of FOVs is
   downloaded and passed through `compute_features` at the pinned knobs, compared at relative
   1e-9. This is the only check that touches the network, and the only one that would catch a
   systematic error in the cohort pass itself rather than in this join.

## The two proofs need different tolerances, and 1e-9 means different things in each

The cohort per-FOV CSVs are written with `run_crowding_pass.FLOAT_FMT = "{:.10g}"` -- ten
*significant* digits. That single fact pushes the two checks in opposite directions.

**Proof 2 compares scores, so the tolerance is absolute 1e-9**, exactly as in
`verify_regression.py`. Composites live in [0, 1], so absolute and relative agree in the middle
of the range but not at the bottom: four of the 1060 FOVs score ~7e-5, meaning nearly every
normalized feature is clipped to 0 and the score is a tiny residual. Input rounding at the tenth
significant digit then moves that residual by ~1e-8 *relatively* while moving it by ~1e-13
absolutely. A relative test would flag those four as failures, but the quantity being checked --
"is this the same score?" -- is absolute, and a 1e-13 disagreement in a number that feeds a
bucket threshold is not a disagreement at all.

**Proof 3 compares raw features, so the tolerance is relative 1e-9.** Here absolute is the
unsatisfiable one: `glcm_contrast` runs to ~160 and `otsu_threshold` to ~170, so ten significant
digits leaves ~1e-7 of absolute slack. `155.83386752280597` is stored as `155.8338675` -- an
absolute delta of 2.3e-8 and a relative delta of 1.5e-10. Requiring absolute 1e-9 there would
fail on correct output.

Both are still an order of magnitude tighter than the storage format's own precision, so each
detects a real disagreement while accepting the rounding the file format imposes.

Usage:
    python scripts/combined/combined-v3/build_features_v3.py
    python scripts/combined/combined-v3/build_features_v3.py --sample-fovs 0   # skip the network
"""
import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent
ROOT = V2_DIR.parent.parent
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(ROOT / "scripts" / "tanzania-complete-081426"))

from _v2_common import (  # noqa: E402
    blur_downsample_from_params,
    compute_features,
    lbp_step_from_params,
    load_image,
    read_csv_dicts,
)
from _slide_common import CROWDING_FOV_DIR, write_csv_atomic  # noqa: E402
from score_fov_v2 import score_features_v2  # noqa: E402
from cohort_ranges import COHORT_RANGES_JSON, FEATURE_NAMES, V22_PARAMS_JSON  # noqa: E402
from merge_labels_v3 import FIELDNAMES as LABEL_FIELDNAMES  # noqa: E402
from merge_labels_v3 import MERGED_LABELS_V3_CSV, RESULTS_DIR  # noqa: E402

FEATURES_V3_CSV = RESULTS_DIR / "features-v3.csv"

# v2.2's own predictions, carried through under a `cohort_` prefix so they can never be mistaken
# for a v3 output. Step 9's gate-replacement section needs `cohort_empty_field_gated` to
# cross-tab what the old `empty_field_override` would have flagged against v3's bottom rungs.
COHORT_FIELDNAMES = ["cohort_density_score", "cohort_density_label", "cohort_overlap_score",
                     "cohort_overlap_label", "cohort_empty_field_gated"]
FIELDNAMES = LABEL_FIELDNAMES + list(FEATURE_NAMES) + COHORT_FIELDNAMES

# Two tolerances, one per proof -- see "The two proofs need different tolerances" above.
# Proof 2 compares bounded composites; proof 3 compares unbounded raw features.
TOL_SCORE_ABS = 1e-9
TOL_FEATURE_REL = 1e-9
DEFAULT_SAMPLE_FOVS = 20
SAMPLE_SEED = 42


def _rel_delta(got, want):
    """|got - want| scaled by the larger magnitude, so it is comparable across features."""
    scale = max(abs(got), abs(want))
    return abs(got - want) / scale if scale else abs(got - want)


def load_cohort_slide(slide, fov_dir=CROWDING_FOV_DIR):
    """-> {fov_id: row} for one slide's cohort per-FOV CSV."""
    path = Path(fov_dir) / f"{slide}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{slide}: no cohort crowding CSV at {path}")
    return {int(r["fov_id"]): r for r in read_csv_dicts(path)}


def join_features(label_rows, fov_dir=CROWDING_FOV_DIR):
    """Left-join labels to cohort features on (slide, fov_id). Fails loudly on any miss."""
    by_slide = {}
    joined, missing, errored = [], [], []
    for r in label_rows:
        slide = r["slide"]
        if slide not in by_slide:
            by_slide[slide] = load_cohort_slide(slide, fov_dir)
        cohort = by_slide[slide].get(int(r["fov_id"]))
        if cohort is None:
            missing.append((slide, r["fov_id"]))
            continue
        if (cohort.get("error") or "").strip():
            errored.append((slide, r["fov_id"], cohort["error"]))
            continue
        out = dict(r)
        for name in FEATURE_NAMES:
            out[name] = float(cohort[name])
        out["cohort_density_score"] = cohort["density_score"]
        out["cohort_density_label"] = cohort["density_label"]
        out["cohort_overlap_score"] = cohort["overlap_score"]
        out["cohort_overlap_label"] = cohort["overlap_label"]
        out["cohort_empty_field_gated"] = cohort["empty_field_gated"]
        joined.append(out)

    if missing:
        raise KeyError(f"{len(missing)} label rows have no cohort feature row: {missing[:5]}")
    if errored:
        raise ValueError(f"{len(errored)} label rows point at a cohort row with an error: "
                         f"{errored[:5]}")
    return joined


def check_roundtrip(rows, params):
    """Proof 2: the joined features must rescore to the committed cohort scores."""
    worst = {"density": 0.0, "overlap": 0.0}
    failures, label_mismatches = [], []
    for r in rows:
        features = {name: r[name] for name in FEATURE_NAMES}
        got = score_features_v2(features, params)
        for axis in ("density", "overlap"):
            want = float(r[f"cohort_{axis}_score"])
            delta = abs(got[f"{axis}_score"] - want)
            worst[axis] = max(worst[axis], delta)
            if delta > TOL_SCORE_ABS:
                failures.append(f"{r['slide']} fov {r['fov_id']} {axis}_score: "
                                f"got={got[f'{axis}_score']!r} cohort={want!r} "
                                f"abs={delta:.3e}")
            if got[f"{axis}_label"] != r[f"cohort_{axis}_label"]:
                label_mismatches.append((r["slide"], r["fov_id"], axis,
                                         got[f"{axis}_label"], r[f"cohort_{axis}_label"]))
    if label_mismatches:
        failures.append(f"{len(label_mismatches)} label mismatches, e.g. {label_mismatches[:3]}")
    return worst, failures


def check_recompute(rows, params, n, seed=SAMPLE_SEED):
    """Proof 3: a sample of FOVs, recomputed from the image at the pinned knobs."""
    if n <= 0:
        return {}, [], 0
    lbp_step = lbp_step_from_params(params)
    blur_downsample = blur_downsample_from_params(params)
    sample = random.Random(seed).sample(rows, min(n, len(rows)))
    worst = {name: 0.0 for name in FEATURE_NAMES}
    failures = []
    for r in sample:
        image = load_image(r["image_path"], grayscale=True)
        fresh = compute_features(image, lbp_step=lbp_step, blur_downsample=blur_downsample)
        for name in FEATURE_NAMES:
            rel = _rel_delta(fresh[name], r[name])
            worst[name] = max(worst[name], rel)
            if rel > TOL_FEATURE_REL:
                failures.append(f"{r['slide']} fov {r['fov_id']} {name}: "
                                f"fresh={fresh[name]!r} cohort={r[name]!r} rel={rel:.3e}")
    return worst, failures, len(sample)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--labels-csv", default=str(MERGED_LABELS_V3_CSV))
    ap.add_argument("--ranges-json", default=str(COHORT_RANGES_JSON))
    ap.add_argument("--params-json", default=str(V22_PARAMS_JSON))
    ap.add_argument("--sample-fovs", type=int, default=DEFAULT_SAMPLE_FOVS,
                    help="how many FOVs to re-extract from the image; 0 skips the network")
    ap.add_argument("--out", default=str(FEATURES_V3_CSV))
    args = ap.parse_args()

    params = json.loads(Path(args.params_json).read_text(encoding="utf-8"))
    ranges_blob = json.loads(Path(args.ranges_json).read_text(encoding="utf-8"))

    # The cohort CSVs, the ranges JSON and the params file must all describe one scoring run.
    # This is the check that stops a re-derived ranges file from being paired with features
    # extracted at different knobs.
    for knob, from_params in (("lbp_step", lbp_step_from_params(params)),
                              ("blur_downsample", blur_downsample_from_params(params))):
        if ranges_blob[knob] != from_params:
            raise ValueError(f"{knob} mismatch: ranges JSON says {ranges_blob[knob]}, "
                             f"{Path(args.params_json).name} says {from_params}")

    label_rows = read_csv_dicts(args.labels_csv)
    rows = join_features(label_rows)
    print(f"joined {len(rows)}/{len(label_rows)} label rows to cohort features "
          f"across {len({r['slide'] for r in rows})} slides "
          f"(lbp_step={ranges_blob['lbp_step']}, "
          f"blur_downsample={ranges_blob['blur_downsample']})")

    worst, failures = check_roundtrip(rows, params)
    if failures:
        print(f"roundtrip vs committed cohort scores: FAILED ({len(failures)} problems)")
    else:
        print(f"roundtrip vs committed cohort scores: max absolute delta "
              f"{max(worst.values()):.3e} (abs_tol {TOL_SCORE_ABS:.0e}), 0 label mismatches")

    r_worst, r_failures, n_sampled = check_recompute(rows, params, args.sample_fovs)
    if n_sampled:
        hottest = max(r_worst, key=r_worst.get)
        print(f"recompute check on {n_sampled} sampled FOVs: max relative delta "
              f"{r_worst[hottest]:.3e} ({hottest}) over all 9 features "
              f"(rel_tol {TOL_FEATURE_REL:.0e})")
    else:
        print("recompute check: skipped (--sample-fovs 0)")

    problems = failures + r_failures
    if problems:
        print(f"\nFAILED: {len(problems)} problem(s)")
        for p in problems[:10]:
            print(f"  - {p}")
        return 1

    write_csv_atomic(args.out, FIELDNAMES, rows)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
