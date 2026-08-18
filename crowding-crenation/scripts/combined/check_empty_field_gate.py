"""Assert the empty-field gate is a no-op on the in-distribution calibration set.

The gate (`empty_field_override` in the params JSON, applied by
`_v2_common.apply_empty_field_override`) forces the bottom bucket on both axes when all four
of otsu_separability / lbp_entropy / glcm_contrast / edge_density_unmasked sit below their
calibration p2 floor. It exists for near-empty fields, where Otsu has nothing bimodal to split
and the composite reads background noise as dense tissue.

A gate that overrides the model is only safe if it never fires on data the model already
handles. This script is the evidence: on the 661-FOV v2.2 set it must fire on exactly 3 FOVs,
all of them manually labeled sparser + no rouleaux and *already* predicted as such, leaving
exact-match bit-for-bit unchanged.

Re-run this after any recalibration -- the floors move with the fit, and the deferred 669-FOV
v2.3 refit will pool in the very FOVs the gate is meant to catch.

**Invariants vs. snapshots.** Two of the things checked here are true of any healthy fit, and
two are just one fit's recorded numbers. The distinction used to be stated in this docstring
but not implemented: the snapshot counts were hard-coded pass/fail criteria, so a *legitimate*
refit -- v2.2-optimized, which differs from v2.2 by one knife-edge FOV -- reported FAILED for
no reason. A check that cries wolf on correct work gets ignored, which is worse than not
having it.

So the invariants always assert:

- every FOV the gate fires on is manually labeled (sparser, no rouleaux) -- it must never
  override a field the model should be describing;
- the gate changes no in-distribution prediction;
- incomplete input fails the gate rather than tripping it.

**Why does it only fire on 3 of 661, when 50 FOVs are manually "sparser"?** Because it is a
near-empty-field guard, not a sparser detector -- most sparser fields have plenty of cells and
the composite reads them fine. Checked directly: of the 10 in-distribution FOVs the density
composite over-estimates by 2+ buckets from sparser/monolayer, **nine have 0 of 4 features
below floor** and the tenth has 1 -- none are near-empty, so the gate correctly stays out. Those
ten are all monolayer -> dense, i.e. a *different* failure mode (cross-stain/texture
over-scoring, the same one `dpc-051-LB-D3-...` illustrates in the v2 report), which this gate
was never meant to address and structurally cannot. The gate under-firing is therefore not the
problem; that over-scoring is, and it is the next thing worth attacking.

This is also why the 8-FOV Nigeria set matters out of proportion to its size: it contains two
genuinely near-empty fields, so it is the only place the gate demonstrably earns its keep
(8/8 with it, 6/8 without).

The snapshots (`--expect-fired`, `--expect-exact`) assert only when passed explicitly. Bare,
they are reported as the numbers to pin. Regression command for the current fit:

    python scripts/combined/check_empty_field_gate.py --expect-fired 3         --expect-exact 0.6959 0.6838

Usage:
    python scripts/combined/check_empty_field_gate.py [--features CSV] [--params JSON]
        [--expect-fired N] [--expect-exact DENSITY OVERLAP]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    RESULTS_DIR,
    apply_label_overrides,
    empty_field_features_below,
    empty_field_fired,
    read_csv_dicts,
)
from src.composite_v2 import bucket, weighted_composite  # noqa: E402

# Defaults are the deployed fit, so a bare run checks what score_fov_v2.py actually uses.
# They are a matched pair -- features extracted at the same lbp_step/blur_downsample the params
# record -- and must be changed together.
DEFAULT_FEATURES = RESULTS_DIR / "features-v2.2-optimized.csv"
DEFAULT_PARAMS = RESULTS_DIR / "density_overlap_v2.2-optimized_params.json"


def score_axis(row, axis_params):
    ranges = {n: (v["min"], v["max"]) for n, v in axis_params["normalization"].items()}
    features = {n: float(row[n]) for n in axis_params["normalization"]}
    score = weighted_composite(features, axis_params["weights"], ranges)
    return bucket(score, axis_params["bucket_thresholds"], axis_params["bucket_labels"])


def main():
    parser = argparse.ArgumentParser(description="Check the empty-field gate against the calibration set.")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    parser.add_argument("--expect-fired", type=int, default=None,
                        help="pin the number of gated FOVs. Unset (default): reported, not "
                             "asserted -- a refit legitimately moves it")
    parser.add_argument("--expect-exact", type=float, nargs=2, metavar=("DENSITY", "OVERLAP"),
                        default=None,
                        help="pin gated exact-match per axis. Unset (default): reported, not "
                             "asserted")
    args = parser.parse_args()

    rows = read_csv_dicts(args.features)
    with open(args.params) as f:
        params = json.load(f)
    cfg = params.get("empty_field_override")
    failures = []

    if not cfg or not cfg.get("enabled"):
        print(f"empty_field_override is absent or disabled in {Path(args.params).name} -- nothing to check.")
        return 0

    print(f"{len(rows)} FOVs from {Path(args.features).name}, gate from {Path(args.params).name}")
    print("floors: " + ", ".join(f"{n}<{v:.6g}" for n, v in cfg["thresholds"].items()))

    fired = [r for r in rows if empty_field_fired(r, cfg)]
    print(f"\ngate fires on {len(fired)}/{len(rows)} FOVs:")
    for r in fired:
        forced_ok = r["density_label"] == cfg["density_label"] and r["overlap_label"] == cfg["overlap_label"]
        print(f"  {r['filename']:<38} manual=({r['density_label']}, {r['overlap_label']})"
              f"{'' if forced_ok else '   <-- MISMATCH'}")
        if not forced_ok:
            failures.append(f"{r['filename']} is not manually labeled "
                            f"({cfg['density_label']}, {cfg['overlap_label']}) but the gate forces it there")

    if args.expect_fired is not None and len(fired) != args.expect_fired:
        failures.append(f"expected the gate to fire on {args.expect_fired} FOVs, got {len(fired)}")

    # near-misses: one short of the full set below floor. Not an error -- the margin between
    # the gate and the rest of the set is the thing worth watching, since a rule one feature
    # weaker than v2.2's does catch a true monolayer.
    n_gate = len(cfg["thresholds"])
    near = [r for r in rows if len(empty_field_features_below(r, cfg)) == n_gate - 1]
    print(f"\n{len(near)} FOVs are one feature short of firing ({n_gate - 1} of {n_gate} below floor):")
    for r in near:
        held = [n for n in cfg["thresholds"] if n not in empty_field_features_below(r, cfg)]
        print(f"  {r['filename']:<38} manual=({r['density_label']}, {r['overlap_label']})"
              f"   held above floor by {held[0]}")

    print()
    counts = {}
    for axis, key in (("density", "density_label"), ("overlap", "overlap_label")):
        ungated = gated = 0
        for r in rows:
            d, o = score_axis(r, params["density"]), score_axis(r, params["overlap"])
            gd, go = apply_label_overrides(d, o, r, params)
            predicted, gated_predicted = (d, gd) if axis == "density" else (o, go)
            ungated += predicted == r[key]
            gated += gated_predicted == r[key]
        counts[axis] = (ungated / len(rows), gated / len(rows))
        flag = "" if ungated == gated else "   <-- GATE CHANGED PREDICTIONS"
        print(f"{axis:<8} exact-match  ungated={ungated / len(rows):.4f}  gated={gated / len(rows):.4f}{flag}")
        if ungated != gated:
            failures.append(f"{axis}: gate changed {abs(gated - ungated)} in-distribution predictions")
        if args.expect_exact is not None:
            expected = args.expect_exact[0] if axis == "density" else args.expect_exact[1]
            if abs(counts[axis][1] - expected) > 5e-5:
                failures.append(f"{axis}: exact-match {counts[axis][1]:.4f} != the pinned "
                                f"baseline {expected}")

    # incomplete input must fail the gate rather than trip it -- it forces the bottom bucket
    if fired:
        probe = dict(fired[0])
        probe[next(iter(cfg["thresholds"]))] = ""
        if empty_field_fired(probe, cfg):
            failures.append("a blank feature value trips the gate; it must fail it")
        probe = {k: v for k, v in fired[0].items() if k != next(iter(cfg["thresholds"]))}
        if empty_field_fired(probe, cfg):
            failures.append("a missing feature trips the gate; it must fail it")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK: the gate fires only on manually-sparse FOVs and changes no prediction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
