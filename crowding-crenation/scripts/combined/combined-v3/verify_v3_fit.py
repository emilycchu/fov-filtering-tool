"""Re-run the v3 calibration into a scratch path and diff it against the committed params.

Determinism is what makes this a diff rather than a range check. Fold order is
`sorted({slide})`, the drop loop is deterministic, the ranges are read from a file rather than
refit, and nothing in the fitting path draws from an RNG -- `bootstrap_median_ci`'s
`default_rng(42)` is the only RNG in `calibrate_v2.py` and v3 does not call it. So a re-run must
land on the same numbers, and any drift is a real change rather than noise.

The tolerance is 1e-9 relative rather than exact equality for the same reason
`verify_regression.py` uses one: BLAS thread counts change the order of accumulation in
`A.T @ A`, so bit-identical output is not guaranteed across machines even for identical inputs.

Also asserts the structural invariants that a passing numeric diff would not catch -- a fit can
reproduce exactly and still be the wrong shape.

Usage:
    python scripts/combined/combined-v3/verify_v3_fit.py
"""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent
ROOT = V2_DIR.parent.parent
sys.path.insert(0, str(V2_DIR))

from _v3_common import DENSITY_LEVELS, OVERLAP_LEVELS  # noqa: E402
from calibrate_v3 import PARAMS_V3_JSON  # noqa: E402

TOL_REL = 1e-9

# Fields that must match exactly, not to a tolerance.
EXACT_FIELDS = ("version", "n_fovs", "train_slides", "alpha", "sample_weights", "sub_delta",
                "pava_include_no_cells", "lbp_step", "blur_downsample", "ranges_n_fovs",
                "ranges_n_slides", "ranges_selector", "non_fitting_density_levels")
EXACT_AXIS_FIELDS = ("feature_names", "bucket_labels", "pava_merged_groups",
                     "dropped_negative_coefficient", "n_fit_rows")

EXPECTED_N_FOVS = 1060
EXPECTED_SLIDES = 7


def _flatten(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def diff_numeric(committed, fresh, fail, tol=TOL_REL):
    fa, fb = dict(_flatten(committed)), dict(_flatten(fresh))
    only_committed = sorted(set(fa) - set(fb))
    only_fresh = sorted(set(fb) - set(fa))
    if only_committed:
        fail(f"fields missing from the re-run: {only_committed[:5]}")
    if only_fresh:
        fail(f"fields the re-run added: {only_fresh[:5]}")

    worst, worst_key, n_numeric = 0.0, None, 0
    for k in sorted(set(fa) & set(fb)):
        a, b = fa[k], fb[k]
        if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) \
                or not isinstance(b, (int, float)):
            if a != b and k != ".generated_from":
                fail(f"{k}: committed={a!r} re-run={b!r}")
            continue
        n_numeric += 1
        scale = max(abs(a), abs(b))
        rel = abs(a - b) / scale if scale else abs(a - b)
        if rel > worst:
            worst, worst_key = rel, k
        if rel > tol:
            fail(f"{k}: committed={a!r} re-run={b!r} rel={rel:.3e}")
    return worst, worst_key, n_numeric


def check_structure(params, fail):
    if params.get("version") != "v3":
        fail(f"version is {params.get('version')!r}, expected 'v3'")
    if params.get("n_fovs") != EXPECTED_N_FOVS:
        fail(f"n_fovs is {params.get('n_fovs')}, expected {EXPECTED_N_FOVS}")
    if len(params.get("train_slides", [])) != EXPECTED_SLIDES:
        fail(f"{len(params.get('train_slides', []))} train slides, expected {EXPECTED_SLIDES}")

    gate = params.get("empty_field_override") or {}
    if gate.get("enabled") is not False:
        fail(f"empty_field_override.enabled is {gate.get('enabled')!r}, expected False -- v3 "
             f"replaces the gate with the bottom rung of the ordinal")

    for axis, levels in (("density", DENSITY_LEVELS), ("overlap", OVERLAP_LEVELS)):
        block = params.get(axis) or {}
        if block.get("bucket_labels") != list(levels):
            fail(f"{axis} bucket_labels are {block.get('bucket_labels')!r}, expected {levels}")
        thr = block.get("bucket_thresholds") or []
        if len(thr) != len(levels) - 1:
            fail(f"{axis}: {len(thr)} thresholds for {len(levels)} levels")
        if any(b < a for a, b in zip(thr, thr[1:])):
            fail(f"{axis} thresholds are not non-decreasing: {thr}")
        names = block.get("feature_names") or []
        if len(names) < 1:
            fail(f"{axis} kept no features")
        if sorted(block.get("weights", {})) != sorted(names):
            fail(f"{axis} weights keys do not match feature_names")
        if sorted(block.get("normalization", {})) != sorted(names):
            fail(f"{axis} normalization keys do not match feature_names")
        total = sum(block.get("weights", {}).values())
        if abs(total - 1.0) > 1e-9:
            fail(f"{axis} weights sum to {total!r}, expected 1.0")

    # The headline structural claim of the 7-level extension.
    merged = params.get("density", {}).get("pava_merged_groups") or []
    if any({0, 1, 2} <= set(g) for g in merged):
        fail("density PAVA merged `no cells` + `few cells` + `sparser` into one block; the "
             "7-level extension failed and that is the README's headline, not a footnote")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--params", default=str(PARAMS_V3_JSON))
    args = ap.parse_args()

    committed_path = Path(args.params)
    if not committed_path.exists():
        print(f"FAILED: no committed params at {committed_path}")
        return 1
    committed = json.loads(committed_path.read_text(encoding="utf-8"))

    failures = []
    fail = failures.append

    check_structure(committed, fail)

    # Re-run with the committed run's own knobs, so this checks reproducibility rather than
    # whether today's defaults happen to match what was shipped.
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cmd = [sys.executable, str(HERE / "calibrate_v3.py"),
               "--sample-weights", str(committed["sample_weights"]),
               "--sub-delta", str(committed["sub_delta"]),
               "--ablation-deltas", str(committed["sub_delta"]),
               "--params-out", str(tmp / "params.json"),
               "--report-out", str(tmp / "report.md"),
               "--fold-detail-out", str(tmp / "folds.csv"),
               "--oof-out", str(tmp / "oof.csv")]
        if committed.get("pava_include_no_cells"):
            cmd.append("--pava-include-no-cells")
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(HERE))
        if proc.returncode != 0:
            print(f"FAILED: the re-run exited {proc.returncode}\n{proc.stderr[-2000:]}")
            return 1
        fresh = json.loads((tmp / "params.json").read_text(encoding="utf-8"))

    worst, worst_key, n_numeric = diff_numeric(committed, fresh, fail)

    print(f"structure: version={committed['version']}, n_fovs={committed['n_fovs']}, "
          f"{len(committed['train_slides'])} train slides, "
          f"density {len(committed['density']['bucket_labels'])} levels / "
          f"overlap {len(committed['overlap']['bucket_labels'])} levels, "
          f"empty_field_override.enabled={committed['empty_field_override']['enabled']}")
    print(f"density PAVA merges: {committed['density']['pava_merged_groups'] or 'none'}")
    print(f"reproducibility: {n_numeric} numeric fields, max relative delta {worst:.3e} "
          f"({worst_key}) against rel_tol {TOL_REL:.0e}")

    if failures:
        print(f"\nFAILED: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
