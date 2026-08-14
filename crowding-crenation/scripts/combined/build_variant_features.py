"""Build one feature CSV per LBP variant by patching only the `lbp_entropy` column.

Takes `features-v2.2.csv` as the base and swaps in the strided LBP entropy values from
`extract_lbp_variants.py`. Nothing else is recomputed, so all seven other feature columns
stay bit-identical to the v2.2 fit and any downstream metric change is attributable to LBP.

The `nolbp` variant keeps the column in the file (so `calibrate_v2.load_features` still
parses) but is excluded from the candidate feature list at refit time by
`compare_lbp_variants.py` -- that exclusion, not a missing column, is what "removed from the
pipeline" means to the model.

Usage:
    python scripts/combined/build_variant_features.py [--variants-csv PATH] [--base PATH]
        [--out-dir PATH]
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    LBP_RUNTIME_DIR,
    LBP_VARIANTS_CSV,
    RESULTS_DIR,
    read_csv_dicts,
    write_csv_dicts,
)
from extract_lbp_variants import column  # noqa: E402

BASE_FEATURES_CSV = RESULTS_DIR / "features-v2.2.csv"


def variant_name(step):
    return f"step{step}"


def build(base_rows, fieldnames, variants_by_key, step, out_path):
    rows = []
    for row in base_rows:
        patched = dict(row)
        patched["lbp_entropy"] = variants_by_key[row["fov_key"]][column(step)]
        rows.append(patched)
    write_csv_dicts(out_path, fieldnames, rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Patch the lbp_entropy column per variant.")
    parser.add_argument("--variants-csv", default=str(LBP_VARIANTS_CSV))
    parser.add_argument("--base", default=str(BASE_FEATURES_CSV))
    parser.add_argument("--out-dir", default=str(LBP_RUNTIME_DIR))
    args = parser.parse_args()

    base_rows = read_csv_dicts(args.base)
    with open(args.base, newline="") as f:
        fieldnames = f.readline().strip().split(",")
    variant_rows = read_csv_dicts(args.variants_csv)
    variants_by_key = {r["fov_key"]: r for r in variant_rows}

    missing = [r["fov_key"] for r in base_rows if r["fov_key"] not in variants_by_key]
    if missing:
        raise SystemExit(f"{len(missing)} base FOVs have no variant row, first: {missing[0]}")

    steps = sorted(int(name[len("lbp_entropy_step"):]) for name in variant_rows[0]
                   if name.startswith("lbp_entropy_step"))
    out_dir = Path(args.out_dir)
    for step in steps:
        out_path = out_dir / f"features-v2.2-{variant_name(step)}.csv"
        build(base_rows, fieldnames, variants_by_key, step, out_path)
        print(f"wrote {out_path.name}  (lbp_entropy from step={step})")

    print(f"\nbase {Path(args.base).name} is reused as-is for the exact and nolbp variants "
          "(the exact kernel reproduces it bit-for-bit, and nolbp drops the feature at refit)")


if __name__ == "__main__":
    main()
