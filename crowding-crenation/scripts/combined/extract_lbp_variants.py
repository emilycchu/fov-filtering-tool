"""Compute LBP entropy at every candidate stride for all 661 calibration FOVs, in one pass.

Only the `lbp_entropy` column is affected by the runtime work, so re-running the whole
`compute_features` vector per variant would burn ~40 minutes recomputing seven identical
columns -- and, worse, would re-stream the 324 tanzania-080526 FOVs from GCS once per variant.
This loads each FOV exactly once, computes every stride from it, and leaves
`build_variant_features.py` to patch the single column. Every other feature is then
bit-identical to `features-v2.2.csv` by construction, so any metric change downstream is
attributable to LBP alone.

Images stream straight from the bucket via `load_image`/`GCSPath` -- nothing is cached to
disk.

`step=1` is included by default and is a free regression check: it must reproduce the stored
`features-v2.2.csv` lbp_entropy column exactly, for all 661 rows.

Usage:
    python scripts/combined/extract_lbp_variants.py [--steps 1 2 4 6 8] [--workers 8]
        [--labels-csv PATH] [--out PATH] [--limit N]
"""
import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    LBP_VARIANTS_CSV,
    RESULTS_DIR,
    load_image,
    read_csv_dicts,
    write_csv_dicts,
)
from src.features.lbp_entropy import _to_gray, lbp_entropy  # noqa: E402

MERGED_LABELS_CSV_V2_2 = RESULTS_DIR / "merged-labels-v2.2.csv"
DEFAULT_STEPS = (1, 2, 4, 6, 8)
KEY_FIELDS = ["fov_key", "dataset", "filename"]


def column(step):
    return f"lbp_entropy_step{step}"


def _score_one(job):
    row, steps = job
    gray = _to_gray(load_image(row["image_path"]))
    # workers=1 is implicit inside a Pool worker (see lbp_entropy._auto_workers) -- the outer
    # pool already owns every core, and nesting threads underneath it slows the batch down.
    values = {column(step): repr(lbp_entropy(gray, step=step)) for step in steps}
    return {**{k: row[k] for k in KEY_FIELDS}, **values}


def main():
    parser = argparse.ArgumentParser(description="Compute LBP entropy at every candidate stride, one pass.")
    parser.add_argument("--labels-csv", default=str(MERGED_LABELS_CSV_V2_2))
    parser.add_argument("--out", default=str(LBP_VARIANTS_CSV))
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--append", action="store_true",
                        help="merge these strides into an existing --out file instead of "
                             "replacing it, so widening the sweep does not re-run the strides "
                             "already measured")
    args = parser.parse_args()

    rows = read_csv_dicts(args.labels_csv)
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} FOVs, strides {args.steps}, {args.workers} workers")

    start = time.perf_counter()
    jobs = [(row, args.steps) for row in rows]
    with Pool(args.workers) as pool:
        results = []
        for i, result in enumerate(pool.imap(_score_one, jobs, chunksize=4), start=1):
            results.append(result)
            if i % 50 == 0 or i == len(jobs):
                rate = (time.perf_counter() - start) / i
                print(f"  {i}/{len(jobs)}  {rate:.2f}s/FOV  "
                      f"eta {(len(jobs) - i) * rate / 60:.1f} min", flush=True)

    steps = list(args.steps)
    if args.append and Path(args.out).exists():
        existing = {r["fov_key"]: r for r in read_csv_dicts(args.out)}
        prior = [int(name[len("lbp_entropy_step"):]) for name in next(iter(existing.values()))
                 if name.startswith("lbp_entropy_step")]
        for result in results:
            result.update({k: v for k, v in existing[result["fov_key"]].items()
                           if k not in result})
        steps = sorted(set(prior) | set(steps))
        print(f"merged with {len(existing)} existing rows; strides now {steps}")

    fieldnames = KEY_FIELDS + [column(step) for step in steps]
    write_csv_dicts(args.out, fieldnames, results)
    print(f"wrote {len(results)} rows to {args.out} in {(time.perf_counter() - start) / 60:.1f} min")


if __name__ == "__main__":
    main()
