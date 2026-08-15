"""Assert the fast LBP kernel is bit-identical to skimage, and time it against the alternatives.

This is the gate on `src/features/lbp_entropy.py`: `lbp_entropy()` is only allowed to be the
fast path because `_lbp_codes()` reproduces `skimage.feature.local_binary_pattern` exactly,
and that claim has to be checked rather than asserted. The exactness check runs on real FOVs
(where flat regions create the floating-point ties that a naive vectorization gets wrong) and
on small random arrays (where every pixel is within `radius` of a border, so the zero-fill
out-of-bounds behaviour is actually exercised).

Also benchmarks the two parallel backends the runtime work considered -- threads over cache
tiles vs. processes over row strips -- since numpy's per-operation dispatch holds the
interpreter lock and caps thread scaling. Strips are exact because skimage reads
out-of-bounds neighbours as 0, so a strip carrying a `ceil(radius)+1` halo reproduces its
interior rows.

Usage:
    python scripts/combined/lbp-optimization/bench_lbp.py [--steps 1 2 4 6 8] [--repeat 1] [--skip-process]
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from skimage.feature import local_binary_pattern

# one deeper than the other scripts/combined scripts, hence the extra .parent
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    INITIAL_IMAGE_DIR,
    LBP_RUNTIME_DIR,
    RESULTS_DIR,
    TANZANIA_080526_BLOB_PREFIX,
    TANZANIA_080526_BUCKET,
    TANZANIA_IMAGE_DIR,
    load_image,
    read_csv_dicts,
    write_csv_dicts,
)
from src.features.lbp_entropy import (  # noqa: E402
    DEFAULT_RADIUS,
    _lbp_codes,
    _to_gray,
    lbp_entropy,
    lbp_entropy_skimage,
)
from src.pipeline import GCSPath  # noqa: E402

N_POINTS = 8 * DEFAULT_RADIUS
HALO = int(np.ceil(DEFAULT_RADIUS)) + 1
BASE_FEATURES_CSV = RESULTS_DIR / "features-v2.2.csv"
MERGED_LABELS_CSV_V2_2 = RESULTS_DIR / "merged-labels-v2.2.csv"


def _strip(args):
    """One row strip, computed with its halo and trimmed back. Top level so it can pickle.

    `row_offset` is not optional: the interpolation weights depend on the absolute row (see
    `_interp_weights`), so a strip that thinks it starts at row 0 produces a handful of
    different codes. Dropping it is what made the first version of this benchmark report
    exact=False.
    """
    sub, row_offset, keep_from, keep_rows, step = args
    codes = _lbp_codes(sub, N_POINTS, DEFAULT_RADIUS, step=step, workers=1, row_offset=row_offset)
    return codes[keep_from:keep_from + keep_rows]


def codes_by_process(gray, step=1, workers=8):
    """`_lbp_codes` split over row strips in separate processes, to dodge the GIL."""
    n_rows = gray.shape[0]
    # Strip boundaries have to land on the sampled grid or the halo trim shifts the output.
    bounds = [b - b % step for b in np.linspace(0, n_rows, workers + 1).astype(int)]
    bounds[-1] = n_rows
    jobs = []
    for start, stop in zip(bounds, bounds[1:]):
        if start >= stop:
            continue
        halo = HALO + (-HALO % step)  # keep the halo a whole number of grid steps
        lo = max(0, start - halo)
        hi = min(n_rows, stop + halo)
        jobs.append((np.ascontiguousarray(gray[lo:hi]), lo, (start - lo) // step,
                     len(range(start, stop, step)), step))
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(_strip, jobs))
    return np.concatenate(parts, axis=0)


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def _exact_entropy(row):
    """Top level so `--full-set`'s Pool can pickle it."""
    return row["fov_key"], repr(lbp_entropy(_to_gray(load_image(row["image_path"]))))


def verify_calibration_set(workers, failures):
    """The fast kernel must reproduce every stored lbp_entropy value in features-v2.2.csv.

    The five-FOV check above is the fast gate; this is the exhaustive one. It re-streams the
    324 tanzania-080526 FOVs from GCS, so it takes ~20 minutes -- run it after touching the
    kernel, not on every invocation.
    """
    rows = read_csv_dicts(MERGED_LABELS_CSV_V2_2)
    stored = {r["fov_key"]: r["lbp_entropy"] for r in read_csv_dicts(BASE_FEATURES_CSV)}
    start = time.perf_counter()
    with Pool(workers) as pool:
        results = pool.map(_exact_entropy, rows, chunksize=4)
    mismatched = [key for key, value in results if value != stored[key]]
    print(f"\nfull calibration set: {len(results) - len(mismatched)}/{len(results)} rows "
          f"bit-identical to features-v2.2.csv ({(time.perf_counter() - start) / 60:.1f} min)")
    for key in mismatched[:5]:
        failures.append(f"{key}: exact kernel does not reproduce the stored lbp_entropy")


def sample_paths():
    """A few local FOVs from both local datasets, plus one streamed from GCS."""
    paths = sorted(TANZANIA_IMAGE_DIR.iterdir())[:2]
    paths += sorted(INITIAL_IMAGE_DIR.iterdir())[:2]
    paths.append(GCSPath(TANZANIA_080526_BUCKET, f"{TANZANIA_080526_BLOB_PREFIX}/dpc-001-KTR-72502946.png"))
    return paths


def check_random_borders(rng, failures):
    """Small arrays: every pixel is a border pixel, so out-of-bounds zero-fill is exercised."""
    for shape in [(31, 31), (60, 70), (7, 513), (513, 7)]:
        image = rng.integers(0, 256, size=shape, dtype=np.uint8)
        reference = local_binary_pattern(image, N_POINTS, DEFAULT_RADIUS, method="uniform")
        for step in (1, 2, 4):
            mine = _lbp_codes(image, N_POINTS, DEFAULT_RADIUS, step=step, workers=1)
            if not np.array_equal(mine.astype(np.float64), reference[::step, ::step]):
                failures.append(f"random {shape} step={step}: codes differ from skimage")
    print(f"random-array border check: {len(failures)} failure(s)")


def main():
    parser = argparse.ArgumentParser(description="Bit-exactness gate and timings for the fast LBP kernel.")
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 6, 8])
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--skip-process", action="store_true",
                        help="skip the process-pool backend benchmark")
    parser.add_argument("--out-csv", default=str(LBP_RUNTIME_DIR / "runtime-bench.csv"),
                        help="per-FOV timings, consumed by plot_lbp_variants.py")
    parser.add_argument("--full-set", action="store_true",
                        help="also check all 661 calibration FOVs against features-v2.2.csv "
                             "(~20 min, re-streams the GCS half)")
    parser.add_argument("--full-set-workers", type=int, default=4,
                        help="4 beats 8 here -- the kernel is memory-bandwidth-bound")
    args = parser.parse_args()

    failures = []
    timings = []
    check_random_borders(np.random.default_rng(0), failures)

    header = f"{'FOV':<44} {'skimage':>9} {'exact':>9} {'speedup':>8} {'exact?':>7}"
    header += "".join(f"{'step' + str(s):>9}" for s in args.steps if s > 1)
    header += f"{'drift(max)':>12}"
    print("\n" + header)

    for path in sample_paths():
        gray = _to_gray(load_image(path))
        reference_codes = local_binary_pattern(gray, N_POINTS, DEFAULT_RADIUS, method="uniform")

        skimage_entropy, t_skimage = timed(lambda: lbp_entropy_skimage(gray))
        t_skimage = min([t_skimage] + [timed(lambda: lbp_entropy_skimage(gray))[1]
                                       for _ in range(args.repeat - 1)])

        exact_entropy, t_exact = timed(lambda: lbp_entropy(gray))
        t_exact = min([t_exact] + [timed(lambda: lbp_entropy(gray))[1] for _ in range(args.repeat - 1)])

        codes_ok = np.array_equal(
            _lbp_codes(gray, N_POINTS, DEFAULT_RADIUS).astype(np.float64), reference_codes
        )
        if not codes_ok:
            failures.append(f"{path.name}: fast codes differ from skimage")
        if exact_entropy != skimage_entropy:
            failures.append(f"{path.name}: entropy {exact_entropy!r} != skimage {skimage_entropy!r}")

        row = f"{path.name[:44]:<44} {t_skimage:>8.2f}s {t_exact:>8.2f}s {t_skimage / t_exact:>7.1f}x"
        row += f"{str(codes_ok and exact_entropy == skimage_entropy):>7}"
        timing = {"filename": path.name, "skimage_s": f"{t_skimage:.4f}", "exact_s": f"{t_exact:.4f}"}
        drift = 0.0
        for step in args.steps:
            if step == 1:
                continue
            value, elapsed = timed(lambda: lbp_entropy(gray, step=step))
            elapsed = min([elapsed] + [timed(lambda: lbp_entropy(gray, step=step))[1]
                                       for _ in range(args.repeat - 1)])
            drift = max(drift, abs(value - skimage_entropy))
            timing[f"step{step}_s"] = f"{elapsed:.4f}"
            timing[f"step{step}_drift"] = f"{value - skimage_entropy:+.6f}"
            row += f"{elapsed:>8.2f}s"
        row += f"{drift:>+12.5f}"
        print(row)
        timings.append(timing)

    if not args.skip_process:
        gray = _to_gray(load_image(sample_paths()[0]))
        reference_codes = local_binary_pattern(gray, N_POINTS, DEFAULT_RADIUS, method="uniform")
        print("\nbackend comparison on one FOV (min of 2):")
        for label, fn in [
            ("serial", lambda: _lbp_codes(gray, N_POINTS, DEFAULT_RADIUS, workers=1)),
            ("threads x8", lambda: _lbp_codes(gray, N_POINTS, DEFAULT_RADIUS, workers=8)),
            ("processes x8", lambda: codes_by_process(gray, workers=8)),
        ]:
            codes, elapsed = timed(fn)
            elapsed = min(elapsed, timed(fn)[1])
            exact = np.array_equal(codes.astype(np.float64), reference_codes)
            print(f"  {label:<14} {elapsed:>6.2f}s  exact={exact}")
            if not exact:
                failures.append(f"backend {label}: codes differ from skimage")

    if timings:
        write_csv_dicts(args.out_csv, list(timings[0]), timings)
        print(f"\nwrote per-FOV timings to {args.out_csv}")

    if args.full_set:
        verify_calibration_set(args.full_set_workers, failures)

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nOK: the fast kernel is bit-identical to skimage on every case checked.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
