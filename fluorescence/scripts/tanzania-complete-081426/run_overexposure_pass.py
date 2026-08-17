"""Run the pixel-level overexposure detector over every fluorescence FOV of every Tanzania
catalog slide, and record how many FOVs it flags per slide.

87,801 FOVs across 271 slides, streamed from GCS. This is the "original image processing
approach" arm of the fluorescence comparison; the crop-outlier arm is `run_crop_outlier_pass.py`
and reads precomputed detection CSVs instead of images.

Three things this does differently from `scripts/run_overexposed_diverse_test.py`, which scores
isolated labeled FOVs:

**Box resolution is read from the index, not probed.** `gcs_fov_multi.load_fov_image` calls
`find_tz_box`, which probes up to 5 `TZ2025-Box<N>/` prefixes *per call* and caches nothing --
up to 439,005 list requests over this sweep. The slide index already resolved every slide's box,
so this constructs the blob name directly and reuses `_download_color`.

**`diffuse_halo_flag` costs no extra fetches.** That flag needs the two preceding FOVs' results
to test whether a candidate matches a neighbouring illumination trend. The diverse test has to
re-download them because it only ever holds one FOV; a full-slide pass already has all 324
results in hand, so the neighbour comparison happens after the pool drains, from memory.

**Masks are dropped immediately.** `OverexposureResult.mask` is a ~400px uint8 array; holding
324 of them is ~52 MB per slide for something never written out. The advisory-flag pass needs
only the scalar diffuse fields, which survive.

`present` is the counted flag -- what production returns today. `present_folded`
(`present or diffuse_halo_flag`) ships alongside as an advisory column, using the same name the
diverse test uses so the two results files stay comparable.

Usage:
    python scripts/tanzania-complete-081426/run_overexposure_pass.py --slides NKR-72502209
    python scripts/tanzania-complete-081426/run_overexposure_pass.py --threads 8
"""
import argparse
import csv
import json
import os
import random
import socket
import ssl
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FLUOR_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(FLUOR_ROOT))

from src.gcs_fov_multi import TZ_BUCKET, _download_color  # noqa: E402
from src.overexposure import detect_overexposure, diffuse_halo_flag  # noqa: E402

# The crowding subproject owns the slide index and the frozen catalog. Read as *data* -- both
# subprojects define a `src` package and only one can be sys.modules["src"], so nothing here
# imports across the boundary.
CROWDING_RESULTS = (FLUOR_ROOT.parent / "crowding-crenation" / "data" / "results"
                    / "tanzania-complete-081426")
SLIDE_INDEX_JSON = CROWDING_RESULTS / "slide-index.json"
SLIDES_CSV = CROWDING_RESULTS / "slides.csv"

RESULTS_DIR = FLUOR_ROOT / "data" / "results" / "tanzania-complete-081426"
FOV_DIR = RESULTS_DIR / "fov" / "overexposure"
LOGS_DIR = RESULTS_DIR / "logs"

# How many preceding FOVs the neighbour-trend check looks back over, matching
# scan_diffuse_candidates.py's NEIGHBOR_WINDOW.
NEIGHBOR_WINDOW = 2

RESULT_COLUMNS = ["present", "present_folded", "diffuse_halo_flag", "confidence",
                  "contrast_ratio", "baseline", "peak", "area_fraction", "solidity",
                  "anisotropy", "radial_rho", "r2_over_r1", "diffuse_radius",
                  "diffuse_circularity", "diffuse_centroid_x", "diffuse_centroid_y"]
FIELDNAMES = ["fov_id"] + RESULT_COLUMNS + ["error"]
ERROR_FIELDNAMES = ["slide_id", "fov_id", "blob", "error"]

FLOAT_FMT = "{:.6g}"
MAX_ERROR_FRACTION = 0.05
MIN_ERRORS_BEFORE_ABANDON = 3

RETRYABLE = (ssl.SSLError, socket.timeout, ConnectionError, TimeoutError)


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return FLOAT_FMT.format(value)
    return value


def _with_retry(fn, *args, attempts=3, base_delay=1.0):
    """Small local copy of the crowding side's retry.

    Deliberate duplication: importing `_slide_common` would put crowding's `src` into
    sys.modules and shadow this subproject's own. ~10 lines is a better price than that.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except RETRYABLE:
            if attempt == attempts:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5))
    raise AssertionError("attempts must be >= 1")


def write_csv_atomic(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_csv_rows(path, fieldnames, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def checkpoint_done(path, expected_rows):
    path = Path(path)
    if not path.exists():
        return False
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return len(list(csv.DictReader(f))) == expected_rows
    except (OSError, UnicodeDecodeError, csv.Error):
        return False


def fluor_blob_name(box, slide_id, fov_id):
    return f"{box}/{slide_id}/fluorescent-{fov_id:03d}-{slide_id}.png"


def detect_one(box, slide_id, fov_id):
    """Detect on one FOV. Returns (fov_id, result_or_None, error_or_None, blob)."""
    blob_name = fluor_blob_name(box, slide_id, fov_id)
    try:
        image = _with_retry(_download_color, TZ_BUCKET, blob_name)
        result = detect_overexposure(image)
        # ~160 KB per FOV; a slide's worth is ~52 MB and nothing downstream reads it.
        result.mask = None
        return fov_id, result, None, blob_name
    except Exception as exc:  # noqa: BLE001 -- one bad FOV must not end the sweep
        return fov_id, None, f"{type(exc).__name__}: {exc}", blob_name


def run_slide(box, slide_id, fov_ids, out_csv, args):
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        outcomes = list(pool.map(lambda f: detect_one(box, slide_id, f), fov_ids))

    by_fov = {fov_id: result for fov_id, result, _err, _blob in outcomes}

    rows, errors = [], []
    for fov_id, result, error, blob_name in sorted(outcomes, key=lambda o: o[0]):
        if error is not None:
            rows.append({"fov_id": fov_id, "error": error})
            errors.append({"slide_id": slide_id, "fov_id": fov_id, "blob": blob_name,
                           "error": error})
            continue

        # Neighbours from memory -- the whole reason to compute this at slide granularity.
        neighbours = [by_fov[n] for n in range(fov_id - NEIGHBOR_WINDOW, fov_id)
                      if by_fov.get(n) is not None]
        flag = diffuse_halo_flag(result, neighbours)

        row = {"fov_id": fov_id, "error": "",
               "present": _fmt(result.present),
               "diffuse_halo_flag": _fmt(flag),
               "present_folded": _fmt(bool(result.present) or bool(flag))}
        for name in RESULT_COLUMNS:
            if name not in row:
                row[name] = _fmt(getattr(result, name, None))
        rows.append(row)

    wall = time.time() - started
    limit = max(MIN_ERRORS_BEFORE_ABANDON, int(MAX_ERROR_FRACTION * len(fov_ids)))
    if len(errors) > limit:
        write_csv_atomic(Path(str(out_csv) + ".failed"), FIELDNAMES, rows)
        print(f"  ABANDONED {slide_id}: {len(errors)}/{len(fov_ids)} FOVs failed", flush=True)
    else:
        write_csv_atomic(out_csv, FIELDNAMES, rows)
    if errors:
        append_csv_rows(LOGS_DIR / "overexposure-errors.csv", ERROR_FIELDNAMES, errors)

    n_present = sum(1 for r in rows if r.get("present") == "True")
    n_folded = sum(1 for r in rows if r.get("present_folded") == "True")
    return rows, errors, wall, n_present, n_folded


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=8,
                        help="concurrent GCS streams; the pipeline-runtime measurements put "
                             "the GCS-bound optimum at 4-8")
    parser.add_argument("--slides", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-fovs", type=int, default=None)
    parser.add_argument("--fov-stride", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--include-non-catalog", action="store_true",
                        help="also score slides the index carries for regression only "
                             "(in_catalog=False), e.g. KTR-72502946")
    args = parser.parse_args()

    if not SLIDE_INDEX_JSON.exists():
        raise SystemExit(f"{SLIDE_INDEX_JSON} not found -- run the crowding subproject's "
                         "build_slide_index.py first")
    with open(SLIDE_INDEX_JSON, encoding="utf-8") as f:
        index = json.load(f)["slides"]
    with open(SLIDES_CSV, newline="", encoding="utf-8") as f:
        slides = [r for r in csv.DictReader(f)]

    # Only the 271 catalog slides; regression-only slides carry in_catalog=False.
    if not args.include_non_catalog:
        slides = [s for s in slides if s.get("in_catalog", "True") == "True"]
    if args.slides:
        wanted = set(args.slides)
        slides = [s for s in slides if s["slide_id"] in wanted]

    targets = []
    skipped = 0
    for slide in slides:
        slide_id = slide["slide_id"]
        entry = index.get(slide_id)
        if entry is None:
            print(f"WARNING {slide_id} missing from slide index; skipping", flush=True)
            continue
        fov_ids = sorted(int(i) for i in entry.get("fluorescent_fov_ids") or [])
        if args.fov_stride > 1:
            fov_ids = fov_ids[::args.fov_stride]
        if args.limit_fovs:
            fov_ids = fov_ids[:args.limit_fovs]

        out_csv = FOV_DIR / f"{slide_id}.csv"
        if not args.force and checkpoint_done(out_csv, len(fov_ids)):
            skipped += 1
            continue
        targets.append((entry["box"], slide_id, fov_ids, out_csv))
        if args.limit and len(targets) >= args.limit:
            break

    print(f"[overexposure] {len(targets)} slides to do, {skipped} already complete, "
          f"{args.threads} threads")
    if not targets:
        return 0

    progress_path = LOGS_DIR / "overexposure-progress.jsonl"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    recent, total_flagged, total_errors = [], 0, 0

    for i, (box, slide_id, fov_ids, out_csv) in enumerate(targets, 1):
        rows, errors, wall, n_present, n_folded = run_slide(box, slide_id, fov_ids, out_csv, args)
        total_flagged += n_present
        total_errors += len(errors)
        recent.append(wall)
        if len(recent) > 10:
            recent.pop(0)
        with open(progress_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"pass": "overexposure", "slide_id": slide_id,
                                "n_fovs": len(rows), "wall_s": round(wall, 2),
                                "n_errors": len(errors), "n_present": n_present,
                                "n_present_folded": n_folded, "ts": time.time()}) + "\n")
        eta = (len(targets) - i) * (sum(recent) / len(recent)) / 60.0
        print(f"[overexposure] {slide_id}: {len(rows)} FOVs in {wall:.1f}s "
              f"({len(rows) / wall:.2f} FOV/s), flagged {n_present} "
              f"(folded {n_folded}), {len(errors)} err | {i}/{len(targets)}, "
              f"ETA {eta:.0f} min", flush=True)

    print(f"[overexposure] done: {len(targets)} slides, {total_flagged} FOVs flagged, "
          f"{total_errors} errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
