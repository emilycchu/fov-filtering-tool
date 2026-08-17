"""Score every DPC FOV of every Tanzania catalog slide for density and Rouleaux severity.

87,799 FOVs across 271 slides, streamed from GCS, nothing written to disk but CSVs.

**Parallelism: threads inside processes.** `data/results/pipeline-runtime/README.md` establishes
that a thread pool beats a process pool here by up to 2.4x -- the work releases the GIL
throughout (blob download is network IO, the compute is cv2/numpy) and threads share one address
space instead of each holding two transient float32 copies of a 7.84M-pixel image. But threads
saturate: measured on 40 FOVs at the adopted params, throughput peaks at **8 threads / ~3.8
FOV/s (2.9x over serial)** and *falls* by 24, because numpy's per-operation dispatch holds the
interpreter lock. At 87,799 FOVs that ceiling is the whole problem -- 3.8 FOV/s is ~6.5 hours.

So `--procs` fans out over processes, each running its own `--threads`-sized thread pool and its
own GIL, on a disjoint shard of the slide list. Sharding is by slide, which keeps the checkpoint
unit intact: no two processes ever touch the same slide, so no locking and no partial-file races.

**Per-FOV output keeps the full feature vector, not just the scores.** Re-deciding the gate rule,
the bucket edges, or the params version must never mean re-streaming 342 GB, and a label-free
check of whether the calibration's p2/p98 ranges clip this cohort needs the raw features.

Usage:
    # smoke test, one slide
    python scripts/tanzania-complete-081426/run_crowding_pass.py --slides NKR-72502209

    # the full run on a 32-vCPU VM
    python scripts/tanzania-complete-081426/run_crowding_pass.py --procs 4 --threads 8
"""
import argparse
import json
import subprocess
import sys
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _slide_common import (  # noqa: E402
    CROWDING_FOV_DIR,
    LOGS_DIR,
    ROOT,
    add_selection_args,
    append_csv_rows,
    dpc_gcs_path,
    iter_target_slides,
    mirror_to_gcs,
    with_retry,
    write_csv_atomic,
)

sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    DEFAULT_SCORING_PARAMS,
    blur_downsample_from_params,
    compute_features,
    lbp_step_from_params,
    load_image,
)
from score_fov_v2 import score_features_v2  # noqa: E402

FEATURE_COLUMNS = ["coverage", "otsu_threshold", "otsu_separability", "saturation_score",
                   "lbp_entropy", "glcm_contrast", "edge_density_unmasked", "tile_glcm_cv",
                   "tile_glcm_patchiness"]

FIELDNAMES = (["fov_id", "density_score", "density_label", "overlap_score", "overlap_label",
               "empty_field_gated"] + FEATURE_COLUMNS + ["error"])

ERROR_FIELDNAMES = ["slide_id", "fov_id", "blob", "error"]

# 10 significant digits: enough for the <1e-9 golden-value regression against the committed
# tanzania-080526 scores, and still ~28 MB for the whole per-FOV tree.
FLOAT_FMT = "{:.10g}"

# A slide is abandoned rather than marked done past this many failures, so a network wobble
# cannot freeze a bad slide as permanently complete.
MAX_ERROR_FRACTION = 0.05
MIN_ERRORS_BEFORE_ABANDON = 3


def _fmt(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return FLOAT_FMT.format(value)
    return value


def score_one_fov(box, slide_id, fov_id, params, lbp_step, blur_downsample):
    """One FOV -> one output row. Never raises; a failure becomes a row with `error` set."""
    path = dpc_gcs_path(box, slide_id, fov_id)
    try:
        # Retry wraps the download only. The ~0.5 s of feature computation below is pure CPU
        # and cannot fail transiently, so including it would re-burn work that succeeded.
        image = with_retry(load_image, path, grayscale=True)
        features = compute_features(image, lbp_step=lbp_step, blur_downsample=blur_downsample)
        scored = score_features_v2(features, params)
    except Exception as exc:  # noqa: BLE001 -- one bad FOV must not end a 271-slide run
        return {"fov_id": fov_id, "error": f"{type(exc).__name__}: {exc}"}, str(path)

    row = {"fov_id": fov_id,
           "density_score": _fmt(scored["density_score"]),
           "density_label": scored["density_label"],
           "overlap_score": _fmt(scored["overlap_score"]),
           "overlap_label": scored["overlap_label"],
           "empty_field_gated": _fmt(scored["empty_field_gated"]),
           "error": ""}
    row.update({name: _fmt(features.get(name)) for name in FEATURE_COLUMNS})
    return row, None


def run_slide(slide, box, fov_ids, out_csv, params, args):
    """Score one slide's FOVs and write its checkpoint. Returns (rows, errors, wall_s)."""
    slide_id = slide["slide_id"]
    lbp_step = lbp_step_from_params(params)
    blur_downsample = blur_downsample_from_params(params)
    started = time.time()

    def work(fov_id):
        return score_one_fov(box, slide_id, fov_id, params, lbp_step, blur_downsample)

    if args.threads > 1:
        with ThreadPool(args.threads) as pool:
            results = pool.map(work, fov_ids)
    else:
        results = [work(fov_id) for fov_id in fov_ids]

    rows = [row for row, _blob in results]
    rows.sort(key=lambda r: r["fov_id"])
    errors = [{"slide_id": slide_id, "fov_id": row["fov_id"], "blob": blob,
               "error": row["error"]}
              for row, blob in results if row.get("error")]

    wall = time.time() - started
    limit = max(MIN_ERRORS_BEFORE_ABANDON, int(MAX_ERROR_FRACTION * len(fov_ids)))
    if len(errors) > limit:
        # Not marked done: write a .failed sidecar so a resumed run retries this slide.
        write_csv_atomic(Path(str(out_csv) + ".failed"), FIELDNAMES, rows)
        print(f"  ABANDONED {slide_id}: {len(errors)}/{len(fov_ids)} FOVs failed (> {limit})",
              flush=True)
    else:
        write_csv_atomic(out_csv, FIELDNAMES, rows)
        if args.mirror_gcs:
            mirror_to_gcs(out_csv)

    if errors:
        append_csv_rows(LOGS_DIR / f"{args.pass_name}-errors.csv", ERROR_FIELDNAMES, errors)
    return rows, errors, wall


def run_shard(args, params):
    targets, progress = iter_target_slides(args, args.pass_name, CROWDING_FOV_DIR,
                                           channel="dpc")
    if args.shard_count > 1:
        targets = [t for i, t in enumerate(targets) if i % args.shard_count == args.shard]
        progress.total = len(targets) + progress.skipped
        progress.pass_name = f"{args.pass_name}:{args.shard}"

    if not targets:
        print(f"[{progress.pass_name}] nothing to do "
              f"({progress.skipped} slides already complete)")
        return 0

    print(f"[{progress.pass_name}] {len(targets)} slides, {args.threads} threads, "
          f"params={Path(args.params).name}", flush=True)

    total_errors = 0
    for slide, box, fov_ids, out_csv in targets:
        rows, errors, wall = run_slide(slide, box, fov_ids, out_csv, params, args)
        total_errors += len(errors)
        progress.record(slide["slide_id"], len(rows), wall, n_errors=len(errors))

    summary = progress.summary()
    print(f"[{progress.pass_name}] done: {json.dumps(summary)} total_errors={total_errors}")
    return 0


def spawn_shards(args):
    """Re-invoke this script once per shard, each with its own interpreter and GIL.

    Subprocesses rather than `multiprocessing.Process` so a child that dies cannot take the
    parent's bookkeeping with it, and so each shard's stdout/exit code stands alone.
    """
    base = [sys.executable, str(Path(__file__).resolve())]
    for flag in ("--params", "--threads", "--fov-stride"):
        base += [flag, str(getattr(args, flag.lstrip("-").replace("-", "_")))]
    if args.slides:
        base += ["--slides", *args.slides]
    if args.limit:
        base += ["--limit", str(args.limit)]
    if args.limit_fovs:
        base += ["--limit-fovs", str(args.limit_fovs)]
    if args.force:
        base += ["--force"]
    if args.mirror_gcs:
        base += ["--mirror-gcs"]

    children = []
    for shard in range(args.procs):
        cmd = base + ["--shard", str(shard), "--shard-count", str(args.procs)]
        children.append(subprocess.Popen(cmd, cwd=str(ROOT)))
    codes = [child.wait() for child in children]
    failed = [i for i, code in enumerate(codes) if code != 0]
    if failed:
        print(f"shards {failed} exited non-zero: {codes}")
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", default=str(DEFAULT_SCORING_PARAMS),
                        help="defaults to the deployed v2.2-optimized fit; lbp_step and "
                             "blur_downsample are read out of this file, never hardcoded")
    parser.add_argument("--threads", type=int, default=8,
                        help="threads per process (8 is the measured throughput knee)")
    parser.add_argument("--procs", type=int, default=1,
                        help="processes to fan out over; each runs its own thread pool on a "
                             "disjoint shard of slides. >1 re-invokes this script per shard")
    parser.add_argument("--shard", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--shard-count", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--mirror-gcs", action="store_true",
                        help="copy each finished slide CSV to the shared bucket (Spot safety)")
    parser.add_argument("--pass-name", default="crowding", help=argparse.SUPPRESS)
    add_selection_args(parser)
    args = parser.parse_args()

    if args.procs > 1 and args.shard_count == 1:
        return spawn_shards(args)

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)
    print(f"params: {params.get('version')} lbp_step={lbp_step_from_params(params)} "
          f"blur_downsample={blur_downsample_from_params(params)}")
    return run_shard(args, params)


if __name__ == "__main__":
    sys.exit(main())
