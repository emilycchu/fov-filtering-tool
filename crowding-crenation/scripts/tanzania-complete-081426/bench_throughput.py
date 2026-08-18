"""Measure the (processes x threads) throughput knee before committing to the full sweep.

Verification step 4. Threads beat processes per-core here, but they saturate -- on the dev
laptop throughput peaked at 8 threads / ~3.8 FOV/s (2.9x over serial) and *fell* by 24, because
numpy's per-operation dispatch holds the interpreter lock. A 32-vCPU VM therefore needs processes
on top of threads, and the right count is a property of that machine's cores and memory
bandwidth, not something to assume from the laptop.

Writes no checkpoints and no per-FOV CSVs -- it scores into the void, so it cannot pollute or
be mistaken for real output. Emits `bench-throughput.json` with the winning config so a headless
run can read it back instead of hardcoding a guess.

Usage:
    python scripts/tanzania-complete-081426/bench_throughput.py --fovs 96
    python scripts/tanzania-complete-081426/bench_throughput.py --threads-list 4 8 16 --procs-list 1 2 4
"""
import argparse
import json
import os
import subprocess
import sys
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _slide_common import (  # noqa: E402
    RESULTS_DIR,
    ROOT,
    dpc_gcs_path,
    load_slide_index,
    load_slides,
    with_retry,
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

OUT_JSON = RESULTS_DIR / "bench-throughput.json"


def pick_work(n_fovs, offset=0):
    """A contiguous run of real FOVs, spread across slides so no single slide's cache dominates."""
    index = load_slide_index()["slides"]
    slides = [s for s in load_slides() if s.get("in_catalog", "True") == "True"]
    work = []
    for slide in slides:
        entry = index.get(slide["slide_id"])
        if not entry:
            continue
        for fov_id in sorted(int(i) for i in entry["dpc_fov_ids"]):
            work.append((entry["box"], slide["slide_id"], fov_id))
            if len(work) >= n_fovs + offset:
                return work[offset:offset + n_fovs]
    return work[offset:offset + n_fovs]


def score(item, params, lbp_step, blur_downsample):
    box, slide_id, fov_id = item
    image = with_retry(load_image, dpc_gcs_path(box, slide_id, fov_id), grayscale=True)
    features = compute_features(image, lbp_step=lbp_step, blur_downsample=blur_downsample)
    return score_features_v2(features, params)["density_score"]


def run_threads(work, params, threads):
    lbp_step = lbp_step_from_params(params)
    blur_downsample = blur_downsample_from_params(params)
    started = time.time()
    if threads > 1:
        with ThreadPool(threads) as pool:
            pool.map(lambda item: score(item, params, lbp_step, blur_downsample), work)
    else:
        for item in work:
            score(item, params, lbp_step, blur_downsample)
    return time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", default=str(DEFAULT_SCORING_PARAMS))
    parser.add_argument("--fovs", type=int, default=96,
                        help="FOVs per configuration (per process)")
    parser.add_argument("--threads-list", nargs="*", type=int, default=[4, 8, 16, 24])
    parser.add_argument("--procs-list", nargs="*", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--shard", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--shard-threads", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--out-json", default=str(OUT_JSON))
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)

    # Child mode: score one shard's slice and print its own rate for the parent to sum.
    if args.shard is not None:
        work = pick_work(args.fovs, offset=args.shard * args.fovs)
        elapsed = run_threads(work, params, args.shard_threads)
        print(json.dumps({"shard": args.shard, "n": len(work), "elapsed_s": elapsed,
                          "fov_per_s": len(work) / elapsed}))
        return 0

    print(f"cpu_count={os.cpu_count()} params={params.get('version')} "
          f"lbp_step={lbp_step_from_params(params)} "
          f"blur_downsample={blur_downsample_from_params(params)}")
    print(f"benchmarking {args.fovs} FOVs per configuration\n")

    results = []

    print("threads (single process):")
    best_threads, best_rate = None, 0.0
    for threads in args.threads_list:
        work = pick_work(args.fovs)
        elapsed = run_threads(work, params, threads)
        rate = len(work) / elapsed
        results.append({"procs": 1, "threads": threads, "fov_per_s": rate,
                        "elapsed_s": elapsed, "n": len(work)})
        marker = ""
        if rate > best_rate:
            best_threads, best_rate, marker = threads, rate, "  <- best so far"
        print(f"  {threads:3d} threads: {elapsed:6.1f}s  {rate:6.2f} FOV/s{marker}", flush=True)

    print(f"\nprocesses x {best_threads} threads:")
    best = {"procs": 1, "threads": best_threads, "fov_per_s": best_rate}
    for procs in args.procs_list:
        if procs == 1:
            print(f"  {procs:3d} proc  : {best_rate:6.2f} FOV/s (from above)")
            continue
        cmd_base = [sys.executable, str(Path(__file__).resolve()), "--params", args.params,
                    "--fovs", str(args.fovs), "--shard-threads", str(best_threads)]
        started = time.time()
        children = [subprocess.Popen(cmd_base + ["--shard", str(i)], cwd=str(ROOT),
                                     stdout=subprocess.PIPE, text=True)
                    for i in range(procs)]
        outs = [child.communicate()[0] for child in children]
        elapsed = time.time() - started
        total = 0
        for out in outs:
            for line in out.strip().splitlines():
                try:
                    total += json.loads(line)["n"]
                except (ValueError, KeyError):
                    pass
        rate = total / elapsed if elapsed else 0.0
        results.append({"procs": procs, "threads": best_threads, "fov_per_s": rate,
                        "elapsed_s": elapsed, "n": total})
        marker = ""
        if rate > best["fov_per_s"]:
            best = {"procs": procs, "threads": best_threads, "fov_per_s": rate}
            marker = "  <- best so far"
        print(f"  {procs:3d} procs : {elapsed:6.1f}s  {rate:6.2f} FOV/s "
              f"({total} FOVs){marker}", flush=True)

    total_fovs = 87799
    hours = total_fovs / best["fov_per_s"] / 3600 if best["fov_per_s"] else float("inf")
    print(f"\nbest: --procs {best['procs']} --threads {best['threads']} "
          f"({best['fov_per_s']:.2f} FOV/s)")
    print(f"projected DPC pass over {total_fovs} FOVs: {hours:.2f} h")

    out = {"cpu_count": os.cpu_count(), "params_version": params.get("version"),
           "fovs_per_config": args.fovs, "results": results, "best": best,
           "projected_dpc_hours": hours}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
