"""Score every FOV of every Tanzania catalog slide with the crop-outlier statistic.

The second fluorescence arm. Unlike the overexposure pass this reads **no images** -- it reads
each slide's precomputed `fov_summary.csv` and compares every FOV's spot count to that slide's
own median/MAD baseline. 271 slides, ~2 minutes, a few MB of network.

**This pass writes `robust_zscore`, never a boolean.** The flagging threshold is applied at
aggregation time (`aggregate_slides.py --crop-outlier-z`, default 6.0), for two reasons: the
existing v1/v2 scripts are pinned at 2.0 and must not be forced to agree, and re-sweeping the
threshold then costs a local re-run of the aggregator instead of another pass over GCS.

**Detection-CSV coverage is split and that is recorded per slide.** Only 187 of the 271 catalog
slides have results under `TZ_MODEL_VERSION`; the other 84 resolve from the annotation bucket,
produced by a different (older) detector. The z-score is computed within a slide, so the model
difference largely cancels -- but `source` travels with every row so the claim stays checkable.

Usage:
    python scripts/tanzania-complete-081426/run_crop_outlier_pass.py
    python scripts/tanzania-complete-081426/run_crop_outlier_pass.py --metric n_positives
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FLUOR_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(FLUOR_ROOT))
sys.path.insert(0, str(FLUOR_ROOT / "scripts" / "crop-outlier-approach"))

from crop_counts import (  # noqa: E402
    METRICS,
    TZ_MODEL_VERSION,
    load_slide_metric_counts_with_source,
    ratio_to_median,
    robust_zscore,
    slide_baseline,
)

CROWDING_RESULTS = (FLUOR_ROOT.parent / "crowding-crenation" / "data" / "results"
                    / "tanzania-complete-081426")
SLIDES_CSV = CROWDING_RESULTS / "slides.csv"

RESULTS_DIR = FLUOR_ROOT / "data" / "results" / "tanzania-complete-081426"
FOV_DIR = RESULTS_DIR / "fov" / "crop-outlier"
SLIDE_CSV = RESULTS_DIR / "crop-outlier-slides.csv"

COUNTRY = "tanzania"

# Same vocabulary and thresholds as analyze_crop_outliers.py, so a flag means the same thing in
# both places. `same_slide_contamination` is absent by construction here: that flag warns when
# several *labeled* FOVs share a slide and thus pollute each other's baseline, whereas this pass
# scores every FOV of every slide, which is the condition itself rather than an exception to it.
SMALL_SLIDE_THRESHOLD = 20
MEAN_MEDIAN_DIVERGENCE_FRAC = 0.25

FOV_FIELDNAMES = ["fov_id", "n_detected", "ratio_to_median", "robust_zscore"]
SLIDE_FIELDNAMES = ["slide_id", "source", "metric", "model_version", "n_fovs_on_slide",
                    "baseline_mean", "baseline_std", "baseline_median", "baseline_mad",
                    "flags", "no_data_reason"]

# The verified split; a change means the buckets moved and the run should stop rather than
# quietly mixing a different population of slides into the comparison.
EXPECTED_SOURCES = {"tz_detection": 187, "annotation_bucket": 84}


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


def process_slide(slide_id, metric):
    counts, source = load_slide_metric_counts_with_source(slide_id, COUNTRY, metric=metric)
    summary = {"slide_id": slide_id, "source": source, "metric": metric,
               "model_version": TZ_MODEL_VERSION if source == "tz_detection" else "",
               "n_fovs_on_slide": "", "baseline_mean": "", "baseline_std": "",
               "baseline_median": "", "baseline_mad": "", "flags": "", "no_data_reason": ""}

    if not counts:
        summary["flags"] = "no_data"
        summary["no_data_reason"] = "no fov_summary.csv in either the detection or annotation bucket"
        return summary, []

    stats = slide_baseline(counts)
    mean, std, median, mad, n_fovs = (stats["mean"], stats["std"], stats["median"],
                                      stats["mad"], stats["n_fovs"])

    flags = []
    if n_fovs < SMALL_SLIDE_THRESHOLD:
        flags.append("small_slide")
    if not median or not mad:
        flags.append("zero_or_undefined_baseline")
    if median and abs(mean - median) > MEAN_MEDIAN_DIVERGENCE_FRAC * median:
        flags.append("mean_median_divergence")

    summary.update({"n_fovs_on_slide": n_fovs, "baseline_mean": round(mean, 3),
                    "baseline_std": round(std, 3), "baseline_median": median,
                    "baseline_mad": round(mad, 4), "flags": ";".join(flags)})

    rows = []
    for fov_id in sorted(counts):
        target = counts[fov_id]
        z = robust_zscore(target, stats)
        ratio = ratio_to_median(target, stats)
        rows.append({"fov_id": fov_id, "n_detected": target,
                     "ratio_to_median": "" if ratio is None else round(ratio, 4),
                     "robust_zscore": "" if z is None else round(z, 6)})
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metric", choices=METRICS, default="n_spots_detected")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--slides", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--assert-sources", action="store_true",
                        help=f"fail unless the source split is exactly {EXPECTED_SOURCES}")
    args = parser.parse_args()

    if not SLIDES_CSV.exists():
        raise SystemExit(f"{SLIDES_CSV} not found -- run the crowding subproject's "
                         "build_slide_index.py first")
    with open(SLIDES_CSV, newline="", encoding="utf-8") as f:
        slides = [r["slide_id"] for r in csv.DictReader(f)
                  if r.get("in_catalog", "True") == "True"]
    if args.slides:
        wanted = set(args.slides)
        slides = [s for s in slides if s in wanted]
    if args.limit:
        slides = slides[:args.limit]

    print(f"[crop-outlier] {len(slides)} slides, metric={args.metric}, "
          f"{args.threads} threads")
    started = time.time()

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        outcomes = list(pool.map(lambda s: (s, process_slide(s, args.metric)), slides))

    summaries = []
    for slide_id, (summary, rows) in outcomes:
        summaries.append(summary)
        if rows:
            write_csv_atomic(FOV_DIR / f"{slide_id}.csv", FOV_FIELDNAMES, rows)

    summaries.sort(key=lambda s: s["slide_id"])
    write_csv_atomic(SLIDE_CSV, SLIDE_FIELDNAMES, summaries)

    sources = Counter(s["source"] for s in summaries)
    flagged = Counter(flag for s in summaries for flag in (s["flags"] or "").split(";") if flag)
    elapsed = time.time() - started
    print(f"\nwrote {SLIDE_CSV}")
    print(f"wrote {sum(1 for _s, (_su, r) in outcomes if r)} per-FOV CSVs to {FOV_DIR}")
    print(f"sources: {dict(sources)}")
    print(f"slide flags: {dict(flagged) or '{}'}")
    print(f"elapsed: {elapsed:.1f}s")

    if args.assert_sources:
        actual = {k: v for k, v in sources.items() if k in EXPECTED_SOURCES or v}
        if actual != EXPECTED_SOURCES:
            print(f"\nFAILED: source split {actual} != expected {EXPECTED_SOURCES}. The "
                  "detection buckets have moved; re-verify before comparing slides.")
            return 1
        print(f"OK: source split matches {EXPECTED_SOURCES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
