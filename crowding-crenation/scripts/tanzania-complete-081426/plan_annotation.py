"""Which slides should be annotated next, to actually move the calibration numbers?

The calibration set is 661 FOVs from 11 slides, of which two slides supply 648. Two consequences,
both measured elsewhere in this dataset's README:

  * **Coverage.** Those two slides sit at the 77th and 93rd percentile of the 271-slide cohort, so
    the labelled pool barely describes the sparse 61% where most slides actually live.
  * **Stability.** Slide-grouped CV has effectively two folds, and they disagree sharply (held-out
    density rho +0.831 vs +0.454). More *slides* -- not more FOVs from the same slides -- is what
    makes a grouped estimate both honest and stable.

So this picks slides to maximise information per annotated FOV, on two criteria:

1. **Spread across the cohort.** Slides are binned into quantiles of their mean density score, and
   one slide is taken per bin. Quantile bins allocate roughly by cohort mass (so the sparse region,
   being most of the cohort, gets most of the slides) while still reaching both extremes, which is
   what threshold derivation needs.
2. **Internal diversity.** Within a bin, prefer the slide with the largest within-slide score
   spread. A slide whose FOVs span 0.1-0.7 yields labels across several buckets; a tight slide
   yields near-duplicates. Since within-slide spread (median std 0.076) is *smaller* than
   between-slide spread (0.140), this is exactly the axis along which slides differ in label value.

Within each chosen slide, the FOV sample is stratified across that slide's own score range rather
than taken at random, so a quarter of a slide still spans its whole range instead of clustering at
its mode.

Already-labelled slides are excluded -- re-annotating KTR-72502948 adds no new group and no new
coverage.

Outputs `annotation-plan.csv` (the per-FOV worklist, with blob URIs) and prints the projected
coverage change.

Usage:
    python scripts/tanzania-complete-081426/plan_annotation.py
    python scripts/tanzania-complete-081426/plan_annotation.py --slides 12 --fraction 0.25
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _slide_common import (  # noqa: E402
    CROWDING_FOV_DIR,
    RESULTS_DIR,
    ROOT,
    SLIDE_SUMMARY_CSV,
    dpc_gcs_path,
    load_slide_index,
    read_csv_dicts,
    write_csv_atomic,
)

sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    DEFAULT_SCORING_PARAMS,
    DENSITY_LEVELS,
    RESULTS_DIR as CALIB_DIR,
)

CALIB_FEATURES = CALIB_DIR / "features-v2.2-optimized.csv"
OUT_CSV = RESULTS_DIR / "annotation-plan.csv"

FIELDNAMES = ["slide_id", "fov_id", "blob", "bin", "slide_density_mean", "slide_density_std",
              "predicted_density_score", "predicted_density_label",
              "predicted_overlap_score", "predicted_overlap_label"]


def labelled_slides_and_counts(features_csv):
    """Slides already in the calibration set, and its manual density-label distribution."""
    import re

    slides, labels = set(), Counter()
    for row in read_csv_dicts(features_csv):
        match = re.match(r"(?:dpc|fluorescent)-\d+-(.+?)\.(?:png|bmp|tiff?|jpe?g)$",
                         row.get("filename", ""), re.IGNORECASE)
        slides.add(match.group(1) if match else row.get("filename", ""))
        labels[row["density_label"]] += 1
    return slides, labels


def quantile_bins(values, n_bins):
    """Bin edges at even quantiles of `values` (so each bin holds ~1/n_bins of the slides)."""
    ordered = sorted(values)
    return [ordered[max(0, min(len(ordered) - 1, round(i * len(ordered) / n_bins)))]
            for i in range(1, n_bins)]


def bin_of(value, edges):
    return sum(1 for e in edges if value >= e)


def stratified_fovs(rows, k):
    """`k` FOVs spanning this slide's own score range: sort by score, take even positions.

    Deliberately not random -- a quarter of a slide taken at random clusters at the slide's mode,
    which is where the model already has the most information.
    """
    ordered = sorted(rows, key=lambda r: r["_score"])
    if k >= len(ordered):
        return ordered
    step = len(ordered) / k
    return [ordered[min(len(ordered) - 1, int(i * step + step / 2))] for i in range(k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-csv", default=str(SLIDE_SUMMARY_CSV))
    parser.add_argument("--features", default=str(CALIB_FEATURES))
    parser.add_argument("--params", default=str(DEFAULT_SCORING_PARAMS))
    parser.add_argument("--slides", type=int, default=8,
                        help="slides to recommend (one per quantile bin); default 8")
    parser.add_argument("--fraction", type=float, default=0.25,
                        help="fraction of each slide's FOVs to annotate; default 0.25")
    parser.add_argument("--out-csv", default=str(OUT_CSV))
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)

    already, labelled_dist = labelled_slides_and_counts(args.features)
    index = load_slide_index()["slides"]

    slides = []
    for row in read_csv_dicts(args.summary_csv):
        try:
            slides.append({"slide_id": row["slide_id"],
                           "mean": float(row["density_mean"]),
                           "std": float(row["density_std"]),
                           "bucket": row["density_bucket"]})
        except (KeyError, TypeError, ValueError):
            continue
    print(f"cohort: {len(slides)} slides | already labelled: "
          f"{len(already & {s['slide_id'] for s in slides})} of them")
    print(f"current labelled pool: {sum(labelled_dist.values())} FOVs from {len(already)} slides")
    print(f"  manual density labels: "
          f"{ {k: labelled_dist.get(k, 0) for k in DENSITY_LEVELS} }\n")

    edges = quantile_bins([s["mean"] for s in slides], args.slides)
    candidates = [s for s in slides if s["slide_id"] not in already]

    chosen = []
    for b in range(args.slides):
        pool = [s for s in candidates if bin_of(s["mean"], edges) == b]
        if not pool:
            continue
        # Most internally diverse slide in this bin -> most buckets covered per annotated FOV.
        pick = max(pool, key=lambda s: s["std"])
        pick["bin"] = b
        chosen.append(pick)

    print(f"recommended {len(chosen)} slides (one per quantile bin of slide mean density):")
    print(f"  {'bin':>3s} {'slide_id':16s} {'mean':>7s} {'std':>7s} {'bucket':16s} "
          f"{'n_fovs':>7s} {'annotate':>9s}")

    out_rows = []
    projected = Counter(labelled_dist)
    for pick in chosen:
        slide_id = pick["slide_id"]
        fov_rows = []
        for r in read_csv_dicts(CROWDING_FOV_DIR / f"{slide_id}.csv"):
            if (r.get("error") or "").strip():
                continue
            try:
                r["_score"] = float(r["density_score"])
            except (TypeError, ValueError):
                continue
            fov_rows.append(r)
        k = max(1, round(args.fraction * len(fov_rows)))
        sample = stratified_fovs(fov_rows, k)
        print(f"  {pick['bin']:3d} {slide_id:16s} {pick['mean']:7.3f} {pick['std']:7.3f} "
              f"{pick['bucket']:16s} {len(fov_rows):7d} {len(sample):9d}")

        box = index[slide_id]["box"]
        for r in sample:
            fov_id = int(r["fov_id"])
            out_rows.append({
                "slide_id": slide_id, "fov_id": fov_id,
                "blob": str(dpc_gcs_path(box, slide_id, fov_id)),
                "bin": pick["bin"],
                "slide_density_mean": round(pick["mean"], 4),
                "slide_density_std": round(pick["std"], 4),
                "predicted_density_score": r["density_score"],
                "predicted_density_label": r["density_label"],
                "predicted_overlap_score": r["overlap_score"],
                "predicted_overlap_label": r["overlap_label"],
            })
            projected[r["density_label"]] += 1

    write_csv_atomic(args.out_csv, FIELDNAMES, out_rows)

    print(f"\ntotal to annotate: {len(out_rows)} FOVs across {len(chosen)} slides")
    print(f"labelled pool would go {sum(labelled_dist.values())} -> "
          f"{sum(labelled_dist.values()) + len(out_rows)} FOVs, "
          f"{len(already)} -> {len(already) + len(chosen)} slides "
          f"({len(already) + len(chosen)} grouped-CV folds)")

    print("\ndensity-label coverage (predicted labels used as a proxy for the new FOVs):")
    print(f"  {'bucket':16s} {'now':>6s} {'added':>7s} {'after':>7s}")
    for level in DENSITY_LEVELS:
        now = labelled_dist.get(level, 0)
        after = projected.get(level, 0)
        print(f"  {level:16s} {now:6d} {after - now:7d} {after:7d}")
    print(f"\nwrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
