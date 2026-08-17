"""Collapse the per-FOV passes into one row per slide.

Reads the three passes' per-FOV CSVs (two of them from the fluorescence subproject) and writes
`slide-summary.csv`: 271 rows, one per catalog slide, always -- a slide missing a pass gets that
pass's columns blank and a `status` saying so, rather than disappearing from the table.

Local only, no network, seconds. That is deliberate: every *decision* in this analysis lives
here rather than in the passes, so re-deciding one costs a re-run of this script instead of
another sweep over 342 GB. The decisions are:

**Gated FOVs count as 0.0 on both axes.** A FOV that trips the empty-field gate is a near-empty
field, so it contributes zero severity to its slide's mean. Two alternative readings are also
computed and stored -- `*_mean_raw` (gate ignored, the composite's own number) and
`*_mean_ungated` (gated FOVs dropped from the mean entirely) -- so the choice is visible and
reversible without recomputing anything.

**The slide bucket comes from bucketing the slide mean** with the same per-FOV thresholds the
model was fit with. Worth being explicit that this is not what those thresholds were fit for: a
mean over ~324 FOVs has far less variance than a single FOV, so slide means compress toward the
middle of the scale. `*_bucket_modal` (the most common per-FOV bucket) and `*_std` are carried
alongside precisely so a compressed grid can be told apart from a genuinely uniform cohort.

**The crop-outlier flag threshold is applied here**, from `--crop-outlier-z` (default 6.0), and
the value used is recorded in every row. The pass itself stores only the z-score.

Usage:
    python scripts/tanzania-complete-081426/aggregate_slides.py
    python scripts/tanzania-complete-081426/aggregate_slides.py --crop-outlier-z 2
"""
import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _slide_common import (  # noqa: E402
    CROWDING_FOV_DIR,
    FLUORESCENCE_RESULTS_DIR,
    ROOT,
    SLIDE_SUMMARY_CSV,
    load_slide_index,
    load_slides,
    read_csv_dicts,
    write_csv_atomic,
)

sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    DEFAULT_SCORING_PARAMS,
    DENSITY_LEVELS,
    OVERLAP_LEVELS,
)
from src.composite_v2 import bucket  # noqa: E402

DEFAULT_CROP_OUTLIER_Z = 6.0

FIELDNAMES = [
    "slide_id", "box", "truth", "truth_warn", "train_test_split", "region",
    "n_fovs_expected", "n_fovs_scored", "n_errors_crowding", "n_empty_field_gated",
    "density_mean", "overlap_mean", "combined_score", "density_bucket", "overlap_bucket",
    "density_bucket_modal", "overlap_bucket_modal",
    "density_mean_raw", "overlap_mean_raw", "density_mean_ungated", "overlap_mean_ungated",
    "density_std", "overlap_std",
    "n_fovs_overexposure", "n_errors_overexposure", "n_flagged_overexposure",
    "frac_flagged_overexposure", "n_flagged_overexposure_folded",
    "n_fovs_crop_outlier", "n_flagged_crop_outlier", "frac_flagged_crop_outlier",
    "crop_outlier_z_threshold", "crop_outlier_source", "crop_outlier_metric",
    "crop_outlier_baseline_median", "crop_outlier_baseline_mad", "crop_outlier_flags",
    "status", "params_version",
]

ROUND = 6


def _num(value):
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_true(value):
    return str(value).strip().lower() == "true"


def _mean(values):
    return statistics.fmean(values) if values else None


def _stdev(values):
    return statistics.stdev(values) if len(values) >= 2 else (0.0 if values else None)


def _r(value):
    return "" if value is None else round(value, ROUND)


def modal_bucket(labels, levels):
    """Most common label, breaking ties by the ordinal scale rather than insertion order.

    `Counter.most_common` resolves ties by first-seen order, which depends on FOV iteration
    order and so is not reproducible across runs. Ties fall to the *lower* severity here.
    """
    if not labels:
        return ""
    counts = Counter(labels)
    return min(counts, key=lambda label: (-counts[label], levels.index(label)))


def summarize_crowding(rows, params):
    """Per-FOV crowding rows -> the density/Rouleaux half of a slide row."""
    ok = [r for r in rows if not (r.get("error") or "").strip()]
    errors = len(rows) - len(ok)

    gated = [r for r in ok if _is_true(r.get("empty_field_gated"))]
    d_raw = [_num(r["density_score"]) for r in ok]
    o_raw = [_num(r["overlap_score"]) for r in ok]
    d_raw = [v for v in d_raw if v is not None]
    o_raw = [v for v in o_raw if v is not None]

    # The headline reading: a gated FOV is a near-empty field and contributes zero severity.
    d_eff = [0.0 if _is_true(r.get("empty_field_gated")) else _num(r["density_score"])
             for r in ok]
    o_eff = [0.0 if _is_true(r.get("empty_field_gated")) else _num(r["overlap_score"])
             for r in ok]
    d_eff = [v for v in d_eff if v is not None]
    o_eff = [v for v in o_eff if v is not None]

    d_ungated = [_num(r["density_score"]) for r in ok
                 if not _is_true(r.get("empty_field_gated"))]
    o_ungated = [_num(r["overlap_score"]) for r in ok
                 if not _is_true(r.get("empty_field_gated"))]
    d_ungated = [v for v in d_ungated if v is not None]
    o_ungated = [v for v in o_ungated if v is not None]

    density_mean, overlap_mean = _mean(d_eff), _mean(o_eff)
    combined = None if density_mean is None or overlap_mean is None else density_mean + overlap_mean

    out = {
        "n_fovs_scored": len(ok),
        "n_errors_crowding": errors,
        "n_empty_field_gated": len(gated),
        "density_mean": _r(density_mean),
        "overlap_mean": _r(overlap_mean),
        "combined_score": _r(combined),
        "density_mean_raw": _r(_mean(d_raw)),
        "overlap_mean_raw": _r(_mean(o_raw)),
        "density_mean_ungated": _r(_mean(d_ungated)),
        "overlap_mean_ungated": _r(_mean(o_ungated)),
        "density_std": _r(_stdev(d_eff)),
        "overlap_std": _r(_stdev(o_eff)),
        "density_bucket": "",
        "overlap_bucket": "",
        "density_bucket_modal": modal_bucket([r["density_label"] for r in ok], DENSITY_LEVELS),
        "overlap_bucket_modal": modal_bucket([r["overlap_label"] for r in ok], OVERLAP_LEVELS),
    }
    if density_mean is not None:
        out["density_bucket"] = bucket(density_mean, params["density"]["bucket_thresholds"],
                                       params["density"]["bucket_labels"])
    if overlap_mean is not None:
        out["overlap_bucket"] = bucket(overlap_mean, params["overlap"]["bucket_thresholds"],
                                       params["overlap"]["bucket_labels"])
    return out


def summarize_overexposure(rows):
    ok = [r for r in rows if not (r.get("error") or "").strip()]
    n_present = sum(1 for r in ok if _is_true(r.get("present")))
    n_folded = sum(1 for r in ok if _is_true(r.get("present_folded")))
    return {
        "n_fovs_overexposure": len(ok),
        "n_errors_overexposure": len(rows) - len(ok),
        "n_flagged_overexposure": n_present,
        "frac_flagged_overexposure": _r(n_present / len(ok)) if ok else "",
        "n_flagged_overexposure_folded": n_folded,
    }


def summarize_crop_outlier(rows, slide_summary, threshold):
    zs = [_num(r.get("robust_zscore")) for r in rows]
    scored = [z for z in zs if z is not None]
    n_flagged = sum(1 for z in scored if z >= threshold)
    out = {
        "n_fovs_crop_outlier": len(rows),
        "n_flagged_crop_outlier": n_flagged if scored else "",
        "frac_flagged_crop_outlier": _r(n_flagged / len(scored)) if scored else "",
        "crop_outlier_z_threshold": threshold,
    }
    if slide_summary:
        out.update({
            "crop_outlier_source": slide_summary.get("source", ""),
            "crop_outlier_metric": slide_summary.get("metric", ""),
            "crop_outlier_baseline_median": slide_summary.get("baseline_median", ""),
            "crop_outlier_baseline_mad": slide_summary.get("baseline_mad", ""),
            "crop_outlier_flags": slide_summary.get("flags", ""),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", default=str(DEFAULT_SCORING_PARAMS))
    parser.add_argument("--crop-outlier-z", type=float, default=DEFAULT_CROP_OUTLIER_Z,
                        help=f"robust_zscore at or above which a FOV counts as flagged "
                             f"(default {DEFAULT_CROP_OUTLIER_Z}; the pass stores the raw "
                             f"z-score so this is free to re-sweep)")
    parser.add_argument("--fluorescence-results-dir", default=str(FLUORESCENCE_RESULTS_DIR))
    parser.add_argument("--out-csv", default=str(SLIDE_SUMMARY_CSV))
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)
    params_version = params.get("version", "")

    slides = [s for s in load_slides() if s.get("in_catalog", "True") == "True"]
    index = load_slide_index()["slides"]

    fluor_dir = Path(args.fluorescence_results_dir)
    overexp_dir = fluor_dir / "fov" / "overexposure"
    crop_dir = fluor_dir / "fov" / "crop-outlier"
    crop_slides_csv = fluor_dir / "crop-outlier-slides.csv"

    crop_summaries = {}
    if crop_slides_csv.exists():
        crop_summaries = {r["slide_id"]: r for r in read_csv_dicts(crop_slides_csv)}

    out_rows = []
    status_counts = Counter()
    for slide in slides:
        slide_id = slide["slide_id"]
        entry = index.get(slide_id, {})

        row = {
            "slide_id": slide_id,
            "box": slide.get("box", ""),
            "truth": slide.get("truth", ""),
            "truth_warn": slide.get("truth_warn", ""),
            "train_test_split": slide.get("train_test_split", ""),
            "region": slide.get("region", ""),
            "n_fovs_expected": entry.get("n_dpc", ""),
            "params_version": params_version,
            "crop_outlier_z_threshold": args.crop_outlier_z,
        }
        missing = []

        crowding_csv = CROWDING_FOV_DIR / f"{slide_id}.csv"
        if crowding_csv.exists():
            row.update(summarize_crowding(read_csv_dicts(crowding_csv), params))
        else:
            missing.append("crowding")

        overexp_csv = overexp_dir / f"{slide_id}.csv"
        if overexp_csv.exists():
            row.update(summarize_overexposure(read_csv_dicts(overexp_csv)))
        else:
            missing.append("overexposure")

        crop_csv = crop_dir / f"{slide_id}.csv"
        if crop_csv.exists():
            row.update(summarize_crop_outlier(read_csv_dicts(crop_csv),
                                              crop_summaries.get(slide_id),
                                              args.crop_outlier_z))
        else:
            missing.append("crop_outlier")

        if not missing:
            expected = entry.get("n_dpc")
            complete = expected in (None, "") or row.get("n_fovs_scored") == expected
            row["status"] = "complete" if complete else "partial"
        elif len(missing) == 3:
            row["status"] = "missing_all"
        else:
            row["status"] = "missing_" + "+".join(missing)
        status_counts[row["status"]] += 1
        out_rows.append(row)

    out_rows.sort(key=lambda r: r["slide_id"])
    write_csv_atomic(args.out_csv, FIELDNAMES, out_rows)

    print(f"wrote {len(out_rows)} slide rows to {args.out_csv}")
    print(f"params: {params_version} | crop-outlier z >= {args.crop_outlier_z}")
    print(f"status: {dict(status_counts)}")

    scored = [r for r in out_rows if r.get("density_mean") not in (None, "")]
    if scored:
        grid = Counter((r["density_bucket"], r["overlap_bucket"]) for r in scored)
        print(f"\nslides with crowding scores: {len(scored)}")
        print(f"density buckets:  {dict(Counter(r['density_bucket'] for r in scored))}")
        print(f"Rouleaux buckets: {dict(Counter(r['overlap_bucket'] for r in scored))}")
        print(f"occupied grid cells: {len(grid)}/25")
        gate_total = sum(int(r.get("n_empty_field_gated") or 0) for r in scored)
        print(f"empty-field-gated FOVs across those slides: {gate_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
