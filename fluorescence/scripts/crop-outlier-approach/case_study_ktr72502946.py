"""Crop-outlier case study: KTR-72502946 (Tanzania), full-slide.

This slide was one of two samples (`KIT-62500763`, `KTR-72502946`) that used to show up as
`no_data` in both v1's results.csv and v2's boundary_negatives.csv -- `tanzania_02032026`'s own
`detection_results/` tree never got this sample mirrored into it. It turns out to have full
detection data all along, in `gs://malaria-annotation-web` (`samples/KTR-72502946/fov_summary.csv`
-- see crop_counts.py's module docstring for the fallback that now finds it).

This script scores every one of this slide's 324 FOVs against the whole-slide baseline
(crop_counts.slide_baseline -- median/MAD over every FOV, the target included, not left out),
the same statistic v1/v2 use, and highlights where the two labeled FOVs (54, 198, both
spot_truth=yes in the 76-row label set) and the two v2 boundary FOVs (1, 324) land in the full
per-slide distribution.

With --previews, also renders raw fluorescence thumbnails for the three FOVs the write-up
discusses (54, 198, 308), streamed from GCS via src.gcs_fov_multi.load_fov_image -- same
downscale-to-700px + caption convention as scripts/render_fn_fp_previews.py, minus the
overexposure-mask overlay (this analysis is about crop counts, not pixel masks).

With --thumbs, renders a smaller (260px) contact-sheet thumbnail for each of the top
TOP_N_THUMBS FOVs by robust_zscore, so the write-up's top-15 table can carry an image per row
and the halo-vs-debris split across the whole flagged tail is eyeballable at a glance rather
than only for the three hand-picked FOVs. Same source images and single-line caption; kept in
previews/thumbs/ and deliberately small (~30-60KB each) since 15 full 700px previews would add
~10MB to the repo.

Usage:
    python scripts/crop-outlier-approach/case_study_ktr72502946.py
    python scripts/crop-outlier-approach/case_study_ktr72502946.py --previews
    python scripts/crop-outlier-approach/case_study_ktr72502946.py --thumbs
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # this dir, for crop_counts (hyphenated package)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root, for src.*

from crop_counts import load_slide_metric_counts, slide_baseline  # noqa: E402  (needs sys.path first)

SAMPLE_ID = "KTR-72502946"
COUNTRY = "Tanzania"
METRIC = "n_spots_detected"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "results" / "crop-outlier-approach" / "crop-outlier-v2"
OUT_CSV = OUT_DIR / "case-study-KTR-72502946.csv"
PREVIEW_DIR = OUT_DIR / "previews"
THUMB_DIR = PREVIEW_DIR / "thumbs"

LABELED_FOVS = {54: "yes", 198: "yes"}  # from data/labels/overexposure-diverse-080726.csv
BOUNDARY_FOVS = (1, 324)
PREVIEW_FOVS = (54, 198, 308)  # the three FOVs the write-up discusses

OUTLIER_ZSCORE = 2.0
TOP_N = 15  # size of the write-up's "top N FOVs by robust_zscore" table, and of --thumbs
PREVIEW_PX = 700  # full previews, as embedded inline in the write-up
THUMB_PX = 260  # contact-sheet thumbs, as embedded in the top-N table's rows


def top_n_rows(rows, n=TOP_N):
    """The write-up's top-N-by-robust_zscore ordering, shared by the printout and --thumbs so
    the table rows and the rendered thumbs can't drift apart."""
    scored = (r for r in rows if r["robust_zscore"] != "")
    return sorted(scored, key=lambda r: r["robust_zscore"], reverse=True)[:n]


def _load_scaled(fov_id, px):
    import cv2

    from src.gcs_fov_multi import load_fov_image

    image, uri = load_fov_image(SAMPLE_ID, fov_id, COUNTRY)
    h, w = image.shape[:2]
    scale = px / max(h, w)
    return cv2.resize(image, (int(w * scale), int(h * scale))), uri


def render_previews(rows_by_fov, median):
    """Render a raw fluorescence thumbnail per PREVIEW_FOVS, captioned with this analysis's own
    numbers so each image is self-describing when embedded in the write-up.
    """
    import cv2

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for fov_id in PREVIEW_FOVS:
        row = rows_by_fov[fov_id]
        preview, uri = _load_scaled(fov_id, PREVIEW_PX)

        truth = LABELED_FOVS.get(fov_id, "(unlabeled)")
        color = (0, 0, 255) if row["high_outlier"] else (0, 255, 0)

        def put(y, text):
            cv2.putText(preview, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        put(20, f"{SAMPLE_ID}  fov {fov_id}  spot_truth={truth}")
        put(42, f"n_spots_detected={row['n_spots_detected']}  (slide median {median:g})")
        put(64, f"ratio={row['ratio_to_median']}  robust_zscore={row['robust_zscore']}")

        out_path = PREVIEW_DIR / f"{SAMPLE_ID}__fov{fov_id}__preview.png"
        cv2.imwrite(str(out_path), preview)
        print(f"  wrote {out_path.name}  (from {uri})")


def render_thumbs(rows):
    """Render one small thumbnail per top-N FOV, for embedding as an image column in the
    write-up's top-N table. The surrounding table row already carries n_spots/ratio/z, so the
    caption here is just the identifying line -- three stacked lines of text would eat most of a
    260px frame and obscure the very halo the thumb exists to show.
    """
    import cv2

    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    for row in top_n_rows(rows):
        fov_id = row["fov_id"]
        thumb, uri = _load_scaled(fov_id, THUMB_PX)
        color = (0, 0, 255) if row["high_outlier"] else (0, 255, 0)
        cv2.putText(thumb, f"fov {fov_id}  z={row['robust_zscore']:g}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        out_path = THUMB_DIR / f"{SAMPLE_ID}__fov{fov_id}__thumb.png"
        cv2.imwrite(str(out_path), thumb)
        print(f"  wrote thumbs/{out_path.name}  ({out_path.stat().st_size // 1024}KB, from {uri})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previews", action="store_true",
                        help=f"also render raw fluorescence thumbnails for FOVs {PREVIEW_FOVS}")
    parser.add_argument("--thumbs", action="store_true",
                        help=f"also render {THUMB_PX}px thumbnails for the top {TOP_N} FOVs by robust_zscore")
    args = parser.parse_args()

    counts = load_slide_metric_counts(SAMPLE_ID, COUNTRY, metric=METRIC)
    if counts is None:
        raise SystemExit(f"no detection data found for {SAMPLE_ID} -- fallback did not resolve")

    stats = slide_baseline(counts)
    mean, std, median, mad, n_fovs = (
        stats["mean"], stats["std"], stats["median"], stats["mad"], stats["n_fovs"],
    )
    print(f"{SAMPLE_ID}: {n_fovs} FOVs, baseline_median={median}, baseline_mad={mad:.3f}, "
          f"baseline_mean={mean:.3f}, baseline_std={std:.3f}")

    rows = []
    for fov_id in sorted(counts):
        target = counts[fov_id]
        ratio_to_median = target / median if median else None
        robust_zscore = (target - median) / mad if mad else None
        role = []
        if fov_id in LABELED_FOVS:
            role.append("labeled")
        if fov_id in BOUNDARY_FOVS:
            role.append("boundary")
        rows.append({
            "fov_id": fov_id,
            "n_spots_detected": target,
            "ratio_to_median": round(ratio_to_median, 3) if ratio_to_median is not None else "",
            "robust_zscore": round(robust_zscore, 3) if robust_zscore is not None else "",
            "role": ";".join(role),
            "high_outlier": robust_zscore is not None and robust_zscore >= OUTLIER_ZSCORE,
        })

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fov_id", "n_spots_detected", "ratio_to_median", "robust_zscore", "role", "high_outlier"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {OUT_CSV}")

    n_outliers = sum(1 for r in rows if r["high_outlier"])
    print(f"\n{n_outliers}/{len(rows)} FOVs at or above robust_zscore >= {OUTLIER_ZSCORE}")

    print("\nlabeled + boundary FOVs:")
    for r in rows:
        if r["role"]:
            print(f"  fov {r['fov_id']:>3} ({r['role']:>16}): n_spots={r['n_spots_detected']:>5}  "
                  f"ratio={r['ratio_to_median']:>7}  z={r['robust_zscore']:>8}")

    print(f"\ntop {TOP_N} FOVs by robust_zscore:")
    for r in top_n_rows(rows):
        print(f"  fov {r['fov_id']:>3} ({r['role'] or '-':>16}): n_spots={r['n_spots_detected']:>5}  "
              f"ratio={r['ratio_to_median']:>7}  z={r['robust_zscore']:>8}")

    if args.previews:
        print(f"\nrendering previews for FOVs {PREVIEW_FOVS}:")
        render_previews({r["fov_id"]: r for r in rows}, median)

    if args.thumbs:
        print(f"\nrendering {THUMB_PX}px thumbs for the top {TOP_N} FOVs:")
        render_thumbs(rows)


if __name__ == "__main__":
    main()
