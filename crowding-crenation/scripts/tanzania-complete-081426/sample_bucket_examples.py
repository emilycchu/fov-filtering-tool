"""Contact sheets of randomly sampled FOVs from the slides in each slide-level bucket.

The slide-level result says 61% of the cohort is `sparser` and nothing is `very dense`. That is a
claim about images, so it should be checkable by looking at images.

**Sampling is deliberately dumb.** FOVs are drawn uniformly at random from the slides whose
*slide-level* bucket is the row's bucket -- not chosen for being clear examples, and not filtered
to FOVs whose own per-FOV label matches the row. Each thumbnail is annotated with its own per-FOV
label and score, so where a randomly drawn FOV disagrees with its slide's bucket, that shows up in
the sheet instead of being quietly excluded. Cherry-picked examples would make the calibration look
better or worse than it is; these cannot.

The seed and the exact (slide, fov) list are written to a manifest CSV, so a reader can pull any
thumbnail back to its source blob and re-render the identical sheet.

Usage:
    python scripts/tanzania-complete-081426/sample_bucket_examples.py
    python scripts/tanzania-complete-081426/sample_bucket_examples.py --axis overlap --per-bucket 6
"""
import argparse
import csv
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _slide_common import (  # noqa: E402
    CROWDING_FOV_DIR,
    PLOTS_DIR,
    RESULTS_DIR,
    ROOT,
    SLIDE_SUMMARY_CSV,
    dpc_gcs_path,
    load_slide_index,
    read_csv_dicts,
    with_retry,
)

sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    DENSITY_LEVELS,
    OVERLAP_LEVELS,
    display_level,
    load_image,
)

SEED = 20260817
THUMB_PX = 460
DPI = 170

# A whole 2800px FOV shrunk to a 460px thumbnail turns individual cells into sub-pixel noise, so
# the sheet showed four rows of identical grey. Instead: a fixed-size crop from the centre at a
# fixed scale. Because the crop size and display size are the same for every thumbnail, cell
# *density* stays directly comparable across rows -- which is the whole point of the sheet.
CROP_PX = 800

# One global display window for every thumbnail, not per-image normalization. These images sit in
# roughly [113, 213] (measured; std 11.2 on a sparse FOV vs 19.0 on a dense one), so the default
# 0-255 window wastes most of the range and renders everything mid-grey. Stretching each image
# independently would make them legible but would also erase the between-bucket brightness and
# contrast differences that the density composite is built on -- exactly the signal being shown.
VMIN, VMAX = 110, 215


def sample_fovs(axis, per_bucket, seed, summary_csv, index):
    """[(bucket, slide_id, fov_id)] -- uniform over (slide, fov) within each bucket."""
    bucket_col = f"{axis}_bucket"
    levels = DENSITY_LEVELS if axis == "density" else OVERLAP_LEVELS

    slides_by_bucket = defaultdict(list)
    for row in read_csv_dicts(summary_csv):
        if row.get(bucket_col):
            slides_by_bucket[row[bucket_col]].append(row["slide_id"])

    rng = random.Random(seed)
    picks = []
    for bucket in levels:
        slides = sorted(slides_by_bucket.get(bucket, []))
        if not slides:
            continue
        # Sample slides without replacement where possible, so one slide cannot dominate a row.
        chosen = (rng.sample(slides, per_bucket) if len(slides) >= per_bucket
                  else [rng.choice(slides) for _ in range(per_bucket)])
        for slide_id in chosen:
            fov_ids = sorted(int(i) for i in index[slide_id]["dpc_fov_ids"])
            picks.append((bucket, slide_id, rng.choice(fov_ids)))
    return picks


def per_fov_row(slide_id, fov_id):
    path = CROWDING_FOV_DIR / f"{slide_id}.csv"
    if not path.exists():
        return {}
    for row in read_csv_dicts(path):
        if int(row["fov_id"]) == fov_id:
            return row
    return {}


def fetch_thumb(item, index, crop_px=CROP_PX):
    bucket, slide_id, fov_id = item
    box = index[slide_id]["box"]
    image = with_retry(load_image, dpc_gcs_path(box, slide_id, fov_id), grayscale=True)

    if crop_px:
        h, w = image.shape[:2]
        side = min(crop_px, h, w)
        top, left = (h - side) // 2, (w - side) // 2
        image = image[top:top + side, left:left + side]

    scale = THUMB_PX / max(image.shape[:2])
    thumb = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return bucket, slide_id, fov_id, thumb, per_fov_row(slide_id, fov_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--axis", choices=("density", "overlap"), default="density")
    parser.add_argument("--per-bucket", type=int, default=5)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--summary-csv", default=str(SLIDE_SUMMARY_CSV))
    parser.add_argument("--plots-dir", default=str(PLOTS_DIR))
    parser.add_argument("--crop-px", type=int, default=CROP_PX,
                        help="native-pixel centre crop per thumbnail; 0 = whole FOV (unreadable "
                             "at thumbnail size, see module constants)")
    parser.add_argument("--vmin", type=int, default=VMIN)
    parser.add_argument("--vmax", type=int, default=VMAX)
    args = parser.parse_args()

    index = load_slide_index()["slides"]
    picks = sample_fovs(args.axis, args.per_bucket, args.seed, args.summary_csv, index)
    if not picks:
        raise SystemExit("no slides with a bucket in the summary -- run aggregate_slides.py first")

    label_col = "density_label" if args.axis == "density" else "overlap_label"
    score_col = "density_score" if args.axis == "density" else "overlap_score"

    print(f"sampling {len(picks)} FOVs (seed {args.seed}) ...", flush=True)
    with ThreadPoolExecutor(args.threads) as pool:
        fetched = list(pool.map(lambda item: fetch_thumb(item, index, args.crop_px), picks))

    buckets = [b for b in (DENSITY_LEVELS if args.axis == "density" else OVERLAP_LEVELS)
               if any(f[0] == b for f in fetched)]
    n_rows, n_cols = len(buckets), args.per_bucket

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.05 * n_cols, 2.35 * n_rows),
                            facecolor=COLOR_SURFACE, squeeze=False)
    axis_name = "Density" if args.axis == "density" else "Rouleaux"

    for r, bucket in enumerate(buckets):
        items = [f for f in fetched if f[0] == bucket]
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_facecolor(COLOR_SURFACE)
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if c >= len(items):
                ax.axis("off")
                continue
            _b, slide_id, fov_id, thumb, row = items[c]
            ax.imshow(thumb, cmap="gray", vmin=args.vmin, vmax=args.vmax,
                      interpolation="nearest")

            own = row.get(label_col, "")
            score = row.get(score_col, "")
            score_txt = f"{float(score):.3f}" if score else "?"
            # Flag disagreement between the FOV's own label and its slide's bucket, rather than
            # hiding it -- these are the informative thumbnails.
            mark = "" if own == bucket else "  (FOV: %s)" % display_level(own) if own else ""
            ax.set_title(f"{slide_id}\nfov {fov_id} - {score_txt}{mark}",
                         fontsize=6.2, color=COLOR_SECONDARY, pad=3)
            if c == 0:
                ax.set_ylabel(display_level(bucket), fontsize=8.5, color=COLOR_PRIMARY,
                              fontweight="bold", rotation=90, labelpad=8)
                ax.axis("on")
                ax.set_xticks([]); ax.set_yticks([])

    # tight_layout first, then place the titles in the space it reserved -- suptitle and a
    # fig.text at a hardcoded y collide once the axes grid is resized, which is what happened at
    # y=0.972 on the first render.
    fig.tight_layout(rect=(0.01, 0, 1, 0.93))
    fig.suptitle(f"Randomly sampled FOVs by slide-level {axis_name.lower()} bucket",
                 x=0.01, y=0.995, ha="left", va="top", color=COLOR_PRIMARY, fontsize=11.5,
                 fontweight="bold")
    crop_note = (f"centre {args.crop_px}px crop of each 2800px FOV, identical scale and grey "
                 f"window ({args.vmin}-{args.vmax}) throughout" if args.crop_px
                 else "whole FOV")
    fig.text(0.01, 0.963,
             f"{args.per_bucket} FOVs per bucket, uniform over (slide, FOV) at seed {args.seed}; "
             f"row = the slide's bucket, caption = that FOV's own score",
             ha="left", va="top", color=COLOR_SECONDARY, fontsize=7.8)
    fig.text(0.01, 0.945, crop_note, ha="left", va="top", color=COLOR_MUTED, fontsize=7.2)

    out_png = Path(args.plots_dir) / f"bucket-examples-{args.axis}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=DPI, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)

    manifest = RESULTS_DIR / f"bucket-examples-{args.axis}.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slide_bucket", "slide_id", "fov_id", "blob",
                         f"fov_{label_col}", f"fov_{score_col}"])
        for bucket, slide_id, fov_id, _thumb, row in fetched:
            writer.writerow([bucket, slide_id, fov_id,
                             str(dpc_gcs_path(index[slide_id]["box"], slide_id, fov_id)),
                             row.get(label_col, ""), row.get(score_col, "")])

    disagree = sum(1 for b, _s, _f, _t, row in fetched if row.get(label_col) != b)
    print(f"wrote {out_png}")
    print(f"wrote {manifest}")
    print(f"FOVs whose own label differs from their slide's bucket: {disagree}/{len(fetched)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
