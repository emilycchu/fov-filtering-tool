"""Run the calibrated v2.2 density/Rouleaux scorer over the Nigeria 081226 mini-dataset
(8 FOVs, 2 slides: HP231668 and HP245487 R) and report where it lands -- plus an
out-of-distribution check against the 661-FOV calibration pool.

Nothing is recalibrated here; this is inference + a distribution check. Manual labels now
exist (`data/labels/nigeria-081226/`), so accuracy is reported against them and the quality
grid plots manual and predicted buckets as two point groups.

The scores go through `apply_label_overrides`, so with v2.2 params the empty-field gate is
live: the two near-empty FOVs trip it and come out Sparser / No Rouleaux instead of the
Dense / Very Dense the raw composite gives them. Pass `--params` a JSON without an enabled
`empty_field_override` to see the ungated behavior.

The OOD check is the point of this script, not an aside. `scripts/combined/README.md`'s
"Known limitation" says every v2 feature is a raw pixel/intensity statistic sensitive to
stain/scanner/illumination, and that a new stain should be spot-checked before its scores
are trusted. That warning was previously only prose; here it is computed: each FOV's
features are compared against the per-feature 2nd-98th percentile normalization range in
density_overlap_v2.2_params.json (the range `normalize_matrix` clips to), so a feature
sitting outside it is one that the composite cannot resolve -- it is pinned at 0.0 or 1.0
by construction.

Outputs (data/results/nigeria-081226/, <suffix> from --suffix, default "v2.2"):
    features-<suffix>.csv        raw feature vector + composite scores + buckets, per FOV
    ood-report-<suffix>.csv      per-FOV x per-feature normalization-range position
    quality-grid-<suffix>.png    manual vs. predicted on the density x Rouleaux grid
    feature-ood-<suffix>.png     each feature vs. the calibration pool's distribution
    fov-thumbnails-<suffix>.png  the 8 FOVs downsampled, captioned with predicted buckets

Usage:
    python scripts/nigeria_081226.py [--workers N] [--params PATH] [--suffix NAME]
"""
import argparse
import csv
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_MINE,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    DENSITY_LEVELS,
    JITTER_SEED,
    NIGERIA_IMAGE_DIR,
    NIGERIA_LABELS_CSV,
    OVERLAP_LEVELS,
    apply_label_overrides,
    compute_features,
    density_ordinal,
    display_level,
    empty_field_fired,
    list_image_paths,
    load_image,
    overlap_ordinal,
    read_csv_dicts,
)
from src.composite_v2 import bucket, weighted_composite  # noqa: E402

COLOR_MODEL = "#eb6834"  # matches scripts/combined/plot_bucket_comparison_v2.py
COLOR_ALERT = "#c2402a"  # out-of-range emphasis

IMAGE_DIR = NIGERIA_IMAGE_DIR
V2_PARAMS = ROOT / "data" / "results" / "density-rouleaux-v2" / "density_overlap_v2.2_params.json"
CALIB_FEATURES = ROOT / "data" / "results" / "density-rouleaux-v2" / "features-v2.2.csv"
OUT_DIR = ROOT / "data" / "results" / "nigeria-081226"

# every candidate feature compute_features() emits, minus the diagnostic-only otsu_threshold
FEATURE_ORDER = [
    "coverage",
    "otsu_separability",
    "saturation_score",
    "lbp_entropy",
    "glcm_contrast",
    "edge_density_unmasked",
    "tile_glcm_cv",
    "tile_glcm_patchiness",
]


def slide_of(filename):
    """'HP245487 R_20251205_041904_2.bmp' -> 'HP245487 R'. The scan timestamp is shared by
    every FOV of a slide, so the slide id is everything before the first underscore."""
    return filename.split("_")[0]


def load_manual_labels(path=NIGERIA_LABELS_CSV):
    """{filename: (density_label, overlap_label)} from the hand-written annotation CSV.

    The `_sparse` filename suffix is deliberately not parsed -- see the note in
    data/labels/nigeria-081226/README.md.
    """
    return {r["filename"]: (r["density"].strip().lower(), r["overlap"].strip().lower())
            for r in read_csv_dicts(path)}


# --- scoring -----------------------------------------------------------------------------

def _extract_one(path):
    features = compute_features(load_image(path))
    features["filename"] = Path(str(path)).name
    return features


def score_axis(features, axis_params):
    ranges = {n: (v["min"], v["max"]) for n, v in axis_params["normalization"].items()}
    score = weighted_composite(features, axis_params["weights"], ranges)
    return score, bucket(score, axis_params["bucket_thresholds"], axis_params["bucket_labels"])


def score_all(paths, params, workers, manual=None):
    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(_extract_one, paths)
    else:
        rows = [_extract_one(p) for p in paths]

    manual = manual or {}
    for r in rows:
        r["slide"] = slide_of(r["filename"])
        r["density_score"], raw_density = score_axis(r, params["density"])
        r["overlap_score"], raw_overlap = score_axis(r, params["overlap"])
        # keep the pre-override buckets: the composite score printed next to a gated FOV
        # would otherwise contradict its label with no way to see why
        r["raw_density_label"], r["raw_overlap_label"] = raw_density, raw_overlap
        r["density_label"], r["overlap_label"] = apply_label_overrides(
            raw_density, raw_overlap, r, params)
        r["gated"] = empty_field_fired(r, params.get("empty_field_override"))
        r["manual_density_label"], r["manual_overlap_label"] = manual.get(r["filename"], ("", ""))
    return rows


# --- out-of-distribution check ------------------------------------------------------------

def normalization_ranges(params):
    """Union of both axes' per-feature normalization ranges (identical where they overlap --
    both axes normalize against the same calibration percentiles)."""
    ranges = {}
    for axis in ("density", "overlap"):
        for name, v in params[axis]["normalization"].items():
            ranges[name] = (v["min"], v["max"])
    return ranges


def range_position(value, lo, hi):
    """Where `value` sits relative to the [lo, hi] normalization range, as the fraction the
    composite would have used before clipping. <0 or >1 means the feature is clipped, i.e.
    the composite cannot distinguish this FOV from any other beyond the range edge."""
    if hi <= lo:
        return float("nan")
    return (value - lo) / (hi - lo)


def build_ood_rows(rows, ranges):
    out = []
    for r in rows:
        for name in FEATURE_ORDER:
            if name not in ranges:
                continue
            lo, hi = ranges[name]
            pos = range_position(r[name], lo, hi)
            out.append({
                "filename": r["filename"],
                "slide": r["slide"],
                "feature": name,
                "value": round(r[name], 6),
                "calib_p2": round(lo, 6),
                "calib_p98": round(hi, 6),
                "range_position": round(pos, 4),
                "clipped": "yes" if (pos < 0 or pos > 1) else "no",
                "x_over_p98": round(r[name] / hi, 3) if hi > 0 else "",
            })
    return out


# --- plots --------------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(COLOR_SURFACE)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.set_axisbelow(True)


def plot_quality_grid(rows, out_path, title_note=""):
    """The density x Rouleaux quality grid, manual annotation vs. model prediction as two
    jittered point groups -- the form scripts/combined/plot_bucket_comparison_v2.py uses,
    now that this dataset has manual labels to put a second group against.

    Where the two disagree, a connector is drawn between them: at n=8 the eye can follow
    individual FOVs, which is the whole value of plotting such a small set.
    """
    rng = random.Random(JITTER_SEED)
    fig, ax = plt.subplots(figsize=(10, 6.8), dpi=150)
    fig.patch.set_facecolor(COLOR_SURFACE)
    _style(ax)

    have_manual = [r for r in rows if r["manual_density_label"]]
    # jitter per FOV, shared by both of its points, so a connector shows only real disagreement
    offsets = {r["filename"]: (rng.uniform(-0.13, 0.13), rng.uniform(-0.15, 0.15)) for r in rows}

    disagreements = [r for r in have_manual
                     if (r["manual_density_label"], r["manual_overlap_label"])
                     != (r["density_label"], r["overlap_label"])]
    for r in disagreements:
        dx, dy = offsets[r["filename"]]
        ax.plot([overlap_ordinal(r["manual_overlap_label"]) - 0.15 + dx,
                 overlap_ordinal(r["overlap_label"]) + 0.15 + dx],
                [density_ordinal(r["manual_density_label"]) + dy,
                 density_ordinal(r["density_label"]) + dy],
                color=COLOR_AXIS, linewidth=1.0, zorder=2, alpha=0.8)

    def scatter_group(xs, ys, color, marker, label):
        ax.scatter(xs, ys, s=110, marker=marker, color=color, alpha=0.8, linewidths=0,
                   zorder=3, label=label)

    scatter_group([overlap_ordinal(r["manual_overlap_label"]) - 0.15 + offsets[r["filename"]][0] for r in have_manual],
                  [density_ordinal(r["manual_density_label"]) + offsets[r["filename"]][1] for r in have_manual],
                  COLOR_MINE, "o", "Mine (manual annotation)")
    scatter_group([overlap_ordinal(r["overlap_label"]) + 0.15 + offsets[r["filename"]][0] for r in rows],
                  [density_ordinal(r["density_label"]) + offsets[r["filename"]][1] for r in rows],
                  COLOR_MODEL, "^", "score_fov_v2 (model)")

    ax.set_xticks(range(len(OVERLAP_LEVELS)))
    ax.set_xticklabels([display_level(l) for l in OVERLAP_LEVELS], fontsize=10, rotation=15, ha="right")
    ax.set_xlim(-0.5, len(OVERLAP_LEVELS) - 0.5)
    ax.set_yticks(range(len(DENSITY_LEVELS)))
    ax.set_yticklabels([display_level(l) for l in DENSITY_LEVELS], fontsize=10)
    ax.set_ylim(-0.5, len(DENSITY_LEVELS) - 0.5)
    for i in range(1, len(OVERLAP_LEVELS)):
        ax.axvline(i - 0.5, color=COLOR_GRID, linewidth=1, zorder=0)
    for i in range(1, len(DENSITY_LEVELS)):
        ax.axhline(i - 0.5, color=COLOR_GRID, linewidth=1, zorder=0)

    # the top-right cell holds 6 of 8 FOVs on both axes; shade it so the plot can't be
    # misread as spread across the grid
    ax.add_patch(plt.Rectangle((len(OVERLAP_LEVELS) - 1.5, len(DENSITY_LEVELS) - 1.5), 1, 1,
                               facecolor=COLOR_ALERT, alpha=0.07, zorder=1))

    ax.set_xlabel("Rouleaux level", color=COLOR_SECONDARY, fontsize=11)
    ax.set_ylabel("Density level", color=COLOR_SECONDARY, fontsize=11)

    # figure-level legend below the axes: inside the axes it sat in the Sparser x No-Rouleaux
    # cell and read as two more plotted FOVs
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_MINE, markersize=9,
               label="Mine (manual annotation)"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor=COLOR_MODEL, markersize=9,
               label="score_fov_v2 (model)"),
    ]
    if disagreements:
        handles.append(Line2D([0], [0], color=COLOR_AXIS, linewidth=1.0, label="disagreement"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=len(handles),
               frameon=False, fontsize=9.5, labelcolor=COLOR_SECONDARY)

    n_agree = sum(1 for r in have_manual
                  if r["manual_density_label"] == r["density_label"]
                  and r["manual_overlap_label"] == r["overlap_label"])
    fig.suptitle("Nigeria 081226 -- manual annotation vs. model, density x Rouleaux",
                 x=0.02, y=0.97, ha="left", color=COLOR_PRIMARY, fontsize=14, fontweight="bold")
    ax.set_title(f"n = {len(rows)} FOVs, {n_agree} matching on both axes"
                 f"{title_note}; shaded cell = top bucket on both axes",
                 loc="left", color=COLOR_SECONDARY, fontsize=10, pad=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out_path, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_feature_ood(rows, calib_rows, ranges, out_path):
    """Per feature: the calibration pool's distribution, its 2nd-98th percentile
    normalization band, and where the 8 Nigeria FOVs fall."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.5), dpi=150)
    fig.patch.set_facecolor(COLOR_SURFACE)
    rng = random.Random(JITTER_SEED)

    for ax, name in zip(axes.ravel(), FEATURE_ORDER):
        _style(ax)
        calib_vals = np.array([float(r[name]) for r in calib_rows])
        nig_vals = np.array([r[name] for r in rows])
        lo, hi = ranges[name]

        ax.axhspan(lo, hi, color=COLOR_MINE, alpha=0.10, zorder=0)
        ax.scatter([0 + rng.uniform(-0.11, 0.11) for _ in calib_vals], calib_vals,
                   s=7, color=COLOR_MINE, alpha=0.28, linewidths=0, zorder=2)
        out_of_range = (nig_vals < lo) | (nig_vals > hi)
        for flag, color in ((~out_of_range, COLOR_MODEL), (out_of_range, COLOR_ALERT)):
            if not flag.any():
                continue
            ax.scatter([1 + rng.uniform(-0.11, 0.11) for _ in nig_vals[flag]], nig_vals[flag],
                       s=52, color=color, alpha=0.85, linewidths=0, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"calibration\n(n={len(calib_vals)})", f"Nigeria\n(n={len(nig_vals)})"], fontsize=9)
        ax.set_xlim(-0.45, 1.45)
        ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)

        n_above = int((nig_vals > hi).sum())
        n_below = int((nig_vals < lo).sum())
        n_clipped = n_above + n_below
        if n_clipped:
            parts = ([f"{n_above} above p98"] if n_above else []) + ([f"{n_below} below p2"] if n_below else [])
            sub = f"{n_clipped}/{len(nig_vals)} clipped: " + ", ".join(parts)
        else:
            sub = "all inside band"
        ax.set_title(name, loc="left", color=COLOR_PRIMARY, fontsize=10.5, fontweight="bold", pad=19)
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=8.5,
                color=COLOR_ALERT if n_clipped else COLOR_SECONDARY)

    fig.suptitle("Nigeria 081226 feature values vs. the v2.2 calibration pool",
                 x=0.01, y=0.985, ha="left", color=COLOR_PRIMARY, fontsize=15, fontweight="bold")
    fig.text(0.01, 0.945, "shaded band = the 2nd-98th percentile range v2.2 normalizes against; "
                          "a point outside it is clipped to 0.0 or 1.0 in the composite",
             ha="left", color=COLOR_SECONDARY, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out_path, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)


def plot_thumbnails(rows, out_path, thumb_px=460):
    """The 8 FOVs downsampled side by side with their predicted buckets -- the visual
    spot-check scripts/combined/README.md asks for before trusting a new stain."""
    n = len(rows)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4.75 * nrows), dpi=130)
    fig.patch.set_facecolor(COLOR_SURFACE)

    for ax, r in zip(axes.ravel(), rows):
        image = cv2.imread(str(IMAGE_DIR / r["filename"]))
        thumb = cv2.cvtColor(cv2.resize(image, (thumb_px, thumb_px), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2RGB)
        ax.imshow(thumb)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(COLOR_AXIS)
        stem = r["filename"].replace(".bmp", "")
        ax.set_title(stem, loc="left", color=COLOR_PRIMARY, fontsize=8.5, fontweight="bold")
        # on a gated FOV the composite score does not correspond to the label -- the score is
        # exactly what the gate overrode, so show it as struck-through provenance, not as the
        # basis for the bucket
        if r["gated"]:
            caption = (f"{display_level(r['density_label'])} / {display_level(r['overlap_label'])}"
                       f"  -- empty-field gate\n"
                       f"composite said {display_level(r['raw_density_label'])} ({r['density_score']:.2f})"
                       f" / {display_level(r['raw_overlap_label'])} ({r['overlap_score']:.2f})")
            color = COLOR_ALERT
        else:
            caption = (f"{display_level(r['density_label'])}  ({r['density_score']:.2f})\n"
                       f"{display_level(r['overlap_label'])}  ({r['overlap_score']:.2f})")
            color = COLOR_SECONDARY
        ax.set_xlabel(caption, color=color, fontsize=9)

    for ax in axes.ravel()[n:]:
        ax.set_visible(False)

    fig.suptitle("Nigeria 081226 FOVs with v2.2 predicted density / Rouleaux (composite score)",
                 x=0.01, y=0.995, ha="left", color=COLOR_PRIMARY, fontsize=14, fontweight="bold")
    # h_pad keeps each row's two-line xlabel clear of the next row's title
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=4.0)
    fig.savefig(out_path, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)


# --- main ---------------------------------------------------------------------------------

def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Score the Nigeria 081226 mini-dataset with v2.2.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--params", default=str(V2_PARAMS),
                        help="params JSON to score with (default: v2.2)")
    parser.add_argument("--suffix", default="v2.2",
                        help="suffix for every output filename, e.g. v2.2-gated")
    args = parser.parse_args()

    params_path = Path(args.params)
    with open(params_path) as f:
        params = json.load(f)

    paths = list_image_paths(IMAGE_DIR)
    manual = load_manual_labels()
    rows = score_all(paths, params, args.workers, manual=manual)
    ranges = normalization_ranges(params)
    calib_rows = read_csv_dicts(CALIB_FEATURES)
    gated = [r for r in rows if r["gated"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / f"features-{args.suffix}.csv",
              ["filename", "slide"] + FEATURE_ORDER + ["otsu_threshold", "density_score",
                                                       "density_label", "overlap_score", "overlap_label",
                                                       "raw_density_label", "raw_overlap_label", "gated",
                                                       "manual_density_label", "manual_overlap_label"],
              rows)
    ood_rows = build_ood_rows(rows, ranges)
    write_csv(OUT_DIR / f"ood-report-{args.suffix}.csv",
              ["filename", "slide", "feature", "value", "calib_p2", "calib_p98",
               "range_position", "clipped", "x_over_p98"],
              ood_rows)

    note = f", {len(gated)} via the empty-field gate" if gated else ""
    plot_quality_grid(rows, OUT_DIR / f"quality-grid-{args.suffix}.png", title_note=note)
    plot_feature_ood(rows, calib_rows, ranges, OUT_DIR / f"feature-ood-{args.suffix}.png")
    plot_thumbnails(rows, OUT_DIR / f"fov-thumbnails-{args.suffix}.png")

    # --- console summary ---
    print(f"n={len(rows)} FOVs from {IMAGE_DIR.name}, scored with {params_path.name}\n")
    for r in rows:
        mark = "  [gated]" if r["gated"] else ""
        print(f"  {r['filename']:<40} {display_level(r['density_label']):<14} ({r['density_score']:.3f})   "
              f"{display_level(r['overlap_label']):<16} ({r['overlap_score']:.3f}){mark}")

    # --- accuracy against the manual labels ---
    labeled = [r for r in rows if r["manual_density_label"]]
    if labeled:
        print(f"\nAgainst manual labels ({NIGERIA_LABELS_CSV.name}, n={len(labeled)}):")
        for axis, pred_key, raw_key, manual_key in (
            ("density", "density_label", "raw_density_label", "manual_density_label"),
            ("Rouleaux", "overlap_label", "raw_overlap_label", "manual_overlap_label"),
        ):
            exact = sum(1 for r in labeled if r[pred_key] == r[manual_key])
            raw_exact = sum(1 for r in labeled if r[raw_key] == r[manual_key])
            delta = "" if exact == raw_exact else f"   (ungated composite: {raw_exact}/{len(labeled)})"
            print(f"  {axis:<9} {exact}/{len(labeled)} exact{delta}")
        both = sum(1 for r in labeled
                   if r["density_label"] == r["manual_density_label"]
                   and r["overlap_label"] == r["manual_overlap_label"])
        print(f"  {'both axes':<9} {both}/{len(labeled)}")
        wrong = [r for r in labeled if r["density_label"] != r["manual_density_label"]
                 or r["overlap_label"] != r["manual_overlap_label"]]
        for r in wrong:
            print(f"    {r['filename']:<40} manual=({r['manual_density_label']}, {r['manual_overlap_label']})"
                  f"  model=({r['density_label']}, {r['overlap_label']})")

    if gated:
        print(f"\nEmpty-field gate fired on {len(gated)}/{len(rows)} FOVs "
              "(all 4 texture features below their calibration p2 floor):")
        for r in gated:
            print(f"  {r['filename']:<40} composite said "
                  f"({r['raw_density_label']}, {r['raw_overlap_label']}) -> forced "
                  f"({r['density_label']}, {r['overlap_label']})")

    print("\nPredicted bucket counts:")
    for axis, levels in (("density_label", DENSITY_LEVELS), ("overlap_label", OVERLAP_LEVELS)):
        counts = {l: sum(1 for r in rows if r[axis] == l) for l in levels}
        print(f"  {axis:<14} " + ", ".join(f"{display_level(l)}={c}" for l, c in counts.items() if c))

    print("\nOut-of-range features (value outside the 2nd-98th percentile calibration band):")
    for name in FEATURE_ORDER:
        lo, hi = ranges[name]
        vals = np.array([r[name] for r in rows])
        n_over, n_under = int((vals > hi).sum()), int((vals < lo).sum())
        band = f"[{lo:.4f}, {hi:.4f}]"
        if not (n_over or n_under):
            print(f"  {name:<24} 0/{len(vals)} clipped, band {band}")
            continue
        # both directions can happen at once here: the near-empty FOVs fall below p2 on the
        # texture features while the textured ones run above p98 -- reporting only one
        # direction would hide half the problem
        detail = []
        if n_over:
            detail.append(f"{n_over} above p98 (max {vals.max():.4f} = {vals.max() / hi:.2f}x edge)")
        if n_under:
            detail.append(f"{n_under} below p2 (min {vals.min():.4f} = {vals.min() / lo:.2f}x edge)")
        print(f"  {name:<24} {n_over + n_under}/{len(vals)} clipped, band {band}")
        for d in detail:
            print(f"  {'':<24}   - {d}")

    n_fully_clipped = sum(1 for r in rows
                          if all(not (lo <= r[n] <= hi) for n, (lo, hi) in
                                 ((n, ranges[n]) for n in ("coverage", "glcm_contrast", "edge_density_unmasked"))))
    print(f"\n{n_fully_clipped}/{len(rows)} FOVs are clipped on all three of "
          "coverage / glcm_contrast / edge_density_unmasked simultaneously.")
    print(f"\nwrote CSVs + 3 plots to {OUT_DIR} with suffix '{args.suffix}'")


if __name__ == "__main__":
    main()
