"""Why did the two visually near-empty Nigeria FOVs score Dense / Very Dense?

`data/results/nigeria-081226/fov-thumbnails-v2.2.png` shows FOV 3 and FOV 4 of slide
HP231668 as essentially blank fields with a few specks, yet v2.2 scored them 0.676 (Dense)
and 0.736 (Very Dense) on density and Heavy Rouleaux on the Rouleaux axis. This script
decomposes those two scores term by term to show the mechanism, and renders the Otsu mask
that starts it.

The mechanism, in short: `weighted_composite` is a sum of non-negative weighted terms and
`normalize()` clamps each normalized feature to [0, 1]. A feature far *below* its
calibration floor therefore contributes exactly 0.0 -- indistinguishable from a feature
sitting exactly at the floor. The composite has no way to express negative evidence. On
these FOVs, the four features that "know" the field is empty (glcm_contrast, edge_density,
lbp_entropy, otsu_separability) are all clipped to 0 and so say nothing at all, while the
three features that misfire on an empty field (coverage, saturation_score,
tile_glcm_patchiness) pin at 1.0 and carry 65% of the density weight between them.

Those same four clipped features are what the empty-field gate keys on: all four below floor
means the composite is describing a field with no cells in it. This script reports the
composite's own verdict *and* the gated one, so the decomposition still explains the label
that actually ships. With v2.2 params the gate is enabled and both FOVs come out Sparser /
No Rouleaux.

Outputs (data/results/nigeria-081226/, <suffix> from --suffix):
    sparse-score-decomposition-<suffix>.csv  per-feature raw/normalized/contribution, per FOV
    sparse-decomposition-<suffix>.png        stacked contribution bars + the Otsu mask evidence

Usage:
    python scripts/nigeria_081226_sparse_diagnosis.py [--params PATH] [--suffix NAME]
"""
import argparse
import csv
import statistics
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "combined"))

import json  # noqa: E402

from _v2_common import (  # noqa: E402
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_MINE,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    apply_label_overrides,
    compute_features,
    display_level,
    empty_field_features_below,
    empty_field_fired,
    load_image,
    read_csv_dicts,
)
from nigeria_081226 import CALIB_FEATURES, FEATURE_ORDER, IMAGE_DIR, OUT_DIR, V2_PARAMS  # noqa: E402
from src.composite_v2 import bucket, normalize  # noqa: E402
from src.segmentation import otsu_segment  # noqa: E402

COLOR_ALERT = "#c2402a"     # features that misfire on an empty field
COLOR_SILENCED = "#8c8a84"  # features clipped to 0 -- present but contributing nothing

# the two FOVs in question, plus a textured one from the same slide for contrast
SPARSE_FOVS = [
    "HP231668_20251210_031002_3_sparse.bmp",
    "HP231668_20251210_031002_4_sparse.bmp",
]
CONTRAST_FOV = "HP231668_20251210_031002_1.bmp"

# features that read an empty field as "dense" vs. features that correctly read it as empty
MISFIRING = {"coverage", "saturation_score", "tile_glcm_patchiness", "tile_glcm_cv"}


def decompose(features, axis_params):
    """Reproduce weighted_composite() term by term, so each feature's actual contribution to
    the final score is visible (weights already sum to 1, so total_weight == 1 here)."""
    weights = axis_params["weights"]
    ranges = {n: (v["min"], v["max"]) for n, v in axis_params["normalization"].items()}
    total_weight = sum(weights.values())

    terms = []
    for name, weight in weights.items():
        lo, hi = ranges[name]
        raw = features[name]
        norm = normalize(raw, lo, hi)
        clip = "low" if raw < lo else ("high" if raw > hi else "")
        terms.append({
            "feature": name,
            "raw": raw,
            "calib_p2": lo,
            "calib_p98": hi,
            "normalized": norm,
            "weight": weight,
            "contribution": weight * norm / total_weight,
            "clipped": clip,
        })
    score = sum(t["contribution"] for t in terms)
    label = bucket(score, axis_params["bucket_thresholds"], axis_params["bucket_labels"])
    terms.sort(key=lambda t: -t["contribution"])
    return score, label, terms


def calibration_sparser_reference(calib_rows):
    """The median feature vector of the calibration pool's manually-labeled Sparser FOVs --
    what a genuinely sparse FOV looks like in feature space."""
    sparser = [r for r in calib_rows if r["density_label"].strip().lower() == "sparser"]
    return sparser, {name: statistics.median(float(r[name]) for r in sparser) for name in FEATURE_ORDER}


def plot_diagnosis(decomps, masks, sparser_ref, out_path):
    any_gated = any(d["gated"] for d in decomps)
    fig = plt.figure(figsize=(16, 9.5), dpi=140)
    fig.patch.set_facecolor(COLOR_SURFACE)
    # explicit geometry rather than tight_layout: the imshow panels make this figure one that
    # tight_layout warns it cannot handle, and it silently leaves the header text overlapping
    # the top axes' title. `top` drops when the gate fires, to fit the extra header line.
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.22,
                          left=0.06, right=0.98, bottom=0.04, top=0.86 if any_gated else 0.89)

    # --- top row: stacked contribution bars per FOV ---
    ax = fig.add_subplot(gs[0, :])
    ax.set_facecolor(COLOR_SURFACE)
    names = [d["short"] for d in decomps]
    xs = np.arange(len(names))
    bottoms = np.zeros(len(names))

    feature_order_by_size = [t["feature"] for t in decomps[0]["terms"]]
    for name in feature_order_by_size:
        vals = np.array([next(t["contribution"] for t in d["terms"] if t["feature"] == name) for d in decomps])
        clipped_low = [next(t["clipped"] for t in d["terms"] if t["feature"] == name) == "low" for d in decomps]
        color = COLOR_ALERT if name in MISFIRING else COLOR_MINE
        ax.bar(xs, vals, bottom=bottoms, width=0.55, color=color,
               edgecolor=COLOR_SURFACE, linewidth=1.2, zorder=3)
        for x, v, b in zip(xs, vals, bottoms):
            if v > 0.035:
                ax.text(x, b + v / 2, f"{name}  {v:.3f}", ha="center", va="center",
                        fontsize=8.5, color="white", fontweight="bold", zorder=4)
        bottoms += vals

    # every clipped-low feature contributes exactly 0, so they'd all land at the same height --
    # list them once per bar as a stacked block instead of overplotting four labels
    for x, d in zip(xs, decomps):
        silenced = [t["feature"] for t in d["terms"] if t["clipped"] == "low"]
        if not silenced:
            continue
        caption = "clipped to 0, contributes nothing:\n" + "\n".join(f"  {s}" for s in silenced)
        if d["gated"]:
            caption += "\n\nall four below floor\n=> empty-field gate fires"
        ax.text(x + 0.30, bottoms[x] - 0.01, caption,
                ha="left", va="top", fontsize=7.8, color=COLOR_SILENCED, zorder=4, linespacing=1.5)

    for x, d in zip(xs, decomps):
        # the composite's own verdict, then what actually ships -- on a gated FOV the score
        # below the bar is precisely the number the gate overrode
        ax.text(x, bottoms[x] + 0.022, f"{d['score']:.3f}  {display_level(d['label'])}",
                ha="center", fontsize=10.5, fontweight="bold",
                color=COLOR_SILENCED if d["gated"] else COLOR_PRIMARY)
        if d["gated"]:
            ax.text(x, bottoms[x] + 0.075, f"gated -> {display_level(d['final_label'])}",
                    ha="center", fontsize=10.5, color=COLOR_ALERT, fontweight="bold")

    for t, lbl in zip(V2_THRESHOLDS, V2_THRESHOLD_LABELS):
        ax.axhline(t, color=COLOR_AXIS, linewidth=1, linestyle="--", zorder=1)
        ax.text(len(names) - 0.42, t, f"  {lbl}", va="center", fontsize=8, color=COLOR_MUTED)

    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, max(bottoms) + (0.16 if any(d["gated"] for d in decomps) else 0.10))
    ax.set_ylabel("contribution to density score", color=COLOR_SECONDARY, fontsize=10.5)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(facecolor=COLOR_ALERT, label="misfires on an empty field (Otsu-derived / tile-ratio)"),
        Patch(facecolor=COLOR_MINE, label="reads the field correctly"),
        Patch(facecolor=COLOR_SILENCED, label="clipped to 0 -> contributes nothing"),
    ], loc="upper left", frameon=False, fontsize=9, labelcolor=COLOR_SECONDARY)
    ax.set_title("Density score, decomposed into per-feature contributions",
                 loc="left", color=COLOR_PRIMARY, fontsize=12, fontweight="bold", pad=10)

    # --- bottom row: the Otsu mask evidence ---
    for i, (title, thumb, mask, coverage, eta) in enumerate(masks):
        sub = fig.add_subplot(gs[1, i])
        sub.set_xticks([])
        sub.set_yticks([])
        for spine in sub.spines.values():
            spine.set_color(COLOR_AXIS)
        overlay = cv2.cvtColor(thumb, cv2.COLOR_GRAY2RGB)
        overlay[mask > 0] = (0.55 * overlay[mask > 0] +
                             0.45 * np.array([194, 64, 42])).astype(np.uint8)
        sub.imshow(overlay)
        sub.set_title(title, loc="left", color=COLOR_PRIMARY, fontsize=9.5, fontweight="bold")
        sub.set_xlabel(f"coverage = {coverage:.3f}   otsu_separability = {eta:.3f}",
                       color=COLOR_SECONDARY, fontsize=9)

    fig.suptitle("Why the two near-empty Nigeria FOVs scored Dense / Very Dense",
                 x=0.01, y=1.00, ha="left", color=COLOR_PRIMARY, fontsize=15, fontweight="bold")
    gated_note = ("\nThe four grey features are the empty-field gate's inputs: all four below floor means no cells "
                  "to describe, so the gate discards the composite and returns the bottom bucket."
                  if any_gated else "")
    fig.text(0.01, 0.945,
             "Red overlay = pixels Otsu calls cell foreground. On an empty field the grayscale histogram is unimodal, "
             "so Otsu splits background noise near its middle and reports ~35-48% coverage;\n"
             "saturation_score = coverage x (1 - separability) then reads that as saturation. Median calibration Sparser FOV, "
             f"for reference: coverage = {sparser_ref['coverage']:.3f}, glcm_contrast = {sparser_ref['glcm_contrast']:.1f}."
             f"{gated_note}",
             ha="left", va="top", color=COLOR_SECONDARY, fontsize=9.5)

    fig.savefig(out_path, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Decompose the near-empty Nigeria FOVs' density scores.")
    parser.add_argument("--params", default=str(V2_PARAMS))
    parser.add_argument("--suffix", default="", help="suffix for output filenames, e.g. gated")
    args = parser.parse_args()

    params_path = Path(args.params)
    with open(params_path) as f:
        params = json.load(f)
    gate_cfg = params.get("empty_field_override")
    tail = f"-{args.suffix}" if args.suffix else ""

    global V2_THRESHOLDS, V2_THRESHOLD_LABELS
    V2_THRESHOLDS = params["density"]["bucket_thresholds"]
    V2_THRESHOLD_LABELS = [display_level(l) for l in params["density"]["bucket_labels"][1:]]

    calib_rows = read_csv_dicts(CALIB_FEATURES)
    sparser_rows, sparser_ref = calibration_sparser_reference(calib_rows)

    decomps, masks, csv_rows = [], [], []
    for filename in SPARSE_FOVS + [CONTRAST_FOV]:
        image = load_image(IMAGE_DIR / filename)
        features = compute_features(image)
        score, label, terms = decompose(features, params["density"])
        ov_score, ov_label, _ = decompose(features, params["overlap"])
        final_label, final_ov_label = apply_label_overrides(label, ov_label, features, params)
        gated = empty_field_fired(features, gate_cfg)
        below = empty_field_features_below(features, gate_cfg)
        short = filename.replace("_20251210_031002", "").replace(".bmp", "")
        decomps.append({"short": short, "score": score, "label": label, "terms": terms,
                        "gated": gated, "final_label": final_label})

        for t in terms:
            csv_rows.append({"filename": filename, "axis": "density", **{
                "feature": t["feature"], "raw": round(t["raw"], 6),
                "calib_p2": round(t["calib_p2"], 6), "calib_p98": round(t["calib_p98"], 6),
                "normalized": round(t["normalized"], 4), "weight": round(t["weight"], 4),
                "contribution": round(t["contribution"], 4), "clipped": t["clipped"],
                "calib_sparser_median": round(sparser_ref[t["feature"]], 6),
                # the gate columns repeat per row, but they are what makes the decomposition
                # explain the label that ships rather than the one the composite computed
                "gate_feature": "yes" if t["feature"] in (gate_cfg or {}).get("thresholds", {}) else "",
                "gate_below_floor": "yes" if t["feature"] in below else "",
                "gate_fired": "yes" if gated else "no",
                "composite_label": label,
                "final_label": final_label,
            }})

        print(f"\n=== {filename} ===")
        print(f"  density = {score:.3f} -> {display_level(label)}    "
              f"Rouleaux = {ov_score:.3f} -> {display_level(ov_label)}")
        if gated:
            print(f"  empty-field gate FIRED (all {len(below)} gate features below floor) -> "
                  f"{display_level(final_label)} / {display_level(final_ov_label)}")
        print(f"  {'feature':<24}{'raw':>11}{'p2':>10}{'p98':>10}{'norm':>7}{'weight':>8}{'contrib':>9}   clip")
        for t in terms:
            print(f"  {t['feature']:<24}{t['raw']:>11.4f}{t['calib_p2']:>10.4f}{t['calib_p98']:>10.4f}"
                  f"{t['normalized']:>7.3f}{t['weight']:>8.3f}{t['contribution']:>9.4f}   {t['clipped']}")
        silenced = [t["feature"] for t in terms if t["clipped"] == "low"]
        driving = [(t["feature"], t["contribution"]) for t in terms if t["contribution"] > 0]
        print(f"  driven by: " + ", ".join(f"{n} ({c / score:.0%})" for n, c in driving))
        if silenced:
            print(f"  silenced (clipped to 0, contribute nothing): {', '.join(silenced)}")

        # mask panel for the two sparse FOVs plus the textured contrast FOV
        mask, _ = otsu_segment(image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray, (420, 420), interpolation=cv2.INTER_AREA)
        mask_small = cv2.resize(mask, (420, 420), interpolation=cv2.INTER_NEAREST)
        masks.append((short, thumb, mask_small, features["coverage"], features["otsu_separability"]))

    print(f"\nCalibration pool has {len(sparser_rows)} manually-labeled Sparser FOVs. Median feature vector:")
    for name in FEATURE_ORDER:
        nig = [next(t["raw"] for t in d["terms"] if t["feature"] == name) for d in decomps[:2]]
        print(f"  {name:<24} sparser median={sparser_ref[name]:>10.4f}    "
              f"nigeria 3/4 = {nig[0]:>9.4f} / {nig[1]:>9.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"sparse-score-decomposition{tail}.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "axis", "feature", "raw", "calib_p2",
                                              "calib_p98", "normalized", "weight", "contribution",
                                              "clipped", "calib_sparser_median", "gate_feature",
                                              "gate_below_floor", "gate_fired", "composite_label",
                                              "final_label"])
        writer.writeheader()
        writer.writerows(csv_rows)

    out_png = OUT_DIR / f"sparse-decomposition{tail}.png"
    plot_diagnosis(decomps, masks, sparser_ref, out_png)
    print(f"\nwrote {out_csv} and {out_png}")


if __name__ == "__main__":
    main()
