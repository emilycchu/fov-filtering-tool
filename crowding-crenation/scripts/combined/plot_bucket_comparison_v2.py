"""Plot manual annotations vs. score_fov_v2 model predictions on the same density x Rouleaux
severity grid, for every FOV in the calibration set -- colored by source (mine vs. model),
jittered within each grid cell. Mirrors scripts/compare_tanzania_labels.py's
plot_overlap_vs_density, extended to the full 5-level density scale (sparser kept distinct)
and the v2 fitted model.

Applies the exact scoring functions score_fov_v2.py runs on new images (weighted_composite +
bucket + saturation override) to the already-extracted features.csv, rather than re-running
image processing on all 337 images -- compute_features() is shared by extract_features_v2.py
and score_fov_v2.py, so this produces identical predictions to actually running the tool.

Usage:
    python scripts/combined/plot_bucket_comparison_v2.py [--features-csv PATH] [--params PATH]
        [--out PATH]
"""
import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from _v2_common import (
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_MINE,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    DENSITY_LEVELS,
    FEATURES_CSV,
    JITTER_SEED,
    OVERLAP_LEVELS,
    PARAMS_JSON,
    PLOTS_DIR,
    apply_label_overrides,
    density_ordinal,
    display_level,
    overlap_ordinal,
    read_csv_dicts,
)
from src.composite_v2 import bucket, weighted_composite

COLOR_MODEL = "#eb6834"  # matches scripts/compare_tanzania_labels.py's COLOR_PROGRAM


def _axis_score_and_label(features, axis_params):
    weights = axis_params["weights"]
    ranges = {n: (v["min"], v["max"]) for n, v in axis_params["normalization"].items()}
    score = weighted_composite(features, weights, ranges)
    label = bucket(score, axis_params["bucket_thresholds"], axis_params["bucket_labels"])
    return score, label


def score_row(row, params):
    _, density_label = _axis_score_and_label(row, params["density"])
    _, overlap_label = _axis_score_and_label(row, params["overlap"])
    density_label, overlap_label = apply_label_overrides(density_label, overlap_label, row, params)
    return density_ordinal(density_label), overlap_ordinal(overlap_label)


def load_rows(features_csv, params):
    rows = read_csv_dicts(features_csv)
    all_feature_names = set(params["density"]["feature_names"]) | set(params["overlap"]["feature_names"])
    for r in rows:
        for name in all_feature_names:
            r[name] = float(r[name])
    return rows


def plot_bucket_comparison(mine, model, n_fovs, out_path):
    rng = random.Random(JITTER_SEED)
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    fig.patch.set_facecolor(COLOR_SURFACE)
    ax.set_facecolor(COLOR_SURFACE)

    def scatter_group(records, color, x_shift, label):
        xs = [ov + x_shift + rng.uniform(-0.14, 0.14) for _, ov in records]
        ys = [dn + rng.uniform(-0.16, 0.16) for dn, _ in records]
        ax.scatter(xs, ys, s=24, color=color, alpha=0.5, linewidths=0, zorder=3, label=label)

    scatter_group(mine, COLOR_MINE, -0.15, "Mine (manual annotation)")
    scatter_group(model, COLOR_MODEL, 0.15, "score_fov_v2 (model)")

    ax.set_xticks(range(len(OVERLAP_LEVELS)))
    ax.set_xticklabels([display_level(l) for l in OVERLAP_LEVELS], fontsize=10, rotation=15, ha="right")
    ax.set_xlim(-0.5, len(OVERLAP_LEVELS) - 0.5)
    for i in range(1, len(OVERLAP_LEVELS)):
        ax.axvline(i - 0.5, color=COLOR_GRID, linewidth=1, zorder=0)

    ax.set_yticks(range(len(DENSITY_LEVELS)))
    ax.set_yticklabels([display_level(l) for l in DENSITY_LEVELS], fontsize=10)
    ax.set_ylim(-0.5, len(DENSITY_LEVELS) - 0.5)
    for i in range(1, len(DENSITY_LEVELS)):
        ax.axhline(i - 0.5, color=COLOR_GRID, linewidth=1, zorder=0)

    ax.tick_params(colors=COLOR_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.set_axisbelow(True)

    ax.set_xlabel("Rouleaux level", color=COLOR_SECONDARY, fontsize=11)
    ax.set_ylabel("Density level", color=COLOR_SECONDARY, fontsize=11)

    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_MINE, markersize=8, label="Mine (manual annotation)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLOR_MODEL, markersize=8, label="score_fov_v2 (model)"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9)

    fig.suptitle("Manual annotations vs. score_fov_v2 predictions -- density x Rouleaux severity grid",
                 x=0.02, y=0.98, ha="left", color=COLOR_PRIMARY, fontsize=14, fontweight="bold")
    ax.set_title(f"n = {n_fovs} FOVs each, jittered within each grid cell to show spread",
                 loc="left", color=COLOR_SECONDARY, fontsize=10, pad=10)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot manual annotations vs. score_fov_v2 predictions on the density x Rouleaux grid.")
    parser.add_argument("--features-csv", default=str(FEATURES_CSV))
    parser.add_argument("--params", default=str(PARAMS_JSON))
    parser.add_argument("--out", default=str(PLOTS_DIR / "bucket-comparison-v2.png"))
    args = parser.parse_args()

    with open(args.params) as f:
        params = json.load(f)
    rows = load_rows(args.features_csv, params)

    mine, model = [], []
    for r in rows:
        mine.append((density_ordinal(r["density_label"]), overlap_ordinal(r["overlap_label"])))
        model.append(score_row(r, params))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_bucket_comparison(mine, model, len(rows), out_path)

    exact_density = sum(1 for (md, _), (pd, _) in zip(mine, model) if md == pd) / len(mine)
    exact_overlap = sum(1 for (_, mo), (_, po) in zip(mine, model) if mo == po) / len(mine)
    print(f"n={len(mine)}, density exact-match={exact_density:.1%}, Rouleaux exact-match={exact_overlap:.1%}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
