"""The two slide-level plots: continuous density-vs-Rouleaux, and the 5x5 calibrated bucket grid.

Both read `slide-summary.csv` and write PNGs to `plots/`.

**Why two panels of the same data.** The continuous plot shows where slides actually sit; the
grid shows what the calibration *calls* them. Read together they answer the question this whole
analysis exists for -- whether the v2.2 bucket edges land sensibly on a real cohort or compress
it into a couple of cells. To make them reconcilable, plot 1 draws the same bucket thresholds as
reference lines, so plot 2 is visibly an aggregation of plot 1 rather than a separate claim.

**Axis order.** Density on x, Rouleaux on y in both, so the panels read against each other. This
inverts `plot_bucket_comparison_v2.py`, which puts Rouleaux on x -- `--transpose` flips both if
matching that is preferred.

**`combined_score` is deliberately not plotted.** It is the sum of the two plotted axes, so on
plot 1 it is exactly the anti-diagonal: plotting it would restate information already fully
visible. It stays a `slide-summary.csv` column for ranking.

Colour follows the dataviz procedure: one hue for a single series and no legend box (the title
names the series), a single-hue sequential ramp for magnitudes -- never a rainbow for a
magnitude -- and the two-category option's pair is validated (protan/deutan dE 24.7 against a
target of 8, normal-vision 33.6 against a floor of 15). Light surface only, matching every other
plot in this repo; a dark variant would need its own steps validated against a dark surface
rather than an automatic flip, which is not worth it for a README figure.

Usage:
    python scripts/tanzania-complete-081426/plot_slide_level.py
    python scripts/tanzania-complete-081426/plot_slide_level.py --color-by overexposure
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _plot_common import (  # noqa: E402
    ANNOTATABLE_RANK_COLORS,
    CALIBRATION_SLIDES,
    calibration_means,
    highlight,
)
from catalog import (  # noqa: E402
    ANNOTATABLE_ORDER,
    ANNOTATABLE_RANKS,
    CATALOG_XLSX,
    SHEET_NAME,
    load_tanzania_slides,
)
from _slide_common import PLOTS_DIR, ROOT, SLIDE_SUMMARY_CSV, read_csv_dicts  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_MINE,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    DEFAULT_SCORING_PARAMS,
    DENSITY_LEVELS,
    JITTER_SEED,
    OVERLAP_LEVELS,
    display_level,
)
from src.composite_v2 import bucket  # noqa: E402

COLOR_ALT = "#eb6834"  # the second categorical hue; pair validated with COLOR_MINE

# Mark spec: at dpi 200 an s=46 marker is ~19 px across and a 0.8 pt ring is ~2 px, which is the
# surface ring that keeps overlapping marks legible. 271 slides overlap heavily without it.
MARK_SIZE = 46
MARK_ALPHA = 0.78
RING_WIDTH = 0.8
DPI = 200

# Reused from plot_bucket_comparison_v2.py so jitter is identical run to run.
JITTER_X, JITTER_Y = 0.16, 0.14

AXIS_PAD_FRAC = 0.04


def style_axes(ax):
    ax.set_facecolor(COLOR_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=COLOR_SECONDARY, labelsize=8, length=3, width=1.0, color=COLOR_AXIS)


def titles(fig, title, subtitle):
    fig.suptitle(title, x=0.02, ha="left", color=COLOR_PRIMARY, fontsize=12,
                 fontweight="bold", y=0.985)
    fig.text(0.02, 0.935, subtitle, ha="left", color=COLOR_SECONDARY, fontsize=8.5)


def sequential_cmap():
    """A single-hue ramp built from the project blue: light -> COLOR_MINE -> dark.

    Explicitly not viridis. A rainbow for a magnitude is the anti-pattern this replaces --
    hue implies category, and these values are counts.
    """
    return LinearSegmentedColormap.from_list("mine_seq", ["#eaf1fb", COLOR_MINE, "#123a68"])


def load_rows(summary_csv):
    rows = []
    for row in read_csv_dicts(summary_csv):
        try:
            row["_d"] = float(row["density_mean"])
            row["_o"] = float(row["overlap_mean"])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append(row)
    return rows


def color_values(rows, color_by):
    if color_by == "overexposure":
        return [float(r.get("frac_flagged_overexposure") or 0.0) for r in rows], \
            "fraction of FOVs flagged (overexposure)"
    if color_by == "crop_outlier":
        return [float(r.get("frac_flagged_crop_outlier") or 0.0) for r in rows], \
            "fraction of FOVs flagged (crop outlier)"
    return None, None


def plot_continuous(rows, params, args, out_path):
    xs = [r["_o"] if args.transpose else r["_d"] for r in rows]
    ys = [r["_d"] if args.transpose else r["_o"] for r in rows]

    x_axis, y_axis = ("overlap", "density") if args.transpose else ("density", "overlap")
    x_label = "Mean Rouleaux score" if args.transpose else "Mean density score"
    y_label = "Mean density score" if args.transpose else "Mean Rouleaux score"
    x_levels = OVERLAP_LEVELS if args.transpose else DENSITY_LEVELS
    y_levels = DENSITY_LEVELS if args.transpose else OVERLAP_LEVELS

    fig, ax = plt.subplots(figsize=(7.4, 6.0), facecolor=COLOR_SURFACE)
    style_axes(ax)

    # Bucket thresholds as reference lines. This is what lets plot 2 read as an aggregate of
    # plot 1, and shows how deep into a bucket each slide sits -- i.e. how fragile its label is.
    for t in params[x_axis]["bucket_thresholds"]:
        ax.axvline(t, color=COLOR_GRID, linewidth=1.0, zorder=0)
    for t in params[y_axis]["bucket_thresholds"]:
        ax.axhline(t, color=COLOR_GRID, linewidth=1.0, zorder=0)

    values, cbar_label = color_values(rows, args.color_by)
    if args.color_by == "annotatable":
        # Ordinal, so a single-hue ramp rather than four hues -- see ANNOTATABLE_RANK_COLORS.
        # A legend is required here because there are 4 series; it is keyed to the rank order.
        for name in ANNOTATABLE_ORDER:
            rank = ANNOTATABLE_RANKS[name]
            idx = [i for i, r in enumerate(rows) if r.get("_annot_rank") == rank]
            if not idx:
                continue
            ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], s=MARK_SIZE,
                       c=ANNOTATABLE_RANK_COLORS[rank], alpha=MARK_ALPHA,
                       edgecolors=COLOR_SURFACE, linewidths=RING_WIDTH,
                       label=f"{rank} - {name} (n={len(idx)})", zorder=3)
        unknown = [i for i, r in enumerate(rows) if not r.get("_annot_rank")]
        if unknown:
            ax.scatter([xs[i] for i in unknown], [ys[i] for i in unknown], s=MARK_SIZE,
                       c=COLOR_MUTED, alpha=0.5, edgecolors=COLOR_SURFACE,
                       linewidths=RING_WIDTH, label=f"unrated (n={len(unknown)})", zorder=2)
        leg = ax.legend(frameon=False, fontsize=7.5, loc="upper left", title="ANNOTATABLE",
                        title_fontsize=8)
        leg.get_title().set_color(COLOR_SECONDARY)
        for text in leg.get_texts():
            text.set_color(COLOR_SECONDARY)
    elif args.color_by == "truth":
        warn = [i for i, r in enumerate(rows) if str(r.get("truth_warn")) == "True"]
        clean = [i for i in range(len(rows)) if i not in set(warn)]
        for idx, color, label in ((clean, COLOR_MINE, "positive"),
                                  (warn, COLOR_ALT, "positive (truth conflict)")):
            ax.scatter([xs[i] for i in idx], [ys[i] for i in idx], s=MARK_SIZE, c=color,
                       alpha=MARK_ALPHA, edgecolors=COLOR_SURFACE, linewidths=RING_WIDTH,
                       label=f"{label} (n={len(idx)})", zorder=3)
        legend = ax.legend(frameon=False, fontsize=8, loc="upper left")
        for text in legend.get_texts():
            text.set_color(COLOR_SECONDARY)
    elif values is not None:
        scatter = ax.scatter(xs, ys, s=MARK_SIZE, c=values, cmap=sequential_cmap(),
                             alpha=MARK_ALPHA, edgecolors=COLOR_SURFACE,
                             linewidths=RING_WIDTH, zorder=3)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label(cbar_label, color=COLOR_SECONDARY, fontsize=8)
        cbar.ax.tick_params(colors=COLOR_SECONDARY, labelsize=7)
        cbar.outline.set_visible(False)
    else:
        ax.scatter(xs, ys, s=MARK_SIZE, c=COLOR_MINE, alpha=MARK_ALPHA,
                   edgecolors=COLOR_SURFACE, linewidths=RING_WIDTH, zorder=3)

    # Data-driven limits: the means occupy a fraction of [0,1], and forcing the full range
    # would waste most of the canvas.
    def limits(vals, thresholds):
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, 1e-6)
        return lo - AXIS_PAD_FRAC * span, hi + AXIS_PAD_FRAC * span

    ax.set_xlim(*limits(xs, params[x_axis]["bucket_thresholds"]))
    ax.set_ylim(*limits(ys, params[y_axis]["bucket_thresholds"]))

    # Bucket names in the margins. Only drawn where the visible band is wide enough to hold the
    # text: the Rouleaux thresholds sit 0.077 and 0.103 apart, so rotated labels for adjacent
    # narrow bands overlap each other ("Some Rouleaux" over "Slight Rouleaux") if drawn
    # unconditionally. A skipped label costs nothing -- the grid line is still there and plot 2
    # names every bucket on its axes.
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    x_span, y_span = x_hi - x_lo, y_hi - y_lo

    def visible_bands(edges, lo, hi):
        for i in range(len(edges) - 1):
            left, right = max(edges[i], lo), min(edges[i + 1], hi)
            yield i, left, right, right - left

    x_edges = [x_lo] + list(params[x_axis]["bucket_thresholds"]) + [x_hi]
    for i, left, right, width in visible_bands(x_edges, x_lo, x_hi):
        if width < 0.10 * x_span:
            continue
        ax.text((left + right) / 2, y_hi, display_level(x_levels[i]), ha="center", va="bottom",
                color=COLOR_MUTED, fontsize=7)

    y_edges = [y_lo] + list(params[y_axis]["bucket_thresholds"]) + [y_hi]
    for i, bottom, top, height in visible_bands(y_edges, y_lo, y_hi):
        if height < 0.16 * y_span:
            continue
        ax.text(x_hi + 0.012 * x_span, (bottom + top) / 2, display_level(y_levels[i]),
                ha="left", va="center", color=COLOR_MUTED, fontsize=7, rotation=270)

    # The two v2.2 calibration slides. ...946 is not in the 271 and so is not in `rows`; its means
    # come from its per-FOV CSV, which is what the summary would have averaged anyway.
    means = calibration_means()
    points = []
    for slide_id in CALIBRATION_SLIDES:
        if slide_id not in means:
            continue
        d, o = means[slide_id]
        points.append((slide_id, o if args.transpose else d, d if args.transpose else o))
    highlight(ax, points, COLOR_PRIMARY, dy=0.035)

    ax.set_xlabel(x_label, color=COLOR_SECONDARY, fontsize=9)
    ax.set_ylabel(y_label, color=COLOR_SECONDARY, fontsize=9)
    titles(fig,
           f"Slide-level density vs. Rouleaux - {len(rows)} Tanzania slides",
           f"Mean of per-FOV {params.get('version', 'v2')} composite scores; "
           f"grid lines = the same fit's per-FOV bucket thresholds")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_grid(rows, params, args, out_path):
    x_key, y_key = ("overlap_bucket", "density_bucket") if args.transpose \
        else ("density_bucket", "overlap_bucket")
    x_levels = OVERLAP_LEVELS if args.transpose else DENSITY_LEVELS
    y_levels = DENSITY_LEVELS if args.transpose else OVERLAP_LEVELS
    x_label = "Rouleaux bucket" if args.transpose else "Density bucket"
    y_label = "Density bucket" if args.transpose else "Rouleaux bucket"

    placed = [r for r in rows if r.get(x_key) in x_levels and r.get(y_key) in y_levels]
    counts = Counter((x_levels.index(r[x_key]), y_levels.index(r[y_key])) for r in placed)

    rng = random.Random(JITTER_SEED)
    fig, ax = plt.subplots(figsize=(7.4, 6.0), facecolor=COLOR_SURFACE)
    style_axes(ax)

    for i in range(len(x_levels) + 1):
        ax.axvline(i - 0.5, color=COLOR_GRID, linewidth=1.0, zorder=0)
    for j in range(len(y_levels) + 1):
        ax.axhline(j - 0.5, color=COLOR_GRID, linewidth=1.0, zorder=0)

    xs, ys = [], []
    calib_points = []
    for row in placed:
        x = x_levels.index(row[x_key]) + rng.uniform(-JITTER_X, JITTER_X)
        y = y_levels.index(row[y_key]) + rng.uniform(-JITTER_Y, JITTER_Y)
        xs.append(x)
        ys.append(y)
        if row["slide_id"] in CALIBRATION_SLIDES:
            calib_points.append((row["slide_id"], x, y))
    ax.scatter(xs, ys, s=MARK_SIZE, c=COLOR_MINE, alpha=MARK_ALPHA,
               edgecolors=COLOR_SURFACE, linewidths=RING_WIDTH, zorder=3)

    # A calibration slide with no summary row still has per-FOV scores, so bucket its means the
    # same way the aggregator would and drop it in its cell.
    placed_ids = {r["slide_id"] for r in placed}
    for slide_id, (d_mean, o_mean) in calibration_means().items():
        if slide_id in placed_ids:
            continue
        d_bucket = bucket(d_mean, params["density"]["bucket_thresholds"],
                          params["density"]["bucket_labels"])
        o_bucket = bucket(o_mean, params["overlap"]["bucket_thresholds"],
                          params["overlap"]["bucket_labels"])
        bx, by = (o_bucket, d_bucket) if args.transpose else (d_bucket, o_bucket)
        if bx not in x_levels or by not in y_levels:
            continue
        x = x_levels.index(bx) + rng.uniform(-JITTER_X, JITTER_X)
        y = y_levels.index(by) + rng.uniform(-JITTER_Y, JITTER_Y)
        ax.scatter([x], [y], s=MARK_SIZE, c=COLOR_MINE, alpha=MARK_ALPHA,
                   edgecolors=COLOR_SURFACE, linewidths=RING_WIDTH, zorder=3)
        calib_points.append((slide_id, x, y))

    # Label below the ring here, not above: the per-cell counts sit at each cell's top-left, and
    # an above-ring label collides with them (it crossed the "11" in slightly-dense/slight).
    highlight(ax, calib_points, COLOR_PRIMARY, dy=-0.075)

    # Per-cell counts. 271 dots over 25 cells will be very unevenly spread, and a count makes
    # the sparse and saturated cells readable without counting dots. No heatmap fill behind
    # them -- that would double-encode the same number the dots already carry.
    for (i, j), n in counts.items():
        ax.text(i - 0.45, j + 0.4, str(n), ha="left", va="top", color=COLOR_MUTED, fontsize=7.5)

    ax.set_xticks(range(len(x_levels)))
    ax.set_xticklabels([display_level(v) for v in x_levels], rotation=15, ha="right")
    ax.set_yticks(range(len(y_levels)))
    ax.set_yticklabels([display_level(v) for v in y_levels])
    ax.set_xlim(-0.5, len(x_levels) - 0.5)
    ax.set_ylim(-0.5, len(y_levels) - 0.5)
    ax.set_xlabel(x_label, color=COLOR_SECONDARY, fontsize=9)
    ax.set_ylabel(y_label, color=COLOR_SECONDARY, fontsize=9)

    titles(fig,
           f"Slide-level bucket grid - {len(placed)} Tanzania slides, "
           f"{len(counts)}/{len(x_levels) * len(y_levels)} cells occupied",
           f"Bucket = {params.get('version', 'v2')} per-FOV thresholds applied to each slide's "
           f"MEAN score; jittered within each cell")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-csv", default=str(SLIDE_SUMMARY_CSV))
    parser.add_argument("--params", default=str(DEFAULT_SCORING_PARAMS))
    parser.add_argument("--xlsx", default=str(CATALOG_XLSX))
    parser.add_argument("--sheet", default=SHEET_NAME)
    parser.add_argument("--plots-dir", default=str(PLOTS_DIR))
    parser.add_argument("--color-by",
                        choices=("none", "overexposure", "crop_outlier", "truth", "annotatable"),
                        default="none")
    parser.add_argument("--transpose", action="store_true",
                        help="Rouleaux on x and density on y, matching "
                             "plot_bucket_comparison_v2.py's axis order")
    parser.add_argument("--suffix", default="", help="appended to both output filenames")
    args = parser.parse_args()

    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)

    rows = load_rows(args.summary_csv)
    if args.color_by == "annotatable":
        # Annotatability lives in the catalog workbook, not slide-summary.csv.
        ranks = {s["slide_id"]: s["annotatable_rank"]
                 for s in load_tanzania_slides(args.xlsx, args.sheet)}
        for row in rows:
            row["_annot_rank"] = ranks.get(row["slide_id"]) or None
    if not rows:
        raise SystemExit(f"no slides with a density_mean in {args.summary_csv} -- run the "
                         "crowding pass and aggregate_slides.py first")

    plots_dir = Path(args.plots_dir)
    suffix = args.suffix or ("-" + args.color_by if args.color_by != "none" else "")
    cont = plot_continuous(rows, params, args,
                           plots_dir / f"slide-density-vs-rouleaux-continuous{suffix}.png")
    grid_path, counts = plot_grid(rows, params, args,
                                 plots_dir / f"slide-density-vs-rouleaux-grid{suffix}.png")

    print(f"slides plotted: {len(rows)}")
    print(f"wrote {cont}")
    print(f"wrote {grid_path}")
    print(f"occupied grid cells: {len(counts)}/25")
    if counts:
        busiest = max(counts.items(), key=lambda kv: kv[1])
        print(f"busiest cell: {busiest[1]} slides")
    return 0


if __name__ == "__main__":
    sys.exit(main())
