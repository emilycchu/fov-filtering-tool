"""Do the catalog's annotatability judgements track any of the automated slide-level indicators?

Three figures, all 271 Tanzania slides:

  1. combined crowding score vs. # FOVs flagged by the fluorescence halo detector
  2. combined crowding score vs. # FOVs flagged by the crop-outlier z-score
  3. a 3-panel strip: annotatability rank vs. each of the three indicators

Figures 1-2 are the same plot against the two different flag sources, coloured by annotatability.
Figure 3 puts annotatability on the x-axis directly, which is the actual question -- 1 and 2 can
only show it as a third variable.

**Colour encodes an ordinal, so it is a single-hue ramp, not four separate hues.** ANNOTATABLE has
a natural order (hard < can spot annotate < annotatable (hard) < annotatable) and the whole point
is whether that order lines up with an indicator. Four categorical hues would throw the ordering
away and invite reading a rainbow as unordered identity. The ramp is magenta rather than blue,
because blue is what every other figure here uses for slides -- see ANNOTATABLE_RANK_COLORS in
`_plot_common.py` for the hue choice and its validation.

`combined_score` is used as the crowding indicator because density and Rouleaux slide means
correlate at 0.995 -- they are one measurement at slide level, so which one is plotted barely
matters. `--crowding-metric` switches it if wanted.

Usage:
    python scripts/tanzania-complete-081426/plot_annotatability.py
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _plot_common import (  # noqa: E402
    ANNOTATABLE_RANK_COLORS as RANK_COLORS,
    CALIBRATION_SLIDES,
    highlight,
    legend_note,
)
from _slide_common import (  # noqa: E402
    PLOTS_DIR,
    RESULTS_DIR,
    ROOT,
    SLIDE_SUMMARY_CSV,
    read_csv_dicts,
    write_csv_atomic,
)
from catalog import (  # noqa: E402
    ANNOTATABLE_ORDER,
    ANNOTATABLE_RANKS,
    CATALOG_XLSX,
    SHEET_NAME,
    load_tanzania_slides,
)

sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
)

MARK_SIZE = 46
MARK_ALPHA = 0.8
RING_WIDTH = 0.8
DPI = 200
JITTER_SEED = 7

INDICATORS = [
    ("combined_score", "Combined crowding score (density + Rouleaux mean)"),
    ("n_flagged_overexposure", "FOVs flagged: fluorescence halo detector"),
    ("n_flagged_crop_outlier", "FOVs flagged: crop-outlier z >= 6"),
]


def style_axes(ax):
    ax.set_facecolor(COLOR_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=COLOR_SECONDARY, labelsize=8, length=3, width=1.0, color=COLOR_AXIS)


def spearman(xs, ys):
    """Rank correlation, ties averaged. Spearman rather than Pearson because annotatability is an
    ordinal rank, and the flag counts are heavily skewed with a long right tail."""
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            mean_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = mean_rank
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def load_rows(summary_csv, xlsx, sheet):
    annot = {s["slide_id"]: s for s in load_tanzania_slides(xlsx, sheet)}
    rows = []
    for row in read_csv_dicts(summary_csv):
        meta = annot.get(row["slide_id"])
        if not meta or not meta["annotatable_rank"]:
            continue
        try:
            row["combined_score"] = float(row["combined_score"])
            row["n_flagged_overexposure"] = int(row["n_flagged_overexposure"])
            row["n_flagged_crop_outlier"] = int(row["n_flagged_crop_outlier"])
        except (TypeError, ValueError):
            continue
        row["annotatable"] = meta["annotatable"]
        row["annotatable_rank"] = int(meta["annotatable_rank"])
        rows.append(row)
    return rows


def legend(ax, counts):
    handles = []
    for name in ANNOTATABLE_ORDER:
        rank = ANNOTATABLE_RANKS[name]
        handles.append(ax.scatter([], [], s=MARK_SIZE, c=RANK_COLORS[rank],
                                  edgecolors=COLOR_SURFACE, linewidths=RING_WIDTH,
                                  label=f"{rank} - {name} (n={counts.get(name, 0)})"))
    leg = ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper left",
                    title="ANNOTATABLE", title_fontsize=8, labelspacing=0.5)
    leg.get_title().set_color(COLOR_SECONDARY)
    for text in leg.get_texts():
        text.set_color(COLOR_SECONDARY)


def scatter_vs_flags(rows, flag_key, flag_label, out_path, counts):
    xs = [r["combined_score"] for r in rows]
    ys = [r[flag_key] for r in rows]
    colors = [RANK_COLORS[r["annotatable_rank"]] for r in rows]

    fig, ax = plt.subplots(figsize=(7.6, 5.9), facecolor=COLOR_SURFACE)
    style_axes(ax)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.scatter(xs, ys, s=MARK_SIZE, c=colors, alpha=MARK_ALPHA, edgecolors=COLOR_SURFACE,
               linewidths=RING_WIDTH, zorder=3)

    rho_all = spearman(xs, ys)
    rho_annot = spearman([r["annotatable_rank"] for r in rows], ys)
    ax.set_xlabel("Combined crowding score (density mean + Rouleaux mean)",
                  color=COLOR_SECONDARY, fontsize=9)
    ax.set_ylabel(flag_label, color=COLOR_SECONDARY, fontsize=9)
    legend(ax, counts)

    by_id = {r["slide_id"]: r for r in rows}
    points = [(s, by_id[s]["combined_score"], by_id[s][flag_key])
              for s in CALIBRATION_SLIDES if s in by_id]
    highlight(ax, points, COLOR_PRIMARY)
    missing = [s for s in CALIBRATION_SLIDES if s not in by_id]

    fig.suptitle(f"{flag_label} vs. crowding, by annotatability", x=0.02, ha="left",
                 color=COLOR_PRIMARY, fontsize=12, fontweight="bold", y=0.985)
    fig.text(0.02, 0.935,
             f"n = {len(rows)} Tanzania slides - Spearman rho: flags vs crowding "
             f"{rho_all:+.3f}, flags vs annotatability rank {rho_annot:+.3f}",
             ha="left", color=COLOR_SECONDARY, fontsize=8.5)
    fig.text(0.02, 0.915, legend_note(missing), ha="left", color=COLOR_MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return rho_all, rho_annot


def strips_vs_rank(rows, out_path):
    """One panel per indicator: annotatability rank on x, indicator on y, jittered."""
    rng = random.Random(JITTER_SEED)
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.5), facecolor=COLOR_SURFACE)
    stats = {}

    for ax, (key, label) in zip(axes, INDICATORS):
        style_axes(ax)
        ax.grid(axis="y", color=COLOR_GRID, linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)
        for rank in (1, 2, 3, 4):
            group = [r for r in rows if r["annotatable_rank"] == rank]
            xs = [rank + rng.uniform(-0.22, 0.22) for _ in group]
            ys = [r[key] for r in group]
            ax.scatter(xs, ys, s=MARK_SIZE, c=RANK_COLORS[rank], alpha=MARK_ALPHA,
                       edgecolors=COLOR_SURFACE, linewidths=RING_WIDTH, zorder=3)
            if ys:
                med = sorted(ys)[len(ys) // 2]
                # A median bar per rank: the eye cannot judge the centre of a jittered
                # cloud, and a monotone trend across ranks is the entire question.
                ax.plot([rank - 0.34, rank + 0.34], [med, med], color=COLOR_PRIMARY,
                        linewidth=1.6, zorder=4, solid_capstyle="butt")

        rho = spearman([r["annotatable_rank"] for r in rows], [r[key] for r in rows])
        stats[key] = rho

        by_id = {r["slide_id"]: r for r in rows}
        points = [(s, by_id[s]["annotatable_rank"], by_id[s][key])
                  for s in CALIBRATION_SLIDES if s in by_id]
        # dy clears the star's upper point, which reaches further than the old ring did.
        highlight(ax, points, COLOR_PRIMARY, fontsize=7, dy=0.055)

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(["1\nhard", "2\ncan spot\nannotate", "3\nannotatable\n(hard)",
                            "4\nannotatable"], fontsize=7.5)
        ax.set_xlim(0.55, 4.45)
        ax.set_xlabel("ANNOTATABLE rank", color=COLOR_SECONDARY, fontsize=8.5)
        ax.set_title(label, color=COLOR_PRIMARY, fontsize=9, pad=6)
        ax.text(0.98, 0.97, f"rho {rho:+.3f}", transform=ax.transAxes, ha="right", va="top",
                color=COLOR_MUTED, fontsize=8)

    fig.suptitle("Slide-level indicators vs. catalog annotatability", x=0.01, ha="left",
                 color=COLOR_PRIMARY, fontsize=12, fontweight="bold", y=0.995)
    missing = [s for s in CALIBRATION_SLIDES if s not in {r["slide_id"] for r in rows}]
    fig.text(0.01, 0.93,
             f"n = {len(rows)} Tanzania slides - black bar = per-rank median - "
             f"rho = Spearman rank correlation vs. the 1-4 rank",
             ha="left", color=COLOR_SECONDARY, fontsize=8.5)
    fig.text(0.01, 0.905, legend_note(missing), ha="left", color=COLOR_MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.885))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-csv", default=str(SLIDE_SUMMARY_CSV))
    parser.add_argument("--xlsx", default=str(CATALOG_XLSX))
    parser.add_argument("--sheet", default=SHEET_NAME)
    parser.add_argument("--plots-dir", default=str(PLOTS_DIR))
    args = parser.parse_args()

    rows = load_rows(args.summary_csv, args.xlsx, args.sheet)
    if not rows:
        raise SystemExit("no slides with both a summary row and an ANNOTATABLE value")
    counts = defaultdict(int)
    for row in rows:
        counts[row["annotatable"]] += 1
    print(f"slides: {len(rows)}  annotatable: {dict(counts)}")

    plots = Path(args.plots_dir)
    r1 = scatter_vs_flags(rows, "n_flagged_overexposure",
                          "FOVs flagged: fluorescence halo detector",
                          plots / "annotatability-crowding-vs-overexposure.png", counts)
    r2 = scatter_vs_flags(rows, "n_flagged_crop_outlier",
                          "FOVs flagged: crop-outlier z >= 6",
                          plots / "annotatability-crowding-vs-crop-outlier.png", counts)
    stats = strips_vs_rank(rows, plots / "annotatability-vs-indicators.png")

    print("\nSpearman rho vs ANNOTATABLE rank (1 hard .. 4 annotatable):")
    for key, _label in INDICATORS:
        print(f"  {key:26s} {stats[key]:+.3f}")
    print(f"\nflags vs crowding: overexposure {r1[0]:+.3f}, crop-outlier {r2[0]:+.3f}")

    # Per-rank medians, so the numbers behind the strip panels are readable without the figure.
    out = []
    for rank in (1, 2, 3, 4):
        group = [r for r in rows if r["annotatable_rank"] == rank]
        row = {"annotatable_rank": rank,
               "annotatable": next(n for n, v in ANNOTATABLE_RANKS.items() if v == rank),
               "n_slides": len(group)}
        for key, _label in INDICATORS:
            values = sorted(r[key] for r in group)
            row[f"median_{key}"] = round(values[len(values) // 2], 4) if values else ""
        out.append(row)
    manifest = RESULTS_DIR / "annotatability-summary.csv"
    write_csv_atomic(manifest, list(out[0].keys()), out)
    print(f"\nwrote {manifest}")
    for row in out:
        print(f"  rank {row['annotatable_rank']} {row['annotatable']:20s} n={row['n_slides']:3d} "
              f"crowding {row['median_combined_score']:.3f}  "
              f"halo {row['median_n_flagged_overexposure']}  "
              f"crop {row['median_n_flagged_crop_outlier']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
