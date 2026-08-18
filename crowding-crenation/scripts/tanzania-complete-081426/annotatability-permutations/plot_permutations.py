"""Figures for the annotatability aggregation sweep. Reads what `build_permutations.py` wrote.

Seven figures:

  1-4  one jittered panel per statistic, grouped by family (whole-slide, percentiles, fractions,
       trimmed means) -- annotatability rank on x, the statistic on y, exactly the treatment
       `plot_annotatability.strips_vs_rank` established, so these read against the original figure
       without translation.
  5    rho vs. sweep parameter for the two swept families, with bootstrap CI bands. 39 jittered
       panels cannot show a monotone trend; this is the figure where the trimmed-mean decline is
       legible.
  6    the summary: all 53 statistics, sorted by |rho|, with paired-difference CI bars against the
       mean baseline.
  7    crowding by collection site, boxes overlaid with the individual slides.

Two deliberate departures from the sibling figures in this dataset:

- **No yellow star callouts.** Both v2.2 calibration slides are excluded from this analysis, so
  there is nothing to call out. Each figure says so in its subtitle rather than leaving a reader to
  wonder whether the stars were forgotten.
- **Figure 6 encodes family by marker shape, not colour.** The magenta ramp is spent on
  annotatability everywhere else in this dataset, and adding a second categorical palette to carry
  family would collide with it. Shape is a free channel and needs no contrast validation.

Usage:
    python scripts/tanzania-complete-081426/annotatability-permutations/plot_permutations.py
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from _plot_common import ANNOTATABLE_RANK_COLORS as RANK_COLORS, CALIBRATION_SLIDES  # noqa: E402
from _slide_common import ROOT, read_csv_dicts  # noqa: E402
from catalog import ANNOTATABLE_ORDER, ANNOTATABLE_RANKS  # noqa: E402
from plot_annotatability import (  # noqa: E402
    DPI,
    JITTER_SEED,
    MARK_ALPHA,
    MARK_SIZE,
    RING_WIDTH,
    spearman,
    style_axes,
)
from build_permutations import (  # noqa: E402
    BASELINE,
    CORRELATIONS_CSV,
    METRICS_CSV,
    PERCENTILES,
    RESULTS_DIR,
    TRIMS,
    WHOLE_SLIDE,
    family_of,
    statistic_names,
)

sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_MINE,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
)

PLOTS_DIR = RESULTS_DIR / "plots"
RANK_TICKS = ["1\nhard", "2\ncan spot\nannotate", "3\nannotatable\n(hard)", "4\nannotatable"]
# Family -> marker, for figure 6. Filled shapes only, so every one carries the surface ring.
FAMILY_MARKERS = {"whole-slide": "o", "percentile": "s", "fraction": "D", "trimmed-mean": "^"}
EXCLUSION_NOTE = (f"calibration slides {', '.join('...' + s[-3:] for s in CALIBRATION_SLIDES)} "
                  f"excluded -- the v2.2 fit was trained on them")


def pretty(name):
    """Statistic key -> axis title."""
    if name in WHOLE_SLIDE:
        return {"mean": "mean (baseline)", "stdev": "std. deviation", "mad": "MAD",
                "iqr": "IQR (p75-p25)", "range": "range (max-min)"}.get(name, name)
    if name.startswith("frac_density_ge_"):
        return "frac FOVs >= " + name[len("frac_density_ge_"):].replace("_", " ")
    if name.startswith("frac_rouleaux_ge_"):
        return "frac FOVs >= " + name[len("frac_rouleaux_ge_"):].replace("_", " ")
    if name.startswith("top"):
        return f"mean of top {name[3:-5]}%"
    return name


def load():
    slides = []
    for row in read_csv_dicts(METRICS_CSV):
        row["annotatable_rank"] = int(row["annotatable_rank"])
        for name in statistic_names():
            row[name] = float(row[name])
        slides.append(row)
    correlations = {}
    for row in read_csv_dicts(CORRELATIONS_CSV):
        for key in ("rho", "rho_ci_lo", "rho_ci_hi", "delta_vs_mean", "delta_ci_lo",
                    "delta_ci_hi", "rho_ungated", "max_abs_rho_perm_p"):
            row[key] = float(row[key])
        row["differs_from_mean"] = row["differs_from_mean"] == "True"
        correlations[row["statistic"]] = row
    return slides, correlations


def _rank_handles(ax, counts):
    return [ax.scatter([], [], s=MARK_SIZE, c=RANK_COLORS[ANNOTATABLE_RANKS[name]],
                       edgecolors=COLOR_SURFACE, linewidths=RING_WIDTH,
                       label=f"{ANNOTATABLE_RANKS[name]} - {name} (n={counts.get(name, 0)})")
            for name in ANNOTATABLE_ORDER]


def _style_legend(leg):
    leg.get_title().set_color(COLOR_SECONDARY)
    for text in leg.get_texts():
        text.set_color(COLOR_SECONDARY)


def rank_legend(ax, counts, loc="upper right", fontsize=7.0, **kwargs):
    leg = ax.legend(handles=_rank_handles(ax, counts), frameon=False, fontsize=fontsize, loc=loc,
                    title="ANNOTATABLE", title_fontsize=fontsize + 0.5, labelspacing=0.45,
                    **kwargs)
    _style_legend(leg)


def header(fig, title, subtitle, note, legend_handles=None, legend_ncol=4):
    """Title block with spacing measured in inches, not axes fraction.

    These figures range from 4.6 to 13 inches tall, so a fixed fractional offset that looks right on
    one is either cramped or floating on another. Returns the `rect` top for `tight_layout`.
    """
    height = fig.get_figheight()
    def y(inches):
        return 1.0 - inches / height
    fig.suptitle(title, x=0.01, ha="left", color=COLOR_PRIMARY, fontsize=12.5,
                 fontweight="bold", y=y(0.30))
    fig.text(0.01, y(0.55), subtitle, ha="left", va="top", color=COLOR_SECONDARY, fontsize=8.5)
    fig.text(0.01, y(0.76), note, ha="left", va="top", color=COLOR_MUTED, fontsize=7.5)
    if legend_handles:
        # The legend lives in the header band, never over the panels: at 270 points per panel there
        # is no reliable empty corner to drop it into, and it covered data in every earlier position.
        leg = fig.legend(handles=legend_handles, frameon=False, fontsize=7.5, ncol=legend_ncol,
                         loc="upper right", bbox_to_anchor=(0.995, y(0.14)),
                         title="ANNOTATABLE", title_fontsize=8, columnspacing=1.4)
        _style_legend(leg)
    return y(1.05)


def strip_panel(ax, slides, key, rho, rng, title_size=8.5):
    """One jittered rank-vs-statistic panel, matching `plot_annotatability.strips_vs_rank`."""
    style_axes(ax)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    for rank in (1, 2, 3, 4):
        group = [s for s in slides if s["annotatable_rank"] == rank]
        xs = [rank + rng.uniform(-0.22, 0.22) for _ in group]
        ys = [s[key] for s in group]
        ax.scatter(xs, ys, s=MARK_SIZE * 0.55, c=RANK_COLORS[rank], alpha=MARK_ALPHA,
                   edgecolors=COLOR_SURFACE, linewidths=RING_WIDTH * 0.7, zorder=3)
        if ys:
            median = sorted(ys)[len(ys) // 2]
            # The eye cannot judge the centre of a jittered cloud, and a monotone trend across
            # ranks is the entire question -- so each rank gets an explicit median bar.
            ax.plot([rank - 0.34, rank + 0.34], [median, median], color=COLOR_PRIMARY,
                    linewidth=1.5, zorder=4, solid_capstyle="butt")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xticklabels(["1", "2", "3", "4"], fontsize=7.5)
    ax.set_xlim(0.55, 4.45)
    # rho goes in the title, not a corner of the panel: the fraction statistics pile up against
    # y=1.0, so every in-panel position collides with data in at least one panel of some grid.
    ax.set_title(f"{pretty(key)}   -   rho {rho:+.3f}", color=COLOR_PRIMARY,
                 fontsize=title_size, pad=5)


def strip_grid(slides, correlations, keys, nrows, ncols, out_path, title, subtitle,
               figsize, counts):
    rng = random.Random(JITTER_SEED)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=COLOR_SURFACE)
    flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, key in zip(flat, keys):
        strip_panel(ax, slides, key, correlations[key]["rho"], rng)
    for ax in flat[len(keys):]:
        ax.axis("off")
    top = header(fig, title, subtitle,
                 f"x = ANNOTATABLE rank (1 hard .. 4 annotatable), one point per slide, jittered - "
                 f"black bar = per-rank median - {EXCLUSION_NOTE}",
                 legend_handles=_rank_handles(flat[0], counts))
    fig.tight_layout(rect=(0, 0, 1, top))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def sweep_curves(correlations, out_path):
    """rho vs. sweep parameter, one panel per swept family.

    Two panels rather than two series on shared axes: a percentile p and a trim fraction X are
    different quantities, and drawing them against one x would invent a correspondence between
    "p20" and "top 20%" that does not exist.
    """
    panels = [
        ([f"p{p}" for p in PERCENTILES], [p for p in PERCENTILES],
         "Percentile of a slide's FOV scores", "Percentile value (p5 .. p100)"),
        ([f"top{pct}_mean" for pct in TRIMS], [pct for pct in TRIMS],
         "Top X% of FOVs averaged (X, descending)", "Trimmed mean (top 95% .. top 5%)"),
    ]
    baseline = correlations[BASELINE]["rho"]
    # sharey because both panels plot the same quantity: on independent scales the trimmed-mean
    # curve's 0.02 drift would render at the same visual slope as a real effect.
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), facecolor=COLOR_SURFACE, sharey=True)
    for ax, (keys, xs, xlabel, title) in zip(axes, panels):
        style_axes(ax)
        ax.grid(axis="y", color=COLOR_GRID, linewidth=1.0, zorder=0)
        ax.set_axisbelow(True)
        rhos = [correlations[k]["rho"] for k in keys]
        los = [correlations[k]["rho_ci_lo"] for k in keys]
        his = [correlations[k]["rho_ci_hi"] for k in keys]
        ax.axhline(baseline, color=COLOR_PRIMARY, linewidth=1.2, zorder=2)
        ax.fill_between(xs, los, his, color=COLOR_MINE, alpha=0.16, linewidth=0, zorder=1)
        ax.plot(xs, rhos, color=COLOR_MINE, linewidth=2.0, zorder=4)
        ax.scatter(xs, rhos, s=26, c=COLOR_MINE, edgecolors=COLOR_SURFACE,
                   linewidths=RING_WIDTH, zorder=5)
        # A corner note, not a label on the line: the trimmed-mean curve runs along the baseline for
        # its whole length, so any inline annotation lands on top of the data.
        ax.text(0.02, 0.04, f"horizontal rule = mean baseline {baseline:+.3f}",
                transform=ax.transAxes, ha="left", va="bottom", color=COLOR_PRIMARY, fontsize=7.5)
        ax.set_xlabel(xlabel, color=COLOR_SECONDARY, fontsize=8.5)
        ax.set_title(title, color=COLOR_PRIMARY, fontsize=9.5, pad=6)
    axes[0].set_ylabel("Spearman rho vs. annotatability rank", color=COLOR_SECONDARY, fontsize=8.5)
    axes[1].invert_xaxis()      # X descending, so "more trimming" reads left-to-right
    top = header(fig, "How the sweep parameter moves the correlation",
                 "band = bootstrap 95% CI on rho (2000 resamples) - the band is wider than the "
                 "entire vertical range of either curve, which is the finding",
                 EXCLUSION_NOTE)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path, dpi=DPI, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def summary_dots(correlations, out_path):
    """All 53 statistics ranked by |rho|, with the paired CI on the difference from the mean.

    The x axis is |rho| - |rho_mean| rather than rho itself, because "does anything beat the mean"
    is the question and the paired difference is the only quantity that can answer it: each
    statistic's own CI is ~0.23 wide, far wider than the 0.11 spread between the best and worst
    entry, so plotting raw rho with marginal CIs would show 53 indistinguishable bars.
    """
    rows = sorted(correlations.values(), key=lambda r: -abs(r["rho"]))
    ys = list(range(len(rows)))[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 13.0), facecolor=COLOR_SURFACE)
    style_axes(ax)
    ax.grid(axis="x", color=COLOR_GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    ax.axvline(0.0, color=COLOR_PRIMARY, linewidth=1.3, zorder=2)

    for y, row in zip(ys, rows):
        ax.plot([row["delta_ci_lo"], row["delta_ci_hi"]], [y, y], color=COLOR_MUTED,
                linewidth=1.3, zorder=3, solid_capstyle="butt")
    for family, marker in FAMILY_MARKERS.items():
        group = [(y, r) for y, r in zip(ys, rows) if r["family"] == family]
        if not group:
            continue
        # A triangle needs more area than a disc to carry the same visual weight, because it tapers.
        ax.scatter([r["delta_vs_mean"] for _, r in group], [y for y, _ in group],
                   s=68 if marker == "^" else 52, marker=marker, c=COLOR_MINE,
                   edgecolors=COLOR_SURFACE, linewidths=RING_WIDTH, zorder=4,
                   label=f"{family} (n={len(group)})")

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r['statistic']}   {r['rho']:+.3f}" for r in rows], fontsize=7.2)
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_color(COLOR_PRIMARY if row["statistic"] == BASELINE else COLOR_SECONDARY)
        if row["statistic"] == BASELINE:
            tick.set_fontweight("bold")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("|rho| - |rho of the mean|      (right = stronger than the mean)",
                  color=COLOR_SECONDARY, fontsize=9)
    leg = ax.legend(frameon=False, fontsize=8, loc="lower right", title="Aggregation family",
                    title_fontsize=8.5, labelspacing=0.5)
    leg.get_title().set_color(COLOR_SECONDARY)
    for text in leg.get_texts():
        text.set_color(COLOR_SECONDARY)

    perm_p = next(iter(correlations.values()))["max_abs_rho_perm_p"]
    n = next(iter(correlations.values()))["n"]
    stronger = [r["statistic"] for r in rows if r["differs_from_mean"] and r["delta_vs_mean"] > 0]
    weaker = [r["statistic"] for r in rows if r["differs_from_mean"] and r["delta_vs_mean"] < 0]
    height = fig.get_figheight()
    fig.suptitle("Does any aggregation beat the mean? No.", x=0.01, ha="left",
                 color=COLOR_PRIMARY, fontsize=13, fontweight="bold", y=1.0 - 0.30 / height)
    fig.text(0.01, 1.0 - 0.55 / height,
             f"n = {n} slides - each row is one way of reducing a slide's ~324 FOV scores to "
             f"one number - bar = paired bootstrap 95% CI on the difference (2000 resamples)",
             ha="left", va="top", color=COLOR_SECONDARY, fontsize=8.5)
    fig.text(0.01, 1.0 - 0.76 / height,
             f"stronger than the mean: {', '.join(stronger) if stronger else 'none'}"
             f"    -    significantly weaker: {', '.join(weaker) if weaker else 'none'}"
             f"    -    max |rho| permutation p = {perm_p:.4f}",
             ha="left", va="top", color=COLOR_SECONDARY, fontsize=8.5)
    fig.text(0.01, 1.0 - 0.97 / height,
             f"y-axis shows each statistic's own Spearman rho - {EXCLUSION_NOTE}",
             ha="left", va="top", color=COLOR_MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 1.0 - 1.20 / height))
    fig.savefig(out_path, dpi=DPI, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def site_boxes(slides, out_path, counts):
    """Crowding score by collection site, boxes overlaid with the individual slides.

    Site comes from the slide_id prefix; the `region` column cannot do this job because all 271
    catalog slides carry the same value there (`Tanzania / Kagera`).
    """
    sites = sorted({s["site"] for s in slides})
    rng = random.Random(JITTER_SEED)
    fig, ax = plt.subplots(figsize=(8.8, 5.8), facecolor=COLOR_SURFACE)
    style_axes(ax)
    ax.grid(axis="y", color=COLOR_GRID, linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)

    groups = {site: [s for s in slides if s["site"] == site] for site in sites}
    # Boxes are context, the slides are the data: hairline, unfilled, solid (matplotlib's default
    # dashed whiskers read as "projection"), and no fliers -- every point is already drawn.
    box = ax.boxplot([[s[BASELINE] for s in groups[site]] for site in sites],
                     positions=range(1, len(sites) + 1), widths=0.52, showfliers=False,
                     patch_artist=False, zorder=2,
                     medianprops=dict(color=COLOR_PRIMARY, linewidth=1.8, solid_capstyle="butt"),
                     boxprops=dict(color=COLOR_AXIS, linewidth=1.1),
                     whiskerprops=dict(color=COLOR_AXIS, linewidth=1.1, linestyle="-"),
                     capprops=dict(color=COLOR_AXIS, linewidth=1.1))
    del box

    for i, site in enumerate(sites, start=1):
        for rank in (1, 2, 3, 4):
            group = [s for s in groups[site] if s["annotatable_rank"] == rank]
            xs = [i + rng.uniform(-0.19, 0.19) for _ in group]
            ax.scatter(xs, [s[BASELINE] for s in group], s=MARK_SIZE * 0.8,
                       c=RANK_COLORS[rank], alpha=MARK_ALPHA, edgecolors=COLOR_SURFACE,
                       linewidths=RING_WIDTH, zorder=4)

    # The per-site stats ride in the tick labels rather than as floating text: as annotations inside
    # the axes they collided with the tall outliers and the last one was clipped by the figure edge.
    labels = []
    for site in sites:
        group = groups[site]
        crowding = sorted(s[BASELINE] for s in group)
        median = crowding[len(crowding) // 2]
        within = spearman([s["annotatable_rank"] for s in group], [s[BASELINE] for s in group])
        labels.append(f"{site}\nn={len(group)}\nmedian {median:.3f}\nrho {within:+.3f}")
    ax.set_xticks(range(1, len(sites) + 1))
    ax.set_xticklabels(labels, fontsize=8.5, color=COLOR_SECONDARY)
    ax.set_xlim(0.5, len(sites) + 0.5)
    ax.set_ylabel("Crowding score (slide mean of density + Rouleaux)",
                  color=COLOR_SECONDARY, fontsize=9)
    # Clears the four-line tick labels; at -0.16 the legend title landed among them.
    rank_legend(ax, counts, loc="upper center", fontsize=7.5, ncol=4,
                bbox_to_anchor=(0.5, -0.26))

    top = header(fig, "Crowding by collection site, coloured by annotatability",
                 f"n = {len(slides)} slides - box = quartiles + 1.5 IQR whiskers, fliers suppressed "
                 f"because every slide is already plotted - rho = within-site Spearman",
                 f"KTR (n={len(groups.get('KTR', []))}) and NKR (n={len(groups.get('NKR', []))}) "
                 f"are small enough that their within-site rho is individually unreliable - "
                 f"{EXCLUSION_NOTE}")
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(out_path, dpi=DPI, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plots-dir", default=str(PLOTS_DIR))
    args = parser.parse_args()

    slides, correlations = load()
    counts = defaultdict(int)
    for slide in slides:
        counts[slide["annotatable"]] += 1
    print(f"slides: {len(slides)}  statistics: {len(correlations)}  "
          f"annotatable: {dict(counts)}")

    plots = Path(args.plots_dir)
    strip_grid(slides, correlations, WHOLE_SLIDE, 2, 3,
               plots / "permutations-whole-slide.png",
               "Whole-slide summary statistics vs. annotatability",
               "Centre (mean, median) tracks annotatability; spread (range, stdev, MAD, IQR) "
               "does not -- the four spread panels are flat.",
               (12.6, 7.4), counts)
    strip_grid(slides, correlations, [f"p{p}" for p in PERCENTILES], 4, 5,
               plots / "permutations-percentiles.png",
               "Percentile of a slide's FOV score distribution vs. annotatability",
               "A slide's pNN is the score at or above NN% of that slide's own FOV scores. "
               "Every panel looks the same, which is the point.",
               (16.0, 12.4), counts)
    strip_grid(slides, correlations,
               [n for n in statistic_names() if n.startswith("frac_")], 2, 4,
               plots / "permutations-fractions.png",
               "Fraction of a slide's FOVs at or above each bucket cut vs. annotatability",
               "The only family whose form is not a mean or a quantile. The gentlest density cut "
               "is the nominal best of all 53 statistics.",
               (16.0, 7.4), counts)
    strip_grid(slides, correlations, [f"top{pct}_mean" for pct in TRIMS], 4, 5,
               plots / "permutations-trimmed-means.png",
               "Mean of a slide's top X% scoring FOVs vs. annotatability",
               "Averaging only the most crowded FOVs. Trimming never helps -- see the sweep-curve "
               "figure for the trend these 19 panels contain.",
               (16.0, 12.4), counts)
    sweep_curves(correlations, plots / "permutations-sweep-curves.png")
    summary_dots(correlations, plots / "permutations-summary.png")
    site_boxes(slides, plots / "crowding-by-site.png", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
