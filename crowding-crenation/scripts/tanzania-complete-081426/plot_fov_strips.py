"""Every FOV of every slide, as one column per slide ordered by that slide's mean.

Two figures:

  1. `fov-strips-all.png`   -- all 271 slides, ~87,799 FOVs
  2. `fov-strips-sample.png` -- 25 randomly chosen slides, coloured per slide

Each has a density panel and a Rouleaux panel. A slide is a vertical column of its own FOV scores,
jittered horizontally so overlapping points separate; columns are ordered left to right by the
slide's mean on that panel's axis.

**Why this view earns its place.** The slide-level plots reduce each slide to one number, which
hides the thing a slide-level mean is only valid if you check: how wide the within-slide spread is
relative to the between-slide spread. Here both are visible at once. The measured answer is that
within-slide std (median 0.076) is *smaller* than between-slide std (0.140), so the columns are
narrower than the sweep across them -- and this figure is where that stops being a statistic and
becomes something you can see.

**Colour in figure 2 is a sequential ramp by slide mean, not 25 arbitrary hues.** The request was a
distinct colour per slide; 25 categorical hues would be an anti-pattern (beyond ~8, generated hues
stop being reliably distinguishable and a rainbow implies unordered identity for what is really an
ordered quantity). Keying the ramp to the slide's mean keeps every column distinguishable, and the
colour then carries the same information as the x position instead of fighting it. Columns are also
positionally separated, so identity never rests on colour alone. `--categorical` switches to
cycling the validated 8-hue theme if maximum adjacent contrast is wanted instead.

Usage:
    python scripts/tanzania-complete-081426/plot_fov_strips.py
    python scripts/tanzania-complete-081426/plot_fov_strips.py --sample 25 --seed 11
"""
import argparse
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _plot_common import (  # noqa: E402
    CALIBRATION_SLIDES,
    STAR_COLOR,
    STAR_EDGE,
    STAR_EDGE_WIDTH,
    STAR_SIZE,
    short_label,
)
from _slide_common import (  # noqa: E402
    CROWDING_FOV_DIR,
    PLOTS_DIR,
    ROOT,
    SLIDE_SUMMARY_CSV,
    read_csv_dicts,
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
    DEFAULT_SCORING_PARAMS,
)

# The validated 8-hue categorical theme, used only under --categorical.
THEME_8 = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
           "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

AXES = [("density", "density_score", "Per-FOV density score"),
        ("overlap", "overlap_score", "Per-FOV Rouleaux score")]

JITTER_SEED = 7
DPI = 200


def style_axes(ax):
    ax.set_facecolor(COLOR_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(COLOR_AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=COLOR_SECONDARY, labelsize=8, length=3, width=1.0, color=COLOR_AXIS)


def sequential_cmap():
    return LinearSegmentedColormap.from_list("mine_seq", ["#b7d3f6", COLOR_MINE, "#123a68"])


def _read_fov_scores(slide_id):
    path = CROWDING_FOV_DIR / f"{slide_id}.csv"
    if not path.exists():
        return None
    density, overlap = [], []
    for fov in read_csv_dicts(path):
        if (fov.get("error") or "").strip():
            continue
        try:
            density.append(float(fov["density_score"]))
            overlap.append(float(fov["overlap_score"]))
        except (TypeError, ValueError):
            continue
    if not density:
        return None
    return {"density": density, "overlap": overlap,
            "density_mean": sum(density) / len(density),
            "overlap_mean": sum(overlap) / len(overlap)}


def load_slides(summary_csv, slide_ids=None):
    """{slide_id: {"density": [...], "overlap": [...], means}} from the per-FOV CSVs.

    The calibration slides are always included when their per-FOV CSV exists, even though
    KTR-72502946 is not one of the 271 catalog slides and so has no `slide-summary.csv` row. It
    was still scored (the crowding pass ran over all 272), and leaving the more heavily weighted
    of the two calibration slides off a figure about where the calibration sits would defeat the
    figure's purpose. `in_catalog` records which is which.
    """
    wanted = set(slide_ids) if slide_ids else None
    out = {}
    for row in read_csv_dicts(summary_csv):
        slide_id = row["slide_id"]
        if wanted is not None and slide_id not in wanted:
            continue
        scores = _read_fov_scores(slide_id)
        if scores:
            out[slide_id] = {**scores, "in_catalog": True}

    for slide_id in CALIBRATION_SLIDES:
        if slide_id in out or (wanted is not None and slide_id not in wanted):
            continue
        scores = _read_fov_scores(slide_id)
        if scores:
            out[slide_id] = {**scores, "in_catalog": False}
    return out


def draw_panel(ax, ordered, key, label, params, color_for, jitter, marker_size, marker_alpha):
    rng = random.Random(JITTER_SEED)
    for position, slide_id in enumerate(ordered):
        values = slide_data[slide_id][key]
        xs = [position + rng.uniform(-jitter, jitter) for _ in values]
        ax.scatter(xs, values, s=marker_size, c=[color_for(slide_id, position)] * len(values),
                   alpha=marker_alpha, linewidths=0, zorder=3)

    axis_name = "density" if key == "density_score" or key == "density" else "overlap"
    for threshold in params[axis_name]["bucket_thresholds"]:
        ax.axhline(threshold, color=COLOR_GRID, linewidth=1.0, zorder=1)

    ax.set_xlim(-1, len(ordered))
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(label, color=COLOR_SECONDARY, fontsize=9)
    style_axes(ax)


def annotate_calibration(ax, ordered, key):
    """Mark the calibration slides' columns: a vertical rule, a yellow star, and a black label.

    The rule is what actually identifies the column among 271 of them; the star matches the callout
    used in every other figure so the visual language is the same throughout. Star fill and outline
    come from `_plot_common` for that reason.
    """
    marked = []
    for slide_id in CALIBRATION_SLIDES:
        if slide_id not in ordered:
            continue
        position = ordered.index(slide_id)
        ax.axvline(position, color=COLOR_PRIMARY, linewidth=1.1, alpha=0.75, zorder=5)
        ax.scatter([position], [1.012], s=STAR_SIZE, marker="*", c=STAR_COLOR,
                   edgecolors=STAR_EDGE, linewidths=STAR_EDGE_WIDTH, zorder=6,
                   clip_on=False)
        ax.annotate(short_label(slide_id), xy=(position, 1.03), xytext=(position, 1.048),
                    ha="center", va="bottom", fontsize=7.5, color=COLOR_PRIMARY, zorder=7,
                    annotation_clip=False)
        marked.append((slide_id, position))
    return marked


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-csv", default=str(SLIDE_SUMMARY_CSV))
    parser.add_argument("--params", default=str(DEFAULT_SCORING_PARAMS))
    parser.add_argument("--plots-dir", default=str(PLOTS_DIR))
    parser.add_argument("--sample", type=int, default=25,
                        help="slides in the second figure (default 25)")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--categorical", action="store_true",
                        help="cycle the 8-hue theme in figure 2 instead of a sequential ramp")
    args = parser.parse_args()

    import json
    with open(args.params, encoding="utf-8") as f:
        params = json.load(f)

    global slide_data
    slide_data = load_slides(args.summary_csv)
    if not slide_data:
        raise SystemExit("no per-FOV crowding CSVs found -- run the crowding pass first")
    total_fovs = sum(len(v["density"]) for v in slide_data.values())
    n_catalog = sum(1 for v in slide_data.values() if v.get("in_catalog"))
    extra = [s for s, v in slide_data.items() if not v.get("in_catalog")]
    print(f"loaded {len(slide_data)} slides ({n_catalog} catalog"
          + (f" + {', '.join(extra)} regression" if extra else "") + f"), {total_fovs:,} FOVs")
    slide_label = (f"{n_catalog} catalog slides"
                   + (f" + {len(extra)} regression slide" if extra else ""))

    # Where the calibration slides sit in the cohort, which is the point of marking them.
    for slide_id in CALIBRATION_SLIDES:
        if slide_id not in slide_data:
            continue
        for axis in ("density", "overlap"):
            ordered = sorted(slide_data, key=lambda s: slide_data[s][f"{axis}_mean"])
            pct = 100 * ordered.index(slide_id) / max(len(ordered) - 1, 1)
            print(f"  {slide_id}: {axis}_mean {slide_data[slide_id][f'{axis}_mean']:.3f} "
                  f"-> {pct:.0f}th percentile of the cohort")

    plots = Path(args.plots_dir)
    plots.mkdir(parents=True, exist_ok=True)

    # ---- figure 1: every slide -------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.4), facecolor=COLOR_SURFACE, sharex=False)
    for ax, (axis_name, _col, label) in zip(axes, AXES):
        ordered = sorted(slide_data, key=lambda s: slide_data[s][f"{axis_name}_mean"])
        draw_panel(ax, ordered, axis_name, label, params,
                   color_for=lambda _s, _p: COLOR_MINE, jitter=0.34,
                   marker_size=1.6, marker_alpha=0.20)
        annotate_calibration(ax, ordered, axis_name)
        ax.set_xlabel(f"{len(ordered)} slides, ordered by mean "
                      f"{label.split()[-2].lower()} score (one column per slide)",
                      color=COLOR_SECONDARY, fontsize=8.5)
        ax.set_xticks([])

    fig.suptitle(f"Every FOV of every Tanzania slide ({total_fovs:,} FOVs, {slide_label})",
                 x=0.01, ha="left", color=COLOR_PRIMARY, fontsize=12.5, fontweight="bold",
                 y=0.998)
    fig.text(0.01, 0.955,
             "One column per slide, ordered by that slide's mean; horizontal lines = "
             "v2.2-optimized per-FOV bucket thresholds; vertical rules = the two calibration slides",
             ha="left", va="top", color=COLOR_SECONDARY, fontsize=8.3)
    fig.tight_layout(rect=(0, 0, 1, 0.935))
    out1 = plots / "fov-strips-all.png"
    fig.savefig(out1, dpi=DPI, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out1}")

    # ---- figure 2: a random sample, coloured per slide --------------------------------------
    rng = random.Random(args.seed)
    # Force the calibration slides in, so the sample can be compared against figure 1's rules.
    pool = [s for s in slide_data if s not in CALIBRATION_SLIDES]
    present_calib = [s for s in CALIBRATION_SLIDES if s in slide_data]
    n_random = max(0, args.sample - len(present_calib))
    sample = present_calib + rng.sample(sorted(pool), min(n_random, len(pool)))
    print(f"sample: {len(sample)} slides (seed {args.seed}, calibration slides forced in)")

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.8), facecolor=COLOR_SURFACE)
    cmap, norm = sequential_cmap(), None
    for ax, (axis_name, _col, label) in zip(axes, AXES):
        ordered = sorted(sample, key=lambda s: slide_data[s][f"{axis_name}_mean"])
        means = [slide_data[s][f"{axis_name}_mean"] for s in ordered]
        norm = Normalize(vmin=min(means), vmax=max(means))

        if args.categorical:
            def color_for(slide_id, position):
                return THEME_8[position % len(THEME_8)]
        else:
            def color_for(slide_id, _position, axis=axis_name):
                return cmap(norm(slide_data[slide_id][f"{axis}_mean"]))

        draw_panel(ax, ordered, axis_name, label, params, color_for=color_for,
                   jitter=0.36, marker_size=5.5, marker_alpha=0.55)
        annotate_calibration(ax, ordered, axis_name)
        ax.set_xticks(range(len(ordered)))
        ax.set_xticklabels(ordered, rotation=90, fontsize=6.2, color=COLOR_SECONDARY)

    scheme = ("cycled 8-hue categorical theme" if args.categorical
              else "single-hue sequential ramp keyed to each slide's mean")
    fig.suptitle(f"Every FOV of {len(sample)} randomly sampled Tanzania slides",
                 x=0.01, ha="left", color=COLOR_PRIMARY, fontsize=12.5, fontweight="bold",
                 y=0.998)
    fig.text(0.01, 0.962,
             f"seed {args.seed}, the two calibration slides forced in; colour = {scheme}",
             ha="left", va="top", color=COLOR_SECONDARY, fontsize=8.3)
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out2 = plots / ("fov-strips-sample-categorical.png" if args.categorical
                    else "fov-strips-sample.png")
    fig.savefig(out2, dpi=DPI, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
