"""Score the ziba-test methanol-evaporation set (24 FOVs) with the calibrated v2.2
density/Rouleaux fit and place every FOV on the density x Rouleaux grid, coloured by
test group.

Nothing is recalibrated here: this is inference plus an out-of-distribution check. There
are no manual labels for this set (no data/labels/ziba-test/), so the grid carries one
point group -- the model's prediction -- rather than the manual-vs-model pair
scripts/nigeria_081226.py draws.

The dataset is a 6 x 4 protocol sweep, not a slide cohort: six methanol volumes
(100/150/200/250/500 uL + a Coplin jar dip) x four fixation/stain conditions (RT vs.
37+Hum incubation, each with and without DAPI). One FOV per cell. So the question the
grid answers is "does fixation condition move where a smear lands on the quality grid", so
each mark carries its test group (see the colour note below) and is direct-labelled with its
methanol volume -- at n=24 a single FOV can then be traced back to its protocol cell, which
is most of the value of plotting a set this small.

The OOD check is not an aside. scripts/combined/README.md's "Known limitation" is that
every v2 feature is a raw pixel/intensity statistic sensitive to stain/scanner/
illumination, and that a new preparation should be spot-checked before its scores are
trusted -- and a methanol-evaporation sweep is exactly a preparation change. Each FOV's
features are compared against the per-feature 2nd-98th percentile normalization range in
density_overlap_v2.2_params.json (the range the composite clips to), so a feature outside
that band is one the composite cannot resolve: it is pinned at 0.0 or 1.0 by construction.

Colour is a blue/red theme rather than four unrelated hues, because the four test groups are
not four arbitrary categories -- they are a 2x2: incubation (RT vs. 37+Hum) x DAPI. So the
encoding is composite and mirrors that structure:

    hue        incubation   blue = RT (cool), red = 37+Hum (warm)
    lightness  DAPI         the darker step of each hue family is the DAPI arm
    shape      DAPI         circle = no DAPI, triangle = DAPI

which means every pair of groups differs in at least two channels, and the pair that shares
a hue (the within-family DAPI contrast) is separated by lightness *and* shape rather than by
colour alone.

Three of the four hexes are documented palette steps: #3987e5 and #184f95 are blue ramp steps
400 and 600, #e34948 is categorical slot 8. Only the dark red is derived -- the reference
palette documents no red *ramp*, so #9e0017 was stepped from slot 8 in OKLCH (hue held at
24.9, lightness walked down, chroma taken as large as stays in gamut), which is the
skill's documented snap-to-passing procedure. The set was then validated on the six checks at
`--pairs all` (the right pairlist for a scatter, where any two marks can end up adjacent):
every mark sits in the light-mode lightness band with chroma >= 0.10 and >= 3.54:1 contrast
against the #fcfcfb surface, worst normal-vision dE 18.2 against a floor of 15, and worst
CVD dE 18.2 under Machado-Oliveira-Fernandes protanopia/deuteranopia at severity 1.0 against
a target of 8. (Checked with a Python port of the skill's validator, reproducing its
documented reference numbers exactly -- 19.6/9.1 adjacent on the default eight and 24.0/9.2
all-pairs on the first three -- because node is not installed in this environment.)

Outputs (data/results/ziba-test/, <suffix> from --suffix, default "v2.2"):
    features-<suffix>.csv          raw feature vector + composite scores + buckets, per FOV
    ood-report-<suffix>.csv        per-FOV x per-feature normalization-range position
    quality-grid-<suffix>.png      the 24 FOVs on the density x Rouleaux grid, by test group
    combined-score-by-group-<suffix>.png   combined severity, one box per test group

Usage:
    python scripts/ziba_test.py [--workers N] [--params PATH] [--suffix NAME]
"""
import argparse
import csv
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

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
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    DENSITY_LEVELS,
    OVERLAP_LEVELS,
    apply_label_overrides,
    blur_downsample_from_params,
    compute_features,
    density_ordinal,
    display_level,
    empty_field_fired,
    lbp_step_from_params,
    list_image_paths,
    load_image,
    overlap_ordinal,
)
from src.composite_v2 import bucket, weighted_composite  # noqa: E402

IMAGE_DIR = ROOT / "data" / "raw" / "ziba-test"
V2_PARAMS = ROOT / "data" / "results" / "density-rouleaux-v2" / "density_overlap_v2.2_params.json"
OUT_DIR = ROOT / "data" / "results" / "ziba-test"

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

# The four test groups. Kept as an explicit list rather than derived from the filenames so a
# missing condition shows up as a dropped legend entry instead of silently reassigning the
# colours of the others -- colour follows the group, never its position in the data.
GROUPS = ["RT", "RT DAPI", "37+Hum", "37+Hum DAPI"]
# hue = incubation, lightness = DAPI. See the module docstring for the validation numbers.
GROUP_COLORS = {
    "RT": "#3987e5",           # blue ramp 400
    "RT DAPI": "#184f95",      # blue ramp 600
    "37+Hum": "#e34948",       # categorical slot 8
    "37+Hum DAPI": "#9e0017",  # slot 8 stepped down in OKLCH (no red ramp is documented)
}
# shape repeats the DAPI split, so the two same-hue groups are never separated by colour alone
GROUP_MARKERS = {"RT": "o", "RT DAPI": "^", "37+Hum": "o", "37+Hum DAPI": "^"}
# a surface-coloured ring, so two marks packed into the same grid cell stay visually separate
MARK_EDGE = COLOR_SURFACE

# volume order along the sweep; Coplin is a jar dip rather than a pipetted volume, so it sorts last
VOLUME_ORDER = ["100uL", "150uL", "200uL", "250uL", "500uL", "Coplin"]


def parse_condition(filename):
    """'20260811_Methanol_Evaporation_250uL_37+Hum_DAPI.png' -> ('250uL', '37+Hum DAPI').

    Layout is <date>_Methanol_Evaporation_<volume>_<incubation>[_DAPI], so the volume and
    incubation tokens are positional and DAPI is an optional trailing token.
    """
    parts = Path(filename).stem.split("_")
    dapi = parts[-1] == "DAPI"
    incubation = parts[-2] if dapi else parts[-1]
    volume = parts[-3] if dapi else parts[-2]
    group = incubation + " DAPI" if dapi else incubation
    if group not in GROUP_COLORS:
        raise ValueError("unrecognized test group " + repr(group) + " from " + repr(filename))
    if volume not in VOLUME_ORDER:
        raise ValueError("unrecognized methanol volume " + repr(volume) + " from " + repr(filename))
    return volume, group


def short_volume(volume):
    """'250uL' -> '250', 'Coplin' -> 'Cop' -- short enough to sit beside a packed marker."""
    return "Cop" if volume == "Coplin" else volume.replace("uL", "")


# --- scoring -----------------------------------------------------------------------------

def _extract_one(args):
    path, lbp_step, blur_downsample = args
    features = compute_features(load_image(path), lbp_step=lbp_step,
                                blur_downsample=blur_downsample)
    features["filename"] = Path(str(path)).name
    return features


def score_axis(features, axis_params):
    ranges = {n: (v["min"], v["max"]) for n, v in axis_params["normalization"].items()}
    score = weighted_composite(features, axis_params["weights"], ranges)
    return score, bucket(score, axis_params["bucket_thresholds"], axis_params["bucket_labels"])


def score_all(paths, params, workers):
    # Both runtime knobs come from the params file, so the features are computed the way the
    # fit being scored against was built (see lbp_step_from_params).
    jobs = [(p, lbp_step_from_params(params), blur_downsample_from_params(params)) for p in paths]
    if workers > 1:
        with Pool(workers) as pool:
            rows = pool.map(_extract_one, jobs)
    else:
        rows = [_extract_one(j) for j in jobs]

    for r in rows:
        r["volume"], r["group"] = parse_condition(r["filename"])
        r["density_score"], raw_density = score_axis(r, params["density"])
        r["overlap_score"], raw_overlap = score_axis(r, params["overlap"])
        # keep the pre-override buckets: on a gated FOV the composite score printed next to a
        # label would otherwise contradict it with no way to see why
        r["raw_density_label"], r["raw_overlap_label"] = raw_density, raw_overlap
        r["density_label"], r["overlap_label"] = apply_label_overrides(
            raw_density, raw_overlap, r, params)
        r["gated"] = empty_field_fired(r, params.get("empty_field_override"))
        r["combined_score"] = combined_score(r)

    rows.sort(key=lambda r: (GROUPS.index(r["group"]), VOLUME_ORDER.index(r["volume"])))
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
    """Where `value` sits in [lo, hi], as the fraction the composite uses before clipping.
    <0 or >1 means the feature is clipped, i.e. the composite cannot distinguish this FOV
    from any other beyond that range edge."""
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
                "group": r["group"],
                "volume": r["volume"],
                "feature": name,
                "value": round(r[name], 6),
                "calib_p2": round(lo, 6),
                "calib_p98": round(hi, 6),
                "range_position": round(pos, 4),
                "clipped": "yes" if (pos < 0 or pos > 1) else "no",
            })
    return out


# --- the grid -----------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(COLOR_SURFACE)
    ax.tick_params(colors=COLOR_MUTED, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLOR_AXIS)
    ax.set_axisbelow(True)


def cell_offsets(count, ncols=2, dx=0.34, dy=0.16):
    """Deterministic packing of `count` marks inside one grid cell.

    Random jitter (what the manual-vs-model grids use, where the *pair* of points per FOV is
    what matters) would overlap marks and scramble the reading order here: with one FOV per
    protocol cell and a volume label on every mark, the useful thing is a stable lattice in
    sweep order, so a cell holding several FOVs reads as a list rather than a cloud.
    """
    nrows = math.ceil(count / ncols) if count else 0
    out = []
    for i in range(count):
        row, col = divmod(i, ncols)
        n_in_row = min(ncols, count - row * ncols)
        out.append(((col - (n_in_row - 1) / 2) * dx,
                    ((nrows - 1) / 2 - row) * dy))
    return out


def plot_quality_grid(rows, out_path, params_name, gated):
    fig, ax = plt.subplots(figsize=(11, 7.2), dpi=150)
    fig.patch.set_facecolor(COLOR_SURFACE)
    _style(ax)

    # group the FOVs by the grid cell they landed in, then pack each cell
    cells = {}
    for r in rows:
        cells.setdefault((overlap_ordinal(r["overlap_label"]),
                          density_ordinal(r["density_label"])), []).append(r)

    plotted = {g: False for g in GROUPS}
    for (x, y), members in cells.items():
        for r, (ox, oy) in zip(members, cell_offsets(len(members))):
            px, py = x + ox, y + oy
            ax.scatter([px], [py], s=95, marker=GROUP_MARKERS[r["group"]],
                       color=GROUP_COLORS[r["group"]], edgecolors=MARK_EDGE, linewidths=1.0,
                       zorder=3)
            ax.text(px + 0.055, py, short_volume(r["volume"]), fontsize=7.5, va="center",
                    ha="left", color=COLOR_SECONDARY, zorder=4)
            plotted[r["group"]] = True

    ax.set_xticks(range(len(OVERLAP_LEVELS)))
    ax.set_xticklabels([display_level(l) for l in OVERLAP_LEVELS], fontsize=10, rotation=15,
                       ha="right")
    ax.set_xlim(-0.5, len(OVERLAP_LEVELS) - 0.5)
    ax.set_yticks(range(len(DENSITY_LEVELS)))
    ax.set_yticklabels([display_level(l) for l in DENSITY_LEVELS], fontsize=10)
    ax.set_ylim(-0.5, len(DENSITY_LEVELS) - 0.5)
    for i in range(1, len(OVERLAP_LEVELS)):
        ax.axvline(i - 0.5, color=COLOR_GRID, linewidth=1, zorder=0)
    for i in range(1, len(DENSITY_LEVELS)):
        ax.axhline(i - 0.5, color=COLOR_GRID, linewidth=1, zorder=0)

    ax.set_xlabel("Rouleaux level", color=COLOR_SECONDARY, fontsize=11)
    ax.set_ylabel("Density level", color=COLOR_SECONDARY, fontsize=11)

    handles = [Line2D([0], [0], marker=GROUP_MARKERS[g], color="none",
                      markerfacecolor=GROUP_COLORS[g], markeredgecolor=MARK_EDGE,
                      markeredgewidth=1.0, markersize=9, label=g)
               for g in GROUPS if plotted[g]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02),
               ncol=len(handles), frameon=False, fontsize=9.5, labelcolor=COLOR_SECONDARY)

    note = "; " + str(len(gated)) + " via the empty-field gate" if gated else ""
    fig.suptitle("ziba-test methanol evaporation -- v2.2 predicted density x Rouleaux",
                 x=0.02, y=0.97, ha="left", color=COLOR_PRIMARY, fontsize=14,
                 fontweight="bold")
    ax.set_title("n = " + str(len(rows)) + " FOVs (6 methanol volumes x 4 test groups), "
                 "landing in " + str(len(cells)) + " of 25 cells" + note + "\n"
                 "model prediction only -- no manual labels for this set; "
                 "point label = methanol volume (" + params_name + ")",
                 loc="left", color=COLOR_SECONDARY, fontsize=10, pad=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(out_path, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)


def combined_score(row):
    """One severity number per FOV: the mean of the two composite scores.

    Defensible here specifically because the two axes are not independent on this dataset --
    they correlate at r = 0.994 / rho = 0.987 across the 24 FOVs (the Rouleaux axis's six
    features are all density features too), so their mean loses almost nothing and the
    alternative, plotting two near-identical panels, would imply two readings where there is
    one. On a set where the axes actually separate, box-plot them separately instead.
    """
    return (row["density_score"] + row["overlap_score"]) / 2.0


def plot_combined_boxplot(rows, out_path, params_name):
    """Combined severity by test group: a box per group with all six of its FOVs overlaid.

    The points are not decoration. At n = 6 per group a box's quartiles rest on one or two
    observations each, so the box alone would overstate how well-determined the spread is --
    showing every FOV, labelled with its methanol volume, is what makes the box honest and is
    also the only way to see that the group spreads are wildly unequal (37+Hum spans 0.13,
    37+Hum DAPI spans 0.59).

    They are also what lets the +/- DAPI pairing be drawn, and that pairing is the figure's
    main finding. Each volume's DAPI and non-DAPI FOV are two fields of the *same slide*, so
    the four boxes are not four independent samples -- the two blue boxes describe one set of
    six slides and the two red boxes another. The connectors say so, and they show that the
    paired same-slide disagreement (mean 0.217 on density) is larger than the RT-vs-37+Hum
    group difference (0.119) the boxes appear to establish.
    """
    fig, ax = plt.subplots(figsize=(9.5, 6.4), dpi=150)
    fig.patch.set_facecolor(COLOR_SURFACE)
    _style(ax)
    ax.grid(True, axis="y", color=COLOR_GRID, linewidth=0.8, zorder=0)

    by_group = {g: [combined_score(r) for r in rows if r["group"] == g] for g in GROUPS}
    present = [g for g in GROUPS if by_group[g]]
    positions = list(range(len(present)))

    bp = ax.boxplot([by_group[g] for g in present], positions=positions, widths=0.5,
                    patch_artist=True, showfliers=False, zorder=2)
    for g, box in zip(present, bp["boxes"]):
        box.set_facecolor(GROUP_COLORS[g])
        box.set_alpha(0.16)
        box.set_edgecolor(GROUP_COLORS[g])
        box.set_linewidth(1.4)
    for g, median in zip(present, bp["medians"]):
        median.set_color(GROUP_COLORS[g])
        median.set_linewidth(2.2)
    for part in ("whiskers", "caps"):
        for artist in bp[part]:
            artist.set_color(COLOR_AXIS)
            artist.set_linewidth(1.0)

    # Every FOV on top of its box, offset to one side so it never hides the median. Points are
    # walked in score order and pushed a column further right whenever the previous one is
    # within LABEL_MIN_GAP -- offsetting by volume order instead collided exactly where it
    # matters, since the FOVs that need separating are the ones close in *score*, which are
    # not the ones adjacent in the sweep.
    LABEL_MIN_GAP = 0.035
    coords = {}
    for x, g in zip(positions, present):
        members = sorted((r for r in rows if r["group"] == g), key=combined_score)
        column = 0
        last = None
        for r in members:
            py = combined_score(r)
            column = column + 1 if last is not None and py - last < LABEL_MIN_GAP else 0
            last = py
            px = x + 0.20 + column * 0.17
            coords[(g, r["volume"])] = (px, py)
            ax.scatter([px], [py], s=58, marker=GROUP_MARKERS[g], color=GROUP_COLORS[g],
                       edgecolors=MARK_EDGE, linewidths=0.9, zorder=4)
            ax.text(px + 0.045, py, short_volume(r["volume"]), fontsize=7.5, va="center",
                    ha="left", color=COLOR_SECONDARY, zorder=5)

    # The +/- DAPI pair at one volume is the SAME SLIDE, so its two marks describe one smear and
    # the connector between them is v2.2 disagreeing with itself -- not a group difference. That
    # makes the connectors the most load-bearing marks here: a steep one is measurement error
    # several buckets wide. Drawn at zorder 1, under the points they comment on.
    n_paired = 0
    for incubation in ("RT", "37+Hum"):
        for volume in VOLUME_ORDER:
            a, b = coords.get((incubation, volume)), coords.get((incubation + " DAPI", volume))
            if not (a and b):
                continue
            ax.plot([a[0], b[0]], [a[1], b[1]], color=COLOR_AXIS, linewidth=0.9, alpha=0.85,
                    zorder=1)
            n_paired += 1

    ax.set_xticks(positions)
    ax.set_xticklabels(present, fontsize=10.5, color=COLOR_SECONDARY)
    ax.set_xlim(-0.55, len(present) - 0.05)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Combined severity score (mean of density and Rouleaux)",
                  color=COLOR_SECONDARY, fontsize=10.5)
    ax.set_xlabel("Test group", color=COLOR_SECONDARY, fontsize=11)

    fig.suptitle("ziba-test methanol evaporation -- combined v2.2 severity by test group",
                 x=0.02, y=0.975, ha="left", color=COLOR_PRIMARY, fontsize=14,
                 fontweight="bold")
    ax.set_title("box = median and quartiles, whiskers 1.5 x IQR; all n = 6 FOVs per group "
                 "plotted and labelled with methanol volume\n"
                 "connectors join the " + str(n_paired) + " +/- DAPI pairs, which are the same "
                 "slide -- a steep connector is v2.2 disagreeing with itself, not a density "
                 "difference\n"
                 "no bucket lines: the two axes have different thresholds, so their mean has "
                 "none (" + params_name + ")",
                 loc="left", color=COLOR_SECONDARY, fontsize=9.5, pad=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, facecolor=COLOR_SURFACE, bbox_inches="tight")
    plt.close(fig)


# --- main ---------------------------------------------------------------------------------

def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Score the ziba-test set with v2.2.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--params", default=str(V2_PARAMS),
                        help="params JSON to score with (default: full-resolution v2.2)")
    parser.add_argument("--suffix", default="v2.2", help="suffix for every output filename")
    args = parser.parse_args()

    params_path = Path(args.params)
    with open(params_path) as f:
        params = json.load(f)

    paths = list_image_paths(IMAGE_DIR)
    rows = score_all(paths, params, args.workers)
    ranges = normalization_ranges(params)
    gated = [r for r in rows if r["gated"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / ("features-" + args.suffix + ".csv"),
              ["filename", "group", "volume"] + FEATURE_ORDER
              + ["otsu_threshold", "density_score", "density_label", "overlap_score",
                 "overlap_label", "combined_score", "raw_density_label",
                 "raw_overlap_label", "gated"],
              rows)
    write_csv(OUT_DIR / ("ood-report-" + args.suffix + ".csv"),
              ["filename", "group", "volume", "feature", "value", "calib_p2", "calib_p98",
               "range_position", "clipped"],
              build_ood_rows(rows, ranges))
    params_name = params_path.stem.replace("density_overlap_", "").replace("_params", "")
    plot_quality_grid(rows, OUT_DIR / ("quality-grid-" + args.suffix + ".png"),
                      params_name, gated)
    plot_combined_boxplot(rows, OUT_DIR / ("combined-score-by-group-" + args.suffix + ".png"),
                          params_name)

    # --- console summary ---
    print("n=" + str(len(rows)) + " FOVs from " + IMAGE_DIR.name
          + ", scored with " + params_path.name + "\n")
    for r in rows:
        mark = "  [gated]" if r["gated"] else ""
        print("  {:<12} {:<7} {:<14} ({:.3f})  {:<16} ({:.3f}){}".format(
            r["group"], r["volume"], display_level(r["density_label"]), r["density_score"],
            display_level(r["overlap_label"]), r["overlap_score"], mark))

    print("\nPredicted bucket counts:")
    for axis, levels in (("density_label", DENSITY_LEVELS), ("overlap_label", OVERLAP_LEVELS)):
        counts = {l: sum(1 for r in rows if r[axis] == l) for l in levels}
        print("  {:<14} ".format(axis)
              + ", ".join(display_level(l) + "=" + str(c) for l, c in counts.items() if c))

    for title, key, order in (("test group", "group", GROUPS),
                              ("methanol volume", "volume", VOLUME_ORDER)):
        print("\nMean composite score by " + title + ":")
        for value in order:
            members = [r for r in rows if r[key] == value]
            if not members:
                continue
            print("  {:<12} n={}  density {:.3f}   Rouleaux {:.3f}".format(
                value, len(members),
                float(np.mean([r["density_score"] for r in members])),
                float(np.mean([r["overlap_score"] for r in members]))))

    print("\nOut-of-range features (value outside the 2nd-98th percentile calibration band):")
    for name in FEATURE_ORDER:
        if name not in ranges:
            print("  {:<24} not in this fit -- no calibration band".format(name))
            continue
        lo, hi = ranges[name]
        vals = np.array([r[name] for r in rows])
        n_over, n_under = int((vals > hi).sum()), int((vals < lo).sum())
        print("  {:<24} {}/{} clipped, band [{:.4f}, {:.4f}], observed [{:.4f}, {:.4f}]".format(
            name, n_over + n_under, len(vals), lo, hi, vals.min(), vals.max()))
        if n_over:
            print("  {:<24}   - {} above p98 (max {:.4f} = {:.2f}x edge)".format(
                "", n_over, vals.max(), vals.max() / hi))
        if n_under:
            print("  {:<24}   - {} below p2 (min {:.4f})".format("", n_under, vals.min()))

    if gated:
        print("\nEmpty-field gate fired on {}/{} FOVs:".format(len(gated), len(rows)))
        for r in gated:
            print("  {:<52} composite said ({}, {}) -> forced ({}, {})".format(
                r["filename"], r["raw_density_label"], r["raw_overlap_label"],
                r["density_label"], r["overlap_label"]))

    print("\nMean combined severity by test group (the box plot's y axis):")
    for g in GROUPS:
        members = [r for r in rows if r["group"] == g]
        if not members:
            continue
        vals = np.array([r["combined_score"] for r in members])
        print("  {:<12} n={}  mean {:.3f}  median {:.3f}  range {:.3f}-{:.3f}".format(
            g, len(members), vals.mean(), float(np.median(vals)), vals.min(), vals.max()))

    print("\nwrote 2 CSVs + 2 plots to " + str(OUT_DIR)
          + " with suffix '" + args.suffix + "'")


if __name__ == "__main__":
    main()
