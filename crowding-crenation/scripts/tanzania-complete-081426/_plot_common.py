"""Shared callout treatment for the two calibration slides, used by every figure in this dataset.

`KTR-72502948` and `KTR-72502946` are the two Tanzania slides whose 648 manually labelled FOVs make
up most of the v2.2 calibration set. Marking where they land in a cohort of 271 is the most direct
available check on the "the fit is centred on the wrong part of the distribution" reading: if the
calibration slides sit at the edge of the cohort rather than in the middle of it, that is the
mis-centring, visible rather than argued.

Two things to know when reading the callouts:

- **`KTR-72502946` is not one of the 271 catalog slides.** It is in the slide index only because
  `build_slide_index.py` adds it as a regression target, and the aggregator filters it out. It does
  have full crowding scores (the crowding pass scored all 272), so it can be placed on crowding
  axes -- but the fluorescence passes filtered to the catalog, so it has **no flag counts** and is
  absent from any figure with a flag axis. Callouts say so rather than leaving a silent gap.
- **The callout is a star, and its outline is load-bearing.** A filled yellow star reads as
  "annotation" rather than "another data point", which an unfilled ring did not — a ring looks like
  a mark in the same series. The colour is the reference palette's categorical slot 4 (`#eda100`),
  but yellow measures only **2.11:1** against the light surface (`#fcfcfb`) — under the 3:1 marker
  floor — while reaching **9.09:1** against black. So the dark outline is not decoration: it is what
  makes the star survive on a pale background, in greyscale print, and under CVD, where the fill
  alone would wash out. The label stays ink-black for the same reason.
"""
CALIBRATION_SLIDES = ("KTR-72502948", "KTR-72502946")

# In the catalog (has a slide-summary row and flag counts) vs. regression-only.
IN_CATALOG = {"KTR-72502948": True, "KTR-72502946": False}

# Reference categorical slot 4. See the docstring for why it always carries a dark edge.
STAR_COLOR = "#eda100"
STAR_EDGE = "#0b0b0b"
# Stars need noticeably more area than a disc to read at the same visual weight, because the
# points taper: at s=210 a star looks smaller than a ring of the same s.
STAR_SIZE = 340
STAR_EDGE_WIDTH = 0.9

# ANNOTATABLE is *ordinal* (1 hard .. 4 annotatable), so it gets a single-hue ramp rather than four
# categorical hues -- the question these figures ask is whether that order lines up with an
# indicator, and four hues would discard the order. Shared here so every figure colours it
# identically.
#
# The hue is magenta, deliberately **not** blue: every other figure in this dataset spends blue on
# slides, so a blue ramp here invited reading one figure's encoding into another's. Magenta is the
# furthest validated option from that blue -- min OKLab dE 24.0 against `#2a78d6`, where a purple
# ramp managed only 10.2 (purple is hue-adjacent to blue) -- while still clearing the yellow star
# callout at 17.6, both far above the dE 8 target.
#
# Validated as an ordinal ramp on the light surface (`#fcfcfb`): adjacent OKLCH dL = 0.105 / 0.135 /
# 0.151 against a 0.06 floor, and the lightest step clears 2.03:1.
ANNOTATABLE_RANK_COLORS = {1: "#e79fbd", 2: "#d4739a", 3: "#ad4470", 4: "#75213f"}


def short_label(slide_id):
    """"KTR-72502948" -> "...948" -- enough to tell the two apart without a long label."""
    return "..." + slide_id[-3:]


def highlight(ax, points, color=STAR_EDGE, fontsize=7.5, dy=0.045, size=STAR_SIZE,
              label_points=True):
    """Yellow star + black label on each (slide_id, x, y) of `ax`.

    `color` sets the label ink only; the star is always `STAR_COLOR` with a `STAR_EDGE` outline, so
    the callout looks identical in every figure regardless of what that figure's palette is doing.

    `dy` is in axes fraction so the label offset is scale-independent -- these figures have wildly
    different y ranges (0-2 for a combined score, 0-60 for a flag count).
    """
    if not points:
        return
    ax.scatter([p[1] for p in points], [p[2] for p in points], s=size, marker="*",
               c=STAR_COLOR, edgecolors=STAR_EDGE, linewidths=STAR_EDGE_WIDTH, zorder=6)
    if not label_points:
        return
    y0, y1 = ax.get_ylim()
    offset = dy * (y1 - y0)
    for slide_id, x, y in points:
        suffix = "" if IN_CATALOG.get(slide_id, True) else " (not in catalog)"
        ax.annotate(f"{short_label(slide_id)}{suffix}", xy=(x, y), xytext=(x, y + offset),
                    ha="center", va="bottom", fontsize=fontsize, color=color, zorder=7,
                    annotation_clip=False)


def calibration_means():
    """{slide_id: (density_mean, overlap_mean)} straight from the per-FOV CSVs.

    Needed because `KTR-72502946` has no `slide-summary.csv` row (not in the catalog) but was
    scored, so crowding-axis figures can place it even though the aggregator filtered it out.
    Computed from the same per-FOV scores the summary averages, so the two agree for `...948`.
    """
    from _slide_common import CROWDING_FOV_DIR, read_csv_dicts

    out = {}
    for slide_id in CALIBRATION_SLIDES:
        path = CROWDING_FOV_DIR / f"{slide_id}.csv"
        if not path.exists():
            continue
        density, overlap = [], []
        for row in read_csv_dicts(path):
            if (row.get("error") or "").strip():
                continue
            try:
                density.append(float(row["density_score"]))
                overlap.append(float(row["overlap_score"]))
            except (TypeError, ValueError):
                continue
        if density:
            out[slide_id] = (sum(density) / len(density), sum(overlap) / len(overlap))
    return out


def legend_note(missing):
    """One line for a figure's subtitle explaining any calibration slide that could not be placed."""
    if not missing:
        return "yellow stars = the two v2.2 calibration slides"
    names = ", ".join(short_label(s) for s in missing)
    return (f"yellow stars = the two v2.2 calibration slides; {names} has no flag counts "
            f"(not in the catalog, so the fluorescence passes skipped it)")
