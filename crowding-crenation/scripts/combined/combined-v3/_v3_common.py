"""v3's label vocabulary: the density axis gains two rungs at the bottom.

Everything else -- `compute_features`, the params-JSON knobs, the overlap vocabulary, the
scoring math -- is imported unchanged from `scripts/combined/_v2_common.py`, so v3 is a
controlled comparison against v2.2 rather than a rewrite.

## Why two new levels

v2.2 handles near-empty fields with a hard pre-filter (`apply_empty_field_override`): when all
four of `otsu_separability`, `lbp_entropy`, `glcm_contrast` and `edge_density_unmasked` fall
below their calibration p2 floor, the composites are discarded and the FOV is forced to
`sparser` + `no rouleaux`. Two problems, both measured:

  * **The gate conflates two different fields.** Across the eight v3 slides it fires on 317
    FOVs, and they are not one population. `KIT-62500670` fov198 is genuinely blank -- flat
    grey, a few specks of debris, nothing to judge. `RUB-72501818` fov107 has hundreds of
    countable, well-separated cells and is perfectly judgeable. Both are forced to `sparser`.
  * **It has never been checked against a label**, and cannot be while it is also the thing
    that selects which FOVs get called empty. `check_empty_field_gate.py` can only assert that
    gated FOVs were already predicted `sparser`, which is circular.

So v3 annotates the distinction instead of asserting it, and the gate becomes a *prediction*
(the bottom rung of the ordinal) rather than a pre-filter.

## Why an ordinal extension rather than a separate axis

`no cells` -> `few cells` -> `sparser` is monotone in the same underlying quantity the density
axis already measures, so a 7-level ordinal represents it without a second model head, and PAVA
derives its cut points the same way it does every other boundary.

**Coverage cannot do this job.** On a flat field Otsu's threshold is arbitrary, so a blank FOV
lands at either end of the coverage range depending on which side of the sensor noise the
threshold falls -- fov198 above reads `coverage=0.9995`. Measured over the eight v3 slides, the
gated FOVs' coverage is bimodal with *nothing* between 0.10 and 0.20: 59 below 0.02, 102 in
0.05-0.10, then 85 above 0.95. Whatever separates the bottom rungs has to be textural, which is
the one part of the old gate's design that was right.

## Migration from the 5-level vocabulary

The 646 existing annotations are unaffected: levels are looked up **by name**, never by index,
so an old `sparser` stays `sparser` and simply takes ordinal 2 instead of 0. No relabelling of
the existing pool is required, and `density_ordinal` below is the only place the mapping lives.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent
sys.path.insert(0, str(V2_DIR))

from _v2_common import (  # noqa: E402,F401 -- re-exported so v3 code has one import site
    DEFAULT_OVERLAP_LABEL,
    OVERLAP_LEVELS,
    OVERLAP_TAGS,
    TILE_GLCM_LEVELS,
    TILE_GRID_SIZE,
    blur_downsample_from_params,
    compute_features,
    lbp_step_from_params,
    load_image,
    overlap_ordinal,
)

# The two new rungs sit below `sparser`; the upper five are v2.2's, unchanged and in order.
DENSITY_LEVELS = ["no cells", "few cells", "sparser", "monolayer", "slightly dense",
                  "dense", "very dense"]
DEFAULT_DENSITY_LABEL = "monolayer"

AXIS_LEVELS = {"density": DENSITY_LEVELS, "overlap": OVERLAP_LEVELS}
AXIS_DISPLAY_NAMES = {"density": "Density", "overlap": "Rouleaux"}

DENSITY_TAGS = {
    "No Cells": "no cells",
    "Few Cells": "few cells",
    "Sparser": "sparser",
    "Monolayer": "monolayer",
    "Slightly Dense": "slightly dense",
    "Dense": "dense",
    "Very Dense": "very dense",
}

# A field with no cells has no packing to describe, so an overlap label on it is a
# contradiction rather than a judgement. `few cells` is deliberately NOT in this set: cells
# that are few can still touch, and that is exactly the density-independent overlap signal v3
# is trying to learn.
NO_OVERLAP_DENSITY_LEVELS = frozenset({"no cells"})

# Levels excluded from the ridge fit and the PAVA medians. A blank field's composite is
# answering a different question, so training on it teaches the regression to predict "no
# texture" rather than "low density". They are still scored at evaluation time, as the
# replacement for the old gate's correctness check.
NON_FITTING_DENSITY_LEVELS = frozenset({"no cells"})

# The quality tags the annotation tool already emits. Recorded, not fitted. `Empty` is accepted
# as a synonym for the `No Cells` density tag rather than a quality flag, so a worklist filled
# in against the interim vocabulary still parses.
QUALITY_TAGS = ("Crenated", "Unfocused", "Overexposed", "Artifact", "Other Dimples",
                "Other", "Large", "Medium")
EMPTY_TAG_SYNONYMS = {"Empty": "no cells", "No Cells": "no cells", "Few Cells": "few cells"}


def density_ordinal(label):
    return DENSITY_LEVELS.index(label.strip().lower())


def display_level(label):
    return label.title()


def parse_tanzania_tags(tags_str, default_overlap=DEFAULT_OVERLAP_LABEL):
    """Free-text `tags` -> (density_label, overlap_label).

    Same contract as `_v2_common.parse_tanzania_tags` -- a missing density tag raises, because
    in this dataset that is a data bug rather than a default -- extended with the two new
    density tags and the `Empty` synonym. Quality tags are ignored here; `parse_quality_tags`
    returns them separately for callers that want them.
    """
    parts = [p.strip() for p in tags_str.split(",") if p.strip()]
    density_label = None
    overlap_label = None
    for part in parts:
        if part in EMPTY_TAG_SYNONYMS:
            density_label = EMPTY_TAG_SYNONYMS[part]
        elif part in DENSITY_TAGS:
            density_label = DENSITY_TAGS[part]
        elif part in OVERLAP_TAGS:
            overlap_label = OVERLAP_TAGS[part]

    if density_label is None:
        raise ValueError(f"no density tag in {tags_str!r}; expected one of "
                         f"{sorted(DENSITY_TAGS)}")

    if density_label in NO_OVERLAP_DENSITY_LEVELS:
        if overlap_label is not None and overlap_label != DEFAULT_OVERLAP_LABEL:
            raise ValueError(f"{density_label!r} cannot carry the overlap tag "
                             f"{overlap_label!r}: {tags_str!r}")
        return density_label, DEFAULT_OVERLAP_LABEL

    return density_label, (default_overlap if overlap_label is None else overlap_label)


def parse_quality_tags(tags_str):
    """The non-density, non-overlap tags, in the order written. Recorded, never fitted."""
    return [p.strip() for p in tags_str.split(",")
            if p.strip() in QUALITY_TAGS]


def collapse_to_v2_density(label):
    """Map a v3 density label onto v2.2's 5-level vocabulary.

    Needed in exactly one place that matters: the blind re-label set was drawn from FOVs
    labelled under the 5-level vocabulary, so a raw comparison would score a vocabulary change
    as annotator disagreement. Collapsing the two new rungs into `sparser` makes the
    self-agreement rate a clean noise measurement, and the redistribution of the old `sparser`
    FOVs across the new rungs is then reported separately as its own result.
    """
    label = label.strip().lower()
    return "sparser" if label in ("no cells", "few cells") else label
