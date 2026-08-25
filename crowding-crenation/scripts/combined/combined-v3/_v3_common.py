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


# Case-folded tag lookups, built once. The annotation tool's casing is not stable -- the v3
# label files write `Few cells` / `No cells` where the tables above say `Few Cells` / `No
# Cells` -- and an exact-match parse silently drops those rungs, which is also why
# `blind-relabels-annotations.txt` parsed 0/50. `EMPTY_TAG_SYNONYMS` is folded into the density
# table rather than tested separately: the only key it adds is `Empty`, and the keys it shares
# with `DENSITY_TAGS` map to identical levels, so the merge removes a special case from the
# parse loop without changing what any tag means. The assertion is what keeps that true.
_DENSITY_TAG_TABLE = {**DENSITY_TAGS, **EMPTY_TAG_SYNONYMS}
assert all(DENSITY_TAGS[k] == v for k, v in EMPTY_TAG_SYNONYMS.items() if k in DENSITY_TAGS), \
    "EMPTY_TAG_SYNONYMS disagrees with DENSITY_TAGS on a shared key"

_DENSITY_LOOKUP = {k.casefold(): v for k, v in _DENSITY_TAG_TABLE.items()}
_OVERLAP_LOOKUP = {k.casefold(): v for k, v in OVERLAP_TAGS.items()}
_QUALITY_LOOKUP = {k.casefold(): k for k in QUALITY_TAGS}


def _tag_parts(tags_str):
    """The written tags, stripped, case-folded, empties dropped."""
    return [p.strip().casefold() for p in tags_str.split(",") if p.strip()]


def _resolve_axis(parts, lookup, levels):
    """-> (most_severe_level, had_multiple) for one axis; (None, False) if the axis is absent.

    Severity is position in `levels`. This replaces v2's last-tag-wins loop, which resolved by
    *write order* and so returned whichever of two tags the annotator happened to type second
    -- yielding `sparser` for "Sparser, Few cells" and the milder `some rouleaux` for
    "Dense, Rouleaux, Some Rouleaux", both silently.

    `had_multiple` is the double-label encoding: two rungs on one axis mean the more severe one,
    at its low end. Keyed on distinct levels rather than tag count, so a repeated tag is not
    read as a hedge.
    """
    found = [lookup[p] for p in parts if p in lookup]
    if not found:
        return None, False
    return max(found, key=levels.index), len(set(found)) > 1


def parse_tanzania_tags_v3(tags_str, default_overlap=DEFAULT_OVERLAP_LABEL):
    """Free-text `tags` -> (density, density_sub_low, overlap, overlap_sub_low).

    Keeps both of the raises it inherited: a missing density tag is a data bug rather than a
    default, and a field with no cells has no packing to describe. The first one is the only
    thing that caught `NKR-72502156` fov 284, so it stays a raise rather than becoming a
    default.

    Both `*_sub_low` flags are False for every one of the 648 legacy rows -- they contain no
    double labels -- which is what makes this a strict extension of the v2 parser rather than a
    reinterpretation of the existing pool. `verify_v3_labels.py` asserts exactly that.
    """
    parts = _tag_parts(tags_str)
    density_label, density_sub_low = _resolve_axis(parts, _DENSITY_LOOKUP, DENSITY_LEVELS)
    overlap_label, overlap_sub_low = _resolve_axis(parts, _OVERLAP_LOOKUP, OVERLAP_LEVELS)

    if density_label is None:
        raise ValueError(f"no density tag in {tags_str!r}; expected one of "
                         f"{sorted(DENSITY_TAGS)}")

    if density_label in NO_OVERLAP_DENSITY_LEVELS:
        if overlap_label is not None and overlap_label != DEFAULT_OVERLAP_LABEL:
            raise ValueError(f"{density_label!r} cannot carry the overlap tag "
                             f"{overlap_label!r}: {tags_str!r}")
        return density_label, density_sub_low, DEFAULT_OVERLAP_LABEL, False

    if overlap_label is None:
        return density_label, density_sub_low, default_overlap, False
    return density_label, density_sub_low, overlap_label, overlap_sub_low


def parse_tanzania_tags(tags_str, default_overlap=DEFAULT_OVERLAP_LABEL):
    """The 2-tuple contract, unchanged, so existing callers do not have to move."""
    density_label, _, overlap_label, _ = parse_tanzania_tags_v3(tags_str, default_overlap)
    return density_label, overlap_label


def has_quality_only(tags_str):
    """True for a row naming no density and no overlap level but at least one quality tag.

    The 23 `Overexposed`-only rows were labelled for a future overexposure study, not for
    crowding, so the merge skips exactly these. Every other row with no density tag still
    raises, which is what stops a genuine omission from hiding behind this predicate.
    """
    parts = _tag_parts(tags_str)
    if any(p in _DENSITY_LOOKUP or p in _OVERLAP_LOOKUP for p in parts):
        return False
    return any(p in _QUALITY_LOOKUP for p in parts)


def ordinal_target(ordinal, sub_low, delta):
    """The fitting target for one row: `delta` below its rung when the label was hedged low.

    The single place the double-label encoding enters the numerics, so `--sub-delta 0` recovers
    the un-nudged model exactly and the ablation is one loop rather than a branch in the fit.
    """
    return float(ordinal) - delta if sub_low else float(ordinal)


def parse_quality_tags(tags_str):
    """The canonical spellings of the quality tags named, in the order written.

    Recorded, never fitted. Case-insensitive for the same reason the axis lookups are, and it
    returns the canonical casing so that downstream counts group instead of splitting on how
    the tag happened to be typed.
    """
    return [_QUALITY_LOOKUP[p] for p in _tag_parts(tags_str) if p in _QUALITY_LOOKUP]


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
