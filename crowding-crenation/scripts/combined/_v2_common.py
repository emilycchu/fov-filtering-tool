"""Shared constants, label parsing, IO, and feature computation for the v2 density/Rouleaux
pipeline (merge_labels_v2.py, extract_features_v2.py, calibrate_v2.py, plot_results_v2.py,
score_fov_v2.py).

compute_features() is the single source of truth for turning an image into the feature
vector both calibration (extract_features_v2.py) and inference (score_fov_v2.py) consume --
importing it from here in both places guarantees they can never drift apart.

The "overlap" axis is internally named "overlap" (matching the source label CSVs' column
name and the existing repo convention), but is always displayed to the user as "Rouleaux"
(see AXIS_DISPLAY_NAMES / display_level) per project convention for this tool.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.features.edge_density import edge_density  # noqa: E402
from src.features.glcm_contrast import glcm_contrast  # noqa: E402
from src.features.lbp_entropy import lbp_entropy  # noqa: E402
from src.features.otsu_separability import otsu_separability  # noqa: E402
from src.features.tile_heterogeneity import coefficient_of_variation, patchiness, tile_statistics  # noqa: E402
from src.pipeline import GCSPath, list_image_paths, load_image  # noqa: E402,F401  (re-exported)
from src.segmentation import cell_coverage, correct_illumination, otsu_segment, to_grayscale  # noqa: E402

# --- paths ---
INITIAL_LABELS_CSV = ROOT / "data" / "labels" / "initial-dataset-071626" / "fovs.csv"
INITIAL_IMAGE_DIR = ROOT / "data" / "raw" / "initial-dataset-071626"
TANZANIA_LABELS_CSV = ROOT / "data" / "labels" / "tanzania-073026" / "KTR-72502948-annotated.csv"
TANZANIA_IMAGE_DIR = ROOT / "data" / "raw" / "new" / "KTR-72502948" / "dpc"
TANZANIA_IMAGE_NAME = "dpc-{fov_id:03d}-KTR-72502948.png"

# Tanzania-080526: KTR-72502946, streamed straight from GCS -- never downloaded locally
# (see data/results/tanzania-080526/README.md for why).
TANZANIA_080526_LABELS_CSV = ROOT / "data" / "labels" / "tanzania-080526" / "KTR-72502946-annotated.csv"
TANZANIA_080526_BUCKET = "tanzania_02032026"
TANZANIA_080526_BLOB_PREFIX = "TZ2025-Box5/KTR-72502946"
TANZANIA_080526_IMAGE_NAME = "dpc-{fov_id:03d}-KTR-72502946.png"

# nigeria-081226: 8 FOVs across 2 slides, local BMPs. Labels live in a hand-written CSV
# (the `_sparse` filename suffix is not parsed -- see data/labels/nigeria-081226/).
NIGERIA_LABELS_CSV = ROOT / "data" / "labels" / "nigeria-081226" / "nigeria-081226-annotated.csv"
NIGERIA_IMAGE_DIR = ROOT / "data" / "raw" / "nigeria-081226"

RESULTS_DIR = ROOT / "data" / "results" / "density-rouleaux-v2"
MERGED_LABELS_CSV = RESULTS_DIR / "merged-labels.csv"
FEATURES_CSV = RESULTS_DIR / "features.csv"

# LBP runtime study (bench_lbp.py -> extract_lbp_variants.py -> build_variant_features.py ->
# compare_lbp_variants.py). Separate results dir so nothing here can overwrite the v2.2 fit.
LBP_RUNTIME_DIR = ROOT / "data" / "results" / "lbp-runtime"
LBP_VARIANTS_CSV = LBP_RUNTIME_DIR / "lbp-variants.csv"
LBP_COMPARISON_CSV = LBP_RUNTIME_DIR / "variant-comparison.csv"
PARAMS_JSON = RESULTS_DIR / "density_overlap_v2_params.json"
REPORT_MD = RESULTS_DIR / "calibration-report.md"
PLOTS_DIR = RESULTS_DIR / "plots"

# --- label vocabulary: 5 levels each, "sparser" kept distinct from "monolayer" ---
DENSITY_LEVELS = ["sparser", "monolayer", "slightly dense", "dense", "very dense"]
OVERLAP_LEVELS = ["no rouleaux", "slight rouleaux", "some rouleaux", "rouleaux", "heavy rouleaux"]
DEFAULT_DENSITY_LABEL = "monolayer"
DEFAULT_OVERLAP_LABEL = "no rouleaux"

AXIS_LEVELS = {"density": DENSITY_LEVELS, "overlap": OVERLAP_LEVELS}
AXIS_DISPLAY_NAMES = {"density": "Density", "overlap": "Rouleaux"}

# Tanzania free-text tag -> our lowercase level vocabulary. Deliberately NOT reusing
# scripts/compare_tanzania_labels.py's fold_density(), which folds "Sparser" into
# "Monolayer" -- this task wants sparser as a genuine 5th level.
DENSITY_TAGS = {
    "Sparser": "sparser",
    "Monolayer": "monolayer",
    "Slightly Dense": "slightly dense",
    "Dense": "dense",
    "Very Dense": "very dense",
}
OVERLAP_TAGS = {
    "Slight Rouleaux": "slight rouleaux",
    "Some Rouleaux": "some rouleaux",
    "Rouleaux": "rouleaux",
    "Heavy Rouleaux": "heavy rouleaux",
}

# tile-grid feature parameters (see src/features/tile_heterogeneity.py)
TILE_GRID_SIZE = 7
TILE_GLCM_LEVELS = 32

# dataviz palette (references/palette.md), matching scripts/tanzania_comparison.py
JITTER_SEED = 7
COLOR_MINE = "#2a78d6"
COLOR_GRID = "#e1e0d9"
COLOR_AXIS = "#c3c2b7"
COLOR_MUTED = "#898781"
COLOR_PRIMARY = "#0b0b0b"
COLOR_SECONDARY = "#52514e"
COLOR_SURFACE = "#fcfcfb"


def density_ordinal(label):
    return DENSITY_LEVELS.index(label.strip().lower())


def overlap_ordinal(label):
    return OVERLAP_LEVELS.index(label.strip().lower())


def display_level(label):
    """'slightly dense' -> 'Slightly Dense', 'no rouleaux' -> 'No Rouleaux'."""
    return label.title()


def parse_tanzania_tags(tags_str):
    """Parse the Tanzania annotated CSV's free-text `tags` column into (density_label,
    overlap_label). The density tag is always present in this dataset (a missing one is a
    real data bug, so this raises); the overlap tag defaults to "no rouleaux" when absent.
    Other tags (Crenated, Unfocused, Artifact, Other Dimples) are ignored.
    """
    parts = [p.strip() for p in tags_str.split(",")]
    density_label = None
    overlap_label = None
    for p in parts:
        if p in DENSITY_TAGS:
            density_label = DENSITY_TAGS[p]
        elif p in OVERLAP_TAGS:
            overlap_label = OVERLAP_TAGS[p]
    if density_label is None:
        raise ValueError(f"no density tag found in {tags_str!r}")
    return density_label, overlap_label or DEFAULT_OVERLAP_LABEL


def read_csv_dicts(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv_dicts(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_features(image):
    """The full candidate feature vector for one FOV image (BGR, as returned by load_image).

    Shared by extract_features_v2.py (calibration) and score_fov_v2.py (inference) so the
    two can never compute features differently.
    """
    gray = to_grayscale(image)
    mask, otsu_threshold = otsu_segment(image)
    coverage = cell_coverage(mask)
    eta = otsu_separability(image)
    saturation_score = float(np.clip(coverage * (1.0 - eta), 0.0, 1.0))

    corrected = correct_illumination(gray, blur_ksize=301)

    def _tile_glcm_contrast(tile):
        quantized = (tile.astype(np.uint16) * TILE_GLCM_LEVELS // 256).astype(np.uint8)
        return glcm_contrast(quantized, levels=TILE_GLCM_LEVELS)

    tile_contrasts = tile_statistics(corrected, grid_size=TILE_GRID_SIZE, stat_fn=_tile_glcm_contrast)

    return {
        "coverage": coverage,
        "otsu_threshold": float(otsu_threshold),
        "otsu_separability": eta,
        "saturation_score": saturation_score,
        "lbp_entropy": lbp_entropy(image),
        "glcm_contrast": glcm_contrast(image),
        "edge_density_unmasked": edge_density(image),
        "tile_glcm_cv": coefficient_of_variation(tile_contrasts),
        "tile_glcm_patchiness": patchiness(tile_contrasts),
    }


def apply_saturation_override(label, features, override_cfg):
    """No-op passthrough while override_cfg is None or disabled (today's behavior, per
    project decision to keep the saturation signal data-driven rather than a hard rule).

    To switch to a hard override later: set override_cfg = {"enabled": true,
    "feature": "saturation_score", "threshold": <fitted cutoff>, "max_label": <top bucket
    label for this axis>} in density_overlap_v2_params.json -- no code restructuring needed.
    """
    if not override_cfg or not override_cfg.get("enabled"):
        return label
    # float() because some callers pass a raw CSV row (strings), not a computed feature dict
    if float(features.get(override_cfg["feature"], 0.0)) >= override_cfg["threshold"]:
        return override_cfg["max_label"]
    return label


# The four features that read a near-empty field correctly: all are texture/contrast measures
# that go to ~0 with no cells present. See data/results/nigeria-081226/README.md.
EMPTY_FIELD_FEATURES = ("otsu_separability", "lbp_entropy", "glcm_contrast", "edge_density_unmasked")


def empty_field_features_below(features, cfg):
    """Which of cfg["thresholds"]'s features sit strictly below their floor.

    A feature that is missing or blank is deliberately NOT counted as below: the gate forces
    the bottom bucket, so incomplete input must fail it rather than trip it.
    """
    if not cfg:
        return []
    below = []
    for name, floor in cfg["thresholds"].items():
        value = features.get(name)
        if value is None or value == "":
            continue
        if float(value) < floor:
            below.append(name)
    return below


def empty_field_fired(features, cfg):
    """True when every gate feature is below its floor and the gate is enabled."""
    if not cfg or not cfg.get("enabled"):
        return False
    rule = cfg.get("rule", "all_below")
    if rule != "all_below":
        raise ValueError(f"unknown empty_field_override rule {rule!r}")
    return len(empty_field_features_below(features, cfg)) == len(cfg["thresholds"])


def apply_empty_field_override(density_label, overlap_label, features, cfg):
    """Force the bottom bucket on *both* axes when the field is empty.

    Unlike apply_saturation_override's per-axis ceiling, this is one joint test with two
    outputs: with no cells present there is nothing for either composite to describe, so its
    output is discarded rather than adjusted. On a near-empty field Otsu has no bimodal
    histogram to split, so `coverage` and `saturation_score` read background noise as dense
    tissue while the four features that know better are clipped to 0 by normalize() -- the
    composite is answering the wrong question, and no reweighting fixes that.

    Measured against the 661-FOV v2.2 calibration set this fires on 3 FOVs, all of them
    manually labeled sparser + no rouleaux and already predicted as such: exact-match is
    unchanged at 0.6974 / 0.6838. See scripts/combined/README.md.
    """
    if not empty_field_fired(features, cfg):
        return density_label, overlap_label
    return cfg["density_label"], cfg["overlap_label"]


def apply_label_overrides(density_label, overlap_label, features, params):
    """Every params-driven post-composite label override, in order.

    Call this rather than the individual override functions -- it is the single place a new
    override has to be added, instead of the four call sites that score a feature vector
    (score_fov_v2.py, plot_bucket_comparison_v2.py, tanzania_080526_rescore.py,
    nigeria_081226.py).
    """
    saturation = params.get("saturation_override") or {}
    density_label = apply_saturation_override(density_label, features, saturation.get("density"))
    overlap_label = apply_saturation_override(overlap_label, features, saturation.get("overlap"))
    return apply_empty_field_override(density_label, overlap_label, features,
                                      params.get("empty_field_override"))
