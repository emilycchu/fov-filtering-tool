"""Merge the v3 label sources into one calibration set, at the 7-level density vocabulary.

  - data/labels/tanzania-combinedv3/{5 slides}_labels.csv   412 FOVs, free-text `tags`
  - data/labels/tanzania-073026/KTR-72502948-annotated.csv   324 FOVs (legacy, exhaustive)
  - data/labels/tanzania-080526/KTR-72502946-annotated.csv   324 FOVs (legacy, exhaustive)

1060 rows across 7 slide groups, up from v2.2's 661 across 2, and for the first time spanning
all four sites and four boxes rather than two slides from one box.

## What is deliberately dropped

`data/labels/initial-dataset-071626/fovs.csv` -- all 13 rows -- is excluded outright. Nine are
Liberia FOVs, which means LB was never actually held out of the v2.2 calibration pool; the other
four are single-FOV Tanzania slides that contribute nothing to a slide-grouped fit and produce
the degenerate 1-FOV folds that made v2.2's leave-one-slide-out Rouleaux rho meaningless.
Nigeria is held out entirely and was never in this pool. The 4 test slides are asserted absent.

## Columns

v2's eight columns, in v2's order, so the legacy rows can be joined against
`merged-labels-v2.2.csv` field-for-field, then six appended: `slide`, `fov_id`, `split`,
`density_sub`, `overlap_sub`, `quality_tags`.

`slide` and `fov_id` are written explicitly rather than regexed back out of the filename by
`calibrate_v2.slide_of`. Slide grouping is the axis the whole v3 design rests on, and
`(slide, fov_id)` is the key `build_features_v3.py` joins on against the cohort per-FOV CSVs,
so both should be auditable in the file rather than reconstructed by every reader.

There is deliberately **no** `density_target` and no `sample_weight` column. Both are functions
of a flag (`--sub-delta`) and of which rows end up in the fit, and materialising them here would
turn the delta ablation into a data-regeneration question instead of one loop in the fit.

Usage:
    python scripts/combined/combined-v3/merge_labels_v3.py [--out PATH] [--excluded-out PATH]
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent
ROOT = V2_DIR.parent.parent
sys.path.insert(0, str(V2_DIR))
sys.path.insert(0, str(ROOT / "scripts" / "tanzania-complete-081426"))

from _v3_common import (  # noqa: E402
    DENSITY_LEVELS,
    OVERLAP_LEVELS,
    density_ordinal,
    has_quality_only,
    overlap_ordinal,
    parse_quality_tags,
    parse_tanzania_tags_v3,
)
from _v2_common import read_csv_dicts  # noqa: E402
from merge_labels_v2 import tanzania_row_paths  # noqa: E402
from _slide_common import TZ_BUCKET, dpc_blob_name, write_csv_atomic  # noqa: E402

FIELDNAMES = ["fov_key", "dataset", "filename", "image_path", "density_label", "overlap_label",
              "density_ord", "overlap_ord",
              "slide", "fov_id", "split", "density_sub", "overlap_sub", "quality_tags"]
EXCLUDED_FIELDNAMES = ["slide", "fov_id", "tags", "reason"]

COMBINEDV3_DATASET = "tanzania-combinedv3"
COMBINEDV3_LABELS_DIR = ROOT / "data" / "labels" / COMBINEDV3_DATASET
RESULTS_DIR = ROOT / "data" / "results" / "combined-v3"
SPLITS_CSV = RESULTS_DIR / "slide-splits.csv"
MERGED_LABELS_V3_CSV = RESULTS_DIR / "merged-labels-v3.csv"
EXCLUDED_CSV = RESULTS_DIR / "excluded-quality-only.csv"

# The legacy sources are one slide each, and the slide id is not recoverable from the dataset
# label, so the mapping is written down rather than parsed back out of a filename.
LEGACY_SLIDE_OF = {"tanzania-073026": "KTR-72502948", "tanzania-080526": "KTR-72502946"}
LEGACY_SOURCES = ("tanzania-073026", "tanzania-080526")

QUALITY_ONLY_REASON = "quality tags only, no density or overlap level"


def load_splits(path=SPLITS_CSV):
    """slide_id -> row, from the committed roster. The single source of truth for box and role."""
    return {r["slide_id"]: r for r in read_csv_dicts(path)}


def _label_row(fov_key, dataset, filename, image_path, slide, fov_id, split, tags):
    density, density_sub_low, overlap, overlap_sub_low = parse_tanzania_tags_v3(tags)
    return {
        "fov_key": fov_key,
        "dataset": dataset,
        "filename": filename,
        "image_path": image_path,
        "density_label": density,
        "overlap_label": overlap,
        "density_ord": density_ordinal(density),
        "overlap_ord": overlap_ordinal(overlap),
        "slide": slide,
        "fov_id": fov_id,
        "split": split,
        "density_sub": "low" if density_sub_low else "",
        "overlap_sub": "low" if overlap_sub_low else "",
        "quality_tags": ";".join(parse_quality_tags(tags)),
    }


def load_combinedv3_slides(splits, labels_dir=COMBINEDV3_LABELS_DIR):
    """-> (rows, excluded). Images are streamed from GCS, so image_path is a gs:// URI."""
    rows, excluded = [], []
    for path in sorted(labels_dir.glob("*_labels.csv")):
        slide = path.name.replace("_labels.csv", "")
        if slide not in splits:
            raise KeyError(f"{path.name}: {slide!r} is not in {SPLITS_CSV}")
        spec = splits[slide]
        if spec["role"] != "train":
            raise ValueError(f"{slide} has role {spec['role']!r}; only train slides may be "
                             f"merged -- the test slides stay unscored until the fit is frozen")
        box = spec["box"]
        for r in read_csv_dicts(path):
            fov_id = int(r["fov_id"])
            tags = r["tags"]
            if has_quality_only(tags):
                excluded.append({"slide": slide, "fov_id": fov_id, "tags": tags,
                                 "reason": QUALITY_ONLY_REASON})
                continue
            filename = f"dpc-{fov_id:03d}-{slide}.png"
            rows.append(_label_row(
                fov_key=f"{COMBINEDV3_DATASET}/{filename}",
                dataset=COMBINEDV3_DATASET,
                filename=filename,
                image_path=f"gs://{TZ_BUCKET}/{dpc_blob_name(box, slide, fov_id)}",
                slide=slide, fov_id=fov_id, split=spec["role"], tags=tags,
            ))
    return rows, excluded


def load_legacy(source, splits):
    """One legacy slide, keyed and pathed by `merge_labels_v2.tanzania_row_paths`.

    Sharing that helper is what makes "the legacy rows are unchanged" checkable rather than two
    copies of the same string formatting happening to agree: the fov_key it builds is the join
    key `verify_v3_labels.py` uses against the committed v2 merged CSV.
    """
    slide = LEGACY_SLIDE_OF[source]
    split = splits[slide]["role"]
    labels_csv = ROOT / "data" / "labels" / source / f"{slide}-annotated.csv"
    rows = []
    for r in read_csv_dicts(labels_csv):
        fov_id = int(r["fov_id"])
        fov_key, dataset, filename, image_path = tanzania_row_paths(source, fov_id)
        rows.append(_label_row(fov_key, dataset, filename, image_path, slide, fov_id, split,
                               r["tags"]))
    return rows


def build_rows(splits):
    """-> (rows, excluded) for the whole v3 pool, new slides first then legacy, in file order."""
    new_rows, excluded = load_combinedv3_slides(splits)
    legacy_rows = [r for source in LEGACY_SOURCES for r in load_legacy(source, splits)]
    return new_rows + legacy_rows, excluded


def check_pool(rows, splits):
    """The two invariants that must hold before anything is written."""
    keys = [r["fov_key"] for r in rows]
    if len(set(keys)) != len(keys):
        seen, dupes = set(), set()
        for k in keys:
            dupes.add(k) if k in seen else seen.add(k)
        raise ValueError(f"duplicate fov_key: {sorted(dupes)[:5]}")
    test_slides = {s for s, r in splits.items() if r["role"] == "test"}
    leaked = sorted({r["slide"] for r in rows} & test_slides)
    if leaked:
        raise ValueError(f"test slides present in the training merge: {leaked}")


def print_level_counts(rows):
    for axis, levels, key in [("density", DENSITY_LEVELS, "density_label"),
                              ("overlap (Rouleaux)", OVERLAP_LEVELS, "overlap_label")]:
        counts = {level: 0 for level in levels}
        for r in rows:
            counts[r[key]] += 1
        print(f"{axis} counts: " + ", ".join(f"{lv}={counts[lv]}" for lv in levels))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(MERGED_LABELS_V3_CSV))
    ap.add_argument("--excluded-out", default=str(EXCLUDED_CSV))
    args = ap.parse_args()

    splits = load_splits()
    rows, excluded = build_rows(splits)
    check_pool(rows, splits)

    write_csv_atomic(args.out, FIELDNAMES, rows)
    write_csv_atomic(args.excluded_out, EXCLUDED_FIELDNAMES, excluded)

    per_slide = {}
    for r in rows:
        per_slide[r["slide"]] = per_slide.get(r["slide"], 0) + 1
    n_new = sum(1 for r in rows if r["dataset"] == COMBINEDV3_DATASET)
    print(f"new: {n_new} FOVs, legacy: {len(rows) - n_new} FOVs, "
          f"merged: {len(rows)} FOVs across {len(per_slide)} slide groups")
    for slide in sorted(per_slide):
        print(f"  {slide}: {per_slide[slide]}")
    print_level_counts(rows)
    print(f"density_sub=low: {sum(1 for r in rows if r['density_sub'])}, "
          f"overlap_sub=low: {sum(1 for r in rows if r['overlap_sub'])}")
    print(f"\nexcluded {len(excluded)} quality-only rows ({QUALITY_ONLY_REASON}):")
    for e in excluded:
        print(f"  {e['slide']} fov {e['fov_id']}: {e['tags']}")
    print(f"\nwrote {args.out}\nwrote {args.excluded_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
