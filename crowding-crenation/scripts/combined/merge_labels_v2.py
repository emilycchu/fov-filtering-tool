"""Merge the manual label sources into one calibration set for the v2 density/Rouleaux
pipeline:

  - data/labels/initial-dataset-071626/fovs.csv       (13 FOVs, clean density/overlap columns)
  - data/labels/tanzania-073026/KTR-72502948-annotated.csv (324 FOVs, free-text `tags` column)
  - data/labels/tanzania-080526/KTR-72502946-annotated.csv (324 FOVs, free-text `tags` column,
    images streamed from GCS -- never downloaded locally)

Density uses the full 5-level scale (sparser, monolayer, slightly dense, dense, very dense)
-- "sparser" is kept distinct from "monolayer", unlike scripts/compare_tanzania_labels.py's
fold_density(), since pooling all three datasets gives enough sparser examples (~50 across
the two Tanzania slides) to no longer need that fold.

Usage:
    python scripts/combined/merge_labels_v2.py [--out PATH]
"""
import argparse

from _v2_common import (
    DEFAULT_DENSITY_LABEL,
    DEFAULT_OVERLAP_LABEL,
    DENSITY_LEVELS,
    INITIAL_IMAGE_DIR,
    INITIAL_LABELS_CSV,
    MERGED_LABELS_CSV,
    OVERLAP_LEVELS,
    TANZANIA_080526_BLOB_PREFIX,
    TANZANIA_080526_BUCKET,
    TANZANIA_080526_IMAGE_NAME,
    TANZANIA_080526_LABELS_CSV,
    TANZANIA_IMAGE_DIR,
    TANZANIA_IMAGE_NAME,
    TANZANIA_LABELS_CSV,
    density_ordinal,
    overlap_ordinal,
    parse_tanzania_tags,
    read_csv_dicts,
    write_csv_dicts,
)

FIELDNAMES = ["fov_key", "dataset", "filename", "image_path", "density_label", "overlap_label",
              "density_ord", "overlap_ord"]


def load_initial_dataset(csv_path, image_dir):
    rows = []
    for r in read_csv_dicts(csv_path):
        filename = r["filename"]
        image_path = image_dir / filename
        if not image_path.exists():
            raise FileNotFoundError(f"initial dataset: missing image for {filename!r}: {image_path}")

        density_label = (r.get("density") or "").strip().lower() or DEFAULT_DENSITY_LABEL
        overlap_label = (r.get("overlap") or "").strip().lower() or DEFAULT_OVERLAP_LABEL

        rows.append({
            "fov_key": f"initial-071626/{filename}",
            "dataset": "initial-071626",
            "filename": filename,
            "image_path": str(image_path),
            "density_label": density_label,
            "overlap_label": overlap_label,
            "density_ord": density_ordinal(density_label),
            "overlap_ord": overlap_ordinal(overlap_label),
        })
    return rows


# The two legacy Tanzania sources, declared once. They differ in exactly one respect -- 073026's
# images were downloaded and 080526's are streamed from GCS -- so the key, dataset and filename
# construction is shared and `image_path` is the only field that branches.
TANZANIA_LEGACY_SOURCES = {
    "tanzania-073026": {
        "labels_csv": TANZANIA_LABELS_CSV,
        "name_template": TANZANIA_IMAGE_NAME,
        "image_dir": TANZANIA_IMAGE_DIR,
        "gcs": None,
    },
    "tanzania-080526": {
        "labels_csv": TANZANIA_080526_LABELS_CSV,
        "name_template": TANZANIA_080526_IMAGE_NAME,
        "image_dir": None,
        "gcs": (TANZANIA_080526_BUCKET, TANZANIA_080526_BLOB_PREFIX),
    },
}


def tanzania_row_paths(source, fov_id):
    """-> (fov_key, dataset, filename, image_path) for one legacy Tanzania FOV.

    Shared with `combined-v3/merge_labels_v3.py` so that "v3 reproduces v2's legacy rows
    exactly" is a structural property rather than two copies of the same string formatting
    happening to agree. `fov_key` is the join key between the two merged files, so it is the
    one string that must not drift.

    For 073026 the images were downloaded, so `image_path` is a local path and its existence is
    checked here -- a missing image is a broken merge, not a row to skip. For 080526 they live
    only in GCS, so `image_path` is a gs:// URI and there is no existence check, trusting the
    caller already confirmed the blobs exist.
    """
    spec = TANZANIA_LEGACY_SOURCES[source]
    filename = spec["name_template"].format(fov_id=fov_id)
    if spec["gcs"] is None:
        local = spec["image_dir"] / filename
        if not local.exists():
            raise FileNotFoundError(f"{source}: missing image for fov_id={fov_id}: {local}")
        image_path = str(local)
    else:
        bucket, blob_prefix = spec["gcs"]
        image_path = f"gs://{bucket}/{blob_prefix}/{filename}"
    return f"{source}/{filename}", source, filename, image_path


def load_tanzania_legacy(source):
    """One legacy Tanzania slide's labelled FOVs, in file order."""
    rows = []
    for r in read_csv_dicts(TANZANIA_LEGACY_SOURCES[source]["labels_csv"]):
        fov_key, dataset, filename, image_path = tanzania_row_paths(source, int(r["fov_id"]))
        density_label, overlap_label = parse_tanzania_tags(r["tags"])
        rows.append({
            "fov_key": fov_key,
            "dataset": dataset,
            "filename": filename,
            "image_path": image_path,
            "density_label": density_label,
            "overlap_label": overlap_label,
            "density_ord": density_ordinal(density_label),
            "overlap_ord": overlap_ordinal(overlap_label),
        })
    return rows


def merge_datasets(*row_lists):
    merged = []
    for rows in row_lists:
        merged.extend(rows)
    return merged


def print_level_counts(rows):
    for axis, levels, key in [("density", DENSITY_LEVELS, "density_label"), ("overlap (Rouleaux)", OVERLAP_LEVELS, "overlap_label")]:
        counts = {level: 0 for level in levels}
        for r in rows:
            counts[r[key]] += 1
        print(f"{axis} counts: " + ", ".join(f"{level}={counts[level]}" for level in levels))


def main():
    parser = argparse.ArgumentParser(description="Merge initial + Tanzania manual labels into one calibration CSV.")
    parser.add_argument("--out", default=str(MERGED_LABELS_CSV))
    args = parser.parse_args()

    initial_rows = load_initial_dataset(INITIAL_LABELS_CSV, INITIAL_IMAGE_DIR)
    tanzania_rows = load_tanzania_legacy("tanzania-073026")
    tanzania_080526_rows = load_tanzania_legacy("tanzania-080526")
    rows = merge_datasets(initial_rows, tanzania_rows, tanzania_080526_rows)

    write_csv_dicts(args.out, FIELDNAMES, rows)

    print(f"initial: {len(initial_rows)} FOVs, tanzania-073026: {len(tanzania_rows)} FOVs, "
          f"tanzania-080526: {len(tanzania_080526_rows)} FOVs, merged: {len(rows)} FOVs")
    if len(rows) != 661:
        print(f"warning: expected 661 merged FOVs, got {len(rows)}")
    print_level_counts(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
