"""Resolve every catalog slide to its TZ2025 box and verify which FOV ids actually exist.

Run once, before the passes. Writes two files the passes then treat as ground truth:

  slides.csv        the frozen catalog (271 rows) + each slide's box
  slide-index.json  per slide: box, per-channel FOV counts and missing ids

Both exist so the workbook is parsed exactly once and every pass -- in either subproject --
works from an identical, auditable slide list.

Two things this replaces, both of which matter at 87,804 FOVs:

**`find_tz_box` is uncached and probes up to 5 prefixes per call.** Calling it per FOV would
be up to 439,020 list requests. Instead, five `list_blobs(prefix="TZ2025-Box<N>/",
delimiter="/")` calls return `.prefixes` -- the full set of slide folders in each box -- so the
whole {slide_id: box} map costs **5 requests**.

**Constructing 324 blob names per channel and hoping.** One `list_blobs` per slide (271
requests, both channels at once) turns "this FOV is missing" from a NotFound discovered an hour
into the run into a recorded fact known before it starts. The names themselves are regular, so
the index stores counts and gaps rather than ~648 names per slide (~35 MB of JSON for nothing).

Usage:
    python scripts/tanzania-complete-081426/build_slide_index.py
    python scripts/tanzania-complete-081426/build_slide_index.py --catalog-only
"""
import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _slide_common import (  # noqa: E402
    DPC_RE,
    EXPECTED_FOVS,
    FLUOR_RE,
    RESULTS_DIR,
    SLIDE_INDEX_JSON,
    SLIDES_CSV,
    TZ_BOXES,
    TZ_BUCKET,
    write_csv_atomic,
)
from catalog import CATALOG_XLSX, SHEET_NAME, load_tanzania_slides  # noqa: E402

SLIDES_FIELDNAMES = ["slide_id", "box", "truth", "truth_warn", "train_test_split", "region",
                     "in_catalog"]

# Slides indexed for validation but deliberately outside the 271-slide analysis. KTR-72502946
# is the tanzania-080526 slide whose 324 per-FOV scores are already committed and verified, so
# it is the only available bit-for-bit regression target for the new harness. It is not in the
# catalog workbook.
REGRESSION_SLIDES = ("KTR-72502946",)

_DPC_RE = re.compile(DPC_RE)
_FLUOR_RE = re.compile(FLUOR_RE)


def _client():
    from src.pipeline import _gcs_client

    return _gcs_client()


def box_map(bucket=TZ_BUCKET, boxes=TZ_BOXES):
    """{slide_id: "TZ2025-Box<N>"} for every slide folder in the bucket, in 5 requests."""
    client = _client()
    mapping = {}
    collisions = []
    for n in boxes:
        box = f"TZ2025-Box{n}"
        iterator = client.list_blobs(bucket, prefix=f"{box}/", delimiter="/")
        list(iterator)  # consume the page so .prefixes is populated
        found = [p.split("/")[1] for p in iterator.prefixes if p.count("/") >= 2]
        for slide_id in found:
            if slide_id in mapping:
                collisions.append((slide_id, mapping[slide_id], box))
            mapping[slide_id] = box
        print(f"  {box}: {len(found)} slide folders")
    if collisions:
        print(f"  WARNING {len(collisions)} slide(s) appear in more than one box: "
              f"{collisions[:5]}")
    return mapping


_CATEGORY_RE = re.compile(r"\d")


def _category(name):
    """"segmentation-mask-017-NKR-x.png" -> "segmentation-mask-*". Collapses a per-FOV family
    to one key so the index records what else is in the prefix without storing 324 names."""
    match = _CATEGORY_RE.search(name)
    return (name[:match.start()] + "*") if match else name


def scan_slide(bucket, box, slide_id):
    """One listing of a slide's prefix -> verified FOV ids for both channels, plus what else
    is in there.

    The "extra blobs" bookkeeping is the point as much as the FOV ids. A slide prefix holds
    ~974 objects, and only 648 of them are FOVs:

      dpc-<nnn>-<slide>.png            324  <- wanted
      fluorescent-<nnn>-<slide>.png    324  <- wanted
      segmentation-mask-<nnn>-...png   324  <- flat, .png, and NOT a FOV
      metadata/dpc-{preview,result}.png, metadata/*-scan.txt   nested

    So `list_image_paths` on this prefix returns three times the FOVs, and the extra third
    would be *scored as if they were FOVs*. Anchoring on `^dpc-\\d{3}-` and rejecting anything
    nested is what keeps that out; recording the counts is what proves it on all 271 slides
    instead of on the one slide someone happened to look at.
    """
    client = _client()
    prefix = f"{box}/{slide_id}/"
    dpc_ids, fluor_ids = [], []
    other = Counter()
    n_nested = 0
    for blob in client.list_blobs(bucket, prefix=prefix):
        name = blob.name[len(prefix):]
        if "/" in name:
            n_nested += 1
            continue
        dpc = _DPC_RE.match(name)
        if dpc:
            dpc_ids.append(int(dpc.group(1)))
            continue
        fluor = _FLUOR_RE.match(name)
        if fluor:
            fluor_ids.append(int(fluor.group(1)))
            continue
        other[_category(name)] += 1
    return sorted(set(dpc_ids)), sorted(set(fluor_ids)), dict(other), n_nested


def missing_ids(present, expected=EXPECTED_FOVS):
    return sorted(set(range(1, expected + 1)) - set(present))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", default=str(CATALOG_XLSX))
    parser.add_argument("--sheet", default=SHEET_NAME)
    parser.add_argument("--catalog-only", action="store_true",
                        help="write slides.csv from the workbook only; no GCS access")
    parser.add_argument("--skip-listing", action="store_true",
                        help="resolve boxes but trust the 1..324 formula instead of listing "
                             "each slide (5 requests instead of 276)")
    parser.add_argument("--slides", nargs="*", default=None,
                        help="restrict to these slide_ids (for smoke tests)")
    parser.add_argument("--extra-slides", nargs="*", default=list(REGRESSION_SLIDES),
                        help="index these slide_ids even though they are not in the catalog. "
                             f"Defaults to {list(REGRESSION_SLIDES)} -- see REGRESSION_SLIDES.")
    args = parser.parse_args()

    catalog = load_tanzania_slides(args.xlsx, args.sheet)
    catalog_ids = {s["slide_id"] for s in catalog}

    # KTR-72502946 is a calibration slide with 324 verified per-FOV rows on disk, which makes
    # it the golden-value regression target -- but it is not one of the 271 catalog slides, so
    # it would otherwise be absent from the index and impossible to run the pass against.
    # Indexed, but flagged in_catalog=False so it never joins the 271-row analysis.
    for slide_id in args.extra_slides or []:
        if slide_id not in catalog_ids:
            catalog.append({"slide_id": slide_id, "truth": "", "truth_norm": "",
                            "truth_warn": False, "train_test_split": "", "region": "",
                            "in_catalog": False})
            catalog_ids.add(slide_id)

    if args.slides:
        wanted = set(args.slides)
        catalog = [s for s in catalog if s["slide_id"] in wanted]

    n_extra = sum(1 for s in catalog if s.get("in_catalog") is False)
    print(f"catalog: {len(catalog) - n_extra} slides"
          + (f" (+{n_extra} regression slide(s) not in the catalog)" if n_extra else ""))

    def slide_row(slide, box):
        return {**slide, "box": box, "in_catalog": slide.get("in_catalog", True)}

    if args.catalog_only:
        write_csv_atomic(SLIDES_CSV, SLIDES_FIELDNAMES,
                         [slide_row(s, "") for s in catalog])
        print(f"wrote {SLIDES_CSV} (no boxes -- --catalog-only)")
        return 0

    print(f"resolving boxes in gs://{TZ_BUCKET} ...")
    boxes = box_map()
    print(f"  total: {len(boxes)} slide folders across {len(TZ_BOXES)} boxes")

    unresolved = [s["slide_id"] for s in catalog if s["slide_id"] not in boxes]
    if unresolved:
        print(f"\nFAILED: {len(unresolved)} catalog slides are in no box: {unresolved[:10]}")
        return 1

    index = {"bucket": TZ_BUCKET, "sheet": args.sheet, "boxes_scanned": len(TZ_BOXES),
             "n_slide_folders_found": len(boxes), "expected_fovs": EXPECTED_FOVS,
             "listed": not args.skip_listing, "slides": {}}

    problems = []
    extras_seen = Counter()
    started = time.time()
    for i, slide in enumerate(catalog, 1):
        slide_id = slide["slide_id"]
        box = boxes[slide_id]

        if args.skip_listing:
            entry = {"box": box, "n_dpc": EXPECTED_FOVS, "n_fluorescent": EXPECTED_FOVS,
                     "dpc_missing": [], "fluorescent_missing": [], "other_flat": {},
                     "n_nested": 0,
                     "dpc_fov_ids": list(range(1, EXPECTED_FOVS + 1)),
                     "fluorescent_fov_ids": list(range(1, EXPECTED_FOVS + 1))}
        else:
            dpc_ids, fluor_ids, other, n_nested = scan_slide(TZ_BUCKET, box, slide_id)
            entry = {"box": box, "n_dpc": len(dpc_ids), "n_fluorescent": len(fluor_ids),
                     "dpc_missing": missing_ids(dpc_ids),
                     "fluorescent_missing": missing_ids(fluor_ids),
                     "other_flat": other, "n_nested": n_nested,
                     "dpc_fov_ids": dpc_ids, "fluorescent_fov_ids": fluor_ids}
            for category, count in other.items():
                extras_seen[category] += count
            if len(dpc_ids) != EXPECTED_FOVS or len(fluor_ids) != EXPECTED_FOVS:
                problems.append((slide_id, len(dpc_ids), len(fluor_ids)))

        index["slides"][slide_id] = entry

        if i % 25 == 0 or i == len(catalog):
            rate = i / max(time.time() - started, 1e-9)
            print(f"  scanned {i}/{len(catalog)} slides ({rate:.1f}/s)", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SLIDE_INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    write_csv_atomic(SLIDES_CSV, SLIDES_FIELDNAMES,
                     [slide_row(s, boxes[s["slide_id"]]) for s in catalog])

    print(f"\nwrote {SLIDE_INDEX_JSON}")
    print(f"wrote {SLIDES_CSV}")
    if extras_seen:
        print(f"non-FOV blobs filtered out (the 327-vs-324 trap): {dict(extras_seen)}")
    print(f"box distribution: {dict(Counter(e['box'] for e in index['slides'].values()))}")

    in_catalog = [s["slide_id"] for s in catalog if s.get("in_catalog", True)]
    total_dpc = sum(index["slides"][s]["n_dpc"] for s in in_catalog)
    total_fluor = sum(index["slides"][s]["n_fluorescent"] for s in in_catalog)
    print(f"catalog FOV totals: {total_dpc} dpc, {total_fluor} fluorescent "
          f"(nominal {len(in_catalog) * EXPECTED_FOVS})")

    # Short slides are a property of the data, not a failure: the passes read the verified FOV
    # ids out of this index, so they score what exists rather than generating NotFounds. Worth
    # naming loudly -- a slide-level mean over 322 FOVs is still sound, but n_fovs_scored in
    # slide-summary.csv will not equal 324 for these and that must not read as a bug later.
    if problems:
        print(f"\nNOTE: {len(problems)} slide(s) have fewer than {EXPECTED_FOVS} FOVs "
              f"(slide_id, n_dpc, n_fluorescent):")
        for row in problems:
            print(f"  - {row[0]}: {row[1]} dpc, {row[2]} fluorescent")
        print("  The passes will score exactly the ids listed in the index for each.")

    print(f"\nOK: {len(catalog)}/{len(catalog)} slides resolved to a box")
    return 0


if __name__ == "__main__":
    sys.exit(main())
