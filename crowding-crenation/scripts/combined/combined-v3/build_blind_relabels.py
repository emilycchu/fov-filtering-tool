"""Package 50 already-labelled FOVs for blind re-annotation, to measure the label-noise ceiling.

Every accuracy number this project reports is bounded by how often the annotator agrees with
themselves, and that bound has never been measured. v2.2's headline is 55% density exact-match
under slide-grouped CV; whether that is a mediocre model or a model at the ceiling is currently
unanswerable, and the answer changes whether more feature work is worth doing at all.

**Why a zip of renamed images rather than a pass through the annotation tool.** The tool shows a
slide's FOVs in order, with the sample id visible, and the prior labels are a click away -- so a
second pass through it measures recall of the first pass, not independent judgement. Blindness
here needs three things, all of which this script does:

  * **Anonymous filenames.** `fov-01.png` .. `fov-50.png`. PNG carries no filename internally,
    so a plain byte copy under a new name leaks nothing.
  * **Shuffled order.** The mapping is a seeded permutation, so neither slide nor label can be
    inferred from position. Both slides are interleaved.
  * **The key is never written into the zip.** It goes to a sibling CSV that the annotator does
    not open. Re-running with the same seed reproduces it, so losing the key is recoverable.

**Sampling: proportional, with a floor.** A pooled self-agreement rate is only an unbiased
ceiling if the sample mirrors the label distribution, so allocation is proportional to each
(density, overlap) cell's frequency in the 648-FOV pool. Pure proportional would spend ~30 of 50
FOVs on `monolayer / no rouleaux` and never show a `dense` or `heavy rouleaux` field, so every
density level and every overlap level is guaranteed at least `--floor` FOVs. That biases the
pooled rate slightly toward the rare buckets, so `sampling_weight` (cell frequency / cell
sampling rate) is recorded in the key -- a weighted mean recovers the unbiased estimate, and an
unweighted one gives the per-bucket view.

Only the two fully-annotated Tanzania slides are eligible. The 13 initial-071626 FOVs are
excluded: 9 are Liberia (held out in v3) and the other 4 are one FOV each from four slides, so
re-labelling them measures nothing about a slide.

Usage:
    python scripts/combined/combined-v3/build_blind_relabels.py
    python scripts/combined/combined-v3/build_blind_relabels.py --n 50 --seed 20260820
"""
import argparse
import csv
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "combined"))

sys.path.insert(0, str(HERE))

from _v2_common import DENSITY_LEVELS as DENSITY_LEVELS_V2  # noqa: E402
from _v2_common import OVERLAP_LEVELS  # noqa: E402
from _v3_common import DENSITY_LEVELS  # noqa: E402,F401 -- the 7-level vocabulary

MERGED_LABELS = ROOT / "data" / "results" / "density-rouleaux-v2" / "merged-labels-v2.2.csv"
OUT_DIR = ROOT / "data" / "results" / "combined-v3"
ELIGIBLE_DATASETS = ("tanzania-073026", "tanzania-080526")

# The v3 vocabulary: density gains `No Cells` and `Few Cells` below `Sparser`, replacing v2.2's
# empty-field gate with an annotated distinction. The pool being re-labelled here was annotated
# under the old 5-level vocabulary, so agreement is scored after collapsing the two new rungs
# back to `Sparser` (see _v3_common.collapse_to_v2_density) and the redistribution of the old
# `Sparser` FOVs is reported separately.
TAG_HELP = [
    "density (pick exactly one):",
    "    No Cells        nothing to judge -- blank field, or debris only",
    "    Few Cells       cells countable one by one; too few to have a packing regime",
    "    Sparser         a real but thin monolayer; packing is judgeable",
    "    Monolayer | Slightly Dense | Dense | Very Dense",
    "overlap (optional; omit for none. Not allowed on No Cells):",
    "    Slight Rouleaux | Some Rouleaux | Rouleaux | Heavy Rouleaux",
    "quality (optional, repeatable):",
    "    Crenated | Unfocused | Overexposed | Artifact | Other Dimples",
]


def load_pool():
    rows = [r for r in csv.DictReader(open(MERGED_LABELS, encoding="utf-8"))
            if r["dataset"] in ELIGIBLE_DATASETS]
    if not rows:
        raise SystemExit(f"no eligible rows in {MERGED_LABELS}")
    return rows


def allocate(rows, n, floor):
    """How many FOVs to draw from each (density, overlap) cell.

    Proportional to cell frequency, then raised so every density level and every overlap level
    reaches `floor` in total, then trimmed back to exactly n from the largest cells.
    """
    cells = defaultdict(list)
    for r in rows:
        cells[(r["density_label"], r["overlap_label"])].append(r)

    total = len(rows)
    alloc = {c: max(1, round(n * len(rs) / total)) if len(rs) else 0
             for c, rs in cells.items()}

    def level_total(index, level):
        return sum(k for c, k in alloc.items() if c[index] == level)

    # Raise the largest cell of any under-represented level until the level clears the floor.
    for index, levels in ((0, DENSITY_LEVELS), (1, OVERLAP_LEVELS)):
        for level in levels:
            present = [c for c in cells if c[index] == level]
            if not present:
                continue
            while level_total(index, level) < floor:
                target = max(present, key=lambda c: len(cells[c]) - alloc[c])
                if alloc[target] >= len(cells[target]):
                    break
                alloc[target] += 1

    # Trim/grow to exactly n, always taking from (or giving to) the cell with the most headroom.
    while sum(alloc.values()) > n:
        target = max((c for c in alloc if alloc[c] > 1),
                     key=lambda c: alloc[c] / max(len(cells[c]), 1), default=None)
        if target is None:
            break
        alloc[target] -= 1
    while sum(alloc.values()) < n:
        target = max(cells, key=lambda c: len(cells[c]) - alloc[c])
        if alloc[target] >= len(cells[target]):
            break
        alloc[target] += 1

    return cells, alloc


def fetch_image(image_path, dest):
    """Copy one FOV to `dest`, from local disk or GCS. KTR-72502946 was never downloaded."""
    if str(image_path).startswith("gs://"):
        from src.pipeline import _gcs_client

        bucket_name, _, blob_name = str(image_path)[len("gs://"):].partition("/")
        _gcs_client().bucket(bucket_name).blob(blob_name).download_to_filename(str(dest))
    else:
        src = Path(image_path)
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copyfile(src, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=50, help="how many FOVs to draw")
    parser.add_argument("--floor", type=int, default=2,
                        help="minimum FOVs per density level and per overlap level")
    parser.add_argument("--seed", type=int, default=20260820,
                        help="fixes both the sample and the shuffled naming")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    rows = load_pool()
    print(f"eligible pool: {len(rows)} FOVs from {len(set(r['dataset'] for r in rows))} slides")

    cells, alloc = allocate(rows, args.n, args.floor)
    rng = random.Random(args.seed)

    picked = []
    for cell, k in sorted(alloc.items()):
        if k <= 0:
            continue
        available = cells[cell]
        k = min(k, len(available))
        for row in rng.sample(available, k):
            # cell frequency / cell sampling rate -- see the module docstring.
            weight = (len(available) / len(rows)) / (k / args.n)
            picked.append((row, cell, weight))

    rng.shuffle(picked)
    print(f"drew {len(picked)} FOVs across {len(set(c for _, c, _ in picked))} label cells")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    staging = out_dir / "blind-relabels"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()

    width = len(str(len(picked)))
    key_rows, template_rows = [], []
    for i, (row, cell, weight) in enumerate(picked, 1):
        blind_id = f"fov-{i:0{width}d}"
        dest = staging / f"{blind_id}.png"
        try:
            fetch_image(row["image_path"], dest)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"could not fetch {row['image_path']}: {exc}") from exc
        key_rows.append({
            "blind_id": blind_id,
            "fov_key": row["fov_key"],
            "dataset": row["dataset"],
            "filename": row["filename"],
            "original_density": row["density_label"],
            "original_overlap": row["overlap_label"],
            "cell_density": cell[0],
            "cell_overlap": cell[1],
            "sampling_weight": f"{weight:.6f}",
        })
        template_rows.append({"blind_id": blind_id, "tags": "", "notes": ""})
        if i % 10 == 0 or i == len(picked):
            print(f"  fetched {i}/{len(picked)}", flush=True)

    # The template ships inside the zip; the key never does.
    template_csv = staging / "annotations.csv"
    with open(template_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["blind_id", "tags", "notes"])
        writer.writeheader()
        writer.writerows(template_rows)

    readme = staging / "README.txt"
    readme.write_text(
        "Blind re-annotation set\n"
        "=======================\n\n"
        f"{len(picked)} FOVs, filenames randomised. Slide identity, FOV number and the previous\n"
        "labels are deliberately not recoverable from this folder -- that is the point, so please\n"
        "do not go looking for them before annotating.\n\n"
        "Fill in the `tags` column of annotations.csv, one row per image, comma-separated:\n\n"
        + "".join(f"  {line}\n" for line in TAG_HELP)
        + "\nExamples:  Slightly Dense, Some Rouleaux, Crenated\n"
        "           Few Cells, Slight Rouleaux\n"
        "           No Cells\n\n"
        "The density question to ask yourself is 'can I judge how packed this field is?'\n"
        "If there is nothing there at all, No Cells. If you could count the cells individually,\n"
        "Few Cells. If it is a thin but genuine monolayer, Sparser.\n\n"
        "Leave `notes` for anything you want to flag. Send back annotations.csv only.\n",
        encoding="utf-8")

    zip_path = out_dir / "blind-relabels.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in sorted(staging.iterdir()):
            zf.write(path, arcname=f"blind-relabels/{path.name}")

    # Written last and outside the zip, so an accidental `zip -r` of the staging dir cannot
    # sweep it in.
    key_csv = out_dir / "blind-relabels-KEY.csv"
    with open(key_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)

    shutil.rmtree(staging)

    size_mb = zip_path.stat().st_size / 1e6
    print(f"\nwrote {zip_path}  ({size_mb:.0f} MB, {len(picked)} images + annotations.csv)")
    print(f"wrote {key_csv}  <- do NOT open before annotating")
    print("\ndensity in the drawn set: "
          f"{dict(Counter(r['original_density'] for r in key_rows))}")
    print("overlap in the drawn set: "
          f"{dict(Counter(r['original_overlap'] for r in key_rows))}")
    print("slides: " + str(dict(Counter(r["dataset"] for r in key_rows))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
