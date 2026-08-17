"""Read the Tanzania slide list out of the annotatability catalog workbook.

The catalog is an .xlsx, and this project's venv has no openpyxl or pandas, so this is a
stdlib `zipfile` + `ElementTree` reader for the one sheet we need. It is deliberately narrow:
it reads a rectangular block of cells and returns rows, nothing more.

Four details of the OOXML format that this file exists to get right:

1. **Sheet order is not file order.** `xl/worksheets/sheet1.xml` is not necessarily the first
   tab. The sheet name maps to a relationship id in `xl/workbook.xml`, which maps to a target
   path in `xl/_rels/workbook.xml.rels`. Hardcoding `sheet1.xml` happens to work on this file
   today and would break silently if a tab were reordered.
2. **Blank cells are omitted entirely.** A `<row>` does not contain a `<c>` for every column,
   so cells must be keyed by the column letter parsed out of the `r` attribute ("AB12" -> AB)
   and read by letter. Reading positionally silently shifts every value right of a blank.
3. **Shared strings can be split into runs.** `<si>` holds either one `<t>` or several
   `<r><t>`, and may carry a `<rPh>` phonetic block whose `<t>` is *not* part of the value.
   So: direct `t` children plus `r/t` grandchildren, which excludes `rPh/t` by construction.
4. **The interesting values contain non-breaking spaces.** Region is `"Tanzania\xa0/ Kagera"`
   and 119 of the TRUTH cells are `"positive\xa0⚠"`. `.strip()` does not touch an interior
   NBSP and `== "positive"` fails on those 119 rows. Raw values are preserved here (they are
   what the source says) and `_clean()` is applied only where a value is being *classified*.

Usage:
    python scripts/tanzania-complete-081426/catalog.py --assert-counts
"""
import argparse
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent.parent

CATALOG_XLSX = (ROOT / "data" / "labels" / "tanzania-081426"
                / "Table_Annotatability__Parasite_Buckets_(VERSION_4)_Tanzania_Liberia_Uganda.xlsx")
SHEET_NAME = "Tanzania+ 3.0"

# Columns on the Tanzania sheet. The three sheets in this workbook have *different* column
# orders (TRAIN TEST SPLIT is I here, C on ELWA, F on Uganda), so these are sheet-specific.
COL_SLIDE = "B"
COL_TRUTH = "C"
COL_REGION = "G"
COL_SPLIT = "I"
COL_ANNOTATABLE = "P"

# The ANNOTATABLE column is an *ordinal* judgement of how workable a slide is for annotation, so
# it gets a rank as well as its raw string. Ranks run 1 (worst) to 4 (best) so that "higher is
# more annotatable" holds when it is correlated against anything.
ANNOTATABLE_RANKS = {
    "annotatable": 4,
    "annotatable (hard)": 3,
    "can spot annotate": 2,
    "hard": 1,
}
# Display order, worst to best -- matches the rank order above.
ANNOTATABLE_ORDER = ["hard", "can spot annotate", "annotatable (hard)", "annotatable"]

HEADER_SENTINEL = "SLIDE NAMES"  # located rather than assumed to be row 1

NS_MAIN = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
NS_REL = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"

# The hazard sign on 119 TRUTH cells. Per the workbook's own NOTES sheet it marks a
# PCR-vs-microscopy truth conflict, and those slides are excluded from the 90-slide split.
HAZARD = "⚠"

_COL_RE = re.compile(r"^([A-Z]+)")

# What a correct parse of the current workbook produces. Checked by --assert-counts so a
# changed workbook or a broken parser fails here rather than 87,804 FOVs later.
EXPECTED_N_SLIDES = 271
EXPECTED_TRUTH = {"positive": 152, f"positive {HAZARD}": 119}
EXPECTED_SPLIT = {"train": 23, "test": 22, "save_for_later": 45, "": 181}
EXPECTED_ANNOTATABLE = {"annotatable": 96, "hard": 106, "can spot annotate": 57,
                        "annotatable (hard)": 12}


def _clean(value):
    """Normalize for *classification* only -- NBSP to space, then strip.

    Never used on a value that gets written back out; see detail 4 in the module docstring.
    """
    return value.replace("\xa0", " ").strip()


def _sheet_target(zf, sheet_name):
    """Resolve a sheet's display name to its path inside the archive."""
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    rel_targets = {rel.get("Id"): rel.get("Target") for rel in rels}

    for sheet in workbook.iter(NS_MAIN + "sheet"):
        if sheet.get("name") == sheet_name:
            target = rel_targets[sheet.get(NS_REL + "id")].lstrip("/")
            return target if target.startswith("xl/") else "xl/" + target

    available = [s.get("name") for s in workbook.iter(NS_MAIN + "sheet")]
    raise KeyError(f"no sheet named {sheet_name!r}; workbook has {available}")


def _shared_strings(zf):
    """The shared-string table, joining multi-run strings and excluding phonetic blocks."""
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []

    table = []
    for si in ET.fromstring(zf.read("xl/sharedStrings.xml")):
        parts = [t.text or "" for t in si.findall(NS_MAIN + "t")]
        for run in si.findall(NS_MAIN + "r"):
            parts.extend(t.text or "" for t in run.findall(NS_MAIN + "t"))
        table.append("".join(parts))
    return table


def _cell_value(cell, shared):
    cell_type = cell.get("t")

    if cell_type == "inlineStr":
        is_el = cell.find(NS_MAIN + "is")
        return "".join(is_el.itertext()) if is_el is not None else ""

    value = cell.find(NS_MAIN + "v")
    if value is None or value.text is None:
        return ""
    if cell_type == "s":
        return shared[int(value.text)]
    return value.text


def _rows(zf, target, shared):
    """{row_number: {column_letter: value}}, blank cells simply absent."""
    sheet = ET.fromstring(zf.read(target))
    rows = {}
    for row in sheet.iter(NS_MAIN + "row"):
        cells = {}
        for cell in row.iter(NS_MAIN + "c"):
            ref = cell.get("r") or ""
            match = _COL_RE.match(ref)
            if match:
                cells[match.group(1)] = _cell_value(cell, shared)
        rows[int(row.get("r"))] = cells
    return rows


def load_tanzania_slides(xlsx_path=CATALOG_XLSX, sheet=SHEET_NAME):
    """One dict per catalog slide, in workbook order, deduplicated on slide_id.

    `truth` and `region` are the raw cell values, NBSPs and all. `truth_norm` and
    `truth_warn` are the classified forms. `train_test_split` is "" for the 181 unassigned
    slides rather than None, so it round-trips through CSV unchanged.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"catalog workbook not found at {xlsx_path}. It is untracked -- see "
            "data/labels/tanzania-081426/."
        )

    with zipfile.ZipFile(xlsx_path) as zf:
        shared = _shared_strings(zf)
        rows = _rows(zf, _sheet_target(zf, sheet), shared)

    header_rows = [n for n, cells in rows.items()
                   if _clean(cells.get(COL_SLIDE, "")).upper() == HEADER_SENTINEL]
    if not header_rows:
        raise ValueError(
            f"no header row found on {sheet!r}: expected {HEADER_SENTINEL!r} in column "
            f"{COL_SLIDE}"
        )
    first_data_row = min(header_rows) + 1

    slides = []
    seen = set()
    for number in sorted(rows):
        if number < first_data_row:
            continue
        cells = rows[number]
        slide_id = _clean(cells.get(COL_SLIDE, ""))
        if not slide_id or slide_id in seen:
            continue
        seen.add(slide_id)

        truth = cells.get(COL_TRUTH, "")
        annotatable = _clean(cells.get(COL_ANNOTATABLE, "")).lower()
        slides.append({
            "slide_id": slide_id,
            "truth": truth,
            "truth_norm": _clean(truth),
            "truth_warn": HAZARD in truth,
            "train_test_split": _clean(cells.get(COL_SPLIT, "")),
            "region": cells.get(COL_REGION, ""),
            "annotatable": annotatable,
            "annotatable_rank": ANNOTATABLE_RANKS.get(annotatable, ""),
        })
    return slides


def summarize(slides):
    return {
        "n_slides": len(slides),
        "truth": Counter(s["truth_norm"] for s in slides),
        "split": Counter(s["train_test_split"] for s in slides),
        "region": Counter(_clean(s["region"]) for s in slides),
        "n_warn": sum(1 for s in slides if s["truth_warn"]),
        "annotatable": Counter(s["annotatable"] for s in slides),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xlsx", default=str(CATALOG_XLSX))
    parser.add_argument("--sheet", default=SHEET_NAME)
    parser.add_argument("--assert-counts", action="store_true",
                        help="fail if the parse does not match the recorded expectations")
    parser.add_argument("--list", action="store_true", help="print one slide_id per line")
    args = parser.parse_args()

    slides = load_tanzania_slides(args.xlsx, args.sheet)

    if args.list:
        for slide in slides:
            print(slide["slide_id"])
        return 0

    stats = summarize(slides)
    print(f"slides: {stats['n_slides']} distinct")
    print(f"truth:  {dict(stats['truth'])}")
    print(f"split:  {dict(stats['split'])}")
    print(f"region: {dict(stats['region'])}")
    print(f"hazard-marked: {stats['n_warn']}")
    print(f"annotatable: {dict(stats['annotatable'])}")

    if not args.assert_counts:
        return 0

    failures = []
    if stats["n_slides"] != EXPECTED_N_SLIDES:
        failures.append(f"expected {EXPECTED_N_SLIDES} slides, got {stats['n_slides']}")
    if dict(stats["truth"]) != EXPECTED_TRUTH:
        failures.append(f"truth {dict(stats['truth'])} != expected {EXPECTED_TRUTH}")
    if dict(stats["split"]) != EXPECTED_SPLIT:
        failures.append(f"split {dict(stats['split'])} != expected {EXPECTED_SPLIT}")
    if dict(stats["annotatable"]) != EXPECTED_ANNOTATABLE:
        failures.append(f"annotatable {dict(stats['annotatable'])} != "
                        f"expected {EXPECTED_ANNOTATABLE}")
    unranked = [s["slide_id"] for s in slides if not s["annotatable_rank"]]
    if unranked:
        failures.append(f"{len(unranked)} slides have an unrecognized ANNOTATABLE value: "
                        f"{unranked[:5]}")

    # The hazard slides are exactly the unassigned ones -- the workbook's NOTES sheet says so,
    # and it is the one cross-field invariant available here to catch a column mix-up.
    assigned_warn = [s["slide_id"] for s in slides if s["truth_warn"] and s["train_test_split"]]
    if assigned_warn:
        failures.append(f"{len(assigned_warn)} hazard slides have a split: {assigned_warn[:5]}")

    if failures:
        print("\nFAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nOK: catalog parse matches expectations")
    return 0


if __name__ == "__main__":
    sys.exit(main())
