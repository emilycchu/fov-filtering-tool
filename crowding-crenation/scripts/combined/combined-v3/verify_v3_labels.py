"""Assert that the v3 label ingest changed nothing it was not supposed to change.

Four checks, in the order they would catch a mistake:

1. **Parser equivalence.** All 648 legacy rows must return `(density, False, overlap, False)`
   from `parse_tanzania_tags_v3`, where `(density, overlap)` is exactly what
   `_v2_common.parse_tanzania_tags` returns. This is the check that proves case-insensitivity
   and max-severity resolution are no-ops on legacy input -- the legacy pool contains no double
   labels and no non-canonical casing, so a difference here means the rewrite changed the
   meaning of an existing label rather than extending the vocabulary.

2. **Merged-file equivalence.** Join the legacy rows of `merged-labels-v3.csv` against the
   committed v2 merged CSV on `fov_key`. The file cannot be byte-identical -- the density
   ordinals shift by 2 by design and five columns are new -- so the assertion is stated on the
   fields that must not move, plus the exact relation on the one that must.

3. **Counts.** Every distribution the v3 write-up quotes, checked rather than hand-computed.

4. **Blind-relabels repair.** The case-sensitive parser scored `blind-relabels-annotations.txt`
   at 0/50 because the annotator wrote lowercase. Case-folding must repair it to 50/50, and the
   resulting agreement rates must reproduce the three numbers the entire v3 justification rests
   on: the 82% / 87% self-agreement ceiling and the 74% both-axes rate.

## Which v2 merged CSV check 2 joins against

`merged-labels-v2.2.csv`, not `merged-labels.csv`. Both are committed, and the latter is the
v2.0-era file: 337 rows, 13 initial-dataset plus only `KTR-72502948`, written before
`KTR-72502946` was annotated. `_v2_common.MERGED_LABELS_CSV` still points at it, which is why
the v2.2 chain passes `--labels-csv` explicitly. `merged-labels-v2.2.csv` is the 661-row file
`merge_labels_v2.py` reproduces byte-for-byte today, and the only one containing all 648 legacy
rows, so it is the only correct join target.

Usage:
    python scripts/combined/combined-v3/verify_v3_labels.py [--assert-counts]
"""
import argparse
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent
ROOT = V2_DIR.parent.parent
sys.path.insert(0, str(V2_DIR))

import _v2_common  # noqa: E402
from _v2_common import read_csv_dicts  # noqa: E402
from _v3_common import (  # noqa: E402
    DENSITY_LEVELS,
    OVERLAP_LEVELS,
    collapse_to_v2_density,
    has_quality_only,
    parse_tanzania_tags_v3,
    _DENSITY_LOOKUP,
    _OVERLAP_LOOKUP,
    _QUALITY_LOOKUP,
    _tag_parts,
)
from merge_labels_v3 import (  # noqa: E402
    COMBINEDV3_DATASET,
    COMBINEDV3_LABELS_DIR,
    LEGACY_SLIDE_OF,
    LEGACY_SOURCES,
    MERGED_LABELS_V3_CSV,
    EXCLUDED_CSV,
    load_splits,
)

MERGED_LABELS_V22_CSV = (ROOT / "data" / "results" / "density-rouleaux-v2"
                         / "merged-labels-v2.2.csv")
BLIND_ANNOTATIONS_TXT = (ROOT / "data" / "labels" / "blind-relabels-082126"
                         / "blind-relabels-annotations.txt")
BLIND_KEY_CSV = ROOT / "data" / "results" / "combined-v3" / "blind-relabels-KEY.csv"

# The density ordinal shifted by exactly this much: v3 prepended `no cells` and `few cells`
# below `sparser`, and levels are looked up by name, so every legacy label keeps its identity
# and gains 2.
ORD_SHIFT = len(DENSITY_LEVELS) - len(_v2_common.DENSITY_LEVELS)

# fov-18, fov-34 and fov-47 were originally `rouleaux`, and the annotator reports having
# forgotten that rung was on the worksheet. The instrument could not agree on those three no
# matter what was on the slide, so they are excluded from the overlap denominator rather than
# scored as disagreements -- counting them would measure the worksheet, not the annotator.
BLIND_OVERLAP_EXCLUDED = ("fov-18", "fov-34", "fov-47")

EXPECTED = {
    "raw_rows": 435,
    "quality_only": {"KIT-62500670": 14, "KIT-62500909": 1, "KTR-72502904": 2,
                     "NKR-72502156": 6, "RUB-72501818": 0},
    "new_per_slide": {"KIT-62500670": 83, "KIT-62500909": 82, "KTR-72502904": 82,
                      "NKR-72502156": 83, "RUB-72501818": 82},
    "new_rows": 412,
    "legacy_rows": 648,
    "total_rows": 1060,
    "n_slides": 7,
    "density_sub_low": 67,
    "overlap_sub_low_ids": {("KIT-62500670", 28), ("KIT-62500670", 64),
                            ("KIT-62500670", 152), ("KIT-62500670", 188)},
    "density_counts": [13, 24, 188, 565, 141, 79, 50],
    "overlap_counts": [743, 129, 66, 56, 66],
    "blind_parsed": 50,
    "blind_density_agree": 41,
    "blind_density_n": 50,
    "blind_overlap_agree": 41,
    "blind_overlap_n": 47,
    "blind_both_agree": 35,
    "blind_both_n": 47,
}


def _legacy_label_rows():
    """(source, fov_id, tags) for every row of the two legacy annotated CSVs, in file order."""
    for source in LEGACY_SOURCES:
        slide = LEGACY_SLIDE_OF[source]
        path = ROOT / "data" / "labels" / source / f"{slide}-annotated.csv"
        for r in read_csv_dicts(path):
            yield source, int(r["fov_id"]), r["tags"]


def _raw_new_rows():
    """(slide, fov_id, tags) for every row of the 5 combinedv3 label CSVs, in file order."""
    for path in sorted(COMBINEDV3_LABELS_DIR.glob("*_labels.csv")):
        slide = path.name.replace("_labels.csv", "")
        for r in read_csv_dicts(path):
            yield slide, int(r["fov_id"]), r["tags"]


def check_parser_equivalence(fail):
    n = 0
    for source, fov_id, tags in _legacy_label_rows():
        n += 1
        want_d, want_o = _v2_common.parse_tanzania_tags(tags)
        got = parse_tanzania_tags_v3(tags)
        if got != (want_d, False, want_o, False):
            fail(f"parser: {source} fov {fov_id} tags={tags!r}: v3 gave {got}, "
                 f"v2 gave ({want_d!r}, {want_o!r})")
    if n != EXPECTED["legacy_rows"]:
        fail(f"parser: read {n} legacy rows, expected {EXPECTED['legacy_rows']}")
    print(f"1. parser equivalence: {n} legacy rows, 0 mismatches")


def check_merged_equivalence(fail, v3_rows):
    v2_rows = {r["fov_key"]: r for r in read_csv_dicts(MERGED_LABELS_V22_CSV)}
    legacy = [r for r in v3_rows if r["dataset"] in LEGACY_SOURCES]

    joined = 0
    for r in legacy:
        v2 = v2_rows.get(r["fov_key"])
        if v2 is None:
            fail(f"merged: {r['fov_key']} is in v3 but not in {MERGED_LABELS_V22_CSV.name}")
            continue
        joined += 1
        for field in ("filename", "image_path", "density_label", "overlap_label"):
            if r[field] != v2[field]:
                fail(f"merged: {r['fov_key']} {field}: v3={r[field]!r} v2={v2[field]!r}")
        if int(r["density_ord"]) != int(v2["density_ord"]) + ORD_SHIFT:
            fail(f"merged: {r['fov_key']} density_ord: v3={r['density_ord']} "
                 f"v2={v2['density_ord']} (expected v2+{ORD_SHIFT})")
        if int(r["overlap_ord"]) != int(v2["overlap_ord"]):
            fail(f"merged: {r['fov_key']} overlap_ord: v3={r['overlap_ord']} "
                 f"v2={v2['overlap_ord']}")
        if r["density_sub"] or r["overlap_sub"]:
            fail(f"merged: {r['fov_key']} legacy row carries a sub flag: "
                 f"density_sub={r['density_sub']!r} overlap_sub={r['overlap_sub']!r}")

    if joined != EXPECTED["legacy_rows"]:
        fail(f"merged: joined {joined} legacy rows, expected {EXPECTED['legacy_rows']}")

    # The 13 v2-side orphans are the initial-dataset rows v3 drops on purpose -- 9 Liberia FOVs
    # and 4 single-FOV Tanzania slides. Any other orphan is a real omission.
    orphans = [k for k, v in v2_rows.items() if k not in {r["fov_key"] for r in legacy}]
    unexpected = [k for k in orphans if not k.startswith("initial-071626/")]
    if unexpected:
        fail(f"merged: v2 rows missing from v3 that are not initial-dataset: {unexpected}")
    print(f"2. merged-file equivalence: {joined} legacy rows joined against "
          f"{MERGED_LABELS_V22_CSV.name}, {len(orphans)} v2-side orphans "
          f"(all initial-071626, dropped by design), 0 v3-side orphans")


def check_counts(fail, v3_rows, splits):
    raw = list(_raw_new_rows())
    if len(raw) != EXPECTED["raw_rows"]:
        fail(f"counts: {len(raw)} raw new rows, expected {EXPECTED['raw_rows']}")

    unknown = Counter()
    overlap_no_density = []
    for slide, fov_id, tags in raw:
        for p in _tag_parts(tags):
            if p not in _DENSITY_LOOKUP and p not in _OVERLAP_LOOKUP and p not in _QUALITY_LOOKUP:
                unknown[p] += 1
        has_d = any(p in _DENSITY_LOOKUP for p in _tag_parts(tags))
        has_o = any(p in _OVERLAP_LOOKUP for p in _tag_parts(tags))
        if has_o and not has_d:
            overlap_no_density.append((slide, fov_id, tags))
    if unknown:
        fail(f"counts: unknown tags {dict(unknown)}")
    if overlap_no_density:
        fail(f"counts: rows with an overlap tag and no density tag (Step 1 not applied?): "
             f"{overlap_no_density}")

    qonly = Counter({s: 0 for s in EXPECTED["quality_only"]})
    for slide, _fov, tags in raw:
        if has_quality_only(tags):
            qonly[slide] += 1
    if dict(qonly) != EXPECTED["quality_only"]:
        fail(f"counts: quality-only per slide {dict(qonly)}, "
             f"expected {EXPECTED['quality_only']}")

    new = [r for r in v3_rows if r["dataset"] == COMBINEDV3_DATASET]
    per_slide = Counter(r["slide"] for r in new)
    if dict(per_slide) != EXPECTED["new_per_slide"]:
        fail(f"counts: new rows per slide {dict(per_slide)}, "
             f"expected {EXPECTED['new_per_slide']}")
    for name, got, want in [
        ("new rows", len(new), EXPECTED["new_rows"]),
        ("legacy rows", len(v3_rows) - len(new), EXPECTED["legacy_rows"]),
        ("total rows", len(v3_rows), EXPECTED["total_rows"]),
        ("distinct slides", len({r["slide"] for r in v3_rows}), EXPECTED["n_slides"]),
    ]:
        if got != want:
            fail(f"counts: {name} {got}, expected {want}")

    d_counts = [sum(1 for r in v3_rows if r["density_label"] == lv) for lv in DENSITY_LEVELS]
    o_counts = [sum(1 for r in v3_rows if r["overlap_label"] == lv) for lv in OVERLAP_LEVELS]
    if d_counts != EXPECTED["density_counts"]:
        fail(f"counts: density {d_counts}, expected {EXPECTED['density_counts']}")
    if o_counts != EXPECTED["overlap_counts"]:
        fail(f"counts: overlap {o_counts}, expected {EXPECTED['overlap_counts']}")

    d_sub = [r for r in v3_rows if r["density_sub"] == "low"]
    if len(d_sub) != EXPECTED["density_sub_low"]:
        fail(f"counts: density_sub=low {len(d_sub)}, expected {EXPECTED['density_sub_low']}")
    legacy_sub = [r["fov_key"] for r in d_sub if r["dataset"] in LEGACY_SOURCES]
    if legacy_sub:
        fail(f"counts: legacy rows carrying density_sub=low: {legacy_sub}")

    o_sub = {(r["slide"], int(r["fov_id"])) for r in v3_rows if r["overlap_sub"] == "low"}
    if o_sub != EXPECTED["overlap_sub_low_ids"]:
        fail(f"counts: overlap_sub=low at {sorted(o_sub)}, "
             f"expected {sorted(EXPECTED['overlap_sub_low_ids'])}")

    test_slides = {s for s, r in splits.items() if r["role"] == "test"}
    leaked = sorted({r["slide"] for r in v3_rows} & test_slides)
    if leaked:
        fail(f"counts: test slides present: {leaked}")
    from_initial = [r["fov_key"] for r in v3_rows if r["dataset"] == "initial-071626"]
    if from_initial:
        fail(f"counts: initial-dataset rows present: {from_initial[:5]}")

    excluded = read_csv_dicts(EXCLUDED_CSV)
    if len(excluded) != sum(EXPECTED["quality_only"].values()):
        fail(f"counts: {len(excluded)} excluded rows, "
             f"expected {sum(EXPECTED['quality_only'].values())}")

    print(f"3. counts: {len(raw)} raw / 0 unknown tags / 0 overlap-without-density / "
          f"{sum(qonly.values())} quality-only / {len(new)} new / "
          f"{len(v3_rows) - len(new)} legacy / {len(v3_rows)} total / "
          f"{len({r['slide'] for r in v3_rows})} slides")
    print(f"   density {d_counts}")
    print(f"   overlap {o_counts}")
    print(f"   density_sub=low {len(d_sub)} (0 legacy), overlap_sub=low {len(o_sub)}")


def _parse_blind_worksheet(path=BLIND_ANNOTATIONS_TXT):
    """blind_id -> written tags, from the `fov-NN: tags` lines of the worksheet.

    The match is anchored at column 0, which is what excludes the worksheet's `Examples:`
    block: those lines are indented. The anchor is load-bearing rather than cosmetic, because
    the examples reuse three real ids (fov-07, fov-08, fov-09), so an unanchored pattern would
    silently overwrite three real answers with example text and still report 50/50.
    """
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(fov-\d+):\s*(.+?)\s*$", line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def check_blind_relabels(fail):
    written = _parse_blind_worksheet()
    key = {r["blind_id"]: r for r in read_csv_dicts(BLIND_KEY_CSV)}

    parsed = {}
    for blind_id in sorted(key):
        tags = written.get(blind_id)
        if tags is None:
            fail(f"blind: no answer line for {blind_id}")
            continue
        try:
            d, _dsub, o, _osub = parse_tanzania_tags_v3(tags)
        except ValueError as e:
            fail(f"blind: {blind_id} tags={tags!r} did not parse: {e}")
            continue
        parsed[blind_id] = (d, o)

    if len(parsed) != EXPECTED["blind_parsed"]:
        fail(f"blind: parsed {len(parsed)}/{len(key)}, expected "
             f"{EXPECTED['blind_parsed']}/{len(key)}")

    d_agree = sum(1 for bid, (d, _o) in parsed.items()
                  if collapse_to_v2_density(d) == key[bid]["original_density"])
    scored = [bid for bid in parsed if bid not in BLIND_OVERLAP_EXCLUDED]
    o_agree = sum(1 for bid in scored if parsed[bid][1] == key[bid]["original_overlap"])
    both = sum(1 for bid in scored
               if collapse_to_v2_density(parsed[bid][0]) == key[bid]["original_density"]
               and parsed[bid][1] == key[bid]["original_overlap"])

    for name, got, want, n, want_n in [
        ("density", d_agree, EXPECTED["blind_density_agree"], len(parsed),
         EXPECTED["blind_density_n"]),
        ("overlap", o_agree, EXPECTED["blind_overlap_agree"], len(scored),
         EXPECTED["blind_overlap_n"]),
        ("both axes", both, EXPECTED["blind_both_agree"], len(scored),
         EXPECTED["blind_both_n"]),
    ]:
        if (got, n) != (want, want_n):
            fail(f"blind: {name} agreement {got}/{n}, expected {want}/{want_n}")

    print(f"4. blind relabels: {len(parsed)}/{len(key)} parsed (was 0/50 case-sensitive); "
          f"density {d_agree}/{len(parsed)} ({d_agree / len(parsed):.0%}), "
          f"overlap {o_agree}/{len(scored)} ({o_agree / len(scored):.0%}), "
          f"both {both}/{len(scored)} ({both / len(scored):.0%})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--assert-counts", action="store_true",
                    help="run check 3 as well (it reads the raw label CSVs, not just the merge)")
    args = ap.parse_args()

    failures = []
    fail = failures.append

    v3_rows = read_csv_dicts(MERGED_LABELS_V3_CSV)
    splits = load_splits()

    check_parser_equivalence(fail)
    check_merged_equivalence(fail, v3_rows)
    if args.assert_counts:
        check_counts(fail, v3_rows, splits)
    else:
        print("3. counts: skipped (pass --assert-counts)")
    check_blind_relabels(fail)

    if failures:
        print(f"\nFAILED: {len(failures)} problem(s)")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
