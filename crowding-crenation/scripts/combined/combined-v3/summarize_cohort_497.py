"""Build the 497-slide Tanzania summary and answer whether negatives differ in crowding.

Everything this project has measured about the Tanzania cohort -- "61% of slides are sparser",
"none in the top bucket", the density CDF the v3 test slate is matched against, the
annotatability correlation -- came from the 271 slides in the annotatability workbook. That
workbook lists positives only. `gs://malaria-annotation-web/catalog.json` records 497 Tanzania
samples: 271 positive and 226 PCR-negative.

So the cohort description covers 55% of the dataset, and one question gates the v3 split:
**do the negatives sit somewhere else in density?** If they overlap the positives, the existing
slate selection stands and the negatives cost no annotation budget. If they don't, the test
slate has to be re-matched against a distribution it has never seen.

This script answers it, and writes the unified summary the selection then runs against:

  slide-summary-497.csv   one row per slide, same schema as the positives' slide-summary.csv
                          minus the fluorescence columns (those passes are positives-only)
  cohort-497-comparison.md  the positive-vs-negative comparison, with the test statistics

`summarize_crowding` is imported from the existing aggregator rather than reimplemented, so the
gated-FOV convention (a gated FOV contributes 0.0 severity) and the bucket-of-mean rule are
identical to the published positives numbers by construction, not by care.

Usage:
    python scripts/combined/combined-v3/summarize_cohort_497.py
"""
import argparse
import csv
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TZ_SCRIPTS = ROOT / "scripts" / "tanzania-complete-081426"
sys.path.insert(0, str(TZ_SCRIPTS))

from _slide_common import (  # noqa: E402
    CROWDING_FOV_DIR,
    RESULTS_DIR as TZ_RESULTS_DIR,
    read_csv_dicts,
    write_csv_atomic,
)
from aggregate_slides import summarize_crowding  # noqa: E402
from build_negatives_index import SLIDES_NEG_CSV, fetch_catalog_json  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import DEFAULT_SCORING_PARAMS  # noqa: E402

OUT_DIR = ROOT / "data" / "results" / "combined-v3"
SUMMARY_CSV = OUT_DIR / "slide-summary-497.csv"
COMPARISON_MD = OUT_DIR / "cohort-497-comparison.md"

FIELDNAMES = [
    "slide_id", "truth", "conflict", "site", "box", "region", "train_test_split",
    "n_fovs_scored", "n_errors_crowding", "n_empty_field_gated",
    "density_mean", "overlap_mean", "combined_score", "density_bucket", "overlap_bucket",
    "density_bucket_modal", "overlap_bucket_modal",
    "density_mean_raw", "overlap_mean_raw", "density_mean_ungated", "overlap_mean_ungated",
    "density_std", "overlap_std", "params_version",
]


def slide_roster():
    """{slide_id: row} for all 497 TZ slides, truth taken from catalog.json.

    catalog.json is the truth source for both halves, not just the negatives -- the workbook's
    TRUTH column carries whitespace corruption on 119 rows and has no entry at all for the 226
    negatives. The workbook is still consulted for `train_test_split`, which catalog.json's TZ
    records do not carry.
    """
    catalog = fetch_catalog_json()
    roster = {}
    for sample in catalog["samples"]:
        if sample.get("country") != "Tanzania" or sample.get("ignored"):
            continue
        truth = sample["sample_truth"]
        roster[sample["id"]] = {
            "slide_id": sample["id"],
            "truth": truth["verdict"],
            "conflict": truth.get("conflict", False),
            "site": sample["id"].split("-")[0],
            "box": sample.get("box") or "",
            "region": sample.get("site_label") or "",
            "train_test_split": "",
        }

    # The parasite-annotation split lives only in the workbook-derived slides.csv.
    positives_csv = TZ_RESULTS_DIR / "slides.csv"
    if positives_csv.exists():
        for row in read_csv_dicts(positives_csv):
            if row["slide_id"] in roster:
                roster[row["slide_id"]]["train_test_split"] = row.get("train_test_split", "")
    return roster


def load_params():
    with open(DEFAULT_SCORING_PARAMS, encoding="utf-8") as f:
        return json.load(f)


def mann_whitney_u(a, b):
    """Two-sided Mann-Whitney U with a normal approximation and tie correction.

    Hand-rolled rather than scipy so this script has no dependency the passes don't already
    have; the sample sizes here (226 vs 271) are far into the regime where the normal
    approximation is exact enough to quote.
    """
    import math

    combined = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i, n = {}, 0, len(combined)
    tie_term = 0.0
    while i < n:
        j = i
        while j + 1 < n and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = avg_rank
        t = j - i + 1
        if t > 1:
            tie_term += t ** 3 - t
        i = j + 1

    r_a = sum(ranks[k] for k, (_v, g) in enumerate(combined) if g == 0)
    na, nb = len(a), len(b)
    u_a = r_a - na * (na + 1) / 2
    mu = na * nb / 2
    sigma_sq = na * nb / 12 * ((n + 1) - tie_term / (n * (n - 1)))
    if sigma_sq <= 0:
        return float("nan"), float("nan"), float("nan")
    z = (u_a - mu) / math.sqrt(sigma_sq)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    # Rank-biserial: the probability a random positive outranks a random negative, rescaled to
    # [-1, 1]. Reported because a p-value on n=497 says "different", not "different enough".
    return z, p, 2 * u_a / (na * nb) - 1


def ks_two_sample(a, b):
    grid = sorted(set(a) | set(b))
    sa, sb = sorted(a), sorted(b)

    def cdf(sorted_vals, x):
        import bisect

        return bisect.bisect_right(sorted_vals, x) / len(sorted_vals)

    return max(abs(cdf(sa, x) - cdf(sb, x)) for x in grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fov-dir", default=str(CROWDING_FOV_DIR))
    args = parser.parse_args()

    roster = slide_roster()
    params = load_params()
    fov_dir = Path(args.fov_dir)
    print(f"roster: {len(roster)} Tanzania slides "
          f"({sum(1 for r in roster.values() if r['truth'] == 'positive')} positive, "
          f"{sum(1 for r in roster.values() if r['truth'] == 'negative')} negative)")

    rows, missing = [], []
    for slide_id, meta in sorted(roster.items()):
        csv_path = fov_dir / f"{slide_id}.csv"
        if not csv_path.exists():
            missing.append(slide_id)
            continue
        summary = summarize_crowding(read_csv_dicts(csv_path), params)
        out = dict(meta)
        out.update({k: v for k, v in summary.items() if k in FIELDNAMES})
        out["params_version"] = params.get("version", "")
        rows.append(out)

    print(f"scored slides found: {len(rows)}; missing: {len(missing)}")
    if missing:
        by_truth = Counter(roster[s]["truth"] for s in missing)
        print(f"  missing by truth: {dict(by_truth)}")
        print(f"  first 10: {missing[:10]}")

    write_csv_atomic(SUMMARY_CSV, FIELDNAMES, rows)
    print(f"wrote {SUMMARY_CSV}")

    pos = [float(r["density_mean"]) for r in rows
           if r["truth"] == "positive" and r["density_mean"] not in ("", None)]
    neg = [float(r["density_mean"]) for r in rows
           if r["truth"] == "negative" and r["density_mean"] not in ("", None)]
    # KTR-72502946 is a negative and has been scored since the tanzania-080526 run, so a bare
    # `if not neg` would let a one-slide "distribution" through into the comparison.
    if len(neg) < 10:
        print(f"\nOnly {len(neg)} negative(s) scored ({[r['slide_id'] for r in rows if r['truth'] == 'negative']}) "
              "-- summary written, comparison skipped until the negatives pass lands.")
        return 0

    z, p, rbc = mann_whitney_u(pos, neg)
    ks = ks_two_sample(pos, neg)

    def line(name, vals):
        q = st.quantiles(vals, n=4)
        return (f"| {name} | {len(vals)} | {min(vals):.3f} | {q[0]:.3f} | "
                f"{st.median(vals):.3f} | {q[2]:.3f} | {max(vals):.3f} | {st.mean(vals):.3f} |")

    md = [
        "# Tanzania cohort at full size: 497 slides, 271 positive + 226 negative\n\n",
        "Slide-level `density_mean` (gated FOVs contribute 0.0, the headline convention).\n\n",
        "| group | n | min | p25 | median | p75 | max | mean |\n",
        "|---|---|---|---|---|---|---|---|\n",
        line("positive", pos) + "\n",
        line("negative", neg) + "\n",
        line("all 497", pos + neg) + "\n\n",
        "## Do the negatives sit elsewhere?\n\n",
        f"- Mann-Whitney z = {z:.2f}, p = {p:.3g}\n",
        f"- rank-biserial correlation = {rbc:+.3f} "
        "(0 = a random positive is as likely to be denser as sparser than a random negative)\n",
        f"- two-sample KS distance = {ks:.3f}\n",
        f"- median difference = {st.median(pos) - st.median(neg):+.3f} "
        f"({st.median(pos):.3f} positive vs {st.median(neg):.3f} negative)\n\n",
    ]

    for name, vals in (("positive", pos), ("negative", neg)):
        counts = Counter(r["density_bucket"] for r in rows if r["truth"] == name)
        md.append(f"- {name} buckets: {dict(counts)}\n")
    md.append("\n")

    sites = sorted({r["site"] for r in rows})
    md.append("## Site is confounded with truth\n\n| site | n pos | n neg | "
              "median dens pos | median dens neg |\n|---|---|---|---|---|\n")
    for site in sites:
        p_v = [float(r["density_mean"]) for r in rows
               if r["site"] == site and r["truth"] == "positive" and r["density_mean"] != ""]
        n_v = [float(r["density_mean"]) for r in rows
               if r["site"] == site and r["truth"] == "negative" and r["density_mean"] != ""]
        md.append(f"| {site} | {len(p_v)} | {len(n_v)} | "
                  f"{st.median(p_v):.3f} | " if p_v else f"| {site} | 0 | {len(n_v)} | - | ")
        md.append(f"{st.median(n_v):.3f} |\n" if n_v else "- |\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_MD.write_text("".join(md), encoding="utf-8")

    print(f"\npositive density_mean: median {st.median(pos):.3f}  (n={len(pos)})")
    print(f"negative density_mean: median {st.median(neg):.3f}  (n={len(neg)})")
    print(f"Mann-Whitney z={z:.2f} p={p:.3g}  rank-biserial={rbc:+.3f}  KS={ks:.3f}")
    print(f"wrote {COMPARISON_MD}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
