"""Choose the v3 train/val and test slides over the full 497-slide Tanzania cohort.

Two different objectives, deliberately not the same one:

  **Test is selected for distributional match.** Its job is an unbiased estimate of deployment
  performance, so the four slides are chosen to minimise the KS distance between their pooled
  per-FOV density CDF and the whole cohort's. Getting this wrong is what produced v2.2's central
  defect -- its two calibration slides sit at the 77th and 93rd percentile of the cohort they
  score, so the fit is centred on the wrong part of the distribution.

  **Train is selected for range coverage.** Its job is to let PAVA derive thresholds across the
  whole scale, so it takes one slide per cohort density quintile and prefers, within each
  quintile, the slide whose own FOVs are most spread out (within-slide std). A slide whose FOVs
  span 0.1-0.7 yields labels in several buckets; a tight slide yields near-duplicates.

Constraints both slates carry, and why:

  * **Truth.** The cohort is 271 positive / 226 negative, and negatives are measurably denser
    (median 0.320 vs 0.247, Mann-Whitney p=5e-06, and the gap holds *within every site*). A
    positives-only slate would repeat the workbook's bias in miniature, so both slates carry
    both classes.
  * **Site.** Site is the strongest structural variable (RUB median 0.187 vs NKR 0.379 among
    negatives) and it is confounded with truth -- negatives are NKR-heavy, positives KIT-heavy.
    **Both slates cover all four sites.** Test did not, until 2026-08-24: it was 3 slides under a
    >=3-site rule, which left KIT untested even though KIT is the largest site among positives
    (104 of 271) and train holds two KIT slides. Going to 4 test slides, one per site, *improves*
    the KS to 0.0138 from 0.0173 -- a 4-slide slate has more freedom to match the cohort CDF than
    a 3-slide one, and that slack more than absorbs the extra constraint. Forcing KIT into a
    3-slide slate would instead have cost 12.7% relative KS (0.0195). Pass `--no-test-all-sites`
    to reproduce the old rule.
  * **Box.** Boxes are scanning batches. Both slates spread across them so a batch effect cannot
    masquerade as a density effect.
  * **No catalog `test` slide enters v3 train.** The workbook's split is a parasite-annotation
    split and is unsuitable for crowding, but keeping its test slides out of our training set
    costs nothing and avoids cross-pipeline contamination. The reverse is harmless.

Already-labelled slides are fixed into train rather than re-selected: KTR-72502948 (positive) and
KTR-72502946 (negative) carry 646 annotated FOVs between them, which is half the v3 label pool.

Train slides are **pinned** by default from the existing roster (`--pin-train`). The five new
train slides are annotated, so they are data now rather than a search result -- re-deriving them
after the test pool changed could silently pick different slides and orphan 412 labelled FOVs.

Writes `slide-splits.csv` -- the committed roster every later v3 step filters on.

Usage:
    python scripts/combined/combined-v3/select_splits.py
    python scripts/combined/combined-v3/select_splits.py --no-test-all-sites --n-test 3
    python scripts/combined/combined-v3/select_splits.py --pin-train ''   # re-search train too
"""
import argparse
import csv
import itertools
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "tanzania-complete-081426"))

from _slide_common import CROWDING_FOV_DIR, read_csv_dicts, write_csv_atomic  # noqa: E402

OUT_DIR = ROOT / "data" / "results" / "combined-v3"
SUMMARY_CSV = OUT_DIR / "slide-summary-497.csv"
SPLITS_CSV = OUT_DIR / "slide-splits.csv"

# The two slides already annotated (646 FOVs). Fixed into train, never re-selected.
PRE_LABELLED = {"KTR-72502948": "positive", "KTR-72502946": "negative"}

# One labelled FOV each, from four different slides. Dropped from the v3 pool: they add no slide
# group, and KIT-62501048 is a catalog `test` slide. KIT-62500652 is additionally a negative.
SINGLETON_LABELLED = {"KIT-62501048", "KIT-62500652", "KIT-62501056", "KIT-62500666"}

SITES = ("KIT", "KTR", "NKR", "RUB")

GRID = np.linspace(0.0, 1.0, 201)
MIN_FOVS = 300

FIELDNAMES = ["slide_id", "role", "truth", "conflict", "site", "box", "region",
              "catalog_split", "n_fovs", "density_mean", "density_std", "density_bucket",
              "overlap_mean", "quintile", "source"]


def load_cohort(summary_csv, fov_dir):
    """Slide metadata plus each slide's per-FOV density scores, gated FOVs read as 0.0.

    The gated->0 convention is the one `aggregate_slides.summarize_crowding` uses for the slide
    mean, so the per-FOV CDF matched here and the slide means quoted everywhere else describe
    the same quantity.
    """
    meta = {r["slide_id"]: r for r in read_csv_dicts(summary_csv)}
    scores = {}
    for slide_id in meta:
        path = Path(fov_dir) / f"{slide_id}.csv"
        if not path.exists():
            continue
        vals = []
        for row in read_csv_dicts(path):
            if row.get("error"):
                continue
            gated = str(row.get("empty_field_gated", "")).strip().lower() == "true"
            try:
                vals.append(0.0 if gated else float(row["density_score"]))
            except (TypeError, ValueError):
                continue
        if vals:
            scores[slide_id] = np.asarray(vals, dtype=np.float64)
    return meta, scores


def hist_counts(values):
    return np.searchsorted(np.sort(values), GRID, side="right").astype(np.float64)


def select_test(meta, scores, eligible, n_test, n_pos, target_cdf):
    """Exhaustive search for the n_pos-positive / (n_test-n_pos)-negative slate closest to the
    cohort CDF. Batched in numpy because the candidate space is ~8M combinations."""
    pos = sorted(s for s in eligible if meta[s]["truth"] == "positive")
    neg = sorted(s for s in eligible if meta[s]["truth"] == "negative")
    n_neg = n_test - n_pos
    order = pos + neg
    idx_of = {s: i for i, s in enumerate(order)}
    H = np.stack([hist_counts(scores[s]) for s in order])
    N = np.array([len(scores[s]) for s in order], dtype=np.float64)
    site = {s: s.split("-")[0] for s in order}

    combos = []
    for pc in itertools.combinations(pos, n_pos):
        for nc in itertools.combinations(neg, n_neg):
            c = pc + nc
            if len({site[s] for s in c}) < 3:
                continue
            if len({meta[s]["box"] for s in c}) < 3:
                continue
            combos.append([idx_of[s] for s in c])
    if not combos:
        raise SystemExit("no test slate satisfies the site/box constraints")
    combos = np.asarray(combos, dtype=np.int32)
    print(f"  test search: {len(combos):,} candidate slates "
          f"({n_pos} positive + {n_neg} negative, >=3 sites, >=3 boxes)")

    best_ks, best = np.inf, None
    for start in range(0, len(combos), 20000):
        chunk = combos[start:start + 20000]
        num = H[chunk].sum(axis=1)
        den = N[chunk].sum(axis=1)[:, None]
        ks = np.abs(num / den - target_cdf).max(axis=1)
        i = int(ks.argmin())
        if ks[i] < best_ks:
            best_ks, best = float(ks[i]), [order[j] for j in chunk[i]]
    return best, best_ks


def select_test_all_sites(meta, scores, eligible, n_pos, target_cdf, sites=SITES,
                          require_distinct_boxes=True):
    """One slide per site -- the only way `len(sites)` slides can cover `len(sites)` sites.

    Brute force is not an option here: 100 KIT-positive x 28 KTR-positive x 78 NKR-negative x 58
    RUB-negative is 12.7M slates for one truth pattern and 73M across all six, times a 201-point
    grid. But KS depends on the slate only through the pooled histogram and the pooled FOV count,
    and both are plain sums over the four slides. So the search factorises: enumerate
    (site0, site1) pairs and (site2, site3) pairs separately, then every full slate is one pair
    from each. That is ~4k iterations over a ~7k x 201 array instead of 73M x 201.

    `require_distinct_boxes` asks for one box per slide as well as one site per slide. Boxes are
    scanning batches, so spreading across them stops a batch effect from masquerading as a
    density effect. Measured, it costs nothing: the constrained and unconstrained optima are the
    same slate.
    """
    all_boxes = sorted({meta[s]["box"] for s in eligible})
    box_bit = {b: 1 << i for i, b in enumerate(all_boxes)}
    site_of = {s: s.split("-")[0] for s in eligible}
    pool = {st: {t: sorted(s for s in eligible
                           if site_of[s] == st and meta[s]["truth"] == t)
                 for t in ("positive", "negative")} for st in sites}

    def pairs(slides_a, slides_b):
        H, N, M, L = [], [], [], []
        for a in slides_a:
            for b in slides_b:
                ma, mb = box_bit[meta[a]["box"]], box_bit[meta[b]["box"]]
                if require_distinct_boxes and ma == mb:
                    continue
                H.append(hist_counts(scores[a]) + hist_counts(scores[b]))
                N.append(float(len(scores[a]) + len(scores[b])))
                M.append(ma | mb)
                L.append((a, b))
        return (np.asarray(H, dtype=np.float64), np.asarray(N, dtype=np.float64),
                np.asarray(M, dtype=np.int64), L)

    best_ks, best = np.inf, None
    n_considered = 0
    for pos_sites in itertools.combinations(sites, n_pos):
        truth = {st: ("positive" if st in pos_sites else "negative") for st in sites}
        left = pairs(pool[sites[0]][truth[sites[0]]], pool[sites[1]][truth[sites[1]]])
        right = pairs(pool[sites[2]][truth[sites[2]]], pool[sites[3]][truth[sites[3]]])
        if not len(left[0]) or not len(right[0]):
            continue
        # Iterate the smaller block so the vectorised side is the larger one.
        if len(left[0]) > len(right[0]):
            left, right = right, left
        Hl, Nl, Ml, Ll = left
        Hr, Nr, Mr, Lr = right
        for i in range(len(Hl)):
            ok = (Mr & Ml[i]) == 0 if require_distinct_boxes else np.ones(len(Mr), bool)
            if not ok.any():
                continue
            n_considered += int(ok.sum())
            S, T = Hl[i] + Hr[ok], Nl[i] + Nr[ok]
            ks = np.abs(S / T[:, None] - target_cdf).max(axis=1)
            j = int(ks.argmin())
            if ks[j] < best_ks:
                best_ks = float(ks[j])
                best = list(Ll[i]) + list(Lr[int(np.flatnonzero(ok)[j])])
    if best is None:
        raise SystemExit("no all-sites test slate satisfies the constraints")
    print(f"  test search: {n_considered:,} candidate slates "
          f"({n_pos} positive + {len(sites) - n_pos} negative, one slide per site"
          f"{', distinct boxes' if require_distinct_boxes else ''})")
    return best, best_ks


def select_train(meta, scores, eligible, n_train, edges):
    """One slide per cohort density quintile, maximising total within-slide spread, subject to
    covering all four sites, >=4 boxes, and >=2 of each truth class."""
    means = {s: float(meta[s]["density_mean"]) for s in eligible}
    stds = {s: float(meta[s]["density_std"] or 0.0) for s in eligible}
    bins = []
    for i in range(n_train):
        lo, hi = edges[i], edges[i + 1]
        members = [s for s in eligible if lo <= means[s] <= hi]
        # Only the most-spread candidates in each bin can win, so trimming keeps the product
        # tractable without changing the optimum.
        bins.append(sorted(members, key=lambda s: -stds[s])[:18])

    best = None
    for combo in itertools.product(*bins):
        if len(set(combo)) < n_train:
            continue
        if len({s.split("-")[0] for s in combo}) < 4:
            continue
        if len({meta[s]["box"] for s in combo}) < 4:
            continue
        truths = Counter(meta[s]["truth"] for s in combo)
        if truths["positive"] < 2 or truths["negative"] < 2:
            continue
        score = sum(stds[s] for s in combo)
        if best is None or score > best[0]:
            best = (score, combo)
    if best is None:
        raise SystemExit("no train slate satisfies the constraints")
    return list(best[1]), best[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-train", type=int, default=5, help="new slides for train/val")
    parser.add_argument("--n-test", type=int, default=4, help="new slides for test")
    parser.add_argument("--test-positives", type=int, default=2,
                        help="how many of the test slides are positive (cohort is 55:45)")
    parser.add_argument("--test-all-sites", action="store_true", default=True,
                        help="require one test slide per site; needs --n-test == 4")
    parser.add_argument("--no-test-all-sites", dest="test_all_sites", action="store_false",
                        help="fall back to the >=3-sites constraint (the pre-2026-08-24 rule)")
    parser.add_argument("--pin-train", default=str(SPLITS_CSV),
                        help="reuse the train slides from this roster instead of re-selecting "
                             "them. The five new train slides are annotated, so they are data "
                             "now, not a search result -- re-deriving them could silently pick "
                             "different slides when the test pool changes. Pass '' to re-search.")
    parser.add_argument("--summary", default=str(SUMMARY_CSV))
    parser.add_argument("--fov-dir", default=str(CROWDING_FOV_DIR))
    args = parser.parse_args()

    meta, scores = load_cohort(args.summary, args.fov_dir)
    print(f"cohort: {len(meta)} slides, {len(scores)} with per-FOV scores")

    cohort = np.concatenate([scores[s] for s in scores])
    target_cdf = np.searchsorted(np.sort(cohort), GRID, side="right") / len(cohort)
    print(f"cohort pooled FOVs: {len(cohort):,}, median {np.median(cohort):.3f}")

    catalog_test = {s for s, m in meta.items() if m.get("train_test_split") == "test"}
    eligible = {s for s in scores
                if s not in PRE_LABELLED and s not in SINGLETON_LABELLED
                and len(scores[s]) >= MIN_FOVS}
    print(f"eligible pool: {len(eligible)} slides "
          f"({sum(1 for s in eligible if meta[s]['truth'] == 'positive')} pos / "
          f"{sum(1 for s in eligible if meta[s]['truth'] == 'negative')} neg)")

    pinned = []
    if args.pin_train and Path(args.pin_train).exists():
        pinned = [r["slide_id"] for r in read_csv_dicts(args.pin_train)
                  if r["role"] == "train" and r["slide_id"] not in PRE_LABELLED]
        print(f"pinned train slides from {Path(args.pin_train).name}: {pinned}")

    # Pinned train slides must not be candidates for test -- they are already annotated.
    eligible -= set(pinned)

    if args.test_all_sites:
        if args.n_test != len(SITES):
            raise SystemExit(f"--test-all-sites needs --n-test {len(SITES)}, got {args.n_test}")
        test, ks = select_test_all_sites(meta, scores, eligible, args.test_positives, target_cdf)
    else:
        test, ks = select_test(meta, scores, eligible, args.n_test, args.test_positives,
                               target_cdf)
    print(f"  -> KS {ks:.4f}")

    edges = np.quantile([float(m["density_mean"]) for m in meta.values()],
                        np.linspace(0, 1, args.n_train + 1))
    if pinned:
        train, spread = pinned, sum(float(meta[s]["density_std"] or 0.0) for s in pinned)
        print(f"  train pinned: total within-slide std {spread:.3f}")
    else:
        train_pool = eligible - set(test) - catalog_test
        train, spread = select_train(meta, scores, train_pool, args.n_train, edges)
        print(f"  train search: total within-slide std {spread:.3f}")
    overlap = sorted(set(train) & set(test))
    if overlap:
        raise SystemExit(f"train and test overlap: {overlap}")

    rows = []
    for slide_id, source in ([(s, "already annotated") for s in PRE_LABELLED]
                             + [(s, "new") for s in train]):
        m = meta[slide_id]
        rows.append({"slide_id": slide_id, "role": "train", "source": source})
    for slide_id in test:
        rows.append({"slide_id": slide_id, "role": "test", "source": "new"})

    quintile_of = {}
    for slide_id in [r["slide_id"] for r in rows]:
        d = float(meta[slide_id]["density_mean"])
        quintile_of[slide_id] = int(np.searchsorted(edges[1:-1], d))

    for row in rows:
        m = meta[row["slide_id"]]
        row.update({
            "truth": m["truth"], "conflict": m["conflict"], "site": m["site"],
            "box": m["box"], "region": m["region"],
            "catalog_split": m.get("train_test_split", ""),
            "n_fovs": len(scores[row["slide_id"]]),
            "density_mean": m["density_mean"], "density_std": m["density_std"],
            "density_bucket": m["density_bucket"], "overlap_mean": m["overlap_mean"],
            "quintile": quintile_of[row["slide_id"]],
        })
    rows.sort(key=lambda r: (r["role"] != "train", float(r["density_mean"])))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(SPLITS_CSV, FIELDNAMES, rows)

    for role in ("train", "test"):
        sel = [r for r in rows if r["role"] == role]
        print(f"\n=== {role.upper()} ({len(sel)} slides, "
              f"{sum(int(r['n_fovs']) for r in sel):,} FOVs) ===")
        for r in sel:
            print(f"  {r['slide_id']:15} {r['truth']:8} dens={float(r['density_mean']):.3f} "
                  f"std={float(r['density_std']):.3f} {r['density_bucket']:15} {r['box']} "
                  f"Q{r['quintile']} catalog={r['catalog_split'] or '-':14} {r['source']}")
        print(f"  sites {sorted({r['site'] for r in sel})}  "
              f"boxes {sorted({r['box'][-4:] for r in sel})}  "
              f"truth {dict(Counter(r['truth'] for r in sel))}")

    pooled = np.concatenate([scores[r["slide_id"]] for r in rows if r["role"] == "test"])
    print(f"\ntest pooled FOV median {np.median(pooled):.3f} vs cohort "
          f"{np.median(cohort):.3f}   KS {ks:.4f}")
    print(f"wrote {SPLITS_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
