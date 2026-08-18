"""Does any way of aggregating a slide's ~324 FOV scores track annotatability better than the mean?

Slide-level crowding has only ever been compared to catalog annotatability one way: the **mean** of
a slide's per-FOV combined scores (`combined_score` in `slide-summary.csv`, plotted by
`plot_annotatability.py`). The mean discards everything about the shape of a slide's FOV
distribution, and nothing on record says it is the right summary. This sweeps 53 alternatives.

Four families, all computed on the same per-FOV combined score (`density_score + overlap_score`,
the slide-level sum convention from `aggregate_slides.py`):

  whole-slide   mean, median, range, stdev, MAD, IQR
  percentiles   p5, p10, ... p100 -- the quantile *value*, i.e. a slide's p80 is the score at or
                above 80% of that slide's own FOV scores. Not band averages, not cumulative subsets.
  fractions     fraction of FOVs at or above each of the 4 density and 4 Rouleaux bucket cuts. The
                only family whose functional form is not a mean or a quantile.
  trimmed means mean of the highest-scoring top X% of FOVs, X = 100, 95, ... 5. "Top" = most
                crowded. X=100 *is* the mean, kept as the sweep's anchor rather than duplicated.

**Every rho carries a bootstrap CI, and there is a permutation test on the largest |rho|.** That is
not decoration. At n=270 the 95% CI on a single rho is about +/-0.11, which is wider than the entire
spread of results -- so a sorted table of 53 rho values invites reading sampling noise as a ranking,
and the nominal winner of 53 tries is a selection artifact until tested against the max-|rho| null.

Gated (near-empty) FOVs contribute 0.0, matching the headline convention in `aggregate_slides.py`;
that is what makes the `mean` row reproduce `combined_score` exactly, which this script asserts as
its check that the FOV-reading path has not drifted from the aggregator. Because 34 slides have >5%
gated FOVs, every statistic is *also* computed with gated FOVs dropped instead of zeroed, reported
as `rho_ungated` so the sensitivity is visible rather than argued.

Excludes the two v2.2 calibration slides so nothing is scored on the data the model was fit to.
Note this drops exactly one slide, not two: `KTR-72502946` is not in the catalog and never had an
ANNOTATABLE value. 271 -> 270.

Usage:
    python scripts/tanzania-complete-081426/annotatability-permutations/build_permutations.py
"""
import argparse
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from _plot_common import CALIBRATION_SLIDES  # noqa: E402
from _slide_common import (  # noqa: E402
    ROOT,
    SLIDE_SUMMARY_CSV,
    CROWDING_FOV_DIR,
    read_csv_dicts,
    write_csv_atomic,
)
from aggregate_slides import _is_true, _mean, _num, _stdev  # noqa: E402
from catalog import CATALOG_XLSX, SHEET_NAME, load_tanzania_slides  # noqa: E402
from plot_annotatability import spearman  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import DENSITY_LEVELS, OVERLAP_LEVELS  # noqa: E402

RESULTS_DIR = ROOT / "data" / "results" / "annotatability-permutations"
METRICS_CSV = RESULTS_DIR / "slide-metrics.csv"
CORRELATIONS_CSV = RESULTS_DIR / "correlations.csv"

# Seeds and resample counts follow `calibrate_v2.py`'s BOOTSTRAP_B / CI_SEED discipline: fixed, so
# re-running produces byte-identical CSVs and a diff means a real change.
BOOTSTRAP_B = 2000
PERMUTATION_B = 2000
CI_SEED = 42
PERM_SEED = 43
ROUND = 6

PERCENTILES = list(range(5, 101, 5))            # p5 .. p100
TRIMS = list(range(95, 0, -5))                  # top 95% .. top 5%; top100 is `mean`
WHOLE_SLIDE = ["mean", "median", "range", "stdev", "mad", "iqr"]
BASELINE = "mean"                               # what every other statistic is compared against

# The 4 non-bottom cuts per axis. `>= sparser` / `>= no rouleaux` are omitted: every FOV clears the
# bottom bucket by definition, so those fractions are the constant 1.0 and have no correlation.
DENSITY_CUTS = DENSITY_LEVELS[1:]
OVERLAP_CUTS = OVERLAP_LEVELS[1:]


def percentile(sorted_values, p):
    """The value at or above p% of `sorted_values`, linearly interpolated between order statistics.

    Linear interpolation rather than nearest-rank because with ~324 FOVs the two differ by less than
    the score's own precision, and interpolation keeps the sweep monotone in p without special-casing.
    """
    if p <= 0:
        return sorted_values[0]
    if p >= 100:
        return sorted_values[-1]
    pos = (len(sorted_values) - 1) * p / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def top_mean(sorted_values, pct):
    """Mean of the highest-scoring `pct`% of FOVs. Trims from the bottom, so gated FOVs (which
    score 0) are the first thing discarded."""
    keep = max(int(round(len(sorted_values) * pct / 100.0)), 1)
    segment = sorted_values[len(sorted_values) - keep:]
    return statistics.fmean(segment)


def fraction_at_or_above(labels, levels, cut):
    """Fraction of FOVs whose bucket is `cut` or more severe."""
    at_or_above = set(levels[levels.index(cut):])
    return sum(1 for label in labels if label in at_or_above) / len(labels)


def statistic_names():
    """Every statistic key, in family order. Shared with `plot_permutations.py` so the two scripts
    cannot disagree about what was computed."""
    names = list(WHOLE_SLIDE)
    names += [f"p{p}" for p in PERCENTILES]
    names += [f"frac_density_ge_{cut.replace(' ', '_')}" for cut in DENSITY_CUTS]
    names += [f"frac_rouleaux_ge_{cut.replace(' ', '_')}" for cut in OVERLAP_CUTS]
    names += [f"top{pct}_mean" for pct in TRIMS]
    return names


def family_of(name):
    if name in WHOLE_SLIDE:
        return "whole-slide"
    if name.startswith("frac_"):
        return "fraction"
    if name.startswith("top"):
        return "trimmed-mean"
    return "percentile"


def sweep_param(name):
    """The x-axis value for the two swept families, so the curve figure can order them. None for
    the unordered families."""
    if family_of(name) == "percentile":
        return int(name[1:])
    if family_of(name) == "trimmed-mean":
        return int(name[3:-5])
    return None


def compute_statistics(scores, density_labels, overlap_labels):
    """All 53 statistics for one slide. `scores` need not be sorted."""
    values = sorted(scores)
    out = {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "range": values[-1] - values[0],
        "stdev": _stdev(values),
        "mad": statistics.median([abs(v - statistics.median(values)) for v in values]),
        "iqr": percentile(values, 75) - percentile(values, 25),
    }
    for p in PERCENTILES:
        out[f"p{p}"] = percentile(values, p)
    for cut in DENSITY_CUTS:
        out[f"frac_density_ge_{cut.replace(' ', '_')}"] = fraction_at_or_above(
            density_labels, DENSITY_LEVELS, cut)
    for cut in OVERLAP_CUTS:
        out[f"frac_rouleaux_ge_{cut.replace(' ', '_')}"] = fraction_at_or_above(
            overlap_labels, OVERLAP_LEVELS, cut)
    for pct in TRIMS:
        out[f"top{pct}_mean"] = top_mean(values, pct)
    return out


def load_slide_fovs(slide_id):
    """(gated-as-zero scores, gated-dropped scores, density labels, Rouleaux labels) for one slide.

    A gated FOV is a near-empty field: the composite is answering the wrong question there, so the
    headline convention reads it as zero severity rather than trusting its score. Both readings are
    returned so the sensitivity can be reported instead of assumed.
    """
    zeroed, dropped, density_labels, overlap_labels = [], [], [], []
    for row in read_csv_dicts(CROWDING_FOV_DIR / f"{slide_id}.csv"):
        if (row.get("error") or "").strip():
            continue
        density, overlap = _num(row["density_score"]), _num(row["overlap_score"])
        if density is None or overlap is None:
            continue
        gated = _is_true(row.get("empty_field_gated"))
        combined = density + overlap
        zeroed.append(0.0 if gated else combined)
        if not gated:
            dropped.append(combined)
        # A gated FOV's own labels are equally untrustworthy, so it reads as the bottom bucket on
        # both axes -- the same substitution `apply_empty_field_override` makes at scoring time.
        density_labels.append(DENSITY_LEVELS[0] if gated else row["density_label"])
        overlap_labels.append(OVERLAP_LEVELS[0] if gated else row["overlap_label"])
    return zeroed, dropped, density_labels, overlap_labels


def load_slides(summary_csv, xlsx, sheet):
    annot = {s["slide_id"]: s for s in load_tanzania_slides(xlsx, sheet)}
    slides, skipped = [], []
    for row in read_csv_dicts(summary_csv):
        slide_id = row["slide_id"]
        meta = annot.get(slide_id)
        if not meta or meta["annotatable_rank"] == "":
            skipped.append(slide_id)
            continue
        if slide_id in CALIBRATION_SLIDES:
            continue
        zeroed, dropped, density_labels, overlap_labels = load_slide_fovs(slide_id)
        if not zeroed:
            skipped.append(slide_id)
            continue
        slides.append({
            "slide_id": slide_id,
            "site": slide_id.split("-")[0],
            "annotatable": meta["annotatable"],
            "annotatable_rank": int(meta["annotatable_rank"]),
            "n_fovs": len(zeroed),
            "summary_combined_score": _num(row["combined_score"]),
            # Kept out of the CSV; only `check_consistency` needs them.
            "fov_min": min(zeroed),
            "fov_max": max(zeroed),
            "stats": compute_statistics(zeroed, density_labels, overlap_labels),
            "stats_ungated": compute_statistics(dropped or zeroed, density_labels, overlap_labels),
        })
    return slides, skipped


def pearson(xs, ys):
    n = len(xs)
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = (sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) ** 0.5
    return num / den if den else float("nan")


def ranks(values):
    """Tie-averaged ranks -- the same treatment `plot_annotatability.spearman` applies internally."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        mean_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            out[order[k]] = mean_rank
        i = j + 1
    return out


def _percentile_ci(draws):
    draws = sorted(draws)
    b = len(draws)
    return draws[int(0.025 * (b - 1))], draws[int(0.975 * (b - 1))]


def bootstrap_all(rank_values, columns, names, baseline=BASELINE, b=BOOTSTRAP_B, seed=CI_SEED):
    """Bootstrap every statistic's rho **and** its difference from the baseline, on shared resamples.

    Resampling slides (not residuals) matches the question: these 270 slides are a sample of a
    cohort, and the CI has to answer "would another 270 slides rank these statistics the same way".

    The reason all 53 statistics share each resample is the `delta` CI, which is the comparison that
    actually answers "does anything beat the mean". A statistic's own CI is ~+/-0.13 wide here --
    wider than the entire spread of results -- so comparing two marginal CIs can detect nothing. But
    these statistics are near-duplicates computed on the same slides, so rho and rho_baseline move
    together under resampling and the CI on their *difference* is an order of magnitude tighter. The
    paired difference is the powerful test; the marginal CI is only there to show the absolute
    uncertainty on each rho.
    """
    rng = random.Random(seed)
    n = len(rank_values)
    draws = {name: [] for name in names}
    deltas = {name: [] for name in names}
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        rx = ranks([rank_values[i] for i in idx])
        per_stat = {}
        for name in names:
            column = columns[name]
            per_stat[name] = pearson(rx, ranks([column[i] for i in idx]))
            draws[name].append(per_stat[name])
        base = per_stat[baseline]
        for name in names:
            # Signed rho is negative throughout (more crowding -> less annotatable), so "stronger"
            # means larger |rho|. Comparing magnitudes keeps the sign convention from inverting the
            # question for any statistic that happened to come out positive.
            deltas[name].append(abs(per_stat[name]) - abs(base))
    return ({name: _percentile_ci(draws[name]) for name in names},
            {name: _percentile_ci(deltas[name]) for name in names})


def max_abs_rho_permutation_p(rank_values, stat_columns, observed_max,
                              b=PERMUTATION_B, seed=PERM_SEED):
    """P(some statistic reaches |rho| >= observed_max | annotatability is unrelated to all of them).

    Testing 53 statistics and reporting the largest |rho| is a selection procedure, so the null has
    to be the *maximum* over 53 under shuffling, not a single correlation. Shuffling the ranks keeps
    each statistic's own distribution and the between-statistic correlation structure intact --
    which matters, because these 53 are near-duplicates of each other, so a Bonferroni-style
    correction assuming independence would be badly conservative.

    Only the annotatability vector is shuffled, so each statistic's rank-transform is computed once
    and reused; a shuffled rank vector is already a rank vector, so the inner loop is a Pearson on
    precomputed ranks rather than a fresh sort.
    """
    rng = random.Random(seed)
    ry_by_stat = [ranks(col) for col in stat_columns]
    rx = ranks(rank_values)
    hits = 0
    for _ in range(b):
        rng.shuffle(rx)
        if max(abs(pearson(rx, ry)) for ry in ry_by_stat) >= observed_max:
            hits += 1
    return (hits + 1) / (b + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-csv", default=str(SLIDE_SUMMARY_CSV))
    parser.add_argument("--xlsx", default=str(CATALOG_XLSX))
    parser.add_argument("--sheet", default=SHEET_NAME)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_B)
    parser.add_argument("--permutations", type=int, default=PERMUTATION_B)
    args = parser.parse_args()

    slides, skipped = load_slides(args.summary_csv, args.xlsx, args.sheet)
    print(f"slides: {len(slides)}  (skipped {len(skipped)} without an ANNOTATABLE value; "
          f"excluded calibration {', '.join(CALIBRATION_SLIDES)})")
    if not slides:
        raise SystemExit("no slides with both a summary row and an ANNOTATABLE value")

    names = statistic_names()
    check_consistency(slides, names)

    results_dir = Path(args.results_dir)
    metrics_path = results_dir / METRICS_CSV.name
    fieldnames = (["slide_id", "site", "annotatable", "annotatable_rank", "n_fovs"] + names)
    write_csv_atomic(metrics_path, fieldnames, [
        dict({"slide_id": s["slide_id"], "site": s["site"], "annotatable": s["annotatable"],
              "annotatable_rank": s["annotatable_rank"], "n_fovs": s["n_fovs"]},
             **{n: round(s["stats"][n], ROUND) for n in names})
        for s in slides
    ])
    print(f"wrote {metrics_path}  ({len(slides)} slides x {len(names)} statistics)")

    rank_values = [s["annotatable_rank"] for s in slides]
    columns = {n: [s["stats"][n] for s in slides] for n in names}
    columns_ungated = {n: [s["stats_ungated"][n] for s in slides] for n in names}

    rows = []
    rho = {n: spearman(rank_values, columns[n]) for n in names}
    baseline_rho = rho[BASELINE]
    print(f"\nbootstrapping {args.bootstrap} resamples x {len(names)} statistics...")
    rho_cis, delta_cis = bootstrap_all(rank_values, columns, names, b=args.bootstrap)
    for name in names:
        lo, hi = rho_cis[name]
        dlo, dhi = delta_cis[name]
        rho_ungated = spearman(rank_values, columns_ungated[name])
        rows.append({
            "statistic": name,
            "family": family_of(name),
            "sweep_param": sweep_param(name) if sweep_param(name) is not None else "",
            "n": len(slides),
            "rho": round(rho[name], ROUND),
            "rho_ci_lo": round(lo, ROUND),
            "rho_ci_hi": round(hi, ROUND),
            "rho_ci_width": round(hi - lo, ROUND),
            # abs(rho) - abs(rho_mean): positive means stronger than the baseline.
            "delta_vs_mean": round(abs(rho[name]) - abs(baseline_rho), ROUND),
            "delta_ci_lo": round(dlo, ROUND),
            "delta_ci_hi": round(dhi, ROUND),
            # The verdict. A paired CI on the difference that excludes 0 is the only evidence here
            # that would justify preferring a statistic over the mean.
            "differs_from_mean": not (dlo <= 0.0 <= dhi),
            "pearson_r": round(pearson(rank_values, columns[name]), ROUND),
            "rho_ungated": round(rho_ungated, ROUND),
            "rho_ungated_delta": round(rho_ungated - rho[name], ROUND),
        })

    observed_max = max(abs(rho[n]) for n in names)
    best = max(names, key=lambda n: abs(rho[n]))
    print(f"permuting {args.permutations} shuffles against max |rho| = {observed_max:.4f} ({best})...")
    perm_p = max_abs_rho_permutation_p(rank_values, [columns[n] for n in names], observed_max,
                                       b=args.permutations)
    for row in rows:
        row["max_abs_rho_perm_p"] = round(perm_p, 6)

    correlations_path = results_dir / CORRELATIONS_CSV.name
    write_csv_atomic(correlations_path, list(rows[0].keys()), rows)
    print(f"wrote {correlations_path}")

    report(rows, slides, baseline_rho, best, observed_max, perm_p)
    return 0


def check_consistency(slides, names):
    """Assertions that would catch a silently wrong statistic. The first one is the important one:
    it ties this script's FOV-reading path to `aggregate_slides.py`'s, so the sweep cannot quietly
    diverge from the `combined_score` everything else in this dataset is built on."""
    for s in slides:
        stats = s["stats"]
        summary = s["summary_combined_score"]
        if summary is not None and abs(stats["mean"] - summary) > 1e-4:
            raise AssertionError(
                f"{s['slide_id']}: mean {stats['mean']:.6f} != slide-summary combined_score "
                f"{summary:.6f} -- the FOV-reading path has diverged from aggregate_slides.py")
        if abs(stats["p50"] - stats["median"]) > 1e-9:
            raise AssertionError(f"{s['slide_id']}: p50 != median")
        if abs(stats["p100"] - s["fov_max"]) > 1e-9:
            raise AssertionError(f"{s['slide_id']}: p100 != slide max")
        if abs(stats["range"] - (s["fov_max"] - s["fov_min"])) > 1e-9:
            raise AssertionError(f"{s['slide_id']}: range != max - min")
        percentile_values = [stats[f"p{p}"] for p in PERCENTILES]
        if any(b < a - 1e-9 for a, b in zip(percentile_values, percentile_values[1:])):
            raise AssertionError(f"{s['slide_id']}: percentiles not non-decreasing in p")
        # Trimming from the bottom can only raise a mean, so as X shrinks the top-X% mean must rise.
        # TRIMS runs 95..5, i.e. X descending, so the sequence must be non-decreasing.
        trim_values = [stats[f"top{pct}_mean"] for pct in TRIMS]
        if any(b < a - 1e-9 for a, b in zip(trim_values, trim_values[1:])):
            raise AssertionError(f"{s['slide_id']}: top-X% means not monotone in X")
        if trim_values[0] < stats["mean"] - 1e-9:
            raise AssertionError(f"{s['slide_id']}: top95_mean < mean (X=100 is the mean, so the "
                                 f"sweep must start at or above it)")
        for name in names:
            if name.startswith("frac_") and not 0.0 <= stats[name] <= 1.0:
                raise AssertionError(f"{s['slide_id']}: {name} outside [0, 1]")
    print(f"consistency checks passed on {len(slides)} slides "
          f"(mean == slide-summary combined_score, p50 == median, p100 == max, range == max-min, "
          f"percentiles monotone, top-X% monotone, fractions in [0,1])")


def report(rows, slides, baseline_rho, best, observed_max, perm_p):
    by_name = {r["statistic"]: r for r in rows}
    print(f"\nbaseline: {BASELINE} rho {baseline_rho:+.4f}  "
          f"CI [{by_name[BASELINE]['rho_ci_lo']:+.3f}, {by_name[BASELINE]['rho_ci_hi']:+.3f}]")
    print(f"strongest: {best} rho {by_name[best]['rho']:+.4f}  "
          f"delta vs mean {by_name[best]['delta_vs_mean']:+.4f}")
    print(f"max |rho| permutation p = {perm_p:.4f}  "
          f"(P that {len(rows)} statistics produce |rho| >= {observed_max:.4f} under no association)")

    stronger = [r for r in rows if r["differs_from_mean"] and r["delta_vs_mean"] > 0]
    weaker = [r for r in rows if r["differs_from_mean"] and r["delta_vs_mean"] < 0]
    print(f"\npaired test vs. the mean (95% CI on |rho| - |rho_mean| excluding 0):")
    print(f"  significantly STRONGER than the mean: "
          f"{', '.join(r['statistic'] for r in stronger) if stronger else 'none'}")
    print(f"  significantly weaker: {len(weaker)} statistics"
          + (f" ({', '.join(r['statistic'] for r in weaker[:6])}"
             f"{', ...' if len(weaker) > 6 else ''})" if weaker else ""))

    print("\nby family (rho range):")
    for family in ("whole-slide", "percentile", "fraction", "trimmed-mean"):
        group = [r for r in rows if r["family"] == family]
        lo = min(group, key=lambda r: abs(r["rho"]))
        hi = max(group, key=lambda r: abs(r["rho"]))
        print(f"  {family:14s} n={len(group):2d}  weakest {lo['statistic']:28s} {lo['rho']:+.4f}"
              f"   strongest {hi['statistic']:28s} {hi['rho']:+.4f}")

    print("\ngated sensitivity (largest |rho_ungated - rho|):")
    for row in sorted(rows, key=lambda r: -abs(r["rho_ungated_delta"]))[:5]:
        print(f"  {row['statistic']:28s} rho {row['rho']:+.4f} -> ungated "
              f"{row['rho_ungated']:+.4f}  ({row['rho_ungated_delta']:+.4f})")

    print("\nper site:")
    for site in sorted({s["site"] for s in slides}):
        group = [s for s in slides if s["site"] == site]
        crowding = [s["stats"][BASELINE] for s in group]
        within = spearman([s["annotatable_rank"] for s in group], crowding)
        print(f"  {site:5s} n={len(group):3d}  median crowding {statistics.median(crowding):.3f}"
              f"  within-site rho {within:+.4f}")


if __name__ == "__main__":
    sys.exit(main())
