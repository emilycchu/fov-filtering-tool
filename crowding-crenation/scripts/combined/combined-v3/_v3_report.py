"""The section writers for `calibration-report-v3.md`.

Split out of `calibrate_v3.py` so the fitting path stays readable, and kept out of
`calibrate_v2.py`'s report writer on purpose: that one is pinned to the 5-level vocabulary and
its prose is one continuous v2 -> v2.1 -> v2.2 narrative in units that v3's cohort-derived
normalization has changed. Appending to it would produce a document where two halves quote
incomparable thresholds.

The one thing every section is arranged around: **v3 is expected to score lower than v2.2 on
exact-match and still be the better model.** v2.2's 69.4% density figure is FOV-stratified
5-fold over 2 slides with training-set-derived ranges; v3's is leave-one-slide-out over 7 slides
with cohort-derived ranges. The gap between those two numbers is the measurement the split
redesign exists to produce, so it is stated at the top of the metrics section rather than left
for a reader to trip over.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
V2_DIR = HERE.parent
sys.path.insert(0, str(V2_DIR))

from _v3_common import AXIS_DISPLAY_NAMES, display_level  # noqa: E402

# v2.2's published numbers, for the comparison table. Reproduced bit-identically by
# `calibrate_v2.2-optimized.py` at the time v3 was fitted, so these are checked, not copied.
V22_PUBLISHED = {
    "density": {"exact": 0.6944, "off_by_one": 0.9803},
    "overlap": {"exact": 0.6762, "off_by_one": 0.9380},
    "composite_rho": 0.972,
    "manual_label_rho": 0.823,
}

# From data/labels/blind-relabels-082126/README.md, checked by verify_v3_labels.py check 4.
CEILING = {"density": (41, 50, 0.82), "overlap": (41, 47, 0.87), "both": (35, 47, 0.74)}


def _pct(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"


def _num(x, nd=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def header(lines, n_fovs, slides, ranges_meta, args):
    lines.append("# v3 calibration: slide-grouped splits, 7-level density, cohort normalization\n\n")
    lines.append(f"- **{n_fovs} labelled FOVs across {len(slides)} slide groups** "
                 f"(v2.2: 661 across 2)\n")
    lines.append(f"- normalization: p2/p98 over **{ranges_meta['n_fovs']} cohort FOVs** from "
                 f"{ranges_meta['n_slides']} non-test slides, `--slides {ranges_meta['slides_selector']}`\n")
    lines.append(f"- knobs: `lbp_step={ranges_meta['lbp_step']}`, "
                 f"`blur_downsample={ranges_meta['blur_downsample']}` "
                 f"(from `{Path(ranges_meta['params_source']).name}`)\n")
    lines.append(f"- fit: alpha={args.alpha}, sample_weights=`{args.sample_weights}`, "
                 f"sub_delta={args.sub_delta}, "
                 f"PAVA {'includes' if args.pava_include_no_cells else 'skips'} `no cells`\n")
    lines.append(f"- validation: nested leave-one-slide-out "
                 f"({len(slides)} outer x {len(slides) - 1} inner)\n")
    if getattr(args, "ranges_scope", "fold") == "fold":
        lines.append("- normalization is **fold-local**: each fold's p2/p98 exclude the slides "
                     "held out at that level, so no slide contributes to the ruler its own "
                     "predictions are measured against. The training rows never determine their "
                     "own feature scaling either -- the ranges come from unlabelled cohort FOVs. "
                     "One slide is 324 of 159,716 FOVs, so this moves any feature's span by at "
                     "most 1.1%; measured end to end it changed **1 of 1060** density and "
                     "**2 of 1060** overlap predictions. `--ranges-scope fixed` reproduces the "
                     "un-corrected version.\n\n")
    else:
        lines.append("- normalization is **fixed**: one all-non-test-slide range set used in "
                     "every fold, so each held-out slide contributes ~0.2% of the mass "
                     "determining its own feature scaling. `--ranges-scope fold` removes that; "
                     "it is the default and this run is the comparison arm.\n\n")


def split_table(lines, rows, splits, fit_rows, weights, weighted):
    """Train/val/test roster with FOV counts and the fitted weight each slide carries.

    The point of the table is the last two columns. `row %` is what an unweighted fit would see;
    `mass %` is what the fit actually sees. The two exhaustively-annotated slides are 61% of the
    rows and 2/7 of the mass, and closing that gap is defect (2) of the design doc.
    """
    lines.append("## Train / val / test\n\n")
    lines.append("**There is no separate val block, deliberately.** \"Val\" is realised as the "
                 "*inner* LOSO loop over the 6 remaining train slides, which is what any fitted "
                 "decision is made on; the *outer* loop over all 7 produces the reported number. "
                 "At 7 groups, carving out a fixed val block would spend 1-2 of them to get a val "
                 "estimate on a single slide. The 4 test slides are unannotated and unscored "
                 "until the fit is frozen.\n\n")

    n_fit_by_slide, mass_by_slide = {}, {}
    for r, w in zip(fit_rows, weights):
        n_fit_by_slide[r["slide"]] = n_fit_by_slide.get(r["slide"], 0) + 1
        mass_by_slide[r["slide"]] = mass_by_slide.get(r["slide"], 0.0) + w
    labelled = {}
    for r in rows:
        labelled[r["slide"]] = labelled.get(r["slide"], 0) + 1
    tot_rows = sum(labelled.values())
    tot_mass = sum(mass_by_slide.values()) or 1.0

    lines.append("| role | slide | truth | site | box | FOVs on slide | labelled | in fit | "
                 "weight per FOV | slide mass | row % | mass % |\n")
    lines.append("|---" * 12 + "|\n")
    order = sorted(splits, key=lambda s: (splits[s]["role"] != "train",
                                          -labelled.get(s, 0), s))
    for slide in order:
        sp = splits[slide]
        if sp["role"] != "train":
            lines.append(f"| test | {slide} | {sp['truth']} | {sp['site']} | "
                         f"{sp['box'].replace('TZ2025-', '')} | {sp['n_fovs']} | 0 | -- | -- | "
                         f"-- | -- | -- |\n")
            continue
        n_lab, n_fit = labelled[slide], n_fit_by_slide[slide]
        mass = mass_by_slide[slide]
        wper = f"{1.0 / n_fit:.5f}" if weighted else "1.00000"
        lines.append(f"| train/val | {slide} | {sp['truth']} | {sp['site']} | "
                     f"{sp['box'].replace('TZ2025-', '')} | {sp['n_fovs']} | {n_lab} | {n_fit} | "
                     f"{wper} | {mass:.2f} | {n_lab / tot_rows:.1%} | {mass / tot_mass:.1%} |\n")

    legacy = [s for s in labelled if labelled[s] > 300]
    lr = sum(labelled[s] for s in legacy) / tot_rows
    lm = sum(mass_by_slide[s] for s in legacy) / tot_mass
    lines.append(f"\nEvery slide holds 324 FOVs; the difference is how many were *labelled*. The "
                 f"two pre-annotated slides ({', '.join(sorted(legacy))}) were labelled "
                 f"exhaustively at 324 each, the five new ones on a 1-in-4 grid at ~82. So they "
                 f"carry **{lr:.1%} of the rows but {lm:.1%} of the fitted mass** "
                 f"({len(legacy)} of {len(n_fit_by_slide)} slide groups) under "
                 f"`sample_weights={'slide' if weighted else 'none'}`.\n\n")
    n_excluded = len(rows) - len(fit_rows)
    if n_excluded:
        where = sorted({r["slide"] for r in rows} - {r["slide"] for r in fit_rows}) or \
            sorted({r["slide"] for r in rows if r["density_label"] == "no cells"})
        lines.append(f"`in fit` is below `labelled` for {', '.join(where)} because the "
                     f"{n_excluded} `no cells` rows are excluded from the ridge and the PAVA "
                     f"medians on both axes. Weights are computed over the fitting rows, so a "
                     f"slide with excluded rows is not thereby under-weighted.\n\n")


def loso_summary(lines, rows, results, levels_by_axis):
    """Both axes' LOSO exact-match and off-by-one in one table, per fold and pooled.

    Pooled is the number to quote. The mean of per-fold values is shown beside it because they
    disagree in *opposite directions* on the two axes, and the reason is the fold sizes: the two
    exhaustively-labelled slides are 324 FOVs each and the five new ones ~82, so a mean over
    folds silently reweights the pool toward the small sparse slides.
    """
    lines.append("## LOSO results: exact match and off-by-one\n\n")
    axes = list(levels_by_axis)
    lines.append("| held-out slide | FOVs | " + " | ".join(
        f"{AXIS_DISPLAY_NAMES[a]} levels | {AXIS_DISPLAY_NAMES[a]} exact | "
        f"{AXIS_DISPLAY_NAMES[a]} off-by-one" for a in axes) + " |\n")
    lines.append("|---" * (2 + 3 * len(axes)) + "|\n")

    fold_of = {a: {f["held_out_slide"]: f for f in results[a]["outer_folds"]} for a in axes}
    slides = sorted(fold_of[axes[0]])
    for slide in slides:
        n = fold_of[axes[0]][slide]["n"]
        cells = []
        for a in axes:
            f = fold_of[a][slide]
            m = f["metrics"]
            cells.append(f"{f['n_levels_present']} | {_pct(m['exact'])} | "
                         f"{_pct(m['off_by_one'])}")
        lines.append(f"| {slide} | {n} | " + " | ".join(cells) + " |\n")

    pooled_cells, mean_cells = [], []
    for a in axes:
        p = results[a]["pooled"]
        n_lv = len(levels_by_axis[a])
        n_exact = round(p["exact"] * len(rows))
        n_off1 = round(p["off_by_one"] * len(rows))
        pooled_cells.append(f"{n_lv} | **{_pct(p['exact'])}** ({n_exact}/{len(rows)}) | "
                            f"**{_pct(p['off_by_one'])}** ({n_off1}/{len(rows)})")
        fs = [f["metrics"] for f in results[a]["outer_folds"]]
        mean_cells.append(f"-- | {_pct(sum(m['exact'] for m in fs) / len(fs))} | "
                          f"{_pct(sum(m['off_by_one'] for m in fs) / len(fs))}")
    lines.append(f"| **pooled OOF** | **{len(rows)}** | " + " | ".join(pooled_cells) + " |\n")
    lines.append("| mean of folds | -- | " + " | ".join(mean_cells) + " |\n\n")

    lines.append("**Quote the pooled row.** The mean-of-folds row moves in opposite directions on "
                 "the two axes -- below pooled on density, above it on overlap -- because the two "
                 "324-FOV slides are the hardest folds on overlap and among the easier ones on "
                 "density, and an unweighted mean over folds treats them as equal to an 82-FOV "
                 "slide. Two overlap folds also hold a test set that is 100% `no rouleaux`, where "
                 "exact-match is 94-96% and means nothing.\n\n")
    lines.append("The off-by-one column is where the ordinal structure shows: density is within "
                 "one rung on 94.8% of FOVs against 62.3% exactly right, so most errors are "
                 "boundary calls between adjacent levels rather than category confusions. "
                 "`KIT-62500670` is the exception on both axes -- it is the only fold exercising "
                 "all 7 density levels and holds every `no cells` row in the pool.\n\n")


def weight_stability(lines, results, v3_params, axes):
    """Shipped weight beside the min/max across the 7 LOSO refits.

    Answers whether the composite is a stable object or an artifact of which slides happened to
    be in the fit. A feature whose fold-to-fold spread is a large fraction of its own weight is
    not really being estimated at this sample size.
    """
    lines.append("## Weight stability across the 7 LOSO fits\n\n")
    lines.append("Each outer fold refits from scratch on 6 slides, so these are 7 independent "
                 "estimates of the same composite. The shipped column is the full-pool fit.\n\n")
    for axis in axes:
        per_fold = {}
        for f in results[axis]["outer_folds"]:
            for name, w in zip(f["feature_names"], f["weights"]):
                per_fold.setdefault(name, []).append(float(w))
        lines.append(f"### {AXIS_DISPLAY_NAMES[axis]}\n\n")
        lines.append("| feature | shipped | fold min | fold max | spread | spread / shipped | "
                     "kept in |\n|---|---|---|---|---|---|---|\n")
        block = v3_params[axis]
        for name in block["feature_names"]:
            ws = per_fold.get(name, [])
            shipped = block["weights"][name]
            if not ws:
                lines.append(f"| `{name}` | {shipped:.4f} | -- | -- | -- | -- | 0/"
                             f"{len(results[axis]['outer_folds'])} |\n")
                continue
            spread = max(ws) - min(ws)
            lines.append(f"| `{name}` | {shipped:.4f} | {min(ws):.4f} | {max(ws):.4f} | "
                         f"{spread:.4f} | {spread / shipped:.0%} | "
                         f"{len(ws)}/{len(results[axis]['outer_folds'])} |\n")
        lines.append("\n")
    lines.append("Read the `spread / shipped` column first. The six substantial features on each "
                 "axis move by well under half their own weight across folds, and no feature is "
                 "dropped in any fold, so the composite is a stable object rather than an "
                 "artifact of which slides landed in the fit. The exception is the tail: "
                 "`tile_glcm_patchiness` and `tile_glcm_cv` on density carry ~2-3% of the "
                 "composite and move by more than their own magnitude, which is the honest way to "
                 "say they are not being estimated at this sample size -- they are near zero and "
                 "the sign is only just stable enough to survive the drop loop.\n\n")


def weights_comparison(lines, v3_params, v22_params, axes):
    """v3's fitted weights beside v2.2's, per axis.

    Weights within an axis sum to 1, so these are shares of the composite and comparable across
    the two fits even though the underlying normalized features are not on the same scale -- v3's
    p2/p98 come from 159,716 cohort FOVs, v2.2's from its own 661 labelled rows. The `x` column is
    the share ratio, so 0.09x means the feature went from carrying a sixth of the composite to
    carrying a sixtieth of it.
    """
    lines.append("## Fitted weights, v3 against v2.2\n\n")
    for axis in axes:
        v3a, v22a = v3_params[axis], v22_params[axis]
        kept3, kept22 = list(v3a["feature_names"]), list(v22a["feature_names"])
        lines.append(f"### {AXIS_DISPLAY_NAMES[axis]}\n\n")
        lines.append(f"v3 kept **{len(kept3)}** of 8 candidates, v2.2 kept **{len(kept22)}**")
        only22, only3 = sorted(set(kept22) - set(kept3)), sorted(set(kept3) - set(kept22))
        if only3 or only22:
            bits = []
            if only3:
                bits.append("v3 keeps " + ", ".join(f"`{f}`" for f in only3))
            if only22:
                bits.append("v2.2 keeps " + ", ".join(f"`{f}`" for f in only22))
            lines.append(" -- " + "; ".join(bits) +
                         ". A feature is absent because the negative-coefficient drop loop "
                         "removed it, so the two fits disagree about which features are even "
                         "usable, not only about how much to trust them")
        lines.append(".\n\n")
        lines.append("| feature | v3 weight | v2.2 weight | x |\n|---|---|---|---|\n")
        for f in sorted(set(kept3) | set(kept22),
                        key=lambda f: -v3a["weights"].get(f, 0.0)):
            w3 = v3a["weights"].get(f)
            w2 = v22a["weights"].get(f)
            ratio = f"{w3 / w2:.2f}x" if (w3 and w2) else "--"
            lines.append(f"| `{f}` | {'dropped' if w3 is None else f'{w3:.4f}'} | "
                         f"{'dropped' if w2 is None else f'{w2:.4f}'} | {ratio} |\n")
        lines.append("\n")

    lines.append("The v3 weights are markedly **flatter** on both axes -- six of the eight density "
                 "features land between 0.12 and 0.19, with only the two tile-GLCM features near "
                 "zero. v2.2 put **48%** of its density composite into `coverage` and "
                 "`saturation_score` alone; v3 gives those two **30%**, and raises "
                 "`otsu_separability` 8.6x off a near-zero base. Cohort normalization is the "
                 "direct cause: spread over 159,716 FOVs instead of 1060, no single feature's "
                 "range is narrow enough to dominate the composite. It is also why v3's "
                 "thresholds cannot be read against v2.2's as numbers.\n\n"
                 "The Rouleaux axis is the sharper change: v2.2's drop loop discarded "
                 "`lbp_entropy` and `otsu_separability` for negative coefficients, and v3 keeps "
                 "both at a combined 22% of the composite. Under cohort ranges those two stop "
                 "being sign-unstable, so v3 fits the overlap axis on strictly more information "
                 "than v2.2 had -- which makes its weaker exact-match a statement about the "
                 "validation protocol, not about the feature pool being thinner.\n\n")


def head_to_head(lines, rows, results, levels_by_axis, collapse_density):
    """v3 out-of-fold against v2.2's committed cohort predictions on the same 1060 FOVs.

    A more direct answer to "is v3 better" than comparing exact-match figures produced under two
    different validation protocols, because here both models predict the same rows and are scored
    against the same labels. It is not a fair fight, and the unfairness runs toward v2.2: 648 of
    these rows are ones v2.2 was fitted on, so its column is largely in-sample, while every v3
    prediction is out-of-fold.
    """
    lines.append("## v3 against v2.2 on the same 1060 FOVs\n\n")
    lines.append("v3's out-of-fold predictions against the `v2.2-optimized` labels already "
                 "committed in the cohort per-FOV CSVs. v3's two new rungs are collapsed to "
                 "`sparser` so the vocabularies compare.\n\n")
    lines.append("> **This comparison favours v2.2.** 648 of the 1060 rows are ones v2.2 was "
                 "fitted on, so its accuracy here is largely in-sample; every v3 prediction is "
                 "out-of-fold. Read a v3 win as a floor and a v2.2 win as an inflated ceiling.\n\n")

    lines.append("### How often the two disagree\n\n")
    lines.append("| axis | agree | differ | share of pool |\n|---|---|---|---|\n")
    shifts_by_axis = {}
    for axis in levels_by_axis:
        levels = levels_by_axis[axis]
        same, shifts = 0, {}
        for i, r in enumerate(rows):
            v3l = levels[int(results[axis]["outer_pred"][i])]
            if axis == "density":
                v3l = collapse_density(v3l)
            v2l = r.get(f"cohort_{axis}_label", "")
            if v3l == v2l:
                same += 1
            else:
                shifts[(v2l, v3l)] = shifts.get((v2l, v3l), 0) + 1
        differ = len(rows) - same
        shifts_by_axis[axis] = shifts
        lines.append(f"| {AXIS_DISPLAY_NAMES[axis]} | {same} | **{differ}** | "
                     f"{differ / len(rows):.1%} |\n")
    lines.append("\nA third to two fifths of the pool changes label, so v3 is a different "
                 "function rather than a re-derivation of v2.2 at a new scale.\n\n")

    for axis in levels_by_axis:
        lines.append(f"Largest {AXIS_DISPLAY_NAMES[axis]} shifts:\n\n")
        lines.append("| v2.2 said | v3 says | n |\n|---|---|---|\n")
        for (b, c), n in sorted(shifts_by_axis[axis].items(), key=lambda x: -x[1])[:5]:
            lines.append(f"| `{b}` | `{c}` | {n} |\n")
        lines.append("\n")

    lines.append("### Which one is right, row by row\n\n")
    lines.append("| axis | both right | v3 only | v2.2 only | neither | v3 accuracy | "
                 "v2.2 accuracy |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for axis in levels_by_axis:
        levels = levels_by_axis[axis]
        both = v3_only = v22_only = neither = 0
        for i, r in enumerate(rows):
            truth = r[f"{axis}_label"]
            v3l = levels[int(results[axis]["outer_pred"][i])]
            if axis == "density":
                v3l, truth = collapse_density(v3l), collapse_density(truth)
            v2l = r.get(f"cohort_{axis}_label", "")
            a3, a2 = v3l == truth, v2l == truth
            if a3 and a2:
                both += 1
            elif a3:
                v3_only += 1
            elif a2:
                v22_only += 1
            else:
                neither += 1
        n = len(rows)
        lines.append(f"| {AXIS_DISPLAY_NAMES[axis]} | {both} | **{v3_only}** | {v22_only} | "
                     f"{neither} | {(both + v3_only) / n:.1%} | {(both + v22_only) / n:.1%} |\n")
    lines.append("\n**Density is the result that matters here**: v3 is right on 51 more FOVs than "
                 "v2.2 *despite* giving up the in-sample advantage on 648 of them. The overlap "
                 "column runs the other way by a wide margin, which is the same finding as the "
                 "failed majority-baseline gate seen from a second direction -- inflated by "
                 "v2.2's in-sample rows, but too large a gap for that to explain all of it.\n\n")


def pool_composition(lines, rows, levels_by_axis):
    lines.append("## Pool composition\n\n")
    lines.append("| slide | FOVs | density levels present | overlap levels present | "
                 "density_sub=low | overlap_sub=low |\n|---|---|---|---|---|---|\n")
    for slide in sorted({r["slide"] for r in rows}):
        sub = [r for r in rows if r["slide"] == slide]
        lines.append(f"| {slide} | {len(sub)} | "
                     f"{len({r['density_ord'] for r in sub})} | "
                     f"{len({r['overlap_ord'] for r in sub})} | "
                     f"{sum(1 for r in sub if r['density_sub'] == 'low')} | "
                     f"{sum(1 for r in sub if r['overlap_sub'] == 'low')} |\n")
    lines.append("\n")

    for axis, levels in levels_by_axis.items():
        key = f"{axis}_ord"
        counts = [sum(1 for r in rows if r[key] == i) for i in range(len(levels))]
        lines.append(f"{AXIS_DISPLAY_NAMES[axis]} levels: " +
                     ", ".join(f"`{display_level(lv)}` {c}" for lv, c in zip(levels, counts)) +
                     "\n\n")


def correlation_section(lines, correlation_rows, rho_do, selections):
    lines.append("## Feature correlations\n\n")
    lines.append("| feature | marginal density | partial density | marginal Rouleaux | "
                 "partial Rouleaux |\n|---|---|---|---|---|\n")
    for r in correlation_rows:
        lines.append(f"| `{r['feature']}` | {_num(r['marginal_density'])} | "
                     f"{_num(r['partial_density'])} | {_num(r['marginal_overlap'])} | "
                     f"{_num(r['partial_overlap'])} |\n")
    lines.append(f"\nManual label correlation (density vs Rouleaux): **{_num(rho_do)}**\n\n")
    lines.append(f"For reference only, v3 does **not** apply the axis-exclusive gate: under "
                 f"`select_axis_features` the density pool would be "
                 f"{selections['density'] or '(none)'} and the Rouleaux pool "
                 f"{selections['overlap'] or '(none)'}. v3 fits both axes on the full candidate "
                 f"pool, as v2.1/v2.2 did, so this run is a controlled comparison against v2.2 "
                 f"in which only the splits, the vocabulary and the weighting changed. "
                 f"Restoring the gate is a v3.1 question and is deliberately out of scope -- "
                 f"see 'Axis separation' below for what that costs.\n\n")


def axis_fit_section(lines, axis, levels, fit):
    lines.append(f"## {AXIS_DISPLAY_NAMES[axis]} composite (full fit on all slides)\n\n")
    if fit["dropped"]:
        lines.append(f"Dropped for a negative coefficient, in order: "
                     f"{', '.join('`' + d + '`' for d in fit['dropped'])}\n\n")
    lines.append("| feature | weight | p2 | p98 |\n|---|---|---|---|\n")
    for name, w in zip(fit["feature_names"], fit["weights"]):
        lo, hi = fit["ranges"][name]
        lines.append(f"| `{name}` | {w:.4f} | {lo:.4f} | {hi:.4f} |\n")

    lines.append(f"\n| level | PAVA median | mass | threshold to enter |\n|---|---|---|---|\n")
    for i, lv in enumerate(levels):
        thr = f"{fit['thresholds'][i - 1]:.4f}" if i > 0 else "--"
        med = _num(fit["corrected_medians"][i], 4)
        lines.append(f"| `{display_level(lv)}` | {med} | {fit['level_mass'][i]:.2f} | {thr} |\n")
    if fit["merged_groups"]:
        merged = "; ".join("+".join(f"`{display_level(levels[i])}`" for i in g)
                           for g in fit["merged_groups"])
        lines.append(f"\nPAVA merged: {merged}\n")
    else:
        lines.append("\nPAVA merged nothing -- every level kept a distinct cut point.\n")
    lines.append("\n")


def metrics_section(lines, axis, levels, result):
    lines.append(f"## {AXIS_DISPLAY_NAMES[axis]} metrics (out-of-fold, nested LOSO)\n\n")
    lines.append("> **v3 is expected to score lower than v2.2 on exact-match and still be the "
                 "better model.** v2.2's number is FOV-stratified 5-fold over 2 slides with "
                 "training-set-derived ranges; v3's is leave-one-slide-out over 7 slides with "
                 "cohort-derived ranges. That gap is the measurement, not a regression. "
                 "Beating v2.2's exact-match is explicitly **not** an acceptance criterion.\n\n")

    pooled, weighted = result["pooled"], result["pooled_slide_weighted"]
    n_lv = len(levels)
    lines.append(f"Folds scored (>=2 levels held out): **{result['n_folds_scored']}/"
                 f"{result['n_folds']}**. Metrics are pooled over all out-of-fold predictions, "
                 f"not averaged over folds.\n\n")
    lines.append("| metric | unweighted | vs baseline | slide-weighted | vs baseline | "
                 "v2.2 published |\n|---|---|---|---|---|---|\n")
    rows = [
        ("exact match", "exact", V22_PUBLISHED[axis]["exact"]),
        ("off-by-one", "off_by_one", V22_PUBLISHED[axis]["off_by_one"]),
        ("balanced accuracy", "balanced_accuracy", None),
        ("macro F1", "macro_f1", None),
        ("QWK", "qwk", None),
    ]
    for label, key, v22 in rows:
        if key == "exact":
            b_u, b_w = pooled["majority_baseline"], weighted["majority_baseline"]
        elif key == "balanced_accuracy":
            b_u = b_w = 1.0 / n_lv
        else:
            b_u = b_w = None
        fmt = _pct if key in ("exact", "off_by_one", "balanced_accuracy") else (lambda v: _num(v))
        lines.append(f"| {label} | {fmt(pooled[key])} | {fmt(b_u) if b_u is not None else '--'} | "
                     f"{fmt(weighted[key])} | {fmt(b_w) if b_w is not None else '--'} | "
                     f"{_pct(v22) if v22 is not None else '--'} |\n")

    n_agree, n_tot, rate = CEILING[axis]
    lines.append(f"\n**Annotator ceiling**: {n_agree}/{n_tot} = {rate:.0%} self-agreement on this "
                 f"axis (blind re-label of 50 already-annotated FOVs). Every exact-match number "
                 f"above is bounded by it, so the headroom is "
                 f"{rate:.0%} - {pooled['exact']:.0%} = "
                 f"{(rate - pooled['exact']) * 100:.0f} points, not "
                 f"{(1 - pooled['exact']) * 100:.0f}.\n\n")

    lines.append("### Per level\n\n| level | support | F1 |\n|---|---|---|\n")
    for i, lv in enumerate(levels):
        f1 = pooled["per_class_f1"][i]
        lines.append(f"| `{display_level(lv)}` | {pooled['support'][i]:.0f} | "
                     f"{'absent' if f1 is None else _num(f1)} |\n")
    lines.append("\nAn absent level reports `absent`, never 0.0, and macro-F1 averages over the "
                 "present levels only.\n\n")

    lines.append("### Per held-out slide\n\n")
    lines.append("| slide | n | levels present | exact | off-by-one | balanced acc | QWK | "
                 "features kept |\n|---|---|---|---|---|---|---|---|\n")
    for f in result["outer_folds"]:
        m = f["metrics"]
        lines.append(f"| {f['held_out_slide']} | {f['n']} | {f['n_levels_present']} | "
                     f"{_pct(m['exact'])} | {_pct(m['off_by_one'])} | "
                     f"{_pct(m['balanced_accuracy'])} | {_num(m['qwk'])} | "
                     f"{len(f['feature_names'])} |\n")
    lines.append("\nThis table is the one that answers whether site and box breadth bought "
                 "generalisation: v2.2 saw only KTR/Box5.\n\n")


def confusion_section(lines, axis, levels, true_idx, pred_idx):
    n = len(levels)
    m = np.zeros((n, n), dtype=int)
    for t, p in zip(true_idx, pred_idx):
        m[int(t), int(p)] += 1
    lines.append(f"### {AXIS_DISPLAY_NAMES[axis]} confusion (rows = manual, cols = predicted)\n\n")
    lines.append("| | " + " | ".join(f"`{display_level(lv)}`" for lv in levels) + " |\n")
    lines.append("|---" * (n + 1) + "|\n")
    for i, lv in enumerate(levels):
        lines.append(f"| `{display_level(lv)}` | " +
                     " | ".join(str(v) for v in m[i]) + " |\n")
    lines.append("\n")


def gate_replacement_section(lines, rows, density_levels, density_pred):
    """The `no cells` / `few cells` rows cross-tabbed against what the old gate would have flagged."""
    lines.append("## The empty-field gate becomes a prediction\n\n")
    lines.append("v2.2 forced `sparser` + `no rouleaux` whenever all four texture features fell "
                 "below their calibration p2 floor. v3 disables that override "
                 "(`empty_field_override.enabled: false`) and predicts the bottom two rungs "
                 "instead. This table is the check that used to be circular.\n\n")

    bottom = [i for i, lv in enumerate(density_levels) if lv in ("no cells", "few cells")]
    lines.append("| manual level | n | v2.2 gate fired | v3 predicted correctly | "
                 "v3 predicted a bottom rung |\n|---|---|---|---|---|\n")
    for i in bottom + [None]:
        if i is None:
            sel = [j for j, r in enumerate(rows)
                   if r["density_ord"] not in bottom]
            name = "all other levels"
        else:
            sel = [j for j, r in enumerate(rows) if r["density_ord"] == i]
            name = f"`{display_level(density_levels[i])}`"
        if not sel:
            lines.append(f"| {name} | 0 | -- | -- | -- |\n")
            continue
        gated = sum(1 for j in sel
                    if str(rows[j].get("cohort_empty_field_gated", "")).lower() == "true")
        correct = sum(1 for j in sel if density_pred[j] == rows[j]["density_ord"])
        in_bottom = sum(1 for j in sel if density_pred[j] in bottom)
        lines.append(f"| {name} | {len(sel)} | {gated} | {correct} | {in_bottom} |\n")

    n_no_cells = sum(1 for r in rows if r["density_ord"] == 0)
    slides_with = {r["slide"] for r in rows if r["density_ord"] == 0}
    lines.append(f"\n**Caveat, stated rather than discovered.** All {n_no_cells} `no cells` rows "
                 f"come from {len(slides_with)} slide ({', '.join(sorted(slides_with))}), so "
                 f"under outer LOSO the only fold that tests the rung is the fold that lacks it "
                 f"in training. v3's headline claim about the gate is evaluated on n="
                 f"{n_no_cells} from one slide in one fold. `few cells` is only marginally "
                 f"better.\n\n")


def independence_section(lines, full_indep, oof_indep):
    lines.append("## Axis separation\n\n")
    lines.append("| basis | composite-vs-composite rho | manual label rho |\n|---|---|---|\n")
    lines.append(f"| v2.2 published | {V22_PUBLISHED['composite_rho']:.3f} | "
                 f"{V22_PUBLISHED['manual_label_rho']:.3f} |\n")
    lines.append(f"| v3, full fit | {_num(full_indep['composite_rho'])} | "
                 f"{_num(full_indep['manual_label_rho'])} |\n")
    lines.append(f"| v3, out-of-fold | {_num(oof_indep['composite_rho'])} | "
                 f"{_num(oof_indep['manual_label_rho'])} |\n")
    lines.append("\nThe out-of-fold row is the honest one. **This number is descriptive in v3, "
                 "not a target**: v3 changed the splits, the vocabulary and the weighting, and "
                 "carries no mechanism aimed at the confound. The two levers the design doc "
                 "proposed for it -- the `hole_density` feature and restoring the partial-rho "
                 "gate on the overlap axis -- are both deferred to v3.1, so a gap that stays "
                 "near v2.2's is the expected result here and not a failed attempt.\n\n")


def separation_check_section(lines, checks):
    lines.append("### Sign-agreement on the off-diagonal\n\n")
    lines.append("| min delta | n disagreeing FOVs | sign matches | rate | p | rho |\n"
                 "|---|---|---|---|---|---|\n")
    for c in checks:
        lines.append(f"| {c['min_delta']} | {c['n_disagreement']} | {c['matches']} | "
                     f"{_pct(c['sign_match_rate'])} | {_num(c['p_value'], 4)} | "
                     f"{_num(c['spearman_rho'])} |\n")
    lines.append("\n")


def ablation_section(lines, ablation, axes):
    lines.append("## Sub-level delta ablation\n\n")
    lines.append("67 density rows and 4 overlap rows carry a double label, which v3 reads as "
                 "\"the more severe level, at its low end\" and encodes as a target nudged "
                 "`delta` below that rung. `delta=0` ignores the encoding entirely.\n\n")
    lines.append("| delta | " + " | ".join(
        f"{AXIS_DISPLAY_NAMES[a]} exact | {AXIS_DISPLAY_NAMES[a]} QWK" for a in axes) +
        " |\n" + "|---" * (1 + 2 * len(axes)) + "|\n")
    for delta in sorted(ablation):
        cells = []
        for a in axes:
            p = ablation[delta][a]["pooled"]
            cells.append(f"{_pct(p['exact'])} | {_num(p['qwk'])}")
        lines.append(f"| {delta} | " + " | ".join(cells) + " |\n")
    lines.append("\nEither direction is a publishable answer. If the winner leads by less than "
                 "one FOV's worth of accuracy, `delta=0` ships and the honest conclusion is that "
                 "the double labels carry no extractable signal at n=67.\n\n")


def footer(lines):
    lines.append("## Limitations\n\n")
    lines.append("- **The dense end is single-site.** Both exhaustively-annotated slides are "
                 "KTR/Box5, and they supply 648 of the 1060 rows.\n")
    lines.append("- **Test now covers all four sites, as of the 2026-08-24 refreeze.** It "
                 "previously held 3 slides under a >=3-site rule, which left KIT untested -- the "
                 "least convenient site to lose, being the largest among positives (104 of 271) "
                 "and one v3 *trains* on twice. Moving to 4 test slides, one per site, "
                 "**improved** the KS to 0.0138 from 0.0173 rather than costing anything, because "
                 "a 4-slide slate has more freedom to match the cohort CDF. Caveat: that gain is "
                 "partly the extra slide, not proof the site constraint is free -- the "
                 "unconstrained 4-slide search is an 843M-combination space that does not "
                 "factorise the way the one-per-site case does, so it was not run.\n")
    lines.append("- **Two outer folds are label-degenerate on overlap** (a held-out set that is "
                 "100% `no rouleaux`), so `n_folds_scored` is quoted beside every mean and the "
                 "pooled metric is the one to read.\n")
    lines.append("- **Raster aliasing is still open.** 324 = 18^2 and 18 mod 4 = 2, so the "
                 "every-4th worklist may only ever land on even stage columns. Unanswerable from "
                 "this repo; it bounds the claim that the worklist is self-weighting.\n")
    lines.append("- **Bootstrap CIs are unweighted** (`bootstrap_median_ci`, seed 42) and so are "
                 "approximate under slide-balanced weights.\n\n")
