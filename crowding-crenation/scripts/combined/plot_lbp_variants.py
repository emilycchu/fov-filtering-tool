"""Plot the LBP runtime/accuracy tradeoff across variants: what each one costs, what it buys.

Two panels rather than one, because runtime (seconds) and accuracy (a rate) are different
scales and a two-y-axis chart would invite reading a crossing point that means nothing. The
variant order is shared down both panels, so a row reads across as one variant.

Inputs are the artifacts of the two measurement scripts -- `runtime-bench.csv` from
`bench_lbp.py` and `variant-comparison.json` from `compare_lbp_variants.py` -- so the figure
cannot disagree with the numbers in the report.

Usage:
    python scripts/combined/plot_lbp_variants.py [--out PATH]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    COLOR_AXIS,
    COLOR_GRID,
    COLOR_MINE,
    COLOR_MUTED,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_SURFACE,
    LBP_RUNTIME_DIR,
    read_csv_dicts,
)

# Second categorical slot, same blue/orange pair the other plot scripts in this repo use.
# Validated light-mode by the dataviz validator: adjacent dE 24.7 protan, 33.6 normal vision.
COLOR_ROULEAUX = "#eb6834"

# Density's LBP weight is 0.065 and Rouleaux drops the feature entirely, so both axes are
# plotted -- the point of the figure is partly that one axis cannot move at all.
AXES = [("density", "Density", COLOR_MINE), ("overlap", "Rouleaux", COLOR_ROULEAUX)]


def variant_label(name):
    return {"skimage": "skimage (v2.2 today)", "exact": "exact (bit-identical)",
            "nolbp": "no LBP"}.get(name, f"stride {name[4:]}")


def median_runtime(timings, key):
    values = [float(r[key]) for r in timings if r.get(key)]
    return statistics.median(values) if values else None


def main():
    parser = argparse.ArgumentParser(description="Plot the LBP variant runtime/accuracy tradeoff.")
    parser.add_argument("--bench-csv", default=str(LBP_RUNTIME_DIR / "runtime-bench.csv"))
    parser.add_argument("--comparison-json", default=str(LBP_RUNTIME_DIR / "variant-comparison.json"))
    parser.add_argument("--out", default=str(LBP_RUNTIME_DIR / "variant-tradeoff.png"))
    args = parser.parse_args()

    timings = read_csv_dicts(args.bench_csv)
    comparison = json.loads(Path(args.comparison_json).read_text())
    refit = comparison["refit"]

    baseline = median_runtime(timings, "skimage_s")
    runtimes = {"exact": median_runtime(timings, "exact_s"), "nolbp": 0.0}
    for key in timings[0]:
        if key.startswith("step") and key.endswith("_s"):
            runtimes[key[:-2]] = median_runtime(timings, key)

    # Only variants that were both benchmarked and refit, so every row has both panels. The
    # stride sweep goes finer than this in variant-comparison.csv; the figure shows the
    # powers of two, which is enough to read the shape.
    strides = sorted(int(name[4:]) for name in runtimes if name.startswith("step"))
    order = ["skimage", "exact"] + [f"step{s}" for s in strides if f"step{s}" in refit] + ["nolbp"]
    order = [name for name in order
             if name == "skimage" or (name in refit and runtimes.get(name) is not None)]
    runtimes["skimage"] = baseline
    positions = list(range(len(order)))[::-1]  # fastest-to-slowest reads top-down

    fig, (ax_time, ax_acc) = plt.subplots(1, 2, figsize=(13.5, 5.0), dpi=150,
                                          gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.07})
    fig.patch.set_facecolor(COLOR_SURFACE)

    # --- panel 1: runtime per FOV (magnitude, one series, direct-labeled) ---
    ax_time.set_facecolor(COLOR_SURFACE)
    for name, y in zip(order, positions):
        seconds = runtimes.get(name)
        is_baseline = name == "skimage"
        ax_time.barh(y, seconds, height=0.45, linewidth=0, zorder=3,
                     color=COLOR_MUTED if is_baseline else COLOR_MINE)
        if is_baseline:
            note = "what v2.2 runs today"
        elif not seconds:
            note = "removed"
        else:
            note = f"{baseline / seconds:.0f}x faster"
        ax_time.annotate(f"  {seconds:.2f}s  ({note})", (seconds, y), va="center",
                         fontsize=9, color=COLOR_PRIMARY, zorder=4)

    ax_time.set_xlim(0, baseline * 1.62)
    ax_time.set_title("LBP runtime per 2800x2800 FOV", fontsize=11.5, color=COLOR_PRIMARY,
                      loc="left", pad=10)
    ax_time.set_xlabel("seconds (median of benchmarked FOVs)", color=COLOR_SECONDARY, fontsize=10)

    # --- panel 2: change in refit accuracy, as a delta so "nothing moves" is legible ---
    # Plotting the two rates directly wastes the axis on the constant ~1.8pt gap between the
    # density and Rouleaux composites; the question is only whether a variant moved either
    # one, so both are drawn against a shared zero.
    ax_acc.set_facecolor(COLOR_SURFACE)
    ax_acc.axvline(0, color=COLOR_AXIS, linewidth=1.5, zorder=2)
    for axis_key, _, color in AXES:
        exact_baseline = refit["exact"][axis_key]["oof_exact_match"]
        for name, y in zip(order, positions):
            if name == "skimage":
                continue  # identical to `exact` by construction; nothing to plot
            delta = 100 * (refit[name][axis_key]["oof_exact_match"] - exact_baseline)
            offset = 0.15 if axis_key == "density" else -0.15
            ax_acc.scatter([delta], [y + offset], s=52, color=color, linewidths=0, zorder=3)
            if abs(delta) >= 0.05:
                ax_acc.annotate(f"{delta:+.2f}", (delta, y + offset), xytext=(7, -3),
                                textcoords="offset points", fontsize=8, color=COLOR_SECONDARY,
                                zorder=4)

    ax_acc.set_xlim(-0.62, 0.62)
    ax_acc.set_title("Change in refit out-of-fold exact match (n=661)", fontsize=11.5,
                     color=COLOR_PRIMARY, loc="left", pad=10)
    ax_acc.set_xlabel("percentage points vs. the v2.2 fit   (1 FOV = 0.15pt)",
                      color=COLOR_SECONDARY, fontsize=10)
    ax_acc.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor=color,
                                  markersize=8, label=label) for _, label, color in AXES],
                  loc="upper right", frameon=False, fontsize=9)

    for ax in (ax_time, ax_acc):
        ax.set_yticks(positions)
        ax.set_ylim(-0.6, len(order) - 0.4)
        ax.tick_params(colors=COLOR_MUTED)
        ax.grid(axis="x", color=COLOR_GRID, linewidth=1)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("left", "bottom"):
            ax.spines[spine].set_color(COLOR_AXIS)
    ax_time.set_yticklabels([variant_label(name) for name in order], fontsize=10)
    ax_acc.set_yticklabels([])  # rows align with panel 1; repeating the labels adds nothing

    fig.suptitle("LBP runtime optimization: what each variant costs and what it changes",
                 fontsize=13, color=COLOR_PRIMARY, x=0.008, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, facecolor=COLOR_SURFACE, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
