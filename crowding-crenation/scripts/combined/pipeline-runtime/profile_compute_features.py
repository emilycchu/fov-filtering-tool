"""Per-stage timings for the v2 feature vector, to say where the per-FOV second actually goes.

Written after the LBP work took `compute_features` from 5.85s to ~1.1s per 2800x2800 FOV, which
moved the bottleneck rather than removing it. Read-only: times the existing code, changes
nothing.

**These stages are not an additive budget, and the script refuses to present them as one.**
Two reasons:

1. Run-to-run variance is large relative to the gaps between stages. The 301-px GaussianBlur
   alone measured 0.539-0.657s across six consecutive calls in one process -- a 0.118s spread,
   which is bigger than every stage below it. An earlier hand-rolled version of this profile
   reported a sum of parts (1.104s) that exceeded its own measured total (1.044s) purely from
   taking min-of-N on each piece independently.
2. Each stage is timed as `compute_features` actually calls it -- passing the colour image, so
   each feature function does its own BGR2GRAY internally. That duplicated conversion is real
   work the pipeline really does (six times), so it is reported as its own row rather than
   quietly shared.

So: min, median and spread per stage, plus the measured whole, and the share column is computed
against the *whole*, not against the sum.

Usage:
    python scripts/combined/pipeline-runtime/profile_compute_features.py [--repeat 5]
        [--fovs N] [--lbp-step 16] [--out PATH]
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "combined"))

from _v2_common import (  # noqa: E402
    INITIAL_IMAGE_DIR,
    TANZANIA_IMAGE_DIR,
    TILE_GLCM_LEVELS,
    TILE_GRID_SIZE,
    compute_features,
    write_csv_dicts,
)
from src.features.edge_density import edge_density  # noqa: E402
from src.features.glcm_contrast import glcm_contrast  # noqa: E402
from src.features.lbp_entropy import lbp_entropy  # noqa: E402
from src.features.otsu_separability import otsu_separability  # noqa: E402
from src.features.tile_heterogeneity import (  # noqa: E402
    coefficient_of_variation,
    patchiness,
    tile_statistics,
)
from src.segmentation import cell_coverage, correct_illumination, otsu_segment, to_grayscale  # noqa: E402

OUT_DIR = ROOT / "data" / "results" / "pipeline-runtime"
DEFAULT_OUT = OUT_DIR / "stage-profile.csv"
BLUR_KSIZE = 301


def timings(fn, repeat):
    out = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        out.append(time.perf_counter() - start)
    return out


def _tile_glcm_contrast(tile):
    quantized = (tile.astype(np.uint16) * TILE_GLCM_LEVELS // 256).astype(np.uint8)
    return glcm_contrast(quantized, levels=TILE_GLCM_LEVELS)


def stages(path, image, gray, corrected, lbp_step):
    """Each stage as compute_features calls it -- colour image in, where that is what it passes."""
    return [
        ("decode: imread colour", lambda: cv2.imread(str(path))),
        ("decode: imread grayscale", lambda: cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)),
        ("to_grayscale", lambda: to_grayscale(image)),
        ("otsu_segment", lambda: otsu_segment(image)),
        ("cell_coverage", lambda: cell_coverage(otsu_segment(image)[0])),
        ("otsu_separability", lambda: otsu_separability(image)),
        (f"correct_illumination({BLUR_KSIZE})", lambda: correct_illumination(gray, blur_ksize=BLUR_KSIZE)),
        ("tile_glcm grid 7x7", lambda: tile_statistics(corrected, grid_size=TILE_GRID_SIZE,
                                                       stat_fn=_tile_glcm_contrast)),
        (f"lbp_entropy(step={lbp_step})", lambda: lbp_entropy(image, step=lbp_step)),
        ("glcm_contrast whole image", lambda: glcm_contrast(image)),
        ("edge_density", lambda: edge_density(image)),
    ]


def sample_paths(n):
    paths = sorted(TANZANIA_IMAGE_DIR.iterdir())[:max(1, n - 1)]
    paths += sorted(INITIAL_IMAGE_DIR.iterdir())[:1]
    return paths[:n]


def main():
    parser = argparse.ArgumentParser(description="Profile compute_features stage by stage.")
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--fovs", type=int, default=3)
    parser.add_argument("--lbp-step", type=int, default=16)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    rows = []
    for path in sample_paths(args.fovs):
        image = cv2.imread(str(path))
        gray = to_grayscale(image)
        corrected = correct_illumination(gray, blur_ksize=BLUR_KSIZE)

        whole = timings(lambda: compute_features(image, lbp_step=args.lbp_step), args.repeat)
        total = min(whole)
        print(f"\n{path.name}")
        print(f"  compute_features(lbp_step={args.lbp_step}): min {total:.3f}s  "
              f"median {statistics.median(whole):.3f}s  spread {max(whole)-min(whole):.3f}s")
        print(f"  {'stage':<34} {'min':>8} {'median':>8} {'spread':>8} {'% of whole':>11}")
        for name, fn in stages(path, image, gray, corrected, args.lbp_step):
            ts = timings(fn, args.repeat)
            lo, med, spread = min(ts), statistics.median(ts), max(ts) - min(ts)
            print(f"  {name:<34} {lo:>7.3f}s {med:>7.3f}s {spread:>7.3f}s {100*lo/total:>10.1f}%")
            rows.append({"filename": path.name, "stage": name, "min_s": f"{lo:.5f}",
                         "median_s": f"{med:.5f}", "spread_s": f"{spread:.5f}",
                         "pct_of_whole": f"{100*lo/total:.2f}",
                         "whole_min_s": f"{total:.5f}",
                         "whole_spread_s": f"{max(whole)-min(whole):.5f}"})

    write_csv_dicts(args.out, list(rows[0]), rows)
    print(f"\nwrote {args.out}")
    print("NOTE: stages do not sum to the whole -- see this script's docstring. Read the share "
          "column as 'fraction of the measured whole', and mind the spread column before "
          "treating any gap under ~0.1s as real.")


if __name__ == "__main__":
    main()
