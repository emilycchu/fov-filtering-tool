"""LBP entropy feature: texture complexity via local binary pattern histogram entropy.

`lbp_entropy()` returns exactly what it always did -- the number is bit-identical to the
`skimage.feature.local_binary_pattern` implementation this module used to call, which is
preserved as `lbp_entropy_skimage()` and asserted against in
`scripts/combined/lbp-optimization/bench_lbp.py`. What changed is how it gets there: skimage
walks 7.84M pixels x 24 neighbours in a generic per-pixel Cython loop (~5s per 2800x2800 FOV,
~82% of `_v2_common.compute_features`), where `_lbp_codes()` does the same arithmetic as
whole-array numpy operations over cache-sized tiles spread across threads.

Reproducing skimage bit-for-bit takes three details that are easy to miss:

1. The neighbour offsets are **rounded to 5 decimals** before use
   (`rp = np.round(-R*sin(2*pi*i/P), 5)`), not used at full precision.
2. skimage computes the interpolation weight as `dr = (row + rp[i]) - floor(row + rp[i])`.
   Subtracting from a coordinate as large as 2799 loses low bits, so `dr` differs slightly
   from row to row. Collapsing it to a single scalar `rp[i] - floor(rp[i])` still matches on
   random noise, but diverges on real images: flat regions are exact ties where
   `(1-dc)*a + dc*a != a` in floating point, so the sign of `interp - centre >= 0` flips on a
   handful of pixels. Hence `_interp_weights()` returns per-row and per-column vectors.
3. skimage's transition count for `method="uniform"` is **non-circular** -- it runs
   `for i in range(P-1)`, so the wrap from bit P-1 back to bit 0 is never counted.

Out-of-bounds neighbours read as 0 (skimage passes `mode='C'`, `cval=0`), which is why zero
padding here is equivalent and why a row strip carrying a `ceil(radius)+1` halo reproduces
its interior rows exactly.

Two rewrites that look free and are not: float32 arithmetic, and folding
`(1-dc)*a + dc*b` into the cheaper `a + dc*(b-a)`. Both change the last bits, and the last
bits are exactly what the `>=` comparison turns into different codes.

`step > 1` evaluates the operator on a stride-`step` grid of *centre* pixels while still
sampling neighbours at the full-resolution radius, so the codes stay bit-identical to the
corresponding pixels of the full map and only the histogram is estimated from a subsample.
That is a deliberately different tradeoff from downsampling the image first
(`lbp_entropy_fast.py`), which changes the operator's spatial scale. Anything with
`step > 1` returns an approximate entropy and must not be used by a calibrated path without
revalidation -- see `data/results/lbp-runtime/README.md`.
"""
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

DEFAULT_RADIUS = 3
# 512 measured fastest on a 2800x2800 FOV; 256 was ~8% slower and 128 was 2.7x slower
# (per-tile numpy dispatch overhead swamps the cache win).
TILE = 512
MAX_THREADS = 8


def _to_gray(image):
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _sample_offsets(n_points, radius):
    """skimage's neighbour offsets, including its round-to-5-decimals step."""
    i = np.arange(n_points, dtype=np.float64)
    rp = np.round(-radius * np.sin(2 * np.pi * i / n_points), 5)
    cp = np.round(radius * np.cos(2 * np.pi * i / n_points), 5)
    return rp, cp


def _interp_weights(n_points, radius, n_rows, n_cols, row_offset=0, col_offset=0):
    """Per-offset (row_shift, col_shift, dr_by_row, dc_by_col) bilinear interpolation terms.

    dr/dc are computed the way skimage's Cython does -- from the absolute coordinate, so they
    carry that subtraction's low-bit loss (see detail 2 in the module docstring).

    `row_offset`/`col_offset` are where this array sits inside the larger image it was cut
    from. They exist because of that same low-bit loss: a row strip handed to a worker starts
    at local row 0 but skimage computed its weights at the absolute row, so splitting an image
    and reassembling the pieces is only exact if each piece knows its origin.
    """
    rp, cp = _sample_offsets(n_points, radius)
    rows = np.arange(n_rows, dtype=np.float64) + row_offset
    cols = np.arange(n_cols, dtype=np.float64) + col_offset

    weights = []
    for i in range(n_points):
        r_coord = rows + rp[i]
        c_coord = cols + cp[i]
        r_floor = np.floor(r_coord)
        c_floor = np.floor(c_coord)
        row_shift = int(np.floor(rp[i]))
        col_shift = int(np.floor(cp[i]))
        # The tiled slicing below addresses neighbours by a constant integer shift, which is
        # only the same thing as flooring each coordinate while no coordinate rounds across an
        # integer boundary. True for every image size this repo sees; fail loudly if not.
        if not (np.array_equal(r_floor, rows + row_shift) and np.array_equal(c_floor, cols + col_shift)):
            raise ValueError(
                f"neighbour {i} of {n_points} (radius {radius}) rounds across an integer "
                f"coordinate boundary on a {n_rows}x{n_cols} image; the tiled kernel cannot "
                "address it by constant shift. Use lbp_entropy_skimage() for this input."
            )
        weights.append((row_shift, col_shift, r_coord - r_floor, c_coord - c_floor))
    return weights


def _tile_codes(padded, centres, weights, n_points, pad, r0, c0, height, width, step):
    """Uniform-method LBP codes for one tile of centre pixels."""
    top = np.empty((height, width), dtype=np.float64)
    bottom = np.empty((height, width), dtype=np.float64)
    interp = np.empty((height, width), dtype=np.float64)
    ones = np.zeros((height, width), dtype=np.uint8)
    transitions = np.zeros((height, width), dtype=np.uint8)
    previous = None

    r_stop = r0 + height * step
    c_stop = c0 + width * step

    for i in range(n_points):
        row_shift, col_shift, dr_by_row, dc_by_col = weights[i]
        dr = dr_by_row[r0:r_stop:step][:, None]
        dc = dc_by_col[c0:c_stop:step][None, :]

        def neighbour(row_offset, col_offset):
            r_start = pad + r0 + row_offset
            c_start = pad + c0 + col_offset
            return padded[r_start:r_start + height * step:step,
                          c_start:c_start + width * step:step]

        np.multiply(1 - dc, neighbour(row_shift, col_shift), out=top)
        top += dc * neighbour(row_shift, col_shift + 1)
        np.multiply(1 - dc, neighbour(row_shift + 1, col_shift), out=bottom)
        bottom += dc * neighbour(row_shift + 1, col_shift + 1)
        np.multiply(1 - dr, top, out=interp)
        interp += dr * bottom

        bit = interp >= centres
        ones += bit
        if i:
            transitions += bit != previous
        previous = bit

    # Uniform patterns (<= 2 transitions) are coded by their popcount; everything else
    # collapses into the single bin n_points + 1.
    return np.where(transitions <= 2, ones, n_points + 1)


def _lbp_codes(gray, n_points, radius, step=1, tile=TILE, workers=None, row_offset=0,
               col_offset=0):
    """Uniform-method LBP codes on a stride-`step` grid of centre pixels.

    Bit-identical to
    `local_binary_pattern(gray, n_points, radius, method="uniform")[::step, ::step]`.

    Pass `row_offset`/`col_offset` when `gray` is a slice of a larger image and the result has
    to match what that larger image would have produced -- see `_interp_weights`.
    """
    image = np.ascontiguousarray(gray, dtype=np.float64)
    n_rows, n_cols = image.shape
    weights = _interp_weights(n_points, radius, n_rows, n_cols, row_offset, col_offset)

    pad = int(np.ceil(radius)) + 1
    padded = np.zeros((n_rows + 2 * pad, n_cols + 2 * pad), dtype=np.float64)
    padded[pad:pad + n_rows, pad:pad + n_cols] = image

    out_rows = len(range(0, n_rows, step))
    out_cols = len(range(0, n_cols, step))
    codes = np.empty((out_rows, out_cols), dtype=np.uint8)

    def fill(block):
        a, b = block
        height = min(a + tile, out_rows) - a
        width = min(b + tile, out_cols) - b
        r0, c0 = a * step, b * step
        centres = image[r0:r0 + height * step:step, c0:c0 + width * step:step]
        codes[a:a + height, b:b + width] = _tile_codes(
            padded, centres, weights, n_points, pad, r0, c0, height, width, step
        )

    blocks = [(a, b) for a in range(0, out_rows, tile) for b in range(0, out_cols, tile)]

    if workers is None:
        workers = _auto_workers()
    if workers <= 1 or len(blocks) == 1:
        for block in blocks:
            fill(block)
    else:
        with ThreadPoolExecutor(workers) as pool:
            list(pool.map(fill, blocks))
    return codes


def _auto_workers():
    """Thread count for one FOV, or 1 when we are already inside a worker process.

    `extract_features_v2.py` and `nigeria_081226.py` fan FOVs out over a `multiprocessing.Pool`;
    spawning threads per FOV underneath that oversubscribes the machine and makes the whole
    batch slower. Pool workers are daemonic, which is the signal to stay serial and let the
    outer pool own the parallelism.
    """
    if multiprocessing.current_process().daemon:
        return 1
    return min(MAX_THREADS, multiprocessing.cpu_count() or 1)


def _entropy(codes, n_points):
    counts = np.bincount(codes.ravel(), minlength=n_points + 2)
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts[counts > 0] / total
    return float(-np.sum(probs * np.log2(probs)))


def lbp_entropy(image, radius=DEFAULT_RADIUS, n_points=None, method="uniform", step=1,
                tile=TILE, workers=None):
    """Shannon entropy of the LBP code histogram.

    With the default `step=1` this is bit-identical to `lbp_entropy_skimage()`, just faster.
    `step > 1` subsamples the centre grid and returns an approximation -- see the module
    docstring.
    """
    gray = _to_gray(image)
    if n_points is None:
        n_points = 8 * radius
    if method != "uniform":
        # Only the uniform code path is reimplemented here; anything else defers to skimage
        # rather than silently returning a different feature.
        return lbp_entropy_skimage(image, radius=radius, n_points=n_points, method=method)

    codes = _lbp_codes(gray, n_points, radius, step=step, tile=tile, workers=workers)
    return _entropy(codes, n_points)


def lbp_entropy_skimage(image, radius=DEFAULT_RADIUS, n_points=None, method="uniform"):
    """The original skimage implementation, kept as the equivalence reference.

    Not called by the pipeline for the uniform method. `scripts/combined/lbp-optimization/bench_lbp.py`
    asserts `_lbp_codes()` reproduces `local_binary_pattern()` exactly; this is the other
    side of that assertion, so do not "clean it up" into a wrapper around the fast path.
    """
    gray = _to_gray(image)
    if n_points is None:
        n_points = 8 * radius

    lbp = local_binary_pattern(gray, n_points, radius, method=method)
    n_bins = int(lbp.max()) + 1
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    probs = hist[hist > 0]
    return float(-np.sum(probs * np.log2(probs)))
