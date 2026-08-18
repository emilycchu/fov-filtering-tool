"""SUPERSEDED. Downsampled variant of lbp_entropy(); measured, and the drift is too large.

Use `lbp_entropy(image, step=N)` instead (src/features/lbp_entropy.py). It subsamples the
grid of *centre* pixels while still sampling neighbours at the full-resolution radius, so
the LBP codes stay bit-identical to the corresponding pixels of the full map and only the
histogram is estimated from a subsample. This module downsamples the *image*, which changes
the operator's spatial scale -- a different feature, not a faster one.

The drift the original version of this docstring asked someone to go measure has now been
measured, on 9 FOVs, against a calibrated feature range of [3.118, 4.196] (span 1.078):

    downsample=2  ->  -0.60 mean drift   (56% of the range)
    downsample=4  ->  -1.52 mean drift  (141% of the range)

versus 0.009 for stride-16 centre subsampling, which is 71x faster than full resolution and
changes none of the 1322 bucket assignments on the 661-FOV v2.2 calibration set. See
data/results/lbp-runtime/README.md.

Kept only because scripts/compare_lbp_entropy_fast.py still imports it to reproduce those
numbers. Do not import it anywhere else, and do not wire it into a calibrated path.

downsample=1 degenerates to the exact original lbp_entropy() (used as a correctness check).
"""
import cv2
import numpy as np
from skimage.feature import local_binary_pattern


def lbp_entropy_downsampled(image, radius=3, n_points=None, method="uniform", downsample=4):
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if n_points is None:
        n_points = 8 * radius
    gray = gray[::downsample, ::downsample]

    lbp = local_binary_pattern(gray, n_points, radius, method=method)
    n_bins = int(lbp.max()) + 1
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)

    probs = hist[hist > 0]
    if probs.size == 0:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))
