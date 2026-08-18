"""Otsu-threshold based cell/foreground segmentation."""
import cv2
import numpy as np


def to_grayscale(image):
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def correct_illumination(gray, blur_ksize=301, downsample=1):
    """Flatten a slow (out-of-focus/vignetting-scale) illumination gradient.

    Estimates the background as a heavily-blurred copy of the image, then subtracts
    it back out and re-centers around the original mean, leaving cell-scale texture
    intact while removing large-scale brightness trends across the FOV.

    ---- OPTIMIZATION: `downsample` ------------------------------------------------------
    This function was ~60% of `compute_features`' cost once LBP was optimized: a 301-px
    Gaussian over 7.84M pixels is 0.53-0.77 s. But the whole point of a 301-px kernel is to
    keep only content far coarser than the sampling grid, so the background can be estimated
    on a shrunken copy and scaled back up with almost no error. `downsample=4` differs from
    the full-resolution background by at most **1 grey level**.

    `downsample=1` (the default) is the original code path, byte for byte, so every params
    file that predates this argument behaves exactly as before.

    Two things measured the hard way, both in data/results/pipeline-runtime/README.md:

    - **`downsample=2` is the WORST factor, not the safest.** It drifts 5-14x more than 4 and
      is the only factor that changes any label on the 661-FOV calibration set (4 Rouleaux
      predictions, all of them correct ones lost). It is not the sigma mismatch from
      integer-dividing the kernel size -- a sigma-matched variant measured identical -- it is
      the resize round-trip, and the mechanism is genuinely unexplained. Use 4.
    - **The stage saturates at `downsample=4`.** The blur itself gets 38x faster, but the
      float32 subtract/clip below is ~0.11 s of fixed cost over the full-resolution image
      regardless, so 8 and 16 are no faster than 4 while drifting more. 6.4x is the ceiling
      for this function, and that tail is now its floor.
    -------------------------------------------------------------------------------------
    """
    if downsample == 1:
        background = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    else:
        # Shrink the kernel with the image so the background keeps the same physical scale.
        ksize = blur_ksize // downsample
        ksize += 1 - ksize % 2  # GaussianBlur requires an odd kernel
        small = cv2.resize(gray, None, fx=1 / downsample, fy=1 / downsample,
                           interpolation=cv2.INTER_AREA)
        background = cv2.resize(cv2.GaussianBlur(small, (ksize, ksize), 0),
                                (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    corrected = gray.astype(np.float32) - background.astype(np.float32) + float(background.mean())
    return np.clip(corrected, 0, 255).astype(np.uint8)


def otsu_segment(image, blur_ksize=5):
    """Segment foreground (cells) from background using Otsu's method.

    Returns the binary mask (0/255) and the threshold value Otsu selected.
    """
    gray = to_grayscale(image)
    if blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    threshold, mask = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask, threshold


def cell_coverage(mask):
    """Fraction of pixels in the mask that belong to the foreground."""
    return float(np.count_nonzero(mask)) / mask.size
