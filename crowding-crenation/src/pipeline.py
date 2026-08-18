"""End-to-end cell crowding scoring pipeline: segment, extract features, score."""
import argparse
import json
import threading
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

from .composite import composite_score
from .features.edge_density import edge_density
from .features.glcm_contrast import glcm_contrast
from .features.lbp_entropy import lbp_entropy
from .segmentation import cell_coverage, otsu_segment

IMAGE_EXTENSIONS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


class GCSPath:
    """Path-like reference to a blob in a GCS bucket (gs://bucket/blob-name), so
    run scripts can treat a bucket prefix the same way they treat a local directory.
    """

    def __init__(self, bucket, blob_name):
        self.bucket = bucket
        self.blob_name = blob_name

    @property
    def name(self):
        return PurePosixPath(self.blob_name).name

    @property
    def suffix(self):
        return PurePosixPath(self.blob_name).suffix

    @property
    def stem(self):
        return PurePosixPath(self.blob_name).stem

    def __truediv__(self, other):
        prefix = self.blob_name.rstrip("/")
        blob_name = f"{prefix}/{other}" if prefix else str(other)
        return GCSPath(self.bucket, blob_name)

    def __str__(self):
        return f"gs://{self.bucket}/{self.blob_name}"

    def __repr__(self):
        return f"GCSPath({str(self)!r})"


def is_gcs_path(path):
    return str(path).startswith("gs://")


_gcs_client_local = threading.local()


def _gcs_client():
    """One storage.Client per thread. google-cloud-storage's Client is not documented
    as safe to share across threads (its underlying requests.Session/connection pool
    isn't guaranteed thread-safe, and sharing one has caused SSL errors under concurrent
    use in the wild), so score_liberia_sample.py's thread pool needs a client per thread
    rather than one shared singleton.
    """
    client = getattr(_gcs_client_local, "client", None)
    if client is None:
        from google.cloud import storage

        client = storage.Client()
        _gcs_client_local.client = client
    return client


def to_dir(directory):
    """Resolve a directory argument (local path or gs:// URI) to a Path or GCSPath root."""
    if is_gcs_path(directory):
        bucket, _, blob_name = str(directory)[len("gs://"):].partition("/")
        return GCSPath(bucket, blob_name)
    return Path(directory)


def load_image(path, grayscale=False):
    """Read an image from a local path or a gs:// blob.

    ---- OPTIMIZATION: `grayscale` -------------------------------------------------------
    Every FOV this repo processes is **already monochrome** -- all three channels equal, in
    the Tanzania PNGs, the Liberia PNGs and the Nigeria BMPs alike. Decoding colour therefore
    pushes three identical channels through libpng and hands back an array the whole v2
    feature vector immediately averages back to grey. `grayscale=True` skips that: 0.18s ->
    0.14s per 2800x2800 FOV, and it lets `compute_features` skip its conversion too.

    It is **bit-identical**, not approximately equal: verified against
    `cvtColor(imread(...), BGR2GRAY)` on 8 local FOVs across all three datasets, on
    GCS-streamed bytes, and on all 661 calibration FOVs (that check is what
    `pipeline-runtime/sweep_blur_downsample.py`'s downsample=1 baseline proves, since it
    loads grayscale where compute_features converts from colour).

    Default stays False because callers that render images still want colour --
    `nigeria_081226.py`'s thumbnail sheet does a BGR2RGB. See
    data/results/pipeline-runtime/README.md.
    -------------------------------------------------------------------------------------
    """
    if isinstance(path, GCSPath):
        gcs_path = path
    elif is_gcs_path(path):
        gcs_path = to_dir(path)
    else:
        gcs_path = None

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    if gcs_path is not None:
        blob = _gcs_client().bucket(gcs_path.bucket).blob(gcs_path.blob_name)
        data = blob.download_as_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), flag)
    else:
        image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def list_image_paths(directory, extensions=IMAGE_EXTENSIONS):
    """List image paths under a directory, local or a gs:// bucket prefix, sorted by name."""
    root = to_dir(directory)
    if isinstance(root, GCSPath):
        prefix = root.blob_name.rstrip("/") + "/" if root.blob_name else ""
        blobs = _gcs_client().list_blobs(root.bucket, prefix=prefix)
        paths = [GCSPath(root.bucket, blob.name) for blob in blobs if PurePosixPath(blob.name).suffix.lower() in extensions]
    else:
        paths = [p for p in root.iterdir() if p.suffix.lower() in extensions]
    return sorted(paths, key=lambda p: p.name)


def score_image(path):
    image = load_image(path)
    mask, otsu_threshold = otsu_segment(image)

    features = {
        "coverage": cell_coverage(mask),
        "edge_density": edge_density(image),
        "glcm_contrast": glcm_contrast(image),
        "lbp_entropy": lbp_entropy(image),
    }

    return {
        "path": str(path),
        "otsu_threshold": otsu_threshold,
        "features": features,
        "score": composite_score(features),
    }


def score_directory(directory, extensions=IMAGE_EXTENSIONS):
    return [score_image(p) for p in list_image_paths(directory, extensions)]


def main():
    parser = argparse.ArgumentParser(
        description="Score cell crowding in microscopy images."
    )
    parser.add_argument(
        "input", help="Path to an image file or a directory of images (local path or gs:// URI)."
    )
    args = parser.parse_args()

    if is_gcs_path(args.input):
        is_single_file = PurePosixPath(args.input).suffix.lower() in IMAGE_EXTENSIONS
        results = [score_image(to_dir(args.input))] if is_single_file else score_directory(args.input)
    else:
        input_path = Path(args.input)
        results = score_directory(input_path) if input_path.is_dir() else [score_image(input_path)]

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
