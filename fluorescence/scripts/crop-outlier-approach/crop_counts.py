"""Resolve (sample_id, fov_id, country) to a detected-crop count (`n_spots_detected`) and to a
whole-slide leave-one-out baseline, reading only precomputed detection output already sitting
in GCS -- never an image.

This deliberately reads the `detection_results/` prefix that `src/gcs_fov.py` and
`src/gcs_fov_multi.py` avoid on purpose (they resolve *raw* images, upstream of any model). That
tree isn't documented anywhere in this repo; the layout below was confirmed by browsing the
buckets directly (`gcloud storage ls`/`cat`), not by reading any existing code:

- Liberia (`gs://liberia-2025`) and Tanzania (`gs://tanzania_02032026`) both have
  `detection_results/<...>/<model_version>/<slide_folder>/fov_summary.csv`, one row per
  `fov_id`, with a `n_spots_detected` column -- the raw count of candidate fluorescent-spot
  crops before ML filtering (chosen over `n_rbcs` because it's mechanistically the metric a
  bright overexposure halo would inflate: these crops come from thresholding local maxima in
  the same blue-channel intensity map the halo lives in).
    * Liberia's slide folder is the same `_Blue` folder `src.gcs_fov.find_slide_blue_folder`
      already resolves, with the `_Blue` suffix stripped, under
      `detection_results/LB25-<batch>/<model_version>/`.
    * Tanzania's slide folder is `sample_id` directly under
      `detection_results/<model_version>/` -- no box-probing needed (unlike raw-image
      resolution), since this tree isn't nested under a box.
    * Verified `fov_id` here uses the exact same raster addressing as the labels CSV (checked
      LB25-D10's sparse-gap pattern against `gcs_fov.py`'s stride-18/ColumnCount=13 special
      case) -- safe to join directly, no row/col decoding needed.
    * Verified `n_spots_detected` is identical across every model-version run folder for a
      slide (the upstream spot-finding step is shared; only the classifier differs) -- this
      standardizes on one folder, `MODEL_VERSION`, per country.
    * Some labeled samples have no `detection_results` under any model version at all (e.g.
      `KTR-72502946`, confirmed to exist as a raw sample but never processed) --
      `load_slide_crop_counts` returns `None` for these rather than raising.
- Uganda (`gs://malaria-annotation-web`) has no per-FOV summary file, but
  `samples/<sample_id>/spots.csv` has the identical per-spot schema
  (`sample_id,fov_id,x,y,log_radius,score,positive`) as Liberia/Tanzania's own `spots.csv` --
  grouping by `fov_id` and counting rows reproduces `n_spots_detected` per FOV without
  touching any image.
"""
import csv
import io
import statistics
import sys
import threading
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.gcs_fov import find_slide_blue_folder, parse_sample_id
from src.gcs_fov_multi import TZ_BUCKET, UG_BUCKET

LB_BUCKET = "liberia-2025"
LB_MODEL_VERSION = "v8_hardneg_single_t0.995"
TZ_MODEL_VERSION = "v8_hardneg_single_t0.995"

MAD_SCALE = 1.4826  # scales MAD to be a normal-consistent estimator of standard deviation

_COUNTRY_ALIASES = {
    "liberia": "lb", "lb": "lb",
    "tanzania": "tz", "tz": "tz",
    "uganda": "ug", "ug": "ug",
}

_client_local = threading.local()
_slide_cache = {}  # (country, sample_id) -> dict[fov_id, n_spots_detected] | None


def _client():
    client = getattr(_client_local, "client", None)
    if client is None:
        from google.cloud import storage

        client = storage.Client()
        _client_local.client = client
    return client


def _try_download_text(bucket, blob_name):
    """Return the blob's text, or None if it doesn't exist (rather than raising) -- a missing
    detection_results tree is an expected, reportable case here, not an error.
    """
    from google.api_core.exceptions import NotFound

    blob = _client().bucket(bucket).blob(blob_name)
    try:
        return blob.download_as_text()
    except NotFound:
        return None


def _parse_fov_summary_csv(text):
    reader = csv.DictReader(io.StringIO(text))
    return {int(row["fov_id"]): int(row["n_spots_detected"]) for row in reader}


def _parse_spots_csv_counts(text):
    """Count rows per fov_id in a spots.csv (sample_id,fov_id,x,y,log_radius,score,positive) --
    reproduces the same n_spots_detected a fov_summary.csv column would report.
    """
    reader = csv.DictReader(io.StringIO(text))
    counts = Counter(int(row["fov_id"]) for row in reader)
    return dict(counts)


def _lb_fov_summary_blob_name(sample_id):
    batch, *_ = parse_sample_id(sample_id)
    blue_folder = find_slide_blue_folder(sample_id, bucket=LB_BUCKET)
    slide_name = blue_folder.rstrip("/").rsplit("/", 1)[-1]
    if slide_name.endswith("_Blue"):
        slide_name = slide_name[: -len("_Blue")]
    return f"detection_results/LB25-{batch}/{LB_MODEL_VERSION}/{slide_name}/fov_summary.csv"


def _tz_fov_summary_blob_name(sample_id):
    return f"detection_results/{TZ_MODEL_VERSION}/{sample_id}/fov_summary.csv"


def _ug_spots_blob_name(sample_id):
    return f"samples/{sample_id}/spots.csv"


def load_slide_crop_counts(sample_id, country):
    """Return {fov_id: n_spots_detected} for every FOV on this slide with detection results, or
    None if this sample has no detection results at all. Cached per (country, sample_id) since
    several labeled rows in the diverse test set share a slide.
    """
    key_country = _COUNTRY_ALIASES.get(country.strip().lower())
    if key_country is None:
        raise ValueError(f"Unknown country: {country!r}")

    cache_key = (key_country, sample_id)
    if cache_key in _slide_cache:
        return _slide_cache[cache_key]

    if key_country == "lb":
        text = _try_download_text(LB_BUCKET, _lb_fov_summary_blob_name(sample_id))
        counts = _parse_fov_summary_csv(text) if text is not None else None
    elif key_country == "tz":
        text = _try_download_text(TZ_BUCKET, _tz_fov_summary_blob_name(sample_id))
        counts = _parse_fov_summary_csv(text) if text is not None else None
    elif key_country == "ug":
        text = _try_download_text(UG_BUCKET, _ug_spots_blob_name(sample_id))
        counts = _parse_spots_csv_counts(text) if text is not None else None
    else:
        raise AssertionError(key_country)  # unreachable, all _COUNTRY_ALIASES values handled above

    _slide_cache[cache_key] = counts
    return counts


def slide_baseline(counts, fov_id):
    """Leave-one-out baseline for one FOV against every other FOV on its slide: mean, std,
    median, and scaled MAD (median absolute deviation), plus how many other FOVs contributed.

    Median/MAD are the primary robust statistic (see module docstring's sibling,
    analyze_crop_outliers.py, for why) -- they tolerate up to ~50% of the slide's *other* FOVs
    being contaminated with elevated counts before breaking, unlike mean/std which even one or
    two such FOVs can already skew. Mean/std are still returned so a large divergence between
    the two is itself visible as a signal.
    """
    others = [v for other_fov, v in counts.items() if other_fov != fov_id]
    n = len(others)
    if n == 0:
        return {"mean": None, "std": None, "median": None, "mad": None, "n_other_fovs": 0}

    mean = statistics.mean(others)
    std = statistics.stdev(others) if n >= 2 else 0.0
    median = statistics.median(others)
    mad = MAD_SCALE * statistics.median(abs(v - median) for v in others)
    return {"mean": mean, "std": std, "median": median, "mad": mad, "n_other_fovs": n}
