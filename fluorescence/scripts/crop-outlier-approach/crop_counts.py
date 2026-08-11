"""Resolve (sample_id, fov_id, country) to a per-FOV count for either of two metrics, and to a
whole-slide baseline (computed over every FOV on the slide, including the target), reading only
precomputed detection output already sitting in GCS -- never an image.

**Two metrics, chosen by the `metric` argument:**
- `"n_spots_detected"` (default) -- the raw count of candidate fluorescent-spot crops *before*
  ML filtering. Chosen over `n_rbcs` because it's mechanistically the metric a bright
  overexposure halo would inflate: these crops come from thresholding local maxima in the same
  blue-channel intensity map the halo lives in.
- `"n_positives"` -- the count of crops the ML classifier actually confirmed as a parasite, per
  FOV. Added to test a different hypothesis: does the halo artifact inflate *confirmed
  parasites*, not just raw candidate crops? (If the classifier correctly rejects halo-caused
  candidates, this should show a much weaker -- or no -- separation between ground-truth
  groups than `n_spots_detected` does; see `data/results/crop-outlier-approach/README.md`'s
  comparison section for the actual result.)

This deliberately reads the `detection_results/` prefix that `src/gcs_fov.py` and
`src/gcs_fov_multi.py` avoid on purpose (they resolve *raw* images, upstream of any model). That
tree isn't documented anywhere in this repo; the layout below was confirmed by browsing the
buckets directly (`gcloud storage ls`/`cat`), not by reading any existing code:

- Liberia (`gs://liberia-2025`) and Tanzania (`gs://tanzania_02032026`) both have
  `detection_results/<...>/<model_version>/<slide_folder>/fov_summary.csv`, one row per
  `fov_id`, with both an `n_spots_detected` and an `n_positives` column.
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
      slide (the upstream spot-finding step is shared); `n_positives` legitimately *does*
      depend on which classifier produced it, since that's the thing that differs between
      model versions -- this standardizes on one folder, `MODEL_VERSION`, per country, for
      both metrics.
    * Tanzania fallback, added after finding `tanzania_02032026`'s own `detection_results/` tree
      is missing 146 slides that were never mirrored there (94 of `TZ2025-Box1`'s 98 slides, 52
      of `Box5`'s 99) -- `gs://malaria-annotation-web` (the annotation tool's bucket) turns out
      to hold `samples/<sample_id>/fov_summary.csv` for every one of those 146 gap slides,
      confirmed byte-for-byte identical to `tanzania_02032026`'s own copy on a sample that has
      both (`RUB-62501326`: 324/324 `n_spots_detected` values match). Tried second, after the
      primary `tanzania_02032026` lookup returns nothing, never instead of it.
- Uganda (`gs://malaria-annotation-web`) has no per-FOV summary file, but
  `samples/<sample_id>/spots.csv` has the identical per-spot schema
  (`fov_id,x,y,radius,score,positive`) as Liberia/Tanzania's own `spots.csv`. **Caveat, found
  2026-08-11 and not yet fixed:** grouping this by `fov_id` and counting rows was assumed to
  reproduce `n_spots_detected`, but cross-checking against a Tanzania sample that has both this
  file and a `fov_summary.csv` (`RUB-62501326`) shows the row count matches `n_spots_filtered`
  exactly (0/324 mismatches) and `n_spots_detected` on zero FOVs. Uganda has no `fov_summary.csv`
  anywhere to source true pre-filter `n_spots_detected` from, so the 10 Uganda rows already in
  `data/results/crop-outlier-approach/results.csv` are on a different, already-partially-filtered
  metric than every Liberia/Tanzania row -- flagged, not corrected, pending a decision on whether
  `n_spots_detected` is recoverable for Uganda at all.
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

METRICS = ("n_spots_detected", "n_positives")

MAD_SCALE = 1.4826  # scales MAD to be a normal-consistent estimator of standard deviation

_COUNTRY_ALIASES = {
    "liberia": "lb", "lb": "lb",
    "tanzania": "tz", "tz": "tz",
    "uganda": "ug", "ug": "ug",
}

_client_local = threading.local()
_slide_cache = {}  # (country, sample_id, metric) -> dict[fov_id, count] | None


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


def _parse_fov_summary_csv(text, metric):
    reader = csv.DictReader(io.StringIO(text))
    return {int(row["fov_id"]): int(row[metric]) for row in reader}


def _parse_spots_csv_counts(text, metric):
    """Aggregate a spots.csv (sample_id,fov_id,x,y,log_radius,score,positive) per fov_id --
    row count reproduces n_spots_detected, sum of the `positive` column reproduces n_positives.
    """
    reader = csv.DictReader(io.StringIO(text))
    counts = Counter()
    if metric == "n_spots_detected":
        for row in reader:
            counts[int(row["fov_id"])] += 1
    elif metric == "n_positives":
        for row in reader:
            counts[int(row["fov_id"])] += int(row["positive"])
    else:
        raise ValueError(f"Unknown metric: {metric!r}")
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


def _annotation_fov_summary_blob_name(sample_id):
    return f"samples/{sample_id}/fov_summary.csv"


def _ug_spots_blob_name(sample_id):
    return f"samples/{sample_id}/spots.csv"


def load_slide_metric_counts(sample_id, country, metric="n_spots_detected"):
    """Return {fov_id: count} for every FOV on this slide with detection results, for either
    `metric` ("n_spots_detected" or "n_positives" -- see module docstring), or None if this
    sample has no detection results at all. Cached per (country, sample_id, metric) since
    several labeled rows in the diverse test set share a slide.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric!r}, expected one of {METRICS}")
    key_country = _COUNTRY_ALIASES.get(country.strip().lower())
    if key_country is None:
        raise ValueError(f"Unknown country: {country!r}")

    cache_key = (key_country, sample_id, metric)
    if cache_key in _slide_cache:
        return _slide_cache[cache_key]

    if key_country == "lb":
        text = _try_download_text(LB_BUCKET, _lb_fov_summary_blob_name(sample_id))
        counts = _parse_fov_summary_csv(text, metric) if text is not None else None
    elif key_country == "tz":
        text = _try_download_text(TZ_BUCKET, _tz_fov_summary_blob_name(sample_id))
        if text is None:
            # fallback: the annotation tool's bucket has fov_summary.csv for slides never
            # mirrored into tanzania_02032026's own detection_results/ tree -- see module
            # docstring's Tanzania fallback note.
            text = _try_download_text(UG_BUCKET, _annotation_fov_summary_blob_name(sample_id))
        counts = _parse_fov_summary_csv(text, metric) if text is not None else None
    elif key_country == "ug":
        text = _try_download_text(UG_BUCKET, _ug_spots_blob_name(sample_id))
        counts = _parse_spots_csv_counts(text, metric) if text is not None else None
    else:
        raise AssertionError(key_country)  # unreachable, all _COUNTRY_ALIASES values handled above

    _slide_cache[cache_key] = counts
    return counts


def slide_baseline(counts):
    """Whole-slide baseline computed once per slide, over *every* FOV with detection results on
    it (the target FOV included, not left out): mean, std, median, and scaled MAD (median
    absolute deviation), plus how many FOVs contributed.

    Previously this excluded the specific FOV being scored (leave-one-out). Changed to include
    every FOV uniformly -- for slides with 300+ FOVs the numeric difference from dropping one
    value is negligible, and a single baseline per slide (computed once, reused for every FOV
    scored against it) is simpler than recomputing per target.

    Median/MAD are the primary robust statistic (see module docstring's sibling,
    analyze_crop_outliers.py, for why) -- they tolerate up to ~50% of a slide's FOVs being
    contaminated with elevated counts before breaking, unlike mean/std which even one or two such
    FOVs can already skew. Mean/std are still returned so a large divergence between the two is
    itself visible as a signal.
    """
    values = list(counts.values())
    n = len(values)
    if n == 0:
        return {"mean": None, "std": None, "median": None, "mad": None, "n_fovs": 0}

    mean = statistics.mean(values)
    std = statistics.stdev(values) if n >= 2 else 0.0
    median = statistics.median(values)
    mad = MAD_SCALE * statistics.median(abs(v - median) for v in values)
    return {"mean": mean, "std": std, "median": median, "mad": mad, "n_fovs": n}
