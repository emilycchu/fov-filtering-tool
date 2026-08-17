"""Layout, atomic IO, retry and progress reporting shared by the tanzania-complete-081426 passes.

This analysis scores 271 slides x 324 FOVs = 87,804 FOVs per image pass, streamed from GCS.
At that size two properties stop being niceties:

**Resumability.** The unit of checkpoint is one slide of one pass. A pass writes a slide's
per-FOV CSV once, atomically, when the slide finishes, and skips any slide whose CSV already
exists with the expected row count. So an interrupted run -- crash, Ctrl-C, Spot preemption --
loses at most the slide in flight, and re-invoking the same command is free for work already
done. `write_csv_atomic` is what makes the skip check trustworthy: a file that exists is
complete, because it is renamed into place only after fsync.

**Per-FOV durability of the *features*, not just the scores.** The per-FOV CSVs keep the full
9-feature vector, so slide aggregates, the empty-field-gate rule, the crop-outlier z threshold,
bucket assignments and both plots are all recomputable locally with zero network. Re-deciding
any of those must never mean re-streaming 342 GB.

Nothing in this module imports from the fluorescence or focus subprojects. All three own a
`src` package and only one can be `sys.modules["src"]`, so the passes are split by subproject
and communicate through `slides.csv` and `slide-index.json` instead of imports.
"""
import csv
import json
import os
import random
import socket
import ssl
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline import GCSPath  # noqa: E402

DATASET = "tanzania-complete-081426"

RESULTS_DIR = ROOT / "data" / "results" / DATASET
SLIDES_CSV = RESULTS_DIR / "slides.csv"
SLIDE_INDEX_JSON = RESULTS_DIR / "slide-index.json"
SLIDE_SUMMARY_CSV = RESULTS_DIR / "slide-summary.csv"
FOV_DIR = RESULTS_DIR / "fov"
CROWDING_FOV_DIR = FOV_DIR / "crowding"
LOGS_DIR = RESULTS_DIR / "logs"
PLOTS_DIR = RESULTS_DIR / "plots"

# The fluorescence passes live in the sibling subproject and write there; the aggregator reads
# across. Resolved rather than hardcoded so a moved checkout still works.
FLUORESCENCE_RESULTS_DIR = ROOT.parent / "fluorescence" / "data" / "results" / DATASET

TZ_BUCKET = "tanzania_02032026"
TZ_BOXES = (1, 2, 3, 4, 5)
EXPECTED_FOVS = 324

# Every slide prefix also holds dpc-preview.png, dpc-result.png and dpc-scan.txt (plus the
# fluorescent equivalents), so a `dpc-` prefix or an image-extension filter yields 327, not
# 324. Anchor on the three-digit FOV id instead.
DPC_RE = "^dpc-(\\d{3})-"
FLUOR_RE = "^fluorescent-(\\d{3})-"

GCS_MIRROR_PREFIX = "gs://malaria-analysis-shared/emily/" + DATASET


def dpc_blob_name(box, slide_id, fov_id):
    return f"{box}/{slide_id}/dpc-{fov_id:03d}-{slide_id}.png"


def fluor_blob_name(box, slide_id, fov_id):
    return f"{box}/{slide_id}/fluorescent-{fov_id:03d}-{slide_id}.png"


def dpc_gcs_path(box, slide_id, fov_id):
    return GCSPath(TZ_BUCKET, dpc_blob_name(box, slide_id, fov_id))


def slide_prefix(box, slide_id):
    return f"gs://{TZ_BUCKET}/{box}/{slide_id}"


# --- IO -------------------------------------------------------------------------------------
# encoding is always explicit. The catalog's TRUTH column contains a hazard sign, and Windows
# defaults text IO to cp1252, which raises UnicodeEncodeError on it. All other values in these
# files are ASCII, where utf-8 is byte-identical, so this costs nothing.

def read_csv_dicts(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_atomic(path, fieldnames, rows):
    """Write a CSV such that its existence implies its completeness.

    `.partial` -> flush -> fsync -> os.replace. os.replace is atomic within a directory on
    both POSIX and Windows. This is the invariant `checkpoint_done` relies on: without it a
    resumed run could read a half-written CSV as finished work, and there is no way to tell
    a truncated CSV from a short one after the fact.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def append_csv_rows(path, fieldnames, rows):
    """Append-only log that survives across resumed invocations."""
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def checkpoint_done(csv_path, expected_rows):
    """True if this slide's pass is already finished.

    Checks the row count as well as existence, so a file left by an older schema or a
    partially-scored slide is redone rather than trusted. Cheap -- one 324-row parse per
    slide, once.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False
    try:
        return len(read_csv_dicts(csv_path)) == expected_rows
    except (OSError, UnicodeDecodeError, csv.Error):
        return False


# --- retry ----------------------------------------------------------------------------------

def _retryable_exceptions():
    """Transient failures worth another attempt. Resolved lazily so this module imports
    without google-cloud-storage present (the aggregator and plots do not need it).
    """
    errors = [ssl.SSLError, socket.timeout, ConnectionError, TimeoutError]
    try:
        from google.api_core import exceptions as gexc

        errors += [gexc.ServerError, gexc.ServiceUnavailable, gexc.TooManyRequests,
                   gexc.InternalServerError, gexc.GatewayTimeout, gexc.BadGateway]
    except ImportError:
        pass
    try:
        from requests import exceptions as rexc

        errors += [rexc.ConnectionError, rexc.ChunkedEncodingError, rexc.Timeout]
    except ImportError:
        pass
    return tuple(errors)


RETRYABLE = _retryable_exceptions()


def _not_found_type():
    try:
        from google.api_core.exceptions import NotFound

        return NotFound
    except ImportError:
        return ()


NOT_FOUND = _not_found_type()

# `except` needs a flat tuple of exception classes -- a nested tuple raises TypeError at the
# moment the handler runs, which for a retry wrapper means only on the failure path.
RETRY_ON = RETRYABLE + (FileNotFoundError,)


def with_retry(fn, *args, attempts=3, base_delay=1.0, **kwargs):
    """Retry `fn` on transient network failures with jittered exponential backoff.

    Two deliberate choices:

    - **`FileNotFoundError` is retried.** `load_image` raises it both for a genuinely absent
      blob *and* when `cv2.imdecode` returns None on a short read. The slide index has already
      proved every blob this is called for exists, so during a pass this means a truncated
      download, which is exactly what a retry fixes. `NotFound` from the GCS layer is a real
      absence and is *not* retried.
    - **The scope is the download only.** Callers wrap `load_image`, not the whole
      score-one-FOV body, so a network blip never re-burns the ~0.5 s of feature computation
      that already succeeded.

    google-cloud-storage already retries 5xx internally; this outer layer exists for SSL
    errors and truncation, which it does not cover.
    """
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args, **kwargs)
        except NOT_FOUND:
            raise
        except RETRY_ON:
            if attempt == attempts:
                raise
            time.sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5))
    raise AssertionError("with_retry requires attempts >= 1")


# --- progress -------------------------------------------------------------------------------

class SlideProgress:
    """Per-slide progress line plus a rolling ETA, appended to a jsonl log.

    A multi-hour run over 271 slides needs to be answerable from outside the process ("is it
    still going, and how long left"), which is what the jsonl is for -- one line per completed
    slide, tailable while the run is live.
    """

    def __init__(self, pass_name, total_slides, window=10, log_dir=LOGS_DIR):
        self.pass_name = pass_name
        self.total = total_slides
        self.window = window
        self.path = Path(log_dir) / f"{pass_name}-progress.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = 0
        self.skipped = 0
        self.recent = []
        self.bytes_total = 0
        self.lock = threading.Lock()
        self.started = time.time()

    def skip(self, slide_id):
        with self.lock:
            self.skipped += 1
            print(f"[{self.pass_name}] skip {slide_id} (already complete)", flush=True)

    def record(self, slide_id, n_fovs, wall_s, n_errors=0, n_bytes=0, **extra):
        with self.lock:
            self.done += 1
            self.recent.append(wall_s)
            if len(self.recent) > self.window:
                self.recent.pop(0)
            self.bytes_total += n_bytes

            entry = {"pass": self.pass_name, "slide_id": slide_id, "n_fovs": n_fovs,
                     "wall_s": round(wall_s, 2), "n_errors": n_errors, "bytes": n_bytes,
                     "ts": time.time(), **extra}
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            remaining = self.total - self.done - self.skipped
            per_slide = sum(self.recent) / len(self.recent)
            eta_min = remaining * per_slide / 60.0
            rate = n_fovs / wall_s if wall_s else 0.0
            print(f"[{self.pass_name}] {slide_id}: {n_fovs} FOVs in {wall_s:.1f}s "
                  f"({rate:.2f} FOV/s, {n_errors} err) | "
                  f"{self.done}/{self.total - self.skipped} done, ETA {eta_min:.0f} min",
                  flush=True)

    def summary(self):
        elapsed = time.time() - self.started
        return {"pass": self.pass_name, "processed": self.done, "skipped": self.skipped,
                "elapsed_s": round(elapsed, 1), "bytes": self.bytes_total}


# --- slide list -----------------------------------------------------------------------------

def load_slide_index(path=SLIDE_INDEX_JSON):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run build_slide_index.py first (it maps each slide to its "
            "TZ2025-Box and verifies which FOV ids exist)."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_slides(slides_csv=SLIDES_CSV):
    path = Path(slides_csv)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found -- run build_slide_index.py first.")
    return read_csv_dicts(path)


def add_selection_args(parser):
    """The slide-selection flags every pass shares, so their semantics cannot drift."""
    parser.add_argument("--slides", nargs="*", default=None,
                        help="score only these slide_ids (default: all in slides.csv)")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many slides (after skipping complete ones)")
    parser.add_argument("--limit-fovs", type=int, default=None,
                        help="score only the first N FOVs of each slide (smoke tests)")
    parser.add_argument("--fov-stride", type=int, default=1,
                        help="score every Nth FOV (default 1 = all 324)")
    parser.add_argument("--force", action="store_true",
                        help="rescore slides that already have a complete CSV")
    return parser


def selected_fov_ids(index_entry, channel, args):
    """The FOV ids to score for one slide, honouring --fov-stride and --limit-fovs.

    Reads the verified ids from the slide index rather than assuming 1..324, so a slide with a
    genuine gap scores what exists instead of generating 324 NotFounds.
    """
    ids = index_entry.get(f"{channel}_fov_ids") or list(range(1, EXPECTED_FOVS + 1))
    ids = sorted(int(i) for i in ids)
    if args.fov_stride and args.fov_stride > 1:
        ids = ids[::args.fov_stride]
    if args.limit_fovs:
        ids = ids[:args.limit_fovs]
    return ids


def iter_target_slides(args, pass_name, fov_dir, channel="dpc",
                       slides_csv=SLIDES_CSV, index_path=SLIDE_INDEX_JSON):
    """Yield (slide_row, box, fov_ids, out_csv) for each slide this invocation should do.

    Applies --slides/--limit/--force and the checkpoint skip, so every pass filters
    identically. `progress` is returned alongside so the caller reports skips consistently.
    """
    slides = load_slides(slides_csv)
    index = load_slide_index(index_path)["slides"]

    wanted = set(args.slides) if args.slides else None
    if wanted:
        known = {s["slide_id"] for s in slides}
        unknown = sorted(wanted - known)
        if unknown:
            raise SystemExit(f"unknown slide_ids: {unknown}")
        slides = [s for s in slides if s["slide_id"] in wanted]

    progress = SlideProgress(pass_name, len(slides))
    targets = []
    for slide in slides:
        slide_id = slide["slide_id"]
        entry = index.get(slide_id)
        if entry is None:
            print(f"[{pass_name}] WARNING {slide_id} missing from slide index; skipping",
                  flush=True)
            continue

        fov_ids = selected_fov_ids(entry, channel, args)
        out_csv = Path(fov_dir) / f"{slide_id}.csv"
        if not args.force and checkpoint_done(out_csv, len(fov_ids)):
            progress.skip(slide_id)
            continue

        targets.append((slide, entry["box"], fov_ids, out_csv))
        if args.limit and len(targets) >= args.limit:
            break

    return targets, progress


def mirror_to_gcs(local_path, prefix=GCS_MIRROR_PREFIX):
    """Copy one finished per-slide CSV to the shared bucket.

    Belt-and-braces for Spot preemption: `--instance-termination-action=STOP` keeps the boot
    disk, but if a VM is ever lost outright these are the only artifacts that took hours to
    produce. 271 files at ~40 KB is negligible next to the 342 GB the pass already reads.
    Failures here are logged, never fatal -- losing a mirror copy must not fail a good slide.
    """
    from src.pipeline import _gcs_client, to_dir

    try:
        root = to_dir(prefix)
        local_path = Path(local_path)
        blob_name = f"{root.blob_name.rstrip('/')}/{local_path.parent.name}/{local_path.name}"
        _gcs_client().bucket(root.bucket).blob(blob_name).upload_from_filename(str(local_path))
        return True
    except Exception as exc:  # noqa: BLE001 -- mirroring is best-effort by design
        print(f"  WARNING mirror of {Path(local_path).name} failed: {exc}", flush=True)
        return False
