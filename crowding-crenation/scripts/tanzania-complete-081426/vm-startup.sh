#!/bin/bash
# Headless driver for the tanzania-complete-081426 sweep.
#
# SSH is unavailable on this project for an external account (project-level enable-oslogin=TRUE),
# so the VM runs the whole analysis unattended and reports through the shared bucket:
#   _status.txt          one line, overwritten -- what stage it is in right now
#   logs/startup.log     this script's full output, pushed every 2 min
#   logs/*.jsonl         one line per completed slide, pushed every 2 min
#   crowding/, fluorescence/  the results trees
#   _DONE.txt / _FAILED.txt   terminal marker
#
# Per-slide CSVs are also mirrored live by --mirror-gcs, so progress is visible as files appear.
set -uo pipefail

LOG=/var/log/crowding-run.log
exec > >(tee -a "$LOG") 2>&1

OUT=gs://malaria-analysis-shared/emily/tanzania-complete-081426
BUNDLE="$OUT/bundle/code.tar.gz"
WORK=/opt/run
DATASET=tanzania-complete-081426

say()    { echo "=== $(date -Is) $* ==="; }
status() { echo "$(date -Is) $*" | gcloud storage cp -q - "$OUT/_status.txt" 2>/dev/null || true; }
pushlog() {
  gcloud storage cp -q "$LOG" "$OUT/logs/startup.log" 2>/dev/null || true
  gcloud storage cp -q -r "$WORK/crowding-crenation/data/results/$DATASET/logs" "$OUT/logs-crowding" 2>/dev/null || true
  gcloud storage cp -q -r "$WORK/fluorescence/data/results/$DATASET/logs" "$OUT/logs-fluorescence" 2>/dev/null || true
}
fail() { say "FAILED at: $*"; status "FAILED at: $*"; pushlog; echo "$*" | gcloud storage cp -q - "$OUT/_FAILED.txt" 2>/dev/null || true; exit 1; }

say "startup begin on $(hostname), $(nproc) vCPU"
status "installing dependencies"

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || fail "apt-get update"
# libgl1 + libglib2.0-0 are opencv-python's runtime shared libs; without them `import cv2`
# fails with a bare ImportError about libGL.so.1 on a minimal Debian image.
apt-get install -y -qq python3-venv python3-pip libgl1 libglib2.0-0 || fail "apt-get install"

say "fetching code bundle"
status "fetching code bundle"
mkdir -p "$WORK" && cd "$WORK" || fail "mkdir $WORK"
gcloud storage cp "$BUNDLE" ./code.tar.gz || fail "download bundle"
tar xzf code.tar.gz || fail "untar bundle"

say "creating venv"
status "installing python packages"
python3 -m venv .venv || fail "venv"
PY="$WORK/.venv/bin/python"
"$PY" -m pip install -q --upgrade pip || fail "pip upgrade"
"$PY" -m pip install -q -r crowding-crenation/requirements.txt || fail "pip crowding reqs"
"$PY" -m pip install -q -r fluorescence/requirements.txt || fail "pip fluorescence reqs"
"$PY" -c "import cv2, numpy, skimage, scipy, matplotlib; print('imports ok', cv2.__version__)" || fail "import check"

# Background reporter: the only way to watch a headless run.
( while true; do sleep 120; pushlog; done ) &
WATCHER=$!
trap 'kill $WATCHER 2>/dev/null || true' EXIT

CROWD="$WORK/crowding-crenation"
FLUOR="$WORK/fluorescence"

say "gate check"
status "gate check"
cd "$CROWD" || fail "cd crowding"
"$PY" scripts/combined/check_empty_field_gate.py || fail "empty-field gate check"

say "golden-value regression (KTR-72502946 vs committed v2.1 scores)"
status "golden-value regression"
"$PY" scripts/tanzania-complete-081426/verify_regression.py --limit-fovs 40 --threads 8 \
  || fail "golden-value regression"

say "throughput bench"
status "throughput bench"
"$PY" scripts/tanzania-complete-081426/bench_throughput.py \
  --fovs 128 --threads-list 4 8 16 24 --procs-list 1 2 4 8 || fail "bench"

BEST_JSON="$CROWD/data/results/$DATASET/bench-throughput.json"
PROCS=$("$PY" -c "import json;print(json.load(open(r'$BEST_JSON'))['best']['procs'])") || fail "read bench procs"
THREADS=$("$PY" -c "import json;print(json.load(open(r'$BEST_JSON'))['best']['threads'])") || fail "read bench threads"
say "chosen config: --procs $PROCS --threads $THREADS"

say "crowding pass (87,799 DPC FOVs)"
status "crowding pass: procs=$PROCS threads=$THREADS"
"$PY" scripts/tanzania-complete-081426/run_crowding_pass.py \
  --procs "$PROCS" --threads "$THREADS" --mirror-gcs || fail "crowding pass"

say "overexposure pass (87,801 fluorescence FOVs)"
status "overexposure pass: procs=$PROCS threads=8"
cd "$FLUOR" || fail "cd fluorescence"
# Same process fan-out as the crowding pass. Without --procs this ran single-process at 19.8 FOV/s
# against the crowding pass's 31.9 -- threads saturate on the GIL long before the machine does.
# Reuses the bench's process count: the bench measures the *crowding* workload (more compute per
# FOV, smaller blobs), so this is a reasonable transfer rather than a separately tuned number.
# Threads stay at 8, the top of the measured GCS-bound range.
"$PY" scripts/tanzania-complete-081426/run_overexposure_pass.py \
  --procs "$PROCS" --threads 8 || fail "overexposure pass"

say "crop-outlier pass"
status "crop-outlier pass"
"$PY" scripts/tanzania-complete-081426/run_crop_outlier_pass.py --threads 16 --assert-sources \
  || fail "crop-outlier pass"

say "aggregate + plots"
status "aggregating"
cd "$CROWD" || fail "cd crowding"
"$PY" scripts/tanzania-complete-081426/aggregate_slides.py || fail "aggregate"
"$PY" scripts/tanzania-complete-081426/plot_slide_level.py || fail "plots"
"$PY" scripts/tanzania-complete-081426/plot_slide_level.py --color-by truth || true
"$PY" scripts/tanzania-complete-081426/plot_slide_level.py --color-by crop_outlier || true

say "uploading results"
status "uploading results"
gcloud storage cp -r "$CROWD/data/results/$DATASET" "$OUT/crowding" || fail "upload crowding"
gcloud storage cp -r "$FLUOR/data/results/$DATASET" "$OUT/fluorescence" || fail "upload fluorescence"

kill $WATCHER 2>/dev/null || true
pushlog
say "ALL DONE"
status "DONE"
date -Is | gcloud storage cp -q - "$OUT/_DONE.txt" 2>/dev/null || true

# Best-effort self-stop so an unattended run does not bill after finishing. The attached service
# account may not hold compute.instances.stop; if not, this logs and the 8 h maxRunDuration
# (terminationAction=STOP) is the backstop.
say "attempting self-stop"
gcloud compute instances stop crowding-tz-081426 --zone=us-central1-a --quiet \
  || say "self-stop not permitted for this service account -- stop it manually or wait for maxRunDuration"
