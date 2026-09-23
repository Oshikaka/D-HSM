#!/usr/bin/env bash
# Download StreamingBench: questions_real.json + Real-Time Visual Understanding videos.
# Only the "real" (Real-Time Visual Understanding) task is used, so the other categories are skipped.
# Target layout:
#   data/streamingbench/questions_real.json
#   data/streamingbench/videos/<video files>
set -euo pipefail

# ---------------------------------------------------------------------------
# Fill this in (or pass it on the command line):
#   ROOT=/path/to/your/workspace bash scripts/download_streamingbench.sh
# data/ is created underneath it.
# ---------------------------------------------------------------------------
ROOT="${ROOT:-}"
: "${ROOT:?ROOT is not set. Edit this script or run: ROOT=/path/to/your/workspace bash scripts/download_streamingbench.sh}"

# Run this inside the activated env (`conda activate vlm`) so `pip` and `hf`
# resolve to that env. Otherwise point PY_BIN at it, e.g.
#   PY_BIN=~/miniconda3/envs/vlm/bin
PY_BIN="${PY_BIN:-}"
PIP="${PY_BIN:+$PY_BIN/}pip"
HF="${PY_BIN:+$PY_BIN/}hf"

DATA="$ROOT/data/streamingbench"
DL="$DATA/_hf_dl"
VIDEOS="$DATA/videos"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$DL" "$VIDEOS"

echo "[1/4] $(date '+%F %T') Downloading questions_real.json"
curl -fL --retry 5 --retry-delay 5 -o "$DATA/questions_real.json" \
    https://raw.githubusercontent.com/THUNLP-MT/StreamingBench/main/src/data/questions_real.json
ls -lh "$DATA/questions_real.json"

echo "[2/4] $(date '+%F %T') Downloading Real-Time Visual Understanding_*.zip from HF"
"$PIP" show hf_transfer >/dev/null 2>&1 || \
    "$PIP" install --quiet hf_transfer
"$HF" download \
    mjuicem/StreamingBench \
    --repo-type dataset \
    --include "Real-Time Visual Understanding_*.zip" \
    --local-dir "$DL"

echo "[3/4] $(date '+%F %T') Extracting zips into $VIDEOS"
cd "$DL"
for z in "Real-Time Visual Understanding"_*.zip; do
    echo "  - unzip: $z"
    unzip -o -q "$z" -x "__MACOSX/*" "*/.DS_Store" -d "$VIDEOS"
done

echo "[4/4] $(date '+%F %T') Sanity check"
du -sh "$VIDEOS"
find "$VIDEOS" -maxdepth 2 | head
ls "$DATA"

echo "=== StreamingBench download finished at $(date '+%F %T') ==="
echo "You can remove $DL once extraction is verified."
