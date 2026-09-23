#!/usr/bin/env bash
# Download OVO-Bench annotations + chunked videos.
# Target layout:
#   data/ovo_bench/ovo_bench_new.json
#   data/ovo_bench/chunked_videos/...
set -euo pipefail

# ---------------------------------------------------------------------------
# Fill this in (or pass it on the command line):
#   ROOT=/path/to/your/workspace bash scripts/download_ovo.sh
# data/ is created underneath it.
# ---------------------------------------------------------------------------
ROOT="${ROOT:-}"
: "${ROOT:?ROOT is not set. Edit this script or run: ROOT=/path/to/your/workspace bash scripts/download_ovo.sh}"

# Run this inside the activated env (`conda activate vlm`) so `pip` and `hf`
# resolve to that env. Otherwise point PY_BIN at it, e.g.
#   PY_BIN=~/miniconda3/envs/vlm/bin
PY_BIN="${PY_BIN:-}"
PIP="${PY_BIN:+$PY_BIN/}pip"
HF="${PY_BIN:+$PY_BIN/}hf"

DATA="$ROOT/data/ovo_bench"
DL="$DATA/_hf_dl"           # raw tar parts from HF
CHUNKED="$DATA/chunked_videos"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$DL"

echo "[1/4] $(date '+%F %T') Downloading annotations ovo_bench_new.json"
curl -fL --retry 5 --retry-delay 5 -o "$DATA/ovo_bench_new.json" \
    https://raw.githubusercontent.com/JoeLeelyf/OVO-Bench/main/data/ovo_bench_new.json
ls -lh "$DATA/ovo_bench_new.json"

echo "[2/4] $(date '+%F %T') Downloading chunked_videos.tar.part[aa-ao] from HF (~144GB)"
# hf_transfer speeds this up substantially. Falls back gracefully if not installed.
"$PIP" show hf_transfer >/dev/null 2>&1 || \
    "$PIP" install --quiet hf_transfer
"$HF" download \
    JoeLeelyf/OVO-Bench \
    --repo-type dataset \
    --include "chunked_videos.tar.part*" \
    --local-dir "$DL"

echo "[3/4] $(date '+%F %T') Concatenating + extracting tar parts into $DATA"
cd "$DL"
# Stream concat → tar to avoid the extra 144GB intermediate file.
# The archive root is chunked_videos/, so extract to $DATA (parent), not $CHUNKED.
cat chunked_videos.tar.parta[a-o] | tar -xvf - -C "$DATA" | tail -5
echo "Extraction complete."

echo "[4/4] $(date '+%F %T') Sanity check"
du -sh "$CHUNKED"
find "$CHUNKED" -maxdepth 2 -type d | head
ls "$DATA"

echo "=== OVO-Bench download finished at $(date '+%F %T') ==="
echo "You can remove $DL to free ~144 GB once extraction is verified."
