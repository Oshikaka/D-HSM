#!/usr/bin/env bash
# Download the model weights: the frozen VLM backbone + the retrieval embedder.
# Target layout:
#   models/Qwen2.5-VL-7B-Instruct/
#   models/bge-small-en-v1.5/
set -euo pipefail

# ---------------------------------------------------------------------------
# Fill this in (or pass it on the command line):
#   ROOT=/path/to/your/workspace bash scripts/download_models.sh
# models/ is created underneath it.
# ---------------------------------------------------------------------------
ROOT="${ROOT:-}"
: "${ROOT:?ROOT is not set. Edit this script or run: ROOT=/path/to/your/workspace bash scripts/download_models.sh}"

# Run this inside the activated env (`conda activate vlm`) so `pip` and `hf`
# resolve to that env. Otherwise point PY_BIN at it, e.g.
#   PY_BIN=~/miniconda3/envs/vlm/bin
PY_BIN="${PY_BIN:-}"
PIP="${PY_BIN:+$PY_BIN/}pip"
HF="${PY_BIN:+$PY_BIN/}hf"

# Swap the backbone here if you are reproducing a different row of the paper,
# e.g. VLM_REPO=Qwen/Qwen3-VL-8B-Instruct.
VLM_REPO="${VLM_REPO:-Qwen/Qwen2.5-VL-7B-Instruct}"
EMBED_REPO="${EMBED_REPO:-BAAI/bge-small-en-v1.5}"

MODELS="$ROOT/models"
export HF_HUB_ENABLE_HF_TRANSFER=1

mkdir -p "$MODELS"

echo "[1/3] $(date '+%F %T') Downloading $VLM_REPO (~16GB)"
"$PIP" show hf_transfer >/dev/null 2>&1 || \
    "$PIP" install --quiet hf_transfer
"$HF" download "$VLM_REPO" --local-dir "$MODELS/${VLM_REPO##*/}"

echo "[2/3] $(date '+%F %T') Downloading $EMBED_REPO (~130MB)"
"$HF" download "$EMBED_REPO" --local-dir "$MODELS/${EMBED_REPO##*/}"

echo "[3/3] $(date '+%F %T') Sanity check"
du -sh "$MODELS"/*
ls "$MODELS/${VLM_REPO##*/}"

echo "=== Model download finished at $(date '+%F %T') ==="
echo "Point the evaluators at them:"
echo "  --model_path  $MODELS/${VLM_REPO##*/}"
echo "  --embed_model $MODELS/${EMBED_REPO##*/}"
