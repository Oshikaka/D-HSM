#!/usr/bin/env bash
# Evaluate D-HSM on OVO-Bench and/or StreamingBench.
#
#   BENCH=ovo  DATA_ROOT=/path/to/data  scripts/run_eval.sh
#   BENCH=sb   DATA_ROOT=/path/to/data  scripts/run_eval.sh
#   BENCH=both DATA_ROOT=/path/to/data  scripts/run_eval.sh     # default
#
set -euo pipefail

# --- edit these for your machine -------------------------------------------
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-small-en-v1.5}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the directory holding ovo_bench/ and streamingbench/}"
OVO_ANNO="${OVO_ANNO:-$DATA_ROOT/ovo_bench/ovo_bench_new.json}"
OVO_VIDEOS="${OVO_VIDEOS:-$DATA_ROOT/ovo_bench/chunked_videos}"
SB_ANNO="${SB_ANNO:-$DATA_ROOT/streamingbench/questions_real.json}"
SB_VIDEOS="${SB_VIDEOS:-$DATA_ROOT/streamingbench/videos}"
BENCH="${BENCH:-both}"          # ovo | sb | both
NPROC="${NPROC:-2}"             # GPUs

# Routing: how D-HSM decides whether a question needs memory at all.
#
#   task_label — read the benchmark's own annotation (OVO task EPM/ASI/HLD vs
#                the Real-Time tasks; StreamingBench required_ability).  This
#                is the setting the reported table numbers were produced with,
#                so it is what this script uses.  It consumes ground-truth
#                metadata at inference time and therefore does not generalise
#                to an arbitrary streaming question.
#
#   keyword    — the paper's §3.3 gate: decide from the question text alone
#                (dhsm/retrieval_gate.py).  This is the default in the code
#                (`--routing` defaults to keyword) and the only mode that runs
#                on questions with no benchmark annotation.
#
# To switch: set ROUTING=keyword (or just drop --routing, since keyword is the
# code default).  Running both and diffing gives the cost of the gate:
#
#   ROUTING=task_label OUT=results/task_label scripts/run_eval.sh
#   ROUTING=keyword    OUT=results/keyword    scripts/run_eval.sh
#   python experiments/compare_gate_ab.py \
#       --ovo_baseline results/task_label/ovo --ovo_gated results/keyword/ovo \
#       --sb_baseline  results/task_label/sb  --sb_gated  results/keyword/sb
#
ROUTING="${ROUTING:-task_label}"
GATE_STRICT_SIM="${GATE_STRICT_SIM:-0.55}"   # ambiguous-bucket floor; keyword only
OUT="${OUT:-results/$ROUTING}"
# ---------------------------------------------------------------------------

# Paper defaults (§4.1): 20 historical chunks, 4 recent frames, dynamic cutoff
# capped at K=12, bge-small-en-v1.5 embeddings.
COMMON=(
  --model_path "$MODEL_PATH"
  --embed_model "$EMBED_MODEL"
  --recent_frames_only 4
  --chunk_duration 1.0
  --fps 1.0
  --max_qa_tokens 256
  --extract_every_n_chunks 1
  --max_extraction_chunks 20
  --sim_threshold 0.25
  --top_k dynamic
  --caption_batch_size 4
  --routing "$ROUTING"
  --gate_strict_sim "$GATE_STRICT_SIM"
)

mkdir -p "$OUT"

if [[ "$BENCH" == "ovo" || "$BENCH" == "both" ]]; then
  echo "=== OVO-Bench  routing=$ROUTING ==="
  accelerate launch --num_processes "$NPROC" \
    experiments/evaluate_ovo.py \
    "${COMMON[@]}" \
    --anno_path "$OVO_ANNO" \
    --chunked_dir "$OVO_VIDEOS" \
    --result_dir "$OUT/ovo" \
    --use_logits \
    2>&1 | tee "$OUT/ovo.log"
fi

if [[ "$BENCH" == "sb" || "$BENCH" == "both" ]]; then
  echo "=== StreamingBench  routing=$ROUTING ==="
  accelerate launch --num_processes "$NPROC" \
    experiments/evaluate_streamingbench.py \
    "${COMMON[@]}" \
    --anno_path "$SB_ANNO" \
    --video_dir "$SB_VIDEOS" \
    --result_dir "$OUT/sb" \
    2>&1 | tee "$OUT/sb.log"
fi

echo
echo "Scores:  $OUT/*/scores_report.json"
echo "Per-question gate decisions (keyword routing) are logged in each results"
echo "row under gate_bucket / gate_memory_cues / gate_recent_cues."
