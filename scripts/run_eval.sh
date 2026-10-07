#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-VL-7B-Instruct}"
EMBED_MODEL="${EMBED_MODEL:-BAAI/bge-small-en-v1.5}"
DATA_ROOT="${DATA_ROOT:?set DATA_ROOT to the directory holding ovo_bench/ and streamingbench/}"
OVO_ANNO="${OVO_ANNO:-$DATA_ROOT/ovo_bench/ovo_bench_new.json}"
OVO_VIDEOS="${OVO_VIDEOS:-$DATA_ROOT/ovo_bench/chunked_videos}"
SB_ANNO="${SB_ANNO:-$DATA_ROOT/streamingbench/questions_real.json}"
SB_VIDEOS="${SB_VIDEOS:-$DATA_ROOT/streamingbench/videos}"
BENCH="${BENCH:-both}"
NPROC="${NPROC:-2}"
RECENT_FRAMES="${RECENT_FRAMES:-4}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
OVO_MCQ_PROMPT_POLICY="${OVO_MCQ_PROMPT_POLICY:-uniform_abstention}"
OVO_HISTORY_MODE="${OVO_HISTORY_MODE:-dhsm}"
OUT="${OUT:-results/keyword_${OVO_MCQ_PROMPT_POLICY}_${OVO_HISTORY_MODE}_${RECENT_FRAMES}f}"

case "$BENCH" in
  ovo|sb|both) ;;
  *) echo "BENCH must be ovo, sb, or both" >&2; exit 2 ;;
esac
if [[ "${ROUTING:-keyword}" != keyword ]]; then
  echo "Only keyword routing is supported; task-label routing has been removed." >&2
  exit 2
fi
if [[ "$OVO_HISTORY_MODE" == recent_only ]]; then
  OVO_SPLITS="${OVO_SPLITS:-backward,realtime}"
else
  OVO_SPLITS="${OVO_SPLITS:-backward,realtime,forward}"
fi

export FORCE_QWENVL_VIDEO_READER=torchcodec
export TORCHCODEC_NUM_THREADS="${TORCHCODEC_NUM_THREADS:-2}"
export MIN_PIXELS="${MIN_PIXELS:-50176}"
export MAX_PIXELS="${MAX_PIXELS:-262144}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

COMMON=(
  --model_path "$MODEL_PATH"
  --embed_model "$EMBED_MODEL"
  --attn_implementation "$ATTN_IMPLEMENTATION"
  --recent_frames_only "$RECENT_FRAMES"
  --chunk_duration 1.0
  --fps 1.0
  --max_qa_tokens 256
  --extract_every_n_chunks 1
  --max_extraction_chunks 20
  --sim_threshold 0.25
  --top_k dynamic
  --caption_batch_size 4
  --routing keyword
  --memory_mode entity_resolved
)
OVO_OPTIONS=()
if [[ -n "${OVO_MEMORY_FLOOR:-}" ]]; then
  OVO_OPTIONS+=(--memory_floor "$OVO_MEMORY_FLOOR")
fi
if [[ -n "${OVO_GATE_STRICT_SIM:-}" ]]; then
  OVO_OPTIONS+=(--gate_strict_sim "$OVO_GATE_STRICT_SIM")
fi
if [[ -n "${OVO_DYNAMIC_TOP_K_MAX:-}" ]]; then
  OVO_OPTIONS+=(--dynamic_top_k_max "$OVO_DYNAMIC_TOP_K_MAX")
fi
SB_OPTIONS=()
if [[ -n "${SB_MEMORY_FLOOR:-}" ]]; then
  SB_OPTIONS+=(--memory_floor "$SB_MEMORY_FLOOR")
fi
if [[ -n "${SB_GATE_STRICT_SIM:-}" ]]; then
  SB_OPTIONS+=(--gate_strict_sim "$SB_GATE_STRICT_SIM")
fi
if [[ -n "${SB_DYNAMIC_TOP_K_MAX:-}" ]]; then
  SB_OPTIONS+=(--dynamic_top_k_max "$SB_DYNAMIC_TOP_K_MAX")
fi
if [[ -n "${SB_SPOKE_ATTACH_THRESHOLD:-}" ]]; then
  SB_OPTIONS+=(--spoke_attach_threshold "$SB_SPOKE_ATTACH_THRESHOLD")
fi
mkdir -p "$OUT"

if [[ "$BENCH" == ovo || "$BENCH" == both ]]; then
  accelerate launch --num_processes "$NPROC" \
    experiments/evaluate_ovo.py "${COMMON[@]}" \
    --anno_path "$OVO_ANNO" --chunked_dir "$OVO_VIDEOS" \
    --result_dir "$OUT/ovo" \
    ${OVO_OPTIONS[@]+"${OVO_OPTIONS[@]}"} \
    --mcq_prompt_policy "$OVO_MCQ_PROMPT_POLICY" \
    --history_mode "$OVO_HISTORY_MODE" --splits "$OVO_SPLITS" \
    --count_question_max_chunks 20 --use_logits \
    2>&1 | tee "$OUT/ovo.log"
fi

if [[ "$BENCH" == sb || "$BENCH" == both ]]; then
  accelerate launch --num_processes "$NPROC" \
    experiments/evaluate_streamingbench.py "${COMMON[@]}" \
    --anno_path "$SB_ANNO" --video_dir "$SB_VIDEOS" \
    --result_dir "$OUT/sb" \
    ${SB_OPTIONS[@]+"${SB_OPTIONS[@]}"} \
    2>&1 | tee "$OUT/sb.log"
fi

echo "Results: $OUT"
