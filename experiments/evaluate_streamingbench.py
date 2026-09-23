"""
Hub-and-Spoke StreamingBench Evaluation
=======================================

StreamingBench evaluator with three readout families:

  - graph_memory
      Episodic-memory questions, except episodic Counting. Build a
      Hub-and-Spoke memory from historical chunks and answer with retrieved
      graph context + recent frames + A/B/C/D logit scoring.

  - recent_window
      Working-memory/current-frame questions. Answer directly from the latest
      visual window with A/B/C/D logit scoring.

  - interval_state
      Episodic Counting questions. Between adjacent question cutoffs for the
      same count question, predict a count delta and accumulate a state, then
      map the accumulated count back to the provided A/B/C/D options.

The data layout follows StreamingBench's questions_real.json:

  [
    {
      "time": "[0:00:00 - 0:00:10]",
      "video_path": "./videos/sample_348_real.mp4",
      "video_categories": "...",
      "questions": [
        {
          "task_type": "Clips Summarize",
          "required_ability": "episodic memory",
          "time_stamp": "00:00:31",
          "question": "...",
          "answer": "C",
          "options": ["A. ...", "B. ...", "C. ...", "D. ..."]
        }
      ]
    }
  ]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "86400")

from accelerate import Accelerator, InitProcessGroupKwargs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dhsm.hub_and_spoke import (
    CAPTION_PROMPT,
    DYNAMIC_TOP_K_MAX,
    DEFAULT_EMBED_MODEL,
    HubAndSpokeEvaluator,
    HubAndSpokeMemory,
    _is_count_question,
    round_sims,
)
from dhsm.retrieval_gate import (
    DEFAULT_GATE_STRICT_SIM,
    MEMORY_MODE_CHOICES,
    ROUTING_CHOICES,
    ROUTING_KEYWORD,
    ROUTING_TASK_LABEL,
    GateDecision,
    evaluator_class_for,
    gate_question,
    resolve_memory_mode,
    strip_prompt_scaffolding,
)
from dhsm import shard_io
from dhsm.video_qa import decode_video_to_chunks_qwen
from dhsm.video_qa_qwen3 import RecentWindowQAModel


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


_DYNAMIC_TOP_K_CHOICES = {"dynamic", "auto", "adaptive"}


def parse_top_k_arg(raw: str) -> int | str:
    token = str(raw).strip().lower()
    if token in _DYNAMIC_TOP_K_CHOICES:
        return "dynamic"
    try:
        top_k = int(token)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "top_k must be a positive integer or one of: dynamic/auto/adaptive."
        ) from exc
    if top_k < 1:
        raise argparse.ArgumentTypeError("top_k must be >= 1.")
    return top_k


def format_top_k(top_k: int | str) -> str:
    if isinstance(top_k, str):
        return f"{top_k}(max={DYNAMIC_TOP_K_MAX})"
    return str(top_k)


PROMPT_TEMPLATE = (
    "You are an advanced video question-answering AI assistant. "
    "You have been provided with video frames and a multiple-choice question. "
    "Analyze only visual evidence available up to the current timestamp and "
    "choose the best answer.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Only give the best option's letter (A, B, C, or D) directly."
)


COUNT_DELTA_PROMPT_TEMPLATE = (
    "You are updating a running count for a streaming video question.\n\n"
    "Question: {question}\n"
    "Previous query time: {previous_time}\n"
    "Current query time: {current_time}\n"
    "Current count before this interval: {current_count}\n\n"
    "{context_note}"
    "Look at the provided chronological frames. Count only NEW instances that "
    "happen AFTER the previous query time and AT OR BEFORE the current query "
    "time. Do not count the same instance twice. If the evidence is unclear, "
    "count 0 new instances.\n\n"
    "Choose the best option:\n"
    "{options}\n\n"
    "Return only the letter."
)


ROUTE_GRAPH = "graph_memory"
ROUTE_RECENT = "recent_window"
ROUTE_INTERVAL = "interval_state"

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def extract_mcq_letter(response: Any) -> str | None:
    """A/B/C/D from free text, bare 1-4 as fallback."""
    if response is None or not str(response).strip():
        return None
    text = str(response).strip().upper()
    if m := re.search(r"\b([A-D])\b", text):
        return m.group(1)
    if m := re.search(r"\b([1-4])\b", text):
        return chr(64 + int(m.group(1)))
    return None


def timestamp_to_seconds(ts: Any) -> float:
    text = str(ts).strip()
    parts = [p for p in text.split(":") if p != ""]
    if not parts:
        return 0.0
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", text)]
    if len(nums) >= 3:
        return nums[-3] * 3600 + nums[-2] * 60 + nums[-1]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return nums[0]


def fmt_time(seconds: float) -> str:
    s = int(max(0.0, seconds))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def make_key(video_basename: str, question: dict[str, Any], question_limit: int = 80) -> str:
    return (
        f"{video_basename}_{question.get('time_stamp', '')}_"
        f"{question.get('task_type', '')}_{question.get('question', '')[:question_limit]}"
    )


def _row_key(row: dict[str, Any]) -> str:
    """Reconstruct a checkpoint key from an already-stripped result row."""
    return make_key(row.get("video", ""), row, 80)


def normalize_question_key(question: str) -> str:
    text = re.sub(r"\s+", " ", question.lower().strip())
    text = re.sub(r"[^\w\s]+", "", text)
    return text


def format_options(options: list[str]) -> str:
    formatted: list[str] = []
    for i, opt in enumerate(options):
        text = str(opt).strip()
        prefix = f"{chr(65 + i)}."
        if not re.match(r"^[A-Z]\s*[\.\)]", text):
            text = f"{prefix} {text}"
        formatted.append(text)
    return "\n".join(formatted)


def build_prompt(question: dict[str, Any]) -> str:
    return PROMPT_TEMPLATE.format(
        question=question.get("question", ""),
        options=format_options(question.get("options", [])),
    )


def build_delta_prompt(
    question: dict[str, Any],
    previous_cutoff: float,
    current_cutoff: float,
    current_count: int,
    max_delta: int,
    has_context: bool,
) -> str:
    options = [
        f"{chr(65 + i)}. {i} new instance(s)"
        for i in range(max_delta + 1)
    ]
    unclear_letter = chr(65 + max_delta + 1)
    options.append(f"{unclear_letter}. unclear, count 0 new instances")
    context_note = (
        "Some early frames may be context from just before the interval; use "
        "them only to understand continuity, not to count earlier events.\n\n"
        if has_context else ""
    )
    return COUNT_DELTA_PROMPT_TEMPLATE.format(
        question=question.get("question", ""),
        previous_time=fmt_time(previous_cutoff),
        current_time=fmt_time(current_cutoff),
        current_count=current_count,
        context_note=context_note,
        options="\n".join(options),
    )


def resolve_video_path(video_path: str, video_dir: str) -> str:
    if video_path.startswith("./videos/"):
        candidate = os.path.join(video_dir, os.path.basename(video_path))
        if os.path.exists(candidate):
            return candidate
        stem, _ = os.path.splitext(os.path.basename(video_path))
        if stem.endswith("_real"):
            stem = stem[: -len("_real")]
        return os.path.join(video_dir, stem, "video.mp4")
    if not os.path.isabs(video_path):
        return os.path.join(video_dir, video_path)
    return video_path


def route_question_by_task_label(question: dict[str, Any]) -> str:
    """Original routing: reads the benchmark's own ``required_ability`` and
    ``task_type`` annotations, i.e. ground-truth metadata.  Kept only so the
    pre-gate numbers can be reproduced with ``--routing task_label``."""
    task = str(question.get("task_type", "")).strip()
    ability = str(question.get("required_ability") or "").strip().lower()

    if task == "Counting" and ability == "episodic memory":
        return ROUTE_INTERVAL
    if ability == "episodic memory":
        return ROUTE_GRAPH
    if ability == "" and task == "Clips Summarize":
        return ROUTE_GRAPH
    return ROUTE_RECENT


def route_question_by_keyword(question: dict[str, Any]) -> tuple[str, GateDecision]:
    """Paper-faithful routing: decide from the question text alone.

    Counting questions that also need history keep the interval-state readout,
    exactly as before — but "is this a counting question" now comes from the
    same surface-phrase test the memory already uses (``_is_count_question``)
    rather than from ``task_type``.
    """
    question_text = str(question.get("question", ""))
    decision = gate_question(question_text, question.get("options") or [])
    if not decision.needs_memory:
        return ROUTE_RECENT, decision
    if _is_count_question(strip_prompt_scaffolding(question_text)):
        return ROUTE_INTERVAL, decision
    return ROUTE_GRAPH, decision


def route_question(question: dict[str, Any], routing: str = ROUTING_KEYWORD) -> str:
    """Route ``question`` under the selected routing mode."""
    if routing == ROUTING_TASK_LABEL:
        return route_question_by_task_label(question)
    return route_question_by_keyword(question)[0]


def option_count_value(option: str) -> int | None:
    text = str(option).strip()
    text = re.sub(r"^[A-Z]\s*[\.\)]\s*", "", text, flags=re.IGNORECASE)
    digit = re.search(r"\d+", text)
    if digit:
        return int(digit.group())
    lower = text.lower()
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lower):
            return value
    return None


def count_to_option_letter(count: int, options: list[str]) -> str:
    parsed: list[tuple[str, int]] = []
    for i, opt in enumerate(options):
        value = option_count_value(opt)
        if value is None:
            continue
        letter = chr(65 + i)
        parsed.append((letter, value))
        if value == count:
            return letter
    if parsed:
        return min(parsed, key=lambda item: abs(item[1] - count))[0]
    return "A"


def answer_gt_letter(question: dict[str, Any]) -> str:
    return (
        extract_mcq_letter(str(question.get("answer", "")))
        or str(question.get("answer", "")).strip().upper()
    )


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    def empty_row() -> dict[str, int]:
        return {"total": 0, "correct": 0}

    by_task: dict[str, dict[str, int]] = defaultdict(empty_row)
    by_route: dict[str, dict[str, int]] = defaultdict(empty_row)
    total = correct = errors = 0

    for result in results:
        task = str(result.get("task_type", "unknown")).strip() or "unknown"
        route = str(result.get("route", "unknown")).strip() or "unknown"
        by_task[task]["total"] += 1
        by_route[route]["total"] += 1
        if result.get("correct", False):
            by_task[task]["correct"] += 1
            by_route[route]["correct"] += 1
            correct += 1
        if result.get("error"):
            errors += 1
        total += 1

    def rows(mapping: dict[str, dict[str, int]], key_name: str) -> list[dict[str, Any]]:
        out = []
        for key in sorted(mapping):
            row = mapping[key]
            t, c = int(row["total"]), int(row["correct"])
            out.append(
                {
                    key_name: key,
                    "total": t,
                    "correct": c,
                    "accuracy": (100.0 * c / t) if t else 0.0,
                }
            )
        return out

    return {
        "overall": {
            "total": total,
            "correct": correct,
            "accuracy": (100.0 * correct / total) if total else 0.0,
        },
        "error_count": errors,
        "routes": rows(by_route, "route"),
        "tasks": rows(by_task, "task_type"),
    }


def print_summary(results: list[dict[str, Any]], label: str) -> None:
    summary = compute_summary(results)
    print("\n" + "=" * 60)
    print(f"StreamingBench {label} Results")
    print("=" * 60)
    print("\nRoutes:")
    for row in summary["routes"]:
        print(
            f"  {row['route']}: {row['accuracy']:.2f}% "
            f"({row['correct']}/{row['total']})"
        )
    print("\nTasks:")
    for row in summary["tasks"]:
        print(
            f"  {row['task_type']}: {row['accuracy']:.2f}% "
            f"({row['correct']}/{row['total']})"
        )
    overall = summary["overall"]
    print(
        f"\n  Overall: {overall['accuracy']:.2f}% "
        f"({overall['correct']}/{overall['total']})"
    )
    print(f"  Errors: {summary['error_count']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Per-video chunk cache
# ---------------------------------------------------------------------------

@dataclass
class CachedChunk:
    chunk_index: int
    start_time: float
    end_time: float
    representative_frame: Any
    representative_ts: float
    frames: list[Any] = field(default_factory=list)
    frame_timestamps: list[float] = field(default_factory=list)
    caption: str | None = None


@dataclass
class CountState:
    previous_cutoff: float = 0.0
    count: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)
    memory: HubAndSpokeMemory = field(default_factory=HubAndSpokeMemory)


def upsert_cached_chunk(
    chunk_cache: dict[int, CachedChunk],
    chunk_order: list[int],
    chunk: Any,
) -> None:
    if not getattr(chunk, "frames", None):
        return

    frame_timestamps = list(getattr(chunk, "frame_timestamps", []) or [])
    if len(frame_timestamps) != len(chunk.frames):
        span = max(float(chunk.end_time) - float(chunk.start_time), 1e-6)
        frame_timestamps = [
            float(chunk.start_time) + span * (i + 0.5) / len(chunk.frames)
            for i in range(len(chunk.frames))
        ]
    mid = len(chunk.frames) // 2
    representative_frame = chunk.frames[mid]
    representative_ts = float(frame_timestamps[mid])

    existing = chunk_cache.get(chunk.chunk_index)
    if existing is None:
        chunk_cache[chunk.chunk_index] = CachedChunk(
            chunk_index=int(chunk.chunk_index),
            start_time=float(chunk.start_time),
            end_time=float(chunk.end_time),
            representative_frame=representative_frame,
            representative_ts=representative_ts,
            frames=list(chunk.frames),
            frame_timestamps=frame_timestamps,
        )
        chunk_order.append(int(chunk.chunk_index))
        return

    existing.start_time = min(existing.start_time, float(chunk.start_time))
    existing.end_time = max(existing.end_time, float(chunk.end_time))
    existing.frames.extend(chunk.frames)
    existing.frame_timestamps.extend(frame_timestamps)
    if existing.frames:
        mid = len(existing.frames) // 2
        existing.representative_frame = existing.frames[mid]
        if existing.frame_timestamps and mid < len(existing.frame_timestamps):
            existing.representative_ts = float(existing.frame_timestamps[mid])
        else:
            existing.representative_ts = (existing.start_time + existing.end_time) / 2.0


def drop_old_recent_frames(
    chunk_cache: dict[int, CachedChunk],
    chunk_order: list[int],
    keep_last: int,
) -> None:
    keep_ids = set(chunk_order[-max(1, keep_last):])
    for chunk_id in chunk_order:
        if chunk_id not in keep_ids:
            chunk_cache[chunk_id].frames = []
            chunk_cache[chunk_id].frame_timestamps = []


def select_history_chunk_ids(
    chunk_ids: list[int],
    extract_every_n_chunks: int,
    max_extraction_chunks: int,
) -> list[int]:
    stride = max(1, int(extract_every_n_chunks))
    strided = chunk_ids[::stride]
    cap = max(0, int(max_extraction_chunks))
    if cap and len(strided) > cap:
        step = len(strided) / cap
        strided = [strided[int(i * step)] for i in range(cap)]
    return strided


def ensure_captions(
    chunks: list[CachedChunk],
    qa_model: RecentWindowQAModel,
    caption_batch_size: int,
) -> None:
    pending = [
        c for c in chunks
        if c.caption is None and c.representative_frame is not None
    ]
    if not pending:
        return
    bs = max(0, int(caption_batch_size))
    if bs <= 1:
        for chunk in pending:
            chunk.caption = qa_model.generate_from_frames(
                [chunk.representative_frame],
                CAPTION_PROMPT,
            ) or ""
        return
    for start in range(0, len(pending), bs):
        group = pending[start:start + bs]
        captions = qa_model.batch_caption_from_frames(
            [[c.representative_frame] for c in group],
            CAPTION_PROMPT,
        )
        for chunk, caption in zip(group, captions):
            chunk.caption = caption or ""


def build_memory_from_cached_chunks(
    selected_chunks: list[CachedChunk],
    qa_model: RecentWindowQAModel,
    embed_model: str,
    embed_device: str,
    sim_threshold: float,
    caption_batch_size: int,
    evaluator: HubAndSpokeEvaluator | None = None,
) -> HubAndSpokeMemory:
    ensure_captions(selected_chunks, qa_model, caption_batch_size)
    if evaluator is not None:
        # Respects the evaluator's memory type (flat_caption ablation) and
        # expand_retrieval flag.
        memory = evaluator._make_memory()
    else:
        memory = HubAndSpokeMemory(
            embed_model=embed_model,
            embed_device=embed_device,
            sim_threshold=sim_threshold,
        )
    for chunk in selected_chunks:
        if chunk.caption:
            memory.update(chunk.caption, chunk.representative_ts)
    return memory


def recent_frames_for_answer(
    chunk_cache: dict[int, CachedChunk],
    chunk_order: list[int],
    window: int,
) -> list[Any]:
    frames: list[Any] = []
    for chunk_id in chunk_order[-max(1, window):]:
        chunk = chunk_cache[chunk_id]
        if chunk.frames:
            frames.extend(chunk.frames)
        elif chunk.representative_frame is not None:
            frames.append(chunk.representative_frame)
    return frames


def cached_frame_items_between(
    chunk_cache: dict[int, CachedChunk],
    chunk_order: list[int],
    start_time: float,
    end_time: float,
    context_seconds: float,
) -> tuple[list[tuple[float, Any]], bool]:
    lower = max(0.0, float(start_time) - max(0.0, float(context_seconds)))
    upper = float(end_time)
    items: list[tuple[float, Any]] = []
    has_context = False

    for chunk_id in chunk_order:
        chunk = chunk_cache[chunk_id]
        if chunk.end_time < lower or chunk.start_time > upper:
            continue
        if chunk.frames and len(chunk.frame_timestamps) == len(chunk.frames):
            pairs = zip(chunk.frame_timestamps, chunk.frames)
        elif chunk.representative_frame is not None:
            pairs = [(chunk.representative_ts, chunk.representative_frame)]
        else:
            continue
        for ts, frame in pairs:
            ts = float(ts)
            if lower < ts <= upper:
                items.append((ts, frame))
                if ts <= start_time:
                    has_context = True

    items.sort(key=lambda item: item[0])
    return items, has_context


def uniform_indices(n: int, k: int) -> list[int]:
    if n <= 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))
    if k == 1:
        return [n - 1]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def sample_frames_between(
    chunk_cache: dict[int, CachedChunk],
    chunk_order: list[int],
    start_time: float,
    end_time: float,
    max_frames: int,
    context_seconds: float,
) -> tuple[list[Any], list[float], bool]:
    items, has_context = cached_frame_items_between(
        chunk_cache,
        chunk_order,
        start_time=start_time,
        end_time=end_time,
        context_seconds=context_seconds,
    )
    selected = [items[i] for i in uniform_indices(len(items), max(1, int(max_frames)))]
    return [frame for _, frame in selected], [ts for ts, _ in selected], has_context


# ---------------------------------------------------------------------------
# Answer paths
# ---------------------------------------------------------------------------

def answer_graph_memory(
    question: dict[str, Any],
    chunk_cache: dict[int, CachedChunk],
    chunk_order: list[int],
    qa: RecentWindowQAModel,
    evaluator: HubAndSpokeEvaluator,
    recent_frames_only: int,
    extract_every_n_chunks: int,
    max_extraction_chunks: int,
    caption_batch_size: int,
    min_evidence_sim: float | None = None,
) -> tuple[str, dict[str, Any]]:
    window = max(1, int(recent_frames_only))
    hist_chunk_ids = chunk_order[:-window] if len(chunk_order) > window else []
    selected_ids = select_history_chunk_ids(
        hist_chunk_ids,
        extract_every_n_chunks=extract_every_n_chunks,
        max_extraction_chunks=max_extraction_chunks,
    )
    selected_chunks = [chunk_cache[idx] for idx in selected_ids]
    memory = build_memory_from_cached_chunks(
        selected_chunks,
        qa,
        embed_model=evaluator.embed_model,
        embed_device=evaluator.embed_device,
        sim_threshold=evaluator.sim_threshold,
        caption_batch_size=caption_batch_size,
        evaluator=evaluator,
    )
    recent_frames = recent_frames_for_answer(chunk_cache, chunk_order, window)
    if not recent_frames:
        raise ValueError("No recent frames available for graph_memory answer")

    evaluator.last_retrieval = None
    response = evaluator.answer_with_memory_mcq(
        memory,
        recent_frames,
        build_prompt(question),
        num_options=4,
        min_evidence_sim=min_evidence_sim,
    )
    retrieval = evaluator.last_retrieval
    metadata = {
        "hist_chunk_candidates": len(hist_chunk_ids),
        "hist_chunk_selected": len(selected_chunks),
        "num_recent_frames": len(recent_frames),
        "route_min_evidence_sim": min_evidence_sim,
        "retrieval_matched_nodes": (
            retrieval.matched_nodes if retrieval is not None else None
        ),
        "retrieval_signal": retrieval.signal if retrieval is not None else None,
        "retrieval_hit_count": retrieval.hit_count if retrieval is not None else None,
        # Raw cosine scores, so the gate's evidence floor can be re-swept
        # offline from this run instead of one evaluation run per threshold.
        "retrieval_hit_sims": round_sims(retrieval, "hit_sims"),
        "retrieval_candidate_sims": round_sims(retrieval, "candidate_sims"),
        **{f"hub_spoke_{k}": v for k, v in memory.stats().items()},
    }
    return response, metadata


def answer_recent_window(
    question: dict[str, Any],
    chunk_cache: dict[int, CachedChunk],
    chunk_order: list[int],
    qa: RecentWindowQAModel,
    recent_frames_only: int,
) -> tuple[str, dict[str, Any]]:
    recent_frames = recent_frames_for_answer(
        chunk_cache,
        chunk_order,
        max(1, int(recent_frames_only)),
    )
    if not recent_frames:
        raise ValueError("No recent frames available for recent_window answer")
    response = qa.score_mcq_from_frames(recent_frames, build_prompt(question), num_options=4)
    return response, {
        "num_recent_frames": len(recent_frames),
        "readout": "recent_window_abcd_logits",
    }


def answer_baseline_hist_recent(
    question: dict[str, Any],
    chunk_cache: dict[int, CachedChunk],
    chunk_order: list[int],
    qa: RecentWindowQAModel,
    recent_frames_only: int,
    hist_frames: int,
    sampling: str = "hist_plus_recent",
) -> tuple[str, dict[str, Any]]:
    """Bare-backbone matched baselines (no memory).

    hist_plus_recent: the same recent window plus the same ΠB-selected
    historical chunks D-HSM would caption, as raw visual frames.
    uniform: hist_frames + recent_frames_only frames sampled uniformly over
    the whole available prefix (matched total budget, no recency guarantee)."""
    window = max(1, int(recent_frames_only))
    if sampling == "uniform":
        selected_ids = select_history_chunk_ids(
            list(chunk_order),
            extract_every_n_chunks=1,
            max_extraction_chunks=hist_frames + window,
        )
        frames = [
            chunk_cache[idx].representative_frame
            for idx in selected_ids
            if chunk_cache[idx].representative_frame is not None
        ]
        meta = {"num_uniform_frames": len(frames),
                "readout": "baseline_uniform_abcd_logits"}
    else:
        hist_chunk_ids = chunk_order[:-window] if len(chunk_order) > window else []
        selected_ids = select_history_chunk_ids(
            hist_chunk_ids,
            extract_every_n_chunks=1,
            max_extraction_chunks=hist_frames,
        )
        hist_frames_list = [
            chunk_cache[idx].representative_frame
            for idx in selected_ids
            if chunk_cache[idx].representative_frame is not None
        ]
        recent_frames = recent_frames_for_answer(chunk_cache, chunk_order, window)
        frames = hist_frames_list + recent_frames
        meta = {"num_hist_frames": len(hist_frames_list),
                "num_recent_frames": len(recent_frames),
                "readout": "baseline_hist_recent_abcd_logits"}
    if not frames:
        raise ValueError("No frames available for baseline answer")
    response = qa.score_mcq_from_frames(frames, build_prompt(question), num_options=4)
    return response, meta


def answer_interval_count(
    question: dict[str, Any],
    chunk_cache: dict[int, CachedChunk],
    chunk_order: list[int],
    qa: RecentWindowQAModel,
    state: CountState,
    current_cutoff: float,
    interval_frames: int,
    interval_context_seconds: float,
    interval_max_delta: int,
) -> tuple[str, dict[str, Any]]:
    state_key = "count"

    def _write_hub_state(
        response_text: str,
        delta: int,
        label: str,
        previous: float,
        cutoff: float,
        num_frames: int,
        timestamps: list[float] | None = None,
        has_context: bool | None = None,
        elapsed: float | None = None,
    ) -> dict[str, Any]:
        task_state = {
            "state_type": "count",
            "update_rule": "interval_delta",
            "response": response_text,
            "new_count": int(delta),
            "label": label,
            "prev_cutoff": previous,
            "cutoff": cutoff,
            "num_frames": int(num_frames),
            "first_ts": timestamps[0] if timestamps else None,
            "last_ts": timestamps[-1] if timestamps else None,
            "has_context": has_context,
            "label_counts": dict(state.label_counts),
            "score_time": elapsed,
        }
        state.memory.update_task_state(state_key, task_state)
        hub_state = state.memory.get_task_state(state_key) or task_state
        return {
            "answer_method": "hub_and_spoke_memory",
            "answer_policy": "interval_delta_count_state",
            "readout": "hub_and_spoke_interval_state",
            "hub_spoke_task_state_key": state_key,
            "hub_spoke_interval_count_response": hub_state.get("response"),
            "hub_spoke_interval_count_new_count": hub_state.get("new_count"),
            "hub_spoke_interval_count_label": hub_state.get("label"),
            "hub_spoke_interval_count_prev_cutoff": hub_state.get("prev_cutoff"),
            "hub_spoke_interval_count_cutoff": hub_state.get("cutoff"),
            "hub_spoke_interval_count_num_frames": hub_state.get("num_frames"),
            "hub_spoke_interval_count_has_context": hub_state.get("has_context"),
            **{f"hub_spoke_{k}": v for k, v in state.memory.stats().items()},
        }

    previous_cutoff = float(state.previous_cutoff)
    if current_cutoff <= previous_cutoff:
        response = count_to_option_letter(state.count, question.get("options", []))
        hub_meta = _write_hub_state(
            response_text=str(state.count),
            delta=0,
            label="A",
            previous=previous_cutoff,
            cutoff=current_cutoff,
            num_frames=0,
        )
        return response, {
            "interval_count_response": str(state.count),
            "interval_count_new_count": 0,
            "interval_count_label": "A",
            "interval_count_prev_cutoff": previous_cutoff,
            "interval_count_cutoff": current_cutoff,
            "interval_count_num_frames": 0,
            "interval_count_label_counts": dict(state.label_counts),
            **hub_meta,
        }

    frames, timestamps, has_context = sample_frames_between(
        chunk_cache,
        chunk_order,
        start_time=previous_cutoff,
        end_time=current_cutoff,
        max_frames=interval_frames,
        context_seconds=interval_context_seconds,
    )
    if not frames:
        label = "A"
        delta = 0
        elapsed = 0.0
    else:
        prompt = build_delta_prompt(
            question,
            previous_cutoff=previous_cutoff,
            current_cutoff=current_cutoff,
            current_count=state.count,
            max_delta=interval_max_delta,
            has_context=has_context,
        )
        t0 = time.perf_counter()
        label = qa.score_mcq_from_frames(
            frames,
            prompt,
            num_options=interval_max_delta + 2,
        )
        elapsed = time.perf_counter() - t0
        max_valid_letter = chr(65 + interval_max_delta + 1)
        if not label or label < "A" or label > max_valid_letter:
            label = max_valid_letter
        option_idx = ord(label) - 65
        delta = option_idx if option_idx <= interval_max_delta else 0

    state.label_counts[label] = state.label_counts.get(label, 0) + 1
    state.count = max(0, state.count + int(delta))
    state.previous_cutoff = current_cutoff
    response = count_to_option_letter(state.count, question.get("options", []))
    hub_meta = _write_hub_state(
        response_text=str(state.count),
        delta=int(delta),
        label=label,
        previous=previous_cutoff,
        cutoff=current_cutoff,
        num_frames=len(frames),
        timestamps=timestamps,
        has_context=has_context,
        elapsed=elapsed,
    )
    return response, {
        "interval_count_response": str(state.count),
        "interval_count_new_count": int(delta),
        "interval_count_label": label,
        "interval_count_prev_cutoff": previous_cutoff,
        "interval_count_cutoff": current_cutoff,
        "interval_count_num_frames": len(frames),
        "interval_count_first_ts": timestamps[0] if timestamps else None,
        "interval_count_last_ts": timestamps[-1] if timestamps else None,
        "interval_count_has_context": has_context,
        "interval_count_label_counts": dict(state.label_counts),
        "interval_count_score_time": elapsed,
        **hub_meta,
    }


# ---------------------------------------------------------------------------
# Per-rank evaluation loop
# ---------------------------------------------------------------------------

def _base_result_row(
    video_basename: str,
    video_path_raw: str,
    video_categories: dict[str, str],
    question: dict[str, Any],
    route: str,
    time_window: str,
    time_seconds: float | None = None,
) -> dict[str, Any]:
    """Fields shared by every result row: success, error, and the
    missing-video placeholder."""
    row: dict[str, Any] = {
        "video": video_basename,
        "video_path": video_path_raw,
        "video_categories": video_categories.get(video_path_raw, ""),
        "time_window": time_window,
        "task_type": question.get("task_type", ""),
        "required_ability": question.get("required_ability", ""),
        "route": route,
        "time_stamp": question.get("time_stamp", ""),
    }
    if time_seconds is not None:
        row["time_seconds"] = time_seconds
    row["question"] = question.get("question", "")
    row["options"] = question.get("options", [])
    return row


def run_rank(
    local_video_paths: list[str],
    video_questions: dict[str, list[dict[str, Any]]],
    video_categories: dict[str, str],
    video_windows: dict[str, list[str]],
    video_dir: str,
    ckpt_path: str,
    qa: RecentWindowQAModel,
    evaluator: HubAndSpokeEvaluator,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    extract_every_n_chunks: int,
    max_extraction_chunks: int,
    caption_batch_size: int,
    count_interval_frames: int,
    count_interval_context_seconds: float,
    count_interval_max_delta: int,
    rank: int,
    baseline_sampling: str | None = None,
    routing: str = ROUTING_KEYWORD,
    gate_strict_sim: float = DEFAULT_GATE_STRICT_SIM,
) -> list[dict[str, Any]]:
    import torch

    all_results, done_keys = shard_io.resume(ckpt_path, key_fn=_row_key)

    def is_done(video_basename: str, question: dict[str, Any]) -> bool:
        return (
            make_key(video_basename, question, 80) in done_keys
            or make_key(video_basename, question, 50) in done_keys
        )

    total_questions = sum(len(video_questions[vp]) for vp in local_video_paths)
    window = max(1, int(recent_frames_only))

    with open(ckpt_path, "a", encoding="utf-8") as ckpt_file:
        processed = 0
        for vi, video_path_raw in enumerate(local_video_paths, 1):
            questions = video_questions[video_path_raw]
            video_path = resolve_video_path(video_path_raw, video_dir)
            video_basename = os.path.basename(video_path)
            logger.info(
                "[rank%d video %d/%d] %s (%d qs)",
                rank,
                vi,
                len(local_video_paths),
                video_basename,
                len(questions),
            )

            if not os.path.exists(video_path):
                logger.warning("Missing: %s", video_path)
                for question in questions:
                    processed += 1
                    if is_done(video_basename, question):
                        continue
                    route = route_question(question, routing)
                    row = {
                        **_base_result_row(
                            video_basename,
                            video_path_raw,
                            video_categories,
                            question,
                            route,
                            time_window=video_windows.get(video_path_raw, [""])[0],
                        ),
                        "answer_gt": answer_gt_letter(question),
                        "response": None,
                        "correct": False,
                        "error": f"Missing video: {video_path}",
                    }
                    key = make_key(video_basename, question, 80)
                    all_results.append(row)
                    done_keys.add(key)
                    shard_io.append_row(ckpt_file, row, key)
                continue

            chunk_cache: dict[int, CachedChunk] = {}
            chunk_order: list[int] = []
            count_states: dict[str, CountState] = {}
            last_decode_end = 0.0
            decode_backend = "cached_window"

            for question in questions:
                processed += 1
                if is_done(video_basename, question):
                    logger.info(
                        "  [rank%d %d/%d] skip %s %s",
                        rank,
                        processed,
                        total_questions,
                        question.get("time_stamp", ""),
                        question.get("task_type", ""),
                    )
                    continue

                gate_decision: GateDecision | None = None
                if baseline_sampling:
                    route = f"baseline_{baseline_sampling}"
                elif routing == ROUTING_TASK_LABEL:
                    route = route_question_by_task_label(question)
                else:
                    route, gate_decision = route_question_by_keyword(question)
                # History is injected under the stricter floor only when the
                # question's wording did not settle whether history is needed.
                min_evidence_sim = (
                    gate_strict_sim
                    if gate_decision is not None and gate_decision.strict_evidence
                    else None
                )
                ts_sec = float(timestamp_to_seconds(question.get("time_stamp", 0.0)))

                try:
                    t0 = time.perf_counter()

                    target_decode_end = max(0.0, ts_sec) + 1e-4
                    if target_decode_end > last_decode_end + 1e-6:
                        chunks, decode_backend = decode_video_to_chunks_qwen(
                            video_path=video_path,
                            chunk_duration=chunk_duration,
                            fps=fps,
                            video_start=last_decode_end,
                            video_end=target_decode_end,
                        )
                        if not chunks and not chunk_order:
                            raise ValueError("No chunks decoded")
                        for chunk in chunks:
                            upsert_cached_chunk(chunk_cache, chunk_order, chunk)
                        last_decode_end = target_decode_end

                    if baseline_sampling:
                        response, metadata = answer_baseline_hist_recent(
                            question,
                            chunk_cache,
                            chunk_order,
                            qa,
                            recent_frames_only=recent_frames_only,
                            hist_frames=max_extraction_chunks,
                            sampling=baseline_sampling,
                        )
                    elif route == ROUTE_GRAPH:
                        response, metadata = answer_graph_memory(
                            question,
                            chunk_cache,
                            chunk_order,
                            qa,
                            evaluator,
                            recent_frames_only=recent_frames_only,
                            extract_every_n_chunks=extract_every_n_chunks,
                            max_extraction_chunks=max_extraction_chunks,
                            caption_batch_size=caption_batch_size,
                            min_evidence_sim=min_evidence_sim,
                        )
                    elif route == ROUTE_INTERVAL:
                        count_key = normalize_question_key(question.get("question", ""))
                        state = count_states.setdefault(count_key, CountState())
                        response, metadata = answer_interval_count(
                            question,
                            chunk_cache,
                            chunk_order,
                            qa,
                            state,
                            current_cutoff=ts_sec,
                            interval_frames=count_interval_frames,
                            interval_context_seconds=count_interval_context_seconds,
                            interval_max_delta=count_interval_max_delta,
                        )
                    else:
                        response, metadata = answer_recent_window(
                            question,
                            chunk_cache,
                            chunk_order,
                            qa,
                            recent_frames_only=recent_frames_only,
                        )

                    generate_time = time.perf_counter() - t0
                    pred = extract_mcq_letter(response)
                    gt = answer_gt_letter(question)
                    correct = bool(pred is not None and pred == gt)
                    row = {
                        **_base_result_row(
                            video_basename,
                            video_path_raw,
                            video_categories,
                            question,
                            route,
                            time_window=question.get("_time_window", ""),
                            time_seconds=ts_sec,
                        ),
                        "answer_gt": gt,
                        "response": response,
                        "correct": correct,
                        "decode_backend": decode_backend,
                        "generate_time": generate_time,
                        "routing_mode": routing,
                        # Agreement audit: what the old annotation-driven rule
                        # would have chosen for this question.
                        "route_task_label": route_question_by_task_label(question),
                        **(gate_decision.as_metadata() if gate_decision else {}),
                        **metadata,
                    }
                    logger.info(
                        "  [rank%d %d/%d] %s %s/%s -> %s (gt=%s)",
                        rank,
                        processed,
                        total_questions,
                        question.get("time_stamp", ""),
                        question.get("task_type", ""),
                        route,
                        response,
                        gt,
                    )

                except Exception as exc:
                    row = {
                        **_base_result_row(
                            video_basename,
                            video_path_raw,
                            video_categories,
                            question,
                            route,
                            time_window=question.get("_time_window", ""),
                            time_seconds=ts_sec,
                        ),
                        "answer_gt": answer_gt_letter(question),
                        "response": None,
                        "correct": False,
                        "error": str(exc),
                    }
                    logger.exception(
                        "  [rank%d %d/%d] %s %s/%s failed for %s: %s",
                        rank,
                        processed,
                        total_questions,
                        question.get("time_stamp", ""),
                        question.get("task_type", ""),
                        route,
                        video_path,
                        exc,
                    )

                key = make_key(video_basename, question, 80)
                all_results.append(row)
                done_keys.add(key)
                shard_io.append_row(ckpt_file, row, key)

                drop_old_recent_frames(chunk_cache, chunk_order, keep_last=window)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hub-and-Spoke StreamingBench evaluation"
    )
    parser.add_argument("--anno_path", "--anno-path", required=True)
    parser.add_argument("--video_dir", "--video-dir", required=True)
    parser.add_argument("--result_dir", "--output-dir", default=None)
    parser.add_argument(
        "--model_path",
        "--qa-model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
    )
    parser.add_argument("--chunk_duration", "--chunk-duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--recent_frames_only", "--recent-frames-only", type=int, default=4)
    parser.add_argument("--max_qa_tokens", "--max-qa-tokens", type=int, default=256)
    parser.add_argument(
        "--extract_every_n_chunks",
        "--extract-every-n-chunks",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_extraction_chunks",
        "--max-extraction-chunks",
        type=int,
        default=30,
        help="Cap on historical chunks for graph_memory; 0=unlimited.",
    )
    parser.add_argument("--sim_threshold", "--sim-threshold", type=float, default=0.25)
    parser.add_argument(
        "--routing",
        choices=ROUTING_CHOICES,
        default=ROUTING_KEYWORD,
        help=(
            "How the memory/recent-window decision is made. 'keyword' (default) "
            "gates on question text only, as described in the paper. 'task_label' "
            "reads the benchmark's required_ability/task_type annotations and is "
            "kept only to reproduce the pre-gate numbers."
        ),
    )
    parser.add_argument(
        "--gate_strict_sim",
        type=float,
        default=DEFAULT_GATE_STRICT_SIM,
        help=(
            "Similarity floor applied when the keyword gate cannot tell whether "
            "history is needed (the 'ambiguous' bucket). History is injected only "
            "if some memory entry clears it. Ignored when --routing task_label."
        ),
    )
    parser.add_argument(
        "--top_k",
        "--top-k",
        type=parse_top_k_arg,
        default="dynamic",
        help=(
            "Fixed retrieval budget (int), or dynamic/auto/adaptive for "
            f"elbow-cutoff mode capped at {DYNAMIC_TOP_K_MAX}. Defaults to "
            "dynamic: a fixed budget is the Table 6 / Fig. 1 'static HSM "
            "retrieval' baseline, not D-HSM."
        ),
    )
    parser.add_argument(
        "--embed_model",
        "--embed-model",
        default=DEFAULT_EMBED_MODEL,
        help="Sentence-transformers model for node/query embeddings.",
    )
    parser.add_argument(
        "--caption_batch_size",
        "--caption-batch-size",
        type=int,
        default=0,
        help="Batch size for graph_memory history captions; 0/1 disables batching.",
    )
    parser.add_argument("--count_interval_frames", type=int, default=16)
    parser.add_argument("--count_interval_context_seconds", type=float, default=1.0)
    parser.add_argument("--count_interval_max_delta", type=int, default=6)
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        help="Attention backend for the VLM (e.g. flash_attention_2, sdpa).",
    )
    parser.add_argument(
        "--no_expansion",
        action="store_true",
        help="Ablation: disable hub-and-spoke expansion during retrieval.",
    )
    parser.add_argument(
        "--memory_mode",
        choices=list(MEMORY_MODE_CHOICES),
        default=None,
        help=(
            "hub_spoke: dhsm/hub_and_spoke.py, the non-provenance variant. "
            "flat_caption: ablation storing each chunk's whole caption as one "
            "retrieval unit (same captions and retrieval hyperparameters, no "
            "hub-and-spoke organization). "
            "incremental: dhsm/hub_and_spoke_incremental.py, the provenance-"
            "tracking variant implementing Algorithm 2. As on OVO, the "
            "streaming path here only calls update(), so this does not "
            "exercise its remove_chunks() removal path. "
            "Defaults to incremental for both routing modes, so an A/B over "
            "--routing isolates the gate; pass it explicitly to override."
        ),
    )
    parser.add_argument(
        "--no_cooccurrence",
        action="store_true",
        help="Ablation: drop co-occurrence edges from expansion.",
    )
    parser.add_argument(
        "--no_next_action",
        action="store_true",
        help="Ablation: drop next-action chains from expansion.",
    )
    parser.add_argument(
        "--no_spoke_merging",
        action="store_true",
        help="Ablation: insert repeated spoke facts as new nodes instead of merging.",
    )
    parser.add_argument(
        "--baseline_sampling",
        choices=["hist_plus_recent", "uniform"],
        default=None,
        help=(
            "Bare-backbone baseline: answer EVERY question from raw frames "
            "(max_extraction_chunks historical representative frames chosen "
            "by the same resampling D-HSM uses, plus the recent window); no "
            "memory, no routing."
        ),
    )
    parser.add_argument("--max_videos", type=int, default=None)
    parser.add_argument("--max_questions_per_video", type=int, default=None)
    args = parser.parse_args()
    # --routing keyword implies the incremental (Algorithm 2) memory unless
    # --memory_mode was given explicitly.
    args.memory_mode = resolve_memory_mode(args.routing, args.memory_mode)

    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=24))]
    )

    with open(args.anno_path) as f:
        all_data = json.load(f)

    video_questions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    video_categories: dict[str, str] = {}
    video_windows: dict[str, list[str]] = defaultdict(list)
    for entry in all_data:
        vp = entry["video_path"]
        video_categories[vp] = entry.get("video_categories", "")
        time_window = entry.get("time", "")
        video_windows[vp].append(time_window)
        for question in entry.get("questions", []):
            item = dict(question)
            item["_time_window"] = time_window
            video_questions[vp].append(item)

    for vp in video_questions:
        video_questions[vp].sort(
            key=lambda question: timestamp_to_seconds(question.get("time_stamp", 0.0))
        )
        if args.max_questions_per_video is not None:
            video_questions[vp] = video_questions[vp][: args.max_questions_per_video]

    all_video_paths = list(video_questions.keys())
    if args.max_videos is not None:
        all_video_paths = all_video_paths[: args.max_videos]

    total_questions = sum(len(video_questions[vp]) for vp in all_video_paths)
    route_counts: dict[str, int] = defaultdict(int)
    task_route_counts: dict[tuple[str, str], int] = defaultdict(int)
    for vp in all_video_paths:
        for question in video_questions[vp]:
            route = route_question(question, args.routing)
            route_counts[route] += 1
            task_route_counts[(str(question.get("task_type", "")), route)] += 1

    if args.result_dir:
        output_dir = args.result_dir
    else:
        model_tag = Path(str(args.model_path).rstrip("/")).name.lower().replace("-instruct", "")
        run_tag = (
            f"hub_and_spoke_{model_tag}"
            f"_recent{args.recent_frames_only}"
            f"_extract{args.extract_every_n_chunks}"
            f"_streamingbench"
        )
        output_dir = os.path.join("results", "streamingbench", run_tag)

    accelerator.print(f"\n{'=' * 60}")
    accelerator.print("Hub-and-Spoke StreamingBench Evaluation")
    accelerator.print(f"{'=' * 60}")
    accelerator.print(
        f"Videos: {len(all_video_paths)}, Questions: {total_questions}, "
        f"Processes: {accelerator.num_processes}"
    )
    accelerator.print(
        "Routes: "
        + ", ".join(f"{k}={route_counts[k]}" for k in sorted(route_counts))
    )
    accelerator.print(
        f"recent_frames={args.recent_frames_only}  "
        f"embed_model={args.embed_model}  sim_threshold={args.sim_threshold}  "
        f"top_k={format_top_k(args.top_k)}  "
        f"routing={args.routing}  "
        f"memory_mode={args.memory_mode}  "
        f"gate_strict_sim={args.gate_strict_sim}  "
        f"caption_batch_size={args.caption_batch_size}  "
        f"count_interval={args.count_interval_frames}f/"
        f"max_delta={args.count_interval_max_delta}"
    )
    accelerator.print(f"Output: {output_dir}")
    accelerator.print(f"{'=' * 60}\n")

    qa = RecentWindowQAModel(
        model_name=args.model_path,
        device=accelerator.device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
    )
    evaluator_cls = evaluator_class_for(args.memory_mode)
    evaluator = evaluator_cls(
        qa_model=qa,
        recent_frames=args.recent_frames_only,
        extract_every_n_chunks=args.extract_every_n_chunks,
        max_extraction_chunks=args.max_extraction_chunks,
        embed_model=args.embed_model,
        sim_threshold=args.sim_threshold,
        top_k=args.top_k,
        caption_batch_size=args.caption_batch_size,
        expand_retrieval=not args.no_expansion,
        expand_co_occurrence=not args.no_cooccurrence,
        expand_next_action=not args.no_next_action,
        merge_spokes=not args.no_spoke_merging,
    )

    with accelerator.split_between_processes(all_video_paths) as local_paths:
        local_video_paths = list(local_paths)

    ckpt_path = shard_io.checkpoint_path(
        output_dir,
        accelerator.process_index,
        accelerator.num_processes,
    )
    done_marker_path = shard_io.done_path(
        output_dir,
        accelerator.process_index,
        accelerator.num_processes,
    )
    if os.path.exists(done_marker_path):
        os.remove(done_marker_path)

    logger.info(
        "[rank%d] assigned %d videos",
        accelerator.process_index,
        len(local_video_paths),
    )

    run_rank(
        local_video_paths=local_video_paths,
        video_questions=video_questions,
        video_categories=video_categories,
        video_windows=video_windows,
        video_dir=args.video_dir,
        ckpt_path=ckpt_path,
        qa=qa,
        evaluator=evaluator,
        chunk_duration=args.chunk_duration,
        fps=args.fps,
        recent_frames_only=args.recent_frames_only,
        extract_every_n_chunks=args.extract_every_n_chunks,
        max_extraction_chunks=args.max_extraction_chunks,
        caption_batch_size=args.caption_batch_size,
        count_interval_frames=args.count_interval_frames,
        count_interval_context_seconds=args.count_interval_context_seconds,
        count_interval_max_delta=args.count_interval_max_delta,
        rank=accelerator.process_index,
        baseline_sampling=args.baseline_sampling,
        routing=args.routing,
        gate_strict_sim=args.gate_strict_sim,
    )

    shard_io.write_done_marker(done_marker_path)

    if accelerator.is_main_process:
        shard_io.wait_for_done_markers(output_dir, accelerator.num_processes)
        all_results = shard_io.merge_shards(output_dir, accelerator.num_processes, key_fn=_row_key)

        model_label = f"HubAndSpoke-{Path(args.model_path.rstrip('/')).name}"
        print_summary(all_results, label=model_label)
        summary = compute_summary(all_results)

        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shard_io.save_json(
            os.path.join(output_dir, f"hub_and_spoke_streamingbench_{ts}.json"),
            {
                "config": vars(args),
                "route_counts": dict(route_counts),
                "task_route_counts": {
                    f"{task}/{route}": count
                    for (task, route), count in sorted(task_route_counts.items())
                },
                "summary": summary,
                "results": all_results,
            },
        )
        shard_io.save_json(os.path.join(output_dir, "scores_report.json"), summary)
        print(f"\nResults saved to: {output_dir}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
'''
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
  experiments/evaluate_streamingbench.py \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --anno_path /data/xinru/dataset/data/streamingbench/questions_real.json \
  --video_dir /data/xinru/dataset/data/streamingbench/videos \
  --result_dir results/streamingbench/qwen2.5_4f_dynamic \
  --recent_frames_only 4 \
  --chunk_duration 1.0 \
  --fps 1.0 \
  --max_qa_tokens 256 \
  --extract_every_n_chunks 1 \
  --max_extraction_chunks 20 \
  --sim_threshold 0.25 \
  --top_k dynamic \
  --caption_batch_size 4 \
  2>&1 | tee results/sb_qwen2.5_4f_dynamic.log
'''
