"""
Hub-and-Spoke StreamingBench Evaluation
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
import hashlib
import importlib.metadata
import json
import logging
import math
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
    ENTITY_LINK_THRESHOLD,
    HubAndSpokeEvaluator,
    HubAndSpokeMemory,
    _is_count_question,
    round_sims,
)
from dhsm.retrieval_gate import (
    DEFAULT_GATE_STRICT_SIM,
    ROUTING_CHOICES,
    ROUTING_KEYWORD,
    GateDecision,
    gate_question,
    strip_prompt_scaffolding,
)
from dhsm import shard_io
from dhsm.evaluation_protocol import ensure_result_protocol
from dhsm.video_qa import decode_video_to_chunks_qwen
from dhsm.video_qa_qwen3 import RecentWindowQAModel
from dhsm.memory_selection import (
    MEMORY_MODE_CHOICES,
    EntityResolvedMemory,
    evaluator_class_for,
    resolve_memory_mode,
)


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


def format_top_k(top_k: int | str, dynamic_top_k_max: int = DYNAMIC_TOP_K_MAX) -> str:
    if isinstance(top_k, str):
        return f"{top_k}(max={dynamic_top_k_max})"
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
    occurrence = question.get("annotation_occurrence_id")
    if occurrence is not None:
        if not isinstance(occurrence, str) or re.fullmatch(r"[0-9]+:[0-9]+", occurrence) is None:
            raise ValueError("Invalid annotation occurrence identity")
        return f"annotation:{occurrence}"
    # Retained for older direct API callers; main assigns an occurrence ID to
    # every question before sorting. Text prefixes are not dataset identity.
    return (
        f"{video_basename}_{question.get('time_stamp', '')}_"
        f"{question.get('task_type', '')}_{question.get('question', '')[:question_limit]}"
    )


def _row_key(row: dict[str, Any]) -> str:
    """Reconstruct a checkpoint key from an already-stripped result row."""
    return make_key(row.get("video", ""), row, 80)


def annotation_question(
    question: dict[str, Any], entry_index: int, question_index: int, time_window: str,
) -> dict[str, Any]:
    """Preserve the raw question and add position-only identity and audit metadata."""
    return {**question, "annotation_occurrence_id": f"{entry_index}:{question_index}",
            "_time_window": time_window}


def validate_result_coverage(results: list[dict[str, Any]], expected_keys: set[str]) -> None:
    """Never publish an aggregate that silently omits an annotation occurrence."""
    keys = [_row_key(row) for row in results]
    if len(keys) != len(set(keys)) or len(keys) != len(expected_keys) or set(keys) != expected_keys:
        raise ValueError(
            "StreamingBench result coverage mismatch: expected every selected annotation "
            "occurrence exactly once before scoring."
        )


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


class SourceMediaUnavailableError(ValueError):
    """The original source has no verified media at this query timestamp."""


def media_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_media_manifest(
    manifest_path: str | None, known_video_paths: set[str],
) -> dict[str, Any] | None:
    # Validate explicit, data-quality-only input substitutions before models load.
 
    if manifest_path is None:
        return None

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate media manifest key: {key}")
            result[key] = value
        return result

    path = Path(manifest_path).resolve()
    raw = path.read_bytes()
    manifest = json.loads(raw, object_pairs_hook=unique_object)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
            or not isinstance(manifest.get("entries"), dict) or not manifest["entries"]):
        raise ValueError("Media manifest requires schema_version=1 and nonempty entries")
    entries = manifest["entries"]
    required = {
        "original_path", "original_sha256", "input_path", "input_sha256",
        "verified_until", "unavailable_after", "provenance",
    }
    for video_path, entry in entries.items():
        if video_path not in known_video_paths:
            raise ValueError(f"Media manifest video is absent from annotations: {video_path}")
        if not isinstance(entry, dict) or set(entry) != required:
            raise ValueError(f"Media manifest entry has unexpected/missing fields: {video_path}")
        if not isinstance(entry["provenance"], dict) or not entry["provenance"]:
            raise ValueError("Media manifest requires source and validation provenance")
        proofs = entry["provenance"].get("proofs")
        if not isinstance(proofs, list) or not proofs:
            raise ValueError("Media manifest requires SHA-bound validation proof files")
        for proof in proofs:
            if (not isinstance(proof, dict) or set(proof) != {"path", "sha256"}
                    or not isinstance(proof["path"], str)
                    or not Path(proof["path"]).is_absolute()
                    or not Path(proof["path"]).is_file()
                    or not isinstance(proof["sha256"], str)
                    or re.fullmatch(r"[0-9a-f]{64}", proof["sha256"]) is None):
                raise ValueError("Media manifest has invalid validation proof binding")
            if media_file_sha256(proof["path"]) != proof["sha256"]:
                raise ValueError("Media manifest validation proof SHA256 mismatch")
        for field in ("verified_until", "unavailable_after"):
            value = entry[field]
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"Media manifest {field} must be finite and nonnegative")
        if entry["verified_until"] != entry["unavailable_after"]:
            raise ValueError("Media manifest verified_until must equal unavailable_after")
        for prefix in ("original", "input"):
            value, expected = entry[f"{prefix}_path"], entry[f"{prefix}_sha256"]
            if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).is_file():
                raise ValueError(f"Media manifest {prefix}_path must be an existing absolute file")
            if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
                raise ValueError(f"Media manifest invalid {prefix} SHA256")
            if media_file_sha256(value) != expected:
                raise ValueError(f"Media manifest {prefix} SHA256 mismatch: {video_path}")
        if os.path.samefile(entry["original_path"], entry["input_path"]):
            raise ValueError("Media manifest must preserve the original file separately")
    return {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "schema_version": 1, "entries": entries}


def source_media_for_video(
    video_path_raw: str, resolved_path: str, media_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    # Record the actual input hash once per video, retaining original identity.
    entry = (media_manifest or {}).get("entries", {}).get(video_path_raw)
    if entry is not None:
        actual = media_file_sha256(entry["input_path"])
        if actual != entry["input_sha256"]:
            raise ValueError(f"Verified prefix changed after media preflight: {video_path_raw}")
        return {key: entry[key] for key in (
            "original_path", "original_sha256", "input_path", "input_sha256",
            "verified_until", "unavailable_after",
        )} | {"manifest_sha256": media_manifest["sha256"], "status": "verified_prefix"}
    actual_path = str(Path(resolved_path).resolve())
    actual_hash = media_file_sha256(actual_path) if Path(actual_path).is_file() else None
    return {"original_path": actual_path, "original_sha256": actual_hash,
            "input_path": actual_path, "input_sha256": actual_hash,
            "verified_until": None, "unavailable_after": None,
            "manifest_sha256": (media_manifest or {}).get("sha256"),
            "status": "original" if actual_hash is not None else "missing"}


def route_question_by_keyword(question: dict[str, Any]) -> tuple[str, GateDecision]:
    # Paper-faithful routing: decide from the question text alone.
    
    question_text = str(question.get("question", ""))
    decision = gate_question(question_text, question.get("options") or [])
    if not decision.needs_memory:
        return ROUTE_RECENT, decision
    if _is_count_question(strip_prompt_scaffolding(question_text)):
        return ROUTE_INTERVAL, decision
    return ROUTE_GRAPH, decision


def route_question(question: dict[str, Any], routing: str = ROUTING_KEYWORD) -> str:
    # Route from question/option text; reject annotation-based routing.
    if routing != ROUTING_KEYWORD:
        raise ValueError("Only question-based keyword routing is supported")
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
    if (len(frame_timestamps) != len(chunk.frames)
            or any(type(t) not in (int, float) or not math.isfinite(t) or t < 0 for t in frame_timestamps)
            or frame_timestamps != sorted(frame_timestamps)):
        raise ValueError("Cached chunks require matching finite, nonnegative, chronological actual PTS")
    if (not math.isfinite(chunk.start_time) or not math.isfinite(chunk.end_time)
            or not 0 <= chunk.start_time <= frame_timestamps[0]
            or frame_timestamps[-1] > chunk.end_time):
        raise ValueError("Actual frame PTS fall outside their chunk bounds")
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

    if (len(existing.frames) != len(existing.frame_timestamps)
            or (existing.frame_timestamps and frame_timestamps[0] < existing.frame_timestamps[-1])):
        raise ValueError("Appending a chunk would lose actual PTS alignment or chronological order")
    existing.start_time = min(existing.start_time, float(chunk.start_time))
    existing.end_time = max(existing.end_time, float(chunk.end_time))
    existing.frames.extend(chunk.frames)
    existing.frame_timestamps.extend(frame_timestamps)
    if existing.frames:
        mid = len(existing.frames) // 2
        existing.representative_frame = existing.frames[mid]
        existing.representative_ts = float(existing.frame_timestamps[mid])


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
        # Respect the selected memory, including explicit legacy ablations.
        memory = evaluator._make_memory()
    else:
        memory = EntityResolvedMemory(
            embed_model=embed_model,
            embed_device=embed_device,
            sim_threshold=sim_threshold,
        )
    for chunk in selected_chunks:
        if chunk.caption:
            if isinstance(memory, EntityResolvedMemory):
                # Caption IDs are local to this actual cached chunk. Keep its
                # provenance independent of rounded representative timestamps.
                memory.update(chunk.caption, chunk.representative_ts, chunk_id=chunk.chunk_index)
            else:
                memory.update(chunk.caption, chunk.representative_ts)
    return memory


def recent_frame_items_for_answer(
    chunk_cache: dict[int, CachedChunk],
    chunk_order: list[int],
    window: int,
    current_cutoff: float | None = None,
) -> list[tuple[float, Any]]:
    """Last N distinct actual frames inside the unchanged recent-chunk region."""
    if current_cutoff is not None and (not math.isfinite(current_cutoff) or current_cutoff < 0):
        raise ValueError("A finite nonnegative question cutoff is required")
    items: list[tuple[float, Any]] = []
    for chunk_id in chunk_order[-max(1, window):]:
        chunk = chunk_cache[chunk_id]
        if chunk.frames:
            if len(chunk.frames) != len(chunk.frame_timestamps):
                raise ValueError("Recent frames lack matching actual PTS")
            items.extend(zip(chunk.frame_timestamps, chunk.frames))
        elif chunk.representative_frame is not None:
            items.append((chunk.representative_ts, chunk.representative_frame))
    times = [t for t, _ in items]
    if (any(type(t) not in (int, float) or not math.isfinite(t) or t < 0
            or (current_cutoff is not None and t > current_cutoff) for t in times)
            or times != sorted(times)):
        raise ValueError("Recent actual PTS are invalid, nonchronological or after the question")
    distinct = {}
    for timestamp, frame in items:
        distinct.setdefault(timestamp, frame)
    return list(distinct.items())[-max(1, int(window)):]


def recent_frames_for_answer(
    chunk_cache: dict[int, CachedChunk], chunk_order: list[int], window: int,
    current_cutoff: float | None = None,
) -> list[Any]:
    return [frame for _, frame in recent_frame_items_for_answer(
        chunk_cache, chunk_order, window, current_cutoff)]


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
    current_cutoff: float | None = None,
) -> tuple[str, dict[str, Any]]:
    window = max(1, int(recent_frames_only))
    recent = recent_frame_items_for_answer(chunk_cache, chunk_order, window, current_cutoff)
    recent_times, recent_frames = [t for t, _ in recent], [f for _, f in recent]
    if not recent_frames:
        raise ValueError("No recent frames available for graph_memory answer")
    hist_chunk_ids = chunk_order[:-window] if len(chunk_order) > window else []
    selected_ids = select_history_chunk_ids(
        hist_chunk_ids,
        extract_every_n_chunks=extract_every_n_chunks,
        max_extraction_chunks=max_extraction_chunks,
    )
    selected_chunks = [chunk_cache[idx] for idx in selected_ids]
    if any(not math.isfinite(c.representative_ts) or c.representative_ts >= recent_times[0]
           for c in selected_chunks):
        raise ValueError("Historical caption frames must precede the recent visual window")
    memory = build_memory_from_cached_chunks(
        selected_chunks,
        qa,
        embed_model=evaluator.embed_model,
        embed_device=evaluator.embed_device,
        sim_threshold=evaluator.sim_threshold,
        caption_batch_size=caption_batch_size,
        evaluator=evaluator,
    )
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
        "recent_frame_timestamps": recent_times,
        "query_cutoff": current_cutoff,
        "history_chunk_ids": selected_ids,
        "history_frame_timestamps": [c.representative_ts for c in selected_chunks],
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
    current_cutoff: float | None = None,
) -> tuple[str, dict[str, Any]]:
    recent = recent_frame_items_for_answer(
        chunk_cache,
        chunk_order,
        max(1, int(recent_frames_only)),
        current_cutoff,
    )
    recent_times, recent_frames = [t for t, _ in recent], [f for _, f in recent]
    if not recent_frames:
        raise ValueError("No recent frames available for recent_window answer")
    response = qa.score_mcq_from_frames(recent_frames, build_prompt(question), num_options=4)
    return response, {
        "num_recent_frames": len(recent_frames),
        "recent_frame_timestamps": recent_times,
        "query_cutoff": current_cutoff,
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
    # Bare-backbone matched baselines (no memory).
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
    source_media: dict[str, Any] | None = None,
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
    if "annotation_occurrence_id" in question:
        row["annotation_occurrence_id"] = question["annotation_occurrence_id"]
    row["question"] = question.get("question", "")
    row["options"] = question.get("options", [])
    if source_media is not None:
        row["source_media"] = dict(source_media)
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
    media_manifest: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if routing != ROUTING_KEYWORD:
        raise ValueError("Only question-based keyword routing is supported")
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
            source_media = source_media_for_video(video_path_raw, video_path, media_manifest)
            video_path = source_media["input_path"]
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
                            time_window=question.get("_time_window", video_windows.get(video_path_raw, [""])[0]),
                            source_media=source_media | {"input_used_for_question": False},
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
            last_decode_end: float | None = None
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
                row_media = source_media | {"input_used_for_question": False}

                try:
                    t0 = time.perf_counter()

                    if not math.isfinite(ts_sec) or ts_sec < 0:
                        raise ValueError("A finite nonnegative question timestamp is required")
                    limit = source_media["verified_until"]
                    if limit is not None and ts_sec > limit:
                        row_media["status"] = "unavailable"
                        raise SourceMediaUnavailableError(
                            f"Source media unavailable at {ts_sec:g}s; verified only through {limit:g}s"
                        )
                    row_media["input_used_for_question"] = True
                    target_decode_end = ts_sec
                    if last_decode_end is None or target_decode_end > last_decode_end:
                        decode_start = 0.0 if last_decode_end is None else math.nextafter(last_decode_end, math.inf)
                        chunks, decode_backend = decode_video_to_chunks_qwen(
                            video_path=video_path,
                            chunk_duration=chunk_duration,
                            fps=fps,
                            video_start=decode_start,
                            video_end=target_decode_end,
                        )
                        if not chunks and not chunk_order:
                            raise ValueError("No chunks decoded")
                        for chunk in chunks:
                            if any(t < decode_start or t > ts_sec for t in chunk.frame_timestamps):
                                raise ValueError("Decoded actual PTS escaped the causal query window")
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
                            current_cutoff=ts_sec,
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
                            current_cutoff=ts_sec,
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
                            source_media=row_media,
                        ),
                        "answer_gt": gt,
                        "response": response,
                        "correct": correct,
                        "decode_backend": decode_backend,
                        "generate_time": generate_time,
                        "routing_mode": routing,
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
                            source_media=row_media,
                        ),
                        "answer_gt": answer_gt_letter(question),
                        "response": None,
                        "correct": False,
                        "error": str(exc),
                        **({"error_code": "source_media_unavailable"}
                           if isinstance(exc, SourceMediaUnavailableError) else {}),
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

def build_result_protocol(
    args: argparse.Namespace, annotation_bytes: bytes, num_processes: int,
) -> dict[str, Any]:
    # Keep old ID handling, data and inference settings out of a new run.
    media_manifest = load_media_manifest(
        getattr(args, "media_manifest", None),
        {entry["video_path"] for entry in json.loads(annotation_bytes)},
    )
    runtime_versions = {}
    for package in (
        "torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils",
        "numpy", "pillow", "av", "decord", "torchcodec", "sentence-transformers",
    ):
        try:
            runtime_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            runtime_versions[package] = None
    source_files = sorted((PROJECT_ROOT / "dhsm").glob("*.py")) + [Path(__file__).resolve()]
    return {
        "benchmark": "streamingbench",
        "config": dict(vars(args)),
        "annotation_sha256": hashlib.sha256(annotation_bytes).hexdigest(),
        "media_manifest": media_manifest,
        "num_processes": num_processes,
        "entity_link_threshold": ENTITY_LINK_THRESHOLD,
        "embed_device": "cpu",
        "processor_overrides": {
            key: int(os.environ[key]) if os.environ.get(key) else None
            for key in ("MIN_PIXELS", "MAX_PIXELS")
        },
        "video_environment": {
            key: os.environ.get(key) for key in (
                "VIDEO_MAX_PIXELS", "FORCE_QWENVL_VIDEO_READER", "TORCHCODEC_NUM_THREADS",
            )
        },
        "runtime_versions": runtime_versions,
        "code_sha256": {
            str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hub-and-Spoke StreamingBench evaluation"
    )
    parser.add_argument("--anno_path", "--anno-path", required=True)
    parser.add_argument("--video_dir", "--video-dir", required=True)
    parser.add_argument(
        "--media_manifest", "--media-manifest", default=None,
        help=("Optional SHA-bound data-quality manifest for verified video prefixes. "
              "Questions beyond a prefix's verified_until become explicit media errors "
              "and remain in the scoring denominator."),
    )
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
        default=20,
        help="Cap on historical chunks for graph_memory (default: 20); 0=unlimited.",
    )
    parser.add_argument("--sim_threshold", "--sim-threshold", type=float, default=0.25)
    parser.add_argument(
        "--routing",
        choices=ROUTING_CHOICES,
        default=ROUTING_KEYWORD,
        help=(
            "Choose memory or recent frames from question/option text only."
        ),
    )
    parser.add_argument(
        "--gate_strict_sim",
        type=float,
        default=DEFAULT_GATE_STRICT_SIM,
        help=(
            "Similarity floor applied when the keyword gate cannot tell whether "
            "history is needed (the 'ambiguous' bucket). History is injected only "
            "if some memory entry clears it."
        ),
    )
    parser.add_argument(
        "--top_k",
        "--top-k",
        type=parse_top_k_arg,
        default="dynamic",
        help=(
            "Fixed retrieval budget (int), or dynamic/auto/adaptive for "
            "elbow-cutoff mode capped by --dynamic_top_k_max. Defaults to "
            "dynamic: a fixed budget is the Table 6 / Fig. 1 'static HSM "
            "retrieval' baseline, not D-HSM."
        ),
    )
    parser.add_argument(
        "--dynamic_top_k_max", type=int, default=12,
        help="Maximum candidate pool for dynamic retrieval (default: 12). "
             "The original elbow rule can select fewer seeds; integer --top_k is unchanged.",
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
        default=4,
        help="Batch size for graph_memory history captions (default: 4); 0/1 disables batching.",
    )
    parser.add_argument("--count_interval_frames", type=int, default=16)
    parser.add_argument("--count_interval_context_seconds", type=float, default=1.0)
    parser.add_argument("--count_interval_max_delta", type=int, default=6)
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        help="Attention backend for the VLM (default: flash_attention_2; pass sdpa to skip flash-attn).",
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
            "entity_resolved (default): incremental memory with chunk-local "
            "caption IDs linked across chunks by entity descriptions. "
            "hub_spoke: dhsm/hub_and_spoke.py, the non-provenance variant. "
            "flat_caption: ablation storing each chunk's whole caption as one "
            "retrieval unit (same captions and retrieval hyperparameters, no "
            "hub-and-spoke organization). "
            "incremental: dhsm/hub_and_spoke_incremental.py, the provenance-"
            "tracking variant implementing Algorithm 2. As on OVO, the "
            "streaming path here only calls update(), so this does not "
            "exercise its remove_chunks() removal path. "
            "Defaults to entity_resolved with chunk-local ID repair; "
            "select an alternative explicitly for an ablation."
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
    if args.dynamic_top_k_max < 1:
        parser.error("--dynamic_top_k_max must be positive.")
    # Use the repaired shared memory unless an ablation is requested.
    args.memory_mode = resolve_memory_mode(args.routing, args.memory_mode)

    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=24))]
    )

    annotation_bytes = Path(args.anno_path).read_bytes()
    all_data = json.loads(annotation_bytes)

    video_questions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    video_categories: dict[str, str] = {}
    video_windows: dict[str, list[str]] = defaultdict(list)
    for entry_index, entry in enumerate(all_data):
        vp = entry["video_path"]
        video_categories[vp] = entry.get("video_categories", "")
        time_window = entry.get("time", "")
        video_windows[vp].append(time_window)
        for question_index, question in enumerate(entry.get("questions", [])):
            item = annotation_question(question, entry_index, question_index, time_window)
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
    expected_keys = {
        make_key(os.path.basename(vp), question)
        for vp in all_video_paths for question in video_questions[vp]
    }
    if len(expected_keys) != total_questions:
        raise ValueError("Selected StreamingBench annotation occurrence identities are not unique")
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
            f"_{args.memory_mode}"
            f"_recent{args.recent_frames_only}"
            f"_extract{args.extract_every_n_chunks}"
            f"_streamingbench"
        )
        output_dir = os.path.join("results", "streamingbench", run_tag)

    result_protocol = build_result_protocol(args, annotation_bytes, accelerator.num_processes)
    ensure_result_protocol(output_dir, result_protocol)
    
    if accelerator.is_main_process:
        shard_io.clear_done_markers(output_dir, accelerator.num_processes)
    accelerator.wait_for_everyone()

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
        f"top_k={format_top_k(args.top_k, args.dynamic_top_k_max)}  "
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
        dynamic_top_k_max=args.dynamic_top_k_max,
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
        media_manifest=result_protocol["media_manifest"],
    )

    shard_io.write_done_marker(done_marker_path)

    if accelerator.is_main_process:
        shard_io.wait_for_done_markers(output_dir, accelerator.num_processes)
        all_results = shard_io.merge_shards(output_dir, accelerator.num_processes, key_fn=_row_key)
        validate_result_coverage(all_results, expected_keys)

        model_label = f"HubAndSpoke-{Path(args.model_path.rstrip('/')).name}"
        print_summary(all_results, label=model_label)
        summary = compute_summary(all_results)

        os.makedirs(output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        shard_io.save_json(
            os.path.join(output_dir, f"hub_and_spoke_streamingbench_{ts}.json"),
            {
                "config": vars(args),
                "media_manifest": result_protocol["media_manifest"],
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
