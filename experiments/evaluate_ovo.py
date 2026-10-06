"""Hub-and-Spoke OVO-Bench evaluation: the full backward/real-time/forward pipeline.

MCQ retrieval uses only question/option text. Task annotations select benchmark
formats and reporting groups; correct answers are used only for scoring.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "86400")

from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dhsm import shard_io
from dhsm.evaluation_protocol import ensure_result_protocol
from dhsm.benchmark_defaults import ovo_defaults
from dhsm.hub_and_spoke import (
    DEFAULT_EMBED_MODEL,
    DYNAMIC_TOP_K_MAX,
    HubAndSpokeEvaluator,
    HubAndSpokeMemory,
    round_sims,
)
from dhsm.retrieval_gate import ROUTING_CHOICES, ROUTING_KEYWORD
from dhsm.memory_selection import (
    MEMORY_MODE_CHOICES, evaluator_class_for, resolve_memory_mode,
)
from dhsm.video_qa import decode_video_to_chunks_qwen
from dhsm.video_qa_qwen3 import RecentWindowQAModel

from experiments.ovo_bench import (
    BACKWARD_TASKS,
    FORWARD_TASKS,
    MCQ_PROMPT_UNIFORM,
    MCQ_PROMPT_POLICIES,
    REAL_TIME_TASKS,
    build_prompt,
    build_mcq_prompt,
    print_report,
)
from experiments.ovo_protocol import (
    HISTORY_DHSM,
    HISTORY_MODES,
    HISTORY_RECENT_ONLY,
    DEFAULT_MEMORY_FLOOR,
    DEFAULT_OVO_GATE_STRICT_SIM,
    UNIFORM_ABSTENTION_INSTRUCTION,
    answer_route_for as _answer_route_for,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def row_key(item: dict[str, Any]) -> str:
    return f"{item.get('task', '')}:{item.get('id')}"


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


def mcq_query_cutoff(anno: dict[str, Any]) -> float:
    cutoff = anno.get("realtime")
    if isinstance(cutoff, bool) or not isinstance(cutoff, (int, float)) or not math.isfinite(cutoff) or cutoff < 0:
        raise ValueError("MCQ annotations require a finite nonnegative realtime query cutoff.")
    return float(cutoff)


def _build_and_answer(
    question: str,
    options: list[str],
    query_cutoff: float,
    video_path: str,
    evaluator: HubAndSpokeEvaluator,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    use_logits: bool = True,
    routing: str = ROUTING_KEYWORD,
    memory_floor: float = DEFAULT_MEMORY_FLOOR,
    gate_strict_sim: float = DEFAULT_OVO_GATE_STRICT_SIM,
    mcq_prompt_policy: str = MCQ_PROMPT_UNIFORM,
    history_mode: str = HISTORY_DHSM,
) -> tuple[str | None, dict[str, Any]]:
    """Infer from observable inputs only; annotations stay in the caller."""
    if not os.path.exists(video_path):
        return None, {"error": "missing_video", "video_path": video_path}
    try:
        query_cutoff = mcq_query_cutoff({"realtime": query_cutoff})
        prompt = build_mcq_prompt(question, options)
        route = _answer_route_for(
            question, options, routing=routing, memory_floor=memory_floor,
            gate_strict_sim=gate_strict_sim,
            mcq_prompt_policy=mcq_prompt_policy,
            history_mode=history_mode,
        )
        chunks, decode_backend = decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
            video_end=query_cutoff,
        )
        if not chunks:
            return None, {"error": "empty_video", "video_path": video_path}
        decoded_timestamps = [timestamp for chunk in chunks
                              for timestamp, _ in chunk_frames_with_timestamps(chunk)]
        if (not decoded_timestamps or decoded_timestamps != sorted(decoded_timestamps)
                or any(timestamp < 0 or timestamp > query_cutoff for timestamp in decoded_timestamps)):
            raise ValueError("Decoded MCQ frame PTS exceed the query cutoff or are unavailable.")
        max_decoded_timestamp = max(decoded_timestamps)

        window = max(1, recent_frames_only)
        hist_chunks = chunks[:-window] if len(chunks) > window else []
        recent_chunks = chunks[-window:]
        # At the default 1 fps / 1 s chunks this is the same selected 4-frame
        # window. Higher sampling rates still respect the actual frame budget.
        recent_items = [item for chunk in recent_chunks
                        for item in chunk_frames_with_timestamps(chunk)][-window:]
        recent_timestamps = [timestamp for timestamp, _ in recent_items]
        recent_frames = [frame for _, frame in recent_items]

        t0 = time.perf_counter()
        memory = (
            evaluator.build_memory_from_chunks(hist_chunks, question=prompt)
            if route.use_memory
            else evaluator._make_memory()
        )
        evaluator.last_retrieval = None
        answer_kwargs = (
            {"answer_instruction": route.answer_instruction}
            if route.answer_instruction else {}
        )
        if use_logits:
            response = evaluator.answer_with_memory_mcq(
                memory,
                recent_frames,
                prompt,
                num_options=route.num_options,
                include_no_match_signal=route.include_no_match_signal,
                min_evidence_sim=route.min_evidence_sim,
                **answer_kwargs,
            )
        else:
            response = evaluator.answer_with_memory(
                memory,
                recent_frames,
                prompt,
                include_no_match_signal=route.include_no_match_signal,
                min_evidence_sim=route.min_evidence_sim,
                **answer_kwargs,
            )
        elapsed = time.perf_counter() - t0
        retrieval = evaluator.last_retrieval

        metadata = {
            "answer_method": route.method_family,
            "answer_policy": route.policy,
            "answer_num_options": route.num_options,
            "routing_mode": routing,
            "mcq_prompt_policy": mcq_prompt_policy,
            "history_mode": history_mode,
            "route_use_memory": route.use_memory,
            "route_min_evidence_sim": route.min_evidence_sim,
            "memory_floor": memory_floor,
            "gate_strict_sim": gate_strict_sim,
            "answer_instruction": route.answer_instruction,
            "include_no_match_signal": route.include_no_match_signal,
            "retrieval_query": prompt,
            "retrieval_context": retrieval.context if retrieval is not None else "",
            "retrieval_dynamic_top_k_max": memory.dynamic_top_k_max,
            **(route.gate.as_metadata() if route.gate is not None else {}),
            "decode_backend": decode_backend,
            "query_cutoff": query_cutoff,
            "max_decoded_timestamp": max_decoded_timestamp,
            "generate_time": elapsed,
            "num_hist_chunks": len(hist_chunks),
            "num_recent_frames": len(recent_frames),
            "recent_frame_timestamps": recent_timestamps,
            "retrieval_matched_nodes": (
                retrieval.matched_nodes if retrieval is not None else None
            ),
            "retrieval_signal": retrieval.signal if retrieval is not None else None,
            "retrieval_hit_count": retrieval.hit_count if retrieval is not None else None,
            "retrieval_hit_sims": round_sims(retrieval, "hit_sims"),
            "retrieval_candidate_sims": round_sims(retrieval, "candidate_sims"),
            **{f"hub_spoke_{k}": v for k, v in memory.stats().items()},
        }
        return response, metadata
    except Exception:
        logger.exception("MCQ sample failed: video=%s", video_path)
        return None, {"error": "exception", "video_path": video_path}


def evaluate_backward_realtime(
    anno: dict[str, Any],
    chunked_dir: str,
    evaluator: HubAndSpokeEvaluator,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    use_logits: bool = True,
    routing: str = ROUTING_KEYWORD,
    memory_floor: float = DEFAULT_MEMORY_FLOOR,
    gate_strict_sim: float = DEFAULT_OVO_GATE_STRICT_SIM,
    mcq_prompt_policy: str = MCQ_PROMPT_UNIFORM,
    history_mode: str = HISTORY_DHSM,
) -> dict[str, Any]:
    video_path = os.path.join(chunked_dir, f"{anno['id']}.mp4")
    response, metadata = _build_and_answer(
        anno["question"], list(anno["options"]), mcq_query_cutoff(anno),
        video_path,
        evaluator,
        chunk_duration,
        fps,
        recent_frames_only,
        use_logits=use_logits,
        routing=routing,
        memory_floor=memory_floor,
        gate_strict_sim=gate_strict_sim,
        mcq_prompt_policy=mcq_prompt_policy,
        history_mode=history_mode,
    )
    return {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "question": anno["question"],
        "options": list(anno["options"]),
        "response": response,
        "ground_truth": chr(65 + anno["gt"]),
        "routing_mode": routing,
        "mcq_prompt_policy": mcq_prompt_policy,
        "history_mode": history_mode,
        **metadata,
    }


# ---------------------------------------------------------------------------
# Forward readouts: shared frame-sampling helpers.
# ---------------------------------------------------------------------------

def fmt_time(seconds: float) -> str:
    whole = int(seconds)
    minutes, secs = divmod(whole, 60)
    return f"{minutes:02d}:{secs:02d}"


def chunk_frames_with_timestamps(chunk) -> list[tuple[float, Any]]:
    frames = getattr(chunk, "frames", None) or []
    if not frames:
        return []
    timestamps = getattr(chunk, "frame_timestamps", None)
    if timestamps is None or len(timestamps) != len(frames):
        raise ValueError("Nonempty video chunks require one actual frame PTS per frame.")
    if any(isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts)
           for ts in timestamps):
        raise ValueError("Video frame PTS must be finite numeric timestamps.")
    return [(float(ts), frame) for ts, frame in zip(timestamps, frames)]


def uniform_indices(total: int, k: int) -> list[int]:
    if k <= 0 or total <= 0:
        return []
    if total <= k:
        return list(range(total))
    if k == 1:
        return [total - 1]
    return sorted({round(i * (total - 1) / (k - 1)) for i in range(k)})


def clamp_count(value: int | None, cap: int = 0) -> int | None:
    if value is None:
        return None
    value = max(0, int(value))
    if cap and cap > 0:
        value = min(value, int(cap))
    return value


def parse_int_response(response: str | None) -> int | None:
    if response is None:
        return None
    match = re.search(r"\d+", str(response))
    return int(match.group()) if match else None


def _sample_frames_between(
    chunks: list,
    start_time: float,
    end_time: float,
    max_frames: int,
    context_seconds: float = 0.0,
) -> tuple[list[Any], list[float], bool]:
    lower = max(0.0, float(start_time) - max(0.0, float(context_seconds)))
    upper = float(end_time)
    items: list[tuple[float, Any]] = []
    has_context = False
    for chunk in chunks:
        for ts, frame in chunk_frames_with_timestamps(chunk):
            if lower < ts <= upper:
                items.append((ts, frame))
                if ts <= start_time:
                    has_context = True
    items.sort(key=lambda item: item[0])
    selected = [items[i] for i in uniform_indices(len(items), max(1, int(max_frames)))]
    return [frame for _, frame in selected], [ts for ts, _ in selected], has_context


def _rec_interval_prompt(
    activity: str,
    previous_cutoff: float,
    current_cutoff: float,
    current_count: int,
    max_delta: int,
    has_context: bool,
) -> str:
    options = [f"{chr(65 + i)}. {i} new complete instance(s)" for i in range(max_delta + 1)]
    unclear_letter = chr(65 + max_delta + 1)
    options.append(f"{unclear_letter}. unclear, count 0 new instances")
    context_note = (
        "Some early frames may be context from just before the interval; use "
        "them only to understand continuity, not to count earlier events.\n"
        if has_context else ""
    )
    return (
        "You are updating a cumulative repetition count for a video prefix.\n\n"
        f"Target activity: {activity}\n"
        f"Previous query time: {fmt_time(previous_cutoff)}\n"
        f"Current query time: {fmt_time(current_cutoff)}\n"
        f"Current count before this interval: {current_count}\n\n"
        f"{context_note}"
        "Look at the provided chronological frames. Count only NEW complete "
        "instances whose defining/peak moment happens AFTER the previous query "
        "time and AT OR BEFORE the current query time. Do not count "
        "preparation, waiting, resetting, continuation, or aftermath.\n\n"
        "Choose the best option:\n" + "\n".join(options) + "\n\nReturn only the letter."
    )


def _crr_interval_prompt(
    question: str,
    previous_cutoff: float,
    current_cutoff: float,
    has_context: bool,
) -> str:
    context_note = (
        "Some earliest frames may be context from just before the interval; "
        "use them only to understand continuity, not as newly appeared answer "
        "evidence.\n\n"
        if has_context else ""
    )
    return (
        "You are updating whether a video question has become answerable.\n\n"
        f"Question: {question}\n"
        f"Previous query time: {fmt_time(previous_cutoff)}\n"
        f"Current query time: {fmt_time(current_cutoff)}\n\n"
        f"{context_note}"
        "Look at the provided chronological frames for this interval. Did this "
        "interval reveal NEW visual evidence needed to answer the question? "
        "The evidence can be an event, action, object, person, location, or "
        "outcome that makes the answer visually determinable.\n\n"
        "Options:\n"
        "A. Yes, the answer evidence appeared in this interval.\n"
        "B. No, the necessary evidence has not appeared yet.\n"
        "C. Unclear, treat as No.\n\n"
        "Respond only with A, B, or C."
    )


def _yes_no_to_ab_prompt(question: str) -> str:
    return (
        f"{question.strip()}\n\n"
        "Options:\n"
        "A. Yes\n"
        "B. No\n\n"
        "Respond only with the letter A or B."
    )


def _ab_to_yes_no(answer: str | None) -> str | None:
    if answer is None:
        return None
    text = str(answer).strip().upper()
    if re.search(r"\bA\b", text):
        return "Yes"
    if re.search(r"\bB\b", text):
        return "No"
    if "YES" in text or text == "Y":
        return "Yes"
    if "NO" in text or text == "N":
        return "No"
    return None


def _ssr_prompt(step: str) -> str:
    return (
        "You are watching the current prefix of a tutorial video. Determine "
        "whether the person is currently performing the specified step in the "
        "latest frames.\n\n"
        f"Step: {step}\n\n"
        "Use the latest frames as the primary evidence. If the step is visibly "
        "underway or the current action is a clear part of that step, answer "
        "Yes. If the step has not started, has already finished, or is not "
        "visible in the latest frames, answer No."
    )


class RecIntervalCounter:
    """REC forward readout: interval-delta counting between adjacent query cutoffs."""

    def __init__(
        self,
        qa_model: RecentWindowQAModel,
        interval_frames: int = 16,
        interval_context_seconds: float = 1.0,
        interval_max_delta: int = 2,
        max_count_cap: int = 0,
    ) -> None:
        self.qa = qa_model
        self.interval_frames = max(1, int(interval_frames))
        self.interval_context_seconds = max(0.0, float(interval_context_seconds))
        self.interval_max_delta = max(1, min(4, int(interval_max_delta)))
        self.max_count_cap = max(0, int(max_count_cap))

    def _count_deltas(
        self,
        activity: str,
        chunks: list,
        sub_tests: list[tuple[int, dict[str, Any]]],
    ) -> dict[int, dict[str, Any]]:
        count = 0
        previous_cutoff = 0.0
        updates: dict[int, dict[str, Any]] = {}
        label_counts = {chr(65 + i): 0 for i in range(self.interval_max_delta + 2)}
        total_time = 0.0

        for orig_idx, ti in sub_tests:
            cutoff = float(ti["realtime"])
            if cutoff <= previous_cutoff:
                updates[orig_idx] = {
                    "interval_delta_response": str(count),
                    "interval_delta_new_count": 0,
                    "interval_delta_label": "A",
                    "interval_delta_num_frames": 0,
                    "interval_delta_frame_timestamps": [],
                    "interval_delta_prev_cutoff": previous_cutoff,
                    "interval_delta_cutoff": cutoff,
                    "interval_delta_label_counts": dict(label_counts),
                    "interval_delta_time_total": total_time,
                }
                previous_cutoff = cutoff
                continue

            frames, timestamps, has_context = _sample_frames_between(
                chunks,
                start_time=previous_cutoff,
                end_time=cutoff,
                max_frames=self.interval_frames,
                context_seconds=self.interval_context_seconds,
            )
            if not frames:
                label = "A"
                delta = 0
                elapsed = 0.0
            else:
                prompt = _rec_interval_prompt(
                    activity,
                    previous_cutoff=previous_cutoff,
                    current_cutoff=cutoff,
                    current_count=count,
                    max_delta=self.interval_max_delta,
                    has_context=has_context,
                )
                t0 = time.perf_counter()
                label = self.qa.score_mcq_from_frames(
                    frames, prompt, num_options=self.interval_max_delta + 2,
                )
                elapsed = time.perf_counter() - t0
                if label not in label_counts:
                    label = chr(65 + self.interval_max_delta + 1)
                option_idx = ord(label) - 65
                delta = option_idx if option_idx <= self.interval_max_delta else 0

            label_counts[label] += 1
            total_time += elapsed
            count += delta
            count = clamp_count(count, self.max_count_cap) or 0
            updates[orig_idx] = {
                "interval_delta_response": str(count),
                "interval_delta_new_count": delta,
                "interval_delta_label": label,
                "interval_delta_num_frames": len(frames),
                "interval_delta_frame_timestamps": timestamps,
                "interval_delta_first_ts": timestamps[0] if timestamps else None,
                "interval_delta_last_ts": timestamps[-1] if timestamps else None,
                "interval_delta_prev_cutoff": previous_cutoff,
                "interval_delta_cutoff": cutoff,
                "interval_delta_has_context": has_context,
                "interval_delta_label_counts": dict(label_counts),
                "interval_delta_time_total": total_time,
            }
            previous_cutoff = cutoff

        return updates

    def evaluate(
        self,
        anno: dict[str, Any],
        chunks: list,
        sub_tests: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        activity = anno["activity"]
        interval_updates = self._count_deltas(activity, chunks, sub_tests)

        results: list[tuple[int, dict[str, Any]]] = []
        prev_count: int | None = None
        for orig_idx, ti in sub_tests:
            cutoff = ti["realtime"]
            prefix_chunks = [c for c in chunks if c.end_time <= cutoff]
            update = dict(interval_updates.get(orig_idx, {
                "interval_delta_response": "0",
                "interval_delta_new_count": 0,
            }))

            current = parse_int_response(update.get("interval_delta_response"))
            if current is not None:
                if prev_count is not None and current < prev_count:
                    update["interval_delta_clamped_from"] = str(current)
                    current = prev_count
                    update["interval_delta_response"] = str(current)
                prev_count = current

            update["response"] = update.get("interval_delta_response")
            update["primary_method"] = "interval_delta"
            update["realtime_cutoff"] = cutoff
            update["num_prefix_chunks"] = len(prefix_chunks)
            results.append((orig_idx, update))

        for chunk in chunks:
            chunk.frames = []
        return results


class CrrIntervalTracker:
    """CRR forward readout: interval-evidence answerability tracking."""

    def __init__(
        self,
        qa_model: RecentWindowQAModel,
        interval_frames: int = 16,
        interval_context_seconds: float = 1.0,
        force_first_no: bool = False,
    ) -> None:
        self.qa = qa_model
        self.interval_frames = max(1, int(interval_frames))
        self.interval_context_seconds = max(0.0, float(interval_context_seconds))
        self.force_first_no = bool(force_first_no)

    def _track_evidence(
        self,
        anno: dict[str, Any],
        chunks: list,
        sub_tests: list[tuple[int, dict[str, Any]]],
    ) -> dict[int, dict[str, Any]]:
        updates: dict[int, dict[str, Any]] = {}
        answerable = False
        previous_cutoff: float | None = None
        label_counts = {"A": 0, "B": 0, "C": 0}
        total_time = 0.0

        for step_idx, (orig_idx, ti) in enumerate(sub_tests):
            cutoff = float(ti["realtime"])
            if previous_cutoff is None:
                previous_cutoff = float(anno.get("ask_time", cutoff))

            if step_idx == 0 and self.force_first_no:
                label = "B"
                response = "No"
                frames: list[Any] = []
                timestamps: list[float] = []
                has_context = False
                elapsed = 0.0
            elif answerable:
                label = "A"
                response = "Yes"
                frames = []
                timestamps = []
                has_context = False
                elapsed = 0.0
            else:
                frames, timestamps, has_context = _sample_frames_between(
                    chunks,
                    start_time=previous_cutoff,
                    end_time=cutoff,
                    max_frames=self.interval_frames,
                    context_seconds=self.interval_context_seconds,
                )
                if not frames:
                    label = "B"
                    response = "No"
                    elapsed = 0.0
                else:
                    prompt = _crr_interval_prompt(
                        anno["question"],
                        previous_cutoff=previous_cutoff,
                        current_cutoff=cutoff,
                        has_context=has_context,
                    )
                    t0 = time.perf_counter()
                    raw = self.qa.score_mcq_from_frames(frames, prompt, num_options=3)
                    elapsed = time.perf_counter() - t0
                    label = raw if raw in label_counts else "C"
                    if label == "A":
                        answerable = True
                    response = "Yes" if answerable else "No"

            label_counts[label] += 1
            total_time += elapsed
            updates[orig_idx] = {
                "interval_evidence_response": response,
                "interval_evidence_label": label,
                "interval_evidence_answerable": answerable,
                "interval_evidence_prev_cutoff": previous_cutoff,
                "interval_evidence_cutoff": cutoff,
                "interval_evidence_num_frames": len(frames),
                "interval_evidence_frame_timestamps": timestamps,
                "interval_evidence_first_ts": timestamps[0] if timestamps else None,
                "interval_evidence_last_ts": timestamps[-1] if timestamps else None,
                "interval_evidence_has_context": has_context,
                "interval_evidence_label_counts": dict(label_counts),
                "interval_evidence_time_total": total_time,
            }
            previous_cutoff = cutoff

        return updates

    def evaluate(
        self,
        anno: dict[str, Any],
        chunks: list,
        sub_tests: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        updates = self._track_evidence(anno, chunks, sub_tests)
        results: list[tuple[int, dict[str, Any]]] = []
        for orig_idx, ti in sub_tests:
            update = dict(updates.get(orig_idx, {}))
            update["response"] = update.get("interval_evidence_response")
            update["primary_method"] = "interval_evidence"
            update["realtime_cutoff"] = ti["realtime"]
            results.append((orig_idx, update))
        for chunk in chunks:
            chunk.frames = []
        return results


class ForwardEvaluator:
    """Combines the REC/SSR/CRR forward readouts for one streaming pass."""

    def __init__(
        self,
        qa_model: RecentWindowQAModel,
        recent_frames: int = 4,
        rec_interval_frames: int = 16,
        rec_interval_context_seconds: float = 1.0,
        rec_interval_max_delta: int = 2,
        crr_interval_frames: int = 16,
        crr_interval_context_seconds: float = 1.0,
        crr_force_first_no: bool = False,
    ) -> None:
        self.qa = qa_model
        self.recent_frames = max(1, int(recent_frames))
        self.rec = RecIntervalCounter(
            qa_model=qa_model,
            interval_frames=rec_interval_frames,
            interval_context_seconds=rec_interval_context_seconds,
            interval_max_delta=rec_interval_max_delta,
        )
        self.crr = CrrIntervalTracker(
            qa_model=qa_model,
            interval_frames=crr_interval_frames,
            interval_context_seconds=crr_interval_context_seconds,
            force_first_no=crr_force_first_no,
        )

    def _score_yes_no(self, frames: list, prompt: str) -> str | None:
        answer = self.qa.score_mcq_from_frames(
            frames, _yes_no_to_ab_prompt(prompt), num_options=2,
        )
        return _ab_to_yes_no(answer)

    def stream_ssr(
        self,
        chunks,
        sub_tests: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        window = self.recent_frames
        recent_buffer: list = []
        results: list[tuple[int, dict[str, Any]]] = []

        chunk_iter = iter(chunks)
        pending = next(chunk_iter, None)

        for orig_idx, ti in sub_tests:
            cutoff = ti["realtime"]
            t0 = time.perf_counter()
            while pending is not None and pending.end_time <= cutoff:
                recent_buffer.append(pending)
                while len(recent_buffer) > window:
                    old = recent_buffer.pop(0)
                    old.frames = []
                pending = next(chunk_iter, None)

            # The decoder provides actual PTS. Filter the frame timestamps as
            # well as chunk boundaries, then log exactly what the model saw.
            recent_items = [
                (timestamp, frame)
                for chunk in recent_buffer
                for timestamp, frame in chunk_frames_with_timestamps(chunk)
                if timestamp <= cutoff
            ]
            recent_frames = [frame for _, frame in recent_items]
            recent_timestamps = [timestamp for timestamp, _ in recent_items]
            prompt = _ssr_prompt(ti["step"])
            response = self._score_yes_no(recent_frames, prompt)
            elapsed = time.perf_counter() - t0
            results.append((
                orig_idx,
                {
                    "response": response,
                    "ssr_response": response,
                    "generate_time": elapsed,
                    "num_recent_chunks": len(recent_buffer),
                    "num_recent_frames": len(recent_frames),
                    "recent_frame_timestamps": recent_timestamps,
                    "recent_first_ts": recent_timestamps[0] if recent_timestamps else None,
                    "recent_last_ts": recent_timestamps[-1] if recent_timestamps else None,
                    "realtime_cutoff": cutoff,
                    "readout": "recent_window_ab_logits",
                    "readout_task": "SSR",
                },
            ))

        for chunk in recent_buffer:
            chunk.frames = []
        return results

    def stream_rec(
        self,
        anno: dict[str, Any],
        chunks,
        sub_tests: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        rows = self.rec.evaluate(anno, chunks, sub_tests)
        memory = HubAndSpokeMemory()
        state_key = "count"
        results: list[tuple[int, dict[str, Any]]] = []
        for orig_idx, update in rows:
            update = dict(update)
            state = {
                "state_type": "count",
                "update_rule": "interval_delta",
                "response": update.get("interval_delta_response"),
                "new_count": update.get("interval_delta_new_count"),
                "label": update.get("interval_delta_label"),
                "prev_cutoff": update.get("interval_delta_prev_cutoff"),
                "cutoff": update.get("interval_delta_cutoff"),
                "num_frames": update.get("interval_delta_num_frames"),
                "first_ts": update.get("interval_delta_first_ts"),
                "last_ts": update.get("interval_delta_last_ts"),
                "has_context": update.get("interval_delta_has_context"),
                "label_counts": update.get("interval_delta_label_counts"),
                "time_total": update.get("interval_delta_time_total"),
            }
            memory.update_task_state(state_key, state)
            hub_state = memory.get_task_state(state_key) or state
            update["response"] = update.get("interval_delta_response")
            update["answer_method"] = "hub_and_spoke_memory"
            update["answer_policy"] = "interval_delta_count_state"
            update["readout"] = "hub_and_spoke_interval_state"
            update["readout_policy"] = "count_delta"
            update["readout_task"] = "REC"
            update["hub_spoke_task_state_key"] = state_key
            update["hub_spoke_interval_delta_response"] = hub_state.get("response")
            update["hub_spoke_interval_delta_new_count"] = hub_state.get("new_count")
            update["hub_spoke_interval_delta_label"] = hub_state.get("label")
            update["hub_spoke_interval_delta_prev_cutoff"] = hub_state.get("prev_cutoff")
            update["hub_spoke_interval_delta_cutoff"] = hub_state.get("cutoff")
            update["hub_spoke_interval_delta_num_frames"] = hub_state.get("num_frames")
            update["hub_spoke_interval_delta_has_context"] = hub_state.get("has_context")
            update.update({f"hub_spoke_{k}": v for k, v in memory.stats().items()})
            results.append((orig_idx, update))
        return results

    def stream_crr(
        self,
        anno: dict[str, Any],
        chunks,
        sub_tests: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        rows = self.crr.evaluate(anno, chunks, sub_tests)
        results: list[tuple[int, dict[str, Any]]] = []
        for orig_idx, update in rows:
            update = dict(update)
            update["response"] = update.get("interval_evidence_response")
            update["readout"] = "interval_state"
            update["readout_policy"] = "evidence_presence"
            update["readout_task"] = "CRR"
            results.append((orig_idx, update))
        return results


def _stream_forward_rows(
    task: str | None,
    anno: dict[str, Any],
    chunks: list[Any],
    sorted_sub: list[tuple[int, dict[str, Any]]],
    evaluator: ForwardEvaluator,
) -> list[tuple[int, dict[str, Any]]]:
    if task == "REC":
        return evaluator.stream_rec(anno, chunks, sorted_sub)
    if task == "SSR":
        return evaluator.stream_ssr(chunks, sorted_sub)
    if task == "CRR":
        return evaluator.stream_crr(anno, chunks, sorted_sub)
    return []


def evaluate_forward(
    anno: dict[str, Any],
    chunked_dir: str,
    evaluator: ForwardEvaluator,
    chunk_duration: float,
    fps: float,
) -> dict[str, Any]:
    result_anno = copy.deepcopy(anno)
    test_info = result_anno.get("test_info", [])
    n_sub = len(test_info)
    if not n_sub:
        return result_anno

    longest_path = os.path.join(chunked_dir, f"{anno['id']}_{n_sub - 1}.mp4")
    if not os.path.exists(longest_path):
        return result_anno

    try:
        chunks, decode_backend = decode_video_to_chunks_qwen(
            video_path=longest_path,
            chunk_duration=chunk_duration,
            fps=fps,
        )
    except Exception:
        logger.exception(
            "decode failed: id=%s task=%s video=%s",
            anno.get("id"), anno.get("task"), longest_path,
        )
        return result_anno
    if not chunks:
        return result_anno

    sorted_sub = sorted(enumerate(test_info), key=lambda kv: kv[1]["realtime"])
    task = anno.get("task")
    try:
        rows = _stream_forward_rows(task, anno, chunks, sorted_sub, evaluator)
    except Exception:
        logger.exception("forward sample failed: id=%s task=%s", anno.get("id"), task)
        return result_anno

    for orig_idx, update in rows:
        update = dict(update)
        update["decode_backend"] = decode_backend
        test_info[orig_idx].update(update)
    return result_anno


# ---------------------------------------------------------------------------
# Checkpointing: thin wrappers over dhsm.shard_io that bucket rows by task.
# ---------------------------------------------------------------------------

def _section_for_task(task: str | None) -> str | None:
    if task in BACKWARD_TASKS:
        return "backward"
    if task in REAL_TIME_TASKS:
        return "realtime"
    if task in FORWARD_TASKS:
        return "forward"
    return None


def _bucket_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    backward, realtime, forward = [], [], []
    for row in rows:
        section = _section_for_task(row.get("task"))
        if section == "backward":
            backward.append(row)
        elif section == "realtime":
            realtime.append(row)
        elif section == "forward":
            forward.append(row)
    return backward, realtime, forward


def load_checkpoint_state(path: str):
    rows, done_keys = shard_io.resume(path, row_key)
    backward, realtime, forward = _bucket_rows(rows)
    return backward, realtime, forward, done_keys


def merge_shard_results(result_dir: str, n_procs: int):
    return _bucket_rows(shard_io.merge_shards(result_dir, n_procs, key_fn=row_key))


def add_crr_first_response_arguments(parser: argparse.ArgumentParser) -> None:
    """Observe first-cutoff evidence by default; retain an explicit legacy flag."""
    first_response = parser.add_mutually_exclusive_group()
    first_response.add_argument(
        "--crr_force_first_no", dest="crr_no_force_first_no", action="store_false",
        help="Legacy CRR diagnostic: force No at the first cutoff without observing frames.",
    )
    first_response.add_argument(
        "--crr_no_force_first_no", dest="crr_no_force_first_no", action="store_true",
        help="Use observed evidence at the first CRR cutoff (default; retained compatibility flag).",
    )
    parser.set_defaults(crr_no_force_first_no=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hub-and-Spoke full OVO-Bench evaluation"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--anno_path",
        default="data/ovo_bench/ovo_bench_new.json",
    )
    parser.add_argument(
        "--chunked_dir",
        default="data/ovo_bench/chunked_videos",
    )
    parser.add_argument("--result_dir", default="results/hub_and_spoke_ovo")
    parser.add_argument("--recent_frames_only", type=int, default=4)
    parser.add_argument("--chunk_duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_qa_tokens", type=int, default=256)
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        help="Attention backend for the VLM (default: flash_attention_2; pass sdpa to skip flash-attn).",
    )
    parser.add_argument("--extract_every_n_chunks", type=int, default=1)
    parser.add_argument("--max_extraction_chunks", type=int, default=20)
    parser.add_argument("--sim_threshold", type=float, default=0.25)
    parser.add_argument(
        "--routing",
        choices=ROUTING_CHOICES,
        default=ROUTING_KEYWORD,
        help=(
            "Gate on question/option text, without MCQ task annotations or GT."
        ),
    )
    parser.add_argument(
        "--history_mode",
        choices=HISTORY_MODES,
        default=HISTORY_DHSM,
        help=(
            "MCQ history ablation: 'dhsm' (default) follows the retrieval gate; "
            "'recent_only' skips all history construction while using the same "
            "recent visual window, MCQ prompt policy and QA method. "
            "recent_only requires --splits backward,realtime (or a subset)."
        ),
    )
    parser.add_argument(
        "--mcq_prompt_policy",
        choices=MCQ_PROMPT_POLICIES,
        default=MCQ_PROMPT_UNIFORM,
        help=(
            "'uniform_abstention' (default) adds one fixed evidence instruction "
            "to every MCQ after retrieval; 'official' uses the original benchmark "
            "QA prompt. Both use the same original caption/retrieval query."
        ),
    )
    parser.add_argument(
        "--memory_floor", type=float, default=None,
        help="Post-filter floor for strong-memory retrieval seeds; default follows the backbone profile.",
    )
    parser.add_argument(
        "--gate_strict_sim", type=float, default=None,
        help="Post-filter floor for ambiguous retrieval seeds; default follows the backbone profile. "
             "Neither floor changes the dynamic cutoff or graph/timeline expansion.",
    )
    parser.add_argument(
        "--top_k",
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
        "--dynamic_top_k_max", type=int, default=None,
        help="Maximum candidate pool for dynamic retrieval; default follows the backbone profile. "
             "The original elbow rule can select fewer seeds; integer --top_k is unchanged.",
    )
    parser.add_argument(
        "--embed_model",
        default=DEFAULT_EMBED_MODEL,
        help="Sentence-transformer for node/query embeddings.",
    )
    parser.add_argument("--count_question_max_chunks", type=int, default=20)
    parser.add_argument(
        "--caption_batch_size", type=int, default=4,
        help="History caption batch size (default: 4); changing it can change captions.",
    )
    parser.add_argument("--max_samples_per_split", type=int, default=None)
    parser.add_argument(
        "--splits",
        default="backward,realtime,forward",
        help="Comma-separated subset of splits to run: backward,realtime,forward.",
    )
    parser.add_argument(
        "--no_expansion",
        action="store_true",
        help=(
            "Ablation: disable hub-and-spoke expansion during retrieval "
            "(render only directly hit memory nodes)."
        ),
    )
    parser.add_argument(
        "--memory_mode",
        choices=list(MEMORY_MODE_CHOICES),
        default=None,
        help="Memory implementation; defaults to entity_resolved with chunk-local ID repair.",
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
        "--use_logits",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rec_interval_frames", type=int, default=16)
    parser.add_argument("--rec_interval_context_seconds", type=float, default=1.0)
    parser.add_argument("--rec_interval_max_delta", type=int, default=2)
    parser.add_argument("--crr_interval_frames", type=int, default=16)
    parser.add_argument("--crr_interval_context_seconds", type=float, default=1.0)
    add_crr_first_response_arguments(parser)
    args = parser.parse_args()
    defaults = ovo_defaults(args.model_path)
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if args.dynamic_top_k_max < 1:
        parser.error("--dynamic_top_k_max must be positive.")
    for name in ("sim_threshold", "memory_floor", "gate_strict_sim"):
        value = getattr(args, name)
        if not math.isfinite(value) or not 0 <= value <= 1:
            parser.error(f"--{name} must be finite and between 0 and 1.")
    if args.recent_frames_only < 1 or args.max_extraction_chunks < 1 or args.count_question_max_chunks < 1:
        parser.error("Recent-frame and historical-caption budgets must be positive.")
    if (not math.isfinite(args.fps) or not math.isfinite(args.chunk_duration)
            or args.fps <= 0 or args.chunk_duration <= 0):
        parser.error("--fps and --chunk_duration must be positive and finite.")
    if args.extract_every_n_chunks < 1 or args.caption_batch_size < 0:
        parser.error("Extraction stride must be positive and caption batch size nonnegative.")
    args.answer_instruction_sha256 = (
        hashlib.sha256(UNIFORM_ABSTENTION_INSTRUCTION.encode()).hexdigest()
        if args.mcq_prompt_policy == MCQ_PROMPT_UNIFORM else None
    )
    # These environment variables change processor resolution. Record their
    # effective overrides so paired runs and resume checks can compare them.
    args.min_pixels = int(os.environ["MIN_PIXELS"]) if os.environ.get("MIN_PIXELS") else None
    args.max_pixels = int(os.environ["MAX_PIXELS"]) if os.environ.get("MAX_PIXELS") else None
    args.video_environment = {
        key: os.environ.get(key)
        for key in ("VIDEO_MAX_PIXELS", "FORCE_QWENVL_VIDEO_READER", "TORCHCODEC_NUM_THREADS")
    }
    args.runtime_versions = {}
    for package in (
        "torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils",
        "numpy", "pillow", "av", "decord", "torchcodec", "sentence-transformers",
    ):
        try:
            args.runtime_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            args.runtime_versions[package] = None
    source_hash = hashlib.sha256()
    source_files = sorted((PROJECT_ROOT / "dhsm").glob("*.py")) + [
        PROJECT_ROOT / "experiments" / name
        for name in ("evaluate_ovo.py", "ovo_bench.py", "ovo_protocol.py")
    ]
    for source_file in source_files:
        source_hash.update(str(source_file.relative_to(PROJECT_ROOT)).encode() + b"\0")
        source_hash.update(source_file.read_bytes())
    args.evaluation_code_sha256 = source_hash.hexdigest()
    # Factory and memory modules are covered by the dhsm/*.py source digest.
    # Keyword routing defaults to the repaired entity-resolved memory.
    args.memory_mode = resolve_memory_mode(args.routing, args.memory_mode)

    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=24))]
    )

    import random
    annotation_bytes = Path(args.anno_path).read_bytes()
    annotations = json.loads(annotation_bytes)
    args.annotation_sha256 = hashlib.sha256(annotation_bytes).hexdigest()
    for annotation in annotations:
        if type(annotation.get("id")) is not int or annotation["id"] < 0:
            raise ValueError("Annotation IDs must be nonnegative integers.")
        if annotation.get("task") in BACKWARD_TASKS + REAL_TIME_TASKS:
            mcq_query_cutoff(annotation)
            build_mcq_prompt(annotation.get("question"), annotation.get("options"))

    active_splits = {s.strip().lower() for s in args.splits.split(",") if s.strip()}
    unknown_splits = active_splits - {"backward", "realtime", "forward"}
    if unknown_splits:
        raise SystemExit(f"Unknown --splits entries: {sorted(unknown_splits)}")
    if args.history_mode == HISTORY_RECENT_ONLY and "forward" in active_splits:
        raise SystemExit(
            "--history_mode recent_only applies to MCQ tasks; "
            "use --splits backward,realtime (or a subset), excluding forward."
        )

    # Every rank checks the same manifest independently before loading models.
    # No rank waits at a barrier if another rejects incompatible old results.
    ensure_result_protocol(
        args.result_dir,
        {
            "benchmark": "ovo",
            "config": vars(args),
            "annotation_sha256": args.annotation_sha256,
            "num_processes": accelerator.num_processes,
        },
    )
    # Remove every stale marker before any rank can finish a resumed shard.
    # Per-rank cleanup races with rank 0 checking other ranks' old markers.
    if accelerator.is_main_process:
        shard_io.clear_done_markers(args.result_dir, accelerator.num_processes)
    accelerator.wait_for_everyone()

    backward_anno = [a for a in annotations if a["task"] in BACKWARD_TASKS]
    realtime_anno = [a for a in annotations if a["task"] in REAL_TIME_TASKS]
    forward_anno = [a for a in annotations if a["task"] in FORWARD_TASKS]
    if "backward" not in active_splits:
        backward_anno = []
    if "realtime" not in active_splits:
        realtime_anno = []
    if "forward" not in active_splits:
        forward_anno = []

    random.seed(42)
    random.shuffle(backward_anno)
    random.shuffle(realtime_anno)
    random.shuffle(forward_anno)
    if args.max_samples_per_split is not None:
        backward_anno = backward_anno[: args.max_samples_per_split]
        realtime_anno = realtime_anno[: args.max_samples_per_split]
        forward_anno = forward_anno[: args.max_samples_per_split]

    accelerator.print(f"\n{'=' * 60}")
    accelerator.print("Hub-and-Spoke OVO-Bench Evaluation")
    accelerator.print(f"{'=' * 60}")
    accelerator.print(
        f"Backward: {len(backward_anno)}, "
        f"Realtime: {len(realtime_anno)}, "
        f"Forward: {len(forward_anno)}"
    )
    accelerator.print(
        f"recent_frames={args.recent_frames_only}  "
        f"embed_model={args.embed_model}  sim_threshold={args.sim_threshold}  "
        f"top_k={format_top_k(args.top_k, args.dynamic_top_k_max)}  "
        f"routing={args.routing}  "
        f"mcq_prompt_policy={args.mcq_prompt_policy}  "
        f"history_mode={args.history_mode}  "
        f"memory_mode={args.memory_mode}  "
        f"memory_floor={args.memory_floor}  gate_strict_sim={args.gate_strict_sim}  "
        f"caption_batch_size={args.caption_batch_size}  "
        f"expansion={'off' if args.no_expansion else 'on'}  "
        f"REC interval={args.rec_interval_frames}f  "
        f"CRR interval={args.crr_interval_frames}f"
    )
    accelerator.print(f"{'=' * 60}\n")

    qa_model = RecentWindowQAModel(
        model_name=args.model_path,
        device=accelerator.device,
        max_new_tokens=args.max_qa_tokens,
        attn_implementation=args.attn_implementation,
    )
    evaluator_cls = evaluator_class_for(args.memory_mode)
    hub_spoke_evaluator = evaluator_cls(
        qa_model=qa_model,
        recent_frames=args.recent_frames_only,
        extract_every_n_chunks=args.extract_every_n_chunks,
        max_extraction_chunks=args.max_extraction_chunks,
        embed_model=args.embed_model,
        sim_threshold=args.sim_threshold,
        top_k=args.top_k,
        dynamic_top_k_max=args.dynamic_top_k_max,
        count_question_max_chunks=args.count_question_max_chunks,
        caption_batch_size=args.caption_batch_size,
        expand_retrieval=not args.no_expansion,
        expand_co_occurrence=not args.no_cooccurrence,
        expand_next_action=not args.no_next_action,
        merge_spokes=not args.no_spoke_merging,
    )
    forward_evaluator = ForwardEvaluator(
        qa_model=qa_model,
        recent_frames=args.recent_frames_only,
        rec_interval_frames=args.rec_interval_frames,
        rec_interval_context_seconds=args.rec_interval_context_seconds,
        rec_interval_max_delta=args.rec_interval_max_delta,
        crr_interval_frames=args.crr_interval_frames,
        crr_interval_context_seconds=args.crr_interval_context_seconds,
        crr_force_first_no=not args.crr_no_force_first_no,
    )

    with accelerator.split_between_processes(backward_anno) as local_backward:
        local_backward = list(local_backward)
    with accelerator.split_between_processes(realtime_anno) as local_realtime:
        local_realtime = list(local_realtime)
    with accelerator.split_between_processes(forward_anno) as local_forward:
        local_forward = list(local_forward)

    ckpt_path = shard_io.checkpoint_path(
        args.result_dir, accelerator.process_index, accelerator.num_processes
    )
    done_marker = shard_io.done_path(
        args.result_dir, accelerator.process_index, accelerator.num_processes
    )
    backward_results, realtime_results, forward_results, done_keys = load_checkpoint_state(
        ckpt_path
    )

    import torch
    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        for anno in tqdm(
            local_backward,
            desc=f"[GPU{accelerator.process_index}] Backward",
            disable=not accelerator.is_local_main_process,
        ):
            key = row_key(anno)
            if key in done_keys:
                continue
            result = evaluate_backward_realtime(
                anno,
                args.chunked_dir,
                hub_spoke_evaluator,
                args.chunk_duration,
                args.fps,
                args.recent_frames_only,
                use_logits=args.use_logits,
                routing=args.routing,
                memory_floor=args.memory_floor,
                gate_strict_sim=args.gate_strict_sim,
                mcq_prompt_policy=args.mcq_prompt_policy,
                history_mode=args.history_mode,
            )
            backward_results.append(result)
            done_keys.add(key)
            shard_io.append_row(ckpt, result, key)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for anno in tqdm(
            local_realtime,
            desc=f"[GPU{accelerator.process_index}] Realtime",
            disable=not accelerator.is_local_main_process,
        ):
            key = row_key(anno)
            if key in done_keys:
                continue
            result = evaluate_backward_realtime(
                anno,
                args.chunked_dir,
                hub_spoke_evaluator,
                args.chunk_duration,
                args.fps,
                args.recent_frames_only,
                use_logits=args.use_logits,
                routing=args.routing,
                memory_floor=args.memory_floor,
                gate_strict_sim=args.gate_strict_sim,
                mcq_prompt_policy=args.mcq_prompt_policy,
                history_mode=args.history_mode,
            )
            realtime_results.append(result)
            done_keys.add(key)
            shard_io.append_row(ckpt, result, key)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for anno in tqdm(
            local_forward,
            desc=f"[GPU{accelerator.process_index}] Forward",
            disable=not accelerator.is_local_main_process,
        ):
            key = row_key(anno)
            if key in done_keys:
                continue
            result = evaluate_forward(
                anno,
                args.chunked_dir,
                forward_evaluator,
                args.chunk_duration,
                args.fps,
            )
            result["routing_mode"] = args.routing
            result["mcq_prompt_policy"] = args.mcq_prompt_policy
            result["history_mode"] = args.history_mode
            forward_results.append(result)
            done_keys.add(key)
            shard_io.append_row(ckpt, result, key)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    shard_io.write_done_marker(done_marker)

    if accelerator.is_main_process:
        shard_io.wait_for_done_markers(args.result_dir, accelerator.num_processes)
        all_backward, all_realtime, all_forward = merge_shard_results(
            args.result_dir, accelerator.num_processes
        )
        model_label = f"HubAndSpoke-{args.model_path.split('/')[-1]}"
        print_report(model_label, all_backward, all_realtime, all_forward)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            args.result_dir, f"hub_and_spoke_results_{timestamp}.json"
        )
        shard_io.save_json(
            output_path,
            {
                "config": vars(args),
                "backward": all_backward,
                "realtime": all_realtime,
                "forward": all_forward,
            },
        )
        print(f"\nResults saved to: {output_path}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
