"""Hub-and-Spoke OVO-Bench evaluation: the full backward/real-time/forward pipeline.

Backward and real-time tasks are answered by retrieving from hub-and-spoke
memory (or a recent-window fallback) via ``dhsm.hub_and_spoke``. The forward
tasks use dedicated causal readouts: REC counts via interval-delta counting
(``RecIntervalCounter``), SSR via recent-window A/B logits, and CRR via
interval-evidence tracking (``CrrIntervalTracker``). Ships one CLI (``main``).

The other REC/CRR counting methods explored during development (storyboard,
segmented, contact_sheet, delta, event, hybrid for REC; tail, tail_mono for
CRR) live in ``evaluate_ovo_ablations.py``, which imports the frame-sampling
helpers defined here.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass
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
from dhsm.hub_and_spoke import (
    DEFAULT_EMBED_MODEL,
    DYNAMIC_TOP_K_MAX,
    HubAndSpokeEvaluator,
    HubAndSpokeMemory,
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
)
from dhsm.video_qa import decode_video_to_chunks_qwen
from dhsm.video_qa_qwen3 import RecentWindowQAModel

from experiments.ovo_bench import (
    BACKWARD_TASKS,
    FORWARD_TASKS,
    REAL_TIME_TASKS,
    build_prompt,
    print_report,
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


def format_top_k(top_k: int | str) -> str:
    if isinstance(top_k, str):
        return f"{top_k}(max={DYNAMIC_TOP_K_MAX})"
    return str(top_k)


@dataclass(frozen=True)
class AnswerRoute:
    method_family: str
    policy: str
    use_memory: bool
    include_no_match_signal: bool
    num_options: int
    min_evidence_sim: float | None = None
    gate: GateDecision | None = None


def _answer_route_for(
    anno: dict[str, Any],
    routing: str = ROUTING_KEYWORD,
    gate_strict_sim: float = DEFAULT_GATE_STRICT_SIM,
) -> AnswerRoute:
    """Decide how one annotation is answered.

    With ``routing="keyword"`` (default) the memory/recent-window decision
    comes from ``dhsm.retrieval_gate`` reading the question text only, which
    is the mechanism the paper describes.  ``routing="task_label"`` restores
    the previous behaviour, which read the OVO task label (EPM/ASI/HLD vs the
    six Real-Time tasks) — i.e. ground-truth benchmark metadata — and is kept
    only so the earlier numbers can be reproduced.

    NOTE: the HLD abstain policy still keys off the task label.  Deriving it
    from text is not possible: HLD questions are lexically identical to EPM
    ones ("Where is the rice cooker?"), so it is a separate discrepancy from
    the retrieval gate and is deliberately left untouched here.
    """
    task = anno.get("task")
    options = anno.get("options") or []
    num_options = len(options) if options else 4

    is_hld = task == "HLD"
    policy = "hld_abstain_on_insufficient_evidence" if is_hld else "standard_mcq"

    if routing == ROUTING_TASK_LABEL:
        decision = None
        use_memory = task in BACKWARD_TASKS
        min_evidence_sim = None
    else:
        decision = gate_question(str(anno.get("question", "")), options)
        use_memory = decision.needs_memory
        min_evidence_sim = gate_strict_sim if decision.strict_evidence else None

    return AnswerRoute(
        method_family="hub_and_spoke_memory" if use_memory else "recent_window",
        policy=policy,
        use_memory=use_memory,
        include_no_match_signal=is_hld,
        num_options=num_options,
        min_evidence_sim=min_evidence_sim,
        gate=decision,
    )


def _build_and_answer(
    anno: dict[str, Any],
    video_path: str,
    evaluator: HubAndSpokeEvaluator,
    chunk_duration: float,
    fps: float,
    recent_frames_only: int,
    prompt: str,
    use_logits: bool = False,
    routing: str = ROUTING_KEYWORD,
    gate_strict_sim: float = DEFAULT_GATE_STRICT_SIM,
) -> tuple[str | None, dict[str, Any]]:
    if not os.path.exists(video_path):
        return None, {}
    try:
        route = _answer_route_for(anno, routing=routing, gate_strict_sim=gate_strict_sim)
        chunks, decode_backend = decode_video_to_chunks_qwen(
            video_path=video_path,
            chunk_duration=chunk_duration,
            fps=fps,
        )
        if not chunks:
            return None, {}

        window = max(1, recent_frames_only)
        hist_chunks = chunks[:-window] if len(chunks) > window else []
        recent_chunks = chunks[-window:]
        recent_frames = [f for c in recent_chunks for f in c.frames]

        t0 = time.perf_counter()
        memory = (
            evaluator.build_memory_from_chunks(hist_chunks, question=prompt)
            if route.use_memory
            else HubAndSpokeMemory(
                embed_model=evaluator.embed_model,
                embed_device=evaluator.embed_device,
                sim_threshold=evaluator.sim_threshold,
            )
        )
        evaluator.last_retrieval = None
        if use_logits:
            response = evaluator.answer_with_memory_mcq(
                memory,
                recent_frames,
                prompt,
                num_options=route.num_options,
                include_no_match_signal=route.include_no_match_signal,
                min_evidence_sim=route.min_evidence_sim,
            )
        else:
            response = evaluator.answer_with_memory(
                memory,
                recent_frames,
                prompt,
                include_no_match_signal=route.include_no_match_signal,
                min_evidence_sim=route.min_evidence_sim,
            )
        elapsed = time.perf_counter() - t0
        retrieval = evaluator.last_retrieval

        metadata = {
            "answer_method": route.method_family,
            "answer_policy": route.policy,
            "answer_num_options": route.num_options,
            "routing_mode": routing,
            "route_use_memory": route.use_memory,
            "route_min_evidence_sim": route.min_evidence_sim,
            # Agreement audit: what the old task-label rule would have decided.
            "route_use_memory_task_label": anno.get("task") in BACKWARD_TASKS,
            **(route.gate.as_metadata() if route.gate is not None else {}),
            "decode_backend": decode_backend,
            "generate_time": elapsed,
            "num_hist_chunks": len(hist_chunks),
            "num_recent_frames": len(recent_frames),
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
    except Exception:
        logger.exception(
            "Sample failed: id=%s task=%s video=%s",
            anno.get("id"), anno.get("task"), video_path,
        )
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
    gate_strict_sim: float = DEFAULT_GATE_STRICT_SIM,
) -> dict[str, Any]:
    video_path = os.path.join(chunked_dir, f"{anno['id']}.mp4")
    prompt = build_prompt(anno["task"], anno)
    response, metadata = _build_and_answer(
        anno,
        video_path,
        evaluator,
        chunk_duration,
        fps,
        recent_frames_only,
        prompt,
        use_logits=use_logits,
        routing=routing,
        gate_strict_sim=gate_strict_sim,
    )
    return {
        "id": anno["id"],
        "video": anno["video"],
        "task": anno["task"],
        "question": anno["question"],
        "response": response,
        "ground_truth": chr(65 + anno["gt"]),
        **metadata,
    }


# ---------------------------------------------------------------------------
# Forward readouts: shared frame-sampling helpers.
#
# Also imported by evaluate_ovo_ablations.py for its own REC methods.
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
    if timestamps and len(timestamps) == len(frames):
        return [(float(ts), frame) for ts, frame in zip(timestamps, frames)]
    span = max(float(chunk.end_time) - float(chunk.start_time), 1e-6)
    count = len(frames)
    return [
        (float(chunk.start_time) + span * (idx + 0.5) / count, frame)
        for idx, frame in enumerate(frames)
    ]


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
        force_first_no: bool = True,
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
        crr_force_first_no: bool = True,
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

            recent_frames = [f for c in recent_buffer for f in c.frames]
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hub-and-Spoke full OVO-Bench evaluation"
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--anno_path",
        default="/data/linzhao/vlm/data/ovo_bench/ovo_bench_new.json",
    )
    parser.add_argument(
        "--chunked_dir",
        default="/data/linzhao/vlm/data/ovo_bench/chunked_videos",
    )
    parser.add_argument("--result_dir", default="results/hub_and_spoke_ovo")
    parser.add_argument("--recent_frames_only", type=int, default=4)
    parser.add_argument("--chunk_duration", type=float, default=1.0)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_qa_tokens", type=int, default=256)
    parser.add_argument(
        "--attn_implementation",
        default="flash_attention_2",
        help="Attention backend for the VLM (e.g. flash_attention_2, sdpa).",
    )
    parser.add_argument("--extract_every_n_chunks", type=int, default=1)
    parser.add_argument("--max_extraction_chunks", type=int, default=20)
    parser.add_argument("--sim_threshold", type=float, default=0.25)
    parser.add_argument(
        "--routing",
        choices=ROUTING_CHOICES,
        default=ROUTING_KEYWORD,
        help=(
            "How the memory/recent-window decision is made. 'keyword' (default) "
            "gates on question text only, as described in the paper. 'task_label' "
            "reads the OVO task annotation (EPM/ASI/HLD vs Real-Time tasks) and is "
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
        default=DEFAULT_EMBED_MODEL,
        help="Sentence-transformer for node/query embeddings.",
    )
    parser.add_argument("--count_question_max_chunks", type=int, default=0)
    parser.add_argument(
        "--caption_batch_size",
        type=int,
        default=0,
        help=(
            "History caption batch size for the Hub-and-Spoke backward/"
            "realtime path. Keep 0 for the baseline HLD/EPM/ASI behavior; "
            ">0 is faster but can change memory captions and scores."
        ),
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
        help=(
            "hub_spoke: dhsm/hub_and_spoke.py, the non-provenance variant. "
            "flat_caption: ablation storing each chunk's whole caption as one "
            "retrieval unit (same captions and retrieval hyperparameters, no "
            "hub-and-spoke organization). "
            "incremental: dhsm/hub_and_spoke_incremental.py, the provenance-"
            "tracking variant implementing Algorithm 2. Note that the "
            "backward/realtime path only calls update(), so this does not "
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
        "--use_logits",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rec_interval_frames", type=int, default=16)
    parser.add_argument("--rec_interval_context_seconds", type=float, default=1.0)
    parser.add_argument("--rec_interval_max_delta", type=int, default=2)
    parser.add_argument("--crr_interval_frames", type=int, default=16)
    parser.add_argument("--crr_interval_context_seconds", type=float, default=1.0)
    parser.add_argument(
        "--crr_no_force_first_no",
        action="store_true",
        help="Do not force the first CRR cutoff to No.",
    )
    args = parser.parse_args()
    # --routing keyword implies the incremental (Algorithm 2) memory unless
    # --memory_mode was given explicitly.
    args.memory_mode = resolve_memory_mode(args.routing, args.memory_mode)

    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=24))]
    )

    import random
    with open(args.anno_path) as handle:
        annotations = json.load(handle)

    active_splits = {s.strip().lower() for s in args.splits.split(",") if s.strip()}
    unknown_splits = active_splits - {"backward", "realtime", "forward"}
    if unknown_splits:
        raise SystemExit(f"Unknown --splits entries: {sorted(unknown_splits)}")

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
        f"top_k={format_top_k(args.top_k)}  "
        f"routing={args.routing}  "
        f"memory_mode={args.memory_mode}  "
        f"gate_strict_sim={args.gate_strict_sim}  "
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
    if os.path.exists(done_marker):
        os.remove(done_marker)
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
                gate_strict_sim=args.gate_strict_sim,
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
                gate_strict_sim=args.gate_strict_sim,
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


'''
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
  experiments/evaluate_ovo.py \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --anno_path /data/xinru/dataset/data/ovo_bench/ovo_bench_new.json \
  --chunked_dir /data/xinru/dataset/data/ovo_bench/chunked_videos \
  --result_dir results/ovo/qwen2.5_4f_batch4_dynamic \
  --recent_frames_only 4 \
  --chunk_duration 1.0 \
  --fps 1.0 \
  --max_qa_tokens 256 \
  --extract_every_n_chunks 1 \
  --max_extraction_chunks 20 \
  --sim_threshold 0.25 \
  --top_k dynamic \
  --use_logits \
  --caption_batch_size 4 \
  2>&1 | tee results/ovo/qwen2.5_4f_batch_4_dynamic.log

'''
