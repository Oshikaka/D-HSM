"""OVO-Bench forward-readout ablations, kept for paper reproducibility.

``evaluate_ovo.py`` runs only the two counting/tracking methods that made it
into the paper: REC via interval-delta counting and CRR via interval-evidence
tracking. This file holds the other methods that were tried along the way,
and imports the frame-sampling helpers it shares with them from
``evaluate_ovo.py`` (never the other way around):

REC (``RecAblationEvaluator``):
  storyboard       Uniformly sample prefix frames, ask the VLM to count
                    complete repetitions directly.
  segmented        Split the prefix into temporal segments, count each
                    separately, and sum the segment counts.
  contact_sheet     Compress sampled frames into one chronological grid image
                    and count from that visual memory image.
  delta             Online counter: update a compact count/event state from
                    only the latest visual window as the stream advances.
  event             Classify each prefix chunk as PEAK / non-peak / none,
                    cluster consecutive PEAK chunks, and return the cluster
                    count.
  hybrid            Use the event-cluster count when it finds events; fall
                    back to storyboard when the detector finds no event.

CRR (``CrrAblationEvaluator``):
  tail              Caption-tail + latest-frame answerability readout.
  tail_mono         Same as ``tail``, followed by monotonic carry-forward:
                    once a prefix is answerable, later prefixes stay so.

Run with ``evaluate_ovo_ablations.py rec ...`` or
``evaluate_ovo_ablations.py crr ...``.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("NCCL_TIMEOUT", "7200")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "0")
os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "86400")

from accelerate import Accelerator, InitProcessGroupKwargs
from PIL import Image, ImageDraw
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dhsm import shard_io
from dhsm.video_qa import decode_video_to_chunks_qwen
from dhsm.video_qa_qwen3 import RecentWindowQAModel

from experiments.evaluate_ovo import (
    chunk_frames_with_timestamps,
    clamp_count,
    fmt_time,
    parse_int_response,
    row_key,
    uniform_indices,
)
from experiments.ovo_bench import score_count, score_yes_no

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
for _noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _parse_methods(methods: Iterable[str], pool: tuple[str, ...]) -> tuple[str, ...]:
    parsed = tuple(m.strip().lower() for m in methods if m.strip())
    unknown = set(parsed) - set(pool)
    if unknown:
        raise ValueError(f"Unknown method(s): {sorted(unknown)}")
    return parsed or pool


# ---------------------------------------------------------------------------
# Checkpointing: generic single-task variants of the shard_io wrappers.
# ---------------------------------------------------------------------------

def _load_checkpoint_rows(path: str, task: str) -> tuple[list[dict[str, Any]], set[str]]:
    rows, done_keys = shard_io.resume(path, row_key)
    return [row for row in rows if row.get("task") == task], done_keys


def _merge_checkpoint_rows(result_dir: str, n_procs: int, task: str) -> list[dict[str, Any]]:
    merged = shard_io.merge_shards(result_dir, n_procs, key_fn=row_key)
    return [row for row in merged if row.get("task") == task]


# ---------------------------------------------------------------------------
# REC ablations
# ---------------------------------------------------------------------------

REC_METHODS = ("storyboard", "segmented", "contact_sheet", "delta", "event", "hybrid")


def _make_storyboard_prompt(activity: str, timestamps: list[float]) -> str:
    ts_text = ", ".join(fmt_time(t) for t in timestamps)
    return (
        "You are given sampled frames from a video prefix in chronological "
        "order. The frames may skip some moments, but the order is preserved.\n\n"
        f"Target activity: {activity}\n"
        f"Frame timestamps: {ts_text}\n\n"
        "Count how many COMPLETE instances of the target activity have already "
        "happened by the final frame.\n"
        "Count one full repetition as one. Do not count preparation, waiting, "
        "resetting, or aftermath as an extra repetition. If the same repetition "
        "appears across multiple sampled frames, count it only once.\n\n"
        "Return only one integer."
    )


def _make_segment_prompt(
    activity: str,
    timestamps: list[float],
    segment_idx: int,
    num_segments: int,
) -> str:
    ts_text = ", ".join(fmt_time(t) for t in timestamps)
    return (
        "You are given sampled frames from ONE temporal segment of a longer "
        "video prefix. The frames are in chronological order.\n\n"
        f"Target activity: {activity}\n"
        f"Segment: {segment_idx + 1} of {num_segments}\n"
        f"Frame timestamps: {ts_text}\n\n"
        "Count how many COMPLETE instances of the target activity have their "
        "defining/peak moment inside this segment.\n"
        "Do not count preparation, waiting, resetting, or aftermath. If the "
        "same repetition appears in multiple sampled frames in this segment, "
        "count it only once. If a repetition is only partially visible and the "
        "defining moment is not visible, do not count it.\n\n"
        "Return only one integer."
    )


def _make_contact_sheet_prompt(activity: str, timestamps: list[float]) -> str:
    ts_text = ", ".join(fmt_time(t) for t in timestamps)
    return (
        "You are given ONE contact-sheet image. It contains sampled frames "
        "from a video prefix in chronological order, arranged left-to-right "
        "and top-to-bottom. Each cell is labeled with its frame index and "
        "timestamp.\n\n"
        f"Target activity: {activity}\n"
        f"Cell timestamps: {ts_text}\n\n"
        "Count how many COMPLETE instances of the target activity have already "
        "happened by the final cell. Count one full repetition as one. Do not "
        "count preparation, waiting, resetting, or aftermath as an extra "
        "repetition. If the same repetition appears in multiple cells, count "
        "it only once.\n\n"
        "Return only one integer."
    )


def _make_delta_prompt(
    activity: str,
    current_count: int,
    last_event_time: float | None,
    current_time: float,
) -> str:
    last_text = "none" if last_event_time is None else fmt_time(last_event_time)
    return (
        "You are updating an online repetition counter from the latest video "
        "frames only.\n\n"
        f"Target activity: {activity}\n"
        f"Current count so far: {current_count}\n"
        f"Last counted event peak time: {last_text}\n"
        f"Current timestamp: {fmt_time(current_time)}\n\n"
        "Look only at the latest frames. Decide whether a NEW complete "
        "instance of the target activity reaches its defining/peak moment "
        "after the last counted event. Do not count preparation, waiting, "
        "resetting, continuation, or aftermath.\n\n"
        "Options:\n"
        "A. 0 new complete instances\n"
        "B. 1 new complete instance\n"
        "C. 2 new complete instances\n"
        "D. unclear, count 0 new instances\n\n"
        "Return only the letter A, B, C, or D."
    )


def _make_phase_prompt(activity: str, ts: float) -> str:
    return (
        "You are labeling the current short video evidence for repetition "
        "counting. Use the images as a tiny chronological clip ending at the "
        "current timestamp.\n\n"
        f"Target activity: {activity}\n"
        f"Current timestamp: {fmt_time(ts)}\n\n"
        "Choose the best label:\n"
        "A. PEAK_COMPLETE - the defining/peak moment of one complete instance "
        "of the target activity is visible now.\n"
        "B. ACTIVE_NONPEAK - the target activity is related or underway, but "
        "this is preparation, continuation, aftermath, or not the countable "
        "peak moment.\n"
        "C. RESET_IDLE - the scene is between attempts or the actor is waiting, "
        "walking, resetting, or holding position.\n"
        "D. OTHER_ACTION - the visible action is not the target activity.\n"
        "E. UNCLEAR - there is not enough visual evidence.\n\n"
        "Return only the letter A, B, C, D, or E."
    )


def _chunk_mid_frame(chunk) -> tuple[float, Any] | None:
    if not getattr(chunk, "frames", None):
        return None
    mid = len(chunk.frames) // 2
    ts = (chunk.start_time + chunk.end_time) / 2.0
    return ts, chunk.frames[mid]


def _sample_storyboard(
    chunks: list,
    max_frames: int,
    tail_frames: int,
) -> tuple[list[Any], list[float]]:
    reps: list[tuple[int, float, Any]] = []
    for idx, chunk in enumerate(chunks):
        for local_idx, (ts, frame) in enumerate(chunk_frames_with_timestamps(chunk)):
            # The first field is only for chronological sorting / tail
            # selection, so a fractional within-chunk id is enough.
            reps.append((idx * 1000 + local_idx, ts, frame))

    if not reps:
        return [], []

    max_frames = max(1, int(max_frames))
    tail_frames = max(0, int(tail_frames))
    if len(reps) <= max_frames:
        selected = reps
    else:
        tail = reps[-min(tail_frames, max_frames):] if tail_frames else []
        tail_ids = {idx for idx, _, _ in tail}
        budget = max_frames - len(tail)
        head = [rep for rep in reps if rep[0] not in tail_ids]
        head_selected = [head[i] for i in uniform_indices(len(head), budget)]
        selected = sorted(head_selected + tail, key=lambda rep: rep[0])

    frames = [frame for _, _, frame in selected]
    timestamps = [ts for _, ts, _ in selected]
    return frames, timestamps


def _make_contact_sheet(
    frames: list[Any],
    timestamps: list[float],
    cols: int,
    cell_size: int,
) -> Image.Image | None:
    if not frames:
        return None
    cols = max(1, int(cols))
    cell_size = max(64, int(cell_size))
    rows = math.ceil(len(frames) / cols)
    sheet = Image.new("RGB", (cols * cell_size, rows * cell_size), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)

    for idx, (frame, ts) in enumerate(zip(frames, timestamps)):
        row, col = divmod(idx, cols)
        x = col * cell_size
        y = row * cell_size
        img = frame.convert("RGB") if hasattr(frame, "convert") else Image.fromarray(frame)
        img.thumbnail((cell_size, cell_size))
        px = x + (cell_size - img.width) // 2
        py = y + (cell_size - img.height) // 2
        sheet.paste(img, (px, py))

        label = f"{idx + 1} {fmt_time(ts)}"
        # A small dark backing keeps labels visible on bright frames.
        draw.rectangle((x + 2, y + 2, x + 92, y + 18), fill=(0, 0, 0))
        draw.text((x + 5, y + 4), label, fill=(255, 255, 255))
        draw.rectangle(
            (x, y, x + cell_size - 1, y + cell_size - 1),
            outline=(180, 180, 180),
        )

    return sheet


def _last_frames_from_chunks(chunks: list, max_frames: int) -> tuple[list[Any], list[float]]:
    items: list[tuple[float, Any]] = []
    for chunk in chunks:
        items.extend(chunk_frames_with_timestamps(chunk))
    items = sorted(items, key=lambda item: item[0])[-max(1, int(max_frames)):]
    return [frame for _, frame in items], [ts for ts, _ in items]


def _phase_context_frames(chunks: list, idx: int, context_chunks: int) -> list[Any]:
    context_chunks = max(1, int(context_chunks))
    start = max(0, idx - context_chunks + 1)
    frames: list[Any] = []
    for chunk in chunks[start: idx + 1]:
        item = _chunk_mid_frame(chunk)
        if item is not None:
            frames.append(item[1])
    return frames


def _cluster_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "start": rows[0]["timestamp"],
        "end": rows[-1]["timestamp"],
        "peak_timestamps": [row["timestamp"] for row in rows],
        "num_peak_chunks": len(rows),
    }


def _cluster_peak_events(
    phase_rows: list[dict[str, Any]],
    merge_gap_seconds: float,
) -> list[dict[str, Any]]:
    peaks = [row for row in phase_rows if row.get("label") == "A"]
    if not peaks:
        return []

    clusters: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = [peaks[0]]
    for row in peaks[1:]:
        prev = current[-1]
        gap = float(row["timestamp"]) - float(prev["timestamp"])
        if gap <= merge_gap_seconds:
            current.append(row)
        else:
            clusters.append(_cluster_summary(current))
            current = [row]
    clusters.append(_cluster_summary(current))
    return clusters


class RecAblationEvaluator:
    """Runs a chosen subset of the REC ablation counting methods."""

    def __init__(
        self,
        qa_model: RecentWindowQAModel,
        methods: Iterable[str],
        storyboard_frames: int = 48,
        storyboard_tail_frames: int = 4,
        segmented_segments: int = 6,
        segmented_frames_per_segment: int = 16,
        contact_sheet_frames: int = 48,
        contact_sheet_tail_frames: int = 4,
        contact_sheet_cols: int = 8,
        contact_sheet_cell_size: int = 192,
        delta_window_frames: int = 4,
        delta_stride: int = 1,
        delta_cooldown_seconds: float = 1.0,
        phase_context_chunks: int = 2,
        event_stride: int = 1,
        event_merge_gap_seconds: float = 2.5,
        max_count_cap: int = 0,
    ) -> None:
        self.qa = qa_model
        self.methods = tuple(methods)
        self.storyboard_frames = max(1, int(storyboard_frames))
        self.storyboard_tail_frames = max(0, int(storyboard_tail_frames))
        self.segmented_segments = max(1, int(segmented_segments))
        self.segmented_frames_per_segment = max(1, int(segmented_frames_per_segment))
        self.contact_sheet_frames = max(1, int(contact_sheet_frames))
        self.contact_sheet_tail_frames = max(0, int(contact_sheet_tail_frames))
        self.contact_sheet_cols = max(1, int(contact_sheet_cols))
        self.contact_sheet_cell_size = max(64, int(contact_sheet_cell_size))
        self.delta_window_frames = max(1, int(delta_window_frames))
        self.delta_stride = max(1, int(delta_stride))
        self.delta_cooldown_seconds = max(0.0, float(delta_cooldown_seconds))
        self.phase_context_chunks = max(1, int(phase_context_chunks))
        self.event_stride = max(1, int(event_stride))
        self.event_merge_gap_seconds = float(event_merge_gap_seconds)
        self.max_count_cap = max(0, int(max_count_cap))

    def _storyboard_count(self, activity: str, prefix_chunks: list) -> dict[str, Any]:
        frames, timestamps = _sample_storyboard(
            prefix_chunks,
            max_frames=self.storyboard_frames,
            tail_frames=self.storyboard_tail_frames,
        )
        if not frames:
            return {"response": "0", "raw_response": None, "num_storyboard_frames": 0}

        prompt = _make_storyboard_prompt(activity, timestamps)
        t0 = time.perf_counter()
        raw = self.qa.generate_from_frames(frames, prompt)
        parsed = clamp_count(parse_int_response(raw), self.max_count_cap)
        elapsed = time.perf_counter() - t0
        response = "0" if parsed is None else str(parsed)
        return {
            "response": response,
            "raw_response": raw,
            "generate_time": elapsed,
            "num_storyboard_frames": len(frames),
            "storyboard_first_ts": timestamps[0] if timestamps else None,
            "storyboard_last_ts": timestamps[-1] if timestamps else None,
        }

    def _segmented_count(self, activity: str, prefix_chunks: list) -> dict[str, Any]:
        if not prefix_chunks:
            return {
                "response": "0",
                "raw_segment_responses": [],
                "segment_counts": [],
                "num_segments_used": 0,
            }

        n_segments = min(self.segmented_segments, len(prefix_chunks))
        chunk_indices = uniform_indices(len(prefix_chunks) + 1, n_segments + 1)
        boundaries = sorted(set(chunk_indices))
        if boundaries[0] != 0:
            boundaries.insert(0, 0)
        if boundaries[-1] != len(prefix_chunks):
            boundaries.append(len(prefix_chunks))

        total = 0
        raw_responses: list[str | None] = []
        segment_counts: list[int] = []
        segment_meta: list[dict[str, Any]] = []
        t0 = time.perf_counter()
        usable_segments = 0

        for start, end in zip(boundaries, boundaries[1:]):
            if end <= start:
                continue
            segment_chunks = prefix_chunks[start:end]
            frames, timestamps = _sample_storyboard(
                segment_chunks,
                max_frames=self.segmented_frames_per_segment,
                tail_frames=0,
            )
            if not frames:
                continue
            usable_segments += 1
            prompt = _make_segment_prompt(
                activity,
                timestamps,
                segment_idx=usable_segments - 1,
                num_segments=max(1, len(boundaries) - 1),
            )
            raw = self.qa.generate_from_frames(frames, prompt)
            count = clamp_count(parse_int_response(raw), self.max_count_cap) or 0
            total += count
            raw_responses.append(raw)
            segment_counts.append(count)
            segment_meta.append(
                {
                    "chunk_start": start,
                    "chunk_end": end,
                    "first_ts": timestamps[0],
                    "last_ts": timestamps[-1],
                    "num_frames": len(frames),
                    "count": count,
                }
            )

        total = clamp_count(total, self.max_count_cap) or 0
        return {
            "response": str(total),
            "raw_segment_responses": raw_responses,
            "segment_counts": segment_counts,
            "segment_meta": segment_meta,
            "generate_time": time.perf_counter() - t0,
            "num_segments_used": usable_segments,
        }

    def _contact_sheet_count(self, activity: str, prefix_chunks: list) -> dict[str, Any]:
        frames, timestamps = _sample_storyboard(
            prefix_chunks,
            max_frames=self.contact_sheet_frames,
            tail_frames=self.contact_sheet_tail_frames,
        )
        if not frames:
            return {
                "response": "0",
                "raw_response": None,
                "num_contact_frames": 0,
                "num_contact_images": 0,
            }

        sheet = _make_contact_sheet(
            frames,
            timestamps,
            cols=self.contact_sheet_cols,
            cell_size=self.contact_sheet_cell_size,
        )
        if sheet is None:
            return {
                "response": "0",
                "raw_response": None,
                "num_contact_frames": 0,
                "num_contact_images": 0,
            }

        prompt = _make_contact_sheet_prompt(activity, timestamps)
        t0 = time.perf_counter()
        raw = self.qa.generate_from_frames([sheet], prompt)
        parsed = clamp_count(parse_int_response(raw), self.max_count_cap)
        elapsed = time.perf_counter() - t0
        response = "0" if parsed is None else str(parsed)
        return {
            "response": response,
            "raw_response": raw,
            "generate_time": elapsed,
            "num_contact_frames": len(frames),
            "num_contact_images": 1,
            "contact_sheet_size": list(sheet.size),
            "contact_first_ts": timestamps[0] if timestamps else None,
            "contact_last_ts": timestamps[-1] if timestamps else None,
        }

    def _delta_stream(self, activity: str, chunks: list) -> list[dict[str, Any]]:
        count = 0
        events: list[dict[str, Any]] = []
        last_event_time: float | None = None
        label_counts = {label: 0 for label in ("A", "B", "C", "D")}
        states: list[dict[str, Any]] = []

        for idx, chunk in enumerate(chunks):
            if idx % self.delta_stride != 0:
                states.append(
                    {
                        "end_time": chunk.end_time,
                        "count": count,
                        "events": list(events),
                        "last_event_time": last_event_time,
                        "label_counts": dict(label_counts),
                        "num_delta_steps": sum(label_counts.values()),
                    }
                )
                continue

            frames, timestamps = _last_frames_from_chunks(
                chunks[: idx + 1],
                max_frames=self.delta_window_frames,
            )
            current_time = timestamps[-1] if timestamps else chunk.end_time
            if not frames:
                label = "D"
            else:
                prompt = _make_delta_prompt(
                    activity,
                    current_count=count,
                    last_event_time=last_event_time,
                    current_time=current_time,
                )
                label = self.qa.score_mcq_from_frames(frames, prompt, num_options=4)
                if label not in label_counts:
                    label = "D"
            label_counts[label] += 1

            delta = {"A": 0, "B": 1, "C": 2, "D": 0}.get(label, 0)
            suppressed = False
            if (
                delta > 0
                and last_event_time is not None
                and current_time - last_event_time <= self.delta_cooldown_seconds
            ):
                delta = 0
                suppressed = True

            if delta > 0:
                for _ in range(delta):
                    events.append(
                        {
                            "event_id": len(events) + 1,
                            "peak_time": current_time,
                            "decision": label,
                        }
                    )
                count += delta
                count = clamp_count(count, self.max_count_cap) or 0
                last_event_time = current_time

            states.append(
                {
                    "end_time": chunk.end_time,
                    "count": count,
                    "events": list(events),
                    "last_event_time": last_event_time,
                    "label_counts": dict(label_counts),
                    "num_delta_steps": sum(label_counts.values()),
                    "last_delta_label": label,
                    "last_delta_suppressed": suppressed,
                }
            )

        return states

    def _phase_rows(self, activity: str, chunks: list) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for idx, chunk in enumerate(chunks):
            if idx % self.event_stride != 0:
                continue
            item = _chunk_mid_frame(chunk)
            if item is None:
                continue
            ts, _ = item
            frames = _phase_context_frames(chunks, idx, self.phase_context_chunks)
            if not frames:
                continue
            prompt = _make_phase_prompt(activity, ts)
            label = self.qa.score_mcq_from_frames(frames, prompt, num_options=5)
            rows.append(
                {
                    "chunk_idx": idx,
                    "timestamp": ts,
                    "start_time": chunk.start_time,
                    "end_time": chunk.end_time,
                    "label": label,
                    "num_phase_frames": len(frames),
                }
            )
        return rows

    def evaluate(
        self,
        anno: dict[str, Any],
        chunks: list,
        sub_tests: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        activity = anno["activity"]
        max_cutoff = max(ti["realtime"] for _, ti in sub_tests)
        eligible_chunks = [c for c in chunks if c.end_time <= max_cutoff]

        phase_rows: list[dict[str, Any]] = []
        phase_elapsed = 0.0
        if "event" in self.methods or "hybrid" in self.methods:
            t0 = time.perf_counter()
            phase_rows = self._phase_rows(activity, eligible_chunks)
            phase_elapsed = time.perf_counter() - t0

        delta_states: list[dict[str, Any]] = []
        delta_elapsed = 0.0
        if "delta" in self.methods:
            t0 = time.perf_counter()
            delta_states = self._delta_stream(activity, eligible_chunks)
            delta_elapsed = time.perf_counter() - t0

        results: list[tuple[int, dict[str, Any]]] = []
        prev_counts: dict[str, int | None] = {m: None for m in self.methods}
        for orig_idx, ti in sub_tests:
            cutoff = ti["realtime"]
            prefix_chunks = [c for c in chunks if c.end_time <= cutoff]
            update: dict[str, Any] = {}

            story_count: int | None = None
            if "storyboard" in self.methods or "hybrid" in self.methods:
                story = self._storyboard_count(activity, prefix_chunks)
                story_count = parse_int_response(story["response"])
                if "storyboard" in self.methods:
                    update["storyboard_response"] = story["response"]
                    update["storyboard_raw_response"] = story["raw_response"]
                    update["storyboard_generate_time"] = story["generate_time"]
                    update["storyboard_num_frames"] = story["num_storyboard_frames"]
                    update["storyboard_first_ts"] = story["storyboard_first_ts"]
                    update["storyboard_last_ts"] = story["storyboard_last_ts"]

            if "segmented" in self.methods:
                segmented = self._segmented_count(activity, prefix_chunks)
                update["segmented_response"] = segmented["response"]
                update["segmented_raw_segment_responses"] = segmented[
                    "raw_segment_responses"
                ]
                update["segmented_segment_counts"] = segmented["segment_counts"]
                update["segmented_segment_meta"] = segmented["segment_meta"]
                update["segmented_generate_time"] = segmented["generate_time"]
                update["segmented_num_segments_used"] = segmented["num_segments_used"]

            if "contact_sheet" in self.methods:
                contact = self._contact_sheet_count(activity, prefix_chunks)
                update["contact_sheet_response"] = contact["response"]
                update["contact_sheet_raw_response"] = contact["raw_response"]
                update["contact_sheet_generate_time"] = contact["generate_time"]
                update["contact_sheet_num_contact_frames"] = contact["num_contact_frames"]
                update["contact_sheet_num_contact_images"] = contact["num_contact_images"]
                update["contact_sheet_size"] = contact.get("contact_sheet_size")
                update["contact_sheet_first_ts"] = contact.get("contact_first_ts")
                update["contact_sheet_last_ts"] = contact.get("contact_last_ts")

            if "delta" in self.methods:
                state = next(
                    (s for s in reversed(delta_states) if s["end_time"] <= cutoff),
                    None,
                )
                if state is None:
                    update["delta_response"] = "0"
                    update["delta_events"] = []
                    update["delta_label_counts"] = {label: 0 for label in ("A", "B", "C", "D")}
                    update["delta_num_steps"] = 0
                    update["delta_stream_time_total"] = delta_elapsed
                else:
                    update["delta_response"] = str(state["count"])
                    update["delta_events"] = state["events"]
                    update["delta_last_event_time"] = state["last_event_time"]
                    update["delta_label_counts"] = state["label_counts"]
                    update["delta_num_steps"] = state["num_delta_steps"]
                    update["delta_last_label"] = state.get("last_delta_label")
                    update["delta_last_suppressed"] = state.get("last_delta_suppressed")
                    update["delta_stream_time_total"] = delta_elapsed

            event_count: int | None = None
            clusters: list[dict[str, Any]] = []
            if "event" in self.methods or "hybrid" in self.methods:
                rows_now = [r for r in phase_rows if r["end_time"] <= cutoff]
                clusters = _cluster_peak_events(rows_now, self.event_merge_gap_seconds)
                event_count = clamp_count(len(clusters), self.max_count_cap)
                update["event_num_phase_rows"] = len(rows_now)
                update["event_num_all_phase_rows"] = len(phase_rows)
                update["event_phase_time_total"] = phase_elapsed
                update["event_label_counts"] = {
                    label: sum(1 for r in rows_now if r["label"] == label)
                    for label in ("A", "B", "C", "D", "E")
                }
                update["event_clusters"] = clusters
                if "event" in self.methods:
                    update["event_response"] = str(event_count or 0)

            if "hybrid" in self.methods:
                if event_count and event_count > 0:
                    hybrid_count = event_count
                    hybrid_source = "event"
                elif story_count is not None:
                    hybrid_count = clamp_count(story_count, self.max_count_cap)
                    hybrid_source = "storyboard_fallback"
                else:
                    hybrid_count = 0
                    hybrid_source = "zero_fallback"
                update["hybrid_response"] = str(hybrid_count or 0)
                update["hybrid_source"] = hybrid_source

            for method in self.methods:
                key = f"{method}_response"
                current = parse_int_response(update.get(key))
                prev = prev_counts.get(method)
                if current is None:
                    continue
                if prev is not None and current < prev:
                    update[f"{method}_clamped_from"] = str(current)
                    current = prev
                    update[key] = str(current)
                prev_counts[method] = current

            primary = self.methods[0]
            update["response"] = update.get(f"{primary}_response")
            update["primary_method"] = primary
            update["realtime_cutoff"] = cutoff
            update["num_prefix_chunks"] = len(prefix_chunks)
            results.append((orig_idx, update))

        for chunk in chunks:
            chunk.frames = []
        return results


def _evaluate_rec(
    anno: dict[str, Any],
    chunked_dir: str,
    evaluator: RecAblationEvaluator,
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
        logger.exception("decode failed: id=%s video=%s", anno.get("id"), longest_path)
        return result_anno
    if not chunks:
        return result_anno

    sorted_sub = sorted(enumerate(test_info), key=lambda kv: kv[1]["realtime"])
    try:
        rows = evaluator.evaluate(anno, chunks, sorted_sub)
    except Exception:
        logger.exception("REC sample failed: id=%s", anno.get("id"))
        return result_anno

    for orig_idx, update in rows:
        update = dict(update)
        update["decode_backend"] = decode_backend
        test_info[orig_idx].update(update)
    return result_anno


def calculate_rec_scores(
    forward: list[dict[str, Any]],
    methods: Iterable[str],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for method in methods:
        vals: list[int] = []
        response_key = f"{method}_response"
        for result in forward:
            for item in result.get("test_info", []):
                vals.append(score_count(item.get(response_key), item["count"]))
        if vals:
            summary[method] = {
                "correct": sum(vals),
                "total": len(vals),
                "accuracy": 100.0 * sum(vals) / len(vals),
            }
    return summary


def print_rec_results(label: str, forward: list[dict[str, Any]], methods: Iterable[str]) -> None:
    summary = calculate_rec_scores(forward, methods)
    print("\n" + "=" * 60)
    print(f"OVO-Bench REC Ablation Results ({label})")
    print("=" * 60)
    for method in methods:
        stats = summary.get(method)
        if not stats:
            continue
        print(
            f"  REC-{method}: {stats['accuracy']:.2f}% "
            f"({stats['correct']}/{stats['total']})"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# CRR ablations
# ---------------------------------------------------------------------------

CRR_METHODS = ("tail", "tail_mono")

PLAIN_CAPTION_PROMPT = (
    "Describe what is happening in this video frame in one or two sentences. "
    "Be specific about people, objects, and actions visible. Be concise."
)


def _tail_ab_to_yes_no(answer: str | None) -> str:
    if answer is None:
        return "No"
    text = str(answer).strip().upper()
    if re.search(r"\bA\b", text) or "YES" in text or text == "Y":
        return "Yes"
    return "No"


def _make_tail_prompt(question: str, caption_context: str) -> str:
    context = f"{caption_context}\n\n" if caption_context else ""
    return (
        f"{context}"
        "You are watching a video prefix ending at the latest frames. Decide "
        "whether the observed visual content up to now provides enough evidence "
        "to answer the question below.\n\n"
        f"Question: {question}\n\n"
        "Answer Yes if the visual answer evidence has already appeared. Answer "
        "No if the needed event, object, person, place, or outcome has not "
        "appeared yet or remains uncertain.\n\n"
        "Options:\n"
        "A. Yes\n"
        "B. No\n\n"
        "Respond only with A or B."
    )


class CrrAblationEvaluator:
    """Runs a chosen subset of the CRR tail-caption ablation methods."""

    def __init__(
        self,
        qa_model: RecentWindowQAModel,
        methods: Iterable[str],
        recent_frames: int = 4,
        crr_caption_tail_seconds: float = 45.0,
        crr_caption_tail_lines: int = 48,
        crr_caption_every_n_chunks: int = 1,
    ) -> None:
        self.qa = qa_model
        self.methods = tuple(methods)
        self.recent_frames = max(1, int(recent_frames))
        self.crr_caption_tail_seconds = float(crr_caption_tail_seconds)
        self.crr_caption_tail_lines = max(0, int(crr_caption_tail_lines))
        self.crr_caption_every_n_chunks = max(1, int(crr_caption_every_n_chunks))

    def _caption_plain_chunk(self, chunk) -> tuple[str | None, float]:
        if not getattr(chunk, "frames", None):
            return None, 0.0
        mid = len(chunk.frames) // 2
        frame = chunk.frames[mid]
        ts = (chunk.start_time + chunk.end_time) / 2.0
        caption = self.qa.generate_from_frames([frame], PLAIN_CAPTION_PROMPT)
        return (caption.strip() or None), ts

    def _caption_context(
        self,
        caption_log: list[tuple[float, str]],
        cutoff: float,
    ) -> str:
        if not caption_log or self.crr_caption_tail_lines <= 0:
            return ""
        lower = max(0.0, cutoff - self.crr_caption_tail_seconds)
        rows = [
            (ts, cap)
            for ts, cap in caption_log
            if lower <= ts <= cutoff and cap.strip()
        ]
        if not rows:
            rows = [(ts, cap) for ts, cap in caption_log if ts <= cutoff and cap.strip()]
        rows = rows[-self.crr_caption_tail_lines:]
        if not rows:
            return ""
        lines = ["[Recent chronological video evidence]"]
        for ts, cap in rows:
            lines.append(f"[{fmt_time(ts)}] {cap[:220]}")
        return "\n".join(lines)

    def _score_tail(
        self,
        anno: dict[str, Any],
        chunks: list,
        sub_tests: list[tuple[int, dict[str, Any]]],
    ) -> dict[int, dict[str, Any]]:
        updates: dict[int, dict[str, Any]] = {}
        window = self.recent_frames
        recent_buffer: list = []
        caption_log: list[tuple[float, str]] = []
        captioned_count = 0
        seen_yes = False

        def evict_oldest() -> None:
            nonlocal captioned_count
            old = recent_buffer.pop(0)
            if old.chunk_index % self.crr_caption_every_n_chunks == 0:
                caption, ts = self._caption_plain_chunk(old)
                if caption:
                    caption_log.append((ts, caption))
                    captioned_count += 1
            old.frames = []

        chunk_iter = iter(chunks)
        pending = next(chunk_iter, None)
        for orig_idx, ti in sub_tests:
            cutoff = ti["realtime"]
            while pending is not None and pending.end_time <= cutoff:
                recent_buffer.append(pending)
                while len(recent_buffer) > window:
                    evict_oldest()
                pending = next(chunk_iter, None)

            recent = [f for c in recent_buffer for f in c.frames]
            caption_context = self._caption_context(caption_log, cutoff)
            prompt = _make_tail_prompt(anno["question"], caption_context)
            t0 = time.perf_counter()
            raw = self.qa.score_mcq_from_frames(recent, prompt, num_options=2)
            elapsed = time.perf_counter() - t0
            response = _tail_ab_to_yes_no(raw)
            if response == "Yes":
                seen_yes = True

            updates[orig_idx] = {
                "tail_response": response,
                "tail_raw_choice": raw,
                "tail_mono_response": "Yes" if seen_yes else "No",
                "tail_generate_time": elapsed,
                "tail_num_caption_lines": len(caption_log),
                "tail_num_caption_lines_used": caption_context.count("\n") if caption_context else 0,
                "tail_num_recent_frames": len(recent),
            }

        for chunk in recent_buffer:
            chunk.frames = []
        return updates

    def evaluate(
        self,
        anno: dict[str, Any],
        chunks: list,
        sub_tests: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[int, dict[str, Any]]]:
        updates_by_idx: dict[int, dict[str, Any]] = defaultdict(dict)

        if "tail" in self.methods or "tail_mono" in self.methods:
            tail_updates = self._score_tail(anno, chunks, sub_tests)
            for idx, update in tail_updates.items():
                updates_by_idx[idx].update(update)

        primary = self.methods[0]
        results: list[tuple[int, dict[str, Any]]] = []
        for orig_idx, ti in sub_tests:
            update = dict(updates_by_idx.get(orig_idx, {}))
            update["response"] = update.get(f"{primary}_response")
            update["primary_method"] = primary
            update["realtime_cutoff"] = ti["realtime"]
            results.append((orig_idx, update))

        for chunk in chunks:
            chunk.frames = []
        return results


def _evaluate_crr(
    anno: dict[str, Any],
    chunked_dir: str,
    evaluator: CrrAblationEvaluator,
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
        logger.exception("decode failed: id=%s video=%s", anno.get("id"), longest_path)
        return result_anno
    if not chunks:
        return result_anno

    sorted_sub = sorted(enumerate(test_info), key=lambda kv: kv[1]["realtime"])
    try:
        rows = evaluator.evaluate(anno, chunks, sorted_sub)
    except Exception:
        logger.exception("CRR sample failed: id=%s", anno.get("id"))
        return result_anno

    for orig_idx, update in rows:
        update = dict(update)
        update["decode_backend"] = decode_backend
        test_info[orig_idx].update(update)
    return result_anno


def calculate_crr_scores(
    forward: list[dict[str, Any]],
    methods: Iterable[str],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for method in methods:
        vals: list[int] = []
        response_key = f"{method}_response"
        for result in forward:
            for item in result.get("test_info", []):
                vals.append(score_yes_no(item.get(response_key), item["type"]))
        if vals:
            summary[method] = {
                "correct": sum(vals),
                "total": len(vals),
                "accuracy": 100.0 * sum(vals) / len(vals),
            }
    return summary


def print_crr_results(label: str, forward: list[dict[str, Any]], methods: Iterable[str]) -> None:
    summary = calculate_crr_scores(forward, methods)
    print("\n" + "=" * 60)
    print(f"OVO-Bench CRR Ablation Results ({label})")
    print("=" * 60)
    for method in methods:
        stats = summary.get(method)
        if not stats:
            continue
        print(
            f"  CRR-{method}: {stats['accuracy']:.2f}% "
            f"({stats['correct']}/{stats['total']})"
        )
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI: `evaluate_ovo_ablations.py rec ...` / `evaluate_ovo_ablations.py crr ...`
# ---------------------------------------------------------------------------

def _add_rec_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--model_path", required=True)
    sub.add_argument(
        "--anno_path",
        default="data/ovo_bench/ovo_bench_new.json",
    )
    sub.add_argument(
        "--chunked_dir",
        default="data/ovo_bench/chunked_videos",
    )
    sub.add_argument("--result_dir", default="results/hub_and_spoke_rec_ablation")
    sub.add_argument("--chunk_duration", type=float, default=1.0)
    sub.add_argument("--fps", type=float, default=1.0)
    sub.add_argument("--max_qa_tokens", type=int, default=64)
    sub.add_argument("--methods", nargs="+", default=list(REC_METHODS))
    sub.add_argument("--storyboard_frames", type=int, default=48)
    sub.add_argument("--storyboard_tail_frames", type=int, default=4)
    sub.add_argument("--segmented_segments", type=int, default=6)
    sub.add_argument("--segmented_frames_per_segment", type=int, default=16)
    sub.add_argument("--contact_sheet_frames", type=int, default=48)
    sub.add_argument("--contact_sheet_tail_frames", type=int, default=4)
    sub.add_argument("--contact_sheet_cols", type=int, default=8)
    sub.add_argument("--contact_sheet_cell_size", type=int, default=192)
    sub.add_argument("--delta_window_frames", type=int, default=4)
    sub.add_argument("--delta_stride", type=int, default=1)
    sub.add_argument("--delta_cooldown_seconds", type=float, default=1.0)
    sub.add_argument("--phase_context_chunks", type=int, default=2)
    sub.add_argument("--event_stride", type=int, default=1)
    sub.add_argument("--event_merge_gap_seconds", type=float, default=2.5)
    sub.add_argument(
        "--max_count_cap",
        type=int,
        default=0,
        help="Optional cap for predicted counts; 0 disables capping.",
    )
    sub.add_argument("--max_samples", type=int, default=None)


def _run_rec(args: argparse.Namespace) -> None:
    methods = _parse_methods(args.methods, REC_METHODS)
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=24))]
    )

    import random
    with open(args.anno_path) as handle:
        annotations = json.load(handle)

    forward_anno = [a for a in annotations if a.get("task") == "REC"]
    random.seed(42)
    random.shuffle(forward_anno)
    if args.max_samples is not None:
        forward_anno = forward_anno[: args.max_samples]

    accelerator.print(f"\n{'=' * 60}")
    accelerator.print("Hub-and-Spoke REC Forward Ablations")
    accelerator.print(f"{'=' * 60}")
    accelerator.print(f"REC items: {len(forward_anno)}  methods={','.join(methods)}")
    accelerator.print(
        f"storyboard_frames={args.storyboard_frames}  "
        f"segmented={args.segmented_segments}x"
        f"{args.segmented_frames_per_segment}  "
        f"contact_sheet={args.contact_sheet_frames}/"
        f"{args.contact_sheet_cols}cols  "
        f"delta_window={args.delta_window_frames}  "
        f"phase_context_chunks={args.phase_context_chunks}  "
        f"event_stride={args.event_stride}  "
        f"event_merge_gap={args.event_merge_gap_seconds}s"
    )
    accelerator.print(f"{'=' * 60}\n")

    qa_model = RecentWindowQAModel(
        model_name=args.model_path,
        device=accelerator.device,
        max_new_tokens=args.max_qa_tokens,
    )
    evaluator = RecAblationEvaluator(
        qa_model=qa_model,
        methods=methods,
        storyboard_frames=args.storyboard_frames,
        storyboard_tail_frames=args.storyboard_tail_frames,
        segmented_segments=args.segmented_segments,
        segmented_frames_per_segment=args.segmented_frames_per_segment,
        contact_sheet_frames=args.contact_sheet_frames,
        contact_sheet_tail_frames=args.contact_sheet_tail_frames,
        contact_sheet_cols=args.contact_sheet_cols,
        contact_sheet_cell_size=args.contact_sheet_cell_size,
        delta_window_frames=args.delta_window_frames,
        delta_stride=args.delta_stride,
        delta_cooldown_seconds=args.delta_cooldown_seconds,
        phase_context_chunks=args.phase_context_chunks,
        event_stride=args.event_stride,
        event_merge_gap_seconds=args.event_merge_gap_seconds,
        max_count_cap=args.max_count_cap,
    )

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
    forward_results, done_keys = _load_checkpoint_rows(ckpt_path, "REC")

    import torch
    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        for anno in tqdm(
            local_forward,
            desc=f"[GPU{accelerator.process_index}] REC",
            disable=not accelerator.is_local_main_process,
        ):
            key = row_key(anno)
            if key in done_keys:
                continue
            result = _evaluate_rec(
                anno, args.chunked_dir, evaluator, args.chunk_duration, args.fps,
            )
            forward_results.append(result)
            done_keys.add(key)
            shard_io.append_row(ckpt, result, key)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    shard_io.write_done_marker(done_marker)

    if accelerator.is_main_process:
        shard_io.wait_for_done_markers(args.result_dir, accelerator.num_processes)
        all_forward = _merge_checkpoint_rows(args.result_dir, accelerator.num_processes, "REC")
        model_label = f"REC-Ablation-{args.model_path.split('/')[-1]}"
        print_rec_results(model_label, all_forward, methods)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            args.result_dir, f"hub_and_spoke_rec_ablation_{timestamp}.json"
        )
        shard_io.save_json(
            output_path,
            {
                "config": vars(args),
                "summary": calculate_rec_scores(all_forward, methods),
                "forward": all_forward,
            },
        )
        print(f"\nResults saved to: {output_path}")

    accelerator.wait_for_everyone()


def _add_crr_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--model_path", required=True)
    sub.add_argument(
        "--anno_path",
        default="data/ovo_bench/ovo_bench_new.json",
    )
    sub.add_argument(
        "--chunked_dir",
        default="data/ovo_bench/chunked_videos",
    )
    sub.add_argument("--result_dir", default="results/hub_and_spoke_crr_interval")
    sub.add_argument("--chunk_duration", type=float, default=1.0)
    sub.add_argument("--fps", type=float, default=1.0)
    sub.add_argument("--max_qa_tokens", type=int, default=128)
    sub.add_argument("--methods", nargs="+", default=["tail_mono", "tail"])
    sub.add_argument("--recent_frames_only", type=int, default=4)
    sub.add_argument("--crr_caption_tail_seconds", type=float, default=45.0)
    sub.add_argument("--crr_caption_tail_lines", type=int, default=48)
    sub.add_argument("--crr_caption_every_n_chunks", type=int, default=1)
    sub.add_argument("--max_samples", type=int, default=None)


def _run_crr(args: argparse.Namespace) -> None:
    methods = _parse_methods(args.methods, CRR_METHODS)
    accelerator = Accelerator(
        kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(hours=24))]
    )

    import random
    with open(args.anno_path) as handle:
        annotations = json.load(handle)

    forward_anno = [a for a in annotations if a.get("task") == "CRR"]
    random.seed(42)
    random.shuffle(forward_anno)
    if args.max_samples is not None:
        forward_anno = forward_anno[: args.max_samples]

    accelerator.print(f"\n{'=' * 60}")
    accelerator.print("Hub-and-Spoke CRR Tail-Caption Ablations")
    accelerator.print(f"{'=' * 60}")
    accelerator.print(f"CRR items: {len(forward_anno)}  methods={','.join(methods)}")
    accelerator.print(
        f"recent_frames={args.recent_frames_only}  "
        f"tail={args.crr_caption_tail_seconds}s/{args.crr_caption_tail_lines} lines"
    )
    accelerator.print(f"{'=' * 60}\n")

    qa_model = RecentWindowQAModel(
        model_name=args.model_path,
        device=accelerator.device,
        max_new_tokens=args.max_qa_tokens,
    )
    evaluator = CrrAblationEvaluator(
        qa_model=qa_model,
        methods=methods,
        recent_frames=args.recent_frames_only,
        crr_caption_tail_seconds=args.crr_caption_tail_seconds,
        crr_caption_tail_lines=args.crr_caption_tail_lines,
        crr_caption_every_n_chunks=args.crr_caption_every_n_chunks,
    )

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
    forward_results, done_keys = _load_checkpoint_rows(ckpt_path, "CRR")

    import torch
    with open(ckpt_path, "a", encoding="utf-8") as ckpt:
        for anno in tqdm(
            local_forward,
            desc=f"[GPU{accelerator.process_index}] CRR",
            disable=not accelerator.is_local_main_process,
        ):
            key = row_key(anno)
            if key in done_keys:
                continue
            result = _evaluate_crr(
                anno, args.chunked_dir, evaluator, args.chunk_duration, args.fps,
            )
            forward_results.append(result)
            done_keys.add(key)
            shard_io.append_row(ckpt, result, key)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    shard_io.write_done_marker(done_marker)

    if accelerator.is_main_process:
        shard_io.wait_for_done_markers(args.result_dir, accelerator.num_processes)
        all_forward = _merge_checkpoint_rows(args.result_dir, accelerator.num_processes, "CRR")
        model_label = f"CRR-Ablation-{args.model_path.split('/')[-1]}"
        print_crr_results(model_label, all_forward, methods)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(
            args.result_dir, f"hub_and_spoke_crr_interval_{timestamp}.json"
        )
        shard_io.save_json(
            output_path,
            {
                "config": vars(args),
                "summary": calculate_crr_scores(all_forward, methods),
                "forward": all_forward,
            },
        )
        print(f"\nResults saved to: {output_path}")

    accelerator.wait_for_everyone()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OVO-Bench forward counting/evidence ablations (paper reproducibility)"
    )
    subparsers = parser.add_subparsers(dest="ablation", required=True)

    rec_parser = subparsers.add_parser(
        "rec", help="REC counting ablations: " + ", ".join(REC_METHODS)
    )
    _add_rec_args(rec_parser)

    crr_parser = subparsers.add_parser(
        "crr", help="CRR answerability ablations: " + ", ".join(CRR_METHODS)
    )
    _add_crr_args(crr_parser)

    args = parser.parse_args()
    if args.ablation == "rec":
        _run_rec(args)
    else:
        _run_crr(args)


if __name__ == "__main__":
    main()


'''
FORCE_QWENVL_VIDEO_READER=decord CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 HF_HOME=/data/huggingface \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
accelerate launch --num_processes 7 \
  experiments/evaluate_ovo_ablations.py rec \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --anno_path data/ovo_bench/ovo_bench_new.json \
  --chunked_dir data/ovo_bench/chunked_videos \
  --result_dir results/hub_and_spoke_rec_ablation_qwen2.5_all \
  --methods storyboard segmented contact_sheet delta event hybrid \
  --storyboard_frames 48 \
  --storyboard_tail_frames 4 \
  --segmented_segments 6 \
  --segmented_frames_per_segment 16 \
  --contact_sheet_frames 48 \
  --contact_sheet_tail_frames 4 \
  --delta_window_frames 4 \
  --delta_stride 1 \
  --delta_cooldown_seconds 1.0 \
  --phase_context_chunks 2 \
  --event_stride 1 \
  --event_merge_gap_seconds 2.5 \
  2>&1 | tee results/hub_and_spoke_rec_ablation_qwen2.5_all.log

accelerate launch --num_processes 7 \
  experiments/evaluate_ovo_ablations.py crr \
  --model_path Qwen/Qwen2.5-VL-7B-Instruct \
  --anno_path data/ovo_bench/ovo_bench_new.json \
  --chunked_dir data/ovo_bench/chunked_videos \
  --result_dir results/hub_and_spoke_crr_tail \
  --methods tail_mono tail \
  2>&1 | tee results/hub_and_spoke_crr_tail.log
'''
