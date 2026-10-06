"""Resumable JSONL checkpointing shared by the benchmark evaluators."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

KEY_FIELD = "_key"


def shard_dir(result_dir: str, rank: int, n_procs: int) -> Path:
    path = Path(result_dir) if n_procs == 1 else Path(result_dir) / f"rank_{rank}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def checkpoint_path(result_dir: str, rank: int, n_procs: int) -> str:
    return str(shard_dir(result_dir, rank, n_procs) / "results_incremental.jsonl")


def done_path(result_dir: str, rank: int, n_procs: int) -> str:
    return str(shard_dir(result_dir, rank, n_procs) / "done")


def clear_done_markers(result_dir: str, n_procs: int) -> None:
    for rank in range(n_procs):
        Path(done_path(result_dir, rank, n_procs)).unlink(missing_ok=True)


def read_rows(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def strip_key(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k != KEY_FIELD}


def append_row(handle: Any, row: dict[str, Any], key: str) -> None:
    handle.write(json.dumps({**row, KEY_FIELD: key}, ensure_ascii=False) + "\n")
    handle.flush()


def resume(path: str, key_fn: Callable[[dict[str, Any]], str]) -> tuple[list[dict[str, Any]], set[str]]:
    rows, keys = [], set()
    for raw in read_rows(path):
        row = strip_key(raw)
        rows.append(row)
        keys.add(raw.get(KEY_FIELD) or key_fn(row))
    return rows, {k for k in keys if k}


def merge_shards(
    result_dir: str,
    n_procs: int,
    key_fn: Callable[[dict[str, Any]], str] | None = None,
) -> list[dict[str, Any]]:
    paths = [
        checkpoint_path(result_dir, rank, n_procs)
        for rank in (range(n_procs) if n_procs > 1 else [0])
    ]
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for raw in read_rows(path):
            row = strip_key(raw)
            key = raw.get(KEY_FIELD) or (key_fn(row) if key_fn else "")
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(row)
    return merged


def write_done_marker(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(datetime.now().isoformat() + "\n", encoding="utf-8")


def wait_for_done_markers(result_dir: str, n_procs: int) -> None:
    if n_procs <= 1:
        return
    timeout = float(os.environ.get("FILE_SYNC_TIMEOUT_SECONDS", "43200"))
    poll = float(os.environ.get("FILE_SYNC_POLL_SECONDS", "10"))
    paths = [done_path(result_dir, rank, n_procs) for rank in range(n_procs)]
    deadline = time.time() + timeout
    while True:
        missing = [p for p in paths if not os.path.exists(p)]
        if not missing:
            return
        if time.time() >= deadline:
            raise RuntimeError(f"Timed out waiting for: {missing}")
        time.sleep(poll)


def flatten_gathered(gathered: list[Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for item in gathered:
        flat.extend(item) if isinstance(item, list) else flat.append(item)
    return flat


def save_json(path: str, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _self_check() -> None:
    import tempfile

    key_of = lambda row: f"{row['task']}:{row['id']}"
    with tempfile.TemporaryDirectory() as result_dir:
        rows = [{"task": "EPM", "id": n, "response": "A"} for n in range(4)]

        # Two ranks each append two rows; rank 0 writes one of them twice.
        for rank in (0, 1):
            with open(checkpoint_path(result_dir, rank, 2), "a", encoding="utf-8") as fh:
                for row in rows[rank * 2:rank * 2 + 2]:
                    append_row(fh, row, key_of(row))
            if rank == 0:
                with open(checkpoint_path(result_dir, rank, 2), "a", encoding="utf-8") as fh:
                    append_row(fh, rows[0], key_of(rows[0]))
            write_done_marker(done_path(result_dir, rank, 2))

        resumed, done = resume(checkpoint_path(result_dir, 0, 2), key_of)
        assert resumed == [rows[0], rows[1], rows[0]], resumed
        assert done == {"EPM:0", "EPM:1"}, done

        wait_for_done_markers(result_dir, 2)
        merged = merge_shards(result_dir, 2, key_fn=key_of)
        assert merged == rows, merged
        assert all(KEY_FIELD not in row for row in merged)

        # A legacy row with no _key is recovered through key_fn, dropped without.
        legacy = Path(checkpoint_path(result_dir, 0, 2))
        legacy.write_text(json.dumps({"task": "HLD", "id": 9}) + "\n", encoding="utf-8")
        assert merge_shards(result_dir, 2, key_fn=key_of)[0] == {"task": "HLD", "id": 9}
        assert merge_shards(result_dir, 2) == rows[2:]

    assert flatten_gathered([[{"a": 1}], {"b": 2}]) == [{"a": 1}, {"b": 2}]
    print("shard_io self-check passed")


if __name__ == "__main__":
    _self_check()
