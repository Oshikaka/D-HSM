"""Refuse to resume evaluation results under a different run protocol."""

from __future__ import annotations
import fcntl
import json
import os
import tempfile
from pathlib import Path
from typing import Any

MANIFEST_NAME = "evaluation_protocol.json"


def ensure_result_protocol(result_dir: str, protocol: dict[str, Any]) -> None:
    # Validate/create a manifest before loading a model or resuming results.

    directory = Path(result_dir)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / MANIFEST_NAME
    expected = {"schema_version": 1, "protocol": protocol}
    # Round-trip to normalize tuples and other JSON-supported values.
    serialized = json.dumps(expected, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    expected = json.loads(serialized)
    with (directory / ".evaluation_protocol.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest.exists():
            try:
                recorded = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError(
                    f"Cannot read evaluation protocol in {manifest}; use a new --result_dir."
                ) from exc
            if recorded != expected:
                raise ValueError(
                    f"Evaluation protocol mismatch in {manifest}. "
                    "The routing, prompt policy, model, data, or evaluation settings changed; "
                    "use a new --result_dir to avoid mixing results."
                )
            return

        artifacts = (
            path for path in directory.rglob("*")
            if path.is_file() and path.suffix in {".json", ".jsonl"} and path.stat().st_size
        )
        if next(artifacts, None) is not None:
            raise ValueError(
                f"Existing results in {directory} have no {MANIFEST_NAME}; "
                "their evaluation protocol is unknown. Use a new --result_dir."
            )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", prefix=".evaluation_protocol.",
                suffix=".tmp", dir=directory, delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, manifest)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
