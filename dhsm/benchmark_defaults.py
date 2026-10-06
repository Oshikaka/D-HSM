"""Lightweight OVO defaults selected from the local model identity."""

from __future__ import annotations

import json
from pathlib import Path
import re


def ovo_defaults(model_path: str | Path) -> dict[str, float | int]:
    """Read local model metadata when available, otherwise recognize its name."""
    model_type = None
    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
        if isinstance(config, dict):
            value = config.get("model_type")
            if value in ("qwen2_5_vl", "qwen3_vl"):
                model_type = value
    except (OSError, ValueError, TypeError):
        pass

    if model_type is None:
        name = str(model_path)
        qwen25 = re.search(r"(?<![a-z0-9])qwen2[._-]5[-_]vl(?![a-z0-9])", name, re.I)
        qwen3 = re.search(r"(?<![a-z0-9])qwen3[-_]vl(?![a-z0-9])", name, re.I)
        if qwen25 and not qwen3:
            model_type = "qwen2_5_vl"

    if model_type == "qwen2_5_vl":
        return {"memory_floor": 0.60, "gate_strict_sim": 0.60, "dynamic_top_k_max": 8}
    return {"memory_floor": 0.55, "gate_strict_sim": 0.70, "dynamic_top_k_max": 12}
