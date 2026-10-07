"""Lightweight benchmark defaults selected from the local model identity."""

from __future__ import annotations

import json
from pathlib import Path
import re


def _model_type(model_path: str | Path) -> str | None:
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
        elif qwen3 and not qwen25:
            model_type = "qwen3_vl"
    return model_type


def ovo_defaults(model_path: str | Path) -> dict[str, float | int]:
    """Preserve the existing OVO profiles, including the unknown-model fallback."""
    if _model_type(model_path) == "qwen2_5_vl":
        return {"memory_floor": 0.60, "gate_strict_sim": 0.60, "dynamic_top_k_max": 8}
    return {"memory_floor": 0.55, "gate_strict_sim": 0.70, "dynamic_top_k_max": 12}


def streamingbench_defaults(
    model_path: str | Path, recent_frames: int, *, memory_mode: str = "entity_resolved",
) -> dict[str, float | int | None]:
    """Use the evaluated SB profiles only for known models and 4/8-frame memory."""
    defaults = {"memory_floor": None, "gate_strict_sim": 0.55,
                "dynamic_top_k_max": 12, "spoke_attach_threshold": 0.65}
    if memory_mode != "entity_resolved":
        return defaults
    profiles = {
        ("qwen2_5_vl", 4): (0.25, 0.55, 12, 0.55),
        ("qwen2_5_vl", 8): (0.635, 0.66, 12, 0.65),
        ("qwen3_vl", 4): (0.25, 0.55, 16, 0.65),
        ("qwen3_vl", 8): (0.25, 0.55, 8, 0.65),
    }
    values = profiles.get((_model_type(model_path), recent_frames))
    if values is not None:
        defaults.update(zip(defaults, values))
    return defaults
