"""Native Qwen-VL evaluation with a patch-embedding optimization."""

from __future__ import annotations

from typing import NoReturn

import torch
from PIL import Image

from .video_qa import RecentWindowQAModel as _BaseQAModel


class RecentWindowQAModel(_BaseQAModel):
    # Use native processor/model paths for answering and history captioning.

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        super().__init__(
            model_name=model_name,
            device=device,
            max_new_tokens=max_new_tokens,
            attn_implementation=attn_implementation,
        )
        self._patch_slow_conv3d_patch_embed()

    def _patch_slow_conv3d_patch_embed(self) -> None:
        # Replace the vision patch-embed conv3d with its exact matmul equivalent.
        visual = self._get_visual_module()
        patch_embed = getattr(visual, "patch_embed", None)
        proj = getattr(patch_embed, "proj", None)
        if proj is None or not hasattr(proj, "kernel_size"):
            return
        if tuple(proj.kernel_size) != tuple(proj.stride) or proj.padding not in ((0, 0, 0), 0):
            return

        weight_2d = proj.weight.detach().reshape(proj.weight.shape[0], -1)
        bias = proj.bias.detach() if proj.bias is not None else None

        def linear_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = hidden_states.reshape(-1, weight_2d.shape[1])
            hidden_states = hidden_states.to(dtype=weight_2d.dtype)
            return torch.nn.functional.linear(hidden_states, weight_2d, bias)

        patch_embed.forward = linear_forward

    @staticmethod
    def _reject_incomplete_cache() -> NoReturn:
        raise RuntimeError(
            "The old cached-vision API omits Qwen3-VL DeepStack features and is disabled. "
            "Use generate_from_frames, score_mcq_from_frames, or batch_caption_from_frames "
            "with the original frames so the model receives complete native visual inputs."
        )

    def encode_vision(self, frames: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        self._reject_incomplete_cache()

    def generate_with_cached_vision(
        self, cached_embeds: torch.Tensor, cached_grid_thw: torch.Tensor, question: str,
    ) -> str:
        self._reject_incomplete_cache()

    def batch_generate_with_cached_vision(
        self, cached_embeds_list: list[torch.Tensor], cached_grid_thw_list: list[torch.Tensor], question: str,
    ) -> list[str]:
        self._reject_incomplete_cache()

    def _score_mcq_with_cached_vision(
        self, cached_embeds: torch.Tensor, cached_grid_thw: torch.Tensor, question: str, num_options: int = 4,
    ) -> str:
        self._reject_incomplete_cache()
