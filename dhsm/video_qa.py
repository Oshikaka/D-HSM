"""Qwen2.5-VL question answering over a window of decoded video frames.

Benchmark-agnostic: decoding, prompting, logit scoring and the token/latency
accounting.  Benchmark-specific prompts and scoring live with the evaluators.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


class _TTFTStreamer:
    def __init__(self, start_time: float) -> None:
        self.start_time = start_time
        self.ttft_seconds: float | None = None

    def put(self, value: torch.Tensor) -> None:
        if self.ttft_seconds is None:
            self.ttft_seconds = time.perf_counter() - self.start_time

    def end(self) -> None:
        pass


@dataclass
class EvalChunk:
    frames: list[Image.Image]
    frame_timestamps: list[float]
    start_time: float
    end_time: float
    chunk_index: int
    fps: float


class RecentWindowQAModel:
    """Minimal Qwen-VL wrapper for the recent-window recency baseline.

    Qwen2.5-VL keeps one ``<|vision_start|>...<|vision_end|>`` block per frame.
    Qwen3-VL overrides this path with the single-block cached builder.
    """

    def __init__(
        self,
        model_name: str,
        device: str | torch.device = "auto",
        max_new_tokens: int = 256,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        from transformers import AutoProcessor

        if "qwen3" in model_name.lower():
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                Qwen3VLForConditionalGeneration as _ModelClass,
            )
        else:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLForConditionalGeneration as _ModelClass,
            )

        self.model_name = model_name
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self._last_ttft_seconds: float = 0.0
        self._last_num_vision_tokens: int = 0
        self._last_num_vision_frames: int = 0

        proc_kwargs: dict[str, Any] = {}
        if os.environ.get("MIN_PIXELS"):
            proc_kwargs["min_pixels"] = int(os.environ["MIN_PIXELS"])
        if os.environ.get("MAX_PIXELS"):
            proc_kwargs["max_pixels"] = int(os.environ["MAX_PIXELS"])
        self.processor = AutoProcessor.from_pretrained(model_name, **proc_kwargs)

        model_kwargs: dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": attn_implementation,
        }
        if device == "auto":
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = str(device)

        _dist = torch.distributed.is_available() and torch.distributed.is_initialized()
        _saved_ws = os.environ.pop("WORLD_SIZE", None)
        try:
            if _dist and torch.distributed.get_rank() != 0:
                torch.distributed.barrier()
            self.model = _ModelClass.from_pretrained(model_name, **model_kwargs)
            if _dist and torch.distributed.get_rank() == 0:
                torch.distributed.barrier()
        finally:
            if _saved_ws is not None:
                os.environ["WORLD_SIZE"] = _saved_ws

        self.model.eval()

        _hf_model = (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )
        self._hf_model = _hf_model
        self.image_token_id = _hf_model.config.image_token_id
        if hasattr(_hf_model, "visual"):
            self._visual = _hf_model.visual
            self._text_model = _hf_model.model
        else:
            # Newer transformers: Qwen2_5_VLForConditionalGeneration wraps
            # visual/language_model under `.model`.
            self._visual = _hf_model.model.visual
            self._text_model = _hf_model.model.language_model
        self.merge_size = getattr(self._visual, "spatial_merge_size", 1)

        tokenizer = self.processor.tokenizer
        self._vision_start_id = tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self._vision_end_id = tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self._im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    def _get_hf_model(self):
        if hasattr(self, "_hf_model"):
            return self._hf_model
        return self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model

    def _get_visual_module(self):
        if hasattr(self, "_visual"):
            return self._visual
        hf_model = self._get_hf_model()
        if hasattr(hf_model, "visual"):
            return hf_model.visual
        return hf_model.model.visual

    def _get_text_model(self):
        if hasattr(self, "_text_model"):
            return self._text_model
        hf_model = self._get_hf_model()
        return hf_model.model if hasattr(hf_model, "model") else hf_model

    def _get_image_feature_model(self):
        hf_model = self._get_hf_model()
        if hasattr(hf_model, "get_image_features"):
            return hf_model
        return hf_model.model

    def _get_visual_dtype(self) -> torch.dtype:
        visual = self._get_visual_module()
        if hasattr(visual, "dtype"):
            return visual.dtype
        if hasattr(self.model, "dtype"):
            return self.model.dtype
        return torch.bfloat16

    def _flatten_vision_features(self, features: Any) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features
        # Handle HuggingFace ModelOutput (e.g., Qwen3-VL BaseModelOutputWithDeepstackFeatures).
        # For Qwen3-VL, get_image_features sets pooler_output to a tuple of per-image embeds;
        # fall back to last_hidden_state / image_embeds for other variants.
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            return self._flatten_vision_features(features.pooler_output)
        if hasattr(features, "image_embeds") and features.image_embeds is not None:
            return self._flatten_vision_features(features.image_embeds)
        if hasattr(features, "last_hidden_state") and features.last_hidden_state is not None:
            return features.last_hidden_state
        if isinstance(features, (tuple, list)):
            if features and all(isinstance(item, torch.Tensor) for item in features):
                return torch.cat(list(features), dim=0)
            first = features[0] if features else None
            if isinstance(first, torch.Tensor):
                return first
            if isinstance(first, (tuple, list)) and first and all(isinstance(item, torch.Tensor) for item in first):
                return torch.cat(list(first), dim=0)
        raise TypeError(f"Unexpected vision feature type: {type(features)}")

    def _infer_module_device(self, module: Any) -> torch.device:
        for parameter in module.parameters():
            return parameter.device
        for buffer in module.buffers():
            return buffer.device
        if hasattr(self.model, "device"):
            return self.model.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_visual_device(self) -> torch.device:
        return self._infer_module_device(self._get_visual_module())

    def _get_text_input_device(self) -> torch.device:
        embeddings = self._get_text_model().get_input_embeddings()
        return self._infer_module_device(embeddings)

    @torch.inference_mode()
    def _generate_from_model_inputs(self, prompt_length: int, **generate_kwargs: Any) -> str:
        """Run generation from prepared model inputs and decode only new tokens."""
        t0 = time.perf_counter()
        streamer = _TTFTStreamer(t0)
        generated_ids = self.model.generate(
            **generate_kwargs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            streamer=streamer,
        )
        self._last_ttft_seconds = (
            streamer.ttft_seconds
            if streamer.ttft_seconds is not None
            else (time.perf_counter() - t0)
        )

        trimmed = [
            generated_ids[0][prompt_length:]
            if generated_ids.shape[1] > prompt_length
            else generated_ids[0]
        ]
        return self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

    @torch.inference_mode()
    def generate_text_only(self, prompt: str, max_new_tokens: int = 8) -> str:
        """Text-only generation (no images). Used for lightweight classification."""
        text_device = self._get_text_input_device()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(text_device)
        attention_mask = inputs["attention_mask"].to(text_device)
        return self._generate_from_model_inputs(
            prompt_length=int(input_ids.shape[1]),
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    @torch.inference_mode()
    def _score_mcq_logits_from_inputs(
        self, prompt_length: int, num_options: int = 4, **fwd_kwargs: Any
    ) -> str:
        """Single forward pass; pick the option whose first-token logit is highest."""
        tokenizer = self.processor.tokenizer
        option_ids = [
            tokenizer.encode(chr(65 + i), add_special_tokens=False)[0]
            for i in range(num_options)
        ]
        outputs = self.model(**fwd_kwargs)
        # logits[0, prompt_length-1] predicts the first generated token
        last_logits = outputs.logits[0, prompt_length - 1, :]
        scores = torch.stack([last_logits[tid] for tid in option_ids])
        return chr(65 + int(scores.argmax().item()))

    @torch.inference_mode()
    def score_mcq_from_frames(
        self, frames: list[Image.Image], question: str, num_options: int = 4
    ) -> str:
        """Logit-based MCQ scoring for Qwen2.5-VL (eliminates position bias)."""
        visual_device = self._get_visual_device()
        text_device = self._get_text_input_device()

        content: list[dict[str, Any]] = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(text_device)
        attention_mask = inputs["attention_mask"].to(text_device)
        pixel_values = inputs["pixel_values"].to(visual_device, dtype=self._get_visual_dtype())
        image_grid_thw = inputs["image_grid_thw"].to(visual_device)

        return self._score_mcq_logits_from_inputs(
            prompt_length=int(input_ids.shape[1]),
            num_options=num_options,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    @torch.inference_mode()
    def generate_from_frames(self, frames: list[Image.Image], question: str) -> str:
        """Generate with the model's native multimodal path for Qwen2.5-VL."""
        visual_device = self._get_visual_device()
        text_device = self._get_text_input_device()

        content: list[dict[str, Any]] = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].to(text_device)
        attention_mask = inputs["attention_mask"].to(text_device)
        pixel_values = inputs["pixel_values"].to(visual_device, dtype=self._get_visual_dtype())
        image_grid_thw = inputs["image_grid_thw"].to(visual_device)

        self._last_num_vision_tokens = int((input_ids == self.image_token_id).sum().item())
        self._last_num_vision_frames = int(image_grid_thw.shape[0])

        return self._generate_from_model_inputs(
            prompt_length=int(input_ids.shape[1]),
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

    @torch.inference_mode()
    def batch_caption_from_frames(
        self, frames_list: list[list[Image.Image]], question: str
    ) -> list[str]:
        """Batched captioning for Qwen2.5-VL — builds a left-padded batch
        from the standard processor outputs and runs one model.generate
        call.  The Qwen3-VL subclass overrides this with its cached-vision
        path."""
        if not frames_list:
            return []
        if len(frames_list) == 1:
            return [self.generate_from_frames(frames_list[0], question)]

        visual_device = self._get_visual_device()
        text_device = self._get_text_input_device()
        visual_dtype = self._get_visual_dtype()
        tokenizer = self.processor.tokenizer

        per_sample: list[dict[str, Any]] = []
        for frames in frames_list:
            content: list[dict[str, Any]] = [
                {"type": "image", "image": frame} for frame in frames
            ]
            content.append({"type": "text", "text": question})
            messages = [{"role": "user", "content": content}]
            per_sample.append(self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ))

        B = len(per_sample)
        seq_lens = [int(s["input_ids"].shape[1]) for s in per_sample]
        max_len = max(seq_lens)
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.eos_token_id or 0)
        )

        # Left-pad input_ids / attention_mask / mm_token_type_ids so each
        # sample's content sits flush-right (last real token at col max_len-1),
        # which is what model.generate expects for batched decoding.
        padded_input_ids = torch.full(
            (B, max_len), pad_id, dtype=torch.long, device=text_device
        )
        attention_mask = torch.zeros(
            (B, max_len), dtype=torch.long, device=text_device
        )
        mm_token_type_ids = torch.zeros(
            (B, max_len), dtype=torch.int32, device=text_device
        )
        for i, s in enumerate(per_sample):
            L = seq_lens[i]
            padded_input_ids[i, max_len - L:] = s["input_ids"][0].to(text_device)
            attention_mask[i, max_len - L:] = 1
            if "mm_token_type_ids" in s:
                mm_token_type_ids[i, max_len - L:] = s["mm_token_type_ids"][0].to(
                    device=text_device, dtype=torch.int32
                )
            else:
                row_image_mask = padded_input_ids[i] == self.image_token_id
                mm_token_type_ids[i, row_image_mask] = 1

        # Cat vision inputs across samples in batch order — the model
        # masked_scatters per-frame features into image_token positions
        # row-major, matching this concat order.
        pixel_values = torch.cat(
            [
                s["pixel_values"].to(visual_device, dtype=visual_dtype)
                for s in per_sample
            ],
            dim=0,
        )
        image_grid_thw = torch.cat(
            [s["image_grid_thw"].to(visual_device) for s in per_sample],
            dim=0,
        )

        # We let the model compute M-RoPE position_ids itself via
        # compute_3d_position_ids → get_rope_index; both handle batched
        # left-padded inputs correctly when mm_token_type_ids is supplied.
        generated_ids = self.model.generate(
            input_ids=padded_input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        new_token_ids = generated_ids[:, max_len:]
        return [
            tokenizer.decode(
                new_token_ids[i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            for i in range(B)
        ]


def decode_video_to_chunks_qwen(
    video_path: str,
    chunk_duration: float,
    fps: float,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[list[EvalChunk], str]:
    try:
        from qwen_vl_utils.vision_process import fetch_video
    except ImportError as exc:
        raise RuntimeError("qwen_vl_utils is required for video decoding.") from exc

    if chunk_duration <= 0:
        raise ValueError(f"chunk_duration must be > 0, got {chunk_duration}")

    video_req: dict[str, Any] = {"video": video_path, "fps": float(fps)}
    if video_start is not None:
        video_req["video_start"] = max(0.0, float(video_start))
    if video_end is not None:
        video_req["video_end"] = max(0.0, float(video_end))

    try:
        video, sample_fps = fetch_video(video_req, return_video_sample_fps=True)
        metadata = {"fps": float(sample_fps)}
    except TypeError:
        # Older qwen_vl_utils that supports return_video_metadata
        video, metadata = fetch_video(video_req, return_video_metadata=True)

    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise ValueError(f"Unexpected qwen_vl_utils output for video={video_path!r}")

    meta = metadata if isinstance(metadata, dict) else {}
    raw_fps = max(float(meta.get("fps", fps if fps > 0 else 1.0)), 1e-6)
    frame_indices = meta.get("frames_indices")
    if isinstance(frame_indices, torch.Tensor):
        frame_indices = frame_indices.detach().cpu().reshape(-1).tolist()
    elif frame_indices is not None and not isinstance(frame_indices, (list, tuple)):
        try:
            frame_indices = list(frame_indices)
        except TypeError:
            frame_indices = None
    if frame_indices is None or len(frame_indices) != int(video.shape[0]):
        start_frame = int(max(0.0, float(video_start or 0.0)) * raw_fps)
        frame_indices = [start_frame + i for i in range(int(video.shape[0]))]
    frame_indices = [int(x) for x in frame_indices]

    if len(frame_indices) > 1:
        sampled_duration = float(frame_indices[-1] - frame_indices[0]) / raw_fps
        sampled_fps = float(len(frame_indices) - 1) / max(sampled_duration, 1e-6)
    else:
        sampled_fps = max(float(fps), 1e-6)
    decode_backend = str(meta.get("video_backend", "unknown"))
    if video_start is not None or video_end is not None:
        decode_backend = f"{decode_backend}_window"

    max_ts = max((float(idx) / raw_fps for idx in frame_indices), default=0.0)
    if len(frame_indices) > 1:
        frame_dt = max(float(frame_indices[-1] - frame_indices[-2]) / raw_fps, 1.0 / raw_fps)
    else:
        frame_dt = 1.0 / raw_fps
    max_valid_end = max_ts + frame_dt

    frame_buckets: dict[int, list[tuple[Image.Image, float]]] = {}
    for i, frame_idx in enumerate(frame_indices):
        ts = float(frame_idx) / raw_fps
        chunk_idx = int(ts // chunk_duration)
        frame = video[i].clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        frame_buckets.setdefault(chunk_idx, []).append((Image.fromarray(frame), ts))
    del video

    chunks: list[EvalChunk] = []
    for chunk_idx in sorted(frame_buckets):
        chunk_frames = frame_buckets[chunk_idx]
        chunks.append(
            EvalChunk(
                frames=[frame for frame, _ in chunk_frames],
                frame_timestamps=[ts for _, ts in chunk_frames],
                start_time=chunk_idx * chunk_duration,
                end_time=min((chunk_idx + 1) * chunk_duration, max_valid_end),
                chunk_index=chunk_idx,
                fps=sampled_fps,
            )
        )
    return chunks, decode_backend
