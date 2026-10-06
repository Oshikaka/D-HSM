"""Qwen2.5-VL question answering over a window of decoded video frames."""

from __future__ import annotations

import os
import math
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

    Qwen2.5-VL and Qwen3-VL both use the processor's native multimodal inputs,
    including the modality IDs needed for M-RoPE and Qwen3 DeepStack features.
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
        # Run generation from prepared model inputs and decode only new tokens.
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
        # Text-only generation (no images). Used for lightweight classification.
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
        # Single forward pass; pick the option whose first-token logit is highest.
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
        # Score MCQ letters using the model's native multimodal inputs.
        """Score MCQ letters using the model's native multimodal inputs."""
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
            **({"mm_token_type_ids": inputs["mm_token_type_ids"].to(text_device)}
               if "mm_token_type_ids" in inputs else {}),
        )

    @torch.inference_mode()
    def generate_from_frames(self, frames: list[Image.Image], question: str) -> str:
        # Generate with the model's native multimodal path.
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
            **({"mm_token_type_ids": inputs["mm_token_type_ids"].to(text_device)}
               if "mm_token_type_ids" in inputs else {}),
        )

    @torch.inference_mode()
    def batch_caption_from_frames(
        self, frames_list: list[list[Image.Image]], question: str
    ) -> list[str]:
        """Batched native captioning — builds a left-padded batch
        from the standard processor outputs and runs one model.generate
        call, preserving modality IDs and all visual features."""
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


def _torchcodec_pts_bound(decoder: Any, seconds: float, *, after: bool = False) -> int:
    # First frame whose real PTS is >= seconds (or > seconds for an end bound).
    lower, upper = 0, int(decoder.metadata.num_frames)
    while lower < upper:
        middle = (lower + upper) // 2
        timestamp = float(decoder.get_frame_at(middle).pts_seconds)
        if not math.isfinite(timestamp):
            raise ValueError("TorchCodec returned a non-finite frame PTS.")
        if timestamp < seconds or (after and timestamp == seconds):
            lower = middle + 1
        else:
            upper = middle
    return lower


def _read_video_with_pts(video_req: dict[str, Any]) -> tuple[torch.Tensor, list[float], list[float]]:
    """Qwen's TorchCodec sampling and resize, retaining native frame metadata.

    qwen-vl-utils 0.0.11 drops FrameBatch.pts_seconds and returns only an average
    sampling rate. It also silently changes backends on failure. Neither behavior
    is suitable for causal cutoffs, so this path requires TorchCodec explicitly.
    Unbounded sampling indices and resized pixels match Qwen's native reader.
    Explicit windows use actual PTS bounds, including for variable frame rates.
    """
    try:
        import qwen_vl_utils.vision_process as vision
        from torchcodec.decoders import VideoDecoder
    except ImportError as exc:
        raise RuntimeError("Causal video decoding requires qwen_vl_utils and torchcodec with native frame PTS.") from exc
    backend = vision.get_video_reader_backend()
    if backend != "torchcodec":
        raise RuntimeError(
            f"Video backend {backend!r} does not provide verified native PTS in this adapter. "
            "Install torchcodec and set FORCE_QWENVL_VIDEO_READER=torchcodec."
        )
    decoder = VideoDecoder(video_req["video"], num_ffmpeg_threads=int(os.environ.get("TORCHCODEC_NUM_THREADS", 8)))
    video_fps = float(decoder.metadata.average_fps)
    total_frames = int(decoder.metadata.num_frames)
    if not math.isfinite(video_fps) or video_fps <= 0 or total_frames <= 0:
        raise ValueError("Invalid TorchCodec frame count or average FPS metadata.")
    if "video_start" in video_req or "video_end" in video_req:
        start_frame = _torchcodec_pts_bound(decoder, video_req["video_start"]) if "video_start" in video_req else 0
        end_frame = (_torchcodec_pts_bound(decoder, video_req["video_end"], after=True) - 1
                     if "video_end" in video_req else total_frames - 1)
        if start_frame > end_frame:
            raise ValueError("No video frames fall within the requested PTS window.")
        total_frames = end_frame - start_frame + 1
    else:
        start_frame, end_frame, total_frames = vision.calculate_video_frame_range(video_req, total_frames, video_fps)
    # Very early queries can have only the first source frame available. Qwen's
    # video sampler requires an even frame count; our downstream API consumes
    # individual images and can use that single real frame without future-frame
    # padding or duplication.
    nframes = 1 if total_frames == 1 else vision.smart_nframes(video_req, total_frames=total_frames, video_fps=video_fps)
    indices = torch.linspace(start_frame, end_frame, nframes).round().long().tolist()
    batch = decoder.get_frames_at(indices=indices)
    video = batch.data
    if not isinstance(video, torch.Tensor) or video.ndim != 4:
        raise ValueError("TorchCodec did not return a TCHW video tensor.")
    try:
        timestamps = batch.pts_seconds.detach().cpu().reshape(-1).tolist()
        durations = batch.duration_seconds.detach().cpu().reshape(-1).tolist()
    except AttributeError as exc:
        raise RuntimeError("TorchCodec frame PTS/durations are unavailable; synthetic timing is refused.") from exc
    if len(timestamps) != len(video) or len(durations) != len(video):
        raise ValueError("TorchCodec frame metadata does not align with decoded frames.")
    if (not all(math.isfinite(t) and t >= 0 for t in timestamps)
            or any(b < a for a, b in zip(timestamps, timestamps[1:]))
            or not all(math.isfinite(d) and d >= 0 for d in durations)):
        raise ValueError("TorchCodec returned invalid or non-monotone PTS/durations.")
    if any(t < video_req.get("video_start", 0.0) or t > video_req.get("video_end", math.inf) for t in timestamps):
        raise ValueError("Decoded frame PTS escaped the requested video window.")

    # Mirror fetch_video's resize exactly; these are the installed package's
    # constants/helpers, including its VIDEO_MAX_PIXELS/total-pixel budget.
    _, _, height, width = video.shape
    min_pixels = video_req.get("min_pixels", vision.VIDEO_MIN_PIXELS)
    total_pixels = video_req.get("total_pixels", vision.VIDEO_TOTAL_PIXELS)
    max_pixels = max(min(vision.VIDEO_MAX_PIXELS, total_pixels / nframes * vision.FRAME_FACTOR), int(min_pixels * 1.05))
    max_pixels = min(video_req.get("max_pixels", max_pixels), max_pixels)
    if "resized_height" in video_req and "resized_width" in video_req:
        resized_height, resized_width = vision.smart_resize(
            video_req["resized_height"], video_req["resized_width"], factor=vision.IMAGE_FACTOR)
    else:
        resized_height, resized_width = vision.smart_resize(
            height, width, factor=vision.IMAGE_FACTOR, min_pixels=min_pixels, max_pixels=max_pixels)
    video = vision.transforms.functional.resize(
        video, [resized_height, resized_width], interpolation=vision.InterpolationMode.BICUBIC,
        antialias=True,
    ).float()
    return video, [float(t) for t in timestamps], [float(d) for d in durations]


def decode_video_to_chunks_qwen(
    video_path: str,
    chunk_duration: float,
    fps: float,
    video_start: float | None = None,
    video_end: float | None = None,
) -> tuple[list[EvalChunk], str]:
    if not math.isfinite(chunk_duration) or chunk_duration <= 0:
        raise ValueError(f"chunk_duration must be > 0, got {chunk_duration}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    video_req: dict[str, Any] = {"video": video_path, "fps": float(fps)}
    for field, value in (("video_start", video_start), ("video_end", video_end)):
        if value is not None:
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite, got {value}")
            video_req[field] = max(0.0, float(value))
    if "video_end" in video_req and video_req["video_end"] < video_req.get("video_start", 0.0):
        raise ValueError("video_end must be >= video_start.")

    video, timestamps, durations = _read_video_with_pts(video_req)
    sampled_fps = ((len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                   if len(timestamps) > 1 and timestamps[-1] > timestamps[0] else float(fps))
    decode_backend = "torchcodec_pts"
    if video_start is not None or video_end is not None:
        decode_backend = f"{decode_backend}_window"
    max_valid_end = max((t + d for t, d in zip(timestamps, durations)), default=0.0)
    if "video_end" in video_req:
        max_valid_end = min(max_valid_end, video_req["video_end"])

    frame_buckets: dict[int, list[tuple[Image.Image, float]]] = {}
    for i, ts in enumerate(timestamps):
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
                end_time=max(max(ts for _, ts in chunk_frames),
                             min((chunk_idx + 1) * chunk_duration, max_valid_end)),
                chunk_index=chunk_idx,
                fps=sampled_fps,
            )
        )
    return chunks, decode_backend
