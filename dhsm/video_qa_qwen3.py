from __future__ import annotations

import torch
from PIL import Image

from .video_qa import RecentWindowQAModel as _BaseQAModel


class RecentWindowQAModel(_BaseQAModel):
    """Qwen3-VL wrapper: one cached vision block reused across the window."""

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
        self.vision_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
        self.vision_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
        self.im_start_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.im_end_id = self.processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.merge_size = self._get_visual_module().spatial_merge_size
        self._patch_slow_conv3d_patch_embed()

    def _patch_slow_conv3d_patch_embed(self) -> None:
        """Replace the vision patch-embed conv3d with its exact matmul equivalent.

        The patch embed uses kernel_size == stride, and the processor already
        feeds pre-flattened (n_patches, C*T*P*P) patches, so each conv window
        maps to exactly one input row: conv3d(x) == x @ W.view(D, -1).T + b.
        On some torch/cuDNN builds (observed: torch 2.9.1+cu128) bf16 conv3d
        picks a pathological algorithm (~450s for 24 frames vs <0.1s as a
        matmul), so bypassing it is both a correctness no-op and a large win."""
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

    @torch.inference_mode()
    def encode_vision(self, frames: list[Image.Image]) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep official preprocessing, but expose encoded vision for explicit input building."""
        content = [{"type": "image", "image": frame} for frame in frames]
        content.append({"type": "text", "text": "."})
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"].to(self.model.device, dtype=self._get_visual_dtype())
        image_grid_thw = inputs["image_grid_thw"].to(self.model.device)
        vision_output = self._get_image_feature_model().get_image_features(pixel_values, image_grid_thw)
        image_embeds = self._flatten_vision_features(vision_output)
        # Cache deepstack features (tuple of tensors, one per deepstack layer) when available,
        # so generation can re-inject them into the language model's forward pass.
        self._last_deepstack_features = getattr(vision_output, "deepstack_features", None)

        del pixel_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return image_embeds, image_grid_thw

    @torch.inference_mode()
    def generate_with_cached_vision(
        self,
        cached_embeds: torch.Tensor,
        cached_grid_thw: torch.Tensor,
        question: str,
    ) -> str:
        device = self.model.device
        tokenizer = self.processor.tokenizer
        text_model = self._get_text_model()

        num_vision_tokens = int(cached_embeds.shape[0])
        self._last_num_vision_tokens = num_vision_tokens
        self._last_num_vision_frames = int(cached_grid_thw.shape[0]) if cached_grid_thw is not None else 0

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        grid_rows = cached_grid_thw.to(device)

        # Per-image token counts (after spatial merge) — must sum to num_vision_tokens.
        merge_sq = int(self.merge_size) ** 2
        per_image_token_counts = [
            int(cached_grid_thw[i].prod().item()) // merge_sq
            for i in range(cached_grid_thw.shape[0])
        ]
        assert sum(per_image_token_counts) == num_vision_tokens, (
            f"per-image token counts {per_image_token_counts} do not sum to "
            f"cached vision tokens {num_vision_tokens}"
        )

        input_ids_list: list[int] = []
        input_ids_list.extend([self.im_start_id])
        input_ids_list.extend(tokenizer.encode("user\n", add_special_tokens=False))
        # Qwen3-VL's get_rope_index splits modality runs via itertools.groupby on
        # mm_token_type_ids, so each image must sit in its own <|vision_start|>...<|vision_end|>
        # block separated by (text-typed) vision_start/vision_end tokens.
        for count in per_image_token_counts:
            input_ids_list.append(self.vision_start_id)
            input_ids_list.extend([self.image_token_id] * count)
            input_ids_list.append(self.vision_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend(question_ids)
        input_ids_list.append(self.im_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend([self.im_start_id])
        input_ids_list.extend(tokenizer.encode("assistant\n", add_special_tokens=False))

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)

        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        cached_embeds = cached_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = input_ids == self.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, cached_embeds)

        # Qwen3-VL moved get_rope_index onto Qwen3VLModel (the parent of language_model)
        # and now requires `mm_token_type_ids` (0 = text, 1 = image, 2 = video).
        rope_model = self._get_hf_model().model
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
        mm_token_type_ids[image_mask] = 1
        position_ids, _ = rope_model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=grid_rows,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        return self._generate_from_model_inputs(
            prompt_length=len(input_ids[0]),
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

    @torch.inference_mode()
    def generate_from_frames(self, frames: list[Image.Image], question: str) -> str:
        cached_embeds, cached_grid_thw = self.encode_vision(frames)
        return self.generate_with_cached_vision(cached_embeds, cached_grid_thw, question)

    @torch.inference_mode()
    def batch_generate_with_cached_vision(
        self,
        cached_embeds_list: list[torch.Tensor],
        cached_grid_thw_list: list[torch.Tensor],
        question: str,
    ) -> list[str]:
        """Batched cached-vision generation: one model.generate() call for B
        samples with the SAME text question but distinct frame sets."""
        B = len(cached_embeds_list)
        if B == 0:
            return []
        if B == 1:
            return [
                self.generate_with_cached_vision(
                    cached_embeds_list[0], cached_grid_thw_list[0], question
                )
            ]

        device = self.model.device
        tokenizer = self.processor.tokenizer
        text_model = self._get_text_model()
        merge_sq = int(self.merge_size) ** 2

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        user_prefix = tokenizer.encode("user\n", add_special_tokens=False)
        nl_ids = tokenizer.encode("\n", add_special_tokens=False)
        asst_prefix = tokenizer.encode("assistant\n", add_special_tokens=False)

        per_sample_ids: list[list[int]] = []
        for i in range(B):
            grid_thw = cached_grid_thw_list[i]
            counts = [
                int(grid_thw[k].prod().item()) // merge_sq
                for k in range(grid_thw.shape[0])
            ]
            assert sum(counts) == int(cached_embeds_list[i].shape[0])
            ids: list[int] = [self.im_start_id, *user_prefix]
            for cnt in counts:
                ids.append(self.vision_start_id)
                ids.extend([self.image_token_id] * cnt)
                ids.append(self.vision_end_id)
            ids.extend(nl_ids)
            ids.extend(question_ids)
            ids.append(self.im_end_id)
            ids.extend(nl_ids)
            ids.append(self.im_start_id)
            ids.extend(asst_prefix)
            per_sample_ids.append(ids)

        max_len = max(len(ids) for ids in per_sample_ids)
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else (tokenizer.eos_token_id or 0)
        )

        # Left-pad: each sample's content sits flush-right so the last token
        # is at column max_len-1, which is required for batched generate.
        padded_input_ids = torch.full(
            (B, max_len), pad_id, dtype=torch.long, device=device
        )
        attention_mask = torch.zeros((B, max_len), dtype=torch.long, device=device)
        for i, ids in enumerate(per_sample_ids):
            L = len(ids)
            padded_input_ids[i, max_len - L:] = torch.tensor(
                ids, dtype=torch.long, device=device
            )
            attention_mask[i, max_len - L:] = 1

        inputs_embeds = text_model.get_input_embeddings()(padded_input_ids)
        target_dtype = inputs_embeds.dtype

        # Cat sample embeds in batch order; masked_scatter on the expanded
        # image mask fills row-major (sample 0 first, then 1, …) which lines
        # up with the cat order.
        all_embeds = torch.cat(
            [e.to(device=device, dtype=target_dtype) for e in cached_embeds_list],
            dim=0,
        )
        image_mask = padded_input_ids == self.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, all_embeds)

        # M-RoPE: compute (rope_dim, 1, L_i) per sample, then pad-cat onto a
        # (rope_dim, B, max_len) tensor with zeros on the left.
        rope_model = self._get_hf_model().model
        per_sample_pos_ids: list[torch.Tensor] = []
        for i in range(B):
            L = len(per_sample_ids[i])
            sample_input_ids = padded_input_ids[i:i + 1, max_len - L:]
            sample_mask = attention_mask[i:i + 1, max_len - L:]
            sample_image_mask = sample_input_ids == self.image_token_id
            mm_tt = torch.zeros_like(sample_input_ids, dtype=torch.int32)
            mm_tt[sample_image_mask] = 1
            sample_pos, _ = rope_model.get_rope_index(
                input_ids=sample_input_ids,
                mm_token_type_ids=mm_tt,
                image_grid_thw=cached_grid_thw_list[i].to(device),
                video_grid_thw=None,
                attention_mask=sample_mask,
            )
            per_sample_pos_ids.append(sample_pos)

        rope_dim = per_sample_pos_ids[0].shape[0]
        position_ids = torch.zeros(
            (rope_dim, B, max_len),
            dtype=per_sample_pos_ids[0].dtype,
            device=device,
        )
        for i, pos in enumerate(per_sample_pos_ids):
            L = pos.shape[-1]
            position_ids[:, i, max_len - L:] = pos[:, 0, :]

        generated_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        # When generate is fed inputs_embeds it returns only new tokens.
        # If a future transformers release ever prepends the prompt we still
        # handle it via the length check.
        if generated_ids.shape[1] > max_len:
            new_token_ids = generated_ids[:, max_len:]
        else:
            new_token_ids = generated_ids

        return [
            tokenizer.decode(
                new_token_ids[i],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()
            for i in range(B)
        ]

    @torch.inference_mode()
    def batch_caption_from_frames(
        self, frames_list: list[list[Image.Image]], question: str
    ) -> list[str]:
        """Batched captioning for Qwen3-VL — encodes each chunk's frames
        independently (variable counts), then runs a single batched generate."""
        if not frames_list:
            return []
        if len(frames_list) == 1:
            return [self.generate_from_frames(frames_list[0], question)]
        cached_embeds_list: list[torch.Tensor] = []
        cached_grid_thw_list: list[torch.Tensor] = []
        for fs in frames_list:
            embeds, grid_thw = self.encode_vision(fs)
            cached_embeds_list.append(embeds)
            cached_grid_thw_list.append(grid_thw)
        return self.batch_generate_with_cached_vision(
            cached_embeds_list, cached_grid_thw_list, question
        )

    @torch.inference_mode()
    def score_mcq_from_frames(
        self, frames: list[Image.Image], question: str, num_options: int = 4
    ) -> str:
        """Logit-based MCQ scoring via cached vision path (Qwen3)."""
        cached_embeds, cached_grid_thw = self.encode_vision(frames)
        return self._score_mcq_with_cached_vision(
            cached_embeds, cached_grid_thw, question, num_options
        )

    @torch.inference_mode()
    def _score_mcq_with_cached_vision(
        self,
        cached_embeds: torch.Tensor,
        cached_grid_thw: torch.Tensor,
        question: str,
        num_options: int = 4,
    ) -> str:
        """Build inputs same as generate_with_cached_vision, score by logit."""
        device = self.model.device
        tokenizer = self.processor.tokenizer
        text_model = self._get_text_model()

        num_vision_tokens = int(cached_embeds.shape[0])
        merge_sq = int(self.merge_size) ** 2
        per_image_token_counts = [
            int(cached_grid_thw[i].prod().item()) // merge_sq
            for i in range(cached_grid_thw.shape[0])
        ]
        assert sum(per_image_token_counts) == num_vision_tokens

        question_ids = tokenizer.encode(question, add_special_tokens=False)
        grid_rows = cached_grid_thw.to(device)

        input_ids_list: list[int] = []
        input_ids_list.extend([self.im_start_id])
        input_ids_list.extend(tokenizer.encode("user\n", add_special_tokens=False))
        for count in per_image_token_counts:
            input_ids_list.append(self.vision_start_id)
            input_ids_list.extend([self.image_token_id] * count)
            input_ids_list.append(self.vision_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend(question_ids)
        input_ids_list.append(self.im_end_id)
        input_ids_list.extend(tokenizer.encode("\n", add_special_tokens=False))
        input_ids_list.extend([self.im_start_id])
        input_ids_list.extend(tokenizer.encode("assistant\n", add_special_tokens=False))

        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        prompt_length = len(input_ids_list)

        inputs_embeds = text_model.get_input_embeddings()(input_ids)
        cached_embeds = cached_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask = input_ids == self.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, cached_embeds)

        rope_model = self._get_hf_model().model
        mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
        mm_token_type_ids[image_mask] = 1
        position_ids, _ = rope_model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=grid_rows,
            video_grid_thw=None,
            attention_mask=attention_mask,
        )

        return self._score_mcq_logits_from_inputs(
            prompt_length=prompt_length,
            num_options=num_options,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
