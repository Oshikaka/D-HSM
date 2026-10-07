"""Shared memory selection for the OVO-Bench and StreamingBench evaluators."""

from .entity_memory import EntityResolvedMemory
from .hub_and_spoke import ENTITY_LINK_THRESHOLD, _is_count_question
from .hub_and_spoke_incremental import IncrementalHubAndSpokeEvaluator
from .retrieval_gate import ROUTING_KEYWORD


DEFAULT_MEMORY_MODE = "entity_resolved"
MEMORY_MODE_CHOICES = (DEFAULT_MEMORY_MODE, "hub_spoke", "flat_caption", "incremental")


def resolve_memory_mode(routing: str, explicit: str | None = None) -> str:
    # Resolve the shared default; routing never selects a memory algorithm.
    if routing != ROUTING_KEYWORD:
        raise ValueError("Only question-based keyword routing is supported")
    mode = DEFAULT_MEMORY_MODE if explicit is None else explicit
    if mode not in MEMORY_MODE_CHOICES:
        raise ValueError(f"Unknown memory_mode {mode!r}; expected one of {MEMORY_MODE_CHOICES}")
    return mode


class EntityResolvedHubAndSpokeEvaluator(IncrementalHubAndSpokeEvaluator):
    # Use chunk-local entity IDs while retaining the existing evaluator API.

    def _make_memory(self):
        for flag in ("expand_retrieval", "expand_co_occurrence", "expand_next_action", "merge_spokes"):
            if getattr(self, flag, True) is not True:
                raise ValueError(
                    f"EntityResolvedMemory does not implement the {flag!r} "
                    "ablation switch; run it with the default."
                )
        memory = EntityResolvedMemory(
            embed_model=self.embed_model,
            embed_device=self.embed_device,
            sim_threshold=self.sim_threshold,
            dynamic_top_k_max=self.dynamic_top_k_max,
        )
        memory.spoke_attach_threshold = getattr(self, "spoke_attach_threshold", ENTITY_LINK_THRESHOLD)
        return memory

    def build_memory_from_chunks(self, chunks, question: str | None = None):
        # Preserve chunk selection/caption order and store actual chunk IDs.
        memory = self._make_memory()
        is_count = question is not None and _is_count_question(question)
        caption_prompt = self._caption_prompt_for(question)
        strided = [chunks[i] for i in range(0, len(chunks), self.extract_every_n_chunks)]
        cap = self.count_question_max_chunks if is_count else self.max_extraction_chunks
        if cap and len(strided) > cap:
            step = len(strided) / cap
            strided = [strided[int(i * step)] for i in range(cap)]
        sampled_ids = {id(chunk) for chunk in strided}
        for chunk in chunks:
            if id(chunk) not in sampled_ids:
                chunk.frames = []
        self._pre_caption_chunks(strided, caption_prompt)
        for chunk in strided:
            caption, timestamp = self._caption_chunk(chunk, caption_prompt)
            chunk.frames = []
            if caption:
                memory.update(caption, timestamp, chunk_id=chunk.chunk_index)
        return memory


def evaluator_class_for(memory_mode: str):
    if memory_mode == DEFAULT_MEMORY_MODE:
        return EntityResolvedHubAndSpokeEvaluator
    if memory_mode == "incremental":
        return IncrementalHubAndSpokeEvaluator
    if memory_mode == "flat_caption":
        from .hub_and_spoke import FlatCaptionEvaluator
        return FlatCaptionEvaluator
    if memory_mode == "hub_spoke":
        from .hub_and_spoke import HubAndSpokeEvaluator
        return HubAndSpokeEvaluator
    raise ValueError(f"Unknown memory_mode {memory_mode!r}; expected one of {MEMORY_MODE_CHOICES}")
