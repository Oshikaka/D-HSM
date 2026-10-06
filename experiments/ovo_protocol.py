"""Label-free OVO MCQ routing and uniform answer policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

from dhsm.retrieval_gate import ROUTING_KEYWORD, GateDecision, gate_question
from experiments.ovo_bench import (
    MCQ_PROMPT_UNIFORM,
    build_mcq_prompt,
    validate_mcq_prompt_policy,
)

HISTORY_DHSM = "dhsm"
HISTORY_RECENT_ONLY = "recent_only"
HISTORY_MODES = (HISTORY_DHSM, HISTORY_RECENT_ONLY)
DEFAULT_MEMORY_FLOOR = 0.60
DEFAULT_OVO_GATE_STRICT_SIM = 0.60

# One fixed instruction for all MCQs, inserted only after retrieval. This is
# method guidance beyond the benchmark's original prompt. No task label or
# uncertainty-option detector controls whether it appears.
UNIFORM_ABSTENTION_INSTRUCTION = (
    'Choose a content option only when the available evidence directly answers the question. '
    'If the evidence is insufficient, only indirectly related, or retrieval reports '
    '"未找到匹配节点", choose an option indicating that the question cannot be answered, '
    'if such an option is available.'
)


@dataclass(frozen=True)
class AnswerRoute:
    method_family: str
    policy: str
    use_memory: bool
    include_no_match_signal: bool
    num_options: int
    min_evidence_sim: float | None = None
    gate: GateDecision | None = None
    answer_instruction: str | None = None


def answer_route_for(
    question: str,
    options: list[str],
    *,
    routing: str = ROUTING_KEYWORD,
    memory_floor: float = DEFAULT_MEMORY_FLOOR,
    gate_strict_sim: float = DEFAULT_OVO_GATE_STRICT_SIM,
    mcq_prompt_policy: str = MCQ_PROMPT_UNIFORM,
    history_mode: str = HISTORY_DHSM,
) -> AnswerRoute:
    # Choose retrieval using observed question/options, never task or GT.
    build_mcq_prompt(question, options)
    validate_mcq_prompt_policy(mcq_prompt_policy)
    if routing != ROUTING_KEYWORD:
        raise ValueError("Only keyword routing is supported.")
    if history_mode not in HISTORY_MODES:
        raise ValueError(f"Unknown history mode {history_mode!r}.")
    for value in (memory_floor, gate_strict_sim):
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Evidence floors must be finite numbers between 0 and 1.")
    decision = gate_question(question, options)
    use_memory = decision.needs_memory and history_mode == HISTORY_DHSM
    minimum = (gate_strict_sim if decision.strict_evidence else memory_floor) if use_memory else None
    uniform = mcq_prompt_policy == MCQ_PROMPT_UNIFORM
    return AnswerRoute(
        method_family="hub_and_spoke_memory" if use_memory else "recent_window",
        policy="uniform_abstention" if uniform else "standard_mcq",
        use_memory=use_memory,
        include_no_match_signal=uniform,
        num_options=len(options),
        min_evidence_sim=minimum,
        gate=decision,
        answer_instruction=UNIFORM_ABSTENTION_INSTRUCTION if uniform else None,
    )
