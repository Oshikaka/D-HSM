"""Question-based memory routing for streaming video understanding."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Bucket names, also used as the value logged into each result row.
STRONG_MEMORY = "strong_memory"
STRONG_RECENT = "strong_recent"
AMBIGUOUS = "ambiguous"

# Production routing uses only question/option text.
ROUTING_KEYWORD = "keyword"
ROUTING_CHOICES = (ROUTING_KEYWORD,)

# Similarity floor applied to the AMBIGUOUS bucket. 
DEFAULT_GATE_STRICT_SIM = 0.55


# ---------------------------------------------------------------------------
# Cue inventories
# ---------------------------------------------------------------------------

_MEMORY_CUES: dict[str, str] = {
    "just now":     r"\bjust\s+now\b",
    "so far":       r"\bso\s+far\b",
    "in total":     r"\bin\s+total\b",
    "until now":    r"\b(?:until|up\s+to|by)\s+now\b",
    "previously":   r"\b(?:previously|earlier|already|moments?\s+ago|a\s+moment\s+ago)\b",
    "before/after": r"\b(?:before|after)\b",
    "summarize":    r"\bsummar\w*",
    "how many times": r"\b(?:how\s+many\s+times|how\s+often)\b",
    "did":          r"\bdid\b",
    "was/were":     r"\b(?:was|were)\b",
    "had":          r"\bhad\b",
    "perfect":      r"\b(?:has|have|had)\s+(?:\w+\s+)?been\b",
    "happened":     r"\bhappened\b",
}

_RECENT_CUES: dict[str, str] = {
    "right now":       r"\bright\s+now\b",
    "currently":       r"\b(?:currently|current)\b",
    "at this moment":  r"\bat\s+(?:this|the)\s+moment\b",
    "is being":        r"\b(?:is|are|am)\s+being\b",
    "about/going to":  r"\b(?:about\s+to|going\s+to)\b",
    "present progressive": r"\b(?:is|are|am)\s+(?:he|she|it|they|we|you|i)?\s*\w+ing\b",
}


_WEAK_MEMORY_CUES = frozenset({"before/after"})

_MEMORY_RE = {name: re.compile(pat, re.IGNORECASE) for name, pat in _MEMORY_CUES.items()}
_RECENT_RE = {name: re.compile(pat, re.IGNORECASE) for name, pat in _RECENT_CUES.items()}


# ---------------------------------------------------------------------------
# Prompt-template scaffolding
# ---------------------------------------------------------------------------
_SCAFFOLD_PATTERNS = (
    r"you(?:'re|r)?\s+watching\s+a\s+video.*?(?=now,\s*answer|$)",
    r"you(?:'re|r)?\s+watching\s+a\s+tutorial\s+video.*?(?=the\s+following\s+is|$)",
    r"you(?:'re|r)?\s+responsible\s+of\s+answering.*?(?=\n|$)",
    r"you(?:'re|r)?\s+task\s+is\s+to.*?(?=\n|$)",
    r"now,\s*answer\s+the\s+following\s+question:\s*",
    r"answer\s+the\s+following\s+question:\s*",
    r"respond\s+only\s+with.*?$",
    r"answer\s+only\s+with.*?$",
    r"provide\s+your\s+answer\s+as.*?$",
    r"do\s+not\s+include\s+any\s+additional\s+text.*?$",
    r"decide\s+whether\s+existing\s+visual\s+content.*?$",
    r"one\s+complete\s+motion\s+counts\s+as\s+one\.?",
    r"the\s+person\s+performing\s+this\s+kind\s+of\s+action.*?$",
    r"the\s+following\s+question\s+are\s+relevant\s+to\s+the\s+latest\s+frames.*?$",
    r"^\s*options?\s*:.*$",
    r"^\s*question\s*:\s*",
)
_SCAFFOLD_RE = [
    re.compile(p, re.IGNORECASE | re.MULTILINE | re.DOTALL) for p in _SCAFFOLD_PATTERNS
]


def strip_prompt_scaffolding(text: str) -> str:
    out = text or ""
    for rx in _SCAFFOLD_RE:
        out = rx.sub(" ", out)
    return re.sub(r"\s+", " ", out).strip()


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateDecision:
    """Outcome of keyword gating for one question."""

    needs_memory: bool
    bucket: str
    memory_cues: tuple[str, ...] = field(default=())
    recent_cues: tuple[str, ...] = field(default=())

    @property
    def strict_evidence(self) -> bool:
        """True when retrieval should run under the stricter threshold."""
        return self.bucket == AMBIGUOUS

    def as_metadata(self) -> dict[str, object]:
        return {
            "gate_needs_memory": self.needs_memory,
            "gate_bucket": self.bucket,
            "gate_memory_cues": list(self.memory_cues),
            "gate_recent_cues": list(self.recent_cues),
        }


def gate_question(question: str, options: list[str] | None = None) -> GateDecision:
    # Decide whether ``question`` needs memory retrieval, from text alone.

    stem = strip_prompt_scaffolding(question)

    mem = tuple(name for name, rx in _MEMORY_RE.items() if rx.search(stem))
    rec = tuple(name for name, rx in _RECENT_RE.items() if rx.search(stem))

    if mem and not (set(mem) <= _WEAK_MEMORY_CUES and rec):
        return GateDecision(True, STRONG_MEMORY, mem, rec)
    if rec:
        return GateDecision(False, STRONG_RECENT, mem, rec)

    if options:

        opts = [strip_prompt_scaffolding(str(o)) for o in options]
        quorum = max(2, (len(opts) + 1) // 2)
        opt_mem = tuple(
            name for name, rx in _MEMORY_RE.items()
            if sum(1 for o in opts if rx.search(o)) >= quorum
        )
        if opt_mem and not set(opt_mem) <= _WEAK_MEMORY_CUES:
            return GateDecision(True, STRONG_MEMORY, opt_mem, rec)


    return GateDecision(True, AMBIGUOUS, mem, rec)


def needs_memory(question: str, options: list[str] | None = None) -> bool:
    # Convenience wrapper returning only the binary decision.
    return gate_question(question, options).needs_memory
