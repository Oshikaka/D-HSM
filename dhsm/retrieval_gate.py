"""Keyword-based retrieval gating for D-HSM.

Implements the first stage of the paper's two-stage retrieval funnel:

    "It first applies keyword-based gating to determine whether memory
     retrieval is needed.  When retrieval is triggered, D-HSM selects a
     memory subset according to the similarity between the question and
     memory entries."

The gate reads ONLY the question text (plus, optionally, the answer options).
It never looks at benchmark task labels, `required_ability` fields, or any
other dataset annotation, so the same code path applies to an arbitrary
streaming question.

Three outcomes
--------------
``STRONG_MEMORY``  an explicit retrospective cue is present ("just now",
                   "so far", past-tense "did/was/were", "before/after", ...).
                   Retrieval runs at the caller's base similarity threshold.

``STRONG_RECENT``  an unambiguous present-moment cue is present and no
                   retrospective cue is ("right now", "currently",
                   "is being", present progressive, ...).  Retrieval is
                   skipped entirely — the recent visual window suffices.

``AMBIGUOUS``      neither, e.g. "Where is the microwave?".  Retrieval runs
                   but under a STRICTER similarity threshold, so history is
                   injected only when memory actually holds a closely
                   matching entry.  This is the "retrieve little or no
                   history" behaviour the paper asks for, realised as
                   evidence gating rather than as a lexical decision.

Why the ambiguous bucket exists
-------------------------------
On OVO-Bench, 35.7% of Backward (EPM/HLD) questions are lexically
indistinguishable from Real-Time ones:

    Backward  "Where is the microwave?"     "What color is the broom?"
    Real-Time "Where is the man sitting?"   "What is the color of the dog?"

The difference is semantic, not lexical: the Backward version asks where an
object was *last seen* once it has left the frame.  No keyword rule can
separate these, so forcing a binary lexical decision necessarily trades one
error for the other.  The gate therefore defers, and lets cosine similarity
against memory make the call — "Where is the microwave?" matches an
``O<N> (microwave)`` hub strongly, a Real-Time spatial question about the
current frame does not.

Cue calibration
---------------
Cue precision measured on the full public annotation sets
(OVO-Bench `ovo_bench_new.json`, n=1468 Backward+Real-Time;
StreamingBench `questions_real.json`, n=2495 episodic+working memory).
See `experiments/verify_retrieval_gate.py` to reproduce.

    bucket          OVO n / purity      StreamingBench n / purity
    STRONG_MEMORY   429 / 94.6%         718 / 93.7%
    STRONG_RECENT   175 / 98.3%         954 / 95.3%
    AMBIGUOUS       864 / --            823 / --

    false negative (history question sent to recent-only, the costly error):
        OVO 0.48%   StreamingBench 4.50%
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Bucket names, also used as the value logged into each result row.
STRONG_MEMORY = "strong_memory"
STRONG_RECENT = "strong_recent"
AMBIGUOUS = "ambiguous"

# Routing modes exposed by the evaluators.
#   keyword     — gate on question text (this module); the paper's mechanism.
#   task_label  — read the benchmark's own task / required_ability annotation.
#                 Kept only to reproduce the pre-gate numbers; it consumes
#                 ground-truth metadata at inference time.
ROUTING_KEYWORD = "keyword"
ROUTING_TASK_LABEL = "task_label"
ROUTING_CHOICES = (ROUTING_KEYWORD, ROUTING_TASK_LABEL)

# Similarity floor applied to the AMBIGUOUS bucket.  0.55 is the value
# `hub_and_spoke.SIM_THRESHOLD` documents as the point where bge-small-en-v1.5
# cosine rises above its baseline for unrelated text (~0.45-0.55), i.e. the
# smallest score at which a memory entry is plausibly *about* the question.
DEFAULT_GATE_STRICT_SIM = 0.55


# ---------------------------------------------------------------------------
# Memory implementation behind the gate
# ---------------------------------------------------------------------------
# Both routing modes run on dhsm/hub_and_spoke_incremental.py, the Algorithm 2
# variant: hubs, spokes and edges carry per-chunk provenance, so evidence can
# be retired when a chunk leaves the window instead of accumulating forever.
# That is the memory the paper describes.
#
# Pinning both modes to the same implementation is deliberate: routing is the
# only variable the A/B is meant to move, so a keyword-vs-task_label delta is
# attributable to the gate alone.  Selecting the memory implementation from
# the routing mode would confound the two.  Pass --memory_mode explicitly to
# run either arm on dhsm/hub_and_spoke.py (hub_spoke) instead.
GATED_MEMORY_MODE = "incremental"
BASELINE_MEMORY_MODE = "incremental"
MEMORY_MODE_CHOICES = ("hub_spoke", "flat_caption", "incremental")


def resolve_memory_mode(routing: str, explicit: str | None = None) -> str:
    """Memory implementation to use, given the routing mode.

    ``explicit`` is the value the caller passed on the command line; when it
    is None both routing modes resolve to the incremental (Algorithm 2)
    memory, so an A/B over ``--routing`` isolates the gate.
    """
    if explicit is not None:
        return explicit
    return GATED_MEMORY_MODE if routing == ROUTING_KEYWORD else BASELINE_MEMORY_MODE


def evaluator_class_for(memory_mode: str):
    """Evaluator class implementing ``memory_mode``.

    Imported lazily so that importing this module stays free of numpy and the
    memory implementations — ``verify_retrieval_gate.py`` audits the gate on
    annotation files alone and should not need them.
    """
    if memory_mode == "flat_caption":
        from .hub_and_spoke import FlatCaptionEvaluator

        return FlatCaptionEvaluator
    if memory_mode == "incremental":
        from .hub_and_spoke_incremental import IncrementalHubAndSpokeEvaluator

        return IncrementalHubAndSpokeEvaluator
    if memory_mode == "hub_spoke":
        from .hub_and_spoke import HubAndSpokeEvaluator

        return HubAndSpokeEvaluator
    raise ValueError(
        f"Unknown memory_mode {memory_mode!r}; expected one of {MEMORY_MODE_CHOICES}."
    )


# ---------------------------------------------------------------------------
# Cue inventories
# ---------------------------------------------------------------------------
# Retrospective cues.  Measured precision for "this question needs history":
# 93-100% on both benchmarks, except "before/after" (see _WEAK_MEMORY_CUES).
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

# Present-moment cues.  Measured precision for "the recent window suffices":
# 95-100% on both benchmarks.
_RECENT_CUES: dict[str, str] = {
    "right now":       r"\bright\s+now\b",
    "currently":       r"\b(?:currently|current)\b",
    "at this moment":  r"\bat\s+(?:this|the)\s+moment\b",
    "is being":        r"\b(?:is|are|am)\s+being\b",
    "about/going to":  r"\b(?:about\s+to|going\s+to)\b",
    # An optional pronoun subject may sit between the copula and the participle
    # ("what is he doing"). Only pronouns are allowed through: a permissive
    # gap would match "where is the building" on the -ing ending alone.
    "present progressive": r"\b(?:is|are|am)\s+(?:he|she|it|they|we|you|i)?\s*\w+ing\b",
}

# "before"/"after" also appear in purely present-tense questions such as
# "What is this person going to do after cleaning the item?" (OVO FPD), where
# they order two *current* events rather than point into history.  When the
# only retrospective evidence is one of these and an unambiguous present-moment
# cue co-occurs, the present reading wins.
_WEAK_MEMORY_CUES = frozenset({"before/after"})

_MEMORY_RE = {name: re.compile(pat, re.IGNORECASE) for name, pat in _MEMORY_CUES.items()}
_RECENT_RE = {name: re.compile(pat, re.IGNORECASE) for name, pat in _RECENT_CUES.items()}


# ---------------------------------------------------------------------------
# Prompt-template scaffolding
# ---------------------------------------------------------------------------
# The gate must see the question, not the benchmark's instruction wrapper.
# OVO's REC template alone contributes "how many times", "did" and "in total"
# to every single question it wraps, and the MCQ templates add boilerplate
# such as "Do not include any additional text".  Strip the known wrappers so
# gating decisions come from the question the user actually asked.
_SCAFFOLD_PATTERNS = (
    # OVO's templates write "You're task is to ..." — match the apostrophe
    # form as well as "your", or the REC wrapper leaks "how many times",
    # "did" and "in total" into every question it wraps.
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
    r"for\s+hld,\s*choose\s+a\s+content\s+option.*?$",
    r"^\s*options?\s*:.*$",
    r"^\s*question\s*:\s*",
)
_SCAFFOLD_RE = [
    re.compile(p, re.IGNORECASE | re.MULTILINE | re.DOTALL) for p in _SCAFFOLD_PATTERNS
]


def strip_prompt_scaffolding(text: str) -> str:
    """Remove known benchmark prompt-template boilerplate from ``text``."""
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
    """Decide whether ``question`` needs memory retrieval, from text alone.

    ``options`` are accepted because multiple-choice option text carries the
    same tense signal as the stem (StreamingBench "Clips Summarize" options
    are uniformly past tense), but they are only consulted when the stem
    itself yields no cue at all — option text is noisier than the question.
    """
    stem = strip_prompt_scaffolding(question)

    mem = tuple(name for name, rx in _MEMORY_RE.items() if rx.search(stem))
    rec = tuple(name for name, rx in _RECENT_RE.items() if rx.search(stem))

    if mem and not (set(mem) <= _WEAK_MEMORY_CUES and rec):
        return GateDecision(True, STRONG_MEMORY, mem, rec)
    if rec:
        return GateDecision(False, STRONG_RECENT, mem, rec)

    if options:
        # A cue in the options counts only when it RECURS across them.  The
        # signal this fallback exists for is a uniformly past-tense option set
        # (StreamingBench "Clips Summarize"), and requiring recurrence keeps
        # it.  A cue in a single option is noise whenever the options are
        # verbatim content rather than paraphrased answers: OVO OCR lists the
        # strings visible on screen, so an option reading "WHAT DID I JUST
        # HEAR" would otherwise route "What is the text on the screen now?"
        # to memory on the strength of its "did".
        opts = [strip_prompt_scaffolding(str(o)) for o in options]
        quorum = max(2, (len(opts) + 1) // 2)
        opt_mem = tuple(
            name for name, rx in _MEMORY_RE.items()
            if sum(1 for o in opts if rx.search(o)) >= quorum
        )
        if opt_mem and not set(opt_mem) <= _WEAK_MEMORY_CUES:
            return GateDecision(True, STRONG_MEMORY, opt_mem, rec)

    # Lexically undecidable.  Retrieve — a missed history question costs ~6.3x
    # a spurious retrieval (Fig. 1: 62.82 vs 55.66 on Backward, 78.98 vs 77.84
    # on Real-Time) — but under the stricter evidence threshold.
    return GateDecision(True, AMBIGUOUS, mem, rec)


def needs_memory(question: str, options: list[str] | None = None) -> bool:
    """Convenience wrapper returning only the binary decision."""
    return gate_question(question, options).needs_memory


if __name__ == "__main__":  # self-test
    # (question, expected bucket)
    CASES = [
        # --- retrospective wording: retrieve ---
        ("What did the person just do?", STRONG_MEMORY),
        ("How many times in total have movie clips been inserted so far?", STRONG_MEMORY),
        ("Which of the following best summarizes the actions taken just now?", STRONG_MEMORY),
        ("Who did I communicate to when chopping egg plants?", STRONG_MEMORY),
        ("What does the person do after load the wheel", STRONG_MEMORY),
        ("What was the monster holding?", STRONG_MEMORY),
        # --- present moment: skip retrieval ---
        ("What is the person doing right now?", STRONG_RECENT),
        ("What action is being performed right now?", STRONG_RECENT),
        ("What is this person about to do?", STRONG_RECENT),
        ("What is he doing?", STRONG_RECENT),
        ("What type of traffic signal is present right now?", STRONG_RECENT),
        # "after" only orders two present events here, so the present reading wins.
        ("What is this person going to do after cleaning the item?", STRONG_RECENT),
        # --- lexically undecidable: retrieve, but evidence-gated ---
        ("Where is the microwave?", AMBIGUOUS),
        ("What color is the broom?", AMBIGUOUS),
        ("How many red cups are on the ground?", AMBIGUOUS),
        ("What is under the monkey?", AMBIGUOUS),
    ]
    failures = 0
    for question, expected in CASES:
        got = gate_question(question)
        ok = got.bucket == expected
        failures += not ok
        print(f"  {'ok  ' if ok else 'FAIL'}  {got.bucket:<14} (want {expected:<14}) {question}")

    # Prompt scaffolding must not leak cues into the gate: OVO's REC wrapper
    # alone contributes "how many times", "did" and "in total".
    from_template = (
        "You're watching a video in which people may perform a certain type of "
        "action repetively.\nYou're task is to count how many times have different "
        "people in the video perform this kind of action in total.\n"
        "One complete motion counts as one.\n"
        "Now, answer the following question: What is the man doing right now?\n"
        "Provide your answer as a single number."
    )
    stripped = strip_prompt_scaffolding(from_template)
    scaffold_ok = gate_question(from_template).bucket == STRONG_RECENT
    failures += not scaffold_ok
    print(f"\n  {'ok  ' if scaffold_ok else 'FAIL'}  scaffolding stripped -> {stripped!r}")

    # Routing selects the memory implementation, so pin that down too.
    from itertools import product

    EXPECTED = {
        (ROUTING_KEYWORD, None): GATED_MEMORY_MODE,
        (ROUTING_TASK_LABEL, None): BASELINE_MEMORY_MODE,
    }
    extra = 0
    for routing, explicit in product(ROUTING_CHOICES, (None,) + MEMORY_MODE_CHOICES):
        got = resolve_memory_mode(routing, explicit)
        want = EXPECTED.get((routing, explicit), explicit)
        ok = got == want
        failures += not ok
        extra += 1
        if not ok:
            print(f"  FAIL  resolve_memory_mode({routing!r}, {explicit!r}) = {got!r}, want {want!r}")
    print(f"\n  ok    resolve_memory_mode over {extra} routing/mode combinations")

    total = len(CASES) + 1 + extra
    print(f"\n{total - failures}/{total} passed")
    raise SystemExit(1 if failures else 0)
