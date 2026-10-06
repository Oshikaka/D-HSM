"""Hub-and-Spoke: caption-driven streaming video memory with semantic retrieval"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# sys.path fix: ensure the conda env's site-packages take precedence over
# ~/.local/lib/python3.x/site-packages.
# ---------------------------------------------------------------------------
import sys as _sys
import site as _site

if hasattr(_site, "getusersitepackages"):
    _user_site = _site.getusersitepackages()
    _in_user = [p for p in _sys.path if p.startswith(_user_site)]
    _not_user = [p for p in _sys.path if not p.startswith(_user_site)]
    _sys.path[:] = _not_user + _in_user
# ---------------------------------------------------------------------------

import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_NODES = 300
# Threshold calibrated for bge-small-en-v1.5, whose cosine baseline for
# unrelated text is around 0.45-0.55.
SIM_THRESHOLD = 0.55
ENTITY_LINK_THRESHOLD = 0.65   # bge baseline shift; was 0.45 for MiniLM

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
DYNAMIC_TOP_K_MAX = 12
_DYNAMIC_TOP_K_MODES = {"dynamic", "auto", "adaptive"}


def _validate_dynamic_top_k_max(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("dynamic_top_k_max must be a positive integer")
    return value


_TEMPORAL_KEYWORDS = {
    "after", "before", "then", "next", "when", "order", "sequence",
    "first", "last", "earlier", "later", "previously", "follow",
    "subsequent", "happen", "did", "while", "during", "step",
}

_RECENCY_PHRASES = (
    "just now",
    "so far",
    "summariz",
    "primary focus",
)
_RECENT_EVENTS_K = 6

_COUNT_PHRASES = (
    "how many times",
    "how often",
    "number of times",
    "count how many",
    "how many",       
)

_COUNT_STOP_WORDS_RAW = (
    "they", "them", "their", "this", "that", "these", "those",
    "what", "when", "where", "which", "while", "until", "after",
    "before", "during", "from", "into", "with", "about",
    "have", "having", "had", "has", "many", "much", "more",
    "time", "times", "video", "scene", "scenes",
    "person", "people", "thing", "things", "happen", "happens",
    "different", "kind", "kinds", "total", "complete", "completes",
    "motion", "count", "counts", "answer", "answers",
    "question", "questions", "follow", "follows", "given",
    "perform", "performs", "show", "shows", "watch", "watches",
    "action", "actions", "event", "events",
    "does", "doing", "doer", "type", "types",
)


# ---------------------------------------------------------------------------
# Caption prompt with stable O<N>/P<N> IDs.
# ---------------------------------------------------------------------------

CAPTION_PROMPT = """\
Describe this video segment for question answering.

Output format (six lines, in this order, each starting with the section name
and a colon). Write NONE (no brackets) if a section has nothing to report.

OBJECTS: <semicolon-separated items>
PEOPLE: <semicolon-separated items>
ACTIONS: <semicolon-separated items>
TEXT: <semicolon-separated items>
SPATIAL: <semicolon-separated items>
EVENT: <one sentence>

Example:
OBJECTS: O1 (red mug); O2 (black laptop); O3 (wooden desk)
PEOPLE: P1 (blue shirt + woman): typing on laptop
ACTIONS: P1 types on O2 with both hands
TEXT: NONE
SPATIAL: O2 is on O3; O1 is left of O2
EVENT: A woman in a blue shirt types on her laptop at a wooden desk.

Section rules:
- OBJECTS: list AT MOST 8 items, each as "O<N> (one-attribute description)"
  where the attribute is color/size/material/state. Use stable IDs O1, O2, ...
  and REUSE the same ID for the same object across segments. STOP after 8 even
  if more are visible. Skip generic background.
- PEOPLE: each person as "P<N> (clothing color + role/age): brief action".
  Use stable IDs P1, P2, ... and reuse the same ID for the same person across
  segments. Always include clothing color when visible.
- ACTIONS: each as "P<N> verb-s O<M> [with tool] [to/towards P<K> or location]".
  Refer to objects by their O<N> IDs and people by their P<N> IDs. Include the
  recipient/target whenever someone hands, gives, shows, talks to, gestures
  to, or looks at another person.
- TEXT: any visible text, verbatim; NONE if no readable text.
- SPATIAL: 3-6 positional relations between named entities (P<N>/O<N>) only,
  as "<A> is left/right/above/below/behind/in front of <B>".
- EVENT: one complete sentence — the most important thing happening in this
  segment.

Hard rules:
- Your first line MUST start with "OBJECTS:".
- Describe ONLY what is visually present. Never infer off-screen events.
- Keep PEOPLE and OBJECT IDs consistent across segments — if P1 was the
  "blue-shirt cook" earlier and O1 was the "red mug" earlier, keep using
  P1 and O1 for them now."""


_SECTION_TO_TYPE: dict[str, str] = {
    "OBJECTS": "entity",
    "PEOPLE":  "entity",
    "ACTIONS": "action",
    "TEXT":    "text_ocr",
    "SPATIAL": "spatial",
    "EVENT":   "event",
}


# ---------------------------------------------------------------------------
# TextNode (with co-occurrence and temporal-next edges)
# ---------------------------------------------------------------------------

@dataclass
class TextNode:
    text: str
    node_type: str
    timestamp: float        # last-seen
    first_seen: float       # never updated
    count: int = 1
    embedding: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    entity_idx: int | None = None      # spoke → hub link (action/spatial/event nodes)
    co_entities: set[int] = field(default_factory=set)  # entity-entity co-occurrence
    next_action: int | None = None     # next action node sharing this entity_idx
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    context: str
    matched_nodes: bool
    signal: str | None = None
    hit_count: int = 0
    # Cosine scores behind the decision. 
    hit_sims: tuple[float, ...] = ()
    candidate_sims: tuple[float, ...] = ()


NO_MATCHING_NODE_SIGNAL = "未找到匹配节点"


def round_sims(retrieval: "RetrievalResult | None", attr: str) -> list[float] | None:
    """Cosine scores off a RetrievalResult, rounded to keep result JSONL small.

    Used by the evaluators to record what the similarity filter saw, so that a
    sweep over the gate's evidence floor can be replayed from a finished run.
    """
    if retrieval is None:
        return None
    return [round(float(s), 4) for s in getattr(retrieval, attr, ()) or ()]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    s = int(seconds)
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


# Crude stemmer — strip common English suffixes. Good enough for keyword overlap without pulling in nltk.
_STEM_SUFFIX_RE = re.compile(r"(ings?|ied|ed|es|s)$", re.IGNORECASE)
_STEM_UNDOUBLE_SUFFIXES = {"ing", "ings", "ed", "ied"}
_STEM_RESTORE_E_SUFFIXES = {"ing", "ings", "ed", "es"}

def _stem(word: str) -> str:
    """Crude rule-based stemmer with two Porter-style fixups:

      1. Undouble a doubled consonant after stripping -ing/-ed:
         "chopping" → "chopp" → "chop", "running" → "runn" → "run".

      2. Restore a silent "e" after stripping -ing/-ed/-es when the resulting
         stem is a length-3 CVC pattern, so "diving"/"dives"/"dive" all map
         to "dive".  Without this, the question (which uses the bare form)
         would not match captions (which use inflections).

    These rules are applied EXCLUSIVELY of each other — undoubling implies
    the doubled consonant was added by the suffix, in which case CVC
    restoration would be wrong (e.g. "running" → "runn" → "run", NOT "rune").
    """
    w = word.lower()
    m = _STEM_SUFFIX_RE.search(w)
    if not m:
        return w
    suffix = m.group(1).lower()
    w = w[: m.start()]
    undoubled = False
    if suffix in _STEM_UNDOUBLE_SUFFIXES and len(w) >= 3:
        if w[-1] == w[-2] and w[-1] not in "aeiou":
            w = w[:-1]
            undoubled = True
    if (
        not undoubled
        and suffix in _STEM_RESTORE_E_SUFFIXES
        and len(w) == 3
        and w[-1] not in "aeiou"
        and w[-2] in "aeiou"
        and w[-3] not in "aeiou"
        and w[-1] not in "wxy"
    ):
        w = w + "e"
    return w or word.lower()


def _keywords(text: str) -> set[str]:
    return {_stem(w) for w in re.findall(r"\b\w{4,}\b", text.lower())}


_COUNT_STOP_STEMS = frozenset(_stem(w) for w in _COUNT_STOP_WORDS_RAW)


def _is_count_question(question: str) -> bool:
    ql = question.lower()
    return any(p in ql for p in _COUNT_PHRASES)


# Pull the activity phrase out of a count question.
_COUNT_FOCUS_RE = re.compile(
    r"how many times\s+"
    r"(?:do(?:es)?|did|has|have|are|is|will)?\s*"
    r"(?:they|he|she|the\s+\S+|people|someone)?\s*"
    r"([^?.\n]+)",
    re.IGNORECASE,
)

def make_activity_aware_caption_prompt(activity: str) -> str:
    # Augment CAPTION_PROMPT with a count-focused activity directive.

    a = activity.strip().strip(".?! ").lower()
    suffix = (
        "\n\n"
        "ACTIVITY FOCUS — counting rule for this video:\n"
        f'This video is being analysed to count COMPLETE instances of: "{a}".\n'
        "\n"
        "Strict rules for the ACTIONS line:\n"
        f'  1. Only write "{a}" if THIS exact chunk shows the PEAK / DEFINING '
        f'moment of one complete instance — the moment the action is being '
        f'executed at its core (e.g., the actual dive entering the water, the '
        f'actual hit landing, the actual lift reaching its top, the actual '
        f'door closing).\n'
        f'  2. DO NOT write "{a}" for chunks that show only:\n'
        f'     - preparation (standing on the board, raising the bat, '
        f'reaching toward the object),\n'
        f'     - the immediate aftermath (swimming away, walking off, '
        f'resetting, recovering),\n'
        f'     - partial overlap where the action is mostly elsewhere,\n'
        f'     - a static pause between attempts.\n'
        f'     For those chunks describe what you actually see with a '
        f'DIFFERENT verb (standing, walking, climbing, swimming, holding, '
        f'looking, etc.).\n'
        f'  3. If one complete instance of "{a}" spans 2-3 consecutive '
        f'chunks, label ONLY the chunk showing the most defining moment. '
        f'Do not repeat the label on the surrounding chunks.\n'
        f'  4. When you DO label this chunk as "{a}", use the EXACT phrase '
        f'"{a}" in the ACTIONS line, e.g. "P1 performs {a}" or "P1 is {a}".'
    )
    return CAPTION_PROMPT + suffix


_COUNT_FOCUS_LEADING_SCAFFOLD = re.compile(
    r"^(?:in total|in all|in the video|total(?:ly)?|so far|now|"
    r"have|has|had|been|did|do|does|are|is|was|were)\s+",
    re.IGNORECASE,
)
_COUNT_FOCUS_TRAILING_SCAFFOLD = re.compile(
    r"\s+(?:so far|in total|up to now|to date|in the video|by now)\b.*$",
    re.IGNORECASE,
)


def _extract_count_focus(question: str) -> str:
    # Return the activity phrase from a count question.

    marker = "answer the following question:"
    idx = question.lower().rfind(marker)
    haystack = question[idx + len(marker):] if idx >= 0 else question

    matches = list(_COUNT_FOCUS_RE.finditer(haystack))
    if not matches:
        return question
    focus = matches[-1].group(1).strip().rstrip("?. ")
    # Strip leading scope adverbs / auxiliaries until the phrase starts with
    # a content word.  Repeat to handle stacked tokens like "in total have".
    for _ in range(4):
        new_focus = _COUNT_FOCUS_LEADING_SCAFFOLD.sub("", focus, count=1)
        if new_focus == focus:
            break
        focus = new_focus
    # Drop trailing temporal scope phrases.
    focus = _COUNT_FOCUS_TRAILING_SCAFFOLD.sub("", focus).strip().rstrip("?. ")
    return focus or question


def _content_stems(text: str) -> set[str]:
    # Question/action content stems with stop words removed.

    return {s for s in _keywords(text) if len(s) >= 3 and s not in _COUNT_STOP_STEMS}


# ---------------------------------------------------------------------------
# Embedder factory
# ---------------------------------------------------------------------------

_EMBEDDER_CACHE: dict[tuple, Any] = {}

def _load_embedder(model_name: str, device: str = "cpu"):
    key = (model_name, device)
    if key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[key]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required to preserve the embedding model's "
            "pooling configuration. Install requirements.txt before evaluation."
        ) from exc
    name = model_name if "/" in model_name else f"sentence-transformers/{model_name}"
    embedder = SentenceTransformer(name, device=device)
    _EMBEDDER_CACHE[key] = embedder
    return embedder


# ---------------------------------------------------------------------------
# Hub-and-Spoke Memory
# ---------------------------------------------------------------------------

# Pattern matching the leading stable ID, e.g. "P1", "P12", "O1", "O7".
_STABLE_ID_RE = re.compile(r"^([PO])(\d+)\b")


class HubAndSpokeMemory:
    """Compact text-based video memory with hub-and-spoke retrieval links."""

    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        sim_threshold: float = SIM_THRESHOLD,
        max_nodes: int = MAX_NODES,
        embed_device: str = "cpu",
        expand_retrieval: bool = True,
        expand_co_occurrence: bool = True,
        expand_next_action: bool = True,
        merge_spokes: bool = True,
        dynamic_top_k_max: int = DYNAMIC_TOP_K_MAX,
    ) -> None:
        self._embed_model_name = embed_model
        self._embed_device = embed_device
        self.sim_threshold = sim_threshold
        self.max_nodes = max_nodes
        # Ablation switches:
        self.expand_retrieval = expand_retrieval
        self.expand_co_occurrence = expand_co_occurrence
        self.expand_next_action = expand_next_action
        self.merge_spokes = merge_spokes
        self.dynamic_top_k_max = _validate_dynamic_top_k_max(dynamic_top_k_max)
        self.nodes: list[TextNode] = []
        self._dedup: dict[str, int] = {}
        self.timeline: list[tuple[float, str]] = []
        self.action_index: dict[str, list[tuple[float, str]]] = {}
        self.action_log: list[tuple[float, str, int]] = []
        self._task_state_index: dict[str, int] = {}
        self._last_action_per_entity: dict[int, int] = {}
        self._embedder = None

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @property
    def _model(self):
        if self._embedder is None:
            self._embedder = _load_embedder(self._embed_model_name, self._embed_device)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    # ------------------------------------------------------------------
    # Entity lookup
    # ------------------------------------------------------------------

    def _find_entity(self, emb: np.ndarray) -> int | None:
        entity_pairs = [(i, n) for i, n in enumerate(self.nodes) if n.node_type == "entity"]
        if not entity_pairs:
            return None
        entity_embs = np.stack([n.embedding for _, n in entity_pairs])
        sims = entity_embs @ emb
        best = int(np.argmax(sims))
        if sims[best] >= ENTITY_LINK_THRESHOLD:
            return entity_pairs[best][0]
        return None

    def _find_entity_by_id(self, stable_id: str) -> int | None:
        # Return the existing entity index for a P<N>/O<N> id, or None.
        key = f"id:{stable_id}"
        return self._dedup.get(key)

    def _format_count_aggregate(
        self,
        question: str,
        embed_sim_threshold: float = 0.60,
    ) -> str | None:
        # Return a [Counting Aggregate] block for count-style questions.

        if not self.action_log and not self.action_index:
            return None

        focus = _extract_count_focus(question)
        fstems = _content_stems(focus)

        # ---- Path 1: stem-based ----
        rows: list[tuple[str, list[float], str]] = []
        for stem in fstems:
            entries = self.action_index.get(stem)
            if not entries:
                continue
            unique_ts = sorted({round(t, 1) for t, _ in entries})
            sample_text = entries[0][1]
            rows.append((stem, unique_ts, sample_text))

        if rows:
            rows.sort(key=lambda r: -len(r[1]))
            lines = ["[Counting Aggregate — observed action occurrences across captioned chunks]"]
            for stem, ts_list, sample in rows[:5]:
                n = len(ts_list)
                preview = ", ".join(_fmt_time(t) for t in ts_list[:8])
                more = ", ..." if n > 8 else ""
                sample_short = sample[:60].rstrip()
                lines.append(
                    f"  '{stem}' (e.g. \"{sample_short}\"): {n} time(s) at [{preview}{more}]"
                )
            return "\n".join(lines)

        # ---- Path 2: embedding fallback ----
        if not self.action_log:
            return None

        f_emb = self._embed([focus])[0]
        action_idxes = [i for i, n in enumerate(self.nodes) if n.node_type == "action"]
        if not action_idxes:
            return None
        embs = np.stack([self.nodes[i].embedding for i in action_idxes])
        sims = embs @ f_emb
        sim_lookup = {idx: float(sims[i]) for i, idx in enumerate(action_idxes)}

        # Reduce to ONE max-similarity per chunk timestamp.  
        chunk_max: dict[float, tuple[float, str]] = {}    # ts → (sim, text)
        for ts, text, idx in self.action_log:
            sim = sim_lookup.get(idx)
            if sim is None:
                continue
            tk = round(ts, 1)
            prev = chunk_max.get(tk)
            if prev is None or sim > prev[0]:
                chunk_max[tk] = (sim, text)

        if not chunk_max:
            return None

        # Keep chunks above threshold.  Order by timestamp for display.
        high = sorted(
            ((ts, sim, text) for ts, (sim, text) in chunk_max.items() if sim >= embed_sim_threshold),
            key=lambda x: x[0],
        )
        if not high:
            return None

        n = len(high)
        ts_list = [t for t, _, _ in high]
        preview = ", ".join(_fmt_time(t) for t in ts_list[:8])
        more = ", ..." if n > 8 else ""

        # Three best-matching distinct caption snippets.
        seen_texts: set[str] = set()
        samples: list[str] = []
        for _, _, text in sorted(high, key=lambda x: -x[1]):
            tn = self._normalise(text)
            if tn in seen_texts:
                continue
            seen_texts.add(tn)
            samples.append(text[:60])
            if len(samples) == 3:
                break
        samples_str = " | ".join(samples)

        return (
            "[Counting Aggregate — semantic match "
            "(no exact verb match; using cosine over action embeddings)]\n"
            f"  Activity \"{focus}\": {n} candidate occurrence(s) at "
            f"[{preview}{more}]\n"
            f"  Matching captions: {samples_str}\n"
            f"  Note: this is an estimate from semantic similarity; the "
            f"actual number may be lower if some matches are false positives."
        )

    @staticmethod
    def _strip_subject(text: str, entity_text: str) -> str:
        ent_lower = entity_text.lower().strip()
        t_lower = text.lower().strip()
        if t_lower.startswith(ent_lower):
            rest = text[len(ent_lower):].lstrip()
            rest = re.sub(r"^(is|are|was|were|has|have)\s+", "", rest, flags=re.IGNORECASE)
            return rest if len(rest) > 3 else text
        return text

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    @staticmethod
    def _stable_id(text: str) -> str | None:
        """Return leading P<N> / O<N> id, or None."""
        m = _STABLE_ID_RE.match(text.strip())
        return f"{m.group(1)}{m.group(2)}" if m else None

    @staticmethod
    def _referenced_ids(text: str) -> list[str]:
        """All P<N>/O<N> ids referenced anywhere in `text`, preserving order."""
        return [f"{m.group(1)}{m.group(2)}"
                for m in re.finditer(r"\b([PO])(\d+)\b", text)]

    def _add_or_merge_node(
        self,
        text: str,
        node_type: str,
        timestamp: float,
    ) -> int | None:
        """Insert or merge a node and return its index (None if rejected)."""
        text = text.strip()
        if not text or text.upper() in {"NONE", "N/A", "-"} or len(text) < 5:
            return None


        if node_type == "action":
            for stem in _content_stems(text):
                self.action_index.setdefault(stem, []).append((timestamp, text))

        # ---- dedup key ----
        sid = self._stable_id(text) if node_type == "entity" else None
        if sid is not None:
            text = text.split(":", 1)[0].strip()
            key = f"id:{sid}"
        else:
            key = self._normalise(text)

        if key in self._dedup and (node_type == "entity" or self.merge_spokes):
            idx = self._dedup[key]
            self.nodes[idx].count += 1
            self.nodes[idx].timestamp = max(self.nodes[idx].timestamp, timestamp)
            return idx

        if len(self.nodes) >= self.max_nodes:
            return None

        emb = self._embed([text])[0]

        # Entity link for spoke nodes.
        entity_idx: int | None = None
        if node_type not in ("entity", "text_ocr"):
            for ref_id in self._referenced_ids(text):
                hit = self._find_entity_by_id(ref_id)
                if hit is not None:
                    entity_idx = hit
                    break
            if entity_idx is None:
                entity_idx = self._find_entity(emb)

        node = TextNode(
            text=text, node_type=node_type,
            timestamp=timestamp, first_seen=timestamp,
            embedding=emb, entity_idx=entity_idx,
        )
        new_idx = len(self.nodes)
        self._dedup[key] = new_idx
        self.nodes.append(node)

        # ---- temporal-next chain for actions ----
        if node_type == "action" and entity_idx is not None:
            prev = self._last_action_per_entity.get(entity_idx)
            if prev is not None and self.nodes[prev].next_action is None:
                self.nodes[prev].next_action = new_idx
            self._last_action_per_entity[entity_idx] = new_idx

        return new_idx

    # ------------------------------------------------------------------
    # Public API: update
    # ------------------------------------------------------------------

    def update(self, caption: str, timestamp: float) -> None:
        """Parse a structured caption and merge facts into hub-and-spoke memory."""
        current_section: str | None = None
        pending: list[str] = []

        # Track entity nodes touched in THIS chunk (for co-occurrence edges).
        chunk_entity_idxes: set[int] = set()

        def flush() -> None:
            nonlocal current_section, pending
            if current_section is None or not pending:
                pending = []
                return
            node_type = _SECTION_TO_TYPE.get(current_section, "event")
            raw = " ".join(pending).strip()
            items = [s.strip() for s in raw.split(";") if s.strip()]
            if not items:
                items = [raw] if raw else []
            for item in items:
                idx = self._add_or_merge_node(item, node_type, timestamp)
                if idx is None:
                    continue
                if node_type == "entity":
                    chunk_entity_idxes.add(idx)
                if current_section == "ACTIONS":
                    self.action_log.append((timestamp, item, idx))
                if current_section == "EVENT":
                    self.timeline.append((timestamp, item))
            pending = []

        for raw_line in caption.splitlines():
            line = raw_line.strip()
            clean = re.sub(r"^[-*•]\s+", "", line)
            matched = False
            for section_key in _SECTION_TO_TYPE:
                m = re.match(rf"^{section_key}\s*:\s*(.*)", clean, re.IGNORECASE)
                if m:
                    flush()
                    current_section = section_key
                    rest = m.group(1).strip()
                    if rest:
                        pending.append(rest)
                    matched = True
                    break
            if not matched and clean:
                pending.append(clean)
        flush()

        # Co-occurrence edges: every pair of entities seen this chunk.
        if len(chunk_entity_idxes) > 1:
            entity_list = list(chunk_entity_idxes)
            for i, a in enumerate(entity_list):
                for b in entity_list[i + 1 :]:
                    self.nodes[a].co_entities.add(b)
                    self.nodes[b].co_entities.add(a)

    def update_task_state(self, key: str, state: dict[str, Any]) -> None:
        """Store a causal readout state as a regular Hub-and-Spoke node."""
        state_key = str(key)
        state_copy = dict(state)
        timestamp = float(
            state_copy.get("cutoff")
            or state_copy.get("current_cutoff")
            or state_copy.get("timestamp")
            or 0.0
        )
        text = (
            f"STATE {state_key}: "
            f"response={state_copy.get('response')}; "
            f"delta={state_copy.get('new_count')}; "
            f"label={state_copy.get('label')}; "
            f"cutoff={state_copy.get('cutoff')}"
        )
        metadata = {
            "task_state_key": state_key,
            "task_state": state_copy,
        }
        idx = self._task_state_index.get(state_key)
        if idx is not None and idx < len(self.nodes):
            node = self.nodes[idx]
            node.text = text
            node.timestamp = max(node.timestamp, timestamp)
            node.count += 1
            node.metadata = metadata
            return
        if len(self.nodes) >= self.max_nodes:
            return
        node = TextNode(
            text=text,
            node_type="task_state",
            timestamp=timestamp,
            first_seen=timestamp,
            embedding=None,
            metadata=metadata,
        )
        self._task_state_index[state_key] = len(self.nodes)
        self.nodes.append(node)

    def get_task_state(self, key: str) -> dict[str, Any] | None:
        idx = self._task_state_index.get(str(key))
        if idx is None or idx >= len(self.nodes):
            return None
        state = self.nodes[idx].metadata.get("task_state")
        return dict(state) if isinstance(state, dict) else None

    # ------------------------------------------------------------------
    # Public API: retrieve
    # ------------------------------------------------------------------

    def _resolve_top_k(self, top_k: int | str | None) -> tuple[int, bool]:
        if isinstance(top_k, str):
            mode = top_k.strip().lower()
            if mode in _DYNAMIC_TOP_K_MODES:
                return _validate_dynamic_top_k_max(
                    getattr(self, "dynamic_top_k_max", DYNAMIC_TOP_K_MAX)
                ), True
            try:
                top_k_val = int(mode)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid top_k value {top_k!r}. Expected a positive integer or one of "
                    f"{sorted(_DYNAMIC_TOP_K_MODES)}."
                ) from exc
        elif top_k is None:
            top_k_val = DYNAMIC_TOP_K_MAX
        else:
            top_k_val = int(top_k)
        return max(1, top_k_val), False

    def _apply_dynamic_top_k(
        self,
        ranked_hits: list[tuple[int, float]],
        base_threshold: float,
    ) -> list[tuple[int, float]]:
        if len(ranked_hits) < 2:
            return ranked_hits
        scores = np.array([score for _, score in ranked_hits], dtype=float)
        gaps = scores[:-1] - scores[1:]
        if gaps.size == 0:
            return ranked_hits

        best_gap_idx = int(np.argmax(gaps))
        best_gap = float(gaps[best_gap_idx])
        median_gap = float(np.median(gaps))
        min_gap = max(0.05, 0.12 * max(0.0, float(scores[0]) - float(base_threshold)))

        if best_gap >= min_gap and best_gap >= 2.0 * median_gap:
            keep = best_gap_idx + 1
            return ranked_hits[:keep]
        return ranked_hits

    def retrieve(
        self,
        question: str,
        top_k: int | str | None = 12,
        threshold: float | None = None,
        return_result: bool = False,
        min_evidence_sim: float | None = None,
    ) -> str | RetrievalResult:
        # Retrieve a question-relevant memory subset.

        if not self.nodes:
            result = RetrievalResult(
                context="",
                matched_nodes=False,
                signal=NO_MATCHING_NODE_SIGNAL,
                hit_count=0,
            )
            return result if return_result else result.context

        thr = threshold if threshold is not None else self.sim_threshold
        top_k_limit, dynamic_top_k = self._resolve_top_k(top_k)
        searchable = [
            (i, n) for i, n in enumerate(self.nodes)
            if n.embedding is not None
        ]
        if searchable:
            q_emb = self._embed([question])[0]
            emb_mat = np.stack([n.embedding for _, n in searchable])
            sims = emb_mat @ q_emb
            top_pos = np.argsort(sims)[::-1][:top_k_limit]
            candidate_sims = tuple(float(sims[int(i)]) for i in top_pos)
            ranked_hits = [
                (searchable[int(i)][0], float(sims[int(i)]))
                for i in top_pos
                if float(sims[int(i)]) >= thr
            ]
            if dynamic_top_k:
                ranked_hits = self._apply_dynamic_top_k(ranked_hits, base_threshold=thr)
            
            if min_evidence_sim is not None:
                ranked_hits = [h for h in ranked_hits if h[1] >= float(min_evidence_sim)]
            hit_idxes = {idx for idx, _ in ranked_hits}
            hit_sims = tuple(score for _, score in ranked_hits)
        else:
            hit_idxes = set()
            hit_sims = ()
            candidate_sims = ()

        q_lower = question.lower()
        is_recency_q = any(p in q_lower for p in _RECENCY_PHRASES)

        is_count_q = _is_count_question(question)

        if not hit_idxes and not (is_recency_q and self.timeline) \
                and not (is_count_q and self.action_index):
            result = RetrievalResult(
                context="",
                matched_nodes=False,
                signal=NO_MATCHING_NODE_SIGNAL,
                hit_count=0,
                hit_sims=hit_sims,
                candidate_sims=candidate_sims,
            )
            return result if return_result else result.context

        # --- Hub-and-spoke expansion ---
        expanded: set[int] = set(hit_idxes)
        if self.expand_retrieval:
            for idx in list(hit_idxes):
                n = self.nodes[idx]
                if n.node_type == "entity":
                    for i, other in enumerate(self.nodes):
                        if other.entity_idx == idx:
                            expanded.add(i)
                    if self.expand_co_occurrence:
                        for co in n.co_entities:
                            expanded.add(co)
                elif n.entity_idx is not None:
                    expanded.add(n.entity_idx)
                    if n.node_type == "action" and self.expand_next_action:
                        nxt = n.next_action
                        depth = 0
                        while nxt is not None and depth < 2:
                            expanded.add(nxt)
                            nxt = self.nodes[nxt].next_action
                            depth += 1

        q_keyword_stems = _keywords(question)

        parts: list[str] = []


        if is_count_q:
            agg = self._format_count_aggregate(question)
            if agg:
                parts.append(agg)

        # Recency bypass
        if is_recency_q and self.timeline:
            recent = sorted(self.timeline, key=lambda x: x[0])[-_RECENT_EVENTS_K:]
            parts.append("[Most recent events]\n" + "\n".join(
                f"  [{_fmt_time(ts)}] {evt}" for ts, evt in recent
            ))

        # Entity profiles
        entity_idxes = sorted(
            (i for i in expanded if self.nodes[i].node_type == "entity"),
            key=lambda i: self.nodes[i].first_seen,
        )
        for eidx in entity_idxes:
            enode = self.nodes[eidx]
            header = f"[{enode.text}]"
            if enode.count > 1:
                header += f" ×{enode.count}"

            linked = [self.nodes[i] for i in expanded if self.nodes[i].entity_idx == eidx]
            if not linked:
                parts.append(header)
                continue

            lines = [header]
            act_evt = sorted(
                [n for n in linked if n.node_type in ("action", "event")],
                key=lambda n: n.first_seen,
            )
            for n in act_evt:
                body = self._strip_subject(n.text, enode.text)
                suffix = f" (×{n.count})" if n.count > 1 else ""
                lines.append(f"  [{_fmt_time(n.first_seen)}] {body}{suffix}")

            for n in linked:
                if n.node_type != "spatial":
                    continue
                body = self._strip_subject(n.text, enode.text)
                suffix = f" (×{n.count})" if n.count > 1 else ""
                lines.append(f"  {body}{suffix}")

            parts.append("\n".join(lines))


        unlinked = [
            self.nodes[i] for i in expanded
            if self.nodes[i].node_type not in ("entity", "text_ocr")
            and (
                self.nodes[i].entity_idx is None
                or self.nodes[i].entity_idx not in expanded
            )
        ]
        if unlinked:
            ul_lines = []
            for n in sorted(unlinked, key=lambda n: n.first_seen):
                t = f"[{_fmt_time(n.first_seen)}] " if n.node_type in ("action", "event") else ""
                suffix = f" (×{n.count})" if n.count > 1 else ""
                ul_lines.append(f"  • {t}{n.text}{suffix}")
            parts.append("[Other]\n" + "\n".join(ul_lines))

        # Screen text
        ocr_nodes = [self.nodes[i] for i in expanded if self.nodes[i].node_type == "text_ocr"]
        if ocr_nodes:
            parts.append("[Screen Text]\n" + "\n".join(f'  • "{n.text}"' for n in ocr_nodes))

        # Timeline
        has_temporal_q = bool(q_keyword_stems & {_stem(w) for w in _TEMPORAL_KEYWORDS})
        has_event_nodes = any(
            self.nodes[i].node_type in ("action", "event") for i in expanded
        )
        if self.timeline and (has_temporal_q or has_event_nodes):
            tl_scored = [
                (ts, evt) for ts, evt in self.timeline
                if q_keyword_stems & _keywords(evt)
            ]
            if not tl_scored:
                tl_scored = self.timeline[-8:]
            tl_sorted = sorted(tl_scored, key=lambda x: x[0])[:10]
            parts.append("[Timeline]\n" + "\n".join(
                f"  [{_fmt_time(ts)}] {evt}" for ts, evt in tl_sorted
            ))

        context = "\n\n".join(parts)
        result = RetrievalResult(
            context=context,
            matched_nodes=bool(hit_idxes),
            signal=None if hit_idxes else NO_MATCHING_NODE_SIGNAL,
            hit_count=len(hit_idxes),
            hit_sims=hit_sims,
            candidate_sims=candidate_sims,
        )
        return result if return_result else result.context

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.nodes)

    def stats(self) -> dict[str, Any]:
        from collections import Counter
        type_counts = Counter(n.node_type for n in self.nodes)
        co_edges = sum(len(n.co_entities) for n in self.nodes) // 2
        action_chains = sum(1 for n in self.nodes if n.next_action is not None)
        return {
            "total_nodes": len(self.nodes),
            "by_type": dict(type_counts),
            "timeline_events": len(self.timeline),
            "action_observations": len(self.action_log),
            "task_states": type_counts.get("task_state", 0),
            "co_occurrence_edges": co_edges,
            "action_chain_links": action_chains,
        }


# ---------------------------------------------------------------------------
# Flat-caption memory (structure ablation baseline)
# ---------------------------------------------------------------------------


class FlatCaptionMemory:
    # Structure-ablation baseline: same captions, no hub-and-spoke organization.

    def __init__(
        self,
        embed_model: str = DEFAULT_EMBED_MODEL,
        sim_threshold: float = SIM_THRESHOLD,
        max_nodes: int = MAX_NODES,
        embed_device: str = "cpu",
        dynamic_top_k_max: int = DYNAMIC_TOP_K_MAX,
    ) -> None:
        self._embed_model_name = embed_model
        self._embed_device = embed_device
        self.sim_threshold = sim_threshold
        self.max_nodes = max_nodes
        self.dynamic_top_k_max = _validate_dynamic_top_k_max(dynamic_top_k_max)
        self.captions: list[tuple[float, str]] = []
        self._embeddings: list[np.ndarray] = []
        self._embedder = None

    @property
    def _model(self):
        if self._embedder is None:
            self._embedder = _load_embedder(self._embed_model_name, self._embed_device)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def update(self, caption: str, timestamp: float) -> None:
        caption = caption.strip()
        if not caption or len(self.captions) >= self.max_nodes:
            return
        self.captions.append((timestamp, caption))
        self._embeddings.append(self._embed([caption])[0])

    # Same signature as HubAndSpokeMemory.retrieve so the evaluator can swap.
    def retrieve(
        self,
        question: str,
        top_k: int | str | None = 12,
        threshold: float | None = None,
        return_result: bool = False,
        min_evidence_sim: float | None = None,
    ) -> str | RetrievalResult:
        if not self.captions:
            result = RetrievalResult(
                context="", matched_nodes=False,
                signal=NO_MATCHING_NODE_SIGNAL, hit_count=0,
            )
            return result if return_result else result.context

        thr = threshold if threshold is not None else self.sim_threshold
        top_k_limit, dynamic_top_k = HubAndSpokeMemory._resolve_top_k(self, top_k)
        q_emb = self._embed([question])[0]
        sims = np.stack(self._embeddings) @ q_emb
        top_pos = np.argsort(sims)[::-1][:top_k_limit]
        candidate_sims = tuple(float(sims[int(i)]) for i in top_pos)
        ranked_hits = [
            (int(i), float(sims[int(i)])) for i in top_pos
            if float(sims[int(i)]) >= thr
        ]
        if dynamic_top_k:
            ranked_hits = HubAndSpokeMemory._apply_dynamic_top_k(
                self, ranked_hits, base_threshold=thr
            )
        if min_evidence_sim is not None:
            ranked_hits = [h for h in ranked_hits if h[1] >= float(min_evidence_sim)]
        if not ranked_hits:
            result = RetrievalResult(
                context="", matched_nodes=False,
                signal=NO_MATCHING_NODE_SIGNAL, hit_count=0,
                candidate_sims=candidate_sims,
            )
            return result if return_result else result.context

        # Render retrieved captions chronologically as flat timestamped blocks.
        blocks = [
            f"[{_fmt_time(self.captions[i][0])}]\n{self.captions[i][1]}"
            for i, _ in sorted(ranked_hits, key=lambda x: self.captions[x[0]][0])
        ]
        result = RetrievalResult(
            context="\n\n".join(blocks),
            matched_nodes=True,
            signal=None,
            hit_count=len(ranked_hits),
            hit_sims=tuple(score for _, score in ranked_hits),
            candidate_sims=candidate_sims,
        )
        return result if return_result else result.context

    def __len__(self) -> int:
        return len(self.captions)

    def stats(self) -> dict[str, Any]:
        return {"total_nodes": len(self.captions), "memory_type": "flat_caption"}


# ---------------------------------------------------------------------------
# Hub-and-Spoke Evaluator
# ---------------------------------------------------------------------------

class HubAndSpokeEvaluator:
    def __init__(
        self,
        qa_model,
        recent_frames: int = 4,
        extract_every_n_chunks: int = 1,
        max_extraction_chunks: int = 30,
        embed_model: str = DEFAULT_EMBED_MODEL,
        embed_device: str = "cpu",
        sim_threshold: float = SIM_THRESHOLD,
        top_k: int | str | None = 12,
        count_question_max_chunks: int = 0,
        caption_batch_size: int = 0,
        expand_retrieval: bool = True,
        expand_co_occurrence: bool = True,
        expand_next_action: bool = True,
        merge_spokes: bool = True,
        dynamic_top_k_max: int = DYNAMIC_TOP_K_MAX,
    ) -> None:
        self.qa = qa_model
        self.recent_frames = recent_frames
        self.extract_every_n_chunks = extract_every_n_chunks
        self.max_extraction_chunks = max_extraction_chunks
        self.embed_model = embed_model
        self.embed_device = embed_device
        self.sim_threshold = sim_threshold
        self.top_k = top_k
        self.expand_retrieval = expand_retrieval
        self.expand_co_occurrence = expand_co_occurrence
        self.expand_next_action = expand_next_action
        self.merge_spokes = merge_spokes
        self.dynamic_top_k_max = _validate_dynamic_top_k_max(dynamic_top_k_max)
        self.count_question_max_chunks = count_question_max_chunks
        self.caption_batch_size = int(caption_batch_size)
        self.last_retrieval: RetrievalResult | None = None

    # ------------------------------------------------------------------
    # Caption-prompt selection
    # ------------------------------------------------------------------

    def _caption_prompt_for(self, question: str | None) -> str:
        """Return an activity-aware caption prompt for count questions, else
        the generic CAPTION_PROMPT."""
        if question and _is_count_question(question):
            return make_activity_aware_caption_prompt(_extract_count_focus(question))
        return CAPTION_PROMPT

    def _caption_chunk(self, chunk, caption_prompt: str) -> tuple[str | None, float]:
        """Caption a single chunk; returns (caption, midpoint_timestamp).
        Caller is responsible for freeing chunk.frames when done with them.
        If a pre-caption pass stashed a caption on the chunk, return that
        instead of running another VLM forward."""
        chunk_ts = (chunk.start_time + chunk.end_time) / 2.0
        cached = getattr(chunk, "_pre_caption", None)
        if cached is not None:
            return (cached or None), chunk_ts
        if not chunk.frames:
            return None, 0.0
        caption = self.qa.generate_from_frames(chunk.frames, caption_prompt)
        return caption, chunk_ts

    def _pre_caption_chunks(self, chunks_to_caption, caption_prompt: str) -> None:
        bs = max(0, int(getattr(self, "caption_batch_size", 0)))
        if bs <= 0:
            return
        ready = [c for c in chunks_to_caption if c.frames and getattr(c, "_pre_caption", None) is None]
        if not ready:
            return
        for start in range(0, len(ready), bs):
            group = ready[start:start + bs]
            if bs == 1 or len(group) == 1:
                for c in group:
                    c._pre_caption = self.qa.generate_from_frames(c.frames, caption_prompt) or ""
            else:
                captions = self.qa.batch_caption_from_frames(
                    [c.frames for c in group], caption_prompt
                )
                for c, cap in zip(group, captions):
                    c._pre_caption = cap or ""

    # ------------------------------------------------------------------
    # Non-streaming build (Backward / Realtime tasks: one question per anno)
    # ------------------------------------------------------------------

    def _make_memory(self):
        return HubAndSpokeMemory(
            embed_model=self.embed_model,
            embed_device=self.embed_device,
            sim_threshold=self.sim_threshold,
            dynamic_top_k_max=self.dynamic_top_k_max,
            expand_retrieval=self.expand_retrieval,
            expand_co_occurrence=self.expand_co_occurrence,
            expand_next_action=self.expand_next_action,
            merge_spokes=self.merge_spokes,
        )

    def build_memory_from_chunks(
        self,
        chunks,
        question: str | None = None,
    ) -> HubAndSpokeMemory:
        memory = self._make_memory()

        is_count = question is not None and _is_count_question(question)
        caption_prompt = self._caption_prompt_for(question)

        strided = [
            chunks[i]
            for i in range(0, len(chunks), self.extract_every_n_chunks)
        ]
        cap = self.count_question_max_chunks if is_count else self.max_extraction_chunks
        if cap and len(strided) > cap:
            n = cap
            step = len(strided) / n
            strided = [strided[int(i * step)] for i in range(n)]

        sampled_ids = {id(c) for c in strided}
        for c in chunks:
            if id(c) not in sampled_ids:
                c.frames = []

        self._pre_caption_chunks(strided, caption_prompt)

        for chunk in strided:
            caption, ts = self._caption_chunk(chunk, caption_prompt)
            chunk.frames = []
            if caption:
                memory.update(caption, ts)

        return memory

    def build_graph_from_chunks(
        self,
        chunks,
        question: str | None = None,
    ) -> HubAndSpokeMemory:
        """Compatibility wrapper; use build_memory_from_chunks in new code."""
        return self.build_memory_from_chunks(chunks, question=question)

    # ------------------------------------------------------------------
    # Streaming build + answer (Forward task: many timed sub-questions
    # against ONE video — caption each chunk once, answer at each cutoff)
    # ------------------------------------------------------------------

    def stream_and_answer(
        self,
        chunks,
        sub_tests: list[tuple[int, dict]],
        prompt_for_sub: "callable",
        use_logits: bool = True,
        num_options: int = 4,
    ) -> list[tuple[int, str | None, dict]]:
        
        memory = self._make_memory()

        caption_prompt = (
            self._caption_prompt_for(prompt_for_sub(sub_tests[0][0]))
            if sub_tests
            else CAPTION_PROMPT
        )

        window = max(1, self.recent_frames)
        recent_buffer: list = []
        captioned_count = 0

        if self.caption_batch_size > 0 and sub_tests:
            max_cutoff = max(ti["realtime"] for _, ti in sub_tests)
            eligible = [c for c in chunks if c.end_time <= max_cutoff]
            if len(eligible) > window:
                self._pre_caption_chunks(eligible[:-window], caption_prompt)

        def _evict_and_caption() -> None:
            nonlocal captioned_count
            old = recent_buffer.pop(0)
            caption, ts = self._caption_chunk(old, caption_prompt)
            if caption:
                memory.update(caption, ts)
                captioned_count += 1
            old.frames = []

        chunk_iter = iter(chunks)
        pending = next(chunk_iter, None)

        results: list[tuple[int, str | None, dict]] = []

        for orig_idx, ti in sub_tests:
            cutoff = ti["realtime"]
            t0 = time.perf_counter()

            while pending is not None and pending.end_time <= cutoff:
                recent_buffer.append(pending)
                while len(recent_buffer) > window:
                    _evict_and_caption()
                pending = next(chunk_iter, None)

            recent = [f for c in recent_buffer for f in c.frames]
            prompt = prompt_for_sub(orig_idx)
            if use_logits:
                response = self.answer_with_memory_mcq(
                    memory, recent, prompt, num_options=num_options
                )
            else:
                response = self.answer_with_memory(memory, recent, prompt)

            elapsed = time.perf_counter() - t0
            meta = {
                "generate_time": elapsed,
                "num_hist_chunks": captioned_count,
                "num_recent_chunks": len(recent_buffer),
                "num_recent_frames": len(recent),
                "realtime_cutoff": cutoff,
                **{f"hub_spoke_{k}": v for k, v in memory.stats().items()},
            }
            results.append((orig_idx, response, meta))


        for chunk in recent_buffer:
            chunk.frames = []

        return results

    # ------------------------------------------------------------------
    # Answer paths shared by both flows
    # ------------------------------------------------------------------

    def answer_with_memory(
        self,
        memory: HubAndSpokeMemory,
        recent_frames: list,
        question: str,
        include_no_match_signal: bool = False,
        min_evidence_sim: float | None = None,
        answer_instruction: str | None = None,
    ) -> str:
        retrieval = memory.retrieve(
            question,
            top_k=self.top_k,
            return_result=True,
            min_evidence_sim=min_evidence_sim,
        )
        self.last_retrieval = retrieval
        full_question = self._question_with_retrieval(
            question, retrieval, include_no_match_signal=include_no_match_signal
        )
        if answer_instruction:
            full_question = f"{answer_instruction}\n\n{full_question}"
        return self.qa.generate_from_frames(recent_frames, full_question)

    def answer_with_memory_mcq(
        self,
        memory: HubAndSpokeMemory,
        recent_frames: list,
        question: str,
        num_options: int = 4,
        include_no_match_signal: bool = False,
        min_evidence_sim: float | None = None,
        answer_instruction: str | None = None,
    ) -> str:
        retrieval = memory.retrieve(
            question,
            top_k=self.top_k,
            return_result=True,
            min_evidence_sim=min_evidence_sim,
        )
        self.last_retrieval = retrieval
        full_question = self._question_with_retrieval(
            question, retrieval, include_no_match_signal=include_no_match_signal
        )
        if answer_instruction:
            full_question = f"{answer_instruction}\n\n{full_question}"
        return self.qa.score_mcq_from_frames(recent_frames, full_question, num_options=num_options)

    def answer_with_graph(
        self,
        memory: HubAndSpokeMemory,
        recent_frames: list,
        question: str,
        include_no_match_signal: bool = False,
    ) -> str:
        """Compatibility wrapper; use answer_with_memory in new code."""
        return self.answer_with_memory(
            memory,
            recent_frames,
            question,
            include_no_match_signal=include_no_match_signal,
        )

    def answer_with_graph_mcq(
        self,
        memory: HubAndSpokeMemory,
        recent_frames: list,
        question: str,
        num_options: int = 4,
        include_no_match_signal: bool = False,
    ) -> str:
        """Compatibility wrapper; use answer_with_memory_mcq in new code."""
        return self.answer_with_memory_mcq(
            memory,
            recent_frames,
            question,
            num_options=num_options,
            include_no_match_signal=include_no_match_signal,
        )

    @staticmethod
    def _question_with_retrieval(
        question: str,
        retrieval: RetrievalResult,
        include_no_match_signal: bool = False,
    ) -> str:
        if retrieval.context:
            return f"{retrieval.context}\n\n{question}"
        if include_no_match_signal and retrieval.signal:
            return f"[Retrieval status]\n{retrieval.signal}\n\n{question}"
        return question


class FlatCaptionEvaluator(HubAndSpokeEvaluator):
    """Structure-ablation evaluator: identical captioning pipeline and
    retrieval hyperparameters, but history is stored as flat whole-caption
    blocks (FlatCaptionMemory) instead of the hub-and-spoke graph."""

    def _make_memory(self):
        return FlatCaptionMemory(
            embed_model=self.embed_model,
            embed_device=self.embed_device,
            sim_threshold=self.sim_threshold,
            dynamic_top_k_max=self.dynamic_top_k_max,
        )
