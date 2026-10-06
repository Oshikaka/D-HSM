"""Incremental Hub-and-Spoke memory with per-chunk provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .hub_and_spoke import (
    ENTITY_LINK_THRESHOLD,
    MAX_NODES,
    NO_MATCHING_NODE_SIGNAL,
    DEFAULT_EMBED_MODEL,
    SIM_THRESHOLD,
    DYNAMIC_TOP_K_MAX,
    _validate_dynamic_top_k_max,
    RetrievalResult,
    HubAndSpokeMemory,
    HubAndSpokeEvaluator,
    _SECTION_TO_TYPE,
    _RECENCY_PHRASES,
    _RECENT_EVENTS_K,
    _TEMPORAL_KEYWORDS,
    _fmt_time,
    _keywords,
    _load_embedder,
    _stem,
)

import re


@dataclass
class IncNode:
    node_id: int
    text: str
    node_type: str
    embedding: np.ndarray | None = field(default=None, repr=False)
    entity_ref: int | None = None          # hub node_id (spokes only)
    attach_method: str | None = None       # 'id' | 'embedding' | None
    stable_id: str | None = None           # 'P1'/'O3' for id-merged entities
    seq: int = 0                           # global seq of first observation
    # provenance: one entry per observation event, multiplicity preserved
    observations: list[tuple[int, float]] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.observations)

    @property
    def first_seen(self) -> float:
        return min(ts for _, ts in self.observations)

    @property
    def last_seen(self) -> float:
        return max(ts for _, ts in self.observations)


class IncrementalHubAndSpokeMemory:
    """Provenance-tracking incremental variant of HubAndSpokeMemory."""

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

        self.nodes: dict[int, IncNode] = {}
        self._next_id = 0
        self._seq = 0
        self._dedup: dict[str, int] = {}
        # co-occurrence edges with per-edge supporting chunk sets
        self.co_edges: dict[frozenset, set[int]] = {}
        # per-hub ordered action spokes (consecutive pairs = next-action links)
        self.action_seq: dict[int, list[int]] = {}
        # timeline entries carry their source chunk for removal
        self.timeline: list[tuple[int, float, str]] = []  # (chunk_id, ts, text)
        # inverted index: chunk -> node ids it supports (edges found via nodes)
        self.chunk_index: dict[int, set[int]] = {}
        # observations rejected by the node cap, for re-admission
        self._overflow: list[tuple[int, int, float, str, str]] = []  # (seq, chunk, ts, text, type)
        self._embedder = None

    # ------------------------------------------------------------------
    @property
    def _model(self):
        if self._embedder is None:
            self._embedder = _load_embedder(self._embed_model_name, self._embed_device)
        return self._embedder

    def _embed(self, texts: list[str]) -> np.ndarray:
        return self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    @staticmethod
    def _normalise(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    @staticmethod
    def _stable_id(text: str) -> str | None:
        m = re.match(r"^([PO])(\d+)\b", text.strip())
        return f"{m.group(1)}{m.group(2)}" if m else None

    @staticmethod
    def _referenced_ids(text: str) -> list[str]:
        return [f"{m.group(1)}{m.group(2)}" for m in re.finditer(r"\b([PO])(\d+)\b", text)]

    def _alive_hubs(self) -> list[IncNode]:
        return [n for n in self.nodes.values() if n.node_type == "entity"]

    def _find_hub_by_embedding(
        self, emb: np.ndarray, max_seq: int | None = None
    ) -> int | None:
        hubs = self._alive_hubs()
        if max_seq is not None:
            hubs = [h for h in hubs if h.seq <= max_seq]
        if not hubs:
            return None
        embs = np.stack([h.embedding for h in hubs])
        sims = embs @ emb
        best = int(np.argmax(sims))
        if sims[best] >= ENTITY_LINK_THRESHOLD:
            return hubs[best].node_id
        return None

    # ------------------------------------------------------------------
    # Additions (Alg. 2)
    # ------------------------------------------------------------------

    def _admit(
        self, text: str, node_type: str, timestamp: float, chunk_id: int, seq: int
    ) -> int | None:
        """Insert or merge one observation; returns node_id or None."""
        text = text.strip()
        if not text or text.upper() in {"NONE", "N/A", "-"} or len(text) < 5:
            return None

        sid = self._stable_id(text) if node_type == "entity" else None
        if sid is not None:
            text = text.split(":", 1)[0].strip()
            key = f"id:{sid}"
        else:
            key = f"{node_type}:{self._normalise(text)}"

        existing = self._dedup.get(key)
        if existing is not None and existing in self.nodes:
            node = self.nodes[existing]
            node.observations.append((chunk_id, timestamp))
            self.chunk_index.setdefault(chunk_id, set()).add(existing)
            return existing

        if len(self.nodes) >= self.max_nodes:
            self._overflow.append((seq, chunk_id, timestamp, text, node_type))
            return None

        emb = self._embed([text])[0]
        entity_ref = None
        attach = None
        if node_type not in ("entity", "text_ocr"):
            for rid in self._referenced_ids(text):
                hub_id = self._dedup.get(f"id:{rid}")
                if hub_id is not None and hub_id in self.nodes:
                    entity_ref, attach = hub_id, "id"
                    break
            if entity_ref is None:
                hit = self._find_hub_by_embedding(emb)
                if hit is not None:
                    entity_ref, attach = hit, "embedding"

        node = IncNode(
            node_id=self._next_id, text=text, node_type=node_type,
            embedding=emb, entity_ref=entity_ref, attach_method=attach,
            stable_id=sid, seq=seq,
            observations=[(chunk_id, timestamp)],
        )
        self.nodes[self._next_id] = node
        self._dedup[key] = self._next_id
        self.chunk_index.setdefault(chunk_id, set()).add(self._next_id)

        if node_type == "action" and entity_ref is not None:
            self.action_seq.setdefault(entity_ref, []).append(self._next_id)

        self._next_id += 1
        return node.node_id

    def update(self, caption: str, timestamp: float, chunk_id: int | None = None) -> None:
        """Parse one chunk's structured caption and merge it incrementally."""
        if chunk_id is None:
            chunk_id = int(round(timestamp * 1000))
        current_section: str | None = None
        pending: list[str] = []
        chunk_entities: set[int] = set()

        def flush() -> None:
            nonlocal current_section, pending
            if current_section is None or not pending:
                pending = []
                return
            node_type = _SECTION_TO_TYPE.get(current_section, "event")
            raw = " ".join(pending).strip()
            items = [s.strip() for s in raw.split(";") if s.strip()] or ([raw] if raw else [])
            for item in items:
                self._seq += 1
                nid = self._admit(item, node_type, timestamp, chunk_id, self._seq)
                if nid is None:
                    continue
                if node_type == "entity":
                    chunk_entities.add(nid)
                if current_section == "EVENT":
                    self.timeline.append((chunk_id, timestamp, item))
            pending = []

        for raw_line in caption.splitlines():
            line = re.sub(r"^[-*•]\s+", "", raw_line.strip())
            matched = False
            for section_key in _SECTION_TO_TYPE:
                m = re.match(rf"^{section_key}\s*:\s*(.*)", line, re.IGNORECASE)
                if m:
                    flush()
                    current_section = section_key
                    rest = m.group(1).strip()
                    if rest:
                        pending.append(rest)
                    matched = True
                    break
            if not matched and line:
                pending.append(line)
        flush()

        ents = sorted(chunk_entities)
        for i, a in enumerate(ents):
            for b in ents[i + 1:]:
                self.co_edges.setdefault(frozenset((a, b)), set()).add(chunk_id)

    # ------------------------------------------------------------------
    # Removal (Alg. 2) — the incremental realization
    # ------------------------------------------------------------------

    def remove_chunks(self, removed: set[int]) -> None:
        removed = set(removed)

        # 1. O(k) lookup of everything the departing chunks supported.
        affected: set[int] = set()
        for cid in removed:
            affected |= self.chunk_index.pop(cid, set())

        # 2. Strip supporter entries; collect nodes whose support is empty.
        deleted: set[int] = set()
        for nid in affected:
            node = self.nodes.get(nid)
            if node is None:
                continue
            node.observations = [(c, ts) for c, ts in node.observations if c not in removed]
            if not node.observations:
                deleted.add(nid)

        for nid in deleted:
            node = self.nodes.pop(nid)
            for key, mapped in list(self._dedup.items()):
                if mapped == nid:
                    del self._dedup[key]
            self.action_seq.pop(nid, None)

        # 3. Re-attach surviving spokes whose hub was deleted.
        for node in self.nodes.values():
            if node.entity_ref in deleted:
                self._reattach(node)

        # 4. Edges: drop removed-chunk support; drop empty / dangling edges.
        for ekey in list(self.co_edges):
            self.co_edges[ekey] -= removed
            if not self.co_edges[ekey] or any(n in deleted for n in ekey):
                del self.co_edges[ekey]

        # 5. Re-link temporal-action chains across deletions.
        for hub_id in list(self.action_seq):
            self.action_seq[hub_id] = [
                nid for nid in self.action_seq[hub_id] if nid in self.nodes
            ]
            if not self.action_seq[hub_id]:
                del self.action_seq[hub_id]

        # 6. Timeline: drop entries from removed chunks.
        self.timeline = [(c, ts, txt) for c, ts, txt in self.timeline if c not in removed]

        # 7. Re-admit overflow observations (original order) into freed space.
        if self._overflow and len(self.nodes) < self.max_nodes:
            pending, self._overflow = self._overflow, []
            for seq, cid, ts, text, ntype in sorted(pending):
                if cid in removed or cid not in self.chunk_index and cid not in {c for c, _, _ in self.timeline}:
                    # chunk itself departed -> its observations are gone too
                    if cid in removed:
                        continue
                if len(self.nodes) >= self.max_nodes:
                    self._overflow.append((seq, cid, ts, text, ntype))
                    continue
                self._admit(text, ntype, ts, cid, seq)

    def _reattach(self, spoke: IncNode) -> None:
        """Reassign an orphaned spoke to the most similar remaining hub."""
        spoke.entity_ref, spoke.attach_method = None, None
        for rid in self._referenced_ids(spoke.text):
            hub_id = self._dedup.get(f"id:{rid}")
            if hub_id is not None and hub_id in self.nodes:
                spoke.entity_ref, spoke.attach_method = hub_id, "id"
                break
        if spoke.entity_ref is None and spoke.node_type not in ("entity", "text_ocr"):
            hit = self._find_hub_by_embedding(spoke.embedding, max_seq=spoke.seq)
            if hit is not None:
                spoke.entity_ref, spoke.attach_method = hit, "embedding"
        if spoke.node_type == "action" and spoke.entity_ref is not None:
            self.action_seq.setdefault(spoke.entity_ref, []).append(spoke.node_id)
            self.action_seq[spoke.entity_ref].sort(key=lambda nid: self.nodes[nid].seq)

    # ------------------------------------------------------------------
    # Retrieval (mirrors HubAndSpokeMemory.retrieve rendering)
    # ------------------------------------------------------------------

    def _next_actions(self, nid: int, depth: int = 2) -> list[int]:
        node = self.nodes[nid]
        if node.entity_ref is None:
            return []
        chain = self.action_seq.get(node.entity_ref, [])
        try:
            i = chain.index(nid)
        except ValueError:
            return []
        return chain[i + 1: i + 1 + depth]

    def __len__(self) -> int:
        return len(self.nodes)

    def retrieve(
        self,
        question: str,
        top_k: int | str | None = 12,
        threshold: float | None = None,
        return_result: bool = False,
        min_evidence_sim: float | None = None,
    ) -> str | RetrievalResult:
        """Mirrors HubAndSpokeMemory.retrieve, including the keyword gate's
        stricter floor for lexically undecidable questions."""
        if not self.nodes:
            result = RetrievalResult("", False, NO_MATCHING_NODE_SIGNAL, 0)
            return result if return_result else result.context

        thr = threshold if threshold is not None else self.sim_threshold
        top_k_limit, dynamic_top_k = HubAndSpokeMemory._resolve_top_k(self, top_k)
        ids = [nid for nid, n in self.nodes.items() if n.embedding is not None]
        q_emb = self._embed([question])[0]
        sims = np.stack([self.nodes[nid].embedding for nid in ids]) @ q_emb
        order = np.argsort(sims)[::-1][:top_k_limit]
        candidate_sims = tuple(float(sims[int(i)]) for i in order)
        ranked = [(ids[int(i)], float(sims[int(i)])) for i in order if float(sims[int(i)]) >= thr]
        if dynamic_top_k:
            ranked = HubAndSpokeMemory._apply_dynamic_top_k(self, ranked, base_threshold=thr)
        # See HubAndSpokeMemory.retrieve: post-filter, do not raise `thr`.
        if min_evidence_sim is not None:
            ranked = [h for h in ranked if h[1] >= float(min_evidence_sim)]
        hit_ids = {nid for nid, _ in ranked}
        hit_sims = tuple(score for _, score in ranked)

        q_lower = question.lower()
        is_recency_q = any(p in q_lower for p in _RECENCY_PHRASES)
        if not hit_ids and not (is_recency_q and self.timeline):
            result = RetrievalResult("", False, NO_MATCHING_NODE_SIGNAL, 0,
                                     hit_sims, candidate_sims)
            return result if return_result else result.context

        expanded: set[int] = set(hit_ids)
        for nid in list(hit_ids):
            n = self.nodes[nid]
            if n.node_type == "entity":
                for oid, other in self.nodes.items():
                    if other.entity_ref == nid:
                        expanded.add(oid)
                for ekey, support in self.co_edges.items():
                    if nid in ekey and support:
                        expanded |= set(ekey)
            elif n.entity_ref is not None:
                expanded.add(n.entity_ref)
                if n.node_type == "action":
                    expanded.update(self._next_actions(nid))

        parts: list[str] = []
        if is_recency_q and self.timeline:
            recent = sorted(self.timeline, key=lambda x: x[1])[-_RECENT_EVENTS_K:]
            parts.append("[Most recent events]\n" + "\n".join(
                f"  [{_fmt_time(ts)}] {evt}" for _, ts, evt in recent))

        def strip_subject(text: str, ent: str) -> str:
            return HubAndSpokeMemory._strip_subject(text, ent)

        hubs = sorted(
            (nid for nid in expanded if self.nodes[nid].node_type == "entity"),
            key=lambda nid: self.nodes[nid].first_seen)
        for hid in hubs:
            hub = self.nodes[hid]
            header = f"[{hub.text}]" + (f" ×{hub.count}" if hub.count > 1 else "")
            linked = [self.nodes[i] for i in expanded
                      if self.nodes[i].entity_ref == hid]
            if not linked:
                parts.append(header)
                continue
            lines = [header]
            for n in sorted((x for x in linked if x.node_type in ("action", "event")),
                            key=lambda x: x.first_seen):
                sfx = f" (×{n.count})" if n.count > 1 else ""
                lines.append(f"  [{_fmt_time(n.first_seen)}] {strip_subject(n.text, hub.text)}{sfx}")
            for n in linked:
                if n.node_type == "spatial":
                    sfx = f" (×{n.count})" if n.count > 1 else ""
                    lines.append(f"  {strip_subject(n.text, hub.text)}{sfx}")
            parts.append("\n".join(lines))

        unlinked = [self.nodes[i] for i in expanded
                    if self.nodes[i].node_type not in ("entity", "text_ocr")
                    and (self.nodes[i].entity_ref is None
                         or self.nodes[i].entity_ref not in expanded)]
        if unlinked:
            ul = []
            for n in sorted(unlinked, key=lambda x: x.first_seen):
                t = f"[{_fmt_time(n.first_seen)}] " if n.node_type in ("action", "event") else ""
                sfx = f" (×{n.count})" if n.count > 1 else ""
                ul.append(f"  • {t}{n.text}{sfx}")
            parts.append("[Other]\n" + "\n".join(ul))

        ocr = [self.nodes[i] for i in expanded if self.nodes[i].node_type == "text_ocr"]
        if ocr:
            parts.append("[Screen Text]\n" + "\n".join(f'  • "{n.text}"' for n in ocr))

        q_stems = _keywords(question)
        has_temporal_q = bool(q_stems & {_stem(w) for w in _TEMPORAL_KEYWORDS})
        has_event_nodes = any(self.nodes[i].node_type in ("action", "event") for i in expanded)
        if self.timeline and (has_temporal_q or has_event_nodes):
            scored = [(ts, evt) for _, ts, evt in self.timeline if q_stems & _keywords(evt)]
            if not scored:
                scored = [(ts, evt) for _, ts, evt in self.timeline[-8:]]
            parts.append("[Timeline]\n" + "\n".join(
                f"  [{_fmt_time(ts)}] {evt}" for ts, evt in sorted(scored)[:10]))

        result = RetrievalResult("\n\n".join(parts), bool(hit_ids),
                                 None if hit_ids else NO_MATCHING_NODE_SIGNAL, len(hit_ids),
                                 hit_sims, candidate_sims)
        return result if return_result else result.context

    # ------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        from collections import Counter
        types = Counter(n.node_type for n in self.nodes.values())
        return {
            "total_nodes": len(self.nodes),
            "by_type": dict(types),
            "co_occurrence_edges": len(self.co_edges),
            "timeline_events": len(self.timeline),
            "overflow_pending": len(self._overflow),
            "memory_type": "incremental_provenance",
        }

    def signature(self) -> list[tuple]:
        """Order-independent state fingerprint for equivalence checks."""
        rows = []
        for n in sorted(self.nodes.values(), key=lambda x: (x.node_type, x.text)):
            hub_text = self.nodes[n.entity_ref].text if n.entity_ref in self.nodes else None
            rows.append((n.node_type, n.text, n.count,
                         round(n.first_seen, 3), round(n.last_seen, 3), hub_text))
        edges = sorted(
            tuple(sorted(self.nodes[x].text for x in ekey)) + (len(sup),)
            for ekey, sup in self.co_edges.items())
        return rows + edges


def verify_equivalence(
    captions: list[tuple[int, float, str]], removed: set[int], **kw
) -> bool:
    """Incremental add+remove must equal a rebuild from the surviving chunks."""
    inc = IncrementalHubAndSpokeMemory(**kw)
    for cid, ts, cap in captions:
        inc.update(cap, ts, chunk_id=cid)
    inc.remove_chunks(removed)

    ref = IncrementalHubAndSpokeMemory(**kw)
    for cid, ts, cap in captions:
        if cid not in removed:
            ref.update(cap, ts, chunk_id=cid)
    return inc.signature() == ref.signature()


if __name__ == "__main__":
    caps = [
        (1, 10.0, "OBJECTS: O1 (red mug); O2 (wooden table)\nPEOPLE: P1 (blue shirt + woman): typing\n"
                  "ACTIONS: P1 picks up O1; P1 places O1 on O2\nTEXT: NONE\nSPATIAL: O1 is on O2\n"
                  "EVENT: A woman picks up a red mug."),
        (2, 20.0, "OBJECTS: O1 (red mug); O3 (silver laptop)\nPEOPLE: P1 (blue shirt + woman): typing\n"
                  "ACTIONS: P1 opens O3; P1 types on O3\nTEXT: SALE 50%\nSPATIAL: O3 is on O2\n"
                  "EVENT: The woman opens a silver laptop."),
        (3, 30.0, "OBJECTS: O1 (red mug)\nPEOPLE: P2 (red hat + man): walking\n"
                  "ACTIONS: P2 walks past O2; P1 places O1 on O2\nTEXT: NONE\nSPATIAL: NONE\n"
                  "EVENT: A man walks past the table."),
    ]
    ok1 = verify_equivalence(caps, removed={2})
    ok2 = verify_equivalence(caps, removed={1, 3})
    ok3 = verify_equivalence(caps, removed=set())
    print(f"remove {{2}}: {'PASS' if ok1 else 'FAIL'}")
    print(f"remove {{1,3}}: {'PASS' if ok2 else 'FAIL'}")
    print(f"remove {{}}: {'PASS' if ok3 else 'FAIL'}")

    m = IncrementalHubAndSpokeMemory()
    for cid, ts, cap in caps:
        m.update(cap, ts, chunk_id=cid)
    m.remove_chunks({2})
    print("\nafter removing chunk 2:", m.stats())
    print(m.retrieve("What did the woman do with the red mug?", top_k=5, threshold=0.25))


class IncrementalHubAndSpokeEvaluator(HubAndSpokeEvaluator):
    """HubAndSpokeEvaluator backed by IncrementalHubAndSpokeMemory.

    Note what this does and does not exercise.  ``build_memory_from_chunks``
    only ever calls ``update()``, so this measures whether the incremental
    rewrite preserves accuracy -- it does NOT exercise ``remove_chunks()``,
    the Algorithm 2 removal path that is the reason this module exists.
    The constructor also drops the expansion/merge ablation switches, which
    IncrementalHubAndSpokeMemory does not implement; passing any of them is
    rejected rather than silently ignored.
    """

    def _make_memory(self):
        for flag, value in (
            ("expand_retrieval", True),
            ("expand_co_occurrence", True),
            ("expand_next_action", True),
            ("merge_spokes", True),
        ):
            if getattr(self, flag, True) is not value:
                raise ValueError(
                    f"IncrementalHubAndSpokeMemory does not implement the "
                    f"{flag!r} ablation switch; run it with the default."
                )
        return IncrementalHubAndSpokeMemory(
            embed_model=self.embed_model,
            embed_device=self.embed_device,
            sim_threshold=self.sim_threshold,
            dynamic_top_k_max=self.dynamic_top_k_max,
        )
