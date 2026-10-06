"""Chunk-local entity references with semantic linking across video chunks."""

from __future__ import annotations
import copy
import re
from typing import Any
import numpy as np
from dhsm.hub_and_spoke import (
    DEFAULT_EMBED_MODEL, DYNAMIC_TOP_K_MAX, ENTITY_LINK_THRESHOLD, MAX_NODES, SIM_THRESHOLD,
    _SECTION_TO_TYPE,
)
from dhsm.hub_and_spoke_incremental import IncNode, IncrementalHubAndSpokeMemory


_ID = re.compile(r"\b([PO])\d+\b")
_UNRESOLVED = "[unresolved {} reference]"


class EntityResolvedMemory(IncrementalHubAndSpokeMemory):
    # Same public update/retrieve/remove/stats interface as incremental memory.

    def __init__(self, embed_model=DEFAULT_EMBED_MODEL, sim_threshold=SIM_THRESHOLD,
                 max_nodes=MAX_NODES, embed_device="cpu", dynamic_top_k_max=DYNAMIC_TOP_K_MAX):
        super().__init__(embed_model=embed_model, sim_threshold=sim_threshold,
                         max_nodes=max_nodes, embed_device=embed_device,
                         dynamic_top_k_max=dynamic_top_k_max)
        self._entities: dict[str, dict[str, Any]] = {}
        self._global_counters = {"O": 0, "P": 0}
        self._description_embeddings: dict[str, np.ndarray] = {}
        self._chunk_references: dict[int, dict[str, Any]] = {}
        # All active observations, including overflow, retain original order.
        self._observation_log: list[tuple[int, int, float, str, str]] = []

    @staticmethod
    def _parse(caption):
        section, pending, result = None, [], []

        def flush():
            if section is not None and pending:
                result.extend((section, item.strip()) for item in " ".join(pending).split(";")
                              if item.strip())
            pending.clear()

        for raw in caption.splitlines():
            line = re.sub(r"^[-*•]\s+", "", raw.strip())
            for name in _SECTION_TO_TYPE:
                match = re.match(rf"^{name}\s*:\s*(.*)", line, re.I)
                if match:
                    flush()
                    section = name
                    if match.group(1).strip():
                        pending.append(match.group(1).strip())
                    break
            else:
                if line:
                    pending.append(line)
        flush()
        return result

    @staticmethod
    def _description(text):
        # A suffix such as ": stirring" is a changing action, not identity.
        text = re.sub(r"^[PO]\d+\b\s*:?\s*", "", text).strip()
        if text.startswith("(") and ")" in text:
            text = text[1:text.index(")")]
        else:
            text = text.split(":", 1)[0]
        # Descriptions can themselves mention a local ID. Never embed or emit
        # that token as though it were a valid global reference.
        text = _ID.sub(lambda m: "person" if m.group(1) == "P" else "object", text)
        return re.sub(r"\s+", " ", text).strip(" ()")

    def _description_embedding(self, description):
        key = self._normalise(description)
        if key not in self._description_embeddings:
            self._description_embeddings[key] = self._embed([description])[0]
        return self._description_embeddings[key]

    def _resolve(self, kind, description, claimed):
        emb = self._description_embedding(description)
        best_id, best_sim = None, -float("inf")
        for gid, entity in self._entities.items():
            if gid[0] != kind or gid in claimed:
                continue
            similarity = float(entity["embedding"] @ emb)
            if similarity > best_sim:
                best_id, best_sim = gid, similarity
        if best_id is not None and best_sim >= ENTITY_LINK_THRESHOLD:
            return best_id, best_sim, "description_similarity"
        self._global_counters[kind] += 1
        gid = f"{kind}{self._global_counters[kind]}"
        self._entities[gid] = {"description": description, "embedding": emb, "support": []}
        return gid, None, "new_description"

    def update(self, caption: str, timestamp: float, chunk_id: int | None = None):
        if chunk_id is None:
            chunk_id = int(round(timestamp * 1000))
        if chunk_id in self._chunk_references:
            raise ValueError(f"Chunk {chunk_id} already active; remove it before replacing it")
        items = self._parse(caption)
        declarations, local, claimed, replacements = [], {}, set(), {}
        # Resolve all declarations before any references, including captions
        # whose ACTIONS section precedes OBJECTS/PEOPLE.
        for index, (section, text) in enumerate(items):
            if _SECTION_TO_TYPE[section] != "entity":
                continue
            description = self._description(text)
            if not description or description.upper() in {"NONE", "N/A", "-"}:
                replacements[index] = None
                continue
            lid = self._stable_id(text)
            kind = lid[0] if lid else ("P" if section == "PEOPLE" else "O")
            previous = local.get(lid) if lid else None
            if previous and self._normalise(previous["description"]) == self._normalise(description):
                gid, similarity, method = previous["global_id"], 1.0, "repeated_declaration"
            else:
                gid, similarity, method = self._resolve(kind, description, claimed)
            claimed.add(gid)
            declaration = {"local_id": lid, "global_id": gid, "description": description,
                           "similarity": similarity, "method": method}
            declarations.append(declaration)
            self._entities[gid]["support"].append((chunk_id, timestamp, description))
            if lid:
                # Conflicting duplicate declarations cannot safely resolve a
                # later reference. Preserve both observations, mark ambiguity.
                if lid in local and (local[lid] is None or local[lid]["global_id"] != gid):
                    local[lid] = None
                else:
                    local[lid] = declaration
            replacements[index] = f"{gid} ({description})"

        reference_events = []

        def rewrite(match):
            lid = match.group(0)
            declaration = local.get(lid)
            reference_events.append({"local_id": lid,
                                     "global_id": declaration["global_id"] if declaration else None})
            if declaration:
                return f'{declaration["global_id"]} ({declaration["description"]})'
            return _UNRESOLVED.format("person" if lid.startswith("P") else "object")

        self._chunk_references[chunk_id] = {
            "chunk_id": chunk_id, "timestamp": timestamp, "declarations": declarations,
            "references": reference_events,
        }
        for index, (section, text) in enumerate(items):
            node_type = _SECTION_TO_TYPE[section]
            if node_type == "entity":
                text = replacements[index]
                if text is None:
                    continue
            elif node_type != "text_ocr":
                text = _ID.sub(rewrite, text)
            self._seq += 1
            record = (self._seq, chunk_id, timestamp, text, node_type)
            self._observation_log.append(record)
            self._admit(text, node_type, timestamp, chunk_id, self._seq)
        self._refresh_auxiliary()

    def _key(self, text, node_type):
        sid = self._stable_id(text) if node_type == "entity" else None
        return f"id:{sid}" if sid else f"{node_type}:{self._normalise(text)}"

    def _attach(self, text, embedding, max_seq=None):
        references = self._referenced_ids(text)
        for rid in references:
            nid = self._dedup.get(f"id:{rid}")
            if nid in self.nodes:
                return nid, "id"
        # A missing/overflowed explicit entity must not bind to an unrelated
        # old hub through an embedding fallback.
        if not references and "[unresolved " not in text:
            nid = self._find_hub_by_embedding(embedding, max_seq=max_seq)
            if nid is not None:
                return nid, "embedding"
        return None, None

    def _admit(self, text, node_type, timestamp, chunk_id, seq):
        text = text.strip()
        if not text or text.upper() in {"NONE", "N/A", "-"} or len(text) < 5:
            return None
        key = self._key(text, node_type)
        existing = self._dedup.get(key)
        if existing in self.nodes:
            self.nodes[existing].observations.append((chunk_id, timestamp))
            self.chunk_index.setdefault(chunk_id, set()).add(existing)
            return existing
        if len(self.nodes) >= self.max_nodes:
            self._overflow.append((seq, chunk_id, timestamp, text, node_type))
            return None
        sid = self._stable_id(text) if node_type == "entity" else None
        emb = self._entities[sid]["embedding"] if sid else self._embed([text])[0]
        ref, method = (self._attach(text, emb) if node_type not in ("entity", "text_ocr")
                       else (None, None))
        nid = self._next_id
        self._next_id += 1
        self.nodes[nid] = IncNode(nid, text, node_type, emb, ref, method, sid, seq,
                                  [(chunk_id, timestamp)])
        self._dedup[key] = nid
        self.chunk_index.setdefault(chunk_id, set()).add(nid)
        return nid

    def _reattach(self, spoke):
        if spoke.node_type not in ("entity", "text_ocr"):
            spoke.entity_ref, spoke.attach_method = self._attach(
                spoke.text, spoke.embedding, max_seq=spoke.seq)

    def _refresh_auxiliary(self):
        self.co_edges, self.timeline, self.action_seq = {}, [], {}
        entities_by_chunk = {}
        for _, cid, ts, text, ntype in self._observation_log:
            nid = self._dedup.get(self._key(text, ntype))
            if nid not in self.nodes or (cid, ts) not in self.nodes[nid].observations:
                continue
            if ntype == "entity":
                entities_by_chunk.setdefault(cid, set()).add(nid)
            elif ntype == "event":
                self.timeline.append((cid, ts, text))
        for cid, ids in entities_by_chunk.items():
            ids = sorted(ids)
            for index, first in enumerate(ids):
                for second in ids[index + 1:]:
                    self.co_edges.setdefault(frozenset((first, second)), set()).add(cid)
        for node in sorted(self.nodes.values(), key=lambda n: n.seq):
            # Newly materialized overflow hubs may satisfy an earlier dangling
            # global reference. Reconsider these without changing resolved IDs.
            if node.entity_ref is None and node.node_type not in ("entity", "text_ocr"):
                self._reattach(node)
            if node.node_type == "action" and node.entity_ref is not None:
                self.action_seq.setdefault(node.entity_ref, []).append(node.node_id)

    def remove_chunks(self, removed):
        removed = set(removed)
        if not removed:
            return
        for cid in removed:
            self._chunk_references.pop(cid, None)
        self._observation_log = [r for r in self._observation_log if r[1] not in removed]
        for gid, entity in list(self._entities.items()):
            entity["support"] = [s for s in entity["support"] if s[0] not in removed]
            if not entity["support"]:
                del self._entities[gid]
            else:
                description = entity["support"][0][2]
                entity.update(description=description, embedding=self._description_embedding(description))
        # Always purge removed overflow, even when the graph remains full.
        pending, self._overflow = self._overflow, []
        super().remove_chunks(removed)
        for seq, cid, ts, text, ntype in sorted(pending):
            if cid not in removed:
                # Admission first checks deduplication, even at the cap.
                self._admit(text, ntype, ts, cid, seq)
        for gid, entity in self._entities.items():
            node = self.nodes.get(self._dedup.get(f"id:{gid}"))
            if node:
                node.text = f'{gid} ({entity["description"]})'
                node.embedding = entity["embedding"]
        self._refresh_auxiliary()
        active_descriptions = {self._normalise(d) for entity in self._entities.values()
                               for _, _, d in entity["support"]}
        self._description_embeddings = {k: v for k, v in self._description_embeddings.items()
                                        if k in active_descriptions}

    def reference_provenance(self):
        return copy.deepcopy(list(self._chunk_references.values()))

    def stats(self):
        result = super().stats()
        result.update(memory_type="entity_resolved_incremental",
                      entity_link_threshold=ENTITY_LINK_THRESHOLD,
                      global_entities=len(self._entities),
                      active_chunks=len(self._chunk_references),
                      unresolved_references=sum(r["global_id"] is None
                                                for c in self._chunk_references.values()
                                                for r in c["references"]))
        return result
