"""Offline audit of the keyword retrieval gate.

Compares the paper-faithful keyword gate (``dhsm/retrieval_gate.py``) against
the routing it replaces, which read the benchmarks' own task annotations.
Requires only the annotation JSONs — no videos, no model weights, no GPU.

    python experiments/verify_retrieval_gate.py \
        --ovo_anno   /path/to/ovo_bench_new.json \
        --sb_anno    /path/to/questions_real.json

Both paths are optional; whichever is supplied is audited.

What it reports
---------------
* bucket sizes and purity for STRONG_MEMORY / STRONG_RECENT / AMBIGUOUS
* the false-negative rate (a history question sent to the recent window)
  and the false-positive rate (the reverse), against the old routing
* an expected-accuracy delta from the two error rates, using the per-question
  costs implied by Fig. 1

Fig. 1 operating points used for the cost model:
    A (recent frames only)  Real-Time 78.98   Backward 55.66
    D (static HSM top-K)    Real-Time 77.84   Backward 61.94
    D-HSM                   Real-Time 78.98   Backward 62.82

so a missed history question costs 62.82-55.66 = 7.16 points and a spurious
retrieval costs 78.98-77.84 = 1.14 points: a 6.3x asymmetry that the gate's
"when in doubt, retrieve" default is built around.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ovo_bench import BACKWARD_TASKS, REAL_TIME_TASKS
from dhsm.retrieval_gate import AMBIGUOUS, STRONG_MEMORY, STRONG_RECENT, gate_question

COST_MISSED_HISTORY = (62.82 - 55.66) / 100.0
COST_SPURIOUS_RETRIEVAL = (78.98 - 77.84) / 100.0


def audit(name: str, items: list[tuple[str, list[str], bool]]) -> dict[str, float]:
    """``items`` is (question, options, old_routing_said_use_memory)."""
    buckets: Counter[str] = Counter()
    bucket_gold: dict[str, Counter[str]] = defaultdict(Counter)
    cue_counts: Counter[str] = Counter()
    fn: list[str] = []
    fp: list[str] = []

    for question, options, gold in items:
        d = gate_question(question, options)
        buckets[d.bucket] += 1
        bucket_gold[d.bucket]["memory" if gold else "recent"] += 1
        for cue in d.memory_cues if d.needs_memory else d.recent_cues:
            cue_counts[cue] += 1
        if gold and not d.needs_memory:
            fn.append(question)
        elif not gold and d.needs_memory:
            fp.append(question)

    n = len(items)
    n_mem = sum(1 for _, _, g in items if g)
    n_rec = n - n_mem
    fn_rate = len(fn) / n_mem if n_mem else 0.0
    fp_rate = len(fp) / n_rec if n_rec else 0.0

    print(f"\n{'=' * 72}\n{name}   n={n}  (old routing: memory={n_mem}, recent={n_rec})\n{'=' * 72}")
    print(f"{'bucket':<16}{'n':>6}{'->memory':>10}{'->recent':>10}{'purity':>9}")
    for b in (STRONG_MEMORY, STRONG_RECENT, AMBIGUOUS):
        tot = buckets[b]
        if not tot:
            continue
        g = bucket_gold[b]
        purity = max(g["memory"], g["recent"]) / tot * 100
        print(f"{b:<16}{tot:>6}{g['memory']:>10}{g['recent']:>10}{purity:>8.1f}%")

    agree = (n - len(fn) - len(fp)) / n * 100
    print(f"\n  agreement with old routing      : {agree:.2f}%")
    print(f"  FALSE NEGATIVE (history->recent): {len(fn):>4}/{n_mem} = {fn_rate * 100:.2f}%   [costly: {COST_MISSED_HISTORY * 100:.2f} pts each]")
    print(f"  false positive (recent->memory) : {len(fp):>4}/{n_rec} = {fp_rate * 100:.2f}%   [cheap:  {COST_SPURIOUS_RETRIEVAL * 100:.2f} pts each]")

    amb_recent = bucket_gold[AMBIGUOUS]["recent"]
    if fp:
        print(f"  ...of which {amb_recent} ({amb_recent / len(fp) * 100:.0f}%) fall in AMBIGUOUS, where the")
        print(f"     stricter similarity floor (--gate_strict_sim) decides, not the keywords.")

    d_mem = -fn_rate * COST_MISSED_HISTORY * 100
    d_rec = -fp_rate * COST_SPURIOUS_RETRIEVAL * 100
    d_all = (n_mem * fn_rate * COST_MISSED_HISTORY + n_rec * fp_rate * COST_SPURIOUS_RETRIEVAL) / n * 100
    print("\n  expected accuracy change (upper bound on the loss — assumes the")
    print("  stricter floor never recovers an AMBIGUOUS case):")
    print(f"     memory split {d_mem:+.2f} pp    recent split {d_rec:+.2f} pp    overall {-d_all:+.2f} pp")

    if fn:
        print("\n  false negatives in full (these are the ones that can cost accuracy):")
        for q in fn[:20]:
            print(f"     - {q[:100]}")
        if len(fn) > 20:
            print(f"     ... and {len(fn) - 20} more")

    print("\n  most frequent firing cues:")
    for cue, c in cue_counts.most_common(10):
        print(f"     {cue:<24} {c}")

    return {"fn_rate": fn_rate, "fp_rate": fp_rate, "agreement": agree}


def load_ovo(path: str) -> list[tuple[str, list[str], bool]]:
    data = json.load(open(path))
    keep = set(BACKWARD_TASKS) | set(REAL_TIME_TASKS)
    return [
        (str(x.get("question", "")), list(x.get("options") or []), x["task"] in BACKWARD_TASKS)
        for x in data
        if x.get("task") in keep
    ]


def load_sb(path: str) -> list[tuple[str, list[str], bool]]:
    data = json.load(open(path))
    out = []
    for video in data:
        for q in video.get("questions", []):
            ability = str(q.get("required_ability") or "").strip().lower()
            if ability not in ("episodic memory", "working memory"):
                continue
            out.append(
                (str(q.get("question", "")), list(q.get("options") or []), ability == "episodic memory")
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ovo_anno", help="Path to ovo_bench_new.json")
    ap.add_argument("--sb_anno", help="Path to StreamingBench questions_real.json")
    args = ap.parse_args()

    if not args.ovo_anno and not args.sb_anno:
        ap.error("supply at least one of --ovo_anno / --sb_anno")

    if args.ovo_anno:
        audit("OVO-Bench  (Backward vs Real-Time)", load_ovo(args.ovo_anno))
    if args.sb_anno:
        audit("StreamingBench  (episodic vs working memory)", load_sb(args.sb_anno))
    print()


if __name__ == "__main__":
    main()
