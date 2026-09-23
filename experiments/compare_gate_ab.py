"""Tabulate the accuracy difference between two routing modes.

Pairs the result directories produced by two otherwise-identical runs that
differ only in ``--routing`` (see ``scripts/run_eval.sh``) and prints the
per-task delta, plus a breakdown of where the keyword gate disagreed with the
old annotation-driven routing.

    python experiments/compare_gate_ab.py \
        --ovo_baseline results/gate_ab/ovo_task_label \
        --ovo_gated    results/gate_ab/ovo_keyword \
        --sb_baseline  results/gate_ab/sb_task_label \
        --sb_gated     results/gate_ab/sb_keyword

Either benchmark pair may be omitted.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ovo_bench import score_count, score_mcq, score_yes_no


def _checkpoint_rows(result_dir: str) -> list[dict[str, Any]]:
    """Per-question rows from results_incremental.jsonl (sharded or not).

    The aggregate JSON is only written when a run finishes, so falling back to
    the checkpoints lets a partially-completed arm still be compared. The
    paired test already restricts itself to questions both arms answered, so a
    short arm costs power, not validity.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    paths = sorted(glob.glob(os.path.join(result_dir, "rank_*", "results_incremental.jsonl")))
    paths += sorted(glob.glob(os.path.join(result_dir, "results_incremental.jsonl")))
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                key = str(item.pop("_key", "")) or json.dumps(item, sort_keys=True)[:200]
                if key in seen:
                    continue
                seen.add(key)
                rows.append(item)
    return rows


def _latest(result_dir: str, pattern: str) -> dict[str, Any] | None:
    hits = sorted(glob.glob(os.path.join(result_dir, pattern)))
    if not hits:
        return None
    with open(hits[-1], encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- OVO

def ovo_scores(result_dir: str) -> tuple[dict[str, tuple[int, int]], list[dict]]:
    payload = _latest(result_dir, "hub_and_spoke_results_*.json")
    per_task: dict[str, list[int]] = defaultdict(list)
    rows: list[dict] = []

    if payload is None:
        rows = _checkpoint_rows(result_dir)
        if not rows:
            raise SystemExit(f"no results (aggregate or checkpoint) in {result_dir}")
        print(f"  [{os.path.basename(result_dir)}] run unfinished; "
              f"using {len(rows)} checkpointed questions")
    else:
        for section in ("backward", "realtime"):
            rows.extend(payload.get(section, []))
        for r in payload.get("forward", []):
            task = r["task"]
            for item in r.get("test_info", []):
                if task == "REC":
                    per_task["REC"].append(score_count(item.get("response"), item["count"]))
                elif task in ("SSR", "CRR"):
                    per_task[task].append(score_yes_no(item.get("response"), item["type"]))
            rows.append(r)

    for r in rows:
        if r.get("task") in ("REC", "SSR", "CRR"):
            continue
        per_task[r["task"]].append(score_mcq(r.get("response"), r.get("ground_truth", "")))

    return {t: (sum(v), len(v)) for t, v in per_task.items()}, rows


OVO_GROUPS = {
    "Backward": ["EPM", "ASI", "HLD"],
    "Real-Time": ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"],
    "Forward": ["REC", "SSR", "CRR"],
}


# ------------------------------------------------------------------ StreamingBench

def sb_scores(result_dir: str) -> tuple[dict[str, tuple[int, int]], list[dict]]:
    payload = _latest(result_dir, "*results*.json")
    if payload is None:
        rows = _checkpoint_rows(result_dir)
        if not rows:
            raise SystemExit(f"no results (aggregate or checkpoint) in {result_dir}")
        print(f"  [{os.path.basename(result_dir)}] run unfinished; "
              f"using {len(rows)} checkpointed questions")
    else:
        rows = payload.get("results", payload if isinstance(payload, list) else [])
    per_task: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        per_task[str(r.get("task_type", "?"))].append(1 if r.get("correct") else 0)
    return {t: (sum(v), len(v)) for t, v in per_task.items()}, rows


# --------------------------------------------------------------------------- report

def paired_tallies(base_rows, gated_rows, key_fn, correct_fn, task_fn):
    """Per-task (correct, n) for both arms over the questions BOTH answered.

    Scoring each arm over its own row set would compare different denominators
    and, worse, render a task the gated arm has not reached yet as 0.00%
    rather than as missing.  Restricting to the shared set keeps the table
    consistent with the paired test below.
    """
    base = {key_fn(r): r for r in base_rows}
    gated = {key_fn(r): r for r in gated_rows}
    shared = set(base) & set(gated)
    b: dict[str, list[int]] = defaultdict(list)
    g: dict[str, list[int]] = defaultdict(list)
    for k in shared:
        t = task_fn(base[k])
        b[t].append(correct_fn(base[k]))
        g[t].append(correct_fn(gated[k]))
    return ({t: (sum(v), len(v)) for t, v in b.items()},
            {t: (sum(v), len(v)) for t, v in g.items()},
            len(shared), len(base), len(gated))


def delta_table(name: str, base: dict, gated: dict, groups: dict[str, list[str]] | None,
                coverage: tuple[int, int, int] | None = None) -> None:
    print(f"\n{'=' * 74}\n{name}\n{'=' * 74}")
    if coverage:
        shared, nb, ng = coverage
        if shared < max(nb, ng):
            print(f"scored on the {shared} questions both arms answered "
                  f"(task_label has {nb}, keyword has {ng})")
    print(f"{'task':<22}{'task_label':>14}{'keyword':>12}{'delta':>10}{'n':>8}")

    def line(label: str, b: tuple[int, int], g: tuple[int, int], indent: str = "") -> None:
        if not b[1] or not g[1]:
            print(f"{indent + label:<22}{'-':>14}{'-':>12}{'not yet run':>10}{0:>8}")
            return
        ba = 100.0 * b[0] / b[1]
        ga = 100.0 * g[0] / g[1]
        mark = "" if abs(ga - ba) < 0.005 else ("  <-- drop" if ga < ba else "  <-- gain")
        print(f"{indent + label:<22}{ba:>13.2f}%{ga:>11.2f}%{ga - ba:>+9.2f}{b[1]:>8}{mark}")

    tasks = groups or {"": sorted(set(base) | set(gated))}
    grand_b = grand_g = grand_n = 0
    for group, members in tasks.items():
        members = [t for t in members if t in base or t in gated]
        if not members:
            continue
        if group:
            present = [t for t in members if base.get(t, (0, 0))[1] and gated.get(t, (0, 0))[1]]
            gb = (sum(base[t][0] for t in present), sum(base[t][1] for t in present))
            gg = (sum(gated[t][0] for t in present), sum(gated[t][1] for t in present))
            line(group, gb, gg)
            grand_b += gb[0]; grand_g += gg[0]; grand_n += gb[1]
        for t in members:
            line(t, base.get(t, (0, 0)), gated.get(t, (0, 0)), indent="  " if group else "")
    if grand_n:
        print(f"{'-' * 74}")
        line("OVERALL", (grand_b, grand_n), (grand_g, grand_n))


def gate_breakdown(rows: list[dict], correct_key) -> None:
    """Accuracy of the gated run split by which bucket the gate chose."""
    by_bucket: dict[str, list[int]] = defaultdict(list)
    flipped: list[dict] = []
    for r in rows:
        bucket = r.get("gate_bucket")
        if bucket is None:
            continue
        by_bucket[bucket].append(correct_key(r))
        old = r.get("route_use_memory_task_label")
        new = r.get("route_use_memory")
        if old is not None and new is not None and bool(old) != bool(new):
            flipped.append(r)
    if not by_bucket:
        return
    print("\n  gated run, by gate bucket:")
    for bucket, vals in sorted(by_bucket.items()):
        print(f"    {bucket:<16} n={len(vals):<6} accuracy={100.0 * sum(vals) / len(vals):.2f}%")
    if flipped:
        vals = [correct_key(r) for r in flipped]
        print(f"\n  questions the gate routed differently from the task label: {len(flipped)}")
        print(f"    accuracy on those: {100.0 * sum(vals) / len(vals):.2f}%")


def mcnemar(name: str, base_rows: list[dict], gated_rows: list[dict],
            key_fn, correct_fn) -> None:
    """Paired significance test on the two arms.

    Comparing marginal accuracies is the wrong test here and badly
    underpowered: at n=1468 the standard error on a ~70% accuracy is ~1.2 pp,
    while the gate's expected effect is ~0.5 pp.  But the arms are paired --
    same questions, same model, same seed, only --routing differs -- and most
    questions get the identical route and therefore the identical answer.
    Only the discordant pairs (one arm right, the other wrong) carry
    information, so McNemar's test on those is far more sensitive.
    """
    base = {key_fn(r): correct_fn(r) for r in base_rows}
    gated = {key_fn(r): correct_fn(r) for r in gated_rows}
    shared = sorted(set(base) & set(gated))
    if not shared:
        print(f"\n  [{name}] no overlapping questions between the arms; skipping paired test")
        return

    both = only_base = only_gated = neither = 0
    for k in shared:
        b, g = base[k], gated[k]
        if b and g:
            both += 1
        elif b and not g:
            only_base += 1
        elif g and not b:
            only_gated += 1
        else:
            neither += 1

    n_disc = only_base + only_gated
    print(f"\n  paired comparison on {len(shared)} shared questions:")
    print(f"    both correct            {both}")
    print(f"    only task_label correct {only_base}   <- gate lost these")
    print(f"    only keyword correct    {only_gated}   <- gate won these")
    print(f"    both wrong              {neither}")
    print(f"    net change              {only_gated - only_base:+d} questions "
          f"({(only_gated - only_base) / len(shared) * 100:+.2f} pp)")

    if n_disc == 0:
        print("    no discordant pairs -- the two arms answered identically")
        return
    # Exact binomial two-sided p under H0: P(discordant favours either arm)=0.5
    from math import comb
    k = min(only_base, only_gated)
    tail = sum(comb(n_disc, i) for i in range(k + 1)) / (2 ** n_disc)
    p = min(1.0, 2 * tail)
    print(f"    McNemar exact (n_discordant={n_disc}): p = {p:.4f}"
          f"{'  <- significant at 0.05' if p < 0.05 else '  (not significant)'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ovo_baseline")
    ap.add_argument("--ovo_gated")
    ap.add_argument("--sb_baseline")
    ap.add_argument("--sb_gated")
    args = ap.parse_args()

    if args.ovo_baseline and args.ovo_gated:
        _, brows = ovo_scores(args.ovo_baseline)
        _, grows = ovo_scores(args.ovo_gated)
        ovo_correct = lambda r: score_mcq(r.get("response"), r.get("ground_truth", ""))
        ovo_key = lambda r: f"{r.get('task')}:{r.get('id')}"
        b, g, shared, nb, ng = paired_tallies(
            brows, grows, ovo_key, ovo_correct, lambda r: r.get("task"))
        delta_table("OVO-Bench", b, g, OVO_GROUPS, (shared, nb, ng))
        gate_breakdown(grows, ovo_correct)
        mcnemar("OVO-Bench", brows, grows, ovo_key, ovo_correct)

    if args.sb_baseline and args.sb_gated:
        _, brows = sb_scores(args.sb_baseline)
        _, grows = sb_scores(args.sb_gated)
        sb_correct = lambda r: 1 if r.get("correct") else 0
        sb_key = lambda r: f"{r.get('video')}|{r.get('time_stamp')}|{str(r.get('question'))[:60]}"
        b, g, shared, nb, ng = paired_tallies(
            brows, grows, sb_key, sb_correct, lambda r: str(r.get("task_type", "?")))
        delta_table("StreamingBench", b, g, None, (shared, nb, ng))
        gate_breakdown(grows, sb_correct)
        mcnemar("StreamingBench", brows, grows, sb_key, sb_correct)

    print()


if __name__ == "__main__":
    main()
