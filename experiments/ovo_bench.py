"""OVO-Bench task spec: prompts, answer parsing, scoring.

Dependency-free on purpose -- the offline tools (verify_retrieval_gate.py,
compare_gate_ab.py) import this without pulling in torch, and so does
evaluate_ovo.py / evaluate_ovo_ablations.py, which hold the actual pipeline.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REAL_TIME_TASKS = ["OCR", "ACR", "ATR", "STU", "FPD", "OJR"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]
MCQ_TASKS = BACKWARD_TASKS + REAL_TIME_TASKS

# Prompt templates verbatim from official OVO-Bench.
MCQ_PROMPT = """
Question: {}
Options:
{}

Respond only with the letter corresponding to your chosen option (e.g., A, B, C).
Do not include any additional text or explanation in your response.
"""

REC_PROMPT = """
You're watching a video in which people may perform a certain type of action repetively.
The person performing this kind of action are referred to as 'they' in the following statement.
You're task is to count how many times have different people in the video perform this kind of action in total.
One complete motion counts as one.
Now, answer the following question: {}
Provide your answer as a single number (e.g., 0, 1, 2, 3…) indicating the total count.
Do not include any additional text or explanation in your response.
"""

SSR_PROMPT = """
You're watching a tutorial video which contain a sequential of steps.
The following is one step from the whole procedures:
{}
Your task is to determine if the man or woman in the video is currently performing this step.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""

CRR_PROMPT = """
You're responsible of answering questions based on the video content.
The following question are relevant to the latest frames, i.e. the end of the video.
{}
Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, provide enough information for answering the question.
Answer only with "Yes" or "No".
Do not include any additional text or explanation in your response.
"""

# HLD options sometimes carry an explicit "not enough information" choice; the
# prompt points the model at it so it abstains instead of guessing a content
# option when retrieval came back empty.
_ABSTAIN_PATTERNS = (
    r"\bnot enough (?:visual )?(?:information|evidence)\b",
    r"\binsufficient (?:visual )?(?:information|evidence)\b",
    r"\bunable to answer\b",
    r"\bcannot answer\b",
    r"\bcan't answer\b",
    r"\bcannot (?:be )?(?:determined|tell)\b",
    r"\bcan't (?:be )?(?:determined|tell)\b",
    r"\bunable to (?:determine|tell)\b",
    r"\bunknown\b",
    r"\bnot (?:shown|visible)\b",
    r"\bnone of the above\b",
)


def find_abstain_option(options: list[str]) -> str | None:
    for idx, option in enumerate(options):
        if any(re.search(p, str(option).lower()) for p in _ABSTAIN_PATTERNS):
            return chr(65 + idx)
    return None


def build_prompt(task: str, anno: dict[str, Any], index: int = 0) -> str:
    if task in MCQ_TASKS:
        options = anno["options"]
        opts = "; ".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options)) + ";"
        prompt = MCQ_PROMPT.format(anno["question"], opts)
        abstain = find_abstain_option(options) if task == "HLD" else None
        if abstain:
            prompt = prompt.replace(
                "Respond only with the letter corresponding to your chosen option",
                "For HLD, choose a content option only when retrieved historical "
                "evidence directly answers the question. If the evidence is "
                "insufficient, only indirectly related, or retrieval reports "
                f'"未找到匹配节点", choose {abstain}.\n'
                "Respond only with the letter corresponding to your chosen option",
                1,
            )
        return prompt
    if task == "REC":
        return REC_PROMPT.format(f"How many times did they {anno['activity']}?")
    if task == "SSR":
        return SSR_PROMPT.format(anno["test_info"][index]["step"])
    if task == "CRR":
        return CRR_PROMPT.format(anno["question"])
    return anno.get("question", "")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def extract_mcq_letter(response: str | None) -> str | None:
    """A/B/C/D from free text, accepting a bare 1-4 as a fallback."""
    if response is None or not str(response).strip():
        return None
    text = str(response).strip().upper()
    if match := re.search(r"\b([A-D])\b", text):
        return match.group(1)
    if match := re.search(r"\b([1-4])\b", text):
        return chr(64 + int(match.group(1)))
    return None


def score_mcq(response: str | None, gt: str) -> int:
    pred = extract_mcq_letter(response)
    return int(pred is not None and pred == str(gt).upper())


def score_count(response: str | None, gt_count: int) -> int:
    nums = re.findall(r"\d+", str(response or ""))
    return int(bool(nums) and "".join(nums) == str(gt_count))


def score_yes_no(response: str | None, gt_type: int) -> int:
    """gt_type: 0 = No, 1 = Yes."""
    text = str(response or "").strip().upper()
    if not text:
        return 0
    if (text == "N" or "NO" in text) and gt_type == 0:
        return 1
    if (text == "Y" or "YES" in text) and gt_type == 1:
        return 1
    return 0


def score_forward_rows(forward: list[dict[str, Any]], response_key: str = "response") -> dict[str, list[int]]:
    """Per-task 0/1 marks for forward rows (each holds a list of sub-tests)."""
    by_task: dict[str, list[int]] = defaultdict(list)
    for result in forward:
        task = result.get("task")
        for item in result.get("test_info", []):
            if task == "REC":
                by_task["REC"].append(score_count(item.get(response_key), item["count"]))
            elif task in {"SSR", "CRR"}:
                by_task[task].append(score_yes_no(item.get(response_key), item["type"]))
    return by_task


def _accuracy_table(by_task: dict[str, list[int]]) -> dict[str, dict[str, Any]]:
    return {
        task: {
            "correct": sum(vals),
            "total": len(vals),
            "accuracy": 100.0 * sum(vals) / len(vals),
        }
        for task, vals in by_task.items()
        if vals
    }


def score_all(
    backward: list[dict[str, Any]],
    realtime: list[dict[str, Any]],
    forward: list[dict[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for section, rows in (("backward", backward), ("realtime", realtime)):
        by_task: dict[str, list[int]] = defaultdict(list)
        for row in rows:
            by_task[row["task"]].append(score_mcq(row.get("response"), row["ground_truth"]))
        summary[section] = _accuracy_table(by_task)
    summary["forward"] = _accuracy_table(score_forward_rows(forward))
    return summary


SECTION_TITLES = (
    ("backward", "Backward Tracing"),
    ("realtime", "Real-time Perception"),
    ("forward", "Forward Responding"),
)


def print_report(
    label: str,
    backward: list[dict[str, Any]],
    realtime: list[dict[str, Any]],
    forward: list[dict[str, Any]],
) -> dict[str, Any]:
    """Print the three-section OVO table; returns the summary it printed."""
    summary = score_all(backward, realtime, forward)
    print("\n" + "=" * 60)
    print(f"OVO-Bench Results ({label})")
    print("=" * 60)

    section_avgs: list[float] = []
    for section, title in SECTION_TITLES:
        rows = summary[section]
        if not rows:
            continue
        print(f"\n{title}:")
        for task, stats in rows.items():
            print(f"  {task}: {stats['accuracy']:.2f}% ({stats['correct']}/{stats['total']})")
        avg = sum(s["accuracy"] for s in rows.values()) / len(rows)
        section_avgs.append(avg)
        print(f"  {title.split()[0]} Avg.: {avg:.2f}%")

    if section_avgs:
        print("\n" + "=" * 60)
        print(f"Total Avg.: {sum(section_avgs) / len(section_avgs):.2f}%")
        print("=" * 60)
    return summary
