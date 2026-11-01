"""Turn scored rollouts into an SFT dataset (rejection sampling).

Only correct episodes that followed the tool protocol are kept, at most `per_task` per task
(the shortest first) and `per_template` per template, so easy templates do not dominate.
Each kept episode is rewritten as a chat with the agent's standard system prompt, whatever
prompt it was sampled with (see `dapt.hint`), and with clean tool calls: repaired arguments,
and a final_answer call where the model had written its answer as text. Assistant text is
dropped, and so are text-only turns (and the agent's nudge after them): the model's actions
are what is learned.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from data_agent_bench.agent import AGENT_TOOL_SPECS, SYSTEM_PROMPT

BAD_FLAGS = {"tool_hallucination", "repeated_call", "no_evidence"}
# Words that would show the model echoing the hinted prompt instead of just following it.
HINT_ECHOES = ("procedure", "instructions say", "as instructed")


def keep(record: dict) -> bool:
    if not record["correct"] or record["stop_reason"] != "final_answer":
        return False
    if BAD_FLAGS & set(record["flags"]):
        return False
    text = json.dumps(record["trajectory"], ensure_ascii=False).lower()
    return not any(echo in text for echo in HINT_ECHOES)


def to_messages(record: dict) -> list[dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": record["question"]},
    ]
    for step in record["trajectory"]:
        if not step["tool_calls"]:
            continue
        calls = [
            {
                "type": "function",
                "function": {"name": ex["name"], "arguments": json.loads(ex["arguments"])},
            }
            for ex in step["tool_calls"]
        ]
        messages.append({"role": "assistant", "content": "", "tool_calls": calls})
        for ex in step["tool_calls"]:
            if ex["name"] != "final_answer":
                messages.append({"role": "tool", "content": ex["output"]})
    return messages


def select(records: list[dict], per_task: int = 1, per_template: int | None = None) -> list[dict]:
    by_task = defaultdict(list)
    for r in records:
        if keep(r):
            by_task[r["task_id"]].append(r)
    chosen, per_tpl = [], defaultdict(int)
    for task_id in sorted(by_task):
        candidates = sorted(by_task[task_id], key=lambda r: (r["n_steps"], r["sample"]))
        for r in candidates[:per_task]:
            key = (r["domain"], r["template"])
            if per_template and per_tpl[key] >= per_template:
                break
            per_tpl[key] += 1
            chosen.append(r)
    return chosen


def build_sft(
    records: list[dict],
    out_path: Path,
    per_task: int = 1,
    per_template: int | None = None,
) -> list[dict]:
    rows = [
        {
            "task_id": r["task_id"],
            "domain": r["domain"],
            "template": r["template"],
            "category": r["category"],
            "messages": to_messages(r),
            "tools": AGENT_TOOL_SPECS,
        }
        for r in select(records, per_task, per_template)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows
