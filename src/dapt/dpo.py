"""Turn scored on-policy rollouts of the SFT model into DPO preference pairs.

The SFT model is sampled several times per training task (`kaggle/dpo-generate`). Tasks where
some samples are correct and some are not give a pair: a correct episode (chosen, filtered as
for SFT, see `dapt.sft.keep`) and a failed one (rejected) for the same question. Tasks the SFT
model never solves can take their chosen episode from other rollouts (`extra_chosen`, e.g. the
guided samples of the base model used for SFT): off-policy, but it is where the model fails.

Both sides are rewritten exactly like the SFT examples (standard system prompt, tool calls only),
so they differ only in the model's actions. Failed episodes are rejected whatever went wrong
(wrong answer, false refusal, answering an unanswerable question, a loop), except API errors:
those are timeouts whose generation was never recorded, so there is nothing to push down.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from data_agent_bench.agent import AGENT_TOOL_SPECS

from dapt.sft import keep, to_messages


def parsable(record: dict) -> bool:
    """Every tool call's arguments are JSON: the chat template needs them as a dict."""
    for step in record["trajectory"]:
        for ex in step["tool_calls"]:
            try:
                if not isinstance(json.loads(ex["arguments"]), dict):
                    return False
            except (TypeError, ValueError):
                return False
    return True


def rejectable(record: dict) -> bool:
    return not record["correct"] and record["outcome"] != "api_error" and parsable(record)


def chosen_order(record: dict) -> tuple:
    # As for SFT: episodes without code errors first, then shortest.
    return ("code_error" in record["flags"], record["n_steps"], record["sample"])


def pair_records(
    records: list[dict],
    extra_chosen: list[dict] | None = None,
    per_task: int = 2,
    per_category: dict[str, int] | None = None,
) -> list[tuple[dict, dict, str]]:
    """(chosen, rejected, source) triples; source is "on_policy" or "off_policy"."""
    chosen, rejected, extra = defaultdict(list), defaultdict(list), defaultdict(list)
    for r in records:
        if keep(r):
            chosen[r["task_id"]].append(r)
        elif rejectable(r):
            rejected[r["task_id"]].append(r)
    for r in extra_chosen or []:
        if keep(r):
            extra[r["task_id"]].append(r)
    task_ids = sorted(rejected)
    random.Random(0).shuffle(task_ids)  # caps cut evenly across domains and seeds
    per_category = per_category or {}
    pairs, per_cat = [], defaultdict(int)
    for task_id in task_ids:
        good, source = chosen[task_id], "on_policy"
        if not good:
            good, source = extra[task_id], "off_policy"
        if not good:
            continue
        good = sorted(good, key=chosen_order)
        # Different failures first, so a task's pairs do not all teach the same thing.
        firsts, rest, seen = [], [], set()
        for r in sorted(rejected[task_id], key=lambda r: r["sample"]):
            (rest if r["outcome"] in seen else firsts).append(r)
            seen.add(r["outcome"])
        for i, r in enumerate((firsts + rest)[:per_task]):
            cat = r["category"]
            if cat in per_category and per_cat[cat] >= per_category[cat]:
                break
            c = good[i % len(good)]
            if to_messages(c) == to_messages(r):
                continue
            per_cat[cat] += 1
            pairs.append((c, r, source))
    return pairs


def build_dpo(
    records: list[dict],
    out_path: Path,
    extra_chosen: list[dict] | None = None,
    per_task: int = 2,
    per_category: dict[str, int] | None = None,
) -> list[dict]:
    rows = [
        {
            "task_id": c["task_id"],
            "domain": c["domain"],
            "template": c["template"],
            "category": c["category"],
            "source": source,
            "rejected_outcome": r["outcome"],
            "chosen": to_messages(c),
            "rejected": to_messages(r),
            "tools": AGENT_TOOL_SPECS,
        }
        for c, r, source in pair_records(records, extra_chosen, per_task, per_category)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows
