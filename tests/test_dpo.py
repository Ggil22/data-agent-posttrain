import json

from dapt.dpo import build_dpo, pair_records, parsable
from dapt.dpo_train import dpo_margin


def exchange(name, arguments, output=""):
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return {"name": name, "arguments": args, "output": output}


def record(task_id="t-0-a", sample=0, answer="42", correct=True, outcome=None, steps=None):
    steps = steps or [
        {"content": None, "tool_calls": [exchange("list_files", {}, "a.csv")]},
        {"content": None, "tool_calls": [exchange("final_answer", {"answer": answer})]},
    ]
    return {
        "task_id": task_id,
        "sample": sample,
        "domain": "t",
        "template": "a",
        "category": "aggregation",
        "question": "How many?",
        "correct": correct,
        "outcome": outcome or ("correct" if correct else "wrong_answer"),
        "stop_reason": "final_answer",
        "flags": [],
        "n_steps": len(steps),
        "trajectory": steps,
    }


def wrong(sample, outcome="wrong_answer", **kw):
    return record(sample=sample, answer=f"wrong-{sample}", correct=False, outcome=outcome, **kw)


def test_pairs_need_a_correct_and_a_failed_episode_of_the_same_task():
    good = record(sample=0)
    pairs = pair_records([good, wrong(1), record(task_id="t-1-a")])
    assert pairs == [(good, wrong(1), "on_policy")]
    assert pair_records([wrong(1), wrong(2)]) == []  # never solved, no other source


def test_never_solved_tasks_take_the_chosen_side_from_other_rollouts():
    teacher = record(sample=0)
    assert pair_records([wrong(1)], extra_chosen=[teacher]) == [(teacher, wrong(1), "off_policy")]
    # An on-policy chosen episode wins over the other rollouts.
    own = record(sample=2)
    assert pair_records([own, wrong(1)], extra_chosen=[teacher])[0][0] is own


def test_rejected_episodes_skip_api_errors_and_unparsable_calls():
    good = record(sample=0)
    timeout = wrong(1, outcome="api_error")
    broken = wrong(2, steps=[{"content": None, "tool_calls": [exchange("run_python", "{oops")]}])
    assert not parsable(broken)
    assert pair_records([good, timeout, broken]) == []


def test_pairs_per_task_put_different_failures_first():
    good = record(sample=0)
    eps = [good, wrong(1), wrong(2), wrong(3, outcome="false_refusal")]
    pairs = pair_records(eps, per_task=2)
    assert [r["sample"] for _, r, _ in pairs] == [1, 3]
    assert len(pair_records(eps, per_task=5)) == 3
    assert len(pair_records(eps, per_task=5, per_category={"aggregation": 1})) == 1


def test_rows_use_sft_style_messages(tmp_path):
    rows = build_dpo([record(sample=0), wrong(1)], tmp_path / "dpo.jsonl")
    (row,) = rows
    assert row["chosen"][-1]["tool_calls"][0]["function"]["arguments"] == {"answer": "42"}
    assert row["rejected"][-1]["tool_calls"][0]["function"]["arguments"] == {"answer": "wrong-1"}
    assert row["chosen"][:4] == row["rejected"][:4]  # same prompt and first action
    assert row["rejected_outcome"] == "wrong_answer" and row["source"] == "on_policy"
    assert json.loads((tmp_path / "dpo.jsonl").read_text(encoding="utf-8")) == row


def test_margin_is_zero_at_the_reference_and_grows_with_chosen():
    assert dpo_margin((-10.0, -12.0), (-10.0, -12.0), 0.1) == 0
    assert dpo_margin((-9.0, -12.0), (-10.0, -12.0), 0.1) > 0
    assert dpo_margin((-10.0, -11.0), (-10.0, -12.0), 0.1) < 0
