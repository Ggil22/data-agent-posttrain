import json

from data_agent_bench.agent import SYSTEM_PROMPT
from data_agent_bench.llm import LLMResponse

from dapt.hint import HINTED_PROMPT, HintingClient
from dapt.sft import keep, select, to_messages


def exchange(name, arguments, output=""):
    return {"name": name, "arguments": json.dumps(arguments), "output": output}


def record(task_id="t-0-a", sample=0, correct=True, steps=None, flags=()):
    steps = steps or [
        {"content": None, "tool_calls": [exchange("list_files", {}, "a.csv")]},
        {"content": "42", "tool_calls": [exchange("final_answer", {"answer": "42"})]},
    ]
    return {
        "task_id": task_id,
        "sample": sample,
        "domain": "t",
        "template": "a",
        "category": "aggregation",
        "question": "How many?",
        "correct": correct,
        "stop_reason": "final_answer",
        "flags": list(flags),
        "n_steps": len(steps),
        "trajectory": steps,
    }


def test_messages_use_the_standard_prompt_and_clean_calls():
    messages = to_messages(record())
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    final = messages[-1]
    assert final["content"] == ""  # the answer written as text is dropped
    assert final["tool_calls"][0]["function"] == {
        "name": "final_answer",
        "arguments": {"answer": "42"},
    }


def test_text_only_turns_are_dropped():
    steps = record()["trajectory"]
    steps.insert(1, {"content": "Let me think about it.", "tool_calls": []})
    roles = [m["role"] for m in to_messages(record(steps=steps))]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]


def test_only_clean_correct_episodes_are_kept():
    assert keep(record())
    assert not keep(record(correct=False))
    assert not keep(record(flags=["repeated_call"]))
    echo = [{"content": None, "tool_calls": [exchange("run_python", {"code": "# procedure"})]}]
    assert not keep(record(steps=echo + record()["trajectory"]))


def test_select_prefers_short_episodes_and_caps_templates():
    long = record(sample=0, steps=[*record()["trajectory"][:1], *record()["trajectory"]])
    short = record(sample=1)
    assert select([long, short]) == [short]
    failed_once = record(sample=1, flags=["code_error"])
    assert select([long, failed_once]) == [long]
    many = [record(task_id=f"t-{i}-a") for i in range(5)]
    assert len(select(many, per_template=3)) == 3


def test_hinting_client_swaps_only_the_system_prompt():
    seen = []

    class Fake:
        def chat(self, messages, tools):
            seen.append(messages)
            return LLMResponse(None, [])

    original = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "q"}]
    HintingClient(Fake()).chat(original, [])
    assert seen[0][0]["content"] == HINTED_PROMPT and seen[0][1] == original[1]
    assert original[0]["content"] == SYSTEM_PROMPT  # the agent's own history is untouched
