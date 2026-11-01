import json

from data_agent_bench.llm import LLMResponse, ToolCall

from dapt.repair import RepairingClient, repair_arguments, unescape_code


def test_string_encoded_arguments_are_decoded():
    raw, applied = repair_arguments("list_files", json.dumps("{}"))
    assert json.loads(raw) == {} and applied == ["string_encoded_arguments"]
    inner = json.dumps({"filename": "a.csv"})
    raw, applied = repair_arguments("inspect_csv", json.dumps(inner))
    assert json.loads(raw) == {"filename": "a.csv"}


def test_double_escaped_code_is_unescaped_only_when_that_fixes_it():
    broken = "import pandas as pd\ndf = pd.read_csv('a.csv')\nprint(len(df))"
    assert unescape_code(broken) == broken.replace("\n", "\n")
    # Code that compiles, or already has real newlines, is left alone.
    assert unescape_code("print('a\nb')") == "print('a\nb')"
    assert unescape_code("x = 1\nprint('\n')") == "x = 1\nprint('\n')"
    # A fix that still does not compile is not applied.
    assert unescape_code("def f(:\n  pass") == "def f(:\n  pass"


def test_valid_arguments_are_untouched():
    raw = json.dumps({"code": "print(1)"})
    assert repair_arguments("run_python", raw) == (raw, [])
    assert repair_arguments("run_python", "not json") == ("not json", [])


def test_repairing_client_counts_repairs():
    class Fake:
        def chat(self, messages, tools):
            return LLMResponse(None, [ToolCall("1", "list_files", json.dumps("{}"))])

    client = RepairingClient(Fake())
    response = client.chat([], [])
    assert response.tool_calls[0].arguments == "{}"
    assert client.repairs == ["string_encoded_arguments"]


def test_short_text_reply_becomes_the_final_answer():
    class Fake:
        def __init__(self, content):
            self.content = content

        def chat(self, messages, tools):
            return LLMResponse(self.content, [])

    client = RepairingClient(Fake("1470"))
    call = client.chat([], []).tool_calls[0]
    assert call.name == "final_answer" and json.loads(call.arguments) == {"answer": "1470"}
    assert client.repairs == ["text_as_final_answer"]
    # Multi-line or long text is reasoning, not an answer.
    assert RepairingClient(Fake("Let me check.\nFirst...")).chat([], []).tool_calls == []
    assert RepairingClient(Fake("x" * 150)).chat([], []).tool_calls == []
