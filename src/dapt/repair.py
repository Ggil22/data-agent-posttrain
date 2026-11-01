"""Optional repair of mechanical tool-call format errors, to separate format from reasoning.

Small models often fail an agent task before any analysis happens: Granite 4.0 Micro encodes
empty arguments as a JSON string ("{}" inside quotes), and Qwen3 models served in fp16 escape
newlines twice, so the code reaches Python as one line full of backslashes. Evaluating a model
both strictly and with these repairs shows how much of its failure rate (and of what training
later fixes) is formatting. Only unambiguous cases are repaired, and every repair is counted.
"""

from __future__ import annotations

import json
from dataclasses import replace

from data_agent_bench.llm import LLMClient, LLMResponse


def repair_arguments(name: str, raw: str) -> tuple[str, list[str]]:
    """Return the repaired raw arguments and the list of repairs applied."""
    repairs = []
    try:
        args = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        return raw, repairs
    if isinstance(args, str):  # a JSON object encoded as a JSON string
        try:
            inner = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return raw, repairs
        if not isinstance(inner, dict):
            return raw, repairs
        args = inner
        repairs.append("string_encoded_arguments")
    if name == "run_python" and isinstance(args, dict) and isinstance(args.get("code"), str):
        fixed = unescape_code(args["code"])
        if fixed != args["code"]:
            args = {**args, "code": fixed}
            repairs.append("double_escaped_code")
    return (json.dumps(args, ensure_ascii=False) if repairs else raw), repairs


def unescape_code(code: str) -> str:
    """Undo a second level of escaping, only when the code is broken and the fix compiles."""
    if "\n" in code or "\\n" not in code or _compiles(code):
        return code
    fixed = code.replace("\\n", "\n").replace("\\t", "\t").replace('\\"', '"')
    return fixed if _compiles(fixed) else code


def _compiles(code: str) -> bool:
    try:
        compile(code, "<agent_code>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


class RepairingClient:
    """Wraps an LLM client and repairs the tool calls of every response it returns."""

    def __init__(self, inner: LLMClient):
        self.inner = inner
        self.repairs: list[str] = []

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        response = self.inner.chat(messages, tools)
        calls = []
        for tc in response.tool_calls:
            arguments, applied = repair_arguments(tc.name, tc.arguments)
            self.repairs += applied
            calls.append(replace(tc, arguments=arguments) if applied else tc)
        return replace(response, tool_calls=calls)
