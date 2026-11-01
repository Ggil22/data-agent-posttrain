"""Guided sampling for SFT data: context distillation of a working procedure.

The base model rarely reads the documentation in its working directory, so it falls for the data
traps, and it often writes its answer as text instead of calling final_answer. To collect correct
trajectories it is sampled with help, and the SFT examples are rewritten with the standard
prompt (see `dapt.sft`), so the fine-tuned model learns to work this way without being told:

- `HintingClient` adds the procedure to the system prompt. On its own this barely changes what
  the model does (pilot: it read the documentation in 41/120 episodes, against 38 without it).
- `GuidingClient` scripts the first two actions: list the files, then read every documentation
  file. The model takes over from there. These actions are kept in the SFT examples: they are
  the habit being taught.

Both are generic: they name no task, table or answer.
"""

from __future__ import annotations

import json

from data_agent_bench.agent import SYSTEM_PROMPT
from data_agent_bench.llm import LLMClient, LLMResponse, ToolCall

PROCEDURE = """
How to work (follow this every time):
1. Call list_files. If there is a documentation file (such as DATA_DICTIONARY.md), read all of \
it with run_python, for example print(open("DATA_DICTIONARY.md").read()).
2. Documentation describes data quality issues and business rules: duplicate rows, test or \
internal records to exclude, sentinel or error codes, mixed units, mixed date formats, \
inconsistent labels, how to treat missing values, fees and prices. Apply every rule that matters \
for the question before computing the answer.
3. Inspect the columns you need, compute the answer with run_python, and check that it makes sense.
4. Submit by calling the final_answer tool. Never write the answer as plain text.
"""

HINTED_PROMPT = SYSTEM_PROMPT + PROCEDURE
DOC_SUFFIXES = (".md", ".txt")


class HintingClient:
    """Wraps an LLM client and replaces the agent's system prompt with the hinted one."""

    def __init__(self, inner: LLMClient):
        self.inner = inner

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        if messages and messages[0]["role"] == "system":
            messages = [{"role": "system", "content": HINTED_PROMPT}, *messages[1:]]
        return self.inner.chat(messages, tools)


class GuidingClient:
    """Wraps an LLM client and scripts the first actions of an episode: list the files, then
    read the documentation files, if any. Later turns go to the wrapped client."""

    def __init__(self, inner: LLMClient):
        self.inner = inner

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        turns = sum(m["role"] == "assistant" for m in messages)
        if turns == 0:
            return _call("guide_list", "list_files", {})
        if turns == 1 and messages[-1]["role"] == "tool":
            names = [line.split("\t")[0] for line in messages[-1]["content"].splitlines()]
            docs = [n for n in names if n.lower().endswith(DOC_SUFFIXES)]
            if docs:
                code = "\n".join(f'print(open("{name}").read())' for name in docs)
                return _call("guide_docs", "run_python", {"code": code})
        return self.inner.chat(messages, tools)


def _call(call_id: str, name: str, arguments: dict) -> LLMResponse:
    return LLMResponse(None, [ToolCall(call_id, name, json.dumps(arguments))])
