"""Hinted sampling for SFT data: context distillation of a working procedure.

The base model rarely reads the documentation in its working directory, so it falls for the data
traps, and it often writes its answer as text instead of calling final_answer. To collect correct
trajectories it is sampled with a system prompt that spells out the procedure. The SFT examples
are then rewritten with the standard prompt (see `dapt.sft`), so the fine-tuned model learns to
follow the procedure without being told. The hint is generic: it names no task, table or answer.
"""

from __future__ import annotations

from data_agent_bench.agent import SYSTEM_PROMPT
from data_agent_bench.llm import LLMClient, LLMResponse

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


class HintingClient:
    """Wraps an LLM client and replaces the agent's system prompt with the hinted one."""

    def __init__(self, inner: LLMClient):
        self.inner = inner

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        if messages and messages[0]["role"] == "system":
            messages = [{"role": "system", "content": HINTED_PROMPT}, *messages[1:]]
        return self.inner.chat(messages, tools)
