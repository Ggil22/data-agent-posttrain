from dapt.train import assistant_char_spans

TEXT = (
    "<|im_start|>system\nYou are...<|im_end|>\n"
    "<|im_start|>user\nHow many?<|im_end|>\n"
    "<|im_start|>assistant\n<tool_call>\na\n</tool_call><|im_end|>\n"
    "<|im_start|>user\n<tool_response>\n3\n</tool_response><|im_end|>\n"
    "<|im_start|>assistant\n<tool_call>\nb\n</tool_call><|im_end|>\n"
)


def test_only_assistant_turns_are_trained_on():
    spans = [TEXT[s:e] for s, e in assistant_char_spans(TEXT)]
    assert spans == [
        "<tool_call>\na\n</tool_call><|im_end|>",
        "<tool_call>\nb\n</tool_call><|im_end|>",
    ]
