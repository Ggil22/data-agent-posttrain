"""SFT of the student with LoRA, on the examples built by `dapt sft-data`.

Meant for Kaggle's 2x T4: `torchrun --nproc_per_node 2 -m dapt.train ...`. T4s have no bf16,
so the frozen base runs in fp16, the same precision vLLM serves it in, and the LoRA weights are
trained in fp32 with fp16 autocast. The loss covers only the assistant turns (the tool calls and
the end-of-turn token); system, user and tool messages are context.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

# Everything the model says: from the assistant header to its end-of-turn token, inclusive.
ASSISTANT_SPAN = re.compile(r"<\|im_start\|>assistant\n(.*?<\|im_end\|>)", re.S)


def assistant_char_spans(text: str) -> list[tuple[int, int]]:
    return [m.span(1) for m in ASSISTANT_SPAN.finditer(text)]


def tokenize(tokenizer, row: dict, max_length: int) -> dict | None:
    """input_ids and labels (-100 outside assistant turns); None if the example is too long."""
    text = tokenizer.apply_chat_template(row["messages"], tools=row["tools"], tokenize=False)
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    if len(enc["input_ids"]) > max_length:
        return None
    spans = assistant_char_spans(text)
    labels = []
    for tok_id, (start, end) in zip(enc["input_ids"], enc["offset_mapping"], strict=True):
        inside = any(s <= start and end <= e for s, e in spans) and end > start
        labels.append(tok_id if inside else -100)
    return {"input_ids": enc["input_ids"], "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    import torch

    width = max(len(b["input_ids"]) for b in batch)
    ids, labels, mask = [], [], []
    for b in batch:
        pad = width - len(b["input_ids"])
        ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        mask.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


def main(argv: list[str] | None = None) -> None:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    p = argparse.ArgumentParser(description="LoRA SFT on dapt examples.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--batch", type=int, default=16, help="Effective batch size (examples).")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--eval-fraction", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=-1, help="Stop early (smoke tests).")
    p.add_argument("--logging-steps", type=int, default=5)
    args = p.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in args.data.open(encoding="utf-8") if line.strip()]
    features = [f for f in (tokenize(tokenizer, r, args.max_length) for r in rows) if f]
    print(f"{len(features)}/{len(rows)} examples within {args.max_length} tokens", flush=True)
    random.Random(0).shuffle(features)
    n_eval = max(1, int(len(features) * args.eval_fraction))
    eval_set, train_set = features[:n_eval], features[n_eval:]

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    world = max(1, torch.cuda.device_count())
    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # a float below 1 is a fraction of the total steps
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=max(1, args.batch // world),
        fp16=torch.cuda.is_available(),  # CPU only for local dry runs
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()
    if trainer.is_world_process_zero():
        adapter = args.out / "adapter"
        model.save_pretrained(adapter)
        tokenizer.save_pretrained(adapter)
        history = trainer.state.log_history
        (args.out / "log_history.json").write_text(json.dumps(history, indent=2))
        print(f"adapter saved to {adapter}", flush=True)


if __name__ == "__main__":
    main()
