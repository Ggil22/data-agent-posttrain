"""SFT of the student with LoRA, on the examples built by `dapt sft-data`.

Meant for Kaggle's 2x T4: `torchrun --nproc_per_node 2 -m dapt.train ...`. T4s have no bf16,
so the frozen base runs in fp16, the same precision vLLM serves it in, and the LoRA weights are
trained in fp32 with fp16 autocast. The loss covers only the assistant turns (the tool calls and
the end-of-turn token); system, user and tool messages are context.
"""

from __future__ import annotations

import argparse
import json
import os
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


def collate(batch: list[dict]) -> dict:
    """One example per device step (no padding). The model computes logits for the last
    position only; the loss computes the rest itself (see ChunkedLoss)."""
    import torch

    (example,) = batch
    labels = torch.tensor([example["labels"]])
    return {
        "input_ids": torch.tensor([example["input_ids"]]),
        "attention_mask": torch.ones_like(labels),
        "labels": labels,
        "logits_to_keep": 1,
    }


class ChunkedLoss:
    """Cross-entropy on the assistant tokens, without materialising the logits.

    Over a 151k vocabulary, logits for the ~2k assistant positions of a long example take more
    memory than a T4 has left (the first full run died there). The final hidden states are
    captured with a hook on the model's last norm, and the output projection plus cross-entropy
    run in chunks under activation checkpointing, so only one chunk of logits exists at a time.
    Used as the Trainer's `compute_loss_func`: returns the summed loss over `num_items_in_batch`.
    """

    def __init__(self, model, chunk: int = 256):
        base = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.lm_head = base.lm_head
        self.hidden = None
        base.model.norm.register_forward_hook(self._capture)
        self.chunk = chunk

    def _capture(self, module, inputs, output):
        self.hidden = output

    def __call__(self, outputs, labels, num_items_in_batch=None):
        import torch
        from torch.utils.checkpoint import checkpoint

        target = labels[0, 1:]
        keep = target != -100
        hidden, target = self.hidden[0, :-1][keep], target[keep]
        self.hidden = None
        total = hidden.new_zeros((), dtype=torch.float32)
        for i in range(0, len(target), self.chunk):
            h, t = hidden[i : i + self.chunk], target[i : i + self.chunk]
            total = total + checkpoint(self._chunk_loss, h, t, use_reentrant=False)
        return total / (num_items_in_batch if num_items_in_batch is not None else len(target))

    def _chunk_loss(self, hidden, target):
        import torch.nn.functional as F

        logits = self.lm_head(hidden).float()
        return F.cross_entropy(logits, target, reduction="sum")


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
    loss_fn = ChunkedLoss(model)

    model.train()  # gradient checkpointing is only active in training mode
    if torch.cuda.is_available():  # fail fast if the longest example does not fit
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        model.to(f"cuda:{torch.cuda.current_device()}")
        longest = max(features, key=lambda f: len(f["input_ids"]))
        batch = {k: v.to(model.device) if torch.is_tensor(v) else v
                 for k, v in collate([longest]).items()}  # fmt: skip
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(**{k: v for k, v in batch.items() if k != "labels"})
            loss_fn(outputs, batch["labels"]).backward()
        model.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 2**30
        # AdamW keeps two fp32 moments per trainable parameter, allocated at the first step.
        adam = 8 * sum(p.numel() for p in model.parameters() if p.requires_grad) / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(
            f"longest example ({len(longest['input_ids'])} tokens): peak {peak:.1f} GiB "
            f"+ {adam:.1f} GiB optimizer state, of {total:.1f} GiB",
            flush=True,
        )

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
        prediction_loss_only=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=collate,
        compute_loss_func=loss_fn,
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
