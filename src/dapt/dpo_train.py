"""DPO of the SFT student on the pairs built by `dapt dpo-data`.

Meant for Kaggle's 2x T4: `torchrun --nproc_per_node 2 -m dapt.dpo_train ...`. A T4 cannot hold
two copies of the model, so the SFT adapter itself is trained further, and the reference model
(the SFT model as it was) is used once, before training: its log-probabilities for every pair
are computed up front. The result is a single adapter on the base model, served and evaluated
exactly like the SFT one.

As in SFT, a trajectory's log-probability covers only the assistant turns (the tool calls and
the end-of-turn tokens); tool outputs are context. Chosen and rejected run as one padded batch
of two, so DDP sees one forward per backward. Precision is as in `dapt.train`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from dapt.train import ChunkedLoss, tokenize


def tokenize_pair(tokenizer, row: dict, max_length: int) -> dict | None:
    sides = [
        tokenize(tokenizer, {"messages": row[side], "tools": row["tools"]}, max_length)
        for side in ("chosen", "rejected")
    ]
    if None in sides:
        return None
    chosen, rejected = sides
    return {"chosen": chosen, "rejected": rejected, "category": row["category"]}


def pad_pair(pair: dict, pad_id: int) -> dict:
    """Chosen and rejected as one right-padded batch of two."""
    import torch

    sides = (pair["chosen"], pair["rejected"])
    length = max(len(s["input_ids"]) for s in sides)
    ids, mask, labels = [], [], []
    for s in sides:
        pad = length - len(s["input_ids"])
        ids.append(s["input_ids"] + [pad_id] * pad)
        mask.append([1] * len(s["input_ids"]) + [0] * pad)
        labels.append(s["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(ids),
        "attention_mask": torch.tensor(mask),
        "labels": torch.tensor(labels),
        "logits_to_keep": 1,
    }


def make_collate(pad_id: int):
    def collate(batch: list[dict]) -> dict:
        import torch

        (pair,) = batch
        return {
            **pad_pair(pair, pad_id),
            "ref": torch.tensor(pair["ref"], dtype=torch.float32),
            "chosen_tokens": torch.tensor(sum(t != -100 for t in pair["chosen"]["labels"])),
        }

    return collate


def dpo_margin(policy, ref, beta: float):
    """beta * (chosen log-ratio - rejected log-ratio); policy and ref are (chosen, rejected)."""
    return beta * ((policy[0] - ref[0]) - (policy[1] - ref[1]))


def main(argv: list[str] | None = None) -> None:
    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    p = argparse.ArgumentParser(description="DPO of the SFT LoRA adapter on dapt pairs.")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--sft-adapter", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument(
        "--nll-weight",
        type=float,
        default=0.0,
        help="Add this times the chosen side's per-token NLL (RPO), so chosen stays likely.",
    )
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch", type=int, default=16, help="Effective batch size (pairs).")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--eval-fraction", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=-1, help="Stop early (smoke tests).")
    p.add_argument("--logging-steps", type=int, default=2)
    args = p.parse_args(argv)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    rows = [json.loads(line) for line in args.data.open(encoding="utf-8") if line.strip()]
    pairs = [x for x in (tokenize_pair(tokenizer, r, args.max_length) for r in rows) if x]
    print(f"{len(pairs)}/{len(rows)} pairs within {args.max_length} tokens", flush=True)
    random.Random(0).shuffle(pairs)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16, attn_implementation="sdpa"
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    # No dropout: the policy's log-probabilities must match the reference's at the start.
    config = LoraConfig.from_pretrained(args.sft_adapter)
    config.lora_dropout, config.inference_mode = 0.0, False
    model = PeftModel.from_pretrained(model, args.sft_adapter, config=config, is_trainable=True)
    model.print_trainable_parameters()
    logprob = ChunkedLoss(model)

    def forward(m, pair: dict):
        batch = pad_pair(pair, pad_id)
        batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        labels = batch.pop("labels")
        with torch.autocast("cuda", dtype=torch.float16, enabled=torch.cuda.is_available()):
            m(**batch)
        return logprob.logprob_sums(labels)

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        model.to(f"cuda:{torch.cuda.current_device()}")
    # Reference log-probabilities: the SFT model, before any update. Every rank computes all of
    # them (the trainer shards the pairs itself), which costs one forward per pair.
    start = time.time()
    model.eval()
    with torch.no_grad():
        for pair in pairs:
            pair["ref"] = forward(model, pair).tolist()
    print(f"reference log-probs for {len(pairs)} pairs in {time.time() - start:.0f}s", flush=True)
    n_eval = max(1, int(len(pairs) * args.eval_fraction))
    eval_set, train_set = pairs[:n_eval], pairs[n_eval:]

    model.train()  # gradient checkpointing is only active in training mode
    if torch.cuda.is_available():  # fail fast if the longest pair does not fit
        longest = max(
            pairs, key=lambda x: max(len(x[s]["input_ids"]) for s in ("chosen", "rejected"))
        )
        forward(model, longest).sum().backward()
        model.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_allocated() / 2**30
        adam = 8 * sum(p.numel() for p in model.parameters() if p.requires_grad) / 2**30
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        n = max(len(longest[s]["input_ids"]) for s in ("chosen", "rejected"))
        print(
            f"longest pair (2 x {n} tokens): peak {peak:.1f} GiB "
            f"+ {adam:.1f} GiB optimizer state, of {total:.1f} GiB",
            flush=True,
        )

    class DPOTrainer(Trainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.model_accepts_loss_kwargs = False  # the trainer averages over accumulation
            self.metrics = {"train": defaultdict(list), "eval": defaultdict(list)}

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels, ref = inputs["labels"], inputs["ref"]
            chosen_tokens = inputs["chosen_tokens"]
            model(
                **{k: v for k, v in inputs.items() if k not in ("labels", "ref", "chosen_tokens")}
            )
            policy = logprob.logprob_sums(labels)
            margin = dpo_margin(policy, ref, args.beta)
            loss = -F.logsigmoid(margin)
            if args.nll_weight:
                loss = loss + args.nll_weight * (-policy[0] / chosen_tokens)
            m = self.metrics["train" if model.training else "eval"]
            m["rewards/margin"].append(margin.item())
            m["rewards/accuracy"].append(float(margin.item() > 0))
            m["rewards/chosen"].append(args.beta * (policy[0] - ref[0]).item())
            m["rewards/rejected"].append(args.beta * (policy[1] - ref[1]).item())
            return (loss, None) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad(), self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            return loss.detach(), None, None

        def log(self, logs, *a, **kw):
            mode = "eval" if any(k.startswith("eval_") for k in logs) else "train"
            prefix = "eval_" if mode == "eval" else ""
            for key, values in self.metrics[mode].items():
                if values:
                    logs[prefix + key] = sum(values) / len(values)
            self.metrics[mode].clear()
            super().log(logs, *a, **kw)

    world = max(1, torch.cuda.device_count())
    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.1,  # a float below 1 is a fraction of the total steps
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=max(1, args.batch // world),
        fp16=torch.cuda.is_available(),
        max_grad_norm=1.0,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=10,
        save_strategy="no",
        report_to="none",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        prediction_loss_only=True,
    )
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=make_collate(pad_id),
    )
    trainer.train()
    if trainer.is_world_process_zero():
        adapter = args.out / "adapter"
        model.save_pretrained(adapter)
        tokenizer.save_pretrained(adapter)
        (args.out / "log_history.json").write_text(json.dumps(trainer.state.log_history, indent=2))
        print(f"adapter saved to {adapter}", flush=True)


if __name__ == "__main__":
    main()
