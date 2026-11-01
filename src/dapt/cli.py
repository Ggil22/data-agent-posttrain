"""Command line interface: `dapt <command>`."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter
from pathlib import Path

from data_agent_bench.llm import OpenAICompatClient, load_model_configs

from dapt.rollout import load_records, rollout
from dapt.tasks import SPLITS, build_split, load_tasks

DEFAULT_TASKS = Path("data/tasks")
DEFAULT_CONFIG = Path("configs/models.yaml")


def cmd_build(args: argparse.Namespace) -> None:
    for split, n in (("train", args.train), ("val", args.val), ("heldout", args.heldout)):
        tasks = build_split(split, n, args.out)
        types = Counter(t.answer_type for t in tasks)
        print(f"{split:<8} {len(tasks):>5} tasks  {dict(types)}")


def cmd_rollout(args: argparse.Namespace) -> None:
    configs = load_model_configs(args.config)
    if args.model not in configs:
        sys.exit(f"Unknown model {args.model!r}; configured: {', '.join(configs)}")
    cfg = configs[args.model]
    if args.temperature is not None:
        cfg = dataclasses.replace(cfg, temperature=args.temperature)
    tasks = load_tasks(args.tasks / f"{args.split}.jsonl")
    if args.subset:  # evenly spread over domains, seeds and templates
        tasks = tasks[:: max(1, len(tasks) // args.subset)][: args.subset]
    out = args.out or Path("runs") / f"{args.model}-{args.split}.jsonl"
    print(f"{len(tasks)} tasks x {args.samples} samples -> {out}")

    def make_client(sample: int) -> OpenAICompatClient:
        return OpenAICompatClient(cfg, cache_dir=Path(".cache/llm"), cache_salt=f"sample={sample}")

    rollout(
        tasks,
        make_client,
        args.tasks / "data",
        out,
        samples=args.samples,
        max_steps=args.max_steps,
        workers=args.workers,
    )
    records = load_records(out)
    outcomes = Counter(r["outcome"] for r in records)
    print(f"{sum(r['correct'] for r in records)}/{len(records)} correct  {dict(outcomes)}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dapt", description="Post-training data agents.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="Generate the train, val and heldout task sets.")
    p.add_argument("--train", type=int, default=60, help="Seeds per training domain.")
    p.add_argument("--val", type=int, default=6, help="Seeds per training domain.")
    p.add_argument("--heldout", type=int, default=12, help="Seeds of the held-out domain.")
    p.add_argument("--out", type=Path, default=DEFAULT_TASKS)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("rollout", help="Run a model on a task set and score each episode.")
    p.add_argument("--model", required=True)
    p.add_argument("--split", choices=list(SPLITS), default="val")
    p.add_argument("--samples", type=int, default=1, help="Episodes per task.")
    p.add_argument("--temperature", type=float, help="Override the configured temperature.")
    p.add_argument("--subset", type=int, help="Only N tasks, evenly spread over the split.")
    p.add_argument("--max-steps", type=int, default=12)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_rollout)
    return parser


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(".env"))
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
