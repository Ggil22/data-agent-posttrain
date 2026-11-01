"""Command line interface: `dapt <command>`."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from collections import Counter
from pathlib import Path

from data_agent_bench.llm import OpenAICompatClient, load_model_configs

from dapt.repair import RepairingClient
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
        repair=args.repair,
    )
    records = load_records(out)
    outcomes = Counter(r["outcome"] for r in records)
    print(f"{sum(r['correct'] for r in records)}/{len(records)} correct  {dict(outcomes)}")


def cmd_bench(args: argparse.Namespace) -> None:
    """data-agent-bench's 65 tasks: the out-of-distribution test set."""
    from data_agent_bench.bench import datasets
    from data_agent_bench.bench.report import write_report
    from data_agent_bench.bench.runner import latest_records, run_benchmark
    from data_agent_bench.bench.runner import load_records as load_bench_records
    from data_agent_bench.bench.tasks import TASKS

    configs = load_model_configs(args.config)
    data_dir = Path("data/bench")
    datasets.write(data_dir)

    def make_client(model: str, repeat: int):
        client = OpenAICompatClient(
            configs[model], cache_dir=Path(".cache/llm"), cache_salt=f"repeat={repeat}"
        )
        return RepairingClient(client) if args.repair else client

    results = run_benchmark(
        [args.model], TASKS, make_client, data_dir, args.out, repeats=args.repeats
    )
    records = latest_records(load_bench_records(results))
    print(f"Report: {write_report(records, args.out, f'{args.model} on data-agent-bench')}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    from dapt.evaluate import evaluate

    summary = evaluate(
        args.hf_model,
        args.out,
        args.tasks,
        lora=args.lora,
        modes=tuple(args.modes),
        splits=tuple(args.splits),
        subset=args.subset,
        workers=args.workers,
    )
    for key, v in summary.items():
        print(f"{key:<16} {v['correct']:>4}/{v['total']:<4} {v['correct'] / v['total']:.1%}")


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
    p.add_argument("--repair", action="store_true", help="Repair tool-call format errors.")
    p.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=cmd_rollout)

    p = sub.add_parser("bench", help="Run a model on data-agent-bench.")
    p.add_argument("--model", required=True)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--repair", action="store_true", help="Repair tool-call format errors.")
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("evaluate", help="Serve a model with vLLM and run every evaluation.")
    p.add_argument("--hf-model", required=True, help="Hugging Face id of the base model.")
    p.add_argument("--lora", type=Path, help="LoRA adapter to evaluate on top of the base.")
    p.add_argument("--modes", nargs="+", default=["strict", "repair"])
    p.add_argument("--splits", nargs="+", default=["val", "heldout", "bench"])
    p.add_argument("--subset", type=int, help="Only N tasks per split (not for bench).")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    from dotenv import load_dotenv

    load_dotenv(Path(".env"))
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
