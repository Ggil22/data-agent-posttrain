"""Evaluate a model served by vLLM on every evaluation level, strictly and with format repair.

Meant for a Kaggle GPU session: one vLLM server per GPU, one evaluation mode per server, both
running at the same time. The same entry point evaluates the base model and, with `lora`, the
fine-tuned adapters.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

from data_agent_bench.bench import datasets as dab_datasets
from data_agent_bench.bench.report import write_report
from data_agent_bench.bench.runner import latest_records, run_benchmark
from data_agent_bench.bench.runner import load_records as load_bench_records
from data_agent_bench.bench.tasks import TASKS
from data_agent_bench.llm import ModelConfig, OpenAICompatClient

from dapt.repair import RepairingClient
from dapt.rollout import load_records, rollout
from dapt.tasks import load_tasks

MODES = ("strict", "repair")
SPLITS = ("val", "heldout", "bench")


def start_server(
    hf_id: str,
    port: int,
    gpu: int,
    log_path: Path,
    lora: Path | None = None,
    extra: tuple[str, ...] = (),
) -> subprocess.Popen:
    cmd = [
        "vllm", "serve", hf_id,
        "--served-model-name", "base",
        "--port", str(port),
        "--dtype", "half",  # T4: no bf16
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.9",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        *extra,
    ]  # fmt: skip
    if lora:
        cmd += ["--enable-lora", "--max-lora-rank", "64", "--lora-modules", f"student={lora}"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = log_path.open("w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 1800) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5)
            return True
        except OSError:
            time.sleep(10)
    return False


def run_mode(
    mode: str,
    cfg: ModelConfig,
    out_dir: Path,
    tasks_dir: Path,
    splits: tuple[str, ...],
    subset: int | None,
    workers: int,
) -> None:
    repair = mode == "repair"
    out_dir.mkdir(parents=True, exist_ok=True)

    def make_client(sample: int):
        return OpenAICompatClient(cfg, cache_salt=f"sample={sample}")

    for split in splits:
        print(f"[{mode}] {split}", flush=True)
        if split == "bench":
            data_dir = out_dir / "bench-data"
            dab_datasets.write(data_dir)

            def bench_client(model: str, repeat: int):
                client = make_client(repeat)
                return RepairingClient(client) if repair else client

            bench_dir = out_dir / "bench"
            results = run_benchmark(
                [cfg.name], TASKS, bench_client, data_dir, bench_dir, log=lambda _: None
            )
            write_report(latest_records(load_bench_records(results)), bench_dir, cfg.name)
            continue
        tasks = load_tasks(tasks_dir / f"{split}.jsonl")
        if subset:
            tasks = tasks[:: max(1, len(tasks) // subset)][:subset]
        rollout(
            tasks,
            make_client,
            tasks_dir / "data",
            out_dir / f"{split}.jsonl",
            workers=workers,
            repair=repair,
            log=lambda msg, m=mode, s=split: print(f"[{m}] {s} {msg}", flush=True),
        )


def evaluate(
    hf_id: str,
    out_dir: Path,
    tasks_dir: Path,
    lora: Path | None = None,
    modes: tuple[str, ...] = MODES,
    splits: tuple[str, ...] = SPLITS,
    subset: int | None = None,
    workers: int = 16,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    served = "student" if lora else "base"
    servers, threads = [], []
    for gpu, mode in enumerate(modes):
        port = 8000 + gpu
        proc = start_server(hf_id, port, gpu, out_dir / f"vllm-{mode}.log", lora)
        servers.append((mode, port, proc))
    try:
        for mode, port, proc in servers:
            if not wait_ready(port, proc):
                raise RuntimeError(f"vLLM for {mode} did not start; see vllm-{mode}.log")
            cfg = ModelConfig(
                name=served,
                provider="vllm",
                model=served,
                base_url=f"http://localhost:{port}/v1",
                api_key_env=None,
            )
            t = threading.Thread(
                target=run_mode,
                args=(mode, cfg, out_dir / mode, tasks_dir, splits, subset, workers),
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
    finally:
        for _, _, proc in servers:
            proc.terminate()
    summary = summarize(out_dir, modes, splits)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def summarize(out_dir: Path, modes=MODES, splits=SPLITS) -> dict:
    summary = {}
    for mode in modes:
        for split in splits:
            if split == "bench":
                path = out_dir / mode / "bench" / "results.jsonl"
                records = latest_records(load_bench_records(path)) if path.exists() else []
            else:
                records = load_records(out_dir / mode / f"{split}.jsonl")
            if records:
                n = sum(r["correct"] for r in records)
                summary[f"{mode}/{split}"] = {"correct": n, "total": len(records)}
    return summary
