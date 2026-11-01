"""Run the data-agent-bench agent on training tasks and record scored trajectories.

Episodes run concurrently, since a local inference server (vLLM) is only efficient when it
batches many requests. Runs are resumable: finished (task, sample) pairs are skipped, except
API errors, which are retried.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

from data_agent_bench.agent import run_agent
from data_agent_bench.bench.failures import classify
from data_agent_bench.bench.scoring import score
from data_agent_bench.llm import LLMClient
from data_agent_bench.sandbox import SubprocessSandbox
from data_agent_bench.tools import ToolExecutor

from dapt.repair import RepairingClient
from dapt.tasks import TrainTask

ClientFactory = Callable[[int], LLMClient]  # sample index -> client


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    latest = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                latest[(r["task_id"], r["sample"])] = r  # retried episodes replace old ones
    return list(latest.values())


def run_episode(
    task: TrainTask,
    sample: int,
    client: LLMClient,
    data_root: Path,
    max_steps: int,
    repair: bool = False,
) -> dict:
    if repair:
        client = RepairingClient(client)
    with tempfile.TemporaryDirectory(prefix="dapt_", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        for f in (data_root / task.data_dir).iterdir():
            shutil.copy2(f, workdir / f.name)
        executor = ToolExecutor(workdir, sandbox=SubprocessSandbox())
        traj = run_agent(client, task.question, executor, max_steps=max_steps)
    sc = score(task.to_dab(), task.truth, traj.final_answer)
    outcome, flags = classify(traj, sc, task.answer_type == "unanswerable")
    return {
        "task_id": task.id,
        "sample": sample,
        "domain": task.domain,
        "template": task.template,
        "category": task.category,
        "question": task.question,
        "truth": task.truth,
        "prediction": traj.final_answer,
        "correct": sc.correct,
        "outcome": outcome,
        "flags": flags,
        "stop_reason": traj.stop_reason,
        "error": traj.error,
        "n_steps": len(traj.steps),
        "prompt_tokens": traj.prompt_tokens,
        "completion_tokens": traj.completion_tokens,
        "repairs": client.repairs if repair else None,
        "trajectory": [asdict(s) for s in traj.steps],
    }


def rollout(
    tasks: list[TrainTask],
    make_client: ClientFactory,
    data_root: Path,
    out_path: Path,
    samples: int = 1,
    max_steps: int = 12,
    workers: int = 16,
    repair: bool = False,
    log: Callable[[str], None] = print,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {
        (r["task_id"], r["sample"]) for r in load_records(out_path) if r["outcome"] != "api_error"
    }
    jobs = [(t, s) for s in range(samples) for t in tasks if (t.id, s) not in done]
    clients = {s: make_client(s) for s in range(samples)}
    n_correct = 0

    with (
        ThreadPoolExecutor(max_workers=workers) as pool,
        out_path.open("a", encoding="utf-8") as f,
    ):
        futures = [
            pool.submit(run_episode, t, s, clients[s], data_root, max_steps, repair)
            for t, s in jobs
        ]
        # Records are written from this thread only, as each episode finishes.
        for i, fut in enumerate(as_completed(futures), 1):
            record = fut.result()
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            n_correct += record["correct"]
            if i % 25 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)} episodes, {n_correct / i:.0%} correct")
    return out_path
