"""Kaggle job: sample SFT trajectories on the train split (rejection sampling, phase 3).

The base model is served on both T4s and sampled with the hinted prompt and format repair; each
GPU takes half the tasks. Episodes are ordered sample by sample, so if the time budget runs out
every task still has its first sample. `dapt sft-data` later keeps the correct ones.

`kaggle kernels push -p kaggle/generate --accelerator NvidiaTeslaT4`
"""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SAMPLES, TEMPERATURE = 2, 0.7
HOURS = 8.0  # sampling budget, well inside Kaggle's 12-hour session limit
GPUS = 2
# Pilot: the same 120 train tasks, one sample each, guided with the hint (GPU 0) and without it
# (GPU 1). Pilot 1 (hint vs nothing) showed the hint alone barely changes behaviour.
PILOT = True

REPO = "https://github.com/Ggil22/data-agent-posttrain"
SRC, VENV = Path("/tmp/repo"), Path("/tmp/venv")
OUT = Path("/kaggle/working/generate")


def sh(cmd: str, **kw) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


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


sh(f"git clone -q {REPO} {SRC}")
sh("pip install -q uv")
sh(f"uv venv -q {VENV} --python 3.12")
sh(f"uv pip install -q --python {VENV}/bin/python vllm -e {SRC}")
os.environ["PATH"] = f"{VENV}/bin:" + os.environ["PATH"]
sh("dapt build --val 0 --heldout 0", cwd=SRC)
OUT.mkdir(parents=True, exist_ok=True)

servers = []
for gpu in range(GPUS):
    port = 8000 + gpu
    cmd = [
        "vllm", "serve", HF_MODEL,
        "--served-model-name", "base",
        "--port", str(port),
        "--dtype", "half",  # T4: no bf16
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.9",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
    ]  # fmt: skip
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = (OUT / f"vllm-{gpu}.log").open("w")
    servers.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))

config = {
    "providers": {
        f"gpu{i}": {"base_url": f"http://localhost:{8000 + i}/v1", "api_key_env": None}
        for i in range(GPUS)
    },
    "models": {f"base{i}": {"provider": f"gpu{i}", "model": "base"} for i in range(GPUS)},
}
(SRC / "configs" / "generate.yaml").write_text(json.dumps(config), encoding="utf-8")

for gpu, proc in enumerate(servers):
    if not wait_ready(8000 + gpu, proc):
        sh(f"tail -n 40 {OUT}/vllm-{gpu}.log")
        raise SystemExit(f"vLLM on GPU {gpu} did not start")

jobs = []
for gpu in range(GPUS):
    if PILOT:
        hint = "--hint" if gpu == 0 else ""
        cmd = (
            f"dapt rollout --model base{gpu} --config configs/generate.yaml --split train "
            f"--subset 120 --temperature {TEMPERATURE} {hint} --guide --repair --workers 16 "
            f"--out {OUT}/pilot-guide-{'hint' if hint else 'nohint'}.jsonl"
        )
    else:
        cmd = (
            f"dapt rollout --model base{gpu} --config configs/generate.yaml --split train "
            f"--samples {SAMPLES} --temperature {TEMPERATURE} --hint --guide --repair "
            f"--shard {gpu}/{GPUS} --hours {HOURS} --workers 16 --out {OUT}/train-{gpu}.jsonl"
        )
    log = (OUT / f"rollout-{gpu}.log").open("w")
    print(f"$ {cmd}", flush=True)
    jobs.append(subprocess.Popen(cmd, shell=True, cwd=SRC, stdout=log, stderr=subprocess.STDOUT))

for gpu, job in enumerate(jobs):
    job.wait()
    print(f"== GPU {gpu}: exit {job.returncode}", flush=True)
    sh(f"tail -n 4 {OUT}/rollout-{gpu}.log")
for proc in servers:
    proc.terminate()
if not PILOT:
    sh(f"dapt sft-data {OUT}/train-*.jsonl --out {OUT}/sft-train.jsonl", cwd=SRC)
