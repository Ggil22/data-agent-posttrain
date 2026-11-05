"""Kaggle job: on-policy samples of the SFT model on the train split, for DPO pairs (phase 4).

The SFT adapter (the dapt-train output, mounted as kernel_sources) is served on both T4s and
sampled several times per task, as it will be used: standard prompt, no hint, no guide, no
format repair. Tasks behind an SFT example are skipped, so sampling goes where the model was
not taught the answer. Each GPU takes half the tasks, task by task, so if the time budget runs
out every task reached has all its samples. `kaggle/dpo-train` turns the samples into pairs.

`kaggle kernels push -p kaggle/dpo-generate --accelerator NvidiaTeslaT4`
"""

import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TASKS, SAMPLES, TEMPERATURE = 640, 4, 0.8
HOURS = 8.0  # sampling budget, well inside Kaggle's 12-hour session limit
GPUS = 2

REPO = "https://github.com/Ggil22/data-agent-posttrain"
SRC, VENV = Path("/tmp/repo"), Path("/tmp/venv")
OUT = Path("/kaggle/working/dpo-generate")


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


(adapter_config,) = Path("/kaggle/input").rglob("adapter/adapter_config.json")
ADAPTER = adapter_config.parent
(SFT_DATA,) = Path("/kaggle/input").rglob("sft-train.jsonl")
print("adapter:", ADAPTER, "\nSFT examples:", SFT_DATA, flush=True)

sh(f"git clone -q {REPO} {SRC}")
sh("pip install -q uv")
sh(f"uv venv -q {VENV} --python 3.12")
sh(f"uv pip install -q --python {VENV}/bin/python vllm -e {SRC}")
os.environ["PATH"] = f"{VENV}/bin:" + os.environ["PATH"]
sh("dapt build --val 0 --heldout 0", cwd=SRC)
OUT.mkdir(parents=True, exist_ok=True)

servers = []
for gpu in range(GPUS):
    cmd = [
        "vllm", "serve", HF_MODEL,
        "--served-model-name", "base",
        "--port", str(8000 + gpu),
        "--dtype", "half",  # T4: no bf16
        "--max-model-len", "16384",
        "--gpu-memory-utilization", "0.9",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "hermes",
        "--enable-lora", "--max-lora-rank", "64",
        "--lora-modules", f"student={ADAPTER}",
    ]  # fmt: skip
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    log = (OUT / f"vllm-{gpu}.log").open("w")
    servers.append(subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env))

config = {
    "providers": {
        f"gpu{i}": {"base_url": f"http://localhost:{8000 + i}/v1", "api_key_env": None}
        for i in range(GPUS)
    },
    "models": {f"student{i}": {"provider": f"gpu{i}", "model": "student"} for i in range(GPUS)},
}
(SRC / "configs" / "dpo-generate.yaml").write_text(json.dumps(config), encoding="utf-8")

for gpu, proc in enumerate(servers):
    if not wait_ready(8000 + gpu, proc):
        sh(f"tail -n 40 {OUT}/vllm-{gpu}.log")
        raise SystemExit(f"vLLM on GPU {gpu} did not start")

jobs = []
for gpu in range(GPUS):
    cmd = (
        f"dapt rollout --model student{gpu} --config configs/dpo-generate.yaml --split train "
        f"--exclude-tasks {SFT_DATA} --subset {TASKS} --shard {gpu}/{GPUS} "
        f"--samples {SAMPLES} --temperature {TEMPERATURE} --by-task --hours {HOURS} "
        f"--workers 16 --out {OUT}/dpo-{gpu}.jsonl"
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
# A first look at the pairs (on-policy only); kaggle/dpo-train builds the real set.
sh(f"dapt dpo-data {OUT}/dpo-*.jsonl --out {OUT}/dpo-pairs-preview.jsonl", cwd=SRC)
