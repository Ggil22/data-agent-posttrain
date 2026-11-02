"""Kaggle job: LoRA SFT of the student on the trajectories sampled by kaggle/generate.

The generate kernel's output is mounted as input (kernel_sources). Its rollouts are filtered
into SFT examples here, so the selection can change without sampling again.

`kaggle kernels push -p kaggle/train --accelerator NvidiaTeslaT4`
"""

import os
import subprocess
from pathlib import Path

SMOKE = False  # a few steps only, to check memory and speed before the full run
HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TRAIN_ARGS = "--epochs 2 --lr 1e-4 --rank 32 --batch 16 --max-length 4096"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
SFT_ARGS = "--per-task 1 --per-category unanswerable=200"

REPO = "https://github.com/Ggil22/data-agent-posttrain"
SRC, VENV = Path("/tmp/repo"), Path("/tmp/venv")
OUT = Path("/kaggle/working/sft")


def sh(cmd: str, **kw) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


sh(f"git clone -q {REPO} {SRC}")
sh("pip install -q uv")
sh(f"uv venv -q {VENV} --python 3.12")
sh(f"uv pip install -q --python {VENV}/bin/python -e {SRC}[train]")
os.environ["PATH"] = f"{VENV}/bin:" + os.environ["PATH"]
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

runs = sorted(Path("/kaggle/input").rglob("train-*.jsonl"))
if not runs:  # the pilot's hinted rollouts, enough for a smoke test
    runs = sorted(Path("/kaggle/input").rglob("pilot-hint.jsonl"))
print("rollouts:", *runs, sep="\n  ", flush=True)
OUT.mkdir(parents=True, exist_ok=True)
data = OUT / "sft-train.jsonl"
sh(f"dapt sft-data {' '.join(map(str, runs))} {SFT_ARGS} --out {data}", cwd=SRC)

steps = "--max-steps 10" if SMOKE else ""
sh(
    f"torchrun --nproc_per_node 2 -m dapt.train --data {data} --model {HF_MODEL} "
    f"--out {OUT} {TRAIN_ARGS} {steps}",
    cwd=SRC,
)
