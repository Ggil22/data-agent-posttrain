"""Kaggle job: DPO of the SFT adapter on pairs from its own samples (phase 4).

Mounted inputs (kernel_sources): dapt-dpo-generate (the on-policy samples, `dpo-*.jsonl`),
dapt-train (the SFT adapter) and dapt-generate (the base model's guided samples,
`train-*.jsonl`, whose correct episodes are the chosen side for tasks the SFT model never
solves). Pairs are built here, so the selection can change without sampling again.

`kaggle kernels push -p kaggle/dpo-train --accelerator NvidiaTeslaT4`
"""

import os
import subprocess
from pathlib import Path

SMOKE = False  # a few steps only, to check memory and speed before the full run
HF_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
TRAIN_ARGS = "--beta 0.1 --nll-weight 0.2 --epochs 2 --lr 2e-5 --batch 16 --max-length 4096"
DPO_ARGS = "--per-task 3"  # 4 samples per task: 3 caps what pairs there are
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO = "https://github.com/Ggil22/data-agent-posttrain"
SRC, VENV = Path("/tmp/repo"), Path("/tmp/venv")
OUT = Path("/kaggle/working/dpo")


def sh(cmd: str, **kw) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


inputs = Path("/kaggle/input")
(adapter_config,) = inputs.rglob("adapter/adapter_config.json")
on_policy = sorted(inputs.rglob("dpo-[0-9]*.jsonl"))
off_policy = sorted(inputs.rglob("train-[0-9]*.jsonl"))
print("SFT adapter:", adapter_config.parent, flush=True)
print("on-policy:", *on_policy, "off-policy:", *off_policy, sep="\n  ", flush=True)

sh(f"git clone -q {REPO} {SRC}")
sh("pip install -q uv")
sh(f"uv venv -q {VENV} --python 3.12")
sh(f"uv pip install -q --python {VENV}/bin/python -e {SRC}[train]")
os.environ["PATH"] = f"{VENV}/bin:" + os.environ["PATH"]
sh("nvidia-smi --query-gpu=name,memory.total --format=csv")

OUT.mkdir(parents=True, exist_ok=True)
data = OUT / "dpo-train.jsonl"
extra = f"--chosen-from {' '.join(map(str, off_policy))}" if off_policy else ""
sh(f"dapt dpo-data {' '.join(map(str, on_policy))} {extra} {DPO_ARGS} --out {data}", cwd=SRC)

steps = "--max-steps 4" if SMOKE else ""
sh(
    f"torchrun --nproc_per_node 2 -m dapt.dpo_train --data {data} --model {HF_MODEL} "
    f"--sft-adapter {adapter_config.parent} --out {OUT} {TRAIN_ARGS} {steps}",
    cwd=SRC,
)
