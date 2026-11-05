"""Kaggle job: evaluate a model on val, heldout and data-agent-bench, strictly and with repair.

Edit RUN below, then `kaggle kernels push -p kaggle/evaluate --accelerator NvidiaTeslaT4`.
"""

import os
import subprocess
from pathlib import Path

RUN = {
    "name": "dpo",
    "hf_model": "Qwen/Qwen3-4B-Instruct-2507",
    # A LoRA adapter to serve on top of the base: "input" finds the one in the mounted
    # kernel_sources (the dapt-train or dapt-dpo-train output); None evaluates the base model.
    "lora": "input",
}
if RUN["lora"] == "input":
    (config,) = Path("/kaggle/input").rglob("adapter/adapter_config.json")
    RUN["lora"] = str(config.parent)
    print("adapter:", RUN["lora"], flush=True)

REPO = "https://github.com/Ggil22/data-agent-posttrain"
SRC, VENV = "/tmp/repo", "/tmp/venv"


def sh(cmd: str, **kw) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


sh(f"git clone -q {REPO} {SRC}")
sh("pip install -q uv")
sh(f"uv venv -q {VENV} --python 3.12")
sh(f"uv pip install -q --python {VENV}/bin/python vllm -e {SRC}")
os.environ["PATH"] = f"{VENV}/bin:" + os.environ["PATH"]
sh("dapt build --train 0", cwd=SRC)
lora = f" --lora {RUN['lora']}" if RUN["lora"] else ""
sh(
    f"dapt evaluate --hf-model {RUN['hf_model']}{lora} --out /kaggle/working/{RUN['name']}",
    cwd=SRC,
)
