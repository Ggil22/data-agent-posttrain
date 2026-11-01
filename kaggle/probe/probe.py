"""Kaggle probe: can vLLM serve each candidate student on a T4, and how well does it do?

Serves one candidate per GPU (2x T4), runs the same validation subset on both and leaves the
scored trajectories and server logs in /kaggle/working.
"""

import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = "https://github.com/Ggil22/data-agent-posttrain"
WORK = Path("/kaggle/working")
SRC = Path("/tmp/repo")
VENV = Path("/tmp/venv")
SUBSET = int(os.environ.get("SUBSET", "60"))

CANDIDATES = [
    # (name, Hugging Face id, extra vLLM arguments)
    ("qwen3-4b-2507", "Qwen/Qwen3-4B-Instruct-2507", ["--tool-call-parser", "hermes"]),
    (
        "qwen3.5-2b",
        "Qwen/Qwen3.5-2B",
        [
            "--tool-call-parser",
            "qwen3_coder",
            "--default-chat-template-kwargs",
            '{"enable_thinking": false}',
            "--limit-mm-per-prompt",
            '{"image": 0, "video": 0}',
        ],
    ),
]


def sh(cmd: str, **kw) -> None:
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, **kw)


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 1200) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://localhost:{port}/v1/models", timeout=5)
            return True
        except Exception:
            time.sleep(10)
    return False


def main() -> None:
    sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv")
    sh(f"git clone -q {REPO} {SRC}")
    sh("pip install -q uv")
    sh(f"uv venv -q {VENV} --python 3.12")
    py = VENV / "bin" / "python"
    sh(f"uv pip install -q --python {py} vllm -e {SRC}")
    sh(f"{py} -c 'import vllm, torch; print(vllm.__version__, torch.__version__)'")
    sh(f"{VENV}/bin/dapt build --val 6 --train 0 --heldout 0", cwd=SRC)

    servers, models = [], {}
    for gpu, (name, hf_id, extra) in enumerate(CANDIDATES):
        port = 8000 + gpu
        log = open(WORK / f"vllm-{name}.log", "w")
        cmd = [
            str(VENV / "bin" / "vllm"),
            "serve",
            hf_id,
            "--served-model-name",
            name,
            "--port",
            str(port),
            "--dtype",
            "half",
            "--max-model-len",
            "16384",
            "--gpu-memory-utilization",
            "0.9",
            "--enable-auto-tool-choice",
            *extra,
        ]
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        servers.append((name, port, proc))
        models[name] = {"provider": f"vllm{port}", "model": name}

    config = {
        "providers": {
            f"vllm{8000 + i}": {"base_url": f"http://localhost:{8000 + i}/v1", "api_key_env": None}
            for i in range(len(CANDIDATES))
        },
        "models": models,
    }
    (SRC / "configs" / "probe.yaml").write_text(json.dumps(config), encoding="utf-8")

    jobs = []
    for name, port, proc in servers:
        if not wait_ready(port, proc):
            print(f"!! {name} did not start; see vllm-{name}.log", flush=True)
            sh(f"tail -n 40 {WORK}/vllm-{name}.log")
            continue
        print(f"{name} ready on port {port}", flush=True)
        out = WORK / f"probe-{name}.jsonl"
        log = open(WORK / f"rollout-{name}.log", "w")
        cmd = [
            str(VENV / "bin" / "dapt"),
            "rollout",
            "--model",
            name,
            "--config",
            "configs/probe.yaml",
            "--split",
            "val",
            "--subset",
            str(SUBSET),
            "--workers",
            "16",
            "--out",
            str(out),
        ]
        jobs.append((name, subprocess.Popen(cmd, cwd=SRC, stdout=log, stderr=subprocess.STDOUT)))

    for name, proc in jobs:
        proc.wait()
        print(f"== {name} (exit {proc.returncode})", flush=True)
        sh(f"tail -n 3 {WORK}/rollout-{name}.log")
    for _, _, proc in servers:
        proc.terminate()
    sys.exit(0 if jobs else 1)


if __name__ == "__main__":
    main()
