"""Build task sets from the domains: data folders on disk plus one JSONL row per task.

Splits:
- train: training domains, seeds 0..n-1.
- val: the same domains on unseen data (seeds from 1000): in-distribution held-out tasks.
- heldout: a domain never trained on (seeds from 2000): cross-domain generalisation.
data-agent-bench itself is the final, out-of-distribution test set.
"""

from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from data_agent_bench.bench import datasets as dab_datasets
from data_agent_bench.bench.tasks import Task

from dapt.domains import DOMAINS
from dapt.domains.base import Domain, Skip

TRAIN_DOMAINS = ("rentals", "clinic", "energy")
HELDOUT_DOMAINS = ("inventory",)
SPLITS = {
    "train": (TRAIN_DOMAINS, 0),
    "val": (TRAIN_DOMAINS, 1000),
    "heldout": (HELDOUT_DOMAINS, 2000),
}


@dataclass(frozen=True)
class TrainTask:
    id: str
    domain: str
    seed: int
    template: str
    category: str
    difficulty: str
    question: str
    answer_type: str
    truth: Any
    decimals: int | None = None

    @property
    def data_dir(self) -> str:
        return f"{self.domain}-{self.seed}"

    def to_dab(self) -> Task:
        """As a data-agent-bench task, to reuse its scoring and failure taxonomy."""
        truth = self.truth
        return Task(
            self.id,
            self.category,
            self.difficulty,
            self.question,
            self.answer_type,
            lambda _: truth,
            self.decimals,
        )


def _key(*parts: str | int) -> list[int]:
    return [p if isinstance(p, int) else zlib.crc32(p.encode()) for p in parts]


def materialize(domain: Domain, seed: int, out_dir: Path) -> Path:
    """Write one dataset instance (CSV tables and data dictionary) to `out_dir/<domain>-<seed>`."""
    path = out_dir / f"{domain.name}-{seed}"
    path.mkdir(parents=True, exist_ok=True)
    data = domain.generate(np.random.default_rng(_key(seed, domain.name)))
    assert set(data) == set(domain.tables), (set(data), domain.tables)
    for name, df in data.items():
        df.to_csv(path / f"{name}.csv", index=False)
    (path / "DATA_DICTIONARY.md").write_text(domain.dictionary, encoding="utf-8")
    return path


def tasks_for(domain: Domain, seed: int, data_dir: Path) -> list[TrainTask]:
    """One task per template, with ground truth computed from the data as the agent sees it."""
    data = dab_datasets.load(data_dir)
    tasks = []
    for t in domain.templates:
        rng = np.random.default_rng(_key(seed, domain.name, t.name))
        try:
            draft = t.fn(data, rng)
        except Skip:
            continue
        tasks.append(
            TrainTask(
                id=f"{domain.name}-{seed}-{t.name}",
                domain=domain.name,
                seed=seed,
                template=t.name,
                category=t.category,
                difficulty=t.difficulty,
                question=draft.question,
                answer_type=draft.answer_type,
                truth=_jsonable(draft.truth),
                decimals=draft.decimals,
            )
        )
    return tasks


def build_split(split: str, n_seeds: int, out_dir: Path) -> list[TrainTask]:
    domains, first_seed = SPLITS[split]
    tasks = []
    for name in domains:
        for seed in range(first_seed, first_seed + n_seeds):
            data_dir = materialize(DOMAINS[name], seed, out_dir / "data")
            tasks += tasks_for(DOMAINS[name], seed, data_dir)
    save_tasks(tasks, out_dir / f"{split}.jsonl")
    return tasks


def save_tasks(tasks: list[TrainTask], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")


def load_tasks(path: Path) -> list[TrainTask]:
    with path.open(encoding="utf-8") as f:
        return [TrainTask(**json.loads(line)) for line in f if line.strip()]


def _jsonable(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value
