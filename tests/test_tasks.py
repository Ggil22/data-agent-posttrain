import re
from collections import Counter

import pandas as pd
import pytest
from data_agent_bench.bench import datasets as dab_datasets
from data_agent_bench.bench.scoring import score
from data_agent_bench.bench.tasks import ANSWER_TYPES, TASKS

from dapt.domains import DOMAINS
from dapt.tasks import SPLITS, materialize, tasks_for

SEEDS = range(8)
STOPWORDS = set(
    "a an the of in on at to for by from and or is are was were what which how many much did "
    "does do with as be it its this that each per all any answer round decimals decimal name "
    "english than more most has have".split()
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("tasks")
    return {
        name: {seed: tasks_for(dom, seed, materialize(dom, seed, out)) for seed in SEEDS}
        for name, dom in DOMAINS.items()
    }, out


def _all(built):
    return [t for by_seed in built[0].values() for ts in by_seed.values() for t in ts]


def _format(task):
    """The answer written the way the agent is told to write it."""
    if task.answer_type == "unanswerable":
        return "UNANSWERABLE"
    if task.answer_type == "bool":
        return "yes" if task.truth else "no"
    if task.answer_type == "list_ordered":
        return ", ".join(str(v) for v in task.truth)
    if task.answer_type == "number" and task.decimals is not None:
        return f"{task.truth:.{task.decimals}f}"
    return str(task.truth)


def test_materialize_is_deterministic(tmp_path):
    dom = DOMAINS["clinic"]
    a, b = materialize(dom, 3, tmp_path / "a"), materialize(dom, 3, tmp_path / "b")
    for f in a.iterdir():
        assert f.read_bytes() == (b / f.name).read_bytes(), f.name


def test_every_template_usually_has_an_answer(built):
    counts = Counter((t.domain, t.template) for t in _all(built))
    for name, dom in DOMAINS.items():
        for t in dom.templates:
            # Templates skip on ties; that must stay rare.
            assert counts[(name, t.name)] >= 0.75 * len(SEEDS), (name, t.name)


def test_tasks_are_well_formed_and_score_as_correct(built):
    tasks = _all(built)
    assert len({t.id for t in tasks}) == len(tasks)
    for t in tasks:
        assert t.answer_type in ANSWER_TYPES, t.id
        assert (t.truth is None) == (t.answer_type == "unanswerable"), t.id
        assert score(t.to_dab(), t.truth, _format(t)).correct, (t.id, t.truth)


def test_unanswerable_share(built):
    share = sum(t.answer_type == "unanswerable" for t in _all(built)) / len(_all(built))
    assert 0.1 <= share <= 0.3


def test_bool_answers_are_not_constant(built):
    by_template = {}
    for t in _all(built):
        if t.answer_type == "bool":
            by_template.setdefault((t.domain, t.template), set()).add(t.truth)
    assert by_template
    for key, values in by_template.items():
        assert values == {True, False}, key


def test_no_overlap_with_the_benchmark(built):
    # Different tables...
    for dom in DOMAINS.values():
        assert not set(dom.tables) & set(dab_datasets.TABLES), dom.name

    # ...and no question close to a benchmark question. Generic analysis phrasing ("in which
    # month were the most...") is shared on purpose; near-copies are not. Measured on content
    # words, the closest pair scores 0.40.
    def words(q):
        return set(re.findall(r"[a-z]+", q.lower())) - STOPWORDS

    bench = [words(t.question) for t in TASKS]
    for t in _all(built):
        w = words(t.question)
        for b in bench:
            assert len(w & b) / len(w | b) < 0.5, (t.question, b)


def test_splits_use_disjoint_seeds_and_domains():
    (train_domains, train_seed), (val_domains, val_seed), (held_domains, held_seed) = (
        SPLITS["train"],
        SPLITS["val"],
        SPLITS["heldout"],
    )
    assert train_domains == val_domains and train_seed < val_seed < held_seed
    assert not set(held_domains) & set(train_domains)


def _load(built, domain, seed=0):
    return dab_datasets.load(built[1] / f"{domain}-{seed}")


def _truth(built, domain, template, seed=0):
    return next(t for t in built[0][domain][seed] if t.template == template)


def test_traps_change_the_answer(built):
    """A naive analysis that ignores the data dictionary gets each of these wrong."""
    trips = _load(built, "rentals")["trips"]
    task = _truth(built, "rentals", "unknown_durations")
    assert (trips["duration_min"] == -1).sum() != task.truth  # duplicates and staff trips

    clinic = _load(built, "clinic")
    task = _truth(built, "clinic", "mean_fee")
    spec = next(s for s in ("Cardiology", "Dermatology", "Pediatrics", "Neurology",
                            "General Practice") if s in task.question)  # fmt: skip
    a = clinic["appointments"].merge(clinic["doctors"], on="doctor_id")
    naive = a.loc[(a["specialty"] == spec) & (a["status"] == "attended"), "fee"].mean()
    assert abs(naive - task.truth) > 10  # fees in cents from the app

    energy = _load(built, "energy")
    task = _truth(built, "energy", "total_consumption")
    month = 1 + ["January", "February", "March"].index(
        re.search(r"in (\w+) 2025", task.question)[1]
    )
    r = energy["readings"]
    naive = r.loc[pd.to_datetime(r["date"]).dt.month == month, "consumption"].sum()
    assert abs(naive - task.truth) > 1  # Wh meters and error codes

    inv = _load(built, "inventory")
    task = _truth(built, "inventory", "distinct_skus_shipped")
    assert inv["movements"]["sku"].nunique() != task.truth  # case variants, TEST, dates
