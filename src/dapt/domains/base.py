"""Building blocks for training-task domains.

A domain is a synthetic dataset generator (several related CSV tables plus a data dictionary
documenting their traps) and a set of question templates. Each template draws its parameters
from a random generator and computes its ground truth from the data as the agent will see it
(round-tripped through CSV), applying the rules the data dictionary documents.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

Data = dict[str, pd.DataFrame]

MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


class Skip(Exception):
    """The template has no well-defined answer for this data (e.g. a tie for the maximum)."""


@dataclass(frozen=True)
class Draft:
    question: str
    answer_type: str  # number | string | bool | list_ordered | unanswerable
    truth: Any
    decimals: int | None = None  # for numbers: None means an exact integer is expected


@dataclass(frozen=True)
class Template:
    name: str
    category: str
    difficulty: str
    fn: Callable[[Data, np.random.Generator], Draft]


def template(category: str, difficulty: str):
    def wrap(fn: Callable[[Data, np.random.Generator], Draft]) -> Template:
        return Template(fn.__name__, category, difficulty, fn)

    return wrap


@dataclass(frozen=True)
class Domain:
    name: str
    tables: tuple[str, ...]
    dictionary: str
    generate: Callable[[np.random.Generator], Data]
    templates: tuple[Template, ...]


def pick(rng: np.random.Generator, options):
    """A random element as a plain Python object (not a numpy scalar)."""
    options = list(options)
    return options[int(rng.integers(len(options)))]


def argmax(series: pd.Series):
    """Index of the maximum; skips the task when the top two values tie."""
    ordered = series.sort_values(ascending=False)
    if len(ordered) == 0:
        raise Skip("empty")
    if len(ordered) > 1 and np.isclose(ordered.iloc[0], ordered.iloc[1]):
        raise Skip(f"tie between {ordered.index[0]!r} and {ordered.index[1]!r}")
    return ordered.index[0]


def top_k(series: pd.Series, k: int) -> list:
    """The k largest, in order; skips the task on a tie inside or at the edge of the top k."""
    ordered = series.sort_values(ascending=False)
    if len(ordered) < k:
        raise Skip("fewer than k groups")
    head = ordered.iloc[: k + 1].to_numpy(dtype=float)
    if np.any(np.isclose(head[:-1], head[1:])):
        raise Skip("tie in the top k")
    return [_plain(i) for i in ordered.index[:k]]


def messy_labels(rng: np.random.Generator, values: np.ndarray, p: float) -> np.ndarray:
    """Hand-typed variants of labels (case and stray spaces) for a fraction `p` of rows."""
    out = values.astype(object).copy()
    for i in np.flatnonzero(rng.random(len(out)) < p):
        v = str(out[i])
        v = pick(rng, (v.upper(), v.capitalize(), v.lower()))
        out[i] = pick(rng, (f" {v}", f"{v} ", v, f"  {v}"))
    return out


def ratio_pct(part: float, whole: float) -> float:
    if whole == 0:
        raise Skip("division by zero")
    return 100 * part / whole


def _plain(value):
    return value.item() if isinstance(value, np.generic) else value
