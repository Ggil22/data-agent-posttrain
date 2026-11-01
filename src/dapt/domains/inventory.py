"""Warehouse stock movements, January to June 2025."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dapt.domains.base import (
    MONTHS,
    WEEKDAYS,
    Data,
    Domain,
    Draft,
    Skip,
    argmax,
    messy_labels,
    pick,
    template,
    top_k,
)

CATEGORIES = {
    "Electronics": 180.0,
    "Furniture": 220.0,
    "Kitchen": 35.0,
    "Garden": 45.0,
    "Toys": 20.0,
}
SUPPLIERS = ("Arcadia Supply", "Brightline", "Cobalt Trading", "Delta Wholesale", "Evergreen Co")
WAREHOUSES = ("W1", "W2", "W3", "W4")

DICTIONARY = """\
# Data dictionary

## products.csv
- sku: product identifier (e.g. SKU-1001).
- category, supplier.
- unit_cost_eur: purchase cost per unit, in EUR.

## movements.csv
Stock movements in the company's warehouses, January to June 2025.
- movement_id: movement identifier.
- date: day of the movement. Two systems wrote this field: YYYY-MM-DD and YYYYMMDD.
- sku: references products.csv. Typed in by hand at some docks, so capitalisation and
  spacing vary (sku-1001 and SKU-1001 are the same product).
- warehouse: W1 to W4. Movements in warehouse TEST are tests of the scanning system, not
  real stock: exclude them from every analysis.
- type: in (goods received) or out (goods shipped).
- quantity: units moved. Always positive: an out movement reduces stock by its quantity.

Every warehouse started 2025 with no stock. Value stock and movements at unit cost.
"""


def generate(rng: np.random.Generator) -> Data:
    n_p = 60
    category = rng.choice(list(CATEGORIES), n_p)
    products = pd.DataFrame(
        {
            "sku": [f"SKU-{1001 + i}" for i in range(n_p)],
            "category": category,
            "supplier": rng.choice(SUPPLIERS, n_p),
            "unit_cost_eur": np.round([CATEGORIES[c] * rng.lognormal(0, 0.4) for c in category], 2),
        }
    )

    n = int(rng.integers(3000, 4500))
    days = pd.date_range("2025-01-01", "2025-06-30", freq="D")
    days = days[days.day_of_week < 6]
    day_w = rng.uniform(0.6, 1.4, len(days))
    date = days[rng.choice(len(days), n, p=day_w / day_w.sum())]
    sku_idx = rng.choice(n_p, n, p=rng.dirichlet(np.full(n_p, 1.5)))
    is_out = rng.random(n) < 0.55
    qty = np.where(is_out, rng.integers(1, 30, n), rng.integers(10, 80, n))

    compact = rng.random(n) < 0.35
    warehouse = rng.choice(WAREHOUSES, n, p=[0.35, 0.25, 0.25, 0.15]).astype(object)
    warehouse[rng.random(n) < 0.02] = "TEST"
    movements = pd.DataFrame(
        {
            "movement_id": np.arange(700001, 700001 + n),
            "date": np.where(compact, date.strftime("%Y%m%d"), date.strftime("%Y-%m-%d")),
            "sku": messy_labels(rng, products["sku"].to_numpy()[sku_idx], 0.1),
            "warehouse": warehouse,
            "type": np.where(is_out, "out", "in"),
            "quantity": qty,
        }
    )
    order = np.argsort(date.to_numpy(), kind="stable")
    movements = movements.iloc[order].reset_index(drop=True)
    movements["movement_id"] = np.arange(700001, 700001 + n)
    return {"products": products, "movements": movements}


def _movements(d: Data) -> pd.DataFrame:
    """Real movements with the documented rules applied, joined to the products."""
    m = d["movements"]
    m = m[m["warehouse"] != "TEST"]
    raw = m["date"].astype(str)
    date = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce").fillna(
        pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    )
    m = m.assign(date=date, sku=m["sku"].str.strip().str.upper())
    m = m.merge(d["products"], on="sku")
    signed = np.where(m["type"] == "out", -m["quantity"], m["quantity"])
    return m.assign(signed=signed, value=m["quantity"] * m["unit_cost_eur"])


def _month(m: pd.DataFrame, month: int) -> pd.DataFrame:
    return m[m["date"].dt.month == month]


@template("aggregation", "medium")
def units_shipped(d, rng):
    month = pick(rng, range(1, 7))
    m = _month(_movements(d), month)
    n = int(m.loc[m["type"] == "out", "quantity"].sum())
    q = f"How many units were shipped out of the warehouses in {MONTHS[month - 1]} 2025?"
    return Draft(q, "number", n)


@template("multi_step", "hard")
def stock_level(d, rng):
    m = _movements(d)
    sku, wh = pick(rng, sorted(m["sku"].unique())), pick(rng, WAREHOUSES)
    stock = int(m.loc[(m["sku"] == sku) & (m["warehouse"] == wh), "signed"].sum())
    q = f"How many units of {sku} were in stock in warehouse {wh} at the end of June 2025?"
    return Draft(q, "number", stock)


@template("join", "medium")
def top_category_shipped(d, rng):
    month = pick(rng, range(1, 7))
    m = _month(_movements(d), month)
    out = m[m["type"] == "out"]
    q = (
        f"Which product category had the highest value of goods shipped out in "
        f"{MONTHS[month - 1]} 2025?"
    )
    return Draft(q, "string", argmax(out.groupby("category")["value"].sum()))


@template("groupby", "easy")
def busiest_warehouse(d, rng):
    month = pick(rng, range(1, 7))
    m = _month(_movements(d), month)
    q = f"Which warehouse recorded the most movements (in and out) in {MONTHS[month - 1]} 2025?"
    return Draft(q, "string", argmax(m.groupby("warehouse").size()))


@template("cleaning", "medium")
def distinct_skus_shipped(d, rng):
    month = pick(rng, range(1, 7))
    m = _month(_movements(d), month)
    n = m.loc[m["type"] == "out", "sku"].nunique()
    q = f"How many distinct products were shipped out at least once in {MONTHS[month - 1]} 2025?"
    return Draft(q, "number", int(n))


@template("time_series", "medium")
def busiest_receiving_month(d, rng):
    m = _movements(d)
    inbound = m[m["type"] == "in"]
    month = argmax(inbound.groupby(inbound["date"].dt.month)["quantity"].sum())
    q = "In which month of 2025 were the most units received? Answer with the month name."
    return Draft(q, "string", MONTHS[month - 1])


@template("statistics", "medium")
def mean_shipment(d, rng):
    cat = pick(rng, CATEGORIES)
    m = _movements(d)
    sel = m[(m["type"] == "out") & (m["category"] == cat)]
    if sel.empty:
        raise Skip("no shipments")
    q = (
        f"What is the average quantity per outbound movement for {cat} products? "
        "Round to 2 decimals."
    )
    return Draft(q, "number", float(sel["quantity"].mean()), decimals=2)


@template("groupby", "medium")
def top_suppliers(d, rng):
    m = _movements(d)
    inbound = m[m["type"] == "in"]
    top = top_k(inbound.groupby("supplier")["value"].sum(), 3)
    q = (
        "List the 3 suppliers whose products had the highest value of goods received, "
        "from highest to lowest."
    )
    return Draft(q, "list_ordered", top)


@template("statistics", "medium")
def compare_warehouses(d, rng):
    a, b = (str(w) for w in rng.choice(WAREHOUSES, 2, replace=False))
    month = pick(rng, range(1, 7))
    m = _month(_movements(d), month)
    out = m[m["type"] == "out"].groupby("warehouse")["value"].sum()
    if np.isclose(out.get(a, 0), out.get(b, 0)):
        raise Skip("equal values")
    q = (
        f"In {MONTHS[month - 1]} 2025, was the value of goods shipped out of {a} higher than "
        f"the value shipped out of {b}?"
    )
    return Draft(q, "bool", bool(out.get(a, 0) > out.get(b, 0)))


@template("business_rules", "hard")
def stock_value(d, rng):
    wh = pick(rng, WAREHOUSES)
    m = _movements(d)
    m = m[m["warehouse"] == wh]
    value = (m["signed"] * m["unit_cost_eur"]).sum()
    q = (
        f"What was the value, in EUR, of the stock held in warehouse {wh} at the end of "
        "June 2025? Round to 2 decimals."
    )
    return Draft(q, "number", float(value), decimals=2)


@template("filtering", "easy")
def large_shipments(d, rng):
    x = pick(rng, (15, 20, 25))
    m = _movements(d)
    n = int(((m["type"] == "out") & (m["quantity"] >= x)).sum())
    return Draft(f"How many outbound movements moved {x} units or more?", "number", n)


@template("time_series", "medium")
def busiest_shipping_weekday(d, rng):
    m = _movements(d)
    out = m[m["type"] == "out"]
    day = argmax(out.groupby(out["date"].dt.day_of_week).size())
    q = (
        "On which day of the week were the most outbound movements recorded? "
        "Answer with the day name in English."
    )
    return Draft(q, "string", WEEKDAYS[day])


@template("unanswerable", "medium")
def un_sales_attributes(d, rng):
    q = pick(
        rng,
        (
            "What is the profit margin on Electronics products?",
            "At what price was SKU-1001 sold to customers?",
            "Which customer received the most shipments?",
            "Which employee processed the most movements?",
        ),
    )
    return Draft(q, "unanswerable", None)


@template("unanswerable", "hard")
def un_out_of_range(d, rng):
    q = pick(
        rng,
        (
            "How many units were shipped out in 2024?",
            "Which warehouse recorded the most movements in August 2025?",
            "What was the value of the stock in warehouse W1 at the end of September 2025?",
            "Which category had the highest value shipped in December 2025?",
        ),
    )
    return Draft(q, "unanswerable", None)


@template("unanswerable", "hard")
def un_missing_measure(d, rng):
    q = pick(
        rng,
        (
            "What is the average delivery time from suppliers, in days?",
            "How many units were damaged in transit?",
            "What is the storage capacity of warehouse W2?",
            "What share of the shipments arrived late to the customer?",
        ),
    )
    return Draft(q, "unanswerable", None)


DOMAIN = Domain(
    name="inventory",
    tables=("products", "movements"),
    dictionary=DICTIONARY,
    generate=generate,
    templates=(
        units_shipped,
        stock_level,
        top_category_shipped,
        busiest_warehouse,
        distinct_skus_shipped,
        busiest_receiving_month,
        mean_shipment,
        top_suppliers,
        compare_warehouses,
        stock_value,
        large_shipments,
        busiest_shipping_weekday,
        un_sales_attributes,
        un_out_of_range,
        un_missing_measure,
    ),
)
