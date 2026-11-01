"""Daily smart-meter readings, January to March 2025."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dapt.domains.base import (
    MONTHS,
    Data,
    Domain,
    Draft,
    Skip,
    argmax,
    pick,
    ratio_pct,
    template,
    top_k,
)

BUILDING_TYPES = {"residential": 12.0, "office": 85.0, "retail": 60.0}
DISTRICTS = ("Docklands", "Greenhill", "Kingsbridge", "Lakeside", "Millbrook")
ERROR_CODE = 99999.9
WEEKDAY_PRICE, WEEKEND_PRICE = 0.18, 0.12

DICTIONARY = f"""\
# Data dictionary

## meters.csv
- meter_id: meter identifier.
- building_type: residential, office or retail.
- district: city district.
- unit: unit the meter reports in. Older meters report Wh (watt-hours), newer ones kWh.
  1 kWh = 1000 Wh.

## readings.csv
One reading per meter and day, January to March 2025.
- date: day of the reading (YYYY-MM-DD).
- meter_id: references meters.csv.
- consumption: energy consumed that day, in the unit given in meters.csv.

A consumption of {ERROR_CODE} is an error code sent by meters that failed to read; it is
not a real value and must be ignored. Days with no row for a meter are readings that were
never received.

Tariff: energy costs {WEEKDAY_PRICE} EUR per kWh on weekdays (Monday to Friday) and
{WEEKEND_PRICE} EUR per kWh on weekends.
"""


def generate(rng: np.random.Generator) -> Data:
    n_m = int(rng.integers(40, 61))
    btype = rng.choice(list(BUILDING_TYPES), n_m, p=[0.6, 0.25, 0.15])
    meters = pd.DataFrame(
        {
            "meter_id": [f"M{i:03d}" for i in range(1, n_m + 1)],
            "building_type": btype,
            "district": rng.choice(DISTRICTS, n_m),
            "unit": np.where(rng.random(n_m) < 0.3, "Wh", "kWh"),
        }
    )

    days = pd.date_range("2025-01-01", "2025-03-31", freq="D")
    weekend_factor = {"residential": 1.25, "office": 0.25, "retail": rng.uniform(0.8, 1.3)}
    rows = []
    for _, m in meters.iterrows():
        base = BUILDING_TYPES[m["building_type"]] * rng.lognormal(0, 0.35)
        season = 1 + rng.uniform(-0.3, 0.1) * np.arange(len(days)) / len(days)
        weekend = np.where(days.day_of_week >= 5, weekend_factor[m["building_type"]], 1.0)
        kwh = base * season * weekend * rng.normal(1, 0.12, len(days))
        value = np.round(np.clip(kwh, 0.1, None), 2)
        if m["unit"] == "Wh":
            value = np.round(value * 1000)
        value[rng.random(len(days)) < 0.01] = ERROR_CODE
        keep = rng.random(len(days)) >= rng.uniform(0, 0.06)
        rows.append(
            pd.DataFrame(
                {"date": days[keep].strftime("%Y-%m-%d"), "meter_id": m["meter_id"]}
                | {"consumption": value[keep]}
            )
        )
    readings = pd.concat(rows).sort_values(["date", "meter_id"]).reset_index(drop=True)
    return {"meters": meters, "readings": readings}


def _readings(d: Data) -> pd.DataFrame:
    """Valid readings in kWh, joined to the meter attributes."""
    r = d["readings"]
    r = r[r["consumption"] != ERROR_CODE].merge(d["meters"], on="meter_id")
    kwh = np.where(r["unit"] == "Wh", r["consumption"] / 1000, r["consumption"])
    date = pd.to_datetime(r["date"])
    return r.assign(kwh=kwh, date=date, weekend=date.dt.day_of_week >= 5)


@template("aggregation", "medium")
def total_consumption(d, rng):
    m = pick(rng, (1, 2, 3))
    r = _readings(d)
    total = r.loc[r["date"].dt.month == m, "kwh"].sum()
    q = (
        f"What was the total energy consumption, in kWh, across all meters in {MONTHS[m - 1]} "
        "2025? Round to 2 decimals."
    )
    return Draft(q, "number", float(total), decimals=2)


@template("groupby", "medium")
def mean_daily_by_type(d, rng):
    btype = pick(rng, BUILDING_TYPES)
    r = _readings(d)
    mean = r.loc[r["building_type"] == btype, "kwh"].mean()
    q = (
        f"What is the average daily consumption, in kWh, of a {btype} meter (the mean over all "
        "valid daily readings)? Round to 2 decimals."
    )
    return Draft(q, "number", float(mean), decimals=2)


@template("join", "medium")
def top_district(d, rng):
    btype = pick(rng, BUILDING_TYPES)
    r = _readings(d)
    r = r[r["building_type"] == btype]
    if r.empty:
        raise Skip("no meters of this type")
    q = f"Which district's {btype} meters consumed the most energy in total?"
    return Draft(q, "string", argmax(r.groupby("district")["kwh"].sum()))


@template("time_series", "hard")
def missing_days(d, rng):
    meter = pick(rng, d["meters"]["meter_id"])
    r = d["readings"]
    received = r.loc[r["meter_id"] == meter, "date"].nunique()
    q = (
        f"How many daily readings were never received from meter {meter} between 1 January and "
        "31 March 2025?"
    )
    return Draft(q, "number", 90 - int(received))


@template("time_series", "medium")
def peak_day(d, rng):
    r = _readings(d)
    day = argmax(r.groupby("date")["kwh"].sum())
    q = (
        "On which day was the total consumption across all meters highest? "
        "Answer with the date as YYYY-MM-DD."
    )
    return Draft(q, "string", day.strftime("%Y-%m-%d"))


@template("business_rules", "hard")
def meter_cost(d, rng):
    m = pick(rng, (1, 2, 3))
    meter = pick(rng, d["meters"]["meter_id"])
    r = _readings(d)
    r = r[(r["meter_id"] == meter) & (r["date"].dt.month == m)]
    cost = (r["kwh"] * np.where(r["weekend"], WEEKEND_PRICE, WEEKDAY_PRICE)).sum()
    q = (
        f"How much did the energy measured by meter {meter} in {MONTHS[m - 1]} 2025 cost, in "
        "EUR, according to the tariff? Round to 2 decimals."
    )
    return Draft(q, "number", float(cost), decimals=2)


@template("filtering", "easy")
def high_readings(d, rng):
    btype = pick(rng, BUILDING_TYPES)
    x = {"residential": 15, "office": 100, "retail": 70}[btype]
    r = _readings(d)
    n = int(((r["building_type"] == btype) & (r["kwh"] > x)).sum())
    q = f"How many daily readings from {btype} meters were above {x} kWh?"
    return Draft(q, "number", n)


@template("statistics", "medium")
def weekend_mean(d, rng):
    btype = pick(rng, BUILDING_TYPES)
    r = _readings(d)
    mean = r.loc[(r["building_type"] == btype) & r["weekend"], "kwh"].mean()
    q = (
        f"What is the average daily consumption, in kWh, of {btype} meters on weekends? "
        "Round to 2 decimals."
    )
    return Draft(q, "number", float(mean), decimals=2)


@template("statistics", "medium")
def weekend_higher(d, rng):
    btype = pick(rng, BUILDING_TYPES)
    r = _readings(d)
    means = r[r["building_type"] == btype].groupby("weekend")["kwh"].mean()
    if len(means) < 2 or np.isclose(means[True], means[False]):
        raise Skip("no difference")
    q = f"Do {btype} meters consume more per day on average on weekends than on weekdays?"
    return Draft(q, "bool", bool(means[True] > means[False]))


@template("groupby", "medium")
def top_meters(d, rng):
    m = pick(rng, (1, 2, 3))
    r = _readings(d)
    top = top_k(r[r["date"].dt.month == m].groupby("meter_id")["kwh"].sum(), 3)
    q = (
        f"List the 3 meters with the highest total consumption in {MONTHS[m - 1]} 2025, "
        "from highest to lowest."
    )
    return Draft(q, "list_ordered", top)


@template("cleaning", "easy")
def error_readings(d, rng):
    district = pick(rng, DISTRICTS)
    r = d["readings"].merge(d["meters"], on="meter_id")
    n = int(((r["district"] == district) & (r["consumption"] == ERROR_CODE)).sum())
    q = f"How many readings from meters in {district} are error codes?"
    return Draft(q, "number", n)


@template("multi_step", "hard")
def consumption_change(d, rng):
    btype = pick(rng, BUILDING_TYPES)
    r = _readings(d)
    r = r[r["building_type"] == btype]
    by_month = r.groupby(r["date"].dt.month)["kwh"].mean()
    change = ratio_pct(by_month[3] - by_month[1], by_month[1])
    q = (
        f"By what percentage did the average daily consumption per {btype} meter change from "
        "January to March 2025? Round to 1 decimal (negative if it fell)."
    )
    return Draft(q, "number", float(change), decimals=1)


@template("multi_step", "hard")
def office_share(d, rng):
    district = pick(rng, DISTRICTS)
    r = _readings(d)
    r = r[r["district"] == district]
    share = ratio_pct(r.loc[r["building_type"] == "office", "kwh"].sum(), r["kwh"].sum())
    q = (
        f"What percentage of the total energy consumed in {district} was consumed by office "
        "meters? Round to 1 decimal."
    )
    return Draft(q, "number", float(share), decimals=1)


@template("unanswerable", "medium")
def un_building_attributes(d, rng):
    q = pick(
        rng,
        (
            "How many people live in the buildings with residential meters?",
            "What is the floor area of the largest office building?",
            "Which energy supplier does meter M001 use?",
            "How old are the buildings in Docklands on average?",
        ),
    )
    return Draft(q, "unanswerable", None)


@template("unanswerable", "hard")
def un_out_of_range(d, rng):
    q = pick(
        rng,
        (
            "What was the total consumption across all meters in April 2025?",
            "Which meter consumed the most energy in 2024?",
            "What was the average daily consumption of office meters in December 2025?",
            "On which day of June 2025 was the total consumption highest?",
        ),
    )
    return Draft(q, "unanswerable", None)


@template("unanswerable", "hard")
def un_missing_measure(d, rng):
    q = pick(
        rng,
        (
            "At what hour of the day does consumption peak?",
            "How many tonnes of CO2 did the residential meters cause?",
            "What share of the energy came from solar panels?",
            "What was the peak power demand, in kW, of meter M002?",
        ),
    )
    return Draft(q, "unanswerable", None)


DOMAIN = Domain(
    name="energy",
    tables=("meters", "readings"),
    dictionary=DICTIONARY,
    generate=generate,
    templates=(
        total_consumption,
        mean_daily_by_type,
        top_district,
        missing_days,
        peak_day,
        meter_cost,
        high_readings,
        weekend_mean,
        weekend_higher,
        top_meters,
        error_readings,
        consumption_change,
        office_share,
        un_building_attributes,
        un_out_of_range,
        un_missing_measure,
    ),
)
