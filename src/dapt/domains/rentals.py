"""Bike-sharing trips, April to June 2025."""

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
    ratio_pct,
    template,
    top_k,
)

DISTRICTS = ("Old Town", "Riverside", "University", "Harbour", "Market", "Hillside")
NAME_PARTS = (
    ("North", "South", "East", "West", "Central", "Upper"),
    ("Square", "Gate", "Park", "Bridge", "Station", "Plaza"),
)
HOUR_PROFILE = np.array(
    [1, 1, 1, 1, 1, 2, 4, 9, 12, 8, 6, 6, 7, 7, 6, 6, 8, 11, 12, 9, 6, 4, 3, 2], dtype=float
)
BIKES = ("classic", "electric")

DICTIONARY = """\
# Data dictionary

## stations.csv
Docking stations of the city bike-sharing service.
- station_id, name, district, capacity (number of docks).

## trips.csv
One row per trip, April to June 2025.
- trip_id: trip identifier.
- started_at: start time (YYYY-MM-DD HH:MM:SS, local time).
- start_station_id, end_station_id: reference stations.csv.
- bike_type: classic or electric.
- rider_type: member, casual or staff. Typed in by operators, so capitalisation and spacing
  vary.
- duration_min: trip duration in minutes. A value of -1 means the GPS signal was lost and the
  duration is unknown: leave those trips out of any duration statistic (they still count as
  trips).
- fare_eur: amount paid, in EUR. Empty when nothing was charged (rides included in a
  membership).

Staff trips are bikes moved between stations by the operator for rebalancing. They are not
customer trips and must be excluded from every analysis.

Known issue: the export duplicated some rows (exact duplicates); count each trip once.
"""


def generate(rng: np.random.Generator) -> Data:
    n_st = 30
    names = [f"{a} {b}" for a in NAME_PARTS[0] for b in NAME_PARTS[1]]
    stations = pd.DataFrame(
        {
            "station_id": np.arange(1, n_st + 1),
            "name": rng.permutation(names)[:n_st],
            "district": rng.choice(DISTRICTS, n_st),
            "capacity": rng.integers(10, 41, n_st),
        }
    )

    n = int(rng.integers(3500, 5000))
    days = pd.date_range("2025-04-01", "2025-06-30", freq="D")
    trend = 1 + rng.uniform(-0.3, 0.6) * np.arange(len(days)) / len(days)
    day_w = np.where(days.day_of_week >= 5, rng.uniform(0.5, 1.3), 1.0) * trend
    day = days[rng.choice(len(days), n, p=day_w / day_w.sum())]
    hour = rng.choice(24, n, p=HOUR_PROFILE / HOUR_PROFILE.sum())
    started = (
        day + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n), unit="s")
    )

    popularity = rng.dirichlet(np.full(n_st, 0.8))
    bike = rng.choice(BIKES, n, p=[0.6, 0.4])
    rider = rng.choice(["member", "casual", "staff"], n, p=[0.62, 0.33, 0.05])
    casual_factor = rng.uniform(0.85, 1.3)
    median = np.where(bike == "electric", 11.0, 14.0) * np.where(
        rider == "casual", casual_factor, 1
    )
    duration = np.clip(np.round(median * rng.lognormal(0, 0.5, n), 1), 1.0, 240.0)

    fare = np.where(
        rider == "casual",
        1.5 + 0.15 * duration + np.where(bike == "electric", 1.0, 0.0),
        np.where((rider == "member") & (bike == "electric"), 0.5, np.nan),
    )
    duration[rng.random(n) < 0.02] = -1

    trips = pd.DataFrame(
        {
            "started_at": started.strftime("%Y-%m-%d %H:%M:%S"),
            "start_station_id": rng.choice(n_st, n, p=popularity) + 1,
            "end_station_id": rng.choice(n_st, n, p=popularity) + 1,
            "bike_type": bike,
            "rider_type": messy_labels(rng, rider, 0.12),
            "duration_min": duration,
            "fare_eur": np.round(fare, 2),
        }
    ).sort_values("started_at", kind="stable")
    trips.insert(0, "trip_id", np.arange(1, n + 1))
    duplicates = trips.iloc[rng.choice(n, 20, replace=False)]
    trips = pd.concat([trips, duplicates]).sort_values("trip_id", kind="stable")
    return {"stations": stations, "trips": trips.reset_index(drop=True)}


def _trips(d: Data) -> pd.DataFrame:
    """Customer trips with the documented rules applied."""
    t = d["trips"].drop_duplicates()
    t = t.assign(rider=t["rider_type"].str.strip().str.lower())
    t = t[t["rider"] != "staff"]
    return t.assign(started=pd.to_datetime(t["started_at"]), fare=t["fare_eur"].fillna(0))


def _timed(d: Data) -> pd.DataFrame:
    t = _trips(d)
    return t[t["duration_min"] != -1]


def _with_district(d: Data) -> pd.DataFrame:
    s = d["stations"]
    t = _trips(d).merge(
        s[["station_id", "district"]], left_on="start_station_id", right_on="station_id"
    )
    return t.merge(
        s[["station_id", "district"]],
        left_on="end_station_id",
        right_on="station_id",
        suffixes=("", "_end"),
    )


@template("aggregation", "easy")
def trips_in_month(d, rng):
    m = pick(rng, (4, 5, 6))
    n = int((_trips(d)["started"].dt.month == m).sum())
    return Draft(f"How many trips started in {MONTHS[m - 1]} 2025?", "number", n)


@template("aggregation", "medium")
def fare_revenue(d, rng):
    rider = pick(rng, ("member", "casual"))
    t = _trips(d)
    total = t.loc[t["rider"] == rider, "fare"].sum()
    q = f"What was the total fare revenue, in EUR, paid by {rider} riders? Round to 2 decimals."
    return Draft(q, "number", float(total), decimals=2)


@template("join", "medium")
def busiest_district(d, rng):
    bike = pick(rng, BIKES)
    t = _with_district(d)
    winner = argmax(t[t["bike_type"] == bike].groupby("district").size())
    return Draft(f"Which district had the most trip starts on {bike} bikes?", "string", winner)


@template("statistics", "medium")
def mean_duration(d, rng):
    bike = pick(rng, BIKES)
    t = _timed(d)
    mean = t.loc[t["bike_type"] == bike, "duration_min"].mean()
    q = f"What is the average duration, in minutes, of trips on {bike} bikes? Round to 2 decimals."
    return Draft(q, "number", float(mean), decimals=2)


@template("filtering", "easy")
def long_trips(d, rng):
    bike, x = pick(rng, BIKES), pick(rng, (20, 30, 45))
    t = _timed(d)
    n = int(((t["bike_type"] == bike) & (t["duration_min"] > x)).sum())
    return Draft(f"How many {bike} bike trips lasted longer than {x} minutes?", "number", n)


@template("time_series", "medium")
def peak_hour(d, rng):
    weekend = pick(rng, (False, True))
    t = _trips(d)
    t = t[(t["started"].dt.day_of_week >= 5) == weekend]
    hour = argmax(t.groupby(t["started"].dt.hour).size())
    days = "weekends (Saturday and Sunday)" if weekend else "weekdays (Monday to Friday)"
    q = f"On {days}, at which hour of the day (0-23) do the most trips start?"
    return Draft(q, "number", int(hour))


@template("time_series", "medium")
def busiest_weekday(d, rng):
    t = _trips(d)
    day = argmax(t.groupby(t["started"].dt.day_of_week).size())
    q = "On which day of the week did the most trips start? Answer with the day name in English."
    return Draft(q, "string", WEEKDAYS[day])


@template("groupby", "medium")
def top_start_stations(d, rng):
    m = pick(rng, (4, 5, 6))
    t = _trips(d)
    counts = t[t["started"].dt.month == m].groupby("start_station_id").size()
    names = d["stations"].set_index("station_id")["name"]
    top = [names[i] for i in top_k(counts, 3)]
    q = (
        f"List the names of the 3 stations with the most trip starts in {MONTHS[m - 1]} 2025, "
        "from most to fewest."
    )
    return Draft(q, "list_ordered", top)


@template("multi_step", "hard")
def cross_district_share(d, rng):
    t = _with_district(d)
    share = ratio_pct((t["district"] != t["district_end"]).sum(), len(t))
    q = (
        "What percentage of trips ended in a different district from the one where they "
        "started? Round to 1 decimal."
    )
    return Draft(q, "number", float(share), decimals=1)


@template("statistics", "medium")
def casual_longer(d, rng):
    bike = pick(rng, BIKES)
    t = _timed(d)
    t = t[t["bike_type"] == bike]
    means = t.groupby("rider")["duration_min"].mean()
    if np.isclose(means["casual"], means["member"]):
        raise Skip("equal means")
    q = f"On {bike} bikes, are trips by casual riders longer on average than trips by members?"
    return Draft(q, "bool", bool(means["casual"] > means["member"]))


@template("join", "hard")
def quiet_stations(d, rng):
    m, k = pick(rng, (4, 5, 6)), pick(rng, (20, 40, 60))
    t = _trips(d)
    counts = t[t["started"].dt.month == m].groupby("start_station_id").size()
    counts = counts.reindex(d["stations"]["station_id"], fill_value=0)
    q = f"How many stations had fewer than {k} trip starts in {MONTHS[m - 1]} 2025?"
    return Draft(q, "number", int((counts < k).sum()))


@template("time_series", "hard")
def trip_growth(d, rng):
    t = _trips(d)
    by_month = t.groupby(t["started"].dt.month).size()
    change = ratio_pct(by_month[6] - by_month[4], by_month[4])
    q = (
        "By what percentage did the number of trips change from April to June 2025? "
        "Round to 1 decimal (negative if it fell)."
    )
    return Draft(q, "number", float(change), decimals=1)


@template("cleaning", "easy")
def unknown_durations(d, rng):
    n = int((_trips(d)["duration_min"] == -1).sum())
    return Draft("How many trips have an unknown duration?", "number", n)


@template("business_rules", "medium")
def member_electric_fees(d, rng):
    m = pick(rng, (4, 5, 6))
    t = _trips(d)
    sel = (t["rider"] == "member") & (t["bike_type"] == "electric") & (t["started"].dt.month == m)
    q = (
        f"How much did members pay in total, in EUR, for electric bike trips in "
        f"{MONTHS[m - 1]} 2025? Round to 2 decimals."
    )
    return Draft(q, "number", float(t.loc[sel, "fare"].sum()), decimals=2)


@template("unanswerable", "medium")
def un_rider_attributes(d, rng):
    q = pick(
        rng,
        (
            "How many riders are under 25 years old?",
            "Which rider made the most trips?",
            "What share of the riders are women?",
            "How many riders live in the Old Town district?",
        ),
    )
    return Draft(q, "unanswerable", None)


@template("unanswerable", "hard")
def un_out_of_range(d, rng):
    q = pick(
        rng,
        (
            "What was the average trip duration in July 2025?",
            "Which station had the most trip starts in March 2025?",
            "Which station had the most trip starts in 2024?",
            "Which district had the most trip starts in December 2025?",
        ),
    )
    return Draft(q, "unanswerable", None)


@template("unanswerable", "hard")
def un_missing_measure(d, rng):
    q = pick(
        rng,
        (
            "What was the average speed of electric bike trips, in km/h?",
            "Which bike, by bike ID, was used most often?",
            "How many trips were taken on rainy days?",
            "What is the average battery level of electric bikes at the start of a trip?",
        ),
    )
    return Draft(q, "unanswerable", None)


DOMAIN = Domain(
    name="rentals",
    tables=("stations", "trips"),
    dictionary=DICTIONARY,
    generate=generate,
    templates=(
        trips_in_month,
        fare_revenue,
        busiest_district,
        mean_duration,
        long_trips,
        peak_hour,
        busiest_weekday,
        top_start_stations,
        cross_district_share,
        casual_longer,
        quiet_stations,
        trip_growth,
        unknown_durations,
        member_electric_fees,
        un_rider_attributes,
        un_out_of_range,
        un_missing_measure,
    ),
)
