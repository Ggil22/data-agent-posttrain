"""Outpatient clinic appointments, January to June 2025."""

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

SPECIALTIES = {
    "Cardiology": 120,
    "Dermatology": 90,
    "Pediatrics": 70,
    "Neurology": 130,
    "General Practice": 50,
}
SURNAMES = (
    "Alvarez", "Brennan", "Costa", "Dubois", "Eriksen", "Fischer", "Garcia", "Haddad", "Ivanova",
    "Jensen", "Kowalski", "Laurent", "Moreau", "Novak", "Okafor", "Petrov", "Quinn", "Rossi",
    "Schmidt", "Tanaka", "Ulrich", "Varga", "Weber", "Yilmaz",
)  # fmt: skip
CITIES = ("Northbury", "Eastwick", "Southport", "Westfield", "Midvale")
CHANNELS = ("phone", "web", "app")
NO_SHOW_PENALTY = 20

DICTIONARY = f"""\
# Data dictionary

## doctors.csv
- doctor_id, name, specialty.

## patients.csv
- patient_id, birth_year, city, insurance (public or private).

Patients with patient_id 9000 or above are test accounts created by the IT team. Exclude
their appointments from every analysis.

## appointments.csv
One row per appointment, January to June 2025.
- appointment_id: appointment identifier.
- patient_id, doctor_id: reference patients.csv and doctors.csv.
- booked_on: date the appointment was booked (YYYY-MM-DD).
- scheduled_date: date of the appointment. Two booking systems wrote this field: YYYY-MM-DD
  and DD.MM.YYYY (day first).
- scheduled_time: start time (HH:MM).
- channel: how it was booked: phone, web or app.
- status: attended, no_show (the patient did not come) or cancelled.
- fee: amount billed for the visit. The app stores it in cents; phone and web bookings store
  it in EUR. Empty when nothing was billed (only attended appointments are billed a fee).

Billing rule: private patients who miss an appointment (status no_show) are charged a
{NO_SHOW_PENALTY} EUR penalty. Penalties are not in the fee column. Public patients are never
charged for a missed appointment.
"""


def generate(rng: np.random.Generator) -> Data:
    n_doc = 18
    specialty = rng.choice(list(SPECIALTIES), n_doc)
    doctors = pd.DataFrame(
        {
            "doctor_id": np.arange(1, n_doc + 1),
            "name": [f"Dr. {s}" for s in rng.permutation(SURNAMES)[:n_doc]],
            "specialty": specialty,
        }
    )

    n_pat = 900
    test_ids = np.arange(9001, 9006)
    patient_id = np.concatenate([np.arange(1, n_pat + 1), test_ids])
    k = len(patient_id)
    patients = pd.DataFrame(
        {
            "patient_id": patient_id,
            "birth_year": rng.integers(1935, 2023, k),
            "city": rng.choice(CITIES, k, p=rng.dirichlet(np.full(len(CITIES), 4))),
            "insurance": rng.choice(["public", "private"], k, p=[0.7, 0.3]),
        }
    )

    n = int(rng.integers(3000, 4200))
    days = pd.date_range("2025-01-01", "2025-06-30", freq="D")
    days = days[days.day_of_week < 6]  # closed on Sundays
    day_w = np.where(days.day_of_week == 5, 0.4, 1.0) * rng.uniform(0.7, 1.3, len(days))
    scheduled = days[rng.choice(len(days), n, p=day_w / day_w.sum())]
    wait = rng.integers(0, 61, n)
    booked = scheduled - pd.to_timedelta(wait, unit="D")
    slot = rng.integers(0, 24, n)  # half-hour slots from 08:00 to 19:30

    pid = np.where(rng.random(n) < 0.02, rng.choice(test_ids, n), rng.integers(1, n_pat + 1, n))
    doc = rng.integers(1, n_doc + 1, n)
    channel = rng.choice(CHANNELS, n, p=[0.35, 0.35, 0.3])

    # Whether booking far ahead makes no-shows likelier or rarer differs between seeds.
    p_no_show = 0.08 + rng.uniform(-0.06, 0.12) * wait / 60
    u = rng.random(n)
    status = np.where(u < 0.1, "cancelled", np.where(u < 0.1 + p_no_show, "no_show", "attended"))
    base_fee = np.array([SPECIALTIES[specialty[i - 1]] for i in doc], dtype=float)
    fee = np.round(base_fee * rng.choice([0.9, 1.0, 1.0, 1.15], n), 2)
    fee = np.where(channel == "app", np.round(fee * 100), fee)
    fee = np.where(status == "attended", fee, np.nan)

    iso = rng.random(n) < 0.7
    scheduled_txt = np.where(iso, scheduled.strftime("%Y-%m-%d"), scheduled.strftime("%d.%m.%Y"))
    appointments = pd.DataFrame(
        {
            "appointment_id": np.arange(50001, 50001 + n),
            "patient_id": pid,
            "doctor_id": doc,
            "booked_on": booked.strftime("%Y-%m-%d"),
            "scheduled_date": scheduled_txt,
            "scheduled_time": [f"{8 + s // 2:02d}:{30 * (s % 2):02d}" for s in slot],
            "channel": channel,
            "status": status,
            "fee": fee,
        }
    )
    return {"doctors": doctors, "patients": patients, "appointments": appointments}


def _appointments(d: Data) -> pd.DataFrame:
    """Appointments with the documented rules applied, joined to patients and doctors."""
    a = d["appointments"]
    a = a[a["patient_id"] < 9000]
    raw = a["scheduled_date"]
    date = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce").fillna(
        pd.to_datetime(raw, format="%d.%m.%Y", errors="coerce")
    )
    fee = np.where(a["channel"] == "app", a["fee"] / 100, a["fee"])
    a = a.assign(
        date=date,
        fee_eur=np.nan_to_num(fee),
        wait=(date - pd.to_datetime(a["booked_on"])).dt.days,
    )
    return a.merge(d["patients"], on="patient_id").merge(d["doctors"], on="doctor_id")


def _attended(d: Data) -> pd.DataFrame:
    a = _appointments(d)
    return a[a["status"] == "attended"]


def _no_show_rate(a: pd.DataFrame) -> float:
    kept = a[a["status"] != "cancelled"]
    return ratio_pct((kept["status"] == "no_show").sum(), len(kept))


@template("aggregation", "easy")
def attended_in_month(d, rng):
    m = pick(rng, range(1, 7))
    a = _attended(d)
    q = f"How many appointments in {MONTHS[m - 1]} 2025 were attended?"
    return Draft(q, "number", int((a["date"].dt.month == m).sum()))


@template("business_rules", "hard")
def billed_revenue(d, rng):
    m = pick(rng, range(1, 7))
    a = _appointments(d)
    a = a[a["date"].dt.month == m]
    penalties = ((a["status"] == "no_show") & (a["insurance"] == "private")).sum()
    total = a["fee_eur"].sum() + NO_SHOW_PENALTY * penalties
    q = (
        f"How much did the clinic bill in total, in EUR, for appointments in {MONTHS[m - 1]} "
        "2025, including no-show penalties? Round to 2 decimals."
    )
    return Draft(q, "number", float(total), decimals=2)


@template("groupby", "medium")
def no_show_specialty(d, rng):
    a = _appointments(d)
    rates = a.groupby("specialty").apply(_no_show_rate, include_groups=False)
    q = (
        "Which specialty has the highest no-show rate? Measure it as no-shows divided by "
        "appointments that were not cancelled."
    )
    return Draft(q, "string", argmax(rates))


@template("cleaning", "medium")
def mean_fee(d, rng):
    spec = pick(rng, SPECIALTIES)
    a = _attended(d)
    mean = a.loc[a["specialty"] == spec, "fee_eur"].mean()
    if np.isnan(mean):
        raise Skip("no appointments")
    q = f"What is the average fee, in EUR, of attended {spec} appointments? Round to 2 decimals."
    return Draft(q, "number", float(mean), decimals=2)


@template("join", "medium")
def top_doctor(d, rng):
    m = pick(rng, range(1, 7))
    a = _attended(d)
    winner = argmax(a[a["date"].dt.month == m].groupby("name").size())
    q = (
        f"Which doctor attended the most appointments in {MONTHS[m - 1]} 2025? "
        "Answer with the doctor's name."
    )
    return Draft(q, "string", winner)


@template("join", "hard")
def older_patients(d, rng):
    year = pick(rng, (1950, 1960, 1970))
    a = _attended(d)
    n = a.loc[a["birth_year"] <= year, "patient_id"].nunique()
    q = f"How many distinct patients born in {year} or earlier attended at least one appointment?"
    return Draft(q, "number", int(n))


@template("time_series", "medium")
def busiest_month(d, rng):
    a = _appointments(d)
    m = argmax(a.groupby(a["date"].dt.month).size())
    q = (
        "In which month of 2025 were the most appointments scheduled (whatever their status)? "
        "Answer with the month name in English."
    )
    return Draft(q, "string", MONTHS[m - 1])


@template("time_series", "medium")
def saturday_appointments(d, rng):
    status = pick(rng, ("attended", "no_show", "cancelled"))
    a = _appointments(d)
    n = int(((a["date"].dt.day_of_week == 5) & (a["status"] == status)).sum())
    q = f"How many appointments scheduled on a Saturday have the status {status}?"
    return Draft(q, "number", n)


@template("statistics", "medium")
def mean_wait(d, rng):
    channel = pick(rng, CHANNELS)
    a = _appointments(d)
    mean = a.loc[a["channel"] == channel, "wait"].mean()
    q = (
        f"On average, how many days in advance were {channel} bookings made (days between "
        "booked_on and the appointment date)? Round to 1 decimal."
    )
    return Draft(q, "number", float(mean), decimals=1)


@template("groupby", "medium")
def cities_by_patients(d, rng):
    a = _attended(d)
    order = top_k(a.groupby("city")["patient_id"].nunique(), len(CITIES))
    q = (
        "List all the cities, ordered by the number of distinct patients who attended at least "
        "one appointment, from most to fewest."
    )
    return Draft(q, "list_ordered", order)


@template("multi_step", "hard")
def private_fee_share(d, rng):
    a = _attended(d)
    share = ratio_pct(a.loc[a["insurance"] == "private", "fee_eur"].sum(), a["fee_eur"].sum())
    q = (
        "What percentage of the fees billed for attended appointments came from private "
        "patients? Do not include no-show penalties. Round to 1 decimal."
    )
    return Draft(q, "number", float(share), decimals=1)


@template("statistics", "hard")
def early_booking_no_shows(d, rng):
    days = pick(rng, (21, 30, 45))
    a = _appointments(d)
    early, late = _no_show_rate(a[a["wait"] > days]), _no_show_rate(a[a["wait"] <= days])
    if np.isclose(early, late):
        raise Skip("equal rates")
    q = (
        f"Is the no-show rate higher for appointments booked more than {days} days in advance "
        f"than for appointments booked {days} days or less in advance? Measure the rate over "
        "appointments that were not cancelled."
    )
    return Draft(q, "bool", bool(early > late))


@template("filtering", "easy")
def cancelled_by_channel(d, rng):
    channel = pick(rng, CHANNELS)
    a = _appointments(d)
    n = int(((a["channel"] == channel) & (a["status"] == "cancelled")).sum())
    return Draft(f"How many appointments booked by {channel} were cancelled?", "number", n)


@template("filtering", "medium")
def afternoon_visits(d, rng):
    spec = pick(rng, SPECIALTIES)
    a = _attended(d)
    n = int(((a["specialty"] == spec) & (a["scheduled_time"] >= "14:00")).sum())
    q = f"How many attended {spec} appointments started at or after 14:00?"
    return Draft(q, "number", n)


@template("unanswerable", "medium")
def un_patient_attributes(d, rng):
    q = pick(
        rng,
        (
            "What is the most common diagnosis among the patients?",
            "What is the average weight of the patients?",
            "How many of the patients are smokers?",
            "How many patients were referred by another hospital?",
        ),
    )
    return Draft(q, "unanswerable", None)


@template("unanswerable", "hard")
def un_out_of_range(d, rng):
    q = pick(
        rng,
        (
            "Which doctor attended the most appointments in December 2025?",
            "What was the average fee of attended appointments in 2024?",
            "What was the no-show rate in September 2025?",
            "Which specialty had the highest average fee in the second half of 2025?",
        ),
    )
    return Draft(q, "unanswerable", None)


@template("unanswerable", "hard")
def un_missing_measure(d, rng):
    q = pick(
        rng,
        (
            "What is the average consultation length, in minutes?",
            "What is the average salary of the doctors?",
            "What is the average patient satisfaction score for Cardiology?",
            "How long did patients wait in the waiting room on average?",
        ),
    )
    return Draft(q, "unanswerable", None)


DOMAIN = Domain(
    name="clinic",
    tables=("doctors", "patients", "appointments"),
    dictionary=DICTIONARY,
    generate=generate,
    templates=(
        attended_in_month,
        billed_revenue,
        no_show_specialty,
        mean_fee,
        top_doctor,
        older_patients,
        busiest_month,
        saturday_appointments,
        mean_wait,
        cities_by_patients,
        private_fee_share,
        early_booking_no_shows,
        cancelled_by_channel,
        afternoon_visits,
        un_patient_attributes,
        un_out_of_range,
        un_missing_measure,
    ),
)
