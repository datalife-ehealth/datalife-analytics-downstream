"""Deterministic synthetic longitudinal data generator (issue #2).

Design notes
------------
* Every subject draws from its own child of ``SeedSequence(seed)``. Subject ``i`` is
  therefore identical whatever the cohort size, and adding a new generator module
  later (which takes a fresh, separately keyed stream) does not shift existing values.
* Distributions are rough, generic, and hand-set. They are not fitted to, sampled
  from, or calibrated against any real patient data and must not be used for
  clinical inference.
* Tables hold canonical text for dates (``YYYY-MM-DD``) and timestamps
  (``YYYY-MM-DDTHH:MM:00Z``) so CSV output is byte-for-byte reproducible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from .contract import Contract, load_contract

GENERATOR_NAME = "datalife-longitudinal"
GENERATOR_VERSION = "0.1.1"

MAX_SUBJECTS = 100_000
MIN_WINDOW_DAYS = 30

SEX = (["F", "M", "U"], [0.49, 0.49, 0.02])
AGE_BANDS = (["18-29", "30-44", "45-59", "60-74", "75+"], [0.22, 0.28, 0.25, 0.18, 0.07])
REGIONS = [f"R0{i}" for i in range(1, 9)]
ENCOUNTER_TYPES = ["routine", "urgent", "telehealth"]
LAB_MARKERS = {"GLU", "HBA1C", "HGB"}

# Probability that a marker is scheduled at an encounter, by encounter type.
SCHEDULE_PROB: dict[str, dict[str, float]] = {
    "routine": {"SBP": 0.95, "DBP": 0.95, "BMI": 0.45, "GLU": 0.35, "HBA1C": 0.30, "HGB": 0.25},
    "urgent": {"SBP": 0.98, "DBP": 0.98, "BMI": 0.10, "GLU": 0.60, "HBA1C": 0.15, "HGB": 0.55},
    "telehealth": {"SBP": 0.50, "DBP": 0.50, "BMI": 0.20, "GLU": 0.05, "HBA1C": 0.0, "HGB": 0.0},
}
# Within-subject visit-to-visit noise (standard deviation), in marker units.
WITHIN_SD = {"SBP": 7.0, "DBP": 5.0, "GLU": 0.5, "HBA1C": 0.2, "HGB": 0.5, "BMI": 0.4}
DECIMALS = {"SBP": 0, "DBP": 0, "GLU": 1, "HBA1C": 1, "HGB": 1, "BMI": 1}


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int
    n_subjects: int = 200
    start: date = date(2024, 1, 1)
    end: date = date(2025, 12, 31)
    not_resulted_rate: float = 0.03
    contract_version: str = "0.1.0"

    def __post_init__(self) -> None:
        problems = []
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            problems.append("seed must be a non-negative integer")
        if not isinstance(self.n_subjects, int) or not 1 <= self.n_subjects <= MAX_SUBJECTS:
            problems.append(f"n_subjects must be an integer between 1 and {MAX_SUBJECTS}")
        if not isinstance(self.start, date) or not isinstance(self.end, date):
            problems.append("start and end must be dates")
        elif (self.end - self.start).days < MIN_WINDOW_DAYS:
            problems.append(f"end must be at least {MIN_WINDOW_DAYS} days after start")
        if not 0.0 <= self.not_resulted_rate < 0.5:
            problems.append("not_resulted_rate must be in [0, 0.5)")
        if problems:
            raise ValueError("Invalid generator configuration: " + "; ".join(problems))

    def as_dict(self) -> dict:
        d = asdict(self)
        d["start"] = self.start.isoformat()
        d["end"] = self.end.isoformat()
        return d


@dataclass
class _Subject:
    row: dict
    encounters: list[dict] = field(default_factory=list)
    observations: list[dict] = field(default_factory=list)


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:00Z")


def _baselines(rng: np.random.Generator, sex: str, age_idx: int, z: float) -> dict[str, float]:
    hba1c = 5.3 + 0.12 * age_idx + 0.45 * z + rng.normal(0, 0.3)
    hgb_mean = {"F": 13.3, "M": 14.8, "U": 14.0}[sex]
    return {
        "SBP": 118 + 6 * age_idx + 8 * z + rng.normal(0, 8),
        "DBP": 76 + 2 * age_idx + 4 * z + rng.normal(0, 6),
        "BMI": 26 + 0.6 * age_idx + 2.5 * z + rng.normal(0, 3.5),
        "HBA1C": max(hba1c, 4.3),
        # Fasting glucose sits below the HbA1c-implied average glucose (1.59 * A1c - 2.59).
        "GLU": max(1.59 * hba1c - 3.3 + rng.normal(0, 0.4), 4.0),
        "HGB": hgb_mean + rng.normal(0, 1.0),
    }


def _annual_trend(rng: np.random.Generator, z: float) -> dict[str, float]:
    pos = max(z, 0.0)
    return {
        "SBP": rng.normal(1.0 * pos, 1.5),
        "DBP": rng.normal(0.4 * pos, 0.8),
        "BMI": rng.normal(0.2 * pos, 0.4),
        "HBA1C": rng.normal(0.15 * pos, 0.1),
        "GLU": rng.normal(0.25 * pos, 0.15),
        "HGB": rng.normal(0.0, 0.1),
    }


def _simulate_subject(
    rng: np.random.Generator, cfg: GeneratorConfig, contract: Contract
) -> _Subject:
    window_days = (cfg.end - cfg.start).days
    key = "syn-" + rng.bytes(8).hex()
    sex = str(rng.choice(SEX[0], p=SEX[1]))
    age_idx = int(rng.choice(len(AGE_BANDS[0]), p=AGE_BANDS[1]))
    region = str(rng.choice(REGIONS))
    enrolled = cfg.start + timedelta(days=int(rng.integers(0, window_days // 2 + 1)))
    subject = _Subject(
        row={
            "subject_key": key,
            "sex": sex,
            "age_band": AGE_BANDS[0][age_idx],
            "region": region,
            "enrolled_on": enrolled.isoformat(),
        }
    )

    # Latent cardiometabolic tendency, shifted with age. Purely synthetic.
    z = float(rng.normal(0, 1) + 0.35 * age_idx - 0.7)
    base = _baselines(rng, sex, age_idx, z)
    trend = _annual_trend(rng, z)

    active_days = (cfg.end - enrolled).days
    if active_days <= 0:
        return subject
    rate = 3.0 + 1.5 * max(z, 0.0)
    n_enc = int(rng.poisson(rate * active_days / 365.25))
    if n_enc == 0:
        return subject

    day_offsets = np.unique(rng.integers(0, active_days, size=n_enc))
    p_urgent = 0.10 + (0.10 if z > 1 else 0.0)
    type_p = [0.80 - p_urgent, p_urgent, 0.20]
    used_days: set[date] = set()

    for off in day_offsets:
        etype = ENCOUNTER_TYPES[int(rng.choice(3, p=type_p))]
        when = _visit_time(rng, enrolled + timedelta(days=int(off)), etype)
        if when is None or when.date() >= cfg.end or when.date() in used_days:
            continue
        used_days.add(when.date())
        t_years = (when.date() - enrolled).days / 365.25
        ts = _ts(when)
        enc_ref = len(subject.encounters)
        subject.encounters.append(
            {"subject_key": key, "encounter_at": ts, "encounter_type": etype, "_ref": enc_ref}
        )
        for code, value in _measure(rng, etype, base, trend, t_years, contract):
            reason = None
            if code in LAB_MARKERS and rng.random() < cfg.not_resulted_rate:
                value, reason = None, "not_resulted"
            source = "xml" if code in LAB_MARKERS and rng.random() < 0.6 else "json"
            subject.observations.append(
                {
                    "subject_key": key,
                    "_enc_ref": enc_ref,
                    "observed_at": ts,
                    "marker_code": code,
                    "value": value,
                    "unit": contract.markers[code]["unit"],
                    "source_kind": source,
                    "missing_reason": reason,
                }
            )
    return subject


def _visit_time(rng: np.random.Generator, day: date, etype: str) -> datetime | None:
    """Routine and telehealth: weekdays 08:00-17:00. Urgent: any day 07:00-23:00."""
    if etype == "urgent":
        minute = int(rng.integers(7 * 60, 23 * 60))
    else:
        if day.weekday() >= 5:  # weekend: move to the following Monday
            day = day + timedelta(days=7 - day.weekday())
        minute = int(rng.integers(8 * 60, 17 * 60))
    return datetime.combine(day, datetime.min.time(), tzinfo=UTC) + timedelta(minutes=minute)


def _measure(
    rng: np.random.Generator,
    etype: str,
    base: dict[str, float],
    trend: dict[str, float],
    t_years: float,
    contract: Contract,
) -> list[tuple[str, float]]:
    """Values for the markers taken at one encounter, in contract marker order."""
    probs = SCHEDULE_PROB[etype]
    taken = {code: rng.random() < probs[code] for code in contract.markers}
    taken["DBP"] = taken["SBP"]  # blood pressure is always recorded as a pair

    def level(code: str, z: float) -> float:
        lo, hi = contract.markers[code]["range"]
        raw = base[code] + trend[code] * t_years + WITHIN_SD[code] * z
        return float(np.clip(raw, lo, hi))

    sbp_z = rng.normal()
    values = {
        "SBP": level("SBP", sbp_z),
        # Diastolic noise is partly shared with systolic, and pulse pressure stays >= 15 mmHg.
        "DBP": level("DBP", 0.6 * sbp_z + 0.8 * rng.normal()),
    }
    values["DBP"] = min(values["DBP"], values["SBP"] - 15)
    for code in ("GLU", "HBA1C", "HGB", "BMI"):
        values[code] = level(code, rng.normal())

    return [(code, round(values[code], DECIMALS[code])) for code in contract.markers if taken[code]]


def generate(
    cfg: GeneratorConfig,
    contract: Contract | None = None,
    progress: Callable[[int], None] | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate the three contract tables for ``cfg``. Output is fully determined by cfg."""
    contract = contract or load_contract(cfg.contract_version)
    children = np.random.SeedSequence(cfg.seed).spawn(cfg.n_subjects)

    subjects = [
        _simulate_subject(np.random.default_rng(child), cfg, contract) for child in children
    ]
    if progress:
        progress(len(subjects))

    keys = [s.row["subject_key"] for s in subjects]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Synthetic subject key collision; choose a different seed")

    subjects.sort(key=lambda s: s.row["subject_key"])
    subj_rows, enc_rows, obs_rows = [], [], []
    enc_counter = obs_counter = 0
    for s in subjects:
        subj_rows.append(s.row)
        ref_to_id: dict[int, str] = {}
        for e in sorted(s.encounters, key=lambda e: (e["encounter_at"], e["_ref"])):
            enc_counter += 1
            ref_to_id[e["_ref"]] = f"enc-{enc_counter:07d}"
            enc_rows.append(
                {
                    "encounter_id": ref_to_id[e["_ref"]],
                    "subject_key": e["subject_key"],
                    "encounter_at": e["encounter_at"],
                    "encounter_type": e["encounter_type"],
                }
            )
        ordered = sorted(
            s.observations, key=lambda o: (o["observed_at"], o["marker_code"], o["_enc_ref"])
        )
        for o in ordered:
            obs_counter += 1
            obs_rows.append(
                {
                    "observation_id": f"obs-{obs_counter:08d}",
                    "subject_key": o["subject_key"],
                    "encounter_id": ref_to_id[o["_enc_ref"]],
                    "observed_at": o["observed_at"],
                    "marker_code": o["marker_code"],
                    "value": o["value"],
                    "unit": o["unit"],
                    "source_kind": o["source_kind"],
                    "missing_reason": o["missing_reason"],
                }
            )

    tables = {
        "subjects": pd.DataFrame(subj_rows, columns=contract.columns("subjects")),
        "encounters": pd.DataFrame(enc_rows, columns=contract.columns("encounters")),
        "observations": pd.DataFrame(obs_rows, columns=contract.columns("observations")),
    }
    tables["observations"]["value"] = tables["observations"]["value"].astype("float64")
    return tables
