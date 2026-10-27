"""Bounded research question and forward-looking examples on the v0.1.0 tables.

Question
    Among synthetic subjects with at least one earlier encounter, can coarse
    demographics and prior measurements predict whether a *later* systolic blood
    pressure is elevated (>= 140 mmHg), better than a prevalence-only naive rule?

Features come only from encounters strictly earlier than the outcome timestamp.
This is a labelling rule on generated data, not a diagnostic definition.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from datalife_analytics.contract import Contract, load_contract

STUDY_ID = "elevated-sbp-forward-v1"
STUDY_QUESTION = (
    "Predict elevated later-encounter systolic blood pressure (>= 140 mmHg) "
    "from earlier encounters of the same synthetic subject."
)
ELEVATED_SBP_THRESHOLD = 140.0
NON_USE = (
    "Not diagnosis, treatment guidance, causal inference, fairness claims, "
    "or an operational public-health signal. Synthetic distributions only."
)


@dataclass(frozen=True)
class ForwardExample:
    """One subject-level prediction example.

    ``feature_dates`` are encounter timestamps that fed the covariates.
    ``outcome_date`` is strictly later than every feature date.
    """

    subject_key: str
    outcome: int
    outcome_date: str
    feature_end_date: str
    feature_dates: tuple[str, ...]
    sex: str
    age_band: str
    region: str
    n_prior: int
    last_encounter_type: str
    mean_bmi: float | None
    last_sbp: float | None
    missing_bmi_rate: float
    missing_sbp_rate: float


def _marker(obs: pd.DataFrame, code: str, name: str) -> pd.DataFrame:
    subset = obs.loc[obs["marker_code"] == code, ["encounter_id", "value"]]
    return subset.rename(columns={"value": name})


def build_forward_examples(
    tables: dict[str, pd.DataFrame],
    contract: Contract | None = None,
) -> tuple[list[ForwardExample], dict[str, int]]:
    """Build one forward example per subject that has a usable later SBP."""
    contract = contract or load_contract()
    subjects = tables["subjects"]
    enc = tables["encounters"].copy()
    obs = tables["observations"]

    enc = enc.merge(_marker(obs, "SBP", "sbp"), on="encounter_id", how="left")
    enc = enc.merge(_marker(obs, "BMI", "bmi"), on="encounter_id", how="left")
    attrs = subjects.set_index("subject_key")

    examples: list[ForwardExample] = []
    flow = {
        "n_subjects": int(len(subjects)),
        "n_encounters": int(len(enc)),
        "n_observations": int(len(obs)),
        "dropped_single_encounter": 0,
        "dropped_no_outcome_sbp": 0,
        "dropped_no_prior": 0,
        "n_examples": 0,
    }

    for subject_key, rows in enc.groupby("subject_key", sort=False):
        ordered = rows.sort_values(["encounter_at", "encounter_id"], kind="mergesort")
        if len(ordered) < 2:
            flow["dropped_single_encounter"] += 1
            continue
        with_sbp = ordered.index[ordered["sbp"].notna()]
        if len(with_sbp) == 0:
            flow["dropped_no_outcome_sbp"] += 1
            continue
        outcome_pos = ordered.index.get_loc(with_sbp[-1])
        if isinstance(outcome_pos, slice):
            outcome_pos = outcome_pos.stop - 1
        if outcome_pos == 0:
            flow["dropped_no_prior"] += 1
            continue

        priors = ordered.iloc[:outcome_pos]
        outcome_row = ordered.iloc[outcome_pos]
        feature_dates = tuple(priors["encounter_at"].astype(str))
        prior_sbp = priors["sbp"].dropna()
        prior_bmi = priors["bmi"].dropna()
        person = attrs.loc[subject_key]

        examples.append(
            ForwardExample(
                subject_key=str(subject_key),
                outcome=int(float(outcome_row["sbp"]) >= ELEVATED_SBP_THRESHOLD),
                outcome_date=str(outcome_row["encounter_at"]),
                feature_end_date=feature_dates[-1],
                feature_dates=feature_dates,
                sex=str(person["sex"]),
                age_band=str(person["age_band"]),
                region=str(person["region"]),
                n_prior=int(len(priors)),
                last_encounter_type=str(priors.iloc[-1]["encounter_type"]),
                mean_bmi=float(prior_bmi.mean()) if len(prior_bmi) else None,
                last_sbp=float(prior_sbp.iloc[-1]) if len(prior_sbp) else None,
                missing_bmi_rate=float(priors["bmi"].isna().mean()),
                missing_sbp_rate=float(priors["sbp"].isna().mean()),
            )
        )

    flow["n_examples"] = len(examples)
    return examples, flow


def missingness_summary(tables: dict[str, pd.DataFrame]) -> dict[str, dict[str, float]]:
    """Fraction of nulls per column, plus structural SBP/BMI absence at encounter grain."""
    summary: dict[str, dict[str, float]] = {}
    for name, df in tables.items():
        n = max(len(df), 1)
        summary[name] = {col: float(df[col].isna().mean()) if n else 0.0 for col in df.columns}
    enc_n = max(len(tables["encounters"]), 1)
    obs = tables["observations"]
    sbp_enc = set(obs.loc[obs["marker_code"] == "SBP", "encounter_id"])
    bmi_enc = set(obs.loc[obs["marker_code"] == "BMI", "encounter_id"])
    enc_ids = tables["encounters"]["encounter_id"]
    summary["encounters_structural"] = {
        "no_sbp_row": float((~enc_ids.isin(sbp_enc)).mean()) if enc_n else 0.0,
        "no_bmi_row": float((~enc_ids.isin(bmi_enc)).mean()) if enc_n else 0.0,
    }
    return summary


def allowed_values(contract: Contract, table: str, field: str) -> tuple[str, ...]:
    for spec in contract.tables[table]["fields"]:
        if spec["name"] == field and spec.get("allowed"):
            return tuple(spec["allowed"])
    raise KeyError(f"{table}.{field} has no allowed set")
