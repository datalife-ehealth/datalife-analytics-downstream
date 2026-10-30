"""Leakage detectors: subject overlap, future features, schema, missing columns."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from datalife_analytics.baseline.models import RidgeLogistic
from datalife_analytics.baseline.pipeline import run_study
from datalife_analytics.baseline.split import (
    LeakageError,
    assert_no_cross_split_time_leakage,
    assert_no_subject_leakage,
    assert_no_time_leakage,
    subject_split,
)
from datalife_analytics.baseline.study import ForwardExample, build_forward_examples
from datalife_analytics.contract import ContractError, validate
from datalife_analytics.generator import GeneratorConfig, generate


def _example(
    subject: str,
    *,
    outcome_date: str,
    feature_dates: tuple[str, ...],
    outcome: int = 0,
) -> ForwardExample:
    return ForwardExample(
        subject_key=subject,
        outcome=outcome,
        outcome_date=outcome_date,
        feature_end_date=feature_dates[-1],
        feature_dates=feature_dates,
        sex="F",
        age_band="45-59",
        region="R01",
        n_prior=len(feature_dates),
        last_encounter_type="routine",
        mean_bmi=24.0,
        last_sbp=120.0,
        missing_bmi_rate=0.0,
        missing_sbp_rate=0.0,
    )


def test_subject_split_has_no_overlapping_subjects(small_tables):
    examples, _ = build_forward_examples(small_tables)
    train, test = subject_split(examples, seed=3)
    train_subjects = {ex.subject_key for ex in train}
    test_subjects = {ex.subject_key for ex in test}
    assert train_subjects.isdisjoint(test_subjects)
    assert train_subjects | test_subjects == {ex.subject_key for ex in examples}


def test_subject_split_is_deterministic(small_tables):
    examples, _ = build_forward_examples(small_tables)
    a_train, a_test = subject_split(examples, seed=11)
    b_train, b_test = subject_split(examples, seed=11)
    assert [ex.subject_key for ex in a_train] == [ex.subject_key for ex in b_train]
    assert [ex.subject_key for ex in a_test] == [ex.subject_key for ex in b_test]


def test_subject_leakage_is_detected():
    shared = _example(
        "syn-aaaaaaaaaaaaaaaa",
        outcome_date="2024-06-01T09:00:00Z",
        feature_dates=("2024-01-01T09:00:00Z",),
    )
    with pytest.raises(LeakageError, match="subject leakage"):
        assert_no_subject_leakage([shared], [shared])


def test_time_leakage_is_detected_when_feature_is_not_before_outcome():
    bad = _example(
        "syn-bbbbbbbbbbbbbbbb",
        outcome_date="2024-03-01T09:00:00Z",
        feature_dates=("2024-01-01T09:00:00Z", "2024-03-01T09:00:00Z"),
    )
    with pytest.raises(LeakageError, match="time leakage"):
        assert_no_time_leakage([bad])


def test_cross_split_time_leakage_is_detected():
    train = [
        _example(
            "syn-cccccccccccccccc",
            outcome_date="2024-09-01T09:00:00Z",
            feature_dates=("2024-08-01T09:00:00Z",),
        )
    ]
    test = [
        _example(
            "syn-cccccccccccccccc",
            outcome_date="2024-03-01T09:00:00Z",
            feature_dates=("2024-01-01T09:00:00Z",),
        )
    ]
    with pytest.raises(LeakageError, match="time leakage"):
        assert_no_cross_split_time_leakage(train, test)


def test_forward_examples_never_use_future_features(small_tables):
    examples, flow = build_forward_examples(small_tables)
    assert flow["n_examples"] == len(examples)
    assert examples
    assert_no_time_leakage(examples)


def test_schema_mismatch_is_rejected(tables_copy):
    tables_copy["subjects"]["email"] = "x"
    with pytest.raises(ContractError, match="forbidden identifier-like column"):
        validate(tables_copy)


def test_missing_columns_are_rejected(tables_copy):
    tables_copy["encounters"] = tables_copy["encounters"].drop(columns=["encounter_type"])
    with pytest.raises(ContractError, match="missing column"):
        validate(tables_copy)


def test_pipeline_validate_rejects_mismatch_before_fit(tables_copy):
    tables_copy["subjects"] = tables_copy["subjects"].drop(columns=["region"])
    with pytest.raises(ContractError, match="missing column"):
        run_study(tables_copy, split_seed=1)


def _next_id(series: pd.Series, prefix: str, width: int) -> str:
    nums = series.astype(str).str.removeprefix(prefix).astype(int)
    return f"{prefix}{int(nums.max()) + 1:0{width}d}"


def _inject_future_bp_encounter(tables: dict[str, pd.DataFrame], subject_key: str, ts: str) -> None:
    """Append a contract-valid future encounter with paired SBP/DBP."""
    enc_id = _next_id(tables["encounters"]["encounter_id"], "enc-", 7)
    obs_n = int(tables["observations"]["observation_id"].str.removeprefix("obs-").astype(int).max())
    encounter = {
        "encounter_id": enc_id,
        "subject_key": subject_key,
        "encounter_at": ts,
        "encounter_type": "routine",
    }
    sbp = {
        "observation_id": f"obs-{obs_n + 1:08d}",
        "subject_key": subject_key,
        "encounter_id": enc_id,
        "observed_at": ts,
        "marker_code": "SBP",
        "value": 150.0,
        "unit": "mmHg",
        "source_kind": "json",
        "missing_reason": None,
    }
    dbp = {**sbp, "observation_id": f"obs-{obs_n + 2:08d}", "marker_code": "DBP", "value": 90.0}
    tables["encounters"] = pd.concat(
        [tables["encounters"], pd.DataFrame([encounter])], ignore_index=True
    )
    tables["observations"] = pd.concat(
        [tables["observations"], pd.DataFrame([sbp, dbp])], ignore_index=True
    )
    tables["encounters"] = (
        tables["encounters"]
        .sort_values(["subject_key", "encounter_at", "encounter_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    tables["observations"] = (
        tables["observations"]
        .sort_values(
            ["subject_key", "observed_at", "marker_code", "observation_id"], kind="mergesort"
        )
        .reset_index(drop=True)
    )


def test_run_study_halts_before_fit_when_future_encounter_leaks_into_features(
    tables_copy, monkeypatch
):
    """Integration: a future-dated row used as a covariate must not reach the model."""
    original = {name: frame.copy() for name, frame in tables_copy.items()}
    examples, flow = build_forward_examples(original)
    target = next(ex for ex in examples if ex.feature_dates)
    future_ts = "2029-12-17T10:00:00Z"
    _inject_future_bp_encounter(tables_copy, target.subject_key, future_ts)
    validate(tables_copy)

    leaked = replace(
        target,
        feature_dates=(*target.feature_dates, future_ts),
        feature_end_date=future_ts,
    )
    rest = [ex for ex in examples if ex.subject_key != target.subject_key]

    def leaky_build(*_args, **_kwargs):
        # Regression stand-in: keep the original (earlier) outcome but fold the
        # injected future encounter into the covariates.
        return [leaked, *rest], flow

    monkeypatch.setattr("datalife_analytics.baseline.pipeline.build_forward_examples", leaky_build)

    fits = {"n": 0}
    original_fit = RidgeLogistic.fit

    def counting_fit(self, *args, **kwargs):
        fits["n"] += 1
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(RidgeLogistic, "fit", counting_fit)

    with pytest.raises(LeakageError, match="time leakage"):
        run_study(tables_copy, split_seed=1)
    assert fits["n"] == 0


def test_run_study_treats_injected_future_encounter_as_outcome_not_a_feature(tables_copy):
    # A later row on the timeline becomes the outcome. That is forward-looking, not
    # leakage, so run_study must still complete.
    examples, _ = build_forward_examples(
        {name: frame.copy() for name, frame in tables_copy.items()}
    )
    target = next(ex for ex in examples if ex.feature_dates)
    future_ts = "2029-12-17T10:00:00Z"
    _inject_future_bp_encounter(tables_copy, target.subject_key, future_ts)
    rebuilt, _ = build_forward_examples(tables_copy)
    leaked = next(ex for ex in rebuilt if ex.subject_key == target.subject_key)
    assert leaked.outcome_date == future_ts
    assert future_ts not in leaked.feature_dates
    result = run_study(tables_copy, split_seed=1)
    assert "ridge_logistic" in result.payload["models"]


def test_generated_tables_match_contract():
    tables = generate(GeneratorConfig(seed=3, n_subjects=20))
    validate(tables)
