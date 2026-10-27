"""Leakage detectors: subject overlap, future features, schema, missing columns."""

from __future__ import annotations

import pytest

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
    from datalife_analytics.baseline.pipeline import run_study

    tables_copy["subjects"] = tables_copy["subjects"].drop(columns=["region"])
    with pytest.raises(ContractError, match="missing column"):
        run_study(tables_copy, split_seed=1)


def test_generated_tables_match_contract():
    tables = generate(GeneratorConfig(seed=3, n_subjects=20))
    validate(tables)
