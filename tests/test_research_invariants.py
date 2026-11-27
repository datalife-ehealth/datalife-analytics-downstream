"""High-signal scientific invariants for the Experiment 0 research baseline."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from datalife_analytics.baseline.pipeline import run_study
from datalife_analytics.baseline.split import (
    LeakageError,
    assert_no_time_leakage,
    subject_split,
)
from datalife_analytics.baseline.study import build_forward_examples
from datalife_analytics.generator import GeneratorConfig, generate


@pytest.mark.invariant
def test_subject_held_out_split_has_zero_subject_overlap(small_tables):
    examples, _ = build_forward_examples(small_tables)
    train, test = subject_split(examples, seed=17)

    train_subjects = {example.subject_key for example in train}
    test_subjects = {example.subject_key for example in test}
    assert train_subjects.isdisjoint(test_subjects)


@pytest.mark.invariant
def test_features_are_strictly_before_outcome_and_leaks_are_rejected(small_tables):
    examples, _ = build_forward_examples(small_tables)
    assert examples

    for example in examples:
        assert all(feature_date < example.outcome_date for feature_date in example.feature_dates)
        assert example.feature_end_date < example.outcome_date
    assert_no_time_leakage(examples)

    leaked = replace(
        examples[0],
        feature_dates=(*examples[0].feature_dates, examples[0].outcome_date),
        feature_end_date=examples[0].outcome_date,
    )
    with pytest.raises(LeakageError, match="time leakage"):
        assert_no_time_leakage([leaked])


@pytest.mark.invariant
def test_same_seed_reproduces_identical_baseline_metrics():
    config = GeneratorConfig(seed=20261006, n_subjects=60)
    first = run_study(
        generate(config),
        split_seed=17,
        generator_seed=config.seed,
        n_subjects=config.n_subjects,
    )
    second = run_study(
        generate(config),
        split_seed=17,
        generator_seed=config.seed,
        n_subjects=config.n_subjects,
    )

    first_metrics = json.dumps(first.payload["models"], sort_keys=True, separators=(",", ":"))
    second_metrics = json.dumps(second.payload["models"], sort_keys=True, separators=(",", ":"))
    assert first_metrics == second_metrics
