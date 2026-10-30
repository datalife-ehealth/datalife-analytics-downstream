"""Validate, split, fit, evaluate, and build the report payload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from datalife_analytics.contract import load_contract, validate

from . import metrics as metrics_mod
from .features import FeatureEncoder
from .lineage import lineage
from .models import NaivePrevalence, RidgeLogistic
from .split import (
    assert_no_cross_split_time_leakage,
    assert_no_subject_leakage,
    assert_no_time_leakage,
    assert_no_time_leakage_against_tables,
    subject_split,
)
from .study import build_forward_examples, missingness_summary


@dataclass
class StudyResult:
    payload: dict[str, Any]


def run_study(
    tables: dict[str, pd.DataFrame],
    *,
    split_seed: int,
    train_fraction: float = 0.7,
    generator_seed: int | None = None,
    n_subjects: int | None = None,
    input_path: str | None = None,
    contract_version: str | None = None,
) -> StudyResult:
    contract = load_contract(contract_version) if contract_version else load_contract()
    validate(tables, contract)
    examples, flow = build_forward_examples(tables, contract)
    # Halt on leaked dates before the split or any model sees the examples.
    assert_no_time_leakage(examples)
    assert_no_time_leakage_against_tables(examples, tables)
    train, test = subject_split(examples, seed=split_seed, train_fraction=train_fraction)
    assert_no_subject_leakage(train, test)
    assert_no_time_leakage(train)
    assert_no_time_leakage(test)
    assert_no_cross_split_time_leakage(train, test)

    encoder = FeatureEncoder().fit(train)
    x_train = encoder.transform(train)
    y_train = encoder.labels(train)
    x_test = encoder.transform(test)
    y_test = encoder.labels(test)

    naive = NaivePrevalence().fit(y_train)
    logistic = RidgeLogistic().fit(x_train, y_train)

    def evaluate(model, rows, labels, examples_side):
        if labels.size == 0:
            return {"n": 0, "note": "empty partition; metrics withheld"}
        scores = model.predict_proba(rows)
        preds = model.predict(rows)
        summary = metrics_mod.summarize(labels, preds, scores)
        summary["uncertainty"] = metrics_mod.bootstrap_ci(labels, scores, preds, seed=split_seed)
        summary["subgroups"] = metrics_mod.subgroup_metrics(examples_side, labels, preds, scores)
        return summary

    payload: dict[str, Any] = {
        "lineage": lineage(
            split_seed=split_seed,
            train_fraction=train_fraction,
            generator_seed=generator_seed,
            n_subjects=n_subjects,
            input_path=input_path,
            contract_version=contract.version,
        ),
        "cohort_flow": flow,
        "split": {
            "n_train": len(train),
            "n_test": len(test),
            "train_subjects": len({ex.subject_key for ex in train}),
            "test_subjects": len({ex.subject_key for ex in test}),
        },
        "missingness": missingness_summary(tables),
        "features": list(encoder.feature_names),
        "impute": {"median_bmi": encoder.median_bmi, "median_sbp": encoder.median_sbp},
        "models": {
            naive.name: {
                "train": evaluate(naive, x_train, y_train, train),
                "test": evaluate(naive, x_test, y_test, test),
                "prevalence": naive.prevalence,
                "majority_class": naive.majority,
            },
            logistic.name: {
                "train": evaluate(logistic, x_train, y_train, train),
                "test": evaluate(logistic, x_test, y_test, test),
            },
        },
        "limitations": [
            "Rows are synthetic; distributions are invented and not derived from patients.",
            "The 140 mmHg cut is a labelling rule for this study, not a diagnostic claim.",
            "On this generator, prior SBP is highly informative by construction; a strong "
            "test AUC is not evidence of a real clinical signal.",
            "Ridge logistic is a transparent baseline, not a candidate for deployment.",
            "Subject-level split blocks subject leakage; features are earlier than the outcome.",
            "Imputation medians are fit on train only. Tiny subgroup cells report n only.",
            "Calibration (ECE) on small test sets is noisy; treat it as a diagnostic.",
            "No causal, fairness, or clinical-effectiveness claim is supported by this run.",
        ],
    }
    return StudyResult(payload=payload)
