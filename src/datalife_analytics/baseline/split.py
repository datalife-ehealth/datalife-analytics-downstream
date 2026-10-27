"""Subject-aware split and leakage detectors."""

from __future__ import annotations

import random
from collections.abc import Sequence

from .study import ForwardExample


class LeakageError(Exception):
    """Raised when a split or example set violates the leakage contract."""

    def __init__(self, issues: Sequence[str]) -> None:
        self.issues = list(issues)
        super().__init__("; ".join(self.issues))


def subject_split(
    examples: Sequence[ForwardExample],
    *,
    seed: int,
    train_fraction: float = 0.7,
) -> tuple[list[ForwardExample], list[ForwardExample]]:
    """Partition examples by subject. No subject appears on both sides."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0.0, 1.0)")
    if not examples:
        return [], []

    subjects = sorted({example.subject_key for example in examples})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    if len(subjects) == 1:
        train_subjects = set(subjects)
        test_subjects: set[str] = set()
    else:
        cut = max(1, min(len(subjects) - 1, round(len(subjects) * train_fraction)))
        train_subjects = set(subjects[:cut])
        test_subjects = set(subjects[cut:])

    train = [ex for ex in examples if ex.subject_key in train_subjects]
    test = [ex for ex in examples if ex.subject_key in test_subjects]
    assert_no_subject_leakage(train, test)
    assert_no_time_leakage(train)
    assert_no_time_leakage(test)
    return train, test


def assert_no_subject_leakage(
    train: Sequence[ForwardExample],
    test: Sequence[ForwardExample],
) -> None:
    overlap = sorted({ex.subject_key for ex in train} & {ex.subject_key for ex in test})
    if overlap:
        raise LeakageError([f"subject leakage: {len(overlap)} subject(s) in both train and test"])


def assert_no_time_leakage(examples: Sequence[ForwardExample]) -> None:
    issues: list[str] = []
    for example in examples:
        if any(date >= example.outcome_date for date in example.feature_dates):
            issues.append(
                f"time leakage: subject {example.subject_key} has a feature date "
                "on or after its outcome date"
            )
        elif example.feature_end_date >= example.outcome_date:
            issues.append(
                f"time leakage: subject {example.subject_key} feature window is not before outcome"
            )
    if issues:
        raise LeakageError(issues)


def assert_no_cross_split_time_leakage(
    train: Sequence[ForwardExample],
    test: Sequence[ForwardExample],
) -> None:
    """If a subject is on both sides, every train date must precede every test date."""
    train_by: dict[str, list[ForwardExample]] = {}
    test_by: dict[str, list[ForwardExample]] = {}
    for example in train:
        train_by.setdefault(example.subject_key, []).append(example)
    for example in test:
        test_by.setdefault(example.subject_key, []).append(example)

    issues: list[str] = []
    for subject_key in sorted(set(train_by) & set(test_by)):
        train_latest = max(ex.outcome_date for ex in train_by[subject_key])
        test_earliest = min(
            min(ex.feature_dates) if ex.feature_dates else ex.outcome_date
            for ex in test_by[subject_key]
        )
        if test_earliest < train_latest:
            issues.append(
                f"time leakage: subject {subject_key} has a test feature before a train outcome"
            )
    if issues:
        raise LeakageError(issues)
