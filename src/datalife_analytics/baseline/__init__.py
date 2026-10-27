"""Leakage-safe baseline study over the v0.1.0 synthetic contract (issue #3)."""

from .pipeline import StudyResult, run_study
from .split import LeakageError, assert_no_subject_leakage, assert_no_time_leakage
from .study import STUDY_ID, STUDY_QUESTION

__all__ = [
    "LeakageError",
    "STUDY_ID",
    "STUDY_QUESTION",
    "StudyResult",
    "assert_no_subject_leakage",
    "assert_no_time_leakage",
    "run_study",
]
