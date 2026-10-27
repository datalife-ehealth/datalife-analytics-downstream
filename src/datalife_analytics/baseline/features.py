"""Train-only feature encoding. Category levels come from the contract, not the split."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from datalife_analytics.contract import load_contract

from .study import ForwardExample, allowed_values


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


@dataclass
class FeatureEncoder:
    median_bmi: float = 0.0
    median_sbp: float = 0.0
    feature_names: tuple[str, ...] = field(default_factory=tuple)

    def _names(self) -> tuple[str, ...]:
        contract = load_contract()
        names = ["intercept"]
        for table, column, prefix in (
            ("subjects", "sex", "sex"),
            ("subjects", "age_band", "age_band"),
            ("subjects", "region", "region"),
            ("encounters", "encounter_type", "last_encounter"),
        ):
            levels = allowed_values(contract, table, column)[1:]
            names += [f"{prefix}={level}" for level in levels]
        names += ["n_prior", "mean_bmi", "last_sbp", "missing_bmi_rate", "missing_sbp_rate"]
        return tuple(names)

    def fit(self, examples: list[ForwardExample]) -> FeatureEncoder:
        self.median_bmi = _median([ex.mean_bmi for ex in examples if ex.mean_bmi is not None])
        self.median_sbp = _median([ex.last_sbp for ex in examples if ex.last_sbp is not None])
        self.feature_names = self._names()
        return self

    def transform(self, examples: list[ForwardExample]) -> np.ndarray:
        contract = load_contract()
        sex = allowed_values(contract, "subjects", "sex")[1:]
        age = allowed_values(contract, "subjects", "age_band")[1:]
        region = allowed_values(contract, "subjects", "region")[1:]
        etype = allowed_values(contract, "encounters", "encounter_type")[1:]
        rows: list[list[float]] = []
        for example in examples:
            row = [1.0]
            row += [1.0 if example.sex == level else 0.0 for level in sex]
            row += [1.0 if example.age_band == level else 0.0 for level in age]
            row += [1.0 if example.region == level else 0.0 for level in region]
            row += [1.0 if example.last_encounter_type == level else 0.0 for level in etype]
            row += [
                float(example.n_prior),
                float(self.median_bmi if example.mean_bmi is None else example.mean_bmi),
                float(self.median_sbp if example.last_sbp is None else example.last_sbp),
                float(example.missing_bmi_rate),
                float(example.missing_sbp_rate),
            ]
            rows.append(row)
        if not rows:
            return np.zeros((0, len(self.feature_names or self._names())))
        return np.asarray(rows, dtype=float)

    def labels(self, examples: list[ForwardExample]) -> np.ndarray:
        return np.asarray([example.outcome for example in examples], dtype=float)
