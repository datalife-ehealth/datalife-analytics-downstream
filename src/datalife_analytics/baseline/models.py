"""Naive comparator and a ridge-regularized logistic baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sigmoid(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


@dataclass
class NaivePrevalence:
    """Always predicts the training positive rate. Hard label is the majority class."""

    prevalence: float = 0.0
    majority: int = 0
    name: str = "naive_prevalence"

    def fit(self, labels: np.ndarray) -> NaivePrevalence:
        if labels.size == 0:
            self.prevalence = 0.0
            self.majority = 0
            return self
        self.prevalence = float(labels.mean())
        self.majority = 1 if self.prevalence >= 0.5 else 0
        return self

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        return np.full(len(rows), self.prevalence, dtype=float)

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return np.full(len(rows), self.majority, dtype=int)


@dataclass
class RidgeLogistic:
    """L2-regularized logistic regression fit by iteratively reweighted least squares."""

    l2: float = 1.0
    max_iter: int = 25
    weights: np.ndarray | None = None
    name: str = "ridge_logistic"

    def fit(self, rows: np.ndarray, labels: np.ndarray) -> RidgeLogistic:
        if rows.size == 0:
            self.weights = np.zeros(0)
            return self
        n_features = rows.shape[1]
        weights = np.zeros(n_features)
        penalty = np.eye(n_features)
        penalty[0, 0] = 0.0
        for _ in range(self.max_iter):
            probs = np.clip(_sigmoid(rows @ weights), 1e-6, 1.0 - 1e-6)
            diag = probs * (1.0 - probs)
            working = (rows @ weights) + (labels - probs) / diag
            xtw = rows.T * diag
            gram = xtw @ rows + self.l2 * penalty
            rhs = xtw @ working
            try:
                weights = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                weights, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
        self.weights = weights
        return self

    def predict_proba(self, rows: np.ndarray) -> np.ndarray:
        if self.weights is None or rows.size == 0:
            return np.full(len(rows), 0.5, dtype=float)
        return _sigmoid(rows @ self.weights)

    def predict(self, rows: np.ndarray) -> np.ndarray:
        return (self.predict_proba(rows) >= 0.5).astype(int)
