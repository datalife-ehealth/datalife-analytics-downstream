"""Discrimination, calibration, uncertainty, and subgroup summaries."""

from __future__ import annotations

import math
import random
from collections import defaultdict

import numpy as np

from .study import ForwardExample

MIN_SUBGROUP = 8


def _finite(value: float) -> float | None:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    return float((labels == preds).mean())


def balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    pos = preds[labels == 1]
    neg = preds[labels == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float((pos.mean() + (1.0 - neg.mean())) / 2.0)


def brier(labels: np.ndarray, scores: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    return float(np.mean((scores - labels) ** 2))


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    n_pos = int(labels.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranked_labels = labels[order]
    ranks = np.empty(labels.size, dtype=float)
    i = 0
    while i < labels.size:
        j = i
        while j < labels.size and scores[order[j]] == scores[order[i]]:
            j += 1
        ranks[i:j] = (i + 1 + j) / 2.0
        i = j
    sum_pos = float(ranks[ranked_labels == 1].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def expected_calibration_error(labels: np.ndarray, scores: np.ndarray, bins: int = 5) -> float:
    if labels.size == 0:
        return float("nan")
    ece = 0.0
    n = labels.size
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        if b == bins - 1:
            mask = (scores >= lo) & (scores <= hi)
        else:
            mask = (scores >= lo) & (scores < hi)
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(float(labels[mask].mean()) - float(scores[mask].mean()))
    return float(ece)


def summarize(
    labels: np.ndarray, preds: np.ndarray, scores: np.ndarray
) -> dict[str, float | int | None]:
    y = labels.astype(int)
    return {
        "n": int(y.size),
        "n_positive": int(y.sum()),
        "n_negative": int(y.size - y.sum()),
        "accuracy": _finite(accuracy(y, preds)),
        "balanced_accuracy": _finite(balanced_accuracy(y, preds)),
        "brier": _finite(brier(y, scores)),
        "roc_auc": _finite(roc_auc(y, scores)),
        "ece": _finite(expected_calibration_error(y, scores)),
    }


def bootstrap_ci(
    labels: np.ndarray,
    scores: np.ndarray,
    preds: np.ndarray,
    *,
    seed: int,
    draws: int = 200,
) -> dict[str, dict[str, float | None]]:
    rng = random.Random(seed)
    n = int(labels.size)
    empty = {"lo": None, "hi": None}
    if n == 0:
        return {"accuracy": empty, "roc_auc": empty}
    accs: list[float] = []
    aucs: list[float] = []
    for _ in range(draws):
        idx = np.array([rng.randrange(n) for _ in range(n)])
        y, s, p = labels[idx], scores[idx], preds[idx]
        accs.append(accuracy(y, p))
        auc = roc_auc(y, s)
        if not math.isnan(auc):
            aucs.append(auc)

    def interval(samples: list[float]) -> dict[str, float | None]:
        if not samples:
            return empty
        ordered = sorted(samples)
        lo = ordered[int(0.025 * (len(ordered) - 1))]
        hi = ordered[int(0.975 * (len(ordered) - 1))]
        return {"lo": lo, "hi": hi}

    return {"accuracy": interval(accs), "roc_auc": interval(aucs)}


def subgroup_metrics(
    examples: list[ForwardExample],
    labels: np.ndarray,
    preds: np.ndarray,
    scores: np.ndarray,
) -> dict[str, list[dict[str, object]]]:
    def by(attr: str) -> list[dict[str, object]]:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, example in enumerate(examples):
            groups[str(getattr(example, attr))].append(index)
        rows: list[dict[str, object]] = []
        for key in sorted(groups):
            idx = np.array(groups[key])
            entry: dict[str, object] = {"level": key, "n": int(idx.size)}
            if idx.size >= MIN_SUBGROUP:
                entry["accuracy"] = _finite(accuracy(labels[idx], preds[idx]))
                entry["roc_auc"] = _finite(roc_auc(labels[idx], scores[idx]))
            else:
                entry["accuracy"] = None
                entry["roc_auc"] = None
                entry["note"] = f"n < {MIN_SUBGROUP}; metric withheld"
            rows.append(entry)
        return rows

    return {"age_band": by("age_band"), "region": by("region"), "sex": by("sex")}
