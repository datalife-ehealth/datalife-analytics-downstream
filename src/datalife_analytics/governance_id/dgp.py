"""Parameterized synthetic cohorts for governance and missingness experiments."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike

from .regimes import (
    FloatArray,
    RegimeM0Data,
    RegimeM1Data,
    RegimeM2Data,
    RegimeM3Data,
)


def _validated_vector(value: ArrayLike, name: str) -> FloatArray:
    """Copy a finite float vector and make it read-only."""
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    array.setflags(write=False)
    return array


def _probability(value: float, name: str) -> float:
    """Validate a probability-valued configuration scalar."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite probability")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite probability") from exc
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


@dataclass(frozen=True)
class GovernanceDGPConfig:
    """Configuration for a governed cohort data-generating process.

    Attributes:
        n_patients: Number of patients in the full reference cohort.
        reason_names: Mutually exclusive missing-reason names.
        m_k: Reference missing-loss mean by reason.
        delta_k: Sensitivity radius by reason.
        baseline_risk_shift: Selection-logit coefficient on baseline risk.
        confusion_matrix: True-to-logged reason transition matrix; defaults to identity.
        epsilon_upper: Upper bound on aggregate audit misclassification.
        structural_zero_prob: Probability that policy eligibility forces non-observation.
    """

    n_patients: int
    reason_names: tuple[str, ...]
    m_k: FloatArray
    delta_k: FloatArray
    baseline_risk_shift: float = 0.0
    confusion_matrix: FloatArray | None = None
    epsilon_upper: float = 0.0
    structural_zero_prob: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.n_patients, Integral) or isinstance(self.n_patients, bool):
            raise ValueError("n_patients must be a positive integer")
        if self.n_patients <= 0:
            raise ValueError("n_patients must be a positive integer")
        object.__setattr__(self, "n_patients", int(self.n_patients))

        names = tuple(self.reason_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("reason_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("reason_names must be unique")
        object.__setattr__(self, "reason_names", names)

        means = _validated_vector(self.m_k, "m_k")
        radii = _validated_vector(self.delta_k, "delta_k")
        if means.size != len(names) or radii.size != len(names):
            raise ValueError("m_k and delta_k must match reason_names")
        if np.any((means < 0.0) | (means > 1.0)):
            raise ValueError("m_k values must be in [0, 1]")
        if np.any((radii < 0.0) | (radii > 1.0)):
            raise ValueError("delta_k values must be in [0, 1]")
        object.__setattr__(self, "m_k", means)
        object.__setattr__(self, "delta_k", radii)

        try:
            risk_shift = float(self.baseline_risk_shift)
        except (TypeError, ValueError) as exc:
            raise ValueError("baseline_risk_shift must be finite") from exc
        if not np.isfinite(risk_shift):
            raise ValueError("baseline_risk_shift must be finite")
        object.__setattr__(self, "baseline_risk_shift", risk_shift)
        object.__setattr__(self, "epsilon_upper", _probability(self.epsilon_upper, "epsilon_upper"))
        object.__setattr__(
            self,
            "structural_zero_prob",
            _probability(self.structural_zero_prob, "structural_zero_prob"),
        )

        if self.confusion_matrix is not None:
            try:
                matrix = np.array(self.confusion_matrix, dtype=np.float64, copy=True)
            except (TypeError, ValueError) as exc:
                raise ValueError("confusion_matrix must be a numeric square matrix") from exc
            if matrix.shape != (len(names), len(names)):
                raise ValueError("confusion_matrix must have shape (n_reasons, n_reasons)")
            if not np.all(np.isfinite(matrix)) or np.any((matrix < 0.0) | (matrix > 1.0)):
                raise ValueError("confusion_matrix entries must be in [0, 1]")
            if not np.allclose(matrix.sum(axis=1), 1.0, rtol=1e-9, atol=1e-9):
                raise ValueError("confusion_matrix rows must sum to one")
            matrix.setflags(write=False)
            object.__setattr__(self, "confusion_matrix", matrix)


@dataclass(frozen=True)
class CohortBundle:
    """Oracle evaluation truth and four progressively disclosed cohort views.

    Attributes:
        oracle_true_psi: Full-cohort mean loss, reserved for internal evaluation.
        true_p_k: True missing-reason shares over the full cohort.
        m0: M0 observed-loss-only view.
        m1: M1 cohort observation summaries.
        m2: M2 reason-level aggregate sensitivity view.
        m3: M3 aggregate view plus full-cohort baseline covariates and observation flags.
    """

    oracle_true_psi: float
    true_p_k: FloatArray
    m0: RegimeM0Data
    m1: RegimeM1Data
    m2: RegimeM2Data
    m3: RegimeM3Data

    def __post_init__(self) -> None:
        try:
            oracle = float(self.oracle_true_psi)
        except (TypeError, ValueError) as exc:
            raise ValueError("oracle_true_psi must be in [0, 1]") from exc
        if not np.isfinite(oracle) or not 0.0 <= oracle <= 1.0:
            raise ValueError("oracle_true_psi must be in [0, 1]")
        true_p_k = _validated_vector(self.true_p_k, "true_p_k")
        if true_p_k.size != len(self.m2.reason_names) or np.any(
            (true_p_k < 0.0) | (true_p_k > 1.0)
        ):
            raise ValueError("true_p_k must match reason_names and be in [0, 1]")
        if not np.isclose(true_p_k.sum(), 1.0 - self.m1.q, rtol=1e-9, atol=1e-9):
            raise ValueError("true_p_k must sum to the full-cohort missing rate")
        if self.m1.total_n != self.m3.total_n or self.m1.total_n != self.m2.total_n:
            raise ValueError("all regime views must describe the same cohort")
        if not np.array_equal(self.m0.observed_losses, self.m1.observed_losses):
            raise ValueError("M0 and M1 must carry the same observed losses")
        if not np.array_equal(
            self.m1.observed_losses, self.m2.observed_losses
        ) or not np.array_equal(self.m1.observed_losses, self.m3.observed_losses):
            raise ValueError("all regime views must carry the same observed losses")
        object.__setattr__(self, "oracle_true_psi", oracle)
        object.__setattr__(self, "true_p_k", true_p_k)


def generate_governed_cohort(config: GovernanceDGPConfig, seed: int = 42) -> CohortBundle:
    """Generate reproducible losses, missingness, audited reasons, and compliant views.

    Baseline risk shifts the observation log-odds, while reason-specific loss means
    make that selection informative when reason composition varies with risk. A
    zero shift and zero structural-zero probability produce MCAR observation.

    Args:
        config: Validated cohort and sensitivity configuration.
        seed: Non-negative seed for NumPy's generator.

    Returns:
        Oracle truth and M0-M3 views for one synthetic reference cohort.
    """
    if not isinstance(config, GovernanceDGPConfig):
        raise TypeError("config must be a GovernanceDGPConfig")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    rng = np.random.default_rng(int(seed))
    n_patients = config.n_patients
    n_reasons = len(config.reason_names)
    matrix = config.confusion_matrix
    if matrix is None:
        matrix = np.eye(n_reasons, dtype=np.float64)

    covariates = rng.normal(size=(n_patients, 3))
    risk = covariates[:, 0]
    reason_slopes = np.linspace(-0.45, 0.45, n_reasons, dtype=np.float64)
    reason_logits = risk[:, None] * reason_slopes[None, :]
    reason_logits -= reason_logits.max(axis=1, keepdims=True)
    reason_probabilities = np.exp(reason_logits)
    reason_probabilities /= reason_probabilities.sum(axis=1, keepdims=True)
    uniforms = rng.random(n_patients)
    true_reason = np.sum(
        uniforms[:, None] > np.cumsum(reason_probabilities, axis=1), axis=1
    ).astype(np.int64)
    true_reason = np.minimum(true_reason, n_reasons - 1)

    lower = np.maximum(0.0, config.m_k - config.delta_k)
    upper = np.minimum(1.0, config.m_k + config.delta_k)
    symmetric_radius = np.minimum(config.m_k - lower, upper - config.m_k)
    losses = (
        config.m_k[true_reason]
        + rng.uniform(-1.0, 1.0, size=n_patients) * symmetric_radius[true_reason]
    )
    losses = np.clip(losses, lower[true_reason], upper[true_reason])

    eligible_structural_zero = rng.random(n_patients) < config.structural_zero_prob
    observation_logits = np.log(0.75 / 0.25) - config.baseline_risk_shift * risk
    observation_probability = 1.0 / (1.0 + np.exp(-np.clip(observation_logits, -30.0, 30.0)))
    observed = (~eligible_structural_zero) & (rng.random(n_patients) < observation_probability)
    missing = ~observed

    logged_reason = np.full(n_patients, -1, dtype=np.int64)
    for reason_index in range(n_reasons):
        reason_rows = np.flatnonzero(missing & (true_reason == reason_index))
        if reason_rows.size:
            logged_reason[reason_rows] = rng.choice(
                n_reasons, size=reason_rows.size, p=matrix[reason_index]
            )

    observed_losses = losses[observed]
    q = float(observed.mean())
    a = float(observed_losses.mean()) if observed_losses.size else 0.0
    n_strata = min(4, n_patients)
    order = np.argsort(risk, kind="stable")
    stratum_index = np.empty(n_patients, dtype=np.int64)
    stratum_index[order] = np.minimum(
        np.arange(n_patients, dtype=np.int64) * n_strata // n_patients, n_strata - 1
    )
    weights = np.empty(n_strata, dtype=np.float64)
    obs_rates = np.empty(n_strata, dtype=np.float64)
    obs_losses = np.empty(n_strata, dtype=np.float64)
    for index in range(n_strata):
        rows = stratum_index == index
        weights[index] = float(rows.mean())
        obs_rates[index] = float(observed[rows].mean())
        stratum_losses = losses[rows & observed]
        obs_losses[index] = float(stratum_losses.mean()) if stratum_losses.size else 0.0

    confusion = matrix
    tilde_p = (
        np.bincount(logged_reason[missing], minlength=n_reasons).astype(np.float64) / n_patients
    )
    true_p_k = (
        np.bincount(true_reason[missing], minlength=n_reasons).astype(np.float64) / n_patients
    )
    m0 = RegimeM0Data(observed_losses=observed_losses)
    m1 = RegimeM1Data(
        observed_losses=observed_losses,
        total_n=n_patients,
        q=q,
        a=a,
        stratum_weights=weights,
        stratum_obs_rates=obs_rates,
        stratum_obs_losses=obs_losses,
    )
    m2 = RegimeM2Data(
        observed_losses=observed_losses,
        total_n=n_patients,
        q=q,
        a=a,
        reason_names=config.reason_names,
        tilde_p=tilde_p,
        m_k=config.m_k,
        delta_k=config.delta_k,
        epsilon=config.epsilon_upper,
        class_acc_lower=np.diag(confusion),
        stratum_weights=weights,
        stratum_obs_rates=obs_rates,
        stratum_obs_losses=obs_losses,
    )
    m3 = RegimeM3Data(
        observed_losses=observed_losses,
        total_n=n_patients,
        q=q,
        a=a,
        reason_names=config.reason_names,
        tilde_p=tilde_p,
        m_k=config.m_k,
        delta_k=config.delta_k,
        epsilon=config.epsilon_upper,
        class_acc_lower=np.diag(confusion),
        stratum_weights=weights,
        stratum_obs_rates=obs_rates,
        stratum_obs_losses=obs_losses,
        H=covariates,
        R=observed.astype(np.bool_),
    )
    return CohortBundle(
        oracle_true_psi=float(losses.mean()),
        true_p_k=true_p_k,
        m0=m0,
        m1=m1,
        m2=m2,
        m3=m3,
    )
