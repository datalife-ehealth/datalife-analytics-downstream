"""Validated data contracts for nested information-disclosure regimes."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _float_array(
    value: ArrayLike,
    name: str,
    *,
    ndim: int,
    bounds: tuple[float, float] | None = None,
) -> FloatArray:
    """Return a finite, validated, read-only float64 copy of an array."""
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if bounds is not None and np.any((array < bounds[0]) | (array > bounds[1])):
        raise ValueError(f"{name} values must be in [{bounds[0]}, {bounds[1]}]")
    array.setflags(write=False)
    return array


def _probability(value: float, name: str) -> float:
    """Validate and normalize a scalar probability."""
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
class RegimeM0Data:
    """Observed losses available under the M0 disclosure regime.

    Attributes:
        observed_losses: One-dimensional observed losses in the unit interval.
    """

    observed_losses: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observed_losses",
            _float_array(self.observed_losses, "observed_losses", ndim=1, bounds=(0.0, 1.0)),
        )


@dataclass(frozen=True)
class RegimeM1Data:
    """Observed losses and cohort-level observation summaries for M1.

    An empty observed cohort uses ``q=0`` and the neutral convention ``a=0``.
    Optional strata are aggregate-only and must reproduce the supplied ``q`` and ``a``.

    Attributes:
        observed_losses: One-dimensional observed losses in the unit interval.
        total_n: Number of patients in the reference cohort.
        q: Cohort observation rate.
        a: Mean observed loss, or zero when no loss is observed.
        stratum_weights: Optional population shares for disjoint strata.
        stratum_obs_rates: Optional observation rates by stratum.
        stratum_obs_losses: Optional observed mean losses by stratum.
    """

    observed_losses: FloatArray
    total_n: int
    q: float
    a: float
    stratum_weights: FloatArray | None = field(default=None, kw_only=True)
    stratum_obs_rates: FloatArray | None = field(default=None, kw_only=True)
    stratum_obs_losses: FloatArray | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        losses = _float_array(self.observed_losses, "observed_losses", ndim=1, bounds=(0.0, 1.0))
        if not isinstance(self.total_n, Integral) or isinstance(self.total_n, bool):
            raise ValueError("total_n must be a positive integer")
        total_n = int(self.total_n)
        if total_n <= 0 or losses.size > total_n:
            raise ValueError("total_n must be positive and at least the observed sample size")
        q = _probability(self.q, "q")
        a = _probability(self.a, "a")
        expected_q = losses.size / total_n
        expected_a = float(np.mean(losses)) if losses.size else 0.0
        if not np.isclose(q, expected_q, rtol=1e-9, atol=1e-12):
            raise ValueError("q must equal len(observed_losses) / total_n")
        if not np.isclose(a, expected_a, rtol=1e-9, atol=1e-12):
            raise ValueError("a must equal the mean of observed_losses")
        object.__setattr__(self, "observed_losses", losses)
        object.__setattr__(self, "total_n", total_n)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "a", a)

        strata = (self.stratum_weights, self.stratum_obs_rates, self.stratum_obs_losses)
        if all(value is None for value in strata):
            return
        if any(value is None for value in strata):
            raise ValueError("all three stratum aggregates must be supplied together")

        weights = _float_array(self.stratum_weights, "stratum_weights", ndim=1, bounds=(0.0, 1.0))
        rates = _float_array(self.stratum_obs_rates, "stratum_obs_rates", ndim=1, bounds=(0.0, 1.0))
        means = _float_array(
            self.stratum_obs_losses, "stratum_obs_losses", ndim=1, bounds=(0.0, 1.0)
        )
        if weights.size == 0 or rates.shape != weights.shape or means.shape != weights.shape:
            raise ValueError("stratum aggregates must be non-empty vectors of equal length")
        if not np.isclose(weights.sum(), 1.0, rtol=1e-9, atol=1e-9):
            raise ValueError("stratum_weights must sum to one")
        aggregate_q = float(np.dot(weights, rates))
        aggregate_a = (
            float(np.dot(weights * rates, means) / aggregate_q) if aggregate_q > 0.0 else 0.0
        )
        if not np.isclose(aggregate_q, q, rtol=1e-9, atol=1e-9):
            raise ValueError("stratum aggregates must reproduce q")
        if not np.isclose(aggregate_a, a, rtol=1e-9, atol=1e-9):
            raise ValueError("stratum aggregates must reproduce a")
        object.__setattr__(self, "stratum_weights", weights)
        object.__setattr__(self, "stratum_obs_rates", rates)
        object.__setattr__(self, "stratum_obs_losses", means)


@dataclass(frozen=True)
class RegimeM2Data(RegimeM1Data):
    """M1 summaries plus aggregate missing-reason sensitivity information.

    No unobserved patient-level records are included. ``a_k`` and ``b_k`` are
    clipped sensitivity endpoints derived from ``m_k`` and ``delta_k``.

    Attributes:
        reason_names: Mutually exclusive missing-reason category names.
        tilde_p: Logged missing-reason shares over the full reference cohort.
        m_k: Reference mean loss for each reason category.
        delta_k: Non-negative sensitivity radius for each reason category.
        epsilon: Upper bound on aggregate audit misclassification.
        class_acc_lower: Optional per-category classification accuracy lower bound.
        a_k: Derived lower loss endpoint by reason.
        b_k: Derived upper loss endpoint by reason.
    """

    reason_names: tuple[str, ...]
    tilde_p: FloatArray
    m_k: FloatArray
    delta_k: FloatArray
    epsilon: float = 0.0
    class_acc_lower: FloatArray | None = field(default=None, kw_only=True)
    a_k: FloatArray = field(init=False)
    b_k: FloatArray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        names = tuple(self.reason_names)
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("reason_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("reason_names must be unique")
        size = len(names)
        tilde_p = _float_array(self.tilde_p, "tilde_p", ndim=1, bounds=(0.0, 1.0))
        means = _float_array(self.m_k, "m_k", ndim=1, bounds=(0.0, 1.0))
        radii = _float_array(self.delta_k, "delta_k", ndim=1, bounds=(0.0, 1.0))
        if tilde_p.size != size or means.size != size or radii.size != size:
            raise ValueError("tilde_p, m_k, and delta_k must match reason_names")
        if not np.isclose(tilde_p.sum(), 1.0 - self.q, rtol=1e-9, atol=1e-9):
            raise ValueError("tilde_p must sum to 1 - q")
        epsilon = _probability(self.epsilon, "epsilon")

        object.__setattr__(self, "reason_names", names)
        object.__setattr__(self, "tilde_p", tilde_p)
        object.__setattr__(self, "m_k", means)
        object.__setattr__(self, "delta_k", radii)
        object.__setattr__(self, "epsilon", epsilon)
        lower = np.maximum(0.0, means - radii)
        upper = np.minimum(1.0, means + radii)
        lower.setflags(write=False)
        upper.setflags(write=False)
        object.__setattr__(self, "a_k", lower)
        object.__setattr__(self, "b_k", upper)

        if self.class_acc_lower is not None:
            accuracy = _float_array(
                self.class_acc_lower, "class_acc_lower", ndim=1, bounds=(0.0, 1.0)
            )
            if accuracy.size != size:
                raise ValueError("class_acc_lower must match reason_names")
            object.__setattr__(self, "class_acc_lower", accuracy)


@dataclass(frozen=True)
class RegimeM3Data(RegimeM2Data):
    """M2 aggregates plus full-cohort baseline covariates and observation flags.

    Attributes:
        H: Finite patient-level baseline covariates with shape ``(total_n, d)``.
        R: Boolean observation indicator with shape ``(total_n,)``.
    """

    H: FloatArray = field(kw_only=True)
    R: BoolArray = field(kw_only=True)

    def __post_init__(self) -> None:
        super().__post_init__()
        covariates = _float_array(self.H, "H", ndim=2)
        if covariates.shape[0] != self.total_n or covariates.shape[1] == 0:
            raise ValueError("H must have shape (total_n, d) with d > 0")
        try:
            raw_indicator = np.asarray(self.R)
        except (TypeError, ValueError) as exc:
            raise ValueError("R must be a one-dimensional binary array") from exc
        if raw_indicator.ndim != 1 or raw_indicator.size != self.total_n:
            raise ValueError("R must have shape (total_n,)")
        if raw_indicator.dtype.kind not in "biuf" or not np.all(np.isfinite(raw_indicator)):
            raise ValueError("R must contain only boolean or numeric binary values")
        if not np.all((raw_indicator == 0) | (raw_indicator == 1)):
            raise ValueError("R must contain only zero or one")
        indicator = np.array(raw_indicator, dtype=np.bool_, copy=True)
        if int(indicator.sum()) != self.observed_losses.size:
            raise ValueError("R must identify exactly the observed_losses entries")
        indicator.setflags(write=False)
        object.__setattr__(self, "H", covariates)
        object.__setattr__(self, "R", indicator)
