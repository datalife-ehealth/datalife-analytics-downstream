"""Day 1 contracts, data-generation invariants, and Manski-bound properties."""

from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from datalife_analytics.governance_id import (
    GovernanceDGPConfig,
    RegimeM0Data,
    RegimeM1Data,
    RegimeM2Data,
    compute_manski_bounds,
    generate_governed_cohort,
)


def _config(
    *,
    n_patients: int = 4_000,
    baseline_risk_shift: float = 0.0,
    structural_zero_prob: float = 0.0,
    confusion_matrix: np.ndarray | None = None,
) -> GovernanceDGPConfig:
    return GovernanceDGPConfig(
        n_patients=n_patients,
        reason_names=("policy", "capacity", "other"),
        m_k=np.array([0.25, 0.45, 0.68]),
        delta_k=np.array([0.10, 0.15, 0.20]),
        baseline_risk_shift=baseline_risk_shift,
        confusion_matrix=confusion_matrix,
        epsilon_upper=0.05,
        structural_zero_prob=structural_zero_prob,
    )


def test_regime_contracts_isolate_disclosed_information_and_reject_invalid_values():
    observed_losses = np.array([0.2, 0.4])
    m0 = RegimeM0Data(observed_losses=observed_losses)
    m1 = RegimeM1Data(observed_losses=observed_losses, total_n=4, q=0.5, a=0.3)
    m2 = RegimeM2Data(
        observed_losses=observed_losses,
        total_n=4,
        q=0.5,
        a=0.3,
        reason_names=("policy", "capacity"),
        tilde_p=np.array([0.25, 0.25]),
        m_k=np.array([0.3, 0.6]),
        delta_k=np.array([0.1, 0.2]),
    )

    assert not hasattr(m0, "total_n")
    assert m1.q == 0.5
    assert {item.name for item in fields(RegimeM0Data)} == {"observed_losses"}
    assert not hasattr(m2, "H")
    assert not hasattr(m2, "R")
    assert not {"H", "R", "unobserved_losses", "patient_losses"}.intersection(
        item.name for item in fields(RegimeM2Data)
    )
    assert np.allclose(m2.a_k, [0.2, 0.4])
    assert np.allclose(m2.b_k, [0.4, 0.8])

    with pytest.raises(ValueError, match="sum to 1 - q"):
        RegimeM2Data(
            observed_losses=observed_losses,
            total_n=4,
            q=0.5,
            a=0.3,
            reason_names=("policy", "capacity"),
            tilde_p=np.array([0.2, 0.2]),
            m_k=np.array([0.3, 0.6]),
            delta_k=np.array([0.1, 0.2]),
        )
    with pytest.raises(ValueError, match="observed_losses values"):
        RegimeM0Data(observed_losses=np.array([0.2, 1.1]))


def test_confusion_matrix_rows_must_be_probabilities():
    with pytest.raises(ValueError, match="rows must sum to one"):
        _config(confusion_matrix=np.array([[0.8, 0.3, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]))


def test_mcar_observed_mean_is_close_to_oracle_mean_at_large_n():
    cohort = generate_governed_cohort(_config(n_patients=20_000), seed=31415)

    assert np.isclose(cohort.m2.tilde_p.sum(), 1.0 - cohort.m1.q)
    assert cohort.m1.q == cohort.m1.observed_losses.size / cohort.m1.total_n
    assert abs(cohort.m1.a - cohort.oracle_true_psi) < 0.015
    assert cohort.m3.H.shape == (20_000, 3)
    assert cohort.m3.R.shape == (20_000,)
    assert cohort.m3.R.sum() == cohort.m1.observed_losses.size


@pytest.mark.parametrize(
    ("baseline_risk_shift", "structural_zero_prob"),
    [(0.0, 0.0), (1.5, 0.0), (-2.0, 0.2), (0.75, 0.65), (0.0, 1.0)],
)
def test_manski_bounds_contain_oracle_truth(
    baseline_risk_shift: float, structural_zero_prob: float
):
    cohort = generate_governed_cohort(
        _config(
            baseline_risk_shift=baseline_risk_shift,
            structural_zero_prob=structural_zero_prob,
        ),
        seed=2718,
    )
    result = compute_manski_bounds(cohort.m1)

    assert result.lower <= cohort.oracle_true_psi <= result.upper
    assert np.isclose(result.width, 1.0 - cohort.m1.q)
