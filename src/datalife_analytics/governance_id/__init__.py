"""Governance-aware identification contracts and synthetic data generation."""

from .dgp import CohortBundle, GovernanceDGPConfig, generate_governed_cohort
from .manski import ManskiBoundsResult, compute_manski_bounds
from .regimes import RegimeM0Data, RegimeM1Data, RegimeM2Data, RegimeM3Data

__all__ = [
    "CohortBundle",
    "GovernanceDGPConfig",
    "ManskiBoundsResult",
    "RegimeM0Data",
    "RegimeM1Data",
    "RegimeM2Data",
    "RegimeM3Data",
    "compute_manski_bounds",
    "generate_governed_cohort",
]
