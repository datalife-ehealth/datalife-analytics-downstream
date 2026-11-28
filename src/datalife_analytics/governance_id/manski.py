"""Classical Manski worst-case bounds for partially observed bounded losses."""

from __future__ import annotations

from dataclasses import dataclass

from .regimes import RegimeM1Data


@dataclass(frozen=True)
class ManskiBoundsResult:
    """Manski interval and its width for a single M1 cohort.

    Attributes:
        lower: Worst-case lower endpoint ``q * a``.
        upper: Worst-case upper endpoint ``q * a + (1 - q)``.
        width: Interval width ``1 - q``.
        q: Cohort observation rate.
        a: Mean observed loss.
    """

    lower: float
    upper: float
    width: float
    q: float
    a: float


def compute_manski_bounds(m1: RegimeM1Data) -> ManskiBoundsResult:
    """Compute the nonparametric worst-case mean-loss interval from M1.

    Args:
        m1: Validated observed-loss and cohort-size summaries.

    Returns:
        The lower and upper endpoints, interval width, and input summaries.
    """
    lower = m1.q * m1.a
    width = 1.0 - m1.q
    return ManskiBoundsResult(
        lower=lower,
        upper=lower + width,
        width=width,
        q=m1.q,
        a=m1.a,
    )
