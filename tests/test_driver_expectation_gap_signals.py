from __future__ import annotations

from decimal import Decimal

import pytest

from moatrader.expectations.driver_signals import (
    DriverName,
    ImpliedSolutionStatus,
    all_driver_solutions,
    implied_driver_solution,
    supported_driver_estimate,
)
from moatrader.valuation.economic_dcf import EconomicDcfEngine


D = Decimal


def history() -> list[tuple[int, dict[str, Decimal | None]]]:
    return [
        (
            2016,
            {
                "revenue": D("1000"),
                "ebit": D("100"),
                "cash": D("50"),
                "debt": D("150"),
                "total_equity": D("500"),
            },
        ),
        (
            2017,
            {
                "revenue": D("1100"),
                "ebit": D("121"),
                "cash": D("60"),
                "debt": D("160"),
                "total_equity": D("540"),
            },
        ),
        (
            2018,
            {
                "revenue": D("1210"),
                "ebit": D("145.2"),
                "cash": D("70"),
                "debt": D("170"),
                "total_equity": D("580"),
            },
        ),
    ]


def test_supported_driver_estimate_is_price_blind_and_deterministic() -> None:
    estimate = supported_driver_estimate(history(), size_bucket="MID", diluted_shares=D("10"))
    assert estimate.growth == D("0.10")
    assert estimate.margin == D("0.0836")
    assert estimate.roiic > 0
    assert D(3) <= estimate.cap_years <= D(15)
    assert "price" not in estimate.model_dump()
    assumptions = estimate.assumptions()
    assert assumptions.stable_roic == D("0.105")
    assert assumptions.driver_evidence_ids == {}


@pytest.mark.parametrize(
    ("driver", "true_value", "tolerance"),
    [
        (DriverName.GROWTH, D("0.08"), D("0.01")),
        (DriverName.MARGIN, D("0.15"), D("0.02")),
        (DriverName.ROIIC, D("0.20"), D("0.04")),
        (DriverName.CAP, D("10"), D("1.0")),
    ],
)
def test_one_driver_reverse_solver_recovers_known_market_expectation(
    driver: DriverName,
    true_value: Decimal,
    tolerance: Decimal,
) -> None:
    estimate = supported_driver_estimate(history(), size_bucket="MID", diluted_shares=D("10"))
    base = estimate.assumptions()
    update = {
        DriverName.GROWTH: {"revenue_growth": true_value},
        DriverName.MARGIN: {"target_nopat_margin": true_value},
        DriverName.ROIIC: {"roiic": true_value},
        DriverName.CAP: {
            "competitive_advantage_period_years": int(true_value),
            "explicit_forecast_years": max(10, int(true_value) + base.fade_years),
        },
    }[driver]
    market_price = EconomicDcfEngine().value(base.model_copy(update=update)).fair_value_per_share
    solution = implied_driver_solution(base=base, current_price=market_price, driver=driver)
    assert solution.status == ImpliedSolutionStatus.SOLVED
    assert solution.implied is not None
    assert abs(solution.implied - true_value) <= tolerance
    assert solution.gap == solution.supported - solution.implied


def test_all_four_named_gaps_are_emitted() -> None:
    estimate = supported_driver_estimate(history(), size_bucket="LARGE", diluted_shares=D("10"))
    supported_price = EconomicDcfEngine().value(estimate.assumptions()).fair_value_per_share
    results = all_driver_solutions(estimate=estimate, current_price=supported_price)
    assert set(results) == {DriverName.GROWTH, DriverName.MARGIN, DriverName.ROIIC, DriverName.CAP}
    for result in results.values():
        assert result.status == ImpliedSolutionStatus.SOLVED
        assert result.gap is not None


def test_supported_driver_requires_two_annual_observations() -> None:
    with pytest.raises(ValueError, match="at least two"):
        supported_driver_estimate(history()[:1], size_bucket="SMALL", diluted_shares=D("10"))
