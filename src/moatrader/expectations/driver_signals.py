from __future__ import annotations

import math
import statistics
from decimal import Decimal
from enum import StrEnum
from typing import Mapping, Sequence

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from moatrader.valuation.economic_dcf import EconomicDcfEngine


TAX_RATE = Decimal("0.24")
STABLE_GROWTH = Decimal("0.02")
WACC_BY_SIZE = {
    "SMALL": Decimal("0.12"),
    "MID": Decimal("0.105"),
    "LARGE": Decimal("0.095"),
}


class DriverName(StrEnum):
    GROWTH = "GROWTH"
    MARGIN = "MARGIN"
    ROIIC = "ROIIC"
    CAP = "CAP"


class ImpliedSolutionStatus(StrEnum):
    SOLVED = "SOLVED"
    CENSORED_LOW = "CENSORED_LOW"
    CENSORED_HIGH = "CENSORED_HIGH"
    NO_ELIGIBLE_VALUATION = "NO_ELIGIBLE_VALUATION"


DRIVER_GRIDS: dict[DriverName, tuple[Decimal, ...]] = {
    DriverName.GROWTH: tuple(
        Decimal(value)
        for value in (
            "-0.20", "-0.10", "-0.05", "0.00", "0.02", "0.05", "0.08",
            "0.10", "0.15", "0.20", "0.30", "0.40", "0.60",
        )
    ),
    DriverName.MARGIN: tuple(
        Decimal(value)
        for value in (
            "-0.30", "-0.20", "-0.10", "0.00", "0.05", "0.10", "0.15",
            "0.20", "0.30", "0.40", "0.60", "0.80",
        )
    ),
    DriverName.ROIIC: tuple(
        Decimal(value)
        for value in (
            "0.01", "0.02", "0.03", "0.05", "0.08", "0.10", "0.15",
            "0.20", "0.30", "0.50", "0.75", "1.00", "2.00", "3.00",
        )
    ),
    DriverName.CAP: tuple(Decimal(value) for value in (0, 1, 2, 3, 5, 7, 10, 12, 15, 20, 25, 30, 40, 50)),
}


def _clamp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    return max(low, min(high, value))


def _median(values: Sequence[Decimal]) -> Decimal:
    return Decimal(str(statistics.median(values)))


class SupportedDriverEstimate(ContractModel):
    base_financial_year: int
    history_years: list[int] = Field(min_length=2)
    base_revenue: Decimal = Field(gt=0)
    base_nopat_margin: Decimal = Field(gt=-1, lt=1)
    base_invested_capital: Decimal = Field(gt=0)
    net_debt: Decimal
    diluted_shares: Decimal = Field(gt=0)
    wacc: Decimal = Field(gt=0, lt=1)
    growth: Decimal = Field(gt=-1, le=3)
    margin: Decimal = Field(gt=-1, lt=1)
    roiic: Decimal = Field(gt=0, le=5)
    cap_years: int = Field(ge=0, le=50)
    historical_growth: list[Decimal] = Field(default_factory=list)
    historical_nopat_margin: list[Decimal] = Field(default_factory=list)
    historical_roiic: list[Decimal] = Field(default_factory=list)
    historical_roic: list[Decimal] = Field(default_factory=list)
    invested_capital_floor_used: bool = False
    roiic_fallback_used: bool = False

    def assumptions(self) -> EconomicDcfAssumptions:
        return EconomicDcfAssumptions(
            scenario="UNSPECIFIED",
            base_period=f"{self.base_financial_year}FY",
            base_revenue=self.base_revenue,
            base_nopat_margin=self.base_nopat_margin,
            base_invested_capital=self.base_invested_capital,
            revenue_growth=self.growth,
            target_nopat_margin=self.margin,
            margin_convergence_years=3,
            roiic=self.roiic,
            competitive_advantage_period_years=self.cap_years,
            fade_years=5,
            explicit_forecast_years=max(10, self.cap_years + 5),
            stable_growth=STABLE_GROWTH,
            stable_nopat_margin=self.margin,
            stable_roic=self.wacc,
            wacc=self.wacc,
            net_debt=self.net_debt,
            diluted_shares=self.diluted_shares,
            assumption_sources={
                "base_revenue": [f"PIT_DART:{self.base_financial_year}FY"],
                "base_nopat_margin": [f"PIT_DART:{self.base_financial_year}FY"],
                "base_invested_capital": [f"PIT_DART:{self.base_financial_year}FY"],
                "revenue_growth": ["PIT_DART:HISTORICAL_MEDIAN"],
                "target_nopat_margin": ["PIT_DART:HISTORICAL_MEDIAN"],
                "roiic": ["PIT_DART:DELTA_NOPAT_OVER_DELTA_INVESTED_CAPITAL"],
                "competitive_advantage_period_years": ["PIT_DART:ROIC_SPREAD_PERSISTENCE_RULE"],
                "stable_growth": ["FROZEN_POLICY:0.02"],
                "stable_nopat_margin": ["PIT_DART:HISTORICAL_MEDIAN"],
                "stable_roic": ["FROZEN_POLICY:WACC"],
                "wacc": ["FROZEN_POLICY:SIZE_BUCKET"],
                "net_debt": [f"PIT_DART:{self.base_financial_year}FY"],
                "diluted_shares": ["PIT_MARCAP:LISTED_SHARES"],
            },
        )


class ImpliedDriverSolution(ContractModel):
    driver: DriverName
    supported: Decimal
    implied: Decimal | None = None
    gap: Decimal | None = None
    status: ImpliedSolutionStatus
    modeled_price: Decimal | None = None
    relative_price_error: Decimal | None = None
    bracket_low: Decimal | None = None
    bracket_high: Decimal | None = None
    grid_point_count: int = Field(ge=1)

    @model_validator(mode="after")
    def gap_matches(self) -> "ImpliedDriverSolution":
        if self.status == ImpliedSolutionStatus.SOLVED:
            if self.implied is None or self.gap is None:
                raise ValueError("solved implied driver requires implied value and gap")
            if self.gap != self.supported - self.implied:
                raise ValueError("driver gap must equal supported minus market implied")
        return self


def supported_driver_estimate(
    history: Sequence[tuple[int, Mapping[str, Decimal | None]]],
    *,
    size_bucket: str,
    diluted_shares: Decimal,
) -> SupportedDriverEstimate:
    observations: list[dict[str, Decimal | int | bool]] = []
    floor_used = False
    for year, metrics in sorted(history, key=lambda item: item[0]):
        revenue = metrics.get("revenue")
        ebit = metrics.get("ebit")
        equity = metrics.get("total_equity")
        if revenue is None or revenue <= 0 or ebit is None or equity is None or equity <= 0:
            continue
        cash = metrics.get("cash") or Decimal(0)
        debt = metrics.get("debt") or Decimal(0)
        invested_capital = equity + debt - cash
        minimum_capital = revenue * Decimal("0.10")
        if invested_capital <= 0:
            invested_capital = minimum_capital
            floor_used = True
        nopat = ebit * (Decimal(1) - TAX_RATE)
        observations.append(
            {
                "year": year,
                "revenue": revenue,
                "nopat": nopat,
                "nopat_margin": nopat / revenue,
                "invested_capital": invested_capital,
                "roic": nopat / invested_capital,
                "cash": cash,
                "debt": debt,
            }
        )
    if len(observations) < 2:
        raise ValueError("at least two positive-revenue/equity PIT annual observations are required")
    growth_values: list[Decimal] = []
    roiic_values: list[Decimal] = []
    for previous, current in zip(observations, observations[1:]):
        prior_revenue = Decimal(previous["revenue"])
        growth_values.append(Decimal(current["revenue"]) / prior_revenue - Decimal(1))
        delta_capital = Decimal(current["invested_capital"]) - Decimal(previous["invested_capital"])
        materiality = abs(Decimal(previous["invested_capital"])) * Decimal("0.01")
        if delta_capital > materiality:
            roiic_values.append(
                (Decimal(current["nopat"]) - Decimal(previous["nopat"])) / delta_capital
            )
    margins = [Decimal(item["nopat_margin"]) for item in observations]
    roics = [Decimal(item["roic"]) for item in observations]
    growth = _clamp(_median(growth_values[-3:]), Decimal("-0.10"), Decimal("0.15"))
    margin = _clamp(_median(margins[-3:]), Decimal("-0.15"), Decimal("0.30"))
    positive_roiic = [value for value in roiic_values[-3:] if value > 0]
    roiic_fallback = not positive_roiic
    roiic_source = _median(positive_roiic) if positive_roiic else max(roics[-1], Decimal("0.03"))
    roiic = _clamp(roiic_source, Decimal("0.03"), Decimal("0.50"))
    wacc = WACC_BY_SIZE.get(size_bucket.upper(), WACC_BY_SIZE["MID"])
    streak = 0
    for roic in reversed(roics):
        if roic > wacc + Decimal("0.02"):
            streak += 1
        else:
            break
    median_roic = _median(roics[-3:])
    cap = int(
        round(
            float(
                _clamp(
                    Decimal(3) + Decimal(20) * max(median_roic - wacc, Decimal(0)) + Decimal(min(streak, 3)),
                    Decimal(3),
                    Decimal(15),
                )
            )
        )
    )
    latest = observations[-1]
    return SupportedDriverEstimate(
        base_financial_year=int(latest["year"]),
        history_years=[int(item["year"]) for item in observations],
        base_revenue=Decimal(latest["revenue"]),
        base_nopat_margin=_clamp(Decimal(latest["nopat_margin"]), Decimal("-0.50"), Decimal("0.50")),
        base_invested_capital=Decimal(latest["invested_capital"]),
        net_debt=Decimal(latest["debt"]) - Decimal(latest["cash"]),
        diluted_shares=diluted_shares,
        wacc=wacc,
        growth=growth,
        margin=margin,
        roiic=roiic,
        cap_years=cap,
        historical_growth=growth_values,
        historical_nopat_margin=margins,
        historical_roiic=roiic_values,
        historical_roic=roics,
        invested_capital_floor_used=floor_used,
        roiic_fallback_used=roiic_fallback,
    )


def _assumptions_with_driver(
    base: EconomicDcfAssumptions,
    driver: DriverName,
    value: Decimal,
) -> EconomicDcfAssumptions:
    update: dict[str, object]
    if driver == DriverName.GROWTH:
        update = {"revenue_growth": value}
    elif driver == DriverName.MARGIN:
        update = {"target_nopat_margin": value}
    elif driver == DriverName.ROIIC:
        update = {"roiic": value}
    else:
        cap = int(value)
        update = {
            "competitive_advantage_period_years": cap,
            "explicit_forecast_years": max(10, cap + base.fade_years),
        }
    return base.model_copy(update=update)


def implied_driver_solution(
    *,
    base: EconomicDcfAssumptions,
    current_price: Decimal,
    driver: DriverName,
    engine: EconomicDcfEngine | None = None,
) -> ImpliedDriverSolution:
    if current_price <= 0:
        raise ValueError("current price must be positive")
    valuation_engine = engine or EconomicDcfEngine()
    supported = {
        DriverName.GROWTH: base.revenue_growth,
        DriverName.MARGIN: base.target_nopat_margin,
        DriverName.ROIIC: base.roiic,
        DriverName.CAP: Decimal(base.competitive_advantage_period_years),
    }[driver]
    points: list[tuple[Decimal, Decimal]] = []
    for value in DRIVER_GRIDS[driver]:
        assumptions = _assumptions_with_driver(base, driver, value)
        valuation = valuation_engine.value(assumptions)
        if valuation.equity_value > 0 and valuation.fair_value_per_share > 0:
            points.append((value, valuation.fair_value_per_share))
    if not points:
        return ImpliedDriverSolution(
            driver=driver,
            supported=supported,
            status=ImpliedSolutionStatus.NO_ELIGIBLE_VALUATION,
            grid_point_count=len(DRIVER_GRIDS[driver]),
        )
    brackets: list[tuple[Decimal, Decimal, Decimal, Decimal]] = []
    for (left_value, left_price), (right_value, right_price) in zip(points, points[1:]):
        left_error = left_price - current_price
        right_error = right_price - current_price
        if left_error == 0:
            brackets.append((left_value, left_value, left_price, left_price))
        elif left_error * right_error <= 0:
            brackets.append((left_value, right_value, left_price, right_price))
    if brackets:
        left_value, right_value, left_price, right_price = min(
            brackets,
            key=lambda item: abs(((item[0] + item[1]) / Decimal(2)) - supported),
        )
        if right_price == left_price or right_value == left_value:
            implied = left_value
        else:
            implied = left_value + (current_price - left_price) * (right_value - left_value) / (right_price - left_price)
        if driver == DriverName.CAP:
            implied = _clamp(implied, left_value, right_value)
        modeled = current_price
        return ImpliedDriverSolution(
            driver=driver,
            supported=supported,
            implied=implied,
            gap=supported - implied,
            status=ImpliedSolutionStatus.SOLVED,
            modeled_price=modeled,
            relative_price_error=Decimal(0),
            bracket_low=left_value,
            bracket_high=right_value,
            grid_point_count=len(DRIVER_GRIDS[driver]),
        )
    nearest_value, nearest_price = min(points, key=lambda item: abs(item[1] - current_price))
    price_values = [price for _value, price in points]
    status = (
        ImpliedSolutionStatus.CENSORED_HIGH
        if current_price > max(price_values)
        else ImpliedSolutionStatus.CENSORED_LOW
    )
    return ImpliedDriverSolution(
        driver=driver,
        supported=supported,
        implied=nearest_value,
        gap=supported - nearest_value,
        status=status,
        modeled_price=nearest_price,
        relative_price_error=(nearest_price - current_price) / current_price,
        grid_point_count=len(DRIVER_GRIDS[driver]),
    )


def all_driver_solutions(
    *,
    estimate: SupportedDriverEstimate,
    current_price: Decimal,
    engine: EconomicDcfEngine | None = None,
) -> dict[DriverName, ImpliedDriverSolution]:
    base = estimate.assumptions()
    valuation_engine = engine or EconomicDcfEngine()
    return {
        driver: implied_driver_solution(
            base=base,
            current_price=current_price,
            driver=driver,
            engine=valuation_engine,
        )
        for driver in DriverName
    }
