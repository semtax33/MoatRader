from __future__ import annotations

import math
from decimal import Decimal
from enum import StrEnum
from typing import Mapping, Sequence

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.expectations.driver_signals import DriverName, TAX_RATE
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from moatrader.valuation.economic_dcf import EconomicDcfEngine


D = Decimal


class RevisionStatus(StrEnum):
    SOLVED = "SOLVED"
    NO_ELIGIBLE_CURVE = "NO_ELIGIBLE_CURVE"
    ENTRY_PRICE_OUTSIDE_CURVE = "ENTRY_PRICE_OUTSIDE_CURVE"
    TARGET_PRICE_OUTSIDE_CURVE = "TARGET_PRICE_OUTSIDE_CURVE"
    NON_POSITIVE_SENSITIVITY = "NON_POSITIVE_SENSITIVITY"


# Wider than the v7.2 static grids.  These limits are model-domain bounds fixed
# before the v7.3 target-price stage is opened; censored observations remain
# excluded from the primary mechanism test.
DYNAMIC_REVISION_GRIDS: dict[DriverName, tuple[Decimal, ...]] = {
    DriverName.GROWTH: tuple(
        D(value)
        for value in (
            "-0.50", "-0.30", "-0.20", "-0.10", "-0.05", "0.00", "0.02",
            "0.05", "0.08", "0.10", "0.15", "0.20", "0.30", "0.40", "0.60",
            "0.80", "1.00",
        )
    ),
    DriverName.MARGIN: tuple(
        D(value)
        for value in (
            "-0.50", "-0.30", "-0.20", "-0.10", "0.00", "0.05", "0.10",
            "0.15", "0.20", "0.30", "0.40", "0.60", "0.80", "0.90",
        )
    ),
    DriverName.ROIIC: tuple(
        D(value)
        for value in (
            "0.005", "0.01", "0.02", "0.03", "0.05", "0.08", "0.10",
            "0.15", "0.20", "0.30", "0.50", "0.75", "1.00", "2.00",
            "3.00", "4.00", "5.00",
        )
    ),
    DriverName.CAP: tuple(D(value) for value in range(0, 51)),
}


DRIVER_SHOCKS: dict[DriverName, Decimal] = {
    DriverName.GROWTH: D("0.01"),
    DriverName.MARGIN: D("0.01"),
    DriverName.ROIIC: D("0.05"),
    DriverName.CAP: D("1"),
}


class PeriodicValueFactorVector(ContractModel):
    """Return-free, same-fiscal-period evidence proxies.

    The vector intentionally does not claim direct volume or price/mix
    identification.  It keeps the observable financial components separate.
    """

    fiscal_year: int
    fiscal_month: int = Field(ge=1, le=12)
    growth_yoy: Decimal | None = None
    growth_acceleration: Decimal | None = None
    nopat_margin_change: Decimal | None = None
    operating_leverage_spread: Decimal | None = None
    roiic_change: Decimal | None = None
    incremental_sales_efficiency_change: Decimal | None = None
    roic_spread_change: Decimal | None = None
    positive_roic_spread_persistence: Decimal | None = None

    def components_for(self, driver: DriverName) -> dict[str, Decimal | None]:
        return {
            DriverName.GROWTH: {
                "growth_yoy": self.growth_yoy,
                "growth_acceleration": self.growth_acceleration,
            },
            DriverName.MARGIN: {
                "nopat_margin_change": self.nopat_margin_change,
                "operating_leverage_spread": self.operating_leverage_spread,
            },
            DriverName.ROIIC: {
                "roiic_change": self.roiic_change,
                "incremental_sales_efficiency_change": (
                    self.incremental_sales_efficiency_change
                ),
            },
            DriverName.CAP: {
                "roic_spread_change": self.roic_spread_change,
                "positive_roic_spread_persistence": (
                    self.positive_roic_spread_persistence
                ),
            },
        }[driver]


class DriverSensitivity(ContractModel):
    driver: DriverName
    shock: Decimal = Field(gt=0)
    base_price: Decimal = Field(gt=0)
    low_price: Decimal | None = None
    high_price: Decimal | None = None
    signed_price_change_per_shock: Decimal | None = None
    absolute_price_change_per_shock: Decimal | None = None
    eligible: bool = False


class DynamicImpliedRevision(ContractModel):
    driver: DriverName
    status: RevisionStatus
    entry_price: Decimal = Field(gt=0)
    target_price: Decimal = Field(gt=0)
    entry_implied: Decimal | None = None
    target_implied: Decimal | None = None
    implied_revision: Decimal | None = None
    entry_bracket_low: Decimal | None = None
    entry_bracket_high: Decimal | None = None
    target_bracket_low: Decimal | None = None
    target_bracket_high: Decimal | None = None
    grid_point_count: int = Field(ge=1)

    @model_validator(mode="after")
    def revision_matches(self) -> "DynamicImpliedRevision":
        if self.status == RevisionStatus.SOLVED:
            if (
                self.entry_implied is None
                or self.target_implied is None
                or self.implied_revision is None
            ):
                raise ValueError("solved revision requires both implied levels")
            if self.implied_revision != self.target_implied - self.entry_implied:
                raise ValueError("implied revision must be target minus entry")
        return self


def _as_decimal(value: Decimal | float | int | None) -> Decimal | None:
    if value is None:
        return None
    result = D(str(value))
    return result if result.is_finite() else None


def _safe_ratio(numerator: Decimal | None, denominator: Decimal | None) -> Decimal | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _invested_capital(metrics: Mapping[str, Decimal | float | int | None]) -> Decimal | None:
    equity = _as_decimal(metrics.get("total_equity"))
    if equity is None or equity <= 0:
        return None
    debt = _as_decimal(metrics.get("debt")) or D(0)
    cash = _as_decimal(metrics.get("cash")) or D(0)
    capital = equity + debt - cash
    return capital if capital > 0 else None


def _period_observation(
    item: tuple[int, int, Mapping[str, Decimal | float | int | None]],
) -> dict[str, Decimal | int | None]:
    year, month, metrics = item
    revenue = _as_decimal(metrics.get("revenue"))
    ebit = _as_decimal(metrics.get("ebit"))
    nopat = ebit * (D(1) - TAX_RATE) if ebit is not None else None
    capital = _invested_capital(metrics)
    return {
        "year": year,
        "month": month,
        "revenue": revenue,
        "nopat": nopat,
        "capital": capital,
        "margin": _safe_ratio(nopat, revenue),
        "roic": _safe_ratio(nopat, capital),
    }


def _material_delta(current: Decimal | None, previous: Decimal | None) -> Decimal | None:
    if current is None or previous is None:
        return None
    change = current - previous
    if abs(change) <= max(abs(previous) * D("0.01"), D("1")):
        return None
    return change


def periodic_value_factor_vector(
    periods: Sequence[
        tuple[int, int, Mapping[str, Decimal | float | int | None]]
    ],
    *,
    wacc: Decimal,
) -> PeriodicValueFactorVector:
    """Build the v7.3 evidence vector without using any market return."""

    if len(periods) < 3:
        raise ValueError("three same-fiscal-month observations are required")
    current, prior, prior2 = [_period_observation(item) for item in periods[:3]]
    months = {int(item["month"]) for item in (current, prior, prior2)}
    years = [int(item["year"]) for item in (current, prior, prior2)]
    if len(months) != 1 or years != [years[0], years[0] - 1, years[0] - 2]:
        raise ValueError("periods must be descending consecutive years with one fiscal month")

    growth = None
    prior_growth = None
    if current["revenue"] is not None and prior["revenue"] is not None and prior["revenue"] > 0:
        growth = current["revenue"] / prior["revenue"] - D(1)
    if prior["revenue"] is not None and prior2["revenue"] is not None and prior2["revenue"] > 0:
        prior_growth = prior["revenue"] / prior2["revenue"] - D(1)

    delta_revenue = _material_delta(current["revenue"], prior["revenue"])
    delta_nopat = (
        current["nopat"] - prior["nopat"]
        if current["nopat"] is not None and prior["nopat"] is not None
        else None
    )
    operating_leverage = _safe_ratio(delta_nopat, delta_revenue)
    operating_leverage_spread = (
        operating_leverage - prior["margin"]
        if operating_leverage is not None and prior["margin"] is not None
        else None
    )

    current_delta_capital = _material_delta(current["capital"], prior["capital"])
    prior_delta_capital = _material_delta(prior["capital"], prior2["capital"])
    current_roiic = (
        _safe_ratio(delta_nopat, current_delta_capital)
        if current_delta_capital is not None and current_delta_capital > 0
        else None
    )
    prior_delta_nopat = (
        prior["nopat"] - prior2["nopat"]
        if prior["nopat"] is not None and prior2["nopat"] is not None
        else None
    )
    prior_roiic = (
        _safe_ratio(prior_delta_nopat, prior_delta_capital)
        if prior_delta_capital is not None and prior_delta_capital > 0
        else None
    )
    current_sales_efficiency = (
        _safe_ratio(delta_revenue, current_delta_capital)
        if current_delta_capital is not None and current_delta_capital > 0
        else None
    )
    prior_delta_revenue = _material_delta(prior["revenue"], prior2["revenue"])
    prior_sales_efficiency = (
        _safe_ratio(prior_delta_revenue, prior_delta_capital)
        if prior_delta_capital is not None and prior_delta_capital > 0
        else None
    )

    current_spread = current["roic"] - wacc if current["roic"] is not None else None
    prior_spread = prior["roic"] - wacc if prior["roic"] is not None else None
    prior2_spread = prior2["roic"] - wacc if prior2["roic"] is not None else None
    persistence = None
    if all(value is not None for value in (current_spread, prior_spread, prior2_spread)):
        persistence = D(sum(value > 0 for value in (current_spread, prior_spread, prior2_spread))) / D(3)

    return PeriodicValueFactorVector(
        fiscal_year=years[0],
        fiscal_month=next(iter(months)),
        growth_yoy=growth,
        growth_acceleration=(growth - prior_growth) if growth is not None and prior_growth is not None else None,
        nopat_margin_change=(
            current["margin"] - prior["margin"]
            if current["margin"] is not None and prior["margin"] is not None
            else None
        ),
        operating_leverage_spread=operating_leverage_spread,
        roiic_change=(current_roiic - prior_roiic) if current_roiic is not None and prior_roiic is not None else None,
        incremental_sales_efficiency_change=(
            current_sales_efficiency - prior_sales_efficiency
            if current_sales_efficiency is not None and prior_sales_efficiency is not None
            else None
        ),
        roic_spread_change=(
            current_spread - prior_spread
            if current_spread is not None and prior_spread is not None
            else None
        ),
        positive_roic_spread_persistence=persistence,
    )


def assumptions_with_driver(
    base: EconomicDcfAssumptions,
    driver: DriverName,
    value: Decimal,
) -> EconomicDcfAssumptions:
    if driver == DriverName.GROWTH:
        update: dict[str, object] = {"revenue_growth": value}
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


def driver_sensitivities(
    base: EconomicDcfAssumptions,
    *,
    engine: EconomicDcfEngine | None = None,
) -> dict[DriverName, DriverSensitivity]:
    valuation_engine = engine or EconomicDcfEngine()
    base_price = valuation_engine.value(base).fair_value_per_share
    if base_price <= 0:
        raise ValueError("base valuation must have positive equity value")
    supported = {
        DriverName.GROWTH: base.revenue_growth,
        DriverName.MARGIN: base.target_nopat_margin,
        DriverName.ROIIC: base.roiic,
        DriverName.CAP: D(base.competitive_advantage_period_years),
    }
    results: dict[DriverName, DriverSensitivity] = {}
    for driver, shock in DRIVER_SHOCKS.items():
        center = supported[driver]
        low = center - shock
        high = center + shock
        domain_low, domain_high = {
            DriverName.GROWTH: (D("-0.99"), D("3")),
            DriverName.MARGIN: (D("-0.99"), D("0.99")),
            DriverName.ROIIC: (D("0.001"), D("5")),
            DriverName.CAP: (D("0"), D("50")),
        }[driver]
        low = max(domain_low, low)
        high = min(domain_high, high)
        if high <= low:
            results[driver] = DriverSensitivity(
                driver=driver,
                shock=shock,
                base_price=base_price,
                eligible=False,
            )
            continue
        try:
            low_price = valuation_engine.value(
                assumptions_with_driver(base, driver, low)
            ).fair_value_per_share
            high_price = valuation_engine.value(
                assumptions_with_driver(base, driver, high)
            ).fair_value_per_share
        except Exception:
            low_price = None
            high_price = None
        denominator = high - low
        signed = (
            ((high_price - low_price) / base_price) / denominator * shock
            if low_price is not None and high_price is not None and denominator > 0
            else None
        )
        eligible = signed is not None and signed > 0 and math.isfinite(float(signed))
        results[driver] = DriverSensitivity(
            driver=driver,
            shock=shock,
            base_price=base_price,
            low_price=low_price,
            high_price=high_price,
            signed_price_change_per_shock=signed,
            absolute_price_change_per_shock=abs(signed) if signed is not None else None,
            eligible=eligible,
        )
    return results


def turbo_driver(sensitivities: Mapping[DriverName, DriverSensitivity]) -> DriverName | None:
    eligible = [item for item in sensitivities.values() if item.eligible]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (
            item.absolute_price_change_per_shock or D(0),
            -list(DriverName).index(item.driver),
        ),
    ).driver


def _valuation_curve(
    base: EconomicDcfAssumptions,
    driver: DriverName,
    engine: EconomicDcfEngine,
) -> list[tuple[Decimal, Decimal]]:
    points: list[tuple[Decimal, Decimal]] = []
    for value in DYNAMIC_REVISION_GRIDS[driver]:
        try:
            price = engine.value(assumptions_with_driver(base, driver, value)).fair_value_per_share
        except Exception:
            continue
        if price > 0:
            points.append((value, price))
    return points


def _roots(
    points: Sequence[tuple[Decimal, Decimal]],
    price: Decimal,
) -> list[tuple[Decimal, Decimal, Decimal]]:
    roots: list[tuple[Decimal, Decimal, Decimal]] = []
    for (left_value, left_price), (right_value, right_price) in zip(points, points[1:]):
        left_error = left_price - price
        right_error = right_price - price
        if left_error == 0:
            roots.append((left_value, left_value, left_value))
        elif left_error * right_error <= 0:
            implied = (
                left_value
                if right_price == left_price
                else left_value
                + (price - left_price)
                * (right_value - left_value)
                / (right_price - left_price)
            )
            roots.append((implied, left_value, right_value))
    if points and points[-1][1] == price:
        roots.append((points[-1][0], points[-1][0], points[-1][0]))
    return roots


def dynamic_implied_revision(
    *,
    base: EconomicDcfAssumptions,
    driver: DriverName,
    entry_price: Decimal,
    target_price: Decimal,
    engine: EconomicDcfEngine | None = None,
) -> DynamicImpliedRevision:
    """Identify a driver revision with one frozen curve and continuous branch.

    WACC, scale, net debt, shares, and the other three drivers are identical for
    the two price observations.  At entry the root nearest the supported value
    is selected; at target the root nearest that entry root is selected.
    """

    if entry_price <= 0 or target_price <= 0:
        raise ValueError("both prices must be positive")
    valuation_engine = engine or EconomicDcfEngine()
    points = _valuation_curve(base, driver, valuation_engine)
    count = len(DYNAMIC_REVISION_GRIDS[driver])
    if len(points) < 2:
        return DynamicImpliedRevision(
            driver=driver,
            status=RevisionStatus.NO_ELIGIBLE_CURVE,
            entry_price=entry_price,
            target_price=target_price,
            grid_point_count=count,
        )
    entry_roots = _roots(points, entry_price)
    if not entry_roots:
        return DynamicImpliedRevision(
            driver=driver,
            status=RevisionStatus.ENTRY_PRICE_OUTSIDE_CURVE,
            entry_price=entry_price,
            target_price=target_price,
            grid_point_count=count,
        )
    supported = {
        DriverName.GROWTH: base.revenue_growth,
        DriverName.MARGIN: base.target_nopat_margin,
        DriverName.ROIIC: base.roiic,
        DriverName.CAP: D(base.competitive_advantage_period_years),
    }[driver]
    entry = min(entry_roots, key=lambda item: abs(item[0] - supported))
    target_roots = _roots(points, target_price)
    if not target_roots:
        return DynamicImpliedRevision(
            driver=driver,
            status=RevisionStatus.TARGET_PRICE_OUTSIDE_CURVE,
            entry_price=entry_price,
            target_price=target_price,
            entry_implied=entry[0],
            entry_bracket_low=entry[1],
            entry_bracket_high=entry[2],
            grid_point_count=count,
        )
    target = min(target_roots, key=lambda item: abs(item[0] - entry[0]))
    return DynamicImpliedRevision(
        driver=driver,
        status=RevisionStatus.SOLVED,
        entry_price=entry_price,
        target_price=target_price,
        entry_implied=entry[0],
        target_implied=target[0],
        implied_revision=target[0] - entry[0],
        entry_bracket_low=entry[1],
        entry_bracket_high=entry[2],
        target_bracket_low=target[1],
        target_bracket_high=target[2],
        grid_point_count=count,
    )
