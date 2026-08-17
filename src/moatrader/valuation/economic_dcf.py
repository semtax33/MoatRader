from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import Field

from moatrader.canonical.models import ContractModel
from moatrader.valuation.assumptions import EconomicDcfAssumptions, ReinvestmentMethod
from moatrader.valuation.terminal import gordon_value, stable_fcff


class EconomicPhase(StrEnum):
    COMPETITIVE_ADVANTAGE = "COMPETITIVE_ADVANTAGE"
    FADE = "FADE"
    STEADY_STATE = "STEADY_STATE"


class EconomicDcfProjection(ContractModel):
    year: int = Field(ge=1)
    phase: EconomicPhase
    revenue_growth: Decimal
    revenue: Decimal
    nopat_margin: Decimal
    nopat: Decimal
    roiic_assumption: Decimal
    reinvestment: Decimal
    realized_roiic: Decimal | None = None
    ending_invested_capital: Decimal
    roic: Decimal
    fcff: Decimal
    discount_factor: Decimal
    present_value: Decimal


class EconomicDcfValuation(ContractModel):
    method: str = "ECONOMIC_FCFF"
    base_period: str | None = None
    scenario: str
    assumptions: EconomicDcfAssumptions
    projections: list[EconomicDcfProjection] = Field(min_length=1)
    terminal_reinvestment_rate: Decimal
    terminal_fcff: Decimal
    terminal_value: Decimal
    terminal_present_value: Decimal
    unadjusted_enterprise_value: Decimal
    enterprise_value: Decimal
    equity_value: Decimal
    fair_value_per_share: Decimal
    terminal_value_share: Decimal
    assets_in_place_value: Decimal
    steady_state_value: Decimal
    pvgo: Decimal
    pvgo_share: Decimal | None = None
    cap_value_contribution: Decimal
    screening_eligible: bool = True
    screening_exclusion_reasons: list[str] = Field(default_factory=list)
    provenance_warnings: list[str] = Field(default_factory=list)


def _lerp(start: Decimal, end: Decimal, fraction: Decimal) -> Decimal:
    return start + (end - start) * fraction


class EconomicDcfEngine:
    """Economic FCFF with explicit growth-reinvestment-ROIIC-CAP consistency."""

    def value(self, assumptions: EconomicDcfAssumptions) -> EconomicDcfValuation:
        core = self._value_core(assumptions)
        no_cap = assumptions.model_copy(
            update={
                "competitive_advantage_period_years": 0,
                "fade_years": 0,
            }
        )
        no_cap_core = self._value_core(no_cap)
        cap_contribution = core["enterprise_value"] - no_cap_core["enterprise_value"]
        enterprise = core["enterprise_value"]
        assets_in_place = self._probability_adjusted(
            assumptions.base_nopat / assumptions.wacc,
            assumptions,
        )
        pvgo = enterprise - assets_in_place
        terminal_share = (
            core["terminal_present_value"] / core["unadjusted_enterprise_value"]
            if core["unadjusted_enterprise_value"]
            else Decimal(0)
        )
        warnings = list(assumptions.provenance_warnings)
        if terminal_share > Decimal("0.75"):
            warnings.append("terminal value exceeds 75% of unadjusted enterprise value")
        exclusions: list[str] = []
        equity = enterprise - assumptions.net_debt
        if enterprise <= 0:
            exclusions.append("NON_POSITIVE_ENTERPRISE_VALUE")
        if equity <= 0:
            exclusions.append("NON_POSITIVE_EQUITY_VALUE")
        if terminal_share < 0 or terminal_share > Decimal("0.90"):
            exclusions.append("UNRELIABLE_TERMINAL_VALUE_SHARE")
        if any(item.fcff < 0 for item in core["projections"][-2:]):
            warnings.append("late explicit-period FCFF is negative")
        return EconomicDcfValuation(
            base_period=assumptions.base_period,
            scenario=assumptions.scenario,
            assumptions=assumptions,
            projections=core["projections"],
            terminal_reinvestment_rate=core["terminal_reinvestment_rate"],
            terminal_fcff=core["terminal_fcff"],
            terminal_value=core["terminal_value"],
            terminal_present_value=core["terminal_present_value"],
            unadjusted_enterprise_value=core["unadjusted_enterprise_value"],
            enterprise_value=enterprise,
            equity_value=equity,
            fair_value_per_share=equity / assumptions.diluted_shares,
            terminal_value_share=terminal_share,
            assets_in_place_value=assets_in_place,
            steady_state_value=no_cap_core["enterprise_value"],
            pvgo=pvgo,
            pvgo_share=pvgo / enterprise if enterprise else None,
            cap_value_contribution=cap_contribution,
            screening_eligible=not exclusions,
            screening_exclusion_reasons=exclusions,
            provenance_warnings=warnings,
        )

    def _value_core(self, assumptions: EconomicDcfAssumptions) -> dict[str, object]:
        revenue = assumptions.base_revenue
        previous_nopat = assumptions.base_nopat
        invested_capital = assumptions.base_invested_capital
        projections: list[EconomicDcfProjection] = []
        for year in range(1, assumptions.explicit_forecast_years + 1):
            growth, margin, roiic, phase = self._year_assumptions(assumptions, year)
            previous_revenue = revenue
            revenue = previous_revenue * (Decimal(1) + growth)
            nopat = revenue * margin
            delta_nopat = nopat - previous_nopat
            if assumptions.reinvestment_method == ReinvestmentMethod.ROIIC:
                reinvestment = max(delta_nopat, Decimal(0)) / roiic
            else:
                assert assumptions.sales_to_capital is not None
                reinvestment = max(revenue - previous_revenue, Decimal(0)) / assumptions.sales_to_capital
            realized_roiic = delta_nopat / reinvestment if reinvestment else None
            invested_capital += reinvestment
            roic = nopat / invested_capital
            fcff = nopat - reinvestment
            discount_factor = Decimal(1) / ((Decimal(1) + assumptions.wacc) ** year)
            projections.append(
                EconomicDcfProjection(
                    year=year,
                    phase=phase,
                    revenue_growth=growth,
                    revenue=revenue,
                    nopat_margin=margin,
                    nopat=nopat,
                    roiic_assumption=roiic,
                    reinvestment=reinvestment,
                    realized_roiic=realized_roiic,
                    ending_invested_capital=invested_capital,
                    roic=roic,
                    fcff=fcff,
                    discount_factor=discount_factor,
                    present_value=fcff * discount_factor,
                )
            )
            previous_nopat = nopat

        terminal_nopat = (
            projections[-1].revenue
            * (Decimal(1) + assumptions.stable_growth)
            * assumptions.stable_nopat_margin
        )
        terminal_fcff = stable_fcff(
            next_period_nopat=terminal_nopat,
            growth=assumptions.stable_growth,
            roic=assumptions.stable_roic,
        )
        terminal_value = gordon_value(
            cash_flow=terminal_fcff,
            discount_rate=assumptions.wacc,
            growth=assumptions.stable_growth,
        )
        terminal_pv = terminal_value * projections[-1].discount_factor
        unadjusted = sum((item.present_value for item in projections), Decimal(0)) + terminal_pv
        return {
            "projections": projections,
            "terminal_reinvestment_rate": assumptions.stable_growth / assumptions.stable_roic,
            "terminal_fcff": terminal_fcff,
            "terminal_value": terminal_value,
            "terminal_present_value": terminal_pv,
            "unadjusted_enterprise_value": unadjusted,
            "enterprise_value": self._probability_adjusted(unadjusted, assumptions),
        }

    @staticmethod
    def _probability_adjusted(
        value: Decimal,
        assumptions: EconomicDcfAssumptions,
    ) -> Decimal:
        survive = Decimal(1) - assumptions.failure_probability
        return (
            value * survive
            + assumptions.distress_recovery_enterprise_value * assumptions.failure_probability
        )

    @staticmethod
    def _year_assumptions(
        assumptions: EconomicDcfAssumptions,
        year: int,
    ) -> tuple[Decimal, Decimal, Decimal, EconomicPhase]:
        cap = assumptions.competitive_advantage_period_years
        fade = assumptions.fade_years
        margin_fraction = min(
            Decimal(year) / Decimal(assumptions.margin_convergence_years),
            Decimal(1),
        )
        target_margin = _lerp(
            assumptions.base_nopat_margin,
            assumptions.target_nopat_margin,
            margin_fraction,
        )
        if year <= cap:
            return (
                assumptions.revenue_growth,
                target_margin,
                assumptions.roiic,
                EconomicPhase.COMPETITIVE_ADVANTAGE,
            )
        if fade and year <= cap + fade:
            fraction = Decimal(year - cap) / Decimal(fade)
            return (
                _lerp(assumptions.revenue_growth, assumptions.stable_growth, fraction),
                _lerp(target_margin, assumptions.stable_nopat_margin, fraction),
                _lerp(assumptions.roiic, assumptions.stable_roic, fraction),
                EconomicPhase.FADE,
            )
        return (
            assumptions.stable_growth,
            assumptions.stable_nopat_margin,
            assumptions.stable_roic,
            EconomicPhase.STEADY_STATE,
        )
