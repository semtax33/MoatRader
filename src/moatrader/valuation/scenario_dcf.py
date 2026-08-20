from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.financial.dcf import DcfAssumptionType
from moatrader.valuation.assumptions import EconomicDcfAssumptions, ReinvestmentMethod
from moatrader.valuation.base import (
    ValuationMethod,
    ValuationResult,
    eligible,
    split_valuation_disclosures,
)
from moatrader.valuation.economic_dcf import EconomicDcfEngine


SCENARIO_DCF_POLICY_VERSION = "scenario-dcf-policy/1"


class ScenarioAnnualObservation(ContractModel):
    fiscal_year: int = Field(ge=1900, le=2200)
    revenue: Decimal = Field(gt=0)
    ebit: Decimal
    source_refs: list[str] = Field(min_length=1)


class ScenarioDcfBuildInput(ContractModel):
    issuer_id: str = Field(min_length=1)
    as_of: str = Field(min_length=10)
    observations: list[ScenarioAnnualObservation] = Field(min_length=3, max_length=5)
    base_period: str = Field(min_length=1)
    base_revenue: Decimal = Field(gt=0)
    base_ebit: Decimal = Field(lt=0)
    base_invested_capital: Decimal = Field(gt=0)
    tax_rate: Decimal = Field(default=Decimal("0.24"), ge=0, lt=1)
    wacc: Decimal = Field(gt=0, lt=1)
    stable_growth: Decimal = Field(default=Decimal("0.02"), ge=-Decimal("0.10"), lt=Decimal("0.20"))
    net_debt: Decimal = Decimal(0)
    diluted_shares: Decimal = Field(gt=0)
    recovery_evidence: list[str] = Field(min_length=1)
    provenance: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def persistent_loss_with_recovery_path(self) -> "ScenarioDcfBuildInput":
        years = [item.fiscal_year for item in self.observations]
        if years != sorted(set(years)):
            raise ValueError("scenario observations must be unique and sorted")
        if not all(item.ebit < 0 for item in self.observations[-2:]):
            raise ValueError("scenario DCF requires at least two persistent annual losses")
        latest_margin = self.observations[-1].ebit / self.observations[-1].revenue
        current_margin = self.base_ebit / self.base_revenue
        if current_margin <= latest_margin:
            raise ValueError("scenario DCF requires an improving current operating margin")
        return self


class ScenarioDcfAssumptions(ContractModel):
    policy_version: Literal["scenario-dcf-policy/1"] = SCENARIO_DCF_POLICY_VERSION
    ranking_basis: Literal["CENTRAL_UNWEIGHTED"] = "CENTRAL_UNWEIGHTED"
    downside: EconomicDcfAssumptions
    central: EconomicDcfAssumptions
    upside: EconomicDcfAssumptions
    downside_probability: Decimal = Field(default=Decimal("0.25"), ge=0, le=1)
    central_probability: Decimal = Field(default=Decimal("0.50"), ge=0, le=1)
    upside_probability: Decimal = Field(default=Decimal("0.25"), ge=0, le=1)
    assumption_confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_scenarios(self) -> "ScenarioDcfAssumptions":
        if (
            self.downside_probability,
            self.central_probability,
            self.upside_probability,
        ) != (Decimal("0.25"), Decimal("0.50"), Decimal("0.25")):
            raise ValueError("scenario diagnostic probabilities are frozen at 25/50/25")
        shares = {item.diluted_shares for item in (self.downside, self.central, self.upside)}
        if len(shares) != 1:
            raise ValueError("scenario DCFs must use the same diluted shares")
        if [item.scenario for item in (self.downside, self.central, self.upside)] != [
            "DOWNSIDE",
            "CENTRAL",
            "UPSIDE",
        ]:
            raise ValueError("scenario DCF bundles must be explicitly labeled")
        return self


def _clamp(value: Decimal, lower: Decimal, upper: Decimal) -> Decimal:
    return max(lower, min(upper, value))


def _quantile(values: list[Decimal], probability: Decimal) -> Decimal:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires observations")
    if len(ordered) == 1:
        return ordered[0]
    position = probability * Decimal(len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - Decimal(lower)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class ScenarioDcfBuilder:
    """Build fixed-policy mature-state scenarios for persistent operating losses."""

    def build(self, source: ScenarioDcfBuildInput) -> ScenarioDcfAssumptions:
        observations = source.observations
        growth_history = [
            current.revenue / previous.revenue - Decimal(1)
            for previous, current in zip(observations, observations[1:])
            if current.fiscal_year == previous.fiscal_year + 1
        ]
        if len(growth_history) < 2:
            raise ValueError("scenario DCF requires consecutive revenue-growth history")
        growth_values = [
            _clamp(_quantile(growth_history, quantile), Decimal("-0.05"), Decimal("0.20"))
            for quantile in (Decimal("0.25"), Decimal("0.50"), Decimal("0.75"))
        ]
        target_margins = (Decimal("0.00"), Decimal("0.08"), Decimal("0.15"))
        roiic_values = (Decimal("0.08"), Decimal("0.12"), Decimal("0.18"))
        stable_roic_values = (Decimal("0.08"), Decimal("0.10"), Decimal("0.12"))
        wacc_values = (
            _clamp(source.wacc + Decimal("0.02"), Decimal("0.03"), Decimal("0.30")),
            source.wacc,
            _clamp(source.wacc - Decimal("0.02"), Decimal("0.03"), Decimal("0.30")),
        )
        convergence_years = (8, 6, 5)
        cap_years = (3, 5, 7)
        failure_probabilities = (Decimal("0.20"), Decimal("0.10"), Decimal("0.05"))
        history_ref = "SCENARIO_HISTORY:" + ",".join(
            str(item.fiscal_year) for item in observations
        )
        candidates: list[EconomicDcfAssumptions] = []
        for growth, margin, roiic, stable_roic, wacc, convergence, cap, failure in zip(
            growth_values,
            target_margins,
            roiic_values,
            stable_roic_values,
            wacc_values,
            convergence_years,
            cap_years,
            failure_probabilities,
        ):
            if wacc <= source.stable_growth:
                raise ValueError("scenario WACC must exceed stable growth")
            candidates.append(
                EconomicDcfAssumptions(
                    scenario="UNSPECIFIED",
                    base_period=source.base_period,
                    base_revenue=source.base_revenue,
                    base_nopat_margin=(source.base_ebit / source.base_revenue)
                    * (Decimal(1) - source.tax_rate),
                    base_invested_capital=source.base_invested_capital,
                    revenue_growth=growth,
                    target_nopat_margin=margin,
                    margin_convergence_years=convergence,
                    roiic=roiic,
                    reinvestment_method=ReinvestmentMethod.ROIIC,
                    competitive_advantage_period_years=cap,
                    fade_years=5,
                    explicit_forecast_years=12,
                    stable_growth=source.stable_growth,
                    stable_nopat_margin=margin,
                    stable_roic=stable_roic,
                    wacc=wacc,
                    net_debt=source.net_debt,
                    diluted_shares=source.diluted_shares,
                    failure_probability=failure,
                    assumption_sources={
                        "base_revenue": [f"PIT_BASE:{source.base_period}"],
                        "base_nopat_margin": [f"PIT_BASE:{source.base_period}"],
                        "base_invested_capital": [f"PIT_BASE:{source.base_period}"],
                        "revenue_growth": [history_ref],
                        "target_nopat_margin": [SCENARIO_DCF_POLICY_VERSION],
                        "roiic": [SCENARIO_DCF_POLICY_VERSION],
                        "competitive_advantage_period_years": [SCENARIO_DCF_POLICY_VERSION],
                        "fade_years": [SCENARIO_DCF_POLICY_VERSION],
                        "stable_growth": ["POLICY:LONG_RUN_GROWTH_2_PERCENT"],
                        "stable_nopat_margin": [SCENARIO_DCF_POLICY_VERSION],
                        "stable_roic": [SCENARIO_DCF_POLICY_VERSION],
                        "wacc": ["FROZEN_WACC_POLICY"],
                        "net_debt": [f"PIT_BASE:{source.base_period}"],
                        "diluted_shares": ["PIT_KRX_LISTED_SHARES"],
                        "failure_probability": [SCENARIO_DCF_POLICY_VERSION],
                    },
                    assumption_types={
                        "base_revenue": DcfAssumptionType.DETERMINISTIC,
                        "base_nopat_margin": DcfAssumptionType.DETERMINISTIC,
                        "base_invested_capital": DcfAssumptionType.DETERMINISTIC,
                        "revenue_growth": DcfAssumptionType.MODEL_INFERENCE,
                        "target_nopat_margin": DcfAssumptionType.DEFAULT,
                        "roiic": DcfAssumptionType.DEFAULT,
                        "competitive_advantage_period_years": DcfAssumptionType.DEFAULT,
                        "fade_years": DcfAssumptionType.DEFAULT,
                        "stable_growth": DcfAssumptionType.DEFAULT,
                        "stable_nopat_margin": DcfAssumptionType.DEFAULT,
                        "stable_roic": DcfAssumptionType.DEFAULT,
                        "wacc": DcfAssumptionType.DETERMINISTIC,
                        "net_debt": DcfAssumptionType.DETERMINISTIC,
                        "diluted_shares": DcfAssumptionType.DETERMINISTIC,
                        "failure_probability": DcfAssumptionType.DEFAULT,
                    },
                    provenance_warnings=[
                        "Mature margins, ROIIC, CAP, and failure rates are frozen policy scenarios, not guidance.",
                        "Central value only is ranking input; probability-weighted value is diagnostic.",
                    ],
                )
            )
        values = [
            EconomicDcfEngine().value(item).fair_value_per_share for item in candidates
        ]
        downside_index = min(range(3), key=values.__getitem__)
        upside_index = max(range(3), key=values.__getitem__)
        return ScenarioDcfAssumptions(
            downside=candidates[downside_index].model_copy(update={"scenario": "DOWNSIDE"}),
            central=candidates[1].model_copy(update={"scenario": "CENTRAL"}),
            upside=candidates[upside_index].model_copy(update={"scenario": "UPSIDE"}),
            assumption_confidence=Decimal("0.50"),
            provenance=source.provenance
            + source.recovery_evidence
            + [
                history_ref,
                SCENARIO_DCF_POLICY_VERSION,
                "RANKING_BASIS:CENTRAL_UNWEIGHTED",
                "SCENARIO_LABEL:INTRINSIC_VALUE_ORDER_NO_MARKET_PRICE",
                "NO_LLM:DETERMINISTIC_BUILDER",
            ],
        )


class ScenarioDcfEngine:
    def value(self, assumptions: ScenarioDcfAssumptions) -> ValuationResult:
        engine = EconomicDcfEngine()
        downside = engine.value(assumptions.downside)
        central = engine.value(assumptions.central)
        upside = engine.value(assumptions.upside)
        values = [
            downside.fair_value_per_share,
            central.fair_value_per_share,
            upside.fair_value_per_share,
        ]
        if values != sorted(values):
            raise ValueError("scenario DCF values must be ordered downside <= central <= upside")
        weighted_diagnostic = (
            values[0] * assumptions.downside_probability
            + values[1] * assumptions.central_probability
            + values[2] * assumptions.upside_probability
        )
        disclosures, trust_warnings = split_valuation_disclosures(
            central.provenance_warnings,
            assumptions.central.provenance_warnings,
        )
        return ValuationResult(
            method=ValuationMethod.SCENARIO_DCF,
            applicability=eligible(
                ValuationMethod.SCENARIO_DCF,
                ["downside", "central", "upside", "scenario_policy"],
            ),
            enterprise_value=central.enterprise_value,
            equity_value=central.equity_value,
            fair_value_per_share=values[1],
            downside_value_per_share=values[0],
            base_value_per_share=values[1],
            upside_value_per_share=values[2],
            assumption_confidence=assumptions.assumption_confidence,
            provenance=assumptions.provenance,
            disclosures=disclosures,
            warnings=trust_warnings + central.screening_exclusion_reasons,
            metadata={
                "screening_eligible": central.screening_eligible,
                "policy_version": assumptions.policy_version,
                "ranking_basis": assumptions.ranking_basis,
                "central_unweighted_value_per_share": str(values[1]),
                "probability_weighted_diagnostic_value_per_share": str(weighted_diagnostic),
                "diagnostic_probabilities": ["0.25", "0.50", "0.25"],
                "probability_weighted": False,
            },
        )
