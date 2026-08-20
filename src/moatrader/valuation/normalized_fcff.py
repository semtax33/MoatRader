from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
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


NORMALIZED_FCFF_POLICY_VERSION = "normalized-fcff-policy/1"


class NormalizationMethod(StrEnum):
    WINSORIZED_MEDIAN = "WINSORIZED_MEDIAN"


class CyclePhase(StrEnum):
    TROUGH = "TROUGH"
    MID_CYCLE = "MID_CYCLE"
    PEAK = "PEAK"


class NormalizedAnnualObservation(ContractModel):
    fiscal_year: int = Field(ge=1900, le=2200)
    revenue: Decimal = Field(gt=0)
    ebit: Decimal
    source_refs: list[str] = Field(min_length=1)


class NormalizationContract(ContractModel):
    contract_version: Literal["normalized-fcff-policy/1"] = NORMALIZED_FCFF_POLICY_VERSION
    window_years: int = Field(default=7, ge=5, le=10)
    minimum_observations: int = Field(default=5, ge=5, le=10)
    method: Literal[NormalizationMethod.WINSORIZED_MEDIAN] = (
        NormalizationMethod.WINSORIZED_MEDIAN
    )
    winsor_lower: Decimal = Field(default=Decimal("0.10"), ge=0, lt=Decimal("0.50"))
    winsor_upper: Decimal = Field(default=Decimal("0.90"), gt=Decimal("0.50"), le=1)
    included_fiscal_years: list[int] = Field(min_length=5, max_length=10)
    excluded_fiscal_years: list[int] = Field(default_factory=list, max_length=10)
    cycle_phase: CyclePhase

    @model_validator(mode="after")
    def coherent_window(self) -> "NormalizationContract":
        included = self.included_fiscal_years
        excluded = self.excluded_fiscal_years
        if included != sorted(set(included)):
            raise ValueError("included fiscal years must be unique and sorted")
        if excluded != sorted(set(excluded)):
            raise ValueError("excluded fiscal years must be unique and sorted")
        if set(included) & set(excluded):
            raise ValueError("a fiscal year cannot be both included and excluded")
        if len(included) < self.minimum_observations:
            raise ValueError("normalization contract lacks minimum observations")
        if included[-1] - included[0] + 1 > self.window_years:
            raise ValueError("included fiscal years exceed the frozen normalization window")
        if self.winsor_lower >= self.winsor_upper:
            raise ValueError("winsor lower bound must be below upper bound")
        return self


class NormalizedFcffBuildInput(ContractModel):
    issuer_id: str = Field(min_length=1)
    as_of: str = Field(min_length=10)
    observations: list[NormalizedAnnualObservation] = Field(min_length=5, max_length=10)
    normalization: NormalizationContract
    base_period: str = Field(min_length=1)
    base_revenue: Decimal = Field(gt=0)
    base_ebit: Decimal
    base_invested_capital: Decimal = Field(gt=0)
    tax_rate: Decimal = Field(default=Decimal("0.24"), ge=0, lt=1)
    wacc: Decimal = Field(gt=0, lt=1)
    stable_growth: Decimal = Field(default=Decimal("0.02"), ge=-Decimal("0.10"), lt=Decimal("0.20"))
    stable_roic: Decimal = Field(default=Decimal("0.10"), gt=0, le=2)
    net_debt: Decimal = Decimal(0)
    diluted_shares: Decimal = Field(gt=0)
    provenance: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def observations_match_contract(self) -> "NormalizedFcffBuildInput":
        years = [item.fiscal_year for item in self.observations]
        if years != sorted(set(years)):
            raise ValueError("normalization observations must be unique and sorted")
        if years != self.normalization.included_fiscal_years:
            raise ValueError("observation years must equal frozen included fiscal years")
        return self


class NormalizedFcffAssumptions(ContractModel):
    downside: EconomicDcfAssumptions
    base: EconomicDcfAssumptions
    upside: EconomicDcfAssumptions
    normalization: NormalizationContract
    normalized_revenue_growth: Decimal
    normalized_nopat_margin: Decimal
    normalized_sales_to_capital: Decimal = Field(gt=0)
    assumption_confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    provenance: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def coherent_scenarios(self) -> "NormalizedFcffAssumptions":
        scenarios = (self.downside, self.base, self.upside)
        if [item.scenario for item in scenarios] != ["DOWNSIDE", "CENTRAL", "UPSIDE"]:
            raise ValueError("normalized FCFF scenarios must be explicitly labeled")
        if len({item.diluted_shares for item in scenarios}) != 1:
            raise ValueError("normalized FCFF scenarios must use the same diluted shares")
        return self


def _clamp(value: Decimal, lower: Decimal, upper: Decimal) -> Decimal:
    return max(lower, min(upper, value))


def _quantile(values: list[Decimal], probability: Decimal) -> Decimal:
    if not values:
        raise ValueError("quantile requires observations")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * Decimal(len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - Decimal(lower_index)
    return ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction


def _winsorized(values: list[Decimal], lower: Decimal, upper: Decimal) -> list[Decimal]:
    low = _quantile(values, lower)
    high = _quantile(values, upper)
    return [_clamp(item, low, high) for item in values]


def infer_cycle_phase(
    observations: list[NormalizedAnnualObservation],
    *,
    current_ebit_margin: Decimal,
) -> CyclePhase:
    margins = [item.ebit / item.revenue for item in observations]
    low = _quantile(margins, Decimal("0.25"))
    high = _quantile(margins, Decimal("0.75"))
    if current_ebit_margin <= low:
        return CyclePhase.TROUGH
    if current_ebit_margin >= high:
        return CyclePhase.PEAK
    return CyclePhase.MID_CYCLE


class NormalizedFcffBuilder:
    """Build price-blind normalized FCFF from a frozen multi-year contract."""

    def build(self, source: NormalizedFcffBuildInput) -> NormalizedFcffAssumptions:
        observations = source.observations
        policy = source.normalization
        margins = [item.ebit / item.revenue for item in observations]
        growth_rates = [
            current.revenue / previous.revenue - Decimal(1)
            for previous, current in zip(observations, observations[1:])
            if current.fiscal_year == previous.fiscal_year + 1
        ]
        if len(growth_rates) < policy.minimum_observations - 1:
            raise ValueError("normalization requires consecutive fiscal-year growth observations")
        winsorized_margins = _winsorized(
            margins, policy.winsor_lower, policy.winsor_upper
        )
        winsorized_growth = _winsorized(
            growth_rates, policy.winsor_lower, policy.winsor_upper
        )
        current_margin = source.base_ebit / source.base_revenue
        inferred_phase = infer_cycle_phase(
            observations,
            current_ebit_margin=current_margin,
        )
        if policy.cycle_phase != inferred_phase:
            raise ValueError(
                "frozen cycle phase does not match deterministic history inference: "
                f"{policy.cycle_phase.value} != {inferred_phase.value}"
            )

        growth_values = [
            _clamp(_quantile(winsorized_growth, quantile), Decimal("-0.10"), Decimal("0.15"))
            for quantile in (Decimal("0.25"), Decimal("0.50"), Decimal("0.75"))
        ]
        nopat_margins = [
            _clamp(
                _quantile(winsorized_margins, quantile) * (Decimal(1) - source.tax_rate),
                Decimal("-0.20"),
                Decimal("0.35"),
            )
            for quantile in (Decimal("0.25"), Decimal("0.50"), Decimal("0.75"))
        ]
        current_nopat_margin = _clamp(
            current_margin * (Decimal(1) - source.tax_rate),
            Decimal("-0.20"),
            Decimal("0.35"),
        )
        sales_to_capital = _clamp(
            source.base_revenue / source.base_invested_capital,
            Decimal("0.10"),
            Decimal("10"),
        )
        wacc_values = (
            _clamp(source.wacc + Decimal("0.015"), Decimal("0.03"), Decimal("0.30")),
            source.wacc,
            _clamp(source.wacc - Decimal("0.015"), Decimal("0.03"), Decimal("0.30")),
        )
        history_ref = (
            "NORMALIZATION_HISTORY:"
            + ",".join(str(item.fiscal_year) for item in observations)
        )
        scenario_assumptions: list[EconomicDcfAssumptions] = []
        for growth, target_margin, wacc in zip(
            growth_values, nopat_margins, wacc_values
        ):
            if wacc <= source.stable_growth:
                raise ValueError("scenario WACC must exceed frozen stable growth")
            scenario_assumptions.append(
                EconomicDcfAssumptions(
                    scenario="UNSPECIFIED",
                    base_period=source.base_period,
                    base_revenue=source.base_revenue,
                    base_nopat_margin=current_nopat_margin,
                    base_invested_capital=source.base_invested_capital,
                    revenue_growth=growth,
                    target_nopat_margin=target_margin,
                    margin_convergence_years=5,
                    roiic=source.stable_roic,
                    reinvestment_method=ReinvestmentMethod.SALES_TO_CAPITAL,
                    sales_to_capital=sales_to_capital,
                    competitive_advantage_period_years=5,
                    fade_years=5,
                    explicit_forecast_years=10,
                    stable_growth=source.stable_growth,
                    stable_nopat_margin=target_margin,
                    stable_roic=source.stable_roic,
                    wacc=wacc,
                    net_debt=source.net_debt,
                    diluted_shares=source.diluted_shares,
                    assumption_sources={
                        "base_revenue": [f"PIT_BASE:{source.base_period}"],
                        "base_nopat_margin": [f"PIT_BASE:{source.base_period}"],
                        "base_invested_capital": [f"PIT_BASE:{source.base_period}"],
                        "revenue_growth": [history_ref, policy.contract_version],
                        "target_nopat_margin": [history_ref, policy.contract_version],
                        "roiic": [policy.contract_version],
                        "stable_growth": ["POLICY:LONG_RUN_GROWTH_2_PERCENT"],
                        "stable_nopat_margin": [history_ref, policy.contract_version],
                        "stable_roic": [policy.contract_version],
                        "wacc": ["FROZEN_WACC_POLICY"],
                        "net_debt": [f"PIT_BASE:{source.base_period}"],
                        "diluted_shares": ["PIT_KRX_LISTED_SHARES"],
                    },
                    assumption_types={
                        "base_revenue": DcfAssumptionType.DETERMINISTIC,
                        "base_nopat_margin": DcfAssumptionType.DETERMINISTIC,
                        "base_invested_capital": DcfAssumptionType.DETERMINISTIC,
                        "revenue_growth": DcfAssumptionType.MODEL_INFERENCE,
                        "target_nopat_margin": DcfAssumptionType.MODEL_INFERENCE,
                        "roiic": DcfAssumptionType.DEFAULT,
                        "stable_growth": DcfAssumptionType.DEFAULT,
                        "stable_nopat_margin": DcfAssumptionType.MODEL_INFERENCE,
                        "stable_roic": DcfAssumptionType.DEFAULT,
                        "wacc": DcfAssumptionType.DETERMINISTIC,
                        "net_debt": DcfAssumptionType.DETERMINISTIC,
                        "diluted_shares": DcfAssumptionType.DETERMINISTIC,
                    },
                    provenance_warnings=[
                        "Normalized values use a frozen winsorized historical policy, not guidance.",
                        "Sales-to-capital and stable ROIC remain policy assumptions.",
                    ],
                )
            )
        # Growth can destroy value when incremental returns trail the cost of
        # capital. Label complete, price-free driver bundles by their computed
        # intrinsic outcome instead of assuming that high growth is always an
        # upside. The central bundle remains the ranked base case.
        candidate_values = [
            EconomicDcfEngine().value(item).fair_value_per_share
            for item in scenario_assumptions
        ]
        downside_index = min(range(3), key=candidate_values.__getitem__)
        upside_index = max(range(3), key=candidate_values.__getitem__)
        labeled_assumptions = [
            scenario_assumptions[downside_index].model_copy(
                update={"scenario": "DOWNSIDE"}
            ),
            scenario_assumptions[1].model_copy(update={"scenario": "CENTRAL"}),
            scenario_assumptions[upside_index].model_copy(
                update={"scenario": "UPSIDE"}
            ),
        ]
        confidence = Decimal("0.75") if len(observations) >= 7 else Decimal("0.60")
        return NormalizedFcffAssumptions(
            downside=labeled_assumptions[0],
            base=labeled_assumptions[1],
            upside=labeled_assumptions[2],
            normalization=policy,
            normalized_revenue_growth=growth_values[1],
            normalized_nopat_margin=nopat_margins[1],
            normalized_sales_to_capital=sales_to_capital,
            assumption_confidence=confidence,
            provenance=source.provenance
            + [
                history_ref,
                policy.contract_version,
                "SCENARIO_LABEL:INTRINSIC_VALUE_ORDER_NO_MARKET_PRICE",
                "NO_LLM:DETERMINISTIC_BUILDER",
            ],
        )


class NormalizedFcffEngine:
    def value(self, assumptions: NormalizedFcffAssumptions) -> ValuationResult:
        engine = EconomicDcfEngine()
        downside, base, upside = (
            engine.value(item)
            for item in (assumptions.downside, assumptions.base, assumptions.upside)
        )
        values = [
            downside.fair_value_per_share,
            base.fair_value_per_share,
            upside.fair_value_per_share,
        ]
        if values != sorted(values):
            raise ValueError("normalized FCFF values must be ordered downside <= base <= upside")
        disclosures, trust_warnings = split_valuation_disclosures(
            base.provenance_warnings,
            assumptions.base.provenance_warnings,
        )
        return ValuationResult(
            method=ValuationMethod.NORMALIZED_FCFF,
            applicability=eligible(
                ValuationMethod.NORMALIZED_FCFF,
                [
                    "normalization_contract",
                    "history_5y",
                    "base_invested_capital",
                    "diluted_shares",
                ],
            ),
            enterprise_value=base.enterprise_value,
            equity_value=base.equity_value,
            fair_value_per_share=values[1],
            downside_value_per_share=values[0],
            base_value_per_share=values[1],
            upside_value_per_share=values[2],
            assumption_confidence=assumptions.assumption_confidence,
            provenance=assumptions.provenance,
            disclosures=disclosures,
            warnings=trust_warnings + base.screening_exclusion_reasons,
            metadata={
                "screening_eligible": base.screening_eligible,
                "normalization_policy": assumptions.normalization.model_dump(mode="json"),
                "normalized_revenue_growth": str(assumptions.normalized_revenue_growth),
                "normalized_nopat_margin": str(assumptions.normalized_nopat_margin),
                "normalized_sales_to_capital": str(
                    assumptions.normalized_sales_to_capital
                ),
                "terminal_value_share": str(base.terminal_value_share),
            },
        )
