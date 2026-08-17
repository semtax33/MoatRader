from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.financial.dcf import DcfAssumptions, DcfEngine
from moatrader.valuation.base import ValuationMethod, ValuationResult, eligible


class LegacyFcffScenarioSet(ContractModel):
    downside: DcfAssumptions
    base: DcfAssumptions
    upside: DcfAssumptions
    method: ValuationMethod = ValuationMethod.ECONOMIC_FCFF
    provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent(self) -> "LegacyFcffScenarioSet":
        if self.method not in {
            ValuationMethod.ECONOMIC_FCFF,
            ValuationMethod.NORMALIZED_FCFF,
            ValuationMethod.SCENARIO_DCF,
        }:
            raise ValueError("legacy FCFF adapter supports FCFF/scenario methods only")
        if len({item.diluted_shares for item in (self.downside, self.base, self.upside)}) != 1:
            raise ValueError("legacy FCFF scenarios must use the same diluted shares")
        return self


def stress_legacy_fcff(base: DcfAssumptions, *, direction: int) -> DcfAssumptions:
    """Return-blind, symmetric operating stress used for engineering audits."""

    if direction not in {-1, 1}:
        raise ValueError("stress direction must be -1 or 1")
    shift = Decimal(direction)
    growth = [
        max(Decimal("-0.90"), item + shift * Decimal("0.03"))
        for item in base.revenue_growth
    ]
    margins = [
        max(Decimal("-0.95"), min(Decimal("0.95"), item + shift * Decimal("0.02")))
        for item in base.ebit_margin
    ]
    wacc = base.wacc - shift * Decimal("0.01")
    terminal_growth = base.terminal_growth + shift * Decimal("0.005")
    if wacc <= terminal_growth:
        raise ValueError("stress produced invalid WACC/terminal-growth relation")
    return base.model_copy(
        update={
            "revenue_growth": growth,
            "ebit_margin": margins,
            "wacc": wacc,
            "terminal_growth": terminal_growth,
        }
    )


class LegacyFcffCommonEngine:
    def value(self, assumptions: LegacyFcffScenarioSet) -> ValuationResult:
        engine = DcfEngine()
        downside, base, upside = (
            engine.value(item)
            for item in (assumptions.downside, assumptions.base, assumptions.upside)
        )
        stressed_values = [
            downside.fair_value_per_share,
            base.fair_value_per_share,
            upside.fair_value_per_share,
        ]
        downside_value = min(stressed_values)
        upside_value = max(stressed_values)
        return ValuationResult(
            method=assumptions.method,
            applicability=eligible(
                assumptions.method,
                ["revenue", "ebit", "invested_capital", "diluted_shares"],
            ),
            enterprise_value=base.enterprise_value,
            equity_value=base.equity_value,
            fair_value_per_share=base.fair_value_per_share,
            downside_value_per_share=downside_value,
            base_value_per_share=base.fair_value_per_share,
            upside_value_per_share=upside_value,
            assumption_confidence=base.assumption_confidence,
            provenance=assumptions.provenance,
            warnings=base.provenance_warnings + base.screening_exclusion_reasons,
            metadata={
                "adapter": "LEGACY_FCFF_PIT_TTM",
                "screening_eligible": base.screening_eligible,
                "terminal_value_share": str(base.terminal_value_share),
                "stress_case_values": [str(value) for value in stressed_values],
            },
        )
