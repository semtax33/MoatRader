from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from moatrader.valuation.base import ValuationMethod, ValuationResult, eligible
from moatrader.valuation.biotech_rnpv import BiotechRnpvAssumptions, BiotechRnpvEngine
from moatrader.valuation.economic_dcf import EconomicDcfEngine


class EconomicFcffScenarioSet(ContractModel):
    downside: EconomicDcfAssumptions
    base: EconomicDcfAssumptions
    upside: EconomicDcfAssumptions
    method: ValuationMethod = ValuationMethod.ECONOMIC_FCFF
    assumption_confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_method_and_shares(self) -> "EconomicFcffScenarioSet":
        if self.method not in {ValuationMethod.ECONOMIC_FCFF, ValuationMethod.NORMALIZED_FCFF}:
            raise ValueError("FCFF scenario set supports economic or normalized FCFF only")
        shares = {item.diluted_shares for item in (self.downside, self.base, self.upside)}
        if len(shares) != 1:
            raise ValueError("FCFF scenarios must use the same diluted shares")
        return self


class CommonEconomicFcffEngine:
    def value(self, assumptions: EconomicFcffScenarioSet) -> ValuationResult:
        engine = EconomicDcfEngine()
        downside, base, upside = (
            engine.value(item) for item in (assumptions.downside, assumptions.base, assumptions.upside)
        )
        values = [downside.fair_value_per_share, base.fair_value_per_share, upside.fair_value_per_share]
        if values != sorted(values):
            raise ValueError("FCFF scenarios must be ordered downside <= base <= upside")
        return ValuationResult(
            method=assumptions.method,
            applicability=eligible(
                assumptions.method,
                ["base_revenue", "base_nopat_margin", "roiic", "wacc", "diluted_shares"],
            ),
            enterprise_value=base.enterprise_value,
            equity_value=base.equity_value,
            fair_value_per_share=values[1],
            downside_value_per_share=values[0],
            base_value_per_share=values[1],
            upside_value_per_share=values[2],
            assumption_confidence=assumptions.assumption_confidence,
            provenance=assumptions.provenance,
            warnings=base.provenance_warnings,
            metadata={"terminal_value_share": str(base.terminal_value_share)},
        )


class RnpvScenarioSet(ContractModel):
    downside: BiotechRnpvAssumptions
    base: BiotechRnpvAssumptions
    upside: BiotechRnpvAssumptions
    assumption_confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def same_shares(self) -> "RnpvScenarioSet":
        shares = {item.diluted_shares for item in (self.downside, self.base, self.upside)}
        if len(shares) != 1:
            raise ValueError("rNPV scenarios must use the same diluted shares")
        return self


class CommonRnpvEngine:
    def value(self, assumptions: RnpvScenarioSet) -> ValuationResult:
        engine = BiotechRnpvEngine()
        downside, base, upside = (
            engine.value(item) for item in (assumptions.downside, assumptions.base, assumptions.upside)
        )
        values = [downside.fair_value_per_share, base.fair_value_per_share, upside.fair_value_per_share]
        if values != sorted(values):
            raise ValueError("rNPV scenarios must be ordered downside <= base <= upside")
        return ValuationResult(
            method=ValuationMethod.RNPV,
            applicability=eligible(
                ValuationMethod.RNPV,
                ["assets", "probability_of_approval", "launch_value", "development_costs"],
            ),
            equity_value=base.equity_value,
            fair_value_per_share=values[1],
            downside_value_per_share=values[0],
            base_value_per_share=values[1],
            upside_value_per_share=values[2],
            assumption_confidence=assumptions.assumption_confidence,
            provenance=assumptions.provenance,
            metadata={"asset_count": len(base.assets)},
        )
