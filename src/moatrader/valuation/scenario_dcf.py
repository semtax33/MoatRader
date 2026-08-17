from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from moatrader.valuation.base import ValuationMethod, ValuationResult, eligible
from moatrader.valuation.economic_dcf import EconomicDcfEngine


class ScenarioDcfAssumptions(ContractModel):
    downside: EconomicDcfAssumptions
    central: EconomicDcfAssumptions
    upside: EconomicDcfAssumptions
    downside_probability: Decimal = Field(ge=0, le=1)
    central_probability: Decimal = Field(ge=0, le=1)
    upside_probability: Decimal = Field(ge=0, le=1)
    assumption_confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def coherent_scenarios(self) -> "ScenarioDcfAssumptions":
        if self.downside_probability + self.central_probability + self.upside_probability != Decimal(1):
            raise ValueError("scenario probabilities must sum exactly to one")
        shares = {item.diluted_shares for item in (self.downside, self.central, self.upside)}
        if len(shares) != 1:
            raise ValueError("scenario DCFs must use the same diluted shares")
        return self


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
        weighted = (
            values[0] * assumptions.downside_probability
            + values[1] * assumptions.central_probability
            + values[2] * assumptions.upside_probability
        )
        shares = assumptions.central.diluted_shares
        return ValuationResult(
            method=ValuationMethod.SCENARIO_DCF,
            applicability=eligible(
                ValuationMethod.SCENARIO_DCF,
                ["downside", "central", "upside", "scenario_probabilities"],
            ),
            enterprise_value=central.enterprise_value,
            equity_value=weighted * shares,
            fair_value_per_share=weighted,
            downside_value_per_share=values[0],
            base_value_per_share=weighted,
            upside_value_per_share=values[2],
            assumption_confidence=assumptions.assumption_confidence,
            provenance=assumptions.provenance,
            warnings=central.provenance_warnings,
            metadata={
                "central_unweighted_value_per_share": str(values[1]),
                "probability_weighted": True,
            },
        )
