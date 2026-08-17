from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from moatrader.valuation.economic_dcf import EconomicDcfEngine, EconomicDcfValuation


class IntrinsicScenarioSet(ContractModel):
    """Price-blind downside/central/upside assumptions."""

    downside: EconomicDcfAssumptions
    central: EconomicDcfAssumptions
    upside: EconomicDcfAssumptions
    evidence_confidence: Decimal = Field(ge=0, le=1)
    maximum_range_expansion: Decimal = Field(default=Decimal("0.50"), ge=0, le=2)

    @model_validator(mode="after")
    def scenario_labels_are_explicit(self) -> "IntrinsicScenarioSet":
        expected = {
            "downside": "DOWNSIDE",
            "central": "CENTRAL",
            "upside": "UPSIDE",
        }
        for name, label in expected.items():
            scenario = getattr(self, name)
            if scenario.scenario != label:
                raise ValueError(f"{name} assumptions must use scenario={label}")
        return self


class IntrinsicValuationRange(ContractModel):
    downside: EconomicDcfValuation
    central: EconomicDcfValuation
    upside: EconomicDcfValuation
    observed_low_per_share: Decimal = Field(ge=0)
    central_per_share: Decimal = Field(ge=0)
    observed_high_per_share: Decimal = Field(ge=0)
    confidence_adjusted_low_per_share: Decimal = Field(ge=0)
    confidence_adjusted_high_per_share: Decimal = Field(ge=0)
    evidence_confidence: Decimal = Field(ge=0, le=1)
    range_expansion_multiplier: Decimal = Field(ge=1)
    price_inputs_used: bool = False

    @model_validator(mode="after")
    def ordered_and_price_blind(self) -> "IntrinsicValuationRange":
        if not self.observed_low_per_share <= self.central_per_share <= self.observed_high_per_share:
            raise ValueError("intrinsic scenario values must bracket the central value")
        if self.confidence_adjusted_low_per_share > self.observed_low_per_share:
            raise ValueError("low confidence must not narrow the downside range")
        if self.confidence_adjusted_high_per_share < self.observed_high_per_share:
            raise ValueError("low confidence must not narrow the upside range")
        if self.price_inputs_used:
            raise ValueError("intrinsic valuation must be price-blind")
        return self


class ScenarioValuationEngine:
    def __init__(self, engine: EconomicDcfEngine | None = None) -> None:
        self.engine = engine or EconomicDcfEngine()

    def value(self, scenarios: IntrinsicScenarioSet) -> IntrinsicValuationRange:
        downside = self.engine.value(scenarios.downside)
        central = self.engine.value(scenarios.central)
        upside = self.engine.value(scenarios.upside)
        invalid = {
            name: valuation.screening_exclusion_reasons
            for name, valuation in (
                ("downside", downside),
                ("central", central),
                ("upside", upside),
            )
            if not valuation.screening_eligible
        }
        if invalid:
            raise ValueError(f"intrinsic scenarios contain ineligible DCF results: {invalid}")
        values = [
            downside.fair_value_per_share,
            central.fair_value_per_share,
            upside.fair_value_per_share,
        ]
        observed_low = min(values)
        observed_high = max(values)
        if not (
            downside.fair_value_per_share
            <= central.fair_value_per_share
            <= upside.fair_value_per_share
        ):
            raise ValueError(
                "scenario labels must be economically ordered: downside <= central <= upside"
            )
        multiplier = Decimal(1) + (
            Decimal(1) - scenarios.evidence_confidence
        ) * scenarios.maximum_range_expansion
        low_distance = central.fair_value_per_share - observed_low
        high_distance = observed_high - central.fair_value_per_share
        return IntrinsicValuationRange(
            downside=downside,
            central=central,
            upside=upside,
            observed_low_per_share=observed_low,
            central_per_share=central.fair_value_per_share,
            observed_high_per_share=observed_high,
            confidence_adjusted_low_per_share=max(
                Decimal(0),
                central.fair_value_per_share - low_distance * multiplier,
            ),
            confidence_adjusted_high_per_share=(
                central.fair_value_per_share + high_distance * multiplier
            ),
            evidence_confidence=scenarios.evidence_confidence,
            range_expansion_multiplier=multiplier,
        )
