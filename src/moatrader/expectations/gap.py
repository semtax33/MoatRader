from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import Field, model_validator

from moatrader.business.competitive_advantage import CapAssessment
from moatrader.business.drivers import ValuationDriver
from moatrader.canonical.models import ContractModel
from moatrader.valuation.reference_class import DecimalRange
from moatrader.valuation.reverse_dcf import ImpliedDriverRange, ImpliedExpectationSurface
from moatrader.valuation.scenarios import IntrinsicScenarioSet, IntrinsicValuationRange
from moatrader.valuation.three_p import (
    CheckStatus,
    PlausibilityStatus,
    ProbabilitySupport,
    ThreePResult,
)


class ExpectationGapDirection(StrEnum):
    FAVORABLE = "FAVORABLE"
    OVERLAP = "OVERLAP"
    UNFAVORABLE = "UNFAVORABLE"
    INDETERMINATE = "INDETERMINATE"


class DriverExpectationGap(ContractModel):
    driver: ValuationDriver
    evidence_based_low: Decimal
    evidence_based_high: Decimal
    market_implied_low: Decimal
    market_implied_high: Decimal
    direction: ExpectationGapDirection
    rationale: str


class ExpectationGapEvaluation(ContractModel):
    current_price: Decimal = Field(gt=0)
    probable_value_low: Decimal
    probable_value_central: Decimal
    probable_value_high: Decimal
    downside_value: Decimal
    central_value_gap: Decimal
    downside_value_gap: Decimal
    valuation_range_width_pct: Decimal
    driver_gaps: list[DriverExpectationGap]
    direction: ExpectationGapDirection
    three_p_verdict: str
    evidence_confidence: Decimal = Field(ge=0, le=1)
    confidence_changes_range_not_value: bool = True
    intrinsic_and_market_lanes_separate: bool = True
    rationale: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_architecture(self) -> "ExpectationGapEvaluation":
        if not self.confidence_changes_range_not_value:
            raise ValueError("confidence must widen a valuation range, not multiply value")
        if not self.intrinsic_and_market_lanes_separate:
            raise ValueError("intrinsic and market lanes must remain separate until gap evaluation")
        return self


def _direction(
    evidence_low: Decimal,
    evidence_high: Decimal,
    implied: ImpliedDriverRange,
) -> ExpectationGapDirection:
    if evidence_low > implied.high:
        return ExpectationGapDirection.FAVORABLE
    if evidence_high < implied.low:
        return ExpectationGapDirection.UNFAVORABLE
    return ExpectationGapDirection.OVERLAP


class ExpectationGapEvaluator:
    def evaluate(
        self,
        *,
        scenarios: IntrinsicScenarioSet,
        intrinsic: IntrinsicValuationRange,
        cap: CapAssessment,
        reverse: ImpliedExpectationSurface,
        central_three_p: ThreePResult,
    ) -> ExpectationGapEvaluation:
        market_price = reverse.market.current_price
        ranges = {
            ValuationDriver.REVENUE_GROWTH: DecimalRange(
                low=min(
                    scenarios.downside.revenue_growth,
                    scenarios.central.revenue_growth,
                    scenarios.upside.revenue_growth,
                ),
                high=max(
                    scenarios.downside.revenue_growth,
                    scenarios.central.revenue_growth,
                    scenarios.upside.revenue_growth,
                ),
            ),
            ValuationDriver.TARGET_MARGIN: DecimalRange(
                low=min(
                    scenarios.downside.target_nopat_margin,
                    scenarios.central.target_nopat_margin,
                    scenarios.upside.target_nopat_margin,
                ),
                high=max(
                    scenarios.downside.target_nopat_margin,
                    scenarios.central.target_nopat_margin,
                    scenarios.upside.target_nopat_margin,
                ),
            ),
            ValuationDriver.ROIIC: DecimalRange(
                low=min(scenarios.downside.roiic, scenarios.central.roiic, scenarios.upside.roiic),
                high=max(scenarios.downside.roiic, scenarios.central.roiic, scenarios.upside.roiic),
            ),
            ValuationDriver.CAP_FADE: DecimalRange(
                low=Decimal(cap.low_years),
                high=Decimal(cap.high_years),
            ),
        }
        implied = {
            ValuationDriver.REVENUE_GROWTH: reverse.implied_revenue_growth,
            ValuationDriver.TARGET_MARGIN: reverse.implied_target_nopat_margin,
            ValuationDriver.ROIIC: reverse.implied_roiic,
            ValuationDriver.CAP_FADE: reverse.implied_cap_years,
        }
        driver_gaps: list[DriverExpectationGap] = []
        for driver, evidence_range in ranges.items():
            market_range = implied[driver]
            direction = _direction(evidence_range.low, evidence_range.high, market_range)
            driver_gaps.append(
                DriverExpectationGap(
                    driver=driver,
                    evidence_based_low=evidence_range.low,
                    evidence_based_high=evidence_range.high,
                    market_implied_low=market_range.low,
                    market_implied_high=market_range.high,
                    direction=direction,
                    rationale=(
                        "Evidence-based probable range is compared with every near-price reverse-DCF "
                        "solution, not with a fabricated single implied assumption."
                    ),
                )
            )

        central_gap = intrinsic.central_per_share / market_price - Decimal(1)
        downside_gap = intrinsic.confidence_adjusted_low_per_share / market_price - Decimal(1)
        width = (
            intrinsic.confidence_adjusted_high_per_share
            - intrinsic.confidence_adjusted_low_per_share
        ) / market_price
        adverse = sum(item.direction == ExpectationGapDirection.UNFAVORABLE for item in driver_gaps)
        favorable = sum(item.direction == ExpectationGapDirection.FAVORABLE for item in driver_gaps)
        valid_story = (
            central_three_p.possible == CheckStatus.PASS
            and central_three_p.plausible != PlausibilityStatus.OUTLIER
            and central_three_p.probable != ProbabilitySupport.CONTRADICTED
        )
        rationale = [
            f"central value gap={central_gap}",
            f"confidence-adjusted downside gap={downside_gap}",
            f"favorable driver gaps={favorable}, unfavorable={adverse}",
        ]
        if not valid_story:
            direction = ExpectationGapDirection.INDETERMINATE
            rationale.append("central assumptions did not clear price-blind 3P validation")
        elif central_gap > 0 and adverse == 0:
            direction = ExpectationGapDirection.FAVORABLE
        elif central_gap < 0 or adverse > favorable:
            direction = ExpectationGapDirection.UNFAVORABLE
        else:
            direction = ExpectationGapDirection.OVERLAP
        return ExpectationGapEvaluation(
            current_price=market_price,
            probable_value_low=intrinsic.confidence_adjusted_low_per_share,
            probable_value_central=intrinsic.central_per_share,
            probable_value_high=intrinsic.confidence_adjusted_high_per_share,
            downside_value=intrinsic.downside.fair_value_per_share,
            central_value_gap=central_gap,
            downside_value_gap=downside_gap,
            valuation_range_width_pct=width,
            driver_gaps=driver_gaps,
            direction=direction,
            three_p_verdict=central_three_p.verdict.value,
            evidence_confidence=intrinsic.evidence_confidence,
            rationale=rationale,
        )
