from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.expectations.gap import ExpectationGapDirection, ExpectationGapEvaluation


class OpportunityCandidate(ContractModel):
    issuer_id: str
    ticker: str
    evaluation: ExpectationGapEvaluation
    valuation_as_of: datetime
    price_as_of: datetime

    @model_validator(mode="after")
    def pit_safe(self) -> "OpportunityCandidate":
        for field in ("valuation_as_of", "price_as_of"):
            value = getattr(self, field)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{field} must be timezone-aware")
        if self.valuation_as_of > self.price_as_of:
            raise ValueError("valuation evidence cutoff must not be after market price")
        return self


class OpportunitySelectorConfig(ContractModel):
    minimum_central_value_gap: Decimal = Decimal(0)
    minimum_downside_value_gap: Decimal = Decimal("-0.30")
    maximum_range_width_pct: Decimal = Field(default=Decimal("3.0"), gt=0)


class RankedOpportunity(ContractModel):
    issuer_id: str
    ticker: str
    direction: ExpectationGapDirection
    central_value_gap: Decimal
    downside_value_gap: Decimal
    valuation_range_width_pct: Decimal
    rank_key: tuple[Decimal, Decimal, Decimal]
    valuation_as_of: datetime
    price_as_of: datetime


class ExpectationOpportunityRanker:
    """Ranks expectation gaps; MOAT score and confidence multipliers are absent."""

    def __init__(self, config: OpportunitySelectorConfig | None = None) -> None:
        self.config = config or OpportunitySelectorConfig()

    def rank(self, candidates: list[OpportunityCandidate]) -> list[RankedOpportunity]:
        ranked: list[RankedOpportunity] = []
        for candidate in candidates:
            gap = candidate.evaluation
            if gap.direction != ExpectationGapDirection.FAVORABLE:
                continue
            if gap.central_value_gap < self.config.minimum_central_value_gap:
                continue
            if gap.downside_value_gap < self.config.minimum_downside_value_gap:
                continue
            if gap.valuation_range_width_pct > self.config.maximum_range_width_pct:
                continue
            key = (
                gap.central_value_gap,
                gap.downside_value_gap,
                -gap.valuation_range_width_pct,
            )
            ranked.append(
                RankedOpportunity(
                    issuer_id=candidate.issuer_id,
                    ticker=candidate.ticker,
                    direction=gap.direction,
                    central_value_gap=gap.central_value_gap,
                    downside_value_gap=gap.downside_value_gap,
                    valuation_range_width_pct=gap.valuation_range_width_pct,
                    rank_key=key,
                    valuation_as_of=candidate.valuation_as_of,
                    price_as_of=candidate.price_as_of,
                )
            )
        return sorted(ranked, key=lambda item: item.rank_key, reverse=True)
