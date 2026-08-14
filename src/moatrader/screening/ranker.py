from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class CandidateInput(ContractModel):
    issuer_id: str
    ticker: str
    current_price: Decimal = Field(gt=0)
    dcf_fair_value: Decimal = Field(gt=0)
    moat_score: Decimal = Field(ge=0, le=10)
    model_confidence: Decimal = Field(ge=0, le=1)
    document_coverage: Decimal = Field(ge=0, le=1)
    valuation_as_of: datetime
    price_as_of: datetime

    @model_validator(mode="after")
    def timestamps_are_pit_safe(self) -> "CandidateInput":
        for name, value in (("valuation_as_of", self.valuation_as_of), ("price_as_of", self.price_as_of)):
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        return self


class SelectorConfig(ContractModel):
    minimum_moat_score: Decimal = Field(default=Decimal("5"), ge=0, le=10)
    minimum_margin_of_safety: Decimal = Field(default=Decimal("0.20"), ge=-1)
    minimum_model_confidence: Decimal = Field(default=Decimal("0.50"), ge=0, le=1)
    minimum_document_coverage: Decimal = Field(default=Decimal("0.50"), ge=0, le=1)


class RankedCandidate(ContractModel):
    issuer_id: str
    ticker: str
    price_to_dcf: Decimal
    margin_of_safety: Decimal
    moat_score: Decimal
    quality_value_score: Decimal
    valuation_as_of: datetime
    price_as_of: datetime


class ValueMoatRanker:
    """Transparent screen, not an alpha claim; backtesting must validate thresholds."""

    def __init__(self, config: SelectorConfig | None = None) -> None:
        self.config = config or SelectorConfig()

    def rank(self, candidates: list[CandidateInput]) -> list[RankedCandidate]:
        ranked: list[RankedCandidate] = []
        for candidate in candidates:
            price_to_dcf = candidate.current_price / candidate.dcf_fair_value
            margin = Decimal(1) - price_to_dcf
            if candidate.moat_score < self.config.minimum_moat_score:
                continue
            if margin < self.config.minimum_margin_of_safety:
                continue
            if candidate.model_confidence < self.config.minimum_model_confidence:
                continue
            if candidate.document_coverage < self.config.minimum_document_coverage:
                continue
            score = (
                (candidate.moat_score / Decimal(10))
                * max(Decimal(0), margin)
                * candidate.model_confidence
                * candidate.document_coverage
            )
            ranked.append(
                RankedCandidate(
                    issuer_id=candidate.issuer_id,
                    ticker=candidate.ticker,
                    price_to_dcf=price_to_dcf,
                    margin_of_safety=margin,
                    moat_score=candidate.moat_score,
                    quality_value_score=score,
                    valuation_as_of=candidate.valuation_as_of,
                    price_as_of=candidate.price_as_of,
                )
            )
        return sorted(ranked, key=lambda item: (item.quality_value_score, -item.price_to_dcf), reverse=True)

