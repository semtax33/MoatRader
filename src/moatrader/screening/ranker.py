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
    moat_rank_weight: Decimal = Field(default=Decimal("0.50"), ge=0, le=1)
    value_rank_weight: Decimal = Field(default=Decimal("0.50"), ge=0, le=1)

    @model_validator(mode="after")
    def rank_weights_sum_to_one(self) -> "SelectorConfig":
        if self.moat_rank_weight + self.value_rank_weight != Decimal(1):
            raise ValueError("moat_rank_weight and value_rank_weight must sum to 1")
        return self


class RankedCandidate(ContractModel):
    issuer_id: str
    ticker: str
    price_to_dcf: Decimal
    margin_of_safety: Decimal
    moat_score: Decimal
    quality_value_score: Decimal
    moat_percentile: Decimal = Decimal(0)
    value_percentile: Decimal = Decimal(0)
    valuation_as_of: datetime
    price_as_of: datetime


class ValueMoatRanker:
    """Transparent screen, not an alpha claim; backtesting must validate thresholds."""

    def __init__(self, config: SelectorConfig | None = None) -> None:
        self.config = config or SelectorConfig()

    def rank(self, candidates: list[CandidateInput]) -> list[RankedCandidate]:
        eligible: list[tuple[CandidateInput, Decimal, Decimal]] = []
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
            eligible.append((candidate, price_to_dcf, margin))

        moat_percentiles = self._percentiles([item[0].moat_score for item in eligible])
        value_percentiles = self._percentiles([item[2] for item in eligible])
        ranked: list[RankedCandidate] = []
        for (candidate, price_to_dcf, margin), moat_percentile, value_percentile in zip(
            eligible,
            moat_percentiles,
            value_percentiles,
            strict=True,
        ):
            score = (
                self.config.moat_rank_weight * moat_percentile
                + self.config.value_rank_weight * value_percentile
            )
            ranked.append(
                RankedCandidate(
                    issuer_id=candidate.issuer_id,
                    ticker=candidate.ticker,
                    price_to_dcf=price_to_dcf,
                    margin_of_safety=margin,
                    moat_score=candidate.moat_score,
                    quality_value_score=score,
                    moat_percentile=moat_percentile,
                    value_percentile=value_percentile,
                    valuation_as_of=candidate.valuation_as_of,
                    price_as_of=candidate.price_as_of,
                )
            )
        return sorted(ranked, key=lambda item: (item.quality_value_score, -item.price_to_dcf), reverse=True)

    @staticmethod
    def _percentiles(values: list[Decimal]) -> list[Decimal]:
        if not values:
            return []
        ordered = sorted(range(len(values)), key=lambda index: values[index])
        result = [Decimal(0)] * len(values)
        cursor = 0
        while cursor < len(ordered):
            end = cursor + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
                end += 1
            average_rank = (Decimal(cursor + 1) + Decimal(end)) / Decimal(2)
            percentile = average_rank / Decimal(len(values))
            for position in range(cursor, end):
                result[ordered[position]] = percentile
            cursor = end
        return result
