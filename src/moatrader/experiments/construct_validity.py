from __future__ import annotations

from datetime import datetime
from math import sqrt

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


def _aware(value: datetime, *, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


class FundamentalSignal(ContractModel):
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    signal_at: datetime
    moat_score: float = Field(ge=0, le=10)
    margin_durability_score: float = Field(ge=0, le=100)
    fragility_score: float = Field(ge=0, le=100)
    baseline_margin: float | None = None

    @model_validator(mode="after")
    def timestamp_is_aware(self) -> "FundamentalSignal":
        _aware(self.signal_at, field="signal_at")
        return self


class FundamentalOutcome(ContractModel):
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    available_at: datetime
    future_roic: float | None = None
    future_margin: float | None = None
    earnings_miss: bool = False
    margin_collapse: bool = False
    negative_fcf: bool = False
    leverage_deterioration: bool = False
    capital_raise: bool = False

    @model_validator(mode="after")
    def timestamp_is_aware(self) -> "FundamentalOutcome":
        _aware(self.available_at, field="available_at")
        return self

    @property
    def deterioration_event(self) -> bool:
        return any(
            (
                self.earnings_miss,
                self.margin_collapse,
                self.negative_fcf,
                self.leverage_deterioration,
                self.capital_raise,
            )
        )


class ConstructValidityReport(ContractModel):
    schema_version: str = "v7-fundamental-construct-validity/1"
    matched_count: int = Field(gt=0)
    moat_future_roic_spearman: float | None = Field(default=None, ge=-1, le=1)
    durability_negative_margin_change_spearman: float | None = Field(default=None, ge=-1, le=1)
    high_fragility_count: int = Field(ge=0)
    lower_fragility_count: int = Field(ge=0)
    high_fragility_event_rate: float | None = Field(default=None, ge=0, le=1)
    lower_fragility_event_rate: float | None = Field(default=None, ge=0, le=1)
    fragility_event_rate_difference: float | None = Field(default=None, ge=-1, le=1)
    return_data_accessed: bool = False

    @model_validator(mode="after")
    def return_blind(self) -> "ConstructValidityReport":
        if self.return_data_accessed:
            raise ValueError("construct validity uses fundamentals, never returns")
        return self


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = (index + 1 + end) / 2
        for position in range(index, end):
            ranks[ordered[position][0]] = average
        index = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_rank, right_rank, strict=True)
    )
    left_scale = sum((value - left_mean) ** 2 for value in left_rank)
    right_scale = sum((value - right_mean) ** 2 for value in right_rank)
    denominator = sqrt(left_scale * right_scale)
    return numerator / denominator if denominator else None


def evaluate_construct_validity(
    *,
    signals: list[FundamentalSignal],
    outcomes: list[FundamentalOutcome],
    high_fragility_threshold: float = 70.0,
) -> ConstructValidityReport:
    signal_by_ticker = {item.ticker: item for item in signals}
    outcome_by_ticker = {item.ticker: item for item in outcomes}
    if len(signal_by_ticker) != len(signals) or len(outcome_by_ticker) != len(outcomes):
        raise ValueError("construct-validity inputs require unique tickers")
    matched = sorted(set(signal_by_ticker) & set(outcome_by_ticker))
    if not matched:
        raise ValueError("construct-validity inputs have no matching tickers")
    for ticker in matched:
        if outcome_by_ticker[ticker].available_at <= signal_by_ticker[ticker].signal_at:
            raise ValueError(f"fundamental outcome must become available after signal: {ticker}")

    moat_pairs = [
        (signal_by_ticker[ticker].moat_score, outcome_by_ticker[ticker].future_roic)
        for ticker in matched
        if outcome_by_ticker[ticker].future_roic is not None
    ]
    margin_pairs = [
        (
            signal_by_ticker[ticker].margin_durability_score,
            -abs(
                outcome_by_ticker[ticker].future_margin
                - signal_by_ticker[ticker].baseline_margin
            ),
        )
        for ticker in matched
        if outcome_by_ticker[ticker].future_margin is not None
        and signal_by_ticker[ticker].baseline_margin is not None
    ]
    high = [
        outcome_by_ticker[ticker].deterioration_event
        for ticker in matched
        if signal_by_ticker[ticker].fragility_score >= high_fragility_threshold
    ]
    lower = [
        outcome_by_ticker[ticker].deterioration_event
        for ticker in matched
        if signal_by_ticker[ticker].fragility_score < high_fragility_threshold
    ]
    high_rate = sum(high) / len(high) if high else None
    lower_rate = sum(lower) / len(lower) if lower else None
    return ConstructValidityReport(
        matched_count=len(matched),
        moat_future_roic_spearman=(
            _correlation(
                [pair[0] for pair in moat_pairs],
                [float(pair[1]) for pair in moat_pairs],
            )
            if moat_pairs
            else None
        ),
        durability_negative_margin_change_spearman=(
            _correlation(
                [pair[0] for pair in margin_pairs],
                [pair[1] for pair in margin_pairs],
            )
            if margin_pairs
            else None
        ),
        high_fragility_count=len(high),
        lower_fragility_count=len(lower),
        high_fragility_event_rate=high_rate,
        lower_fragility_event_rate=lower_rate,
        fragility_event_rate_difference=(
            high_rate - lower_rate
            if high_rate is not None and lower_rate is not None
            else None
        ),
    )
