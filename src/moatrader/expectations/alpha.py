from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel

if TYPE_CHECKING:
    from moatrader.valuation.base import ValuationResult


class AlphaSignalStatus(StrEnum):
    VALID = "VALID"
    MODEL_NOT_APPLICABLE = "MODEL_NOT_APPLICABLE"
    INVALID_VALUATION = "INVALID_VALUATION"
    MISSING_MARKET_PRICE = "MISSING_MARKET_PRICE"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class CheapSignal(ContractModel):
    """Method-appropriate valuation gap used as the sole production alpha rank.

    Percentiles are deliberately stored alongside the raw gap.  Cross-method
    ranking must use ``method_archetype_percentile`` once it is available; the
    raw gap remains an explanation field and is never mixed with risk signals.
    """

    valuation_method: str = Field(min_length=1)
    economic_archetype: str = Field(min_length=1)
    market_price: Decimal | None = Field(default=None, gt=0)
    primary_fair_value_per_share: Decimal | None = Field(default=None, ge=0)
    raw_expectation_gap: Decimal | None = None
    method_percentile: float | None = Field(default=None, ge=0, le=100)
    method_archetype_percentile: float | None = Field(default=None, ge=0, le=100)
    status: AlphaSignalStatus = AlphaSignalStatus.VALID
    rank_eligible: bool = True

    @model_validator(mode="after")
    def consistent_gap_and_eligibility(self) -> "CheapSignal":
        complete = (
            self.market_price is not None
            and self.primary_fair_value_per_share is not None
            and self.raw_expectation_gap is not None
        )
        if self.status == AlphaSignalStatus.VALID and not complete:
            raise ValueError("VALID Cheap signals require price, fair value, and raw gap")
        if complete:
            expected = self.primary_fair_value_per_share / self.market_price - Decimal(1)
            if abs(self.raw_expectation_gap - expected) > Decimal("0.00000001"):
                raise ValueError("raw_expectation_gap must equal fair_value / market_price - 1")
        elif any(
            value is not None
            for value in (
                self.market_price,
                self.primary_fair_value_per_share,
                self.raw_expectation_gap,
            )
        ):
            raise ValueError("partial Cheap valuation inputs are not allowed")
        if self.rank_eligible != (self.status == AlphaSignalStatus.VALID):
            raise ValueError("rank eligibility must exactly match VALID Cheap status")
        if not self.rank_eligible and (
            self.method_percentile is not None
            or self.method_archetype_percentile is not None
        ):
            raise ValueError("non-rank-eligible Cheap signals cannot carry percentiles")
        return self

    @classmethod
    def from_values(
        cls,
        *,
        valuation_method: str,
        economic_archetype: str,
        market_price: Decimal,
        primary_fair_value_per_share: Decimal,
        status: AlphaSignalStatus = AlphaSignalStatus.VALID,
        rank_eligible: bool = True,
    ) -> "CheapSignal":
        return cls(
            valuation_method=valuation_method,
            economic_archetype=economic_archetype,
            market_price=market_price,
            primary_fair_value_per_share=primary_fair_value_per_share,
            raw_expectation_gap=primary_fair_value_per_share / market_price - Decimal(1),
            status=status,
            rank_eligible=rank_eligible,
        )

    @classmethod
    def from_valuation(
        cls,
        *,
        valuation: "ValuationResult",
        economic_archetype: str,
        market_price: Decimal,
    ) -> "CheapSignal":
        from moatrader.valuation.base import ApplicabilityStatus

        if valuation.fair_value_per_share is None:
            raise ValueError("valuation has no primary fair value")
        route_eligible = valuation.applicability.status == ApplicabilityStatus.ELIGIBLE
        positive_value = valuation.fair_value_per_share > 0
        status = (
            AlphaSignalStatus.VALID
            if route_eligible and positive_value
            else AlphaSignalStatus.INVALID_VALUATION
            if route_eligible
            else AlphaSignalStatus.MODEL_NOT_APPLICABLE
        )
        return cls.from_values(
            valuation_method=valuation.method.value,
            economic_archetype=economic_archetype,
            market_price=market_price,
            primary_fair_value_per_share=max(valuation.fair_value_per_share, Decimal(0)),
            status=status,
            rank_eligible=status == AlphaSignalStatus.VALID,
        )


class AlphaSignal(ContractModel):
    """Production alpha interface.

    Cheap is intentionally the only field.  Improving, 3P, fragility, MOAT,
    and industry evidence belong to confirmation or risk contracts and cannot
    silently become alpha weights through this interface.
    """

    cheap: CheapSignal

    @property
    def rank_value(self) -> float:
        if not self.cheap.rank_eligible:
            raise ValueError("Cheap signal is not rank eligible")
        if self.cheap.method_archetype_percentile is not None:
            return self.cheap.method_archetype_percentile
        if self.cheap.method_percentile is not None:
            return self.cheap.method_percentile
        if self.cheap.raw_expectation_gap is None:
            raise ValueError("rank-eligible Cheap signal has no raw expectation gap")
        return float(self.cheap.raw_expectation_gap)


def assign_method_archetype_percentiles(signals: list[CheapSignal]) -> list[CheapSignal]:
    """Normalize only comparable, eligible signals; preserve raw method gaps."""

    def percentiles(values: list[Decimal]) -> list[float]:
        if len(values) == 1:
            return [50.0]
        order = sorted(range(len(values)), key=lambda index: values[index])
        result = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position + 1
            while end < len(order) and values[order[end]] == values[order[position]]:
                end += 1
            rank = (position + end - 1) / 2.0
            value = 100.0 * rank / (len(values) - 1)
            for offset in range(position, end):
                result[order[offset]] = value
            position = end
        return result

    output = [item.model_copy(deep=True) for item in signals]
    for signal in output:
        signal.method_percentile = None
        signal.method_archetype_percentile = None
    method_groups: dict[str, list[int]] = {}
    archetype_groups: dict[tuple[str, str], list[int]] = {}
    for index, signal in enumerate(output):
        if not signal.rank_eligible:
            continue
        method_groups.setdefault(signal.valuation_method, []).append(index)
        archetype_groups.setdefault(
            (signal.valuation_method, signal.economic_archetype), []
        ).append(index)
    for indices in method_groups.values():
        values = [output[index].raw_expectation_gap for index in indices]
        if any(value is None for value in values):
            raise ValueError("rank-eligible Cheap signal has no raw expectation gap")
        for index, percentile in zip(indices, percentiles(values), strict=True):
            output[index].method_percentile = percentile
    for indices in archetype_groups.values():
        values = [output[index].raw_expectation_gap for index in indices]
        if any(value is None for value in values):
            raise ValueError("rank-eligible Cheap signal has no raw expectation gap")
        for index, percentile in zip(indices, percentiles(values), strict=True):
            output[index].method_archetype_percentile = percentile
    return output
