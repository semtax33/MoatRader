from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import Field, computed_field, model_validator

from moatrader.canonical.models import ContractModel

if TYPE_CHECKING:
    from moatrader.valuation.base import ValuationResult


class AlphaSignalStatus(StrEnum):
    VALID = "VALID"
    MODEL_NOT_APPLICABLE = "MODEL_NOT_APPLICABLE"
    INVALID_VALUATION = "INVALID_VALUATION"
    MISSING_MARKET_PRICE = "MISSING_MARKET_PRICE"
    NOT_COMPARABLE = "NOT_COMPARABLE"
    UNTRUSTED_VALUATION = "UNTRUSTED_VALUATION"


class ValuationTrustPolicy(ContractModel):
    """Frozen, method-neutral gate. Failed valuations remain explainable but unrankable."""

    min_assumption_confidence: Decimal = Field(default=Decimal("0.50"), ge=0, le=1)
    max_warning_count: int = Field(default=3, ge=0)
    require_screening_eligible: bool = True


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
    reference_class: str = Field(min_length=1)
    assumption_confidence: Decimal | None = Field(default=None, ge=0, le=1)
    warning_count: int = Field(default=0, ge=0)
    trust_reason_codes: list[str] = Field(default_factory=list)
    status: AlphaSignalStatus = AlphaSignalStatus.VALID
    rank_eligible: bool = True

    @model_validator(mode="before")
    @classmethod
    def default_reference_class(cls, values: object) -> object:
        if isinstance(values, dict):
            values = dict(values)
            supplied_score = values.pop("unified_value_score", None)
            percentile = values.get("method_archetype_percentile")
            if supplied_score is not None and supplied_score != percentile:
                raise ValueError(
                    "unified value score must equal method-archetype percentile"
                )
        if isinstance(values, dict) and not values.get("reference_class"):
            method = values.get("valuation_method")
            archetype = values.get("economic_archetype")
            if method and archetype:
                values["reference_class"] = f"{method}::{archetype}"
        return values

    @computed_field(return_type=float | None)
    @property
    def unified_value_score(self) -> float | None:
        return self.method_archetype_percentile

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
        if self.rank_eligible and self.trust_reason_codes:
            raise ValueError("rank-eligible Cheap signals cannot carry trust failures")
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
        assumption_confidence: Decimal | None = None,
        warning_count: int = 0,
        trust_reason_codes: list[str] | None = None,
        reference_class: str | None = None,
    ) -> "CheapSignal":
        return cls(
            valuation_method=valuation_method,
            economic_archetype=economic_archetype,
            market_price=market_price,
            primary_fair_value_per_share=primary_fair_value_per_share,
            raw_expectation_gap=primary_fair_value_per_share / market_price - Decimal(1),
            reference_class=(
                reference_class or f"{valuation_method}::{economic_archetype}"
            ),
            assumption_confidence=assumption_confidence,
            warning_count=warning_count,
            trust_reason_codes=trust_reason_codes or [],
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
        trust_policy: ValuationTrustPolicy | None = None,
    ) -> "CheapSignal":
        from moatrader.valuation.base import ApplicabilityStatus

        if valuation.fair_value_per_share is None:
            raise ValueError("valuation has no primary fair value")
        policy = trust_policy or ValuationTrustPolicy()
        route_eligible = valuation.applicability.status == ApplicabilityStatus.ELIGIBLE
        positive_value = valuation.fair_value_per_share > 0
        trust_failures: list[str] = []
        if policy.require_screening_eligible and not bool(
            valuation.metadata.get("screening_eligible", True)
        ):
            trust_failures.append("SCREENING_INELIGIBLE")
        if (
            valuation.assumption_confidence is None
            or valuation.assumption_confidence < policy.min_assumption_confidence
        ):
            trust_failures.append("LOW_OR_MISSING_ASSUMPTION_CONFIDENCE")
        if len(valuation.warnings) > policy.max_warning_count:
            trust_failures.append("TOO_MANY_VALUATION_WARNINGS")
        status = (
            AlphaSignalStatus.VALID
            if route_eligible and positive_value and not trust_failures
            else AlphaSignalStatus.INVALID_VALUATION
            if route_eligible and not positive_value
            else AlphaSignalStatus.UNTRUSTED_VALUATION
            if route_eligible and trust_failures
            else AlphaSignalStatus.MODEL_NOT_APPLICABLE
        )
        return cls.from_values(
            valuation_method=valuation.method.value,
            economic_archetype=economic_archetype,
            market_price=market_price,
            primary_fair_value_per_share=max(valuation.fair_value_per_share, Decimal(0)),
            assumption_confidence=valuation.assumption_confidence,
            warning_count=len(valuation.warnings),
            trust_reason_codes=trust_failures,
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
        if self.cheap.unified_value_score is None:
            raise ValueError("Cheap signal is not normalized to its route reference class")
        return self.cheap.unified_value_score


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

    output: list[CheapSignal] = []
    for signal in signals:
        data = signal.model_dump(exclude={"unified_value_score"})
        data.update(
            method_percentile=None,
            method_archetype_percentile=None,
            reference_class=f"{signal.valuation_method}::{signal.economic_archetype}",
        )
        output.append(CheapSignal.model_validate(data))
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
            data = output[index].model_dump(exclude={"unified_value_score"})
            data["method_percentile"] = percentile
            output[index] = CheapSignal.model_validate(data)
    for indices in archetype_groups.values():
        values = [output[index].raw_expectation_gap for index in indices]
        if any(value is None for value in values):
            raise ValueError("rank-eligible Cheap signal has no raw expectation gap")
        for index, percentile in zip(indices, percentiles(values), strict=True):
            data = output[index].model_dump(exclude={"unified_value_score"})
            data["method_archetype_percentile"] = percentile
            output[index] = CheapSignal.model_validate(data)
    return output
