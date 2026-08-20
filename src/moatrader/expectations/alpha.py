from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

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
    INSUFFICIENT_REFERENCE_CLASS = "INSUFFICIENT_REFERENCE_CLASS"


class ValuationTrustPolicy(ContractModel):
    """Frozen, method-neutral gate. Failed valuations remain explainable but unrankable."""

    contract_version: Literal["valuation-trust/2"] = "valuation-trust/2"
    min_assumption_confidence: Decimal = Field(default=Decimal("0.50"), ge=0, le=1)
    max_warning_count: int = Field(default=3, ge=0)
    require_screening_eligible: bool = True
    warning_count_basis: Literal["STRUCTURED_TRUST_WARNINGS"] = (
        "STRUCTURED_TRUST_WARNINGS"
    )


class UnifiedValueReferenceLevel(StrEnum):
    METHOD_ARCHETYPE = "METHOD_ARCHETYPE"
    METHOD = "METHOD"
    MODEL_FAMILY = "MODEL_FAMILY"


MODEL_FAMILY_BY_METHOD = {
    "ECONOMIC_FCFF": "OPERATING_CASH_FLOW",
    "NORMALIZED_FCFF": "OPERATING_CASH_FLOW",
    "SCENARIO_DCF": "OPERATING_CASH_FLOW",
    "APV": "OPERATING_CASH_FLOW",
    "RIM": "RESIDUAL_INCOME",
    "RNPV": "PIPELINE_PROBABILITY_WEIGHTED",
    "NAV": "ASSET_AND_SUM_OF_PARTS",
    "SOTP": "ASSET_AND_SUM_OF_PARTS",
}


class UnifiedValueNormalizationPolicy(ContractModel):
    """Return-blind hierarchy; it never falls back across model families."""

    contract_version: Literal["unified-value-normalization/2"] = (
        "unified-value-normalization/2"
    )
    min_reference_class_size: int = Field(default=20, ge=2)
    small_class_action: Literal["HIERARCHICAL_FALLBACK_THEN_UNRANKABLE"] = (
        "HIERARCHICAL_FALLBACK_THEN_UNRANKABLE"
    )
    parent_class_fallback: Literal[True] = True
    reference_class_hierarchy: tuple[
        Literal["METHOD_ARCHETYPE", "METHOD", "MODEL_FAMILY"], ...
    ] = ("METHOD_ARCHETYPE", "METHOD", "MODEL_FAMILY")


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
    raw_value_gap: Decimal | None = None
    method_percentile: float | None = Field(default=None, ge=0, le=100)
    method_archetype_percentile: float | None = Field(default=None, ge=0, le=100)
    reference_class: str = Field(min_length=1)
    reference_class_size: int = Field(default=0, ge=0)
    method_archetype_reference_size: int = Field(default=0, ge=0)
    method_reference_size: int = Field(default=0, ge=0)
    model_family_reference_size: int = Field(default=0, ge=0)
    normalization_level: UnifiedValueReferenceLevel | None = None
    normalization_fallback_used: bool = False
    assumption_confidence: Decimal | None = Field(default=None, ge=0, le=1)
    warning_count: int = Field(default=0, ge=0)
    disclosure_count: int = Field(default=0, ge=0)
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
            legacy_gap = values.pop("raw_expectation_gap", None)
            current_gap = values.get("raw_value_gap")
            if current_gap is None and legacy_gap is not None:
                values["raw_value_gap"] = legacy_gap
            elif legacy_gap is not None and legacy_gap != current_gap:
                raise ValueError("raw_value_gap conflicts with legacy raw_expectation_gap")
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

    @property
    def raw_expectation_gap(self) -> Decimal | None:
        """Deprecated read alias. New artifacts use raw_value_gap."""

        return self.raw_value_gap

    @model_validator(mode="after")
    def consistent_gap_and_eligibility(self) -> "CheapSignal":
        complete = (
            self.market_price is not None
            and self.primary_fair_value_per_share is not None
            and self.raw_value_gap is not None
        )
        if self.status == AlphaSignalStatus.VALID and not complete:
            raise ValueError("VALID Cheap signals require price, fair value, and raw gap")
        if complete:
            expected = self.primary_fair_value_per_share / self.market_price - Decimal(1)
            if abs(self.raw_value_gap - expected) > Decimal("0.00000001"):
                raise ValueError("raw_value_gap must equal fair_value / market_price - 1")
        elif any(
            value is not None
            for value in (
                self.market_price,
                self.primary_fair_value_per_share,
                self.raw_value_gap,
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
        disclosure_count: int = 0,
        trust_reason_codes: list[str] | None = None,
        reference_class: str | None = None,
    ) -> "CheapSignal":
        return cls(
            valuation_method=valuation_method,
            economic_archetype=economic_archetype,
            market_price=market_price,
            primary_fair_value_per_share=primary_fair_value_per_share,
            raw_value_gap=primary_fair_value_per_share / market_price - Decimal(1),
            reference_class=(
                reference_class or f"{valuation_method}::{economic_archetype}"
            ),
            assumption_confidence=assumption_confidence,
            warning_count=warning_count,
            disclosure_count=disclosure_count,
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
        if not positive_value:
            trust_failures.append("NON_POSITIVE_FAIR_VALUE")
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
            disclosure_count=len(valuation.disclosures),
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


def assign_method_archetype_percentiles(
    signals: list[CheapSignal],
    policy: UnifiedValueNormalizationPolicy | None = None,
) -> list[CheapSignal]:
    """Normalize trusted values through a frozen, return-blind class hierarchy."""

    normalization = policy or UnifiedValueNormalizationPolicy()

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
            reference_class_size=0,
            method_archetype_reference_size=0,
            method_reference_size=0,
            model_family_reference_size=0,
            normalization_level=None,
            normalization_fallback_used=False,
        )
        output.append(CheapSignal.model_validate(data))

    initially_trusted = [index for index, signal in enumerate(output) if signal.rank_eligible]
    method_archetype_groups: dict[tuple[str, str], list[int]] = {}
    method_groups: dict[str, list[int]] = {}
    family_groups: dict[str, list[int]] = {}
    for index in initially_trusted:
        signal = output[index]
        family = MODEL_FAMILY_BY_METHOD.get(
            signal.valuation_method,
            f"METHOD_ONLY_{signal.valuation_method}",
        )
        method_archetype_groups.setdefault(
            (signal.valuation_method, signal.economic_archetype), []
        ).append(index)
        method_groups.setdefault(signal.valuation_method, []).append(index)
        family_groups.setdefault(family, []).append(index)

    selected: dict[int, tuple[UnifiedValueReferenceLevel, object, list[int]]] = {}
    for index, signal in enumerate(output):
        family = MODEL_FAMILY_BY_METHOD.get(
            signal.valuation_method,
            f"METHOD_ONLY_{signal.valuation_method}",
        )
        method_archetype_key = (signal.valuation_method, signal.economic_archetype)
        local_indices = method_archetype_groups.get(method_archetype_key, [])
        parent_indices = method_groups.get(signal.valuation_method, [])
        family_indices = family_groups.get(family, [])
        data = signal.model_dump(exclude={"unified_value_score"})
        data.update(
            method_archetype_reference_size=len(local_indices),
            method_reference_size=len(parent_indices),
            model_family_reference_size=len(family_indices),
        )
        if not signal.rank_eligible:
            output[index] = CheapSignal.model_validate(data)
            continue

        candidates = (
            (
                UnifiedValueReferenceLevel.METHOD_ARCHETYPE,
                method_archetype_key,
                local_indices,
                f"{signal.valuation_method}::{signal.economic_archetype}",
            ),
            (
                UnifiedValueReferenceLevel.METHOD,
                signal.valuation_method,
                parent_indices,
                f"METHOD::{signal.valuation_method}",
            ),
            (
                UnifiedValueReferenceLevel.MODEL_FAMILY,
                family,
                family_indices,
                f"MODEL_FAMILY::{family}",
            ),
        )
        chosen = next(
            (
                candidate
                for candidate in candidates
                if candidate[0].value in normalization.reference_class_hierarchy
                and len(candidate[2]) >= normalization.min_reference_class_size
            ),
            None,
        )
        if chosen is None:
            attempted = [
                candidate
                for candidate in candidates
                if candidate[0].value in normalization.reference_class_hierarchy
            ]
            last = attempted[-1]
            data.update(
                reference_class=last[3],
                reference_class_size=len(last[2]),
                normalization_level=last[0],
                normalization_fallback_used=len(attempted) > 1,
                status=AlphaSignalStatus.INSUFFICIENT_REFERENCE_CLASS,
                rank_eligible=False,
                trust_reason_codes=[
                    *signal.trust_reason_codes,
                    f"REFERENCE_CLASS_N_LT_{normalization.min_reference_class_size}",
                ],
            )
        else:
            level, key, indices, label = chosen
            data.update(
                reference_class=label,
                reference_class_size=len(indices),
                normalization_level=level,
                normalization_fallback_used=(
                    level != UnifiedValueReferenceLevel.METHOD_ARCHETYPE
                ),
            )
            selected[index] = (level, key, indices)
        output[index] = CheapSignal.model_validate(data)

    for method, indices in method_groups.items():
        eligible_indices = [index for index in indices if index in selected]
        if not eligible_indices:
            continue
        values = [output[index].raw_value_gap for index in indices]
        if any(value is None for value in values):
            raise ValueError(f"trusted {method} signal has no raw value gap")
        ranks = dict(zip(indices, percentiles(values), strict=True))
        for index in eligible_indices:
            data = output[index].model_dump(exclude={"unified_value_score"})
            data["method_percentile"] = ranks[index]
            output[index] = CheapSignal.model_validate(data)

    selected_groups: dict[tuple[UnifiedValueReferenceLevel, object], list[int]] = {}
    reference_members: dict[tuple[UnifiedValueReferenceLevel, object], list[int]] = {}
    for index, (level, key, indices) in selected.items():
        selected_groups.setdefault((level, key), []).append(index)
        reference_members[(level, key)] = indices
    for group_key, selected_indices in selected_groups.items():
        members = reference_members[group_key]
        values = [output[index].raw_value_gap for index in members]
        if any(value is None for value in values):
            raise ValueError("trusted reference-class signal has no raw value gap")
        ranks = dict(zip(members, percentiles(values), strict=True))
        for index in selected_indices:
            data = output[index].model_dump(exclude={"unified_value_score"})
            data["method_archetype_percentile"] = ranks[index]
            output[index] = CheapSignal.model_validate(data)
    return output
