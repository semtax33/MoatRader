from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Any, Protocol, TypeVar

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class ValuationMethod(StrEnum):
    ECONOMIC_FCFF = "ECONOMIC_FCFF"
    NORMALIZED_FCFF = "NORMALIZED_FCFF"
    RIM = "RIM"
    RNPV = "RNPV"
    SCENARIO_DCF = "SCENARIO_DCF"
    APV = "APV"
    NAV = "NAV"
    SOTP = "SOTP"
    REVERSE_DCF = "REVERSE_DCF"
    PB_ROE_CROSS_CHECK = "PB_ROE_CROSS_CHECK"
    NORMALIZED_EARNINGS = "NORMALIZED_EARNINGS"
    # Legacy route names remain readable while the new router is rolled out.
    STEADY_STATE_FCFF = "STEADY_STATE_FCFF"
    EXCESS_RETURN_EQUITY = "EXCESS_RETURN_EQUITY"
    BIOTECH_RNPV = "BIOTECH_RNPV"
    NARRATIVE_DCF = "NARRATIVE_DCF"
    FAILURE_ADJUSTED_DCF = "FAILURE_ADJUSTED_DCF"


class ApplicabilityStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INAPPLICABLE = "INAPPLICABLE"


class ModelApplicability(ContractModel):
    method: ValuationMethod
    status: ApplicabilityStatus
    required_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def fail_closed(self) -> "ModelApplicability":
        if self.status == ApplicabilityStatus.ELIGIBLE and self.missing_fields:
            raise ValueError("ELIGIBLE valuation cannot have missing required fields")
        if self.status == ApplicabilityStatus.INSUFFICIENT_DATA and not self.missing_fields:
            raise ValueError("INSUFFICIENT_DATA must identify missing fields")
        return self


class ValuationResult(ContractModel):
    """Method-neutral result. Primary and cross-check values are never averaged."""

    method: ValuationMethod
    applicability: ModelApplicability
    enterprise_value: Decimal | None = None
    equity_value: Decimal | None = None
    fair_value_per_share: Decimal | None = None
    downside_value_per_share: Decimal | None = None
    base_value_per_share: Decimal | None = None
    upside_value_per_share: Decimal | None = None
    assumption_confidence: Decimal | None = Field(default=None, ge=0, le=1)
    provenance: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    cross_check_method: ValuationMethod | None = None
    cross_check_value_per_share: Decimal | None = None

    @model_validator(mode="after")
    def consistent_result(self) -> "ValuationResult":
        if self.applicability.method != self.method:
            raise ValueError("applicability method must match valuation result method")
        values = (
            self.equity_value,
            self.fair_value_per_share,
            self.downside_value_per_share,
            self.base_value_per_share,
            self.upside_value_per_share,
        )
        if self.applicability.status in {
            ApplicabilityStatus.ELIGIBLE,
            ApplicabilityStatus.REVIEW_REQUIRED,
        } and any(value is None for value in values):
            raise ValueError("valued results require equity and downside/base/upside values")
        if self.base_value_per_share != self.fair_value_per_share:
            raise ValueError("fair_value_per_share must be the primary base value")
        if (
            self.downside_value_per_share is not None
            and self.base_value_per_share is not None
            and self.upside_value_per_share is not None
            and not self.downside_value_per_share
            <= self.base_value_per_share
            <= self.upside_value_per_share
        ):
            raise ValueError("valuation scenarios must be ordered downside <= base <= upside")
        if (self.cross_check_method is None) != (self.cross_check_value_per_share is None):
            raise ValueError("cross-check method and value must be supplied together")
        return self


AssumptionsT = TypeVar("AssumptionsT")


class ValuationEngine(Protocol[AssumptionsT]):
    def value(self, assumptions: AssumptionsT) -> ValuationResult: ...


def eligible(method: ValuationMethod, required_fields: list[str]) -> ModelApplicability:
    return ModelApplicability(
        method=method,
        status=ApplicabilityStatus.ELIGIBLE,
        required_fields=required_fields,
    )
