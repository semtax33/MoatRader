from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.apv import ApvAssumptions, ApvEngine
from moatrader.valuation.base import ApplicabilityStatus, ValuationMethod, ValuationResult
from moatrader.valuation.common_engines import (
    CommonEconomicFcffEngine,
    CommonRnpvEngine,
    EconomicFcffScenarioSet,
    RnpvScenarioSet,
)
from moatrader.valuation.nav import NavAssumptions, NavEngine
from moatrader.valuation.rim import CommonRimEngine, RimScenarioSet
from moatrader.valuation.router import REQUIRED_DATA, ValuationRoute
from moatrader.valuation.scenario_dcf import ScenarioDcfAssumptions, ScenarioDcfEngine
from moatrader.valuation.sotp import SotpAssumptions, SotpEngine


ROUTED_VALUATION_INPUT_VERSION = "routed-valuation-input/1"


class RoutedValuationInput(ContractModel):
    """PIT, price-free input for exactly one routed valuation engine."""

    schema_version: Literal["routed-valuation-input/1"] = ROUTED_VALUATION_INPUT_VERSION
    issuer_id: str = Field(min_length=1)
    as_of: date
    method: ValuationMethod
    assumptions: dict[str, Any]
    source_refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_sources(self) -> "RoutedValuationInput":
        if len(self.source_refs) != len(set(self.source_refs)):
            raise ValueError("routed valuation source_refs must be unique")
        return self


AssumptionSet = (
    EconomicFcffScenarioSet
    | RimScenarioSet
    | RnpvScenarioSet
    | ScenarioDcfAssumptions
    | ApvAssumptions
    | NavAssumptions
    | SotpAssumptions
)


@dataclass(frozen=True)
class PreparedValuationInput:
    envelope: RoutedValuationInput
    assumptions: AssumptionSet
    actual_engine: str

    @property
    def available_data(self) -> tuple[str, ...]:
        return REQUIRED_DATA[self.envelope.method]


class ExecutionStatus(StrEnum):
    VALUED = "VALUED"
    ROUTE_NOT_APPLICABLE = "ROUTE_NOT_APPLICABLE"
    MISSING_METHOD_INPUT = "MISSING_METHOD_INPUT"
    INVALID_METHOD_INPUT = "INVALID_METHOD_INPUT"
    INPUT_ROUTE_MISMATCH = "INPUT_ROUTE_MISMATCH"
    VALUATION_ERROR = "VALUATION_ERROR"


class RoutedValuationExecution(ContractModel):
    route: ValuationRoute
    status: ExecutionStatus
    actual_engine: str | None = None
    valuation: ValuationResult | None = None
    input_source_refs: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def fail_closed_and_method_consistent(self) -> "RoutedValuationExecution":
        if self.status == ExecutionStatus.VALUED:
            if self.valuation is None or not self.actual_engine:
                raise ValueError("VALUED execution requires valuation and actual engine")
            if self.valuation.method != self.route.primary_method:
                raise ValueError("valuation result method must match routed primary method")
        elif self.valuation is not None:
            raise ValueError("failed execution cannot carry a valuation")
        if self.status != ExecutionStatus.VALUED and not self.reason_codes:
            raise ValueError("failed execution must carry reason codes")
        return self


_ASSUMPTION_MODELS: dict[ValuationMethod, type[ContractModel]] = {
    ValuationMethod.ECONOMIC_FCFF: EconomicFcffScenarioSet,
    ValuationMethod.NORMALIZED_FCFF: EconomicFcffScenarioSet,
    ValuationMethod.RIM: RimScenarioSet,
    ValuationMethod.RNPV: RnpvScenarioSet,
    ValuationMethod.SCENARIO_DCF: ScenarioDcfAssumptions,
    ValuationMethod.APV: ApvAssumptions,
    ValuationMethod.NAV: NavAssumptions,
    ValuationMethod.SOTP: SotpAssumptions,
}


_ENGINE_NAMES: dict[ValuationMethod, str] = {
    ValuationMethod.ECONOMIC_FCFF: "CommonEconomicFcffEngine",
    ValuationMethod.NORMALIZED_FCFF: "CommonEconomicFcffEngine",
    ValuationMethod.RIM: "CommonRimEngine",
    ValuationMethod.RNPV: "CommonRnpvEngine",
    ValuationMethod.SCENARIO_DCF: "ScenarioDcfEngine",
    ValuationMethod.APV: "ApvEngine",
    ValuationMethod.NAV: "NavEngine",
    ValuationMethod.SOTP: "SotpEngine",
}


class RoutedValuationExecutor:
    """Deterministic model registry. No LLM and no cross-method fallback."""

    @staticmethod
    def prepare(payload: dict[str, Any] | RoutedValuationInput) -> PreparedValuationInput:
        envelope = (
            payload
            if isinstance(payload, RoutedValuationInput)
            else RoutedValuationInput.model_validate(payload)
        )
        model = _ASSUMPTION_MODELS.get(envelope.method)
        if model is None:
            raise ValueError(f"no executable engine registered for {envelope.method.value}")
        assumptions = model.model_validate(envelope.assumptions)
        if isinstance(assumptions, EconomicFcffScenarioSet):
            if assumptions.method != envelope.method:
                raise ValueError(
                    "FCFF assumptions method must match routed input method: "
                    f"{assumptions.method.value} != {envelope.method.value}"
                )
        return PreparedValuationInput(
            envelope=envelope,
            assumptions=assumptions,  # type: ignore[arg-type]
            actual_engine=_ENGINE_NAMES[envelope.method],
        )

    def execute(
        self,
        route: ValuationRoute,
        prepared: PreparedValuationInput | None,
    ) -> RoutedValuationExecution:
        if route.applicability.status != ApplicabilityStatus.ELIGIBLE:
            return RoutedValuationExecution(
                route=route,
                status=ExecutionStatus.ROUTE_NOT_APPLICABLE,
                reason_codes=route.applicability.reason_codes,
            )
        if prepared is None:
            return RoutedValuationExecution(
                route=route,
                status=ExecutionStatus.MISSING_METHOD_INPUT,
                reason_codes=[f"MISSING_INPUT_FOR_{route.primary_method.value}"],
            )
        envelope = prepared.envelope
        mismatch: list[str] = []
        if envelope.issuer_id != route.issuer_id:
            mismatch.append("ISSUER_ID_MISMATCH")
        if envelope.as_of.isoformat() != route.as_of:
            mismatch.append("AS_OF_MISMATCH")
        if envelope.method != route.primary_method:
            mismatch.append("METHOD_MISMATCH")
        if mismatch:
            return RoutedValuationExecution(
                route=route,
                status=ExecutionStatus.INPUT_ROUTE_MISMATCH,
                input_source_refs=envelope.source_refs,
                reason_codes=mismatch,
            )
        try:
            valuation = self._value(envelope.method, prepared.assumptions)
        except Exception as exc:
            return RoutedValuationExecution(
                route=route,
                status=ExecutionStatus.VALUATION_ERROR,
                actual_engine=prepared.actual_engine,
                input_source_refs=envelope.source_refs,
                reason_codes=[f"{type(exc).__name__}:{exc}"],
            )
        return RoutedValuationExecution(
            route=route,
            status=ExecutionStatus.VALUED,
            actual_engine=prepared.actual_engine,
            valuation=valuation,
            input_source_refs=envelope.source_refs,
        )

    @staticmethod
    def invalid_input(
        route: ValuationRoute,
        exc: Exception,
    ) -> RoutedValuationExecution:
        return RoutedValuationExecution(
            route=route,
            status=ExecutionStatus.INVALID_METHOD_INPUT,
            reason_codes=[f"{type(exc).__name__}:{exc}"],
        )

    @staticmethod
    def _value(method: ValuationMethod, assumptions: AssumptionSet) -> ValuationResult:
        if method in {ValuationMethod.ECONOMIC_FCFF, ValuationMethod.NORMALIZED_FCFF}:
            if not isinstance(assumptions, EconomicFcffScenarioSet):
                raise TypeError("FCFF route requires EconomicFcffScenarioSet")
            return CommonEconomicFcffEngine().value(assumptions)
        if method == ValuationMethod.RIM:
            if not isinstance(assumptions, RimScenarioSet):
                raise TypeError("RIM route requires RimScenarioSet")
            return CommonRimEngine().value(assumptions)
        if method == ValuationMethod.RNPV:
            if not isinstance(assumptions, RnpvScenarioSet):
                raise TypeError("rNPV route requires RnpvScenarioSet")
            return CommonRnpvEngine().value(assumptions)
        if method == ValuationMethod.SCENARIO_DCF:
            if not isinstance(assumptions, ScenarioDcfAssumptions):
                raise TypeError("scenario DCF route requires ScenarioDcfAssumptions")
            return ScenarioDcfEngine().value(assumptions)
        if method == ValuationMethod.APV:
            if not isinstance(assumptions, ApvAssumptions):
                raise TypeError("APV route requires ApvAssumptions")
            return ApvEngine().value(assumptions)
        if method == ValuationMethod.NAV:
            if not isinstance(assumptions, NavAssumptions):
                raise TypeError("NAV route requires NavAssumptions")
            return NavEngine().value(assumptions)
        if method == ValuationMethod.SOTP:
            if not isinstance(assumptions, SotpAssumptions):
                raise TypeError("SOTP route requires SotpAssumptions")
            return SotpEngine().value(assumptions)
        raise ValueError(f"no executable engine registered for {method.value}")
