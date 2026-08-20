from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.base import ValuationMethod, ValuationResult, eligible


SOTP_POLICY_VERSION = "sotp-policy/2"


class SotpValueBasis(StrEnum):
    ENTERPRISE = "ENTERPRISE_VALUE"
    EQUITY = "EQUITY_VALUE"


class SotpPartBuildInput(ContractModel):
    name: str = Field(min_length=1)
    method: ValuationMethod
    assumptions: dict[str, Any]
    value_basis: SotpValueBasis = SotpValueBasis.EQUITY
    ownership_pct: Decimal = Field(default=Decimal(1), gt=0, le=1)
    ownership_applied: bool = False
    net_debt_adjustment: Decimal = Field(default=Decimal(0), ge=0)
    net_debt_scope_id: str = Field(min_length=1)
    nci_adjustment: Decimal = Field(default=Decimal(0), ge=0)
    nci_scope_id: str = Field(min_length=1)
    cashflow_scope_id: str = Field(min_length=1)
    included_cashflows: list[str] = Field(min_length=1)
    excluded_cashflows: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def supported_submodel(self) -> "SotpPartBuildInput":
        if self.method == ValuationMethod.SOTP:
            raise ValueError("SOTP cannot recursively value an SOTP part")
        if self.value_basis == SotpValueBasis.EQUITY and (
            self.net_debt_adjustment or self.nci_adjustment
        ):
            raise ValueError("equity-value part must not deduct net debt or NCI again")
        return self


class SotpPart(ContractModel):
    name: str = Field(min_length=1)
    method: ValuationMethod
    value_basis: SotpValueBasis
    downside_value: Decimal
    base_value: Decimal
    upside_value: Decimal
    ownership_pct: Decimal = Field(default=Decimal(1), gt=0, le=1)
    ownership_applied: bool
    net_debt_adjustment: Decimal = Field(ge=0)
    net_debt_scope_id: str = Field(min_length=1)
    nci_adjustment: Decimal = Field(ge=0)
    nci_scope_id: str = Field(min_length=1)
    cashflow_scope_id: str = Field(min_length=1)
    included_cashflows: list[str] = Field(min_length=1)
    excluded_cashflows: list[str] = Field(default_factory=list)
    actual_engine: str = Field(min_length=1)
    submodel_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_status: Literal["VALUED"] = "VALUED"
    provenance: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_and_disjoint(self) -> "SotpPart":
        if not self.downside_value <= self.base_value <= self.upside_value:
            raise ValueError("SOTP part values must be ordered downside <= base <= upside")
        overlap = set(self.included_cashflows) & set(self.excluded_cashflows)
        if overlap:
            raise ValueError(
                f"cash flows cannot be both included and excluded: {sorted(overlap)}"
            )
        if self.value_basis == SotpValueBasis.EQUITY and (
            self.net_debt_adjustment or self.nci_adjustment
        ):
            raise ValueError("equity-value part must not deduct net debt or NCI again")
        return self


class SotpAssumptions(ContractModel):
    parts: list[SotpPart] = Field(min_length=2)
    parent_cash: Decimal = Field(default=Decimal(0), ge=0)
    parent_debt: Decimal = Field(default=Decimal(0), ge=0)
    parent_net_debt_scope_id: str = Field(default="PARENT_NET_DEBT", min_length=1)
    parent_nci: Decimal = Field(default=Decimal(0), ge=0)
    parent_nci_scope_id: str = Field(default="PARENT_NCI", min_length=1)
    other_adjustments: Decimal = Decimal(0)
    intersegment_adjustments: Decimal = Decimal(0)
    diluted_shares: Decimal = Field(gt=0)
    assumption_confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def prevent_double_counting(self) -> "SotpAssumptions":
        scope_groups = {
            "cash-flow": [item.cashflow_scope_id for item in self.parts],
            "net-debt": [item.net_debt_scope_id for item in self.parts]
            + [self.parent_net_debt_scope_id],
            "NCI": [item.nci_scope_id for item in self.parts]
            + [self.parent_nci_scope_id],
        }
        for scope_type, scopes in scope_groups.items():
            duplicates = sorted({scope for scope in scopes if scopes.count(scope) > 1})
            if duplicates:
                raise ValueError(f"duplicate {scope_type} scope: {duplicates}")
        owners: dict[str, str] = {}
        for part in self.parts:
            for cashflow in part.included_cashflows:
                previous = owners.setdefault(cashflow, part.name)
                if previous != part.name:
                    raise ValueError(
                        f"cash-flow scope {cashflow!r} is included by both "
                        f"{previous!r} and {part.name!r}"
                    )
        return self


class SotpBuildInput(ContractModel):
    policy_version: Literal["sotp-policy/2"] = SOTP_POLICY_VERSION
    parts: list[SotpPartBuildInput] = Field(min_length=2)
    parent_cash: Decimal = Field(default=Decimal(0), ge=0)
    parent_debt: Decimal = Field(default=Decimal(0), ge=0)
    parent_net_debt_scope_id: str = Field(default="PARENT_NET_DEBT", min_length=1)
    parent_nci: Decimal = Field(default=Decimal(0), ge=0)
    parent_nci_scope_id: str = Field(default="PARENT_NCI", min_length=1)
    other_adjustments: Decimal = Decimal(0)
    intersegment_adjustments: Decimal = Decimal(0)
    diluted_shares: Decimal = Field(gt=0)
    assumption_confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    provenance: list[str] = Field(min_length=1)


def _fingerprint(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _execute_submodel(
    method: ValuationMethod, payload: dict[str, Any]
) -> tuple[ValuationResult, Decimal, str]:
    if method == ValuationMethod.ECONOMIC_FCFF:
        from moatrader.valuation.common_engines import CommonEconomicFcffEngine, EconomicFcffScenarioSet

        assumptions = EconomicFcffScenarioSet.model_validate(payload)
        return CommonEconomicFcffEngine().value(assumptions), assumptions.base.diluted_shares, "CommonEconomicFcffEngine"
    if method == ValuationMethod.NORMALIZED_FCFF:
        from moatrader.valuation.normalized_fcff import NormalizedFcffAssumptions, NormalizedFcffEngine

        assumptions = NormalizedFcffAssumptions.model_validate(payload)
        return NormalizedFcffEngine().value(assumptions), assumptions.base.diluted_shares, "NormalizedFcffEngine"
    if method == ValuationMethod.SCENARIO_DCF:
        from moatrader.valuation.scenario_dcf import ScenarioDcfAssumptions, ScenarioDcfEngine

        assumptions = ScenarioDcfAssumptions.model_validate(payload)
        return ScenarioDcfEngine().value(assumptions), assumptions.central.diluted_shares, "ScenarioDcfEngine"
    if method == ValuationMethod.RIM:
        from moatrader.valuation.rim import CommonRimEngine, RimScenarioSet

        assumptions = RimScenarioSet.model_validate(payload)
        return CommonRimEngine().value(assumptions), assumptions.base.diluted_shares, "CommonRimEngine"
    if method == ValuationMethod.RNPV:
        from moatrader.valuation.common_engines import CommonRnpvEngine, RnpvScenarioSet

        assumptions = RnpvScenarioSet.model_validate(payload)
        return CommonRnpvEngine().value(assumptions), assumptions.base.diluted_shares, "CommonRnpvEngine"
    if method == ValuationMethod.NAV:
        from moatrader.valuation.nav import NavAssumptions, NavEngine

        assumptions = NavAssumptions.model_validate(payload)
        return NavEngine().value(assumptions), assumptions.diluted_shares, "NavEngine"
    if method == ValuationMethod.APV:
        from moatrader.valuation.apv import ApvAssumptions, ApvEngine

        assumptions = ApvAssumptions.model_validate(payload)
        return ApvEngine().value(assumptions), assumptions.diluted_shares, "ApvEngine"
    raise ValueError(f"unsupported SOTP submodel: {method.value}")


class SotpBuilder:
    """Execute every part's declared submodel before SOTP aggregation."""

    def build(self, source: SotpBuildInput) -> SotpAssumptions:
        parts: list[SotpPart] = []
        for item in source.parts:
            result, shares, engine_name = _execute_submodel(item.method, item.assumptions)
            if item.value_basis != SotpValueBasis.EQUITY:
                raise ValueError(
                    "actual submodel execution requires EQUITY_VALUE because "
                    "scenario enterprise values are unavailable"
                )
            values = (
                result.downside_value_per_share,
                result.base_value_per_share,
                result.upside_value_per_share,
            )
            if any(value is None for value in values):
                raise ValueError(f"submodel {item.name} lacks scenario equity values")
            parts.append(
                SotpPart(
                    name=item.name,
                    method=item.method,
                    value_basis=item.value_basis,
                    downside_value=values[0] * shares,
                    base_value=values[1] * shares,
                    upside_value=values[2] * shares,
                    ownership_pct=item.ownership_pct,
                    ownership_applied=item.ownership_applied,
                    net_debt_adjustment=item.net_debt_adjustment,
                    net_debt_scope_id=item.net_debt_scope_id,
                    nci_adjustment=item.nci_adjustment,
                    nci_scope_id=item.nci_scope_id,
                    cashflow_scope_id=item.cashflow_scope_id,
                    included_cashflows=item.included_cashflows,
                    excluded_cashflows=item.excluded_cashflows,
                    actual_engine=engine_name,
                    submodel_input_sha256=_fingerprint(item.assumptions),
                    provenance=item.provenance
                    + [f"ACTUAL_ENGINE:{engine_name}", SOTP_POLICY_VERSION],
                )
            )
        assumptions = SotpAssumptions(
            parts=parts,
            parent_cash=source.parent_cash,
            parent_debt=source.parent_debt,
            parent_net_debt_scope_id=source.parent_net_debt_scope_id,
            parent_nci=source.parent_nci,
            parent_nci_scope_id=source.parent_nci_scope_id,
            other_adjustments=source.other_adjustments,
            intersegment_adjustments=source.intersegment_adjustments,
            diluted_shares=source.diluted_shares,
            assumption_confidence=source.assumption_confidence,
            provenance=source.provenance
            + [SOTP_POLICY_VERSION, "NO_LLM:DETERMINISTIC_SUBMODEL_EXECUTION"],
        )
        SotpEngine().value(assumptions)
        return assumptions


class SotpEngine:
    def value(self, assumptions: SotpAssumptions) -> ValuationResult:
        def part_equity(item: SotpPart, field: str) -> Decimal:
            value = getattr(item, field)
            if item.value_basis == SotpValueBasis.ENTERPRISE:
                value -= item.net_debt_adjustment + item.nci_adjustment
            return value if item.ownership_applied else value * item.ownership_pct

        def total(field: str) -> Decimal:
            return (
                sum((part_equity(item, field) for item in assumptions.parts), Decimal(0))
                + assumptions.parent_cash
                - assumptions.parent_debt
                - assumptions.parent_nci
                + assumptions.other_adjustments
                + assumptions.intersegment_adjustments
            )

        downside, base, upside = (
            total(field)
            for field in ("downside_value", "base_value", "upside_value")
        )
        shares = assumptions.diluted_shares
        return ValuationResult(
            method=ValuationMethod.SOTP,
            applicability=eligible(
                ValuationMethod.SOTP,
                [
                    "parts",
                    "actual_engine",
                    "cashflow_scope_id",
                    "ownership_pct",
                    "net_debt_scope_id",
                    "nci_scope_id",
                    "diluted_shares",
                ],
            ),
            equity_value=base,
            fair_value_per_share=base / shares,
            downside_value_per_share=downside / shares,
            base_value_per_share=base / shares,
            upside_value_per_share=upside / shares,
            assumption_confidence=assumptions.assumption_confidence,
            provenance=assumptions.provenance,
            metadata={
                "part_count": len(assumptions.parts),
                "part_methods": [item.method.value for item in assumptions.parts],
                "part_actual_engines": [item.actual_engine for item in assumptions.parts],
                "part_execution_statuses": [item.execution_status for item in assumptions.parts],
            },
        )
