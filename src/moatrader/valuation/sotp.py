from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.base import ValuationMethod, ValuationResult, eligible


class SotpValueBasis(StrEnum):
    ENTERPRISE = "ENTERPRISE"
    EQUITY = "EQUITY"


class SotpPart(ContractModel):
    name: str = Field(min_length=1)
    method: ValuationMethod
    value_basis: SotpValueBasis
    downside_value: Decimal
    base_value: Decimal
    upside_value: Decimal
    ownership_pct: Decimal = Field(default=Decimal(1), gt=0, le=1)
    included_cashflows: list[str] = Field(min_length=1)
    excluded_cashflows: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_and_disjoint(self) -> "SotpPart":
        if not self.downside_value <= self.base_value <= self.upside_value:
            raise ValueError("SOTP part values must be ordered downside <= base <= upside")
        overlap = set(self.included_cashflows) & set(self.excluded_cashflows)
        if overlap:
            raise ValueError(f"cash flows cannot be both included and excluded: {sorted(overlap)}")
        return self


class SotpAssumptions(ContractModel):
    parts: list[SotpPart] = Field(min_length=2)
    parent_cash: Decimal = Field(default=Decimal(0), ge=0)
    parent_debt: Decimal = Field(default=Decimal(0), ge=0)
    other_adjustments: Decimal = Decimal(0)
    intersegment_adjustments: Decimal = Decimal(0)
    diluted_shares: Decimal = Field(gt=0)
    assumption_confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def prevent_double_counting(self) -> "SotpAssumptions":
        owners: dict[str, str] = {}
        for part in self.parts:
            for cashflow in part.included_cashflows:
                previous = owners.setdefault(cashflow, part.name)
                if previous != part.name:
                    raise ValueError(
                        f"cash-flow scope {cashflow!r} is included by both {previous!r} and {part.name!r}"
                    )
        return self


class SotpEngine:
    def value(self, assumptions: SotpAssumptions) -> ValuationResult:
        enterprise_parts = [item for item in assumptions.parts if item.value_basis == SotpValueBasis.ENTERPRISE]
        equity_parts = [item for item in assumptions.parts if item.value_basis == SotpValueBasis.EQUITY]

        def total(field: str) -> Decimal:
            enterprise = sum(
                (getattr(item, field) * item.ownership_pct for item in enterprise_parts), Decimal(0)
            )
            equity = sum(
                (getattr(item, field) * item.ownership_pct for item in equity_parts), Decimal(0)
            )
            return (
                enterprise
                + equity
                + assumptions.parent_cash
                - assumptions.parent_debt
                + assumptions.other_adjustments
                + assumptions.intersegment_adjustments
            )

        downside, base, upside = (total(field) for field in ("downside_value", "base_value", "upside_value"))
        shares = assumptions.diluted_shares
        return ValuationResult(
            method=ValuationMethod.SOTP,
            applicability=eligible(
                ValuationMethod.SOTP,
                ["parts", "included_cashflows", "ownership_pct", "diluted_shares"],
            ),
            enterprise_value=sum(
                (item.base_value * item.ownership_pct for item in enterprise_parts), Decimal(0)
            ) or None,
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
            },
        )
