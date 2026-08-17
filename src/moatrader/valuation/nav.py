from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.base import ValuationMethod, ValuationResult, eligible


class NavAsset(ContractModel):
    name: str = Field(min_length=1)
    base_value: Decimal = Field(ge=0)
    downside_haircut: Decimal = Field(default=Decimal("0.20"), ge=0, le=1)
    upside_premium: Decimal = Field(default=Decimal(0), ge=0, le=3)
    ownership_pct: Decimal = Field(default=Decimal(1), gt=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)


class NavAssumptions(ContractModel):
    assets: list[NavAsset] = Field(min_length=1)
    cash: Decimal = Field(default=Decimal(0), ge=0)
    debt: Decimal = Field(default=Decimal(0), ge=0)
    other_liabilities: Decimal = Field(default=Decimal(0), ge=0)
    diluted_shares: Decimal = Field(gt=0)
    assumption_confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    provenance: list[str] = Field(default_factory=list)


class NavEngine:
    def value(self, assumptions: NavAssumptions) -> ValuationResult:
        base_assets = sum(
            (item.base_value * item.ownership_pct for item in assumptions.assets), Decimal(0)
        )
        downside_assets = sum(
            (
                item.base_value
                * (Decimal(1) - item.downside_haircut)
                * item.ownership_pct
                for item in assumptions.assets
            ),
            Decimal(0),
        )
        upside_assets = sum(
            (
                item.base_value
                * (Decimal(1) + item.upside_premium)
                * item.ownership_pct
                for item in assumptions.assets
            ),
            Decimal(0),
        )
        liabilities = assumptions.debt + assumptions.other_liabilities
        base_equity = base_assets + assumptions.cash - liabilities
        downside_equity = downside_assets + assumptions.cash - liabilities
        upside_equity = upside_assets + assumptions.cash - liabilities
        shares = assumptions.diluted_shares
        return ValuationResult(
            method=ValuationMethod.NAV,
            applicability=eligible(
                ValuationMethod.NAV,
                ["assets", "debt", "other_liabilities", "diluted_shares"],
            ),
            equity_value=base_equity,
            fair_value_per_share=base_equity / shares,
            downside_value_per_share=downside_equity / shares,
            base_value_per_share=base_equity / shares,
            upside_value_per_share=upside_equity / shares,
            assumption_confidence=assumptions.assumption_confidence,
            provenance=assumptions.provenance,
            metadata={"asset_count": len(assumptions.assets)},
        )
