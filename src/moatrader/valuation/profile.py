from __future__ import annotations

import hashlib
import json
from datetime import date
from enum import StrEnum

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class EconomicArchetype(StrEnum):
    FINANCIAL_INTERMEDIARY = "FINANCIAL_INTERMEDIARY"
    PRE_REVENUE_BIOTECH = "PRE_REVENUE_BIOTECH"
    COMMERCIAL_PLUS_PIPELINE = "COMMERCIAL_PLUS_PIPELINE"
    MULTI_BUSINESS = "MULTI_BUSINESS"
    LEVERAGE_DRIVEN = "LEVERAGE_DRIVEN"
    ASSET_BACKED = "ASSET_BACKED"
    LOSS_MAKING_GROWTH = "LOSS_MAKING_GROWTH"
    CYCLICAL_OPERATING = "CYCLICAL_OPERATING"
    GENERAL_OPERATING = "GENERAL_OPERATING"


class ValuationProfile(ContractModel):
    """PIT, price-blind economic structure used before any valuation output exists."""

    issuer_id: str = Field(min_length=1)
    as_of: date
    sector: str = Field(min_length=1)
    industry: str = Field(min_length=1)
    economic_archetype: EconomicArchetype
    is_financial_intermediary: bool = False
    is_reit: bool = False
    is_resource_company: bool = False
    revenue_positive: bool | None = None
    ebit_positive: bool | None = None
    fcf_positive: bool | None = None
    pipeline_assets_material: bool = False
    multi_segment: bool = False
    segment_heterogeneity_material: bool = False
    leverage_path_material: bool = False
    asset_value_primary: bool = False
    materially_cyclical: bool = False
    available_data: list[str] = Field(default_factory=list)
    provenance: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_data_and_consistent_archetype(self) -> "ValuationProfile":
        if len(self.available_data) != len(set(self.available_data)):
            raise ValueError("available_data must not contain duplicates")
        if self.is_financial_intermediary and self.economic_archetype != EconomicArchetype.FINANCIAL_INTERMEDIARY:
            raise ValueError("financial intermediary must use FINANCIAL_INTERMEDIARY archetype")
        if (
            self.economic_archetype == EconomicArchetype.FINANCIAL_INTERMEDIARY
            and not self.is_financial_intermediary
        ):
            raise ValueError("FINANCIAL_INTERMEDIARY archetype requires financial flag")
        pipeline_archetypes = {
            EconomicArchetype.PRE_REVENUE_BIOTECH,
            EconomicArchetype.COMMERCIAL_PLUS_PIPELINE,
        }
        if self.pipeline_assets_material != (self.economic_archetype in pipeline_archetypes):
            raise ValueError("material pipeline flag and pipeline archetype must agree")
        sotp_trigger = self.multi_segment and self.segment_heterogeneity_material
        if sotp_trigger != (self.economic_archetype == EconomicArchetype.MULTI_BUSINESS):
            raise ValueError("heterogeneous multi-segment flag and MULTI_BUSINESS archetype must agree")
        if self.leverage_path_material != (
            self.economic_archetype == EconomicArchetype.LEVERAGE_DRIVEN
        ):
            raise ValueError("material leverage path and LEVERAGE_DRIVEN archetype must agree")
        if self.materially_cyclical != (
            self.economic_archetype == EconomicArchetype.CYCLICAL_OPERATING
        ):
            raise ValueError("cyclical flag and CYCLICAL_OPERATING archetype must agree")
        if (
            self.economic_archetype == EconomicArchetype.LOSS_MAKING_GROWTH
            and self.ebit_positive is not False
        ):
            raise ValueError("LOSS_MAKING_GROWTH archetype requires nonpositive EBIT evidence")
        if (
            self.ebit_positive is False
            and self.economic_archetype
            not in {
                EconomicArchetype.LOSS_MAKING_GROWTH,
                *pipeline_archetypes,
            }
        ):
            raise ValueError("nonpositive EBIT must use loss-making or pipeline archetype")
        if (self.is_reit or self.is_resource_company or self.asset_value_primary) and self.economic_archetype not in {
            EconomicArchetype.ASSET_BACKED,
            EconomicArchetype.MULTI_BUSINESS,
        }:
            raise ValueError("asset-primary companies must use ASSET_BACKED or MULTI_BUSINESS")
        if self.economic_archetype == EconomicArchetype.ASSET_BACKED and not (
            self.is_reit or self.is_resource_company or self.asset_value_primary
        ):
            raise ValueError("ASSET_BACKED archetype requires asset-primary evidence")
        return self

    def fingerprint(self) -> str:
        payload = self.model_dump(mode="json")
        payload["available_data"] = sorted(payload["available_data"])
        payload["provenance"] = sorted(payload["provenance"])
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
