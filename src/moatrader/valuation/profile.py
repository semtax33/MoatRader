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
    PIPELINE_ADJUDICATION_REQUIRED = "PIPELINE_ADJUDICATION_REQUIRED"
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
    pipeline_adjudication_required: bool = False
    multi_segment: bool = False
    segment_heterogeneity_material: bool = False
    leverage_path_material: bool = False
    asset_value_primary: bool = False
    materially_cyclical: bool = False
    persistent_loss: bool = False
    path_to_positive_unit_economics: bool = False
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
        if (
            self.economic_archetype
            == EconomicArchetype.PIPELINE_ADJUDICATION_REQUIRED
            and not self.pipeline_adjudication_required
        ):
            raise ValueError("pipeline candidate requires adjudication flag")
        if self.pipeline_adjudication_required and self.economic_archetype not in {
            EconomicArchetype.PIPELINE_ADJUDICATION_REQUIRED,
            *pipeline_archetypes,
        }:
            raise ValueError("pipeline adjudication evidence conflicts with lower-priority archetype")
        if (
            self.economic_archetype in pipeline_archetypes
            and not self.pipeline_assets_material
        ):
            raise ValueError("pipeline archetype requires material pipeline evidence")
        if self.pipeline_assets_material and self.economic_archetype not in {
            EconomicArchetype.FINANCIAL_INTERMEDIARY,
            EconomicArchetype.MULTI_BUSINESS,
            *pipeline_archetypes,
        }:
            raise ValueError("pipeline evidence conflicts with lower-priority archetype")
        sotp_trigger = self.multi_segment and self.segment_heterogeneity_material
        if self.economic_archetype == EconomicArchetype.MULTI_BUSINESS and not sotp_trigger:
            raise ValueError("MULTI_BUSINESS archetype requires heterogeneous segments")
        if sotp_trigger and self.economic_archetype not in {
            EconomicArchetype.FINANCIAL_INTERMEDIARY,
            EconomicArchetype.MULTI_BUSINESS,
        }:
            raise ValueError("heterogeneous segments conflict with lower-priority archetype")
        if (
            self.economic_archetype == EconomicArchetype.LEVERAGE_DRIVEN
            and not self.leverage_path_material
        ):
            raise ValueError("LEVERAGE_DRIVEN archetype requires material leverage path")
        if self.leverage_path_material and self.economic_archetype not in {
            EconomicArchetype.FINANCIAL_INTERMEDIARY,
            EconomicArchetype.MULTI_BUSINESS,
            *pipeline_archetypes,
            EconomicArchetype.ASSET_BACKED,
            EconomicArchetype.LEVERAGE_DRIVEN,
        }:
            raise ValueError("material leverage conflicts with lower-priority archetype")
        if (
            self.economic_archetype == EconomicArchetype.CYCLICAL_OPERATING
            and not self.materially_cyclical
        ):
            raise ValueError("CYCLICAL_OPERATING archetype requires cyclical evidence")
        if (
            self.materially_cyclical
            and self.economic_archetype == EconomicArchetype.GENERAL_OPERATING
        ):
            raise ValueError("cyclical evidence conflicts with GENERAL_OPERATING archetype")
        if (
            self.economic_archetype == EconomicArchetype.LOSS_MAKING_GROWTH
            and (self.ebit_positive is not False or not self.persistent_loss)
        ):
            raise ValueError(
                "LOSS_MAKING_GROWTH archetype requires persistent nonpositive EBIT evidence"
            )
        if (
            self.ebit_positive is False
            and self.persistent_loss
            and self.economic_archetype
            not in {
                EconomicArchetype.FINANCIAL_INTERMEDIARY,
                EconomicArchetype.MULTI_BUSINESS,
                EconomicArchetype.LOSS_MAKING_GROWTH,
                EconomicArchetype.PIPELINE_ADJUDICATION_REQUIRED,
                *pipeline_archetypes,
                EconomicArchetype.ASSET_BACKED,
                EconomicArchetype.LEVERAGE_DRIVEN,
                EconomicArchetype.CYCLICAL_OPERATING,
            }
        ):
            raise ValueError(
                "nonpositive EBIT must use loss-making, pipeline, or cyclical archetype"
            )
        if self.path_to_positive_unit_economics and not self.persistent_loss:
            raise ValueError("unit-economics recovery path requires persistent-loss evidence")
        if (self.is_reit or self.is_resource_company or self.asset_value_primary) and self.economic_archetype not in {
            EconomicArchetype.FINANCIAL_INTERMEDIARY,
            EconomicArchetype.MULTI_BUSINESS,
            *pipeline_archetypes,
            EconomicArchetype.ASSET_BACKED,
        }:
            raise ValueError("asset-primary evidence conflicts with lower-priority archetype")
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
