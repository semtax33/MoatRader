from __future__ import annotations

from decimal import Decimal
from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.biotech_rnpv import BiotechRnpvAssumptions, PipelineAsset
from moatrader.valuation.common_engines import RnpvScenarioSet


RNPV_POLICY_VERSION = "rnpv-policy/1"
RNPV_POS_REFERENCE = "doi:10.1093/biostatistics/kxx069:table-1:path-by-path"


class ClinicalPhase(StrEnum):
    PHASE_1 = "PHASE_1"
    PHASE_2 = "PHASE_2"
    PHASE_3 = "PHASE_3"
    APPROVED = "APPROVED"


PHASE_PROBABILITY_OF_APPROVAL: dict[ClinicalPhase, Decimal] = {
    ClinicalPhase.PHASE_1: Decimal("0.138"),
    ClinicalPhase.PHASE_2: Decimal("0.210"),
    ClinicalPhase.PHASE_3: Decimal("0.590"),
    ClinicalPhase.APPROVED: Decimal("1.000"),
}
PHASE_YEARS_TO_LAUNCH: dict[ClinicalPhase, int] = {
    ClinicalPhase.PHASE_1: 7,
    ClinicalPhase.PHASE_2: 5,
    ClinicalPhase.PHASE_3: 2,
    ClinicalPhase.APPROVED: 0,
}


class PipelineAssetEvidence(ContractModel):
    asset_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    indication: str = Field(min_length=1)
    clinical_phase: ClinicalPhase
    ownership_pct: Decimal = Field(gt=0, le=1)
    peak_sales: Decimal = Field(gt=0)
    operating_cash_margin: Decimal = Field(gt=0, le=1)
    years_to_peak: int = Field(ge=1, le=10)
    commercial_years: int = Field(ge=3, le=20)
    remaining_development_costs: list[Decimal] = Field(min_length=1, max_length=10)
    phase_evidence_refs: list[str] = Field(min_length=1)
    ownership_evidence_refs: list[str] = Field(min_length=1)
    market_evidence_refs: list[str] = Field(min_length=1)
    cost_evidence_refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_evidence(self) -> "PipelineAssetEvidence":
        refs = (
            self.phase_evidence_refs
            + self.ownership_evidence_refs
            + self.market_evidence_refs
            + self.cost_evidence_refs
        )
        if len(refs) != len(set(refs)):
            raise ValueError("rNPV evidence refs must be unique across evidence roles")
        return self


class RnpvBuildInput(ContractModel):
    policy_version: Literal["rnpv-policy/1"] = RNPV_POLICY_VERSION
    issuer_id: str = Field(min_length=1)
    as_of: str = Field(min_length=10)
    assets: list[PipelineAssetEvidence] = Field(min_length=1)
    net_cash: Decimal
    diluted_shares: Decimal = Field(gt=0)
    evidence_available_at: dict[str, date] = Field(min_length=1)
    provenance: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def evidence_is_complete_and_pit(self) -> "RnpvBuildInput":
        refs = {
            ref
            for asset in self.assets
            for ref in (
                asset.phase_evidence_refs
                + asset.ownership_evidence_refs
                + asset.market_evidence_refs
                + asset.cost_evidence_refs
            )
        }
        if set(self.evidence_available_at) != refs:
            raise ValueError("rNPV evidence availability must cover exactly all role refs")
        cutoff = date.fromisoformat(self.as_of[:10])
        future = sorted(
            ref
            for ref, available_at in self.evidence_available_at.items()
            if available_at > cutoff
        )
        if future:
            raise ValueError(f"rNPV evidence is future-dated: {future}")
        return self


def _launch_value_at_launch(
    asset: PipelineAssetEvidence,
    *,
    sales_multiplier: Decimal,
    margin_multiplier: Decimal,
    discount_rate: Decimal,
) -> Decimal:
    peak_sales = asset.peak_sales * sales_multiplier
    margin = min(asset.operating_cash_margin * margin_multiplier, Decimal("0.90"))
    value = Decimal(0)
    for year in range(1, asset.commercial_years + 1):
        ramp = min(Decimal(year) / Decimal(asset.years_to_peak), Decimal(1))
        cash_flow = peak_sales * ramp * margin * asset.ownership_pct
        value += cash_flow / ((Decimal(1) + discount_rate) ** year)
    return value


class RnpvBuilder:
    """Build rNPV with frozen phase POS and role-separated asset evidence."""

    def build(self, source: RnpvBuildInput) -> RnpvScenarioSet:
        scenario_parameters = (
            (Decimal("0.70"), Decimal("0.80"), Decimal("1.20"), Decimal("0.14")),
            (Decimal("1.00"), Decimal("1.00"), Decimal("1.00"), Decimal("0.12")),
            (Decimal("1.30"), Decimal("1.20"), Decimal("0.80"), Decimal("0.10")),
        )
        cases: list[BiotechRnpvAssumptions] = []
        for sales_multiple, margin_multiple, cost_multiple, discount_rate in scenario_parameters:
            assets: list[PipelineAsset] = []
            for evidence in source.assets:
                probability = PHASE_PROBABILITY_OF_APPROVAL[evidence.clinical_phase]
                launch_value = _launch_value_at_launch(
                    evidence,
                    sales_multiplier=sales_multiple,
                    margin_multiplier=margin_multiple,
                    discount_rate=discount_rate,
                )
                costs = [
                    cost * evidence.ownership_pct * cost_multiple
                    for cost in evidence.remaining_development_costs
                ]
                refs = list(
                    dict.fromkeys(
                        evidence.phase_evidence_refs
                        + evidence.ownership_evidence_refs
                        + evidence.market_evidence_refs
                        + evidence.cost_evidence_refs
                        + [RNPV_POS_REFERENCE, RNPV_POLICY_VERSION]
                    )
                )
                assets.append(
                    PipelineAsset(
                        name=f"{evidence.name} — {evidence.indication}",
                        years_to_launch=PHASE_YEARS_TO_LAUNCH[evidence.clinical_phase],
                        probability_of_approval=probability,
                        launch_value=launch_value,
                        remaining_development_costs=costs,
                        evidence_ids=refs,
                    )
                )
            cases.append(
                BiotechRnpvAssumptions(
                    assets=assets,
                    discount_rate=discount_rate,
                    net_cash=source.net_cash,
                    diluted_shares=source.diluted_shares,
                )
            )
        return RnpvScenarioSet(
            downside=cases[0],
            base=cases[1],
            upside=cases[2],
            assumption_confidence=Decimal("0.50"),
            provenance=source.provenance
            + [
                RNPV_POLICY_VERSION,
                RNPV_POS_REFERENCE,
                "NO_LLM:DETERMINISTIC_BUILDER",
            ],
        )
