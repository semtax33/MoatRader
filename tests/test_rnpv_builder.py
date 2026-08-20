from decimal import Decimal

import pytest

from moatrader.valuation import (
    ClinicalPhase,
    CommonRnpvEngine,
    PipelineAssetEvidence,
    RnpvBuildInput,
    RnpvBuilder,
    ValuationMethod,
)


D = Decimal


def source() -> RnpvBuildInput:
    return RnpvBuildInput(
        issuer_id="BIO1",
        as_of="2025-08-31",
        assets=[
            PipelineAssetEvidence(
                asset_id="A1",
                name="Asset A",
                indication="Disease A",
                clinical_phase=ClinicalPhase.PHASE_2,
                ownership_pct=D("0.80"),
                peak_sales=D("1000"),
                operating_cash_margin=D("0.30"),
                years_to_peak=4,
                commercial_years=10,
                remaining_development_costs=[D("40"), D("50"), D("60")],
                phase_evidence_refs=["CTG:NCT1:PHASE2"],
                ownership_evidence_refs=["DART:LICENSE:A1"],
                market_evidence_refs=["PEER_SALES:A1"],
                cost_evidence_refs=["DART:RND:A1"],
            )
        ],
        net_cash=D("100"),
        diluted_shares=D("10"),
        evidence_available_at={
            "CTG:NCT1:PHASE2": "2025-01-01",
            "DART:LICENSE:A1": "2025-01-01",
            "PEER_SALES:A1": "2025-01-01",
            "DART:RND:A1": "2025-01-01",
        },
        provenance=["PIT:BIO1"],
    )


def test_rnpv_builder_uses_frozen_pos_and_executes_engine() -> None:
    assumptions = RnpvBuilder().build(source())
    result = CommonRnpvEngine().value(assumptions)

    assert assumptions.base.assets[0].probability_of_approval == D("0.210")
    assert assumptions.base.assets[0].years_to_launch == 5
    assert result.method == ValuationMethod.RNPV
    assert result.downside_value_per_share < result.base_value_per_share
    assert result.base_value_per_share < result.upside_value_per_share
    assert "NO_LLM:DETERMINISTIC_BUILDER" in result.provenance


def test_rnpv_evidence_roles_cannot_reuse_same_reference() -> None:
    payload = source().assets[0].model_dump(mode="json")
    payload["cost_evidence_refs"] = ["PEER_SALES:A1"]

    with pytest.raises(ValueError, match="unique across evidence roles"):
        PipelineAssetEvidence.model_validate(payload)


def test_rnpv_build_input_rejects_future_evidence() -> None:
    payload = source().model_dump(mode="json")
    payload["evidence_available_at"]["PEER_SALES:A1"] = "2025-09-01"

    with pytest.raises(ValueError, match="future-dated"):
        RnpvBuildInput.model_validate(payload)
