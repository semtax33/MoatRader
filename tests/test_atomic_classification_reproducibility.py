from __future__ import annotations

from moatrader.evidence.models import (
    AtomicEvidenceExtraction,
    AtomicMoatRole,
    EvidenceDirection,
    EvidenceType,
)
from scripts.audit_atomic_classification_reproducibility import evaluate_case, summarize


def _vote(role: AtomicMoatRole, evidence_type: EvidenceType, direction: EvidenceDirection):
    return AtomicEvidenceExtraction(
        is_investment_relevant=role != AtomicMoatRole.NONE,
        moat_role=role,
        evidence_type=evidence_type,
        direction=direction,
        fact="fact",
    )


def _case() -> dict[str, object]:
    return {
        "ticker": "000001",
        "atomic_evidence_key": "AEK_TEST",
        "source_type": "IR",
        "document_id": "D1",
        "baseline_signature": ["MECHANISM", True, "COST_ADVANTAGE", "MOAT_POSITIVE", "COMPANY"],
        "repeat_signature": ["NONE", False, "OTHER", "NEUTRAL", "COMPANY"],
        "role_transition": True,
        "chunk": {
            "chunk_id": "AU1",
            "document_id": "D1",
            "node_ids": ["N1"],
            "chunk_type": "atomic_evidence",
            "markdown": "수율 개선으로 낮은 제조비용을 달성했다.",
            "token_count": 10,
            "metadata": {"atomic_evidence_key": "AEK_TEST"},
        },
    }


def test_two_independent_consensus_groups_match() -> None:
    mechanism = _vote(
        AtomicMoatRole.MECHANISM,
        EvidenceType.COST_ADVANTAGE,
        EvidenceDirection.MOAT_POSITIVE,
    )
    irrelevant = AtomicEvidenceExtraction()
    result = evaluate_case(
        _case(),
        [mechanism, mechanism, irrelevant, mechanism, mechanism, mechanism],
    )

    assert result["consensus_role_match"] is True
    assert result["consensus_signature_match"] is True
    assert result["score_route_conflict"] is False


def test_summary_requires_consensus_stability_improvement() -> None:
    row = {
        "baseline_signature": ["MECHANISM"],
        "repeat_signature": ["NONE"],
        "vote_count": 10,
        "raw_modal_agreement_rate": 0.8,
        "raw_pairwise_agreement_rate": 0.7,
        "raw_route_modal_agreement_rate": 0.9,
        "raw_route_pairwise_agreement_rate": 0.8,
        "consensus_role_match": True,
        "consensus_route_signature_match": True,
        "consensus_signature_match": True,
        "consensus_scope_match": True,
        "score_route_conflict": False,
        "production_three_vote": {
            "route_match_to_full_rate": 1.0,
            "any_route_mismatch_rate": 0.0,
            "moat_route_conflict_rate": 0.0,
        },
    }
    result = summarize([row] * 10, requested_votes=10)

    assert result["baseline_exact_signature_rate"] == 0.0
    assert result["baseline_exact_route_rate"] == 0.0
    assert result["independent_consensus_route_match_rate"] == 1.0
    assert result["production_three_vote_supported"] is True
    assert result["independent_consensus_signature_match_rate"] == 1.0
    assert result["classifier_stability_supported"] is True
