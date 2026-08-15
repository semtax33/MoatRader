from __future__ import annotations

from datetime import date

from moatrader.canonical.models import SourceType, StatementType
from moatrader.evidence.models import (
    CoverageMetrics,
    Durability,
    EconomicScope,
    EvidenceCard,
    EvidenceDirection,
    EvidenceType,
    MoatMechanismScore,
    MoatScore,
)
from moatrader.evidence.validation import derive_moat_score, validate_moat_score


def _card(
    evidence_id: str,
    evidence_type: EvidenceType,
    direction: EvidenceDirection,
    *,
    strength: float = 0.8,
    reliability: float = 0.8,
) -> EvidenceCard:
    return EvidenceCard(
        evidence_id=evidence_id,
        source_chunk_id=f"C-{evidence_id}",
        node_ids=[f"N-{evidence_id}"],
        evidence_type=evidence_type,
        statement_type=StatementType.DISCLOSED_FACT,
        fact="Grounded company-specific evidence.",
        mechanism=["barrier", "customer friction"],
        direction=direction,
        strength=strength,
        reliability=reliability,
        source_type=SourceType.DART,
        economic_scope=EconomicScope.COMPANY,
        raw_quote="Grounded company-specific evidence.",
    )


def _score(*, evidence_type: EvidenceType, evidence_ids: list[str], durability: Durability = Durability.HIGH) -> MoatScore:
    return MoatScore(
        as_of=date(2026, 5, 31),
        economic_moat_score=10,
        mechanisms=[
            MoatMechanismScore(
                evidence_type=evidence_type,
                score=10,
                evidence_ids=evidence_ids,
                rationale="Proposed mechanism.",
            )
        ],
        durability=durability,
        model_confidence=0.95,
        document_coverage=CoverageMetrics(),
    )


def test_score_rejects_non_structural_and_type_mismatched_mechanisms() -> None:
    cards = [_card("E1", EvidenceType.MARGIN_STABILITY, EvidenceDirection.MOAT_POSITIVE)]
    score = _score(evidence_type=EvidenceType.MARGIN_STABILITY, evidence_ids=["E1"])
    assert any("not a structural moat mechanism" in error for error in validate_moat_score(score, cards))

    score.mechanisms[0].evidence_type = EvidenceType.SWITCHING_COST
    assert any("type is MARGIN_STABILITY" in error for error in validate_moat_score(score, cards))


def test_score_requires_positive_direction_and_negative_counterevidence() -> None:
    cards = [
        _card("E1", EvidenceType.SWITCHING_COST, EvidenceDirection.NEUTRAL),
        _card("E2", EvidenceType.COMPETITIVE_THREAT, EvidenceDirection.NEUTRAL),
    ]
    score = _score(evidence_type=EvidenceType.SWITCHING_COST, evidence_ids=["E1"])
    score.counterevidence_ids = ["E2"]
    errors = validate_moat_score(score, cards)
    assert any("expected MOAT_POSITIVE" in error for error in errors)
    assert any("expected MOAT_NEGATIVE" in error for error in errors)


def test_published_score_ignores_llm_numeric_score_and_durability() -> None:
    cards = [_card("E1", EvidenceType.SWITCHING_COST, EvidenceDirection.MOAT_POSITIVE)]
    proposed = _score(
        evidence_type=EvidenceType.SWITCHING_COST,
        evidence_ids=["E1"],
        durability=Durability.LOW,
    )
    assert validate_moat_score(proposed, cards) == []

    derived = derive_moat_score(proposed, cards)

    assert derived.llm_proposed_score == 10
    assert derived.mechanisms[0].score == 8.0
    assert derived.economic_moat_score == 5.0
    assert derived.durability == Durability.MEDIUM
    assert derived.model_confidence == 0.8

    different_proposal = proposed.model_copy(update={"economic_moat_score": 1, "durability": Durability.HIGH})
    different_proposal.mechanisms[0].score = 1
    repeated = derive_moat_score(different_proposal, cards)
    assert repeated.economic_moat_score == derived.economic_moat_score
    assert repeated.mechanisms == derived.mechanisms
    assert repeated.durability == derived.durability
