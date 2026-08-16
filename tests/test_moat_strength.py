from __future__ import annotations

from datetime import date

from moatrader.canonical.models import SectionRole, SourceRef, SourceType, StatementType
from moatrader.context import ContextEvidenceReference, MoatStrengthContextBuilder
from moatrader.evidence.models import (
    CandidateAtomicAuditDecision,
    CandidateAtomicAuditResult,
    CandidateAuditReason,
    CandidateSupportStatus,
    ContextualMechanismAssessment,
    ContextualMoatAssessment,
    CoverageMetrics,
    EconomicScope,
    EvidenceCard,
    EvidenceDirection,
    EvidenceType,
    MoatAuditStatus,
)
from moatrader.evidence.atomic import select_context_cited_atomic_units
from moatrader.evidence.validation import (
    build_candidate_manifest,
    derive_audited_moat_score,
    normalize_contextual_moat_assessment,
    reconcile_context_and_claims,
    validate_contextual_moat_assessment,
)
from moatrader.retrieval import ChunkMoatStrengthRetriever
from moatrader.semantic.chunker import HeuristicTokenCounter, SemanticChunk


def _chunk(
    chunk_id: str,
    text: str,
    *,
    role: SectionRole = SectionRole.BUSINESS,
    generated: bool = False,
) -> SemanticChunk:
    return SemanticChunk(
        chunk_id=chunk_id,
        document_id="D1",
        section_path=[role.value],
        section_role=role,
        node_ids=[f"N-{chunk_id}"],
        chunk_type="prose",
        markdown=text,
        token_count=HeuristicTokenCounter().count(text),
        source_refs=[
            SourceRef(
                source_type=(SourceType.GENERATED_SUMMARY if generated else SourceType.DART),
                document_id="D1",
                uri="https://example.test/source",
                source_hash="0" * 64,
            )
        ],
        metadata={"generated_summary": generated},
    )


def _atomic_unit(chunk: SemanticChunk) -> SemanticChunk:
    return SemanticChunk(
        chunk_id="AU1",
        document_id=chunk.document_id,
        section_path=chunk.section_path,
        section_role=chunk.section_role,
        node_ids=chunk.node_ids,
        chunk_type="atomic_evidence",
        markdown="Customers sign five-year contracts.",
        token_count=6,
        source_refs=chunk.source_refs,
        metadata={
            "atomic_evidence_key": "AEK1",
            "origin_chunk_ids": [chunk.chunk_id],
        },
    )


def _positive_card(*, reliability: float = 0.9) -> EvidenceCard:
    return EvidenceCard(
        evidence_id="E1",
        source_chunk_id="AU1",
        node_ids=["N-C1"],
        evidence_type=EvidenceType.SWITCHING_COST,
        statement_type=StatementType.DISCLOSED_FACT,
        fact="Customers sign five-year contracts.",
        mechanism=["contractual switching friction"],
        direction=EvidenceDirection.MOAT_POSITIVE,
        source_type=SourceType.DART,
        economic_scope=EconomicScope.COMPANY,
        raw_quote="Customers sign five-year contracts.",
        reliability=reliability,
    )


def _reference(chunk: SemanticChunk) -> ContextEvidenceReference:
    return ContextEvidenceReference(
        ref_id="R1",
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        node_ids=chunk.node_ids,
        raw_quote=chunk.markdown,
        source_types=[SourceType.DART],
    )


def _assessment(_chunk: SemanticChunk) -> ContextualMoatAssessment:
    return ContextualMoatAssessment(
        evidence_sufficiency=4,
        mechanisms=[
            ContextualMechanismAssessment(
                evidence_type=EvidenceType.SWITCHING_COST,
                strength_bucket=4,
                scope_materiality_bucket=4,
                durability_bucket=4,
                economic_scope=EconomicScope.COMPANY,
                reference_ids=["R1"],
                rationale="Long contracts create company-wide switching friction.",
            )
        ],
    )


def _candidate_audit(
    assessment: ContextualMoatAssessment,
    *,
    supported: bool,
) -> CandidateAtomicAuditResult:
    candidate = build_candidate_manifest(assessment)[0]
    return CandidateAtomicAuditResult(
        decisions=[
            CandidateAtomicAuditDecision(
                candidate_id=candidate.candidate_id,
                support=(
                    CandidateSupportStatus.SUPPORTED
                    if supported
                    else CandidateSupportStatus.INSUFFICIENT
                ),
                reason=(
                    CandidateAuditReason.EXPLICIT_CAUSAL_BARRIER
                    if supported
                    else CandidateAuditReason.INSUFFICIENT_ATOMIC_EVIDENCE
                ),
                supporting_atomic_evidence_ids=["E1"] if supported else [],
            )
        ]
    )


def test_strength_context_is_broad_balanced_and_excludes_generated_summaries() -> None:
    chunks = [
        _chunk("C1", "Five-year customer contracts and renewal retention.", role=SectionRole.CUSTOMERS),
        _chunk("C2", "Competitors face patent and qualification entry barriers.", role=SectionRole.COMPETITION),
        _chunk("C3", "Operating margin and free cash flow persisted for five years.", role=SectionRole.FINANCIALS),
        _chunk("C4", "Substitution and technology risk could weaken market share.", role=SectionRole.RISK),
        _chunk("C5", "Management says everything is wonderful.", generated=True),
    ]

    context = MoatStrengthContextBuilder().build(chunks)

    assert set(context.selected_chunk_ids) == {"C1", "C2", "C3", "C4"}
    assert "C5" not in context.markdown
    assert context.token_count <= context.token_budget
    assert {"MECHANISM", "CUSTOMER", "OUTCOME", "PERSISTENCE", "COMPETITION", "COUNTER"} <= set(
        context.question_coverage
    )
    reordered = MoatStrengthContextBuilder().build(list(reversed(chunks)))
    assert reordered.selected_chunk_ids == context.selected_chunk_ids
    assert reordered.markdown == context.markdown
    assert len(context.references) == 4
    assert "[R_" in context.markdown
    assert "## CHUNK " not in context.markdown
    assert "Node IDs:" not in context.markdown


def test_strength_retrieval_matches_korean_disclosure_terms() -> None:
    chunks = [
        _chunk("C1", "고객은 장기 계약과 인증 절차 때문에 공급자 전환이 어렵습니다."),
        _chunk("C2", "신규 경쟁사와 대체재 출현으로 시장점유율 하락 위험이 있습니다."),
    ]

    result = ChunkMoatStrengthRetriever(top_k_per_question=1).retrieve(chunks)
    by_lane = {hit.lane: hit.chunk_id for hit in result.hits}

    assert by_lane["CUSTOMER"] == "C1"
    assert by_lane["COMPETITION"] == "C2"


def test_contextual_reference_contract_repairs_unknown_ref_number_and_duplicate() -> None:
    chunk = _chunk("C1", "Customers sign five-year contracts.")
    invalid = ContextualMoatAssessment(
        evidence_sufficiency=3,
        mechanisms=[
            ContextualMechanismAssessment(
                evidence_type=EvidenceType.SWITCHING_COST,
                strength_bucket=3,
                scope_materiality_bucket=3,
                durability_bucket=3,
                economic_scope=EconomicScope.COMPANY,
                reference_ids=["R1", "R-invented"],
                rationale="The relationship persists for 99 years.",
            ),
            ContextualMechanismAssessment(
                evidence_type=EvidenceType.SWITCHING_COST,
                strength_bucket=2,
                scope_materiality_bucket=2,
                durability_bucket=2,
                economic_scope=EconomicScope.COMPANY,
                reference_ids=["R1"],
                rationale="Duplicate candidate.",
            ),
        ],
    )

    errors = validate_contextual_moat_assessment(invalid, [_reference(chunk)])
    assert any("unknown references" in error for error in errors)
    assert any("contains digits" in error for error in errors)
    assert any("duplicate" in error for error in errors)

    repaired, report = normalize_contextual_moat_assessment(invalid, [_reference(chunk)])
    assert validate_contextual_moat_assessment(repaired, [_reference(chunk)]) == []
    assert len(repaired.mechanisms) == 1
    assert repaired.mechanisms[0].strength_bucket == 2
    assert repaired.mechanisms[0].reference_ids == ["R1"]
    assert not any(character.isdigit() for character in repaired.mechanisms[0].rationale)
    assert report["action_count"] >= 3


def test_contextual_mechanism_requires_matching_atomic_claim() -> None:
    chunk = _chunk("C1", "Customers sign five-year contracts.")

    assessment = _assessment(chunk)
    candidates = build_candidate_manifest(assessment)
    reconciled = reconcile_context_and_claims(
        assessment,
        [],
        contextual_chunks=[chunk],
        atomic_units=[_atomic_unit(chunk)],
        references=[_reference(chunk)],
        candidate_manifest=candidates,
        candidate_audit=_candidate_audit(assessment, supported=False),
    )
    score = derive_audited_moat_score(
        reconciled,
        [],
        issuer_id="ISSUER",
        as_of=date(2026, 5, 31),
        document_coverage=CoverageMetrics(moat_evidence_coverage=1.0),
    )

    assert reconciled.mechanisms == []
    assert reconciled.audit_status == MoatAuditStatus.PARTIAL
    assert score.economic_moat_score == 0
    assert score.score_eligible is False
    assert score.eligibility_status.value == "BRIDGE_FAIL"


def test_context_citation_expands_atomic_audit_beyond_baseline_sample() -> None:
    chunk = _chunk(
        "C1",
        "Customers sign five-year contracts. Qualification takes two years.",
    )
    first = _atomic_unit(chunk)
    second = first.model_copy(
        update={
            "chunk_id": "AU2",
            "markdown": "Qualification takes two years.",
            "metadata": {
                "atomic_evidence_key": "AEK2",
                "origin_chunk_ids": ["C1"],
            },
        }
    )
    assessment = _assessment(chunk)

    selected = select_context_cited_atomic_units(
        [first, second],
        assessment,
        chunk_id_by_ref={"R1": "C1"},
        raw_quote_by_ref={"R1": "Customers sign five-year contracts."},
    )

    assert [unit.chunk_id for unit in selected] == ["AU1"]


def test_economic_strength_is_independent_from_evidence_reliability() -> None:
    chunk = _chunk("C1", "Customers sign five-year contracts.")
    assessment = _assessment(chunk)
    atomic = [_atomic_unit(chunk)]
    candidates = build_candidate_manifest(assessment)
    candidate_audit = _candidate_audit(assessment, supported=True)

    high_card = _positive_card(reliability=0.95)
    low_card = _positive_card(reliability=0.40)
    high_reconciled = reconcile_context_and_claims(
        assessment,
        [high_card],
        contextual_chunks=[chunk],
        atomic_units=atomic,
        references=[_reference(chunk)],
        candidate_manifest=candidates,
        candidate_audit=candidate_audit,
    )
    low_reconciled = reconcile_context_and_claims(
        assessment,
        [low_card],
        contextual_chunks=[chunk],
        atomic_units=atomic,
        references=[_reference(chunk)],
        candidate_manifest=candidates,
        candidate_audit=candidate_audit,
    )
    coverage = CoverageMetrics(moat_evidence_coverage=1.0)
    high = derive_audited_moat_score(
        high_reconciled,
        [high_card],
        issuer_id="ISSUER",
        as_of=date(2026, 5, 31),
        document_coverage=coverage,
    )
    low = derive_audited_moat_score(
        low_reconciled,
        [low_card],
        issuer_id="ISSUER",
        as_of=date(2026, 5, 31),
        document_coverage=coverage,
    )

    assert high.economic_moat_score == low.economic_moat_score == 7.5
    assert high.evidence_confidence > low.evidence_confidence
    assert high.scoring_method == "DUAL_LANE_CONTEXTUAL_STRENGTH_REDUCER_V1"


def test_atomic_counterevidence_is_retained_when_context_omits_it() -> None:
    chunk = _chunk("C1", "Customers sign five-year contracts.")
    positive = _positive_card()
    negative = EvidenceCard(
        evidence_id="E2",
        source_chunk_id="AU2",
        node_ids=["N-C1"],
        evidence_type=EvidenceType.COMPETITIVE_THREAT,
        statement_type=StatementType.DISCLOSED_FACT,
        fact="A competitor entered the market.",
        direction=EvidenceDirection.MOAT_NEGATIVE,
        source_type=SourceType.DART,
        economic_scope=EconomicScope.COMPANY,
        raw_quote="A competitor entered the market.",
        reliability=0.9,
    )
    counter_atomic = _atomic_unit(chunk).model_copy(
        update={
            "chunk_id": "AU2",
            "markdown": "A competitor entered the market.",
            "metadata": {
                "atomic_evidence_key": "AEK2",
                "origin_chunk_ids": ["C1"],
            },
        }
    )

    assessment = _assessment(chunk)
    reconciled = reconcile_context_and_claims(
        assessment,
        [positive, negative],
        contextual_chunks=[chunk],
        atomic_units=[_atomic_unit(chunk), counter_atomic],
        references=[_reference(chunk)],
        candidate_manifest=build_candidate_manifest(assessment),
        candidate_audit=_candidate_audit(assessment, supported=True),
    )

    assert any(item.atomic_evidence_ids == ["E2"] for item in reconciled.counterevidence)
    assert "E2" in reconciled.atomic_evidence_ids
