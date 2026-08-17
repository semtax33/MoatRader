from __future__ import annotations

import hashlib
from datetime import date

from moatrader.canonical.models import SectionRole, SourceRef, SourceType, StatementType
from moatrader.evidence.atomic import (
    build_atomic_evidence_units,
    select_atomic_evidence_units,
    split_atomic_evidence_text,
)
from moatrader.evidence.models import (
    AtomicEvidenceExtraction,
    CanonicalClaimSignature,
    DcfLink,
    EconomicScope,
    EvidenceCard,
    EvidenceDirection,
    EvidenceType,
    ForwardDriverType,
)
from moatrader.evidence.processing import atomic_extraction_to_judgment, build_canonical_claim_set
from moatrader.evidence.metamorphic import _is_reducer_input
from moatrader.evidence.validation import derive_moat_score
from moatrader.llm.contracts import build_atomic_evidence_request
from moatrader.llm.replay import LLMReplayCache
from moatrader.semantic.chunker import SemanticChunk


def _chunk(
    chunk_id: str,
    text: str,
    *,
    role: SectionRole = SectionRole.BUSINESS,
    source_type: SourceType = SourceType.DART,
) -> SemanticChunk:
    return SemanticChunk(
        chunk_id=chunk_id,
        document_id="DOC-1",
        section_path=["Business"],
        section_role=role,
        node_ids=[f"N-{chunk_id}"],
        chunk_type="text",
        markdown=text,
        token_count=max(1, len(text.split())),
        source_refs=[SourceRef(source_type=source_type, document_id="DOC-1")],
    )


def _selected_keys(chunks: list[SemanticChunk]) -> set[str]:
    units = select_atomic_evidence_units(
        build_atomic_evidence_units(chunks, issuer_id="ISSUER-1"),
        20,
    )
    return {str(unit.metadata["atomic_evidence_key"]) for unit in units}


def test_atomic_evidence_set_ignores_order_duplicate_summary_and_formatting() -> None:
    first = "고객 시스템에 당사 솔루션이 통합되어 있다. 교체에는 상당한 비용과 시간이 필요하다."
    second = "주요 고객 계약은 장기간 유지되며 경쟁 제품으로 전환하기 어렵다."
    baseline = [_chunk("C1", first), _chunk("C2", second, role=SectionRole.CUSTOMERS)]
    expected = _selected_keys(baseline)

    shuffled = [
        _chunk("C2", second, role=SectionRole.CUSTOMERS),
        _chunk("C1", "교체에는 상당한 비용과 시간이 필요하다. 고객 시스템에 당사 솔루션이 통합되어 있다."),
    ]
    duplicate = [*baseline, _chunk("C1-DUP", first)]
    formatted = [
        _chunk("C1", "# 새 제목\n\n-   고객 시스템에   당사 솔루션이 통합되어 있다.\n\n교체에는 상당한 비용과 시간이 필요하다."),
        _chunk("C2", second, role=SectionRole.CUSTOMERS),
    ]
    summary = _chunk(
        "SUMMARY",
        "회사는 강력한 고객 lock-in을 보유한다.",
        source_type=SourceType.GENERATED_SUMMARY,
    ).model_copy(update={"metadata": {"generated_summary": True}})
    boilerplate = _chunk(
        "BOILER",
        "This administrative notice contains no material source assertion.",
        role=SectionRole.OTHER,
    )

    assert _selected_keys(shuffled) == expected
    assert _selected_keys(duplicate) == expected
    assert _selected_keys(formatted) == expected
    assert _selected_keys([summary, *baseline]) == expected
    assert _selected_keys([*baseline, boilerplate]) == expected


def test_atomic_split_is_idempotent_for_long_single_cell_dart_table() -> None:
    sentences = [
        f"계약 고객은 통합 시스템을 사용하며 전환에는 검증 비용이 발생합니다. 문장 {index}"
        for index in range(40)
    ]
    markdown = "| Column 1 |\n|---|\n| " + "<br><br>".join(sentences) + " |"

    first = split_atomic_evidence_text(markdown)
    rebuilt = "\n\n".join(reversed(first))
    second = split_atomic_evidence_text(rebuilt)

    assert len(first) > 1
    assert set(second) == set(first)


def test_atomic_split_preserves_short_relational_table_row() -> None:
    markdown = "| 고객 | 계약기간 | 갱신율 |\n|---|---:|---:|\n| A사 | 5년 | 93% |"

    assert split_atomic_evidence_text(markdown) == [
        "A사 | 5년 | 93%",
        "고객 | 계약기간 | 갱신율",
    ]


def test_atomic_replay_identity_uses_evidence_key_not_full_prompt(tmp_path) -> None:
    unit = build_atomic_evidence_units(
        [_chunk("C1", "고객 시스템 통합으로 전환 비용이 발생한다.")],
        issuer_id="ISSUER-1",
    )[0]
    request = build_atomic_evidence_request(unit, issuer_id="ISSUER-1")
    changed_user = request.user + "\nPresentation-only suffix"
    changed = request.model_copy(
        update={
            "user": changed_user,
            "input_sha256": hashlib.sha256(
                f"{request.system}\n\n{changed_user}".encode("utf-8")
            ).hexdigest(),
        }
    )
    cache = LLMReplayCache(
        tmp_path,
        experiment_id="experiment",
        summary_model="gpt-5-nano",
        moat_model="gpt-5.6-luna",
        summary_reasoning_effort="low",
        atomic_reasoning_effort="low",
        moat_reasoning_effort="medium",
        engine_version="0.8.0",
    )

    assert cache.identity(request, AtomicEvidenceExtraction)[0] == cache.identity(
        changed,
        AtomicEvidenceExtraction,
    )[0]


def test_atomic_request_exposes_issuer_identity_for_third_party_guard() -> None:
    unit = build_atomic_evidence_units(
        [_chunk("C1", "GLOBAL BLUE / 글로벌 1위 사업자")],
        issuer_id="204620",
    )[0]

    request = build_atomic_evidence_request(
        unit,
        issuer_id="204620",
        issuer_name="글로벌텍스프리",
    )

    assert "Issuer ID: 204620" in request.user
    assert "Issuer name: 글로벌텍스프리" in request.user
    assert request.metadata["issuer_name"] == "글로벌텍스프리"


def test_atomic_request_prioritizes_explicit_observable_outcome_anchors() -> None:
    unit = build_atomic_evidence_units(
        [_chunk("C1", "회사의 제품은 시장점유율 1위이다.")],
        issuer_id="ISSUER-1",
    )[0]

    request = build_atomic_evidence_request(
        unit,
        issuer_id="ISSUER-1",
        issuer_name="Issuer",
    )

    assert request.metadata["prompt_version"] == "atomic-evidence-classifier/10"
    assert request.metadata["rubric_version"] == "structural-moat-rubric/9"
    assert "Positive routing priority" in request.system
    assert "does not mean an observable outcome should be returned as NONE" in request.system
    assert "issuer-owned product, brand, or platform outcome belongs to COMPANY" in request.system
    assert "Do not use PRODUCT_CATEGORY for an issuer-owned product's market share" in request.system


def test_atomic_api_schema_uses_explicit_aliases_without_changing_internal_fields() -> None:
    schema = AtomicEvidenceExtraction.model_json_schema()
    properties = schema["properties"]

    assert {
        "relevant",
        "type",
        "direction",
        "fact",
        "mechanism",
        "scope",
        "segment",
        "subject",
        "predicate",
        "horizon",
        "metric",
    } <= set(properties)
    assert "is_investment_relevant" not in properties
    parsed = AtomicEvidenceExtraction.model_validate(
        {
            "relevant": True,
            "type": "SWITCHING_COST",
            "fact": "Five-year contracts create switching friction.",
            "mechanism": ["long contract"],
            "direction": "MOAT_POSITIVE",
            "scope": "COMPANY",
            "segment": None,
            "horizon": "LONG",
            "subject": "customer contract",
            "predicate": "switching friction",
        }
    )

    assert parsed.is_investment_relevant is True
    assert parsed.claim_subject == "customer contract"
    assert "is_investment_relevant" in parsed.model_dump()


def test_minimal_atomic_output_is_enriched_from_source_without_llm_arithmetic() -> None:
    unit = build_atomic_evidence_units(
        [
            _chunk(
                "IR",
                "회사는 2026년 생산능력(CAPA)을 30% 확대할 계획이다.",
                role=SectionRole.GUIDANCE,
                source_type=SourceType.IR,
            )
        ],
        issuer_id="ISSUER-1",
    )[0]
    extraction = AtomicEvidenceExtraction(
        is_investment_relevant=True,
        evidence_type=EvidenceType.OPERATING_DRIVER,
        direction=EvidenceDirection.NEUTRAL,
        fact="2026년 생산능력을 30% 확대할 계획",
        mechanism=["capacity expansion"],
        economic_scope=EconomicScope.COMPANY,
        claim_subject="production capacity",
        claim_predicate="planned expansion",
        claim_horizon="2026",
        claim_metric="capacity growth",
    )

    judgment = atomic_extraction_to_judgment(extraction, unit)

    assert judgment.statement_type == StatementType.MANAGEMENT_CLAIM
    assert judgment.strength == 0.5
    assert judgment.forward_driver_type == ForwardDriverType.CAPACITY
    assert judgment.dcf_links == [DcfLink.REVENUE, DcfLink.CAPEX]
    assert any(str(metric.value) == "30" and metric.unit == "%" for metric in judgment.metrics)
    assert judgment.period == "2026년"


def _card(evidence_id: str, predicate: str, reliability: float = 0.8) -> EvidenceCard:
    return EvidenceCard(
        evidence_id=evidence_id,
        source_chunk_id=f"C-{evidence_id}",
        node_ids=[f"N-{evidence_id}"],
        evidence_type=EvidenceType.SWITCHING_COST,
        statement_type=StatementType.DISCLOSED_FACT,
        fact=predicate,
        mechanism=["workflow integration", "switching friction"],
        direction=EvidenceDirection.MOAT_POSITIVE,
        source_type=SourceType.DART,
        economic_scope=EconomicScope.COMPANY,
        raw_quote=predicate,
        reliability=reliability,
        claim_signature=CanonicalClaimSignature(
            moat_source=EvidenceType.SWITCHING_COST,
            subject="customer workflow",
            predicate=predicate,
            direction=EvidenceDirection.MOAT_POSITIVE,
            horizon="LONG",
        ),
    )


def test_claim_set_and_python_reducer_are_commutative_associative_idempotent() -> None:
    first = _card("E1", "integration friction", 0.9)
    duplicate_claim = _card("E2", "integration friction", 0.7)
    second = _card("E3", "qualification cost", 0.8)
    inputs = [first, duplicate_claim, second]
    canonical, clusters = build_canonical_claim_set(inputs, issuer_id="ISSUER-1")

    assert len(canonical) == 2
    assert any(cluster.supporting_evidence_ids == ["E2"] for cluster in clusters)
    variants = [
        inputs,
        list(reversed(inputs)),
        [*inputs, *inputs],
        [first, second, duplicate_claim],
    ]
    scores = [
        derive_moat_score(
            None,
            variant,
            issuer_id="ISSUER-1",
            as_of=date(2026, 5, 31),
        )
        for variant in variants
    ]

    assert {score.economic_moat_score for score in scores} == {scores[0].economic_moat_score}
    assert {tuple(score.canonical_claim_ids) for score in scores} == {
        tuple(scores[0].canonical_claim_ids)
    }
    assert {
        tuple((item.evidence_type, item.score, tuple(item.evidence_ids)) for item in score.mechanisms)
        for score in scores
    } == {
        tuple(
            (item.evidence_type, item.score, tuple(item.evidence_ids))
            for item in scores[0].mechanisms
        )
    }


def test_metamorphic_reducer_contract_keeps_out_of_scope_negative_claims() -> None:
    negative = _card("E-NEG", "industry regulation pressure")
    negative.direction = EvidenceDirection.MOAT_NEGATIVE
    negative.economic_scope = EconomicScope.INDUSTRY
    neutral = _card("E-NEUTRAL", "industry context")
    neutral.direction = EvidenceDirection.NEUTRAL
    neutral.economic_scope = EconomicScope.INDUSTRY

    assert _is_reducer_input(negative) is True
    assert _is_reducer_input(neutral) is False
