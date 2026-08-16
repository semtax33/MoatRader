from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from moatrader.canonical.models import SourceType, StatementType
from moatrader.context import (
    DynamicTokenBudgetAllocator,
    EvidencePackBuilder,
    build_financial_feature_vector,
)
from moatrader.evidence.models import (
    CompanyDossier,
    ClaimCluster,
    EvidenceCard,
    EvidenceDirection,
    EvidenceType,
    MoatRankRefinement,
    MoatRankRefinementStatus,
)
from moatrader.financial import FinancialSnapshot, FinancialSnapshotBuilder
from moatrader.financial.snapshot import DerivedMetric, FinancialPoint, FinancialSeries
from moatrader.screening import CandidateInput, ValueMoatRanker
from moatrader.semantic import SemanticChunker
from moatrader.semantic.chunker import SemanticChunk

from conftest import build_dart_bundle


def test_evidence_pack_has_three_layers_and_traceable_ids():
    bundle = build_dart_bundle(
        "<html><body><h1>II. 사업의 내용</h1><p>신규 공급자 인증에는 18개월이 필요합니다.</p></body></html>"
    )
    chunk = SemanticChunker().chunk(bundle)[0]
    card = EvidenceCard(
        evidence_id="E001",
        source_chunk_id=chunk.chunk_id,
        node_ids=chunk.node_ids,
        evidence_type=EvidenceType.SWITCHING_COST,
        statement_type=StatementType.DISCLOSED_FACT,
        fact="신규 공급자 인증에는 18개월이 필요함",
        raw_quote="신규 공급자 인증에는 18개월이 필요합니다.",
        mechanism=["long qualification", "switching friction"],
        direction=EvidenceDirection.MOAT_POSITIVE,
        strength=0.8,
        reliability=0.8,
        source_type=SourceType.DART,
    )
    dossier = CompanyDossier(
        issuer_name="테스트전자",
        as_of=bundle.metadata.available_at,
        source_document_ids=[bundle.metadata.source_document_id],
        business_summary="고객 승인 절차가 길다.",
        evidence=[card],
    )
    snapshot = FinancialSnapshotBuilder().build([bundle], as_of=bundle.metadata.available_at)
    pack = EvidencePackBuilder().build(dossier, snapshot, [chunk])
    assert "# L1. Structural Summary" in pack.markdown
    assert "# L2. Grounded Structural Evidence Cards" in pack.markdown
    assert "# L3. Raw Evidence Appendix" in pack.markdown
    assert "[E001]" in pack.markdown
    assert chunk.chunk_id in pack.raw_chunk_ids
    assert "## Financial Economics" not in pack.markdown
    assert "고객 승인 절차가 길다." not in pack.markdown
    assert "Display-only LLM summaries are intentionally excluded" in pack.markdown


def test_compact_factor_pack_preserves_counterevidence_and_uses_on_demand_quotes():
    bundle = build_dart_bundle(
        "<html><body><h1>II. 사업의 내용</h1><p>고객 계약은 5년이다. 경쟁사의 무상 전환 도구가 출시됐다.</p></body></html>"
    )
    chunk = SemanticChunker().chunk(bundle)[0]
    positive = EvidenceCard(
        evidence_id="E_POS",
        claim_id="CL_POS",
        source_chunk_id=chunk.chunk_id,
        node_ids=chunk.node_ids,
        evidence_type=EvidenceType.SWITCHING_COST,
        statement_type=StatementType.DISCLOSED_FACT,
        fact="고객 계약은 5년",
        raw_quote="고객 계약은 5년이다.",
        mechanism=["장기 계약", "전환 마찰"],
        direction=EvidenceDirection.MOAT_POSITIVE,
        source_type=SourceType.DART,
    )
    negative = positive.model_copy(
        update={
            "evidence_id": "E_NEG",
            "claim_id": "CL_NEG",
            "fact": "경쟁사의 무상 전환 도구 출시",
            "raw_quote": "경쟁사의 무상 전환 도구가 출시됐다.",
            "mechanism": [],
            "direction": EvidenceDirection.MOAT_NEGATIVE,
        }
    )
    dossier = CompanyDossier(
        issuer_name="테스트전자",
        as_of=bundle.metadata.available_at,
        source_document_ids=[bundle.metadata.source_document_id],
        evidence=[positive, negative],
    )
    snapshot = FinancialSnapshotBuilder().build([bundle], as_of=bundle.metadata.available_at)
    pack = EvidencePackBuilder().build(
        dossier,
        snapshot,
        [],
        [
            ClaimCluster(
                claim_id="CL_POS",
                canonical_evidence_id="E_POS",
                supporting_evidence_ids=["E_POS_2"],
            ),
            ClaimCluster(
                claim_id="CL_NEG",
                canonical_evidence_id="E_NEG",
            ),
        ],
    )

    assert pack.claim_ids == ["CL_NEG", "CL_POS"]
    assert pack.counterevidence_ids == ["E_NEG"]
    assert "[E_POS_2]" in pack.markdown
    assert "[E_NEG]" in pack.markdown
    assert positive.raw_quote not in pack.markdown
    assert negative.raw_quote not in pack.markdown
    assert pack.raw_evidence_artifact == "evidence.jsonl"


def test_financial_feature_vector_is_exact_python_numeric_compression():
    timestamp = datetime(2025, 3, 31, tzinfo=timezone.utc)
    point = FinancialPoint(
        period=timestamp.date(),
        period_basis="FY",
        value=Decimal("1000"),
        unit="KRW",
        source_fact_ids=["F_REVENUE"],
        available_at=timestamp,
    )
    snapshot = FinancialSnapshot(
        as_of=timestamp,
        issuer_id="ISSUER",
        series=[FinancialSeries(concept="REVENUE", points=[point])],
        derived_metrics=[
            DerivedMetric(
                name="EBIT_MARGIN",
                period=timestamp.date(),
                value=Decimal("0.17"),
                unit="RATIO",
                derived_from_fact_ids=["F_REVENUE", "F_EBIT"],
            )
        ],
    )

    vector = build_financial_feature_vector(snapshot)

    assert "REVENUE|2025-03-31:1000:KRW|src=F_REVENUE" in vector.markdown
    assert "EBIT_MARGIN|2025-03-31:0.17:RATIO|src=F_EBIT,F_REVENUE" in vector.markdown
    assert vector.schema_version == "financial-feature-vector/1"


def test_value_moat_ranker_filters_and_orders_by_quality_adjusted_discount():
    timestamp = datetime(2026, 8, 13, tzinfo=timezone.utc)
    common = {
        "current_price": Decimal("60"),
        "dcf_fair_value": Decimal("100"),
        "moat_score": Decimal("8"),
        "model_confidence": Decimal("0.8"),
        "document_coverage": Decimal("0.9"),
        "valuation_as_of": timestamp,
        "price_as_of": timestamp,
    }
    candidates = [
        CandidateInput(issuer_id="1", ticker="AAA", **common),
        CandidateInput(
            issuer_id="2",
            ticker="BBB",
            **{
                **common,
                "current_price": Decimal("50"),
                "moat_score": Decimal("7"),
            },
        ),
        CandidateInput(
            issuer_id="3",
            ticker="LOW",
            **{**common, "moat_score": Decimal("3")},
        ),
    ]
    ranked = ValueMoatRanker().rank(candidates)
    assert [item.ticker for item in ranked] == ["BBB", "AAA"]
    assert all(item.price_to_dcf < 1 for item in ranked)


def test_value_moat_ranker_refines_only_within_equal_public_score() -> None:
    timestamp = datetime(2026, 8, 13, tzinfo=timezone.utc)
    common = {
        "current_price": Decimal("60"),
        "dcf_fair_value": Decimal("100"),
        "moat_score": Decimal("6"),
        "model_confidence": Decimal("0.8"),
        "document_coverage": Decimal("0.9"),
        "valuation_as_of": timestamp,
        "price_as_of": timestamp,
    }
    def refinement(mechanism: float) -> dict[str, object]:
        return {
            "rank_refinement": MoatRankRefinement(
                mechanism_component=mechanism,
                outcome_component=0,
                durability_component=1,
                counter_component=1,
            ),
            "rank_refinement_status": MoatRankRefinementStatus.STABLE_COMPONENTS,
        }

    ranked = ValueMoatRanker().rank(
        [
            CandidateInput(
                issuer_id="1", ticker="REFINE_HIGH", **refinement(4), **common
            ),
            CandidateInput(
                issuer_id="2", ticker="REFINE_LOW", **refinement(2), **common
            ),
            CandidateInput(
                issuer_id="4",
                ticker="PUBLIC_HIGH",
                moat_score=Decimal("7"),
                **refinement(0),
                **{key: value for key, value in common.items() if key != "moat_score"},
            ),
            CandidateInput(
                issuer_id="3",
                ticker="PUBLIC_FAIL",
                moat_score=Decimal("4"),
                **refinement(4),
                **{key: value for key, value in common.items() if key != "moat_score"},
            ),
        ]
    )

    assert [item.ticker for item in ranked] == [
        "PUBLIC_HIGH",
        "REFINE_HIGH",
        "REFINE_LOW",
    ]
    assert ranked[0].moat_rank_key[0] > ranked[1].moat_rank_key[0]
    assert ranked[1].moat_percentile > ranked[2].moat_percentile
    assert ranked[1].moat_score == ranked[2].moat_score == Decimal("6")


def test_value_moat_ranker_preserves_tie_when_refinement_is_incomplete() -> None:
    timestamp = datetime(2026, 8, 13, tzinfo=timezone.utc)
    common = {
        "current_price": Decimal("60"),
        "dcf_fair_value": Decimal("100"),
        "moat_score": Decimal("6"),
        "model_confidence": Decimal("0.8"),
        "document_coverage": Decimal("0.9"),
        "valuation_as_of": timestamp,
        "price_as_of": timestamp,
    }
    stable = MoatRankRefinement(
        mechanism_component=4,
        outcome_component=0,
        durability_component=1,
        counter_component=1,
    )
    ranked = ValueMoatRanker().rank(
        [
            CandidateInput(
                issuer_id="1",
                ticker="HAS_COMPONENTS",
                rank_refinement=stable,
                rank_refinement_status=MoatRankRefinementStatus.STABLE_COMPONENTS,
                **common,
            ),
            CandidateInput(issuer_id="2", ticker="PUBLIC_ONLY", **common),
        ]
    )

    assert ranked[0].moat_percentile == ranked[1].moat_percentile
    assert ranked[0].moat_rank_key == ranked[1].moat_rank_key == (Decimal("6"),)


def test_allocator_treats_unretrieved_chunks_as_low_relevance() -> None:
    missed = SemanticChunk(
        chunk_id="MISS",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown="missed",
        token_count=700,
    )
    hit = missed.model_copy(update={"chunk_id": "HIT", "node_ids": ["N2"], "markdown": "hit"})
    allocation = DynamicTokenBudgetAllocator(
        model_context_tokens=9_000,
        prompt_reserve_tokens=8_000,
    ).allocate([missed, hit], relevance={"HIT": 1.0})

    assert [chunk.chunk_id for chunk in allocation.selected] == ["HIT"]


def test_structural_pack_excludes_positive_financial_outcomes() -> None:
    bundle = build_dart_bundle("<html><body><p>Operating margin increased.</p></body></html>")
    chunk = SemanticChunker().chunk(bundle)[0]
    card = EvidenceCard(
        evidence_id="OUTCOME",
        source_chunk_id=chunk.chunk_id,
        node_ids=chunk.node_ids,
        evidence_type=EvidenceType.MARGIN_STABILITY,
        statement_type=StatementType.DISCLOSED_FACT,
        fact="Operating margin increased.",
        raw_quote="Operating margin increased.",
        direction=EvidenceDirection.MOAT_POSITIVE,
        source_type=SourceType.DART,
    )
    dossier = CompanyDossier(
        issuer_name="Fixture",
        as_of=bundle.metadata.available_at,
        source_document_ids=[bundle.metadata.source_document_id],
        evidence=[card],
    )

    pack = EvidencePackBuilder().build(
        dossier,
        FinancialSnapshotBuilder().build([bundle], as_of=bundle.metadata.available_at),
        [],
    )

    assert "[OUTCOME]" not in pack.markdown
    assert "No positive company-specific structural evidence" in pack.markdown
