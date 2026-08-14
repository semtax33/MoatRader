from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from moatrader.canonical.models import SourceType, StatementType
from moatrader.context import EvidencePackBuilder
from moatrader.evidence.models import (
    CompanyDossier,
    EvidenceCard,
    EvidenceDirection,
    EvidenceType,
)
from moatrader.financial import FinancialSnapshotBuilder
from moatrader.screening import CandidateInput, ValueMoatRanker
from moatrader.semantic import SemanticChunker

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
    assert "# L1. Structured Summary" in pack.markdown
    assert "# L2. Evidence Cards" in pack.markdown
    assert "# L3. Raw Evidence Appendix" in pack.markdown
    assert "[E001]" in pack.markdown
    assert chunk.chunk_id in pack.raw_chunk_ids


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
            **{**common, "current_price": Decimal("50"), "moat_score": Decimal("7")},
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

