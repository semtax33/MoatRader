from __future__ import annotations

from moatrader.canonical.models import SectionRole, SourceRef, SourceType
from moatrader.runner.selection import batch_evidence_chunks, select_evidence_chunks
from moatrader.semantic import SemanticChunk


def _chunk(index: int, role: SectionRole, text: str, tokens: int = 100) -> SemanticChunk:
    return SemanticChunk(
        chunk_id=f"C{index}",
        document_id="DOC1",
        section_path=[role.value, str(index)],
        section_role=role,
        node_ids=[f"N{index}"],
        chunk_type="paragraph",
        markdown=text,
        token_count=tokens,
        source_refs=[SourceRef(source_type=SourceType.DART, document_id="DOC1")],
    )


def test_evidence_selection_prefers_moat_and_counterevidence() -> None:
    chunks = [
        _chunk(1, SectionRole.GOVERNANCE, "이사회 및 임원 보수"),
        _chunk(2, SectionRole.BUSINESS, "장기 고객 계약과 재계약 유지율"),
        _chunk(3, SectionRole.RISK, "고객 집중과 대체 기술 위험"),
        _chunk(4, SectionRole.PRODUCTS, "특허 기반 독점 제품과 가격 경쟁력"),
    ]

    selected = select_evidence_chunks(chunks, 2)

    assert {chunk.chunk_id for chunk in selected} <= {"C2", "C3", "C4"}
    assert "C1" not in {chunk.chunk_id for chunk in selected}


def test_evidence_batches_respect_document_and_token_limit() -> None:
    chunks = [
        _chunk(1, SectionRole.BUSINESS, "a", 400),
        _chunk(2, SectionRole.RISK, "b", 400),
        _chunk(3, SectionRole.PRODUCTS, "c", 400).model_copy(update={"document_id": "DOC2"}),
    ]

    batches = batch_evidence_chunks(chunks, 1_000)

    assert [[chunk.chunk_id for chunk in batch] for batch in batches] == [["C1", "C2"], ["C3"]]
