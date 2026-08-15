from __future__ import annotations

from moatrader.canonical.models import SectionRole, SourceRef, SourceType
from moatrader.semantic import SemanticChunk, deduplicate_chunks


def _chunk(chunk_id: str, markdown: str) -> SemanticChunk:
    return SemanticChunk(
        chunk_id=chunk_id,
        document_id="DOC-1",
        section_path=["Financial statements"],
        section_role=SectionRole.FINANCIALS,
        node_ids=[f"N-{chunk_id}"],
        chunk_type="table",
        markdown=markdown,
        token_count=10,
        source_refs=[
            SourceRef(
                source_type=SourceType.DART,
                document_id="DOC-1",
            )
        ],
    )


def test_chunk_dedup_keeps_exact_and_numeric_change_semantics() -> None:
    exact = "Revenue was 1,000 and operating income was 200."
    changed = "Revenue was 1,100 and operating income was 220."
    result = deduplicate_chunks(
        [
            _chunk("C1", exact),
            _chunk("C2", exact),
            _chunk("C3", changed),
            _chunk("C4", "Completely unrelated governance disclosure."),
        ]
    )

    assert [chunk.chunk_id for chunk in result.kept] == ["C1", "C3", "C4"]
    assert [(item.duplicate_chunk_id, item.canonical_chunk_id) for item in result.duplicates] == [
        ("C2", "C1")
    ]
    assert any(
        change.older_chunk_id == "C3" and change.newer_chunk_id == "C1"
        for change in result.changes
    )


def test_chunk_dedup_does_not_compare_across_section_roles() -> None:
    first = _chunk("C1", "Identical disclosure 1,000.")
    second = _chunk("C2", "Identical disclosure 1,000.").model_copy(
        update={"section_role": SectionRole.RISK}
    )

    result = deduplicate_chunks([first, second])

    assert [chunk.chunk_id for chunk in result.kept] == ["C1", "C2"]
    assert not result.duplicates


def test_long_numeric_update_survives_shingle_candidate_blocking() -> None:
    rows = " ".join(
        f"Product {index} revenue {index * 100:,} operating margin 20 percent."
        for index in range(100)
    )
    updated = rows.replace("Product 50 revenue 5,000", "Product 50 revenue 5,500")

    result = deduplicate_chunks([_chunk("C1", rows), _chunk("C2", updated)])

    assert [chunk.chunk_id for chunk in result.kept] == ["C1", "C2"]
    assert len(result.changes) == 1
    assert result.changes[0].older_chunk_id == "C2"
    assert result.changes[0].newer_chunk_id == "C1"
