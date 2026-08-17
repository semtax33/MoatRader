from __future__ import annotations

from datetime import datetime, timezone

from moatrader.canonical.models import (
    CanonicalDocumentBundle,
    DocumentAST,
    DocumentMetadata,
    ProvenanceIndex,
    QualityMetrics,
    SourceType,
)
from moatrader.quality import ParserQualityGateConfig, assess_parser_quality

from conftest import build_dart_bundle


def _bundle(quality: QualityMetrics) -> CanonicalDocumentBundle:
    return CanonicalDocumentBundle(
        metadata=DocumentMetadata(
            source_type=SourceType.DART,
            source_document_id="DOC1",
            available_at=datetime.now(timezone.utc),
            availability_source="fixture",
            raw_sha256="0" * 64,
            parser_version="fixture",
        ),
        ast=DocumentAST(document_id="DOC1"),
        provenance=ProvenanceIndex(),
        quality=quality,
    )


def test_quality_gate_rejects_low_retention_and_missing_table() -> None:
    assessment = assess_parser_quality(
        _bundle(
            QualityMetrics(
                raw_visible_chars=100,
                ast_chars=80,
                text_retention=0.8,
                raw_table_count=2,
                ast_table_count=1,
                raw_numeric_cell_count=10,
                numeric_cell_count=8,
                numeric_retention=0.8,
            )
        ),
        ParserQualityGateConfig(minimum_text_retention=0.95),
    )

    assert assessment.passed is False
    assert len(assessment.failures) == 3


def test_quality_gate_can_allow_nested_table_count_difference() -> None:
    assessment = assess_parser_quality(
        _bundle(QualityMetrics(text_retention=1.0, raw_table_count=2, ast_table_count=1)),
        ParserQualityGateConfig(require_table_count_match=False),
    )

    assert assessment.passed is True


def test_quality_gate_rejects_tagged_facts_that_were_not_canonicalized() -> None:
    assessment = assess_parser_quality(
        _bundle(
            QualityMetrics(
                text_retention=1.0,
                raw_structured_fact_count=10,
                structured_fact_count=8,
                structured_fact_retention=0.8,
            )
        )
    )

    assert assessment.passed is False
    assert assessment.failures == [
        "structured fact retention 0.8000 is below 0.9900"
    ]


def test_quality_gate_rejects_numeric_financial_table_without_semantics() -> None:
    bundle = build_dart_bundle(
        "<html><body><h1>재무제표</h1><table>"
        "<tr><td>매출액</td><td>100</td></tr><tr><td>영업이익</td><td>10</td></tr>"
        "</table></body></html>"
    )

    assessment = assess_parser_quality(bundle)

    assert assessment.passed is False
    assert any("column-header mapping" in failure for failure in assessment.failures)
    assert any("reporting-period context" in failure for failure in assessment.failures)
    assert any("unit context" in failure for failure in assessment.failures)


def test_quality_gate_recognizes_statement_specific_korean_heading() -> None:
    bundle = build_dart_bundle(
        "<html><body><h1>연결 재무상태표</h1>"
        "<table><tr><td>자산</td><td>100</td></tr>"
        "<tr><td>부채</td><td>50</td></tr></table></body></html>"
    )

    assessment = assess_parser_quality(bundle)

    assert any("column-header mapping" in failure for failure in assessment.failures)


def test_quality_gate_rejects_ir_pdf_table_with_collapsed_numeric_grid() -> None:
    bundle = build_dart_bundle(
        "<html><body><table>"
        "<tr><th>구분</th><th>20.3Q</th><th>20.4Q</th><th>21.1Q</th>"
        "<th>21.2Q</th><th>21.3Q</th></tr>"
        "<tr><td>Revenue 58,252 56,594 100,754 52,918<br>"
        "GP 30,886 27,382 49,591 24,348<br>"
        "OP 14,397 6,857 30,765 6,234</td>"
        "<td></td><td></td><td></td><td></td><td>94,261</td></tr>"
        "<tr><td></td><td></td><td></td><td></td><td></td><td>45,889</td></tr>"
        "</table></body></html>"
    )
    table = next(node for node in bundle.ast.walk() if node.kind == "table")
    table.attributes["table_extraction_strategy"] = "lines"
    bundle = bundle.model_copy(
        update={
            "metadata": bundle.metadata.model_copy(
                update={"source_type": SourceType.IR}
            )
        }
    )

    assessment = assess_parser_quality(bundle)

    assert assessment.passed is False
    assert any("collapsed multi-column grid" in item for item in assessment.failures)


def test_dart_summary_financial_table_preserves_unknown_source_omitted_unit() -> None:
    bundle = build_dart_bundle(
        "<html><body><h1>1. 요약재무정보</h1><table>"
        "<tr><th>과목</th><th>제25기 3분기말</th><th>제24기말</th></tr>"
        "<tr><td>자산총계</td><td>70,277,930,552</td><td>74,421,441,480</td></tr>"
        "<tr><td>부채총계</td><td>19,399,346,522</td><td>19,904,477,492</td></tr>"
        "</table></body></html>",
        period_end="2025-09-30",
    )

    assessment = assess_parser_quality(bundle)

    assert assessment.passed is True
    assert any(
        "preserves unknown unit because the source section omits unit context" in warning
        for warning in assessment.warnings
    )


def test_dart_summary_unit_marker_for_later_table_does_not_fail_earlier_unknown_unit() -> None:
    bundle = build_dart_bundle(
        "<html><body><h1>1. 요약재무정보</h1>"
        "<p>가. 요약 연결 재무정보</p>"
        "<table><tr><th>구분</th><th>2026년 3월말</th></tr>"
        "<tr><td>자산총계</td><td>2,781,388</td></tr>"
        "<tr><td>부채총계</td><td>2,057,513</td></tr></table>"
        "<p>나. 요약 별도재무정보</p><p>(단위 : 백만원)</p>"
        "<table><tr><th>구분</th><th>2026년 3월말</th></tr>"
        "<tr><td>자산총계</td><td>1,000</td></tr>"
        "<tr><td>부채총계</td><td>500</td></tr></table>"
        "</body></html>",
        period_end="2026-03-31",
    )

    assessment = assess_parser_quality(bundle)

    assert assessment.passed is True
    assert any(
        "preserves unknown unit because the source section omits unit context" in warning
        for warning in assessment.warnings
    )
