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
