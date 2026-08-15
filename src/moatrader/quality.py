from __future__ import annotations

from pydantic import Field

from moatrader.canonical.models import CanonicalDocumentBundle, ContractModel


class ParserQualityGateConfig(ContractModel):
    minimum_text_retention: float = Field(default=0.95, ge=0.0, le=1.0)
    minimum_numeric_retention: float = Field(default=0.99, ge=0.0, le=1.0)
    minimum_structured_fact_retention: float = Field(default=0.99, ge=0.0, le=1.0)
    require_table_count_match: bool = True


class ParserQualityAssessment(ContractModel):
    source_document_id: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def assess_parser_quality(
    bundle: CanonicalDocumentBundle,
    config: ParserQualityGateConfig | None = None,
) -> ParserQualityAssessment:
    config = config or ParserQualityGateConfig()
    quality = bundle.quality
    failures: list[str] = []
    if quality.text_retention is None:
        failures.append("visible text retention is unavailable")
    elif quality.text_retention < config.minimum_text_retention:
        failures.append(
            f"visible text retention {quality.text_retention:.4f} is below "
            f"{config.minimum_text_retention:.4f}"
        )
    if config.require_table_count_match and quality.raw_table_count != quality.ast_table_count:
        failures.append(
            f"raw/canonical table counts differ: {quality.raw_table_count}/{quality.ast_table_count}"
        )
    if quality.raw_numeric_cell_count and quality.numeric_retention is None:
        failures.append("numeric cell retention is unavailable")
    elif quality.numeric_retention is not None and quality.numeric_retention < config.minimum_numeric_retention:
        failures.append(
            f"numeric cell retention {quality.numeric_retention:.4f} is below "
            f"{config.minimum_numeric_retention:.4f}"
        )
    if quality.raw_structured_fact_count and quality.structured_fact_retention is None:
        failures.append("structured fact retention is unavailable")
    elif (
        quality.structured_fact_retention is not None
        and quality.structured_fact_retention < config.minimum_structured_fact_retention
    ):
        failures.append(
            f"structured fact retention {quality.structured_fact_retention:.4f} is below "
            f"{config.minimum_structured_fact_retention:.4f}"
        )
    return ParserQualityAssessment(
        source_document_id=bundle.metadata.source_document_id,
        passed=not failures,
        failures=failures,
        warnings=quality.warnings,
    )
