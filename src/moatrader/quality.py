from __future__ import annotations

import re

from pydantic import Field

from moatrader.canonical.models import CanonicalDocumentBundle, ContractModel, TableNode


class ParserQualityGateConfig(ContractModel):
    minimum_text_retention: float = Field(default=0.95, ge=0.0, le=1.0)
    minimum_numeric_retention: float = Field(default=0.99, ge=0.0, le=1.0)
    minimum_structured_fact_retention: float = Field(default=0.99, ge=0.0, le=1.0)
    require_table_count_match: bool = True
    require_financial_table_semantics: bool = True


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
    warnings = list(quality.warnings)
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
    section_period_context = {
        tuple(node.section_path)
        for node in bundle.ast.walk()
        if isinstance(node, TableNode)
        and (node.period is not None or re.search(r"20\d{2}", node.raw_text))
    }
    for node in bundle.ast.walk():
        if not isinstance(node, TableNode):
            continue
        numeric_cells = [cell for row in node.rows for cell in row.cells if cell.numeric_value is not None]
        if not numeric_cells:
            continue
        section_text = " ".join(node.section_path)
        financial_context = bool(
            re.search(
                r"요약재무|(?:^|[ >])(?:연결|별도)?재무제표(?:$|[ >])|재무상태표|손익계산서|포괄손익|현금흐름표|자본변동표|financial\s+statements?",
                section_text,
                re.IGNORECASE,
            )
            and "주석" not in section_text
        )
        table_width = max((len(row.cells) for row in node.rows), default=0)
        is_data_table = table_width >= 2 and len(node.rows) >= 2 and len(numeric_cells) >= 2
        nonempty_headers = [header for header in node.column_headers if any(item.strip() for item in header.path)]
        has_period_context = (
            node.period is not None
            or tuple(node.section_path) in section_period_context
            or any(
                re.search(
                    r"20\d{2}|제\s*\d+\s*기|당기|전기|분기|반기|current|prior",
                    " ".join(header.path),
                    re.I,
                )
                for header in node.column_headers
            )
        )
        has_unit_context = node.unit is not None or any(cell.unit is not None for cell in numeric_cells)
        if financial_context and is_data_table and config.require_financial_table_semantics:
            if not nonempty_headers:
                failures.append(f"financial table {node.node_id} has numeric values but no column-header mapping")
            if not has_period_context:
                failures.append(f"financial table {node.node_id} has no reporting-period context")
            if not has_unit_context:
                failures.append(f"financial table {node.node_id} has no unit context")
        else:
            if not nonempty_headers:
                warnings.append(f"numeric table {node.node_id} has no column-header mapping")
            if not has_unit_context:
                warnings.append(f"numeric table {node.node_id} has no unit context")
    return ParserQualityAssessment(
        source_document_id=bundle.metadata.source_document_id,
        passed=not failures,
        failures=failures,
        warnings=list(dict.fromkeys(warnings)),
    )
