from __future__ import annotations

import re

from pydantic import Field

from moatrader.canonical.models import (
    CanonicalDocumentBundle,
    ContractModel,
    SourceType,
    TableNode,
)


_PACKED_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])[-(]?\d[\d,]*(?:\.\d+)?%?\)?"
)


def _collapsed_pdf_grid_failures(bundle: CanonicalDocumentBundle) -> list[str]:
    """Detect table-finder grids that preserved text but destroyed cell relations."""

    if bundle.metadata.source_type != SourceType.IR:
        return []
    failures: list[str] = []
    for node in bundle.ast.walk():
        if not isinstance(node, TableNode):
            continue
        if not node.attributes.get("table_extraction_strategy"):
            continue
        width = max((len(row.cells) for row in node.rows), default=0)
        if width < 4 or len(node.rows) < 3:
            continue
        header_rows = max(1, node.header_row_count)
        header = node.rows[0].cells
        if sum(bool(cell.normalized_text.strip()) for cell in header) < 3:
            continue
        for row in node.rows[header_rows:]:
            nonempty = [cell for cell in row.cells if cell.normalized_text.strip()]
            if len(nonempty) > 2:
                continue
            for cell in nonempty:
                text = cell.raw_text
                line_count = len([line for line in text.splitlines() if line.strip()])
                numeric_count = len(_PACKED_NUMERIC_TOKEN_RE.findall(text))
                if line_count >= 2 and numeric_count >= 2 * (width - 1):
                    failures.append(
                        f"IR table {node.node_id} has a collapsed multi-column grid: "
                        f"{numeric_count} numeric tokens are packed into one multiline cell"
                    )
                    break
            if failures and node.node_id in failures[-1]:
                break
    return failures


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
    failures.extend(_collapsed_pdf_grid_failures(bundle))
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
        ownership_or_structure_table = bool(
            re.search(
                r"지분구조|계열회사|주주\s*현황|ownership\s+structure|affiliates?",
                node.raw_text,
                re.IGNORECASE,
            )
        )
        financial_context = financial_context and not ownership_or_structure_table
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
        source_omitted_dart_summary_unit = (
            bundle.metadata.source_type == SourceType.DART
            and "요약재무" in section_text
        )
        if financial_context and is_data_table and config.require_financial_table_semantics:
            if not nonempty_headers:
                failures.append(f"financial table {node.node_id} has numeric values but no column-header mapping")
            if not has_period_context:
                failures.append(f"financial table {node.node_id} has no reporting-period context")
            if not has_unit_context:
                if source_omitted_dart_summary_unit:
                    warnings.append(
                        f"DART summary financial table {node.node_id} preserves unknown unit because the source section omits unit context"
                    )
                else:
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
