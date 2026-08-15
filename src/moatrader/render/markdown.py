from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from decimal import Decimal

from moatrader.canonical.models import (
    CanonicalDocumentBundle,
    CanonicalNode,
    FigureNode,
    ListNode,
    NoteNode,
    PageBreakNode,
    ParagraphNode,
    SectionNode,
    SourceRef,
    StructuredFact,
    TableNode,
    TableRow,
    UnknownBlockNode,
)


def _escape_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", "<br>") or " "


def _format_decimal(value: Decimal) -> str:
    if value == value.to_integral_value():
        return f"{int(value):,}"
    return f"{value.normalize():,}"


def _source_label(ref: SourceRef) -> str:
    location = ref.xpath or (f"page {ref.page}" if ref.page else None) or (f"slide {ref.slide}" if ref.slide else None)
    return f"{ref.source_type.value}:{ref.document_id}" + (f" @ {location}" if location else "")


class CanonicalMarkdownRenderer:
    """Render canonical data for humans/LLMs without becoming the source of truth."""

    def __init__(self, *, include_provenance: bool = True, include_node_ids: bool = True) -> None:
        self.include_provenance = include_provenance
        self.include_node_ids = include_node_ids

    def render_document(self, bundle: CanonicalDocumentBundle, *, include_facts: bool = True) -> str:
        metadata = bundle.metadata
        lines = [
            "# CANONICAL FINANCIAL DOCUMENT",
            "",
            "## 0. Metadata",
            "",
            f"- Company: {metadata.issuer_name or 'Unknown'}",
            f"- Ticker: {metadata.ticker or 'Unknown'}",
            f"- Source: {metadata.source_type.value}",
            f"- Source Document ID: {metadata.source_document_id}",
            f"- Document Type: {metadata.document_type.value}",
            f"- Available At: {metadata.available_at.isoformat()}",
            f"- Availability Precision: {metadata.availability_precision.value}",
            f"- Parser: {metadata.parser_version}",
            f"- Raw SHA-256: `{metadata.raw_sha256}`",
            "",
            "---",
            "",
            "# 1. Document Structure",
            "",
        ]
        for node in bundle.ast.children:
            rendered = self.render_node(node)
            if rendered:
                lines.extend([rendered, ""])
        if include_facts and bundle.facts:
            lines.extend(["---", "", "# 2. Structured Facts", "", self.render_facts(bundle.facts), ""])
        lines.extend(
            [
                "---",
                "",
                "# 3. Coverage",
                "",
                f"- Visible text retention: {self._ratio(bundle.quality.text_retention)}",
                f"- Tables: {bundle.quality.ast_table_count}/{bundle.quality.raw_table_count}",
                f"- Numeric cells: {bundle.quality.numeric_cell_count}/{bundle.quality.raw_numeric_cell_count}",
                f"- Numeric retention: {self._ratio(bundle.quality.numeric_retention)}",
                f"- Structured facts: {bundle.quality.structured_fact_count}/{bundle.quality.raw_structured_fact_count}",
                f"- Structured fact retention: {self._ratio(bundle.quality.structured_fact_retention)}",
                f"- Paragraphs: {bundle.quality.paragraph_count}",
                f"- Headings: {bundle.quality.heading_count}",
                f"- Unknown blocks: {bundle.quality.unknown_block_count}",
                f"- Duplicate text ratio: {self._ratio(bundle.quality.duplicate_text_ratio)}",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _ratio(value: float | None) -> str:
        return "n/a" if value is None else f"{value:.1%}"

    def render_node(self, node: CanonicalNode) -> str:
        if isinstance(node, SectionNode):
            return self.render_section(node)
        if isinstance(node, TableNode):
            return self.render_table(node)
        if isinstance(node, ParagraphNode):
            return self._with_anchor(node.normalized_text, node)
        if isinstance(node, NoteNode):
            text = "\n".join(f"> {line}" for line in node.normalized_text.splitlines())
            return self._with_anchor(text, node)
        if isinstance(node, ListNode):
            lines = []
            for index, item in enumerate(node.items, start=1):
                marker = f"{index}." if node.ordered else "-"
                lines.append(f"{marker} {item.normalized_text}")
            return self._with_anchor("\n".join(lines), node)
        if isinstance(node, FigureNode):
            label = node.caption or node.alt_text or "Figure"
            return self._with_anchor(f"[Figure: {label}]", node)
        if isinstance(node, PageBreakNode):
            return "---"
        if isinstance(node, UnknownBlockNode):
            return self._with_anchor(f"> [Unclassified] {node.normalized_text}", node)
        return ""

    def render_section(self, section: SectionNode) -> str:
        level = min(6, max(2, section.level + 1))
        role = f" `{section.role.value}`" if section.role else ""
        anchor = f" [{section.node_id}]" if self.include_node_ids else ""
        lines = [f"{'#' * level} {section.title_normalized}{anchor}{role}", ""]
        if self.include_provenance:
            lines.extend([f"_Source: {_source_label(section.source_refs[0])}_", ""])
        for child in section.children:
            rendered = self.render_node(child)
            if rendered:
                lines.extend([rendered, ""])
        return "\n".join(lines).rstrip()

    def render_table(
        self,
        table: TableNode,
        *,
        rows: Iterable[TableRow] | None = None,
        columns: Iterable[int] | None = None,
        include_footnotes: bool = True,
    ) -> str:
        selected_rows = list(rows) if rows is not None else table.rows[table.header_row_count :]
        title = table.caption or "Table"
        heading = f"### {title}" + (f" [{table.node_id}]" if self.include_node_ids else "")
        lines = [heading, ""]
        if table.unit:
            canonical = f" ({table.unit.canonical})" if table.unit.canonical else ""
            lines.append(f"- Unit: {table.unit.raw}{canonical}")
        if table.period:
            lines.append(f"- Period: {table.period.raw_label or table.period.fiscal_period or table.period.fiscal_year or table.period.kind.value}")
        if table.section_path:
            lines.append(f"- Section: {' > '.join(table.section_path)}")
        if self.include_provenance:
            lines.append(f"- Source: {_source_label(table.source_refs[0])}")
        if len(lines) > 2:
            lines.append("")

        full_width = len(table.column_headers)
        if not full_width and selected_rows:
            full_width = len(selected_rows[0].cells)
        selected_columns = list(columns) if columns is not None else list(range(full_width))
        if any(column < 0 or column >= full_width for column in selected_columns):
            raise ValueError("table column selection is out of bounds")
        headers = [
            (
                " > ".join(table.column_headers[column].path)
                if table.column_headers and table.column_headers[column].path
                else f"Column {column + 1}"
            )
            for column in selected_columns
        ]
        lines.append("| " + " | ".join(_escape_cell(value) for value in headers) + " |")
        numeric_columns = []
        for col in selected_columns:
            values = [row.cells[col].numeric_value for row in selected_rows if col < len(row.cells) and row.cells[col].normalized_text]
            numeric_columns.append(bool(values) and len(values) == sum(1 for value in values if value is not None))
        lines.append("|" + "|".join("---:" if numeric else "---" for numeric in numeric_columns) + "|")
        for row in selected_rows:
            values = [
                _escape_cell(row.cells[column].normalized_text)
                if column < len(row.cells)
                else ""
                for column in selected_columns
            ]
            lines.append("| " + " | ".join(values) + " |")
        if not selected_rows:
            lines.append("| " + " | ".join(" " for _ in selected_columns) + " |")
        if include_footnotes and table.footnotes:
            lines.append("")
            lines.extend(f"- Footnote {note.marker or ''}: {note.text}".rstrip() for note in table.footnotes)
        return "\n".join(lines)

    def render_facts(self, facts: Iterable[StructuredFact]) -> str:
        lines = ["| Concept | Period | Value | Unit | Scope | Fact ID |", "|---|---|---:|---|---|---|"]
        for fact in facts:
            if fact.period.instant:
                period = fact.period.instant.isoformat()
            elif fact.period.end:
                period = fact.period.end.isoformat()
            else:
                period = fact.period.raw_label or "unknown"
            value = _format_decimal(fact.numeric_value) if fact.numeric_value is not None else str(fact.value or "")
            unit = fact.unit.canonical if fact.unit and fact.unit.canonical else (fact.unit.raw if fact.unit else "")
            lines.append(
                f"| {_escape_cell(fact.canonical_concept or fact.concept)} | {period} | {value} | {unit} | {fact.scope.value} | {fact.fact_id} |"
            )
        return "\n".join(lines)

    def _with_anchor(self, rendered: str, node: CanonicalNode) -> str:
        metadata: list[str] = []
        if self.include_node_ids:
            metadata.append(node.node_id)
        if self.include_provenance:
            metadata.append(_source_label(node.source_refs[0]))
        return rendered if not metadata else f"<!-- {' | '.join(metadata)} -->\n{rendered}"
