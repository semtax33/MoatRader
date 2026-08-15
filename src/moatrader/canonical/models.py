from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator


SCHEMA_VERSION = "1.0.0"


class ContractModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        use_enum_values=False,
        str_strip_whitespace=False,
    )


class SourceType(StrEnum):
    DART = "DART"
    SEC_EDGAR = "SEC_EDGAR"
    IR = "IR"
    ANALYST = "ANALYST"
    INDUSTRY = "INDUSTRY"
    OTHER = "OTHER"


class DocumentType(StrEnum):
    ANNUAL_REPORT = "ANNUAL_REPORT"
    QUARTERLY_REPORT = "QUARTERLY_REPORT"
    CURRENT_REPORT = "CURRENT_REPORT"
    REGISTRATION_STATEMENT = "REGISTRATION_STATEMENT"
    IR_PRESENTATION = "IR_PRESENTATION"
    EARNINGS_RELEASE = "EARNINGS_RELEASE"
    ANALYST_REPORT = "ANALYST_REPORT"
    OTHER = "OTHER"


class SectionRole(StrEnum):
    COMPANY_OVERVIEW = "COMPANY_OVERVIEW"
    BUSINESS = "BUSINESS"
    PRODUCTS = "PRODUCTS"
    CUSTOMERS = "CUSTOMERS"
    SUPPLIERS = "SUPPLIERS"
    COMPETITION = "COMPETITION"
    RISK = "RISK"
    MDA = "MDA"
    FINANCIALS = "FINANCIALS"
    NOTES = "NOTES"
    GOVERNANCE = "GOVERNANCE"
    GUIDANCE = "GUIDANCE"
    OTHER = "OTHER"


class StatementType(StrEnum):
    DISCLOSED_FACT = "DISCLOSED_FACT"
    MANAGEMENT_CLAIM = "MANAGEMENT_CLAIM"
    ANALYST_INTERPRETATION = "ANALYST_INTERPRETATION"
    INDUSTRY_INTERPRETATION = "INDUSTRY_INTERPRETATION"
    FORECAST = "FORECAST"
    DERIVED_METRIC = "DERIVED_METRIC"


class PeriodKind(StrEnum):
    INSTANT = "INSTANT"
    DURATION = "DURATION"
    FOREVER = "FOREVER"
    UNKNOWN = "UNKNOWN"


class ConsolidationScope(StrEnum):
    CONSOLIDATED = "CONSOLIDATED"
    SEPARATE = "SEPARATE"
    SEGMENT = "SEGMENT"
    UNKNOWN = "UNKNOWN"


class AssetKind(StrEnum):
    IMAGE = "IMAGE"
    CHART = "CHART"
    DIAGRAM = "DIAGRAM"
    ATTACHMENT = "ATTACHMENT"
    OTHER = "OTHER"


class AvailabilityPrecision(StrEnum):
    EXACT = "EXACT"
    DAY = "DAY"
    INFERRED = "INFERRED"


class BoundingBox(ContractModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def ordered(self) -> "BoundingBox":
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bounding box coordinates must be ordered")
        return self


class SourceRef(ContractModel):
    source_type: SourceType
    document_id: str = Field(min_length=1)
    uri: str | None = None
    xpath: str | None = None
    page: int | None = Field(default=None, ge=1)
    slide: int | None = Field(default=None, ge=1)
    bbox: BoundingBox | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    table_row: int | None = Field(default=None, ge=0)
    table_col: int | None = Field(default=None, ge=0)
    source_hash: str | None = None

    @model_validator(mode="after")
    def char_range_is_valid(self) -> "SourceRef":
        if self.char_start is not None and self.char_end is not None and self.char_end < self.char_start:
            raise ValueError("char_end must not precede char_start")
        return self


class ReportingPeriod(ContractModel):
    kind: PeriodKind = PeriodKind.UNKNOWN
    start: date | None = None
    end: date | None = None
    instant: date | None = None
    fiscal_year: int | None = None
    fiscal_period: str | None = None
    raw_label: str | None = None

    @model_validator(mode="after")
    def period_shape(self) -> "ReportingPeriod":
        if self.kind == PeriodKind.INSTANT and self.instant is None:
            raise ValueError("instant period requires instant")
        if self.kind == PeriodKind.DURATION and (self.start is None or self.end is None):
            raise ValueError("duration period requires start and end")
        if self.start and self.end and self.end < self.start:
            raise ValueError("period end must not precede start")
        return self


class DocumentMetadata(ContractModel):
    schema_version: str = SCHEMA_VERSION
    source_type: SourceType
    source_document_id: str = Field(min_length=1)
    document_type: DocumentType = DocumentType.OTHER
    issuer_id: str | None = None
    issuer_name: str | None = None
    ticker: str | None = None
    market: str | None = None
    title: str | None = None
    published_at: datetime | None = None
    available_at: datetime
    availability_precision: AvailabilityPrecision = AvailabilityPrecision.EXACT
    availability_source: str = Field(min_length=1)
    reporting_period: ReportingPeriod | None = None
    language: str = "und"
    jurisdiction: str | None = None
    is_amendment: bool = False
    amends_document_id: str | None = None
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_version: str
    source_specific: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def point_in_time_and_amendment_rules(self) -> "DocumentMetadata":
        if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
            raise ValueError("available_at must be timezone-aware for point-in-time safety")
        if self.published_at is not None and (
            self.published_at.tzinfo is None or self.published_at.utcoffset() is None
        ):
            raise ValueError("published_at must be timezone-aware when supplied")
        if self.is_amendment and not self.amends_document_id:
            raise ValueError("an amendment must identify amends_document_id")
        return self


class ClassificationTrace(ContractModel):
    rule_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class BaseNode(ContractModel):
    node_id: str
    kind: str
    order: int = Field(ge=0)
    raw_text: str = ""
    normalized_text: str = ""
    section_path: list[str] = Field(default_factory=list)
    source_refs: list[SourceRef] = Field(min_length=1)
    classification: ClassificationTrace | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)


class ParagraphNode(BaseNode):
    kind: Literal["paragraph"] = "paragraph"


class NoteNode(BaseNode):
    kind: Literal["note"] = "note"
    marker: str | None = None
    target_node_ids: list[str] = Field(default_factory=list)


class UnknownBlockNode(BaseNode):
    kind: Literal["unknown_block"] = "unknown_block"
    reason: str = "unclassified_visible_block"


class PageBreakNode(BaseNode):
    kind: Literal["page_break"] = "page_break"
    page_after: int | None = Field(default=None, ge=1)


class FigureNode(BaseNode):
    kind: Literal["figure"] = "figure"
    asset_id: str | None = None
    caption: str | None = None
    alt_text: str | None = None


class ListItemNode(BaseNode):
    kind: Literal["list_item"] = "list_item"
    ordinal: int | None = None


class ListNode(BaseNode):
    kind: Literal["list"] = "list"
    ordered: bool = False
    items: list[ListItemNode] = Field(default_factory=list)


class UnitSpec(ContractModel):
    raw: str
    canonical: str | None = None
    scale: Decimal = Decimal("1")
    currency: str | None = None


class TableCell(ContractModel):
    row: int = Field(ge=0)
    col: int = Field(ge=0)
    origin_row: int = Field(ge=0)
    origin_col: int = Field(ge=0)
    raw_text: str = ""
    normalized_text: str = ""
    is_header: bool = False
    propagated: bool = False
    source_rowspan: int = Field(default=1, ge=1)
    source_colspan: int = Field(default=1, ge=1)
    numeric_value: Decimal | None = None
    unit: UnitSpec | None = None
    source_ref: SourceRef | None = None


class TableRow(ContractModel):
    index: int = Field(ge=0)
    cells: list[TableCell]

    @model_validator(mode="after")
    def positions_match(self) -> "TableRow":
        if any(cell.row != self.index for cell in self.cells):
            raise ValueError("every cell row must equal its TableRow index")
        if [cell.col for cell in self.cells] != list(range(len(self.cells))):
            raise ValueError("table cells must be rectangular and ordered by column")
        return self


class TableHeader(ContractModel):
    col: int = Field(ge=0)
    path: list[str] = Field(default_factory=list)


class TableFootnote(ContractModel):
    marker: str | None = None
    text: str
    node_id: str | None = None


class TableNode(BaseNode):
    kind: Literal["table"] = "table"
    caption: str | None = None
    unit: UnitSpec | None = None
    period: ReportingPeriod | None = None
    column_headers: list[TableHeader] = Field(default_factory=list)
    header_row_count: int = Field(default=0, ge=0)
    rows: list[TableRow] = Field(default_factory=list)
    footnotes: list[TableFootnote] = Field(default_factory=list)

    @model_validator(mode="after")
    def rectangular(self) -> "TableNode":
        widths = {len(row.cells) for row in self.rows}
        if len(widths) > 1:
            raise ValueError("all canonical table rows must have equal width")
        width = next(iter(widths), 0)
        if self.column_headers and (
            [header.col for header in self.column_headers] != list(range(width))
        ):
            raise ValueError("column_headers must cover every column in order")
        if self.header_row_count > len(self.rows):
            raise ValueError("header_row_count exceeds row count")
        return self


class SectionNode(BaseNode):
    kind: Literal["section"] = "section"
    title_raw: str
    title_normalized: str
    level: int = Field(ge=1)
    role: SectionRole | None = None
    heading_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    explicit_level: int | None = Field(default=None, ge=1)
    inferred_level: int | None = Field(default=None, ge=1)
    children: list["CanonicalNode"] = Field(default_factory=list)


CanonicalNode: TypeAlias = Annotated[
    SectionNode
    | ParagraphNode
    | TableNode
    | ListNode
    | NoteNode
    | FigureNode
    | PageBreakNode
    | UnknownBlockNode,
    Field(discriminator="kind"),
]


SectionNode.model_rebuild(_types_namespace={"CanonicalNode": CanonicalNode})


def walk_nodes(nodes: Iterable[CanonicalNode]) -> Iterator[CanonicalNode]:
    for node in nodes:
        yield node
        if isinstance(node, SectionNode):
            yield from walk_nodes(node.children)
        elif isinstance(node, ListNode):
            yield from node.items


class DocumentAST(ContractModel):
    document_id: str
    children: list[CanonicalNode] = Field(default_factory=list)

    def walk(self) -> Iterator[CanonicalNode]:
        return walk_nodes(self.children)

    def node_index(self) -> dict[str, CanonicalNode]:
        return {node.node_id: node for node in self.walk()}


class FactDimension(ContractModel):
    axis: str
    member: str
    typed_value: str | None = None


class StructuredFact(ContractModel):
    fact_id: str
    concept: str
    canonical_concept: str | None = None
    label: str | None = None
    value: Decimal | str | bool | None
    numeric_value: Decimal | None = None
    unit: UnitSpec | None = None
    period: ReportingPeriod
    scope: ConsolidationScope = ConsolidationScope.UNKNOWN
    dimensions: list[FactDimension] = Field(default_factory=list)
    segment: str | None = None
    context_id: str | None = None
    decimals: int | Literal["INF"] | None = None
    statement_type: StatementType = StatementType.DISCLOSED_FACT
    available_at: datetime
    is_restated: bool = False
    source_refs: list[SourceRef] = Field(min_length=1)
    derived_from_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def fact_is_pit_safe(self) -> "StructuredFact":
        if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
            raise ValueError("fact available_at must be timezone-aware")
        if self.statement_type == StatementType.DERIVED_METRIC and not self.derived_from_ids:
            raise ValueError("derived metrics must identify their inputs")
        return self


class DocumentAsset(ContractModel):
    asset_id: str
    kind: AssetKind
    media_type: str | None = None
    uri: str | None = None
    sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    caption: str | None = None
    alt_text: str | None = None
    extracted_text: str | None = None
    source_refs: list[SourceRef] = Field(min_length=1)


class ProvenanceRecord(ContractModel):
    object_id: str
    source_refs: list[SourceRef] = Field(default_factory=list)
    derived_from_ids: list[str] = Field(default_factory=list)
    transform: str | None = None
    transform_version: str | None = None


class ProvenanceIndex(ContractModel):
    records: dict[str, ProvenanceRecord] = Field(default_factory=dict)

    @model_validator(mode="after")
    def record_keys_match(self) -> "ProvenanceIndex":
        for key, record in self.records.items():
            if key != record.object_id:
                raise ValueError(f"provenance key {key!r} does not match object_id")
        return self


class QualityMetrics(ContractModel):
    raw_visible_chars: int = Field(default=0, ge=0)
    ast_chars: int = Field(default=0, ge=0)
    text_retention: float | None = Field(default=None, ge=0.0)
    raw_table_count: int = Field(default=0, ge=0)
    ast_table_count: int = Field(default=0, ge=0)
    raw_numeric_cell_count: int = Field(default=0, ge=0)
    numeric_cell_count: int = Field(default=0, ge=0)
    numeric_retention: float | None = Field(default=None, ge=0.0)
    raw_structured_fact_count: int = Field(default=0, ge=0)
    structured_fact_count: int = Field(default=0, ge=0)
    structured_fact_retention: float | None = Field(default=None, ge=0.0)
    paragraph_count: int = Field(default=0, ge=0)
    heading_count: int = Field(default=0, ge=0)
    unknown_block_count: int = Field(default=0, ge=0)
    duplicate_text_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    warnings: list[str] = Field(default_factory=list)


class CanonicalDocumentBundle(ContractModel):
    schema_version: str = SCHEMA_VERSION
    metadata: DocumentMetadata
    ast: DocumentAST
    facts: list[StructuredFact] = Field(default_factory=list)
    assets: list[DocumentAsset] = Field(default_factory=list)
    provenance: ProvenanceIndex = Field(default_factory=ProvenanceIndex)
    quality: QualityMetrics = Field(default_factory=QualityMetrics)

    @model_validator(mode="after")
    def identifiers_and_lineage_are_consistent(self) -> "CanonicalDocumentBundle":
        document_id = self.metadata.source_document_id
        if self.ast.document_id != document_id:
            raise ValueError("AST document_id must equal metadata source_document_id")

        ids = [node.node_id for node in self.ast.walk()]
        ids.extend(fact.fact_id for fact in self.facts)
        ids.extend(asset.asset_id for asset in self.assets)
        if len(ids) != len(set(ids)):
            raise ValueError("node, fact, and asset IDs must be unique within a bundle")

        known = set(ids)
        asset_ids = {asset.asset_id for asset in self.assets}
        for node in self.ast.walk():
            if isinstance(node, FigureNode) and node.asset_id and node.asset_id not in asset_ids:
                raise ValueError(f"figure {node.node_id} references missing asset {node.asset_id}")
        for fact in self.facts:
            missing = set(fact.derived_from_ids) - known
            if missing:
                raise ValueError(f"fact {fact.fact_id} has missing lineage: {sorted(missing)}")
        for key, record in self.provenance.records.items():
            if key not in known:
                raise ValueError(f"provenance record references unknown object {key}")
            missing = set(record.derived_from_ids) - known
            if missing:
                raise ValueError(f"provenance record {key} has missing lineage: {sorted(missing)}")
        return self

    def as_of(self, cutoff: datetime) -> "CanonicalDocumentBundle | None":
        """Return this immutable point-in-time view only when it was market-visible."""
        if cutoff.tzinfo is None or cutoff.utcoffset() is None:
            raise ValueError("cutoff must be timezone-aware")
        return self if self.metadata.available_at <= cutoff else None

    def to_json(self, *, indent: int | None = 2) -> str:
        return self.model_dump_json(indent=indent, exclude_none=True)
