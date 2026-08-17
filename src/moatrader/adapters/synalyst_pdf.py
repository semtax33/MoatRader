from __future__ import annotations

import importlib
import os
import sys
import tempfile
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from moatrader.adapters.base import RawDocument
from moatrader.adapters.pdf import (
    IrPdfAdapter,
    ParsedPdf,
    PdfImage,
    PdfPage,
    PdfTable,
    PdfTextBlock,
    _hint_source,
)
from moatrader.canonical.ids import content_hash
from moatrader.canonical.models import (
    AssetKind,
    AvailabilityPrecision,
    DocumentMetadata,
    DocumentType,
    SourceType,
    StatementType,
)


_KST = ZoneInfo("Asia/Seoul")
SYNALYST_BRIDGE_VERSION = "synalyst-industry-bridge/1.0.0"
SUPPORTED_SYNALYST_PARSER_VERSIONS = frozenset({"0.2.15"})


def _aware_datetime(value: Any, *, default_zone: ZoneInfo = _KST) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
    else:
        normalized = str(value or "").strip().replace("Z", "+00:00")
        if not normalized:
            raise ValueError("a PIT-safe timestamp is required")
        parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.replace(tzinfo=default_zone)
    return parsed


def _optional_datetime(value: Any) -> datetime | None:
    return _aware_datetime(value) if value not in (None, "") else None


def _default_synalyst_root() -> Path | None:
    configured = os.getenv("MOATRADER_SYNALYST_ROOT", "").strip()
    candidates = [
        Path(configured) if configured else None,
        Path(__file__).resolve().parents[4] / "Synalyst",
    ]
    for candidate in candidates:
        if candidate is not None and (candidate / "synalyst" / "pdf_pipeline").is_dir():
            return candidate.resolve()
    return None


class IndustryPdfAdapter(IrPdfAdapter):
    """Canonical industry-report adapter backed by Synalyst's reviewed PDF parser.

    Synalyst owns layout, table, OCR-plan, and provenance-aware page parsing.  This
    bridge only translates its source-neutral records into MoatRader's canonical
    document model; it deliberately does not maintain a second PDF parser.
    """

    source_type = SourceType.INDUSTRY
    document_type = DocumentType.ANALYST_REPORT
    statement_type = StatementType.INDUSTRY_INTERPRETATION
    source_label = "Industry analyst report"
    text_transform = "synalyst_silver_to_canonical_ast"
    asset_transform = "synalyst_figure_region_to_canonical_asset"
    asset_kind = AssetKind.CHART
    always_emit_figure_nodes = True
    table_text_suppression_threshold = 0.10

    def __init__(
        self,
        *,
        synalyst_root: str | Path | None = None,
        synalyst_parser: Any | None = None,
    ) -> None:
        # OCR is configured inside Synalyst, not in the inherited PyMuPDF lane.
        super().__init__(ocr_adapter=None)
        self.synalyst_root = (
            Path(synalyst_root).expanduser().resolve()
            if synalyst_root is not None
            else _default_synalyst_root()
        )
        self._injected_parser = synalyst_parser
        self._document_input_type: Any | None = None
        self._synalyst_parser_version = "unknown"
        self.parser_version = f"{SYNALYST_BRIDGE_VERSION}+synalyst/unknown"

    def detect(self, source: RawDocument) -> bool:
        hinted = _hint_source(source)
        is_pdf = (
            (source.media_type or "").casefold() == "application/pdf"
            or source.content.startswith(b"%PDF-")
        )
        return is_pdf and hinted in {
            "INDUSTRY",
            "INDUSTRY_REPORT",
            "ANALYST_INDUSTRY",
        }

    def _parser(self) -> tuple[Any, Any]:
        if self._injected_parser is not None:
            if self._document_input_type is None:
                models = importlib.import_module("synalyst.pdf_pipeline.models")
                parser_module = importlib.import_module("synalyst.pdf_pipeline.parser")
                self._document_input_type = models.DocumentInput
                self._synalyst_parser_version = str(parser_module.PARSER_VERSION)
                self._validate_parser_version()
                self.parser_version = (
                    f"{SYNALYST_BRIDGE_VERSION}+synalyst/{self._synalyst_parser_version}"
                )
            return self._injected_parser, self._document_input_type

        if self.synalyst_root is not None:
            package = self.synalyst_root / "synalyst" / "pdf_pipeline"
            if not package.is_dir():
                raise FileNotFoundError(
                    f"Synalyst PDF package not found below {self.synalyst_root}"
                )
            root_text = str(self.synalyst_root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
        try:
            public = importlib.import_module("synalyst.pdf_pipeline")
            parser_module = importlib.import_module("synalyst.pdf_pipeline.parser")
        except ImportError as exc:
            raise RuntimeError(
                "Synalyst is required for industry PDF parsing. Set "
                "MOATRADER_SYNALYST_ROOT to the Synalyst checkout."
            ) from exc
        self._document_input_type = public.DocumentInput
        self._synalyst_parser_version = str(parser_module.PARSER_VERSION)
        if self.synalyst_root is not None:
            module_path = Path(str(public.__file__)).resolve()
            try:
                module_path.relative_to(self.synalyst_root)
            except ValueError as exc:
                raise RuntimeError(
                    "loaded synalyst package does not come from the configured "
                    f"checkout: {module_path}"
                ) from exc
        self._validate_parser_version()
        self.parser_version = (
            f"{SYNALYST_BRIDGE_VERSION}+synalyst/{self._synalyst_parser_version}"
        )
        self._injected_parser = public.StructuredPdfParser()
        return self._injected_parser, self._document_input_type

    def _validate_parser_version(self) -> None:
        if self._synalyst_parser_version not in SUPPORTED_SYNALYST_PARSER_VERSIONS:
            supported = ", ".join(sorted(SUPPORTED_SYNALYST_PARSER_VERSIONS))
            raise RuntimeError(
                "unreviewed Synalyst parser version "
                f"{self._synalyst_parser_version}; supported: {supported}"
            )

    def extract_metadata(self, source: RawDocument) -> DocumentMetadata:
        self._parser()
        hints = dict(source.hints)
        available_at = _aware_datetime(
            hints.get("available_at")
            or hints.get("registered_at")
            or source.fetched_at
        )
        published_at = _optional_datetime(
            hints.get("published_at") or hints.get("report_date")
        )
        if published_at is not None and available_at < published_at:
            raise ValueError("available_at cannot be earlier than published_at")
        raw_hash = content_hash(source.content)
        report_id = str(
            hints.get("source_document_id")
            or hints.get("report_id")
            or raw_hash[:24]
        )
        source_specific = dict(hints.get("source_specific") or {})
        industry_code = str(
            hints.get("industry_code")
            or source_specific.get("industry_code")
            or "UNKNOWN"
        ).strip()
        industry_name = str(
            hints.get("industry_name")
            or source_specific.get("industry_name")
            or hints.get("issuer_name")
            or "Unknown industry"
        ).strip()
        source_specific.update(
            {
                "statement_type": self.statement_type.value,
                "economic_scope": "INDUSTRY",
                "industry_code": industry_code,
                "industry_name": industry_name,
                "publisher": (
                    hints.get("publisher")
                    or hints.get("office_name")
                    or source_specific.get("publisher")
                ),
                "author": (
                    hints.get("author")
                    or hints.get("report_writer")
                    or source_specific.get("author")
                ),
                "source_system": (
                    hints.get("source_system")
                    or source_specific.get("source_system")
                    or "hankyung_consensus"
                ),
                "synalyst_parser_version": self._synalyst_parser_version,
                "bridge_version": SYNALYST_BRIDGE_VERSION,
            }
        )
        return DocumentMetadata(
            source_type=self.source_type,
            source_document_id=report_id,
            document_type=self.document_type,
            # Industry reports are reference-class evidence, not issuer filings.
            issuer_id=hints.get("issuer_id") or f"INDUSTRY:{industry_code}",
            issuer_name=hints.get("issuer_name") or industry_name,
            ticker=hints.get("ticker") or f"INDUSTRY-{industry_code}",
            market=hints.get("market"),
            title=hints.get("title") or hints.get("report_name"),
            published_at=published_at,
            available_at=available_at,
            availability_precision=AvailabilityPrecision(
                str(hints.get("availability_precision") or "EXACT").upper()
            ),
            availability_source=str(
                hints.get("availability_source") or "hankyung_register_date"
            ),
            language=str(hints.get("language") or "ko"),
            jurisdiction=str(hints.get("jurisdiction") or "KR"),
            raw_sha256=raw_hash,
            parser_version=self.parser_version,
            source_specific={key: value for key, value in source_specific.items() if value is not None},
        )

    def parse_structure(self, source: RawDocument) -> ParsedPdf:
        parser, document_input_type = self._parser()
        metadata = self.extract_metadata(source)
        local_path = Path(str(source.hints.get("local_path") or "")).expanduser()
        temporary_path: Path | None = None
        if local_path.is_file():
            resolved = local_path.resolve()
            if content_hash(resolved.read_bytes()) != metadata.raw_sha256:
                raise ValueError("local_path bytes do not match RawDocument content")
            pdf_path = resolved
        else:
            handle = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            try:
                handle.write(source.content)
                handle.flush()
            finally:
                handle.close()
            temporary_path = Path(handle.name)
            pdf_path = temporary_path
        try:
            document_input = document_input_type(
                path=pdf_path,
                source="hankyung-industry",
                published_at=metadata.published_at or metadata.available_at,
                available_as_of=metadata.available_at,
                event_at=None,
                title=metadata.title,
                external_id=metadata.source_document_id,
            )
            structured = parser.parse(document_input)
            return self._translate(structured)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    @staticmethod
    def _translate(structured: Any) -> ParsedPdf:
        pages: list[PdfPage] = []
        candidate_table_count = 0
        accepted_table_count = 0
        review_item_count = 0
        for parsed_page in structured.pages:
            candidate_table_count += len(parsed_page.tables)
            review_item_count += len(parsed_page.review_items)
            page_record = parsed_page.page
            provenance = {
                item.provenance_id: item for item in parsed_page.provenance
            }
            blocks: list[PdfTextBlock] = []
            images: list[PdfImage] = []
            for index, node in enumerate(parsed_page.nodes):
                node_type = str(getattr(node.node_type, "value", node.node_type))
                if node_type == "figure":
                    images.append(
                        PdfImage(
                            page=node.page_number,
                            index=index,
                            bbox=(node.bbox.x0, node.bbox.y0, node.bbox.x1, node.bbox.y1),
                        )
                    )
                    if not str(node.text or "").strip():
                        continue
                if node_type in {"header", "footer", "disclaimer"}:
                    continue
                source_provenance = provenance.get(node.provenance_id)
                heading = node_type in {"title", "heading"}
                blocks.append(
                    PdfTextBlock(
                        page=node.page_number,
                        native_index=index,
                        text=node.text,
                        bbox=(node.bbox.x0, node.bbox.y0, node.bbox.x1, node.bbox.y1),
                        max_font_size=18.0 if node_type == "title" else 14.0 if heading else 10.0,
                        mean_font_size=18.0 if node_type == "title" else 14.0 if heading else 10.0,
                        bold_ratio=1.0 if heading else 0.0,
                        line_count=max(1, node.text.count("\n") + 1),
                        engine=(
                            f"synalyst:{source_provenance.extraction_engine}"
                            if source_provenance is not None
                            else "synalyst"
                        ),
                        confidence=node.confidence,
                    )
                )

            cells_by_table: dict[str, list[Any]] = {}
            for cell in parsed_page.cells:
                cells_by_table.setdefault(cell.table_id, []).append(cell)
            tables: list[PdfTable] = []
            bridge_warnings: list[str] = []
            for table in parsed_page.tables:
                page_area = float(page_record.width) * float(page_record.height)
                table_area = float(table.bbox.width) * float(table.bbox.height)
                page_ratio = table_area / page_area if page_area else 0.0
                # Synalyst deliberately marks ambiguous layout grids for review.
                # Treating a full-page newsletter grid as a semantic table then
                # suppresses every underlying narrative node. Fail closed to the
                # reviewed reading-order text while preserving the table warning.
                if (
                    table.needs_review
                    and not table.header_resolved
                    and page_ratio >= 0.60
                ):
                    bridge_warnings.append(
                        "ambiguous full-page layout table retained as narrative text: "
                        f"table_id={table.table_id}, area_ratio={page_ratio:.3f}"
                    )
                    continue
                matrix = [["" for _ in range(table.column_count)] for _ in range(table.row_count)]
                boxes: list[list[tuple[float, float, float, float] | None]] = [
                    [None for _ in range(table.column_count)] for _ in range(table.row_count)
                ]
                for cell in cells_by_table.get(table.table_id, []):
                    if cell.row_index >= table.row_count or cell.column_index >= table.column_count:
                        continue
                    matrix[cell.row_index][cell.column_index] = cell.text
                    if cell.bbox is not None:
                        boxes[cell.row_index][cell.column_index] = (
                            cell.bbox.x0,
                            cell.bbox.y0,
                            cell.bbox.x1,
                            cell.bbox.y1,
                        )
                tables.append(
                    PdfTable(
                        page=table.page_number,
                        strategy=f"synalyst:{table.strategy}",
                        bbox=(table.bbox.x0, table.bbox.y0, table.bbox.x1, table.bbox.y1),
                        matrix=tuple(tuple(row) for row in matrix),
                        cell_bboxes=tuple(tuple(row) for row in boxes),
                    )
                )
                accepted_table_count += 1
            warnings = [
                f"{getattr(item.reason, 'value', item.reason)}: {item.details}"
                for item in parsed_page.review_items
            ]
            warnings.extend(bridge_warnings)
            warnings.extend(
                f"quality:{item.metric_name}={item.metric_value:.4f} below {item.threshold:.4f}"
                for item in parsed_page.quality_metrics
                if not item.passed
            )
            ocr_applied = any(
                "ocr" in str(item.extraction_engine).casefold()
                for item in parsed_page.provenance
            )
            pages.append(
                PdfPage(
                    number=page_record.page_number,
                    width=page_record.width,
                    height=page_record.height,
                    blocks=blocks,
                    tables=tables,
                    images=images,
                    warnings=warnings,
                    needs_ocr=page_record.needs_ocr,
                    ocr_applied=ocr_applied,
                    ocr_engine="synalyst" if ocr_applied else None,
                )
            )
        document = structured.document
        pdf_metadata = {
            str(key): str(value) for key, value in (document.pdf_metadata or {}).items()
        }
        pdf_metadata.update(
            {
                "synalyst_document_id": str(document.document_id),
                "synalyst_source_path": str(document.source_path),
                "synalyst_candidate_table_count": str(candidate_table_count),
                "synalyst_accepted_table_count": str(accepted_table_count),
                "synalyst_review_item_count": str(review_item_count),
            }
        )
        return ParsedPdf(
            raw_sha256=str(document.checksum_sha256),
            pages=pages,
            metadata=pdf_metadata,
        )
