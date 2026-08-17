from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, time
from decimal import Decimal, InvalidOperation
from statistics import median
from typing import Any
from zoneinfo import ZoneInfo

from moatrader.adapters.base import RawDocument, SourceAdapter
from moatrader.adapters.html import _section_role
from moatrader.adapters.ocr import PdfOcrAdapter, PdfOcrBlock, PdfOcrResult
from moatrader.canonical.ids import content_hash, normalize_text, stable_id
from moatrader.canonical.models import (
    AssetKind,
    AvailabilityPrecision,
    BoundingBox,
    CanonicalDocumentBundle,
    DocumentAST,
    DocumentAsset,
    DocumentMetadata,
    DocumentType,
    FigureNode,
    PageBreakNode,
    ParagraphNode,
    PeriodKind,
    ProvenanceIndex,
    ProvenanceRecord,
    QualityMetrics,
    ReportingPeriod,
    SectionNode,
    SourceRef,
    SourceType,
    StatementType,
    TableCell,
    TableHeader,
    TableNode,
    TableRow,
    UnitSpec,
)


PARSER_VERSION = "pymupdf-ir/0.2.0"
_KST = ZoneInfo("Asia/Seoul")
_NUMBER = re.compile(r"^\(?[-+]?\d[\d,]*(?:\.\d+)?\)?%?$")
_YEAR = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_UNIT_CONTEXT = re.compile(
    r"(?:단위|unit)\s*[:：]?\s*[\(\[]?\s*"
    r"(억\s*원|백\s*만\s*원|천\s*원|원|미화\s*달러|달러|USD|KRW|%)(?:\s*[,/]\s*%)?",
    re.IGNORECASE,
)
_PAREN_UNIT_CONTEXT = re.compile(
    r"[\(\[]\s*(억\s*원|백\s*만\s*원|천\s*원|미화\s*달러|달러|USD|KRW)\s*[\)\]]",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PdfTextBlock:
    page: int
    native_index: int
    text: str
    bbox: tuple[float, float, float, float]
    max_font_size: float
    mean_font_size: float
    bold_ratio: float
    line_count: int
    engine: str = "pymupdf"
    confidence: float | None = None


@dataclass(frozen=True, slots=True)
class PdfTable:
    page: int
    strategy: str
    bbox: tuple[float, float, float, float]
    matrix: tuple[tuple[str, ...], ...]
    cell_bboxes: tuple[tuple[tuple[float, float, float, float] | None, ...], ...]


@dataclass(frozen=True, slots=True)
class PdfImage:
    page: int
    index: int
    bbox: tuple[float, float, float, float]


@dataclass(slots=True)
class PdfPage:
    number: int
    width: float
    height: float
    blocks: list[PdfTextBlock] = field(default_factory=list)
    tables: list[PdfTable] = field(default_factory=list)
    images: list[PdfImage] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    needs_ocr: bool = False
    ocr_applied: bool = False
    ocr_engine: str | None = None
    ocr_dpi: int | None = None
    ocr_mean_confidence: float | None = None
    ocr_block_count: int = 0


@dataclass(slots=True)
class ParsedPdf:
    raw_sha256: str
    pages: list[PdfPage]
    metadata: dict[str, str]


def enrich_ir_table_semantics(
    bundle: CanonicalDocumentBundle,
) -> CanonicalDocumentBundle:
    """Attach page-grounded period/unit context omitted by PDF table finders."""

    if bundle.metadata.source_type != SourceType.IR:
        return bundle
    nodes = list(bundle.ast.walk())
    page_text: dict[int, list[str]] = {}
    for node in nodes:
        if isinstance(node, TableNode) or not node.raw_text:
            continue
        pages = {
            reference.page
            for reference in node.source_refs
            if reference.page is not None
        }
        for page in pages:
            page_text.setdefault(page, []).append(node.raw_text)
    repairs = 0
    for node in nodes:
        if not isinstance(node, TableNode):
            continue
        page = next(
            (
                reference.page
                for reference in node.source_refs
                if reference.page is not None
            ),
            None,
        )
        context = "\n".join(
            [node.raw_text, *page_text.get(page, []), *node.section_path]
        )
        updates: dict[str, object] = {}
        match = _UNIT_CONTEXT.search(context) or _PAREN_UNIT_CONTEXT.search(context)
        if node.unit is None and match:
            raw = match.group(0).strip()
            value = match.group(1).upper().replace(" ", "")
            mapping: dict[str, tuple[str | None, Decimal, str | None]] = {
                "억원": ("KRW", Decimal("100000000"), "KRW"),
                "백만원": ("KRW", Decimal("1000000"), "KRW"),
                "천원": ("KRW", Decimal("1000"), "KRW"),
                "원": ("KRW", Decimal("1"), "KRW"),
                "미화달러": ("USD", Decimal("1"), "USD"),
                "달러": ("USD", Decimal("1"), "USD"),
                "USD": ("USD", Decimal("1"), "USD"),
                "KRW": ("KRW", Decimal("1"), "KRW"),
                "%": ("PERCENT", Decimal("0.01"), None),
            }
            canonical, scale, currency = mapping.get(
                value,
                (None, Decimal("1"), None),
            )
            updates["unit"] = UnitSpec(
                raw=raw,
                canonical=canonical,
                scale=scale,
                currency=currency,
            )
        if node.period is None and (year_match := _YEAR.search(context)):
            updates["period"] = ReportingPeriod(
                kind=PeriodKind.UNKNOWN,
                fiscal_year=int(year_match.group(1)),
                raw_label=year_match.group(0),
            )
        if updates:
            for key, value in updates.items():
                setattr(node, key, value)
            node.attributes = {
                **node.attributes,
                "page_context_semantics_enrichment": "ir-pdf/1",
            }
            repairs += 1
    tables = [node for node in nodes if isinstance(node, TableNode)]
    for node in tables:
        if node.unit is not None and node.period is not None:
            continue
        page = next(
            (reference.page for reference in node.source_refs if reference.page is not None),
            None,
        )
        siblings = [
            other
            for other in tables
            if other is not node
            and other.section_path == node.section_path
            and any(reference.page == page for reference in other.source_refs)
        ]
        updates: dict[str, object] = {}
        if node.unit is None:
            donor = next((other.unit for other in siblings if other.unit is not None), None)
            if donor is not None:
                updates["unit"] = donor
        if node.period is None:
            donor_period = next(
                (other.period for other in siblings if other.period is not None),
                None,
            )
            if donor_period is not None:
                updates["period"] = donor_period
        if updates:
            for key, value in updates.items():
                setattr(node, key, value)
            node.attributes = {
                **node.attributes,
                "page_context_semantics_enrichment": "ir-pdf/1",
                "semantics_propagated_from_same_page_section": True,
            }
            repairs += 1
    if not repairs:
        return bundle
    return bundle.model_copy(
        update={
            "metadata": bundle.metadata.model_copy(
                update={
                    "source_specific": {
                        **bundle.metadata.source_specific,
                        "table_semantics_enrichment_count": repairs,
                    }
                }
            )
        }
    )


def _hint_source(source: RawDocument) -> str:
    return str(source.hints.get("source_type", "")).upper().replace(" ", "_")


def _parse_datetime(value: Any, default_zone: ZoneInfo) -> tuple[datetime, AvailabilityPrecision]:
    if isinstance(value, datetime):
        parsed = value
        precision = AvailabilityPrecision.EXACT
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.min)
        precision = AvailabilityPrecision.DAY
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        precision = AvailabilityPrecision.EXACT
    else:
        raise ValueError("available_at (or fetched_at) is required")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=default_zone)
        precision = AvailabilityPrecision.INFERRED
    return parsed, precision


def _optional_datetime(value: Any, default_zone: ZoneInfo) -> datetime | None:
    return _parse_datetime(value, default_zone)[0] if value is not None else None


def _optional_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _bbox(value: tuple[float, float, float, float]) -> BoundingBox:
    return BoundingBox(x0=value[0], y0=value[1], x1=value[2], y1=value[3])


def _area(value: tuple[float, float, float, float]) -> float:
    return max(0.0, value[2] - value[0]) * max(0.0, value[3] - value[1])


def _coverage_by(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    x0, y0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    x1, y1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    return intersection / _area(inner) if _area(inner) else 0.0


def _overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = _area(left) + _area(right) - intersection
    return intersection / union if union else 0.0


def _numeric_value(value: str) -> Decimal | None:
    token = re.sub(r"\s+", "", value)
    if not token or not _NUMBER.fullmatch(token):
        return None
    negative = token.startswith("(") and token.endswith(")")
    token = token.strip("()%").replace(",", "")
    try:
        number = Decimal(token)
    except InvalidOperation:
        return None
    return -number if negative else number


def _encoding_corrupt(text: str) -> bool:
    visible = [character for character in text if not character.isspace()]
    if not visible:
        return False
    if "\ufffd" in text or any(ord(character) < 32 and character not in "\t\r\n" for character in text):
        return True
    suspicious = sum(0x00A1 <= ord(character) <= 0x00FF for character in visible)
    hangul = sum("가" <= character <= "힣" for character in visible)
    return len(visible) >= 8 and suspicious >= 4 and hangul / len(visible) < 0.10 and suspicious / len(visible) >= 0.25


def _safe_text(value: Any) -> str:
    """Replace illegal lone UTF-16 surrogates emitted by damaged PDF maps."""
    return str(value).encode("utf-8", errors="replace").decode("utf-8")


def _deduplicate_blocks(blocks: list[PdfTextBlock]) -> tuple[list[PdfTextBlock], int]:
    kept: list[PdfTextBlock] = []
    duplicates = 0
    for block in blocks:
        normalized = normalize_text(block.text).casefold()
        duplicate = next(
            (
                prior
                for prior in kept
                if normalize_text(prior.text).casefold() == normalized
                and all(abs(a - b) <= 1.5 for a, b in zip(prior.bbox, block.bbox, strict=True))
            ),
            None,
        )
        if duplicate is None:
            kept.append(block)
        else:
            duplicates += 1
    return kept, duplicates


class IrPdfAdapter(SourceAdapter[ParsedPdf]):
    source_type = SourceType.IR
    default_zone = _KST

    def __init__(
        self,
        *,
        ocr_adapter: PdfOcrAdapter | None = None,
        ocr_dpi: int = 200,
        retry_ocr_dpi: int = 300,
        minimum_ocr_confidence: float = 0.80,
        ocr_native_text_threshold: int = 80,
        ocr_image_area_threshold: float = 0.35,
    ) -> None:
        if ocr_dpi < 72 or retry_ocr_dpi < ocr_dpi:
            raise ValueError("OCR DPI must be at least 72 and retry DPI must not be lower")
        if not 0.0 <= minimum_ocr_confidence <= 1.0:
            raise ValueError("minimum OCR confidence must be between zero and one")
        if ocr_native_text_threshold < 0:
            raise ValueError("OCR native text threshold must not be negative")
        if not 0.0 <= ocr_image_area_threshold <= 1.0:
            raise ValueError("OCR image area threshold must be between zero and one")
        self.ocr_adapter = ocr_adapter
        self.ocr_dpi = ocr_dpi
        self.retry_ocr_dpi = retry_ocr_dpi
        self.minimum_ocr_confidence = minimum_ocr_confidence
        self.ocr_native_text_threshold = ocr_native_text_threshold
        self.ocr_image_area_threshold = ocr_image_area_threshold

    def detect(self, source: RawDocument) -> bool:
        hinted = _hint_source(source)
        is_pdf = (source.media_type or "").casefold() == "application/pdf" or source.content.startswith(b"%PDF-")
        return is_pdf and hinted in {"IR", "INVESTOR_RELATIONS"}

    def extract_metadata(self, source: RawDocument) -> DocumentMetadata:
        hints = dict(source.hints)
        raw_hash = content_hash(source.content)
        document_id = str(hints.get("source_document_id") or raw_hash[:24])
        available_at, precision = _parse_datetime(
            hints.get("available_at") or source.fetched_at,
            self.default_zone,
        )
        if hints.get("availability_precision"):
            precision = AvailabilityPrecision(str(hints["availability_precision"]).upper())
        report_date = _optional_date(hints.get("report_date") or hints.get("period_end"))
        reporting_period = (
            ReportingPeriod(
                kind=PeriodKind.UNKNOWN,
                end=report_date,
                fiscal_year=report_date.year,
                raw_label=report_date.isoformat(),
            )
            if report_date
            else None
        )
        source_specific = dict(hints.get("source_specific", {}))
        source_specific.setdefault("statement_type", StatementType.MANAGEMENT_CLAIM.value)
        return DocumentMetadata(
            source_type=SourceType.IR,
            source_document_id=document_id,
            document_type=DocumentType.IR_PRESENTATION,
            issuer_id=hints.get("issuer_id") or hints.get("corp_code"),
            issuer_name=hints.get("issuer_name"),
            ticker=hints.get("ticker") or hints.get("stock_code"),
            market=hints.get("market"),
            title=hints.get("title") or hints.get("report_name"),
            published_at=_optional_datetime(hints.get("published_at"), self.default_zone),
            available_at=available_at,
            availability_precision=precision,
            availability_source=str(
                hints.get("availability_source")
                or ("source_metadata" if hints.get("available_at") else "fetched_at")
            ),
            reporting_period=reporting_period,
            language=str(hints.get("language") or "ko"),
            jurisdiction=str(hints.get("jurisdiction") or "KR"),
            raw_sha256=raw_hash,
            parser_version=PARSER_VERSION,
            source_specific=source_specific,
        )

    def parse_structure(self, source: RawDocument) -> ParsedPdf:
        try:
            import fitz
        except ImportError as exc:  # pragma: no cover - packaging failure
            raise RuntimeError("PyMuPDF is required to parse PDF documents") from exc
        try:
            document = fitz.open(stream=source.content, filetype="pdf")
        except Exception as exc:
            raise ValueError(f"unable to open PDF: {exc}") from exc
        try:
            if document.needs_pass:
                raise ValueError("password-protected PDF is unsupported")
            pages: list[PdfPage] = []
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                fitz.TOOLS.set_small_glyph_heights(False)
                snapshot = PdfPage(
                    number=page_index + 1,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                )
                self._extract_text(page, snapshot)
                self._extract_images(page, snapshot)
                self._extract_tables(page, snapshot)
                visible_chars = sum(
                    len(re.sub(r"\s+", "", block.text)) for block in snapshot.blocks
                )
                image_area = sum(_area(image.bbox) for image in snapshot.images)
                page_area = snapshot.width * snapshot.height
                image_ratio = image_area / page_area if page_area else 0.0
                encoding_corrupt = any(
                    _encoding_corrupt(block.text) for block in snapshot.blocks
                )
                snapshot.needs_ocr = encoding_corrupt or (
                    visible_chars < self.ocr_native_text_threshold
                    and image_ratio >= self.ocr_image_area_threshold
                )
                if snapshot.needs_ocr and not encoding_corrupt:
                    snapshot.warnings.append(
                        "OCR_REQUIRED: image-dominant page has insufficient native text"
                    )
                if encoding_corrupt:
                    snapshot.warnings.append(
                        "OCR_REQUIRED: native text encoding appears corrupt"
                    )
                if snapshot.needs_ocr and self.ocr_adapter is not None:
                    self._apply_ocr(
                        page,
                        snapshot,
                        replace_native=encoding_corrupt or visible_chars < 40,
                    )
                pages.append(snapshot)
            return ParsedPdf(
                raw_sha256=content_hash(source.content),
                pages=pages,
                metadata={
                    _safe_text(key): _safe_text(value)
                    for key, value in (document.metadata or {}).items()
                    if value not in (None, "")
                },
            )
        finally:
            document.close()

    @staticmethod
    def _extract_text(page: Any, snapshot: PdfPage) -> None:
        try:
            payload = page.get_text("dict", sort=False)
            blocks: list[PdfTextBlock] = []
            native_index = -1
            for raw in payload.get("blocks", []):
                if int(raw.get("type", -1)) != 0:
                    continue
                native_index += 1
                lines: list[str] = []
                sizes: list[float] = []
                bold_chars = 0
                total_chars = 0
                for line in raw.get("lines", []):
                    spans = line.get("spans", [])
                    line_text = "".join(_safe_text(span.get("text", "")) for span in spans).strip()
                    if line_text:
                        lines.append(line_text)
                    for span in spans:
                        text = _safe_text(span.get("text", ""))
                        count = len(text.strip())
                        if not count:
                            continue
                        sizes.append(float(span.get("size", 0.0)))
                        total_chars += count
                        if int(span.get("flags", 0)) & 16 or "bold" in str(span.get("font", "")).casefold():
                            bold_chars += count
                text = "\n".join(lines).strip()
                if not text:
                    continue
                bbox = tuple(map(float, raw["bbox"]))
                blocks.append(
                    PdfTextBlock(
                        page=snapshot.number,
                        native_index=native_index,
                        text=text,
                        bbox=bbox,  # type: ignore[arg-type]
                        max_font_size=max(sizes, default=0.0),
                        mean_font_size=sum(sizes) / len(sizes) if sizes else 0.0,
                        bold_ratio=bold_chars / total_chars if total_chars else 0.0,
                        line_count=len(lines),
                        engine="pymupdf",
                    )
                )
            snapshot.blocks, duplicate_count = _deduplicate_blocks(blocks)
            if duplicate_count:
                snapshot.warnings.append(
                    f"removed {duplicate_count} visually overprinted native text block(s)"
                )
        except Exception as exc:
            snapshot.warnings.append(f"text extraction failed: {exc}")

    @staticmethod
    def _extract_images(page: Any, snapshot: PdfPage) -> None:
        try:
            for index, image in enumerate(page.get_image_info(xrefs=True)):
                raw_bbox = image.get("bbox")
                if raw_bbox is None:
                    continue
                bbox = tuple(map(float, raw_bbox))
                if _area(bbox) > 0:  # type: ignore[arg-type]
                    snapshot.images.append(
                        PdfImage(
                            page=snapshot.number,
                            index=index,
                            bbox=bbox,  # type: ignore[arg-type]
                        )
                    )
        except Exception as exc:
            snapshot.warnings.append(f"image extraction failed: {exc}")

    @staticmethod
    def _extract_tables(page: Any, snapshot: PdfPage) -> None:
        candidates: list[PdfTable] = []
        for strategy in ("lines", "text"):
            if strategy == "text" and any(table.strategy == "lines" for table in candidates):
                break
            try:
                kwargs: dict[str, Any] = {"strategy": strategy}
                if strategy == "text":
                    kwargs.update(min_words_vertical=2, min_words_horizontal=1)
                finder = page.find_tables(**kwargs)
                for table in finder.tables:
                    raw_matrix = [
                        ["" if value is None else normalize_text(_safe_text(value)) for value in row]
                        for row in table.extract()
                    ]
                    width = max((len(row) for row in raw_matrix), default=0)
                    if len(raw_matrix) < 2 or not 2 <= width <= 24:
                        continue
                    matrix = tuple(tuple([*row, *([""] * (width - len(row)))]) for row in raw_matrix)
                    bbox = tuple(map(float, table.bbox))
                    nonempty = [value for row in matrix for value in row if value]
                    numeric_ratio = (
                        sum(_numeric_value(value) is not None for value in nonempty) / len(nonempty)
                        if nonempty
                        else 0.0
                    )
                    nonempty_ratio = len(nonempty) / (len(matrix) * width)
                    page_area = snapshot.width * snapshot.height
                    if strategy == "text" and page_area and _area(bbox) / page_area > 0.55 and numeric_ratio < 0.35:  # type: ignore[arg-type]
                        continue
                    if strategy == "text" and numeric_ratio < 0.20 and nonempty_ratio < 0.55:
                        # Sparse text clusters around diagrams and agenda braces are not
                        # tables. Their native blocks remain available as paragraphs.
                        continue
                    cell_rows: list[tuple[tuple[float, float, float, float] | None, ...]] = []
                    for row_index in range(len(matrix)):
                        raw_cells = table.rows[row_index].cells if row_index < len(table.rows) else []
                        converted = [
                            tuple(map(float, raw_cells[column])) if column < len(raw_cells) and raw_cells[column] is not None else None
                            for column in range(width)
                        ]
                        cell_rows.append(tuple(converted))
                    candidates.append(
                        PdfTable(
                            page=snapshot.number,
                            strategy=strategy,
                            bbox=bbox,  # type: ignore[arg-type]
                            matrix=matrix,
                            cell_bboxes=tuple(cell_rows),
                        )
                    )
            except Exception as exc:
                snapshot.warnings.append(f"table strategy {strategy!r} failed: {exc}")

        selected: list[PdfTable] = []
        for candidate in sorted(
            candidates,
            key=lambda item: (0 if item.strategy == "lines" else 1, item.bbox[1], item.bbox[0]),
        ):
            if any(_overlap(candidate.bbox, prior.bbox) >= 0.60 for prior in selected):
                continue
            selected.append(candidate)
        snapshot.tables = sorted(selected, key=lambda item: (item.bbox[1], item.bbox[0]))

    def _apply_ocr(
        self,
        page: Any,
        snapshot: PdfPage,
        *,
        replace_native: bool,
    ) -> None:
        assert self.ocr_adapter is not None
        try:
            result = self.ocr_adapter.extract_page(page, dpi=self.ocr_dpi)
            if result.mean_confidence < self.minimum_ocr_confidence:
                retry = self.ocr_adapter.extract_page(page, dpi=self.retry_ocr_dpi)
                if (
                    retry.mean_confidence > result.mean_confidence
                    or len(retry.blocks) > len(result.blocks)
                ):
                    result = retry
            ocr_blocks = self._ocr_text_blocks(snapshot.number, result)
            if not ocr_blocks:
                snapshot.warnings.append("OCR_FAILED: engine returned no recognized text")
                return
            if replace_native:
                snapshot.blocks = ocr_blocks
            else:
                snapshot.blocks = self._supplement_native_text(snapshot.blocks, ocr_blocks)
            snapshot.blocks.sort(key=lambda item: (item.bbox[1], item.bbox[0], item.native_index))
            snapshot.ocr_applied = True
            snapshot.ocr_engine = result.engine
            snapshot.ocr_dpi = result.dpi
            snapshot.ocr_mean_confidence = result.mean_confidence
            snapshot.ocr_block_count = len(ocr_blocks)
            snapshot.warnings.append(
                "OCR_APPLIED: "
                f"engine={result.engine}, dpi={result.dpi}, "
                f"blocks={len(ocr_blocks)}, confidence={result.mean_confidence:.3f}"
            )
        except Exception as exc:
            snapshot.warnings.append(f"OCR_FAILED: {type(exc).__name__}: {exc}")

    def _ocr_text_blocks(
        self,
        page_number: int,
        result: PdfOcrResult,
    ) -> list[PdfTextBlock]:
        """Keep reliable OCR while retaining moderately confident numeric tokens.

        Presentation charts often produce dozens of low-confidence glyph
        fragments. Those fragments are more harmful than omitted prose to the
        evidence lane, while compact numeric labels legitimately score a bit
        lower. The asymmetric threshold preserves the latter without admitting
        arbitrary chart noise.
        """

        def accepted(block: PdfOcrBlock) -> bool:
            text = normalize_text(block.text)
            if not text:
                return False
            numeric = bool(re.search(r"\d", text))
            threshold = (
                max(0.55, self.minimum_ocr_confidence - 0.20)
                if numeric
                else self.minimum_ocr_confidence
            )
            if block.confidence < threshold:
                return False
            meaningful = sum(character.isalnum() for character in text)
            if meaningful == 0:
                return False
            return len(text) > 1 or numeric

        return [
            PdfTextBlock(
                page=page_number,
                native_index=-(index + 1),
                text=block.text,
                bbox=block.bbox,
                max_font_size=max(6.0, block.bbox[3] - block.bbox[1]),
                mean_font_size=max(6.0, block.bbox[3] - block.bbox[1]),
                bold_ratio=0.0,
                line_count=1,
                engine=result.engine,
                confidence=block.confidence,
            )
            for index, block in enumerate(result.blocks)
            if accepted(block)
        ]

    @staticmethod
    def _supplement_native_text(
        native: list[PdfTextBlock],
        ocr: list[PdfTextBlock],
    ) -> list[PdfTextBlock]:
        result = list(native)
        for candidate in ocr:
            candidate_text = normalize_text(candidate.text).casefold()
            duplicate = any(
                (
                    candidate_text == normalize_text(item.text).casefold()
                    or candidate_text in normalize_text(item.text).casefold()
                    or normalize_text(item.text).casefold() in candidate_text
                )
                and (
                    _coverage_by(candidate.bbox, item.bbox) >= 0.45
                    or _coverage_by(item.bbox, candidate.bbox) >= 0.45
                )
                for item in native
                if normalize_text(item.text)
            )
            if not duplicate:
                result.append(candidate)
        return result

    def convert(self, source: RawDocument) -> CanonicalDocumentBundle:
        metadata = self.extract_metadata(source)
        parsed = self.parse_structure(source)
        metadata = metadata.model_copy(
            update={
                "source_specific": {
                    **metadata.source_specific,
                    "pdf_page_count": len(parsed.pages),
                    "pdf_metadata": parsed.metadata,
                    "ocr": {
                        "configured": self.ocr_adapter is not None,
                        "engine": self.ocr_adapter.name if self.ocr_adapter else None,
                        "required_page_count": sum(page.needs_ocr for page in parsed.pages),
                        "applied_page_count": sum(page.ocr_applied for page in parsed.pages),
                        "failed_page_count": sum(
                            page.needs_ocr and not page.ocr_applied for page in parsed.pages
                        ),
                    },
                }
            }
        )
        ast, assets = self._build_ast(parsed, metadata, source.uri)
        nodes = list(ast.walk())
        tables = [node for node in nodes if isinstance(node, TableNode)]
        raw_visible_chars = sum(
            len(re.sub(r"\s+", "", block.text))
            for page in parsed.pages
            for block in page.blocks
        )
        ast_chars = sum(
            len(re.sub(r"\s+", "", node.raw_text))
            for node in nodes
            if node.raw_text
        )
        raw_numeric = sum(
            _numeric_value(value) is not None
            for page in parsed.pages
            for table in page.tables
            for row in table.matrix
            for value in row
        )
        canonical_numeric = sum(
            cell.numeric_value is not None
            for table in tables
            for row in table.rows
            for cell in row.cells
        )
        warnings = [
            f"page {page.number}: {warning}"
            for page in parsed.pages
            for warning in page.warnings
        ]
        if any(page.needs_ocr and not page.ocr_applied for page in parsed.pages):
            warnings.append(
                "OCR was required on one or more pages but no reviewed OCR adapter was configured; native evidence was preserved"
            )
        text_retention = min(1.0, ast_chars / raw_visible_chars) if raw_visible_chars else None
        numeric_retention = min(1.0, canonical_numeric / raw_numeric) if raw_numeric else None
        if text_retention is not None and text_retention < 0.90:
            warnings.append("visible text retention is below the 90% IR PDF target")
        if numeric_retention is not None and numeric_retention < 0.99:
            warnings.append("numeric cell retention is below the 99% IR PDF target")

        records: dict[str, ProvenanceRecord] = {}
        for node in nodes:
            records[node.node_id] = ProvenanceRecord(
                object_id=node.node_id,
                source_refs=node.source_refs,
                transform="pymupdf_page_layout_to_canonical_ast",
                transform_version=PARSER_VERSION,
            )
        for asset in assets:
            records[asset.asset_id] = ProvenanceRecord(
                object_id=asset.asset_id,
                source_refs=asset.source_refs,
                transform="pymupdf_image_region_extract",
                transform_version=PARSER_VERSION,
            )
        values = [
            normalize_text(node.normalized_text).casefold()
            for node in nodes
            if not isinstance(node, SectionNode) and normalize_text(node.normalized_text)
        ]
        duplicate_ratio = (
            (len(values) - len(set(values))) / len(values) if values else None
        )
        return enrich_ir_table_semantics(CanonicalDocumentBundle(
            metadata=metadata,
            ast=ast,
            facts=[],
            assets=assets,
            provenance=ProvenanceIndex(records=records),
            quality=QualityMetrics(
                raw_visible_chars=raw_visible_chars,
                ast_chars=ast_chars,
                text_retention=text_retention,
                raw_table_count=sum(len(page.tables) for page in parsed.pages),
                ast_table_count=len(tables),
                raw_numeric_cell_count=raw_numeric,
                numeric_cell_count=canonical_numeric,
                numeric_retention=numeric_retention,
                raw_structured_fact_count=0,
                structured_fact_count=0,
                structured_fact_retention=None,
                paragraph_count=sum(isinstance(node, ParagraphNode) for node in nodes),
                heading_count=sum(isinstance(node, SectionNode) for node in nodes),
                duplicate_text_ratio=duplicate_ratio,
                warnings=warnings,
            ),
        ))

    def _build_ast(
        self,
        parsed: ParsedPdf,
        metadata: DocumentMetadata,
        uri: str | None,
    ) -> tuple[DocumentAST, list[DocumentAsset]]:
        root: list[Any] = []
        stack: list[SectionNode] = []
        assets: list[DocumentAsset] = []
        order = 0

        def append(node: Any) -> None:
            if stack:
                stack[-1].children.append(node)
            else:
                root.append(node)

        for page in parsed.pages:
            font_sizes = [block.mean_font_size for block in page.blocks if block.mean_font_size > 0]
            body_size = median(font_sizes) if font_sizes else 10.0
            table_boxes = [table.bbox for table in page.tables]
            entries: list[tuple[float, float, str, Any]] = []
            for block in page.blocks:
                if any(_coverage_by(block.bbox, table_bbox) >= 0.60 for table_bbox in table_boxes):
                    continue
                entries.append((block.bbox[1], block.bbox[0], "text", block))
            for table in page.tables:
                entries.append((table.bbox[1], table.bbox[0], "table", table))

            for _y, _x, kind, item in sorted(entries, key=lambda value: (value[0], value[1], value[2])):
                if kind == "table":
                    node = self._table_node(item, metadata, uri, order)
                    append(node)
                    order += 1
                    continue
                block: PdfTextBlock = item
                normalized = normalize_text(block.text)
                if not normalized:
                    continue
                is_heading = (
                    len(normalized) <= 140
                    and block.line_count <= 3
                    and (
                        block.max_font_size >= body_size * 1.28
                        or (block.bold_ratio >= 0.75 and block.max_font_size >= body_size * 1.05)
                    )
                )
                ref = SourceRef(
                    source_type=SourceType.IR,
                    document_id=metadata.source_document_id,
                    uri=uri,
                    page=page.number,
                    bbox=_bbox(block.bbox),
                    source_hash=parsed.raw_sha256,
                )
                attributes = {
                    "statement_type": StatementType.MANAGEMENT_CLAIM.value,
                    "native_block_index": block.native_index,
                    "font_size": block.max_font_size,
                    "bold_ratio": block.bold_ratio,
                    "text_engine": block.engine,
                    "ocr_confidence": block.confidence,
                }
                if is_heading:
                    level = 1 if block.max_font_size >= body_size * 1.60 else 2
                    section = SectionNode(
                        node_id=stable_id(
                            "PS",
                            metadata.source_document_id,
                            page.number,
                            block.native_index,
                            normalized,
                        ),
                        kind="section",
                        order=order,
                        raw_text=block.text,
                        normalized_text=normalized,
                        source_refs=[ref],
                        attributes=attributes,
                        title_raw=block.text,
                        title_normalized=normalized,
                        level=level,
                        role=_section_role(normalized),
                        heading_confidence=min(
                            1.0,
                            max(block.max_font_size / max(body_size * 1.6, 0.1), block.bold_ratio),
                        ),
                        inferred_level=level,
                    )
                    while stack and stack[-1].level >= level:
                        stack.pop()
                    if stack:
                        stack[-1].children.append(section)
                    else:
                        root.append(section)
                    stack.append(section)
                    order += 1
                else:
                    append(
                        ParagraphNode(
                            node_id=stable_id(
                                "PP",
                                metadata.source_document_id,
                                page.number,
                                block.native_index,
                                normalized,
                            ),
                            order=order,
                            raw_text=block.text,
                            normalized_text=normalized,
                            source_refs=[ref],
                            attributes=attributes,
                        )
                    )
                    order += 1

            for image in page.images:
                ref = SourceRef(
                    source_type=SourceType.IR,
                    document_id=metadata.source_document_id,
                    uri=uri,
                    page=page.number,
                    bbox=_bbox(image.bbox),
                    source_hash=parsed.raw_sha256,
                )
                asset_id = stable_id(
                    "PA",
                    metadata.source_document_id,
                    page.number,
                    image.index,
                    image.bbox,
                )
                assets.append(
                    DocumentAsset(
                        asset_id=asset_id,
                        kind=AssetKind.IMAGE,
                        media_type="image/unknown",
                        uri=uri,
                        alt_text=f"PDF image region on page {page.number}",
                        source_refs=[ref],
                    )
                )
                if page.needs_ocr:
                    append(
                        FigureNode(
                            node_id=stable_id("PF", asset_id),
                            order=order,
                            raw_text="",
                            normalized_text="",
                            source_refs=[ref],
                            attributes={
                                "statement_type": StatementType.MANAGEMENT_CLAIM.value,
                                "ocr_status": (
                                    "APPLIED" if page.ocr_applied else "REQUIRED_NOT_CONFIGURED"
                                ),
                                "ocr_engine": page.ocr_engine,
                                "ocr_dpi": page.ocr_dpi,
                                "ocr_mean_confidence": page.ocr_mean_confidence,
                            },
                            asset_id=asset_id,
                            alt_text=(
                                f"Image-dominant IR page {page.number}; OCR applied"
                                if page.ocr_applied
                                else f"Image-dominant IR page {page.number}; OCR required"
                            ),
                        )
                    )
                    order += 1
            page_ref = SourceRef(
                source_type=SourceType.IR,
                document_id=metadata.source_document_id,
                uri=uri,
                page=page.number,
                source_hash=parsed.raw_sha256,
            )
            append(
                PageBreakNode(
                    node_id=stable_id("PB", metadata.source_document_id, page.number),
                    order=order,
                    source_refs=[page_ref],
                    page_after=page.number,
                    attributes={"statement_type": StatementType.MANAGEMENT_CLAIM.value},
                )
            )
            order += 1

        def assign_paths(nodes: list[Any], parent: list[str]) -> None:
            for node in nodes:
                if isinstance(node, SectionNode):
                    path = [*parent, node.title_normalized]
                    node.section_path = path
                    assign_paths(node.children, path)
                else:
                    node.section_path = list(parent)

        assign_paths(root, [])
        return DocumentAST(document_id=metadata.source_document_id, children=root), assets

    @staticmethod
    def _table_node(
        table: PdfTable,
        metadata: DocumentMetadata,
        uri: str | None,
        order: int,
    ) -> TableNode:
        ref = SourceRef(
            source_type=SourceType.IR,
            document_id=metadata.source_document_id,
            uri=uri,
            page=table.page,
            bbox=_bbox(table.bbox),
            source_hash=metadata.raw_sha256,
        )
        header_count = 1 if table.matrix and any(_numeric_value(value) is None for value in table.matrix[0] if value) else 0
        rows: list[TableRow] = []
        for row_index, values in enumerate(table.matrix):
            cells: list[TableCell] = []
            for column, value in enumerate(values):
                raw_cell_bbox = (
                    table.cell_bboxes[row_index][column]
                    if row_index < len(table.cell_bboxes)
                    and column < len(table.cell_bboxes[row_index])
                    else None
                )
                cell_ref = ref.model_copy(
                    update={
                        "bbox": _bbox(raw_cell_bbox) if raw_cell_bbox is not None else ref.bbox,
                        "table_row": row_index,
                        "table_col": column,
                    }
                )
                cells.append(
                    TableCell(
                        row=row_index,
                        col=column,
                        origin_row=row_index,
                        origin_col=column,
                        raw_text=value,
                        normalized_text=normalize_text(value),
                        is_header=row_index < header_count,
                        numeric_value=_numeric_value(value),
                        source_ref=cell_ref,
                    )
                )
            rows.append(TableRow(index=row_index, cells=cells))
        width = len(table.matrix[0]) if table.matrix else 0
        headers = [
            TableHeader(
                col=column,
                path=(
                    [table.matrix[0][column]]
                    if header_count and table.matrix[0][column]
                    else [f"Column {column + 1}"]
                ),
            )
            for column in range(width)
        ]
        label = next(
            (
                value
                for row in table.matrix[:2]
                for value in row
                if value and _YEAR.search(value)
            ),
            None,
        )
        raw_text = "\n".join("\t".join(row) for row in table.matrix)
        return TableNode(
            node_id=stable_id(
                "PT",
                metadata.source_document_id,
                table.page,
                table.bbox,
                table.matrix,
            ),
            order=order,
            raw_text=raw_text,
            normalized_text=normalize_text(raw_text),
            source_refs=[ref],
            attributes={
                "statement_type": StatementType.MANAGEMENT_CLAIM.value,
                "table_extraction_strategy": table.strategy,
            },
            caption=f"IR table — page {table.page}",
            period=(
                ReportingPeriod(
                    kind=PeriodKind.UNKNOWN,
                    fiscal_year=int(_YEAR.search(label).group(1)),
                    raw_label=label,
                )
                if label and _YEAR.search(label)
                else None
            ),
            column_headers=headers,
            header_row_count=header_count,
            rows=rows,
        )
