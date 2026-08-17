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


PARSER_VERSION = "pymupdf-ir/0.4.0"
_KST = ZoneInfo("Asia/Seoul")
_NUMBER = re.compile(r"^\(?[-+]?\d[\d,]*(?:\.\d+)?\)?%?$")
_PACKED_NUMBER = re.compile(r"(?<![A-Za-z0-9])[-(]?\d[\d,]*(?:\.\d+)?%?\)?")
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


def _collapsed_grid(matrix: tuple[tuple[str, ...], ...]) -> bool:
    width = len(matrix[0]) if matrix else 0
    if width < 3 or len(matrix) < 2:
        return False
    for row in matrix[1:]:
        nonempty = [value for value in row if value]
        packed_counts = [len(_PACKED_NUMBER.findall(value)) for value in nonempty]
        if sum(count >= 2 for count in packed_counts) >= 2 or any(
            count >= 3 for count in packed_counts
        ):
            return True
    return False


def _numeric_word_groups(
    words: list[Any],
    *,
    minimum_y: float,
) -> list[list[Any]]:
    numeric_words = [
        word
        for word in words
        if (float(word[1]) + float(word[3])) / 2 >= minimum_y
        and _NUMBER.fullmatch(normalize_text(_safe_text(word[4])))
    ]
    if not numeric_words:
        return []
    word_height = median(float(word[3]) - float(word[1]) for word in numeric_words)
    tolerance = max(2.0, word_height * 0.40)
    groups: list[list[Any]] = []
    for word in sorted(
        numeric_words,
        key=lambda item: ((float(item[1]) + float(item[3])) / 2, float(item[0])),
    ):
        center = (float(word[1]) + float(word[3])) / 2
        if groups:
            prior_center = median(
                (float(item[1]) + float(item[3])) / 2 for item in groups[-1]
            )
            if abs(center - prior_center) <= tolerance:
                groups[-1].append(word)
                continue
        groups.append([word])
    return groups


def _matrix_from_axes(
    words: list[Any],
    columns: list[tuple[float, float]],
    rows: list[tuple[float, float]],
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[tuple[float, float, float, float], ...], ...],
]:
    rebuilt: list[tuple[str, ...]] = []
    cell_rows: list[tuple[tuple[float, float, float, float], ...]] = []
    for y0, y1 in rows:
        values: list[str] = []
        boxes: list[tuple[float, float, float, float]] = []
        for x0, x1 in columns:
            tokens = [
                word
                for word in words
                if x0 <= (float(word[0]) + float(word[2])) / 2 < x1
                and y0 <= (float(word[1]) + float(word[3])) / 2 < y1
            ]
            tokens.sort(
                key=lambda word: (
                    int(word[5]) if len(word) > 5 else 0,
                    int(word[6]) if len(word) > 6 else 0,
                    int(word[7]) if len(word) > 7 else 0,
                    float(word[1]),
                    float(word[0]),
                )
            )
            values.append(
                normalize_text(" ".join(_safe_text(word[4]) for word in tokens))
            )
            boxes.append((x0, y0, x1, y1))
        rebuilt.append(tuple(values))
        cell_rows.append(tuple(boxes))
    return tuple(rebuilt), tuple(cell_rows)


def _repair_collapsed_grid(
    page: Any,
    table: Any,
    matrix: tuple[tuple[str, ...], ...],
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[tuple[float, float, float, float] | None, ...], ...],
] | None:
    """Rebuild a collapsed ruled grid from native word centers and surviving axes."""

    if not _collapsed_grid(matrix) or not table.rows:
        return None
    width = len(matrix[0])
    header_cells = table.rows[0].cells
    if len(header_cells) < width or any(cell is None for cell in header_cells[:width]):
        return None
    columns = [tuple(map(float, cell)) for cell in header_cells[:width]]
    if any(columns[index][2] > columns[index + 1][0] + 1.0 for index in range(width - 1)):
        return None

    try:
        words = list(page.get_text("words", clip=table.bbox, sort=True))
    except TypeError:
        words = list(page.get_text("words", sort=True))
    header_y0 = min(column[1] for column in columns)
    header_y1 = max(column[3] for column in columns)
    groups = _numeric_word_groups(words, minimum_y=header_y1)
    groups = [group for group in groups if len(group) >= max(2, width // 2)]
    if len(groups) < 2:
        return None
    centers = [
        median((float(word[1]) + float(word[3])) / 2 for word in group)
        for group in groups
    ]
    table_bottom = float(table.bbox[3])
    boundaries = [
        header_y1,
        *[
            (centers[index] + centers[index + 1]) / 2
            for index in range(len(centers) - 1)
        ],
        table_bottom,
    ]
    rows: list[tuple[float, float]] = [(header_y0, header_y1)]
    rows.extend(
        (boundaries[index], boundaries[index + 1])
        for index in range(len(centers))
    )
    rebuilt_matrix, cell_rows = _matrix_from_axes(
        words,
        [(column[0], column[2]) for column in columns],
        rows,
    )
    if _collapsed_grid(rebuilt_matrix):
        return None
    prior_nonempty = sum(bool(value) for row in matrix for value in row)
    rebuilt_nonempty = sum(bool(value) for row in rebuilt_matrix for value in row)
    if rebuilt_nonempty <= prior_nonempty:
        return None
    return rebuilt_matrix, cell_rows


def _repair_collapsed_numeric_grid(
    page: Any,
    table: Any,
    matrix: tuple[tuple[str, ...], ...],
) -> tuple[
    tuple[tuple[str, ...], ...],
    tuple[tuple[tuple[float, float, float, float] | None, ...], ...],
] | None:
    """Infer lost numeric columns when even the ruled header axes collapsed."""

    if not _collapsed_grid(matrix) or not table.rows:
        return None
    try:
        words = list(page.get_text("words", clip=table.bbox, sort=True))
    except TypeError:
        words = list(page.get_text("words", sort=True))
    if not words:
        return None

    header_cells = [cell for cell in table.rows[0].cells if cell is not None]
    if not header_cells:
        return None
    header_y0 = min(float(cell[1]) for cell in header_cells)
    header_y1 = max(float(cell[3]) for cell in header_cells)
    row_groups = _numeric_word_groups(words, minimum_y=header_y1)
    row_groups = [group for group in row_groups if len(group) >= 3]
    if len(row_groups) < 2:
        return None

    numeric_words = [word for group in row_groups for word in group]
    word_width = median(float(word[2]) - float(word[0]) for word in numeric_words)
    x_tolerance = max(4.0, word_width * 0.75)
    x_clusters: list[list[Any]] = []
    # Financial tables normally right-align numeric values, so the right edge is
    # substantially more stable than the word centre when digit counts differ.
    for word in sorted(numeric_words, key=lambda item: float(item[2])):
        anchor = float(word[2])
        if x_clusters:
            prior_anchor = median(float(item[2]) for item in x_clusters[-1])
            if abs(anchor - prior_anchor) <= x_tolerance:
                x_clusters[-1].append(word)
                continue
        x_clusters.append([word])

    row_tolerance = max(
        2.0,
        median(float(word[3]) - float(word[1]) for word in numeric_words) * 0.40,
    )

    def row_count(cluster: list[Any]) -> int:
        centers = sorted((float(word[1]) + float(word[3])) / 2 for word in cluster)
        count = 0
        prior: float | None = None
        for center in centers:
            if prior is None or abs(center - prior) > row_tolerance:
                count += 1
                prior = center
        return count

    minimum_occurrences = max(2, (len(row_groups) + 1) // 2)
    x_clusters = [
        cluster for cluster in x_clusters if row_count(cluster) >= minimum_occurrences
    ]
    if len(x_clusters) < 2:
        return None
    x_clusters.sort(key=lambda cluster: median(float(word[2]) for word in cluster))
    anchors = [median(float(word[2]) for word in cluster) for cluster in x_clusters]
    if any(right - left < max(8.0, word_width) for left, right in zip(anchors, anchors[1:])):
        return None

    first_numeric_x0 = min(float(word[0]) for word in x_clusters[0])
    data_label_right = max(
        (
            float(word[2])
            for word in words
            if (float(word[1]) + float(word[3])) / 2 >= header_y1
            and not _NUMBER.fullmatch(normalize_text(_safe_text(word[4])))
            and float(word[2]) < first_numeric_x0
        ),
        default=float(table.bbox[0]),
    )
    first_boundary = (data_label_right + first_numeric_x0) / 2
    if not float(table.bbox[0]) < first_boundary < anchors[0]:
        return None
    x_boundaries = [
        first_boundary,
        *[(anchors[index] + anchors[index + 1]) / 2 for index in range(len(anchors) - 1)],
    ]
    columns = [(float(table.bbox[0]), first_boundary)]
    columns.extend(
        (x_boundaries[index], x_boundaries[index + 1])
        for index in range(len(x_boundaries) - 1)
    )
    columns.append((x_boundaries[-1], float(table.bbox[2]) + 0.01))

    y_centers = [
        median((float(word[1]) + float(word[3])) / 2 for word in group)
        for group in row_groups
    ]
    y_boundaries = [
        header_y1,
        *[(y_centers[index] + y_centers[index + 1]) / 2 for index in range(len(y_centers) - 1)],
        float(table.bbox[3]) + 0.01,
    ]
    rows: list[tuple[float, float]] = [(header_y0, header_y1)]
    rows.extend(
        (y_boundaries[index], y_boundaries[index + 1])
        for index in range(len(y_centers))
    )
    rebuilt_matrix, cell_rows = _matrix_from_axes(words, columns, rows)
    if _collapsed_grid(rebuilt_matrix):
        return None
    if any(
        sum(_numeric_value(value) is not None for value in row) < len(x_clusters) // 2
        for row in rebuilt_matrix[1:]
    ):
        return None
    return rebuilt_matrix, cell_rows


def _numeric_coordinate_match_rate(
    page: Any,
    matrix: tuple[tuple[str, ...], ...],
    cell_rows: tuple[
        tuple[tuple[float, float, float, float] | None, ...], ...
    ],
) -> tuple[int, float | None]:
    """Verify that numeric cell text exists as one native word inside its bbox."""

    checked = 0
    matched = 0
    for row_index, row in enumerate(matrix):
        for column, value in enumerate(row):
            if _numeric_value(value) is None:
                continue
            bbox = (
                cell_rows[row_index][column]
                if row_index < len(cell_rows) and column < len(cell_rows[row_index])
                else None
            )
            if bbox is None:
                continue
            checked += 1
            try:
                words = page.get_text("words", clip=bbox, sort=True)
            except TypeError:
                words = page.get_text("words", sort=True)
            target = normalize_text(value).replace("−", "-")
            if any(
                normalize_text(_safe_text(word[4])).replace("−", "-") == target
                for word in words
            ):
                matched += 1
    return checked, (matched / checked if checked else None)


def enrich_pdf_table_semantics(
    bundle: CanonicalDocumentBundle,
) -> CanonicalDocumentBundle:
    """Attach page-grounded period/unit context omitted by PDF table finders."""

    if bundle.metadata.source_type not in {
        SourceType.IR,
        SourceType.ANALYST,
        SourceType.INDUSTRY,
    }:
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


# Backwards-compatible public name retained for callers that imported the IR-only
# helper before analyst and industry PDFs shared the same canonical table lane.
enrich_ir_table_semantics = enrich_pdf_table_semantics


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


def _page_bbox(
    page: Any,
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Map table-finder coordinates into the visible page coordinate system."""

    if not getattr(page, "rotation", 0):
        return value
    import fitz

    rect = fitz.Rect(value) * page.derotation_matrix
    return tuple(map(float, rect))


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
    token = value.strip().replace("−", "-")
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
    document_type = DocumentType.IR_PRESENTATION
    statement_type = StatementType.MANAGEMENT_CLAIM
    source_label = "IR"
    parser_version = PARSER_VERSION
    text_transform = "pymupdf_page_layout_to_canonical_ast"
    asset_transform = "pymupdf_image_region_extract"
    asset_kind = AssetKind.IMAGE
    always_emit_figure_nodes = False
    table_text_suppression_threshold = 0.60

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
        source_specific.setdefault("statement_type", self.statement_type.value)
        return DocumentMetadata(
            source_type=self.source_type,
            source_document_id=document_id,
            document_type=self.document_type,
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
            parser_version=self.parser_version,
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
            if strategy == "text" and any(
                table.strategy.startswith("lines") for table in candidates
            ):
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
                    if len(raw_matrix) < 2 or not 2 <= width <= 40:
                        continue
                    matrix = tuple(tuple([*row, *([""] * (width - len(row)))]) for row in raw_matrix)
                    raw_bbox = tuple(map(float, table.bbox))
                    bbox = _page_bbox(page, raw_bbox)
                    repaired = (
                        _repair_collapsed_grid(page, table, matrix)
                        if strategy == "lines"
                        else None
                    )
                    if repaired is None and strategy == "lines":
                        repaired = _repair_collapsed_numeric_grid(page, table, matrix)
                    if repaired is not None:
                        matrix, repaired_cell_rows = repaired
                        snapshot.warnings.append(
                            "TABLE_GRID_REPAIRED: rebuilt collapsed ruled table "
                            "from native word coordinates"
                        )
                    else:
                        repaired_cell_rows = None
                    if _collapsed_grid(matrix):
                        snapshot.warnings.append(
                            "TABLE_CANDIDATE_REJECTED_COLLAPSED_GRID: native text retained"
                        )
                        continue
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
                    cell_rows: list[
                        tuple[tuple[float, float, float, float] | None, ...]
                    ] = []
                    if repaired_cell_rows is not None:
                        cell_rows.extend(
                            tuple(
                                _page_bbox(page, cell) if cell is not None else None
                                for cell in row
                            )
                            for row in repaired_cell_rows
                        )
                    else:
                        for row_index in range(len(matrix)):
                            raw_cells = (
                                table.rows[row_index].cells
                                if row_index < len(table.rows)
                                else []
                            )
                            converted = [
                                _page_bbox(
                                    page,
                                    tuple(map(float, raw_cells[column])),
                                )
                                if column < len(raw_cells)
                                and raw_cells[column] is not None
                                else None
                                for column in range(width)
                            ]
                            cell_rows.append(tuple(converted))
                    checked, coordinate_match_rate = _numeric_coordinate_match_rate(
                        page,
                        matrix,
                        tuple(cell_rows),
                    )
                    if checked >= 8 and (
                        coordinate_match_rate is None or coordinate_match_rate < 0.90
                    ):
                        snapshot.warnings.append(
                            "TABLE_CANDIDATE_REJECTED_NUMERIC_COORDINATE_MISMATCH: "
                            f"{coordinate_match_rate:.3f} across {checked} numeric cells; "
                            "native text retained"
                        )
                        continue
                    candidates.append(
                        PdfTable(
                            page=snapshot.number,
                            strategy=(
                                f"{strategy}-coordinate-repair"
                                if repaired is not None
                                else strategy
                            ),
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
            key=lambda item: (
                0 if item.strategy.startswith("lines") else 1,
                item.bbox[1],
                item.bbox[0],
            ),
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
            warnings.append(
                f"visible text retention is below the 90% {self.source_label} PDF target"
            )
        if numeric_retention is not None and numeric_retention < 0.99:
            warnings.append("numeric cell retention is below the 99% IR PDF target")

        records: dict[str, ProvenanceRecord] = {}
        for node in nodes:
            records[node.node_id] = ProvenanceRecord(
                object_id=node.node_id,
                source_refs=node.source_refs,
                transform=self.text_transform,
                transform_version=self.parser_version,
            )
        for asset in assets:
            records[asset.asset_id] = ProvenanceRecord(
                object_id=asset.asset_id,
                source_refs=asset.source_refs,
                transform=self.asset_transform,
                transform_version=self.parser_version,
            )
        values = [
            normalize_text(node.normalized_text).casefold()
            for node in nodes
            if not isinstance(node, SectionNode) and normalize_text(node.normalized_text)
        ]
        duplicate_ratio = (
            (len(values) - len(set(values))) / len(values) if values else None
        )
        return enrich_pdf_table_semantics(CanonicalDocumentBundle(
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
                if any(
                    _coverage_by(block.bbox, table_bbox)
                    >= self.table_text_suppression_threshold
                    for table_bbox in table_boxes
                ):
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
                    source_type=self.source_type,
                    document_id=metadata.source_document_id,
                    uri=uri,
                    page=page.number,
                    bbox=_bbox(block.bbox),
                    source_hash=parsed.raw_sha256,
                )
                attributes = {
                    "statement_type": self.statement_type.value,
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
                    source_type=self.source_type,
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
                        kind=self.asset_kind,
                        media_type="image/unknown",
                        uri=uri,
                        alt_text=f"PDF image region on page {page.number}",
                        source_refs=[ref],
                    )
                )
                if page.needs_ocr or self.always_emit_figure_nodes:
                    append(
                        FigureNode(
                            node_id=stable_id("PF", asset_id),
                            order=order,
                            raw_text="",
                            normalized_text="",
                            source_refs=[ref],
                            attributes={
                                "statement_type": self.statement_type.value,
                                "ocr_status": (
                                    "APPLIED"
                                    if page.ocr_applied
                                    else "REQUIRED_NOT_CONFIGURED"
                                    if page.needs_ocr
                                    else "NOT_REQUIRED"
                                ),
                                "ocr_engine": page.ocr_engine,
                                "ocr_dpi": page.ocr_dpi,
                                "ocr_mean_confidence": page.ocr_mean_confidence,
                            },
                            asset_id=asset_id,
                            alt_text=(
                                f"Image-dominant {self.source_label} page {page.number}; OCR applied"
                                if page.ocr_applied
                                else f"Image-dominant {self.source_label} page {page.number}; OCR required"
                                if page.needs_ocr
                                else f"{self.source_label} chart/figure region on page {page.number}"
                            ),
                        )
                    )
                    order += 1
            page_ref = SourceRef(
                source_type=self.source_type,
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
                    attributes={"statement_type": self.statement_type.value},
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

    def _table_node(
        self,
        table: PdfTable,
        metadata: DocumentMetadata,
        uri: str | None,
        order: int,
    ) -> TableNode:
        ref = SourceRef(
            source_type=self.source_type,
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
                "statement_type": self.statement_type.value,
                "table_extraction_strategy": table.strategy,
            },
            caption=f"{self.source_label} table — page {table.page}",
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
