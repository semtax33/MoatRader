from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Literal
from zoneinfo import ZoneInfo

from lxml import etree, html

from moatrader.adapters.base import RawDocument, SourceAdapter
from moatrader.canonical.ids import content_hash, normalize_text, stable_id
from moatrader.canonical.models import (
    AssetKind,
    AvailabilityPrecision,
    CanonicalDocumentBundle,
    ClassificationTrace,
    ConsolidationScope,
    DocumentAST,
    DocumentAsset,
    DocumentMetadata,
    DocumentType,
    FactDimension,
    FigureNode,
    ListItemNode,
    ListNode,
    NoteNode,
    ParagraphNode,
    PeriodKind,
    ProvenanceIndex,
    ProvenanceRecord,
    QualityMetrics,
    ReportingPeriod,
    SectionNode,
    SectionRole,
    SourceRef,
    SourceType,
    StructuredFact,
    TableCell,
    TableFootnote,
    TableHeader,
    TableNode,
    TableRow,
    UnitSpec,
    UnknownBlockNode,
)


PARSER_VERSION = "html-ast/0.2.0"
_DROP_TAGS = {"script", "style", "meta", "link", "noscript", "template", "head"}
_XBRL_INFRA_TAGS = {"context", "unit", "schemaref", "resources", "references", "hidden"}
_DART_SECTION_TAG_RE = re.compile(r"^section-(\d+)$")
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "div",
    "document",
    "dl",
    "fieldset",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "ul",
}
_NOTE_RE = re.compile(r"^\s*(?P<marker>주\s*\d+\)|※|\*|주석\s*[:：])")
_UNIT_RE = re.compile(r"(?:\(\s*)?단위\s*[:：]\s*(?P<unit>[^)\n]+)\)?", re.IGNORECASE)
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?:\s*년)?(?:\s*(Q[1-4]|[1-4]Q|[1-4]분기|반기))?")
_HIDDEN_RE = re.compile(r"(?:display\s*:\s*none|visibility\s*:\s*hidden)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?%?$")
_PAREN_NUMBER_RE = re.compile(r"^\((\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\)$")
_BYTE_CHARSET_RE = re.compile(
    br"(?:charset\s*=\s*[\"']?\s*|encoding\s*=\s*[\"']\s*)([A-Za-z0-9._-]+)",
    re.IGNORECASE,
)
_XML_DECLARATION_RE = re.compile(r"^\ufeff?\s*<\?xml[^>]*\?>", re.IGNORECASE)

_HEADING_PATTERNS: list[tuple[str, re.Pattern[str], int]] = [
    ("roman", re.compile(r"^\s*[IVXLCDMⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.\s*"), 1),
    ("decimal_nested", re.compile(r"^\s*\d+(?:\.\d+)+\s+"), 3),
    ("decimal", re.compile(r"^\s*\d+\.\s+"), 2),
    ("korean", re.compile(r"^\s*[가-힣]\.\s+"), 3),
    ("paren_num", re.compile(r"^\s*\(\d+\)\s*"), 4),
    ("circled", re.compile(r"^\s*[①②③④⑤⑥⑦⑧⑨⑩]\s*"), 5),
]

_ROLE_PATTERNS: list[tuple[SectionRole, re.Pattern[str]]] = [
    (SectionRole.FINANCIALS, re.compile(r"재무제표|financial statements?", re.I)),
    (SectionRole.MDA, re.compile(r"경영진.{0,12}(논의|분석)|management.{0,20}discussion|md&a", re.I)),
    (SectionRole.RISK, re.compile(r"위험\s*요인|risk factors?", re.I)),
    (SectionRole.COMPETITION, re.compile(r"경쟁\s*(상황|환경|현황)|competition|competitive landscape", re.I)),
    (SectionRole.CUSTOMERS, re.compile(r"고객|customer", re.I)),
    (SectionRole.SUPPLIERS, re.compile(r"공급|supplier", re.I)),
    (SectionRole.PRODUCTS, re.compile(r"제품|서비스|products?|services?", re.I)),
    (SectionRole.GUIDANCE, re.compile(r"가이던스|전망|guidance|outlook", re.I)),
    (SectionRole.BUSINESS, re.compile(r"사업의\s*내용|사업\s*개요|business(?:\s+overview)?", re.I)),
    (SectionRole.COMPANY_OVERVIEW, re.compile(r"회사의\s*개요|company\s+overview", re.I)),
    (SectionRole.NOTES, re.compile(r"주석|notes?\s+to", re.I)),
    (SectionRole.GOVERNANCE, re.compile(r"지배구조|governance", re.I)),
]


@dataclass(slots=True)
class ParsedHtml:
    root: html.HtmlElement
    raw_sha256: str
    encoding: str | None
    raw_visible_chars: int
    raw_table_count: int
    raw_numeric_cell_count: int


@dataclass(slots=True)
class BlockEvent:
    event_type: Literal["HEADING", "PARAGRAPH", "TABLE", "LIST", "NOTE", "FIGURE", "UNKNOWN"]
    payload: Any
    level: int | None = None
    explicit_level: int | None = None
    inferred_level: int | None = None
    confidence: float = 1.0


@dataclass(slots=True)
class ParseState:
    document_id: str
    source_type: SourceType
    source_hash: str
    uri: str | None
    events: list[BlockEvent] = field(default_factory=list)
    order: int = 0

    def next_order(self) -> int:
        result = self.order
        self.order += 1
        return result


def decode_html_document(content: bytes) -> tuple[str, str]:
    """Decode deterministically; Korean filings often omit or misplace charset metadata."""
    if content.startswith(b"\xef\xbb\xbf"):
        return content.decode("utf-8-sig"), "utf-8-sig"
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16"), "utf-16"
    header = content[:4096]
    match = _BYTE_CHARSET_RE.search(header)
    candidates: list[str] = []
    if match:
        declared = match.group(1).decode("ascii", errors="ignore").lower()
        candidates.append({"euc-kr": "cp949", "ks_c_5601-1987": "cp949"}.get(declared, declared))
    candidates.extend(["utf-8", "cp949"])
    tried: set[str] = set()
    for encoding in candidates:
        if not encoding or encoding in tried:
            continue
        tried.add(encoding)
        try:
            return content.decode(encoding), encoding
        except (LookupError, UnicodeDecodeError):
            continue
    return content.decode("utf-8", errors="replace"), "utf-8-replace"


def _tag(element: etree._Element) -> str:
    if not isinstance(element.tag, str):
        return ""
    return element.tag.rsplit("}", 1)[-1].split(":")[-1].lower()


def _attribute(element: etree._Element, name: str) -> str | None:
    target = name.lower()
    for key, value in element.attrib.items():
        if key.rsplit("}", 1)[-1].split(":")[-1].lower() == target:
            return value
    return None


def _xpath(element: etree._Element) -> str:
    return element.getroottree().getpath(element)


def _source_ref(element: etree._Element, state: ParseState, **kwargs: Any) -> SourceRef:
    return SourceRef(
        source_type=state.source_type,
        document_id=state.document_id,
        uri=state.uri,
        xpath=_xpath(element),
        source_hash=state.source_hash,
        **kwargs,
    )


def _inline_text(element: etree._Element, *, skip_nested_tables: bool = True) -> str:
    parts: list[str] = []

    def visit(node: etree._Element, root: etree._Element) -> None:
        if node.text:
            parts.append(node.text)
        for child in node:
            tag = _tag(child)
            if tag == "br":
                parts.append("\n")
            elif tag == "table" and skip_nested_tables and child is not root:
                pass
            elif tag not in _DROP_TAGS:
                visit(child, root)
            if child.tail:
                parts.append(child.tail)

    visit(element, element)
    return normalize_text("".join(parts))


def _meaningful_fragment(value: str | None) -> str:
    return normalize_text(value or "")


def _has_direct_block_children(element: etree._Element) -> bool:
    return any(
        _tag(child) in _BLOCK_TAGS or _DART_SECTION_TAG_RE.fullmatch(_tag(child)) is not None
        for child in element
    )


def _heading_pattern(text: str) -> tuple[str, int] | None:
    for name, pattern, level in _HEADING_PATTERNS:
        if pattern.search(text):
            return name, level
    return None


def _style_is_bold(element: etree._Element) -> bool:
    if _tag(element) in {"b", "strong"}:
        return True
    style = (_attribute(element, "style") or "").lower()
    return "font-weight:bold" in style.replace(" ", "") or "font-weight:700" in style.replace(" ", "")


def _heading_score(element: etree._Element, text: str) -> tuple[float, int | None, list[str]]:
    tag = _tag(element)
    reasons: list[str] = []
    score = 0.0
    inferred: int | None = None
    if tag in {f"h{i}" for i in range(1, 7)}:
        score += 1.0
        inferred = int(tag[1])
        reasons.append(f"explicit_tag={tag}")
    pattern = _heading_pattern(text)
    if pattern:
        name, inferred = pattern
        score += 0.45
        reasons.append(f"numbering_pattern={name}")
    if _style_is_bold(element):
        score += 0.2
        reasons.append("bold=true")
    if len(text) <= 80:
        score += 0.15
        reasons.append(f"short_text={len(text)}")
    if text.endswith((".", "다.", "요.", "?", "!")):
        score -= 0.3
        reasons.append("sentence_ending=true")
    if len(text) > 160:
        score -= 0.5
        reasons.append("long_text=true")
    return max(0.0, min(score, 1.0)), inferred, reasons


def _section_role(title: str) -> SectionRole | None:
    for role, pattern in _ROLE_PATTERNS:
        if pattern.search(title):
            return role
    return None


def _canonical_unit(raw: str) -> UnitSpec:
    compact = re.sub(r"\s+", "", raw).upper()
    mappings: list[tuple[re.Pattern[str], str, Decimal, str | None]] = [
        (re.compile(r"백만(?:원|KRW)|KRW백만"), "KRW_MILLION", Decimal("1000000"), "KRW"),
        (re.compile(r"천(?:원|KRW)|KRW천"), "KRW_THOUSAND", Decimal("1000"), "KRW"),
        (re.compile(r"억(?:원|KRW)"), "KRW_HUNDRED_MILLION", Decimal("100000000"), "KRW"),
        (re.compile(r"백만USD|USD백만"), "USD_MILLION", Decimal("1000000"), "USD"),
        (re.compile(r"천USD|USD천"), "USD_THOUSAND", Decimal("1000"), "USD"),
        (re.compile(r"USD|달러"), "USD", Decimal("1"), "USD"),
        (re.compile(r"KRW|원"), "KRW", Decimal("1"), "KRW"),
        (re.compile(r"%|PERCENT"), "PERCENT", Decimal("1"), None),
        (re.compile(r"명|PERSON"), "PERSON", Decimal("1"), None),
    ]
    for pattern, canonical, scale, currency in mappings:
        if pattern.search(compact):
            return UnitSpec(raw=raw, canonical=canonical, scale=scale, currency=currency)
    return UnitSpec(raw=raw, canonical=None)


def _numeric_value(raw: str) -> tuple[Decimal | None, UnitSpec | None]:
    value = normalize_text(raw).replace(" ", "")
    if not value or value in {"-", "—", "–", "N/A", "n/a"}:
        return None, None
    negative = False
    match = _PAREN_NUMBER_RE.match(value)
    if match:
        value = match.group(1)
        negative = True
    is_percent = value.endswith("%")
    if is_percent:
        value = value[:-1]
    if not _NUMBER_RE.match(value + ("%" if is_percent else "")):
        return None, None
    try:
        parsed = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None, None
    if negative:
        parsed = -parsed
    if is_percent:
        return parsed / Decimal("100"), UnitSpec(raw="%", canonical="PERCENT")
    return parsed, None


def _period_from_text(text: str) -> ReportingPeriod | None:
    match = _YEAR_RE.search(text)
    if not match:
        return None
    return ReportingPeriod(
        kind=PeriodKind.UNKNOWN,
        fiscal_year=int(match.group(1)),
        fiscal_period=match.group(2),
        raw_label=match.group(0),
    )


def _nearest_table(element: etree._Element) -> etree._Element | None:
    current = element.getparent()
    while current is not None:
        if _tag(current) == "table":
            return current
        current = current.getparent()
    return None


def _direct_rows(table: etree._Element) -> list[etree._Element]:
    return [row for row in table.iterdescendants() if _tag(row) == "tr" and _nearest_table(row) is table]


def _parse_table(element: etree._Element, state: ParseState) -> TableNode:
    source_ref = _source_ref(element, state)
    original_rows = _direct_rows(element)
    grid: dict[tuple[int, int], TableCell] = {}
    max_col = 0
    max_row = len(original_rows)

    for row_index, row_element in enumerate(original_rows):
        col_index = 0
        cells = [child for child in row_element if _tag(child) in {"th", "td"}]
        for cell_element in cells:
            while (row_index, col_index) in grid:
                col_index += 1
            try:
                rowspan = max(1, int(_attribute(cell_element, "rowspan") or "1"))
            except ValueError:
                rowspan = 1
            try:
                colspan = max(1, int(_attribute(cell_element, "colspan") or "1"))
            except ValueError:
                colspan = 1
            raw_text = _inline_text(cell_element)
            normalized = normalize_text(raw_text)
            numeric, cell_unit = _numeric_value(normalized)
            is_header = _tag(cell_element) == "th"
            cell_ref = _source_ref(cell_element, state, table_row=row_index, table_col=col_index)
            for target_row in range(row_index, row_index + rowspan):
                for target_col in range(col_index, col_index + colspan):
                    grid[(target_row, target_col)] = TableCell(
                        row=target_row,
                        col=target_col,
                        origin_row=row_index,
                        origin_col=col_index,
                        raw_text=raw_text,
                        normalized_text=normalized,
                        is_header=is_header,
                        propagated=(target_row != row_index or target_col != col_index),
                        source_rowspan=rowspan,
                        source_colspan=colspan,
                        numeric_value=numeric,
                        unit=cell_unit,
                        source_ref=cell_ref,
                    )
            col_index += colspan
            max_col = max(max_col, col_index)
            max_row = max(max_row, row_index + rowspan)

    canonical_rows: list[TableRow] = []
    for row_index in range(max_row):
        canonical_cells: list[TableCell] = []
        for col_index in range(max_col):
            cell = grid.get((row_index, col_index))
            if cell is None:
                cell = TableCell(
                    row=row_index,
                    col=col_index,
                    origin_row=row_index,
                    origin_col=col_index,
                    source_ref=source_ref,
                )
            canonical_cells.append(cell)
        canonical_rows.append(TableRow(index=row_index, cells=canonical_cells))

    header_row_count = 0
    for row in canonical_rows:
        nonempty = [cell for cell in row.cells if cell.normalized_text]
        if nonempty and all(cell.is_header for cell in nonempty):
            header_row_count += 1
        else:
            break

    column_headers: list[TableHeader] = []
    for col_index in range(max_col):
        path: list[str] = []
        for row_index in range(header_row_count):
            text = canonical_rows[row_index].cells[col_index].normalized_text
            if text and (not path or text != path[-1]):
                path.append(text)
        column_headers.append(TableHeader(col=col_index, path=path))

    previous_texts: list[str] = []
    for event in reversed(state.events[-4:]):
        if event.event_type == "HEADING":
            break
        payload_text = getattr(event.payload, "normalized_text", "")
        if payload_text:
            previous_texts.append(payload_text)
    previous_texts.reverse()
    context_text = "\n".join(previous_texts)
    unit_match = _UNIT_RE.search(context_text)
    unit = _canonical_unit(unit_match.group("unit").strip()) if unit_match else None
    caption_elements = [child for child in element if _tag(child) == "caption"]
    caption = _inline_text(caption_elements[0]) if caption_elements else None
    if not caption:
        for candidate in reversed(previous_texts):
            if not _UNIT_RE.search(candidate) and not _NOTE_RE.search(candidate) and len(candidate) <= 160:
                caption = candidate
                break
    period = _period_from_text("\n".join(filter(None, [caption or "", context_text])))
    normalized_table_text = "\n".join(
        " | ".join(cell.normalized_text for cell in row.cells) for row in canonical_rows
    )
    node_id = stable_id("T", state.document_id, source_ref.xpath, "table", normalized_table_text)
    return TableNode(
        node_id=node_id,
        order=state.next_order(),
        raw_text=_inline_text(element),
        normalized_text=normalized_table_text,
        source_refs=[source_ref],
        caption=caption,
        unit=unit,
        period=period,
        column_headers=column_headers,
        header_row_count=header_row_count,
        rows=canonical_rows,
        classification=ClassificationTrace(
            rule_id="R013_TABLE",
            confidence=1.0,
            reasons=["tag=table", "rowspan_colspan_expanded=true"],
        ),
    )


def _emit_text_event(
    element: etree._Element,
    text: str,
    state: ParseState,
    *,
    forced_rule: str | None = None,
    forced_explicit_level: int | None = None,
) -> None:
    normalized = normalize_text(text)
    if not normalized:
        return
    source_ref = _source_ref(element, state)
    note_match = _NOTE_RE.match(normalized)
    score, inferred_level, reasons = _heading_score(element, normalized)
    tag = _tag(element)
    explicit_level = forced_explicit_level or (
        int(tag[1]) if tag in {f"h{i}" for i in range(1, 7)} else None
    )
    if forced_explicit_level is not None:
        reasons.append(f"dart_section_tag_level={forced_explicit_level}")

    if note_match:
        node_id = stable_id("N", state.document_id, source_ref.xpath, "note", normalized)
        node = NoteNode(
            node_id=node_id,
            order=state.next_order(),
            raw_text=text,
            normalized_text=normalized,
            source_refs=[source_ref],
            marker=note_match.group("marker"),
            classification=ClassificationTrace(
                rule_id="R014_NOTE",
                confidence=0.95,
                reasons=["note_pattern=true"],
            ),
        )
        state.events.append(BlockEvent("NOTE", node))
        return

    if explicit_level is not None or score >= 0.75:
        level = explicit_level or inferred_level or 2
        node_id = stable_id("S", state.document_id, source_ref.xpath, "section", normalized)
        node = SectionNode(
            node_id=node_id,
            order=state.next_order(),
            raw_text=text,
            normalized_text=normalized,
            source_refs=[source_ref],
            title_raw=text,
            title_normalized=normalized,
            level=level,
            role=_section_role(normalized),
            heading_confidence=1.0 if explicit_level is not None else score,
            explicit_level=explicit_level,
            inferred_level=inferred_level,
            classification=ClassificationTrace(
                rule_id=forced_rule
                or ("R003_EXPLICIT_HEADING" if explicit_level is not None else "R005_INFERRED_HEADING"),
                confidence=1.0 if explicit_level is not None else score,
                reasons=reasons,
            ),
        )
        state.events.append(
            BlockEvent(
                "HEADING",
                node,
                level=level,
                explicit_level=explicit_level,
                inferred_level=inferred_level,
                confidence=node.heading_confidence,
            )
        )
        return

    node_id = stable_id("P", state.document_id, source_ref.xpath, "paragraph", normalized, state.order)
    node = ParagraphNode(
        node_id=node_id,
        order=state.next_order(),
        raw_text=text,
        normalized_text=normalized,
        source_refs=[source_ref],
        classification=ClassificationTrace(
            rule_id=forced_rule or "R006_PARAGRAPH",
            confidence=0.9 if tag == "p" else 0.75,
            reasons=[f"tag={tag}", "visible_text=true", "inside_table=false"],
        ),
    )
    state.events.append(BlockEvent("PARAGRAPH", node))


def _parse_list(element: etree._Element, state: ParseState) -> ListNode:
    source_ref = _source_ref(element, state)
    items: list[ListItemNode] = []
    for ordinal, child in enumerate(element):
        if _tag(child) != "li":
            continue
        text = _inline_text(child)
        if not text:
            continue
        child_ref = _source_ref(child, state)
        items.append(
            ListItemNode(
                node_id=stable_id("LI", state.document_id, child_ref.xpath, text),
                order=state.next_order(),
                raw_text=text,
                normalized_text=normalize_text(text),
                source_refs=[child_ref],
                ordinal=ordinal + 1 if _tag(element) == "ol" else None,
                classification=ClassificationTrace(rule_id="R012_LIST_ITEM", confidence=1.0),
            )
        )
    text = "\n".join(item.normalized_text for item in items)
    return ListNode(
        node_id=stable_id("L", state.document_id, source_ref.xpath, text),
        order=state.next_order(),
        raw_text=text,
        normalized_text=text,
        source_refs=[source_ref],
        ordered=_tag(element) == "ol",
        items=items,
        classification=ClassificationTrace(rule_id="R011_LIST", confidence=1.0),
    )


def _walk_dom(element: etree._Element, state: ParseState) -> None:
    tag = _tag(element)
    if not tag or tag in _DROP_TAGS or tag in _XBRL_INFRA_TAGS:
        return
    if _HIDDEN_RE.search(_attribute(element, "style") or "") or _attribute(element, "hidden") is not None:
        return
    dart_section = _DART_SECTION_TAG_RE.fullmatch(tag)
    if dart_section:
        level = int(dart_section.group(1))
        title = next(
            (child for child in element if _tag(child) in {"title", "section-title"}),
            None,
        )
        if title is not None:
            _emit_text_event(
                title,
                _inline_text(title),
                state,
                forced_rule="R002_DART_EXPLICIT_SECTION",
                forced_explicit_level=level,
            )
        for child in element:
            if child is not title:
                _walk_dom(child, state)
            tail = _meaningful_fragment(child.tail)
            if tail:
                _emit_text_event(element, tail, state, forced_rule="R016_WRAPPER_TAIL_TEXT")
        return
    if tag == "table":
        state.events.append(BlockEvent("TABLE", _parse_table(element, state)))
        # Nested tables are independent information-bearing objects. The parent
        # grid excludes their text, so preserve each direct nested table once.
        for descendant in element.iterdescendants():
            if _tag(descendant) == "table" and _nearest_table(descendant) is element:
                _walk_dom(descendant, state)
        return
    if tag in {"ul", "ol"}:
        node = _parse_list(element, state)
        if node.items:
            state.events.append(BlockEvent("LIST", node))
        return
    if tag in {"figure", "img"}:
        image = element if tag == "img" else next((item for item in element.iter() if _tag(item) == "img"), element)
        source_ref = _source_ref(element, state)
        asset_source_ref = _source_ref(image, state)
        uri = _attribute(image, "src")
        alt = _attribute(image, "alt")
        caption = _inline_text(element) if tag == "figure" else None
        asset_id = stable_id("A", state.document_id, asset_source_ref.xpath, uri, alt)
        figure = FigureNode(
            node_id=stable_id("F", state.document_id, source_ref.xpath, uri, caption),
            order=state.next_order(),
            raw_text=caption or alt or "",
            normalized_text=normalize_text(caption or alt or ""),
            source_refs=[source_ref],
            asset_id=asset_id,
            caption=caption,
            alt_text=alt,
            classification=ClassificationTrace(rule_id="R015_FIGURE", confidence=1.0),
            attributes={"uri": uri} if uri else {},
        )
        state.events.append(BlockEvent("FIGURE", figure))
        return
    if tag in {f"h{i}" for i in range(1, 7)} | {"title", "section-title"}:
        _emit_text_event(element, _inline_text(element), state)
        return
    if tag in {"p", "pre", "blockquote", "li"}:
        _emit_text_event(element, _inline_text(element), state)
        return

    if not _has_direct_block_children(element):
        text = _inline_text(element)
        if text and tag not in {"html", "body", "tbody", "thead", "tfoot", "tr", "td", "th"}:
            _emit_text_event(element, text, state, forced_rule="R007_LEAF_CONTAINER")
        return

    leading = _meaningful_fragment(element.text)
    if leading:
        _emit_text_event(element, leading, state, forced_rule="R016_WRAPPER_DIRECT_TEXT")
    for child in element:
        _walk_dom(child, state)
        tail = _meaningful_fragment(child.tail)
        if tail:
            _emit_text_event(element, tail, state, forced_rule="R016_WRAPPER_TAIL_TEXT")


def _attach_table_footnotes(events: list[BlockEvent]) -> None:
    last_table: TableNode | None = None
    for event in events:
        if event.event_type == "TABLE":
            last_table = event.payload
        elif event.event_type == "NOTE" and last_table is not None:
            note: NoteNode = event.payload
            note.target_node_ids.append(last_table.node_id)
            last_table.footnotes.append(
                TableFootnote(marker=note.marker, text=note.normalized_text, node_id=note.node_id)
            )
        elif event.event_type not in {"FIGURE"}:
            last_table = None


def _build_section_tree(document_id: str, events: list[BlockEvent]) -> DocumentAST:
    root_children: list[Any] = []
    stack: list[SectionNode] = []
    for event in events:
        node = event.payload
        if event.event_type == "HEADING":
            while stack and stack[-1].level >= node.level:
                stack.pop()
            if stack:
                stack[-1].children.append(node)
            else:
                root_children.append(node)
            stack.append(node)
        elif stack:
            stack[-1].children.append(node)
        else:
            root_children.append(node)

    def assign_paths(nodes: list[Any], parent_path: list[str]) -> None:
        for node in nodes:
            if isinstance(node, SectionNode):
                path = [*parent_path, node.title_normalized]
                node.section_path = path
                assign_paths(node.children, path)
            else:
                node.section_path = list(parent_path)
                if isinstance(node, ListNode):
                    for item in node.items:
                        item.section_path = list(parent_path)

    assign_paths(root_children, [])
    return DocumentAST(document_id=document_id, children=root_children)


def _parse_datetime(value: Any, default_zone: ZoneInfo) -> tuple[datetime, AvailabilityPrecision]:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, time.min)
    elif isinstance(value, str):
        normalized = value.strip().replace("Z", "+00:00")
        result = datetime.fromisoformat(normalized)
    else:
        raise ValueError("available_at (or fetched_at) is required")
    precision = AvailabilityPrecision.DAY if result.time() == time.min and not isinstance(value, datetime) else AvailabilityPrecision.EXACT
    if result.tzinfo is None:
        result = result.replace(tzinfo=default_zone)
        precision = AvailabilityPrecision.DAY if result.time() == time.min else AvailabilityPrecision.INFERRED
    return result, precision


def _optional_datetime(value: Any, default_zone: ZoneInfo) -> datetime | None:
    if value is None:
        return None
    return _parse_datetime(value, default_zone)[0]


def _optional_date(value: Any) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _document_type(hints: dict[str, Any]) -> DocumentType:
    explicit = hints.get("document_type")
    if explicit:
        return DocumentType(str(explicit).upper())
    label = " ".join(str(hints.get(key, "")) for key in ("report_name", "form_type", "title")).lower()
    if any(term in label for term in ("사업보고서", "annual", "10-k", "20-f")):
        return DocumentType.ANNUAL_REPORT
    if any(term in label for term in ("분기보고서", "반기보고서", "quarter", "10-q", "6-k")):
        return DocumentType.QUARTERLY_REPORT
    if any(term in label for term in ("8-k", "current report")):
        return DocumentType.CURRENT_REPORT
    if any(term in label for term in ("ir", "presentation", "investor relations")):
        return DocumentType.IR_PRESENTATION
    if any(term in label for term in ("earnings", "실적발표")):
        return DocumentType.EARNINGS_RELEASE
    return DocumentType.OTHER


class BaseHtmlFinancialAdapter(SourceAdapter[ParsedHtml]):
    source_type: SourceType
    default_zone = ZoneInfo("UTC")

    def parse_structure(self, source: RawDocument) -> ParsedHtml:
        decoded, encoding = decode_html_document(source.content)
        try:
            # lxml rejects a Unicode string that still contains an XML encoding
            # declaration. OpenDART primary documents are XML-shaped HTML and
            # commonly include exactly that declaration, so remove it after the
            # bytes have already been decoded deterministically.
            root = html.document_fromstring(_XML_DECLARATION_RE.sub("", decoded, count=1))
        except (etree.ParserError, ValueError) as exc:
            raise ValueError(f"unable to parse HTML document: {exc}") from exc
        body = next((element for element in root.iter() if _tag(element) == "body"), root)
        raw_visible = _inline_text(body, skip_nested_tables=False)
        raw_tables = sum(1 for element in root.iter() if _tag(element) == "table")
        raw_numeric_cells = sum(
            1
            for element in root.iter()
            if _tag(element) in {"th", "td"} and _numeric_value(_inline_text(element))[0] is not None
        )
        return ParsedHtml(
            root=root,
            raw_sha256=content_hash(source.content),
            encoding=encoding,
            raw_visible_chars=len(re.sub(r"\s+", "", raw_visible)),
            raw_table_count=raw_tables,
            raw_numeric_cell_count=raw_numeric_cells,
        )

    def extract_metadata(self, source: RawDocument) -> DocumentMetadata:
        hints = dict(source.hints)
        raw_hash = content_hash(source.content)
        document_id = str(
            hints.get("source_document_id")
            or hints.get("rcept_no")
            or hints.get("accession_number")
            or raw_hash[:24]
        )
        available_raw = hints.get("available_at") or source.fetched_at
        available_at, precision = _parse_datetime(available_raw, self.default_zone)
        if hints.get("availability_precision"):
            precision = AvailabilityPrecision(str(hints["availability_precision"]).upper())
        start = _optional_date(hints.get("period_start"))
        end = _optional_date(hints.get("period_end"))
        instant = _optional_date(hints.get("period_instant"))
        reporting_period: ReportingPeriod | None = None
        if start and end:
            reporting_period = ReportingPeriod(
                kind=PeriodKind.DURATION,
                start=start,
                end=end,
                fiscal_year=hints.get("fiscal_year"),
                fiscal_period=hints.get("fiscal_period"),
            )
        elif instant:
            reporting_period = ReportingPeriod(
                kind=PeriodKind.INSTANT,
                instant=instant,
                fiscal_year=hints.get("fiscal_year"),
                fiscal_period=hints.get("fiscal_period"),
            )
        source_specific = dict(hints.get("source_specific", {}))
        for key in ("rcept_no", "corp_code", "accession_number", "form_type", "cik"):
            if hints.get(key) is not None:
                source_specific[key] = hints[key]
        return DocumentMetadata(
            source_type=self.source_type,
            source_document_id=document_id,
            document_type=_document_type(hints),
            issuer_id=hints.get("issuer_id") or hints.get("corp_code") or hints.get("cik"),
            issuer_name=hints.get("issuer_name"),
            ticker=hints.get("ticker") or hints.get("stock_code"),
            market=hints.get("market"),
            title=hints.get("title") or hints.get("report_name"),
            published_at=_optional_datetime(hints.get("published_at"), self.default_zone),
            available_at=available_at,
            availability_precision=precision,
            availability_source=str(hints.get("availability_source") or ("source_metadata" if hints.get("available_at") else "fetched_at")),
            reporting_period=reporting_period,
            language=str(hints.get("language") or "und"),
            jurisdiction=hints.get("jurisdiction"),
            is_amendment=bool(hints.get("is_amendment", False)),
            amends_document_id=hints.get("amends_document_id"),
            raw_sha256=raw_hash,
            parser_version=PARSER_VERSION,
            source_specific=source_specific,
        )

    def build_ast(self, parsed: ParsedHtml, metadata: DocumentMetadata, uri: str | None) -> DocumentAST:
        state = ParseState(
            document_id=metadata.source_document_id,
            source_type=self.source_type,
            source_hash=parsed.raw_sha256,
            uri=uri,
        )
        body = next((element for element in parsed.root.iter() if _tag(element) == "body"), parsed.root)
        _walk_dom(body, state)
        _attach_table_footnotes(state.events)
        return _build_section_tree(metadata.source_document_id, state.events)

    def _context_index(self, parsed: ParsedHtml) -> dict[str, tuple[ReportingPeriod, list[FactDimension], str | None]]:
        contexts: dict[str, tuple[ReportingPeriod, list[FactDimension], str | None]] = {}
        for element in parsed.root.iter():
            if _tag(element) != "context":
                continue
            context_id = _attribute(element, "id")
            if not context_id:
                continue
            values: dict[str, str] = {}
            dimensions: list[FactDimension] = []
            entity: str | None = None
            for descendant in element.iterdescendants():
                tag = _tag(descendant)
                text = normalize_text(descendant.text or "")
                if tag in {"startdate", "enddate", "instant"} and text:
                    values[tag] = text
                elif tag == "identifier" and text:
                    entity = text
                elif tag in {"explicitmember", "typedmember"}:
                    axis = _attribute(descendant, "dimension") or "unknown"
                    dimensions.append(FactDimension(axis=axis, member=text or "typed", typed_value=text if tag == "typedmember" else None))
            if "instant" in values:
                period = ReportingPeriod(kind=PeriodKind.INSTANT, instant=date.fromisoformat(values["instant"][:10]))
            elif "startdate" in values and "enddate" in values:
                period = ReportingPeriod(
                    kind=PeriodKind.DURATION,
                    start=date.fromisoformat(values["startdate"][:10]),
                    end=date.fromisoformat(values["enddate"][:10]),
                )
            else:
                period = ReportingPeriod(kind=PeriodKind.UNKNOWN)
            contexts[context_id] = (period, dimensions, entity)
        return contexts

    def extract_facts(
        self,
        parsed: ParsedHtml,
        metadata: DocumentMetadata,
        uri: str | None,
    ) -> list[StructuredFact]:
        contexts = self._context_index(parsed)
        facts: list[StructuredFact] = []
        for element in parsed.root.iter():
            if _tag(element) not in {"nonfraction", "nonnumeric"}:
                continue
            concept = _attribute(element, "name")
            if not concept:
                continue
            raw = _inline_text(element)
            context_id = _attribute(element, "contextref")
            period, dimensions, _entity = contexts.get(
                context_id or "",
                (metadata.reporting_period or ReportingPeriod(kind=PeriodKind.UNKNOWN), [], None),
            )
            numeric, inferred_unit = _numeric_value(raw)
            scale_raw = _attribute(element, "scale")
            if numeric is not None and scale_raw:
                try:
                    numeric *= Decimal(10) ** int(scale_raw)
                except (ValueError, InvalidOperation):
                    pass
            unit_ref = _attribute(element, "unitref")
            unit = _canonical_unit(unit_ref) if unit_ref else inferred_unit
            source_ref = SourceRef(
                source_type=self.source_type,
                document_id=metadata.source_document_id,
                uri=uri,
                xpath=_xpath(element),
                source_hash=parsed.raw_sha256,
            )
            decimals_raw = _attribute(element, "decimals")
            decimals: int | Literal["INF"] | None = None
            if decimals_raw:
                try:
                    decimals = int(decimals_raw)
                except ValueError:
                    if decimals_raw.upper() == "INF":
                        decimals = "INF"
            fact_id = stable_id("XF", metadata.source_document_id, source_ref.xpath, concept, context_id, raw)
            facts.append(
                StructuredFact(
                    fact_id=fact_id,
                    concept=concept,
                    value=numeric if numeric is not None else raw,
                    numeric_value=numeric,
                    unit=unit,
                    period=period,
                    dimensions=copy.deepcopy(dimensions),
                    context_id=context_id,
                    decimals=decimals,
                    available_at=metadata.available_at,
                    source_refs=[source_ref],
                )
            )
        return facts

    def extract_assets(
        self,
        parsed: ParsedHtml,
        metadata: DocumentMetadata,
        uri: str | None,
    ) -> list[DocumentAsset]:
        assets: list[DocumentAsset] = []
        for element in parsed.root.iter():
            if _tag(element) != "img":
                continue
            source_ref = SourceRef(
                source_type=self.source_type,
                document_id=metadata.source_document_id,
                uri=uri,
                xpath=_xpath(element),
                source_hash=parsed.raw_sha256,
            )
            image_uri = _attribute(element, "src")
            alt = _attribute(element, "alt")
            assets.append(
                DocumentAsset(
                    asset_id=stable_id("A", metadata.source_document_id, source_ref.xpath, image_uri, alt),
                    kind=AssetKind.IMAGE,
                    uri=image_uri,
                    alt_text=alt,
                    source_refs=[source_ref],
                )
            )
        return assets

    def convert(self, source: RawDocument) -> CanonicalDocumentBundle:
        metadata = self.extract_metadata(source)
        parsed = self.parse_structure(source)
        ast = self.build_ast(parsed, metadata, source.uri)
        facts = self.extract_facts(parsed, metadata, source.uri)
        assets = self.extract_assets(parsed, metadata, source.uri)
        records: dict[str, ProvenanceRecord] = {}
        for node in ast.walk():
            records[node.node_id] = ProvenanceRecord(
                object_id=node.node_id,
                source_refs=node.source_refs,
                transform="html_dom_to_canonical_ast",
                transform_version=PARSER_VERSION,
            )
        for fact in facts:
            records[fact.fact_id] = ProvenanceRecord(
                object_id=fact.fact_id,
                source_refs=fact.source_refs,
                derived_from_ids=fact.derived_from_ids,
                transform="inline_xbrl_extract",
                transform_version=PARSER_VERSION,
            )
        for asset in assets:
            records[asset.asset_id] = ProvenanceRecord(
                object_id=asset.asset_id,
                source_refs=asset.source_refs,
                transform="html_asset_extract",
                transform_version=PARSER_VERSION,
            )
        nodes = list(ast.walk())
        # Use the unexpanded node text for retention; canonical rowspan propagation
        # intentionally repeats values and must not inflate source coverage.
        ast_chars = sum(len(re.sub(r"\s+", "", normalize_text(node.raw_text))) for node in nodes)
        tables = [node for node in nodes if isinstance(node, TableNode)]
        numeric_count = sum(
            1
            for table in tables
            for row in table.rows
            for cell in row.cells
            if cell.numeric_value is not None and not cell.propagated
        )
        retention = ast_chars / parsed.raw_visible_chars if parsed.raw_visible_chars else None
        numeric_retention = (
            numeric_count / parsed.raw_numeric_cell_count if parsed.raw_numeric_cell_count else None
        )
        text_values = [
            normalize_text(node.normalized_text).casefold()
            for node in nodes
            if not isinstance(node, SectionNode) and normalize_text(node.normalized_text)
        ]
        duplicate_text_ratio = (
            (len(text_values) - len(set(text_values))) / len(text_values)
            if text_values
            else None
        )
        warnings: list[str] = []
        if retention is not None and retention < 0.95:
            warnings.append("visible text retention is below the 95% MVP target")
        if len(tables) != parsed.raw_table_count:
            warnings.append("canonical table count differs from raw DOM table count (nested tables may explain this)")
        if numeric_retention is not None and numeric_retention < 0.99:
            warnings.append("numeric cell retention is below the 99% MVP target")
        quality = QualityMetrics(
            raw_visible_chars=parsed.raw_visible_chars,
            ast_chars=ast_chars,
            text_retention=retention,
            raw_table_count=parsed.raw_table_count,
            ast_table_count=len(tables),
            raw_numeric_cell_count=parsed.raw_numeric_cell_count,
            numeric_cell_count=numeric_count,
            numeric_retention=numeric_retention,
            paragraph_count=sum(isinstance(node, ParagraphNode) for node in nodes),
            heading_count=sum(isinstance(node, SectionNode) for node in nodes),
            unknown_block_count=sum(isinstance(node, UnknownBlockNode) for node in nodes),
            duplicate_text_ratio=duplicate_text_ratio,
            warnings=warnings,
        )
        return CanonicalDocumentBundle(
            metadata=metadata,
            ast=ast,
            facts=facts,
            assets=assets,
            provenance=ProvenanceIndex(records=records),
            quality=quality,
        )
