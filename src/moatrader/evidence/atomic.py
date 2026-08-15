from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict

from moatrader.canonical.ids import stable_id
from moatrader.canonical.models import SectionRole, SourceRef, SourceType
from moatrader.semantic.chunker import HeuristicTokenCounter, SemanticChunk


ATOMIC_SEGMENTATION_VERSION = "atomic-evidence/2"
ATOMIC_RUBRIC_VERSION = "structural-moat-rubric/3"

_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HTML_BREAK_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+")
_SPACE_RE = re.compile(r"\s+")
_NON_CONTENT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
_MOAT_TERMS_RE = re.compile(
    r"고객|거래처|계약|점유율|브랜드|특허|인허가|독점|진입\s*장벽|연구개발|"
    r"규모의\s*경제|네트워크|전환\s*비용|가격\s*결정|유지율|의존|집중|대체|경쟁|규제|"
    r"customer|contract|patent|licen[cs]e|switching|network\s*effect|retention|churn|"
    r"competition|dependen|concentrat|substitut|regulat",
    re.IGNORECASE,
)
_ROLE_WEIGHT = {
    SectionRole.COMPETITION: 16,
    SectionRole.RISK: 15,
    SectionRole.CUSTOMERS: 14,
    SectionRole.PRODUCTS: 13,
    SectionRole.BUSINESS: 12,
    SectionRole.SUPPLIERS: 11,
    SectionRole.MDA: 8,
    SectionRole.GUIDANCE: 6,
    SectionRole.COMPANY_OVERVIEW: 5,
    SectionRole.NOTES: 3,
    SectionRole.FINANCIALS: 2,
    SectionRole.GOVERNANCE: 0,
    SectionRole.OTHER: 1,
}


def normalize_atomic_text(value: str) -> str:
    """Canonical text used by classification keys and whitespace metamorphs."""

    text = unicodedata.normalize("NFKC", value)
    text = _HTML_COMMENT_RE.sub(" ", text)
    text = _HTML_BREAK_RE.sub(" ", text)
    text = _HEADING_RE.sub("", text)
    text = _LIST_PREFIX_RE.sub("", text)
    text = text.replace("\u00a0", " ")
    return _SPACE_RE.sub(" ", text).strip()


def _bounded_fragments(value: str, maximum_words: int = 180) -> list[str]:
    words = value.split()
    if len(words) <= maximum_words:
        return [value]
    return [" ".join(words[index : index + maximum_words]) for index in range(0, len(words), maximum_words)]


def _prose_fragments(value: str) -> list[str]:
    """Return sentence-bounded units that are stable when parsed again."""

    normalized_value = normalize_atomic_text(value)
    fragments: list[str] = []
    for sentence in _SENTENCE_BOUNDARY_RE.split(normalized_value):
        normalized_sentence = normalize_atomic_text(sentence)
        if normalized_sentence:
            fragments.extend(_bounded_fragments(normalized_sentence))
    return fragments


def _table_fragments(cells: list[str], maximum_words: int = 180) -> list[str]:
    """Keep ordinary rows intact and canonically split narrative table cells.

    DART commonly stores entire note paragraphs in a one-cell HTML table with
    ``<br>`` separators.  Treating those paragraphs as one table row made the
    first segmentation differ from a rebuild of the same atomic text.  Short
    relational rows remain rows; long narrative rows are atomized per cell so
    every emitted unit is safe to feed back through the splitter.
    """

    content_cells = [normalize_atomic_text(cell) for cell in cells]
    content_cells = [cell for cell in content_cells if cell]
    if not content_cells:
        return []
    row = " | ".join(content_cells)
    if len(content_cells) > 1 and len(row.split()) <= maximum_words:
        return [row]
    return [
        fragment
        for cell in content_cells
        for fragment in _prose_fragments(cell)
    ]


def split_atomic_evidence_text(markdown: str) -> list[str]:
    """Split source markdown into order-independent sentence/row units.

    Headings and presentation markup are context, not evidence.  Tables are
    split by row; prose is split by deterministic punctuation boundaries.
    Exact duplicate units are collapsed and the result is content-sorted.
    """

    cleaned = _HTML_COMMENT_RE.sub("\n", unicodedata.normalize("NFKC", markdown))
    candidates: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        value = " ".join(paragraph)
        paragraph.clear()
        candidates.extend(_prose_fragments(value))

    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        if _HEADING_RE.match(line) or _TABLE_SEPARATOR_RE.match(line):
            flush_paragraph()
            continue
        if "|" in line:
            flush_paragraph()
            candidates.extend(_table_fragments(line.strip("|").split("|")))
            continue
        paragraph.append(_LIST_PREFIX_RE.sub("", line))
    flush_paragraph()

    unique = {
        normalized
        for candidate in candidates
        if (normalized := normalize_atomic_text(candidate))
        and len(normalized) >= 8
        and not _NON_CONTENT_RE.fullmatch(normalized)
    }
    return sorted(unique, key=lambda item: (item.casefold(), item))


def is_generated_summary_chunk(chunk: SemanticChunk) -> bool:
    if chunk.metadata.get("generated_summary") is True:
        return True
    source_types = {ref.source_type for ref in chunk.source_refs}
    return SourceType.GENERATED_SUMMARY in source_types


def _source_type(chunk: SemanticChunk) -> SourceType:
    source_types = sorted({ref.source_type for ref in chunk.source_refs}, key=lambda item: item.value)
    return source_types[0] if source_types else SourceType.OTHER


def atomic_classification_key(
    *,
    issuer_id: str | None,
    text: str,
    source_type: SourceType,
    section_role: SectionRole | None,
) -> str:
    return stable_id(
        "AEK",
        ATOMIC_SEGMENTATION_VERSION,
        ATOMIC_RUBRIC_VERSION,
        issuer_id or "UNKNOWN_ISSUER",
        source_type.value,
        (section_role or SectionRole.OTHER).value,
        normalize_atomic_text(text).casefold(),
    )


def build_atomic_evidence_units(
    chunks: list[SemanticChunk],
    *,
    issuer_id: str | None,
) -> list[SemanticChunk]:
    """Create a canonical set of independently classifiable evidence units."""

    grouped: dict[tuple[str, str], list[tuple[SemanticChunk, str]]] = defaultdict(list)
    for chunk in chunks:
        if is_generated_summary_chunk(chunk):
            continue
        source_type = _source_type(chunk)
        for text in split_atomic_evidence_text(chunk.markdown):
            key = atomic_classification_key(
                issuer_id=issuer_id,
                text=text,
                source_type=source_type,
                section_role=chunk.section_role,
            )
            # Keep separate source documents for audit validation while using
            # the document-independent key for judgment replay.
            grouped[(key, chunk.document_id)].append((chunk, text))

    counter = HeuristicTokenCounter()
    units: list[SemanticChunk] = []
    for (classification_key, document_id), origins in sorted(grouped.items()):
        origin_chunks = [item[0] for item in origins]
        text = min((item[1] for item in origins), key=lambda item: (item.casefold(), item))
        canonical_origin = min(
            origin_chunks,
            key=lambda item: (
                (item.section_role or SectionRole.OTHER).value,
                tuple(item.section_path),
                item.chunk_id,
            ),
        )
        node_ids = sorted({node_id for chunk in origin_chunks for node_id in chunk.node_ids})
        refs_by_json: dict[str, SourceRef] = {}
        for chunk in origin_chunks:
            for ref in chunk.source_refs:
                refs_by_json[ref.model_dump_json()] = ref
        units.append(
            SemanticChunk(
                chunk_id=stable_id("AU", classification_key, document_id),
                document_id=document_id,
                section_path=list(canonical_origin.section_path),
                section_role=canonical_origin.section_role,
                node_ids=node_ids,
                chunk_type="atomic_evidence",
                markdown=text,
                token_count=counter.count(text),
                source_refs=[refs_by_json[key] for key in sorted(refs_by_json)],
                metadata={
                    "atomic_evidence_key": classification_key,
                    "atomic_segmentation_version": ATOMIC_SEGMENTATION_VERSION,
                    "atomic_rubric_version": ATOMIC_RUBRIC_VERSION,
                    "origin_chunk_ids": sorted({chunk.chunk_id for chunk in origin_chunks}),
                    "source_type": _source_type(canonical_origin).value,
                },
            )
        )
    return units


def select_atomic_evidence_units(
    units: list[SemanticChunk],
    maximum: int | None,
) -> list[SemanticChunk]:
    """Content-ranked set selection with no presentation-order tie breaker."""

    unique: dict[str, SemanticChunk] = {}
    for unit in units:
        key = str(unit.metadata["atomic_evidence_key"])
        current = unique.get(key)
        if current is None or (unit.document_id, unit.chunk_id) < (current.document_id, current.chunk_id):
            unique[key] = unit
    def relevance(unit: SemanticChunk) -> tuple[int, int]:
        keyword_count = min(20, len(_MOAT_TERMS_RE.findall(unit.markdown)))
        return _ROLE_WEIGHT.get(unit.section_role or SectionRole.OTHER, 1) + keyword_count, keyword_count

    eligible = [
        unit
        for unit in unique.values()
        if relevance(unit)[1] > 0
        or unit.section_role
        in {
            SectionRole.BUSINESS,
            SectionRole.PRODUCTS,
            SectionRole.CUSTOMERS,
            SectionRole.COMPETITION,
            SectionRole.RISK,
        }
    ]
    ranked = sorted(
        eligible,
        key=lambda unit: (
            -relevance(unit)[0],
            -relevance(unit)[1],
            unit.token_count,
            str(unit.metadata["atomic_evidence_key"]),
        ),
    )
    selected = ranked if maximum is None else ranked[:maximum]
    return sorted(selected, key=lambda unit: str(unit.metadata["atomic_evidence_key"]))


def atomic_unit_set_sha256(units: list[SemanticChunk]) -> str:
    keys = sorted({str(unit.metadata["atomic_evidence_key"]) for unit in units})
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()
