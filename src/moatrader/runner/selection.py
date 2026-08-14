from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable

from moatrader.canonical.models import SectionRole
from moatrader.semantic.chunker import SemanticChunk


_ROLE_WEIGHT = {
    SectionRole.COMPETITION: 16,
    SectionRole.RISK: 15,
    SectionRole.CUSTOMERS: 14,
    SectionRole.PRODUCTS: 13,
    SectionRole.BUSINESS: 12,
    SectionRole.SUPPLIERS: 11,
    SectionRole.MDA: 10,
    SectionRole.GUIDANCE: 8,
    SectionRole.COMPANY_OVERVIEW: 6,
    SectionRole.FINANCIALS: 4,
    SectionRole.NOTES: 3,
    SectionRole.GOVERNANCE: 0,
    SectionRole.OTHER: 1,
}

_POSITIVE_TERMS = re.compile(
    r"시장점유|점유율|경쟁|경쟁력|고객|거래처|수주|계약|재계약|브랜드|특허|지식재산|"
    r"연구개발|R&D|인허가|허가|독점|진입장벽|원가|규모의 경제|네트워크|전환비용|"
    r"가격|단가|마진|영업이익|반복매출|구독|유지율|해외시장|market share|competition|"
    r"customer|contract|patent|license|switching|network effect|pricing|margin|retention",
    re.IGNORECASE,
)
_NEGATIVE_TERMS = re.compile(
    r"위험|리스크|의존|집중|대체|경쟁심화|가격하락|원재료|규제|소송|분쟁|적자|손상|"
    r"중단|감소|불확실|risk|dependen|concentrat|substitut|litigation|decline|uncertain",
    re.IGNORECASE,
)
_LOW_VALUE_TERMS = re.compile(
    r"임원.{0,8}보수|이사회|감사제도|주주총회|자본금 변동|주식의 총수|정관|"
    r"executive compensation|board of directors",
    re.IGNORECASE,
)


def evidence_chunk_score(chunk: SemanticChunk) -> tuple[int, int, int]:
    text = f"{' '.join(chunk.section_path)}\n{chunk.markdown}"
    role_weight = _ROLE_WEIGHT.get(chunk.section_role or SectionRole.OTHER, 1)
    keyword_score = min(20, len(_POSITIVE_TERMS.findall(text)) + 2 * len(_NEGATIVE_TERMS.findall(text)))
    penalty = 12 if _LOW_VALUE_TERMS.search(text) else 0
    return role_weight + keyword_score - penalty, keyword_score, -chunk.token_count


def select_evidence_chunks(chunks: list[SemanticChunk], maximum: int | None) -> list[SemanticChunk]:
    if maximum is None or len(chunks) <= maximum:
        return list(chunks)
    if maximum <= 0:
        raise ValueError("maximum evidence chunks must be positive")

    ranked = sorted(
        enumerate(chunks),
        key=lambda item: (*evidence_chunk_score(item[1]), -item[0]),
        reverse=True,
    )
    selected: list[tuple[int, SemanticChunk]] = []
    used_sections: defaultdict[tuple[str, ...], int] = defaultdict(int)

    def reserve(predicate: Callable[[SemanticChunk], bool]) -> None:
        if len(selected) >= maximum:
            return
        for index, chunk in ranked:
            if chunk.chunk_id in {item.chunk_id for _position, item in selected}:
                continue
            if predicate(chunk):
                selected.append((index, chunk))
                used_sections[tuple(chunk.section_path)] += 1
                return

    # A scorer needs the actual business model before it can interpret isolated
    # patent, contract, risk, or financial-note disclosures.
    reserve(
        lambda chunk: chunk.section_role == SectionRole.BUSINESS
        and re.search(r"사업의 개요|사업 개요|business overview", " ".join(chunk.section_path), re.IGNORECASE)
    )
    reserve(lambda chunk: chunk.section_role == SectionRole.PRODUCTS)
    reserve(lambda chunk: bool(_NEGATIVE_TERMS.search(f"{' '.join(chunk.section_path)}\n{chunk.markdown}")))

    # First pass maximizes section diversity so a long financial-note section
    # cannot crowd out business, customer, competition, and risk disclosures.
    for index, chunk in ranked:
        if len(selected) >= maximum:
            break
        if chunk.chunk_id in {item.chunk_id for _position, item in selected}:
            continue
        path = tuple(chunk.section_path)
        if used_sections[path] >= 1:
            continue
        selected.append((index, chunk))
        used_sections[path] += 1
        if len(selected) == maximum:
            break

    if len(selected) < maximum:
        selected_ids = {chunk.chunk_id for _index, chunk in selected}
        for index, chunk in ranked:
            if chunk.chunk_id in selected_ids:
                continue
            selected.append((index, chunk))
            selected_ids.add(chunk.chunk_id)
            if len(selected) == maximum:
                break

    return [chunk for _index, chunk in sorted(selected, key=lambda item: item[0])]


def batch_evidence_chunks(chunks: list[SemanticChunk], maximum_tokens: int) -> list[list[SemanticChunk]]:
    if maximum_tokens <= 0:
        raise ValueError("evidence batch token limit must be positive")
    batches: list[list[SemanticChunk]] = []
    current: list[SemanticChunk] = []
    current_tokens = 0
    current_document: str | None = None
    for chunk in chunks:
        document_changed = current_document is not None and chunk.document_id != current_document
        would_overflow = current and current_tokens + chunk.token_count > maximum_tokens
        if document_changed or would_overflow:
            batches.append(current)
            current = []
            current_tokens = 0
        current.append(chunk)
        current_tokens += chunk.token_count
        current_document = chunk.document_id
    if current:
        batches.append(current)
    return batches
