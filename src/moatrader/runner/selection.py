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
    SectionRole.FINANCIALS: 2,
    SectionRole.NOTES: 3,
    SectionRole.GOVERNANCE: 0,
    SectionRole.OTHER: 1,
}

_POSITIVE_TERMS = re.compile(
    r"시장\s*점유|경쟁\s*우위|고객|거래처|수주|계약|점유율|브랜드|특허|인허가|독점|진입\s*장벽|"
    r"연구개발|R&D|규모의\s*경제|네트워크|전환\s*비용|가격\s*결정|마진|영업이익|반복\s*매출|구독|"
    r"market\s*share|competition|customer|contract|patent|license|switching|network\s*effect|pricing|margin|retention",
    re.IGNORECASE,
)
_NEGATIVE_TERMS = re.compile(
    r"위험|리스크|의존|집중|대체|경쟁\s*심화|가격\s*하락|원재료|규제|소송|분쟁|적자|손상|중단|감소|불확실|"
    r"risk|dependen|concentrat|substitut|litigation|decline|uncertain",
    re.IGNORECASE,
)
_LOW_VALUE_TERMS = re.compile(
    r"임원.{0,8}보수|이사의\s*보수|주주총회|자본금\s*변동|주식의\s*총수|정관|"
    r"executive\s*compensation|board\s*of\s*directors",
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
    selected_ids: set[str] = set()
    used_sections: defaultdict[tuple[str, ...], int] = defaultdict(int)

    def reserve(predicate: Callable[[SemanticChunk], bool]) -> None:
        if len(selected) >= maximum:
            return
        for index, chunk in ranked:
            if chunk.chunk_id in selected_ids:
                continue
            if predicate(chunk):
                selected.append((index, chunk))
                selected_ids.add(chunk.chunk_id)
                used_sections[tuple(chunk.section_path)] += 1
                return

    # Preserve enough business context to interpret isolated patent, contract,
    # risk, or customer disclosures while guaranteeing counterevidence.
    reserve(
        lambda chunk: chunk.section_role == SectionRole.BUSINESS
        and bool(
            re.search(
                r"사업의\s*내용|사업\s*개요|business\s+overview",
                " ".join(chunk.section_path),
                re.IGNORECASE,
            )
        )
    )
    reserve(lambda chunk: chunk.section_role == SectionRole.PRODUCTS)
    reserve(lambda chunk: bool(_NEGATIVE_TERMS.search(f"{' '.join(chunk.section_path)}\n{chunk.markdown}")))

    # First pass maximizes section diversity so a long financial-note section
    # cannot crowd out business, customer, competition, and risk disclosures.
    for index, chunk in ranked:
        if len(selected) >= maximum:
            break
        if chunk.chunk_id in selected_ids:
            continue
        path = tuple(chunk.section_path)
        if used_sections[path] >= 1:
            continue
        selected.append((index, chunk))
        selected_ids.add(chunk.chunk_id)
        used_sections[path] += 1

    for index, chunk in ranked:
        if len(selected) >= maximum:
            break
        if chunk.chunk_id in selected_ids:
            continue
        selected.append((index, chunk))
        selected_ids.add(chunk.chunk_id)

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
        would_overflow = bool(current) and current_tokens + chunk.token_count > maximum_tokens
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
