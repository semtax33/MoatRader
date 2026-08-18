from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable

import fitz

from moatrader.evidence.historical_overlay import HistoricalExcerpt, HistoricalSourceRole


_SPACE = re.compile(r"[ \t\u00a0]+")
_PARAGRAPH = re.compile(r"\n\s*\n+")
_MARKET_OPINION = re.compile(
    r"(?:투자의견|목표주가|적정주가|현재주가|상승여력|target\s*price|\b(?:buy|hold|sell)\b|"
    r"매수(?:\s*\(|\s*$)|중립(?:\s*\(|\s*$)|매도(?:\s*\(|\s*$)|\bPER\b|\bPBR\b)",
    flags=re.IGNORECASE,
)
_EVIDENCE_TERMS = (
    "가격", "판가", "인상", "인하", "마진", "수익성", "원가", "비용", "점유율",
    "경쟁", "진입", "특허", "허가", "인증", "독점", "네트워크", "플랫폼", "규모",
    "전환", "교체", "재구매", "갱신", "해지", "고객", "공급", "계약", "반복", "유지",
    "집중", "의존", "규제", "소송", "위험", "감소", "하락", "악화", "회복", "증가",
    "pricing", "switch", "retention", "renewal", "churn", "barrier", "competition", "risk",
)


def opaque_unit_id(source_id: str, text: str) -> str:
    digest = hashlib.sha256(f"{source_id}\n{text}".encode("utf-8")).hexdigest()
    return f"U_{digest[:20]}"


def quarantine_market_opinion(text: str) -> tuple[str, int]:
    """Remove market-price/recommendation lines before any LLM can see them."""

    kept: list[str] = []
    removed = 0
    for raw in text.splitlines():
        line = _SPACE.sub(" ", raw).strip()
        if not line:
            kept.append("")
        elif _MARKET_OPINION.search(line):
            removed += 1
        else:
            kept.append(line)
    return "\n".join(kept), removed


def pdf_text(path: str | Path) -> str:
    source = Path(path).resolve()
    with fitz.open(source) as document:
        return "\n\n".join(page.get_text("text", sort=True) for page in document)


def _bounded_blocks(text: str, *, maximum_chars: int = 1200) -> list[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    paragraphs = [_SPACE.sub(" ", item).strip() for item in _PARAGRAPH.split(normalized)]
    blocks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) < 40:
            continue
        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + maximum_chars)
            if end < len(paragraph):
                split = max(paragraph.rfind(". ", start, end), paragraph.rfind("다. ", start, end))
                if split > start + 200:
                    end = split + 2
            block = paragraph[start:end].strip()
            if len(block) >= 40:
                blocks.append(block)
            start = end
    return blocks


def _evidence_score(text: str) -> tuple[int, int]:
    lowered = text.casefold()
    term_hits = sum(lowered.count(term.casefold()) for term in _EVIDENCE_TERMS)
    numeric = min(8, len(re.findall(r"\d+(?:[.,]\d+)?\s*%", text)))
    realized = sum(lowered.count(term) for term in ("실적", "기록", "달성", "감소", "증가", "하락"))
    return term_hits * 3 + numeric * 2 + realized, len(text)


def excerpts_from_text(
    *,
    source_id: str,
    source_role: HistoricalSourceRole,
    available_at: datetime,
    text: str,
    maximum: int = 24,
    quarantine_opinion: bool = False,
) -> tuple[list[HistoricalExcerpt], int]:
    removed = 0
    if quarantine_opinion:
        text, removed = quarantine_market_opinion(text)
    candidates = _bounded_blocks(text)
    ranked = sorted(enumerate(candidates), key=lambda item: (_evidence_score(item[1]), -item[0]), reverse=True)
    selected = sorted(ranked[:maximum], key=lambda item: item[0])
    excerpts = [
        HistoricalExcerpt.create(
            unit_id=opaque_unit_id(source_id, value),
            source_id=source_id,
            source_role=source_role,
            available_at=available_at,
            text=value,
        )
        for index, value in selected
    ]
    return excerpts, removed


def dart_original_excerpts(
    filing_dir: str | Path,
    *,
    maximum: int = 30,
) -> list[HistoricalExcerpt]:
    root = Path(filing_dir).resolve()
    index = json.loads((root / "evidence-index.json").read_text(encoding="utf-8-sig"))
    excerpts: list[HistoricalExcerpt] = []
    for item in index:
        text_path = root / str(item["text_file"])
        text = text_path.read_text(encoding="utf-8-sig")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != item["text_sha256"]:
            raise ValueError(f"DART evidence text hash mismatch: {text_path}")
        available_at = datetime.fromisoformat(item["available_at"])
        found, _ = excerpts_from_text(
            source_id=item["source_id"],
            source_role=HistoricalSourceRole.DART_ORIGINAL,
            available_at=available_at,
            text=text,
            maximum=maximum,
        )
        excerpts.extend(found)
    return sorted(excerpts, key=lambda item: _evidence_score(item.text), reverse=True)[:maximum]


def pdf_excerpts(
    path: str | Path,
    *,
    source_id: str,
    source_role: HistoricalSourceRole,
    available_at: datetime,
    maximum: int = 24,
) -> tuple[list[HistoricalExcerpt], int]:
    return excerpts_from_text(
        source_id=source_id,
        source_role=source_role,
        available_at=available_at,
        text=pdf_text(path),
        maximum=maximum,
        quarantine_opinion=source_role in {
            HistoricalSourceRole.COMPANY_ANALYST,
            HistoricalSourceRole.INDUSTRY_ANALYST,
        },
    )


def unique_excerpts(groups: Iterable[Iterable[HistoricalExcerpt]]) -> list[HistoricalExcerpt]:
    result: dict[str, HistoricalExcerpt] = {}
    for group in groups:
        for item in group:
            prior = result.get(item.unit_id)
            if prior is not None and prior != item:
                raise ValueError(f"conflicting historical excerpt: {item.unit_id}")
            result[item.unit_id] = item
    return list(result.values())
