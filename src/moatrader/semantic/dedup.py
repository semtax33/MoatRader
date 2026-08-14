from __future__ import annotations

import re
from difflib import SequenceMatcher

from pydantic import Field

from moatrader.canonical.ids import normalize_text
from moatrader.canonical.models import ContractModel
from moatrader.semantic.chunker import SemanticChunk


_NUMBER_RE = re.compile(r"[-+]?\d[\d,.]*(?:%|개월|년|월|일)?")


class ChunkDuplicate(ContractModel):
    duplicate_chunk_id: str
    canonical_chunk_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    kind: str


class ChunkChange(ContractModel):
    older_chunk_id: str
    newer_chunk_id: str
    similarity: float = Field(ge=0.0, le=1.0)
    older_numbers: list[str]
    newer_numbers: list[str]


class ChunkDeduplicationResult(ContractModel):
    kept: list[SemanticChunk]
    duplicates: list[ChunkDuplicate] = Field(default_factory=list)
    changes: list[ChunkChange] = Field(default_factory=list)


def _comparison_text(chunk: SemanticChunk) -> str:
    return re.sub(r"\s+", " ", normalize_text(chunk.markdown)).strip().casefold()


def deduplicate_chunks(
    chunks: list[SemanticChunk],
    *,
    near_duplicate_threshold: float = 0.94,
    change_candidate_threshold: float = 0.80,
) -> ChunkDeduplicationResult:
    """Keep input order; callers should pass newest filings first."""
    kept: list[SemanticChunk] = []
    kept_text: list[str] = []
    duplicates: list[ChunkDuplicate] = []
    changes: list[ChunkChange] = []
    for chunk in chunks:
        text = _comparison_text(chunk)
        numbers = _NUMBER_RE.findall(text)
        matched = False
        for canonical, candidate_text in zip(kept, kept_text, strict=True):
            if chunk.section_role != canonical.section_role:
                continue
            similarity = 1.0 if text == candidate_text else SequenceMatcher(None, text, candidate_text).ratio()
            candidate_numbers = _NUMBER_RE.findall(candidate_text)
            if similarity >= near_duplicate_threshold and numbers == candidate_numbers:
                duplicates.append(
                    ChunkDuplicate(
                        duplicate_chunk_id=chunk.chunk_id,
                        canonical_chunk_id=canonical.chunk_id,
                        similarity=similarity,
                        kind="EXACT" if similarity == 1.0 else "NEAR",
                    )
                )
                matched = True
                break
            if similarity >= change_candidate_threshold and numbers != candidate_numbers:
                changes.append(
                    ChunkChange(
                        older_chunk_id=chunk.chunk_id,
                        newer_chunk_id=canonical.chunk_id,
                        similarity=similarity,
                        older_numbers=numbers,
                        newer_numbers=candidate_numbers,
                    )
                )
        if not matched:
            kept.append(chunk)
            kept_text.append(text)
    return ChunkDeduplicationResult(kept=kept, duplicates=duplicates, changes=changes)

