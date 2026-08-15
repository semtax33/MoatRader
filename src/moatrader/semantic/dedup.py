from __future__ import annotations

import re
from difflib import SequenceMatcher

from pydantic import Field

from moatrader.canonical.ids import normalize_text
from moatrader.canonical.models import ContractModel
from moatrader.semantic.chunker import SemanticChunk


_NUMBER_RE = re.compile(r"[-+]?\d[\d,.]*(?:%|개월|년|월|일)?")
_TOKEN_RE = re.compile(r"[0-9A-Za-z_가-힣]+|[^\s\w]", re.UNICODE)
_SHINGLE_SIZE = 3
_MIN_SHINGLES_FOR_BLOCKING = 20
_MIN_SHINGLE_DICE = 0.45


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


def _comparison_shingles(text: str) -> frozenset[tuple[str, ...]]:
    tokens = _TOKEN_RE.findall(text)
    if len(tokens) < _SHINGLE_SIZE:
        return frozenset({tuple(tokens)}) if tokens else frozenset()
    return frozenset(
        tuple(tokens[index : index + _SHINGLE_SIZE])
        for index in range(len(tokens) - _SHINGLE_SIZE + 1)
    )


def deduplicate_chunks(
    chunks: list[SemanticChunk],
    *,
    near_duplicate_threshold: float = 0.94,
    change_candidate_threshold: float = 0.80,
) -> ChunkDeduplicationResult:
    """Keep input order; callers should pass newest filings first."""
    kept: list[SemanticChunk] = []
    kept_text: list[str] = []
    kept_numbers: list[list[str]] = []
    kept_shingles: list[frozenset[tuple[str, ...]]] = []
    duplicates: list[ChunkDuplicate] = []
    changes: list[ChunkChange] = []
    for chunk in chunks:
        text = _comparison_text(chunk)
        numbers = _NUMBER_RE.findall(text)
        shingles = _comparison_shingles(text)
        matched = False
        for canonical, candidate_text, candidate_numbers, candidate_shingles in zip(
            kept,
            kept_text,
            kept_numbers,
            kept_shingles,
            strict=True,
        ):
            if chunk.section_role != canonical.section_role:
                continue
            if text == candidate_text:
                similarity = 1.0
            else:
                if min(len(shingles), len(candidate_shingles)) >= _MIN_SHINGLES_FOR_BLOCKING:
                    common = len(shingles & candidate_shingles)
                    dice = (2 * common) / (len(shingles) + len(candidate_shingles))
                    if dice < _MIN_SHINGLE_DICE:
                        continue
                matcher = SequenceMatcher(None, text, candidate_text)
                # Both methods are documented upper bounds on ratio(). Skipping
                # below the lower change threshold preserves the old matching
                # semantics while avoiding quadratic alignment for unrelated
                # long financial-table chunks.
                if matcher.real_quick_ratio() < change_candidate_threshold:
                    continue
                if matcher.quick_ratio() < change_candidate_threshold:
                    continue
                similarity = matcher.ratio()
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
            kept_numbers.append(numbers)
            kept_shingles.append(shingles)
    return ChunkDeduplicationResult(kept=kept, duplicates=duplicates, changes=changes)
