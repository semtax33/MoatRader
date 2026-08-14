from __future__ import annotations

from collections import defaultdict

from pydantic import Field

from moatrader.canonical.models import ContractModel, SectionRole
from moatrader.semantic.chunker import SemanticChunk


DEFAULT_WEIGHTS: dict[SectionRole | None, float] = {
    SectionRole.BUSINESS: 0.18,
    SectionRole.COMPETITION: 0.14,
    SectionRole.CUSTOMERS: 0.10,
    SectionRole.SUPPLIERS: 0.06,
    SectionRole.PRODUCTS: 0.08,
    SectionRole.FINANCIALS: 0.16,
    SectionRole.MDA: 0.10,
    SectionRole.RISK: 0.10,
    SectionRole.GUIDANCE: 0.03,
    None: 0.05,
}

DEFAULT_MINIMUMS: dict[SectionRole | None, int] = {
    SectionRole.BUSINESS: 800,
    SectionRole.COMPETITION: 600,
    SectionRole.CUSTOMERS: 400,
    SectionRole.FINANCIALS: 1_000,
    SectionRole.RISK: 500,
}


class AllocationResult(ContractModel):
    token_budget: int = Field(ge=0)
    used_tokens: int = Field(ge=0)
    selected: list[SemanticChunk] = Field(default_factory=list)
    dropped_chunk_ids: list[str] = Field(default_factory=list)


class DynamicTokenBudgetAllocator:
    """Allocate only across roles present in a document and preserve document order."""

    def __init__(
        self,
        *,
        model_context_tokens: int,
        prompt_reserve_tokens: int,
        weights: dict[SectionRole | None, float] | None = None,
        minimums: dict[SectionRole | None, int] | None = None,
    ) -> None:
        if model_context_tokens <= prompt_reserve_tokens:
            raise ValueError("model context must exceed prompt reserve")
        self.available = model_context_tokens - prompt_reserve_tokens
        self.weights = weights or DEFAULT_WEIGHTS
        self.minimums = minimums or DEFAULT_MINIMUMS

    def allocate(
        self,
        chunks: list[SemanticChunk],
        *,
        relevance: dict[str, float] | None = None,
    ) -> AllocationResult:
        relevance = relevance or {}
        selected_ids: set[str] = set()
        used = 0
        grouped: dict[SectionRole | None, list[SemanticChunk]] = defaultdict(list)
        for chunk in chunks:
            grouped[chunk.section_role].append(chunk)

        def rank(chunk: SemanticChunk) -> float:
            signal = max(0.01, relevance.get(chunk.chunk_id, 1.0))
            role_weight = self.weights.get(chunk.section_role, self.weights.get(None, 0.01))
            return role_weight * signal / max(1.0, chunk.token_count ** 0.5)

        for role, candidates in grouped.items():
            quota = self.minimums.get(role, 0)
            role_used = 0
            for chunk in sorted(candidates, key=rank, reverse=True):
                if role_used >= quota or used + chunk.token_count > self.available:
                    continue
                selected_ids.add(chunk.chunk_id)
                role_used += chunk.token_count
                used += chunk.token_count

        remaining = [chunk for chunk in chunks if chunk.chunk_id not in selected_ids]
        for chunk in sorted(remaining, key=rank, reverse=True):
            if used + chunk.token_count <= self.available:
                selected_ids.add(chunk.chunk_id)
                used += chunk.token_count

        selected = [chunk for chunk in chunks if chunk.chunk_id in selected_ids]
        dropped = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in selected_ids]
        return AllocationResult(
            token_budget=self.available,
            used_tokens=used,
            selected=selected,
            dropped_chunk_ids=dropped,
        )

