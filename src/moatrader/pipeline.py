from __future__ import annotations

from dataclasses import dataclass

from moatrader.adapters import AdapterRegistry, DartHtmlAdapter, EdgarHtmlAdapter, IrHtmlAdapter, RawDocument
from moatrader.canonical.models import CanonicalDocumentBundle
from moatrader.context import AllocationResult, DynamicTokenBudgetAllocator
from moatrader.financial import FinancialSnapshot, FinancialSnapshotBuilder
from moatrader.llm import LLMRequest, build_evidence_request
from moatrader.render import CanonicalMarkdownRenderer
from moatrader.semantic import SemanticChunk, SemanticChunker


def default_registry() -> AdapterRegistry:
    return AdapterRegistry([DartHtmlAdapter(), EdgarHtmlAdapter(), IrHtmlAdapter()])


@dataclass(slots=True)
class PreparedDocument:
    bundle: CanonicalDocumentBundle
    structured_markdown: str
    chunks: list[SemanticChunk]
    selected_context: AllocationResult | None
    evidence_requests: list[LLMRequest]
    financial_snapshot: FinancialSnapshot


class CanonicalFinancialDocumentPipeline:
    def __init__(
        self,
        *,
        registry: AdapterRegistry | None = None,
        renderer: CanonicalMarkdownRenderer | None = None,
        chunker: SemanticChunker | None = None,
    ) -> None:
        self.registry = registry or default_registry()
        self.renderer = renderer or CanonicalMarkdownRenderer()
        self.chunker = chunker or SemanticChunker(renderer=self.renderer)
        self.snapshots = FinancialSnapshotBuilder()

    def ingest(self, source: RawDocument) -> CanonicalDocumentBundle:
        return self.registry.convert(source)

    def prepare_for_llm(
        self,
        source: RawDocument,
        *,
        model_context_tokens: int | None = None,
        prompt_reserve_tokens: int = 8_000,
    ) -> PreparedDocument:
        bundle = self.ingest(source)
        markdown = self.renderer.render_document(bundle)
        chunks = self.chunker.chunk(bundle)
        allocation = None
        if model_context_tokens is not None:
            allocation = DynamicTokenBudgetAllocator(
                model_context_tokens=model_context_tokens,
                prompt_reserve_tokens=prompt_reserve_tokens,
            ).allocate(chunks)
        requests = [build_evidence_request(chunk) for chunk in chunks]
        snapshot = self.snapshots.build([bundle], as_of=bundle.metadata.available_at)
        return PreparedDocument(
            bundle=bundle,
            structured_markdown=markdown,
            chunks=chunks,
            selected_context=allocation,
            evidence_requests=requests,
            financial_snapshot=snapshot,
        )

