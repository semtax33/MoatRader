from __future__ import annotations

from dataclasses import dataclass

from moatrader.adapters import (
    AdapterRegistry,
    DartHtmlAdapter,
    EdgarHtmlAdapter,
    IndustryPdfAdapter,
    IrHtmlAdapter,
    IrPdfAdapter,
    PdfOcrAdapter,
    RawDocument,
)
from moatrader.canonical.models import CanonicalDocumentBundle, SourceType
from moatrader.context import AllocationResult, DynamicTokenBudgetAllocator
from moatrader.evidence import build_atomic_evidence_units, select_valuation_evidence_units
from moatrader.financial import FinancialSnapshot, FinancialSnapshotBuilder
from moatrader.llm import LLMRequest, build_evidence_request, build_valuation_driver_request
from moatrader.render import CanonicalMarkdownRenderer
from moatrader.semantic import SemanticChunk, SemanticChunker


def default_registry(
    *,
    ir_ocr_adapter: PdfOcrAdapter | None = None,
    synalyst_root: str | None = None,
) -> AdapterRegistry:
    return AdapterRegistry(
        [
            DartHtmlAdapter(),
            EdgarHtmlAdapter(),
            IrHtmlAdapter(),
            IrPdfAdapter(ocr_adapter=ir_ocr_adapter),
            IndustryPdfAdapter(synalyst_root=synalyst_root),
        ]
    )


@dataclass(slots=True)
class PreparedDocument:
    bundle: CanonicalDocumentBundle
    structured_markdown: str
    chunks: list[SemanticChunk]
    selected_context: AllocationResult | None
    evidence_requests: list[LLMRequest]
    valuation_evidence_units: list[SemanticChunk]
    valuation_evidence_requests: list[LLMRequest]
    financial_snapshot: FinancialSnapshot


class CanonicalFinancialDocumentPipeline:
    def __init__(
        self,
        *,
        registry: AdapterRegistry | None = None,
        renderer: CanonicalMarkdownRenderer | None = None,
        chunker: SemanticChunker | None = None,
        ir_ocr_adapter: PdfOcrAdapter | None = None,
        synalyst_root: str | None = None,
    ) -> None:
        if registry is not None and (
            ir_ocr_adapter is not None or synalyst_root is not None
        ):
            raise ValueError(
                "adapter-specific configuration cannot be combined with an explicit registry"
            )
        self.registry = registry or default_registry(
            ir_ocr_adapter=ir_ocr_adapter,
            synalyst_root=synalyst_root,
        )
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
        maximum_valuation_units: int | None = None,
    ) -> PreparedDocument:
        if maximum_valuation_units is not None and maximum_valuation_units < 1:
            raise ValueError("maximum_valuation_units must be positive when supplied")
        bundle = self.ingest(source)
        markdown = self.renderer.render_document(bundle)
        chunks = self.chunker.chunk(bundle)
        allocation = None
        if model_context_tokens is not None:
            allocation = DynamicTokenBudgetAllocator(
                model_context_tokens=model_context_tokens,
                prompt_reserve_tokens=prompt_reserve_tokens,
            ).allocate(chunks)
        # Industry material is reference-class/valuation context. It must never
        # enter the issuer MOAT lane as if an industry-wide fact were a company
        # mechanism or outcome.
        requests = (
            []
            if bundle.metadata.source_type == SourceType.INDUSTRY
            else [build_evidence_request(chunk) for chunk in chunks]
        )
        atomic_units = build_atomic_evidence_units(
            chunks,
            issuer_id=bundle.metadata.issuer_id,
        )
        valuation_units = select_valuation_evidence_units(
            atomic_units,
            maximum=maximum_valuation_units,
        )
        valuation_requests = [
            build_valuation_driver_request(
                unit,
                issuer_id=bundle.metadata.issuer_id,
                issuer_name=bundle.metadata.issuer_name,
            )
            for unit in valuation_units
        ]
        snapshot = self.snapshots.build([bundle], as_of=bundle.metadata.available_at)
        return PreparedDocument(
            bundle=bundle,
            structured_markdown=markdown,
            chunks=chunks,
            selected_context=allocation,
            evidence_requests=requests,
            valuation_evidence_units=valuation_units,
            valuation_evidence_requests=valuation_requests,
            financial_snapshot=snapshot,
        )
