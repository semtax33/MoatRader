from __future__ import annotations

import hashlib
import inspect
import json
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from moatrader.adapters import RawDocument
from moatrader.audit import RunManifest
from moatrader.canonical.ids import stable_id
from moatrader.canonical.models import CanonicalDocumentBundle, SectionNode, TableNode
from moatrader.context import DynamicTokenBudgetAllocator, EvidencePackBuilder
from moatrader.evidence.models import (
    CompanyDossier,
    CoverageMetrics,
    EvidenceBatchExtractionResult,
    EvidenceCard,
    EvidenceDirection,
    EvidenceExtractionResult,
    EvidenceRelation,
    MoatScore,
    OUTCOME_CORROBORATION_TYPES,
    STRUCTURAL_MOAT_TYPES,
    SectionSummary,
)
from moatrader.evidence.validation import (
    derive_moat_score,
    validate_evidence_batch_result,
    validate_evidence_result,
    validate_moat_score,
)
from moatrader.evidence.processing import (
    build_evidence_relations,
    build_forward_driver_cards,
    calibrate_card_reliability,
    cluster_duplicate_evidence,
    normalize_card_semantics,
)
from moatrader.financial import (
    DcfAssumptions,
    DcfAssumptionType,
    DcfEngine,
    FinancialSnapshot,
    FinancialSnapshotBuilder,
)
from moatrader.llm import (
    LLMRequest,
    LLMTransport,
    build_evidence_batch_request,
    build_evidence_request,
    build_moat_pack_request,
    build_section_summary_request,
)
from moatrader.llm.transport import TransportResult, TransportUsage
from moatrader.pipeline import CanonicalFinancialDocumentPipeline
from moatrader.quality import ParserQualityGateConfig, assess_parser_quality
from moatrader.retrieval import EvidenceRetriever, RetrievalResult
from moatrader.runstore import RunStore
from moatrader.runner.models import (
    CompanyRunResult,
    CompanyRunStatus,
    LLMCallAudit,
    UniverseRunConfig,
    UniverseRunResult,
)
from moatrader.runner.report import rank_run_result, ranking_csv, results_csv
from moatrader.runner.selection import batch_evidence_chunks, select_evidence_chunks
from moatrader.semantic import SemanticChunk, deduplicate_chunks
from moatrader.universe import CompanyInput, UniverseManifest


ResponseT = TypeVar("ResponseT", bound=BaseModel)
RUNNER_VERSION = "0.5.1"


class MoatUniverseRunner:
    def __init__(
        self,
        *,
        config: UniverseRunConfig,
        output_directory: str | Path,
        transport: LLMTransport | None,
        pipeline: CanonicalFinancialDocumentPipeline | None = None,
    ) -> None:
        if transport is None and not config.dry_run:
            raise ValueError("an LLM transport is required unless dry_run=true")
        self.config = config
        self.transport = transport
        self.pipeline = pipeline or CanonicalFinancialDocumentPipeline()
        self.store = RunStore(Path(output_directory) / config.run_id)
        self.snapshots = FinancialSnapshotBuilder()
        self.retriever = EvidenceRetriever()
        self.pack_builder = EvidencePackBuilder()

    def run(self, manifest: UniverseManifest, companies: list[CompanyInput]) -> UniverseRunResult:
        started = datetime.now(timezone.utc)
        self.store.write_json(
            self.store.root / "run-config.json",
            {
                **self.config.model_dump(mode="json"),
                "manifest": manifest.path,
                "tickers": [company.ticker for company in companies],
            },
        )
        results: list[CompanyRunResult] = []
        if self.config.workers == 1:
            for company in companies:
                results.append(self._safe_run_company(company))
        else:
            with ThreadPoolExecutor(max_workers=self.config.workers, thread_name_prefix="moat") as executor:
                futures = {executor.submit(self._safe_run_company, company): company.ticker for company in companies}
                for future in as_completed(futures):
                    results.append(future.result())
        order = {company.ticker: index for index, company in enumerate(companies)}
        results.sort(key=lambda result: order[result.ticker])
        unranked = UniverseRunResult(
            run_id=self.config.run_id,
            as_of=self.config.as_of,
            started_at=started,
            completed_at=datetime.now(timezone.utc),
            companies=results,
        )
        final = unranked.model_copy(update={"ranking": rank_run_result(unranked)})
        self.store.write_json(self.store.root / "run-result.json", final)
        self.store.write_text(self.store.root / "results.csv", results_csv(final))
        self.store.write_text(self.store.root / "ranking.csv", ranking_csv(final))
        return final

    def _safe_run_company(self, company: CompanyInput) -> CompanyRunResult:
        try:
            return self._run_company(company)
        except Exception as exc:
            company_dir = self.store.company_dir(company.ticker)
            signature = stable_id("RUN", self.config.run_id, company.ticker, "unavailable")
            error = f"{type(exc).__name__}: {exc}"
            self.store.write_text(company_dir / "error.txt", error + "\n\n" + traceback.format_exc())
            result = self._failed_result(company, signature, company_dir, error)
            self.store.write_json(company_dir / "result.json", result)
            self._checkpoint(company_dir / "checkpoint.json", signature, "FAILED", error=error)
            return result

    def _run_company(self, company: CompanyInput) -> CompanyRunResult:
        company_dir = self.store.company_dir(company.ticker)
        signature = self._company_signature(company)
        result_path = company_dir / "result.json"
        checkpoint_path = company_dir / "checkpoint.json"
        if self.config.resume and result_path.is_file():
            existing = CompanyRunResult.model_validate(self.store.read_json(result_path))
            if existing.run_signature != signature:
                return self._failed_result(company, signature, company_dir, "resume signature mismatch")
            reusable = {CompanyRunStatus.COMPLETE, CompanyRunStatus.NO_PIT_DOCUMENTS}
            if existing.status in reusable or (
                existing.status == CompanyRunStatus.PREPARED and self.config.dry_run
            ):
                return existing

        audit_path = company_dir / "llm-calls.jsonl"
        audits: list[LLMCallAudit] = []
        if self.config.resume and audit_path.is_file():
            audits = [
                LLMCallAudit.model_validate_json(line)
                for line in audit_path.read_text(encoding="utf-8-sig").splitlines()
                if line
            ]
        source_document_ids: list[str] = []
        try:
            if company.price_as_of is not None and company.price_as_of > self.config.as_of:
                raise ValueError(
                    f"price_as_of {company.price_as_of.isoformat()} is after run as_of "
                    f"{self.config.as_of.isoformat()}"
                )
            if company.price_as_of is not None and (
                self.config.as_of - company.price_as_of
            ) > timedelta(days=self.config.maximum_price_age_days):
                raise ValueError(
                    f"price_as_of {company.price_as_of.isoformat()} is older than the configured "
                    f"{self.config.maximum_price_age_days}-day limit"
                )
            self._checkpoint(checkpoint_path, signature, "INGESTING")
            bundles, chunks = self._ingest_company(company, company_dir)
            if not bundles:
                result = CompanyRunResult(
                    ticker=company.ticker,
                    issuer_id=company.issuer_id,
                    issuer_name=company.issuer_name,
                    status=CompanyRunStatus.NO_PIT_DOCUMENTS,
                    run_signature=signature,
                    artifact_directory=str(company_dir),
                    runner_version=RUNNER_VERSION,
                )
                self.store.write_json(result_path, result)
                self._checkpoint(checkpoint_path, signature, "NO_PIT_DOCUMENTS")
                return result
            source_document_ids = [bundle.metadata.source_document_id for bundle in bundles]
            quality_assessments = [
                assess_parser_quality(
                    bundle,
                    ParserQualityGateConfig(
                        minimum_text_retention=self.config.minimum_text_retention,
                        minimum_numeric_retention=self.config.minimum_numeric_retention,
                        minimum_structured_fact_retention=self.config.minimum_structured_fact_retention,
                        require_table_count_match=self.config.require_table_count_match,
                        require_financial_table_semantics=self.config.require_financial_table_semantics,
                    ),
                )
                for bundle in bundles
            ]
            self.store.write_json(company_dir / "quality-gate.json", quality_assessments)
            rejected = [assessment for assessment in quality_assessments if not assessment.passed]
            if rejected and not self.config.allow_low_quality:
                details = "; ".join(
                    f"{assessment.source_document_id}: {', '.join(assessment.failures)}"
                    for assessment in rejected
                )
                raise ValueError(f"parser quality gate failed: {details}")
            snapshot = self.snapshots.build(bundles, as_of=self.config.as_of)
            self.store.write_json(company_dir / "financial-snapshot.json", snapshot)
            self.store.write_text(company_dir / "financial-snapshot.md", snapshot.to_markdown())
            dcf = self._calculate_dcf(company, company_dir, snapshot)

            dedup = deduplicate_chunks(chunks)
            chunks = dedup.kept
            self.store.write_json(company_dir / "chunk-dedup.json", dedup)
            self.store.write_jsonl(company_dir / "chunks.jsonl", chunks)

            evidence_chunks = select_evidence_chunks(chunks, self.config.maximum_evidence_chunks)
            evidence_batches = (
                batch_evidence_chunks(evidence_chunks, self.config.evidence_batch_max_tokens)
                if self.config.evidence_batch_max_tokens is not None
                else [[chunk] for chunk in evidence_chunks]
            )
            self.store.write_json(
                company_dir / "evidence-chunk-selection.json",
                {
                    "method": "role_keyword_diversity/1",
                    "available_chunk_count": len(chunks),
                    "selected_chunk_count": len(evidence_chunks),
                    "selected_chunk_ids": [chunk.chunk_id for chunk in evidence_chunks],
                    "batch_count": len(evidence_batches),
                    "batch_token_counts": [sum(chunk.token_count for chunk in batch) for batch in evidence_batches],
                },
            )

            evidence_requests = [
                build_evidence_request(batch[0])
                if len(batch) == 1
                else build_evidence_batch_request(batch)
                for batch in evidence_batches
            ]
            self.store.write_jsonl(company_dir / "evidence-requests.jsonl", evidence_requests)
            if self.config.dry_run:
                result = CompanyRunResult(
                    ticker=company.ticker,
                    issuer_id=company.issuer_id or bundles[0].metadata.issuer_id,
                    issuer_name=company.issuer_name or bundles[0].metadata.issuer_name,
                    status=CompanyRunStatus.PREPARED,
                    run_signature=signature,
                    source_document_ids=source_document_ids,
                    chunk_count=len(chunks),
                    dcf=dcf,
                    current_price=company.current_price,
                    price_as_of=company.price_as_of,
                    valuation_as_of=self.config.as_of,
                    artifact_directory=str(company_dir),
                    runner_version=RUNNER_VERSION,
                )
                self.store.write_json(result_path, result)
                self._checkpoint(checkpoint_path, signature, "PREPARED")
                return result

            self._checkpoint(checkpoint_path, signature, "EXTRACTING_EVIDENCE")
            cards = self._load_or_extract_evidence(company_dir, evidence_chunks, bundles, audits)
            self.store.write_json(
                company_dir / "forward-driver-cards.json",
                build_forward_driver_cards(cards),
            )
            relations = build_evidence_relations(cards)
            evidence_clusters = cluster_duplicate_evidence(cards, relations)
            self.store.write_json(company_dir / "evidence-clusters.json", evidence_clusters)
            self.store.write_json(
                company_dir / "moat-outcome-corroboration.json",
                [
                    card
                    for card in cards
                    if card.evidence_type in OUTCOME_CORROBORATION_TYPES
                ],
            )

            self._checkpoint(checkpoint_path, signature, "SUMMARIZING")
            summaries = self._load_or_summarize(company_dir, evidence_chunks, cards, audits)
            dossier = self._build_dossier(company, bundles, snapshot, cards, relations, summaries)
            self.store.write_json(company_dir / "dossier.json", dossier)

            canonical_ids = {cluster.canonical_evidence_id for cluster in evidence_clusters}
            structural_score_ids = {
                card.evidence_id
                for card in dossier.evidence
                if card.evidence_id in canonical_ids
                and (
                    (
                        card.direction == EvidenceDirection.MOAT_POSITIVE
                        and card.evidence_type in STRUCTURAL_MOAT_TYPES
                    )
                    or card.direction == EvidenceDirection.MOAT_NEGATIVE
                )
            }
            scoring_base_dossier = self._filter_dossier_evidence(dossier, structural_score_ids)
            retrieval = self.retriever.retrieve(scoring_base_dossier.evidence)
            self.store.write_json(company_dir / "retrieval.json", retrieval)
            scoring_chunk_ids = {
                card.source_chunk_id for card in scoring_base_dossier.evidence
            }
            raw_context_candidates = (
                [chunk for chunk in chunks if chunk.chunk_id in scoring_chunk_ids]
                if self.config.include_raw_moat_appendix
                else []
            )
            scoring_dossier, pruning = self._fit_dossier_to_context(
                scoring_base_dossier,
                snapshot,
                retrieval,
                raw_context_candidates,
            )
            self.store.write_json(company_dir / "scoring-dossier.json", scoring_dossier)
            self.store.write_json(company_dir / "evidence-pruning.json", pruning)
            preliminary = self.pack_builder.build(scoring_dossier, snapshot, [])
            remaining_context = self.config.context_tokens - preliminary.token_count
            if remaining_context <= self.config.prompt_reserve_tokens:
                raise ValueError(
                    f"summary/evidence layers use {preliminary.token_count} tokens and exceed the available context"
                )
            cited_chunk_ids = {card.source_chunk_id for card in scoring_dossier.evidence}
            raw_candidates = (
                [chunk for chunk in chunks if chunk.chunk_id in cited_chunk_ids]
                if self.config.include_raw_moat_appendix
                else []
            )
            allocation = DynamicTokenBudgetAllocator(
                model_context_tokens=remaining_context,
                prompt_reserve_tokens=self.config.prompt_reserve_tokens,
            ).allocate(raw_candidates, relevance=retrieval.chunk_relevance)
            pack = self.pack_builder.build(scoring_dossier, snapshot, allocation.selected)
            selected_chunks = list(allocation.selected)
            dropped_ids = list(allocation.dropped_chunk_ids)
            while pack.token_count + self.config.prompt_reserve_tokens > self.config.context_tokens and selected_chunks:
                weakest = min(
                    selected_chunks,
                    key=lambda chunk: (
                        retrieval.chunk_relevance.get(chunk.chunk_id, 0.0),
                        -chunk.token_count,
                    ),
                )
                selected_chunks.remove(weakest)
                dropped_ids.append(weakest.chunk_id)
                pack = self.pack_builder.build(scoring_dossier, snapshot, selected_chunks)
            if pack.token_count + self.config.prompt_reserve_tokens > self.config.context_tokens:
                raise ValueError("fixed evidence pack exceeds configured model context")
            allocation = allocation.model_copy(
                update={
                    "selected": selected_chunks,
                    "dropped_chunk_ids": dropped_ids,
                    "used_tokens": sum(chunk.token_count for chunk in selected_chunks),
                }
            )
            self.store.write_json(company_dir / "context-allocation.json", allocation)
            self.store.write_json(company_dir / "evidence-pack.json", pack)
            self.store.write_text(company_dir / "evidence-pack.md", pack.markdown)

            self._checkpoint(checkpoint_path, signature, "SCORING")
            score_path = company_dir / "moat-score.json"
            score_request = build_moat_pack_request(scoring_dossier, pack)
            self.store.write_json(company_dir / "moat-request.json", score_request)
            if self.config.resume and score_path.is_file():
                score = MoatScore.model_validate(self.store.read_json(score_path))
            else:
                score = self._execute_validated(
                    score_request,
                    MoatScore,
                    lambda value: self._validate_score(
                        value,
                        scoring_dossier.evidence,
                    ),
                    audits,
                    company_dir,
                )
                score = derive_moat_score(score, scoring_dossier.evidence)
                total_structural_evidence = len(structural_score_ids)
                score = score.model_copy(
                    update={
                        "issuer_id": dossier.issuer_id,
                        "as_of": self.config.as_of.date(),
                        "document_coverage": self._coverage(
                            bundles,
                            chunks,
                            allocation.selected,
                            total_evidence=total_structural_evidence,
                            selected_evidence=len(scoring_dossier.evidence),
                        ),
                    }
                )
                self.store.write_json(score_path, score)

            score_audits = [
                audit for audit in audits if audit.task == "FINAL_MOAT_SCORING"
            ]
            effective_score_audit = score_audits[-1] if score_audits else None
            actual_or_estimated_tokens = (
                effective_score_audit.usage.input_tokens
                if effective_score_audit and effective_score_audit.usage.input_tokens > 0
                else pack.token_count
            )
            self.store.write_json(
                company_dir / "run-manifest.json",
                RunManifest(
                    run_id=f"{self.config.run_id}:{company.ticker}",
                    signal_at=self.config.as_of,
                    evidence_cutoff=self.config.as_of,
                    model=effective_score_audit.model if effective_score_audit else self.config.moat_model,
                    parser_version=",".join(sorted({bundle.metadata.parser_version for bundle in bundles})),
                    renderer_version="canonical-markdown/1",
                    prompt_version="structural-moat-pack/2",
                    token_budget=self.config.context_tokens,
                    input_tokens=actual_or_estimated_tokens,
                    input_sha256=(
                        effective_score_audit.input_sha256
                        if effective_score_audit
                        else score_request.input_sha256
                    ),
                    temperature=score_request.temperature,
                    created_at=datetime.now(timezone.utc),
                ),
            )

            usage = self._sum_usage(audits)
            self.store.write_jsonl(company_dir / "llm-calls.jsonl", audits)
            result = CompanyRunResult(
                ticker=company.ticker,
                issuer_id=dossier.issuer_id,
                issuer_name=dossier.issuer_name,
                status=CompanyRunStatus.COMPLETE,
                run_signature=signature,
                source_document_ids=source_document_ids,
                evidence_count=len(cards),
                chunk_count=len(chunks),
                selected_chunk_count=len(allocation.selected),
                moat_score=score,
                dcf=dcf,
                current_price=company.current_price,
                price_as_of=company.price_as_of,
                valuation_as_of=self.config.as_of,
                artifact_directory=str(company_dir),
                llm_usage=usage,
                runner_version=RUNNER_VERSION,
            )
            self.store.write_json(result_path, result)
            self._checkpoint(checkpoint_path, signature, "COMPLETE")
            return result
        except Exception as exc:
            if audits:
                self.store.write_jsonl(company_dir / "llm-calls.jsonl", audits)
            error = f"{type(exc).__name__}: {exc}"
            self.store.write_text(company_dir / "error.txt", error + "\n\n" + traceback.format_exc())
            result = self._failed_result(
                company,
                signature,
                company_dir,
                error,
                source_document_ids=source_document_ids,
                usage=self._sum_usage(audits),
            )
            self.store.write_json(result_path, result)
            self._checkpoint(checkpoint_path, signature, "FAILED", error=error)
            return result

    def _ingest_company(
        self,
        company: CompanyInput,
        company_dir: Path,
    ) -> tuple[list[CanonicalDocumentBundle], list[SemanticChunk]]:
        bundles: list[CanonicalDocumentBundle] = []
        for document in company.documents:
            input_path = Path(document.input_path)
            metadata = json.loads(Path(document.metadata_path).read_text(encoding="utf-8-sig"))
            metadata["source_type"] = document.source.value
            metadata.setdefault("ticker", company.ticker)
            metadata.setdefault("issuer_id", company.issuer_id)
            metadata.setdefault("issuer_name", company.issuer_name)
            source_uri = str(metadata.get("primary_document_url") or input_path.resolve().as_uri())
            media_type = (
                "application/xml"
                if input_path.suffix.lower() == ".xml"
                else "application/xhtml+xml"
                if input_path.suffix.lower() == ".xhtml"
                else "text/html"
            )
            raw = RawDocument(
                content=input_path.read_bytes(),
                uri=source_uri,
                fetched_at=datetime.fromtimestamp(input_path.stat().st_mtime, timezone.utc),
                media_type=media_type,
                hints=metadata,
            )
            bundle = self.pipeline.ingest(raw)
            if bundle.metadata.available_at <= self.config.as_of:
                bundles.append(bundle)
        bundles.sort(key=lambda bundle: bundle.metadata.available_at, reverse=True)
        chunks: list[SemanticChunk] = []
        for bundle in bundles:
            artifact = company_dir / "documents" / stable_id("D", bundle.metadata.source_document_id)
            self.store.write_json(artifact / "bundle.json", bundle)
            self.store.write_text(artifact / "document.md", self.pipeline.renderer.render_document(bundle))
            document_chunks = []
            for chunk in self.pipeline.chunker.chunk(bundle):
                metadata = {
                    **chunk.metadata,
                    "available_at": bundle.metadata.available_at.isoformat(),
                    "source_document_id": bundle.metadata.source_document_id,
                }
                document_chunks.append(chunk.model_copy(update={"metadata": metadata}))
            self.store.write_jsonl(artifact / "chunks.jsonl", document_chunks)
            chunks.extend(document_chunks)
        return bundles, chunks

    def _load_or_extract_evidence(
        self,
        company_dir: Path,
        chunks: list[SemanticChunk],
        bundles: list[CanonicalDocumentBundle],
        audits: list[LLMCallAudit],
    ) -> list[EvidenceCard]:
        path = company_dir / "evidence.jsonl"
        checkpoint_dir = company_dir / "evidence-by-chunk"
        bundle_by_document = {bundle.ast.document_id: bundle for bundle in bundles}
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        cards: list[EvidenceCard] = []
        batches = (
            batch_evidence_chunks(chunks, self.config.evidence_batch_max_tokens)
            if self.config.evidence_batch_max_tokens is not None
            else [[chunk] for chunk in chunks]
        )
        for batch in batches:
            batch_ids = [chunk.chunk_id for chunk in batch]
            checkpoint = checkpoint_dir / f"{stable_id('EC', *batch_ids)}.json"
            result: EvidenceExtractionResult | EvidenceBatchExtractionResult | None = None
            if self.config.resume and checkpoint.is_file():
                stored = self.store.read_json(checkpoint)
                if len(batch) == 1:
                    candidate = EvidenceExtractionResult.model_validate(stored)
                    validation_errors = validate_evidence_result(
                        candidate,
                        batch[0],
                        bundle_by_document[batch[0].document_id],
                        discard_invalid_cards=True,
                    )
                else:
                    candidate = EvidenceBatchExtractionResult.model_validate(stored)
                    validation_errors = validate_evidence_batch_result(
                        candidate,
                        batch,
                        bundle_by_document,
                        discard_invalid_cards=True,
                    )
                if not validation_errors:
                    result = candidate
            if result is None and len(batch) == 1:
                chunk = batch[0]
                request = build_evidence_request(chunk)
                result = self._execute_validated(
                    request,
                    EvidenceExtractionResult,
                    lambda value, c=chunk: validate_evidence_result(
                        value,
                        c,
                        bundle_by_document[c.document_id],
                        discard_invalid_cards=True,
                    ),
                    audits,
                    company_dir,
                )
            elif result is None:
                request = build_evidence_batch_request(batch)
                result = self._execute_validated(
                    request,
                    EvidenceBatchExtractionResult,
                    lambda value, b=batch: validate_evidence_batch_result(
                        value,
                        b,
                        bundle_by_document,
                        discard_invalid_cards=True,
                    ),
                    audits,
                    company_dir,
                )
            chunk_cards: list[EvidenceCard] = []
            for index, card in enumerate(result.cards):
                source_chunk = chunk_by_id[card.source_chunk_id]
                source_type = (
                    source_chunk.source_refs[0].source_type
                    if source_chunk.source_refs
                    else card.source_type
                )
                normalized = normalize_card_semantics(
                    card.model_copy(update={"source_type": source_type})
                )
                evidence_id = stable_id(
                    "E",
                    normalized.source_chunk_id,
                    index,
                    normalized.evidence_type.value,
                    normalized.direction.value,
                    normalized.fact,
                )
                chunk_cards.append(
                    calibrate_card_reliability(
                        normalized.model_copy(update={"evidence_id": evidence_id})
                    )
                )
            normalized_result: EvidenceExtractionResult | EvidenceBatchExtractionResult
            if len(batch) == 1:
                normalized_result = EvidenceExtractionResult(chunk_id=batch[0].chunk_id, cards=chunk_cards)
            else:
                normalized_result = EvidenceBatchExtractionResult(cards=chunk_cards)
            self.store.write_json(checkpoint, normalized_result)
            cards.extend(chunk_cards)
            self.store.write_jsonl(path, cards)
        return cards

    def _load_or_summarize(
        self,
        company_dir: Path,
        chunks: list[SemanticChunk],
        cards: list[EvidenceCard],
        audits: list[LLMCallAudit],
    ) -> list[SectionSummary]:
        path = company_dir / "section-summaries.json"
        checkpoint_dir = company_dir / "section-summary-by-path"
        path_by_chunk = {chunk.chunk_id: chunk.section_path for chunk in chunks}
        grouped: dict[tuple[str, ...], list[EvidenceCard]] = defaultdict(list)
        if self.config.consolidate_section_summaries and cards:
            grouped[("Selected MOAT evidence",)] = list(cards)
        else:
            for card in cards:
                grouped[tuple(path_by_chunk.get(card.source_chunk_id, []))].append(card)
        summaries: list[SectionSummary] = []
        for path_tuple, section_cards in grouped.items():
            allowed = {card.evidence_id for card in section_cards}
            checkpoint = checkpoint_dir / f"{stable_id('SS', *path_tuple)}.json"
            summary = None
            if self.config.resume and checkpoint.is_file():
                candidate = SectionSummary.model_validate(self.store.read_json(checkpoint))
                if not self._validate_summary(candidate, list(path_tuple), allowed):
                    summary = candidate
            if summary is None:
                request = build_section_summary_request(list(path_tuple), section_cards)
                summary = self._execute_validated(
                    request,
                    SectionSummary,
                    lambda value, path_value=list(path_tuple), ids=allowed: self._validate_summary(value, path_value, ids),
                    audits,
                    company_dir,
                )
            self.store.write_json(checkpoint, summary)
            summaries.append(summary)
            self.store.write_json(path, [item.model_dump(mode="json") for item in summaries])
        if not grouped:
            self.store.write_json(path, [])
        return summaries

    def _calculate_dcf(
        self,
        company: CompanyInput,
        company_dir: Path,
        snapshot: FinancialSnapshot,
    ):
        if not company.dcf_assumptions_path:
            return None
        if not snapshot.series:
            raise ValueError(
                "DCF hard fail: FinancialSnapshot has no canonical numeric series; "
                "a fair value must not be generated from detached assumptions"
            )
        assumptions_text = Path(company.dcf_assumptions_path).read_text(encoding="utf-8-sig")
        assumptions = DcfAssumptions.model_validate_json(assumptions_text)
        diluted_series = snapshot.series_index().get("DILUTED_SHARES")
        diluted_points = [
            point
            for point in (diluted_series.points if diluted_series else [])
            if point.period <= self.config.as_of.date() and point.value > 0
        ]
        share_count_basis = "PIT_KRX_LISTED_SHARES_NOT_FULLY_DILUTED"
        if diluted_points:
            point = max(diluted_points, key=lambda item: (item.period, item.available_at))
            conservative_shares = max(assumptions.diluted_shares, point.value)
            sources = dict(assumptions.assumption_sources)
            sources["diluted_shares"] = [
                f"PIT_KRX_LISTED_SHARES+FINANCIAL_SNAPSHOT_DILUTED_SHARES:{point.period.isoformat()}"
            ]
            types = dict(assumptions.assumption_types)
            types["diluted_shares"] = DcfAssumptionType.DISCLOSED_FACT
            warnings = [
                warning
                for warning in assumptions.provenance_warnings
                if "potential options/convertibles" not in warning
            ]
            assumptions = assumptions.model_copy(
                update={
                    "diluted_shares": conservative_shares,
                    "assumption_sources": sources,
                    "assumption_types": types,
                    "provenance_warnings": warnings,
                }
            )
            share_count_basis = "MAX_OF_PIT_KRX_LISTED_AND_DISCLOSED_DILUTED_SHARES"
        assumptions_hash = hashlib.sha256(assumptions_text.encode("utf-8")).hexdigest()
        effective_assumptions_hash = hashlib.sha256(
            assumptions.model_dump_json(exclude_none=True).encode("utf-8")
        ).hexdigest()
        dcf = DcfEngine().value(assumptions)
        self.store.write_json(company_dir / "dcf-assumptions.json", assumptions)
        self.store.write_json(
            company_dir / "dcf-manifest.json",
            {
                "valuation_as_of": self.config.as_of.isoformat(),
                "assumptions_input_sha256": assumptions_hash,
                "effective_assumptions_sha256": effective_assumptions_hash,
                "engine_version": "unlevered-dcf/2",
                "calculation_mode": "deterministic_python",
                "llm_model": None,
                "method": dcf.method,
                "base_period": dcf.base_period,
                "assumption_sources": assumptions.assumption_sources,
                "assumption_types": {
                    key: value.value for key, value in assumptions.assumption_types.items()
                },
                "assumption_confidence": dcf.assumption_confidence,
                "confidence_penalty": dcf.confidence_penalty,
                "default_assumptions": dcf.default_assumptions,
                "terminal_value_share": dcf.terminal_value_share,
                "provenance_warnings": dcf.provenance_warnings,
                "share_count_basis": share_count_basis,
                "financial_snapshot_series": [series.concept for series in snapshot.series],
            },
        )
        self.store.write_json(company_dir / "dcf.json", dcf)
        return dcf

    def _execute_validated(
        self,
        request: LLMRequest,
        response_model: type[ResponseT],
        validator: Any,
        audits: list[LLMCallAudit],
        company_dir: Path,
    ) -> ResponseT:
        assert self.transport is not None
        current = request
        errors: list[str] = []
        for attempt in range(self.config.validation_attempts):
            result: TransportResult[ResponseT] = self.transport.execute(current, response_model)
            raw_output = result.raw_output_text or result.parsed.model_dump_json(exclude_none=True)
            normalized_output = json.dumps(
                result.parsed.model_dump(mode="json", exclude_none=True),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            raw_response_sha256 = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
            normalized_output_sha256 = hashlib.sha256(normalized_output.encode("utf-8")).hexdigest()
            raw_path = company_dir / "llm-raw" / (
                stable_id("LLM", current.task.value, current.input_sha256, attempt) + ".json"
            )
            self.store.write_json(
                raw_path,
                {
                    "task": current.task.value,
                    "input_sha256": current.input_sha256,
                    "model": result.model,
                    "provider": result.provider,
                    "response_id": result.response_id,
                    "raw_output_text": raw_output,
                    "raw_response_sha256": raw_response_sha256,
                    "normalized_output": result.parsed,
                    "normalized_output_sha256": normalized_output_sha256,
                },
            )
            audits.append(
                LLMCallAudit(
                    task=current.task.value,
                    input_sha256=current.input_sha256,
                    provider=result.provider,
                    model=result.model,
                    response_id=result.response_id,
                    usage=result.usage,
                    created_at=datetime.now(timezone.utc),
                    raw_response_path=str(raw_path),
                    raw_response_sha256=raw_response_sha256,
                    normalized_output_sha256=normalized_output_sha256,
                )
            )
            self.store.write_jsonl(company_dir / "llm-calls.jsonl", audits)
            errors = list(validator(result.parsed))
            if not errors:
                return result.parsed
            current = self._repair_request(request, errors, attempt + 1)
        raise ValueError("LLM output failed validation: " + "; ".join(errors))

    @staticmethod
    def _repair_request(request: LLMRequest, errors: list[str], attempt: int) -> LLMRequest:
        user = (
            request.user
            + "\n\nYour previous output failed deterministic validation. Return the complete corrected object.\n"
            + "Validation errors:\n- "
            + "\n- ".join(errors)
        )
        input_hash = hashlib.sha256(f"{request.system}\n\n{user}".encode("utf-8")).hexdigest()
        return request.model_copy(
            update={"user": user, "input_sha256": input_hash, "metadata": {**request.metadata, "repair_attempt": attempt}}
        )

    def _build_dossier(
        self,
        company: CompanyInput,
        bundles: list[CanonicalDocumentBundle],
        snapshot: FinancialSnapshot,
        cards: list[EvidenceCard],
        relations: list[EvidenceRelation],
        summaries: list[SectionSummary],
    ) -> CompanyDossier:
        business_summary = self._render_business_summary(summaries)
        table_ids = [
            node.node_id
            for bundle in bundles
            for node in bundle.ast.walk()
            if isinstance(node, TableNode)
        ]
        metadata = bundles[0].metadata
        return CompanyDossier(
            issuer_id=company.issuer_id or metadata.issuer_id,
            issuer_name=company.issuer_name or metadata.issuer_name or company.ticker,
            ticker=company.ticker,
            as_of=self.config.as_of,
            source_document_ids=[bundle.metadata.source_document_id for bundle in bundles],
            business_summary=business_summary,
            financial_summary=snapshot.to_markdown(),
            evidence=cards,
            relations=relations,
            section_summaries=summaries,
            key_table_ids=table_ids,
        )

    @staticmethod
    def _render_business_summary(summaries: list[SectionSummary]) -> str | None:
        business_lines: list[str] = []
        for summary in summaries:
            business_lines.append(f"### {' > '.join(summary.section_path) or '(root)'}")
            if summary.positive_evidence_ids:
                business_lines.append("- Positive: " + ", ".join(f"[{item}]" for item in summary.positive_evidence_ids))
            if summary.negative_evidence_ids:
                business_lines.append("- Negative: " + ", ".join(f"[{item}]" for item in summary.negative_evidence_ids))
            if summary.key_mechanisms:
                business_lines.append(
                    "- Mechanisms: "
                    + "; ".join(
                        f"[{', '.join(claim.evidence_ids)}] {claim.text}"
                        for claim in summary.key_mechanisms
                    )
                )
            if summary.key_kpis:
                business_lines.append(
                    "- KPIs: "
                    + "; ".join(
                        f"[{', '.join(claim.evidence_ids)}] {claim.text}"
                        for claim in summary.key_kpis
                    )
                )
            if summary.uncertainties:
                business_lines.append(
                    "- Uncertainties: "
                    + "; ".join(
                        f"[{', '.join(claim.evidence_ids)}] {claim.text}"
                        for claim in summary.uncertainties
                    )
                )
        return "\n".join(business_lines) if business_lines else None

    def _fit_dossier_to_context(
        self,
        dossier: CompanyDossier,
        snapshot: FinancialSnapshot,
        retrieval: RetrievalResult,
        chunks: list[SemanticChunk],
    ) -> tuple[CompanyDossier, dict[str, Any]]:
        raw_reserve = min(8_000, sum(chunk.token_count for chunk in chunks))
        target = self.config.context_tokens - self.config.prompt_reserve_tokens - raw_reserve
        if target <= 0:
            raise ValueError("context is too small after prompt and raw-evidence reserves")
        full_tokens = self.pack_builder.build(dossier, snapshot, []).token_count
        if full_tokens <= target:
            return dossier, {
                "pruned": False,
                "target_tokens": target,
                "full_tokens": full_tokens,
                "selected_tokens": full_tokens,
                "selected_evidence_ids": [card.evidence_id for card in dossier.evidence],
                "dropped_evidence_ids": [],
            }

        retrieved = {hit.evidence_id for hit in retrieval.hits}
        ordered = sorted(
            dossier.evidence,
            key=lambda card: (
                card.evidence_id in retrieved,
                card.direction.value == "MOAT_NEGATIVE",
                card.strength * card.reliability,
            ),
            reverse=True,
        )

        def filtered(count: int) -> CompanyDossier:
            selected_cards = ordered[:count]
            selected_ids = {card.evidence_id for card in selected_cards}
            summaries: list[SectionSummary] = []
            for summary in dossier.section_summaries:
                filtered_summary = self._filter_summary(summary, selected_ids)
                if filtered_summary is not None:
                    summaries.append(filtered_summary)
            relations = [
                relation
                for relation in dossier.relations
                if relation.from_evidence_id in selected_ids and relation.to_evidence_id in selected_ids
            ]
            return dossier.model_copy(
                update={
                    "business_summary": self._render_business_summary(summaries),
                    "evidence": selected_cards,
                    "relations": relations,
                    "section_summaries": summaries,
                }
            )

        empty = filtered(0)
        empty_tokens = self.pack_builder.build(empty, snapshot, []).token_count
        if empty_tokens > target:
            raise ValueError(
                f"fixed structural metadata uses {empty_tokens} tokens, above evidence-layer target {target}"
            )
        low, high = 0, len(ordered)
        while low < high:
            middle = (low + high + 1) // 2
            candidate = filtered(middle)
            if self.pack_builder.build(candidate, snapshot, []).token_count <= target:
                low = middle
            else:
                high = middle - 1
        selected = filtered(low)
        selected_pack_tokens = self.pack_builder.build(selected, snapshot, []).token_count
        selected_ids = {card.evidence_id for card in selected.evidence}
        return selected, {
            "pruned": True,
            "target_tokens": target,
            "full_tokens": full_tokens,
            "selected_tokens": selected_pack_tokens,
            "selected_evidence_ids": [card.evidence_id for card in selected.evidence],
            "dropped_evidence_ids": [
                card.evidence_id for card in dossier.evidence if card.evidence_id not in selected_ids
            ],
        }

    def _filter_dossier_evidence(
        self,
        dossier: CompanyDossier,
        selected_ids: set[str],
    ) -> CompanyDossier:
        selected_cards = [card for card in dossier.evidence if card.evidence_id in selected_ids]
        summaries: list[SectionSummary] = []
        for summary in dossier.section_summaries:
            filtered_summary = self._filter_summary(summary, selected_ids)
            if filtered_summary is not None:
                summaries.append(filtered_summary)
        relations = [
            relation
            for relation in dossier.relations
            if relation.from_evidence_id in selected_ids and relation.to_evidence_id in selected_ids
        ]
        return dossier.model_copy(
            update={
                "business_summary": self._render_business_summary(summaries),
                "evidence": selected_cards,
                "relations": relations,
                "section_summaries": summaries,
            }
        )

    @staticmethod
    def _filter_summary(summary: SectionSummary, selected_ids: set[str]) -> SectionSummary | None:
        positive = [item for item in summary.positive_evidence_ids if item in selected_ids]
        negative = [item for item in summary.negative_evidence_ids if item in selected_ids]

        def claims(values: list[Any]) -> list[Any]:
            filtered: list[Any] = []
            for claim in values:
                evidence_ids = [item for item in claim.evidence_ids if item in selected_ids]
                if evidence_ids:
                    filtered.append(claim.model_copy(update={"evidence_ids": evidence_ids}))
            return filtered

        mechanisms = claims(summary.key_mechanisms)
        kpis = claims(summary.key_kpis)
        uncertainties = claims(summary.uncertainties)
        if not positive and not negative and not mechanisms and not kpis and not uncertainties:
            return None
        return summary.model_copy(
            update={
                "positive_evidence_ids": positive,
                "negative_evidence_ids": negative,
                "key_mechanisms": mechanisms,
                "key_kpis": kpis,
                "uncertainties": uncertainties,
            }
        )

    @staticmethod
    def _validate_summary(summary: SectionSummary, path: list[str], evidence_ids: set[str]) -> list[str]:
        summary.section_path = list(path)
        summary.positive_evidence_ids = [item for item in summary.positive_evidence_ids if item in evidence_ids]
        summary.negative_evidence_ids = [item for item in summary.negative_evidence_ids if item in evidence_ids]

        def grounded_claims(values: list[Any]) -> list[Any]:
            result = []
            for claim in values:
                grounded = [item for item in claim.evidence_ids if item in evidence_ids]
                if grounded:
                    result.append(claim.model_copy(update={"evidence_ids": grounded}))
            return result

        summary.key_mechanisms = grounded_claims(summary.key_mechanisms)
        summary.key_kpis = grounded_claims(summary.key_kpis)
        summary.uncertainties = grounded_claims(summary.uncertainties)
        return []

    def _validate_score(
        self,
        score: MoatScore,
        evidence: list[EvidenceCard],
    ) -> list[str]:
        score.as_of = self.config.as_of.date()
        return validate_moat_score(score, evidence)

    @staticmethod
    def _coverage(
        bundles: list[CanonicalDocumentBundle],
        chunks: list[SemanticChunk],
        selected: list[SemanticChunk],
        *,
        total_evidence: int,
        selected_evidence: int,
    ) -> CoverageMetrics:
        raw_chars = sum(bundle.quality.raw_visible_chars for bundle in bundles)
        ast_chars = sum(bundle.quality.ast_chars for bundle in bundles)
        total_tokens = sum(chunk.token_count for chunk in chunks)
        selected_tokens = sum(chunk.token_count for chunk in selected)
        all_sections = {tuple(node.section_path) for bundle in bundles for node in bundle.ast.walk() if isinstance(node, SectionNode)}
        selected_sections = {tuple(chunk.section_path) for chunk in selected if chunk.section_path}
        all_tables = {
            node.node_id: node
            for bundle in bundles
            for node in bundle.ast.walk()
            if isinstance(node, TableNode)
        }
        selected_table_cells: dict[str, set[tuple[int, int]] | None] = {}
        for chunk in selected:
            table_id = chunk.metadata.get("table_id")
            if not table_id:
                for node_id in chunk.node_ids:
                    if node_id in all_tables:
                        selected_table_cells[node_id] = None
                continue
            if selected_table_cells.get(table_id) is None and table_id in selected_table_cells:
                continue
            row_start = chunk.metadata.get("row_start")
            row_end = chunk.metadata.get("row_end")
            if isinstance(row_start, int) and isinstance(row_end, int):
                selected_table_cells.setdefault(table_id, set())
                cells = selected_table_cells[table_id]
                table = all_tables.get(table_id)
                columns = chunk.metadata.get("column_indices")
                if not isinstance(columns, list) and table is not None:
                    columns = list(range(len(table.column_headers)))
                if cells is not None and isinstance(columns, list):
                    cells.update(
                        (row_index, column)
                        for row_index in range(row_start, row_end + 1)
                        for column in columns
                        if isinstance(column, int)
                    )
            else:
                selected_table_cells[table_id] = None
        total_numeric = 0
        selected_numeric = 0
        for table_id, table in all_tables.items():
            for row in table.rows:
                selection = selected_table_cells.get(table_id, set())
                for cell in row.cells:
                    if cell.numeric_value is None:
                        continue
                    total_numeric += 1
                    if selection is None or (row.index, cell.col) in selection:
                        selected_numeric += 1
        return CoverageMetrics(
            char_retention=min(1.0, ast_chars / raw_chars) if raw_chars else None,
            token_retention=min(1.0, selected_tokens / total_tokens) if total_tokens else None,
            evidence_retention=min(1.0, selected_evidence / total_evidence) if total_evidence else None,
            section_retention=min(1.0, len(selected_sections & all_sections) / len(all_sections)) if all_sections else None,
            table_retention=min(1.0, len(selected_table_cells) / len(all_tables)) if all_tables else None,
            numeric_retention=min(1.0, selected_numeric / total_numeric) if total_numeric else None,
            moat_evidence_coverage=(
                min(1.0, selected_evidence / total_evidence)
                if total_evidence
                else 0.0
            ),
        )

    def _company_signature(self, company: CompanyInput) -> str:
        digest = hashlib.sha256()
        prompt_contract = "\n\n".join(
            inspect.getsource(builder)
            for builder in (
                build_evidence_request,
                build_evidence_batch_request,
                build_section_summary_request,
                build_moat_pack_request,
            )
        )
        schema_contract = json.dumps(
            MoatScore.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest.update(
            json.dumps(
                {
                    "ticker": company.ticker,
                    "issuer_id": company.issuer_id,
                    "issuer_name": company.issuer_name,
                    "current_price": str(company.current_price) if company.current_price is not None else None,
                    "price_as_of": company.price_as_of.isoformat() if company.price_as_of else None,
                    "as_of": self.config.as_of.isoformat(),
                    "summary_model": self.config.summary_model,
                    "moat_model": self.config.moat_model,
                    "summary_reasoning_effort": self.config.summary_reasoning_effort,
                    "moat_reasoning_effort": self.config.moat_reasoning_effort,
                    "context_tokens": self.config.context_tokens,
                    "prompt_reserve_tokens": self.config.prompt_reserve_tokens,
                    "max_output_tokens": self.config.max_output_tokens,
                    "minimum_text_retention": self.config.minimum_text_retention,
                    "minimum_numeric_retention": self.config.minimum_numeric_retention,
                    "minimum_structured_fact_retention": self.config.minimum_structured_fact_retention,
                    "require_table_count_match": self.config.require_table_count_match,
                    "require_financial_table_semantics": self.config.require_financial_table_semantics,
                    "allow_low_quality": self.config.allow_low_quality,
                    "maximum_price_age_days": self.config.maximum_price_age_days,
                    "maximum_evidence_chunks": self.config.maximum_evidence_chunks,
                    "evidence_batch_max_tokens": self.config.evidence_batch_max_tokens,
                    "consolidate_section_summaries": self.config.consolidate_section_summaries,
                    "include_raw_moat_appendix": self.config.include_raw_moat_appendix,
                    "validation_attempts": self.config.validation_attempts,
                    "runner_version": RUNNER_VERSION,
                    "prompt_contract_sha256": hashlib.sha256(prompt_contract.encode("utf-8")).hexdigest(),
                    "response_schema_sha256": hashlib.sha256(schema_contract.encode("utf-8")).hexdigest(),
                },
                sort_keys=True,
            ).encode("utf-8")
        )
        for document in company.documents:
            digest.update(Path(document.input_path).read_bytes())
            digest.update(Path(document.metadata_path).read_bytes())
        if company.dcf_assumptions_path:
            digest.update(Path(company.dcf_assumptions_path).read_bytes())
        return digest.hexdigest()

    def _checkpoint(self, path: Path, signature: str, stage: str, error: str | None = None) -> None:
        self.store.write_json(
            path,
            {
                "run_signature": signature,
                "stage": stage,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "error": error,
            },
        )

    @staticmethod
    def _sum_usage(audits: list[LLMCallAudit]) -> TransportUsage:
        return TransportUsage(
            input_tokens=sum(audit.usage.input_tokens for audit in audits),
            output_tokens=sum(audit.usage.output_tokens for audit in audits),
            cached_input_tokens=sum(audit.usage.cached_input_tokens for audit in audits),
        )

    @staticmethod
    def _failed_result(
        company: CompanyInput,
        signature: str,
        company_dir: Path,
        error: str,
        *,
        source_document_ids: list[str] | None = None,
        usage: TransportUsage | None = None,
    ) -> CompanyRunResult:
        return CompanyRunResult(
            ticker=company.ticker,
            issuer_id=company.issuer_id,
            issuer_name=company.issuer_name,
            status=CompanyRunStatus.FAILED,
            run_signature=signature,
            source_document_ids=source_document_ids or [],
            error=error,
            artifact_directory=str(company_dir),
            llm_usage=usage or TransportUsage(),
            runner_version=RUNNER_VERSION,
        )
