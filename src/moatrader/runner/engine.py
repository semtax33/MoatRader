from __future__ import annotations

import hashlib
import inspect
import json
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from moatrader.adapters import RawDocument
from moatrader.audit import RunManifest
from moatrader.canonical.ids import stable_id
from moatrader.canonical.models import CanonicalDocumentBundle, SectionNode, TableNode
from moatrader.context import (
    DynamicTokenBudgetAllocator,
    EvidencePackBuilder,
    MoatStrengthContextBuilder,
    build_financial_feature_vector,
)
from moatrader.evidence.models import (
    AtomicEvidenceExtraction,
    AtomicEvidenceJudgment,
    CompanyDossier,
    ContextualMoatAssessment,
    CandidateAtomicAuditResult,
    CoverageMetrics,
    EvidenceCard,
    EvidenceDirection,
    EvidenceExtractionResult,
    EvidenceRelation,
    MoatScore,
    ReconciledMoatAssessment,
    OUTCOME_CORROBORATION_TYPES,
    STRUCTURAL_MOAT_TYPES,
    SectionSummary,
)
from moatrader.evidence.ledger import EvidenceLedgerStore
from moatrader.evidence.validation import (
    build_candidate_manifest,
    derive_audited_moat_rank_score,
    derive_audited_moat_score,
    reconcile_context_and_claims,
    normalize_candidate_atomic_audit,
    normalize_contextual_moat_assessment,
    normalize_contextual_moat_rank_assessment,
    validate_contextual_moat_assessment,
    validate_evidence_result,
    validate_moat_score,
)
from moatrader.evidence.processing import (
    assign_canonical_claim_identity,
    atomic_extraction_to_judgment,
    atomic_judgment_to_card,
    build_canonical_claim_set,
    build_evidence_relations,
    build_evidence_preserving_summaries,
    build_forward_driver_cards,
    cluster_duplicate_evidence,
    normalize_atomic_extraction,
)
from moatrader.evidence.atomic import (
    ATOMIC_RUBRIC_VERSION,
    ATOMIC_SEGMENTATION_VERSION,
    atomic_unit_set_sha256,
    build_atomic_evidence_units,
    build_candidate_atomic_evidence_allowlist,
    build_candidate_targeted_atomic_audit,
    map_context_references_to_atomic_units,
    select_atomic_evidence_units,
    select_context_cited_atomic_units,
)
from moatrader.evidence.metamorphic import audit_company_metamorphs
from moatrader.financial import (
    DcfAssumptions,
    DcfAssumptionType,
    DcfEngine,
    FinancialSnapshot,
    FinancialSnapshotBuilder,
)
from moatrader.llm import (
    LLMReplayCache,
    LLMRequest,
    LLMTransport,
    build_atomic_evidence_request,
    build_candidate_atomic_audit_request,
    build_contextual_moat_strength_request,
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
from moatrader.semantic import HeuristicTokenCounter, SemanticChunk, deduplicate_chunks
from moatrader.universe import CompanyInput, UniverseManifest


ResponseT = TypeVar("ResponseT", bound=BaseModel)
RUNNER_VERSION = "0.9.2"


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
        self.strength_context_builder = MoatStrengthContextBuilder(
            model_context_tokens=config.strength_context_tokens,
            prompt_reserve_tokens=config.strength_prompt_reserve_tokens,
        )
        self.replay_cache = (
            LLMReplayCache(
                config.llm_replay_cache_directory,
                experiment_id=config.experiment_id,
                summary_model=config.summary_model,
                moat_model=config.moat_model,
                summary_reasoning_effort=config.summary_reasoning_effort,
                atomic_reasoning_effort=config.atomic_reasoning_effort,
                moat_reasoning_effort=config.moat_reasoning_effort,
                engine_version=RUNNER_VERSION,
            )
            if config.llm_replay_cache_directory and config.experiment_id and not config.dry_run
            else None
        )
        self.evidence_ledger = (
            EvidenceLedgerStore(
                config.evidence_ledger_directory,
                experiment_id=config.experiment_id,
            )
            if config.evidence_ledger_directory and config.experiment_id and not config.dry_run
            else None
        )

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
            feature_vector = build_financial_feature_vector(snapshot)
            self.store.write_json(company_dir / "financial-feature-vector.json", feature_vector)
            self.store.write_text(company_dir / "financial-feature-vector.md", feature_vector.markdown)
            dcf = self._calculate_dcf(company, company_dir, snapshot)

            dedup = deduplicate_chunks(chunks)
            chunks = dedup.kept
            self.store.write_json(company_dir / "chunk-dedup.json", dedup)
            self.store.write_jsonl(company_dir / "chunks.jsonl", chunks)

            issuer_id = company.issuer_id or bundles[0].metadata.issuer_id
            strength_context = self.strength_context_builder.build(chunks)
            selected_strength_chunk_ids = set(strength_context.selected_chunk_ids)
            strength_chunks = [
                chunk
                for chunk in chunks
                if chunk.chunk_id in selected_strength_chunk_ids
            ]
            strength_request = build_contextual_moat_strength_request(
                strength_context,
                issuer_id=issuer_id,
                as_of=self.config.as_of.date(),
            )
            self.store.write_json(company_dir / "moat-strength-context.json", strength_context)
            self.store.write_text(
                company_dir / "moat-strength-context.md",
                strength_context.markdown,
            )
            self.store.write_json(
                company_dir / "moat-strength-retrieval.json",
                strength_context.retrieval,
            )
            self.store.write_json(
                company_dir / "moat-strength-request.json",
                strength_request,
            )
            all_atomic_units = build_atomic_evidence_units(chunks, issuer_id=issuer_id)
            evidence_chunks = select_atomic_evidence_units(
                all_atomic_units,
                self.config.maximum_atomic_evidence_units,
            )
            self.store.write_jsonl(company_dir / "atomic-evidence-units.jsonl", evidence_chunks)
            self.store.write_json(
                company_dir / "evidence-chunk-selection.json",
                {
                    "method": "atomic_content_set/1",
                    "segmentation_version": ATOMIC_SEGMENTATION_VERSION,
                    "rubric_version": ATOMIC_RUBRIC_VERSION,
                    "available_chunk_count": len(chunks),
                    "available_atomic_unit_count": len(all_atomic_units),
                    "selected_atomic_unit_count": len(evidence_chunks),
                    "atomic_unit_set_sha256": atomic_unit_set_sha256(evidence_chunks),
                    "selected_chunk_ids": [chunk.chunk_id for chunk in evidence_chunks],
                },
            )

            evidence_requests = [
                build_atomic_evidence_request(chunk, issuer_id=issuer_id)
                for chunk in evidence_chunks
            ]
            self.store.write_jsonl(company_dir / "evidence-requests.jsonl", evidence_requests)
            token_counter = HeuristicTokenCounter()
            minimal_schema_json = json.dumps(
                AtomicEvidenceExtraction.model_json_schema(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            prior_schema_json = json.dumps(
                AtomicEvidenceJudgment.model_json_schema(by_alias=False),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            minimal_schema_tokens = token_counter.count(minimal_schema_json)
            prior_schema_tokens = token_counter.count(prior_schema_json)
            static_prefix_tokens = (
                token_counter.count(evidence_requests[0].system + minimal_schema_json)
                if evidence_requests
                else 0
            )
            dynamic_suffix_tokens = sum(
                token_counter.count(request.user) for request in evidence_requests
            )
            source_evidence_tokens = sum(
                token_counter.count(chunk.markdown) for chunk in evidence_chunks
            )
            estimated_atomic_input_tokens = (
                static_prefix_tokens * len(evidence_requests) + dynamic_suffix_tokens
            )
            strength_schema_json = json.dumps(
                ContextualMoatAssessment.model_json_schema(),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            strength_static_prefix_tokens = token_counter.count(
                strength_request.system + strength_schema_json
            )
            strength_dynamic_suffix_tokens = token_counter.count(strength_request.user)
            estimated_strength_input_tokens = (
                strength_static_prefix_tokens + strength_dynamic_suffix_tokens
            )
            estimated_cacheable_prefix_tokens = (
                (static_prefix_tokens * len(evidence_requests) if static_prefix_tokens >= 1_024 else 0)
                + (
                    strength_static_prefix_tokens
                    if strength_static_prefix_tokens >= 1_024
                    else 0
                )
            )
            estimated_dynamic_suffix_tokens = (
                dynamic_suffix_tokens + strength_dynamic_suffix_tokens
            )
            expected_output_token_cap = (
                len(evidence_requests) * min(self.config.max_output_tokens, 2_000)
                + min(self.config.max_output_tokens, 8_000)
            )
            token_budget_audit = {
                "schema_version": "llm-token-budget/4",
                "optimization_priority": "CORRECTNESS_THEN_COST",
                "full_context_tokens_estimated": strength_context.token_count,
                "compact_context_tokens_estimated": strength_context.token_count,
                "strength_context_compression_ratio": 1.0,
                "strength_context_compressed": False,
                "estimated_cacheable_prefix_tokens_total": estimated_cacheable_prefix_tokens,
                "estimated_dynamic_suffix_tokens_all_tasks": estimated_dynamic_suffix_tokens,
                "expected_output_token_cap_total": expected_output_token_cap,
                "atomic_request_count": len(evidence_requests),
                "estimated_atomic_input_tokens": estimated_atomic_input_tokens,
                "estimated_static_prefix_tokens_per_request": static_prefix_tokens,
                "estimated_dynamic_suffix_tokens_total": dynamic_suffix_tokens,
                "estimated_source_evidence_tokens_total": source_evidence_tokens,
                "estimated_useful_source_token_ratio": (
                    source_evidence_tokens / estimated_atomic_input_tokens
                    if estimated_atomic_input_tokens
                    else 0.0
                ),
                "minimal_atomic_schema_tokens": minimal_schema_tokens,
                "prior_full_atomic_schema_tokens": prior_schema_tokens,
                "schema_token_reduction_fraction": (
                    max(0.0, 1.0 - minimal_schema_tokens / prior_schema_tokens)
                    if prior_schema_tokens
                    else 0.0
                ),
                "estimated_schema_tokens_avoided": max(
                    0,
                    (prior_schema_tokens - minimal_schema_tokens) * len(evidence_requests),
                ),
                "atomic_reasoning_effort": self.config.atomic_reasoning_effort,
                "atomic_max_output_tokens": min(self.config.max_output_tokens, 2_000),
                "strength_request_count": 1,
                "strength_selected_chunk_count": len(strength_context.selected_chunk_ids),
                "strength_context_token_budget": strength_context.token_budget,
                "strength_context_tokens_estimated": strength_context.token_count,
                "estimated_strength_input_tokens": estimated_strength_input_tokens,
                "estimated_strength_static_prefix_tokens": strength_static_prefix_tokens,
                "estimated_strength_dynamic_suffix_tokens": strength_dynamic_suffix_tokens,
                "strength_reasoning_effort": self.config.moat_reasoning_effort,
                "strength_max_output_tokens": min(self.config.max_output_tokens, 8_000),
                "strength_context_policy": "ALWAYS_ON_BROAD_BALANCED_CANONICAL_CHUNKS",
                "strength_compression_ablation_enabled": False,
                "canonical_schema_serialization": True,
                "prompt_cache_mode": "explicit",
                "prompt_cache_ttl": "30m",
                "prompt_cache_breakpoint_count": sum(
                    request.prompt_cache_breakpoint for request in evidence_requests
                ) + int(strength_request.prompt_cache_breakpoint),
                "prompt_cache_minimum_prefix_tokens": 1_024,
                "estimated_prefix_cache_eligible": static_prefix_tokens >= 1_024,
                "cache_eligibility_note": (
                    "heuristic only; provider-rendered prefix length controls eligibility"
                ),
                "context_strategy": "DUAL_LANE_ATOMIC_AUDIT_PLUS_BROAD_CONTEXTUAL_STRENGTH",
                "exact_usage_source": "llm-calls.jsonl provider response usage",
                "exact_counting_endpoint": "POST /v1/responses/input_tokens",
            }
            self.store.write_json(company_dir / "llm-token-budget.json", token_budget_audit)
            if self.config.dry_run:
                result = CompanyRunResult(
                    ticker=company.ticker,
                    issuer_id=company.issuer_id or bundles[0].metadata.issuer_id,
                    issuer_name=company.issuer_name or bundles[0].metadata.issuer_name,
                    status=CompanyRunStatus.PREPARED,
                    run_signature=signature,
                    source_document_ids=source_document_ids,
                    chunk_count=len(chunks),
                    selected_chunk_count=len(evidence_chunks),
                    strength_context_chunk_count=len(strength_context.selected_chunk_ids),
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

            self._checkpoint(checkpoint_path, signature, "ASSESSING_MOAT_STRENGTH")
            strength_assessment_path = company_dir / "contextual-moat-assessment.json"
            raw_strength_assessment_path = (
                company_dir / "contextual-moat-assessment-raw.json"
            )
            if self.config.resume and raw_strength_assessment_path.is_file():
                raw_strength_assessment = ContextualMoatAssessment.model_validate(
                    self.store.read_json(raw_strength_assessment_path)
                )
            else:
                # Broad context is read exactly once. All contract repairs below
                # are deterministic field/item operations and never resend the
                # 100K-class context because one field was malformed.
                raw_strength_assessment = self._execute_validated(
                    strength_request,
                    ContextualMoatAssessment,
                    lambda _value: [],
                    audits,
                    company_dir,
                )
                self.store.write_json(
                    raw_strength_assessment_path,
                    raw_strength_assessment,
                )
            strength_assessment, strength_repair = normalize_contextual_moat_assessment(
                raw_strength_assessment,
                strength_context.references,
            )
            rank_strength_assessment, rank_strength_repair = (
                normalize_contextual_moat_rank_assessment(
                    raw_strength_assessment,
                    strength_context.references,
                )
            )
            strength_errors = validate_contextual_moat_assessment(
                strength_assessment,
                strength_context.references,
            )
            if strength_errors:
                raise ValueError(
                    "normalized contextual assessment failed validation: "
                    + "; ".join(strength_errors)
                )
            self.store.write_json(strength_assessment_path, strength_assessment)
            self.store.write_json(
                company_dir / "contextual-moat-field-repair.json",
                strength_repair,
            )
            self.store.write_json(
                company_dir / "contextual-moat-rank-field-repair.json",
                rank_strength_repair,
            )
            candidate_manifest = build_candidate_manifest(strength_assessment)
            self.store.write_json(
                company_dir / "moat-candidate-manifest.json",
                candidate_manifest,
            )

            baseline_atomic_units = list(evidence_chunks)
            chunk_id_by_ref = {
                reference.ref_id: reference.chunk_id
                for reference in strength_context.references
            }
            raw_quote_by_ref = {
                reference.ref_id: reference.raw_quote
                for reference in strength_context.references
            }
            atomic_unit_ids_by_ref = map_context_references_to_atomic_units(
                all_atomic_units,
                chunk_id_by_ref=chunk_id_by_ref,
                raw_quote_by_ref=raw_quote_by_ref,
            )
            cited_atomic_units = select_context_cited_atomic_units(
                all_atomic_units,
                strength_assessment,
                chunk_id_by_ref=chunk_id_by_ref,
                raw_quote_by_ref=raw_quote_by_ref,
            )
            units_by_id = {
                unit.chunk_id: unit
                for unit in [*baseline_atomic_units, *cited_atomic_units]
            }
            evidence_chunks = sorted(
                units_by_id.values(),
                key=lambda unit: str(unit.metadata["atomic_evidence_key"]),
            )
            self.store.write_jsonl(
                company_dir / "atomic-audit-baseline-units.jsonl",
                baseline_atomic_units,
            )
            self.store.write_jsonl(
                company_dir / "atomic-evidence-units.jsonl",
                evidence_chunks,
            )
            self.store.write_json(
                company_dir / "evidence-chunk-selection.json",
                {
                    "method": "dual_lane_citation_audit/1",
                    "segmentation_version": ATOMIC_SEGMENTATION_VERSION,
                    "rubric_version": ATOMIC_RUBRIC_VERSION,
                    "available_chunk_count": len(chunks),
                    "available_atomic_unit_count": len(all_atomic_units),
                    "baseline_atomic_unit_count": len(baseline_atomic_units),
                    "context_cited_atomic_unit_count": len(cited_atomic_units),
                    "selected_atomic_unit_count": len(evidence_chunks),
                    "atomic_unit_set_sha256": atomic_unit_set_sha256(evidence_chunks),
                    "selected_chunk_ids": [unit.chunk_id for unit in evidence_chunks],
                    "parent_fallback_enabled": False,
                },
            )
            evidence_requests = [
                build_atomic_evidence_request(chunk, issuer_id=issuer_id)
                for chunk in evidence_chunks
            ]
            self.store.write_jsonl(
                company_dir / "evidence-requests.jsonl",
                evidence_requests,
            )
            self._refresh_atomic_token_budget(
                company_dir,
                evidence_requests,
                evidence_chunks,
            )

            self._checkpoint(checkpoint_path, signature, "EXTRACTING_EVIDENCE")
            current_cards = self._load_or_extract_evidence(
                company_dir,
                evidence_chunks,
                bundles,
                audits,
                issuer_id=issuer_id,
            )
            self.store.write_jsonl(company_dir / "current-evidence.jsonl", current_cards)
            ledger_merge = None
            active_ledger_document_ids: list[str] = []
            cards = current_cards
            if self.evidence_ledger is not None:
                ledger_merge = self.evidence_ledger.merge(
                    company.ticker,
                    as_of=self.config.as_of,
                    current_cards=current_cards,
                    chunks=evidence_chunks,
                    document_available_at={
                        bundle.ast.document_id: bundle.metadata.available_at for bundle in bundles
                    },
                )
                cards = ledger_merge.cards
            cards = [
                assign_canonical_claim_identity(card, issuer_id=issuer_id)
                for card in cards
            ]
            self.store.write_jsonl(company_dir / "evidence.jsonl", cards)

            allowed_candidate_evidence = build_candidate_atomic_evidence_allowlist(
                candidate_manifest,
                cards,
                evidence_chunks,
                atomic_unit_ids_by_ref=atomic_unit_ids_by_ref,
            )
            self.store.write_json(
                company_dir / "candidate-atomic-allowlist.json",
                allowed_candidate_evidence,
            )
            candidate_audit = build_candidate_targeted_atomic_audit(
                candidate_manifest,
                cards,
                allowed_evidence_ids=allowed_candidate_evidence,
            )
            self.store.write_json(
                company_dir / "candidate-atomic-audit-method.json",
                {
                    "schema_version": "candidate-targeted-atomic-audit/1",
                    "method": "PYTHON_EXACT_CANDIDATE_REF_TYPE_SCOPE_BINDING",
                    "additional_llm_calls": 0,
                },
            )
            candidate_audit, candidate_audit_repair = normalize_candidate_atomic_audit(
                candidate_audit,
                candidate_manifest,
                cards,
                allowed_evidence_ids=allowed_candidate_evidence,
            )
            self.store.write_json(
                company_dir / "candidate-atomic-audit.json",
                candidate_audit,
            )
            self.store.write_json(
                company_dir / "candidate-atomic-audit-repair.json",
                candidate_audit_repair,
            )
            self.store.write_json(
                company_dir / "forward-driver-cards.json",
                build_forward_driver_cards(current_cards),
            )
            relations = build_evidence_relations(cards)
            if self.evidence_ledger is not None and ledger_merge is not None:
                invalidated = self.evidence_ledger.apply_relations(
                    company.ticker,
                    as_of=self.config.as_of,
                    current_evidence_ids=ledger_merge.current_evidence_ids,
                    relations=relations,
                )
                if invalidated:
                    cards = [card for card in cards if card.evidence_id not in invalidated]
                    relations = build_evidence_relations(cards)
                    self.store.write_jsonl(company_dir / "evidence.jsonl", cards)
                ledger_records = self.evidence_ledger.records(company.ticker)
                active_ledger_document_ids = self.evidence_ledger.active_source_document_ids(
                    company.ticker,
                    as_of=self.config.as_of,
                )
                self.store.write_json(
                    company_dir / "evidence-ledger-snapshot.json",
                    {
                        "experiment_id": self.config.experiment_id,
                        "as_of": self.config.as_of.isoformat(),
                        "current_evidence_count": len(current_cards),
                        "carried_evidence_count": len(cards) - len(current_cards),
                        "invalidated_evidence_ids": sorted(invalidated),
                        "records": ledger_records,
                    },
                )
            evidence_clusters = cluster_duplicate_evidence(cards, relations)
            self.store.write_json(company_dir / "evidence-clusters.json", evidence_clusters)
            claim_cards, claim_clusters = build_canonical_claim_set(cards, issuer_id=issuer_id)
            self.store.write_jsonl(company_dir / "canonical-claim-set.jsonl", claim_cards)
            self.store.write_json(company_dir / "claim-clusters.json", claim_clusters)
            self.store.write_json(
                company_dir / "moat-outcome-corroboration.json",
                [
                    card
                    for card in cards
                    if card.evidence_type in OUTCOME_CORROBORATION_TYPES
                ],
            )

            reconciled = reconcile_context_and_claims(
                strength_assessment,
                cards,
                contextual_chunks=strength_chunks,
                atomic_units=evidence_chunks,
                references=strength_context.references,
                candidate_manifest=candidate_manifest,
                candidate_audit=candidate_audit,
            )
            self.store.write_json(
                company_dir / "moat-reconciliation.json",
                reconciled,
            )

            self._checkpoint(checkpoint_path, signature, "SUMMARIZING")
            summaries = self._load_or_summarize(company_dir, evidence_chunks, claim_cards, audits)
            dossier = self._build_dossier(company, bundles, snapshot, cards, relations, summaries)
            if ledger_merge is not None:
                dossier = dossier.model_copy(
                    update={
                        "source_document_ids": list(
                            dict.fromkeys(
                                [*dossier.source_document_ids, *active_ledger_document_ids]
                            )
                        )
                    }
                )
            self.store.write_json(company_dir / "dossier.json", dossier)

            structural_score_ids = {
                card.evidence_id
                for card in claim_cards
                if (
                    (
                        card.direction == EvidenceDirection.MOAT_POSITIVE
                        and card.evidence_type in STRUCTURAL_MOAT_TYPES
                    )
                    or card.direction == EvidenceDirection.MOAT_NEGATIVE
                )
            }
            scoring_base_dossier = self._filter_dossier_evidence(dossier, structural_score_ids).model_copy(
                update={"business_summary": None, "section_summaries": []}
            )
            retrieval = self.retriever.retrieve(scoring_base_dossier.evidence)
            self.store.write_json(company_dir / "retrieval.json", retrieval)
            scoring_dossier = scoring_base_dossier
            pruning = {
                "pruned": False,
                "reason": "deterministic reducer consumes the complete canonical claim set",
                "selected_evidence_ids": [card.evidence_id for card in scoring_dossier.evidence],
                "dropped_evidence_ids": [],
            }
            self.store.write_json(company_dir / "scoring-dossier.json", scoring_dossier)
            self.store.write_json(company_dir / "evidence-pruning.json", pruning)
            cited_chunk_ids = {card.source_chunk_id for card in scoring_dossier.evidence}
            selected_scoring_units = [
                unit for unit in evidence_chunks if unit.chunk_id in cited_chunk_ids
            ]
            allocation = DynamicTokenBudgetAllocator(
                model_context_tokens=self.config.context_tokens,
                prompt_reserve_tokens=self.config.prompt_reserve_tokens,
            ).allocate([], relevance={})
            pack = self.pack_builder.build(scoring_dossier, snapshot, [], claim_clusters)
            self.store.write_json(company_dir / "context-allocation.json", allocation)
            self.store.write_json(company_dir / "evidence-pack.json", pack)
            self.store.write_text(company_dir / "evidence-pack.md", pack.markdown)

            self._checkpoint(checkpoint_path, signature, "SCORING")
            score_path = company_dir / "moat-score.json"
            reducer_payload = {
                "schema_version": "dual-lane-strength-reducer/3",
                "issuer_id": dossier.issuer_id,
                "as_of": self.config.as_of.date().isoformat(),
                "contextual_assessment": strength_assessment.model_dump(
                    mode="json", exclude_none=True
                ),
                "raw_rank_assessment": rank_strength_assessment.model_dump(
                    mode="json", exclude_none=True
                ),
                "candidate_manifest": [
                    item.model_dump(mode="json", exclude_none=True)
                    for item in candidate_manifest
                ],
                "candidate_atomic_audit": candidate_audit.model_dump(
                    mode="json", exclude_none=True
                ),
                "reconciled_attributes": reconciled.model_dump(
                    mode="json", exclude_none=True
                ),
                "canonical_claims": [
                    card.model_dump(mode="json", exclude_none=True)
                    for card in sorted(
                        scoring_dossier.evidence,
                        key=lambda item: (item.claim_id or "", item.evidence_id),
                    )
                ],
            }
            reducer_json = json.dumps(
                reducer_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            reducer_sha256 = hashlib.sha256(reducer_json.encode("utf-8")).hexdigest()
            self.store.write_json(company_dir / "moat-reducer-input.json", reducer_payload)
            score_coverage = self._coverage(
                bundles,
                chunks,
                strength_chunks,
                total_evidence=len(structural_score_ids),
                selected_evidence=len(scoring_dossier.evidence),
            )
            if self.config.resume and score_path.is_file():
                score = MoatScore.model_validate(self.store.read_json(score_path))
            else:
                score = derive_audited_moat_score(
                    reconciled,
                    cards,
                    issuer_id=dossier.issuer_id,
                    as_of=self.config.as_of.date(),
                    document_coverage=score_coverage,
                )
            rank_score = derive_audited_moat_rank_score(
                rank_strength_assessment,
                reconciled,
                score_eligible=score.score_eligible,
            )
            score = score.model_copy(
                update={"economic_moat_rank_score": rank_score}
            )
            self.store.write_json(score_path, score)

            expected_claim_ids = {
                card.claim_id for card in scoring_dossier.evidence if card.claim_id
            }
            packed_claim_ids = set(pack.claim_ids)
            claim_union = expected_claim_ids | packed_claim_ids
            claim_jaccard = (
                len(expected_claim_ids & packed_claim_ids) / len(claim_union)
                if claim_union
                else 1.0
            )
            expected_counter_ids = {
                card.evidence_id
                for card in scoring_dossier.evidence
                if card.direction == EvidenceDirection.MOAT_NEGATIVE
            }
            packed_counter_ids = set(pack.counterevidence_ids)
            counter_recall = (
                len(expected_counter_ids & packed_counter_ids) / len(expected_counter_ids)
                if expected_counter_ids
                else 1.0
            )
            semantic_passed = (
                claim_jaccard == 1.0
                and counter_recall == 1.0
            )
            summary_manifest = self.store.read_json(company_dir / "section-summary-manifest.json")
            token_budget_audit = self.store.read_json(company_dir / "llm-token-budget.json")
            usage_so_far = self._sum_usage(audits)
            actual_provider_tokens = usage_so_far.input_tokens + usage_so_far.output_tokens
            token_budget_audit.update(
                {
                    "actual_llm_call_count": len(audits),
                    "actual_input_tokens": usage_so_far.input_tokens,
                    "actual_output_tokens": usage_so_far.output_tokens,
                    "actual_provider_tokens": actual_provider_tokens,
                    "actual_cached_input_tokens": usage_so_far.cached_input_tokens,
                    "actual_cache_write_tokens": usage_so_far.cache_write_tokens,
                    "actual_cache_read_fraction_of_input": (
                        usage_so_far.cached_input_tokens / usage_so_far.input_tokens
                        if usage_so_far.input_tokens
                        else 0.0
                    ),
                    "actual_cache_write_fraction_of_input": (
                        usage_so_far.cache_write_tokens / usage_so_far.input_tokens
                        if usage_so_far.input_tokens
                        else 0.0
                    ),
                    "actual_provider_tokens_per_call": (
                        actual_provider_tokens / len(audits) if audits else 0.0
                    ),
                }
            )
            self.store.write_json(company_dir / "llm-token-budget.json", token_budget_audit)
            compression_audit = {
                "schema_version": "compression-invariance/1",
                "passed": semantic_passed,
                "claim_jaccard": claim_jaccard,
                "counterevidence_recall": counter_recall,
                "moat_score_delta": None,
                "factor_scores_equal": None,
                "score_invariance_not_applicable": (
                    "economic strength is derived from the uncompressed broad contextual lane"
                ),
                "expected_claim_ids": sorted(expected_claim_ids),
                "packed_claim_ids": sorted(packed_claim_ids),
                "expected_counterevidence_ids": sorted(expected_counter_ids),
                "packed_counterevidence_ids": sorted(packed_counter_ids),
                "compact_pack_tokens_estimated": pack.token_count,
                "expanded_context_tokens_estimated": pack.expanded_context_token_count,
                "pack_token_reduction_fraction": pack.token_reduction_fraction,
                "strength_context_compressed": False,
                "strength_context_tokens_estimated": strength_context.token_count,
                "strength_selected_chunk_count": len(strength_context.selected_chunk_ids),
                "summary_llm_calls_avoided": summary_manifest["llm_calls_avoided"],
                "summary_input_tokens_avoided_estimated": summary_manifest[
                    "estimated_llm_input_tokens_avoided"
                ],
                "atomic_schema_token_reduction_fraction_estimated": token_budget_audit[
                    "schema_token_reduction_fraction"
                ],
                "atomic_schema_tokens_avoided_estimated": token_budget_audit[
                    "estimated_schema_tokens_avoided"
                ],
                "actual_llm_call_count": len(audits),
                "replayed_llm_call_count": sum(audit.replayed for audit in audits),
                "actual_input_tokens": usage_so_far.input_tokens,
                "actual_output_tokens": usage_so_far.output_tokens,
                "cached_input_tokens": usage_so_far.cached_input_tokens,
                "cache_write_tokens": usage_so_far.cache_write_tokens,
            }
            self.store.write_json(company_dir / "compression-audit.json", compression_audit)
            if not semantic_passed:
                raise ValueError(
                    "audit-lane compression gate failed: claims or counterevidence changed"
                )

            metamorphic_audit = audit_company_metamorphs(
                company_dir,
                issuer_id=dossier.issuer_id,
                maximum_atomic_units=self.config.maximum_atomic_evidence_units,
            )
            self.store.write_json(company_dir / "metamorphic-audit.json", metamorphic_audit)
            if metamorphic_audit["passed"] is not True:
                raise ValueError(
                    "MOAT metamorphic gate failed: "
                    + "; ".join(str(item) for item in metamorphic_audit["failures"])
                )

            self.store.write_json(
                company_dir / "run-manifest.json",
                RunManifest(
                    run_id=f"{self.config.run_id}:{company.ticker}",
                    signal_at=self.config.as_of,
                    evidence_cutoff=self.config.as_of,
                    model=f"{self.config.moat_model}+deterministic-python",
                    parser_version=",".join(sorted({bundle.metadata.parser_version for bundle in bundles})),
                    renderer_version="canonical-markdown/1",
                    prompt_version="dual-lane-moat/1",
                    token_budget=self.config.strength_context_tokens,
                    input_tokens=min(
                        strength_context.token_count,
                        self.config.strength_context_tokens,
                    ),
                    input_sha256=reducer_sha256,
                    temperature=0.0,
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
                selected_chunk_count=len(evidence_chunks),
                strength_context_chunk_count=len(strength_context.selected_chunk_ids),
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
        *,
        issuer_id: str | None,
    ) -> list[EvidenceCard]:
        path = company_dir / "evidence.jsonl"
        checkpoint_dir = company_dir / "atomic-judgment-by-key"
        bundle_by_document = {bundle.ast.document_id: bundle for bundle in bundles}
        cards: list[EvidenceCard] = []
        for chunk in sorted(chunks, key=lambda item: str(item.metadata["atomic_evidence_key"])):
            atomic_key = str(chunk.metadata["atomic_evidence_key"])
            checkpoint = checkpoint_dir / f"{atomic_key}.json"
            judgment: AtomicEvidenceJudgment | None = None
            if self.config.resume and checkpoint.is_file():
                candidate = AtomicEvidenceJudgment.model_validate(self.store.read_json(checkpoint))
                validation_errors = self._validate_atomic_judgment(
                    candidate,
                    chunk,
                    bundle_by_document[chunk.document_id],
                    issuer_id=issuer_id,
                )
                if not validation_errors:
                    judgment = candidate
            if judgment is None:
                request = build_atomic_evidence_request(chunk, issuer_id=issuer_id)
                extraction = self._execute_validated(
                    request,
                    AtomicEvidenceExtraction,
                    lambda _value: [],
                    audits,
                    company_dir,
                )
                extraction, repair_actions = normalize_atomic_extraction(extraction)
                judgment = atomic_extraction_to_judgment(extraction, chunk)
                validation_errors = self._validate_atomic_judgment(
                    judgment,
                    chunk,
                    bundle_by_document[chunk.document_id],
                    issuer_id=issuer_id,
                )
                repair_payload: dict[str, object] = {
                    "schema_version": "atomic-field-repair/1",
                    "atomic_evidence_key": atomic_key,
                    "strategy": "DETERMINISTIC_FIELD_REPAIR_NO_COMPANY_RETRY",
                    "actions": repair_actions,
                    "validation_errors": validation_errors,
                    "item_accepted": not validation_errors,
                }
                if validation_errors:
                    judgment = AtomicEvidenceJudgment(
                        is_investment_relevant=False,
                        fact="Atomic item excluded after deterministic validation.",
                    )
                    repair_payload["fallback"] = "EXCLUDE_INVALID_ATOMIC_ITEM"
                if repair_actions or validation_errors:
                    self.store.write_json(
                        company_dir / "atomic-repair-by-key" / f"{atomic_key}.json",
                        repair_payload,
                    )
            self.store.write_json(checkpoint, judgment)
            if judgment.is_investment_relevant:
                cards.append(
                    self._card_from_atomic_judgment(
                        judgment,
                        chunk,
                        issuer_id=issuer_id,
                    )
                )
            self.store.write_jsonl(path, cards)
        return sorted({card.evidence_id: card for card in cards}.values(), key=lambda card: card.evidence_id)

    def _card_from_atomic_judgment(
        self,
        judgment: AtomicEvidenceJudgment,
        chunk: SemanticChunk,
        *,
        issuer_id: str | None,
    ) -> EvidenceCard:
        return atomic_judgment_to_card(judgment, chunk, issuer_id=issuer_id)

    def _validate_atomic_judgment(
        self,
        judgment: AtomicEvidenceJudgment,
        chunk: SemanticChunk,
        bundle: CanonicalDocumentBundle,
        *,
        issuer_id: str | None,
    ) -> list[str]:
        if not judgment.is_investment_relevant:
            return []
        errors: list[str] = []
        if judgment.claim_signature is None:
            errors.append("relevant atomic evidence requires a canonical claim_signature")
        try:
            card = self._card_from_atomic_judgment(judgment, chunk, issuer_id=issuer_id)
            result = EvidenceExtractionResult(chunk_id=chunk.chunk_id, cards=[card])
            errors.extend(
                validate_evidence_result(
                    result,
                    chunk,
                    bundle,
                    discard_invalid_cards=False,
                )
            )
        except ValueError as exc:
            errors.append(str(exc))
        return errors

    @staticmethod
    def _evidence_collision_key(card: EvidenceCard) -> tuple[int, str, str, str]:
        # One verbatim source span is one evidence unit. Duplicate model
        # annotations are collapsed without depending on output order.
        direction_priority = {
            EvidenceDirection.MOAT_NEGATIVE: 0,
            EvidenceDirection.MOAT_POSITIVE: 1,
            EvidenceDirection.NEUTRAL: 2,
        }
        return (
            direction_priority[card.direction],
            card.evidence_type.value,
            card.statement_type.value,
            card.fact,
        )

    def _load_or_summarize(
        self,
        company_dir: Path,
        chunks: list[SemanticChunk],
        cards: list[EvidenceCard],
        audits: list[LLMCallAudit],
    ) -> list[SectionSummary]:
        _ = audits
        path = company_dir / "section-summaries.json"
        path_by_chunk = {chunk.chunk_id: chunk.section_path for chunk in chunks}
        summaries = build_evidence_preserving_summaries(
            cards,
            section_path_by_chunk=path_by_chunk,
            consolidate=self.config.consolidate_section_summaries,
        )
        self.store.write_json(path, summaries)

        # Estimate the input eliminated by replacing the former generative
        # summary pass. This is deliberately local and conservative; exact API
        # usage remains sourced from provider response usage fields.
        token_counter = HeuristicTokenCounter()
        card_by_id = {card.evidence_id: card for card in cards}
        manifest_items: list[dict[str, Any]] = []
        estimated_avoided = 0
        for summary in summaries:
            evidence_ids = list(
                dict.fromkeys(
                    [*summary.positive_evidence_ids, *summary.negative_evidence_ids]
                    + [
                        evidence_id
                        for claim in [
                            *summary.key_mechanisms,
                            *summary.key_kpis,
                            *summary.uncertainties,
                        ]
                        for evidence_id in claim.evidence_ids
                    ]
                )
            )
            section_cards = [card_by_id[item] for item in evidence_ids if item in card_by_id]
            avoided = 0
            if section_cards:
                legacy_request = build_section_summary_request(summary.section_path, section_cards)
                avoided = token_counter.count(
                    legacy_request.system
                    + legacy_request.user
                    + json.dumps(legacy_request.response_schema, ensure_ascii=False, sort_keys=True)
                )
                estimated_avoided += avoided
            manifest_items.append(
                {
                    "summary_id": stable_id("SS", *summary.section_path, *sorted(evidence_ids)),
                    "summary_version": "canonical-evidence-summary/1",
                    "as_of_date": self.config.as_of.date().isoformat(),
                    "source_evidence_ids": sorted(evidence_ids),
                    "prompt_version": None,
                    "model_snapshot": None,
                    "generator": "deterministic-python",
                    "estimated_llm_input_tokens_avoided": avoided,
                }
            )
        self.store.write_json(
            company_dir / "section-summary-manifest.json",
            {
                "schema_version": "section-summary-manifest/1",
                "llm_calls_avoided": len(summaries),
                "estimated_llm_input_tokens_avoided": estimated_avoided,
                "summaries": manifest_items,
            },
        )
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
        value_fields = (
            "base_revenue",
            "revenue_growth",
            "ebit_margin",
            "tax_rate",
            "depreciation_pct_revenue",
            "capex_pct_revenue",
            "nwc_pct_revenue",
            "wacc",
            "terminal_growth",
            "net_debt",
            "diluted_shares",
        )
        assumption_payload = assumptions.model_dump(mode="json")
        valuation_items = [
            {
                "name": field,
                "value": assumption_payload[field],
                "assumption_type": assumptions.type_for(field).value,
                "supporting_evidence_ids_or_source_refs": assumptions.assumption_sources.get(field, []),
            }
            for field in value_fields
        ]
        valuation_summary = {
            "schema_version": "valuation-summary/1",
            "as_of": self.config.as_of.isoformat(),
            "calculation": "deterministic-python",
            "items": valuation_items,
        }
        self.store.write_json(company_dir / "valuation-summary.json", valuation_summary)
        valuation_lines = [
            "# VALUATION SUMMARY",
            f"as_of={self.config.as_of.isoformat()}",
            "calculation=deterministic-python",
            "format=ASSUMPTION|VALUE|TYPE|SOURCES",
            "",
            *[
                f"{item['name']}|{item['value']}|{item['assumption_type']}|"
                f"src={','.join(item['supporting_evidence_ids_or_source_refs'])}"
                for item in valuation_items
            ],
        ]
        self.store.write_text(
            company_dir / "valuation-summary.md",
            "\n".join(valuation_lines).rstrip() + "\n",
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
            replayed = False
            replay_cache_key: str | None = None
            if self.replay_cache is None:
                result: TransportResult[ResponseT] = self.transport.execute(current, response_model)
                errors = list(validator(result.parsed))
            else:
                with self.replay_cache.locked(current, response_model) as replay_cache_key:
                    _, cached = self.replay_cache.load(current, response_model)
                    if cached is not None:
                        cached_errors = list(validator(cached.parsed))
                        if cached_errors:
                            self.replay_cache.discard(current.task, replay_cache_key)
                        else:
                            result = cached
                            errors = []
                            replayed = True
                    if not replayed:
                        result = self.transport.execute(current, response_model)
                        errors = list(validator(result.parsed))
                        if not errors:
                            self.replay_cache.store(current, response_model, result)
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
                stable_id(
                    "LLM",
                    current.task.value,
                    current.input_sha256,
                    attempt,
                    raw_response_sha256,
                )
                + ".json"
            )
            self.store.write_json(
                raw_path,
                {
                    "task": current.task.value,
                    "input_sha256": current.input_sha256,
                    "model": result.model,
                    "provider": result.provider,
                    "response_id": result.response_id,
                    "prompt_cache_key": current.prompt_cache_key,
                    "raw_output_text": raw_output,
                    "raw_response_sha256": raw_response_sha256,
                    "normalized_output": result.parsed,
                    "normalized_output_sha256": normalized_output_sha256,
                    "replayed": replayed,
                    "replay_cache_key": replay_cache_key,
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
                    prompt_cache_key=current.prompt_cache_key,
                    replayed=replayed,
                    replay_cache_key=replay_cache_key,
                )
            )
            self.store.write_jsonl(company_dir / "llm-calls.jsonl", audits)
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
        selected_cards = sorted(
            (card for card in dossier.evidence if card.evidence_id in selected_ids),
            key=lambda card: (card.claim_id or "", card.evidence_id),
        )
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
                build_atomic_evidence_request,
                build_contextual_moat_strength_request,
                build_section_summary_request,
                build_evidence_preserving_summaries,
                EvidencePackBuilder.build,
                MoatStrengthContextBuilder.build,
                build_financial_feature_vector,
                normalize_contextual_moat_assessment,
                normalize_contextual_moat_rank_assessment,
                reconcile_context_and_claims,
                derive_audited_moat_rank_score,
                derive_audited_moat_score,
            )
        )
        schema_contract = json.dumps(
            {
                "atomic_extraction": AtomicEvidenceExtraction.model_json_schema(),
                "atomic_judgment": AtomicEvidenceJudgment.model_json_schema(),
                "contextual_moat_assessment": ContextualMoatAssessment.model_json_schema(),
                "reconciled_moat_assessment": ReconciledMoatAssessment.model_json_schema(),
                "section_summary": SectionSummary.model_json_schema(),
                "moat_score": MoatScore.model_json_schema(),
            },
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
                    "atomic_reasoning_effort": self.config.atomic_reasoning_effort,
                    "moat_reasoning_effort": self.config.moat_reasoning_effort,
                    "context_tokens": self.config.context_tokens,
                    "prompt_reserve_tokens": self.config.prompt_reserve_tokens,
                    "strength_context_tokens": self.config.strength_context_tokens,
                    "strength_prompt_reserve_tokens": (
                        self.config.strength_prompt_reserve_tokens
                    ),
                    "max_output_tokens": self.config.max_output_tokens,
                    "minimum_text_retention": self.config.minimum_text_retention,
                    "minimum_numeric_retention": self.config.minimum_numeric_retention,
                    "minimum_structured_fact_retention": self.config.minimum_structured_fact_retention,
                    "require_table_count_match": self.config.require_table_count_match,
                    "require_financial_table_semantics": self.config.require_financial_table_semantics,
                    "allow_low_quality": self.config.allow_low_quality,
                    "maximum_price_age_days": self.config.maximum_price_age_days,
                    "maximum_atomic_evidence_units": self.config.maximum_atomic_evidence_units,
                    "consolidate_section_summaries": self.config.consolidate_section_summaries,
                    "validation_attempts": self.config.validation_attempts,
                    "experiment_id": self.config.experiment_id,
                    "llm_replay_enabled": bool(self.config.llm_replay_cache_directory),
                    "evidence_ledger_enabled": bool(self.config.evidence_ledger_directory),
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

    def _refresh_atomic_token_budget(
        self,
        company_dir: Path,
        requests: list[LLMRequest],
        units: list[SemanticChunk],
    ) -> None:
        path = company_dir / "llm-token-budget.json"
        payload = self.store.read_json(path)
        counter = HeuristicTokenCounter()
        schema_json = json.dumps(
            AtomicEvidenceExtraction.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        static_tokens = (
            counter.count(requests[0].system + schema_json) if requests else 0
        )
        dynamic_tokens = sum(counter.count(request.user) for request in requests)
        source_tokens = sum(counter.count(unit.markdown) for unit in units)
        total_tokens = static_tokens * len(requests) + dynamic_tokens
        strength_static_tokens = int(
            payload.get("estimated_strength_static_prefix_tokens", 0)
        )
        strength_dynamic_tokens = int(
            payload.get("estimated_strength_dynamic_suffix_tokens", 0)
        )
        strength_breakpoints = int(payload.get("strength_request_count", 0))
        payload.update(
            {
                "atomic_request_count": len(requests),
                "estimated_atomic_input_tokens": total_tokens,
                "estimated_static_prefix_tokens_per_request": static_tokens,
                "estimated_dynamic_suffix_tokens_total": dynamic_tokens,
                "estimated_source_evidence_tokens_total": source_tokens,
                "estimated_useful_source_token_ratio": (
                    source_tokens / total_tokens if total_tokens else 0.0
                ),
                "prompt_cache_breakpoint_count": sum(
                    request.prompt_cache_breakpoint for request in requests
                ) + strength_breakpoints,
                "estimated_prefix_cache_eligible": static_tokens >= 1_024,
                "estimated_cacheable_prefix_tokens_total": (
                    static_tokens * len(requests) if static_tokens >= 1_024 else 0
                )
                + (strength_static_tokens if strength_static_tokens >= 1_024 else 0),
                "estimated_dynamic_suffix_tokens_all_tasks": (
                    dynamic_tokens + strength_dynamic_tokens
                ),
                "expected_output_token_cap_total": (
                    len(requests) * min(self.config.max_output_tokens, 2_000)
                    + min(self.config.max_output_tokens, 8_000)
                ),
                "atomic_selection_policy": "BASELINE_PLUS_CONTEXT_CITATION_AUDIT",
            }
        )
        self.store.write_json(path, payload)

    @staticmethod
    def _sum_usage(audits: list[LLMCallAudit]) -> TransportUsage:
        return TransportUsage(
            input_tokens=sum(audit.usage.input_tokens for audit in audits),
            output_tokens=sum(audit.usage.output_tokens for audit in audits),
            cached_input_tokens=sum(audit.usage.cached_input_tokens for audit in audits),
            cache_write_tokens=sum(audit.usage.cache_write_tokens for audit in audits),
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
