from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from moatrader.canonical.models import SourceType, StatementType
from moatrader.evidence.models import (
    AtomicEvidenceExtraction,
    CandidateAtomicAuditDecision,
    CandidateAtomicAuditResult,
    CandidateAuditReason,
    CandidateSupportStatus,
    CitedSummaryClaim,
    ContextualMechanismAssessment,
    ContextualMoatAssessment,
    CoverageMetrics,
    Durability,
    EconomicScope,
    EvidenceCard,
    EvidenceDirection,
    EvidenceExtractionResult,
    EvidenceType,
    MoatMechanismScore,
    MoatScore,
    SectionSummary,
)
from moatrader.llm import FunctionTransport, LLMRequest, LLMTask
from moatrader.runner import CompanyRunStatus, MoatUniverseRunner, UniverseRunConfig
from moatrader.universe import load_universe_manifest


ROOT = Path(__file__).resolve().parents[1]


def _fixture_handler(request: LLMRequest, _response_model: type[Any]) -> Any:
    if request.task == LLMTask.LOCAL_EVIDENCE_EXTRACTION:
        return AtomicEvidenceExtraction(
            is_investment_relevant=True,
            evidence_type=EvidenceType.SWITCHING_COST,
            fact="The disclosure supports a repeatable customer relationship.",
            mechanism=["workflow integration", "switching friction"],
            direction=EvidenceDirection.MOAT_POSITIVE,
            claim_subject="customer workflow",
            claim_predicate="switching friction",
            claim_horizon="LONG",
        )
    if request.task == LLMTask.CONTEXTUAL_MOAT_STRENGTH:
        return ContextualMoatAssessment(
            evidence_sufficiency=3,
            mechanisms=[
                ContextualMechanismAssessment(
                    evidence_type=EvidenceType.SWITCHING_COST,
                    strength_bucket=3,
                    scope_materiality_bucket=4,
                    durability_bucket=3,
                    economic_scope=EconomicScope.COMPANY,
                    reference_ids=[str(request.metadata["reference_ids"][0])],
                    rationale="Grounded context indicates switching friction.",
                )
            ],
        )
    if request.task == LLMTask.CANDIDATE_ATOMIC_AUDIT:
        candidate_id = str(request.metadata["candidate_ids"][0])
        evidence_ids = list(request.metadata["allowed_evidence_ids"][candidate_id])
        return CandidateAtomicAuditResult(
            decisions=[
                CandidateAtomicAuditDecision(
                    candidate_id=candidate_id,
                    support=CandidateSupportStatus.SUPPORTED,
                    reason=CandidateAuditReason.EXPLICIT_CAUSAL_BARRIER,
                    supporting_atomic_evidence_ids=evidence_ids,
                )
            ]
        )
    if request.task == LLMTask.SECTION_SUMMARY:
        evidence_ids = list(request.metadata["evidence_ids"])
        return SectionSummary(
            section_path=list(request.metadata["section_path"]),
            positive_evidence_ids=evidence_ids,
            key_mechanisms=[
                CitedSummaryClaim(
                    text="workflow integration raises switching friction",
                    evidence_ids=evidence_ids,
                )
            ],
        )
    if request.task == LLMTask.FINAL_MOAT_SCORING:
        evidence_ids = list(request.metadata["positive_evidence_ids"])
        negative_ids = list(request.metadata["negative_evidence_ids"])
        return MoatScore(
            issuer_id=str(request.metadata["issuer_id"]),
            as_of=date.fromisoformat(str(request.metadata["as_of"])[:10]),
            economic_moat_score=7.0,
            mechanisms=[
                MoatMechanismScore(
                    evidence_type=EvidenceType.SWITCHING_COST,
                    score=7.0,
                    evidence_ids=[evidence_ids[0]],
                    rationale="Validated evidence indicates switching friction.",
                )
            ],
            durability=Durability.MEDIUM_HIGH,
            model_confidence=0.8,
            document_coverage=CoverageMetrics(),
            counterevidence_ids=negative_ids[:1],
        )
    raise AssertionError(request.task)


def _config(run_id: str, *, resume: bool = False, dry_run: bool = False, workers: int = 1) -> UniverseRunConfig:
    return UniverseRunConfig(
        run_id=run_id,
        as_of=datetime.fromisoformat("2025-05-16T00:00:00+09:00"),
        resume=resume,
        dry_run=dry_run,
        workers=workers,
    )


def test_moat_model_latest_alias_is_rejected() -> None:
    with pytest.raises(ValueError, match="exact pinned model ID"):
        UniverseRunConfig(
            run_id="unpinned",
            as_of=datetime.fromisoformat("2025-05-16T00:00:00+09:00"),
            moat_model="gpt-5-chat-latest",
        )


def test_runner_completes_scores_dcf_ranking_and_manifest(tmp_path: Path) -> None:
    universe = load_universe_manifest(ROOT / "examples" / "universe.csv")
    runner = MoatUniverseRunner(
        config=_config("integration"),
        output_directory=tmp_path,
        transport=FunctionTransport(_fixture_handler),
    )

    result = runner.run(universe, universe.companies)

    company = result.companies[0]
    assert company.status == CompanyRunStatus.COMPLETE
    assert company.moat_score is not None
    assert company.moat_score.economic_moat_score == 3.75
    assert company.moat_score.economic_moat_rank_score == 5.625
    assert company.moat_score.llm_proposed_score is None
    assert company.dcf is not None
    assert company.valuation_as_of == datetime.fromisoformat("2025-05-16T00:00:00+09:00")
    assert result.ranking == []  # fixture DCF has intentionally insufficient provenance
    company_dir = tmp_path / "integration" / "companies" / "SAMPLE"
    assert (company_dir / "evidence-pack.md").is_file()
    assert (company_dir / "evidence-clusters.json").is_file()
    assert (company_dir / "claim-clusters.json").is_file()
    assert (company_dir / "metamorphic-audit.json").is_file()
    assert (company_dir / "compression-audit.json").is_file()
    assert (company_dir / "financial-feature-vector.json").is_file()
    assert (company_dir / "financial-feature-vector.md").is_file()
    raw_strength = json.loads(
        (company_dir / "contextual-moat-assessment-raw.json").read_text(encoding="utf-8")
    )
    public_strength = json.loads(
        (company_dir / "contextual-moat-assessment.json").read_text(encoding="utf-8")
    )
    assert raw_strength["mechanisms"][0]["strength_bucket"] == 3
    assert public_strength["mechanisms"][0]["strength_bucket"] == 2
    assert (company_dir / "contextual-moat-rank-field-repair.json").is_file()
    assert (company_dir / "section-summary-manifest.json").is_file()
    assert (company_dir / "llm-token-budget.json").is_file()
    assert (company_dir / "run-manifest.json").is_file()
    assert (company_dir / "dcf-assumptions.json").is_file()
    assert (company_dir / "dcf-manifest.json").is_file()
    assert (company_dir / "valuation-summary.json").is_file()
    assert (company_dir / "valuation-summary.md").is_file()
    raw_calls = list((company_dir / "llm-raw").glob("*.json"))
    assert raw_calls
    call_audit = [
        json.loads(line)
        for line in (company_dir / "llm-calls.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert all(item.get("raw_response_sha256") for item in call_audit)
    assert all(Path(item["raw_response_path"]).is_file() for item in call_audit)
    assert {item["task"] for item in call_audit} == {
        "LOCAL_EVIDENCE_EXTRACTION",
        "CONTEXTUAL_MOAT_STRENGTH",
    }
    summary_manifest = json.loads(
        (company_dir / "section-summary-manifest.json").read_text(encoding="utf-8")
    )
    assert summary_manifest["llm_calls_avoided"] == 1
    assert summary_manifest["summaries"][0]["generator"] == "deterministic-python"
    compression = json.loads((company_dir / "compression-audit.json").read_text(encoding="utf-8"))
    assert compression["passed"] is True
    assert compression["claim_jaccard"] == 1.0
    assert compression["counterevidence_recall"] == 1.0
    token_budget = json.loads((company_dir / "llm-token-budget.json").read_text(encoding="utf-8"))
    assert token_budget["schema_version"] == "llm-token-budget/4"
    assert token_budget["schema_token_reduction_fraction"] > 0
    assert token_budget["estimated_schema_tokens_avoided"] > 0
    assert token_budget["prompt_cache_mode"] == "explicit"
    assert token_budget["prompt_cache_breakpoint_count"] == (
        token_budget["atomic_request_count"] + token_budget["strength_request_count"]
    )
    assert token_budget["estimated_cacheable_prefix_tokens_total"] >= 0
    assert token_budget["estimated_dynamic_suffix_tokens_all_tasks"] > 0
    assert token_budget["expected_output_token_cap_total"] >= 8_000
    assert token_budget["strength_context_compression_ratio"] == 1.0
    assert token_budget["actual_llm_call_count"] == len(call_audit)
    assert token_budget["actual_provider_tokens_per_call"] >= 0
    assert token_budget["atomic_reasoning_effort"] == "medium"
    assert token_budget["atomic_max_output_tokens"] == 2_000
    assert token_budget["strength_request_count"] == 1
    assert token_budget["strength_compression_ablation_enabled"] is False
    assert token_budget["atomic_selection_policy"] == "BASELINE_PLUS_CONTEXT_CITATION_AUDIT"
    selection = json.loads(
        (company_dir / "evidence-chunk-selection.json").read_text(encoding="utf-8")
    )
    assert selection["method"] == "dual_lane_citation_audit/1"
    assert selection["parent_fallback_enabled"] is False
    dcf_manifest = json.loads((company_dir / "dcf-manifest.json").read_text(encoding="utf-8"))
    assert dcf_manifest["calculation_mode"] == "deterministic_python"
    assert dcf_manifest["llm_model"] is None
    assert dcf_manifest["method"] == "FCFF"
    assert dcf_manifest["assumption_confidence"] == "0.10"
    assert dcf_manifest["confidence_penalty"] == "0.90"
    dcf_payload = json.loads((company_dir / "dcf.json").read_text(encoding="utf-8"))
    assert dcf_payload["method"] == "FCFF"
    assert dcf_payload["assumptions"]["base_revenue"] == "1000"
    assert "terminal_value_share" in dcf_payload
    run_manifest = json.loads((company_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["model"] == "gpt-5.6-luna+deterministic-python"
    assert (company_dir / "moat-reducer-input.json").is_file()
    assert (company_dir / "moat-strength-context.md").is_file()
    assert (company_dir / "contextual-moat-assessment.json").is_file()
    assert (company_dir / "moat-reconciliation.json").is_file()
    assert all(item["task"] != "FINAL_MOAT_SCORING" for item in call_audit)


def test_runner_preserves_official_primary_document_url_in_provenance(tmp_path: Path) -> None:
    source = tmp_path / "source.xml"
    source.write_text("<DOCUMENT><BODY><P>Official source.</P></BODY></DOCUMENT>", encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "source_type": "DART",
                "rcept_no": "20250101000001",
                "corp_code": "00126380",
                "available_at": "2025-01-01T23:59:59+09:00",
                "primary_document_url": "https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20250101000001",
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "universe.csv"
    manifest_path.write_text(
        "ticker,source,input,metadata,issuer_id,issuer_name\n"
        f"SAMPLE,DART,{source},{metadata},00126380,Sample\n",
        encoding="utf-8",
    )
    universe = load_universe_manifest(manifest_path)
    runner = MoatUniverseRunner(
        config=UniverseRunConfig(
            run_id="official-uri",
            as_of=datetime.fromisoformat("2025-01-02T00:00:00+09:00"),
            dry_run=True,
        ),
        output_directory=tmp_path / "runs",
        transport=None,
    )

    result = runner.run(universe, universe.companies)

    bundle_path = next((tmp_path / "runs" / "official-uri" / "companies" / "SAMPLE" / "documents").glob("*/bundle.json"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    source_ref = next(iter(bundle["provenance"]["records"].values()))["source_refs"][0]
    assert source_ref["uri"].startswith("https://dart.fss.or.kr/")


def test_runner_dry_run_makes_no_transport_calls(tmp_path: Path) -> None:
    universe = load_universe_manifest(ROOT / "examples" / "universe.csv")

    result = MoatUniverseRunner(
        config=_config("dry", dry_run=True),
        output_directory=tmp_path,
        transport=None,
    ).run(universe, universe.companies)

    assert result.companies[0].status == CompanyRunStatus.PREPARED
    assert result.companies[0].dcf is not None
    assert (tmp_path / "dry" / "companies" / "SAMPLE" / "evidence-requests.jsonl").is_file()


def test_runner_rejects_market_price_from_after_signal(tmp_path: Path) -> None:
    universe = load_universe_manifest(ROOT / "examples" / "universe.csv")
    universe.companies[0].price_as_of = datetime.fromisoformat("2025-05-17T00:00:00+09:00")

    result = MoatUniverseRunner(
        config=_config("future-price", dry_run=True),
        output_directory=tmp_path,
        transport=None,
    ).run(universe, universe.companies)

    assert result.companies[0].status == CompanyRunStatus.FAILED
    assert "after run as_of" in (result.companies[0].error or "")


def test_runner_rejects_stale_market_price(tmp_path: Path) -> None:
    universe = load_universe_manifest(ROOT / "examples" / "universe.csv")
    universe.companies[0].price_as_of = datetime.fromisoformat("2025-05-01T00:00:00+09:00")

    result = MoatUniverseRunner(
        config=_config("stale-price", dry_run=True),
        output_directory=tmp_path,
        transport=None,
    ).run(universe, universe.companies)

    assert result.companies[0].status == CompanyRunStatus.FAILED
    assert "older than" in (result.companies[0].error or "")


def test_runner_resume_reuses_completed_result(tmp_path: Path) -> None:
    universe = load_universe_manifest(ROOT / "examples" / "universe.csv")
    MoatUniverseRunner(
        config=_config("resume"),
        output_directory=tmp_path,
        transport=FunctionTransport(_fixture_handler),
    ).run(universe, universe.companies)

    def fail_if_called(_request: LLMRequest, _model: type[Any]) -> Any:
        raise AssertionError("transport should not be called for a completed checkpoint")

    resumed = MoatUniverseRunner(
        config=_config("resume", resume=True),
        output_directory=tmp_path,
        transport=FunctionTransport(fail_if_called),
    ).run(universe, universe.companies)

    assert resumed.companies[0].status == CompanyRunStatus.COMPLETE


def test_experiment_replay_reuses_validated_outputs_across_run_ids(tmp_path: Path) -> None:
    universe = load_universe_manifest(ROOT / "examples" / "universe.csv")
    cache = tmp_path / "replay"
    first_config = _config("replay-first").model_copy(
        update={
            "experiment_id": "experiment-a",
            "llm_replay_cache_directory": str(cache),
        }
    )
    first = MoatUniverseRunner(
        config=first_config,
        output_directory=tmp_path,
        transport=FunctionTransport(_fixture_handler),
    ).run(universe, universe.companies)

    def fail_if_called(_request: LLMRequest, _model: type[Any]) -> Any:
        raise AssertionError("validated response should be replayed")

    second_config = _config("replay-second").model_copy(
        update={
            "experiment_id": "experiment-a",
            "llm_replay_cache_directory": str(cache),
        }
    )
    second = MoatUniverseRunner(
        config=second_config,
        output_directory=tmp_path,
        transport=FunctionTransport(fail_if_called),
    ).run(universe, universe.companies)

    assert first.companies[0].moat_score == second.companies[0].moat_score
    audit_path = tmp_path / "replay-second" / "companies" / "SAMPLE" / "llm-calls.jsonl"
    calls = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert calls
    assert all(call["replayed"] is True for call in calls)
    assert all(call["replay_cache_key"] for call in calls)

    fresh_calls = 0

    def count_fresh_calls(request: LLMRequest, response_model: type[Any]) -> Any:
        nonlocal fresh_calls
        fresh_calls += 1
        return _fixture_handler(request, response_model)

    isolated_config = _config("replay-isolated").model_copy(
        update={
            "experiment_id": "experiment-b",
            "llm_replay_cache_directory": str(cache),
        }
    )
    MoatUniverseRunner(
        config=isolated_config,
        output_directory=tmp_path,
        transport=FunctionTransport(count_fresh_calls),
    ).run(universe, universe.companies)
    assert fresh_calls > 0


def test_resume_signature_rejects_changed_market_snapshot(tmp_path: Path) -> None:
    universe = load_universe_manifest(ROOT / "examples" / "universe.csv")
    MoatUniverseRunner(
        config=_config("resume-price"),
        output_directory=tmp_path,
        transport=FunctionTransport(_fixture_handler),
    ).run(universe, universe.companies)
    universe.companies[0].current_price = universe.companies[0].current_price + 1

    resumed = MoatUniverseRunner(
        config=_config("resume-price", resume=True),
        output_directory=tmp_path,
        transport=FunctionTransport(_fixture_handler),
    ).run(universe, universe.companies)

    assert resumed.companies[0].status == CompanyRunStatus.FAILED
    assert "signature mismatch" in (resumed.companies[0].error or "")


def test_dry_run_can_resume_into_real_execution(tmp_path: Path) -> None:
    universe = load_universe_manifest(ROOT / "examples" / "universe.csv")
    prepared = MoatUniverseRunner(
        config=_config("prepared-resume", dry_run=True),
        output_directory=tmp_path,
        transport=None,
    ).run(universe, universe.companies)
    assert prepared.companies[0].status == CompanyRunStatus.PREPARED

    completed = MoatUniverseRunner(
        config=_config("prepared-resume", resume=True),
        output_directory=tmp_path,
        transport=FunctionTransport(_fixture_handler),
    ).run(universe, universe.companies)

    assert completed.companies[0].status == CompanyRunStatus.COMPLETE


def test_partial_evidence_checkpoints_resume_per_chunk(tmp_path: Path) -> None:
    universe = load_universe_manifest(ROOT / "examples" / "universe.csv")
    successful_chunk: list[str] = []
    local_calls = 0

    def fail_on_second_chunk(request: LLMRequest, response_model: type[Any]) -> Any:
        nonlocal local_calls
        if request.task == LLMTask.LOCAL_EVIDENCE_EXTRACTION:
            local_calls += 1
            if local_calls == 2:
                raise RuntimeError("interrupted fixture")
            successful_chunk.append(str(request.metadata["chunk_id"]))
        return _fixture_handler(request, response_model)

    failed = MoatUniverseRunner(
        config=_config("partial"),
        output_directory=tmp_path,
        transport=FunctionTransport(fail_on_second_chunk),
    ).run(universe, universe.companies)
    assert failed.companies[0].status == CompanyRunStatus.FAILED

    resumed_local_chunks: list[str] = []

    def record_resume(request: LLMRequest, response_model: type[Any]) -> Any:
        if request.task == LLMTask.LOCAL_EVIDENCE_EXTRACTION:
            resumed_local_chunks.append(str(request.metadata["chunk_id"]))
        return _fixture_handler(request, response_model)

    resumed = MoatUniverseRunner(
        config=_config("partial", resume=True),
        output_directory=tmp_path,
        transport=FunctionTransport(record_resume),
    ).run(universe, universe.companies)

    assert resumed.companies[0].status == CompanyRunStatus.COMPLETE
    assert successful_chunk[0] not in resumed_local_chunks
    assert resumed_local_chunks


def test_canonical_claim_set_is_not_pruned_by_llm_context_budget(tmp_path: Path) -> None:
    universe = load_universe_manifest(ROOT / "examples" / "universe.csv")

    config = _config("pruning")
    config.context_tokens = 9_000
    config.prompt_reserve_tokens = 1_000
    result = MoatUniverseRunner(
        config=config,
        output_directory=tmp_path,
        transport=FunctionTransport(_fixture_handler),
    ).run(universe, universe.companies)

    company = result.companies[0]
    assert company.status == CompanyRunStatus.COMPLETE
    assert company.moat_score is not None
    assert company.moat_score.document_coverage.evidence_retention is not None
    assert company.moat_score.document_coverage.evidence_retention == 1
    pruning = json.loads(
        (tmp_path / "pruning" / "companies" / "SAMPLE" / "evidence-pruning.json").read_text(encoding="utf-8")
    )
    assert pruning["pruned"] is False
    assert pruning["dropped_evidence_ids"] == []


def test_company_failure_does_not_abort_other_tickers(tmp_path: Path) -> None:
    source = ROOT / "examples" / "sample-dart.html"
    metadata = ROOT / "examples" / "sample-dart-metadata.json"
    good_dcf = ROOT / "examples" / "sample-dcf-assumptions.json"
    bad_dcf = tmp_path / "bad-dcf.json"
    bad_dcf.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "universe.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "ticker",
                "source",
                "input",
                "metadata",
                "issuer_id",
                "issuer_name",
                "current_price",
                "price_as_of",
                "dcf_assumptions",
            ]
        )
        writer.writerow(["GOOD", "DART", source, metadata, "GOOD", "Good", "10", "2025-05-15T16:00:00+09:00", good_dcf])
        writer.writerow(["BAD", "DART", source, metadata, "BAD", "Bad", "10", "2025-05-15T16:00:00+09:00", bad_dcf])
    universe = load_universe_manifest(manifest_path)

    result = MoatUniverseRunner(
        config=_config("isolation", workers=2),
        output_directory=tmp_path,
        transport=FunctionTransport(_fixture_handler),
    ).run(universe, universe.companies)

    statuses = {company.ticker: company.status for company in result.companies}
    assert statuses == {"GOOD": CompanyRunStatus.COMPLETE, "BAD": CompanyRunStatus.FAILED}
    assert result.failed_count == 1


def test_failure_before_company_signature_is_isolated(tmp_path: Path) -> None:
    universe = load_universe_manifest(ROOT / "examples" / "universe.csv")
    universe.companies[0].documents[0].input_path = str(tmp_path / "disappeared.html")

    result = MoatUniverseRunner(
        config=_config("early-failure"),
        output_directory=tmp_path,
        transport=FunctionTransport(_fixture_handler),
    ).run(universe, universe.companies)

    assert result.companies[0].status == CompanyRunStatus.FAILED
    assert "FileNotFoundError" in (result.companies[0].error or "")


def test_empty_financial_snapshot_hard_fails_before_dcf_output(tmp_path: Path) -> None:
    source = tmp_path / "no-financials.html"
    source.write_text("<html><body><p>Only qualitative disclosure.</p></body></html>", encoding="utf-8")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "rcept_no": "20250515000123",
                "corp_code": "00126380",
                "issuer_name": "No Financials",
                "available_at": "2025-05-15T09:01:02+09:00",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "empty-universe.csv"
    manifest.write_text(
        "ticker,source,input,metadata,current_price,price_as_of,dcf_assumptions\n"
        f"EMPTY,DART,{source},{metadata},10,2025-05-15T16:00:00+09:00,"
        f"{ROOT / 'examples' / 'sample-dcf-assumptions.json'}\n",
        encoding="utf-8",
    )
    universe = load_universe_manifest(manifest)

    result = MoatUniverseRunner(
        config=_config("empty-hard-fail", dry_run=True),
        output_directory=tmp_path,
        transport=None,
    ).run(universe, universe.companies)

    company = result.companies[0]
    assert company.status == CompanyRunStatus.FAILED
    assert "DCF hard fail" in (company.error or "")
    assert not (tmp_path / "empty-hard-fail" / "companies" / "EMPTY" / "dcf.json").exists()
