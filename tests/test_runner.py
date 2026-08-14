from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from moatrader.canonical.models import SourceType, StatementType
from moatrader.evidence.models import (
    CitedSummaryClaim,
    CoverageMetrics,
    Durability,
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
        chunk_id = str(request.metadata["chunk_id"])
        return EvidenceExtractionResult(
            chunk_id=chunk_id,
            cards=[
                EvidenceCard(
                    evidence_id="pending",
                    source_chunk_id=chunk_id,
                    node_ids=[str(request.metadata["node_ids"][0])],
                    evidence_type=EvidenceType.SWITCHING_COST,
                    statement_type=StatementType.DISCLOSED_FACT,
                    fact="The disclosure supports a repeatable customer relationship.",
                    mechanism=["workflow integration", "switching friction"],
                    direction=EvidenceDirection.MOAT_POSITIVE,
                    strength=0.7,
                    source_type=SourceType(str(request.metadata["source_type"])),
                    reliability=0.8,
                )
            ],
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
        evidence_ids = list(request.metadata["evidence_ids"])
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
    assert company.moat_score.economic_moat_score == 7.0
    assert company.dcf is not None
    assert company.valuation_as_of == datetime.fromisoformat("2025-05-16T00:00:00+09:00")
    assert result.ranking and result.ranking[0].ticker == "SAMPLE"
    company_dir = tmp_path / "integration" / "companies" / "SAMPLE"
    assert (company_dir / "evidence-pack.md").is_file()
    assert (company_dir / "evidence-clusters.json").is_file()
    assert (company_dir / "run-manifest.json").is_file()
    assert (company_dir / "dcf-assumptions.json").is_file()
    assert (company_dir / "dcf-manifest.json").is_file()
    dcf_manifest = json.loads((company_dir / "dcf-manifest.json").read_text(encoding="utf-8"))
    assert dcf_manifest["calculation_mode"] == "deterministic_python"
    assert dcf_manifest["llm_model"] is None
    run_manifest = json.loads((company_dir / "run-manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["input_sha256"] == json.loads(
        (company_dir / "moat-request.json").read_text(encoding="utf-8")
    )["input_sha256"]


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
    assert len(resumed_local_chunks) == 1


def test_large_evidence_layer_is_pruned_to_context_budget(tmp_path: Path) -> None:
    universe = load_universe_manifest(ROOT / "examples" / "universe.csv")

    def verbose_handler(request: LLMRequest, response_model: type[Any]) -> Any:
        if request.task != LLMTask.LOCAL_EVIDENCE_EXTRACTION:
            return _fixture_handler(request, response_model)
        chunk_id = str(request.metadata["chunk_id"])
        cards = []
        for index in range(24):
            cards.append(
                EvidenceCard(
                    evidence_id=f"pending-{index}",
                    source_chunk_id=chunk_id,
                    node_ids=[str(request.metadata["node_ids"][0])],
                    evidence_type=EvidenceType.SWITCHING_COST,
                    statement_type=StatementType.DISCLOSED_FACT,
                    fact=("Detailed grounded competitive evidence and operating context. " * 25) + str(index),
                    mechanism=["workflow integration", "qualification period", "switching friction"],
                    direction=(
                        EvidenceDirection.MOAT_NEGATIVE if index % 7 == 0 else EvidenceDirection.MOAT_POSITIVE
                    ),
                    strength=0.6,
                    source_type=SourceType(str(request.metadata["source_type"])),
                    reliability=0.7,
                )
            )
        return EvidenceExtractionResult(chunk_id=chunk_id, cards=cards)

    config = _config("pruning")
    config.context_tokens = 9_000
    config.prompt_reserve_tokens = 1_000
    result = MoatUniverseRunner(
        config=config,
        output_directory=tmp_path,
        transport=FunctionTransport(verbose_handler),
    ).run(universe, universe.companies)

    company = result.companies[0]
    assert company.status == CompanyRunStatus.COMPLETE
    assert company.moat_score is not None
    assert company.moat_score.document_coverage.evidence_retention is not None
    assert company.moat_score.document_coverage.evidence_retention < 1
    pruning = json.loads(
        (tmp_path / "pruning" / "companies" / "SAMPLE" / "evidence-pruning.json").read_text(encoding="utf-8")
    )
    assert pruning["pruned"] is True
    assert pruning["selected_tokens"] <= pruning["target_tokens"]


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
