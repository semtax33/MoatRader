from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from moatrader.adapters import PaddlePdfOcrAdapter, RawDocument
from moatrader.backtest import (
    BacktestConfig,
    PointInTimeBacktester,
    PricePanel,
    equity_csv,
    load_price_panel,
    rebalances_csv,
)
from moatrader.canonical.models import CanonicalDocumentBundle
from moatrader.ingestion import (
    DEFAULT_SEC_FORMS,
    BronzeFilingStore,
    DartCollector,
    DartOpenApiClient,
    KindCompanyIdentity,
    KindIrClient,
    KindIrCollector,
    ResilientHttpClient,
    SecEdgarClient,
    SecEdgarCollector,
    write_collected_universe_manifest,
)
from moatrader.llm import OpenAIResponsesTransport
from moatrader.pipeline import CanonicalFinancialDocumentPipeline
from moatrader.preflight import (
    find_workspace_manifest,
    validate_preflight_approval,
    validate_preflight_sample_selection,
)
from moatrader.runner import MoatUniverseRunner, UniverseRunConfig, UniverseRunResult
from moatrader.runner.engine import RUNNER_VERSION
from moatrader.runner.report import (
    opportunities_csv,
    rank_expectation_opportunities,
    rank_run_result,
    ranking_csv,
)
from moatrader.runstore import RunStore
from moatrader.screening import SelectorConfig
from moatrader.universe import load_universe_manifest


def _preflight_universe_tickers(
    workspace_manifest: Path | None,
    fallback_tickers: list[str],
) -> list[str]:
    """Return the experiment universe, including companies with no PIT filing.

    Date manifests intentionally omit companies for which no source document was
    available.  The preflight approval, however, covers the original experiment
    universe so the final signal panel can retain those companies as
    ``NO_PIT_DOCUMENT`` rows.
    """

    if workspace_manifest is None:
        return fallback_tickers
    universe_path = workspace_manifest.parent / "inputs" / "universe.csv"
    if not universe_path.is_file():
        return fallback_tickers
    with universe_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    tickers = [
        str(row.get("stock_code") or row.get("ticker") or "").strip().zfill(6)
        for row in rows
    ]
    return [ticker for ticker in tickers if ticker] or fallback_tickers


def _ir_ocr_adapter(args: argparse.Namespace) -> PaddlePdfOcrAdapter | None:
    engine = str(getattr(args, "ir_ocr_engine", "none") or "none").casefold()
    if engine == "none":
        return None
    if engine != "paddle":
        raise ValueError(f"unsupported IR OCR engine: {engine}")
    return PaddlePdfOcrAdapter(
        device=str(getattr(args, "ir_ocr_device", "cpu")),
        cpu_threads=int(getattr(args, "ir_ocr_cpu_threads", 6)),
    )


def _ingest(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8-sig"))
    metadata["source_type"] = args.source.upper()
    raw = RawDocument(
        content=input_path.read_bytes(),
        uri=str(metadata.get("primary_document_url") or input_path.resolve().as_uri()),
        fetched_at=datetime.now(timezone.utc),
        media_type=(
            "application/xml"
            if input_path.suffix.lower() == ".xml"
            else "application/xhtml+xml"
            if input_path.suffix.lower() == ".xhtml"
            else "application/pdf"
            if input_path.suffix.lower() == ".pdf"
            else "text/html"
        ),
        hints=metadata,
    )
    prepared = CanonicalFinancialDocumentPipeline(
        ir_ocr_adapter=_ir_ocr_adapter(args)
    ).prepare_for_llm(
        raw,
        model_context_tokens=args.context_tokens,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "bundle.json").write_text(prepared.bundle.to_json(), encoding="utf-8")
    (output / "document.md").write_text(prepared.structured_markdown, encoding="utf-8")
    (output / "financial-snapshot.md").write_text(prepared.financial_snapshot.to_markdown(), encoding="utf-8")
    (output / "chunks.jsonl").write_text(
        "\n".join(chunk.model_dump_json(exclude_none=True) for chunk in prepared.chunks) + "\n",
        encoding="utf-8",
    )
    (output / "evidence-requests.jsonl").write_text(
        "\n".join(request.model_dump_json(exclude_none=True) for request in prepared.evidence_requests) + "\n",
        encoding="utf-8",
    )
    if prepared.selected_context:
        (output / "context-allocation.json").write_text(
            prepared.selected_context.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )
    return 0


def _schema(_args: argparse.Namespace) -> int:
    print(json.dumps(CanonicalDocumentBundle.model_json_schema(), ensure_ascii=False, indent=2))
    return 0


def _calendar_date(value: str) -> date:
    normalized = value.strip()
    try:
        if len(normalized) == 8 and normalized.isdigit():
            return datetime.strptime(normalized, "%Y%m%d").date()
        return date.fromisoformat(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; use YYYY-MM-DD or YYYYMMDD") from exc


def _identifier_values(direct: list[str] | None, files: list[str] | None) -> list[str]:
    values = list(direct or [])
    for filename in files or []:
        path = Path(filename).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"identifier file not found: {path}")
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            content = line.split("#", 1)[0]
            values.extend(item.strip() for item in content.split(",") if item.strip())
    return list(dict.fromkeys(values))


def _collection_result_path(output: str | Path, source: str, started_at: datetime) -> Path:
    stamp = started_at.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path(output).resolve() / "collections" / f"{stamp}-{source}.json"


def _finish_collection(args: argparse.Namespace, result: object, source: str) -> int:
    from moatrader.ingestion import CollectionResult

    typed = CollectionResult.model_validate(result)
    manifest_path: Path | None = None
    if typed.filings:
        manifest_path = Path(args.manifest).resolve() if args.manifest else Path(args.output).resolve() / "collected-universe.csv"
        write_collected_universe_manifest(BronzeFilingStore(args.output), manifest_path)
        typed = typed.model_copy(update={"manifest_path": str(manifest_path)})
    result_path = _collection_result_path(args.output, source, typed.started_at)
    RunStore(args.output).write_json(result_path, typed)
    print(f"source={typed.source_type.value}")
    print(f"discovered={typed.discovered_count}")
    print(f"downloaded={typed.downloaded_count}")
    print(f"unchanged={typed.unchanged_count}")
    print(f"failed={len(typed.failures)}")
    print(f"result={result_path}")
    if manifest_path:
        print(f"manifest={manifest_path}")
    for failure in typed.failures:
        print(f"{failure.source_document_id}\tFAILED\t{failure.message}", file=sys.stderr)
    return 2 if typed.failures else 0


def _collect_dart(args: argparse.Namespace) -> int:
    api_key = os.getenv(args.api_key_env, "").strip()
    if not api_key:
        raise ValueError(
            f"OpenDART API key is missing; set environment variable {args.api_key_env}"
        )
    http = ResilientHttpClient(
        user_agent="MoatRader OpenDART collector",
        requests_per_second=args.requests_per_second,
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        default_max_bytes=int(args.max_download_mb * 1024 * 1024),
    )
    collector = DartCollector(
        DartOpenApiClient(http, api_key),
        BronzeFilingStore(args.output),
        max_archive_bytes=int(args.max_download_mb * 1024 * 1024),
        max_extracted_bytes=int(args.max_extracted_mb * 1024 * 1024),
    )
    report_kinds = set(args.report_kind or ["annual", "semiannual", "quarterly"])
    result = collector.collect(
        begin_date=args.begin_date,
        end_date=args.end_date,
        corp_codes=_identifier_values(args.corp_code, args.corp_code_file),
        stock_codes=_identifier_values(args.stock_code, args.stock_code_file),
        all_companies=args.all_companies,
        report_kinds=report_kinds,
        final_only=args.final_only,
        refresh=args.refresh,
        max_filings=args.max_filings,
    )
    return _finish_collection(args, result, "dart")


def _collect_sec(args: argparse.Namespace) -> int:
    user_agent = os.getenv(args.user_agent_env, "").strip()
    if not user_agent:
        raise ValueError(
            f"SEC declared User-Agent is missing; set {args.user_agent_env} to "
            "'Application Name contact@example.com'"
        )
    if args.requests_per_second > 10:
        raise ValueError("SEC automated access must not exceed 10 requests per second")
    http = ResilientHttpClient(
        user_agent=user_agent,
        requests_per_second=args.requests_per_second,
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        default_max_bytes=int(args.max_download_mb * 1024 * 1024),
    )
    client = SecEdgarClient(http, declared_user_agent=user_agent)
    collector = SecEdgarCollector(
        client,
        BronzeFilingStore(args.output),
        max_download_bytes=int(args.max_download_mb * 1024 * 1024),
        availability_lag_minutes=args.availability_lag_minutes,
    )
    result = collector.collect(
        begin_date=args.begin_date,
        end_date=args.end_date,
        tickers=_identifier_values(args.ticker, args.ticker_file),
        ciks=_identifier_values(args.cik, args.cik_file),
        forms=set(args.form or DEFAULT_SEC_FORMS),
        refresh=args.refresh,
        max_filings=args.max_filings,
    )
    return _finish_collection(args, result, "sec-edgar")


def _kind_company_identities(path: str | Path) -> list[KindCompanyIdentity]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"KIND company identity CSV not found: {source}")
    identities: dict[str, KindCompanyIdentity] = {}
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=2):
            raw_ticker = str(row.get("ticker") or row.get("stock_code") or "").strip()
            ticker = raw_ticker.zfill(6)
            issuer_id = str(row.get("issuer_id") or row.get("corp_code") or "").strip()
            issuer_name = str(row.get("issuer_name") or row.get("name") or row.get("corp_name") or "").strip()
            if not raw_ticker or not issuer_id or not issuer_name:
                raise ValueError(
                    f"row {row_number}: company file requires ticker/stock_code, "
                    "issuer_id/corp_code, and issuer_name/name"
                )
            identity = KindCompanyIdentity(
                ticker=ticker,
                issuer_id=issuer_id,
                issuer_name=issuer_name,
                kind_company_code=(str(row.get("kind_company_code") or "").strip() or None),
            )
            prior = identities.get(ticker)
            if prior is not None and prior != identity:
                raise ValueError(f"row {row_number}: conflicting identity for ticker {ticker}")
            identities[ticker] = identity
    if not identities:
        raise ValueError("KIND company identity CSV contains no companies")
    return list(identities.values())


def _collect_kind_ir(args: argparse.Namespace) -> int:
    http = ResilientHttpClient(
        user_agent="MoatRader KIND IR collector",
        requests_per_second=args.requests_per_second,
        timeout_seconds=args.timeout,
        max_retries=args.retries,
        default_max_bytes=int(args.max_download_mb * 1024 * 1024),
    )
    collector = KindIrCollector(
        KindIrClient(http),
        BronzeFilingStore(args.output),
        max_download_bytes=int(args.max_download_mb * 1024 * 1024),
    )
    result = collector.collect(
        begin_date=args.begin_date,
        end_date=args.end_date,
        companies=_kind_company_identities(args.company_file),
        refresh=args.refresh,
        max_materials=args.max_materials,
        max_materials_per_company=args.max_materials_per_company,
    )
    return _finish_collection(args, result, "kind-ir")


def _collect_manifest(args: argparse.Namespace) -> int:
    output = write_collected_universe_manifest(BronzeFilingStore(args.bronze_root), args.output)
    print(f"manifest={output}")
    return 0


def _aware_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid ISO-8601 timestamp: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include Z or a UTC offset (for example +09:00)")
    return parsed


def _selected_tickers(args: argparse.Namespace) -> set[str] | None:
    values: list[str] = list(args.ticker or [])
    for group in args.tickers or []:
        values.extend(group.split(","))
    selected = {value.strip() for value in values if value.strip()}
    return selected or None


def _moat_run(args: argparse.Namespace) -> int:
    if args.resume and not args.run_id:
        raise ValueError("--resume requires --run-id so the existing checkpoints can be located")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = (Path(args.output) / run_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise ValueError(f"run directory already exists; choose another --run-id or use --resume: {run_dir}")
    manifest = load_universe_manifest(args.universe)
    companies = manifest.select(_selected_tickers(args))
    workspace_manifest = find_workspace_manifest(args.universe, args.output)
    workspace_payload = (
        json.loads(workspace_manifest.read_text(encoding="utf-8-sig"))
        if workspace_manifest is not None
        else {}
    )
    experiment_id = args.experiment_id or workspace_payload.get("experiment_id")
    if not experiment_id:
        output_identity = hashlib.sha256(str(Path(args.output).resolve()).encode("utf-8")).hexdigest()[:16]
        experiment_id = f"standalone-{output_identity}"
    replay_cache = Path(args.llm_replay_cache).resolve() if args.llm_replay_cache else (
        workspace_manifest.parent / "llm-replay"
        if workspace_manifest is not None
        else Path(args.output).resolve() / ".llm-replay"
    )
    evidence_ledger = (
        workspace_manifest.parent / "evidence-ledger"
        if workspace_manifest is not None
        else Path(args.output).resolve() / ".evidence-ledger"
    )
    config = UniverseRunConfig(
        run_id=run_id,
        as_of=args.as_of,
        summary_model=args.model or args.summary_model,
        moat_model=args.model or args.moat_model,
        summary_reasoning_effort=args.reasoning_effort or args.summary_reasoning_effort,
        atomic_reasoning_effort=args.reasoning_effort or args.atomic_reasoning_effort,
        moat_reasoning_effort=args.reasoning_effort or args.moat_reasoning_effort,
        context_tokens=args.context_tokens,
        prompt_reserve_tokens=args.prompt_reserve_tokens,
        strength_context_tokens=args.strength_context_tokens,
        strength_prompt_reserve_tokens=args.strength_prompt_reserve_tokens,
        max_output_tokens=args.max_output_tokens,
        minimum_text_retention=args.minimum_text_retention,
        minimum_numeric_retention=args.minimum_numeric_retention,
        minimum_structured_fact_retention=args.minimum_structured_fact_retention,
        require_table_count_match=not args.allow_table_count_mismatch,
        require_financial_table_semantics=not args.allow_incomplete_financial_table_semantics,
        allow_low_quality=args.allow_low_quality,
        maximum_price_age_days=args.maximum_price_age_days,
        maximum_atomic_evidence_units=args.maximum_atomic_evidence_units,
        maximum_ir_atomic_evidence_units=args.maximum_ir_atomic_evidence_units,
        atomic_classification_votes=args.atomic_classification_votes,
        maximum_valuation_atomic_evidence_units=args.maximum_valuation_atomic_evidence_units,
        maximum_ir_valuation_atomic_evidence_units=(
            args.maximum_ir_valuation_atomic_evidence_units
        ),
        valuation_classification_votes=args.valuation_classification_votes,
        incremental_ir_mode=args.incremental_ir,
        longitudinal_ir_mode=args.longitudinal_ir,
        minimum_longitudinal_ir_years=args.minimum_longitudinal_ir_years,
        consolidate_section_summaries=args.consolidate_section_summaries,
        ir_ocr_engine=args.ir_ocr_engine,
        ir_ocr_device=args.ir_ocr_device,
        ir_ocr_cpu_threads=args.ir_ocr_cpu_threads,
        workers=args.workers,
        resume=args.resume,
        dry_run=args.dry_run,
        validation_attempts=args.validation_attempts,
        experiment_id=experiment_id,
        llm_replay_cache_directory=str(replay_cache),
        evidence_ledger_directory=str(evidence_ledger),
        enable_legacy_moat_ranking=args.enable_legacy_moat_ranking,
    )
    is_large_manifest = len(manifest.companies) > 5
    if is_large_manifest and args.preflight_sample:
        validate_preflight_sample_selection(
            (company.ticker for company in companies),
            workspace_manifest=workspace_manifest,
        )
    elif is_large_manifest:
        if not args.preflight_report:
            raise ValueError(
                "a manifest with more than 5 companies requires a passed --preflight-report; "
                "run exactly 3..5 tickers twice with --preflight-sample first"
            )
        approval_path = Path(args.preflight_report).resolve()
        expected_approval_hash = workspace_payload.get("preflight_report_sha256")
        if workspace_manifest is not None and (
            workspace_payload.get("preflight_status") != "PASSED"
            or not approval_path.is_file()
            or not expected_approval_hash
            or hashlib.sha256(approval_path.read_bytes()).hexdigest() != expected_approval_hash
        ):
            raise ValueError(
                "preflight report is not the passed report recorded by workspace-manifest.json"
            )
        validate_preflight_approval(
            approval_path,
            universe_tickers=_preflight_universe_tickers(
                workspace_manifest,
                [company.ticker for company in manifest.companies],
            ),
            as_of_date=config.as_of.date().isoformat(),
            config=config,
            runner_version=RUNNER_VERSION,
        )
    elif args.preflight_sample:
        validate_preflight_sample_selection(
            (company.ticker for company in companies),
            workspace_manifest=workspace_manifest,
        )
    transport = None
    if not args.dry_run:
        transport = OpenAIResponsesTransport(
            summary_model=config.summary_model,
            moat_model=config.moat_model,
            summary_reasoning_effort=config.summary_reasoning_effort,
            atomic_reasoning_effort=config.atomic_reasoning_effort,
            moat_reasoning_effort=config.moat_reasoning_effort,
            max_output_tokens=args.max_output_tokens,
            max_retries=args.api_retries,
            timeout_seconds=args.api_timeout,
        )
    runner = MoatUniverseRunner(
        config=config,
        output_directory=args.output,
        transport=transport,
        pipeline=CanonicalFinancialDocumentPipeline(
            ir_ocr_adapter=_ir_ocr_adapter(args)
        ),
    )
    result = runner.run(manifest, companies)
    print(f"run_id={result.run_id}")
    print(f"run_dir={run_dir}")
    for company in result.companies:
        suffix = f" error={company.error}" if company.error else ""
        print(f"{company.ticker}\t{company.status.value}{suffix}")
    print(f"complete={sum(item.status.value == 'COMPLETE' for item in result.companies)}")
    print(f"failed={result.failed_count}")
    print(f"expectation_opportunities={len(result.opportunities)}")
    print(f"legacy_moat_ranked={len(result.ranking)}")
    return 2 if result.failed_count else 0


def _load_run_result(run_dir: str) -> tuple[Path, UniverseRunResult]:
    directory = Path(run_dir).resolve()
    path = directory / "run-result.json"
    if not path.is_file():
        raise FileNotFoundError(f"run result not found: {path}")
    return directory, UniverseRunResult.model_validate_json(path.read_text(encoding="utf-8-sig"))


def _moat_status(args: argparse.Namespace) -> int:
    directory = Path(args.run_dir).resolve()
    result_path = directory / "run-result.json"
    if not result_path.is_file():
        config_path = directory / "run-config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"run configuration not found: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        print(f"run_id={config.get('run_id', directory.name)}")
        print(f"run_dir={directory}")
        print(f"as_of={config.get('as_of', '-')}")
        for ticker in config.get("tickers", []):
            checkpoint = directory / "companies" / ticker / "checkpoint.json"
            state = json.loads(checkpoint.read_text(encoding="utf-8-sig")) if checkpoint.is_file() else {}
            print(f"{ticker}\t{state.get('stage', 'PENDING')}")
        return 0
    result = UniverseRunResult.model_validate_json(result_path.read_text(encoding="utf-8-sig"))
    print(f"run_id={result.run_id}")
    print(f"run_dir={directory}")
    print(f"as_of={result.as_of.isoformat()}")
    for company in result.companies:
        expectation = company.expectation_analysis
        gap = expectation.expectation_gap.direction.value if expectation else "-"
        central = (
            expectation.intrinsic_valuation.central_value_per_share
            if expectation
            else "-"
        )
        probable = expectation.three_p["central"].verdict.value if expectation else "-"
        print(
            f"{company.ticker}\t{company.status.value}\tgap={gap}"
            f"\tintrinsic_central={central}\t3p={probable}"
        )
    print(f"failed={result.failed_count}")
    print(f"expectation_opportunities={len(result.opportunities)}")
    print(f"legacy_moat_ranked={len(result.ranking)}")
    return 0


def _screen_rank(args: argparse.Namespace) -> int:
    directory, result = _load_run_result(args.run_dir)
    selector = SelectorConfig(
        minimum_moat_score=args.minimum_moat_score,
        minimum_margin_of_safety=args.minimum_margin_of_safety,
        minimum_model_confidence=args.minimum_model_confidence,
        minimum_document_coverage=args.minimum_document_coverage,
    )
    ranked = rank_run_result(result, selector)
    reranked = result.model_copy(update={"ranking": ranked})
    output = Path(args.output).resolve() if args.output else directory / "ranking.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(ranking_csv(reranked), encoding="utf-8")
    for rank, candidate in enumerate(ranked, start=1):
        print(
            f"{rank}\t{candidate.ticker}\tmoat={candidate.moat_score}"
            f"\tprice/dcf={candidate.price_to_dcf:.4f}"
            f"\tmargin={candidate.margin_of_safety:.4f}"
            f"\tquality_value={candidate.quality_value_score:.6f}"
        )
    print(f"ranked={len(ranked)}")
    print(f"output={output}")
    return 0


def _screen_expectations(args: argparse.Namespace) -> int:
    directory, result = _load_run_result(args.run_dir)
    ranked = rank_expectation_opportunities(result)
    reranked = result.model_copy(update={"opportunities": ranked})
    output = Path(args.output).resolve() if args.output else directory / "opportunities.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(opportunities_csv(reranked), encoding="utf-8")
    for rank, candidate in enumerate(ranked, start=1):
        print(
            f"{rank}\t{candidate.ticker}\tdirection={candidate.direction.value}"
            f"\tcentral_gap={candidate.central_value_gap:.4f}"
            f"\tdownside_gap={candidate.downside_value_gap:.4f}"
            f"\trange_width={candidate.valuation_range_width_pct:.4f}"
        )
    print(f"ranked={len(ranked)}")
    print(f"output={output}")
    return 0


def _backtest_run(args: argparse.Namespace) -> int:
    result_paths: list[Path] = []
    for value in args.run_dir or []:
        directory = Path(value).resolve()
        path = directory if directory.name == "run-result.json" else directory / "run-result.json"
        if not path.is_file():
            raise FileNotFoundError(f"run result not found: {path}")
        result_paths.append(path)
    if args.runs_root:
        root = Path(args.runs_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"runs root not found: {root}")
        result_paths.extend(sorted(root.glob("*/run-result.json")))
    unique_paths = list(dict.fromkeys(path.resolve() for path in result_paths))
    if not unique_paths:
        raise ValueError("provide at least one --run-dir or --runs-root containing run-result.json files")
    runs = [
        UniverseRunResult.model_validate_json(path.read_text(encoding="utf-8-sig"))
        for path in unique_paths
    ]
    config = BacktestConfig(
        end_at=args.end_at,
        top_n=args.top_n,
        execution_lag_days=args.execution_lag_days,
        transaction_cost_bps=args.transaction_cost_bps,
        slippage_bps=args.slippage_bps,
        maximum_turnover=args.maximum_turnover,
        enforce_capacity=args.enforce_capacity,
        maximum_participation_rate=args.maximum_participation_rate,
        benchmark_ticker=args.benchmark_ticker,
        initial_capital=args.initial_capital,
        liquidate_at_end=not args.no_terminal_liquidation,
        maximum_signal_price_age_days=args.maximum_signal_price_age_days,
        missing_exit_return=args.missing_exit_return,
        adjusted_close_includes_distributions=not args.unadjusted_distributions,
    )
    result = PointInTimeBacktester(config).run(runs, PricePanel(load_price_panel(args.prices)))
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError(f"backtest output directory is not empty; use --overwrite: {output}")
    store = RunStore(output)
    store.write_json(output / "backtest-result.json", result)
    store.write_text(output / "equity.csv", equity_csv(result))
    store.write_text(output / "rebalances.csv", rebalances_csv(result))
    performance = result.performance
    print(f"output={output}")
    print(f"runs={len(result.source_run_ids)}")
    print(f"total_return={performance.total_return}")
    print(f"cagr={performance.cagr if performance.cagr is not None else '-'}")
    print(f"max_drawdown={performance.max_drawdown}")
    print(f"transaction_cost={performance.total_transaction_cost}")
    print(f"slippage_cost={performance.total_slippage_cost}")
    if performance.benchmark_total_return is not None:
        print(f"benchmark_total_return={performance.benchmark_total_return}")
        print(f"excess_total_return={performance.excess_total_return}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="moatrader")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="collect immutable Bronze filings from official APIs")
    collect_subparsers = collect.add_subparsers(dest="collect_command", required=True)

    collect_dart = collect_subparsers.add_parser(
        "dart",
        help="discover and download DART filings through OpenDART",
    )
    collect_dart.add_argument("--from", dest="begin_date", required=True, type=_calendar_date)
    collect_dart.add_argument("--to", dest="end_date", required=True, type=_calendar_date)
    collect_dart.add_argument("--corp-code", action="append", help="8-digit DART corporation code; repeatable")
    collect_dart.add_argument("--stock-code", action="append", help="6-digit listed stock code; repeatable")
    collect_dart.add_argument("--corp-code-file", action="append", help="UTF-8 comma/newline corporation-code file")
    collect_dart.add_argument("--stock-code-file", action="append", help="UTF-8 comma/newline stock-code file")
    collect_dart.add_argument(
        "--all-companies",
        action="store_true",
        help="collect all companies (OpenDART restricts no-corp searches to about three months)",
    )
    collect_dart.add_argument(
        "--report-kind",
        action="append",
        choices=["annual", "semiannual", "quarterly"],
        help="periodic report kind; repeatable (default: all three)",
    )
    collect_dart.add_argument("--final-only", action="store_true", help="ask OpenDART for final reports only")
    collect_dart.add_argument("--output", default="data-lake/bronze", help="Bronze root directory")
    collect_dart.add_argument("--manifest", help="combined latest-filings universe CSV output")
    collect_dart.add_argument("--api-key-env", default="DART_API_KEY")
    collect_dart.add_argument("--requests-per-second", type=float, default=2.0)
    collect_dart.add_argument("--timeout", type=float, default=60.0)
    collect_dart.add_argument("--retries", type=int, default=4)
    collect_dart.add_argument("--max-download-mb", type=float, default=256.0)
    collect_dart.add_argument("--max-extracted-mb", type=float, default=1024.0)
    collect_dart.add_argument("--max-filings", type=int)
    collect_dart.add_argument("--refresh", action="store_true", help="redownload known IDs to detect revisions")
    collect_dart.set_defaults(handler=_collect_dart)

    collect_sec = collect_subparsers.add_parser(
        "sec",
        help="discover via data.sec.gov and download original EDGAR filings",
    )
    collect_sec.add_argument("--from", dest="begin_date", required=True, type=_calendar_date)
    collect_sec.add_argument("--to", dest="end_date", required=True, type=_calendar_date)
    collect_sec.add_argument("--ticker", action="append", help="SEC company ticker; repeatable")
    collect_sec.add_argument("--cik", action="append", help="SEC CIK; repeatable")
    collect_sec.add_argument("--ticker-file", action="append", help="UTF-8 comma/newline ticker file")
    collect_sec.add_argument("--cik-file", action="append", help="UTF-8 comma/newline CIK file")
    collect_sec.add_argument(
        "--form",
        action="append",
        help="form type; repeatable (default: 10-K/10-Q/20-F/40-F and amendments)",
    )
    collect_sec.add_argument("--output", default="data-lake/bronze", help="Bronze root directory")
    collect_sec.add_argument("--manifest", help="combined latest-filings universe CSV output")
    collect_sec.add_argument("--user-agent-env", default="SEC_USER_AGENT")
    collect_sec.add_argument("--requests-per-second", type=float, default=5.0)
    collect_sec.add_argument("--timeout", type=float, default=60.0)
    collect_sec.add_argument("--retries", type=int, default=4)
    collect_sec.add_argument("--max-download-mb", type=float, default=256.0)
    collect_sec.add_argument(
        "--availability-lag-minutes",
        type=int,
        default=5,
        help="conservative lag added to SEC acceptanceDateTime (default: 5)",
    )
    collect_sec.add_argument("--max-filings", type=int)
    collect_sec.add_argument("--refresh", action="store_true", help="redownload known IDs to detect revisions")
    collect_sec.set_defaults(handler=_collect_sec)

    collect_kind = collect_subparsers.add_parser(
        "kind-ir",
        help="discover and download PIT-safe IR PDFs from the official KIND IR materials list",
    )
    collect_kind.add_argument("--from", dest="begin_date", required=True, type=_calendar_date)
    collect_kind.add_argument("--to", dest="end_date", required=True, type=_calendar_date)
    collect_kind.add_argument(
        "--company-file",
        required=True,
        help="UTF-8 CSV with ticker, issuer_id, issuer_name (DART manifest columns are accepted)",
    )
    collect_kind.add_argument("--output", default="data-lake/bronze", help="Bronze root directory")
    collect_kind.add_argument("--manifest", help="combined latest-filings universe CSV output")
    collect_kind.add_argument("--requests-per-second", type=float, default=2.0)
    collect_kind.add_argument("--timeout", type=float, default=60.0)
    collect_kind.add_argument("--retries", type=int, default=4)
    collect_kind.add_argument("--max-download-mb", type=float, default=256.0)
    collect_kind.add_argument("--max-materials", type=int)
    collect_kind.add_argument(
        "--max-materials-per-company",
        type=int,
        help="keep only the latest N available PDF attachments per matched company",
    )
    collect_kind.add_argument("--refresh", action="store_true", help="redownload known IDs to detect revisions")
    collect_kind.set_defaults(handler=_collect_kind_ir)

    collect_manifest = collect_subparsers.add_parser(
        "manifest",
        help="rebuild a runner-compatible universe CSV from Bronze latest pointers",
    )
    collect_manifest.add_argument("--bronze-root", default="data-lake/bronze")
    collect_manifest.add_argument("--output", required=True)
    collect_manifest.set_defaults(handler=_collect_manifest)

    ingest = subparsers.add_parser("ingest-html", help="convert DART/EDGAR/IR HTML or IR PDF to canonical artifacts")
    ingest.add_argument("input")
    ingest.add_argument("--metadata", required=True, help="UTF-8 JSON metadata/hints")
    ingest.add_argument("--source", required=True, choices=["dart", "sec_edgar", "ir"])
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--context-tokens", type=int)
    ingest.add_argument(
        "--ir-ocr-engine",
        choices=["none", "paddle"],
        default=os.getenv("MOATRADER_IR_OCR_ENGINE", "none"),
    )
    ingest.add_argument(
        "--ir-ocr-device",
        default=os.getenv("MOATRADER_IR_OCR_DEVICE", "cpu"),
    )
    ingest.add_argument("--ir-ocr-cpu-threads", type=int, default=6)
    ingest.set_defaults(handler=_ingest)
    schema = subparsers.add_parser("schema", help="print CanonicalDocumentBundle JSON Schema")
    schema.set_defaults(handler=_schema)

    moat = subparsers.add_parser(
        "analyze",
        aliases=["moat"],
        help="run or inspect evidence, intrinsic valuation, and expectation-gap analysis",
    )
    moat_subparsers = moat.add_subparsers(dest="moat_command", required=True)
    moat_run = moat_subparsers.add_parser("run", help="analyze one, several, or all manifest tickers")
    moat_run.add_argument("--universe", required=True, help="universe CSV manifest")
    moat_run.add_argument("--ticker", action="append", help="one ticker; may be repeated")
    moat_run.add_argument("--tickers", action="append", help="comma-separated ticker list")
    moat_run.add_argument("--as-of", required=True, type=_aware_datetime, help="PIT cutoff with timezone")
    moat_run.add_argument("--output", default="data-lake/gold/runs", help="parent run directory")
    moat_run.add_argument("--run-id", help="stable run ID; required with --resume")
    moat_run.add_argument("--resume", action="store_true", help="reuse valid per-company checkpoints")
    moat_run.add_argument("--dry-run", action="store_true", help="ingest and emit LLM requests without API calls")
    moat_run.add_argument(
        "--preflight-sample",
        action="store_true",
        help="allow only the workspace's 3..5 ticker preflight sample",
    )
    moat_run.add_argument(
        "--preflight-report",
        help="passed preflight JSON required whenever the input manifest contains more than 5 companies",
    )
    moat_run.add_argument(
        "--experiment-id",
        help="fresh experiment namespace; auto-discovered from workspace-manifest.json",
    )
    moat_run.add_argument(
        "--llm-replay-cache",
        help="experiment-scoped content-addressed response cache directory",
    )
    moat_run.add_argument(
        "--summary-model",
        default=os.getenv("MOATRADER_SUMMARY_MODEL", "gpt-5-nano"),
        help="model reserved for optional generative sentence summaries (default path uses deterministic summaries)",
    )
    moat_run.add_argument(
        "--moat-model",
        default=os.getenv("MOATRADER_MOAT_MODEL", "gpt-5.6-luna"),
        help="pinned model for atomic MOAT evidence classification (default: gpt-5.6-luna)",
    )
    moat_run.add_argument("--model", help=argparse.SUPPRESS)
    moat_run.add_argument(
        "--summary-reasoning-effort",
        choices=["minimal", "low", "medium", "high"],
        default="low",
    )
    moat_run.add_argument(
        "--atomic-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="medium",
        help="reasoning effort for audit-lane atomic classification (default: medium)",
    )
    moat_run.add_argument(
        "--moat-reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="medium",
    )
    moat_run.add_argument("--reasoning-effort", help=argparse.SUPPRESS)
    moat_run.add_argument("--context-tokens", type=int, default=64_000)
    moat_run.add_argument("--prompt-reserve-tokens", type=int, default=8_000)
    moat_run.add_argument("--strength-context-tokens", type=int, default=100_000)
    moat_run.add_argument("--strength-prompt-reserve-tokens", type=int, default=12_000)
    moat_run.add_argument("--max-output-tokens", type=int, default=8_000)
    moat_run.add_argument("--minimum-text-retention", type=float, default=0.95)
    moat_run.add_argument("--minimum-numeric-retention", type=float, default=0.99)
    moat_run.add_argument("--minimum-structured-fact-retention", type=float, default=0.99)
    moat_run.add_argument("--allow-table-count-mismatch", action="store_true")
    moat_run.add_argument(
        "--allow-low-quality",
        action="store_true",
        help="continue despite parser quality gate failures (recorded in quality-gate.json)",
    )
    moat_run.add_argument(
        "--allow-incomplete-financial-table-semantics",
        action="store_true",
        help="do not hard-fail numeric financial tables missing headers, periods, or units",
    )
    moat_run.add_argument("--maximum-price-age-days", type=int, default=7)
    moat_run.add_argument(
        "--maximum-atomic-evidence-units",
        "--maximum-evidence-chunks",
        dest="maximum_atomic_evidence_units",
        type=int,
        default=24,
        help="classify at most this many content-ranked atomic evidence units; the old chunk flag is an alias",
    )
    moat_run.add_argument(
        "--maximum-ir-atomic-evidence-units",
        type=int,
        default=12,
        help="classify IR atomic evidence in a separate slot so it cannot displace DART evidence",
    )
    moat_run.add_argument(
        "--atomic-classification-votes",
        type=int,
        choices=[1, 3, 5, 7, 9],
        default=3,
        help="independent votes per frozen atomic unit; strict majority or fail closed (default: 3)",
    )
    moat_run.add_argument(
        "--incremental-ir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="freeze the DART assessment and evaluate IR only as a deterministic incremental delta",
    )
    moat_run.add_argument(
        "--longitudinal-ir",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="preserve dated IR document coverage and require multi-year accepted evidence",
    )
    moat_run.add_argument(
        "--minimum-longitudinal-ir-years",
        type=int,
        default=3,
        help="minimum distinct usable IR years required for longitudinal treatment",
    )
    moat_run.add_argument(
        "--consolidate-section-summaries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="group deterministic evidence-preserving summaries at company level",
    )
    moat_run.add_argument(
        "--ir-ocr-engine",
        choices=["none", "paddle"],
        default=os.getenv("MOATRADER_IR_OCR_ENGINE", "none"),
        help="selective OCR engine for image-dominant or encoding-corrupt IR PDF pages",
    )
    moat_run.add_argument(
        "--ir-ocr-device",
        default=os.getenv("MOATRADER_IR_OCR_DEVICE", "cpu"),
        help="PaddleOCR device, for example cpu or gpu:0",
    )
    moat_run.add_argument("--ir-ocr-cpu-threads", type=int, default=6)
    moat_run.add_argument("--workers", type=int, default=1)
    moat_run.add_argument("--validation-attempts", type=int, default=2)
    moat_run.add_argument(
        "--enable-legacy-moat-ranking",
        action="store_true",
        help="emit the deprecated MOAT×DCF ranking for diagnostics only",
    )
    moat_run.add_argument(
        "--maximum-valuation-atomic-evidence-units",
        type=int,
        default=24,
        help="valuation-only atomic units selected independently from the frozen MOAT sensor",
    )
    moat_run.add_argument(
        "--maximum-ir-valuation-atomic-evidence-units",
        type=int,
        default=12,
        help="IR valuation-only units; enabled only when expectation_assumptions is supplied",
    )
    moat_run.add_argument(
        "--valuation-classification-votes",
        type=int,
        choices=[1, 3, 5, 7, 9],
        default=3,
        help="strict-majority votes for the valuation-driver lane",
    )
    moat_run.add_argument("--api-retries", type=int, default=4)
    moat_run.add_argument("--api-timeout", type=float, default=180.0)
    moat_run.set_defaults(handler=_moat_run)

    moat_status = moat_subparsers.add_parser("status", help="show a completed or prepared run")
    moat_status.add_argument("--run-dir", required=True)
    moat_status.set_defaults(handler=_moat_status)

    screen = subparsers.add_parser("screen", help="rank completed expectation-gap results")
    screen_subparsers = screen.add_subparsers(dest="screen_command", required=True)
    expectations = screen_subparsers.add_parser(
        "expectations",
        help="rank the primary Reverse-DCF/3P expectation-gap output",
    )
    expectations.add_argument("--run-dir", required=True)
    expectations.add_argument("--output")
    expectations.set_defaults(handler=_screen_expectations)

    rank = screen_subparsers.add_parser(
        "rank",
        help="deprecated diagnostic: re-rank legacy MOAT×DCF output",
    )
    rank.add_argument("--run-dir", required=True)
    rank.add_argument("--minimum-moat-score", type=Decimal, default=Decimal("5"))
    rank.add_argument("--minimum-margin-of-safety", type=Decimal, default=Decimal("0.20"))
    rank.add_argument("--minimum-model-confidence", type=Decimal, default=Decimal("0.50"))
    rank.add_argument("--minimum-document-coverage", type=Decimal, default=Decimal("0.50"))
    rank.add_argument("--output")
    rank.set_defaults(handler=_screen_rank)

    backtest = subparsers.add_parser("backtest", help="PIT backtest historical run rankings")
    backtest_subparsers = backtest.add_subparsers(dest="backtest_command", required=True)
    backtest_run = backtest_subparsers.add_parser("run", help="backtest one or more historical run results")
    backtest_run.add_argument("--run-dir", action="append", help="run directory; may be repeated")
    backtest_run.add_argument("--runs-root", help="directory whose immediate children contain run-result.json")
    backtest_run.add_argument("--prices", required=True, help="adjusted price CSV")
    backtest_run.add_argument("--end-at", required=True, type=_aware_datetime)
    backtest_run.add_argument("--output", required=True)
    backtest_run.add_argument("--top-n", type=int, default=10)
    backtest_run.add_argument("--execution-lag-days", type=int, default=1)
    backtest_run.add_argument("--transaction-cost-bps", type=Decimal, default=Decimal("10"))
    backtest_run.add_argument("--slippage-bps", type=Decimal, default=Decimal("5"))
    backtest_run.add_argument("--maximum-turnover", type=Decimal, default=Decimal("1"))
    backtest_run.add_argument("--enforce-capacity", action="store_true")
    backtest_run.add_argument(
        "--maximum-participation-rate",
        type=Decimal,
        default=Decimal("0.05"),
        help="maximum trade notional / daily dollar volume when capacity is enforced",
    )
    backtest_run.add_argument("--benchmark-ticker")
    backtest_run.add_argument("--initial-capital", type=Decimal, default=Decimal("100000000"))
    backtest_run.add_argument("--no-terminal-liquidation", action="store_true")
    backtest_run.add_argument("--maximum-signal-price-age-days", type=int, default=7)
    backtest_run.add_argument(
        "--missing-exit-return",
        type=Decimal,
        default=Decimal("-1"),
        help="conservative return assigned when a held security has no exit price",
    )
    backtest_run.add_argument(
        "--unadjusted-distributions",
        action="store_true",
        help="declare that adjusted_close does not include dividends/distributions",
    )
    backtest_run.add_argument("--overwrite", action="store_true")
    backtest_run.set_defaults(handler=_backtest_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
