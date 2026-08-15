from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from moatrader.preflight import (
    PREFLIGHT_SCHEMA_VERSION,
    contract_sha256,
    execution_contract,
    ticker_set_sha256,
)
from moatrader.runner.engine import RUNNER_VERSION
from moatrader.runner.models import CompanyRunStatus, UniverseRunConfig, UniverseRunResult
try:
    from scripts.audit_moat_reproducibility import compare_runs
    from scripts.merge_kr_signal_panel import spearman
except ModuleNotFoundError:  # Direct `python scripts\...py` execution.
    from audit_moat_reproducibility import compare_runs
    from merge_kr_signal_panel import spearman


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _parse_run_arguments(values: list[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        date, separator, path = value.partition("=")
        if not separator or not date or date in result:
            raise ValueError(f"invalid or duplicate --{label} value: {value}")
        result[date] = Path(path).resolve()
    return result


def _load_run(path: Path, date: str, sample: set[str]) -> UniverseRunResult:
    result = UniverseRunResult.model_validate_json(path.read_text(encoding="utf-8-sig"))
    if result.as_of.date().isoformat() != date:
        raise ValueError(f"preflight run date mismatch: {path} has {result.as_of.isoformat()}, expected {date}")
    actual = {company.ticker for company in result.companies}
    if actual != sample:
        raise ValueError(f"preflight sample mismatch for {date}: expected={sorted(sample)}, actual={sorted(actual)}")
    incomplete = [
        company.ticker
        for company in result.companies
        if company.status != CompanyRunStatus.COMPLETE or company.moat_score is None
    ]
    if incomplete:
        raise ValueError(f"preflight run has incomplete scores for {date}: {incomplete}")
    versions = {company.runner_version for company in result.companies}
    if versions != {RUNNER_VERSION}:
        raise ValueError(f"preflight runner versions are not current: {versions}")
    return result


def _run_contract(path: Path) -> dict[str, Any]:
    payload = json.loads((path.parent / "run-config.json").read_text(encoding="utf-8-sig"))
    fields = UniverseRunConfig.model_fields
    config = UniverseRunConfig.model_validate({key: value for key, value in payload.items() if key in fields})
    return execution_contract(config)


def _candidate_replay_audit(result: UniverseRunResult) -> tuple[int, int, list[str]]:
    total = 0
    replayed = 0
    failures: list[str] = []
    for company in result.companies:
        path = Path(company.artifact_directory) / "llm-calls.jsonl"
        calls = [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]
        total += len(calls)
        replayed += sum(call.get("replayed") is True for call in calls)
        if any(not call.get("replay_cache_key") for call in calls):
            failures.append(f"{company.ticker}: call missing replay cache key")
    if total == 0:
        failures.append("candidate preflight contains no LLM calls")
    elif replayed != total:
        failures.append(f"candidate replay coverage is {replayed}/{total}, expected 100%")
    return total, replayed, failures


def _metamorphic_audit(result: UniverseRunResult) -> tuple[dict[str, Any], list[str]]:
    reports: dict[str, Any] = {}
    failures: list[str] = []
    required = {
        "sentence_shuffle",
        "paragraph_shuffle",
        "duplicate_evidence",
        "summary_injection",
        "whitespace_heading_change",
        "irrelevant_boilerplate_injection",
        "node_order_change",
    }
    for company in result.companies:
        path = Path(company.artifact_directory) / "metamorphic-audit.json"
        if not path.is_file():
            failures.append(f"{company.ticker}: missing metamorphic audit")
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        reports[company.ticker] = payload
        transformations = payload.get("transformations") or {}
        missing = required - set(transformations)
        if payload.get("passed") is not True or missing:
            failures.append(
                f"{company.ticker}: metamorphic gate failed or incomplete; missing={sorted(missing)}"
            )
            continue
        for name in required:
            item = transformations[name]
            if (
                item.get("atomic_key_jaccard") != 1.0
                or item.get("evidence_jaccard") != 1.0
                or item.get("claim_jaccard") != 1.0
                or item.get("score_delta") != 0.0
            ):
                failures.append(f"{company.ticker}: {name} is not zero-tolerance invariant")
    return reports, failures


def _compression_audit(result: UniverseRunResult) -> tuple[dict[str, Any], list[str]]:
    reports: dict[str, Any] = {}
    failures: list[str] = []
    for company in result.companies:
        path = Path(company.artifact_directory) / "compression-audit.json"
        if not path.is_file():
            failures.append(f"{company.ticker}: missing compression audit")
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        reports[company.ticker] = payload
        if (
            payload.get("passed") is not True
            or payload.get("claim_jaccard") != 1.0
            or payload.get("counterevidence_recall") != 1.0
            or payload.get("moat_score_delta") != 0.0
            or payload.get("factor_scores_equal") is not True
        ):
            failures.append(
                f"{company.ticker}: compression changed claims, counterevidence, or score"
            )
    return reports, failures


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Approve a 3..5-company repeated-run and adjacent-date MOAT preflight before a large run."
    )
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--baseline", action="append", required=True, help="DATE=sample run-result.json")
    parser.add_argument("--candidate", action="append", required=True, help="DATE=repeated sample run-result.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-repeat-spearman", type=float, default=1.0)
    parser.add_argument("--minimum-evidence-jaccard", type=float, default=1.0)
    parser.add_argument("--minimum-claim-jaccard", type=float, default=1.0)
    parser.add_argument("--minimum-adjacent-spearman", type=float, default=0.50)
    parser.add_argument("--minimum-positive-sample-companies", type=int, default=1)
    parser.add_argument("--minimum-distinct-scores-per-date", type=int, default=2)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    workspace_manifest_path = workspace / "workspace-manifest.json"
    workspace_manifest = json.loads(workspace_manifest_path.read_text(encoding="utf-8-sig"))
    dates = list(workspace_manifest["dates"])
    sample_tickers = list(workspace_manifest.get("preflight_sample_tickers") or [])
    if not 3 <= len(set(sample_tickers)) <= 5:
        raise ValueError("workspace preflight sample must contain 3 to 5 unique tickers")
    sample = set(sample_tickers)
    universe_rows = _read_csv(workspace / "inputs" / "universe.csv")
    universe_tickers = [row["stock_code"].zfill(6) for row in universe_rows]
    baseline_paths = _parse_run_arguments(args.baseline, "baseline")
    candidate_paths = _parse_run_arguments(args.candidate, "candidate")
    if set(baseline_paths) != set(dates) or set(candidate_paths) != set(dates):
        raise ValueError(
            f"preflight must cover every date: expected={dates}, "
            f"baseline={sorted(baseline_paths)}, candidate={sorted(candidate_paths)}"
        )

    baselines = {date: _load_run(baseline_paths[date], date, sample) for date in dates}
    candidates = {date: _load_run(candidate_paths[date], date, sample) for date in dates}
    contracts = [
        _run_contract(path)
        for date in dates
        for path in (baseline_paths[date], candidate_paths[date])
    ]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("sample runs used different execution contracts")

    failures: list[str] = []
    repeat_reports: dict[str, Any] = {}
    replay_total = 0
    replay_hits = 0
    metamorphic_reports: dict[str, Any] = {}
    compression_reports: dict[str, Any] = {}
    for date in dates:
        report = compare_runs(
            baselines[date],
            candidates[date],
            minimum_score_spearman=args.minimum_repeat_spearman,
            minimum_mean_evidence_jaccard=args.minimum_evidence_jaccard,
            minimum_mean_claim_jaccard=args.minimum_claim_jaccard,
            maximum_median_score_delta=0.0,
            maximum_company_score_delta=0.0,
        )
        repeat_reports[date] = report
        failures.extend(f"{date}: {failure}" for failure in report["failures"])
        before = {company.ticker: company for company in baselines[date].companies}
        after = {company.ticker: company for company in candidates[date].companies}
        changed_sources = [
            ticker
            for ticker in sample
            if before[ticker].source_document_ids != after[ticker].source_document_ids
        ]
        if changed_sources:
            failures.append(f"{date}: repeated run source documents changed for {sorted(changed_sources)}")
        total, hits, replay_failures = _candidate_replay_audit(candidates[date])
        replay_total += total
        replay_hits += hits
        failures.extend(f"{date}: {failure}" for failure in replay_failures)
        date_metamorphic, metamorphic_failures = _metamorphic_audit(baselines[date])
        metamorphic_reports[date] = date_metamorphic
        failures.extend(f"{date}: {failure}" for failure in metamorphic_failures)
        baseline_compression, baseline_compression_failures = _compression_audit(baselines[date])
        candidate_compression, candidate_compression_failures = _compression_audit(candidates[date])
        compression_reports[date] = {
            "baseline": baseline_compression,
            "candidate": candidate_compression,
        }
        failures.extend(f"{date}: baseline {failure}" for failure in baseline_compression_failures)
        failures.extend(f"{date}: candidate {failure}" for failure in candidate_compression_failures)
        baseline_scores = [
            float(company.moat_score.economic_moat_score)
            for company in baselines[date].companies
        ]
        if len(set(baseline_scores)) < args.minimum_distinct_scores_per_date:
            failures.append(
                f"{date}: sample has only {len(set(baseline_scores))} distinct MOAT score(s); "
                f"required={args.minimum_distinct_scores_per_date}"
            )

    positive_sample_tickers = sorted(
        {
            company.ticker
            for result in baselines.values()
            for company in result.companies
            if float(company.moat_score.economic_moat_score) > 0
        }
    )
    if len(positive_sample_tickers) < args.minimum_positive_sample_companies:
        failures.append(
            f"sample contains only {len(positive_sample_tickers)} positive-MOAT companies; "
            f"required={args.minimum_positive_sample_companies}"
        )

    adjacent_reports: list[dict[str, Any]] = []
    for prior_date, current_date in zip(dates, dates[1:]):
        prior = {company.ticker: float(company.moat_score.economic_moat_score) for company in baselines[prior_date].companies}
        current = {company.ticker: float(company.moat_score.economic_moat_score) for company in baselines[current_date].companies}
        ordered = sorted(sample)
        correlation = spearman([prior[ticker] for ticker in ordered], [current[ticker] for ticker in ordered])
        deltas = [abs(current[ticker] - prior[ticker]) for ticker in ordered]
        adjacent = {
            "prior_date": prior_date,
            "current_date": current_date,
            "company_count": len(ordered),
            "spearman": correlation,
            "median_absolute_delta": statistics.median(deltas),
        }
        adjacent_reports.append(adjacent)
        if correlation is None or correlation < args.minimum_adjacent_spearman:
            failures.append(
                f"{prior_date}->{current_date}: adjacent Spearman {correlation!r} "
                f"is below {args.minimum_adjacent_spearman:.3f}"
            )

    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "passed": not failures,
        "failures": failures,
        "runner_version": RUNNER_VERSION,
        "experiment_id": workspace_manifest.get("experiment_id"),
        "dates": dates,
        "sample_tickers": sorted(sample),
        "sample_size": len(sample),
        "approved_universe_tickers_sha256": ticker_set_sha256(universe_tickers),
        "execution_contract": contracts[0],
        "execution_contract_sha256": contract_sha256(contracts[0]),
        "minimum_repeat_spearman": args.minimum_repeat_spearman,
        "minimum_evidence_jaccard": args.minimum_evidence_jaccard,
        "minimum_claim_jaccard": args.minimum_claim_jaccard,
        "minimum_adjacent_spearman": args.minimum_adjacent_spearman,
        "minimum_positive_sample_companies": args.minimum_positive_sample_companies,
        "minimum_distinct_scores_per_date": args.minimum_distinct_scores_per_date,
        "positive_sample_tickers": positive_sample_tickers,
        "candidate_replay_calls": replay_total,
        "candidate_replay_hits": replay_hits,
        "repeatability": repeat_reports,
        "metamorphic": metamorphic_reports,
        "compression_invariance": compression_reports,
        "adjacent_stability": adjacent_reports,
        "inputs": {
            "universe_sha256": hashlib.sha256((workspace / "inputs" / "universe.csv").read_bytes()).hexdigest(),
            "dates_sha256": hashlib.sha256((workspace / "inputs" / "dates.csv").read_bytes()).hexdigest(),
        },
    }
    output = (args.output or workspace / "diagnostics" / "moat-preflight.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not failures:
        workspace_manifest["preflight_status"] = "PASSED"
        workspace_manifest["preflight_report"] = str(output)
        workspace_manifest["preflight_report_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        workspace_manifest_path.write_text(
            json.dumps(workspace_manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise RuntimeError("MOAT preflight failed; full-universe execution remains blocked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
