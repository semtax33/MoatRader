from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path

from moatrader.evidence.models import EvidenceCard, EvidenceDirection, STRUCTURAL_MOAT_TYPES
from moatrader.evidence.validation import validate_moat_score
from moatrader.runner.engine import RUNNER_VERSION
from moatrader.runner.models import CompanyRunStatus, UniverseRunResult
try:
    from scripts.merge_kr_signal_panel import decimal_percentiles
except ModuleNotFoundError:  # Direct `python scripts\...py` execution.
    from merge_kr_signal_panel import decimal_percentiles


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _signal_expected(company: object) -> tuple[bool, Decimal | None, list[str]]:
    score = getattr(company, "moat_score", None)
    dcf = getattr(company, "dcf", None)
    price = getattr(company, "current_price", None)
    fair_value = dcf.fair_value_per_share if dcf else None
    ratio = price / fair_value if price is not None and fair_value is not None and fair_value > 0 else None
    margin = Decimal(1) - ratio if ratio is not None else None
    reasons: list[str] = []
    if getattr(company, "status", None) != CompanyRunStatus.COMPLETE:
        reasons.append(getattr(company, "status").value)
    if score is None:
        reasons.append("NO_MOAT_SCORE")
    elif Decimal(str(score.economic_moat_score)) < Decimal("5"):
        reasons.append("MOAT_BELOW_5")
    if margin is None:
        reasons.append("NO_POSITIVE_DCF")
    elif margin < Decimal("0.20"):
        reasons.append("MARGIN_BELOW_20PCT")
    if dcf is not None and not dcf.screening_eligible:
        reasons.extend(dcf.screening_exclusion_reasons)
    if score is not None and Decimal(str(score.model_confidence)) < Decimal("0.50"):
        reasons.append("CONFIDENCE_BELOW_0_50")
    if score is not None and Decimal(str(score.document_coverage.moat_evidence_coverage or 0)) < Decimal("0.50"):
        reasons.append("MOAT_COVERAGE_BELOW_0_50")
    eligible = not reasons
    return eligible, None, reasons


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--signals", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    signals = (args.signals or workspace / "signals" / "moat-dcf-signals.csv").resolve()
    output = (args.output or workspace / "signals" / "signal-audit.json").resolve()
    rows = read_csv(signals)
    universe = read_csv(workspace / "inputs" / "universe.csv")
    date_rows = read_csv(workspace / "inputs" / "dates.csv")
    dates = tuple((row.get("date") or row.get("as_of") or "").strip() for row in date_rows)
    expected_tickers = {row["stock_code"].zfill(6) for row in universe}
    expected_keys = {(as_of, ticker) for as_of in dates for ticker in expected_tickers}
    actual_keys = {(row["date"], row["ticker"]) for row in rows}
    if len(rows) != len(expected_keys) or actual_keys != expected_keys:
        raise RuntimeError("signal panel does not match the exact input universe x dates grid")
    for row in rows:
        eligible = row["signal_eligible"] == "1"
        if eligible != bool(row["signal"]):
            raise RuntimeError(f"signal missingness disagrees with eligibility: {row['date']} {row['ticker']}")
        if row["signal"]:
            Decimal(row["signal"])

    future_sources: list[dict[str, str]] = []
    future_prices: list[dict[str, str]] = []
    external_paths: list[dict[str, str]] = []
    stale_periods: list[dict[str, object]] = []
    manifest_counts: dict[str, int] = {}
    for as_of in dates:
        cutoff = date.fromisoformat(as_of)
        manifest_rows = read_csv(workspace / "date-inputs" / as_of / "universe-manifest.csv")
        manifest_counts[as_of] = len({row["ticker"] for row in manifest_rows})
        for row in manifest_rows:
            paths = [Path(row["input"]).resolve(), Path(row["metadata"]).resolve()]
            if row.get("dcf_assumptions"):
                paths.append(Path(row["dcf_assumptions"]).resolve())
            for path in paths:
                if not _inside(path, workspace):
                    external_paths.append({"date": as_of, "ticker": row["ticker"], "path": str(path)})
            metadata = json.loads(Path(row["metadata"]).read_text(encoding="utf-8-sig"))
            available_at = datetime.fromisoformat(str(metadata["available_at"]).replace("Z", "+00:00"))
            if available_at.date() > cutoff:
                future_sources.append({"date": as_of, "ticker": row["ticker"], "available_at": metadata["available_at"]})
            if datetime.fromisoformat(row["price_as_of"]).date() > cutoff:
                future_prices.append({"date": as_of, "ticker": row["ticker"], "price_as_of": row["price_as_of"]})
        latest_period_by_ticker: dict[str, date] = {}
        for row in manifest_rows:
            metadata = json.loads(Path(row["metadata"]).read_text(encoding="utf-8-sig"))
            period_text = row.get("selection_period_end") or metadata.get("period_end")
            if not period_text:
                continue
            period_end = date.fromisoformat(str(period_text)[:10])
            ticker = row["ticker"]
            latest_period_by_ticker[ticker] = max(
                period_end,
                latest_period_by_ticker.get(ticker, period_end),
            )
        for ticker, period_end in latest_period_by_ticker.items():
            age = (cutoff - period_end).days
            if age > 370:
                stale_periods.append({"date": as_of, "ticker": ticker, "age_days": age})
    if future_sources or future_prices or external_paths:
        raise RuntimeError("PIT or fresh-workspace path audit failed")

    signal_manifest = json.loads(signals.with_name("signal-manifest.json").read_text(encoding="utf-8"))
    run_results: dict[str, dict[str, object]] = {}
    score_contract_errors: list[str] = []
    signal_errors: list[str] = []
    raw_audit_errors: list[str] = []
    routing_errors: list[str] = []
    models: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    live_response_ids: list[str] = []
    replay_cache_outputs: dict[str, str] = {}
    replayed_call_count = 0

    for as_of, paths in signal_manifest["run_results"].items():
        by_ticker: dict[str, object] = {}
        for result_path_text in paths:
            result_path = Path(result_path_text).resolve()
            if not _inside(result_path, workspace):
                raise RuntimeError(f"run result is outside workspace: {result_path}")
            result = UniverseRunResult.model_validate_json(result_path.read_text(encoding="utf-8-sig"))
            if result.as_of.date().isoformat() != as_of:
                raise RuntimeError(
                    f"run as_of mismatch for {as_of}: {result.run_id} has {result.as_of.isoformat()}"
                )
            for company in result.companies:
                if company.ticker in by_ticker:
                    raise RuntimeError(f"duplicate run-result company for {as_of}/{company.ticker}")
                if company.runner_version != RUNNER_VERSION:
                    raise RuntimeError(
                        f"incompatible runner version for {as_of}/{company.ticker}: "
                        f"{company.runner_version!r}, expected {RUNNER_VERSION}"
                    )
                by_ticker[company.ticker] = company
        run_results[as_of] = by_ticker

    row_by_key = {(row["date"], row["ticker"]): row for row in rows}
    for as_of, by_ticker in run_results.items():
        for ticker, company in by_ticker.items():
            row = row_by_key.get((as_of, ticker))
            if row is None:
                continue
            eligible, _expected_signal, _reasons = _signal_expected(company)
            if eligible != (row["signal_eligible"] == "1"):
                signal_errors.append(f"{as_of}/{ticker}")
            if company.status != CompanyRunStatus.COMPLETE or company.moat_score is None:
                continue
            artifact = Path(company.artifact_directory).resolve()
            if not _inside(artifact, workspace):
                raw_audit_errors.append(f"{as_of}/{ticker}: artifact directory is outside workspace")
                continue
            evidence_path = artifact / "evidence.jsonl"
            cards = [
                EvidenceCard.model_validate_json(line)
                for line in evidence_path.read_text(encoding="utf-8-sig").splitlines()
                if line.strip()
            ]
            errors = validate_moat_score(company.moat_score, cards)
            if errors:
                score_contract_errors.extend(f"{as_of}/{ticker}: {error}" for error in errors)
            if company.moat_score.llm_proposed_score is None:
                score_contract_errors.append(f"{as_of}/{ticker}: missing llm_proposed_score audit field")
            for mechanism in company.moat_score.mechanisms:
                if mechanism.evidence_type not in STRUCTURAL_MOAT_TYPES:
                    score_contract_errors.append(f"{as_of}/{ticker}: non-structural mechanism")
            if company.moat_score.economic_moat_score > 0 and not any(
                card.direction == EvidenceDirection.MOAT_POSITIVE
                and card.evidence_type in STRUCTURAL_MOAT_TYPES
                for card in cards
            ):
                score_contract_errors.append(f"{as_of}/{ticker}: positive score without structural evidence")

            audit_path = artifact / "llm-calls.jsonl"
            for line in audit_path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                call = json.loads(line)
                models[call["model"]] += 1
                tasks[call["task"]] += 1
                if call["task"] in {"LOCAL_EVIDENCE_EXTRACTION", "FINAL_MOAT_SCORING"} and not str(
                    call["model"]
                ).startswith(("gpt-5-luna", "gpt-5.6-luna", "fixture")):
                    routing_errors.append(f"{as_of}/{ticker}: {call['task']} -> {call['model']}")
                if call["task"] == "SECTION_SUMMARY" and not str(call["model"]).startswith(
                    ("gpt-5-nano", "fixture")
                ):
                    routing_errors.append(f"{as_of}/{ticker}: SECTION_SUMMARY -> {call['model']}")
                if call.get("replayed"):
                    replayed_call_count += 1
                    cache_key = str(call.get("replay_cache_key") or "")
                    normalized_hash = str(call.get("normalized_output_sha256") or "")
                    if not cache_key:
                        raw_audit_errors.append(f"{as_of}/{ticker}: replayed call has no cache key")
                    elif cache_key in replay_cache_outputs and replay_cache_outputs[cache_key] != normalized_hash:
                        raw_audit_errors.append(f"{as_of}/{ticker}: replay cache key maps to different outputs")
                    else:
                        replay_cache_outputs[cache_key] = normalized_hash
                elif call.get("response_id"):
                    live_response_ids.append(call["response_id"])
                for key, value in call.get("usage", {}).items():
                    usage[key] += int(value or 0)
                raw_path = Path(call.get("raw_response_path") or "")
                if not _inside(raw_path.resolve(), artifact):
                    raw_audit_errors.append(f"{as_of}/{ticker}: raw LLM artifact is outside company directory")
                elif not raw_path.is_file() or sha256(raw_path) == "":
                    raw_audit_errors.append(f"{as_of}/{ticker}: missing raw LLM artifact")
                else:
                    payload = json.loads(raw_path.read_text(encoding="utf-8"))
                    actual_hash = hashlib.sha256(payload["raw_output_text"].encode("utf-8")).hexdigest()
                    if actual_hash != call.get("raw_response_sha256"):
                        raw_audit_errors.append(f"{as_of}/{ticker}: raw response hash mismatch")
                    normalized = json.dumps(
                        payload["normalized_output"],
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    normalized_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                    if normalized_hash != call.get("normalized_output_sha256"):
                        raw_audit_errors.append(f"{as_of}/{ticker}: normalized response hash mismatch")

    for as_of in dates:
        eligible_rows = [row for row in rows if row["date"] == as_of and row["signal_eligible"] == "1"]
        moat_percentiles = decimal_percentiles([Decimal(row["moat_score"]) for row in eligible_rows])
        value_percentiles = decimal_percentiles([Decimal(row["margin_of_safety"]) for row in eligible_rows])
        for row, moat_percentile, value_percentile in zip(
            eligible_rows,
            moat_percentiles,
            value_percentiles,
            strict=True,
        ):
            expected = (moat_percentile + value_percentile) / Decimal(2)
            if (
                Decimal(row["signal"]) != expected
                or Decimal(row["moat_percentile"]) != moat_percentile
                or Decimal(row["value_percentile"]) != value_percentile
            ):
                signal_errors.append(f"{as_of}/{row['ticker']}: percentile signal mismatch")

    if score_contract_errors or signal_errors or raw_audit_errors or routing_errors:
        raise RuntimeError(
            "semantic audit failed: "
            f"score={len(score_contract_errors)} signal={len(signal_errors)} "
            f"raw={len(raw_audit_errors)} routing={len(routing_errors)}"
        )
    if len(live_response_ids) != len(set(live_response_ids)):
        raise RuntimeError("duplicate live provider response IDs")
    status_counts = Counter(row["status"] for row in rows)
    eligible_counts = Counter(row["date"] for row in rows if row["signal_eligible"] == "1")
    audit = {
        "schema_version": "moatrader-kr-signal-audit/2",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "fresh_workspace_only": True,
        "row_count": len(rows),
        "unique_date_ticker_count": len(actual_keys),
        "universe_count": len(expected_tickers),
        "dates": list(dates),
        "rows_per_date": dict(Counter(row["date"] for row in rows)),
        "status_counts": dict(status_counts),
        "eligible_counts": dict(eligible_counts),
        "manifest_counts": manifest_counts,
        "pit_audit": {
            "future_source_count": len(future_sources),
            "future_price_count": len(future_prices),
            "external_path_count": len(external_paths),
            "stale_period_count": len(stale_periods),
            "stale_periods": stale_periods,
        },
        "semantic_audit": {
            "score_contract_error_count": 0,
            "signal_recomputation_error_count": 0,
            "raw_llm_audit_error_count": 0,
            "model_routing_error_count": 0,
        },
        "llm_audit": {
            "call_count": sum(tasks.values()),
            "live_call_count": sum(tasks.values()) - replayed_call_count,
            "replayed_call_count": replayed_call_count,
            "unique_live_response_id_count": len(set(live_response_ids)),
            "unique_replay_cache_key_count": len(replay_cache_outputs),
            "models": dict(models),
            "tasks": dict(tasks),
            "usage": dict(usage),
        },
        "sha256": {
            "universe.csv": sha256(workspace / "inputs" / "universe.csv"),
            "dates.csv": sha256(workspace / "inputs" / "dates.csv"),
            "moat-dcf-signals.csv": sha256(signals),
            "signal-coverage.csv": sha256(signals.with_name("signal-coverage.csv")),
            "signal-manifest.json": sha256(signals.with_name("signal-manifest.json")),
        },
    }
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
