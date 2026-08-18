from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from moatrader.financial import DcfAssumptions, DcfEngine
from moatrader.backtest.historical import (
    HistoricalSignalStatus,
    compound_change_ratios,
    latest_revenue_continuity,
    latest_pit_filing_versions,
    quarterly_signal_dates,
    sample_statistics,
)
from moatrader.experiments.historical_validation import HistoricalValidationContract
from moatrader.experiments.integrity import snapshot_protected_files
from prepare_kr_dcf_manifest import assumptions_from_history


SEOUL = ZoneInfo("Asia/Seoul")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _truth(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "y"}


def _price_at_or_before(frame: pd.DataFrame, signal_date: date) -> pd.Series | None:
    eligible = frame[frame["date"] <= signal_date]
    if eligible.empty:
        return None
    point = eligible.iloc[-1]
    if (signal_date - point["date"]).days > 10:
        return None
    return point


def _forward_return(frame: pd.DataFrame, *, entry_date: date, horizon_days: int) -> tuple[float, date] | None:
    target = entry_date + timedelta(days=horizon_days)
    window = frame[(frame["date"] > entry_date) & (frame["date"] <= target)]
    if window.empty:
        return None
    exit_date = window.iloc[-1]["date"]
    if (target - exit_date).days > 10:
        return None
    value = compound_change_ratios(window["changes_ratio_percent"].tolist())
    return float(value), exit_date


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the deterministic v7 Data-PIT Cheap event study on the frozen 150-stock universe."
    )
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--filings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v6-contract", type=Path, required=True)
    parser.add_argument("--v6-stability-a", type=Path, required=True)
    parser.add_argument("--v6-stability-b", type=Path, required=True)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--last-signal", type=date.fromisoformat, default=date(2025, 9, 30))
    parser.add_argument("--horizon-days", type=int, default=77)
    parser.add_argument("--top-n", type=int, default=15)
    args = parser.parse_args()

    output = args.output.resolve()
    if "v7" not in output.as_posix().casefold():
        raise ValueError("historical backtest output must be a v7 path")
    if output.exists():
        raise FileExistsError(f"historical backtest output must be new: {output}")
    if args.horizon_days <= 0 or args.top_n <= 0:
        raise ValueError("horizon-days and top-n must be positive")

    universe_path = args.universe.resolve()
    price_path = args.prices.resolve()
    filing_root = args.filings.resolve()
    universe = _read_csv(universe_path)
    if len(universe) != 150:
        raise ValueError(f"backtest requires the frozen 150-stock universe, got {len(universe)}")
    universe_hash = _sha256(universe_path)
    price_manifest = json.loads((price_path.parent / "manifest.json").read_text(encoding="utf-8"))
    filing_manifest = json.loads((filing_root / "manifest.json").read_text(encoding="utf-8"))
    if price_manifest["universe_sha256"] != universe_hash or filing_manifest["universe_sha256"] != universe_hash:
        raise ValueError("v7 price/filing inputs do not match the frozen universe hash")

    v6_before = snapshot_protected_files(
        repository_root=Path.cwd(),
        contract_path=args.v6_contract.resolve(),
        stability_directories=[args.v6_stability_a.resolve(), args.v6_stability_b.resolve()],
    )
    signal_dates = quarterly_signal_dates(start=args.start, end=args.last_signal)
    contract = HistoricalValidationContract.create(
        frozen_on=date(2026, 8, 18),
        start_date=args.start,
        end_date=date(2025, 12, 31),
        signal_dates=signal_dates,
        universe_sha256=universe_hash,
        universe_selected_as_of=date.fromisoformat(universe[0]["as_of"]),
    )

    price_frame = pd.read_csv(price_path, encoding="utf-8-sig")
    price_frame["timestamp"] = pd.to_datetime(price_frame["timestamp"], utc=True)
    price_frame["date"] = price_frame["timestamp"].dt.tz_convert("Asia/Seoul").dt.date
    price_frame["ticker"] = price_frame["ticker"].astype(str).str.zfill(6)
    prices_by_ticker = {
        ticker: frame.sort_values("date").reset_index(drop=True)
        for ticker, frame in price_frame.groupby("ticker", sort=False)
    }

    filings_by_ticker: dict[str, list[dict[str, object]]] = defaultdict(list)
    for metadata_path in sorted(filing_root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["metadata_path"] = str(metadata_path)
        filings_by_ticker[str(metadata["ticker"]).zfill(6)].append(metadata)

    output.mkdir(parents=True)
    _write_json(output / "historical-validation-contract.json", contract.model_dump(mode="json"))
    signal_rows: list[dict[str, object]] = []
    engine = DcfEngine()
    for signal_date in signal_dates:
        cutoff = datetime.combine(signal_date, time.max, tzinfo=SEOUL)
        date_rows: list[dict[str, object]] = []
        for item in universe:
            ticker = item["stock_code"].zfill(6)
            row: dict[str, object] = {
                "signal_date": signal_date.isoformat(),
                "cutoff": cutoff.isoformat(),
                "ticker": ticker,
                "name": item["name"],
                "market": item["market"],
                "size_bucket": item["size_bucket"],
                "status": "",
                "status_detail": "",
                "price_date": "",
                "price": "",
                "market_cap": "",
                "listed_shares": "",
                "latest_rcept_no": "",
                "latest_available_at": "",
                "latest_fiscal_period_end": "",
                "latest_is_amendment": "",
                "original_archive_sha256": "",
                "xbrl_archive_sha256": "",
                "metric_coverage_count": "",
                "history_periods": "",
                "fair_value_per_share": "",
                "cheap": "",
                "cheap_rank": "",
                "cheap_quintile": "",
                "assumption_confidence": "",
                "terminal_value_share": "",
                "screening_exclusion_reasons": "",
            }
            if _truth(item["finance_hint"]) or _truth(item["holding_hint"]):
                row["status"] = HistoricalSignalStatus.EXCLUDED_ARCHETYPE
                row["status_detail"] = "FCFF_NOT_COMPARABLE_FOR_FINANCE_OR_HOLDING_HINT"
                date_rows.append(row)
                continue
            ticker_prices = prices_by_ticker.get(ticker)
            point = _price_at_or_before(ticker_prices, signal_date) if ticker_prices is not None else None
            if point is None:
                row["status"] = HistoricalSignalStatus.NOT_LISTED_OR_NO_PRICE
                date_rows.append(row)
                continue
            row.update(
                {
                    "price_date": point["date"].isoformat(),
                    "price": str(point["close"]),
                    "market_cap": str(point["market_cap"]),
                    "listed_shares": str(int(point["listed_shares"])),
                }
            )
            versions = latest_pit_filing_versions(filings_by_ticker.get(ticker, []), cutoff=cutoff)
            usable = [
                record
                for record in versions
                if record.get("metrics") is not None
                and record["metrics"].get("revenue") not in {None, "", "0", 0}
            ]
            if not usable:
                row["status"] = HistoricalSignalStatus.NO_PIT_FINANCIALS
                date_rows.append(row)
                continue
            latest = usable[-1]
            coverage = int(latest["metrics"]["metric_coverage_count"])
            row.update(
                {
                    "latest_rcept_no": latest["rcept_no"],
                    "latest_available_at": latest["available_at"],
                    "latest_fiscal_period_end": latest["fiscal_period_end"],
                    "latest_is_amendment": latest["is_amendment"],
                    "original_archive_sha256": latest["original_archive_sha256"],
                    "xbrl_archive_sha256": latest["xbrl_archive_sha256"] or "",
                    "metric_coverage_count": coverage,
                    "history_periods": "|".join(record["fiscal_period_end"] for record in usable),
                }
            )
            if coverage < 4:
                row["status"] = HistoricalSignalStatus.INSUFFICIENT_FINANCIAL_COVERAGE
                date_rows.append(row)
                continue
            try:
                history = [
                    (
                        date.fromisoformat(record["fiscal_period_end"]).year,
                        {
                            key: Decimal(str(record["metrics"][key]))
                            if record["metrics"].get(key) is not None
                            else None
                            for key in ("revenue", "ebit", "capex", "depreciation", "cash", "debt", "nwc")
                        },
                    )
                    for record in usable
                ]
                revenue_stable, revenue_ratio = latest_revenue_continuity(history)
                if not revenue_stable:
                    row["status"] = HistoricalSignalStatus.FINANCIAL_DISCONTINUITY
                    row["status_detail"] = (
                        "LATEST_REVENUE_RATIO_OUTSIDE_0.1_TO_10:"
                        f"{revenue_ratio}"
                    )
                    date_rows.append(row)
                    continue
                assumptions_payload, _audit = assumptions_from_history(
                    history,
                    item["size_bucket"],
                    Decimal(str(int(point["listed_shares"]))),
                )
                valuation = engine.value(DcfAssumptions.model_validate(assumptions_payload))
                cheap = valuation.fair_value_per_share / Decimal(str(point["close"])) - Decimal(1)
                row.update(
                    {
                        "fair_value_per_share": str(valuation.fair_value_per_share),
                        "cheap": str(cheap),
                        "assumption_confidence": str(valuation.assumption_confidence),
                        "terminal_value_share": str(valuation.terminal_value_share),
                        "screening_exclusion_reasons": "|".join(valuation.screening_exclusion_reasons),
                    }
                )
                if not valuation.screening_eligible:
                    row["status"] = HistoricalSignalStatus.DCF_SCREENING_EXCLUSION
                    row["status_detail"] = "|".join(valuation.screening_exclusion_reasons)
                else:
                    row["status"] = HistoricalSignalStatus.ELIGIBLE
            except Exception as exc:
                row["status"] = HistoricalSignalStatus.VALUATION_ERROR
                row["status_detail"] = f"{type(exc).__name__}: {exc}"
            date_rows.append(row)

        eligible = sorted(
            (row for row in date_rows if row["status"] == HistoricalSignalStatus.ELIGIBLE),
            key=lambda row: (Decimal(str(row["cheap"])), str(row["ticker"])),
            reverse=True,
        )
        count = len(eligible)
        for rank, row in enumerate(eligible, start=1):
            row["cheap_rank"] = rank
            row["cheap_quintile"] = min(5, math.floor((rank - 1) * 5 / count) + 1)
        signal_rows.extend(date_rows)
        print(
            f"signal {signal_date}: eligible={count} statuses={dict(Counter(str(row['status']) for row in date_rows))}",
            flush=True,
        )

    signal_path = output / "signals.csv"
    fields = list(signal_rows[0])
    with signal_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(signal_rows)
    signal_hash = _sha256(signal_path)
    _write_json(
        output / "signals-seal.json",
        {
            "schema_version": "v7-historical-signal-seal/1",
            "signals_sha256": signal_hash,
            "rank_signal": "CHEAP_ONLY",
            "return_data_accessed_before_seal": False,
            "llm_may_change_rank": False,
        },
    )

    # Return access starts only after the immutable signal CSV hash is sealed.
    sealed_rows = _read_csv(signal_path)
    event_rows: list[dict[str, object]] = []
    for signal_date in signal_dates:
        current = [row for row in sealed_rows if row["signal_date"] == signal_date.isoformat()]
        eligible = [row for row in current if row["status"] == HistoricalSignalStatus.ELIGIBLE]
        returns: dict[str, float] = {}
        exits: list[date] = []
        for row in eligible:
            frame = prices_by_ticker[row["ticker"]]
            result = _forward_return(
                frame,
                entry_date=date.fromisoformat(row["price_date"]),
                horizon_days=args.horizon_days,
            )
            if result is not None:
                returns[row["ticker"]] = result[0]
                exits.append(result[1])
        top = [returns[row["ticker"]] for row in eligible if int(row["cheap_rank"]) <= args.top_n and row["ticker"] in returns]
        bottom_threshold = max(1, len(eligible) - args.top_n + 1)
        bottom = [returns[row["ticker"]] for row in eligible if int(row["cheap_rank"]) >= bottom_threshold and row["ticker"] in returns]
        quintiles = {
            number: [returns[row["ticker"]] for row in eligible if int(row["cheap_quintile"]) == number and row["ticker"] in returns]
            for number in range(1, 6)
        }
        universe_returns: list[float] = []
        for item in universe:
            ticker = item["stock_code"].zfill(6)
            frame = prices_by_ticker.get(ticker)
            point = _price_at_or_before(frame, signal_date) if frame is not None else None
            if point is None:
                continue
            result = _forward_return(frame, entry_date=point["date"], horizon_days=args.horizon_days)
            if result is not None:
                universe_returns.append(result[0])
        mean = lambda values: sum(values) / len(values) if values else None
        top_return = mean(top)
        bottom_return = mean(bottom)
        benchmark = mean(universe_returns)
        event_rows.append(
            {
                "signal_date": signal_date.isoformat(),
                "horizon_days": args.horizon_days,
                "max_exit_date": max(exits).isoformat() if exits else "",
                "eligible_count": len(eligible),
                "return_observation_count": len(returns),
                "top_count": len(top),
                "bottom_count": len(bottom),
                "universe_return_count": len(universe_returns),
                "top15_mean_return": top_return,
                "bottom15_mean_return": bottom_return,
                "cheap_spread": top_return - bottom_return if top_return is not None and bottom_return is not None else None,
                "universe_equal_weight_return": benchmark,
                "top15_excess_return": top_return - benchmark if top_return is not None and benchmark is not None else None,
                **{f"cheap_quintile_{number}_return": mean(quintiles[number]) for number in range(1, 6)},
            }
        )

    event_path = output / "quarterly-event-results.csv"
    with event_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(event_rows[0]))
        writer.writeheader()
        writer.writerows(event_rows)
    top_values = [float(row["top15_mean_return"]) for row in event_rows if row["top15_mean_return"] is not None]
    spread_values = [float(row["cheap_spread"]) for row in event_rows if row["cheap_spread"] is not None]
    excess_values = [float(row["top15_excess_return"]) for row in event_rows if row["top15_excess_return"] is not None]
    benchmark_values = [float(row["universe_equal_weight_return"]) for row in event_rows if row["universe_equal_weight_return"] is not None]
    status_counts = Counter(row["status"] for row in signal_rows)
    summary = {
        "schema_version": "v7-data-pit-cheap-event-study/1",
        "validation_grade": "DATA_PIT_HISTORICAL",
        "historical_label": "PSEUDO_OOS_ONLY_FOR_ANY_MODERN_LLM_OVERLAY",
        "signal_period": [signal_dates[0].isoformat(), signal_dates[-1].isoformat()],
        "signal_count": len(signal_dates),
        "universe_count": len(universe),
        "fixed_2025_universe_backcast": True,
        "survivorship_and_membership_bias": True,
        "horizon_days": args.horizon_days,
        "top_n": args.top_n,
        "return_basis": "KRX_DAILY_CHANGES_RATIO_PRICE_RETURN_EXCLUDES_CASH_DISTRIBUTIONS",
        "top15_statistics": sample_statistics(top_values),
        "bottom_spread_statistics": sample_statistics(spread_values),
        "top15_excess_statistics": sample_statistics(excess_values),
        "universe_equal_weight_statistics": sample_statistics(benchmark_values),
        "signal_status_counts": dict(sorted(status_counts.items())),
        "signals_sha256": signal_hash,
        "event_results_sha256": _sha256(event_path),
        "llm_overlay_executed": False,
        "llm_overlay_reason": "NO_APPROVED_OPENAI_MODEL_RUN; DETERMINISTIC_ABLATION_IS_PRIMARY",
        "openai_api_key_present_at_runtime": bool(os.environ.get("OPENAI_API_KEY")),
    }
    _write_json(output / "summary.json", summary)
    ablation_path = output / "ablation.csv"
    with ablation_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["track", "status", "validation_grade", "rank_control", "notes"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "track": "B_DETERMINISTIC_CHEAP_ONLY",
                    "status": "COMPLETED",
                    "validation_grade": "DATA_PIT_HISTORICAL",
                    "rank_control": "CHEAP_ONLY",
                    "notes": "Official OpenDART cutoff filings plus pinned marcap PIT price/shares.",
                },
                {
                    "track": "A_LLM_OVERLAY",
                    "status": "NOT_RUN",
                    "validation_grade": "LLM_PIT_PSEUDO_OOS",
                    "rank_control": "FORBIDDEN",
                    "notes": "Requires exact cutoff citations, entailment, future traps, and anonymization stability; no approved model run was made.",
                },
            ]
        )

    v6_after = snapshot_protected_files(
        repository_root=Path.cwd(),
        contract_path=args.v6_contract.resolve(),
        stability_directories=[args.v6_stability_a.resolve(), args.v6_stability_b.resolve()],
    )
    v6_changed = sorted(key for key in set(v6_before) | set(v6_after) if v6_before.get(key) != v6_after.get(key))
    integrity = {
        "schema_version": "v7-build-v6-integrity/1",
        "v6_unchanged_during_backtest": not v6_changed,
        "changed_paths": v6_changed,
        "protected_file_count": len(v6_after),
        "protected_sha256": v6_after,
    }
    _write_json(output / "v6-integrity.json", integrity)
    if v6_changed:
        raise RuntimeError("v6 changed during v7 historical backtest: " + ", ".join(v6_changed))
    manifest = {
        "schema_version": "v7-historical-backtest-build/1",
        "contract_sha256": contract.contract_sha256,
        "universe_sha256": universe_hash,
        "price_manifest_sha256": _sha256(price_path.parent / "manifest.json"),
        "filing_manifest_sha256": _sha256(filing_root / "manifest.json"),
        "source_urls": [
            "https://opendart.fss.or.kr/api/list.json",
            "https://opendart.fss.or.kr/api/document.xml",
            "https://opendart.fss.or.kr/api/fnlttXbrl.xml",
            "https://github.com/FinanceData/marcap",
        ],
        "artifacts": {
            path.name: _sha256(path)
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "build-manifest.json"
        },
        "v6_unchanged": True,
    }
    _write_json(output / "build-manifest.json", manifest)
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
