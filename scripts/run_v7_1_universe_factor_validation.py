from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from moatrader.backtest.historical import latest_revenue_continuity, quarterly_signal_dates
from moatrader.backtest.universe_corrected import (
    build_historical_universe,
    extract_arcana_annual_metrics,
    forward_return,
    moving_block_bootstrap_mean,
    newey_west_mean,
    previous_price_point,
    residualize_cross_section,
    sha256_file,
    spearman_ic,
    trailing_beta,
    trailing_momentum,
)
from moatrader.financial import DcfAssumptions, DcfEngine
from prepare_kr_dcf_manifest import assumptions_from_history


REPOSITORY = Path(__file__).resolve().parents[1]
OLD_DETERMINISTIC = REPOSITORY / "data-lake/experiments/historical-validation-v7-2020-2025/results-v7-quality-gated"
OLD_LLM = REPOSITORY / "data-lake/experiments/historical-llm-overlay-v7-2020-2025"
OLD_PRICE_SOURCE = REPOSITORY / "data-lake/experiments/historical-validation-v7-2020-2025/prices/source"
OLD_DART_FILINGS = REPOSITORY / "data-lake/experiments/historical-validation-v7-2020-2025/opendart-original"
ARCANA_ROOT = Path(r"D:\Programming\python_example\Arcana\data-lake\silver\dart")
ARCANA_SNAPSHOTS = ARCANA_ROOT / "normalized-snapshots"
ARCANA_METADATA = ARCANA_ROOT / "kr_report_metadata.csv"
DEFAULT_OUTPUT = REPOSITORY / "data-lake/experiments/universe-corrected-residual-v7-1-2020-2025"
MARCAP_COMMIT = "5e8e4e57f3fcb129a6ff20751f643f67d3592c82"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def file_map(root: Path) -> dict[str, str]:
    return {
        path.relative_to(REPOSITORY).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def protected_v6_hashes() -> dict[str, str]:
    integrity = json.loads((OLD_DETERMINISTIC / "v6-integrity.json").read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for relative in integrity["protected_sha256"]:
        path = REPOSITORY / relative
        if not path.exists():
            raise FileNotFoundError(f"protected v6 file missing: {path}")
        result[relative] = sha256_file(path)
    return result


def seal_sources(output: Path) -> dict[str, Any]:
    old_v7 = {**file_map(OLD_DETERMINISTIC), **file_map(OLD_LLM)}
    payload = {
        "schema_version": "moatrader-v7.1-protected-source-seal/1",
        "v6": protected_v6_hashes(),
        "v7": old_v7,
        "v6_file_count": len(protected_v6_hashes()),
        "v7_file_count": len(old_v7),
        "mutation_policy": "READ_ONLY; ALL NEW ARTIFACTS UNDER V7.1 OUTPUT",
    }
    write_json(output / "seal-before.json", payload)
    return payload


def assert_seal_unchanged(before: dict[str, Any], output: Path) -> None:
    current_v6 = protected_v6_hashes()
    current_v7 = {**file_map(OLD_DETERMINISTIC), **file_map(OLD_LLM)}
    v6_changed = sorted(key for key in set(before["v6"]) | set(current_v6) if before["v6"].get(key) != current_v6.get(key))
    v7_changed = sorted(key for key in set(before["v7"]) | set(current_v7) if before["v7"].get(key) != current_v7.get(key))
    payload = {
        "schema_version": "moatrader-v7.1-protected-source-integrity/1",
        "v6_unchanged": not v6_changed,
        "v7_unchanged": not v7_changed,
        "v6_changed_paths": v6_changed,
        "v7_changed_paths": v7_changed,
    }
    write_json(output / "integrity-after.json", payload)
    if v6_changed or v7_changed:
        raise RuntimeError("protected v6/v7 artifact changed during v7.1 validation")


def read_marcap(output: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    columns = [
        "Code", "Name", "Close", "Amount", "Marcap", "Stocks", "Market", "MarketId",
        "Rank", "Date", "ChangesRatio", "Dept",
    ]
    pieces: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for year in range(2019, 2026):
        if year < 2020:
            path = output / "inputs/marcap" / f"marcap-{year}.parquet"
        else:
            path = OLD_PRICE_SOURCE / f"marcap-{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=columns)
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame["Code"] = frame["Code"].astype(str).str.zfill(6)
        pieces.append(frame)
        sources.append(
            {
                "year": year,
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "url": f"https://raw.githubusercontent.com/FinanceData/marcap/{MARCAP_COMMIT}/data/marcap-{year}.parquet",
                "reuse_mode": "V7_READ_ONLY" if year >= 2020 else "V7.1_NEW_INPUT",
            }
        )
    result = pd.concat(pieces, ignore_index=True)
    return result, sources


def load_current_sector_map(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    tables = pd.read_html(path, encoding="euc-kr")
    if not tables or tables[0].shape[1] < 4:
        raise ValueError("KRX KIND company list did not contain the expected four columns")
    frame = tables[0]
    codes = frame.iloc[:, 2].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(6)
    sectors = frame.iloc[:, 3].astype(str).str.strip()
    mapping = dict(zip(codes, sectors, strict=False))
    return mapping, {
        "path": str(path),
        "sha256": sha256_file(path),
        "row_count": len(frame),
        "source_url": "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13",
        "pit_grade": "CURRENT_2026_CLASSIFICATION_NOT_HISTORICAL_PIT",
        "allowed_use": "SECTOR_NEUTRALIZATION_SENSITIVITY_ONLY",
    }


class ArcanaFinancialStore:
    def __init__(self) -> None:
        metadata = pd.read_csv(
            ARCANA_METADATA,
            dtype={"stock_code": str, "rcept_no": str},
            low_memory=False,
        )
        metadata["stock_code"] = metadata["stock_code"].str.zfill(6)
        metadata["report_date"] = pd.to_datetime(metadata["report_date"])
        self.metadata = metadata[
            (metadata["source_type"] == "statement") & (metadata["fiscal_month"] == 12)
        ].copy()
        self.by_ticker = {
            str(ticker): frame.sort_values(["fiscal_year", "report_date", "rcept_no"])
            for ticker, frame in self.metadata.groupby("stock_code", sort=False)
        }
        self.cache: dict[tuple[str, int], dict[str, float | int | None] | None] = {}

    def load(self, ticker: str, fiscal_year: int) -> dict[str, float | int | None] | None:
        key = (ticker, fiscal_year)
        if key in self.cache:
            return self.cache[key]
        path = ARCANA_SNAPSHOTS / f"kr_normalized_{ticker}_{fiscal_year}.12.csv"
        if not path.exists():
            self.cache[key] = None
            return None
        try:
            value = extract_arcana_annual_metrics(pd.read_csv(path, low_memory=False))
        except Exception:
            value = None
        self.cache[key] = value
        return value

    def history(self, ticker: str, cutoff: date) -> tuple[list[tuple[int, dict[str, float | int | None]]], list[dict[str, Any]]]:
        ticker_rows = self.by_ticker.get(ticker)
        if ticker_rows is None:
            return [], []
        rows = ticker_rows[ticker_rows["report_date"] <= pd.Timestamp(cutoff)]
        rows = rows.drop_duplicates("fiscal_year", keep="last")
        history: list[tuple[int, dict[str, float | int | None]]] = []
        sources: list[dict[str, Any]] = []
        for row in rows.itertuples(index=False):
            metrics = self.load(ticker, int(row.fiscal_year))
            if metrics is None or not metrics.get("revenue"):
                continue
            history.append((int(row.fiscal_year), metrics))
            sources.append(
                {
                    "fiscal_year": int(row.fiscal_year),
                    "report_date": pd.Timestamp(row.report_date).date().isoformat(),
                    "rcept_no": str(row.rcept_no),
                    "source_url": str(row.source_url),
                    "snapshot": str(ARCANA_SNAPSHOTS / f"kr_normalized_{ticker}_{int(row.fiscal_year)}.12.csv"),
                }
            )
        return history, sources


def decimal_metrics(metrics: dict[str, float | int | None]) -> dict[str, Decimal | None]:
    return {
        key: Decimal(str(metrics[key])) if metrics.get(key) is not None else None
        for key in ("revenue", "ebit", "capex", "depreciation", "cash", "debt", "nwc")
    }


def cross_source_reproduction_audit(store: ArcanaFinancialStore, output: Path) -> dict[str, Any]:
    comparisons: dict[str, list[float]] = defaultdict(list)
    exact: Counter[str] = Counter()
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(OLD_DART_FILINGS.rglob("metadata.json")):
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(old.get("metrics"), dict):
            continue
        ticker = str(old["ticker"]).zfill(6)
        year_value = old.get("fiscal_year") or old["metrics"].get("fiscal_year")
        if year_value is None:
            continue
        year = int(year_value)
        arcana = store.load(ticker, year)
        if not arcana:
            continue
        item: dict[str, Any] = {"ticker": ticker, "fiscal_year": year, "rcept_no": old["rcept_no"]}
        for metric in ("revenue", "ebit", "capex", "depreciation", "cash", "debt", "nwc"):
            left = old.get("metrics", {}).get(metric)
            right = arcana.get(metric)
            if left is None or right is None or float(left) == 0:
                continue
            ratio = abs(float(right) / float(left))
            counts[metric] += 1
            comparisons[metric].append(ratio)
            is_exact = math.isclose(float(right), float(left), rel_tol=1e-9, abs_tol=1.0)
            exact[metric] += int(is_exact)
            item[f"{metric}_ratio"] = ratio
            item[f"{metric}_exact"] = is_exact
        rows.append(item)
    pd.DataFrame(rows).to_csv(output / "audits/arcana-vs-opendart-xbrl.csv", index=False, encoding="utf-8-sig")
    metrics = {}
    for metric, values in comparisons.items():
        metrics[metric] = {
            "n": counts[metric],
            "exact_count": exact[metric],
            "exact_rate": exact[metric] / counts[metric],
            "median_ratio": float(np.median(values)),
            "p10_ratio": float(np.quantile(values, 0.10)),
            "p90_ratio": float(np.quantile(values, 0.90)),
        }
    result = {
        "schema_version": "moatrader-v7.1-financial-source-reproduction/1",
        "comparison": "REUSED_ARCANA_NORMALIZED_DART_VS_FROZEN_OFFICIAL_OPENDART_XBRL",
        "metrics": metrics,
        "gate": {
            "rank_reproduction_exactly_proven": False,
            "reason": "DEBT_AND_NWC_AGGREGATION_DIFFERENCES; NEW_UNIVERSE_HAS_NO_FROZEN_XBRL_BASELINE",
        },
    }
    write_json(output / "audits/arcana-vs-opendart-xbrl-summary.json", result)
    return result


def llm_return_free_audit(output: Path) -> dict[str, Any]:
    failures = json.loads((OLD_LLM / "validation-failures.json").read_text(encoding="utf-8"))
    failure_by_pack: dict[str, str] = {}
    for item in failures:
        failure_by_pack.setdefault(str(item["pack_id"]), str(item["error"]))
    packs = {
        path.stem: path
        for path in (OLD_LLM / "packs").rglob("*.json")
        if path.parent.name == "candidates"
    }
    validated_paths = sorted((OLD_LLM / "validated").glob("*.json"))
    exact_total = 0
    exact_found = 0
    offset_total = 0
    offset_exact = 0
    entailment_passed_claims = 0
    for path in validated_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        pack = json.loads(packs[record["pack_id"]].read_text(encoding="utf-8"))
        units = {str(item["unit_id"]): str(item["text"]) for item in pack["excerpts"]}
        for claim in record["validated_claims"]:
            exact_total += 1
            entailment_passed_claims += 1
            text = units.get(str(claim["unit_id"]), "")
            quote = str(claim["exact_quote"])
            exact_found += int(quote in text)
            if claim.get("char_start") is not None and claim.get("char_end") is not None:
                offset_total += 1
                offset_exact += int(text[int(claim["char_start"]) : int(claim["char_end"])] == quote)
    all_ids = sorted(packs)
    rng = np.random.default_rng(42)
    selected = rng.choice(np.asarray(all_ids), size=min(200, len(all_ids)), replace=False)
    audit_rows = []
    validated_ids = {path.stem for path in validated_paths}
    for pack_id in sorted(selected):
        audit_rows.append(
            {
                "pack_id": pack_id,
                "pipeline_outcome": "PASS" if pack_id in validated_ids else "FAIL_CLOSED",
                "failure_reason": failure_by_pack.get(pack_id, ""),
                "human_axis_label": "",
                "human_direction_label": "",
                "human_entailment_label": "",
                "contains_return_label": False,
            }
        )
    pd.DataFrame(audit_rows).to_csv(
        output / "llm-validator/return-free-audit-sample-200.csv",
        index=False,
        encoding="utf-8-sig",
    )
    evaluation = json.loads((OLD_LLM / "evaluation.json").read_text(encoding="utf-8"))
    attempted = int(evaluation["sample"]["eligible_pack_count"])
    passed = int(evaluation["sample"]["validated_pack_count"])
    failure_reasons = evaluation.get("validation_failure_reasons", {})
    anonymization_failures = int(failure_reasons.get("ANONYMIZATION_CLASSIFICATION_INSTABILITY", 0)) + int(
        failure_reasons.get("ANONYMIZATION_CONFIDENCE_INSTABILITY", 0)
    )
    anonymization_stable = attempted - anonymization_failures
    result = {
        "schema_version": "moatrader-v7.1-return-free-llm-validator-audit/1",
        "models": evaluation["models"],
        "return_labels_used": False,
        "eligible_pack_count": attempted,
        "fail_closed_pack_count": attempted - passed,
        "pack_acceptance_rate": passed / attempted,
        "validated_claim_count": entailment_passed_claims,
        "exact_quote_presence": {"n": exact_total, "accuracy": exact_found / exact_total if exact_total else None},
        "exact_quote_offset": {"n": offset_total, "accuracy": offset_exact / offset_total if offset_total else None},
        "independent_entailment_gate_precision": {
            "observed_on_accepted_claims": 1.0 if entailment_passed_claims else None,
            "n": entailment_passed_claims,
            "caveat": "PASS RATE OF THE INDEPENDENT GATE, NOT HUMAN-GOLD PRECISION",
        },
        "anonymization_stability": {
            "stable_pack_count": anonymization_stable,
            "attempted_pack_count": attempted,
            "strict_classification_and_confidence_stability_rate": anonymization_stable / attempted,
            "classification_instability_count": int(
                failure_reasons.get("ANONYMIZATION_CLASSIFICATION_INSTABILITY", 0)
            ),
            "confidence_instability_count": int(
                failure_reasons.get("ANONYMIZATION_CONFIDENCE_INSTABILITY", 0)
            ),
        },
        "human_gold_classification_accuracy": None,
        "human_gold_status": "NOT_OBSERVED; 200-ROW RETURN-FREE ANNOTATION PACK CREATED",
        "promotion_gate": "FAIL_PENDING_HUMAN_GOLD",
    }
    write_json(output / "llm-validator/quality-summary.json", result)
    return result


def factors_for_row(
    item: pd.Series,
    ticker_frame: pd.DataFrame,
    market_return: pd.Series,
    latest: dict[str, float | int | None] | None,
    signal_date: date,
) -> dict[str, float | None]:
    assets = float(latest.get("total_assets") or 0) if latest else 0.0
    equity = float(latest.get("total_equity") or 0) if latest else 0.0
    ebit = float(latest.get("ebit") or 0) if latest else 0.0
    cfo = float(latest.get("cfo") or 0) if latest else 0.0
    debt = float(latest.get("debt") or 0) if latest else 0.0
    market_cap = float(item["market_cap"])
    return {
        "value_btm": equity / market_cap if equity > 0 and market_cap > 0 else None,
        "quality_roa_cfo_leverage": (ebit / assets) + (cfo / assets) - (debt / assets) if assets > 0 else None,
        "momentum_12_1": trailing_momentum(ticker_frame, as_of=signal_date),
        "size_log_mcap": math.log(market_cap) if market_cap > 0 else None,
        "beta_252": trailing_beta(ticker_frame, market_return, as_of=signal_date),
    }


def build_signals(
    marcap: pd.DataFrame,
    sector_map: dict[str, str],
    store: ArcanaFinancialStore,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_dates = quarterly_signal_dates(start=date(2020, 1, 1), end=date(2025, 9, 30))
    market_daily = (
        marcap[marcap["MarketId"].isin(["STK", "KSQ"])]
        .groupby(["MarketId", "Date"])["ChangesRatio"]
        .mean()
        .div(100.0)
    )
    price_groups = marcap.groupby("Code", sort=False)
    prices_by_ticker: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    masters: list[pd.DataFrame] = []
    eligibles: list[pd.DataFrame] = []
    selections: list[pd.DataFrame] = []
    engine = DcfEngine()
    for signal_date in signal_dates:
        universe_window = marcap[
            (marcap["Date"] >= pd.Timestamp(signal_date) - pd.Timedelta(days=400))
            & (marcap["Date"] <= pd.Timestamp(signal_date))
        ]
        build = build_historical_universe(universe_window, as_of=signal_date)
        for frame, collection in ((build.master, masters), (build.eligible, eligibles), (build.selected, selections)):
            copy = frame.copy()
            copy.insert(0, "signal_date", signal_date.isoformat())
            collection.append(copy)
        for _, item in build.selected.iterrows():
            ticker = str(item["stock_code"]).zfill(6)
            if ticker not in prices_by_ticker:
                prices_by_ticker[ticker] = price_groups.get_group(ticker).sort_values("Date")
            prices = prices_by_ticker[ticker]
            point = previous_price_point(prices, as_of=signal_date)
            history, sources = store.history(ticker, signal_date)
            latest = history[-1][1] if history else None
            market_id = "STK" if item["market"] == "KOSPI" else "KSQ"
            factor_values = factors_for_row(
                item,
                prices,
                market_daily.loc[market_id],
                latest,
                signal_date,
            )
            row: dict[str, Any] = {
                "signal_date": signal_date.isoformat(),
                "universe_actual_as_of": build.actual_as_of.isoformat(),
                "ticker": ticker,
                "name": item["name"],
                "market": item["market"],
                "size_bucket": str(item["size_bucket"]),
                "current_sector": sector_map.get(ticker, "UNKNOWN_CURRENT_SECTOR"),
                "current_sector_pit": False,
                "price_date": pd.Timestamp(point["Date"]).date().isoformat() if point is not None else "",
                "price": float(point["Close"]) if point is not None else None,
                "market_cap": float(item["market_cap"]),
                "listed_shares": int(item["listed_shares"]),
                "finance_hint": bool(item["finance_hint"]),
                "holding_hint": bool(item["holding_hint"]),
                "status": "",
                "status_detail": "",
                "latest_rcept_no": sources[-1]["rcept_no"] if sources else "",
                "latest_report_date": sources[-1]["report_date"] if sources else "",
                "latest_fiscal_year": history[-1][0] if history else None,
                "history_years": "|".join(str(year) for year, _ in history),
                "metric_coverage_count": int(latest["metric_coverage_count"]) if latest else 0,
                "fair_value_per_share": None,
                "cheap": None,
                "cheap_rank": None,
                "terminal_value_share": None,
                "assumption_confidence": None,
                **factor_values,
            }
            if item["finance_hint"] or item["holding_hint"]:
                row["status"] = "EXCLUDED_ARCHETYPE"
                row["status_detail"] = "FCFF_NOT_COMPARABLE_FOR_FINANCE_OR_HOLDING_HINT"
            elif point is None:
                row["status"] = "NO_PRICE"
            elif not history:
                row["status"] = "NO_PIT_FINANCIALS"
            elif int(latest["metric_coverage_count"]) < 4:
                row["status"] = "INSUFFICIENT_FINANCIAL_COVERAGE"
            else:
                try:
                    decimal_history = [(year, decimal_metrics(metrics)) for year, metrics in history]
                    stable, ratio = latest_revenue_continuity(decimal_history)
                    if not stable:
                        row["status"] = "FINANCIAL_DISCONTINUITY"
                        row["status_detail"] = f"LATEST_REVENUE_RATIO_OUTSIDE_0.1_TO_10:{ratio}"
                    else:
                        assumptions, _audit = assumptions_from_history(
                            decimal_history,
                            str(item["size_bucket"]),
                            Decimal(str(int(item["listed_shares"]))),
                        )
                        valuation = engine.value(DcfAssumptions.model_validate(assumptions))
                        cheap = valuation.fair_value_per_share / Decimal(str(point["Close"])) - Decimal(1)
                        row.update(
                            {
                                "fair_value_per_share": float(valuation.fair_value_per_share),
                                "cheap": float(cheap),
                                "terminal_value_share": float(valuation.terminal_value_share),
                                "assumption_confidence": float(valuation.assumption_confidence),
                                "status": "ELIGIBLE" if valuation.screening_eligible else "DCF_SCREENING_EXCLUSION",
                                "status_detail": "|".join(valuation.screening_exclusion_reasons),
                            }
                        )
                except Exception as exc:
                    row["status"] = "VALUATION_ERROR"
                    row["status_detail"] = f"{type(exc).__name__}: {exc}"
            rows.append(row)
        print(f"signal {signal_date}: {Counter(row['status'] for row in rows if row['signal_date'] == signal_date.isoformat())}", flush=True)
    signals = pd.DataFrame(rows)
    residual_specs = {
        "cheap_resid_value": (["value_btm"], []),
        "cheap_resid_value_quality": (["value_btm", "quality_roa_cfo_leverage"], []),
        "cheap_resid_value_quality_momentum": (["value_btm", "quality_roa_cfo_leverage", "momentum_12_1"], []),
        "cheap_resid_value_quality_momentum_size": (["value_btm", "quality_roa_cfo_leverage", "momentum_12_1", "size_log_mcap"], []),
        "cheap_resid_full": (["value_btm", "quality_roa_cfo_leverage", "momentum_12_1", "size_log_mcap", "beta_252"], ["current_sector"]),
        "cheap_resid_sector_only": ([], ["current_sector"]),
    }
    signals["cheap_rank_high"] = np.nan
    signals["value_quality_composite"] = np.nan
    for signal_date, index in signals.groupby("signal_date").groups.items():
        group = signals.loc[index]
        valid = group["status"].eq("ELIGIBLE")
        rank = group.loc[valid, "cheap"].rank(method="first", ascending=False)
        signals.loc[rank.index, "cheap_rank"] = rank
        signals.loc[index, "cheap_rank_high"] = group["cheap"].rank(pct=True)
        value_rank = group["value_btm"].rank(pct=True)
        quality_rank = group["quality_roa_cfo_leverage"].rank(pct=True)
        signals.loc[index, "value_quality_composite"] = (value_rank + quality_rank) / 2.0
        eligible_group = signals.loc[index].copy()
        eligible_group.loc[~valid, "cheap"] = np.nan
        for name, (numeric, categorical) in residual_specs.items():
            signals.loc[index, name] = residualize_cross_section(
                eligible_group,
                target="cheap",
                numeric_controls=numeric,
                categorical_controls=categorical,
            )
    universe_frame = pd.concat(masters, ignore_index=True)
    eligible_frame = pd.concat(eligibles, ignore_index=True)
    selected_frame = pd.concat(selections, ignore_index=True)
    (output / "universes").mkdir(parents=True, exist_ok=True)
    universe_frame.to_csv(output / "universes/master-by-date.csv", index=False, encoding="utf-8-sig")
    eligible_frame.to_csv(output / "universes/eligible-by-date.csv", index=False, encoding="utf-8-sig")
    selected_frame.to_csv(output / "universes/selected-150-by-date.csv", index=False, encoding="utf-8-sig")
    return signals, selected_frame


def evaluate_returns(signals: pd.DataFrame, marcap: pd.DataFrame, output: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    signals = signals.copy()
    price_groups = marcap.groupby("Code", sort=False)
    prices_by_ticker: dict[str, pd.DataFrame] = {}
    signals["forward_77d_return"] = np.nan
    signals["exit_date"] = ""
    for index, row in signals.iterrows():
        if not row["price_date"]:
            continue
        ticker = str(row["ticker"])
        if ticker not in prices_by_ticker:
            prices_by_ticker[ticker] = price_groups.get_group(ticker).sort_values("Date")
        prices = prices_by_ticker[ticker]
        result = forward_return(prices, entry_date=date.fromisoformat(row["price_date"]), horizon_days=77)
        if result is not None:
            signals.at[index, "forward_77d_return"] = result[0]
            signals.at[index, "exit_date"] = result[1].isoformat()

    signal_columns = [
        "cheap", "cheap_resid_value", "cheap_resid_value_quality",
        "cheap_resid_value_quality_momentum", "cheap_resid_value_quality_momentum_size",
        "cheap_resid_full",
    ]
    event_rows: list[dict[str, Any]] = []
    for signal_date, group in signals.groupby("signal_date", sort=True):
        all_returns = group["forward_77d_return"].dropna()
        eligible = group[group["status"].eq("ELIGIBLE") & group["forward_77d_return"].notna()].copy()
        event: dict[str, Any] = {
            "signal_date": signal_date,
            "selected_universe_count": len(group),
            "selected_return_count": len(all_returns),
            "eligible_common_count": len(eligible),
            "benchmark_equal_weight_all": all_returns.mean(),
            "benchmark_equal_weight_common": eligible["forward_77d_return"].mean(),
        }
        for column in signal_columns:
            event[f"{column}_ic"] = spearman_ic(eligible, column, "forward_77d_return")
            ranked = eligible[[column, "forward_77d_return"]].dropna().sort_values(column)
            if len(ranked) >= 10:
                ranked["quintile"] = pd.qcut(
                    ranked[column].rank(method="first"),
                    5,
                    labels=[1, 2, 3, 4, 5],
                ).astype(int)
                means = ranked.groupby("quintile")["forward_77d_return"].mean()
                for quintile in range(1, 6):
                    event[f"{column}_q{quintile}"] = means.get(quintile, np.nan)
                event[f"{column}_q5_minus_q1"] = means.get(5, np.nan) - means.get(1, np.nan)
        cheap_sorted = eligible.sort_values("cheap", ascending=False)
        event["benchmark_cheap_top15"] = cheap_sorted.head(15)["forward_77d_return"].mean()

        def top_quintile(column: str) -> float:
            subset = eligible[[column, "forward_77d_return"]].dropna().sort_values(column, ascending=False)
            return float(subset.head(max(1, math.ceil(len(subset) / 5)))["forward_77d_return"].mean()) if len(subset) else np.nan

        event["benchmark_value_top_quintile"] = top_quintile("value_btm")
        event["benchmark_quality_top_quintile"] = top_quintile("quality_roa_cfo_leverage")
        event["benchmark_momentum_top_quintile"] = top_quintile("momentum_12_1")
        event["benchmark_value_quality_top_quintile"] = top_quintile("value_quality_composite")
        event["benchmark_sector_neutral_cheap_top_quintile"] = top_quintile("cheap_resid_sector_only")
        for benchmark in (
            "benchmark_cheap_top15",
            "benchmark_value_top_quintile",
            "benchmark_quality_top_quintile",
            "benchmark_momentum_top_quintile",
            "benchmark_value_quality_top_quintile",
            "benchmark_sector_neutral_cheap_top_quintile",
        ):
            event[f"{benchmark}_excess_vs_common"] = (
                event[benchmark] - event["benchmark_equal_weight_common"]
            )
        event_rows.append(event)
    events = pd.DataFrame(event_rows)
    summary: dict[str, Any] = {
        "schema_version": "moatrader-v7.1-universe-corrected-factor-validation/1",
        "validation_grade": "DATA_PIT_HISTORICAL_WITH_CURRENT_SECTOR_SENSITIVITY",
        "historical_label": "PIT_HISTORICAL_VALIDATION; NOT_TRUE_LIVE_OOS",
        "horizon_days": 77,
        "overlapping_window_inference": {"newey_west_lag": 1, "moving_block_length_quarters": 4},
        "series": {},
    }
    metric_columns = [
        column for column in events.columns
        if column not in {"signal_date", "selected_universe_count", "selected_return_count", "eligible_common_count"}
    ]
    for column in metric_columns:
        values = pd.to_numeric(events[column], errors="coerce").dropna().tolist()
        summary["series"][column] = {
            "newey_west": newey_west_mean(values, lag=1),
            "moving_block_bootstrap": moving_block_bootstrap_mean(values, block_length=4, repetitions=10_000, seed=42),
        }
    summary["quintile_profiles"] = {}
    for column in signal_columns:
        profile = [float(pd.to_numeric(events[f"{column}_q{number}"], errors="coerce").mean()) for number in range(1, 6)]
        summary["quintile_profiles"][column] = {
            "q1_to_q5_mean_returns": profile,
            "strictly_monotonic_ascending": all(left < right for left, right in zip(profile, profile[1:])),
            "q5_minus_q1": profile[-1] - profile[0],
        }
    raw = summary["series"].get("cheap_q5_minus_q1", {}).get("newey_west", {}).get("mean", float("nan"))
    full = summary["series"].get("cheap_resid_full_q5_minus_q1", {}).get("newey_west", {}).get("mean", float("nan"))
    full_ci = summary["series"].get("cheap_resid_full_q5_minus_q1", {}).get("moving_block_bootstrap", {})
    if math.isfinite(float(full)) and float(full) > 0 and float(full_ci.get("ci_low", -1)) > 0:
        judgment = "PROMOTE_TO_FRESH_HOLDOUT"
    elif math.isfinite(float(raw)) and float(raw) > 0:
        judgment = "KEEP_AS_RESEARCH_CANDIDATE; RESIDUAL_SIGNIFICANCE_NOT_PROVEN"
    else:
        judgment = "DO_NOT_PROMOTE"
    summary["pre_registered_judgment"] = judgment
    summary["sector_limitation"] = "SECTOR CONTROL USES CURRENT 2026 KRX KIND CLASSIFICATION AND IS SENSITIVITY-ONLY"
    signals.to_csv(output / "results/signals-with-returns.csv", index=False, encoding="utf-8-sig")
    events.to_csv(output / "results/quarterly-factor-results.csv", index=False, encoding="utf-8-sig")
    write_json(output / "results/statistical-summary.json", summary)
    return events, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run universe-corrected v7.1 Cheap/factor/validator validation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if "v7-1" not in output.as_posix().casefold() and "v7.1" not in output.as_posix().casefold():
        raise ValueError("output must be an explicit v7.1 path")
    results = output / "results"
    if (output / "FINAL-RESULT.json").exists():
        raise FileExistsError(f"completed v7.1 result is immutable: {output}")
    results.mkdir(parents=True, exist_ok=True)
    (output / "audits").mkdir(parents=True, exist_ok=True)
    (output / "llm-validator").mkdir(parents=True, exist_ok=True)
    before = seal_sources(output)
    marcap, marcap_sources = read_marcap(output)
    sector_map, sector_manifest = load_current_sector_map(output / "inputs/krx-kind-current-company-list.xls")
    write_json(
        output / "input-manifest.json",
        {
            "schema_version": "moatrader-v7.1-inputs/1",
            "marcap_provider_commit": MARCAP_COMMIT,
            "marcap_sources": marcap_sources,
            "sector_source": sector_manifest,
            "dart_normalized_metadata": {
                "path": str(ARCANA_METADATA),
                "sha256": sha256_file(ARCANA_METADATA),
                "reuse_mode": "READ_ONLY_ARCANA_DART_CACHE",
            },
        },
    )
    store = ArcanaFinancialStore()
    cross_source = cross_source_reproduction_audit(store, output)
    signals, selected = build_signals(marcap, sector_map, store, output)

    # Exact port checkpoint against the original 2025-08-01 universe.
    checkpoint_window = marcap[
        (marcap["Date"] >= pd.Timestamp("2024-06-01")) & (marcap["Date"] <= pd.Timestamp("2025-08-01"))
    ]
    checkpoint = build_historical_universe(checkpoint_window, as_of=date(2025, 8, 1))
    original_master = pd.read_csv(
        Path(r"D:\Programming\python_example\MoatPoC\universe_master.csv"), dtype={"stock_code": str}
    )
    original_eligible = pd.read_csv(
        Path(r"D:\Programming\python_example\MoatPoC\universe_eligible.csv"), dtype={"stock_code": str}
    )
    original_selected = pd.read_csv(
        Path(r"D:\Programming\python_example\MoatPoC\universe.csv"), dtype={"stock_code": str}
    )
    for frame in (original_master, original_eligible, original_selected):
        frame["stock_code"] = frame["stock_code"].str.zfill(6)
    checkpoint_audit = {
        "as_of": "2025-08-01",
        "master_count_reproduced": len(checkpoint.master),
        "master_count_original": len(original_master),
        "master_membership_exact": set(checkpoint.master.stock_code) == set(original_master.stock_code),
        "eligible_count_reproduced": len(checkpoint.eligible),
        "eligible_count_original": len(original_eligible),
        "eligible_membership_exact": set(checkpoint.eligible.stock_code) == set(original_eligible.stock_code),
        "selected_count_reproduced": len(checkpoint.selected),
        "selected_count_original": len(original_selected),
        "selected_membership_exact": set(checkpoint.selected.stock_code) == set(original_selected.stock_code),
        "selected_order_exact": checkpoint.selected.stock_code.tolist() == original_selected.stock_code.tolist(),
    }
    write_json(output / "audits/universe-port-2025-08-01.json", checkpoint_audit)

    pre_return_path = output / "results/signals-pre-return.csv"
    signals.to_csv(pre_return_path, index=False, encoding="utf-8-sig")
    write_json(
        output / "results/signals-seal.json",
        {
            "schema_version": "moatrader-v7.1-signal-seal/1",
            "signals_sha256": sha256_file(pre_return_path),
            "rank_signal": "FROZEN_CHEAP_WITH_PRE_REGISTERED_RESIDUAL_SPECS",
            "return_labels_used_to_define_signal": False,
            "formula_changed_after_observing_returns": False,
            "residual_controls": ["VALUE", "QUALITY", "MOMENTUM_12_1", "SIZE", "BETA_252", "CURRENT_SECTOR_SENSITIVITY"],
        },
    )
    events, statistical = evaluate_returns(signals, marcap, output)
    llm_quality = llm_return_free_audit(output)
    status_counts = signals["status"].value_counts().to_dict()
    final = {
        "schema_version": "moatrader-v7.1-final-validation/1",
        "period": [signals.signal_date.min(), signals.signal_date.max()],
        "signal_date_count": signals.signal_date.nunique(),
        "selected_observation_count": len(signals),
        "unique_selected_ticker_count": signals.ticker.nunique(),
        "status_counts": status_counts,
        "universe_checkpoint": checkpoint_audit,
        "financial_reproduction_gate": cross_source["gate"],
        "statistical_judgment": statistical["pre_registered_judgment"],
        "llm_validator_promotion_gate": llm_quality["promotion_gate"],
        "overall_judgment": (
            "RESEARCH_CANDIDATE_ONLY_UNTIL_FINANCIAL_RANK_REPRODUCTION_AND_HUMAN_GOLD_LLM_GATE_PASS"
        ),
        "true_live_oos": False,
        "label": "PIT_HISTORICAL_VALIDATION / MODERN_LLM_PSEUDO_OOS",
    }
    write_json(output / "FINAL-RESULT.json", final)
    assert_seal_unchanged(before, output)
    write_json(
        output / "build-manifest.json",
        {
            "schema_version": "moatrader-v7.1-build/1",
            "artifacts": {
                path.relative_to(output).as_posix(): sha256_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "build-manifest.json"
            },
            "v6_unchanged": True,
            "sealed_v7_unchanged": True,
            "credentials_persisted": False,
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
