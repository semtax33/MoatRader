from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from moatrader.backtest.universe_corrected import (
    build_historical_universe,
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
from moatrader.expectations.driver_signals import (
    DRIVER_GRIDS,
    TAX_RATE,
    DriverName,
    ImpliedSolutionStatus,
    all_driver_solutions,
    supported_driver_estimate,
)
from moatrader.financial.arcana_pit import ArcanaAnnualPitStore


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_DEFAULT = REPOSITORY / "data-lake/experiments/driver-expectation-gap-v7-2-2018-2019"
V7_1_ROOT = REPOSITORY / "data-lake/experiments/universe-corrected-residual-v7-1-2020-2025"
ARCANA_DART = Path(r"D:\Programming\python_example\Arcana\data-lake\silver\dart")
ARCANA_METADATA = ARCANA_DART / "kr_report_metadata.csv"
ARCANA_SNAPSHOTS = ARCANA_DART / "normalized-snapshots"
MARCAP_COMMIT = "5e8e4e57f3fcb129a6ff20751f643f67d3592c82"
SIGNAL_START = date(2018, 4, 30)
SIGNAL_END = date(2019, 9, 30)
HORIZON_DAYS = 77
HAC_LAG = 3
BLOCK_LENGTH = 4


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


def current_hashes(expected: dict[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in expected:
        path = REPOSITORY / relative
        if not path.exists():
            raise FileNotFoundError(f"protected file missing: {path}")
        result[relative] = sha256_file(path)
    return result


def protect_prior_versions(output: Path) -> dict[str, Any]:
    inherited = json.loads((V7_1_ROOT / "seal-before.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": "moatrader-v7.2-prior-version-seal/1",
        "v6": current_hashes(inherited["v6"]),
        "v7": current_hashes(inherited["v7"]),
        "v7_1": file_map(V7_1_ROOT),
        "mutation_policy": "V6_V7_V7.1_READ_ONLY; V7.2_OUTPUT_ONLY",
    }
    write_json(output / "seal-before.json", payload)
    return payload


def assert_prior_versions_unchanged(before: dict[str, Any], output: Path) -> None:
    current = {
        "v6": current_hashes(before["v6"]),
        "v7": current_hashes(before["v7"]),
        "v7_1": file_map(V7_1_ROOT),
    }
    changes = {
        version: sorted(
            key
            for key in set(before[version]) | set(current[version])
            if before[version].get(key) != current[version].get(key)
        )
        for version in current
    }
    payload = {
        "schema_version": "moatrader-v7.2-prior-version-integrity/1",
        **{f"{version}_unchanged": not paths for version, paths in changes.items()},
        "changed_paths": changes,
    }
    write_json(output / "integrity-after.json", payload)
    if any(changes.values()):
        raise RuntimeError("protected v6/v7/v7.1 artifacts changed")


def signal_dates() -> list[date]:
    return [timestamp.date() for timestamp in pd.date_range(SIGNAL_START, SIGNAL_END, freq="ME")]


def freeze_contract(output: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "moatrader-v7.2-driver-expectation-gap-freeze/1",
        "frozen_on": "2026-08-19",
        "hypothesis": (
            "Among firms at the same book-to-market level, favorable evidence-supported minus "
            "market-implied Growth, Margin, ROIIC, or CAP predicts higher forward returns."
        ),
        "prohibited_research_period": ["2020-01-01", "2025-12-31"],
        "test_period": [SIGNAL_START.isoformat(), SIGNAL_END.isoformat()],
        "signal_dates": [value.isoformat() for value in signal_dates()],
        "formation_frequency": "MONTH_END",
        "forward_horizon_calendar_days": HORIZON_DAYS,
        "driver_gap_direction": "EVIDENCE_SUPPORTED_MINUS_MARKET_IMPLIED; HIGH_IS_FAVORABLE",
        "reverse_method": (
            "ONE_DRIVER_AT_A_TIME; OTHER THREE DRIVERS HELD AT PRICE_BLIND_SUPPORTED VALUES; "
            "LINEAR_INTERPOLATION_INSIDE_FROZEN_GRID; CENSORED SOLUTIONS EXCLUDED"
        ),
        "supported_driver_rules": {
            "growth": "median of up to last 3 annual revenue growth observations, clipped [-10%,15%]",
            "margin": "median of up to last 3 annual NOPAT margins, clipped [-15%,30%]",
            "roiic": (
                "median positive delta-NOPAT/delta-invested-capital for material positive capital "
                "changes, fallback latest ROIC, clipped [3%,50%]"
            ),
            "cap": (
                "round(clip(3 + 20*max(median historical ROIC-WACC,0) + min(latest positive-spread "
                "streak,3),3,15))"
            ),
            "invested_capital": "total equity + debt - cash; 10% revenue floor only when non-positive",
            "tax_rate": str(TAX_RATE),
            "wacc": {"SMALL": "0.12", "MID": "0.105", "LARGE": "0.095"},
        },
        "implied_driver_grids": {
            driver.value: [str(value) for value in values]
            for driver, values in DRIVER_GRIDS.items()
        },
        "primary_signal": "rank-normal driver gap residual after same-date book-to-market control",
        "secondary_signal": (
            "rank-normal residual after book-to-market, quality, 12-1 momentum, log size, and beta"
        ),
        "primary_tests": [
            f"{driver.value.lower()}_gap_vn_mean_spearman_ic_and_q5_minus_q1"
            for driver in DriverName
        ],
        "multiple_testing": "HOLM_TWO_SIDED_ACROSS_FOUR_PRIMARY_MEAN_IC_TESTS",
        "overlap_inference": {
            "newey_west_lag_months": HAC_LAG,
            "moving_block_length_months": BLOCK_LENGTH,
            "bootstrap_repetitions": 10_000,
            "bootstrap_seed": 42,
        },
        "promotion_gate": (
            "positive mean IC, positive Q5-Q1, Holm p<0.05, IC block-bootstrap lower bound>0, "
            "and monotonic average Q1-Q5"
        ),
        "llm_used_for_alpha": False,
        "returns_permitted_before_signal_seal": False,
    }
    write_json(output / "frozen-contract.json", payload)
    return payload


def load_marcap(output: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    columns = [
        "Code", "Name", "Close", "Amount", "Marcap", "Stocks", "Market", "MarketId",
        "Rank", "Date", "ChangesRatio", "Dept",
    ]
    pieces = []
    sources = []
    for year in (2017, 2018, 2019):
        path = output / "inputs/marcap" / f"marcap-{year}.parquet"
        frame = pd.read_parquet(path, columns=columns)
        frame["Code"] = frame["Code"].astype(str).str.zfill(6)
        frame["Date"] = pd.to_datetime(frame["Date"])
        pieces.append(frame)
        sources.append(
            {
                "year": year,
                "path": str(path),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "url": f"https://raw.githubusercontent.com/FinanceData/marcap/{MARCAP_COMMIT}/data/marcap-{year}.parquet",
            }
        )
    marcap = pd.concat(pieces, ignore_index=True)
    if marcap["Date"].max() >= pd.Timestamp("2020-01-01"):
        raise ValueError("v7.2 holdout input must not contain 2020+ market data")
    return marcap, sources


def decimal_history(
    history: list[tuple[int, dict[str, float | int | None]]],
) -> list[tuple[int, dict[str, Decimal | None]]]:
    keys = ("revenue", "ebit", "cash", "debt", "total_equity", "total_assets", "cfo")
    return [
        (
            year,
            {
                key: Decimal(str(metrics[key])) if metrics.get(key) is not None else None
                for key in keys
            },
        )
        for year, metrics in history
    ]


def build_pre_return_signals(
    *,
    marcap: pd.DataFrame,
    store: ArcanaAnnualPitStore,
    output: Path,
) -> pd.DataFrame:
    price_groups = marcap.groupby("Code", sort=False)
    price_cache: dict[str, pd.DataFrame] = {}
    market_daily = (
        marcap[marcap["MarketId"].isin(["STK", "KSQ"])]
        .groupby(["MarketId", "Date"])["ChangesRatio"]
        .mean()
        .div(100.0)
    )
    rows: list[dict[str, Any]] = []
    universe_frames: list[pd.DataFrame] = []
    for signal_date in signal_dates():
        universe_window = marcap[
            (marcap["Date"] >= pd.Timestamp(signal_date) - pd.Timedelta(days=400))
            & (marcap["Date"] <= pd.Timestamp(signal_date))
        ]
        build = build_historical_universe(universe_window, as_of=signal_date)
        selected_copy = build.selected.copy()
        selected_copy.insert(0, "signal_date", signal_date.isoformat())
        universe_frames.append(selected_copy)
        date_rows: list[dict[str, Any]] = []
        for _, item in build.selected.iterrows():
            ticker = str(item["stock_code"]).zfill(6)
            if ticker not in price_cache:
                price_cache[ticker] = price_groups.get_group(ticker).sort_values("Date")
            prices = price_cache[ticker]
            point = previous_price_point(prices, as_of=signal_date)
            history, sources = store.history(ticker, signal_date)
            latest = history[-1][1] if history else None
            market_id = "STK" if item["market"] == "KOSPI" else "KSQ"
            market_return = market_daily.loc[market_id]
            market_cap = float(item["market_cap"])
            assets = float(latest.get("total_assets") or 0) if latest else 0.0
            equity = float(latest.get("total_equity") or 0) if latest else 0.0
            ebit = float(latest.get("ebit") or 0) if latest else 0.0
            cfo = float(latest.get("cfo") or 0) if latest else 0.0
            debt = float(latest.get("debt") or 0) if latest else 0.0
            row: dict[str, Any] = {
                "signal_date": signal_date.isoformat(),
                "universe_actual_as_of": build.actual_as_of.isoformat(),
                "ticker": ticker,
                "name": item["name"],
                "market": item["market"],
                "size_bucket": str(item["size_bucket"]),
                "price_date": pd.Timestamp(point["Date"]).date().isoformat() if point is not None else "",
                "price": float(point["Close"]) if point is not None else None,
                "market_cap": market_cap,
                "listed_shares": int(item["listed_shares"]),
                "finance_hint": bool(item["finance_hint"]),
                "holding_hint": bool(item["holding_hint"]),
                "status": "",
                "status_detail": "",
                "latest_fiscal_year": history[-1][0] if history else None,
                "latest_report_date": sources[-1]["report_date"] if sources else "",
                "latest_rcept_no": sources[-1]["rcept_no"] if sources else "",
                "history_years": "|".join(str(year) for year, _metrics in history),
                "value_btm": equity / market_cap if equity > 0 and market_cap > 0 else None,
                "quality_roa_cfo_leverage": (ebit + cfo - debt) / assets if assets > 0 else None,
                "momentum_12_1": trailing_momentum(prices, as_of=signal_date),
                "size_log_mcap": math.log(market_cap) if market_cap > 0 else None,
                "beta_252": trailing_beta(prices, market_return, as_of=signal_date),
            }
            if item["finance_hint"] or item["holding_hint"]:
                row["status"] = "EXCLUDED_ARCHETYPE"
            elif point is None:
                row["status"] = "NO_PRICE"
            else:
                try:
                    estimate = supported_driver_estimate(
                        decimal_history(history),
                        size_bucket=str(item["size_bucket"]),
                        diluted_shares=Decimal(str(int(item["listed_shares"]))),
                    )
                    if estimate.base_nopat_margin <= 0:
                        raise ValueError("latest NOPAT margin must be positive for driver reverse DCF")
                    solutions = all_driver_solutions(
                        estimate=estimate,
                        current_price=Decimal(str(point["Close"])),
                    )
                    row.update(
                        {
                            "status": "TESTABLE",
                            "supported_growth": float(estimate.growth),
                            "supported_margin": float(estimate.margin),
                            "supported_roiic": float(estimate.roiic),
                            "supported_cap": estimate.cap_years,
                            "invested_capital_floor_used": estimate.invested_capital_floor_used,
                            "roiic_fallback_used": estimate.roiic_fallback_used,
                        }
                    )
                    for driver, solution in solutions.items():
                        prefix = driver.value.lower()
                        row[f"{prefix}_implied"] = float(solution.implied) if solution.implied is not None else None
                        row[f"{prefix}_gap_raw"] = float(solution.gap) if solution.gap is not None else None
                        row[f"{prefix}_gap_status"] = solution.status.value
                        row[f"{prefix}_gap_signal"] = (
                            float(solution.gap)
                            if solution.status == ImpliedSolutionStatus.SOLVED and solution.gap is not None
                            else None
                        )
                        row[f"{prefix}_implied_price_error"] = (
                            float(solution.relative_price_error)
                            if solution.relative_price_error is not None
                            else None
                        )
                except Exception as exc:
                    row["status"] = "UNTESTABLE_FINANCIAL_OR_MODEL"
                    row["status_detail"] = f"{type(exc).__name__}: {exc}"
            date_rows.append(row)
        rows.extend(date_rows)
        print(f"signal {signal_date}: {dict(Counter(row['status'] for row in date_rows))}", flush=True)
    signals = pd.DataFrame(rows)
    for signal_date, indices in signals.groupby("signal_date").groups.items():
        group = signals.loc[indices].copy()
        for driver in DriverName:
            prefix = driver.value.lower()
            target = f"{prefix}_gap_signal"
            if target not in group:
                continue
            signals.loc[indices, f"{prefix}_gap_vn"] = residualize_cross_section(
                group,
                target=target,
                numeric_controls=["value_btm"],
            )
            signals.loc[indices, f"{prefix}_gap_full_resid"] = residualize_cross_section(
                group,
                target=target,
                numeric_controls=[
                    "value_btm", "quality_roa_cfo_leverage", "momentum_12_1",
                    "size_log_mcap", "beta_252",
                ],
            )
    pd.concat(universe_frames, ignore_index=True).to_csv(
        output / "universes/selected-150-by-date.csv", index=False, encoding="utf-8-sig"
    )
    return signals


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        value = min(1.0, (count - index) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def evaluate_after_seal(
    *,
    signals: pd.DataFrame,
    marcap: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = signals.copy()
    result["forward_77d_return"] = np.nan
    result["exit_date"] = ""
    groups = marcap.groupby("Code", sort=False)
    cache: dict[str, pd.DataFrame] = {}
    for index, row in result.iterrows():
        if not row["price_date"]:
            continue
        ticker = str(row["ticker"]).zfill(6)
        if ticker not in cache:
            cache[ticker] = groups.get_group(ticker).sort_values("Date")
        value = forward_return(
            cache[ticker],
            entry_date=date.fromisoformat(str(row["price_date"])),
            horizon_days=HORIZON_DAYS,
        )
        if value is not None:
            if value[1] >= date(2020, 1, 1):
                raise ValueError("v7.2 holdout return crossed into prohibited 2020+ period")
            result.at[index, "forward_77d_return"] = value[0]
            result.at[index, "exit_date"] = value[1].isoformat()

    specs = []
    for driver in DriverName:
        prefix = driver.value.lower()
        specs.extend([f"{prefix}_gap_signal", f"{prefix}_gap_vn", f"{prefix}_gap_full_resid"])
    event_rows: list[dict[str, Any]] = []
    for signal_date, group in result.groupby("signal_date", sort=True):
        event: dict[str, Any] = {
            "signal_date": signal_date,
            "selected_count": len(group),
            "all_equal_weight_return": group["forward_77d_return"].mean(),
        }
        value_sample = group[["value_btm", "forward_77d_return"]].dropna().sort_values("value_btm", ascending=False)
        event["value_top_quintile_return"] = value_sample.head(max(1, math.ceil(len(value_sample) / 5)))["forward_77d_return"].mean()
        for signal in specs:
            sample = group[[signal, "forward_77d_return"]].dropna().sort_values(signal)
            event[f"{signal}_count"] = len(sample)
            event[f"{signal}_common_return"] = sample["forward_77d_return"].mean()
            event[f"{signal}_ic"] = spearman_ic(sample, signal, "forward_77d_return")
            if len(sample) >= 10:
                sample["quintile"] = pd.qcut(
                    sample[signal].rank(method="first"),
                    5,
                    labels=[1, 2, 3, 4, 5],
                ).astype(int)
                means = sample.groupby("quintile")["forward_77d_return"].mean()
                for quintile in range(1, 6):
                    event[f"{signal}_q{quintile}"] = means.get(quintile, np.nan)
                event[f"{signal}_q5_minus_q1"] = means.get(5, np.nan) - means.get(1, np.nan)
                top = sample[sample["quintile"] == 5]["forward_77d_return"].mean()
                event[f"{signal}_top_quintile_return"] = top
                event[f"{signal}_top_quintile_excess"] = top - sample["forward_77d_return"].mean()
        event_rows.append(event)
    events = pd.DataFrame(event_rows)
    summary: dict[str, Any] = {
        "schema_version": "moatrader-v7.2-driver-gap-holdout/1",
        "validation_grade": "PRE_2020_PIT_HISTORICAL_HOLDOUT_NOT_LIVE_OOS",
        "period": [SIGNAL_START.isoformat(), SIGNAL_END.isoformat()],
        "signal_date_count": len(events),
        "horizon_days": HORIZON_DAYS,
        "series": {},
        "quintile_profiles": {},
    }
    excluded_columns = {"signal_date", "selected_count"}
    for column in events.columns:
        if column in excluded_columns or column.endswith("_count"):
            continue
        values = pd.to_numeric(events[column], errors="coerce").dropna().tolist()
        summary["series"][column] = {
            "newey_west": newey_west_mean(values, lag=HAC_LAG),
            "moving_block_bootstrap": moving_block_bootstrap_mean(
                values,
                block_length=BLOCK_LENGTH,
                repetitions=10_000,
                seed=42,
            ),
        }
    primary_p: dict[str, float] = {}
    for driver in DriverName:
        signal = f"{driver.value.lower()}_gap_vn"
        ic_key = f"{signal}_ic"
        nw = summary["series"][ic_key]["newey_west"]
        t_value = float(nw["t"])
        primary_p[driver.value] = float(2 * stats.norm.sf(abs(t_value))) if math.isfinite(t_value) else 1.0
        profile = [
            float(pd.to_numeric(events[f"{signal}_q{quintile}"], errors="coerce").mean())
            for quintile in range(1, 6)
        ]
        summary["quintile_profiles"][driver.value] = {
            "q1_to_q5_mean_returns": profile,
            "q5_minus_q1": profile[-1] - profile[0],
            "strictly_monotonic_ascending": all(left < right for left, right in zip(profile, profile[1:])),
        }
    adjusted = holm_adjust(primary_p)
    summary["multiple_testing"] = {
        driver: {"two_sided_hac_normal_p": primary_p[driver], "holm_adjusted_p": adjusted[driver]}
        for driver in primary_p
    }
    judgments = {}
    for driver in DriverName:
        signal = f"{driver.value.lower()}_gap_vn"
        ic_stats = summary["series"][f"{signal}_ic"]
        spread_stats = summary["series"][f"{signal}_q5_minus_q1"]
        profile = summary["quintile_profiles"][driver.value]
        strict = (
            float(ic_stats["newey_west"]["mean"]) > 0
            and float(spread_stats["newey_west"]["mean"]) > 0
            and adjusted[driver.value] < 0.05
            and float(ic_stats["moving_block_bootstrap"]["ci_low"]) > 0
            and profile["strictly_monotonic_ascending"]
        )
        weak = float(ic_stats["newey_west"]["mean"]) > 0 and float(spread_stats["newey_west"]["mean"]) > 0
        judgments[driver.value] = "PROMOTE" if strict else ("WEAK_POSITIVE_NOT_PROMOTED" if weak else "FAILED")
    summary["driver_judgments"] = judgments
    summary["overall_judgment"] = (
        "AT_LEAST_ONE_DRIVER_PROMOTED" if "PROMOTE" in judgments.values() else "NO_DRIVER_CLEARED_PROMOTION_GATE"
    )
    result.to_csv(output / "results/signals-with-returns.csv", index=False, encoding="utf-8-sig")
    events.to_csv(output / "results/monthly-driver-gap-results.csv", index=False, encoding="utf-8-sig")
    write_json(output / "results/statistical-summary.json", summary)
    return events, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pre-2020 driver-level Expectation Gap holdout.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    if "v7-2" not in output.as_posix().casefold() and "v7.2" not in output.as_posix().casefold():
        raise ValueError("output must be an explicit v7.2 path")
    if (output / "FINAL-RESULT.json").exists():
        raise FileExistsError(f"completed result is immutable: {output}")
    (output / "results").mkdir(parents=True, exist_ok=True)
    (output / "universes").mkdir(parents=True, exist_ok=True)
    before = protect_prior_versions(output)
    contract = freeze_contract(output)
    marcap, marcap_sources = load_marcap(output)
    store = ArcanaAnnualPitStore(metadata_path=ARCANA_METADATA, snapshot_root=ARCANA_SNAPSHOTS)
    coverage = store.annual_coverage()
    write_json(
        output / "input-manifest.json",
        {
            "schema_version": "moatrader-v7.2-inputs/1",
            "marcap_commit": MARCAP_COMMIT,
            "marcap_sources": marcap_sources,
            "arcana_dart_metadata": {
                "path": str(ARCANA_METADATA),
                "sha256": sha256_file(ARCANA_METADATA),
                "mode": "READ_ONLY_REUSE",
            },
            "annual_dart_coverage": coverage,
            "coverage_decision": (
                "2010-2017 NOT USED FOR PERFORMANCE: two-year broad annual DART history is unavailable; "
                "2018-2019 is the earliest broad pre-2020 testable interval"
            ),
            "return_inputs_after_2019_present": False,
        },
    )
    signals = build_pre_return_signals(marcap=marcap, store=store, output=output)
    signal_path = output / "results/signals-pre-return.csv"
    signals.to_csv(signal_path, index=False, encoding="utf-8-sig")
    write_json(
        output / "results/signals-seal.json",
        {
            "schema_version": "moatrader-v7.2-driver-gap-signal-seal/1",
            "signals_sha256": sha256_file(signal_path),
            "contract_sha256": sha256_file(output / "frozen-contract.json"),
            "driver_signal_code_sha256": sha256_file(
                REPOSITORY / "src/moatrader/expectations/driver_signals.py"
            ),
            "return_labels_opened_before_seal": False,
            "formula_changed_after_return_access": False,
            "llm_used_for_rank": False,
        },
    )
    _events, summary = evaluate_after_seal(signals=signals, marcap=marcap, output=output)
    final = {
        "schema_version": "moatrader-v7.2-final-result/1",
        "contract_schema_version": contract["schema_version"],
        "period": summary["period"],
        "signal_date_count": summary["signal_date_count"],
        "selected_observation_count": len(signals),
        "unique_ticker_count": int(signals["ticker"].nunique()),
        "base_status_counts": signals["status"].value_counts().to_dict(),
        "driver_solution_status_counts": {
            driver.value: signals[f"{driver.value.lower()}_gap_status"].fillna("NO_BASE_ESTIMATE").value_counts().to_dict()
            for driver in DriverName
        },
        "driver_judgments": summary["driver_judgments"],
        "overall_judgment": summary["overall_judgment"],
        "true_live_oos": False,
        "validation_grade": summary["validation_grade"],
        "llm_used_for_alpha": False,
        "data_limitations": [
            "Only 2018-2019 has broad pre-2020 two-year PIT DART coverage.",
            "Arcana canonical debt aggregation does not exactly reproduce all original XBRL debt facts.",
            "Eighteen monthly observations with overlapping 77-day returns have low effective time-series power.",
        ],
    }
    write_json(output / "FINAL-RESULT.json", final)
    assert_prior_versions_unchanged(before, output)
    write_json(
        output / "build-manifest.json",
        {
            "schema_version": "moatrader-v7.2-build/1",
            "artifacts": {
                path.relative_to(output).as_posix(): sha256_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "build-manifest.json"
            },
            "v6_v7_v7_1_unchanged": True,
            "credentials_persisted": False,
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
