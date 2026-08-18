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

from moatrader.backtest.historical import quarterly_signal_dates
from moatrader.backtest.universe_corrected import (
    build_historical_universe,
    moving_block_bootstrap_mean,
    newey_west_mean,
    previous_price_point,
    rank_normal_score,
    residualize_cross_section,
    sha256_file,
    spearman_ic,
    trailing_beta,
    trailing_momentum,
)
from moatrader.expectations.driver_signals import (
    WACC_BY_SIZE,
    DriverName,
    supported_driver_estimate,
)
from moatrader.expectations.revision import (
    DRIVER_SHOCKS,
    DYNAMIC_REVISION_GRIDS,
    EXPECTATION_SURFACE_LEVELS,
    SURFACE_KERNEL_BANDWIDTH,
    RevisionStatus,
    SurfaceStatus,
    driver_sensitivities,
    dynamic_implied_revision,
    expectation_surface_revision,
    periodic_value_factor_vector,
    turbo_driver,
)
from moatrader.financial.arcana_pit import ArcanaAnnualPitStore, ArcanaPeriodicPitStore
from moatrader.valuation.assumptions import EconomicDcfAssumptions


REPOSITORY = Path(__file__).resolve().parents[1]
OUTPUT_DEFAULT = (
    REPOSITORY
    / "data-lake/experiments/dynamic-expectation-revision-v7-3-2020-2025-diagnostic"
)
V7_2_ROOT = REPOSITORY / "data-lake/experiments/driver-expectation-gap-v7-2-2018-2019"
V7_1_ROOT = REPOSITORY / "data-lake/experiments/universe-corrected-residual-v7-1-2020-2025"
V7_PRICE_ROOT = REPOSITORY / "data-lake/experiments/historical-validation-v7-2020-2025/prices/source"
ARCANA_DART = Path(r"D:\Programming\python_example\Arcana\data-lake\silver\dart")
ARCANA_METADATA = ARCANA_DART / "kr_report_metadata.csv"
ARCANA_SNAPSHOTS = ARCANA_DART / "normalized-snapshots"
SECTOR_SOURCE = V7_1_ROOT / "inputs/krx-kind-current-company-list.xls"
MARCAP_COMMIT = "5e8e4e57f3fcb129a6ff20751f643f67d3592c82"
SIGNAL_START = date(2020, 3, 31)
SIGNAL_END = date(2025, 9, 30)
HAC_LAG = 1
BLOCK_LENGTH = 4


COMPONENT_DRIVER: dict[str, DriverName] = {
    "growth_yoy": DriverName.GROWTH,
    "growth_acceleration": DriverName.GROWTH,
    "nopat_margin_change": DriverName.MARGIN,
    "operating_leverage_spread": DriverName.MARGIN,
    "roiic_change": DriverName.ROIIC,
    "incremental_sales_efficiency_change": DriverName.ROIIC,
    "roic_spread_change": DriverName.CAP,
    "positive_roic_spread_persistence": DriverName.CAP,
}


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
    inherited = json.loads((V7_2_ROOT / "seal-before.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": "moatrader-v7.3-prior-version-seal/1",
        "v6": current_hashes(inherited["v6"]),
        "v7": current_hashes(inherited["v7"]),
        "v7_1": current_hashes(inherited["v7_1"]),
        "v7_2": file_map(V7_2_ROOT),
        "mutation_policy": "V6_V7_V7.1_V7.2_READ_ONLY; V7.3_OUTPUT_ONLY",
    }
    write_json(output / "seal-before.json", payload)
    return payload


def assert_prior_versions_unchanged(before: dict[str, Any], output: Path) -> None:
    current = {
        "v6": current_hashes(before["v6"]),
        "v7": current_hashes(before["v7"]),
        "v7_1": current_hashes(before["v7_1"]),
        "v7_2": file_map(V7_2_ROOT),
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
        "schema_version": "moatrader-v7.3-prior-version-integrity/1",
        **{f"{version}_unchanged": not paths for version, paths in changes.items()},
        "changed_paths": changes,
    }
    write_json(output / "integrity-after.json", payload)
    if any(changes.values()):
        raise RuntimeError("protected v6/v7/v7.1/v7.2 artifacts changed")


def signal_dates() -> list[date]:
    return quarterly_signal_dates(start=SIGNAL_START, end=SIGNAL_END)


def next_quarter_end(value: date) -> date:
    lookup = {
        3: (value.year, 6, 30),
        6: (value.year, 9, 30),
        9: (value.year, 12, 31),
        12: (value.year + 1, 3, 31),
    }
    if value.month not in lookup:
        raise ValueError("signal date must be a calendar quarter end")
    return date(*lookup[value.month])


def freeze_contract(output: Path) -> dict[str, Any]:
    payload = {
        "schema_version": "moatrader-v7.3-dynamic-expectation-revision-freeze/1",
        "frozen_on": "2026-08-19",
        "validation_grade": "2020_2025_DEVELOPMENT_DIAGNOSTIC_NOT_OOS",
        "hypothesis": (
            "Return-free DART value-factor changes observed at t predict the same-method, "
            "one-turbo-driver market-implied expectation revision at the next quarter end."
        ),
        "period": [SIGNAL_START.isoformat(), SIGNAL_END.isoformat()],
        "signal_dates": [value.isoformat() for value in signal_dates()],
        "target_dates": [next_quarter_end(value).isoformat() for value in signal_dates()],
        "sequence": [
            "PIT_VALUE_FACTOR_VECTOR",
            "DCF_SENSITIVITY_TURBO_TRIGGER",
            "PREDICTION_SEAL",
            "NEXT_QUARTER_IDENTIFIED_IMPLIED_DRIVER_REVISION",
            "MECHANISM_GATE",
            "CONDITIONAL_FACTOR_NEUTRAL_RETURN_TEST_ONLY_IF_GATE_PASSES",
        ],
        "identification_contract": {
            "primary_surface_sensor": (
                "ENTRY_AND_TARGET_USE_ONE_FROZEN_T_5X5X5X5_GROWTH_MARGIN_ROIIC_CAP_SURFACE; "
                "UNIFORM_EXPLICIT_STATE_PRIOR; LOG_PRICE_KERNEL_WEIGHTING; WACC_SCALE_NET_DEBT_"
                "SHARES_FIXED; SPARSE_OR_OUTSIDE_SURFACE_EXCLUDED"
            ),
            "secondary_one_driver_slice": (
                "OTHER_THREE_DRIVERS_FIXED; ENTRY_ROOT_NEAREST_SUPPORTED; TARGET_ROOT_NEAREST_"
                "ENTRY_ROOT; CENSORED_PAIRS_EXCLUDED"
            ),
        },
        "turbo_trigger": {
            "rule": "largest positive central DCF equity-price sensitivity",
            "normalized_shocks": {
                driver.value: str(shock) for driver, shock in DRIVER_SHOCKS.items()
            },
            "future_returns_used": False,
        },
        "value_factor_mapping": {
            "GROWTH": ["growth_yoy", "growth_acceleration"],
            "MARGIN": ["nopat_margin_change", "operating_leverage_spread"],
            "ROIIC": ["roiic_change", "incremental_sales_efficiency_change"],
            "CAP": ["roic_spread_change", "positive_roic_spread_persistence"],
        },
        "proxy_warning": (
            "Structured DART statements do not directly identify volume or price/mix. "
            "Growth and margin components are observable composite proxies, not direct labels."
        ),
        "component_policy": (
            "raw components and cross-sectional rank-normal components retained separately; "
            "within-trigger mean is used only for directional mechanism evaluation; no "
            "Sensitivity*Revision*Confidence aggregate alpha score"
        ),
        "dynamic_grids": {
            driver.value: [str(value) for value in values]
            for driver, values in DYNAMIC_REVISION_GRIDS.items()
        },
        "expectation_surface": {
            "role": "PRIMARY_MARKET_EXPECTATION_SENSOR",
            "levels": {
                driver.value: [str(value) for value in values]
                for driver, values in EXPECTATION_SURFACE_LEVELS.items()
            },
            "design_points": math.prod(
                len(values) for values in EXPECTATION_SURFACE_LEVELS.values()
            ),
            "uniform_state_prior": True,
            "log_price_kernel_bandwidth": SURFACE_KERNEL_BANDWIDTH,
            "maximum_nearest_relative_price_error": 0.10,
            "minimum_effective_point_count": 5,
        },
        "primary_mechanism_metrics": [
            "quarterly cross-sectional Spearman IC: turbo factor score vs revision in driver shocks",
            "direction accuracy vs actual implied revision sign",
            "Q4-minus-Q1 mean revision in driver shocks",
            "confidence calibration",
        ],
        "mechanism_gate": {
            "minimum_solved_pair_coverage": 0.25,
            "minimum_dates_with_ic": 12,
            "mean_ic_positive": True,
            "newey_west_ic_t_minimum": 1.96,
            "moving_block_ic_lower_bound_positive": True,
            "direction_accuracy_minimum": 0.55,
            "direction_accuracy_above_majority_baseline_by": 0.02,
            "mean_q4_minus_q1_revision_positive": True,
        },
        "conditional_return_controls": [
            "VALUE_BOOK_TO_MARKET",
            "QUALITY_ROA_CFO_LEVERAGE",
            "MOMENTUM_12_1",
            "LOG_SIZE",
            "BETA_252",
            "CURRENT_2026_SECTOR_SENSITIVITY_ONLY",
        ],
        "llm_used_for_primary_signal": False,
        "llm_rationale": (
            "The prior sealed LLM overlay classified moat/fragility axes, not the frozen "
            "value-factor taxonomy; v7.3 first isolates construct validity with deterministic "
            "DART financial evidence."
        ),
        "cheap_role": "NOT_ALPHA_SIGNAL; RESERVED_FOR_LATER_PAYOFF_OR_MARGIN_OF_SAFETY",
        "target_prices_permitted_before_prediction_seal": False,
    }
    write_json(output / "frozen-contract.json", payload)
    return payload


def read_marcap() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    columns = [
        "Code", "Name", "Close", "Amount", "Marcap", "Stocks", "Market", "MarketId",
        "Rank", "Date", "ChangesRatio", "Dept",
    ]
    pieces: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    for year in range(2019, 2026):
        path = (
            V7_1_ROOT / "inputs/marcap/marcap-2019.parquet"
            if year == 2019
            else V7_PRICE_ROOT / f"marcap-{year}.parquet"
        )
        if not path.exists():
            raise FileNotFoundError(path)
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
                "url": (
                    f"https://raw.githubusercontent.com/FinanceData/marcap/{MARCAP_COMMIT}/"
                    f"data/marcap-{year}.parquet"
                ),
                "reuse_mode": "SEALED_V7_READ_ONLY" if year >= 2020 else "SEALED_V7.1_READ_ONLY",
            }
        )
    return pd.concat(pieces, ignore_index=True), sources


def current_sector_map() -> tuple[dict[str, str], dict[str, Any]]:
    if not SECTOR_SOURCE.exists():
        return {}, {"available": False, "pit": False, "path": str(SECTOR_SOURCE)}
    table = pd.read_html(SECTOR_SOURCE, encoding="euc-kr")[0]
    ticker_column = next(column for column in table.columns if "종목코드" in str(column))
    sector_column = next(column for column in table.columns if "업종" in str(column))
    mapping = {
        str(ticker).strip().zfill(6): str(sector)
        for ticker, sector in zip(table[ticker_column], table[sector_column])
        if pd.notna(ticker) and pd.notna(sector)
    }
    return mapping, {
        "available": True,
        "pit": False,
        "allowed_use": "CONDITIONAL_RETURN_SECTOR_SENSITIVITY_ONLY",
        "path": str(SECTOR_SOURCE),
        "sha256": sha256_file(SECTOR_SOURCE),
        "ticker_count": len(mapping),
    }


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


def factor_values(
    latest: dict[str, float | int | None] | None,
    *,
    market_cap: float,
    prices: pd.DataFrame,
    market_return: pd.Series,
    signal_date: date,
) -> dict[str, float | None]:
    if latest is None:
        return {
            "value_btm": None,
            "quality_roa_cfo_leverage": None,
            "momentum_12_1": trailing_momentum(prices, as_of=signal_date),
            "size_log_mcap": math.log(market_cap) if market_cap > 0 else None,
            "beta_252": trailing_beta(prices, market_return, as_of=signal_date),
        }
    equity = float(latest.get("total_equity") or 0)
    assets = float(latest.get("total_assets") or 0)
    ebit = float(latest.get("ebit") or 0)
    cfo = float(latest.get("cfo") or 0)
    debt = float(latest.get("debt") or 0)
    return {
        "value_btm": equity / market_cap if equity > 0 and market_cap > 0 else None,
        "quality_roa_cfo_leverage": (ebit + cfo - debt) / assets if assets > 0 else None,
        "momentum_12_1": trailing_momentum(prices, as_of=signal_date),
        "size_log_mcap": math.log(market_cap) if market_cap > 0 else None,
        "beta_252": trailing_beta(prices, market_return, as_of=signal_date),
    }


def build_predictions_pre_target(
    *,
    marcap: pd.DataFrame,
    annual_store: ArcanaAnnualPitStore,
    periodic_store: ArcanaPeriodicPitStore,
    sectors: dict[str, str],
    output: Path,
) -> pd.DataFrame:
    grouped_prices = marcap.groupby("Code", sort=False)
    price_cache: dict[str, pd.DataFrame] = {}
    market_daily = (
        marcap[marcap["MarketId"].isin(["STK", "KSQ"])]
        .groupby(["MarketId", "Date"])["ChangesRatio"]
        .mean()
        .div(100.0)
    )
    rows: list[dict[str, Any]] = []
    universes: list[pd.DataFrame] = []
    for signal_date in signal_dates():
        window = marcap[
            (marcap["Date"] >= pd.Timestamp(signal_date) - pd.Timedelta(days=400))
            & (marcap["Date"] <= pd.Timestamp(signal_date))
        ]
        build = build_historical_universe(window, as_of=signal_date)
        selected = build.selected.copy()
        selected.insert(0, "signal_date", signal_date.isoformat())
        universes.append(selected)
        date_rows: list[dict[str, Any]] = []
        for _, item in build.selected.iterrows():
            ticker = str(item["stock_code"]).zfill(6)
            if ticker not in price_cache:
                price_cache[ticker] = grouped_prices.get_group(ticker).sort_values("Date")
            prices = price_cache[ticker]
            point = previous_price_point(prices, as_of=signal_date)
            annual_history, annual_sources = annual_store.history(ticker, signal_date)
            periodic, periodic_sources = periodic_store.latest_with_comparables(
                ticker, signal_date, prior_years=2
            )
            latest = annual_history[-1][1] if annual_history else None
            market_id = "STK" if item["market"] == "KOSPI" else "KSQ"
            row: dict[str, Any] = {
                "signal_date": signal_date.isoformat(),
                "target_date": next_quarter_end(signal_date).isoformat(),
                "universe_actual_as_of": build.actual_as_of.isoformat(),
                "ticker": ticker,
                "name": item["name"],
                "market": item["market"],
                "size_bucket": str(item["size_bucket"]),
                "current_sector": sectors.get(ticker, "UNKNOWN_CURRENT_SECTOR"),
                "current_sector_pit": False,
                "market_cap": float(item["market_cap"]),
                "listed_shares": int(item["listed_shares"]),
                "finance_hint": bool(item["finance_hint"]),
                "holding_hint": bool(item["holding_hint"]),
                "entry_price_date": (
                    pd.Timestamp(point["Date"]).date().isoformat() if point is not None else ""
                ),
                "entry_price": float(point["Close"]) if point is not None else None,
                "latest_annual_fiscal_year": annual_history[-1][0] if annual_history else None,
                "latest_annual_report_date": annual_sources[-1]["report_date"] if annual_sources else "",
                "latest_annual_rcept_no": annual_sources[-1]["rcept_no"] if annual_sources else "",
                "periodic_fiscal_year": periodic[0][0] if periodic else None,
                "periodic_fiscal_month": periodic[0][1] if periodic else None,
                "periodic_report_date": periodic_sources[0]["report_date"] if periodic_sources else "",
                "periodic_rcept_no": periodic_sources[0]["rcept_no"] if periodic_sources else "",
                "periodic_comparable_count": len(periodic),
                "status": "",
                "status_detail": "",
                "base_assumptions_json": "",
                "turbo_driver": "",
                **factor_values(
                    latest,
                    market_cap=float(item["market_cap"]),
                    prices=prices,
                    market_return=market_daily.loc[market_id],
                    signal_date=signal_date,
                ),
            }
            if item["finance_hint"] or item["holding_hint"]:
                row["status"] = "EXCLUDED_ARCHETYPE"
            elif point is None:
                row["status"] = "NO_ENTRY_PRICE"
            elif len(periodic) < 3:
                row["status"] = "NO_THREE_PERIODIC_COMPARABLES"
            else:
                try:
                    estimate = supported_driver_estimate(
                        decimal_history(annual_history),
                        size_bucket=str(item["size_bucket"]),
                        diluted_shares=Decimal(str(int(item["listed_shares"]))),
                    )
                    base = estimate.assumptions()
                    vector = periodic_value_factor_vector(
                        periodic,
                        wacc=WACC_BY_SIZE[str(item["size_bucket"]).upper()],
                    )
                    sensitivities = driver_sensitivities(base)
                    selected_driver = turbo_driver(sensitivities)
                    row.update(
                        {
                            **{
                                name: float(value) if value is not None else None
                                for name, value in vector.model_dump().items()
                                if name not in {"fiscal_year", "fiscal_month"}
                            },
                            "supported_growth": float(estimate.growth),
                            "supported_margin": float(estimate.margin),
                            "supported_roiic": float(estimate.roiic),
                            "supported_cap": estimate.cap_years,
                            "base_assumptions_json": base.model_dump_json(),
                        }
                    )
                    for driver, sensitivity in sensitivities.items():
                        prefix = driver.value.lower()
                        row[f"{prefix}_sensitivity_signed"] = (
                            float(sensitivity.signed_price_change_per_shock)
                            if sensitivity.signed_price_change_per_shock is not None
                            else None
                        )
                        row[f"{prefix}_sensitivity_abs"] = (
                            float(sensitivity.absolute_price_change_per_shock)
                            if sensitivity.absolute_price_change_per_shock is not None
                            else None
                        )
                        row[f"{prefix}_sensitivity_eligible"] = sensitivity.eligible
                    if selected_driver is None:
                        row["status"] = "NO_POSITIVE_TURBO_SENSITIVITY"
                    else:
                        row["turbo_driver"] = selected_driver.value
                        row["status"] = "PREDICTION_CANDIDATE"
                except Exception as exc:
                    row["status"] = "UNTESTABLE_FINANCIAL_OR_MODEL"
                    row["status_detail"] = f"{type(exc).__name__}: {exc}"
            date_rows.append(row)
        rows.extend(date_rows)
        print(
            f"formation {signal_date}: {dict(Counter(row['status'] for row in date_rows))}",
            flush=True,
        )

    signals = pd.DataFrame(rows)
    for signal_date, indices in signals.groupby("signal_date").groups.items():
        group = signals.loc[indices]
        candidate = group["status"].eq("PREDICTION_CANDIDATE")
        for component in COMPONENT_DRIVER:
            values = group[component] if component in group else pd.Series(np.nan, index=group.index)
            values = values.where(candidate)
            signals.loc[indices, f"{component}_rn"] = rank_normal_score(values)
        for driver in DriverName:
            components = [
                name for name, mapped_driver in COMPONENT_DRIVER.items() if mapped_driver == driver
            ]
            normalized = [f"{name}_rn" for name in components]
            part = signals.loc[indices, normalized]
            count = part.notna().sum(axis=1)
            score = part.mean(axis=1, skipna=True).where(count > 0)
            same_direction = (
                np.sign(part.iloc[:, 0]) == np.sign(part.iloc[:, 1])
            ) & part.notna().all(axis=1)
            consistency = pd.Series(0.5, index=part.index)
            consistency.loc[same_direction] = 1.0
            signals.loc[indices, f"{driver.value.lower()}_factor_component_count"] = count
            signals.loc[indices, f"{driver.value.lower()}_factor_score"] = score
            signals.loc[indices, f"{driver.value.lower()}_factor_confidence"] = (
                count / len(components) * consistency
            ).where(count > 0)

    for index, row in signals[signals["status"].eq("PREDICTION_CANDIDATE")].iterrows():
        prefix = str(row["turbo_driver"]).lower()
        score = pd.to_numeric(pd.Series([row.get(f"{prefix}_factor_score")]), errors="coerce").iloc[0]
        confidence = pd.to_numeric(
            pd.Series([row.get(f"{prefix}_factor_confidence")]), errors="coerce"
        ).iloc[0]
        if pd.isna(score):
            signals.at[index, "status"] = "NO_TURBO_FACTOR_SCORE"
        elif float(score) == 0:
            signals.at[index, "status"] = "NO_PREDICTED_DIRECTION"
        else:
            signals.at[index, "turbo_factor_score"] = float(score)
            signals.at[index, "turbo_factor_confidence"] = float(confidence)
            signals.at[index, "predicted_revision_sign"] = 1 if float(score) > 0 else -1
            signals.at[index, "status"] = "PREDICTION_READY"

    selected = pd.concat(universes, ignore_index=True)
    selected.to_csv(
        output / "universes/selected-150-by-date.csv", index=False, encoding="utf-8-sig"
    )
    return signals


def _quartile_spread(sample: pd.DataFrame) -> tuple[float, list[float]]:
    if len(sample) < 12:
        return float("nan"), [float("nan")] * 4
    work = sample.copy()
    work["quartile"] = pd.qcut(
        work["turbo_factor_score"].rank(method="first"),
        4,
        labels=[1, 2, 3, 4],
    ).astype(int)
    means = work.groupby("quartile")["surface_revision_in_shocks"].mean()
    profile = [float(means.get(value, np.nan)) for value in range(1, 5)]
    return profile[-1] - profile[0], profile


def evaluate_mechanism_after_seal(
    *,
    predictions: pd.DataFrame,
    marcap: pd.DataFrame,
    output: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    result = predictions.copy()
    for column, default in (
        ("target_price_date", ""),
        ("target_price", np.nan),
        ("price_return", np.nan),
        ("revision_status", "NOT_EVALUATED"),
        ("entry_implied_driver", np.nan),
        ("target_implied_driver", np.nan),
        ("implied_driver_revision", np.nan),
        ("revision_in_shocks", np.nan),
        ("surface_status", "NOT_EVALUATED"),
        ("surface_revision_in_shocks", np.nan),
    ):
        result[column] = default
    grouped = marcap.groupby("Code", sort=False)
    cache: dict[str, pd.DataFrame] = {}
    ready = result["status"].eq("PREDICTION_READY")
    for position, (index, row) in enumerate(result[ready].iterrows(), start=1):
        ticker = str(row["ticker"]).zfill(6)
        if ticker not in cache:
            cache[ticker] = grouped.get_group(ticker).sort_values("Date")
        point = previous_price_point(
            cache[ticker], as_of=date.fromisoformat(str(row["target_date"]))
        )
        if point is None:
            result.at[index, "revision_status"] = "NO_TARGET_PRICE"
            continue
        target_price = Decimal(str(point["Close"]))
        entry_price = Decimal(str(row["entry_price"]))
        base = EconomicDcfAssumptions.model_validate_json(str(row["base_assumptions_json"]))
        driver = DriverName(str(row["turbo_driver"]))
        revision = dynamic_implied_revision(
            base=base,
            driver=driver,
            entry_price=entry_price,
            target_price=target_price,
        )
        result.at[index, "target_price_date"] = pd.Timestamp(point["Date"]).date().isoformat()
        result.at[index, "target_price"] = float(target_price)
        result.at[index, "price_return"] = float(target_price / entry_price - Decimal(1))
        result.at[index, "revision_status"] = revision.status.value
        result.at[index, "entry_implied_driver"] = (
            float(revision.entry_implied) if revision.entry_implied is not None else np.nan
        )
        result.at[index, "target_implied_driver"] = (
            float(revision.target_implied) if revision.target_implied is not None else np.nan
        )
        result.at[index, "implied_driver_revision"] = (
            float(revision.implied_revision) if revision.implied_revision is not None else np.nan
        )
        if revision.status == RevisionStatus.SOLVED and revision.implied_revision is not None:
            result.at[index, "revision_in_shocks"] = float(
                revision.implied_revision / DRIVER_SHOCKS[driver]
            )
        surface = expectation_surface_revision(
            base=base,
            entry_price=entry_price,
            target_price=target_price,
        )
        result.at[index, "surface_status"] = surface.status.value
        if surface.entry is not None:
            result.at[index, "surface_entry_nearest_price_error"] = float(
                surface.entry.nearest_relative_price_error
            )
            result.at[index, "surface_entry_effective_points"] = float(
                surface.entry.effective_point_count
            )
            for surface_driver in DriverName:
                prefix = surface_driver.value.lower()
                result.at[index, f"surface_entry_{prefix}_mean"] = float(
                    surface.entry.driver_mean[surface_driver]
                )
                result.at[index, f"surface_entry_{prefix}_p10"] = float(
                    surface.entry.driver_p10[surface_driver]
                )
                result.at[index, f"surface_entry_{prefix}_p90"] = float(
                    surface.entry.driver_p90[surface_driver]
                )
        if surface.target is not None:
            result.at[index, "surface_target_nearest_price_error"] = float(
                surface.target.nearest_relative_price_error
            )
            result.at[index, "surface_target_effective_points"] = float(
                surface.target.effective_point_count
            )
            for surface_driver in DriverName:
                prefix = surface_driver.value.lower()
                result.at[index, f"surface_target_{prefix}_mean"] = float(
                    surface.target.driver_mean[surface_driver]
                )
                result.at[index, f"surface_target_{prefix}_p10"] = float(
                    surface.target.driver_p10[surface_driver]
                )
                result.at[index, f"surface_target_{prefix}_p90"] = float(
                    surface.target.driver_p90[surface_driver]
                )
        if surface.status == SurfaceStatus.SOLVED:
            for surface_driver in DriverName:
                prefix = surface_driver.value.lower()
                driver_revision = surface.driver_revision[surface_driver]
                result.at[index, f"surface_{prefix}_revision"] = float(driver_revision)
                result.at[index, f"surface_{prefix}_revision_in_shocks"] = float(
                    driver_revision / DRIVER_SHOCKS[surface_driver]
                )
            turbo_revision = surface.driver_revision[driver]
            result.at[index, "surface_revision_in_shocks"] = float(
                turbo_revision / DRIVER_SHOCKS[driver]
            )
        if position % 250 == 0:
            print(f"target evaluation {position}/{int(ready.sum())}", flush=True)

    event_rows: list[dict[str, Any]] = []
    for signal_date, group in result.groupby("signal_date", sort=True):
        solved = group[group["surface_status"].eq(SurfaceStatus.SOLVED.value)].copy()
        direction = solved[
            solved["predicted_revision_sign"].notna()
            & solved["surface_revision_in_shocks"].notna()
            & solved["surface_revision_in_shocks"].ne(0)
        ].copy()
        direction["direction_correct"] = (
            direction["predicted_revision_sign"] * direction["surface_revision_in_shocks"] > 0
        ).astype(float)
        spread, profile = _quartile_spread(solved)
        event: dict[str, Any] = {
            "signal_date": signal_date,
            "prediction_ready_count": int(group["status"].eq("PREDICTION_READY").sum()),
            "solved_pair_count": len(solved),
            "mechanism_ic": spearman_ic(
                solved, "turbo_factor_score", "surface_revision_in_shocks"
            ),
            "direction_count": len(direction),
            "direction_accuracy": direction["direction_correct"].mean(),
            "q4_minus_q1_revision_in_shocks": spread,
            **{f"revision_q{number}_mean": profile[number - 1] for number in range(1, 5)},
        }
        for driver in DriverName:
            subset = solved[solved["turbo_driver"].eq(driver.value)]
            directed = direction[direction["turbo_driver"].eq(driver.value)]
            prefix = driver.value.lower()
            event[f"{prefix}_count"] = len(subset)
            event[f"{prefix}_ic"] = spearman_ic(
                subset, "turbo_factor_score", "surface_revision_in_shocks"
            )
            event[f"{prefix}_direction_accuracy"] = directed["direction_correct"].mean()
        for component, driver in COMPONENT_DRIVER.items():
            subset = solved[solved["turbo_driver"].eq(driver.value)]
            event[f"{component}_ic"] = spearman_ic(
                subset, f"{component}_rn", "surface_revision_in_shocks"
            )
        slice_comparable = solved[
            solved["revision_status"].eq(RevisionStatus.SOLVED.value)
            & solved["revision_in_shocks"].notna()
            & solved["revision_in_shocks"].ne(0)
            & solved["surface_revision_in_shocks"].ne(0)
        ]
        event["surface_slice_sign_agreement"] = (
            np.sign(slice_comparable["revision_in_shocks"])
            == np.sign(slice_comparable["surface_revision_in_shocks"])
        ).mean()
        event["surface_slice_comparable_count"] = len(slice_comparable)
        for driver in DriverName:
            prefix = driver.value.lower()
            event[f"{prefix}_all_vector_surface_ic"] = spearman_ic(
                solved,
                f"{prefix}_factor_score",
                f"surface_{prefix}_revision_in_shocks",
            )
        event_rows.append(event)
    events = pd.DataFrame(event_rows)

    solved_all = result[result["surface_status"].eq(SurfaceStatus.SOLVED.value)].copy()
    directed_all = solved_all[
        solved_all["predicted_revision_sign"].notna()
        & solved_all["surface_revision_in_shocks"].notna()
        & solved_all["surface_revision_in_shocks"].ne(0)
    ].copy()
    directed_all["direction_correct"] = (
        directed_all["predicted_revision_sign"] * directed_all["surface_revision_in_shocks"] > 0
    ).astype(float)
    positive_rate = float((directed_all["surface_revision_in_shocks"] > 0).mean()) if len(directed_all) else float("nan")
    majority = max(positive_rate, 1 - positive_rate) if math.isfinite(positive_rate) else float("nan")
    accuracy = float(directed_all["direction_correct"].mean()) if len(directed_all) else float("nan")
    ic_values = pd.to_numeric(events["mechanism_ic"], errors="coerce").dropna().tolist()
    spread_values = pd.to_numeric(
        events["q4_minus_q1_revision_in_shocks"], errors="coerce"
    ).dropna().tolist()
    ic_nw = newey_west_mean(ic_values, lag=HAC_LAG)
    ic_boot = moving_block_bootstrap_mean(
        ic_values, block_length=BLOCK_LENGTH, repetitions=10_000, seed=42
    )
    spread_nw = newey_west_mean(spread_values, lag=HAC_LAG)
    prediction_count = int(result["status"].eq("PREDICTION_READY").sum())
    coverage = len(solved_all) / prediction_count if prediction_count else 0.0
    gate_checks = {
        "solved_pair_coverage_ge_0_25": coverage >= 0.25,
        "dates_with_ic_ge_12": len(ic_values) >= 12,
        "mean_ic_positive": float(ic_nw["mean"]) > 0,
        "newey_west_ic_t_ge_1_96": float(ic_nw["t"]) >= 1.96,
        "block_bootstrap_ic_lower_bound_positive": float(ic_boot["ci_low"]) > 0,
        "direction_accuracy_ge_0_55": math.isfinite(accuracy) and accuracy >= 0.55,
        "accuracy_beats_majority_by_0_02": (
            math.isfinite(accuracy) and math.isfinite(majority) and accuracy >= majority + 0.02
        ),
        "mean_q4_minus_q1_revision_positive": float(spread_nw["mean"]) > 0,
    }
    gate_passed = all(gate_checks.values())

    confidence_calibration: list[dict[str, Any]] = []
    if len(directed_all):
        bins = pd.cut(
            directed_all["turbo_factor_confidence"],
            bins=[-np.inf, 0.49, 0.74, np.inf],
            labels=["LOW", "MEDIUM", "HIGH"],
        )
        for label, subset in directed_all.groupby(bins, observed=True):
            confidence_calibration.append(
                {
                    "confidence_bin": str(label),
                    "count": len(subset),
                    "mean_confidence": float(subset["turbo_factor_confidence"].mean()),
                    "direction_accuracy": float(subset["direction_correct"].mean()),
                }
            )

    summary = {
        "schema_version": "moatrader-v7.3-dynamic-revision-mechanism/1",
        "validation_grade": "2020_2025_DEVELOPMENT_DIAGNOSTIC_NOT_OOS",
        "signal_date_count": len(events),
        "prediction_ready_count": prediction_count,
        "surface_status_counts": result["surface_status"].value_counts().to_dict(),
        "one_driver_slice_status_counts": result["revision_status"].value_counts().to_dict(),
        "solved_pair_count": len(solved_all),
        "solved_pair_coverage": coverage,
        "turbo_driver_counts_ready": (
            result[result["status"].eq("PREDICTION_READY")]["turbo_driver"].value_counts().to_dict()
        ),
        "turbo_driver_counts_solved": solved_all["turbo_driver"].value_counts().to_dict(),
        "surface_slice_sign_agreement": {
            "comparable_count": int(
                pd.to_numeric(events["surface_slice_comparable_count"], errors="coerce").sum()
            ),
            "mean_quarterly_agreement": float(
                pd.to_numeric(events["surface_slice_sign_agreement"], errors="coerce").mean()
            ),
        },
        "mechanism_ic": {
            "newey_west": ic_nw,
            "moving_block_bootstrap": ic_boot,
        },
        "direction": {
            "count": len(directed_all),
            "accuracy": accuracy,
            "actual_positive_rate": positive_rate,
            "majority_baseline_accuracy": majority,
        },
        "q4_minus_q1_revision_in_shocks": {
            "newey_west": spread_nw,
            "moving_block_bootstrap": moving_block_bootstrap_mean(
                spread_values, block_length=BLOCK_LENGTH, repetitions=10_000, seed=42
            ),
        },
        "confidence_calibration": confidence_calibration,
        "driver_diagnostics": {},
        "component_diagnostics": {},
        "mechanism_gate_checks": gate_checks,
        "mechanism_gate_passed": gate_passed,
        "judgment": (
            "MECHANISM_SURVIVED_DEVELOPMENT_DIAGNOSTIC"
            if gate_passed
            else "MECHANISM_NOT_ESTABLISHED; DO_NOT_PROMOTE_OR_BUILD_COMPOSITE_ALPHA"
        ),
        "identification_limitation": (
            "The primary sensor retains a kernel-weighted multidimensional region, but its fixed "
            "uniform state grid is still an identifying prior. It is not a uniquely observed "
            "market belief. The one-driver curve is reported only as a slice diagnostic."
        ),
    }
    for driver in DriverName:
        prefix = driver.value.lower()
        values = pd.to_numeric(events[f"{prefix}_ic"], errors="coerce").dropna().tolist()
        subset = directed_all[directed_all["turbo_driver"].eq(driver.value)]
        summary["driver_diagnostics"][driver.value] = {
            "solved_count": int((solved_all["turbo_driver"] == driver.value).sum()),
            "direction_count": len(subset),
            "direction_accuracy": (
                float(subset["direction_correct"].mean()) if len(subset) else None
            ),
            "ic_newey_west": newey_west_mean(values, lag=HAC_LAG),
            "all_vector_to_surface_ic_newey_west": newey_west_mean(
                pd.to_numeric(
                    events[f"{prefix}_all_vector_surface_ic"], errors="coerce"
                ).dropna().tolist(),
                lag=HAC_LAG,
            ),
        }
    for component in COMPONENT_DRIVER:
        values = pd.to_numeric(events[f"{component}_ic"], errors="coerce").dropna().tolist()
        summary["component_diagnostics"][component] = {
            "driver": COMPONENT_DRIVER[component].value,
            "ic_newey_west": newey_west_mean(values, lag=HAC_LAG),
        }

    result.to_csv(
        output / "results/predictions-with-implied-revisions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    events.to_csv(
        output / "results/quarterly-mechanism-results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(output / "results/mechanism-summary.json", summary)
    return result, events, summary


def conditional_return_test(
    result: pd.DataFrame,
    mechanism: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    if not mechanism["mechanism_gate_passed"]:
        payload = {
            "schema_version": "moatrader-v7.3-conditional-return/1",
            "status": "NOT_RUN_MECHANISM_GATE_FAILED",
            "reason": (
                "The pre-registered Evidence→Implied-Revision mechanism gate failed; opening a "
                "factor-neutral alpha interpretation would violate the research sequence."
            ),
        }
        write_json(output / "results/conditional-return-stage.json", payload)
        return payload

    evaluated = result[result["surface_status"].eq(SurfaceStatus.SOLVED.value)].copy()
    events: list[dict[str, Any]] = []
    evaluated["price_return_full_resid"] = np.nan
    for signal_date, indices in evaluated.groupby("signal_date").groups.items():
        group = evaluated.loc[indices]
        evaluated.loc[indices, "price_return_full_resid"] = residualize_cross_section(
            group,
            target="price_return",
            numeric_controls=[
                "value_btm",
                "quality_roa_cfo_leverage",
                "momentum_12_1",
                "size_log_mcap",
                "beta_252",
            ],
            categorical_controls=["current_sector"],
        )
        events.append(
            {
                "signal_date": signal_date,
                "count": len(group),
                "raw_return_ic": spearman_ic(
                    group, "turbo_factor_score", "price_return"
                ),
                "full_residual_return_ic": spearman_ic(
                    evaluated.loc[indices],
                    "turbo_factor_score",
                    "price_return_full_resid",
                ),
            }
        )
    event_frame = pd.DataFrame(events)
    raw = pd.to_numeric(event_frame["raw_return_ic"], errors="coerce").dropna().tolist()
    residual = pd.to_numeric(
        event_frame["full_residual_return_ic"], errors="coerce"
    ).dropna().tolist()
    payload = {
        "schema_version": "moatrader-v7.3-conditional-return/1",
        "status": "RUN_AFTER_MECHANISM_GATE_PASS",
        "validation_grade": "DEVELOPMENT_DIAGNOSTIC_NOT_OOS_NOT_PROMOTABLE",
        "raw_return_ic": {
            "newey_west": newey_west_mean(raw, lag=HAC_LAG),
            "moving_block_bootstrap": moving_block_bootstrap_mean(
                raw, block_length=BLOCK_LENGTH, repetitions=10_000, seed=42
            ),
        },
        "full_residual_return_ic": {
            "newey_west": newey_west_mean(residual, lag=HAC_LAG),
            "moving_block_bootstrap": moving_block_bootstrap_mean(
                residual, block_length=BLOCK_LENGTH, repetitions=10_000, seed=42
            ),
        },
        "sector_warning": "CURRENT_2026_SECTOR_CLASSIFICATION; SENSITIVITY_ONLY_NOT_PIT",
        "promotion_allowed": False,
    }
    evaluated.to_csv(
        output / "results/conditional-return-observations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    event_frame.to_csv(
        output / "results/quarterly-conditional-return-results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_json(output / "results/conditional-return-stage.json", payload)
    return payload


def write_final_report(
    output: Path,
    mechanism: dict[str, Any],
    conditional: dict[str, Any],
    status_counts: dict[str, int],
) -> None:
    ic = mechanism["mechanism_ic"]["newey_west"]
    bootstrap = mechanism["mechanism_ic"]["moving_block_bootstrap"]
    direction = mechanism["direction"]
    spread = mechanism["q4_minus_q1_revision_in_shocks"]["newey_west"]
    checks = "\n".join(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in mechanism["mechanism_gate_checks"].items()
    )
    driver_rows = "\n".join(
        "| {driver} | {count} | {accuracy} | {ic_mean} |".format(
            driver=driver,
            count=values["solved_count"],
            accuracy=(
                f"{values['direction_accuracy']:.3f}"
                if values["direction_accuracy"] is not None
                else "N/A"
            ),
            ic_mean=(
                f"{values['ic_newey_west']['mean']:.4f}"
                if math.isfinite(float(values["ic_newey_west"]["mean"]))
                else "N/A"
            ),
        )
        for driver, values in mechanism["driver_diagnostics"].items()
    )
    report = f"""# MoatRader v7.3 Dynamic Expectation Revision — Final Diagnostic

## 판정

**{mechanism['judgment']}**

이 결과는 2020–2025 데이터를 본 뒤 설계된 공식으로 같은 기간을 진단한 것이므로
`development diagnostic`이며 OOS 또는 pseudo-OOS로 부르지 않는다. Cheap은 이번 알파
신호에서 제외했고 향후 payoff/margin-of-safety 층에만 남겼다.

## 1차 메커니즘 결과

- Prediction-ready: {mechanism['prediction_ready_count']:,}
- 양 끝 가격 모두 surface eligible: {mechanism['solved_pair_count']:,} ({mechanism['solved_pair_coverage']:.1%})
- 분기 평균 Spearman IC: {ic['mean']:.4f} (NW t={ic['t']:.2f})
- IC moving-block 95% CI: [{bootstrap['ci_low']:.4f}, {bootstrap['ci_high']:.4f}]
- 방향 정확도: {direction['accuracy']:.1%} (majority baseline {direction['majority_baseline_accuracy']:.1%})
- Q4-Q1 implied revision (driver-shock units): {spread['mean']:.4f}
- Surface/one-driver slice 부호 일치율(분기 평균): {mechanism['surface_slice_sign_agreement']['mean_quarterly_agreement']:.1%}

### Gate

{checks}

| Turbo driver | Solved N | Direction accuracy | Mean within-driver IC |
|---|---:|---:|---:|
{driver_rows}

## 연구 순서

조건부 수익률 단계 상태: **{conditional['status']}**

메커니즘 게이트를 통과하지 못하면 수익률 단계는 실행하지 않는다는 계약을 그대로
적용했다. 통과하더라도 이 기간의 수익 결과는 개발 진단일 뿐 승격 근거가 될 수 없다.

## 구현 범위와 한계

- `t`와 `t+1`은 동일한 t-시점 625개 상태의 Growth·Margin·ROIIC·CAP surface와 동일한
  WACC, scale, net debt, shares를 사용한다. 가격 근접 조합 분포를 같은 방식으로 비교한다.
- 주가 하나가 네 Driver를 유일하게 식별하지는 못한다. 균등 상태 prior와 log-price
  kernel을 명시적으로 둔 identified distribution이며, 관찰된 시장 신념 그 자체는 아니다.
- 나머지 세 Driver를 고정하는 one-driver 역산은 주 판정이 아니라 slice 진단으로만
  보존했다. Surface와 slice의 부호 일치율도 함께 보고해 단일축 의존성을 드러냈다.
- 구조화 DART 재무는 Volume과 Price/Mix를 직접 분해하지 못한다. 성장 가속, 마진 변화,
  영업레버리지, 증분자본효율, ROIC 지속성을 명시적 composite proxy로 사용했다.
- prior LLM corpus의 축은 moat/fragility였고 새 Value Factor taxonomy가 아니어서, 기존
  LLM 점수를 억지로 재사용하지 않았다. 텍스트 LLM 센서는 별도 construct-validity
  gold set을 만든 뒤 이 벡터의 추가 component로 검증해야 한다.

## Formation status

```json
{json.dumps(status_counts, ensure_ascii=False, indent=2)}
```
"""
    (output / "FINAL-REPORT.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run v7.3 Value-Factor to dynamic implied-revision diagnostic."
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    output = args.output.resolve()
    normalized = output.as_posix().casefold()
    if "v7-3" not in normalized and "v7.3" not in normalized:
        raise ValueError("output must be an explicit v7.3 path")
    if (output / "FINAL-RESULT.json").exists():
        raise FileExistsError(f"completed v7.3 result is immutable: {output}")
    for directory in ("results", "universes", "audits"):
        (output / directory).mkdir(parents=True, exist_ok=True)

    prior = protect_prior_versions(output)
    freeze_contract(output)
    marcap, marcap_sources = read_marcap()
    sectors, sector_manifest = current_sector_map()
    write_json(
        output / "input-manifest.json",
        {
            "schema_version": "moatrader-v7.3-inputs/1",
            "marcap_provider_commit": MARCAP_COMMIT,
            "marcap_sources": marcap_sources,
            "arcana_dart": {
                "metadata_path": str(ARCANA_METADATA),
                "metadata_sha256": sha256_file(ARCANA_METADATA),
                "snapshot_root": str(ARCANA_SNAPSHOTS),
                "reuse_mode": "READ_ONLY_EXISTING_ARCANA_DART_CACHE",
            },
            "sector_source": sector_manifest,
            "credentials_persisted": False,
        },
    )
    annual_store = ArcanaAnnualPitStore(
        metadata_path=ARCANA_METADATA,
        snapshot_root=ARCANA_SNAPSHOTS,
    )
    periodic_store = ArcanaPeriodicPitStore(
        metadata_path=ARCANA_METADATA,
        snapshot_root=ARCANA_SNAPSHOTS,
    )
    write_json(
        output / "audits/arcana-periodic-coverage.json",
        {
            "schema_version": "moatrader-v7.3-arcana-periodic-coverage/1",
            "coverage": [
                row
                for row in periodic_store.periodic_coverage()
                if 2018 <= row["fiscal_year"] <= 2025
            ],
        },
    )
    predictions = build_predictions_pre_target(
        marcap=marcap,
        annual_store=annual_store,
        periodic_store=periodic_store,
        sectors=sectors,
        output=output,
    )
    prediction_path = output / "results/predictions-pre-target.csv"
    predictions.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    write_json(
        output / "results/prediction-seal.json",
        {
            "schema_version": "moatrader-v7.3-prediction-seal/1",
            "prediction_sha256": sha256_file(prediction_path),
            "sealed_signal": "TURBO_DRIVER_PLUS_RETURN_FREE_DART_VALUE_FACTOR_VECTOR",
            "target_prices_opened_before_seal": False,
            "future_returns_used_to_select_turbo": False,
            "future_returns_used_to_define_factor_mapping": False,
            "aggregate_sensitivity_revision_confidence_score_created": False,
        },
    )
    result, _events, mechanism = evaluate_mechanism_after_seal(
        predictions=predictions,
        marcap=marcap,
        output=output,
    )
    conditional = conditional_return_test(result, mechanism, output)
    status_counts = predictions["status"].value_counts().to_dict()
    final = {
        "schema_version": "moatrader-v7.3-final-diagnostic/1",
        "validation_grade": "2020_2025_DEVELOPMENT_DIAGNOSTIC_NOT_OOS",
        "period": [SIGNAL_START.isoformat(), SIGNAL_END.isoformat()],
        "signal_date_count": len(signal_dates()),
        "selected_observation_count": len(predictions),
        "formation_status_counts": status_counts,
        "prediction_ready_count": mechanism["prediction_ready_count"],
        "solved_pair_count": mechanism["solved_pair_count"],
        "solved_pair_coverage": mechanism["solved_pair_coverage"],
        "mechanism_gate_passed": mechanism["mechanism_gate_passed"],
        "mechanism_judgment": mechanism["judgment"],
        "conditional_return_stage": conditional["status"],
        "overall_judgment": (
            "DEVELOPMENT_MECHANISM_SURVIVED_BUT_REQUIRES_FRESH_HOLDOUT"
            if mechanism["mechanism_gate_passed"]
            else "DYNAMIC_EXPECTATION_REVISION_MECHANISM_NOT_ESTABLISHED"
        ),
        "llm_used_for_primary_signal": False,
        "cheap_used_for_primary_signal": False,
        "true_live_oos": False,
        "pseudo_oos": False,
        "credentials_persisted": False,
    }
    write_json(output / "FINAL-RESULT.json", final)
    write_final_report(output, mechanism, conditional, status_counts)
    assert_prior_versions_unchanged(prior, output)
    write_json(
        output / "build-manifest.json",
        {
            "schema_version": "moatrader-v7.3-build/1",
            "artifacts": {
                path.relative_to(output).as_posix(): sha256_file(path)
                for path in sorted(output.rglob("*"))
                if path.is_file() and path.name != "build-manifest.json"
            },
            "prior_versions_unchanged": True,
            "credentials_persisted": False,
        },
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
