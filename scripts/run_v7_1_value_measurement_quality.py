from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from moatrader.backtest.universe_corrected import (
    compound_percent,
    moving_block_bootstrap_mean,
    newey_west_mean,
    rank_normal_score,
    residualize_cross_section,
    sha256_file,
    spearman_ic,
)
from scripts.run_v7_1_value_neutral_sensitivity import extract_value_fundamentals


REPOSITORY = Path(__file__).resolve().parents[1]
NEUTRAL_ROOT = (
    REPOSITORY
    / "data-lake/experiments/expectation-gap-v7-1-multi-value-neutral-sensitivity-2020-2025"
)
SIGNALS_DEFAULT = NEUTRAL_ROOT / "results/value-enriched-signals.csv"
BASE_ROOT = REPOSITORY / "data-lake/experiments/universe-corrected-residual-v7-1-2020-2025"
PRICE_ROOT = REPOSITORY / "data-lake/experiments/historical-validation-v7-2020-2025/prices/source"
ARCANA_ROOT = Path(r"D:\Programming\python_example\Arcana\data-lake\silver\dart")
ARCANA_METADATA = ARCANA_ROOT / "kr_report_metadata.csv"
ARCANA_SNAPSHOTS = ARCANA_ROOT / "normalized-snapshots"
OUTPUT_DEFAULT = (
    REPOSITORY / "data-lake/experiments/value-measurement-quality-v7-1-2020-2025"
)
HORIZONS = (77, 182, 365, 730)
HORIZON_LAGS = {77: 1, 182: 2, 365: 4, 730: 8}
BLOCK_LENGTHS = {77: 4, 182: 4, 365: 4, 730: 8}
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 42
UNKNOWN_SECTOR = "UNKNOWN_CURRENT_SECTOR"
MIN_SECTOR_SIZE = 3
STRUCTURE_COLUMNS = (
    "structure_asset_intensity",
    "structure_leverage",
    "structure_accrual_intensity",
    "structure_operating_margin",
)


@dataclass(frozen=True)
class MeasureSpec:
    key: str
    label: str
    column: str | None


MEASURES = (
    MeasureSpec("dcf_cheap", "DCF Cheap", "cheap"),
    MeasureSpec("pbr", "PBR (B/M)", "value_btm"),
    MeasureSpec("per", "PER (E/P)", "value_earnings_yield"),
    MeasureSpec("simple_per_pbr", "Simple PER+PBR", None),
)
BROAD_VALUE = MeasureSpec("broad_value", "Broad Value composite", "value_core_composite")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def source_hashes() -> dict[str, str]:
    paths = [
        SIGNALS_DEFAULT,
        BASE_ROOT / "FINAL-RESULT.json",
        BASE_ROOT / "results/signals-with-returns.csv",
        ARCANA_METADATA,
        *[PRICE_ROOT / f"marcap-{year}.parquet" for year in range(2020, 2026)],
    ]
    return {str(path): sha256_file(path) for path in paths}


def _safe_divide(top: pd.Series, bottom: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(top, errors="coerce")
    denominator = pd.to_numeric(bottom, errors="coerce")
    result = numerator / denominator
    return result.where((denominator > 0) & np.isfinite(result))


def add_structure_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["structure_asset_intensity"] = _safe_divide(
        result["fund_total_assets"], result["fund_revenue"]
    )
    result["structure_leverage"] = _safe_divide(
        pd.to_numeric(result["fund_debt"], errors="coerce").fillna(0.0),
        result["fund_total_assets"],
    )
    result["structure_accrual_intensity"] = _safe_divide(
        pd.to_numeric(result["fund_net_income"], errors="coerce")
        - pd.to_numeric(result["fund_cfo"], errors="coerce"),
        result["fund_total_assets"],
    )
    result["structure_operating_margin"] = _safe_divide(
        result["fund_ebit"], result["fund_revenue"]
    )
    result["structure_rnd_intensity"] = _safe_divide(
        result["fund_rnd"], result["fund_revenue"]
    )
    return result


def load_prices(
    price_root: Path,
    *,
    tickers: set[str],
) -> tuple[pd.Timestamp, dict[str, pd.DataFrame]]:
    pieces = []
    market_max_date = pd.Timestamp.min
    for year in range(2020, 2026):
        path = price_root / f"marcap-{year}.parquet"
        frame = pd.read_parquet(path, columns=["Code", "Date", "ChangesRatio"])
        frame["Code"] = frame["Code"].astype(str).str.zfill(6)
        frame["Date"] = pd.to_datetime(frame["Date"])
        market_max_date = max(market_max_date, pd.Timestamp(frame["Date"].max()))
        pieces.append(frame[frame["Code"].isin(tickers)])
    prices = pd.concat(pieces, ignore_index=True).sort_values(["Code", "Date"])
    groups = {
        str(ticker): group.reset_index(drop=True)
        for ticker, group in prices.groupby("Code", sort=False)
    }
    return market_max_date, groups


def add_horizon_outcomes(
    frame: pd.DataFrame,
    *,
    price_groups: dict[str, pd.DataFrame],
    market_max_date: pd.Timestamp,
) -> pd.DataFrame:
    result = frame.copy()
    for horizon in HORIZONS:
        result[f"forward_{horizon}d_return"] = np.nan
        result[f"forward_{horizon}d_exit_date"] = ""
        result[f"forward_{horizon}d_attrition"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        result[f"forward_{horizon}d_dataset_censored"] = False
    for row_number, (index, row) in enumerate(result.iterrows(), start=1):
        if not isinstance(row.get("price_date"), str) or not row["price_date"]:
            continue
        prices = price_groups.get(str(row["ticker"]).zfill(6))
        if prices is None:
            continue
        entry = pd.Timestamp(row["price_date"])
        for horizon in HORIZONS:
            target = entry + pd.Timedelta(days=horizon)
            if target > market_max_date:
                result.at[index, f"forward_{horizon}d_dataset_censored"] = True
                continue
            window = prices[(prices["Date"] > entry) & (prices["Date"] <= target)]
            if window.empty:
                result.at[index, f"forward_{horizon}d_attrition"] = True
                continue
            exit_date = pd.Timestamp(window.iloc[-1]["Date"])
            complete = (target - exit_date).days <= 10
            result.at[index, f"forward_{horizon}d_exit_date"] = exit_date.date().isoformat()
            result.at[index, f"forward_{horizon}d_attrition"] = not complete
            if complete:
                result.at[index, f"forward_{horizon}d_return"] = compound_percent(
                    window["ChangesRatio"]
                )
        if row_number % 750 == 0:
            print(f"computed horizon outcomes: {row_number}/{len(result)}", flush=True)
    reproduced = pd.to_numeric(result["forward_77d_return"], errors="coerce")
    original = pd.to_numeric(frame["forward_77d_return"], errors="coerce")
    pair = pd.concat([reproduced.rename("reproduced"), original.rename("original")], axis=1).dropna()
    max_diff = float(np.max(np.abs(pair["reproduced"] - pair["original"])))
    if (
        max_diff > 1e-12
        or reproduced.notna().sum() != original.notna().sum()
        or len(pair) != int(original.notna().sum())
    ):
        raise RuntimeError("77-day return reproduction failed")
    result.attrs["return_77d_reproduction"] = {
        "matched_rows": len(pair),
        "max_abs_difference": max_diff,
    }
    return result


def load_future_fundamental_index(metadata_path: Path) -> dict[str, pd.DataFrame]:
    metadata = pd.read_csv(
        metadata_path,
        dtype={"stock_code": str, "rcept_no": str},
        low_memory=False,
    )
    metadata["stock_code"] = metadata["stock_code"].str.zfill(6)
    metadata["report_date"] = pd.to_datetime(metadata["report_date"])
    annual = metadata[
        (metadata["source_type"] == "statement")
        & (pd.to_numeric(metadata["fiscal_month"], errors="coerce") == 12)
    ].copy()
    annual["fiscal_year"] = pd.to_numeric(annual["fiscal_year"], errors="coerce").astype("Int64")
    annual = annual.dropna(subset=["fiscal_year", "report_date"])
    annual = annual.sort_values(["stock_code", "report_date", "rcept_no"])
    return {
        str(ticker): group.reset_index(drop=True)
        for ticker, group in annual.groupby("stock_code", sort=False)
    }


def add_next_fundamentals(
    frame: pd.DataFrame,
    *,
    metadata_by_ticker: dict[str, pd.DataFrame],
    snapshot_root: Path,
) -> pd.DataFrame:
    result = frame.copy()
    output_columns = (
        "next_report_date",
        "next_fiscal_year",
        "next_fund_revenue",
        "next_fund_ebit",
        "next_fund_cfo",
    )
    result["next_report_date"] = ""
    for column in output_columns[1:]:
        result[column] = np.nan
    snapshot_cache: dict[tuple[str, int], dict[str, float | None] | None] = {}
    for index, row in result.iterrows():
        ticker = str(row["ticker"]).zfill(6)
        current_year_value = pd.to_numeric(pd.Series([row["latest_fiscal_year"]]), errors="coerce").iloc[0]
        if pd.isna(current_year_value):
            continue
        current_year = int(current_year_value)
        signal_date = pd.Timestamp(row["signal_date"])
        metadata = metadata_by_ticker.get(ticker)
        if metadata is None:
            continue
        future = metadata[
            (metadata["fiscal_year"].astype(int) > current_year)
            & (metadata["report_date"] > signal_date)
            & (metadata["report_date"] <= signal_date + pd.Timedelta(days=550))
        ]
        if future.empty:
            continue
        source = future.sort_values(["report_date", "fiscal_year", "rcept_no"]).iloc[0]
        fiscal_year = int(source["fiscal_year"])
        cache_key = (ticker, fiscal_year)
        if cache_key not in snapshot_cache:
            path = snapshot_root / f"kr_normalized_{ticker}_{fiscal_year}.12.csv"
            snapshot_cache[cache_key] = (
                extract_value_fundamentals(pd.read_csv(path, low_memory=False))
                if path.exists()
                else None
            )
        metrics = snapshot_cache[cache_key]
        if metrics is None:
            continue
        result.at[index, "next_report_date"] = pd.Timestamp(source["report_date"]).date().isoformat()
        result.at[index, "next_fiscal_year"] = fiscal_year
        result.at[index, "next_fund_revenue"] = metrics["fund_revenue"]
        result.at[index, "next_fund_ebit"] = metrics["fund_ebit"]
        result.at[index, "next_fund_cfo"] = metrics["fund_cfo"]
    current_ebit = pd.to_numeric(result["fund_ebit"], errors="coerce")
    next_ebit = pd.to_numeric(result["next_fund_ebit"], errors="coerce")
    result["next_ebit_growth"] = next_ebit / current_ebit - 1.0
    known = (current_ebit > 0) & next_ebit.notna()
    deterioration = pd.Series(pd.NA, index=result.index, dtype="boolean")
    deterioration.loc[known] = (
        (next_ebit.loc[known] <= 0)
        | (result.loc[known, "next_ebit_growth"] <= -0.30)
    )
    result["next_ebit_deterioration"] = deterioration
    return result


def non_archetype(frame: pd.DataFrame) -> pd.Series:
    return ~frame["finance_hint"].astype(bool) & ~frame["holding_hint"].astype(bool)


def availability_mask(frame: pd.DataFrame, measure: str) -> pd.Series:
    base = non_archetype(frame)
    if measure == "dcf_cheap":
        return base & frame["status"].eq("ELIGIBLE") & pd.to_numeric(frame["cheap"], errors="coerce").notna()
    if measure == "dcf_calculable":
        return base & pd.to_numeric(frame["cheap"], errors="coerce").notna()
    if measure == "pbr":
        return base & pd.to_numeric(frame["value_btm"], errors="coerce").notna()
    if measure == "per":
        return base & pd.to_numeric(frame["value_earnings_yield"], errors="coerce").notna()
    if measure == "simple_per_pbr":
        return availability_mask(frame, "pbr") & availability_mask(frame, "per")
    if measure == "broad_value":
        return base & pd.to_numeric(frame["value_core_composite"], errors="coerce").notna()
    raise KeyError(measure)


def score_series(frame: pd.DataFrame, measure: str) -> pd.Series:
    if measure == "dcf_cheap":
        return pd.to_numeric(frame["cheap"], errors="coerce")
    if measure == "pbr":
        return pd.to_numeric(frame["value_btm"], errors="coerce")
    if measure == "per":
        return pd.to_numeric(frame["value_earnings_yield"], errors="coerce")
    if measure == "simple_per_pbr":
        return pd.concat(
            [
                rank_normal_score(frame["value_btm"]).rename("pbr"),
                rank_normal_score(frame["value_earnings_yield"]).rename("per"),
            ],
            axis=1,
        ).mean(axis=1).where(
            pd.to_numeric(frame["value_btm"], errors="coerce").notna()
            & pd.to_numeric(frame["value_earnings_yield"], errors="coerce").notna()
        )
    if measure == "broad_value":
        return pd.to_numeric(frame["value_core_composite"], errors="coerce")
    raise KeyError(measure)


def common_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        availability_mask(frame, "dcf_cheap")
        & availability_mask(frame, "pbr")
        & availability_mask(frame, "per")
    )


def measure_label(key: str) -> str:
    for spec in (*MEASURES, BROAD_VALUE):
        if spec.key == key:
            return spec.label
    return key


def inference(values: Sequence[float], *, horizon: int) -> dict[str, Any]:
    clean = [float(value) for value in values if pd.notna(value) and math.isfinite(float(value))]
    nw = newey_west_mean(clean, lag=HORIZON_LAGS[horizon])
    bootstrap = moving_block_bootstrap_mean(
        clean,
        block_length=BLOCK_LENGTHS[horizon],
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
    )
    t_value = float(nw["t"])
    p_value = float(2 * stats.norm.sf(abs(t_value))) if math.isfinite(t_value) else float("nan")
    return {"newey_west": nw, "moving_block_bootstrap": bootstrap, "hac_normal_p": p_value}


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    finite = {key: value for key, value in p_values.items() if math.isfinite(float(value))}
    ordered = sorted(finite, key=finite.get)
    result: dict[str, float] = {key: float("nan") for key in p_values}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        adjusted = min(1.0, (count - index) * finite[key])
        running = max(running, adjusted)
        result[key] = running
    return result


def paired_dcf_differences(
    frame: pd.DataFrame,
    *,
    metrics: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for benchmark in ("pbr", "per", "simple_per_pbr"):
        for metric, horizon in metrics.items():
            pivot = frame.pivot(index="signal_date", columns="measure", values=metric)
            pair = pivot[["dcf_cheap", benchmark]].dropna()
            delta = pair["dcf_cheap"] - pair[benchmark]
            stats_map = inference(delta.tolist(), horizon=horizon)
            key = f"{benchmark}::{metric}"
            p_values[key] = float(stats_map["hac_normal_p"])
            rows.append(
                {
                    "benchmark": benchmark,
                    "benchmark_label": measure_label(benchmark),
                    "metric": metric,
                    "paired_quarters": len(delta),
                    "dcf_minus_benchmark_mean": float(stats_map["newey_west"]["mean"]),
                    "difference_hac_t": float(stats_map["newey_west"]["t"]),
                    "difference_hac_p": float(stats_map["hac_normal_p"]),
                    "difference_boot_ci_low": float(
                        stats_map["moving_block_bootstrap"]["ci_low"]
                    ),
                    "difference_boot_ci_high": float(
                        stats_map["moving_block_bootstrap"]["ci_high"]
                    ),
                    "_key": key,
                }
            )
    adjusted = holm_adjust(p_values)
    result = pd.DataFrame(rows)
    result["holm_adjusted_p"] = result["_key"].map(adjusted)
    return result.drop(columns="_key")


def portfolio_metrics(frame: pd.DataFrame, *, signal: str, returns: str) -> dict[str, float | int]:
    work = frame[[signal, returns]].apply(pd.to_numeric, errors="coerce").dropna()
    output: dict[str, float | int] = {
        "n": len(work),
        "ic": spearman_ic(work, signal, returns),
        "mean_return": float(work[returns].mean()) if len(work) else np.nan,
    }
    if len(work) < 10:
        output.update(
            {
                "q1": np.nan,
                "q5": np.nan,
                "q5_minus_q1": np.nan,
                "top_quintile_return": np.nan,
                "top_quintile_excess": np.nan,
            }
        )
        return output
    work = work.copy()
    work["quintile"] = pd.qcut(
        work[signal].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]
    ).astype(int)
    means = work.groupby("quintile")[returns].mean()
    q1 = float(means.get(1, np.nan))
    q5 = float(means.get(5, np.nan))
    output.update(
        {
            "q1": q1,
            "q5": q5,
            "q5_minus_q1": q5 - q1,
            "top_quintile_return": q5,
            "top_quintile_excess": q5 - float(work[returns].mean()),
        }
    )
    return output


def coverage_table(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = frame[non_archetype(frame)].copy()
    net_income = pd.to_numeric(base["fund_net_income"], errors="coerce")
    ebit = pd.to_numeric(base["fund_ebit"], errors="coerce")
    segments = {
        "all_non_archetype": pd.Series(True, index=base.index),
        "positive_net_income": net_income > 0,
        "nonpositive_net_income": net_income <= 0,
        "operating_profit_net_loss": (net_income <= 0) & (ebit > 0),
    }
    measures = (
        ("dcf_calculable", "DCF calculable"),
        ("dcf_cheap", "DCF trusted/eligible"),
        ("pbr", "PBR"),
        ("per", "PER"),
        ("simple_per_pbr", "Simple PER+PBR"),
        ("broad_value", "Broad Value composite"),
    )
    rows = []
    for segment, segment_mask in segments.items():
        denominator = int(segment_mask.sum())
        for key, label in measures:
            valid = availability_mask(base, key) & segment_mask
            rows.append(
                {
                    "segment": segment,
                    "measure": key,
                    "measure_label": label,
                    "available_rows": int(valid.sum()),
                    "segment_rows": denominator,
                    "coverage": float(valid.sum() / denominator) if denominator else np.nan,
                    "unique_tickers": int(base.loc[valid, "ticker"].nunique()),
                }
            )
    loss_excluded = base[
        (net_income <= 0)
        & pd.to_numeric(base["cheap"], errors="coerce").notna()
        & ~base["status"].eq("ELIGIBLE")
    ]
    reasons: Counter[str] = Counter()
    for value in loss_excluded["status_detail"].fillna("").astype(str):
        tokens = [token for token in value.split("|") if token]
        reasons.update(tokens or ["NO_DETAIL"])
    reason_rows = pd.DataFrame(
        [
            {"reason": reason, "count": count, "share_of_reason_tokens": count / sum(reasons.values())}
            for reason, count in reasons.most_common()
        ]
    )
    return pd.DataFrame(rows), reason_rows


def formation_frame(frame: pd.DataFrame, *, lane: str, measure: str) -> pd.DataFrame:
    if lane == "common":
        result = frame[common_mask(frame)].copy()
    elif lane == "own_coverage":
        result = frame[availability_mask(frame, measure)].copy()
    else:
        raise KeyError(lane)
    result["_score"] = score_series(result, measure)
    return result.dropna(subset=["_score"])


def evaluate_horizons(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    measure_keys = [spec.key for spec in MEASURES]
    for lane in ("common", "own_coverage"):
        for signal_date, date_frame in frame.groupby("signal_date", sort=True):
            for measure in measure_keys:
                formed = formation_frame(date_frame, lane=lane, measure=measure)
                for horizon in HORIZONS:
                    returns = f"forward_{horizon}d_return"
                    metrics = portfolio_metrics(formed, signal="_score", returns=returns)
                    observable = ~formed[f"forward_{horizon}d_dataset_censored"].astype(bool)
                    attrition = formed.loc[observable, f"forward_{horizon}d_attrition"].astype("boolean")
                    rows.append(
                        {
                            "lane": lane,
                            "signal_date": signal_date,
                            "measure": measure,
                            "measure_label": measure_label(measure),
                            "horizon_days": horizon,
                            "formation_n": len(formed),
                            "observable_n": int(observable.sum()),
                            "attrition_n": int(attrition.fillna(False).sum()),
                            "attrition_rate": float(attrition.mean()) if len(attrition) else np.nan,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def summarize_horizons(
    quarterly: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    metrics = ("ic", "q5_minus_q1", "top_quintile_return", "top_quintile_excess", "attrition_rate")
    for (lane, measure, horizon), group in quarterly.groupby(
        ["lane", "measure", "horizon_days"], sort=False
    ):
        valid_ic = pd.to_numeric(group["ic"], errors="coerce").notna()
        observed_group = group[valid_ic]
        stats_map = {
            metric: inference(pd.to_numeric(group[metric], errors="coerce").dropna().tolist(), horizon=int(horizon))
            for metric in metrics
        }
        rows.append(
            {
                "lane": lane,
                "measure": measure,
                "measure_label": str(group["measure_label"].iloc[0]),
                "horizon_days": int(horizon),
                "quarters": int(valid_ic.sum()),
                "average_n": float(pd.to_numeric(observed_group["n"], errors="coerce").mean()),
                "mean_ic": float(stats_map["ic"]["newey_west"]["mean"]),
                "ic_hac_t": float(stats_map["ic"]["newey_west"]["t"]),
                "ic_boot_ci_low": float(stats_map["ic"]["moving_block_bootstrap"]["ci_low"]),
                "ic_boot_ci_high": float(stats_map["ic"]["moving_block_bootstrap"]["ci_high"]),
                "positive_ic_quarter_share": float(
                    (pd.to_numeric(observed_group["ic"], errors="coerce") > 0).mean()
                ),
                "mean_q5_minus_q1": float(stats_map["q5_minus_q1"]["newey_west"]["mean"]),
                "q5_minus_q1_hac_t": float(stats_map["q5_minus_q1"]["newey_west"]["t"]),
                "mean_top_quintile_return": float(
                    stats_map["top_quintile_return"]["newey_west"]["mean"]
                ),
                "mean_top_quintile_excess": float(
                    stats_map["top_quintile_excess"]["newey_west"]["mean"]
                ),
                "mean_attrition_rate": float(stats_map["attrition_rate"]["newey_west"]["mean"]),
            }
        )
        detail[f"{lane}::{measure}::{horizon}"] = stats_map
    summary = pd.DataFrame(rows)
    differences: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    common = quarterly[quarterly["lane"] == "common"]
    for horizon in HORIZONS:
        horizon_frame = common[common["horizon_days"] == horizon]
        for benchmark in ("pbr", "per", "simple_per_pbr"):
            for metric in ("ic", "q5_minus_q1", "top_quintile_return"):
                pivot = horizon_frame.pivot(index="signal_date", columns="measure", values=metric)
                pair = pivot[["dcf_cheap", benchmark]].dropna()
                delta = pair["dcf_cheap"] - pair[benchmark]
                stats_map = inference(delta.tolist(), horizon=horizon)
                key = f"{horizon}::{benchmark}::{metric}"
                p_values[key] = float(stats_map["hac_normal_p"])
                differences.append(
                    {
                        "horizon_days": horizon,
                        "benchmark": benchmark,
                        "benchmark_label": measure_label(benchmark),
                        "metric": metric,
                        "paired_quarters": len(delta),
                        "dcf_minus_benchmark_mean": float(stats_map["newey_west"]["mean"]),
                        "difference_hac_t": float(stats_map["newey_west"]["t"]),
                        "difference_hac_p": float(stats_map["hac_normal_p"]),
                        "difference_boot_ci_low": float(
                            stats_map["moving_block_bootstrap"]["ci_low"]
                        ),
                        "difference_boot_ci_high": float(
                            stats_map["moving_block_bootstrap"]["ci_high"]
                        ),
                        "_key": key,
                    }
                )
    adjusted = holm_adjust(p_values)
    difference_frame = pd.DataFrame(differences)
    difference_frame["holm_adjusted_p"] = difference_frame["_key"].map(adjusted)
    difference_frame = difference_frame.drop(columns="_key")
    return summary, difference_frame, detail


def pooled_sector(frame: pd.DataFrame) -> pd.Series:
    sectors = frame["current_sector"].fillna(UNKNOWN_SECTOR).astype(str)
    counts = sectors.value_counts()
    rare = set(counts[counts < MIN_SECTOR_SIZE].index) | {UNKNOWN_SECTOR}
    return sectors.where(~sectors.isin(rare), "OTHER_RARE_OR_UNKNOWN")


def regression_r2(y: pd.Series, numeric: pd.DataFrame, categories: pd.Series | None = None) -> float:
    parts = [rank_normal_score(y).rename("_y")]
    ranked_numeric = pd.DataFrame(
        {column: rank_normal_score(numeric[column]) for column in numeric.columns},
        index=numeric.index,
    )
    parts.append(ranked_numeric)
    work = pd.concat(parts, axis=1)
    if categories is not None:
        work["_category"] = categories.astype(str)
    required = ["_y", *ranked_numeric.columns]
    valid = work[required].notna().all(axis=1)
    if int(valid.sum()) <= len(required) + 2:
        return float("nan")
    x_parts = [np.ones((int(valid.sum()), 1))]
    if len(ranked_numeric.columns):
        x_parts.append(work.loc[valid, ranked_numeric.columns].to_numpy(dtype=float))
    if categories is not None:
        dummies = pd.get_dummies(work.loc[valid, "_category"], drop_first=True, dtype=float)
        if not dummies.empty:
            x_parts.append(dummies.to_numpy(dtype=float))
    x = np.column_stack(x_parts)
    y_array = work.loc[valid, "_y"].to_numpy(dtype=float)
    fitted = x @ np.linalg.lstsq(x, y_array, rcond=None)[0]
    total = float(np.sum((y_array - y_array.mean()) ** 2))
    return 1.0 - float(np.sum((y_array - fitted) ** 2)) / total if total > 0 else float("nan")


def top_sector_hhi(frame: pd.DataFrame, score: pd.Series, sectors: pd.Series) -> float:
    work = pd.DataFrame({"score": score, "sector": sectors}).dropna()
    if len(work) < 10:
        return float("nan")
    count = max(1, math.ceil(len(work) / 5))
    shares = work.nlargest(count, "score")["sector"].value_counts(normalize=True)
    return float(np.sum(np.square(shares.to_numpy(dtype=float))))


def evaluate_industry(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    quarterly_rows: list[dict[str, Any]] = []
    for signal_date, date_frame in frame.groupby("signal_date", sort=True):
        common = date_frame[common_mask(date_frame)].copy()
        sectors = pooled_sector(common)
        raw_sectors = common["current_sector"].fillna(UNKNOWN_SECTOR).astype(str)
        for spec in MEASURES:
            score = score_series(common, spec.key)
            temp = common.copy()
            temp["_score"] = score
            temp["_sector"] = sectors
            temp["_sector_residual"] = residualize_cross_section(
                temp,
                target="_score",
                numeric_controls=[],
                categorical_controls=["_sector"],
            )
            row: dict[str, Any] = {
                "signal_date": signal_date,
                "measure": spec.key,
                "measure_label": spec.label,
                "n": int(score.notna().sum()),
                "pooled_sector_count": int(sectors.nunique()),
                "sector_r2": regression_r2(score, pd.DataFrame(index=common.index), sectors),
                "top_quintile_sector_hhi": top_sector_hhi(common, score, raw_sectors),
            }
            for horizon in (77, 365):
                metrics = portfolio_metrics(
                    temp,
                    signal="_sector_residual",
                    returns=f"forward_{horizon}d_return",
                )
                row[f"sector_neutral_{horizon}d_n"] = metrics["n"]
                row[f"sector_neutral_{horizon}d_ic"] = metrics["ic"]
                row[f"sector_neutral_{horizon}d_q5_minus_q1"] = metrics["q5_minus_q1"]
            quarterly_rows.append(row)
    quarterly = pd.DataFrame(quarterly_rows)
    summary_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for measure, group in quarterly.groupby("measure", sort=False):
        stats_map: dict[str, Any] = {}
        for metric, horizon in (
            ("sector_r2", 77),
            ("top_quintile_sector_hhi", 77),
            ("sector_neutral_77d_ic", 77),
            ("sector_neutral_77d_q5_minus_q1", 77),
            ("sector_neutral_365d_ic", 365),
            ("sector_neutral_365d_q5_minus_q1", 365),
        ):
            stats_map[metric] = inference(
                pd.to_numeric(group[metric], errors="coerce").dropna().tolist(),
                horizon=horizon,
            )
        summary_rows.append(
            {
                "measure": measure,
                "measure_label": str(group["measure_label"].iloc[0]),
                "mean_sector_r2": float(stats_map["sector_r2"]["newey_west"]["mean"]),
                "mean_top_quintile_sector_hhi": float(
                    stats_map["top_quintile_sector_hhi"]["newey_west"]["mean"]
                ),
                "sector_neutral_77d_ic": float(
                    stats_map["sector_neutral_77d_ic"]["newey_west"]["mean"]
                ),
                "sector_neutral_77d_ic_hac_t": float(
                    stats_map["sector_neutral_77d_ic"]["newey_west"]["t"]
                ),
                "sector_neutral_365d_ic": float(
                    stats_map["sector_neutral_365d_ic"]["newey_west"]["mean"]
                ),
                "sector_neutral_365d_ic_hac_t": float(
                    stats_map["sector_neutral_365d_ic"]["newey_west"]["t"]
                ),
                "sector_neutral_77d_q5_minus_q1": float(
                    stats_map["sector_neutral_77d_q5_minus_q1"]["newey_west"]["mean"]
                ),
                "sector_neutral_365d_q5_minus_q1": float(
                    stats_map["sector_neutral_365d_q5_minus_q1"]["newey_west"]["mean"]
                ),
            }
        )
        detail[measure] = stats_map
    return quarterly, pd.DataFrame(summary_rows), detail


def evaluate_accounting_structure(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exposure_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for signal_date, date_frame in frame.groupby("signal_date", sort=True):
        common = date_frame[common_mask(date_frame)].copy()
        sectors = pooled_sector(common)
        attributes = common[list(STRUCTURE_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        for spec in MEASURES:
            score = score_series(common, spec.key)
            direct_r2 = regression_r2(score, attributes)
            sector_r2 = regression_r2(score, pd.DataFrame(index=common.index), sectors)
            full_r2 = regression_r2(score, attributes, sectors)
            row: dict[str, Any] = {
                "signal_date": signal_date,
                "measure": spec.key,
                "measure_label": spec.label,
                "n": int(pd.concat([score, attributes], axis=1).dropna().shape[0]),
                "structure_joint_r2": direct_r2,
                "sector_r2": sector_r2,
                "sector_plus_structure_r2": full_r2,
                "incremental_structure_r2_over_sector": full_r2 - sector_r2,
            }
            for attribute in STRUCTURE_COLUMNS:
                pair = pd.concat(
                    [rank_normal_score(score).rename("score"), rank_normal_score(common[attribute]).rename("attribute")],
                    axis=1,
                ).dropna()
                row[f"corr__{attribute}"] = (
                    float(pair["score"].corr(pair["attribute"])) if len(pair) > 2 else np.nan
                )
                valid = common[[attribute, "forward_365d_return"]].copy()
                valid["_score"] = score
                valid = valid.dropna()
                if len(valid) >= 15:
                    valid["structure_tercile"] = pd.qcut(
                        valid[attribute].rank(method="first"), 3, labels=[1, 2, 3]
                    ).astype(int)
                    for tercile, subgroup in valid.groupby("structure_tercile"):
                        bin_rows.append(
                            {
                                "signal_date": signal_date,
                                "measure": spec.key,
                                "measure_label": spec.label,
                                "attribute": attribute,
                                "tercile": int(tercile),
                                "n": len(subgroup),
                                "ic_365d": spearman_ic(
                                    subgroup, "_score", "forward_365d_return"
                                ),
                            }
                        )
            row["mean_abs_structure_corr"] = float(
                np.nanmean(
                    [abs(float(row[f"corr__{attribute}"])) for attribute in STRUCTURE_COLUMNS]
                )
            )
            exposure_rows.append(row)
    exposures = pd.DataFrame(exposure_rows)
    bins = pd.DataFrame(bin_rows)
    summary_rows = []
    for measure, group in exposures.groupby("measure", sort=False):
        mean_corr = float(
            np.nanmean(
                np.abs(
                    group[[f"corr__{column}" for column in STRUCTURE_COLUMNS]].to_numpy(
                        dtype=float
                    )
                )
            )
        )
        measure_bins = bins[bins["measure"] == measure]
        bin_means = (
            measure_bins.groupby(["attribute", "tercile"])["ic_365d"].mean().dropna()
        )
        attribute_dispersion = (
            bin_means.groupby("attribute").std(ddof=0) if len(bin_means) else pd.Series(dtype=float)
        )
        summary_rows.append(
            {
                "measure": measure,
                "measure_label": str(group["measure_label"].iloc[0]),
                "mean_structure_joint_r2": float(group["structure_joint_r2"].mean()),
                "mean_incremental_structure_r2_over_sector": float(
                    group["incremental_structure_r2_over_sector"].mean()
                ),
                "mean_abs_structure_corr": mean_corr,
                "mean_365d_ic_dispersion_across_structure_terciles": float(
                    attribute_dispersion.mean()
                ),
                "positive_structure_tercile_share": float((bin_means > 0).mean()),
                "structure_bin_count": len(bin_means),
            }
        )
    return exposures, bins, pd.DataFrame(summary_rows)


def _top_quintile(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) < 10:
        return frame.iloc[0:0]
    count = max(1, math.ceil(len(frame) / 5))
    return frame.nlargest(count, "_score")


def evaluate_value_traps(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    quarterly_rows: list[dict[str, Any]] = []
    selection_sets: dict[tuple[str, str, str], set[str]] = {}
    for lane in ("common", "own_coverage"):
        for signal_date, date_frame in frame.groupby("signal_date", sort=True):
            for spec in MEASURES:
                formed = formation_frame(date_frame, lane=lane, measure=spec.key)
                observable = ~formed["forward_365d_dataset_censored"].astype(bool)
                formed = formed[observable].copy()
                top = _top_quintile(formed)
                selection_sets[(lane, signal_date, spec.key)] = set(top["ticker"].astype(str))
                if top.empty:
                    continue
                attrition = top["forward_365d_attrition"].astype("boolean")
                returns = pd.to_numeric(top["forward_365d_return"], errors="coerce")
                severe = returns <= -0.20
                complete_base_returns = pd.to_numeric(
                    formed["forward_365d_return"], errors="coerce"
                ).dropna()
                bottom_cutoff = (
                    float(complete_base_returns.quantile(0.20))
                    if len(complete_base_returns)
                    else np.nan
                )
                relative = (
                    (returns <= bottom_cutoff).where(returns.notna())
                    if math.isfinite(bottom_cutoff)
                    else pd.Series(np.nan, index=top.index)
                )
                deterioration = top["next_ebit_deterioration"].astype("boolean")
                combined_known = deterioration.notna() & attrition.notna()
                combined = (
                    severe.fillna(False)
                    | attrition.fillna(False)
                    | deterioration.fillna(False)
                )
                quarterly_rows.append(
                    {
                        "lane": lane,
                        "signal_date": signal_date,
                        "measure": spec.key,
                        "measure_label": spec.label,
                        "formation_n": len(formed),
                        "top_n": len(top),
                        "complete_return_n": int(returns.notna().sum()),
                        "next_ebit_n": int(deterioration.notna().sum()),
                        "combined_known_n": int(combined_known.sum()),
                        "mean_complete_return": float(returns.mean()),
                        "severe_loss_rate": float(severe.mean()),
                        "relative_bottom_quintile_rate": float(relative.mean()),
                        "attrition_rate": float(attrition.mean()),
                        "ebit_deterioration_rate": float(deterioration.mean()),
                        "combined_trap_rate": float(combined[combined_known].mean())
                        if combined_known.any()
                        else np.nan,
                    }
                )
    quarterly = pd.DataFrame(quarterly_rows)
    summary_rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    metrics = (
        "mean_complete_return",
        "severe_loss_rate",
        "relative_bottom_quintile_rate",
        "attrition_rate",
        "ebit_deterioration_rate",
        "combined_trap_rate",
    )
    for (lane, measure), group in quarterly.groupby(["lane", "measure"], sort=False):
        stats_map = {
            metric: inference(pd.to_numeric(group[metric], errors="coerce").dropna().tolist(), horizon=365)
            for metric in metrics
        }
        summary_rows.append(
            {
                "lane": lane,
                "measure": measure,
                "measure_label": str(group["measure_label"].iloc[0]),
                "quarters": int(group["signal_date"].nunique()),
                "average_top_n": float(group["top_n"].mean()),
                "average_combined_known_n": float(group["combined_known_n"].mean()),
                "mean_complete_return": float(stats_map["mean_complete_return"]["newey_west"]["mean"]),
                "mean_severe_loss_rate": float(stats_map["severe_loss_rate"]["newey_west"]["mean"]),
                "mean_relative_bottom_quintile_rate": float(
                    stats_map["relative_bottom_quintile_rate"]["newey_west"]["mean"]
                ),
                "mean_attrition_rate": float(stats_map["attrition_rate"]["newey_west"]["mean"]),
                "mean_ebit_deterioration_rate": float(
                    stats_map["ebit_deterioration_rate"]["newey_west"]["mean"]
                ),
                "mean_combined_trap_rate": float(
                    stats_map["combined_trap_rate"]["newey_west"]["mean"]
                ),
                "combined_trap_rate_hac_t": float(
                    stats_map["combined_trap_rate"]["newey_west"]["t"]
                ),
            }
        )
        detail[f"{lane}::{measure}"] = stats_map
    differences: list[dict[str, Any]] = []
    common = quarterly[quarterly["lane"] == "common"]
    for benchmark in ("pbr", "per", "simple_per_pbr"):
        for metric in (
            "mean_complete_return",
            "severe_loss_rate",
            "relative_bottom_quintile_rate",
            "ebit_deterioration_rate",
            "combined_trap_rate",
        ):
            pivot = common.pivot(index="signal_date", columns="measure", values=metric)
            pair = pivot[["dcf_cheap", benchmark]].dropna()
            delta = pair["dcf_cheap"] - pair[benchmark]
            stats_map = inference(delta.tolist(), horizon=365)
            overlaps = []
            for signal_date in pair.index:
                dcf = selection_sets.get(("common", signal_date, "dcf_cheap"), set())
                other = selection_sets.get(("common", signal_date, benchmark), set())
                union = dcf | other
                overlaps.append(len(dcf & other) / len(union) if union else np.nan)
            differences.append(
                {
                    "benchmark": benchmark,
                    "benchmark_label": measure_label(benchmark),
                    "metric": metric,
                    "paired_quarters": len(delta),
                    "dcf_minus_benchmark_mean": float(stats_map["newey_west"]["mean"]),
                    "difference_hac_t": float(stats_map["newey_west"]["t"]),
                    "difference_hac_p": float(stats_map["hac_normal_p"]),
                    "difference_boot_ci_low": float(
                        stats_map["moving_block_bootstrap"]["ci_low"]
                    ),
                    "difference_boot_ci_high": float(
                        stats_map["moving_block_bootstrap"]["ci_high"]
                    ),
                    "mean_top_selection_jaccard": float(np.nanmean(overlaps)),
                }
            )
    difference_frame = pd.DataFrame(differences)
    trap_p = {
        f"{row.benchmark}::{row.metric}": float(row.difference_hac_p)
        for row in difference_frame.itertuples(index=False)
    }
    trap_adjusted = holm_adjust(trap_p)
    difference_frame["holm_adjusted_p"] = [
        trap_adjusted[f"{row.benchmark}::{row.metric}"]
        for row in difference_frame.itertuples(index=False)
    ]
    return quarterly, pd.DataFrame(summary_rows), difference_frame, detail


def measurement_pair_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal_date, date_frame in frame.groupby("signal_date", sort=True):
        common = date_frame[common_mask(date_frame)].copy()
        scores = {
            spec.key: score_series(common, spec.key)
            for spec in MEASURES
        }
        for left_index, left in enumerate(MEASURES):
            for right in MEASURES[left_index + 1 :]:
                pair = pd.concat(
                    [scores[left.key].rename("left"), scores[right.key].rename("right")],
                    axis=1,
                ).dropna()
                rows.append(
                    {
                        "signal_date": signal_date,
                        "left": left.key,
                        "right": right.key,
                        "n": len(pair),
                        "spearman_correlation": float(pair["left"].corr(pair["right"], method="spearman")),
                    }
                )
    return pd.DataFrame(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(numeric) else f"{numeric:.{digits}f}"


def _pct(value: Any, digits: int = 1) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "NA"
    return "NA" if not math.isfinite(numeric) else f"{100 * numeric:.{digits}f}%"


def write_report(
    output: Path,
    *,
    coverage: pd.DataFrame,
    horizon_summary: pd.DataFrame,
    horizon_differences: pd.DataFrame,
    industry_summary: pd.DataFrame,
    industry_differences: pd.DataFrame,
    structure_summary: pd.DataFrame,
    structure_differences: pd.DataFrame,
    trap_summary: pd.DataFrame,
    trap_differences: pd.DataFrame,
    correlations: pd.DataFrame,
    return_reproduction: dict[str, Any],
) -> None:
    horizon_lines = [
        "| Horizon | Measure | Avg N | Mean IC | HAC t | Q5-Q1 | Top return |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    primary_horizons = horizon_summary[horizon_summary["lane"] == "common"].copy()
    for horizon in HORIZONS:
        for row in primary_horizons[primary_horizons["horizon_days"] == horizon].itertuples(
            index=False
        ):
            horizon_lines.append(
                f"| {horizon}d | {row.measure_label} | {row.average_n:.1f} | "
                f"{row.mean_ic:.4f} | {row.ic_hac_t:.2f} | {row.mean_q5_minus_q1:.2%} | "
                f"{row.mean_top_quintile_return:.2%} |"
            )
    coverage_lines = [
        "| Segment | Measure | Available | Coverage |",
        "|---|---|---:|---:|",
    ]
    coverage_view = coverage[
        coverage["segment"].isin(["all_non_archetype", "nonpositive_net_income", "operating_profit_net_loss"])
        & coverage["measure"].isin(["dcf_calculable", "dcf_cheap", "pbr", "per"])
    ]
    for row in coverage_view.itertuples(index=False):
        coverage_lines.append(
            f"| {row.segment} | {row.measure_label} | {row.available_rows}/{row.segment_rows} | "
            f"{row.coverage:.1%} |"
        )
    industry_lines = [
        "| Measure | Sector R² | Top-sector HHI | Sector-neutral 77d IC | 365d IC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in industry_summary.itertuples(index=False):
        industry_lines.append(
            f"| {row.measure_label} | {row.mean_sector_r2:.1%} | "
            f"{row.mean_top_quintile_sector_hhi:.3f} | {row.sector_neutral_77d_ic:.4f} | "
            f"{row.sector_neutral_365d_ic:.4f} |"
        )
    structure_lines = [
        "| Measure | Structure R² | Incremental over sector | Mean |corr| | 365d IC dispersion | Positive bins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in structure_summary.itertuples(index=False):
        structure_lines.append(
            f"| {row.measure_label} | {row.mean_structure_joint_r2:.1%} | "
            f"{row.mean_incremental_structure_r2_over_sector:.1%} | "
            f"{row.mean_abs_structure_corr:.3f} | "
            f"{row.mean_365d_ic_dispersion_across_structure_terciles:.4f} | "
            f"{row.positive_structure_tercile_share:.1%} |"
        )
    trap_lines = [
        "| Lane | Measure | Top N | Combined N | 365d return | Severe loss | Attrition | EBIT deterioration | Combined trap |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in trap_summary.itertuples(index=False):
        trap_lines.append(
            f"| {row.lane} | {row.measure_label} | {row.average_top_n:.1f} | "
            f"{row.average_combined_known_n:.1f} | "
            f"{row.mean_complete_return:.2%} | {row.mean_severe_loss_rate:.1%} | "
            f"{row.mean_attrition_rate:.1%} | {row.mean_ebit_deterioration_rate:.1%} | "
            f"{row.mean_combined_trap_rate:.1%} |"
        )
    corr_view = correlations[
        (correlations["left"] == "dcf_cheap")
        & correlations["right"].isin(["pbr", "per", "simple_per_pbr"])
    ]
    corr_means = corr_view.groupby("right")["spearman_correlation"].mean().to_dict()
    long_dcf = primary_horizons[primary_horizons["measure"] == "dcf_cheap"].set_index(
        "horizon_days"
    )
    common_traps = trap_summary[trap_summary["lane"] == "common"].set_index("measure")
    loss_coverage = coverage[
        (coverage["segment"] == "nonpositive_net_income")
        & coverage["measure"].isin(["dcf_calculable", "dcf_cheap", "pbr", "per"])
    ].set_index("measure")
    dcf_vs_simple_365_ic = horizon_differences[
        (horizon_differences["horizon_days"] == 365)
        & (horizon_differences["benchmark"] == "simple_per_pbr")
        & (horizon_differences["metric"] == "ic")
    ].iloc[0]
    dcf_vs_simple_trap_return = trap_differences[
        (trap_differences["benchmark"] == "simple_per_pbr")
        & (trap_differences["metric"] == "mean_complete_return")
    ].iloc[0]
    dcf_vs_simple_trap_ebit = trap_differences[
        (trap_differences["benchmark"] == "simple_per_pbr")
        & (trap_differences["metric"] == "ebit_deterioration_rate")
    ].iloc[0]
    text = f"""# DCF Cheap vs simple Value: measurement-quality diagnostics

## 범위와 판정 원칙

이 분석은 기존 2020-03~2025-09 v7.1 결과를 본 뒤 수행한 **사후 진단**입니다. DCF Cheap이 독립 알파인지가 아니라 PER/PBR보다 더 좋은 Value 측정치인지 산업 비교력, 적자 커버리지, 회계구조 민감도, value trap, 장기 horizon의 다섯 축에서 비교합니다.

- `common` lane: DCF trusted, PBR, PER가 모두 존재하는 동일 종목에서 순위 품질 비교
- `own_coverage` lane: 각 측정치가 실제 사용할 수 있는 자체 표본에서 구현 결과 비교
- 산업은 현재 2026 KRX KIND 분류이므로 **비-PIT sensitivity only**입니다. 날짜별 3개 미만 업종과 미분류는 하나로 묶었습니다.
- 수익률은 현금분배 미포함 Marcap 가격수익률입니다. 목표일 10일 전부터 가격이 끊긴 종목은 장기수익률에서 누락시키지 않고 `attrition` trap으로 별도 집계했습니다.

## 핵심 관찰

- DCF와 단순 지표의 평균 순위상관은 PBR `{corr_means.get('pbr', np.nan):.3f}`, PER `{corr_means.get('per', np.nan):.3f}`, PER+PBR `{corr_means.get('simple_per_pbr', np.nan):.3f}`입니다.
- 적자 행에서 DCF는 계산 자체는 `{loss_coverage.loc['dcf_calculable', 'coverage']:.1%}` 가능하지만 frozen trusted screening을 통과하는 비율은 `{loss_coverage.loc['dcf_cheap', 'coverage']:.1%}`입니다. PBR은 `{loss_coverage.loc['pbr', 'coverage']:.1%}`, PER는 `{loss_coverage.loc['per', 'coverage']:.1%}`입니다.
- 공통표본 DCF IC는 77/182/365/730일에 각각 `{long_dcf.loc[77, 'mean_ic']:.4f}`, `{long_dcf.loc[182, 'mean_ic']:.4f}`, `{long_dcf.loc[365, 'mean_ic']:.4f}`, `{long_dcf.loc[730, 'mean_ic']:.4f}`입니다.
- 공통표본 DCF top-quintile의 365일 combined trap rate는 `{common_traps.loc['dcf_cheap', 'mean_combined_trap_rate']:.1%}`입니다. PBR `{common_traps.loc['pbr', 'mean_combined_trap_rate']:.1%}`, PER `{common_traps.loc['per', 'mean_combined_trap_rate']:.1%}`, PER+PBR `{common_traps.loc['simple_per_pbr', 'mean_combined_trap_rate']:.1%}`와 paired 차이는 아래 원자료에서 확인할 수 있습니다.

## 최종 판정

| 검정 축 | 판정 | 근거 |
|---|---|---|
| 산업 간 비교 | **열위~혼합** | DCF sector R²는 PBR보다 약간 낮지만 PER보다 높고, sector-neutral 77·365일 IC는 PBR 및 PER+PBR보다 낮음; paired 차이는 Holm 보정 후 유의하지 않음 |
| 적자기업 coverage | **계산 가능, trusted 사용은 열위** | 적자에서 DCF 계산 97.4%지만 frozen screening 통과 9.9%; PBR 99.2%, PER 0% |
| 회계구조 차이 | **혼합** | DCF 구조 R²는 PBR보다 7.6%p 낮아 유의하지만 PER+PBR composite와는 차이가 없고, 구조 tercile 성과 일관성이 우월하지 않음 |
| Value trap | **혼합** | 다음 EBIT 악화는 PER+PBR보다 `{abs(dcf_vs_simple_trap_ebit.dcf_minus_benchmark_mean):.1%}p` 낮음(Holm p=`{dcf_vs_simple_trap_ebit.holm_adjusted_p:.4g}`); top 365일 수익률도 `{abs(dcf_vs_simple_trap_return.dcf_minus_benchmark_mean):.1%}p` 낮음(Holm p=`{dcf_vs_simple_trap_return.holm_adjusted_p:.4g}`) |
| 장기 horizon | **열위** | 365일 IC가 PER+PBR보다 `{abs(dcf_vs_simple_365_ic.dcf_minus_benchmark_mean):.4f}` 낮고 paired HAC t=`{dcf_vs_simple_365_ic.difference_hac_t:.2f}`, Holm p=`{dcf_vs_simple_365_ic.holm_adjusted_p:.4f}` |

**해석 범위:** 이 실험은 model router 없이 동일한 historical FCFF DCF를 적용한 Cheap 한 종류만 검정했습니다. 따라서 아래 열위 판정은 `one-size-fits-all historical FCFF Cheap ranker`에 한정됩니다. RIM/rNPV/SOTP/NAV/APV/Mid-cycle을 실제 실행하고 route별 reference-class percentile로 통합한 Unified/Universal Value는 이 실험에서 테스트하지 않았습니다.

**종합:** 현재 static historical-FCFF Cheap은 가격수익률 순위와 coverage를 함께 보면 주력 Value ranker로서 단순 PER+PBR 또는 PBR보다 약했습니다. 다음 EBIT 악화 위험을 줄이는 보조 진단 가능성은 남지만, 이 결과를 multi-model Unified/Universal Value의 실패로 확대해석할 수 없습니다.

## 장기 horizon — 공통표본

{chr(10).join(horizon_lines)}

DCF 우월성은 `results/horizon-dcf-differences.csv`의 동일 분기 paired 차이와 Holm 보정 p-value로 판단해야 합니다. 단순히 DCF 행의 IC가 양수라는 이유로 우월하다고 보지 않습니다.

## 적자기업 및 전체 커버리지

{chr(10).join(coverage_lines)}

`DCF calculable`과 `DCF trusted/eligible`을 분리했습니다. 전자는 공정가치 숫자가 계산됐다는 뜻이고, 후자는 기존 frozen screening을 통과해 실제 성과평가에 사용된 신호입니다.

## 산업 간 비교력

{chr(10).join(industry_lines)}

Sector R²가 낮을수록 측정치 순위가 특정 산업에 덜 좌우되고, sector-neutral IC가 높을수록 업종 내부에서도 가격발견력이 남는다는 진단입니다. 현재 산업분류라는 한계 때문에 이 결과는 결정적 증거가 아닙니다.

## 회계구조 민감도

{chr(10).join(structure_lines)}

구조 변수는 자산집약도, 부채/자산, accrual `(순이익-CFO)/자산`, 영업마진입니다. 낮은 노출만으로 좋은 측정치라고 할 수 없으므로 365일 IC가 각 구조 tercile에서 얼마나 일관적인지도 함께 봅니다.

## Value trap

{chr(10).join(trap_lines)}

Severe loss는 365일 수익률 -20% 이하, EBIT deterioration은 다음 가용 연차 EBIT이 음수이거나 30% 이상 감소, combined trap은 severe loss·가격 attrition·EBIT deterioration 중 하나입니다. `results/value-trap-differences.csv`가 DCF와 단순 지표의 분기별 paired 차이를 제공합니다.

## 검증과 제한

- 기존 77일 수익률 재현: `{return_reproduction['matched_rows']}`행, 최대 절대차 `{return_reproduction['max_abs_difference']:.3e}`
- 730일 결과는 완결 가능한 초기 분기만 사용하므로 표본 기간이 더 짧습니다.
- 무작위 150종목 표본의 연속 분기 공통 종목은 평균 5개 미만이어서 순위 turnover 자체는 신뢰성 있게 비교하지 않았습니다.
- Value trap 정의와 모든 비교축은 기존 수익률을 본 뒤 정한 사후 진단입니다. 새 OOS 증거가 아닙니다.

## 산출물

- `results/horizon-summary.csv`, `results/horizon-dcf-differences.csv`
- `results/coverage.csv`, `results/dcf-loss-screening-reasons.csv`
- `results/industry-summary.csv`, `results/accounting-structure-summary.csv`
- `results/industry-differences.csv`, `results/accounting-structure-differences.csv`
- `results/value-trap-summary.csv`, `results/value-trap-differences.csv`
- `results/analysis-panel.csv`, `results/statistical-detail.json`
"""
    (output / "FINAL-REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare DCF Cheap with PER/PBR as broad Value measurements."
    )
    parser.add_argument("--signals", type=Path, default=SIGNALS_DEFAULT)
    parser.add_argument("--price-root", type=Path, default=PRICE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    signals_path = args.signals.resolve()
    price_root = args.price_root.resolve()
    output = args.output.resolve()
    if (output / "FINAL-RESULT.json").exists():
        raise FileExistsError(f"completed measurement-quality result is immutable: {output}")
    (output / "results").mkdir(parents=True, exist_ok=True)
    hashes_before = source_hashes()
    frame = pd.read_csv(signals_path, dtype={"ticker": str}, low_memory=False)
    frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    if (
        len(frame) != 3450
        or frame["signal_date"].nunique() != 23
        or str(frame["signal_date"].min()) != "2020-03-31"
        or str(frame["signal_date"].max()) != "2025-09-30"
    ):
        raise ValueError("input is not the frozen v7.1 2020-2025 signal panel")
    frame = add_structure_fields(frame)
    market_max_date, price_groups = load_prices(
        price_root,
        tickers=set(frame["ticker"].astype(str)),
    )
    frame = add_horizon_outcomes(
        frame,
        price_groups=price_groups,
        market_max_date=market_max_date,
    )
    return_reproduction = dict(frame.attrs["return_77d_reproduction"])
    del price_groups
    metadata_index = load_future_fundamental_index(ARCANA_METADATA)
    frame = add_next_fundamentals(
        frame,
        metadata_by_ticker=metadata_index,
        snapshot_root=ARCANA_SNAPSHOTS,
    )
    coverage, loss_reasons = coverage_table(frame)
    horizon_quarterly = evaluate_horizons(frame)
    horizon_summary, horizon_differences, horizon_detail = summarize_horizons(
        horizon_quarterly
    )
    industry_quarterly, industry_summary, industry_detail = evaluate_industry(frame)
    industry_differences = paired_dcf_differences(
        industry_quarterly,
        metrics={
            "sector_r2": 77,
            "top_quintile_sector_hhi": 77,
            "sector_neutral_77d_ic": 77,
            "sector_neutral_365d_ic": 365,
        },
    )
    structure_exposure, structure_bins, structure_summary = evaluate_accounting_structure(frame)
    structure_differences = paired_dcf_differences(
        structure_exposure,
        metrics={
            "structure_joint_r2": 77,
            "incremental_structure_r2_over_sector": 77,
            "mean_abs_structure_corr": 77,
        },
    )
    trap_quarterly, trap_summary, trap_differences, trap_detail = evaluate_value_traps(frame)
    correlations = measurement_pair_correlations(frame)
    outputs = {
        "analysis-panel.csv": frame,
        "coverage.csv": coverage,
        "dcf-loss-screening-reasons.csv": loss_reasons,
        "horizon-quarterly.csv": horizon_quarterly,
        "horizon-summary.csv": horizon_summary,
        "horizon-dcf-differences.csv": horizon_differences,
        "industry-quarterly.csv": industry_quarterly,
        "industry-summary.csv": industry_summary,
        "industry-differences.csv": industry_differences,
        "accounting-structure-quarterly.csv": structure_exposure,
        "accounting-structure-bins.csv": structure_bins,
        "accounting-structure-summary.csv": structure_summary,
        "accounting-structure-differences.csv": structure_differences,
        "value-trap-quarterly.csv": trap_quarterly,
        "value-trap-summary.csv": trap_summary,
        "value-trap-differences.csv": trap_differences,
        "measure-correlations.csv": correlations,
    }
    for filename, table in outputs.items():
        table.to_csv(output / "results" / filename, index=False, encoding="utf-8-sig")
    write_json(
        output / "results/statistical-detail.json",
        {
            "schema_version": "moatrader-v7.1-value-measurement-quality-statistics/1",
            "horizon": horizon_detail,
            "industry": industry_detail,
            "value_trap": trap_detail,
        },
    )
    hashes_after = source_hashes()
    changed = sorted(
        key
        for key in set(hashes_before) | set(hashes_after)
        if hashes_before.get(key) != hashes_after.get(key)
    )
    integrity = {
        "schema_version": "moatrader-v7.1-value-measurement-source-integrity/1",
        "sources_unchanged": not changed,
        "changed_paths": changed,
        "source_file_count": len(hashes_after),
    }
    write_json(output / "source-integrity.json", integrity)
    if changed:
        raise RuntimeError(f"source artifacts changed during measurement test: {changed}")
    input_manifest = {
        "schema_version": "moatrader-v7.1-value-measurement-inputs/1",
        "analysis_grade": "EX_POST_MEASUREMENT_QUALITY_DIAGNOSTIC_NOT_NEW_HOLDOUT",
        "signals": str(signals_path),
        "signals_sha256": sha256_file(signals_path),
        "price_root": str(price_root),
        "price_max_date": market_max_date.date().isoformat(),
        "arcana_metadata": str(ARCANA_METADATA),
        "sector_limitation": "CURRENT_2026_KRX_KIND_NOT_HISTORICAL_PIT; SENSITIVITY_ONLY",
        "horizons": list(HORIZONS),
        "horizon_hac_lags": HORIZON_LAGS,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "return_77d_reproduction": return_reproduction,
        "lanes": {
            "common": "DCF trusted + positive PBR + positive PER same rows",
            "own_coverage": "each measure's own non-archetype available rows",
        },
    }
    write_json(output / "input-manifest.json", input_manifest)
    write_report(
        output,
        coverage=coverage,
        horizon_summary=horizon_summary,
        horizon_differences=horizon_differences,
        industry_summary=industry_summary,
        industry_differences=industry_differences,
        structure_summary=structure_summary,
        structure_differences=structure_differences,
        trap_summary=trap_summary,
        trap_differences=trap_differences,
        correlations=correlations,
        return_reproduction=return_reproduction,
    )
    final = {
        "schema_version": "moatrader-v7.1-value-measurement-quality-final/1",
        "analysis_grade": "EX_POST_MEASUREMENT_QUALITY_DIAGNOSTIC_NOT_NEW_HOLDOUT",
        "period": ["2020-03-31", "2025-09-30"],
        "signal_date_count": 23,
        "dimensions": {
            "industry_comparability": industry_summary.to_dict("records"),
            "industry_comparability_differences": industry_differences.to_dict("records"),
            "loss_coverage": coverage[
                coverage["segment"] == "nonpositive_net_income"
            ].to_dict("records"),
            "accounting_structure": structure_summary.to_dict("records"),
            "accounting_structure_differences": structure_differences.to_dict("records"),
            "value_traps": trap_summary.to_dict("records"),
            "long_horizons": horizon_summary[
                horizon_summary["lane"] == "common"
            ].to_dict("records"),
        },
        "return_77d_reproduction": return_reproduction,
        "source_integrity": integrity,
        "report": str(output / "FINAL-REPORT.md"),
    }
    write_json(output / "FINAL-RESULT.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
