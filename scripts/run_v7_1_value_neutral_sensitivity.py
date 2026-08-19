from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from moatrader.backtest.universe_corrected import (
    moving_block_bootstrap_mean,
    newey_west_mean,
    rank_normal_score,
    residualize_cross_section,
    sha256_file,
    spearman_ic,
)


REPOSITORY = Path(__file__).resolve().parents[1]
BASE_ROOT = REPOSITORY / "data-lake/experiments/universe-corrected-residual-v7-1-2020-2025"
BASE_SIGNALS = BASE_ROOT / "results/signals-with-returns.csv"
OUTPUT_DEFAULT = (
    REPOSITORY
    / "data-lake/experiments/expectation-gap-v7-1-multi-value-neutral-sensitivity-2020-2025"
)
ARCANA_SNAPSHOTS = Path(
    r"D:\Programming\python_example\Arcana\data-lake\silver\dart\normalized-snapshots"
)
ARCANA_METADATA = Path(
    r"D:\Programming\python_example\Arcana\data-lake\silver\dart\kr_report_metadata.csv"
)
DRIVERS = ("cheap",)
HAC_LAG = 1
BLOCK_LENGTH = 4
BOOTSTRAP_REPETITIONS = 10_000
BOOTSTRAP_SEED = 42


@dataclass(frozen=True)
class ValueSpec:
    key: str
    label: str
    controls: tuple[str, ...]
    definition: str


CORE_VALUE_COLUMNS = (
    "value_btm",
    "value_earnings_yield",
    "value_fcf_yield",
    "value_sales_yield",
    "value_cfo_yield",
    "value_ebitda_ev_yield",
)


VALUE_SPECS = (
    ValueSpec("pbr_btm", "PBR (B/M)", ("value_btm",), "positive book equity / market cap"),
    ValueSpec(
        "per_earnings",
        "PER (E/P)",
        ("value_earnings_yield",),
        "positive net income / market cap",
    ),
    ValueSpec(
        "pfcf",
        "P/FCF (FCF/P)",
        ("value_fcf_yield",),
        "positive (CFO - CAPEX) / market cap",
    ),
    ValueSpec(
        "psr",
        "PSR (Sales/P)",
        ("value_sales_yield",),
        "positive revenue / market cap",
    ),
    ValueSpec(
        "pcr",
        "PCR (CFO/P)",
        ("value_cfo_yield",),
        "positive operating cash flow / market cap",
    ),
    ValueSpec(
        "ev_ebitda",
        "EV/EBITDA (EBITDA/EV)",
        ("value_ebitda_ev_yield",),
        "positive EBITDA / positive enterprise value",
    ),
    ValueSpec(
        "ev_ebit",
        "EV/EBIT (EBIT/EV)",
        ("value_ebit_ev_yield",),
        "positive operating income / positive enterprise value",
    ),
    ValueSpec(
        "por",
        "POR (Operating income/P)",
        ("value_operating_income_yield",),
        "positive operating income / market cap",
    ),
    ValueSpec(
        "pgpr",
        "PGPR (Gross profit/P)",
        ("value_gross_profit_yield",),
        "positive gross profit / market cap",
    ),
    ValueSpec(
        "prr_rnd",
        "PRR (R&D/P)",
        ("value_rnd_yield",),
        "positive research and development expense / market cap; requested RPR interpreted as PRR",
    ),
    ValueSpec(
        "retained_earnings",
        "Retained earnings/P",
        ("value_retained_earnings_yield",),
        "positive retained earnings / market cap; alternate literal RPR sensitivity",
    ),
    ValueSpec(
        "par",
        "PAR (Assets/P)",
        ("value_assets_yield",),
        "positive total assets / market cap",
    ),
    ValueSpec(
        "ncav",
        "NCAV/P",
        ("value_ncav_yield",),
        "positive current assets less total liabilities / market cap",
    ),
    ValueSpec(
        "core_composite",
        "Core value composite",
        ("value_core_composite",),
        "mean same-date rank-normal score of PBR, PER, P/FCF, PSR, PCR, and EV/EBITDA on complete cases",
    ),
    ValueSpec(
        "core_multivariate",
        "Core value multivariate",
        CORE_VALUE_COLUMNS,
        "simultaneous controls for PBR, PER, P/FCF, PSR, PCR, and EV/EBITDA on complete cases",
    ),
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(REPOSITORY).as_posix(): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _max_abs_signed(frame: pd.DataFrame, account_ids: Sequence[str]) -> float | None:
    values = pd.to_numeric(
        frame.loc[frame["canonical_account_id"].isin(account_ids), "normalized_amount"],
        errors="coerce",
    ).dropna()
    if values.empty:
        return None
    return float(values.iloc[np.abs(values.to_numpy(dtype=float)).argmax()])


def _sum_unique_abs(frame: pd.DataFrame, account_ids: Sequence[str]) -> float | None:
    values = pd.to_numeric(
        frame.loc[frame["canonical_account_id"].isin(account_ids), "normalized_amount"],
        errors="coerce",
    ).dropna()
    unique = {abs(float(value)) for value in values if float(value) != 0.0}
    return float(sum(unique)) if unique else None


def extract_value_fundamentals(frame: pd.DataFrame) -> dict[str, float | None]:
    required = {"canonical_account_id", "normalized_amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"normalized snapshot missing columns: {sorted(missing)}")
    capex_frame = frame.copy()
    capex_ids = ["CAPEX_PPE", "CAPEX_INTANG"]
    if "cash_direction" in capex_frame:
        capex_rows = capex_frame["canonical_account_id"].isin(capex_ids)
        direction = capex_frame["cash_direction"].astype(str).str.casefold()
        if (capex_rows & direction.eq("outflow")).any():
            capex_frame = capex_frame[~capex_rows | direction.eq("outflow")]
    capex = _sum_unique_abs(capex_frame, capex_ids)
    dna = _max_abs_signed(frame, ["DNA_IS"])
    if dna is None:
        depreciation = _max_abs_signed(frame, ["DEPRECIATION_EXPENSE"])
        amortization = _max_abs_signed(frame, ["AMORTIZATION"])
        if depreciation is not None or amortization is not None:
            dna = abs(depreciation or 0.0) + abs(amortization or 0.0)
    return {
        "fund_revenue": _max_abs_signed(frame, ["REVENUE"]),
        "fund_net_income": _max_abs_signed(frame, ["NET_INCOME"]),
        "fund_cfo": _max_abs_signed(frame, ["CFO"]),
        "fund_capex": capex,
        "fund_ebit": _max_abs_signed(frame, ["OPERATING_INCOME"]),
        "fund_dna": abs(dna) if dna is not None else None,
        "fund_gross_profit": _max_abs_signed(frame, ["GROSS_PROFIT"]),
        "fund_rnd": _max_abs_signed(frame, ["RND"]),
        "fund_retained_earnings": _max_abs_signed(frame, ["RETAINED_EARNINGS"]),
        "fund_total_assets": _max_abs_signed(frame, ["TOTAL_ASSETS"]),
        "fund_total_equity": _max_abs_signed(frame, ["TOTAL_EQUITY"]),
        "fund_current_assets": _max_abs_signed(frame, ["CURRENT_ASSETS"]),
        "fund_total_liabilities": _max_abs_signed(frame, ["TOTAL_LIABILITIES"]),
        "fund_cash": _max_abs_signed(frame, ["CASH_AND_EQUIVALENTS"]),
        "fund_debt": _sum_unique_abs(
            frame,
            ["SHORT_TERM_DEBT", "LONG_TERM_DEBT", "LEASE_LIABILITY"],
        ),
    }


def _positive_yield(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    top = pd.to_numeric(numerator, errors="coerce")
    bottom = pd.to_numeric(denominator, errors="coerce")
    result = top / bottom
    return result.where((top > 0) & (bottom > 0) & np.isfinite(result))


def enrich_value_metrics(signals: pd.DataFrame, snapshot_root: Path) -> pd.DataFrame:
    result = signals.copy()
    result["ticker"] = result["ticker"].astype(str).str.zfill(6)
    result["latest_fiscal_year_int"] = pd.to_numeric(
        result["latest_fiscal_year"], errors="coerce"
    ).astype("Int64")
    keys = result.dropna(subset=["latest_fiscal_year_int"])[
        ["ticker", "latest_fiscal_year_int"]
    ].drop_duplicates()
    fundamental_rows: list[dict[str, Any]] = []
    for index, row in enumerate(keys.itertuples(index=False), start=1):
        ticker = str(row.ticker).zfill(6)
        fiscal_year = int(row.latest_fiscal_year_int)
        path = snapshot_root / f"kr_normalized_{ticker}_{fiscal_year}.12.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, low_memory=False)
        fundamental_rows.append(
            {
                "ticker": ticker,
                "latest_fiscal_year_int": fiscal_year,
                "fund_snapshot_path": str(path),
                **extract_value_fundamentals(frame),
            }
        )
        if index % 250 == 0:
            print(f"loaded value fundamentals: {index}/{len(keys)}", flush=True)
    fundamentals = pd.DataFrame(fundamental_rows)
    result = result.merge(
        fundamentals,
        on=["ticker", "latest_fiscal_year_int"],
        how="left",
        validate="many_to_one",
    )
    market_cap = pd.to_numeric(result["market_cap"], errors="coerce")
    result["value_earnings_yield"] = _positive_yield(result["fund_net_income"], market_cap)
    fcf = pd.to_numeric(result["fund_cfo"], errors="coerce") - pd.to_numeric(
        result["fund_capex"], errors="coerce"
    )
    result["fund_fcf"] = fcf
    result["value_fcf_yield"] = _positive_yield(fcf, market_cap)
    result["value_sales_yield"] = _positive_yield(result["fund_revenue"], market_cap)
    result["value_cfo_yield"] = _positive_yield(result["fund_cfo"], market_cap)
    debt = pd.to_numeric(result["fund_debt"], errors="coerce").fillna(0.0)
    cash = pd.to_numeric(result["fund_cash"], errors="coerce")
    result["fund_enterprise_value"] = market_cap + debt - cash
    result["fund_ebitda"] = pd.to_numeric(result["fund_ebit"], errors="coerce") + pd.to_numeric(
        result["fund_dna"], errors="coerce"
    )
    result["value_ebitda_ev_yield"] = _positive_yield(
        result["fund_ebitda"], result["fund_enterprise_value"]
    )
    result["value_ebit_ev_yield"] = _positive_yield(
        result["fund_ebit"], result["fund_enterprise_value"]
    )
    result["value_operating_income_yield"] = _positive_yield(result["fund_ebit"], market_cap)
    result["value_gross_profit_yield"] = _positive_yield(
        result["fund_gross_profit"], market_cap
    )
    result["value_rnd_yield"] = _positive_yield(result["fund_rnd"], market_cap)
    result["value_retained_earnings_yield"] = _positive_yield(
        result["fund_retained_earnings"], market_cap
    )
    result["value_assets_yield"] = _positive_yield(result["fund_total_assets"], market_cap)
    ncav = pd.to_numeric(result["fund_current_assets"], errors="coerce") - pd.to_numeric(
        result["fund_total_liabilities"], errors="coerce"
    )
    result["fund_ncav"] = ncav
    result["value_ncav_yield"] = _positive_yield(ncav, market_cap)
    result["value_core_composite"] = np.nan
    for _signal_date, indices in result.groupby("signal_date").groups.items():
        ranked = pd.DataFrame(
            {column: rank_normal_score(result.loc[indices, column]) for column in CORE_VALUE_COLUMNS},
            index=indices,
        )
        complete = ranked.notna().all(axis=1)
        result.loc[complete.index[complete], "value_core_composite"] = ranked.loc[
            complete
        ].mean(axis=1)
    return result


def neutralized_column(driver: str, spec: ValueSpec) -> str:
    return f"{driver}_vn__{spec.key}"


def add_neutralized_signals(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for _signal_date, indices in result.groupby("signal_date").groups.items():
        group = result.loc[indices]
        for driver in DRIVERS:
            target = driver
            eligible_group = group.copy()
            eligible_group.loc[~eligible_group["status"].eq("ELIGIBLE"), target] = np.nan
            for spec in VALUE_SPECS:
                result.loc[indices, neutralized_column(driver, spec)] = residualize_cross_section(
                    eligible_group,
                    target=target,
                    numeric_controls=spec.controls,
                )
    return result


def _correlation(left: pd.Series, right: pd.Series, *, method: str = "pearson") -> float:
    pair = pd.concat(
        [pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce")], axis=1
    ).dropna()
    if len(pair) < 3 or pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def neutralization_exposure(
    group: pd.DataFrame,
    *,
    target: str,
    residual: str,
    controls: Sequence[str],
) -> dict[str, float | int]:
    work = group[[target, residual, *controls]].apply(pd.to_numeric, errors="coerce")
    y = rank_normal_score(work[target])
    z = pd.DataFrame({column: rank_normal_score(work[column]) for column in controls})
    valid_mask = pd.concat([y.rename("_target_rank"), work[[residual]], z], axis=1).notna().all(axis=1)
    if int(valid_mask.sum()) <= len(controls) + 2:
        return {"n": int(valid_mask.sum())}
    valid = work.loc[valid_mask]
    y = y.loc[valid_mask]
    z = z.loc[valid_mask]
    resid = work.loc[valid_mask, residual]
    x = np.column_stack([np.ones(len(valid)), z.to_numpy(dtype=float)])
    y_array = y.to_numpy(dtype=float)
    fitted = x @ np.linalg.lstsq(x, y_array, rcond=None)[0]
    total = float(np.sum((y_array - y_array.mean()) ** 2))
    unexplained = float(np.sum((y_array - fitted) ** 2))
    pre = [_correlation(y, z[column]) for column in controls]
    post = [_correlation(resid, z[column]) for column in controls]
    return {
        "n": len(valid),
        "value_exposure_r2": 1.0 - unexplained / total if total > 0 else float("nan"),
        "residual_variance_share": unexplained / total if total > 0 else float("nan"),
        "rank_retention": _correlation(valid[target], resid, method="spearman"),
        "mean_abs_pre_control_corr": float(np.nanmean(np.abs(pre))),
        "max_abs_pre_control_corr": float(np.nanmax(np.abs(pre))),
        "mean_abs_post_control_corr": float(np.nanmean(np.abs(post))),
        "max_abs_post_control_corr": float(np.nanmax(np.abs(post))),
    }


def portfolio_metrics(sample: pd.DataFrame, signal: str) -> dict[str, float | int]:
    work = sample[[signal, "forward_77d_return"]].apply(pd.to_numeric, errors="coerce").dropna()
    result: dict[str, float | int] = {
        "n": len(work),
        "ic": spearman_ic(work, signal, "forward_77d_return"),
    }
    if len(work) < 10:
        result.update({"q1": np.nan, "q5": np.nan, "q5_minus_q1": np.nan, "top_excess": np.nan})
        return result
    work = work.copy()
    work["quintile"] = pd.qcut(
        work[signal].rank(method="first"),
        5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)
    means = work.groupby("quintile")["forward_77d_return"].mean()
    common = float(work["forward_77d_return"].mean())
    q1 = float(means.get(1, np.nan))
    q5 = float(means.get(5, np.nan))
    result.update(
        {
            "q1": q1,
            "q5": q5,
            "q5_minus_q1": q5 - q1,
            "top_excess": q5 - common,
        }
    )
    return result


def evaluate_monthly(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    performance_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    for signal_date, group in frame.groupby("signal_date", sort=True):
        for driver in DRIVERS:
            target = driver
            for spec in VALUE_SPECS:
                residual = neutralized_column(driver, spec)
                sample = group[[target, residual, "forward_77d_return"]].dropna()
                raw = portfolio_metrics(sample, target)
                neutral = portfolio_metrics(sample, residual)
                performance_rows.append(
                    {
                        "signal_date": signal_date,
                        "driver": driver,
                        "neutralizer": spec.key,
                        "neutralizer_label": spec.label,
                        "n": int(raw["n"]),
                        **{f"raw_{key}": value for key, value in raw.items() if key != "n"},
                        **{f"neutral_{key}": value for key, value in neutral.items() if key != "n"},
                        **{
                            f"delta_{key}": float(neutral[key]) - float(raw[key])
                            for key in ("ic", "q5_minus_q1", "top_excess")
                        },
                    }
                )
                exposure_rows.append(
                    {
                        "signal_date": signal_date,
                        "driver": driver,
                        "neutralizer": spec.key,
                        "neutralizer_label": spec.label,
                        **neutralization_exposure(
                            group,
                            target=target,
                            residual=residual,
                            controls=spec.controls,
                        ),
                    }
                )
    return pd.DataFrame(performance_rows), pd.DataFrame(exposure_rows)


def inference(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if pd.notna(value) and math.isfinite(float(value))]
    nw = newey_west_mean(clean, lag=HAC_LAG)
    bootstrap = moving_block_bootstrap_mean(
        clean,
        block_length=BLOCK_LENGTH,
        repetitions=BOOTSTRAP_REPETITIONS,
        seed=BOOTSTRAP_SEED,
    )
    t_value = float(nw["t"])
    p_value = float(2 * stats.norm.sf(abs(t_value))) if math.isfinite(t_value) else float("nan")
    return {"newey_west": nw, "moving_block_bootstrap": bootstrap, "hac_normal_p": p_value}


def _safe_attenuation(raw: float, neutral: float) -> float:
    return (raw - neutral) / raw if raw >= 0.01 else float("nan")


def summarize_driver_results(
    monthly: pd.DataFrame,
    exposures: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for (driver, neutralizer), group in monthly.groupby(["driver", "neutralizer"], sort=False):
        exp = exposures[
            (exposures["driver"] == driver) & (exposures["neutralizer"] == neutralizer)
        ]
        stats_map = {
            column: inference(pd.to_numeric(group[column], errors="coerce").dropna().tolist())
            for column in (
                "raw_ic",
                "neutral_ic",
                "delta_ic",
                "raw_q5_minus_q1",
                "neutral_q5_minus_q1",
                "delta_q5_minus_q1",
                "raw_top_excess",
                "neutral_top_excess",
                "delta_top_excess",
            )
        }
        raw_ic = float(stats_map["raw_ic"]["newey_west"]["mean"])
        neutral_ic = float(stats_map["neutral_ic"]["newey_west"]["mean"])
        raw_spread = float(stats_map["raw_q5_minus_q1"]["newey_west"]["mean"])
        neutral_spread = float(stats_map["neutral_q5_minus_q1"]["newey_west"]["mean"])
        row = {
            "driver": driver,
            "neutralizer": neutralizer,
            "neutralizer_label": str(group["neutralizer_label"].iloc[0]),
            "quarters": int(group["signal_date"].nunique()),
            "average_n": float(pd.to_numeric(group["n"], errors="coerce").mean()),
            "minimum_n": int(pd.to_numeric(group["n"], errors="coerce").min()),
            "raw_ic_mean": raw_ic,
            "neutral_ic_mean": neutral_ic,
            "delta_ic_mean": float(stats_map["delta_ic"]["newey_west"]["mean"]),
            "neutral_ic_hac_t": float(stats_map["neutral_ic"]["newey_west"]["t"]),
            "delta_ic_hac_t": float(stats_map["delta_ic"]["newey_west"]["t"]),
            "delta_ic_boot_ci_low": float(
                stats_map["delta_ic"]["moving_block_bootstrap"]["ci_low"]
            ),
            "delta_ic_boot_ci_high": float(
                stats_map["delta_ic"]["moving_block_bootstrap"]["ci_high"]
            ),
            "ic_attenuation": _safe_attenuation(raw_ic, neutral_ic),
            "raw_q5_minus_q1_mean": raw_spread,
            "neutral_q5_minus_q1_mean": neutral_spread,
            "delta_q5_minus_q1_mean": float(
                stats_map["delta_q5_minus_q1"]["newey_west"]["mean"]
            ),
            "neutral_q5_minus_q1_hac_t": float(
                stats_map["neutral_q5_minus_q1"]["newey_west"]["t"]
            ),
            "delta_q5_minus_q1_hac_t": float(
                stats_map["delta_q5_minus_q1"]["newey_west"]["t"]
            ),
            "spread_attenuation": _safe_attenuation(raw_spread, neutral_spread),
            "mean_value_exposure_r2": float(
                pd.to_numeric(exp["value_exposure_r2"], errors="coerce").mean()
            ),
            "mean_rank_retention": float(pd.to_numeric(exp["rank_retention"], errors="coerce").mean()),
            "mean_abs_pre_control_corr": float(
                pd.to_numeric(exp["mean_abs_pre_control_corr"], errors="coerce").mean()
            ),
            "mean_abs_post_control_corr": float(
                pd.to_numeric(exp["mean_abs_post_control_corr"], errors="coerce").mean()
            ),
            "max_abs_post_control_corr": float(
                pd.to_numeric(exp["max_abs_post_control_corr"], errors="coerce").max()
            ),
        }
        rows.append(row)
        detail[f"{driver}::{neutralizer}"] = stats_map
    return pd.DataFrame(rows), detail


def summarize_aggregate_results(
    monthly: pd.DataFrame,
    exposures: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metric_columns = [
        "n",
        "raw_ic",
        "neutral_ic",
        "delta_ic",
        "raw_q5_minus_q1",
        "neutral_q5_minus_q1",
        "delta_q5_minus_q1",
        "raw_top_excess",
        "neutral_top_excess",
        "delta_top_excess",
    ]
    date_average = (
        monthly.groupby(["neutralizer", "neutralizer_label", "signal_date"], as_index=False)[
            metric_columns
        ].mean()
    )
    rows: list[dict[str, Any]] = []
    detail: dict[str, Any] = {}
    for neutralizer, group in date_average.groupby("neutralizer", sort=False):
        exp = exposures[exposures["neutralizer"] == neutralizer]
        stats_map = {
            column: inference(pd.to_numeric(group[column], errors="coerce").dropna().tolist())
            for column in metric_columns
            if column != "n"
        }
        raw_ic = float(stats_map["raw_ic"]["newey_west"]["mean"])
        neutral_ic = float(stats_map["neutral_ic"]["newey_west"]["mean"])
        raw_spread = float(stats_map["raw_q5_minus_q1"]["newey_west"]["mean"])
        neutral_spread = float(stats_map["neutral_q5_minus_q1"]["newey_west"]["mean"])
        rows.append(
            {
                "neutralizer": neutralizer,
                "neutralizer_label": str(group["neutralizer_label"].iloc[0]),
                "quarters": int(group["signal_date"].nunique()),
                "average_n": float(group["n"].mean()),
                "raw_ic_mean": raw_ic,
                "neutral_ic_mean": neutral_ic,
                "delta_ic_mean": float(stats_map["delta_ic"]["newey_west"]["mean"]),
                "neutral_ic_hac_t": float(stats_map["neutral_ic"]["newey_west"]["t"]),
                "delta_ic_hac_t": float(stats_map["delta_ic"]["newey_west"]["t"]),
                "delta_ic_boot_ci_low": float(
                    stats_map["delta_ic"]["moving_block_bootstrap"]["ci_low"]
                ),
                "delta_ic_boot_ci_high": float(
                    stats_map["delta_ic"]["moving_block_bootstrap"]["ci_high"]
                ),
                "ic_attenuation": _safe_attenuation(raw_ic, neutral_ic),
                "raw_q5_minus_q1_mean": raw_spread,
                "neutral_q5_minus_q1_mean": neutral_spread,
                "delta_q5_minus_q1_mean": float(
                    stats_map["delta_q5_minus_q1"]["newey_west"]["mean"]
                ),
                "neutral_q5_minus_q1_hac_t": float(
                    stats_map["neutral_q5_minus_q1"]["newey_west"]["t"]
                ),
                "delta_q5_minus_q1_hac_t": float(
                    stats_map["delta_q5_minus_q1"]["newey_west"]["t"]
                ),
                "spread_attenuation": _safe_attenuation(raw_spread, neutral_spread),
                "mean_value_exposure_r2": float(
                    pd.to_numeric(exp["value_exposure_r2"], errors="coerce").mean()
                ),
                "mean_rank_retention": float(
                    pd.to_numeric(exp["rank_retention"], errors="coerce").mean()
                ),
                "mean_abs_pre_control_corr": float(
                    pd.to_numeric(exp["mean_abs_pre_control_corr"], errors="coerce").mean()
                ),
                "mean_abs_post_control_corr": float(
                    pd.to_numeric(exp["mean_abs_post_control_corr"], errors="coerce").mean()
                ),
                "max_abs_post_control_corr": float(
                    pd.to_numeric(exp["max_abs_post_control_corr"], errors="coerce").max()
                ),
            }
        )
        detail[neutralizer] = stats_map
    return pd.DataFrame(rows), detail


def metric_coverage(frame: pd.DataFrame) -> pd.DataFrame:
    columns = sorted({control for spec in VALUE_SPECS for control in spec.controls})
    rows = []
    for column in columns:
        valid = pd.to_numeric(frame[column], errors="coerce").notna()
        by_date = frame.assign(_valid=valid).groupby("signal_date")["_valid"].mean()
        rows.append(
            {
                "metric_column": column,
                "valid_rows": int(valid.sum()),
                "total_rows": len(frame),
                "coverage": float(valid.mean()),
                "minimum_quarterly_coverage": float(by_date.min()),
                "maximum_quarterly_coverage": float(by_date.max()),
            }
        )
    return pd.DataFrame(rows)


def _percent(value: float) -> str:
    return "NA" if not math.isfinite(value) else f"{100.0 * value:.1f}%"


def write_report(
    output: Path,
    aggregate: pd.DataFrame,
    drivers: pd.DataFrame,
    coverage: pd.DataFrame,
) -> None:
    display = aggregate.copy().sort_values("neutral_ic_mean", ascending=False)
    table_lines = [
        "| Value neutralizer | avg N | raw IC | neutral IC | ΔIC | IC attenuation | raw Q5-Q1 | neutral Q5-Q1 | Δspread | value R² | rank retained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in display.itertuples(index=False):
        table_lines.append(
            "| {label} | {n:.1f} | {raw_ic:.4f} | {neutral_ic:.4f} | {delta_ic:+.4f} | {ic_att} | "
            "{raw_spread:.2%} | {neutral_spread:.2%} | {delta_spread:+.2%} | {r2:.1%} | {rank:.1%} |".format(
                label=row.neutralizer_label,
                n=row.average_n,
                raw_ic=row.raw_ic_mean,
                neutral_ic=row.neutral_ic_mean,
                delta_ic=row.delta_ic_mean,
                ic_att=_percent(row.ic_attenuation),
                raw_spread=row.raw_q5_minus_q1_mean,
                neutral_spread=row.neutral_q5_minus_q1_mean,
                delta_spread=row.delta_q5_minus_q1_mean,
                r2=row.mean_value_exposure_r2,
                rank=row.mean_rank_retention,
            )
        )
    pbr = aggregate.loc[aggregate["neutralizer"] == "pbr_btm"].iloc[0]
    core = aggregate.loc[aggregate["neutralizer"] == "core_multivariate"].iloc[0]
    requested = {
        key: aggregate.loc[aggregate["neutralizer"] == key].iloc[0]
        for key in ("per_earnings", "pfcf", "psr", "pcr", "ev_ebitda", "prr_rnd")
    }
    maximum_residual_t = float(aggregate["neutral_ic_hac_t"].abs().max())
    strongest = display.iloc[0]
    weakest = display.iloc[-1]
    driver_pbr = drivers[drivers["neutralizer"] == "pbr_btm"].sort_values("driver")
    driver_lines = [
        "| Signal | raw IC | B/M-neutral IC | ΔIC | raw Q5-Q1 | B/M-neutral Q5-Q1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in driver_pbr.itertuples(index=False):
        driver_lines.append(
            f"| {row.driver} | {row.raw_ic_mean:.4f} | {row.neutral_ic_mean:.4f} | "
            f"{row.delta_ic_mean:+.4f} | {row.raw_q5_minus_q1_mean:.2%} | "
            f"{row.neutral_q5_minus_q1_mean:.2%} |"
        )
    coverage_map = dict(zip(coverage["metric_column"], coverage["coverage"], strict=False))
    text = f"""# Expected Gap (Cheap) multi-value neutralization sensitivity

## 결론

- 이 결과는 2020-03~2025-09의 기존 v7.1 PIT Cheap/Expectation Gap 신호·77일 수익률을 재사용한 **사후(ex-post) 민감도 분석**입니다. 새 OOS 홀드아웃이나 프로모션 검정이 아닙니다.
- 기존 PBR(B/M) 중립화 결과를 재현하면 IC는 `{pbr.raw_ic_mean:.4f}`에서 `{pbr.neutral_ic_mean:.4f}`로, Q5-Q1은 `{pbr.raw_q5_minus_q1_mean:.2%}`에서 `{pbr.neutral_q5_minus_q1_mean:.2%}`로 바뀝니다.
- 요청 지표별 IC 감소율은 PER `{requested['per_earnings'].ic_attenuation:.1%}`, P/FCF `{requested['pfcf'].ic_attenuation:.1%}`, PSR `{requested['psr'].ic_attenuation:.1%}`, PCR `{requested['pcr'].ic_attenuation:.1%}`, EV/EBITDA `{requested['ev_ebitda'].ic_attenuation:.1%}`, PRR(R&D/P) `{requested['prr_rnd'].ic_attenuation:.1%}`입니다. 100% 초과는 잔존 IC의 부호가 음(-)으로 뒤집혔다는 뜻이고, 음(-)의 감소율은 중립화 후 IC가 오히려 소폭 커졌다는 뜻입니다.
- 어느 사양도 잔존 IC가 통계적으로 유의하지 않았습니다(15개 사양 중 최대 `|HAC t|={maximum_residual_t:.2f}`). PCR·P/FCF·R&D/P는 PBR보다 IC를 덜 제거했지만 독립 알파로 볼 유의성은 없습니다.
- 6개 핵심 Value 지표를 동시에 통제하면 IC는 `{core.raw_ic_mean:.4f}`에서 `{core.neutral_ic_mean:.4f}`, Q5-Q1은 `{core.raw_q5_minus_q1_mean:.2%}`에서 `{core.neutral_q5_minus_q1_mean:.2%}`가 됐습니다. 평균 신호 순위 보존율은 `{core.mean_rank_retention:.1%}`입니다.
- 중립화 후 평균 IC가 가장 높은 단일/복합 사양은 **{strongest.neutralizer_label}** (`{strongest.neutral_ic_mean:.4f}`), 가장 낮은 사양은 **{weakest.neutralizer_label}** (`{weakest.neutral_ic_mean:.4f}`)입니다. 이는 우열 추천이 아니라 Value 중복 노출의 위치를 보여주는 진단입니다.
- 모든 회귀 중립 신호는 사용한 rank-normal Value 통제변수와 수치상 직교합니다(전체 분기 중 최대 절대 사후 상관 `{aggregate.max_abs_post_control_corr.max():.3e}`). 따라서 표의 Value R²는 제거한 신호 분산, rank retained는 제거 후 남은 원신호 순위 정보를 뜻합니다.

## 지표별 비교

{chr(10).join(table_lines)}

`raw`는 각 Value 지표가 관측되는 동일 표본에서의 원 Expected Gap입니다. 따라서 지표별 raw 값 차이는 커버리지 차이이며, `Δ`가 순수한 같은-표본 중립화 변화입니다. IC/Q5-Q1은 분기별 값의 평균이고, 통계 요약에는 Newey-West lag 1 및 4분기 moving-block bootstrap이 들어 있습니다.

## 기존 PBR 결과 재현

{chr(10).join(driver_lines)}

## Value 정의와 커버리지

- 모든 지표는 **높을수록 cheap**이 되도록 배수의 역수(yield) 방향으로 통일했습니다. 음(-) 또는 0인 이익·현금흐름·EBITDA는 해당 배수에서 제외했습니다.
- PER: E/P, P/FCF: (CFO-CAPEX)/P, PSR: Sales/P, PCR: CFO/P, EV/EBITDA: EBITDA/EV입니다.
- 요청의 `RPR`은 표준적인 Value 배수 명칭이 불명확해, 일반적인 퀀트 명칭 `PRR=Price/R&D`의 역수 R&D/P와 문자 그대로의 대안인 Retained earnings/P를 모두 계산했습니다.
- 주요 전체 행 커버리지: E/P `{coverage_map['value_earnings_yield']:.1%}`, FCF/P `{coverage_map['value_fcf_yield']:.1%}`, Sales/P `{coverage_map['value_sales_yield']:.1%}`, CFO/P `{coverage_map['value_cfo_yield']:.1%}`, EBITDA/EV `{coverage_map['value_ebitda_ev_yield']:.1%}`, R&D/P `{coverage_map['value_rnd_yield']:.1%}`.
- Core composite와 Core multivariate는 PBR, PER, P/FCF, PSR, PCR, EV/EBITDA가 모두 있는 complete-case 표본입니다.

## 해석 제한

- 분기 수가 23개뿐입니다. HAC와 block bootstrap을 써도 검정력은 제한적입니다.
- 재무 지표는 기존 v7.1과 같은 시점가용 연차 DART snapshot을 사용하며, 분기/TTM 지표는 아닙니다.
- 지표별로 양(+)의 분모만 남기므로 표본이 달라집니다. 그래서 지표 간 raw 수준을 직접 비교하지 않고 각 행의 paired Δ를 봐야 합니다.
- Value proxy를 선택한 시점에 이미 수익률을 확인했으므로 결과를 새 알파 발견이나 OOS 증거로 해석하면 안 됩니다.

## 산출물

- `results/aggregate-comparison.csv`: Value 중립화 사양별 비교
- `results/signal-comparison.csv`: Cheap 신호 상세 비교
- `results/quarterly-paired-results.csv`: 분기별 같은-표본 raw/neutral 결과
- `results/quarterly-neutralization-exposure.csv`: 제거된 Value 노출과 순위 보존
- `results/value-metric-coverage.csv`: Value 지표 커버리지
- `results/value-enriched-signals.csv`: 재무 Value와 모든 neutral signal
- `results/statistical-summary.json`: HAC 및 bootstrap 전체 통계
"""
    (output / "FINAL-REPORT.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare v7.1 Cheap/Expected Gap neutralization across multiple value metrics."
    )
    parser.add_argument("--base-signals", type=Path, default=BASE_SIGNALS)
    parser.add_argument("--snapshot-root", type=Path, default=ARCANA_SNAPSHOTS)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()
    base_signals = args.base_signals.resolve()
    snapshot_root = args.snapshot_root.resolve()
    output = args.output.resolve()
    if (output / "FINAL-RESULT.json").exists():
        raise FileExistsError(f"completed sensitivity result is immutable: {output}")
    (output / "results").mkdir(parents=True, exist_ok=True)
    if not base_signals.exists():
        raise FileNotFoundError(base_signals)
    if not snapshot_root.exists():
        raise FileNotFoundError(snapshot_root)
    base_hashes_before = file_hashes(BASE_ROOT)
    signals = pd.read_csv(base_signals, dtype={"ticker": str}, low_memory=False)
    exits = pd.to_datetime(signals["exit_date"], errors="coerce").dropna()
    signal_dates = sorted(signals["signal_date"].astype(str).unique().tolist())
    if (
        exits.empty
        or len(signal_dates) != 23
        or signal_dates[0] != "2020-03-31"
        or signal_dates[-1] != "2025-09-30"
    ):
        raise ValueError("source result is not the frozen 2020-2025 v7.1 Cheap validation")
    enriched = enrich_value_metrics(signals, snapshot_root)
    enriched = add_neutralized_signals(enriched)
    reproduced = pd.to_numeric(enriched["cheap_vn__pbr_btm"], errors="coerce")
    frozen = pd.to_numeric(enriched["cheap_resid_value"], errors="coerce")
    reproduction_pair = pd.concat(
        [reproduced.rename("reproduced"), frozen.rename("frozen")], axis=1
    ).dropna()
    reproduction_max_abs_diff = float(
        np.max(np.abs(reproduction_pair["reproduced"] - reproduction_pair["frozen"]))
    )
    if (
        len(reproduction_pair) != int(frozen.notna().sum())
        or len(reproduction_pair) != int(reproduced.notna().sum())
        or reproduction_max_abs_diff > 1e-12
    ):
        raise RuntimeError("PBR neutralization did not exactly reproduce frozen v7.1")
    monthly, exposures = evaluate_monthly(enriched)
    driver_comparison, driver_detail = summarize_driver_results(monthly, exposures)
    aggregate_comparison, aggregate_detail = summarize_aggregate_results(monthly, exposures)
    coverage = metric_coverage(enriched)
    enriched.to_csv(
        output / "results/value-enriched-signals.csv", index=False, encoding="utf-8-sig"
    )
    monthly.to_csv(
        output / "results/quarterly-paired-results.csv", index=False, encoding="utf-8-sig"
    )
    exposures.to_csv(
        output / "results/quarterly-neutralization-exposure.csv",
        index=False,
        encoding="utf-8-sig",
    )
    driver_comparison.to_csv(
        output / "results/signal-comparison.csv", index=False, encoding="utf-8-sig"
    )
    aggregate_comparison.to_csv(
        output / "results/aggregate-comparison.csv", index=False, encoding="utf-8-sig"
    )
    coverage.to_csv(
        output / "results/value-metric-coverage.csv", index=False, encoding="utf-8-sig"
    )
    write_json(
        output / "results/statistical-summary.json",
        {
            "schema_version": "moatrader-v7.1-multi-value-neutral-sensitivity/1",
            "analysis_grade": "EX_POST_SENSITIVITY_NOT_NEW_HOLDOUT",
            "signal_results": driver_detail,
            "aggregate_results": aggregate_detail,
        },
    )
    input_payload = {
        "schema_version": "moatrader-v7.1-multi-value-neutral-inputs/1",
        "base_signals": str(base_signals),
        "base_signals_sha256": sha256_file(base_signals),
        "arcana_snapshot_root": str(snapshot_root),
        "arcana_metadata": str(ARCANA_METADATA),
        "arcana_metadata_sha256": sha256_file(ARCANA_METADATA) if ARCANA_METADATA.exists() else None,
        "signal_rows": len(signals),
        "signal_dates": signal_dates,
        "exit_date_max": exits.max().date().isoformat(),
        "value_specs": [
            {
                "key": spec.key,
                "label": spec.label,
                "controls": list(spec.controls),
                "definition": spec.definition,
            }
            for spec in VALUE_SPECS
        ],
        "eligibility": "positive numerator and denominator for inverse multiples; complete cases per residualization",
        "comparison": "pairwise matched raw versus neutral within neutralizer/date/Cheap signal",
        "pbr_reproduction": {
            "matched_rows": len(reproduction_pair),
            "max_abs_signal_difference": reproduction_max_abs_diff,
            "exact_within_1e-12": True,
        },
        "inference": {
            "newey_west_lag": HAC_LAG,
            "moving_block_length": BLOCK_LENGTH,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
    }
    write_json(output / "input-manifest.json", input_payload)
    base_hashes_after = file_hashes(BASE_ROOT)
    changed = sorted(
        key
        for key in set(base_hashes_before) | set(base_hashes_after)
        if base_hashes_before.get(key) != base_hashes_after.get(key)
    )
    integrity = {
        "schema_version": "moatrader-v7.1-multi-value-neutral-source-integrity/1",
        "base_v7_1_unchanged": not changed,
        "changed_paths": changed,
        "base_file_count": len(base_hashes_after),
    }
    write_json(output / "source-integrity.json", integrity)
    if changed:
        raise RuntimeError(f"base v7.1 source changed during sensitivity run: {changed}")
    write_report(output, aggregate_comparison, driver_comparison, coverage)
    final = {
        "schema_version": "moatrader-v7.1-multi-value-neutral-final/1",
        "analysis_grade": "EX_POST_SENSITIVITY_NOT_NEW_HOLDOUT",
        "period": [str(signals["signal_date"].min()), str(signals["signal_date"].max())],
        "signal_date_count": int(signals["signal_date"].nunique()),
        "signal_count": len(DRIVERS),
        "neutralizer_count": len(VALUE_SPECS),
        "pbr_reproduction": input_payload["pbr_reproduction"],
        "aggregate_comparison": aggregate_comparison.to_dict("records"),
        "max_abs_post_control_corr": float(aggregate_comparison["max_abs_post_control_corr"].max()),
        "source_integrity": integrity,
        "report": str(output / "FINAL-REPORT.md"),
    }
    write_json(output / "FINAL-RESULT.json", final)
    print(json.dumps(final, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
