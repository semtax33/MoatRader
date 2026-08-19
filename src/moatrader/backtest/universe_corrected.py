from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats


PREFERRED_RE = re.compile(r"(?:우|우B|우C|\d우|\d우B|\d우C)$", re.IGNORECASE)
SPAC_RE = re.compile(r"(스팩|기업인수목적)", re.IGNORECASE)
REIT_RE = re.compile(r"(리츠|REIT)", re.IGNORECASE)
CURRENT_REIT_RE = re.compile(
    r"(?:리츠|REIT)(?:우|우B|우C|\d우|\d우B|\d우C)?$",
    re.IGNORECASE,
)
FINANCE_HINT_RE = re.compile(r"(금융지주|증권|생명|손해보험|화재|은행|캐피탈|저축은행)", re.IGNORECASE)
HOLDING_HINT_RE = re.compile(r"(홀딩스|지주)", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_security(name: object) -> str:
    """Preserve the frozen v7 historical-universe classifier exactly."""
    value = str(name).strip()
    if SPAC_RE.search(value):
        return "SPAC"
    if REIT_RE.search(value):
        return "REIT"
    if PREFERRED_RE.search(value):
        return "PREFERRED"
    return "COMMON"


def classify_current_security(name: object) -> str:
    """Classify a live security without treating company names like 메리츠 as REITs."""
    value = str(name).strip()
    if SPAC_RE.search(value):
        return "SPAC"
    if CURRENT_REIT_RE.search(value):
        return "REIT"
    if PREFERRED_RE.search(value):
        return "PREFERRED"
    return "COMMON"


def _ticker(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.zfill(6)


def normalize_marcap(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Code", "Name", "Close", "Amount", "Marcap", "Stocks", "MarketId", "Date",
        "ChangesRatio",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"marcap frame missing columns: {sorted(missing)}")
    result = frame.copy()
    result["Date"] = pd.to_datetime(result["Date"])
    result["Code"] = _ticker(result["Code"])
    return result


def nearest_trading_date(frame: pd.DataFrame, requested: date | pd.Timestamp) -> pd.Timestamp:
    requested_ts = pd.Timestamp(requested)
    dates = frame.loc[
        frame["MarketId"].isin(["STK", "KSQ"]) & (frame["Date"] <= requested_ts), "Date"
    ]
    if dates.empty:
        raise ValueError(f"no KOSPI/KOSDAQ date at or before {requested_ts.date()}")
    return pd.Timestamp(dates.max())


def assign_size_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    ranked = result["market_cap"].rank(method="first")
    result["size_bucket"] = pd.qcut(
        ranked,
        q=3,
        labels=["SMALL", "MID", "LARGE"],
    )
    return result


def _sample_one_size_bucket(
    group: pd.DataFrame,
    *,
    n: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    selected_indices: list[int] = []
    base = n // 2
    for market in ("KOSPI", "KOSDAQ"):
        candidates = group[group["market"] == market].index.to_numpy()
        rng.shuffle(candidates)
        selected_indices.extend(candidates[: min(base, len(candidates))].tolist())
    remaining = n - len(selected_indices)
    if remaining > 0:
        leftover = group[~group.index.isin(selected_indices)].index.to_numpy()
        rng.shuffle(leftover)
        selected_indices.extend(leftover[:remaining].tolist())
    return group.loc[selected_indices]


def stratified_sample(frame: pd.DataFrame, *, n: int = 150, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    base = n // 3
    for index, bucket in enumerate(("SMALL", "MID", "LARGE")):
        target = base if index < 2 else n - sum(len(piece) for piece in pieces)
        pieces.append(
            _sample_one_size_bucket(
                frame[frame["size_bucket"] == bucket],
                n=target,
                rng=rng,
            )
        )
    result = pd.concat(pieces, ignore_index=True)
    if len(result) < n:
        remaining = frame[~frame["stock_code"].isin(result["stock_code"])].copy()
        indices = remaining.index.to_numpy()
        rng.shuffle(indices)
        result = pd.concat([result, remaining.loc[indices[: n - len(result)]]], ignore_index=True)
    return result


@dataclass(frozen=True)
class UniverseBuild:
    requested_as_of: date
    actual_as_of: date
    age_check_date: date
    adtv_start: date
    master: pd.DataFrame
    eligible: pd.DataFrame
    selected: pd.DataFrame


def build_historical_universe(
    marcap: pd.DataFrame,
    *,
    as_of: date,
    n: int = 150,
    liquidity_quantile: float = 0.25,
    seed: int = 42,
) -> UniverseBuild:
    """Port the originally frozen 2025 universe algorithm to any historical date.

    Only rows observable on or before ``as_of`` are used.  The master ordering is
    intentionally KOSPI then KOSDAQ, with ticker sorting inside each market, because
    the frozen NumPy sample depends on the original DataFrame index.
    """

    frame = normalize_marcap(marcap)
    actual = nearest_trading_date(frame, as_of)
    old_target = actual - pd.Timedelta(days=365)
    age_check = nearest_trading_date(frame, old_target)
    trading_dates = sorted(
        pd.to_datetime(
            frame.loc[
                frame["MarketId"].isin(["STK", "KSQ"]) & (frame["Date"] <= actual),
                "Date",
            ].unique()
        )
    )
    if len(trading_dates) < 60:
        raise ValueError(f"fewer than 60 trading dates available before {actual.date()}")
    last_60 = trading_dates[-60:]

    snapshot = frame[
        (frame["Date"] == actual) & frame["MarketId"].isin(["STK", "KSQ"])
    ].copy()
    previous_codes = set(
        frame.loc[
            (frame["Date"] == age_check) & frame["MarketId"].isin(["STK", "KSQ"]),
            "Code",
        ]
    )
    traded = (
        frame[
            frame["Date"].isin(last_60) & frame["MarketId"].isin(["STK", "KSQ"])
        ]
        .groupby("Code", sort=False)["Amount"]
        .sum()
        .rename("trading_value_60d")
    )
    snapshot = snapshot.merge(traded, left_on="Code", right_index=True, how="left")
    snapshot["trading_value_60d"] = snapshot["trading_value_60d"].fillna(0.0)
    snapshot["market"] = snapshot["MarketId"].map({"STK": "KOSPI", "KSQ": "KOSDAQ"})
    snapshot["market_order"] = snapshot["MarketId"].map({"STK": 0, "KSQ": 1})
    snapshot = snapshot.sort_values(["market_order", "Code"], kind="stable")
    master = pd.DataFrame(
        {
            "stock_code": snapshot["Code"],
            "name": snapshot["Name"].astype(str),
            "market": snapshot["market"],
            "market_cap": pd.to_numeric(snapshot["Marcap"], errors="coerce"),
            "listed_shares": pd.to_numeric(snapshot["Stocks"], errors="coerce"),
            "trading_value_60d": pd.to_numeric(snapshot["trading_value_60d"], errors="coerce"),
        }
    ).reset_index(drop=True)
    master["adtv60"] = master["trading_value_60d"] / 60.0
    master["security_type"] = master["name"].map(classify_security)
    master["finance_hint"] = master["name"].map(lambda x: bool(FINANCE_HINT_RE.search(str(x))))
    master["holding_hint"] = master["name"].map(lambda x: bool(HOLDING_HINT_RE.search(str(x))))
    master["listed_1y_flag"] = master["stock_code"].isin(previous_codes)
    master["age_check_date"] = age_check.strftime("%Y%m%d")

    eligible = master[
        (master["security_type"] == "COMMON")
        & master["market_cap"].notna()
        & (master["market_cap"] > 0)
        & master["listed_1y_flag"]
    ].copy()
    eligible["liquidity_pct"] = eligible["adtv60"].rank(method="average", pct=True)
    eligible = eligible[eligible["liquidity_pct"] >= liquidity_quantile].copy()
    eligible = assign_size_bucket(eligible)
    if len(eligible) < n:
        raise ValueError(f"eligible universe {len(eligible)} is smaller than sample {n}")
    selected = stratified_sample(eligible, n=n, seed=seed)
    for output in (master, eligible, selected):
        output["as_of"] = actual.date().isoformat()
    selected["selection_seed"] = seed
    selected["selection_rule"] = (
        f"COMMON|LISTED_1Y=True|LIQ_PCT>={liquidity_quantile}|SIZE_3_BUCKET|MARKET_BALANCED"
    )
    selected = selected.sort_values(
        ["size_bucket", "market", "market_cap"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)
    return UniverseBuild(
        requested_as_of=as_of,
        actual_as_of=actual.date(),
        age_check_date=age_check.date(),
        adtv_start=pd.Timestamp(last_60[0]).date(),
        master=master,
        eligible=eligible,
        selected=selected,
    )


def compound_percent(values: Iterable[float]) -> float:
    result = 1.0
    used = 0
    for raw in values:
        value = float(raw)
        if math.isfinite(value):
            result *= 1.0 + value / 100.0
            used += 1
    return result - 1.0 if used else float("nan")


def trailing_momentum(frame: pd.DataFrame, *, as_of: date) -> float:
    start = pd.Timestamp(as_of) - pd.Timedelta(days=365)
    end = pd.Timestamp(as_of) - pd.Timedelta(days=30)
    window = frame[(frame["Date"] > start) & (frame["Date"] <= end)]
    return compound_percent(window["ChangesRatio"]) if len(window) >= 120 else float("nan")


def trailing_beta(
    frame: pd.DataFrame,
    market_returns: pd.Series,
    *,
    as_of: date,
) -> float:
    window = frame[frame["Date"] <= pd.Timestamp(as_of)].tail(252).copy()
    if window.empty:
        return float("nan")
    asset = pd.Series(
        pd.to_numeric(window["ChangesRatio"], errors="coerce").to_numpy() / 100.0,
        index=pd.to_datetime(window["Date"]),
    )
    paired = pd.concat([asset.rename("asset"), market_returns.rename("market")], axis=1).dropna()
    if len(paired) < 120 or paired["market"].var(ddof=1) <= 0:
        return float("nan")
    return float(paired.cov().loc["asset", "market"] / paired["market"].var(ddof=1))


def rank_normal_score(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    result = pd.Series(np.nan, index=series.index, dtype=float)
    valid = values.notna()
    if valid.sum() < 2:
        return result
    pct = values[valid].rank(method="average", pct=True)
    clipped = pct.clip(1e-6, 1 - 1e-6)
    result.loc[valid] = stats.norm.ppf(clipped)
    return result


def residualize_cross_section(
    frame: pd.DataFrame,
    *,
    target: str,
    numeric_controls: Sequence[str],
    categorical_controls: Sequence[str] = (),
) -> pd.Series:
    columns = [target, *numeric_controls, *categorical_controls]
    work = frame[columns].copy()
    work[target] = rank_normal_score(work[target])
    for column in numeric_controls:
        work[column] = rank_normal_score(work[column])
    valid = work[[target, *numeric_controls]].notna().all(axis=1)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if valid.sum() <= len(numeric_controls) + 2:
        return result
    x_parts = [np.ones((int(valid.sum()), 1), dtype=float)]
    if numeric_controls:
        x_parts.append(work.loc[valid, list(numeric_controls)].to_numpy(dtype=float))
    if categorical_controls:
        dummies = pd.get_dummies(
            work.loc[valid, list(categorical_controls)].astype(str),
            drop_first=True,
            dtype=float,
        )
        if not dummies.empty:
            x_parts.append(dummies.to_numpy(dtype=float))
    x = np.column_stack(x_parts)
    y = work.loc[valid, target].to_numpy(dtype=float)
    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    result.loc[valid] = y - x @ coefficients
    return result


def spearman_ic(frame: pd.DataFrame, signal: str, returns: str) -> float:
    pair = frame[[signal, returns]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 5 or pair[signal].nunique() < 2 or pair[returns].nunique() < 2:
        return float("nan")
    return float(stats.spearmanr(pair[signal], pair[returns]).statistic)


def newey_west_mean(values: Sequence[float], *, lag: int = 1) -> dict[str, float | int]:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    n = len(array)
    if n < 2:
        return {"n": n, "mean": float("nan"), "se": float("nan"), "t": float("nan"), "lag": lag}
    demeaned = array - array.mean()
    long_run = float(demeaned @ demeaned / n)
    effective_lag = min(lag, n - 1)
    for offset in range(1, effective_lag + 1):
        gamma = float(demeaned[offset:] @ demeaned[:-offset] / n)
        long_run += 2.0 * (1.0 - offset / (effective_lag + 1.0)) * gamma
    variance = max(long_run / n, 0.0)
    se = math.sqrt(variance)
    return {
        "n": n,
        "mean": float(array.mean()),
        "se": se,
        "t": float(array.mean() / se) if se > 0 else float("nan"),
        "lag": effective_lag,
    }


def moving_block_bootstrap_mean(
    values: Sequence[float],
    *,
    block_length: int = 4,
    repetitions: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int]:
    array = np.asarray([value for value in values if math.isfinite(float(value))], dtype=float)
    n = len(array)
    if n == 0:
        return {"n": 0, "mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    block = max(1, min(block_length, n))
    starts = np.arange(0, n - block + 1)
    blocks_needed = math.ceil(n / block)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([array[start : start + block] for start in chosen])[:n]
        means[index] = sample.mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "n": n,
        "mean": float(array.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "block_length": block,
        "repetitions": repetitions,
        "seed": seed,
    }


def forward_return(
    frame: pd.DataFrame,
    *,
    entry_date: date,
    horizon_days: int = 77,
) -> tuple[float, date] | None:
    target = pd.Timestamp(entry_date) + pd.Timedelta(days=horizon_days)
    window = frame[(frame["Date"] > pd.Timestamp(entry_date)) & (frame["Date"] <= target)]
    if window.empty:
        return None
    exit_date = pd.Timestamp(window.iloc[-1]["Date"])
    if (target - exit_date).days > 10:
        return None
    return compound_percent(window["ChangesRatio"]), exit_date.date()


def previous_price_point(frame: pd.DataFrame, *, as_of: date, max_staleness_days: int = 10) -> pd.Series | None:
    eligible = frame[frame["Date"] <= pd.Timestamp(as_of)]
    if eligible.empty:
        return None
    point = eligible.iloc[-1]
    if (pd.Timestamp(as_of) - pd.Timestamp(point["Date"])).days > max_staleness_days:
        return None
    return point


def _max_abs_amount(frame: pd.DataFrame, account_ids: Sequence[str]) -> float | None:
    values = pd.to_numeric(
        frame.loc[frame["canonical_account_id"].isin(account_ids), "normalized_amount"],
        errors="coerce",
    ).dropna()
    if values.empty:
        return None
    return float(values.iloc[np.abs(values.to_numpy()).argmax()])


def _sum_unique_abs_amount(frame: pd.DataFrame, account_ids: Sequence[str]) -> float | None:
    values = pd.to_numeric(
        frame.loc[frame["canonical_account_id"].isin(account_ids), "normalized_amount"],
        errors="coerce",
    ).dropna()
    unique = {abs(float(value)) for value in values if float(value) != 0.0}
    return float(sum(unique)) if unique else None


def extract_arcana_annual_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Adapt Arcana's normalized annual DART snapshot to the frozen DCF schema.

    Aggregation intentionally mirrors ``parse_dart_ifrs_archive``: max-absolute
    single facts, unique absolute CAPEX/debt components, and max receivable/payable
    facts to avoid double-counting combined and disaggregated concepts.
    """

    required = {"canonical_account_id", "normalized_amount"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Arcana normalized snapshot missing columns: {sorted(missing)}")
    revenue = _max_abs_amount(frame, ["REVENUE"])
    ebit = _max_abs_amount(frame, ["OPERATING_INCOME"])
    capex_frame = frame.copy()
    if "cash_direction" in capex_frame:
        capex_rows = capex_frame["canonical_account_id"].isin(["CAPEX_PPE", "CAPEX_INTANG"])
        direction = capex_frame["cash_direction"].astype(str).str.casefold()
        if (capex_rows & direction.eq("outflow")).any():
            capex_frame = capex_frame[~capex_rows | direction.eq("outflow")]
    capex = _sum_unique_abs_amount(capex_frame, ["CAPEX_PPE", "CAPEX_INTANG"])
    depreciation = _max_abs_amount(frame, ["DEPRECIATION_EXPENSE", "AMORTIZATION"])
    cash = _max_abs_amount(frame, ["CASH_AND_EQUIVALENTS"])
    debt = _sum_unique_abs_amount(frame, ["SHORT_TERM_DEBT", "LONG_TERM_DEBT", "LEASE_LIABILITY"])
    receivables = _max_abs_amount(
        frame,
        ["TRADE_AND_OTHER_RECEIVABLES", "TRADE_RECEIVABLES"],
    )
    inventories = _max_abs_amount(frame, ["INVENTORIES"])
    payables = _max_abs_amount(
        frame,
        ["TRADE_AND_OTHER_PAYABLES", "TRADE_PAYABLES"],
    )
    nwc = None
    if any(value is not None for value in (receivables, inventories, payables)):
        nwc = (receivables or 0.0) + (inventories or 0.0) - (payables or 0.0)
    total_assets = _max_abs_amount(frame, ["TOTAL_ASSETS"])
    total_equity = _max_abs_amount(frame, ["TOTAL_EQUITY"])
    cfo = _max_abs_amount(frame, ["CFO"])
    metrics = [revenue, ebit, capex, depreciation, cash, debt, nwc]
    return {
        "fiscal_year": int(pd.to_numeric(frame.get("fiscal_year"), errors="coerce").dropna().max())
        if "fiscal_year" in frame and pd.to_numeric(frame["fiscal_year"], errors="coerce").notna().any()
        else None,
        "revenue": revenue,
        "ebit": ebit,
        "capex": capex,
        "depreciation": depreciation,
        "cash": cash,
        "debt": debt,
        "nwc": nwc,
        "total_assets": total_assets,
        "total_equity": total_equity,
        "cfo": cfo,
        "metric_coverage_count": sum(value is not None for value in metrics),
    }
