from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from moatrader.backtest.universe_corrected import (
    build_historical_universe,
    classify_security,
    extract_arcana_annual_metrics,
    moving_block_bootstrap_mean,
    newey_west_mean,
    residualize_cross_section,
)


def test_security_classification_matches_frozen_order() -> None:
    assert classify_security("테스트스팩3호") == "SPAC"
    assert classify_security("테스트리츠우") == "REIT"
    assert classify_security("테스트3우B") == "PREFERRED"
    assert classify_security("테스트") == "COMMON"


def test_arcana_adapter_uses_only_capex_outflows_and_avoids_combined_nwc_double_count() -> None:
    frame = pd.DataFrame(
        [
            ("REVENUE", 1000, ""),
            ("OPERATING_INCOME", 100, ""),
            ("CAPEX_PPE", 60, "outflow"),
            ("CAPEX_PPE", 10, "inflow"),
            ("CAPEX_INTANG", 20, "outflow"),
            ("DEPRECIATION_EXPENSE", 30, ""),
            ("AMORTIZATION", 5, ""),
            ("CASH_AND_EQUIVALENTS", 50, ""),
            ("SHORT_TERM_DEBT", 40, ""),
            ("LONG_TERM_DEBT", 60, ""),
            ("TRADE_AND_OTHER_RECEIVABLES", 120, ""),
            ("TRADE_RECEIVABLES", 80, ""),
            ("INVENTORIES", 40, ""),
            ("TRADE_AND_OTHER_PAYABLES", 70, ""),
            ("TRADE_PAYABLES", 50, ""),
            ("TOTAL_ASSETS", 2000, ""),
            ("TOTAL_EQUITY", 1000, ""),
            ("CFO", 150, ""),
        ],
        columns=["canonical_account_id", "normalized_amount", "cash_direction"],
    )
    frame["fiscal_year"] = 2024
    metrics = extract_arcana_annual_metrics(frame)
    assert metrics["capex"] == 80
    assert metrics["depreciation"] == 30
    assert metrics["debt"] == 100
    assert metrics["nwc"] == 90


def test_residualization_removes_linear_value_exposure() -> None:
    value = np.linspace(-2, 2, 50)
    frame = pd.DataFrame({"cheap": value * 3 + np.sin(value), "value": value})
    residual = residualize_cross_section(frame, target="cheap", numeric_controls=["value"])
    assert residual.notna().sum() == 50
    assert residual.abs().max() < 1e-10


def test_overlap_robust_statistics_are_deterministic() -> None:
    values = [0.01, 0.02, -0.01, 0.03, 0.04, 0.00]
    hac = newey_west_mean(values, lag=1)
    assert hac["n"] == 6
    assert hac["se"] > 0
    first = moving_block_bootstrap_mean(values, block_length=4, repetitions=100, seed=7)
    second = moving_block_bootstrap_mean(values, block_length=4, repetitions=100, seed=7)
    assert first == second


def test_2025_checkpoint_reproduces_original_universe_exactly() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "data-lake/experiments/historical-validation-v7-2020-2025/prices/source"
    original_path = Path(r"D:\Programming\python_example\MoatPoC\universe.csv")
    if not original_path.exists() or not (source / "marcap-2024.parquet").exists():
        pytest.skip("local frozen reference inputs unavailable")
    marcap = pd.concat(
        [pd.read_parquet(source / f"marcap-{year}.parquet") for year in (2024, 2025)],
        ignore_index=True,
    )
    marcap["Date"] = pd.to_datetime(marcap["Date"])
    build = build_historical_universe(
        marcap[(marcap["Date"] >= "2024-06-01") & (marcap["Date"] <= "2025-08-01")],
        as_of=date(2025, 8, 1),
    )
    original = pd.read_csv(original_path, dtype={"stock_code": str})
    original["stock_code"] = original["stock_code"].str.zfill(6)
    assert len(build.master) == 2759
    assert len(build.eligible) == 1825
    assert build.selected["stock_code"].tolist() == original["stock_code"].tolist()
