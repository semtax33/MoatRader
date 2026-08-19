from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from moatrader.backtest.universe_corrected import rank_normal_score
from scripts.run_v7_1_value_neutral_sensitivity import (
    VALUE_SPECS,
    extract_value_fundamentals,
    neutralized_column,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data-lake/experiments/universe-corrected-residual-v7-1-2020-2025"
OUTPUT = (
    ROOT
    / "data-lake/experiments/expectation-gap-v7-1-multi-value-neutral-sensitivity-2020-2025"
)


def test_extract_value_fundamentals_uses_consistent_positive_expense_conventions() -> None:
    frame = pd.DataFrame(
        [
            ("REVENUE", 1000, ""),
            ("NET_INCOME", 100, ""),
            ("CFO", 150, ""),
            ("CAPEX_PPE", 60, "outflow"),
            ("CAPEX_PPE", 10, "inflow"),
            ("CAPEX_INTANG", 20, "outflow"),
            ("OPERATING_INCOME", 120, ""),
            ("DNA_IS", 30, ""),
            ("GROSS_PROFIT", 400, ""),
            ("RND", 25, ""),
            ("RETAINED_EARNINGS", 200, ""),
            ("TOTAL_ASSETS", 2000, ""),
            ("TOTAL_EQUITY", 1000, ""),
            ("CURRENT_ASSETS", 800, ""),
            ("TOTAL_LIABILITIES", 1000, ""),
            ("CASH_AND_EQUIVALENTS", 50, ""),
            ("SHORT_TERM_DEBT", 40, ""),
            ("LONG_TERM_DEBT", 60, ""),
        ],
        columns=["canonical_account_id", "normalized_amount", "cash_direction"],
    )
    metrics = extract_value_fundamentals(frame)
    assert metrics["fund_net_income"] == 100
    assert metrics["fund_capex"] == 80
    assert metrics["fund_dna"] == 30
    assert metrics["fund_rnd"] == 25
    assert metrics["fund_debt"] == 100


def test_pbr_neutralization_exactly_reproduces_frozen_v7_1_signal() -> None:
    enriched = pd.read_csv(OUTPUT / "results/value-enriched-signals.csv", low_memory=False)
    reproduced = pd.to_numeric(enriched[neutralized_column("cheap", VALUE_SPECS[0])], errors="coerce")
    frozen = pd.to_numeric(enriched["cheap_resid_value"], errors="coerce")
    pair = pd.concat([reproduced, frozen], axis=1).dropna()
    assert len(pair) == 1209
    assert reproduced.notna().sum() == frozen.notna().sum() == 1209
    assert np.max(np.abs(pair.iloc[:, 0] - pair.iloc[:, 1])) < 1e-12


def test_every_neutral_signal_is_orthogonal_to_the_rank_normal_controls_used() -> None:
    enriched = pd.read_csv(OUTPUT / "results/value-enriched-signals.csv", low_memory=False)
    maximum = 0.0
    for _signal_date, group in enriched.groupby("signal_date"):
        for spec in VALUE_SPECS:
            residual = pd.to_numeric(
                group[neutralized_column("cheap", spec)], errors="coerce"
            )
            for control in spec.controls:
                control_rank = rank_normal_score(group[control])
                pair = pd.concat([residual.rename("residual"), control_rank.rename("control")], axis=1).dropna()
                if len(pair) > 2 and pair["residual"].std() > 1e-12:
                    correlation = abs(float(pair["residual"].corr(pair["control"])))
                    maximum = max(maximum, correlation)
    assert maximum < 1e-10


def test_completed_result_covers_requested_metrics_and_has_no_significant_residual_ic() -> None:
    final = json.loads((OUTPUT / "FINAL-RESULT.json").read_text(encoding="utf-8"))
    rows = {row["neutralizer"]: row for row in final["aggregate_comparison"]}
    assert final["signal_date_count"] == 23
    assert final["neutralizer_count"] == 15
    assert final["pbr_reproduction"]["exact_within_1e-12"] is True
    assert {
        "pbr_btm",
        "per_earnings",
        "pfcf",
        "psr",
        "pcr",
        "ev_ebitda",
        "prr_rnd",
        "core_multivariate",
    }.issubset(rows)
    assert np.isclose(rows["pbr_btm"]["raw_ic_mean"], 0.056829256913638826)
    assert np.isclose(rows["pbr_btm"]["neutral_ic_mean"], 0.0008125518013634374)
    assert all(abs(float(row["neutral_ic_hac_t"])) < 1.96 for row in rows.values())
    assert final["max_abs_post_control_corr"] < 1e-10


def test_base_v7_1_artifacts_remained_unchanged() -> None:
    integrity = json.loads((OUTPUT / "source-integrity.json").read_text(encoding="utf-8"))
    assert integrity["base_v7_1_unchanged"] is True
    assert integrity["changed_paths"] == []
    assert (BASE / "FINAL-RESULT.json").exists()
