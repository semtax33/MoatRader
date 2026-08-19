from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "data-lake/experiments/value-measurement-quality-v7-1-2020-2025"
)
RESULTS = OUTPUT / "results"
ENRICHED = (
    ROOT
    / "data-lake/experiments/expectation-gap-v7-1-multi-value-neutral-sensitivity-2020-2025"
    / "results/value-enriched-signals.csv"
)


def _final_result() -> dict:
    return json.loads((OUTPUT / "FINAL-RESULT.json").read_text(encoding="utf-8"))


def _row(rows: list[dict], **keys: object) -> dict:
    matches = [row for row in rows if all(row[key] == value for key, value in keys.items())]
    assert len(matches) == 1, keys
    return matches[0]


def test_result_is_complete_and_sources_are_unchanged() -> None:
    final = _final_result()
    assert final["schema_version"] == "moatrader-v7.1-value-measurement-quality-final/1"
    assert final["analysis_grade"] == "EX_POST_MEASUREMENT_QUALITY_DIAGNOSTIC_NOT_NEW_HOLDOUT"
    assert final["period"] == ["2020-03-31", "2025-09-30"]
    assert final["signal_date_count"] == 23
    assert final["source_integrity"]["sources_unchanged"] is True
    assert final["source_integrity"]["changed_paths"] == []
    assert final["source_integrity"]["source_file_count"] == 10


def test_recomputed_77d_returns_exactly_reproduce_the_frozen_input() -> None:
    panel = pd.read_csv(
        RESULTS / "analysis-panel.csv",
        usecols=["signal_date", "ticker", "forward_77d_return"],
        dtype={"ticker": str},
    )
    frozen = pd.read_csv(
        ENRICHED,
        usecols=["signal_date", "ticker", "forward_77d_return"],
        dtype={"ticker": str},
    )
    joined = panel.merge(
        frozen,
        on=["signal_date", "ticker"],
        how="inner",
        suffixes=("_recomputed", "_frozen"),
        validate="one_to_one",
    ).dropna(subset=["forward_77d_return_recomputed", "forward_77d_return_frozen"])
    difference = (
        joined["forward_77d_return_recomputed"] - joined["forward_77d_return_frozen"]
    ).abs()
    final_reproduction = _final_result()["return_77d_reproduction"]
    assert len(joined) == final_reproduction["matched_rows"] == 3446
    assert float(difference.max()) < 1e-12
    assert final_reproduction["max_abs_difference"] < 1e-12


def test_loss_company_coverage_separates_calculable_from_trusted_dcf() -> None:
    rows = _final_result()["dimensions"]["loss_coverage"]
    calculable = _row(rows, segment="nonpositive_net_income", measure="dcf_calculable")
    trusted = _row(rows, segment="nonpositive_net_income", measure="dcf_cheap")
    pbr = _row(rows, segment="nonpositive_net_income", measure="pbr")
    per = _row(rows, segment="nonpositive_net_income", measure="per")

    assert calculable["segment_rows"] == trusted["segment_rows"] == 957
    assert calculable["available_rows"] == 932
    assert calculable["coverage"] > 0.97
    assert trusted["available_rows"] == 95
    assert np.isclose(trusted["coverage"], 95 / 957)
    assert pbr["coverage"] > 0.99
    assert per["coverage"] == 0.0


def test_long_horizon_comparison_uses_equal_common_samples_and_favors_simple_value() -> None:
    summary = pd.read_csv(RESULTS / "horizon-summary.csv")
    common = summary.loc[summary["lane"] == "common"].copy()
    assert set(common["measure"]) == {"dcf_cheap", "pbr", "per", "simple_per_pbr"}
    assert common.groupby("horizon_days")["average_n"].nunique().eq(1).all()
    assert (
        common.groupby("horizon_days")["quarters"].first().astype(int).to_dict()
        == {77: 23, 182: 22, 365: 20, 730: 16}
    )

    indexed = common.set_index(["horizon_days", "measure"])
    for horizon in (77, 182, 365, 730):
        assert indexed.loc[(horizon, "dcf_cheap"), "mean_ic"] < indexed.loc[
            (horizon, "simple_per_pbr"), "mean_ic"
        ]
    assert np.isclose(indexed.loc[(730, "dcf_cheap"), "positive_ic_quarter_share"], 0.5625)
    assert indexed.loc[(730, "pbr"), "positive_ic_quarter_share"] == 1.0
    assert indexed.loc[(730, "simple_per_pbr"), "positive_ic_quarter_share"] == 1.0

    differences = pd.read_csv(RESULTS / "horizon-dcf-differences.csv")
    d365 = differences.loc[
        (differences["horizon_days"] == 365)
        & (differences["benchmark"] == "simple_per_pbr")
        & (differences["metric"] == "ic")
    ].iloc[0]
    assert d365["dcf_minus_benchmark_mean"] < -0.08
    assert d365["holm_adjusted_p"] < 0.05


def test_dcf_reduces_ebit_deterioration_but_not_price_loss_or_return_traps() -> None:
    differences = pd.read_csv(RESULTS / "value-trap-differences.csv")
    simple = differences.loc[differences["benchmark"] == "simple_per_pbr"].set_index("metric")

    assert simple.loc["ebit_deterioration_rate", "dcf_minus_benchmark_mean"] < -0.08
    assert simple.loc["ebit_deterioration_rate", "holm_adjusted_p"] < 0.05
    assert simple.loc["mean_complete_return", "dcf_minus_benchmark_mean"] < -0.05
    assert simple.loc["mean_complete_return", "holm_adjusted_p"] < 0.05
    assert simple.loc["severe_loss_rate", "dcf_minus_benchmark_mean"] > 0.0
    assert simple.loc["severe_loss_rate", "holm_adjusted_p"] > 0.05
    assert simple.loc["combined_trap_rate", "dcf_minus_benchmark_mean"] < 0.0
    assert simple.loc["combined_trap_rate", "holm_adjusted_p"] > 0.05


def test_accounting_structure_advantage_is_only_relative_to_pbr() -> None:
    rows = _final_result()["dimensions"]["accounting_structure"]
    dcf = _row(rows, measure="dcf_cheap")
    pbr = _row(rows, measure="pbr")
    simple = _row(rows, measure="simple_per_pbr")
    assert dcf["mean_structure_joint_r2"] < pbr["mean_structure_joint_r2"]
    assert dcf["mean_structure_joint_r2"] > simple["mean_structure_joint_r2"]

    differences = pd.read_csv(RESULTS / "accounting-structure-differences.csv")
    joint = differences.loc[differences["metric"] == "structure_joint_r2"].set_index("benchmark")
    assert joint.loc["pbr", "dcf_minus_benchmark_mean"] < -0.07
    assert joint.loc["pbr", "holm_adjusted_p"] < 0.05
    assert joint.loc["simple_per_pbr", "holm_adjusted_p"] > 0.05


def test_industry_comparability_has_no_holm_significant_dcf_advantage() -> None:
    rows = _final_result()["dimensions"]["industry_comparability"]
    dcf = _row(rows, measure="dcf_cheap")
    pbr = _row(rows, measure="pbr")
    simple = _row(rows, measure="simple_per_pbr")
    assert dcf["sector_neutral_365d_ic"] < pbr["sector_neutral_365d_ic"]
    assert dcf["sector_neutral_365d_ic"] < simple["sector_neutral_365d_ic"]

    differences = pd.read_csv(RESULTS / "industry-differences.csv")
    assert (differences["holm_adjusted_p"] > 0.05).all()


def test_final_report_records_the_ex_post_universal_value_rejection() -> None:
    report = (OUTPUT / "FINAL-REPORT.md").read_text(encoding="utf-8")
    assert "사후 진단" in report
    assert "비-PIT sensitivity only" in report
    assert "Universal Value measurement" in report
    assert "주력 Value ranker로는 단순 PER+PBR 또는 PBR보다 약했습니다" in report
