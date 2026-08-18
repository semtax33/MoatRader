from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from moatrader.backtest.universe_corrected import rank_normal_score, sha256_file
from scripts.run_v7_2_driver_expectation_gaps import holm_adjust, signal_dates


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data-lake/experiments/driver-expectation-gap-v7-2-2018-2019"


def test_holdout_dates_are_pre_2020_and_monthly() -> None:
    dates = signal_dates()
    assert len(dates) == 18
    assert dates[0].isoformat() == "2018-04-30"
    assert dates[-1].isoformat() == "2019-09-30"
    assert all(value.year < 2020 for value in dates)


def test_holm_adjustment_is_ordered_and_never_smaller_than_raw_p() -> None:
    raw = {"A": 0.01, "B": 0.02, "C": 0.20, "D": 0.80}
    adjusted = holm_adjust(raw)
    assert all(adjusted[key] >= raw[key] for key in raw)
    ordered = [adjusted[key] for key in sorted(raw, key=raw.get)]
    assert ordered == sorted(ordered)


def test_completed_signal_seal_and_pre_2020_return_boundary() -> None:
    seal = json.loads((OUTPUT / "results/signals-seal.json").read_text(encoding="utf-8"))
    signal_path = OUTPUT / "results/signals-pre-return.csv"
    assert seal["signals_sha256"] == sha256_file(signal_path)
    assert seal["return_labels_opened_before_seal"] is False
    results = pd.read_csv(OUTPUT / "results/signals-with-returns.csv")
    exits = pd.to_datetime(results["exit_date"], errors="coerce").dropna()
    assert exits.max() < pd.Timestamp("2020-01-01")


def test_value_neutral_signals_are_orthogonal_to_rank_normal_book_to_market() -> None:
    signals = pd.read_csv(OUTPUT / "results/signals-pre-return.csv")
    for _signal_date, group in signals.groupby("signal_date"):
        value = rank_normal_score(group["value_btm"])
        for driver in ("growth", "margin", "roiic", "cap"):
            residual = pd.to_numeric(group[f"{driver}_gap_vn"], errors="coerce")
            pair = pd.concat([residual.rename("residual"), value.rename("value")], axis=1).dropna()
            if len(pair) > 2 and pair["residual"].std() > 1e-12:
                assert abs(np.corrcoef(pair["residual"], pair["value"])[0, 1]) < 1e-8


def test_no_driver_was_promoted_and_prior_versions_remained_unchanged() -> None:
    final = json.loads((OUTPUT / "FINAL-RESULT.json").read_text(encoding="utf-8"))
    assert final["overall_judgment"] == "NO_DRIVER_CLEARED_PROMOTION_GATE"
    assert "PROMOTE" not in final["driver_judgments"].values()
    integrity = json.loads((OUTPUT / "integrity-after.json").read_text(encoding="utf-8"))
    assert integrity["v6_unchanged"]
    assert integrity["v7_unchanged"]
    assert integrity["v7_1_unchanged"]
