from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from moatrader.backtest.universe_corrected import sha256_file
from scripts.run_v7_3_dynamic_expectation_revision import next_quarter_end, signal_dates


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "data-lake/experiments/dynamic-expectation-revision-v7-3-2020-2025-diagnostic"
)


def test_v7_3_uses_23_quarterly_development_dates() -> None:
    dates = signal_dates()
    assert len(dates) == 23
    assert dates[0].isoformat() == "2020-03-31"
    assert dates[-1].isoformat() == "2025-09-30"
    assert next_quarter_end(dates[-1]).isoformat() == "2025-12-31"


def test_prediction_was_sealed_before_target_columns_existed() -> None:
    path = OUTPUT / "results/predictions-pre-target.csv"
    seal = json.loads((OUTPUT / "results/prediction-seal.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(path, nrows=5)
    assert seal["prediction_sha256"] == sha256_file(path)
    assert seal["target_prices_opened_before_seal"] is False
    assert seal["aggregate_sensitivity_revision_confidence_score_created"] is False
    assert "target_price" not in frame.columns
    assert "implied_driver_revision" not in frame.columns
    assert "expected_revision_score" not in frame.columns


def test_multidimensional_surface_is_primary_and_improves_sensor_coverage() -> None:
    contract = json.loads((OUTPUT / "frozen-contract.json").read_text(encoding="utf-8"))
    summary = json.loads(
        (OUTPUT / "results/mechanism-summary.json").read_text(encoding="utf-8")
    )
    assert contract["expectation_surface"]["role"] == "PRIMARY_MARKET_EXPECTATION_SENSOR"
    assert contract["expectation_surface"]["design_points"] == 625
    assert summary["solved_pair_count"] == 1146
    assert summary["one_driver_slice_status_counts"]["SOLVED"] == 640
    assert summary["solved_pair_count"] > summary["one_driver_slice_status_counts"]["SOLVED"]
    assert 0 <= summary["surface_slice_sign_agreement"]["mean_quarterly_agreement"] <= 1


def test_failed_mechanism_gate_prevented_return_stage() -> None:
    final = json.loads((OUTPUT / "FINAL-RESULT.json").read_text(encoding="utf-8"))
    conditional = json.loads(
        (OUTPUT / "results/conditional-return-stage.json").read_text(encoding="utf-8")
    )
    assert final["validation_grade"] == "2020_2025_DEVELOPMENT_DIAGNOSTIC_NOT_OOS"
    assert final["mechanism_gate_passed"] is False
    assert final["true_live_oos"] is False
    assert final["pseudo_oos"] is False
    assert conditional["status"] == "NOT_RUN_MECHANISM_GATE_FAILED"
    assert not (OUTPUT / "results/conditional-return-observations.csv").exists()


def test_v6_through_v7_2_remained_unchanged() -> None:
    integrity = json.loads((OUTPUT / "integrity-after.json").read_text(encoding="utf-8"))
    assert integrity["v6_unchanged"]
    assert integrity["v7_unchanged"]
    assert integrity["v7_1_unchanged"]
    assert integrity["v7_2_unchanged"]
    assert all(not paths for paths in integrity["changed_paths"].values())


def test_v7_3_build_manifest_matches_every_generated_artifact() -> None:
    manifest = json.loads((OUTPUT / "build-manifest.json").read_text(encoding="utf-8"))
    assert manifest["credentials_persisted"] is False
    for relative, expected in manifest["artifacts"].items():
        assert sha256_file(OUTPUT / relative) == expected
