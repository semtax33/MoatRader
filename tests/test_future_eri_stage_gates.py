from __future__ import annotations

import json
from pathlib import Path

from scripts.run_future_eri_downstream_validation import run as run_downstream
from scripts.run_historical_future_eri_outcomes import run as run_outcomes


def test_outcome_runner_does_not_open_any_vault_when_feature_gate_is_closed(
    tmp_path: Path,
) -> None:
    feature_build = tmp_path / "feature-build"
    feature_build.mkdir()
    (feature_build / "stage-status.json").write_text(
        json.dumps({"outcome_stage_authorized": False}),
        encoding="utf-8",
    )

    result = run_outcomes(
        feature_build=feature_build,
        expectation_input=tmp_path / "does-not-exist-expectations.jsonl",
        outcome_input=tmp_path / "does-not-exist-outcomes.jsonl",
        trading_sessions_path=tmp_path / "does-not-exist-calendar.csv",
        output=tmp_path / "result",
    )

    assert result["status"] == "BLOCKED_FEATURE_COVERAGE_OR_QUALITY_GATE"
    assert result["expectation_input_opened"] is False
    assert result["outcome_vault_opened"] is False
    assert result["return_data_opened"] is False


def test_downstream_runner_does_not_open_optional_inputs_when_mechanism_fails(
    tmp_path: Path,
) -> None:
    mechanism = tmp_path / "mechanism.json"
    mechanism.write_text(
        json.dumps({"downstream_stage_authorized": False}),
        encoding="utf-8",
    )

    result = run_downstream(
        mechanism_stage_status=mechanism,
        output=tmp_path / "downstream",
        analyst_input=tmp_path / "does-not-exist-analyst.jsonl",
        fundamental_input=tmp_path / "does-not-exist-fundamentals.jsonl",
        return_input=tmp_path / "does-not-exist-returns.jsonl",
        authorize_return_stage=True,
    )

    assert result["status"] == "BLOCKED_ERI_MECHANISM_GATE"
    assert result["analyst_input_opened"] is False
    assert result["fundamental_input_opened"] is False
    assert result["return_input_opened"] is False
