from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from moatrader.backtest.universe_corrected import (
    rank_normal_score,
    residualize_cross_section,
    spearman_ic,
)
from moatrader.expectations.historical_evidence import sha256_file
from scripts.prepare_historical_evidence_index_eri_inputs_v2 import (
    _git_state,
    _load_sessions,
)
from scripts.run_historical_evidence_index_value_neutralization_v2 import (
    _inference,
    _read_json,
    _read_records,
    _write_json,
    _write_jsonl,
)


WINDOW_DEFINITIONS_V2: dict[str, dict[str, int]] = {
    "momentum_1m": {
        "lookback_sessions": 21,
        "skip_most_recent_sessions": 0,
        "expected_return_observations": 21,
        "minimum_return_observations": 17,
    },
    "momentum_3_1": {
        "lookback_sessions": 63,
        "skip_most_recent_sessions": 21,
        "expected_return_observations": 42,
        "minimum_return_observations": 34,
    },
    "momentum_6_1": {
        "lookback_sessions": 126,
        "skip_most_recent_sessions": 21,
        "expected_return_observations": 105,
        "minimum_return_observations": 84,
    },
    "momentum_12_1": {
        "lookback_sessions": 252,
        "skip_most_recent_sessions": 21,
        "expected_return_observations": 231,
        "minimum_return_observations": 185,
    },
}

CONTROL_SPECS_V2: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("momentum_1m", "Momentum 1M / Pre-filing 21 sessions", ("momentum_1m",)),
    ("momentum_3_1", "Momentum 3-1", ("momentum_3_1",)),
    ("momentum_6_1", "Momentum 6-1", ("momentum_6_1",)),
    ("momentum_12_1", "Momentum 12-1 exact sessions", ("momentum_12_1",)),
    (
        "joint_3_6_12",
        "Momentum 3-1 + 6-1 + 12-1",
        ("momentum_3_1", "momentum_6_1", "momentum_12_1"),
    ),
    (
        "joint_all_four",
        "Momentum 1M + 3-1 + 6-1 + 12-1",
        ("momentum_1m", "momentum_3_1", "momentum_6_1", "momentum_12_1"),
    ),
)

HORIZONS_V2 = (21, 42, 63)
BAND_ORDER_V2 = ("STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL")


def _contract_payload(contract_path: Path, *, eri_build: Path) -> dict[str, Any]:
    contract = _read_json(contract_path)
    frozen = contract.get("frozen_inputs") or {}
    control = contract.get("historical_momentum_controls") or {}
    stage_path = eri_build / "stage-status.json"
    feature_path = eri_build / "features-with-frozen-expectations-pre-outcome.jsonl"
    if not (
        contract.get("status") == "PREREGISTERED_BEFORE_SHORT_MOMENTUM_RESULTS"
        and frozen.get("eri_stage_status_sha256") == sha256_file(stage_path)
        and frozen.get("feature_rows_sha256") == sha256_file(feature_path)
        and frozen.get("pre_outcome_feature_count") == 1673
        and frozen.get("panel_observation_count") == 1640
        and control.get("definitions") == WINDOW_DEFINITIONS_V2
        and control.get("signal_day_included") is False
        and control.get("minimum_coverage_ratio") == 0.8
        and contract.get("evidence_definition_changed") is False
        and contract.get("future_eri_definition_changed") is False
        and contract.get("source_files_may_be_modified") is False
    ):
        raise ValueError("short-momentum preregistration contract does not match frozen inputs")
    return contract


def _assert_pre_outcome_rows(rows: Iterable[dict[str, Any]]) -> None:
    prohibited = ("future_eri", "future_return", "forward_return")

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                if any(part in str(key).casefold() for part in prohibited):
                    raise ValueError(f"pre-outcome input contains downstream value: {key}")
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for row in rows:
        visit(row)


def _load_return_maps(
    files: Sequence[Path], tickers: set[str]
) -> tuple[dict[str, dict[date, float]], dict[str, str]]:
    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    selected = sorted(tickers)
    for path in files:
        hashes[str(path.resolve())] = sha256_file(path)
        frame = pd.read_parquet(
            path,
            columns=["Date", "Code", "ChangesRatio"],
            filters=[("Code", "in", selected)],
        )
        if frame.empty:
            continue
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.date
        frame["Code"] = frame["Code"].astype(str).str.zfill(6)
        frame["ChangesRatio"] = pd.to_numeric(frame["ChangesRatio"], errors="coerce")
        pieces.append(frame)
    if not pieces:
        return {}, hashes
    combined = pd.concat(pieces, ignore_index=True).sort_values(["Code", "Date"])
    result: dict[str, dict[date, float]] = {}
    for ticker, group in combined.groupby("Code", sort=False):
        values: dict[date, float] = {}
        for row in group[["Date", "ChangesRatio"]].itertuples(index=False):
            value = float(row.ChangesRatio)
            if math.isfinite(value):
                values.setdefault(row.Date, value)
        result[str(ticker)] = values
    return result, hashes


def _compound_percent(values: Iterable[float]) -> float | None:
    total = 1.0
    used = 0
    for raw in values:
        value = float(raw)
        if not math.isfinite(value):
            continue
        total *= 1.0 + value / 100.0
        used += 1
    result = total - 1.0
    return result if used and math.isfinite(result) else None


def historical_momentum_window(
    returns: dict[date, float],
    sessions: Sequence[date],
    *,
    signal_date: date,
    lookback_sessions: int,
    skip_most_recent_sessions: int,
    minimum_return_observations: int,
) -> dict[str, Any]:
    location = bisect.bisect_left(sessions, signal_date)
    prior = list(sessions[:location])
    expected = lookback_sessions - skip_most_recent_sessions
    if len(prior) < lookback_sessions or expected <= 0:
        return {
            "value": None,
            "observed_return_count": 0,
            "expected_return_count": expected,
            "window_start": None,
            "window_end": None,
        }
    selected = prior[-lookback_sessions:]
    if skip_most_recent_sessions:
        selected = selected[:-skip_most_recent_sessions]
    values = [returns[item] for item in selected if item in returns]
    return {
        "value": (
            _compound_percent(values)
            if len(values) >= minimum_return_observations
            else None
        ),
        "observed_return_count": len(values),
        "expected_return_count": len(selected),
        "window_start": selected[0].isoformat(),
        "window_end": selected[-1].isoformat(),
    }


def future_momentum_windows(
    returns: dict[date, float],
    sessions: Sequence[date],
    *,
    signal_date: date,
    horizon: int,
) -> dict[str, Any]:
    location = bisect.bisect_left(sessions, signal_date)
    post = list(sessions[location : location + horizon])
    if len(post) < horizon:
        return {
            "forward_return": None,
            "future_1m_momentum": None,
            "target_session": None,
            "forward_observed_count": 0,
            "future_1m_observed_count": 0,
        }
    forward_values = [returns[item] for item in post if item in returns]
    rolling_sessions = post[-21:]
    rolling_values = [returns[item] for item in rolling_sessions if item in returns]
    return {
        "forward_return": (
            _compound_percent(forward_values)
            if len(forward_values) >= math.ceil(0.8 * horizon)
            else None
        ),
        "future_1m_momentum": (
            _compound_percent(rolling_values) if len(rolling_values) >= 17 else None
        ),
        "target_session": post[-1].isoformat(),
        "forward_observed_count": len(forward_values),
        "future_1m_observed_count": len(rolling_values),
    }


def _band(value: float) -> str:
    if value == -1.0:
        return "STRONG_BEAR"
    if -1.0 < value < 0.0:
        return "BEAR"
    if value == 0.0:
        return "NEUTRAL"
    if 0.0 < value < 1.0:
        return "BULL"
    if value == 1.0:
        return "STRONG_BULL"
    raise ValueError(f"invalid frozen Evidence value: {value}")


def _verify_sources_unchanged(hashes: dict[str, str]) -> None:
    changed = [path for path, expected in hashes.items() if sha256_file(Path(path)) != expected]
    if changed:
        raise RuntimeError(f"a MARCAP source changed during momentum analysis: {changed[:3]}")


def prepare_momentum_controls_pre_outcome(
    *,
    workspace: Path,
    eri_build: Path,
    contract_path: Path,
    marcap_files: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production momentum-control preparation requires a clean worktree")
    contract = _contract_payload(contract_path, eri_build=eri_build)
    eri_stage_path = eri_build / "stage-status.json"
    feature_path = eri_build / "features-with-frozen-expectations-pre-outcome.jsonl"
    eri_stage = _read_json(eri_stage_path)
    if not (
        eri_stage.get("status")
        in {"FULL_PRIMARY_MECHANISM_PASSED", "FULL_PRIMARY_MECHANISM_REJECTED_OR_INCONCLUSIVE"}
        and eri_stage.get("future_eri_used_as_signal") is False
        and eri_stage.get("future_eri_used_as_ranking") is False
        and eri_stage.get("return_data_opened") is False
    ):
        raise ValueError("sealed ERI stage is not authorized for a parallel diagnostic")
    features = _read_records(feature_path)
    _assert_pre_outcome_rows(features)
    frozen = contract["frozen_inputs"]
    if len(features) != int(frozen["pre_outcome_feature_count"]):
        raise ValueError("frozen ERI feature count changed")
    tickers = {str(item["issuer_id"]).zfill(6) for item in features}
    sessions = _load_sessions(marcap_files)
    returns_by_ticker, marcap_hashes = _load_return_maps(marcap_files, tickers)
    rows: list[dict[str, Any]] = []
    coverage: Counter[str] = Counter()
    for feature in sorted(features, key=lambda item: str(item["observation_id"])):
        ticker = str(feature["issuer_id"]).zfill(6)
        signal = pd.Timestamp(feature["signal_timestamp"])
        ticker_returns = returns_by_ticker.get(ticker, {})
        values: dict[str, float | None] = {}
        diagnostics: dict[str, dict[str, Any]] = {}
        for field, definition in WINDOW_DEFINITIONS_V2.items():
            calculated = historical_momentum_window(
                ticker_returns,
                sessions,
                signal_date=signal.date(),
                lookback_sessions=definition["lookback_sessions"],
                skip_most_recent_sessions=definition["skip_most_recent_sessions"],
                minimum_return_observations=definition["minimum_return_observations"],
            )
            values[field] = calculated.pop("value")
            diagnostics[field] = calculated
            coverage[field] += int(values[field] is not None)
        rows.append(
            {
                "schema_version": "moatrader-short-momentum-control-row-v2/1",
                "observation_id": str(feature["observation_id"]),
                "issuer_id": ticker,
                "signal_timestamp": signal.isoformat(),
                "factor_available_at": signal.isoformat(),
                "prefiling_return_21": values["momentum_1m"],
                "window_diagnostics": diagnostics,
                **values,
            }
        )
    thresholds = contract["historical_momentum_controls"]
    coverage_checks = {
        field: count >= int(thresholds["minimum_complete_control_observations"])
        for field, count in sorted(coverage.items())
    }
    coverage_passed = len(coverage_checks) == len(WINDOW_DEFINITIONS_V2) and all(
        coverage_checks.values()
    )
    _verify_sources_unchanged(marcap_hashes)
    output.mkdir(parents=True, exist_ok=True)
    controls_path = output / "momentum-controls-pre-outcome.jsonl"
    _write_jsonl(controls_path, rows)
    seal = {
        "schema_version": "moatrader-short-momentum-control-seal-v2/1",
        "status": "SHORT_MOMENTUM_CONTROLS_SEALED_OUTCOMES_CLOSED",
        "git_commit": commit,
        "worktree_dirty": False,
        "script_sha256": sha256_file(Path(__file__)),
        "contract_sha256": sha256_file(contract_path),
        "eri_stage_sha256": sha256_file(eri_stage_path),
        "feature_rows_sha256": sha256_file(feature_path),
        "expected_future_eri_labels_sha256": frozen["future_eri_labels_sha256"],
        "momentum_controls_sha256": sha256_file(controls_path),
        "marcap_source_hashes": marcap_hashes,
        "observation_count": len(rows),
        "coverage_counts": dict(sorted(coverage.items())),
        "coverage_checks": coverage_checks,
        "coverage_gate_passed": coverage_passed,
        "window_definitions": WINDOW_DEFINITIONS_V2,
        "future_eri_values_opened": False,
        "forward_price_returns_opened": False,
        "signal_day_return_used_in_controls": False,
        "source_files_modified": False,
        "outcome_stage_authorized": coverage_passed,
    }
    seal_path = output / "momentum-controls-seal.json"
    _write_json(seal_path, seal)
    status = {**seal, "momentum_controls_seal_sha256": sha256_file(seal_path)}
    _write_json(output / "stage-status.json", status)
    return status


def _prepare_neutral_panel(
    *,
    features: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    controls: list[dict[str, Any]],
) -> pd.DataFrame:
    feature_by_id = {str(item["observation_id"]): item for item in features}
    label_by_id = {str(item["observation_id"]): item for item in labels}
    control_by_id = {str(item["observation_id"]): item for item in controls}
    if not set(label_by_id).issubset(feature_by_id) or not set(label_by_id).issubset(
        control_by_id
    ):
        raise ValueError("final Future ERI IDs are not covered by feature/control rows")
    rows: list[dict[str, Any]] = []
    for observation_id in sorted(label_by_id):
        feature = feature_by_id[observation_id]
        control = control_by_id[observation_id]
        signal = pd.Timestamp(feature["signal_timestamp"])
        if signal != pd.Timestamp(control["signal_timestamp"]):
            raise ValueError("momentum control timestamp differs from Evidence timestamp")
        rows.append(
            {
                "observation_id": observation_id,
                "issuer_id": str(feature["issuer_id"]),
                "signal_month": signal.strftime("%Y-%m"),
                "full_evidence_index": float(feature["full_evidence_index"]),
                "future_eri": float(label_by_id[observation_id]["future_eri"]),
                **{
                    field: (
                        float(control[field]) if control.get(field) is not None else np.nan
                    )
                    for field in WINDOW_DEFINITIONS_V2
                },
            }
        )
    return pd.DataFrame(rows)


def _neutral_monthly(panel: pd.DataFrame, *, minimum_n: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal_month, month in panel.groupby("signal_month", sort=True):
        for key, label, controls in CONTROL_SPECS_V2:
            sample = month[["full_evidence_index", "future_eri", *controls]].dropna().copy()
            base = {
                "signal_month": signal_month,
                "test": key,
                "test_label": label,
                "controls": list(controls),
                "n": len(sample),
                "same_sample_raw_and_neutral": True,
            }
            if len(sample) < max(minimum_n, len(controls) + 3):
                rows.append({**base, "status": "INSUFFICIENT_MONTHLY_OBSERVATIONS"})
                continue
            sample["neutral_evidence"] = residualize_cross_section(
                sample,
                target="full_evidence_index",
                numeric_controls=controls,
            )
            sample = sample.dropna(subset=["neutral_evidence"])
            raw_ic = spearman_ic(sample, "full_evidence_index", "future_eri")
            neutral_ic = spearman_ic(sample, "neutral_evidence", "future_eri")
            post_correlations = [
                abs(float(sample["neutral_evidence"].corr(sample[field], method="spearman")))
                for field in controls
            ]
            rows.append(
                {
                    **base,
                    "status": "EVALUATED_SAME_SAMPLE",
                    "n": len(sample),
                    "raw_ic": raw_ic,
                    "neutral_ic": neutral_ic,
                    "delta_ic": neutral_ic - raw_ic,
                    "max_abs_post_control_spearman": max(post_correlations),
                }
            )
    return rows


def _inference_triplet(
    selected: list[dict[str, Any]], *, inference: dict[str, Any]
) -> dict[str, Any]:
    return {
        field: _inference(
            [float(item[field]) for item in selected],
            hac_lag_months=int(inference["newey_west_lag_months"]),
            block_length_months=int(inference["moving_block_length_months"]),
            bootstrap_repetitions=int(inference["bootstrap_repetitions"]),
            bootstrap_seed=int(inference["bootstrap_seed"]),
        )
        for field in ("raw_ic", "neutral_ic", "delta_ic")
    }


def _neutral_summary(
    panel: pd.DataFrame, monthly: list[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    inference_config = contract["inference"]
    tests: dict[str, Any] = {}
    for key, label, controls in CONTROL_SPECS_V2:
        selected = [
            item
            for item in monthly
            if item["test"] == key and item["status"] == "EVALUATED_SAME_SAMPLE"
        ]
        values = _inference_triplet(selected, inference=inference_config)
        raw = values["raw_ic"]["newey_west"]["mean"]
        neutral = values["neutral_ic"]["newey_west"]["mean"]
        retention = (
            float(neutral) / float(raw)
            if raw is not None and neutral is not None and abs(float(raw)) > 1e-12
            else None
        )
        tests[key] = {
            "test_label": label,
            "controls": list(controls),
            "complete_control_observation_count": int(
                panel[list(controls)].notna().all(axis=1).sum()
            ),
            "valid_month_count": len(selected),
            "same_sample_raw_and_neutral": True,
            **values,
            "signed_ic_retention_ratio": retention,
            "absolute_ic_attenuation": (
                1.0 - abs(float(neutral)) / abs(float(raw))
                if retention is not None
                else None
            ),
            "max_abs_post_control_spearman": (
                max(float(item["max_abs_post_control_spearman"]) for item in selected)
                if selected
                else None
            ),
        }
    prefiling_monthly = []
    for signal_month, month in panel.groupby("signal_month", sort=True):
        sample = month[["full_evidence_index", "momentum_1m"]].dropna()
        if len(sample) >= 5:
            prefiling_monthly.append(
                {
                    "signal_month": signal_month,
                    "n": len(sample),
                    "spearman": spearman_ic(
                        sample, "full_evidence_index", "momentum_1m"
                    ),
                }
            )
    prefiling_inference = _inference(
        [float(item["spearman"]) for item in prefiling_monthly],
        hac_lag_months=int(inference_config["newey_west_lag_months"]),
        block_length_months=int(inference_config["moving_block_length_months"]),
        bootstrap_repetitions=int(inference_config["bootstrap_repetitions"]),
        bootstrap_seed=int(inference_config["bootstrap_seed"]),
    )
    return {
        "schema_version": "moatrader-short-momentum-neutralization-summary-v2/1",
        "status": "SHORT_MOMENTUM_NEUTRALIZATION_EVALUATED",
        "panel_observation_count": len(panel),
        "signal_month_count": int(panel["signal_month"].nunique()),
        "tests": tests,
        "prefiling_return_21_evidence_correlation": prefiling_inference,
        "prefiling_return_monthly": prefiling_monthly,
        "ranking_output_produced": False,
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "forward_price_returns_opened": False,
    }


def evaluate_lead_lag_authorization(
    summary: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    gate = contract["lead_lag_authorization_gate"]
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    for key in gate["required_tests"]:
        result = summary["tests"][key]
        retention = result["signed_ic_retention_ratio"]
        lower = result["neutral_ic"]["moving_block_bootstrap"]["ci_low"]
        months = result["valid_month_count"]
        checks[f"{key}:retention"] = retention is not None and float(retention) >= float(
            gate["minimum_signed_ic_retention"]
        )
        checks[f"{key}:neutral_bootstrap_lower"] = lower is not None and float(lower) > float(
            gate["neutral_ic_moving_block_bootstrap_lower_bound_must_exceed"]
        )
        checks[f"{key}:valid_months"] = int(months) >= int(gate["minimum_valid_months"])
        details[key] = {
            "signed_ic_retention_ratio": retention,
            "neutral_ic_bootstrap_ci_low": lower,
            "valid_month_count": months,
        }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "details": details,
        "forward_price_returns_opened_for_gate": False,
    }


def run_eri_momentum_neutralization(
    *,
    workspace: Path,
    eri_build: Path,
    contract_path: Path,
    pre_outcome_build: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production momentum neutralization requires a clean worktree")
    contract = _contract_payload(contract_path, eri_build=eri_build)
    pre_stage_path = pre_outcome_build / "stage-status.json"
    controls_path = pre_outcome_build / "momentum-controls-pre-outcome.jsonl"
    pre_stage = _read_json(pre_stage_path)
    if not (
        pre_stage.get("status") == "SHORT_MOMENTUM_CONTROLS_SEALED_OUTCOMES_CLOSED"
        and pre_stage.get("outcome_stage_authorized") is True
        and pre_stage.get("future_eri_values_opened") is False
        and pre_stage.get("forward_price_returns_opened") is False
        and pre_stage.get("momentum_controls_sha256") == sha256_file(controls_path)
        and pre_stage.get("contract_sha256") == sha256_file(contract_path)
    ):
        raise ValueError("pre-outcome short-momentum controls are not sealed or authorized")
    feature_path = eri_build / "features-with-frozen-expectations-pre-outcome.jsonl"
    labels_path = eri_build / "future-eri-labels.jsonl"
    if sha256_file(labels_path) != contract["frozen_inputs"]["future_eri_labels_sha256"]:
        raise ValueError("Future ERI labels differ from the preregistered hash")
    input_hashes = {
        "contract": sha256_file(contract_path),
        "pre_stage": sha256_file(pre_stage_path),
        "controls": sha256_file(controls_path),
        "features": sha256_file(feature_path),
        "labels": sha256_file(labels_path),
    }
    panel = _prepare_neutral_panel(
        features=_read_records(feature_path),
        labels=_read_records(labels_path),
        controls=_read_records(controls_path),
    )
    minimum_n = int(contract["lead_lag_test_if_authorized"]["minimum_monthly_observations"])
    monthly = _neutral_monthly(panel, minimum_n=minimum_n)
    summary = _neutral_summary(panel, monthly, contract)
    authorization = evaluate_lead_lag_authorization(summary, contract)
    summary["lead_lag_authorization_gate"] = authorization
    output.mkdir(parents=True, exist_ok=True)
    monthly_path = output / "monthly-short-momentum-neutralization.jsonl"
    summary_path = output / "short-momentum-neutralization-summary.json"
    _write_jsonl(monthly_path, monthly)
    _write_json(summary_path, summary)
    after = {
        "contract": sha256_file(contract_path),
        "pre_stage": sha256_file(pre_stage_path),
        "controls": sha256_file(controls_path),
        "features": sha256_file(feature_path),
        "labels": sha256_file(labels_path),
    }
    if input_hashes != after:
        raise RuntimeError("a sealed Momentum or ERI input changed during neutralization")
    status = {
        "schema_version": "moatrader-short-momentum-neutralization-stage-v2/1",
        "status": "SHORT_MOMENTUM_NEUTRALIZATION_COMPLETE",
        "git_commit": commit,
        "worktree_dirty": False,
        "panel_observation_count": len(panel),
        "lead_lag_stage_authorized": authorization["status"] == "PASS",
        "future_eri_values_opened": True,
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "forward_price_returns_opened": False,
        "ranking_output_produced": False,
        "source_files_modified": False,
    }
    stage_path = output / "stage-status.json"
    _write_json(stage_path, status)
    _write_json(
        output / "build-manifest.json",
        {
            **status,
            "input_hashes": input_hashes,
            "output_hashes": {
                "monthly": sha256_file(monthly_path),
                "summary": sha256_file(summary_path),
                "stage": sha256_file(stage_path),
            },
        },
    )
    return status


def _prepare_lead_lag_panel(
    *,
    features: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    returns_by_ticker: dict[str, dict[date, float]],
    sessions: Sequence[date],
) -> pd.DataFrame:
    control_by_id = {str(item["observation_id"]): item for item in controls}
    rows: list[dict[str, Any]] = []
    for feature in sorted(features, key=lambda item: str(item["observation_id"])):
        observation_id = str(feature["observation_id"])
        control = control_by_id[observation_id]
        ticker = str(feature["issuer_id"]).zfill(6)
        signal = pd.Timestamp(feature["signal_timestamp"])
        row: dict[str, Any] = {
            "observation_id": observation_id,
            "issuer_id": ticker,
            "signal_month": signal.strftime("%Y-%m"),
            "full_evidence_index": float(feature["full_evidence_index"]),
            "band": _band(float(feature["full_evidence_index"])),
            "prefiling_return_21": (
                float(control["prefiling_return_21"])
                if control.get("prefiling_return_21") is not None
                else np.nan
            ),
        }
        for horizon in HORIZONS_V2:
            values = future_momentum_windows(
                returns_by_ticker.get(ticker, {}),
                sessions,
                signal_date=signal.date(),
                horizon=horizon,
            )
            row[f"forward_return_{horizon}"] = values["forward_return"]
            row[f"future_1m_momentum_{horizon}"] = values["future_1m_momentum"]
            row[f"target_session_{horizon}"] = values["target_session"]
            row[f"forward_observed_count_{horizon}"] = values["forward_observed_count"]
            row[f"future_1m_observed_count_{horizon}"] = values[
                "future_1m_observed_count"
            ]
        rows.append(row)
    panel = pd.DataFrame(rows)
    for horizon in HORIZONS_V2:
        current = "prefiling_return_21"
        future = f"future_1m_momentum_{horizon}"
        current_rank = f"prefiling_rank_{horizon}"
        future_rank = f"future_momentum_rank_{horizon}"
        panel[current_rank] = np.nan
        panel[future_rank] = np.nan
        for _, indices in panel.groupby("signal_month", sort=True).groups.items():
            subset = panel.loc[indices, [current, future]].dropna()
            if len(subset) < 5:
                continue
            panel.loc[subset.index, current_rank] = subset[current].rank(
                method="average", pct=True
            )
            panel.loc[subset.index, future_rank] = subset[future].rank(
                method="average", pct=True
            )
        panel[f"delta_momentum_rank_{horizon}"] = (
            panel[future_rank] - panel[current_rank]
        )
    return panel


def _lead_lag_results(
    panel: pd.DataFrame, contract: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    minimum_n = int(contract["lead_lag_test_if_authorized"]["minimum_monthly_observations"])
    inference_config = contract["inference"]
    monthly: list[dict[str, Any]] = []
    horizons: dict[str, Any] = {}
    for horizon in HORIZONS_V2:
        delta_field = f"delta_momentum_rank_{horizon}"
        return_field = f"forward_return_{horizon}"
        for signal_month, month in panel.groupby("signal_month", sort=True):
            delta_sample = month[["full_evidence_index", delta_field]].dropna()
            return_sample = month[["full_evidence_index", return_field]].dropna()
            monthly.append(
                {
                    "signal_month": signal_month,
                    "horizon": horizon,
                    "delta_rank_n": len(delta_sample),
                    "delta_rank_ic": (
                        spearman_ic(delta_sample, "full_evidence_index", delta_field)
                        if len(delta_sample) >= minimum_n
                        else None
                    ),
                    "forward_return_n": len(return_sample),
                    "forward_return_ic": (
                        spearman_ic(return_sample, "full_evidence_index", return_field)
                        if len(return_sample) >= minimum_n
                        else None
                    ),
                }
            )
        selected = [
            item
            for item in monthly
            if item["horizon"] == horizon and item["delta_rank_ic"] is not None
        ]
        returns = [
            item
            for item in monthly
            if item["horizon"] == horizon and item["forward_return_ic"] is not None
        ]
        bands: list[dict[str, Any]] = []
        for band in BAND_ORDER_V2:
            sample = panel.loc[panel["band"] == band, [delta_field, return_field]].dropna()
            bands.append(
                {
                    "band": band,
                    "n": len(sample),
                    "mean_delta_momentum_rank": (
                        float(sample[delta_field].mean()) if len(sample) else None
                    ),
                    "median_delta_momentum_rank": (
                        float(sample[delta_field].median()) if len(sample) else None
                    ),
                    "mean_forward_return": (
                        float(sample[return_field].mean()) if len(sample) else None
                    ),
                    "median_forward_return": (
                        float(sample[return_field].median()) if len(sample) else None
                    ),
                }
            )
        horizons[str(horizon)] = {
            "delta_momentum_rank_ic": _inference(
                [float(item["delta_rank_ic"]) for item in selected],
                hac_lag_months=int(inference_config["newey_west_lag_months"]),
                block_length_months=int(inference_config["moving_block_length_months"]),
                bootstrap_repetitions=int(inference_config["bootstrap_repetitions"]),
                bootstrap_seed=int(inference_config["bootstrap_seed"]),
            ),
            "forward_return_ic": _inference(
                [float(item["forward_return_ic"]) for item in returns],
                hac_lag_months=int(inference_config["newey_west_lag_months"]),
                block_length_months=int(inference_config["moving_block_length_months"]),
                bootstrap_repetitions=int(inference_config["bootstrap_repetitions"]),
                bootstrap_seed=int(inference_config["bootstrap_seed"]),
            ),
            "five_band": bands,
        }
    positive_lower = 0
    positive_spread = 0
    for horizon in HORIZONS_V2:
        result = horizons[str(horizon)]
        lower = result["delta_momentum_rank_ic"]["moving_block_bootstrap"]["ci_low"]
        positive_lower += int(lower is not None and float(lower) > 0.0)
        band_by_name = {item["band"]: item for item in result["five_band"]}
        bull = band_by_name["STRONG_BULL"]["median_delta_momentum_rank"]
        bear = band_by_name["STRONG_BEAR"]["median_delta_momentum_rank"]
        spread = float(bull) - float(bear) if bull is not None and bear is not None else None
        result["strong_bull_minus_strong_bear_median_delta_rank"] = spread
        positive_spread += int(spread is not None and spread > 0.0)
    policy = contract["lead_lag_test_if_authorized"]
    gate = {
        "positive_bootstrap_lower_bound_horizon_count": positive_lower,
        "required_positive_bootstrap_lower_bound_horizon_count": int(
            policy["minimum_horizons_with_positive_bootstrap_lower_bound"]
        ),
        "positive_strong_bull_minus_strong_bear_spread_horizon_count": positive_spread,
        "required_positive_strong_bull_minus_strong_bear_spread_horizon_count": int(
            policy[
                "minimum_horizons_with_positive_strong_bull_minus_strong_bear_median_delta_rank"
            ]
        ),
    }
    gate["status"] = (
        "PASS"
        if positive_lower >= gate["required_positive_bootstrap_lower_bound_horizon_count"]
        and positive_spread
        >= gate["required_positive_strong_bull_minus_strong_bear_spread_horizon_count"]
        else "FAIL"
    )
    return {
        "schema_version": "moatrader-evidence-future-momentum-lead-lag-summary-v2/1",
        "status": "FUTURE_MOMENTUM_LEAD_LAG_EVALUATED",
        "panel_observation_count": len(panel),
        "horizons": horizons,
        "lead_lag_gate": gate,
        "causal_claim_allowed": False,
        "ranking_output_produced": False,
        "future_returns_used_as_ranking": False,
    }, monthly


def run_future_momentum_lead_lag(
    *,
    workspace: Path,
    eri_build: Path,
    contract_path: Path,
    pre_outcome_build: Path,
    neutralization_build: Path,
    marcap_files: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production future-momentum lead-lag test requires a clean worktree")
    contract = _contract_payload(contract_path, eri_build=eri_build)
    neutral_stage_path = neutralization_build / "stage-status.json"
    neutral_summary_path = neutralization_build / "short-momentum-neutralization-summary.json"
    neutral_stage = _read_json(neutral_stage_path)
    neutral_summary = _read_json(neutral_summary_path)
    authorization = evaluate_lead_lag_authorization(neutral_summary, contract)
    if not (
        neutral_stage.get("status") == "SHORT_MOMENTUM_NEUTRALIZATION_COMPLETE"
        and neutral_stage.get("lead_lag_stage_authorized") is True
        and neutral_stage.get("forward_price_returns_opened") is False
        and authorization["status"] == "PASS"
    ):
        raise ValueError("lead-lag gate failed; forward price returns must remain closed")
    feature_path = eri_build / "features-with-frozen-expectations-pre-outcome.jsonl"
    controls_path = pre_outcome_build / "momentum-controls-pre-outcome.jsonl"
    label_ids = {
        str(item["observation_id"])
        for item in _read_records(eri_build / "future-eri-labels.jsonl")
    }
    features = [
        item for item in _read_records(feature_path) if str(item["observation_id"]) in label_ids
    ]
    controls = _read_records(controls_path)
    tickers = {str(item["issuer_id"]).zfill(6) for item in features}
    sessions = _load_sessions(marcap_files)
    returns_by_ticker, marcap_hashes = _load_return_maps(marcap_files, tickers)
    panel = _prepare_lead_lag_panel(
        features=features,
        controls=controls,
        returns_by_ticker=returns_by_ticker,
        sessions=sessions,
    )
    summary, monthly = _lead_lag_results(panel, contract)
    _verify_sources_unchanged(marcap_hashes)
    output.mkdir(parents=True, exist_ok=True)
    rows_path = output / "future-momentum-lead-lag-observations.jsonl"
    monthly_path = output / "monthly-future-momentum-lead-lag.jsonl"
    summary_path = output / "future-momentum-lead-lag-summary.json"
    export = panel.replace({np.nan: None}).to_dict(orient="records")
    _write_jsonl(rows_path, export)
    _write_jsonl(monthly_path, monthly)
    _write_json(summary_path, summary)
    status = {
        "schema_version": "moatrader-future-momentum-lead-lag-stage-v2/1",
        "status": "FUTURE_MOMENTUM_LEAD_LAG_COMPLETE",
        "git_commit": commit,
        "worktree_dirty": False,
        "panel_observation_count": len(panel),
        "forward_price_returns_opened": True,
        "future_returns_used_as_signal": False,
        "future_returns_used_as_ranking": False,
        "ranking_output_produced": False,
        "source_files_modified": False,
    }
    stage_path = output / "stage-status.json"
    _write_json(stage_path, status)
    _write_json(
        output / "build-manifest.json",
        {
            **status,
            "contract_sha256": sha256_file(contract_path),
            "neutral_stage_sha256": sha256_file(neutral_stage_path),
            "neutral_summary_sha256": sha256_file(neutral_summary_path),
            "feature_rows_sha256": sha256_file(feature_path),
            "controls_sha256": sha256_file(controls_path),
            "marcap_source_hashes": marcap_hashes,
            "output_hashes": {
                "rows": sha256_file(rows_path),
                "monthly": sha256_file(monthly_path),
                "summary": sha256_file(summary_path),
                "stage": sha256_file(stage_path),
            },
        },
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run preregistered short-Momentum neutralization and gated lead-lag tests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser("pre-outcome")
    pre.add_argument("--workspace", type=Path, default=Path.cwd())
    pre.add_argument("--eri-build", type=Path, required=True)
    pre.add_argument("--contract-path", type=Path, required=True)
    pre.add_argument("--marcap-files", type=Path, nargs="+", required=True)
    pre.add_argument("--output", type=Path, required=True)
    neutral = subparsers.add_parser("eri-neutralization")
    neutral.add_argument("--workspace", type=Path, default=Path.cwd())
    neutral.add_argument("--eri-build", type=Path, required=True)
    neutral.add_argument("--contract-path", type=Path, required=True)
    neutral.add_argument("--pre-outcome-build", type=Path, required=True)
    neutral.add_argument("--output", type=Path, required=True)
    lead = subparsers.add_parser("lead-lag")
    lead.add_argument("--workspace", type=Path, default=Path.cwd())
    lead.add_argument("--eri-build", type=Path, required=True)
    lead.add_argument("--contract-path", type=Path, required=True)
    lead.add_argument("--pre-outcome-build", type=Path, required=True)
    lead.add_argument("--neutralization-build", type=Path, required=True)
    lead.add_argument("--marcap-files", type=Path, nargs="+", required=True)
    lead.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = vars(args)
    command = values.pop("command")
    if command == "pre-outcome":
        result = prepare_momentum_controls_pre_outcome(**values)
    elif command == "eri-neutralization":
        result = run_eri_momentum_neutralization(**values)
    else:
        result = run_future_momentum_lead_lag(**values)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
