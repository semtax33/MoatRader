from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from moatrader.backtest.universe_corrected import (
    rank_normal_score,
    residualize_cross_section,
    spearman_ic,
)
from moatrader.expectations.historical_evidence import sha256_file
from scripts.prepare_historical_evidence_index_neutral_controls_v2 import (
    FACTOR_CONTROL_FIELDS_V2,
    FACTOR_DEFINITIONS_V2,
)
from scripts.run_historical_evidence_index_value_neutralization_v2 import (
    _inference,
    _read_json,
    _read_records,
    _validate_eri_gate,
    _write_json,
    _write_jsonl,
)


@dataclass(frozen=True)
class FactorSpecV2:
    key: str
    label: str
    controls: tuple[str, ...]


FACTOR_SPECS_V2 = (
    FactorSpecV2("momentum", "Momentum 12-1", ("factor_momentum_12_1",)),
    FactorSpecV2("growth", "Revenue Growth", ("factor_revenue_growth_yoy",)),
    FactorSpecV2(
        "quality",
        "Quality ROA+CFO-Leverage",
        ("factor_quality_roa_cfo_leverage",),
    ),
    FactorSpecV2(
        "momentum_growth_quality",
        "Momentum + Growth + Quality",
        (
            "factor_momentum_12_1",
            "factor_revenue_growth_yoy",
            "factor_quality_roa_cfo_leverage",
        ),
    ),
    FactorSpecV2(
        "analyst_eps_forecast",
        "Analyst forward EPS yield",
        ("factor_analyst_forward_eps_yield",),
    ),
    FactorSpecV2(
        "analyst_eps_revision",
        "Analyst EPS 30d revision",
        ("factor_analyst_eps_revision_30d",),
    ),
)


def _assert_factor_rows_are_outcome_blind(rows: list[dict[str, Any]]) -> None:
    prohibited = ("future_eri", "future_return", "forward_return", "actual_market_price")
    for row in rows:
        for key in row:
            if any(fragment in str(key).casefold() for fragment in prohibited):
                raise ValueError(f"factor-control input contains downstream field: {key}")


def _validate_manifest(
    *, manifest: dict[str, Any], factor_input: Path, eri_build: Path
) -> None:
    feature_path = eri_build / "features-with-frozen-expectations-pre-outcome.jsonl"
    labels_path = eri_build / "future-eri-labels.jsonl"
    checks = {
        "schema": manifest.get("schema_version")
        == "moatrader-evidence-index-factor-controls-v2/1",
        "status": manifest.get("status") == "V2_FACTOR_CONTROLS_PREPARED_AFTER_ERI_GATE",
        "input": manifest.get("factor_input_sha256") == sha256_file(factor_input),
        "eri_stage": manifest.get("eri_stage_status_sha256")
        == sha256_file(eri_build / "stage-status.json"),
        "eri_build": manifest.get("eri_build_manifest_sha256")
        == sha256_file(eri_build / "build-manifest.json"),
        "features": manifest.get("feature_input_sha256") == sha256_file(feature_path),
        "labels": manifest.get("future_eri_labels_sha256") == sha256_file(labels_path),
        "pit": manifest.get("point_in_time_at_signal_verified") is True,
        "availability": manifest.get("factor_available_no_later_than_signal_verified") is True,
        "outcome_blind": manifest.get("future_eri_used_to_construct_factor_controls") is False,
        "return_closed": manifest.get("return_data_opened") is False,
        "read_only": manifest.get("source_files_read_only") is True,
        "unmodified": manifest.get("source_files_modified") is False,
        "integrity": manifest.get("source_integrity_verification_status")
        == "PASS_NO_SOURCE_MUTATION",
        "fields": manifest.get("control_fields") == list(FACTOR_CONTROL_FIELDS_V2),
        "definitions": manifest.get("control_definitions") == FACTOR_DEFINITIONS_V2,
        "ranking": manifest.get("ranking_policy") == "NO_FACTOR_BASED_RANKING",
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"invalid factor-control manifest fields: {failed}")


def _prepare_panel(
    *, features: list[dict[str, Any]], labels: list[dict[str, Any]], factors: list[dict[str, Any]]
) -> pd.DataFrame:
    feature_by_id = {str(row["observation_id"]): row for row in features}
    label_by_id = {str(row["observation_id"]): row for row in labels}
    factor_by_id = {str(row["observation_id"]): row for row in factors}
    if len(feature_by_id) != len(features) or len(label_by_id) != len(labels):
        raise ValueError("ERI feature/label IDs must be unique")
    if len(factor_by_id) != len(factors) or set(factor_by_id) != set(label_by_id):
        raise ValueError("factor controls must explicitly cover every ERI label")
    _assert_factor_rows_are_outcome_blind(factors)
    rows: list[dict[str, Any]] = []
    for observation_id in sorted(label_by_id):
        feature = feature_by_id[observation_id]
        factor = factor_by_id[observation_id]
        signal = pd.Timestamp(feature["signal_timestamp"])
        factor_signal = pd.Timestamp(factor["signal_timestamp"])
        available = pd.Timestamp(factor["factor_available_at"])
        if signal.tzinfo is None or signal != factor_signal or available > signal:
            raise ValueError("factor-control PIT timestamp mismatch")
        row = {
            "observation_id": observation_id,
            "issuer_id": str(feature["issuer_id"]),
            "signal_month": signal.strftime("%Y-%m"),
            "full_evidence_index": float(feature["full_evidence_index"]),
            "future_eri": float(label_by_id[observation_id]["future_eri"]),
        }
        for field in FACTOR_CONTROL_FIELDS_V2:
            raw = factor.get(field)
            value = float(raw) if raw is not None else np.nan
            if not pd.isna(value) and not math.isfinite(value):
                raise ValueError(f"factor value must be finite or null: {field}")
            row[field] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _exposure_diagnostics(
    sample: pd.DataFrame, *, residual_column: str, controls: Sequence[str]
) -> dict[str, Any]:
    target = rank_normal_score(sample["full_evidence_index"])
    residual = pd.to_numeric(sample[residual_column], errors="coerce")
    ranked_controls = pd.DataFrame(
        {field: rank_normal_score(sample[field]) for field in controls}, index=sample.index
    )
    valid = pd.concat([target.rename("target"), residual.rename("residual"), ranked_controls], axis=1).dropna()
    if len(valid) <= len(controls) + 2:
        return {"control_exposure_r_squared": None, "max_abs_post_control_spearman": None}
    x = np.column_stack(
        [np.ones(len(valid)), valid[list(controls)].to_numpy(dtype=float)]
    )
    y = valid["target"].to_numpy(dtype=float)
    fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    total = float(np.sum((y - y.mean()) ** 2))
    unexplained = float(np.sum((y - fitted) ** 2))
    correlations = [
        abs(float(valid["residual"].corr(valid[field], method="spearman")))
        for field in controls
    ]
    return {
        "control_exposure_r_squared": 1.0 - unexplained / total if total > 0 else None,
        "max_abs_post_control_spearman": max(correlations) if correlations else None,
    }


def _monthly_results(
    panel: pd.DataFrame, *, minimum_monthly_observations: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal_month, month in panel.groupby("signal_month", sort=True):
        for spec in FACTOR_SPECS_V2:
            columns = ["full_evidence_index", "future_eri", *spec.controls]
            sample = month[columns].dropna().copy()
            base = {
                "signal_month": signal_month,
                "factor": spec.key,
                "factor_label": spec.label,
                "controls": list(spec.controls),
                "n": len(sample),
                "same_sample_raw_and_neutral": True,
            }
            if len(sample) < max(minimum_monthly_observations, len(spec.controls) + 3):
                rows.append({**base, "status": "INSUFFICIENT_MONTHLY_OBSERVATIONS"})
                continue
            sample["neutral_full_evidence_index"] = residualize_cross_section(
                sample,
                target="full_evidence_index",
                numeric_controls=spec.controls,
            )
            complete = sample.dropna(subset=["neutral_full_evidence_index"])
            raw_ic = spearman_ic(complete, "full_evidence_index", "future_eri")
            neutral_ic = spearman_ic(complete, "neutral_full_evidence_index", "future_eri")
            diagnostics = _exposure_diagnostics(
                complete,
                residual_column="neutral_full_evidence_index",
                controls=spec.controls,
            )
            rows.append(
                {
                    **base,
                    "status": (
                        "EVALUATED_SAME_SAMPLE"
                        if math.isfinite(raw_ic) and math.isfinite(neutral_ic)
                        else "INSUFFICIENT_VARIATION"
                    ),
                    "n": len(complete),
                    "raw_ic": raw_ic,
                    "neutral_ic": neutral_ic,
                    "delta_ic": neutral_ic - raw_ic,
                    **diagnostics,
                }
            )
    return rows


def _summary(
    panel: pd.DataFrame,
    monthly: list[dict[str, Any]],
    *,
    hac_lag_months: int,
    block_length_months: int,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    factors: dict[str, Any] = {}
    for spec in FACTOR_SPECS_V2:
        selected = [
            row
            for row in monthly
            if row["factor"] == spec.key and row["status"] == "EVALUATED_SAME_SAMPLE"
        ]
        inference = {
            name: _inference(
                [float(row[name]) for row in selected],
                hac_lag_months=hac_lag_months,
                block_length_months=block_length_months,
                bootstrap_repetitions=bootstrap_repetitions,
                bootstrap_seed=bootstrap_seed,
            )
            for name in ("raw_ic", "neutral_ic", "delta_ic")
        }
        raw_mean = inference["raw_ic"]["newey_west"]["mean"]
        neutral_mean = inference["neutral_ic"]["newey_west"]["mean"]
        retention = (
            float(neutral_mean) / float(raw_mean)
            if raw_mean is not None
            and neutral_mean is not None
            and math.isfinite(float(raw_mean))
            and abs(float(raw_mean)) > 1e-12
            else None
        )
        complete_count = int(panel[list(spec.controls)].notna().all(axis=1).sum())
        factors[spec.key] = {
            "factor_label": spec.label,
            "controls": list(spec.controls),
            "panel_observation_count": len(panel),
            "complete_control_observation_count": complete_count,
            "valid_month_count": len(selected),
            "same_sample_raw_and_neutral": True,
            "raw_ic": inference["raw_ic"],
            "neutral_ic": inference["neutral_ic"],
            "delta_ic": inference["delta_ic"],
            "signed_ic_retention_ratio": retention,
            "absolute_ic_attenuation": (
                1.0 - abs(float(neutral_mean)) / abs(float(raw_mean))
                if retention is not None and neutral_mean is not None
                else None
            ),
            "mean_control_exposure_r_squared": (
                float(np.mean([row["control_exposure_r_squared"] for row in selected]))
                if selected
                else None
            ),
            "max_abs_post_control_spearman": (
                max(float(row["max_abs_post_control_spearman"]) for row in selected)
                if selected
                else None
            ),
        }
    return {
        "schema_version": "moatrader-evidence-index-factor-neutralization-summary-v2/1",
        "status": "EVALUATED_PARALLEL_FACTOR_SENSITIVITIES",
        "signal": "FULL_EVIDENCE_INDEX",
        "outcome": "FUTURE_ERI_T63",
        "panel_observation_count": len(panel),
        "signal_month_count": int(panel["signal_month"].nunique()),
        "ranking_output_produced": False,
        "same_sample_policy": "PER_FACTOR_MONTH_IDENTICAL_COMPLETE_CASE_RAW_AND_NEUTRAL",
        "factors": factors,
    }


def run_factor_neutralization_v2(
    *,
    eri_build: Path,
    factor_input: Path,
    factor_manifest: Path,
    output: Path,
    minimum_monthly_observations: int = 5,
    hac_lag_months: int = 3,
    block_length_months: int = 4,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    gate = _validate_eri_gate(eri_build=eri_build, output=output)
    if isinstance(gate, dict):
        return gate
    eri_stage, _eri_manifest, feature_path, labels_path = gate
    manifest = _read_json(factor_manifest)
    _validate_manifest(manifest=manifest, factor_input=factor_input, eri_build=eri_build)
    input_hashes = {
        "eri_stage": sha256_file(eri_build / "stage-status.json"),
        "eri_build": sha256_file(eri_build / "build-manifest.json"),
        "features": sha256_file(feature_path),
        "labels": sha256_file(labels_path),
        "factor_manifest": sha256_file(factor_manifest),
        "factor_input": sha256_file(factor_input),
    }
    panel = _prepare_panel(
        features=_read_records(feature_path),
        labels=_read_records(labels_path),
        factors=_read_records(factor_input),
    )
    monthly = _monthly_results(
        panel, minimum_monthly_observations=minimum_monthly_observations
    )
    summary = _summary(
        panel,
        monthly,
        hac_lag_months=hac_lag_months,
        block_length_months=block_length_months,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    )
    monthly_path = output / "monthly-factor-neutralization.jsonl"
    summary_path = output / "factor-neutralization-summary.json"
    _write_jsonl(monthly_path, monthly)
    _write_json(summary_path, summary)
    after = {
        "eri_stage": sha256_file(eri_build / "stage-status.json"),
        "eri_build": sha256_file(eri_build / "build-manifest.json"),
        "features": sha256_file(feature_path),
        "labels": sha256_file(labels_path),
        "factor_manifest": sha256_file(factor_manifest),
        "factor_input": sha256_file(factor_input),
    }
    if input_hashes != after:
        raise RuntimeError("an ERI or factor input changed during neutralization")
    evaluated = sum(int(item["valid_month_count"] > 0) for item in summary["factors"].values())
    status = {
        "schema_version": "moatrader-evidence-index-factor-neutralization-stage-v2/1",
        "status": (
            "V2_FACTOR_NEUTRALIZATION_COMPLETE"
            if evaluated == len(FACTOR_SPECS_V2)
            else "V2_FACTOR_NEUTRALIZATION_PARTIAL_COVERAGE"
        ),
        "eri_primary_result_status": eri_stage["status"],
        "panel_observation_count": len(panel),
        "factor_test_count": len(FACTOR_SPECS_V2),
        "evaluated_factor_test_count": evaluated,
        "return_data_opened": False,
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run PIT Momentum, Growth, Quality, joint, and analyst EPS neutral tests."
    )
    parser.add_argument("--eri-build", type=Path, required=True)
    parser.add_argument("--factor-input", type=Path, required=True)
    parser.add_argument("--factor-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-monthly-observations", type=int, default=5)
    parser.add_argument("--hac-lag-months", type=int, default=3)
    parser.add_argument("--block-length-months", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    result = run_factor_neutralization_v2(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
