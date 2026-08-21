from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from moatrader.backtest.universe_corrected import (
    moving_block_bootstrap_mean,
    newey_west_mean,
    rank_normal_score,
    residualize_cross_section,
    spearman_ic,
)
from moatrader.expectations.historical_evidence import sha256_file


@dataclass(frozen=True)
class ValueMetricSpecV2:
    key: str
    label: str
    field: str
    definition: str


# Every input is oriented so that a larger value means cheaper.  The metrics are
# deliberately parallel sensitivities: their order here is serialization order,
# not research or ranking priority.
VALUE_METRIC_SPECS_V2 = (
    ValueMetricSpecV2(
        "pbr_btm",
        "PBR (B/M)",
        "value_btm",
        "book equity / market capitalization",
    ),
    ValueMetricSpecV2(
        "per_earnings_yield",
        "PER (E/P)",
        "value_earnings_yield",
        "earnings / market capitalization",
    ),
    ValueMetricSpecV2(
        "p_fcf",
        "P/FCF (FCF/P)",
        "value_fcf_yield",
        "free cash flow / market capitalization",
    ),
    ValueMetricSpecV2(
        "psr",
        "PSR (Sales/P)",
        "value_sales_yield",
        "sales / market capitalization",
    ),
    ValueMetricSpecV2(
        "pcr",
        "PCR (CFO/P)",
        "value_cfo_yield",
        "operating cash flow / market capitalization",
    ),
    ValueMetricSpecV2(
        "ev_ebitda",
        "EV/EBITDA (EBITDA/EV)",
        "value_ebitda_ev_yield",
        "EBITDA / enterprise value",
    ),
    ValueMetricSpecV2(
        "ev_ebit",
        "EV/EBIT (EBIT/EV)",
        "value_ebit_ev_yield",
        "operating income / enterprise value",
    ),
    ValueMetricSpecV2(
        "por",
        "POR (Operating income/P)",
        "value_operating_income_yield",
        "operating income / market capitalization",
    ),
    ValueMetricSpecV2(
        "pgpr",
        "PGPR (Gross profit/P)",
        "value_gross_profit_yield",
        "gross profit / market capitalization",
    ),
    ValueMetricSpecV2(
        "rpr_prr_rnd",
        "RPR/PRR (R&D/P)",
        "value_rnd_yield",
        "research and development expense / market capitalization",
    ),
    ValueMetricSpecV2(
        "retained_earnings",
        "Retained earnings/P",
        "value_retained_earnings_yield",
        "retained earnings / market capitalization",
    ),
    ValueMetricSpecV2(
        "par_assets",
        "PAR (Assets/P)",
        "value_assets_yield",
        "total assets / market capitalization",
    ),
    ValueMetricSpecV2(
        "ncav",
        "NCAV/P",
        "value_ncav_yield",
        "net current asset value / market capitalization",
    ),
)

VALUE_METRIC_FIELDS_V2 = tuple(spec.field for spec in VALUE_METRIC_SPECS_V2)
NEUTRALIZER_PRIORITY_V2 = "NONE_PARALLEL_SENSITIVITY"


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _json_safe(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    raw = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(raw, list) or any(not isinstance(item, dict) for item in raw):
        raise ValueError(f"JSON object records required: {path}")
    return [dict(item) for item in raw]


def _blocked(output: Path, status: str, **extra: object) -> dict[str, Any]:
    payload = {
        "schema_version": "moatrader-evidence-index-value-neutralization-stage-v2/1",
        "status": status,
        "eri_stage_opened": False,
        "eri_labels_opened": False,
        "value_manifest_opened": False,
        "value_data_opened": False,
        "return_data_opened": False,
        "future_eri_role": "DOWNSTREAM_OUTCOME_ONLY_NOT_SIGNAL_OR_RANKING",
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "ranking_output_produced": False,
        "neutralizer_priority": NEUTRALIZER_PRIORITY_V2,
        "per_pbr_joint_primary": False,
        "per_pbr_primary_ranking": False,
        **extra,
    }
    _write_json(output / "stage-status.json", payload)
    return payload


def _aware_timestamp(value: object, *, field: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return timestamp


def _assert_no_downstream_fields(records: list[dict[str, Any]]) -> None:
    prohibited = (
        "future_eri",
        "future_return",
        "forward_return",
        "actual_market_price",
        "target_price",
        "counterfactual_value",
    )

    def inspect(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                normalized = str(key).casefold()
                if any(fragment in normalized for fragment in prohibited):
                    raise ValueError(
                        f"value-control input contains prohibited downstream field: {path}.{key}"
                    )
                inspect(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{path}[{index}]")

    inspect(records, "value_controls")


def _validate_eri_gate(
    *, eri_build: Path, output: Path
) -> tuple[dict[str, Any], dict[str, Any], Path, Path] | dict[str, Any]:
    stage_path = eri_build / "stage-status.json"
    build_path = eri_build / "build-manifest.json"
    if not stage_path.is_file() or not build_path.is_file():
        return _blocked(output, "BLOCKED_ERI_STAGE_OR_BUILD_MANIFEST_MISSING")
    stage = _read_json(stage_path)
    allowed_statuses = {
        "FULL_PRIMARY_MECHANISM_PASSED",
        "FULL_PRIMARY_MECHANISM_REJECTED_OR_INCONCLUSIVE",
    }
    valid_stage = (
        stage.get("status") in allowed_statuses
        and stage.get("outcome_vault_opened") is True
        and int(stage.get("label_count", 0)) > 0
        and stage.get("primary_endpoint")
        == "FULL_EVIDENCE_INDEX_TO_FUTURE_ERI_T63"
        and stage.get("future_eri_role")
        == "DOWNSTREAM_OUTCOME_ONLY_NOT_SIGNAL_OR_RANKING"
        and stage.get("future_eri_used_as_signal") is False
        and stage.get("future_eri_used_as_ranking") is False
        and stage.get("return_data_opened") is False
        and stage.get("primary_ranking_policy") == "NONE_MECHANISM_ONLY"
        and stage.get("per_pbr_role") == "NOT_USED"
        and stage.get("value_neutralization_stage_authorized") is True
    )
    if not valid_stage:
        return _blocked(
            output,
            "BLOCKED_ERI_MECHANISM_EVALUATION_NOT_COMPLETE",
            eri_stage_opened=True,
            eri_stage_status=stage.get("status"),
        )
    build = _read_json(build_path)
    feature_path = eri_build / "features-with-frozen-expectations-pre-outcome.jsonl"
    labels_path = eri_build / "future-eri-labels.jsonl"
    valid_build = (
        build.get("schema_version")
        == "moatrader-historical-evidence-index-eri-build-v2/1"
        and build.get("stage_status_sha256") == sha256_file(stage_path)
        and feature_path.is_file()
        and labels_path.is_file()
        and build.get("feature_input_sha256") == sha256_file(feature_path)
        and build.get("future_eri_labels_sha256") == sha256_file(labels_path)
        and build.get("return_data_opened") is False
        and build.get("per_pbr_role") == "NOT_USED"
    )
    if not valid_build:
        return _blocked(
            output,
            "BLOCKED_ERI_ARTIFACT_LINEAGE_MISMATCH",
            eri_stage_opened=True,
            eri_stage_status=stage.get("status"),
        )
    return stage, build, feature_path, labels_path


def _validate_value_manifest(
    *,
    manifest: dict[str, Any],
    value_input: Path,
    eri_stage_path: Path,
    eri_build_path: Path,
    feature_path: Path,
    labels_path: Path,
) -> None:
    expected_orientation = {field: "HIGHER_IS_CHEAPER" for field in VALUE_METRIC_FIELDS_V2}
    checks = {
        "schema_version": manifest.get("schema_version")
        == "moatrader-evidence-index-value-controls-v2/1",
        "status": manifest.get("status")
        == "V2_VALUE_CONTROLS_PREPARED_AFTER_ERI_GATE",
        "value_input_sha256": manifest.get("value_input_sha256")
        == sha256_file(value_input),
        "eri_stage_status_sha256": manifest.get("eri_stage_status_sha256")
        == sha256_file(eri_stage_path),
        "eri_build_manifest_sha256": manifest.get("eri_build_manifest_sha256")
        == sha256_file(eri_build_path),
        "feature_input_sha256": manifest.get("feature_input_sha256")
        == sha256_file(feature_path),
        "future_eri_labels_sha256": manifest.get("future_eri_labels_sha256")
        == sha256_file(labels_path),
        "point_in_time": manifest.get("point_in_time_at_signal_verified") is True,
        "availability": manifest.get("value_available_no_later_than_signal_verified")
        is True,
        "outcome_blind_construction": manifest.get(
            "future_eri_used_to_construct_value_controls"
        )
        is False,
        "return_closed": manifest.get("return_data_opened") is False,
        "source_read_only": manifest.get("source_files_read_only") is True,
        "source_unmodified": manifest.get("source_files_modified") is False,
        "source_integrity": manifest.get("source_integrity_verification_status")
        == "PASS_NO_SOURCE_MUTATION",
        "metric_fields": manifest.get("metric_fields") == list(VALUE_METRIC_FIELDS_V2),
        "metric_orientation": manifest.get("metric_orientation") == expected_orientation,
        "neutralizer_priority": manifest.get("neutralizer_priority")
        == NEUTRALIZER_PRIORITY_V2,
        "per_pbr_joint_primary": manifest.get("per_pbr_joint_primary") is False,
        "per_pbr_primary_ranking": manifest.get("per_pbr_primary_ranking") is False,
        "ranking_policy": manifest.get("ranking_policy") == "NO_VALUE_BASED_RANKING",
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"invalid V2 value-control manifest fields: {failed}")


def _prepare_panel(
    *,
    features: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    values: list[dict[str, Any]],
) -> pd.DataFrame:
    feature_by_id: dict[str, dict[str, Any]] = {}
    for record in features:
        observation_id = str(record.get("observation_id", "")).strip()
        if not observation_id or observation_id in feature_by_id:
            raise ValueError("ERI feature observation IDs must be nonblank and unique")
        if (
            record.get("schema_version")
            != "moatrader-evidence-index-future-eri-feature-v2/1"
            or record.get("outcome_value_used_as_signal") is not False
            or record.get("outcome_value_used_as_ranking") is not False
            or record.get("return_data_accessed") is not False
            or record.get("per_pbr_role") != "NOT_USED"
        ):
            raise ValueError("invalid or contaminated V2 ERI feature row")
        _aware_timestamp(record.get("signal_timestamp"), field="feature.signal_timestamp")
        feature_by_id[observation_id] = record

    label_by_id: dict[str, dict[str, Any]] = {}
    for record in labels:
        observation_id = str(record.get("observation_id", "")).strip()
        if not observation_id or observation_id in label_by_id:
            raise ValueError("Future ERI label observation IDs must be nonblank and unique")
        if (
            record.get("schema_version") != "moatrader-future-eri-label-v1/1"
            or int(record.get("horizon_trading_days", 0)) != 63
            or record.get("return_data_accessed") is not False
        ):
            raise ValueError("invalid or return-contaminated Future ERI label row")
        future_eri = float(record["future_eri"])
        if not math.isfinite(future_eri):
            raise ValueError("Future ERI label must be finite")
        if observation_id not in feature_by_id:
            raise ValueError("Future ERI label has no sealed feature row")
        label_by_id[observation_id] = record

    _assert_no_downstream_fields(values)
    value_by_id: dict[str, dict[str, Any]] = {}
    for record in values:
        observation_id = str(record.get("observation_id", "")).strip()
        if not observation_id or observation_id in value_by_id:
            raise ValueError("value-control observation IDs must be nonblank and unique")
        if record.get("schema_version") != "moatrader-evidence-index-value-control-row-v2/1":
            raise ValueError("invalid V2 value-control row schema")
        if not all(field in record for field in VALUE_METRIC_FIELDS_V2):
            raise ValueError("each value-control row must explicitly carry every metric field")
        source_ids = record.get("value_source_ids")
        if not isinstance(source_ids, list) or not source_ids or any(
            not str(item).strip() for item in source_ids
        ):
            raise ValueError("value-control rows require nonblank source IDs")
        feature = feature_by_id.get(observation_id)
        if feature is None:
            raise ValueError("value-control row has no sealed feature row")
        feature_signal = _aware_timestamp(
            feature["signal_timestamp"], field="feature.signal_timestamp"
        )
        value_signal = _aware_timestamp(
            record.get("signal_timestamp"), field="value.signal_timestamp"
        )
        available_at = _aware_timestamp(
            record.get("value_available_at"), field="value.value_available_at"
        )
        if value_signal != feature_signal:
            raise ValueError("value-control signal timestamp does not match ERI feature")
        if available_at > feature_signal:
            raise ValueError("value control was not available at the signal timestamp")
        for field in VALUE_METRIC_FIELDS_V2:
            raw = record[field]
            if raw is not None and not math.isfinite(float(raw)):
                raise ValueError(f"value metric must be finite or null: {field}")
        value_by_id[observation_id] = record

    if set(value_by_id) != set(label_by_id):
        missing = sorted(set(label_by_id) - set(value_by_id))[:5]
        extra = sorted(set(value_by_id) - set(label_by_id))[:5]
        raise ValueError(
            "value-control panel must explicitly cover every ERI label "
            f"(missing={missing}, extra={extra})"
        )

    panel_rows: list[dict[str, Any]] = []
    for observation_id in sorted(label_by_id):
        feature = feature_by_id[observation_id]
        label = label_by_id[observation_id]
        value = value_by_id[observation_id]
        signal = _aware_timestamp(feature["signal_timestamp"], field="signal_timestamp")
        row: dict[str, Any] = {
            "observation_id": observation_id,
            "issuer_id": str(feature["issuer_id"]),
            "signal_timestamp": signal,
            "signal_month": signal.strftime("%Y-%m"),
            "full_evidence_index": float(feature["full_evidence_index"]),
            "future_eri": float(label["future_eri"]),
        }
        for field in VALUE_METRIC_FIELDS_V2:
            row[field] = float(value[field]) if value[field] is not None else np.nan
        panel_rows.append(row)
    return pd.DataFrame(panel_rows)


def _exposure_diagnostics(
    sample: pd.DataFrame, *, residual_column: str, value_field: str
) -> dict[str, float]:
    target = rank_normal_score(sample["full_evidence_index"])
    control = rank_normal_score(sample[value_field])
    residual = pd.to_numeric(sample[residual_column], errors="coerce")
    valid = pd.concat(
        [target.rename("target"), control.rename("control"), residual.rename("residual")],
        axis=1,
    ).dropna()
    if len(valid) < 5:
        return {
            "value_exposure_r_squared": float("nan"),
            "raw_value_spearman": float("nan"),
            "neutral_value_spearman": float("nan"),
            "rank_retention": float("nan"),
        }
    x = np.column_stack([np.ones(len(valid)), valid["control"].to_numpy(dtype=float)])
    y = valid["target"].to_numpy(dtype=float)
    fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
    total = float(np.sum((y - y.mean()) ** 2))
    unexplained = float(np.sum((y - fitted) ** 2))
    return {
        "value_exposure_r_squared": 1.0 - unexplained / total if total > 0 else float("nan"),
        "raw_value_spearman": float(valid["target"].corr(valid["control"], method="spearman")),
        "neutral_value_spearman": float(
            valid["residual"].corr(valid["control"], method="spearman")
        ),
        "rank_retention": float(valid["target"].corr(valid["residual"], method="spearman")),
    }


def _monthly_results(
    panel: pd.DataFrame, *, minimum_monthly_observations: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal_month, month in panel.groupby("signal_month", sort=True):
        for spec in VALUE_METRIC_SPECS_V2:
            sample = month[
                ["full_evidence_index", "future_eri", spec.field]
            ].dropna()
            base = {
                "signal_month": signal_month,
                "metric": spec.key,
                "metric_label": spec.label,
                "metric_field": spec.field,
                "metric_definition": spec.definition,
                "metric_orientation": "HIGHER_IS_CHEAPER",
                "metric_role": "PARALLEL_SENSITIVITY",
                "priority_rank": None,
                "n": len(sample),
                "same_sample_raw_and_neutral": True,
            }
            if len(sample) < minimum_monthly_observations:
                rows.append(
                    {
                        **base,
                        "status": "INSUFFICIENT_MONTHLY_OBSERVATIONS",
                        "raw_ic": None,
                        "neutral_ic": None,
                        "delta_ic": None,
                        "value_exposure_r_squared": None,
                        "raw_value_spearman": None,
                        "neutral_value_spearman": None,
                        "rank_retention": None,
                    }
                )
                continue
            sample = sample.copy()
            sample["neutral_full_evidence_index"] = residualize_cross_section(
                sample,
                target="full_evidence_index",
                numeric_controls=(spec.field,),
            )
            complete = sample.dropna(subset=["neutral_full_evidence_index"])
            raw_ic = spearman_ic(complete, "full_evidence_index", "future_eri")
            neutral_ic = spearman_ic(
                complete, "neutral_full_evidence_index", "future_eri"
            )
            diagnostics = _exposure_diagnostics(
                complete,
                residual_column="neutral_full_evidence_index",
                value_field=spec.field,
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


def _inference(
    values: Sequence[float],
    *,
    hac_lag_months: int,
    block_length_months: int,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return {
        "newey_west": newey_west_mean(clean, lag=hac_lag_months),
        "moving_block_bootstrap": moving_block_bootstrap_mean(
            clean,
            block_length=block_length_months,
            repetitions=bootstrap_repetitions,
            seed=bootstrap_seed,
        ),
    }


def _summary(
    panel: pd.DataFrame,
    monthly: list[dict[str, Any]],
    *,
    hac_lag_months: int,
    block_length_months: int,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for spec in VALUE_METRIC_SPECS_V2:
        selected = [
            row
            for row in monthly
            if row["metric"] == spec.key and row["status"] == "EVALUATED_SAME_SAMPLE"
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
        signed_retention = (
            float(neutral_mean) / float(raw_mean)
            if raw_mean is not None
            and neutral_mean is not None
            and math.isfinite(float(raw_mean))
            and math.isfinite(float(neutral_mean))
            and abs(float(raw_mean)) > 1e-12
            else float("nan")
        )
        metrics[spec.key] = {
            "metric_label": spec.label,
            "metric_field": spec.field,
            "metric_definition": spec.definition,
            "metric_orientation": "HIGHER_IS_CHEAPER",
            "metric_role": "PARALLEL_SENSITIVITY",
            "priority_rank": None,
            "panel_observation_count": len(panel),
            "metric_available_observation_count": int(panel[spec.field].notna().sum()),
            "valid_month_count": len(selected),
            "same_sample_raw_and_neutral": True,
            "raw_ic": inference["raw_ic"],
            "neutral_ic": inference["neutral_ic"],
            "delta_ic": inference["delta_ic"],
            "signed_ic_retention_ratio": signed_retention,
            "absolute_ic_attenuation": (
                1.0 - abs(float(neutral_mean)) / abs(float(raw_mean))
                if math.isfinite(signed_retention)
                else float("nan")
            ),
            "mean_value_exposure_r_squared": (
                float(np.mean([row["value_exposure_r_squared"] for row in selected]))
                if selected
                else float("nan")
            ),
            "mean_neutral_value_spearman": (
                float(np.mean([row["neutral_value_spearman"] for row in selected]))
                if selected
                else float("nan")
            ),
        }
    return {
        "schema_version": "moatrader-evidence-index-value-neutralization-summary-v2/1",
        "status": "EVALUATED_PARALLEL_VALUE_SENSITIVITIES",
        "outcome": "FUTURE_ERI_T63",
        "signal": "FULL_EVIDENCE_INDEX",
        "panel_observation_count": len(panel),
        "signal_month_count": int(panel["signal_month"].nunique()),
        "neutralizer_priority": NEUTRALIZER_PRIORITY_V2,
        "per_pbr_joint_primary": False,
        "per_pbr_primary_ranking": False,
        "ranking_output_produced": False,
        "same_sample_policy": (
            "FOR_EACH_METRIC_MONTH_COMPARE_RAW_AND_NEUTRAL_IC_ON_IDENTICAL_COMPLETE_CASES"
        ),
        "interpretation_policy": (
            "DESCRIPTIVE_MECHANISM_LINK_ATTENUATION_NOT_RETURN_ALPHA_OR_TRADING_RANKING"
        ),
        "inference": {
            "calendar_unit": "SIGNAL_MONTH",
            "newey_west_lag_months": hac_lag_months,
            "moving_block_length_months": block_length_months,
            "bootstrap_repetitions": bootstrap_repetitions,
            "bootstrap_seed": bootstrap_seed,
        },
        "metrics": metrics,
    }


def run_evidence_index_value_neutralization_v2(
    *,
    eri_build: Path,
    value_input: Path,
    value_manifest: Path,
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
    if minimum_monthly_observations < 5:
        raise ValueError("minimum_monthly_observations must be at least 5")
    if min(hac_lag_months, block_length_months, bootstrap_repetitions) < 1:
        raise ValueError("inference parameters must be positive")

    eri_gate = _validate_eri_gate(eri_build=eri_build, output=output)
    if isinstance(eri_gate, dict):
        return eri_gate
    eri_stage, _eri_manifest, feature_path, labels_path = eri_gate

    # The ERI artifacts are validated before the optional Value paths are inspected.
    features = _read_records(feature_path)
    labels = _read_records(labels_path)
    if len(labels) != int(eri_stage["label_count"]):
        return _blocked(
            output,
            "BLOCKED_ERI_LABEL_COUNT_MISMATCH",
            eri_stage_opened=True,
            eri_labels_opened=True,
        )

    if not value_manifest.is_file() or not value_input.is_file():
        raise FileNotFoundError("authorized Value stage requires both manifest and input")
    input_hashes_before = {
        "eri_stage_status": sha256_file(eri_build / "stage-status.json"),
        "eri_build_manifest": sha256_file(eri_build / "build-manifest.json"),
        "eri_features": sha256_file(feature_path),
        "future_eri_labels": sha256_file(labels_path),
        "value_manifest": sha256_file(value_manifest),
        "value_input": sha256_file(value_input),
    }
    manifest = _read_json(value_manifest)
    _validate_value_manifest(
        manifest=manifest,
        value_input=value_input,
        eri_stage_path=eri_build / "stage-status.json",
        eri_build_path=eri_build / "build-manifest.json",
        feature_path=feature_path,
        labels_path=labels_path,
    )
    values = _read_records(value_input)
    panel = _prepare_panel(features=features, labels=labels, values=values)
    monthly = _monthly_results(
        panel,
        minimum_monthly_observations=minimum_monthly_observations,
    )
    summary = _summary(
        panel,
        monthly,
        hac_lag_months=hac_lag_months,
        block_length_months=block_length_months,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    )
    monthly_path = output / "monthly-value-neutralization.jsonl"
    summary_path = output / "value-neutralization-summary.json"
    _write_jsonl(monthly_path, monthly)
    _write_json(summary_path, summary)

    input_hashes_after = {
        "eri_stage_status": sha256_file(eri_build / "stage-status.json"),
        "eri_build_manifest": sha256_file(eri_build / "build-manifest.json"),
        "eri_features": sha256_file(feature_path),
        "future_eri_labels": sha256_file(labels_path),
        "value_manifest": sha256_file(value_manifest),
        "value_input": sha256_file(value_input),
    }
    if input_hashes_before != input_hashes_after:
        raise RuntimeError("an ERI or Value input changed during neutralization")

    evaluated_metrics = sum(
        int(metric["valid_month_count"] > 0) for metric in summary["metrics"].values()
    )
    status = {
        "schema_version": "moatrader-evidence-index-value-neutralization-stage-v2/1",
        "status": (
            "V2_VALUE_NEUTRALIZATION_COMPLETE_PARALLEL_SENSITIVITY"
            if evaluated_metrics == len(VALUE_METRIC_SPECS_V2)
            else "V2_VALUE_NEUTRALIZATION_COMPLETE_PARTIAL_METRIC_COVERAGE"
        ),
        "eri_stage_opened": True,
        "eri_labels_opened": True,
        "value_manifest_opened": True,
        "value_data_opened": True,
        "return_data_opened": False,
        "future_eri_role": "DOWNSTREAM_OUTCOME_ONLY_NOT_SIGNAL_OR_RANKING",
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "ranking_output_produced": False,
        "signal": "FULL_EVIDENCE_INDEX",
        "outcome": "FUTURE_ERI_T63",
        "panel_observation_count": len(panel),
        "metric_count": len(VALUE_METRIC_SPECS_V2),
        "evaluated_metric_count": evaluated_metrics,
        "neutralizer_priority": NEUTRALIZER_PRIORITY_V2,
        "per_pbr_joint_primary": False,
        "per_pbr_primary_ranking": False,
        "per_pbr_role": "PARALLEL_SENSITIVITY_ONLY",
        "source_files_read_only": True,
        "source_files_modified": False,
        "inputs_unchanged_during_run": True,
    }
    stage_path = output / "stage-status.json"
    _write_json(stage_path, status)
    _write_json(
        output / "build-manifest.json",
        {
            "schema_version": "moatrader-evidence-index-value-neutralization-build-v2/1",
            "eri_primary_result_status": eri_stage["status"],
            "input_hashes": input_hashes_before,
            "output_hashes": {
                "monthly_value_neutralization": sha256_file(monthly_path),
                "value_neutralization_summary": sha256_file(summary_path),
                "stage_status": sha256_file(stage_path),
            },
            "metric_fields": list(VALUE_METRIC_FIELDS_V2),
            "neutralizer_priority": NEUTRALIZER_PRIORITY_V2,
            "per_pbr_joint_primary": False,
            "per_pbr_primary_ranking": False,
            "ranking_output_produced": False,
            "future_eri_used_as_signal": False,
            "future_eri_used_as_ranking": False,
            "return_data_opened": False,
            "source_files_modified": False,
        },
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "After the V2 t+63 ERI evaluation, compare Full Evidence Index IC before "
            "and after parallel PIT Value neutralizers without producing a ranking."
        )
    )
    parser.add_argument("--eri-build", type=Path, required=True)
    parser.add_argument("--value-input", type=Path, required=True)
    parser.add_argument("--value-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-monthly-observations", type=int, default=5)
    parser.add_argument("--hac-lag-months", type=int, default=3)
    parser.add_argument("--block-length-months", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    result = run_evidence_index_value_neutralization_v2(
        eri_build=args.eri_build,
        value_input=args.value_input,
        value_manifest=args.value_manifest,
        output=args.output,
        minimum_monthly_observations=args.minimum_monthly_observations,
        hac_lag_months=args.hac_lag_months,
        block_length_months=args.block_length_months,
        bootstrap_repetitions=args.bootstrap_repetitions,
        bootstrap_seed=args.bootstrap_seed,
    )
    print(json.dumps(_json_safe(result), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
