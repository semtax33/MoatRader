from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from moatrader.expectations.historical_evidence import sha256_file
from scripts.run_historical_evidence_index_value_neutralization_v2 import (
    NEUTRALIZER_PRIORITY_V2,
    VALUE_METRIC_FIELDS_V2,
    VALUE_METRIC_SPECS_V2,
    run_evidence_index_value_neutralization_v2,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def test_value_stage_does_not_open_value_paths_before_eri_is_complete(
    tmp_path: Path,
) -> None:
    eri_build = tmp_path / "eri"
    eri_build.mkdir()

    result = run_evidence_index_value_neutralization_v2(
        eri_build=eri_build,
        value_input=tmp_path / "must-not-be-opened-value.jsonl",
        value_manifest=tmp_path / "must-not-be-opened-manifest.json",
        output=tmp_path / "result",
        bootstrap_repetitions=10,
    )

    assert result["status"] == "BLOCKED_ERI_STAGE_OR_BUILD_MANIFEST_MISSING"
    assert result["eri_labels_opened"] is False
    assert result["value_manifest_opened"] is False
    assert result["value_data_opened"] is False
    assert result["return_data_opened"] is False
    assert result["future_eri_used_as_signal"] is False
    assert result["future_eri_used_as_ranking"] is False
    assert result["neutralizer_priority"] == NEUTRALIZER_PRIORITY_V2
    assert result["per_pbr_joint_primary"] is False


def _authorized_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    eri_build = tmp_path / "eri"
    eri_build.mkdir()
    features: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    values: list[dict[str, object]] = []
    for month_index, month in enumerate((1, 2, 3), start=1):
        signal = datetime(2024, month, 20, 15, 30, tzinfo=timezone.utc)
        for issuer_index in range(8):
            observation_id = f"OBS:{month}:{issuer_index}"
            evidence_index = (issuer_index - 3.5) / 3.5
            future_eri = evidence_index + ((issuer_index * 3 + month_index) % 5) / 20
            features.append(
                {
                    "schema_version": "moatrader-evidence-index-future-eri-feature-v2/1",
                    "observation_id": observation_id,
                    "issuer_id": f"ISSUER:{issuer_index}",
                    "signal_timestamp": signal.isoformat(),
                    "full_evidence_index": evidence_index,
                    "outcome_value_used_as_signal": False,
                    "outcome_value_used_as_ranking": False,
                    "return_data_accessed": False,
                    "per_pbr_role": "NOT_USED",
                }
            )
            labels.append(
                {
                    "schema_version": "moatrader-future-eri-label-v1/1",
                    "observation_id": observation_id,
                    "horizon_trading_days": 63,
                    "future_eri": future_eri,
                    "return_data_accessed": False,
                }
            )
            value_row: dict[str, object] = {
                "schema_version": "moatrader-evidence-index-value-control-row-v2/1",
                "observation_id": observation_id,
                "signal_timestamp": signal.isoformat(),
                "value_available_at": (signal - timedelta(days=1)).isoformat(),
                "value_source_ids": [f"DART:{issuer_index}:{month}", f"KRX:{issuer_index}:{month}"],
            }
            for metric_index, field in enumerate(VALUE_METRIC_FIELDS_V2, start=1):
                value_row[field] = (
                    evidence_index * (0.35 + metric_index / 40)
                    + ((issuer_index * (metric_index + 2) + month_index) % 7) / 10
                )
            values.append(value_row)

    feature_path = eri_build / "features-with-frozen-expectations-pre-outcome.jsonl"
    labels_path = eri_build / "future-eri-labels.jsonl"
    _write_jsonl(feature_path, features)
    _write_jsonl(labels_path, labels)
    stage_path = eri_build / "stage-status.json"
    _write_json(
        stage_path,
        {
            "schema_version": "moatrader-historical-evidence-index-eri-stage-v2/1",
            "status": "FULL_PRIMARY_MECHANISM_PASSED",
            "outcome_vault_opened": True,
            "label_count": len(labels),
            "primary_endpoint": "FULL_EVIDENCE_INDEX_TO_FUTURE_ERI_T63",
            "future_eri_role": "DOWNSTREAM_OUTCOME_ONLY_NOT_SIGNAL_OR_RANKING",
            "future_eri_used_as_signal": False,
            "future_eri_used_as_ranking": False,
            "return_data_opened": False,
            "primary_ranking_policy": "NONE_MECHANISM_ONLY",
            "per_pbr_role": "NOT_USED",
            "value_neutralization_stage_authorized": True,
        },
    )
    build_path = eri_build / "build-manifest.json"
    _write_json(
        build_path,
        {
            "schema_version": "moatrader-historical-evidence-index-eri-build-v2/1",
            "stage_status_sha256": sha256_file(stage_path),
            "feature_input_sha256": sha256_file(feature_path),
            "future_eri_labels_sha256": sha256_file(labels_path),
            "return_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )

    value_path = tmp_path / "value-controls.jsonl"
    _write_jsonl(value_path, values)
    manifest_path = tmp_path / "value-controls-manifest.json"
    _write_json(
        manifest_path,
        {
            "schema_version": "moatrader-evidence-index-value-controls-v2/1",
            "status": "V2_VALUE_CONTROLS_PREPARED_AFTER_ERI_GATE",
            "value_input_sha256": sha256_file(value_path),
            "eri_stage_status_sha256": sha256_file(stage_path),
            "eri_build_manifest_sha256": sha256_file(build_path),
            "feature_input_sha256": sha256_file(feature_path),
            "future_eri_labels_sha256": sha256_file(labels_path),
            "point_in_time_at_signal_verified": True,
            "value_available_no_later_than_signal_verified": True,
            "future_eri_used_to_construct_value_controls": False,
            "return_data_opened": False,
            "source_files_read_only": True,
            "source_files_modified": False,
            "source_integrity_verification_status": "PASS_NO_SOURCE_MUTATION",
            "metric_fields": list(VALUE_METRIC_FIELDS_V2),
            "metric_orientation": {
                field: "HIGHER_IS_CHEAPER" for field in VALUE_METRIC_FIELDS_V2
            },
            "neutralizer_priority": NEUTRALIZER_PRIORITY_V2,
            "per_pbr_joint_primary": False,
            "per_pbr_primary_ranking": False,
            "ranking_policy": "NO_VALUE_BASED_RANKING",
        },
    )
    return eri_build, value_path, manifest_path


def test_parallel_value_neutralization_uses_same_samples_and_no_ranking(
    tmp_path: Path,
) -> None:
    eri_build, value_path, manifest_path = _authorized_inputs(tmp_path)
    protected_before = {
        "features": sha256_file(
            eri_build / "features-with-frozen-expectations-pre-outcome.jsonl"
        ),
        "labels": sha256_file(eri_build / "future-eri-labels.jsonl"),
        "value": sha256_file(value_path),
        "manifest": sha256_file(manifest_path),
    }

    output = tmp_path / "neutralization"
    result = run_evidence_index_value_neutralization_v2(
        eri_build=eri_build,
        value_input=value_path,
        value_manifest=manifest_path,
        output=output,
        minimum_monthly_observations=5,
        bootstrap_repetitions=100,
        bootstrap_seed=7,
    )

    assert result["status"] == "V2_VALUE_NEUTRALIZATION_COMPLETE_PARALLEL_SENSITIVITY"
    assert result["metric_count"] == len(VALUE_METRIC_SPECS_V2)
    assert result["evaluated_metric_count"] == len(VALUE_METRIC_SPECS_V2)
    assert result["neutralizer_priority"] == NEUTRALIZER_PRIORITY_V2
    assert result["per_pbr_joint_primary"] is False
    assert result["per_pbr_primary_ranking"] is False
    assert result["ranking_output_produced"] is False
    assert result["future_eri_used_as_signal"] is False
    assert result["future_eri_used_as_ranking"] is False
    assert result["return_data_opened"] is False
    summary = json.loads(
        (output / "value-neutralization-summary.json").read_text(encoding="utf-8")
    )
    assert set(summary["metrics"]) == {spec.key for spec in VALUE_METRIC_SPECS_V2}
    assert summary["neutralizer_priority"] == NEUTRALIZER_PRIORITY_V2
    assert summary["per_pbr_joint_primary"] is False
    assert summary["ranking_output_produced"] is False
    for metric in summary["metrics"].values():
        assert metric["metric_role"] == "PARALLEL_SENSITIVITY"
        assert metric["priority_rank"] is None
        assert metric["same_sample_raw_and_neutral"] is True
        assert metric["valid_month_count"] == 3
    monthly = [
        json.loads(line)
        for line in (output / "monthly-value-neutralization.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(monthly) == 3 * len(VALUE_METRIC_SPECS_V2)
    assert all(row["same_sample_raw_and_neutral"] is True for row in monthly)
    assert all(row["priority_rank"] is None for row in monthly)

    protected_after = {
        "features": sha256_file(
            eri_build / "features-with-frozen-expectations-pre-outcome.jsonl"
        ),
        "labels": sha256_file(eri_build / "future-eri-labels.jsonl"),
        "value": sha256_file(value_path),
        "manifest": sha256_file(manifest_path),
    }
    assert protected_after == protected_before
