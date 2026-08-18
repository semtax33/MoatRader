from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from moatrader.experiments.contract import compute_contract_sha256
from moatrader.experiments.coverage import build_coverage_report
from moatrader.experiments.integrity import (
    audit_v6_integrity,
    ensure_new_experiment_output,
    sha256_file,
)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_coverage_report_partitions_valid_invalid_and_missing(tmp_path: Path) -> None:
    routing = tmp_path / "routing.csv"
    signals = tmp_path / "signals.csv"
    _write_csv(
        routing,
        [
            {
                "date": "2026-05-31",
                "ticker": "000001",
                "primary_method": "ECONOMIC_FCFF",
                "applicability_status": "ELIGIBLE",
                "missing_fields": "",
                "valuation_generated": 1,
            },
            {
                "date": "2026-05-31",
                "ticker": "000002",
                "primary_method": "SCENARIO_DCF",
                "applicability_status": "ELIGIBLE",
                "missing_fields": "",
                "valuation_generated": 1,
            },
            {
                "date": "2026-05-31",
                "ticker": "000003",
                "primary_method": "SOTP",
                "applicability_status": "INSUFFICIENT_DATA",
                "missing_fields": "segment_values;ownership_pct",
                "valuation_generated": 0,
            },
        ],
    )
    _write_csv(
        signals,
        [
            {
                "date": "2026-05-31",
                "ticker": "000001",
                "method": "ECONOMIC_FCFF",
                "rank_eligible": 1,
                "alpha_status": "VALID",
            },
            {
                "date": "2026-05-31",
                "ticker": "000002",
                "method": "SCENARIO_DCF",
                "rank_eligible": 0,
                "alpha_status": "INVALID_VALUATION",
            },
            {
                "date": "2026-05-31",
                "ticker": "000003",
                "method": "SOTP",
                "rank_eligible": 0,
                "alpha_status": "MODEL_NOT_APPLICABLE",
            },
        ],
    )

    report = build_coverage_report(routing_path=routing, signals_path=signals)

    assert report.row_count == 3
    assert report.router_eligible_count == 2
    assert report.valuation_generated_count == 2
    assert report.rank_eligible_count == 1
    assert report.invalid_valuation_count == 1
    assert report.model_not_applicable_count == 1
    assert report.missing_input_counts == {"ownership_pct": 1, "segment_values": 1}
    assert report.return_data_accessed is False


def test_v7_output_guard_rejects_v6_overlap_and_overwrite(tmp_path: Path) -> None:
    protected = tmp_path / "candidate-v6"
    protected.mkdir()
    with pytest.raises(ValueError, match="protected path"):
        ensure_new_experiment_output(
            protected / "child-v7", required_label="v7", protected_roots=[protected]
        )
    existing = tmp_path / "candidate-v7"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="must be new"):
        ensure_new_experiment_output(
            existing, required_label="v7", protected_roots=[protected]
        )


def test_v6_integrity_audit_reports_artifact_drift_without_writing(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("frozen\n", encoding="utf-8")
    stability = tmp_path / "stability-v6"
    stability.mkdir()
    artifact = stability / "signals.csv"
    artifact.write_text("frozen\n", encoding="utf-8")
    contract_dir = tmp_path / "contract-v6"
    contract_dir.mkdir()
    contract_path = contract_dir / "frozen-contract.json"
    payload = {
        "frozen_source_sha256": {"source.py": sha256_file(source)},
        "engineering_stability_sha256": {"signals.csv": sha256_file(artifact)},
    }
    payload["contract_sha256"] = compute_contract_sha256(payload)
    contract_path.write_text(json.dumps(payload), encoding="utf-8")

    pristine = audit_v6_integrity(
        repository_root=tmp_path,
        contract_path=contract_path,
        stability_directory=stability,
    )
    assert pristine.pristine_against_contract is True

    artifact.write_text("drifted\n", encoding="utf-8")
    drifted = audit_v6_integrity(
        repository_root=tmp_path,
        contract_path=contract_path,
        stability_directory=stability,
    )
    assert drifted.pristine_against_contract is False
    assert [item.path for item in drifted.engineering_artifact_mismatches] == [
        "stability-v6/signals.csv"
    ]
