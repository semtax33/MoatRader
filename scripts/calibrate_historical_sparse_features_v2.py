from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from moatrader.expectations.historical_evidence import sha256_file
from moatrader.expectations.historical_evidence_v2 import (
    HistoricalSparseEvidenceFeatureRowV2,
    SparseCoverageGatePolicyV2,
    calibrate_sparse_band_contract_v2,
    evaluate_sparse_coverage_gate_v2,
    seal_sparse_features_v2,
    sparse_band_diagnostics_v2,
    sparse_feature_coverage_report,
)


SEOUL = ZoneInfo("Asia/Seoul")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_features(path: Path) -> list[HistoricalSparseEvidenceFeatureRowV2]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            HistoricalSparseEvidenceFeatureRowV2.model_validate_json(line)
            for line in handle
            if line.strip()
        ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[HistoricalSparseEvidenceFeatureRowV2]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _validate_outcome_blind_manifest(
    path: Path,
    *,
    expected_status: str,
    description: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _read_json(path)
    if payload.get("status") != expected_status:
        raise ValueError(f"{description} has not passed: {payload.get('status')}")
    if payload.get("outcome_vault_opened", False) or payload.get("return_data_opened", False):
        raise ValueError(f"{description} is contaminated by downstream data")
    return payload


def calibrate_sparse_features(
    *,
    feature_build: Path,
    output: Path,
    freeze: bool = False,
    minimum_observed_axes: int | None = None,
    parser_validation_manifest: Path | None = None,
    abstention_audit_manifest: Path | None = None,
    contract_freeze_manifest: Path | None = None,
    coverage_gate_policy: Path | None = None,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    feature_path = feature_build / "sparse-features-all-pairs.jsonl"
    upstream_status_path = feature_build / "stage-status.json"
    for path in (feature_path, upstream_status_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    upstream = _read_json(upstream_status_path)
    if upstream.get("status") != "SPARSE_FEATURES_BUILT_AWAITING_OUTCOME_BLIND_CALIBRATION":
        raise ValueError("V2 sparse feature build is incomplete")
    if upstream.get("outcome_vault_opened", False) or upstream.get("return_data_opened", False):
        raise ValueError("V2 sparse feature build is contaminated by downstream data")

    rows = _read_features(feature_path)
    if len({row.observation_id for row in rows}) != len(rows):
        raise ValueError("V2 sparse feature observation IDs must be unique")
    output.mkdir(parents=True, exist_ok=True)
    coverage = sparse_feature_coverage_report(rows)
    coverage["freeze_requested"] = freeze
    coverage["minimum_observed_axes_selected"] = minimum_observed_axes
    _write_json(output / "outcome-blind-coverage-diagnostics.json", coverage)

    base_status: dict[str, Any] = {
        "schema_version": "moatrader-historical-sparse-calibration-stage-v2/1",
        "status": "OUTCOME_BLIND_DIAGNOSTICS_COMPLETE_AWAITING_EXPLICIT_FREEZE",
        "feature_count": len(rows),
        "minimum_observed_axes": minimum_observed_axes,
        "feature_dataset_sealed": False,
        "outcome_stage_authorized": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "minimum_observed_axes_auto_selected": False,
        "band_rules_outcome_blind": True,
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
        "input_hashes": {
            "sparse_features": sha256_file(feature_path),
            "upstream_stage": sha256_file(upstream_status_path),
        },
    }
    if not freeze:
        _write_json(output / "stage-status.json", base_status)
        return base_status
    if minimum_observed_axes is None:
        raise ValueError("freeze requires an explicit --minimum-observed-axes; it is never inferred")
    if (
        parser_validation_manifest is None
        or abstention_audit_manifest is None
        or contract_freeze_manifest is None
        or coverage_gate_policy is None
    ):
        raise ValueError(
            "freeze requires dual LOCKED validation, abstention audit, contract freeze, and coverage policy"
        )
    parser_validation = _validate_outcome_blind_manifest(
        parser_validation_manifest,
        expected_status="V2_LOCKED_TESTS_PASSED",
        description="dual Natural and Balanced V2 parser LOCKED tests",
    )
    if not parser_validation.get("natural_frequency_gate_passed", False):
        raise ValueError("V2 parser validation lacks the Natural-frequency LOCKED gate")
    if not parser_validation.get("directional_strata_gate_passed", False):
        raise ValueError("balanced V2 parser LOCKED test lacks directional strata coverage")
    abstention_audit = _validate_outcome_blind_manifest(
        abstention_audit_manifest,
        expected_status="V2_ABSTENTION_AUDIT_PASSED",
        description="V2 abstention audit",
    )
    if not abstention_audit.get("upstream_extraction_gate_passed", False):
        raise ValueError("upstream retrieval/table/period failures exceed the audit threshold")
    contract_freeze = _validate_outcome_blind_manifest(
        contract_freeze_manifest,
        expected_status="V2_PRE_OUTCOME_CONTRACT_FROZEN",
        description="V2 pre-outcome measurement contract",
    )
    if contract_freeze.get("dry_run_only", True) or contract_freeze.get("worktree_dirty", True):
        raise ValueError("production pre-outcome seal requires a clean committed contract freeze")
    gate_policy = SparseCoverageGatePolicyV2.model_validate_json(
        coverage_gate_policy.read_text(encoding="utf-8")
    )
    upstream_hashes = upstream.get("input_hashes", {})
    if upstream_hashes.get("parser_validation_manifest") != sha256_file(
        parser_validation_manifest
    ):
        raise ValueError("feature build did not use this V2 parser validation manifest")
    if upstream_hashes.get("contract_freeze_manifest") != sha256_file(
        contract_freeze_manifest
    ):
        raise ValueError("feature build did not use this V2 contract freeze")
    if contract_freeze.get("parser_freeze_sha256") != parser_validation.get(
        "parser_freeze_sha256"
    ):
        raise ValueError("contract freeze and dual LOCKED validation use different parser freezes")
    if not upstream.get("applicability_contract_complete", False):
        raise ValueError("feature freeze requires complete outcome-blind PIT applicability decisions")
    if not upstream.get("deterministic_pit_priority_applied", False):
        raise ValueError("feature freeze requires deterministic PIT evidence priority")

    contract = calibrate_sparse_band_contract_v2(
        rows,
        minimum_observed_axes=minimum_observed_axes,
        minimum_rows_per_band=gate_policy.minimum_rows_per_band,
    )
    contract_path = output / "sparse-breadth-band-contract.json"
    _write_json(contract_path, contract.model_dump(mode="json"))
    base_status.update(
        band_contract_sha256=sha256_file(contract_path),
        five_band_counts={key.value: value for key, value in contract.band_counts.items()},
        all_five_bands_sufficient=contract.all_bands_sufficient,
    )
    band_diagnostics = sparse_band_diagnostics_v2(rows, band_contract=contract)
    band_diagnostics_path = output / "sparse-band-diagnostics.json"
    _write_json(band_diagnostics_path, band_diagnostics)
    coverage_gate = evaluate_sparse_coverage_gate_v2(
        band_diagnostics,
        policy=gate_policy,
    )
    coverage_gate_path = output / "measurement-coverage-gate.json"
    _write_json(coverage_gate_path, coverage_gate)
    base_status.update(
        corr_abs_signed_breadth_coverage=band_diagnostics[
            "corr_abs_signed_breadth_coverage"
        ],
        corr_abs_signed_breadth_nobs=band_diagnostics[
            "corr_abs_signed_breadth_nobs"
        ],
        measurement_coverage_gate_passed=coverage_gate["gate_passed"],
        band_diagnostics_sha256=sha256_file(band_diagnostics_path),
        coverage_gate_report_sha256=sha256_file(coverage_gate_path),
    )
    if not contract.all_bands_sufficient or not coverage_gate["gate_passed"]:
        base_status["status"] = "V2_BAND_COVERAGE_INSUFFICIENT_OUTCOME_GATE_CLOSED"
        _write_json(output / "stage-status.json", base_status)
        return base_status

    eligible = sorted(
        (row for row in rows if row.signed_score_axis_count >= minimum_observed_axes),
        key=lambda row: row.observation_id,
    )
    eligible_path = output / "features-pre-outcome.jsonl"
    _write_jsonl(eligible_path, eligible)
    seal = seal_sparse_features_v2(
        eligible,
        band_contract=contract,
        parser_validation_manifest_sha256=sha256_file(parser_validation_manifest),
        contract_freeze_manifest_sha256=sha256_file(contract_freeze_manifest),
        abstention_audit_manifest_sha256=sha256_file(abstention_audit_manifest),
        coverage_gate_report_sha256=sha256_file(coverage_gate_path),
        sealed_at=datetime.now(SEOUL),
    )
    seal_path = output / "sparse-feature-seal.json"
    _write_json(seal_path, seal.model_dump(mode="json"))
    pre_outcome_manifest = {
        "schema_version": "moatrader-v2-pre-outcome-manifest/2",
        "status": "V2_PRE_OUTCOME_SEALED",
        "contract_tag": contract_freeze["contract_tag"],
        "git_commit": contract_freeze["git_commit"],
        "feature_policy_sha": contract_freeze["feature_policy_sha256"],
        "feature_policy_sha256": contract_freeze["feature_policy_sha256"],
        "applicability_policy_sha": contract_freeze["applicability_policy_sha256"],
        "applicability_policy_sha256": contract_freeze["applicability_policy_sha256"],
        "deterministic_axis_policy_sha": contract_freeze[
            "deterministic_axis_policy_sha256"
        ],
        "deterministic_axis_policy_sha256": contract_freeze[
            "deterministic_axis_policy_sha256"
        ],
        "evidence_priority_sha256": contract_freeze["evidence_priority_sha256"],
        "parser_prompt_sha": contract_freeze["parser_prompt_sha256"],
        "parser_prompt_sha256": contract_freeze["parser_prompt_sha256"],
        "locked_test_sha": sha256_file(parser_validation_manifest),
        "locked_test_sha256": sha256_file(parser_validation_manifest),
        "natural_locked_gate_passed": True,
        "balanced_locked_gate_passed": True,
        "abstention_audit_sha256": sha256_file(abstention_audit_manifest),
        "min_nobs": minimum_observed_axes,
        "minimum_observed_axes": minimum_observed_axes,
        "banding_method": contract.calibration_method,
        "band_rules": {
            key.value: value for key, value in contract.band_rules.items()
        },
        "band_contract_sha256": sha256_file(contract_path),
        "coverage_gate_policy_sha256": sha256_file(coverage_gate_policy),
        "coverage_gate_report_sha256": sha256_file(coverage_gate_path),
        "coverage_gate_passed": True,
        "coverage_gate": {
            "policy_sha256": sha256_file(coverage_gate_policy),
            "report_sha256": sha256_file(coverage_gate_path),
            "passed": True,
        },
        "signal_timestamp_policy": contract_freeze["signal_timestamp_policy"],
        "last_grounded_days": contract_freeze["last_grounded_days"],
        "last_grounded_role": contract_freeze["last_grounded_role"],
        "feature_seal_sha256": sha256_file(seal_path),
        "feature_dataset_sha256": sha256_file(eligible_path),
        "feature_contract_modification_forbidden": True,
        "next_stage": "SEPARATE_ELIGIBILITY_AND_REVERSE_DCF_IMPLEMENTATION_REQUIRED",
        "outcome_stage_authorized": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
    }
    pre_outcome_path = output / "pre-outcome-manifest.json"
    _write_json(pre_outcome_path, pre_outcome_manifest)
    base_status.update(
        status="V2_FEATURE_ONLY_CALIBRATION_SEALED_OUTCOMES_CLOSED",
        eligible_feature_count=len(eligible),
        feature_dataset_sealed=True,
        outcome_stage_authorized=False,
        next_gate="IMPLEMENT_AND_PASS_V2_ELIGIBILITY_THEN_REVERSE_DCF_BEFORE_ERI",
        sealed_feature_sha256=sha256_file(eligible_path),
        feature_seal_sha256=sha256_file(seal_path),
        parser_validation_manifest_sha256=sha256_file(parser_validation_manifest),
        abstention_audit_manifest_sha256=sha256_file(abstention_audit_manifest),
        contract_freeze_manifest_sha256=sha256_file(contract_freeze_manifest),
        coverage_gate_policy_sha256=sha256_file(coverage_gate_policy),
        pre_outcome_manifest_sha256=sha256_file(pre_outcome_path),
    )
    _write_json(output / "stage-status.json", base_status)
    return base_status


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose V2 sparse coverage without outcomes, then optionally freeze an explicit "
            "Nobs threshold and feature-only five-band contract after every V2 gate passes."
        )
    )
    parser.add_argument("--feature-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--minimum-observed-axes", type=int)
    parser.add_argument("--parser-validation-manifest", type=Path)
    parser.add_argument("--abstention-audit-manifest", type=Path)
    parser.add_argument("--contract-freeze-manifest", type=Path)
    parser.add_argument("--coverage-gate-policy", type=Path)
    args = parser.parse_args()
    result = calibrate_sparse_features(
        feature_build=args.feature_build,
        output=args.output,
        freeze=args.freeze,
        minimum_observed_axes=args.minimum_observed_axes,
        parser_validation_manifest=args.parser_validation_manifest,
        abstention_audit_manifest=args.abstention_audit_manifest,
        contract_freeze_manifest=args.contract_freeze_manifest,
        coverage_gate_policy=args.coverage_gate_policy,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
