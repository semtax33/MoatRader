from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from moatrader.canonical.models import ContractModel
from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    HistoricalFilingPair,
    canonical_payload_sha256,
    sha256_file,
)
from moatrader.expectations.historical_evidence_v2 import (
    DETERMINISTIC_CORE_AXES_V2,
    DeterministicCoreIndexCoveragePolicyV2,
    DeterministicCoreIndexRowV2,
    EvidenceIndexContractV2,
    SparseBreadthBandV2,
    build_deterministic_core_index_row_v2,
)
from scripts.build_historical_sparse_features_v2 import DeterministicAxisEvidenceInputV2


SEOUL = ZoneInfo("Asia/Seoul")
DEFAULT_COVERAGE_POLICY = DeterministicCoreIndexCoveragePolicyV2(
    minimum_rows_per_band=500,
    minimum_unique_issuers_per_band=500,
    minimum_unique_signal_months_per_band=24,
    minimum_total_unique_issuers=2_000,
    minimum_total_unique_signal_months=48,
    maximum_top_issuer_share_per_band=Decimal("0.02"),
    maximum_top_month_share_per_band=Decimal("0.25"),
    maximum_top_year_share_per_band=Decimal("0.30"),
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[ContractModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _git_state(workspace: Path) -> tuple[str, list[str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return commit, dirty


def _require_closed(payload: dict[str, Any], description: str) -> None:
    for key in (
        "outcome_vault_opened",
        "outcome_data_accessed",
        "outcomes_opened",
        "return_data_opened",
        "return_data_accessed",
        "returns_opened",
        "value_data_opened",
        "value_data_accessed",
    ):
        if payload.get(key, False):
            raise ValueError(f"{description} opened forbidden downstream data: {key}")
    if payload.get("per_pbr_role", "NOT_USED") != "NOT_USED":
        raise ValueError(f"{description} used PER/PBR before the index contract was frozen")


def _share(counter: Counter[str], denominator: int) -> Decimal:
    if not counter or denominator == 0:
        return Decimal(0)
    return Decimal(counter.most_common(1)[0][1]) / Decimal(denominator)


def deterministic_core_diagnostics_v2(
    rows: list[DeterministicCoreIndexRowV2],
    *,
    policy: DeterministicCoreIndexCoveragePolicyV2,
) -> dict[str, Any]:
    eligible = [row for row in rows if row.eligible]
    if not eligible:
        raise ValueError("Deterministic Core Index eligible subset is empty")
    all_issuers = {row.issuer_id for row in eligible}
    all_months = {row.signal_timestamp.strftime("%Y-%m") for row in eligible}
    by_band: dict[str, Any] = {}
    band_gate_passed = True
    for band in SparseBreadthBandV2:
        selected = [row for row in eligible if row.band == band]
        issuers = Counter(row.issuer_id for row in selected)
        months = Counter(row.signal_timestamp.strftime("%Y-%m") for row in selected)
        years = Counter(row.signal_timestamp.strftime("%Y") for row in selected)
        nobs = Counter(str(row.nobs) for row in selected)
        fractions = Counter(row.core_evidence_index_fraction for row in selected)
        checks = {
            "minimum_rows": len(selected) >= policy.minimum_rows_per_band,
            "minimum_unique_issuers": (
                len(issuers) >= policy.minimum_unique_issuers_per_band
            ),
            "minimum_unique_signal_months": (
                len(months) >= policy.minimum_unique_signal_months_per_band
            ),
            "issuer_concentration": (
                _share(issuers, len(selected))
                <= policy.maximum_top_issuer_share_per_band
            ),
            "month_concentration": (
                _share(months, len(selected))
                <= policy.maximum_top_month_share_per_band
            ),
            "year_concentration": (
                _share(years, len(selected))
                <= policy.maximum_top_year_share_per_band
            ),
        }
        passed = all(checks.values())
        band_gate_passed &= passed
        by_band[band.value] = {
            "row_count": len(selected),
            "row_share": Decimal(len(selected)) / Decimal(len(eligible)),
            "unique_issuers": len(issuers),
            "unique_signal_months": len(months),
            "unique_signal_years": len(years),
            "top_issuer_share": _share(issuers, len(selected)),
            "top_month_share": _share(months, len(selected)),
            "top_year_share": _share(years, len(selected)),
            "issuer_distribution": dict(sorted(issuers.items())),
            "signal_month_distribution": dict(sorted(months.items())),
            "signal_year_distribution": dict(sorted(years.items())),
            "nobs_distribution": dict(sorted(nobs.items())),
            "exact_index_fraction_distribution": dict(sorted(fractions.items())),
            "coverage_checks": checks,
            "coverage_gate_passed": passed,
        }
    global_checks = {
        "minimum_total_unique_issuers": (
            len(all_issuers) >= policy.minimum_total_unique_issuers
        ),
        "minimum_total_unique_signal_months": (
            len(all_months) >= policy.minimum_total_unique_signal_months
        ),
    }
    gate_passed = band_gate_passed and all(global_checks.values())
    nobs_all = Counter(str(row.nobs) for row in rows)
    nobs_eligible = Counter(str(row.nobs) for row in eligible)
    fractions = Counter(row.core_evidence_index_fraction for row in eligible)
    return {
        "schema_version": "moatrader-deterministic-core-index-diagnostics-v2/1",
        "status": "PASSED" if gate_passed else "FAILED_MEASUREMENT_COVERAGE",
        "coverage_gate_passed": gate_passed,
        "pair_count": len(rows),
        "eligible_row_count": len(eligible),
        "eligible_rate": Decimal(len(eligible)) / Decimal(len(rows)),
        "minimum_observed_axes": 2,
        "nobs_all_pairs": dict(sorted(nobs_all.items())),
        "nobs_eligible": dict(sorted(nobs_eligible.items())),
        "exact_index_fraction_distribution": dict(sorted(fractions.items())),
        "unique_issuers": len(all_issuers),
        "unique_signal_months": len(all_months),
        "global_checks": global_checks,
        "by_band": by_band,
        "capex_included": False,
        "score_and_coverage_separate": True,
        "index_multiplied_by_coverage": False,
        "outcome_data_accessed": False,
        "return_data_accessed": False,
        "value_data_accessed": False,
        "per_pbr_role": "NOT_USED",
    }


def _validate_source_inputs(
    *,
    filing_pair_input: Path,
    deterministic_evidence_input: Path,
    deterministic_stage_manifest: Path,
    measurement_audit_report: Path,
    pit_input_stage_manifest: Path,
    source_audit_report: Path,
) -> dict[str, Any]:
    deterministic = _read_json(deterministic_stage_manifest)
    measurement = _read_json(measurement_audit_report)
    pit = _read_json(pit_input_stage_manifest)
    source = _read_json(source_audit_report)
    for payload, description in (
        (deterministic, "deterministic evidence stage"),
        (measurement, "deterministic measurement audit"),
        (pit, "PIT input stage"),
        (source, "source audit"),
    ):
        _require_closed(payload, description)
    if deterministic.get("status") != "DETERMINISTIC_PIT_EVIDENCE_COMPLETE_OUTCOME_BLIND":
        raise ValueError("deterministic evidence stage is not complete and outcome blind")
    if deterministic.get("current_evidence_carry_forward") is not False:
        raise ValueError("current evidence carry-forward must be disabled")
    if deterministic.get("last_grounded_staleness_days") != 450:
        raise ValueError("last-grounded staleness must remain 450 days")
    if deterministic.get("last_grounded_role") != (
        "PREVIOUS_COMPARISON_BASIS_ONLY_NEVER_CURRENT_EVIDENCE"
    ):
        raise ValueError("last-grounded evidence role changed")
    if deterministic.get("deterministic_evidence_sha256") != sha256_file(
        deterministic_evidence_input
    ):
        raise ValueError("deterministic evidence hash does not match its stage manifest")
    if measurement.get("status") != "DETERMINISTIC_MEASUREMENT_AUDITED_OUTCOME_BLIND":
        raise ValueError("deterministic measurement audit is not complete and outcome blind")
    if measurement.get("input_hashes", {}).get("selection_2_evidence") != sha256_file(
        deterministic_evidence_input
    ):
        raise ValueError("measurement audit did not inspect the selected evidence input")
    if measurement.get("input_hashes", {}).get("filing_pairs") != sha256_file(
        filing_pair_input
    ):
        raise ValueError("measurement audit and filing-pair input differ")
    if measurement.get("source_files_modified") is not False:
        raise ValueError("measurement audit does not prove source files were preserved")
    if measurement.get("source_write_operations") != 0:
        raise ValueError("measurement audit recorded a source write")
    if pit.get("status") != "PIT_INPUTS_PREPARED_OUTCOME_BLIND":
        raise ValueError("PIT inputs are not complete and outcome blind")
    if not all(
        pit.get(key) is True
        for key in (
            "arcana_business_info_used",
            "arcana_finance_comment_used",
            "arcana_finance_statement_used",
            "moatrader_original_used",
            "all_available_source_variants_read",
        )
    ):
        raise ValueError(
            "PIT inputs did not read all three Arcana sections and MoatRader originals"
        )
    if pit.get("source_files_modified") is not False:
        raise ValueError("PIT input stage does not prove source preservation")
    if (
        pit.get("source_hash_mismatch_count") != 0
        or pit.get("source_write_operations") != 0
    ):
        raise ValueError("PIT input stage found a source mutation or write")
    if pit.get("expected_source_path_count_by_origin") != pit.get(
        "verified_source_path_count_by_origin"
    ):
        raise ValueError("PIT input stage did not verify every expected source path")
    if not all(
        source.get(key) is True
        for key in (
            "all_arcana_sections_discovered",
            "all_arcana_sections_read_for_pairs",
            "all_arcana_sections_contributed_to_packets",
            "both_source_systems_used",
        )
    ):
        raise ValueError("source audit lacks Arcana three-section or MoatRader coverage")
    if source.get("arcana_section_selection") != [
        "business-info",
        "finance-comment",
        "finance-statement",
    ]:
        raise ValueError("source audit Arcana section selection changed")
    if source.get("source_files_modified") is not False:
        raise ValueError("source audit reports source modification")
    if source.get("moatrader_original_audit", {}).get("archive_hash_mismatch_count") != 0:
        raise ValueError("MoatRader original archive hash mismatch detected")
    return {
        "deterministic_stage": deterministic,
        "measurement_audit": measurement,
        "pit_stage": pit,
        "source_audit": source,
    }


def freeze_historical_evidence_index_v2(
    *,
    workspace: Path,
    filing_pair_input: Path,
    deterministic_evidence_input: Path,
    deterministic_stage_manifest: Path,
    measurement_audit_report: Path,
    pit_input_stage_manifest: Path,
    source_audit_report: Path,
    output: Path,
    contract_tag: str,
    dry_run: bool = False,
    coverage_policy: DeterministicCoreIndexCoveragePolicyV2 = DEFAULT_COVERAGE_POLICY,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if not contract_tag.strip():
        raise ValueError("contract_tag cannot be blank")
    commit, dirty = _git_state(workspace)
    if dirty and not dry_run:
        raise ValueError("production Evidence Index freeze requires a clean worktree")
    inputs = _validate_source_inputs(
        filing_pair_input=filing_pair_input,
        deterministic_evidence_input=deterministic_evidence_input,
        deterministic_stage_manifest=deterministic_stage_manifest,
        measurement_audit_report=measurement_audit_report,
        pit_input_stage_manifest=pit_input_stage_manifest,
        source_audit_report=source_audit_report,
    )
    pairs = _read_jsonl(filing_pair_input, HistoricalFilingPair)
    if len({pair.pair_id for pair in pairs}) != len(pairs):
        raise ValueError("filing-pair IDs must be unique")
    declared_pair_counts = {
        inputs["deterministic_stage"].get("pair_count"),
        inputs["measurement_audit"].get("pair_count"),
        inputs["pit_stage"].get("pair_count"),
        inputs["source_audit"].get("regular_pair_count"),
    }
    if declared_pair_counts != {len(pairs)}:
        raise ValueError("upstream manifests do not agree on the filing-pair count")
    evidence_rows = _read_jsonl(
        deterministic_evidence_input, DeterministicAxisEvidenceInputV2
    )
    evidence = {(row.pair_id, row.evidence.axis): row.evidence for row in evidence_rows}
    if len(evidence) != len(evidence_rows):
        raise ValueError("deterministic evidence must be unique by pair and axis")
    required_axes = {*DETERMINISTIC_CORE_AXES_V2, OperatingEvidenceAxis.CAPACITY_CAPEX}
    expected_keys = {(pair.pair_id, axis) for pair in pairs for axis in required_axes}
    if set(evidence) != expected_keys:
        raise ValueError("deterministic evidence is not a complete four-axis pair panel")
    rows = sorted(
        (
            build_deterministic_core_index_row_v2(
                pair=pair,
                axis_evidence=[evidence[(pair.pair_id, axis)] for axis in required_axes],
            )
            for pair in pairs
        ),
        key=lambda row: row.observation_id,
    )
    diagnostics = deterministic_core_diagnostics_v2(rows, policy=coverage_policy)
    expected_nobs2 = inputs["measurement_audit"][
        "selection_2_primary_deterministic_features"
    ]["nobs_at_least_2"]
    if diagnostics["eligible_row_count"] != expected_nobs2:
        raise ValueError("Core Index eligible count differs from the frozen measurement audit")
    if not diagnostics["coverage_gate_passed"]:
        raise ValueError("Core Index failed its outcome-blind measurement coverage gate")

    output.mkdir(parents=True, exist_ok=True)
    contract = EvidenceIndexContractV2()
    contract_path = output / "evidence-index-contract.json"
    all_path = output / "deterministic-core-index-all-pairs.jsonl"
    eligible_path = output / "deterministic-core-index-eligible-nobs2.jsonl"
    diagnostics_path = output / "deterministic-core-index-diagnostics.json"
    _write_json(contract_path, contract.model_dump(mode="json"))
    _write_jsonl(all_path, rows)
    _write_jsonl(eligible_path, (row for row in rows if row.eligible))
    _write_json(diagnostics_path, diagnostics)
    artifact_hashes = {
        "evidence_index_contract": sha256_file(contract_path),
        "deterministic_core_index_all_pairs": sha256_file(all_path),
        "deterministic_core_index_eligible_nobs2": sha256_file(eligible_path),
        "deterministic_core_index_diagnostics": sha256_file(diagnostics_path),
    }
    input_hashes = {
        "filing_pairs": sha256_file(filing_pair_input),
        "deterministic_evidence": sha256_file(deterministic_evidence_input),
        "deterministic_stage_manifest": sha256_file(deterministic_stage_manifest),
        "measurement_audit_report": sha256_file(measurement_audit_report),
        "pit_input_stage_manifest": sha256_file(pit_input_stage_manifest),
        "source_audit_report": sha256_file(source_audit_report),
    }
    code_paths = (
        workspace / "src/moatrader/expectations/historical_evidence_v2.py",
        workspace / "scripts/freeze_historical_evidence_index_v2.py",
        workspace / "scripts/build_historical_deterministic_pit_evidence_v2.py",
        workspace / "scripts/prepare_historical_deterministic_pit_inputs_v2.py",
        workspace / "scripts/audit_historical_deterministic_measurement_v2.py",
    )
    code_hashes = {
        str(path.relative_to(workspace)): sha256_file(path) for path in code_paths
    }
    now = datetime.now(SEOUL)
    status = (
        "DRY_RUN_V2_EVIDENCE_INDEX_CONTRACT_VALIDATED_OUTCOMES_CLOSED"
        if dry_run
        else "V2_EVIDENCE_INDEX_CONTRACT_FROZEN_OUTCOMES_CLOSED"
    )
    pre_outcome = {
        "schema_version": "moatrader-pre-outcome-evidence-index-manifest-v2/1",
        "status": status,
        "frozen_at": now.isoformat(),
        "contract_tag": contract_tag.strip(),
        "git_commit": commit,
        "worktree_dirty": bool(dirty),
        "dry_run_only": dry_run,
        "evidence_index_contract_sha256": artifact_hashes["evidence_index_contract"],
        "coverage_policy": coverage_policy.model_dump(mode="json"),
        "coverage_policy_sha256": canonical_payload_sha256(
            coverage_policy.model_dump(mode="json")
        ),
        "coverage_gate_passed": diagnostics["coverage_gate_passed"],
        "pair_count": len(rows),
        "eligible_row_count": diagnostics["eligible_row_count"],
        "minimum_observed_axes": 2,
        "banding_method": contract.banding_method,
        "band_rules": {key.value: value for key, value in contract.band_rules.items()},
        "primary_index": contract.primary_index,
        "primary_measurement_status": contract.primary_measurement_status,
        "full_index_materialized": False,
        "secondary_index": contract.secondary_index,
        "deterministic_core_materialized": True,
        "capex_role": "DIAGNOSTIC_ONLY",
        "last_grounded_days": 450,
        "current_evidence_carry_forward": False,
        "artifact_hashes": artifact_hashes,
        "input_hashes": input_hashes,
        "code_hashes": code_hashes,
        "source_provenance_gate": {
            "arcana_business_info_read": True,
            "arcana_finance_comment_read": True,
            "arcana_finance_statement_read": True,
            "moatrader_original_regular_filings_read": True,
            "all_expected_source_paths_verified": True,
            "source_integrity_record_count": inputs["source_audit"].get(
                "source_integrity_record_count"
            ),
            "source_files_modified": False,
            "source_write_operations": 0,
            "source_hash_mismatch_count": 0,
        },
        "next_gate": "LOCK_DEMAND_AND_PRICE_MIX_SEMANTIC_PARSER_THEN_BUILD_FULL_INDEX",
        "outcome_stage_authorized": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
    }
    manifest_path = output / "pre-outcome-index-manifest.json"
    _write_json(manifest_path, pre_outcome)
    stage_status = {
        **pre_outcome,
        "schema_version": "moatrader-evidence-index-freeze-stage-v2/1",
        "pre_outcome_index_manifest_sha256": sha256_file(manifest_path),
    }
    _write_json(output / "stage-status.json", stage_status)
    return stage_status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze the outcome-blind V2 Evidence Index contract and Core baseline."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--filing-pair-input", type=Path, required=True)
    parser.add_argument("--deterministic-evidence-input", type=Path, required=True)
    parser.add_argument("--deterministic-stage-manifest", type=Path, required=True)
    parser.add_argument("--measurement-audit-report", type=Path, required=True)
    parser.add_argument("--pit-input-stage-manifest", type=Path, required=True)
    parser.add_argument("--source-audit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contract-tag", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    status = freeze_historical_evidence_index_v2(**vars(args))
    print(json.dumps(status, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
