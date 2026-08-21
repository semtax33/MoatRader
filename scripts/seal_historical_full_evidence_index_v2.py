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
from moatrader.expectations.historical_evidence import canonical_payload_sha256, sha256_file
from moatrader.expectations.historical_evidence_v2 import (
    FullEvidenceIndexCoveragePolicyV2,
    FullEvidenceIndexRowV2,
    HistoricalSparseEvidenceFeatureRowV2,
    SparseBreadthBandV2,
    build_full_evidence_index_row_v2,
)
from scripts.classify_historical_future_eri_evidence import (
    ParserProfile,
    parser_spec,
)


D = Decimal
SEOUL = ZoneInfo("Asia/Seoul")
DEFAULT_FULL_COVERAGE_POLICY = FullEvidenceIndexCoveragePolicyV2(
    minimum_rows_per_band=500,
    minimum_unique_issuers_per_band=500,
    minimum_unique_signal_months_per_band=24,
    minimum_total_unique_issuers=2_000,
    minimum_total_unique_signal_months=48,
    maximum_top_issuer_share_per_band=D("0.02"),
    maximum_top_month_share_per_band=D("0.25"),
    maximum_top_year_share_per_band=D("0.30"),
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
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[ContractModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


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
        raise ValueError(f"{description} used PER/PBR before the Full Index seal")


def _share(counter: Counter[str], denominator: int) -> D:
    if not counter or denominator == 0:
        return D(0)
    return D(counter.most_common(1)[0][1]) / D(denominator)


def _validate_frozen_execution_code(
    *, workspace: Path, sparse_stage: dict[str, Any], current_commit: str
) -> None:
    if sparse_stage.get("measurement_contract_frozen") is not True:
        raise ValueError("Full Index seal requires the frozen V2 measurement contract")
    if sparse_stage.get("measurement_contract_git_commit") != current_commit:
        raise ValueError("measurement code commit changed after the V2 contract freeze")
    frozen = sparse_stage.get("measurement_contract_code_sha256", {})
    if not isinstance(frozen, dict):
        raise ValueError("sparse feature stage lacks frozen measurement code hashes")
    required = {
        "semantic_classifier": workspace
        / "scripts"
        / "classify_historical_future_eri_evidence.py",
        "semantic_cost_preparer": workspace
        / "scripts"
        / "prepare_historical_semantic_cost_manifest_v2.py",
        "sparse_builder": workspace / "scripts" / "build_historical_sparse_features_v2.py",
        "full_index_sealer": workspace
        / "scripts"
        / "seal_historical_full_evidence_index_v2.py",
        "eri_runner": workspace / "scripts" / "run_historical_evidence_index_eri_v2.py",
        "value_neutralization_runner": workspace
        / "scripts"
        / "run_historical_evidence_index_value_neutralization_v2.py",
    }
    for name, path in required.items():
        if frozen.get(name) != sha256_file(path):
            raise ValueError(f"V2 measurement code changed after contract freeze: {name}")


def full_evidence_index_diagnostics_v2(
    rows: list[FullEvidenceIndexRowV2],
    *,
    policy: FullEvidenceIndexCoveragePolicyV2,
) -> dict[str, Any]:
    eligible = [row for row in rows if row.eligible]
    if not eligible:
        raise ValueError("Full Evidence Index eligible subset is empty")
    by_band: dict[str, Any] = {}
    band_passed = True
    for band in SparseBreadthBandV2:
        selected = [row for row in eligible if row.band == band]
        issuers = Counter(row.issuer_id for row in selected)
        months = Counter(row.signal_timestamp.strftime("%Y-%m") for row in selected)
        years = Counter(row.signal_timestamp.strftime("%Y") for row in selected)
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
        band_passed &= passed
        by_band[band.value] = {
            "row_count": len(selected),
            "unique_issuers": len(issuers),
            "unique_signal_months": len(months),
            "top_issuer_share": _share(issuers, len(selected)),
            "top_month_share": _share(months, len(selected)),
            "top_year_share": _share(years, len(selected)),
            "nobs_distribution": dict(sorted(Counter(str(row.nobs) for row in selected).items())),
            "semantic_grounded_axis_count_distribution": dict(
                sorted(
                    Counter(
                        str(row.semantic_grounded_axis_count) for row in selected
                    ).items()
                )
            ),
            "coverage_checks": checks,
            "coverage_gate_passed": passed,
        }
    all_issuers = {row.issuer_id for row in eligible}
    all_months = {row.signal_timestamp.strftime("%Y-%m") for row in eligible}
    global_checks = {
        "minimum_total_unique_issuers": (
            len(all_issuers) >= policy.minimum_total_unique_issuers
        ),
        "minimum_total_unique_signal_months": (
            len(all_months) >= policy.minimum_total_unique_signal_months
        ),
    }
    passed = band_passed and all(global_checks.values())
    return {
        "schema_version": "moatrader-full-evidence-index-diagnostics-v2/1",
        "status": "PASSED" if passed else "FAILED_MEASUREMENT_COVERAGE",
        "coverage_gate_passed": passed,
        "pair_count": len(rows),
        "eligible_row_count": len(eligible),
        "eligible_rate": D(len(eligible)) / D(len(rows)),
        "minimum_observed_axes": 2,
        "nobs_all_pairs": dict(sorted(Counter(str(row.nobs) for row in rows).items())),
        "nobs_eligible": dict(sorted(Counter(str(row.nobs) for row in eligible).items())),
        "unique_issuers": len(all_issuers),
        "unique_signal_months": len(all_months),
        "global_checks": global_checks,
        "by_band": by_band,
        "full_axes": ["DEMAND", "PRICE_MIX", "MARGIN", "INVENTORY_MISMATCH", "BACKLOG"],
        "capex_included": False,
        "score_and_coverage_separate": True,
        "index_multiplied_by_coverage": False,
        "outcome_data_accessed": False,
        "return_data_accessed": False,
        "value_data_accessed": False,
        "per_pbr_role": "NOT_USED",
    }


def _validate_gates(
    *,
    sparse_feature_input: Path,
    sparse_stage_manifest: Path,
    dual_locked_manifest: Path,
    semantic_classification_stage_manifest: Path,
    semantic_selection_manifest: Path,
    cost_manifest: Path,
    core_pre_outcome_manifest: Path,
) -> dict[str, dict[str, Any]]:
    sparse = _read_json(sparse_stage_manifest)
    locked = _read_json(dual_locked_manifest)
    classification = _read_json(semantic_classification_stage_manifest)
    selection = _read_json(semantic_selection_manifest)
    cost = _read_json(cost_manifest)
    core = _read_json(core_pre_outcome_manifest)
    for payload, description in (
        (sparse, "sparse feature stage"),
        (locked, "dual LOCKED gate"),
        (classification, "semantic classification stage"),
        (selection, "semantic selection"),
        (cost, "semantic cost prespecification"),
        (core, "Core Index contract"),
    ):
        _require_closed(payload, description)
    if locked.get("status") != "V2_LOCKED_TESTS_PASSED":
        raise ValueError("Demand/PriceMix dual LOCKED tests have not passed")
    if not all(
        locked.get(key) is True
        for key in ("natural_frequency_gate_passed", "directional_strata_gate_passed")
    ):
        raise ValueError("both Natural and Balanced LOCKED gate flags must be true")
    spec = parser_spec(ParserProfile.DEMAND_PRICE_MIX_V2)
    expected_parser = {
        "parser_version": spec.parser_version,
        "prompt_sha256": spec.prompt_sha256,
        "requested_model": "gpt-5.6-luna",
    }
    for payload, description in (
        (locked, "dual LOCKED gate"),
        (classification, "semantic classification stage"),
    ):
        for key, expected in expected_parser.items():
            if payload.get(key) != expected:
                raise ValueError(f"{description} does not match frozen {key}")
    if classification.get("parser_profile") != spec.profile.value:
        raise ValueError("full classification did not use the semantic V2 parser profile")
    if classification.get("status") != (
        "FULL_SEMANTIC_CLASSIFICATION_COMPLETE_OUTCOMES_CLOSED"
    ):
        raise ValueError("full semantic classification is incomplete")
    if classification.get("semantic_execution_scope") != "FULL_HISTORICAL" or (
        classification.get("full_historical_execution_authorized") is not True
    ):
        raise ValueError("semantic classifications lack the full-run execution gate")
    for key, path in (
        ("dual_locked_manifest_sha256", dual_locked_manifest),
        ("semantic_selection_manifest_sha256", semantic_selection_manifest),
        ("semantic_cost_manifest_sha256", cost_manifest),
    ):
        if classification.get(key) != sha256_file(path):
            raise ValueError(f"semantic classification authorization hash mismatch: {key}")
    if classification.get("classification_count") != classification.get("packet_count"):
        raise ValueError("semantic classification count is incomplete")
    if classification.get("input_blinded_packet_sha256") != selection.get(
        "output_packet_sha256"
    ):
        raise ValueError("classification input differs from semantic selection")
    if classification.get("packet_count") != selection.get("selected_packet_count"):
        raise ValueError("classification packet count differs from semantic selection")
    if cost.get("status") != "FULL_SEMANTIC_RUN_COST_PRESPECIFIED_NO_EXTERNAL_CALL":
        raise ValueError("full semantic cost was not prespecified")
    if cost.get("api_calls_executed") is not False:
        raise ValueError("cost manifest was created after calls were executed")
    token_estimation = cost.get("token_estimation", {})
    if (
        token_estimation.get("pilot_prompt_differs_from_frozen_full_prompt") is not False
        or token_estimation.get("pilot_contract_matches_frozen_full_prompt") is not True
    ):
        raise ValueError(
            "Full Index seal requires cost estimation from exact frozen V2 pilots"
        )
    for key, expected in (
        ("parser_profile", spec.profile.value),
        ("parser_version", spec.parser_version),
        ("prompt_sha256", spec.prompt_sha256),
        ("model", "gpt-5.6-luna"),
    ):
        if cost.get(key) != expected:
            raise ValueError(f"cost manifest does not match frozen {key}")
    if cost.get("exact_packet_count") != selection.get("selected_packet_count"):
        raise ValueError("cost manifest packet count differs from semantic selection")
    cost_inputs = cost.get("inputs", {})
    if cost_inputs.get("dual_locked_manifest_sha256") != sha256_file(
        dual_locked_manifest
    ):
        raise ValueError("cost manifest does not seal the dual LOCKED lineage")
    if sparse.get("status") != "SPARSE_FEATURES_BUILT_AWAITING_OUTCOME_BLIND_CALIBRATION":
        raise ValueError("sparse full feature stage is incomplete")
    if sparse.get("parser_directional_validation_passed") is not True:
        raise ValueError("sparse feature stage lacks the parser gate")
    if sparse.get("missing_is_neutral") is not False:
        raise ValueError("sparse feature stage silently converted missing to neutral")
    if sparse.get("deterministic_pit_priority_applied") is not True:
        raise ValueError("deterministic PIT priority was not applied")
    sparse_hashes = sparse.get("input_hashes", {})
    if sparse_hashes.get("sparse_features") != sha256_file(sparse_feature_input):
        raise ValueError("sparse feature file changed after its stage manifest")
    if sparse_hashes.get("classifications") != classification.get("classification_sha256"):
        raise ValueError("sparse features do not use the sealed semantic classifications")
    if core.get("status") != "V2_EVIDENCE_INDEX_CONTRACT_FROZEN_OUTCOMES_CLOSED":
        raise ValueError("Core Index contract is not frozen")
    if core.get("deterministic_core_materialized") is not True:
        raise ValueError("Core secondary baseline is absent")
    provenance = core.get("source_provenance_gate", {})
    if not all(
        provenance.get(key) is True
        for key in (
            "arcana_business_info_read",
            "arcana_finance_comment_read",
            "arcana_finance_statement_read",
            "moatrader_original_regular_filings_read",
            "all_expected_source_paths_verified",
        )
    ):
        raise ValueError("Arcana three-section or MoatRader original source proof is missing")
    if provenance.get("source_files_modified") is not False:
        raise ValueError("source provenance does not prove original-file preservation")
    source_contract = sparse.get("source_contract", {})
    if source_contract.get("status") != (
        "ARCANA_AND_MOATRADER_ORIGINALS_VERIFIED_READ_ONLY"
    ) or source_contract.get("verified") is not True:
        raise ValueError("sparse features lack the production three-section source contract")
    for classification_key, source_key in (
        ("semantic_source_audit_sha256", "source_audit_sha256"),
        ("semantic_source_build_manifest_sha256", "build_manifest_sha256"),
        (
            "semantic_source_integrity_before_sha256",
            "source_integrity_before_sha256",
        ),
        (
            "semantic_source_integrity_after_sha256",
            "source_integrity_after_sha256",
        ),
    ):
        if classification.get(classification_key) != source_contract.get(source_key):
            raise ValueError(
                f"semantic classification and sparse source lineage differ: {classification_key}"
            )
    return {
        "sparse": sparse,
        "locked": locked,
        "classification": classification,
        "selection": selection,
        "cost": cost,
        "core": core,
    }


def seal_full_evidence_index_v2(
    *,
    workspace: Path,
    sparse_feature_input: Path,
    sparse_stage_manifest: Path,
    dual_locked_manifest: Path,
    semantic_classification_stage_manifest: Path,
    semantic_selection_manifest: Path,
    cost_manifest: Path,
    core_pre_outcome_manifest: Path,
    output: Path,
    seal_tag: str,
    dry_run: bool = False,
    coverage_policy: FullEvidenceIndexCoveragePolicyV2 = DEFAULT_FULL_COVERAGE_POLICY,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if not seal_tag.strip():
        raise ValueError("seal_tag cannot be blank")
    gates = _validate_gates(
        sparse_feature_input=sparse_feature_input,
        sparse_stage_manifest=sparse_stage_manifest,
        dual_locked_manifest=dual_locked_manifest,
        semantic_classification_stage_manifest=semantic_classification_stage_manifest,
        semantic_selection_manifest=semantic_selection_manifest,
        cost_manifest=cost_manifest,
        core_pre_outcome_manifest=core_pre_outcome_manifest,
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _validate_frozen_execution_code(
        workspace=workspace,
        sparse_stage=gates["sparse"],
        current_commit=commit,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if dirty and not dry_run:
        raise ValueError("production Full Index seal requires a clean worktree")
    sparse_rows = _read_jsonl(sparse_feature_input, HistoricalSparseEvidenceFeatureRowV2)
    rows = sorted(
        (build_full_evidence_index_row_v2(row) for row in sparse_rows),
        key=lambda row: row.observation_id,
    )
    if len(rows) != gates["sparse"].get("pair_count"):
        raise ValueError("Full Index row count differs from sparse feature stage")
    if len({row.observation_id for row in rows}) != len(rows):
        raise ValueError("Full Index observation IDs must be unique")
    diagnostics = full_evidence_index_diagnostics_v2(rows, policy=coverage_policy)
    if not diagnostics["coverage_gate_passed"]:
        raise ValueError("Full Evidence Index failed the outcome-blind coverage gate")

    output.mkdir(parents=True, exist_ok=True)
    all_path = output / "full-evidence-index-all-pairs.jsonl"
    eligible_path = output / "full-evidence-index-eligible-nobs2.jsonl"
    diagnostics_path = output / "full-evidence-index-diagnostics.json"
    _write_jsonl(all_path, rows)
    _write_jsonl(eligible_path, (row for row in rows if row.eligible))
    _write_json(diagnostics_path, diagnostics)
    artifact_hashes = {
        "full_evidence_index_all_pairs": sha256_file(all_path),
        "full_evidence_index_eligible_nobs2": sha256_file(eligible_path),
        "full_evidence_index_diagnostics": sha256_file(diagnostics_path),
    }
    input_hashes = {
        "sparse_features": sha256_file(sparse_feature_input),
        "sparse_stage_manifest": sha256_file(sparse_stage_manifest),
        "dual_locked_manifest": sha256_file(dual_locked_manifest),
        "semantic_classification_stage_manifest": sha256_file(
            semantic_classification_stage_manifest
        ),
        "semantic_selection_manifest": sha256_file(semantic_selection_manifest),
        "semantic_cost_manifest": sha256_file(cost_manifest),
        "core_pre_outcome_manifest": sha256_file(core_pre_outcome_manifest),
    }
    authorized = not dry_run
    manifest = {
        "schema_version": "moatrader-full-evidence-index-seal-v2/1",
        "status": (
            "V2_FULL_EVIDENCE_INDEX_SEALED_OUTCOMES_CLOSED"
            if authorized
            else "DRY_RUN_V2_FULL_EVIDENCE_INDEX_VALIDATED_OUTCOMES_CLOSED"
        ),
        "sealed_at": datetime.now(SEOUL).isoformat(),
        "seal_tag": seal_tag.strip(),
        "git_commit": commit,
        "worktree_dirty": bool(dirty),
        "dry_run_only": dry_run,
        "primary_index": "FULL_EVIDENCE_SIGNED_BREADTH_V2",
        "primary_axes": [
            "DEMAND",
            "PRICE_MIX",
            "MARGIN",
            "INVENTORY_MISMATCH",
            "BACKLOG",
        ],
        "primary_measurement_status": "MATERIALIZED_AND_COVERAGE_SEALED",
        "full_index_materialized": True,
        "secondary_index": "DETERMINISTIC_CORE_SIGNED_BREADTH_V2",
        "deterministic_core_materialized": True,
        "minimum_observed_axes": 2,
        "banding_method": "FIXED_ECONOMIC_SIGN_BANDS_V2",
        "coverage_gate_passed": True,
        "coverage_policy": coverage_policy.model_dump(mode="json"),
        "coverage_policy_sha256": canonical_payload_sha256(
            coverage_policy.model_dump(mode="json")
        ),
        "pair_count": len(rows),
        "eligible_row_count": diagnostics["eligible_row_count"],
        "capex_role": "DIAGNOSTIC_ONLY",
        "capex_included": False,
        "score_and_coverage_separate": True,
        "index_multiplied_by_coverage": False,
        "semantic_parser_gate_passed": True,
        "natural_frequency_locked_gate_passed": True,
        "balanced_directional_locked_gate_passed": True,
        "parser_version": gates["locked"]["parser_version"],
        "prompt_sha256": gates["locked"]["prompt_sha256"],
        "model": gates["locked"]["requested_model"],
        "source_provenance_gate": gates["core"]["source_provenance_gate"],
        "artifact_hashes": artifact_hashes,
        "input_hashes": input_hashes,
        "outcome_stage_authorized": authorized,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "future_eri_role": "DOWNSTREAM_OUTCOME_ONLY_NOT_SIGNAL_OR_RANKING",
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
    }
    manifest_path = output / "full-evidence-index-seal.json"
    _write_json(manifest_path, manifest)
    status = {
        **manifest,
        "schema_version": "moatrader-full-evidence-index-stage-v2/1",
        "full_evidence_index_seal_sha256": sha256_file(manifest_path),
    }
    _write_json(output / "stage-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seal the five-axis Full Evidence Index before any t+63 outcome access."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--sparse-feature-input", type=Path, required=True)
    parser.add_argument("--sparse-stage-manifest", type=Path, required=True)
    parser.add_argument("--dual-locked-manifest", type=Path, required=True)
    parser.add_argument("--semantic-classification-stage-manifest", type=Path, required=True)
    parser.add_argument("--semantic-selection-manifest", type=Path, required=True)
    parser.add_argument("--cost-manifest", type=Path, required=True)
    parser.add_argument("--core-pre-outcome-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seal-tag", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = seal_full_evidence_index_v2(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
