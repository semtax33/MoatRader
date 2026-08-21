from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from moatrader.expectations.historical_evidence import canonical_payload_sha256, sha256_file
from moatrader.expectations.historical_evidence_v2 import (
    EvidenceIndexContractV2,
    PITApplicabilityRulesV2,
)
from scripts.classify_historical_future_eri_evidence import ParserProfile, parser_spec


SEOUL = ZoneInfo("Asia/Seoul")
EVIDENCE_PRIORITY = (
    "DETERMINISTIC_NUMERIC",
    "STRUCTURED_TABLE",
    "LLM_NARRATIVE",
)
DETERMINISTIC_AXIS_POLICY = {
    "MARGIN": "OPERATING_MARGIN_DELTA_NUMERIC",
    "INVENTORY_MISMATCH": "INVENTORY_GROWTH_MINUS_REVENUE_GROWTH_NUMERIC",
    "BACKLOG": "BACKLOG_GROWTH_STRUCTURED_TABLE",
    "CAPACITY_CAPEX": "RAW_INVESTMENT_DIRECTION_DIAGNOSTIC_ONLY",
}
FEATURE_POLICY = {
    "states": ["-1", "0", "+1", "NA", "NOT_APPLICABLE"],
    "zero_definition": "GROUNDED_CURRENT_AND_PREVIOUS_WITH_NO_DIRECTIONAL_CHANGE",
    "na_definition": "NO_CURRENTLY_GROUNDED_COMPARABLE_EVIDENCE",
    "not_applicable_definition": "AXIS_HAS_NO_ECONOMIC_MEANING_UNDER_PIT_RULE",
    "nobs": "COUNT_OF_GROUNDED_PRIMARY_-1_0_+1_EXCLUDING_CAPEX",
    "minimum_nobs": 2,
    "n_directional": "COUNT_OF_GROUNDED_ABSOLUTE_DIRECTION_1_DIAGNOSTIC_ONLY",
    "signed_breadth": "(N_POSITIVE-N_NEGATIVE)/NOBS",
    "coverage": "NOBS/N_APPLICABLE_PRIMARY_AXES_EXCLUDING_CAPEX",
    "score_and_coverage_separate": True,
    "index_multiplied_by_coverage": False,
    "banding_method": "FIXED_ECONOMIC_SIGN_BANDS_V2",
    "primary_index": "FULL_EVIDENCE_SIGNED_BREADTH_V2",
    "secondary_index": "DETERMINISTIC_CORE_SIGNED_BREADTH_V2",
    "last_grounded_role": "PREVIOUS_COMPARISON_BASIS_ONLY",
    "current_evidence_carry_forward": False,
    "primary_ranking_policy": "NONE_MECHANISM_ONLY",
    "per_pbr_role": "NOT_USED",
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _git_value(args: list[str], *, workspace: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def freeze_contract(
    *,
    workspace: Path,
    rules_input: Path,
    parser_freeze_manifest: Path,
    source_build: Path,
    contract_tag: str,
    output: Path,
    allow_dirty_for_dry_run: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"V2 contract freeze already exists: {output}")
    if not contract_tag.strip():
        raise ValueError("contract_tag cannot be blank")
    source_audit_path = source_build / "source-audit.json"
    source_manifest_path = source_build / "build-manifest.json"
    code_paths = {
        "feature_contract": workspace / "src" / "moatrader" / "expectations" / "historical_evidence_v2.py",
        "deterministic_builder": workspace / "scripts" / "build_historical_deterministic_pit_evidence_v2.py",
        "semantic_selector": workspace / "scripts" / "prepare_historical_semantic_packets_v2.py",
        "sparse_builder": workspace / "scripts" / "build_historical_sparse_features_v2.py",
        "calibrator": workspace / "scripts" / "calibrate_historical_sparse_features_v2.py",
        "locked_evaluator": workspace / "scripts" / "evaluate_historical_evidence_parser_v2.py",
        "locked_set_preparer": workspace / "scripts" / "prepare_historical_locked_sets_v2.py",
        "abstention_audit": workspace / "scripts" / "audit_historical_evidence_abstentions_v2.py",
        "evidence_index_freezer": workspace / "scripts" / "freeze_historical_evidence_index_v2.py",
        "semantic_classifier": workspace
        / "scripts"
        / "classify_historical_future_eri_evidence.py",
        "full_index_sealer": workspace
        / "scripts"
        / "seal_historical_full_evidence_index_v2.py",
        "eri_runner": workspace / "scripts" / "run_historical_evidence_index_eri_v2.py",
        "value_neutralization_runner": workspace
        / "scripts"
        / "run_historical_evidence_index_value_neutralization_v2.py",
    }
    for path in (
        rules_input,
        parser_freeze_manifest,
        source_audit_path,
        source_manifest_path,
        *code_paths.values(),
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    rules = PITApplicabilityRulesV2.model_validate_json(rules_input.read_text(encoding="utf-8"))
    if rules.last_grounded_staleness_days != 450:
        raise ValueError("V2 contract fixes last-grounded previous-basis staleness at 450 days")
    parser_freeze = json.loads(parser_freeze_manifest.read_text(encoding="utf-8"))
    if parser_freeze.get("schema_version") != "moatrader-historical-evidence-parser-freeze-v2/2":
        raise ValueError("contract freeze requires the dual independent V2 parser freeze")
    if parser_freeze.get("status") != (
        "V2_PARSER_FROZEN_AWAITING_DUAL_INDEPENDENT_LOCKED_TESTS"
    ):
        raise ValueError("contract freeze requires the pending dual-LOCKED parser freeze")
    spec = parser_spec(ParserProfile.DEMAND_PRICE_MIX_V2)
    for key, expected in (
        ("parser_profile", spec.profile.value),
        ("parser_version", spec.parser_version),
        ("prompt_sha256", spec.prompt_sha256),
        ("requested_model", "gpt-5.6-luna"),
    ):
        if parser_freeze.get(key) != expected:
            raise ValueError(f"contract freeze parser does not match semantic V2 {key}")
    for key in ("outcome_vault_opened", "return_data_opened", "value_data_opened"):
        if parser_freeze.get(key, False):
            raise ValueError(f"parser freeze opened forbidden downstream data: {key}")
    if parser_freeze.get("per_pbr_role", "NOT_USED") != "NOT_USED":
        raise ValueError("parser freeze used PER/PBR before the Full Index seal")
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if not source_audit.get("both_source_systems_used", False):
        raise ValueError("contract freeze requires both Arcana and MoatRader original sources")
    if not source_audit.get("all_arcana_sections_discovered", False) or not source_audit.get(
        "all_arcana_sections_read_for_pairs", False
    ):
        raise ValueError(
            "contract freeze requires Arcana business-info, finance-comment, and "
            "finance-statement discovery and extraction"
        )
    if not source_audit.get("all_arcana_sections_contributed_to_packets", False):
        raise ValueError("contract freeze requires packet evidence from all three Arcana sections")
    if source_audit.get("source_files_modified", True) or source_manifest.get(
        "source_files_modified", True
    ):
        raise ValueError("original-source build reports mutation")
    git_commit = _git_value(["rev-parse", "HEAD"], workspace=workspace)
    git_status = _git_value(["status", "--porcelain"], workspace=workspace)
    dirty = bool(git_status)
    if dirty and not allow_dirty_for_dry_run:
        raise ValueError("production V2 contract freeze requires a clean committed worktree")
    code_hashes = {name: sha256_file(path) for name, path in code_paths.items()}
    evidence_index_contract = EvidenceIndexContractV2()
    payload = {
        "schema_version": "moatrader-historical-sparse-contract-freeze-v2/2",
        "status": "V2_PRE_OUTCOME_CONTRACT_FROZEN",
        "contract_tag": contract_tag.strip(),
        "frozen_at": datetime.now(SEOUL).isoformat(),
        "git_commit": git_commit,
        "worktree_dirty": dirty,
        "dry_run_only": dirty,
        "feature_policy": FEATURE_POLICY,
        "feature_policy_sha256": canonical_payload_sha256(FEATURE_POLICY),
        "evidence_index_contract": evidence_index_contract.model_dump(mode="json"),
        "evidence_index_contract_sha256": canonical_payload_sha256(
            evidence_index_contract.model_dump(mode="json")
        ),
        "applicability_policy_sha256": canonical_payload_sha256(
            rules.model_dump(mode="json")
        ),
        "deterministic_axis_policy": DETERMINISTIC_AXIS_POLICY,
        "deterministic_axis_policy_sha256": canonical_payload_sha256(
            DETERMINISTIC_AXIS_POLICY
        ),
        "evidence_priority": list(EVIDENCE_PRIORITY),
        "evidence_priority_sha256": canonical_payload_sha256(EVIDENCE_PRIORITY),
        "score_averaging_across_source_types": False,
        "parser_version": parser_freeze["parser_version"],
        "parser_prompt_sha256": parser_freeze["prompt_sha256"],
        "parser_freeze_sha256": sha256_file(parser_freeze_manifest),
        "signal_timestamp_policy": "FIRST_TRADABLE_TIMESTAMP_AFTER_CURRENT_REGULAR_FILING_AVAILABLE_AT",
        "last_grounded_days": rules.last_grounded_staleness_days,
        "last_grounded_role": "PREVIOUS_COMPARISON_BASIS_ONLY_NEVER_CURRENT_EVIDENCE",
        "code_sha256": code_hashes,
        "source_audit_sha256": sha256_file(source_audit_path),
        "source_build_manifest_sha256": sha256_file(source_manifest_path),
        "regular_pair_count": source_audit.get("regular_pair_count"),
        "arcana_regular_filing_count": source_audit.get("arcana_regular_filing_count"),
        "moatrader_regular_original_filing_count": source_audit.get(
            "moatrader_regular_original_filing_count"
        ),
        "outcome_stage_authorized": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze and tag the final V2 pre-outcome measurement contract before LOCKED use."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--rules-input", type=Path, required=True)
    parser.add_argument("--parser-freeze-manifest", type=Path, required=True)
    parser.add_argument("--source-build", type=Path, required=True)
    parser.add_argument("--contract-tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-dirty-for-dry-run", action="store_true")
    args = parser.parse_args()
    result = freeze_contract(
        workspace=args.workspace,
        rules_input=args.rules_input,
        parser_freeze_manifest=args.parser_freeze_manifest,
        source_build=args.source_build,
        contract_tag=args.contract_tag,
        output=args.output,
        allow_dirty_for_dry_run=args.allow_dirty_for_dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
