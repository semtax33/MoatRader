from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from moatrader.expectations.historical_evidence import canonical_payload_sha256, sha256_file


SEOUL = ZoneInfo("Asia/Seoul")
ORIGINAL_V1_TAG = "future-eri-v1-preoutcome"
V1R_TAG = "future-eri-v1r-three-section-preoutcome"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _git(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _git_optional(workspace: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def freeze_v1r_contract(
    *,
    workspace: Path,
    source_build: Path,
    original_v1_contract: Path,
    locked_set_preparation_manifest: Path,
    parser_freeze_manifest: Path,
    output: Path,
    expected_pair_count: int = 43_752,
    allow_dirty_for_dry_run: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"V1R contract output already exists: {output}")
    source_contract_path = source_build / "frozen-contract.json"
    source_audit_path = source_build / "source-audit.json"
    source_manifest_path = source_build / "build-manifest.json"
    code_paths = {
        "source_builder": workspace / "scripts" / "build_historical_future_eri_evidence.py",
        "historical_contract": workspace
        / "src"
        / "moatrader"
        / "expectations"
        / "historical_evidence.py",
        "locked_preparer": workspace / "scripts" / "prepare_historical_v1r_locked_set.py",
        "locked_evaluator": workspace
        / "scripts"
        / "evaluate_historical_evidence_parser_v1r.py",
        "feature_builder": workspace / "scripts" / "build_historical_complete_features_v1r.py",
        "feasibility_auditor": workspace / "scripts" / "audit_historical_v1r_feasibility.py",
    }
    for path in (
        original_v1_contract,
        locked_set_preparation_manifest,
        parser_freeze_manifest,
        source_contract_path,
        source_audit_path,
        source_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    # During initial implementation the two downstream code files may not yet exist. Production
    # freezes, however, must hash every file and therefore fail until the implementation is whole.
    missing_code = [path for path in code_paths.values() if not path.is_file()]
    if missing_code:
        raise FileNotFoundError(missing_code[0])

    original = json.loads(original_v1_contract.read_text(encoding="utf-8"))
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    prepared = json.loads(locked_set_preparation_manifest.read_text(encoding="utf-8"))
    parser_freeze = json.loads(parser_freeze_manifest.read_text(encoding="utf-8"))
    if original.get("schema_version") != "moatrader-historical-future-eri-feature-v1/1":
        raise ValueError("original V1 reference contract schema mismatch")
    if source_contract.get("schema_version") != "moatrader-historical-future-eri-feature-v1r/1":
        raise ValueError("source build is not a V1R contract build")
    if source_contract.get("feature") != original.get("feature"):
        raise ValueError("V1R changed the frozen V1 feature rule")
    if source_contract.get("feature_bands") != original.get("feature_bands"):
        raise ValueError("V1R changed the frozen V1 five-band rule")
    original_sources = set(original.get("source_scope", {}).get("included", []))
    v1r_sources = set(source_contract.get("source_scope", {}).get("included", []))
    required_new_sources = {
        "Arcana data-lake raw finance-comment HTML",
        "Arcana data-lake raw finance-statement HTML",
    }
    if not original_sources < v1r_sources or not required_new_sources.issubset(v1r_sources):
        raise ValueError("V1R source scope is not the strict three-section extension of V1")
    if source_audit.get("research_variant") != "V1R":
        raise ValueError("V1R freeze requires a V1R source audit")
    for key in (
        "all_arcana_sections_discovered",
        "all_arcana_sections_read_for_pairs",
        "all_arcana_sections_contributed_to_packets",
    ):
        if not source_audit.get(key, False):
            raise ValueError(f"V1R source audit did not pass {key}")
    if source_audit.get("regular_pair_count") != expected_pair_count:
        raise ValueError("V1R source build does not cover the frozen filing-pair universe")
    if source_audit.get("source_files_modified", True) or source_manifest.get(
        "source_files_modified", True
    ):
        raise ValueError("V1R source build reports original-source mutation")
    if prepared.get("status") != "V1R_SOURCE_STRATIFIED_LOCKED_SET_PREPARED_OUTCOME_BLIND":
        raise ValueError("V1R LOCKED set preparation has not passed")
    if prepared.get("v1_locked_rows_reused", True):
        raise ValueError("V1R LOCKED preparation reused V1 rows")
    if parser_freeze.get("schema_version") != "moatrader-historical-evidence-parser-freeze-v1r/1":
        raise ValueError("V1R parser freeze schema mismatch")
    if parser_freeze.get("locked_set_preparation_manifest_sha256") != sha256_file(
        locked_set_preparation_manifest
    ):
        raise ValueError("V1R parser freeze does not use the supplied LOCKED preparation")
    if any(
        payload.get("outcome_vault_opened", False)
        or payload.get("return_data_opened", False)
        or payload.get("value_data_opened", False)
        for payload in (source_audit, prepared, parser_freeze)
    ):
        raise ValueError("V1R pre-outcome inputs are contaminated")

    original_tag_commit = _git(workspace, "rev-list", "-n", "1", ORIGINAL_V1_TAG)
    if not original_tag_commit:
        raise ValueError("original V1 tag must remain present")
    git_commit = _git(workspace, "rev-parse", "HEAD")
    git_status = _git(workspace, "status", "--porcelain")
    dirty = bool(git_status)
    v1r_tag_commit = _git_optional(workspace, "rev-list", "-n", "1", V1R_TAG)
    tag_exists = bool(v1r_tag_commit)
    if dirty and not allow_dirty_for_dry_run:
        raise ValueError("production V1R freeze requires a clean committed worktree")
    production_frozen = not dirty and tag_exists and v1r_tag_commit == git_commit
    if not production_frozen and not allow_dirty_for_dry_run:
        raise ValueError("production V1R freeze requires the new V1R tag at HEAD")
    code_hashes = {name: sha256_file(path) for name, path in code_paths.items()}
    payload = {
        "schema_version": "moatrader-historical-v1r-preoutcome-contract/1",
        "status": (
            "V1R_PREOUTCOME_CONTRACT_FROZEN"
            if production_frozen
            else "V1R_PREOUTCOME_CONTRACT_DRY_RUN"
        ),
        "contract_tag": V1R_TAG,
        "contract_tag_exists": tag_exists,
        "contract_tag_commit": v1r_tag_commit or None,
        "original_v1_tag": ORIGINAL_V1_TAG,
        "original_v1_tag_commit": original_tag_commit,
        "original_v1_tag_preserved": True,
        "git_commit": git_commit,
        "worktree_dirty": dirty,
        "dry_run_only": not production_frozen,
        "frozen_at": datetime.now(SEOUL).isoformat(),
        "research_design": {
            "A_V1": "BUSINESS_INFO_PLUS_MOATRADER__SIX_AXIS_COMPLETE",
            "B_V1R": "THREE_ARCANA_SECTIONS_PLUS_MOATRADER__SIX_AXIS_COMPLETE",
            "C_V2": "THREE_ARCANA_SECTIONS_PLUS_MOATRADER__SPARSE_BREADTH",
            "A_TO_B": "SOURCE_COVERAGE_EFFECT_ONLY",
            "B_TO_C": "FEATURE_CONTRACT_EFFECT_ONLY",
        },
        "same_hypothesis_as_v1": True,
        "same_feature_policy_as_v1": True,
        "same_band_policy_as_v1": True,
        "feature_policy_sha256": canonical_payload_sha256(source_contract["feature"]),
        "band_policy_sha256": canonical_payload_sha256(source_contract["feature_bands"]),
        "source_policy_sha256": canonical_payload_sha256(source_contract["source_scope"]),
        "original_v1_contract_sha256": sha256_file(original_v1_contract),
        "source_contract_sha256": sha256_file(source_contract_path),
        "source_audit_sha256": sha256_file(source_audit_path),
        "source_build_manifest_sha256": sha256_file(source_manifest_path),
        "locked_set_preparation_manifest_sha256": sha256_file(
            locked_set_preparation_manifest
        ),
        "parser_freeze_sha256": sha256_file(parser_freeze_manifest),
        "parser_prompt_sha256": parser_freeze["prompt_sha256"],
        "minimum_rows_per_band": 20,
        "complete_case_required_axes": 6,
        "code_sha256": code_hashes,
        "outcome_stage_authorized": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze V1R as a source-corrected replication without changing V1."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--source-build", type=Path, required=True)
    parser.add_argument("--original-v1-contract", type=Path, required=True)
    parser.add_argument("--locked-set-preparation-manifest", type=Path, required=True)
    parser.add_argument("--parser-freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pair-count", type=int, default=43_752)
    parser.add_argument("--allow-dirty-for-dry-run", action="store_true")
    args = parser.parse_args()
    result = freeze_v1r_contract(
        workspace=args.workspace,
        source_build=args.source_build,
        original_v1_contract=args.original_v1_contract,
        locked_set_preparation_manifest=args.locked_set_preparation_manifest,
        parser_freeze_manifest=args.parser_freeze_manifest,
        output=args.output,
        expected_pair_count=args.expected_pair_count,
        allow_dirty_for_dry_run=args.allow_dirty_for_dry_run,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
