from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from moatrader.expectations.eri_null_fixtures import run_production_eri_null_fixtures
from moatrader.expectations.future_eri import (
    CurrentExpectationStateV1,
    EvidenceIndexFutureEriFeatureRowV2,
    seal_evidence_index_feature_dataset_v2,
    target_trading_session,
)
from moatrader.expectations.historical_evidence import sha256_file
from moatrader.expectations.historical_evidence_v2 import (
    DeterministicCoreIndexRowV2,
    FullEvidenceIndexRowV2,
)
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from scripts.audit_historical_future_eri_outcome_eligibility import (
    OutcomeEligibilityInventoryRowV1,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


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


def _read_sessions(path: Path) -> list[date]:
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    sessions = sorted({date.fromisoformat(str(value)[:10]) for value in raw})
    if not sessions:
        raise ValueError("trading session input is empty")
    return sessions


def _git_state(workspace: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _assert_pre_outcome(records: list[dict[str, Any]]) -> None:
    prohibited = (
        "future_eri",
        "future_return",
        "forward_return",
        "actual_market_price",
        "target_price_value",
        "target_close",
    )

    def inspect(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if any(fragment in str(key).casefold() for fragment in prohibited):
                    raise ValueError(f"pre-outcome input contains outcome field: {path}.{key}")
                inspect(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{path}[{index}]")

    inspect(records, "pre_outcome")


def _blocked(output: Path, status: str, **extra: object) -> dict[str, Any]:
    payload = {
        "schema_version": "moatrader-eri-feature-panel-stage-v2/1",
        "status": status,
        "feature_panel_sealed": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "per_pbr_role": "NOT_USED",
        **extra,
    }
    _write_json(output / "stage-status.json", payload)
    return payload


def seal_eri_feature_panel_v2(
    *,
    workspace: Path,
    full_index_build: Path,
    core_index_build: Path,
    pre_outcome_build: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production ERI feature sealing requires a clean worktree")

    full_stage_path = full_index_build / "stage-status.json"
    full_seal_path = full_index_build / "full-evidence-index-seal.json"
    core_manifest_path = core_index_build / "pre-outcome-index-manifest.json"
    pre_stage_path = pre_outcome_build / "stage-status.json"
    pre_seal_path = pre_outcome_build / "pre-outcome-input-seal.json"
    required = (
        full_stage_path,
        full_seal_path,
        core_manifest_path,
        pre_stage_path,
        pre_seal_path,
    )
    if any(not path.is_file() for path in required):
        return _blocked(output, "BLOCKED_REQUIRED_PRE_OUTCOME_SEAL_MISSING")

    full_stage = _read_json(full_stage_path)
    full_seal = _read_json(full_seal_path)
    core_manifest = _read_json(core_manifest_path)
    pre_stage = _read_json(pre_stage_path)
    pre_seal = _read_json(pre_seal_path)
    full_seal_sha = sha256_file(full_seal_path)
    if not (
        full_stage.get("status") == "V2_FULL_EVIDENCE_INDEX_SEALED_OUTCOMES_CLOSED"
        and full_stage.get("outcome_stage_authorized") is True
        and full_stage.get("full_evidence_index_seal_sha256") == full_seal_sha
        and full_seal.get("outcome_vault_opened") is False
        and full_seal.get("return_data_opened") is False
        and full_seal.get("value_data_opened") is False
        and full_seal.get("per_pbr_role") == "NOT_USED"
    ):
        return _blocked(output, "BLOCKED_FULL_INDEX_SEAL_INVALID")
    if not (
        core_manifest.get("status") == "V2_EVIDENCE_INDEX_CONTRACT_FROZEN_OUTCOMES_CLOSED"
        and core_manifest.get("outcome_vault_opened") is False
        and core_manifest.get("return_data_opened") is False
        and core_manifest.get("value_data_opened") is False
        and full_seal.get("input_hashes", {}).get("core_pre_outcome_manifest")
        == sha256_file(core_manifest_path)
    ):
        return _blocked(output, "BLOCKED_CORE_INDEX_SEAL_INVALID")
    if not (
        pre_stage.get("status") == "ERI_PRE_OUTCOME_INPUTS_PREPARED_OUTCOMES_CLOSED"
        and pre_stage.get("pre_outcome_input_seal_sha256") == sha256_file(pre_seal_path)
        and pre_seal.get("full_index_seal_sha256") == full_seal_sha
        and pre_seal.get("core_pre_outcome_manifest_sha256") == sha256_file(core_manifest_path)
        and pre_seal.get("outcome_stage_authorized") is True
        and pre_seal.get("outcome_vault_opened") is False
        and pre_seal.get("return_data_opened") is False
        and pre_seal.get("value_data_opened") is False
    ):
        return _blocked(output, "BLOCKED_PRE_OUTCOME_INPUT_SEAL_INVALID")

    full_path = full_index_build / "full-evidence-index-eligible-nobs2.jsonl"
    core_path = core_index_build / "deterministic-core-index-eligible-nobs2.jsonl"
    expectation_path = pre_outcome_build / "expectations-pre-outcome.jsonl"
    inventory_path = pre_outcome_build / "outcome-eligibility-inventory.jsonl"
    sessions_path = pre_outcome_build / "trading-sessions.json"
    if any(not path.is_file() for path in (full_path, core_path, expectation_path, inventory_path, sessions_path)):
        return _blocked(output, "BLOCKED_PRE_OUTCOME_ARTIFACT_MISSING")
    expected_hashes = pre_seal.get("artifact_hashes", {})
    if not (
        full_seal.get("artifact_hashes", {}).get("full_evidence_index_eligible_nobs2")
        == sha256_file(full_path)
        and core_manifest.get("artifact_hashes", {}).get("deterministic_core_index_eligible_nobs2")
        == sha256_file(core_path)
        and expected_hashes.get("expectations_pre_outcome") == sha256_file(expectation_path)
        and expected_hashes.get("outcome_eligibility_inventory") == sha256_file(inventory_path)
        and expected_hashes.get("trading_sessions") == sha256_file(sessions_path)
    ):
        return _blocked(output, "BLOCKED_PRE_OUTCOME_ARTIFACT_HASH_MISMATCH")

    full_records = _read_records(full_path)
    core_records = _read_records(core_path)
    expectation_records = _read_records(expectation_path)
    inventory_records = _read_records(inventory_path)
    _assert_pre_outcome(expectation_records)
    _assert_pre_outcome(inventory_records)
    full_rows = {
        row.observation_id: row
        for row in (FullEvidenceIndexRowV2.model_validate(item) for item in full_records)
    }
    core_rows = {
        row.observation_id: row
        for row in (DeterministicCoreIndexRowV2.model_validate(item) for item in core_records)
    }
    expectation_by_id = {str(item["observation_id"]): item for item in expectation_records}
    inventory_rows = [OutcomeEligibilityInventoryRowV1.model_validate(item) for item in inventory_records]
    inventory_by_id = {item.observation_id: item for item in inventory_rows}
    if len(full_rows) != len(full_records) or len(core_rows) != len(core_records):
        raise ValueError("Full/Core Index observation IDs must be unique")
    if len(expectation_by_id) != len(expectation_records) or len(inventory_by_id) != len(inventory_rows):
        raise ValueError("pre-outcome observation IDs must be unique")
    sessions = _read_sessions(sessions_path)

    feature_rows: list[EvidenceIndexFutureEriFeatureRowV2] = []
    exclusions: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    common_ids = set(full_rows) & set(core_rows)
    for observation_id in sorted(common_ids):
        full = full_rows[observation_id]
        core = core_rows[observation_id]
        reasons: list[str] = []
        record = expectation_by_id.get(observation_id)
        expectation_state = None
        assumptions = None
        if record is None:
            reasons.append("MISSING_T_REVERSE_DCF")
        else:
            try:
                expectation_state = CurrentExpectationStateV1.model_validate(record["expectation_state"])
                assumptions = EconomicDcfAssumptions.model_validate(record["frozen_expectation_assumptions"])
            except (KeyError, TypeError, ValueError):
                reasons.append("INVALID_T_REVERSE_DCF")
        try:
            target = target_trading_session(full.signal_timestamp.date(), sessions, horizon=63)
        except ValueError:
            target = None
            reasons.append("MISSING_EXACT_T_PLUS_63_SESSION")
        inventory = inventory_by_id.get(observation_id)
        if inventory is None:
            reasons.append("MISSING_OUTCOME_ELIGIBILITY_INVENTORY")
        elif target is not None:
            if inventory.target_session != target:
                reasons.append("TARGET_SESSION_NOT_EXACT_T_PLUS_63")
            if not (inventory.target_price_at and inventory.target_price_source_id):
                reasons.append("MISSING_TARGET_PRICE_METADATA")
            if not (
                inventory.realized_financials_available_at
                and inventory.realized_financial_source_ids
                and inventory.target_price_at
                and inventory.realized_financials_available_at <= inventory.target_price_at
            ):
                reasons.append("MISSING_T_PLUS_63_PIT_FINANCIALS")
            if not (inventory.net_debt_source_id and inventory.diluted_shares_source_id):
                reasons.append("MISSING_EQUITY_BRIDGE_INPUTS")
            if not inventory.wacc_source_id:
                reasons.append("MISSING_WACC_SOURCE")
        if not reasons and expectation_state is not None and assumptions is not None:
            if full.full_evidence_index is None or core.core_evidence_index is None:
                raise AssertionError("eligible Full/Core rows must have indices")
            feature_rows.append(
                EvidenceIndexFutureEriFeatureRowV2(
                    observation_id=observation_id,
                    issuer_id=full.issuer_id,
                    signal_timestamp=full.signal_timestamp,
                    full_evidence_index=full.full_evidence_index,
                    full_nobs=full.nobs,
                    core_evidence_index=core.core_evidence_index,
                    core_nobs=core.nobs,
                    full_index_row_sha256=full.row_sha256,
                    core_index_row_sha256=core.row_sha256,
                    full_index_seal_sha256=full_seal_sha,
                    expectation_state=expectation_state,
                    frozen_expectation_assumptions=assumptions,
                )
            )
        else:
            unique_reasons = sorted(set(reasons))
            reason_counts.update(unique_reasons)
            exclusions.append({"observation_id": observation_id, "reasons": unique_reasons})

    exclusion_path = output / "outcome-eligibility-exclusions.json"
    report_path = output / "outcome-eligibility-report.json"
    _write_json(exclusion_path, exclusions)
    _write_json(
        report_path,
        {
            "schema_version": "moatrader-evidence-index-outcome-eligibility-v2/1",
            "common_full_core_index_count": len(common_ids),
            "label_eligible_count": len(feature_rows),
            "exclusion_reason_counts": dict(sorted(reason_counts.items())),
            "exact_horizon_trading_sessions": 63,
            "outcome_values_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    if not feature_rows:
        return _blocked(output, "BLOCKED_NO_T63_LABEL_ELIGIBLE_COMMON_PANEL")

    feature_seal = seal_evidence_index_feature_dataset_v2(
        feature_rows,
        sealed_at=max(item.signal_timestamp for item in feature_rows),
        full_index_seal_sha256=full_seal_sha,
    )
    feature_path = output / "features-with-frozen-expectations-pre-outcome.jsonl"
    feature_seal_path = output / "feature-seal-pre-outcome.json"
    null_fixture_path = output / "eri-null-fixtures.json"
    _write_jsonl(feature_path, feature_rows)
    _write_json(feature_seal_path, feature_seal.model_dump(mode="json"))
    null_fixtures = run_production_eri_null_fixtures()
    _write_json(null_fixture_path, null_fixtures)
    if not null_fixtures.get("all_passed", False):
        return _blocked(output, "BLOCKED_PRODUCTION_ERI_NULL_FIXTURE")

    stage = {
        "schema_version": "moatrader-eri-feature-panel-stage-v2/1",
        "status": "ERI_FEATURE_PANEL_SEALED_OUTCOMES_CLOSED",
        "git_commit": commit,
        "worktree_dirty": False,
        "script_sha256": sha256_file(Path(__file__)),
        "feature_panel_sealed": True,
        "feature_count": len(feature_rows),
        "full_index_seal_sha256": full_seal_sha,
        "pre_outcome_input_seal_sha256": sha256_file(pre_seal_path),
        "feature_dataset_sha256": feature_seal.feature_dataset_sha256,
        "feature_artifact_sha256": sha256_file(feature_path),
        "feature_seal_sha256": sha256_file(feature_seal_path),
        "eligibility_report_sha256": sha256_file(report_path),
        "eligibility_exclusions_sha256": sha256_file(exclusion_path),
        "production_null_fixture_sha256": sha256_file(null_fixture_path),
        "production_null_fixtures_passed": True,
        "outcome_stage_authorized": True,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "per_pbr_role": "NOT_USED",
    }
    stage_path = output / "stage-status.json"
    _write_json(stage_path, stage)
    _write_json(
        output / "build-manifest.json",
        {
            **stage,
            "stage_status_sha256": sha256_file(stage_path),
            "expectation_input_sha256": sha256_file(expectation_path),
            "eligibility_inventory_sha256": sha256_file(inventory_path),
            "trading_sessions_sha256": sha256_file(sessions_path),
            "outcome_values_opened_before_feature_seal": False,
        },
    )
    return stage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Seal the common Full/Core ERI feature panel before opening t+63 outcomes."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--full-index-build", type=Path, required=True)
    parser.add_argument("--core-index-build", type=Path, required=True)
    parser.add_argument("--pre-outcome-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = seal_eri_feature_panel_v2(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
