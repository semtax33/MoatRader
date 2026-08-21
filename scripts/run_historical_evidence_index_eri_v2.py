from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from moatrader.expectations.eri_null_fixtures import run_production_eri_null_fixtures
from moatrader.expectations.eri_validation import (
    EvidenceIndexEriObservationV2,
    evaluate_dual_evidence_index_eri_mechanism_v2,
)
from moatrader.expectations.future_eri import (
    CurrentExpectationStateV1,
    EvidenceIndexFutureEriFeatureRowV2,
    FutureEriOutcomeInputV1,
    build_evidence_index_future_eri_label_v2,
    seal_evidence_index_feature_dataset_v2,
    target_trading_session,
)
from moatrader.expectations.historical_evidence import sha256_file
from moatrader.expectations.historical_evidence_v2 import (
    DeterministicCoreIndexRowV2,
    FullEvidenceIndexRowV2,
    fixed_economic_breadth_band_v2,
)
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from scripts.audit_historical_future_eri_outcome_eligibility import (
    OutcomeEligibilityInventoryRowV1,
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        return [dict(item) for item in json.loads(text)]
    return [dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def _read_sessions(path: Path) -> list[date]:
    text = path.read_text(encoding="utf-8-sig").strip()
    raw = (
        json.loads(text)
        if text.startswith("[")
        else [line.split(",")[0] for line in text.splitlines()]
    )
    if raw and str(raw[0]).strip().lower() in {"date", "session", "trading_date"}:
        raw = raw[1:]
    sessions = sorted(
        {
            date.fromisoformat(str(value).strip()[:10])
            for value in raw
            if str(value).strip()
        }
    )
    if not sessions:
        raise ValueError("trading session input is empty")
    return sessions


def _assert_pre_outcome(records: list[dict[str, Any]]) -> None:
    prohibited = ("future_eri", "future_return", "actual_market_price")

    def inspect(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if any(fragment in str(key).lower() for fragment in prohibited):
                    raise ValueError(f"pre-outcome input contains outcome field: {path}.{key}")
                inspect(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{path}[{index}]")

    inspect(records, "pre_outcome")


def _blocked(output: Path, status: str, **extra: object) -> dict[str, Any]:
    payload = {
        "schema_version": "moatrader-historical-evidence-index-eri-stage-v2/1",
        "status": status,
        "expectation_input_opened": False,
        "eligibility_inventory_opened": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "downstream_stage_authorized": False,
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "per_pbr_role": "NOT_USED",
        **extra,
    }
    _write_json(output / "stage-status.json", payload)
    return payload


def run_evidence_index_eri_v2(
    *,
    full_index_build: Path,
    core_index_build: Path,
    expectation_input: Path,
    eligibility_inventory_input: Path,
    outcome_input: Path,
    trading_sessions_path: Path,
    output: Path,
    minimum_observations_per_band: int = 20,
    hac_lag_months: int = 3,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    full_stage_path = full_index_build / "stage-status.json"
    full_seal_path = full_index_build / "full-evidence-index-seal.json"
    if not full_stage_path.is_file() or not full_seal_path.is_file():
        return _blocked(output, "BLOCKED_FULL_INDEX_SEAL_MISSING")
    full_stage = _read_json(full_stage_path)
    full_seal = _read_json(full_seal_path)
    if not (
        full_stage.get("status") == "V2_FULL_EVIDENCE_INDEX_SEALED_OUTCOMES_CLOSED"
        and full_stage.get("outcome_stage_authorized") is True
        and full_stage.get("full_index_materialized") is True
        and full_stage.get("coverage_gate_passed") is True
        and full_stage.get("semantic_parser_gate_passed") is True
        and full_stage.get("full_evidence_index_seal_sha256")
        == sha256_file(full_seal_path)
        and full_seal.get("outcome_vault_opened") is False
        and full_seal.get("return_data_opened") is False
        and full_seal.get("value_data_opened") is False
        and full_seal.get("per_pbr_role") == "NOT_USED"
    ):
        return _blocked(output, "BLOCKED_FULL_INDEX_COVERAGE_OR_PARSER_GATE")

    core_manifest_path = core_index_build / "pre-outcome-index-manifest.json"
    if not core_manifest_path.is_file():
        return _blocked(output, "BLOCKED_CORE_SECONDARY_MISSING")
    core_manifest = _read_json(core_manifest_path)
    if not (
        core_manifest.get("status") == "V2_EVIDENCE_INDEX_CONTRACT_FROZEN_OUTCOMES_CLOSED"
        and core_manifest.get("deterministic_core_materialized") is True
        and core_manifest.get("outcome_vault_opened") is False
        and core_manifest.get("return_data_opened") is False
        and core_manifest.get("value_data_opened") is False
        and core_manifest.get("per_pbr_role") == "NOT_USED"
        and full_seal.get("input_hashes", {}).get("core_pre_outcome_manifest")
        == sha256_file(core_manifest_path)
    ):
        return _blocked(output, "BLOCKED_CORE_SECONDARY_SEAL_MISMATCH")

    full_path = full_index_build / "full-evidence-index-eligible-nobs2.jsonl"
    core_path = core_index_build / "deterministic-core-index-eligible-nobs2.jsonl"
    if not full_path.is_file() or not core_path.is_file():
        return _blocked(output, "BLOCKED_INDEX_ROW_ARTIFACT_MISSING")
    if full_seal.get("artifact_hashes", {}).get(
        "full_evidence_index_eligible_nobs2"
    ) != sha256_file(full_path):
        return _blocked(output, "BLOCKED_FULL_INDEX_ROW_HASH_MISMATCH")
    if core_manifest.get("artifact_hashes", {}).get(
        "deterministic_core_index_eligible_nobs2"
    ) != sha256_file(core_path):
        return _blocked(output, "BLOCKED_CORE_INDEX_ROW_HASH_MISMATCH")

    full_rows = {
        row.observation_id: row
        for row in (
            FullEvidenceIndexRowV2.model_validate(record)
            for record in _read_records(full_path)
        )
    }
    core_rows = {
        row.observation_id: row
        for row in (
            DeterministicCoreIndexRowV2.model_validate(record)
            for record in _read_records(core_path)
        )
    }
    if len(full_rows) != len(_read_records(full_path)):
        raise ValueError("Full Index observation IDs must be unique")
    if len(core_rows) != len(_read_records(core_path)):
        raise ValueError("Core Index observation IDs must be unique")
    common_index_ids = set(full_rows) & set(core_rows)
    if not common_index_ids:
        return _blocked(output, "BLOCKED_NO_COMMON_FULL_CORE_INDEX_PANEL")

    expectation_records = _read_records(expectation_input)
    _assert_pre_outcome(expectation_records)
    expectation_by_id = {
        str(item["observation_id"]): item for item in expectation_records
    }
    if len(expectation_by_id) != len(expectation_records):
        raise ValueError("expectation observation IDs must be unique")
    inventory_records = _read_records(eligibility_inventory_input)
    _assert_pre_outcome(inventory_records)
    inventory_rows = [
        OutcomeEligibilityInventoryRowV1.model_validate(item)
        for item in inventory_records
    ]
    inventory_by_id = {item.observation_id: item for item in inventory_rows}
    if len(inventory_by_id) != len(inventory_rows):
        raise ValueError("eligibility inventory observation IDs must be unique")
    sessions = _read_sessions(trading_sessions_path)

    feature_rows: list[EvidenceIndexFutureEriFeatureRowV2] = []
    exclusions: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    full_seal_sha = sha256_file(full_seal_path)
    for observation_id in sorted(common_index_ids):
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
                expectation_state = CurrentExpectationStateV1.model_validate(
                    record["expectation_state"]
                )
                assumptions = EconomicDcfAssumptions.model_validate(
                    record["frozen_expectation_assumptions"]
                )
            except (KeyError, TypeError, ValueError):
                reasons.append("INVALID_T_REVERSE_DCF")
        target = None
        try:
            target = target_trading_session(
                full.signal_timestamp.date(), sessions, horizon=63
            )
        except ValueError:
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
            for reason in reasons:
                reason_counts[reason] += 1
            exclusions.append(
                {"observation_id": observation_id, "reasons": sorted(set(reasons))}
            )
    _write_json(output / "outcome-eligibility-exclusions.json", exclusions)
    eligibility_report = {
        "schema_version": "moatrader-evidence-index-outcome-eligibility-v2/1",
        "common_full_core_index_count": len(common_index_ids),
        "label_eligible_count": len(feature_rows),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "exact_horizon_trading_sessions": 63,
        "outcome_values_opened": False,
        "return_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    eligibility_path = output / "outcome-eligibility-report.json"
    _write_json(eligibility_path, eligibility_report)
    if not feature_rows:
        status = _blocked(output, "BLOCKED_NO_T63_LABEL_ELIGIBLE_COMMON_PANEL")
        status["expectation_input_opened"] = True
        status["eligibility_inventory_opened"] = True
        _write_json(output / "stage-status.json", status)
        return status

    feature_seal = seal_evidence_index_feature_dataset_v2(
        feature_rows,
        sealed_at=max(item.signal_timestamp for item in feature_rows),
        full_index_seal_sha256=full_seal_sha,
    )
    feature_path = output / "features-with-frozen-expectations-pre-outcome.jsonl"
    feature_seal_path = output / "feature-seal-pre-outcome.json"
    _write_jsonl(feature_path, feature_rows)
    _write_json(feature_seal_path, feature_seal.model_dump(mode="json"))

    null_fixtures = run_production_eri_null_fixtures()
    null_fixture_path = output / "eri-null-fixtures.json"
    _write_json(null_fixture_path, null_fixtures)
    if not null_fixtures.get("all_passed", False):
        status = _blocked(output, "BLOCKED_PRODUCTION_ERI_NULL_FIXTURE")
        status["expectation_input_opened"] = True
        status["eligibility_inventory_opened"] = True
        status["production_null_fixtures_passed"] = False
        _write_json(output / "stage-status.json", status)
        return status

    # Outcome values are opened only after the common Full/Core feature panel is sealed.
    outcome_records = _read_records(outcome_input)
    outcomes = {
        item.observation_id: item
        for item in (
            FutureEriOutcomeInputV1.model_validate(record) for record in outcome_records
        )
    }
    if len(outcomes) != len(outcome_records):
        raise ValueError("outcome observation IDs must be unique")
    labels = []
    missing_outcomes: list[str] = []
    for feature in feature_rows:
        outcome = outcomes.get(feature.observation_id)
        if outcome is None:
            missing_outcomes.append(feature.observation_id)
            continue
        labels.append(
            build_evidence_index_future_eri_label_v2(
                feature=feature,
                outcome=outcome,
                feature_seal=feature_seal,
                trading_sessions=sessions,
            )
        )
    _write_jsonl(output / "future-eri-labels.jsonl", labels)
    _write_json(output / "missing-outcomes.json", missing_outcomes)
    feature_by_id = {item.observation_id: item for item in feature_rows}
    primary_rows = [
        EvidenceIndexEriObservationV2(
            observation_id=label.observation_id,
            issuer_id=feature_by_id[label.observation_id].issuer_id,
            signal_timestamp=feature_by_id[label.observation_id].signal_timestamp,
            index_role="FULL_PRIMARY",
            evidence_index=feature_by_id[label.observation_id].full_evidence_index,
            evidence_band=fixed_economic_breadth_band_v2(
                feature_by_id[label.observation_id].full_evidence_index
            ),
            nobs=feature_by_id[label.observation_id].full_nobs,
            future_eri=label.future_eri,
        )
        for label in labels
    ]
    secondary_rows = [
        EvidenceIndexEriObservationV2(
            observation_id=label.observation_id,
            issuer_id=feature_by_id[label.observation_id].issuer_id,
            signal_timestamp=feature_by_id[label.observation_id].signal_timestamp,
            index_role="CORE_SECONDARY",
            evidence_index=feature_by_id[label.observation_id].core_evidence_index,
            evidence_band=fixed_economic_breadth_band_v2(
                feature_by_id[label.observation_id].core_evidence_index
            ),
            nobs=feature_by_id[label.observation_id].core_nobs,
            future_eri=label.future_eri,
        )
        for label in labels
    ]
    report = (
        evaluate_dual_evidence_index_eri_mechanism_v2(
            primary_full=primary_rows,
            secondary_core=secondary_rows,
            minimum_observations_per_band=minimum_observations_per_band,
            hac_lag_months=hac_lag_months,
        )
        if labels
        else None
    )
    report_path = output / "dual-evidence-index-eri-report.json"
    _write_json(
        report_path,
        report.model_dump(mode="json")
        if report is not None
        else {"status": "NOT_EVALUATED_NO_LABELS", "mechanism_gate_passed": False},
    )
    primary_passed = bool(report and report.primary_full.mechanism_gate_passed)
    status = {
        "schema_version": "moatrader-historical-evidence-index-eri-stage-v2/1",
        "status": (
            "FULL_PRIMARY_MECHANISM_PASSED"
            if primary_passed
            else "FULL_PRIMARY_MECHANISM_REJECTED_OR_INCONCLUSIVE"
        ),
        "expectation_input_opened": True,
        "eligibility_inventory_opened": True,
        "outcome_vault_opened": True,
        "label_count": len(labels),
        "common_full_core_panel": True,
        "primary_endpoint": "FULL_EVIDENCE_INDEX_TO_FUTURE_ERI_T63",
        "secondary_endpoint": "CORE_EVIDENCE_INDEX_TO_FUTURE_ERI_T63",
        "future_eri_role": "DOWNSTREAM_OUTCOME_ONLY_NOT_SIGNAL_OR_RANKING",
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "return_data_opened": False,
        "downstream_stage_authorized": False,
        "production_null_fixtures_passed": True,
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "stage-status.json", status)
    _write_json(
        output / "build-manifest.json",
        {
            "schema_version": "moatrader-historical-evidence-index-eri-build-v2/1",
            "full_index_seal_sha256": full_seal_sha,
            "core_pre_outcome_manifest_sha256": sha256_file(core_manifest_path),
            "expectation_input_sha256": sha256_file(expectation_input),
            "eligibility_inventory_sha256": sha256_file(eligibility_inventory_input),
            "trading_sessions_sha256": sha256_file(trading_sessions_path),
            "outcome_input_sha256": sha256_file(outcome_input),
            "feature_seal_pre_outcome_sha256": sha256_file(feature_seal_path),
            "eligibility_report_sha256": sha256_file(eligibility_path),
            "production_null_fixture_sha256": sha256_file(null_fixture_path),
            "dual_mechanism_report_sha256": sha256_file(report_path),
            "outcome_opened_only_after_common_feature_seal": True,
            "exact_horizon_trading_sessions": 63,
            "return_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Full primary and Core secondary Evidence Indices against t+63 Future ERI."
    )
    parser.add_argument("--full-index-build", type=Path, required=True)
    parser.add_argument("--core-index-build", type=Path, required=True)
    parser.add_argument("--expectation-input", type=Path, required=True)
    parser.add_argument("--eligibility-inventory", type=Path, required=True)
    parser.add_argument("--outcome-input", type=Path, required=True)
    parser.add_argument("--trading-sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-observations-per-band", type=int, default=20)
    parser.add_argument("--hac-lag-months", type=int, default=3)
    args = parser.parse_args()
    result = run_evidence_index_eri_v2(
        full_index_build=args.full_index_build,
        core_index_build=args.core_index_build,
        expectation_input=args.expectation_input,
        eligibility_inventory_input=args.eligibility_inventory,
        outcome_input=args.outcome_input,
        trading_sessions_path=args.trading_sessions,
        output=args.output,
        minimum_observations_per_band=args.minimum_observations_per_band,
        hac_lag_months=args.hac_lag_months,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
