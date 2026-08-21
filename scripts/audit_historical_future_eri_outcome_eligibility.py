from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.expectations.eri_null_fixtures import run_production_eri_null_fixtures
from moatrader.expectations.future_eri import (
    CurrentExpectationStateV1,
    FutureEriFeatureRowV1,
    seal_feature_dataset,
    target_trading_session,
)
from moatrader.expectations.historical_evidence import (
    HistoricalEvidenceDatasetSealV1,
    HistoricalEvidenceFeatureRowV1,
    canonical_payload_sha256,
    sha256_file,
)
from moatrader.valuation.assumptions import EconomicDcfAssumptions


class OutcomeEligibilityInventoryRowV1(ContractModel):
    schema_version: str = "moatrader-outcome-eligibility-inventory-row-v1/1"
    observation_id: str = Field(min_length=1)
    target_session: date
    target_price_at: datetime | None = None
    target_price_source_id: str | None = None
    realized_financials_available_at: datetime | None = None
    realized_financial_source_ids: list[str] = Field(default_factory=list)
    net_debt_source_id: str | None = None
    diluted_shares_source_id: str | None = None
    wacc_source_id: str | None = None
    outcome_values_included: Literal[False] = False
    return_values_included: Literal[False] = False

    @model_validator(mode="after")
    def metadata_is_point_in_time(self) -> "OutcomeEligibilityInventoryRowV1":
        if self.target_price_at is not None:
            if self.target_price_at.tzinfo is None or self.target_price_at.utcoffset() is None:
                raise ValueError("target_price_at must be timezone-aware")
            if self.target_price_at.date() != self.target_session:
                raise ValueError("target_price_at must fall on target_session")
        if self.realized_financials_available_at is not None:
            if (
                self.realized_financials_available_at.tzinfo is None
                or self.realized_financials_available_at.utcoffset() is None
            ):
                raise ValueError("realized_financials_available_at must be timezone-aware")
            if (
                self.target_price_at is not None
                and self.realized_financials_available_at > self.target_price_at
            ):
                raise ValueError("realized financial metadata is not PIT at target price")
        return self


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        return [dict(item) for item in json.loads(text)]
    return [dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def _read_sessions(path: Path) -> list[date]:
    text = path.read_text(encoding="utf-8-sig").strip()
    raw = json.loads(text) if text.startswith("[") else [line.split(",")[0] for line in text.splitlines()]
    if raw and str(raw[0]).strip().lower() in {"date", "session", "trading_date"}:
        raw = raw[1:]
    sessions = sorted({date.fromisoformat(str(value).strip()[:10]) for value in raw if str(value).strip()})
    if not sessions:
        raise ValueError("trading session input is empty")
    return sessions


def _assert_pre_outcome_expectations(records: list[dict[str, Any]]) -> None:
    prohibited = ("future_eri", "future_return", "target_price", "actual_market_price")

    def inspect(value: object, path: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if any(fragment in str(key).lower() for fragment in prohibited):
                    raise ValueError(f"expectation input contains outcome field: {path}.{key}")
                inspect(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                inspect(child, f"{path}[{index}]")

    inspect(records, "expectations")


def audit_outcome_eligibility(
    *,
    feature_build: Path,
    expectation_input: Path,
    eligibility_inventory_input: Path,
    trading_sessions_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    stage_path = feature_build / "stage-status.json"
    if not stage_path.is_file():
        raise FileNotFoundError("feature stage status is missing")
    feature_stage = json.loads(stage_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    if not feature_stage.get("outcome_stage_authorized", False):
        status = {
            "schema_version": "moatrader-outcome-eligibility-stage-v1/1",
            "status": "BLOCKED_FEATURE_COVERAGE_OR_QUALITY_GATE",
            "expectation_input_opened": False,
            "eligibility_inventory_opened": False,
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "outcome_stage_authorized": False,
        }
        _write_json(output / "stage-status.json", status)
        return status

    feature_path = feature_build / "features-pre-outcome.jsonl"
    historical_seal_path = feature_build / "feature-seal.json"
    historical_rows = [
        HistoricalEvidenceFeatureRowV1.model_validate(record)
        for record in _read_records(feature_path)
    ]
    historical_seal = HistoricalEvidenceDatasetSealV1.model_validate_json(
        historical_seal_path.read_text(encoding="utf-8")
    )
    sorted_rows = sorted(historical_rows, key=lambda item: item.observation_id)
    if [item.observation_id for item in sorted_rows] != historical_seal.observation_ids:
        raise ValueError("historical features do not match sealed observation IDs")
    if canonical_payload_sha256([item.model_dump(mode="json") for item in sorted_rows]) != (
        historical_seal.feature_dataset_sha256
    ):
        raise ValueError("historical feature dataset changed after sealing")

    expectation_records = _read_records(expectation_input)
    _assert_pre_outcome_expectations(expectation_records)
    expectation_by_id = {str(item["observation_id"]): item for item in expectation_records}
    if len(expectation_by_id) != len(expectation_records):
        raise ValueError("expectation observation IDs must be unique")
    inventory_rows = [
        OutcomeEligibilityInventoryRowV1.model_validate(item)
        for item in _read_records(eligibility_inventory_input)
    ]
    inventory_by_id = {item.observation_id: item for item in inventory_rows}
    if len(inventory_by_id) != len(inventory_rows):
        raise ValueError("eligibility inventory observation IDs must be unique")
    sessions = _read_sessions(trading_sessions_path)

    eligible_features: list[FutureEriFeatureRowV1] = []
    exclusions: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    counts = Counter(
        {
            "sealed_feature_rows": len(historical_rows),
            "has_t_reverse_dcf": 0,
            "has_exact_t_plus_63_session": 0,
            "has_target_price_metadata": 0,
            "has_t_plus_63_pit_financials": 0,
            "valid_equity_bridge_inputs": 0,
            "has_wacc_source": 0,
            "label_eligible": 0,
        }
    )
    for historical in historical_rows:
        reasons: list[str] = []
        record = expectation_by_id.get(historical.observation_id)
        feature: FutureEriFeatureRowV1 | None = None
        if record is None:
            reasons.append("MISSING_T_REVERSE_DCF")
        else:
            try:
                feature = FutureEriFeatureRowV1(
                    observation_id=historical.observation_id,
                    evidence=historical.evidence,
                    expectation_state=CurrentExpectationStateV1.model_validate(
                        record["expectation_state"]
                    ),
                    frozen_expectation_assumptions=EconomicDcfAssumptions.model_validate(
                        record["frozen_expectation_assumptions"]
                    ),
                )
                counts["has_t_reverse_dcf"] += 1
            except (KeyError, TypeError, ValueError):
                reasons.append("INVALID_T_REVERSE_DCF")

        expected_target: date | None = None
        try:
            expected_target = target_trading_session(
                historical.signal_timestamp.date(), sessions, horizon=63
            )
            counts["has_exact_t_plus_63_session"] += 1
        except ValueError as exc:
            reason = (
                "OUTCOME_WINDOW_INCOMPLETE"
                if "does not cover" in str(exc)
                else "SIGNAL_SESSION_ABSENT_FROM_CALENDAR"
            )
            reasons.append(reason)

        inventory = inventory_by_id.get(historical.observation_id)
        if inventory is None:
            reasons.append("MISSING_OUTCOME_ELIGIBILITY_INVENTORY")
        elif expected_target is not None:
            if inventory.target_session != expected_target:
                reasons.append("TARGET_SESSION_NOT_EXACT_T_PLUS_63")
            target_price_ok = bool(
                inventory.target_price_at is not None and inventory.target_price_source_id
            )
            if target_price_ok:
                counts["has_target_price_metadata"] += 1
            else:
                reasons.append("MISSING_TARGET_PRICE_METADATA")
            pit_financials_ok = bool(
                inventory.realized_financials_available_at is not None
                and inventory.realized_financial_source_ids
                and inventory.target_price_at is not None
                and inventory.realized_financials_available_at <= inventory.target_price_at
            )
            if pit_financials_ok:
                counts["has_t_plus_63_pit_financials"] += 1
            else:
                reasons.append("MISSING_T_PLUS_63_PIT_FINANCIALS")
            bridge_ok = bool(inventory.net_debt_source_id and inventory.diluted_shares_source_id)
            if bridge_ok:
                counts["valid_equity_bridge_inputs"] += 1
            else:
                reasons.append("MISSING_EQUITY_BRIDGE_INPUTS")
            if inventory.wacc_source_id:
                counts["has_wacc_source"] += 1
            else:
                reasons.append("MISSING_WACC_SOURCE")

        if not reasons and feature is not None:
            counts["label_eligible"] += 1
            eligible_features.append(feature)
        else:
            for reason in reasons:
                reason_counts[reason] += 1
            exclusions.append(
                {"observation_id": historical.observation_id, "reasons": sorted(set(reasons))}
            )

    _write_json(output / "outcome-eligibility-exclusions.json", exclusions)
    _write_jsonl(
        output / "eligible-features-with-frozen-expectations-pre-outcome.jsonl",
        (item.model_dump(mode="json") for item in eligible_features),
    )
    if eligible_features:
        seal = seal_feature_dataset(
            eligible_features,
            sealed_at=max(item.evidence.signal_timestamp for item in eligible_features),
        )
        _write_json(output / "eligible-feature-seal.json", seal.model_dump(mode="json"))

    null_fixtures = run_production_eri_null_fixtures()
    null_fixture_path = output / "eri-null-fixtures.json"
    _write_json(null_fixture_path, null_fixtures)
    report = {
        "schema_version": "moatrader-outcome-eligibility-report-v1/1",
        **dict(counts),
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "production_null_fixtures_passed": bool(null_fixtures["all_passed"]),
        "exact_horizon_trading_sessions": 63,
        "outcome_values_opened": False,
        "return_data_opened": False,
    }
    report_path = output / "outcome-eligibility-report.json"
    _write_json(report_path, report)
    authorized = bool(eligible_features) and bool(null_fixtures["all_passed"])
    status = {
        "schema_version": "moatrader-outcome-eligibility-stage-v1/1",
        "status": (
            "OUTCOME_ELIGIBILITY_AUDITED_AND_AUTHORIZED"
            if authorized
            else "OUTCOME_ELIGIBILITY_INCONCLUSIVE_OR_EMPTY"
        ),
        "expectation_input_opened": True,
        "eligibility_inventory_opened": True,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "outcome_stage_authorized": authorized,
        "label_eligible_count": len(eligible_features),
        "input_hashes": {
            "historical_feature_seal": sha256_file(historical_seal_path),
            "expectation_input": sha256_file(expectation_input),
            "eligibility_inventory": sha256_file(eligibility_inventory_input),
            "trading_sessions": sha256_file(trading_sessions_path),
            "eligibility_report": sha256_file(report_path),
            "production_null_fixtures": sha256_file(null_fixture_path),
        },
    }
    _write_json(output / "stage-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit exact t+63 outcome eligibility without opening outcome values."
    )
    parser.add_argument("--feature-build", type=Path, required=True)
    parser.add_argument("--expectation-input", type=Path, required=True)
    parser.add_argument("--eligibility-inventory", type=Path, required=True)
    parser.add_argument("--trading-sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_outcome_eligibility(
        feature_build=args.feature_build,
        expectation_input=args.expectation_input,
        eligibility_inventory_input=args.eligibility_inventory,
        trading_sessions_path=args.trading_sessions,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
