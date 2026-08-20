from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any, Iterable

from moatrader.expectations.eri_validation import (
    ClusteredEriObservationV1,
    evaluate_clustered_eri_mechanism,
)
from moatrader.expectations.future_eri import (
    CurrentExpectationStateV1,
    FutureEriFeatureRowV1,
    FutureEriOutcomeInputV1,
    build_future_eri_label,
    seal_feature_dataset,
)
from moatrader.expectations.historical_evidence import (
    HistoricalEvidenceDatasetSealV1,
    HistoricalEvidenceFeatureRowV1,
    canonical_payload_sha256,
    sha256_file,
)
from moatrader.valuation.assumptions import EconomicDcfAssumptions


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


def _assert_expectation_input_is_pre_outcome(records: list[dict[str, Any]]) -> None:
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


def run(
    *,
    feature_build: Path,
    expectation_input: Path,
    outcome_input: Path,
    trading_sessions_path: Path,
    output: Path,
    minimum_observations_per_band: int = 20,
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
            "schema_version": "moatrader-historical-future-eri-outcome-stage-v1/1",
            "status": "BLOCKED_FEATURE_COVERAGE_OR_QUALITY_GATE",
            "expectation_input_opened": False,
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "downstream_stage_authorized": False,
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
    if [item.observation_id for item in sorted(historical_rows, key=lambda item: item.observation_id)] != (
        historical_seal.observation_ids
    ):
        raise ValueError("historical feature rows do not match the sealed observation set")
    if canonical_payload_sha256(
        [item.model_dump(mode="json") for item in sorted(historical_rows, key=lambda item: item.observation_id)]
    ) != historical_seal.feature_dataset_sha256:
        raise ValueError("historical evidence dataset changed after sealing")

    expectation_records = _read_records(expectation_input)
    _assert_expectation_input_is_pre_outcome(expectation_records)
    expectation_by_id = {str(item["observation_id"]): item for item in expectation_records}
    if len(expectation_by_id) != len(expectation_records):
        raise ValueError("expectation observation IDs must be unique")
    final_features: list[FutureEriFeatureRowV1] = []
    missing_expectations: list[str] = []
    for historical in historical_rows:
        record = expectation_by_id.get(historical.observation_id)
        if record is None:
            missing_expectations.append(historical.observation_id)
            continue
        final_features.append(
            FutureEriFeatureRowV1(
                observation_id=historical.observation_id,
                evidence=historical.evidence,
                expectation_state=CurrentExpectationStateV1.model_validate(
                    record["expectation_state"]
                ),
                frozen_expectation_assumptions=EconomicDcfAssumptions.model_validate(
                    record["frozen_expectation_assumptions"]
                ),
            )
        )
    if not final_features:
        status = {
            "schema_version": "moatrader-historical-future-eri-outcome-stage-v1/1",
            "status": "BLOCKED_NO_COMPLETE_EXPECTATION_STATES",
            "expectation_input_opened": True,
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "downstream_stage_authorized": False,
        }
        _write_json(output / "missing-expectations.json", missing_expectations)
        _write_json(output / "stage-status.json", status)
        return status

    final_seal = seal_feature_dataset(
        final_features,
        sealed_at=max(item.evidence.signal_timestamp for item in final_features),
    )
    _write_jsonl(
        output / "features-with-frozen-expectations-pre-outcome.jsonl",
        (item.model_dump(mode="json") for item in final_features),
    )
    _write_json(output / "feature-seal-with-expectations.json", final_seal.model_dump(mode="json"))
    _write_json(output / "missing-expectations.json", missing_expectations)

    # The outcome vault is deliberately opened only after the final feature seal exists.
    outcome_records = _read_records(outcome_input)
    outcomes = {
        item.observation_id: item
        for item in (
            FutureEriOutcomeInputV1.model_validate(record) for record in outcome_records
        )
    }
    if len(outcomes) != len(outcome_records):
        raise ValueError("outcome observation IDs must be unique")
    sessions = _read_sessions(trading_sessions_path)
    labels = []
    missing_outcomes: list[str] = []
    feature_by_id = {item.observation_id: item for item in final_features}
    for feature in final_features:
        outcome = outcomes.get(feature.observation_id)
        if outcome is None:
            missing_outcomes.append(feature.observation_id)
            continue
        labels.append(
            build_future_eri_label(
                feature=feature,
                outcome=outcome,
                feature_seal=final_seal,
                trading_sessions=sessions,
            )
        )
    _write_jsonl(output / "future-eri-labels.jsonl", (item.model_dump(mode="json") for item in labels))
    _write_json(output / "missing-outcomes.json", missing_outcomes)

    reports: dict[str, Any] = {}
    for metric, attribute in (
        ("FROZEN_EQUITY_ERI_V1", "future_eri"),
        ("ENTERPRISE_ERI_DIAGNOSTIC", "enterprise_future_eri"),
    ):
        if labels:
            rows = [
                ClusteredEriObservationV1(
                    observation_id=label.observation_id,
                    issuer_id=feature_by_id[label.observation_id].evidence.issuer_id,
                    signal_timestamp=feature_by_id[label.observation_id].evidence.signal_timestamp,
                    evidence_f_score=feature_by_id[label.observation_id].evidence.evidence_f_score,
                    future_eri=getattr(label, attribute),
                )
                for label in labels
            ]
            reports[metric] = evaluate_clustered_eri_mechanism(
                rows,
                minimum_observations_per_band=minimum_observations_per_band,
                outcome_metric=metric,
            ).model_dump(mode="json")
        else:
            reports[metric] = {"status": "NOT_EVALUATED_NO_LABELS", "mechanism_gate_passed": False}
    _write_json(output / "mechanism-reports.json", reports)
    primary_passed = bool(reports["FROZEN_EQUITY_ERI_V1"].get("mechanism_gate_passed", False))
    status = {
        "schema_version": "moatrader-historical-future-eri-outcome-stage-v1/1",
        "status": "MECHANISM_PASSED" if primary_passed else "MECHANISM_REJECTED_OR_INCONCLUSIVE",
        "expectation_input_opened": True,
        "outcome_vault_opened": True,
        "label_count": len(labels),
        "return_data_opened": False,
        "downstream_stage_authorized": primary_passed,
        "primary_endpoint": "FROZEN_EQUITY_ERI_V1_FIXED_FIVE_BAND_MONOTONICITY",
        "enterprise_eri_role": "EV_BRIDGE_DIAGNOSTIC_ONLY",
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "stage-status.json", status)
    _write_json(
        output / "build-manifest.json",
        {
            "schema_version": "moatrader-historical-future-eri-outcome-build-v1/1",
            "historical_feature_seal_sha256": sha256_file(historical_seal_path),
            "expectation_input_sha256": sha256_file(expectation_input),
            "outcome_input_sha256": sha256_file(outcome_input),
            "trading_sessions_sha256": sha256_file(trading_sessions_path),
            "outcome_opened_only_after_final_feature_seal": True,
            "return_data_opened": False,
        },
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open 63-session ERI outcomes only after the historical evidence feature seal."
    )
    parser.add_argument("--feature-build", type=Path, required=True)
    parser.add_argument("--expectation-input", type=Path, required=True)
    parser.add_argument("--outcome-input", type=Path, required=True)
    parser.add_argument("--trading-sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-observations-per-band", type=int, default=20)
    args = parser.parse_args()
    result = run(
        feature_build=args.feature_build,
        expectation_input=args.expectation_input,
        outcome_input=args.outcome_input,
        trading_sessions_path=args.trading_sessions,
        output=args.output,
        minimum_observations_per_band=args.minimum_observations_per_band,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
