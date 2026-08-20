from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

from moatrader.expectations.future_eri import (
    CurrentExpectationStateV1,
    EriMechanismObservationV1,
    EriMonotonicityPolicyV1,
    EvidenceObservation,
    EvidenceVectorStatus,
    FutureEriFeatureRowV1,
    FutureEriOutcomeInputV1,
    build_fcff_evidence_vector,
    build_future_eri_label,
    evaluate_future_eri_monotonicity,
    seal_feature_dataset,
)
from moatrader.valuation.assumptions import EconomicDcfAssumptions


CONTRACT = {
    "schema_version": "moatrader-future-eri-v1-frozen-contract/1",
    "route": "FCFF",
    "horizon_trading_days": 63,
    "evidence_axes": [
        "DEMAND",
        "PRICE_MIX",
        "BACKLOG",
        "MARGIN",
        "INVENTORY_MISMATCH",
        "CAPACITY_CAPEX",
    ],
    "feature": {
        "primary_score": "equal-weight sum of six {-1,0,+1} comparable-period deltas",
        "materiality_score": "secondary diagnostic; direction * capped materiality",
        "missing_axis_policy": "EXCLUDE_NO_IMPUTATION",
        "llm_role": "STRUCTURED_FACT_CLASSIFICATION_ONLY",
        "required_trace": [
            "source_span",
            "classification",
            "confidence",
            "materiality_rule_id",
            "materiality numerator/denominator source IDs when numeric",
        ],
    },
    "label": "log(actual_market_price_t_plus_63 / counterfactual_fcff_value_t_plus_63)",
    "price_timestamps": {
        "entry": "Reverse DCF price timestamp must equal signal timestamp",
        "target": "target-session timestamp with an explicit source ID",
    },
    "counterfactual": {
        "updated": [
            "realized base revenue",
            "realized base NOPAT margin",
            "realized invested capital",
            "net debt",
            "diluted shares",
            "WACC",
        ],
        "frozen": [
            "revenue growth",
            "target NOPAT margin",
            "ROIIC",
            "stable-state assumptions",
            "reinvestment method",
        ],
        "cap_clock": "consume only completed signal-date anniversaries",
    },
    "mechanism_test": {
        "bands": ["-6:-3", "-2:-1", "0", "1:2", "3:6"],
        "required": [
            "all adjacent mean ERIs nondecreasing",
            "Q5 minus Q1 mean ERI positive",
            "score-to-ERI Spearman nonnegative",
            "minimum count satisfied in every band",
        ],
    },
    "prohibited_in_v1": [
        "machine learning",
        "future return test",
        "residual return test",
        "PER+PBR primary ranking",
        "any value-factor primary ranking",
    ],
    "primary_ranking_policy": "NONE_MECHANISM_ONLY",
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    stripped = text.lstrip()
    if not stripped:
        return []
    if stripped.startswith("["):
        payload = json.loads(text)
        if not isinstance(payload, list):
            raise ValueError(f"expected a JSON array: {path}")
        return [dict(item) for item in payload]
    return [dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def _read_sessions(path: Path) -> list[date]:
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError("trading-session file is empty")
    if text.startswith("["):
        values = json.loads(text)
    else:
        values = [line.split(",", maxsplit=1)[0].strip() for line in text.splitlines()]
        if values and values[0].lower() in {"date", "session", "trading_date"}:
            values = values[1:]
    sessions = sorted({date.fromisoformat(str(value)[:10]) for value in values if value})
    if not sessions:
        raise ValueError("trading-session file contains no dates")
    return sessions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_feature_rows(
    records: Sequence[dict[str, Any]],
) -> tuple[list[FutureEriFeatureRowV1], list[dict[str, Any]]]:
    rows: list[FutureEriFeatureRowV1] = []
    exclusions: list[dict[str, Any]] = []
    for record in records:
        observation_id = str(record.get("observation_id") or "")
        expectation = CurrentExpectationStateV1.model_validate(record["expectation_state"])
        current = [EvidenceObservation.model_validate(item) for item in record["current_observations"]]
        prior = [EvidenceObservation.model_validate(item) for item in record["prior_observations"]]
        vector = build_fcff_evidence_vector(
            issuer_id=expectation.issuer_id,
            signal_timestamp=expectation.signal_timestamp,
            current=current,
            prior=prior,
        )
        if vector.status != EvidenceVectorStatus.COMPLETE:
            exclusions.append(
                {
                    "observation_id": observation_id,
                    "reason": "INCOMPLETE_SIX_AXIS_EVIDENCE",
                    "missing_axes": [item.value for item in vector.missing_axes],
                }
            )
            continue
        rows.append(
            FutureEriFeatureRowV1(
                observation_id=observation_id,
                evidence=vector,
                expectation_state=expectation,
                frozen_expectation_assumptions=EconomicDcfAssumptions.model_validate(
                    record["frozen_expectation_assumptions"]
                ),
            )
        )
    return rows, exclusions


def run(
    *,
    feature_input: Path,
    outcome_input: Path,
    trading_sessions_path: Path,
    output: Path,
    minimum_observations_per_band: int = 20,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "frozen-contract.json", CONTRACT)

    # Stage 1: outcome_input is deliberately not opened before these artifacts
    # are written and the feature dataset is cryptographically sealed.
    feature_records = _read_records(feature_input)
    features, exclusions = build_feature_rows(feature_records)
    if not features:
        raise ValueError("no complete six-axis feature rows are available to seal")
    feature_payload = [item.model_dump(mode="json") for item in features]
    _write_jsonl(output / "features-pre-label.jsonl", feature_payload)
    sealed_at = max(item.evidence.signal_timestamp for item in features)
    seal = seal_feature_dataset(features, sealed_at=sealed_at)
    _write_json(output / "feature-seal.json", seal.model_dump(mode="json"))
    _write_json(output / "feature-exclusions.json", exclusions)

    # Stage 2: labels become visible only after feature-seal.json exists.
    outcome_records = _read_records(outcome_input)
    outcomes = {
        item.observation_id: item
        for item in (
            FutureEriOutcomeInputV1.model_validate(record) for record in outcome_records
        )
    }
    if len(outcomes) != len(outcome_records):
        raise ValueError("outcome observation_id values must be unique")
    sessions = _read_sessions(trading_sessions_path)
    labels = []
    missing_outcomes = []
    for feature in features:
        outcome = outcomes.get(feature.observation_id)
        if outcome is None:
            missing_outcomes.append(feature.observation_id)
            continue
        labels.append(
            build_future_eri_label(
                feature=feature,
                outcome=outcome,
                feature_seal=seal,
                trading_sessions=sessions,
            )
        )
    _write_jsonl(
        output / "future-eri-labels.jsonl",
        [item.model_dump(mode="json") for item in labels],
    )
    _write_json(output / "missing-outcomes.json", missing_outcomes)

    feature_by_id = {item.observation_id: item for item in features}
    mechanism_rows = [
        EriMechanismObservationV1(
            observation_id=label.observation_id,
            signal_timestamp=feature_by_id[label.observation_id].evidence.signal_timestamp,
            evidence_f_score=feature_by_id[label.observation_id].evidence.evidence_f_score,
            future_eri=label.future_eri,
        )
        for label in labels
    ]
    if mechanism_rows:
        report: dict[str, Any] = evaluate_future_eri_monotonicity(
            mechanism_rows,
            policy=EriMonotonicityPolicyV1(
                minimum_observations_per_band=minimum_observations_per_band
            ),
        ).model_dump(mode="json")
    else:
        report = {
            "schema_version": "moatrader-future-eri-monotonicity-v1/1",
            "status": "NOT_EVALUATED_NO_63_SESSION_LABELS",
            "mechanism_gate_passed": False,
            "ml_stage_authorized": False,
            "return_stage_status": "BLOCKED_MECHANISM_GATE_FAILED",
            "return_data_accessed": False,
            "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        }
    _write_json(output / "mechanism-report.json", report)

    final = {
        "schema_version": "moatrader-future-eri-v1-result/1",
        "feature_input_count": len(feature_records),
        "sealed_feature_count": len(features),
        "incomplete_feature_count": len(exclusions),
        "future_eri_label_count": len(labels),
        "missing_outcome_count": len(missing_outcomes),
        "mechanism_gate_passed": bool(report["mechanism_gate_passed"]),
        "ml_stage_authorized": bool(report["ml_stage_authorized"]),
        "return_stage_status": report["return_stage_status"],
        "return_data_accessed": False,
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "FINAL-RESULT.json", final)

    artifacts = {
        path.relative_to(output).as_posix(): _sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "build-manifest.json"
    }
    _write_json(
        output / "build-manifest.json",
        {
            "schema_version": "moatrader-future-eri-v1-build-manifest/1",
            "feature_input_sha256": _sha256_file(feature_input),
            "outcome_input_sha256": _sha256_file(outcome_input),
            "trading_sessions_sha256": _sha256_file(trading_sessions_path),
            "artifacts": artifacts,
            "credentials_persisted": False,
        },
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the return-blind FCFF Evidence -> 63-session Future ERI V1 dataset."
    )
    parser.add_argument("--feature-input", type=Path, required=True)
    parser.add_argument("--outcome-input", type=Path, required=True)
    parser.add_argument("--trading-sessions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-observations-per-band", type=int, default=20)
    args = parser.parse_args()
    final = run(
        feature_input=args.feature_input,
        outcome_input=args.outcome_input,
        trading_sessions_path=args.trading_sessions,
        output=args.output,
        minimum_observations_per_band=args.minimum_observations_per_band,
    )
    print(json.dumps(final, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
