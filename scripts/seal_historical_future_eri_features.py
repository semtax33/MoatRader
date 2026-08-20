from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from moatrader.canonical.models import StatementType
from moatrader.expectations.future_eri import (
    EvidenceObservation,
    EvidenceScoreBand,
    EvidenceState,
    OperatingEvidenceAxis,
    evidence_score_band,
)
from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    AxisPairClassification,
    HistoricalEvidenceFeatureRowV1,
    HistoricalFilingPair,
    PairedAxisPacket,
    build_historical_evidence_feature_row,
    packet_id,
    seal_historical_evidence_features,
    sha256_file,
    validate_classification_grounding,
)


SEOUL = ZoneInfo("Asia/Seoul")


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


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _human_classification(row: dict[str, str]) -> AxisPairClassification | None:
    packet_id_value = str(row.get("packet_id") or "").strip()
    axis_value = str(row.get("axis") or "").strip()
    status_value = str(row.get("human_status") or "").strip()
    if not packet_id_value or not axis_value or not status_value:
        return None
    status = AxisClassificationStatus(status_value)
    payload: dict[str, Any] = {
        "packet_id": packet_id_value,
        "axis": axis_value,
        "status": status,
        "confidence": 1.0,
    }
    if status == AxisClassificationStatus.COMPLETE:
        payload.update(
            previous_state=int(str(row["human_previous_state"]).strip()),
            current_state=int(str(row["human_current_state"]).strip()),
            previous_source_id=str(row["human_previous_source_id"]).strip(),
            current_source_id=str(row["human_current_source_id"]).strip(),
            previous_source_span=str(row["human_previous_source_span"]).strip(),
            current_source_span=str(row["human_current_source_span"]).strip(),
        )
    return AxisPairClassification.model_validate(payload)


def evaluate_human_gold_quality(
    *,
    human_gold_path: Path,
    classifications: dict[str, AxisPairClassification],
    packets: dict[str, PairedAxisPacket],
    minimum_gold_per_axis: int = 20,
    minimum_overall_agreement: float = 0.80,
    minimum_axis_agreement: float = 0.70,
) -> dict[str, Any]:
    reviewed: list[tuple[AxisPairClassification, AxisPairClassification]] = []
    invalid_rows: list[dict[str, Any]] = []
    with human_gold_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for number, row in enumerate(csv.DictReader(handle), start=2):
            try:
                human = _human_classification(dict(row))
                if human is None:
                    continue
                packet = packets[human.packet_id]
                validate_classification_grounding(human, packet)
                machine = classifications[human.packet_id]
                reviewed.append((human, machine))
            except (KeyError, TypeError, ValueError) as exc:
                invalid_rows.append({"row": number, "error": str(exc)})

    by_axis: dict[str, dict[str, Any]] = {}
    all_matches = 0
    confusion: Counter[str] = Counter()
    for axis in OperatingEvidenceAxis:
        values = [(human, machine) for human, machine in reviewed if human.axis == axis]
        matches = 0
        for human, machine in values:
            human_key = (human.status, human.previous_state, human.current_state)
            machine_key = (machine.status, machine.previous_state, machine.current_state)
            matches += int(human_key == machine_key)
            confusion[f"{human.status.value}->{machine.status.value}"] += 1
        all_matches += matches
        agreement = matches / len(values) if values else 0.0
        by_axis[axis.value] = {
            "reviewed": len(values),
            "exact_status_and_state_pair_matches": matches,
            "agreement": agreement,
            "minimum_count_passed": len(values) >= minimum_gold_per_axis,
            "agreement_passed": agreement >= minimum_axis_agreement,
        }
    overall = all_matches / len(reviewed) if reviewed else 0.0
    gate = (
        not invalid_rows
        and bool(reviewed)
        and overall >= minimum_overall_agreement
        and all(
            item["minimum_count_passed"] and item["agreement_passed"]
            for item in by_axis.values()
        )
    )
    if not reviewed:
        status = "NOT_EVALUATED_NO_COMPLETED_HUMAN_GOLD"
    elif invalid_rows:
        status = "FAILED_INVALID_HUMAN_GOLD"
    elif gate:
        status = "PASSED"
    else:
        status = "FAILED_AGREEMENT_OR_COVERAGE"
    return {
        "schema_version": "moatrader-historical-label-quality-v1/1",
        "status": status,
        "gate_passed": gate,
        "reviewed_count": len(reviewed),
        "overall_exact_status_and_state_pair_agreement": overall,
        "minimum_gold_per_axis": minimum_gold_per_axis,
        "minimum_overall_agreement": minimum_overall_agreement,
        "minimum_axis_agreement": minimum_axis_agreement,
        "by_axis": by_axis,
        "status_confusion": dict(sorted(confusion.items())),
        "invalid_rows": invalid_rows,
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }


def _evidence_id(pair_id_value: str, axis: OperatingEvidenceAxis, side: str) -> str:
    digest = hashlib.sha256(f"{pair_id_value}|{axis.value}|{side}".encode("utf-8")).hexdigest()
    return f"EVID_{digest[:24]}"


def _observation(
    *,
    pair: HistoricalFilingPair,
    classification: AxisPairClassification,
    private: dict[str, Any],
    side: str,
) -> EvidenceObservation:
    if classification.status != AxisClassificationStatus.COMPLETE:
        raise ValueError("cannot build an observation from an abstention")
    is_previous = side == "previous"
    source_id = (
        classification.previous_source_id if is_previous else classification.current_source_id
    )
    source_span = (
        classification.previous_source_span if is_previous else classification.current_source_span
    )
    state = classification.previous_state if is_previous else classification.current_state
    assert source_id is not None and source_span is not None and state is not None
    source = private["sources"][source_id]
    if source["side"] != side:
        raise ValueError("classification source side does not match private source map")
    filing = pair.previous if is_previous else pair.current
    return EvidenceObservation(
        observation_id=_evidence_id(pair.pair_id, classification.axis, side.upper()),
        issuer_id=pair.ticker,
        fiscal_period=filing.fiscal_period_end.isoformat(),
        axis=classification.axis,
        state=state,
        source_document_id=filing.rcept_no,
        source_span=source_span,
        source_published_at=filing.published_at,
        available_at=filing.available_at,
        signal_timestamp=filing.signal_timestamp,
        statement_type=StatementType.DISCLOSED_FACT,
        classification_rule_id="BLINDED_PAIRED_AXIS_LLM_V1",
        materiality_rule_id="QUALITATIVE_AXIS_STATE_EQUAL_UNIT_V1",
        confidence=Decimal(str(classification.confidence)),
        materiality=Decimal(1),
    )


def run(
    *,
    input_build: Path,
    classification_build: Path,
    human_gold_path: Path,
    output: Path,
    minimum_gold_per_axis: int = 20,
    minimum_overall_agreement: float = 0.80,
    minimum_axis_agreement: float = 0.70,
    minimum_feature_rows_per_band: int = 20,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    packet_path = input_build / "llm" / "blinded-packets.jsonl"
    classification_path = classification_build / "classifications.jsonl"
    if not packet_path.is_file() or not classification_path.is_file():
        raise FileNotFoundError("blinded packets and completed classifications are required")
    packets = {item.packet_id: item for item in _read_jsonl(packet_path, PairedAxisPacket)}
    classifications = {
        item.packet_id: item for item in _read_jsonl(classification_path, AxisPairClassification)
    }
    if len(classifications) != len(set(classifications)):
        raise ValueError("classification packet IDs must be unique")
    for packet_id_value, classification in classifications.items():
        validate_classification_grounding(classification, packets[packet_id_value])

    quality = evaluate_human_gold_quality(
        human_gold_path=human_gold_path,
        classifications=classifications,
        packets=packets,
        minimum_gold_per_axis=minimum_gold_per_axis,
        minimum_overall_agreement=minimum_overall_agreement,
        minimum_axis_agreement=minimum_axis_agreement,
    )
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "label-quality-report.json", quality)
    if not quality["gate_passed"]:
        status = {
            "schema_version": "moatrader-historical-feature-seal-stage-v1/1",
            "status": "BLOCKED_LABEL_QUALITY_GATE",
            "feature_dataset_sealed": False,
            "private_source_map_opened": False,
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "per_pbr_role": "NOT_USED",
        }
        _write_json(output / "stage-status.json", status)
        return status

    pair_path = input_build / "private" / "filing-pairs.jsonl"
    private_path = input_build / "private" / "pair-source-map.jsonl"
    pairs = _read_jsonl(pair_path, HistoricalFilingPair)
    private_rows = {
        str(item["pair_id"]): item
        for item in (
            json.loads(line)
            for line in private_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    features: list[HistoricalEvidenceFeatureRowV1] = []
    exclusions: list[dict[str, Any]] = []
    axis_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    for pair in pairs:
        private = private_rows[pair.pair_id]
        previous: list[EvidenceObservation] = []
        current: list[EvidenceObservation] = []
        missing: list[dict[str, str]] = []
        for axis in OperatingEvidenceAxis:
            classification = classifications.get(packet_id(pair.pair_id, axis))
            if classification is None:
                missing.append({"axis": axis.value, "reason": "MISSING_CLASSIFICATION"})
                continue
            if classification.status != AxisClassificationStatus.COMPLETE:
                missing.append({"axis": axis.value, "reason": classification.status.value})
                continue
            previous.append(
                _observation(
                    pair=pair,
                    classification=classification,
                    private=private,
                    side="previous",
                )
            )
            current.append(
                _observation(
                    pair=pair,
                    classification=classification,
                    private=private,
                    side="current",
                )
            )
            assert classification.delta is not None
            axis_distribution[axis.value][str(classification.delta)] += 1
        if missing:
            exclusions.append({"pair_id": pair.pair_id, "reasons": missing})
            continue
        features.append(
            build_historical_evidence_feature_row(
                pair=pair,
                previous_observations=previous,
                current_observations=current,
                coverage_sector=str(private.get("coverage_sector") or "UNMAPPED"),
            )
        )

    _write_jsonl(
        output / "features-pre-outcome.jsonl",
        (item.model_dump(mode="json") for item in features),
    )
    _write_json(output / "feature-exclusions.json", exclusions)
    bands: Counter[str] = Counter()
    years: Counter[str] = Counter()
    sectors: Counter[str] = Counter()
    for feature in features:
        assert feature.evidence.evidence_f_score is not None
        bands[evidence_score_band(feature.evidence.evidence_f_score).value] += 1
        years[str(feature.signal_timestamp.year)] += 1
        sectors[feature.coverage_sector] += 1
    band_counts = {band.value: bands[band.value] for band in EvidenceScoreBand}
    feature_coverage = {
        "schema_version": "moatrader-historical-feature-coverage-v1/1",
        "total_filing_pairs": len(pairs),
        "six_axis_complete_features": len(features),
        "coverage": len(features) / len(pairs) if pairs else 0.0,
        "unique_issuers": len({item.issuer_id for item in features}),
        "by_signal_year": dict(sorted(years.items())),
        "by_sector": dict(sorted(sectors.items())),
        "axis_delta_distribution": {
            axis.value: {
                "-1": axis_distribution[axis.value]["-1"],
                "0": axis_distribution[axis.value]["0"],
                "1": axis_distribution[axis.value]["1"],
            }
            for axis in OperatingEvidenceAxis
        },
        "feature_band_counts": band_counts,
        "minimum_feature_rows_per_band": minimum_feature_rows_per_band,
        "all_bands_sufficient": all(
            value >= minimum_feature_rows_per_band for value in band_counts.values()
        ),
        "outcomes_opened": False,
        "returns_opened": False,
    }
    _write_json(output / "feature-coverage-report.json", feature_coverage)

    if not features:
        status = {
            "schema_version": "moatrader-historical-feature-seal-stage-v1/1",
            "status": "INCONCLUSIVE_DUE_TO_FEATURE_COVERAGE",
            "feature_dataset_sealed": False,
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "per_pbr_role": "NOT_USED",
        }
        _write_json(output / "stage-status.json", status)
        return status

    seal = seal_historical_evidence_features(features, sealed_at=datetime.now(SEOUL))
    _write_json(output / "feature-seal.json", seal.model_dump(mode="json"))
    outcome_gate = bool(feature_coverage["all_bands_sufficient"])
    status = {
        "schema_version": "moatrader-historical-feature-seal-stage-v1/1",
        "status": (
            "FEATURE_SEALED_OUTCOME_GATE_PASSED"
            if outcome_gate
            else "FEATURE_SEALED_INCONCLUSIVE_BAND_COVERAGE"
        ),
        "feature_dataset_sealed": True,
        "sealed_feature_count": len(features),
        "outcome_stage_authorized": outcome_gate,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
        "input_hashes": {
            "blinded_packets": sha256_file(packet_path),
            "classifications": sha256_file(classification_path),
            "human_gold": sha256_file(human_gold_path),
        },
    }
    _write_json(output / "stage-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the human-gold gate, build six-axis features, and seal before ERI outcomes."
    )
    parser.add_argument("--input-build", type=Path, required=True)
    parser.add_argument("--classification-build", type=Path, required=True)
    parser.add_argument("--human-gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-gold-per-axis", type=int, default=20)
    parser.add_argument("--minimum-overall-agreement", type=float, default=0.80)
    parser.add_argument("--minimum-axis-agreement", type=float, default=0.70)
    parser.add_argument("--minimum-feature-rows-per-band", type=int, default=20)
    args = parser.parse_args()
    result = run(
        input_build=args.input_build,
        classification_build=args.classification_build,
        human_gold_path=args.human_gold,
        output=args.output,
        minimum_gold_per_axis=args.minimum_gold_per_axis,
        minimum_overall_agreement=args.minimum_overall_agreement,
        minimum_axis_agreement=args.minimum_axis_agreement,
        minimum_feature_rows_per_band=args.minimum_feature_rows_per_band,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
