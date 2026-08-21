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


def _read_packets_by_id(path: Path, wanted: set[str]) -> dict[str, PairedAxisPacket]:
    packets: dict[str, PairedAxisPacket] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            packet = PairedAxisPacket.model_validate_json(line)
            if packet.packet_id in wanted:
                if packet.packet_id in packets:
                    raise ValueError(f"duplicate blinded packet id: {packet.packet_id}")
                packets[packet.packet_id] = packet
    missing = wanted - set(packets)
    if missing:
        raise KeyError(f"classifications reference missing packets: {sorted(missing)[:5]}")
    return packets


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
    gold_split: str | None = None,
) -> dict[str, Any]:
    reviewed: list[tuple[AxisPairClassification, AxisPairClassification]] = []
    invalid_rows: list[dict[str, Any]] = []
    grounding_validated_count = 0
    with human_gold_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if gold_split is not None and "gold_split" not in (reader.fieldnames or []):
            invalid_rows.append({"row": 1, "error": "human gold is missing gold_split"})
        for number, row in enumerate(reader, start=2):
            if gold_split is not None and str(row.get("gold_split") or "").strip() != gold_split:
                continue
            try:
                human = _human_classification(dict(row))
                if human is None:
                    continue
                packet = packets[human.packet_id]
                validate_classification_grounding(human, packet)
                machine = classifications[human.packet_id]
                validate_classification_grounding(machine, packet)
                grounding_validated_count += 1
                reviewed.append((human, machine))
            except (KeyError, TypeError, ValueError) as exc:
                invalid_rows.append({"row": number, "error": str(exc)})

    by_axis: dict[str, dict[str, Any]] = {}
    all_matches = 0
    confusion: Counter[str] = Counter()
    state_confusion: Counter[str] = Counter()
    human_status_distribution: Counter[str] = Counter()
    machine_status_distribution: Counter[str] = Counter()
    fatal_direction_flip_count = 0
    false_stable_count = 0
    machine_abstention_count = 0
    for axis in OperatingEvidenceAxis:
        values = [(human, machine) for human, machine in reviewed if human.axis == axis]
        matches = 0
        axis_fatal_flips = 0
        axis_false_stable = 0
        axis_machine_abstentions = 0
        for human, machine in values:
            human_key = (human.status, human.previous_state, human.current_state)
            machine_key = (machine.status, machine.previous_state, machine.current_state)
            matches += int(human_key == machine_key)
            confusion[f"{human.status.value}->{machine.status.value}"] += 1
            human_status_distribution[human.status.value] += 1
            machine_status_distribution[machine.status.value] += 1
            if machine.status != AxisClassificationStatus.COMPLETE:
                axis_machine_abstentions += 1
            if machine.status == AxisClassificationStatus.COMPLETE:
                machine_states = (machine.previous_state, machine.current_state)
                if human.status == AxisClassificationStatus.COMPLETE:
                    human_states = (human.previous_state, human.current_state)
                    for human_state, machine_state in zip(human_states, machine_states, strict=True):
                        assert human_state is not None and machine_state is not None
                        state_confusion[f"{human_state.value}->{machine_state.value}"] += 1
                        if human_state.value * machine_state.value == -1:
                            axis_fatal_flips += 1
                        if machine_state.value == 0 and human_state.value != 0:
                            axis_false_stable += 1
                elif any(state is not None and state.value == 0 for state in machine_states):
                    axis_false_stable += 1
        all_matches += matches
        fatal_direction_flip_count += axis_fatal_flips
        false_stable_count += axis_false_stable
        machine_abstention_count += axis_machine_abstentions
        agreement = matches / len(values) if values else 0.0
        by_axis[axis.value] = {
            "reviewed": len(values),
            "exact_status_and_state_pair_matches": matches,
            "agreement": agreement,
            "machine_abstention_count": axis_machine_abstentions,
            "machine_abstention_rate": axis_machine_abstentions / len(values) if values else 0.0,
            "fatal_direction_flip_count": axis_fatal_flips,
            "false_stable_count": axis_false_stable,
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
        "evaluated_gold_split": gold_split or "ALL",
        "reviewed_count": len(reviewed),
        "overall_exact_status_and_state_pair_agreement": overall,
        "minimum_gold_per_axis": minimum_gold_per_axis,
        "minimum_overall_agreement": minimum_overall_agreement,
        "minimum_axis_agreement": minimum_axis_agreement,
        "by_axis": by_axis,
        "status_confusion": dict(sorted(confusion.items())),
        "state_confusion": dict(sorted(state_confusion.items())),
        "human_status_distribution": dict(sorted(human_status_distribution.items())),
        "machine_status_distribution": dict(sorted(machine_status_distribution.items())),
        "machine_abstention_count": machine_abstention_count,
        "machine_abstention_rate": machine_abstention_count / len(reviewed) if reviewed else 0.0,
        "fatal_direction_flip_count": fatal_direction_flip_count,
        "false_stable_count": false_stable_count,
        "source_span_grounding_validated_count": grounding_validated_count,
        "source_span_grounding_rate": grounding_validated_count / len(reviewed) if reviewed else 0.0,
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
    quality_classification_build: Path | None = None,
    human_gold_path: Path,
    output: Path,
    minimum_gold_per_axis: int = 20,
    minimum_overall_agreement: float = 0.80,
    minimum_axis_agreement: float = 0.70,
    minimum_feature_rows_per_band: int = 20,
    gold_split: str = "LOCKED_TEST",
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    packet_path = input_build / "llm" / "blinded-packets.jsonl"
    classification_path = classification_build / "classifications.jsonl"
    quality_classification_path = (
        quality_classification_build / "classifications.jsonl"
        if quality_classification_build is not None
        else classification_path
    )
    if (
        not packet_path.is_file()
        or not classification_path.is_file()
        or not quality_classification_path.is_file()
    ):
        raise FileNotFoundError("blinded packets and completed classifications are required")
    classification_rows = _read_jsonl(classification_path, AxisPairClassification)
    quality_classification_rows = _read_jsonl(
        quality_classification_path, AxisPairClassification
    )
    classifications = {item.packet_id: item for item in classification_rows}
    quality_classifications = {
        item.packet_id: item for item in quality_classification_rows
    }
    if len(classifications) != len(classification_rows):
        raise ValueError("feature classification packet IDs must be unique")
    if len(quality_classifications) != len(quality_classification_rows):
        raise ValueError("quality classification packet IDs must be unique")
    packets = _read_packets_by_id(
        packet_path,
        set(classifications) | set(quality_classifications),
    )
    for packet_id_value, classification in classifications.items():
        validate_classification_grounding(classification, packets[packet_id_value])
    for packet_id_value, classification in quality_classifications.items():
        validate_classification_grounding(classification, packets[packet_id_value])

    quality = evaluate_human_gold_quality(
        human_gold_path=human_gold_path,
        classifications=quality_classifications,
        packets=packets,
        minimum_gold_per_axis=minimum_gold_per_axis,
        minimum_overall_agreement=minimum_overall_agreement,
        minimum_axis_agreement=minimum_axis_agreement,
        gold_split=gold_split,
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
            "evaluated_gold_split": gold_split,
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
    axis_status_distribution: dict[str, Counter[str]] = defaultdict(Counter)
    exclusion_reason_counts: Counter[str] = Counter()
    for classification in classifications.values():
        axis_status_distribution[classification.axis.value][classification.status.value] += 1
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
            for item in missing:
                exclusion_reason_counts[f"{item['axis']}:{item['reason']}"] += 1
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
    issuers: Counter[str] = Counter()
    months: Counter[str] = Counter()
    for feature in features:
        assert feature.evidence.evidence_f_score is not None
        bands[evidence_score_band(feature.evidence.evidence_f_score).value] += 1
        years[str(feature.signal_timestamp.year)] += 1
        sectors[feature.coverage_sector] += 1
        issuers[feature.issuer_id] += 1
        months[feature.signal_timestamp.strftime("%Y-%m")] += 1
    band_counts = {band.value: bands[band.value] for band in EvidenceScoreBand}
    issuer_total = sum(issuers.values())
    month_total = sum(months.values())
    top_issuer_counts = issuers.most_common(10)
    top_month_counts = months.most_common(10)
    feature_coverage = {
        "schema_version": "moatrader-historical-feature-coverage-v1/1",
        "total_filing_pairs": len(pairs),
        "six_axis_complete_features": len(features),
        "excluded_pair_count": len(exclusions),
        "coverage": len(features) / len(pairs) if pairs else 0.0,
        "unique_issuers": len({item.issuer_id for item in features}),
        "unique_signal_months": len(months),
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
        "axis_classification_status_distribution": {
            axis.value: {
                status.value: axis_status_distribution[axis.value][status.value]
                for status in AxisClassificationStatus
            }
            for axis in OperatingEvidenceAxis
        },
        "llm_abstention_by_axis": {
            axis.value: (
                axis_status_distribution[axis.value][AxisClassificationStatus.INSUFFICIENT_EVIDENCE.value]
                + axis_status_distribution[axis.value][AxisClassificationStatus.AMBIGUOUS.value]
            )
            for axis in OperatingEvidenceAxis
        },
        "exclusion_reason_counts": dict(sorted(exclusion_reason_counts.items())),
        "issuer_concentration": {
            "top_issuer_count": top_issuer_counts[0][1] if top_issuer_counts else 0,
            "top_issuer_share": top_issuer_counts[0][1] / issuer_total if top_issuer_counts else 0.0,
            "top_10_issuer_share": sum(count for _, count in top_issuer_counts) / issuer_total
            if issuer_total
            else 0.0,
            "top_10": [{"issuer_id": key, "count": count} for key, count in top_issuer_counts],
        },
        "month_concentration": {
            "top_month_count": top_month_counts[0][1] if top_month_counts else 0,
            "top_month_share": top_month_counts[0][1] / month_total if top_month_counts else 0.0,
            "top_10_month_share": sum(count for _, count in top_month_counts) / month_total
            if month_total
            else 0.0,
            "top_10": [{"signal_month": key, "count": count} for key, count in top_month_counts],
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
            "status": "INCONCLUSIVE_DUE_TO_COMPLETE_CASE_COVERAGE_COLLAPSE",
            "research_interpretation": "V1_HYPOTHESIS_NOT_REJECTED_FEATURE_CONTRACT_COLLAPSED_SAMPLE",
            "tombstone": True,
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
            else "INCONCLUSIVE_DUE_TO_COMPLETE_CASE_COVERAGE_COLLAPSE"
        ),
        "research_interpretation": (
            "OUTCOME_GATE_PASSED"
            if outcome_gate
            else "V1_HYPOTHESIS_NOT_REJECTED_FEATURE_CONTRACT_COLLAPSED_SAMPLE"
        ),
        "tombstone": not outcome_gate,
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
            "quality_classifications": sha256_file(quality_classification_path),
        },
        "evaluated_gold_split": gold_split,
    }
    _write_json(output / "stage-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply the human-gold gate, build six-axis features, and seal before ERI outcomes."
    )
    parser.add_argument("--input-build", type=Path, required=True)
    parser.add_argument("--classification-build", type=Path, required=True)
    parser.add_argument(
        "--quality-classification-build",
        type=Path,
        help="Optional separate classification run used only for the human-gold gate.",
    )
    parser.add_argument("--human-gold", type=Path, required=True)
    parser.add_argument("--gold-split", choices=["DEV", "LOCKED_TEST"], default="LOCKED_TEST")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-gold-per-axis", type=int, default=20)
    parser.add_argument("--minimum-overall-agreement", type=float, default=0.80)
    parser.add_argument("--minimum-axis-agreement", type=float, default=0.70)
    parser.add_argument("--minimum-feature-rows-per-band", type=int, default=20)
    args = parser.parse_args()
    result = run(
        input_build=args.input_build,
        classification_build=args.classification_build,
        quality_classification_build=args.quality_classification_build,
        human_gold_path=args.human_gold,
        output=args.output,
        minimum_gold_per_axis=args.minimum_gold_per_axis,
        minimum_overall_agreement=args.minimum_overall_agreement,
        minimum_axis_agreement=args.minimum_axis_agreement,
        minimum_feature_rows_per_band=args.minimum_feature_rows_per_band,
        gold_split=args.gold_split,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
