from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    AxisPairClassification,
    PairedAxisPacket,
    sha256_file,
    validate_classification_grounding,
)
from scripts.prepare_historical_v1r_locked_set import V1RSourceStratum
from scripts.seal_historical_future_eri_features import evaluate_human_gold_quality


SEOUL = ZoneInfo("Asia/Seoul")
GOLD_SPLIT = "V1R_LOCKED_TEST"
GOLD_CONTRACT = "V1R_THREE_SECTION_SOURCE_STRATIFIED_LOCKED"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def _human_classification(row: dict[str, str]) -> AxisPairClassification:
    status = AxisClassificationStatus(str(row.get("human_status") or "").strip())
    payload: dict[str, Any] = {
        "packet_id": str(row.get("packet_id") or "").strip(),
        "axis": str(row.get("axis") or "").strip(),
        "status": status,
        "confidence": 1.0,
    }
    if status == AxisClassificationStatus.COMPLETE:
        payload.update(
            previous_state=int(str(row.get("human_previous_state") or "").strip()),
            current_state=int(str(row.get("human_current_state") or "").strip()),
            previous_source_id=str(row.get("human_previous_source_id") or "").strip(),
            current_source_id=str(row.get("human_current_source_id") or "").strip(),
            previous_source_span=str(row.get("human_previous_source_span") or "").strip(),
            current_source_span=str(row.get("human_current_source_span") or "").strip(),
        )
    return AxisPairClassification.model_validate(payload)


def _exact_match(human: AxisPairClassification, machine: AxisPairClassification) -> bool:
    return human.status == machine.status and (
        human.status != AxisClassificationStatus.COMPLETE
        or (
            human.previous_state == machine.previous_state
            and human.current_state == machine.current_state
        )
    )


def create_v1r_parser_freeze(
    *,
    dev_evaluation_manifest: Path,
    locked_set_preparation_manifest: Path,
    locked_packet_input: Path,
    source_strata_input: Path,
    human_gold: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"V1R parser freeze already exists: {output}")
    for path in (
        dev_evaluation_manifest,
        locked_set_preparation_manifest,
        locked_packet_input,
        source_strata_input,
        human_gold,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    dev = json.loads(dev_evaluation_manifest.read_text(encoding="utf-8"))
    if dev.get("status") not in {
        "DEV_PASSED_PARSER_READY_TO_FREEZE",
        "DEV_PASSED_PARSER_FROZEN",
    }:
        raise ValueError("V1R freeze requires a passing outcome-blind DEV evaluation")
    if (
        dev.get("outcome_vault_opened", False)
        or dev.get("return_data_opened", False)
        or dev.get("value_data_opened", False)
    ):
        raise ValueError("V1R DEV evaluation is contaminated by downstream data")
    prepared = json.loads(locked_set_preparation_manifest.read_text(encoding="utf-8"))
    if prepared.get("status") != "V1R_SOURCE_STRATIFIED_LOCKED_SET_PREPARED_OUTCOME_BLIND":
        raise ValueError("V1R source-stratified LOCKED preparation has not passed")
    if prepared.get("v1_locked_rows_reused", True) or prepared.get("dev_rows_reused", True):
        raise ValueError("V1R cannot reuse V1 LOCKED or DEV rows")
    expected = {
        "locked_packet_sha256": sha256_file(locked_packet_input),
        "source_strata_sha256": sha256_file(source_strata_input),
    }
    for key, value in expected.items():
        if prepared.get(key) != value:
            raise ValueError(f"V1R prepared artifact hash mismatch: {key}")
    payload = {
        "schema_version": "moatrader-historical-evidence-parser-freeze-v1r/1",
        "status": "V1R_PARSER_FROZEN_AWAITING_SINGLE_USE_LOCKED_TEST",
        "contract_tag": "future-eri-v1r-three-section-preoutcome",
        "frozen_at": datetime.now(SEOUL).isoformat(),
        "parser_version": dev["parser_version"],
        "prompt_sha256": dev["prompt_sha256"],
        "requested_model": dev["requested_model"],
        "dev_evaluation_manifest_sha256": sha256_file(dev_evaluation_manifest),
        "locked_set_preparation_manifest_sha256": sha256_file(
            locked_set_preparation_manifest
        ),
        "locked_packet_sha256": sha256_file(locked_packet_input),
        "source_strata_sha256": sha256_file(source_strata_input),
        "human_gold_sha256": sha256_file(human_gold),
        "minimum_per_axis_source_stratum": prepared[
            "minimum_per_axis_source_stratum"
        ],
        "source_strata": prepared["source_strata"],
        "v1_locked_rows_reused": False,
        "dev_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output, payload)
    return payload


def evaluate_v1r_locked_parser(
    *,
    packet_input: Path,
    source_strata_input: Path,
    classification_build: Path,
    human_gold: Path,
    parser_freeze_manifest: Path,
    locked_consumption_record: Path,
    output: Path,
    minimum_overall_agreement: float = 0.80,
    minimum_axis_agreement: float = 0.70,
    minimum_source_stratum_agreement: float = 0.70,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if locked_consumption_record.exists():
        raise FileExistsError("V1R LOCKED test was already consumed")
    classification_path = classification_build / "classifications.jsonl"
    classification_status_path = classification_build / "stage-status.json"
    for path in (
        packet_input,
        source_strata_input,
        classification_path,
        classification_status_path,
        human_gold,
        parser_freeze_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    freeze = json.loads(parser_freeze_manifest.read_text(encoding="utf-8"))
    if freeze.get("status") != "V1R_PARSER_FROZEN_AWAITING_SINGLE_USE_LOCKED_TEST":
        raise ValueError("V1R parser is not frozen for its LOCKED test")
    if freeze.get("v1_locked_rows_reused", True) or freeze.get("dev_rows_reused", True):
        raise ValueError("V1R freeze reports reused V1 or DEV rows")
    expected_hashes = {
        "locked_packet_sha256": sha256_file(packet_input),
        "source_strata_sha256": sha256_file(source_strata_input),
        "human_gold_sha256": sha256_file(human_gold),
    }
    for key, value in expected_hashes.items():
        if freeze.get(key) != value:
            raise ValueError(f"V1R frozen input changed: {key}")
    classification_status = json.loads(
        classification_status_path.read_text(encoding="utf-8")
    )
    if classification_status.get("status") != "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE":
        raise ValueError("V1R LOCKED classifications are incomplete")
    if (
        classification_status.get("outcome_vault_opened", False)
        or classification_status.get("return_data_opened", False)
        or classification_status.get("value_data_opened", False)
    ):
        raise ValueError("V1R LOCKED classifications are contaminated by downstream data")
    if classification_status.get("input_blinded_packet_sha256") != sha256_file(packet_input):
        raise ValueError("V1R classification packet hash mismatch")
    for key in ("parser_version", "prompt_sha256", "requested_model"):
        if classification_status.get(key) != freeze.get(key):
            raise ValueError(f"V1R LOCKED classification does not match frozen {key}")

    _write_json(
        locked_consumption_record,
        {
            "schema_version": "moatrader-locked-parser-test-consumption-v1r/1",
            "status": "STARTED_SINGLE_USE",
            "started_at": datetime.now(SEOUL).isoformat(),
            "parser_freeze_sha256": sha256_file(parser_freeze_manifest),
            "locked_packet_sha256": sha256_file(packet_input),
            "classification_sha256": sha256_file(classification_path),
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
        },
    )
    packet_rows = _read_jsonl(packet_input, PairedAxisPacket)
    machine_rows = _read_jsonl(classification_path, AxisPairClassification)
    packets = {row.packet_id: row for row in packet_rows}
    machine = {row.packet_id: row for row in machine_rows}
    if len(packets) != len(packet_rows) or len(machine) != len(machine_rows):
        raise ValueError("V1R packet and classification IDs must be unique")
    if set(packets) != set(machine):
        raise ValueError("V1R classifications must exactly cover LOCKED packets")
    strata_rows = [
        json.loads(line)
        for line in source_strata_input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    strata = {
        str(row["packet_id"]): V1RSourceStratum(str(row["source_stratum"]))
        for row in strata_rows
    }
    source_origins = {
        str(row["packet_id"]): {
            str(source_id): str(origin)
            for source_id, origin in dict(row.get("source_origins") or {}).items()
        }
        for row in strata_rows
    }
    if len(strata) != len(strata_rows) or set(strata) != set(packets):
        raise ValueError("V1R source strata must exactly cover LOCKED packets")
    expected_origin = {
        V1RSourceStratum.BUSINESS_INFO: "ARCANA_BUSINESS_HTML",
        V1RSourceStratum.FINANCE_COMMENT: "ARCANA_FINANCE_COMMENT_HTML",
        V1RSourceStratum.FINANCE_STATEMENT: "ARCANA_FINANCE_STATEMENT_HTML",
    }
    arcana_origins = set(expected_origin.values())
    for packet_id, packet in packets.items():
        packet_source_ids = {
            excerpt.source_id
            for excerpt in (*packet.previous_excerpts, *packet.current_excerpts)
        }
        if set(source_origins[packet_id]) != packet_source_ids:
            raise ValueError("V1R source-origin map must exactly cover packet excerpts")
        stratum = strata[packet_id]
        previous_origins = {
            source_origins[packet_id][excerpt.source_id]
            for excerpt in packet.previous_excerpts
        }
        current_origins = {
            source_origins[packet_id][excerpt.source_id]
            for excerpt in packet.current_excerpts
        }
        if stratum in expected_origin:
            required = expected_origin[stratum]
            if previous_origins != {required} or current_origins != {required}:
                raise ValueError("V1R single-source stratum contains another source")
        elif (
            "MOATRADER_OPENDART_ARCHIVE" not in previous_origins
            or "MOATRADER_OPENDART_ARCHIVE" not in current_origins
            or not (previous_origins & arcana_origins)
            or not (current_origins & arcana_origins)
        ):
            raise ValueError("V1R overlap stratum lacks Arcana or MoatRader evidence")

    common_report = evaluate_human_gold_quality(
        human_gold_path=human_gold,
        classifications=machine,
        packets=packets,
        minimum_gold_per_axis=int(freeze["minimum_per_axis_source_stratum"])
        * len(V1RSourceStratum),
        minimum_overall_agreement=minimum_overall_agreement,
        minimum_axis_agreement=minimum_axis_agreement,
        gold_split=GOLD_SPLIT,
    )
    reviewed: list[tuple[AxisPairClassification, AxisPairClassification, V1RSourceStratum]] = []
    invalid_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with human_gold.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"gold_split", "gold_contract_version", "source_stratum"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("V1R human gold lacks split, contract, or source stratum")
        for number, raw in enumerate(reader, start=2):
            if str(raw.get("gold_split") or "").strip() != GOLD_SPLIT:
                continue
            try:
                if str(raw.get("gold_contract_version") or "").strip() != GOLD_CONTRACT:
                    raise ValueError("row is not from the frozen V1R gold contract")
                packet_id = str(raw.get("packet_id") or "").strip()
                if packet_id in seen:
                    raise ValueError(f"duplicate V1R human-gold packet ID: {packet_id}")
                seen.add(packet_id)
                if V1RSourceStratum(str(raw.get("source_stratum") or "").strip()) != strata[
                    packet_id
                ]:
                    raise ValueError("human-gold source stratum changed")
                human = _human_classification(dict(raw))
                validate_classification_grounding(human, packets[packet_id])
                validate_classification_grounding(machine[packet_id], packets[packet_id])
                reviewed.append((human, machine[packet_id], strata[packet_id]))
            except (KeyError, TypeError, ValueError) as exc:
                invalid_rows.append({"row": number, "error": str(exc)})
    if seen != set(packets):
        invalid_rows.append({"row": 0, "error": "V1R gold must exactly cover LOCKED packets"})

    by_axis_source: dict[str, dict[str, Any]] = {}
    source_confusion: Counter[str] = Counter()
    source_gate = True
    minimum_count = int(freeze["minimum_per_axis_source_stratum"])
    for axis in OperatingEvidenceAxis:
        for stratum in V1RSourceStratum:
            values = [
                (human, predicted)
                for human, predicted, row_stratum in reviewed
                if human.axis == axis and row_stratum == stratum
            ]
            matches = sum(_exact_match(human, predicted) for human, predicted in values)
            agreement = matches / len(values) if values else 0.0
            count_passed = len(values) >= minimum_count
            agreement_passed = agreement >= minimum_source_stratum_agreement
            source_gate = source_gate and count_passed and agreement_passed
            key = f"{axis.value}|{stratum.value}"
            by_axis_source[key] = {
                "reviewed": len(values),
                "exact_matches": matches,
                "agreement": agreement,
                "machine_abstention_count": sum(
                    predicted.status != AxisClassificationStatus.COMPLETE
                    for _human, predicted in values
                ),
                "machine_abstention_rate": (
                    sum(
                        predicted.status != AxisClassificationStatus.COMPLETE
                        for _human, predicted in values
                    )
                    / len(values)
                    if values
                    else 0.0
                ),
                "false_stable_count": sum(
                    1
                    for human, predicted in values
                    if predicted.status == AxisClassificationStatus.COMPLETE
                    for human_state, predicted_state in zip(
                        (human.previous_state, human.current_state),
                        (predicted.previous_state, predicted.current_state),
                        strict=True,
                    )
                    if predicted_state is not None
                    and predicted_state.value == 0
                    and (human_state is None or human_state.value != 0)
                ),
                "neutral_to_bullish_count": sum(
                    1
                    for human, predicted in values
                    if human.status == AxisClassificationStatus.COMPLETE
                    and predicted.status == AxisClassificationStatus.COMPLETE
                    for human_state, predicted_state in zip(
                        (human.previous_state, human.current_state),
                        (predicted.previous_state, predicted.current_state),
                        strict=True,
                    )
                    if human_state is not None
                    and predicted_state is not None
                    and human_state.value == 0
                    and predicted_state.value == 1
                ),
                "state_confusion": dict(
                    sorted(
                        Counter(
                            f"{human_state.value}->{predicted_state.value}"
                            for human, predicted in values
                            if human.status == AxisClassificationStatus.COMPLETE
                            and predicted.status == AxisClassificationStatus.COMPLETE
                            for human_state, predicted_state in zip(
                                (human.previous_state, human.current_state),
                                (predicted.previous_state, predicted.current_state),
                                strict=True,
                            )
                            if human_state is not None and predicted_state is not None
                        ).items()
                    )
                ),
                "human_selected_source_origins": dict(
                    sorted(
                        Counter(
                            source_origins[human.packet_id][source_id]
                            for human, _predicted in values
                            if human.status == AxisClassificationStatus.COMPLETE
                            for source_id in (
                                human.previous_source_id,
                                human.current_source_id,
                            )
                            if source_id is not None
                        ).items()
                    )
                ),
                "machine_selected_source_origins": dict(
                    sorted(
                        Counter(
                            source_origins[predicted.packet_id][source_id]
                            for _human, predicted in values
                            if predicted.status == AxisClassificationStatus.COMPLETE
                            for source_id in (
                                predicted.previous_source_id,
                                predicted.current_source_id,
                            )
                            if source_id is not None
                        ).items()
                    )
                ),
                "source_span_grounding_validated_count": len(values),
                "source_span_grounding_rate": 1.0 if values else 0.0,
                "minimum_count_passed": count_passed,
                "agreement_passed": agreement_passed,
            }
            for human, predicted in values:
                source_confusion[
                    f"{stratum.value}|{human.status.value}->{predicted.status.value}"
                ] += 1
    gate = bool(common_report["gate_passed"]) and source_gate and not invalid_rows
    report = {
        "schema_version": "moatrader-historical-label-quality-v1r/1",
        "status": "PASSED" if gate else "FAILED_V1R_SOURCE_STRATIFIED_LOCKED_GATE",
        "gate_passed": gate,
        "common_v1_exact_quality_gate_passed": bool(common_report["gate_passed"]),
        "source_stratum_gate_passed": source_gate,
        "minimum_source_stratum_agreement": minimum_source_stratum_agreement,
        "minimum_per_axis_source_stratum": minimum_count,
        "reviewed_count": len(reviewed),
        "by_axis_source_stratum": by_axis_source,
        "source_status_confusion": dict(sorted(source_confusion.items())),
        "common_v1_quality_report": common_report,
        "invalid_rows": invalid_rows,
        "v1_locked_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "parser-quality-report-v1r.json"
    _write_json(report_path, report)
    stage = {
        "schema_version": "moatrader-historical-parser-evaluation-stage-v1r/1",
        "status": "V1R_LOCKED_TEST_PASSED" if gate else "V1R_EVIDENCE_PARSER_NOT_VALIDATED",
        "gate_passed": gate,
        "source_stratum_gate_passed": source_gate,
        "parser_version": classification_status["parser_version"],
        "prompt_sha256": classification_status["prompt_sha256"],
        "requested_model": classification_status["requested_model"],
        "parser_freeze_sha256": sha256_file(parser_freeze_manifest),
        "quality_report_sha256": sha256_file(report_path),
        "v1_locked_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "stage-status.json", stage)
    consumption = json.loads(locked_consumption_record.read_text(encoding="utf-8"))
    consumption.update(
        status="COMPLETED_SINGLE_USE",
        completed_at=datetime.now(SEOUL).isoformat(),
        gate_passed=gate,
        quality_report_sha256=sha256_file(report_path),
    )
    _write_json(locked_consumption_record, consumption)
    return stage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze or consume the source-stratified V1R LOCKED parser test."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--dev-evaluation-manifest", type=Path, required=True)
    freeze.add_argument("--locked-set-preparation-manifest", type=Path, required=True)
    freeze.add_argument("--locked-packet-input", type=Path, required=True)
    freeze.add_argument("--source-strata-input", type=Path, required=True)
    freeze.add_argument("--human-gold", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--packet-input", type=Path, required=True)
    evaluate.add_argument("--source-strata-input", type=Path, required=True)
    evaluate.add_argument("--classification-build", type=Path, required=True)
    evaluate.add_argument("--human-gold", type=Path, required=True)
    evaluate.add_argument("--parser-freeze-manifest", type=Path, required=True)
    evaluate.add_argument("--locked-consumption-record", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--minimum-overall-agreement", type=float, default=0.80)
    evaluate.add_argument("--minimum-axis-agreement", type=float, default=0.70)
    evaluate.add_argument("--minimum-source-stratum-agreement", type=float, default=0.70)
    args = parser.parse_args()
    if args.command == "freeze":
        result = create_v1r_parser_freeze(
            dev_evaluation_manifest=args.dev_evaluation_manifest,
            locked_set_preparation_manifest=args.locked_set_preparation_manifest,
            locked_packet_input=args.locked_packet_input,
            source_strata_input=args.source_strata_input,
            human_gold=args.human_gold,
            output=args.output,
        )
    else:
        result = evaluate_v1r_locked_parser(
            packet_input=args.packet_input,
            source_strata_input=args.source_strata_input,
            classification_build=args.classification_build,
            human_gold=args.human_gold,
            parser_freeze_manifest=args.parser_freeze_manifest,
            locked_consumption_record=args.locked_consumption_record,
            output=args.output,
            minimum_overall_agreement=args.minimum_overall_agreement,
            minimum_axis_agreement=args.minimum_axis_agreement,
            minimum_source_stratum_agreement=args.minimum_source_stratum_agreement,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
