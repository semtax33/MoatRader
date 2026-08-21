from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    AxisPairClassification,
    PairedAxisPacket,
    sha256_file,
    validate_classification_grounding,
)


SEOUL = ZoneInfo("Asia/Seoul")
LOCKED_CONFIG = {
    "NATURAL": {
        "split": "V2_NATURAL_LOCKED_TEST",
        "contract": "V2_NATURAL_FREQUENCY_LOCKED",
        "packet_hash_key": "natural_locked_packet_sha256",
    },
    "BALANCED": {
        "split": "V2_BALANCED_LOCKED_TEST",
        "contract": "V2_DIRECTIONAL_BALANCED_LOCKED",
        "packet_hash_key": "balanced_locked_packet_sha256",
    },
}
V2_STRATA = (
    "COMPLETE_NEGATIVE",
    "COMPLETE_NEUTRAL",
    "COMPLETE_POSITIVE",
    "INSUFFICIENT_EVIDENCE",
    "AMBIGUOUS",
)
SEMANTIC_PARSER_AXES = (
    OperatingEvidenceAxis.DEMAND,
    OperatingEvidenceAxis.PRICE_MIX,
)


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


def _stratum(value: AxisPairClassification) -> str:
    if value.status == AxisClassificationStatus.INSUFFICIENT_EVIDENCE:
        return "INSUFFICIENT_EVIDENCE"
    if value.status == AxisClassificationStatus.AMBIGUOUS:
        return "AMBIGUOUS"
    assert value.delta is not None
    return {
        -1: "COMPLETE_NEGATIVE",
        0: "COMPLETE_NEUTRAL",
        1: "COMPLETE_POSITIVE",
    }[value.delta]


def _packet_ids(path: Path) -> set[str]:
    rows = _read_jsonl(path, PairedAxisPacket)
    result = {row.packet_id for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"LOCKED packet IDs must be unique: {path}")
    return result


def create_v2_parser_freeze(
    *,
    dev_evaluation_manifest: Path,
    locked_set_preparation_manifest: Path,
    natural_locked_packet_input: Path,
    balanced_locked_packet_input: Path,
    human_gold: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"V2 parser freeze already exists: {output}")
    for path in (
        dev_evaluation_manifest,
        locked_set_preparation_manifest,
        natural_locked_packet_input,
        balanced_locked_packet_input,
        human_gold,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    dev = json.loads(dev_evaluation_manifest.read_text(encoding="utf-8"))
    if dev.get("status") not in {
        "DEV_PASSED_PARSER_READY_TO_FREEZE",
        "DEV_PASSED_PARSER_FROZEN",
    }:
        raise ValueError("V2 freeze requires a passing outcome-blind DEV evaluation")
    if dev.get("outcome_vault_opened", False) or dev.get("return_data_opened", False):
        raise ValueError("DEV evaluation is contaminated by downstream data")
    preparation = json.loads(locked_set_preparation_manifest.read_text(encoding="utf-8"))
    if preparation.get("status") != "V2_DUAL_INDEPENDENT_LOCKED_SETS_PREPARED_OUTCOME_BLIND":
        raise ValueError("V2 freeze requires the independent LOCKED-set preparation gate")
    if preparation.get("v1_locked_rows_reused", True):
        raise ValueError("V1 LOCKED rows cannot be reused in V2")
    if not preparation.get("locked_sets_disjoint", False):
        raise ValueError("Natural and Balanced preparation sets are not disjoint")
    if preparation.get("outcome_vault_opened", False) or preparation.get(
        "return_data_opened", False
    ):
        raise ValueError("LOCKED-set preparation is contaminated by downstream data")
    expected_prepared_hashes = {
        "natural_locked_packet_sha256": sha256_file(natural_locked_packet_input),
        "balanced_locked_packet_sha256": sha256_file(balanced_locked_packet_input),
        "human_gold_sha256": sha256_file(human_gold),
    }
    for key, value in expected_prepared_hashes.items():
        if preparation.get(key) != value:
            raise ValueError(f"prepared V2 LOCKED artifact hash mismatch: {key}")
    natural_ids = _packet_ids(natural_locked_packet_input)
    balanced_ids = _packet_ids(balanced_locked_packet_input)
    overlap = natural_ids & balanced_ids
    if overlap:
        raise ValueError(
            f"Natural and Balanced LOCKED sets must be independent: {sorted(overlap)[:5]}"
        )
    payload = {
        "schema_version": "moatrader-historical-evidence-parser-freeze-v2/2",
        "status": "V2_PARSER_FROZEN_AWAITING_DUAL_INDEPENDENT_LOCKED_TESTS",
        "frozen_at": datetime.now(SEOUL).isoformat(),
        "parser_version": dev["parser_version"],
        "prompt_sha256": dev["prompt_sha256"],
        "requested_model": dev["requested_model"],
        "dev_evaluation_manifest_sha256": sha256_file(dev_evaluation_manifest),
        "locked_set_preparation_manifest_sha256": sha256_file(
            locked_set_preparation_manifest
        ),
        "natural_locked_packet_sha256": sha256_file(natural_locked_packet_input),
        "balanced_locked_packet_sha256": sha256_file(balanced_locked_packet_input),
        "human_gold_sha256": sha256_file(human_gold),
        "natural_locked_gold_contract": LOCKED_CONFIG["NATURAL"]["contract"],
        "balanced_locked_gold_contract": LOCKED_CONFIG["BALANCED"]["contract"],
        "locked_sets_disjoint": True,
        "v1_locked_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    _write_json(output, payload)
    return payload


def evaluate_v2_locked_parser(
    *,
    packet_input: Path,
    classification_build: Path,
    human_gold: Path,
    parser_freeze_manifest: Path,
    locked_consumption_record: Path,
    output: Path,
    locked_kind: Literal["NATURAL", "BALANCED"] = "BALANCED",
    minimum_per_axis_stratum: int = 5,
    minimum_natural_per_axis: int = 20,
    minimum_overall_directional_agreement: float = 0.80,
    minimum_axis_directional_agreement: float = 0.70,
    maximum_neutral_to_bullish_rate: float = 0.10,
    maximum_false_stable_rate: float = 0.05,
    maximum_opposite_direction_count: int = 0,
) -> dict[str, Any]:
    config = LOCKED_CONFIG[locked_kind]
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if locked_consumption_record.exists():
        raise FileExistsError(f"V2 {locked_kind} LOCKED test was already consumed")
    classification_path = classification_build / "classifications.jsonl"
    classification_status_path = classification_build / "stage-status.json"
    for path in (
        packet_input,
        classification_path,
        classification_status_path,
        human_gold,
        parser_freeze_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    freeze = json.loads(parser_freeze_manifest.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "moatrader-historical-evidence-parser-freeze-v2/2":
        raise ValueError("V1 or single-set LOCKED artifacts cannot validate V2 measurement")
    if freeze.get("status") != "V2_PARSER_FROZEN_AWAITING_DUAL_INDEPENDENT_LOCKED_TESTS":
        raise ValueError("V2 parser is not frozen for dual independent LOCKED tests")
    if freeze.get("v1_locked_rows_reused", True) or not freeze.get("locked_sets_disjoint", False):
        raise ValueError("V2 LOCKED sets must be new, mutually disjoint, and independent of V1")
    if freeze.get(str(config["packet_hash_key"])) != sha256_file(packet_input):
        raise ValueError(f"V2 {locked_kind} LOCKED packet input changed after parser freeze")
    if freeze.get("human_gold_sha256") != sha256_file(human_gold):
        raise ValueError("V2 human gold changed after parser freeze")
    classification_stage = json.loads(classification_status_path.read_text(encoding="utf-8"))
    if classification_stage.get("status") != "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE":
        raise ValueError(f"V2 {locked_kind} LOCKED classifications are incomplete")
    if classification_stage.get("input_blinded_packet_sha256") != sha256_file(packet_input):
        raise ValueError(f"V2 {locked_kind} LOCKED classification packet hash mismatch")
    for key in ("parser_version", "prompt_sha256", "requested_model"):
        if classification_stage.get(key) != freeze.get(key):
            raise ValueError(f"V2 {locked_kind} classification does not match frozen {key}")

    _write_json(
        locked_consumption_record,
        {
            "schema_version": "moatrader-locked-parser-test-consumption-v2/2",
            "status": "STARTED_SINGLE_USE",
            "locked_kind": locked_kind,
            "started_at": datetime.now(SEOUL).isoformat(),
            "parser_freeze_sha256": sha256_file(parser_freeze_manifest),
            "locked_packet_sha256": sha256_file(packet_input),
            "classification_sha256": sha256_file(classification_path),
            "v1_locked_rows_reused": False,
            "outcome_vault_opened": False,
            "return_data_opened": False,
        },
    )

    packet_rows = _read_jsonl(packet_input, PairedAxisPacket)
    classification_rows = _read_jsonl(classification_path, AxisPairClassification)
    packets = {row.packet_id: row for row in packet_rows}
    machine = {row.packet_id: row for row in classification_rows}
    if len(packets) != len(packet_rows) or len(machine) != len(classification_rows):
        raise ValueError("V2 LOCKED packet and classification IDs must be unique")
    if set(machine) != set(packets):
        raise ValueError("V2 LOCKED classifications must exactly match packet IDs")
    if any(packet.axis not in SEMANTIC_PARSER_AXES for packet in packets.values()):
        raise ValueError("V2 parser LOCKED sets may contain only frozen semantic-parser axes")

    reviewed: list[tuple[AxisPairClassification, AxisPairClassification]] = []
    invalid_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with human_gold.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"gold_split", "gold_contract_version"}.issubset(reader.fieldnames or []):
            raise ValueError("V2 human gold lacks split or measurement-contract columns")
        for number, raw in enumerate(reader, start=2):
            if str(raw.get("gold_split") or "").strip() != config["split"]:
                continue
            try:
                if str(raw.get("gold_contract_version") or "").strip() != config["contract"]:
                    raise ValueError(f"row is not from the V2 {locked_kind} gold contract")
                if str(raw.get("reviewer") or "").strip() != "HUMAN":
                    raise ValueError("V2 gold reviewer must be tagged exactly HUMAN")
                human = _human_classification(dict(raw))
                if human.packet_id in seen:
                    raise ValueError(f"duplicate human-gold packet ID: {human.packet_id}")
                seen.add(human.packet_id)
                packet = packets[human.packet_id]
                validate_classification_grounding(human, packet)
                validate_classification_grounding(machine[human.packet_id], packet)
                reviewed.append((human, machine[human.packet_id]))
            except (KeyError, TypeError, ValueError) as exc:
                invalid_rows.append({"row": number, "error": str(exc)})
    if seen != set(packets):
        invalid_rows.append(
            {"row": 0, "error": f"V2 {locked_kind} gold must exactly cover its packet input"}
        )

    strata_counts = {
        axis.value: {stratum: 0 for stratum in V2_STRATA}
        for axis in SEMANTIC_PARSER_AXES
    }
    by_axis: dict[str, dict[str, Any]] = {}
    confusion: Counter[str] = Counter()
    directional_matches = 0
    neutral_human_count = 0
    neutral_to_bullish_count = 0
    human_missing_count = 0
    false_stable_count = 0
    human_directional_count = 0
    opposite_direction_count = 0
    for human, predicted in reviewed:
        human_stratum = _stratum(human)
        predicted_stratum = _stratum(predicted)
        strata_counts[human.axis.value][human_stratum] += 1
        confusion[f"{human_stratum}->{predicted_stratum}"] += 1
        directional_matches += int(human_stratum == predicted_stratum)
        if human_stratum == "COMPLETE_NEUTRAL":
            neutral_human_count += 1
            neutral_to_bullish_count += int(predicted_stratum == "COMPLETE_POSITIVE")
        if human_stratum in {"INSUFFICIENT_EVIDENCE", "AMBIGUOUS"}:
            human_missing_count += 1
            false_stable_count += int(predicted_stratum == "COMPLETE_NEUTRAL")
        if human_stratum in {"COMPLETE_NEGATIVE", "COMPLETE_POSITIVE"}:
            human_directional_count += 1
            opposite_direction_count += int(
                (human_stratum, predicted_stratum)
                in {
                    ("COMPLETE_NEGATIVE", "COMPLETE_POSITIVE"),
                    ("COMPLETE_POSITIVE", "COMPLETE_NEGATIVE"),
                }
            )
    for axis in SEMANTIC_PARSER_AXES:
        values = [(human, pred) for human, pred in reviewed if human.axis == axis]
        matches = sum(_stratum(human) == _stratum(pred) for human, pred in values)
        agreement = matches / len(values) if values else 0.0
        by_axis[axis.value] = {
            "reviewed": len(values),
            "directional_matches": matches,
            "directional_agreement": agreement,
            "agreement_passed": agreement >= minimum_axis_directional_agreement,
            "stratum_counts": strata_counts[axis.value],
            "natural_frequency_count_passed": len(values) >= minimum_natural_per_axis,
            "balanced_strata_passed": all(
                count >= minimum_per_axis_stratum
                for count in strata_counts[axis.value].values()
            ),
        }
    overall = directional_matches / len(reviewed) if reviewed else 0.0
    neutral_to_bullish_rate = (
        neutral_to_bullish_count / neutral_human_count if neutral_human_count else 0.0
    )
    false_stable_rate = false_stable_count / human_missing_count if human_missing_count else 0.0
    opposite_direction_rate = (
        opposite_direction_count / human_directional_count if human_directional_count else 0.0
    )
    common_quality = (
        bool(reviewed)
        and not invalid_rows
        and overall >= minimum_overall_directional_agreement
        and neutral_to_bullish_rate <= maximum_neutral_to_bullish_rate
        and false_stable_rate <= maximum_false_stable_rate
        and opposite_direction_count <= maximum_opposite_direction_count
        and all(row["agreement_passed"] for row in by_axis.values())
    )
    natural_gate = common_quality and all(
        row["natural_frequency_count_passed"] for row in by_axis.values()
    )
    balanced_gate = common_quality and all(
        row["balanced_strata_passed"] for row in by_axis.values()
    )
    gate = natural_gate if locked_kind == "NATURAL" else balanced_gate
    report = {
        "schema_version": "moatrader-historical-label-quality-v2/2",
        "status": "PASSED" if gate else "FAILED_LOCKED_MEASUREMENT_QUALITY",
        "locked_kind": locked_kind,
        "gate_passed": gate,
        "natural_frequency_gate_passed": natural_gate if locked_kind == "NATURAL" else False,
        "directional_strata_gate_passed": balanced_gate if locked_kind == "BALANCED" else False,
        "reviewed_count": len(reviewed),
        "minimum_per_axis_stratum": minimum_per_axis_stratum,
        "minimum_natural_per_axis": minimum_natural_per_axis,
        "minimum_overall_directional_agreement": minimum_overall_directional_agreement,
        "minimum_axis_directional_agreement": minimum_axis_directional_agreement,
        "overall_directional_agreement": overall,
        "maximum_neutral_to_bullish_rate": maximum_neutral_to_bullish_rate,
        "neutral_human_count": neutral_human_count,
        "neutral_to_bullish_count": neutral_to_bullish_count,
        "neutral_to_bullish_rate": neutral_to_bullish_rate,
        "maximum_false_stable_rate": maximum_false_stable_rate,
        "human_missing_count": human_missing_count,
        "false_stable_count": false_stable_count,
        "false_stable_rate": false_stable_rate,
        "maximum_opposite_direction_count": maximum_opposite_direction_count,
        "human_directional_count": human_directional_count,
        "opposite_direction_count": opposite_direction_count,
        "opposite_direction_rate": opposite_direction_rate,
        "by_axis": by_axis,
        "directional_confusion": dict(sorted(confusion.items())),
        "invalid_rows": invalid_rows,
        "gold_split": config["split"],
        "gold_contract_version": config["contract"],
        "v1_locked_rows_reused": False,
        "semantic_parser_axes": [axis.value for axis in SEMANTIC_PARSER_AXES],
        "gold_label_authority": "HUMAN",
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "parser-quality-report-v2.json"
    _write_json(report_path, report)
    stage = {
        "schema_version": "moatrader-historical-parser-evaluation-stage-v2/2",
        "status": (
            f"V2_{locked_kind}_LOCKED_TEST_PASSED"
            if gate
            else f"V2_{locked_kind}_EVIDENCE_PARSER_NOT_VALIDATED"
        ),
        "locked_kind": locked_kind,
        "gate_passed": gate,
        "natural_frequency_gate_passed": natural_gate if locked_kind == "NATURAL" else False,
        "directional_strata_gate_passed": balanced_gate if locked_kind == "BALANCED" else False,
        "parser_version": classification_stage["parser_version"],
        "prompt_sha256": classification_stage["prompt_sha256"],
        "requested_model": classification_stage["requested_model"],
        "parser_quality_report_sha256": sha256_file(report_path),
        "parser_freeze_sha256": sha256_file(parser_freeze_manifest),
        "v1_locked_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
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


def combine_v2_locked_evaluations(
    *,
    natural_evaluation_manifest: Path,
    balanced_evaluation_manifest: Path,
    parser_freeze_manifest: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"combined LOCKED manifest output must be new: {output}")
    for path in (
        natural_evaluation_manifest,
        balanced_evaluation_manifest,
        parser_freeze_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    natural = json.loads(natural_evaluation_manifest.read_text(encoding="utf-8"))
    balanced = json.loads(balanced_evaluation_manifest.read_text(encoding="utf-8"))
    expected_freeze_hash = sha256_file(parser_freeze_manifest)
    if natural.get("status") != "V2_NATURAL_LOCKED_TEST_PASSED":
        raise ValueError("Natural-frequency V2 LOCKED test has not passed")
    if balanced.get("status") != "V2_BALANCED_LOCKED_TEST_PASSED":
        raise ValueError("Directional-balanced V2 LOCKED test has not passed")
    if not natural.get("natural_frequency_gate_passed", False):
        raise ValueError("Natural-frequency LOCKED gate flag is false")
    if not balanced.get("directional_strata_gate_passed", False):
        raise ValueError("Balanced directional-strata gate flag is false")
    for manifest in (natural, balanced):
        if manifest.get("parser_freeze_sha256") != expected_freeze_hash:
            raise ValueError("LOCKED evaluations do not share the supplied parser freeze")
        if manifest.get("outcome_vault_opened", False) or manifest.get("return_data_opened", False):
            raise ValueError("LOCKED evaluation is contaminated by downstream data")
    for key in ("parser_version", "prompt_sha256", "requested_model"):
        if natural.get(key) != balanced.get(key):
            raise ValueError(f"Natural and Balanced LOCKED disagree on {key}")
    payload = {
        "schema_version": "moatrader-historical-parser-dual-locked-stage-v2/1",
        "status": "V2_LOCKED_TESTS_PASSED",
        "natural_frequency_gate_passed": True,
        "directional_strata_gate_passed": True,
        "parser_version": natural["parser_version"],
        "prompt_sha256": natural["prompt_sha256"],
        "requested_model": natural["requested_model"],
        "parser_freeze_sha256": expected_freeze_hash,
        "natural_evaluation_manifest_sha256": sha256_file(natural_evaluation_manifest),
        "balanced_evaluation_manifest_sha256": sha256_file(balanced_evaluation_manifest),
        "v1_locked_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    _write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze, consume, or combine independent Natural and Balanced V2 LOCKED tests."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--dev-evaluation-manifest", type=Path, required=True)
    freeze_parser.add_argument("--locked-set-preparation-manifest", type=Path, required=True)
    freeze_parser.add_argument("--natural-locked-packet-input", type=Path, required=True)
    freeze_parser.add_argument("--balanced-locked-packet-input", type=Path, required=True)
    freeze_parser.add_argument("--human-gold", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--locked-kind", choices=["NATURAL", "BALANCED"], required=True)
    evaluate_parser.add_argument("--packet-input", type=Path, required=True)
    evaluate_parser.add_argument("--classification-build", type=Path, required=True)
    evaluate_parser.add_argument("--human-gold", type=Path, required=True)
    evaluate_parser.add_argument("--parser-freeze-manifest", type=Path, required=True)
    evaluate_parser.add_argument("--locked-consumption-record", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--minimum-per-axis-stratum", type=int, default=5)
    evaluate_parser.add_argument("--minimum-natural-per-axis", type=int, default=20)
    evaluate_parser.add_argument("--minimum-overall-directional-agreement", type=float, default=0.80)
    evaluate_parser.add_argument("--minimum-axis-directional-agreement", type=float, default=0.70)
    evaluate_parser.add_argument("--maximum-neutral-to-bullish-rate", type=float, default=0.10)
    evaluate_parser.add_argument("--maximum-false-stable-rate", type=float, default=0.05)
    evaluate_parser.add_argument("--maximum-opposite-direction-count", type=int, default=0)
    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument("--natural-evaluation-manifest", type=Path, required=True)
    combine_parser.add_argument("--balanced-evaluation-manifest", type=Path, required=True)
    combine_parser.add_argument("--parser-freeze-manifest", type=Path, required=True)
    combine_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        result = create_v2_parser_freeze(
            dev_evaluation_manifest=args.dev_evaluation_manifest,
            locked_set_preparation_manifest=args.locked_set_preparation_manifest,
            natural_locked_packet_input=args.natural_locked_packet_input,
            balanced_locked_packet_input=args.balanced_locked_packet_input,
            human_gold=args.human_gold,
            output=args.output,
        )
    elif args.command == "evaluate":
        result = evaluate_v2_locked_parser(
            packet_input=args.packet_input,
            classification_build=args.classification_build,
            human_gold=args.human_gold,
            parser_freeze_manifest=args.parser_freeze_manifest,
            locked_consumption_record=args.locked_consumption_record,
            output=args.output,
            locked_kind=args.locked_kind,
            minimum_per_axis_stratum=args.minimum_per_axis_stratum,
            minimum_natural_per_axis=args.minimum_natural_per_axis,
            minimum_overall_directional_agreement=args.minimum_overall_directional_agreement,
            minimum_axis_directional_agreement=args.minimum_axis_directional_agreement,
            maximum_neutral_to_bullish_rate=args.maximum_neutral_to_bullish_rate,
            maximum_false_stable_rate=args.maximum_false_stable_rate,
            maximum_opposite_direction_count=args.maximum_opposite_direction_count,
        )
    else:
        result = combine_v2_locked_evaluations(
            natural_evaluation_manifest=args.natural_evaluation_manifest,
            balanced_evaluation_manifest=args.balanced_evaluation_manifest,
            parser_freeze_manifest=args.parser_freeze_manifest,
            output=args.output,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
