from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from moatrader.expectations.historical_evidence import (
    AxisPairClassification,
    PairedAxisPacket,
    sha256_file,
)
from scripts.seal_historical_future_eri_features import evaluate_human_gold_quality


SEOUL = ZoneInfo("Asia/Seoul")


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def evaluate_parser(
    *,
    packet_input: Path,
    classification_build: Path,
    human_gold: Path,
    split: str,
    output: Path,
    minimum_gold_per_axis: int = 20,
    minimum_overall_agreement: float = 0.80,
    minimum_axis_agreement: float = 0.70,
    freeze_manifest_output: Path | None = None,
    locked_packet_input: Path | None = None,
    parser_freeze_manifest: Path | None = None,
    locked_consumption_record: Path | None = None,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    classification_path = classification_build / "classifications.jsonl"
    classification_status_path = classification_build / "stage-status.json"
    for required in (packet_input, classification_path, classification_status_path, human_gold):
        if not required.is_file():
            raise FileNotFoundError(required)
    classification_status = json.loads(classification_status_path.read_text(encoding="utf-8"))
    if classification_status.get("status") != "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE":
        raise ValueError("classification stage is not complete")
    packet_hash = sha256_file(packet_input)
    if classification_status.get("input_blinded_packet_sha256") != packet_hash:
        raise ValueError("classification stage packet hash does not match evaluation input")
    if classification_status.get("classification_sha256") != sha256_file(
        classification_path
    ):
        raise ValueError("classification file changed after its stage manifest")

    if split == "LOCKED_TEST":
        if parser_freeze_manifest is None or locked_consumption_record is None:
            raise ValueError(
                "LOCKED_TEST requires parser_freeze_manifest and locked_consumption_record"
            )
        if locked_consumption_record.exists():
            raise FileExistsError("locked test was already consumed for this parser freeze")
        freeze = json.loads(parser_freeze_manifest.read_text(encoding="utf-8"))
        for key in ("parser_version", "prompt_sha256", "requested_model"):
            if classification_status.get(key) != freeze.get(key):
                raise ValueError(f"locked classification does not match frozen {key}")
        if freeze.get("human_gold_sha256") != sha256_file(human_gold):
            raise ValueError("human gold changed after parser freeze")
        if freeze.get("locked_packet_sha256") != packet_hash:
            raise ValueError("locked packet input changed after parser freeze")
        _write_json(
            locked_consumption_record,
            {
                "schema_version": "moatrader-locked-parser-test-consumption-v1/1",
                "status": "STARTED_SINGLE_USE",
                "started_at": datetime.now(SEOUL).isoformat(),
                "parser_freeze_sha256": sha256_file(parser_freeze_manifest),
                "locked_packet_sha256": packet_hash,
                "classification_sha256": sha256_file(classification_path),
                "outcome_vault_opened": False,
                "return_data_opened": False,
            },
        )

    packets_list = _read_jsonl(packet_input, PairedAxisPacket)
    classifications_list = _read_jsonl(classification_path, AxisPairClassification)
    packets = {item.packet_id: item for item in packets_list}
    classifications = {item.packet_id: item for item in classifications_list}
    if len(packets) != len(packets_list) or len(classifications) != len(classifications_list):
        raise ValueError("packet and classification IDs must be unique")
    if set(packets) != set(classifications):
        raise ValueError("classification IDs must exactly match evaluation packet IDs")
    if classification_status.get("packet_count") != len(packets_list) or (
        classification_status.get("classification_count") != len(classifications_list)
    ):
        raise ValueError("classification stage counts do not match evaluation artifacts")

    report = evaluate_human_gold_quality(
        human_gold_path=human_gold,
        classifications=classifications,
        packets=packets,
        minimum_gold_per_axis=minimum_gold_per_axis,
        minimum_overall_agreement=minimum_overall_agreement,
        minimum_axis_agreement=minimum_axis_agreement,
        gold_split=split,
        required_axes={packet.axis for packet in packets_list},
    )
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "parser-quality-report.json"
    _write_json(report_path, report)
    if split == "DEV":
        stage = (
            "DEV_PASSED_PARSER_READY_TO_FREEZE"
            if report["gate_passed"]
            else "DEV_FAILED_PROMPT_REVISION_ALLOWED"
        )
        if report["gate_passed"] and freeze_manifest_output is not None:
            if locked_packet_input is None or not locked_packet_input.is_file():
                raise ValueError("locked_packet_input is required to freeze a passing DEV parser")
            if freeze_manifest_output.exists():
                raise FileExistsError(f"parser freeze already exists: {freeze_manifest_output}")
            _write_json(
                freeze_manifest_output,
                {
                    "schema_version": "moatrader-historical-evidence-parser-freeze-v1/1",
                    "frozen_at": datetime.now(SEOUL).isoformat(),
                    "parser_version": classification_status["parser_version"],
                    "prompt_sha256": classification_status["prompt_sha256"],
                    "requested_model": classification_status["requested_model"],
                    "dev_packet_sha256": packet_hash,
                    "locked_packet_sha256": sha256_file(locked_packet_input),
                    "dev_classification_sha256": sha256_file(classification_path),
                    "dev_quality_report_sha256": sha256_file(report_path),
                    "human_gold_sha256": sha256_file(human_gold),
                    "primary_gate": {
                        "minimum_gold_per_axis": minimum_gold_per_axis,
                        "minimum_overall_agreement": minimum_overall_agreement,
                        "minimum_axis_agreement": minimum_axis_agreement,
                    },
                    "outcome_vault_opened": False,
                    "return_data_opened": False,
                },
            )
            stage = "DEV_PASSED_PARSER_FROZEN"
    else:
        stage = (
            "LOCKED_TEST_PASSED"
            if report["gate_passed"]
            else "EVIDENCE_PARSER_NOT_VALIDATED"
        )
        assert locked_consumption_record is not None
        consumption = json.loads(locked_consumption_record.read_text(encoding="utf-8"))
        consumption.update(
            status="COMPLETED_SINGLE_USE",
            completed_at=datetime.now(SEOUL).isoformat(),
            gate_passed=bool(report["gate_passed"]),
            quality_report_sha256=sha256_file(report_path),
        )
        _write_json(locked_consumption_record, consumption)

    status = {
        "schema_version": "moatrader-historical-parser-evaluation-stage-v1/1",
        "status": stage,
        "split": split,
        "gate_passed": bool(report["gate_passed"]),
        "parser_profile": classification_status.get("parser_profile"),
        "parser_version": classification_status["parser_version"],
        "prompt_sha256": classification_status["prompt_sha256"],
        "requested_model": classification_status["requested_model"],
        "semantic_execution_scope": classification_status.get(
            "semantic_execution_scope"
        ),
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "stage-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate DEV or single-use LOCKED_TEST parser quality without opening private/outcome data."
    )
    parser.add_argument("--packet-input", type=Path, required=True)
    parser.add_argument("--classification-build", type=Path, required=True)
    parser.add_argument("--human-gold", type=Path, required=True)
    parser.add_argument("--split", choices=["DEV", "LOCKED_TEST"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-gold-per-axis", type=int, default=20)
    parser.add_argument("--minimum-overall-agreement", type=float, default=0.80)
    parser.add_argument("--minimum-axis-agreement", type=float, default=0.70)
    parser.add_argument("--freeze-manifest-output", type=Path)
    parser.add_argument("--locked-packet-input", type=Path)
    parser.add_argument("--parser-freeze-manifest", type=Path)
    parser.add_argument("--locked-consumption-record", type=Path)
    args = parser.parse_args()
    result = evaluate_parser(
        packet_input=args.packet_input,
        classification_build=args.classification_build,
        human_gold=args.human_gold,
        split=args.split,
        output=args.output,
        minimum_gold_per_axis=args.minimum_gold_per_axis,
        minimum_overall_agreement=args.minimum_overall_agreement,
        minimum_axis_agreement=args.minimum_axis_agreement,
        freeze_manifest_output=args.freeze_manifest_output,
        locked_packet_input=args.locked_packet_input,
        parser_freeze_manifest=args.parser_freeze_manifest,
        locked_consumption_record=args.locked_consumption_record,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
