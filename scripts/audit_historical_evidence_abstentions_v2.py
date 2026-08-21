from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    AxisPairClassification,
    PairedAxisPacket,
    sha256_file,
)
from moatrader.expectations.historical_evidence_v2 import AbstentionReasonV2


AUDIT_REASON_CODES = (
    AbstentionReasonV2.TRUE_NO_MENTION.value,
    AbstentionReasonV2.ONE_PERIOD_ONLY.value,
    AbstentionReasonV2.RETRIEVAL_MISS.value,
    AbstentionReasonV2.TABLE_EXTRACTION_FAIL.value,
    AbstentionReasonV2.PERIOD_MISMATCH.value,
    AbstentionReasonV2.AMBIGUOUS_HUMAN_TOO.value,
    AbstentionReasonV2.NOT_APPLICABLE.value,
)
AUDIT_COLUMNS = (
    "packet_id",
    "axis",
    "machine_status",
    "previous_excerpts_json",
    "current_excerpts_json",
    "abstention_reason",
    "reviewer",
    "notes",
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


def _stable_key(packet_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{packet_id}".encode("utf-8")).hexdigest()


def _stratified_sample(
    rows: list[tuple[PairedAxisPacket, AxisPairClassification]],
    *,
    sample_size: int,
    seed: str,
) -> list[tuple[PairedAxisPacket, AxisPairClassification]]:
    groups: dict[tuple[str, str], list[tuple[PairedAxisPacket, AxisPairClassification]]] = (
        defaultdict(list)
    )
    for packet, classification in rows:
        groups[(packet.axis.value, classification.status.value)].append((packet, classification))
    for group in groups.values():
        group.sort(key=lambda item: _stable_key(item[0].packet_id, seed))
    selected: list[tuple[PairedAxisPacket, AxisPairClassification]] = []
    ordered_keys = sorted(groups)
    cursor = 0
    while len(selected) < sample_size:
        made_progress = False
        for key in ordered_keys:
            if cursor < len(groups[key]):
                selected.append(groups[key][cursor])
                made_progress = True
                if len(selected) == sample_size:
                    break
        if not made_progress:
            break
        cursor += 1
    if len(selected) != sample_size:
        raise ValueError(
            f"only {len(selected)} abstentions are available for requested audit size {sample_size}"
        )
    return sorted(selected, key=lambda item: item[0].packet_id)


def prepare_abstention_audit(
    *,
    packet_input: Path,
    classification_build: Path,
    output: Path,
    sample_size: int = 300,
    seed: str = "MOATRADER_V2_ABSTENTION_AUDIT_20260821",
) -> dict[str, Any]:
    if not 200 <= sample_size <= 500:
        raise ValueError("V2 abstention audit sample size must be in [200, 500]")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    classification_path = classification_build / "classifications.jsonl"
    status_path = classification_build / "stage-status.json"
    for path in (packet_input, classification_path, status_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    stage = json.loads(status_path.read_text(encoding="utf-8"))
    if stage.get("status") != "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE":
        raise ValueError("classification build is incomplete")
    if stage.get("input_blinded_packet_sha256") != sha256_file(packet_input):
        raise ValueError("classification input hash does not match the audit packet input")
    packets_list = _read_jsonl(packet_input, PairedAxisPacket)
    classifications_list = _read_jsonl(classification_path, AxisPairClassification)
    packets = {item.packet_id: item for item in packets_list}
    classifications = {item.packet_id: item for item in classifications_list}
    if len(packets) != len(packets_list) or len(classifications) != len(classifications_list):
        raise ValueError("audit packet and classification IDs must be unique")
    if set(packets) != set(classifications):
        raise ValueError("audit classifications must exactly cover the supplied packets")
    candidates = [
        (packets[packet_id], classification)
        for packet_id, classification in classifications.items()
        if classification.status != AxisClassificationStatus.COMPLETE
    ]
    selected = _stratified_sample(candidates, sample_size=sample_size, seed=seed)

    output.mkdir(parents=True, exist_ok=True)
    template_path = output / "abstention-audit-template.csv"
    with template_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for packet, classification in selected:
            writer.writerow(
                {
                    "packet_id": packet.packet_id,
                    "axis": packet.axis.value,
                    "machine_status": classification.status.value,
                    "previous_excerpts_json": json.dumps(
                        [item.model_dump(mode="json") for item in packet.previous_excerpts],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "current_excerpts_json": json.dumps(
                        [item.model_dump(mode="json") for item in packet.current_excerpts],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    "abstention_reason": "",
                    "reviewer": "",
                    "notes": "",
                }
            )
    strata = Counter(
        f"{packet.axis.value}|{classification.status.value}"
        for packet, classification in selected
    )
    manifest = {
        "schema_version": "moatrader-historical-abstention-audit-v2/1",
        "status": "V2_ABSTENTION_AUDIT_PREPARED",
        "sample_size": sample_size,
        "sampling_method": "DETERMINISTIC_ROUND_ROBIN_AXIS_BY_MACHINE_STATUS",
        "sampling_seed": seed,
        "sample_strata": dict(sorted(strata.items())),
        "allowed_reason_codes": list(AUDIT_REASON_CODES),
        "packet_input_sha256": sha256_file(packet_input),
        "classification_sha256": sha256_file(classification_path),
        "audit_template_sha256": sha256_file(template_path),
        "identifiers_masked": True,
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    _write_json(output / "stage-status.json", manifest)
    return manifest


def validate_abstention_audit(
    *,
    prepared_build: Path,
    completed_audit: Path,
    output: Path,
    maximum_upstream_failure_rate: float = 0.25,
) -> dict[str, Any]:
    if not 0 <= maximum_upstream_failure_rate <= 1:
        raise ValueError("maximum_upstream_failure_rate must be in [0, 1]")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    prepared_status_path = prepared_build / "stage-status.json"
    template_path = prepared_build / "abstention-audit-template.csv"
    for path in (prepared_status_path, template_path, completed_audit):
        if not path.is_file():
            raise FileNotFoundError(path)
    prepared = json.loads(prepared_status_path.read_text(encoding="utf-8"))
    if prepared.get("status") != "V2_ABSTENTION_AUDIT_PREPARED":
        raise ValueError("abstention audit was not prepared by the V2 blinded workflow")
    if prepared.get("audit_template_sha256") != sha256_file(template_path):
        raise ValueError("prepared abstention template changed after sampling")

    def rows_by_id(path: Path) -> dict[str, dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not set(AUDIT_COLUMNS).issubset(reader.fieldnames or []):
                raise ValueError("completed audit is missing required columns")
            rows: dict[str, dict[str, str]] = {}
            for row in reader:
                packet_id = str(row.get("packet_id") or "").strip()
                if packet_id in rows:
                    raise ValueError(f"audit contains a duplicate packet ID: {packet_id}")
                rows[packet_id] = dict(row)
        if "" in rows:
            raise ValueError("completed audit contains a blank packet ID")
        return rows

    template_rows = rows_by_id(template_path)
    completed_rows = rows_by_id(completed_audit)
    if len(template_rows) != prepared["sample_size"]:
        raise ValueError("prepared audit row count does not match its manifest")
    if set(completed_rows) != set(template_rows):
        raise ValueError("completed audit must contain exactly the prepared packet IDs")
    reasons: Counter[str] = Counter()
    reviewers: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    protected = (
        "axis",
        "machine_status",
        "previous_excerpts_json",
        "current_excerpts_json",
    )
    for packet_id, row in completed_rows.items():
        expected = template_rows[packet_id]
        if any(str(row.get(key) or "") != str(expected.get(key) or "") for key in protected):
            errors.append({"packet_id": packet_id, "error": "blinded evidence columns changed"})
        reason = str(row.get("abstention_reason") or "").strip()
        reviewer = str(row.get("reviewer") or "").strip()
        if reason not in AUDIT_REASON_CODES:
            errors.append({"packet_id": packet_id, "error": f"invalid reason code: {reason}"})
        else:
            reasons[reason] += 1
        if not reviewer:
            errors.append({"packet_id": packet_id, "error": "reviewer is required"})
        else:
            reviewers[reviewer] += 1
    upstream_failure_count = sum(
        reasons[code]
        for code in (
            AbstentionReasonV2.RETRIEVAL_MISS.value,
            AbstentionReasonV2.TABLE_EXTRACTION_FAIL.value,
            AbstentionReasonV2.PERIOD_MISMATCH.value,
        )
    )
    sparse_design_count = sum(
        reasons[code]
        for code in (
            AbstentionReasonV2.TRUE_NO_MENTION.value,
            AbstentionReasonV2.NOT_APPLICABLE.value,
        )
    )
    upstream_failure_rate = (
        upstream_failure_count / len(completed_rows) if completed_rows else 0.0
    )
    sparse_design_reason_rate = (
        sparse_design_count / len(completed_rows) if completed_rows else 0.0
    )
    structurally_valid = not errors and 200 <= len(completed_rows) <= 500
    upstream_gate_passed = upstream_failure_rate <= maximum_upstream_failure_rate
    passed = structurally_valid and upstream_gate_passed
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": "moatrader-historical-abstention-audit-report-v2/1",
        "status": (
            "PASSED"
            if passed
            else (
                "FAILED_UPSTREAM_EXTRACTION_GATE"
                if structurally_valid and not upstream_gate_passed
                else "FAILED_INVALID_OR_INCOMPLETE_AUDIT"
            )
        ),
        "gate_passed": passed,
        "reviewed_count": len(completed_rows),
        "reason_distribution": dict(sorted(reasons.items())),
        "maximum_upstream_failure_rate": maximum_upstream_failure_rate,
        "upstream_failure_count": upstream_failure_count,
        "upstream_failure_rate": upstream_failure_rate,
        "upstream_extraction_gate_passed": upstream_gate_passed,
        "sparse_design_reason_count": sparse_design_count,
        "sparse_design_reason_rate": sparse_design_reason_rate,
        "measurement_interpretation": (
            "UPSTREAM_EXTRACTION_REMEDIATION_REQUIRED"
            if not upstream_gate_passed
            else (
                "SPARSE_DESIGN_SUPPORTED"
                if sparse_design_reason_rate >= 0.5
                else "MIXED_ABSTENTION_STRUCTURE"
            )
        ),
        "reviewer_distribution": dict(sorted(reviewers.items())),
        "errors": errors,
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    report_path = output / "abstention-audit-report.json"
    _write_json(report_path, report)
    status = {
        "schema_version": "moatrader-historical-abstention-audit-stage-v2/1",
        "status": (
            "V2_ABSTENTION_AUDIT_PASSED"
            if passed
            else (
                "V2_ABSTENTION_AUDIT_FAILED_UPSTREAM_EXTRACTION"
                if structurally_valid and not upstream_gate_passed
                else "V2_ABSTENTION_AUDIT_FAILED_INVALID_OR_INCOMPLETE"
            )
        ),
        "gate_passed": passed,
        "upstream_extraction_gate_passed": upstream_gate_passed,
        "upstream_failure_rate": upstream_failure_rate,
        "reviewed_count": len(completed_rows),
        "prepared_manifest_sha256": sha256_file(prepared_status_path),
        "completed_audit_sha256": sha256_file(completed_audit),
        "audit_report_sha256": sha256_file(report_path),
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    _write_json(output / "stage-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare or validate a 200-500 row blinded V2 abstention reason-code audit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--packet-input", type=Path, required=True)
    prepare.add_argument("--classification-build", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--sample-size", type=int, default=300)
    prepare.add_argument("--seed", default="MOATRADER_V2_ABSTENTION_AUDIT_20260821")
    validate = subparsers.add_parser("validate")
    validate.add_argument("--prepared-build", type=Path, required=True)
    validate.add_argument("--completed-audit", type=Path, required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--maximum-upstream-failure-rate", type=float, default=0.25)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare_abstention_audit(
            packet_input=args.packet_input,
            classification_build=args.classification_build,
            output=args.output,
            sample_size=args.sample_size,
            seed=args.seed,
        )
    else:
        result = validate_abstention_audit(
            prepared_build=args.prepared_build,
            completed_audit=args.completed_audit,
            output=args.output,
            maximum_upstream_failure_rate=args.maximum_upstream_failure_rate,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
