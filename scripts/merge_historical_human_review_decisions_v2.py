from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Sequence

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    sha256_file,
)


PACKET_ID_RE = re.compile(r"^PKT_[0-9a-f]{24}$")
COMPLETED_IMPORT_STATUS = "HUMAN_REVIEW_DECISIONS_IMPORTED_OUTCOME_BLIND"
BALANCED_RETEST_REVIEW_TYPE = "balanced-retest-1"
COMBINED_BALANCED_RETEST_STATUS = (
    "V2_BALANCED_RETEST_1_COMBINED_HUMAN_REVIEW_DECISIONS_READY_OUTCOME_BLIND"
)
SEMANTIC_AXES = {
    OperatingEvidenceAxis.DEMAND.value,
    OperatingEvidenceAxis.PRICE_MIX.value,
}
DECISION_EVIDENCE_FIELDS = (
    "previous_state",
    "current_state",
    "previous_anchor",
    "current_anchor",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_decision(
    decision: object, *, input_path: Path, item_number: int
) -> dict[str, Any]:
    if not isinstance(decision, dict):
        raise ValueError(f"review decision must be an object: {input_path}:{item_number}")
    row = dict(decision)
    packet_id = str(row.get("packet_id") or "").strip()
    if not PACKET_ID_RE.fullmatch(packet_id):
        raise ValueError(f"invalid packet ID: {input_path}:{item_number}: {packet_id!r}")
    row["packet_id"] = packet_id
    axis = str(row.get("axis") or "").strip()
    if axis and axis not in SEMANTIC_AXES:
        raise ValueError(f"invalid semantic axis: {input_path}:{item_number}: {axis!r}")
    if axis:
        row["axis"] = axis
    try:
        status = AxisClassificationStatus(str(row.get("status") or "").strip())
    except ValueError as exc:
        raise ValueError(
            f"invalid HUMAN status: {input_path}:{item_number}: {row.get('status')!r}"
        ) from exc
    row["status"] = status.value
    if status == AxisClassificationStatus.COMPLETE:
        for field in ("previous_state", "current_state"):
            value = row.get(field)
            if isinstance(value, bool) or value not in (-1, 0, 1):
                raise ValueError(
                    f"COMPLETE state must be -1, 0, or 1: "
                    f"{input_path}:{item_number}:{field}"
                )
        for field in ("previous_anchor", "current_anchor"):
            value = str(row.get(field) or "").strip()
            if not value:
                raise ValueError(
                    f"COMPLETE decision requires a grounded anchor: "
                    f"{input_path}:{item_number}:{field}"
                )
            row[field] = value
    elif any(row.get(field) not in (None, "") for field in DECISION_EVIDENCE_FIELDS):
        raise ValueError(
            "INSUFFICIENT_EVIDENCE and AMBIGUOUS decisions must leave states and "
            f"anchors empty: {input_path}:{item_number}"
        )
    return row


def _read_packet_ids(candidate_build: Path) -> set[str]:
    packet_path = candidate_build / "balanced-retest-candidate-packets.jsonl"
    if not packet_path.is_file():
        raise FileNotFoundError(packet_path)
    packet_ids: list[str] = []
    with packet_path.open("r", encoding="utf-8") as handle:
        for item_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(
                    f"candidate packet must be an object: {packet_path}:{item_number}"
                )
            packet_id = str(row.get("packet_id") or "").strip()
            if not PACKET_ID_RE.fullmatch(packet_id):
                raise ValueError(
                    f"invalid candidate packet ID: {packet_path}:{item_number}: "
                    f"{packet_id!r}"
                )
            packet_ids.append(packet_id)
    if len(packet_ids) != len(set(packet_ids)):
        raise ValueError(f"candidate packet IDs must be unique: {packet_path}")
    return set(packet_ids)


def _validate_candidate_build(candidate_build: Path) -> tuple[set[str], Path]:
    manifest_path = candidate_build / "balanced-retest-preparation-manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"candidate manifest must contain an object: {manifest_path}")
    if manifest.get("status") != (
        "V2_BALANCED_RETEST_1_CANDIDATES_PREPARED_OUTCOME_BLIND"
    ):
        raise ValueError(f"Balanced retest candidate build is not ready: {manifest_path}")
    for key in ("outcome_vault_opened", "return_data_opened", "value_data_opened"):
        if manifest.get(key) is not False:
            raise ValueError(f"candidate build must keep {key} closed: {manifest_path}")
    if manifest.get("per_pbr_role") != "NOT_USED":
        raise ValueError(f"candidate build used PER/PBR before Full Index: {manifest_path}")
    packet_ids = _read_packet_ids(candidate_build)
    if manifest.get("candidate_packet_count") != len(packet_ids):
        raise ValueError(f"candidate manifest count does not match packets: {manifest_path}")
    packet_path = candidate_build / "balanced-retest-candidate-packets.jsonl"
    if manifest.get("balanced_retest_candidate_packet_sha256") != sha256_file(packet_path):
        raise ValueError(f"candidate packet hash changed: {packet_path}")
    return packet_ids, manifest_path


def _validate_completed_import(
    *,
    payload: dict[str, Any],
    input_path: Path,
    candidate_build: Path,
    validated_decisions: Sequence[dict[str, Any]],
) -> tuple[str, str]:
    if payload.get("status") != COMPLETED_IMPORT_STATUS:
        raise ValueError(f"HUMAN review import is not complete: {input_path}")
    if payload.get("review_type") != BALANCED_RETEST_REVIEW_TYPE:
        raise ValueError(f"unexpected Balanced retest review type: {input_path}")
    reviewer_name = str(payload.get("human_reviewer_name") or "").strip()
    if not reviewer_name:
        raise ValueError(f"actual HUMAN reviewer name is missing: {input_path}")
    if payload.get("attestation") != "YES":
        raise ValueError(f"HUMAN attestation must be exactly YES: {input_path}")
    review_date = str(payload.get("review_date") or "").strip()
    if not review_date:
        raise ValueError(f"HUMAN review date is missing: {input_path}")
    for key in ("outcome_vault_opened", "return_data_opened", "value_data_opened"):
        if payload.get(key) is not False:
            raise ValueError(f"HUMAN review import must keep {key} closed: {input_path}")
    if payload.get("per_pbr_role") != "NOT_USED":
        raise ValueError(f"HUMAN review import used PER/PBR before Full Index: {input_path}")
    for key in (
        "candidate_excerpts_verified",
        "decision_export_formulas_verified",
        "workbook_read_only_verified",
    ):
        if payload.get(key) is not True:
            raise ValueError(f"completed HUMAN review import is missing {key}: {input_path}")
    if payload.get("pending_count") != 0 or payload.get("row_error_count") != 0:
        raise ValueError(f"HUMAN review import still has pending rows or errors: {input_path}")

    expected_ids, manifest_path = _validate_candidate_build(candidate_build)
    actual_ids = {row["packet_id"] for row in validated_decisions}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "HUMAN review import does not exactly cover its candidate build; "
            f"missing={missing[:5]} extra={extra[:5]}: {input_path}"
        )
    expected_count = len(expected_ids)
    for key in ("candidate_count", "decision_count", "reviewed_count"):
        if payload.get(key) != expected_count:
            raise ValueError(f"HUMAN review import has an invalid {key}: {input_path}")
    if payload.get("candidate_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError(f"HUMAN review import candidate manifest changed: {input_path}")
    return reviewer_name, review_date


def merge_human_review_decisions(
    *,
    inputs: Sequence[Path],
    output: Path,
    input_candidate_builds: Sequence[Path] | None = None,
    combined_candidate_build: Path | None = None,
) -> dict[str, Any]:
    """Merge independent HUMAN review batches without altering either source file."""

    if output.exists():
        raise FileExistsError(f"output file already exists: {output}")
    if len(inputs) < 2:
        raise ValueError("at least two HUMAN decision files are required")
    strict_balanced_retest = (
        input_candidate_builds is not None or combined_candidate_build is not None
    )
    if strict_balanced_retest:
        if input_candidate_builds is None or combined_candidate_build is None:
            raise ValueError(
                "strict Balanced retest merge requires both input candidate builds "
                "and the combined candidate build"
            )
        if len(input_candidate_builds) != len(inputs):
            raise ValueError("each HUMAN decision file requires its own candidate build")
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_files: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    reviewer_names: list[str] = []
    review_dates: list[str] = []
    for input_index, input_path in enumerate(inputs):
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"review decision file must contain an object: {input_path}")
        if payload.get("reviewer") != "HUMAN":
            raise ValueError(f"reviewer must be tagged exactly HUMAN: {input_path}")
        if payload.get("outcome_vault_opened") is not False:
            raise ValueError(f"outcome vault must remain closed: {input_path}")
        if payload.get("return_data_opened") is not False:
            raise ValueError(f"return data must remain closed: {input_path}")
        rows = payload.get("decisions")
        if not isinstance(rows, list):
            raise ValueError(f"review decisions must contain a decisions list: {input_path}")
        validated = [
            _validate_decision(row, input_path=input_path, item_number=number)
            for number, row in enumerate(rows, start=1)
        ]
        candidate_build = None
        if strict_balanced_retest:
            assert input_candidate_builds is not None
            candidate_build = input_candidate_builds[input_index]
            reviewer_name, review_date = _validate_completed_import(
                payload=payload,
                input_path=input_path,
                candidate_build=candidate_build,
                validated_decisions=validated,
            )
            reviewer_names.append(reviewer_name)
            review_dates.append(review_date)
        for row in validated:
            packet_id = row["packet_id"]
            if packet_id in seen:
                raise ValueError(f"duplicate packet ID across HUMAN batches: {packet_id}")
            seen.add(packet_id)
            decisions.append(row)
            status = row["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        source_files.append(
            {
                "path": str(input_path.resolve()),
                "sha256": sha256_file(input_path),
                "decision_count": len(validated),
                **(
                    {
                        "candidate_build": str(candidate_build.resolve()),
                        "candidate_manifest_sha256": sha256_file(
                            candidate_build
                            / "balanced-retest-preparation-manifest.json"
                        ),
                    }
                    if candidate_build is not None
                    else {}
                ),
            }
        )
    decisions.sort(key=lambda row: row["packet_id"])
    combined_manifest_path: Path | None = None
    if strict_balanced_retest:
        assert combined_candidate_build is not None
        combined_ids, combined_manifest_path = _validate_candidate_build(
            combined_candidate_build
        )
        decision_ids = {row["packet_id"] for row in decisions}
        if decision_ids != combined_ids:
            missing = sorted(combined_ids - decision_ids)
            extra = sorted(decision_ids - combined_ids)
            raise ValueError(
                "merged HUMAN decisions do not exactly cover combined candidates; "
                f"missing={missing[:5]} extra={extra[:5]}"
            )
        if len(set(reviewer_names)) != 1:
            raise ValueError(
                "actual HUMAN reviewer names differ across batches; "
                "reviewer identity cannot be silently collapsed"
            )
    result = {
        "schema_version": "moatrader-v2-human-review-decisions-merged/1",
        **(
            {
                "status": COMBINED_BALANCED_RETEST_STATUS,
                "review_type": BALANCED_RETEST_REVIEW_TYPE,
                "human_reviewer_name": reviewer_names[0],
                "attestation": "YES",
                "review_date": max(review_dates),
                "batch_review_dates": review_dates,
                "candidate_count": len(decisions),
                "candidate_manifest_sha256": sha256_file(combined_manifest_path),
            }
            if combined_manifest_path is not None
            else {}
        ),
        "reviewer": "HUMAN",
        "decision_count": len(decisions),
        "status_counts": dict(sorted(status_counts.items())),
        "source_decision_files": source_files,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
        "decisions": decisions,
    }
    _write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge disjoint, outcome-blind V2 HUMAN review decision batches."
    )
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument(
        "--input-candidate-build", type=Path, action="append", default=None
    )
    parser.add_argument("--combined-candidate-build", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge_human_review_decisions(
        inputs=args.input,
        output=args.output,
        input_candidate_builds=args.input_candidate_build,
        combined_candidate_build=args.combined_candidate_build,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
