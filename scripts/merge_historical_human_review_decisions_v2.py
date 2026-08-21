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


def merge_human_review_decisions(
    *, inputs: Sequence[Path], output: Path
) -> dict[str, Any]:
    """Merge independent HUMAN review batches without altering either source file."""

    if output.exists():
        raise FileExistsError(f"output file already exists: {output}")
    if len(inputs) < 2:
        raise ValueError("at least two HUMAN decision files are required")
    decisions: list[dict[str, Any]] = []
    seen: set[str] = set()
    source_files: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    for input_path in inputs:
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
            }
        )
    decisions.sort(key=lambda row: row["packet_id"])
    result = {
        "schema_version": "moatrader-v2-human-review-decisions-merged/1",
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge_human_review_decisions(inputs=args.input, output=args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
