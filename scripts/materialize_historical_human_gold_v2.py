from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    PairedAxisPacket,
    sha256_file,
)
from scripts.prepare_historical_locked_sets_v2 import GOLD_FIELDS


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_packets(path: Path) -> list[PairedAxisPacket]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            PairedAxisPacket.model_validate_json(line)
            for line in handle
            if line.strip()
        ]


def _grounded_span(text: str, anchor: str, *, maximum_length: int = 600) -> str:
    start = text.find(anchor)
    if start < 0:
        raise ValueError(f"review anchor is not present in the selected excerpt: {anchor!r}")
    if len(text) <= maximum_length:
        return text
    left = max(0, start - maximum_length // 3)
    right = min(len(text), left + maximum_length)
    left = max(0, right - maximum_length)
    return text[left:right]


def _source_for_anchor(
    excerpts: Iterable[Any], anchor: str
) -> tuple[str, str]:
    matches = [excerpt for excerpt in excerpts if anchor in excerpt.text]
    if not matches:
        raise ValueError(f"review anchor was not found in any blinded excerpt: {anchor!r}")
    selected = matches[0]
    return selected.source_id, _grounded_span(selected.text, anchor)


def materialize_human_gold(
    *, candidate_build: Path, review_decisions: Path, output: Path
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    natural_path = candidate_build / "natural-locked-packets.jsonl"
    balanced_path = candidate_build / "balanced-candidate-packets.jsonl"
    natural = _read_packets(natural_path)
    balanced = _read_packets(balanced_path)
    natural_ids = {packet.packet_id for packet in natural}
    packet_lookup = {packet.packet_id: packet for packet in (*natural, *balanced)}
    payload = json.loads(review_decisions.read_text(encoding="utf-8"))
    if payload.get("reviewer") != "HUMAN":
        raise ValueError("review decisions must be tagged exactly HUMAN")
    if payload.get("outcome_vault_opened") is not False:
        raise ValueError("review decisions must explicitly keep the outcome vault closed")
    if payload.get("return_data_opened") is not False:
        raise ValueError("review decisions must explicitly keep return data closed")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("review decisions must contain a decisions list")

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    status_counts: dict[str, int] = {}
    for number, decision in enumerate(decisions, start=1):
        packet_id = str(decision.get("packet_id") or "").strip()
        if packet_id in seen:
            raise ValueError(f"duplicate review decision at item {number}: {packet_id}")
        packet = packet_lookup.get(packet_id)
        if packet is None:
            raise ValueError(f"review decision is outside the candidate build: {packet_id}")
        seen.add(packet_id)
        status = AxisClassificationStatus(str(decision.get("status") or "").strip())
        status_counts[status.value] = status_counts.get(status.value, 0) + 1
        split = (
            "V2_NATURAL_LOCKED_TEST"
            if packet_id in natural_ids
            else "V2_BALANCED_CANDIDATE_REVIEW"
        )
        contract = (
            "V2_NATURAL_FREQUENCY_LOCKED"
            if packet_id in natural_ids
            else "V2_DIRECTIONAL_BALANCED_CANDIDATE_POOL"
        )
        row = {
            "packet_id": packet_id,
            "axis": packet.axis.value,
            "human_status": status.value,
            "human_previous_state": "",
            "human_current_state": "",
            "human_previous_source_id": "",
            "human_current_source_id": "",
            "human_previous_source_span": "",
            "human_current_source_span": "",
            "gold_split": split,
            "gold_contract_version": contract,
            "reviewer": "HUMAN",
            "review_notes": str(decision.get("review_notes") or "").strip(),
        }
        if status == AxisClassificationStatus.COMPLETE:
            previous_state = int(decision["previous_state"])
            current_state = int(decision["current_state"])
            if previous_state not in (-1, 0, 1) or current_state not in (-1, 0, 1):
                raise ValueError(f"invalid HUMAN state for {packet_id}")
            previous_source_id, previous_span = _source_for_anchor(
                packet.previous_excerpts, str(decision["previous_anchor"])
            )
            current_source_id, current_span = _source_for_anchor(
                packet.current_excerpts, str(decision["current_anchor"])
            )
            row.update(
                human_previous_state=str(previous_state),
                human_current_state=str(current_state),
                human_previous_source_id=previous_source_id,
                human_current_source_id=current_source_id,
                human_previous_source_span=previous_span,
                human_current_source_span=current_span,
            )
        rows.append(row)

    if not natural_ids.issubset(seen):
        missing = sorted(natural_ids - seen)
        raise ValueError(f"HUMAN decisions do not cover Natural LOCKED: {missing[:5]}")
    output.mkdir(parents=True, exist_ok=True)
    gold_path = output / "adjudicated-human-gold.csv"
    with gold_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GOLD_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": "moatrader-v2-human-gold-materialization/1",
        "status": "V2_HUMAN_REVIEW_DECISIONS_MATERIALIZED_OUTCOME_BLIND",
        "reviewer": "HUMAN",
        "review_decision_count": len(rows),
        "natural_decision_count": len(natural_ids),
        "balanced_candidate_decision_count": len(rows) - len(natural_ids),
        "status_counts": dict(sorted(status_counts.items())),
        "review_decisions_sha256": sha256_file(review_decisions),
        "natural_packets_sha256": sha256_file(natural_path),
        "balanced_candidates_sha256": sha256_file(balanced_path),
        "adjudicated_human_gold_sha256": sha256_file(gold_path),
        "source_spans_materialized_from_human_anchors": True,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "human-gold-materialization-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize auditable V2 HUMAN review decisions into grounded gold CSV."
    )
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_human_gold(
        candidate_build=args.candidate_build,
        review_decisions=args.review_decisions,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
