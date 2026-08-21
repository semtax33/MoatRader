from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterator

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    HistoricalFilingPair,
    PairedAxisPacket,
    packet_id,
    sha256_file,
)
from moatrader.expectations.historical_evidence_v2 import (
    AxisApplicabilityV2,
    SparseAxisAvailabilityV2,
)
from scripts.build_historical_sparse_features_v2 import (
    AxisApplicabilityDecisionInputV2,
    DeterministicAxisEvidenceInputV2,
)


SEMANTIC_PRIMARY_AXES = {
    OperatingEvidenceAxis.DEMAND,
    OperatingEvidenceAxis.PRICE_MIX,
}
DETERMINISTIC_PRIORITY_AXES = {
    OperatingEvidenceAxis.MARGIN,
    OperatingEvidenceAxis.INVENTORY_MISMATCH,
    OperatingEvidenceAxis.BACKLOG,
    OperatingEvidenceAxis.CAPACITY_CAPEX,
}


def _read_jsonl(path: Path, model: type):
    with path.open("r", encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def _packet_groups(path: Path) -> Iterator[list[PairedAxisPacket]]:
    group: list[PairedAxisPacket] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            group.append(PairedAxisPacket.model_validate_json(line))
            if len(group) == len(OperatingEvidenceAxis):
                if {packet.axis for packet in group} != set(OperatingEvidenceAxis):
                    raise ValueError("packet group must contain exactly six axes")
                yield group
                group = []
    if group:
        raise ValueError("packet input has a trailing incomplete pair group")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def prepare_semantic_packets(
    *,
    filing_pair_input: Path,
    packet_input: Path,
    deterministic_evidence_input: Path,
    applicability_input: Path,
    output: Path,
    include_qualitative_diagnostics: bool = False,
    expected_pair_count: int | None = None,
) -> dict[str, object]:
    for path in (
        filing_pair_input,
        packet_input,
        deterministic_evidence_input,
        applicability_input,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if output.exists():
        raise FileExistsError(f"output must be new: {output}")
    deterministic_rows = _read_jsonl(
        deterministic_evidence_input, DeterministicAxisEvidenceInputV2
    )
    applicability_rows = _read_jsonl(
        applicability_input, AxisApplicabilityDecisionInputV2
    )
    deterministic = {
        (row.pair_id, row.evidence.axis): row.evidence for row in deterministic_rows
    }
    applicability = {(row.pair_id, row.axis): row for row in applicability_rows}
    if len(deterministic) != len(deterministic_rows):
        raise ValueError("deterministic input must be unique by pair and axis")
    if len(applicability) != len(applicability_rows):
        raise ValueError("applicability input must be unique by pair and axis")

    selected_axis = Counter()
    decision_counts = Counter()
    selected_pairs: set[str] = set()
    used_deterministic: set[tuple[str, OperatingEvidenceAxis]] = set()
    used_applicability: set[tuple[str, OperatingEvidenceAxis]] = set()
    pair_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with filing_pair_input.open("r", encoding="utf-8") as pair_handle, output.open(
        "w", encoding="utf-8", newline="\n"
    ) as output_handle:
        pair_lines = (line for line in pair_handle if line.strip())
        for pair_line, packets in zip(pair_lines, _packet_groups(packet_input), strict=True):
            pair = HistoricalFilingPair.model_validate_json(pair_line)
            pair_count += 1
            by_axis = {packet.axis: packet for packet in packets}
            if {packet.packet_id for packet in packets} != {
                packet_id(pair.pair_id, axis) for axis in OperatingEvidenceAxis
            }:
                raise ValueError(f"packet group does not match pair: {pair.pair_id}")
            for axis in OperatingEvidenceAxis:
                key = (pair.pair_id, axis)
                decision = applicability.get(key)
                if decision is None:
                    raise ValueError(f"missing applicability decision: {pair.pair_id}/{axis.value}")
                used_applicability.add(key)
                deterministic_item = deterministic.get(key)
                if axis in DETERMINISTIC_PRIORITY_AXES and deterministic_item is None:
                    raise ValueError(f"missing deterministic priority axis: {pair.pair_id}/{axis.value}")
                if deterministic_item is not None:
                    used_deterministic.add(key)
                packet = by_axis[axis]
                if decision.applicability == AxisApplicabilityV2.NOT_APPLICABLE:
                    decision_counts["SKIP_NOT_APPLICABLE"] += 1
                    continue
                if deterministic_item is not None and deterministic_item.availability in {
                    SparseAxisAvailabilityV2.GROUNDED,
                    SparseAxisAvailabilityV2.NOT_APPLICABLE,
                }:
                    decision_counts["SKIP_HIGHER_PRIORITY_EVIDENCE"] += 1
                    continue
                both_periods = bool(packet.previous_excerpts and packet.current_excerpts)
                if not both_periods:
                    decision_counts["SKIP_NO_TWO_PERIOD_SEMANTIC_CANDIDATE"] += 1
                    continue
                selected = axis in SEMANTIC_PRIMARY_AXES or (
                    axis == OperatingEvidenceAxis.CAPACITY_CAPEX
                    and deterministic_item is not None
                    and deterministic_item.availability == SparseAxisAvailabilityV2.NA
                )
                if include_qualitative_diagnostics and axis in DETERMINISTIC_PRIORITY_AXES:
                    selected = selected or (
                        deterministic_item is not None
                        and deterministic_item.availability == SparseAxisAvailabilityV2.NA
                    )
                if not selected:
                    decision_counts["SKIP_NOT_SEMANTIC_PRIMARY"] += 1
                    continue
                output_handle.write(packet.model_dump_json() + "\n")
                selected_axis[axis.value] += 1
                selected_pairs.add(pair.pair_id)
                decision_counts["SELECT_SEMANTIC_REQUIRED"] += 1
    if expected_pair_count is not None and pair_count != expected_pair_count:
        raise ValueError(f"filing pair count mismatch: {pair_count} != {expected_pair_count}")
    if set(deterministic) != used_deterministic:
        raise ValueError("deterministic input contains rows outside the filing-pair universe")
    if set(applicability) != used_applicability:
        raise ValueError("applicability input contains rows outside the filing-pair universe")
    manifest = {
        "schema_version": "moatrader-semantic-packet-selection-v2/1",
        "status": "SEMANTIC_REQUIRED_PACKETS_PREPARED_OUTCOME_BLIND",
        "pair_count": pair_count,
        "selected_pair_count": len(selected_pairs),
        "selected_packet_count": sum(selected_axis.values()),
        "selected_axis_counts": {
            axis.value: selected_axis[axis.value] for axis in OperatingEvidenceAxis
        },
        "decision_counts": dict(sorted(decision_counts.items())),
        "semantic_primary_axes": sorted(axis.value for axis in SEMANTIC_PRIMARY_AXES),
        "capacity_narrative_fallback_enabled": True,
        "qualitative_diagnostics_for_numeric_axes": include_qualitative_diagnostics,
        "evidence_priority": [
            "DETERMINISTIC_NUMERIC",
            "STRUCTURED_TABLE",
            "LLM_NARRATIVE",
        ],
        "source_hashes": {
            "filing_pairs": sha256_file(filing_pair_input),
            "blinded_packets": sha256_file(packet_input),
            "deterministic_evidence": sha256_file(deterministic_evidence_input),
            "applicability": sha256_file(applicability_input),
        },
        "output_packet_sha256": sha256_file(output),
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    _write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Select only semantic packets still needed after deterministic PIT coverage: "
            "Demand, Price/Mix, and Capacity narrative fallback by default."
        )
    )
    parser.add_argument("--filing-pair-input", type=Path, required=True)
    parser.add_argument("--packet-input", type=Path, required=True)
    parser.add_argument("--deterministic-evidence-input", type=Path, required=True)
    parser.add_argument("--applicability-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-qualitative-diagnostics", action="store_true")
    parser.add_argument("--expected-pair-count", type=int)
    args = parser.parse_args()
    result = prepare_semantic_packets(
        filing_pair_input=args.filing_pair_input,
        packet_input=args.packet_input,
        deterministic_evidence_input=args.deterministic_evidence_input,
        applicability_input=args.applicability_input,
        output=args.output,
        include_qualitative_diagnostics=args.include_qualitative_diagnostics,
        expected_pair_count=args.expected_pair_count,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
