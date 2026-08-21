from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Iterator

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import PairedAxisPacket, sha256_file


GOLD_SPLITS = {"DEV", "LOCKED_TEST"}
ALL_AXES = {axis for axis in OperatingEvidenceAxis}


def _packet_lines(path: Path) -> Iterator[tuple[str, PairedAxisPacket]]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line.rstrip("\r\n"), PairedAxisPacket.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"invalid packet at line {number}: {exc}") from exc


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _gold_ids(path: Path, split: str) -> set[str]:
    if split not in GOLD_SPLITS:
        raise ValueError(f"unsupported gold split: {split}")
    selected: set[str] = set()
    axis_counts = {axis.value: 0 for axis in OperatingEvidenceAxis}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "gold_split" not in (reader.fieldnames or []):
            raise ValueError("human gold file is missing gold_split")
        for row in reader:
            if str(row.get("gold_split") or "").strip() != split:
                continue
            packet_id = str(row.get("packet_id") or "").strip()
            axis = str(row.get("axis") or "").strip()
            if not packet_id or packet_id in selected:
                raise ValueError(f"missing or duplicate gold packet id: {packet_id}")
            if axis not in axis_counts:
                raise ValueError(f"invalid gold axis: {axis}")
            selected.add(packet_id)
            axis_counts[axis] += 1
    expected = {axis.value: 20 for axis in OperatingEvidenceAxis}
    if axis_counts != expected:
        raise ValueError(f"gold split must contain 20 rows per axis: {axis_counts}")
    return selected


def _groups_of_six(
    packets: Iterable[tuple[str, PairedAxisPacket]],
) -> Iterator[list[tuple[str, PairedAxisPacket]]]:
    group: list[tuple[str, PairedAxisPacket]] = []
    for item in packets:
        group.append(item)
        if len(group) == len(OperatingEvidenceAxis):
            axes = {packet.axis for _, packet in group}
            if axes != ALL_AXES:
                raise ValueError(f"canonical packet order lost six-axis pair grouping: {axes}")
            yield group
            group = []
    if group:
        raise ValueError(f"trailing incomplete packet group: {len(group)}")


def prepare_subset(
    *,
    packet_input: Path,
    output: Path,
    mode: str,
    human_gold: Path | None = None,
    expected_candidate_pairs: int | None = None,
) -> dict[str, object]:
    if not packet_input.is_file():
        raise FileNotFoundError(packet_input)
    if output.exists():
        raise FileExistsError(f"output must be new: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_packets = 0
    selected_pairs = 0
    total_pairs = 0
    axis_counts = {axis.value: 0 for axis in OperatingEvidenceAxis}
    availability_histogram = {str(value): 0 for value in range(7)}
    co_observation = {
        left.value: {right.value: 0 for right in OperatingEvidenceAxis}
        for left in OperatingEvidenceAxis
    }

    with output.open("w", encoding="utf-8", newline="\n") as handle:
        if mode in GOLD_SPLITS:
            if human_gold is None:
                raise ValueError("human_gold is required for DEV/LOCKED_TEST")
            wanted = _gold_ids(human_gold, mode)
            found: set[str] = set()
            for line, packet in _packet_lines(packet_input):
                if packet.packet_id not in wanted:
                    continue
                handle.write(line + "\n")
                found.add(packet.packet_id)
                selected_packets += 1
                axis_counts[packet.axis.value] += 1
            missing = sorted(wanted - found)
            if missing:
                raise ValueError(f"gold packets missing from blinded input: {missing[:5]}")
        elif mode == "CANDIDATE_COMPLETE":
            for group in _groups_of_six(_packet_lines(packet_input)):
                total_pairs += 1
                if not all(packet.previous_excerpts and packet.current_excerpts for _, packet in group):
                    continue
                selected_pairs += 1
                for line, packet in group:
                    handle.write(line + "\n")
                    selected_packets += 1
                    axis_counts[packet.axis.value] += 1
            if expected_candidate_pairs is not None and selected_pairs != expected_candidate_pairs:
                raise ValueError(
                    f"candidate-complete pair mismatch: {selected_pairs} != {expected_candidate_pairs}"
                )
        elif mode == "AXIS_AVAILABLE":
            for group in _groups_of_six(_packet_lines(packet_input)):
                total_pairs += 1
                available = [
                    (line, packet)
                    for line, packet in group
                    if packet.previous_excerpts and packet.current_excerpts
                ]
                available_axes = {packet.axis for _, packet in available}
                availability_histogram[str(len(available_axes))] += 1
                for left in OperatingEvidenceAxis:
                    for right in OperatingEvidenceAxis:
                        co_observation[left.value][right.value] += int(
                            left in available_axes and right in available_axes
                        )
                if available:
                    selected_pairs += 1
                for line, packet in available:
                    handle.write(line + "\n")
                    selected_packets += 1
                    axis_counts[packet.axis.value] += 1
            if expected_candidate_pairs is not None and total_pairs != expected_candidate_pairs:
                raise ValueError(
                    f"all-pair packet count mismatch: {total_pairs} != {expected_candidate_pairs}"
                )
        else:
            raise ValueError(f"unsupported mode: {mode}")

    manifest = {
        "schema_version": "moatrader-historical-classification-subset-v1/1",
        "mode": mode,
        "source_packet_sha256": sha256_file(packet_input),
        "human_gold_sha256": sha256_file(human_gold) if human_gold is not None else None,
        "selected_packet_count": selected_packets,
        "selected_pair_count": (
            selected_pairs if mode in {"CANDIDATE_COMPLETE", "AXIS_AVAILABLE"} else None
        ),
        "total_pair_count": total_pairs if mode in {"CANDIDATE_COMPLETE", "AXIS_AVAILABLE"} else None,
        "axis_packet_counts": axis_counts,
        "candidate_grounded_axis_count_histogram": (
            availability_histogram if mode == "AXIS_AVAILABLE" else None
        ),
        "candidate_pairwise_co_observation_matrix": (
            co_observation if mode == "AXIS_AVAILABLE" else None
        ),
        "selection_policy": (
            "PAIR_AXIS_INDEPENDENT_BOTH_PERIODS_AVAILABLE_NO_SIX_AXIS_PREFILTER"
            if mode == "AXIS_AVAILABLE"
            else None
        ),
        "output_packet_sha256": sha256_file(output),
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    _write_json(output.with_suffix(output.suffix + ".manifest.json"), manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create outcome-blind DEV, LOCKED_TEST, V1 six-axis-complete, or V2 "
            "axis-independent packet inputs."
        )
    )
    parser.add_argument("--packet-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["DEV", "LOCKED_TEST", "CANDIDATE_COMPLETE", "AXIS_AVAILABLE"],
        required=True,
    )
    parser.add_argument("--human-gold", type=Path)
    parser.add_argument("--expected-candidate-pairs", type=int)
    args = parser.parse_args()
    result = prepare_subset(
        packet_input=args.packet_input,
        output=args.output,
        mode=args.mode,
        human_gold=args.human_gold,
        expected_candidate_pairs=args.expected_candidate_pairs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
