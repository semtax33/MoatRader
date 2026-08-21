from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    AxisPairClassification,
    PairedAxisPacket,
    sha256_file,
    validate_classification_grounding,
)


SEMANTIC_AXES = (
    OperatingEvidenceAxis.DEMAND,
    OperatingEvidenceAxis.PRICE_MIX,
    OperatingEvidenceAxis.CAPACITY_CAPEX,
)
STRATA = (
    "COMPLETE_NEGATIVE",
    "COMPLETE_NEUTRAL",
    "COMPLETE_POSITIVE",
    "INSUFFICIENT_EVIDENCE",
    "AMBIGUOUS",
)
GOLD_FIELDS = (
    "packet_id",
    "axis",
    "human_status",
    "human_previous_state",
    "human_current_state",
    "human_previous_source_id",
    "human_current_source_id",
    "human_previous_source_span",
    "human_current_source_span",
    "gold_split",
    "gold_contract_version",
    "reviewer",
    "review_notes",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[PairedAxisPacket]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _read_packets(path: Path) -> list[PairedAxisPacket]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [PairedAxisPacket.model_validate_json(line) for line in handle if line.strip()]
    if len({row.packet_id for row in rows}) != len(rows):
        raise ValueError(f"packet IDs must be unique: {path}")
    return rows


def _packet_ids(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return {
                str(row.get("packet_id") or "").strip()
                for row in csv.DictReader(handle)
                if str(row.get("packet_id") or "").strip()
            }
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            packet_id = str(payload.get("packet_id") or "").strip()
            if packet_id:
                result.add(packet_id)
    return result


def _selection_key(packet: PairedAxisPacket, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{packet.packet_id}".encode("utf-8")).hexdigest()


def _blank_gold_rows(
    packets: Sequence[PairedAxisPacket], *, split: str, contract: str
) -> list[dict[str, str]]:
    return [
        {
            "packet_id": packet.packet_id,
            "axis": packet.axis.value,
            "human_status": "",
            "human_previous_state": "",
            "human_current_state": "",
            "human_previous_source_id": "",
            "human_current_source_id": "",
            "human_previous_source_span": "",
            "human_current_source_span": "",
            "gold_split": split,
            "gold_contract_version": contract,
            "reviewer": "",
            "review_notes": "",
        }
        for packet in packets
    ]


def _write_gold(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GOLD_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_locked_candidates(
    *,
    packet_input: Path,
    prior_v1_inputs: Sequence[Path],
    dev_inputs: Sequence[Path],
    output: Path,
    natural_per_axis: int = 40,
    balanced_candidates_per_axis: int = 250,
    seed: str = "MOATRADER_V2_LOCKED_20260821",
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if not prior_v1_inputs:
        raise ValueError("explicit prior V1 inputs are required to prove no V1 row reuse")
    if not dev_inputs:
        raise ValueError("explicit DEV inputs are required to keep LOCKED rows independent")
    if natural_per_axis < 1 or balanced_candidates_per_axis < 5:
        raise ValueError("LOCKED candidate sample sizes are too small")
    packets = _read_packets(packet_input)
    if any(packet.axis not in SEMANTIC_AXES for packet in packets):
        raise ValueError("V2 LOCKED candidate input may contain only semantic-parser axes")
    prior_ids = set().union(*(_packet_ids(path) for path in prior_v1_inputs))
    dev_ids = set().union(*(_packet_ids(path) for path in dev_inputs))
    excluded = prior_ids | dev_ids
    eligible = [packet for packet in packets if packet.packet_id not in excluded]
    by_axis: dict[OperatingEvidenceAxis, list[PairedAxisPacket]] = defaultdict(list)
    for packet in eligible:
        by_axis[packet.axis].append(packet)

    natural: list[PairedAxisPacket] = []
    balanced_candidates: list[PairedAxisPacket] = []
    for axis in SEMANTIC_AXES:
        ordered = sorted(by_axis[axis], key=lambda row: _selection_key(row, seed))
        if len(ordered) < natural_per_axis + balanced_candidates_per_axis:
            raise ValueError(
                f"not enough independent {axis.value} packets for Natural and Balanced pools"
            )
        natural.extend(ordered[:natural_per_axis])
        balanced_candidates.extend(
            ordered[natural_per_axis : natural_per_axis + balanced_candidates_per_axis]
        )
    natural.sort(key=lambda row: (row.axis.value, row.packet_id))
    balanced_candidates.sort(key=lambda row: (row.axis.value, row.packet_id))
    natural_ids = {row.packet_id for row in natural}
    balanced_ids = {row.packet_id for row in balanced_candidates}
    if natural_ids & balanced_ids or (natural_ids | balanced_ids) & excluded:
        raise AssertionError("LOCKED selection independence invariant failed")

    natural_path = output / "natural-locked-packets.jsonl"
    balanced_path = output / "balanced-candidate-packets.jsonl"
    _write_jsonl(natural_path, natural)
    _write_jsonl(balanced_path, balanced_candidates)
    _write_gold(
        output / "natural-human-gold-template.csv",
        _blank_gold_rows(
            natural,
            split="V2_NATURAL_LOCKED_TEST",
            contract="V2_NATURAL_FREQUENCY_LOCKED",
        ),
    )
    _write_gold(
        output / "balanced-candidate-human-gold-template.csv",
        _blank_gold_rows(
            balanced_candidates,
            split="V2_BALANCED_CANDIDATE_REVIEW",
            contract="V2_DIRECTIONAL_BALANCED_CANDIDATE_POOL",
        ),
    )
    manifest = {
        "schema_version": "moatrader-v2-locked-candidate-preparation/1",
        "status": "V2_INDEPENDENT_LOCKED_CANDIDATES_PREPARED_OUTCOME_BLIND",
        "selection_seed": seed,
        "semantic_parser_axes": [axis.value for axis in SEMANTIC_AXES],
        "natural_per_axis": natural_per_axis,
        "balanced_candidates_per_axis": balanced_candidates_per_axis,
        "source_packet_sha256": sha256_file(packet_input),
        "prior_v1_input_sha256": [sha256_file(path) for path in prior_v1_inputs],
        "dev_input_sha256": [sha256_file(path) for path in dev_inputs],
        "prior_v1_packet_id_count": len(prior_ids),
        "dev_packet_id_count": len(dev_ids),
        "natural_locked_packet_sha256": sha256_file(natural_path),
        "balanced_candidate_packet_sha256": sha256_file(balanced_path),
        "locked_sets_disjoint": True,
        "v1_locked_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    _write_json(output / "candidate-preparation-manifest.json", manifest)
    return manifest


def _classification(row: dict[str, str]) -> AxisPairClassification:
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
    return {
        -1: "COMPLETE_NEGATIVE",
        0: "COMPLETE_NEUTRAL",
        1: "COMPLETE_POSITIVE",
    }[value.delta]


def finalize_locked_sets(
    *,
    candidate_build: Path,
    adjudicated_human_gold: Path,
    output: Path,
    minimum_per_axis_stratum: int = 5,
    seed: str = "MOATRADER_V2_BALANCED_20260821",
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if minimum_per_axis_stratum < 5:
        raise ValueError("Balanced LOCKED requires at least five cases per axis/stratum")
    candidate_manifest_path = candidate_build / "candidate-preparation-manifest.json"
    natural_path = candidate_build / "natural-locked-packets.jsonl"
    balanced_candidate_path = candidate_build / "balanced-candidate-packets.jsonl"
    for path in (candidate_manifest_path, natural_path, balanced_candidate_path, adjudicated_human_gold):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if candidate_manifest.get("status") != "V2_INDEPENDENT_LOCKED_CANDIDATES_PREPARED_OUTCOME_BLIND":
        raise ValueError("V2 LOCKED candidate preparation has not passed")
    if candidate_manifest.get("natural_locked_packet_sha256") != sha256_file(natural_path):
        raise ValueError("Natural LOCKED packets changed after candidate preparation")
    if candidate_manifest.get("balanced_candidate_packet_sha256") != sha256_file(
        balanced_candidate_path
    ):
        raise ValueError("Balanced candidate packets changed after candidate preparation")

    natural = _read_packets(natural_path)
    candidates = _read_packets(balanced_candidate_path)
    packet_lookup = {packet.packet_id: packet for packet in (*natural, *candidates)}
    human_rows: dict[str, dict[str, str]] = {}
    human_labels: dict[str, AxisPairClassification] = {}
    with adjudicated_human_gold.open("r", encoding="utf-8-sig", newline="") as handle:
        for number, raw in enumerate(csv.DictReader(handle), start=2):
            packet_id = str(raw.get("packet_id") or "").strip()
            if packet_id not in packet_lookup:
                continue
            if packet_id in human_rows:
                raise ValueError(f"duplicate human gold packet ID at row {number}: {packet_id}")
            row = dict(raw)
            label = _classification(row)
            validate_classification_grounding(label, packet_lookup[packet_id])
            human_rows[packet_id] = row
            human_labels[packet_id] = label
    natural_ids = {packet.packet_id for packet in natural}
    if not natural_ids.issubset(human_rows):
        raise ValueError("adjudicated human gold must cover every Natural LOCKED packet")

    buckets: dict[tuple[OperatingEvidenceAxis, str], list[PairedAxisPacket]] = defaultdict(list)
    for packet in candidates:
        label = human_labels.get(packet.packet_id)
        if label is not None:
            buckets[(packet.axis, _stratum(label))].append(packet)
    balanced: list[PairedAxisPacket] = []
    stratum_counts: dict[str, dict[str, int]] = {}
    for axis in SEMANTIC_AXES:
        stratum_counts[axis.value] = {}
        for stratum in STRATA:
            rows = sorted(
                buckets[(axis, stratum)], key=lambda row: _selection_key(row, seed)
            )
            if len(rows) < minimum_per_axis_stratum:
                raise ValueError(
                    f"insufficient human-adjudicated {axis.value}/{stratum} Balanced cases"
                )
            chosen = rows[:minimum_per_axis_stratum]
            balanced.extend(chosen)
            stratum_counts[axis.value][stratum] = len(chosen)
    balanced.sort(key=lambda row: (row.axis.value, row.packet_id))
    if natural_ids & {packet.packet_id for packet in balanced}:
        raise AssertionError("Natural and Balanced LOCKED sets overlap")

    output.mkdir(parents=True, exist_ok=True)
    output_natural = output / "natural-locked-packets.jsonl"
    output_balanced = output / "balanced-locked-packets.jsonl"
    _write_jsonl(output_natural, natural)
    _write_jsonl(output_balanced, balanced)
    gold_rows: list[dict[str, str]] = []
    for packet, split, contract in (
        *(
            (item, "V2_NATURAL_LOCKED_TEST", "V2_NATURAL_FREQUENCY_LOCKED")
            for item in natural
        ),
        *(
            (item, "V2_BALANCED_LOCKED_TEST", "V2_DIRECTIONAL_BALANCED_LOCKED")
            for item in balanced
        ),
    ):
        row = {field: str(human_rows[packet.packet_id].get(field) or "") for field in GOLD_FIELDS}
        row["packet_id"] = packet.packet_id
        row["axis"] = packet.axis.value
        row["gold_split"] = split
        row["gold_contract_version"] = contract
        gold_rows.append(row)
    gold_path = output / "v2-locked-human-gold.csv"
    _write_gold(gold_path, gold_rows)
    manifest = {
        "schema_version": "moatrader-v2-locked-set-preparation/1",
        "status": "V2_DUAL_INDEPENDENT_LOCKED_SETS_PREPARED_OUTCOME_BLIND",
        "candidate_preparation_manifest_sha256": sha256_file(candidate_manifest_path),
        "adjudicated_human_gold_sha256": sha256_file(adjudicated_human_gold),
        "natural_locked_packet_sha256": sha256_file(output_natural),
        "balanced_locked_packet_sha256": sha256_file(output_balanced),
        "human_gold_sha256": sha256_file(gold_path),
        "natural_packet_count": len(natural),
        "balanced_packet_count": len(balanced),
        "balanced_stratum_counts": stratum_counts,
        "minimum_per_axis_stratum": minimum_per_axis_stratum,
        "semantic_parser_axes": [axis.value for axis in SEMANTIC_AXES],
        "locked_sets_disjoint": True,
        "v1_locked_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    _write_json(output / "locked-set-preparation-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare new, independent Natural and directional-balanced V2 LOCKED sets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-candidates")
    prepare.add_argument("--packet-input", type=Path, required=True)
    prepare.add_argument("--prior-v1-input", type=Path, action="append", required=True)
    prepare.add_argument("--dev-input", type=Path, action="append", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--natural-per-axis", type=int, default=40)
    prepare.add_argument("--balanced-candidates-per-axis", type=int, default=250)
    prepare.add_argument("--seed", default="MOATRADER_V2_LOCKED_20260821")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--candidate-build", type=Path, required=True)
    finalize.add_argument("--adjudicated-human-gold", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--minimum-per-axis-stratum", type=int, default=5)
    finalize.add_argument("--seed", default="MOATRADER_V2_BALANCED_20260821")
    args = parser.parse_args()
    if args.command == "prepare-candidates":
        result = prepare_locked_candidates(
            packet_input=args.packet_input,
            prior_v1_inputs=args.prior_v1_input,
            dev_inputs=args.dev_input,
            output=args.output,
            natural_per_axis=args.natural_per_axis,
            balanced_candidates_per_axis=args.balanced_candidates_per_axis,
            seed=args.seed,
        )
    else:
        result = finalize_locked_sets(
            candidate_build=args.candidate_build,
            adjudicated_human_gold=args.adjudicated_human_gold,
            output=args.output,
            minimum_per_axis_stratum=args.minimum_per_axis_stratum,
            seed=args.seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
