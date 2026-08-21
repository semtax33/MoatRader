from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from moatrader.expectations.historical_evidence import (
    AxisPairClassification,
    PairedAxisPacket,
    sha256_file,
    validate_classification_grounding,
)
from scripts.prepare_historical_locked_sets_v2 import (
    GOLD_FIELDS,
    SEMANTIC_AXES,
    STRATA,
    _classification,
    _read_packets,
    _review_row_is_blank,
    _selection_key,
    _stratum,
    _write_gold,
    _write_json,
    _write_jsonl,
)


def _require_manifest(path: Path, status: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != status:
        raise ValueError(f"upstream manifest has not passed: {path}")
    for key in ("outcome_vault_opened", "return_data_opened", "value_data_opened"):
        if payload.get(key, False):
            raise ValueError(f"upstream manifest opened forbidden downstream data: {key}")
    return payload


def _round_robin_select(
    buckets: dict[str, list[PairedAxisPacket]],
    *,
    count: int,
    seed: str,
    already_selected: set[str] | None = None,
) -> list[PairedAxisPacket]:
    selected: list[PairedAxisPacket] = []
    used = set(already_selected or ())
    ordered = {
        stratum: sorted(rows, key=lambda row: _selection_key(row, seed))
        for stratum, rows in buckets.items()
    }
    positions = {stratum: 0 for stratum in ordered}
    while len(selected) < count:
        advanced = False
        for stratum in STRATA:
            rows = ordered.get(stratum, [])
            index = positions.get(stratum, 0)
            while index < len(rows) and rows[index].packet_id in used:
                index += 1
            positions[stratum] = index
            if index >= len(rows):
                continue
            packet = rows[index]
            positions[stratum] = index + 1
            used.add(packet.packet_id)
            selected.append(packet)
            advanced = True
            if len(selected) == count:
                break
        if not advanced:
            raise ValueError("insufficient disjoint HUMAN-reviewed packets for semantic DEV")
    return selected


def prepare_semantic_dev(
    *,
    candidate_build: Path,
    adjudicated_human_gold: Path,
    human_gold_materialization_manifest: Path,
    locked_set_build: Path,
    output: Path,
    per_axis: int = 30,
    minimum_complete_per_axis: int = 5,
    minimum_ambiguous_per_axis: int = 5,
    seed: str = "MOATRADER_V2_SEMANTIC_DEV_20260822",
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if per_axis < 20:
        raise ValueError("semantic DEV requires at least 20 HUMAN cases per axis")
    if minimum_complete_per_axis < 1 or minimum_ambiguous_per_axis < 1:
        raise ValueError("semantic DEV must include COMPLETE and AMBIGUOUS cases")
    if minimum_complete_per_axis + minimum_ambiguous_per_axis > per_axis:
        raise ValueError("semantic DEV minimum strata exceed per-axis size")

    candidate_manifest_path = candidate_build / "candidate-preparation-manifest.json"
    candidate_manifest = _require_manifest(
        candidate_manifest_path,
        "V2_INDEPENDENT_LOCKED_CANDIDATES_PREPARED_OUTCOME_BLIND",
    )
    materialization = _require_manifest(
        human_gold_materialization_manifest,
        "V2_HUMAN_REVIEW_DECISIONS_MATERIALIZED_OUTCOME_BLIND",
    )
    locked_manifest_path = locked_set_build / "locked-set-preparation-manifest.json"
    locked_manifest = _require_manifest(
        locked_manifest_path,
        "V2_DUAL_INDEPENDENT_LOCKED_SETS_PREPARED_OUTCOME_BLIND",
    )
    natural_candidate_path = candidate_build / "natural-locked-packets.jsonl"
    balanced_candidate_path = candidate_build / "balanced-candidate-packets.jsonl"
    natural_locked_path = locked_set_build / "natural-locked-packets.jsonl"
    balanced_locked_path = locked_set_build / "balanced-locked-packets.jsonl"
    for path in (
        adjudicated_human_gold,
        natural_candidate_path,
        balanced_candidate_path,
        natural_locked_path,
        balanced_locked_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if candidate_manifest.get("natural_locked_packet_sha256") != sha256_file(
        natural_candidate_path
    ):
        raise ValueError("candidate Natural packet hash mismatch")
    if candidate_manifest.get("balanced_candidate_packet_sha256") != sha256_file(
        balanced_candidate_path
    ):
        raise ValueError("candidate Balanced packet hash mismatch")
    if materialization.get("adjudicated_human_gold_sha256") != sha256_file(
        adjudicated_human_gold
    ):
        raise ValueError("materialized HUMAN gold hash mismatch")
    for key, path in (
        ("natural_locked_packet_sha256", natural_locked_path),
        ("balanced_locked_packet_sha256", balanced_locked_path),
    ):
        if locked_manifest.get(key) != sha256_file(path):
            raise ValueError(f"LOCKED packet hash mismatch: {key}")

    candidate_packets = _read_packets(natural_candidate_path) + _read_packets(
        balanced_candidate_path
    )
    packet_lookup = {packet.packet_id: packet for packet in candidate_packets}
    if len(packet_lookup) != len(candidate_packets):
        raise ValueError("candidate Natural and Balanced packet pools overlap")
    locked_packets = _read_packets(natural_locked_path) + _read_packets(
        balanced_locked_path
    )
    locked_ids = {packet.packet_id for packet in locked_packets}
    if len(locked_ids) != len(locked_packets):
        raise ValueError("Natural and Balanced LOCKED packet pools overlap")
    if not locked_ids.issubset(packet_lookup):
        raise ValueError("LOCKED packets are not descendants of the supplied candidate build")

    human_rows: dict[str, dict[str, str]] = {}
    labels: dict[str, AxisPairClassification] = {}
    with adjudicated_human_gold.open("r", encoding="utf-8-sig", newline="") as handle:
        for number, raw in enumerate(csv.DictReader(handle), start=2):
            packet_id = str(raw.get("packet_id") or "").strip()
            if packet_id not in packet_lookup or packet_id in locked_ids:
                continue
            if _review_row_is_blank(raw):
                continue
            if packet_id in human_rows:
                raise ValueError(f"duplicate HUMAN gold packet ID at row {number}: {packet_id}")
            if str(raw.get("reviewer") or "").strip() != "HUMAN":
                raise ValueError(f"semantic DEV reviewer must be HUMAN at row {number}")
            label = _classification(raw)
            validate_classification_grounding(label, packet_lookup[packet_id])
            human_rows[packet_id] = dict(raw)
            labels[packet_id] = label

    selected: list[PairedAxisPacket] = []
    stratum_counts: dict[str, dict[str, int]] = {}
    for axis in SEMANTIC_AXES:
        axis_buckets: dict[str, list[PairedAxisPacket]] = defaultdict(list)
        for packet_id, label in labels.items():
            packet = packet_lookup[packet_id]
            if packet.axis == axis:
                axis_buckets[_stratum(label)].append(packet)
        complete_buckets = {
            stratum: axis_buckets[stratum]
            for stratum in STRATA
            if stratum.startswith("COMPLETE_")
        }
        complete = _round_robin_select(
            complete_buckets,
            count=minimum_complete_per_axis,
            seed=f"{seed}|{axis.value}|COMPLETE",
        )
        ambiguous = _round_robin_select(
            {"AMBIGUOUS": axis_buckets["AMBIGUOUS"]},
            count=minimum_ambiguous_per_axis,
            seed=f"{seed}|{axis.value}|AMBIGUOUS",
        )
        chosen = complete + ambiguous
        chosen_ids = {packet.packet_id for packet in chosen}
        chosen += _round_robin_select(
            axis_buckets,
            count=per_axis - len(chosen),
            seed=f"{seed}|{axis.value}|FILL",
            already_selected=chosen_ids,
        )
        selected.extend(chosen)
        counts = defaultdict(int)
        for packet in chosen:
            counts[_stratum(labels[packet.packet_id])] += 1
        stratum_counts[axis.value] = {stratum: counts[stratum] for stratum in STRATA}

    selected.sort(key=lambda packet: (packet.axis.value, packet.packet_id))
    selected_ids = {packet.packet_id for packet in selected}
    if selected_ids & locked_ids:
        raise AssertionError("semantic DEV overlaps a LOCKED packet")
    if len(selected_ids) != len(selected):
        raise AssertionError("semantic DEV packet IDs are not unique")

    output.mkdir(parents=True, exist_ok=True)
    packet_output = output / "dev-packets.jsonl"
    gold_output = output / "dev-human-gold.csv"
    _write_jsonl(packet_output, selected)
    gold_rows: list[dict[str, str]] = []
    for packet in selected:
        row = {
            field: str(human_rows[packet.packet_id].get(field) or "")
            for field in GOLD_FIELDS
        }
        row["packet_id"] = packet.packet_id
        row["axis"] = packet.axis.value
        row["gold_split"] = "DEV"
        row["gold_contract_version"] = "V2_SEMANTIC_DEV_OUTCOME_BLIND"
        gold_rows.append(row)
    _write_gold(gold_output, gold_rows)
    manifest = {
        "schema_version": "moatrader-v2-semantic-dev-preparation/1",
        "status": "V2_SEMANTIC_DEV_PREPARED_OUTCOME_BLIND",
        "candidate_preparation_manifest_sha256": sha256_file(candidate_manifest_path),
        "human_gold_materialization_manifest_sha256": sha256_file(
            human_gold_materialization_manifest
        ),
        "locked_set_preparation_manifest_sha256": sha256_file(locked_manifest_path),
        "adjudicated_human_gold_sha256": sha256_file(adjudicated_human_gold),
        "dev_packet_sha256": sha256_file(packet_output),
        "dev_human_gold_sha256": sha256_file(gold_output),
        "packet_count": len(selected),
        "per_axis": per_axis,
        "minimum_complete_per_axis": minimum_complete_per_axis,
        "minimum_ambiguous_per_axis": minimum_ambiguous_per_axis,
        "stratum_counts": stratum_counts,
        "semantic_parser_axes": [axis.value for axis in SEMANTIC_AXES],
        "gold_label_authority": "HUMAN",
        "locked_packet_overlap_count": 0,
        "locked_sets_consumed": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "semantic-dev-preparation-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a HUMAN-labelled semantic V2 DEV set disjoint from both LOCKED sets."
    )
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--adjudicated-human-gold", type=Path, required=True)
    parser.add_argument("--human-gold-materialization-manifest", type=Path, required=True)
    parser.add_argument("--locked-set-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-axis", type=int, default=30)
    parser.add_argument("--minimum-complete-per-axis", type=int, default=5)
    parser.add_argument("--minimum-ambiguous-per-axis", type=int, default=5)
    parser.add_argument("--seed", default="MOATRADER_V2_SEMANTIC_DEV_20260822")
    args = parser.parse_args()
    result = prepare_semantic_dev(
        candidate_build=args.candidate_build,
        adjudicated_human_gold=args.adjudicated_human_gold,
        human_gold_materialization_manifest=args.human_gold_materialization_manifest,
        locked_set_build=args.locked_set_build,
        output=args.output,
        per_axis=args.per_axis,
        minimum_complete_per_axis=args.minimum_complete_per_axis,
        minimum_ambiguous_per_axis=args.minimum_ambiguous_per_axis,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
