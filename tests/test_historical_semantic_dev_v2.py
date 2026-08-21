from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    BlindedExcerpt,
    PairedAxisPacket,
    sha256_file,
)
from scripts.prepare_historical_locked_sets_v2 import GOLD_FIELDS
from scripts.prepare_historical_semantic_dev_v2 import prepare_semantic_dev


def _packet(index: int, axis: OperatingEvidenceAxis) -> PairedAxisPacket:
    return PairedAxisPacket(
        packet_id=f"PKT_{index:024x}",
        axis=axis,
        previous_excerpts=[BlindedExcerpt(source_id=f"SRC_{index * 2:020x}", text="이전")],
        current_excerpts=[
            BlindedExcerpt(source_id=f"SRC_{index * 2 + 1:020x}", text="현재")
        ],
    )


def _write_packets(path: Path, packets: list[PairedAxisPacket]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(packet.model_dump_json() + "\n" for packet in packets),
        encoding="utf-8",
    )


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_semantic_dev_is_human_labelled_and_disjoint_from_both_locked_sets(
    tmp_path: Path,
) -> None:
    axes = (OperatingEvidenceAxis.DEMAND, OperatingEvidenceAxis.PRICE_MIX)
    by_axis = {
        axis: [
            _packet(axis_index * 1_000 + index, axis)
            for index in range(36)
        ]
        for axis_index, axis in enumerate(axes, start=1)
    }
    natural_candidates = [by_axis[axis][0] for axis in axes]
    balanced_candidates = [
        packet for axis in axes for packet in by_axis[axis][1:]
    ]
    candidate_build = tmp_path / "candidates"
    natural_candidate_path = candidate_build / "natural-locked-packets.jsonl"
    balanced_candidate_path = candidate_build / "balanced-candidate-packets.jsonl"
    _write_packets(natural_candidate_path, natural_candidates)
    _write_packets(balanced_candidate_path, balanced_candidates)
    _write_manifest(
        candidate_build / "candidate-preparation-manifest.json",
        {
            "status": "V2_INDEPENDENT_LOCKED_CANDIDATES_PREPARED_OUTCOME_BLIND",
            "natural_locked_packet_sha256": sha256_file(natural_candidate_path),
            "balanced_candidate_packet_sha256": sha256_file(balanced_candidate_path),
            "outcome_vault_opened": False,
            "return_data_opened": False,
        },
    )

    locked_natural = natural_candidates
    locked_balanced = [packet for axis in axes for packet in by_axis[axis][1:3]]
    locked_build = tmp_path / "locked"
    natural_locked_path = locked_build / "natural-locked-packets.jsonl"
    balanced_locked_path = locked_build / "balanced-locked-packets.jsonl"
    _write_packets(natural_locked_path, locked_natural)
    _write_packets(balanced_locked_path, locked_balanced)
    _write_manifest(
        locked_build / "locked-set-preparation-manifest.json",
        {
            "status": "V2_DUAL_INDEPENDENT_LOCKED_SETS_PREPARED_OUTCOME_BLIND",
            "natural_locked_packet_sha256": sha256_file(natural_locked_path),
            "balanced_locked_packet_sha256": sha256_file(balanced_locked_path),
            "outcome_vault_opened": False,
            "return_data_opened": False,
        },
    )

    adjudicated = tmp_path / "adjudicated-human-gold.csv"
    with adjudicated.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GOLD_FIELDS))
        writer.writeheader()
        for axis in axes:
            for index, packet in enumerate(by_axis[axis]):
                if index < 13:
                    status = "COMPLETE"
                elif index < 24:
                    status = "AMBIGUOUS"
                else:
                    status = "INSUFFICIENT_EVIDENCE"
                row = {field: "" for field in GOLD_FIELDS}
                row.update(
                    packet_id=packet.packet_id,
                    axis=axis.value,
                    human_status=status,
                    reviewer="HUMAN",
                    review_notes="outcome-blind fixture review",
                )
                if status == "COMPLETE":
                    row.update(
                        human_previous_state="0",
                        human_current_state="0",
                        human_previous_source_id=packet.previous_excerpts[0].source_id,
                        human_current_source_id=packet.current_excerpts[0].source_id,
                        human_previous_source_span="이전",
                        human_current_source_span="현재",
                    )
                writer.writerow(row)
    materialization_manifest = tmp_path / "human-gold-materialization-manifest.json"
    _write_manifest(
        materialization_manifest,
        {
            "status": "V2_HUMAN_REVIEW_DECISIONS_MATERIALIZED_OUTCOME_BLIND",
            "adjudicated_human_gold_sha256": sha256_file(adjudicated),
            "outcome_vault_opened": False,
            "return_data_opened": False,
        },
    )

    output = tmp_path / "dev"
    manifest = prepare_semantic_dev(
        candidate_build=candidate_build,
        adjudicated_human_gold=adjudicated,
        human_gold_materialization_manifest=materialization_manifest,
        locked_set_build=locked_build,
        output=output,
        per_axis=20,
        minimum_complete_per_axis=3,
        minimum_ambiguous_per_axis=3,
    )

    assert manifest["status"] == "V2_SEMANTIC_DEV_PREPARED_OUTCOME_BLIND"
    assert manifest["packet_count"] == 40
    assert manifest["locked_packet_overlap_count"] == 0
    assert manifest["per_pbr_role"] == "NOT_USED"
    assert all(
        counts["COMPLETE_NEUTRAL"] >= 3 and counts["AMBIGUOUS"] >= 3
        for counts in manifest["stratum_counts"].values()
    )
    dev_packets = {
        PairedAxisPacket.model_validate_json(line).packet_id
        for line in (output / "dev-packets.jsonl").read_text(encoding="utf-8").splitlines()
    }
    locked_ids = {packet.packet_id for packet in locked_natural + locked_balanced}
    assert dev_packets.isdisjoint(locked_ids)
    with pytest.raises(FileExistsError, match="new or empty"):
        prepare_semantic_dev(
            candidate_build=candidate_build,
            adjudicated_human_gold=adjudicated,
            human_gold_materialization_manifest=materialization_manifest,
            locked_set_build=locked_build,
            output=output,
            per_axis=20,
            minimum_complete_per_axis=3,
            minimum_ambiguous_per_axis=3,
        )
