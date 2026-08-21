from __future__ import annotations

import json
from pathlib import Path

import pytest

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    BlindedExcerpt,
    PairedAxisPacket,
    sha256_file,
)
from scripts.merge_historical_human_review_decisions_v2 import (
    merge_human_review_decisions,
)
from scripts.prepare_historical_locked_sets_v2 import (
    extend_locked_candidates,
    prepare_supplemental_candidates,
)


def _packet(
    index: int,
    axis: OperatingEvidenceAxis,
    previous_text: str,
    current_text: str,
) -> PairedAxisPacket:
    return PairedAxisPacket(
        packet_id=f"PKT_{index:024x}",
        axis=axis,
        previous_excerpts=[
            BlindedExcerpt(
                source_id=f"SRC_{index * 2:020x}",
                text=previous_text,
            )
        ],
        current_excerpts=[
            BlindedExcerpt(
                source_id=f"SRC_{index * 2 + 1:020x}",
                text=current_text,
            )
        ],
    )


def _write_packets(path: Path, rows: list[PairedAxisPacket]) -> None:
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _base_candidate_build(
    root: Path, source_packet_input: Path
) -> tuple[Path, list[PairedAxisPacket]]:
    build = root / "base-candidates"
    build.mkdir()
    natural = [
        _packet(1, OperatingEvidenceAxis.DEMAND, "수요 유지", "수요 유지"),
        _packet(2, OperatingEvidenceAxis.PRICE_MIX, "판매가격 유지", "판매가격 유지"),
    ]
    balanced = [
        _packet(3, OperatingEvidenceAxis.DEMAND, "수요 유지", "수요 유지"),
        _packet(4, OperatingEvidenceAxis.PRICE_MIX, "판매가격 유지", "판매가격 유지"),
    ]
    natural_path = build / "natural-locked-packets.jsonl"
    balanced_path = build / "balanced-candidate-packets.jsonl"
    hint_path = build / "balanced-candidate-selection-hints.jsonl"
    _write_packets(natural_path, natural)
    _write_packets(balanced_path, balanced)
    hint_path.write_text(
        "".join(
            json.dumps(
                {
                    "packet_id": packet.packet_id,
                    "axis": packet.axis.value,
                    "selection_hint": "COMPLETE_NEUTRAL",
                    "gold_label": False,
                    "human_review_required": True,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for packet in balanced
        ),
        encoding="utf-8",
    )
    _write_json(
        build / "candidate-preparation-manifest.json",
        {
            "schema_version": "moatrader-v2-locked-candidate-preparation/1",
            "status": "V2_INDEPENDENT_LOCKED_CANDIDATES_PREPARED_OUTCOME_BLIND",
            "source_packet_count": 26,
            "source_packet_sha256": sha256_file(source_packet_input),
            "natural_locked_packet_sha256": sha256_file(natural_path),
            "balanced_candidate_packet_sha256": sha256_file(balanced_path),
            "balanced_candidate_selection_hint_sha256": sha256_file(hint_path),
            "v1_locked_rows_reused": False,
            "outcome_vault_opened": False,
            "return_data_opened": False,
        },
    )
    return build, [*natural, *balanced]


def test_prepare_and_extend_directional_supplement(tmp_path: Path) -> None:
    excluded_prior = _packet(
        5,
        OperatingEvidenceAxis.DEMAND,
        "수요가 증가하였습니다",
        "수요가 감소하였습니다",
    )
    excluded_dev = _packet(
        6,
        OperatingEvidenceAxis.PRICE_MIX,
        "판매가격이 하락하였습니다",
        "판매가격이 상승하였습니다",
    )
    directional: list[PairedAxisPacket] = []
    index = 100
    for axis, subject in (
        (OperatingEvidenceAxis.DEMAND, "수요가"),
        (OperatingEvidenceAxis.PRICE_MIX, "판매가격이"),
    ):
        for _ in range(5):
            directional.append(
                _packet(
                    index,
                    axis,
                    f"{subject} 증가하였습니다",
                    f"{subject} 감소하였습니다",
                )
            )
            index += 1
        for _ in range(5):
            directional.append(
                _packet(
                    index,
                    axis,
                    f"{subject} 감소하였습니다",
                    f"{subject} 증가하였습니다",
                )
            )
            index += 1

    packet_input = tmp_path / "semantic-packets.jsonl"
    placeholder_base = [
        _packet(1, OperatingEvidenceAxis.DEMAND, "수요 유지", "수요 유지"),
        _packet(2, OperatingEvidenceAxis.PRICE_MIX, "판매가격 유지", "판매가격 유지"),
        _packet(3, OperatingEvidenceAxis.DEMAND, "수요 유지", "수요 유지"),
        _packet(4, OperatingEvidenceAxis.PRICE_MIX, "판매가격 유지", "판매가격 유지"),
    ]
    _write_packets(
        packet_input,
        [*placeholder_base, excluded_prior, excluded_dev, *directional],
    )
    base_build, base_packets = _base_candidate_build(tmp_path, packet_input)
    prior_input = tmp_path / "prior-v1.jsonl"
    dev_input = tmp_path / "dev.jsonl"
    _write_packets(prior_input, [excluded_prior])
    _write_packets(dev_input, [excluded_dev])

    supplement_build = tmp_path / "supplement"
    prepared = prepare_supplemental_candidates(
        packet_input=packet_input,
        base_candidate_build=base_build,
        prior_v1_inputs=[prior_input],
        dev_inputs=[dev_input],
        output=supplement_build,
        minimum_per_axis_direction=5,
    )
    assert prepared["supplemental_packet_count"] == 20
    assert set(prepared["supplemental_directional_cue_counts"].values()) == {5}
    assert prepared["selection_hint_is_gold_label"] is False
    selected = [
        PairedAxisPacket.model_validate_json(line)
        for line in (supplement_build / "supplemental-candidate-packets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {packet.packet_id for packet in selected}.isdisjoint(
        {packet.packet_id for packet in (*base_packets, excluded_prior, excluded_dev)}
    )

    extended = tmp_path / "extended-candidates"
    manifest = extend_locked_candidates(
        base_candidate_build=base_build,
        supplemental_candidate_build=supplement_build,
        output=extended,
    )
    assert manifest["candidate_extension"] is True
    assert manifest["balanced_candidate_count"] == 22
    assert manifest["balanced_candidate_axis_counts"] == {
        "DEMAND": 11,
        "PRICE_MIX": 11,
    }
    assert manifest["outcome_vault_opened"] is False
    assert manifest["return_data_opened"] is False


def test_merge_human_review_batches_and_reject_duplicates(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    shared = {
        "reviewer": "HUMAN",
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    complete = {
        "packet_id": "PKT_000000000000000000000001",
        "axis": "DEMAND",
        "status": "COMPLETE",
        "previous_state": 0,
        "current_state": 1,
        "previous_anchor": "수요 유지",
        "current_anchor": "수요 증가",
    }
    abstention = {
        "packet_id": "PKT_000000000000000000000002",
        "axis": "PRICE_MIX",
        "status": "INSUFFICIENT_EVIDENCE",
        "previous_state": None,
        "current_state": None,
        "previous_anchor": None,
        "current_anchor": None,
    }
    _write_json(first, {**shared, "decisions": [complete]})
    _write_json(second, {**shared, "decisions": [abstention]})
    output = tmp_path / "merged.json"
    merged = merge_human_review_decisions(inputs=[first, second], output=output)
    assert merged["decision_count"] == 2
    assert merged["status_counts"] == {
        "COMPLETE": 1,
        "INSUFFICIENT_EVIDENCE": 1,
    }
    assert len(merged["source_decision_files"]) == 2
    assert output.is_file()

    duplicate = tmp_path / "duplicate.json"
    _write_json(duplicate, {**shared, "decisions": [complete]})
    with pytest.raises(ValueError, match="duplicate packet ID"):
        merge_human_review_decisions(
            inputs=[first, duplicate], output=tmp_path / "not-written.json"
        )
