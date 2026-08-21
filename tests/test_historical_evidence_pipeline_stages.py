from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from moatrader.expectations.eri_null_fixtures import run_production_eri_null_fixtures
from moatrader.expectations.future_eri import EvidenceState, OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisPairClassification,
    BlindedExcerpt,
    PairedAxisPacket,
    sha256_file,
    validate_classification_grounding,
)
from moatrader.llm import FunctionTransport
from scripts.classify_historical_future_eri_evidence import (
    AxisPairClassificationPayload,
    run as run_classifier,
)
from scripts.evaluate_historical_evidence_parser import evaluate_parser
from scripts.prepare_historical_evidence_classification_subset import prepare_subset
from scripts.seal_historical_future_eri_features import evaluate_human_gold_quality


def test_production_eri_null_fixtures_all_pass_before_outcome_open() -> None:
    report = run_production_eri_null_fixtures()

    assert report["all_passed"] is True
    assert report["status"] == "PASSED"
    assert report["fixture_count"] == 6
    assert report["outcome_vault_opened"] is False
    assert report["return_data_opened"] is False
    assert all(item["horizon_trading_days"] == 63 for item in report["cases"])


def test_classifier_runs_packet_subset_concurrently_and_replays_without_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    packets = [_packet(index, axis) for index, axis in enumerate(OperatingEvidenceAxis, start=1)]
    packet_input = tmp_path / "packets.jsonl"
    _write_packets(packet_input, packets)
    output = tmp_path / "classification"

    def handler(request, _response_model):
        packet = json.loads(request.user)
        previous = packet["previous_excerpts"][0]
        current = packet["current_excerpts"][0]
        return AxisPairClassification(
            packet_id=packet["packet_id"],
            axis=packet["axis"],
            previous_state=EvidenceState.STABLE,
            current_state=EvidenceState.IMPROVING,
            previous_source_id=previous["source_id"],
            current_source_id=current["source_id"],
            previous_source_span=previous["text"],
            current_source_span=current["text"],
            confidence=1,
        ).model_dump(mode="python")

    first = run_classifier(
        input_build=tmp_path,
        packet_input=packet_input,
        output=output,
        execute=True,
        workers=3,
        transport=FunctionTransport(handler),
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    second = run_classifier(
        input_build=tmp_path,
        packet_input=packet_input,
        output=output,
        execute=True,
        workers=2,
    )
    stored = [
        json.loads(line)
        for line in (output / "classifications.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert first["classification_count"] == 6
    assert second["classification_count"] == 6
    assert second["usage"]["input_tokens"] == 0
    assert [item["packet_id"] for item in stored] == [packet.packet_id for packet in packets]


def test_classifier_retries_non_verbatim_grounding_before_cache(tmp_path: Path) -> None:
    packet = _packet(1, OperatingEvidenceAxis.DEMAND)
    packet_input = tmp_path / "packets.jsonl"
    _write_packets(packet_input, [packet])
    calls = 0

    def handler(request, _response_model):
        nonlocal calls
        calls += 1
        payload = json.loads(request.user)
        previous = payload["previous_excerpts"][0]
        current = payload["current_excerpts"][0]
        return AxisPairClassification(
            packet_id=payload["packet_id"],
            axis=payload["axis"],
            previous_state=EvidenceState.STABLE,
            current_state=EvidenceState.IMPROVING,
            previous_source_id=(
                "SRC_ffffffffffffffffffff" if calls == 1 else previous["source_id"]
            ),
            current_source_id=current["source_id"],
            previous_source_span=previous["text"],
            current_source_span=current["text"],
            confidence=1,
        ).model_dump(mode="python")

    status = run_classifier(
        input_build=tmp_path,
        packet_input=packet_input,
        output=tmp_path / "classification",
        execute=True,
        workers=1,
        transport=FunctionTransport(handler),
    )

    assert calls == 2
    assert status["classification_count"] == 1
    assert (tmp_path / "classification" / "responses" / f"{packet.packet_id}.json").is_file()


def test_classifier_canonicalizes_paraphrased_span_to_exact_source(tmp_path: Path) -> None:
    packet = _packet(1, OperatingEvidenceAxis.DEMAND)
    packet_input = tmp_path / "packets.jsonl"
    _write_packets(packet_input, [packet])

    def handler(request, _response_model):
        payload = json.loads(request.user)
        previous = payload["previous_excerpts"][0]
        current = payload["current_excerpts"][0]
        return AxisPairClassificationPayload(
            packet_id=payload["packet_id"],
            axis=payload["axis"],
            previous_state=EvidenceState.STABLE,
            current_state=EvidenceState.IMPROVING,
            previous_source_id=previous["source_id"],
            current_source_id=current["source_id"],
            previous_source_span="상태 유지",
            current_source_span="상태 개선",
            confidence=1,
        )

    run_classifier(
        input_build=tmp_path,
        packet_input=packet_input,
        output=tmp_path / "classification",
        execute=True,
        workers=1,
        transport=FunctionTransport(handler),
    )
    stored = json.loads(
        (tmp_path / "classification" / "classifications.jsonl").read_text(encoding="utf-8")
    )

    assert stored["previous_source_span"] == packet.previous_excerpts[0].text
    assert stored["current_source_span"] == packet.current_excerpts[0].text


def test_classifier_uses_input_identity_instead_of_model_copy(tmp_path: Path) -> None:
    packet = _packet(1, OperatingEvidenceAxis.DEMAND)
    packet_input = tmp_path / "packets.jsonl"
    _write_packets(packet_input, [packet])

    def handler(request, _response_model):
        payload = json.loads(request.user)
        previous = payload["previous_excerpts"][0]
        current = payload["current_excerpts"][0]
        return AxisPairClassificationPayload(
            packet_id="PKT_ffffffffffffffffffffffff",
            axis=OperatingEvidenceAxis.MARGIN,
            previous_state=EvidenceState.STABLE,
            current_state=EvidenceState.IMPROVING,
            previous_source_id=previous["source_id"],
            current_source_id=current["source_id"],
            previous_source_span=previous["text"],
            current_source_span=current["text"],
            confidence=1,
        )

    run_classifier(
        input_build=tmp_path,
        packet_input=packet_input,
        output=tmp_path / "classification",
        execute=True,
        workers=1,
        transport=FunctionTransport(handler),
    )
    stored = json.loads(
        (tmp_path / "classification" / "classifications.jsonl").read_text(encoding="utf-8")
    )

    assert stored["packet_id"] == packet.packet_id
    assert stored["axis"] == packet.axis.value


def test_classifier_retries_complete_payload_missing_grounding_fields(tmp_path: Path) -> None:
    packet = _packet(1, OperatingEvidenceAxis.DEMAND)
    packet_input = tmp_path / "packets.jsonl"
    _write_packets(packet_input, [packet])
    calls = 0

    def handler(request, _response_model):
        nonlocal calls
        calls += 1
        payload = json.loads(request.user)
        if calls == 1:
            return AxisPairClassificationPayload(
                packet_id=payload["packet_id"],
                axis=payload["axis"],
                status="COMPLETE",
                confidence=0.5,
            )
        previous = payload["previous_excerpts"][0]
        current = payload["current_excerpts"][0]
        return AxisPairClassificationPayload(
            packet_id=payload["packet_id"],
            axis=payload["axis"],
            previous_state=EvidenceState.STABLE,
            current_state=EvidenceState.IMPROVING,
            previous_source_id=previous["source_id"],
            current_source_id=current["source_id"],
            previous_source_span=previous["text"],
            current_source_span=current["text"],
            confidence=1,
        )

    status = run_classifier(
        input_build=tmp_path,
        packet_input=packet_input,
        output=tmp_path / "classification",
        execute=True,
        workers=1,
        transport=FunctionTransport(handler),
    )

    assert calls == 2
    assert status["classification_count"] == 1


def test_grounding_accepts_any_excerpt_sharing_the_same_document_source_id() -> None:
    source_id = "SRC_00000000000000000001"
    packet = PairedAxisPacket(
        packet_id="PKT_000000000000000000000001",
        axis=OperatingEvidenceAxis.DEMAND,
        previous_excerpts=[
            BlindedExcerpt(source_id=source_id, text="수요가 전년보다 감소했습니다."),
            BlindedExcerpt(source_id=source_id, text="매출유형"),
        ],
        current_excerpts=[
            BlindedExcerpt(source_id=source_id, text="수요가 전년보다 증가했습니다."),
            BlindedExcerpt(source_id=source_id, text="매출액"),
        ],
    )
    classification = AxisPairClassification(
        packet_id=packet.packet_id,
        axis=packet.axis,
        previous_state=EvidenceState.WEAKENING,
        current_state=EvidenceState.IMPROVING,
        previous_source_id=source_id,
        current_source_id=source_id,
        previous_source_span="전년보다 감소",
        current_source_span="전년보다 증가",
        confidence=1,
    )

    validate_classification_grounding(classification, packet)


def test_classifier_clears_model_states_when_status_abstains(tmp_path: Path) -> None:
    packet = _packet(1, OperatingEvidenceAxis.CAPACITY_CAPEX)
    packet_input = tmp_path / "packets.jsonl"
    _write_packets(packet_input, [packet])

    def handler(request, _response_model):
        payload = json.loads(request.user)
        previous = payload["previous_excerpts"][0]
        current = payload["current_excerpts"][0]
        return AxisPairClassificationPayload(
            packet_id=payload["packet_id"],
            axis=payload["axis"],
            status="AMBIGUOUS",
            previous_state=EvidenceState.STABLE,
            current_state=EvidenceState.WEAKENING,
            previous_source_id=previous["source_id"],
            current_source_id=current["source_id"],
            previous_source_span=previous["text"],
            current_source_span=current["text"],
            confidence=0.5,
        )

    run_classifier(
        input_build=tmp_path,
        packet_input=packet_input,
        output=tmp_path / "classification",
        execute=True,
        workers=1,
        transport=FunctionTransport(handler),
    )
    stored = json.loads(
        (tmp_path / "classification" / "classifications.jsonl").read_text(encoding="utf-8")
    )

    assert stored["status"] == "AMBIGUOUS"
    assert stored["previous_state"] is None
    assert stored["current_state"] is None
    assert stored["previous_source_span"] is None
    assert stored["current_source_span"] is None


def test_dev_freeze_and_locked_test_are_split_aware_and_single_use(tmp_path: Path) -> None:
    dev_packets = [_packet(index, axis) for index, axis in enumerate(OperatingEvidenceAxis, start=1)]
    locked_packets = [
        _packet(index + 10, axis) for index, axis in enumerate(OperatingEvidenceAxis, start=1)
    ]
    dev_input = tmp_path / "dev.jsonl"
    locked_input = tmp_path / "locked.jsonl"
    _write_packets(dev_input, dev_packets)
    _write_packets(locked_input, locked_packets)
    rows: list[dict[str, str]] = []

    def classification(packet: PairedAxisPacket) -> AxisPairClassification:
        previous = packet.previous_excerpts[0]
        current = packet.current_excerpts[0]
        return AxisPairClassification(
            packet_id=packet.packet_id,
            axis=packet.axis,
            previous_state=EvidenceState.STABLE,
            current_state=EvidenceState.IMPROVING,
            previous_source_id=previous.source_id,
            current_source_id=current.source_id,
            previous_source_span=previous.text,
            current_source_span=current.text,
            confidence=1,
        )

    def classification_build(path: Path, packets: list[PairedAxisPacket], packet_path: Path) -> None:
        path.mkdir()
        (path / "classifications.jsonl").write_text(
            "".join(classification(packet).model_dump_json() + "\n" for packet in packets),
            encoding="utf-8",
        )
        (path / "stage-status.json").write_text(
            json.dumps(
                {
                    "status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE",
                    "input_blinded_packet_sha256": sha256_file(packet_path),
                    "parser_version": "parser-test-v1",
                    "prompt_sha256": "a" * 64,
                    "requested_model": "fixture",
                }
            ),
            encoding="utf-8",
        )

    for split, packets in (("DEV", dev_packets), ("LOCKED_TEST", locked_packets)):
        for packet in packets:
            label = classification(packet)
            rows.append(
                {
                    "packet_id": packet.packet_id,
                    "axis": packet.axis.value,
                    "human_status": "COMPLETE",
                    "human_previous_state": "0",
                    "human_current_state": "1",
                    "human_previous_source_id": str(label.previous_source_id),
                    "human_current_source_id": str(label.current_source_id),
                    "human_previous_source_span": str(label.previous_source_span),
                    "human_current_source_span": str(label.current_source_span),
                    "gold_split": split,
                }
            )
    gold = tmp_path / "gold.csv"
    with gold.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dev_build = tmp_path / "dev-build"
    locked_build = tmp_path / "locked-build"
    classification_build(dev_build, dev_packets, dev_input)
    classification_build(locked_build, locked_packets, locked_input)
    freeze = tmp_path / "parser-freeze.json"

    dev_status = evaluate_parser(
        packet_input=dev_input,
        classification_build=dev_build,
        human_gold=gold,
        split="DEV",
        output=tmp_path / "dev-eval",
        minimum_gold_per_axis=1,
        minimum_overall_agreement=1,
        minimum_axis_agreement=1,
        freeze_manifest_output=freeze,
        locked_packet_input=locked_input,
    )
    consumption = tmp_path / "locked-consumption.json"
    locked_status = evaluate_parser(
        packet_input=locked_input,
        classification_build=locked_build,
        human_gold=gold,
        split="LOCKED_TEST",
        output=tmp_path / "locked-eval",
        minimum_gold_per_axis=1,
        minimum_overall_agreement=1,
        minimum_axis_agreement=1,
        parser_freeze_manifest=freeze,
        locked_consumption_record=consumption,
    )

    assert dev_status["status"] == "DEV_PASSED_PARSER_FROZEN"
    assert locked_status["status"] == "LOCKED_TEST_PASSED"
    assert json.loads(consumption.read_text(encoding="utf-8"))["status"] == "COMPLETED_SINGLE_USE"
    with pytest.raises(FileExistsError, match="already consumed"):
        evaluate_parser(
            packet_input=locked_input,
            classification_build=locked_build,
            human_gold=gold,
            split="LOCKED_TEST",
            output=tmp_path / "locked-eval-second",
            minimum_gold_per_axis=1,
            minimum_overall_agreement=1,
            minimum_axis_agreement=1,
            parser_freeze_manifest=freeze,
            locked_consumption_record=consumption,
        )


def _packet(index: int, axis: OperatingEvidenceAxis, *, complete: bool = True) -> PairedAxisPacket:
    previous = BlindedExcerpt(source_id=f"SRC_{index * 2:020x}", text="상태가 유지되고 있습니다.")
    current = BlindedExcerpt(source_id=f"SRC_{index * 2 + 1:020x}", text="상태가 개선되고 있습니다.")
    return PairedAxisPacket(
        packet_id=f"PKT_{index:024x}",
        axis=axis,
        previous_excerpts=[previous],
        current_excerpts=[current] if complete else [],
    )


def _write_packets(path: Path, packets: list[PairedAxisPacket]) -> None:
    path.write_text(
        "".join(
            json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, sort_keys=True) + "\n"
            for packet in packets
        ),
        encoding="utf-8",
    )


def test_gold_subset_is_exactly_twenty_per_axis(tmp_path: Path) -> None:
    packets: list[PairedAxisPacket] = []
    rows: list[dict[str, str]] = []
    index = 1
    for axis in OperatingEvidenceAxis:
        for axis_index in range(40):
            packet = _packet(index, axis)
            packets.append(packet)
            rows.append(
                {
                    "packet_id": packet.packet_id,
                    "axis": axis.value,
                    "gold_split": "DEV" if axis_index < 20 else "LOCKED_TEST",
                }
            )
            index += 1
    packet_input = tmp_path / "all.jsonl"
    gold = tmp_path / "gold.csv"
    output = tmp_path / "dev.jsonl"
    _write_packets(packet_input, packets)
    with gold.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = prepare_subset(
        packet_input=packet_input,
        output=output,
        mode="DEV",
        human_gold=gold,
    )

    assert manifest["selected_packet_count"] == 120
    assert set(manifest["axis_packet_counts"].values()) == {20}
    assert len(output.read_text(encoding="utf-8").splitlines()) == 120


def test_candidate_subset_keeps_only_pairs_with_all_six_axes(tmp_path: Path) -> None:
    packets: list[PairedAxisPacket] = []
    index = 1
    for pair_index in range(2):
        for axis in OperatingEvidenceAxis:
            packets.append(
                _packet(
                    index,
                    axis,
                    complete=not (pair_index == 1 and axis == OperatingEvidenceAxis.BACKLOG),
                )
            )
            index += 1
    packet_input = tmp_path / "all.jsonl"
    output = tmp_path / "candidate.jsonl"
    _write_packets(packet_input, packets)

    manifest = prepare_subset(
        packet_input=packet_input,
        output=output,
        mode="CANDIDATE_COMPLETE",
        expected_candidate_pairs=1,
    )

    assert manifest["selected_pair_count"] == 1
    assert manifest["selected_packet_count"] == 6
    assert set(manifest["axis_packet_counts"].values()) == {1}


def test_quality_gate_reads_only_requested_gold_split(tmp_path: Path) -> None:
    packets: dict[str, PairedAxisPacket] = {}
    classifications: dict[str, AxisPairClassification] = {}
    rows: list[dict[str, str]] = []
    index = 1
    for axis in OperatingEvidenceAxis:
        for split in ("DEV", "LOCKED_TEST"):
            packet = _packet(index, axis)
            packets[packet.packet_id] = packet
            previous = packet.previous_excerpts[0]
            current = packet.current_excerpts[0]
            if split == "LOCKED_TEST":
                classifications[packet.packet_id] = AxisPairClassification(
                    packet_id=packet.packet_id,
                    axis=axis,
                    previous_state=EvidenceState.STABLE,
                    current_state=EvidenceState.IMPROVING,
                    previous_source_id=previous.source_id,
                    current_source_id=current.source_id,
                    previous_source_span=previous.text,
                    current_source_span=current.text,
                    confidence=1,
                )
            rows.append(
                {
                    "packet_id": packet.packet_id,
                    "axis": axis.value,
                    "human_status": "COMPLETE",
                    "human_previous_state": "0",
                    "human_current_state": "1",
                    "human_previous_source_id": previous.source_id,
                    "human_current_source_id": current.source_id,
                    "human_previous_source_span": previous.text,
                    "human_current_source_span": current.text,
                    "gold_split": split,
                }
            )
            index += 1
    gold = tmp_path / "gold.csv"
    with gold.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    quality = evaluate_human_gold_quality(
        human_gold_path=gold,
        classifications=classifications,
        packets=packets,
        minimum_gold_per_axis=1,
        minimum_overall_agreement=1,
        minimum_axis_agreement=1,
        gold_split="LOCKED_TEST",
    )

    assert quality["gate_passed"] is True
    assert quality["reviewed_count"] == 6
    assert quality["evaluated_gold_split"] == "LOCKED_TEST"
    assert quality["source_span_grounding_rate"] == 1
    assert all(item["reviewed"] == 1 for item in quality["by_axis"].values())
