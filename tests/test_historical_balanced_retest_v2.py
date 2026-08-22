from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisPairClassification,
    BlindedExcerpt,
    PairedAxisPacket,
    sha256_file,
)
from scripts.classify_historical_future_eri_evidence import (
    ParserProfile,
    SemanticExecutionScope,
    parser_spec,
)
from scripts.evaluate_historical_evidence_parser_v2 import (
    combine_v2_locked_evaluations,
    evaluate_v2_locked_parser,
)
from scripts.merge_historical_human_review_decisions_v2 import (
    merge_human_review_decisions,
)
from scripts.materialize_historical_balanced_retest_v2 import (
    freeze_balanced_retest_measurement,
    materialize_balanced_retest_human_gold,
)
from scripts.prepare_historical_balanced_retest_v2 import (
    RETEST_CONTRACT,
    RETEST_SPLIT,
    prepare_balanced_retest_candidates,
)
from scripts.prepare_historical_locked_sets_v2 import GOLD_FIELDS


AXES = (OperatingEvidenceAxis.DEMAND, OperatingEvidenceAxis.PRICE_MIX)


def _packet(index: int, axis: OperatingEvidenceAxis, previous: str, current: str) -> PairedAxisPacket:
    return PairedAxisPacket(
        packet_id=f"PKT_{index:024x}",
        axis=axis,
        previous_excerpts=[
            BlindedExcerpt(source_id=f"SRC_{index * 2:020x}", text=previous)
        ],
        current_excerpts=[
            BlindedExcerpt(source_id=f"SRC_{index * 2 + 1:020x}", text=current)
        ],
    )


def _write_packets(path: Path, rows: list[PairedAxisPacket]) -> None:
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_strict_candidate_subset(
    path: Path, packets: list[PairedAxisPacket]
) -> Path:
    path.mkdir()
    packet_path = path / "balanced-retest-candidate-packets.jsonl"
    _write_packets(packet_path, packets)
    manifest_path = path / "balanced-retest-preparation-manifest.json"
    _write_json(
        manifest_path,
        {
            "status": "V2_BALANCED_RETEST_1_CANDIDATES_PREPARED_OUTCOME_BLIND",
            "candidate_packet_count": len(packets),
            "balanced_retest_candidate_packet_sha256": sha256_file(packet_path),
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    return path


def _write_completed_review_import(
    path: Path,
    *,
    decisions: list[dict[str, object]],
    candidate_build: Path,
    review_date: str,
) -> None:
    _write_json(
        path,
        {
            "status": "HUMAN_REVIEW_DECISIONS_IMPORTED_OUTCOME_BLIND",
            "review_type": "balanced-retest-1",
            "reviewer": "HUMAN",
            "human_reviewer_name": "Test Reviewer",
            "attestation": "YES",
            "review_date": review_date,
            "candidate_count": len(decisions),
            "decision_count": len(decisions),
            "reviewed_count": len(decisions),
            "pending_count": 0,
            "row_error_count": 0,
            "candidate_manifest_sha256": sha256_file(
                candidate_build / "balanced-retest-preparation-manifest.json"
            ),
            "candidate_excerpts_verified": True,
            "decision_export_formulas_verified": True,
            "workbook_read_only_verified": True,
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
            "decisions": decisions,
        },
    )


def _gold_row(
    packet: PairedAxisPacket,
    *,
    status: str,
    previous_state: int | None = None,
    current_state: int | None = None,
) -> dict[str, str]:
    row = {field: "" for field in GOLD_FIELDS}
    row.update(
        packet_id=packet.packet_id,
        axis=packet.axis.value,
        human_status=status,
        reviewer="HUMAN",
        gold_split="V2_BALANCED_CANDIDATE_REVIEW",
        gold_contract_version="V2_DIRECTIONAL_BALANCED_CANDIDATE_POOL",
        review_notes="독립 HUMAN 사전 판정",
    )
    if status == "COMPLETE":
        assert previous_state is not None and current_state is not None
        row.update(
            human_previous_state=str(previous_state),
            human_current_state=str(current_state),
            human_previous_source_id=packet.previous_excerpts[0].source_id,
            human_current_source_id=packet.current_excerpts[0].source_id,
            human_previous_source_span=packet.previous_excerpts[0].text,
            human_current_source_span=packet.current_excerpts[0].text,
        )
    return row


def _fixture(tmp_path: Path) -> dict[str, object]:
    packets: list[PairedAxisPacket] = []
    prior_rows: list[dict[str, str]] = []
    fresh_labels: dict[str, tuple[str, int | None, int | None]] = {}

    for axis_index, axis in enumerate(AXES, start=1):
        base = axis_index * 100_000
        if axis == OperatingEvidenceAxis.DEMAND:
            neutral_text = "현재 수요는 전년 대비 동일하게 유지되었습니다."
        else:
            neutral_text = "현재 평균 판매가격은 전년 대비 동일하게 유지되었습니다."
        for offset in range(7):
            packet = _packet(base + 100 + offset, axis, neutral_text, neutral_text)
            packets.append(packet)
            prior_rows.append(
                _gold_row(packet, status="COMPLETE", previous_state=0, current_state=0)
            )
            fresh_labels[packet.packet_id] = ("COMPLETE", 0, 0)
        for offset in range(7):
            packet = _packet(
                base + 200 + offset,
                axis,
                "사업 현황 일반 설명입니다.",
                "현재 계획만 설명합니다.",
            )
            packets.append(packet)
            prior_rows.append(_gold_row(packet, status="INSUFFICIENT_EVIDENCE"))
            fresh_labels[packet.packet_id] = ("INSUFFICIENT_EVIDENCE", None, None)
        for offset in range(7):
            packet = _packet(
                base + 300 + offset,
                axis,
                "증가와 감소가 함께 나타나 방향이 충돌합니다.",
                "상승과 하락이 함께 나타나 방향이 충돌합니다.",
            )
            packets.append(packet)
            prior_rows.append(_gold_row(packet, status="AMBIGUOUS"))
            fresh_labels[packet.packet_id] = ("AMBIGUOUS", None, None)

        if axis == OperatingEvidenceAxis.DEMAND:
            negative_pair = (
                "전년 대비 제품 수요가 증가했습니다.",
                "전년 대비 제품 수요가 감소했습니다.",
            )
            positive_pair = (negative_pair[1], negative_pair[0])
        else:
            negative_pair = (
                "전년 대비 평균 판매가격이 상승했습니다.",
                "전년 대비 평균 판매가격이 하락했습니다.",
            )
            positive_pair = (negative_pair[1], negative_pair[0])
        for offset in range(12):
            packet = _packet(base + 1_000 + offset, axis, *negative_pair)
            packets.append(packet)
            fresh_labels[packet.packet_id] = ("COMPLETE", 1, -1)
        for offset in range(12):
            packet = _packet(base + 2_000 + offset, axis, *positive_pair)
            packets.append(packet)
            fresh_labels[packet.packet_id] = ("COMPLETE", -1, 1)

    exclusion_packets = [
        _packet(1, OperatingEvidenceAxis.DEMAND, "이전", "현재"),
        _packet(2, OperatingEvidenceAxis.PRICE_MIX, "이전", "현재"),
        _packet(3, OperatingEvidenceAxis.DEMAND, "이전", "현재"),
        _packet(4, OperatingEvidenceAxis.PRICE_MIX, "이전", "현재"),
    ]
    packets.extend(exclusion_packets)
    packet_input = tmp_path / "semantic-packets.jsonl"
    _write_packets(packet_input, packets)

    paths: dict[str, object] = {
        "packet_input": packet_input,
        "fresh_labels": fresh_labels,
    }
    for name, rows in (
        ("prior_v1", [exclusion_packets[0]]),
        ("dev", [exclusion_packets[1]]),
        ("old_natural", [exclusion_packets[2]]),
        ("old_balanced", [exclusion_packets[3]]),
    ):
        path = tmp_path / f"{name}.jsonl"
        _write_packets(path, rows)
        paths[name] = path

    prior_human_gold = tmp_path / "prior-human-gold.csv"
    with prior_human_gold.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GOLD_FIELDS))
        writer.writeheader()
        writer.writerows(prior_rows)
    paths["prior_human_gold"] = prior_human_gold
    prior_human_manifest = tmp_path / "prior-human-materialization.json"
    _write_json(
        prior_human_manifest,
        {
            "status": "V2_HUMAN_REVIEW_DECISIONS_MATERIALIZED_OUTCOME_BLIND",
            "reviewer": "HUMAN",
            "adjudicated_human_gold_sha256": sha256_file(prior_human_gold),
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    paths["prior_human_manifest"] = prior_human_manifest

    spec = parser_spec(ParserProfile.DEMAND_PRICE_MIX_V2)
    freeze = tmp_path / "parser-freeze.json"
    _write_json(
        freeze,
        {
            "schema_version": "moatrader-historical-evidence-parser-freeze-v2/2",
            "status": "V2_PARSER_FROZEN_AWAITING_DUAL_INDEPENDENT_LOCKED_TESTS",
            "parser_profile": spec.profile.value,
            "parser_version": spec.parser_version,
            "prompt_sha256": spec.prompt_sha256,
            "requested_model": "gpt-5.6-luna",
            "natural_locked_packet_sha256": sha256_file(paths["old_natural"]),
            "balanced_locked_packet_sha256": sha256_file(paths["old_balanced"]),
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    paths["freeze"] = freeze

    evaluation = tmp_path / "failed-balanced-evaluation.json"
    _write_json(
        evaluation,
        {
            "status": "V2_BALANCED_EVIDENCE_PARSER_NOT_VALIDATED",
            "locked_kind": "BALANCED",
            "gate_passed": False,
            "parser_freeze_sha256": sha256_file(freeze),
            "input_blinded_packet_sha256": sha256_file(paths["old_balanced"]),
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    paths["evaluation"] = evaluation
    consumption = tmp_path / "balanced-consumption.json"
    _write_json(
        consumption,
        {
            "status": "COMPLETED_SINGLE_USE",
            "locked_kind": "BALANCED",
            "gate_passed": False,
            "parser_freeze_sha256": sha256_file(freeze),
            "locked_packet_sha256": sha256_file(paths["old_balanced"]),
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    paths["consumption"] = consumption
    return paths


def _prepare(paths: dict[str, object], output: Path) -> dict[str, object]:
    return prepare_balanced_retest_candidates(
        packet_input=paths["packet_input"],
        prior_v1_inputs=[paths["prior_v1"]],
        dev_inputs=[paths["dev"]],
        prior_v2_locked_inputs=[paths["old_natural"], paths["old_balanced"]],
        prior_human_gold=paths["prior_human_gold"],
        prior_human_gold_materialization_manifest=paths["prior_human_manifest"],
        failed_balanced_evaluation_manifest=paths["evaluation"],
        failed_balanced_consumption_record=paths["consumption"],
        parser_freeze_manifest=paths["freeze"],
        output=output,
        directional_candidates_per_axis_stratum=10,
        nondirectional_candidates_per_axis_stratum=5,
        seed="TEST_BALANCED_RETEST",
    )


def test_prepare_balanced_retest_is_disjoint_and_model_blind(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "balanced-retest-candidates"
    result = _prepare(paths, output)
    assert result["status"] == "V2_BALANCED_RETEST_1_CANDIDATES_PREPARED_OUTCOME_BLIND"
    assert result["candidate_packet_count"] == 70
    assert result["selection_used_parser_classifications"] is False
    assert result["selection_used_post_test_disagreement_rows"] is False
    assert result["prior_human_labels_accepted_as_retest_gold"] is False
    assert result["fresh_independent_human_review_required"] is True
    assert result["first_balanced_test_remains_consumed"] is True
    assert result["first_balanced_result_superseded"] is False
    assert result["outcome_vault_opened"] is False
    assert result["per_pbr_role"] == "NOT_USED"

    selected = {
        PairedAxisPacket.model_validate_json(line).packet_id
        for line in (output / "balanced-retest-candidate-packets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    }
    excluded: set[str] = set()
    for key in ("prior_v1", "dev", "old_natural", "old_balanced"):
        excluded.update(
            PairedAxisPacket.model_validate_json(line).packet_id
            for line in paths[key].read_text(encoding="utf-8").splitlines()
            if line
        )
    assert not selected & excluded


def test_balanced_retest_materialize_freeze_evaluate_and_combine(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    candidate_build = tmp_path / "balanced-retest-candidates"
    _prepare(paths, candidate_build)
    packet_path = candidate_build / "balanced-retest-candidate-packets.jsonl"
    packets = [
        PairedAxisPacket.model_validate_json(line)
        for line in packet_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    fresh_labels = paths["fresh_labels"]
    decisions: list[dict[str, object]] = []
    for packet in packets:
        status, previous_state, current_state = fresh_labels[packet.packet_id]
        decision: dict[str, object] = {
            "packet_id": packet.packet_id,
            "axis": packet.axis.value,
            "status": status,
            "review_notes": "새 독립 HUMAN 재검토 판정",
            "contract_self_check": "YES",
        }
        if status == "COMPLETE":
            decision.update(
                previous_state=previous_state,
                current_state=current_state,
                previous_anchor=packet.previous_excerpts[0].text,
                current_anchor=packet.current_excerpts[0].text,
            )
        decisions.append(decision)
    split = len(packets) // 2
    first_candidate_build = _write_strict_candidate_subset(
        tmp_path / "balanced-retest-first-candidates", packets[:split]
    )
    second_candidate_build = _write_strict_candidate_subset(
        tmp_path / "balanced-retest-second-candidates", packets[split:]
    )
    first_review_path = tmp_path / "balanced-retest-first-review-import.json"
    second_review_path = tmp_path / "balanced-retest-second-review-import.json"
    _write_completed_review_import(
        first_review_path,
        decisions=decisions[:split],
        candidate_build=first_candidate_build,
        review_date="2026-08-21",
    )
    _write_completed_review_import(
        second_review_path,
        decisions=decisions[split:],
        candidate_build=second_candidate_build,
        review_date="2026-08-22",
    )
    source_hashes_before = {
        path: sha256_file(path)
        for path in (
            first_review_path,
            second_review_path,
            first_candidate_build / "balanced-retest-candidate-packets.jsonl",
            second_candidate_build / "balanced-retest-candidate-packets.jsonl",
            packet_path,
        )
    }
    review_path = tmp_path / "balanced-retest-review.json"
    merged = merge_human_review_decisions(
        inputs=[first_review_path, second_review_path],
        output=review_path,
        input_candidate_builds=[first_candidate_build, second_candidate_build],
        combined_candidate_build=candidate_build,
    )
    assert merged["human_reviewer_name"] == "Test Reviewer"
    assert merged["attestation"] == "YES"
    assert merged["review_date"] == "2026-08-22"
    assert merged["candidate_count"] == len(packets)
    assert {path: sha256_file(path) for path in source_hashes_before} == (
        source_hashes_before
    )
    human_gold_build = tmp_path / "balanced-retest-human-gold"
    materialized = materialize_balanced_retest_human_gold(
        candidate_build=candidate_build,
        review_decisions=review_path,
        output=human_gold_build,
    )
    assert materialized["status"] == (
        "V2_BALANCED_RETEST_1_HUMAN_GOLD_MATERIALIZED_OUTCOME_BLIND"
    )
    assert materialized["selected_packet_count"] == 50
    assert materialized["model_fields_accepted"] is False
    for counts in materialized["selected_stratum_counts"].values():
        assert set(counts.values()) == {5}

    retest_freeze = tmp_path / "balanced-retest-freeze.json"
    frozen = freeze_balanced_retest_measurement(
        parser_freeze_manifest=paths["freeze"],
        candidate_build=candidate_build,
        human_gold_build=human_gold_build,
        output=retest_freeze,
    )
    assert frozen["status"] == "V2_BALANCED_RETEST_1_FROZEN_AWAITING_SINGLE_USE_TEST"
    assert frozen["semantic_parser_root_freeze_sha256"] == sha256_file(paths["freeze"])

    selected_packet_path = human_gold_build / "balanced-retest-packets.jsonl"
    selected_packets = [
        PairedAxisPacket.model_validate_json(line)
        for line in selected_packet_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    classifications: list[AxisPairClassification] = []
    for packet in selected_packets:
        status, previous_state, current_state = fresh_labels[packet.packet_id]
        payload: dict[str, object] = {
            "packet_id": packet.packet_id,
            "axis": packet.axis,
            "status": status,
            "confidence": 1,
        }
        if status == "COMPLETE":
            payload.update(
                previous_state=previous_state,
                current_state=current_state,
                previous_source_id=packet.previous_excerpts[0].source_id,
                current_source_id=packet.current_excerpts[0].source_id,
                previous_source_span=packet.previous_excerpts[0].text,
                current_source_span=packet.current_excerpts[0].text,
            )
        classifications.append(AxisPairClassification.model_validate(payload))
    classification_build = tmp_path / "balanced-retest-classification"
    classification_build.mkdir()
    classification_path = classification_build / "classifications.jsonl"
    classification_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in classifications),
        encoding="utf-8",
    )
    spec = parser_spec(ParserProfile.DEMAND_PRICE_MIX_V2)
    _write_json(
        classification_build / "stage-status.json",
        {
            "status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE",
            "input_blinded_packet_sha256": sha256_file(selected_packet_path),
            "classification_sha256": sha256_file(classification_path),
            "packet_count": len(selected_packets),
            "classification_count": len(classifications),
            "parser_profile": spec.profile.value,
            "parser_version": spec.parser_version,
            "prompt_sha256": spec.prompt_sha256,
            "requested_model": "gpt-5.6-luna",
            "semantic_execution_scope": SemanticExecutionScope.PILOT_OR_LOCKED_VALIDATION.value,
            "full_historical_execution_authorized": False,
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    balanced_output = tmp_path / "balanced-retest-evaluation"
    balanced_consumption = tmp_path / "balanced-retest-consumption.json"
    balanced = evaluate_v2_locked_parser(
        packet_input=selected_packet_path,
        classification_build=classification_build,
        human_gold=human_gold_build / "balanced-retest-human-gold.csv",
        parser_freeze_manifest=retest_freeze,
        locked_consumption_record=balanced_consumption,
        output=balanced_output,
        locked_kind="BALANCED_RETEST_1",
        minimum_per_axis_stratum=5,
        minimum_overall_directional_agreement=1,
        minimum_axis_directional_agreement=1,
        maximum_neutral_to_bullish_rate=0,
        maximum_false_stable_rate=0,
        maximum_opposite_direction_count=0,
    )
    assert balanced["status"] == "V2_BALANCED_RETEST_1_LOCKED_TEST_PASSED"
    assert balanced["directional_strata_gate_passed"] is True
    assert balanced["first_balanced_test_remains_consumed"] is True
    assert balanced["first_balanced_result_superseded"] is False

    natural = tmp_path / "natural-evaluation.json"
    _write_json(
        natural,
        {
            "status": "V2_NATURAL_RETEST_1_LOCKED_TEST_PASSED",
            "locked_kind": "NATURAL_RETEST_1",
            "gate_passed": True,
            "natural_frequency_gate_passed": True,
            "directional_strata_gate_passed": False,
            "parser_profile": spec.profile.value,
            "parser_version": spec.parser_version,
            "prompt_sha256": spec.prompt_sha256,
            "requested_model": "gpt-5.6-luna",
            "classification_stage_sha256": "a" * 64,
            "classification_sha256": "b" * 64,
            "input_blinded_packet_sha256": "c" * 64,
            "parser_freeze_sha256": "d" * 64,
            "semantic_parser_root_freeze_sha256": sha256_file(paths["freeze"]),
            "retest_number": 1,
            "first_natural_test_remains_consumed": True,
            "first_natural_result_superseded": False,
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    combined_path = tmp_path / "combined.json"
    combined = combine_v2_locked_evaluations(
        natural_evaluation_manifest=natural,
        balanced_evaluation_manifest=balanced_output / "stage-status.json",
        parser_freeze_manifest=paths["freeze"],
        output=combined_path,
    )
    assert combined["status"] == "V2_LOCKED_TESTS_PASSED"
    assert combined["balanced_locked_kind"] == "BALANCED_RETEST_1"
    assert combined["balanced_retest_number"] == 1

    with pytest.raises(FileExistsError, match="already consumed"):
        evaluate_v2_locked_parser(
            packet_input=selected_packet_path,
            classification_build=classification_build,
            human_gold=human_gold_build / "balanced-retest-human-gold.csv",
            parser_freeze_manifest=retest_freeze,
            locked_consumption_record=balanced_consumption,
            output=tmp_path / "balanced-retest-second-evaluation",
            locked_kind="BALANCED_RETEST_1",
        )


def test_balanced_retest_materialization_rejects_model_fields(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    candidate_build = tmp_path / "balanced-retest-candidates"
    _prepare(paths, candidate_build)
    review_path = tmp_path / "contaminated-review.json"
    _write_json(
        review_path,
        {
            "reviewer": "HUMAN",
            "human_reviewer_name": "Test Reviewer",
            "attestation": "YES",
            "review_date": "2026-08-22",
            "model_label": "COMPLETE_NEUTRAL",
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
            "decisions": [],
        },
    )
    with pytest.raises(ValueError, match="forbidden model-derived fields"):
        materialize_balanced_retest_human_gold(
            candidate_build=candidate_build,
            review_decisions=review_path,
            output=tmp_path / "human-gold",
        )
