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
from scripts.materialize_historical_natural_retest_v2 import (
    freeze_natural_retest_measurement,
    materialize_natural_retest_human_gold,
)
from scripts.prepare_historical_natural_retest_v2 import (
    RETEST_CONTRACT,
    RETEST_SPLIT,
    prepare_natural_retest_candidates,
)


def _packet(index: int, axis: OperatingEvidenceAxis) -> PairedAxisPacket:
    return PairedAxisPacket(
        packet_id=f"PKT_{index:024x}",
        axis=axis,
        previous_excerpts=[
            BlindedExcerpt(source_id=f"SRC_{index * 2:020x}", text="이전 공시 원문")
        ],
        current_excerpts=[
            BlindedExcerpt(
                source_id=f"SRC_{index * 2 + 1:020x}", text="현재 공시 원문"
            )
        ],
    )


def _write_packets(path: Path, rows: list[PairedAxisPacket]) -> None:
    path.write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> dict[str, Path]:
    axes = (OperatingEvidenceAxis.DEMAND, OperatingEvidenceAxis.PRICE_MIX)
    packets = [
        _packet(axis_index * 10_000 + index, axis)
        for axis_index, axis in enumerate(axes, start=1)
        for index in range(70)
    ]
    packet_input = tmp_path / "semantic-packets.jsonl"
    _write_packets(packet_input, packets)

    by_axis = {
        axis: [packet for packet in packets if packet.axis == axis] for axis in axes
    }
    prior_v1_rows = [by_axis[axis][0] for axis in axes]
    dev_rows = [by_axis[axis][1] for axis in axes]
    old_natural_rows = [by_axis[axis][2] for axis in axes]
    old_balanced_rows = [by_axis[axis][3] for axis in axes]
    paths: dict[str, Path] = {"packet_input": packet_input}
    for name, rows in (
        ("prior_v1", prior_v1_rows),
        ("dev", dev_rows),
        ("old_natural", old_natural_rows),
        ("old_balanced", old_balanced_rows),
    ):
        path = tmp_path / f"{name}.jsonl"
        _write_packets(path, rows)
        paths[name] = path

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

    evaluation = tmp_path / "failed-natural-evaluation.json"
    _write_json(
        evaluation,
        {
            "status": "V2_NATURAL_EVIDENCE_PARSER_NOT_VALIDATED",
            "locked_kind": "NATURAL",
            "gate_passed": False,
            "parser_freeze_sha256": sha256_file(freeze),
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    paths["evaluation"] = evaluation

    consumption = tmp_path / "natural-consumption.json"
    _write_json(
        consumption,
        {
            "status": "COMPLETED_SINGLE_USE",
            "locked_kind": "NATURAL",
            "gate_passed": False,
            "parser_freeze_sha256": sha256_file(freeze),
            "locked_packet_sha256": sha256_file(paths["old_natural"]),
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    paths["consumption"] = consumption
    return paths


def test_prepare_natural_retest_is_disjoint_and_outcome_blind(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    output = tmp_path / "natural-retest"
    result = prepare_natural_retest_candidates(
        packet_input=paths["packet_input"],
        prior_v1_inputs=[paths["prior_v1"]],
        dev_inputs=[paths["dev"]],
        prior_v2_locked_inputs=[paths["old_natural"], paths["old_balanced"]],
        failed_natural_evaluation_manifest=paths["evaluation"],
        failed_natural_consumption_record=paths["consumption"],
        parser_freeze_manifest=paths["freeze"],
        output=output,
        per_axis=20,
        seed="TEST_NATURAL_RETEST",
    )

    assert result["status"] == "V2_NATURAL_RETEST_1_PREPARED_OUTCOME_BLIND"
    assert result["packet_count"] == 40
    assert result["axis_counts"] == {"DEMAND": 20, "PRICE_MIX": 20}
    assert result["selection_used_parser_classifications"] is False
    assert result["selection_used_post_test_disagreement_rows"] is False
    assert result["first_natural_test_remains_consumed"] is True
    assert result["first_natural_result_superseded"] is False
    assert result["outcome_vault_opened"] is False
    assert result["return_data_opened"] is False
    assert result["value_data_opened"] is False
    assert result["per_pbr_role"] == "NOT_USED"

    selected_ids = {
        PairedAxisPacket.model_validate_json(line).packet_id
        for line in (output / "natural-retest-packets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    }
    excluded_ids: set[str] = set()
    for key in ("prior_v1", "dev", "old_natural", "old_balanced"):
        excluded_ids.update(
            PairedAxisPacket.model_validate_json(line).packet_id
            for line in paths[key].read_text(encoding="utf-8").splitlines()
            if line
        )
    assert not selected_ids & excluded_ids

    with (output / "natural-retest-human-gold-template.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    assert {row["gold_split"] for row in rows} == {RETEST_SPLIT}
    assert {row["gold_contract_version"] for row in rows} == {RETEST_CONTRACT}
    assert all(not row["human_status"] and not row["reviewer"] for row in rows)


def test_prepare_natural_retest_requires_both_frozen_v2_inputs(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    with pytest.raises(ValueError, match="both prior V2 Natural and Balanced"):
        prepare_natural_retest_candidates(
            packet_input=paths["packet_input"],
            prior_v1_inputs=[paths["prior_v1"]],
            dev_inputs=[paths["dev"]],
            prior_v2_locked_inputs=[paths["old_natural"]],
            failed_natural_evaluation_manifest=paths["evaluation"],
            failed_natural_consumption_record=paths["consumption"],
            parser_freeze_manifest=paths["freeze"],
            output=tmp_path / "natural-retest",
            per_axis=20,
        )


def test_prepare_natural_retest_rejects_unconsumed_or_passing_first_test(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    consumption = json.loads(paths["consumption"].read_text(encoding="utf-8"))
    consumption["gate_passed"] = True
    _write_json(paths["consumption"], consumption)
    with pytest.raises(ValueError, match="consumed failing Natural test"):
        prepare_natural_retest_candidates(
            packet_input=paths["packet_input"],
            prior_v1_inputs=[paths["prior_v1"]],
            dev_inputs=[paths["dev"]],
            prior_v2_locked_inputs=[
                paths["old_natural"],
                paths["old_balanced"],
            ],
            failed_natural_evaluation_manifest=paths["evaluation"],
            failed_natural_consumption_record=paths["consumption"],
            parser_freeze_manifest=paths["freeze"],
            output=tmp_path / "natural-retest",
            per_axis=20,
        )


def test_natural_retest_human_gold_freeze_evaluation_and_combine(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    candidate_build = tmp_path / "natural-retest-candidates"
    prepare_natural_retest_candidates(
        packet_input=paths["packet_input"],
        prior_v1_inputs=[paths["prior_v1"]],
        dev_inputs=[paths["dev"]],
        prior_v2_locked_inputs=[paths["old_natural"], paths["old_balanced"]],
        failed_natural_evaluation_manifest=paths["evaluation"],
        failed_natural_consumption_record=paths["consumption"],
        parser_freeze_manifest=paths["freeze"],
        output=candidate_build,
        per_axis=20,
        seed="TEST_NATURAL_RETEST_MATERIALIZATION",
    )
    packet_path = candidate_build / "natural-retest-packets.jsonl"
    packets = [
        PairedAxisPacket.model_validate_json(line)
        for line in packet_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    review_path = tmp_path / "natural-retest-review.json"
    _write_json(
        review_path,
        {
            "reviewer": "HUMAN",
            "human_reviewer_name": "Test Reviewer",
            "attestation": "YES",
            "review_date": "2026-08-22",
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
            "decisions": [
                {
                    "packet_id": packet.packet_id,
                    "axis": packet.axis.value,
                    "status": "COMPLETE",
                    "previous_state": 0,
                    "current_state": 0,
                    "previous_anchor": "이전 공시 원문",
                    "current_anchor": "현재 공시 원문",
                    "review_notes": "두 기간 모두 명시적 안정 상태로 독립 판정",
                    "contract_self_check": "YES",
                }
                for packet in packets
            ],
        },
    )
    human_gold_build = tmp_path / "natural-retest-human-gold"
    materialized = materialize_natural_retest_human_gold(
        candidate_build=candidate_build,
        review_decisions=review_path,
        output=human_gold_build,
    )
    assert materialized["status"] == (
        "V2_NATURAL_RETEST_1_HUMAN_GOLD_MATERIALIZED_OUTCOME_BLIND"
    )
    assert materialized["review_decision_count"] == 40
    assert materialized["model_fields_accepted"] is False
    assert materialized["contract_self_check_required"] is True

    retest_freeze = tmp_path / "natural-retest-freeze.json"
    frozen = freeze_natural_retest_measurement(
        parser_freeze_manifest=paths["freeze"],
        candidate_build=candidate_build,
        human_gold_build=human_gold_build,
        output=retest_freeze,
    )
    assert frozen["status"] == (
        "V2_NATURAL_RETEST_1_FROZEN_AWAITING_SINGLE_USE_TEST"
    )
    assert frozen["semantic_parser_root_freeze_sha256"] == sha256_file(
        paths["freeze"]
    )

    classifications = [
        AxisPairClassification(
            packet_id=packet.packet_id,
            axis=packet.axis,
            status="COMPLETE",
            confidence=1,
            previous_state=0,
            current_state=0,
            previous_source_id=packet.previous_excerpts[0].source_id,
            current_source_id=packet.current_excerpts[0].source_id,
            previous_source_span=packet.previous_excerpts[0].text,
            current_source_span=packet.current_excerpts[0].text,
        )
        for packet in packets
    ]
    classification_build = tmp_path / "natural-retest-classification"
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
            "input_blinded_packet_sha256": sha256_file(packet_path),
            "classification_sha256": sha256_file(classification_path),
            "packet_count": len(packets),
            "classification_count": len(classifications),
            "parser_profile": spec.profile.value,
            "parser_version": spec.parser_version,
            "prompt_sha256": spec.prompt_sha256,
            "requested_model": "gpt-5.6-luna",
            "semantic_execution_scope": (
                SemanticExecutionScope.PILOT_OR_LOCKED_VALIDATION.value
            ),
            "full_historical_execution_authorized": False,
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    natural_output = tmp_path / "natural-retest-evaluation"
    consumption = tmp_path / "natural-retest-consumption.json"
    natural = evaluate_v2_locked_parser(
        packet_input=packet_path,
        classification_build=classification_build,
        human_gold=human_gold_build / "natural-retest-human-gold.csv",
        parser_freeze_manifest=retest_freeze,
        locked_consumption_record=consumption,
        output=natural_output,
        locked_kind="NATURAL_RETEST_1",
        minimum_natural_per_axis=20,
        minimum_overall_directional_agreement=1,
        minimum_axis_directional_agreement=1,
        maximum_neutral_to_bullish_rate=0,
        maximum_false_stable_rate=0,
        maximum_opposite_direction_count=0,
    )
    assert natural["status"] == "V2_NATURAL_RETEST_1_LOCKED_TEST_PASSED"
    assert natural["natural_frequency_gate_passed"] is True
    assert natural["first_natural_test_remains_consumed"] is True
    assert natural["first_natural_result_superseded"] is False
    assert json.loads(consumption.read_text(encoding="utf-8"))["status"] == (
        "COMPLETED_SINGLE_USE"
    )

    balanced = tmp_path / "balanced-evaluation.json"
    _write_json(
        balanced,
        {
            "status": "V2_BALANCED_LOCKED_TEST_PASSED",
            "locked_kind": "BALANCED",
            "gate_passed": True,
            "natural_frequency_gate_passed": False,
            "directional_strata_gate_passed": True,
            "parser_profile": spec.profile.value,
            "parser_version": spec.parser_version,
            "prompt_sha256": spec.prompt_sha256,
            "requested_model": "gpt-5.6-luna",
            "classification_stage_sha256": "b" * 64,
            "classification_sha256": "c" * 64,
            "input_blinded_packet_sha256": "d" * 64,
            "parser_freeze_sha256": sha256_file(paths["freeze"]),
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    )
    combined_path = tmp_path / "combined.json"
    combined = combine_v2_locked_evaluations(
        natural_evaluation_manifest=natural_output / "stage-status.json",
        balanced_evaluation_manifest=balanced,
        parser_freeze_manifest=paths["freeze"],
        output=combined_path,
    )
    assert combined["status"] == "V2_LOCKED_TESTS_PASSED"
    assert combined["natural_locked_kind"] == "NATURAL_RETEST_1"
    assert combined["natural_retest_number"] == 1

    with pytest.raises(FileExistsError, match="already consumed"):
        evaluate_v2_locked_parser(
            packet_input=packet_path,
            classification_build=classification_build,
            human_gold=human_gold_build / "natural-retest-human-gold.csv",
            parser_freeze_manifest=retest_freeze,
            locked_consumption_record=consumption,
            output=tmp_path / "natural-retest-second-evaluation",
            locked_kind="NATURAL_RETEST_1",
        )


def test_natural_retest_materialization_rejects_model_fields(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    candidate_build = tmp_path / "natural-retest-candidates"
    prepare_natural_retest_candidates(
        packet_input=paths["packet_input"],
        prior_v1_inputs=[paths["prior_v1"]],
        dev_inputs=[paths["dev"]],
        prior_v2_locked_inputs=[paths["old_natural"], paths["old_balanced"]],
        failed_natural_evaluation_manifest=paths["evaluation"],
        failed_natural_consumption_record=paths["consumption"],
        parser_freeze_manifest=paths["freeze"],
        output=candidate_build,
        per_axis=20,
    )
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
        materialize_natural_retest_human_gold(
            candidate_build=candidate_build,
            review_decisions=review_path,
            output=tmp_path / "human-gold",
        )


def test_natural_retest_materialization_requires_contract_self_check(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    candidate_build = tmp_path / "natural-retest-candidates"
    prepare_natural_retest_candidates(
        packet_input=paths["packet_input"],
        prior_v1_inputs=[paths["prior_v1"]],
        dev_inputs=[paths["dev"]],
        prior_v2_locked_inputs=[paths["old_natural"], paths["old_balanced"]],
        failed_natural_evaluation_manifest=paths["evaluation"],
        failed_natural_consumption_record=paths["consumption"],
        parser_freeze_manifest=paths["freeze"],
        output=candidate_build,
        per_axis=20,
    )
    packet = PairedAxisPacket.model_validate_json(
        (candidate_build / "natural-retest-packets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()[0]
    )
    review_path = tmp_path / "missing-contract-self-check.json"
    _write_json(
        review_path,
        {
            "reviewer": "HUMAN",
            "human_reviewer_name": "Test Reviewer",
            "attestation": "YES",
            "review_date": "2026-08-22",
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
            "decisions": [
                {
                    "packet_id": packet.packet_id,
                    "axis": packet.axis.value,
                    "status": "INSUFFICIENT_EVIDENCE",
                    "review_notes": "계약상 두 기간의 실현 상태가 부족함",
                }
            ],
        },
    )
    with pytest.raises(ValueError, match="contract_self_check must be exactly YES"):
        materialize_natural_retest_human_gold(
            candidate_build=candidate_build,
            review_decisions=review_path,
            output=tmp_path / "human-gold",
        )
