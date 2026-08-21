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
    ParserProfile,
    SEMANTIC_PARSER_VERSION_V2,
    SEMANTIC_PROMPT_SHA256_V2,
    SemanticExecutionScope,
    build_request,
    run as run_classifier,
)
from scripts.evaluate_historical_evidence_parser import evaluate_parser
from scripts.prepare_historical_evidence_classification_subset import prepare_subset
from scripts.prepare_historical_semantic_cost_manifest_v2 import prepare_cost_manifest
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


def test_semantic_v2_profile_is_demand_price_mix_only_and_never_defaults_na_to_zero(
    tmp_path: Path,
) -> None:
    demand = _packet(1, OperatingEvidenceAxis.DEMAND)
    request = build_request(
        demand,
        parser_profile=ParserProfile.DEMAND_PRICE_MIX_V2,
    )

    assert request.metadata["parser_version"] == SEMANTIC_PARSER_VERSION_V2
    assert "Missing is NA, never neutral" in request.system
    assert "current_state minus previous_state" in request.system
    assert "Revenue alone is not demand" in request.system
    assert "Pricing policy" in request.system
    assert request.prompt_cache_key == "moatrader:historical-demand-price-mix-v2-0"

    with pytest.raises(ValueError, match="only Demand and PriceMix"):
        build_request(
            _packet(2, OperatingEvidenceAxis.MARGIN),
            parser_profile=ParserProfile.DEMAND_PRICE_MIX_V2,
        )

    packet_input = tmp_path / "semantic.jsonl"
    _write_packets(packet_input, [demand, _packet(3, OperatingEvidenceAxis.PRICE_MIX)])
    status = run_classifier(
        input_build=tmp_path,
        packet_input=packet_input,
        output=tmp_path / "semantic-prep",
        parser_profile=ParserProfile.DEMAND_PRICE_MIX_V2,
    )
    assert status["parser_profile"] == ParserProfile.DEMAND_PRICE_MIX_V2.value
    assert status["parser_version"] == SEMANTIC_PARSER_VERSION_V2
    assert status["prompt_sha256"] == SEMANTIC_PROMPT_SHA256_V2
    assert status["outcome_vault_opened"] is False


def test_semantic_v2_profile_rejects_nonsemantic_packet_file(tmp_path: Path) -> None:
    packet_input = tmp_path / "forbidden.jsonl"
    _write_packets(packet_input, [_packet(1, OperatingEvidenceAxis.BACKLOG)])

    with pytest.raises(ValueError, match="forbidden axis"):
        run_classifier(
            input_build=tmp_path,
            packet_input=packet_input,
            output=tmp_path / "semantic-prep",
            parser_profile=ParserProfile.DEMAND_PRICE_MIX_V2,
        )


def test_full_semantic_execution_requires_dual_locked_and_cost_authorization(
    tmp_path: Path,
) -> None:
    packets = [
        _packet(1, OperatingEvidenceAxis.DEMAND),
        _packet(2, OperatingEvidenceAxis.PRICE_MIX),
    ]
    packet_input = tmp_path / "semantic.jsonl"
    _write_packets(packet_input, packets)
    calls = 0

    def handler(request, _response_model):
        nonlocal calls
        calls += 1
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

    with pytest.raises(ValueError, match="explicit semantic execution scope"):
        run_classifier(
            input_build=tmp_path,
            packet_input=packet_input,
            output=tmp_path / "missing-scope",
            execute=True,
            parser_profile=ParserProfile.DEMAND_PRICE_MIX_V2,
            transport=FunctionTransport(handler),
        )
    assert calls == 0

    pilot = run_classifier(
        input_build=tmp_path,
        packet_input=packet_input,
        output=tmp_path / "pilot",
        execute=True,
        parser_profile=ParserProfile.DEMAND_PRICE_MIX_V2,
        semantic_execution_scope=SemanticExecutionScope.PILOT_OR_LOCKED_VALIDATION,
        transport=FunctionTransport(handler),
    )
    assert pilot["full_historical_execution_authorized"] is False
    assert calls == 2

    (tmp_path / "private").mkdir()
    (tmp_path / "llm").mkdir()
    pair_source = tmp_path / "private" / "filing-pairs.jsonl"
    blinded_source = tmp_path / "llm" / "blinded-packets.jsonl"
    before = tmp_path / "private" / "source-integrity-before.json"
    after = tmp_path / "private" / "source-integrity-after.json"
    source_audit = tmp_path / "source-audit.json"
    source_build_manifest = tmp_path / "build-manifest.json"
    pair_source.write_text('{"fixture": "pair"}\n', encoding="utf-8")
    blinded_source.write_text(packet_input.read_text(encoding="utf-8"), encoding="utf-8")
    integrity_records = [{"path": "readonly/source.html", "sha256": "a" * 64}]
    before.write_text(
        json.dumps(
            {
                "mutation_policy": "ARCANA_AND_MOATRADER_SOURCE_FILES_READ_ONLY",
                "records": integrity_records,
            }
        ),
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(
            {
                "verification_status": "PASS_NO_SOURCE_MUTATION",
                "records": integrity_records,
            }
        ),
        encoding="utf-8",
    )
    source_audit.write_text(
        json.dumps(
            {
                "schema_version": "moatrader-historical-source-audit-v1/2",
                "both_source_systems_used": True,
                "all_arcana_sections_discovered": True,
                "all_arcana_sections_read_for_pairs": True,
                "all_arcana_sections_contributed_to_packets": True,
                "source_files_modified": False,
            }
        ),
        encoding="utf-8",
    )
    source_build_manifest.write_text(
        json.dumps(
            {
                "source_files_modified": False,
                "artifacts": {
                    "source-audit.json": sha256_file(source_audit),
                    "private/filing-pairs.jsonl": sha256_file(pair_source),
                    "llm/blinded-packets.jsonl": sha256_file(blinded_source),
                    "private/source-integrity-before.json": sha256_file(before),
                    "private/source-integrity-after.json": sha256_file(after),
                },
            }
        ),
        encoding="utf-8",
    )

    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "status": "SEMANTIC_REQUIRED_PACKETS_PREPARED_OUTCOME_BLIND",
                "selected_packet_count": 2,
                "semantic_primary_axes": ["DEMAND", "PRICE_MIX"],
                "output_packet_sha256": sha256_file(packet_input),
                "source_hashes": {
                    "filing_pairs": sha256_file(pair_source),
                    "blinded_packets": sha256_file(blinded_source),
                },
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    locked = tmp_path / "dual-locked.json"
    locked.write_text(
        json.dumps(
            {
                "status": "V2_LOCKED_TESTS_PASSED",
                "natural_frequency_gate_passed": True,
                "directional_strata_gate_passed": True,
                "parser_version": SEMANTIC_PARSER_VERSION_V2,
                "prompt_sha256": SEMANTIC_PROMPT_SHA256_V2,
                "requested_model": "gpt-5.6-luna",
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    cost = tmp_path / "cost.json"
    cost.write_text(
        json.dumps(
            {
                "status": "FULL_SEMANTIC_RUN_COST_PRESPECIFIED_NO_EXTERNAL_CALL",
                "api_calls_executed": False,
                "parser_profile": "DEMAND_PRICE_MIX_V2",
                "parser_version": SEMANTIC_PARSER_VERSION_V2,
                "prompt_sha256": SEMANTIC_PROMPT_SHA256_V2,
                "model": "gpt-5.6-luna",
                "exact_packet_count": 2,
                "token_estimation": {
                    "pilot_prompt_differs_from_frozen_full_prompt": False,
                    "pilot_contract_matches_frozen_full_prompt": True,
                },
                "inputs": {
                    "semantic_packet_sha256": sha256_file(packet_input),
                    "semantic_selection_manifest_sha256": sha256_file(selection),
                    "pilot_stage_manifests": [
                        {
                            "parser_profile": "DEMAND_PRICE_MIX_V2",
                            "parser_version": SEMANTIC_PARSER_VERSION_V2,
                            "prompt_sha256": SEMANTIC_PROMPT_SHA256_V2,
                            "requested_model": "gpt-5.6-luna",
                            "semantic_execution_scope": "PILOT_OR_LOCKED_VALIDATION",
                        },
                        {
                            "parser_profile": "DEMAND_PRICE_MIX_V2",
                            "parser_version": SEMANTIC_PARSER_VERSION_V2,
                            "prompt_sha256": SEMANTIC_PROMPT_SHA256_V2,
                            "requested_model": "gpt-5.6-luna",
                            "semantic_execution_scope": "PILOT_OR_LOCKED_VALIDATION",
                        },
                    ],
                },
                "outcome_vault_opened": False,
                "return_data_opened": False,
                "value_data_opened": False,
                "per_pbr_role": "NOT_USED",
            }
        ),
        encoding="utf-8",
    )
    failed_locked = tmp_path / "failed-dual-locked.json"
    failed_payload = json.loads(locked.read_text(encoding="utf-8"))
    failed_payload["status"] = "V2_LOCKED_TESTS_FAILED"
    failed_locked.write_text(json.dumps(failed_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="dual LOCKED tests have not passed"):
        run_classifier(
            input_build=tmp_path,
            packet_input=packet_input,
            output=tmp_path / "failed-full",
            execute=True,
            parser_profile=ParserProfile.DEMAND_PRICE_MIX_V2,
            semantic_execution_scope=SemanticExecutionScope.FULL_HISTORICAL,
            dual_locked_manifest=failed_locked,
            semantic_selection_manifest=selection,
            semantic_cost_manifest=cost,
            transport=FunctionTransport(handler),
        )
    assert calls == 2

    source_audit_text = source_audit.read_text(encoding="utf-8")
    invalid_source_audit = json.loads(source_audit_text)
    invalid_source_audit["all_arcana_sections_read_for_pairs"] = False
    source_audit.write_text(json.dumps(invalid_source_audit), encoding="utf-8")
    with pytest.raises(ValueError, match="all_arcana_sections_read_for_pairs"):
        run_classifier(
            input_build=tmp_path,
            packet_input=packet_input,
            output=tmp_path / "failed-source-full",
            execute=True,
            parser_profile=ParserProfile.DEMAND_PRICE_MIX_V2,
            semantic_execution_scope=SemanticExecutionScope.FULL_HISTORICAL,
            dual_locked_manifest=locked,
            semantic_selection_manifest=selection,
            semantic_cost_manifest=cost,
            transport=FunctionTransport(handler),
        )
    assert calls == 2
    source_audit.write_text(source_audit_text, encoding="utf-8")

    legacy_cost = tmp_path / "legacy-cost.json"
    legacy_cost_payload = json.loads(cost.read_text(encoding="utf-8"))
    legacy_cost_payload["token_estimation"] = {
        "pilot_prompt_differs_from_frozen_full_prompt": True,
        "pilot_contract_matches_frozen_full_prompt": False,
    }
    legacy_cost.write_text(json.dumps(legacy_cost_payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exact frozen V2 pilot executions"):
        run_classifier(
            input_build=tmp_path,
            packet_input=packet_input,
            output=tmp_path / "failed-legacy-cost-full",
            execute=True,
            parser_profile=ParserProfile.DEMAND_PRICE_MIX_V2,
            semantic_execution_scope=SemanticExecutionScope.FULL_HISTORICAL,
            dual_locked_manifest=locked,
            semantic_selection_manifest=selection,
            semantic_cost_manifest=legacy_cost,
            transport=FunctionTransport(handler),
        )
    assert calls == 2

    full = run_classifier(
        input_build=tmp_path,
        packet_input=packet_input,
        output=tmp_path / "full",
        execute=True,
        parser_profile=ParserProfile.DEMAND_PRICE_MIX_V2,
        semantic_execution_scope=SemanticExecutionScope.FULL_HISTORICAL,
        dual_locked_manifest=locked,
        semantic_selection_manifest=selection,
        semantic_cost_manifest=cost,
        transport=FunctionTransport(handler),
    )
    assert full["full_historical_execution_authorized"] is True
    assert full["semantic_execution_scope"] == "FULL_HISTORICAL"
    assert full["status"] == "FULL_SEMANTIC_CLASSIFICATION_COMPLETE_OUTCOMES_CLOSED"
    assert full["dual_locked_manifest_sha256"] == sha256_file(locked)
    assert calls == 4


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
        classification_path = path / "classifications.jsonl"
        classification_path.write_text(
            "".join(classification(packet).model_dump_json() + "\n" for packet in packets),
            encoding="utf-8",
        )
        (path / "stage-status.json").write_text(
            json.dumps(
                {
                    "status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE",
                    "input_blinded_packet_sha256": sha256_file(packet_path),
                    "classification_sha256": sha256_file(classification_path),
                    "packet_count": len(packets),
                    "classification_count": len(packets),
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


def test_semantic_cost_manifest_freezes_calls_tokens_cost_and_prompt_before_run(
    tmp_path: Path,
) -> None:
    packet_input = tmp_path / "semantic.jsonl"
    packets = [
        _packet(1, OperatingEvidenceAxis.DEMAND),
        _packet(2, OperatingEvidenceAxis.PRICE_MIX),
    ]
    _write_packets(packet_input, packets)
    selection = tmp_path / "semantic.jsonl.manifest.json"
    selection.write_text(
        json.dumps(
            {
                "selected_packet_count": 2,
                "output_packet_sha256": sha256_file(packet_input),
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    pilot_paths: list[Path] = []
    for index in range(2):
        path = tmp_path / f"pilot-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE",
                    "packet_count": 1,
                    "classification_count": 1,
                    "parser_profile": "DEMAND_PRICE_MIX_V2",
                    "parser_version": SEMANTIC_PARSER_VERSION_V2,
                    "prompt_sha256": SEMANTIC_PROMPT_SHA256_V2,
                    "requested_model": "gpt-5.6-luna",
                    "semantic_execution_scope": "PILOT_OR_LOCKED_VALIDATION",
                    "full_historical_execution_authorized": False,
                    "outcome_vault_opened": False,
                    "return_data_opened": False,
                    "value_data_opened": False,
                    "per_pbr_role": "NOT_USED",
                    "credentials_persisted": False,
                    "usage": {
                        "input_tokens": 100,
                        "cached_input_tokens": 20,
                        "cache_write_tokens": 10,
                        "output_tokens": 30,
                    },
                }
            ),
            encoding="utf-8",
        )
        pilot_paths.append(path)

    result = prepare_cost_manifest(
        semantic_packet_input=packet_input,
        semantic_selection_manifest=selection,
        pilot_stage_manifests=pilot_paths,
        output=tmp_path / "cost.json",
    )

    assert result["exact_expected_api_calls_without_retries"] == 2
    assert result["axis_counts"] == {"DEMAND": 1, "PRICE_MIX": 1}
    assert result["parser_profile"] == "DEMAND_PRICE_MIX_V2"
    assert result["token_estimation"]["expected_tokens"] == {
        "input_tokens": 200,
        "cached_input_tokens": 40,
        "cache_write_tokens": 20,
        "output_tokens": 60,
    }
    assert result["pricing"]["expected_cost"]["total"] == "0.000106"
    assert result["api_calls_executed"] is False
    assert result["outcome_vault_opened"] is False
    assert result["per_pbr_role"] == "NOT_USED"
    assert result["token_estimation"]["pilot_prompt_differs_from_frozen_full_prompt"] is False
    assert result["token_estimation"]["pilot_contract_matches_frozen_full_prompt"] is True


def test_semantic_cost_manifest_rejects_legacy_pilot_prompt(tmp_path: Path) -> None:
    packet_input = tmp_path / "semantic.jsonl"
    _write_packets(
        packet_input,
        [
            _packet(1, OperatingEvidenceAxis.DEMAND),
            _packet(2, OperatingEvidenceAxis.PRICE_MIX),
        ],
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected_packet_count": 2,
                "output_packet_sha256": sha256_file(packet_input),
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    pilot_paths: list[Path] = []
    for index in range(2):
        path = tmp_path / f"legacy-pilot-{index}.json"
        path.write_text(
            json.dumps(
                {
                    "status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE",
                    "packet_count": 1,
                    "classification_count": 1,
                    "parser_profile": "LEGACY_V1",
                    "parser_version": "historical-evidence-parser-v1.2.0",
                    "prompt_sha256": "a" * 64,
                    "requested_model": "gpt-5.6-luna",
                    "semantic_execution_scope": "PILOT_OR_LOCKED_VALIDATION",
                    "full_historical_execution_authorized": False,
                    "outcome_vault_opened": False,
                    "return_data_opened": False,
                    "value_data_opened": False,
                    "credentials_persisted": False,
                    "per_pbr_role": "NOT_USED",
                    "usage": {
                        "input_tokens": 1,
                        "cached_input_tokens": 0,
                        "cache_write_tokens": 0,
                        "output_tokens": 1,
                    },
                }
            ),
            encoding="utf-8",
        )
        pilot_paths.append(path)

    with pytest.raises(ValueError, match="frozen semantic V2 parser_profile"):
        prepare_cost_manifest(
            semantic_packet_input=packet_input,
            semantic_selection_manifest=selection,
            pilot_stage_manifests=pilot_paths,
            output=tmp_path / "cost.json",
        )


def test_semantic_cost_manifest_rejects_nonsemantic_axis(tmp_path: Path) -> None:
    packet_input = tmp_path / "semantic.jsonl"
    _write_packets(packet_input, [_packet(1, OperatingEvidenceAxis.MARGIN)])
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "selected_packet_count": 1,
                "output_packet_sha256": sha256_file(packet_input),
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only Demand and PriceMix"):
        prepare_cost_manifest(
            semantic_packet_input=packet_input,
            semantic_selection_manifest=selection,
            pilot_stage_manifests=[tmp_path / "missing-a", tmp_path / "missing-b"],
            output=tmp_path / "cost.json",
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
