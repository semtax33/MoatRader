from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from moatrader.expectations.future_eri import EvidenceState, OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisPairClassification,
    BlindedExcerpt,
    HistoricalFilingPair,
    HistoricalRegularFiling,
    HistoricalSourceOrigin,
    HistoricalSourceVariant,
    PairedAxisPacket,
    ReceiptLinkage,
    historical_pair_id,
    packet_id,
    sha256_file,
)
from scripts.audit_historical_v1r_feasibility import audit_v1r_feasibility
from scripts.build_historical_complete_features_v1r import build_v1r_features
from scripts.build_historical_future_eri_evidence import (
    FROZEN_FEATURE_CONTRACT_V1,
    FROZEN_FEATURE_CONTRACT_V1R,
)
from scripts.evaluate_historical_evidence_parser_v1r import (
    create_v1r_parser_freeze,
    evaluate_v1r_locked_parser,
)
from scripts.freeze_historical_v1r_contract import freeze_v1r_contract
from scripts.prepare_historical_v1r_locked_set import prepare_v1r_locked_set


SEOUL = ZoneInfo("Asia/Seoul")
ORIGINS = (
    HistoricalSourceOrigin.ARCANA_BUSINESS_HTML,
    HistoricalSourceOrigin.ARCANA_FINANCE_COMMENT_HTML,
    HistoricalSourceOrigin.ARCANA_FINANCE_STATEMENT_HTML,
    HistoricalSourceOrigin.MOATRADER_OPENDART_ARCHIVE,
)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_id(pair_index: int, side: int, origin_index: int) -> str:
    return f"SRC_{pair_index * 100 + side * 10 + origin_index:020x}"


def _filing(*, ticker: str, pair_index: int, current: bool) -> HistoricalRegularFiling:
    rcept_no = "20200814000001" if current else "20200330000001"
    period = date(2020, 6, 30) if current else date(2020, 3, 31)
    timestamp = datetime(2020, 8 if current else 3, 14 if current else 30, 16, tzinfo=SEOUL)
    variants = [
        HistoricalSourceVariant(
            origin=origin,
            path=f"C:/v1r-fixture/{pair_index}/{current}/{origin.value}.source",
            raw_sha256=_hash(f"{pair_index}|{current}|{origin.value}"),
            byte_count=1,
            receipt_linkage=(
                ReceiptLinkage.EXACT_METADATA
                if origin == HistoricalSourceOrigin.MOATRADER_OPENDART_ARCHIVE
                else ReceiptLinkage.INFERRED_TICKER_PERIOD
            ),
        )
        for origin in ORIGINS
    ]
    return HistoricalRegularFiling(
        ticker=ticker,
        issuer_name=f"회사{ticker}",
        rcept_no=rcept_no,
        report_name="반기보고서" if current else "분기보고서",
        report_code="11012" if current else "11013",
        fiscal_period_end=period,
        published_at=timestamp,
        available_at=timestamp,
        signal_timestamp=timestamp,
        source_variants=variants,
    )


def _pair(pair_index: int) -> HistoricalFilingPair:
    ticker = f"{pair_index + 1:06d}"
    previous = _filing(ticker=ticker, pair_index=pair_index, current=False)
    current = _filing(ticker=ticker, pair_index=pair_index, current=True)
    return HistoricalFilingPair(
        pair_id=historical_pair_id(ticker, previous.rcept_no, current.rcept_no),
        ticker=ticker,
        previous=previous,
        current=current,
    )


def _packet_and_private(
    pair: HistoricalFilingPair, pair_index: int, axis: OperatingEvidenceAxis
) -> tuple[PairedAxisPacket, dict[str, dict[str, str]]]:
    sources: dict[str, dict[str, str]] = {}
    previous: list[BlindedExcerpt] = []
    current: list[BlindedExcerpt] = []
    for origin_index, origin in enumerate(ORIGINS, start=1):
        previous_id = _source_id(pair_index, 0, origin_index)
        current_id = _source_id(pair_index, 1, origin_index)
        previous.append(
            BlindedExcerpt(source_id=previous_id, text=f"이전 {origin.value} {axis.value}")
        )
        current.append(
            BlindedExcerpt(source_id=current_id, text=f"현재 {origin.value} {axis.value}")
        )
        sources[previous_id] = {"side": "previous", "origin": origin.value}
        sources[current_id] = {"side": "current", "origin": origin.value}
    return (
        PairedAxisPacket(
            packet_id=packet_id(pair.pair_id, axis),
            axis=axis,
            previous_excerpts=previous,
            current_excerpts=current,
        ),
        sources,
    )


def _states_for_band(pair_index: int, axis_index: int, repeats_per_band: int) -> tuple[int, int]:
    band_index = pair_index // repeats_per_band
    if band_index == 0:
        return 0, -1
    if band_index == 1:
        return (0, -1) if axis_index < 2 else (0, 0)
    if band_index == 2:
        return 0, 0
    if band_index == 3:
        return (0, 1) if axis_index < 2 else (0, 0)
    return 0, 1


def _classification(
    packet: PairedAxisPacket, *, previous_state: int, current_state: int
) -> AxisPairClassification:
    previous = packet.previous_excerpts[0]
    current = packet.current_excerpts[0]
    return AxisPairClassification(
        packet_id=packet.packet_id,
        axis=packet.axis,
        previous_state=EvidenceState(previous_state),
        current_state=EvidenceState(current_state),
        previous_source_id=previous.source_id,
        current_source_id=current.source_id,
        previous_source_span=previous.text,
        current_source_span=current.text,
        confidence=1,
    )


def _write_source_build(tmp_path: Path, *, repeats_per_band: int) -> tuple[Path, list[PairedAxisPacket], list[AxisPairClassification]]:
    pair_count = repeats_per_band * 5
    build = tmp_path / "source-build"
    (build / "llm").mkdir(parents=True)
    (build / "private").mkdir()
    pairs: list[HistoricalFilingPair] = []
    packets: list[PairedAxisPacket] = []
    classifications: list[AxisPairClassification] = []
    private_rows: list[dict[str, object]] = []
    for pair_index in range(pair_count):
        pair = _pair(pair_index)
        pairs.append(pair)
        pair_sources: dict[str, dict[str, str]] = {}
        for axis_index, axis in enumerate(OperatingEvidenceAxis):
            packet, sources = _packet_and_private(pair, pair_index, axis)
            packets.append(packet)
            pair_sources.update(sources)
            previous_state, current_state = _states_for_band(
                pair_index, axis_index, repeats_per_band
            )
            classifications.append(
                _classification(
                    packet,
                    previous_state=previous_state,
                    current_state=current_state,
                )
            )
        private_rows.append(
            {
                "pair_id": pair.pair_id,
                "ticker": pair.ticker,
                "issuer_name": f"회사{pair.ticker}",
                "previous_rcept_no": pair.previous.rcept_no,
                "current_rcept_no": pair.current.rcept_no,
                "coverage_sector": "TEST",
                "sources": pair_sources,
            }
        )

    def write_jsonl(path: Path, rows: list[object]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
                handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")

    pair_path = build / "private" / "filing-pairs.jsonl"
    private_path = build / "private" / "pair-source-map.jsonl"
    packet_path = build / "llm" / "blinded-packets.jsonl"
    write_jsonl(pair_path, pairs)
    write_jsonl(private_path, private_rows)
    write_jsonl(packet_path, packets)
    source_audit = {
        "schema_version": "moatrader-historical-source-audit-v1/2",
        "research_variant": "V1R",
        "source_contract_tag": "future-eri-v1r-three-section-preoutcome",
        "same_feature_rule_as_v1": True,
        "arcana_section_selection": [
            "business-info",
            "finance-comment",
            "finance-statement",
        ],
        "arcana_regular_filing_count": pair_count * 2,
        "moatrader_regular_original_filing_count": pair_count * 2,
        "regular_pair_count": pair_count,
        "both_source_systems_used": True,
        "all_arcana_sections_discovered": True,
        "all_arcana_sections_read_for_pairs": True,
        "all_arcana_sections_contributed_to_packets": True,
        "source_effect_audit": {
            "filing_origin_patterns": {"ARCANA_MOATRADER_OVERLAP": pair_count * 2},
            "outcomes_opened": False,
            "returns_opened": False,
            "value_data_opened": False,
        },
        "source_integrity_record_count": 0,
        "source_files_modified": False,
    }
    source_audit_path = build / "source-audit.json"
    source_audit_path.write_text(json.dumps(source_audit), encoding="utf-8")
    (build / "frozen-contract.json").write_text(
        json.dumps(FROZEN_FEATURE_CONTRACT_V1R), encoding="utf-8"
    )
    before = {"mutation_policy": "ARCANA_AND_MOATRADER_SOURCE_FILES_READ_ONLY", "records": []}
    after = {**before, "verification_status": "PASS_NO_SOURCE_MUTATION"}
    before_path = build / "private" / "source-integrity-before.json"
    after_path = build / "private" / "source-integrity-after.json"
    before_path.write_text(json.dumps(before), encoding="utf-8")
    after_path.write_text(json.dumps(after), encoding="utf-8")
    artifacts = {
        "source-audit.json": sha256_file(source_audit_path),
        "private/filing-pairs.jsonl": sha256_file(pair_path),
        "private/pair-source-map.jsonl": sha256_file(private_path),
        "llm/blinded-packets.jsonl": sha256_file(packet_path),
        "private/source-integrity-before.json": sha256_file(before_path),
        "private/source-integrity-after.json": sha256_file(after_path),
    }
    (build / "build-manifest.json").write_text(
        json.dumps(
            {
                "research_variant": "V1R",
                "artifacts": artifacts,
                "source_files_modified": False,
                "outcome_data_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    return build, packets, classifications


def _classification_build(
    path: Path,
    packet_path: Path,
    classifications: list[AxisPairClassification],
) -> None:
    path.mkdir()
    (path / "classifications.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in classifications),
        encoding="utf-8",
    )
    (path / "stage-status.json").write_text(
        json.dumps(
            {
                "status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE",
                "input_blinded_packet_sha256": sha256_file(packet_path),
                "parser_version": "parser-v1r-fixture",
                "prompt_sha256": "a" * 64,
                "requested_model": "fixture",
            }
        ),
        encoding="utf-8",
    )


def _prepare_and_evaluate_locked(
    tmp_path: Path, source_build: Path
) -> tuple[Path, Path, Path]:
    prior = tmp_path / "prior-v1.jsonl"
    dev = tmp_path / "dev.jsonl"
    prior.write_text(json.dumps({"packet_id": "PKT_" + "e" * 24}) + "\n", encoding="utf-8")
    dev.write_text(json.dumps({"packet_id": "PKT_" + "d" * 24}) + "\n", encoding="utf-8")
    locked = tmp_path / "locked"
    prepare_v1r_locked_set(
        input_build=source_build,
        prior_v1_inputs=[prior],
        dev_inputs=[dev],
        output=locked,
        minimum_per_axis_source_stratum=1,
    )
    packet_path = locked / "v1r-locked-packets.jsonl"
    packets = [
        PairedAxisPacket.model_validate_json(line)
        for line in packet_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gold_template = locked / "v1r-locked-human-gold-template.csv"
    with gold_template.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    packet_by_id = {packet.packet_id: packet for packet in packets}
    labels: list[AxisPairClassification] = []
    for row in rows:
        packet = packet_by_id[row["packet_id"]]
        label = _classification(packet, previous_state=0, current_state=1)
        labels.append(label)
        row.update(
            human_status="COMPLETE",
            human_previous_state="0",
            human_current_state="1",
            human_previous_source_id=str(label.previous_source_id),
            human_current_source_id=str(label.current_source_id),
            human_previous_source_span=str(label.previous_source_span),
            human_current_source_span=str(label.current_source_span),
            reviewer="human",
        )
    gold = tmp_path / "v1r-gold.csv"
    with gold.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    locked_classification = tmp_path / "locked-classification"
    _classification_build(locked_classification, packet_path, labels)
    dev_manifest = tmp_path / "dev-evaluation.json"
    dev_manifest.write_text(
        json.dumps(
            {
                "status": "DEV_PASSED_PARSER_READY_TO_FREEZE",
                "parser_version": "parser-v1r-fixture",
                "prompt_sha256": "a" * 64,
                "requested_model": "fixture",
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    parser_freeze = tmp_path / "parser-freeze-v1r.json"
    create_v1r_parser_freeze(
        dev_evaluation_manifest=dev_manifest,
        locked_set_preparation_manifest=locked / "locked-set-preparation-manifest.json",
        locked_packet_input=packet_path,
        source_strata_input=locked / "v1r-locked-source-strata.jsonl",
        human_gold=gold,
        output=parser_freeze,
    )
    evaluation = tmp_path / "locked-evaluation"
    stage = evaluate_v1r_locked_parser(
        packet_input=packet_path,
        source_strata_input=locked / "v1r-locked-source-strata.jsonl",
        classification_build=locked_classification,
        human_gold=gold,
        parser_freeze_manifest=parser_freeze,
        locked_consumption_record=tmp_path / "locked-consumption.json",
        output=evaluation,
        minimum_overall_agreement=1,
        minimum_axis_agreement=1,
        minimum_source_stratum_agreement=1,
    )
    assert stage["status"] == "V1R_LOCKED_TEST_PASSED"
    return locked, parser_freeze, evaluation / "stage-status.json"


def _run_pipeline(tmp_path: Path, *, repeats_per_band: int) -> tuple[dict[str, object], Path]:
    source_build, packets, full_labels = _write_source_build(
        tmp_path, repeats_per_band=repeats_per_band
    )
    locked, parser_freeze, parser_validation = _prepare_and_evaluate_locked(
        tmp_path, source_build
    )
    original_contract = tmp_path / "original-v1-contract.json"
    original_contract.write_text(json.dumps(FROZEN_FEATURE_CONTRACT_V1), encoding="utf-8")
    contract_freeze = tmp_path / "v1r-contract-freeze.json"
    freeze_v1r_contract(
        workspace=Path(__file__).resolve().parents[1],
        source_build=source_build,
        original_v1_contract=original_contract,
        locked_set_preparation_manifest=locked / "locked-set-preparation-manifest.json",
        parser_freeze_manifest=parser_freeze,
        output=contract_freeze,
        expected_pair_count=repeats_per_band * 5,
        allow_dirty_for_dry_run=True,
    )
    full_classification = tmp_path / "full-classification"
    _classification_build(
        full_classification,
        source_build / "llm" / "blinded-packets.jsonl",
        full_labels,
    )
    feature_build = tmp_path / "feature-build"
    feature_stage = build_v1r_features(
        input_build=source_build,
        classification_build=full_classification,
        parser_validation_manifest=parser_validation,
        contract_freeze_manifest=contract_freeze,
        output=feature_build,
        allow_dry_run_contract=True,
    )
    assert feature_stage["outcome_stage_authorized"] is False
    feasibility = tmp_path / "feasibility"
    result = audit_v1r_feasibility(
        feature_build=feature_build,
        contract_freeze_manifest=contract_freeze,
        parser_validation_manifest=parser_validation,
        output=feasibility,
        allow_dry_run_contract=True,
    )
    return result, feasibility


def test_v1r_contract_is_source_only_extension_and_preserves_v1() -> None:
    assert FROZEN_FEATURE_CONTRACT_V1R["feature"] == FROZEN_FEATURE_CONTRACT_V1["feature"]
    assert FROZEN_FEATURE_CONTRACT_V1R["feature_bands"] == FROZEN_FEATURE_CONTRACT_V1[
        "feature_bands"
    ]
    original_sources = set(FROZEN_FEATURE_CONTRACT_V1["source_scope"]["included"])
    v1r_sources = set(FROZEN_FEATURE_CONTRACT_V1R["source_scope"]["included"])
    assert original_sources < v1r_sources
    assert "Arcana data-lake raw finance-comment HTML" in v1r_sources
    assert "Arcana data-lake raw finance-statement HTML" in v1r_sources
    assert FROZEN_FEATURE_CONTRACT_V1R["intended_freeze_tag"] != "future-eri-v1-preoutcome"


def test_v1r_source_stratified_locked_is_new_and_single_use(tmp_path: Path) -> None:
    source_build, _, _ = _write_source_build(tmp_path, repeats_per_band=1)
    locked, parser_freeze, parser_validation = _prepare_and_evaluate_locked(
        tmp_path, source_build
    )
    preparation = json.loads(
        (locked / "locked-set-preparation-manifest.json").read_text(encoding="utf-8")
    )
    assert preparation["selected_packet_count"] == 24
    assert preparation["v1_locked_rows_reused"] is False
    assert set(preparation["source_strata"]) == {
        "BUSINESS_INFO_EVIDENCE",
        "FINANCE_COMMENT_EVIDENCE",
        "FINANCE_STATEMENT_EVIDENCE",
        "MULTI_SOURCE_MOATRADER_OVERLAP_EVIDENCE",
    }
    evaluation = json.loads(parser_validation.read_text(encoding="utf-8"))
    assert evaluation["source_stratum_gate_passed"] is True
    quality = json.loads(
        (parser_validation.parent / "parser-quality-report-v1r.json").read_text(
            encoding="utf-8"
        )
    )
    assert all(
        row["source_span_grounding_rate"] == 1
        and "neutral_to_bullish_count" in row
        and "false_stable_count" in row
        for row in quality["by_axis_source_stratum"].values()
    )
    with pytest.raises(FileExistsError, match="already consumed"):
        evaluate_v1r_locked_parser(
            packet_input=locked / "v1r-locked-packets.jsonl",
            source_strata_input=locked / "v1r-locked-source-strata.jsonl",
            classification_build=tmp_path / "locked-classification",
            human_gold=tmp_path / "v1r-gold.csv",
            parser_freeze_manifest=parser_freeze,
            locked_consumption_record=tmp_path / "locked-consumption.json",
            output=tmp_path / "second-evaluation",
        )


def test_v1r_feasibility_passes_only_when_each_original_band_has_twenty(
    tmp_path: Path,
) -> None:
    result, feasibility = _run_pipeline(tmp_path, repeats_per_band=20)
    assert result["status"] == "V1R_FEASIBILITY_PASSED_ERI_MECHANISM_ELIGIBLE"
    assert result["six_axis_complete_features"] == 100
    assert result["all_five_bands_at_least_20"] is True
    assert result["outcome_stage_authorized"] is True
    report = json.loads(
        (feasibility / "v1r-feasibility-report.json").read_text(encoding="utf-8")
    )
    assert set(report["five_band_counts"].values()) == {20}
    assert report["outcomes_opened"] is False
    assert report["returns_opened"] is False
    assert report["value_data_opened"] is False
    assert (feasibility / "feature-seal.json").is_file()


def test_v1r_feasibility_tombstones_small_complete_case_sample(tmp_path: Path) -> None:
    result, feasibility = _run_pipeline(tmp_path, repeats_per_band=1)
    assert result["status"] == "V1R_FEASIBILITY_FAILED_COMPLETE_CASE_COVERAGE_COLLAPSE"
    assert result["six_axis_complete_features"] == 5
    assert result["all_five_bands_at_least_20"] is False
    assert result["outcome_stage_authorized"] is False
    manifest = json.loads(
        (feasibility / "pre-outcome-manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "V1R_FEASIBILITY_TOMBSTONED_COMPLETE_CASE_COLLAPSE"
    assert manifest["original_v1_tag_preserved"] is True
    assert manifest["per_pbr_role"] == "NOT_USED"
