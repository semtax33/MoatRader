from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from moatrader.expectations.future_eri import EvidenceState, OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
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
from moatrader.expectations.historical_evidence_v2 import (
    AbstentionReasonV2,
    AxisApplicabilityV2,
    AxisEvidenceProvenanceV2,
    AxisSignedScoreRoleV2,
    GroundedAxisStateSnapshotV2,
    PITApplicabilityRulesV2,
    PITOperatingSnapshotV2,
    PreviousEvidenceBasisV2,
    SparseAxisAvailabilityV2,
    SparseAxisEvidenceV2,
    SparseCoverageGatePolicyV2,
    build_deterministic_pit_axis_evidence,
    build_last_grounded_axis_evidence,
    build_sparse_feature_row_v2,
    calibrate_sparse_band_contract_v2,
    evaluate_sparse_coverage_gate_v2,
    merge_axis_evidence_v2,
    sparse_band_diagnostics_v2,
    sparse_feature_coverage_report,
)
from scripts.audit_historical_evidence_abstentions_v2 import (
    prepare_abstention_audit,
    validate_abstention_audit,
)
from scripts.build_historical_sparse_features_v2 import (
    AxisApplicabilityDecisionInputV2,
    DeterministicAxisEvidenceInputV2,
    build_sparse_features,
)
from scripts.calibrate_historical_sparse_features_v2 import calibrate_sparse_features
from scripts.evaluate_historical_evidence_parser_v2 import (
    combine_v2_locked_evaluations,
    create_v2_parser_freeze,
    evaluate_v2_locked_parser,
)
from scripts.prepare_historical_evidence_classification_subset import prepare_subset
from scripts.prepare_historical_locked_sets_v2 import (
    finalize_locked_sets,
    prepare_locked_candidates,
)
from scripts.materialize_historical_human_gold_v2 import materialize_human_gold
from scripts.prepare_historical_deterministic_pit_inputs_v2 import (
    FilingSource,
    FilingTask,
    _extract_task,
    extract_pit_metrics_from_html,
)
from scripts.prepare_historical_semantic_packets_v2 import prepare_semantic_packets


D = Decimal
SEOUL = ZoneInfo("Asia/Seoul")


def _filing(ticker: str, rcept_no: str, period: date) -> HistoricalRegularFiling:
    timestamp = datetime.strptime(rcept_no[:8], "%Y%m%d").replace(
        hour=16,
        tzinfo=SEOUL,
    )
    return HistoricalRegularFiling(
        ticker=ticker,
        issuer_name="테스트",
        rcept_no=rcept_no,
        report_name="정기보고서",
        report_code={3: "11013", 6: "11012", 9: "11014", 12: "11011"}[period.month],
        fiscal_period_end=period,
        published_at=timestamp,
        available_at=timestamp,
        signal_timestamp=timestamp,
        source_variants=[
            HistoricalSourceVariant(
                origin=HistoricalSourceOrigin.ARCANA_BUSINESS_HTML,
                path=f"readonly/{rcept_no}.html",
                raw_sha256=hashlib.sha256(rcept_no.encode()).hexdigest(),
                byte_count=10,
                receipt_linkage=ReceiptLinkage.EXACT_METADATA,
            )
        ],
    )


def _pair(index: int = 1) -> HistoricalFilingPair:
    ticker = f"{index:06d}"
    previous = _filing(ticker, f"20200330{index:06d}", date(2020, 3, 31))
    current = _filing(ticker, f"20200814{index:06d}", date(2020, 6, 30))
    return HistoricalFilingPair(
        pair_id=historical_pair_id(ticker, previous.rcept_no, current.rcept_no),
        ticker=ticker,
        previous=previous,
        current=current,
    )


def _packet(pair: HistoricalFilingPair, axis: OperatingEvidenceAxis, *, both: bool = True) -> PairedAxisPacket:
    axis_index = list(OperatingEvidenceAxis).index(axis)
    return PairedAxisPacket(
        packet_id=packet_id(pair.pair_id, axis),
        axis=axis,
        previous_excerpts=[
            BlindedExcerpt(source_id=f"SRC_{axis_index * 2 + 1:020x}", text="이전 상태")
        ],
        current_excerpts=(
            [BlindedExcerpt(source_id=f"SRC_{axis_index * 2 + 2:020x}", text="현재 개선")]
            if both
            else []
        ),
    )


def _grounded(
    axis: OperatingEvidenceAxis,
    direction: EvidenceState,
    *,
    pair: HistoricalFilingPair,
) -> SparseAxisEvidenceV2:
    return SparseAxisEvidenceV2(
        axis=axis,
        applicability=AxisApplicabilityV2.APPLICABLE,
        availability=SparseAxisAvailabilityV2.GROUNDED,
        direction=direction,
        provenance=AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC,
        previous_evidence_basis=PreviousEvidenceBasisV2.IMMEDIATE_PREVIOUS_FILING,
        confidence=D(1),
        source_ids=[f"SOURCE_{axis.value}"],
        previous_evidence_at=pair.previous.available_at,
        current_evidence_at=pair.current.available_at,
        applicability_rule_id="TEST_PIT_RULE",
        signed_score_role=(
            AxisSignedScoreRoleV2.RAW_DIRECTION_ONLY
            if axis == OperatingEvidenceAxis.CAPACITY_CAPEX
            else AxisSignedScoreRoleV2.PRIMARY_SIGNED_SCORE
        ),
    )


def _na(axis: OperatingEvidenceAxis) -> SparseAxisEvidenceV2:
    return SparseAxisEvidenceV2(
        axis=axis,
        applicability=AxisApplicabilityV2.APPLICABLE,
        availability=SparseAxisAvailabilityV2.NA,
        abstention_reason=AbstentionReasonV2.TRUE_NO_MENTION,
        applicability_rule_id="TEST_APPLICABLE",
        signed_score_role=(
            AxisSignedScoreRoleV2.RAW_DIRECTION_ONLY
            if axis == OperatingEvidenceAxis.CAPACITY_CAPEX
            else AxisSignedScoreRoleV2.PRIMARY_SIGNED_SCORE
        ),
    )


def _not_applicable(axis: OperatingEvidenceAxis) -> SparseAxisEvidenceV2:
    return SparseAxisEvidenceV2(
        axis=axis,
        applicability=AxisApplicabilityV2.NOT_APPLICABLE,
        availability=SparseAxisAvailabilityV2.NOT_APPLICABLE,
        applicability_rule_id="TEST_NOT_APPLICABLE",
        signed_score_role=(
            AxisSignedScoreRoleV2.RAW_DIRECTION_ONLY
            if axis == OperatingEvidenceAxis.CAPACITY_CAPEX
            else AxisSignedScoreRoleV2.PRIMARY_SIGNED_SCORE
        ),
    )


def test_sparse_contract_keeps_na_and_not_applicable_distinct_from_neutral() -> None:
    pair = _pair()
    evidence = [
        _grounded(OperatingEvidenceAxis.DEMAND, EvidenceState.IMPROVING, pair=pair),
        _grounded(OperatingEvidenceAxis.PRICE_MIX, EvidenceState.STABLE, pair=pair),
        _na(OperatingEvidenceAxis.BACKLOG),
        _grounded(OperatingEvidenceAxis.MARGIN, EvidenceState.WEAKENING, pair=pair),
        _not_applicable(OperatingEvidenceAxis.INVENTORY_MISMATCH),
        _grounded(OperatingEvidenceAxis.CAPACITY_CAPEX, EvidenceState.IMPROVING, pair=pair),
    ]

    row = build_sparse_feature_row_v2(pair=pair, axis_evidence=evidence)

    assert row.observed_axis_count == 4
    assert row.applicable_axis_count == 5
    assert row.unavailable_axis_count == 1
    assert row.not_applicable_axis_count == 1
    assert row.signed_score_axis_count == 3
    assert row.raw_direction_only_axis_count == 1
    assert row.neutral_axis_count == 1
    assert row.signed_breadth == D("0")
    assert row.coverage == D("0.8")
    with pytest.raises(ValueError, match="NA axis cannot contain"):
        SparseAxisEvidenceV2(
            axis=OperatingEvidenceAxis.BACKLOG,
            applicability=AxisApplicabilityV2.APPLICABLE,
            availability=SparseAxisAvailabilityV2.NA,
            direction=EvidenceState.STABLE,
            abstention_reason=AbstentionReasonV2.TRUE_NO_MENTION,
            applicability_rule_id="BAD_ZERO_IMPUTATION",
        )


def _pit_snapshot(
    *,
    current: bool,
    sources: bool = True,
) -> PITOperatingSnapshotV2:
    suffix = "CURRENT" if current else "PREVIOUS"
    return PITOperatingSnapshotV2(
        issuer_id="000001",
        fiscal_period_end=date(2020, 6 if current else 3, 30 if current else 31),
        available_at=datetime(2020, 8 if current else 3, 14 if current else 30, 16, tzinfo=SEOUL),
        source_ids=(
            {
                metric: [f"{suffix}_{metric}"]
                for metric in (
                    "revenue",
                    "operating_profit",
                    "inventory",
                    "assets",
                    "backlog",
                    "capex",
                    "ppe",
                )
            }
            if sources
            else {}
        ),
        revenue=D(120 if current else 100),
        operating_profit=D(18 if current else 10),
        inventory=D(15 if current else 10),
        assets=D(200),
        backlog=D(130 if current else 100),
        capex=D(9 if current else 5),
        ppe=D(80),
        backlog_disclosed=True,
        capacity_disclosed=True,
    )


def test_deterministic_pit_axes_have_priority_ready_directions() -> None:
    evidence = build_deterministic_pit_axis_evidence(
        previous=_pit_snapshot(current=False),
        current=_pit_snapshot(current=True),
        rules=PITApplicabilityRulesV2(),
    )

    assert evidence[OperatingEvidenceAxis.MARGIN].direction == EvidenceState.IMPROVING
    assert evidence[OperatingEvidenceAxis.INVENTORY_MISMATCH].direction == EvidenceState.WEAKENING
    assert evidence[OperatingEvidenceAxis.BACKLOG].direction == EvidenceState.IMPROVING
    assert evidence[OperatingEvidenceAxis.CAPACITY_CAPEX].direction == EvidenceState.IMPROVING
    assert evidence[OperatingEvidenceAxis.MARGIN].provenance == (
        AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC
    )
    assert evidence[OperatingEvidenceAxis.INVENTORY_MISMATCH].provenance == (
        AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC
    )
    assert evidence[OperatingEvidenceAxis.BACKLOG].provenance == (
        AxisEvidenceProvenanceV2.STRUCTURED_TABLE
    )
    assert evidence[OperatingEvidenceAxis.CAPACITY_CAPEX].provenance == (
        AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC
    )
    assert evidence[OperatingEvidenceAxis.CAPACITY_CAPEX].signed_score_role == (
        AxisSignedScoreRoleV2.RAW_DIRECTION_ONLY
    )


def test_capex_axis_uses_net_ppe_intensity_fallback_as_raw_direction() -> None:
    previous = _pit_snapshot(current=False).model_copy(
        update={
            "capex": None,
            "ppe": D(50),
            "source_ids": {
                key: value
                for key, value in _pit_snapshot(current=False).source_ids.items()
                if key != "capex"
            },
        }
    )
    current = _pit_snapshot(current=True).model_copy(
        update={
            "capex": None,
            "ppe": D(80),
            "source_ids": {
                key: value
                for key, value in _pit_snapshot(current=True).source_ids.items()
                if key != "capex"
            },
        }
    )

    capex = build_deterministic_pit_axis_evidence(
        previous=previous,
        current=current,
        rules=PITApplicabilityRulesV2(),
    )[OperatingEvidenceAxis.CAPACITY_CAPEX]

    assert capex.availability == SparseAxisAvailabilityV2.GROUNDED
    assert capex.deterministic_metric_name == "NET_PPE_TO_ASSETS"
    assert capex.direction == EvidenceState.IMPROVING
    assert capex.signed_score_role == AxisSignedScoreRoleV2.RAW_DIRECTION_ONLY


def test_pit_availability_order_violation_is_na_not_lookahead() -> None:
    previous = _pit_snapshot(current=False).model_copy(
        update={"available_at": datetime(2020, 8, 20, 16, tzinfo=SEOUL)}
    )
    current = _pit_snapshot(current=True)

    evidence = build_deterministic_pit_axis_evidence(
        previous=previous,
        current=current,
        rules=PITApplicabilityRulesV2(),
    )

    assert set(evidence) == {
        OperatingEvidenceAxis.MARGIN,
        OperatingEvidenceAxis.INVENTORY_MISMATCH,
        OperatingEvidenceAxis.BACKLOG,
        OperatingEvidenceAxis.CAPACITY_CAPEX,
    }
    assert all(
        row.availability == SparseAxisAvailabilityV2.NA
        and row.abstention_reason == AbstentionReasonV2.PERIOD_MISMATCH
        and row.direction is None
        for row in evidence.values()
    )


def test_pit_html_extractor_uses_current_cumulative_values_and_structured_backlog() -> None:
    document = """
    <html><body>
      <p>(단위 : 백만원)</p>
      <table><thead><tr><th>계정</th><th>당기 3개월</th><th>당기 누적</th><th>전기 3개월</th><th>전기 누적</th></tr></thead>
        <tbody>
          <tr><td>매출액</td><td>100</td><td>200</td><td>80</td><td>160</td></tr>
          <tr><td>영업이익(손실)</td><td>10</td><td>20</td><td>8</td><td>16</td></tr>
        </tbody>
      </table>
      <p>(단위 : 백만원)</p>
      <table><tbody>
        <tr><td>재고자산</td><td>50</td><td>45</td></tr>
        <tr><td>유형자산</td><td>200</td><td>180</td></tr>
        <tr><td>자산총계</td><td>500</td><td>450</td></tr>
      </tbody></table>
      <p>(단위 : 백만원)</p>
      <table><tbody>
        <tr><td>유형자산의 취득</td><td>(30)</td><td>(25)</td></tr>
        <tr><td>무형자산의 취득</td><td>(5)</td><td>(4)</td></tr>
      </tbody></table>
      <p>생산설비 증설</p>
      <p>(단위 : 백만원)</p>
      <table><thead><tr><th>공사명</th><th>계약잔액</th></tr></thead>
        <tbody>
          <tr><td>A</td><td>30</td></tr><tr><td>B</td><td>40</td></tr>
          <tr><td>합계</td><td>70</td></tr>
        </tbody>
      </table>
    </body></html>
    """

    metrics = extract_pit_metrics_from_html(
        document,
        fiscal_period_end=date(2024, 6, 30),
    )

    assert metrics["revenue"] == D(200_000_000)
    assert metrics["operating_profit"] == D(20_000_000)
    assert metrics["inventory"] == D(50_000_000)
    assert metrics["assets"] == D(500_000_000)
    assert metrics["ppe"] == D(200_000_000)
    assert metrics["capex"] == D(35_000_000)
    assert metrics["backlog"] == D(70_000_000)
    assert metrics["backlog_disclosed"] is True
    assert metrics["capacity_disclosed"] is True

    embedded_unit_metrics = extract_pit_metrics_from_html(
        """
        <html><body><table>
          <tr><td>단위 : 억원</td></tr>
          <tr><td>매출액</td><td>3</td></tr>
          <tr><td>영업이익</td><td>1</td></tr>
        </table></body></html>
        """,
        fiscal_period_end=date(2024, 12, 31),
    )
    assert embedded_unit_metrics["revenue"] == D(300_000_000)
    assert embedded_unit_metrics["operating_profit"] == D(100_000_000)


def test_pit_filing_task_reads_all_arcana_sections_and_moatrader_original(
    tmp_path: Path,
) -> None:
    documents = {
        "finance-statement.html": "<html><body><p>재무제표 본문</p></body></html>",
        "finance-comment.html": """
            <html><body><table>
              <tr><td>매출액</td><td>100</td></tr>
              <tr><td>영업이익</td><td>10</td></tr>
              <tr><td>재고자산</td><td>20</td></tr>
              <tr><td>자산총계</td><td>200</td></tr>
            </table></body></html>
        """,
        "business-info.html": """
            <html><body><table>
              <tr><th>공사명</th><th>수주잔고</th></tr>
              <tr><td>합계</td><td>80</td></tr>
            </table></body></html>
        """,
    }
    paths: dict[str, Path] = {}
    for name, document in documents.items():
        path = tmp_path / name
        path.write_text(document, encoding="utf-8")
        paths[name] = path
    archive_path = tmp_path / "original.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "report.xml",
            """
            <html><body><table>
              <tr><td>유형자산의 취득</td><td>(30)</td></tr>
            </table></body></html>
            """,
        )

    def source(name: str, origin: str) -> FilingSource:
        path = archive_path if name == "original.zip" else paths[name]
        digest = sha256_file(path)
        return FilingSource(
            source_id=f"PIT_SRC_{digest[:20]}",
            origin=origin,
            path=str(path),
            raw_sha256=digest,
        )

    result = _extract_task(
        FilingTask(
            ticker="000001",
            rcept_no="20240515000001",
            fiscal_period_end="2024-03-31",
            available_at="2024-05-15T16:00:00+09:00",
            finance_statement=source(
                "finance-statement.html", "ARCANA_FINANCE_STATEMENT_HTML"
            ),
            finance_comment=source(
                "finance-comment.html", "ARCANA_FINANCE_COMMENT_HTML"
            ),
            business_info=source("business-info.html", "ARCANA_BUSINESS_HTML"),
            moatrader_original=source("original.zip", "MOATRADER_OPENDART_ARCHIVE"),
        )
    )

    assert len(result["verified_hashes"]) == 4
    assert result["origins"]["revenue"] == "ARCANA_FINANCE_COMMENT_HTML"
    assert result["origins"]["backlog"] == "ARCANA_BUSINESS_HTML"
    assert result["origins"]["capex"] == "MOATRADER_OPENDART_ARCHIVE"
    assert result["snapshot"]["capacity_disclosed"] is True


def test_last_grounded_respects_frozen_staleness_window() -> None:
    current = GroundedAxisStateSnapshotV2(
        axis=OperatingEvidenceAxis.DEMAND,
        state=EvidenceState.IMPROVING,
        fiscal_period_end=date(2021, 6, 30),
        available_at=datetime(2021, 8, 15, tzinfo=SEOUL),
        source_ids=["CURRENT"],
    )
    recent = GroundedAxisStateSnapshotV2(
        axis=OperatingEvidenceAxis.DEMAND,
        state=EvidenceState.STABLE,
        fiscal_period_end=date(2020, 6, 30),
        available_at=current.available_at - timedelta(days=449),
        source_ids=["PRIOR"],
    )
    stale = recent.model_copy(
        update={"available_at": current.available_at - timedelta(days=451)}
    )

    grounded = build_last_grounded_axis_evidence(
        current=current,
        history=[recent],
        staleness_limit_days=450,
        applicability_rule_id="LAST_GROUNDED_450D",
    )
    unavailable = build_last_grounded_axis_evidence(
        current=current,
        history=[stale],
        staleness_limit_days=450,
        applicability_rule_id="LAST_GROUNDED_450D",
    )

    assert grounded.provenance == AxisEvidenceProvenanceV2.LLM_NARRATIVE
    assert grounded.previous_evidence_basis == (
        PreviousEvidenceBasisV2.LAST_GROUNDED_WITHIN_STALENESS
    )
    assert grounded.prior_age_days == 449
    assert unavailable.availability == SparseAxisAvailabilityV2.NA
    assert unavailable.abstention_reason == AbstentionReasonV2.STALE_PRIOR_STATE


def test_last_grounded_never_replaces_missing_current_evidence() -> None:
    history = [
        GroundedAxisStateSnapshotV2(
            axis=OperatingEvidenceAxis.DEMAND,
            state=EvidenceState.IMPROVING,
            fiscal_period_end=date(2023, 3, 31),
            available_at=datetime(2023, 5, 15, tzinfo=SEOUL),
            source_ids=["2023Q1"],
        )
    ]

    result = build_last_grounded_axis_evidence(
        current=None,
        history=history,
        staleness_limit_days=450,
        applicability_rule_id="LAST_GROUNDED_450D",
        axis=OperatingEvidenceAxis.DEMAND,
    )

    assert result.availability == SparseAxisAvailabilityV2.NA
    assert result.direction is None
    assert result.abstention_reason == AbstentionReasonV2.TRUE_NO_MENTION


def test_numeric_beats_table_and_llm_without_averaging() -> None:
    pair = _pair()
    llm = _grounded(OperatingEvidenceAxis.MARGIN, EvidenceState.STABLE, pair=pair).model_copy(
        update={"provenance": AxisEvidenceProvenanceV2.LLM_NARRATIVE}
    )
    numeric = _grounded(
        OperatingEvidenceAxis.MARGIN,
        EvidenceState.WEAKENING,
        pair=pair,
    )

    merged = merge_axis_evidence_v2(llm, numeric)

    assert merged.direction == EvidenceState.WEAKENING
    assert merged.provenance == AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC


def _breadth_row(index: int, directions: tuple[EvidenceState, EvidenceState]):
    pair = _pair(index)
    axes = list(OperatingEvidenceAxis)
    evidence = [
        _grounded(axes[0], directions[0], pair=pair),
        _grounded(axes[1], directions[1], pair=pair),
        *[_na(axis) for axis in axes[2:]],
    ]
    return build_sparse_feature_row_v2(pair=pair, axis_evidence=evidence)


def test_feature_only_band_calibration_requires_explicit_nobs_and_covers_five_bands() -> None:
    rows = [
        _breadth_row(1, (EvidenceState.WEAKENING, EvidenceState.WEAKENING)),
        _breadth_row(2, (EvidenceState.WEAKENING, EvidenceState.STABLE)),
        _breadth_row(3, (EvidenceState.STABLE, EvidenceState.STABLE)),
        _breadth_row(4, (EvidenceState.IMPROVING, EvidenceState.STABLE)),
        _breadth_row(5, (EvidenceState.IMPROVING, EvidenceState.IMPROVING)),
    ]
    contract = calibrate_sparse_band_contract_v2(
        rows,
        minimum_observed_axes=2,
        minimum_rows_per_band=1,
    )
    report = sparse_feature_coverage_report(rows)
    diagnostics = sparse_band_diagnostics_v2(rows, band_contract=contract)
    gate = evaluate_sparse_coverage_gate_v2(
        diagnostics,
        policy=SparseCoverageGatePolicyV2(
            minimum_rows_per_band=1,
            minimum_unique_issuers_per_band=1,
            minimum_unique_signal_months_per_band=1,
            minimum_total_unique_issuers=5,
            minimum_total_unique_signal_months=1,
            maximum_top_issuer_share_per_band=D(1),
            maximum_top_year_share_per_band=D(1),
            maximum_top_evidence_source_share_per_band=D(1),
        ),
    )
    concentrated_gate = evaluate_sparse_coverage_gate_v2(
        diagnostics,
        policy=SparseCoverageGatePolicyV2(
            minimum_rows_per_band=1,
            minimum_unique_issuers_per_band=1,
            minimum_unique_signal_months_per_band=1,
            minimum_total_unique_issuers=5,
            minimum_total_unique_signal_months=1,
            maximum_top_issuer_share_per_band=D("0.5"),
            maximum_top_year_share_per_band=D(1),
            maximum_top_evidence_source_share_per_band=D(1),
        ),
    )

    assert contract.all_bands_sufficient is True
    assert set(contract.band_counts.values()) == {1}
    assert all(D(-1) < value < D(1) for value in contract.cut_points)
    assert report["grounded_axis_count_histogram"]["2"] == 5
    assert len(report["signed_breadth_distribution"]) == 5
    assert report["n_directional_histogram"]["0"] == 1
    assert report["by_exact_nobs"]["2"]["row_count"] == 5
    assert len(report["by_exact_nobs"]["2"]["observations"]) == 5
    assert report["by_exact_nobs"]["2"]["observations"][0]["directional_ratio"] is not None
    assert diagnostics["corr_abs_signed_breadth_nobs"] is None
    assert set(diagnostics["by_band"]) == {band.value for band in contract.band_counts}
    assert gate["gate_passed"] is True
    assert concentrated_gate["gate_passed"] is False


def test_axis_available_subset_does_not_require_six_axis_complete_pair(tmp_path: Path) -> None:
    pair = _pair()
    packets = [
        _packet(pair, axis, both=axis != OperatingEvidenceAxis.BACKLOG)
        for axis in OperatingEvidenceAxis
    ]
    packet_input = tmp_path / "all.jsonl"
    packet_input.write_text(
        "".join(item.model_dump_json() + "\n" for item in packets),
        encoding="utf-8",
    )
    output = tmp_path / "axis-available.jsonl"

    manifest = prepare_subset(
        packet_input=packet_input,
        output=output,
        mode="AXIS_AVAILABLE",
        expected_candidate_pairs=1,
    )

    assert manifest["total_pair_count"] == 1
    assert manifest["selected_pair_count"] == 1
    assert manifest["selected_packet_count"] == 5
    assert manifest["candidate_grounded_axis_count_histogram"]["5"] == 1


def test_semantic_selection_keeps_only_demand_and_price_mix(
    tmp_path: Path,
) -> None:
    pair = _pair()
    packets = [_packet(pair, axis) for axis in OperatingEvidenceAxis]
    pair_path = tmp_path / "pairs.jsonl"
    packet_path = tmp_path / "packets.jsonl"
    pair_path.write_text(pair.model_dump_json() + "\n", encoding="utf-8")
    packet_path.write_text(
        "".join(packet.model_dump_json() + "\n" for packet in packets),
        encoding="utf-8",
    )
    deterministic: list[DeterministicAxisEvidenceInputV2] = []
    for axis in (
        OperatingEvidenceAxis.MARGIN,
        OperatingEvidenceAxis.INVENTORY_MISMATCH,
        OperatingEvidenceAxis.BACKLOG,
        OperatingEvidenceAxis.CAPACITY_CAPEX,
    ):
        if axis == OperatingEvidenceAxis.MARGIN:
            evidence = _grounded(axis, EvidenceState.WEAKENING, pair=pair)
        elif axis == OperatingEvidenceAxis.BACKLOG:
            evidence = _not_applicable(axis)
        else:
            evidence = _na(axis)
        deterministic.append(
            DeterministicAxisEvidenceInputV2(pair_id=pair.pair_id, evidence=evidence)
        )
    applicability = [
        AxisApplicabilityDecisionInputV2(
            pair_id=pair.pair_id,
            axis=axis,
            applicability=(
                AxisApplicabilityV2.NOT_APPLICABLE
                if axis == OperatingEvidenceAxis.BACKLOG
                else AxisApplicabilityV2.APPLICABLE
            ),
            rule_id="TEST_PIT_APPLICABILITY",
        )
        for axis in OperatingEvidenceAxis
    ]
    deterministic_path = tmp_path / "deterministic.jsonl"
    applicability_path = tmp_path / "applicability.jsonl"
    deterministic_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in deterministic), encoding="utf-8"
    )
    applicability_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in applicability), encoding="utf-8"
    )
    output = tmp_path / "semantic.jsonl"

    manifest = prepare_semantic_packets(
        filing_pair_input=pair_path,
        packet_input=packet_path,
        deterministic_evidence_input=deterministic_path,
        applicability_input=applicability_path,
        output=output,
        expected_pair_count=1,
    )
    selected = [
        PairedAxisPacket.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert {packet.axis for packet in selected} == {
        OperatingEvidenceAxis.DEMAND,
        OperatingEvidenceAxis.PRICE_MIX,
    }
    assert manifest["selected_packet_count"] == 2
    assert manifest["capacity_narrative_fallback_enabled"] is False
    assert manifest["qualitative_diagnostics_for_numeric_axes"] is False


def test_sparse_builder_retains_all_pairs_and_unclassified_axes_as_na(tmp_path: Path) -> None:
    pair = _pair()
    input_build = tmp_path / "input"
    (input_build / "private").mkdir(parents=True)
    (input_build / "llm").mkdir()
    (input_build / "private" / "filing-pairs.jsonl").write_text(
        pair.model_dump_json() + "\n", encoding="utf-8"
    )
    packets = [_packet(pair, axis) for axis in OperatingEvidenceAxis]
    (input_build / "llm" / "blinded-packets.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in packets), encoding="utf-8"
    )
    demand = packets[0]
    classification = AxisPairClassification(
        packet_id=demand.packet_id,
        axis=demand.axis,
        previous_state=EvidenceState.STABLE,
        current_state=EvidenceState.IMPROVING,
        previous_source_id=demand.previous_excerpts[0].source_id,
        current_source_id=demand.current_excerpts[0].source_id,
        previous_source_span=demand.previous_excerpts[0].text,
        current_source_span=demand.current_excerpts[0].text,
        confidence=1,
    )
    classification_build = tmp_path / "classification"
    classification_build.mkdir()
    (classification_build / "classifications.jsonl").write_text(
        classification.model_dump_json() + "\n", encoding="utf-8"
    )
    (classification_build / "stage-status.json").write_text(
        json.dumps({"status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE"}),
        encoding="utf-8",
    )

    result = build_sparse_features(
        input_build=input_build,
        classification_build=classification_build,
        output=tmp_path / "features",
        coverage_only_unvalidated=True,
        allow_test_input_without_source_audit=True,
    )
    stored = json.loads(
        (tmp_path / "features" / "sparse-features-all-pairs.jsonl").read_text(
            encoding="utf-8"
        )
    )

    assert result["pair_count"] == 1
    assert stored["observed_axis_count"] == 1
    assert stored["unavailable_axis_count"] == 5
    assert stored["signed_breadth"] == "1"
    assert result["outcome_stage_authorized"] is False


def test_calibration_diagnostics_never_auto_select_nobs(tmp_path: Path) -> None:
    rows = [
        _breadth_row(index, directions)
        for index, directions in enumerate(
            (
                (EvidenceState.WEAKENING, EvidenceState.WEAKENING),
                (EvidenceState.WEAKENING, EvidenceState.STABLE),
                (EvidenceState.STABLE, EvidenceState.STABLE),
                (EvidenceState.IMPROVING, EvidenceState.STABLE),
                (EvidenceState.IMPROVING, EvidenceState.IMPROVING),
            ),
            start=1,
        )
    ]
    feature_build = tmp_path / "feature-build"
    feature_build.mkdir()
    (feature_build / "sparse-features-all-pairs.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    (feature_build / "stage-status.json").write_text(
        json.dumps(
            {
                "status": "SPARSE_FEATURES_BUILT_AWAITING_OUTCOME_BLIND_CALIBRATION",
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )

    status = calibrate_sparse_features(
        feature_build=feature_build,
        output=tmp_path / "diagnostics",
    )

    assert status["minimum_observed_axes"] is None
    assert status["minimum_observed_axes_auto_selected"] is False
    assert status["outcome_stage_authorized"] is False


def test_sparse_freeze_opens_gate_only_after_all_v2_manifests_pass(tmp_path: Path) -> None:
    rows = [
        _breadth_row(index, directions)
        for index, directions in enumerate(
            (
                (EvidenceState.WEAKENING, EvidenceState.WEAKENING),
                (EvidenceState.WEAKENING, EvidenceState.STABLE),
                (EvidenceState.STABLE, EvidenceState.STABLE),
                (EvidenceState.IMPROVING, EvidenceState.STABLE),
                (EvidenceState.IMPROVING, EvidenceState.IMPROVING),
            ),
            start=1,
        )
    ]
    parser_manifest = tmp_path / "parser-v2.json"
    parser_manifest.write_text(
        json.dumps(
            {
                "status": "V2_LOCKED_TESTS_PASSED",
                "natural_frequency_gate_passed": True,
                "directional_strata_gate_passed": True,
                "parser_freeze_sha256": "f" * 64,
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    audit_manifest = tmp_path / "audit-v2.json"
    audit_manifest.write_text(
        json.dumps(
            {
                "status": "V2_ABSTENTION_AUDIT_PASSED",
                "upstream_extraction_gate_passed": True,
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    contract_manifest = tmp_path / "contract-v2.json"
    contract_manifest.write_text(
        json.dumps(
            {
                "status": "V2_PRE_OUTCOME_CONTRACT_FROZEN",
                "contract_tag": "v2-fixture",
                "git_commit": "a" * 40,
                "worktree_dirty": False,
                "dry_run_only": False,
                "parser_freeze_sha256": "f" * 64,
                "feature_policy_sha256": "1" * 64,
                "applicability_policy_sha256": "2" * 64,
                "deterministic_axis_policy_sha256": "3" * 64,
                "evidence_priority_sha256": "4" * 64,
                "parser_prompt_sha256": "5" * 64,
                "signal_timestamp_policy": (
                    "FIRST_TRADABLE_TIMESTAMP_AFTER_CURRENT_REGULAR_FILING_AVAILABLE_AT"
                ),
                "last_grounded_days": 450,
                "last_grounded_role": (
                    "PREVIOUS_COMPARISON_BASIS_ONLY_NEVER_CURRENT_EVIDENCE"
                ),
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    coverage_policy = tmp_path / "coverage-policy.json"
    coverage_policy.write_text(
        SparseCoverageGatePolicyV2(
            minimum_rows_per_band=1,
            minimum_unique_issuers_per_band=1,
            minimum_unique_signal_months_per_band=1,
            minimum_total_unique_issuers=5,
            minimum_total_unique_signal_months=1,
            maximum_top_issuer_share_per_band=D(1),
            maximum_top_year_share_per_band=D(1),
            maximum_top_evidence_source_share_per_band=D(1),
        ).model_dump_json(),
        encoding="utf-8",
    )
    feature_build = tmp_path / "freeze-feature-build"
    feature_build.mkdir()
    (feature_build / "sparse-features-all-pairs.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    (feature_build / "stage-status.json").write_text(
        json.dumps(
            {
                "status": "SPARSE_FEATURES_BUILT_AWAITING_OUTCOME_BLIND_CALIBRATION",
                "outcome_vault_opened": False,
                "return_data_opened": False,
                "applicability_contract_complete": True,
                "deterministic_pit_priority_applied": True,
                "input_hashes": {
                    "parser_validation_manifest": sha256_file(parser_manifest),
                    "contract_freeze_manifest": sha256_file(contract_manifest),
                },
            }
        ),
        encoding="utf-8",
    )

    result = calibrate_sparse_features(
        feature_build=feature_build,
        output=tmp_path / "frozen",
        freeze=True,
        minimum_observed_axes=2,
        parser_validation_manifest=parser_manifest,
        abstention_audit_manifest=audit_manifest,
        contract_freeze_manifest=contract_manifest,
        coverage_gate_policy=coverage_policy,
    )

    assert result["status"] == "V2_FEATURE_ONLY_CALIBRATION_SEALED_OUTCOMES_CLOSED"
    assert result["outcome_stage_authorized"] is False
    assert result["all_five_bands_sufficient"] is True
    assert (tmp_path / "frozen" / "sparse-feature-seal.json").is_file()
    pre_outcome = json.loads(
        (tmp_path / "frozen" / "pre-outcome-manifest.json").read_text(encoding="utf-8")
    )
    assert pre_outcome["min_nobs"] == 2
    assert pre_outcome["coverage_gate"]["passed"] is True
    assert pre_outcome["last_grounded_days"] == 450
    assert pre_outcome["outcome_stage_authorized"] is False


def _locked_packet(index: int, axis: OperatingEvidenceAxis) -> PairedAxisPacket:
    return PairedAxisPacket(
        packet_id=f"PKT_{index:024x}",
        axis=axis,
        previous_excerpts=[BlindedExcerpt(source_id=f"SRC_{index * 2:020x}", text="이전")],
        current_excerpts=[
            BlindedExcerpt(source_id=f"SRC_{index * 2 + 1:020x}", text="현재")
        ],
    )


def test_v2_dual_locked_sets_are_independent_balanced_and_single_use(tmp_path: Path) -> None:
    balanced_packets: list[PairedAxisPacket] = []
    balanced_labels: list[AxisPairClassification] = []
    natural_packets: list[PairedAxisPacket] = []
    natural_labels: list[AxisPairClassification] = []
    gold_rows: list[dict[str, str]] = []

    def add_case(
        *,
        index: int,
        axis: OperatingEvidenceAxis,
        status: str,
        previous_state: int | None,
        current_state: int | None,
        split: str,
        contract: str,
        packets: list[PairedAxisPacket],
        labels: list[AxisPairClassification],
    ) -> None:
        packet = _locked_packet(index, axis)
        packets.append(packet)
        payload = {
            "packet_id": packet.packet_id,
            "axis": axis,
            "status": status,
            "confidence": 1,
        }
        row = {
            "packet_id": packet.packet_id,
            "axis": axis.value,
            "human_status": status,
            "human_previous_state": "",
            "human_current_state": "",
            "human_previous_source_id": "",
            "human_current_source_id": "",
            "human_previous_source_span": "",
            "human_current_source_span": "",
            "gold_split": split,
            "gold_contract_version": contract,
            "reviewer": "HUMAN",
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
            row.update(
                human_previous_state=str(previous_state),
                human_current_state=str(current_state),
                human_previous_source_id=packet.previous_excerpts[0].source_id,
                human_current_source_id=packet.current_excerpts[0].source_id,
                human_previous_source_span=packet.previous_excerpts[0].text,
                human_current_source_span=packet.current_excerpts[0].text,
            )
        labels.append(AxisPairClassification.model_validate(payload))
        gold_rows.append(row)

    index = 1
    semantic_axes = (
        OperatingEvidenceAxis.DEMAND,
        OperatingEvidenceAxis.PRICE_MIX,
    )
    for axis in semantic_axes:
        for status, previous_state, current_state in (
            ("COMPLETE", 1, 0),
            ("COMPLETE", 0, 0),
            ("COMPLETE", 0, 1),
            ("INSUFFICIENT_EVIDENCE", None, None),
            ("AMBIGUOUS", None, None),
        ):
            add_case(
                index=index,
                axis=axis,
                status=status,
                previous_state=previous_state,
                current_state=current_state,
                split="V2_BALANCED_LOCKED_TEST",
                contract="V2_DIRECTIONAL_BALANCED_LOCKED",
                packets=balanced_packets,
                labels=balanced_labels,
            )
            index += 1
    for offset, axis in enumerate(semantic_axes, start=1000):
        add_case(
            index=offset,
            axis=axis,
            status="COMPLETE",
            previous_state=0,
            current_state=1,
            split="V2_NATURAL_LOCKED_TEST",
            contract="V2_NATURAL_FREQUENCY_LOCKED",
            packets=natural_packets,
            labels=natural_labels,
        )

    balanced_input = tmp_path / "balanced.jsonl"
    natural_input = tmp_path / "natural.jsonl"
    for path, packets in (
        (balanced_input, balanced_packets),
        (natural_input, natural_packets),
    ):
        path.write_text(
            "".join(item.model_dump_json() + "\n" for item in packets), encoding="utf-8"
        )
    gold = tmp_path / "gold-v2.csv"
    with gold.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gold_rows[0]))
        writer.writeheader()
        writer.writerows(gold_rows)

    def classification_build(
        path: Path,
        packet_path: Path,
        labels: list[AxisPairClassification],
    ) -> None:
        path.mkdir()
        (path / "classifications.jsonl").write_text(
            "".join(item.model_dump_json() + "\n" for item in labels), encoding="utf-8"
        )
        (path / "stage-status.json").write_text(
            json.dumps(
                {
                    "status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE",
                    "input_blinded_packet_sha256": sha256_file(packet_path),
                    "parser_version": "parser-v2-test",
                    "prompt_sha256": "a" * 64,
                    "requested_model": "fixture",
                }
            ),
            encoding="utf-8",
        )

    balanced_build = tmp_path / "balanced-classifications"
    natural_build = tmp_path / "natural-classifications"
    classification_build(balanced_build, balanced_input, balanced_labels)
    classification_build(natural_build, natural_input, natural_labels)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "schema_version": "moatrader-historical-evidence-parser-freeze-v2/2",
                "status": "V2_PARSER_FROZEN_AWAITING_DUAL_INDEPENDENT_LOCKED_TESTS",
                "parser_version": "parser-v2-test",
                "prompt_sha256": "a" * 64,
                "requested_model": "fixture",
                "natural_locked_packet_sha256": sha256_file(natural_input),
                "balanced_locked_packet_sha256": sha256_file(balanced_input),
                "human_gold_sha256": sha256_file(gold),
                "locked_sets_disjoint": True,
                "v1_locked_rows_reused": False,
            }
        ),
        encoding="utf-8",
    )
    balanced_consumption = tmp_path / "balanced-consumption.json"
    natural_consumption = tmp_path / "natural-consumption.json"
    balanced_output = tmp_path / "balanced-evaluation"
    natural_output = tmp_path / "natural-evaluation"
    balanced = evaluate_v2_locked_parser(
        packet_input=balanced_input,
        classification_build=balanced_build,
        human_gold=gold,
        parser_freeze_manifest=freeze,
        locked_consumption_record=balanced_consumption,
        output=balanced_output,
        locked_kind="BALANCED",
        minimum_per_axis_stratum=1,
        minimum_overall_directional_agreement=1,
        minimum_axis_directional_agreement=1,
        maximum_neutral_to_bullish_rate=0,
    )
    natural = evaluate_v2_locked_parser(
        packet_input=natural_input,
        classification_build=natural_build,
        human_gold=gold,
        parser_freeze_manifest=freeze,
        locked_consumption_record=natural_consumption,
        output=natural_output,
        locked_kind="NATURAL",
        minimum_natural_per_axis=1,
        minimum_overall_directional_agreement=1,
        minimum_axis_directional_agreement=1,
        maximum_neutral_to_bullish_rate=0,
    )
    combined_path = tmp_path / "combined.json"
    combined = combine_v2_locked_evaluations(
        natural_evaluation_manifest=natural_output / "stage-status.json",
        balanced_evaluation_manifest=balanced_output / "stage-status.json",
        parser_freeze_manifest=freeze,
        output=combined_path,
    )

    assert balanced["status"] == "V2_BALANCED_LOCKED_TEST_PASSED"
    assert natural["status"] == "V2_NATURAL_LOCKED_TEST_PASSED"
    assert combined["status"] == "V2_LOCKED_TESTS_PASSED"
    balanced_quality = json.loads(
        (balanced_output / "parser-quality-report-v2.json").read_text(encoding="utf-8")
    )
    assert balanced_quality["false_stable_count"] == 0
    assert balanced_quality["false_stable_rate"] == 0
    assert balanced_quality["opposite_direction_count"] == 0
    with pytest.raises(FileExistsError, match="already consumed"):
        evaluate_v2_locked_parser(
            packet_input=balanced_input,
            classification_build=balanced_build,
            human_gold=gold,
            parser_freeze_manifest=freeze,
            locked_consumption_record=balanced_consumption,
            output=tmp_path / "second-evaluation",
            locked_kind="BALANCED",
            minimum_per_axis_stratum=1,
        )


def test_prepare_new_independent_natural_and_balanced_locked_sets(tmp_path: Path) -> None:
    semantic_axes = (
        OperatingEvidenceAxis.DEMAND,
        OperatingEvidenceAxis.PRICE_MIX,
    )
    packets = [
        _locked_packet(axis_index * 100 + index, axis)
        for axis_index, axis in enumerate(semantic_axes, start=1)
        for index in range(1, 29)
    ]
    prior = [packets[index * 28] for index in range(2)]
    dev = [packets[index * 28 + 1] for index in range(2)]

    def write_packets(path: Path, rows: list[PairedAxisPacket]) -> None:
        path.write_text(
            "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
        )

    packet_input = tmp_path / "semantic-packets.jsonl"
    prior_input = tmp_path / "v1-locked.jsonl"
    dev_input = tmp_path / "dev.jsonl"
    write_packets(packet_input, packets)
    write_packets(prior_input, prior)
    write_packets(dev_input, dev)
    candidates = tmp_path / "locked-candidates"
    prepared = prepare_locked_candidates(
        packet_input=packet_input,
        prior_v1_inputs=[prior_input],
        dev_inputs=[dev_input],
        output=candidates,
        natural_per_axis=1,
        balanced_candidates_per_axis=25,
    )
    assert prepared["v1_locked_rows_reused"] is False
    assert prepared["locked_sets_disjoint"] is True

    natural = [
        PairedAxisPacket.model_validate_json(line)
        for line in (candidates / "natural-locked-packets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    balanced_candidates = [
        PairedAxisPacket.model_validate_json(line)
        for line in (candidates / "balanced-candidate-packets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    natural_ids = {packet.packet_id for packet in natural}
    by_axis_index: dict[OperatingEvidenceAxis, int] = {axis: 0 for axis in semantic_axes}
    decisions: list[dict[str, object]] = []
    for packet in (*natural, *balanced_candidates):
        if packet.packet_id in natural_ids:
            status, previous_state, current_state = "COMPLETE", 0, 0
        else:
            stratum_index = by_axis_index[packet.axis] % 5
            by_axis_index[packet.axis] += 1
            status, previous_state, current_state = (
                ("COMPLETE", 1, 0),
                ("COMPLETE", 0, 0),
                ("COMPLETE", 0, 1),
                ("INSUFFICIENT_EVIDENCE", None, None),
                ("AMBIGUOUS", None, None),
            )[stratum_index]
        decision: dict[str, object] = {
            "packet_id": packet.packet_id,
            "status": status,
            "review_notes": "fixture HUMAN review",
        }
        if status == "COMPLETE":
            decision.update(
                previous_state=previous_state,
                current_state=current_state,
                previous_anchor="이전",
                current_anchor="현재",
            )
        decisions.append(decision)
    review_decisions = tmp_path / "human-review-decisions.json"
    review_decisions.write_text(
        json.dumps(
            {
                "reviewer": "HUMAN",
                "outcome_vault_opened": False,
                "return_data_opened": False,
                "decisions": decisions,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    materialized = tmp_path / "human-gold-materialized"
    materialization = materialize_human_gold(
        candidate_build=candidates,
        review_decisions=review_decisions,
        output=materialized,
    )
    assert materialization["reviewer"] == "HUMAN"
    assert materialization["review_decision_count"] == 52
    adjudicated = materialized / "adjudicated-human-gold.csv"
    final = tmp_path / "locked-final"
    manifest = finalize_locked_sets(
        candidate_build=candidates,
        adjudicated_human_gold=adjudicated,
        output=final,
        minimum_per_axis_stratum=5,
    )
    assert manifest["status"] == "V2_DUAL_INDEPENDENT_LOCKED_SETS_PREPARED_OUTCOME_BLIND"
    assert manifest["natural_packet_count"] == 2
    assert manifest["balanced_packet_count"] == 50
    assert manifest["gold_label_authority"] == "HUMAN"
    assert all(
        count == 5
        for axis_counts in manifest["balanced_stratum_counts"].values()
        for count in axis_counts.values()
    )
    dev_evaluation = tmp_path / "dev-evaluation.json"
    dev_evaluation.write_text(
        json.dumps(
            {
                "status": "DEV_PASSED_PARSER_READY_TO_FREEZE",
                "parser_version": "parser-v2-test",
                "prompt_sha256": "a" * 64,
                "requested_model": "fixture",
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    parser_freeze = create_v2_parser_freeze(
        dev_evaluation_manifest=dev_evaluation,
        locked_set_preparation_manifest=final / "locked-set-preparation-manifest.json",
        natural_locked_packet_input=final / "natural-locked-packets.jsonl",
        balanced_locked_packet_input=final / "balanced-locked-packets.jsonl",
        human_gold=final / "v2-locked-human-gold.csv",
        output=tmp_path / "parser-freeze.json",
    )
    assert parser_freeze["v1_locked_rows_reused"] is False
    assert parser_freeze["locked_sets_disjoint"] is True
    assert parser_freeze["locked_set_preparation_manifest_sha256"] == sha256_file(
        final / "locked-set-preparation-manifest.json"
    )


def test_abstention_reason_audit_requires_200_grounded_human_reasons(tmp_path: Path) -> None:
    packets = [
        _locked_packet(index, list(OperatingEvidenceAxis)[index % 6])
        for index in range(1, 201)
    ]
    classifications = [
        AxisPairClassification(
            packet_id=packet.packet_id,
            axis=packet.axis,
            status=(
                AxisClassificationStatus.INSUFFICIENT_EVIDENCE
                if index % 2
                else AxisClassificationStatus.AMBIGUOUS
            ),
            confidence=1,
        )
        for index, packet in enumerate(packets)
    ]
    packet_input = tmp_path / "abstentions.jsonl"
    packet_input.write_text(
        "".join(item.model_dump_json() + "\n" for item in packets), encoding="utf-8"
    )
    build = tmp_path / "classification"
    build.mkdir()
    classification_path = build / "classifications.jsonl"
    classification_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in classifications), encoding="utf-8"
    )
    (build / "stage-status.json").write_text(
        json.dumps(
            {
                "status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE",
                "input_blinded_packet_sha256": sha256_file(packet_input),
            }
        ),
        encoding="utf-8",
    )
    prepared = tmp_path / "prepared"
    prepare_status = prepare_abstention_audit(
        packet_input=packet_input,
        classification_build=build,
        output=prepared,
        sample_size=200,
    )
    completed = tmp_path / "completed.csv"
    with (prepared / "abstention-audit-template.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as source, completed.open("w", encoding="utf-8-sig", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row["abstention_reason"] = "AMBIGUOUS_HUMAN_TOO"
            row["reviewer"] = "HUMAN_REVIEWER"
            writer.writerow(row)
    validated = validate_abstention_audit(
        prepared_build=prepared,
        completed_audit=completed,
        output=tmp_path / "validated",
    )

    assert prepare_status["sample_size"] == 200
    assert validated["status"] == "V2_ABSTENTION_AUDIT_PASSED"
    assert validated["reviewed_count"] == 200
