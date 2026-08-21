from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from moatrader.canonical.models import StatementType
from moatrader.expectations.future_eri import EvidenceObservation, EvidenceState, OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    AxisPairClassification,
    BlindedExcerpt,
    HistoricalRegularFiling,
    HistoricalSourceIntegrityManifest,
    HistoricalSourceOrigin,
    HistoricalSourceVariant,
    PairedAxisPacket,
    ReceiptLinkage,
    build_blinded_packets,
    build_historical_evidence_feature_row,
    build_regular_filing_pairs,
    discover_arcana_business_sources,
    discover_moatrader_original_sources,
    merge_historical_sources,
    packet_coverage_report,
    seal_historical_evidence_features,
    source_integrity_record,
    validate_classification_grounding,
    validate_packet_anonymization,
    verify_source_integrity,
)
from scripts.build_historical_future_eri_evidence import run as run_historical_builder
from scripts.classify_historical_future_eri_evidence import build_request
from scripts.seal_historical_future_eri_features import evaluate_human_gold_quality


SEOUL = ZoneInfo("Asia/Seoul")
TRADING_SESSIONS = [date(2020, 3, 31), date(2020, 4, 1), date(2020, 8, 17)]


def _write_zip(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("report.xml", f"<html><body>{text}</body></html>")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _filing(
    *,
    ticker: str,
    rcept_no: str,
    period: date,
    source: Path,
    origin: HistoricalSourceOrigin = HistoricalSourceOrigin.ARCANA_BUSINESS_HTML,
    linkage: ReceiptLinkage = ReceiptLinkage.INFERRED_TICKER_PERIOD,
) -> HistoricalRegularFiling:
    published = datetime.strptime(rcept_no[:8], "%Y%m%d").replace(
        hour=23,
        minute=59,
        tzinfo=SEOUL,
    )
    return HistoricalRegularFiling(
        ticker=ticker,
        issuer_name="테스트회사",
        rcept_no=rcept_no,
        report_name="분기보고서",
        report_code={3: "11013", 6: "11012", 9: "11014", 12: "11011"}[period.month],
        fiscal_period_end=period,
        published_at=published,
        available_at=published,
        signal_timestamp=published,
        source_variants=[
            HistoricalSourceVariant(
                origin=origin,
                path=str(source.resolve()),
                raw_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                byte_count=source.stat().st_size,
                receipt_linkage=linkage,
            )
        ],
    )


def test_integrity_manifest_detects_mutation_of_temporary_copy(tmp_path: Path) -> None:
    copied_source = tmp_path / "copied-source.html"
    copied_source.write_text("immutable source copy", encoding="utf-8")
    manifest = HistoricalSourceIntegrityManifest(
        created_at=datetime.now(SEOUL),
        records=[source_integrity_record(copied_source)],
    )

    verify_source_integrity(manifest)
    copied_source.write_text("changed temporary copy", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable historical source changed"):
        verify_source_integrity(manifest)


def test_arcana_discovery_keeps_regular_original_and_records_amendment(tmp_path: Path) -> None:
    root = tmp_path / "business-info"
    source = root / "005930" / "business_info_(2020.03).html"
    source.parent.mkdir(parents=True)
    source.write_text("<html><body>수주잔고 증가</body></html>", encoding="utf-8")
    comment = (
        tmp_path
        / "finance-comment"
        / "005930"
        / "finance_statement_comment_(2020.03).html"
    )
    comment.parent.mkdir(parents=True)
    comment.write_text("<html><body>영업이익률 개선</body></html>", encoding="utf-8")
    statement = (
        tmp_path
        / "finance-statement"
        / "005930"
        / "finance_statement_(2020.03).html"
    )
    statement.parent.mkdir(parents=True)
    statement.write_text("<html><body>재고자산 감소</body></html>", encoding="utf-8")
    metadata = tmp_path / "kr_report_metadata.csv"
    rows = [
        {
            "source_type": "comment",
            "period_end_date": "2020-03-31",
            "report_name": "분기보고서 (2020.03)",
            "stock_code": "005930",
            "rcept_no": "20200330000001",
            "corp_name": "테스트회사",
        },
        {
            "source_type": "comment",
            "period_end_date": "2020-03-31",
            "report_name": "[정정]분기보고서 (2020.03)",
            "stock_code": "005930",
            "rcept_no": "20200402000001",
            "corp_name": "테스트회사",
        },
    ]
    with metadata.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    filings, amendments = discover_arcana_business_sources(
        metadata_path=metadata,
        business_html_root=root,
        trading_sessions=TRADING_SESSIONS,
        begin_year=2020,
        end_year=2020,
    )

    assert len(filings) == 1
    assert len(amendments) == 1
    assert {item.origin for item in filings[0].source_variants} == {
        HistoricalSourceOrigin.ARCANA_BUSINESS_HTML,
        HistoricalSourceOrigin.ARCANA_FINANCE_COMMENT_HTML,
        HistoricalSourceOrigin.ARCANA_FINANCE_STATEMENT_HTML,
    }
    assert filings[0].signal_timestamp.date() == date(2020, 3, 31)


def test_moatrader_discovery_finds_exact_original_and_deduplicates_copies(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / "experiments" / "historical-validation-v7-2020-2025", tmp_path / "other"]
    archive_hash = ""
    for root in roots:
        filing_dir = root / "filings" / "005930" / "20200330000001"
        archive = filing_dir / "original-document.zip"
        archive_hash = _write_zip(archive, "수주잔고 증가")
        (filing_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "source_document_id": "20200330000001",
                    "ticker": "005930",
                    "title": "분기보고서 (2020.03)",
                    "period_end": "2020-03-31",
                    "original_archive_sha256": archive_hash,
                    "is_amendment": False,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    missing_ticker_dir = tmp_path / "other" / "filings" / "unknown" / "20200330000002"
    missing_ticker_archive = missing_ticker_dir / "original.zip"
    missing_ticker_hash = _write_zip(missing_ticker_archive, "수요 증가")
    (missing_ticker_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_document_id": "20200330000002",
                "title": "분기보고서 (2020.03)",
                "period_end": "2020-03-31",
                "source_specific": {"archive_sha256": missing_ticker_hash},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    filings, audit = discover_moatrader_original_sources(
        data_lake_root=tmp_path,
        trading_sessions=TRADING_SESSIONS,
        begin_year=2020,
        end_year=2020,
    )

    assert len(filings) == 1
    assert audit["duplicate_archive_copies_removed"] == 1
    assert audit["unique_regular_receipts"] == 1
    assert audit["archive_hash_mismatch_count"] == 0
    assert filings[0].source_variants[0].raw_sha256 == archive_hash
    assert "historical-validation-v7-2020-2025" in filings[0].source_variants[0].path


def test_merge_retains_both_independent_source_variants(tmp_path: Path) -> None:
    html_source = tmp_path / "business.html"
    html_source.write_text("<html>수요 증가</html>", encoding="utf-8")
    archive = tmp_path / "original.zip"
    _write_zip(archive, "수요 증가와 수익성 개선")
    period = date(2020, 3, 31)
    arcana = _filing(
        ticker="005930",
        rcept_no="20200330000002",
        period=period,
        source=html_source,
    )
    moatrader = _filing(
        ticker="005930",
        rcept_no="20200330000001",
        period=period,
        source=archive,
        origin=HistoricalSourceOrigin.MOATRADER_OPENDART_ARCHIVE,
        linkage=ReceiptLinkage.EXACT_METADATA,
    )

    merged, audit = merge_historical_sources([arcana], [moatrader])

    assert len(merged) == 1
    assert merged[0].rcept_no == moatrader.rcept_no
    assert {item.origin for item in merged[0].source_variants} == {
        HistoricalSourceOrigin.ARCANA_BUSINESS_HTML,
        HistoricalSourceOrigin.MOATRADER_OPENDART_ARCHIVE,
    }
    assert audit["dual_source_filing_count"] == 1
    assert audit["receipt_conflict_count"] == 1


def test_consecutive_pair_blinds_identifiers_dates_and_market_context(tmp_path: Path) -> None:
    previous_source = tmp_path / "previous.html"
    current_source = tmp_path / "current.html"
    previous_source.write_text(
        "<html><body>테스트회사 005930의 2020년 3월 수주잔고가 감소했다.</body></html>",
        encoding="utf-8",
    )
    current_source.write_text(
        "<html><body>테스트회사 005930의 2020년 6월 수주잔고가 증가했다.</body></html>",
        encoding="utf-8",
    )
    previous = _filing(
        ticker="005930",
        rcept_no="20200330000001",
        period=date(2020, 3, 31),
        source=previous_source,
    )
    current = _filing(
        ticker="005930",
        rcept_no="20200814000001",
        period=date(2020, 6, 30),
        source=current_source,
    )

    pairs = build_regular_filing_pairs([current, previous])
    packets, private = build_blinded_packets(pairs[0])
    backlog = next(item for item in packets if item.axis == OperatingEvidenceAxis.BACKLOG)

    assert len(pairs) == 1
    assert private["ticker"] == "005930"
    assert backlog.market_data_included is False
    assert backlog.future_context_included is False
    combined = " ".join(item.text for item in backlog.previous_excerpts + backlog.current_excerpts)
    assert "테스트회사" not in combined
    assert "005930" not in combined
    assert "2020" not in combined
    assert "[ENTITY]" in combined and "[DATE]" in combined

    classification = AxisPairClassification(
        packet_id=backlog.packet_id,
        axis=backlog.axis,
        previous_state=EvidenceState.WEAKENING,
        current_state=EvidenceState.IMPROVING,
        previous_source_id=backlog.previous_excerpts[0].source_id,
        current_source_id=backlog.current_excerpts[0].source_id,
        previous_source_span="수주잔고가 감소",
        current_source_span="수주잔고가 증가",
        confidence=0.9,
    )
    validate_classification_grounding(classification, backlog)


def test_blinded_packet_round_robins_arcana_section_sources_before_excerpt_cap(
    tmp_path: Path,
) -> None:
    origins = (
        HistoricalSourceOrigin.ARCANA_BUSINESS_HTML,
        HistoricalSourceOrigin.ARCANA_FINANCE_COMMENT_HTML,
        HistoricalSourceOrigin.ARCANA_FINANCE_STATEMENT_HTML,
    )

    def filing_with_sections(*, current: bool) -> HistoricalRegularFiling:
        paths: list[Path] = []
        for index, origin in enumerate(origins):
            path = tmp_path / f"{'current' if current else 'previous'}-{origin.value}.html"
            repeats = 12 if index == 0 else 1
            path.write_text(
                "<html><body>"
                + "\n".join(f"수요 근거 {index}-{item}" for item in range(repeats))
                + "</body></html>",
                encoding="utf-8",
            )
            paths.append(path)
        base = _filing(
            ticker="005930",
            rcept_no="20200814000001" if current else "20200330000001",
            period=date(2020, 6, 30) if current else date(2020, 3, 31),
            source=paths[0],
        )
        return base.model_copy(
            update={
                "source_variants": [
                    HistoricalSourceVariant(
                        origin=origin,
                        path=str(path.resolve()),
                        raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        byte_count=path.stat().st_size,
                        receipt_linkage=ReceiptLinkage.INFERRED_TICKER_PERIOD,
                    )
                    for origin, path in zip(origins, paths, strict=True)
                ]
            }
        )

    pair = build_regular_filing_pairs(
        [filing_with_sections(current=False), filing_with_sections(current=True)]
    )[0]
    demand = next(
        packet
        for packet in build_blinded_packets(pair)[0]
        if packet.axis == OperatingEvidenceAxis.DEMAND
    )

    assert len({item.source_id for item in demand.previous_excerpts}) == 3
    assert len({item.source_id for item in demand.current_excerpts}) == 3


def test_packet_coverage_reports_complete_six_axis_pair(tmp_path: Path) -> None:
    previous_source = tmp_path / "previous.html"
    current_source = tmp_path / "current.html"
    all_keywords = " 수요 가격 수주 마진 재고 생산능력 "
    previous_source.write_text(f"<html>{all_keywords}감소</html>", encoding="utf-8")
    current_source.write_text(f"<html>{all_keywords}증가</html>", encoding="utf-8")
    previous = _filing(
        ticker="005930",
        rcept_no="20200330000001",
        period=date(2020, 3, 31),
        source=previous_source,
    )
    current = _filing(
        ticker="005930",
        rcept_no="20200814000001",
        period=date(2020, 6, 30),
        source=current_source,
    )
    pair = build_regular_filing_pairs([previous, current])[0]
    packets, private = build_blinded_packets(pair)

    report = packet_coverage_report(packets, private_rows=[private])

    assert report["axis_packet_count"] == 6
    assert report["six_axis_candidate_complete"] == 1
    assert report["candidate_coverage"] == 1.0
    assert report["outcomes_opened"] is False
    assert report["returns_opened"] is False


def test_historical_builder_uses_both_sources_without_opening_outcomes(tmp_path: Path) -> None:
    arcana_root = tmp_path / "arcana-business"
    ticker_root = arcana_root / "005930"
    ticker_root.mkdir(parents=True)
    evidence = "수요 가격 수주 마진 재고 생산능력 모두 개선"
    for period in ("2020.03", "2020.06"):
        (ticker_root / f"business_info_({period}).html").write_text(
            f"<html><body>테스트회사 {period} {evidence}</body></html>",
            encoding="utf-8",
        )
        comment = (
            tmp_path
            / "finance-comment"
            / "005930"
            / f"finance_statement_comment_({period}).html"
        )
        comment.parent.mkdir(parents=True, exist_ok=True)
        comment.write_text(
            f"<html><body>테스트회사 {period} 마진 원가 개선</body></html>",
            encoding="utf-8",
        )
        statement = (
            tmp_path
            / "finance-statement"
            / "005930"
            / f"finance_statement_({period}).html"
        )
        statement.parent.mkdir(parents=True, exist_ok=True)
        statement.write_text(
            f"<html><body>테스트회사 {period} 매출 재고 설비투자</body></html>",
            encoding="utf-8",
        )
    arcana_metadata = tmp_path / "kr_report_metadata.csv"
    metadata_rows = [
        {
            "source_type": "comment",
            "period_end_date": "2020-03-31",
            "report_name": "분기보고서 (2020.03)",
            "stock_code": "005930",
            "rcept_no": "20200330000001",
            "corp_name": "테스트회사",
        },
        {
            "source_type": "comment",
            "period_end_date": "2020-06-30",
            "report_name": "반기보고서 (2020.06)",
            "stock_code": "005930",
            "rcept_no": "20200814000001",
            "corp_name": "테스트회사",
        },
    ]
    with arcana_metadata.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metadata_rows[0]))
        writer.writeheader()
        writer.writerows(metadata_rows)

    moat_filing = tmp_path / "moat-data-lake" / "filings" / "005930" / "20200814000001"
    archive = moat_filing / "original-document.zip"
    archive_hash = _write_zip(archive, evidence)
    (moat_filing / "metadata.json").write_text(
        json.dumps(
            {
                "source_document_id": "20200814000001",
                "ticker": "005930",
                "title": "반기보고서 (2020.06)",
                "period_end": "2020-06-30",
                "original_archive_sha256": archive_hash,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    calendar = tmp_path / "calendar.csv"
    calendar.write_text("Date\n2020-03-31\n2020-08-17\n", encoding="utf-8")
    sector_map = tmp_path / "sectors.csv"
    sector_map.write_text("ticker,sector\n005930,반도체\n", encoding="utf-8-sig")
    output = tmp_path / "result"

    result = run_historical_builder(
        arcana_metadata=arcana_metadata,
        arcana_business_html=arcana_root,
        moatrader_data_lake=tmp_path / "moat-data-lake",
        trading_calendar=calendar,
        output=output,
        begin_year=2020,
        end_year=2020,
        sector_map_path=sector_map,
        tickers={"005930"},
        gold_per_axis=1,
    )

    assert result["both_source_systems_used"] is True
    assert result["regular_pair_count"] == 1
    assert result["outcome_vault_opened"] is False
    assert result["return_data_opened"] is False
    assert result["source_files_modified"] is False
    assert result["all_arcana_sections_discovered"] is True
    assert result["all_arcana_sections_read_for_pairs"] is True
    assert result["all_arcana_sections_contributed_to_packets"] is True
    assert set(result["pair_source_extraction_by_origin"]) >= {
        "ARCANA_BUSINESS_HTML",
        "ARCANA_FINANCE_COMMENT_HTML",
        "ARCANA_FINANCE_STATEMENT_HTML",
    }
    after = json.loads(
        (output / "private" / "source-integrity-after.json").read_text(encoding="utf-8")
    )
    assert after["verification_status"] == "PASS_NO_SOURCE_MUTATION"
    packet_text = (output / "llm" / "blinded-packets.jsonl").read_text(encoding="utf-8")
    assert "005930" not in packet_text
    assert "테스트회사" not in packet_text
    assert not (output / "future-eri-labels.jsonl").exists()


def test_abstained_axis_classification_cannot_invent_a_state() -> None:
    with pytest.raises(ValueError, match="abstained classification"):
        AxisPairClassification(
            packet_id="PKT_" + "1" * 24,
            axis=OperatingEvidenceAxis.DEMAND,
            status=AxisClassificationStatus.INSUFFICIENT_EVIDENCE,
            previous_state=EvidenceState.STABLE,
            confidence=1,
        )


def test_blinded_llm_request_contains_no_private_identity_or_market_data(tmp_path: Path) -> None:
    previous_source = tmp_path / "previous.html"
    current_source = tmp_path / "current.html"
    previous_source.write_text("<html>테스트회사 2020년 수요 감소</html>", encoding="utf-8")
    current_source.write_text("<html>테스트회사 2020년 수요 증가</html>", encoding="utf-8")
    pair = build_regular_filing_pairs(
        [
            _filing(
                ticker="005930",
                rcept_no="20200330000001",
                period=date(2020, 3, 31),
                source=previous_source,
            ),
            _filing(
                ticker="005930",
                rcept_no="20200814000001",
                period=date(2020, 6, 30),
                source=current_source,
            ),
        ]
    )[0]
    packet = next(
        item for item in build_blinded_packets(pair)[0] if item.axis == OperatingEvidenceAxis.DEMAND
    )

    request = build_request(packet)

    assert "테스트회사" not in request.user
    assert "005930" not in request.user
    assert "2020" not in request.user
    assert "market_data_included\":false" in request.user
    assert "future_context_included\":false" in request.user


def test_pair_level_issuer_name_masks_previous_filing_with_missing_metadata(
    tmp_path: Path,
) -> None:
    previous_source = tmp_path / "previous.html"
    current_source = tmp_path / "current.html"
    previous_source.write_text("<html>동화약품 생산능력 증가</html>", encoding="utf-8")
    current_source.write_text("<html>동화약품 생산능력 유지</html>", encoding="utf-8")
    previous = _filing(
        ticker="000020",
        rcept_no="20200330000001",
        period=date(2020, 3, 31),
        source=previous_source,
    ).model_copy(update={"issuer_name": ""})
    current = _filing(
        ticker="000020",
        rcept_no="20200814000001",
        period=date(2020, 6, 30),
        source=current_source,
    ).model_copy(update={"issuer_name": "동화약품"})
    pair = build_regular_filing_pairs([previous, current])[0]

    packets, _ = build_blinded_packets(pair)
    combined = " ".join(
        excerpt.text
        for packet in packets
        for excerpt in (*packet.previous_excerpts, *packet.current_excerpts)
    )

    assert "동화약품" not in combined
    assert "[ENTITY]" in combined


def test_anonymization_validator_does_not_invent_date_across_excerpt_boundary() -> None:
    packet = PairedAxisPacket(
        packet_id="PKT_" + "1" * 24,
        axis=OperatingEvidenceAxis.CAPACITY_CAPEX,
        previous_excerpts=[
            BlindedExcerpt(source_id="SRC_" + "1" * 20, text="설비 연혁 2018"),
            BlindedExcerpt(source_id="SRC_" + "2" * 20, text=". 05 생산설비 증설"),
        ],
        current_excerpts=[],
    )

    validate_packet_anonymization(
        packet,
        issuer_name="테스트회사",
        ticker="005930",
        rcept_numbers=["20200330000001", "20200814000001"],
    )


def test_human_gold_quality_and_feature_seal_are_outcome_blind(tmp_path: Path) -> None:
    previous_source = tmp_path / "previous.html"
    current_source = tmp_path / "current.html"
    keywords = "수요 가격 수주 마진 재고 생산능력"
    previous_source.write_text(f"<html>{keywords} 유지</html>", encoding="utf-8")
    current_source.write_text(f"<html>{keywords} 개선</html>", encoding="utf-8")
    pair = build_regular_filing_pairs(
        [
            _filing(
                ticker="005930",
                rcept_no="20200330000001",
                period=date(2020, 3, 31),
                source=previous_source,
            ),
            _filing(
                ticker="005930",
                rcept_no="20200814000001",
                period=date(2020, 6, 30),
                source=current_source,
            ),
        ]
    )[0]
    packet_values = build_blinded_packets(pair)[0]
    packets = {item.packet_id: item for item in packet_values}
    classifications: dict[str, AxisPairClassification] = {}
    gold_rows: list[dict[str, str]] = []
    previous_observations: list[EvidenceObservation] = []
    current_observations: list[EvidenceObservation] = []
    for packet in packet_values:
        previous_excerpt = packet.previous_excerpts[0]
        current_excerpt = packet.current_excerpts[0]
        previous_span = previous_excerpt.text[-10:]
        current_span = current_excerpt.text[-10:]
        classification = AxisPairClassification(
            packet_id=packet.packet_id,
            axis=packet.axis,
            previous_state=EvidenceState.STABLE,
            current_state=EvidenceState.IMPROVING,
            previous_source_id=previous_excerpt.source_id,
            current_source_id=current_excerpt.source_id,
            previous_source_span=previous_span,
            current_source_span=current_span,
            confidence=0.9,
        )
        classifications[packet.packet_id] = classification
        gold_rows.append(
            {
                "packet_id": packet.packet_id,
                "axis": packet.axis.value,
                "human_status": "COMPLETE",
                "human_previous_state": "0",
                "human_current_state": "1",
                "human_previous_source_id": previous_excerpt.source_id,
                "human_current_source_id": current_excerpt.source_id,
                "human_previous_source_span": previous_span,
                "human_current_source_span": current_span,
            }
        )
        for side, state, span, filing in (
            ("previous", EvidenceState.STABLE, previous_span, pair.previous),
            ("current", EvidenceState.IMPROVING, current_span, pair.current),
        ):
            observation = EvidenceObservation(
                observation_id=f"{packet.packet_id}:{side}",
                issuer_id=pair.ticker,
                fiscal_period=filing.fiscal_period_end.isoformat(),
                axis=packet.axis,
                state=state,
                source_document_id=filing.rcept_no,
                source_span=span,
                source_published_at=filing.published_at,
                available_at=filing.available_at,
                signal_timestamp=filing.signal_timestamp,
                statement_type=StatementType.DISCLOSED_FACT,
                classification_rule_id="TEST_RULE",
                materiality_rule_id="QUALITATIVE_AXIS_STATE_EQUAL_UNIT_V1",
                confidence=Decimal("0.9"),
                materiality=Decimal(1),
            )
            (previous_observations if side == "previous" else current_observations).append(
                observation
            )

    gold_path = tmp_path / "gold.csv"
    with gold_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gold_rows[0]))
        writer.writeheader()
        writer.writerows(gold_rows)
    quality = evaluate_human_gold_quality(
        human_gold_path=gold_path,
        classifications=classifications,
        packets=packets,
        minimum_gold_per_axis=1,
        minimum_overall_agreement=1,
        minimum_axis_agreement=1,
    )
    feature = build_historical_evidence_feature_row(
        pair=pair,
        previous_observations=previous_observations,
        current_observations=current_observations,
        coverage_sector="반도체",
    )
    seal = seal_historical_evidence_features(
        [feature],
        sealed_at=pair.current.signal_timestamp,
    )

    assert quality["gate_passed"] is True
    assert feature.evidence.evidence_f_score == 6
    assert feature.outcome_data_accessed is False
    assert feature.return_data_accessed is False
    assert seal.outcome_source_opened_before_seal is False
    assert seal.return_data_accessed is False
    tampered = feature.model_dump(mode="json")
    tampered["coverage_sector"] = "변조"
    with pytest.raises(ValueError, match="feature_hash"):
        type(feature).model_validate(tampered)
