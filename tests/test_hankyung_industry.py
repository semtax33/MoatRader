from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from moatrader.business import (
    ValuationDriver,
    ValuationDriverExtraction,
    ValuationDriverMapper,
    ValuationEvidenceRole,
)
from moatrader.canonical.models import FigureNode, SourceType, TableNode
from moatrader.canonical.models import SourceRef, StatementType
from moatrader.ingestion import (
    BronzeFilingStore,
    HankyungIndustryClient,
    HankyungIndustryCollector,
    HankyungIndustryReport,
    HttpResponse,
    ResilientHttpClient,
    raw_document_from_synalyst_pdf,
)
from moatrader.pipeline import CanonicalFinancialDocumentPipeline
from moatrader.cli import main
from moatrader.semantic import SemanticChunk


def _metadata_row(report_id: int = 643479) -> dict[str, Any]:
    return {
        "REPORT_IDX": report_id,
        "REPORT_TYPE": "IN",
        "INDUSTRY_CODE": "021",
        "INDUSTRY_NAME": "금융업",
        "REPORT_TITLE": "산업 점검",
        "REPORT_CONTENT": "산업 점검",
        "REPORT_FILENAME": "산업:점검.pdf",
        "REPORT_FILEPATH": "https://example.test/report.pdf",
        "REPORT_DATE": "2025-09-02",
        "REGISTER_DATE": "20250902102714",
        "OFFICE_NAME": "테스트증권",
        "REPORT_WRITER": "분석가",
    }


def _pdf_bytes() -> bytes:
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Industry demand increased 12%.")
    content = document.tobytes()
    document.close()
    return content


class _FakeHttp:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> HttpResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected HTTP request")
        return self.responses.pop(0)

    def post_form(self, url: str, **kwargs: Any) -> HttpResponse:
        raise AssertionError("Hankyung collector must not POST")


def test_hankyung_metadata_preserves_exact_pit_and_industry_scope() -> None:
    report = HankyungIndustryReport.from_api(_metadata_row())

    assert report.report_id == "643479"
    assert report.industry_name == "금융업"
    assert report.registered_at.isoformat() == "2025-09-02T10:27:14+09:00"
    assert report.filename == "643479_산업_점검.pdf"
    assert report.adapter_hints()["source_type"] == "INDUSTRY"
    assert report.adapter_hints()["source_specific"]["report_type"] == "IN"
    descriptor = report.descriptor()
    assert descriptor.source_type == SourceType.INDUSTRY
    assert descriptor.available_at == report.registered_at
    assert descriptor.issuer_id == "INDUSTRY:021"


def test_hankyung_client_uses_industry_filter_and_bounded_valid_pdf() -> None:
    payload = {
        "current_page": 1,
        "last_page": 1,
        "data": [_metadata_row(), {**_metadata_row(2), "REPORT_TYPE": "CO"}],
    }
    fake = _FakeHttp(
        [
            HttpResponse(
                url="https://example.test/search",
                status_code=200,
                headers={},
                content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            ),
            HttpResponse(
                url="https://example.test/report.pdf",
                status_code=200,
                headers={"content-type": "application/pdf"},
                content=_pdf_bytes(),
            ),
        ]
    )
    client = HankyungIndustryClient(fake, "token")

    reports = client.search(
        begin_date=date(2025, 9, 1),
        end_date=date(2025, 9, 30),
        industry_codes={"021"},
    )
    content = client.download_pdf(reports[0], max_bytes=1024 * 1024)

    assert len(reports) == 1
    assert content.startswith(b"%PDF-")
    assert fake.calls[0]["query"]["reportType"] == "IN"
    assert fake.calls[0]["query"]["reportRange"] == 500
    assert fake.calls[1]["max_bytes"] == 1024 * 1024
    assert fake.calls[1]["headers"]["Authorization"] == "Bearer token"


def test_hankyung_collect_cli_fails_before_network_without_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("HANKYUNG_BEARER_TOKEN", raising=False)

    exit_code = main(
        [
            "collect",
            "hankyung-industry",
            "--from",
            "2025-01-01",
            "--to",
            "2025-01-31",
            "--output",
            str(tmp_path),
        ]
    )

    assert exit_code == 2
    assert "HANKYUNG_BEARER_TOKEN" in capsys.readouterr().err


def test_hankyung_collector_writes_immutable_industry_bronze_without_redownload(
    tmp_path: Path,
) -> None:
    report = HankyungIndustryReport.from_api(_metadata_row())

    class FakeClient:
        downloads = 0

        def search(self, **_kwargs: Any) -> list[HankyungIndustryReport]:
            return [report]

        def download_pdf(self, _report: HankyungIndustryReport, *, max_bytes: int) -> bytes:
            assert max_bytes > 0
            self.downloads += 1
            return _pdf_bytes()

    client = FakeClient()
    store = BronzeFilingStore(tmp_path / "bronze")
    collector = HankyungIndustryCollector(client, store)
    first = collector.collect(
        begin_date=date(2025, 9, 1),
        end_date=date(2025, 9, 30),
    )
    second = collector.collect(
        begin_date=date(2025, 9, 1),
        end_date=date(2025, 9, 30),
    )

    assert client.downloads == 1
    assert first.downloaded_count == 1
    assert second.unchanged_count == 1
    saved = second.filings[0]
    assert "hankyung-industry" in Path(saved.input_path).parts
    metadata = json.loads(Path(saved.metadata_path).read_text(encoding="utf-8"))
    assert metadata["economic_scope"] == "INDUSTRY"
    assert metadata["available_at"] == "2025-09-02T10:27:14+09:00"


def test_industry_forward_text_is_forecast_not_management_claim() -> None:
    chunk = SemanticChunk(
        chunk_id="AU_industry",
        document_id="HANKYUNG_IN_1",
        node_ids=["N1"],
        chunk_type="atomic_evidence",
        markdown="2026년 산업 수요는 12% 증가할 것으로 전망한다.",
        token_count=12,
        source_refs=[
            SourceRef(
                source_type=SourceType.INDUSTRY,
                document_id="HANKYUNG_IN_1",
                page=1,
            )
        ],
        metadata={"atomic_evidence_key": "AEK_industry"},
    )
    extraction = ValuationDriverExtraction(
        relevant=True,
        primary_driver=ValuationDriver.REVENUE_GROWTH,
        role=ValuationEvidenceRole.SCENARIO_INPUT,
        fact="산업 수요 전망",
    )

    bundle = ValuationDriverMapper().map_atomic_extractions(
        issuer_id="INDUSTRY:001",
        as_of=datetime(2025, 1, 1, tzinfo=timezone.utc),
        extractions=[(extraction, chunk)],
    )

    assert bundle.evidence[0].statement_type == StatementType.FORECAST
    assert bundle.evidence[0].source_type == SourceType.INDUSTRY
    assert bundle.evidence[0].range_widening_required is True


def _synalyst_fixture(name: str) -> tuple[Path, Path, Path]:
    root = Path(r"D:\Programming\python_example\Synalyst")
    pdf = root / "data-lake" / "bronze" / "consensus" / "hankyung" / "industry" / "2025" / "pdf" / name
    reports = root / "data-lake" / "bronze" / "consensus" / "hankyung" / "industry" / "2025" / "json" / "reports.json"
    if not pdf.is_file() or not reports.is_file():
        pytest.skip("local Synalyst industry Bronze fixture is unavailable")
    return root, pdf, reports


def test_existing_synalyst_newsletter_pdf_fails_closed_from_ambiguous_layout_table() -> None:
    root, pdf, reports = _synalyst_fixture("643479_20250901134606295K_02.pdf")
    raw = raw_document_from_synalyst_pdf(pdf, reports)

    prepared = CanonicalFinancialDocumentPipeline(
        synalyst_root=str(root)
    ).prepare_for_llm(raw)
    bundle = prepared.bundle

    assert bundle.metadata.source_type == SourceType.INDUSTRY
    assert bundle.metadata.available_at.isoformat() == "2025-09-02T10:27:14+09:00"
    assert bundle.metadata.parser_version.endswith("synalyst/0.2.15")
    assert bundle.quality.text_retention == 1.0
    assert bundle.quality.ast_table_count == 0
    assert any("ambiguous full-page layout table" in item for item in bundle.quality.warnings)
    assert prepared.evidence_requests == []
    assert prepared.valuation_evidence_units
    assert prepared.valuation_evidence_requests
    assert all(
        unit.metadata["available_at"] == "2025-09-02T10:27:14+09:00"
        for unit in prepared.valuation_evidence_units
    )
    assert all(
        request.metadata["available_at"] == "2025-09-02T10:27:14+09:00"
        and request.metadata["economic_scope"] == "INDUSTRY"
        for request in prepared.valuation_evidence_requests
    )
    assert all(
        reference.source_type == SourceType.INDUSTRY
        for node in bundle.ast.walk()
        for reference in node.source_refs
    )
    assert "external INDUSTRY reference-class evidence" in prepared.valuation_evidence_requests[0].system


def test_existing_synalyst_multipage_pdf_preserves_tables_figures_and_provenance() -> None:
    root, pdf, reports = _synalyst_fixture("645095_20251210074402055_0_ko.pdf")
    raw = raw_document_from_synalyst_pdf(pdf, reports)

    prepared = CanonicalFinancialDocumentPipeline(
        synalyst_root=str(root)
    ).prepare_for_llm(raw)
    bundle = prepared.bundle
    nodes = list(bundle.ast.walk())

    assert bundle.metadata.source_specific["pdf_page_count"] == 4
    assert any(isinstance(node, TableNode) for node in nodes)
    assert any(isinstance(node, FigureNode) for node in nodes)
    assert bundle.assets
    assert len(prepared.valuation_evidence_units) >= 5
    for node in nodes:
        for reference in node.source_refs:
            assert reference.source_type == SourceType.INDUSTRY
            assert reference.page is not None
            assert reference.source_hash == bundle.metadata.raw_sha256
    assert all(
        record.transform_version == bundle.metadata.parser_version
        for record in bundle.provenance.records.values()
    )


def test_industry_prepare_cli_uses_existing_synalyst_pdf_without_collection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, pdf, reports = _synalyst_fixture("643479_20250901134606295K_02.pdf")
    monkeypatch.setattr(
        ResilientHttpClient,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("industry prepare must not access the network")
        ),
    )

    exit_code = main(
        [
            "industry",
            "prepare",
            "--pdf-root",
            str(pdf),
            "--reports-json",
            str(reports),
            "--synalyst-root",
            str(root),
            "--output",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["success_count"] == 1
    assert manifest["failure_count"] == 0
    document = tmp_path / "HANKYUNG_IN_643479"
    assert (document / "bundle.json").is_file()
    assert (document / "industry-evidence-units.jsonl").is_file()
    assert (document / "valuation-evidence-requests.jsonl").is_file()
    assert not (tmp_path / "collections").exists()
