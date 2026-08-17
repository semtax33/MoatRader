from __future__ import annotations

import json
import re
import unicodedata
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator

from moatrader.adapters import RawDocument
from moatrader.canonical.models import AvailabilityPrecision, ContractModel, SourceType
from moatrader.ingestion.http import HttpClient, HttpRequestError
from moatrader.ingestion.models import (
    CollectionFailure,
    CollectionResult,
    FilingDescriptor,
)
from moatrader.ingestion.store import BronzeFilingStore


HANKYUNG_REPORT_API_URL = "https://markets.hankyung.com/api/v2/consensus/search/report"
HANKYUNG_PDF_DOWNLOAD_URL = "https://consensus.hankyung.com/analysis/downpdf"
HANKYUNG_CONSENSUS_URL = "https://markets.hankyung.com/consensus"
HANKYUNG_INDUSTRY_REPORT_TYPE = "IN"
SEOUL = ZoneInfo("Asia/Seoul")
_INVALID_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _parse_registered_at(value: Any) -> datetime:
    normalized = str(value or "").strip()
    try:
        return datetime.strptime(normalized, "%Y%m%d%H%M%S").replace(tzinfo=SEOUL)
    except ValueError as exc:
        raise ValueError(f"invalid Hankyung REGISTER_DATE: {normalized!r}") from exc


def _parse_report_date(value: Any) -> date:
    normalized = str(value or "").strip()
    try:
        return date.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"invalid Hankyung REPORT_DATE: {normalized!r}") from exc


def _safe_filename(value: str, maximum: int = 140) -> str:
    normalized = unicodedata.normalize("NFKC", value)
    cleaned = _INVALID_FILENAME.sub("_", normalized)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ._")
    if cleaned.casefold().endswith(".pdf"):
        cleaned = cleaned[:-4].rstrip(" ._")
    return (cleaned[:maximum].rstrip(" ._") or "report") + ".pdf"


class HankyungIndustryReport(ContractModel):
    report_id: str = Field(min_length=1)
    industry_code: str = Field(min_length=1)
    industry_name: str = Field(min_length=1)
    title: str = Field(min_length=1)
    publisher: str | None = None
    author: str | None = None
    report_date: date
    registered_at: datetime
    filename: str = Field(min_length=1)
    file_url: str | None = None
    raw_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_point_in_time(self) -> "HankyungIndustryReport":
        if self.registered_at.tzinfo is None or self.registered_at.utcoffset() is None:
            raise ValueError("registered_at must be timezone-aware")
        published_at = datetime.combine(self.report_date, time.min, tzinfo=SEOUL)
        if self.registered_at < published_at:
            raise ValueError("REGISTER_DATE cannot precede REPORT_DATE")
        return self

    @classmethod
    def from_api(cls, row: dict[str, Any]) -> "HankyungIndustryReport":
        if str(row.get("REPORT_TYPE") or "").upper() != HANKYUNG_INDUSTRY_REPORT_TYPE:
            raise ValueError("Hankyung record is not an industry report")
        report_id = str(row.get("REPORT_IDX") or row.get("LINK_IDX") or "").strip()
        if not report_id:
            raise ValueError("Hankyung industry report is missing REPORT_IDX/LINK_IDX")
        industry_code = str(row.get("INDUSTRY_CODE") or "UNKNOWN").strip() or "UNKNOWN"
        industry_name = str(row.get("INDUSTRY_NAME") or "Unknown industry").strip()
        title = str(row.get("REPORT_TITLE") or row.get("REPORT_CONTENT") or "").strip()
        if not title:
            title = f"Hankyung industry report {report_id}"
        original_filename = str(row.get("REPORT_FILENAME") or title)
        return cls(
            report_id=report_id,
            industry_code=industry_code,
            industry_name=industry_name,
            title=title,
            publisher=str(row.get("OFFICE_NAME") or "").strip() or None,
            author=str(row.get("REPORT_WRITER") or "").strip() or None,
            report_date=_parse_report_date(row.get("REPORT_DATE")),
            registered_at=_parse_registered_at(row.get("REGISTER_DATE")),
            filename=_safe_filename(f"{report_id}_{original_filename}"),
            file_url=(
                str(row.get("REPORT_FILEPATH")).strip()
                if str(row.get("REPORT_FILEPATH") or "").startswith(("https://", "http://"))
                else None
            ),
            raw_metadata=dict(row),
        )

    @property
    def source_document_id(self) -> str:
        return f"HANKYUNG_IN_{self.report_id}"

    def adapter_hints(self) -> dict[str, Any]:
        return {
            "source_type": SourceType.INDUSTRY.value,
            "source_document_id": self.source_document_id,
            "report_id": self.report_id,
            "industry_code": self.industry_code,
            "industry_name": self.industry_name,
            "issuer_id": f"INDUSTRY:{self.industry_code}",
            "issuer_name": self.industry_name,
            "ticker": f"INDUSTRY-{self.industry_code}",
            "title": self.title,
            "report_name": self.title,
            "publisher": self.publisher,
            "author": self.author,
            "report_date": self.report_date.isoformat(),
            "published_at": datetime.combine(
                self.report_date, time.min, tzinfo=SEOUL
            ).isoformat(),
            "available_at": self.registered_at.isoformat(),
            "availability_precision": AvailabilityPrecision.EXACT.value,
            "availability_source": "hankyung_REGISTER_DATE",
            "source_system": "hankyung_consensus",
            "language": "ko",
            "jurisdiction": "KR",
            "source_specific": {
                "report_type": HANKYUNG_INDUSTRY_REPORT_TYPE,
                "publisher": self.publisher,
                "author": self.author,
                "industry_code": self.industry_code,
                "industry_name": self.industry_name,
                "hankyung_report_id": self.report_id,
            },
        }

    def descriptor(self) -> FilingDescriptor:
        published_at = datetime.combine(self.report_date, time.min, tzinfo=SEOUL)
        return FilingDescriptor(
            source_type=SourceType.INDUSTRY,
            source_document_id=self.source_document_id,
            issuer_id=f"INDUSTRY:{self.industry_code}",
            issuer_name=self.industry_name,
            ticker=f"INDUSTRY-{self.industry_code}",
            report_name=self.title,
            form_type="INDUSTRY_REPORT",
            filing_date=self.report_date,
            report_date=self.report_date,
            published_at=published_at,
            available_at=self.registered_at,
            availability_precision=AvailabilityPrecision.EXACT,
            availability_source="hankyung_REGISTER_DATE",
            primary_document_name="report.pdf",
            primary_document_url=self.file_url or HANKYUNG_PDF_DOWNLOAD_URL,
            source_specific=self.adapter_hints()["source_specific"],
        )


def load_hankyung_industry_reports(path: str | Path) -> dict[str, HankyungIndustryReport]:
    metadata_path = Path(path).expanduser().resolve()
    payload = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        raise ValueError("Hankyung reports.json must contain a list")
    reports: dict[str, HankyungIndustryReport] = {}
    for row in payload:
        if not isinstance(row, dict) or str(row.get("REPORT_TYPE") or "").upper() != "IN":
            continue
        report = HankyungIndustryReport.from_api(row)
        prior = reports.get(report.report_id)
        if prior is not None and prior.raw_metadata != report.raw_metadata:
            raise ValueError(f"conflicting Hankyung report metadata: {report.report_id}")
        reports[report.report_id] = report
    return reports


def raw_document_from_synalyst_pdf(
    pdf_path: str | Path,
    reports_json: str | Path,
) -> RawDocument:
    """Join an already-downloaded Synalyst PDF to its authoritative API metadata."""

    path = Path(pdf_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    report_id = path.stem.partition("_")[0]
    report = load_hankyung_industry_reports(reports_json).get(report_id)
    if report is None:
        raise ValueError(f"no REPORT_IDX={report_id} metadata for {path.name}")
    return RawDocument(
        content=path.read_bytes(),
        uri=report.file_url or path.as_uri(),
        fetched_at=report.registered_at.astimezone(timezone.utc),
        media_type="application/pdf",
        hints={**report.adapter_hints(), "local_path": str(path)},
    )


class HankyungIndustryClient:
    """Small bounded client adapted from Synalyst's Hankyung consensus crawler."""

    def __init__(self, http: HttpClient, bearer_token: str) -> None:
        token = bearer_token.strip()
        if not token:
            raise ValueError("Hankyung bearer token is required")
        self.http = http
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json, application/pdf;q=0.9, */*;q=0.8",
            "Referer": HANKYUNG_CONSENSUS_URL,
        }

    def search(
        self,
        *,
        begin_date: date,
        end_date: date,
        industry_codes: set[str] | None = None,
        page_size: int = 500,
        maximum: int | None = None,
    ) -> list[HankyungIndustryReport]:
        if end_date < begin_date:
            raise ValueError("end_date must not precede begin_date")
        if not 1 <= page_size <= 1000:
            raise ValueError("page_size must be between 1 and 1000")
        wanted = {str(value).strip() for value in (industry_codes or set()) if str(value).strip()}
        reports: dict[str, HankyungIndustryReport] = {}
        page = 1
        last_page = 1
        while page <= last_page:
            response = self.http.get(
                HANKYUNG_REPORT_API_URL,
                query={
                    "page": page,
                    "reportType": HANKYUNG_INDUSTRY_REPORT_TYPE,
                    "fromDate": begin_date.isoformat(),
                    "toDate": end_date.isoformat(),
                    "gradeCode": "ALL",
                    "changePrices": "ALL",
                    "searchType": "ALL",
                    "reportRange": page_size,
                },
                headers=self.headers,
                max_bytes=64 * 1024 * 1024,
            )
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Hankyung search response must be a JSON object")
            raw_rows = payload.get("data")
            rows = raw_rows if isinstance(raw_rows, list) else list(raw_rows.values()) if isinstance(raw_rows, dict) else None
            if rows is None:
                raise ValueError("Hankyung search response data must be a list or object")
            for row in rows:
                if not isinstance(row, dict):
                    continue
                try:
                    report = HankyungIndustryReport.from_api(row)
                except ValueError:
                    continue
                if wanted and report.industry_code not in wanted:
                    continue
                reports.setdefault(report.report_id, report)
                if maximum is not None and len(reports) >= maximum:
                    break
            if maximum is not None and len(reports) >= maximum:
                break
            try:
                last_page = max(page, int(payload.get("last_page") or page))
            except (TypeError, ValueError) as exc:
                raise ValueError("Hankyung last_page is invalid") from exc
            page += 1
        return sorted(
            reports.values(),
            key=lambda item: (item.registered_at, item.report_id),
        )

    def download_pdf(
        self,
        report: HankyungIndustryReport,
        *,
        max_bytes: int,
    ) -> bytes:
        candidates: list[tuple[str, dict[str, Any] | None]] = [
            (HANKYUNG_PDF_DOWNLOAD_URL, {"report_idx": report.report_id})
        ]
        if report.file_url:
            candidates.append((report.file_url, None))
        failures: list[str] = []
        for url, query in candidates:
            try:
                response = self.http.get(
                    url,
                    query=query,
                    headers=self.headers,
                    max_bytes=max_bytes,
                )
                self._validate_pdf(response.content)
                return response.content
            except (HttpRequestError, ValueError) as exc:
                failures.append(f"{url}: {exc}")
        raise HttpRequestError("all Hankyung PDF candidates failed: " + " | ".join(failures))

    @staticmethod
    def _validate_pdf(content: bytes) -> None:
        if not content.startswith(b"%PDF-"):
            raise ValueError("downloaded response is not a PDF")
        try:
            import fitz

            document = fitz.open(stream=content, filetype="pdf")
            try:
                if document.needs_pass:
                    raise ValueError("password-protected PDF is unsupported")
                if document.page_count < 1:
                    raise ValueError("PDF has no pages")
            finally:
                document.close()
        except ImportError as exc:  # pragma: no cover - packaging failure
            raise RuntimeError("PyMuPDF is required to validate downloaded PDFs") from exc
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"downloaded PDF is structurally invalid: {exc}") from exc


class HankyungIndustryCollector:
    def __init__(
        self,
        client: HankyungIndustryClient,
        store: BronzeFilingStore,
        *,
        max_pdf_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if max_pdf_bytes <= 0:
            raise ValueError("max_pdf_bytes must be positive")
        self.client = client
        self.store = store
        self.max_pdf_bytes = max_pdf_bytes

    def collect(
        self,
        *,
        begin_date: date,
        end_date: date,
        industry_codes: set[str] | None = None,
        maximum: int | None = None,
        refresh: bool = False,
    ) -> CollectionResult:
        started = datetime.now(timezone.utc)
        reports = self.client.search(
            begin_date=begin_date,
            end_date=end_date,
            industry_codes=industry_codes,
            maximum=maximum,
        )
        filings = []
        failures: list[CollectionFailure] = []
        for report in reports:
            try:
                current = self.store.current(SourceType.INDUSTRY, report.source_document_id)
                if current is not None and not refresh:
                    filings.append(current)
                    continue
                content = self.client.download_pdf(report, max_bytes=self.max_pdf_bytes)
                filings.append(
                    self.store.save(
                        report.descriptor(),
                        files={"report.pdf": content},
                        primary_path="report.pdf",
                    )
                )
            except Exception as exc:  # isolate one damaged/missing report
                failures.append(
                    CollectionFailure(
                        source_document_id=report.source_document_id,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
        return CollectionResult(
            source_type=SourceType.INDUSTRY,
            started_at=started,
            completed_at=datetime.now(timezone.utc),
            query={
                "from": begin_date.isoformat(),
                "to": end_date.isoformat(),
                "industry_codes": sorted(industry_codes or set()),
                "maximum": maximum,
            },
            discovered_count=len(reports),
            filings=filings,
            failures=failures,
        )

