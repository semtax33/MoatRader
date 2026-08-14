from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import Field, model_validator

from moatrader.canonical.models import AvailabilityPrecision, ContractModel, SourceType
from moatrader.ingestion.http import HttpClient
from moatrader.ingestion.models import (
    CollectedFiling,
    CollectionFailure,
    CollectionResult,
    FilingDescriptor,
)
from moatrader.ingestion.store import BronzeFilingStore, safe_relative_path


SEC_DATA_URL = "https://data.sec.gov"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_SEC_FORMS = frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A"})
_ACCESSION_PATTERN = re.compile(r"^\d{10}-\d{2}-\d{6}$")


class SecCompany(ContractModel):
    cik: str = Field(pattern=r"^\d{10}$")
    ticker: str
    title: str


class SecFiling(ContractModel):
    accession_number: str = Field(pattern=r"^\d{10}-\d{2}-\d{6}$")
    filing_date: date
    report_date: date | None = None
    acceptance_at: datetime
    form: str
    primary_document: str
    primary_document_description: str | None = None
    file_number: str | None = None
    items: str | None = None
    size: int | None = Field(default=None, ge=0)
    is_xbrl: bool = False
    is_inline_xbrl: bool = False

    @model_validator(mode="after")
    def acceptance_is_aware(self) -> "SecFiling":
        if self.acceptance_at.tzinfo is None or self.acceptance_at.utcoffset() is None:
            raise ValueError("SEC acceptanceDateTime must be timezone-aware")
        return self

    @property
    def is_amendment(self) -> bool:
        return self.form.upper().endswith("/A")

    @property
    def base_form(self) -> str:
        return self.form[:-2] if self.is_amendment else self.form


class SecSubmissionHistory(ContractModel):
    cik: str = Field(pattern=r"^\d{10}$")
    name: str
    tickers: list[str] = Field(default_factory=list)
    exchanges: list[str] = Field(default_factory=list)
    filings: list[SecFiling] = Field(default_factory=list)


class SecEdgarClient:
    def __init__(
        self,
        http: HttpClient,
        *,
        declared_user_agent: str,
        data_url: str = SEC_DATA_URL,
        archives_url: str = SEC_ARCHIVES_URL,
        tickers_url: str = SEC_TICKERS_URL,
    ) -> None:
        validate_sec_user_agent(declared_user_agent)
        self.http = http
        self.declared_user_agent = declared_user_agent.strip()
        self.data_url = data_url.rstrip("/")
        self.archives_url = archives_url.rstrip("/")
        self.tickers_url = tickers_url

    def company_tickers(self) -> list[SecCompany]:
        response = self.http.get(self.tickers_url, headers={"Accept": "application/json"})
        try:
            payload = response.json()
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("SEC company ticker catalog is not valid JSON") from exc
        companies: list[SecCompany] = []
        values = payload.values() if isinstance(payload, dict) else payload
        for row in values:
            if not isinstance(row, dict) or row.get("cik_str") is None:
                continue
            companies.append(
                SecCompany(
                    cik=normalize_cik(str(row["cik_str"])),
                    ticker=str(row.get("ticker") or "").upper(),
                    title=str(row.get("title") or ""),
                )
            )
        return companies

    def submission_history(
        self,
        cik: str,
        *,
        begin_date: date,
        end_date: date,
        amendment_lookback_days: int = 550,
    ) -> SecSubmissionHistory:
        normalized_cik = normalize_cik(cik)
        if end_date < begin_date:
            raise ValueError("end_date must not precede begin_date")
        response = self.http.get(
            f"{self.data_url}/submissions/CIK{normalized_cik}.json",
            headers={"Accept": "application/json"},
        )
        try:
            payload = response.json()
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"SEC submissions response is not valid JSON for CIK {normalized_cik}") from exc
        recent = ((payload.get("filings") or {}).get("recent") or {})
        filings = parse_sec_filing_columns(recent)
        context_begin = begin_date - timedelta(days=amendment_lookback_days)
        for descriptor in (payload.get("filings") or {}).get("files") or []:
            name = str(descriptor.get("name") or "")
            if not re.fullmatch(r"[A-Za-z0-9._-]+\.json", name):
                raise ValueError(f"unsafe SEC submissions history filename: {name!r}")
            filing_from = _optional_date(descriptor.get("filingFrom"))
            filing_to = _optional_date(descriptor.get("filingTo"))
            if filing_from and filing_to and (filing_to < context_begin or filing_from > end_date):
                continue
            historical_response = self.http.get(
                f"{self.data_url}/submissions/{name}",
                headers={"Accept": "application/json"},
            )
            try:
                historical_payload = historical_response.json()
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"SEC historical submissions file is not valid JSON: {name}") from exc
            columns = historical_payload.get("filings", {}).get("recent") if isinstance(historical_payload, dict) else None
            filings.extend(parse_sec_filing_columns(columns or historical_payload))
        unique = {filing.accession_number: filing for filing in filings}
        return SecSubmissionHistory(
            cik=normalized_cik,
            name=str(payload.get("name") or ""),
            tickers=[str(value) for value in payload.get("tickers") or []],
            exchanges=[str(value) for value in payload.get("exchanges") or []],
            filings=sorted(unique.values(), key=lambda item: (item.acceptance_at, item.accession_number)),
        )

    def filing_urls(self, cik: str, filing: SecFiling) -> tuple[str, str]:
        cik_number = str(int(normalize_cik(cik)))
        accession_directory = filing.accession_number.replace("-", "")
        primary_path = safe_relative_path(filing.primary_document).as_posix()
        quoted_primary = "/".join(quote(part, safe="") for part in primary_path.split("/"))
        directory = f"{self.archives_url}/{cik_number}/{accession_directory}"
        return f"{directory}/{quoted_primary}", f"{directory}/{filing.accession_number}.txt"

    def download_filing(
        self,
        cik: str,
        filing: SecFiling,
        *,
        max_bytes: int | None = None,
    ) -> tuple[bytes, bytes, str, str]:
        primary_url, complete_url = self.filing_urls(cik, filing)
        primary = self.http.get(primary_url, max_bytes=max_bytes).content
        complete = self.http.get(complete_url, max_bytes=max_bytes).content
        return primary, complete, primary_url, complete_url


class SecEdgarCollector:
    def __init__(
        self,
        client: SecEdgarClient,
        store: BronzeFilingStore,
        *,
        max_download_bytes: int = 256 * 1024 * 1024,
        availability_lag_minutes: int = 5,
    ) -> None:
        if max_download_bytes <= 0:
            raise ValueError("max_download_bytes must be positive")
        if availability_lag_minutes < 0:
            raise ValueError("availability_lag_minutes must be non-negative")
        self.client = client
        self.store = store
        self.max_download_bytes = max_download_bytes
        self.availability_lag_minutes = availability_lag_minutes

    def collect(
        self,
        *,
        begin_date: date,
        end_date: date,
        tickers: list[str] | None = None,
        ciks: list[str] | None = None,
        forms: set[str] | None = None,
        refresh: bool = False,
        max_filings: int | None = None,
    ) -> CollectionResult:
        started = datetime.now(timezone.utc)
        requested_tickers = [value.strip().upper() for value in dict.fromkeys(tickers or [])]
        requested_ciks = [normalize_cik(value) for value in dict.fromkeys(ciks or [])]
        if not requested_tickers and not requested_ciks:
            raise ValueError("provide at least one SEC --ticker or --cik")
        selected_forms = {value.strip().upper() for value in (forms or DEFAULT_SEC_FORMS) if value.strip()}
        if not selected_forms:
            raise ValueError("at least one SEC form is required")

        catalog = self.client.company_tickers() if requested_tickers else []
        by_ticker = {item.ticker.upper(): item for item in catalog}
        by_cik: dict[str, list[SecCompany]] = {}
        for company in catalog:
            by_cik.setdefault(company.cik, []).append(company)
        missing = [ticker for ticker in requested_tickers if ticker not in by_ticker]
        if missing:
            raise ValueError(f"SEC tickers not found in official catalog: {missing}")
        ticker_by_cik = {by_ticker[ticker].cik: ticker for ticker in requested_tickers}
        requested_ciks.extend(by_ticker[ticker].cik for ticker in requested_tickers)
        requested_ciks = list(dict.fromkeys(requested_ciks))

        histories: dict[str, SecSubmissionHistory] = {}
        candidates: list[tuple[SecSubmissionHistory, SecFiling]] = []
        amendment_links: dict[str, str] = {}
        for cik in requested_ciks:
            history = self.client.submission_history(cik, begin_date=begin_date, end_date=end_date)
            histories[cik] = history
            amendment_links.update(_sec_amendment_links(history.filings))
            for filing in history.filings:
                if begin_date <= filing.filing_date <= end_date and filing.form.upper() in selected_forms:
                    candidates.append((history, filing))
        candidates.sort(key=lambda item: (item[1].acceptance_at, item[1].accession_number))
        if max_filings is not None:
            if max_filings <= 0:
                raise ValueError("max_filings must be positive")
            candidates = candidates[:max_filings]

        collected: list[CollectedFiling] = []
        failures: list[CollectionFailure] = []
        for history, filing in candidates:
            current = self.store.current(SourceType.SEC_EDGAR, filing.accession_number)
            if current is not None and not refresh:
                collected.append(current)
                continue
            try:
                extension = Path(filing.primary_document).suffix.lower()
                if extension not in {".htm", ".html", ".xhtml"}:
                    raise ValueError(
                        f"primary SEC document is not HTML/XHTML: {filing.primary_document}"
                    )
                primary, complete, primary_url, complete_url = self.client.download_filing(
                    history.cik,
                    filing,
                    max_bytes=self.max_download_bytes,
                )
                linked_original = amendment_links.get(filing.accession_number)
                available_at = filing.acceptance_at + timedelta(minutes=self.availability_lag_minutes)
                ticker = ticker_by_cik.get(history.cik)
                if ticker is None:
                    catalog_matches = by_cik.get(history.cik) or []
                    ticker = (
                        sorted((item.ticker for item in catalog_matches), key=str.upper)[0]
                        if catalog_matches
                        else (history.tickers[0] if history.tickers else None)
                    )
                descriptor = FilingDescriptor(
                    source_type=SourceType.SEC_EDGAR,
                    source_document_id=filing.accession_number,
                    issuer_id=history.cik,
                    issuer_name=history.name,
                    ticker=ticker,
                    report_name=f"Form {filing.form}",
                    form_type=filing.form,
                    filing_date=filing.filing_date,
                    report_date=filing.report_date,
                    published_at=filing.acceptance_at,
                    available_at=available_at,
                    availability_precision=(
                        AvailabilityPrecision.EXACT
                        if self.availability_lag_minutes == 0
                        else AvailabilityPrecision.INFERRED
                    ),
                    availability_source=(
                        "SEC_SUBMISSIONS_ACCEPTANCE_DATETIME"
                        if self.availability_lag_minutes == 0
                        else f"SEC_ACCEPTANCE_PLUS_{self.availability_lag_minutes}M_CONSERVATIVE"
                    ),
                    primary_document_name=filing.primary_document,
                    primary_document_url=primary_url,
                    archive_url=complete_url,
                    is_amendment=linked_original is not None,
                    amends_document_id=linked_original,
                    source_specific={
                        "accession_number": filing.accession_number,
                        "cik": history.cik,
                        "form_type": filing.form,
                        "filing_date": filing.filing_date.isoformat(),
                        "report_date": filing.report_date.isoformat() if filing.report_date else None,
                        "acceptance_datetime": filing.acceptance_at.isoformat(),
                        "primary_document": filing.primary_document,
                        "primary_document_description": filing.primary_document_description,
                        "file_number": filing.file_number,
                        "items": filing.items,
                        "size": filing.size,
                        "is_xbrl": filing.is_xbrl,
                        "is_inline_xbrl": filing.is_inline_xbrl,
                        "reported_as_amendment": filing.is_amendment,
                        "amendment_link_status": "LINKED" if linked_original else ("UNRESOLVED" if filing.is_amendment else "NOT_APPLICABLE"),
                        "amendment_link_method": "BASE_FORM_AND_REPORT_DATE" if linked_original else None,
                        "complete_submission_url": complete_url,
                        "exchanges": history.exchanges,
                        "all_tickers": history.tickers,
                    },
                )
                primary_relative = f"documents/{safe_relative_path(filing.primary_document).as_posix()}"
                collected.append(
                    self.store.save(
                        descriptor,
                        files={
                            primary_relative: primary,
                            "original-submission.txt": complete,
                        },
                        primary_path=primary_relative,
                    )
                )
            except Exception as exc:
                failures.append(
                    CollectionFailure(
                        source_document_id=filing.accession_number,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
        return CollectionResult(
            source_type=SourceType.SEC_EDGAR,
            started_at=started,
            completed_at=datetime.now(timezone.utc),
            query={
                "begin_date": begin_date.isoformat(),
                "end_date": end_date.isoformat(),
                "tickers": requested_tickers,
                "ciks": requested_ciks,
                "forms": sorted(selected_forms),
                "availability_lag_minutes": self.availability_lag_minutes,
                "refresh": refresh,
                "max_filings": max_filings,
            },
            discovered_count=len(candidates),
            filings=collected,
            failures=failures,
        )


def validate_sec_user_agent(value: str) -> None:
    normalized = value.strip()
    if len(normalized) < 8 or "@" not in normalized or normalized.startswith("@"):
        raise ValueError(
            "SEC_USER_AGENT must identify the application/company and include a contact email, "
            "for example: 'MoatRader admin@example.com'"
        )


def normalize_cik(value: str) -> str:
    normalized = value.strip().upper().removeprefix("CIK")
    if not normalized.isdigit() or len(normalized) > 10:
        raise ValueError(f"invalid SEC CIK: {value!r}")
    return normalized.zfill(10)


def parse_sec_filing_columns(columns: Any) -> list[SecFiling]:
    if not isinstance(columns, dict):
        return []
    accessions = columns.get("accessionNumber") or []
    if not isinstance(accessions, list):
        raise ValueError("SEC submissions accessionNumber must be an array")
    filings: list[SecFiling] = []
    for index, accession_value in enumerate(accessions):
        accession = str(accession_value or "")
        if not _ACCESSION_PATTERN.fullmatch(accession):
            raise ValueError(f"invalid SEC accession number: {accession!r}")

        def value(key: str) -> Any:
            values = columns.get(key) or []
            return values[index] if isinstance(values, list) and index < len(values) else None

        acceptance_raw = str(value("acceptanceDateTime") or "")
        try:
            acceptance = datetime.fromisoformat(acceptance_raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid SEC acceptanceDateTime for {accession}: {acceptance_raw!r}") from exc
        if acceptance.tzinfo is None or acceptance.utcoffset() is None:
            raise ValueError(f"SEC acceptanceDateTime lacks timezone for {accession}: {acceptance_raw!r}")
        filing_date = _optional_date(value("filingDate"))
        if filing_date is None:
            raise ValueError(f"SEC filingDate is missing for {accession}")
        size_value = value("size")
        filings.append(
            SecFiling(
                accession_number=accession,
                filing_date=filing_date,
                report_date=_optional_date(value("reportDate")),
                acceptance_at=acceptance,
                form=str(value("form") or ""),
                primary_document=str(value("primaryDocument") or ""),
                primary_document_description=str(value("primaryDocDescription") or "") or None,
                file_number=str(value("fileNumber") or "") or None,
                items=str(value("items") or "") or None,
                size=int(size_value) if size_value not in (None, "") else None,
                is_xbrl=str(value("isXBRL") or "0") in {"1", "true", "True"},
                is_inline_xbrl=str(value("isInlineXBRL") or "0") in {"1", "true", "True"},
            )
        )
    return filings


def _sec_amendment_links(filings: list[SecFiling]) -> dict[str, str]:
    originals: dict[tuple[str, date | None], list[SecFiling]] = {}
    for filing in filings:
        if not filing.is_amendment:
            originals.setdefault((filing.form.upper(), filing.report_date), []).append(filing)
    links: dict[str, str] = {}
    for filing in filings:
        if not filing.is_amendment:
            continue
        candidates = [
            item
            for item in originals.get((filing.base_form.upper(), filing.report_date), [])
            if (item.acceptance_at, item.accession_number) < (filing.acceptance_at, filing.accession_number)
        ]
        if candidates:
            links[filing.accession_number] = max(
                candidates,
                key=lambda item: (item.acceptance_at, item.accession_number),
            ).accession_number
    return links


def _optional_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid SEC date: {value!r}") from exc
