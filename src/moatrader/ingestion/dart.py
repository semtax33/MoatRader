from __future__ import annotations

import calendar
import hashlib
import io
import json
import re
import time as time_module
import zipfile
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from lxml import etree
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


DART_BASE_URL = "https://opendart.fss.or.kr/api"
DART_VIEWER_URL = "https://dart.fss.or.kr/dsaf001/main.do"
_KST = ZoneInfo("Asia/Seoul")
_AMENDMENT_PREFIX = re.compile(
    r"^\s*\[(?:기재정정|첨부정정|첨부추가|변경등록|연장결정|발행조건확정|정정명령부과|정정제출요구)\]\s*"
)
_REPORT_KINDS = {
    "annual": "사업보고서",
    "semiannual": "반기보고서",
    "quarterly": "분기보고서",
}


class DartApiError(RuntimeError):
    def __init__(self, status: str, message: str) -> None:
        super().__init__(f"OpenDART API error {status}: {message}")
        self.status = status


class DartSearchQuery(ContractModel):
    begin_date: date
    end_date: date
    corp_code: str | None = Field(default=None, pattern=r"^\d{8}$")
    disclosure_type: str | None = Field(default="A", min_length=1, max_length=1)
    detail_type: str | None = Field(default=None, pattern=r"^[A-J]\d{3}$")
    final_only: bool = False

    @model_validator(mode="after")
    def date_range_is_valid(self) -> "DartSearchQuery":
        if self.end_date < self.begin_date:
            raise ValueError("end_date must not precede begin_date")
        if self.corp_code is None and (self.end_date - self.begin_date).days > 92:
            raise ValueError("OpenDART searches without corp_code are limited to approximately three months")
        return self


class DartCorporation(ContractModel):
    corp_code: str = Field(pattern=r"^\d{8}$")
    corp_name: str
    corp_eng_name: str | None = None
    stock_code: str | None = None
    modify_date: date | None = None


class DartFiling(ContractModel):
    corp_cls: str | None = None
    corp_name: str
    corp_code: str = Field(pattern=r"^\d{8}$")
    stock_code: str | None = None
    report_name: str
    rcept_no: str = Field(pattern=r"^\d{14}$")
    filer_name: str | None = None
    receipt_date: date
    remarks: str | None = None

    @property
    def reported_as_amendment(self) -> bool:
        return _AMENDMENT_PREFIX.match(self.report_name) is not None

    @property
    def normalized_report_name(self) -> str:
        return re.sub(r"\s+", " ", _AMENDMENT_PREFIX.sub("", self.report_name)).strip()


class DartOpenApiClient:
    def __init__(
        self,
        http: HttpClient,
        api_key: str,
        *,
        base_url: str = DART_BASE_URL,
        api_max_retries: int = 2,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenDART API key is required")
        if api_max_retries < 0:
            raise ValueError("api_max_retries must be non-negative")
        self.http = http
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.api_max_retries = api_max_retries

    def list_filings(self, query: DartSearchQuery) -> list[DartFiling]:
        page = 1
        filings: list[DartFiling] = []
        while True:
            payload = self._get_json(
                "list.json",
                {
                    "corp_code": query.corp_code,
                    "bgn_de": query.begin_date.strftime("%Y%m%d"),
                    "end_de": query.end_date.strftime("%Y%m%d"),
                    "last_reprt_at": "Y" if query.final_only else "N",
                    "pblntf_ty": query.disclosure_type,
                    "pblntf_detail_ty": query.detail_type,
                    "sort": "date",
                    "sort_mth": "asc",
                    "page_no": page,
                    "page_count": 100,
                },
                no_data_is_empty=True,
            )
            if payload is None:
                return []
            for row in payload.get("list") or []:
                filings.append(self._parse_filing(row))
            total_page = int(payload.get("total_page") or 1)
            if page >= total_page:
                break
            page += 1
        unique = {filing.rcept_no: filing for filing in filings}
        return sorted(unique.values(), key=lambda item: (item.receipt_date, item.rcept_no))

    def download_document_archive(self, rcept_no: str, *, max_bytes: int | None = None) -> bytes:
        return self._get_zip(
            "document.xml",
            {"rcept_no": rcept_no},
            max_bytes=max_bytes,
        )

    def list_corporations(self, *, max_bytes: int | None = None) -> tuple[list[DartCorporation], bytes]:
        archive = self._get_zip("corpCode.xml", {}, max_bytes=max_bytes)
        members = extract_zip_members(archive, max_total_bytes=max_bytes or 256 * 1024 * 1024)
        xml_candidates = [(name, content) for name, content in members.items() if name.lower().endswith(".xml")]
        if not xml_candidates:
            raise ValueError("OpenDART corporation archive contains no XML file")
        _, xml_bytes = max(xml_candidates, key=lambda item: len(item[1]))
        parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False, huge_tree=False)
        root = etree.fromstring(xml_bytes, parser=parser)
        corporations: list[DartCorporation] = []
        for element in root.xpath(".//list"):
            values = {child.tag: (child.text or "").strip() for child in element}
            if not values.get("corp_code"):
                continue
            modify = values.get("modify_date")
            corporations.append(
                DartCorporation(
                    corp_code=values["corp_code"],
                    corp_name=values.get("corp_name", ""),
                    corp_eng_name=values.get("corp_eng_name") or None,
                    stock_code=values.get("stock_code") or None,
                    modify_date=datetime.strptime(modify, "%Y%m%d").date() if modify else None,
                )
            )
        return corporations, archive

    def _get_json(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        no_data_is_empty: bool = False,
    ) -> dict[str, Any] | None:
        for attempt in range(self.api_max_retries + 1):
            response = self.http.get(
                f"{self.base_url}/{endpoint}",
                query={"crtfc_key": self.api_key, **params},
                headers={"Accept": "application/json"},
            )
            try:
                payload = response.json()
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"OpenDART returned invalid JSON for {endpoint}") from exc
            status = str(payload.get("status") or "")
            if status == "000":
                return payload
            if status == "013" and no_data_is_empty:
                return None
            if status in {"800", "900"} and attempt < self.api_max_retries:
                time_module.sleep(min(30.0, 1.5 * (2**attempt)))
                continue
            raise DartApiError(status or "UNKNOWN", str(payload.get("message") or "unknown error"))
        raise AssertionError("unreachable OpenDART retry state")

    def _get_zip(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        max_bytes: int | None,
    ) -> bytes:
        for attempt in range(self.api_max_retries + 1):
            response = self.http.get(
                f"{self.base_url}/{endpoint}",
                query={"crtfc_key": self.api_key, **params},
                max_bytes=max_bytes,
            )
            if zipfile.is_zipfile(io.BytesIO(response.content)):
                return response.content
            status, message = _parse_dart_error(response.content)
            if status in {"800", "900"} and attempt < self.api_max_retries:
                time_module.sleep(min(30.0, 1.5 * (2**attempt)))
                continue
            raise DartApiError(status or "INVALID_ARCHIVE", message or "response is not a ZIP archive")
        raise AssertionError("unreachable OpenDART ZIP retry state")

    @staticmethod
    def _parse_filing(row: dict[str, Any]) -> DartFiling:
        receipt = str(row.get("rcept_dt") or "")
        return DartFiling(
            corp_cls=str(row.get("corp_cls") or "") or None,
            corp_name=str(row.get("corp_name") or ""),
            corp_code=str(row.get("corp_code") or ""),
            stock_code=str(row.get("stock_code") or "") or None,
            report_name=str(row.get("report_nm") or ""),
            rcept_no=str(row.get("rcept_no") or ""),
            filer_name=str(row.get("flr_nm") or "") or None,
            receipt_date=datetime.strptime(receipt, "%Y%m%d").date(),
            remarks=str(row.get("rm") or "") or None,
        )


class DartCollector:
    def __init__(
        self,
        client: DartOpenApiClient,
        store: BronzeFilingStore,
        *,
        max_archive_bytes: int = 256 * 1024 * 1024,
        max_extracted_bytes: int = 1024 * 1024 * 1024,
    ) -> None:
        if max_archive_bytes <= 0 or max_extracted_bytes <= 0:
            raise ValueError("DART archive size limits must be positive")
        self.client = client
        self.store = store
        self.max_archive_bytes = max_archive_bytes
        self.max_extracted_bytes = max_extracted_bytes

    def collect(
        self,
        *,
        begin_date: date,
        end_date: date,
        corp_codes: list[str] | None = None,
        stock_codes: list[str] | None = None,
        all_companies: bool = False,
        report_kinds: set[str] | None = None,
        final_only: bool = False,
        refresh: bool = False,
        max_filings: int | None = None,
    ) -> CollectionResult:
        started = datetime.now(timezone.utc)
        requested_corp_codes = list(dict.fromkeys(value.strip() for value in (corp_codes or [])))
        requested_stock_codes = list(dict.fromkeys(value.strip() for value in (stock_codes or [])))
        invalid_corp_codes = [value for value in requested_corp_codes if not re.fullmatch(r"\d{8}", value)]
        invalid_stock_codes = [value for value in requested_stock_codes if not re.fullmatch(r"\d{6}", value)]
        if invalid_corp_codes:
            raise ValueError(f"invalid DART corporation codes: {invalid_corp_codes}")
        if invalid_stock_codes:
            raise ValueError(f"invalid DART stock codes: {invalid_stock_codes}")
        if all_companies and (requested_corp_codes or requested_stock_codes):
            raise ValueError("--all-companies cannot be combined with specific DART company codes")
        if not requested_corp_codes and not requested_stock_codes and not all_companies:
            raise ValueError("provide --corp-code/--stock-code or explicitly select --all-companies")
        if report_kinds:
            unknown = report_kinds - set(_REPORT_KINDS)
            if unknown:
                raise ValueError(f"unsupported DART report kinds: {sorted(unknown)}")

        if requested_stock_codes:
            corporations, _ = self.client.list_corporations(max_bytes=self.max_archive_bytes)
            by_stock = {item.stock_code: item for item in corporations if item.stock_code}
            missing = [stock for stock in requested_stock_codes if stock not in by_stock]
            if missing:
                raise ValueError(f"DART stock codes not found in corporation catalog: {missing}")
            requested_corp_codes.extend(by_stock[stock].corp_code for stock in requested_stock_codes)
            requested_corp_codes = list(dict.fromkeys(requested_corp_codes))

        searches = requested_corp_codes or [None]
        discovered: dict[str, DartFiling] = {}
        for corp_code in searches:
            query = DartSearchQuery(
                begin_date=begin_date,
                end_date=end_date,
                corp_code=corp_code,
                disclosure_type="A",
                final_only=final_only,
            )
            for filing in self.client.list_filings(query):
                if report_kinds and not any(
                    label in filing.normalized_report_name for kind, label in _REPORT_KINDS.items() if kind in report_kinds
                ):
                    continue
                discovered[filing.rcept_no] = filing
        filings = sorted(discovered.values(), key=lambda item: (item.receipt_date, item.rcept_no))
        if max_filings is not None:
            if max_filings <= 0:
                raise ValueError("max_filings must be positive")
            filings = filings[:max_filings]
        amendment_links = _dart_amendment_links(filings)

        collected: list[CollectedFiling] = []
        failures: list[CollectionFailure] = []
        for filing in filings:
            current = self.store.current(SourceType.DART, filing.rcept_no)
            if current is not None and not refresh:
                collected.append(current)
                continue
            try:
                archive = self.client.download_document_archive(
                    filing.rcept_no,
                    max_bytes=self.max_archive_bytes,
                )
                members = extract_zip_members(
                    archive,
                    max_total_bytes=self.max_extracted_bytes,
                )
                primary_member = select_dart_primary_document(members, filing.rcept_no)
                amends = amendment_links.get(filing.rcept_no)
                report_date = _period_end_from_report_name(filing.normalized_report_name)
                descriptor = FilingDescriptor(
                    source_type=SourceType.DART,
                    source_document_id=filing.rcept_no,
                    issuer_id=filing.corp_code,
                    issuer_name=filing.corp_name,
                    ticker=filing.stock_code,
                    report_name=filing.report_name,
                    filing_date=filing.receipt_date,
                    report_date=report_date,
                    available_at=datetime.combine(filing.receipt_date, time.max, tzinfo=_KST),
                    availability_precision=AvailabilityPrecision.DAY,
                    availability_source="DART_LIST_RCEPT_DATE_CONSERVATIVE_EOD_KST",
                    primary_document_name=primary_member,
                    primary_document_url=f"{DART_VIEWER_URL}?rcpNo={filing.rcept_no}",
                    archive_url=f"{self.client.base_url}/document.xml?rcept_no={filing.rcept_no}",
                    is_amendment=amends is not None,
                    amends_document_id=amends,
                    source_specific={
                        "rcept_no": filing.rcept_no,
                        "corp_code": filing.corp_code,
                        "stock_code": filing.stock_code,
                        "corp_cls": filing.corp_cls,
                        "report_name": filing.report_name,
                        "normalized_report_name": filing.normalized_report_name,
                        "filer_name": filing.filer_name,
                        "rcept_dt": filing.receipt_date.strftime("%Y%m%d"),
                        "remarks": filing.remarks,
                        "reported_as_amendment": filing.reported_as_amendment,
                        "amendment_link_status": "LINKED" if amends else ("UNRESOLVED" if filing.reported_as_amendment else "NOT_APPLICABLE"),
                        "archive_sha256": hashlib.sha256(archive).hexdigest(),
                        "archive_members": sorted(members),
                    },
                )
                stored_files = {"original.zip": archive}
                stored_files.update({f"documents/{name}": content for name, content in members.items()})
                collected.append(
                    self.store.save(
                        descriptor,
                        files=stored_files,
                        primary_path=f"documents/{primary_member}",
                    )
                )
            except Exception as exc:
                failures.append(
                    CollectionFailure(
                        source_document_id=filing.rcept_no,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
        return CollectionResult(
            source_type=SourceType.DART,
            started_at=started,
            completed_at=datetime.now(timezone.utc),
            query={
                "begin_date": begin_date.isoformat(),
                "end_date": end_date.isoformat(),
                "corp_codes": requested_corp_codes,
                "stock_codes": requested_stock_codes,
                "all_companies": all_companies,
                "report_kinds": sorted(report_kinds or set(_REPORT_KINDS)),
                "final_only": final_only,
                "refresh": refresh,
                "max_filings": max_filings,
            },
            discovered_count=len(filings),
            filings=collected,
            failures=failures,
        )


def extract_zip_members(
    archive: bytes,
    *,
    max_total_bytes: int,
    max_entries: int = 10_000,
) -> dict[str, bytes]:
    if max_total_bytes <= 0 or max_entries <= 0:
        raise ValueError("ZIP limits must be positive")
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        infos = [info for info in source.infolist() if not info.is_dir()]
        if len(infos) > max_entries:
            raise ValueError(f"ZIP contains {len(infos)} files, above limit {max_entries}")
        total = sum(info.file_size for info in infos)
        if total > max_total_bytes:
            raise ValueError(f"ZIP expands to {total} bytes, above limit {max_total_bytes}")
        members: dict[str, bytes] = {}
        for info in infos:
            if info.flag_bits & 0x1:
                raise ValueError(f"encrypted ZIP member is unsupported: {info.filename}")
            safe = safe_relative_path(info.filename).as_posix()
            if safe in members:
                raise ValueError(f"duplicate ZIP member path: {safe}")
            content = source.read(info)
            if len(content) != info.file_size:
                raise ValueError(f"ZIP member size mismatch: {safe}")
            members[safe] = content
    if not members:
        raise ValueError("ZIP archive contains no files")
    return members


def select_dart_primary_document(members: dict[str, bytes], rcept_no: str) -> str:
    candidates = [
        (name, content)
        for name, content in members.items()
        if Path(name).suffix.lower() in {".html", ".htm", ".xhtml", ".xml"}
    ]
    if not candidates:
        raise ValueError("DART archive contains no HTML/XML document")

    def score(item: tuple[str, bytes]) -> tuple[int, int, int, str]:
        name, content = item
        basename = Path(name).stem.lower()
        exact_receipt = int(basename == rcept_no.lower())
        contains_receipt = int(rcept_no.lower() in basename)
        return (exact_receipt, contains_receipt, len(content), name)

    return max(candidates, key=score)[0]


def _parse_dart_error(content: bytes) -> tuple[str | None, str | None]:
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True)
        root = etree.fromstring(content, parser=parser)
    except etree.XMLSyntaxError:
        return None, None
    status = root.findtext(".//status")
    message = root.findtext(".//message")
    return status.strip() if status else None, message.strip() if message else None


def _dart_amendment_links(filings: list[DartFiling]) -> dict[str, str]:
    originals: dict[tuple[str, str], list[DartFiling]] = {}
    for filing in filings:
        if not filing.reported_as_amendment:
            originals.setdefault((filing.corp_code, filing.normalized_report_name), []).append(filing)
    links: dict[str, str] = {}
    for filing in filings:
        if not filing.reported_as_amendment:
            continue
        candidates = [
            item
            for item in originals.get((filing.corp_code, filing.normalized_report_name), [])
            if (item.receipt_date, item.rcept_no) < (filing.receipt_date, filing.rcept_no)
        ]
        if candidates:
            links[filing.rcept_no] = max(candidates, key=lambda item: (item.receipt_date, item.rcept_no)).rcept_no
    return links


def _period_end_from_report_name(report_name: str) -> date | None:
    match = re.search(r"\((\d{4})[.\-/](\d{1,2})(?:[.\-/](\d{1,2}))?\)", report_name)
    if not match:
        return None
    year, month = int(match.group(1)), int(match.group(2))
    day = int(match.group(3)) if match.group(3) else calendar.monthrange(year, month)[1]
    try:
        return date(year, month, day)
    except ValueError:
        return None
