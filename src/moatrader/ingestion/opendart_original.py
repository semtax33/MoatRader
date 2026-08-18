from __future__ import annotations

import calendar
import hashlib
import io
import re
import zipfile
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from lxml import etree, html
from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.ingestion.dart import (
    DART_BASE_URL,
    DartApiError,
    DartOpenApiClient,
    DartSearchQuery,
    extract_zip_members,
)
from moatrader.ingestion.http import HttpClient


SEOUL = ZoneInfo("Asia/Seoul")
_ANNUAL_PERIOD = re.compile(r"사업보고서\s*\((?P<year>\d{4})[.](?P<month>\d{2})\)")
_TEXT_SUFFIXES = {".htm", ".html", ".xhtml", ".xml", ".txt"}


class OpenDartAnnualFiling(ContractModel):
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    corp_code: str = Field(pattern=r"^[0-9]{8}$")
    corp_name: str = Field(min_length=1)
    rcept_no: str = Field(pattern=r"^[0-9]{14}$")
    rcept_date: date
    available_at: datetime
    report_name: str = Field(min_length=1)
    normalized_report_name: str = Field(min_length=1)
    fiscal_period_end: date
    is_amendment: bool
    amends_rcept_no: str | None = Field(default=None, pattern=r"^[0-9]{14}$")
    viewer_url: str

    @model_validator(mode="after")
    def conservative_receipt_availability(self) -> "OpenDartAnnualFiling":
        expected = datetime.combine(self.rcept_date, time.max, tzinfo=SEOUL)
        if self.available_at != expected:
            raise ValueError("OpenDART filing must become eligible at receipt-date EOD Seoul")
        return self


class OriginalEvidenceMember(ContractModel):
    source_id: str = Field(min_length=1)
    rcept_no: str = Field(pattern=r"^[0-9]{14}$")
    archive_member: str = Field(min_length=1)
    available_at: datetime
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    char_count: int = Field(gt=0)
    text_file: str = Field(min_length=1)


def _parse_api_error(content: bytes) -> tuple[str, str]:
    try:
        parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=True)
        root = etree.fromstring(content, parser=parser)
    except etree.XMLSyntaxError:
        return "INVALID_ARCHIVE", "response is not a ZIP archive"
    status = (root.findtext(".//status") or "INVALID_ARCHIVE").strip()
    message = (root.findtext(".//message") or "response is not a ZIP archive").strip()
    return status, message


class OpenDartOriginalClient:
    """Official OpenDART source client for original filing and XBRL ZIPs.

    The key is used only as an HTTP query value.  ResilientHttpClient redacts
    it from URLs retained in errors, and no model or manifest contains it.
    """

    def __init__(self, http: HttpClient, api_key: str) -> None:
        key = api_key.strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", key):
            raise ValueError("OpenDART API key must contain 40 hexadecimal characters")
        self.http = http
        self._api_key = key
        self._base = DartOpenApiClient(http, key)

    def list_annual_filings(
        self,
        *,
        ticker: str,
        corp_code: str,
        begin_date: date,
        end_date: date,
    ) -> list[OpenDartAnnualFiling]:
        rows = self._base.list_filings(
            DartSearchQuery(
                begin_date=begin_date,
                end_date=end_date,
                corp_code=corp_code,
                disclosure_type="A",
                final_only=False,
            )
        )
        result: list[OpenDartAnnualFiling] = []
        for row in rows:
            matched = _ANNUAL_PERIOD.search(row.normalized_report_name)
            if matched is None:
                continue
            year = int(matched.group("year"))
            month = int(matched.group("month"))
            period_end = date(year, month, calendar.monthrange(year, month)[1])
            result.append(
                OpenDartAnnualFiling(
                    ticker=ticker.zfill(6),
                    corp_code=corp_code,
                    corp_name=row.corp_name,
                    rcept_no=row.rcept_no,
                    rcept_date=row.receipt_date,
                    available_at=datetime.combine(row.receipt_date, time.max, tzinfo=SEOUL),
                    report_name=row.report_name,
                    normalized_report_name=row.normalized_report_name,
                    fiscal_period_end=period_end,
                    is_amendment=row.reported_as_amendment,
                    viewer_url=f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={row.rcept_no}",
                )
            )
        originals: dict[str, list[OpenDartAnnualFiling]] = {}
        for filing in result:
            if not filing.is_amendment:
                originals.setdefault(filing.normalized_report_name, []).append(filing)
        linked: list[OpenDartAnnualFiling] = []
        for filing in result:
            if not filing.is_amendment:
                linked.append(filing)
                continue
            candidates = [
                item
                for item in originals.get(filing.normalized_report_name, [])
                if (item.rcept_date, item.rcept_no) < (filing.rcept_date, filing.rcept_no)
            ]
            prior = max(candidates, key=lambda item: (item.rcept_date, item.rcept_no)) if candidates else None
            linked.append(
                filing.model_copy(update={"amends_rcept_no": prior.rcept_no if prior else None})
            )
        return sorted(linked, key=lambda item: (item.rcept_date, item.rcept_no))

    def download_original_archive(self, rcept_no: str) -> bytes:
        return self._base.download_document_archive(rcept_no, max_bytes=512 * 1024 * 1024)

    def download_xbrl_archive(self, rcept_no: str) -> bytes:
        response = self.http.get(
            f"{DART_BASE_URL}/fnlttXbrl.xml",
            query={
                "crtfc_key": self._api_key,
                "rcept_no": rcept_no,
                "reprt_code": "11011",
            },
            max_bytes=512 * 1024 * 1024,
        )
        if zipfile.is_zipfile(io.BytesIO(response.content)):
            return response.content
        status, message = _parse_api_error(response.content)
        raise DartApiError(status, message)


def _visible_text(content: bytes, *, suffix: str) -> str:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            decoded = content.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        return ""
    if suffix == ".txt":
        lines = decoded.splitlines()
    else:
        try:
            # Decode before parsing. DART documents do not always repeat the
            # response/member encoding in an HTML meta declaration, and raw
            # byte parsing can otherwise turn Korean UTF-8 into mojibake.
            # lxml rejects a Unicode string that still contains a byte-level
            # XML encoding declaration, so remove that declaration only after
            # successful decoding.
            decoded = re.sub(r"^\s*<\?xml[^>]*\?>", "", decoded, count=1, flags=re.I)
            root = html.fromstring(decoded)
        except (etree.ParserError, ValueError):
            return ""
        for node in root.xpath("//script|//style|//noscript"):
            node.drop_tree()
        lines = root.xpath("//text()")
    normalized = [" ".join(str(line).replace("\xa0", " ").split()) for line in lines]
    return "\n".join(line for line in normalized if line).strip()


def extract_original_evidence(
    archive: bytes,
    *,
    rcept_no: str,
    available_at: datetime,
) -> tuple[list[OriginalEvidenceMember], dict[str, str]]:
    members = extract_zip_members(archive, max_total_bytes=2 * 1024 * 1024 * 1024)
    records: list[OriginalEvidenceMember] = []
    texts: dict[str, str] = {}
    for member_name, content in sorted(members.items()):
        suffix = Path(member_name).suffix.lower()
        if suffix not in _TEXT_SUFFIXES:
            continue
        text = _visible_text(content, suffix=suffix)
        if not text:
            continue
        raw_hash = hashlib.sha256(content).hexdigest()
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        source_id = f"opendart:{rcept_no}:{raw_hash[:20]}"
        text_file = f"evidence/{len(records):04d}-{text_hash[:16]}.txt"
        records.append(
            OriginalEvidenceMember(
                source_id=source_id,
                rcept_no=rcept_no,
                archive_member=member_name,
                available_at=available_at,
                raw_sha256=raw_hash,
                text_sha256=text_hash,
                char_count=len(text),
                text_file=text_file,
            )
        )
        texts[text_file] = text
    if not records:
        raise ValueError(f"OpenDART original archive has no extractable text: {rcept_no}")
    return records, texts
