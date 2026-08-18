from __future__ import annotations

import hashlib
import io
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from lxml import html
from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


SEOUL = ZoneInfo("Asia/Seoul")
DART_SEARCH_URL = "https://dart.fss.or.kr/dsab007/detailSearch.ax"
DART_VIEWER_URL = "https://dart.fss.or.kr/dsaf001/main.do"
DART_DOWNLOAD_MAIN = "https://dart.fss.or.kr/pdf/download/main.do"
DART_IFRS_DOWNLOAD = "https://dart.fss.or.kr/pdf/download/ifrs.do"
_ANNUAL_RE = re.compile(r"사업보고서\s*\((?P<year>\d{4})[.]12\)")
_RECEIPT_RE = re.compile(r"rcpNo=(\d{14})")
_DCM_RE = re.compile(r"openPdfDownload\(['\"](?P<rcp>\d{14})['\"],\s*['\"](?P<dcm>\d+)['\"]\)")


class DartWebAnnualFiling(ContractModel):
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    corp_code: str = Field(pattern=r"^[0-9]{8}$")
    corp_name: str = Field(min_length=1)
    rcept_no: str = Field(pattern=r"^[0-9]{14}$")
    rcept_date: date
    available_at: datetime
    report_name: str = Field(min_length=1)
    fiscal_year: int = Field(ge=1990, le=2200)
    is_amendment: bool = False
    viewer_url: str
    dcm_no: str | None = Field(default=None, pattern=r"^[0-9]+$")

    @model_validator(mode="after")
    def receipt_controls_availability(self) -> "DartWebAnnualFiling":
        expected = datetime.combine(self.rcept_date, datetime_time.max, tzinfo=SEOUL)
        if self.available_at != expected:
            raise ValueError("DART web filing availability must use conservative receipt-date EOD")
        return self


@dataclass
class DartWebResponse:
    content: bytes
    status_code: int
    headers: dict[str, str]


class DartWebClient:
    """Public DART website client for immutable filed XBRL archives.

    This is intentionally separate from the OpenDART API client. It records
    receipt-number timestamps and downloads the exact filing artifact shown by
    the public viewer without requiring an API credential.
    """

    def __init__(self, *, requests_per_second: float = 2.0, session: Any | None = None) -> None:
        if requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive")
        if session is None:
            import requests

            session = requests.Session()
        self.session = session
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
                "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.5",
            }
        )
        self.minimum_interval = 1.0 / requests_per_second
        self.last_request = 0.0
        self.pdf_session_initialized = False

    def _request(self, method: str, url: str, **kwargs: object) -> Any:
        remaining = self.minimum_interval - (time.monotonic() - self.last_request)
        if remaining > 0:
            time.sleep(remaining)
        try:
            response = self.session.request(method, url, timeout=60, **kwargs)
        finally:
            self.last_request = time.monotonic()
        response.raise_for_status()
        return response

    def list_annual_filings(
        self,
        *,
        ticker: str,
        corp_code: str,
        corp_name: str,
        begin_date: date,
        end_date: date,
    ) -> list[DartWebAnnualFiling]:
        payload = {
            "currentPage": "1",
            "maxResults": "100",
            "maxLinks": "10",
            "sort": "date",
            "series": "asc",
            "textCrpCik": corp_code,
            "textCrpNm": corp_name,
            "startDate": begin_date.strftime("%Y%m%d"),
            "endDate": end_date.strftime("%Y%m%d"),
            "publicType": "A001",
            "finalReport": "",
        }
        response = self._request("POST", DART_SEARCH_URL, data=payload)
        # The DART search response declares UTF-8 in the HTTP header but does
        # not consistently repeat it in an HTML meta tag.  Parse requests'
        # decoded text so Korean report names remain Unicode and the annual-
        # report pattern cannot silently miss every filing.
        root = html.fromstring(response.text)
        filings: dict[str, DartWebAnnualFiling] = {}
        for anchor in root.xpath("//a[contains(@href, 'rcpNo=')]"):
            match = _RECEIPT_RE.search(anchor.get("href") or "")
            if not match:
                continue
            report_name = " ".join(anchor.text_content().split())
            annual = _ANNUAL_RE.search(report_name)
            if not annual:
                continue
            rcept_no = match.group(1)
            rcept_date = datetime.strptime(rcept_no[:8], "%Y%m%d").date()
            filings[rcept_no] = DartWebAnnualFiling(
                ticker=ticker.zfill(6),
                corp_code=corp_code,
                corp_name=corp_name,
                rcept_no=rcept_no,
                rcept_date=rcept_date,
                available_at=datetime.combine(rcept_date, datetime_time.max, tzinfo=SEOUL),
                report_name=report_name,
                fiscal_year=int(annual.group("year")),
                is_amendment=bool(re.search(r"정정", report_name)),
                viewer_url=f"{DART_VIEWER_URL}?rcpNo={rcept_no}",
            )
        return sorted(filings.values(), key=lambda item: (item.rcept_date, item.rcept_no))

    def resolve_dcm_no(self, filing: DartWebAnnualFiling) -> str:
        response = self._request("GET", filing.viewer_url)
        text = response.text
        candidates = [match for match in _DCM_RE.finditer(text) if match.group("rcp") == filing.rcept_no]
        if not candidates:
            raise ValueError(f"DART viewer has no PDF/XBRL download number: {filing.rcept_no}")
        values = {match.group("dcm") for match in candidates}
        if len(values) != 1:
            raise ValueError(f"DART viewer has ambiguous download numbers: {filing.rcept_no}")
        return next(iter(values))

    def download_ifrs_archive(self, filing: DartWebAnnualFiling, *, dcm_no: str) -> bytes:
        main_url = f"{DART_DOWNLOAD_MAIN}?rcp_no={filing.rcept_no}&dcm_no={dcm_no}"
        if not self.pdf_session_initialized:
            self._request("GET", main_url, headers={"Referer": filing.viewer_url})
            self.pdf_session_initialized = True
        url = f"{DART_IFRS_DOWNLOAD}?rcp_no={filing.rcept_no}&dcm_no={dcm_no}&lang=ko"
        response = self._request("GET", url, headers={"Referer": main_url, "Accept": "*/*"})
        content = response.content
        if not content:
            # DART can rotate its PDF-session cookie. Reinitialize once.
            self._request("GET", main_url, headers={"Referer": filing.viewer_url})
            response = self._request("GET", url, headers={"Referer": main_url, "Accept": "*/*"})
            content = response.content
        if not zipfile.is_zipfile(io.BytesIO(content)):
            raise ValueError(f"DART IFRS response is not a ZIP archive: {filing.rcept_no}")
        return content


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
