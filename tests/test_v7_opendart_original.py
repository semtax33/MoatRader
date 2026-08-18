from __future__ import annotations

import io
import json
import zipfile
from datetime import date
from typing import Any

from moatrader.ingestion.http import HttpResponse
from moatrader.ingestion.opendart_original import (
    OpenDartOriginalClient,
    extract_original_evidence,
)


def _zip(files: dict[str, bytes]) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as output:
        for name, content in files.items():
            output.writestr(name, content)
    return target.getvalue()


class _Http:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get(
        self,
        url: str,
        *,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int | None = None,
    ) -> HttpResponse:
        del headers, max_bytes
        params = query or {}
        self.calls.append((url, params))
        if url.endswith("/list.json"):
            content = json.dumps(
                {
                    "status": "000",
                    "page_no": 1,
                    "total_page": 1,
                    "list": [
                        {
                            "corp_cls": "Y",
                            "corp_name": "Sample",
                            "corp_code": "00126380",
                            "stock_code": "005930",
                            "report_nm": "사업보고서 (2019.12)",
                            "rcept_no": "20200330000001",
                            "flr_nm": "Sample",
                            "rcept_dt": "20200330",
                            "rm": "",
                        },
                        {
                            "corp_cls": "Y",
                            "corp_name": "Sample",
                            "corp_code": "00126380",
                            "stock_code": "005930",
                            "report_nm": "[기재정정] 사업보고서 (2019.12)",
                            "rcept_no": "20200401000002",
                            "flr_nm": "Sample",
                            "rcept_dt": "20200401",
                            "rm": "",
                        },
                    ],
                },
                ensure_ascii=False,
            ).encode("utf-8")
        elif url.endswith("/document.xml"):
            content = _zip({"primary.xml": b"<html><body>Original text</body></html>"})
        elif url.endswith("/fnlttXbrl.xml"):
            content = _zip({"sample.xbrl": b"<xbrl/>"})
        else:
            raise AssertionError(url)
        return HttpResponse(url=url, status_code=200, headers={}, content=content)


def test_official_opendart_lists_original_and_amendment_and_downloads_both_archives() -> None:
    http = _Http()
    client = OpenDartOriginalClient(http, "a" * 40)
    filings = client.list_annual_filings(
        ticker="005930",
        corp_code="00126380",
        begin_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
    )

    assert len(filings) == 2
    assert filings[1].is_amendment is True
    assert filings[1].amends_rcept_no == filings[0].rcept_no
    assert filings[0].available_at.isoformat() == "2020-03-30T23:59:59.999999+09:00"
    assert zipfile.is_zipfile(io.BytesIO(client.download_original_archive(filings[0].rcept_no)))
    assert zipfile.is_zipfile(io.BytesIO(client.download_xbrl_archive(filings[0].rcept_no)))
    list_call = next(item for item in http.calls if item[0].endswith("/list.json"))
    assert list_call[1]["last_reprt_at"] == "N"


def test_original_archive_becomes_exact_citable_cutoff_text() -> None:
    archive = _zip(
        {
            "primary.xml": (
                "<?xml version='1.0' encoding='UTF-8'?>"
                "<html><body><h1>사업 내용</h1><p>신규 고객 매출이 증가했다.</p>"
                "<script>futureKnowledge()</script></body></html>"
            ).encode("utf-8"),
            "image.png": b"not text",
        }
    )
    filing = OpenDartOriginalClient(_Http(), "a" * 40).list_annual_filings(
        ticker="005930",
        corp_code="00126380",
        begin_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
    )[0]
    records, texts = extract_original_evidence(
        archive,
        rcept_no=filing.rcept_no,
        available_at=filing.available_at,
    )

    assert len(records) == 1
    text = texts[records[0].text_file]
    assert "신규 고객 매출이 증가했다." in text
    assert "futureKnowledge" not in text
    assert records[0].char_count == len(text)
    assert records[0].source_id.startswith(f"opendart:{filing.rcept_no}:")
