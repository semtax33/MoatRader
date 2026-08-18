from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from moatrader.ingestion.dart_web import DartWebClient


SEOUL = ZoneInfo("Asia/Seoul")


class _Response:
    def __init__(self, content: bytes) -> None:
        self.content = content
        self.text = content.decode("utf-8")

    def raise_for_status(self) -> None:
        return None


class _Session:
    def __init__(self, responses: list[_Response]) -> None:
        self.headers: dict[str, str] = {}
        self.responses = responses
        self.calls: list[tuple[str, str, dict[str, object]]] = []

    def request(self, method: str, url: str, **kwargs: object) -> _Response:
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def test_dart_web_search_keeps_original_and_amendment_with_eod_availability() -> None:
    body = b"""<html><head><meta charset='utf-8'></head><body>
      <a href='/dsaf001/main.do?rcpNo=20200330004254'>\xec\x82\xac\xec\x97\x85\xeb\xb3\xb4\xea\xb3\xa0\xec\x84\x9c (2019.12)</a>
      <a href='/dsaf001/main.do?rcpNo=20200401001922'>[\xec\xa0\x95\xec\xa0\x95] \xec\x82\xac\xec\x97\x85\xeb\xb3\xb4\xea\xb3\xa0\xec\x84\x9c (2019.12)</a>
      <a href='/dsaf001/main.do?rcpNo=20200515000001'>\xeb\xb6\x84\xea\xb8\xb0\xeb\xb3\xb4\xea\xb3\xa0\xec\x84\x9c (2020.03)</a>
    </body></html>"""
    session = _Session([_Response(body)])
    client = DartWebClient(requests_per_second=1000, session=session)

    filings = client.list_annual_filings(
        ticker="5930",
        corp_code="00126380",
        corp_name="Samsung Electronics",
        begin_date=date(2020, 1, 1),
        end_date=date(2020, 12, 31),
    )

    assert [item.rcept_no for item in filings] == ["20200330004254", "20200401001922"]
    assert filings[0].ticker == "005930"
    assert filings[0].is_amendment is False
    assert filings[1].is_amendment is True
    assert filings[0].available_at == datetime.combine(
        date(2020, 3, 30), time.max, tzinfo=SEOUL
    )
    assert session.calls[0][2]["data"]["finalReport"] == ""


def test_dart_web_resolves_exact_download_number() -> None:
    viewer = b"<script>openPdfDownload('20200330004254', '7222135');</script>"
    session = _Session([_Response(viewer)])
    client = DartWebClient(requests_per_second=1000, session=session)
    from moatrader.ingestion.dart_web import DartWebAnnualFiling

    record = DartWebAnnualFiling(
        ticker="005930",
        corp_code="00126380",
        corp_name="삼성전자",
        rcept_no="20200330004254",
        rcept_date=date(2020, 3, 30),
        available_at=datetime.combine(date(2020, 3, 30), time.max, tzinfo=SEOUL),
        report_name="사업보고서 (2019.12)",
        fiscal_year=2019,
        viewer_url="https://dart.fss.or.kr/dsaf001/main.do?rcpNo=20200330004254",
    )

    assert client.resolve_dcm_no(record) == "7222135"
