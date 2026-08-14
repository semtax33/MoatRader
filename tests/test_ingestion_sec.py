from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from moatrader.ingestion import (
    BronzeFilingStore,
    HttpResponse,
    SecEdgarClient,
    SecEdgarCollector,
    normalize_cik,
    validate_sec_user_agent,
)


def _columns() -> dict[str, list[Any]]:
    return {
        "accessionNumber": ["0000320193-25-000001", "0000320193-25-000002"],
        "filingDate": ["2025-01-31", "2025-02-03"],
        "reportDate": ["2024-12-28", "2024-12-28"],
        "acceptanceDateTime": ["2025-01-31T21:00:00.000Z", "2025-02-03T21:01:00.000Z"],
        "form": ["10-K", "10-K/A"],
        "primaryDocument": ["aapl-20241228.htm", "aapl-20241228x10ka.htm"],
        "primaryDocDescription": ["FORM 10-K", "FORM 10-K/A"],
        "fileNumber": ["001-36743", "001-36743"],
        "items": ["", ""],
        "size": [100, 120],
        "isXBRL": [1, 1],
        "isInlineXBRL": [1, 1],
    }


class FakeSecHttp:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        query: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        max_bytes: int | None = None,
    ) -> HttpResponse:
        del query, headers, max_bytes
        self.calls.append(url)
        if url.endswith("company_tickers.json"):
            payload: Any = {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}}
            content = json.dumps(payload).encode("utf-8")
        elif url.endswith("CIK0000320193.json"):
            payload = {
                "cik": "0000320193",
                "name": "Apple Inc.",
                "tickers": ["AAPL"],
                "exchanges": ["Nasdaq"],
                "filings": {"recent": _columns(), "files": []},
            }
            content = json.dumps(payload).encode("utf-8")
        elif url.endswith(".htm"):
            content = f"<html><body><h1>{Path(url).name}</h1></body></html>".encode("utf-8")
        elif url.endswith(".txt"):
            content = b"<SEC-DOCUMENT><SEC-HEADER>complete submission</SEC-HEADER>"
        else:
            raise AssertionError(url)
        return HttpResponse(url=url, status_code=200, headers={}, content=content)


def test_sec_collector_uses_official_urls_exact_acceptance_and_complete_submission(tmp_path: Path) -> None:
    http = FakeSecHttp()
    store = BronzeFilingStore(tmp_path / "bronze")
    client = SecEdgarClient(http, declared_user_agent="MoatRader admin@example.com")
    collector = SecEdgarCollector(client, store, availability_lag_minutes=5)

    result = collector.collect(
        begin_date=date(2025, 1, 1),
        end_date=date(2025, 2, 28),
        tickers=["aapl"],
        forms={"10-K", "10-K/A"},
    )

    assert result.discovered_count == 2
    assert result.downloaded_count == 2
    assert not result.failures
    amended = next(item for item in result.filings if item.source_document_id.endswith("000002"))
    metadata = json.loads(Path(amended.metadata_path).read_text(encoding="utf-8"))
    assert metadata["published_at"] == "2025-02-03T21:01:00+00:00"
    assert metadata["available_at"] == "2025-02-03T21:06:00+00:00"
    assert metadata["availability_precision"] == "INFERRED"
    assert metadata["is_amendment"] is True
    assert metadata["amends_document_id"] == "0000320193-25-000001"
    assert metadata["source_specific"]["reported_as_amendment"] is True
    assert (Path(amended.version_directory) / "original-submission.txt").is_file()
    assert any(
        url
        == "https://www.sec.gov/Archives/edgar/data/320193/000032019325000002/aapl-20241228x10ka.htm"
        for url in http.calls
    )
    assert any(url.endswith("/0000320193-25-000002.txt") for url in http.calls)


def test_sec_validation_normalizes_cik_and_requires_declared_contact() -> None:
    assert normalize_cik("CIK320193") == "0000320193"
    validate_sec_user_agent("MoatRader admin@example.com")

    with pytest.raises(ValueError, match="contact email"):
        validate_sec_user_agent("anonymous-bot")


def test_sec_history_loads_official_additional_submission_files() -> None:
    class HistoricalHttp:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def get(self, url: str, **_kwargs: Any) -> HttpResponse:
            self.calls.append(url)
            if url.endswith("CIK0000320193.json"):
                payload: Any = {
                    "name": "Apple Inc.",
                    "tickers": ["AAPL"],
                    "exchanges": ["Nasdaq"],
                    "filings": {
                        "recent": {key: [] for key in _columns()},
                        "files": [
                            {
                                "name": "CIK0000320193-submissions-001.json",
                                "filingFrom": "2009-01-01",
                                "filingTo": "2011-12-31",
                            }
                        ],
                    },
                }
            elif url.endswith("CIK0000320193-submissions-001.json"):
                columns = _columns()
                columns["filingDate"] = ["2010-01-31", "2010-02-03"]
                columns["reportDate"] = ["2009-12-28", "2009-12-28"]
                columns["acceptanceDateTime"] = [
                    "2010-01-31T21:00:00.000Z",
                    "2010-02-03T21:01:00.000Z",
                ]
                payload = columns
            else:
                raise AssertionError(url)
            return HttpResponse(url=url, status_code=200, headers={}, content=json.dumps(payload).encode())

    http = HistoricalHttp()
    client = SecEdgarClient(http, declared_user_agent="MoatRader admin@example.com")

    history = client.submission_history(
        "320193",
        begin_date=date(2010, 1, 1),
        end_date=date(2010, 12, 31),
    )

    assert len(history.filings) == 2
    assert any(url.endswith("CIK0000320193-submissions-001.json") for url in http.calls)
