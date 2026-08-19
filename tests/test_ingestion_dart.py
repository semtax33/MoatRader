from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pytest

from moatrader.canonical.models import SourceType
from moatrader.ingestion import (
    BronzeFilingStore,
    DartCollector,
    DartOpenApiClient,
    HttpResponse,
    extract_zip_members,
    write_collected_universe_manifest,
)
from moatrader.runner import CompanyRunStatus, MoatUniverseRunner, UniverseRunConfig
from moatrader.universe import load_universe_manifest


def _zip(files: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()


class FakeDartHttp:
    def __init__(self, document_archives: dict[str, bytes]) -> None:
        self.document_archives = document_archives
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
        if url.endswith("/corpCode.xml"):
            body = _zip(
                {
                    "CORPCODE.xml": b"""<?xml version='1.0' encoding='UTF-8'?>
<result><list><corp_code>00126380</corp_code><corp_name>Sample</corp_name>
<corp_eng_name>Sample Inc.</corp_eng_name><stock_code>005930</stock_code>
<modify_date>20250101</modify_date></list>
<list><corp_code>00999999</corp_code><corp_name>Alpha Code Sample</corp_name>
<corp_eng_name>Alpha Code Sample Inc.</corp_eng_name><stock_code>00680K</stock_code>
<modify_date>20260101</modify_date></list></result>"""
                }
            )
        elif url.endswith("/list.json"):
            body = json.dumps(
                {
                    "status": "000",
                    "message": "정상",
                    "page_no": 1,
                    "total_page": 1,
                    "list": [
                        {
                            "corp_cls": "Y",
                            "corp_name": "Sample",
                            "corp_code": "00126380",
                            "stock_code": "005930",
                            "report_nm": "사업보고서 (2024.12)",
                            "rcept_no": "20250315000001",
                            "flr_nm": "Sample",
                            "rcept_dt": "20250315",
                            "rm": "연정",
                        },
                        {
                            "corp_cls": "Y",
                            "corp_name": "Sample",
                            "corp_code": "00126380",
                            "stock_code": "005930",
                            "report_nm": "[기재정정] 사업보고서 (2024.12)",
                            "rcept_no": "20250320000002",
                            "flr_nm": "Sample",
                            "rcept_dt": "20250320",
                            "rm": "연",
                        },
                    ],
                },
                ensure_ascii=False,
            ).encode("utf-8")
        elif url.endswith("/document.xml"):
            body = self.document_archives[str(params["rcept_no"])]
        else:
            raise AssertionError(url)
        return HttpResponse(url=url, status_code=200, headers={}, content=body)


def test_dart_collector_resolves_stock_downloads_raw_and_links_amendment(tmp_path: Path) -> None:
    originals = {
        "20250315000001": b"<html><body><h1>Business</h1><p>Original</p></body></html>",
        "20250320000002": b"<html><body><h1>Business</h1><p>Amended</p></body></html>",
    }
    http = FakeDartHttp(
        {
            rcept_no: _zip({f"{rcept_no}.xml": content, "attachment.xml": b"<attachment/>"})
            for rcept_no, content in originals.items()
        }
    )
    store = BronzeFilingStore(tmp_path / "bronze")
    collector = DartCollector(DartOpenApiClient(http, "x" * 40), store)

    result = collector.collect(
        begin_date=date(2025, 1, 1),
        end_date=date(2025, 3, 31),
        stock_codes=["005930"],
        report_kinds={"annual"},
    )

    assert result.discovered_count == 2
    assert not result.failures
    assert result.downloaded_count == 2
    amended = next(item for item in result.filings if item.source_document_id == "20250320000002")
    metadata = json.loads(Path(amended.metadata_path).read_text(encoding="utf-8"))
    assert metadata["available_at"] == "2025-03-20T23:59:59.999999+09:00"
    assert metadata["availability_precision"] == "DAY"
    assert metadata["is_amendment"] is True
    assert metadata["amends_document_id"] == "20250315000001"
    assert metadata["source_specific"]["archive_sha256"]
    assert "crtfc_key" not in json.dumps(metadata)
    assert Path(amended.input_path).read_bytes() == originals["20250320000002"]
    assert metadata["raw_sha256"] == hashlib.sha256(originals["20250320000002"]).hexdigest()
    assert (Path(amended.version_directory) / "original.zip").is_file()
    assert (Path(amended.version_directory) / "sha256.txt").is_file()

    manifest = write_collected_universe_manifest(store, tmp_path / "collected.csv")
    universe = load_universe_manifest(manifest)
    assert universe.companies[0].ticker == "005930"
    assert len(universe.companies[0].documents) == 2

    run_result = MoatUniverseRunner(
        config=UniverseRunConfig(
            run_id="collected-dart",
            as_of=datetime.fromisoformat("2025-03-21T00:00:00+09:00"),
            dry_run=True,
        ),
        output_directory=tmp_path / "runs",
        transport=None,
    ).run(universe, universe.companies)
    assert run_result.companies[0].status == CompanyRunStatus.PREPARED
    assert run_result.companies[0].source_document_ids == ["20250320000002", "20250315000001"]

    second = collector.collect(
        begin_date=date(2025, 1, 1),
        end_date=date(2025, 3, 31),
        stock_codes=["005930"],
        report_kinds={"annual"},
    )
    assert second.unchanged_count == 2
    document_calls = [url for url, _ in http.calls if url.endswith("/document.xml")]
    assert len(document_calls) == 2


def test_dart_collector_accepts_krx_alphanumeric_stock_codes(tmp_path: Path) -> None:
    archive = _zip({"20250315000001.xml": b"<html><body>Report</body></html>"})
    http = FakeDartHttp(
        {
            "20250315000001": archive,
            "20250320000002": archive,
        }
    )
    collector = DartCollector(
        DartOpenApiClient(http, "x" * 40),
        BronzeFilingStore(tmp_path / "bronze"),
    )

    collector.collect(
        begin_date=date(2025, 1, 1),
        end_date=date(2025, 3, 31),
        stock_codes=["00680k"],
        report_kinds={"annual"},
        max_filings=1,
    )

    list_calls = [params for url, params in http.calls if url.endswith("/list.json")]
    assert list_calls[0]["corp_code"] == "00999999"


def test_dart_zip_extraction_rejects_path_traversal() -> None:
    archive = _zip({"../escape.xml": b"bad"})

    with pytest.raises(ValueError, match="unsafe artifact path"):
        extract_zip_members(archive, max_total_bytes=1024)


def test_dart_zip_extraction_rejects_expansion_above_limit() -> None:
    archive = _zip({"large.xml": b"x" * 100})

    with pytest.raises(ValueError, match="above limit"):
        extract_zip_members(archive, max_total_bytes=99)
