from __future__ import annotations

from datetime import date
from pathlib import Path

from moatrader.canonical.models import SourceType
from moatrader.ingestion import (
    BronzeFilingStore,
    HttpResponse,
    KindCompanyIdentity,
    KindIrClient,
    KindIrCollector,
    KindIrMaterial,
)


LIST_HTML = """
<html><body><table>
<tr><th>번호</th><th>회사명</th><th>일자</th><th>제목</th><th>첨부</th></tr>
<tr><td>1</td>
  <td><a href="#" onclick="companysummary_open('05847'); return false;">리노공업</a></td>
  <td>2025-08-29</td>
  <td><a href="#" onclick="fnDetailView('17299','2'); return false;">기업설명회(IR) 개최</a></td>
  <td><a href="/external/dst/irReference/17299/리노 IR Book.pdf">리노 IR Book.pdf</a></td>
</tr>
</table></body></html>
"""


class FakeKindHttp:
    def __init__(self) -> None:
        self.forms: list[dict[str, object]] = []
        self.get_urls: list[str] = []

    def post_form(self, url: str, *, form: dict[str, object], **_kwargs: object) -> HttpResponse:
        self.forms.append(form)
        return HttpResponse(
            url=url,
            status_code=200,
            headers={"content-type": "text/html;charset=EUC-KR"},
            content=LIST_HTML.encode("euc-kr"),
        )

    def get(self, url: str, **_kwargs: object) -> HttpResponse:
        self.get_urls.append(url)
        return HttpResponse(
            url=url,
            status_code=200,
            headers={"content-type": "application/pdf"},
            content=b"%PDF-1.7\nfixture",
        )


def test_kind_client_parses_euc_kr_list_and_quotes_pdf_url() -> None:
    http = FakeKindHttp()
    client = KindIrClient(http)

    materials = client.search_materials(
        begin_date=date(2025, 8, 1),
        end_date=date(2025, 8, 31),
        page_size=100,
    )
    content = client.download_pdf(materials[0], max_bytes=1024)

    assert len(materials) == 1
    assert materials[0].source_document_id == "KINDIR_17299_1"
    assert materials[0].company_name == "리노공업"
    assert materials[0].kind_company_code == "05847"
    assert http.forms[0]["method"] == "searchIRMaterialsSub"
    assert http.forms[0]["pageIndex"] == 1
    assert "%EB%A6%AC%EB%85%B8" in http.get_urls[0]
    assert content.startswith(b"%PDF-")


def test_kind_collector_uses_conservative_day_level_pit_and_ir_namespace(tmp_path: Path) -> None:
    http = FakeKindHttp()
    store = BronzeFilingStore(tmp_path)
    result = KindIrCollector(KindIrClient(http), store).collect(
        begin_date=date(2025, 8, 1),
        end_date=date(2025, 8, 31),
        companies=[
            KindCompanyIdentity(
                ticker="058470",
                issuer_id="00369657",
                issuer_name="리노공업",
            )
        ],
    )

    assert result.source_type == SourceType.IR
    assert result.discovered_count == 1
    assert not result.failures
    collected = result.filings[0]
    assert Path(collected.input_path).suffix == ".pdf"
    assert "kind-ir" in Path(collected.input_path).parts
    metadata = Path(collected.metadata_path).read_text(encoding="utf-8")
    assert '"available_at": "2025-08-30T00:00:00+09:00"' in metadata
    assert '"statement_type": "MANAGEMENT_CLAIM"' in metadata
    assert '"jurisdiction": "KR"' in metadata


def test_kind_collector_selection_is_identity_only(tmp_path: Path) -> None:
    result = KindIrCollector(
        KindIrClient(FakeKindHttp()),
        BronzeFilingStore(tmp_path),
    ).collect(
        begin_date=date(2025, 8, 1),
        end_date=date(2025, 8, 31),
        companies=[
            KindCompanyIdentity(
                ticker="000000",
                issuer_id="00000000",
                issuer_name="다른회사",
            )
        ],
    )

    assert result.discovered_count == 0
    assert result.filings == []
    assert result.query["selection_policy"] == "availability-and-identity-only; no return data"


class FakeMaterialClient:
    search_url = "https://example.test/ir"

    def __init__(self, materials: list[KindIrMaterial]) -> None:
        self.materials = materials

    def search_materials(self, **_kwargs: object) -> list[KindIrMaterial]:
        return self.materials

    @staticmethod
    def download_pdf(_material: KindIrMaterial, *, max_bytes: int) -> bytes:
        assert max_bytes > 0
        return b"%PDF-1.7\nfixture"


def test_kind_collector_selects_latest_material_from_each_recent_year(
    tmp_path: Path,
) -> None:
    identity = KindCompanyIdentity(
        ticker="123456",
        issuer_id="issuer-1",
        issuer_name="테스트",
    )
    materials = [
        KindIrMaterial(
            ir_seq=str(100 + index),
            resoroom_type="1",
            company_name="테스트",
            listed_on=date(year, month, 1),
            title=f"IR {year}-{month}",
            attachment_index=1,
            attachment_name=f"{year}-{month}.pdf",
            attachment_url=f"https://example.test/{year}-{month}.pdf",
        )
        for index, (year, month) in enumerate(
            [(2021, 6), (2022, 3), (2022, 11), (2023, 5), (2024, 8)]
        )
    ]
    collector = KindIrCollector(
        FakeMaterialClient(materials),  # type: ignore[arg-type]
        BronzeFilingStore(tmp_path / "bronze"),
    )

    result = collector.collect(
        begin_date=date(2021, 1, 1),
        end_date=date(2024, 12, 31),
        companies=[identity],
        max_materials_per_company_per_year=1,
        max_years_per_company=3,
    )

    assert [filing.source_document_id for filing in result.filings] == [
        "KINDIR_102_1",
        "KINDIR_103_1",
        "KINDIR_104_1",
    ]
