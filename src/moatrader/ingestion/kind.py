from __future__ import annotations

import re
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from urllib.parse import quote, urljoin, urlsplit, urlunsplit
from zoneinfo import ZoneInfo

from lxml import html
from pydantic import Field

from moatrader.canonical.models import AvailabilityPrecision, ContractModel, SourceType
from moatrader.ingestion.http import HttpClient
from moatrader.ingestion.models import (
    CollectedFiling,
    CollectionFailure,
    CollectionResult,
    FilingDescriptor,
)
from moatrader.ingestion.store import BronzeFilingStore


KIND_IR_URL = "https://kind.krx.co.kr/corpgeneral/irschedule.do"
KIND_BASE_URL = "https://kind.krx.co.kr"
_KST = ZoneInfo("Asia/Seoul")
_DETAIL_PATTERN = re.compile(r"fnDetailView\('(?P<seq>\d+)'\s*,\s*'(?P<room>\d+)'\)")
_COMPANY_PATTERN = re.compile(r"companysummary_open\('(?P<code>\d+)'\)")
_CORPORATE_WORDS = re.compile(
    r"(?:주식회사|\(주\)|㈜|유한회사|코스닥|코스피)",
    flags=re.IGNORECASE,
)
_NON_WORD = re.compile(r"[^0-9a-z가-힣]+", flags=re.IGNORECASE)


class KindCompanyIdentity(ContractModel):
    ticker: str = Field(min_length=1)
    issuer_id: str = Field(min_length=1)
    issuer_name: str = Field(min_length=1)
    kind_company_code: str | None = None


class KindIrMaterial(ContractModel):
    ir_seq: str = Field(pattern=r"^\d+$")
    resoroom_type: str = Field(pattern=r"^\d+$")
    kind_company_code: str | None = None
    company_name: str
    listed_on: date
    title: str
    attachment_index: int = Field(ge=1)
    attachment_name: str
    attachment_url: str

    @property
    def source_document_id(self) -> str:
        return f"KINDIR_{self.ir_seq}_{self.attachment_index}"


def normalize_company_name(value: str) -> str:
    return _NON_WORD.sub("", _CORPORATE_WORDS.sub("", value)).casefold()


def _quoted_url(value: str) -> str:
    parts = urlsplit(value)
    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            quote(parts.path, safe="/%:@!$&'()*+,;=-._~"),
            quote(parts.query, safe="=&%:@!$'()*+,;/?-._~"),
            parts.fragment,
        )
    )


class KindIrClient:
    def __init__(
        self,
        http: HttpClient,
        *,
        search_url: str = KIND_IR_URL,
        base_url: str = KIND_BASE_URL,
    ) -> None:
        self.http = http
        self.search_url = search_url
        self.base_url = base_url.rstrip("/")

    def search_materials(
        self,
        *,
        begin_date: date,
        end_date: date,
        page_size: int = 3000,
        max_pages: int | None = None,
    ) -> list[KindIrMaterial]:
        if end_date < begin_date:
            raise ValueError("end_date must not precede begin_date")
        if not 1 <= page_size <= 3000:
            raise ValueError("page_size must be between 1 and 3000")
        page = 1
        materials: dict[str, KindIrMaterial] = {}
        while max_pages is None or page <= max_pages:
            response = self.http.post_form(
                self.search_url,
                form=self._search_form(begin_date, end_date, page, page_size),
                headers={"Accept": "text/html, */*;q=0.8"},
                max_bytes=32 * 1024 * 1024,
            )
            rows = self._parse_list(response.content)
            new_count = 0
            for material in rows:
                if material.source_document_id not in materials:
                    materials[material.source_document_id] = material
                    new_count += 1
            if len(rows) < page_size or new_count == 0:
                break
            page += 1
        return sorted(
            materials.values(),
            key=lambda item: (item.listed_on, int(item.ir_seq), item.attachment_index),
        )

    def download_pdf(self, material: KindIrMaterial, *, max_bytes: int) -> bytes:
        response = self.http.get(
            _quoted_url(material.attachment_url),
            headers={"Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1"},
            max_bytes=max_bytes,
        )
        content = response.content
        if not content.startswith(b"%PDF-"):
            raise ValueError(
                f"KIND attachment is not a PDF: {material.source_document_id}"
            )
        return content

    @staticmethod
    def _search_form(
        begin_date: date,
        end_date: date,
        page: int,
        page_size: int,
    ) -> dict[str, Any]:
        start = begin_date.isoformat()
        end = end_date.isoformat()
        return {
            "method": "searchIRMaterialsSub",
            "forward": "searchirmaterials_sub",
            "currentPageSize": page_size,
            "pageIndex": page,
            "searchCodeType": "",
            "repIsuSrtCd": "",
            "irSeq": "",
            "searchCorpName": "",
            "resoroomType": "",
            "searchFromDate": start,
            "searchToDate": end,
            "marketType": "",
            "searchName": "",
            "kosdaqSegment": "",
            "title": "",
            "fromDate": start,
            "toDate": end,
        }

    def _parse_list(self, content: bytes) -> list[KindIrMaterial]:
        text = content.decode("euc-kr", errors="replace")
        root = html.fromstring(text)
        materials: list[KindIrMaterial] = []
        for row in root.xpath("//tr[td]"):
            cells = row.xpath("./td")
            if len(cells) < 5:
                continue
            company_name = " ".join(cells[1].text_content().split())
            title = " ".join(cells[3].text_content().split())
            try:
                listed_on = date.fromisoformat(" ".join(cells[2].text_content().split()))
            except ValueError:
                continue
            company_match = next(
                (
                    _COMPANY_PATTERN.search(anchor.get("onclick") or "")
                    for anchor in cells[1].xpath(".//a")
                    if _COMPANY_PATTERN.search(anchor.get("onclick") or "")
                ),
                None,
            )
            detail_match = next(
                (
                    _DETAIL_PATTERN.search(anchor.get("onclick") or "")
                    for anchor in cells[3].xpath(".//a")
                    if _DETAIL_PATTERN.search(anchor.get("onclick") or "")
                ),
                None,
            )
            if detail_match is None:
                continue
            attachments = [
                anchor
                for anchor in cells[4].xpath(".//a[@href]")
                if ".pdf" in (anchor.get("href") or "").casefold()
            ]
            for index, anchor in enumerate(attachments, start=1):
                href = str(anchor.get("href") or "")
                name = " ".join(anchor.text_content().split()) or href.rsplit("/", 1)[-1]
                materials.append(
                    KindIrMaterial(
                        ir_seq=detail_match.group("seq"),
                        resoroom_type=detail_match.group("room"),
                        kind_company_code=(
                            company_match.group("code") if company_match is not None else None
                        ),
                        company_name=company_name,
                        listed_on=listed_on,
                        title=title,
                        attachment_index=index,
                        attachment_name=name,
                        attachment_url=urljoin(self.base_url + "/", href),
                    )
                )
        return materials


class KindIrCollector:
    def __init__(
        self,
        client: KindIrClient,
        store: BronzeFilingStore,
        *,
        max_download_bytes: int = 256 * 1024 * 1024,
    ) -> None:
        if max_download_bytes <= 0:
            raise ValueError("max_download_bytes must be positive")
        self.client = client
        self.store = store
        self.max_download_bytes = max_download_bytes

    def collect(
        self,
        *,
        begin_date: date,
        end_date: date,
        companies: list[KindCompanyIdentity],
        refresh: bool = False,
        max_materials: int | None = None,
        max_materials_per_company: int | None = None,
    ) -> CollectionResult:
        if not companies:
            raise ValueError("at least one company identity is required")
        started = datetime.now(timezone.utc)
        identities = self._identity_index(companies)
        discovered = self.client.search_materials(
            begin_date=begin_date,
            end_date=end_date,
        )
        selected: list[tuple[KindIrMaterial, KindCompanyIdentity]] = []
        for material in discovered:
            identity = identities.get(normalize_company_name(material.company_name))
            if identity is None and material.kind_company_code:
                identity = identities.get(f"kind:{material.kind_company_code}")
            if identity is not None:
                selected.append((material, identity))
        if max_materials_per_company is not None:
            if max_materials_per_company <= 0:
                raise ValueError("max_materials_per_company must be positive")
            latest: list[tuple[KindIrMaterial, KindCompanyIdentity]] = []
            per_ticker: dict[str, int] = {}
            for material, identity in reversed(selected):
                count = per_ticker.get(identity.ticker, 0)
                if count >= max_materials_per_company:
                    continue
                latest.append((material, identity))
                per_ticker[identity.ticker] = count + 1
            selected = sorted(
                latest,
                key=lambda item: (
                    item[0].listed_on,
                    int(item[0].ir_seq),
                    item[0].attachment_index,
                ),
            )
        if max_materials is not None:
            selected = selected[:max_materials]

        filings: list[CollectedFiling] = []
        failures: list[CollectionFailure] = []
        for material, identity in selected:
            try:
                current = self.store.current(SourceType.IR, material.source_document_id)
                if current is not None and not refresh:
                    filings.append(current)
                    continue
                content = self.client.download_pdf(
                    material,
                    max_bytes=self.max_download_bytes,
                )
                published_at = datetime.combine(material.listed_on, time.min, tzinfo=_KST)
                available_at = datetime.combine(
                    material.listed_on + timedelta(days=1),
                    time.min,
                    tzinfo=_KST,
                )
                descriptor = FilingDescriptor(
                    source_type=SourceType.IR,
                    source_document_id=material.source_document_id,
                    issuer_id=identity.issuer_id,
                    issuer_name=identity.issuer_name,
                    ticker=identity.ticker,
                    report_name=material.title,
                    form_type="KIND_IR_MATERIAL",
                    filing_date=material.listed_on,
                    published_at=published_at,
                    available_at=available_at,
                    availability_precision=AvailabilityPrecision.DAY,
                    availability_source=(
                        "KIND IR material list date; conservatively available from "
                        "00:00 Asia/Seoul on the following calendar day"
                    ),
                    primary_document_name="document.pdf",
                    primary_document_url=material.attachment_url,
                    archive_url=self.client.search_url,
                    source_specific={
                        "provider": "KIND",
                        "ir_seq": material.ir_seq,
                        "resoroom_type": material.resoroom_type,
                        "kind_company_code": material.kind_company_code,
                        "kind_company_name": material.company_name,
                        "listed_on": material.listed_on.isoformat(),
                        "attachment_index": material.attachment_index,
                        "original_filename": material.attachment_name,
                        "statement_type": "MANAGEMENT_CLAIM",
                    },
                )
                filings.append(
                    self.store.save(
                        descriptor,
                        files={"documents/document.pdf": content},
                        primary_path="documents/document.pdf",
                    )
                )
            except Exception as exc:
                failures.append(
                    CollectionFailure(
                        source_document_id=material.source_document_id,
                        message=f"{type(exc).__name__}: {exc}",
                    )
                )
        return CollectionResult(
            source_type=SourceType.IR,
            started_at=started,
            completed_at=datetime.now(timezone.utc),
            query={
                "provider": "KIND",
                "begin_date": begin_date.isoformat(),
                "end_date": end_date.isoformat(),
                "company_count": len(companies),
                "matched_material_count": len(selected),
                "selection_policy": "availability-and-identity-only; no return data",
                "refresh": refresh,
                "max_materials": max_materials,
                "max_materials_per_company": max_materials_per_company,
            },
            discovered_count=len(selected),
            filings=filings,
            failures=failures,
        )

    @staticmethod
    def _identity_index(
        companies: list[KindCompanyIdentity],
    ) -> dict[str, KindCompanyIdentity]:
        result: dict[str, KindCompanyIdentity] = {}
        for company in companies:
            key = normalize_company_name(company.issuer_name)
            if key in result and result[key] != company:
                raise ValueError(f"ambiguous normalized company name: {company.issuer_name}")
            result[key] = company
            if company.kind_company_code:
                result[f"kind:{company.kind_company_code}"] = company
        return result
