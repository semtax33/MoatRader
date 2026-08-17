from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from moatrader.canonical.models import AvailabilityPrecision, ContractModel, SourceType


COLLECTOR_VERSION = "0.3.0"


class CollectionAction(StrEnum):
    DOWNLOADED = "DOWNLOADED"
    UNCHANGED = "UNCHANGED"
    REVISED = "REVISED"


class FilingDescriptor(ContractModel):
    source_type: SourceType
    source_document_id: str = Field(min_length=1)
    issuer_id: str = Field(min_length=1)
    issuer_name: str | None = None
    ticker: str | None = None
    report_name: str | None = None
    form_type: str | None = None
    filing_date: date
    report_date: date | None = None
    published_at: datetime | None = None
    available_at: datetime
    availability_precision: AvailabilityPrecision
    availability_source: str = Field(min_length=1)
    primary_document_name: str = Field(min_length=1)
    primary_document_url: str
    archive_url: str | None = None
    is_amendment: bool = False
    amends_document_id: str | None = None
    source_specific: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def timestamps_and_amendment_are_valid(self) -> "FilingDescriptor":
        for name, value in (("published_at", self.published_at), ("available_at", self.available_at)):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must be timezone-aware")
        if self.is_amendment and not self.amends_document_id:
            raise ValueError("linked amendments must identify amends_document_id")
        return self

    def adapter_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "source_type": self.source_type.value,
            "source_document_id": self.source_document_id,
            "issuer_id": self.issuer_id,
            "issuer_name": self.issuer_name,
            "ticker": self.ticker,
            "title": self.report_name or self.form_type,
            "report_name": self.report_name,
            "form_type": self.form_type,
            "filing_date": self.filing_date.isoformat(),
            "report_date": self.report_date.isoformat() if self.report_date else None,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "available_at": self.available_at.isoformat(),
            "availability_precision": self.availability_precision.value,
            "availability_source": self.availability_source,
            "period_end": self.report_date.isoformat() if self.report_date else None,
            "is_amendment": self.is_amendment,
            "amends_document_id": self.amends_document_id,
            "primary_document_name": self.primary_document_name,
            "primary_document_url": self.primary_document_url,
            "archive_url": self.archive_url,
            "source_specific": self.source_specific,
        }
        if self.source_type == SourceType.DART:
            metadata.update(
                {
                    "rcept_no": self.source_document_id,
                    "corp_code": self.issuer_id,
                    "stock_code": self.ticker,
                    "jurisdiction": "KR",
                    "language": "ko",
                }
            )
        elif self.source_type == SourceType.SEC_EDGAR:
            metadata.update(
                {
                    "accession_number": self.source_document_id,
                    "cik": self.issuer_id,
                    "jurisdiction": "US",
                    "language": "en",
                }
            )
        elif self.source_type == SourceType.IR:
            metadata.update(
                {
                    "jurisdiction": "KR",
                    "language": "ko",
                }
            )
        elif self.source_type == SourceType.INDUSTRY:
            metadata.update(
                {
                    "jurisdiction": "KR",
                    "language": "ko",
                    "economic_scope": "INDUSTRY",
                    "industry_code": self.source_specific.get("industry_code"),
                    "industry_name": self.source_specific.get("industry_name"),
                    "publisher": self.source_specific.get("publisher"),
                    "author": self.source_specific.get("author"),
                    "source_system": "hankyung_consensus",
                }
            )
        return {key: value for key, value in metadata.items() if value is not None}


class CollectedFiling(ContractModel):
    source_type: SourceType
    source_document_id: str
    issuer_id: str
    issuer_name: str | None = None
    ticker: str
    action: CollectionAction
    raw_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    version_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    downloaded_at: datetime
    input_path: str
    metadata_path: str
    version_directory: str

    @model_validator(mode="after")
    def downloaded_at_is_aware(self) -> "CollectedFiling":
        if self.downloaded_at.tzinfo is None or self.downloaded_at.utcoffset() is None:
            raise ValueError("downloaded_at must be timezone-aware")
        return self

    def verify_files(self) -> None:
        if not Path(self.input_path).is_file():
            raise FileNotFoundError(f"collected primary document not found: {self.input_path}")
        if not Path(self.metadata_path).is_file():
            raise FileNotFoundError(f"collected metadata not found: {self.metadata_path}")


class CollectionFailure(ContractModel):
    source_document_id: str
    message: str


class CollectionResult(ContractModel):
    collector_version: str = COLLECTOR_VERSION
    source_type: SourceType
    started_at: datetime
    completed_at: datetime
    query: dict[str, Any] = Field(default_factory=dict)
    discovered_count: int = Field(ge=0)
    filings: list[CollectedFiling] = Field(default_factory=list)
    failures: list[CollectionFailure] = Field(default_factory=list)
    manifest_path: str | None = None

    @property
    def downloaded_count(self) -> int:
        return sum(item.action in {CollectionAction.DOWNLOADED, CollectionAction.REVISED} for item in self.filings)

    @property
    def unchanged_count(self) -> int:
        return sum(item.action == CollectionAction.UNCHANGED for item in self.filings)
