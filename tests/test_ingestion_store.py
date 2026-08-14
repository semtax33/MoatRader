from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from moatrader.canonical.models import AvailabilityPrecision, SourceType
from moatrader.ingestion import (
    BronzeFilingStore,
    CollectedFiling,
    CollectionAction,
    FilingDescriptor,
    safe_relative_path,
    write_universe_manifest,
)


def _descriptor() -> FilingDescriptor:
    return FilingDescriptor(
        source_type=SourceType.DART,
        source_document_id="20250101000001",
        issuer_id="00126380",
        issuer_name="Sample",
        ticker="005930",
        report_name="사업보고서 (2024.12)",
        filing_date=date(2025, 1, 1),
        available_at=datetime(2025, 1, 1, 23, 59, tzinfo=timezone.utc),
        availability_precision=AvailabilityPrecision.DAY,
        availability_source="fixture",
        primary_document_name="main.xml",
        primary_document_url="https://example.test/main",
    )


def test_bronze_store_preserves_immutable_revisions_and_latest_pointer(tmp_path: Path) -> None:
    store = BronzeFilingStore(tmp_path)

    first = store.save(
        _descriptor(),
        files={"original.zip": b"zip-v1", "documents/main.xml": b"v1"},
        primary_path="documents/main.xml",
    )
    unchanged = store.save(
        _descriptor(),
        files={"original.zip": b"zip-v1", "documents/main.xml": b"v1"},
        primary_path="documents/main.xml",
    )
    revised = store.save(
        _descriptor(),
        files={"original.zip": b"zip-v2", "documents/main.xml": b"v2"},
        primary_path="documents/main.xml",
    )

    assert first.action == CollectionAction.DOWNLOADED
    assert unchanged.action == CollectionAction.UNCHANGED
    assert revised.action == CollectionAction.REVISED
    assert first.version_directory != revised.version_directory
    assert Path(first.input_path).read_bytes() == b"v1"
    assert Path(revised.input_path).read_bytes() == b"v2"
    current = store.current(SourceType.DART, _descriptor().source_document_id)
    assert current is not None and current.version_sha256 == revised.version_sha256
    assert len(list((tmp_path / "dart" / _descriptor().source_document_id / "versions").iterdir())) == 2


def test_safe_relative_path_rejects_absolute_and_parent_paths() -> None:
    with pytest.raises(ValueError):
        safe_relative_path("../outside")
    with pytest.raises(ValueError):
        safe_relative_path("C:\\outside")


def test_manifest_generation_fails_closed_for_conflicting_company_identity(tmp_path: Path) -> None:
    input_path = tmp_path / "input.html"
    metadata_path = tmp_path / "metadata.json"
    input_path.write_text("<html/>", encoding="utf-8")
    metadata_path.write_text("{}", encoding="utf-8")

    def filing(issuer_id: str) -> CollectedFiling:
        return CollectedFiling(
            source_type=SourceType.DART,
            source_document_id="20250101000001" + issuer_id[-1],
            issuer_id=issuer_id,
            issuer_name="Sample",
            ticker="SAMPLE",
            action=CollectionAction.DOWNLOADED,
            raw_sha256="0" * 64,
            version_sha256="1" * 64,
            downloaded_at=datetime.now(timezone.utc),
            input_path=str(input_path),
            metadata_path=str(metadata_path),
            version_directory=str(tmp_path),
        )

    with pytest.raises(ValueError, match="conflicting issuer IDs"):
        write_universe_manifest([filing("00126380"), filing("00999999")], tmp_path / "universe.csv")
