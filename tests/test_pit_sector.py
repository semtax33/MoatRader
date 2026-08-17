from __future__ import annotations

from datetime import date, datetime, timezone

from moatrader.financial.pit_sector import PitSectorRecord, resolve_pit_sector


def _record(*, sector: str, effective: str, published: str) -> PitSectorRecord:
    return PitSectorRecord(
        ticker="000001",
        sector=sector,
        effective_from=date.fromisoformat(effective),
        source_published_at=datetime.fromisoformat(published),
        source="KRX_PIT_SNAPSHOT",
        evidence_ref=f"snapshot:{effective}",
    )


def test_sector_resolution_rejects_future_published_classification() -> None:
    records = [
        _record(
            sector="Old",
            effective="2025-01-01",
            published="2025-01-02T00:00:00+09:00",
        ),
        _record(
            sector="Future",
            effective="2025-06-01",
            published="2025-09-01T00:00:00+09:00",
        ),
    ]
    result = resolve_pit_sector(
        records,
        ticker="000001",
        as_of=datetime.fromisoformat("2025-08-31T23:59:59+09:00"),
    )
    assert result is not None
    assert result.sector == "Old"


def test_sector_resolution_requires_timezone_aware_cutoff() -> None:
    try:
        resolve_pit_sector([], ticker="000001", as_of=datetime(2025, 1, 1))
    except ValueError as exc:
        assert "timezone-aware" in str(exc)
    else:
        raise AssertionError("naive sector cutoff was accepted")
