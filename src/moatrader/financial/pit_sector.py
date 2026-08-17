from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class PitSectorRecord(ContractModel):
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    sector: str = Field(min_length=1)
    industry_code: str | None = None
    effective_from: date
    effective_to: date | None = None
    source_published_at: datetime
    source: str = Field(min_length=1)
    evidence_ref: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_interval(self) -> "PitSectorRecord":
        if self.effective_to is not None and self.effective_to < self.effective_from:
            raise ValueError("sector effective_to must not precede effective_from")
        if self.source_published_at.tzinfo is None or self.source_published_at.utcoffset() is None:
            raise ValueError("sector source_published_at must be timezone-aware")
        return self

    def available_for(self, as_of: datetime) -> bool:
        return (
            self.source_published_at <= as_of
            and self.effective_from <= as_of.date()
            and (self.effective_to is None or as_of.date() <= self.effective_to)
        )


def resolve_pit_sector(
    records: list[PitSectorRecord],
    *,
    ticker: str,
    as_of: datetime,
) -> PitSectorRecord | None:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("as_of must be timezone-aware")
    candidates = [
        item for item in records if item.ticker == ticker and item.available_for(as_of)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.effective_from, item.source_published_at))


def load_pit_sector_csv(path: Path) -> list[PitSectorRecord]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [
        PitSectorRecord(
            ticker=str(row.get("ticker") or "").strip().zfill(6),
            sector=str(row.get("sector") or "").strip(),
            industry_code=str(row.get("industry_code") or "").strip() or None,
            effective_from=date.fromisoformat(str(row.get("effective_from") or "")),
            effective_to=(
                date.fromisoformat(str(row["effective_to"]))
                if str(row.get("effective_to") or "").strip()
                else None
            ),
            source_published_at=datetime.fromisoformat(
                str(row.get("source_published_at") or "")
            ),
            source=str(row.get("source") or "").strip(),
            evidence_ref=str(row.get("evidence_ref") or "").strip(),
        )
        for row in rows
    ]
