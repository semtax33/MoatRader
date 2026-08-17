from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel, SourceType


TICKER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$"


class DocumentInput(ContractModel):
    ticker: str = Field(pattern=TICKER_PATTERN)
    source: SourceType
    input_path: str
    metadata_path: str


class CompanyInput(ContractModel):
    ticker: str = Field(pattern=TICKER_PATTERN)
    issuer_id: str | None = None
    issuer_name: str | None = None
    documents: list[DocumentInput] = Field(min_length=1)
    current_price: Decimal | None = Field(default=None, gt=0)
    price_as_of: datetime | None = None
    dcf_assumptions_path: str | None = None
    expectation_assumptions_path: str | None = None

    @model_validator(mode="after")
    def market_data_is_complete(self) -> "CompanyInput":
        if (self.current_price is None) != (self.price_as_of is None):
            raise ValueError("current_price and price_as_of must be supplied together")
        if self.price_as_of is not None and (
            self.price_as_of.tzinfo is None or self.price_as_of.utcoffset() is None
        ):
            raise ValueError("price_as_of must be timezone-aware")
        if any(document.ticker != self.ticker for document in self.documents):
            raise ValueError("all documents must belong to the company ticker")
        return self


class UniverseManifest(ContractModel):
    path: str
    companies: list[CompanyInput] = Field(min_length=1)

    def select(self, tickers: set[str] | None = None) -> list[CompanyInput]:
        if not tickers:
            return list(self.companies)
        selected = [company for company in self.companies if company.ticker in tickers]
        missing = tickers - {company.ticker for company in selected}
        if missing:
            raise ValueError(f"tickers not found in manifest: {sorted(missing)}")
        return selected


def _resolve(base: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    return str(path if path.is_absolute() else (base / path).resolve())


def _parse_datetime(value: str | None, field: str, row_number: int) -> datetime | None:
    if not value:
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"row {row_number}: invalid {field}: {value}") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"row {row_number}: {field} must include timezone offset")
    return result


def load_universe_manifest(path: str | Path) -> UniverseManifest:
    manifest_path = Path(path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"universe manifest not found: {manifest_path}")
    rows_by_ticker: dict[str, list[dict[str, str]]] = defaultdict(list)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"ticker", "source", "input", "metadata"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"manifest is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            ticker = (row.get("ticker") or "").strip()
            if not ticker:
                raise ValueError(f"row {row_number}: ticker is required")
            row["_row_number"] = str(row_number)
            rows_by_ticker[ticker].append(row)

    companies: list[CompanyInput] = []
    for ticker, rows in rows_by_ticker.items():
        first = rows[0]
        base = manifest_path.parent
        documents: list[DocumentInput] = []
        for row in rows:
            row_number = int(row["_row_number"])
            source_text = (row.get("source") or "").strip().upper()
            if source_text in {"SEC", "EDGAR"}:
                source_text = "SEC_EDGAR"
            try:
                source = SourceType(source_text)
            except ValueError as exc:
                raise ValueError(f"row {row_number}: unsupported source {source_text!r}") from exc
            input_path = _resolve(base, row.get("input"))
            metadata_path = _resolve(base, row.get("metadata"))
            assert input_path is not None and metadata_path is not None
            if not Path(input_path).is_file():
                raise FileNotFoundError(f"row {row_number}: input not found: {input_path}")
            if not Path(metadata_path).is_file():
                raise FileNotFoundError(f"row {row_number}: metadata not found: {metadata_path}")
            documents.append(
                DocumentInput(
                    ticker=ticker,
                    source=source,
                    input_path=input_path,
                    metadata_path=metadata_path,
                )
            )

        def consistent(field: str) -> str | None:
            values = {(row.get(field) or "").strip() for row in rows} - {""}
            if len(values) > 1:
                raise ValueError(f"ticker {ticker}: conflicting {field} values: {sorted(values)}")
            return next(iter(values), None)

        current_price_text = consistent("current_price")
        price_as_of_text = consistent("price_as_of")
        dcf_path = _resolve(base, consistent("dcf_assumptions"))
        if dcf_path and not Path(dcf_path).is_file():
            raise FileNotFoundError(f"ticker {ticker}: DCF assumptions not found: {dcf_path}")
        expectation_path = _resolve(base, consistent("expectation_assumptions"))
        if expectation_path and not Path(expectation_path).is_file():
            raise FileNotFoundError(
                f"ticker {ticker}: expectation assumptions not found: {expectation_path}"
            )
        companies.append(
            CompanyInput(
                ticker=ticker,
                issuer_id=consistent("issuer_id"),
                issuer_name=consistent("issuer_name"),
                documents=documents,
                current_price=Decimal(current_price_text) if current_price_text else None,
                price_as_of=_parse_datetime(price_as_of_text, "price_as_of", int(first["_row_number"])),
                dcf_assumptions_path=dcf_path,
                expectation_assumptions_path=expectation_path,
            )
        )
    return UniverseManifest(path=str(manifest_path), companies=companies)
