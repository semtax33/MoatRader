from __future__ import annotations

import csv
import io
import os
from collections import defaultdict
from pathlib import Path

from moatrader.canonical.models import SourceType
from moatrader.ingestion.models import CollectedFiling
from moatrader.ingestion.store import BronzeFilingStore


UNIVERSE_COLUMNS = (
    "ticker",
    "source",
    "input",
    "metadata",
    "issuer_id",
    "issuer_name",
    "current_price",
    "price_as_of",
    "dcf_assumptions",
)


def write_collected_universe_manifest(
    store: BronzeFilingStore,
    output: str | Path,
    *,
    sources: set[SourceType] | None = None,
) -> Path:
    filings = store.iter_current(sources)
    if not filings:
        raise ValueError("no collected Bronze filings are available for the requested sources")
    return write_universe_manifest(filings, output)


def write_universe_manifest(filings: list[CollectedFiling], output: str | Path) -> Path:
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(UNIVERSE_COLUMNS), lineterminator="\n")
    writer.writeheader()
    ordered = sorted(
        filings,
        key=lambda item: (item.ticker.upper(), item.source_type.value, item.source_document_id),
    )
    grouped: dict[str, list[CollectedFiling]] = defaultdict(list)
    for filing in ordered:
        grouped[filing.ticker].append(filing)
    identity: dict[str, tuple[str, str]] = {}
    for ticker, company_filings in grouped.items():
        issuer_ids = {item.issuer_id for item in company_filings}
        if len(issuer_ids) != 1:
            raise ValueError(
                f"ticker {ticker}: collected sources use conflicting issuer IDs; "
                "supply a curated company identity manifest"
            )
        latest_named = next(
            (item for item in reversed(company_filings) if item.issuer_name),
            company_filings[-1],
        )
        identity[ticker] = (next(iter(issuer_ids)), latest_named.issuer_name or "")

    for filing in ordered:
        filing.verify_files()
        issuer_id, issuer_name = identity[filing.ticker]
        writer.writerow(
            {
                "ticker": filing.ticker,
                "source": filing.source_type.value,
                "input": os.path.relpath(filing.input_path, output_path.parent),
                "metadata": os.path.relpath(filing.metadata_path, output_path.parent),
                "issuer_id": issuer_id,
                "issuer_name": issuer_name,
                "current_price": "",
                "price_as_of": "",
                "dcf_assumptions": "",
            }
        )
    _atomic_write(output_path, stream.getvalue().encode("utf-8-sig"))
    return output_path


def _atomic_write(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
