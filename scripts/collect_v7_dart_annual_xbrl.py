from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

from moatrader.financial.historical_xbrl import parse_dart_ifrs_archive
from moatrader.ingestion.dart_web import DartWebClient, sha256_bytes


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _corporation_map(path: Path) -> dict[str, str]:
    archive = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        members = [name for name in source.namelist() if name.lower().endswith(".xml")]
        if not members:
            raise ValueError("DART corporation archive contains no XML")
        root = ElementTree.fromstring(source.read(members[0]))
    result: dict[str, str] = {}
    for item in root.findall(".//list"):
        ticker = (item.findtext("stock_code") or "").strip()
        corp_code = (item.findtext("corp_code") or "").strip()
        if len(ticker) == 6 and len(corp_code) == 8:
            result[ticker] = corp_code
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collect_company(
    row: dict[str, str],
    *,
    corp_by_ticker: dict[str, str],
    output: Path,
    begin_date: date,
    end_date: date,
    requests_per_second: float,
) -> dict[str, object]:
    ticker = row["stock_code"].zfill(6)
    corp_code = corp_by_ticker.get(ticker)
    if corp_code is None:
        raise ValueError(f"DART corporation code is missing for {ticker}")
    corp_name = row["name"]
    client = DartWebClient(requests_per_second=requests_per_second)
    filings = client.list_annual_filings(
        ticker=ticker,
        corp_code=corp_code,
        corp_name=corp_name,
        begin_date=begin_date,
        end_date=end_date,
    )
    collected = 0
    reused = 0
    failures: list[dict[str, str]] = []
    for filing in filings:
        filing_dir = output / "filings" / ticker / filing.rcept_no
        archive_path = filing_dir / "ifrs.zip"
        metadata_path = filing_dir / "metadata.json"
        if archive_path.is_file() and metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if _sha256(archive_path) != metadata.get("archive_sha256"):
                raise ValueError(f"existing DART archive hash mismatch: {archive_path}")
            reused += 1
            continue
        try:
            dcm_no = client.resolve_dcm_no(filing)
            archive = client.download_ifrs_archive(filing, dcm_no=dcm_no)
            metrics = parse_dart_ifrs_archive(archive, fiscal_year=filing.fiscal_year)
            filing_dir.mkdir(parents=True, exist_ok=True)
            archive_path.write_bytes(archive)
            metadata = {
                **filing.model_dump(mode="json"),
                "dcm_no": dcm_no,
                "archive_sha256": sha256_bytes(archive),
                "archive_bytes": len(archive),
                "download_url": (
                    "https://dart.fss.or.kr/pdf/download/ifrs.do?"
                    f"rcp_no={filing.rcept_no}&dcm_no={dcm_no}&lang=ko"
                ),
                "metrics": metrics.model_dump(mode="json"),
                "data_pit_eligible_from": filing.available_at.isoformat(),
            }
            metadata_path.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            collected += 1
        except Exception as exc:
            failures.append(
                {
                    "ticker": ticker,
                    "rcept_no": filing.rcept_no,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "ticker": ticker,
        "corp_code": corp_code,
        "discovered": len(filings),
        "collected": collected,
        "reused": reused,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect historical annual DART XBRL filings into a separate v7 PIT archive."
    )
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--corp-code-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--begin-date", type=date.fromisoformat, default=date(2019, 1, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-second", type=float, default=2.0)
    parser.add_argument("--max-tickers", type=int)
    args = parser.parse_args()
    if "v7" not in args.output.resolve().as_posix().casefold():
        raise ValueError("historical DART output must be a v7 path")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8")
    universe = _read_csv(args.universe.resolve())
    if len(universe) != 150:
        raise ValueError(f"historical validation requires the frozen 150-stock universe, got {len(universe)}")
    if args.max_tickers is not None:
        universe = universe[: args.max_tickers]
    corp_by_ticker = _corporation_map(args.corp_code_zip.resolve())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_company,
                row,
                corp_by_ticker=corp_by_ticker,
                output=output,
                begin_date=args.begin_date,
                end_date=args.end_date,
                requests_per_second=args.requests_per_second,
            ): row["stock_code"].zfill(6)
            for row in universe
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "ticker": ticker,
                    "corp_code": corp_by_ticker.get(ticker),
                    "discovered": 0,
                    "collected": 0,
                    "reused": 0,
                    "failures": [
                        {"ticker": ticker, "rcept_no": "", "error": f"{type(exc).__name__}: {exc}"}
                    ],
                }
            with lock:
                results.append(result)
                print(
                    f"[{completed}/{len(universe)}] {ticker} "
                    f"filings={result['discovered']} collected={result['collected']} "
                    f"reused={result['reused']} failures={len(result['failures'])}",
                    flush=True,
                )

    metadata_files = sorted((output / "filings").rglob("metadata.json")) if (output / "filings").is_dir() else []
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_files]
    failures = [item for result in results for item in result["failures"]]
    manifest = {
        "schema_version": "v7-dart-web-annual-xbrl/1",
        "universe_path": str(args.universe.resolve()),
        "universe_sha256": _sha256(args.universe.resolve()),
        "requested_ticker_count": len(universe),
        "requested_begin_date": args.begin_date.isoformat(),
        "requested_end_date": args.end_date.isoformat(),
        "filing_count": len(metadata),
        "ticker_with_filing_count": len({item["ticker"] for item in metadata}),
        "fiscal_year_counts": {
            str(year): sum(item["fiscal_year"] == year for item in metadata)
            for year in range(2018, 2025)
        },
        "failure_count": len(failures),
        "failures": failures,
        "source": "DART_PUBLIC_VIEWER_FILED_IFRS_ARCHIVE",
        "source_url": "https://dart.fss.or.kr/",
        "pit_policy": "RECEIPT_NUMBER_DATE_CONSERVATIVE_EOD_ASIA_SEOUL",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output / "manifest.json")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
