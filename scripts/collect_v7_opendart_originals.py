from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import io
import json
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from xml.etree import ElementTree

from moatrader.ingestion.dart import DartApiError
from moatrader.ingestion.http import ResilientHttpClient
from moatrader.financial.historical_xbrl import parse_dart_ifrs_archive
from moatrader.ingestion.opendart_original import (
    OpenDartOriginalClient,
    extract_original_evidence,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _corporation_map(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(io.BytesIO(path.read_bytes())) as source:
        xml_members = [name for name in source.namelist() if name.lower().endswith(".xml")]
        if not xml_members:
            raise ValueError("DART corporation archive contains no XML")
        root = ElementTree.fromstring(source.read(xml_members[0]))
    result: dict[str, str] = {}
    for item in root.findall(".//list"):
        ticker = (item.findtext("stock_code") or "").strip()
        corp_code = (item.findtext("corp_code") or "").strip()
        if len(ticker) == 6 and len(corp_code) == 8:
            result[ticker] = corp_code
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _collect_company(
    row: dict[str, str],
    *,
    corp_codes: dict[str, str],
    client: OpenDartOriginalClient,
    output: Path,
    begin_date: date,
    end_date: date,
) -> dict[str, object]:
    ticker = row["stock_code"].zfill(6)
    corp_code = corp_codes.get(ticker)
    if corp_code is None:
        raise ValueError(f"OpenDART corporation code missing for {ticker}")
    filings = client.list_annual_filings(
        ticker=ticker,
        corp_code=corp_code,
        begin_date=begin_date,
        end_date=end_date,
    )
    reused = 0
    collected = 0
    original_failures: list[dict[str, str]] = []
    xbrl_unavailable: list[dict[str, str]] = []
    for filing in filings:
        filing_dir = output / "filings" / ticker / filing.rcept_no
        metadata_path = filing_dir / "metadata.json"
        original_path = filing_dir / "original-document.zip"
        if metadata_path.is_file() and original_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if _sha256(original_path) != metadata.get("original_archive_sha256"):
                raise ValueError(f"existing OpenDART original hash mismatch: {original_path}")
            reused += 1
            continue
        try:
            original = client.download_original_archive(filing.rcept_no)
            filing_dir.mkdir(parents=True, exist_ok=True)
            original_path.write_bytes(original)
            evidence, texts = extract_original_evidence(
                original, rcept_no=filing.rcept_no, available_at=filing.available_at
            )
            xbrl_status = "NOT_REQUESTED"
            xbrl_hash = None
            xbrl_bytes = None
            metrics = None
            try:
                xbrl_bytes = client.download_xbrl_archive(filing.rcept_no)
                xbrl_hash = hashlib.sha256(xbrl_bytes).hexdigest()
                parsed = parse_dart_ifrs_archive(
                    xbrl_bytes,
                    fiscal_year=filing.fiscal_period_end.year,
                    period_end=filing.fiscal_period_end,
                )
                metrics = parsed.model_dump(mode="json")
                xbrl_status = "COLLECTED_AND_PARSED"
            except DartApiError as exc:
                xbrl_status = f"NOT_AVAILABLE_{exc.status}"
                xbrl_unavailable.append(
                    {"ticker": ticker, "rcept_no": filing.rcept_no, "status": exc.status}
                )
            except Exception as exc:
                xbrl_status = f"COLLECTED_PARSE_FAILED_{type(exc).__name__}"
                xbrl_unavailable.append(
                    {
                        "ticker": ticker,
                        "rcept_no": filing.rcept_no,
                        "status": xbrl_status,
                        "error": str(exc),
                    }
                )

            for relative, text in texts.items():
                target = filing_dir / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            if xbrl_bytes is not None:
                (filing_dir / "financial-statements-xbrl.zip").write_bytes(xbrl_bytes)
            _write_json(
                filing_dir / "evidence-index.json",
                [item.model_dump(mode="json") for item in evidence],
            )
            metadata = {
                **filing.model_dump(mode="json"),
                "source_role": "PRIMARY_HISTORICAL_EVIDENCE",
                "original_api_endpoint": "https://opendart.fss.or.kr/api/document.xml",
                "xbrl_api_endpoint": "https://opendart.fss.or.kr/api/fnlttXbrl.xml",
                "original_archive_sha256": hashlib.sha256(original).hexdigest(),
                "original_archive_bytes": len(original),
                "evidence_document_count": len(evidence),
                "evidence_character_count": sum(item.char_count for item in evidence),
                "xbrl_status": xbrl_status,
                "xbrl_archive_sha256": xbrl_hash,
                "metrics": metrics,
                "data_pit_eligible_from": filing.available_at.isoformat(),
            }
            _write_json(metadata_path, metadata)
            collected += 1
        except Exception as exc:
            original_failures.append(
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
        "original_failures": original_failures,
        "xbrl_unavailable": xbrl_unavailable,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect official OpenDART original disclosure ZIPs as the primary v7 evidence corpus."
    )
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--corp-code-zip", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--begin-date", type=date.fromisoformat, default=date(2019, 1, 1))
    parser.add_argument("--end-date", type=date.fromisoformat, default=date(2025, 12, 31))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--requests-per-second", type=float, default=4.0)
    parser.add_argument("--max-tickers", type=int)
    parser.add_argument("--api-key-env", default="DART_API_KEY")
    parser.add_argument("--prompt-api-key", action="store_true")
    args = parser.parse_args()

    output = args.output.resolve()
    if "v7" not in output.as_posix().casefold():
        raise ValueError("OpenDART original output must be a v7 path")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8")
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key and args.prompt_api_key:
        api_key = getpass.getpass("OpenDART API key: ").strip()
    if not api_key:
        raise ValueError(
            f"OpenDART API key is missing; set {args.api_key_env} or use --prompt-api-key"
        )

    universe_path = args.universe.resolve()
    universe = _read_csv(universe_path)
    if len(universe) != 150:
        raise ValueError(f"historical validation requires 150 universe rows, got {len(universe)}")
    if args.max_tickers is not None:
        universe = universe[: args.max_tickers]
    corp_codes = _corporation_map(args.corp_code_zip.resolve())
    output.mkdir(parents=True, exist_ok=True)
    http = ResilientHttpClient(
        user_agent="MoatRader-v7-historical-validation/1.0",
        requests_per_second=args.requests_per_second,
        timeout_seconds=120,
        max_retries=4,
        default_max_bytes=512 * 1024 * 1024,
    )
    client = OpenDartOriginalClient(http, api_key)
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _collect_company,
                row,
                corp_codes=corp_codes,
                client=client,
                output=output,
                begin_date=args.begin_date,
                end_date=args.end_date,
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
                    "corp_code": corp_codes.get(ticker),
                    "discovered": 0,
                    "collected": 0,
                    "reused": 0,
                    "original_failures": [
                        {"ticker": ticker, "rcept_no": "", "error": f"{type(exc).__name__}: {exc}"}
                    ],
                    "xbrl_unavailable": [],
                }
            results.append(result)
            print(
                f"[{completed}/{len(universe)}] {ticker} filings={result['discovered']} "
                f"collected={result['collected']} reused={result['reused']} "
                f"original_failures={len(result['original_failures'])} "
                f"xbrl_unavailable={len(result['xbrl_unavailable'])}",
                flush=True,
            )

    metadata_files = sorted((output / "filings").rglob("metadata.json")) if (output / "filings").is_dir() else []
    metadata = [json.loads(path.read_text(encoding="utf-8")) for path in metadata_files]
    original_failures = [item for result in results for item in result["original_failures"]]
    xbrl_unavailable = [item for result in results for item in result["xbrl_unavailable"]]
    manifest = {
        "schema_version": "v7-opendart-original-evidence/1",
        "universe_path": str(universe_path),
        "universe_sha256": _sha256(universe_path),
        "requested_ticker_count": len(universe),
        "requested_begin_date": args.begin_date.isoformat(),
        "requested_end_date": args.end_date.isoformat(),
        "filing_count": len(metadata),
        "ticker_with_filing_count": len({item["ticker"] for item in metadata}),
        "original_failure_count": len(original_failures),
        "original_failures": original_failures,
        "xbrl_unavailable_count": len(xbrl_unavailable),
        "xbrl_unavailable": xbrl_unavailable,
        "primary_source": "OPENDART_DOCUMENT_XML_ORIGINAL_ZIP",
        "structured_companion_source": "OPENDART_FNLTT_XBRL_ORIGINAL_ZIP",
        "list_endpoint": "https://opendart.fss.or.kr/api/list.json",
        "original_endpoint": "https://opendart.fss.or.kr/api/document.xml",
        "xbrl_endpoint": "https://opendart.fss.or.kr/api/fnlttXbrl.xml",
        "api_key_persisted": False,
        "pit_policy": "ALL_ORIGINAL_AND_AMENDED_RECEIPTS; RECEIPT_DATE_EOD_ASIA_SEOUL",
    }
    _write_json(output / "manifest.json", manifest)
    print(output / "manifest.json")
    return 1 if original_failures else 0


if __name__ == "__main__":
    sys.exit(main())
