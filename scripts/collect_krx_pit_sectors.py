from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date, datetime, time as datetime_time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from moatrader.ingestion.krx import KrxDataClient


SEOUL = ZoneInfo("Asia/Seoul")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def ticker(value: object) -> str:
    return str(value or "").strip().zfill(6)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect official historical KRX sector snapshots.")
    parser.add_argument("--dates", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-lookback-days", type=int, default=10)
    parser.add_argument("--insecure-tls", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"KRX PIT sector output already exists: {output}")
    dates = [date.fromisoformat(str(next(iter(row.values()))).strip()) for row in read_csv(args.dates.resolve())]
    universe = {ticker(row.get("stock_code")) for row in read_csv(args.universe.resolve())}
    client = KrxDataClient(verify_tls=not args.insecure_tls)
    collected: list[dict[str, str]] = []
    manifest: list[dict[str, object]] = []
    for signal_date in dates:
        found = None
        for offset in range(args.maximum_lookback_days + 1):
            candidate = signal_date - timedelta(days=offset)
            rows = client.sector_snapshot(candidate)
            matching = {item.ticker: item for item in rows if item.ticker in universe}
            if len(matching) == len(universe):
                found = (candidate, matching)
                break
        if found is None:
            raise ValueError(f"no complete KRX sector snapshot found before {signal_date}")
        snapshot_date, matching = found
        published = datetime.combine(snapshot_date, datetime_time(23, 59, 59), tzinfo=SEOUL)
        response_hashes = sorted({item.raw_response_sha256 for item in matching.values()})
        for code, item in sorted(matching.items()):
            collected.append(
                {
                    "ticker": code,
                    "sector": item.sector,
                    "industry_code": f"KRX:{item.market}:{item.sector}",
                    "effective_from": snapshot_date.isoformat(),
                    "effective_to": signal_date.isoformat(),
                    "source_published_at": published.isoformat(),
                    "source": "KRX_MDCSTAT03901_HISTORICAL_SNAPSHOT",
                    "evidence_ref": f"KRX:MDCSTAT03901:{snapshot_date.isoformat()}:{item.raw_response_sha256}:{code}",
                }
            )
        manifest.append(
            {
                "signal_date": signal_date.isoformat(),
                "snapshot_date": snapshot_date.isoformat(),
                "universe_count": len(matching),
                "raw_response_sha256": response_hashes,
                "retrieved_at": datetime.now(tz=SEOUL).isoformat(),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ticker", "sector", "industry_code", "effective_from", "effective_to",
        "source_published_at", "source", "evidence_ref",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(collected)
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
