from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


SEOUL = ZoneInfo("Asia/Seoul")


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze a current sector classification for future PIT use.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot-date", type=date.fromisoformat, required=True)
    parser.add_argument("--source", default="NAVER_FINANCE_CURRENT_CLASSIFICATION_SNAPSHOT")
    args = parser.parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"PIT sector snapshot already exists: {output}")
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    published = datetime.combine(args.snapshot_date, datetime.max.time(), tzinfo=SEOUL).replace(microsecond=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "ticker",
        "sector",
        "industry_code",
        "effective_from",
        "effective_to",
        "source_published_at",
        "source",
        "evidence_ref",
    ]
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item.get("ticker") or "")):
            code = str(row.get("ticker") or "").strip().zfill(6)
            writer.writerow(
                {
                    "ticker": code,
                    "sector": str(row.get("sector") or "UNKNOWN").strip(),
                    "industry_code": str(row.get("industry_code") or "").strip(),
                    "effective_from": args.snapshot_date.isoformat(),
                    "effective_to": "",
                    "source_published_at": published.isoformat(),
                    "source": args.source,
                    "evidence_ref": f"sha256:{source_sha}:ticker:{code}",
                }
            )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
