from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--dates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    universe_path = args.universe.resolve()
    dates_path = args.dates.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"fresh workspace is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    inputs = output / "inputs"
    inputs.mkdir()
    universe_copy = inputs / "universe.csv"
    dates_copy = inputs / "dates.csv"
    shutil.copy2(universe_path, universe_copy)
    shutil.copy2(dates_path, dates_copy)

    universe = read_csv(universe_copy)
    date_rows = read_csv(dates_copy)
    tickers = [row["stock_code"].zfill(6) for row in universe]
    dates = [(row.get("date") or row.get("as_of") or "").strip() for row in date_rows]
    if len(tickers) != len(set(tickers)):
        raise ValueError("universe stock_code values must be unique")
    if any(not value for value in dates) or len(dates) != len(set(dates)):
        raise ValueError("dates must be non-empty and unique")
    (inputs / "stock-codes.txt").write_text("\n".join(tickers) + "\n", encoding="utf-8")
    (output / "workspace-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "moatrader-kr-signal-backtest/1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "fresh_run": True,
                "source_result_reuse": False,
                "universe_count": len(tickers),
                "dates": dates,
                "expected_signal_count": len(tickers) * len(dates),
                "input_sha256": {
                    "universe.csv": sha256(universe_copy),
                    "dates.csv": sha256(dates_copy),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"workspace={output}")
    print(f"universe={len(tickers)} dates={len(dates)} expected={len(tickers) * len(dates)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
