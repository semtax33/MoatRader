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


def select_preflight_sample(
    universe: list[dict[str, str]],
    *,
    sample_size: int,
    explicit_tickers: list[str],
) -> list[str]:
    tickers = [row["stock_code"].zfill(6) for row in universe]
    available = set(tickers)
    explicit = list(dict.fromkeys(value.zfill(6) for value in explicit_tickers))
    if explicit:
        if not 3 <= len(explicit) <= 5:
            raise ValueError("explicit preflight sample must contain 3 to 5 unique tickers")
        missing = sorted(set(explicit) - available)
        if missing:
            raise ValueError(f"preflight sample tickers are outside the universe: {missing}")
        return sorted(explicit)

    groups: dict[tuple[str, str], list[str]] = {}
    for row in universe:
        key = (row.get("market", ""), row.get("size_bucket", ""))
        groups.setdefault(key, []).append(row["stock_code"].zfill(6))
    seed = next((row.get("selection_seed") for row in universe if row.get("selection_seed")), "0")
    for values in groups.values():
        values.sort(key=lambda ticker: hashlib.sha256(f"{seed}:{ticker}".encode("utf-8")).hexdigest())
    selected: list[str] = []
    ordered_groups = sorted(groups)
    cursor = 0
    while len(selected) < sample_size:
        progressed = False
        for key in ordered_groups:
            values = groups[key]
            if cursor < len(values):
                selected.append(values[cursor])
                progressed = True
                if len(selected) == sample_size:
                    break
        if not progressed:
            break
        cursor += 1
    if len(selected) != sample_size:
        raise ValueError(f"universe has only {len(selected)} selectable preflight companies")
    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--dates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, choices=range(3, 6), default=5)
    parser.add_argument("--sample-ticker", action="append", default=[])
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
    if not universe:
        raise ValueError("universe must contain at least one company")
    tickers = [row["stock_code"].zfill(6) for row in universe]
    dates = [(row.get("date") or row.get("as_of") or "").strip() for row in date_rows]
    if len(tickers) != len(set(tickers)):
        raise ValueError("universe stock_code values must be unique")
    if any(not value for value in dates) or len(dates) != len(set(dates)):
        raise ValueError("dates must be non-empty and unique")
    if dates != sorted(dates):
        raise ValueError("dates must be chronological so the PIT evidence ledger cannot run out of order")
    sample_tickers = (
        select_preflight_sample(
            universe,
            sample_size=args.sample_size,
            explicit_tickers=args.sample_ticker,
        )
        if len(tickers) > 5
        else sorted(tickers)
    )
    sample_rows = [row for row in universe if row["stock_code"].zfill(6) in set(sample_tickers)]
    (inputs / "stock-codes.txt").write_text("\n".join(tickers) + "\n", encoding="utf-8")
    (inputs / "preflight-sample.txt").write_text(
        "\n".join(sample_tickers) + "\n",
        encoding="utf-8",
    )
    with (inputs / "preflight-sample.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(universe[0]))
        writer.writeheader()
        writer.writerows(sample_rows)
    universe_hash = sha256(universe_copy)
    dates_hash = sha256(dates_copy)
    experiment_id = f"{output.name}-{hashlib.sha256(f'{universe_hash}:{dates_hash}'.encode('utf-8')).hexdigest()[:12]}"
    (output / "workspace-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "moatrader-kr-signal-backtest/2",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "experiment_id": experiment_id,
                "fresh_run": True,
                "source_result_reuse": False,
                "universe_count": len(tickers),
                "dates": dates,
                "expected_signal_count": len(tickers) * len(dates),
                "preflight_required": len(tickers) > 5,
                "preflight_status": "PENDING" if len(tickers) > 5 else "NOT_REQUIRED",
                "preflight_sample_tickers": sample_tickers,
                "preflight_sample_size": len(sample_tickers),
                "input_sha256": {
                    "universe.csv": universe_hash,
                    "dates.csv": dates_hash,
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"workspace={output}")
    print(f"universe={len(tickers)} dates={len(dates)} expected={len(tickers) * len(dates)}")
    print(f"experiment_id={experiment_id}")
    print(f"preflight_sample={','.join(sample_tickers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
