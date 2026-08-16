from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "moatrader-moat-contract-v2-validation-samples/1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _band(score: float) -> str:
    if score <= 25:
        return "LOW"
    if score <= 50:
        return "MID"
    return "HIGH"


def _balanced_sample(
    rows: list[dict[str, Any]], *, count: int, seed: str
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["market"], row["size_bucket"], row["old_score_band"])].append(row)
    for values in groups.values():
        values.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['ticker']}".encode("utf-8")
            ).hexdigest()
        )
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    cursor = 0
    while len(selected) < count:
        made_progress = False
        for offset in range(len(keys)):
            key = keys[(cursor + offset) % len(keys)]
            if groups[key]:
                selected.append(groups[key].pop(0))
                made_progress = True
                if len(selected) == count:
                    break
        if not made_progress:
            raise RuntimeError(f"requested {count} rows but only selected {len(selected)}")
        cursor = (cursor + 1) % len(keys)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze unseen and economic MOAT v2 samples.")
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--old-scores", required=True, type=Path)
    parser.add_argument("--notebook", required=True, type=Path)
    parser.add_argument("--frozen-sample", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--structural-count", type=int, default=10)
    parser.add_argument("--economic-count", type=int, default=24)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    workspace = args.workspace.resolve()
    workspace_manifest = json.loads(
        (workspace / "workspace-manifest.json").read_text(encoding="utf-8-sig")
    )
    dates = list(workspace_manifest["dates"])
    signal_date, return_date = dates[0], dates[1]
    structural_date = dates[-1]

    universe_rows = _read_csv(args.universe.resolve())
    universe = {row["stock_code"].zfill(6): row for row in universe_rows}
    notebook_text = args.notebook.resolve().read_text(encoding="utf-8")
    notebook_tickers = set(re.findall(r"(?<!\d)\d{6}(?!\d)", notebook_text))
    frozen_tickers = {
        row["ticker"].zfill(6) for row in _read_csv(args.frozen_sample.resolve())
    }
    old_scores = {
        (row["stock_code"].zfill(6), row["as_of"]): float(row["economic_moat_score_100"])
        for row in _read_csv(args.old_scores.resolve())
        if row.get("economic_moat_score_100") not in {None, ""}
    }
    manifests = {
        date: _read_csv(workspace / "date-inputs" / date / "universe-manifest.csv")
        for date in dates
    }
    available = {
        date: {row["ticker"].zfill(6) for row in rows}
        for date, rows in manifests.items()
    }

    candidates: list[dict[str, Any]] = []
    for ticker, row in universe.items():
        if ticker in notebook_tickers or ticker in frozen_tickers:
            continue
        if not all(ticker in available[date] for date in dates):
            continue
        if not all((ticker, date) in old_scores for date in dates):
            continue
        score = old_scores[(ticker, signal_date)]
        candidates.append(
            {
                "ticker": ticker,
                "company_name": row.get("name", ""),
                "market": row.get("market", ""),
                "size_bucket": row.get("size_bucket", ""),
                "old_score_signal_date": score,
                "old_score_band": _band(score),
            }
        )
    if len(candidates) < args.economic_count:
        raise RuntimeError(
            f"only {len(candidates)} notebook-unmentioned, non-frozen candidates; "
            f"need {args.economic_count}"
        )

    structural = _balanced_sample(
        [dict(row) for row in candidates],
        count=args.structural_count,
        seed="moat-v2-structural-20260816",
    )
    structural_tickers = {row["ticker"] for row in structural}
    remaining = [dict(row) for row in candidates if row["ticker"] not in structural_tickers]
    economic = structural + _balanced_sample(
        remaining,
        count=args.economic_count - len(structural),
        seed="moat-v2-economic-20260816",
    )
    economic_tickers = {row["ticker"] for row in economic}

    sample_rows = [
        {**row, "sample": "STRUCTURAL_AND_ECONOMIC"}
        for row in structural
    ] + [
        {**row, "sample": "ECONOMIC_ONLY"}
        for row in economic
        if row["ticker"] not in structural_tickers
    ]
    _write_csv(
        output / "validation-samples.csv",
        sample_rows,
        (
            "sample",
            "ticker",
            "company_name",
            "market",
            "size_bucket",
            "old_score_signal_date",
            "old_score_band",
        ),
    )

    manifest_hashes: dict[str, str] = {}
    for kind, date, tickers, batch_size in (
        ("structural", structural_date, [row["ticker"] for row in structural], 5),
        ("economic", signal_date, [row["ticker"] for row in economic], 4),
    ):
        source_rows = manifests[date]
        fields = list(source_rows[0])
        for index in range(0, len(tickers), batch_size):
            batch = set(tickers[index : index + batch_size])
            rows = [row for row in source_rows if row["ticker"].zfill(6) in batch]
            found = {row["ticker"].zfill(6) for row in rows}
            if found != batch:
                raise RuntimeError(f"{kind} batch missing: {sorted(batch - found)}")
            path = output / "manifests" / kind / f"batch-{index // batch_size + 1}.csv"
            _write_csv(path, rows, fields)
            manifest_hashes[path.relative_to(output).as_posix()] = _sha256(path)

    prices: dict[tuple[str, str], tuple[str, str]] = {}
    for date in (signal_date, return_date):
        for row in manifests[date]:
            ticker = row["ticker"].zfill(6)
            if ticker in economic_tickers and row.get("current_price"):
                prices.setdefault(
                    (ticker, date), (row["current_price"], row.get("price_as_of", ""))
                )
    price_rows = [
        {
            "ticker": ticker,
            "signal_date": signal_date,
            "signal_price": prices[(ticker, signal_date)][0],
            "signal_price_as_of": prices[(ticker, signal_date)][1],
            "return_date": return_date,
            "return_price": prices[(ticker, return_date)][0],
            "return_price_as_of": prices[(ticker, return_date)][1],
        }
        for ticker in sorted(economic_tickers)
    ]
    _write_csv(
        output / "economic-forward-prices.csv",
        price_rows,
        price_rows[0].keys(),
    )

    protocol = {
        "schema_version": SCHEMA_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "selection": (
            "notebook-unmentioned; frozen-8 excluded; complete four-date inputs; "
            "market-size-old-score-band round-robin with SHA256 seeded order"
        ),
        "candidate_count": len(candidates),
        "notebook_ticker_count": len(notebook_tickers & set(universe)),
        "excluded_frozen_tickers": sorted(frozen_tickers),
        "structural": {
            "date": structural_date,
            "count": len(structural),
            "tickers": [row["ticker"] for row in structural],
        },
        "economic": {
            "signal_date": signal_date,
            "return_date": return_date,
            "count": len(economic),
            "tickers": [row["ticker"] for row in economic],
        },
        "inputs": {
            "workspace": str(workspace),
            "workspace_manifest_sha256": _sha256(workspace / "workspace-manifest.json"),
            "universe": str(args.universe.resolve()),
            "universe_sha256": _sha256(args.universe.resolve()),
            "notebook": str(args.notebook.resolve()),
            "notebook_sha256": _sha256(args.notebook.resolve()),
            "old_scores": str(args.old_scores.resolve()),
            "old_scores_sha256": _sha256(args.old_scores.resolve()),
            "frozen_sample": str(args.frozen_sample.resolve()),
            "frozen_sample_sha256": _sha256(args.frozen_sample.resolve()),
            "manifest_sha256": manifest_hashes,
        },
    }
    (output / "validation-protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
