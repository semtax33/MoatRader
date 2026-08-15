from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path


DATES = ("2025-08-31", "2025-11-30", "2026-02-28", "2026-05-31")


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
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--signals", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    signals = (args.signals or workspace / "signals" / "moat-dcf-signals.csv").resolve()
    output = (args.output or workspace / "signals" / "signal-audit.json").resolve()
    rows = read_csv(signals)
    universe = read_csv(workspace / "inputs" / "universe.csv")
    expected_tickers = {row["stock_code"].zfill(6) for row in universe}
    expected_keys = {(as_of, ticker) for as_of in DATES for ticker in expected_tickers}
    actual_keys = {(row["date"], row["ticker"]) for row in rows}
    if len(rows) != 600 or actual_keys != expected_keys:
        raise RuntimeError("signal panel is not the exact 150 x 4 universe")
    for row in rows:
        Decimal(row["signal"])

    future_sources: list[dict[str, str]] = []
    future_prices: list[dict[str, str]] = []
    external_paths: list[dict[str, str]] = []
    manifest_counts: dict[str, int] = {}
    for as_of in DATES:
        cutoff = date.fromisoformat(as_of)
        manifest_rows = read_csv(workspace / "date-inputs" / as_of / "universe-manifest.csv")
        manifest_counts[as_of] = len(manifest_rows)
        for row in manifest_rows:
            paths = [Path(row["input"]).resolve(), Path(row["metadata"]).resolve()]
            if row.get("dcf_assumptions"):
                paths.append(Path(row["dcf_assumptions"]).resolve())
            for path in paths:
                if workspace not in path.parents:
                    external_paths.append({"date": as_of, "ticker": row["ticker"], "path": str(path)})
            metadata = json.loads(Path(row["metadata"]).read_text(encoding="utf-8"))
            if datetime.fromisoformat(metadata["available_at"]).date() > cutoff:
                future_sources.append(
                    {"date": as_of, "ticker": row["ticker"], "available_at": metadata["available_at"]}
                )
            if datetime.fromisoformat(row["price_as_of"]).date() > cutoff:
                future_prices.append(
                    {"date": as_of, "ticker": row["ticker"], "price_as_of": row["price_as_of"]}
                )
    if future_sources or future_prices or external_paths:
        raise RuntimeError("PIT or fresh-workspace path audit failed")

    models: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    usage: Counter[str] = Counter()
    response_ids: list[str] = []
    invalid_lines = 0
    for path in (workspace / "runs").glob("kr-signal-*/companies/*/llm-calls.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_lines += 1
                continue
            models[row["model"]] += 1
            tasks[row["task"]] += 1
            response_ids.append(row["response_id"])
            for key, value in row.get("usage", {}).items():
                usage[key] += int(value or 0)
    if invalid_lines or len(response_ids) != len(set(response_ids)):
        raise RuntimeError("LLM audit log integrity failed")
    status_counts = Counter(row["status"] for row in rows)
    expected_moat_calls = status_counts["COMPLETE"]
    if (
        tasks["FINAL_MOAT_SCORING"] != expected_moat_calls
        or models["gpt-5.6-luna"] != expected_moat_calls
    ):
        raise RuntimeError("MOAT model audit count mismatch")
    if set(models) != {"gpt-5-nano-2025-08-07", "gpt-5.6-luna"}:
        raise RuntimeError(f"unexpected model IDs: {sorted(models)}")

    eligible_counts = Counter(row["date"] for row in rows if row["signal_eligible"] == "1")
    audit = {
        "schema_version": "moatrader-kr-signal-audit/1",
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "workspace": str(workspace),
        "fresh_workspace_only": True,
        "row_count": len(rows),
        "unique_date_ticker_count": len(actual_keys),
        "universe_count": len(expected_tickers),
        "dates": list(DATES),
        "rows_per_date": dict(Counter(row["date"] for row in rows)),
        "status_counts": dict(status_counts),
        "eligible_counts": dict(eligible_counts),
        "manifest_counts": manifest_counts,
        "pit_audit": {
            "future_source_count": len(future_sources),
            "future_price_count": len(future_prices),
            "external_path_count": len(external_paths),
        },
        "llm_audit": {
            "call_count": len(response_ids),
            "unique_response_id_count": len(set(response_ids)),
            "models": dict(models),
            "tasks": dict(tasks),
            "usage": dict(usage),
        },
        "sha256": {
            "universe.csv": sha256(workspace / "inputs" / "universe.csv"),
            "dates.csv": sha256(workspace / "inputs" / "dates.csv"),
            "moat-dcf-signals.csv": sha256(signals),
            "signal-coverage.csv": sha256(signals.with_name("signal-coverage.csv")),
            "signal-manifest.json": sha256(signals.with_name("signal-manifest.json")),
        },
    }
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
