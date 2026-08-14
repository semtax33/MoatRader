from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

from moatrader.runner.models import CompanyRunStatus, UniverseRunResult


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def coverage_min(company: object) -> Decimal:
    score = getattr(company, "moat_score", None)
    if score is None:
        return Decimal(0)
    coverage = score.document_coverage
    values = [
        value
        for value in (
            coverage.token_retention,
            coverage.evidence_retention,
            coverage.char_retention,
            coverage.section_retention,
            coverage.table_retention,
            coverage.numeric_retention,
        )
        if value is not None
    ]
    return Decimal(str(min(values))) if values else Decimal(0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True, help="DATE=RUN_RESULT_JSON")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    universe = read_csv(workspace / "inputs" / "universe.csv")
    date_rows = read_csv(workspace / "inputs" / "dates.csv")
    tickers = [row["stock_code"].zfill(6) for row in universe]
    universe_name = {row["stock_code"].zfill(6): row.get("name", "") for row in universe}
    dates = [(row.get("date") or row.get("as_of") or "").strip() for row in date_rows]
    if len(set(tickers)) != len(tickers):
        raise ValueError("universe contains duplicate stock codes")
    if len(set(dates)) != len(dates) or any(not date for date in dates):
        raise ValueError("dates input contains duplicate or blank dates")
    run_paths: dict[str, list[Path]] = {}
    for value in args.run:
        date, separator, path = value.partition("=")
        if not separator:
            raise ValueError(f"invalid --run value: {value}")
        run_paths.setdefault(date, []).append(Path(path).resolve())
    if set(run_paths) != set(dates):
        raise ValueError(f"run dates mismatch: expected={dates}, actual={sorted(run_paths)}")

    rows: list[dict[str, object]] = []
    for date in dates:
        by_ticker = {}
        for run_path in run_paths[date]:
            result = UniverseRunResult.model_validate_json(run_path.read_text(encoding="utf-8-sig"))
            # Later --run arguments are repair overlays and intentionally win.
            by_ticker.update({company.ticker: company for company in result.companies})
        for ticker in tickers:
            company = by_ticker.get(ticker)
            if company is None:
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "issuer_name": universe_name[ticker],
                        "status": "NO_PIT_DOCUMENT",
                        "signal_eligible": 0,
                        "signal": "0",
                        "signal_rank": "",
                        "moat_score": "",
                        "model_confidence": "",
                        "document_coverage": "",
                        "dcf_fair_value": "",
                        "current_price": "",
                        "price_to_dcf": "",
                        "margin_of_safety": "",
                    }
                )
                continue

            score = company.moat_score
            fair_value = company.dcf.fair_value_per_share if company.dcf else None
            price = company.current_price
            ratio = price / fair_value if price is not None and fair_value is not None and fair_value > 0 else None
            margin = Decimal(1) - ratio if ratio is not None else None
            coverage = coverage_min(company)
            eligible = (
                company.status == CompanyRunStatus.COMPLETE
                and score is not None
                and ratio is not None
                and margin is not None
                and margin > 0
            )
            signal = (
                Decimal(str(score.economic_moat_score))
                / Decimal(10)
                * margin
                * Decimal(str(score.model_confidence))
                * coverage
                if eligible and score is not None and margin is not None
                else Decimal(0)
            )
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "issuer_name": company.issuer_name or universe_name[ticker],
                    "status": company.status.value,
                    "signal_eligible": int(eligible),
                    "signal": signal,
                    "signal_rank": "",
                    "moat_score": score.economic_moat_score if score else "",
                    "model_confidence": score.model_confidence if score else "",
                    "document_coverage": coverage if score else "",
                    "dcf_fair_value": fair_value or "",
                    "current_price": price or "",
                    "price_to_dcf": ratio or "",
                    "margin_of_safety": margin if margin is not None else "",
                }
            )

    for date in dates:
        ranked = [row for row in rows if row["date"] == date and row["signal_eligible"]]
        ranked.sort(key=lambda row: (-Decimal(str(row["signal"])), str(row["ticker"])))
        for rank, row in enumerate(ranked, start=1):
            row["signal_rank"] = rank

    fields = [
        "date", "ticker", "issuer_name", "status", "signal_eligible", "signal",
        "signal_rank", "moat_score", "model_confidence", "document_coverage",
        "dcf_fair_value", "current_price", "price_to_dcf", "margin_of_safety",
    ]
    output = args.output.resolve()
    write_csv(output, rows, fields)
    coverage_rows = []
    for date in dates:
        subset = [row for row in rows if row["date"] == date]
        statuses = Counter(str(row["status"]) for row in subset)
        coverage_rows.append(
            {
                "date": date,
                "row_count": len(subset),
                "eligible_count": sum(int(row["signal_eligible"]) for row in subset),
                "complete_count": statuses["COMPLETE"],
                "no_pit_document_count": statuses["NO_PIT_DOCUMENT"],
                "other_status_count": len(subset) - statuses["COMPLETE"] - statuses["NO_PIT_DOCUMENT"],
            }
        )
    write_csv(output.with_name("signal-coverage.csv"), coverage_rows, list(coverage_rows[0]))
    manifest = {
        "schema_version": "moatrader-moat-dcf-signal/1",
        "signal_formula": "(moat_score/10) * max(1-price/dcf, 0) * model_confidence * min_document_coverage",
        "fallback_signal": 0,
        "row_count": len(rows),
        "expected_row_count": len(tickers) * len(dates),
        "universe_count": len(tickers),
        "dates": dates,
        "run_results": {
            date: [str(path) for path in paths]
            for date, paths in run_paths.items()
        },
        "status_counts": dict(Counter(str(row["status"]) for row in rows)),
    }
    output.with_name("signal-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if len(rows) != len(tickers) * len(dates):
        raise RuntimeError("signal panel row count mismatch")
    keys = {(str(row["date"]), str(row["ticker"])) for row in rows}
    if len(keys) != len(rows):
        raise RuntimeError("signal panel contains duplicate date/ticker rows")
    if any(row["signal"] == "" for row in rows):
        raise RuntimeError("signal panel contains blank signal values")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
