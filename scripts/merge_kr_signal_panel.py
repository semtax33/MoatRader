from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from decimal import Decimal
from pathlib import Path

from moatrader.runner.models import CompanyRunStatus, UniverseRunResult
from moatrader.runner.engine import RUNNER_VERSION


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def moat_coverage(company: object) -> Decimal:
    score = getattr(company, "moat_score", None)
    if score is None:
        return Decimal(0)
    coverage = score.document_coverage
    return Decimal(str(coverage.moat_evidence_coverage or 0))


def _average_ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2
        for position in range(cursor, end):
            result[ordered[position]] = average
        cursor = end
    return result


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_mean = statistics.mean(left_rank)
    right_mean = statistics.mean(right_rank)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_rank, right_rank, strict=True))
    denominator = math.sqrt(
        sum((value - left_mean) ** 2 for value in left_rank)
        * sum((value - right_mean) ** 2 for value in right_rank)
    )
    return numerator / denominator if denominator else None


def decimal_percentiles(values: list[Decimal]) -> list[Decimal]:
    ranks = _average_ranks([float(value) for value in values])
    return [Decimal(str(rank)) / Decimal(len(values)) for rank in ranks] if values else []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True, help="DATE=RUN_RESULT_JSON")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-adjacent-moat-spearman", type=float, default=0.50)
    parser.add_argument("--minimum-stability-companies", type=int, default=20)
    parser.add_argument(
        "--skip-stability-gate",
        action="store_true",
        help="diagnostic ablations only; production signal panels should never use this",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    universe = read_csv(workspace / "inputs" / "universe.csv")
    date_rows = read_csv(workspace / "inputs" / "dates.csv")
    tickers = [row["stock_code"].zfill(6) for row in universe]
    universe_name = {row["stock_code"].zfill(6): row.get("name", "") for row in universe}
    universe_market = {row["stock_code"].zfill(6): row.get("market", "") for row in universe}
    universe_size = {row["stock_code"].zfill(6): row.get("size_bucket", "") for row in universe}
    universe_sector = {row["stock_code"].zfill(6): row.get("sector", "") for row in universe}
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
            if result.as_of.date().isoformat() != date:
                raise ValueError(
                    f"run as_of mismatch for {date}: {result.run_id} has {result.as_of.isoformat()}"
                )
            for company in result.companies:
                if company.ticker in by_ticker:
                    raise ValueError(
                        f"duplicate ticker across run inputs for {date}: {company.ticker}; "
                        "old/new or repair-overlay mixing is prohibited"
                    )
                if company.runner_version != RUNNER_VERSION:
                    raise ValueError(
                        f"incompatible runner version for {date}/{company.ticker}: "
                        f"{company.runner_version!r}, expected {RUNNER_VERSION}"
                    )
                by_ticker[company.ticker] = company
        for ticker in tickers:
            company = by_ticker.get(ticker)
            if company is None:
                rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "issuer_name": universe_name[ticker],
                        "market": universe_market[ticker],
                        "size_bucket": universe_size[ticker],
                        "sector": universe_sector[ticker],
                        "status": "NO_PIT_DOCUMENT",
                        "signal_eligible": 0,
                        "eligibility_reason": "NO_PIT_DOCUMENT",
                        "signal": "",
                        "signal_rank": "",
                        "moat_percentile": "",
                        "value_percentile": "",
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
            coverage = moat_coverage(company)
            reasons = []
            if company.status != CompanyRunStatus.COMPLETE:
                reasons.append(company.status.value)
            if score is None:
                reasons.append("NO_MOAT_SCORE")
            elif Decimal(str(score.economic_moat_score)) < Decimal("5"):
                reasons.append("MOAT_BELOW_5")
            if ratio is None or margin is None:
                reasons.append("NO_POSITIVE_DCF")
            elif margin < Decimal("0.20"):
                reasons.append("MARGIN_BELOW_20PCT")
            if company.dcf is not None and not company.dcf.screening_eligible:
                reasons.extend(company.dcf.screening_exclusion_reasons)
            if score is not None and Decimal(str(score.model_confidence)) < Decimal("0.50"):
                reasons.append("CONFIDENCE_BELOW_0_50")
            if score is not None and coverage < Decimal("0.50"):
                reasons.append("MOAT_COVERAGE_BELOW_0_50")
            eligible = not reasons
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "issuer_name": company.issuer_name or universe_name[ticker],
                    "market": universe_market[ticker],
                    "size_bucket": universe_size[ticker],
                    "sector": universe_sector[ticker],
                    "status": company.status.value,
                    "signal_eligible": int(eligible),
                    "eligibility_reason": "ELIGIBLE" if eligible else ";".join(reasons),
                    "signal": "",
                    "signal_rank": "",
                    "moat_percentile": "",
                    "value_percentile": "",
                    "moat_score": score.economic_moat_score if score else "",
                    "model_confidence": score.model_confidence if score else "",
                    "document_coverage": coverage if score else "",
                    "dcf_fair_value": fair_value or "",
                    "current_price": price or "",
                    "price_to_dcf": ratio or "",
                    "margin_of_safety": margin if margin is not None else "",
                }
            )

    stability = []
    for prior_date, current_date in zip(dates, dates[1:]):
        prior = {
            str(row["ticker"]): float(row["moat_score"])
            for row in rows
            if row["date"] == prior_date and row["moat_score"] != ""
        }
        current = {
            str(row["ticker"]): float(row["moat_score"])
            for row in rows
            if row["date"] == current_date and row["moat_score"] != ""
        }
        common = sorted(set(prior) & set(current))
        correlation = spearman([prior[ticker] for ticker in common], [current[ticker] for ticker in common])
        deltas = [abs(current[ticker] - prior[ticker]) for ticker in common]
        result = {
            "prior_date": prior_date,
            "current_date": current_date,
            "common_company_count": len(common),
            "spearman": correlation,
            "median_absolute_delta": statistics.median(deltas) if deltas else None,
            "delta_ge_5_count": sum(delta >= 5 for delta in deltas),
        }
        stability.append(result)
        if (
            not args.skip_stability_gate
            and len(common) < args.minimum_stability_companies
        ):
            raise RuntimeError(
                f"adjacent-quarter MOAT stability sample is too small for {prior_date}->{current_date}: "
                f"n={len(common)}, required={args.minimum_stability_companies}"
            )
        if (
            not args.skip_stability_gate
            and (correlation is None or correlation < args.minimum_adjacent_moat_spearman)
        ):
            raise RuntimeError(
                f"adjacent-quarter MOAT stability gate failed for {prior_date}->{current_date}: "
                f"rho={correlation}, required={args.minimum_adjacent_moat_spearman}"
            )

    for date in dates:
        ranked = [row for row in rows if row["date"] == date and row["signal_eligible"]]
        moat_percentiles = decimal_percentiles([Decimal(str(row["moat_score"])) for row in ranked])
        value_percentiles = decimal_percentiles([Decimal(str(row["margin_of_safety"])) for row in ranked])
        for row, moat_percentile, value_percentile in zip(
            ranked,
            moat_percentiles,
            value_percentiles,
            strict=True,
        ):
            row["moat_percentile"] = moat_percentile
            row["value_percentile"] = value_percentile
            row["signal"] = (moat_percentile + value_percentile) / Decimal(2)
        ranked.sort(key=lambda row: (-Decimal(str(row["signal"])), str(row["ticker"])))
        for rank, row in enumerate(ranked, start=1):
            row["signal_rank"] = rank

    fields = [
        "date", "ticker", "issuer_name", "market", "size_bucket", "sector",
        "status", "signal_eligible", "eligibility_reason", "signal",
        "signal_rank", "moat_percentile", "value_percentile",
        "moat_score", "model_confidence", "document_coverage",
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
        "schema_version": "moatrader-moat-dcf-signal/2",
        "signal_formula": "0.5 * cross_sectional_percentile(moat_score) + 0.5 * cross_sectional_percentile(margin_of_safety); confidence and MOAT coverage are eligibility gates",
        "fallback_signal": None,
        "row_count": len(rows),
        "expected_row_count": len(tickers) * len(dates),
        "universe_count": len(tickers),
        "dates": dates,
        "run_results": {
            date: [str(path) for path in paths]
            for date, paths in run_paths.items()
        },
        "moat_stability": stability,
        "minimum_adjacent_moat_spearman": args.minimum_adjacent_moat_spearman,
        "minimum_stability_companies": args.minimum_stability_companies,
        "stability_gate_skipped": args.skip_stability_gate,
        "universe_source_as_of": sorted({row.get("as_of", "") for row in universe if row.get("as_of")}),
        "universe_reconstitution": False,
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
    if any(bool(row["signal_eligible"]) != (row["signal"] != "") for row in rows):
        raise RuntimeError("signal eligibility and missing-value semantics disagree")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
