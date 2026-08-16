from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "moatrader-moat-contract-v2-holdouts/1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2
        for index in ordered[cursor:end]:
            ranks[index] = average
        cursor = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left, right, strict=True)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    return _pearson(_rank(left), _rank(right))


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _winsorize(values: list[float], lower: float = 0.05, upper: float = 0.95) -> list[float]:
    floor = _percentile(values, lower)
    ceiling = _percentile(values, upper)
    return [min(ceiling, max(floor, value)) for value in values]


def _load_companies(root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("run-result.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for company in payload.get("companies", []):
            ticker = str(company["ticker"]).zfill(6)
            if ticker in result:
                raise RuntimeError(f"duplicate company: {ticker}")
            result[ticker] = company
    return result


def _call_audit(root: Path) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for path in root.rglob("llm-calls.jsonl"):
        calls.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    tasks = Counter(str(call.get("task")) for call in calls)
    live = [call for call in calls if not call.get("replayed")]
    return {
        "call_count": len(calls),
        "live_call_count": len(live),
        "replayed_call_count": len(calls) - len(live),
        "task_counts": dict(tasks),
        "live_input_tokens": sum(
            int((call.get("usage") or {}).get("input_tokens") or 0) for call in live
        ),
        "live_output_tokens": sum(
            int((call.get("usage") or {}).get("output_tokens") or 0) for call in live
        ),
        "live_contextual_call_count": sum(
            call.get("task") == "CONTEXTUAL_MOAT_STRENGTH" for call in live
        ),
    }


def _bootstrap_spearman(
    scores: list[float], returns: list[float], *, samples: int = 10_000
) -> dict[str, float | int | None]:
    if len(scores) < 3:
        return {"samples": 0, "lower_95": None, "upper_95": None}
    rng = random.Random(20260816)
    values: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(scores)) for _ in scores]
        rho = _spearman([scores[i] for i in indices], [returns[i] for i in indices])
        if rho is not None:
            values.append(rho)
    return {
        "samples": len(values),
        "lower_95": _percentile(values, 0.025) if values else None,
        "upper_95": _percentile(values, 0.975) if values else None,
    }


def _permutation_p_value(
    scores: list[float], returns: list[float], observed: float | None, *, samples: int = 10_000
) -> float | None:
    if observed is None or len(scores) < 3:
        return None
    rng = random.Random(20260816)
    exceed = 0
    candidate = list(returns)
    for _ in range(samples):
        rng.shuffle(candidate)
        rho = _spearman(scores, candidate)
        if rho is not None and abs(rho) >= abs(observed):
            exceed += 1
    return (exceed + 1) / (samples + 1)


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    structural = report["structural_holdout"]
    economic = report["economic_validation"]
    lines = [
        "# MOAT contract v2 unseen and economic validation",
        "",
        f"- Structural holdout: **{'PASS' if structural['passed'] else 'FAIL'}** "
        f"({structural['completed_count']}/{structural['expected_count']} complete)",
        f"- Economic sample: {economic['completed_count']}/{economic['expected_count']} complete, "
        f"{economic['eligible_count']} score-eligible",
        f"- Forward MOAT rank IC: {_fmt(economic['moat_rank_ic'])}",
        f"- Winsorized rank IC: {_fmt(economic['winsorized_moat_rank_ic'])}",
        f"- Highest-minus-lowest score-bucket return: {_fmt(economic['highest_minus_lowest_bucket_return'])}",
        f"- Directional support: **{'YES' if economic['direction_supported'] else 'NO'}**",
        "",
        "## Structural holdout",
        "",
        f"- Audit FAIL: {structural['audit_fail_count']}",
        f"- Eligibility: {json.dumps(structural['eligibility_counts'], ensure_ascii=False)}",
        f"- Distinct eligible scores: {structural['distinct_eligible_scores']}",
        f"- Contextual calls: {structural['llm_calls']['live_contextual_call_count']}",
        "",
        "## Forward-return relationship by score bucket",
        "",
        "| MOAT score | Count | Mean return | Median return |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in economic["score_buckets"]:
        lines.append(
            f"| {row['score']:.2f} | {row['count']} | {row['mean_return']:.4f} | "
            f"{row['median_return']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Statistical caution",
            "",
            f"The deterministic bootstrap interval is "
            f"[{_fmt(economic['bootstrap_rank_ic_95']['lower_95'])}, "
            f"{_fmt(economic['bootstrap_rank_ic_95']['upper_95'])}] and the two-sided "
            f"permutation p-value is {_fmt(economic['rank_ic_permutation_p_value'])}. "
            "This is one cross-section, so it is a directional experiment rather than a "
            "claim of investable alpha.",
            "",
            "BRIDGE_FAIL and other ineligible records are excluded, not encoded as zero. "
            "DCF and combined metrics are reported only on the much smaller DCF-eligible subset.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate unseen and economic MOAT v2 holdouts.")
    parser.add_argument("--sample-root", required=True, type=Path)
    parser.add_argument("--structural-root", required=True, type=Path)
    parser.add_argument("--economic-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sample_root = args.sample_root.resolve()
    protocol = json.loads(
        (sample_root / "validation-protocol.json").read_text(encoding="utf-8-sig")
    )
    sample_meta = {
        row["ticker"].zfill(6): row
        for row in _read_csv(sample_root / "validation-samples.csv")
    }
    prices = {
        row["ticker"].zfill(6): row
        for row in _read_csv(sample_root / "economic-forward-prices.csv")
    }
    structural_companies = _load_companies(args.structural_root.resolve())
    economic_companies = _load_companies(args.economic_root.resolve())

    structural_tickers = [str(value).zfill(6) for value in protocol["structural"]["tickers"]]
    structural_rows = [structural_companies.get(ticker) for ticker in structural_tickers]
    structural_completed = [
        row for row in structural_rows if row and row.get("status") == "COMPLETE"
    ]
    structural_scores = [row["moat_score"] for row in structural_completed]
    structural_calls = _call_audit(args.structural_root.resolve())
    structural = {
        "expected_count": len(structural_tickers),
        "completed_count": len(structural_completed),
        "audit_fail_count": sum(score.get("audit_status") == "FAIL" for score in structural_scores),
        "eligibility_counts": dict(Counter(score.get("eligibility_status") for score in structural_scores)),
        "eligible_count": sum(bool(score.get("score_eligible", True)) for score in structural_scores),
        "distinct_eligible_scores": len(
            {
                float(score["economic_moat_score"])
                for score in structural_scores
                if score.get("score_eligible", True)
            }
        ),
        "llm_calls": structural_calls,
    }
    structural["passed"] = (
        structural["completed_count"] == structural["expected_count"]
        and structural["audit_fail_count"] == 0
        and structural_calls["live_contextual_call_count"] == structural["expected_count"]
    )

    economic_tickers = [str(value).zfill(6) for value in protocol["economic"]["tickers"]]
    panel: list[dict[str, Any]] = []
    for ticker in economic_tickers:
        company = economic_companies.get(ticker)
        price = prices[ticker]
        forward_return = float(price["return_price"]) / float(price["signal_price"]) - 1
        score = company.get("moat_score") if company else None
        dcf = company.get("dcf") if company else None
        dcf_eligible = bool(dcf and dcf.get("screening_eligible"))
        dcf_upside = (
            float(Decimal(str(dcf["fair_value_per_share"])) / Decimal(str(company["current_price"])) - 1)
            if dcf_eligible
            else None
        )
        panel.append(
            {
                "ticker": ticker,
                "company_name": sample_meta[ticker]["company_name"],
                "market": sample_meta[ticker]["market"],
                "size_bucket": sample_meta[ticker]["size_bucket"],
                "status": company.get("status") if company else "MISSING",
                "audit_status": score.get("audit_status") if score else None,
                "eligibility_status": score.get("eligibility_status") if score else None,
                "score_eligible": bool(score and score.get("score_eligible", True)),
                "moat_score": float(score["economic_moat_score"]) if score else None,
                "dcf_eligible": dcf_eligible,
                "dcf_upside": dcf_upside,
                "forward_return": forward_return,
                "old_score": float(sample_meta[ticker]["old_score_signal_date"]),
            }
        )

    completed = [row for row in panel if row["status"] == "COMPLETE"]
    eligible = [row for row in completed if row["score_eligible"]]
    scores = [float(row["moat_score"]) for row in eligible]
    returns = [float(row["forward_return"]) for row in eligible]
    rank_ic = _spearman(scores, returns)
    winsorized_returns = _winsorize(returns)
    winsorized_rank_ic = _spearman(scores, winsorized_returns)
    group_returns: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in eligible:
        group_returns[(row["market"], row["size_bucket"])].append(float(row["forward_return"]))
    group_means = {key: statistics.mean(values) for key, values in group_returns.items()}
    neutral_returns = [
        float(row["forward_return"]) - group_means[(row["market"], row["size_bucket"])]
        for row in eligible
    ]
    by_score: dict[float, list[float]] = defaultdict(list)
    for score, value in zip(scores, returns, strict=True):
        by_score[score].append(value)
    score_buckets = [
        {
            "score": score,
            "count": len(values),
            "mean_return": statistics.mean(values),
            "median_return": statistics.median(values),
        }
        for score, values in sorted(by_score.items())
    ]
    bucket_spread = (
        score_buckets[-1]["mean_return"] - score_buckets[0]["mean_return"]
        if len(score_buckets) >= 2
        else None
    )

    dcf_rows = [row for row in eligible if row["dcf_eligible"]]
    dcf_upside = [float(row["dcf_upside"]) for row in dcf_rows]
    dcf_returns = [float(row["forward_return"]) for row in dcf_rows]
    dcf_moat_scores = [float(row["moat_score"]) for row in dcf_rows]
    combined_rank = [
        a + b
        for a, b in zip(_rank(dcf_moat_scores), _rank(dcf_upside), strict=True)
    ] if dcf_rows else []
    economic_calls = _call_audit(args.economic_root.resolve())
    economic = {
        "expected_count": len(economic_tickers),
        "completed_count": len(completed),
        "audit_fail_count": sum(row["audit_status"] == "FAIL" for row in completed),
        "eligibility_counts": dict(Counter(row["eligibility_status"] for row in completed)),
        "eligible_count": len(eligible),
        "distinct_eligible_scores": len(by_score),
        "moat_rank_ic": rank_ic,
        "winsorized_moat_rank_ic": winsorized_rank_ic,
        "market_size_neutral_rank_ic": _spearman(scores, neutral_returns),
        "old_score_rank_ic": _spearman([float(row["old_score"]) for row in eligible], returns),
        "highest_minus_lowest_bucket_return": bucket_spread,
        "score_buckets": score_buckets,
        "bootstrap_rank_ic_95": _bootstrap_spearman(scores, returns),
        "rank_ic_permutation_p_value": _permutation_p_value(scores, returns, rank_ic),
        "dcf_eligible_count": len(dcf_rows),
        "dcf_upside_rank_ic": _spearman(dcf_upside, dcf_returns),
        "combined_moat_dcf_rank_ic": _spearman(combined_rank, dcf_returns),
        "direction_supported": bool(
            rank_ic is not None and rank_ic > 0 and bucket_spread is not None and bucket_spread > 0
        ),
        "llm_calls": economic_calls,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(sample_root / "validation-protocol.json"),
        "structural_holdout": structural,
        "economic_validation": economic,
        "interpretation": {
            "structural": "A fresh 10-name, notebook-unmentioned holdout after the frozen regression gate.",
            "economic": "One 24-name cross-section with one forward period; exploratory, not a production backtest.",
            "missing_scores": "Ineligible scores are excluded and are never replaced with zero.",
        },
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "economic-company-panel.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(panel[0]))
        writer.writeheader()
        writer.writerows(panel)
    (output / "holdout-validation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "holdout-validation-report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if structural["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
