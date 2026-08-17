from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "moatrader-moat-contract-v2-regression/1"


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


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    left_rank = _rank(left)
    right_rank = _rank(right)
    left_mean = statistics.mean(left_rank)
    right_mean = statistics.mean(right_rank)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_rank, right_rank, strict=True)
    )
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left_rank)
        * sum((b - right_mean) ** 2 for b in right_rank)
    )
    return numerator / denominator if denominator else None


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _load_runs(root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_date: dict[str, dict[str, dict[str, Any]]] = {}
    for path in sorted(root.rglob("run-result.json")):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        as_of = str(payload["as_of"])[:10]
        companies = by_date.setdefault(as_of, {})
        for company in payload.get("companies", []):
            ticker = str(company["ticker"]).zfill(6)
            if ticker in companies:
                raise RuntimeError(f"duplicate company for {as_of}: {ticker}")
            companies[ticker] = company
    return by_date


def _artifact(company: dict[str, Any], name: str) -> dict[str, Any]:
    path = Path(str(company["artifact_directory"])) / name
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.is_file() else {}


def _eligible_score(company: dict[str, Any] | None) -> dict[str, Any] | None:
    if not company or company.get("status") != "COMPLETE":
        return None
    score = company.get("moat_score")
    if not score or not score.get("score_eligible", True):
        return None
    return score


def _score_evidence_ids(score: dict[str, Any]) -> set[str]:
    return {
        evidence_id
        for mechanism in score.get("mechanisms", [])
        for evidence_id in mechanism.get("evidence_ids", [])
    } | set(score.get("counterevidence_ids", []))


def _reconciled_strength_signature(company: dict[str, Any]) -> tuple[Any, ...]:
    payload = _artifact(company, "moat-reconciliation.json")
    mechanisms = tuple(
        sorted(
            (
                item.get("candidate_id"),
                item.get("evidence_type"),
                item.get("strength_bucket"),
                item.get("scope_materiality_bucket"),
                item.get("durability_bucket"),
            )
            for item in payload.get("mechanisms", [])
        )
    )
    outcomes = tuple(
        sorted(
            (
                item.get("evidence_type"),
                item.get("strength_bucket"),
                item.get("persistence_bucket"),
            )
            for item in payload.get("outcomes", [])
        )
    )
    return mechanisms, outcomes


def _selected_context_ids(company: dict[str, Any]) -> set[str]:
    payload = _artifact(company, "moat-strength-context.json")
    return set(payload.get("selected_chunk_ids", []))


def _old_scores(protocol: dict[str, Any]) -> dict[tuple[str, str], float]:
    path = Path(protocol["inputs"]["old_scores"])
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {
            (row["stock_code"].zfill(6), row["as_of"]): float(row["economic_moat_score_100"])
            for row in csv.DictReader(stream)
            if row.get("economic_moat_score_100") not in {None, ""}
        }


def _check(criterion: str, value: Any, threshold: Any, passed: bool) -> dict[str, Any]:
    return {"criterion": criterion, "value": value, "threshold": threshold, "pass": passed}


def _fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MOAT contract v2 frozen regression",
        "",
        f"- Verdict: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Completed baseline cells: {report['completed_cell_count']}/{report['expected_cell_count']}",
        f"- Eligible baseline scores: {report['eligible_score_count']}",
        f"- Independent repeat cells: {report['repeatability']['common_company_count']}",
        "",
        "## Gate checks",
        "",
        "| Group | Criterion | Value | Threshold | Pass |",
        "| --- | --- | ---: | ---: | :---: |",
    ]
    for group in ("structural_checks", "repeat_checks", "legacy_protocol_diagnostics"):
        label = {
            "structural_checks": "Structural",
            "repeat_checks": "Repeat",
            "legacy_protocol_diagnostics": "Diagnostic",
        }[group]
        for item in report[group]:
            lines.append(
                f"| {label} | {item['criterion']} | {_fmt(item['value'])} | "
                f"{_fmt(item['threshold'])} | {'PASS' if item['pass'] else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            "## Date distribution",
            "",
            "| Date | Complete | Eligible | Distinct | Max single-score share | Old-score rho |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["date_metrics"]:
        lines.append(
            f"| {row['date']} | {row['completed_count']} | {row['eligible_count']} | "
            f"{row['distinct_scores']} | {_fmt(row['maximum_single_score_share'])} | "
            f"{_fmt(row['old_holistic_spearman'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Completion and score eligibility are intentionally separate. BRIDGE_FAIL, "
            "INSUFFICIENT, and validation failures do not enter score distributions or correlations. "
            "Candidate identity is the primary structural repeat metric; free-form canonical claim "
            "identity remains diagnostic only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate MOAT contract-v2 frozen and repeat runs.")
    parser.add_argument("--protocol-root", required=True, type=Path)
    parser.add_argument("--baseline-root", required=True, type=Path)
    parser.add_argument("--repeat-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    protocol_root = args.protocol_root.resolve()
    protocol = json.loads(
        (protocol_root / "FROZEN_protocol.json").read_text(encoding="utf-8-sig")
    )
    dates = list(protocol["dates"])
    tickers = [str(value).zfill(6) for value in protocol["sample_tickers"]]
    repeat_date = str(protocol["repeat_date"])
    thresholds = protocol["thresholds"]
    baseline = _load_runs(args.baseline_root.resolve())
    repeat = _load_runs(args.repeat_root.resolve())
    old_scores = _old_scores(protocol)

    expected = len(dates) * len(tickers)
    completed = 0
    eligible_count = 0
    audit_fail_count = 0
    date_metrics: list[dict[str, Any]] = []
    old_rhos: list[float] = []
    for date in dates:
        companies = baseline.get(date, {})
        values: list[float] = []
        anchors: list[float] = []
        completed_for_date = 0
        for ticker in tickers:
            company = companies.get(ticker)
            if company and company.get("status") == "COMPLETE":
                completed += 1
                completed_for_date += 1
                if (company.get("moat_score") or {}).get("audit_status") == "FAIL":
                    audit_fail_count += 1
            score = _eligible_score(company)
            if score is None:
                continue
            eligible_count += 1
            values.append(float(score["economic_moat_score"]))
            if (ticker, date) in old_scores:
                anchors.append(old_scores[(ticker, date)])
        counts = Counter(values)
        rho = _spearman(values, anchors) if len(values) == len(anchors) else None
        if rho is not None:
            old_rhos.append(rho)
        date_metrics.append(
            {
                "date": date,
                "completed_count": completed_for_date,
                "eligible_count": len(values),
                "distinct_scores": len(counts),
                "maximum_single_score_share": (
                    max(counts.values()) / len(values) if values else None
                ),
                "old_holistic_spearman": rho,
            }
        )

    baseline_repeat = baseline.get(repeat_date, {})
    independent_repeat = repeat.get(repeat_date, {})
    common = sorted(set(baseline_repeat) & set(independent_repeat) & set(tickers))
    eligibility_matches: list[bool] = []
    repeat_details: list[dict[str, Any]] = []
    before: list[float] = []
    after: list[float] = []
    for ticker in common:
        left_company = baseline_repeat[ticker]
        right_company = independent_repeat[ticker]
        left = _eligible_score(left_company)
        right = _eligible_score(right_company)
        eligibility_matches.append((left is None) == (right is None))
        detail: dict[str, Any] = {
            "ticker": ticker,
            "baseline_eligible": left is not None,
            "repeat_eligible": right is not None,
        }
        if left is not None and right is not None:
            left_value = float(left["economic_moat_score"])
            right_value = float(right["economic_moat_score"])
            before.append(left_value)
            after.append(right_value)
            detail.update(
                baseline_score=left_value,
                repeat_score=right_value,
                absolute_score_delta=abs(left_value - right_value),
                candidate_jaccard=_jaccard(
                    set(left.get("candidate_ids", [])), set(right.get("candidate_ids", []))
                ),
                evidence_jaccard=_jaccard(
                    _score_evidence_ids(left), _score_evidence_ids(right)
                ),
                selected_context_jaccard=_jaccard(
                    _selected_context_ids(left_company), _selected_context_ids(right_company)
                ),
                used_reference_jaccard=_jaccard(
                    set(left.get("context_reference_ids", [])),
                    set(right.get("context_reference_ids", [])),
                ),
                canonical_claim_jaccard=_jaccard(
                    set(left.get("canonical_claim_ids", [])),
                    set(right.get("canonical_claim_ids", [])),
                ),
                strength_attributes_equal=(
                    _reconciled_strength_signature(left_company)
                    == _reconciled_strength_signature(right_company)
                ),
            )
        repeat_details.append(detail)

    comparable = [row for row in repeat_details if "absolute_score_delta" in row]
    deltas = [float(row["absolute_score_delta"]) for row in comparable]
    score_rho = _spearman(before, after)
    if score_rho is None and deltas and all(delta == 0 for delta in deltas):
        score_rho = 1.0
    repeatability = {
        "common_company_count": len(common),
        "common_eligible_count": len(comparable),
        "eligibility_match_rate": statistics.mean(eligibility_matches) if eligibility_matches else None,
        "score_spearman": score_rho,
        "median_absolute_score_delta": statistics.median(deltas) if deltas else None,
        "maximum_absolute_score_delta": max(deltas) if deltas else None,
        "mean_candidate_jaccard": statistics.mean(row["candidate_jaccard"] for row in comparable) if comparable else None,
        "mean_evidence_jaccard": statistics.mean(row["evidence_jaccard"] for row in comparable) if comparable else None,
        "mean_selected_context_jaccard": statistics.mean(row["selected_context_jaccard"] for row in comparable) if comparable else None,
        "mean_used_reference_jaccard": statistics.mean(row["used_reference_jaccard"] for row in comparable) if comparable else None,
        "mean_canonical_claim_jaccard": statistics.mean(row["canonical_claim_jaccard"] for row in comparable) if comparable else None,
        "strength_attribute_match_rate": statistics.mean(row["strength_attributes_equal"] for row in comparable) if comparable else None,
        "companies": repeat_details,
    }

    structural_checks = [
        _check("completion_coverage", completed / expected, thresholds["coverage"], completed / expected >= thresholds["coverage"]),
        _check("audit_fail_count", audit_fail_count, thresholds["audit_fail_count"], audit_fail_count <= thresholds["audit_fail_count"]),
    ]
    repeat_checks = [
        _check("repeat_eligibility_match", repeatability["eligibility_match_rate"], 1.0, repeatability["eligibility_match_rate"] == 1.0),
        _check("repeat_score_spearman", score_rho, thresholds["minimum_repeat_score_spearman"], score_rho is not None and score_rho >= thresholds["minimum_repeat_score_spearman"]),
        _check("repeat_median_score_delta", repeatability["median_absolute_score_delta"], thresholds["maximum_repeat_median_score_delta"], repeatability["median_absolute_score_delta"] is not None and repeatability["median_absolute_score_delta"] <= thresholds["maximum_repeat_median_score_delta"]),
        _check("repeat_max_score_delta", repeatability["maximum_absolute_score_delta"], thresholds["maximum_repeat_company_score_delta"], repeatability["maximum_absolute_score_delta"] is not None and repeatability["maximum_absolute_score_delta"] <= thresholds["maximum_repeat_company_score_delta"]),
        _check("repeat_strength_attribute_match", repeatability["strength_attribute_match_rate"], thresholds["minimum_repeat_strength_attribute_match"], (repeatability["strength_attribute_match_rate"] or 0) >= thresholds["minimum_repeat_strength_attribute_match"]),
        _check("repeat_candidate_jaccard", repeatability["mean_candidate_jaccard"], thresholds["minimum_repeat_claim_jaccard"], (repeatability["mean_candidate_jaccard"] or 0) >= thresholds["minimum_repeat_claim_jaccard"]),
        _check("repeat_selected_context_jaccard", repeatability["mean_selected_context_jaccard"], thresholds["minimum_repeat_context_jaccard"], (repeatability["mean_selected_context_jaccard"] or 0) >= thresholds["minimum_repeat_context_jaccard"]),
    ]
    minimum_old_rho = thresholds["minimum_mean_old_holistic_spearman"]
    legacy_protocol_diagnostics = [
        _check("minimum_distinct_eligible_scores_each_date", min(row["distinct_scores"] for row in date_metrics), thresholds["minimum_distinct_scores_each_date"], all(row["distinct_scores"] >= thresholds["minimum_distinct_scores_each_date"] for row in date_metrics)),
        _check("maximum_single_eligible_score_share_each_date", max((row["maximum_single_score_share"] or 1.0) for row in date_metrics), thresholds["maximum_single_score_share_each_date"], all((row["maximum_single_score_share"] or 1.0) <= thresholds["maximum_single_score_share_each_date"] for row in date_metrics)),
        _check("mean_old_holistic_spearman_on_eligible", statistics.mean(old_rhos) if old_rhos else None, minimum_old_rho, bool(old_rhos) and statistics.mean(old_rhos) >= minimum_old_rho),
        _check("positive_old_holistic_dates", sum(value > 0 for value in old_rhos), thresholds["minimum_positive_old_holistic_dates"], sum(value > 0 for value in old_rhos) >= thresholds["minimum_positive_old_holistic_dates"]),
        _check("repeat_atomic_evidence_jaccard", repeatability["mean_evidence_jaccard"], thresholds["minimum_repeat_evidence_jaccard"], (repeatability["mean_evidence_jaccard"] or 0) >= thresholds["minimum_repeat_evidence_jaccard"]),
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_root / "FROZEN_protocol.json"),
        "baseline_root": str(args.baseline_root.resolve()),
        "repeat_root": str(args.repeat_root.resolve()),
        "expected_cell_count": expected,
        "completed_cell_count": completed,
        "eligible_score_count": eligible_count,
        "audit_fail_count": audit_fail_count,
        "passed": all(item["pass"] for item in [*structural_checks, *repeat_checks]),
        "structural_checks": structural_checks,
        "repeat_checks": repeat_checks,
        "legacy_protocol_diagnostics": legacy_protocol_diagnostics,
        "date_metrics": date_metrics,
        "repeatability": repeatability,
        "diagnostics": {
            "canonical_claim_jaccard": repeatability["mean_canonical_claim_jaccard"],
            "used_reference_jaccard": repeatability["mean_used_reference_jaccard"],
            "gate_semantics": (
                "Contract-v2 pass uses the feedback-specified completion, audit, "
                "eligibility, score/attribute stability, Candidate ID, and selected-context "
                "checks. Legacy score-distribution and raw atomic-ID thresholds remain "
                "reported but cannot override structural identity after Candidate ID adoption."
            ),
        },
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "frozen-regression-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "frozen-regression-report.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
