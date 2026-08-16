from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_STRATA = (
    ("KOSDAQ", "LARGE", "HIGH"),
    ("KOSDAQ", "LARGE", "LOW"),
    ("KOSDAQ", "MID", "MID"),
    ("KOSDAQ", "SMALL", "LOW"),
    ("KOSPI", "LARGE", "HIGH"),
    ("KOSPI", "MID", "LOW"),
    ("KOSPI", "MID", "MID"),
    ("KOSPI", "SMALL", "HIGH"),
)
PROTOCOL_SCHEMA = "moatrader-moat-shadow-holdout/1"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _band(score: float) -> str:
    if score <= 25:
        return "LOW"
    if score <= 50:
        return "MID"
    return "HIGH"


def _rank(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
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


def _prepare(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"shadow holdout directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    workspace = args.workspace.resolve()
    workspace_manifest_path = workspace / "workspace-manifest.json"
    workspace_manifest = json.loads(workspace_manifest_path.read_text(encoding="utf-8-sig"))
    dates = list(workspace_manifest["dates"])
    if len(dates) != 4:
        raise ValueError(f"expected four dates, got {dates}")

    universe_rows = _read_csv(args.universe.resolve())
    universe = {row["stock_code"].zfill(6): row for row in universe_rows}
    notebook_text = args.notebook.resolve().read_text(encoding="utf-8")
    mentioned = set(re.findall(r"(?<!\d)\d{6}(?!\d)", notebook_text)) & set(universe)

    old_rows = _read_csv(args.old_scores.resolve())
    old_scores = {
        (row["stock_code"].zfill(6), row["as_of"]): float(row["economic_moat_score_100"])
        for row in old_rows
        if row.get("economic_moat_score_100") not in {None, ""}
    }

    manifests: dict[str, list[dict[str, str]]] = {}
    prices: dict[tuple[str, str], tuple[float, str]] = {}
    for date in dates:
        rows = _read_csv(workspace / "date-inputs" / date / "universe-manifest.csv")
        manifests[date] = rows
        for row in rows:
            ticker = row["ticker"].zfill(6)
            key = (ticker, date)
            if row.get("current_price") and key not in prices:
                prices[key] = (float(row["current_price"]), row.get("price_as_of", ""))

    candidates: list[dict[str, Any]] = []
    for ticker, row in universe.items():
        if ticker in mentioned:
            continue
        if not all((ticker, date) in old_scores for date in dates):
            continue
        if not all((ticker, date) in prices for date in dates):
            continue
        score = old_scores[(ticker, dates[0])]
        selection_hash = hashlib.sha256(f"{args.seed}:{ticker}".encode("utf-8")).hexdigest()
        candidates.append(
            {
                "ticker": ticker,
                "company_name": row.get("name", ""),
                "market": row.get("market", ""),
                "size_bucket": row.get("size_bucket", ""),
                "old_holistic_score_first_date": score,
                "old_holistic_band": _band(score),
                "selection_hash": selection_hash,
            }
        )

    selected: list[dict[str, Any]] = []
    for market, size_bucket, band in DEFAULT_STRATA:
        matching = [
            row
            for row in candidates
            if (row["market"], row["size_bucket"], row["old_holistic_band"])
            == (market, size_bucket, band)
        ]
        if not matching:
            raise RuntimeError(f"no candidate for frozen stratum {(market, size_bucket, band)}")
        chosen = min(matching, key=lambda row: row["selection_hash"])
        chosen["selection_stratum"] = f"{market}|{size_bucket}|{band}"
        selected.append(chosen)

    tickers = [row["ticker"] for row in selected]
    sample_fields = (
        "ticker",
        "company_name",
        "market",
        "size_bucket",
        "old_holistic_score_first_date",
        "old_holistic_band",
        "selection_stratum",
        "selection_hash",
    )
    _write_csv(output / "FROZEN_sample.csv", selected, sample_fields)

    price_rows = []
    for date in dates:
        for ticker in tickers:
            price, price_as_of = prices[(ticker, date)]
            price_rows.append(
                {"date": date, "ticker": ticker, "price": price, "price_as_of": price_as_of}
            )
    _write_csv(output / "prices.csv", price_rows, ("date", "ticker", "price", "price_as_of"))

    batches = [tickers[:4], tickers[4:]]
    manifest_hashes: dict[str, str] = {}
    for date in dates:
        source_rows = manifests[date]
        fieldnames = list(source_rows[0])
        for batch_index, batch in enumerate(batches, start=1):
            batch_rows = [row for row in source_rows if row["ticker"].zfill(6) in set(batch)]
            found = {row["ticker"].zfill(6) for row in batch_rows}
            if found != set(batch):
                raise RuntimeError(f"{date} batch {batch_index} missing tickers: {sorted(set(batch) - found)}")
            path = output / "manifests" / date / f"batch-{batch_index}.csv"
            _write_csv(path, batch_rows, fieldnames)
            manifest_hashes[path.relative_to(output).as_posix()] = _sha256(path)

    sample_hash = _sha256(output / "FROZEN_sample.csv")
    experiment_id = f"moat-shadow-{args.seed}-{sample_hash[:12]}"
    thresholds = {
        "coverage": 1.0,
        "audit_fail_count": 0,
        "minimum_distinct_scores_each_date": 3,
        "maximum_single_score_share_each_date": 0.625,
        "minimum_mean_old_holistic_spearman": 0.50,
        "minimum_positive_old_holistic_dates": 3,
        "minimum_repeat_score_spearman": 0.90,
        "maximum_repeat_median_score_delta": 0.50,
        "maximum_repeat_company_score_delta": 2.0,
        "minimum_repeat_strength_attribute_match": 0.90,
        "minimum_repeat_evidence_jaccard": 0.50,
        "minimum_repeat_claim_jaccard": 0.50,
        "minimum_repeat_context_jaccard": 1.0,
        "exploratory_minimum_mean_forward_rank_ic": 0.10,
        "exploratory_minimum_positive_return_periods": 2,
        "exploratory_minimum_mean_top2_minus_bottom2": 0.0,
    }
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": experiment_id,
        "seed": args.seed,
        "selection_method": "notebook-unmentioned; complete four-date data; fixed market-size-old-score strata; minimum SHA256(seed:ticker)",
        "notebook_mentioned_ticker_count": len(mentioned),
        "eligible_candidate_count": len(candidates),
        "sample_tickers": tickers,
        "dates": dates,
        "unique_baseline_cells": len(tickers) * len(dates),
        "repeat_date": dates[0],
        "unique_repeat_cells": len(tickers),
        "total_fresh_scoring_cells": len(tickers) * len(dates) + len(tickers),
        "thresholds": thresholds,
        "inputs": {
            "universe": str(args.universe.resolve()),
            "universe_sha256": _sha256(args.universe.resolve()),
            "dates_sha256": _sha256(workspace / "inputs" / "dates.csv"),
            "notebook": str(args.notebook.resolve()),
            "notebook_sha256": _sha256(args.notebook.resolve()),
            "old_scores": str(args.old_scores.resolve()),
            "old_scores_sha256": _sha256(args.old_scores.resolve()),
            "source_tree_sha256": _tree_sha256(args.source_root.resolve()),
            "manifest_sha256": manifest_hashes,
        },
        "interpretation": {
            "old_holistic_relation": "bridge diagnostic, not ground truth",
            "forward_return_relation": "exploratory because n=8 per date and three realized periods",
            "production_decision": "requires both structural/repeat gates; return relation alone cannot approve production",
        },
    }
    (output / "FROZEN_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "workspace-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": PROTOCOL_SCHEMA,
                "created_at": protocol["frozen_at"],
                "experiment_id": experiment_id,
                "fresh_run": True,
                "source_result_reuse": False,
                "universe_count": len(tickers),
                "dates": dates,
                "expected_signal_count": len(tickers) * len(dates),
                "preflight_required": False,
                "preflight_status": "NOT_REQUIRED",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(protocol, ensure_ascii=False, indent=2))
    return 0


def _company_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["ticker"].zfill(6): item
        for item in result.get("companies", [])
        if item.get("status") == "COMPLETE" and item.get("moat_score") is not None
    }


def _load_merged_runs(root: Path, dates: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for date in dates:
        companies: dict[str, dict[str, Any]] = {}
        for path in sorted((root / date).glob("batch-*/run-result.json")):
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            overlap = set(companies) & set(_company_map(payload))
            if overlap:
                raise RuntimeError(f"duplicate companies in {date}: {sorted(overlap)}")
            companies.update(_company_map(payload))
        merged[date] = companies
    return merged


def _load_all_runs(root: Path, dates: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    merged: dict[str, dict[str, dict[str, Any]]] = {}
    for date in dates:
        companies: dict[str, dict[str, Any]] = {}
        for path in sorted((root / date).glob("batch-*/run-result.json")):
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
            for item in payload.get("companies", []):
                ticker = item["ticker"].zfill(6)
                if ticker in companies:
                    raise RuntimeError(f"duplicate company in {date}: {ticker}")
                companies[ticker] = item
        merged[date] = companies
    return merged


def _failure_categories(error: str) -> list[str]:
    patterns = {
        "RAW_QUOTE_MISMATCH": "raw quote is not in",
        "NODE_ID_OUTSIDE_CHUNK": "node IDs outside",
        "UNSUPPORTED_NUMBER": "unsupported numbers",
        "UNKNOWN_CHUNK": "unknown chunk",
        "INVALID_RISK_RUBRIC_ENUM": "outside the risk rubric",
        "INDUSTRY_SCOPE_AS_COMPANY_MOAT": "scope is INDUSTRY",
        "PRODUCT_CATEGORY_SCOPE_AS_COMPANY_MOAT": "scope is PRODUCT_CATEGORY",
        "DUPLICATE_ASSESSMENT": "duplicate ",
        "POSITIVE_DURABILITY_WITHOUT_MECHANISM": "positive durability requires",
    }
    return [category for category, pattern in patterns.items() if pattern in error]


def _call_audit(root: Path) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for path in root.rglob("llm-calls.jsonl"):
        with path.open("r", encoding="utf-8-sig") as stream:
            calls.extend(json.loads(line) for line in stream if line.strip())
    by_task: dict[str, Any] = {}
    for task in sorted({str(call.get("task")) for call in calls}):
        task_calls = [call for call in calls if str(call.get("task")) == task]
        live = [call for call in task_calls if not call.get("replayed")]
        by_task[task] = {
            "call_count": len(task_calls),
            "live_call_count": len(live),
            "replayed_call_count": len(task_calls) - len(live),
            "live_input_tokens": sum(int((call.get("usage") or {}).get("input_tokens") or 0) for call in live),
            "live_output_tokens": sum(int((call.get("usage") or {}).get("output_tokens") or 0) for call in live),
            "live_cached_input_tokens": sum(int((call.get("usage") or {}).get("cached_input_tokens") or 0) for call in live),
        }
    return {
        "call_count": len(calls),
        "live_call_count": sum(not call.get("replayed") for call in calls),
        "replayed_call_count": sum(bool(call.get("replayed")) for call in calls),
        "models": dict(Counter(str(call.get("model")) for call in calls)),
        "by_task": by_task,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# MOAT Dual-Lane shadow holdout",
        "",
        f"- Verdict: **{'PASS' if report['production_ready'] else 'FAIL — not production-ready'}**",
        f"- Baseline coverage: {report['scored_cell_count']}/{report['baseline_cell_count']} "
        f"({report['scored_cell_count'] / report['baseline_cell_count']:.1%})",
        f"- Structural bridge: {'PASS' if report['structural_bridge_passed'] else 'FAIL'}",
        f"- Independent repeat: {'PASS' if report['independent_repeat_passed'] else 'FAIL'}",
        f"- Exploratory return direction: {'PASS' if report['exploratory_return_direction_passed'] else 'FAIL'}",
        "",
        "## Pre-registered checks",
        "",
        "| Group | Criterion | Value | Threshold | Pass |",
        "| --- | --- | ---: | ---: | :---: |",
    ]
    for group, checks in (
        ("Structural", report["structural_checks"]),
        ("Repeat", report["repeat_checks"]),
        ("Exploratory", report["exploratory_checks"]),
    ):
        for check in checks:
            value = check["value"]
            value_text = "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)
            lines.append(
                f"| {group} | {check['criterion']} | {value_text} | {check['threshold']} | "
                f"{'✅' if check['pass'] else '❌'} |"
            )
    lines.extend(
        [
            "",
            "## Coverage and relationship by date",
            "",
            "| Date | Scored | Distinct | Max bucket | Old holistic rho | Forward rank IC | Top2-Bottom2 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["date_metrics"]:
        def fmt(value: Any) -> str:
            return "N/A" if value is None else f"{value:.4f}"

        lines.append(
            f"| {row['date']} | {row['scored_count']} | {row['distinct_scores']} | "
            f"{fmt(row['maximum_single_score_share'])} | {fmt(row['old_holistic_spearman'])} | "
            f"{fmt(row['forward_rank_ic'])} | {fmt(row['top2_minus_bottom2_return'])} |"
        )
    lines.extend(
        [
            "",
            "## Independent repeat status transitions",
            "",
            "| Transition | Count |",
            "| --- | ---: |",
        ]
    )
    for transition, count in report["repeat_status_transitions"].items():
        lines.append(f"| {transition} | {count} |")
    lines.extend(
        [
            "",
            "## Baseline failure categories",
            "",
            "| Category | Failed companies |",
            "| --- | ---: |",
        ]
    )
    for category, count in report["failure_analysis"]["baseline_category_company_counts"].items():
        lines.append(f"| {category} | {count} |")
    baseline_calls = report["llm_call_audit"]["baseline"]
    repeat_calls = report["llm_call_audit"]["independent_repeat"]
    lines.extend(
        [
            "",
            "## API usage audit",
            "",
            f"- Baseline: {baseline_calls['live_call_count']} live calls, "
            f"{sum(item['live_input_tokens'] for item in baseline_calls['by_task'].values()):,} input tokens.",
            f"- Independent repeat: {repeat_calls['live_call_count']} live calls, "
            f"{sum(item['live_input_tokens'] for item in repeat_calls['by_task'].values()):,} input tokens.",
            f"- Baseline contextual lane: "
            f"{baseline_calls['by_task'].get('CONTEXTUAL_MOAT_STRENGTH', {}).get('live_input_tokens', 0):,} input tokens.",
            "",
            "## Interpretation",
            "",
            "The run fails before a credible economic-signal conclusion can be drawn. Coverage, audit status, "
            "independent-repeat stability, bridge correlation, and the exploratory return direction all miss their "
            "frozen thresholds. The strict validator is successfully rejecting ungrounded outputs, but the current "
            "prompt/response contract does not produce valid grounded assessments reliably enough.",
            "",
            "The return statistics use only completed names (3–6 per date), so they are descriptive and must not be "
            "treated as evidence that the factor is economically dead or alive.",
            "",
        ]
    )
    return "\n".join(lines)


def _strength_signature(score: dict[str, Any]) -> tuple[Any, ...]:
    mechanisms = tuple(
        sorted(
            (
                item.get("evidence_type"),
                item.get("strength_bucket"),
                item.get("scope_materiality_bucket"),
            )
            for item in score.get("mechanisms", [])
        )
    )
    outcomes = tuple(
        sorted(
            (
                item.get("evidence_type"),
                item.get("strength_bucket"),
                item.get("persistence_bucket"),
            )
            for item in score.get("outcome_strengths", [])
        )
    )
    return (
        mechanisms,
        outcomes,
        score.get("durability"),
        score.get("audit_status"),
        score.get("scoring_method"),
    )


def _score_evidence_ids(score: dict[str, Any]) -> set[str]:
    return {
        evidence_id
        for item in score.get("mechanisms", [])
        for evidence_id in item.get("evidence_ids", [])
    } | set(score.get("counterevidence_ids", []))


def _repeat_report(
    baseline: dict[str, dict[str, Any]], repeat: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    common = sorted(set(baseline) & set(repeat))
    before = [float(baseline[t]["moat_score"]["economic_moat_score"]) for t in common]
    after = [float(repeat[t]["moat_score"]["economic_moat_score"]) for t in common]
    deltas = [abs(a - b) for a, b in zip(before, after, strict=True)]
    details = []
    for ticker, left_value, right_value, delta in zip(common, before, after, deltas, strict=True):
        left = baseline[ticker]["moat_score"]
        right = repeat[ticker]["moat_score"]
        details.append(
            {
                "ticker": ticker,
                "baseline_score": left_value,
                "repeat_score": right_value,
                "absolute_score_delta": delta,
                "evidence_jaccard": _jaccard(_score_evidence_ids(left), _score_evidence_ids(right)),
                "claim_jaccard": _jaccard(
                    set(left.get("canonical_claim_ids", [])), set(right.get("canonical_claim_ids", []))
                ),
                "context_jaccard": _jaccard(
                    set(left.get("context_chunk_ids", [])), set(right.get("context_chunk_ids", []))
                ),
                "strength_attributes_equal": _strength_signature(left) == _strength_signature(right),
            }
        )
    score_rho = _spearman(before, after)
    if score_rho is None and all(delta == 0 for delta in deltas):
        score_rho = 1.0
    return {
        "common_company_count": len(common),
        "missing_tickers": sorted(set(baseline) - set(repeat)),
        "added_tickers": sorted(set(repeat) - set(baseline)),
        "score_spearman": score_rho,
        "median_absolute_score_delta": statistics.median(deltas) if deltas else None,
        "maximum_absolute_score_delta": max(deltas) if deltas else None,
        "mean_evidence_jaccard": statistics.mean(x["evidence_jaccard"] for x in details) if details else None,
        "mean_claim_jaccard": statistics.mean(x["claim_jaccard"] for x in details) if details else None,
        "mean_context_jaccard": statistics.mean(x["context_jaccard"] for x in details) if details else None,
        "strength_attribute_match_rate": statistics.mean(x["strength_attributes_equal"] for x in details) if details else None,
        "companies": details,
    }


def _analyze(args: argparse.Namespace) -> int:
    root = args.output.resolve()
    protocol_path = root / "FROZEN_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    if protocol.get("schema_version") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported shadow holdout protocol")
    dates = list(protocol["dates"])
    tickers = list(protocol["sample_tickers"])
    thresholds = protocol["thresholds"]
    baseline = _load_merged_runs(root / "runs" / "baseline", dates)
    repeat_date = protocol["repeat_date"]
    repeat = _load_merged_runs(root / "runs" / "repeat", [repeat_date])[repeat_date]
    baseline_all = _load_all_runs(root / "runs" / "baseline", dates)
    repeat_all = _load_all_runs(root / "runs" / "repeat", [repeat_date])[repeat_date]

    old_scores = {
        (row["stock_code"].zfill(6), row["as_of"]): float(row["economic_moat_score_100"])
        for row in _read_csv(Path(protocol["inputs"]["old_scores"]))
        if row.get("economic_moat_score_100") not in {None, ""}
    }
    legacy_rows = _read_csv(args.legacy_signals.resolve())
    legacy_scores = {
        (row["ticker"].zfill(6), row["date"]): float(row["moat_score"])
        for row in legacy_rows
        if row.get("moat_score") not in {None, ""}
    }
    prices = {
        (row["ticker"].zfill(6), row["date"]): float(row["price"])
        for row in _read_csv(root / "prices.csv")
    }

    company_rows: list[dict[str, Any]] = []
    date_metrics: list[dict[str, Any]] = []
    for date_index, date in enumerate(dates):
        current = baseline[date]
        scores: list[float] = []
        anchors: list[float] = []
        legacy: list[float] = []
        forward: list[float] = []
        forward_scores: list[float] = []
        forward_tickers: list[str] = []
        for ticker in tickers:
            company = current.get(ticker)
            score_payload = company.get("moat_score") if company else None
            score = float(score_payload["economic_moat_score"]) if score_payload else None
            next_return = None
            if date_index + 1 < len(dates):
                next_date = dates[date_index + 1]
                next_return = prices[(ticker, next_date)] / prices[(ticker, date)] - 1
            company_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "status": company.get("status") if company else "MISSING",
                    "current_score": score,
                    "audit_status": score_payload.get("audit_status") if score_payload else None,
                    "evidence_confidence": score_payload.get("evidence_confidence") if score_payload else None,
                    "mechanism_count": len(score_payload.get("mechanisms", [])) if score_payload else None,
                    "outcome_count": len(score_payload.get("outcome_strengths", [])) if score_payload else None,
                    "old_holistic_score": old_scores.get((ticker, date)),
                    "legacy_atomic_score": legacy_scores.get((ticker, date)),
                    "next_period_return": next_return,
                }
            )
            if score is None:
                continue
            scores.append(score)
            anchors.append(old_scores[(ticker, date)])
            legacy.append(legacy_scores[(ticker, date)])
            if next_return is not None:
                forward_scores.append(score)
                forward.append(next_return)
                forward_tickers.append(ticker)
        counts = Counter(scores)
        top_bottom = None
        if forward_scores:
            ordered = sorted(
                zip(forward_scores, forward_tickers, forward, strict=True),
                key=lambda x: (x[0], x[1]),
            )
            top_bottom = statistics.mean(x[2] for x in ordered[-2:]) - statistics.mean(
                x[2] for x in ordered[:2]
            )
        date_metrics.append(
            {
                "date": date,
                "scored_count": len(scores),
                "distinct_scores": len(counts),
                "maximum_single_score_share": max(counts.values()) / len(scores) if scores else None,
                "old_holistic_spearman": _spearman(scores, anchors),
                "legacy_atomic_spearman": _spearman(scores, legacy),
                "forward_rank_ic": _spearman(forward_scores, forward) if forward_scores else None,
                "top2_minus_bottom2_return": top_bottom,
            }
        )

    expected = len(tickers) * len(dates)
    scored = sum(row["current_score"] is not None for row in company_rows)
    audit_fail_count = sum(row["audit_status"] == "FAIL" for row in company_rows)
    old_rhos = [row["old_holistic_spearman"] for row in date_metrics if row["old_holistic_spearman"] is not None]
    return_rhos = [row["forward_rank_ic"] for row in date_metrics if row["forward_rank_ic"] is not None]
    spreads = [
        row["top2_minus_bottom2_return"]
        for row in date_metrics
        if row["top2_minus_bottom2_return"] is not None
    ]
    repeat_report = _repeat_report(baseline[repeat_date], repeat)
    status_transitions = Counter(
        (
            baseline_all[repeat_date].get(ticker, {}).get("status", "MISSING"),
            repeat_all.get(ticker, {}).get("status", "MISSING"),
        )
        for ticker in tickers
    )
    failure_category_counts: Counter[str] = Counter()
    failed_company_count = 0
    for companies in baseline_all.values():
        for company in companies.values():
            if company.get("status") != "FAILED":
                continue
            failed_company_count += 1
            failure_category_counts.update(_failure_categories(company.get("error") or ""))
    repeat_failure_category_counts: Counter[str] = Counter()
    repeat_failed_company_count = 0
    for company in repeat_all.values():
        if company.get("status") != "FAILED":
            continue
        repeat_failed_company_count += 1
        repeat_failure_category_counts.update(_failure_categories(company.get("error") or ""))

    def usage_summary(all_runs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, int]:
        usage = Counter()
        for companies in all_runs.values():
            for company in companies.values():
                for key in ("input_tokens", "output_tokens", "cached_input_tokens", "cache_write_tokens"):
                    usage[key] += int((company.get("llm_usage") or {}).get(key) or 0)
        return dict(usage)

    structural_checks = [
        {"criterion": "coverage", "value": scored / expected, "threshold": thresholds["coverage"], "pass": scored / expected >= thresholds["coverage"]},
        {"criterion": "audit_fail_count", "value": audit_fail_count, "threshold": thresholds["audit_fail_count"], "pass": audit_fail_count <= thresholds["audit_fail_count"]},
        {"criterion": "minimum_distinct_scores_each_date", "value": min(row["distinct_scores"] for row in date_metrics), "threshold": thresholds["minimum_distinct_scores_each_date"], "pass": all(row["distinct_scores"] >= thresholds["minimum_distinct_scores_each_date"] for row in date_metrics)},
        {"criterion": "maximum_single_score_share_each_date", "value": max(row["maximum_single_score_share"] or 1 for row in date_metrics), "threshold": thresholds["maximum_single_score_share_each_date"], "pass": all((row["maximum_single_score_share"] or 1) <= thresholds["maximum_single_score_share_each_date"] for row in date_metrics)},
        {"criterion": "mean_old_holistic_spearman", "value": statistics.mean(old_rhos) if old_rhos else None, "threshold": thresholds["minimum_mean_old_holistic_spearman"], "pass": bool(old_rhos) and statistics.mean(old_rhos) >= thresholds["minimum_mean_old_holistic_spearman"]},
        {"criterion": "positive_old_holistic_dates", "value": sum(value > 0 for value in old_rhos), "threshold": thresholds["minimum_positive_old_holistic_dates"], "pass": sum(value > 0 for value in old_rhos) >= thresholds["minimum_positive_old_holistic_dates"]},
    ]
    repeat_checks = [
        {"criterion": "repeat_score_spearman", "value": repeat_report["score_spearman"], "threshold": thresholds["minimum_repeat_score_spearman"], "pass": (repeat_report["score_spearman"] or -1) >= thresholds["minimum_repeat_score_spearman"]},
        {"criterion": "repeat_median_score_delta", "value": repeat_report["median_absolute_score_delta"], "threshold": thresholds["maximum_repeat_median_score_delta"], "pass": repeat_report["median_absolute_score_delta"] is not None and repeat_report["median_absolute_score_delta"] <= thresholds["maximum_repeat_median_score_delta"]},
        {"criterion": "repeat_max_score_delta", "value": repeat_report["maximum_absolute_score_delta"], "threshold": thresholds["maximum_repeat_company_score_delta"], "pass": repeat_report["maximum_absolute_score_delta"] is not None and repeat_report["maximum_absolute_score_delta"] <= thresholds["maximum_repeat_company_score_delta"]},
        {"criterion": "repeat_strength_attribute_match", "value": repeat_report["strength_attribute_match_rate"], "threshold": thresholds["minimum_repeat_strength_attribute_match"], "pass": (repeat_report["strength_attribute_match_rate"] or 0) >= thresholds["minimum_repeat_strength_attribute_match"]},
        {"criterion": "repeat_evidence_jaccard", "value": repeat_report["mean_evidence_jaccard"], "threshold": thresholds["minimum_repeat_evidence_jaccard"], "pass": (repeat_report["mean_evidence_jaccard"] or 0) >= thresholds["minimum_repeat_evidence_jaccard"]},
        {"criterion": "repeat_claim_jaccard", "value": repeat_report["mean_claim_jaccard"], "threshold": thresholds["minimum_repeat_claim_jaccard"], "pass": (repeat_report["mean_claim_jaccard"] or 0) >= thresholds["minimum_repeat_claim_jaccard"]},
        {"criterion": "repeat_context_jaccard", "value": repeat_report["mean_context_jaccard"], "threshold": thresholds["minimum_repeat_context_jaccard"], "pass": (repeat_report["mean_context_jaccard"] or 0) >= thresholds["minimum_repeat_context_jaccard"]},
    ]
    exploratory_checks = [
        {"criterion": "mean_forward_rank_ic", "value": statistics.mean(return_rhos) if return_rhos else None, "threshold": thresholds["exploratory_minimum_mean_forward_rank_ic"], "pass": bool(return_rhos) and statistics.mean(return_rhos) >= thresholds["exploratory_minimum_mean_forward_rank_ic"]},
        {"criterion": "positive_return_periods", "value": sum(value > 0 for value in return_rhos), "threshold": thresholds["exploratory_minimum_positive_return_periods"], "pass": sum(value > 0 for value in return_rhos) >= thresholds["exploratory_minimum_positive_return_periods"]},
        {"criterion": "mean_top2_minus_bottom2", "value": statistics.mean(spreads) if spreads else None, "threshold": thresholds["exploratory_minimum_mean_top2_minus_bottom2"], "pass": bool(spreads) and statistics.mean(spreads) >= thresholds["exploratory_minimum_mean_top2_minus_bottom2"]},
    ]
    report = {
        "schema_version": PROTOCOL_SCHEMA,
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": protocol["experiment_id"],
        "sample_size": len(tickers),
        "baseline_cell_count": expected,
        "scored_cell_count": scored,
        "structural_bridge_passed": all(row["pass"] for row in structural_checks),
        "independent_repeat_passed": all(row["pass"] for row in repeat_checks),
        "exploratory_return_direction_passed": all(row["pass"] for row in exploratory_checks),
        "production_ready": all(row["pass"] for row in [*structural_checks, *repeat_checks]),
        "structural_checks": structural_checks,
        "repeat_checks": repeat_checks,
        "exploratory_checks": exploratory_checks,
        "date_metrics": date_metrics,
        "repeatability": repeat_report,
        "completion_by_date": {
            date: Counter(company.get("status", "MISSING") for company in baseline_all[date].values())
            for date in dates
        },
        "repeat_status_transitions": {
            f"{before}->{after}": count
            for (before, after), count in sorted(status_transitions.items())
        },
        "failure_analysis": {
            "baseline_failed_company_count": failed_company_count,
            "baseline_category_company_counts": dict(failure_category_counts.most_common()),
            "repeat_failed_company_count": repeat_failed_company_count,
            "repeat_category_company_counts": dict(repeat_failure_category_counts.most_common()),
        },
        "llm_usage": {
            "baseline": usage_summary(baseline_all),
            "independent_repeat": usage_summary({repeat_date: repeat_all}),
        },
        "llm_call_audit": {
            "baseline": _call_audit(root / "runs" / "baseline"),
            "independent_repeat": _call_audit(root / "runs" / "repeat"),
        },
        "limitations": [
            "The old holistic score is a bridge reference, not ground truth.",
            "Eight names and three realized return periods are a directional diagnostic, not a backtest.",
            "The sample excludes every universe ticker textually mentioned in the supplied notebook.",
        ],
    }
    diagnostics = root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    _write_csv(diagnostics / "company-panel.csv", company_rows, company_rows[0].keys())
    _write_csv(diagnostics / "date-metrics.csv", date_metrics, date_metrics[0].keys())
    (diagnostics / "shadow-holdout-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (diagnostics / "shadow-holdout-report.md").write_text(
        _render_markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["production_ready"] else 2


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or evaluate the frozen small-sample MOAT shadow holdout.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--workspace", required=True, type=Path)
    prepare.add_argument("--universe", required=True, type=Path)
    prepare.add_argument("--old-scores", required=True, type=Path)
    prepare.add_argument("--notebook", required=True, type=Path)
    prepare.add_argument("--source-root", default=Path("src"), type=Path)
    prepare.add_argument("--output", required=True, type=Path)
    prepare.add_argument("--seed", type=int, default=20260816)
    prepare.set_defaults(handler=_prepare)
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--legacy-signals", required=True, type=Path)
    analyze.set_defaults(handler=_analyze)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
