from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.evaluate_signal_panel import (
        _randomized_quantile_spreads,
        _sample_percentile,
        signal_tie_diagnostics,
        winsorize,
    )
    from scripts.materialize_moat_rank_shadow import (
        CANDIDATE_STABLE,
        _average_ranks,
        _rank_keys,
        materialize_root,
    )
    from scripts.merge_kr_signal_panel import spearman
except ModuleNotFoundError:  # Direct ``python scripts\...py`` execution.
    from evaluate_signal_panel import (
        _randomized_quantile_spreads,
        _sample_percentile,
        signal_tie_diagnostics,
        winsorize,
    )
    from materialize_moat_rank_shadow import (
        CANDIDATE_STABLE,
        _average_ranks,
        _rank_keys,
        materialize_root,
    )
    from merge_kr_signal_panel import spearman


SCHEMA_VERSION = "moatrader-multi-period-economic-evaluation/1"
OLD = "OLD_HOLISTIC"
PUBLIC = "CURRENT_PUBLIC"
STABLE = "CURRENT_STABLE_RANK_KEY"
SIGNALS = (OLD, PUBLIC, STABLE)


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
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _neutral_ic(scores: list[float], returns: list[float], groups: list[str]) -> dict[str, Any]:
    if not (len(scores) == len(returns) == len(groups)):
        raise ValueError("neutral IC inputs must have equal lengths")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        if not group:
            raise ValueError("neutral group is blank")
        grouped[group].append(index)
    usable = [index for indices in grouped.values() if len(indices) >= 2 for index in indices]
    if len(usable) < 3:
        return {
            "rank_ic": None,
            "observation_count": len(usable),
            "group_count": sum(len(indices) >= 2 for indices in grouped.values()),
            "singleton_excluded_count": len(scores) - len(usable),
        }
    score_means = {
        group: statistics.mean(scores[index] for index in indices)
        for group, indices in grouped.items()
        if len(indices) >= 2
    }
    return_means = {
        group: statistics.mean(returns[index] for index in indices)
        for group, indices in grouped.items()
        if len(indices) >= 2
    }
    neutral_scores = [scores[index] - score_means[groups[index]] for index in usable]
    neutral_returns = [returns[index] - return_means[groups[index]] for index in usable]
    return {
        "rank_ic": spearman(neutral_scores, neutral_returns),
        "observation_count": len(usable),
        "group_count": len(score_means),
        "singleton_excluded_count": len(scores) - len(usable),
    }


def _rank_percentiles(values: list[float]) -> list[float]:
    ranks = _average_ranks([(value,) for value in values])
    return [rank / len(values) for rank in ranks]


def _date_metrics(
    rows: list[dict[str, Any]],
    *,
    signal: str,
    tie_simulations: int,
    seed: str,
) -> tuple[dict[str, Any], list[float]]:
    scores = [float(row[signal]) for row in rows]
    returns = [float(row["forward_return"]) for row in rows]
    clipped = winsorize(returns, 0.01, 0.99)
    tickers = [str(row["ticker"]) for row in rows]
    sectors = [str(row["sector"]) for row in rows]
    market_size = [f"{row['market']}|{row['size_bucket']}" for row in rows]
    tail_count = len(rows) // 5
    spreads = _randomized_quantile_spreads(
        scores,
        clipped,
        tickers,
        tail_count=tail_count,
        simulations=tie_simulations,
        seed=seed,
    )
    return (
        {
            "observation_count": len(rows),
            "rank_ic": spearman(scores, returns),
            "winsorized_rank_ic": spearman(scores, clipped),
            "sector_neutral": _neutral_ic(scores, clipped, sectors),
            "market_size_neutral": _neutral_ic(scores, clipped, market_size),
            "equal_count_q5_minus_q1": {
                "mean": statistics.mean(spreads),
                "median": statistics.median(spreads),
                "p05_tie_assignment": _sample_percentile(spreads, 0.05),
                "p95_tie_assignment": _sample_percentile(spreads, 0.95),
                "positive_share": sum(value > 0 for value in spreads) / len(spreads),
                "top_count": tail_count,
                "bottom_count": tail_count,
            },
            "tie_diagnostics": signal_tie_diagnostics(scores),
        },
        spreads,
    )


def _summary(
    panel: list[dict[str, Any]],
    *,
    dates: list[str],
    tie_simulations: int,
    seed: str,
) -> dict[str, Any]:
    metrics_by_signal: dict[str, Any] = {}
    spreads_by_signal: dict[str, list[list[float]]] = defaultdict(list)
    for signal in SIGNALS:
        date_metrics: list[dict[str, Any]] = []
        pooled_signal_ranks: list[float] = []
        pooled_return_ranks: list[float] = []
        for date in dates:
            rows = [row for row in panel if row["date"] == date]
            metrics, spreads = _date_metrics(
                rows,
                signal=signal,
                tie_simulations=tie_simulations,
                seed=f"{seed}|{date}",
            )
            date_metrics.append({"date": date, **metrics})
            spreads_by_signal[signal].append(spreads)
            pooled_signal_ranks.extend(_rank_percentiles([float(row[signal]) for row in rows]))
            pooled_return_ranks.extend(
                _rank_percentiles([float(row["forward_return"]) for row in rows])
            )
        aggregate_spreads = [
            statistics.mean(period[simulation] for period in spreads_by_signal[signal])
            for simulation in range(tie_simulations)
        ]
        rank_ics = [float(row["rank_ic"]) for row in date_metrics if row["rank_ic"] is not None]
        neutral_ics = [
            float(row["sector_neutral"]["rank_ic"])
            for row in date_metrics
            if row["sector_neutral"]["rank_ic"] is not None
        ]
        market_size_ics = [
            float(row["market_size_neutral"]["rank_ic"])
            for row in date_metrics
            if row["market_size_neutral"]["rank_ic"] is not None
        ]
        metrics_by_signal[signal] = {
            "date_count": len(date_metrics),
            "mean_date_rank_ic": statistics.mean(rank_ics) if rank_ics else None,
            "median_date_rank_ic": statistics.median(rank_ics) if rank_ics else None,
            "positive_rank_ic_dates": sum(value > 0 for value in rank_ics),
            "mean_date_sector_neutral_rank_ic": (
                statistics.mean(neutral_ics) if neutral_ics else None
            ),
            "positive_sector_neutral_dates": sum(value > 0 for value in neutral_ics),
            "mean_date_market_size_neutral_rank_ic": (
                statistics.mean(market_size_ics) if market_size_ics else None
            ),
            "pooled_within_date_rank_ic": spearman(
                pooled_signal_ranks, pooled_return_ranks
            ),
            "mean_date_equal_count_q5_minus_q1": {
                "mean": statistics.mean(aggregate_spreads),
                "median": statistics.median(aggregate_spreads),
                "p05_tie_assignment": _sample_percentile(aggregate_spreads, 0.05),
                "p95_tie_assignment": _sample_percentile(aggregate_spreads, 0.95),
                "positive_share": (
                    sum(value > 0 for value in aggregate_spreads)
                    / len(aggregate_spreads)
                ),
            },
            "dates": date_metrics,
        }

    comparisons: dict[str, Any] = {}
    for left, right, label in (
        (OLD, STABLE, "OLD_MINUS_CURRENT_STABLE"),
        (OLD, PUBLIC, "OLD_MINUS_CURRENT_PUBLIC"),
        (STABLE, PUBLIC, "CURRENT_STABLE_MINUS_PUBLIC"),
    ):
        left_metrics = metrics_by_signal[left]
        right_metrics = metrics_by_signal[right]
        spread_deltas = [
            statistics.mean(left_period[simulation] - right_period[simulation] for left_period, right_period in zip(spreads_by_signal[left], spreads_by_signal[right], strict=True))
            for simulation in range(tie_simulations)
        ]
        comparisons[label] = {
            "mean_date_rank_ic_delta": (
                left_metrics["mean_date_rank_ic"] - right_metrics["mean_date_rank_ic"]
            ),
            "mean_date_sector_neutral_rank_ic_delta": (
                left_metrics["mean_date_sector_neutral_rank_ic"]
                - right_metrics["mean_date_sector_neutral_rank_ic"]
            ),
            "mean_q5_minus_q1_delta": statistics.mean(spread_deltas),
            "q5_minus_q1_delta_p05_tie_assignment": _sample_percentile(spread_deltas, 0.05),
            "q5_minus_q1_delta_p95_tie_assignment": _sample_percentile(spread_deltas, 0.95),
            "q5_minus_q1_delta_positive_share": (
                sum(value > 0 for value in spread_deltas) / len(spread_deltas)
            ),
        }
    return {"signals": metrics_by_signal, "comparisons": comparisons}


def _date_block_bootstrap(
    panel: list[dict[str, Any]],
    *,
    dates: list[str],
    samples: int,
    seed: str,
) -> dict[str, Any]:
    by_date = {date: [row for row in panel if row["date"] == date] for date in dates}
    per_date_ic = {
        signal: {
            date: spearman(
                [float(row[signal]) for row in rows],
                winsorize([float(row["forward_return"]) for row in rows], 0.01, 0.99),
            )
            for date, rows in by_date.items()
        }
        for signal in SIGNALS
    }
    rng = random.Random(f"{seed}|date-block-bootstrap")
    values: dict[str, list[float]] = {signal: [] for signal in SIGNALS}
    deltas: dict[str, list[float]] = {
        "OLD_MINUS_CURRENT_STABLE": [],
        "OLD_MINUS_CURRENT_PUBLIC": [],
        "CURRENT_STABLE_MINUS_PUBLIC": [],
    }
    for _ in range(samples):
        sampled_dates = [dates[rng.randrange(len(dates))] for _ in dates]
        means = {
            signal: statistics.mean(float(per_date_ic[signal][date]) for date in sampled_dates)
            for signal in SIGNALS
        }
        for signal, value in means.items():
            values[signal].append(value)
        deltas["OLD_MINUS_CURRENT_STABLE"].append(means[OLD] - means[STABLE])
        deltas["OLD_MINUS_CURRENT_PUBLIC"].append(means[OLD] - means[PUBLIC])
        deltas["CURRENT_STABLE_MINUS_PUBLIC"].append(means[STABLE] - means[PUBLIC])

    def interval(items: list[float]) -> dict[str, Any]:
        return {
            "lower_95": _sample_percentile(items, 0.025),
            "upper_95": _sample_percentile(items, 0.975),
            "positive_share": sum(value > 0 for value in items) / len(items),
        }

    return {
        "samples": samples,
        "resampling_unit": "DATE",
        "signals": {signal: interval(items) for signal, items in values.items()},
        "comparisons": {label: interval(items) for label, items in deltas.items()},
    }


def _llm_audit(root: Path) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []
    for path in root.rglob("llm-calls.jsonl"):
        calls.extend(
            json.loads(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    live = [call for call in calls if not call.get("replayed")]
    return {
        "call_count": len(calls),
        "live_call_count": len(live),
        "replayed_call_count": len(calls) - len(live),
        "models": dict(Counter(str(call.get("model")) for call in calls)),
        "tasks": dict(Counter(str(call.get("task")) for call in calls)),
        "live_input_tokens": sum(
            int((call.get("usage") or {}).get("input_tokens") or 0) for call in live
        ),
        "live_output_tokens": sum(
            int((call.get("usage") or {}).get("output_tokens") or 0) for call in live
        ),
    }


def _source_pit_audit(root: Path, primary_dates: list[str]) -> dict[str, Any]:
    future_source: list[dict[str, str]] = []
    future_price: list[dict[str, str]] = []
    row_count = 0
    for date in primary_dates:
        cutoff = datetime.fromisoformat(f"{date}T23:59:59+09:00")
        for manifest in (root / "manifests" / date).glob("batch-*.csv"):
            for row in _read_csv(manifest):
                row_count += 1
                metadata_path = Path(row["metadata"])
                metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
                available_at = datetime.fromisoformat(str(metadata["available_at"]).replace("Z", "+00:00"))
                if available_at > cutoff:
                    future_source.append(
                        {"date": date, "ticker": row["ticker"], "available_at": available_at.isoformat()}
                    )
                price_as_of = datetime.fromisoformat(str(row["price_as_of"]).replace("Z", "+00:00"))
                if price_as_of > cutoff:
                    future_price.append(
                        {"date": date, "ticker": row["ticker"], "price_as_of": price_as_of.isoformat()}
                    )
    return {
        "manifest_document_row_count": row_count,
        "future_source_count": len(future_source),
        "future_price_count": len(future_price),
        "future_sources": future_source,
        "future_prices": future_price,
        "passed": not future_source and not future_price,
    }


def build_panel(
    *, root: Path, anchor_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = json.loads((root / "FROZEN_protocol.json").read_text(encoding="utf-8-sig"))
    base_rows = _read_csv(root / "FROZEN_panel.csv")
    anchor_rows = materialize_root(anchor_root)
    primary_rows = materialize_root(root / "runs")
    current_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for row in [*anchor_rows, *primary_rows]:
        key = (str(row["date"]), str(row["ticker"]).zfill(6))
        if key in current_by_key:
            raise ValueError(f"duplicate current score row: {key}")
        current_by_key[key] = row

    expected = {(row["date"], row["ticker"].zfill(6)) for row in base_rows}
    missing = sorted(expected - set(current_by_key))
    extra = sorted(set(current_by_key) - expected)
    if missing or extra:
        raise ValueError(f"current score panel mismatch: missing={missing}, extra={extra}")

    eligible: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    by_date_current: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in base_rows:
        current = current_by_key[(row["date"], row["ticker"].zfill(6))]
        if _truthy(row["return_eligible"]) and bool(current["score_eligible"]):
            by_date_current[row["date"]].append(current)
        else:
            exclusions.append(
                {
                    "date": row["date"],
                    "ticker": row["ticker"].zfill(6),
                    "return_eligible": _truthy(row["return_eligible"]),
                    "score_eligible": bool(current["score_eligible"]),
                    "eligibility_status": current["eligibility_status"],
                }
            )

    stable_rank_by_key: dict[tuple[str, str], float] = {}
    for date, rows in by_date_current.items():
        keys = _rank_keys(rows, CANDIDATE_STABLE)
        ranks = _average_ranks(keys)
        for row, rank in zip(rows, ranks, strict=True):
            stable_rank_by_key[(date, str(row["ticker"]).zfill(6))] = rank

    for row in base_rows:
        key = (row["date"], row["ticker"].zfill(6))
        current = current_by_key[key]
        if key not in stable_rank_by_key:
            continue
        eligible.append(
            {
                "date": row["date"],
                "split": row["split"],
                "ticker": row["ticker"].zfill(6),
                "company_name": row["company_name"],
                "market": row["market"],
                "size_bucket": row["size_bucket"],
                "industry_code": row["industry_code"],
                "sector": row["sector"],
                OLD: float(row["old_holistic_score"]),
                PUBLIC: float(current["economic_moat_score"]),
                STABLE: stable_rank_by_key[key],
                "rank_refinement_status": current["rank_refinement_status"],
                "audit_status": current["audit_status"],
                "eligibility_status": current["eligibility_status"],
                "forward_return": float(row["forward_return"]),
                "signal_session": row["signal_session"],
                "return_session": row["return_session"],
            }
        )
    counts = Counter(row["date"] for row in eligible)
    minimum = int(protocol["evaluation"]["minimum_eligible_per_date"])
    audit = {
        "expected_stock_date_count": len(expected),
        "materialized_stock_date_count": len(current_by_key),
        "eligible_stock_date_count": len(eligible),
        "excluded_stock_date_count": len(exclusions),
        "eligible_by_date": {date: counts[date] for date in protocol["dates"]},
        "minimum_eligible_per_date": minimum,
        "minimum_eligible_gate_passed": all(
            counts[date] >= minimum for date in protocol["dates"]
        ),
        "exclusions": exclusions,
    }
    return eligible, audit


def _fmt(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value * 100:.2f}%" if percent else f"{value:.4f}"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    primary = report["primary_holdout"]
    all_dates = report["all_dates_with_seen_anchor"]
    panel = report["panel_audit"]
    lines = [
        "# MOAT multi-period economic holdout",
        "",
        f"- Frozen stock×date cells: **{panel['expected_stock_date_count']}**",
        f"- Score/return eligible cells: **{panel['eligible_stock_date_count']}**",
        f"- Eligible by date: `{json.dumps(panel['eligible_by_date'], ensure_ascii=False)}`",
        f"- Minimum 20/date gate: **{'PASS' if panel['minimum_eligible_gate_passed'] else 'FAIL'}**",
        "- Return horizon: **50 market sessions, close-to-close**",
        "",
        "## Primary temporal holdout (three unseen dates)",
        "",
        "| Signal | Mean date IC | Mean sector-neutral IC | Mean market-size-neutral IC | Pooled within-date IC | Mean Q5−Q1 | Positive IC dates |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for signal in SIGNALS:
        value = primary["signals"][signal]
        lines.append(
            f"| {signal} | {_fmt(value['mean_date_rank_ic'])} | "
            f"{_fmt(value['mean_date_sector_neutral_rank_ic'])} | "
            f"{_fmt(value['mean_date_market_size_neutral_rank_ic'])} | "
            f"{_fmt(value['pooled_within_date_rank_ic'])} | "
            f"{_fmt(value['mean_date_equal_count_q5_minus_q1']['mean'], percent=True)} | "
            f"{value['positive_rank_ic_dates']}/{value['date_count']} |"
        )
    lines.extend(
        [
            "",
            "## Primary holdout by date",
            "",
            "| Date | Signal | N | IC | Sector-neutral IC | Market-size-neutral IC | Q5−Q1 | Distinct scores |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for date in report["design_audit"]["primary_holdout_dates"]:
        for signal in SIGNALS:
            value = next(
                row for row in primary["signals"][signal]["dates"] if row["date"] == date
            )
            lines.append(
                f"| {date} | {signal} | {value['observation_count']} | "
                f"{_fmt(value['rank_ic'])} | "
                f"{_fmt(value['sector_neutral']['rank_ic'])} | "
                f"{_fmt(value['market_size_neutral']['rank_ic'])} | "
                f"{_fmt(value['equal_count_q5_minus_q1']['mean'], percent=True)} | "
                f"{value['tie_diagnostics']['distinct_signal_count']} |"
            )
    lines.extend(
        [
            "",
            "## All four dates including the seen anchor",
            "",
            "| Signal | Mean date IC | Mean sector-neutral IC | Mean Q5−Q1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for signal in SIGNALS:
        value = all_dates["signals"][signal]
        lines.append(
            f"| {signal} | {_fmt(value['mean_date_rank_ic'])} | "
            f"{_fmt(value['mean_date_sector_neutral_rank_ic'])} | "
            f"{_fmt(value['mean_date_equal_count_q5_minus_q1']['mean'], percent=True)} |"
        )
    lines.extend(
        [
            "",
            "## Direct OLD vs CURRENT comparisons",
            "",
            "| Comparison | Mean IC delta | Mean sector-neutral IC delta | Mean Q5−Q1 delta | Date-block bootstrap 95% IC-delta interval |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, comparison_name in (
        ("OLD−CURRENT PUBLIC", "OLD_MINUS_CURRENT_PUBLIC"),
        ("OLD−CURRENT STABLE", "OLD_MINUS_CURRENT_STABLE"),
    ):
        comparison = primary["comparisons"][comparison_name]
        bootstrap = report["primary_date_block_bootstrap"]["comparisons"][
            comparison_name
        ]
        lines.append(
            f"| {label} | {_fmt(comparison['mean_date_rank_ic_delta'])} | "
            f"{_fmt(comparison['mean_date_sector_neutral_rank_ic_delta'])} | "
            f"{_fmt(comparison['mean_q5_minus_q1_delta'], percent=True)} | "
            f"[{_fmt(bootstrap['lower_95'])}, {_fmt(bootstrap['upper_95'])}] |"
        )
    design = report["design_audit"]
    source_pit = design["source_pit"]
    lines.extend(
        [
            "",
            "## Design and provenance audit",
            "",
            f"- Protocol was frozen before primary runs: **{'PASS' if design['freeze_precedes_primary_runs'] else 'FAIL'}**",
            f"- Candidate selection used forward returns: **{str(design['candidate_selection_used_forward_returns']).upper()}**",
            f"- Future-dated source rows: **{source_pit['future_source_count']}**",
            f"- Future-dated price rows: **{source_pit['future_price_count']}**",
            f"- Evaluation API calls: **{report['evaluation_api_calls']}**",
            f"- Runner versions: `{json.dumps(design['runner_versions'])}`",
            f"- LLM models observed in primary run artifacts: `{json.dumps(report['llm_audit']['models'], sort_keys=True)}`",
            "",
            "The first date is reported only as a seen anchor. Production interpretation must use "
            "the three-date primary temporal holdout. Tie Monte Carlo intervals measure cutoff-tie "
            "assignment uncertainty, while the date-block bootstrap measures the much coarser "
            "three-period sampling uncertainty.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen OLD, CURRENT public, and stable RankKey signals over four dates."
    )
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--anchor-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tie-simulations", type=int, default=10_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    args = parser.parse_args()
    root = args.root.resolve()
    anchor_root = args.anchor_root.resolve()
    protocol_path = root / "FROZEN_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    panel, panel_audit = build_panel(root=root, anchor_root=anchor_root)
    seed = str(protocol["evaluation"]["seed"])
    primary_dates = list(protocol["primary_holdout_dates"])
    dates = list(protocol["dates"])
    primary_panel = [row for row in panel if row["date"] in primary_dates]
    primary = _summary(
        primary_panel,
        dates=primary_dates,
        tie_simulations=args.tie_simulations,
        seed=seed,
    )
    all_dates = _summary(
        panel,
        dates=dates,
        tie_simulations=args.tie_simulations,
        seed=seed,
    )
    bootstrap = _date_block_bootstrap(
        primary_panel,
        dates=primary_dates,
        samples=args.bootstrap_samples,
        seed=seed,
    )
    pit_audit = _source_pit_audit(root, primary_dates)
    llm_audit = _llm_audit(root / "runs")
    started_at = []
    runner_versions: set[str] = set()
    for path in (root / "runs").rglob("run-result.json"):
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("started_at"):
            started_at.append(str(payload["started_at"]))
        for company in payload.get("companies", []):
            if company.get("runner_version"):
                runner_versions.add(str(company["runner_version"]))
    frozen_at = datetime.fromisoformat(str(protocol["frozen_at"]).replace("Z", "+00:00"))
    parsed_starts = [
        datetime.fromisoformat(value.replace("Z", "+00:00")) for value in started_at
    ]
    freeze_precedes_primary_runs = bool(parsed_starts) and frozen_at < min(parsed_starts)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": _sha256(protocol_path),
        "evaluation_api_calls": 0,
        "panel_audit": panel_audit,
        "design_audit": {
            "candidate_selection_used_forward_returns": protocol[
                "candidate_selection_used_forward_returns"
            ],
            "freeze_precedes_primary_runs": freeze_precedes_primary_runs,
            "seen_anchor_dates": protocol["seen_anchor_dates"],
            "primary_holdout_dates": primary_dates,
            "source_pit": pit_audit,
            "runner_versions": sorted(runner_versions),
            "passed": (
                not protocol["candidate_selection_used_forward_returns"]
                and freeze_precedes_primary_runs
                and pit_audit["passed"]
            ),
        },
        "llm_audit": llm_audit,
        "primary_holdout": primary,
        "all_dates_with_seen_anchor": all_dates,
        "primary_date_block_bootstrap": bootstrap,
        "interpretation_constraints": {
            "primary_period_count": len(primary_dates),
            "seen_anchor_excluded_from_primary": True,
            "sector_source": "OpenDART current company industry code mapped to frozen broad sectors; singleton sectors excluded from neutral IC.",
            "tie_interval": "cutoff assignment uncertainty, not investment performance confidence",
            "date_bootstrap": "only three primary periods; interval is descriptive and low-power",
        },
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output / "multi-period-economic-panel.csv",
        panel,
        (
            "date",
            "split",
            "ticker",
            "company_name",
            "market",
            "size_bucket",
            "industry_code",
            "sector",
            OLD,
            PUBLIC,
            STABLE,
            "rank_refinement_status",
            "audit_status",
            "eligibility_status",
            "forward_return",
            "signal_session",
            "return_session",
        ),
    )
    (output / "multi-period-economic-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "multi-period-economic-report.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    gates_passed = panel_audit["minimum_eligible_gate_passed"] and report["design_audit"]["passed"]
    return 0 if gates_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
