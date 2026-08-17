from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from moatrader.experiments import (
    FrozenExpectationGapContract,
    compute_contract_sha256,
    verify_frozen_sources,
)


SEOUL = ZoneInfo("Asia/Seoul")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    output = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2.0
        for offset in range(position, end):
            output[order[offset]] = rank
        position = end
    return output


def spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = ranks(left)
    y = ranks(right)
    mx = statistics.mean(x)
    my = statistics.mean(y)
    numerator = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def group_demean(values: list[float], groups: list[str]) -> list[float]:
    buckets: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        buckets[group].append(value)
    means = {group: statistics.mean(items) for group, items in buckets.items()}
    return [value - means[group] for value, group in zip(values, groups, strict=True)]


def weighted_mean(values: list[float], weights: list[float]) -> float | None:
    total = sum(weights)
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total if total else None


def metrics(rows: list[dict[str, Any]], candidate: str) -> dict[str, Any]:
    eligible_field = f"candidate_{candidate.lower()}_eligible"
    dated: list[dict[str, Any]] = []
    for signal_date in sorted({row["signal_date"] for row in rows}):
        group = [
            row
            for row in rows
            if row["signal_date"] == signal_date
            and row[eligible_field]
            and number(row.get("cheap_rank")) is not None
            and number(row.get("forward_return")) is not None
        ]
        scores = [float(row["cheap_rank"]) for row in group]
        returns = [float(row["forward_return"]) for row in group]
        sectors = [str(row["sector"]) for row in group]
        raw_ic = spearman(scores, returns) if len(group) >= 20 else None
        neutral_ic = (
            spearman(group_demean(scores, sectors), group_demean(returns, sectors))
            if len(group) >= 20
            else None
        )
        count = max(1, len(group) // 5) if group else 0
        order = sorted(range(len(group)), key=lambda index: (scores[index], group[index]["ticker"]))
        bottom = order[:count]
        top = order[-count:]
        multipliers = [
            float(row["candidate_c_position_multiplier"]) if candidate == "C" else 1.0
            for row in group
        ]
        top_return = weighted_mean(
            [returns[index] for index in top], [multipliers[index] for index in top]
        )
        bottom_return = weighted_mean(
            [returns[index] for index in bottom], [multipliers[index] for index in bottom]
        )
        top_returns = [returns[index] for index in top]
        worst_decile = None
        if top_returns:
            worst_index = max(0, math.ceil(0.10 * len(top_returns)) - 1)
            worst_decile = sorted(top_returns)[worst_index]
        negative_universe = [value for value in returns if value < 0]
        negative_top_indices = [index for index in top if returns[index] < 0]
        negative_top_mean = weighted_mean(
            [returns[index] for index in negative_top_indices],
            [multipliers[index] for index in negative_top_indices],
        )
        downside_capture = (
            abs(negative_top_mean) / abs(statistics.mean(negative_universe))
            if negative_universe and negative_top_mean is not None
            else 0.0
            if negative_universe
            else None
        )
        dated.append(
            {
                "signal_date": signal_date,
                "eligible_count": len(group),
                "top_count": len(top),
                "raw_ic": raw_ic,
                "sector_neutral_ic": neutral_ic,
                "q5_minus_q1": (
                    top_return - bottom_return
                    if top_return is not None and bottom_return is not None
                    else None
                ),
                "top_portfolio_return": top_return,
                "top_portfolio_worst_decile": worst_decile,
                "downside_capture": downside_capture,
            }
        )

    def mean(field: str) -> float | None:
        values = [float(item[field]) for item in dated if item[field] is not None]
        return statistics.mean(values) if values else None

    wealth = 1.0
    peak = 1.0
    drawdown = 0.0
    for item in dated:
        period_return = item["top_portfolio_return"]
        if period_return is None:
            continue
        wealth *= 1.0 + float(period_return)
        peak = max(peak, wealth)
        drawdown = min(drawdown, wealth / peak - 1.0)
    return {
        "candidate": candidate,
        "date_metrics": dated,
        "mean_raw_ic": mean("raw_ic"),
        "mean_sector_neutral_ic": mean("sector_neutral_ic"),
        "mean_q5_minus_q1": mean("q5_minus_q1"),
        "mean_top_portfolio_worst_decile": mean("top_portfolio_worst_decile"),
        "mean_downside_capture": mean("downside_capture"),
        "top_portfolio_compound_return": wealth - 1.0,
        "top_portfolio_maximum_drawdown": drawdown,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate sealed Cheap vs Cheap+Risk holdouts after embargo.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--sealed-root", type=Path, required=True)
    parser.add_argument("--returns", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"holdout evaluation output already exists: {output}")
    raw_contract = json.loads(args.contract.resolve().read_text(encoding="utf-8-sig"))
    if compute_contract_sha256(raw_contract) != raw_contract.get("contract_sha256"):
        raise ValueError("frozen contract content hash is invalid")
    contract = FrozenExpectationGapContract.model_validate(raw_contract)
    verify_frozen_sources(contract, repository_root=args.repository_root.resolve())
    embargo = max(contract.holdout_dates) + timedelta(days=contract.forward_return_calendar_days)
    if datetime.now(tz=SEOUL).date() < embargo:
        raise ValueError(f"holdout returns remain embargoed until {embargo}")
    merged: list[dict[str, Any]] = []
    signals_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    candidates_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for signal_date in contract.holdout_dates:
        folder = args.sealed_root.resolve() / signal_date.isoformat()
        seal_path = folder / "seal.json"
        signal_path = folder / "sealed-signals.json"
        candidate_path = folder / "candidates.csv"
        seal = json.loads(seal_path.read_text(encoding="utf-8-sig"))
        if seal.get("signal_date") != signal_date.isoformat():
            raise ValueError(f"seal date mismatch for {signal_date}")
        if seal.get("return_data_accessed") is not False:
            raise ValueError(f"seal accessed return data for {signal_date}")
        if seal.get("contract_sha256") != contract.contract_sha256:
            raise ValueError(f"seal contract mismatch for {signal_date}")
        if sha256(signal_path) != seal.get("sealed_signals_sha256") or sha256(candidate_path) != seal.get("candidates_sha256"):
            raise ValueError(f"sealed signal content changed for {signal_date}")
        for row in json.loads(signal_path.read_text(encoding="utf-8-sig")):
            key = (str(row["signal_date"]), str(row["ticker"]))
            if key in signals_by_key:
                raise ValueError(f"duplicate sealed signal key: {key}")
            signals_by_key[key] = row
        for row in read_csv(candidate_path):
            key = (str(row["signal_date"]), str(row["ticker"]))
            if key in candidates_by_key:
                raise ValueError(f"duplicate candidate key: {key}")
            row["cheap_rank"] = number(row.get("cheap_rank"))
            for field in ("candidate_a_eligible", "candidate_b_eligible", "candidate_c_eligible"):
                row[field] = str(row[field]).lower() == "true"
            row["candidate_c_position_multiplier"] = float(row["candidate_c_position_multiplier"])
            candidates_by_key[key] = row
    expected_count = contract.universe_count * len(contract.holdout_dates)
    if len(signals_by_key) != expected_count or len(candidates_by_key) != expected_count:
        raise ValueError("sealed signal panel does not match frozen holdout shape")
    returns = read_csv(args.returns.resolve())
    if len(returns) != expected_count:
        raise ValueError("returns panel does not match frozen holdout shape")
    seen_return_keys: set[tuple[str, str]] = set()
    for row in returns:
        key = (str(row.get("signal_date") or row.get("date")), str(row.get("ticker") or "").zfill(6))
        if key in seen_return_keys:
            raise ValueError(f"duplicate return key: {key}")
        seen_return_keys.add(key)
        if key not in signals_by_key or key not in candidates_by_key:
            raise ValueError(f"return key was not sealed: {key}")
        value = number(row.get("forward_return"))
        if value is None:
            raise ValueError(f"missing forward return for {key}")
        signal = signals_by_key[key]
        candidate = candidates_by_key[key]
        merged.append(
            {
                **candidate,
                "sector": signal["sector"],
                "forward_return": value,
            }
        )
    results = {name: metrics(merged, name) for name in ("A", "B", "C")}
    a = results["A"]
    c = results["C"]
    ic_ok = (
        a["mean_sector_neutral_ic"] is not None
        and c["mean_sector_neutral_ic"] is not None
        and c["mean_sector_neutral_ic"]
        >= a["mean_sector_neutral_ic"] - contract.maximum_sector_neutral_ic_sacrifice
    )
    worst_ok = (
        a["mean_top_portfolio_worst_decile"] is not None
        and c["mean_top_portfolio_worst_decile"] is not None
        and c["mean_top_portfolio_worst_decile"]
        >= a["mean_top_portfolio_worst_decile"] + contract.minimum_worst_decile_improvement
    )
    capture_ok = (
        a["mean_downside_capture"] is not None
        and c["mean_downside_capture"] is not None
        and c["mean_downside_capture"]
        <= a["mean_downside_capture"] - contract.minimum_downside_capture_improvement
    )
    report = {
        "schema_version": "frozen-expectation-gap-holdout-evaluation/1",
        "contract_sha256": contract.contract_sha256,
        "returns_sha256": sha256(args.returns.resolve()),
        "success_criteria": {
            "maximum_sector_neutral_ic_sacrifice": contract.maximum_sector_neutral_ic_sacrifice,
            "minimum_worst_decile_improvement": contract.minimum_worst_decile_improvement,
            "minimum_downside_capture_improvement": contract.minimum_downside_capture_improvement,
        },
        "candidates": results,
        "risk_overlay_success": bool(ic_ok and (worst_ok or capture_ok)),
        "criteria_checks": {
            "ic_preserved": ic_ok,
            "worst_decile_improved": worst_ok,
            "downside_capture_improved": capture_ok,
        },
    }
    output.mkdir(parents=True)
    (output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Frozen Expectation GAP holdout",
        "",
        f"Contract: `{contract.contract_sha256}`",
        "",
        "| Candidate | Raw IC | Sector-neutral IC | Q5-Q1 | Worst decile | Downside capture | Compound top return |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("A", "B", "C"):
        item = results[name]
        fmt = lambda value: "NA" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {name} | {fmt(item['mean_raw_ic'])} | {fmt(item['mean_sector_neutral_ic'])} | "
            f"{fmt(item['mean_q5_minus_q1'])} | {fmt(item['mean_top_portfolio_worst_decile'])} | "
            f"{fmt(item['mean_downside_capture'])} | {fmt(item['top_portfolio_compound_return'])} |"
        )
    lines.extend(["", f"Risk overlay success: **{report['risk_overlay_success']}**", ""])
    (output / "evaluation.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "evaluation.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
