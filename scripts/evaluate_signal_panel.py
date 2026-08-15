from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from scripts.merge_kr_signal_panel import spearman


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _number(value: str, *, field: str, key: tuple[str, str]) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} for {key[0]}/{key[1]}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field} for {key[0]}/{key[1]}")
    return result


def winsorize(values: list[float], lower: float = 0.01, upper: float = 0.99) -> list[float]:
    if not values:
        return []
    ordered = sorted(values)

    def percentile(probability: float) -> float:
        position = probability * (len(ordered) - 1)
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return ordered[low]
        fraction = position - low
        return ordered[low] * (1 - fraction) + ordered[high] * fraction

    floor = percentile(lower)
    ceiling = percentile(upper)
    return [min(ceiling, max(floor, value)) for value in values]


def group_demean(values: list[float], groups: list[str]) -> list[float]:
    if len(values) != len(groups):
        raise ValueError("values and groups must have the same length")
    by_group: dict[str, list[float]] = defaultdict(list)
    for value, group in zip(values, groups, strict=True):
        if not group:
            raise ValueError("sector/group is blank; neutral IC would be misleading")
        by_group[group].append(value)
    means = {group: statistics.mean(items) for group, items in by_group.items()}
    return [value - means[group] for value, group in zip(values, groups, strict=True)]


def residualize(values: list[float], controls: list[list[float]]) -> list[float]:
    """Return intercept-adjusted OLS residuals via Gram-Schmidt projections."""
    if not values:
        return []
    if any(len(control) != len(values) for control in controls):
        raise ValueError("every factor control must match the value count")
    residual = [value - statistics.mean(values) for value in values]
    basis: list[list[float]] = []
    for control in controls:
        vector = [value - statistics.mean(control) for value in control]
        for prior in basis:
            denominator = sum(value * value for value in prior)
            if denominator:
                coefficient = sum(a * b for a, b in zip(vector, prior, strict=True)) / denominator
                vector = [a - coefficient * b for a, b in zip(vector, prior, strict=True)]
        denominator = sum(value * value for value in vector)
        if denominator <= 1e-20:
            continue
        basis.append(vector)
        coefficient = sum(a * b for a, b in zip(residual, vector, strict=True)) / denominator
        residual = [a - coefficient * b for a, b in zip(residual, vector, strict=True)]
    return residual


def nonoverlapping_quantile_spread(
    signals: list[float],
    returns: list[float],
    tickers: list[str],
    *,
    quantiles: int = 5,
) -> tuple[float | None, int, int]:
    if not (len(signals) == len(returns) == len(tickers)):
        raise ValueError("signals, returns, and tickers must have the same length")
    if len(signals) < quantiles:
        return None, 0, 0
    ordered = sorted(range(len(signals)), key=lambda index: (signals[index], tickers[index]))
    buckets: list[list[int]] = [[] for _ in range(quantiles)]
    for position, index in enumerate(ordered):
        bucket = min(quantiles - 1, position * quantiles // len(ordered))
        buckets[bucket].append(index)
    low = buckets[0]
    high = buckets[-1]
    if set(low) & set(high):
        raise RuntimeError("quantile construction produced overlapping tails")
    spread = statistics.mean(returns[index] for index in high) - statistics.mean(
        returns[index] for index in low
    )
    return spread, len(high), len(low)


def evaluate_date(
    rows: list[dict[str, object]],
    *,
    date: str,
    minimum_observations: int,
    factor_columns: list[str] | None = None,
) -> dict[str, object]:
    signals = [float(row["signal"]) for row in rows]
    returns = [float(row["forward_return"]) for row in rows]
    tickers = [str(row["ticker"]) for row in rows]
    groups = [str(row["neutral_group"]) for row in rows]
    clipped_returns = winsorize(returns)
    enough = len(rows) >= minimum_observations
    raw_ic = spearman(signals, clipped_returns) if enough else None
    neutral_ic = None
    factor_neutral_ic = None
    if enough:
        neutral_signals = group_demean(signals, groups)
        neutral_returns = group_demean(clipped_returns, groups)
        neutral_ic = spearman(neutral_signals, neutral_returns)
        if factor_columns:
            controls = [
                group_demean([float(row["factors"][column]) for row in rows], groups)
                for column in factor_columns
            ]
            factor_neutral_ic = spearman(
                residualize(neutral_signals, controls),
                residualize(neutral_returns, controls),
            )
    spread, top_count, bottom_count = nonoverlapping_quantile_spread(
        signals,
        clipped_returns,
        tickers,
    )
    return {
        "date": date,
        "observation_count": len(rows),
        "raw_spearman_ic": raw_ic,
        "group_neutral_spearman_ic": neutral_ic,
        "group_and_factor_neutral_spearman_ic": factor_neutral_ic,
        "winsorized_q5_minus_q1": spread if enough else None,
        "top_quantile_count": top_count,
        "bottom_quantile_count": bottom_count,
    }


def _mean(values: list[object]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.mean(present) if present else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate PIT signal IC and non-overlapping quantile spread against realized returns."
    )
    parser.add_argument("--signals", required=True, type=Path)
    parser.add_argument("--returns", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--return-column", default="forward_return")
    parser.add_argument("--group-column", default="sector")
    parser.add_argument(
        "--factor-column",
        action="append",
        default=[],
        help="numeric exposure column to neutralize after sector demeaning; may be repeated",
    )
    parser.add_argument("--minimum-observations", type=int, default=20)
    args = parser.parse_args()
    if args.minimum_observations < 5:
        raise ValueError("minimum observations must be at least five")

    signal_rows = read_csv(args.signals.resolve())
    return_rows = read_csv(args.returns.resolve())
    return_by_key: dict[tuple[str, str], dict[str, str]] = {}
    for row in return_rows:
        key = ((row.get("date") or "").strip(), (row.get("ticker") or "").strip())
        if not all(key):
            raise ValueError("returns require non-blank date and ticker")
        if key in return_by_key:
            raise ValueError(f"duplicate realized return: {key[0]}/{key[1]}")
        return_by_key[key] = row

    by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    eligible_count = 0
    missing_return_count = 0
    for row in signal_rows:
        if (row.get("signal_eligible") or "").strip() != "1":
            continue
        eligible_count += 1
        date = (row.get("date") or "").strip()
        ticker = (row.get("ticker") or "").strip()
        key = (date, ticker)
        signal_text = (row.get("signal") or "").strip()
        group = (row.get(args.group_column) or "").strip()
        if not signal_text:
            raise ValueError(f"eligible signal is blank: {date}/{ticker}")
        realized_row = return_by_key.get(key)
        return_text = (realized_row.get(args.return_column) or "").strip() if realized_row else ""
        if not return_text:
            missing_return_count += 1
            continue
        factors: dict[str, float] = {}
        for column in args.factor_column:
            factor_text = (row.get(column) or "").strip()
            if not factor_text and realized_row is not None:
                factor_text = (realized_row.get(column) or "").strip()
            if not factor_text:
                raise ValueError(f"missing factor {column} for {date}/{ticker}")
            factors[column] = _number(factor_text, field=column, key=key)
        by_date[date].append(
            {
                "ticker": ticker,
                "signal": _number(signal_text, field="signal", key=key),
                "forward_return": _number(return_text, field=args.return_column, key=key),
                "neutral_group": group,
                "factors": factors,
            }
        )

    metrics = [
        evaluate_date(
            rows,
            date=date,
            minimum_observations=args.minimum_observations,
            factor_columns=args.factor_column,
        )
        for date, rows in sorted(by_date.items())
    ]
    report = {
        "schema_version": "moatrader-signal-evaluation/1",
        "signals": str(args.signals.resolve()),
        "returns": str(args.returns.resolve()),
        "return_column": args.return_column,
        "neutral_group_column": args.group_column,
        "factor_columns": args.factor_column,
        "minimum_observations": args.minimum_observations,
        "eligible_signal_count": eligible_count,
        "matched_return_count": eligible_count - missing_return_count,
        "missing_return_count": missing_return_count,
        "date_count": len(metrics),
        "mean_raw_spearman_ic": _mean([row["raw_spearman_ic"] for row in metrics]),
        "mean_group_neutral_spearman_ic": _mean(
            [row["group_neutral_spearman_ic"] for row in metrics]
        ),
        "mean_group_and_factor_neutral_spearman_ic": _mean(
            [row["group_and_factor_neutral_spearman_ic"] for row in metrics]
        ),
        "mean_winsorized_q5_minus_q1": _mean(
            [row["winsorized_q5_minus_q1"] for row in metrics]
        ),
        "dates": metrics,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
