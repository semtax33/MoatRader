from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_signal_panel import (
        _randomized_quantile_spreads,
        _sample_percentile,
        group_demean,
        signal_tie_diagnostics,
        winsorize,
    )
    from scripts.merge_kr_signal_panel import spearman
except ModuleNotFoundError:  # Direct ``python scripts\...py`` execution.
    from evaluate_signal_panel import (
        _randomized_quantile_spreads,
        _sample_percentile,
        group_demean,
        signal_tie_diagnostics,
        winsorize,
    )
    from merge_kr_signal_panel import spearman


SCHEMA_VERSION = "moatrader-old-vs-current-economic/1"
OLD_LABEL = "OLD_HOLISTIC"
CURRENT_LABEL = "CURRENT_PUBLIC"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ticker(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("ticker is blank")
    return text.zfill(6)


def _number(value: object, *, field: str, ticker: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field} for {ticker}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {field} for {ticker}")
    return result


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _index(
    rows: list[dict[str, str]],
    *,
    ticker_column: str,
    source: str,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        ticker = _ticker(row.get(ticker_column))
        if ticker in result:
            raise ValueError(f"duplicate ticker in {source}: {ticker}")
        result[ticker] = row
    return result


def _assert_close(left: float, right: float, *, label: str, ticker: str) -> None:
    if not math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{label} mismatch for {ticker}: {left} != {right}")


def build_exact_panel(
    *,
    protocol_path: Path,
    sample_metadata_path: Path,
    old_scores_path: Path,
    current_shadow_path: Path,
    forward_prices_path: Path,
    reference_panel_path: Path,
    signal_date: str,
    expected_eligible_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8-sig"))
    cohort = [_ticker(value) for value in protocol["economic"]["tickers"]]
    if len(cohort) != len(set(cohort)):
        raise ValueError("economic protocol contains duplicate tickers")

    metadata = _index(
        _read_csv(sample_metadata_path),
        ticker_column="ticker",
        source="sample metadata",
    )
    old_rows = [
        row
        for row in _read_csv(old_scores_path)
        if (row.get("as_of") or "").strip() == signal_date
    ]
    old_scores = _index(old_rows, ticker_column="stock_code", source="old scores")
    shadow_rows = _read_csv(current_shadow_path)
    shadow_dates = {(row.get("date") or "").strip() for row in shadow_rows}
    if shadow_dates != {signal_date}:
        raise ValueError(
            f"current shadow date mismatch: expected {signal_date}, found {sorted(shadow_dates)}"
        )
    current = _index(shadow_rows, ticker_column="ticker", source="current shadow")
    prices = _index(
        [
            row
            for row in _read_csv(forward_prices_path)
            if (row.get("signal_date") or "").strip() == signal_date
        ],
        ticker_column="ticker",
        source="forward prices",
    )
    reference = _index(
        _read_csv(reference_panel_path),
        ticker_column="ticker",
        source="reference panel",
    )

    sources = {
        "sample_metadata": metadata,
        "old_scores": old_scores,
        "current_shadow": current,
        "forward_prices": prices,
        "reference_panel": reference,
    }
    for name, indexed in sources.items():
        missing = sorted(set(cohort) - set(indexed))
        if missing:
            raise ValueError(f"{name} is missing cohort tickers: {missing}")

    panel: list[dict[str, Any]] = []
    excluded: list[dict[str, str]] = []
    return_dates: set[str] = set()
    for ticker in cohort:
        meta = metadata[ticker]
        old_row = old_scores[ticker]
        current_row = current[ticker]
        price_row = prices[ticker]
        reference_row = reference[ticker]
        eligible = _truthy(current_row.get("score_eligible"))
        if eligible != _truthy(reference_row.get("score_eligible")):
            raise ValueError(f"eligibility mismatch for {ticker}")

        old_score = _number(
            old_row.get("economic_moat_score_100"),
            field="old holistic score",
            ticker=ticker,
        )
        current_score = _number(
            current_row.get("economic_moat_score"),
            field="current public score",
            ticker=ticker,
        )
        signal_price = _number(price_row.get("signal_price"), field="signal price", ticker=ticker)
        return_price = _number(price_row.get("return_price"), field="return price", ticker=ticker)
        if signal_price <= 0:
            raise ValueError(f"signal price must be positive for {ticker}")
        forward_return = return_price / signal_price - 1
        return_dates.add((price_row.get("return_date") or "").strip())

        _assert_close(
            old_score,
            _number(meta.get("old_score_signal_date"), field="metadata old score", ticker=ticker),
            label="old source-to-metadata",
            ticker=ticker,
        )
        _assert_close(
            old_score,
            _number(reference_row.get("old_score"), field="reference old score", ticker=ticker),
            label="old source-to-reference",
            ticker=ticker,
        )
        _assert_close(
            current_score,
            _number(reference_row.get("moat_score"), field="reference current score", ticker=ticker),
            label="current source-to-reference",
            ticker=ticker,
        )
        _assert_close(
            forward_return,
            _number(reference_row.get("forward_return"), field="reference return", ticker=ticker),
            label="price-derived return-to-reference",
            ticker=ticker,
        )

        if not eligible:
            excluded.append(
                {
                    "ticker": ticker,
                    "eligibility_status": str(current_row.get("eligibility_status") or ""),
                }
            )
            continue
        panel.append(
            {
                "date": signal_date,
                "return_date": (price_row.get("return_date") or "").strip(),
                "ticker": ticker,
                "company_name": meta.get("company_name") or "",
                "market": meta.get("market") or "",
                "size_bucket": meta.get("size_bucket") or "",
                "old_holistic_score": old_score,
                "current_public_score": current_score,
                "forward_return": forward_return,
            }
        )

    if len(panel) != expected_eligible_count:
        raise ValueError(
            f"eligible count mismatch: expected {expected_eligible_count}, found {len(panel)}"
        )
    if return_dates == {""} or len(return_dates) != 1:
        raise ValueError(f"forward return dates are not singular: {sorted(return_dates)}")

    audit = {
        "protocol_cohort_count": len(cohort),
        "eligible_count": len(panel),
        "excluded_count": len(excluded),
        "excluded": excluded,
        "signal_date": signal_date,
        "return_date": next(iter(return_dates)),
        "source_value_checks": {
            "old_source_matches_metadata_and_reference": True,
            "current_shadow_matches_reference": True,
            "price_derived_returns_match_reference": True,
            "eligibility_matches_reference": True,
        },
    }
    return panel, audit


def _bootstrap_score_metrics(
    old_scores: list[float],
    current_scores: list[float],
    returns: list[float],
    *,
    samples: int,
    seed: str,
) -> dict[str, Any]:
    rng = random.Random(f"{seed}|bootstrap")
    old_values: list[float] = []
    current_values: list[float] = []
    deltas: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(returns)) for _ in returns]
        sampled_returns = [returns[index] for index in indices]
        old_ic = spearman([old_scores[index] for index in indices], sampled_returns)
        current_ic = spearman([current_scores[index] for index in indices], sampled_returns)
        if old_ic is None or current_ic is None:
            continue
        old_values.append(old_ic)
        current_values.append(current_ic)
        deltas.append(old_ic - current_ic)

    def interval(values: list[float]) -> dict[str, float | int | None]:
        return {
            "valid_samples": len(values),
            "lower_95": _sample_percentile(values, 0.025),
            "upper_95": _sample_percentile(values, 0.975),
            "positive_share": sum(value > 0 for value in values) / len(values) if values else None,
        }

    return {
        "requested_samples": samples,
        OLD_LABEL: interval(old_values),
        CURRENT_LABEL: interval(current_values),
        "OLD_MINUS_CURRENT": interval(deltas),
    }


def _permutation_p_values(
    old_scores: list[float],
    current_scores: list[float],
    returns: list[float],
    *,
    samples: int,
    seed: str,
) -> dict[str, float | int | None]:
    observed = {
        OLD_LABEL: spearman(old_scores, returns),
        CURRENT_LABEL: spearman(current_scores, returns),
    }
    exceed = {OLD_LABEL: 0, CURRENT_LABEL: 0}
    rng = random.Random(f"{seed}|permutation")
    candidate = list(returns)
    for _ in range(samples):
        rng.shuffle(candidate)
        for label, scores in ((OLD_LABEL, old_scores), (CURRENT_LABEL, current_scores)):
            rho = spearman(scores, candidate)
            if rho is not None and observed[label] is not None and abs(rho) >= abs(observed[label]):
                exceed[label] += 1
    return {
        "samples": samples,
        OLD_LABEL: (exceed[OLD_LABEL] + 1) / (samples + 1),
        CURRENT_LABEL: (exceed[CURRENT_LABEL] + 1) / (samples + 1),
    }


def _score_metrics(
    scores: list[float],
    returns: list[float],
    winsorized_returns: list[float],
    tickers: list[str],
    groups: list[str],
    *,
    simulations: int,
    seed: str,
) -> tuple[dict[str, Any], list[float]]:
    tail_count = len(scores) // 5
    spreads = _randomized_quantile_spreads(
        scores,
        winsorized_returns,
        tickers,
        tail_count=tail_count,
        simulations=simulations,
        seed=seed,
    )
    neutral_scores = group_demean(scores, groups)
    neutral_returns = group_demean(winsorized_returns, groups)
    return (
        {
            "rank_ic": spearman(scores, returns),
            "winsorized_rank_ic": spearman(scores, winsorized_returns),
            "market_size_return_neutral_rank_ic": spearman(scores, neutral_returns),
            "market_size_both_sides_neutral_rank_ic": spearman(
                neutral_scores, neutral_returns
            ),
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


def compare_panel(
    panel: list[dict[str, Any]],
    *,
    tie_simulations: int,
    bootstrap_samples: int,
    permutation_samples: int,
    seed: str,
) -> dict[str, Any]:
    if len(panel) < 5:
        raise ValueError("at least five eligible observations are required")
    if min(tie_simulations, bootstrap_samples, permutation_samples) < 1:
        raise ValueError("simulation counts must be positive")

    tickers = [str(row["ticker"]) for row in panel]
    old_scores = [float(row["old_holistic_score"]) for row in panel]
    current_scores = [float(row["current_public_score"]) for row in panel]
    returns = [float(row["forward_return"]) for row in panel]
    clipped_returns = winsorize(returns, 0.01, 0.99)
    groups = [f"{row['market']}|{row['size_bucket']}" for row in panel]

    old_metrics, old_spreads = _score_metrics(
        old_scores,
        returns,
        clipped_returns,
        tickers,
        groups,
        simulations=tie_simulations,
        seed=seed,
    )
    current_metrics, current_spreads = _score_metrics(
        current_scores,
        returns,
        clipped_returns,
        tickers,
        groups,
        simulations=tie_simulations,
        seed=seed,
    )
    spread_deltas = [
        old - current
        for old, current in zip(old_spreads, current_spreads, strict=True)
    ]
    bootstrap = _bootstrap_score_metrics(
        old_scores,
        current_scores,
        clipped_returns,
        samples=bootstrap_samples,
        seed=seed,
    )
    permutation = _permutation_p_values(
        old_scores,
        current_scores,
        clipped_returns,
        samples=permutation_samples,
        seed=seed,
    )
    return {
        "observation_count": len(panel),
        "forward_return_winsorization": {"lower": 0.01, "upper": 0.99},
        "score_scale_note": "Spearman and equal-count ranking are invariant to the different old/current score scales.",
        "tie_method": "FIXED_SEED_MONTE_CARLO_EQUAL_TAILS",
        "tie_seed": seed,
        "tie_simulations": tie_simulations,
        "metrics": {
            OLD_LABEL: old_metrics,
            CURRENT_LABEL: current_metrics,
        },
        "direct_comparison": {
            "old_minus_current_rank_ic": old_metrics["rank_ic"] - current_metrics["rank_ic"],
            "old_minus_current_winsorized_rank_ic": (
                old_metrics["winsorized_rank_ic"] - current_metrics["winsorized_rank_ic"]
            ),
            "old_minus_current_market_size_return_neutral_rank_ic": (
                old_metrics["market_size_return_neutral_rank_ic"]
                - current_metrics["market_size_return_neutral_rank_ic"]
            ),
            "old_minus_current_market_size_both_sides_neutral_rank_ic": (
                old_metrics["market_size_both_sides_neutral_rank_ic"]
                - current_metrics["market_size_both_sides_neutral_rank_ic"]
            ),
            "old_vs_current_score_spearman": spearman(old_scores, current_scores),
            "old_minus_current_q5_minus_q1": {
                "mean": statistics.mean(spread_deltas),
                "median": statistics.median(spread_deltas),
                "p05_tie_assignment": _sample_percentile(spread_deltas, 0.05),
                "p95_tie_assignment": _sample_percentile(spread_deltas, 0.95),
                "positive_share": sum(value > 0 for value in spread_deltas) / len(spread_deltas),
            },
        },
        "paired_row_bootstrap_rank_ic": bootstrap,
        "two_sided_return_permutation_p_value": permutation,
    }


def _fmt(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value * 100:.2f}%" if percent else f"{value:.4f}"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    comparison = report["comparison"]
    metrics = comparison["metrics"]
    old = metrics[OLD_LABEL]
    current = metrics[CURRENT_LABEL]
    direct = comparison["direct_comparison"]
    bootstrap_delta = comparison["paired_row_bootstrap_rank_ic"]["OLD_MINUS_CURRENT"]
    lines = [
        "# OLD holistic vs CURRENT public — exact same economic 22",
        "",
        f"- Signal date: **{report['panel_audit']['signal_date']}**",
        f"- Forward return date: **{report['panel_audit']['return_date']}**",
        f"- Eligible observations: **{comparison['observation_count']}** (same names and returns)",
        "- API / LLM calls: **0 / 0**",
        "",
        "## Result",
        "",
        "| Metric | OLD holistic | CURRENT public | OLD − CURRENT |",
        "| --- | ---: | ---: | ---: |",
        f"| Rank IC | {_fmt(old['rank_ic'])} | {_fmt(current['rank_ic'])} | "
        f"{_fmt(direct['old_minus_current_rank_ic'])} |",
        f"| Winsorized Rank IC | {_fmt(old['winsorized_rank_ic'])} | "
        f"{_fmt(current['winsorized_rank_ic'])} | "
        f"{_fmt(direct['old_minus_current_winsorized_rank_ic'])} |",
        f"| Market×size return-neutral IC | {_fmt(old['market_size_return_neutral_rank_ic'])} | "
        f"{_fmt(current['market_size_return_neutral_rank_ic'])} | "
        f"{_fmt(direct['old_minus_current_market_size_return_neutral_rank_ic'])} |",
        f"| Market×size both-sides-neutral IC | {_fmt(old['market_size_both_sides_neutral_rank_ic'])} | "
        f"{_fmt(current['market_size_both_sides_neutral_rank_ic'])} | "
        f"{_fmt(direct['old_minus_current_market_size_both_sides_neutral_rank_ic'])} |",
        f"| Equal-count Q5−Q1 mean | {_fmt(old['equal_count_q5_minus_q1']['mean'], percent=True)} | "
        f"{_fmt(current['equal_count_q5_minus_q1']['mean'], percent=True)} | "
        f"{_fmt(direct['old_minus_current_q5_minus_q1']['mean'], percent=True)} |",
        f"| Q5−Q1 positive tie assignments | {_fmt(old['equal_count_q5_minus_q1']['positive_share'], percent=True)} | "
        f"{_fmt(current['equal_count_q5_minus_q1']['positive_share'], percent=True)} | "
        f"{(old['equal_count_q5_minus_q1']['positive_share'] - current['equal_count_q5_minus_q1']['positive_share']) * 100:+.2f} pp |",
        "",
        f"The OLD and CURRENT score ordering has Spearman rho "
        f"**{_fmt(direct['old_vs_current_score_spearman'])}** on these 22 names.",
        f"Across paired tie assignments, OLD Q5−Q1 exceeds CURRENT in "
        f"**{_fmt(direct['old_minus_current_q5_minus_q1']['positive_share'], percent=True)}** "
        "of simulations.",
        "",
        "## Uncertainty",
        "",
        f"- OLD Rank IC two-sided permutation p-value: "
        f"{_fmt(comparison['two_sided_return_permutation_p_value'][OLD_LABEL])}",
        f"- CURRENT Rank IC two-sided permutation p-value: "
        f"{_fmt(comparison['two_sided_return_permutation_p_value'][CURRENT_LABEL])}",
        f"- Paired-bootstrap 95% interval for OLD−CURRENT Rank IC: "
        f"[{_fmt(bootstrap_delta['lower_95'])}, {_fmt(bootstrap_delta['upper_95'])}]",
        "- The Q5−Q1 p05/p95 ranges measure cutoff-tie assignment uncertainty only; "
        "they are not confidence intervals for future performance.",
        "",
        "## Interpretation",
        "",
        "OLD is directionally stronger in the raw cross-section: it has the higher Rank IC "
        "and a positive equal-count tail spread, while CURRENT has a negative tail spread. "
        "This is evidence consistent with economically useful ordering being lost in the CURRENT "
        "public reduction, but the market×size-neutral diagnostics are near zero for both. It is "
        "therefore not proof: the paired-bootstrap interval spans zero and this is only one "
        "22-name period.",
        "",
        "The result supports keeping the current 0–4 ordinal/public-score design frozen and moving to a fresh "
        "multi-period holdout before changing score resolution. It does not support a 0–10 scale change.",
        "",
        "## Method",
        "",
        "Both scores use the exact same eligible tickers and price-derived forward returns. Returns "
        "are winsorized at 1%/99% for tail portfolios. Q1 and Q5 each contain exactly four names. "
        f"All cutoff ties are integrated over {comparison['tie_simulations']:,} deterministic "
        "Monte Carlo assignments using the same seed for both scores.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    output: Path,
    panel: list[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "old-vs-current-economic-panel.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(panel[0]))
        writer.writeheader()
        writer.writerows(panel)
    (output / "old-vs-current-economic-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "old-vs-current-economic-report.md").write_text(
        _markdown(report),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="API-zero OLD holistic vs CURRENT public comparison on one exact cohort."
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--sample-metadata", required=True, type=Path)
    parser.add_argument("--old-scores", required=True, type=Path)
    parser.add_argument("--current-shadow", required=True, type=Path)
    parser.add_argument("--forward-prices", required=True, type=Path)
    parser.add_argument("--reference-panel", required=True, type=Path)
    parser.add_argument("--signal-date", required=True)
    parser.add_argument("--expected-eligible-count", type=int, default=22)
    parser.add_argument("--tie-simulations", type=int, default=10_000)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--permutation-samples", type=int, default=10_000)
    parser.add_argument("--seed", default="economic-old-vs-current-v1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    paths = {
        "protocol": args.protocol.resolve(),
        "sample_metadata": args.sample_metadata.resolve(),
        "old_scores": args.old_scores.resolve(),
        "current_shadow": args.current_shadow.resolve(),
        "forward_prices": args.forward_prices.resolve(),
        "reference_panel": args.reference_panel.resolve(),
    }
    panel, panel_audit = build_exact_panel(
        protocol_path=paths["protocol"],
        sample_metadata_path=paths["sample_metadata"],
        old_scores_path=paths["old_scores"],
        current_shadow_path=paths["current_shadow"],
        forward_prices_path=paths["forward_prices"],
        reference_panel_path=paths["reference_panel"],
        signal_date=args.signal_date,
        expected_eligible_count=args.expected_eligible_count,
    )
    comparison = compare_panel(
        panel,
        tie_simulations=args.tie_simulations,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
        seed=args.seed,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "api_calls": 0,
            "llm_calls": 0,
            "network_access_required": False,
        },
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "panel_audit": panel_audit,
        "comparison": comparison,
        "conclusion": {
            "classification": "OLD_RAW_DIRECTIONALLY_STRONGER_NEUTRAL_INCONCLUSIVE_SINGLE_CROSS_SECTION",
            "supports_zero_to_ten_change": False,
            "next_test": "FRESH_MULTI_PERIOD_ECONOMIC_HOLDOUT",
            "caution": "One 22-name period cannot identify standalone alpha or prove reducer causality.",
        },
    }
    write_outputs(args.output.resolve(), panel, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
