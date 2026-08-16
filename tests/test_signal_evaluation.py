from __future__ import annotations

import pytest

from scripts.evaluate_signal_panel import (
    evaluate_date,
    group_demean,
    nonoverlapping_quantile_spread,
    residualize,
    signal_tie_diagnostics,
    winsorize,
)


def test_nonoverlapping_quantile_spread_uses_disjoint_tails() -> None:
    spread, top_count, bottom_count = nonoverlapping_quantile_spread(
        [float(value) for value in range(10)],
        [float(value) / 10 for value in range(10)],
        [f"T{value}" for value in range(10)],
    )

    assert spread == pytest.approx(0.8)
    assert top_count == 2
    assert bottom_count == 2


def test_quantile_spread_randomizes_cutoff_ties_instead_of_using_ticker_order() -> None:
    signals = [1.0] * 10
    returns = [float(value) for value in range(10)]
    tickers = [f"T{value}" for value in range(10)]

    first = nonoverlapping_quantile_spread(
        signals, returns, tickers, simulations=2000, seed="fixed"
    )
    reordered = nonoverlapping_quantile_spread(
        list(reversed(signals)),
        list(reversed(returns)),
        list(reversed(tickers)),
        simulations=2000,
        seed="fixed",
    )
    diagnostics = signal_tie_diagnostics(signals)

    assert first == reordered
    assert abs(first[0]) < 0.25
    assert first[1:] == (2, 2)
    assert diagnostics == {
        "distinct_signal_count": 1,
        "max_single_signal_share": 1.0,
        "top_boundary_tie_count": 10,
        "bottom_boundary_tie_count": 10,
    }


def test_group_neutral_evaluation_removes_between_sector_level_effect() -> None:
    rows = []
    for index in range(10):
        group = "A" if index < 5 else "B"
        within = index % 5
        rows.append(
            {
                "ticker": f"T{index}",
                "signal": float(index),
                "forward_return": float((100 if group == "B" else 0) - within),
                "neutral_group": group,
            }
        )

    metrics = evaluate_date(rows, date="2025-01-01", minimum_observations=5)

    assert metrics["raw_spearman_ic"] > 0
    assert metrics["group_neutral_spearman_ic"] < -0.95
    assert metrics["distinct_signal_count"] == 10
    assert metrics["top_quantile_count"] == metrics["bottom_quantile_count"]


def test_group_demean_rejects_missing_sector() -> None:
    with pytest.raises(ValueError, match="blank"):
        group_demean([1.0], [""])


def test_winsorize_clips_extreme_returns() -> None:
    clipped = winsorize([0.0] * 99 + [1000.0])

    assert clipped[-1] < 1000.0


def test_residualize_removes_linear_factor_exposure() -> None:
    factor = [float(value) for value in range(10)]
    values = [2 * value + (1 if index % 2 else -1) for index, value in enumerate(factor)]

    residuals = residualize(values, [factor])

    assert abs(sum(a * b for a, b in zip(residuals, factor, strict=True))) < 1e-9
