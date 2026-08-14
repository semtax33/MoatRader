from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from pathlib import Path

import pytest

from moatrader.backtest import BacktestConfig, PointInTimeBacktester, PricePanel, PricePoint, load_price_panel
from moatrader.runner import UniverseRunResult
from moatrader.screening import RankedCandidate


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _run(run_id: str, signal_at: str, ticker: str, *, future_data: bool = False) -> UniverseRunResult:
    signal = _dt(signal_at)
    data_at = signal.replace(day=signal.day + 1) if future_data else signal
    candidate = RankedCandidate(
        issuer_id=ticker,
        ticker=ticker,
        price_to_dcf=Decimal("0.5"),
        margin_of_safety=Decimal("0.5"),
        moat_score=Decimal("7"),
        quality_value_score=Decimal("0.2"),
        valuation_as_of=data_at,
        price_as_of=data_at,
    )
    return UniverseRunResult(
        run_id=run_id,
        as_of=signal,
        started_at=signal,
        completed_at=signal,
        companies=[],
        ranking=[candidate],
    )


def _panel() -> PricePanel:
    return PricePanel(
        [
            PricePoint(timestamp=_dt("2025-01-02T16:00:00+00:00"), ticker="AAA", adjusted_close=100),
            PricePoint(timestamp=_dt("2025-01-02T16:00:00+00:00"), ticker="BBB", adjusted_close=50),
            PricePoint(timestamp=_dt("2025-02-02T16:00:00+00:00"), ticker="AAA", adjusted_close=110),
            PricePoint(timestamp=_dt("2025-02-02T16:00:00+00:00"), ticker="BBB", adjusted_close=50),
            PricePoint(timestamp=_dt("2025-03-02T16:00:00+00:00"), ticker="AAA", adjusted_close=120),
            PricePoint(timestamp=_dt("2025-03-02T16:00:00+00:00"), ticker="BBB", adjusted_close=60),
        ]
    )


def test_backtest_rotates_at_next_tradable_timestamp_without_lookahead() -> None:
    result = PointInTimeBacktester(
        BacktestConfig(
            end_at=_dt("2025-03-01T00:00:00+00:00"),
            top_n=1,
            execution_lag_days=1,
            transaction_cost_bps=0,
            initial_capital=Decimal("1000"),
        )
    ).run(
        [
            _run("r1", "2025-01-01T00:00:00+00:00", "AAA"),
            _run("r2", "2025-02-01T00:00:00+00:00", "BBB"),
        ],
        _panel(),
    )

    assert [record.execution_at for record in result.rebalances] == [
        _dt("2025-01-02T16:00:00+00:00"),
        _dt("2025-02-02T16:00:00+00:00"),
    ]
    assert result.performance.ending_capital == Decimal("1320")
    assert result.performance.total_return == Decimal("0.32")
    assert result.performance.max_drawdown == 0


def test_backtest_rejects_future_signal_inputs() -> None:
    with pytest.raises(ValueError, match="future valuation_as_of"):
        PointInTimeBacktester(
            BacktestConfig(end_at=_dt("2025-03-01T00:00:00+00:00"))
        ).run([_run("r1", "2025-01-01T00:00:00+00:00", "AAA", future_data=True)], _panel())


def test_backtest_rejects_stale_signal_price() -> None:
    run = _run("r1", "2025-01-20T00:00:00+00:00", "AAA")
    run.ranking[0].price_as_of = _dt("2025-01-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="stale price_as_of"):
        PointInTimeBacktester(
            BacktestConfig(end_at=_dt("2025-03-01T00:00:00+00:00"))
        ).run([run], _panel())


def test_backtest_fails_when_held_security_has_no_exit_price() -> None:
    points = [
        PricePoint(timestamp=_dt("2025-01-02T16:00:00+00:00"), ticker="AAA", adjusted_close=100)
    ]
    with pytest.raises(ValueError, match="no common tradable"):
        PointInTimeBacktester(
            BacktestConfig(end_at=_dt("2025-03-01T00:00:00+00:00"), transaction_cost_bps=0)
        ).run([_run("r1", "2025-01-01T00:00:00+00:00", "AAA")], PricePanel(points))


def test_transaction_costs_reduce_ending_value() -> None:
    zero_cost = PointInTimeBacktester(
        BacktestConfig(
            end_at=_dt("2025-03-01T00:00:00+00:00"),
            transaction_cost_bps=0,
            initial_capital=1000,
        )
    ).run([_run("r1", "2025-01-01T00:00:00+00:00", "AAA")], _panel())
    with_cost = PointInTimeBacktester(
        BacktestConfig(
            end_at=_dt("2025-03-01T00:00:00+00:00"),
            transaction_cost_bps=100,
            initial_capital=1000,
        )
    ).run([_run("r1", "2025-01-01T00:00:00+00:00", "AAA")], _panel())

    assert with_cost.performance.ending_capital < zero_cost.performance.ending_capital
    assert with_cost.performance.total_transaction_cost > 0


def test_equity_curve_marks_intermediate_drawdown() -> None:
    panel = PricePanel(
        [
            PricePoint(timestamp=_dt("2025-01-02T16:00:00+00:00"), ticker="AAA", adjusted_close=100),
            PricePoint(timestamp=_dt("2025-01-15T16:00:00+00:00"), ticker="AAA", adjusted_close=50),
            PricePoint(timestamp=_dt("2025-03-02T16:00:00+00:00"), ticker="AAA", adjusted_close=120),
        ]
    )
    result = PointInTimeBacktester(
        BacktestConfig(
            end_at=_dt("2025-03-01T00:00:00+00:00"),
            transaction_cost_bps=0,
            initial_capital=1000,
        )
    ).run([_run("r1", "2025-01-01T00:00:00+00:00", "AAA")], panel)

    assert result.performance.max_drawdown == Decimal("-0.5")
    assert any(point.timestamp == _dt("2025-01-15T16:00:00+00:00") for point in result.equity_curve)


def test_price_panel_loader_requires_timezone_and_adjusted_prices(tmp_path: Path) -> None:
    path = tmp_path / "prices.csv"
    path.write_text(
        "timestamp,ticker,adjusted_close,tradable\n"
        "2025-01-02T16:00:00+00:00,AAA,100,true\n"
        "2025-01-02T16:00:00+00:00,BBB,50,false\n",
        encoding="utf-8",
    )

    points = load_price_panel(path)

    assert [(point.ticker, point.adjusted_close) for point in points] == [("AAA", Decimal("100"))]
