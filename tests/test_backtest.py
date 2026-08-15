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
            slippage_bps=0,
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


def test_backtest_uses_conservative_settlement_when_exit_price_is_missing() -> None:
    points = [
        PricePoint(timestamp=_dt("2025-01-02T16:00:00+00:00"), ticker="AAA", adjusted_close=100)
    ]
    result = PointInTimeBacktester(
            BacktestConfig(end_at=_dt("2025-03-01T00:00:00+00:00"), transaction_cost_bps=0, slippage_bps=0)
        ).run([_run("r1", "2025-01-01T00:00:00+00:00", "AAA")], PricePanel(points))

    assert result.performance.ending_capital == 0
    assert result.forced_settlements[0].ticker == "AAA"
    assert result.forced_settlements[0].assumed_return == -1


def test_transaction_costs_reduce_ending_value() -> None:
    zero_cost = PointInTimeBacktester(
        BacktestConfig(
            end_at=_dt("2025-03-01T00:00:00+00:00"),
            transaction_cost_bps=0,
            slippage_bps=0,
            initial_capital=1000,
        )
    ).run([_run("r1", "2025-01-01T00:00:00+00:00", "AAA")], _panel())
    with_cost = PointInTimeBacktester(
        BacktestConfig(
            end_at=_dt("2025-03-01T00:00:00+00:00"),
            transaction_cost_bps=100,
            slippage_bps=0,
            initial_capital=1000,
        )
    ).run([_run("r1", "2025-01-01T00:00:00+00:00", "AAA")], _panel())

    assert with_cost.performance.ending_capital < zero_cost.performance.ending_capital
    assert with_cost.performance.total_transaction_cost > 0


def test_slippage_is_reported_separately_from_transaction_cost() -> None:
    result = PointInTimeBacktester(
        BacktestConfig(
            end_at=_dt("2025-03-01T00:00:00+00:00"),
            transaction_cost_bps=0,
            slippage_bps=100,
            initial_capital=1000,
        )
    ).run([_run("r1", "2025-01-01T00:00:00+00:00", "AAA")], _panel())

    assert result.performance.total_transaction_cost == 0
    assert result.performance.total_slippage_cost > 0


def test_suspended_selection_does_not_delay_the_whole_rebalance() -> None:
    run = _run("r1", "2025-01-01T00:00:00+00:00", "AAA")
    run.ranking.append(
        RankedCandidate(
            issuer_id="BBB",
            ticker="BBB",
            price_to_dcf=Decimal("0.5"),
            margin_of_safety=Decimal("0.5"),
            moat_score=Decimal("7"),
            quality_value_score=Decimal("0.1"),
            valuation_as_of=run.as_of,
            price_as_of=run.as_of,
        )
    )
    panel = PricePanel(
        [
            PricePoint(timestamp=_dt("2025-01-02T16:00:00+00:00"), ticker="AAA", adjusted_close=100),
            PricePoint(timestamp=_dt("2025-01-03T16:00:00+00:00"), ticker="BBB", adjusted_close=50),
            PricePoint(timestamp=_dt("2025-03-02T16:00:00+00:00"), ticker="AAA", adjusted_close=110),
            PricePoint(timestamp=_dt("2025-03-02T16:00:00+00:00"), ticker="BBB", adjusted_close=55),
        ]
    )

    result = PointInTimeBacktester(
        BacktestConfig(
            end_at=_dt("2025-03-01T00:00:00+00:00"),
            top_n=2,
            transaction_cost_bps=0,
            slippage_bps=0,
        )
    ).run([run], panel)

    assert result.rebalances[0].execution_at == _dt("2025-01-02T16:00:00+00:00")
    assert result.rebalances[0].selected_tickers == ["AAA"]
    assert result.rebalances[0].unexecuted_tickers == ["BBB"]


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
            slippage_bps=0,
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


def test_backtest_reports_benchmark_and_excess_return() -> None:
    panel = PricePanel(
        [
            PricePoint(timestamp=_dt("2025-01-02T16:00:00+00:00"), ticker="AAA", adjusted_close=100),
            PricePoint(timestamp=_dt("2025-03-02T16:00:00+00:00"), ticker="AAA", adjusted_close=120),
            PricePoint(timestamp=_dt("2025-01-02T16:00:00+00:00"), ticker="INDEX", adjusted_close=200),
            PricePoint(timestamp=_dt("2025-03-02T16:00:00+00:00"), ticker="INDEX", adjusted_close=220),
        ]
    )
    result = PointInTimeBacktester(
        BacktestConfig(
            end_at=_dt("2025-03-01T00:00:00+00:00"),
            transaction_cost_bps=0,
            slippage_bps=0,
            benchmark_ticker="INDEX",
        )
    ).run([_run("r1", "2025-01-01T00:00:00+00:00", "AAA")], panel)

    assert result.performance.total_return == Decimal("0.2")
    assert result.performance.benchmark_total_return == Decimal("0.1")
    assert result.performance.excess_total_return == Decimal("0.1")


def test_capacity_gate_uses_trade_notional_and_requires_volume() -> None:
    points = [
        PricePoint(
            timestamp=_dt("2025-01-02T16:00:00+00:00"),
            ticker="AAA",
            adjusted_close=100,
            dollar_volume=1000,
        ),
        PricePoint(timestamp=_dt("2025-03-02T16:00:00+00:00"), ticker="AAA", adjusted_close=110),
    ]
    with pytest.raises(ValueError, match="daily dollar volume"):
        PointInTimeBacktester(
            BacktestConfig(
                end_at=_dt("2025-03-01T00:00:00+00:00"),
                initial_capital=1000,
                transaction_cost_bps=0,
                slippage_bps=0,
                enforce_capacity=True,
                maximum_participation_rate=Decimal("0.05"),
            )
        ).run([_run("r1", "2025-01-01T00:00:00+00:00", "AAA")], PricePanel(points))


def test_backtest_rejects_price_series_without_distribution_adjustment() -> None:
    with pytest.raises(ValueError, match="including distributions"):
        BacktestConfig(
            end_at=_dt("2025-03-01T00:00:00+00:00"),
            adjusted_close_includes_distributions=False,
        )
