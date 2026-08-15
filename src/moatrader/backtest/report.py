from __future__ import annotations

import csv
from io import StringIO

from moatrader.backtest.models import BacktestResult


def equity_csv(result: BacktestResult) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["timestamp", "portfolio_value", "drawdown"])
    writer.writeheader()
    for point in result.equity_curve:
        writer.writerow(
            {
                "timestamp": point.timestamp.isoformat(),
                "portfolio_value": point.portfolio_value,
                "drawdown": point.drawdown,
            }
        )
    return stream.getvalue()


def rebalances_csv(result: BacktestResult) -> str:
    stream = StringIO(newline="")
    fieldnames = [
        "run_id",
        "signal_at",
        "execution_at",
        "selected_tickers",
        "requested_tickers",
        "unexecuted_tickers",
        "locked_tickers",
        "pre_trade_value",
        "post_trade_value",
        "turnover",
        "transaction_cost",
        "slippage_cost",
        "maximum_capacity_utilization",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for record in result.rebalances:
        writer.writerow(
            {
                "run_id": record.run_id,
                "signal_at": record.signal_at.isoformat(),
                "execution_at": record.execution_at.isoformat(),
                "selected_tickers": ",".join(record.selected_tickers),
                "requested_tickers": ",".join(record.requested_tickers),
                "unexecuted_tickers": ",".join(record.unexecuted_tickers),
                "locked_tickers": ",".join(record.locked_tickers),
                "pre_trade_value": record.pre_trade_value,
                "post_trade_value": record.post_trade_value,
                "turnover": record.turnover,
                "transaction_cost": record.transaction_cost,
                "slippage_cost": record.slippage_cost,
                "maximum_capacity_utilization": record.maximum_capacity_utilization,
            }
        )
    return stream.getvalue()
