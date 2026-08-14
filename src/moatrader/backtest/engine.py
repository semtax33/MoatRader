from __future__ import annotations

import math
import statistics
from datetime import datetime, timedelta
from decimal import Decimal

from moatrader.backtest.models import (
    BacktestConfig,
    BacktestPerformance,
    BacktestResult,
    EquityPoint,
    RebalanceRecord,
)
from moatrader.backtest.prices import PricePanel
from moatrader.runner.models import UniverseRunResult


class PointInTimeBacktester:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config

    def run(self, runs: list[UniverseRunResult], prices: PricePanel) -> BacktestResult:
        signals = sorted((run for run in runs if run.as_of < self.config.end_at), key=lambda run: run.as_of)
        if not signals:
            raise ValueError("no run signals occur before end_at")
        if len({run.as_of for run in signals}) != len(signals):
            raise ValueError("only one universe run is allowed per signal timestamp")
        self._validate_signal_pit(signals)

        cash = self.config.initial_capital
        units: dict[str, Decimal] = {}
        rebalances: list[RebalanceRecord] = []
        equity: list[EquityPoint] = [
            EquityPoint(timestamp=signals[0].as_of, portfolio_value=cash, drawdown=Decimal(0))
        ]
        peak = cash
        last_execution = signals[0].as_of
        cost_rate = self.config.transaction_cost_bps / Decimal(10_000)

        for signal in signals:
            selected = [candidate.ticker for candidate in signal.ranking[: self.config.top_n]]
            target_tickers = set(selected)
            lookup_tickers = set(units) | target_tickers
            eligible_at = signal.as_of + timedelta(days=self.config.execution_lag_days)
            execution_at = prices.common_timestamp(lookup_tickers, eligible_at)
            if execution_at <= last_execution:
                raise ValueError("rebalance executions must be strictly increasing")
            if units:
                peak = self._append_marks(
                    equity,
                    prices,
                    units,
                    cash,
                    after=last_execution,
                    through=execution_at,
                    peak=peak,
                )
            execution_prices = prices.prices_at(lookup_tickers, execution_at)
            current_values = {ticker: quantity * execution_prices[ticker] for ticker, quantity in units.items()}
            pre_value = cash + sum(current_values.values(), Decimal(0))
            if pre_value <= 0:
                raise ValueError("portfolio value became non-positive")
            current_weights = {ticker: value / pre_value for ticker, value in current_values.items()}
            current_cash_weight = cash / pre_value
            target_weight = Decimal(1) / Decimal(len(selected)) if selected else Decimal(0)
            asset_turnover = sum(
                abs((target_weight if ticker in target_tickers else Decimal(0)) - current_weights.get(ticker, Decimal(0)))
                for ticker in lookup_tickers
            )
            target_cash_weight = Decimal(0) if selected else Decimal(1)
            turnover = (asset_turnover + abs(target_cash_weight - current_cash_weight)) / Decimal(2)
            transaction_cost = pre_value * cost_rate * turnover
            post_value = pre_value - transaction_cost
            if post_value <= 0:
                raise ValueError("transaction costs exhausted the portfolio")
            if selected:
                allocation = post_value / Decimal(len(selected))
                units = {ticker: allocation / execution_prices[ticker] for ticker in selected}
                cash = Decimal(0)
            else:
                units = {}
                cash = post_value
            rebalances.append(
                RebalanceRecord(
                    run_id=signal.run_id,
                    signal_at=signal.as_of,
                    execution_at=execution_at,
                    selected_tickers=selected,
                    pre_trade_value=pre_value,
                    post_trade_value=post_value,
                    turnover=turnover,
                    transaction_cost=transaction_cost,
                )
            )
            peak = max(peak, post_value)
            self._replace_or_append_equity(
                equity,
                EquityPoint(
                    timestamp=execution_at,
                    portfolio_value=post_value,
                    drawdown=post_value / peak - Decimal(1),
                ),
            )
            last_execution = execution_at

        exit_at = prices.common_timestamp(set(units), self.config.end_at)
        if exit_at <= last_execution:
            raise ValueError("end_at must resolve after the final rebalance execution")
        if units:
            peak = self._append_marks(
                equity,
                prices,
                units,
                cash,
                after=last_execution,
                through=exit_at,
                peak=peak,
            )
        exit_prices = prices.prices_at(set(units), exit_at)
        ending = cash + sum((quantity * exit_prices[ticker] for ticker, quantity in units.items()), Decimal(0))
        terminal_cost = Decimal(0)
        if self.config.liquidate_at_end and units:
            terminal_cost = ending * cost_rate
            ending -= terminal_cost
        peak = max(peak, ending)
        self._replace_or_append_equity(
            equity,
            EquityPoint(timestamp=exit_at, portfolio_value=ending, drawdown=ending / peak - Decimal(1)),
        )
        performance = self._performance(equity, rebalances, terminal_cost)
        return BacktestResult(
            started_at=equity[0].timestamp,
            ended_at=equity[-1].timestamp,
            config=self.config,
            source_run_ids=[run.run_id for run in signals],
            rebalances=rebalances,
            equity_curve=equity,
            performance=performance,
        )

    @staticmethod
    def _replace_or_append_equity(equity: list[EquityPoint], point: EquityPoint) -> None:
        if equity and equity[-1].timestamp == point.timestamp:
            equity[-1] = point
        else:
            equity.append(point)

    def _append_marks(
        self,
        equity: list[EquityPoint],
        prices: PricePanel,
        units: dict[str, Decimal],
        cash: Decimal,
        *,
        after: datetime,
        through: datetime,
        peak: Decimal,
    ) -> Decimal:
        tickers = set(units)
        for timestamp in prices.common_timestamps(tickers, after=after, through=through):
            point_prices = prices.prices_at(tickers, timestamp)
            value = cash + sum(
                (quantity * point_prices[ticker] for ticker, quantity in units.items()),
                Decimal(0),
            )
            peak = max(peak, value)
            self._replace_or_append_equity(
                equity,
                EquityPoint(timestamp=timestamp, portfolio_value=value, drawdown=value / peak - Decimal(1)),
            )
        return peak

    def _validate_signal_pit(self, signals: list[UniverseRunResult]) -> None:
        for run in signals:
            for candidate in run.ranking:
                if candidate.valuation_as_of > run.as_of:
                    raise ValueError(
                        f"run {run.run_id} candidate {candidate.ticker} has future valuation_as_of"
                    )
                if candidate.price_as_of > run.as_of:
                    raise ValueError(f"run {run.run_id} candidate {candidate.ticker} has future price_as_of")
                if (run.as_of - candidate.price_as_of) > timedelta(
                    days=self.config.maximum_signal_price_age_days
                ):
                    raise ValueError(
                        f"run {run.run_id} candidate {candidate.ticker} has stale price_as_of"
                    )

    def _performance(
        self,
        equity: list[EquityPoint],
        rebalances: list[RebalanceRecord],
        terminal_cost: Decimal,
    ) -> BacktestPerformance:
        initial = self.config.initial_capital
        ending = equity[-1].portfolio_value
        total_return = ending / initial - Decimal(1)
        elapsed_days = (equity[-1].timestamp - equity[0].timestamp).total_seconds() / 86_400
        cagr = None
        if elapsed_days > 0 and ending > 0:
            cagr = float((ending / initial) ** (Decimal(str(365.25 / elapsed_days))) - Decimal(1))
        returns: list[float] = []
        intervals: list[float] = []
        for previous, current in zip(equity, equity[1:]):
            days = (current.timestamp - previous.timestamp).total_seconds() / 86_400
            if days > 0 and previous.portfolio_value > 0:
                returns.append(float(current.portfolio_value / previous.portfolio_value - Decimal(1)))
                intervals.append(days)
        volatility = None
        if len(returns) >= 2 and intervals:
            volatility = statistics.stdev(returns) * math.sqrt(365.25 / statistics.mean(intervals))
        average_turnover = (
            sum((record.turnover for record in rebalances), Decimal(0)) / Decimal(len(rebalances))
            if rebalances
            else Decimal(0)
        )
        total_cost = sum((record.transaction_cost for record in rebalances), Decimal(0)) + terminal_cost
        return BacktestPerformance(
            initial_capital=initial,
            ending_capital=ending,
            total_return=total_return,
            cagr=cagr,
            annualized_volatility=volatility,
            max_drawdown=min(point.drawdown for point in equity),
            average_turnover=average_turnover,
            total_transaction_cost=total_cost,
            rebalance_count=len(rebalances),
        )
