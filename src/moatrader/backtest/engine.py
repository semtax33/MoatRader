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
    ForcedSettlement,
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
        transaction_cost_rate = self.config.transaction_cost_bps / Decimal(10_000)
        slippage_rate = self.config.slippage_bps / Decimal(10_000)
        forced_settlements: list[ForcedSettlement] = []

        for signal in signals:
            requested = [candidate.ticker for candidate in signal.ranking[: self.config.top_n]]
            eligible_at = signal.as_of + timedelta(days=self.config.execution_lag_days)
            execution_at = prices.market_timestamp(eligible_at)
            if execution_at is None:
                raise ValueError(f"no market timestamp at or after {eligible_at.isoformat()}")
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
            lookup_tickers = set(units) | set(requested)
            exact_prices = prices.tradable_prices_at(lookup_tickers, execution_at)
            mark_prices = prices.prices_at_or_before(set(units), execution_at) if units else {}
            locked_tickers = set(units) - set(exact_prices)
            selected = [ticker for ticker in requested if ticker in exact_prices]
            target_tickers = set(selected)
            current_values = {ticker: quantity * mark_prices[ticker] for ticker, quantity in units.items()}
            pre_value = cash + sum(current_values.values(), Decimal(0))
            if pre_value <= 0:
                raise ValueError("portfolio value became non-positive")
            locked_value = sum((current_values[ticker] for ticker in locked_tickers), Decimal(0))
            current_weights = {
                ticker: value / pre_value
                for ticker, value in current_values.items()
                if ticker not in locked_tickers
            }
            current_cash_weight = cash / pre_value
            investable_weight = max(Decimal(0), Decimal(1) - locked_value / pre_value)
            target_weight = investable_weight / Decimal(len(selected)) if selected else Decimal(0)
            asset_turnover = sum(
                abs((target_weight if ticker in target_tickers else Decimal(0)) - current_weights.get(ticker, Decimal(0)))
                for ticker in (set(current_weights) | target_tickers)
            )
            target_cash_weight = Decimal(0) if selected else investable_weight
            turnover = (asset_turnover + abs(target_cash_weight - current_cash_weight)) / Decimal(2)
            if turnover > self.config.maximum_turnover:
                raise ValueError(
                    f"rebalance turnover {turnover} exceeds configured maximum {self.config.maximum_turnover}"
                )
            transaction_cost = pre_value * transaction_cost_rate * turnover
            slippage_cost = pre_value * slippage_rate * turnover
            post_value = pre_value - transaction_cost - slippage_cost
            if post_value <= 0:
                raise ValueError("transaction costs exhausted the portfolio")
            locked_units = {ticker: units[ticker] for ticker in locked_tickers}
            investable_value = post_value - locked_value
            if investable_value < 0:
                raise ValueError("locked holdings exceed post-cost portfolio value")
            if selected:
                allocation = investable_value / Decimal(len(selected))
                capacity_utilizations: list[Decimal] = []
                if self.config.enforce_capacity:
                    for ticker in selected:
                        dollar_volume = prices.dollar_volume_at(ticker, execution_at)
                        if dollar_volume is None:
                            raise ValueError(
                                f"capacity enforcement requires dollar_volume for {ticker} at {execution_at.isoformat()}"
                            )
                        current_value = current_values.get(ticker, Decimal(0))
                        trade_notional = abs(allocation - current_value)
                        utilization = trade_notional / dollar_volume
                        capacity_utilizations.append(utilization)
                        if utilization > self.config.maximum_participation_rate:
                            raise ValueError(
                                f"target trade for {ticker} uses {utilization:.4f} of daily dollar volume, "
                                f"above {self.config.maximum_participation_rate:.4f}"
                            )
                units = {
                    **locked_units,
                    **{ticker: allocation / exact_prices[ticker] for ticker in selected},
                }
                cash = Decimal(0)
            else:
                capacity_utilizations = []
                units = locked_units
                cash = investable_value
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
                    slippage_cost=slippage_cost,
                    requested_tickers=requested,
                    unexecuted_tickers=[ticker for ticker in requested if ticker not in target_tickers],
                    locked_tickers=sorted(locked_tickers),
                    maximum_capacity_utilization=(
                        max(capacity_utilizations) if capacity_utilizations else None
                    ),
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

        exit_at = prices.market_timestamp(self.config.end_at) or self.config.end_at
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
        exit_prices = prices.tradable_prices_at(set(units), exit_at)
        ending = cash
        for ticker, quantity in units.items():
            if ticker in exit_prices:
                ending += quantity * exit_prices[ticker]
                continue
            last_at, last_price = prices.last_point_at_or_before(ticker, exit_at)
            settlement_value = quantity * last_price * (Decimal(1) + self.config.missing_exit_return)
            ending += settlement_value
            forced_settlements.append(
                ForcedSettlement(
                    ticker=ticker,
                    settlement_at=exit_at,
                    last_price_at=last_at,
                    last_price=last_price,
                    assumed_return=self.config.missing_exit_return,
                    settlement_value=settlement_value,
                    reason="NO_EXIT_PRICE_CONSERVATIVE_SETTLEMENT",
                )
            )
        terminal_transaction_cost = Decimal(0)
        terminal_slippage_cost = Decimal(0)
        if self.config.liquidate_at_end and units:
            terminal_transaction_cost = ending * transaction_cost_rate
            terminal_slippage_cost = ending * slippage_rate
            ending -= terminal_transaction_cost + terminal_slippage_cost
        peak = max(peak, ending)
        self._replace_or_append_equity(
            equity,
            EquityPoint(timestamp=exit_at, portfolio_value=ending, drawdown=ending / peak - Decimal(1)),
        )
        benchmark_return = None
        if self.config.benchmark_ticker:
            benchmark_prices = prices.tradable_prices_at(
                {self.config.benchmark_ticker},
                rebalances[0].execution_at,
            )
            benchmark_end_prices = prices.tradable_prices_at(
                {self.config.benchmark_ticker},
                exit_at,
            )
            if self.config.benchmark_ticker not in benchmark_prices or self.config.benchmark_ticker not in benchmark_end_prices:
                raise ValueError(
                    f"benchmark {self.config.benchmark_ticker} requires prices exactly at "
                    f"{rebalances[0].execution_at.isoformat()} and {exit_at.isoformat()}"
                )
            benchmark_start = benchmark_prices[self.config.benchmark_ticker]
            benchmark_end = benchmark_end_prices[self.config.benchmark_ticker]
            benchmark_return = benchmark_end / benchmark_start - Decimal(1)
        performance = self._performance(
            equity,
            rebalances,
            terminal_transaction_cost,
            terminal_slippage_cost,
            benchmark_return,
        )
        return BacktestResult(
            started_at=equity[0].timestamp,
            ended_at=equity[-1].timestamp,
            config=self.config,
            source_run_ids=[run.run_id for run in signals],
            rebalances=rebalances,
            equity_curve=equity,
            performance=performance,
            forced_settlements=forced_settlements,
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
        for timestamp in prices.mark_timestamps(tickers, after=after, through=through):
            point_prices = prices.prices_at_or_before(tickers, timestamp)
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
        terminal_transaction_cost: Decimal,
        terminal_slippage_cost: Decimal,
        benchmark_return: Decimal | None,
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
        total_transaction_cost = (
            sum((record.transaction_cost for record in rebalances), Decimal(0))
            + terminal_transaction_cost
        )
        total_slippage_cost = (
            sum((record.slippage_cost for record in rebalances), Decimal(0))
            + terminal_slippage_cost
        )
        return BacktestPerformance(
            initial_capital=initial,
            ending_capital=ending,
            total_return=total_return,
            cagr=cagr,
            annualized_volatility=volatility,
            max_drawdown=min(point.drawdown for point in equity),
            average_turnover=average_turnover,
            total_transaction_cost=total_transaction_cost,
            total_slippage_cost=total_slippage_cost,
            rebalance_count=len(rebalances),
            benchmark_total_return=benchmark_return,
            excess_total_return=total_return - benchmark_return if benchmark_return is not None else None,
        )
