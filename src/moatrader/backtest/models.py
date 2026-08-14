from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class BacktestConfig(ContractModel):
    end_at: datetime
    top_n: int = Field(default=10, ge=1)
    execution_lag_days: int = Field(default=1, ge=0, le=30)
    transaction_cost_bps: Decimal = Field(default=Decimal("10"), ge=0)
    initial_capital: Decimal = Field(default=Decimal("100000000"), gt=0)
    liquidate_at_end: bool = True
    maximum_signal_price_age_days: int = Field(default=7, ge=0, le=366)

    @model_validator(mode="after")
    def timestamps_are_aware(self) -> "BacktestConfig":
        if self.end_at.tzinfo is None or self.end_at.utcoffset() is None:
            raise ValueError("end_at must be timezone-aware")
        return self


class PricePoint(ContractModel):
    timestamp: datetime
    ticker: str = Field(min_length=1)
    adjusted_close: Decimal = Field(gt=0)

    @model_validator(mode="after")
    def timestamp_is_aware(self) -> "PricePoint":
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("price timestamp must be timezone-aware")
        return self


class RebalanceRecord(ContractModel):
    run_id: str
    signal_at: datetime
    execution_at: datetime
    selected_tickers: list[str]
    pre_trade_value: Decimal
    post_trade_value: Decimal
    turnover: Decimal = Field(ge=0)
    transaction_cost: Decimal = Field(ge=0)


class EquityPoint(ContractModel):
    timestamp: datetime
    portfolio_value: Decimal = Field(ge=0)
    drawdown: Decimal = Field(ge=-1, le=0)


class BacktestPerformance(ContractModel):
    initial_capital: Decimal
    ending_capital: Decimal
    total_return: Decimal
    cagr: float | None = None
    annualized_volatility: float | None = None
    max_drawdown: Decimal
    average_turnover: Decimal
    total_transaction_cost: Decimal
    rebalance_count: int = Field(ge=0)


class BacktestResult(ContractModel):
    started_at: datetime
    ended_at: datetime
    config: BacktestConfig
    source_run_ids: list[str]
    rebalances: list[RebalanceRecord]
    equity_curve: list[EquityPoint]
    performance: BacktestPerformance
