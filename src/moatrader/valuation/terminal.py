from __future__ import annotations

from decimal import Decimal


def stable_reinvestment_rate(*, growth: Decimal, roic: Decimal) -> Decimal:
    if roic <= growth:
        raise ValueError("stable ROIC must exceed stable growth")
    return growth / roic


def stable_fcff(*, next_period_nopat: Decimal, growth: Decimal, roic: Decimal) -> Decimal:
    return next_period_nopat * (Decimal(1) - stable_reinvestment_rate(growth=growth, roic=roic))


def gordon_value(*, cash_flow: Decimal, discount_rate: Decimal, growth: Decimal) -> Decimal:
    if discount_rate <= growth:
        raise ValueError("discount rate must exceed stable growth")
    return cash_flow / (discount_rate - growth)
