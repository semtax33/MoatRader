from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.base import ValuationMethod, ValuationResult, eligible
from moatrader.valuation.terminal import gordon_value


class ApvCase(ContractModel):
    unlevered_fcff: list[Decimal] = Field(min_length=1, max_length=60)
    terminal_cash_flow: Decimal
    terminal_growth: Decimal = Field(ge=-0.10, lt=0.20)
    unlevered_cost_of_capital: Decimal = Field(gt=0, lt=1)
    tax_shields: list[Decimal] = Field(default_factory=list, max_length=60)
    tax_shield_discount_rate: Decimal = Field(gt=0, lt=1)
    expected_distress_cost: Decimal = Field(default=Decimal(0), ge=0)

    @model_validator(mode="after")
    def finite_terminal(self) -> "ApvCase":
        if self.unlevered_cost_of_capital <= self.terminal_growth:
            raise ValueError("unlevered cost of capital must exceed terminal growth")
        return self


class ApvAssumptions(ContractModel):
    downside: ApvCase
    base: ApvCase
    upside: ApvCase
    debt: Decimal = Field(ge=0)
    cash: Decimal = Field(default=Decimal(0), ge=0)
    diluted_shares: Decimal = Field(gt=0)
    assumption_confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    provenance: list[str] = Field(default_factory=list)


class ApvEngine:
    @staticmethod
    def _enterprise(case: ApvCase) -> Decimal:
        explicit = sum(
            (
                cash_flow / ((Decimal(1) + case.unlevered_cost_of_capital) ** year)
                for year, cash_flow in enumerate(case.unlevered_fcff, start=1)
            ),
            Decimal(0),
        )
        terminal = gordon_value(
            cash_flow=case.terminal_cash_flow,
            discount_rate=case.unlevered_cost_of_capital,
            growth=case.terminal_growth,
        ) / ((Decimal(1) + case.unlevered_cost_of_capital) ** len(case.unlevered_fcff))
        shield = sum(
            (
                cash_flow / ((Decimal(1) + case.tax_shield_discount_rate) ** year)
                for year, cash_flow in enumerate(case.tax_shields, start=1)
            ),
            Decimal(0),
        )
        return explicit + terminal + shield - case.expected_distress_cost

    def value(self, assumptions: ApvAssumptions) -> ValuationResult:
        enterprises = [self._enterprise(case) for case in (assumptions.downside, assumptions.base, assumptions.upside)]
        equities = [value + assumptions.cash - assumptions.debt for value in enterprises]
        if equities != sorted(equities):
            raise ValueError("APV cases must be ordered downside <= base <= upside")
        shares = assumptions.diluted_shares
        return ValuationResult(
            method=ValuationMethod.APV,
            applicability=eligible(
                ValuationMethod.APV,
                ["unlevered_fcff", "tax_shields", "debt", "diluted_shares"],
            ),
            enterprise_value=enterprises[1],
            equity_value=equities[1],
            fair_value_per_share=equities[1] / shares,
            downside_value_per_share=equities[0] / shares,
            base_value_per_share=equities[1] / shares,
            upside_value_per_share=equities[2] / shares,
            assumption_confidence=assumptions.assumption_confidence,
            provenance=assumptions.provenance,
        )
