from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.base import ValuationMethod, ValuationResult, eligible


class RimAssumptions(ContractModel):
    book_equity: Decimal = Field(gt=0)
    roe_path: list[Decimal] = Field(min_length=1, max_length=30)
    cost_of_equity: Decimal = Field(gt=0, lt=1)
    payout_ratio: Decimal = Field(ge=0, le=1)
    terminal_roe: Decimal = Field(gt=-1, lt=1)
    terminal_growth: Decimal = Field(ge=-0.10, lt=0.20)
    diluted_shares: Decimal = Field(gt=0)
    assumption_confidence: Decimal = Field(default=Decimal("0.5"), ge=0, le=1)
    provenance: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def terminal_is_finite(self) -> "RimAssumptions":
        if self.cost_of_equity <= self.terminal_growth:
            raise ValueError("cost of equity must exceed terminal growth")
        return self


class RimProjection(ContractModel):
    year: int = Field(ge=1)
    beginning_book_equity: Decimal
    roe: Decimal
    net_income: Decimal
    residual_income: Decimal
    ending_book_equity: Decimal
    present_value: Decimal


class RimValuation(ContractModel):
    projections: list[RimProjection]
    terminal_residual_income: Decimal
    terminal_value: Decimal
    equity_value: Decimal
    fair_value_per_share: Decimal


class RimEngine:
    def value(self, assumptions: RimAssumptions) -> RimValuation:
        book = assumptions.book_equity
        projections: list[RimProjection] = []
        for year, roe in enumerate(assumptions.roe_path, start=1):
            net_income = book * roe
            residual_income = net_income - assumptions.cost_of_equity * book
            ending_book = book + net_income * (Decimal(1) - assumptions.payout_ratio)
            present = residual_income / ((Decimal(1) + assumptions.cost_of_equity) ** year)
            projections.append(
                RimProjection(
                    year=year,
                    beginning_book_equity=book,
                    roe=roe,
                    net_income=net_income,
                    residual_income=residual_income,
                    ending_book_equity=ending_book,
                    present_value=present,
                )
            )
            book = ending_book
        terminal_book = book * (Decimal(1) + assumptions.terminal_growth)
        terminal_ri = terminal_book * (assumptions.terminal_roe - assumptions.cost_of_equity)
        terminal_value = terminal_ri / (
            assumptions.cost_of_equity - assumptions.terminal_growth
        )
        terminal_present = terminal_value / (
            (Decimal(1) + assumptions.cost_of_equity) ** len(projections)
        )
        equity = assumptions.book_equity + sum(
            (item.present_value for item in projections), Decimal(0)
        ) + terminal_present
        return RimValuation(
            projections=projections,
            terminal_residual_income=terminal_ri,
            terminal_value=terminal_value,
            equity_value=equity,
            fair_value_per_share=equity / assumptions.diluted_shares,
        )


class RimScenarioSet(ContractModel):
    downside: RimAssumptions
    base: RimAssumptions
    upside: RimAssumptions

    @model_validator(mode="after")
    def same_capital_base(self) -> "RimScenarioSet":
        keys = {(item.book_equity, item.diluted_shares) for item in (self.downside, self.base, self.upside)}
        if len(keys) != 1:
            raise ValueError("RIM scenarios must share PIT book equity and diluted shares")
        return self


class CommonRimEngine:
    def value(self, assumptions: RimScenarioSet) -> ValuationResult:
        engine = RimEngine()
        downside = engine.value(assumptions.downside)
        base = engine.value(assumptions.base)
        upside = engine.value(assumptions.upside)
        return ValuationResult(
            method=ValuationMethod.RIM,
            applicability=eligible(
                ValuationMethod.RIM,
                ["book_equity", "roe_path", "cost_of_equity", "payout_ratio", "diluted_shares"],
            ),
            equity_value=base.equity_value,
            fair_value_per_share=base.fair_value_per_share,
            downside_value_per_share=downside.fair_value_per_share,
            base_value_per_share=base.fair_value_per_share,
            upside_value_per_share=upside.fair_value_per_share,
            assumption_confidence=assumptions.base.assumption_confidence,
            provenance=assumptions.base.provenance,
            metadata={"terminal_residual_income": str(base.terminal_residual_income)},
        )
