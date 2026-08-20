from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Literal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.base import ValuationMethod, ValuationResult, eligible


RIM_POLICY_VERSION = "rim-policy/1"
RIM_COST_OF_EQUITY_BY_SIZE: dict[str, Decimal] = {
    "LARGE": Decimal("0.09"),
    "MID": Decimal("0.10"),
    "SMALL": Decimal("0.12"),
}


class RimBuildInput(ContractModel):
    """PIT accounting inputs used to build a deterministic RIM scenario set."""

    policy_version: Literal["rim-policy/1"] = RIM_POLICY_VERSION
    issuer_id: str = Field(min_length=1)
    as_of: str = Field(min_length=10)
    book_equity: Decimal = Field(gt=0)
    prior_fy_net_income: Decimal
    current_ytd_net_income: Decimal
    prior_ytd_net_income: Decimal
    diluted_shares: Decimal = Field(gt=0)
    size_bucket: Literal["LARGE", "MID", "SMALL"]
    evidence_available_at: dict[str, date] = Field(min_length=2)
    provenance: list[str] = Field(min_length=2)

    @property
    def ttm_net_income(self) -> Decimal:
        return (
            self.prior_fy_net_income
            + self.current_ytd_net_income
            - self.prior_ytd_net_income
        )

    @model_validator(mode="after")
    def evidence_is_pit(self) -> "RimBuildInput":
        cutoff = date.fromisoformat(self.as_of[:10])
        future = sorted(
            ref
            for ref, available_at in self.evidence_available_at.items()
            if available_at > cutoff
        )
        if future:
            raise ValueError(f"RIM evidence is future-dated: {future}")
        evidence_refs = set(self.evidence_available_at)
        if not evidence_refs.issubset(set(self.provenance)):
            raise ValueError("RIM provenance must include every dated evidence ref")
        roe = self.ttm_net_income / self.book_equity
        if abs(roe) > Decimal("0.50"):
            raise ValueError("RIM_ROE_OUTSIDE_ENGINEERING_BOUND")
        return self


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


class RimBuilder:
    """Build RIM without LLM inference or outcome-conditioned parameters."""

    def build(self, source: RimBuildInput) -> RimScenarioSet:
        cost = RIM_COST_OF_EQUITY_BY_SIZE[source.size_bucket]
        central_roe = source.ttm_net_income / source.book_equity
        cases: list[RimAssumptions] = []
        for shift in (Decimal("-0.03"), Decimal(0), Decimal("0.03")):
            start_roe = central_roe + shift
            terminal_roe = cost + (start_roe - cost) * Decimal("0.25")
            roe_path = [
                start_roe
                + (terminal_roe - start_roe) * Decimal(year) / Decimal(5)
                for year in range(1, 6)
            ]
            cases.append(
                RimAssumptions(
                    book_equity=source.book_equity,
                    roe_path=roe_path,
                    cost_of_equity=cost,
                    payout_ratio=Decimal("0.40"),
                    terminal_roe=terminal_roe,
                    terminal_growth=Decimal("0.02"),
                    diluted_shares=source.diluted_shares,
                    assumption_confidence=Decimal("0.60"),
                    provenance=list(
                        dict.fromkeys(
                            source.provenance
                            + [RIM_POLICY_VERSION, "NO_LLM:DETERMINISTIC_BUILDER"]
                        )
                    ),
                )
            )
        return RimScenarioSet(
            downside=cases[0],
            base=cases[1],
            upside=cases[2],
        )
