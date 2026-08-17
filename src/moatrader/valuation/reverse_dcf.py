from __future__ import annotations

import itertools
from datetime import datetime
from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from moatrader.valuation.economic_dcf import EconomicDcfEngine


class MarketPriceInput(ContractModel):
    current_price: Decimal = Field(gt=0)
    price_as_of: datetime
    evidence_cutoff: datetime

    @model_validator(mode="after")
    def pit_safe(self) -> "MarketPriceInput":
        for name in ("price_as_of", "evidence_cutoff"):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must be timezone-aware")
        if self.evidence_cutoff > self.price_as_of:
            raise ValueError("reverse DCF cannot use evidence published after the market price")
        return self


class ReverseDcfGrid(ContractModel):
    revenue_growth: list[Decimal] = Field(min_length=1, max_length=30)
    target_nopat_margin: list[Decimal] = Field(min_length=1, max_length=30)
    roiic: list[Decimal] = Field(min_length=1, max_length=30)
    cap_years: list[int] = Field(min_length=1, max_length=30)
    price_tolerance: Decimal = Field(default=Decimal("0.03"), gt=0, le=1)
    nearest_points_when_no_solution: int = Field(default=12, ge=1, le=100)

    @model_validator(mode="after")
    def bounded_surface(self) -> "ReverseDcfGrid":
        point_count = (
            len(self.revenue_growth)
            * len(self.target_nopat_margin)
            * len(self.roiic)
            * len(self.cap_years)
        )
        if point_count > 100_000:
            raise ValueError("reverse DCF surface is limited to 100,000 deterministic points")
        if any(value <= -1 or value > 3 for value in self.revenue_growth):
            raise ValueError("revenue-growth grid values must be inside (-100%, 300%]")
        if any(not -1 < value < 1 for value in self.target_nopat_margin):
            raise ValueError("target-margin grid values must be inside (-1, 1)")
        if any(value <= 0 or value > 5 for value in self.roiic):
            raise ValueError("ROIIC grid values must be inside (0, 500%]")
        if any(value < 0 or value > 50 for value in self.cap_years):
            raise ValueError("CAP grid values must be between 0 and 50")
        return self


class ImpliedExpectationPoint(ContractModel):
    revenue_growth: Decimal
    target_nopat_margin: Decimal
    roiic: Decimal
    cap_years: int
    modeled_price: Decimal
    relative_price_error: Decimal
    within_tolerance: bool
    model_screening_eligible: bool
    model_exclusion_reasons: list[str] = Field(default_factory=list)


class ImpliedDriverRange(ContractModel):
    low: Decimal
    high: Decimal

    @model_validator(mode="after")
    def ordered(self) -> "ImpliedDriverRange":
        if self.high < self.low:
            raise ValueError("implied range must be ordered")
        return self


class ImpliedExpectationSurface(ContractModel):
    market: MarketPriceInput
    point_count: int = Field(ge=1)
    eligible_point_count: int = Field(ge=1)
    solution_count: int = Field(ge=0)
    tolerance: Decimal
    solution_points: list[ImpliedExpectationPoint] = Field(default_factory=list)
    representative_points: list[ImpliedExpectationPoint] = Field(min_length=1)
    implied_revenue_growth: ImpliedDriverRange
    implied_target_nopat_margin: ImpliedDriverRange
    implied_roiic: ImpliedDriverRange
    implied_cap_years: ImpliedDriverRange
    multiple_solutions_required: bool = True

    @model_validator(mode="after")
    def preserve_surface_not_false_single_solution(self) -> "ImpliedExpectationSurface":
        if not self.multiple_solutions_required:
            raise ValueError("reverse DCF must expose a surface/range, not a single implied future")
        if self.solution_count != len(self.solution_points):
            raise ValueError("solution_count must match solution_points")
        if not self.solution_count <= self.eligible_point_count <= self.point_count:
            raise ValueError("reverse DCF point counts must satisfy solution <= eligible <= total")
        if any(not item.model_screening_eligible for item in self.representative_points):
            raise ValueError("representative reverse-DCF points must pass model screening")
        return self


class ReverseDcfEngine:
    def __init__(self, engine: EconomicDcfEngine | None = None) -> None:
        self.engine = engine or EconomicDcfEngine()

    def surface(
        self,
        *,
        base_assumptions: EconomicDcfAssumptions,
        market: MarketPriceInput,
        grid: ReverseDcfGrid,
    ) -> ImpliedExpectationSurface:
        points: list[ImpliedExpectationPoint] = []
        for growth, margin, roiic, cap_years in itertools.product(
            grid.revenue_growth,
            grid.target_nopat_margin,
            grid.roiic,
            grid.cap_years,
        ):
            payload = base_assumptions.model_dump(mode="python")
            payload.update(
                {
                    "scenario": "UNSPECIFIED",
                    "revenue_growth": growth,
                    "target_nopat_margin": margin,
                    "roiic": roiic,
                    "competitive_advantage_period_years": cap_years,
                    "explicit_forecast_years": max(
                        base_assumptions.explicit_forecast_years,
                        cap_years + base_assumptions.fade_years,
                    ),
                }
            )
            assumptions = EconomicDcfAssumptions.model_validate(payload)
            valuation = self.engine.value(assumptions)
            modeled = valuation.fair_value_per_share
            error = (modeled - market.current_price) / market.current_price
            points.append(
                ImpliedExpectationPoint(
                    revenue_growth=growth,
                    target_nopat_margin=margin,
                    roiic=roiic,
                    cap_years=cap_years,
                    modeled_price=modeled,
                    relative_price_error=error,
                    within_tolerance=(
                        valuation.screening_eligible
                        and abs(error) <= grid.price_tolerance
                    ),
                    model_screening_eligible=valuation.screening_eligible,
                    model_exclusion_reasons=valuation.screening_exclusion_reasons,
                )
            )
        points.sort(
            key=lambda item: (
                abs(item.relative_price_error),
                item.cap_years,
                item.revenue_growth,
                item.target_nopat_margin,
                item.roiic,
            )
        )
        eligible_points = [item for item in points if item.model_screening_eligible]
        if not eligible_points:
            raise ValueError("reverse DCF grid produced no screening-eligible valuation points")
        solutions = [item for item in eligible_points if item.within_tolerance]
        representatives = solutions or eligible_points[: grid.nearest_points_when_no_solution]

        def value_range(name: str) -> ImpliedDriverRange:
            values = [Decimal(str(getattr(item, name))) for item in representatives]
            return ImpliedDriverRange(low=min(values), high=max(values))

        return ImpliedExpectationSurface(
            market=market,
            point_count=len(points),
            eligible_point_count=len(eligible_points),
            solution_count=len(solutions),
            tolerance=grid.price_tolerance,
            solution_points=solutions,
            representative_points=representatives,
            implied_revenue_growth=value_range("revenue_growth"),
            implied_target_nopat_margin=value_range("target_nopat_margin"),
            implied_roiic=value_range("roiic"),
            implied_cap_years=value_range("cap_years"),
        )
