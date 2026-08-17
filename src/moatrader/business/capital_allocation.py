from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class CapitalPeriod(ContractModel):
    period: str = Field(min_length=1)
    revenue: Decimal = Field(gt=0)
    reported_nopat: Decimal
    reported_invested_capital: Decimal = Field(gt=0)
    reinvestment: Decimal
    rd_expense: Decimal = Field(default=Decimal(0), ge=0)
    sga_expense: Decimal = Field(default=Decimal(0), ge=0)
    acquisition_spend: Decimal = Field(default=Decimal(0), ge=0)
    buybacks: Decimal = Field(default=Decimal(0), ge=0)
    dividends: Decimal = Field(default=Decimal(0), ge=0)


class IntangibleAdjustmentPolicy(ContractModel):
    useful_life_years: int = Field(default=5, ge=1, le=20)
    sga_capitalization_pct: Decimal = Field(default=Decimal(0), ge=0, le=1)
    tax_rate: Decimal = Field(default=Decimal("0.25"), ge=0, le=1)


class CapitalPeriodAnalysis(ContractModel):
    period: str
    reported_roic: Decimal
    adjusted_economic_roic: Decimal
    reported_roiic: Decimal | None = None
    adjusted_roiic: Decimal | None = None
    reinvestment_rate: Decimal | None = None
    adjusted_reinvestment: Decimal
    adjusted_reinvestment_rate: Decimal | None = None
    intangible_investment: Decimal
    intangible_amortization: Decimal
    unamortized_intangible_asset: Decimal
    adjusted_nopat: Decimal
    adjusted_invested_capital: Decimal


class CapitalAllocationProfile(ContractModel):
    periods: list[CapitalPeriodAnalysis] = Field(min_length=1)
    median_reported_roic: Decimal
    median_adjusted_economic_roic: Decimal
    latest_reported_roiic: Decimal | None = None
    latest_adjusted_roiic: Decimal | None = None
    cumulative_acquisition_spend: Decimal = Decimal(0)
    cumulative_buybacks: Decimal = Decimal(0)
    cumulative_dividends: Decimal = Decimal(0)
    price_inputs_used: bool = False

    @model_validator(mode="after")
    def price_blind(self) -> "CapitalAllocationProfile":
        if self.price_inputs_used:
            raise ValueError("capital allocation analysis must be price-blind")
        return self


def _median(values: list[Decimal]) -> Decimal:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


class CapitalAllocationAnalyzer:
    """Keeps reported and intangible-adjusted economics side by side."""

    def analyze(
        self,
        periods: list[CapitalPeriod],
        policy: IntangibleAdjustmentPolicy | None = None,
    ) -> CapitalAllocationProfile:
        if not periods:
            raise ValueError("at least one capital period is required")
        policy = policy or IntangibleAdjustmentPolicy()
        vintages: list[tuple[Decimal, int]] = []
        analyses: list[CapitalPeriodAnalysis] = []
        previous: CapitalPeriod | None = None
        previous_adjusted_nopat: Decimal | None = None

        for item in periods:
            intangible_investment = item.rd_expense + item.sga_expense * policy.sga_capitalization_pct
            amortization = sum(
                amount / Decimal(policy.useful_life_years)
                for amount, age in vintages
                if age < policy.useful_life_years
            )
            aged = [(amount, age + 1) for amount, age in vintages if age + 1 < policy.useful_life_years]
            aged.append((intangible_investment, 0))
            vintages = aged
            unamortized = sum(
                amount * Decimal(policy.useful_life_years - age) / Decimal(policy.useful_life_years)
                for amount, age in vintages
            )
            tax_factor = Decimal(1) - policy.tax_rate
            adjusted_nopat = item.reported_nopat + (intangible_investment - amortization) * tax_factor
            adjusted_capital = item.reported_invested_capital + unamortized
            adjusted_reinvestment = item.reinvestment + intangible_investment - amortization
            reported_roic = item.reported_nopat / item.reported_invested_capital
            adjusted_roic = adjusted_nopat / adjusted_capital
            reported_roiic: Decimal | None = None
            adjusted_roiic: Decimal | None = None
            if previous is not None and item.reinvestment != 0:
                reported_roiic = (item.reported_nopat - previous.reported_nopat) / item.reinvestment
                if previous_adjusted_nopat is not None and adjusted_reinvestment != 0:
                    adjusted_roiic = (
                        adjusted_nopat - previous_adjusted_nopat
                    ) / adjusted_reinvestment
            reinvestment_rate = (
                item.reinvestment / item.reported_nopat if item.reported_nopat != 0 else None
            )
            adjusted_reinvestment_rate = (
                adjusted_reinvestment / adjusted_nopat if adjusted_nopat != 0 else None
            )
            analyses.append(
                CapitalPeriodAnalysis(
                    period=item.period,
                    reported_roic=reported_roic,
                    adjusted_economic_roic=adjusted_roic,
                    reported_roiic=reported_roiic,
                    adjusted_roiic=adjusted_roiic,
                    reinvestment_rate=reinvestment_rate,
                    adjusted_reinvestment=adjusted_reinvestment,
                    adjusted_reinvestment_rate=adjusted_reinvestment_rate,
                    intangible_investment=intangible_investment,
                    intangible_amortization=amortization,
                    unamortized_intangible_asset=unamortized,
                    adjusted_nopat=adjusted_nopat,
                    adjusted_invested_capital=adjusted_capital,
                )
            )
            previous = item
            previous_adjusted_nopat = adjusted_nopat

        return CapitalAllocationProfile(
            periods=analyses,
            median_reported_roic=_median([item.reported_roic for item in analyses]),
            median_adjusted_economic_roic=_median(
                [item.adjusted_economic_roic for item in analyses]
            ),
            latest_reported_roiic=analyses[-1].reported_roiic,
            latest_adjusted_roiic=analyses[-1].adjusted_roiic,
            cumulative_acquisition_spend=sum(
                (item.acquisition_spend for item in periods), Decimal(0)
            ),
            cumulative_buybacks=sum((item.buybacks for item in periods), Decimal(0)),
            cumulative_dividends=sum((item.dividends for item in periods), Decimal(0)),
        )
