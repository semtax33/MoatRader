from __future__ import annotations

from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class DcfAssumptions(ContractModel):
    base_revenue: Decimal = Field(gt=0)
    revenue_growth: list[Decimal] = Field(min_length=1, max_length=20)
    ebit_margin: list[Decimal] = Field(min_length=1, max_length=20)
    tax_rate: Decimal = Field(ge=0, le=1)
    depreciation_pct_revenue: Decimal = Field(ge=0)
    capex_pct_revenue: Decimal = Field(ge=0)
    nwc_pct_revenue: Decimal = Field(ge=0)
    wacc: Decimal = Field(gt=0, lt=1)
    terminal_growth: Decimal = Field(ge=-0.1, lt=0.2)
    net_debt: Decimal = Decimal(0)
    diluted_shares: Decimal = Field(gt=0)

    @model_validator(mode="after")
    def forecast_shape(self) -> "DcfAssumptions":
        if len(self.revenue_growth) != len(self.ebit_margin):
            raise ValueError("revenue_growth and ebit_margin must have equal forecast lengths")
        if self.wacc <= self.terminal_growth:
            raise ValueError("WACC must exceed terminal growth")
        return self


class DcfProjection(ContractModel):
    year: int
    revenue: Decimal
    ebit: Decimal
    nopat: Decimal
    unlevered_fcf: Decimal
    discount_factor: Decimal
    present_value: Decimal


class DcfValuation(ContractModel):
    projections: list[DcfProjection]
    terminal_value: Decimal
    terminal_present_value: Decimal
    enterprise_value: Decimal
    equity_value: Decimal
    fair_value_per_share: Decimal


class DcfEngine:
    """Transparent unlevered DCF; assumptions remain separate from LLM reasoning."""

    def value(self, assumptions: DcfAssumptions) -> DcfValuation:
        revenue = assumptions.base_revenue
        previous_nwc = revenue * assumptions.nwc_pct_revenue
        projections: list[DcfProjection] = []
        for year, (growth, margin) in enumerate(
            zip(assumptions.revenue_growth, assumptions.ebit_margin, strict=True),
            start=1,
        ):
            revenue *= Decimal(1) + growth
            ebit = revenue * margin
            nopat = ebit * (Decimal(1) - assumptions.tax_rate)
            depreciation = revenue * assumptions.depreciation_pct_revenue
            capex = revenue * assumptions.capex_pct_revenue
            nwc = revenue * assumptions.nwc_pct_revenue
            delta_nwc = nwc - previous_nwc
            previous_nwc = nwc
            fcf = nopat + depreciation - capex - delta_nwc
            discount_factor = Decimal(1) / ((Decimal(1) + assumptions.wacc) ** year)
            projections.append(
                DcfProjection(
                    year=year,
                    revenue=revenue,
                    ebit=ebit,
                    nopat=nopat,
                    unlevered_fcf=fcf,
                    discount_factor=discount_factor,
                    present_value=fcf * discount_factor,
                )
            )
        last_fcf = projections[-1].unlevered_fcf
        terminal = last_fcf * (Decimal(1) + assumptions.terminal_growth) / (
            assumptions.wacc - assumptions.terminal_growth
        )
        terminal_pv = terminal * projections[-1].discount_factor
        enterprise = sum((item.present_value for item in projections), Decimal(0)) + terminal_pv
        equity = enterprise - assumptions.net_debt
        return DcfValuation(
            projections=projections,
            terminal_value=terminal,
            terminal_present_value=terminal_pv,
            enterprise_value=enterprise,
            equity_value=equity,
            fair_value_per_share=equity / assumptions.diluted_shares,
        )

