from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class DcfAssumptionType(StrEnum):
    DISCLOSED_FACT = "DISCLOSED_FACT"
    DETERMINISTIC = "DETERMINISTIC"
    MODEL_INFERENCE = "MODEL_INFERENCE"
    MANAGEMENT_GUIDANCE = "MANAGEMENT_GUIDANCE"
    EXTERNAL_FORECAST = "EXTERNAL_FORECAST"
    DEFAULT = "DEFAULT"
    UNSPECIFIED = "UNSPECIFIED"


DCF_VALUE_FIELDS = (
    "base_revenue",
    "revenue_growth",
    "ebit_margin",
    "tax_rate",
    "depreciation_pct_revenue",
    "capex_pct_revenue",
    "nwc_pct_revenue",
    "wacc",
    "terminal_growth",
    "net_debt",
    "diluted_shares",
)


class DcfAssumptions(ContractModel):
    method: Literal["FCFF"] = "FCFF"
    base_period: str | None = None
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
    assumption_sources: dict[str, list[str]] = Field(default_factory=dict)
    assumption_types: dict[str, DcfAssumptionType] = Field(default_factory=dict)
    provenance_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def forecast_shape(self) -> "DcfAssumptions":
        if len(self.revenue_growth) != len(self.ebit_margin):
            raise ValueError("revenue_growth and ebit_margin must have equal forecast lengths")
        if self.wacc <= self.terminal_growth:
            raise ValueError("WACC must exceed terminal growth")
        allowed = set(DCF_VALUE_FIELDS)
        invalid_sources = set(self.assumption_sources) - allowed
        invalid_types = set(self.assumption_types) - allowed
        if invalid_sources or invalid_types:
            raise ValueError(
                "DCF provenance contains unknown assumption fields: "
                f"{sorted(invalid_sources | invalid_types)}"
            )
        return self

    def type_for(self, field: str) -> DcfAssumptionType:
        return self.assumption_types.get(field, DcfAssumptionType.UNSPECIFIED)

    def confidence_score(self) -> Decimal:
        weights = {
            DcfAssumptionType.DISCLOSED_FACT: Decimal("1.00"),
            DcfAssumptionType.DETERMINISTIC: Decimal("0.95"),
            DcfAssumptionType.EXTERNAL_FORECAST: Decimal("0.75"),
            DcfAssumptionType.MANAGEMENT_GUIDANCE: Decimal("0.70"),
            DcfAssumptionType.MODEL_INFERENCE: Decimal("0.55"),
            DcfAssumptionType.DEFAULT: Decimal("0.25"),
            DcfAssumptionType.UNSPECIFIED: Decimal("0.10"),
        }
        return sum((weights[self.type_for(field)] for field in DCF_VALUE_FIELDS), Decimal(0)) / Decimal(
            len(DCF_VALUE_FIELDS)
        )


class DcfProjection(ContractModel):
    year: int
    revenue: Decimal
    ebit: Decimal
    nopat: Decimal
    unlevered_fcf: Decimal
    discount_factor: Decimal
    present_value: Decimal


class DcfValuation(ContractModel):
    method: Literal["FCFF"] = "FCFF"
    base_period: str | None = None
    assumptions: DcfAssumptions
    projections: list[DcfProjection]
    terminal_value: Decimal
    terminal_present_value: Decimal
    enterprise_value: Decimal
    equity_value: Decimal
    fair_value_per_share: Decimal
    terminal_value_share: Decimal
    assumption_confidence: Decimal = Field(ge=0, le=1)
    confidence_penalty: Decimal = Field(ge=0, le=1)
    default_assumptions: list[str] = Field(default_factory=list)
    provenance_warnings: list[str] = Field(default_factory=list)
    screening_eligible: bool = True
    screening_exclusion_reasons: list[str] = Field(default_factory=list)


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
        confidence = assumptions.confidence_score()
        defaults = [
            field
            for field in DCF_VALUE_FIELDS
            if assumptions.type_for(field) == DcfAssumptionType.DEFAULT
        ]
        unspecified = [
            field
            for field in DCF_VALUE_FIELDS
            if assumptions.type_for(field) == DcfAssumptionType.UNSPECIFIED
        ]
        warnings = list(assumptions.provenance_warnings)
        if unspecified:
            warnings.append(f"missing assumption provenance: {', '.join(unspecified)}")
        terminal_share = terminal_pv / enterprise if enterprise else Decimal(0)
        if terminal_share > Decimal("0.70"):
            warnings.append("terminal value exceeds 70% of enterprise value")
        if terminal_share < 0 or terminal_share > 1:
            warnings.append("terminal value share is outside [0, 1] because explicit-period or enterprise value is non-positive")
        exclusion_reasons: list[str] = []
        if enterprise <= 0:
            exclusion_reasons.append("NON_POSITIVE_ENTERPRISE_VALUE")
        if equity <= 0:
            exclusion_reasons.append("NON_POSITIVE_EQUITY_VALUE")
        if last_fcf <= 0:
            exclusion_reasons.append("NON_POSITIVE_TERMINAL_FCF")
        if terminal_share < 0 or terminal_share > Decimal("0.85"):
            exclusion_reasons.append("UNRELIABLE_TERMINAL_VALUE_SHARE")
        if confidence < Decimal("0.50"):
            exclusion_reasons.append("LOW_ASSUMPTION_CONFIDENCE")
        return DcfValuation(
            method=assumptions.method,
            base_period=assumptions.base_period,
            assumptions=assumptions,
            projections=projections,
            terminal_value=terminal,
            terminal_present_value=terminal_pv,
            enterprise_value=enterprise,
            equity_value=equity,
            fair_value_per_share=equity / assumptions.diluted_shares,
            terminal_value_share=terminal_share,
            assumption_confidence=confidence,
            confidence_penalty=Decimal(1) - confidence,
            default_assumptions=defaults,
            provenance_warnings=warnings,
            screening_eligible=not exclusion_reasons,
            screening_exclusion_reasons=exclusion_reasons,
        )
