from __future__ import annotations

from decimal import Decimal
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from moatrader.business.drivers import ValuationDriver
from moatrader.canonical.models import ContractModel
from moatrader.financial.dcf import DcfAssumptionType


class ReinvestmentMethod(StrEnum):
    ROIIC = "ROIIC"
    SALES_TO_CAPITAL = "SALES_TO_CAPITAL"


ECONOMIC_DCF_VALUE_FIELDS = (
    "base_revenue",
    "base_nopat_margin",
    "base_invested_capital",
    "revenue_growth",
    "target_nopat_margin",
    "roiic",
    "competitive_advantage_period_years",
    "fade_years",
    "stable_growth",
    "stable_nopat_margin",
    "stable_roic",
    "wacc",
    "net_debt",
    "diluted_shares",
    "failure_probability",
)


class EconomicDcfAssumptions(ContractModel):
    """Price-blind economic DCF assumptions.

    Growth, reinvestment, ROIIC and competitive-advantage fade are explicit.
    ``current_price`` is intentionally absent and extra fields are forbidden by
    ContractModel, making accidental price anchoring fail closed.
    """

    method: Literal["ECONOMIC_FCFF"] = "ECONOMIC_FCFF"
    scenario: Literal["DOWNSIDE", "CENTRAL", "UPSIDE", "UNSPECIFIED"] = "UNSPECIFIED"
    base_period: str | None = None
    base_revenue: Decimal = Field(gt=0)
    base_nopat_margin: Decimal = Field(gt=-1, lt=1)
    base_invested_capital: Decimal = Field(gt=0)
    revenue_growth: Decimal = Field(gt=-1, le=3)
    target_nopat_margin: Decimal = Field(gt=-1, lt=1)
    margin_convergence_years: int = Field(default=3, ge=1, le=30)
    roiic: Decimal = Field(gt=0, le=5)
    reinvestment_method: ReinvestmentMethod = ReinvestmentMethod.ROIIC
    sales_to_capital: Decimal | None = Field(default=None, gt=0, le=100)
    competitive_advantage_period_years: int = Field(ge=0, le=50)
    fade_years: int = Field(default=5, ge=0, le=30)
    explicit_forecast_years: int = Field(default=10, ge=1, le=60)
    stable_growth: Decimal = Field(ge=-0.10, lt=0.20)
    stable_nopat_margin: Decimal = Field(gt=-1, lt=1)
    stable_roic: Decimal = Field(gt=0, le=2)
    wacc: Decimal = Field(gt=0, lt=1)
    net_debt: Decimal = Decimal(0)
    diluted_shares: Decimal = Field(gt=0)
    failure_probability: Decimal = Field(default=Decimal(0), ge=0, lt=1)
    distress_recovery_enterprise_value: Decimal = Field(default=Decimal(0), ge=0)
    driver_evidence_ids: dict[ValuationDriver, list[str]] = Field(default_factory=dict)
    assumption_sources: dict[str, list[str]] = Field(default_factory=dict)
    assumption_types: dict[str, DcfAssumptionType] = Field(default_factory=dict)
    provenance_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def economically_consistent(self) -> "EconomicDcfAssumptions":
        if self.wacc <= self.stable_growth:
            raise ValueError("WACC must exceed stable growth")
        if self.stable_roic <= self.stable_growth:
            raise ValueError("stable ROIC must exceed stable growth")
        if self.explicit_forecast_years < self.competitive_advantage_period_years + self.fade_years:
            raise ValueError("explicit forecast must cover the complete CAP and fade periods")
        if self.reinvestment_method == ReinvestmentMethod.SALES_TO_CAPITAL:
            if self.sales_to_capital is None:
                raise ValueError("sales_to_capital is required for SALES_TO_CAPITAL reinvestment")
        elif self.sales_to_capital is not None:
            raise ValueError("sales_to_capital is only valid with SALES_TO_CAPITAL reinvestment")
        allowed = set(ECONOMIC_DCF_VALUE_FIELDS)
        invalid = (set(self.assumption_sources) | set(self.assumption_types)) - allowed
        if invalid:
            raise ValueError(f"unknown economic DCF provenance fields: {sorted(invalid)}")
        for driver, evidence_ids in self.driver_evidence_ids.items():
            if len(evidence_ids) != len(set(evidence_ids)):
                raise ValueError(f"duplicate evidence IDs for {driver.value}")
        applied: dict[str, ValuationDriver] = {}
        for driver, evidence_ids in self.driver_evidence_ids.items():
            for evidence_id in evidence_ids:
                previous = applied.setdefault(evidence_id, driver)
                if previous != driver:
                    raise ValueError(
                        f"evidence {evidence_id} is applied to both "
                        f"{previous.value} and {driver.value}; use related drivers for diagnostics only"
                    )
        return self

    @property
    def base_nopat(self) -> Decimal:
        return self.base_revenue * self.base_nopat_margin

    def type_for(self, field: str) -> DcfAssumptionType:
        return self.assumption_types.get(field, DcfAssumptionType.UNSPECIFIED)
