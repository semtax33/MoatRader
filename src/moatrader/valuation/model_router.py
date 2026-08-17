from pydantic import Field

from moatrader.business.lifecycle import CompanyType, LifeCycleStage
from moatrader.canonical.models import ContractModel
from moatrader.valuation.base import ValuationMethod


class ValuationModelRoute(ContractModel):
    company_type: CompanyType
    life_cycle_stage: LifeCycleStage
    method: ValuationMethod
    implemented: bool
    rationale: list[str] = Field(default_factory=list)


class ValuationModelRouter:
    """Pre-declared economic-structure routing; never selected after seeing price."""

    def route(
        self,
        *,
        company_type: CompanyType,
        life_cycle_stage: LifeCycleStage,
    ) -> ValuationModelRoute:
        if company_type == CompanyType.FINANCIAL:
            return ValuationModelRoute(
                company_type=company_type,
                life_cycle_stage=life_cycle_stage,
                method=ValuationMethod.EXCESS_RETURN_EQUITY,
                implemented=False,
                rationale=["Bank/insurance cash flows and leverage require an equity excess-return model."],
            )
        if company_type == CompanyType.PRE_REVENUE_BIOTECH:
            return ValuationModelRoute(
                company_type=company_type,
                life_cycle_stage=life_cycle_stage,
                method=ValuationMethod.BIOTECH_RNPV,
                implemented=True,
                rationale=["Pre-revenue pipeline assets require probability- and timing-adjusted rNPV."],
            )
        if company_type == CompanyType.DISTRESSED:
            return ValuationModelRoute(
                company_type=company_type,
                life_cycle_stage=life_cycle_stage,
                method=ValuationMethod.FAILURE_ADJUSTED_DCF,
                implemented=True,
                rationale=["Distress probability and recovery value must be explicit."],
            )
        if company_type == CompanyType.HIGH_GROWTH_PLATFORM:
            return ValuationModelRoute(
                company_type=company_type,
                life_cycle_stage=life_cycle_stage,
                method=ValuationMethod.NARRATIVE_DCF,
                implemented=True,
                rationale=["Narrative drivers are constrained by economic FCFF and reinvestment."],
            )
        if company_type == CompanyType.MATURE_NON_FINANCIAL:
            return ValuationModelRoute(
                company_type=company_type,
                life_cycle_stage=life_cycle_stage,
                method=ValuationMethod.STEADY_STATE_FCFF,
                implemented=True,
                rationale=["Mature economics emphasize fade and steady-state value."],
            )
        return ValuationModelRoute(
            company_type=company_type,
            life_cycle_stage=life_cycle_stage,
            method=ValuationMethod.ECONOMIC_FCFF,
            implemented=True,
            rationale=["General non-financial company uses growth-reinvestment-ROIIC economic FCFF."],
        )
