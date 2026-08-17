from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.business.capital_allocation import (
    CapitalAllocationAnalyzer,
    CapitalAllocationProfile,
    CapitalPeriod,
    IntangibleAdjustmentPolicy,
)
from moatrader.business.competitive_advantage import (
    CapAssessment,
    CapEngine,
    CapPrior,
    CompetitiveAdvantageProfile,
)
from moatrader.business.drivers import (
    ValuationDriverEvidenceBundle,
    ValuationDriverMapper,
)
from moatrader.business.lifecycle import CompanyType, LifeCycleStage
from moatrader.canonical.models import ContractModel
from moatrader.evidence.models import EvidenceCard
from moatrader.expectations.gap import ExpectationGapEvaluation, ExpectationGapEvaluator
from moatrader.valuation.model_router import (
    ValuationMethod,
    ValuationModelRoute,
    ValuationModelRouter,
)
from moatrader.valuation.reference_class import PlausibilityReferenceClass
from moatrader.valuation.reverse_dcf import (
    ImpliedExpectationSurface,
    MarketPriceInput,
    ReverseDcfEngine,
    ReverseDcfGrid,
)
from moatrader.valuation.scenarios import (
    IntrinsicScenarioSet,
    IntrinsicValuationRange,
    ScenarioValuationEngine,
)
from moatrader.valuation.three_p import (
    PlausibilityStatus,
    PossibleContext,
    ThreePEngine,
    ThreePResult,
)


EXPECTATION_ANALYSIS_SCHEMA_VERSION = "expectation-analysis/1"


class ExpectationAnalysisRequest(ContractModel):
    """Price-blind research input loaded before the market lane exists."""

    schema_version: str = "expectation-analysis-request/1"
    evidence_cutoff: datetime
    company_type: CompanyType = CompanyType.GENERAL_NON_FINANCIAL
    life_cycle_stage: LifeCycleStage = LifeCycleStage.MATURE_GROWTH
    intrinsic_scenarios: IntrinsicScenarioSet
    possible_context: PossibleContext = Field(default_factory=PossibleContext)
    reference_class: PlausibilityReferenceClass | None = None
    cap_prior: CapPrior
    reverse_grid: ReverseDcfGrid
    capital_periods: list[CapitalPeriod] = Field(default_factory=list)
    intangible_adjustment_policy: IntangibleAdjustmentPolicy = Field(
        default_factory=IntangibleAdjustmentPolicy
    )

    @model_validator(mode="after")
    def cutoff_is_timezone_aware(self) -> "ExpectationAnalysisRequest":
        if self.evidence_cutoff.tzinfo is None or self.evidence_cutoff.utcoffset() is None:
            raise ValueError("expectation evidence_cutoff must be timezone-aware")
        return self


class ExpectationAnalysis(ContractModel):
    schema_version: str = EXPECTATION_ANALYSIS_SCHEMA_VERSION
    issuer_id: str
    as_of: datetime
    model_route: ValuationModelRoute
    driver_evidence: ValuationDriverEvidenceBundle
    competitive_advantage_profile: CompetitiveAdvantageProfile
    cap_assessment: CapAssessment
    capital_allocation_profile: CapitalAllocationProfile | None = None
    intrinsic_valuation: IntrinsicValuationRange
    three_p: dict[str, ThreePResult]
    implied_expectations: ImpliedExpectationSurface
    expectation_gap: ExpectationGapEvaluation
    analysis_warnings: list[str] = Field(default_factory=list)
    old_moat_score_used_for_valuation: bool = False
    price_lane_started_after_intrinsic_lane: bool = True

    @model_validator(mode="after")
    def architecture_isolated(self) -> "ExpectationAnalysis":
        if self.old_moat_score_used_for_valuation:
            raise ValueError("the deprecated scalar MOAT score must not drive valuation")
        if not self.price_lane_started_after_intrinsic_lane:
            raise ValueError("market price may enter only after intrinsic valuation and 3P")
        return self


class ExpectationAnalysisEngine:
    def __init__(self) -> None:
        self.mapper = ValuationDriverMapper()
        self.cap_engine = CapEngine()
        self.capital_allocation_analyzer = CapitalAllocationAnalyzer()
        self.scenario_engine = ScenarioValuationEngine()
        self.three_p_engine = ThreePEngine()
        self.reverse_engine = ReverseDcfEngine()
        self.gap_engine = ExpectationGapEvaluator()
        self.router = ValuationModelRouter()

    def analyze(
        self,
        *,
        issuer_id: str,
        as_of: datetime,
        cards: list[EvidenceCard],
        request: ExpectationAnalysisRequest,
        current_price: Decimal,
        price_as_of: datetime,
        driver_evidence: ValuationDriverEvidenceBundle | None = None,
    ) -> ExpectationAnalysis:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        if request.evidence_cutoff != as_of:
            raise ValueError(
                "expectation request evidence_cutoff must exactly match the run as_of"
            )
        if request.reference_class and request.reference_class.as_of > as_of:
            raise ValueError("reference-class data is after the evidence cutoff")
        if request.cap_prior.as_of > as_of:
            raise ValueError("CAP prior data is after the evidence cutoff")
        route = self.router.route(
            company_type=request.company_type,
            life_cycle_stage=request.life_cycle_stage,
        )
        supported_economic_routes = {
            ValuationMethod.ECONOMIC_FCFF,
            ValuationMethod.STEADY_STATE_FCFF,
            ValuationMethod.NARRATIVE_DCF,
            ValuationMethod.FAILURE_ADJUSTED_DCF,
        }
        if not route.implemented or route.method not in supported_economic_routes:
            raise ValueError(
                f"expectation pipeline does not support routed method {route.method.value}; "
                "use the dedicated valuation engine"
            )

        # Intrinsic lane: no price object exists before all steps below finish.
        evidence = driver_evidence or self.mapper.map_cards(
            issuer_id=issuer_id,
            as_of=as_of,
            cards=cards,
        )
        if evidence.issuer_id != issuer_id or evidence.as_of != as_of:
            raise ValueError("precomputed valuation evidence has a different issuer or PIT cutoff")
        self._validate_assumption_evidence_links(request, evidence)
        profile = CompetitiveAdvantageProfile.from_driver_evidence(evidence)
        cap = self.cap_engine.assess(profile, request.cap_prior)
        capital_allocation = (
            self.capital_allocation_analyzer.analyze(
                request.capital_periods,
                request.intangible_adjustment_policy,
            )
            if request.capital_periods
            else None
        )
        intrinsic = self.scenario_engine.value(request.intrinsic_scenarios)
        three_p = {
            name: self.three_p_engine.evaluate(
                assumptions,
                evidence=evidence,
                possible_context=request.possible_context,
                reference_class=request.reference_class,
                capital_allocation=capital_allocation,
            )
            for name, assumptions in (
                ("downside", request.intrinsic_scenarios.downside),
                ("central", request.intrinsic_scenarios.central),
                ("upside", request.intrinsic_scenarios.upside),
            )
        }

        # Market lane begins only here. No price-derived value is fed back into
        # assumptions, CAP, evidence routing, scenarios, or 3P.
        market = MarketPriceInput(
            current_price=current_price,
            price_as_of=price_as_of,
            evidence_cutoff=as_of,
        )
        implied = self.reverse_engine.surface(
            base_assumptions=request.intrinsic_scenarios.central,
            market=market,
            grid=request.reverse_grid,
        )
        gap = self.gap_engine.evaluate(
            scenarios=request.intrinsic_scenarios,
            intrinsic=intrinsic,
            cap=cap,
            reverse=implied,
            central_three_p=three_p["central"],
        )
        warnings: list[str] = []
        central_cap = request.intrinsic_scenarios.central.competitive_advantage_period_years
        if not cap.low_years <= central_cap <= cap.high_years:
            warnings.append(
                f"central CAP {central_cap}y is outside evidence/reference range "
                f"{cap.low_years}-{cap.high_years}y"
            )
        historical_roiic_check = next(
            (
                item
                for item in three_p["central"].plausible_checks
                if item.name == "COMPANY_HISTORICAL_ROIIC"
            ),
            None,
        )
        if (
            historical_roiic_check is not None
            and historical_roiic_check.status == PlausibilityStatus.OUTLIER
        ):
            warnings.append(
                "central ROIIC is outside the PIT company historical economic ROIIC range"
            )
        if implied.solution_count == 0:
            warnings.append("reverse DCF grid found no point inside price tolerance; nearest points shown")
        return ExpectationAnalysis(
            issuer_id=issuer_id,
            as_of=as_of,
            model_route=route,
            driver_evidence=evidence,
            competitive_advantage_profile=profile,
            cap_assessment=cap,
            capital_allocation_profile=capital_allocation,
            intrinsic_valuation=intrinsic,
            three_p=three_p,
            implied_expectations=implied,
            expectation_gap=gap,
            analysis_warnings=warnings,
        )

    @staticmethod
    def _validate_assumption_evidence_links(
        request: ExpectationAnalysisRequest,
        evidence: ValuationDriverEvidenceBundle,
    ) -> None:
        by_id = {item.evidence_id: item for item in evidence.evidence}
        for scenario_name, assumptions in (
            ("downside", request.intrinsic_scenarios.downside),
            ("central", request.intrinsic_scenarios.central),
            ("upside", request.intrinsic_scenarios.upside),
        ):
            for driver, evidence_ids in assumptions.driver_evidence_ids.items():
                for evidence_id in evidence_ids:
                    item = by_id.get(evidence_id)
                    if item is None:
                        raise ValueError(
                            f"{scenario_name} references unknown valuation evidence {evidence_id}"
                        )
                    if item.primary_driver != driver:
                        raise ValueError(
                            f"{scenario_name} applies {evidence_id} to {driver.value}, but its "
                            f"exclusive primary driver is {item.primary_driver.value}"
                        )
