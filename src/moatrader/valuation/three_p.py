from __future__ import annotations

from decimal import Decimal
from enum import StrEnum

from pydantic import Field, model_validator

from moatrader.business.capital_allocation import CapitalAllocationProfile
from moatrader.business.drivers import (
    ValuationDriver,
    ValuationDriverEvidenceBundle,
    ValuationEvidenceRole,
)
from moatrader.canonical.models import ContractModel
from moatrader.valuation.assumptions import EconomicDcfAssumptions, ReinvestmentMethod
from moatrader.valuation.economic_dcf import EconomicDcfEngine
from moatrader.valuation.reference_class import PlausibilityReferenceClass


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNKNOWN = "UNKNOWN"


class PlausibilityStatus(StrEnum):
    IN_RANGE = "IN_RANGE"
    OUTLIER = "OUTLIER"
    UNKNOWN = "UNKNOWN"


class ProbabilitySupport(StrEnum):
    SUPPORTED = "SUPPORTED"
    MIXED = "MIXED"
    WEAK = "WEAK"
    CONTRADICTED = "CONTRADICTED"


class ThreePVerdict(StrEnum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


class PossibleContext(ContractModel):
    tam_ceiling: Decimal | None = Field(default=None, gt=0)
    gross_margin_ceiling: Decimal | None = Field(default=None, gt=-1, lt=1)
    stable_growth_ceiling: Decimal = Field(default=Decimal("0.05"), ge=-0.10, lt=0.20)
    maximum_revenue_growth: Decimal = Field(default=Decimal("1.00"), ge=0, le=5)
    minimum_positive_reinvestment: Decimal = Field(default=Decimal("0.000001"), ge=0)
    source_refs: list[str] = Field(default_factory=list)


class ThreePCheck(ContractModel):
    name: str
    status: CheckStatus | PlausibilityStatus
    observed: str
    boundary: str
    rationale: str


class DriverProbabilityAssessment(ContractModel):
    driver: ValuationDriver
    status: ProbabilitySupport
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    counterevidence_ids: list[str] = Field(default_factory=list)
    scenario_evidence_ids: list[str] = Field(default_factory=list)
    range_widener_evidence_ids: list[str] = Field(default_factory=list)


class ThreePResult(ContractModel):
    scenario: str
    possible: CheckStatus
    possible_checks: list[ThreePCheck]
    plausible: PlausibilityStatus
    plausible_checks: list[ThreePCheck]
    probable: ProbabilitySupport
    driver_assessments: list[DriverProbabilityAssessment]
    verdict: ThreePVerdict
    price_inputs_used: bool = False
    probability_is_calibrated: bool = False

    @model_validator(mode="after")
    def qualitative_probability_and_price_blind(self) -> "ThreePResult":
        if self.price_inputs_used:
            raise ValueError("3P validation must be price-blind")
        if self.probability_is_calibrated:
            raise ValueError("v1 has no historical calibration for numeric probabilities")
        return self


class ThreePEngine:
    """Possible/Plausible/Probable is an assumption validator, not a DCF formula."""

    def evaluate(
        self,
        assumptions: EconomicDcfAssumptions,
        *,
        evidence: ValuationDriverEvidenceBundle,
        possible_context: PossibleContext,
        reference_class: PlausibilityReferenceClass | None,
        capital_allocation: CapitalAllocationProfile | None = None,
    ) -> ThreePResult:
        possible_checks = self._possible(assumptions, possible_context)
        possible = (
            CheckStatus.FAIL
            if any(item.status == CheckStatus.FAIL for item in possible_checks)
            else CheckStatus.PASS
        )
        plausible_checks = self._plausible(
            assumptions,
            reference_class,
            capital_allocation,
        )
        known_plausibility = [
            item.status
            for item in plausible_checks
            if item.status != PlausibilityStatus.UNKNOWN
        ]
        known_reference_class_checks = [
            item
            for item in plausible_checks
            if item.name not in {"REFERENCE_CLASS", "COMPANY_HISTORICAL_ROIIC"}
            and item.status != PlausibilityStatus.UNKNOWN
        ]
        plausible = (
            PlausibilityStatus.OUTLIER
            if any(item == PlausibilityStatus.OUTLIER for item in known_plausibility)
            else PlausibilityStatus.IN_RANGE
            if known_reference_class_checks
            else PlausibilityStatus.UNKNOWN
        )
        driver_assessments = self._probable(evidence)
        statuses = [item.status for item in driver_assessments]
        if ProbabilitySupport.CONTRADICTED in statuses:
            probable = ProbabilitySupport.CONTRADICTED
        elif ProbabilitySupport.MIXED in statuses:
            probable = ProbabilitySupport.MIXED
        elif statuses and all(item == ProbabilitySupport.SUPPORTED for item in statuses):
            probable = ProbabilitySupport.SUPPORTED
        elif ProbabilitySupport.SUPPORTED in statuses:
            probable = ProbabilitySupport.MIXED
        else:
            probable = ProbabilitySupport.WEAK

        if possible == CheckStatus.FAIL:
            verdict = ThreePVerdict.FAIL
        elif plausible == PlausibilityStatus.OUTLIER or probable == ProbabilitySupport.CONTRADICTED:
            verdict = ThreePVerdict.REVIEW
        elif plausible == PlausibilityStatus.IN_RANGE and probable == ProbabilitySupport.SUPPORTED:
            verdict = ThreePVerdict.PASS
        else:
            verdict = ThreePVerdict.REVIEW
        return ThreePResult(
            scenario=assumptions.scenario,
            possible=possible,
            possible_checks=possible_checks,
            plausible=plausible,
            plausible_checks=plausible_checks,
            probable=probable,
            driver_assessments=driver_assessments,
            verdict=verdict,
        )

    @staticmethod
    def _possible(
        assumptions: EconomicDcfAssumptions,
        context: PossibleContext,
    ) -> list[ThreePCheck]:
        valuation = EconomicDcfEngine().value(assumptions)
        final_revenue = valuation.projections[-1].revenue
        final_reinvestment = valuation.projections[-1].reinvestment
        checks = [
            ThreePCheck(
                name="WACC_EXCEEDS_STABLE_GROWTH",
                status=CheckStatus.PASS if assumptions.wacc > assumptions.stable_growth else CheckStatus.FAIL,
                observed=str(assumptions.wacc),
                boundary=f"> {assumptions.stable_growth}",
                rationale="Gordon growth requires discount rate above stable growth.",
            ),
            ThreePCheck(
                name="STABLE_ROIC_FUNDS_GROWTH",
                status=CheckStatus.PASS if assumptions.stable_roic > assumptions.stable_growth else CheckStatus.FAIL,
                observed=str(assumptions.stable_roic),
                boundary=f"> {assumptions.stable_growth}",
                rationale="Stable growth must be funded by a feasible reinvestment rate below 100%.",
            ),
            ThreePCheck(
                name="REVENUE_GROWTH_CEILING",
                status=(
                    CheckStatus.PASS
                    if assumptions.revenue_growth <= context.maximum_revenue_growth
                    else CheckStatus.FAIL
                ),
                observed=str(assumptions.revenue_growth),
                boundary=f"<= {context.maximum_revenue_growth}",
                rationale="The configured physical/economic growth ceiling is deterministic.",
            ),
            ThreePCheck(
                name="STABLE_GROWTH_CEILING",
                status=(
                    CheckStatus.PASS
                    if assumptions.stable_growth <= context.stable_growth_ceiling
                    else CheckStatus.FAIL
                ),
                observed=str(assumptions.stable_growth),
                boundary=f"<= {context.stable_growth_ceiling}",
                rationale="Stable growth cannot exceed the configured long-run economy ceiling.",
            ),
        ]
        checks.append(
            ThreePCheck(
                name="TAM_CEILING",
                status=(
                    CheckStatus.UNKNOWN
                    if context.tam_ceiling is None
                    else CheckStatus.PASS
                    if final_revenue <= context.tam_ceiling
                    else CheckStatus.FAIL
                ),
                observed=str(final_revenue),
                boundary=str(context.tam_ceiling) if context.tam_ceiling is not None else "not supplied",
                rationale="Explicit-period revenue must fit inside a PIT TAM ceiling.",
            )
        )
        checks.append(
            ThreePCheck(
                name="MARGIN_BELOW_GROSS_MARGIN",
                status=(
                    CheckStatus.UNKNOWN
                    if context.gross_margin_ceiling is None
                    else CheckStatus.PASS
                    if assumptions.target_nopat_margin <= context.gross_margin_ceiling
                    else CheckStatus.FAIL
                ),
                observed=str(assumptions.target_nopat_margin),
                boundary=(
                    str(context.gross_margin_ceiling)
                    if context.gross_margin_ceiling is not None
                    else "not supplied"
                ),
                rationale="NOPAT margin cannot exceed the supplied gross-margin ceiling.",
            )
        )
        growth_requires_reinvestment = assumptions.revenue_growth <= 0 or any(
            item.reinvestment >= context.minimum_positive_reinvestment
            for item in valuation.projections[: max(1, assumptions.competitive_advantage_period_years)]
        )
        checks.append(
            ThreePCheck(
                name="GROWTH_REQUIRES_REINVESTMENT",
                status=CheckStatus.PASS if growth_requires_reinvestment else CheckStatus.FAIL,
                observed=str(final_reinvestment),
                boundary=f">= {context.minimum_positive_reinvestment} during growth",
                rationale="Growth is linked to ROIIC or sales-to-capital rather than free cash creation.",
            )
        )
        return checks

    @staticmethod
    def _plausible(
        assumptions: EconomicDcfAssumptions,
        reference: PlausibilityReferenceClass | None,
        capital_allocation: CapitalAllocationProfile | None,
    ) -> list[ThreePCheck]:
        if reference is None:
            result = [
                ThreePCheck(
                    name="REFERENCE_CLASS",
                    status=PlausibilityStatus.UNKNOWN,
                    observed="none",
                    boundary="PIT industry/reference-class data required",
                    rationale="Plausibility is not inferred without an external base rate.",
                )
            ]
        else:
            fields = [
                ("REVENUE_GROWTH", assumptions.revenue_growth, reference.revenue_growth),
                ("TARGET_NOPAT_MARGIN", assumptions.target_nopat_margin, reference.nopat_margin),
                ("ROIIC", assumptions.roiic, reference.roiic),
                ("CAP_YEARS", assumptions.competitive_advantage_period_years, reference.cap_years),
                ("STABLE_GROWTH", assumptions.stable_growth, reference.stable_growth),
            ]
            if assumptions.reinvestment_method == ReinvestmentMethod.SALES_TO_CAPITAL:
                fields.append(
                    ("SALES_TO_CAPITAL", assumptions.sales_to_capital, reference.sales_to_capital)
                )
            result = []
            for name, value, bounds in fields:
                if bounds is None or value is None:
                    status = PlausibilityStatus.UNKNOWN
                    boundary = "not supplied"
                else:
                    status = (
                        PlausibilityStatus.IN_RANGE
                        if bounds.contains(value)
                        else PlausibilityStatus.OUTLIER
                    )
                    boundary = f"[{bounds.low}, {bounds.high}]"
                result.append(
                    ThreePCheck(
                        name=name,
                        status=status,
                        observed=str(value),
                        boundary=boundary,
                        rationale=f"Compared with PIT reference class {reference.name}.",
                    )
                )

        historical_roiic = (
            [
                item.adjusted_roiic
                for item in capital_allocation.periods
                if item.adjusted_roiic is not None
            ]
            if capital_allocation is not None
            else []
        )
        if not historical_roiic and capital_allocation is not None:
            historical_roiic = [
                item.reported_roiic
                for item in capital_allocation.periods
                if item.reported_roiic is not None
            ]
        if historical_roiic:
            low = min(historical_roiic)
            high = max(historical_roiic)
            result.append(
                ThreePCheck(
                    name="COMPANY_HISTORICAL_ROIIC",
                    status=(
                        PlausibilityStatus.IN_RANGE
                        if low <= assumptions.roiic <= high
                        else PlausibilityStatus.OUTLIER
                    ),
                    observed=str(assumptions.roiic),
                    boundary=f"[{low}, {high}]",
                    rationale=(
                        "Compared with the company's PIT historical intangible-adjusted "
                        "ROIIC range (reported ROIIC fallback)."
                    ),
                )
            )
        elif capital_allocation is not None:
            result.append(
                ThreePCheck(
                    name="COMPANY_HISTORICAL_ROIIC",
                    status=PlausibilityStatus.UNKNOWN,
                    observed=str(assumptions.roiic),
                    boundary="at least two usable capital periods required",
                    rationale="No historical incremental-return observation was available.",
                )
            )
        return result

    @staticmethod
    def _probable(
        evidence: ValuationDriverEvidenceBundle,
    ) -> list[DriverProbabilityAssessment]:
        grouped = evidence.by_driver()
        result: list[DriverProbabilityAssessment] = []
        for driver in ValuationDriver:
            items = grouped[driver]
            support = sorted(
                item.evidence_id for item in items if item.role == ValuationEvidenceRole.SUPPORT
            )
            counter = sorted(
                item.evidence_id for item in items if item.role == ValuationEvidenceRole.COUNTER
            )
            scenario = sorted(
                item.evidence_id
                for item in items
                if item.role == ValuationEvidenceRole.SCENARIO_INPUT
            )
            wideners = sorted(
                item.evidence_id
                for item in items
                if item.range_widening_required
                or item.role == ValuationEvidenceRole.RANGE_WIDENER
            )
            if counter and not support:
                status = ProbabilitySupport.CONTRADICTED
            elif counter and support:
                status = ProbabilitySupport.MIXED
            elif support:
                status = ProbabilitySupport.SUPPORTED
            else:
                status = ProbabilitySupport.WEAK
            result.append(
                DriverProbabilityAssessment(
                    driver=driver,
                    status=status,
                    supporting_evidence_ids=support,
                    counterevidence_ids=counter,
                    scenario_evidence_ids=scenario,
                    range_widener_evidence_ids=wideners,
                )
            )
        return result
