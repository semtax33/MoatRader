from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from moatrader.business import (
    CapitalAllocationAnalyzer,
    CapitalPeriod,
    CapEngine,
    CapPrior,
    CompetitiveAdvantageProfile,
    IntangibleAdjustmentPolicy,
    ValuationDriver,
    ValuationDriverMapper,
    ValuationDriverExtraction,
    ValuationEvidenceRole,
    build_valuation_driver_consensus,
)
from moatrader.business.lifecycle import CompanyType, LifeCycleStage
from moatrader.canonical.models import SectionRole, SourceRef, SourceType, StatementType
from moatrader.evidence.atomic import select_atomic_evidence_units, select_valuation_evidence_units
from moatrader.evidence.models import (
    AtomicMoatRole,
    DcfLink,
    EvidenceCard,
    EvidenceDirection,
    EvidenceType,
    ForwardDriverType,
)
from moatrader.expectations import (
    ExpectationAnalysisEngine,
    ExpectationAnalysisRequest,
    ExpectationGapDirection,
)
from moatrader.llm import LLMTask, build_valuation_driver_request
from moatrader.screening import (
    ExpectationOpportunityRanker,
    OpportunityCandidate,
)
from moatrader.valuation import (
    EconomicDcfAssumptions,
    EconomicDcfEngine,
    IntrinsicScenarioSet,
    MarketPriceInput,
    PossibleContext,
    ReverseDcfEngine,
    ReverseDcfGrid,
    ScenarioValuationEngine,
    ValuationMethod,
    ValuationModelRouter,
)
from moatrader.valuation.biotech_rnpv import (
    BiotechRnpvAssumptions,
    BiotechRnpvEngine,
    PipelineAsset,
)
from moatrader.valuation.reference_class import (
    DecimalRange,
    IntegerRange,
    PlausibilityReferenceClass,
)
from moatrader.semantic import SemanticChunk


D = Decimal
AS_OF = datetime(2026, 8, 14, 15, 30, tzinfo=timezone.utc)


def _assumptions(
    scenario: str = "CENTRAL",
    *,
    growth: str = "0.10",
    margin: str = "0.15",
    roiic: str = "0.20",
    cap: int = 5,
) -> EconomicDcfAssumptions:
    return EconomicDcfAssumptions(
        scenario=scenario,
        base_period="2025_TTM",
        base_revenue=D("1000"),
        base_nopat_margin=D("0.10"),
        base_invested_capital=D("500"),
        revenue_growth=D(growth),
        target_nopat_margin=D(margin),
        margin_convergence_years=3,
        roiic=D(roiic),
        competitive_advantage_period_years=cap,
        fade_years=3,
        explicit_forecast_years=max(10, cap + 3),
        stable_growth=D("0.025"),
        stable_nopat_margin=D("0.10"),
        stable_roic=D("0.09"),
        wacc=D("0.09"),
        net_debt=D("100"),
        diluted_shares=D("10"),
    )


def _card(
    evidence_id: str,
    evidence_type: EvidenceType,
    role: AtomicMoatRole,
    direction: EvidenceDirection,
    fact: str,
    *,
    statement_type: StatementType = StatementType.DISCLOSED_FACT,
    forward_driver: ForwardDriverType | None = None,
    dcf_links: list[DcfLink] | None = None,
) -> EvidenceCard:
    return EvidenceCard(
        evidence_id=evidence_id,
        source_chunk_id=f"CHUNK-{evidence_id}",
        node_ids=[f"NODE-{evidence_id}"],
        moat_role=role,
        evidence_type=evidence_type,
        statement_type=statement_type,
        fact=fact,
        raw_quote=fact,
        mechanism=["observable company-specific relation"],
        direction=direction,
        source_type=SourceType.IR,
        reliability=0.9,
        forward_driver_type=forward_driver,
        dcf_links=dcf_links or [],
    )


def _cards() -> list[EvidenceCard]:
    return [
        _card(
            "E_COST",
            EvidenceType.COST_ADVANTAGE,
            AtomicMoatRole.MECHANISM,
            EvidenceDirection.MOAT_POSITIVE,
            "Direct production removes more than 30 process steps and lowers manufacturing cost.",
        ),
        _card(
            "E_SHARE",
            EvidenceType.MARKET_SHARE,
            AtomicMoatRole.OUTCOME,
            EvidenceDirection.MOAT_POSITIVE,
            "The company retained the number-one market position for 11 years.",
        ),
        _card(
            "E_PIPELINE",
            EvidenceType.OTHER,
            AtomicMoatRole.NONE,
            EvidenceDirection.NEUTRAL,
            "A disclosed phase-three pipeline is expected to launch in 2028.",
            statement_type=StatementType.MANAGEMENT_CLAIM,
            forward_driver=ForwardDriverType.MARKET_GROWTH,
            dcf_links=[DcfLink.REVENUE],
        ),
    ]


def _scenarios(confidence: str = "0.8") -> IntrinsicScenarioSet:
    return IntrinsicScenarioSet(
        downside=_assumptions("DOWNSIDE", growth="0.04", margin="0.10", roiic="0.12", cap=3),
        central=_assumptions(),
        upside=_assumptions("UPSIDE", growth="0.15", margin="0.18", roiic="0.25", cap=8),
        evidence_confidence=D(confidence),
    )


def _reference() -> PlausibilityReferenceClass:
    return PlausibilityReferenceClass(
        name="industrial-growth-peer-set",
        as_of=AS_OF,
        source_refs=["INDUSTRY:PIT:2026-08-14"],
        revenue_growth=DecimalRange(low=D("0.02"), high=D("0.20")),
        nopat_margin=DecimalRange(low=D("0.05"), high=D("0.20")),
        roiic=DecimalRange(low=D("0.08"), high=D("0.30")),
        cap_years=IntegerRange(low=2, high=12),
        stable_growth=DecimalRange(low=D("0"), high=D("0.04")),
    )


def test_valuation_mapper_preserves_moat_none_forward_evidence_and_prevents_double_counting() -> None:
    bundle = ValuationDriverMapper().map_cards(issuer_id="ISSUER", as_of=AS_OF, cards=_cards())
    cost = next(item for item in bundle.evidence if item.evidence_id == "E_COST")
    pipeline = next(item for item in bundle.evidence if item.evidence_id == "E_PIPELINE")

    assert cost.primary_driver == ValuationDriver.TARGET_MARGIN
    assert cost.related_drivers == [ValuationDriver.ROIIC, ValuationDriver.CAP_FADE]
    assert cost.numeric_adjustment_allowed is False
    assert cost.exclusive_application_key == "E_COST"
    assert pipeline.primary_driver == ValuationDriver.REVENUE_GROWTH
    assert pipeline.role == ValuationEvidenceRole.SCENARIO_INPUT
    assert pipeline.moat_role == AtomicMoatRole.NONE


def test_persistence_parser_ignores_implausible_large_year_counts() -> None:
    assert ValuationDriverMapper._persistence_years_text("10년 연속 유지") == 10
    assert ValuationDriverMapper._persistence_years_text("987년 수치가 표에 있음") is None


def test_legacy_positive_disclosed_fact_is_valuation_support_not_claim_support() -> None:
    disclosed = _cards()[0].model_copy(
        update={
            "moat_role": AtomicMoatRole.NONE,
            "direction": EvidenceDirection.MOAT_POSITIVE,
            "statement_type": StatementType.DISCLOSED_FACT,
            "reliability": 0.8,
        }
    )
    forecast = disclosed.model_copy(update={"statement_type": StatementType.FORECAST})

    assert ValuationDriverMapper._role(disclosed) == ValuationEvidenceRole.SUPPORT
    assert ValuationDriverMapper._role(forecast) == ValuationEvidenceRole.SCENARIO_INPUT


def test_valuation_selector_and_prompt_are_independent_from_frozen_moat_sensor() -> None:
    chunk = SemanticChunk(
        chunk_id="PIPELINE",
        document_id="DOC",
        node_ids=["N1"],
        chunk_type="atomic_evidence",
        section_path=["Pipeline"],
        section_role=SectionRole.GUIDANCE,
        markdown="The phase-three clinical program is planned for commercial launch in 2028.",
        token_count=12,
        source_refs=[SourceRef(source_type=SourceType.IR, document_id="DOC", page=4)],
        metadata={"atomic_evidence_key": "KEY-PIPELINE"},
    )

    assert select_atomic_evidence_units([chunk], 1) == []
    assert select_valuation_evidence_units([chunk], 1) == [chunk]
    request = build_valuation_driver_request(chunk, issuer_id="ISSUER")
    assert request.task == LLMTask.VALUATION_DRIVER_CLASSIFICATION
    assert request.metadata["price_inputs_present"] is False
    assert "MOAT_NONE" in request.system
    assert "Current price" not in request.user


def test_valuation_driver_consensus_requires_strict_route_majority() -> None:
    revenue = ValuationDriverExtraction(
        relevant=True,
        primary_driver=ValuationDriver.REVENUE_GROWTH,
        role=ValuationEvidenceRole.SCENARIO_INPUT,
        fact="Pipeline timing constrains revenue timing.",
    )
    margin = revenue.model_copy(update={"primary_driver": ValuationDriver.TARGET_MARGIN})
    consensus, diagnostics = build_valuation_driver_consensus([revenue, revenue, margin])
    assert consensus.primary_driver == ValuationDriver.REVENUE_GROWTH
    assert diagnostics["status"] == "STRICT_MAJORITY"

    failed, failed_diagnostics = build_valuation_driver_consensus(
        [
            revenue,
            margin,
            revenue.model_copy(update={"primary_driver": ValuationDriver.RISK}),
        ]
    )
    assert failed.relevant is False
    assert failed_diagnostics["status"] == "NO_STRICT_MAJORITY"


def test_competitive_advantage_profile_and_cap_use_evidence_not_public_moat_score() -> None:
    bundle = ValuationDriverMapper().map_cards(issuer_id="ISSUER", as_of=AS_OF, cards=_cards())
    profile = CompetitiveAdvantageProfile.from_driver_evidence(bundle)
    assessment = CapEngine().assess(
        profile,
        CapPrior(
            reference_class="industry-base-rate",
            as_of=AS_OF,
            low_years=3,
            central_years=5,
            high_years=8,
            source_refs=["INDUSTRY:CAP"],
        ),
    )

    assert EvidenceType.COST_ADVANTAGE in profile.mechanism_evidence_ids
    assert assessment.price_inputs_used is False
    assert assessment.low_years <= assessment.central_years <= assessment.high_years
    assert "E_SHARE" in assessment.supporting_evidence_ids
    assert "E_COST" not in assessment.supporting_evidence_ids
    assert not hasattr(assessment, "moat_score")


def test_economic_dcf_links_growth_reinvestment_roiic_and_fades_to_steady_state() -> None:
    assumptions = _assumptions()
    valuation = EconomicDcfEngine().value(assumptions)

    assert valuation.enterprise_value - assumptions.net_debt == valuation.equity_value
    assert valuation.fair_value_per_share == valuation.equity_value / assumptions.diluted_shares
    assert valuation.projections[0].realized_roiic == assumptions.roiic
    assert valuation.projections[5].phase.value == "FADE"
    assert valuation.projections[-1].phase.value == "STEADY_STATE"
    assert valuation.terminal_reinvestment_rate == assumptions.stable_growth / assumptions.stable_roic
    assert valuation.cap_value_contribution > 0
    assert valuation.pvgo_share is not None


def test_economic_intrinsic_assumptions_reject_market_price_leakage() -> None:
    payload = _assumptions().model_dump(mode="python")
    payload["current_price"] = D("80")
    with pytest.raises(ValidationError, match="current_price"):
        EconomicDcfAssumptions.model_validate(payload)


def test_one_evidence_id_cannot_be_applied_to_multiple_dcf_levers() -> None:
    payload = _assumptions().model_dump(mode="python")
    payload["driver_evidence_ids"] = {
        ValuationDriver.TARGET_MARGIN: ["E_COST"],
        ValuationDriver.ROIIC: ["E_COST"],
    }

    with pytest.raises(ValidationError, match="applied to both"):
        EconomicDcfAssumptions.model_validate(payload)


def test_scenario_evidence_must_match_its_exclusive_primary_driver() -> None:
    scenarios = _scenarios().model_copy(
        update={
            "central": _scenarios().central.model_copy(
                update={
                    "driver_evidence_ids": {
                        ValuationDriver.ROIIC: ["E_COST"],
                    }
                }
            )
        }
    )
    request = ExpectationAnalysisRequest(
        evidence_cutoff=AS_OF,
        intrinsic_scenarios=scenarios,
        cap_prior=CapPrior(
            reference_class="peer",
            as_of=AS_OF,
            low_years=2,
            central_years=4,
            high_years=6,
            source_refs=["SRC"],
        ),
        reverse_grid=ReverseDcfGrid(
            revenue_growth=[D("0.05")],
            target_nopat_margin=[D("0.10")],
            roiic=[D("0.15")],
            cap_years=[4],
        ),
    )

    with pytest.raises(ValueError, match="exclusive primary driver is TARGET_MARGIN"):
        ExpectationAnalysisEngine().analyze(
            issuer_id="ISSUER",
            as_of=AS_OF,
            cards=_cards(),
            request=request,
            current_price=D("80"),
            price_as_of=AS_OF,
        )


def test_low_confidence_widens_value_range_without_changing_central_value() -> None:
    high = ScenarioValuationEngine().value(_scenarios("0.9"))
    low = ScenarioValuationEngine().value(_scenarios("0.2"))

    assert high.central_per_share == low.central_per_share
    assert low.confidence_adjusted_low_per_share < high.confidence_adjusted_low_per_share
    assert low.confidence_adjusted_high_per_share > high.confidence_adjusted_high_per_share
    assert low.price_inputs_used is False


def test_reverse_dcf_outputs_a_surface_and_enforces_pit_market_cutoff() -> None:
    assumptions = _assumptions()
    market = MarketPriceInput(current_price=D("100"), price_as_of=AS_OF, evidence_cutoff=AS_OF)
    surface = ReverseDcfEngine().surface(
        base_assumptions=assumptions,
        market=market,
        grid=ReverseDcfGrid(
            revenue_growth=[D("0.04"), D("0.10")],
            target_nopat_margin=[D("0.10"), D("0.15")],
            roiic=[D("0.12"), D("0.20")],
            cap_years=[2, 5, 8],
            price_tolerance=D("0.05"),
        ),
    )

    assert surface.point_count == 24
    assert 0 < surface.eligible_point_count <= surface.point_count
    assert surface.multiple_solutions_required is True

    with pytest.raises(ValidationError, match="300%"):
        ReverseDcfGrid(
            revenue_growth=[D("4")],
            target_nopat_margin=[D("0.10")],
            roiic=[D("0.20")],
            cap_years=[5],
        )
    assert surface.representative_points
    assert surface.implied_cap_years.low <= surface.implied_cap_years.high
    with pytest.raises(ValidationError, match="after the market price"):
        MarketPriceInput(
            current_price=D("100"),
            price_as_of=AS_OF,
            evidence_cutoff=datetime(2026, 8, 15, tzinfo=timezone.utc),
        )


def test_end_to_end_expectation_analysis_keeps_price_out_of_intrinsic_and_drops_moat_score() -> None:
    request = ExpectationAnalysisRequest(
        evidence_cutoff=AS_OF,
        intrinsic_scenarios=_scenarios(),
        possible_context=PossibleContext(
            tam_ceiling=D("10000"),
            gross_margin_ceiling=D("0.60"),
            source_refs=["INDUSTRY:TAM"],
        ),
        reference_class=_reference(),
        cap_prior=CapPrior(
            reference_class="industrial-growth-peer-set",
            as_of=AS_OF,
            low_years=3,
            central_years=5,
            high_years=8,
            source_refs=["INDUSTRY:CAP"],
        ),
        reverse_grid=ReverseDcfGrid(
            revenue_growth=[D("0.02"), D("0.05")],
            target_nopat_margin=[D("0.08"), D("0.10")],
            roiic=[D("0.10"), D("0.15")],
            cap_years=[1, 3, 5],
            price_tolerance=D("0.05"),
        ),
    )
    analysis = ExpectationAnalysisEngine().analyze(
        issuer_id="ISSUER",
        as_of=AS_OF,
        cards=_cards(),
        request=request,
        current_price=D("80"),
        price_as_of=AS_OF,
    )

    assert analysis.intrinsic_valuation.price_inputs_used is False
    assert analysis.three_p["central"].price_inputs_used is False
    assert analysis.old_moat_score_used_for_valuation is False
    assert analysis.implied_expectations.market.current_price == D("80")
    assert analysis.expectation_gap.direction in {
        ExpectationGapDirection.FAVORABLE,
        ExpectationGapDirection.OVERLAP,
        ExpectationGapDirection.UNFAVORABLE,
        ExpectationGapDirection.INDETERMINATE,
    }
    assert analysis.expectation_gap.confidence_changes_range_not_value is True

    with pytest.raises(ValueError, match="exactly match"):
        ExpectationAnalysisEngine().analyze(
            issuer_id="ISSUER",
            as_of=AS_OF,
            cards=_cards(),
            request=request.model_copy(
                update={"evidence_cutoff": datetime(2026, 8, 13, tzinfo=timezone.utc)}
            ),
            current_price=D("80"),
            price_as_of=AS_OF,
        )
    naive_payload = request.model_dump(mode="python")
    naive_payload["evidence_cutoff"] = datetime(2026, 8, 14)
    with pytest.raises(ValidationError, match="timezone-aware"):
        ExpectationAnalysisRequest.model_validate(naive_payload)

    with pytest.raises(ValueError, match="CAP prior data is after"):
        ExpectationAnalysisEngine().analyze(
            issuer_id="ISSUER",
            as_of=AS_OF,
            cards=_cards(),
            request=request.model_copy(
                update={
                    "cap_prior": request.cap_prior.model_copy(
                        update={"as_of": datetime(2026, 8, 15, tzinfo=timezone.utc)}
                    )
                }
            ),
            current_price=D("80"),
            price_as_of=datetime(2026, 8, 15, tzinfo=timezone.utc),
        )


def test_possible_stage_fails_when_revenue_exceeds_tam() -> None:
    request = ExpectationAnalysisRequest(
        evidence_cutoff=AS_OF,
        intrinsic_scenarios=_scenarios(),
        possible_context=PossibleContext(tam_ceiling=D("1001")),
        reference_class=_reference(),
        cap_prior=CapPrior(
            reference_class="peer",
            as_of=AS_OF,
            low_years=2,
            central_years=4,
            high_years=6,
            source_refs=["SRC"],
        ),
        reverse_grid=ReverseDcfGrid(
            revenue_growth=[D("0.01")],
            target_nopat_margin=[D("0.08")],
            roiic=[D("0.10")],
            cap_years=[1],
        ),
    )
    analysis = ExpectationAnalysisEngine().analyze(
        issuer_id="ISSUER",
        as_of=AS_OF,
        cards=_cards(),
        request=request,
        current_price=D("80"),
        price_as_of=AS_OF,
    )
    assert analysis.three_p["central"].possible.value == "FAIL"
    assert analysis.expectation_gap.direction == ExpectationGapDirection.INDETERMINATE


def test_capital_allocation_keeps_reported_and_intangible_adjusted_roic() -> None:
    profile = CapitalAllocationAnalyzer().analyze(
        [
            CapitalPeriod(
                period="2024",
                revenue=D("1000"),
                reported_nopat=D("100"),
                reported_invested_capital=D("500"),
                reinvestment=D("80"),
                rd_expense=D("50"),
            ),
            CapitalPeriod(
                period="2025",
                revenue=D("1150"),
                reported_nopat=D("125"),
                reported_invested_capital=D("580"),
                reinvestment=D("90"),
                rd_expense=D("60"),
            ),
        ],
        IntangibleAdjustmentPolicy(useful_life_years=5, tax_rate=D("0.25")),
    )

    assert profile.periods[-1].reported_roic != profile.periods[-1].adjusted_economic_roic
    assert profile.latest_reported_roiic == D("25") / D("90")
    assert profile.periods[-1].adjusted_reinvestment == D("140")
    assert profile.latest_adjusted_roiic == D("25") / D("140")
    assert profile.price_inputs_used is False


def test_capital_allocation_history_validates_but_does_not_overwrite_roiic() -> None:
    request = ExpectationAnalysisRequest(
        evidence_cutoff=AS_OF,
        intrinsic_scenarios=_scenarios(),
        reference_class=_reference(),
        cap_prior=CapPrior(
            reference_class="peer",
            as_of=AS_OF,
            low_years=2,
            central_years=4,
            high_years=6,
            source_refs=["SRC"],
        ),
        reverse_grid=ReverseDcfGrid(
            revenue_growth=[D("0.05")],
            target_nopat_margin=[D("0.10")],
            roiic=[D("0.15")],
            cap_years=[4],
            price_tolerance=D("0.50"),
        ),
        capital_periods=[
            CapitalPeriod(
                period="2024",
                revenue=D("1000"),
                reported_nopat=D("100"),
                reported_invested_capital=D("500"),
                reinvestment=D("80"),
                rd_expense=D("50"),
            ),
            CapitalPeriod(
                period="2025",
                revenue=D("1150"),
                reported_nopat=D("125"),
                reported_invested_capital=D("580"),
                reinvestment=D("90"),
                rd_expense=D("60"),
            ),
        ],
    )

    analysis = ExpectationAnalysisEngine().analyze(
        issuer_id="ISSUER",
        as_of=AS_OF,
        cards=_cards(),
        request=request,
        current_price=D("80"),
        price_as_of=AS_OF,
    )

    assert analysis.capital_allocation_profile is not None
    roiic_check = next(
        item
        for item in analysis.three_p["central"].plausible_checks
        if item.name == "COMPANY_HISTORICAL_ROIIC"
    )
    assert roiic_check.status.value == "OUTLIER"
    assert analysis.intrinsic_valuation.central.assumptions.roiic == D("0.20")


def test_scenario_labels_must_also_be_ordered_by_value() -> None:
    scenarios = IntrinsicScenarioSet(
        downside=_assumptions(
            "DOWNSIDE",
            growth="0.20",
            margin="0.30",
            roiic="0.40",
            cap=8,
        ),
        central=_assumptions(
            "CENTRAL",
            growth="0.04",
            margin="0.10",
            roiic="0.12",
            cap=3,
        ),
        upside=_assumptions(
            "UPSIDE",
            growth="0.06",
            margin="0.12",
            roiic="0.15",
            cap=4,
        ),
        evidence_confidence=D("0.8"),
    )

    with pytest.raises(ValueError, match="downside <= central <= upside"):
        ScenarioValuationEngine().value(scenarios)


def test_model_router_and_biotech_rnpv_fail_closed_by_economic_structure() -> None:
    router = ValuationModelRouter()
    bank = router.route(company_type=CompanyType.FINANCIAL, life_cycle_stage=LifeCycleStage.STABLE)
    biotech = router.route(
        company_type=CompanyType.PRE_REVENUE_BIOTECH,
        life_cycle_stage=LifeCycleStage.PRE_REVENUE,
    )
    assert bank.method == ValuationMethod.EXCESS_RETURN_EQUITY
    assert bank.implemented is False
    assert biotech.method == ValuationMethod.BIOTECH_RNPV

    valuation = BiotechRnpvEngine().value(
        BiotechRnpvAssumptions(
            assets=[
                PipelineAsset(
                    name="Asset A",
                    years_to_launch=3,
                    probability_of_approval=D("0.4"),
                    launch_value=D("1000"),
                    remaining_development_costs=[D("50"), D("40"), D("30")],
                    evidence_ids=["PIPELINE-STAGE-A"],
                )
            ],
            discount_rate=D("0.10"),
            net_cash=D("100"),
            diluted_shares=D("10"),
        )
    )
    assert valuation.assets[0].probability_adjusted_launch_value > 0
    assert valuation.fair_value_per_share == valuation.equity_value / D("10")


def test_opportunity_ranker_uses_gap_and_downside_not_moat_or_confidence_multiplier() -> None:
    request = ExpectationAnalysisRequest(
        evidence_cutoff=AS_OF,
        intrinsic_scenarios=_scenarios(),
        possible_context=PossibleContext(tam_ceiling=D("10000")),
        reference_class=_reference(),
        cap_prior=CapPrior(
            reference_class="peer",
            as_of=AS_OF,
            low_years=3,
            central_years=5,
            high_years=8,
            source_refs=["SRC"],
        ),
        reverse_grid=ReverseDcfGrid(
            revenue_growth=[D("0.01")],
            target_nopat_margin=[D("0.07")],
            roiic=[D("0.09")],
            cap_years=[1],
        ),
    )
    analysis = ExpectationAnalysisEngine().analyze(
        issuer_id="ISSUER",
        as_of=AS_OF,
        cards=_cards(),
        request=request,
        current_price=D("80"),
        price_as_of=AS_OF,
    )
    candidate = OpportunityCandidate(
        issuer_id="ISSUER",
        ticker="AAA",
        evaluation=analysis.expectation_gap,
        valuation_as_of=AS_OF,
        price_as_of=AS_OF,
    )
    ranked = ExpectationOpportunityRanker().rank([candidate])
    if analysis.expectation_gap.direction == ExpectationGapDirection.FAVORABLE:
        assert ranked[0].ticker == "AAA"
        assert not hasattr(ranked[0], "moat_score")
        assert not hasattr(ranked[0], "quality_value_score")
    else:
        assert ranked == []
