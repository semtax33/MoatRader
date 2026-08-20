from __future__ import annotations

import inspect
from datetime import date
from decimal import Decimal as D

import pytest
from pydantic import ValidationError

import moatrader.valuation.execution as execution_module
from moatrader.valuation import (
    ApvAssumptions,
    ApvCase,
    EconomicArchetype,
    EconomicDcfAssumptions,
    EconomicFcffScenarioSet,
    ExecutionStatus,
    NavAsset,
    NavAssumptions,
    CyclePhase,
    NormalizationContract,
    NormalizedFcffAssumptions,
    PipelineAsset,
    RimAssumptions,
    RimScenarioSet,
    RnpvScenarioSet,
    RoutedValuationExecutor,
    RoutedValuationInput,
    ScenarioDcfAssumptions,
    SotpAssumptions,
    SotpPart,
    SotpValueBasis,
    ValuationMethod,
    ValuationProfile,
    ValuationProfileRouter,
)
from moatrader.valuation.biotech_rnpv import BiotechRnpvAssumptions
from moatrader.valuation.router import REQUIRED_DATA


AS_OF = date(2026, 5, 31)


def _economic(growth: str) -> EconomicDcfAssumptions:
    return EconomicDcfAssumptions(
        base_revenue=D("1000"),
        base_nopat_margin=D("0.10"),
        base_invested_capital=D("800"),
        revenue_growth=D(growth),
        target_nopat_margin=D("0.12"),
        roiic=D("0.15"),
        competitive_advantage_period_years=3,
        fade_years=2,
        explicit_forecast_years=5,
        stable_growth=D("0.02"),
        stable_nopat_margin=D("0.10"),
        stable_roic=D("0.10"),
        wacc=D("0.10"),
        diluted_shares=D("10"),
    )


def _fcff(method: ValuationMethod) -> EconomicFcffScenarioSet:
    return EconomicFcffScenarioSet(
        downside=_economic("0.01"),
        base=_economic("0.05"),
        upside=_economic("0.09"),
        method=method,
        assumption_confidence=D("0.8"),
        provenance=["PIT:FCFF"],
    )


def _rim(roe: str) -> RimAssumptions:
    return RimAssumptions(
        book_equity=D("1000"),
        roe_path=[D(roe)] * 5,
        cost_of_equity=D("0.10"),
        payout_ratio=D("0.40"),
        terminal_roe=D(roe),
        terminal_growth=D("0.03"),
        diluted_shares=D("10"),
        assumption_confidence=D("0.8"),
        provenance=["PIT:RIM"],
    )


def _rnpv(probability: str) -> BiotechRnpvAssumptions:
    return BiotechRnpvAssumptions(
        assets=[
            PipelineAsset(
                name="Asset A",
                years_to_launch=3,
                probability_of_approval=D(probability),
                launch_value=D("1000"),
                remaining_development_costs=[D("20")] * 3,
                evidence_ids=["PIT:RNPV"],
            )
        ],
        discount_rate=D("0.12"),
        net_cash=D("100"),
        diluted_shares=D("10"),
    )


def _apv_case(fcff: str, shield: str) -> ApvCase:
    return ApvCase(
        unlevered_fcff=[D(fcff)] * 5,
        terminal_cash_flow=D(fcff),
        terminal_growth=D("0.02"),
        unlevered_cost_of_capital=D("0.10"),
        tax_shields=[D(shield)] * 5,
        tax_shield_discount_rate=D("0.06"),
    )


def _sotp_part(name: str, scope: str, value: str) -> SotpPart:
    base = D(value)
    return SotpPart(
        name=name,
        method=ValuationMethod.ECONOMIC_FCFF,
        value_basis=SotpValueBasis.ENTERPRISE,
        downside_value=base * D("0.8"),
        base_value=base,
        upside_value=base * D("1.2"),
        ownership_applied=False,
        net_debt_adjustment=D("0"),
        net_debt_scope_id=f"NET_DEBT:{name}",
        nci_adjustment=D("0"),
        nci_scope_id=f"NCI:{name}",
        cashflow_scope_id=f"SCOPE:{name}",
        included_cashflows=[scope],
        actual_engine="TEST_FIXTURE_ENGINE",
        submodel_input_sha256="0" * 64,
        provenance=[f"PIT:{name}"],
    )


def _case(method: ValuationMethod) -> tuple[ValuationProfile, object, str]:
    common: dict[str, object] = {
        "issuer_id": "C1",
        "as_of": AS_OF,
        "sector": "Sector",
        "industry": "Industry",
        "economic_archetype": EconomicArchetype.GENERAL_OPERATING,
        "available_data": list(REQUIRED_DATA[method]),
        "provenance": ["PIT:PROFILE"],
    }
    if method == ValuationMethod.ECONOMIC_FCFF:
        assumptions = _fcff(method)
        engine = "CommonEconomicFcffEngine"
    elif method == ValuationMethod.NORMALIZED_FCFF:
        common.update(
            economic_archetype=EconomicArchetype.CYCLICAL_OPERATING,
            materially_cyclical=True,
        )
        assumptions = NormalizedFcffAssumptions(
            downside=_economic("0.01").model_copy(update={"scenario": "DOWNSIDE"}),
            base=_economic("0.05").model_copy(update={"scenario": "CENTRAL"}),
            upside=_economic("0.09").model_copy(update={"scenario": "UPSIDE"}),
            normalization=NormalizationContract(
                included_fiscal_years=[2020, 2021, 2022, 2023, 2024],
                cycle_phase=CyclePhase.MID_CYCLE,
            ),
            normalized_revenue_growth=D("0.05"),
            normalized_nopat_margin=D("0.12"),
            normalized_sales_to_capital=D("1.25"),
            assumption_confidence=D("0.8"),
            provenance=["PIT:NORMALIZED_FCFF"],
        )
        engine = "NormalizedFcffEngine"
    elif method == ValuationMethod.RIM:
        common.update(
            economic_archetype=EconomicArchetype.FINANCIAL_INTERMEDIARY,
            is_financial_intermediary=True,
        )
        assumptions = RimScenarioSet(
            downside=_rim("0.08"), base=_rim("0.12"), upside=_rim("0.16")
        )
        engine = "CommonRimEngine"
    elif method == ValuationMethod.RNPV:
        common.update(
            economic_archetype=EconomicArchetype.PRE_REVENUE_BIOTECH,
            pipeline_assets_material=True,
            ebit_positive=False,
        )
        assumptions = RnpvScenarioSet(
            downside=_rnpv("0.20"), base=_rnpv("0.50"), upside=_rnpv("0.80")
        )
        engine = "CommonRnpvEngine"
    elif method == ValuationMethod.SCENARIO_DCF:
        common.update(
            economic_archetype=EconomicArchetype.LOSS_MAKING_GROWTH,
            ebit_positive=False,
            persistent_loss=True,
            path_to_positive_unit_economics=True,
        )
        assumptions = ScenarioDcfAssumptions(
            downside=_economic("0.01").model_copy(update={"scenario": "DOWNSIDE"}),
            central=_economic("0.05").model_copy(update={"scenario": "CENTRAL"}),
            upside=_economic("0.09").model_copy(update={"scenario": "UPSIDE"}),
            assumption_confidence=D("0.8"),
            provenance=["PIT:SCENARIO"],
        )
        engine = "ScenarioDcfEngine"
    elif method == ValuationMethod.APV:
        common.update(
            economic_archetype=EconomicArchetype.LEVERAGE_DRIVEN,
            leverage_path_material=True,
        )
        assumptions = ApvAssumptions(
            downside=_apv_case("70", "5"),
            base=_apv_case("100", "10"),
            upside=_apv_case("130", "15"),
            debt=D("300"),
            cash=D("50"),
            diluted_shares=D("10"),
            assumption_confidence=D("0.8"),
            provenance=["PIT:APV"],
        )
        engine = "ApvEngine"
    elif method == ValuationMethod.NAV:
        common.update(
            economic_archetype=EconomicArchetype.ASSET_BACKED,
            asset_value_primary=True,
        )
        assumptions = NavAssumptions(
            assets=[NavAsset(name="Asset", base_value=D("1000"), evidence_ids=["PIT:NAV"])],
            cash=D("100"),
            debt=D("300"),
            diluted_shares=D("10"),
            assumption_confidence=D("0.8"),
            provenance=["PIT:NAV"],
        )
        engine = "NavEngine"
    elif method == ValuationMethod.SOTP:
        common.update(
            economic_archetype=EconomicArchetype.MULTI_BUSINESS,
            multi_segment=True,
            segment_heterogeneity_material=True,
        )
        assumptions = SotpAssumptions(
            parts=[
                _sotp_part("Part A", "A", "1000"),
                _sotp_part("Part B", "B", "500"),
            ],
            parent_cash=D("100"),
            parent_debt=D("300"),
            diluted_shares=D("10"),
            assumption_confidence=D("0.8"),
            provenance=["PIT:SOTP"],
        )
        engine = "SotpEngine"
    else:  # pragma: no cover - test case list is explicit
        raise AssertionError(method)
    return ValuationProfile(**common), assumptions, engine


@pytest.mark.parametrize(
    "method",
    [
        ValuationMethod.ECONOMIC_FCFF,
        ValuationMethod.NORMALIZED_FCFF,
        ValuationMethod.RIM,
        ValuationMethod.RNPV,
        ValuationMethod.SCENARIO_DCF,
        ValuationMethod.APV,
        ValuationMethod.NAV,
        ValuationMethod.SOTP,
    ],
)
def test_every_route_executes_its_real_engine(method: ValuationMethod) -> None:
    profile, assumptions, expected_engine = _case(method)
    route = ValuationProfileRouter().route(profile)
    payload = RoutedValuationInput(
        issuer_id=profile.issuer_id,
        as_of=profile.as_of,
        method=method,
        assumptions=assumptions.model_dump(mode="json"),
        source_refs=["PIT:MODEL_INPUT"],
    )

    prepared = RoutedValuationExecutor.prepare(payload)
    execution = RoutedValuationExecutor().execute(route, prepared)

    assert route.primary_method == method
    assert execution.status == ExecutionStatus.VALUED
    assert execution.actual_engine == expected_engine
    assert execution.valuation is not None
    assert execution.valuation.method == method


def test_wrong_method_input_fails_closed_without_fcff_fallback() -> None:
    nav_profile, _, _ = _case(ValuationMethod.NAV)
    route = ValuationProfileRouter().route(nav_profile)
    _, rim_assumptions, _ = _case(ValuationMethod.RIM)
    prepared = RoutedValuationExecutor.prepare(
        RoutedValuationInput(
            issuer_id="C1",
            as_of=AS_OF,
            method=ValuationMethod.RIM,
            assumptions=rim_assumptions.model_dump(mode="json"),
            source_refs=["PIT:WRONG_MODEL"],
        )
    )

    execution = RoutedValuationExecutor().execute(route, prepared)

    assert execution.status == ExecutionStatus.INPUT_ROUTE_MISMATCH
    assert execution.valuation is None
    assert execution.actual_engine is None
    assert execution.reason_codes == ["METHOD_MISMATCH"]


def test_missing_method_input_is_unvalued_not_fallback_fcff() -> None:
    profile, _, _ = _case(ValuationMethod.RNPV)
    route = ValuationProfileRouter().route(profile)

    execution = RoutedValuationExecutor().execute(route, None)

    assert execution.status == ExecutionStatus.MISSING_METHOD_INPUT
    assert execution.valuation is None
    assert execution.reason_codes == ["MISSING_INPUT_FOR_RNPV"]


def test_normalized_fcff_cannot_relabel_economic_fcff_assumptions() -> None:
    with pytest.raises(ValueError, match="normalization"):
        RoutedValuationExecutor.prepare(
            RoutedValuationInput(
                issuer_id="C1",
                as_of=AS_OF,
                method=ValuationMethod.NORMALIZED_FCFF,
                assumptions=_fcff(ValuationMethod.ECONOMIC_FCFF).model_dump(mode="json"),
                source_refs=["PIT:MISMATCH"],
            )
        )


def test_execution_path_has_no_llm_dependency() -> None:
    source = inspect.getsource(execution_module).lower()
    assert "openai" not in source
    assert "llm" in source  # explicit contract documentation: no LLM


@pytest.mark.parametrize(
    "update",
    [
        {"schema_version": "routed-valuation-input/999"},
        {"source_refs": ["PIT:X", "PIT:X"]},
        {"current_price": "100"},
        {"forward_return": "0.25"},
    ],
)
def test_routed_input_rejects_wrong_schema_duplicate_sources_and_market_data(
    update: dict[str, object],
) -> None:
    _, assumptions, _ = _case(ValuationMethod.RIM)
    payload: dict[str, object] = {
        "issuer_id": "C1",
        "as_of": AS_OF,
        "method": ValuationMethod.RIM,
        "assumptions": assumptions.model_dump(mode="json"),
        "source_refs": ["PIT:X"],
    }
    payload.update(update)

    with pytest.raises(ValidationError):
        RoutedValuationInput.model_validate(payload)
