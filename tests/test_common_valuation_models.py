from __future__ import annotations

from decimal import Decimal as D

import pytest
from pydantic import ValidationError

from moatrader.valuation import (
    ApvAssumptions,
    ApvCase,
    ApvEngine,
    BiotechRnpvAssumptions,
    CommonEconomicFcffEngine,
    CommonRimEngine,
    CommonRnpvEngine,
    EconomicDcfAssumptions,
    EconomicFcffScenarioSet,
    NavAsset,
    NavAssumptions,
    NavEngine,
    PipelineAsset,
    RimAssumptions,
    RimScenarioSet,
    RnpvScenarioSet,
    ScenarioDcfAssumptions,
    ScenarioDcfEngine,
    SotpAssumptions,
    SotpEngine,
    SotpPart,
    SotpValueBasis,
    ValuationMethod,
)


def _economic_dcf(growth: str) -> EconomicDcfAssumptions:
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


def test_economic_fcff_returns_common_ordered_equity_result() -> None:
    result = CommonEconomicFcffEngine().value(
        EconomicFcffScenarioSet(
            downside=_economic_dcf("0.01"),
            base=_economic_dcf("0.05"),
            upside=_economic_dcf("0.09"),
        )
    )

    assert result.method == ValuationMethod.ECONOMIC_FCFF
    assert result.enterprise_value is not None
    assert result.downside_value_per_share < result.base_value_per_share < result.upside_value_per_share
    assert result.fair_value_per_share == result.base_value_per_share


def test_scenario_dcf_returns_probability_weighted_common_result() -> None:
    result = ScenarioDcfEngine().value(
        ScenarioDcfAssumptions(
            downside=_economic_dcf("0.01"),
            central=_economic_dcf("0.05"),
            upside=_economic_dcf("0.09"),
            downside_probability=D("0.20"),
            central_probability=D("0.50"),
            upside_probability=D("0.30"),
        )
    )

    assert result.method == ValuationMethod.SCENARIO_DCF
    assert result.metadata["probability_weighted"] is True
    assert result.downside_value_per_share < result.base_value_per_share < result.upside_value_per_share
    assert result.fair_value_per_share == result.base_value_per_share


def _rnpv(probability: str) -> BiotechRnpvAssumptions:
    return BiotechRnpvAssumptions(
        assets=[
            PipelineAsset(
                name="Asset A",
                years_to_launch=3,
                probability_of_approval=D(probability),
                launch_value=D("1000"),
                remaining_development_costs=[D("20"), D("20"), D("20")],
                evidence_ids=["PIT:E1"],
            )
        ],
        discount_rate=D("0.12"),
        net_cash=D("100"),
        diluted_shares=D("10"),
    )


def test_rnpv_returns_common_probability_adjusted_result() -> None:
    result = CommonRnpvEngine().value(
        RnpvScenarioSet(
            downside=_rnpv("0.20"),
            base=_rnpv("0.50"),
            upside=_rnpv("0.80"),
        )
    )

    assert result.method == ValuationMethod.RNPV
    assert result.enterprise_value is None
    assert result.downside_value_per_share < result.base_value_per_share < result.upside_value_per_share
    assert result.metadata["asset_count"] == 1


def _rim(roe: str) -> RimAssumptions:
    return RimAssumptions(
        book_equity=D("1000"),
        roe_path=[D(roe)] * 5,
        cost_of_equity=D("0.10"),
        payout_ratio=D("0.40"),
        terminal_roe=D(roe),
        terminal_growth=D("0.03"),
        diluted_shares=D("10"),
        provenance=["PIT:financial-snapshot"],
    )


def test_rim_returns_common_ordered_equity_result() -> None:
    result = CommonRimEngine().value(
        RimScenarioSet(downside=_rim("0.08"), base=_rim("0.12"), upside=_rim("0.16"))
    )

    assert result.method == ValuationMethod.RIM
    assert result.enterprise_value is None
    assert result.downside_value_per_share < result.base_value_per_share < result.upside_value_per_share
    assert result.fair_value_per_share == result.base_value_per_share


def test_nav_keeps_asset_haircuts_out_of_alpha_weights() -> None:
    result = NavEngine().value(
        NavAssumptions(
            assets=[
                NavAsset(
                    name="Property A",
                    base_value=D("1000"),
                    downside_haircut=D("0.25"),
                    upside_premium=D("0.10"),
                    evidence_ids=["E1"],
                )
            ],
            cash=D("100"),
            debt=D("300"),
            diluted_shares=D("10"),
        )
    )

    assert result.method == ValuationMethod.NAV
    assert result.base_value_per_share == D("80")
    assert result.downside_value_per_share == D("55")
    assert result.upside_value_per_share == D("90")


def _apv_case(fcff: str, shield: str) -> ApvCase:
    return ApvCase(
        unlevered_fcff=[D(fcff)] * 5,
        terminal_cash_flow=D(fcff),
        terminal_growth=D("0.02"),
        unlevered_cost_of_capital=D("0.10"),
        tax_shields=[D(shield)] * 5,
        tax_shield_discount_rate=D("0.06"),
    )


def test_apv_separates_financing_effects() -> None:
    result = ApvEngine().value(
        ApvAssumptions(
            downside=_apv_case("70", "5"),
            base=_apv_case("100", "10"),
            upside=_apv_case("130", "15"),
            debt=D("300"),
            cash=D("50"),
            diluted_shares=D("10"),
        )
    )
    assert result.method == ValuationMethod.APV
    assert result.enterprise_value is not None
    assert result.downside_value_per_share < result.base_value_per_share < result.upside_value_per_share


def _part(name: str, scope: str, value: str, basis: SotpValueBasis) -> SotpPart:
    base = D(value)
    return SotpPart(
        name=name,
        method=ValuationMethod.ECONOMIC_FCFF,
        value_basis=basis,
        downside_value=base * D("0.8"),
        base_value=base,
        upside_value=base * D("1.2"),
        included_cashflows=[scope],
        provenance=[f"PIT:{name}"],
    )


def test_sotp_rejects_duplicate_cashflow_scope() -> None:
    with pytest.raises(ValidationError, match="included by both"):
        SotpAssumptions(
            parts=[
                _part("Operating", "PRODUCT_A", "1000", SotpValueBasis.ENTERPRISE),
                _part("Pipeline", "PRODUCT_A", "500", SotpValueBasis.EQUITY),
            ],
            diluted_shares=D("10"),
        )


def test_sotp_aggregates_heterogeneous_parts_without_averaging_methods() -> None:
    result = SotpEngine().value(
        SotpAssumptions(
            parts=[
                _part("Operating", "PRODUCT_A", "1000", SotpValueBasis.ENTERPRISE),
                _part("Pipeline", "ASSET_B", "500", SotpValueBasis.EQUITY),
            ],
            parent_cash=D("100"),
            parent_debt=D("300"),
            diluted_shares=D("10"),
        )
    )
    assert result.method == ValuationMethod.SOTP
    assert result.base_value_per_share == D("130")
    assert result.metadata["part_count"] == 2
