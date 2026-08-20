from __future__ import annotations

from datetime import date
from decimal import Decimal as D

import pytest
from pydantic import ValidationError

from moatrader.expectations import (
    CheapSignal,
    UnifiedValueNormalizationPolicy,
    assign_method_archetype_percentiles,
)
from moatrader.valuation import (
    ApplicabilityStatus,
    EconomicArchetype,
    ModelApplicability,
    ValuationMethod,
    ValuationProfile,
    ValuationProfileRouter,
    ValuationResult,
)


def _profile(**updates: object) -> ValuationProfile:
    values: dict[str, object] = {
        "issuer_id": "C1",
        "as_of": date(2026, 5, 31),
        "sector": "Industrials",
        "industry": "Machinery",
        "economic_archetype": EconomicArchetype.GENERAL_OPERATING,
        "revenue_positive": True,
        "ebit_positive": True,
        "fcf_positive": True,
        "available_data": [
            "revenue",
            "ebit",
            "invested_capital",
            "valuation_assumptions",
            "diluted_shares",
        ],
        "provenance": ["PIT:DART:2026-05-31"],
    }
    values.update(updates)
    return ValuationProfile(**values)


@pytest.mark.parametrize(
    ("profile", "method"),
    [
        (
            _profile(
                economic_archetype=EconomicArchetype.FINANCIAL_INTERMEDIARY,
                is_financial_intermediary=True,
                available_data=["book_equity", "net_income", "diluted_shares"],
            ),
            ValuationMethod.RIM,
        ),
        (
            _profile(
                economic_archetype=EconomicArchetype.PRE_REVENUE_BIOTECH,
                pipeline_assets_material=True,
                revenue_positive=False,
                ebit_positive=False,
                available_data=[
                    "pipeline_assets",
                    "clinical_phase",
                    "reference_pos",
                    "launch_value",
                    "development_costs",
                    "diluted_shares",
                ],
            ),
            ValuationMethod.RNPV,
        ),
        (
            _profile(
                economic_archetype=EconomicArchetype.MULTI_BUSINESS,
                multi_segment=True,
                segment_heterogeneity_material=True,
                available_data=[
                    "segment_submodel_inputs",
                    "valuation_basis",
                    "cashflow_scope_id",
                    "net_debt_scope_id",
                    "nci_scope_id",
                    "diluted_shares",
                ],
            ),
            ValuationMethod.SOTP,
        ),
        (
            _profile(
                economic_archetype=EconomicArchetype.ASSET_BACKED,
                is_reit=True,
                available_data=["asset_values", "debt", "diluted_shares"],
            ),
            ValuationMethod.NAV,
        ),
        (
            _profile(
                economic_archetype=EconomicArchetype.LOSS_MAKING_GROWTH,
                ebit_positive=False,
                persistent_loss=True,
                path_to_positive_unit_economics=True,
                available_data=[
                    "revenue",
                    "persistent_loss",
                        "path_to_positive_unit_economics",
                        "scenario_policy",
                        "scenario_assumptions",
                    "valuation_assumptions",
                    "diluted_shares",
                ],
            ),
            ValuationMethod.SCENARIO_DCF,
        ),
        (
            _profile(
                economic_archetype=EconomicArchetype.LEVERAGE_DRIVEN,
                leverage_path_material=True,
                available_data=[
                    "unlevered_cashflows",
                    "debt_schedule",
                    "tax_shields",
                    "diluted_shares",
                ],
            ),
            ValuationMethod.APV,
        ),
        (
            _profile(
                economic_archetype=EconomicArchetype.CYCLICAL_OPERATING,
                materially_cyclical=True,
                available_data=[
                    "history_5y",
                    "normalization_contract",
                    "base_invested_capital",
                    "diluted_shares",
                ],
            ),
            ValuationMethod.NORMALIZED_FCFF,
        ),
    ],
)
def test_router_uses_economic_structure_before_value(profile: ValuationProfile, method: ValuationMethod) -> None:
    route = ValuationProfileRouter().route(profile)
    assert route.primary_method == method
    assert route.applicability.status == ApplicabilityStatus.ELIGIBLE
    assert route.profile_sha256 == profile.fingerprint()


def test_router_fails_closed_when_method_specific_data_is_missing() -> None:
    route = ValuationProfileRouter().route(
        _profile(
            economic_archetype=EconomicArchetype.FINANCIAL_INTERMEDIARY,
            is_financial_intermediary=True,
            available_data=["book_equity"],
        )
    )
    assert route.primary_method == ValuationMethod.RIM
    assert route.applicability.status == ApplicabilityStatus.INSUFFICIENT_DATA
    assert route.applicability.missing_fields == ["diluted_shares", "net_income"]


def test_profile_contract_rejects_price_or_return_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        _profile(current_price=100, forward_return=0.2)


@pytest.mark.parametrize(
    "updates",
    [
        {
            "economic_archetype": EconomicArchetype.GENERAL_OPERATING,
            "pipeline_assets_material": True,
        },
        {
            "economic_archetype": EconomicArchetype.MULTI_BUSINESS,
            "multi_segment": False,
            "segment_heterogeneity_material": False,
        },
        {
            "economic_archetype": EconomicArchetype.LOSS_MAKING_GROWTH,
            "ebit_positive": False,
            "persistent_loss": False,
        },
        {
            "economic_archetype": EconomicArchetype.ASSET_BACKED,
            "asset_value_primary": False,
        },
    ],
)
def test_profile_rejects_archetype_flag_mismatch(updates: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _profile(**updates)


def test_structural_cyclical_route_survives_a_transient_ebit_loss() -> None:
    profile = _profile(
        economic_archetype=EconomicArchetype.CYCLICAL_OPERATING,
        ebit_positive=False,
        materially_cyclical=True,
        available_data=[
            "history_5y",
            "normalization_contract",
            "base_invested_capital",
            "diluted_shares",
        ],
    )

    assert ValuationProfileRouter().route(profile).primary_method == ValuationMethod.NORMALIZED_FCFF


def test_pipeline_candidate_fails_closed_to_rnpv_before_scenario() -> None:
    profile = _profile(
        economic_archetype=EconomicArchetype.PIPELINE_ADJUDICATION_REQUIRED,
        ebit_positive=False,
        persistent_loss=True,
        path_to_positive_unit_economics=True,
        pipeline_adjudication_required=True,
        available_data=["revenue", "diluted_shares"],
    )

    route = ValuationProfileRouter().route(profile)

    assert route.primary_method == ValuationMethod.RNPV
    assert route.applicability.status == ApplicabilityStatus.INSUFFICIENT_DATA
    assert "pipeline ownership" in route.rationale[0]


def _valuation(method: ValuationMethod, fair_value: str) -> ValuationResult:
    value = D(fair_value)
    return ValuationResult(
        method=method,
        applicability=ModelApplicability(
            method=method,
            status=ApplicabilityStatus.ELIGIBLE,
        ),
        equity_value=value * D("10"),
        fair_value_per_share=value,
        downside_value_per_share=value * D("0.8"),
        base_value_per_share=value,
        upside_value_per_share=value * D("1.2"),
        assumption_confidence=D("0.8"),
    )


def test_common_cheap_is_value_over_price_and_normalized_within_method_archetype() -> None:
    signals = [
        CheapSignal.from_valuation(
            valuation=_valuation(ValuationMethod.RIM, fair),
            economic_archetype=EconomicArchetype.FINANCIAL_INTERMEDIARY.value,
            market_price=D("100"),
        )
        for fair in ("90", "120", "150")
    ]
    normalized = assign_method_archetype_percentiles(
        signals,
        UnifiedValueNormalizationPolicy(min_reference_class_size=2),
    )

    assert [item.raw_value_gap for item in normalized] == [D("-0.1"), D("0.2"), D("0.5")]
    assert [item.method_archetype_percentile for item in normalized] == [0.0, 50.0, 100.0]
    assert [item.method_percentile for item in normalized] == [0.0, 50.0, 100.0]
    assert [item.unified_value_score for item in normalized] == [0.0, 50.0, 100.0]
    assert all(
        item.reference_class == "RIM::FINANCIAL_INTERMEDIARY" for item in normalized
    )
