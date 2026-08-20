from datetime import date
from decimal import Decimal

import pytest

from moatrader.valuation import (
    CyclePhase,
    EconomicArchetype,
    ExecutionStatus,
    NormalizationContract,
    NormalizedAnnualObservation,
    NormalizedFcffBuildInput,
    NormalizedFcffBuilder,
    NormalizedFcffEngine,
    RoutedValuationExecutor,
    RoutedValuationInput,
    ValuationMethod,
    ValuationProfile,
    ValuationProfileRouter,
    infer_cycle_phase,
)


D = Decimal


def observations() -> list[NormalizedAnnualObservation]:
    values = [
        (2018, "800", "40"),
        (2019, "900", "72"),
        (2020, "700", "-14"),
        (2021, "1100", "165"),
        (2022, "1250", "225"),
        (2023, "1000", "90"),
        (2024, "1050", "84"),
    ]
    return [
        NormalizedAnnualObservation(
            fiscal_year=year,
            revenue=D(revenue),
            ebit=D(ebit),
            source_refs=[f"DART:{year}"],
        )
        for year, revenue, ebit in values
    ]


def build_input() -> NormalizedFcffBuildInput:
    history = observations()
    current_margin = D("80") / D("1000")
    phase = infer_cycle_phase(history, current_ebit_margin=current_margin)
    return NormalizedFcffBuildInput(
        issuer_id="000001",
        as_of="2025-08-31",
        observations=history,
        normalization=NormalizationContract(
            included_fiscal_years=[item.fiscal_year for item in history],
            excluded_fiscal_years=[],
            cycle_phase=phase,
        ),
        base_period="2025H1_TTM",
        base_revenue=D("1000"),
        base_ebit=D("80"),
        base_invested_capital=D("700"),
        wacc=D("0.105"),
        net_debt=D("100"),
        diluted_shares=D("10"),
        provenance=["PIT:DART:000001"],
    )


def test_normalized_builder_freezes_history_and_runs_real_engine() -> None:
    assumptions = NormalizedFcffBuilder().build(build_input())
    result = NormalizedFcffEngine().value(assumptions)

    assert result.method == ValuationMethod.NORMALIZED_FCFF
    assert result.downside_value_per_share <= result.base_value_per_share
    assert result.base_value_per_share <= result.upside_value_per_share
    assert result.fair_value_per_share == result.base_value_per_share
    assert result.metadata["normalization_policy"]["window_years"] == 7
    assert result.metadata["normalization_policy"]["included_fiscal_years"] == list(
        range(2018, 2025)
    )
    assert "NO_LLM:DETERMINISTIC_BUILDER" in result.provenance
    assert "SCENARIO_LABEL:INTRINSIC_VALUE_ORDER_NO_MARKET_PRICE" in result.provenance


def test_normalized_routed_input_executes_normalized_engine() -> None:
    build = build_input()
    assumptions = NormalizedFcffBuilder().build(build)
    profile = ValuationProfile(
        issuer_id="000001",
        as_of=date(2025, 8, 31),
        sector="Chemicals",
        industry="Cyclical",
        economic_archetype=EconomicArchetype.CYCLICAL_OPERATING,
        revenue_positive=True,
        ebit_positive=True,
        materially_cyclical=True,
        available_data=[
            "history_5y",
            "normalization_contract",
            "base_invested_capital",
            "diluted_shares",
        ],
        provenance=["PIT:DART:000001"],
    )
    route = ValuationProfileRouter().route(profile)
    prepared = RoutedValuationExecutor.prepare(
        RoutedValuationInput(
            issuer_id="000001",
            as_of=date(2025, 8, 31),
            method=ValuationMethod.NORMALIZED_FCFF,
            assumptions=assumptions.model_dump(mode="json"),
            source_refs=["PIT:DART:000001", "POLICY:normalized-fcff-policy/1"],
        )
    )
    execution = RoutedValuationExecutor().execute(route, prepared)

    assert execution.status == ExecutionStatus.VALUED
    assert execution.actual_engine == "NormalizedFcffEngine"
    assert execution.valuation is not None
    assert execution.valuation.method == ValuationMethod.NORMALIZED_FCFF


def test_normalization_contract_rejects_short_or_mismatched_history() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        NormalizationContract(
            included_fiscal_years=[2021, 2022, 2023, 2024],
            cycle_phase=CyclePhase.MID_CYCLE,
        )

    source = build_input().model_copy(
        update={
            "normalization": NormalizationContract(
                included_fiscal_years=list(range(2017, 2024)),
                cycle_phase=CyclePhase.MID_CYCLE,
            )
        }
    )
    with pytest.raises(ValueError, match="observation years"):
        NormalizedFcffBuildInput.model_validate(source.model_dump(mode="json"))


def test_cycle_phase_cannot_be_overridden_after_seeing_outputs() -> None:
    source = build_input()
    wrong = source.model_copy(
        update={
            "normalization": source.normalization.model_copy(
                update={"cycle_phase": CyclePhase.PEAK}
            )
        }
    )
    if source.normalization.cycle_phase == CyclePhase.PEAK:
        wrong = source.model_copy(
            update={
                "normalization": source.normalization.model_copy(
                    update={"cycle_phase": CyclePhase.TROUGH}
                )
            }
        )
    with pytest.raises(ValueError, match="cycle phase"):
        NormalizedFcffBuilder().build(wrong)
