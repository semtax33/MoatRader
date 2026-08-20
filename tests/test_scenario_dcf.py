from decimal import Decimal

import pytest

from moatrader.valuation import (
    ScenarioAnnualObservation,
    ScenarioDcfAssumptions,
    ScenarioDcfBuildInput,
    ScenarioDcfBuilder,
    ScenarioDcfEngine,
    ValuationMethod,
)


D = Decimal


def build_source() -> ScenarioDcfBuildInput:
    return ScenarioDcfBuildInput(
        issuer_id="C1",
        as_of="2025-08-31",
        observations=[
            ScenarioAnnualObservation(
                fiscal_year=2022,
                revenue=D("700"),
                ebit=D("-140"),
                source_refs=["PIT:2022"],
            ),
            ScenarioAnnualObservation(
                fiscal_year=2023,
                revenue=D("850"),
                ebit=D("-150"),
                source_refs=["PIT:2023"],
            ),
            ScenarioAnnualObservation(
                fiscal_year=2024,
                revenue=D("1000"),
                ebit=D("-120"),
                source_refs=["PIT:2024"],
            ),
        ],
        base_period="2025H1_TTM",
        base_revenue=D("1150"),
        base_ebit=D("-100"),
        base_invested_capital=D("900"),
        wacc=D("0.12"),
        net_debt=D("100"),
        diluted_shares=D("10"),
        recovery_evidence=["PIT:MARGIN_IMPROVEMENT"],
        provenance=["PIT:DART:C1"],
    )


def test_builder_executes_real_engine_and_central_is_rank_value() -> None:
    assumptions = ScenarioDcfBuilder().build(build_source())
    result = ScenarioDcfEngine().value(assumptions)

    assert result.method == ValuationMethod.SCENARIO_DCF
    assert result.fair_value_per_share == result.base_value_per_share
    assert result.metadata["ranking_basis"] == "CENTRAL_UNWEIGHTED"
    assert result.metadata["probability_weighted"] is False
    assert result.downside_value_per_share <= result.base_value_per_share
    assert result.base_value_per_share <= result.upside_value_per_share
    assert "NO_LLM:DETERMINISTIC_BUILDER" in result.provenance


def test_probabilities_are_frozen_and_cannot_be_tuned() -> None:
    assumptions = ScenarioDcfBuilder().build(build_source())
    payload = assumptions.model_dump(mode="json")
    payload["downside_probability"] = "0.20"
    payload["upside_probability"] = "0.30"

    with pytest.raises(ValueError, match="frozen at 25/50/25"):
        ScenarioDcfAssumptions.model_validate(payload)


def test_builder_rejects_transient_loss_or_missing_recovery() -> None:
    payload = build_source().model_dump(mode="json")
    payload["observations"][-1]["ebit"] = "10"
    with pytest.raises(ValueError, match="persistent annual losses"):
        ScenarioDcfBuildInput.model_validate(payload)

    payload = build_source().model_dump(mode="json")
    payload["base_ebit"] = "-200"
    with pytest.raises(ValueError, match="improving current operating margin"):
        ScenarioDcfBuildInput.model_validate(payload)
