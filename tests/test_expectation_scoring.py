from __future__ import annotations

import pytest

from moatrader.expectations.scoring import (
    ExpectationScoreStatus,
    FragilityComponents,
    ThreeAxisPercentiles,
    average_tie_percentiles,
    build_three_axis_score,
    weighted_geometric_score,
)


def _fragility(**updates: float) -> FragilityComponents:
    values = {
        "wacc_sensitivity": 20.0,
        "terminal_growth_sensitivity": 20.0,
        "scenario_dispersion": 20.0,
        "terminal_value_share": 20.0,
        "single_driver_dependence": 20.0,
        "evidence_weakness": 20.0,
    }
    values.update(updates)
    return FragilityComponents(**values)


def test_fixed_three_axis_weights_and_geometric_composite() -> None:
    inputs = ThreeAxisPercentiles(
        expectation_gap=80,
        probable_mos=60,
        plausible_mos=40,
        probable_value_revision=70,
        plausible_value_revision=50,
        driver_breadth=90,
        evidence_revision=30,
    )
    result = build_three_axis_score(inputs, _fragility())
    assert result.cheap == pytest.approx(66.0)
    assert result.improving == pytest.approx(64.0)
    assert result.non_fragile == pytest.approx(80.0)
    assert result.composite == pytest.approx(weighted_geometric_score(66, 64, 80))
    assert result.status == ExpectationScoreStatus.VALID
    assert result.diagnostic_only
    assert not result.rank_eligible


def test_missing_revision_is_insufficient_evidence_not_a_zero() -> None:
    result = build_three_axis_score(
        ThreeAxisPercentiles(expectation_gap=50, probable_mos=50, plausible_mos=50),
        _fragility(),
    )
    assert result.status == ExpectationScoreStatus.INSUFFICIENT_EVIDENCE
    assert result.improving is None
    assert result.composite is None
    assert not result.rank_eligible


def test_non_fragile_gate_is_separate_from_score() -> None:
    inputs = ThreeAxisPercentiles(
        expectation_gap=80,
        probable_mos=80,
        plausible_mos=80,
        probable_value_revision=80,
        plausible_value_revision=80,
        driver_breadth=80,
        evidence_revision=80,
    )
    result = build_three_axis_score(
        inputs,
        FragilityComponents(
            wacc_sensitivity=90,
            terminal_growth_sensitivity=90,
            scenario_dispersion=90,
            terminal_value_share=90,
            single_driver_dependence=90,
            evidence_weakness=90,
        ),
    )
    assert result.non_fragile == pytest.approx(10)
    assert result.composite is not None
    assert result.status == ExpectationScoreStatus.HIGH_FRAGILITY
    assert not result.rank_eligible


def test_average_tie_percentiles_are_order_invariant() -> None:
    assert average_tie_percentiles([10, 20, 20, 40]) == pytest.approx(
        [0.0, 50.0, 50.0, 100.0]
    )
