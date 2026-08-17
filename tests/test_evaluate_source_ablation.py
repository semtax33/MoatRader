from __future__ import annotations

from types import SimpleNamespace

import pytest

from scripts.evaluate_source_ablation import _treatment_effect_repeatability


def _company(score: float) -> SimpleNamespace:
    return SimpleNamespace(
        moat_score=SimpleNamespace(economic_moat_score=score),
    )


def test_treatment_effect_repeatability_compares_paired_deltas() -> None:
    dart = {"A": _company(1.0), "B": _company(2.0), "C": _company(3.0)}
    ir = {"A": _company(2.0), "B": _company(2.0), "C": _company(2.0)}
    dart_repeat = {
        "A": _company(1.5),
        "B": _company(2.0),
        "C": _company(2.5),
    }
    ir_repeat = {
        "A": _company(2.5),
        "B": _company(2.0),
        "C": _company(1.5),
    }

    result = _treatment_effect_repeatability(
        dart,
        ir,
        dart_repeat,
        ir_repeat,
    )

    assert result is not None
    assert result["company_count"] == 3
    assert result["treatment_delta_spearman"] == pytest.approx(1.0)
    assert result["exact_treatment_delta_match_rate"] == 1.0
    assert result["treatment_direction_match_rate"] == 1.0
    assert result["maximum_absolute_treatment_delta_difference"] == 0.0


def test_treatment_effect_repeatability_requires_both_repeat_lanes() -> None:
    assert _treatment_effect_repeatability({}, {}, None, {}) is None


def test_treatment_effect_repeatability_can_limit_to_compliant_tickers() -> None:
    dart = {"A": _company(1.0), "B": _company(2.0), "C": _company(3.0)}
    ir = {"A": _company(2.0), "B": _company(3.0), "C": _company(2.0)}
    dart_repeat = {"A": _company(1.0), "B": _company(2.0), "C": _company(3.0)}
    ir_repeat = {"A": _company(2.0), "B": _company(2.0), "C": _company(4.0)}

    result = _treatment_effect_repeatability(
        dart,
        ir,
        dart_repeat,
        ir_repeat,
        tickers={"A", "B"},
    )

    assert result is not None
    assert result["company_count"] == 2
    assert result["exact_treatment_delta_match_rate"] == 0.5
    assert result["treatment_direction_match_rate"] == 0.5
