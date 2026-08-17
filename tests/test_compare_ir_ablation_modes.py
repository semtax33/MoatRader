from __future__ import annotations

from scripts.compare_ir_ablation_modes import compare


def _report(*, longitudinal: bool) -> dict[str, object]:
    n = 10
    return {
        "company_count": n,
        "paired": {
            "ir_usable_company_count": 9,
            "treatment_compliant_company_count": 6,
            "longitudinal_treatment_compliant_company_count": (
                5 if longitudinal else 6
            ),
            "multi_year_accepted_ir_company_count": 5 if longitudinal else 0,
            "evidence_sufficiency_increase_count": 5,
            "mechanism_coverage_increase_count": 4,
            "outcome_coverage_increase_count": 4,
            "persistence_coverage_increase_count": 4 if longitudinal else 1,
            "counterevidence_increase_count": 2,
            "score_increase_count": 2,
            "score_decrease_count": 1,
            "noncompliant_score_change_count": 0,
        },
        "repeatability": {
            "dart_plus_ir": {
                "score_spearman": 0.70 if longitudinal else 0.60,
                "exact_score_match_rate": 0.80,
            },
            "treatment_effect_compliant_in_both": {
                "company_count": 5,
                "treatment_delta_spearman": 0.60 if longitudinal else 0.20,
                "exact_treatment_delta_match_rate": 0.80,
                "treatment_direction_match_rate": 0.80,
            },
        },
    }


def test_longitudinal_ir_comparison_uses_normalized_frozen_criteria() -> None:
    result = compare(_report(longitudinal=False), _report(longitudinal=True))

    assert result["return_data_used"] is False
    assert result["longitudinal_ir"]["multi_year_accepted_ir_company_rate"] == 0.5
    assert result["pre_registered_diagnostic_criteria"][
        "persistence_coverage_rate_improved"
    ] is True
    assert result["sensor_connection_supported"] is True
