from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data-lake/experiments/universal-value-150-20260820-v5"
READINESS = OUTPUT / "factor-readiness"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def test_return_blind_architecture_is_repeatable_without_changing_primary_rank() -> None:
    final = _json(OUTPUT / "FINAL-RESULT.json")
    policy = _json(OUTPUT / "preregistered-policy.json")
    coverage = final["coverage"]

    assert final["schema_version"] == "unified-value-architecture-calibration/4"
    assert final["verdict"] == "PASS"
    assert final["repeatability_pass"] is True
    assert final["return_data_accessed"] is False
    assert policy["return_inputs_forbidden"] is True
    assert policy["primary_ranking_policy_changed"] is False
    assert policy["broad_value_role"] == "COMPARISON_BASELINE_ONLY_NOT_PRIMARY_RANK"
    assert policy["trust_policy"] == {
        "contract_version": "valuation-trust/2",
        "max_warning_count": 3,
        "min_assumption_confidence": "0.50",
        "require_screening_eligible": True,
        "warning_count_basis": "STRUCTURED_TRUST_WARNINGS",
    }
    assert coverage["normalization_policy"]["reference_class_hierarchy"] == [
        "METHOD_ARCHETYPE",
        "METHOD",
        "MODEL_FAMILY",
    ]

    for name in ("routing.csv", "signals.csv", "coverage.json", "audit-contract.json"):
        assert (OUTPUT / "run-a" / name).read_bytes() == (OUTPUT / "run-b" / name).read_bytes()


def test_reason_fix_generates_scores_without_relaxing_trust_thresholds() -> None:
    coverage = _json(OUTPUT / "run-a/coverage.json")
    assert coverage["valuation_generated_count"] == 119
    assert coverage["trust_gate_pass_count"] == 59
    assert coverage["rank_eligible_count"] == 54
    assert coverage["pre_normalization_status_counts"] == {
        "INVALID_VALUATION": 48,
        "MODEL_NOT_APPLICABLE": 31,
        "UNTRUSTED_VALUATION": 12,
        "VALID": 59,
    }
    assert coverage["alpha_status_counts"] == {
        "INSUFFICIENT_REFERENCE_CLASS": 5,
        "INVALID_VALUATION": 48,
        "MODEL_NOT_APPLICABLE": 31,
        "UNTRUSTED_VALUATION": 12,
        "VALID": 54,
    }
    reasons = coverage["trust_reason_counts_by_status"]
    assert reasons["INVALID_VALUATION"]["NON_POSITIVE_FAIR_VALUE"] == 48
    assert reasons["UNTRUSTED_VALUATION"] == {"SCREENING_INELIGIBLE": 12}
    assert reasons["INSUFFICIENT_REFERENCE_CLASS"] == {"REFERENCE_CLASS_N_LT_20": 5}
    assert coverage["fallback_fcff_count"] == 0
    assert coverage["llm_call_count"] == 0


def test_disclosures_are_preserved_but_do_not_consume_warning_budget() -> None:
    signals = pd.read_csv(OUTPUT / "run-a/signals.csv", dtype={"ticker": str})
    economic = signals.loc[
        (signals["method"] == "ECONOMIC_FCFF") & signals["actual_engine"].notna()
    ]
    assert len(economic) == 78
    assert economic["valuation_disclosure_count"].eq(4).all()
    valid = signals.loc[signals["pre_normalization_status"] == "VALID"]
    assert not valid["trust_reason_codes"].fillna("").str.contains(
        "TOO_MANY_VALUATION_WARNINGS"
    ).any()
    invalid = signals.loc[signals["pre_normalization_status"] == "INVALID_VALUATION"]
    assert len(invalid) == 48
    assert invalid["primary_fair_value_per_share"].eq(0).all()
    assert invalid["trust_reason_codes"].str.contains("NON_POSITIVE_FAIR_VALUE").all()


def test_hierarchical_normalization_is_fail_closed_outside_model_family() -> None:
    signals = pd.read_csv(OUTPUT / "run-a/signals.csv", dtype={"ticker": str})
    ranked = signals.loc[signals["rank_eligible"] == 1]
    assert len(ranked) == 54
    assert ranked["unified_value_score"].between(0, 100).all()
    assert signals["unified_value_score"].notna().equals(signals["rank_eligible"].eq(1))

    local = ranked.loc[ranked["reference_class"] == "ECONOMIC_FCFF::GENERAL_OPERATING"]
    family = ranked.loc[
        ranked["reference_class"] == "MODEL_FAMILY::OPERATING_CASH_FLOW"
    ]
    assert len(local) == 44
    assert local["normalization_level"].eq("METHOD_ARCHETYPE").all()
    assert local["reference_class_size"].eq(44).all()
    assert len(family) == 10
    assert family["method"].eq("NORMALIZED_FCFF").all()
    assert family["normalization_level"].eq("MODEL_FAMILY").all()
    assert family["reference_class_size"].eq(54).all()
    assert family["normalization_fallback_used"].eq(1).all()
    assert not signals.loc[signals["method"].isin(["RIM", "RNPV"]), "rank_eligible"].any()


def test_factor_readiness_remains_not_ready_and_broad_value_is_only_a_baseline() -> None:
    readiness = _json(READINESS / "FACTOR-READINESS.json")
    assert readiness["schema_version"] == "universal-value-factor-readiness/2"
    assert readiness["factor_verdict"] == "NOT_READY_AS_UNIVERSAL_VALUE_FACTOR"
    assert readiness["rank_eligible_count"] == 54
    assert np.isclose(readiness["rank_eligible_coverage"], 0.36)
    assert readiness["broad_value_overlap_count"] == 54
    assert readiness["primary_ranking_policy_changed"] is False
    assert readiness["broad_value_role"] == "COMPARISON_BASELINE_ONLY_NOT_PRIMARY_RANK"
    assert readiness["factor_gate_failures"] == [
        "RANK_ELIGIBLE_COVERAGE_LT_60_PERCENT",
        "FEWER_THAN_3_SCORE_BEARING_REFERENCE_CLASSES",
        "NO_FORWARD_RETURN_FOR_2026_08_19_SIGNAL",
    ]
    expanded = next(
        row
        for row in readiness["universal_score_broad_value_correlations"]
        if row["comparison"] == "broad_value_expanded"
    )
    assert expanded["n"] == 54
    assert np.isclose(expanded["spearman"], 0.6832475700400228)

    report = (READINESS / "FINAL-REPORT.md").read_text(encoding="utf-8")
    assert "PER+PBR은 비교 baseline일 뿐 우선 랭킹으로 전환하지 않았고" in report
    assert "Universal Value Factor로 사용할 수 없습니다" in report
    assert (READINESS / "FAILURE-REASON-BREAKDOWN.csv").exists()
    assert (READINESS / "NORMALIZATION-CLASS-COVERAGE.csv").exists()
