from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path

import pytest

from moatrader.experiments import (
    FrozenExpectationGapContract,
    compute_contract_sha256,
    verify_frozen_sources,
)
from scripts.freeze_expectation_gap_contract import validate_engineering_coverage


def _payload(source_hashes: dict[str, str]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "expectation-gap-production-candidate/1",
        "frozen_on": "2026-08-18",
        "development_dates": ["2026-05-31"],
        "holdout_dates": ["2026-08-31"],
        "universe_sha256": "a" * 64,
        "universe_count": 150,
        "valuation_methods": ["ECONOMIC_FCFF", "RIM"],
        "router_contract_version": "valuation-router/1",
        "cheap_definition": "value / price - 1",
        "percentile_cohort": "date x method x archetype",
        "risk_policy": {"contract_version": "risk-overlay/1"},
        "legacy_composite_role": "DIAGNOSTIC_ONLY",
        "improving_role": "CONFIRMATION_ONLY",
        "sector_policy": "PIT_ONLY",
        "source_cutoff_policy": "AVAILABLE_AT_LTE_AS_OF",
        "signal_seal_required": True,
        "return_inputs_forbidden_before_signal_seal": True,
        "forward_return_calendar_days": 77,
        "maximum_sector_neutral_ic_sacrifice": 0.05,
        "minimum_worst_decile_improvement": 0.03,
        "minimum_downside_capture_improvement": 0.10,
        "frozen_source_sha256": source_hashes,
        "engineering_stability_sha256": {"signals.csv": "b" * 64},
        "engineering_return_data_accessed": False,
    }
    payload["contract_sha256"] = compute_contract_sha256(payload)
    return payload


def test_freeze_contract_rejects_seen_date_as_holdout() -> None:
    payload = _payload({})
    payload["holdout_dates"] = ["2026-05-31"]
    payload["contract_sha256"] = compute_contract_sha256(payload)
    with pytest.raises(ValueError, match="must not overlap"):
        FrozenExpectationGapContract.model_validate(payload)


def test_frozen_source_verification_detects_drift(tmp_path: Path) -> None:
    source = tmp_path / "alpha.py"
    source.write_text("frozen\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    contract = FrozenExpectationGapContract.model_validate(_payload({"alpha.py": digest}))
    verify_frozen_sources(contract, repository_root=tmp_path)

    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source verification failed"):
        verify_frozen_sources(contract, repository_root=tmp_path)


def _engineering_coverage() -> dict[str, object]:
    return {
        "schema_version": "expanded-valuation-signal-audit/4",
        "row_count": 600,
        "pit_sector_count": 600,
        "return_data_accessed": False,
        "fallback_fcff_count": 0,
        "llm_call_count": 0,
        "route_actual_engine_match_rate": 1.0,
        "architecture_gate_pass": True,
        "route_stability": 0.99,
        "normalization_policy": {
            "contract_version": "unified-value-normalization/1",
            "min_reference_class_size": 20,
            "small_class_action": "UNRANKABLE",
            "parent_class_fallback": False,
        },
        "valuation_generated_count": 500,
        "rank_eligible_count": 400,
        "actual_engine_counts": {
            "CommonEconomicFcffEngine": 180,
            "CommonNormalizedFcffEngine": 80,
            "ScenarioDcfEngine": 80,
            "CommonRimEngine": 45,
            "CommonRnpvEngine": 25,
            "ApvEngine": 25,
            "NavEngine": 35,
            "SotpEngine": 30,
        },
        "method_audit": {
            "ECONOMIC_FCFF": {"routed_count": 200, "eligible_route_count": 180, "valuation_generated_count": 180, "rank_eligible_count": 160, "max_reference_class_size": 160},
            "NORMALIZED_FCFF": {"routed_count": 100, "eligible_route_count": 80, "valuation_generated_count": 80, "rank_eligible_count": 60, "max_reference_class_size": 60},
            "SCENARIO_DCF": {"routed_count": 100, "eligible_route_count": 80, "valuation_generated_count": 80, "rank_eligible_count": 60, "max_reference_class_size": 60},
            "RIM": {"routed_count": 50, "eligible_route_count": 45, "valuation_generated_count": 45, "rank_eligible_count": 35, "max_reference_class_size": 35},
            "RNPV": {"routed_count": 30, "eligible_route_count": 25, "valuation_generated_count": 25, "rank_eligible_count": 20, "max_reference_class_size": 20},
            "APV": {"routed_count": 30, "eligible_route_count": 25, "valuation_generated_count": 25, "rank_eligible_count": 20, "max_reference_class_size": 20},
            "NAV": {"routed_count": 40, "eligible_route_count": 35, "valuation_generated_count": 35, "rank_eligible_count": 25, "max_reference_class_size": 25},
            "SOTP": {"routed_count": 50, "eligible_route_count": 30, "valuation_generated_count": 30, "rank_eligible_count": 20, "max_reference_class_size": 20},
        },
    }


def test_engineering_freeze_gate_accepts_real_multi_engine_zero_llm_audit() -> None:
    validate_engineering_coverage(_engineering_coverage())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": "expanded-valuation-signal-audit/3"}, "schema v4"),
        ({"fallback_fcff_count": 1}, "fallback FCFF"),
        ({"llm_call_count": 1}, "zero LLM"),
        ({"route_actual_engine_match_rate": 0.99}, "100% route-to-actual-engine"),
        ({"architecture_gate_pass": False}, "passing architecture"),
        (
            {
                "normalization_policy": {
                    "min_reference_class_size": 10,
                    "parent_class_fallback": False,
                }
            },
            "N=20",
        ),
        ({"actual_engine_counts": {"LegacyFcffCommonEngine": 485}}, "three distinct"),
        (
            {
                "actual_engine_counts": {"LegacyFcffCommonEngine": 460, "CommonRimEngine": 20, "NavEngine": 20}
            },
            "exceeding 90%",
        ),
        (
            {
                "method_audit": {
                    "SOTP": {"routed_count": 36, "eligible_route_count": 1, "valuation_generated_count": 0, "rank_eligible_count": 0}
                }
            },
            "eligible route execution gaps",
        ),
        (
            {"valuation_generated_count": 499},
            "generated counts must match",
        ),
        (
            {"rank_eligible_count": 399},
            "trusted counts must match",
        ),
        (
            {
                "method_audit": {
                    "RIM": {"routed_count": 10, "eligible_route_count": 8, "valuation_generated_count": 8, "rank_eligible_count": 9}
                }
            },
            "coverage ordering",
        ),
    ],
)
def test_engineering_freeze_gate_rejects_fake_or_dead_multi_model_audit(
    updates: dict[str, object],
    message: str,
) -> None:
    coverage = _engineering_coverage()
    coverage.update(updates)
    with pytest.raises(ValueError, match=message):
        validate_engineering_coverage(coverage)
