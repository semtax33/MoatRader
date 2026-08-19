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
        "schema_version": "expanded-valuation-signal-audit/2",
        "row_count": 600,
        "pit_sector_count": 600,
        "return_data_accessed": False,
        "fallback_fcff_count": 0,
        "llm_call_count": 0,
        "valuation_generated_count": 500,
        "rank_eligible_count": 400,
        "actual_engine_counts": {"CommonEconomicFcffEngine": 300, "CommonRimEngine": 100, "NavEngine": 100},
        "method_audit": {
            "ECONOMIC_FCFF": {"routed_count": 400, "valuation_generated_count": 300, "rank_eligible_count": 250},
            "RIM": {"routed_count": 100, "valuation_generated_count": 100, "rank_eligible_count": 80},
            "NAV": {"routed_count": 100, "valuation_generated_count": 100, "rank_eligible_count": 70},
        },
    }


def test_engineering_freeze_gate_accepts_real_multi_engine_zero_llm_audit() -> None:
    validate_engineering_coverage(_engineering_coverage())


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"schema_version": "expanded-valuation-signal-audit/1"}, "schema v2"),
        ({"fallback_fcff_count": 1}, "fallback FCFF"),
        ({"llm_call_count": 1}, "zero LLM"),
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
                    "SOTP": {"routed_count": 36, "valuation_generated_count": 0, "rank_eligible_count": 0}
                }
            },
            "zero generated/trusted",
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
                    "RIM": {"routed_count": 10, "valuation_generated_count": 8, "rank_eligible_count": 9}
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
