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
