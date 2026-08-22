from __future__ import annotations

import pytest

from scripts.audit_historical_eri_eligibility_bridge_v2 import (
    PRECHECK_REASON_PRIORITY,
    T63_REASON_PRIORITY,
    _coverage_rows,
    _issuer_concentration,
    _primary_reason,
    _reverse_model_status,
    _stage_bridge,
)


def _row(
    observation_id: str,
    issuer_id: str,
    sector: str,
    stages: tuple[bool, bool, bool, bool, bool],
) -> dict[str, object]:
    price, reverse, t63, valid, final = stages
    return {
        "observation_id": observation_id,
        "issuer_id": issuer_id,
        "sector": sector,
        "evidence_eligible": True,
        "price_pit_available": price,
        "reverse_valuation_available": reverse,
        "t63_snapshot_available": t63,
        "eri_decomposition_valid": valid,
        "final_common": final,
        "primary_exclusion_stage": None,
        "primary_exclusion_reason": None,
        "all_exclusion_reasons": [],
    }


def test_primary_reason_uses_upstream_priority_and_requires_a_reason() -> None:
    reasons = ["FEWER_THAN_TWO_VALID_PIT_ANNUALS", "NO_EXACT_SIGNAL_OPEN_PRICE"]
    assert (
        _primary_reason(reasons, PRECHECK_REASON_PRIORITY)
        == "NO_EXACT_SIGNAL_OPEN_PRICE"
    )
    with pytest.raises(ValueError, match="at least one"):
        _primary_reason([], PRECHECK_REASON_PRIORITY)
    assert (
        _primary_reason(
            [
                "MISSING_OUTCOME_ELIGIBILITY_INVENTORY",
                "MISSING_EXACT_T_PLUS_63_SESSION",
            ],
            T63_REASON_PRIORITY,
        )
        == "MISSING_EXACT_T_PLUS_63_SESSION"
    )


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        (
            "REVERSE_DCF_ERROR:ValueError:REVERSE_DCF_CENSORED_HIGH",
            "REVERSE_DCF_CENSORED_HIGH",
        ),
        (
            "REVERSE_DCF_ERROR:ValueError:base valuation must have positive equity value",
            "REVERSE_DCF_BASE_VALUATION_NON_POSITIVE_EQUITY",
        ),
    ],
)
def test_reverse_model_status_is_stable(reason: str, expected: str) -> None:
    assert _reverse_model_status(reason) == expected


def test_bridge_and_dimension_coverage_are_additive() -> None:
    ledger = [
        _row("o1", "a", "TECH", (True, True, True, True, True)),
        _row("o2", "a", "TECH", (True, True, True, False, False)),
        _row("o3", "b", "BANK", (True, True, False, False, False)),
        _row("o4", "c", "BANK", (False, False, False, False, False)),
    ]
    ledger[1]["primary_exclusion_stage"] = "T63_SNAPSHOT_TO_ERI_DECOMPOSITION"
    ledger[1]["primary_exclusion_reason"] = "INVALID_ERI_ENTERPRISE_VALUE"
    ledger[1]["all_exclusion_reasons"] = ["INVALID_ERI_ENTERPRISE_VALUE"]
    bridge = _stage_bridge(ledger)
    assert [row["observation_count"] for row in bridge] == [4, 3, 3, 2, 1, 1]
    assert [row["loss_from_previous_stage"] for row in bridge] == [0, 1, 0, 1, 1, 0]
    assert bridge[-1]["issuer_count"] == 1

    coverage = _coverage_rows(ledger, dimension="sector")
    tech = next(row for row in coverage if row["dimension_value"] == "TECH")
    bank = next(row for row in coverage if row["dimension_value"] == "BANK")
    assert tech["evidence_eligible_count"] == 2
    assert tech["final_common_count"] == 1
    assert tech["lost_at_eri_decomposition_count"] == 1
    assert tech["primary_exclusion_reason_counts"] == {
        "INVALID_ERI_ENTERPRISE_VALUE": 1
    }
    assert bank["lost_before_price_pit_count"] == 1
    assert bank["lost_at_t63_snapshot_count"] == 1
    assert sum(row["final_common_count"] for row in coverage) == 1


def test_issuer_concentration_uses_observation_shares() -> None:
    ledger = [
        _row("o1", "a", "TECH", (True, True, True, True, True)),
        _row("o2", "a", "TECH", (True, True, True, True, True)),
        _row("o3", "b", "BANK", (True, True, True, True, True)),
    ]
    result = _issuer_concentration(ledger, stage="final_common")
    assert result["issuer_count"] == 2
    assert result["largest_issuer_share"] == pytest.approx(2 / 3)
    assert result["issuer_hhi"] == pytest.approx(5 / 9)
