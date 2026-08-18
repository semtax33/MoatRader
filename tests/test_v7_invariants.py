from __future__ import annotations

from moatrader.financial.invariants import run_reference_invariant_suite


def test_v7_reference_invariant_suite_runs_at_least_one_hundred_checks() -> None:
    report = run_reference_invariant_suite()

    assert len(report.checks) == 120
    assert report.passed_count == 120
    assert report.failed_count == 0
    assert report.return_data_accessed is False
    assert {item.invariant for item in report.checks} == {
        "FCFF_WACC_UP_FAIR_VALUE_DOWN",
        "RIM_EXCESS_ROE_UP_FAIR_VALUE_UP",
        "RNPV_SUCCESS_PROBABILITY_UP_VALUE_UP",
        "NAV_ASSET_VALUE_UP_EQUITY_VALUE_UP",
        "SOTP_PART_PLUS_100_OWNERSHIP_ADJUSTED",
        "APV_TAX_SHIELD_UP_VALUE_UP",
    }
