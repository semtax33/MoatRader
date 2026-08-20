from decimal import Decimal

import pytest

from moatrader.valuation import (
    CommonRimEngine,
    RimBuildInput,
    RimBuilder,
    ValuationMethod,
)


D = Decimal


def source() -> RimBuildInput:
    refs = ["DART:ANNUAL", "DART:HALF"]
    return RimBuildInput(
        issuer_id="BANK1",
        as_of="2026-08-18",
        book_equity=D("1000"),
        prior_fy_net_income=D("100"),
        current_ytd_net_income=D("60"),
        prior_ytd_net_income=D("50"),
        diluted_shares=D("10"),
        size_bucket="LARGE",
        evidence_available_at={refs[0]: "2026-03-20", refs[1]: "2026-08-14"},
        provenance=refs,
    )


def test_rim_builder_uses_ttm_and_executes_ordered_scenarios() -> None:
    build_input = source()
    assumptions = RimBuilder().build(build_input)
    result = CommonRimEngine().value(assumptions)

    assert build_input.ttm_net_income == D("110")
    assert assumptions.base.cost_of_equity == D("0.09")
    assert result.method == ValuationMethod.RIM
    assert result.downside_value_per_share < result.base_value_per_share
    assert result.base_value_per_share < result.upside_value_per_share
    assert "NO_LLM:DETERMINISTIC_BUILDER" in result.provenance


def test_rim_build_input_rejects_future_evidence() -> None:
    payload = source().model_dump(mode="json")
    payload["evidence_available_at"]["DART:HALF"] = "2026-08-19"

    with pytest.raises(ValueError, match="future-dated"):
        RimBuildInput.model_validate(payload)
