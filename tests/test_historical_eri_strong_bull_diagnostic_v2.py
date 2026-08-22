from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from decimal import Decimal

import pytest

from moatrader.expectations.future_eri import (
    CurrentExpectationStateV1,
    EvidenceIndexFutureEriFeatureRowV2,
    FutureEriOutcomeInputV1,
    RealizedFcffStateV1,
    build_evidence_index_future_eri_label_v2,
    roll_forward_evidence_index_expectations_v2,
    seal_evidence_index_feature_dataset_v2,
    target_trading_session,
)
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from moatrader.valuation.economic_dcf import EconomicDcfEngine
from scripts.diagnose_historical_eri_strong_bull_v2 import (
    TOLERANCE,
    _group_summaries,
    decompose_eri_bridge_v2,
)


D = Decimal
KST = timezone(timedelta(hours=9))
SIGNAL = datetime(2024, 5, 16, 9, 0, tzinfo=KST)


def _assumptions() -> EconomicDcfAssumptions:
    return EconomicDcfAssumptions(
        base_period="2023FY",
        base_revenue=D("1000"),
        base_nopat_margin=D("0.10"),
        base_invested_capital=D("800"),
        revenue_growth=D("0.06"),
        target_nopat_margin=D("0.12"),
        roiic=D("0.20"),
        competitive_advantage_period_years=5,
        fade_years=4,
        explicit_forecast_years=10,
        stable_growth=D("0.02"),
        stable_nopat_margin=D("0.10"),
        stable_roic=D("0.12"),
        wacc=D("0.09"),
        net_debt=D("100"),
        diluted_shares=D("10"),
    )


def _fixture():
    assumptions = _assumptions()
    fitted_price = EconomicDcfEngine().value(assumptions).fair_value_per_share
    feature = EvidenceIndexFutureEriFeatureRowV2(
        observation_id="OBS_TEST_DECOMPOSITION",
        issuer_id="000001",
        signal_timestamp=SIGNAL,
        full_evidence_index=D("1"),
        full_nobs=4,
        core_evidence_index=D("0.5"),
        core_nobs=3,
        full_index_row_sha256="a" * 64,
        core_index_row_sha256="b" * 64,
        full_index_seal_sha256="c" * 64,
        expectation_state=CurrentExpectationStateV1(
            issuer_id="000001",
            signal_timestamp=SIGNAL,
            market_price=fitted_price,
            market_price_at=SIGNAL,
            market_price_source_id="TEST:SIGNAL:PRICE",
            implied_growth=assumptions.revenue_growth,
            implied_margin=assumptions.target_nopat_margin,
            implied_roiic=assumptions.roiic,
            implied_cap_years=D(assumptions.competitive_advantage_period_years),
            reverse_dcf_method="TEST_REVERSE_DCF",
            reverse_dcf_input_sha256="d" * 64,
        ),
        frozen_expectation_assumptions=assumptions,
    )
    sessions = [SIGNAL.date() + timedelta(days=offset) for offset in range(100)]
    target = target_trading_session(SIGNAL.date(), sessions, horizon=63)
    realized = RealizedFcffStateV1(
        available_at=datetime.combine(target, time(8), tzinfo=KST),
        base_period="2024FY",
        base_revenue=D("1080"),
        base_nopat_margin=D("0.11"),
        base_invested_capital=D("850"),
        net_debt=D("90"),
        diluted_shares=D("10"),
        wacc=D("0.10"),
        wacc_source_id="TEST:TARGET:WACC",
        source_document_ids=["TEST:REALIZED"],
    )
    provisional = FutureEriOutcomeInputV1(
        observation_id=feature.observation_id,
        target_session=target,
        target_price_at=datetime.combine(target, time(15, 30), tzinfo=KST),
        actual_market_price=D("1"),
        target_price_source_id="TEST:TARGET:PRICE",
        realized_state=realized,
    )
    rolled = roll_forward_evidence_index_expectations_v2(feature, provisional)
    counterfactual = EconomicDcfEngine().value(rolled).fair_value_per_share
    outcome = provisional.model_copy(
        update={"actual_market_price": counterfactual * D("1.10")}
    )
    seal = seal_evidence_index_feature_dataset_v2(
        [feature],
        sealed_at=SIGNAL + timedelta(minutes=1),
        full_index_seal_sha256="c" * 64,
    )
    label = build_evidence_index_future_eri_label_v2(
        feature=feature,
        outcome=outcome,
        feature_seal=seal,
        trading_sessions=sessions,
    )
    return feature, outcome, label


def test_decomposition_is_exact_and_preserves_the_sealed_future_eri() -> None:
    feature, outcome, label = _fixture()
    result = decompose_eri_bridge_v2(
        feature=feature,
        outcome=outcome,
        label=label,
    )
    total = (
        result["realization_component"]
        + result["discount_rate_component"]
        + result["expectation_revision_component"]
    )
    assert abs(total - result["total_log_price_bridge"]) <= TOLERANCE
    assert (
        abs(
            result["signal_reverse_fit_log_gap"]
            + result["total_log_price_bridge"]
            - result["market_total_log_price_bridge"]
        )
        <= TOLERANCE
    )
    assert result["expectation_revision_component"] == label.future_eri
    assert result["expectation_revision_component"] == D("1.10").ln()
    assert result["discount_rate_component"] < 0
    assert abs(result["signal_reverse_fit_log_gap"]) <= TOLERANCE


def test_group_summary_reports_requested_quantiles_and_small_cells() -> None:
    base = {
        "issuer_id": "A",
        "full_evidence_band": "STRONG_BULL",
        **{
            field: str(index / 100)
            for index, field in enumerate(
                (
                    "realization_component",
                    "discount_rate_component",
                    "expectation_revision_component",
                    "total_log_price_bridge",
                    "market_total_log_price_bridge",
                    "enterprise_realization_component",
                    "enterprise_discount_rate_component",
                    "enterprise_expectation_revision_component",
                    "enterprise_total_log_bridge",
                    "capital_structure_bridge_effect",
                    "signal_reverse_fit_log_gap",
                    "wacc_change",
                    "realized_minus_signal_nopat_margin",
                    "log_realized_to_signal_revenue",
                )
            )
        },
    }
    rows = [dict(base), {**base, "issuer_id": "B", "expectation_revision_component": "-0.1"}]
    summary = _group_summaries(rows, dimensions=("full_evidence_band",))[0]
    assert summary["observation_count"] == 2
    assert summary["issuer_count"] == 2
    assert summary["small_cell_less_than_20"] is True
    assert summary["negative_expectation_revision_share"] == pytest.approx(0.5)
    assert summary["component_statistics"]["expectation_revision_component"]["p50"] == pytest.approx(-0.04)
