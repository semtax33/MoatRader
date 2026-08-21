from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal

from moatrader.canonical.models import StatementType
from moatrader.expectations.future_eri import (
    CurrentExpectationStateV1,
    EvidenceObservation,
    EvidenceState,
    FutureEriFeatureRowV1,
    FutureEriOutcomeInputV1,
    OperatingEvidenceAxis,
    RealizedFcffStateV1,
    build_fcff_evidence_vector,
    build_future_eri_label,
    roll_forward_frozen_expectations,
    seal_feature_dataset,
    target_trading_session,
)
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from moatrader.valuation.economic_dcf import EconomicDcfEngine


D = Decimal
KST = timezone(timedelta(hours=9))
SIGNAL_AT = datetime(2024, 5, 16, 9, 0, tzinfo=KST)
PRIOR_SIGNAL_AT = datetime(2024, 2, 15, 9, 0, tzinfo=KST)
TOLERANCE = D("1e-24")


def _observation(
    axis: OperatingEvidenceAxis,
    state: EvidenceState,
    *,
    current: bool,
) -> EvidenceObservation:
    available = (
        datetime(2024, 5, 15, 17, 30, tzinfo=KST)
        if current
        else datetime(2024, 2, 14, 12, 0, tzinfo=KST)
    )
    return EvidenceObservation(
        observation_id=f"NULL:{'CURRENT' if current else 'PRIOR'}:{axis.value}",
        issuer_id="NULL_FIXTURE_ISSUER",
        fiscal_period="2024Q1" if current else "2023Q4",
        axis=axis,
        state=state,
        source_document_id=f"NULL:DART:{'2024Q1' if current else '2023Q4'}",
        source_span=f"{axis.value} deterministic null fixture",
        source_published_at=available - timedelta(minutes=5),
        available_at=available,
        signal_timestamp=SIGNAL_AT if current else PRIOR_SIGNAL_AT,
        statement_type=StatementType.DISCLOSED_FACT,
        classification_rule_id="PRODUCTION_NULL_FIXTURE_V1",
        materiality_rule_id="QUALITATIVE_AXIS_STATE_EQUAL_UNIT_V1",
        confidence=D(1),
        materiality=D(1),
    )


def _assumptions() -> EconomicDcfAssumptions:
    return EconomicDcfAssumptions(
        base_period="2024Q1",
        base_revenue=D("1000"),
        base_nopat_margin=D("0.10"),
        base_invested_capital=D("800"),
        revenue_growth=D("0.08"),
        target_nopat_margin=D("0.12"),
        roiic=D("0.20"),
        competitive_advantage_period_years=6,
        fade_years=4,
        explicit_forecast_years=10,
        stable_growth=D("0.025"),
        stable_nopat_margin=D("0.10"),
        stable_roic=D("0.12"),
        wacc=D("0.09"),
        net_debt=D("100"),
        diluted_shares=D("10"),
    )


def _feature() -> FutureEriFeatureRowV1:
    current = [
        _observation(axis, EvidenceState.STABLE, current=True)
        for axis in OperatingEvidenceAxis
    ]
    prior = [
        _observation(axis, EvidenceState.STABLE, current=False)
        for axis in OperatingEvidenceAxis
    ]
    assumptions = _assumptions()
    return FutureEriFeatureRowV1(
        observation_id="NULL_FIXTURE_OBSERVATION",
        evidence=build_fcff_evidence_vector(
            issuer_id="NULL_FIXTURE_ISSUER",
            signal_timestamp=SIGNAL_AT,
            current=current,
            prior=prior,
        ),
        expectation_state=CurrentExpectationStateV1(
            issuer_id="NULL_FIXTURE_ISSUER",
            signal_timestamp=SIGNAL_AT,
            market_price=D("100"),
            market_price_at=SIGNAL_AT,
            market_price_source_id="NULL:PRICE:T",
            implied_growth=assumptions.revenue_growth,
            implied_margin=assumptions.target_nopat_margin,
            implied_roiic=assumptions.roiic,
            implied_cap_years=D(assumptions.competitive_advantage_period_years),
            reverse_dcf_method="FCFF_FROZEN_PATH_V1",
            reverse_dcf_input_sha256="0" * 64,
        ),
        frozen_expectation_assumptions=assumptions,
    )


def _sessions() -> list[date]:
    return [SIGNAL_AT.date() + timedelta(days=offset) for offset in range(100)]


def _outcome(feature: FutureEriFeatureRowV1, realized: RealizedFcffStateV1) -> FutureEriOutcomeInputV1:
    target = target_trading_session(feature.evidence.signal_timestamp.date(), _sessions())
    return FutureEriOutcomeInputV1(
        observation_id=feature.observation_id,
        target_session=target,
        target_price_at=datetime.combine(target, time(15, 30), tzinfo=KST),
        actual_market_price=D(1),
        target_price_source_id="NULL:PRICE:T_PLUS_63",
        realized_state=realized,
    )


def _evaluate_case(case_id: str, realized: RealizedFcffStateV1) -> dict[str, object]:
    feature = _feature()
    outcome = _outcome(feature, realized)
    rolled = roll_forward_frozen_expectations(feature, outcome)
    explained_price = EconomicDcfEngine().value(rolled).fair_value_per_share
    outcome = outcome.model_copy(update={"actual_market_price": explained_price})
    seal = seal_feature_dataset([feature], sealed_at=SIGNAL_AT + timedelta(minutes=1))
    label = build_future_eri_label(
        feature=feature,
        outcome=outcome,
        feature_seal=seal,
        trading_sessions=_sessions(),
    )
    passed = (
        abs(label.future_eri) <= TOLERANCE
        and abs(label.enterprise_future_eri) <= TOLERANCE
        and abs(label.capital_structure_bridge_effect) <= TOLERANCE
    )
    return {
        "case_id": case_id,
        "passed": passed,
        "future_eri": str(label.future_eri),
        "enterprise_future_eri": str(label.enterprise_future_eri),
        "capital_structure_bridge_effect": str(label.capital_structure_bridge_effect),
        "horizon_trading_days": label.horizon_trading_days,
        "production_label_path_used": True,
    }


def run_production_eri_null_fixtures() -> dict[str, object]:
    assumptions = _assumptions()
    feature = _feature()
    target = target_trading_session(feature.evidence.signal_timestamp.date(), _sessions())
    available_at = datetime.combine(target, time(8), tzinfo=KST)

    def realized(
        case_id: str,
        *,
        revenue: Decimal = assumptions.base_revenue,
        margin: Decimal = assumptions.base_nopat_margin,
        capital: Decimal = assumptions.base_invested_capital,
        net_debt: Decimal = assumptions.net_debt,
        shares: Decimal = assumptions.diluted_shares,
        wacc: Decimal = assumptions.wacc,
    ) -> RealizedFcffStateV1:
        return RealizedFcffStateV1(
            available_at=available_at,
            base_period=assumptions.base_period or "2024Q1",
            base_revenue=revenue,
            base_nopat_margin=margin,
            base_invested_capital=capital,
            net_debt=net_debt,
            diluted_shares=shares,
            wacc=wacc,
            wacc_source_id=f"NULL:{case_id}:WACC",
            source_document_ids=[f"NULL:{case_id}:REALIZED"],
        )

    cases = [
        _evaluate_case("UNCHANGED_PRICE_AND_EXPECTATIONS", realized("A")),
        _evaluate_case("WACC_ONLY", realized("B", wacc=D("0.07"))),
        _evaluate_case(
            "EXPECTED_FUNDAMENTAL_REALIZATION",
            realized(
                "C",
                revenue=D("1080"),
                margin=D("0.108"),
                capital=D("840"),
                net_debt=D("95"),
            ),
        ),
        _evaluate_case("SPLIT", realized("D_SPLIT", shares=D("20"))),
        _evaluate_case("DIVIDEND", realized("D_DIVIDEND", net_debt=D("110"))),
        _evaluate_case(
            "RIGHTS_ISSUE",
            realized("D_RIGHTS", net_debt=D("80"), shares=D("12")),
        ),
    ]
    return {
        "schema_version": "moatrader-production-eri-null-fixtures-v1/1",
        "status": "PASSED" if all(bool(item["passed"]) for item in cases) else "FAILED",
        "all_passed": all(bool(item["passed"]) for item in cases),
        "tolerance": str(TOLERANCE),
        "fixture_count": len(cases),
        "cases": cases,
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
