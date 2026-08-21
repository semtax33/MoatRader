from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
import json
from pathlib import Path

import pytest

from moatrader.canonical.models import StatementType
from moatrader.expectations.future_eri import (
    CurrentExpectationStateV1,
    EvidenceIndexFutureEriFeatureRowV2,
    EriMechanismObservationV1,
    EriMonotonicityPolicyV1,
    EvidenceObservation,
    EvidenceScoreBand,
    EvidenceState,
    EvidenceVectorStatus,
    FcffEvidenceVectorV1,
    FutureEriFeatureRowV1,
    FutureEriOutcomeInputV1,
    MaterialityBasis,
    OperatingEvidenceAxis,
    RealizedFcffStateV1,
    build_fcff_evidence_vector,
    build_evidence_index_future_eri_label_v2,
    build_future_eri_label,
    evaluate_future_eri_monotonicity,
    next_usable_signal_timestamp,
    roll_forward_frozen_expectations,
    scaled_materiality,
    seal_feature_dataset,
    seal_evidence_index_feature_dataset_v2,
    target_trading_session,
)
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from moatrader.valuation.economic_dcf import EconomicDcfEngine
from scripts.run_future_eri_v1 import run


D = Decimal
KST = timezone(timedelta(hours=9))
SIGNAL_AT = datetime(2024, 5, 16, 9, 0, tzinfo=KST)
PRIOR_SIGNAL_AT = datetime(2024, 2, 15, 9, 0, tzinfo=KST)


def _observation(
    axis: OperatingEvidenceAxis,
    state: EvidenceState,
    *,
    current: bool,
    materiality: str = "0.5",
) -> EvidenceObservation:
    available = (
        datetime(2024, 5, 15, 17, 30, tzinfo=KST)
        if current
        else datetime(2024, 2, 14, 12, 0, tzinfo=KST)
    )
    return EvidenceObservation(
        observation_id=f"{'CURRENT' if current else 'PRIOR'}:{axis.value}",
        issuer_id="00126380",
        fiscal_period="2024Q1" if current else "2023Q4",
        axis=axis,
        state=state,
        source_document_id=f"DART:{'2024Q1' if current else '2023Q4'}",
        source_span=f"{axis.value} disclosed evidence span",
        source_published_at=available - timedelta(minutes=5),
        available_at=available,
        signal_timestamp=SIGNAL_AT if current else PRIOR_SIGNAL_AT,
        statement_type=StatementType.DISCLOSED_FACT,
        classification_rule_id="FCFF_EVIDENCE_V1",
        materiality_rule_id="SEMANTIC_BUCKET_V1",
        confidence=D("0.9"),
        materiality=D(materiality),
    )


def _complete_vector() -> FcffEvidenceVectorV1:
    current_states = {
        OperatingEvidenceAxis.DEMAND: EvidenceState.IMPROVING,
        OperatingEvidenceAxis.PRICE_MIX: EvidenceState.IMPROVING,
        OperatingEvidenceAxis.BACKLOG: EvidenceState.STABLE,
        OperatingEvidenceAxis.MARGIN: EvidenceState.IMPROVING,
        OperatingEvidenceAxis.INVENTORY_MISMATCH: EvidenceState.WEAKENING,
        OperatingEvidenceAxis.CAPACITY_CAPEX: EvidenceState.IMPROVING,
    }
    current = [_observation(axis, state, current=True) for axis, state in current_states.items()]
    prior = [
        _observation(axis, EvidenceState.STABLE, current=False, materiality="0.25")
        for axis in OperatingEvidenceAxis
    ]
    return build_fcff_evidence_vector(
        issuer_id="00126380",
        signal_timestamp=SIGNAL_AT,
        current=current,
        prior=prior,
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
    return FutureEriFeatureRowV1(
        observation_id="00126380:2024-05-16",
        evidence=_complete_vector(),
        expectation_state=CurrentExpectationStateV1(
            issuer_id="00126380",
            signal_timestamp=SIGNAL_AT,
            market_price=D("100"),
            market_price_at=SIGNAL_AT,
            market_price_source_id="KRX:005930:2024-05-16:OPEN",
            implied_growth=D("0.08"),
            implied_margin=D("0.12"),
            implied_roiic=D("0.20"),
            implied_cap_years=D("6"),
            reverse_dcf_method="FCFF_FROZEN_PATH_V1",
            reverse_dcf_input_sha256="a" * 64,
        ),
        frozen_expectation_assumptions=_assumptions(),
    )


def _sessions() -> list[date]:
    return [SIGNAL_AT.date() + timedelta(days=offset) for offset in range(100)]


def _outcome(feature: FutureEriFeatureRowV1) -> FutureEriOutcomeInputV1:
    target = target_trading_session(feature.evidence.signal_timestamp.date(), _sessions())
    return FutureEriOutcomeInputV1(
        observation_id=feature.observation_id,
        target_session=target,
        target_price_at=datetime.combine(target, time(15, 30), tzinfo=KST),
        actual_market_price=D("125"),
        target_price_source_id=f"KRX:005930:{target}:CLOSE",
        realized_state=RealizedFcffStateV1(
            available_at=datetime.combine(target, time(8, 0), tzinfo=KST),
            base_period="2024Q2",
            base_revenue=D("1040"),
            base_nopat_margin=D("0.11"),
            base_invested_capital=D("820"),
            net_debt=D("90"),
            diluted_shares=D("10"),
            wacc=D("0.085"),
            wacc_source_id=f"FROZEN_WACC_POLICY:{target}",
            source_document_ids=["DART:2024Q2"],
        ),
    )


def test_after_close_evidence_becomes_usable_at_next_session_open() -> None:
    sessions = [date(2024, 5, 15), date(2024, 5, 16), date(2024, 5, 17)]
    published = datetime(2024, 5, 15, 17, 30, tzinfo=KST)
    assert next_usable_signal_timestamp(published, trading_sessions=sessions) == SIGNAL_AT


def test_evidence_observation_rejects_future_and_non_auditable_inputs() -> None:
    payload = _observation(
        OperatingEvidenceAxis.DEMAND,
        EvidenceState.IMPROVING,
        current=True,
    ).model_dump()
    payload["available_at"] = SIGNAL_AT + timedelta(minutes=1)
    with pytest.raises(ValueError, match="not available"):
        EvidenceObservation.model_validate(payload)
    payload = _observation(
        OperatingEvidenceAxis.DEMAND,
        EvidenceState.IMPROVING,
        current=True,
    ).model_dump()
    payload["statement_type"] = StatementType.FORECAST
    with pytest.raises(ValueError, match="fact, claim, or derived metric"):
        EvidenceObservation.model_validate(payload)


def test_materiality_ratio_is_deterministic_and_capped() -> None:
    computation = scaled_materiality(
        basis=MaterialityBasis.CONTRACT_VALUE_TTM_REVENUE,
        numerator=D("1200"),
        denominator=D("1000"),
        numerator_source_id="DART:CONTRACT",
        denominator_source_id="DART:TTM_REVENUE",
    )
    assert computation.raw_ratio == D("1.2")
    assert computation.capped_materiality == D("1")
    payload = computation.model_dump()
    payload["raw_ratio"] = D("0.8")
    with pytest.raises(ValueError, match="numerator / denominator"):
        type(computation).model_validate(payload)


def test_six_axis_vector_is_delta_based_and_missing_is_not_neutral() -> None:
    vector = _complete_vector()
    assert vector.status == EvidenceVectorStatus.COMPLETE
    assert vector.evidence_f_score == 3
    assert vector.materiality_weighted_score == D("1.5")
    assert vector.primary_ranking_policy == "NONE_MECHANISM_ONLY"

    incomplete = build_fcff_evidence_vector(
        issuer_id="00126380",
        signal_timestamp=SIGNAL_AT,
        current=[
            _observation(
                OperatingEvidenceAxis.DEMAND,
                EvidenceState.IMPROVING,
                current=True,
            )
        ],
        prior=[
            _observation(
                OperatingEvidenceAxis.DEMAND,
                EvidenceState.STABLE,
                current=False,
            )
        ],
    )
    assert incomplete.status == EvidenceVectorStatus.INCOMPLETE
    assert incomplete.evidence_f_score is None
    assert len(incomplete.missing_axes) == 5


def test_feature_seal_precedes_outcome_and_is_deterministic() -> None:
    feature = _feature()
    first = seal_feature_dataset([feature], sealed_at=SIGNAL_AT + timedelta(minutes=1))
    second = seal_feature_dataset([feature], sealed_at=SIGNAL_AT + timedelta(minutes=2))
    assert first.feature_dataset_sha256 == second.feature_dataset_sha256
    assert first.feature_row_sha256[feature.observation_id]
    assert first.outcome_source_opened_before_seal is False
    assert first.return_data_accessed is False
    assert "future_eri" not in feature.model_dump()
    assert "future_return" not in feature.model_dump()


def test_counterfactual_reanchors_realized_facts_but_freezes_expectations() -> None:
    feature = _feature()
    outcome = _outcome(feature)
    rolled = roll_forward_frozen_expectations(feature, outcome)
    assert rolled.base_revenue == D("1040")
    assert rolled.base_nopat_margin == D("0.11")
    assert rolled.wacc == D("0.085")
    assert rolled.revenue_growth == feature.frozen_expectation_assumptions.revenue_growth
    assert rolled.target_nopat_margin == feature.frozen_expectation_assumptions.target_nopat_margin
    assert rolled.roiic == feature.frozen_expectation_assumptions.roiic
    assert rolled.competitive_advantage_period_years == 6


def test_future_eri_label_is_exactly_63_sessions_and_not_a_return_label() -> None:
    feature = _feature()
    seal = seal_feature_dataset([feature], sealed_at=SIGNAL_AT + timedelta(minutes=1))
    outcome = _outcome(feature)
    label = build_future_eri_label(
        feature=feature,
        outcome=outcome,
        feature_seal=seal,
        trading_sessions=_sessions(),
    )
    expected = (label.actual_market_price / label.counterfactual_value_per_share).ln()
    assert label.future_eri == expected
    assert label.horizon_trading_days == 63
    assert label.return_data_accessed is False
    assert "future_return" not in label.model_dump()
    assert (
        label.enterprise_equity_bridge.counterfactual_enterprise_value
        - label.enterprise_equity_bridge.counterfactual_net_debt
        == label.enterprise_equity_bridge.counterfactual_equity_value
    )

    changed_feature = feature.model_copy(
        update={
            "expectation_state": feature.expectation_state.model_copy(
                update={"market_price": D("101")}
            )
        }
    )
    with pytest.raises(ValueError, match="changed after"):
        build_future_eri_label(
            feature=changed_feature,
            outcome=outcome,
            feature_seal=seal,
            trading_sessions=_sessions(),
        )

    invalid = outcome.model_copy(
        update={"target_session": outcome.target_session + timedelta(days=1)}
    )
    with pytest.raises(ValueError, match=r"exactly t\+63"):
        build_future_eri_label(
            feature=feature,
            outcome=invalid,
            feature_seal=seal,
            trading_sessions=_sessions(),
        )


def test_v2_evidence_index_feature_is_sealed_before_exact_t63_future_eri() -> None:
    base = _feature()
    feature = EvidenceIndexFutureEriFeatureRowV2(
        observation_id=base.observation_id,
        issuer_id=base.expectation_state.issuer_id,
        signal_timestamp=base.expectation_state.signal_timestamp,
        full_evidence_index=D("0.5"),
        full_nobs=4,
        core_evidence_index=D("0"),
        core_nobs=3,
        full_index_row_sha256="b" * 64,
        core_index_row_sha256="c" * 64,
        full_index_seal_sha256="d" * 64,
        expectation_state=base.expectation_state,
        frozen_expectation_assumptions=base.frozen_expectation_assumptions,
    )
    seal = seal_evidence_index_feature_dataset_v2(
        [feature],
        sealed_at=SIGNAL_AT + timedelta(minutes=1),
        full_index_seal_sha256="d" * 64,
    )
    label = build_evidence_index_future_eri_label_v2(
        feature=feature,
        outcome=_outcome(base),
        feature_seal=seal,
        trading_sessions=_sessions(),
    )

    assert label.horizon_trading_days == 63
    assert feature.downstream_outcome_role == "EVIDENCE_INDEX_PREDICTS_T63_ERI"
    assert feature.outcome_value_used_as_signal is False
    assert feature.outcome_value_used_as_ranking is False
    assert feature.per_pbr_role == "NOT_USED"
    assert seal.outcome_source_opened_before_seal is False
    assert label.return_data_accessed is False

    wrong_target = _outcome(base).model_copy(
        update={"target_session": _outcome(base).target_session + timedelta(days=1)}
    )
    with pytest.raises(ValueError, match=r"exactly t\+63"):
        build_evidence_index_future_eri_label_v2(
            feature=feature,
            outcome=wrong_target,
            feature_seal=seal,
            trading_sessions=_sessions(),
        )

def _null_case_label(realized: RealizedFcffStateV1):
    feature = _feature()
    base_outcome = _outcome(feature)
    outcome = base_outcome.model_copy(update={"realized_state": realized})
    rolled = roll_forward_frozen_expectations(feature, outcome)
    explained_price = EconomicDcfEngine().value(rolled).fair_value_per_share
    explained_outcome = outcome.model_copy(update={"actual_market_price": explained_price})
    seal = seal_feature_dataset([feature], sealed_at=SIGNAL_AT + timedelta(minutes=1))
    return build_future_eri_label(
        feature=feature,
        outcome=explained_outcome,
        feature_seal=seal,
        trading_sessions=_sessions(),
    )


def test_eri_null_case_a_unchanged_price_and_expectations_is_zero() -> None:
    assumptions = _assumptions()
    target = _outcome(_feature()).target_session
    realized = RealizedFcffStateV1(
        available_at=datetime.combine(target, time(8), tzinfo=KST),
        base_period=assumptions.base_period or "2024Q1",
        base_revenue=assumptions.base_revenue,
        base_nopat_margin=assumptions.base_nopat_margin,
        base_invested_capital=assumptions.base_invested_capital,
        net_debt=assumptions.net_debt,
        diluted_shares=assumptions.diluted_shares,
        wacc=assumptions.wacc,
        wacc_source_id="NULL:A:WACC",
        source_document_ids=["NULL:A"],
    )
    label = _null_case_label(realized)
    assert label.future_eri == 0
    assert label.enterprise_future_eri == 0
    assert label.capital_structure_bridge_effect == 0


def test_eri_null_case_b_wacc_only_move_is_removed() -> None:
    base = _outcome(_feature()).realized_state
    label = _null_case_label(
        base.model_copy(
            update={
                "base_revenue": _assumptions().base_revenue,
                "base_nopat_margin": _assumptions().base_nopat_margin,
                "base_invested_capital": _assumptions().base_invested_capital,
                "net_debt": _assumptions().net_debt,
                "diluted_shares": _assumptions().diluted_shares,
                "wacc": D("0.07"),
                "wacc_source_id": "NULL:B:LOWER_WACC",
                "source_document_ids": ["NULL:B"],
            }
        )
    )
    assert label.future_eri == 0
    assert label.enterprise_future_eri == 0


def test_eri_null_case_c_expected_fundamental_realization_is_removed() -> None:
    base = _outcome(_feature()).realized_state
    label = _null_case_label(
        base.model_copy(
            update={
                "base_revenue": D("1080"),
                "base_nopat_margin": D("0.108"),
                "base_invested_capital": D("840"),
                "net_debt": D("95"),
                "diluted_shares": D("10"),
                "wacc": D("0.09"),
                "wacc_source_id": "NULL:C:WACC",
                "source_document_ids": ["NULL:C:REALIZED"],
            }
        )
    )
    assert label.future_eri == 0
    assert label.enterprise_future_eri == 0


@pytest.mark.parametrize(
    ("case_id", "net_debt", "shares"),
    [
        ("SPLIT", D("100"), D("20")),
        ("DIVIDEND", D("110"), D("10")),
        ("RIGHTS_ISSUE", D("80"), D("12")),
    ],
)
def test_eri_null_case_d_corporate_actions_are_removed_by_ev_equity_bridge(
    case_id: str,
    net_debt: Decimal,
    shares: Decimal,
) -> None:
    assumptions = _assumptions()
    base = _outcome(_feature()).realized_state
    label = _null_case_label(
        base.model_copy(
            update={
                "base_revenue": assumptions.base_revenue,
                "base_nopat_margin": assumptions.base_nopat_margin,
                "base_invested_capital": assumptions.base_invested_capital,
                "net_debt": net_debt,
                "diluted_shares": shares,
                "wacc": assumptions.wacc,
                "wacc_source_id": f"NULL:D:{case_id}:WACC",
                "source_document_ids": [f"NULL:D:{case_id}"],
            }
        )
    )
    assert label.future_eri == 0
    assert abs(label.enterprise_future_eri) < D("1e-24")
    assert abs(label.capital_structure_bridge_effect) < D("1e-24")

def test_five_band_monotonicity_gate_opens_ml_but_never_runs_return_stage() -> None:
    scores = [-4, -3, -2, -1, 0, 0, 1, 2, 3, 5]
    observations = [
        EriMechanismObservationV1(
            observation_id=f"ROW:{index}",
            signal_timestamp=SIGNAL_AT,
            evidence_f_score=score,
            future_eri=D(score) / D(100),
        )
        for index, score in enumerate(scores)
    ]
    report = evaluate_future_eri_monotonicity(
        observations,
        policy=EriMonotonicityPolicyV1(minimum_observations_per_band=2),
    )
    assert [item.band for item in report.bands] == list(EvidenceScoreBand)
    assert report.adjacent_nondecreasing_count == 4
    assert report.mechanism_gate_passed
    assert report.ml_stage_authorized
    assert report.return_stage_status == "NOT_RUN_V1_MECHANISM_ONLY"
    assert report.return_data_accessed is False


def test_non_monotonic_eri_keeps_downstream_stages_blocked() -> None:
    scores = [-4, -3, -2, -1, 0, 0, 1, 2, 3, 5]
    eri = [D("-0.04"), D("-0.03"), D("0.20"), D("0.20"), D("0"), D("0"), D("0.01"), D("0.02"), D("0.03"), D("0.05")]
    observations = [
        EriMechanismObservationV1(
            observation_id=f"ROW:{index}",
            signal_timestamp=SIGNAL_AT,
            evidence_f_score=score,
            future_eri=value,
        )
        for index, (score, value) in enumerate(zip(scores, eri, strict=True))
    ]
    report = evaluate_future_eri_monotonicity(
        observations,
        policy=EriMonotonicityPolicyV1(minimum_observations_per_band=2),
    )
    assert not report.mechanism_gate_passed
    assert not report.ml_stage_authorized
    assert report.return_stage_status == "BLOCKED_MECHANISM_GATE_FAILED"


def test_runner_seals_features_before_building_eri_labels(tmp_path: Path) -> None:
    current_states = {
        OperatingEvidenceAxis.DEMAND: EvidenceState.IMPROVING,
        OperatingEvidenceAxis.PRICE_MIX: EvidenceState.IMPROVING,
        OperatingEvidenceAxis.BACKLOG: EvidenceState.STABLE,
        OperatingEvidenceAxis.MARGIN: EvidenceState.IMPROVING,
        OperatingEvidenceAxis.INVENTORY_MISMATCH: EvidenceState.WEAKENING,
        OperatingEvidenceAxis.CAPACITY_CAPEX: EvidenceState.IMPROVING,
    }
    feature = _feature()
    outcome = _outcome(feature)
    feature_input = tmp_path / "feature-input.json"
    outcome_input = tmp_path / "outcome-input.json"
    sessions_input = tmp_path / "sessions.json"
    output = tmp_path / "result"
    feature_input.write_text(
        json.dumps(
            [
                {
                    "observation_id": feature.observation_id,
                    "current_observations": [
                        _observation(axis, state, current=True).model_dump(mode="json")
                        for axis, state in current_states.items()
                    ],
                    "prior_observations": [
                        _observation(
                            axis,
                            EvidenceState.STABLE,
                            current=False,
                            materiality="0.25",
                        ).model_dump(mode="json")
                        for axis in OperatingEvidenceAxis
                    ],
                    "expectation_state": feature.expectation_state.model_dump(mode="json"),
                    "frozen_expectation_assumptions": (
                        feature.frozen_expectation_assumptions.model_dump(mode="json")
                    ),
                }
            ]
        ),
        encoding="utf-8",
    )
    outcome_input.write_text(
        json.dumps([outcome.model_dump(mode="json")]),
        encoding="utf-8",
    )
    sessions_input.write_text(
        json.dumps([item.isoformat() for item in _sessions()]),
        encoding="utf-8",
    )

    final = run(
        feature_input=feature_input,
        outcome_input=outcome_input,
        trading_sessions_path=sessions_input,
        output=output,
        minimum_observations_per_band=1,
    )
    seal = json.loads((output / "feature-seal.json").read_text(encoding="utf-8"))
    stored_feature = json.loads(
        (output / "features-pre-label.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert seal["outcome_source_opened_before_seal"] is False
    assert "future_eri" not in stored_feature
    assert "future_return" not in stored_feature
    assert final["future_eri_label_count"] == 1
    assert final["return_data_accessed"] is False
    assert final["per_pbr_role"] == "NOT_USED"
    assert (output / "build-manifest.json").is_file()
