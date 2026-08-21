from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from moatrader.expectations.future_eri import (
    CurrentExpectationStateV1,
    EvidenceState,
    FutureEriOutcomeInputV1,
    OperatingEvidenceAxis,
    RealizedFcffStateV1,
    target_trading_session,
)
from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    AxisPairClassification,
    BlindedExcerpt,
    HistoricalFilingPair,
    HistoricalRegularFiling,
    HistoricalSourceOrigin,
    HistoricalSourceVariant,
    PairedAxisPacket,
    ReceiptLinkage,
    historical_pair_id,
    packet_id,
    sha256_file,
)
from moatrader.expectations.historical_evidence_v2 import (
    AbstentionReasonV2,
    AxisApplicabilityV2,
    AxisEvidenceProvenanceV2,
    AxisSignedScoreRoleV2,
    DeterministicCoreIndexCoveragePolicyV2,
    EvidenceIndexContractV2,
    FullEvidenceIndexCoveragePolicyV2,
    GroundedAxisStateSnapshotV2,
    PITApplicabilityRulesV2,
    PITOperatingSnapshotV2,
    PreviousEvidenceBasisV2,
    SparseAxisAvailabilityV2,
    SparseAxisEvidenceV2,
    SparseCoverageGatePolicyV2,
    build_deterministic_pit_axis_evidence,
    build_deterministic_core_index_row_v2,
    build_full_evidence_index_row_v2,
    build_last_grounded_axis_evidence,
    build_sparse_feature_row_v2,
    calibrate_sparse_band_contract_v2,
    evaluate_sparse_coverage_gate_v2,
    fixed_economic_breadth_band_v2,
    merge_axis_evidence_v2,
    sparse_band_diagnostics_v2,
    sparse_feature_coverage_report,
)
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from scripts.audit_historical_evidence_abstentions_v2 import (
    prepare_abstention_audit,
    validate_abstention_audit,
)
from scripts.audit_historical_future_eri_outcome_eligibility import (
    OutcomeEligibilityInventoryRowV1,
)
from scripts.audit_historical_deterministic_measurement_v2 import (
    _direction as inventory_economic_direction,
)
from scripts.build_historical_deterministic_pit_evidence_v2 import (
    LastGroundedDeterministicBasisInputV2,
    PITOperatingPairInputV2,
    build_pit_evidence,
)
from scripts.build_historical_sparse_features_v2 import (
    AxisApplicabilityDecisionInputV2,
    DeterministicAxisEvidenceInputV2,
    build_sparse_features,
)
from scripts.calibrate_historical_sparse_features_v2 import calibrate_sparse_features
from scripts.classify_historical_future_eri_evidence import ParserProfile, parser_spec
from scripts.evaluate_historical_evidence_parser_v2 import (
    combine_v2_locked_evaluations,
    create_v2_parser_freeze,
    evaluate_v2_locked_parser,
)
from scripts.freeze_historical_evidence_index_v2 import (
    deterministic_core_diagnostics_v2,
)
from scripts.seal_historical_full_evidence_index_v2 import (
    full_evidence_index_diagnostics_v2,
    seal_full_evidence_index_v2,
)
from scripts.run_historical_evidence_index_eri_v2 import run_evidence_index_eri_v2
from scripts.prepare_historical_evidence_classification_subset import prepare_subset
from scripts.prepare_historical_locked_sets_v2 import (
    _classification,
    _review_row_is_blank,
    finalize_locked_sets,
    prepare_locked_candidates,
)
from scripts.materialize_historical_human_gold_v2 import materialize_human_gold
from scripts.prepare_historical_deterministic_pit_inputs_v2 import (
    FilingSource,
    FilingTask,
    _extract_task,
    extract_pit_metrics_from_html,
)
from scripts.prepare_historical_last_grounded_inputs_v2 import (
    prepare_last_grounded_inputs,
)
from scripts.prepare_historical_semantic_packets_v2 import prepare_semantic_packets


D = Decimal
SEOUL = ZoneInfo("Asia/Seoul")


def _filing(ticker: str, rcept_no: str, period: date) -> HistoricalRegularFiling:
    timestamp = datetime.strptime(rcept_no[:8], "%Y%m%d").replace(
        hour=16,
        tzinfo=SEOUL,
    )
    return HistoricalRegularFiling(
        ticker=ticker,
        issuer_name="테스트",
        rcept_no=rcept_no,
        report_name="정기보고서",
        report_code={3: "11013", 6: "11012", 9: "11014", 12: "11011"}[period.month],
        fiscal_period_end=period,
        published_at=timestamp,
        available_at=timestamp,
        signal_timestamp=timestamp,
        source_variants=[
            HistoricalSourceVariant(
                origin=HistoricalSourceOrigin.ARCANA_BUSINESS_HTML,
                path=f"readonly/{rcept_no}.html",
                raw_sha256=hashlib.sha256(rcept_no.encode()).hexdigest(),
                byte_count=10,
                receipt_linkage=ReceiptLinkage.EXACT_METADATA,
            )
        ],
    )


def _pair(index: int = 1) -> HistoricalFilingPair:
    ticker = f"{index:06d}"
    previous = _filing(ticker, f"20200330{index:06d}", date(2020, 3, 31))
    current = _filing(ticker, f"20200814{index:06d}", date(2020, 6, 30))
    return HistoricalFilingPair(
        pair_id=historical_pair_id(ticker, previous.rcept_no, current.rcept_no),
        ticker=ticker,
        previous=previous,
        current=current,
    )


def _packet(pair: HistoricalFilingPair, axis: OperatingEvidenceAxis, *, both: bool = True) -> PairedAxisPacket:
    axis_index = list(OperatingEvidenceAxis).index(axis)
    return PairedAxisPacket(
        packet_id=packet_id(pair.pair_id, axis),
        axis=axis,
        previous_excerpts=[
            BlindedExcerpt(source_id=f"SRC_{axis_index * 2 + 1:020x}", text="이전 상태")
        ],
        current_excerpts=(
            [BlindedExcerpt(source_id=f"SRC_{axis_index * 2 + 2:020x}", text="현재 개선")]
            if both
            else []
        ),
    )


def _grounded(
    axis: OperatingEvidenceAxis,
    direction: EvidenceState,
    *,
    pair: HistoricalFilingPair,
) -> SparseAxisEvidenceV2:
    return SparseAxisEvidenceV2(
        axis=axis,
        applicability=AxisApplicabilityV2.APPLICABLE,
        availability=SparseAxisAvailabilityV2.GROUNDED,
        direction=direction,
        provenance=AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC,
        previous_evidence_basis=PreviousEvidenceBasisV2.IMMEDIATE_PREVIOUS_FILING,
        confidence=D(1),
        source_ids=[f"SOURCE_{axis.value}"],
        previous_evidence_at=pair.previous.available_at,
        current_evidence_at=pair.current.available_at,
        applicability_rule_id="TEST_PIT_RULE",
        signed_score_role=(
            AxisSignedScoreRoleV2.RAW_DIRECTION_ONLY
            if axis == OperatingEvidenceAxis.CAPACITY_CAPEX
            else AxisSignedScoreRoleV2.PRIMARY_SIGNED_SCORE
        ),
    )


def _na(axis: OperatingEvidenceAxis) -> SparseAxisEvidenceV2:
    return SparseAxisEvidenceV2(
        axis=axis,
        applicability=AxisApplicabilityV2.APPLICABLE,
        availability=SparseAxisAvailabilityV2.NA,
        abstention_reason=AbstentionReasonV2.TRUE_NO_MENTION,
        applicability_rule_id="TEST_APPLICABLE",
        signed_score_role=(
            AxisSignedScoreRoleV2.RAW_DIRECTION_ONLY
            if axis == OperatingEvidenceAxis.CAPACITY_CAPEX
            else AxisSignedScoreRoleV2.PRIMARY_SIGNED_SCORE
        ),
    )


def _not_applicable(axis: OperatingEvidenceAxis) -> SparseAxisEvidenceV2:
    return SparseAxisEvidenceV2(
        axis=axis,
        applicability=AxisApplicabilityV2.NOT_APPLICABLE,
        availability=SparseAxisAvailabilityV2.NOT_APPLICABLE,
        applicability_rule_id="TEST_NOT_APPLICABLE",
        signed_score_role=(
            AxisSignedScoreRoleV2.RAW_DIRECTION_ONLY
            if axis == OperatingEvidenceAxis.CAPACITY_CAPEX
            else AxisSignedScoreRoleV2.PRIMARY_SIGNED_SCORE
        ),
    )


def test_sparse_contract_keeps_na_and_not_applicable_distinct_from_neutral() -> None:
    pair = _pair()
    evidence = [
        _grounded(OperatingEvidenceAxis.DEMAND, EvidenceState.IMPROVING, pair=pair),
        _grounded(OperatingEvidenceAxis.PRICE_MIX, EvidenceState.STABLE, pair=pair),
        _na(OperatingEvidenceAxis.BACKLOG),
        _grounded(OperatingEvidenceAxis.MARGIN, EvidenceState.WEAKENING, pair=pair),
        _not_applicable(OperatingEvidenceAxis.INVENTORY_MISMATCH),
        _grounded(OperatingEvidenceAxis.CAPACITY_CAPEX, EvidenceState.IMPROVING, pair=pair),
    ]

    row = build_sparse_feature_row_v2(pair=pair, axis_evidence=evidence)

    assert row.observed_axis_count == 4
    assert row.applicable_axis_count == 5
    assert row.unavailable_axis_count == 1
    assert row.not_applicable_axis_count == 1
    assert row.signed_score_axis_count == 3
    assert row.raw_direction_only_axis_count == 1
    assert row.neutral_axis_count == 1
    assert row.signed_breadth == D("0")
    assert row.coverage == D("0.8")
    with pytest.raises(ValueError, match="NA axis cannot contain"):
        SparseAxisEvidenceV2(
            axis=OperatingEvidenceAxis.BACKLOG,
            applicability=AxisApplicabilityV2.APPLICABLE,
            availability=SparseAxisAvailabilityV2.NA,
            direction=EvidenceState.STABLE,
            abstention_reason=AbstentionReasonV2.TRUE_NO_MENTION,
            applicability_rule_id="BAD_ZERO_IMPUTATION",
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("-1", "STRONG_BEAR"),
        ("-0.999", "BEAR"),
        ("-0.5", "BEAR"),
        ("0", "NEUTRAL"),
        ("0.5", "BULL"),
        ("0.999", "BULL"),
        ("1", "STRONG_BULL"),
    ],
)
def test_fixed_economic_bands_do_not_depend_on_sample_quantiles(
    value: str,
    expected: str,
) -> None:
    assert fixed_economic_breadth_band_v2(D(value)).value == expected


def test_evidence_index_contract_prespecifies_full_and_freezes_core() -> None:
    contract = EvidenceIndexContractV2()

    assert contract.primary_index == "FULL_EVIDENCE_SIGNED_BREADTH_V2"
    assert contract.primary_measurement_status == (
        "PRESPECIFIED_PENDING_DEMAND_PRICE_MIX_GATE"
    )
    assert contract.full_index_materialized is False
    assert contract.secondary_index == "DETERMINISTIC_CORE_SIGNED_BREADTH_V2"
    assert contract.minimum_observed_axes == 2
    assert contract.capex_role == "DIAGNOSTIC_ONLY"
    assert contract.index_multiplied_by_coverage is False
    assert contract.outcome_stage_authorized is False
    assert contract.value_data_opened is False
    assert contract.per_pbr_role == "NOT_USED"


def _core_row(
    index: int,
    directions: tuple[EvidenceState | None, EvidenceState | None, EvidenceState | None],
    *,
    capex_direction: EvidenceState | None = None,
):
    pair = _pair(index)
    evidence = []
    for axis, direction in zip(
        (
            OperatingEvidenceAxis.MARGIN,
            OperatingEvidenceAxis.INVENTORY_MISMATCH,
            OperatingEvidenceAxis.BACKLOG,
        ),
        directions,
        strict=True,
    ):
        evidence.append(
            _grounded(axis, direction, pair=pair) if direction is not None else _na(axis)
        )
    if capex_direction is None:
        evidence.append(_na(OperatingEvidenceAxis.CAPACITY_CAPEX))
    else:
        evidence.append(
            _grounded(
                OperatingEvidenceAxis.CAPACITY_CAPEX,
                capex_direction,
                pair=pair,
            ).model_copy(
                update={
                    "deterministic_metric_name": "CAPEX_TO_REVENUE",
                    "deterministic_previous_value": D("0.1"),
                    "deterministic_current_value": D("0.2"),
                    "deterministic_delta": D("0.1"),
                }
            )
        )
    return build_deterministic_core_index_row_v2(pair=pair, axis_evidence=evidence)


def test_deterministic_core_score_excludes_capex_and_keeps_coverage_separate() -> None:
    bullish_capex = _core_row(
        101,
        (EvidenceState.IMPROVING, EvidenceState.WEAKENING, None),
        capex_direction=EvidenceState.IMPROVING,
    )
    bearish_capex = _core_row(
        102,
        (EvidenceState.IMPROVING, EvidenceState.WEAKENING, None),
        capex_direction=EvidenceState.WEAKENING,
    )

    assert bullish_capex.nobs == 2
    assert bullish_capex.core_evidence_index == D(0)
    assert bullish_capex.core_evidence_index_fraction == "0"
    assert bullish_capex.coverage == D(2) / D(3)
    assert bullish_capex.band.value == "NEUTRAL"
    assert bullish_capex.eligible is True
    assert bullish_capex.capex_in_index is False
    assert bearish_capex.core_evidence_index == bullish_capex.core_evidence_index
    assert bearish_capex.nobs == bullish_capex.nobs
    assert bearish_capex.row_sha256 != bullish_capex.row_sha256


def test_deterministic_core_requires_nobs_two_before_assigning_a_band() -> None:
    row = _core_row(103, (EvidenceState.IMPROVING, None, None))

    assert row.nobs == 1
    assert row.core_evidence_index == D(1)
    assert row.eligible is False
    assert row.band is None


def test_full_evidence_index_uses_exactly_five_axes_and_never_imputes_na() -> None:
    pair = _pair(104)
    capex = _grounded(
        OperatingEvidenceAxis.CAPACITY_CAPEX,
        EvidenceState.IMPROVING,
        pair=pair,
    ).model_copy(
        update={
            "deterministic_metric_name": "CAPEX_TO_REVENUE",
            "deterministic_previous_value": D("0.1"),
            "deterministic_current_value": D("0.2"),
            "deterministic_delta": D("0.1"),
        }
    )
    sparse = build_sparse_feature_row_v2(
        pair=pair,
        axis_evidence=[
            _grounded(OperatingEvidenceAxis.DEMAND, EvidenceState.IMPROVING, pair=pair),
            _na(OperatingEvidenceAxis.PRICE_MIX),
            _grounded(OperatingEvidenceAxis.MARGIN, EvidenceState.WEAKENING, pair=pair),
            _grounded(OperatingEvidenceAxis.INVENTORY_MISMATCH, EvidenceState.STABLE, pair=pair),
            _not_applicable(OperatingEvidenceAxis.BACKLOG),
            capex,
        ],
    )

    full = build_full_evidence_index_row_v2(sparse)

    assert full.nobs == 3
    assert full.net_evidence == 0
    assert full.full_evidence_index == D(0)
    assert full.coverage == D(3) / D(4)
    assert full.semantic_grounded_axis_count == 1
    assert full.deterministic_core_grounded_axis_count == 2
    assert full.full_axis_states[OperatingEvidenceAxis.PRICE_MIX].value == "NA"
    assert full.full_axis_states[OperatingEvidenceAxis.BACKLOG].value == "NOT_APPLICABLE"
    assert full.capex_in_index is False
    assert full.band.value == "NEUTRAL"
    assert full.per_pbr_role == "NOT_USED"


def test_full_evidence_index_requires_nobs_two_for_band() -> None:
    pair = _pair(105)
    sparse = build_sparse_feature_row_v2(
        pair=pair,
        axis_evidence=[
            _grounded(OperatingEvidenceAxis.DEMAND, EvidenceState.IMPROVING, pair=pair),
            _na(OperatingEvidenceAxis.PRICE_MIX),
            _na(OperatingEvidenceAxis.MARGIN),
            _na(OperatingEvidenceAxis.INVENTORY_MISMATCH),
            _not_applicable(OperatingEvidenceAxis.BACKLOG),
            _na(OperatingEvidenceAxis.CAPACITY_CAPEX),
        ],
    )

    full = build_full_evidence_index_row_v2(sparse)

    assert full.nobs == 1
    assert full.full_evidence_index == D(1)
    assert full.eligible is False
    assert full.band is None


def _full_row(
    index: int,
    directions: tuple[
        EvidenceState | None,
        EvidenceState | None,
        EvidenceState | None,
        EvidenceState | None,
        EvidenceState | None,
    ],
):
    pair = _pair(index)
    evidence = [
        _grounded(axis, direction, pair=pair) if direction is not None else _na(axis)
        for axis, direction in zip(
            (
                OperatingEvidenceAxis.DEMAND,
                OperatingEvidenceAxis.PRICE_MIX,
                OperatingEvidenceAxis.MARGIN,
                OperatingEvidenceAxis.INVENTORY_MISMATCH,
                OperatingEvidenceAxis.BACKLOG,
            ),
            directions,
            strict=True,
        )
    ]
    evidence.append(_na(OperatingEvidenceAxis.CAPACITY_CAPEX))
    return build_full_evidence_index_row_v2(
        build_sparse_feature_row_v2(pair=pair, axis_evidence=evidence)
    )


def test_full_evidence_index_coverage_gate_reports_all_fixed_bands() -> None:
    rows = [
        _full_row(121, (EvidenceState.WEAKENING, EvidenceState.WEAKENING, None, None, None)),
        _full_row(122, (EvidenceState.WEAKENING, EvidenceState.STABLE, None, None, None)),
        _full_row(123, (EvidenceState.WEAKENING, EvidenceState.IMPROVING, None, None, None)),
        _full_row(124, (EvidenceState.IMPROVING, EvidenceState.STABLE, None, None, None)),
        _full_row(125, (EvidenceState.IMPROVING, EvidenceState.IMPROVING, None, None, None)),
    ]
    report = full_evidence_index_diagnostics_v2(
        rows,
        policy=FullEvidenceIndexCoveragePolicyV2(
            minimum_rows_per_band=1,
            minimum_unique_issuers_per_band=1,
            minimum_unique_signal_months_per_band=1,
            minimum_total_unique_issuers=5,
            minimum_total_unique_signal_months=1,
            maximum_top_issuer_share_per_band=D(1),
            maximum_top_month_share_per_band=D(1),
            maximum_top_year_share_per_band=D(1),
        ),
    )

    assert report["coverage_gate_passed"] is True
    assert report["eligible_row_count"] == 5
    assert set(report["by_band"]) == {
        "STRONG_BEAR",
        "BEAR",
        "NEUTRAL",
        "BULL",
        "STRONG_BULL",
    }
    assert report["capex_included"] is False
    assert report["per_pbr_role"] == "NOT_USED"


def test_full_index_seal_fails_before_features_when_dual_locked_gate_has_not_passed(
    tmp_path: Path,
) -> None:
    manifests = {}
    payloads = {
        "sparse": {"outcome_vault_opened": False, "return_data_opened": False},
        "locked": {
            "status": "V2_LOCKED_TESTS_FAILED",
            "outcome_vault_opened": False,
            "return_data_opened": False,
        },
        "classification": {
            "outcome_vault_opened": False,
            "return_data_opened": False,
        },
        "selection": {"outcome_vault_opened": False, "return_data_opened": False},
        "cost": {
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
        },
        "core": {
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "value_data_opened": False,
            "per_pbr_role": "NOT_USED",
        },
    }
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        manifests[name] = path

    with pytest.raises(ValueError, match="dual LOCKED tests have not passed"):
        seal_full_evidence_index_v2(
            workspace=tmp_path,
            sparse_feature_input=tmp_path / "must-not-be-opened.jsonl",
            sparse_stage_manifest=manifests["sparse"],
            dual_locked_manifest=manifests["locked"],
            semantic_classification_stage_manifest=manifests["classification"],
            semantic_selection_manifest=manifests["selection"],
            cost_manifest=manifests["cost"],
            core_pre_outcome_manifest=manifests["core"],
            output=tmp_path / "full-seal",
            seal_tag="TEST",
            dry_run=True,
        )
    assert not (tmp_path / "full-seal").exists()


def test_full_index_dry_run_seals_five_axis_primary_after_every_prior_gate(
    tmp_path: Path,
) -> None:
    sparse_rows = [
        _breadth_row(201, (EvidenceState.WEAKENING, EvidenceState.WEAKENING)),
        _breadth_row(202, (EvidenceState.WEAKENING, EvidenceState.STABLE)),
        _breadth_row(203, (EvidenceState.STABLE, EvidenceState.STABLE)),
        _breadth_row(204, (EvidenceState.IMPROVING, EvidenceState.STABLE)),
        _breadth_row(205, (EvidenceState.IMPROVING, EvidenceState.IMPROVING)),
    ]
    sparse_input = tmp_path / "sparse-features.jsonl"
    sparse_input.write_text(
        "".join(row.model_dump_json() + "\n" for row in sparse_rows),
        encoding="utf-8",
    )
    spec = parser_spec(ParserProfile.DEMAND_PRICE_MIX_V2)
    classification_sha = "c" * 64
    packet_sha = "b" * 64

    def write_manifest(name: str, payload: dict[str, object]) -> Path:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    sparse_manifest = write_manifest(
        "sparse-stage",
        {
            "status": "SPARSE_FEATURES_BUILT_AWAITING_OUTCOME_BLIND_CALIBRATION",
            "pair_count": len(sparse_rows),
            "parser_directional_validation_passed": True,
            "missing_is_neutral": False,
            "deterministic_pit_priority_applied": True,
            "input_hashes": {
                "sparse_features": sha256_file(sparse_input),
                "classifications": classification_sha,
            },
        },
    )
    locked_manifest = write_manifest(
        "dual-locked",
        {
            "status": "V2_LOCKED_TESTS_PASSED",
            "natural_frequency_gate_passed": True,
            "directional_strata_gate_passed": True,
            "parser_version": spec.parser_version,
            "prompt_sha256": spec.prompt_sha256,
            "requested_model": "gpt-5.6-luna",
        },
    )
    classification_manifest = write_manifest(
        "classification-stage",
        {
            "status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE",
            "parser_profile": spec.profile.value,
            "parser_version": spec.parser_version,
            "prompt_sha256": spec.prompt_sha256,
            "requested_model": "gpt-5.6-luna",
            "classification_count": 5,
            "packet_count": 5,
            "input_blinded_packet_sha256": packet_sha,
            "classification_sha256": classification_sha,
        },
    )
    selection_manifest = write_manifest(
        "semantic-selection",
        {"output_packet_sha256": packet_sha, "selected_packet_count": 5},
    )
    cost_manifest = write_manifest(
        "cost",
        {
            "status": "FULL_SEMANTIC_RUN_COST_PRESPECIFIED_NO_EXTERNAL_CALL",
            "api_calls_executed": False,
            "parser_profile": spec.profile.value,
            "parser_version": spec.parser_version,
            "prompt_sha256": spec.prompt_sha256,
            "model": "gpt-5.6-luna",
            "exact_packet_count": 5,
        },
    )
    core_manifest = write_manifest(
        "core",
        {
            "status": "V2_EVIDENCE_INDEX_CONTRACT_FROZEN_OUTCOMES_CLOSED",
            "deterministic_core_materialized": True,
            "per_pbr_role": "NOT_USED",
            "source_provenance_gate": {
                "arcana_business_info_read": True,
                "arcana_finance_comment_read": True,
                "arcana_finance_statement_read": True,
                "moatrader_original_regular_filings_read": True,
                "all_expected_source_paths_verified": True,
                "source_files_modified": False,
            },
        },
    )
    output = tmp_path / "full-seal"
    status = seal_full_evidence_index_v2(
        workspace=Path.cwd(),
        sparse_feature_input=sparse_input,
        sparse_stage_manifest=sparse_manifest,
        dual_locked_manifest=locked_manifest,
        semantic_classification_stage_manifest=classification_manifest,
        semantic_selection_manifest=selection_manifest,
        cost_manifest=cost_manifest,
        core_pre_outcome_manifest=core_manifest,
        output=output,
        seal_tag="SYNTHETIC_SUCCESS_PATH",
        dry_run=True,
        coverage_policy=FullEvidenceIndexCoveragePolicyV2(
            minimum_rows_per_band=1,
            minimum_unique_issuers_per_band=1,
            minimum_unique_signal_months_per_band=1,
            minimum_total_unique_issuers=5,
            minimum_total_unique_signal_months=1,
            maximum_top_issuer_share_per_band=D(1),
            maximum_top_month_share_per_band=D(1),
            maximum_top_year_share_per_band=D(1),
        ),
    )

    assert status["status"] == (
        "DRY_RUN_V2_FULL_EVIDENCE_INDEX_VALIDATED_OUTCOMES_CLOSED"
    )
    assert status["full_index_materialized"] is True
    assert status["coverage_gate_passed"] is True
    assert status["outcome_stage_authorized"] is False
    assert status["primary_index"] == "FULL_EVIDENCE_SIGNED_BREADTH_V2"
    assert status["secondary_index"] == "DETERMINISTIC_CORE_SIGNED_BREADTH_V2"
    assert status["capex_included"] is False
    assert status["per_pbr_role"] == "NOT_USED"
    assert status["future_eri_role"] == "DOWNSTREAM_OUTCOME_ONLY_NOT_SIGNAL_OR_RANKING"
    assert (output / "full-evidence-index-eligible-nobs2.jsonl").is_file()


def test_t63_runner_opens_outcomes_only_after_full_primary_and_core_secondary_seals(
    tmp_path: Path,
) -> None:
    direction_rows = (
        (EvidenceState.WEAKENING,) * 5,
        (
            EvidenceState.WEAKENING,
            EvidenceState.WEAKENING,
            EvidenceState.WEAKENING,
            EvidenceState.STABLE,
            EvidenceState.STABLE,
        ),
        (
            EvidenceState.WEAKENING,
            EvidenceState.IMPROVING,
            EvidenceState.WEAKENING,
            EvidenceState.IMPROVING,
            EvidenceState.STABLE,
        ),
        (
            EvidenceState.IMPROVING,
            EvidenceState.IMPROVING,
            EvidenceState.IMPROVING,
            EvidenceState.STABLE,
            EvidenceState.STABLE,
        ),
        (EvidenceState.IMPROVING,) * 5,
    )
    core_directions = (
        (EvidenceState.WEAKENING,) * 3,
        (EvidenceState.WEAKENING, EvidenceState.WEAKENING, EvidenceState.STABLE),
        (EvidenceState.WEAKENING, EvidenceState.IMPROVING, EvidenceState.STABLE),
        (EvidenceState.IMPROVING, EvidenceState.IMPROVING, EvidenceState.STABLE),
        (EvidenceState.IMPROVING,) * 3,
    )
    full_rows = [
        _full_row(index, directions)
        for index, directions in zip(range(301, 306), direction_rows, strict=True)
    ]
    core_rows = [
        _core_row(index, directions)
        for index, directions in zip(range(301, 306), core_directions, strict=True)
    ]
    assert [row.band.value for row in full_rows] == [
        "STRONG_BEAR",
        "BEAR",
        "NEUTRAL",
        "BULL",
        "STRONG_BULL",
    ]
    assert [row.band.value for row in core_rows] == [
        "STRONG_BEAR",
        "BEAR",
        "NEUTRAL",
        "BULL",
        "STRONG_BULL",
    ]

    def write_jsonl(path: Path, rows: list[object]) -> None:
        path.write_text(
            "".join(
                json.dumps(
                    row.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
                for row in rows
            ),
            encoding="utf-8",
        )

    full_build = tmp_path / "full-index"
    core_build = tmp_path / "core-index"
    full_build.mkdir()
    core_build.mkdir()
    full_path = full_build / "full-evidence-index-eligible-nobs2.jsonl"
    core_path = core_build / "deterministic-core-index-eligible-nobs2.jsonl"
    write_jsonl(full_path, full_rows)
    write_jsonl(core_path, core_rows)
    core_manifest_path = core_build / "pre-outcome-index-manifest.json"
    core_manifest_path.write_text(
        json.dumps(
            {
                "status": "V2_EVIDENCE_INDEX_CONTRACT_FROZEN_OUTCOMES_CLOSED",
                "deterministic_core_materialized": True,
                "outcome_vault_opened": False,
                "return_data_opened": False,
                "value_data_opened": False,
                "per_pbr_role": "NOT_USED",
                "artifact_hashes": {
                    "deterministic_core_index_eligible_nobs2": sha256_file(core_path)
                },
            }
        ),
        encoding="utf-8",
    )
    full_seal_path = full_build / "full-evidence-index-seal.json"
    full_seal_path.write_text(
        json.dumps(
            {
                "outcome_vault_opened": False,
                "return_data_opened": False,
                "value_data_opened": False,
                "per_pbr_role": "NOT_USED",
                "artifact_hashes": {
                    "full_evidence_index_eligible_nobs2": sha256_file(full_path)
                },
                "input_hashes": {
                    "core_pre_outcome_manifest": sha256_file(core_manifest_path)
                },
            }
        ),
        encoding="utf-8",
    )
    (full_build / "stage-status.json").write_text(
        json.dumps(
            {
                "status": "V2_FULL_EVIDENCE_INDEX_SEALED_OUTCOMES_CLOSED",
                "outcome_stage_authorized": True,
                "full_index_materialized": True,
                "coverage_gate_passed": True,
                "semantic_parser_gate_passed": True,
                "full_evidence_index_seal_sha256": sha256_file(full_seal_path),
            }
        ),
        encoding="utf-8",
    )

    assumptions = EconomicDcfAssumptions(
        base_period="2020Q2",
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
    session_dates = [full_rows[0].signal_timestamp.date() + timedelta(days=i) for i in range(100)]
    target = target_trading_session(full_rows[0].signal_timestamp.date(), session_dates)
    target_price_at = datetime.combine(target, time(15, 30), tzinfo=SEOUL)
    realized_at = datetime.combine(target, time(8, 0), tzinfo=SEOUL)
    expectations: list[dict[str, object]] = []
    inventory: list[OutcomeEligibilityInventoryRowV1] = []
    outcomes: list[FutureEriOutcomeInputV1] = []
    for rank, row in enumerate(full_rows, start=1):
        expectations.append(
            {
                "observation_id": row.observation_id,
                "expectation_state": CurrentExpectationStateV1(
                    issuer_id=row.issuer_id,
                    signal_timestamp=row.signal_timestamp,
                    market_price=D("100"),
                    market_price_at=row.signal_timestamp,
                    market_price_source_id=f"KRX:{row.issuer_id}:SIGNAL",
                    implied_growth=D("0.08"),
                    implied_margin=D("0.12"),
                    implied_roiic=D("0.20"),
                    implied_cap_years=D("6"),
                    reverse_dcf_method="FCFF_FROZEN_PATH_V1",
                    reverse_dcf_input_sha256=f"{rank:x}" * 64,
                ).model_dump(mode="json"),
                "frozen_expectation_assumptions": assumptions.model_dump(mode="json"),
            }
        )
        inventory.append(
            OutcomeEligibilityInventoryRowV1(
                observation_id=row.observation_id,
                target_session=target,
                target_price_at=target_price_at,
                target_price_source_id=f"KRX:{row.issuer_id}:{target}:CLOSE",
                realized_financials_available_at=realized_at,
                realized_financial_source_ids=[f"DART:{row.issuer_id}:2020Q3"],
                net_debt_source_id=f"DART:{row.issuer_id}:NET_DEBT",
                diluted_shares_source_id=f"DART:{row.issuer_id}:SHARES",
                wacc_source_id="FROZEN_WACC_POLICY",
            )
        )
        outcomes.append(
            FutureEriOutcomeInputV1(
                observation_id=row.observation_id,
                target_session=target,
                target_price_at=target_price_at,
                actual_market_price=D(80 + rank * 10),
                target_price_source_id=f"KRX:{row.issuer_id}:{target}:CLOSE",
                realized_state=RealizedFcffStateV1(
                    available_at=realized_at,
                    base_period="2020Q3",
                    base_revenue=D("1040"),
                    base_nopat_margin=D("0.11"),
                    base_invested_capital=D("820"),
                    net_debt=D("90"),
                    diluted_shares=D("10"),
                    wacc=D("0.085"),
                    wacc_source_id="FROZEN_WACC_POLICY",
                    source_document_ids=[f"DART:{row.issuer_id}:2020Q3"],
                ),
            )
        )

    expectation_path = tmp_path / "expectations.jsonl"
    expectation_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in expectations),
        encoding="utf-8",
    )
    inventory_path = tmp_path / "eligibility.jsonl"
    outcome_path = tmp_path / "outcomes.jsonl"
    write_jsonl(inventory_path, inventory)
    write_jsonl(outcome_path, outcomes)
    sessions_path = tmp_path / "sessions.json"
    sessions_path.write_text(
        json.dumps([value.isoformat() for value in session_dates]),
        encoding="utf-8",
    )
    output = tmp_path / "t63-result"
    status = run_evidence_index_eri_v2(
        full_index_build=full_build,
        core_index_build=core_build,
        expectation_input=expectation_path,
        eligibility_inventory_input=inventory_path,
        outcome_input=outcome_path,
        trading_sessions_path=sessions_path,
        output=output,
        minimum_observations_per_band=1,
        hac_lag_months=1,
    )

    assert status["label_count"] == 5
    assert status["common_full_core_panel"] is True
    assert status["primary_endpoint"] == "FULL_EVIDENCE_INDEX_TO_FUTURE_ERI_T63"
    assert status["secondary_endpoint"] == "CORE_EVIDENCE_INDEX_TO_FUTURE_ERI_T63"
    assert status["outcome_vault_opened"] is True
    assert status["future_eri_used_as_signal"] is False
    assert status["future_eri_used_as_ranking"] is False
    assert status["return_data_opened"] is False
    assert status["per_pbr_role"] == "NOT_USED"
    build_manifest = json.loads((output / "build-manifest.json").read_text(encoding="utf-8"))
    assert build_manifest["outcome_opened_only_after_common_feature_seal"] is True
    assert (output / "feature-seal-pre-outcome.json").is_file()


def test_deterministic_core_diagnostics_report_all_fixed_bands() -> None:
    rows = [
        _core_row(111, (EvidenceState.WEAKENING, EvidenceState.WEAKENING, None)),
        _core_row(112, (EvidenceState.WEAKENING, EvidenceState.STABLE, None)),
        _core_row(113, (EvidenceState.WEAKENING, EvidenceState.IMPROVING, None)),
        _core_row(114, (EvidenceState.IMPROVING, EvidenceState.STABLE, None)),
        _core_row(115, (EvidenceState.IMPROVING, EvidenceState.IMPROVING, None)),
    ]
    diagnostics = deterministic_core_diagnostics_v2(
        rows,
        policy=DeterministicCoreIndexCoveragePolicyV2(
            minimum_rows_per_band=1,
            minimum_unique_issuers_per_band=1,
            minimum_unique_signal_months_per_band=1,
            minimum_total_unique_issuers=5,
            minimum_total_unique_signal_months=1,
            maximum_top_issuer_share_per_band=D(1),
            maximum_top_month_share_per_band=D(1),
            maximum_top_year_share_per_band=D(1),
        ),
    )

    assert diagnostics["coverage_gate_passed"] is True
    assert diagnostics["eligible_row_count"] == 5
    assert set(diagnostics["by_band"]) == {
        "STRONG_BEAR",
        "BEAR",
        "NEUTRAL",
        "BULL",
        "STRONG_BULL",
    }
    assert {values["row_count"] for values in diagnostics["by_band"].values()} == {1}


def _pit_snapshot(
    *,
    current: bool,
    sources: bool = True,
) -> PITOperatingSnapshotV2:
    suffix = "CURRENT" if current else "PREVIOUS"
    return PITOperatingSnapshotV2(
        issuer_id="000001",
        fiscal_period_end=date(2020, 6 if current else 3, 30 if current else 31),
        available_at=datetime(2020, 8 if current else 3, 14 if current else 30, 16, tzinfo=SEOUL),
        source_ids=(
            {
                metric: [f"{suffix}_{metric}"]
                for metric in (
                    "revenue",
                    "operating_profit",
                    "inventory",
                    "assets",
                    "backlog",
                    "capex",
                    "ppe",
                )
            }
            if sources
            else {}
        ),
        revenue=D(120 if current else 100),
        operating_profit=D(18 if current else 10),
        inventory=D(15 if current else 10),
        assets=D(200),
        backlog=D(130 if current else 100),
        capex=D(9 if current else 5),
        ppe=D(80),
        backlog_disclosed=True,
        capacity_disclosed=True,
    )


def test_deterministic_pit_axes_have_priority_ready_directions() -> None:
    evidence = build_deterministic_pit_axis_evidence(
        previous=_pit_snapshot(current=False),
        current=_pit_snapshot(current=True),
        rules=PITApplicabilityRulesV2(),
    )

    assert evidence[OperatingEvidenceAxis.MARGIN].direction == EvidenceState.IMPROVING
    assert evidence[OperatingEvidenceAxis.INVENTORY_MISMATCH].direction == EvidenceState.WEAKENING
    assert evidence[OperatingEvidenceAxis.BACKLOG].direction == EvidenceState.IMPROVING
    assert evidence[OperatingEvidenceAxis.CAPACITY_CAPEX].direction == EvidenceState.IMPROVING
    assert evidence[OperatingEvidenceAxis.MARGIN].provenance == (
        AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC
    )
    assert evidence[OperatingEvidenceAxis.INVENTORY_MISMATCH].provenance == (
        AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC
    )
    assert evidence[OperatingEvidenceAxis.BACKLOG].provenance == (
        AxisEvidenceProvenanceV2.STRUCTURED_TABLE
    )
    assert evidence[OperatingEvidenceAxis.CAPACITY_CAPEX].provenance == (
        AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC
    )
    assert evidence[OperatingEvidenceAxis.CAPACITY_CAPEX].signed_score_role == (
        AxisSignedScoreRoleV2.RAW_DIRECTION_ONLY
    )


def test_capex_axis_uses_net_ppe_intensity_fallback_as_raw_direction() -> None:
    previous = _pit_snapshot(current=False).model_copy(
        update={
            "capex": None,
            "ppe": D(50),
            "source_ids": {
                key: value
                for key, value in _pit_snapshot(current=False).source_ids.items()
                if key != "capex"
            },
        }
    )
    current = _pit_snapshot(current=True).model_copy(
        update={
            "capex": None,
            "ppe": D(80),
            "source_ids": {
                key: value
                for key, value in _pit_snapshot(current=True).source_ids.items()
                if key != "capex"
            },
        }
    )

    capex = build_deterministic_pit_axis_evidence(
        previous=previous,
        current=current,
        rules=PITApplicabilityRulesV2(),
    )[OperatingEvidenceAxis.CAPACITY_CAPEX]

    assert capex.availability == SparseAxisAvailabilityV2.GROUNDED
    assert capex.deterministic_metric_name == "NET_PPE_TO_ASSETS"
    assert capex.direction == EvidenceState.IMPROVING
    assert capex.signed_score_role == AxisSignedScoreRoleV2.RAW_DIRECTION_ONLY


def test_pit_availability_order_violation_is_na_not_lookahead() -> None:
    previous = _pit_snapshot(current=False).model_copy(
        update={"available_at": datetime(2020, 8, 20, 16, tzinfo=SEOUL)}
    )
    current = _pit_snapshot(current=True)

    evidence = build_deterministic_pit_axis_evidence(
        previous=previous,
        current=current,
        rules=PITApplicabilityRulesV2(),
    )

    assert set(evidence) == {
        OperatingEvidenceAxis.MARGIN,
        OperatingEvidenceAxis.INVENTORY_MISMATCH,
        OperatingEvidenceAxis.BACKLOG,
        OperatingEvidenceAxis.CAPACITY_CAPEX,
    }
    assert all(
        row.availability == SparseAxisAvailabilityV2.NA
        and row.abstention_reason == AbstentionReasonV2.PERIOD_MISMATCH
        and row.direction is None
        for row in evidence.values()
    )


def test_pit_html_extractor_uses_current_cumulative_values_and_structured_backlog() -> None:
    document = """
    <html><body>
      <p>(단위 : 백만원)</p>
      <table><thead><tr><th>계정</th><th>당기 3개월</th><th>당기 누적</th><th>전기 3개월</th><th>전기 누적</th></tr></thead>
        <tbody>
          <tr><td>매출액</td><td>100</td><td>200</td><td>80</td><td>160</td></tr>
          <tr><td>영업이익(손실)</td><td>10</td><td>20</td><td>8</td><td>16</td></tr>
        </tbody>
      </table>
      <p>(단위 : 백만원)</p>
      <table><tbody>
        <tr><td>재고자산</td><td>50</td><td>45</td></tr>
        <tr><td>유형자산</td><td>200</td><td>180</td></tr>
        <tr><td>자산총계</td><td>500</td><td>450</td></tr>
      </tbody></table>
      <p>(단위 : 백만원)</p>
      <table><tbody>
        <tr><td>유형자산의 취득</td><td>(30)</td><td>(25)</td></tr>
        <tr><td>무형자산의 취득</td><td>(5)</td><td>(4)</td></tr>
      </tbody></table>
      <p>생산설비 증설</p>
      <p>(단위 : 백만원)</p>
      <table><thead><tr><th>공사명</th><th>계약잔액</th></tr></thead>
        <tbody>
          <tr><td>A</td><td>30</td></tr><tr><td>B</td><td>40</td></tr>
          <tr><td>합계</td><td>70</td></tr>
        </tbody>
      </table>
    </body></html>
    """

    metrics = extract_pit_metrics_from_html(
        document,
        fiscal_period_end=date(2024, 6, 30),
    )

    assert metrics["revenue"] == D(200_000_000)
    assert metrics["operating_profit"] == D(20_000_000)
    assert metrics["inventory"] == D(50_000_000)
    assert metrics["assets"] == D(500_000_000)
    assert metrics["ppe"] == D(200_000_000)
    assert metrics["capex"] == D(35_000_000)
    assert metrics["backlog"] == D(70_000_000)
    assert metrics["backlog_disclosed"] is True
    assert metrics["capacity_disclosed"] is True

    embedded_unit_metrics = extract_pit_metrics_from_html(
        """
        <html><body><table>
          <tr><td>단위 : 억원</td></tr>
          <tr><td>매출액</td><td>3</td></tr>
          <tr><td>영업이익</td><td>1</td></tr>
        </table></body></html>
        """,
        fiscal_period_end=date(2024, 12, 31),
    )
    assert embedded_unit_metrics["revenue"] == D(300_000_000)
    assert embedded_unit_metrics["operating_profit"] == D(100_000_000)


def test_pit_html_extractor_ignores_row_indices_and_trailing_note_references() -> None:
    metrics = extract_pit_metrics_from_html(
        """
        <html><body><p>(단위 : 백만원)</p>
          <table><tbody>
            <tr><td>(6)재고자산</td><td>50</td><td>45</td></tr>
            <tr><td>(4)유형자산 (주11, 14)</td><td>200</td><td>180</td></tr>
            <tr><td>자산총계</td><td>500</td><td>450</td></tr>
          </tbody></table>
          <table><tbody>
            <tr><td>Ⅰ.매출액 (주34)</td><td>100</td><td>90</td></tr>
            <tr><td>V.영업이익(손실) (주35)</td><td>10</td><td>8</td></tr>
          </tbody></table>
          <table><tbody>
            <tr><td>(20)유형자산의 취득 (주14)</td><td>(30)</td><td>(25)</td></tr>
            <tr><td>(21)무형자산의 취득 [주석 15]</td><td>(5)</td><td>(4)</td></tr>
          </tbody></table>
        </body></html>
        """,
        fiscal_period_end=date(2024, 12, 31),
    )

    assert metrics["revenue"] == D(100_000_000)
    assert metrics["operating_profit"] == D(10_000_000)
    assert metrics["inventory"] == D(50_000_000)
    assert metrics["assets"] == D(500_000_000)
    assert metrics["ppe"] == D(200_000_000)
    assert metrics["capex"] == D(35_000_000)

    exact_variant_metrics = extract_pit_metrics_from_html(
        """
        <html><body><table>
          <tr><td>매출액 계</td><td>120</td></tr>
          <tr><td>영업손익</td><td>(12)</td></tr>
        </table></body></html>
        """,
        fiscal_period_end=date(2024, 12, 31),
    )
    duplicate_label_metrics = extract_pit_metrics_from_html(
        """
        <html><body><table>
          <tr><td>매출액(매출액)</td><td>130</td></tr>
          <tr><td>영업이익</td><td>13</td></tr>
        </table></body></html>
        """,
        fiscal_period_end=date(2024, 12, 31),
    )
    assert exact_variant_metrics["revenue"] == D(120)
    assert exact_variant_metrics["operating_profit"] == D(-12)
    assert duplicate_label_metrics["revenue"] == D(130)
    assert duplicate_label_metrics["operating_profit"] == D(13)


def test_pit_filing_task_reads_all_arcana_sections_and_moatrader_original(
    tmp_path: Path,
) -> None:
    documents = {
        "finance-statement.html": "<html><body><p>재무제표 본문</p></body></html>",
        "finance-comment.html": """
            <html><body><table>
              <tr><td>매출액</td><td>100</td></tr>
              <tr><td>영업이익</td><td>10</td></tr>
              <tr><td>재고자산</td><td>20</td></tr>
              <tr><td>자산총계</td><td>200</td></tr>
            </table></body></html>
        """,
        "business-info.html": """
            <html><body><table>
              <tr><th>공사명</th><th>수주잔고</th></tr>
              <tr><td>합계</td><td>80</td></tr>
            </table></body></html>
        """,
    }
    paths: dict[str, Path] = {}
    for name, document in documents.items():
        path = tmp_path / name
        path.write_text(document, encoding="utf-8")
        paths[name] = path
    archive_path = tmp_path / "original.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "report.xml",
            """
            <html><body><table>
              <tr><td>유형자산의 취득</td><td>(30)</td></tr>
            </table></body></html>
            """,
        )

    def source(name: str, origin: str) -> FilingSource:
        path = archive_path if name == "original.zip" else paths[name]
        digest = sha256_file(path)
        return FilingSource(
            source_id=f"PIT_SRC_{digest[:20]}",
            origin=origin,
            path=str(path),
            raw_sha256=digest,
        )

    result = _extract_task(
        FilingTask(
            ticker="000001",
            rcept_no="20240515000001",
            fiscal_period_end="2024-03-31",
            available_at="2024-05-15T16:00:00+09:00",
            finance_statement=source(
                "finance-statement.html", "ARCANA_FINANCE_STATEMENT_HTML"
            ),
            finance_comment=source(
                "finance-comment.html", "ARCANA_FINANCE_COMMENT_HTML"
            ),
            business_info=source("business-info.html", "ARCANA_BUSINESS_HTML"),
            moatrader_original=source("original.zip", "MOATRADER_OPENDART_ARCHIVE"),
        )
    )

    assert len(result["verified_hashes"]) == 4
    assert result["origins"]["revenue"] == "ARCANA_FINANCE_COMMENT_HTML"
    assert result["origins"]["backlog"] == "ARCANA_BUSINESS_HTML"
    assert result["origins"]["capex"] == "MOATRADER_OPENDART_ARCHIVE"
    assert result["snapshot"]["capacity_disclosed"] is True


def test_last_grounded_respects_frozen_staleness_window() -> None:
    current = GroundedAxisStateSnapshotV2(
        axis=OperatingEvidenceAxis.DEMAND,
        state=EvidenceState.IMPROVING,
        fiscal_period_end=date(2021, 6, 30),
        available_at=datetime(2021, 8, 15, tzinfo=SEOUL),
        source_ids=["CURRENT"],
    )
    recent = GroundedAxisStateSnapshotV2(
        axis=OperatingEvidenceAxis.DEMAND,
        state=EvidenceState.STABLE,
        fiscal_period_end=date(2020, 6, 30),
        available_at=current.available_at - timedelta(days=449),
        source_ids=["PRIOR"],
    )
    stale = recent.model_copy(
        update={"available_at": current.available_at - timedelta(days=451)}
    )

    grounded = build_last_grounded_axis_evidence(
        current=current,
        history=[recent],
        staleness_limit_days=450,
        applicability_rule_id="LAST_GROUNDED_450D",
    )
    unavailable = build_last_grounded_axis_evidence(
        current=current,
        history=[stale],
        staleness_limit_days=450,
        applicability_rule_id="LAST_GROUNDED_450D",
    )

    assert grounded.provenance == AxisEvidenceProvenanceV2.LLM_NARRATIVE
    assert grounded.previous_evidence_basis == (
        PreviousEvidenceBasisV2.LAST_GROUNDED_WITHIN_STALENESS
    )
    assert grounded.prior_age_days == 449
    assert unavailable.availability == SparseAxisAvailabilityV2.NA
    assert unavailable.abstention_reason == AbstentionReasonV2.STALE_PRIOR_STATE


def test_last_grounded_never_replaces_missing_current_evidence() -> None:
    history = [
        GroundedAxisStateSnapshotV2(
            axis=OperatingEvidenceAxis.DEMAND,
            state=EvidenceState.IMPROVING,
            fiscal_period_end=date(2023, 3, 31),
            available_at=datetime(2023, 5, 15, tzinfo=SEOUL),
            source_ids=["2023Q1"],
        )
    ]

    result = build_last_grounded_axis_evidence(
        current=None,
        history=history,
        staleness_limit_days=450,
        applicability_rule_id="LAST_GROUNDED_450D",
        axis=OperatingEvidenceAxis.DEMAND,
    )

    assert result.availability == SparseAxisAvailabilityV2.NA
    assert result.direction is None
    assert result.abstention_reason == AbstentionReasonV2.TRUE_NO_MENTION


def test_inventory_economic_polarity_is_inverted_from_raw_mismatch() -> None:
    assert inventory_economic_direction(D("-0.051"), D("0.05")) == EvidenceState.IMPROVING
    assert inventory_economic_direction(D("0.051"), D("0.05")) == EvidenceState.WEAKENING
    assert inventory_economic_direction(D("0.05"), D("0.05")) == EvidenceState.STABLE


def test_selection_2_uses_last_grounded_only_as_previous_numeric_basis(
    tmp_path: Path,
) -> None:
    def snapshot(
        *,
        period: date,
        available_at: datetime,
        revenue: int,
        inventory: int | None,
    ) -> PITOperatingSnapshotV2:
        values = {
            "revenue": D(revenue),
            "operating_profit": D(revenue) / D(10),
            "inventory": D(inventory) if inventory is not None else None,
            "assets": D(500),
            "backlog": D(revenue),
            "capex": D(20),
            "ppe": D(200),
        }
        return PITOperatingSnapshotV2(
            issuer_id="000001",
            fiscal_period_end=period,
            available_at=available_at,
            source_ids={
                metric: [f"SRC_{period.strftime('%Y%m%d')}_{metric}"]
                for metric, value in values.items()
                if value is not None
            },
            backlog_disclosed=True,
            capacity_disclosed=True,
            **values,
        )

    q1 = snapshot(
        period=date(2023, 3, 31),
        available_at=datetime(2023, 5, 15, 16, tzinfo=SEOUL),
        revenue=100,
        inventory=20,
    )
    q2 = snapshot(
        period=date(2023, 6, 30),
        available_at=datetime(2023, 8, 14, 16, tzinfo=SEOUL),
        revenue=110,
        inventory=None,
    )
    q3 = snapshot(
        period=date(2023, 9, 30),
        available_at=datetime(2023, 11, 14, 16, tzinfo=SEOUL),
        revenue=130,
        inventory=18,
    )
    pairs = [
        PITOperatingPairInputV2(pair_id=f"PAIR_{1:024x}", previous=q1, current=q2),
        PITOperatingPairInputV2(pair_id=f"PAIR_{2:024x}", previous=q2, current=q3),
    ]
    pair_input = tmp_path / "pit-pairs.jsonl"
    pair_input.write_text(
        "".join(pair.model_dump_json() + "\n" for pair in pairs),
        encoding="utf-8",
    )
    rules_input = tmp_path / "rules.json"
    rules_input.write_text(PITApplicabilityRulesV2().model_dump_json(), encoding="utf-8")
    prepared_output = tmp_path / "last-grounded"
    prepared = prepare_last_grounded_inputs(
        pit_pair_input=pair_input,
        rules_input=rules_input,
        output=prepared_output,
    )
    last_input = prepared_output / "last-grounded-deterministic-bases.jsonl"
    rows = [
        LastGroundedDeterministicBasisInputV2.model_validate_json(line)
        for line in last_input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert prepared["recovered_count"] == 1
    assert len(rows) == 1
    assert rows[0].pair_id == pairs[1].pair_id
    assert rows[0].axis == OperatingEvidenceAxis.INVENTORY_MISMATCH
    assert rows[0].previous.fiscal_period_end == q1.fiscal_period_end
    assert rows[0].current_evidence_carried_forward is False

    evidence_output = tmp_path / "evidence"
    stage = build_pit_evidence(
        pit_pair_input=pair_input,
        rules_input=rules_input,
        last_grounded_input=last_input,
        output=evidence_output,
    )
    evidence_rows = [
        DeterministicAxisEvidenceInputV2.model_validate_json(line)
        for line in (evidence_output / "deterministic-axis-evidence.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    recovered = next(
        row.evidence
        for row in evidence_rows
        if row.pair_id == pairs[1].pair_id
        and row.evidence.axis == OperatingEvidenceAxis.INVENTORY_MISMATCH
    )
    missing_current = next(
        row.evidence
        for row in evidence_rows
        if row.pair_id == pairs[0].pair_id
        and row.evidence.axis == OperatingEvidenceAxis.INVENTORY_MISMATCH
    )

    assert stage["last_grounded_row_count"] == 1
    assert recovered.availability == SparseAxisAvailabilityV2.GROUNDED
    assert recovered.previous_evidence_basis == (
        PreviousEvidenceBasisV2.LAST_GROUNDED_WITHIN_STALENESS
    )
    assert recovered.previous_evidence_at == q1.available_at
    assert recovered.current_evidence_at == q3.available_at
    assert missing_current.availability == SparseAxisAvailabilityV2.NA
    assert missing_current.direction is None


def test_numeric_beats_table_and_llm_without_averaging() -> None:
    pair = _pair()
    llm = _grounded(OperatingEvidenceAxis.MARGIN, EvidenceState.STABLE, pair=pair).model_copy(
        update={"provenance": AxisEvidenceProvenanceV2.LLM_NARRATIVE}
    )
    numeric = _grounded(
        OperatingEvidenceAxis.MARGIN,
        EvidenceState.WEAKENING,
        pair=pair,
    )

    merged = merge_axis_evidence_v2(llm, numeric)

    assert merged.direction == EvidenceState.WEAKENING
    assert merged.provenance == AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC


def _breadth_row(index: int, directions: tuple[EvidenceState, EvidenceState]):
    pair = _pair(index)
    axes = list(OperatingEvidenceAxis)
    evidence = [
        _grounded(axes[0], directions[0], pair=pair),
        _grounded(axes[1], directions[1], pair=pair),
        *[_na(axis) for axis in axes[2:]],
    ]
    return build_sparse_feature_row_v2(pair=pair, axis_evidence=evidence)


def test_feature_only_band_calibration_requires_explicit_nobs_and_covers_five_bands() -> None:
    rows = [
        _breadth_row(1, (EvidenceState.WEAKENING, EvidenceState.WEAKENING)),
        _breadth_row(2, (EvidenceState.WEAKENING, EvidenceState.STABLE)),
        _breadth_row(3, (EvidenceState.STABLE, EvidenceState.STABLE)),
        _breadth_row(4, (EvidenceState.IMPROVING, EvidenceState.STABLE)),
        _breadth_row(5, (EvidenceState.IMPROVING, EvidenceState.IMPROVING)),
    ]
    contract = calibrate_sparse_band_contract_v2(
        rows,
        minimum_observed_axes=2,
        minimum_rows_per_band=1,
    )
    report = sparse_feature_coverage_report(rows)
    diagnostics = sparse_band_diagnostics_v2(rows, band_contract=contract)
    gate = evaluate_sparse_coverage_gate_v2(
        diagnostics,
        policy=SparseCoverageGatePolicyV2(
            minimum_rows_per_band=1,
            minimum_unique_issuers_per_band=1,
            minimum_unique_signal_months_per_band=1,
            minimum_total_unique_issuers=5,
            minimum_total_unique_signal_months=1,
            maximum_top_issuer_share_per_band=D(1),
            maximum_top_year_share_per_band=D(1),
            maximum_top_evidence_source_share_per_band=D(1),
        ),
    )
    concentrated_gate = evaluate_sparse_coverage_gate_v2(
        diagnostics,
        policy=SparseCoverageGatePolicyV2(
            minimum_rows_per_band=1,
            minimum_unique_issuers_per_band=1,
            minimum_unique_signal_months_per_band=1,
            minimum_total_unique_issuers=5,
            minimum_total_unique_signal_months=1,
            maximum_top_issuer_share_per_band=D("0.5"),
            maximum_top_year_share_per_band=D(1),
            maximum_top_evidence_source_share_per_band=D(1),
        ),
    )

    assert contract.all_bands_sufficient is True
    assert set(contract.band_counts.values()) == {1}
    assert contract.calibration_method == "FIXED_ECONOMIC_SIGN_BANDS_V2"
    assert contract.band_for(D(-1)).value == "STRONG_BEAR"
    assert contract.band_for(D(0)).value == "NEUTRAL"
    assert contract.band_for(D(1)).value == "STRONG_BULL"
    assert report["grounded_axis_count_histogram"]["2"] == 5
    assert len(report["signed_breadth_distribution"]) == 5
    assert report["n_directional_histogram"]["0"] == 1
    assert report["by_exact_nobs"]["2"]["row_count"] == 5
    assert len(report["by_exact_nobs"]["2"]["observations"]) == 5
    assert report["by_exact_nobs"]["2"]["observations"][0]["directional_ratio"] is not None
    assert diagnostics["corr_abs_signed_breadth_nobs"] is None
    assert set(diagnostics["by_band"]) == {band.value for band in contract.band_counts}
    assert gate["gate_passed"] is True
    assert concentrated_gate["gate_passed"] is False


def test_axis_available_subset_does_not_require_six_axis_complete_pair(tmp_path: Path) -> None:
    pair = _pair()
    packets = [
        _packet(pair, axis, both=axis != OperatingEvidenceAxis.BACKLOG)
        for axis in OperatingEvidenceAxis
    ]
    packet_input = tmp_path / "all.jsonl"
    packet_input.write_text(
        "".join(item.model_dump_json() + "\n" for item in packets),
        encoding="utf-8",
    )
    output = tmp_path / "axis-available.jsonl"

    manifest = prepare_subset(
        packet_input=packet_input,
        output=output,
        mode="AXIS_AVAILABLE",
        expected_candidate_pairs=1,
    )

    assert manifest["total_pair_count"] == 1
    assert manifest["selected_pair_count"] == 1
    assert manifest["selected_packet_count"] == 5
    assert manifest["candidate_grounded_axis_count_histogram"]["5"] == 1


def test_semantic_selection_keeps_only_demand_and_price_mix(
    tmp_path: Path,
) -> None:
    pair = _pair()
    packets = [_packet(pair, axis) for axis in OperatingEvidenceAxis]
    pair_path = tmp_path / "pairs.jsonl"
    packet_path = tmp_path / "packets.jsonl"
    pair_path.write_text(pair.model_dump_json() + "\n", encoding="utf-8")
    packet_path.write_text(
        "".join(packet.model_dump_json() + "\n" for packet in packets),
        encoding="utf-8",
    )
    deterministic: list[DeterministicAxisEvidenceInputV2] = []
    for axis in (
        OperatingEvidenceAxis.MARGIN,
        OperatingEvidenceAxis.INVENTORY_MISMATCH,
        OperatingEvidenceAxis.BACKLOG,
        OperatingEvidenceAxis.CAPACITY_CAPEX,
    ):
        if axis == OperatingEvidenceAxis.MARGIN:
            evidence = _grounded(axis, EvidenceState.WEAKENING, pair=pair)
        elif axis == OperatingEvidenceAxis.BACKLOG:
            evidence = _not_applicable(axis)
        else:
            evidence = _na(axis)
        deterministic.append(
            DeterministicAxisEvidenceInputV2(pair_id=pair.pair_id, evidence=evidence)
        )
    applicability = [
        AxisApplicabilityDecisionInputV2(
            pair_id=pair.pair_id,
            axis=axis,
            applicability=(
                AxisApplicabilityV2.NOT_APPLICABLE
                if axis == OperatingEvidenceAxis.BACKLOG
                else AxisApplicabilityV2.APPLICABLE
            ),
            rule_id="TEST_PIT_APPLICABILITY",
        )
        for axis in OperatingEvidenceAxis
    ]
    deterministic_path = tmp_path / "deterministic.jsonl"
    applicability_path = tmp_path / "applicability.jsonl"
    deterministic_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in deterministic), encoding="utf-8"
    )
    applicability_path.write_text(
        "".join(row.model_dump_json() + "\n" for row in applicability), encoding="utf-8"
    )
    output = tmp_path / "semantic.jsonl"

    manifest = prepare_semantic_packets(
        filing_pair_input=pair_path,
        packet_input=packet_path,
        deterministic_evidence_input=deterministic_path,
        applicability_input=applicability_path,
        output=output,
        expected_pair_count=1,
    )
    selected = [
        PairedAxisPacket.model_validate_json(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]

    assert {packet.axis for packet in selected} == {
        OperatingEvidenceAxis.DEMAND,
        OperatingEvidenceAxis.PRICE_MIX,
    }
    assert manifest["selected_packet_count"] == 2
    assert manifest["capacity_narrative_fallback_enabled"] is False
    assert manifest["qualitative_diagnostics_for_numeric_axes"] is False


def test_sparse_builder_retains_all_pairs_and_unclassified_axes_as_na(tmp_path: Path) -> None:
    pair = _pair()
    input_build = tmp_path / "input"
    (input_build / "private").mkdir(parents=True)
    (input_build / "llm").mkdir()
    (input_build / "private" / "filing-pairs.jsonl").write_text(
        pair.model_dump_json() + "\n", encoding="utf-8"
    )
    packets = [_packet(pair, axis) for axis in OperatingEvidenceAxis]
    (input_build / "llm" / "blinded-packets.jsonl").write_text(
        "".join(item.model_dump_json() + "\n" for item in packets), encoding="utf-8"
    )
    demand = packets[0]
    classification = AxisPairClassification(
        packet_id=demand.packet_id,
        axis=demand.axis,
        previous_state=EvidenceState.STABLE,
        current_state=EvidenceState.IMPROVING,
        previous_source_id=demand.previous_excerpts[0].source_id,
        current_source_id=demand.current_excerpts[0].source_id,
        previous_source_span=demand.previous_excerpts[0].text,
        current_source_span=demand.current_excerpts[0].text,
        confidence=1,
    )
    classification_build = tmp_path / "classification"
    classification_build.mkdir()
    (classification_build / "classifications.jsonl").write_text(
        classification.model_dump_json() + "\n", encoding="utf-8"
    )
    (classification_build / "stage-status.json").write_text(
        json.dumps({"status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE"}),
        encoding="utf-8",
    )

    result = build_sparse_features(
        input_build=input_build,
        classification_build=classification_build,
        output=tmp_path / "features",
        coverage_only_unvalidated=True,
        allow_test_input_without_source_audit=True,
    )
    stored = json.loads(
        (tmp_path / "features" / "sparse-features-all-pairs.jsonl").read_text(
            encoding="utf-8"
        )
    )

    assert result["pair_count"] == 1
    assert stored["observed_axis_count"] == 1
    assert stored["unavailable_axis_count"] == 5
    assert stored["signed_breadth"] == "1"
    assert result["outcome_stage_authorized"] is False


def test_calibration_diagnostics_never_auto_select_nobs(tmp_path: Path) -> None:
    rows = [
        _breadth_row(index, directions)
        for index, directions in enumerate(
            (
                (EvidenceState.WEAKENING, EvidenceState.WEAKENING),
                (EvidenceState.WEAKENING, EvidenceState.STABLE),
                (EvidenceState.STABLE, EvidenceState.STABLE),
                (EvidenceState.IMPROVING, EvidenceState.STABLE),
                (EvidenceState.IMPROVING, EvidenceState.IMPROVING),
            ),
            start=1,
        )
    ]
    feature_build = tmp_path / "feature-build"
    feature_build.mkdir()
    (feature_build / "sparse-features-all-pairs.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    (feature_build / "stage-status.json").write_text(
        json.dumps(
            {
                "status": "SPARSE_FEATURES_BUILT_AWAITING_OUTCOME_BLIND_CALIBRATION",
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )

    status = calibrate_sparse_features(
        feature_build=feature_build,
        output=tmp_path / "diagnostics",
    )

    assert status["minimum_observed_axes"] is None
    assert status["minimum_observed_axes_auto_selected"] is False
    assert status["outcome_stage_authorized"] is False


def test_sparse_freeze_opens_gate_only_after_all_v2_manifests_pass(tmp_path: Path) -> None:
    rows = [
        _breadth_row(index, directions)
        for index, directions in enumerate(
            (
                (EvidenceState.WEAKENING, EvidenceState.WEAKENING),
                (EvidenceState.WEAKENING, EvidenceState.STABLE),
                (EvidenceState.STABLE, EvidenceState.STABLE),
                (EvidenceState.IMPROVING, EvidenceState.STABLE),
                (EvidenceState.IMPROVING, EvidenceState.IMPROVING),
            ),
            start=1,
        )
    ]
    parser_manifest = tmp_path / "parser-v2.json"
    parser_manifest.write_text(
        json.dumps(
            {
                "status": "V2_LOCKED_TESTS_PASSED",
                "natural_frequency_gate_passed": True,
                "directional_strata_gate_passed": True,
                "parser_freeze_sha256": "f" * 64,
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    audit_manifest = tmp_path / "audit-v2.json"
    audit_manifest.write_text(
        json.dumps(
            {
                "status": "V2_ABSTENTION_AUDIT_PASSED",
                "upstream_extraction_gate_passed": True,
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    contract_manifest = tmp_path / "contract-v2.json"
    contract_manifest.write_text(
        json.dumps(
            {
                "status": "V2_PRE_OUTCOME_CONTRACT_FROZEN",
                "contract_tag": "v2-fixture",
                "git_commit": "a" * 40,
                "worktree_dirty": False,
                "dry_run_only": False,
                "parser_freeze_sha256": "f" * 64,
                "feature_policy_sha256": "1" * 64,
                "applicability_policy_sha256": "2" * 64,
                "deterministic_axis_policy_sha256": "3" * 64,
                "evidence_priority_sha256": "4" * 64,
                "parser_prompt_sha256": "5" * 64,
                "signal_timestamp_policy": (
                    "FIRST_TRADABLE_TIMESTAMP_AFTER_CURRENT_REGULAR_FILING_AVAILABLE_AT"
                ),
                "last_grounded_days": 450,
                "last_grounded_role": (
                    "PREVIOUS_COMPARISON_BASIS_ONLY_NEVER_CURRENT_EVIDENCE"
                ),
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    coverage_policy = tmp_path / "coverage-policy.json"
    coverage_policy.write_text(
        SparseCoverageGatePolicyV2(
            minimum_rows_per_band=1,
            minimum_unique_issuers_per_band=1,
            minimum_unique_signal_months_per_band=1,
            minimum_total_unique_issuers=5,
            minimum_total_unique_signal_months=1,
            maximum_top_issuer_share_per_band=D(1),
            maximum_top_year_share_per_band=D(1),
            maximum_top_evidence_source_share_per_band=D(1),
        ).model_dump_json(),
        encoding="utf-8",
    )
    feature_build = tmp_path / "freeze-feature-build"
    feature_build.mkdir()
    (feature_build / "sparse-features-all-pairs.jsonl").write_text(
        "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
    )
    (feature_build / "stage-status.json").write_text(
        json.dumps(
            {
                "status": "SPARSE_FEATURES_BUILT_AWAITING_OUTCOME_BLIND_CALIBRATION",
                "outcome_vault_opened": False,
                "return_data_opened": False,
                "applicability_contract_complete": True,
                "deterministic_pit_priority_applied": True,
                "input_hashes": {
                    "parser_validation_manifest": sha256_file(parser_manifest),
                    "contract_freeze_manifest": sha256_file(contract_manifest),
                },
            }
        ),
        encoding="utf-8",
    )

    result = calibrate_sparse_features(
        feature_build=feature_build,
        output=tmp_path / "frozen",
        freeze=True,
        minimum_observed_axes=2,
        parser_validation_manifest=parser_manifest,
        abstention_audit_manifest=audit_manifest,
        contract_freeze_manifest=contract_manifest,
        coverage_gate_policy=coverage_policy,
    )

    assert result["status"] == "V2_FEATURE_ONLY_CALIBRATION_SEALED_OUTCOMES_CLOSED"
    assert result["outcome_stage_authorized"] is False
    assert result["all_five_bands_sufficient"] is True
    assert (tmp_path / "frozen" / "sparse-feature-seal.json").is_file()
    pre_outcome = json.loads(
        (tmp_path / "frozen" / "pre-outcome-manifest.json").read_text(encoding="utf-8")
    )
    assert pre_outcome["min_nobs"] == 2
    assert pre_outcome["coverage_gate"]["passed"] is True
    assert pre_outcome["last_grounded_days"] == 450
    assert pre_outcome["outcome_stage_authorized"] is False


def _locked_packet(index: int, axis: OperatingEvidenceAxis) -> PairedAxisPacket:
    return PairedAxisPacket(
        packet_id=f"PKT_{index:024x}",
        axis=axis,
        previous_excerpts=[BlindedExcerpt(source_id=f"SRC_{index * 2:020x}", text="이전")],
        current_excerpts=[
            BlindedExcerpt(source_id=f"SRC_{index * 2 + 1:020x}", text="현재")
        ],
    )


def test_v2_dual_locked_sets_are_independent_balanced_and_single_use(tmp_path: Path) -> None:
    balanced_packets: list[PairedAxisPacket] = []
    balanced_labels: list[AxisPairClassification] = []
    natural_packets: list[PairedAxisPacket] = []
    natural_labels: list[AxisPairClassification] = []
    gold_rows: list[dict[str, str]] = []

    def add_case(
        *,
        index: int,
        axis: OperatingEvidenceAxis,
        status: str,
        previous_state: int | None,
        current_state: int | None,
        split: str,
        contract: str,
        packets: list[PairedAxisPacket],
        labels: list[AxisPairClassification],
    ) -> None:
        packet = _locked_packet(index, axis)
        packets.append(packet)
        payload = {
            "packet_id": packet.packet_id,
            "axis": axis,
            "status": status,
            "confidence": 1,
        }
        row = {
            "packet_id": packet.packet_id,
            "axis": axis.value,
            "human_status": status,
            "human_previous_state": "",
            "human_current_state": "",
            "human_previous_source_id": "",
            "human_current_source_id": "",
            "human_previous_source_span": "",
            "human_current_source_span": "",
            "gold_split": split,
            "gold_contract_version": contract,
            "reviewer": "HUMAN",
        }
        if status == "COMPLETE":
            payload.update(
                previous_state=previous_state,
                current_state=current_state,
                previous_source_id=packet.previous_excerpts[0].source_id,
                current_source_id=packet.current_excerpts[0].source_id,
                previous_source_span=packet.previous_excerpts[0].text,
                current_source_span=packet.current_excerpts[0].text,
            )
            row.update(
                human_previous_state=str(previous_state),
                human_current_state=str(current_state),
                human_previous_source_id=packet.previous_excerpts[0].source_id,
                human_current_source_id=packet.current_excerpts[0].source_id,
                human_previous_source_span=packet.previous_excerpts[0].text,
                human_current_source_span=packet.current_excerpts[0].text,
            )
        labels.append(AxisPairClassification.model_validate(payload))
        gold_rows.append(row)

    index = 1
    semantic_axes = (
        OperatingEvidenceAxis.DEMAND,
        OperatingEvidenceAxis.PRICE_MIX,
    )
    for axis in semantic_axes:
        for status, previous_state, current_state in (
            ("COMPLETE", 1, 0),
            ("COMPLETE", 0, 0),
            ("COMPLETE", 0, 1),
            ("INSUFFICIENT_EVIDENCE", None, None),
            ("AMBIGUOUS", None, None),
        ):
            add_case(
                index=index,
                axis=axis,
                status=status,
                previous_state=previous_state,
                current_state=current_state,
                split="V2_BALANCED_LOCKED_TEST",
                contract="V2_DIRECTIONAL_BALANCED_LOCKED",
                packets=balanced_packets,
                labels=balanced_labels,
            )
            index += 1
    for offset, axis in enumerate(semantic_axes, start=1000):
        add_case(
            index=offset,
            axis=axis,
            status="COMPLETE",
            previous_state=0,
            current_state=1,
            split="V2_NATURAL_LOCKED_TEST",
            contract="V2_NATURAL_FREQUENCY_LOCKED",
            packets=natural_packets,
            labels=natural_labels,
        )

    balanced_input = tmp_path / "balanced.jsonl"
    natural_input = tmp_path / "natural.jsonl"
    for path, packets in (
        (balanced_input, balanced_packets),
        (natural_input, natural_packets),
    ):
        path.write_text(
            "".join(item.model_dump_json() + "\n" for item in packets), encoding="utf-8"
        )
    gold = tmp_path / "gold-v2.csv"
    with gold.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(gold_rows[0]))
        writer.writeheader()
        writer.writerows(gold_rows)

    def classification_build(
        path: Path,
        packet_path: Path,
        labels: list[AxisPairClassification],
    ) -> None:
        path.mkdir()
        (path / "classifications.jsonl").write_text(
            "".join(item.model_dump_json() + "\n" for item in labels), encoding="utf-8"
        )
        (path / "stage-status.json").write_text(
            json.dumps(
                {
                    "status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE",
                    "input_blinded_packet_sha256": sha256_file(packet_path),
                    "parser_version": "parser-v2-test",
                    "prompt_sha256": "a" * 64,
                    "requested_model": "fixture",
                }
            ),
            encoding="utf-8",
        )

    balanced_build = tmp_path / "balanced-classifications"
    natural_build = tmp_path / "natural-classifications"
    classification_build(balanced_build, balanced_input, balanced_labels)
    classification_build(natural_build, natural_input, natural_labels)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(
        json.dumps(
            {
                "schema_version": "moatrader-historical-evidence-parser-freeze-v2/2",
                "status": "V2_PARSER_FROZEN_AWAITING_DUAL_INDEPENDENT_LOCKED_TESTS",
                "parser_version": "parser-v2-test",
                "prompt_sha256": "a" * 64,
                "requested_model": "fixture",
                "natural_locked_packet_sha256": sha256_file(natural_input),
                "balanced_locked_packet_sha256": sha256_file(balanced_input),
                "human_gold_sha256": sha256_file(gold),
                "locked_sets_disjoint": True,
                "v1_locked_rows_reused": False,
            }
        ),
        encoding="utf-8",
    )
    balanced_consumption = tmp_path / "balanced-consumption.json"
    natural_consumption = tmp_path / "natural-consumption.json"
    balanced_output = tmp_path / "balanced-evaluation"
    natural_output = tmp_path / "natural-evaluation"
    balanced = evaluate_v2_locked_parser(
        packet_input=balanced_input,
        classification_build=balanced_build,
        human_gold=gold,
        parser_freeze_manifest=freeze,
        locked_consumption_record=balanced_consumption,
        output=balanced_output,
        locked_kind="BALANCED",
        minimum_per_axis_stratum=1,
        minimum_overall_directional_agreement=1,
        minimum_axis_directional_agreement=1,
        maximum_neutral_to_bullish_rate=0,
    )
    natural = evaluate_v2_locked_parser(
        packet_input=natural_input,
        classification_build=natural_build,
        human_gold=gold,
        parser_freeze_manifest=freeze,
        locked_consumption_record=natural_consumption,
        output=natural_output,
        locked_kind="NATURAL",
        minimum_natural_per_axis=1,
        minimum_overall_directional_agreement=1,
        minimum_axis_directional_agreement=1,
        maximum_neutral_to_bullish_rate=0,
    )
    combined_path = tmp_path / "combined.json"
    combined = combine_v2_locked_evaluations(
        natural_evaluation_manifest=natural_output / "stage-status.json",
        balanced_evaluation_manifest=balanced_output / "stage-status.json",
        parser_freeze_manifest=freeze,
        output=combined_path,
    )

    assert balanced["status"] == "V2_BALANCED_LOCKED_TEST_PASSED"
    assert natural["status"] == "V2_NATURAL_LOCKED_TEST_PASSED"
    assert combined["status"] == "V2_LOCKED_TESTS_PASSED"
    balanced_quality = json.loads(
        (balanced_output / "parser-quality-report-v2.json").read_text(encoding="utf-8")
    )
    assert balanced_quality["false_stable_count"] == 0
    assert balanced_quality["false_stable_rate"] == 0
    assert balanced_quality["opposite_direction_count"] == 0
    with pytest.raises(FileExistsError, match="already consumed"):
        evaluate_v2_locked_parser(
            packet_input=balanced_input,
            classification_build=balanced_build,
            human_gold=gold,
            parser_freeze_manifest=freeze,
            locked_consumption_record=balanced_consumption,
            output=tmp_path / "second-evaluation",
            locked_kind="BALANCED",
            minimum_per_axis_stratum=1,
        )


def test_prepare_new_independent_natural_and_balanced_locked_sets(tmp_path: Path) -> None:
    semantic_axes = (
        OperatingEvidenceAxis.DEMAND,
        OperatingEvidenceAxis.PRICE_MIX,
    )
    packets = [
        _locked_packet(axis_index * 100 + index, axis)
        for axis_index, axis in enumerate(semantic_axes, start=1)
        for index in range(1, 29)
    ]
    prior = [packets[index * 28] for index in range(2)]
    dev = [packets[index * 28 + 1] for index in range(2)]

    def write_packets(path: Path, rows: list[PairedAxisPacket]) -> None:
        path.write_text(
            "".join(row.model_dump_json() + "\n" for row in rows), encoding="utf-8"
        )

    packet_input = tmp_path / "semantic-packets.jsonl"
    prior_input = tmp_path / "v1-locked.jsonl"
    dev_input = tmp_path / "dev.jsonl"
    write_packets(packet_input, packets)
    write_packets(prior_input, prior)
    write_packets(dev_input, dev)
    candidates = tmp_path / "locked-candidates"
    prepared = prepare_locked_candidates(
        packet_input=packet_input,
        prior_v1_inputs=[prior_input],
        dev_inputs=[dev_input],
        output=candidates,
        natural_per_axis=1,
        balanced_candidates_per_axis=25,
    )
    assert prepared["v1_locked_rows_reused"] is False
    assert prepared["locked_sets_disjoint"] is True

    natural = [
        PairedAxisPacket.model_validate_json(line)
        for line in (candidates / "natural-locked-packets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    balanced_candidates = [
        PairedAxisPacket.model_validate_json(line)
        for line in (candidates / "balanced-candidate-packets.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    natural_ids = {packet.packet_id for packet in natural}
    by_axis_index: dict[OperatingEvidenceAxis, int] = {axis: 0 for axis in semantic_axes}
    decisions: list[dict[str, object]] = []
    for packet in (*natural, *balanced_candidates):
        if packet.packet_id in natural_ids:
            status, previous_state, current_state = "COMPLETE", 0, 0
        else:
            stratum_index = by_axis_index[packet.axis] % 5
            by_axis_index[packet.axis] += 1
            status, previous_state, current_state = (
                ("COMPLETE", 1, 0),
                ("COMPLETE", 0, 0),
                ("COMPLETE", 0, 1),
                ("INSUFFICIENT_EVIDENCE", None, None),
                ("AMBIGUOUS", None, None),
            )[stratum_index]
        decision: dict[str, object] = {
            "packet_id": packet.packet_id,
            "status": status,
            "review_notes": "fixture HUMAN review",
        }
        if status == "COMPLETE":
            decision.update(
                previous_state=previous_state,
                current_state=current_state,
                previous_anchor="이전",
                current_anchor="현재",
            )
        decisions.append(decision)
    review_decisions = tmp_path / "human-review-decisions.json"
    review_decisions.write_text(
        json.dumps(
            {
                "reviewer": "HUMAN",
                "outcome_vault_opened": False,
                "return_data_opened": False,
                "decisions": decisions,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    materialized = tmp_path / "human-gold-materialized"
    materialization = materialize_human_gold(
        candidate_build=candidates,
        review_decisions=review_decisions,
        output=materialized,
    )
    assert materialization["reviewer"] == "HUMAN"
    assert materialization["review_decision_count"] == 52
    adjudicated = materialized / "adjudicated-human-gold.csv"
    final = tmp_path / "locked-final"
    manifest = finalize_locked_sets(
        candidate_build=candidates,
        adjudicated_human_gold=adjudicated,
        output=final,
        minimum_per_axis_stratum=5,
    )
    assert manifest["status"] == "V2_DUAL_INDEPENDENT_LOCKED_SETS_PREPARED_OUTCOME_BLIND"
    assert manifest["natural_packet_count"] == 2
    assert manifest["balanced_packet_count"] == 50
    assert manifest["gold_label_authority"] == "HUMAN"
    assert all(
        count == 5
        for axis_counts in manifest["balanced_stratum_counts"].values()
        for count in axis_counts.values()
    )
    dev_evaluation = tmp_path / "dev-evaluation.json"
    dev_evaluation.write_text(
        json.dumps(
            {
                "status": "DEV_PASSED_PARSER_READY_TO_FREEZE",
                "parser_version": "parser-v2-test",
                "prompt_sha256": "a" * 64,
                "requested_model": "fixture",
                "outcome_vault_opened": False,
                "return_data_opened": False,
            }
        ),
        encoding="utf-8",
    )
    parser_freeze = create_v2_parser_freeze(
        dev_evaluation_manifest=dev_evaluation,
        locked_set_preparation_manifest=final / "locked-set-preparation-manifest.json",
        natural_locked_packet_input=final / "natural-locked-packets.jsonl",
        balanced_locked_packet_input=final / "balanced-locked-packets.jsonl",
        human_gold=final / "v2-locked-human-gold.csv",
        output=tmp_path / "parser-freeze.json",
    )
    assert parser_freeze["v1_locked_rows_reused"] is False
    assert parser_freeze["locked_sets_disjoint"] is True
    assert parser_freeze["locked_set_preparation_manifest_sha256"] == sha256_file(
        final / "locked-set-preparation-manifest.json"
    )


def test_human_review_rows_distinguish_unreviewed_candidates_from_abstentions() -> None:
    assert _review_row_is_blank({"packet_id": "candidate", "human_status": ""}) is True
    assert (
        _review_row_is_blank(
            {
                "packet_id": "candidate",
                "human_status": "INSUFFICIENT_EVIDENCE",
                "reviewer": "HUMAN",
            }
        )
        is False
    )

    with pytest.raises(
        ValueError,
        match="must leave every state/source field blank",
    ):
        _classification(
            {
                "packet_id": "candidate",
                "axis": "DEMAND",
                "human_status": "AMBIGUOUS",
                "human_previous_source_id": "SRC_must_not_survive_abstention",
            }
        )


def test_abstention_reason_audit_requires_200_grounded_human_reasons(tmp_path: Path) -> None:
    packets = [
        _locked_packet(index, list(OperatingEvidenceAxis)[index % 6])
        for index in range(1, 201)
    ]
    classifications = [
        AxisPairClassification(
            packet_id=packet.packet_id,
            axis=packet.axis,
            status=(
                AxisClassificationStatus.INSUFFICIENT_EVIDENCE
                if index % 2
                else AxisClassificationStatus.AMBIGUOUS
            ),
            confidence=1,
        )
        for index, packet in enumerate(packets)
    ]
    packet_input = tmp_path / "abstentions.jsonl"
    packet_input.write_text(
        "".join(item.model_dump_json() + "\n" for item in packets), encoding="utf-8"
    )
    build = tmp_path / "classification"
    build.mkdir()
    classification_path = build / "classifications.jsonl"
    classification_path.write_text(
        "".join(item.model_dump_json() + "\n" for item in classifications), encoding="utf-8"
    )
    (build / "stage-status.json").write_text(
        json.dumps(
            {
                "status": "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE",
                "input_blinded_packet_sha256": sha256_file(packet_input),
            }
        ),
        encoding="utf-8",
    )
    prepared = tmp_path / "prepared"
    prepare_status = prepare_abstention_audit(
        packet_input=packet_input,
        classification_build=build,
        output=prepared,
        sample_size=200,
    )
    completed = tmp_path / "completed.csv"
    with (prepared / "abstention-audit-template.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as source, completed.open("w", encoding="utf-8-sig", newline="") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row["abstention_reason"] = "AMBIGUOUS_HUMAN_TOO"
            row["reviewer"] = "HUMAN_REVIEWER"
            writer.writerow(row)
    validated = validate_abstention_audit(
        prepared_build=prepared,
        completed_audit=completed,
        output=tmp_path / "validated",
    )

    assert prepare_status["sample_size"] == 200
    assert validated["status"] == "V2_ABSTENTION_AUDIT_PASSED"
    assert validated["reviewed_count"] == 200
