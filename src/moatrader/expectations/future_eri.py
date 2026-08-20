from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, time
from decimal import Decimal
from enum import IntEnum, StrEnum
from math import sqrt
from statistics import median
from typing import Literal, Sequence

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel, StatementType
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from moatrader.valuation.economic_dcf import EconomicDcfEngine


D = Decimal
FCFF_EVIDENCE_AXES = (
    "DEMAND",
    "PRICE_MIX",
    "BACKLOG",
    "MARGIN",
    "INVENTORY_MISMATCH",
    "CAPACITY_CAPEX",
)
FORBIDDEN_FEATURE_FIELD_FRAGMENTS = (
    "future_price",
    "future_return",
    "forward_return",
    "target_price",
    "future_eri",
    "realized_future",
)


def _aware(value: datetime, *, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _sha256_payload(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def model_sha256(model: ContractModel) -> str:
    return _sha256_payload(model.model_dump(mode="json"))


class OperatingEvidenceAxis(StrEnum):
    DEMAND = "DEMAND"
    PRICE_MIX = "PRICE_MIX"
    BACKLOG = "BACKLOG"
    MARGIN = "MARGIN"
    INVENTORY_MISMATCH = "INVENTORY_MISMATCH"
    CAPACITY_CAPEX = "CAPACITY_CAPEX"


class EvidenceState(IntEnum):
    WEAKENING = -1
    STABLE = 0
    IMPROVING = 1


class EvidenceVectorStatus(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


class MaterialityBasis(StrEnum):
    CONTRACT_VALUE_TTM_REVENUE = "CONTRACT_VALUE_TTM_REVENUE"
    ABSOLUTE_CHANGE_TTM_REVENUE = "ABSOLUTE_CHANGE_TTM_REVENUE"
    CAPEX_TTM_REVENUE = "CAPEX_TTM_REVENUE"
    INVENTORY_TTM_REVENUE = "INVENTORY_TTM_REVENUE"
    OTHER_SCALED_FACT = "OTHER_SCALED_FACT"


class MaterialityComputationV1(ContractModel):
    basis: MaterialityBasis
    numerator: Decimal = Field(ge=0)
    denominator: Decimal = Field(gt=0)
    raw_ratio: Decimal = Field(ge=0)
    capped_materiality: Decimal = Field(ge=0, le=1)
    numerator_source_id: str = Field(min_length=1)
    denominator_source_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def deterministic_ratio(self) -> "MaterialityComputationV1":
        expected_ratio = self.numerator / self.denominator
        if self.raw_ratio != expected_ratio:
            raise ValueError("raw materiality ratio must equal numerator / denominator")
        if self.capped_materiality != min(expected_ratio, D(1)):
            raise ValueError("capped materiality must equal min(raw ratio, 1)")
        return self


def scaled_materiality(
    *,
    basis: MaterialityBasis,
    numerator: Decimal,
    denominator: Decimal,
    numerator_source_id: str,
    denominator_source_id: str,
) -> MaterialityComputationV1:
    if denominator <= 0:
        raise ValueError("materiality denominator must be positive")
    if numerator < 0:
        raise ValueError("materiality numerator must be nonnegative")
    ratio = numerator / denominator
    return MaterialityComputationV1(
        basis=basis,
        numerator=numerator,
        denominator=denominator,
        raw_ratio=ratio,
        capped_materiality=min(ratio, D(1)),
        numerator_source_id=numerator_source_id,
        denominator_source_id=denominator_source_id,
    )


class EvidenceObservation(ContractModel):
    """One auditable fact classification; it is not an outlook prediction."""

    observation_id: str = Field(min_length=1)
    issuer_id: str = Field(min_length=1)
    fiscal_period: str = Field(min_length=1)
    axis: OperatingEvidenceAxis
    state: EvidenceState
    source_document_id: str = Field(min_length=1)
    source_span: str = Field(min_length=1)
    source_published_at: datetime
    available_at: datetime
    signal_timestamp: datetime
    statement_type: StatementType
    classification_rule_id: str = Field(min_length=1)
    materiality_rule_id: str = Field(min_length=1)
    confidence: Decimal = Field(ge=0, le=1)
    materiality: Decimal = Field(ge=0, le=1)
    raw_materiality_ratio: Decimal | None = Field(default=None, ge=0)
    materiality_computation: MaterialityComputationV1 | None = None

    @model_validator(mode="after")
    def auditable_and_point_in_time(self) -> "EvidenceObservation":
        for field, value in (
            ("source_published_at", self.source_published_at),
            ("available_at", self.available_at),
            ("signal_timestamp", self.signal_timestamp),
        ):
            _aware(value, field=field)
        if self.source_published_at > self.available_at:
            raise ValueError("source_published_at cannot be later than available_at")
        if self.available_at > self.signal_timestamp:
            raise ValueError("evidence was not available at signal_timestamp")
        allowed = {
            StatementType.DISCLOSED_FACT,
            StatementType.MANAGEMENT_CLAIM,
            StatementType.DERIVED_METRIC,
        }
        if self.statement_type not in allowed:
            raise ValueError("V1 evidence must be a fact, claim, or derived metric")
        if self.raw_materiality_ratio is not None:
            expected = min(self.raw_materiality_ratio, D(1))
            if self.materiality != expected:
                raise ValueError("materiality must equal min(raw_materiality_ratio, 1)")
        if self.materiality_computation is not None:
            if self.raw_materiality_ratio != self.materiality_computation.raw_ratio:
                raise ValueError("observation raw ratio does not match materiality computation")
            if self.materiality != self.materiality_computation.capped_materiality:
                raise ValueError("observation materiality does not match materiality computation")
        return self


class ComparableEvidenceDelta(ContractModel):
    axis: OperatingEvidenceAxis
    current_state: EvidenceState
    prior_state: EvidenceState
    raw_state_change: int = Field(ge=-2, le=2)
    direction: int = Field(ge=-1, le=1)
    materiality: Decimal = Field(ge=0, le=1)
    materiality_weighted_direction: Decimal = Field(ge=-1, le=1)
    current_observation_ids: list[str] = Field(min_length=1)
    prior_observation_ids: list[str] = Field(min_length=1)
    current_source_document_ids: list[str] = Field(min_length=1)
    prior_source_document_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def arithmetic_matches(self) -> "ComparableEvidenceDelta":
        raw = self.current_state.value - self.prior_state.value
        direction = (raw > 0) - (raw < 0)
        if self.raw_state_change != raw or self.direction != direction:
            raise ValueError("evidence delta does not match current minus prior state")
        if self.materiality_weighted_direction != D(direction) * self.materiality:
            raise ValueError("materiality-weighted direction does not match delta")
        return self


def _aggregate_observations(
    observations: Sequence[EvidenceObservation],
) -> tuple[EvidenceState, Decimal]:
    if not observations:
        raise ValueError("at least one evidence observation is required")
    axes = {item.axis for item in observations}
    issuers = {item.issuer_id for item in observations}
    periods = {item.fiscal_period for item in observations}
    if len(axes) != 1 or len(issuers) != 1 or len(periods) != 1:
        raise ValueError("observations must share one issuer, period, and axis")
    weighted = sum(
        (D(item.state.value) * item.confidence * item.materiality for item in observations),
        D(0),
    )
    state = EvidenceState((weighted > 0) - (weighted < 0))
    materiality = max((item.materiality for item in observations), default=D(0))
    return state, materiality


def comparable_evidence_delta(
    *,
    current: Sequence[EvidenceObservation],
    prior: Sequence[EvidenceObservation],
) -> ComparableEvidenceDelta:
    current_state, current_materiality = _aggregate_observations(current)
    prior_state, prior_materiality = _aggregate_observations(prior)
    if current[0].issuer_id != prior[0].issuer_id or current[0].axis != prior[0].axis:
        raise ValueError("current and prior evidence must share issuer and axis")
    signal_timestamp = current[0].signal_timestamp
    if any(item.signal_timestamp != signal_timestamp for item in current):
        raise ValueError("current observations must share signal_timestamp")
    if any(item.available_at > signal_timestamp for item in prior):
        raise ValueError("prior evidence was not available at current signal")
    if current[0].fiscal_period == prior[0].fiscal_period:
        raise ValueError("current and prior evidence must be comparable different periods")
    raw = current_state.value - prior_state.value
    direction = (raw > 0) - (raw < 0)
    materiality = max(current_materiality, prior_materiality)
    return ComparableEvidenceDelta(
        axis=current[0].axis,
        current_state=current_state,
        prior_state=prior_state,
        raw_state_change=raw,
        direction=direction,
        materiality=materiality,
        materiality_weighted_direction=D(direction) * materiality,
        current_observation_ids=sorted(item.observation_id for item in current),
        prior_observation_ids=sorted(item.observation_id for item in prior),
        current_source_document_ids=sorted({item.source_document_id for item in current}),
        prior_source_document_ids=sorted({item.source_document_id for item in prior}),
    )


class FcffEvidenceVectorV1(ContractModel):
    schema_version: str = "moatrader-fcff-evidence-vector-v1/1"
    route: Literal["FCFF"] = "FCFF"
    issuer_id: str = Field(min_length=1)
    signal_timestamp: datetime
    status: EvidenceVectorStatus
    axis_deltas: dict[OperatingEvidenceAxis, ComparableEvidenceDelta]
    missing_axes: list[OperatingEvidenceAxis] = Field(default_factory=list)
    evidence_f_score: int | None = Field(default=None, ge=-6, le=6)
    materiality_weighted_score: Decimal | None = Field(default=None, ge=-6, le=6)
    future_label_accessed: Literal[False] = False
    return_data_accessed: Literal[False] = False
    primary_ranking_policy: Literal["NONE_MECHANISM_ONLY"] = "NONE_MECHANISM_ONLY"

    @model_validator(mode="after")
    def complete_vector_arithmetic(self) -> "FcffEvidenceVectorV1":
        _aware(self.signal_timestamp, field="signal_timestamp")
        expected = set(OperatingEvidenceAxis)
        actual = set(self.axis_deltas)
        missing = sorted(expected - actual, key=lambda item: item.value)
        if self.missing_axes != missing:
            raise ValueError("missing_axes does not match axis_deltas")
        complete = not missing
        if complete != (self.status == EvidenceVectorStatus.COMPLETE):
            raise ValueError("vector status does not match six-axis completeness")
        score = sum(item.direction for item in self.axis_deltas.values())
        weighted = sum(
            (item.materiality_weighted_direction for item in self.axis_deltas.values()),
            D(0),
        )
        if complete:
            if self.evidence_f_score != score or self.materiality_weighted_score != weighted:
                raise ValueError("complete vector scores do not match axis deltas")
        elif self.evidence_f_score is not None or self.materiality_weighted_score is not None:
            raise ValueError("an incomplete vector must not silently impute missing axes")
        return self


def build_fcff_evidence_vector(
    *,
    issuer_id: str,
    signal_timestamp: datetime,
    current: Sequence[EvidenceObservation],
    prior: Sequence[EvidenceObservation],
) -> FcffEvidenceVectorV1:
    _aware(signal_timestamp, field="signal_timestamp")
    current_by_axis: dict[OperatingEvidenceAxis, list[EvidenceObservation]] = {}
    prior_by_axis: dict[OperatingEvidenceAxis, list[EvidenceObservation]] = {}
    for destination, observations in ((current_by_axis, current), (prior_by_axis, prior)):
        for item in observations:
            if item.issuer_id != issuer_id:
                raise ValueError("all observations must match vector issuer_id")
            if item.available_at > signal_timestamp:
                raise ValueError("vector contains evidence unavailable at signal")
            if destination is current_by_axis and item.signal_timestamp != signal_timestamp:
                raise ValueError("current evidence signal_timestamp must match vector signal")
            destination.setdefault(item.axis, []).append(item)
    deltas = {
        axis: comparable_evidence_delta(
            current=current_by_axis[axis],
            prior=prior_by_axis[axis],
        )
        for axis in OperatingEvidenceAxis
        if axis in current_by_axis and axis in prior_by_axis
    }
    missing = sorted(set(OperatingEvidenceAxis) - set(deltas), key=lambda item: item.value)
    complete = not missing
    return FcffEvidenceVectorV1(
        issuer_id=issuer_id,
        signal_timestamp=signal_timestamp,
        status=(EvidenceVectorStatus.COMPLETE if complete else EvidenceVectorStatus.INCOMPLETE),
        axis_deltas=deltas,
        missing_axes=missing,
        evidence_f_score=(sum(item.direction for item in deltas.values()) if complete else None),
        materiality_weighted_score=(
            sum((item.materiality_weighted_direction for item in deltas.values()), D(0))
            if complete
            else None
        ),
    )


class CurrentExpectationStateV1(ContractModel):
    route: Literal["FCFF"] = "FCFF"
    issuer_id: str = Field(min_length=1)
    signal_timestamp: datetime
    market_price: Decimal = Field(gt=0)
    market_price_at: datetime
    market_price_source_id: str = Field(min_length=1)
    implied_growth: Decimal = Field(gt=-1, le=3)
    implied_margin: Decimal = Field(gt=-1, lt=1)
    implied_roiic: Decimal = Field(gt=0, le=5)
    implied_cap_years: Decimal = Field(ge=0, le=50)
    reverse_dcf_method: str = Field(min_length=1)
    reverse_dcf_input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def signal_is_aware(self) -> "CurrentExpectationStateV1":
        _aware(self.signal_timestamp, field="signal_timestamp")
        _aware(self.market_price_at, field="market_price_at")
        if self.market_price_at != self.signal_timestamp:
            raise ValueError("Reverse DCF market price must be timestamped at the signal")
        return self


class FutureEriFeatureRowV1(ContractModel):
    schema_version: str = "moatrader-future-eri-feature-v1/1"
    observation_id: str = Field(min_length=1)
    evidence: FcffEvidenceVectorV1
    expectation_state: CurrentExpectationStateV1
    frozen_expectation_assumptions: EconomicDcfAssumptions

    @model_validator(mode="after")
    def one_point_in_time_state(self) -> "FutureEriFeatureRowV1":
        if self.evidence.status != EvidenceVectorStatus.COMPLETE:
            raise ValueError("only complete six-axis evidence vectors can enter the V1 dataset")
        if self.evidence.issuer_id != self.expectation_state.issuer_id:
            raise ValueError("evidence and expectation state issuer mismatch")
        if self.evidence.signal_timestamp != self.expectation_state.signal_timestamp:
            raise ValueError("evidence and expectation state signal mismatch")
        expected = self.expectation_state
        frozen = self.frozen_expectation_assumptions
        if (
            expected.implied_growth != frozen.revenue_growth
            or expected.implied_margin != frozen.target_nopat_margin
            or expected.implied_roiic != frozen.roiic
            or expected.implied_cap_years != D(frozen.competitive_advantage_period_years)
        ):
            raise ValueError("frozen DCF path must match the current implied expectation state")
        return self


class FeatureDatasetSealV1(ContractModel):
    schema_version: str = "moatrader-future-eri-feature-seal-v1/1"
    sealed_at: datetime
    feature_count: int = Field(gt=0)
    observation_ids: list[str] = Field(min_length=1)
    feature_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_row_sha256: dict[str, str]
    outcome_source_opened_before_seal: Literal[False] = False
    forbidden_outcome_fields_present: Literal[False] = False
    return_data_accessed: Literal[False] = False

    @model_validator(mode="after")
    def valid_seal(self) -> "FeatureDatasetSealV1":
        _aware(self.sealed_at, field="sealed_at")
        if self.feature_count != len(self.observation_ids):
            raise ValueError("feature_count does not match observation_ids")
        if self.observation_ids != sorted(set(self.observation_ids)):
            raise ValueError("observation_ids must be unique and sorted")
        if set(self.feature_row_sha256) != set(self.observation_ids):
            raise ValueError("feature_row_sha256 keys must match observation_ids")
        if any(
            len(value) != 64 or any(char not in "0123456789abcdef" for char in value)
            for value in self.feature_row_sha256.values()
        ):
            raise ValueError("feature row hashes must be lowercase SHA-256 values")
        return self


def _assert_no_outcome_fields(value: object, *, path: str = "feature") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            if any(fragment in normalized for fragment in FORBIDDEN_FEATURE_FIELD_FRAGMENTS):
                raise ValueError(f"forbidden outcome field in feature payload: {path}.{key}")
            _assert_no_outcome_fields(child, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_no_outcome_fields(child, path=f"{path}[{index}]")


def seal_feature_dataset(
    rows: Sequence[FutureEriFeatureRowV1],
    *,
    sealed_at: datetime,
) -> FeatureDatasetSealV1:
    _aware(sealed_at, field="sealed_at")
    if not rows:
        raise ValueError("cannot seal an empty feature dataset")
    ids = sorted(item.observation_id for item in rows)
    if len(ids) != len(set(ids)):
        raise ValueError("feature observation_id values must be unique")
    payload = [item.model_dump(mode="json") for item in sorted(rows, key=lambda row: row.observation_id)]
    _assert_no_outcome_fields(payload)
    if any(item.evidence.signal_timestamp > sealed_at for item in rows):
        raise ValueError("sealed_at cannot precede a signal timestamp")
    return FeatureDatasetSealV1(
        sealed_at=sealed_at,
        feature_count=len(rows),
        observation_ids=ids,
        feature_dataset_sha256=_sha256_payload(payload),
        feature_row_sha256={
            item.observation_id: _sha256_payload(item.model_dump(mode="json"))
            for item in rows
        },
    )


def next_usable_signal_timestamp(
    available_at: datetime,
    *,
    trading_sessions: Sequence[date],
    market_open: time = time(9, 0),
    market_close: time = time(15, 30),
) -> datetime:
    """Respect after-close availability without inventing exchange holidays."""

    _aware(available_at, field="available_at")
    sessions = sorted(set(trading_sessions))
    if not sessions:
        raise ValueError("trading_sessions cannot be empty")
    zone = available_at.tzinfo
    assert zone is not None
    if available_at.date() in sessions:
        opened = datetime.combine(available_at.date(), market_open, tzinfo=zone)
        closed = datetime.combine(available_at.date(), market_close, tzinfo=zone)
        if available_at <= opened:
            return opened
        if available_at <= closed:
            return available_at
    for session in sessions:
        if session > available_at.date():
            return datetime.combine(session, market_open, tzinfo=zone)
    raise ValueError("trading calendar does not extend beyond available_at")


def target_trading_session(
    signal_session: date,
    trading_sessions: Sequence[date],
    *,
    horizon: int = 63,
) -> date:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    sessions = sorted(set(trading_sessions))
    try:
        index = sessions.index(signal_session)
    except ValueError:
        raise ValueError("signal_session is absent from trading calendar") from None
    target = index + horizon
    if target >= len(sessions):
        raise ValueError("trading calendar does not cover the requested horizon")
    return sessions[target]


class RealizedFcffStateV1(ContractModel):
    available_at: datetime
    base_period: str = Field(min_length=1)
    base_revenue: Decimal = Field(gt=0)
    base_nopat_margin: Decimal = Field(gt=-1, lt=1)
    base_invested_capital: Decimal = Field(gt=0)
    net_debt: Decimal
    diluted_shares: Decimal = Field(gt=0)
    wacc: Decimal = Field(gt=0, lt=1)
    wacc_source_id: str = Field(min_length=1)
    source_document_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def availability_is_aware(self) -> "RealizedFcffStateV1":
        _aware(self.available_at, field="available_at")
        return self


class FutureEriOutcomeInputV1(ContractModel):
    observation_id: str = Field(min_length=1)
    target_session: date
    target_price_at: datetime
    actual_market_price: Decimal = Field(gt=0)
    target_price_source_id: str = Field(min_length=1)
    realized_state: RealizedFcffStateV1
    price_basis: Literal["RAW_CLOSE_WITH_REALIZED_EV_EQUITY_BRIDGE"] = (
        "RAW_CLOSE_WITH_REALIZED_EV_EQUITY_BRIDGE"
    )

    @model_validator(mode="after")
    def outcome_is_point_in_time(self) -> "FutureEriOutcomeInputV1":
        _aware(self.target_price_at, field="target_price_at")
        if self.target_price_at.date() != self.target_session:
            raise ValueError("target_price_at must fall on target_session")
        if self.realized_state.available_at > self.target_price_at:
            raise ValueError("counterfactual cannot use fundamentals published after target price")
        return self


class EnterpriseEquityBridgeV1(ContractModel):
    counterfactual_enterprise_value: Decimal = Field(gt=0)
    counterfactual_net_debt: Decimal
    counterfactual_equity_value: Decimal = Field(gt=0)
    actual_equity_value: Decimal = Field(gt=0)
    actual_enterprise_value: Decimal = Field(gt=0)
    diluted_shares: Decimal = Field(gt=0)
    counterfactual_value_per_share: Decimal = Field(gt=0)
    actual_market_price: Decimal = Field(gt=0)
    enterprise_future_eri: Decimal
    equity_future_eri: Decimal
    capital_structure_bridge_effect: Decimal
    capital_structure_source_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def exact_ev_equity_bridge(self) -> "EnterpriseEquityBridgeV1":
        tolerance = D("1e-24")
        if self.counterfactual_equity_value != (
            self.counterfactual_enterprise_value - self.counterfactual_net_debt
        ):
            raise ValueError("counterfactual EV-to-equity bridge is inconsistent")
        if self.actual_equity_value != self.actual_market_price * self.diluted_shares:
            raise ValueError("actual equity value must equal price times realized diluted shares")
        if self.actual_enterprise_value != self.actual_equity_value + self.counterfactual_net_debt:
            raise ValueError("actual EV must use the same realized net-debt bridge")
        if self.counterfactual_value_per_share != (
            self.counterfactual_equity_value / self.diluted_shares
        ):
            raise ValueError("counterfactual per-share bridge is inconsistent")
        expected_enterprise = (
            self.actual_enterprise_value / self.counterfactual_enterprise_value
        ).ln()
        expected_equity = (
            self.actual_equity_value / self.counterfactual_equity_value
        ).ln()
        if abs(self.enterprise_future_eri - expected_enterprise) > tolerance:
            raise ValueError("enterprise ERI does not match bridged EV ratio")
        if abs(self.equity_future_eri - expected_equity) > tolerance:
            raise ValueError("equity ERI does not match bridged equity ratio")
        if abs(
            self.capital_structure_bridge_effect
            - (self.equity_future_eri - self.enterprise_future_eri)
        ) > tolerance:
            raise ValueError("capital-structure bridge effect is inconsistent")
        return self


class FutureEriLabelV1(ContractModel):
    schema_version: str = "moatrader-future-eri-label-v1/1"
    observation_id: str = Field(min_length=1)
    horizon_trading_days: Literal[63] = 63
    target_session: date
    target_price_at: datetime
    actual_market_price: Decimal = Field(gt=0)
    counterfactual_value_per_share: Decimal = Field(gt=0)
    future_eri: Decimal
    enterprise_future_eri: Decimal
    capital_structure_bridge_effect: Decimal
    enterprise_equity_bridge: EnterpriseEquityBridgeV1
    feature_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    frozen_assumptions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    counterfactual_assumptions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    realized_fundamentals_available_at: datetime
    return_data_accessed: Literal[False] = False

    @model_validator(mode="after")
    def eri_matches_value_ratio(self) -> "FutureEriLabelV1":
        _aware(self.target_price_at, field="target_price_at")
        _aware(
            self.realized_fundamentals_available_at,
            field="realized_fundamentals_available_at",
        )
        expected = (self.actual_market_price / self.counterfactual_value_per_share).ln()
        if abs(self.future_eri - expected) > D("1e-24"):
            raise ValueError("future_eri must equal log(actual/counterfactual)")
        if self.future_eri != self.enterprise_equity_bridge.equity_future_eri:
            raise ValueError("primary V1 ERI must remain the frozen equity-bridge ERI")
        if self.enterprise_future_eri != self.enterprise_equity_bridge.enterprise_future_eri:
            raise ValueError("enterprise ERI must match the EV bridge")
        if (
            self.capital_structure_bridge_effect
            != self.enterprise_equity_bridge.capital_structure_bridge_effect
        ):
            raise ValueError("label bridge effect must match EV bridge")
        return self


def roll_forward_frozen_expectations(
    feature: FutureEriFeatureRowV1,
    outcome: FutureEriOutcomeInputV1,
) -> EconomicDcfAssumptions:
    """Re-anchor on realized facts while freezing forward expectation drivers.

    EconomicDcfAssumptions uses whole-year CAP. V1 therefore consumes a CAP
    year only after a full signal-date anniversary; the 63-session horizon
    normally consumes none. This rule is explicit and deterministic.
    """

    signal_date = feature.evidence.signal_timestamp.date()
    target = outcome.target_session
    completed_years = target.year - signal_date.year
    if (target.month, target.day) < (signal_date.month, signal_date.day):
        completed_years -= 1
    completed_years = max(completed_years, 0)
    frozen = feature.frozen_expectation_assumptions
    remaining_cap = max(frozen.competitive_advantage_period_years - completed_years, 0)
    remaining_forecast = max(
        frozen.explicit_forecast_years - completed_years,
        remaining_cap + frozen.fade_years,
        1,
    )
    realized = outcome.realized_state
    payload = frozen.model_dump()
    payload.update(
        {
            "base_period": realized.base_period,
            "base_revenue": realized.base_revenue,
            "base_nopat_margin": realized.base_nopat_margin,
            "base_invested_capital": realized.base_invested_capital,
            "net_debt": realized.net_debt,
            "diluted_shares": realized.diluted_shares,
            "wacc": realized.wacc,
            "competitive_advantage_period_years": remaining_cap,
            "explicit_forecast_years": remaining_forecast,
        }
    )
    return EconomicDcfAssumptions.model_validate(payload)


def build_future_eri_label(
    *,
    feature: FutureEriFeatureRowV1,
    outcome: FutureEriOutcomeInputV1,
    feature_seal: FeatureDatasetSealV1,
    trading_sessions: Sequence[date],
    engine: EconomicDcfEngine | None = None,
) -> FutureEriLabelV1:
    if outcome.observation_id != feature.observation_id:
        raise ValueError("feature and outcome observation_id mismatch")
    if feature.observation_id not in feature_seal.observation_ids:
        raise ValueError("feature is not part of the sealed dataset")
    if feature_seal.feature_row_sha256[feature.observation_id] != model_sha256(feature):
        raise ValueError("feature row changed after the dataset was sealed")
    expected_target = target_trading_session(
        feature.evidence.signal_timestamp.date(),
        trading_sessions,
        horizon=63,
    )
    if outcome.target_session != expected_target:
        raise ValueError("outcome target is not exactly t+63 trading sessions")
    counterfactual = roll_forward_frozen_expectations(feature, outcome)
    valuation = (engine or EconomicDcfEngine()).value(counterfactual)
    value = valuation.fair_value_per_share
    if value <= 0:
        raise ValueError("counterfactual equity value per share must be positive")
    eri = (outcome.actual_market_price / value).ln()
    actual_equity = outcome.actual_market_price * outcome.realized_state.diluted_shares
    actual_enterprise = actual_equity + outcome.realized_state.net_debt
    if actual_enterprise <= 0 or valuation.enterprise_value <= 0:
        raise ValueError("actual and counterfactual enterprise values must be positive")
    enterprise_eri = (actual_enterprise / valuation.enterprise_value).ln()
    bridge = EnterpriseEquityBridgeV1(
        counterfactual_enterprise_value=valuation.enterprise_value,
        counterfactual_net_debt=outcome.realized_state.net_debt,
        counterfactual_equity_value=valuation.equity_value,
        actual_equity_value=actual_equity,
        actual_enterprise_value=actual_enterprise,
        diluted_shares=outcome.realized_state.diluted_shares,
        counterfactual_value_per_share=value,
        actual_market_price=outcome.actual_market_price,
        enterprise_future_eri=enterprise_eri,
        equity_future_eri=eri,
        capital_structure_bridge_effect=eri - enterprise_eri,
        capital_structure_source_ids=outcome.realized_state.source_document_ids,
    )
    return FutureEriLabelV1(
        observation_id=feature.observation_id,
        target_session=outcome.target_session,
        target_price_at=outcome.target_price_at,
        actual_market_price=outcome.actual_market_price,
        counterfactual_value_per_share=value,
        future_eri=eri,
        enterprise_future_eri=enterprise_eri,
        capital_structure_bridge_effect=eri - enterprise_eri,
        enterprise_equity_bridge=bridge,
        feature_dataset_sha256=feature_seal.feature_dataset_sha256,
        frozen_assumptions_sha256=model_sha256(feature.frozen_expectation_assumptions),
        counterfactual_assumptions_sha256=model_sha256(counterfactual),
        realized_fundamentals_available_at=outcome.realized_state.available_at,
    )


class EvidenceScoreBand(StrEnum):
    Q1 = "Q1_-6_TO_-3"
    Q2 = "Q2_-2_TO_-1"
    Q3 = "Q3_0"
    Q4 = "Q4_1_TO_2"
    Q5 = "Q5_3_TO_6"


def evidence_score_band(score: int) -> EvidenceScoreBand:
    if not -6 <= score <= 6:
        raise ValueError("evidence score must be between -6 and 6")
    if score <= -3:
        return EvidenceScoreBand.Q1
    if score <= -1:
        return EvidenceScoreBand.Q2
    if score == 0:
        return EvidenceScoreBand.Q3
    if score <= 2:
        return EvidenceScoreBand.Q4
    return EvidenceScoreBand.Q5


class EriMechanismObservationV1(ContractModel):
    observation_id: str = Field(min_length=1)
    signal_timestamp: datetime
    evidence_f_score: int = Field(ge=-6, le=6)
    future_eri: Decimal

    @model_validator(mode="after")
    def signal_is_aware(self) -> "EriMechanismObservationV1":
        _aware(self.signal_timestamp, field="signal_timestamp")
        return self


class EriMonotonicityPolicyV1(ContractModel):
    minimum_observations_per_band: int = Field(default=20, ge=1)
    minimum_spearman: Decimal = Field(default=D(0), ge=-1, le=1)
    require_all_adjacent_means_nondecreasing: Literal[True] = True
    require_positive_q5_minus_q1: Literal[True] = True


class EriBandSummaryV1(ContractModel):
    band: EvidenceScoreBand
    count: int = Field(ge=0)
    mean_future_eri: Decimal | None = None
    median_future_eri: Decimal | None = None


class EriMonotonicityReportV1(ContractModel):
    schema_version: str = "moatrader-future-eri-monotonicity-v1/1"
    observation_count: int = Field(gt=0)
    bands: list[EriBandSummaryV1] = Field(min_length=5, max_length=5)
    score_eri_spearman: Decimal | None = Field(default=None, ge=-1, le=1)
    adjacent_nondecreasing_count: int = Field(ge=0, le=4)
    q5_minus_q1_mean_eri: Decimal | None = None
    mechanism_gate_passed: bool
    ml_stage_authorized: bool
    return_stage_status: Literal[
        "NOT_RUN_V1_MECHANISM_ONLY",
        "BLOCKED_MECHANISM_GATE_FAILED",
    ]
    return_data_accessed: Literal[False] = False
    primary_ranking_policy: Literal["NONE_MECHANISM_ONLY"] = "NONE_MECHANISM_ONLY"

    @model_validator(mode="after")
    def downstream_gates_match(self) -> "EriMonotonicityReportV1":
        if self.ml_stage_authorized != self.mechanism_gate_passed:
            raise ValueError("ML stage can open only after the ERI mechanism gate passes")
        expected = (
            "NOT_RUN_V1_MECHANISM_ONLY"
            if self.mechanism_gate_passed
            else "BLOCKED_MECHANISM_GATE_FAILED"
        )
        if self.return_stage_status != expected:
            raise ValueError("return stage status does not match mechanism gate")
        return self


def _average_ranks(values: Sequence[Decimal]) -> list[Decimal]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [D(0)] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        average = D(index + 1 + end) / D(2)
        for position in range(index, end):
            ranks[ordered[position][0]] = average
        index = end
    return ranks


def _spearman(left: Sequence[Decimal], right: Sequence[Decimal]) -> Decimal | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = _average_ranks(left)
    y = _average_ranks(right)
    x_mean = sum(x, D(0)) / D(len(x))
    y_mean = sum(y, D(0)) / D(len(y))
    numerator = sum(((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True)), D(0))
    x_scale = sum(((value - x_mean) ** 2 for value in x), D(0))
    y_scale = sum(((value - y_mean) ** 2 for value in y), D(0))
    denominator = sqrt(float(x_scale * y_scale))
    return D(str(float(numerator) / denominator)) if denominator else None


def evaluate_future_eri_monotonicity(
    observations: Sequence[EriMechanismObservationV1],
    *,
    policy: EriMonotonicityPolicyV1 | None = None,
) -> EriMonotonicityReportV1:
    if not observations:
        raise ValueError("mechanism evaluation requires observations")
    ids = [item.observation_id for item in observations]
    if len(ids) != len(set(ids)):
        raise ValueError("mechanism observations require unique observation_id values")
    rule = policy or EriMonotonicityPolicyV1()
    ordered_bands = list(EvidenceScoreBand)
    grouped = {
        band: [
            item.future_eri
            for item in observations
            if evidence_score_band(item.evidence_f_score) == band
        ]
        for band in ordered_bands
    }
    summaries = [
        EriBandSummaryV1(
            band=band,
            count=len(values),
            mean_future_eri=(sum(values, D(0)) / D(len(values)) if values else None),
            median_future_eri=(D(str(median(values))) if values else None),
        )
        for band, values in grouped.items()
    ]
    means = [item.mean_future_eri for item in summaries]
    adjacent = sum(
        left is not None and right is not None and right >= left
        for left, right in zip(means, means[1:])
    )
    spread = (
        means[-1] - means[0]
        if means[0] is not None and means[-1] is not None
        else None
    )
    rho = _spearman(
        [D(item.evidence_f_score) for item in observations],
        [item.future_eri for item in observations],
    )
    enough = all(item.count >= rule.minimum_observations_per_band for item in summaries)
    passed = bool(
        enough
        and adjacent == 4
        and spread is not None
        and spread > 0
        and rho is not None
        and rho >= rule.minimum_spearman
    )
    return EriMonotonicityReportV1(
        observation_count=len(observations),
        bands=summaries,
        score_eri_spearman=rho,
        adjacent_nondecreasing_count=adjacent,
        q5_minus_q1_mean_eri=spread,
        mechanism_gate_passed=passed,
        ml_stage_authorized=passed,
        return_stage_status=(
            "NOT_RUN_V1_MECHANISM_ONLY"
            if passed
            else "BLOCKED_MECHANISM_GATE_FAILED"
        ),
    )
