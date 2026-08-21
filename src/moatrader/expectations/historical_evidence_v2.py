from __future__ import annotations

from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from math import sqrt
from typing import Literal, Sequence

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.expectations.future_eri import EvidenceState, OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    AxisPairClassification,
    HistoricalFilingPair,
    PairedAxisPacket,
    canonical_payload_sha256,
    historical_observation_id,
)


D = Decimal


class AxisApplicabilityV2(StrEnum):
    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SparseAxisAvailabilityV2(StrEnum):
    GROUNDED = "GROUNDED"
    NA = "NA"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AxisEvidenceProvenanceV2(StrEnum):
    LLM_NARRATIVE = "LLM_NARRATIVE"
    STRUCTURED_TABLE = "STRUCTURED_TABLE"
    DETERMINISTIC_NUMERIC = "DETERMINISTIC_NUMERIC"


class PreviousEvidenceBasisV2(StrEnum):
    IMMEDIATE_PREVIOUS_FILING = "IMMEDIATE_PREVIOUS_FILING"
    LAST_GROUNDED_WITHIN_STALENESS = "LAST_GROUNDED_WITHIN_STALENESS"


class AbstentionReasonV2(StrEnum):
    TRUE_NO_MENTION = "TRUE_NO_MENTION"
    ONE_PERIOD_ONLY = "ONE_PERIOD_ONLY"
    RETRIEVAL_MISS = "RETRIEVAL_MISS"
    TABLE_EXTRACTION_FAIL = "TABLE_EXTRACTION_FAIL"
    PERIOD_MISMATCH = "PERIOD_MISMATCH"
    AMBIGUOUS_HUMAN_TOO = "AMBIGUOUS_HUMAN_TOO"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    STALE_PRIOR_STATE = "STALE_PRIOR_STATE"
    CLASSIFICATION_NOT_RUN = "CLASSIFICATION_NOT_RUN"
    UNADJUDICATED_ABSTENTION = "UNADJUDICATED_ABSTENTION"


class SparseBreadthBandV2(StrEnum):
    STRONG_BEAR = "STRONG_BEAR"
    BEAR = "BEAR"
    NEUTRAL = "NEUTRAL"
    BULL = "BULL"
    STRONG_BULL = "STRONG_BULL"


class SparseAxisEvidenceV2(ContractModel):
    schema_version: str = "moatrader-sparse-axis-evidence-v2/1"
    axis: OperatingEvidenceAxis
    applicability: AxisApplicabilityV2
    availability: SparseAxisAvailabilityV2
    direction: EvidenceState | None = None
    provenance: AxisEvidenceProvenanceV2 | None = None
    previous_evidence_basis: PreviousEvidenceBasisV2 | None = None
    confidence: Decimal | None = Field(default=None, ge=0, le=1)
    source_ids: list[str] = Field(default_factory=list)
    source_spans: list[str] = Field(default_factory=list)
    previous_evidence_at: datetime | None = None
    current_evidence_at: datetime | None = None
    prior_age_days: int | None = Field(default=None, ge=0)
    staleness_limit_days: int | None = Field(default=None, gt=0)
    abstention_reason: AbstentionReasonV2 | None = None
    classification_packet_id: str | None = Field(
        default=None, pattern=r"^PKT_[0-9a-f]{24}$"
    )
    applicability_rule_id: str = Field(min_length=1)
    deterministic_metric_name: str | None = None
    deterministic_previous_value: Decimal | None = None
    deterministic_current_value: Decimal | None = None
    deterministic_delta: Decimal | None = None

    @model_validator(mode="after")
    def missing_never_becomes_neutral(self) -> "SparseAxisEvidenceV2":
        for name, value in (
            ("previous_evidence_at", self.previous_evidence_at),
            ("current_evidence_at", self.current_evidence_at),
        ):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must be timezone-aware")
        if self.applicability == AxisApplicabilityV2.NOT_APPLICABLE:
            if self.availability != SparseAxisAvailabilityV2.NOT_APPLICABLE:
                raise ValueError("not-applicable axis must use NOT_APPLICABLE availability")
            if any(
                value is not None
                for value in (
                    self.direction,
                    self.provenance,
                    self.previous_evidence_basis,
                    self.confidence,
                    self.abstention_reason,
                )
            ):
                raise ValueError("not-applicable axis cannot contain a score or abstention")
            if self.source_ids or self.source_spans:
                raise ValueError("not-applicable axis cannot contain evidence sources")
            return self
        if self.availability == SparseAxisAvailabilityV2.NOT_APPLICABLE:
            raise ValueError("applicable axis cannot use NOT_APPLICABLE availability")
        if self.availability == SparseAxisAvailabilityV2.NA:
            if any(
                value is not None
                for value in (
                    self.direction,
                    self.provenance,
                    self.previous_evidence_basis,
                    self.confidence,
                    self.previous_evidence_at,
                    self.current_evidence_at,
                )
            ):
                raise ValueError("NA axis cannot contain a direction, evidence basis, or confidence")
            if self.source_ids or self.source_spans:
                raise ValueError("NA axis cannot contain evidence sources")
            if self.abstention_reason is None:
                raise ValueError("NA axis requires an explicit abstention reason")
            return self
        if (
            self.direction is None
            or self.provenance is None
            or self.previous_evidence_basis is None
            or self.confidence is None
        ):
            raise ValueError(
                "grounded axis requires direction, provenance, previous basis, and confidence"
            )
        if self.abstention_reason is not None:
            raise ValueError("grounded axis cannot contain an abstention reason")
        if not self.source_ids:
            raise ValueError("grounded axis requires auditable source IDs")
        if self.previous_evidence_at is None or self.current_evidence_at is None:
            raise ValueError("grounded axis requires previous/current evidence timestamps")
        if self.previous_evidence_at > self.current_evidence_at:
            raise ValueError("axis evidence timestamps must be chronological")
        if self.previous_evidence_basis == PreviousEvidenceBasisV2.LAST_GROUNDED_WITHIN_STALENESS:
            if self.prior_age_days is None or self.staleness_limit_days is None:
                raise ValueError("last-grounded previous basis requires age and staleness limit")
            if self.prior_age_days > self.staleness_limit_days:
                raise ValueError("stale last-grounded previous basis cannot be scored")
        elif self.prior_age_days is not None or self.staleness_limit_days is not None:
            raise ValueError("staleness metadata is allowed only for a last-grounded previous basis")
        return self


class HistoricalSparseEvidenceFeatureRowV2(ContractModel):
    schema_version: str = "moatrader-historical-sparse-evidence-feature-v2/1"
    observation_id: str = Field(pattern=r"^OBS_[0-9a-f]{24}$")
    pair_id: str = Field(pattern=r"^PAIR_[0-9a-f]{24}$")
    issuer_id: str = Field(pattern=r"^[0-9]{6}$")
    signal_timestamp: datetime
    axis_evidence: dict[OperatingEvidenceAxis, SparseAxisEvidenceV2]
    applicable_axis_count: int = Field(ge=0, le=6)
    observed_axis_count: int = Field(ge=0, le=6)
    unavailable_axis_count: int = Field(ge=0, le=6)
    not_applicable_axis_count: int = Field(ge=0, le=6)
    positive_axis_count: int = Field(ge=0, le=6)
    neutral_axis_count: int = Field(ge=0, le=6)
    negative_axis_count: int = Field(ge=0, le=6)
    n_directional: int = Field(ge=0, le=6)
    directional_event_count: int = Field(ge=0, le=6)
    net_evidence: int = Field(ge=-6, le=6)
    signed_breadth: Decimal | None = Field(default=None, ge=-1, le=1)
    coverage: Decimal | None = Field(default=None, ge=0, le=1)
    feature_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    applicability_decided_outcome_blind: Literal[True] = True
    score_and_coverage_separate: Literal[True] = True
    outcome_data_accessed: Literal[False] = False
    return_data_accessed: Literal[False] = False
    primary_ranking_policy: Literal["NONE_MECHANISM_ONLY"] = "NONE_MECHANISM_ONLY"
    per_pbr_role: Literal["NOT_USED"] = "NOT_USED"

    @model_validator(mode="after")
    def sparse_arithmetic_and_hash(self) -> "HistoricalSparseEvidenceFeatureRowV2":
        if self.signal_timestamp.tzinfo is None or self.signal_timestamp.utcoffset() is None:
            raise ValueError("signal_timestamp must be timezone-aware")
        expected_axes = set(OperatingEvidenceAxis)
        if set(self.axis_evidence) != expected_axes:
            raise ValueError("V2 feature must retain all six axes as explicit states")
        if any(key != value.axis for key, value in self.axis_evidence.items()):
            raise ValueError("axis evidence key does not match embedded axis")
        applicable = sum(
            item.applicability == AxisApplicabilityV2.APPLICABLE
            for item in self.axis_evidence.values()
        )
        observed = sum(
            item.availability == SparseAxisAvailabilityV2.GROUNDED
            for item in self.axis_evidence.values()
        )
        unavailable = sum(
            item.availability == SparseAxisAvailabilityV2.NA
            for item in self.axis_evidence.values()
        )
        not_applicable = sum(
            item.availability == SparseAxisAvailabilityV2.NOT_APPLICABLE
            for item in self.axis_evidence.values()
        )
        directions = [
            item.direction.value
            for item in self.axis_evidence.values()
            if item.direction is not None
        ]
        positive = sum(value > 0 for value in directions)
        neutral = sum(value == 0 for value in directions)
        negative = sum(value < 0 for value in directions)
        net = sum(directions)
        expected_breadth = D(net) / D(observed) if observed else None
        expected_coverage = D(observed) / D(applicable) if applicable else None
        expected = (
            applicable,
            observed,
            unavailable,
            not_applicable,
            positive,
            neutral,
            negative,
            positive + negative,
            positive + negative,
            net,
            expected_breadth,
            expected_coverage,
        )
        actual = (
            self.applicable_axis_count,
            self.observed_axis_count,
            self.unavailable_axis_count,
            self.not_applicable_axis_count,
            self.positive_axis_count,
            self.neutral_axis_count,
            self.negative_axis_count,
            self.n_directional,
            self.directional_event_count,
            self.net_evidence,
            self.signed_breadth,
            self.coverage,
        )
        if actual != expected:
            raise ValueError("sparse feature arithmetic does not match axis evidence")
        payload = self.model_dump(mode="json")
        actual_hash = str(payload.pop("feature_hash"))
        if actual_hash != canonical_payload_sha256(payload):
            raise ValueError("V2 feature hash does not match row payload")
        return self


class SparseBreadthBandContractV2(ContractModel):
    schema_version: str = "moatrader-sparse-breadth-band-contract-v2/1"
    minimum_observed_axes: int = Field(ge=1, le=6)
    cut_points: tuple[Decimal, Decimal, Decimal, Decimal]
    calibration_method: Literal["FEATURE_ONLY_UNIQUE_VALUE_QUINTILES_V2"] = (
        "FEATURE_ONLY_UNIQUE_VALUE_QUINTILES_V2"
    )
    calibration_feature_count: int = Field(gt=0)
    calibration_feature_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    band_counts: dict[SparseBreadthBandV2, int]
    minimum_rows_per_band: int = Field(gt=0)
    all_bands_sufficient: bool
    outcome_data_accessed: Literal[False] = False
    return_data_accessed: Literal[False] = False

    @model_validator(mode="after")
    def feature_only_band_contract(self) -> "SparseBreadthBandContractV2":
        if not all(D(-1) < value < D(1) for value in self.cut_points):
            raise ValueError("V2 breadth cut points must lie strictly inside [-1, 1]")
        if tuple(sorted(self.cut_points)) != self.cut_points or len(set(self.cut_points)) != 4:
            raise ValueError("V2 breadth cut points must be strictly increasing")
        if set(self.band_counts) != set(SparseBreadthBandV2):
            raise ValueError("V2 band counts must cover exactly five bands")
        sufficient = all(value >= self.minimum_rows_per_band for value in self.band_counts.values())
        if self.all_bands_sufficient != sufficient:
            raise ValueError("V2 band sufficiency flag does not match counts")
        return self

    def band_for(self, value: Decimal) -> SparseBreadthBandV2:
        first, second, third, fourth = self.cut_points
        if value <= first:
            return SparseBreadthBandV2.STRONG_BEAR
        if value <= second:
            return SparseBreadthBandV2.BEAR
        if value <= third:
            return SparseBreadthBandV2.NEUTRAL
        if value <= fourth:
            return SparseBreadthBandV2.BULL
        return SparseBreadthBandV2.STRONG_BULL


class SparseCoverageGatePolicyV2(ContractModel):
    schema_version: str = "moatrader-sparse-coverage-gate-policy-v2/1"
    minimum_rows_per_band: int = Field(ge=1)
    minimum_unique_issuers_per_band: int = Field(ge=1)
    minimum_unique_signal_months_per_band: int = Field(ge=1)
    minimum_total_unique_issuers: int = Field(ge=1)
    minimum_total_unique_signal_months: int = Field(ge=1)
    maximum_top_issuer_share_per_band: Decimal = Field(gt=0, le=1)
    maximum_top_year_share_per_band: Decimal = Field(gt=0, le=1)
    maximum_top_evidence_source_share_per_band: Decimal = Field(gt=0, le=1)
    outcome_data_accessed: Literal[False] = False
    return_data_accessed: Literal[False] = False


class HistoricalSparseEvidenceDatasetSealV2(ContractModel):
    schema_version: str = "moatrader-historical-sparse-evidence-seal-v2/1"
    sealed_at: datetime
    feature_count: int = Field(gt=0)
    observation_ids: list[str] = Field(min_length=1)
    feature_dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_row_sha256: dict[str, str]
    band_contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_validation_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_freeze_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    abstention_audit_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    coverage_gate_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    minimum_observed_axes: int = Field(ge=1, le=6)
    natural_frequency_locked_gate_passed: Literal[True] = True
    balanced_directional_locked_gate_passed: Literal[True] = True
    measurement_coverage_gate_passed: Literal[True] = True
    applicability_decided_outcome_blind: Literal[True] = True
    signal_timestamp_policy: Literal[
        "FIRST_TRADABLE_TIMESTAMP_AFTER_CURRENT_REGULAR_FILING_AVAILABLE_AT"
    ] = "FIRST_TRADABLE_TIMESTAMP_AFTER_CURRENT_REGULAR_FILING_AVAILABLE_AT"
    last_grounded_days: Literal[450] = 450
    last_grounded_role: Literal[
        "PREVIOUS_COMPARISON_BASIS_ONLY_NEVER_CURRENT_EVIDENCE"
    ] = "PREVIOUS_COMPARISON_BASIS_ONLY_NEVER_CURRENT_EVIDENCE"
    outcome_source_opened_before_seal: Literal[False] = False
    return_data_accessed: Literal[False] = False
    primary_ranking_policy: Literal["NONE_MECHANISM_ONLY"] = "NONE_MECHANISM_ONLY"
    per_pbr_role: Literal["NOT_USED"] = "NOT_USED"

    @model_validator(mode="after")
    def valid_sparse_seal(self) -> "HistoricalSparseEvidenceDatasetSealV2":
        if self.sealed_at.tzinfo is None or self.sealed_at.utcoffset() is None:
            raise ValueError("sealed_at must be timezone-aware")
        if self.observation_ids != sorted(set(self.observation_ids)):
            raise ValueError("V2 seal observation IDs must be sorted and unique")
        if self.feature_count != len(self.observation_ids):
            raise ValueError("V2 seal feature count mismatch")
        if set(self.feature_row_sha256) != set(self.observation_ids):
            raise ValueError("V2 seal row-hash keys mismatch")
        return self


class PITApplicabilityRulesV2(ContractModel):
    schema_version: str = "moatrader-pit-applicability-rules-v2/1"
    margin_change_neutral_tolerance: Decimal = Field(default=D("0.005"), ge=0)
    inventory_mismatch_neutral_tolerance: Decimal = Field(default=D("0.05"), ge=0)
    backlog_growth_neutral_tolerance: Decimal = Field(default=D("0.05"), ge=0)
    capex_intensity_neutral_tolerance: Decimal = Field(default=D("0.01"), ge=0)
    inventory_assets_applicability_threshold: Decimal = Field(default=D("0.01"), ge=0)
    ppe_assets_applicability_threshold: Decimal = Field(default=D("0.05"), ge=0)
    last_grounded_staleness_days: int = Field(default=450, gt=0)
    outcome_data_accessed: Literal[False] = False
    return_data_accessed: Literal[False] = False


class PITOperatingSnapshotV2(ContractModel):
    schema_version: str = "moatrader-pit-operating-snapshot-v2/1"
    issuer_id: str = Field(pattern=r"^[0-9]{6}$")
    fiscal_period_end: date
    available_at: datetime
    source_ids: dict[str, list[str]] = Field(default_factory=dict)
    revenue: Decimal | None = None
    operating_profit: Decimal | None = None
    inventory: Decimal | None = None
    assets: Decimal | None = None
    backlog: Decimal | None = None
    capex: Decimal | None = None
    ppe: Decimal | None = None
    backlog_disclosed: bool = False
    capacity_disclosed: bool = False

    @model_validator(mode="after")
    def point_in_time_snapshot(self) -> "PITOperatingSnapshotV2":
        if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
            raise ValueError("PIT snapshot available_at must be timezone-aware")
        for name in ("revenue", "inventory", "assets", "backlog", "capex", "ppe"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        return self


class GroundedAxisStateSnapshotV2(ContractModel):
    axis: OperatingEvidenceAxis
    state: EvidenceState
    fiscal_period_end: date
    available_at: datetime
    source_ids: list[str] = Field(min_length=1)
    source_spans: list[str] = Field(default_factory=list)
    confidence: Decimal = Field(default=D(1), ge=0, le=1)
    provenance: AxisEvidenceProvenanceV2 = AxisEvidenceProvenanceV2.LLM_NARRATIVE

    @model_validator(mode="after")
    def aware_timestamp(self) -> "GroundedAxisStateSnapshotV2":
        if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
            raise ValueError("grounded state available_at must be timezone-aware")
        return self


def _direction(value: Decimal, tolerance: Decimal, *, invert: bool = False) -> EvidenceState:
    result = (value > tolerance) - (value < -tolerance)
    if invert:
        result = -result
    return EvidenceState(result)


def _source_ids(*snapshots: PITOperatingSnapshotV2, metrics: Sequence[str]) -> list[str]:
    return sorted(
        {
            source_id
            for snapshot in snapshots
            for metric in metrics
            for source_id in snapshot.source_ids.get(metric, [])
        }
    )


def _na_axis(
    axis: OperatingEvidenceAxis,
    reason: AbstentionReasonV2,
    *,
    rule_id: str,
) -> SparseAxisEvidenceV2:
    return SparseAxisEvidenceV2(
        axis=axis,
        applicability=AxisApplicabilityV2.APPLICABLE,
        availability=SparseAxisAvailabilityV2.NA,
        abstention_reason=reason,
        applicability_rule_id=rule_id,
    )


def _not_applicable_axis(axis: OperatingEvidenceAxis, *, rule_id: str) -> SparseAxisEvidenceV2:
    return SparseAxisEvidenceV2(
        axis=axis,
        applicability=AxisApplicabilityV2.NOT_APPLICABLE,
        availability=SparseAxisAvailabilityV2.NOT_APPLICABLE,
        applicability_rule_id=rule_id,
    )


def _deterministic_axis(
    *,
    axis: OperatingEvidenceAxis,
    direction: EvidenceState,
    previous: PITOperatingSnapshotV2,
    current: PITOperatingSnapshotV2,
    metric_name: str,
    previous_value: Decimal,
    current_value: Decimal,
    delta: Decimal,
    metrics: Sequence[str],
    rule_id: str,
    provenance: AxisEvidenceProvenanceV2,
) -> SparseAxisEvidenceV2:
    sources = _source_ids(previous, current, metrics=metrics)
    if not sources:
        return _na_axis(axis, AbstentionReasonV2.TABLE_EXTRACTION_FAIL, rule_id=rule_id)
    return SparseAxisEvidenceV2(
        axis=axis,
        applicability=AxisApplicabilityV2.APPLICABLE,
        availability=SparseAxisAvailabilityV2.GROUNDED,
        direction=direction,
        provenance=provenance,
        previous_evidence_basis=PreviousEvidenceBasisV2.IMMEDIATE_PREVIOUS_FILING,
        confidence=D(1),
        source_ids=sources,
        previous_evidence_at=previous.available_at,
        current_evidence_at=current.available_at,
        applicability_rule_id=rule_id,
        deterministic_metric_name=metric_name,
        deterministic_previous_value=previous_value,
        deterministic_current_value=current_value,
        deterministic_delta=delta,
    )


def build_deterministic_pit_axis_evidence(
    *,
    previous: PITOperatingSnapshotV2,
    current: PITOperatingSnapshotV2,
    rules: PITApplicabilityRulesV2,
) -> dict[OperatingEvidenceAxis, SparseAxisEvidenceV2]:
    if previous.issuer_id != current.issuer_id:
        raise ValueError("PIT snapshots must share one issuer")
    if previous.fiscal_period_end >= current.fiscal_period_end:
        raise ValueError("PIT snapshots must be chronological")
    if previous.available_at > current.available_at:
        raise ValueError("PIT availability must be chronological")
    result: dict[OperatingEvidenceAxis, SparseAxisEvidenceV2] = {}

    margin_rule = "PIT_OPERATING_MARGIN_CHANGE_V2"
    if (
        previous.revenue not in (None, D(0))
        and current.revenue not in (None, D(0))
        and previous.operating_profit is not None
        and current.operating_profit is not None
    ):
        prior_margin = previous.operating_profit / previous.revenue
        current_margin = current.operating_profit / current.revenue
        delta = current_margin - prior_margin
        result[OperatingEvidenceAxis.MARGIN] = _deterministic_axis(
            axis=OperatingEvidenceAxis.MARGIN,
            direction=_direction(delta, rules.margin_change_neutral_tolerance),
            previous=previous,
            current=current,
            metric_name="OPERATING_MARGIN",
            previous_value=prior_margin,
            current_value=current_margin,
            delta=delta,
            metrics=("revenue", "operating_profit"),
            rule_id=margin_rule,
            provenance=AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC,
        )
    else:
        result[OperatingEvidenceAxis.MARGIN] = _na_axis(
            OperatingEvidenceAxis.MARGIN,
            AbstentionReasonV2.TABLE_EXTRACTION_FAIL,
            rule_id=margin_rule,
        )

    inventory_rule = "PIT_INVENTORY_GROWTH_MINUS_REVENUE_GROWTH_V2"
    inventory_ratios = [
        snapshot.inventory / snapshot.assets
        for snapshot in (previous, current)
        if snapshot.inventory is not None and snapshot.assets not in (None, D(0))
    ]
    if inventory_ratios and max(inventory_ratios) < rules.inventory_assets_applicability_threshold:
        result[OperatingEvidenceAxis.INVENTORY_MISMATCH] = _not_applicable_axis(
            OperatingEvidenceAxis.INVENTORY_MISMATCH,
            rule_id=inventory_rule,
        )
    elif all(
        value not in (None, D(0))
        for value in (previous.inventory, previous.revenue)
    ) and current.inventory is not None and current.revenue is not None:
        assert previous.inventory is not None and previous.revenue is not None
        inventory_growth = current.inventory / previous.inventory - D(1)
        revenue_growth = current.revenue / previous.revenue - D(1)
        mismatch = inventory_growth - revenue_growth
        result[OperatingEvidenceAxis.INVENTORY_MISMATCH] = _deterministic_axis(
            axis=OperatingEvidenceAxis.INVENTORY_MISMATCH,
            direction=_direction(
                mismatch,
                rules.inventory_mismatch_neutral_tolerance,
                invert=True,
            ),
            previous=previous,
            current=current,
            metric_name="INVENTORY_GROWTH_MINUS_REVENUE_GROWTH",
            previous_value=D(0),
            current_value=mismatch,
            delta=mismatch,
            metrics=("inventory", "revenue"),
            rule_id=inventory_rule,
            provenance=AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC,
        )
    else:
        result[OperatingEvidenceAxis.INVENTORY_MISMATCH] = _na_axis(
            OperatingEvidenceAxis.INVENTORY_MISMATCH,
            AbstentionReasonV2.TABLE_EXTRACTION_FAIL,
            rule_id=inventory_rule,
        )

    backlog_rule = "PIT_BACKLOG_GROWTH_V2"
    if not (previous.backlog_disclosed or current.backlog_disclosed):
        result[OperatingEvidenceAxis.BACKLOG] = _not_applicable_axis(
            OperatingEvidenceAxis.BACKLOG,
            rule_id=backlog_rule,
        )
    elif previous.backlog not in (None, D(0)) and current.backlog is not None:
        assert previous.backlog is not None
        growth = current.backlog / previous.backlog - D(1)
        result[OperatingEvidenceAxis.BACKLOG] = _deterministic_axis(
            axis=OperatingEvidenceAxis.BACKLOG,
            direction=_direction(growth, rules.backlog_growth_neutral_tolerance),
            previous=previous,
            current=current,
            metric_name="BACKLOG_GROWTH",
            previous_value=previous.backlog,
            current_value=current.backlog,
            delta=growth,
            metrics=("backlog",),
            rule_id=backlog_rule,
            provenance=AxisEvidenceProvenanceV2.STRUCTURED_TABLE,
        )
    else:
        result[OperatingEvidenceAxis.BACKLOG] = _na_axis(
            OperatingEvidenceAxis.BACKLOG,
            AbstentionReasonV2.TABLE_EXTRACTION_FAIL,
            rule_id=backlog_rule,
        )

    capex_rule = "PIT_CAPEX_INTENSITY_CHANGE_V2"
    ppe_ratios = [
        snapshot.ppe / snapshot.assets
        for snapshot in (previous, current)
        if snapshot.ppe is not None and snapshot.assets not in (None, D(0))
    ]
    if not (previous.capacity_disclosed or current.capacity_disclosed) and (
        not ppe_ratios or max(ppe_ratios) < rules.ppe_assets_applicability_threshold
    ):
        result[OperatingEvidenceAxis.CAPACITY_CAPEX] = _not_applicable_axis(
            OperatingEvidenceAxis.CAPACITY_CAPEX,
            rule_id=capex_rule,
        )
    elif (
        previous.capex is not None
        and current.capex is not None
        and previous.revenue not in (None, D(0))
        and current.revenue not in (None, D(0))
    ):
        prior_intensity = previous.capex / previous.revenue
        current_intensity = current.capex / current.revenue
        delta = current_intensity - prior_intensity
        result[OperatingEvidenceAxis.CAPACITY_CAPEX] = _deterministic_axis(
            axis=OperatingEvidenceAxis.CAPACITY_CAPEX,
            direction=_direction(delta, rules.capex_intensity_neutral_tolerance),
            previous=previous,
            current=current,
            metric_name="CAPEX_TO_REVENUE",
            previous_value=prior_intensity,
            current_value=current_intensity,
            delta=delta,
            metrics=("capex", "revenue"),
            rule_id=capex_rule,
            provenance=AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC,
        )
    else:
        result[OperatingEvidenceAxis.CAPACITY_CAPEX] = _na_axis(
            OperatingEvidenceAxis.CAPACITY_CAPEX,
            AbstentionReasonV2.TABLE_EXTRACTION_FAIL,
            rule_id=capex_rule,
        )
    return result


def build_last_grounded_axis_evidence(
    *,
    current: GroundedAxisStateSnapshotV2 | None,
    history: Sequence[GroundedAxisStateSnapshotV2],
    staleness_limit_days: int,
    applicability_rule_id: str,
    axis: OperatingEvidenceAxis | None = None,
) -> SparseAxisEvidenceV2:
    if staleness_limit_days < 1:
        raise ValueError("staleness_limit_days must be positive")
    if current is None:
        if axis is None:
            raise ValueError("axis is required when the current filing has no grounded evidence")
        return _na_axis(
            axis,
            AbstentionReasonV2.TRUE_NO_MENTION,
            rule_id=applicability_rule_id,
        )
    if axis is not None and axis != current.axis:
        raise ValueError("explicit axis does not match current grounded evidence")
    prior_candidates = sorted(
        (
            item
            for item in history
            if item.axis == current.axis and item.available_at < current.available_at
        ),
        key=lambda item: item.available_at,
    )
    if not prior_candidates:
        return _na_axis(
            current.axis,
            AbstentionReasonV2.ONE_PERIOD_ONLY,
            rule_id=applicability_rule_id,
        )
    prior = prior_candidates[-1]
    age = (current.available_at.date() - prior.available_at.date()).days
    if age > staleness_limit_days:
        return _na_axis(
            current.axis,
            AbstentionReasonV2.STALE_PRIOR_STATE,
            rule_id=applicability_rule_id,
        )
    raw = current.state.value - prior.state.value
    direction = EvidenceState((raw > 0) - (raw < 0))
    return SparseAxisEvidenceV2(
        axis=current.axis,
        applicability=AxisApplicabilityV2.APPLICABLE,
        availability=SparseAxisAvailabilityV2.GROUNDED,
        direction=direction,
        provenance=current.provenance,
        previous_evidence_basis=PreviousEvidenceBasisV2.LAST_GROUNDED_WITHIN_STALENESS,
        confidence=min(prior.confidence, current.confidence),
        source_ids=sorted(set(prior.source_ids + current.source_ids)),
        source_spans=prior.source_spans + current.source_spans,
        previous_evidence_at=prior.available_at,
        current_evidence_at=current.available_at,
        prior_age_days=age,
        staleness_limit_days=staleness_limit_days,
        applicability_rule_id=applicability_rule_id,
    )


def qualitative_axis_evidence(
    *,
    classification: AxisPairClassification | None,
    packet: PairedAxisPacket,
    pair: HistoricalFilingPair,
    applicability: AxisApplicabilityV2 = AxisApplicabilityV2.APPLICABLE,
    applicability_rule_id: str = "UNIVERSAL_APPLICABLE_CALIBRATION_ONLY_V2",
) -> SparseAxisEvidenceV2:
    if applicability == AxisApplicabilityV2.NOT_APPLICABLE:
        return _not_applicable_axis(packet.axis, rule_id=applicability_rule_id)
    if classification is None:
        if packet.previous_excerpts and packet.current_excerpts:
            reason = AbstentionReasonV2.CLASSIFICATION_NOT_RUN
        elif packet.previous_excerpts or packet.current_excerpts:
            reason = AbstentionReasonV2.ONE_PERIOD_ONLY
        else:
            reason = AbstentionReasonV2.UNADJUDICATED_ABSTENTION
        return _na_axis(packet.axis, reason, rule_id=applicability_rule_id)
    if classification.status != AxisClassificationStatus.COMPLETE:
        reason = (
            AbstentionReasonV2.AMBIGUOUS_HUMAN_TOO
            if classification.status == AxisClassificationStatus.AMBIGUOUS
            else AbstentionReasonV2.UNADJUDICATED_ABSTENTION
        )
        return _na_axis(packet.axis, reason, rule_id=applicability_rule_id)
    assert classification.delta is not None
    assert classification.previous_source_id is not None
    assert classification.current_source_id is not None
    assert classification.previous_source_span is not None
    assert classification.current_source_span is not None
    return SparseAxisEvidenceV2(
        axis=packet.axis,
        applicability=AxisApplicabilityV2.APPLICABLE,
        availability=SparseAxisAvailabilityV2.GROUNDED,
        direction=EvidenceState(classification.delta),
        provenance=AxisEvidenceProvenanceV2.LLM_NARRATIVE,
        previous_evidence_basis=PreviousEvidenceBasisV2.IMMEDIATE_PREVIOUS_FILING,
        confidence=D(str(classification.confidence)),
        source_ids=[classification.previous_source_id, classification.current_source_id],
        source_spans=[classification.previous_source_span, classification.current_source_span],
        previous_evidence_at=pair.previous.available_at,
        current_evidence_at=pair.current.available_at,
        classification_packet_id=classification.packet_id,
        applicability_rule_id=applicability_rule_id,
    )


def merge_axis_evidence_v2(
    qualitative: SparseAxisEvidenceV2,
    deterministic: SparseAxisEvidenceV2 | None,
) -> SparseAxisEvidenceV2:
    if deterministic is None:
        return qualitative
    if qualitative.axis != deterministic.axis:
        raise ValueError("cannot merge different evidence axes")
    if deterministic.availability == SparseAxisAvailabilityV2.NOT_APPLICABLE:
        return deterministic
    if deterministic.availability == SparseAxisAvailabilityV2.GROUNDED:
        rank = {
            AxisEvidenceProvenanceV2.LLM_NARRATIVE: 1,
            AxisEvidenceProvenanceV2.STRUCTURED_TABLE: 2,
            AxisEvidenceProvenanceV2.DETERMINISTIC_NUMERIC: 3,
        }
        if qualitative.availability != SparseAxisAvailabilityV2.GROUNDED:
            return deterministic
        assert deterministic.provenance is not None and qualitative.provenance is not None
        return (
            deterministic
            if rank[deterministic.provenance] >= rank[qualitative.provenance]
            else qualitative
        )
    return qualitative


def build_sparse_feature_row_v2(
    *,
    pair: HistoricalFilingPair,
    axis_evidence: Sequence[SparseAxisEvidenceV2],
) -> HistoricalSparseEvidenceFeatureRowV2:
    by_axis = {item.axis: item for item in axis_evidence}
    if len(by_axis) != len(axis_evidence):
        raise ValueError("V2 sparse feature axis evidence must be unique")
    applicable = sum(
        item.applicability == AxisApplicabilityV2.APPLICABLE for item in by_axis.values()
    )
    observed = sum(
        item.availability == SparseAxisAvailabilityV2.GROUNDED for item in by_axis.values()
    )
    unavailable = sum(item.availability == SparseAxisAvailabilityV2.NA for item in by_axis.values())
    not_applicable = sum(
        item.availability == SparseAxisAvailabilityV2.NOT_APPLICABLE
        for item in by_axis.values()
    )
    directions = [item.direction.value for item in by_axis.values() if item.direction is not None]
    positive = sum(value > 0 for value in directions)
    neutral = sum(value == 0 for value in directions)
    negative = sum(value < 0 for value in directions)
    net = sum(directions)
    draft = HistoricalSparseEvidenceFeatureRowV2.model_construct(
        observation_id=historical_observation_id(pair.pair_id),
        pair_id=pair.pair_id,
        issuer_id=pair.ticker,
        signal_timestamp=pair.current.signal_timestamp,
        axis_evidence=by_axis,
        applicable_axis_count=applicable,
        observed_axis_count=observed,
        unavailable_axis_count=unavailable,
        not_applicable_axis_count=not_applicable,
        positive_axis_count=positive,
        neutral_axis_count=neutral,
        negative_axis_count=negative,
        n_directional=positive + negative,
        directional_event_count=positive + negative,
        net_evidence=net,
        signed_breadth=D(net) / D(observed) if observed else None,
        coverage=D(observed) / D(applicable) if applicable else None,
        feature_hash="0" * 64,
    )
    payload = draft.model_dump(mode="json", exclude={"feature_hash"})
    return HistoricalSparseEvidenceFeatureRowV2.model_validate(
        {**payload, "feature_hash": canonical_payload_sha256(payload)}
    )


def sparse_feature_coverage_report(
    rows: Sequence[HistoricalSparseEvidenceFeatureRowV2],
) -> dict[str, object]:
    nobs = Counter(item.observed_axis_count for item in rows)
    napplicable = Counter(item.applicable_axis_count for item in rows)
    axes = list(OperatingEvidenceAxis)
    co_observation = {
        left.value: {
            right.value: sum(
                row.axis_evidence[left].availability == SparseAxisAvailabilityV2.GROUNDED
                and row.axis_evidence[right].availability == SparseAxisAvailabilityV2.GROUNDED
                for row in rows
            )
            for right in axes
        }
        for left in axes
    }
    thresholds: dict[str, object] = {}
    for minimum in range(1, 7):
        selected = [item for item in rows if item.observed_axis_count >= minimum]
        issuers = Counter(item.issuer_id for item in selected)
        months = Counter(item.signal_timestamp.strftime("%Y-%m") for item in selected)
        thresholds[str(minimum)] = {
            "eligible_observations": len(selected),
            "unique_issuers": len(issuers),
            "unique_signal_months": len(months),
            "top_issuer_share": (
                issuers.most_common(1)[0][1] / len(selected) if selected else 0.0
            ),
            "top_month_share": months.most_common(1)[0][1] / len(selected) if selected else 0.0,
            "signed_breadth_unique_values": len(
                {item.signed_breadth for item in selected if item.signed_breadth is not None}
            ),
        }
    exact_nobs: dict[str, object] = {}
    for value in range(7):
        selected = [item for item in rows if item.observed_axis_count == value]
        breadth = Counter(
            item.signed_breadth for item in selected if item.signed_breadth is not None
        )
        exact_nobs[str(value)] = {
            "row_count": len(selected),
            "unique_issuers": len({item.issuer_id for item in selected}),
            "unique_signal_months": len(
                {item.signal_timestamp.strftime("%Y-%m") for item in selected}
            ),
            "signed_breadth_distribution": {
                str(key): breadth[key] for key in sorted(breadth)
            },
            "rows_with_directional_event": sum(item.n_directional > 0 for item in selected),
            "rows_with_directional_event_ratio": (
                sum(item.n_directional > 0 for item in selected) / len(selected)
                if selected
                else 0.0
            ),
            "directional_axis_share_of_observed": (
                sum(item.n_directional for item in selected)
                / sum(item.observed_axis_count for item in selected)
                if sum(item.observed_axis_count for item in selected)
                else 0.0
            ),
            "observations": [
                {
                    "observation_id": item.observation_id,
                    "issuer_id": item.issuer_id,
                    "signal_month": item.signal_timestamp.strftime("%Y-%m"),
                    "signed_breadth": item.signed_breadth,
                    "n_directional": item.n_directional,
                    "directional_ratio": (
                        D(item.n_directional) / D(item.observed_axis_count)
                        if item.observed_axis_count
                        else None
                    ),
                }
                for item in sorted(selected, key=lambda row: row.observation_id)
            ],
        }
    axis_report: dict[str, object] = {}
    for axis in axes:
        evidence = [row.axis_evidence[axis] for row in rows]
        directions = Counter(
            item.direction.value for item in evidence if item.direction is not None
        )
        abstentions = Counter(
            item.abstention_reason.value
            for item in evidence
            if item.abstention_reason is not None
        )
        axis_report[axis.value] = {
            "applicable": sum(item.applicability == AxisApplicabilityV2.APPLICABLE for item in evidence),
            "observed": sum(item.availability == SparseAxisAvailabilityV2.GROUNDED for item in evidence),
            "na": sum(item.availability == SparseAxisAvailabilityV2.NA for item in evidence),
            "not_applicable": sum(
                item.availability == SparseAxisAvailabilityV2.NOT_APPLICABLE for item in evidence
            ),
            "-1": directions[-1],
            "0": directions[0],
            "+1": directions[1],
            "abstention_reason_distribution": dict(sorted(abstentions.items())),
        }
    provenance = Counter(
        item.provenance.value
        for row in rows
        for item in row.axis_evidence.values()
        if item.provenance is not None
    )
    years = Counter(item.signal_timestamp.strftime("%Y") for item in rows)
    ndir = Counter(item.n_directional for item in rows)
    return {
        "schema_version": "moatrader-sparse-feature-coverage-report-v2/1",
        "pair_count": len(rows),
        "grounded_axis_count_histogram": {str(key): nobs[key] for key in range(7)},
        "applicable_axis_count_histogram": {
            str(key): napplicable[key] for key in range(7)
        },
        "pairwise_co_observation_matrix": co_observation,
        "minimum_observed_axis_scenarios": thresholds,
        "by_exact_nobs": exact_nobs,
        "axis_measurement_distribution": axis_report,
        "n_directional_histogram": {str(key): ndir[key] for key in range(7)},
        "directional_axis_event_count": sum(item.n_directional for item in rows),
        "grounded_axis_observation_count": sum(item.observed_axis_count for item in rows),
        "signed_breadth_distribution": {
            str(value): count
            for value, count in sorted(
                Counter(
                    item.signed_breadth
                    for item in rows
                    if item.signed_breadth is not None
                ).items()
            )
        },
        "signal_year_distribution": dict(sorted(years.items())),
        "source_type_distribution": dict(sorted(provenance.items())),
        "outcomes_opened": False,
        "returns_opened": False,
        "per_pbr_role": "NOT_USED",
    }


def _pearson(left: Sequence[Decimal], right: Sequence[Decimal]) -> Decimal | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = sum(left, D(0)) / D(len(left))
    right_mean = sum(right, D(0)) / D(len(right))
    numerator = sum(
        ((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True)),
        D(0),
    )
    left_scale = sum(((value - left_mean) ** 2 for value in left), D(0))
    right_scale = sum(((value - right_mean) ** 2 for value in right), D(0))
    denominator = sqrt(float(left_scale * right_scale))
    return D(str(float(numerator) / denominator)) if denominator else None


def sparse_band_diagnostics_v2(
    rows: Sequence[HistoricalSparseEvidenceFeatureRowV2],
    *,
    band_contract: SparseBreadthBandContractV2,
) -> dict[str, object]:
    selected = [
        row
        for row in rows
        if row.observed_axis_count >= band_contract.minimum_observed_axes
        and row.signed_breadth is not None
    ]
    by_band: dict[str, object] = {}
    for band in SparseBreadthBandV2:
        band_rows = [
            row for row in selected if band_contract.band_for(row.signed_breadth) == band
        ]
        issuers = Counter(row.issuer_id for row in band_rows)
        months = Counter(row.signal_timestamp.strftime("%Y-%m") for row in band_rows)
        years = Counter(row.signal_timestamp.strftime("%Y") for row in band_rows)
        nobs = Counter(row.observed_axis_count for row in band_rows)
        ndir = Counter(row.n_directional for row in band_rows)
        provenance = Counter(
            item.provenance.value
            for row in band_rows
            for item in row.axis_evidence.values()
            if item.provenance is not None
        )
        sources = Counter(
            source_id
            for row in band_rows
            for item in row.axis_evidence.values()
            for source_id in item.source_ids
        )
        by_band[band.value] = {
            "row_count": len(band_rows),
            "unique_issuers": len(issuers),
            "unique_signal_months": len(months),
            "top_issuer_share": (
                D(issuers.most_common(1)[0][1]) / D(len(band_rows)) if band_rows else D(0)
            ),
            "top_year_share": (
                D(years.most_common(1)[0][1]) / D(len(band_rows)) if band_rows else D(0)
            ),
            "top_evidence_source_share": (
                D(sources.most_common(1)[0][1]) / D(sum(sources.values()))
                if sources
                else D(0)
            ),
            "nobs_distribution": {str(key): nobs[key] for key in range(7)},
            "n_directional_distribution": {str(key): ndir[key] for key in range(7)},
            "signal_year_distribution": dict(sorted(years.items())),
            "source_type_distribution": dict(sorted(provenance.items())),
        }
    abs_breadth = [abs(row.signed_breadth) for row in selected]
    return {
        "schema_version": "moatrader-sparse-band-diagnostics-v2/1",
        "eligible_row_count": len(selected),
        "unique_issuers": len({row.issuer_id for row in selected}),
        "unique_signal_months": len(
            {row.signal_timestamp.strftime("%Y-%m") for row in selected}
        ),
        "corr_abs_signed_breadth_coverage": _pearson(
            abs_breadth,
            [row.coverage for row in selected if row.coverage is not None],
        ),
        "corr_abs_signed_breadth_nobs": _pearson(
            abs_breadth,
            [D(row.observed_axis_count) for row in selected],
        ),
        "by_band": by_band,
        "outcomes_opened": False,
        "returns_opened": False,
    }


def evaluate_sparse_coverage_gate_v2(
    diagnostics: dict[str, object],
    *,
    policy: SparseCoverageGatePolicyV2,
) -> dict[str, object]:
    band_results: dict[str, object] = {}
    by_band = dict(diagnostics["by_band"])
    for band in SparseBreadthBandV2:
        values = dict(by_band[band.value])
        checks = {
            "minimum_rows": values["row_count"] >= policy.minimum_rows_per_band,
            "minimum_unique_issuers": (
                values["unique_issuers"] >= policy.minimum_unique_issuers_per_band
            ),
            "minimum_unique_signal_months": (
                values["unique_signal_months"] >= policy.minimum_unique_signal_months_per_band
            ),
            "issuer_concentration": (
                values["top_issuer_share"] <= policy.maximum_top_issuer_share_per_band
            ),
            "year_concentration": (
                values["top_year_share"] <= policy.maximum_top_year_share_per_band
            ),
            "evidence_source_concentration": (
                values["top_evidence_source_share"]
                <= policy.maximum_top_evidence_source_share_per_band
            ),
        }
        band_results[band.value] = {
            "passed": all(checks.values()),
            "checks": checks,
            "observed": values,
        }
    global_checks = {
        "minimum_total_unique_issuers": (
            diagnostics["unique_issuers"] >= policy.minimum_total_unique_issuers
        ),
        "minimum_total_unique_signal_months": (
            diagnostics["unique_signal_months"] >= policy.minimum_total_unique_signal_months
        ),
    }
    passed = all(global_checks.values()) and all(
        dict(value)["passed"] for value in band_results.values()
    )
    return {
        "schema_version": "moatrader-sparse-coverage-gate-report-v2/1",
        "status": "PASSED" if passed else "FAILED_MEASUREMENT_COVERAGE",
        "gate_passed": passed,
        "policy": policy.model_dump(mode="json"),
        "global_checks": global_checks,
        "by_band": band_results,
        "outcome_data_accessed": False,
        "return_data_accessed": False,
    }


def calibrate_sparse_band_contract_v2(
    rows: Sequence[HistoricalSparseEvidenceFeatureRowV2],
    *,
    minimum_observed_axes: int,
    minimum_rows_per_band: int,
) -> SparseBreadthBandContractV2:
    if not 1 <= minimum_observed_axes <= 6:
        raise ValueError("minimum_observed_axes must be in [1, 6]")
    selected = [
        item
        for item in rows
        if item.observed_axis_count >= minimum_observed_axes
        and item.signed_breadth is not None
    ]
    unique = sorted({item.signed_breadth for item in selected if item.signed_breadth is not None})
    if len(unique) < 5:
        raise ValueError("at least five unique breadth values are required to freeze five bands")
    boundaries: list[Decimal] = []
    for quantile in range(1, 5):
        upper_index = max(1, min(len(unique) - 1, (len(unique) * quantile) // 5))
        lower = unique[upper_index - 1]
        upper = unique[upper_index]
        boundaries.append((lower + upper) / D(2))
    if len(set(boundaries)) != 4:
        raise ValueError("feature-only unique-value quantiles did not produce five bands")
    provisional = SparseBreadthBandContractV2(
        minimum_observed_axes=minimum_observed_axes,
        cut_points=tuple(boundaries),  # type: ignore[arg-type]
        calibration_feature_count=len(selected),
        calibration_feature_dataset_sha256=canonical_payload_sha256(
            [item.model_dump(mode="json") for item in sorted(rows, key=lambda row: row.observation_id)]
        ),
        band_counts={band: 0 for band in SparseBreadthBandV2},
        minimum_rows_per_band=minimum_rows_per_band,
        all_bands_sufficient=False,
    )
    counts = Counter(
        provisional.band_for(item.signed_breadth)
        for item in selected
        if item.signed_breadth is not None
    )
    band_counts = {band: counts[band] for band in SparseBreadthBandV2}
    return SparseBreadthBandContractV2.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "band_counts": band_counts,
            "all_bands_sufficient": all(
                value >= minimum_rows_per_band for value in band_counts.values()
            ),
        }
    )


def seal_sparse_features_v2(
    rows: Sequence[HistoricalSparseEvidenceFeatureRowV2],
    *,
    band_contract: SparseBreadthBandContractV2,
    parser_validation_manifest_sha256: str,
    contract_freeze_manifest_sha256: str,
    abstention_audit_manifest_sha256: str,
    coverage_gate_report_sha256: str,
    sealed_at: datetime,
) -> HistoricalSparseEvidenceDatasetSealV2:
    if not band_contract.all_bands_sufficient:
        raise ValueError("cannot seal V2 outcome features before all bands are sufficient")
    eligible = sorted(
        (item for item in rows if item.observed_axis_count >= band_contract.minimum_observed_axes),
        key=lambda item: item.observation_id,
    )
    if not eligible:
        raise ValueError("cannot seal an empty V2 sparse feature dataset")
    payload = [item.model_dump(mode="json") for item in eligible]
    return HistoricalSparseEvidenceDatasetSealV2(
        sealed_at=sealed_at,
        feature_count=len(eligible),
        observation_ids=[item.observation_id for item in eligible],
        feature_dataset_sha256=canonical_payload_sha256(payload),
        feature_row_sha256={item.observation_id: item.feature_hash for item in eligible},
        band_contract_sha256=canonical_payload_sha256(band_contract.model_dump(mode="json")),
        parser_validation_manifest_sha256=parser_validation_manifest_sha256,
        contract_freeze_manifest_sha256=contract_freeze_manifest_sha256,
        abstention_audit_manifest_sha256=abstention_audit_manifest_sha256,
        coverage_gate_report_sha256=coverage_gate_report_sha256,
        minimum_observed_axes=band_contract.minimum_observed_axes,
    )
