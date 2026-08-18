from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel, SourceType
from moatrader.valuation.base import ValuationMethod


def canonical_sha256(payload: dict[str, object], *, excluded: set[str] | None = None) -> str:
    canonical = dict(payload)
    for field in excluded or set():
        canonical.pop(field, None)
    def json_default(value: object) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=json_default,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _aware(value: datetime, *, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


class ShadowRankStatus(StrEnum):
    VALID = "VALID"
    MODEL_NOT_APPLICABLE = "MODEL_NOT_APPLICABLE"
    INVALID_VALUATION = "INVALID_VALUATION"


class ShadowSourceReference(ContractModel):
    document_id: str = Field(min_length=1)
    source_type: SourceType
    available_at: datetime
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def timestamp_is_aware(self) -> "ShadowSourceReference":
        _aware(self.available_at, field="available_at")
        return self


class ShadowCompanySignal(ContractModel):
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    signal_at: datetime
    valuation_method: ValuationMethod
    economic_archetype: str = Field(min_length=1)
    status: ShadowRankStatus
    market_price: Decimal | None = Field(default=None, gt=0)
    fair_value_per_share: Decimal | None = None
    expectation_gap: Decimal | None = None
    rank_percentile: Decimal | None = Field(default=None, ge=0, le=100)
    source_references: list[ShadowSourceReference] = Field(min_length=1)
    diagnostics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def status_controls_rank_fields(self) -> "ShadowCompanySignal":
        _aware(self.signal_at, field="signal_at")
        if self.status == ShadowRankStatus.VALID:
            required = (
                self.market_price,
                self.fair_value_per_share,
                self.expectation_gap,
                self.rank_percentile,
            )
            if any(value is None for value in required):
                raise ValueError("VALID shadow signal requires price, fair value, gap, and rank")
            assert self.market_price is not None
            assert self.fair_value_per_share is not None
            assert self.expectation_gap is not None
            if self.fair_value_per_share <= 0:
                raise ValueError("VALID shadow signal requires positive fair value")
            expected_gap = self.fair_value_per_share / self.market_price - Decimal(1)
            if abs(expected_gap - self.expectation_gap) > Decimal("1e-12"):
                raise ValueError("shadow expectation gap must equal fair value / price - 1")
        elif self.expectation_gap is not None or self.rank_percentile is not None:
            raise ValueError("invalid/inapplicable shadow signals cannot publish a Cheap rank")
        return self


class ExpectationGapResearchContract(ContractModel):
    schema_version: str = "expectation-gap-v7-research/1"
    created_at: datetime
    parent_v6_contract_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_v6_contract_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_engineering_input_sha256: dict[str, str]
    expected_universe_count: int = Field(gt=0)
    scheduled_signal_at: list[datetime] = Field(min_length=1)
    cadence: str = "WEEKLY"
    research_horizons_calendar_days: list[int] = Field(default_factory=lambda: [21, 42, 77])
    primary_horizon_calendar_days: int = 77
    overlapping_horizons_are_not_independent: bool = True
    v6_results_must_not_modify_v7: bool = True
    v7_results_must_only_modify_v8: bool = True
    analyst_market_opinion_intrinsic_access: bool = False
    return_inputs_forbidden_before_signal_seal: bool = True
    return_data_accessed: bool = False
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def frozen_research_rules(self) -> "ExpectationGapResearchContract":
        _aware(self.created_at, field="created_at")
        if self.return_data_accessed or not self.return_inputs_forbidden_before_signal_seal:
            raise ValueError("v7 research contract must be return-blind before sealing")
        if self.analyst_market_opinion_intrinsic_access:
            raise ValueError("market opinion cannot enter the intrinsic valuation lane")
        if sorted(set(self.research_horizons_calendar_days)) != [21, 42, 77]:
            raise ValueError("v7 research horizons must be preregistered as 21/42/77 days")
        if self.primary_horizon_calendar_days != 77:
            raise ValueError("v7 primary horizon remains 77 calendar days")
        for item in self.scheduled_signal_at:
            _aware(item, field="scheduled_signal_at")
        if len(set(self.scheduled_signal_at)) != len(self.scheduled_signal_at):
            raise ValueError("shadow schedule timestamps must be unique")
        ordered = sorted(self.scheduled_signal_at)
        if ordered != self.scheduled_signal_at:
            raise ValueError("shadow schedule must be chronological")
        if any(right - left != timedelta(days=7) for left, right in zip(ordered, ordered[1:])):
            raise ValueError("shadow schedule must use an exact weekly cadence")
        actual = canonical_sha256(
            self.model_dump(mode="json"), excluded={"contract_sha256"}
        )
        if actual != self.contract_sha256:
            raise ValueError("v7 research contract hash mismatch")
        return self

    @classmethod
    def create(cls, **payload: object) -> "ExpectationGapResearchContract":
        draft = cls.model_construct(contract_sha256="0" * 64, **payload)
        canonical = draft.model_dump(mode="json", exclude={"contract_sha256"})
        canonical["contract_sha256"] = canonical_sha256(canonical)
        return cls.model_validate(canonical)


class ShadowSnapshot(ContractModel):
    schema_version: str = "v7-shadow-snapshot/1"
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    signal_at: datetime
    sealed_at: datetime
    signals: list[ShadowCompanySignal] = Field(min_length=1)
    return_data_accessed: bool = False
    snapshot_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def immutable_snapshot_contract(self) -> "ShadowSnapshot":
        _aware(self.signal_at, field="signal_at")
        _aware(self.sealed_at, field="sealed_at")
        if self.return_data_accessed:
            raise ValueError("shadow signals must be sealed before return access")
        actual = canonical_sha256(
            self.model_dump(mode="json"), excluded={"snapshot_sha256"}
        )
        if actual != self.snapshot_sha256:
            raise ValueError("shadow snapshot hash mismatch")
        return self


def seal_shadow_snapshot(
    *,
    contract: ExpectationGapResearchContract,
    signal_at: datetime,
    sealed_at: datetime,
    signals: list[ShadowCompanySignal],
    output_path: Path,
) -> ShadowSnapshot:
    if output_path.exists():
        raise FileExistsError(f"shadow snapshot is immutable and already exists: {output_path}")
    if signal_at not in contract.scheduled_signal_at:
        raise ValueError("signal timestamp is not in the frozen weekly research schedule")
    if len(signals) != contract.expected_universe_count:
        raise ValueError(
            f"shadow snapshot requires {contract.expected_universe_count} companies, got {len(signals)}"
        )
    tickers = [item.ticker for item in signals]
    if len(tickers) != len(set(tickers)):
        raise ValueError("shadow snapshot tickers must be unique")
    for signal in signals:
        if signal.signal_at != signal_at:
            raise ValueError(f"signal timestamp mismatch for {signal.ticker}")
        for source in signal.source_references:
            if source.available_at > signal_at:
                raise ValueError(f"future-source leakage for {signal.ticker}: {source.document_id}")
    _aware(sealed_at, field="sealed_at")
    if sealed_at < signal_at:
        raise ValueError("snapshot cannot be sealed before its signal timestamp")
    payload: dict[str, object] = {
        "schema_version": "v7-shadow-snapshot/1",
        "contract_sha256": contract.contract_sha256,
        "signal_at": signal_at,
        "sealed_at": sealed_at,
        "signals": [item.model_dump(mode="json") for item in sorted(signals, key=lambda item: item.ticker)],
        "return_data_accessed": False,
    }
    payload["snapshot_sha256"] = canonical_sha256(payload)
    snapshot = ShadowSnapshot.model_validate(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return snapshot
