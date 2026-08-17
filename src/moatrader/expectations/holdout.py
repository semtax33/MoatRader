from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel, SourceType
from moatrader.expectations.alpha import (
    AlphaSignal,
    assign_method_archetype_percentiles,
)
from moatrader.expectations.risk import (
    FrozenRiskOverlayPolicy,
    RiskOverlayDecision,
    RiskProfile,
    ThesisConfirmation,
)


SEOUL = ZoneInfo("Asia/Seoul")


class HoldoutSourceReference(ContractModel):
    document_id: str = Field(min_length=1)
    source_type: SourceType
    available_at: datetime

    @model_validator(mode="after")
    def timezone_aware(self) -> "HoldoutSourceReference":
        if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
            raise ValueError("holdout source available_at must be timezone-aware")
        return self


class HoldoutResearchInput(ContractModel):
    """Return-blind research vector assembled before the signal is sealed."""

    ticker: str = Field(pattern=r"^[0-9]{6}$")
    risk: RiskProfile
    confirmation: ThesisConfirmation
    source_references: list[HoldoutSourceReference] = Field(min_length=1)
    legacy_composite_diagnostic: float | None = Field(default=None, ge=0, le=100)


class HoldoutSignal(ContractModel):
    signal_date: date
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    issuer_name: str
    sector: str
    sector_snapshot_date: date
    sector_evidence_ref: str = Field(min_length=1)
    alpha: AlphaSignal
    risk: RiskProfile
    confirmation: ThesisConfirmation
    route_profile_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_references: list[HoldoutSourceReference] = Field(min_length=1)
    legacy_composite_diagnostic: float | None = Field(default=None, ge=0, le=100)

    @model_validator(mode="after")
    def production_rank_is_method_archetype_cheap(self) -> "HoldoutSignal":
        cheap = self.alpha.cheap
        if cheap.rank_eligible and cheap.method_archetype_percentile is None:
            raise ValueError("holdout Cheap rank requires method x archetype percentile")
        if self.sector_snapshot_date > self.signal_date:
            raise ValueError("holdout sector snapshot cannot be after the signal date")
        cutoff = datetime.combine(self.signal_date, time.max, tzinfo=SEOUL)
        if any(item.available_at > cutoff for item in self.source_references):
            raise ValueError("holdout source was not available by the signal-date cutoff")
        return self


def verify_and_normalize_holdout_ranks(
    signals: list[HoldoutSignal],
) -> list[HoldoutSignal]:
    """Recompute the frozen cohort rank and reject caller-supplied rank drift."""

    normalized_cheap = assign_method_archetype_percentiles(
        [item.alpha.cheap for item in signals]
    )
    output: list[HoldoutSignal] = []
    for signal, cheap in zip(signals, normalized_cheap, strict=True):
        supplied = signal.alpha.cheap
        if supplied.method_percentile != cheap.method_percentile:
            raise ValueError(f"method percentile drift for {signal.ticker}")
        if supplied.method_archetype_percentile != cheap.method_archetype_percentile:
            raise ValueError(f"method x archetype percentile drift for {signal.ticker}")
        output.append(
            signal.model_copy(
                update={"alpha": signal.alpha.model_copy(update={"cheap": cheap})},
                deep=True,
            )
        )
    return output


class HoldoutCandidates(ContractModel):
    signal_date: date
    ticker: str = Field(pattern=r"^[0-9]{6}$")
    cheap_rank: float | None = Field(default=None, ge=0, le=100)
    candidate_a_eligible: bool
    candidate_b_eligible: bool
    candidate_c_eligible: bool
    candidate_c_position_multiplier: float = Field(ge=0, le=1)
    risk_decision: RiskOverlayDecision
    risk_reason_codes: list[str] = Field(default_factory=list)


def build_holdout_candidates(
    signal: HoldoutSignal,
    *,
    policy: FrozenRiskOverlayPolicy,
) -> HoldoutCandidates:
    cheap = signal.alpha.cheap
    rank = cheap.method_archetype_percentile if cheap.rank_eligible else None
    candidate_a = rank is not None
    candidate_b = candidate_a and signal.risk.three_p.hard_gate_pass
    overlay = policy.apply(signal.risk)
    candidate_c = candidate_b and overlay.decision != RiskOverlayDecision.EXCLUDED
    multiplier = overlay.position_multiplier if candidate_c else 0.0
    return HoldoutCandidates(
        signal_date=signal.signal_date,
        ticker=signal.ticker,
        cheap_rank=rank,
        candidate_a_eligible=candidate_a,
        candidate_b_eligible=candidate_b,
        candidate_c_eligible=candidate_c,
        candidate_c_position_multiplier=multiplier,
        risk_decision=overlay.decision,
        risk_reason_codes=overlay.reason_codes,
    )
