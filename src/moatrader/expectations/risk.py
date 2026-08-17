from __future__ import annotations

from enum import StrEnum

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.valuation.three_p import (
    CheckStatus,
    PlausibilityStatus,
    ProbabilitySupport,
    ThreePResult,
)


class ConfirmationStatus(StrEnum):
    AVAILABLE = "AVAILABLE"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ThesisConfirmation(ContractModel):
    """Improving is a diagnostic, not an alpha weight or risk score."""

    improving: float | None = Field(default=None, ge=0, le=100)
    status: ConfirmationStatus

    @model_validator(mode="after")
    def consistent_availability(self) -> "ThesisConfirmation":
        if self.status == ConfirmationStatus.AVAILABLE and self.improving is None:
            raise ValueError("AVAILABLE confirmation requires an improving value")
        if self.status == ConfirmationStatus.INSUFFICIENT_EVIDENCE and self.improving is not None:
            raise ValueError("insufficient confirmation evidence must not be imputed")
        return self


class ThreePValidity(ContractModel):
    possible: CheckStatus
    plausible: PlausibilityStatus
    probable: ProbabilitySupport
    hard_gate_pass: bool
    review_required: bool

    @model_validator(mode="after")
    def deterministic_gate(self) -> "ThreePValidity":
        expected_gate = self.possible != CheckStatus.FAIL
        expected_review = (
            self.plausible in {PlausibilityStatus.OUTLIER, PlausibilityStatus.UNKNOWN}
            or self.probable
            in {
                ProbabilitySupport.CONTRADICTED,
                ProbabilitySupport.MIXED,
                ProbabilitySupport.WEAK,
            }
        )
        if self.hard_gate_pass != expected_gate:
            raise ValueError("3P hard gate is determined only by Possible != FAIL")
        if self.review_required != expected_review:
            raise ValueError("3P review flag must preserve plausible/probable uncertainty")
        return self

    @classmethod
    def from_result(cls, result: ThreePResult) -> "ThreePValidity":
        return cls(
            possible=result.possible,
            plausible=result.plausible,
            probable=result.probable,
            hard_gate_pass=result.possible != CheckStatus.FAIL,
            review_required=(
                result.plausible in {PlausibilityStatus.OUTLIER, PlausibilityStatus.UNKNOWN}
                or result.probable
                in {
                    ProbabilitySupport.CONTRADICTED,
                    ProbabilitySupport.MIXED,
                    ProbabilitySupport.WEAK,
                }
            ),
        )


class RiskProfile(ContractModel):
    """Unweighted risk vector frozen before any holdout return is joined."""

    fragility_score: float | None = Field(default=None, ge=0, le=100)
    three_p: ThreePValidity
    industry_counterevidence_count: int | None = Field(default=None, ge=0)
    industry_range_widener_count: int | None = Field(default=None, ge=0)
    industry_evidence_available: bool = False
    reason_codes: list[str] = Field(default_factory=list)


class ValuationFragilityDiagnostics(ContractModel):
    """Method-neutral, return-blind fragility inputs from a scenario valuation."""

    downside_value_per_share: float
    base_value_per_share: float
    upside_value_per_share: float
    assumption_confidence: float | None = Field(default=None, ge=0, le=1)
    warning_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def ordered_scenarios(self) -> "ValuationFragilityDiagnostics":
        if not (
            self.downside_value_per_share
            <= self.base_value_per_share
            <= self.upside_value_per_share
        ):
            raise ValueError("fragility scenarios must be ordered downside <= base <= upside")
        return self

    def score(self) -> float | None:
        """Return a 0-100 risk score without fitting weights to returns."""

        scale = abs(self.base_value_per_share)
        if scale <= 0:
            return None
        downside_loss = min(
            100.0,
            100.0
            * max(0.0, self.base_value_per_share - self.downside_value_per_share)
            / scale,
        )
        scenario_dispersion = min(
            100.0,
            50.0
            * abs(self.upside_value_per_share - self.downside_value_per_share)
            / scale,
        )
        confidence_weakness = (
            100.0 * (1.0 - self.assumption_confidence)
            if self.assumption_confidence is not None
            else 100.0
        )
        warning_burden = min(100.0, 20.0 * self.warning_count)
        return max(
            0.0,
            min(
                100.0,
                0.35 * downside_loss
                + 0.35 * scenario_dispersion
                + 0.20 * confidence_weakness
                + 0.10 * warning_burden,
            ),
        )


class RiskOverlayDecision(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    POSITION_CAP = "POSITION_CAP"
    EXCLUDED = "EXCLUDED"


class RiskOverlayResult(ContractModel):
    """Frozen policy output; never changes the Cheap rank value."""

    decision: RiskOverlayDecision
    position_multiplier: float = Field(ge=0, le=1)
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_position(self) -> "RiskOverlayResult":
        if self.decision == RiskOverlayDecision.EXCLUDED and self.position_multiplier != 0:
            raise ValueError("excluded candidates must have zero position multiplier")
        if self.decision == RiskOverlayDecision.ELIGIBLE and self.position_multiplier != 1:
            raise ValueError("eligible candidates must retain full position multiplier")
        if self.decision == RiskOverlayDecision.POSITION_CAP and not 0 < self.position_multiplier < 1:
            raise ValueError("position caps require a multiplier strictly between zero and one")
        return self


class FrozenRiskOverlayPolicy(ContractModel):
    """Return-blind production-candidate policy frozen after engineering stability.

    Thresholds express model-risk severity, not historical return optimization.
    Missing industry context is reported but does not punish source availability.
    """

    contract_version: str = "risk-overlay/1"
    high_fragility_threshold: float = Field(default=70.0, ge=0, le=100)
    severe_industry_counterevidence_min: int = Field(default=2, ge=1)
    severe_industry_range_widener_min: int = Field(default=3, ge=1)
    position_cap_multiplier: float = Field(default=0.5, gt=0, lt=1)
    exclude_possible_fail: bool = True
    exclude_combined_severe_risks: bool = True

    def apply(self, profile: RiskProfile) -> RiskOverlayResult:
        reasons: list[str] = []
        if self.exclude_possible_fail and not profile.three_p.hard_gate_pass:
            return RiskOverlayResult(
                decision=RiskOverlayDecision.EXCLUDED,
                position_multiplier=0.0,
                reason_codes=["THREE_P_POSSIBLE_FAIL"],
            )
        high_fragility = (
            profile.fragility_score is not None
            and profile.fragility_score >= self.high_fragility_threshold
        )
        if profile.fragility_score is None:
            reasons.append("FRAGILITY_MISSING")
        elif high_fragility:
            reasons.append("HIGH_FRAGILITY")
        severe_industry = bool(
            profile.industry_evidence_available
            and (
                (profile.industry_counterevidence_count or 0)
                >= self.severe_industry_counterevidence_min
                or (profile.industry_range_widener_count or 0)
                >= self.severe_industry_range_widener_min
            )
        )
        if not profile.industry_evidence_available:
            reasons.append("INDUSTRY_CONTEXT_MISSING")
        elif severe_industry:
            reasons.append("SEVERE_INDUSTRY_RISK")
        if self.exclude_combined_severe_risks and high_fragility and severe_industry:
            return RiskOverlayResult(
                decision=RiskOverlayDecision.EXCLUDED,
                position_multiplier=0.0,
                reason_codes=reasons,
            )
        if high_fragility or severe_industry or profile.fragility_score is None:
            return RiskOverlayResult(
                decision=RiskOverlayDecision.POSITION_CAP,
                position_multiplier=self.position_cap_multiplier,
                reason_codes=reasons,
            )
        return RiskOverlayResult(
            decision=RiskOverlayDecision.ELIGIBLE,
            position_multiplier=1.0,
            reason_codes=reasons,
        )
