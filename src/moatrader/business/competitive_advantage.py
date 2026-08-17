from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import Field, model_validator

from moatrader.business.drivers import (
    ValuationDriver,
    ValuationDriverEvidenceBundle,
    ValuationEvidenceRole,
)
from moatrader.canonical.models import ContractModel
from moatrader.evidence.models import AtomicMoatRole, EvidenceType


class CapEvidenceStrength(StrEnum):
    UNSUPPORTED = "UNSUPPORTED"
    BASE_RATE_ONLY = "BASE_RATE_ONLY"
    CORROBORATED = "CORROBORATED"
    STRONG = "STRONG"
    ERODING = "ERODING"


class CompetitiveAdvantageProfile(ContractModel):
    """Evidence profile, deliberately not a scalar MOAT score."""

    issuer_id: str = Field(min_length=1)
    mechanism_evidence_ids: dict[EvidenceType, list[str]] = Field(default_factory=dict)
    outcome_evidence_ids: list[str] = Field(default_factory=list)
    counterevidence_ids: list[str] = Field(default_factory=list)
    cap_support_evidence_ids: list[str] = Field(default_factory=list)
    cap_erosion_evidence_ids: list[str] = Field(default_factory=list)
    observed_persistence_years: list[int] = Field(default_factory=list)
    range_widener_evidence_ids: list[str] = Field(default_factory=list)

    @classmethod
    def from_driver_evidence(
        cls,
        bundle: ValuationDriverEvidenceBundle,
    ) -> "CompetitiveAdvantageProfile":
        mechanisms: dict[EvidenceType, list[str]] = {}
        outcomes: list[str] = []
        counters: list[str] = []
        cap_support: list[str] = []
        cap_erosion: list[str] = []
        persistence: list[int] = []
        wideners: list[str] = []
        for item in bundle.evidence:
            if item.moat_role == AtomicMoatRole.MECHANISM:
                mechanisms.setdefault(item.evidence_type, []).append(item.evidence_id)
            elif item.moat_role == AtomicMoatRole.OUTCOME:
                outcomes.append(item.evidence_id)
            elif item.moat_role == AtomicMoatRole.COUNTER:
                counters.append(item.evidence_id)
            # Related drivers are explanatory only. Counting them here would
            # apply the same fact to (for example) both target margin and CAP.
            cap_related = item.primary_driver == ValuationDriver.CAP_FADE
            if cap_related and item.role == ValuationEvidenceRole.COUNTER:
                cap_erosion.append(item.evidence_id)
            elif cap_related and item.role == ValuationEvidenceRole.SUPPORT:
                cap_support.append(item.evidence_id)
            if cap_related and item.persistence_years_observed is not None:
                persistence.append(item.persistence_years_observed)
            if cap_related and item.range_widening_required:
                wideners.append(item.evidence_id)
        return cls(
            issuer_id=bundle.issuer_id,
            mechanism_evidence_ids=mechanisms,
            outcome_evidence_ids=outcomes,
            counterevidence_ids=counters,
            cap_support_evidence_ids=cap_support,
            cap_erosion_evidence_ids=cap_erosion,
            observed_persistence_years=sorted(persistence),
            range_widener_evidence_ids=wideners,
        )


class CapPrior(ContractModel):
    reference_class: str = Field(min_length=1)
    as_of: datetime
    low_years: int = Field(ge=0, le=50)
    central_years: int = Field(ge=0, le=50)
    high_years: int = Field(ge=0, le=50)
    maximum_years: int = Field(default=30, ge=1, le=100)
    source_refs: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered(self) -> "CapPrior":
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("CAP prior as_of must be timezone-aware")
        if not self.low_years <= self.central_years <= self.high_years <= self.maximum_years:
            raise ValueError("CAP prior must satisfy low <= central <= high <= maximum")
        return self


class CapAssessment(ContractModel):
    reference_class: str
    low_years: int = Field(ge=0)
    central_years: int = Field(ge=0)
    high_years: int = Field(ge=0)
    strength: CapEvidenceStrength
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    erosion_evidence_ids: list[str] = Field(default_factory=list)
    rationale: list[str] = Field(default_factory=list)
    price_inputs_used: bool = False

    @model_validator(mode="after")
    def ordered_and_price_blind(self) -> "CapAssessment":
        if not self.low_years <= self.central_years <= self.high_years:
            raise ValueError("CAP assessment must satisfy low <= central <= high")
        if self.price_inputs_used:
            raise ValueError("CAP assessment must be price-blind")
        return self


class CapEngine:
    """Converts mechanism/durability evidence into a bounded CAP range.

    The reducer starts from an external reference-class prior. It never maps a
    public MOAT score to years and never changes growth, margin, WACC, or terminal
    growth at the same time.
    """

    def assess(
        self,
        profile: CompetitiveAdvantageProfile,
        prior: CapPrior,
    ) -> CapAssessment:
        cap_support_ids = set(profile.cap_support_evidence_ids)
        independent_mechanisms = len(
            [
                kind
                for kind, ids in profile.mechanism_evidence_ids.items()
                if cap_support_ids.intersection(ids)
            ]
        )
        support = len(cap_support_ids)
        erosion = len(set(profile.cap_erosion_evidence_ids))
        has_outcome = bool(cap_support_ids.intersection(profile.outcome_evidence_ids))
        rationale = [
            f"reference-class prior={prior.low_years}-{prior.high_years} years",
            f"independent mechanisms={independent_mechanisms}",
            f"CAP support={support}, erosion={erosion}",
        ]

        low, central, high = prior.low_years, prior.central_years, prior.high_years
        strength = CapEvidenceStrength.BASE_RATE_ONLY
        if erosion > support:
            strength = CapEvidenceStrength.ERODING
            low = max(0, low - 2)
            central = max(low, central - 2)
            high = max(central, high - 1)
            rationale.append("erosion evidence dominates support; CAP range shortened")
        elif independent_mechanisms >= 1 and has_outcome and support >= 3:
            strength = CapEvidenceStrength.STRONG
            low = min(prior.maximum_years, low + 1)
            central = min(prior.maximum_years, central + 2)
            high = min(prior.maximum_years, high + 3)
            rationale.append("multiple mechanisms plus realized corroboration support a longer CAP")
        elif support >= 1:
            strength = CapEvidenceStrength.CORROBORATED
            central = min(prior.maximum_years, central + 1)
            high = min(prior.maximum_years, high + 1)
            rationale.append("primary CAP evidence supports a bounded one-step extension")
        elif not support and not independent_mechanisms:
            strength = CapEvidenceStrength.UNSUPPORTED
            rationale.append("no company-specific support; retain a wide base-rate range")

        # Historical persistence is corroboration, not a forecast-year floor.
        if profile.observed_persistence_years:
            observed = max(profile.observed_persistence_years)
            rationale.append(f"longest observed historical persistence={observed} years")
            if observed >= prior.central_years and strength == CapEvidenceStrength.CORROBORATED:
                high = min(prior.maximum_years, high + 1)

        if profile.range_widener_evidence_ids:
            low = max(0, low - 1)
            high = min(prior.maximum_years, high + 1)
            rationale.append("low-reliability/forward evidence widens, but does not shift, the range")

        central = min(max(central, low), high)
        return CapAssessment(
            reference_class=prior.reference_class,
            low_years=low,
            central_years=central,
            high_years=high,
            strength=strength,
            supporting_evidence_ids=sorted(set(profile.cap_support_evidence_ids)),
            erosion_evidence_ids=sorted(set(profile.cap_erosion_evidence_ids)),
            rationale=rationale,
        )


def cap_range_width(assessment: CapAssessment) -> Decimal:
    return Decimal(assessment.high_years - assessment.low_years)
