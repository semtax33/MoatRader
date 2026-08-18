from __future__ import annotations

import hashlib
import re
from datetime import datetime
from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.business.drivers import (
    EvidenceApplicationPolicy,
    ValuationDriver,
    ValuationDriverEvidence,
    ValuationDriverEvidenceBundle,
    ValuationEvidenceRole,
)
from moatrader.canonical.models import ContractModel, SourceType, StatementType
from moatrader.evidence.models import AtomicMoatRole, EvidenceType


_PRICE_OR_OPINION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\btarget\s+price\b",
        r"\bcurrent\s+(?:share\s+)?price\b",
        r"\bmarket\s+price\b",
        r"\b(?:strong\s+)?(?:buy|sell)\b",
        r"\bhold\s+rating\b",
        r"\binvestment\s+recommendation\b",
        r"\bupside\s+potential\b",
        r"목표\s*주가",
        r"목표가",
        r"현재\s*주가",
        r"투자\s*의견",
        r"매수\s*의견",
        r"매도\s*의견",
        r"상승\s*여력",
    )
)


def _aware(value: datetime, *, field: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")


def _contains_price_or_opinion(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PRICE_OR_OPINION_PATTERNS)


class ReportEvidenceItem(ContractModel):
    item_id: str = Field(min_length=1)
    primary_driver: ValuationDriver
    fact: str = Field(min_length=1)
    observable_anchor: str = Field(min_length=1)
    source_document_id: str = Field(min_length=1)
    source_chunk_id: str = Field(min_length=1)
    node_ids: list[str] = Field(min_length=1)
    available_at: datetime
    statement_type: StatementType
    role: ValuationEvidenceRole = ValuationEvidenceRole.SCENARIO_INPUT
    value: Decimal | str | None = None
    unit: str | None = None
    period: str | None = None
    reliability: float = Field(default=0.70, ge=0, le=1)

    @model_validator(mode="after")
    def source_and_role_are_safe(self) -> "ReportEvidenceItem":
        _aware(self.available_at, field="available_at")
        if self.role not in {
            ValuationEvidenceRole.SCENARIO_INPUT,
            ValuationEvidenceRole.RANGE_WIDENER,
        }:
            raise ValueError("analyst-report evidence may only shape scenarios or widen ranges")
        return self


class MarketOpinion(ContractModel):
    source_document_id: str = Field(min_length=1)
    available_at: datetime
    recommendation: str | None = None
    target_price: Decimal | None = Field(default=None, gt=0)
    current_price: Decimal | None = Field(default=None, gt=0)
    stated_upside: Decimal | None = None
    commentary: str | None = None

    @model_validator(mode="after")
    def opinion_is_explicit_and_pit(self) -> "MarketOpinion":
        _aware(self.available_at, field="available_at")
        if not any(
            value is not None
            for value in (
                self.recommendation,
                self.target_price,
                self.current_price,
                self.stated_upside,
                self.commentary,
            )
        ):
            raise ValueError("market opinion requires at least one opinion field")
        return self


class IntrinsicResearchBundle(ContractModel):
    issuer_id: str = Field(min_length=1)
    as_of: datetime
    observed_facts: list[ReportEvidenceItem] = Field(default_factory=list)
    peer_comparisons: list[ReportEvidenceItem] = Field(default_factory=list)
    analyst_estimates: list[ReportEvidenceItem] = Field(default_factory=list)
    interpretation: list[ReportEvidenceItem] = Field(default_factory=list)
    price_leakage_detected: bool = False

    @model_validator(mode="after")
    def intrinsic_lane_is_pit_and_price_blind(self) -> "IntrinsicResearchBundle":
        _aware(self.as_of, field="as_of")
        if self.price_leakage_detected:
            raise ValueError("intrinsic research bundle cannot contain price leakage")
        for item in self.items():
            if item.available_at > self.as_of:
                raise ValueError(f"future analyst evidence is not PIT eligible: {item.item_id}")
            searchable = " ".join(
                value
                for value in (
                    item.fact,
                    item.observable_anchor,
                    item.unit or "",
                    str(item.value) if item.value is not None else "",
                )
                if value
            )
            if _contains_price_or_opinion(searchable):
                raise ValueError(f"price/opinion leakage in intrinsic lane: {item.item_id}")
        return self

    def items(self) -> list[ReportEvidenceItem]:
        return [
            *self.observed_facts,
            *self.peer_comparisons,
            *self.analyst_estimates,
            *self.interpretation,
        ]

    def to_valuation_driver_bundle(self) -> ValuationDriverEvidenceBundle:
        evidence: list[ValuationDriverEvidence] = []
        for item in self.items():
            digest = hashlib.sha256(
                f"{item.source_document_id}|{item.item_id}".encode("utf-8")
            ).hexdigest()[:24]
            evidence.append(
                ValuationDriverEvidence(
                    evidence_id=f"ANALYST-{digest}",
                    primary_driver=item.primary_driver,
                    role=item.role,
                    application_policy=EvidenceApplicationPolicy.PRIMARY_DRIVER_ONLY,
                    fact=item.fact,
                    observable_anchor=item.observable_anchor,
                    source_type=SourceType.ANALYST,
                    statement_type=item.statement_type,
                    evidence_type=EvidenceType.OTHER,
                    moat_role=AtomicMoatRole.NONE,
                    reliability=item.reliability,
                    source_chunk_id=item.source_chunk_id,
                    node_ids=item.node_ids,
                    period=item.period,
                    range_widening_required=(
                        item.role == ValuationEvidenceRole.RANGE_WIDENER
                        or item.reliability < 0.65
                        or item.statement_type == StatementType.FORECAST
                    ),
                    numeric_adjustment_allowed=False,
                    exclusive_application_key=f"ANALYST:{item.item_id}",
                )
            )
        return ValuationDriverEvidenceBundle(
            issuer_id=self.issuer_id,
            as_of=self.as_of,
            evidence=evidence,
        )


class ResearchReportBundle(ContractModel):
    """Analyst-report lanes; market opinion is quarantined from intrinsic inputs."""

    issuer_id: str = Field(min_length=1)
    observed_facts: list[ReportEvidenceItem] = Field(default_factory=list)
    peer_comparisons: list[ReportEvidenceItem] = Field(default_factory=list)
    analyst_estimates: list[ReportEvidenceItem] = Field(default_factory=list)
    interpretation: list[ReportEvidenceItem] = Field(default_factory=list)
    market_opinion: list[MarketOpinion] = Field(default_factory=list)

    def intrinsic_view(self, *, as_of: datetime) -> IntrinsicResearchBundle:
        return IntrinsicResearchBundle(
            issuer_id=self.issuer_id,
            as_of=as_of,
            observed_facts=self.observed_facts,
            peer_comparisons=self.peer_comparisons,
            analyst_estimates=self.analyst_estimates,
            interpretation=self.interpretation,
            price_leakage_detected=False,
        )

    def to_valuation_driver_bundle(self, *, as_of: datetime) -> ValuationDriverEvidenceBundle:
        return self.intrinsic_view(as_of=as_of).to_valuation_driver_bundle()
