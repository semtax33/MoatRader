from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel, SourceType, StatementType
from moatrader.canonical.ids import stable_id
from moatrader.evidence.models import (
    AtomicMoatRole,
    DcfLink,
    EvidenceCard,
    EvidenceDirection,
    EvidenceType,
    ForwardDriverType,
)
from moatrader.semantic.chunker import SemanticChunk


VALUATION_DRIVER_SCHEMA_VERSION = "valuation-driver-evidence/1"


class ValuationDriver(StrEnum):
    """Small, intentionally non-overlapping economic assumption ontology."""

    REVENUE_GROWTH = "REVENUE_GROWTH"
    TARGET_MARGIN = "TARGET_MARGIN"
    REINVESTMENT_EFFICIENCY = "REINVESTMENT_EFFICIENCY"
    ROIIC = "ROIIC"
    CAP_FADE = "CAP_FADE"
    RISK = "RISK"


class ValuationEvidenceRole(StrEnum):
    SUPPORT = "SUPPORT"
    COUNTER = "COUNTER"
    RANGE_WIDENER = "RANGE_WIDENER"
    SCENARIO_INPUT = "SCENARIO_INPUT"


class EvidenceApplicationPolicy(StrEnum):
    """Prevents one economic fact from being counted in several DCF levers."""

    PRIMARY_DRIVER_ONLY = "PRIMARY_DRIVER_ONLY"
    DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"


class ValuationDriverExtraction(ContractModel):
    """Minimal LLM classification for one valuation-selected atomic source unit."""

    relevant: bool = False
    primary_driver: ValuationDriver | None = None
    related_drivers: list[ValuationDriver] = Field(default_factory=list)
    role: ValuationEvidenceRole = ValuationEvidenceRole.SCENARIO_INPUT
    fact: str = "No valuation-driver evidence"
    economic_mechanism: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def valid_route(self) -> "ValuationDriverExtraction":
        if self.relevant and self.primary_driver is None:
            raise ValueError("relevant valuation extraction requires primary_driver")
        if not self.relevant and self.primary_driver is not None:
            raise ValueError("irrelevant valuation extraction cannot publish a driver")
        if self.primary_driver in self.related_drivers:
            raise ValueError("primary driver cannot be repeated as related")
        if len(self.related_drivers) != len(set(self.related_drivers)):
            raise ValueError("related drivers must be unique")
        return self


class ValuationDriverEvidence(ContractModel):
    evidence_id: str = Field(min_length=1)
    claim_id: str | None = None
    primary_driver: ValuationDriver
    related_drivers: list[ValuationDriver] = Field(default_factory=list)
    role: ValuationEvidenceRole
    application_policy: EvidenceApplicationPolicy = EvidenceApplicationPolicy.PRIMARY_DRIVER_ONLY
    fact: str = Field(min_length=1)
    observable_anchor: str = Field(min_length=1)
    economic_mechanism: list[str] = Field(default_factory=list)
    source_type: SourceType
    statement_type: StatementType
    evidence_type: EvidenceType
    moat_role: AtomicMoatRole
    reliability: float = Field(ge=0.0, le=1.0)
    source_chunk_id: str = Field(min_length=1)
    node_ids: list[str] = Field(min_length=1)
    period: str | None = None
    persistence_years_observed: int | None = Field(default=None, ge=0, le=200)
    range_widening_required: bool = False
    numeric_adjustment_allowed: bool = False
    exclusive_application_key: str = Field(min_length=1)

    @model_validator(mode="after")
    def no_duplicate_or_primary_related_driver(self) -> "ValuationDriverEvidence":
        if self.primary_driver in self.related_drivers:
            raise ValueError("primary_driver must not be repeated in related_drivers")
        if len(set(self.related_drivers)) != len(self.related_drivers):
            raise ValueError("related_drivers must be unique")
        if self.numeric_adjustment_allowed:
            raise ValueError(
                "v1 evidence records may validate assumptions but must not apply numeric DCF bumps"
            )
        return self


class ValuationDriverEvidenceBundle(ContractModel):
    schema_version: str = VALUATION_DRIVER_SCHEMA_VERSION
    issuer_id: str = Field(min_length=1)
    as_of: datetime
    evidence: list[ValuationDriverEvidence] = Field(default_factory=list)
    unmapped_evidence_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def pit_cutoff_is_timezone_aware(self) -> "ValuationDriverEvidenceBundle":
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        ids = [item.evidence_id for item in self.evidence]
        if len(ids) != len(set(ids)):
            raise ValueError("each atomic evidence card may create only one primary valuation record")
        application_keys = [item.exclusive_application_key for item in self.evidence]
        if len(application_keys) != len(set(application_keys)):
            raise ValueError("one atomic fact cannot be applied to multiple primary DCF drivers")
        return self

    def by_driver(self) -> dict[ValuationDriver, list[ValuationDriverEvidence]]:
        grouped = {driver: [] for driver in ValuationDriver}
        for item in self.evidence:
            grouped[item.primary_driver].append(item)
        return grouped


_PERSISTENCE_RE = re.compile(
    r"(?P<years>\d{1,3})\s*(?:개?년|years?)\s*(?:연속|동안|이상|간)?",
    re.IGNORECASE,
)


_EVIDENCE_DRIVER_MAP: dict[EvidenceType, tuple[ValuationDriver, tuple[ValuationDriver, ...]]] = {
    EvidenceType.SWITCHING_COST: (ValuationDriver.CAP_FADE, (ValuationDriver.REVENUE_GROWTH,)),
    EvidenceType.NETWORK_EFFECT: (ValuationDriver.CAP_FADE, (ValuationDriver.REVENUE_GROWTH,)),
    EvidenceType.COST_ADVANTAGE: (
        ValuationDriver.TARGET_MARGIN,
        (ValuationDriver.ROIIC, ValuationDriver.CAP_FADE),
    ),
    EvidenceType.INTANGIBLE_ASSET: (ValuationDriver.CAP_FADE, (ValuationDriver.TARGET_MARGIN,)),
    EvidenceType.SCALE_ADVANTAGE: (
        ValuationDriver.REINVESTMENT_EFFICIENCY,
        (ValuationDriver.TARGET_MARGIN, ValuationDriver.CAP_FADE),
    ),
    EvidenceType.REGULATORY_BARRIER: (ValuationDriver.CAP_FADE, (ValuationDriver.RISK,)),
    EvidenceType.PRICING_POWER: (ValuationDriver.TARGET_MARGIN, (ValuationDriver.CAP_FADE,)),
    EvidenceType.CUSTOMER_RETENTION: (ValuationDriver.CAP_FADE, (ValuationDriver.REVENUE_GROWTH,)),
    EvidenceType.MARKET_SHARE: (ValuationDriver.CAP_FADE, (ValuationDriver.REVENUE_GROWTH,)),
    EvidenceType.MARGIN_STABILITY: (ValuationDriver.TARGET_MARGIN, (ValuationDriver.CAP_FADE,)),
    EvidenceType.ROIC_QUALITY: (ValuationDriver.ROIIC, (ValuationDriver.CAP_FADE,)),
    EvidenceType.FCF_QUALITY: (
        ValuationDriver.REINVESTMENT_EFFICIENCY,
        (ValuationDriver.ROIIC,),
    ),
    EvidenceType.COMPETITIVE_THREAT: (ValuationDriver.CAP_FADE, (ValuationDriver.RISK,)),
    EvidenceType.CUSTOMER_CONCENTRATION: (ValuationDriver.RISK, (ValuationDriver.REVENUE_GROWTH,)),
    EvidenceType.SUBSTITUTION_RISK: (ValuationDriver.CAP_FADE, (ValuationDriver.RISK,)),
    EvidenceType.TECHNOLOGY_RISK: (ValuationDriver.RISK, (ValuationDriver.CAP_FADE,)),
    EvidenceType.CAPITAL_INTENSITY: (
        ValuationDriver.REINVESTMENT_EFFICIENCY,
        (ValuationDriver.ROIIC,),
    ),
    EvidenceType.MARKET_DEMAND: (ValuationDriver.REVENUE_GROWTH, ()),
    EvidenceType.CATEGORY_RECURRING_DEMAND: (ValuationDriver.REVENUE_GROWTH, (ValuationDriver.CAP_FADE,)),
    EvidenceType.CAPACITY_UTILIZATION: (
        ValuationDriver.REVENUE_GROWTH,
        (ValuationDriver.REINVESTMENT_EFFICIENCY,),
    ),
    EvidenceType.EXPORT_MIX: (ValuationDriver.REVENUE_GROWTH, (ValuationDriver.TARGET_MARGIN,)),
    EvidenceType.OPERATING_DRIVER: (ValuationDriver.REVENUE_GROWTH, ()),
}


_FORWARD_DRIVER_MAP: dict[ForwardDriverType, tuple[ValuationDriver, tuple[ValuationDriver, ...]]] = {
    ForwardDriverType.VOLUME: (ValuationDriver.REVENUE_GROWTH, ()),
    ForwardDriverType.ASP: (ValuationDriver.REVENUE_GROWTH, (ValuationDriver.TARGET_MARGIN,)),
    ForwardDriverType.CAPACITY: (
        ValuationDriver.REINVESTMENT_EFFICIENCY,
        (ValuationDriver.REVENUE_GROWTH,),
    ),
    ForwardDriverType.UTILIZATION: (
        ValuationDriver.REVENUE_GROWTH,
        (ValuationDriver.TARGET_MARGIN,),
    ),
    ForwardDriverType.PRODUCT_MIX: (
        ValuationDriver.TARGET_MARGIN,
        (ValuationDriver.REVENUE_GROWTH,),
    ),
    ForwardDriverType.EXPORT_MIX: (
        ValuationDriver.REVENUE_GROWTH,
        (ValuationDriver.TARGET_MARGIN,),
    ),
    ForwardDriverType.MARKET_GROWTH: (ValuationDriver.REVENUE_GROWTH, ()),
    ForwardDriverType.MARGIN: (ValuationDriver.TARGET_MARGIN, ()),
    ForwardDriverType.CAPEX: (ValuationDriver.REINVESTMENT_EFFICIENCY, (ValuationDriver.ROIIC,)),
    ForwardDriverType.WORKING_CAPITAL: (ValuationDriver.REINVESTMENT_EFFICIENCY, ()),
    ForwardDriverType.RAW_MATERIAL_COST: (ValuationDriver.TARGET_MARGIN, (ValuationDriver.RISK,)),
}


_DCF_LINK_MAP: dict[DcfLink, ValuationDriver] = {
    DcfLink.REVENUE: ValuationDriver.REVENUE_GROWTH,
    DcfLink.EBIT_MARGIN: ValuationDriver.TARGET_MARGIN,
    DcfLink.CAPEX: ValuationDriver.REINVESTMENT_EFFICIENCY,
    DcfLink.DEPRECIATION: ValuationDriver.REINVESTMENT_EFFICIENCY,
    DcfLink.NWC: ValuationDriver.REINVESTMENT_EFFICIENCY,
    DcfLink.WACC: ValuationDriver.RISK,
    # Terminal growth is not a moat sink. Durability evidence belongs in CAP/fade.
    DcfLink.TERMINAL_GROWTH: ValuationDriver.CAP_FADE,
}


class ValuationDriverMapper:
    """Deterministically re-routes atomic evidence without assigning DCF numbers."""

    def map_cards(
        self,
        *,
        issuer_id: str,
        as_of: datetime,
        cards: list[EvidenceCard],
    ) -> ValuationDriverEvidenceBundle:
        mapped: list[ValuationDriverEvidence] = []
        unmapped: list[str] = []
        for card in cards:
            route = self._route(card)
            if route is None:
                unmapped.append(card.evidence_id)
                continue
            primary, related = route
            role = self._role(card)
            persistence = self._persistence_years(card)
            anchor = (card.raw_quote or card.fact).strip()
            mapped.append(
                ValuationDriverEvidence(
                    evidence_id=card.evidence_id,
                    claim_id=card.claim_id,
                    primary_driver=primary,
                    related_drivers=list(related),
                    role=role,
                    fact=card.fact,
                    observable_anchor=anchor,
                    economic_mechanism=list(card.mechanism),
                    source_type=card.source_type,
                    statement_type=card.statement_type,
                    evidence_type=card.evidence_type,
                    moat_role=card.moat_role,
                    reliability=card.reliability,
                    source_chunk_id=card.source_chunk_id,
                    node_ids=list(card.node_ids),
                    period=card.period,
                    persistence_years_observed=persistence,
                    range_widening_required=(
                        role == ValuationEvidenceRole.RANGE_WIDENER
                        or card.reliability < 0.65
                        or card.statement_type
                        in {StatementType.MANAGEMENT_CLAIM, StatementType.FORECAST}
                    ),
                    exclusive_application_key=card.evidence_id,
                )
            )
        return ValuationDriverEvidenceBundle(
            issuer_id=issuer_id,
            as_of=as_of,
            evidence=mapped,
            unmapped_evidence_ids=unmapped,
        )

    def map_atomic_extractions(
        self,
        *,
        issuer_id: str,
        as_of: datetime,
        extractions: list[tuple[ValuationDriverExtraction, SemanticChunk]],
    ) -> ValuationDriverEvidenceBundle:
        evidence: list[ValuationDriverEvidence] = []
        unmapped: list[str] = []
        for extraction, chunk in extractions:
            atomic_key = str(chunk.metadata.get("atomic_evidence_key") or chunk.chunk_id)
            evidence_id = stable_id("VDE", chunk.document_id, atomic_key)
            if not extraction.relevant or extraction.primary_driver is None:
                unmapped.append(evidence_id)
                continue
            source_type = (
                chunk.source_refs[0].source_type if chunk.source_refs else SourceType.OTHER
            )
            statement_type = self._statement_type_from_source(chunk.markdown, source_type)
            evidence.append(
                ValuationDriverEvidence(
                    evidence_id=evidence_id,
                    primary_driver=extraction.primary_driver,
                    related_drivers=list(extraction.related_drivers),
                    role=extraction.role,
                    fact=extraction.fact,
                    # Python preserves the complete source unit as the authority;
                    # an LLM compression is never the provenance anchor.
                    observable_anchor=chunk.markdown,
                    economic_mechanism=list(extraction.economic_mechanism),
                    source_type=source_type,
                    statement_type=statement_type,
                    evidence_type=EvidenceType.OTHER,
                    moat_role=AtomicMoatRole.NONE,
                    reliability=self._source_reliability(statement_type),
                    source_chunk_id=chunk.chunk_id,
                    node_ids=list(chunk.node_ids),
                    period=None,
                    persistence_years_observed=self._persistence_years_text(chunk.markdown),
                    range_widening_required=(
                        extraction.role == ValuationEvidenceRole.RANGE_WIDENER
                        or statement_type
                        in {StatementType.MANAGEMENT_CLAIM, StatementType.FORECAST}
                    ),
                    exclusive_application_key=atomic_key,
                )
            )
        return ValuationDriverEvidenceBundle(
            issuer_id=issuer_id,
            as_of=as_of,
            evidence=evidence,
            unmapped_evidence_ids=unmapped,
        )

    @staticmethod
    def merge_bundles(
        primary: ValuationDriverEvidenceBundle,
        supplemental: ValuationDriverEvidenceBundle,
    ) -> ValuationDriverEvidenceBundle:
        if primary.issuer_id != supplemental.issuer_id or primary.as_of != supplemental.as_of:
            raise ValueError("valuation evidence bundles must share issuer and PIT cutoff")
        # Existing source-grounded MOAT/outcome cards are authoritative for the
        # same atomic unit. The valuation-only lane fills previously discarded
        # NONE facts, not overwrite frozen sensor decisions.
        by_chunk = {item.source_chunk_id: item for item in primary.evidence}
        for item in supplemental.evidence:
            by_chunk.setdefault(item.source_chunk_id, item)
        return ValuationDriverEvidenceBundle(
            issuer_id=primary.issuer_id,
            as_of=primary.as_of,
            evidence=sorted(by_chunk.values(), key=lambda item: item.evidence_id),
            unmapped_evidence_ids=sorted(
                set(primary.unmapped_evidence_ids) | set(supplemental.unmapped_evidence_ids)
            ),
        )

    @staticmethod
    def _route(
        card: EvidenceCard,
    ) -> tuple[ValuationDriver, tuple[ValuationDriver, ...]] | None:
        # Explicit forward facts take precedence for MOAT_NONE evidence. This is
        # what preserves pipeline, capacity, mix, and reinvestment information.
        if card.moat_role == AtomicMoatRole.NONE and card.forward_driver_type is not None:
            return _FORWARD_DRIVER_MAP.get(card.forward_driver_type)
        mapped = _EVIDENCE_DRIVER_MAP.get(card.evidence_type)
        if mapped is not None:
            return mapped
        if card.forward_driver_type is not None:
            mapped = _FORWARD_DRIVER_MAP.get(card.forward_driver_type)
            if mapped is not None:
                return mapped
        drivers: list[ValuationDriver] = []
        for link in card.dcf_links:
            driver = _DCF_LINK_MAP.get(link)
            if driver is not None and driver not in drivers:
                drivers.append(driver)
        if not drivers:
            return None
        return drivers[0], tuple(drivers[1:])

    @staticmethod
    def _role(card: EvidenceCard) -> ValuationEvidenceRole:
        if card.direction == EvidenceDirection.MOAT_NEGATIVE or card.moat_role == AtomicMoatRole.COUNTER:
            return ValuationEvidenceRole.COUNTER
        if (
            card.direction == EvidenceDirection.MOAT_POSITIVE
            and card.statement_type
            not in {StatementType.MANAGEMENT_CLAIM, StatementType.FORECAST}
            and card.reliability >= 0.50
        ):
            # Legacy evidence checkpoints may predate the explicit moat_role
            # field. A source-grounded positive disclosed fact still supports
            # its exclusively mapped valuation driver; claims/forecasts do not.
            return ValuationEvidenceRole.SUPPORT
        if card.moat_role == AtomicMoatRole.NONE or card.statement_type in {
            StatementType.MANAGEMENT_CLAIM,
            StatementType.FORECAST,
        }:
            return ValuationEvidenceRole.SCENARIO_INPUT
        if card.reliability < 0.50:
            return ValuationEvidenceRole.RANGE_WIDENER
        return ValuationEvidenceRole.SUPPORT

    @staticmethod
    def _persistence_years(card: EvidenceCard) -> int | None:
        match = _PERSISTENCE_RE.search(" ".join((card.fact, card.raw_quote or "")))
        years = int(match.group("years")) if match else None
        return years if years is not None and years <= 200 else None

    @staticmethod
    def _persistence_years_text(text: str) -> int | None:
        match = _PERSISTENCE_RE.search(text)
        years = int(match.group("years")) if match else None
        return years if years is not None and years <= 200 else None

    @staticmethod
    def _statement_type_from_source(text: str, source_type: SourceType) -> StatementType:
        if re.search(r"전망|계획|예상|기대|목표|추정|will|expect|plan|target|forecast", text, re.I):
            return (
                StatementType.MANAGEMENT_CLAIM
                if source_type in {SourceType.DART, SourceType.SEC_EDGAR, SourceType.IR}
                else StatementType.FORECAST
            )
        if source_type == SourceType.ANALYST:
            return StatementType.ANALYST_INTERPRETATION
        if source_type == SourceType.INDUSTRY:
            return StatementType.INDUSTRY_INTERPRETATION
        return StatementType.DISCLOSED_FACT

    @staticmethod
    def _source_reliability(statement_type: StatementType) -> float:
        return {
            StatementType.DISCLOSED_FACT: 0.95,
            StatementType.DERIVED_METRIC: 0.90,
            StatementType.ANALYST_INTERPRETATION: 0.75,
            StatementType.INDUSTRY_INTERPRETATION: 0.75,
            StatementType.MANAGEMENT_CLAIM: 0.65,
            StatementType.FORECAST: 0.60,
        }[statement_type]


def build_valuation_driver_consensus(
    votes: list[ValuationDriverExtraction],
) -> tuple[ValuationDriverExtraction, dict[str, object]]:
    """Strict-majority route consensus; disagreement fails closed."""

    if not votes:
        return ValuationDriverExtraction(), {"status": "NO_VOTES", "vote_count": 0}
    signatures = [
        (
            vote.relevant,
            vote.primary_driver.value if vote.primary_driver else None,
            vote.role.value,
        )
        for vote in votes
    ]
    counts = Counter(signatures)
    signature, count = min(
        counts.items(),
        key=lambda item: (-item[1], str(item[0])),
    )
    required = len(votes) // 2 + 1
    diagnostics: dict[str, object] = {
        "vote_count": len(votes),
        "required_majority": required,
        "route_counts": {str(key): value for key, value in sorted(counts.items(), key=lambda item: str(item[0]))},
    }
    if count < required:
        diagnostics["status"] = "NO_STRICT_MAJORITY"
        return ValuationDriverExtraction(), diagnostics
    matching = [vote for vote, item in zip(votes, signatures, strict=True) if item == signature]
    canonical = min(matching, key=lambda item: item.model_dump_json())
    related_counts = Counter(driver for vote in matching for driver in vote.related_drivers)
    related = sorted(
        (
            driver
            for driver, driver_count in related_counts.items()
            if driver_count >= len(matching) // 2 + 1 and driver != canonical.primary_driver
        ),
        key=lambda item: item.value,
    )[:2]
    diagnostics["status"] = "STRICT_MAJORITY"
    diagnostics["winning_count"] = count
    return canonical.model_copy(update={"related_drivers": related}), diagnostics
