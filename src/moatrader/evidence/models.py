from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import ConfigDict, Field, field_validator, model_validator

from moatrader.canonical.models import ContractModel, SourceType, StatementType


class EvidenceType(StrEnum):
    SWITCHING_COST = "SWITCHING_COST"
    NETWORK_EFFECT = "NETWORK_EFFECT"
    COST_ADVANTAGE = "COST_ADVANTAGE"
    INTANGIBLE_ASSET = "INTANGIBLE_ASSET"
    SCALE_ADVANTAGE = "SCALE_ADVANTAGE"
    REGULATORY_BARRIER = "REGULATORY_BARRIER"
    PRICING_POWER = "PRICING_POWER"
    CUSTOMER_RETENTION = "CUSTOMER_RETENTION"
    MARKET_SHARE = "MARKET_SHARE"
    MARGIN_STABILITY = "MARGIN_STABILITY"
    COMPETITIVE_THREAT = "COMPETITIVE_THREAT"
    CUSTOMER_CONCENTRATION = "CUSTOMER_CONCENTRATION"
    SUBSTITUTION_RISK = "SUBSTITUTION_RISK"
    TECHNOLOGY_RISK = "TECHNOLOGY_RISK"
    CAPITAL_INTENSITY = "CAPITAL_INTENSITY"
    ROIC_QUALITY = "ROIC_QUALITY"
    FCF_QUALITY = "FCF_QUALITY"
    MARKET_DEMAND = "MARKET_DEMAND"
    CATEGORY_RECURRING_DEMAND = "CATEGORY_RECURRING_DEMAND"
    CAPACITY_UTILIZATION = "CAPACITY_UTILIZATION"
    EXPORT_MIX = "EXPORT_MIX"
    OPERATING_DRIVER = "OPERATING_DRIVER"
    OTHER = "OTHER"


# Only these categories describe a causal, company-specific barrier that can
# be scored as an economic-moat mechanism.  The remaining EvidenceType values
# are outcomes, context, operating drivers, or risks and may corroborate or
# weaken a moat, but must never be promoted to a mechanism by themselves.
STRUCTURAL_MOAT_TYPES: frozenset[EvidenceType] = frozenset(
    {
        EvidenceType.SWITCHING_COST,
        EvidenceType.NETWORK_EFFECT,
        EvidenceType.COST_ADVANTAGE,
        EvidenceType.INTANGIBLE_ASSET,
        EvidenceType.SCALE_ADVANTAGE,
        EvidenceType.REGULATORY_BARRIER,
    }
)

OUTCOME_CORROBORATION_TYPES: frozenset[EvidenceType] = frozenset(
    {
        EvidenceType.PRICING_POWER,
        EvidenceType.CUSTOMER_RETENTION,
        EvidenceType.MARKET_SHARE,
        EvidenceType.MARGIN_STABILITY,
        EvidenceType.ROIC_QUALITY,
        EvidenceType.FCF_QUALITY,
    }
)


class EvidenceDirection(StrEnum):
    MOAT_POSITIVE = "MOAT_POSITIVE"
    MOAT_NEGATIVE = "MOAT_NEGATIVE"
    NEUTRAL = "NEUTRAL"


class EconomicScope(StrEnum):
    COMPANY = "COMPANY"
    SEGMENT = "SEGMENT"
    PRODUCT_CATEGORY = "PRODUCT_CATEGORY"
    INDUSTRY = "INDUSTRY"
    MACRO = "MACRO"


class ForwardDriverType(StrEnum):
    VOLUME = "VOLUME"
    ASP = "ASP"
    CAPACITY = "CAPACITY"
    UTILIZATION = "UTILIZATION"
    PRODUCT_MIX = "PRODUCT_MIX"
    EXPORT_MIX = "EXPORT_MIX"
    MARKET_GROWTH = "MARKET_GROWTH"
    MARGIN = "MARGIN"
    CAPEX = "CAPEX"
    WORKING_CAPITAL = "WORKING_CAPITAL"
    RAW_MATERIAL_COST = "RAW_MATERIAL_COST"


class DcfLink(StrEnum):
    REVENUE = "REVENUE"
    EBIT_MARGIN = "EBIT_MARGIN"
    CAPEX = "CAPEX"
    DEPRECIATION = "DEPRECIATION"
    NWC = "NWC"
    WACC = "WACC"
    TERMINAL_GROWTH = "TERMINAL_GROWTH"


class EvidenceMetric(ContractModel):
    # JSON repair can preserve a malformed presentation fragment as an extra
    # key. Only the canonical name/value/unit fields are material.
    model_config = ConfigDict(extra="ignore")

    name: str
    value: Decimal | str
    unit: str | None = None

    @field_validator("value", mode="before")
    @classmethod
    def unwrap_scalar_value(cls, value: object) -> object:
        """Recover provider wrappers while keeping metric values scalar."""
        candidate = value
        for _ in range(3):
            if isinstance(candidate, list):
                if len(candidate) != 1:
                    return value
                candidate = candidate[0]
                continue
            if isinstance(candidate, dict):
                nested = next(
                    (
                        candidate[key]
                        for key in ("value", "decimal", "number", "amount")
                        if key in candidate
                    ),
                    None,
                )
                if nested is None:
                    return value
                candidate = nested
                continue
            break
        return candidate


class CanonicalClaimSignature(ContractModel):
    """Semantic slots used for deterministic claim-level de-duplication.

    The LLM may label these slots, but Python overwrites the type/direction and
    canonicalizes every string before hashing.  The resulting claim_id is the
    scoring identity; evidence_id remains the source-span audit identity.
    """

    # The API-facing aliases keep the Structured Outputs schema compact while
    # populate_by_name preserves the expanded internal/checkpoint contract.
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    moat_source: EvidenceType = Field(default=EvidenceType.OTHER, alias="type")
    subject: str = Field(default="company", min_length=1)
    predicate: str = Field(default="unspecified", min_length=1)
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    horizon: str = "UNSPECIFIED"
    metric: str | None = None
    value_bucket: str | None = Field(default=None, alias="bucket")


class AtomicEvidenceExtraction(ContractModel):
    """Minimal sufficient LLM output for one atomic source unit.

    Numeric parsing, statement provenance, forward-driver routing, DCF links,
    strength and final claim identity are deterministic Python responsibilities.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    is_investment_relevant: bool = Field(default=False, alias="relevant")
    evidence_type: EvidenceType = Field(default=EvidenceType.OTHER, alias="type")
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    fact: str = "No investment-relevant evidence"
    mechanism: list[str] = Field(default_factory=list)
    economic_scope: EconomicScope = Field(default=EconomicScope.COMPANY, alias="scope")
    segment: str | None = None
    claim_subject: str = Field(default="company", alias="subject")
    claim_predicate: str = Field(default="unspecified", alias="predicate")
    claim_horizon: str | None = Field(default=None, alias="horizon")
    claim_metric: str | None = Field(default=None, alias="metric")


class AtomicEvidenceJudgment(ContractModel):
    """Classification of one deterministic, source-grounded atomic unit."""

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    is_investment_relevant: bool = Field(default=False, alias="relevant")
    evidence_type: EvidenceType = Field(default=EvidenceType.OTHER, alias="type")
    statement_type: StatementType = Field(default=StatementType.MANAGEMENT_CLAIM, alias="stmt")
    fact: str = "No investment-relevant evidence"
    mechanism: list[str] = Field(default_factory=list)
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    economic_scope: EconomicScope = Field(default=EconomicScope.COMPANY, alias="scope")
    segment: str | None = None
    metrics: list[EvidenceMetric] = Field(default_factory=list)
    unit: str | None = None
    period: str | None = None
    forward_driver_type: ForwardDriverType | None = Field(default=None, alias="driver")
    dcf_links: list[DcfLink] = Field(default_factory=list, alias="dcf")
    forecast_horizon: str | None = Field(default=None, alias="horizon")
    claim_signature: CanonicalClaimSignature | None = Field(default=None, alias="claim")


class EvidenceCard(ContractModel):
    # Provider models occasionally add harmless presentation-only fields such
    # as period_unit. Core grounding fields remain required and are validated
    # independently against the canonical chunk.
    model_config = ConfigDict(extra="ignore")

    evidence_id: str
    source_chunk_id: str
    node_ids: list[str] = Field(min_length=1)
    evidence_type: EvidenceType = EvidenceType.OTHER
    statement_type: StatementType = StatementType.MANAGEMENT_CLAIM
    fact: str = Field(min_length=1)
    mechanism: list[str] = Field(default_factory=list)
    direction: EvidenceDirection = EvidenceDirection.NEUTRAL
    strength: float = Field(default=0.5, ge=0.0, le=1.0)
    source_type: SourceType = SourceType.OTHER
    company_scope: str = "COMPANY"
    economic_scope: EconomicScope = EconomicScope.COMPANY
    segment: str | None = None
    metrics: list[EvidenceMetric] = Field(default_factory=list)
    unit: str | None = None
    period: str | None = None
    raw_quote: str | None = None
    reliability: float = Field(default=0.5, ge=0.0, le=1.0)
    forward_driver_type: ForwardDriverType | None = None
    dcf_links: list[DcfLink] = Field(default_factory=list)
    forecast_horizon: str | None = None
    atomic_evidence_key: str | None = None
    claim_signature: CanonicalClaimSignature | None = None
    claim_id: str | None = None

    @field_validator("metrics", mode="before")
    @classmethod
    def null_metrics_are_empty(cls, value: object) -> object:
        if value is None or value == "" or value == {}:
            return []
        if isinstance(value, list):
            return [
                item
                for item in value
                if isinstance(item, EvidenceMetric)
                or (
                    isinstance(item, dict)
                    and item.get("name")
                    and item.get("value") is not None
                )
            ]
        return value

    @field_validator("period", mode="before")
    @classmethod
    def numeric_period_is_string(cls, value: object) -> object:
        if value is None or isinstance(value, str):
            return value
        if isinstance(value, (int, float, Decimal, date, datetime)):
            return str(value)
        return value

    @field_validator("statement_type", mode="before")
    @classmethod
    def normalize_statement_type_typo(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized in {"DISLOSED_FACT", "DISCLOSE_FACT", "DISCLOSEDFACT"}:
                return StatementType.DISCLOSED_FACT
            if normalized not in {item.value for item in StatementType}:
                return StatementType.MANAGEMENT_CLAIM
            return normalized
        return value

    @field_validator("evidence_type", mode="before")
    @classmethod
    def unknown_evidence_type_is_other(cls, value: object) -> object:
        if isinstance(value, list):
            if len(value) != 1:
                return EvidenceType.OTHER
            value = value[0]
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized not in {item.value for item in EvidenceType}:
                return EvidenceType.OTHER
            return normalized
        return value

    @field_validator("direction", mode="before")
    @classmethod
    def unknown_direction_is_neutral(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized not in {item.value for item in EvidenceDirection}:
                return EvidenceDirection.NEUTRAL
            return normalized
        return value

    @field_validator("source_type", mode="before")
    @classmethod
    def normalize_source_type(cls, value: object) -> object:
        if isinstance(value, str):
            normalized = value.strip().upper().replace(" ", "_")
            if normalized in {"SEC", "EDGAR"}:
                return SourceType.SEC_EDGAR
            if normalized not in {item.value for item in SourceType}:
                return SourceType.OTHER
            return normalized
        return SourceType.OTHER if value is None else value

    @field_validator("mechanism", mode="before")
    @classmethod
    def null_mechanism_is_empty(cls, value: object) -> object:
        if value is None or value == {} or value == "":
            return []
        if isinstance(value, list):
            return [
                item.get("text") if isinstance(item, dict) and isinstance(item.get("text"), str) else item
                for item in value
            ]
        return value

    @field_validator("strength", "reliability", mode="before")
    @classmethod
    def null_confidence_is_neutral(cls, value: object) -> object:
        if value is None or (isinstance(value, str) and not value.strip()):
            return 0.5
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return value
        if numeric < 0:
            return 0.0
        if numeric <= 1:
            return numeric
        if numeric <= 5:
            return numeric / 5
        if numeric <= 10:
            return numeric / 10
        if numeric <= 100:
            return numeric / 100
        return 1.0

    @field_validator("company_scope", mode="before")
    @classmethod
    def null_scope_is_company(cls, value: object) -> object:
        return "COMPANY" if value is None else value

    @field_validator("economic_scope", mode="before")
    @classmethod
    def normalize_economic_scope(cls, value: object) -> object:
        if value is None:
            return EconomicScope.COMPANY
        if isinstance(value, str):
            normalized = value.strip().upper().replace(" ", "_")
            if normalized not in {item.value for item in EconomicScope}:
                return EconomicScope.COMPANY
            return normalized
        return value

    @field_validator("forward_driver_type", mode="before")
    @classmethod
    def normalize_forward_driver_type(cls, value: object) -> object:
        if value is None or value == "":
            return None
        candidate = value
        for _ in range(3):
            if isinstance(candidate, list):
                if len(candidate) != 1:
                    return None
                candidate = candidate[0]
                continue
            if isinstance(candidate, dict):
                nested = next(
                    (
                        candidate[key]
                        for key in (
                            "forward_driver_type",
                            "ForwardDriverType",
                            "driver_type",
                            "value",
                        )
                        if key in candidate
                    ),
                    None,
                )
                if nested is None:
                    return None
                candidate = nested
                continue
            break
        if isinstance(candidate, str):
            normalized = candidate.strip().upper().replace(" ", "_")
            if normalized not in {item.value for item in ForwardDriverType}:
                return None
            return normalized
        return None

    @field_validator("dcf_links", mode="before")
    @classmethod
    def normalize_dcf_links(cls, value: object) -> object:
        if value is None or value == "":
            return []
        if isinstance(value, dict):
            value = next(
                (
                    value[key]
                    for key in ("dcf_links", "items", "values")
                    if isinstance(value.get(key), list)
                ),
                [],
            )
        if not isinstance(value, list):
            return []
        allowed = {item.value for item in DcfLink}
        normalized_links = []
        for item in value:
            if isinstance(item, dict):
                item = next(
                    (
                        item[key]
                        for key in ("dcf_link", "name", "value", "type")
                        if isinstance(item.get(key), str)
                    ),
                    None,
                )
            if not isinstance(item, str):
                continue
            normalized = item.strip().upper().replace(" ", "_")
            if normalized in allowed:
                normalized_links.append(normalized)
        return normalized_links


class ForwardDriverCard(ContractModel):
    driver_id: str
    source_evidence_id: str
    source_chunk_id: str
    node_ids: list[str] = Field(min_length=1)
    driver_type: ForwardDriverType
    evidence: str = Field(min_length=1)
    implication: list[str] = Field(default_factory=list)
    dcf_links: list[DcfLink] = Field(min_length=1)
    statement_type: StatementType
    economic_scope: EconomicScope
    segment: str | None = None
    period: str | None = None
    forecast_horizon: str | None = None
    reliability: float = Field(ge=0.0, le=1.0)


class EvidenceExtractionResult(ContractModel):
    model_config = ConfigDict(extra="ignore")

    chunk_id: str
    cards: list[EvidenceCard] = Field(default_factory=list)

    @model_validator(mode="after")
    def cards_belong_to_chunk(self) -> "EvidenceExtractionResult":
        if any(card.source_chunk_id != self.chunk_id for card in self.cards):
            raise ValueError("every evidence card must cite the result chunk_id")
        return self


class EvidenceBatchExtractionResult(ContractModel):
    """Evidence extracted from multiple canonical chunks in one bounded call."""

    model_config = ConfigDict(extra="ignore")

    cards: list[EvidenceCard] = Field(default_factory=list)


class EvidenceRelationType(StrEnum):
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    WEAKENS = "WEAKENS"
    UPDATES = "UPDATES"
    DUPLICATES = "DUPLICATES"


class EvidenceRelation(ContractModel):
    from_evidence_id: str
    to_evidence_id: str
    relation: EvidenceRelationType


class EvidenceCluster(ContractModel):
    canonical_evidence_id: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class ClaimCluster(ContractModel):
    claim_id: str
    canonical_evidence_id: str
    supporting_evidence_ids: list[str] = Field(default_factory=list)


class CitedSummaryClaim(ContractModel):
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)


class SectionSummary(ContractModel):
    section_path: list[str]
    positive_evidence_ids: list[str] = Field(default_factory=list)
    negative_evidence_ids: list[str] = Field(default_factory=list)
    key_mechanisms: list[CitedSummaryClaim] = Field(default_factory=list)
    key_kpis: list[CitedSummaryClaim] = Field(default_factory=list)
    uncertainties: list[CitedSummaryClaim] = Field(default_factory=list)


class CompanyDossier(ContractModel):
    issuer_id: str | None = None
    issuer_name: str
    ticker: str | None = None
    as_of: datetime
    source_document_ids: list[str] = Field(min_length=1)
    business_summary: str | None = None
    financial_summary: str | None = None
    evidence: list[EvidenceCard] = Field(default_factory=list)
    relations: list[EvidenceRelation] = Field(default_factory=list)
    section_summaries: list[SectionSummary] = Field(default_factory=list)
    key_table_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def dossier_references_are_internal(self) -> "CompanyDossier":
        ids = {card.evidence_id for card in self.evidence}
        if len(ids) != len(self.evidence):
            raise ValueError("evidence IDs must be unique")
        referenced: set[str] = set()
        for summary in self.section_summaries:
            referenced.update(summary.positive_evidence_ids)
            referenced.update(summary.negative_evidence_ids)
            for claim in [*summary.key_mechanisms, *summary.key_kpis, *summary.uncertainties]:
                referenced.update(claim.evidence_ids)
        for relation in self.relations:
            referenced.add(relation.from_evidence_id)
            referenced.add(relation.to_evidence_id)
        missing = referenced - ids
        if missing:
            raise ValueError(f"dossier references missing evidence: {sorted(missing)}")
        return self


class Durability(StrEnum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    MEDIUM_HIGH = "MEDIUM_HIGH"
    HIGH = "HIGH"


class MoatMechanismScore(ContractModel):
    evidence_type: EvidenceType
    score: float = Field(ge=0.0, le=10.0)
    evidence_ids: list[str] = Field(min_length=1)
    rationale: str


class CoverageMetrics(ContractModel):
    char_retention: float | None = Field(default=None, ge=0.0)
    token_retention: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_retention: float | None = Field(default=None, ge=0.0, le=1.0)
    section_retention: float | None = Field(default=None, ge=0.0, le=1.0)
    table_retention: float | None = Field(default=None, ge=0.0, le=1.0)
    numeric_retention: float | None = Field(default=None, ge=0.0, le=1.0)
    # This is the only coverage value used for MOAT eligibility/ranking.  The
    # fields above remain parser/context diagnostics and are not collapsed into
    # a single minimum that penalizes table-heavy filings.
    moat_evidence_coverage: float | None = Field(default=None, ge=0.0, le=1.0)


class MoatScore(ContractModel):
    issuer_id: str | None = None
    as_of: date
    economic_moat_score: float = Field(ge=0.0, le=10.0)
    mechanisms: list[MoatMechanismScore] = Field(default_factory=list)
    counterevidence_ids: list[str] = Field(default_factory=list)
    canonical_claim_ids: list[str] = Field(default_factory=list)
    durability: Durability
    model_confidence: float = Field(ge=0.0, le=1.0)
    document_coverage: CoverageMetrics
    caveats: list[str] = Field(default_factory=list)
    # Preserves the model's holistic proposal for audit.  The public economic
    # moat score is recomputed deterministically from validated mechanisms.
    llm_proposed_score: float | None = Field(default=None, ge=0.0, le=10.0)

    @model_validator(mode="after")
    def positive_score_has_evidence(self) -> "MoatScore":
        if self.economic_moat_score > 0 and not self.mechanisms:
            raise ValueError("a positive moat score requires cited mechanisms")
        return self
