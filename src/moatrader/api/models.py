from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


Tone = Literal["positive", "neutral", "negative", "warning"]
ConfidenceLevel = Literal["HIGH", "MEDIUM", "LOW"]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReportSummary(ApiModel):
    ticker: str
    issuer_name: str
    as_of: date
    status: str
    moat_score: float | None = None
    current_price: float | None = None
    currency: str = "KRW"


class ReportCatalog(ApiModel):
    schema_version: str
    reports: list[ReportSummary]


class ReportMeta(ApiModel):
    schema_version: str
    report_id: str
    generated_at: datetime
    as_of: datetime
    evidence_cutoff: datetime
    price_as_of: datetime
    data_grade: Literal["RESEARCH", "LIMITED", "INSUFFICIENT"]
    source_document_count: int = Field(ge=0)
    evidence_count: int = Field(ge=0)
    warnings: list[str] = Field(default_factory=list)


class Metric(ApiModel):
    id: str
    label: str
    value: float | None = None
    unit: str = ""
    period: str | None = None
    trend: Tone = "neutral"


class MixItem(ApiModel):
    name: str
    value: float
    share_pct: float | None = None
    unit: str = "KRW"
    period: str


class CompanyProfile(ApiModel):
    ticker: str
    issuer_id: str | None = None
    issuer_name: str
    currency: str = "KRW"
    business_summary: str
    business_model: str
    industry_label: str
    revenue_mix: list[MixItem] = Field(default_factory=list)
    geography: list[MixItem] = Field(default_factory=list)
    key_metrics: list[Metric] = Field(default_factory=list)


class IndustryForce(ApiModel):
    id: str
    label: str
    status: str
    tone: Tone
    description: str
    evidence_ids: list[str] = Field(default_factory=list)


class ValueLink(ApiModel):
    stage: str
    title: str
    description: str


class IndustryAnalysis(ApiModel):
    structure_summary: str
    forces: list[IndustryForce]
    value_driver_chain: list[ValueLink]


class MoatAxis(ApiModel):
    id: str
    label: str
    status: Literal["SUPPORTED", "MIXED", "NOT_OBSERVED"]
    score: float | None = Field(default=None, ge=0, le=10)
    explanation: str
    evidence_ids: list[str] = Field(default_factory=list)
    counterevidence_ids: list[str] = Field(default_factory=list)


class MoatAnalysis(ApiModel):
    score: float = Field(ge=0, le=10)
    rating: Literal["WIDE", "NARROW", "NONE", "INSUFFICIENT"]
    durability: str
    confidence: float = Field(ge=0, le=1)
    evidence_coverage: float = Field(ge=0, le=1)
    primary_sources: list[str]
    summary: str
    axes: list[MoatAxis]


class ModelRoute(ApiModel):
    primary_model: str
    base_period: str | None = None
    rationale: str
    cross_checks: list[str] = Field(default_factory=list)


class ValuationScenario(ApiModel):
    id: Literal["bear", "base", "bull"]
    label: str
    low: float
    central: float
    high: float
    upside_pct: float
    assumptions: list[str]


class EconomicValueScore(ApiModel):
    percentile: float | None = Field(default=None, ge=0, le=100)
    label: str
    reference_class: str
    sample_size: int = Field(ge=0)
    confidence: ConfidenceLevel
    fragility: ConfidenceLevel
    coverage: float = Field(ge=0, le=1)
    caveat: str


class Assumption(ApiModel):
    id: str
    label: str
    value: str
    source_type: str
    sources: list[str]
    editable: bool = True


class ValuationAnalysis(ApiModel):
    route: ModelRoute
    currency: str
    current_price: float
    base_fair_value: float
    base_value_gap_pct: float
    scenarios: list[ValuationScenario]
    economic_value: EconomicValueScore
    assumptions: list[Assumption]


class DriverRange(ApiModel):
    low: float
    high: float


class ImpliedDriver(ApiModel):
    id: Literal["growth", "margin", "roiic", "cap"]
    label: str
    unit: str
    implied: DriverRange
    base_case: float
    interpretation: str


class ImpliedPoint(ApiModel):
    growth_pct: float
    margin_pct: float
    roiic_pct: float
    cap_years: int
    modeled_price: float
    relative_error_pct: float


class MarketExpectations(ApiModel):
    status: Literal["AVAILABLE", "UNAVAILABLE"]
    method: str
    solution_count: int = Field(ge=0)
    evaluated_point_count: int = Field(ge=0)
    tolerance_pct: float = Field(gt=0)
    drivers: list[ImpliedDriver]
    representative_points: list[ImpliedPoint]
    headline: str
    identification_caveat: str


class SensitivityDriver(ApiModel):
    id: Literal["growth", "margin", "roiic", "cap"]
    label: str
    assumed_change: str
    value_impact_pct: float
    tone: Tone


class SensitivityAnalysis(ApiModel):
    primary_driver_id: str
    primary_driver_label: str
    drivers: list[SensitivityDriver]
    turbo_trigger: str
    fragility: ConfidenceLevel


class PriceExplanation(ApiModel):
    headline: str
    summary: str
    core_question: str
    market_concern: str
    rerating_condition: str


class EvidenceSource(ApiModel):
    source_type: str
    document_id: str
    title: str
    available_at: datetime
    url: str


class EvidenceItem(ApiModel):
    id: str
    direction: Literal["positive", "negative", "neutral"]
    evidence_type: str
    fact: str
    exact_quote: str
    mechanism: list[str]
    strength: float = Field(ge=0, le=1)
    reliability: float = Field(ge=0, le=1)
    period: str | None = None
    linked_drivers: list[str] = Field(default_factory=list)
    source: EvidenceSource


class ThesisMonitorItem(ApiModel):
    id: str
    label: str
    status: Literal["INTACT", "WATCH", "WEAKENING", "UNKNOWN"]
    tone: Tone
    detail: str
    evidence_ids: list[str] = Field(default_factory=list)


class ThesisChange(ApiModel):
    label: str
    previous: str
    current: str
    tone: Tone


class ThesisAnalysis(ApiModel):
    core_thesis: str
    supporting_evidence_ids: list[str]
    breakers: list[str]
    breaker_evidence_ids: list[str]
    monitor: list[ThesisMonitorItem]
    changes_since_previous: list[ThesisChange]


class DecisionSupport(ApiModel):
    value_trap_diagnosis: str
    payoff_profile: str
    what_to_watch_next: list[str]
    use_boundary: str
    disclaimer: str


class VersionInfo(ApiModel):
    runner: str
    model: str
    prompt: str
    parser: str
    calculation: str


class ResearchReport(ApiModel):
    meta: ReportMeta
    company: CompanyProfile
    industry: IndustryAnalysis
    moat: MoatAnalysis
    valuation: ValuationAnalysis
    market_expectations: MarketExpectations
    sensitivity: SensitivityAnalysis
    price_explanation: PriceExplanation
    evidence: list[EvidenceItem]
    thesis: ThesisAnalysis
    decision_support: DecisionSupport
    versions: VersionInfo
