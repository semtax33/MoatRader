from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.evidence.models import MoatScore
from moatrader.financial.dcf import DcfValuation
from moatrader.llm.transport import TransportUsage
from moatrader.screening.ranker import RankedCandidate


class CompanyRunStatus(StrEnum):
    PREPARED = "PREPARED"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"
    NO_PIT_DOCUMENTS = "NO_PIT_DOCUMENTS"


class UniverseRunConfig(ContractModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    as_of: datetime
    summary_model: str = Field(default="gpt-5-nano", min_length=1)
    moat_model: str = Field(default="gpt-5.6-luna", min_length=1)
    summary_reasoning_effort: str = Field(default="low", min_length=1)
    moat_reasoning_effort: str = Field(default="medium", min_length=1)
    context_tokens: int = Field(default=64_000, gt=8_000)
    prompt_reserve_tokens: int = Field(default=8_000, ge=1_000)
    max_output_tokens: int = Field(default=8_000, ge=1_000, le=100_000)
    minimum_text_retention: float = Field(default=0.95, ge=0.0, le=1.0)
    minimum_numeric_retention: float = Field(default=0.99, ge=0.0, le=1.0)
    minimum_structured_fact_retention: float = Field(default=0.99, ge=0.0, le=1.0)
    require_table_count_match: bool = True
    require_financial_table_semantics: bool = True
    allow_low_quality: bool = False
    maximum_price_age_days: int = Field(default=7, ge=0, le=366)
    maximum_evidence_chunks: int | None = Field(default=24, ge=1, le=1000)
    evidence_batch_max_tokens: int | None = Field(default=4_000, ge=500, le=100_000)
    consolidate_section_summaries: bool = True
    include_raw_moat_appendix: bool = False
    workers: int = Field(default=1, ge=1, le=32)
    resume: bool = False
    dry_run: bool = False
    validation_attempts: int = Field(default=2, ge=1, le=5)
    experiment_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    llm_replay_cache_directory: str | None = None
    evidence_ledger_directory: str | None = None

    @model_validator(mode="after")
    def config_is_valid(self) -> "UniverseRunConfig":
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        if self.context_tokens <= self.prompt_reserve_tokens:
            raise ValueError("context_tokens must exceed prompt_reserve_tokens")
        if bool(self.experiment_id) != bool(self.llm_replay_cache_directory):
            raise ValueError(
                "experiment_id and llm_replay_cache_directory must be configured together"
            )
        if self.evidence_ledger_directory and not self.experiment_id:
            raise ValueError("evidence_ledger_directory requires experiment_id")
        return self


class LLMCallAudit(ContractModel):
    task: str
    input_sha256: str
    provider: str
    model: str
    response_id: str | None = None
    usage: TransportUsage = Field(default_factory=TransportUsage)
    created_at: datetime
    raw_response_path: str | None = None
    raw_response_sha256: str | None = None
    normalized_output_sha256: str | None = None
    replayed: bool = False
    replay_cache_key: str | None = None


class CompanyRunResult(ContractModel):
    ticker: str
    issuer_id: str | None = None
    issuer_name: str | None = None
    status: CompanyRunStatus
    run_signature: str
    source_document_ids: list[str] = Field(default_factory=list)
    evidence_count: int = Field(default=0, ge=0)
    chunk_count: int = Field(default=0, ge=0)
    selected_chunk_count: int = Field(default=0, ge=0)
    moat_score: MoatScore | None = None
    dcf: DcfValuation | None = None
    current_price: Decimal | None = None
    price_as_of: datetime | None = None
    valuation_as_of: datetime | None = None
    error: str | None = None
    artifact_directory: str
    llm_usage: TransportUsage = Field(default_factory=TransportUsage)
    runner_version: str | None = None


class UniverseRunResult(ContractModel):
    run_id: str
    as_of: datetime
    started_at: datetime
    completed_at: datetime
    companies: list[CompanyRunResult]
    ranking: list[RankedCandidate] = Field(default_factory=list)

    @property
    def failed_count(self) -> int:
        return sum(company.status == CompanyRunStatus.FAILED for company in self.companies)
