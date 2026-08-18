from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from enum import StrEnum

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class ValidationGrade(StrEnum):
    DATA_PIT_HISTORICAL = "DATA_PIT_HISTORICAL"
    LLM_PIT_PSEUDO_OOS = "LLM_PIT_PSEUDO_OOS"
    TRUE_LIVE_OOS = "TRUE_LIVE_OOS"


class EntailmentVerdict(StrEnum):
    ENTAILED = "ENTAILED"
    NOT_ENTAILED = "NOT_ENTAILED"
    UNKNOWN = "UNKNOWN"


class TrapAnswer(StrEnum):
    UNKNOWN_FROM_CUTOFF_EVIDENCE = "UNKNOWN_FROM_CUTOFF_EVIDENCE"
    CLAIMED_KNOWLEDGE = "CLAIMED_KNOWLEDGE"


class CutoffDocument(ContractModel):
    source_id: str = Field(min_length=1)
    available_at: datetime
    text: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def text_hash_and_timestamp(self) -> "CutoffDocument":
        if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
            raise ValueError("cutoff document available_at must be timezone-aware")
        actual = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if actual != self.sha256:
            raise ValueError("cutoff document text hash mismatch")
        return self


class ExactEvidenceCitation(ContractModel):
    source_id: str = Field(min_length=1)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    exact_quote: str = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_span(self) -> "ExactEvidenceCitation":
        if self.char_end <= self.char_start:
            raise ValueError("citation char_end must follow char_start")
        return self


class EvidenceJudgment(ContractModel):
    judgment_id: str = Field(min_length=1)
    classification: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    citations: list[ExactEvidenceCitation] = Field(min_length=1)
    entailment: EntailmentVerdict
    entailment_checker: str = Field(min_length=1)


class FutureKnowledgeTrapResult(ContractModel):
    trap_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answer: TrapAnswer


class AnonymizationPair(ContractModel):
    pair_id: str = Field(min_length=1)
    original_classification: str = Field(min_length=1)
    anonymized_classification: str = Field(min_length=1)
    original_score: float | None = None
    anonymized_score: float | None = None


class HistoricalValidationContract(ContractModel):
    schema_version: str = "moatrader-v7-historical-validation/1"
    frozen_on: date
    start_date: date
    end_date: date
    signal_dates: list[date] = Field(min_length=1)
    universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    universe_count: int = 150
    universe_selected_as_of: date
    fixed_future_universe_backcast: bool = True
    survivorship_and_membership_bias_disclosed: bool = True
    deterministic_alpha_grade: ValidationGrade = ValidationGrade.DATA_PIT_HISTORICAL
    llm_overlay_grade: ValidationGrade = ValidationGrade.LLM_PIT_PSEUDO_OOS
    live_grade: ValidationGrade = ValidationGrade.TRUE_LIVE_OOS
    deterministic_rank_signal: str = "CHEAP"
    llm_may_change_rank: bool = False
    exact_cutoff_citations_required: bool = True
    entailment_required: bool = True
    future_knowledge_traps_required: bool = True
    anonymization_stability_required: bool = True
    deterministic_llm_ablation_required: bool = True
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def honest_historical_labels(self) -> "HistoricalValidationContract":
        if self.end_date < self.start_date:
            raise ValueError("historical validation end_date precedes start_date")
        if self.signal_dates != sorted(set(self.signal_dates)):
            raise ValueError("historical signal dates must be unique and chronological")
        if any(item < self.start_date or item > self.end_date for item in self.signal_dates):
            raise ValueError("historical signal date is outside the contract period")
        if not self.fixed_future_universe_backcast or not self.survivorship_and_membership_bias_disclosed:
            raise ValueError("fixed 2025 universe backcast bias must be explicit")
        if self.llm_may_change_rank:
            raise ValueError("LLM overlay cannot control the historical Cheap rank")
        gates = (
            self.exact_cutoff_citations_required,
            self.entailment_required,
            self.future_knowledge_traps_required,
            self.anonymization_stability_required,
            self.deterministic_llm_ablation_required,
        )
        if not all(gates):
            raise ValueError("historical LLM validation gates must all be enabled")
        payload = self.model_dump(mode="json", exclude={"contract_sha256"})
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if hashlib.sha256(encoded.encode("utf-8")).hexdigest() != self.contract_sha256:
            raise ValueError("historical validation contract hash mismatch")
        return self

    @classmethod
    def create(cls, **values: object) -> "HistoricalValidationContract":
        draft = cls.model_construct(contract_sha256="0" * 64, **values)
        payload = draft.model_dump(mode="json", exclude={"contract_sha256"})
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        payload["contract_sha256"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        return cls.model_validate(payload)


def enforce_cutoff_evidence_gate(
    *,
    cutoff: datetime,
    documents: list[CutoffDocument],
    judgments: list[EvidenceJudgment],
) -> None:
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("historical cutoff must be timezone-aware")
    by_id = {document.source_id: document for document in documents}
    if len(by_id) != len(documents):
        raise ValueError("cutoff document IDs must be unique")
    for judgment in judgments:
        if judgment.entailment != EntailmentVerdict.ENTAILED:
            raise ValueError(f"judgment is not entailed by cutoff evidence: {judgment.judgment_id}")
        for citation in judgment.citations:
            document = by_id.get(citation.source_id)
            if document is None:
                raise ValueError(f"citation source is absent: {citation.source_id}")
            if document.available_at > cutoff:
                raise ValueError(f"future evidence at historical cutoff: {citation.source_id}")
            if citation.char_end > len(document.text):
                raise ValueError(f"citation span exceeds source length: {citation.source_id}")
            if document.text[citation.char_start : citation.char_end] != citation.exact_quote:
                raise ValueError(f"citation is not an exact source span: {citation.source_id}")


def enforce_future_knowledge_traps(results: list[FutureKnowledgeTrapResult]) -> None:
    if not results:
        raise ValueError("future-knowledge trap suite cannot be empty")
    failed = [item.trap_id for item in results if item.answer != TrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE]
    if failed:
        raise ValueError("future-knowledge trap failure: " + ", ".join(failed))


def enforce_anonymization_stability(
    pairs: list[AnonymizationPair],
    *,
    maximum_score_delta: float = 0.5,
) -> None:
    if not pairs:
        raise ValueError("anonymization stability suite cannot be empty")
    failures: list[str] = []
    for pair in pairs:
        if pair.original_classification != pair.anonymized_classification:
            failures.append(pair.pair_id)
            continue
        if (pair.original_score is None) != (pair.anonymized_score is None):
            failures.append(pair.pair_id)
            continue
        if (
            pair.original_score is not None
            and pair.anonymized_score is not None
            and abs(pair.original_score - pair.anonymized_score) > maximum_score_delta
        ):
            failures.append(pair.pair_id)
    if failures:
        raise ValueError("anonymization instability: " + ", ".join(failures))
