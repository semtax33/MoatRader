from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class HistoricalSourceRole(StrEnum):
    DART_ORIGINAL = "DART_ORIGINAL"
    IR = "IR"
    COMPANY_ANALYST = "COMPANY_ANALYST"
    INDUSTRY_ANALYST = "INDUSTRY_ANALYST"


class HistoricalEvidenceAxis(StrEnum):
    PRICING_POWER = "PRICING_POWER"
    SWITCHING_COST = "SWITCHING_COST"
    OTHER_MOAT = "OTHER_MOAT"
    FRAGILITY = "FRAGILITY"
    CAP_SUPPORT = "CAP_SUPPORT"


class HistoricalEvidenceDirection(StrEnum):
    SUPPORTIVE = "SUPPORTIVE"
    EROSIVE = "EROSIVE"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class HistoricalTrapAnswer(StrEnum):
    UNKNOWN_FROM_CUTOFF_EVIDENCE = "UNKNOWN_FROM_CUTOFF_EVIDENCE"
    CLAIMED_FUTURE_KNOWLEDGE = "CLAIMED_FUTURE_KNOWLEDGE"


class HistoricalEntailment(StrEnum):
    ENTAILED = "ENTAILED"
    NOT_ENTAILED = "NOT_ENTAILED"
    UNKNOWN = "UNKNOWN"


class HistoricalRiskAction(StrEnum):
    PASS = "PASS"
    POSITION_CAP = "POSITION_CAP"
    VETO = "VETO"


class HistoricalExcerpt(ContractModel):
    unit_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_role: HistoricalSourceRole
    available_at: datetime
    text: str = Field(min_length=1)
    text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def valid_pit_excerpt(self) -> "HistoricalExcerpt":
        if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
            raise ValueError("historical excerpt available_at must be timezone-aware")
        actual = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        if actual != self.text_sha256:
            raise ValueError("historical excerpt text hash mismatch")
        return self

    @classmethod
    def create(cls, **values: object) -> "HistoricalExcerpt":
        text = str(values["text"])
        return cls(text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(), **values)


class HistoricalPreprocessSelection(ContractModel):
    pack_id: str = Field(min_length=1)
    selected_unit_ids: list[str] = Field(min_length=1, max_length=16)
    industry_codes: list[str] = Field(default_factory=list, max_length=2)

class HistoricalPreprocessBatch(ContractModel):
    packs: list[HistoricalPreprocessSelection] = Field(min_length=1, max_length=6)


class HistoricalClaim(ContractModel):
    judgment_id: str = Field(min_length=1)
    axis: HistoricalEvidenceAxis
    direction: HistoricalEvidenceDirection
    claim: str = Field(min_length=1)
    unit_id: str = Field(min_length=1)
    exact_quote: str = Field(min_length=8)
    confidence: float = Field(ge=0, le=1)


class HistoricalPackAssessment(ContractModel):
    pack_id: str = Field(min_length=1)
    claims: list[HistoricalClaim] = Field(default_factory=list, max_length=8)
    future_trap_answer: HistoricalTrapAnswer

class HistoricalAssessmentBatch(ContractModel):
    packs: list[HistoricalPackAssessment] = Field(min_length=1, max_length=6)


class HistoricalEntailmentDecision(ContractModel):
    judgment_id: str = Field(min_length=1)
    verdict: HistoricalEntailment


class HistoricalEntailmentBatch(ContractModel):
    decisions: list[HistoricalEntailmentDecision] = Field(default_factory=list)


class ValidatedHistoricalClaim(ContractModel):
    judgment_id: str
    axis: HistoricalEvidenceAxis
    direction: HistoricalEvidenceDirection
    claim: str
    source_id: str
    source_role: HistoricalSourceRole
    available_at: datetime
    unit_id: str
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    exact_quote: str
    confidence: float = Field(ge=0, le=1)


class HistoricalOverlayDecision(ContractModel):
    pack_id: str
    action: HistoricalRiskAction
    validated_claims: list[ValidatedHistoricalClaim]
    supportive_count: int = Field(ge=0)
    erosive_count: int = Field(ge=0)
    source_role_count: int = Field(ge=0)
    validation_grade: str = "LLM_PIT_PSEUDO_OOS"
    llm_changed_cheap_rank: bool = False


def validate_preprocess_selection(
    selection: HistoricalPreprocessSelection,
    excerpts: list[HistoricalExcerpt],
) -> list[HistoricalExcerpt]:
    by_id = {item.unit_id: item for item in excerpts}
    if len(by_id) != len(excerpts):
        raise ValueError("historical excerpt unit IDs must be unique")
    missing = [item for item in selection.selected_unit_ids if item not in by_id]
    if missing:
        raise ValueError("preprocessor selected absent units: " + ", ".join(missing))
    return [by_id[item] for item in selection.selected_unit_ids]


def sanitize_preprocess_selection(
    selection: HistoricalPreprocessSelection,
    excerpts: list[HistoricalExcerpt],
    *,
    allowed_industry_codes: set[str],
) -> HistoricalPreprocessSelection:
    """Drop model-invented/duplicate routing IDs without adding evidence."""

    allowed_units = {item.unit_id for item in excerpts}
    selected_units = list(
        dict.fromkeys(item for item in selection.selected_unit_ids if item in allowed_units)
    )
    if not selected_units:
        raise ValueError("preprocessor returned no valid supplied unit IDs")
    industries = list(
        dict.fromkeys(item for item in selection.industry_codes if item in allowed_industry_codes)
    )[:2]
    return HistoricalPreprocessSelection(
        pack_id=selection.pack_id,
        selected_unit_ids=selected_units[:16],
        industry_codes=industries,
    )


def _classification_signature(assessment: HistoricalPackAssessment) -> list[tuple[str, str]]:
    return sorted((item.axis.value, item.direction.value) for item in assessment.claims)


def validate_historical_assessment(
    *,
    cutoff: datetime,
    excerpts: list[HistoricalExcerpt],
    original: HistoricalPackAssessment,
    anonymized: HistoricalPackAssessment,
    entailment: HistoricalEntailmentBatch,
    maximum_confidence_delta: float = 0.15,
) -> list[ValidatedHistoricalClaim]:
    if cutoff.tzinfo is None or cutoff.utcoffset() is None:
        raise ValueError("historical cutoff must be timezone-aware")
    if original.pack_id != anonymized.pack_id:
        raise ValueError("original/anonymized pack IDs differ")
    if original.future_trap_answer != HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE:
        raise ValueError("original future-knowledge trap failed")
    if anonymized.future_trap_answer != HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE:
        raise ValueError("anonymized future-knowledge trap failed")
    if _classification_signature(original) != _classification_signature(anonymized):
        raise ValueError("anonymization classification instability")
    original_conf = sorted(item.confidence for item in original.claims)
    anonymized_conf = sorted(item.confidence for item in anonymized.claims)
    if any(abs(left - right) > maximum_confidence_delta for left, right in zip(original_conf, anonymized_conf)):
        raise ValueError("anonymization confidence instability")
    decisions = {item.judgment_id: item.verdict for item in entailment.decisions}
    if len(decisions) != len(entailment.decisions):
        raise ValueError("entailment decision IDs must be unique")
    by_unit = {item.unit_id: item for item in excerpts}
    validated: list[ValidatedHistoricalClaim] = []
    for claim in original.claims:
        if claim.direction in {HistoricalEvidenceDirection.UNKNOWN, HistoricalEvidenceDirection.MIXED}:
            continue
        if decisions.get(claim.judgment_id) != HistoricalEntailment.ENTAILED:
            raise ValueError(f"claim is not independently entailed: {claim.judgment_id}")
        excerpt = by_unit.get(claim.unit_id)
        if excerpt is None:
            raise ValueError(f"claim cites absent unit: {claim.unit_id}")
        if excerpt.available_at > cutoff:
            raise ValueError(f"claim cites future evidence: {claim.unit_id}")
        start = excerpt.text.find(claim.exact_quote)
        if start < 0:
            raise ValueError(f"claim quote is not an exact source span: {claim.judgment_id}")
        validated.append(
            ValidatedHistoricalClaim(
                **claim.model_dump(),
                source_id=excerpt.source_id,
                source_role=excerpt.source_role,
                available_at=excerpt.available_at,
                char_start=start,
                char_end=start + len(claim.exact_quote),
            )
        )
    return validated


def deterministic_overlay_decision(
    pack_id: str,
    claims: list[ValidatedHistoricalClaim],
) -> HistoricalOverlayDecision:
    supportive = [item for item in claims if item.direction == HistoricalEvidenceDirection.SUPPORTIVE]
    erosive = [item for item in claims if item.direction == HistoricalEvidenceDirection.EROSIVE]
    erosive_roles = {item.source_role for item in erosive}
    severe_axes = {
        HistoricalEvidenceAxis.PRICING_POWER,
        HistoricalEvidenceAxis.SWITCHING_COST,
        HistoricalEvidenceAxis.OTHER_MOAT,
        HistoricalEvidenceAxis.FRAGILITY,
    }
    severe = [item for item in erosive if item.axis in severe_axes]
    if len(severe) >= 2 and len({item.source_role for item in severe}) >= 2:
        action = HistoricalRiskAction.VETO
    elif erosive:
        action = HistoricalRiskAction.POSITION_CAP
    else:
        action = HistoricalRiskAction.PASS
    return HistoricalOverlayDecision(
        pack_id=pack_id,
        action=action,
        validated_claims=claims,
        supportive_count=len(supportive),
        erosive_count=len(erosive),
        source_role_count=len({item.source_role for item in claims}),
    )


def anonymize_text(text: str, issuer_name: str, ticker: str) -> str:
    result = text.replace(ticker, "COMPANY_CODE")
    variants = {issuer_name, re.sub(r"(?:주식회사|\(주\)|㈜)", "", issuer_name).strip()}
    for value in sorted((item for item in variants if item), key=len, reverse=True):
        result = re.sub(re.escape(value), "COMPANY_A", result, flags=re.IGNORECASE)
    return result
