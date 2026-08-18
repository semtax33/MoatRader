from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from moatrader.experiments.historical_validation import (
    AnonymizationPair,
    CutoffDocument,
    EntailmentVerdict,
    EvidenceJudgment,
    ExactEvidenceCitation,
    FutureKnowledgeTrapResult,
    HistoricalValidationContract,
    TrapAnswer,
    ValidationGrade,
    enforce_anonymization_stability,
    enforce_cutoff_evidence_gate,
    enforce_future_knowledge_traps,
)


SEOUL = ZoneInfo("Asia/Seoul")


def _document(*, available_at: datetime | None = None) -> CutoffDocument:
    text = "매출액은 전년 대비 12% 증가했다."
    return CutoffDocument(
        source_id="dart:20200330000001",
        available_at=available_at or datetime(2020, 3, 30, 23, 59, 59, tzinfo=SEOUL),
        text=text,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )


def _judgment(document: CutoffDocument) -> EvidenceJudgment:
    quote = "12% 증가"
    start = document.text.index(quote)
    return EvidenceJudgment(
        judgment_id="growth-1",
        classification="SUPPORTED",
        claim="매출액이 전년 대비 12% 증가했다.",
        citations=[
            ExactEvidenceCitation(
                source_id=document.source_id,
                char_start=start,
                char_end=start + len(quote),
                exact_quote=quote,
            )
        ],
        entailment=EntailmentVerdict.ENTAILED,
        entailment_checker="deterministic-test-v1",
    )


def test_historical_contract_is_hashed_and_uses_honest_validation_grades() -> None:
    contract = HistoricalValidationContract.create(
        frozen_on=date(2026, 8, 18),
        start_date=date(2020, 1, 1),
        end_date=date(2025, 12, 31),
        signal_dates=[date(2020, 2, 29), date(2020, 5, 31)],
        universe_sha256="a" * 64,
        universe_selected_as_of=date(2025, 8, 1),
    )

    assert contract.contract_sha256 != "0" * 64
    assert contract.deterministic_alpha_grade == ValidationGrade.DATA_PIT_HISTORICAL
    assert contract.llm_overlay_grade == ValidationGrade.LLM_PIT_PSEUDO_OOS
    assert contract.live_grade == ValidationGrade.TRUE_LIVE_OOS
    assert contract.llm_may_change_rank is False
    assert contract.survivorship_and_membership_bias_disclosed is True


def test_cutoff_evidence_gate_accepts_exact_pre_cutoff_citation() -> None:
    document = _document()
    enforce_cutoff_evidence_gate(
        cutoff=datetime(2020, 3, 31, tzinfo=SEOUL),
        documents=[document],
        judgments=[_judgment(document)],
    )


@pytest.mark.parametrize("failure", ["future", "span", "entailment"])
def test_cutoff_evidence_gate_rejects_non_pit_or_unsupported_claims(failure: str) -> None:
    cutoff = datetime(2020, 3, 31, tzinfo=SEOUL)
    document = _document(
        available_at=cutoff + timedelta(days=1) if failure == "future" else None
    )
    judgment = _judgment(document)
    if failure == "span":
        citation = judgment.citations[0]
        judgment = judgment.model_copy(
            update={
                "citations": [
                    citation.model_copy(update={"exact_quote": "미래를 안다"})
                ]
            }
        )
    if failure == "entailment":
        judgment = judgment.model_copy(update={"entailment": EntailmentVerdict.UNKNOWN})

    with pytest.raises(ValueError):
        enforce_cutoff_evidence_gate(
            cutoff=cutoff,
            documents=[document],
            judgments=[judgment],
        )


def test_future_knowledge_traps_require_unknown_from_cutoff_evidence() -> None:
    enforce_future_knowledge_traps(
        [
            FutureKnowledgeTrapResult(
                trap_id="future-acquisition",
                question="이 회사는 2024년에 인수되는가?",
                answer=TrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE,
            )
        ]
    )
    with pytest.raises(ValueError, match="future-knowledge trap failure"):
        enforce_future_knowledge_traps(
            [
                FutureKnowledgeTrapResult(
                    trap_id="future-acquisition",
                    question="이 회사는 2024년에 인수되는가?",
                    answer=TrapAnswer.CLAIMED_KNOWLEDGE,
                )
            ]
        )


def test_anonymization_stability_rejects_identity_dependent_output() -> None:
    enforce_anonymization_stability(
        [
            AnonymizationPair(
                pair_id="stable",
                original_classification="PASS",
                anonymized_classification="PASS",
                original_score=3.0,
                anonymized_score=3.2,
            )
        ]
    )
    with pytest.raises(ValueError, match="anonymization instability"):
        enforce_anonymization_stability(
            [
                AnonymizationPair(
                    pair_id="unstable",
                    original_classification="PASS",
                    anonymized_classification="FAIL",
                )
            ]
        )
