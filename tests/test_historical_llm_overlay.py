from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from moatrader.evidence.historical_corpus import opaque_unit_id, quarantine_market_opinion
from moatrader.evidence.historical_overlay import (
    HistoricalEntailment,
    HistoricalEntailmentBatch,
    HistoricalEntailmentDecision,
    HistoricalEvidenceAxis,
    HistoricalEvidenceDirection,
    HistoricalExcerpt,
    HistoricalPackAssessment,
    HistoricalClaim,
    HistoricalPreprocessSelection,
    HistoricalRiskAction,
    HistoricalSourceRole,
    HistoricalTrapAnswer,
    deterministic_overlay_decision,
    sanitize_preprocess_selection,
    validate_historical_assessment,
    validate_preprocess_selection,
)
from moatrader.ingestion.hankyung import load_hankyung_company_reports
from moatrader.llm.contracts import LLMTask
from moatrader.llm.transport import OpenAIResponsesTransport


SEOUL = ZoneInfo("Asia/Seoul")


def _excerpt(role: HistoricalSourceRole, text: str, suffix: str) -> HistoricalExcerpt:
    return HistoricalExcerpt.create(
        unit_id=f"unit-{suffix}",
        source_id=f"source-{suffix}",
        source_role=role,
        available_at=datetime(2020, 3, 1, tzinfo=SEOUL),
        text=text,
    )


def _assessment(pack_id: str, claims: list[HistoricalClaim]) -> HistoricalPackAssessment:
    return HistoricalPackAssessment(
        pack_id=pack_id,
        claims=claims,
        future_trap_answer=HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE,
    )


def test_company_report_loader_uses_register_date_and_excludes_opinion_fields(tmp_path: Path) -> None:
    metadata = tmp_path / "reports.json"
    metadata.write_text(
        json.dumps(
            [
                {
                    "REPORT_IDX": 123,
                    "REPORT_TYPE": "CO",
                    "BUSINESS_CODE": "5930",
                    "BUSINESS_NAME": "삼성전자",
                    "REPORT_TITLE": "기업 분석",
                    "REPORT_FILENAME": "report.pdf",
                    "REPORT_FILEPATH": "https://example.test/report",
                    "REPORT_DATE": "2020-03-02",
                    "REGISTER_DATE": "20200302091530",
                    "TARGET_STOCK_PRICES": "999999",
                    "GRADE_VALUE": "Buy",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    report = load_hankyung_company_reports(metadata)["123"]
    assert report.ticker == "005930"
    assert report.registered_at.isoformat() == "2020-03-02T09:15:30+09:00"
    hints = report.adapter_hints()
    serialized = json.dumps(hints, ensure_ascii=False)
    assert "999999" not in serialized
    assert "Buy" not in serialized
    assert hints["source_specific"]["market_opinion_fields_quarantined"] is True


def test_company_report_loader_isolates_rows_without_company_identity(tmp_path: Path) -> None:
    metadata = tmp_path / "reports.json"
    metadata.write_text(
        json.dumps(
            [
                {
                    "REPORT_IDX": 999,
                    "REPORT_TYPE": "CO",
                    "BUSINESS_CODE": "",
                    "BUSINESS_NAME": "",
                    "REPORT_DATE": "2020-01-01",
                    "REGISTER_DATE": "20200101090000",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert load_hankyung_company_reports(metadata) == {}


def test_market_opinion_quarantine_removes_prices_and_recommendations() -> None:
    cleaned, removed = quarantine_market_opinion(
        "가격 전가로 매출총이익률이 개선됐다.\n목표주가 80,000원, 투자의견 Buy\n고객 갱신율은 92%다."
    )
    assert removed == 1
    assert "80,000" not in cleaned
    assert "Buy" not in cleaned
    assert "갱신율" in cleaned


def test_llm_visible_unit_id_is_short_opaque_and_stable() -> None:
    value = opaque_unit_id("opendart:very:long:source", "가격을 인상했다.")
    assert value == opaque_unit_id("opendart:very:long:source", "가격을 인상했다.")
    assert value.startswith("U_")
    assert len(value) == 22


def test_strict_gate_accepts_exact_pit_entailed_stable_claims() -> None:
    quote = "원재료 상승분을 판매가격에 반영하였다."
    dart = _excerpt(HistoricalSourceRole.DART_ORIGINAL, f"회사는 {quote} 이후 매출을 공시했다.", "dart")
    claim = HistoricalClaim(
        judgment_id="j1",
        axis=HistoricalEvidenceAxis.PRICING_POWER,
        direction=HistoricalEvidenceDirection.SUPPORTIVE,
        claim="회사가 원재료 상승분을 판매가격에 반영했다.",
        unit_id=dart.unit_id,
        exact_quote=quote,
        confidence=0.8,
    )
    original = _assessment("p1", [claim])
    anonymous = _assessment("p1", [claim.model_copy(update={"confidence": 0.75})])
    entailed = HistoricalEntailmentBatch(
        decisions=[HistoricalEntailmentDecision(judgment_id="j1", verdict=HistoricalEntailment.ENTAILED)]
    )
    validated = validate_historical_assessment(
        cutoff=datetime(2020, 3, 31, 23, 59, tzinfo=SEOUL),
        excerpts=[dart],
        original=original,
        anonymized=anonymous,
        entailment=entailed,
    )
    assert validated[0].exact_quote == quote
    assert deterministic_overlay_decision("p1", validated).action == HistoricalRiskAction.PASS


def test_gate_fails_closed_on_future_trap_quote_instability_and_preprocessor_hallucination() -> None:
    excerpt = _excerpt(HistoricalSourceRole.IR, "고객 계약 갱신률이 하락했다.", "ir")
    selection = HistoricalPreprocessSelection(pack_id="p1", selected_unit_ids=["missing"])
    with pytest.raises(ValueError, match="absent units"):
        validate_preprocess_selection(selection, [excerpt])
    sanitized = sanitize_preprocess_selection(
        HistoricalPreprocessSelection(
            pack_id="p1",
            selected_unit_ids=["missing", excerpt.unit_id, excerpt.unit_id],
            industry_codes=["159", "159"],
        ),
        [excerpt],
        allowed_industry_codes={"159"},
    )
    assert sanitized.selected_unit_ids == [excerpt.unit_id]
    assert sanitized.industry_codes == ["159"]
    claim = HistoricalClaim(
        judgment_id="j1",
        axis=HistoricalEvidenceAxis.SWITCHING_COST,
        direction=HistoricalEvidenceDirection.EROSIVE,
        claim="계약 갱신률이 하락했다.",
        unit_id=excerpt.unit_id,
        exact_quote="문서에 없는 인용문",
        confidence=0.9,
    )
    original = _assessment("p1", [claim])
    unstable = _assessment("p1", [claim.model_copy(update={"direction": HistoricalEvidenceDirection.SUPPORTIVE})])
    entailed = HistoricalEntailmentBatch(
        decisions=[HistoricalEntailmentDecision(judgment_id="j1", verdict=HistoricalEntailment.ENTAILED)]
    )
    with pytest.raises(ValueError, match="instability"):
        validate_historical_assessment(
            cutoff=datetime(2020, 3, 31, tzinfo=SEOUL),
            excerpts=[excerpt],
            original=original,
            anonymized=unstable,
            entailment=entailed,
        )


def test_two_source_erosion_is_veto_but_llm_never_changes_rank() -> None:
    dart = _excerpt(HistoricalSourceRole.DART_ORIGINAL, "주요 고객 이탈이 증가했다.", "d")
    analyst = _excerpt(HistoricalSourceRole.COMPANY_ANALYST, "계약 갱신률 하락이 확인된다.", "a")
    claims = []
    for index, (excerpt, axis) in enumerate(
        ((dart, HistoricalEvidenceAxis.FRAGILITY), (analyst, HistoricalEvidenceAxis.SWITCHING_COST)),
        start=1,
    ):
        quote = excerpt.text
        claims.append(
            {
                "judgment_id": f"j{index}",
                "axis": axis,
                "direction": HistoricalEvidenceDirection.EROSIVE,
                "claim": quote,
                "source_id": excerpt.source_id,
                "source_role": excerpt.source_role,
                "available_at": excerpt.available_at,
                "unit_id": excerpt.unit_id,
                "char_start": 0,
                "char_end": len(quote),
                "exact_quote": quote,
                "confidence": 0.9,
            }
        )
    from moatrader.evidence.historical_overlay import ValidatedHistoricalClaim

    decision = deterministic_overlay_decision("p1", [ValidatedHistoricalClaim(**item) for item in claims])
    assert decision.action == HistoricalRiskAction.VETO
    assert decision.llm_changed_cheap_rank is False


def test_requested_models_route_to_nano_and_luna() -> None:
    transport = OpenAIResponsesTransport(
        summary_model="gpt-5-nano-2025-08-07",
        moat_model="gpt-5.6-luna",
    )
    assert transport._model_for(LLMTask.HISTORICAL_PREPROCESSING) == "gpt-5-nano-2025-08-07"
    assert transport._model_for(LLMTask.HISTORICAL_EVIDENCE_CLASSIFICATION) == "gpt-5.6-luna"
    assert transport._model_for(LLMTask.HISTORICAL_ENTAILMENT_CHECK) == "gpt-5.6-luna"


def test_empty_claim_pack_needs_no_entailment_decisions() -> None:
    assert HistoricalEntailmentBatch(decisions=[]).decisions == []
