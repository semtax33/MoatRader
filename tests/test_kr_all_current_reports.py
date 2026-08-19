from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from scripts import run_kr_all_current_reports as current
from moatrader.evidence.historical_overlay import (
    HistoricalClaim,
    HistoricalEntailment,
    HistoricalEntailmentBatch,
    HistoricalEntailmentDecision,
    HistoricalEvidenceAxis,
    HistoricalEvidenceDirection,
    HistoricalExcerpt,
    HistoricalPackAssessment,
    HistoricalSourceRole,
    HistoricalTrapAnswer,
)


CUTOFF = datetime.fromisoformat("2026-08-18T23:59:59+09:00")


def _csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_dart_text_extraction_is_deterministic_and_deduplicates_nested_blocks(
    tmp_path: Path,
) -> None:
    source = tmp_path / "filing.xml"
    source.write_text(
        "<DOCUMENT><DIV><P>고객 계약 갱신률과 가격 인상 근거를 설명하는 충분히 긴 원문 문장입니다.</P>"
        "<P>고객 계약 갱신률과 가격 인상 근거를 설명하는 충분히 긴 원문 문장입니다.</P>"
        "</DIV></DOCUMENT>",
        encoding="utf-8",
    )

    first = current._dart_document_text(source)
    second = current._dart_document_text(source)

    assert first == second
    assert "고객 계약 갱신률" in first
    assert first.split("\n\n").count(
        "고객 계약 갱신률과 가격 인상 근거를 설명하는 충분히 긴 원문 문장입니다."
    ) == 1


def test_prepare_deduplicates_common_issuer_pack_and_keeps_every_security(
    tmp_path: Path,
    monkeypatch,
) -> None:
    universe = tmp_path / "universe.csv"
    _csv(
        universe,
        [
            {"stock_code": "000001", "name": "보통주", "market": "KOSPI", "security_type": "COMMON"},
            {"stock_code": "000002", "name": "우선주", "market": "KOSPI", "security_type": "PREFERRED"},
            {"stock_code": "000003", "name": "무공시", "market": "KONEX", "security_type": "COMMON"},
        ],
        ["stock_code", "name", "market", "security_type"],
    )
    document = tmp_path / "dart.xml"
    document.write_text(
        "<DOCUMENT><P>회사는 장기 고객 계약과 반복 매출, 가격 조정 조항을 보유하고 있으며 경쟁 위험도 공시한다.</P></DOCUMENT>",
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.json"
    metadata.write_text(
        json.dumps(
            {
                "source_document_id": "DART-1",
                "available_at": "2026-08-14T18:00:00+09:00",
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.csv"
    _csv(
        manifest,
        [
            {
                "ticker": ticker,
                "filing_ticker": "000001",
                "source": "DART",
                "input": str(document),
                "metadata": str(metadata),
                "issuer_id": "ISSUER-1",
                "issuer_name": "테스트회사",
            }
            for ticker in ("000001", "000002")
        ],
        ["ticker", "filing_ticker", "source", "input", "metadata", "issuer_id", "issuer_name"],
    )
    monkeypatch.setattr(current, "_report_catalog", lambda *_args, **_kwargs: ({}, {}, {}))
    monkeypatch.setattr(current, "_ir_catalog", lambda *_args, **_kwargs: {})
    output = tmp_path / "reports"

    pack_ids = current.prepare(
        universe_path=universe,
        dart_manifest_path=manifest,
        ir_manifest_path=None,
        synalyst_hankyung_root=tmp_path / "synalyst",
        output=output,
        cutoff=CUTOFF,
    )

    assert pack_ids == ["KR-2026-08-18-000001"]
    pack = current.read_json(output / "packs" / "candidates" / f"{pack_ids[0]}.json")
    assert pack["security_tickers"] == ["000001", "000002"]
    assert len(pack["source_assignment"]["dart_documents"]) == 1
    security_map = current.read_json(output / "security-map.json")
    assert len(security_map) == 3
    assert security_map[-1]["status"] == "NO_PERIODIC_PIT_FILING"


def test_ir_catalog_rejects_non_ir_rows_from_combined_bronze_manifest(
    tmp_path: Path,
) -> None:
    ir_input = tmp_path / "ir.pdf"
    ir_input.write_bytes(b"%PDF-test")
    dart_input = tmp_path / "dart.xml"
    dart_input.write_text("<DOCUMENT/>", encoding="utf-8")
    ir_metadata = tmp_path / "ir.json"
    dart_metadata = tmp_path / "dart.json"
    for path in (ir_metadata, dart_metadata):
        path.write_text(
            json.dumps({"available_at": "2026-08-14T00:00:00+09:00"}),
            encoding="utf-8",
        )
    manifest = tmp_path / "combined.csv"
    _csv(
        manifest,
        [
            {
                "ticker": "000001",
                "source": "DART",
                "input": dart_input.name,
                "metadata": dart_metadata.name,
            },
            {
                "ticker": "000001",
                "source": "IR",
                "input": ir_input.name,
                "metadata": ir_metadata.name,
            },
        ],
        ["ticker", "source", "input", "metadata"],
    )

    catalog = current._ir_catalog(manifest, cutoff=CUTOFF)

    assert len(catalog["000001"]) == 1
    assert catalog["000001"][0]["source"] == "IR"
    assert catalog["000001"][0]["input"] == str(ir_input.resolve())


def test_seal_emits_one_fail_closed_report_per_universe_security(tmp_path: Path) -> None:
    universe = tmp_path / "universe.csv"
    _csv(
        universe,
        [
            {
                "stock_code": ticker,
                "name": f"회사{ticker}",
                "market": "KOSPI",
                "security_type": "COMMON",
                "price_as_of": "2026-08-14T16:00:00+09:00",
                "price_source": "FINANCEDATA_MARCAP_PINNED",
            }
            for ticker in ("000001", "000002")
        ],
        ["stock_code", "name", "market", "security_type", "price_as_of", "price_source"],
    )
    manifest = tmp_path / "manifest.csv"
    _csv(
        manifest,
        [{"ticker": "000001", "filing_ticker": "000001", "dcf_assumptions": ""}],
        ["ticker", "filing_ticker", "dcf_assumptions"],
    )
    audit = tmp_path / "dcf-audit.csv"
    _csv(audit, [], ["stock_code"])
    exclusions = tmp_path / "exclusions.csv"
    _csv(
        exclusions,
        [{"stock_code": "000002", "name": "회사000002", "reason": "NO_PERIODIC_PIT_FILING"}],
        ["stock_code", "name", "reason"],
    )
    output = tmp_path / "output"
    current.write_json(
        output / "corpus-manifest.json",
        {"pack_ids": ["KR-2026-08-18-000001"]},
    )
    current.write_json(
        output / "security-map.json",
        [
            {
                "ticker": "000001",
                "filing_ticker": "000001",
                "pack_id": "KR-2026-08-18-000001",
                "status": "PIT_DOCUMENTS_AVAILABLE",
            },
            {
                "ticker": "000002",
                "filing_ticker": "",
                "pack_id": "",
                "status": "NO_PERIODIC_PIT_FILING",
            },
        ],
    )

    current.seal(
        output=output,
        universe_path=universe,
        dart_manifest_path=manifest,
        dcf_audit_path=audit,
        exclusions_path=exclusions,
        cutoff=CUTOFF,
    )

    coverage = current.read_json(output / "coverage.json")
    assert coverage["report_count"] == 2
    assert coverage["all_universe_securities_have_report"] is True
    assert (output / "reports" / "000001" / "report.json").is_file()
    missing = current.read_json(output / "reports" / "000002" / "report.json")
    assert missing["status"] == "NO_PERIODIC_PIT_FILING"
    assert missing["evidence_overlay"]["action"] == "FAIL_CLOSED"


def test_llm_stage_fails_closed_when_preprocess_remains_partial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    current.write_json(
        tmp_path / "corpus-manifest.json",
        {"pack_ids": ["P1", "P2"]},
    )
    monkeypatch.setattr(current, "_transport", lambda: object())
    monkeypatch.setattr(current, "preprocess", lambda **_kwargs: None)

    try:
        current.run_llm(output=tmp_path, batch_size=2, workers=1)
    except RuntimeError as exc:
        assert "preprocess remained incomplete" in str(exc)
    else:
        raise AssertionError("partial preprocessing must fail closed")


def test_current_validation_keeps_only_anonymization_stable_claims() -> None:
    excerpt = HistoricalExcerpt.create(
        unit_id="U_1",
        source_id="DART-1",
        source_role=HistoricalSourceRole.DART_ORIGINAL,
        available_at=datetime.fromisoformat("2026-08-14T00:00:00+09:00"),
        text="고객은 장기 계약을 갱신하고 있으며 가격 조정 조항을 적용받는다.",
    )
    stable = HistoricalClaim(
        judgment_id="P:J01",
        axis=HistoricalEvidenceAxis.SWITCHING_COST,
        direction=HistoricalEvidenceDirection.SUPPORTIVE,
        claim="장기 계약 갱신은 전환 마찰을 시사한다.",
        unit_id="U_1",
        exact_quote="고객은 장기 계약을 갱신하고 있으며",
        confidence=0.7,
    )
    unstable = stable.model_copy(
        update={
            "judgment_id": "P:J02",
            "axis": HistoricalEvidenceAxis.PRICING_POWER,
            "claim": "가격 조정 조항이 있다.",
            "exact_quote": "가격 조정 조항을 적용받는다",
        }
    )
    original = HistoricalPackAssessment(
        pack_id="P",
        claims=[stable, unstable],
        future_trap_answer=HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE,
    )
    anonymous = HistoricalPackAssessment(
        pack_id="P",
        claims=[stable.model_copy(update={"confidence": 0.72})],
        future_trap_answer=HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE,
    )
    entailment = HistoricalEntailmentBatch(
        decisions=[
            HistoricalEntailmentDecision(
                judgment_id="P:J01",
                verdict=HistoricalEntailment.ENTAILED,
            ),
            HistoricalEntailmentDecision(
                judgment_id="P:J02",
                verdict=HistoricalEntailment.ENTAILED,
            ),
        ]
    )

    claims, audit = current.validate_current_assessment(
        cutoff=CUTOFF,
        excerpts=[excerpt],
        original=original,
        anonymized=anonymous,
        entailment=entailment,
    )

    assert [claim.judgment_id for claim in claims] == ["P:J01"]
    assert audit["anonymization_instability_detected"] is True
    assert audit["discarded_original_claim_count"] == 1


def test_current_validation_rejects_same_axis_from_different_evidence_units() -> None:
    excerpts = [
        HistoricalExcerpt.create(
            unit_id="U_1",
            source_id="DART-1",
            source_role=HistoricalSourceRole.DART_ORIGINAL,
            available_at=datetime.fromisoformat("2026-08-14T00:00:00+09:00"),
            text="품목 허가 전에는 판매와 대금 수령이 없다고 공시했다.",
        ),
        HistoricalExcerpt.create(
            unit_id="U_2",
            source_id="DART-1",
            source_role=HistoricalSourceRole.DART_ORIGINAL,
            available_at=datetime.fromisoformat("2026-08-14T00:00:00+09:00"),
            text="전환가격의 최저 조정 한도는 발행 시점 가격의 85%이다.",
        ),
    ]
    original_claim = HistoricalClaim(
        judgment_id="P:J01",
        axis=HistoricalEvidenceAxis.FRAGILITY,
        direction=HistoricalEvidenceDirection.EROSIVE,
        claim="상업화 전 매출 부재 위험이 있다.",
        unit_id="U_1",
        exact_quote="품목 허가 전에는 판매와 대금 수령이 없다고 공시했다.",
        confidence=0.9,
    )
    anonymous_claim = original_claim.model_copy(
        update={
            "unit_id": "U_2",
            "exact_quote": "전환가격의 최저 조정 한도는 발행 시점 가격의 85%이다.",
            "confidence": 0.91,
        }
    )
    original = HistoricalPackAssessment(
        pack_id="P",
        claims=[original_claim],
        future_trap_answer=HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE,
    )
    anonymous = HistoricalPackAssessment(
        pack_id="P",
        claims=[anonymous_claim],
        future_trap_answer=HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE,
    )
    entailment = HistoricalEntailmentBatch(
        decisions=[
            HistoricalEntailmentDecision(
                judgment_id="P:J01",
                verdict=HistoricalEntailment.ENTAILED,
            )
        ]
    )

    claims, audit = current.validate_current_assessment(
        cutoff=CUTOFF,
        excerpts=excerpts,
        original=original,
        anonymized=anonymous,
        entailment=entailment,
    )

    assert claims == []
    assert audit["stable_claim_count"] == 0
    assert audit["anonymization_instability_detected"] is True


def test_current_validation_matches_the_same_quote_after_identity_masking() -> None:
    excerpt = HistoricalExcerpt.create(
        unit_id="U_1",
        source_id="DART-1",
        source_role=HistoricalSourceRole.DART_ORIGINAL,
        available_at=datetime.fromisoformat("2026-08-14T00:00:00+09:00"),
        text="테스트회사는 장기 공급 계약을 반복해서 갱신하고 있다.",
    )
    original_claim = HistoricalClaim(
        judgment_id="P:J01",
        axis=HistoricalEvidenceAxis.SWITCHING_COST,
        direction=HistoricalEvidenceDirection.SUPPORTIVE,
        claim="장기 공급 계약이 반복 갱신된다.",
        unit_id="U_1",
        exact_quote="테스트회사는 장기 공급 계약을 반복해서 갱신하고 있다.",
        confidence=0.8,
    )
    anonymous_claim = original_claim.model_copy(
        update={
            "exact_quote": "COMPANY_A는 장기 공급 계약을 반복해서 갱신하고 있다.",
            "confidence": 0.81,
        }
    )
    original = HistoricalPackAssessment(
        pack_id="P",
        claims=[original_claim],
        future_trap_answer=HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE,
    )
    anonymous = HistoricalPackAssessment(
        pack_id="P",
        claims=[anonymous_claim],
        future_trap_answer=HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE,
    )
    entailment = HistoricalEntailmentBatch(
        decisions=[
            HistoricalEntailmentDecision(
                judgment_id="P:J01",
                verdict=HistoricalEntailment.ENTAILED,
            )
        ]
    )

    claims, audit = current.validate_current_assessment(
        cutoff=CUTOFF,
        excerpts=[excerpt],
        original=original,
        anonymized=anonymous,
        entailment=entailment,
        issuer_name="테스트회사",
        ticker="000001",
    )

    assert [claim.judgment_id for claim in claims] == ["P:J01"]
    assert audit["stable_claim_count"] == 1


def test_current_validation_discards_only_the_unentailed_claim() -> None:
    excerpt = HistoricalExcerpt.create(
        unit_id="U_1",
        source_id="DART-1",
        source_role=HistoricalSourceRole.DART_ORIGINAL,
        available_at=datetime.fromisoformat("2026-08-14T00:00:00+09:00"),
        text="장기 계약이 유지되지만 원재료 가격 상승 위험도 함께 공시한다.",
    )
    kept = HistoricalClaim(
        judgment_id="P:J01",
        axis=HistoricalEvidenceAxis.SWITCHING_COST,
        direction=HistoricalEvidenceDirection.SUPPORTIVE,
        claim="장기 계약이 유지된다.",
        unit_id="U_1",
        exact_quote="장기 계약이 유지되지만",
        confidence=0.7,
    )
    rejected = HistoricalClaim(
        judgment_id="P:J02",
        axis=HistoricalEvidenceAxis.FRAGILITY,
        direction=HistoricalEvidenceDirection.EROSIVE,
        claim="원재료 가격 상승 위험이 있다.",
        unit_id="U_1",
        exact_quote="원재료 가격 상승 위험도 함께 공시한다",
        confidence=0.8,
    )
    original = HistoricalPackAssessment(
        pack_id="P",
        claims=[kept, rejected],
        future_trap_answer=HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE,
    )
    anonymous = original.model_copy(
        update={
            "claims": [
                kept.model_copy(update={"confidence": 0.72}),
                rejected.model_copy(update={"confidence": 0.78}),
            ]
        }
    )
    entailment = HistoricalEntailmentBatch(
        decisions=[
            HistoricalEntailmentDecision(
                judgment_id="P:J01",
                verdict=HistoricalEntailment.ENTAILED,
            ),
            HistoricalEntailmentDecision(
                judgment_id="P:J02",
                verdict=HistoricalEntailment.NOT_ENTAILED,
            ),
        ]
    )

    claims, audit = current.validate_current_assessment(
        cutoff=CUTOFF,
        excerpts=[excerpt],
        original=original,
        anonymized=anonymous,
        entailment=entailment,
    )

    assert [claim.judgment_id for claim in claims] == ["P:J01"]
    assert audit["discarded_entailment_claim_count"] == 1
    assert audit["validated_claim_count"] == 1


def test_current_unit_selection_is_dart_first_and_source_balanced() -> None:
    def excerpt(index: int, role: HistoricalSourceRole) -> HistoricalExcerpt:
        return HistoricalExcerpt.create(
            unit_id=f"U_{index}",
            source_id=f"S_{role.value}",
            source_role=role,
            available_at=datetime.fromisoformat("2026-08-14T00:00:00+09:00"),
            text=f"충분히 긴 테스트 증거 문장 {index} 고객 계약 가격 경쟁 위험 반복 매출",
        )

    values = [excerpt(index, HistoricalSourceRole.DART_ORIGINAL) for index in range(20)]
    values += [excerpt(100 + index, HistoricalSourceRole.IR) for index in range(5)]
    values += [excerpt(200 + index, HistoricalSourceRole.COMPANY_ANALYST) for index in range(5)]

    selected = current.deterministic_current_units(values)

    assert len(selected) == 16
    assert sum(item.source_role == HistoricalSourceRole.DART_ORIGINAL for item in selected) == 12
    assert sum(item.source_role == HistoricalSourceRole.IR for item in selected) == 2
    assert sum(item.source_role == HistoricalSourceRole.COMPANY_ANALYST for item in selected) == 2


def test_industry_routing_requires_explicit_cutoff_keywords() -> None:
    semiconductor = HistoricalExcerpt.create(
        unit_id="U_SEMI",
        source_id="DART-1",
        source_role=HistoricalSourceRole.DART_ORIGINAL,
        available_at=datetime.fromisoformat("2026-08-14T00:00:00+09:00"),
        text="회사는 반도체 메모리 웨이퍼 공정 장비를 공급한다.",
    )
    generic = HistoricalExcerpt.create(
        unit_id="U_GENERIC",
        source_id="DART-2",
        source_role=HistoricalSourceRole.DART_ORIGINAL,
        available_at=datetime.fromisoformat("2026-08-14T00:00:00+09:00"),
        text="회사는 다양한 고객에게 제품과 서비스를 공급한다.",
    )

    assert current.deterministic_industry_codes(
        [semiconductor], allowed_codes={"159", "066"}
    ) == ["159"]
    assert current.deterministic_industry_codes(
        [generic], allowed_codes={"159", "066"}
    ) == []


def test_classification_consensus_keeps_only_exact_repeated_claims(
    tmp_path: Path,
) -> None:
    stable = HistoricalClaim(
        judgment_id="old-1",
        axis=HistoricalEvidenceAxis.SWITCHING_COST,
        direction=HistoricalEvidenceDirection.SUPPORTIVE,
        claim="반복된 claim",
        unit_id="U_1",
        exact_quote="동일하게 반복되는 정확한 원문 인용",
        confidence=0.8,
    )
    unstable = stable.model_copy(
        update={
            "judgment_id": "old-2",
            "axis": HistoricalEvidenceAxis.PRICING_POWER,
            "claim": "한 vote에만 있는 claim",
        }
    )
    for vote, claims in (
        (1, [stable, unstable]),
        (2, [stable.model_copy(update={"confidence": 0.7})]),
    ):
        current.write_json(
            tmp_path
            / "llm"
            / "original-votes"
            / f"vote-{vote:02d}"
            / "P.json",
            HistoricalPackAssessment(
                pack_id="P",
                claims=claims,
                future_trap_answer=HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE,
            ).model_dump(mode="json"),
        )

    current.consolidate_classification_votes(
        pack_ids=["P"],
        output=tmp_path,
        anonymized=False,
    )

    payload = current.read_json(tmp_path / "llm" / "original" / "P.json")
    assert len(payload["claims"]) == 1
    assert payload["claims"][0]["axis"] == "SWITCHING_COST"
    assert payload["claims"][0]["confidence"] == 0.7
    assert payload["classification_consensus"]["vote_claim_counts"] == [2, 1]
