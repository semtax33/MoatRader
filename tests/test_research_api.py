from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi.testclient import TestClient

from moatrader.api.app import create_app
from moatrader.api.repository import ResearchArtifactRepository
from moatrader.api.service import FundamentalResearchService


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _fixture_run(root: Path, run_name: str, *, model: str, evidence_count: int = 2) -> Path:
    directory = root / "backtests" / "sample" / "runs" / run_name / "companies" / "TST"
    assumptions = {
        "method": "FCFF",
        "base_period": "2025FY",
        "base_revenue": "1000000000",
        "revenue_growth": ["0.08", "0.07", "0.06", "0.04", "0.02"],
        "ebit_margin": ["0.15", "0.155", "0.16", "0.165", "0.17"],
        "tax_rate": "0.24",
        "depreciation_pct_revenue": "0.03",
        "capex_pct_revenue": "0.05",
        "nwc_pct_revenue": "0.04",
        "wacc": "0.09",
        "terminal_growth": "0.02",
        "net_debt": "100000000",
        "diluted_shares": "1000000",
        "assumption_sources": {
            "revenue_growth": ["ANNUAL_HISTORY"],
            "ebit_margin": ["PIT_TTM"],
            "wacc": ["SIZE_BUCKET"],
            "terminal_growth": ["POLICY_DEFAULT"],
        },
        "assumption_types": {
            "base_revenue": "DETERMINISTIC",
            "revenue_growth": "MODEL_INFERENCE",
            "ebit_margin": "MODEL_INFERENCE",
            "tax_rate": "DEFAULT",
            "depreciation_pct_revenue": "MODEL_INFERENCE",
            "capex_pct_revenue": "MODEL_INFERENCE",
            "nwc_pct_revenue": "MODEL_INFERENCE",
            "wacc": "DETERMINISTIC",
            "terminal_growth": "DEFAULT",
            "net_debt": "DETERMINISTIC",
            "diluted_shares": "DETERMINISTIC",
        },
        "provenance_warnings": ["test warning"],
    }
    moat = {
        "economic_moat_score": 6.4,
        "durability": "MEDIUM_HIGH",
        "model_confidence": 0.82,
        "document_coverage": {"moat_evidence_coverage": 0.9},
        "mechanisms": [
            {
                "evidence_type": "SWITCHING_COST",
                "score": 7.2,
                "evidence_ids": ["E_POS"],
            }
        ],
        "counterevidence_ids": ["E_NEG"],
    }
    result = {
        "ticker": "TST",
        "issuer_id": "issuer-1",
        "issuer_name": "테스트기업",
        "status": "COMPLETE",
        "run_signature": f"{run_name:0<64}"[:64],
        "source_document_ids": ["20260201000001"],
        "evidence_count": evidence_count,
        "moat_score": moat,
        "dcf": {
            "fair_value_per_share": "1500",
            "assumption_confidence": "0.64",
        },
        "current_price": "1200",
        "price_as_of": "2026-02-27T16:00:00+09:00",
        "valuation_as_of": "2026-02-28T23:59:59+09:00",
        "runner_version": "1.0.0",
    }
    dossier = {
        "evidence": [
            {
                "evidence_id": "E_POS",
                "evidence_type": "SWITCHING_COST",
                "fact": "장기 계약이 고객 전환비용을 높인다.",
                "mechanism": ["업무 프로세스 통합"],
                "direction": "MOAT_POSITIVE",
                "strength": 0.8,
                "reliability": 0.95,
                "source_type": "DART",
                "raw_quote": "고객은 장기 계약에 따라 서비스를 이용합니다.",
                "period": "2025년",
                "dcf_links": ["CAP"],
            },
            {
                "evidence_id": "E_NEG",
                "evidence_type": "REGULATORY_BARRIER",
                "fact": "규제 비용이 증가할 수 있다.",
                "mechanism": ["규제 불확실성"],
                "direction": "MOAT_NEGATIVE",
                "strength": 0.7,
                "reliability": 0.9,
                "source_type": "DART",
                "raw_quote": "관련 규제 절차가 진행 중입니다.",
                "period": "2025년",
                "dcf_links": [],
            },
        ]
    }
    snapshot = {
        "series": [
            {
                "concept": "REVENUE",
                "points": [
                    {"period": "2025-12-31", "period_basis": "FY", "value": "1000000000", "unit": "KRW"}
                ],
            }
        ],
        "breakdowns": [
            {
                "concept": "REVENUE",
                "dimension": "- 플랫폼",
                "points": [
                    {"period": "2025-12-31", "period_basis": "FY", "value": "700000000", "unit": "KRW"}
                ],
            },
            {
                "concept": "REVENUE",
                "dimension": "- 서비스",
                "points": [
                    {"period": "2025-12-31", "period_basis": "FY", "value": "300000000", "unit": "KRW"}
                ],
            },
        ],
        "derived_metrics": [
            {"name": "REVENUE_CAGR", "value": "0.08", "unit": "RATIO"},
            {"name": "EBIT_MARGIN", "period": "2025-12-31", "value": "0.15", "unit": "RATIO"},
            {"name": "FCF_MARGIN", "period": "2025-12-31", "value": "0.09", "unit": "RATIO"},
        ],
    }
    manifest = {
        "run_id": run_name,
        "model": model,
        "created_at": "2026-03-01T00:00:00Z",
        "evidence_cutoff": "2026-02-28T23:59:59+09:00",
        "prompt_version": "test/1",
        "parser_version": "test/1",
    }
    ledger = {
        "records": [
            {
                "evidence_id": evidence_id,
                "source_document_id": "20260201000001",
                "source_available_at": "2026-02-01T23:59:59+09:00",
            }
            for evidence_id in ("E_POS", "E_NEG")
        ]
    }
    for name, payload in (
        ("result.json", result),
        ("dossier.json", dossier),
        ("moat-score.json", moat),
        ("dcf-assumptions.json", assumptions),
        ("financial-snapshot.json", snapshot),
        ("run-manifest.json", manifest),
        ("evidence-ledger-snapshot.json", ledger),
    ):
        _write_json(directory / name, payload)
    return directory


def test_research_api_exposes_grounded_product_contract(tmp_path: Path) -> None:
    _fixture_run(tmp_path, "luna", model="gpt-5.6-luna")
    client = TestClient(create_app(data_root=tmp_path))

    response = client.get("/api/research/TST")

    assert response.status_code == 200
    payload = response.json()
    assert payload["meta"]["schema_version"] == "fundamental-research/1.0"
    assert payload["company"]["revenue_mix"][0]["name"] == "플랫폼"
    assert payload["moat"]["axes"][0]["status"] == "SUPPORTED"
    assert len(payload["valuation"]["scenarios"]) == 3
    assert payload["market_expectations"]["evaluated_point_count"] > 100
    assert payload["market_expectations"]["identification_caveat"]
    assert payload["evidence"][0]["source"]["url"].endswith("20260201000001")
    assert "종목 추천" in payload["decision_support"]["use_boundary"]


def test_research_api_keeps_report_available_when_reverse_dcf_has_no_positive_equity_value(
    tmp_path: Path,
) -> None:
    directory = _fixture_run(tmp_path, "distressed", model="gpt-5.6-luna")
    assumptions_path = directory / "dcf-assumptions.json"
    assumptions = json.loads(assumptions_path.read_text(encoding="utf-8"))
    assumptions["net_debt"] = "1000000000000"
    _write_json(assumptions_path, assumptions)
    client = TestClient(create_app(data_root=tmp_path))

    response = client.get("/api/research/TST")

    assert response.status_code == 200
    market_expectations = response.json()["market_expectations"]
    assert market_expectations["status"] == "UNAVAILABLE"
    assert market_expectations["evaluated_point_count"] == 0
    assert market_expectations["drivers"] == []
    assert "식별하지 못했습니다" in market_expectations["headline"]


def test_research_api_ignores_incomplete_optional_previous_report(tmp_path: Path) -> None:
    current_directory = _fixture_run(tmp_path, "current", model="gpt-5.6-luna")
    current_result = json.loads(
        (current_directory / "result.json").read_text(encoding="utf-8")
    )
    previous_result = {
        **current_result,
        "run_signature": "previous".ljust(64, "0"),
        "valuation_as_of": "2026-01-31T23:59:59+09:00",
        "price_as_of": "2026-01-30T16:00:00+09:00",
    }
    incomplete_directory = (
        tmp_path
        / "backtests"
        / "sample"
        / "runs"
        / "previous"
        / "companies"
        / "TST"
    )
    _write_json(incomplete_directory / "result.json", previous_result)
    client = TestClient(create_app(data_root=tmp_path))

    response = client.get("/api/research/TST")

    assert response.status_code == 200
    assert response.json()["thesis"]["changes_since_previous"] == []


def test_catalog_and_not_found_are_explicit(tmp_path: Path) -> None:
    _fixture_run(tmp_path, "luna", model="gpt-5.6-luna")
    client = TestClient(create_app(data_root=tmp_path))

    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/api/research").json()["reports"][0]["ticker"] == "TST"
    missing = client.get("/api/research/MISSING")
    assert missing.status_code == 404
    assert "not found" in missing.json()["detail"]


def test_repository_prefers_luna_evidence_run_on_same_pit_date(tmp_path: Path) -> None:
    _fixture_run(tmp_path, "deterministic", model="deterministic-python", evidence_count=20)
    expected = _fixture_run(tmp_path, "luna", model="gpt-5.6-luna", evidence_count=2)

    selected = ResearchArtifactRepository(tmp_path).load("TST")

    assert selected.directory == expected.resolve()


def test_current_all_security_report_api_includes_fail_closed_securities(
    tmp_path: Path,
) -> None:
    report_dir = (
        tmp_path
        / "backtests"
        / "kr-all-research-20260818-v2"
        / "research-reports"
        / "2026-08-18"
        / "reports"
        / "005935"
    )
    _write_json(
        report_dir / "report.json",
        {
            "schema_version": "kr-all-current-reports/1",
            "ticker": "005935",
            "name": "삼성전자우",
            "market": "KOSPI",
            "security_type": "PREFERRED",
            "cutoff": "2026-08-18T23:59:59+09:00",
            "status": "COMPLETE_DATA_ONLY",
            "valuation": {"status": "NOT_APPLICABLE_OR_UNAVAILABLE"},
            "evidence_overlay": {"action": "FAIL_CLOSED"},
        },
    )
    client = TestClient(create_app(data_root=tmp_path))

    catalog = client.get("/api/current-research")
    detail = client.get("/api/current-research/005935")

    assert catalog.status_code == 200
    assert catalog.json()["count"] == 1
    assert catalog.json()["reports"][0]["status"] == "COMPLETE_DATA_ONLY"
    assert detail.status_code == 200
    assert detail.json()["ticker"] == "005935"


def test_research_api_distinguishes_unsupported_current_report_from_missing(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "research-reports" / "2026-08-18" / "reports" / "006220"
    _write_json(
        report_dir / "report.json",
        {
            "ticker": "006220",
            "name": "제주은행",
            "cutoff": "2026-08-18T23:59:59+09:00",
            "status": "COMPLETE",
            "source_exclusions": ["SPECIAL_COMPANY_DCF_MODEL_MISMATCH"],
            "valuation": {"status": "NOT_APPLICABLE_OR_UNAVAILABLE"},
        },
    )
    client = TestClient(create_app(data_root=tmp_path))

    response = client.get("/api/research/006220")

    assert response.status_code == 400
    assert "보고서는 존재하지만" in response.json()["detail"]
    assert "SPECIAL_COMPANY_DCF_MODEL_MISMATCH" in response.json()["detail"]


def test_research_api_adapts_ready_all_security_report(tmp_path: Path) -> None:
    cutoff = "2026-08-18T23:59:59+09:00"
    report_root = tmp_path / "research-reports" / "2026-08-18"
    report_dir = report_root / "reports" / "035420"
    _write_json(
        report_dir / "report.json",
        {
            "schema_version": "kr-all-current-reports/1",
            "ticker": "035420",
            "name": "NAVER",
            "market": "KOSPI",
            "security_type": "COMMON",
            "cutoff": cutoff,
            "price_as_of": "2026-08-14T16:00:00+09:00",
            "status": "COMPLETE",
            "valuation": {
                "status": "READY",
                "current_price": "228000",
                "fair_value_per_share": "163479.88",
                "assumption_confidence": "0.64",
                "terminal_value_share": "0.71",
            },
            "evidence_overlay": {
                "pack_id": "KR-2026-08-18-035420",
                "validated_claims": [
                    {
                        "judgment_id": "KR-2026-08-18-035420:J01",
                        "axis": "FRAGILITY",
                        "direction": "EROSIVE",
                        "claim": "환율 변동 위험에 노출되어 있다.",
                        "source_id": "20260814002266",
                        "available_at": "2026-08-14T23:59:59+09:00",
                        "exact_quote": "미국달러화 관련 환율 변동 위험에 노출되어 있습니다.",
                        "confidence": 0.99,
                    }
                ],
                "anonymization_audit": {"original_claim_count": 1},
            },
            "model_contract": {"main_model": "gpt-5.6-luna"},
        },
    )
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "report-catalog.csv").write_text(
        "ticker,name,market,security_type,status,filing_ticker,overlay_action,"
        "validated_claim_count,valuation_status,report_json,report_markdown\n"
        "035420,NAVER,KOSPI,COMMON,COMPLETE,035420,POSITION_CAP,1,READY,,\n",
        encoding="utf-8",
    )
    _write_json(
        report_root / "packs" / "selected" / "KR-2026-08-18-035420.json",
        {
            "ticker": "035420",
            "excerpts": [
                {
                    "unit_id": "U_SEGMENT",
                    "source_id": "20260814002266",
                    "source_role": "DART_ORIGINAL",
                    "text": (
                        "가. 매출실적 부문별로는 전년 동기 대비 플랫폼 부문이 "
                        "5.7% 증가, 커머스 부문은 3.2% 감소하였습니다."
                    ),
                },
                {
                    "unit_id": "U_RESULT",
                    "source_id": "20260814002266",
                    "source_role": "DART_ORIGINAL",
                    "text": "당사 매출액은 10% 증가했고 영업이익은 20% 증가했습니다.",
                },
            ],
            "source_assignment": {
                "dart_documents": [
                    {
                        "source_id": "20260814002266",
                        "available_at": "2026-08-14T23:59:59+09:00",
                    }
                ]
            },
        },
    )
    assumptions = {
        "method": "FCFF",
        "base_period": "2026H1",
        "base_revenue": "12962954994050",
        "revenue_growth": ["0.10", "0.08", "0.06", "0.04", "0.02"],
        "ebit_margin": ["0.17", "0.17", "0.175", "0.175", "0.18"],
        "tax_rate": "0.24",
        "depreciation_pct_revenue": "0.05",
        "capex_pct_revenue": "0.07",
        "nwc_pct_revenue": "0.03",
        "wacc": "0.095",
        "terminal_growth": "0.02",
        "net_debt": "-1184631171506",
        "diluted_shares": "156977585",
        "assumption_sources": {},
        "assumption_types": {},
        "provenance_warnings": [],
    }
    date_inputs = tmp_path / "date-inputs" / "2026-08-18"
    _write_json(date_inputs / "assumptions" / "035420.json", assumptions)
    _write_json(
        date_inputs / "dcf-inputs" / "035420.json",
        {
            "ticker": "035420",
            "metrics": {"revenue": "12962954994050", "ebit": "2243349860301"},
            "pit": {"latest_report_period": "2026H1"},
            "annual_history": [
                {"year": 2023, "metrics": {"revenue": "9670643576585"}},
                {"year": 2025, "metrics": {"revenue": "12035007218975"}},
            ],
        },
    )
    # A legacy COMPLETE result without its required detail artifacts must not
    # mask the usable all-security report for the same ticker.
    _write_json(
        tmp_path / "runs" / "incomplete" / "companies" / "035420" / "result.json",
        {
            "ticker": "035420",
            "issuer_name": "NAVER",
            "status": "COMPLETE",
            "valuation_as_of": cutoff,
        },
    )
    client = TestClient(create_app(data_root=tmp_path))

    catalog = client.get("/api/research")
    detail = client.get("/api/research/035420")

    assert catalog.status_code == 200
    assert [item["ticker"] for item in catalog.json()["reports"]] == ["035420"]
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["meta"]["schema_version"] == "fundamental-research/1.0"
    assert payload["company"]["issuer_name"] == "NAVER"
    assert [item["name"] for item in payload["company"]["segment_trends"]] == [
        "플랫폼 부문",
        "커머스 부문",
    ]
    assert payload["company"]["segment_trends"][1]["change_pct"] == -3.2
    assert payload["valuation"]["current_price"] == 228000
    assert payload["valuation"]["route"]["primary_model"] == "FCFF DCF"
    assert payload["valuation"]["economic_value"]["percentile"] is None
    assert payload["moat"]["rating"] == "INSUFFICIENT"
    assert payload["evidence"][0]["source"]["document_id"] == "20260814002266"
    assert any(item["evidence_type"] == "SEGMENT_TREND" for item in payload["evidence"])
    assert payload["thesis"]["context_evidence_ids"]
    assert "검증된 경쟁우위" not in payload["thesis"]["core_thesis"]


def test_current_context_evidence_surfaces_hynix_style_disclosures() -> None:
    cutoff = datetime.fromisoformat("2026-08-18T23:59:59+09:00")
    pack = {
        "excerpts": [
            {
                "unit_id": "U_INDUSTRY",
                "source_id": "20260814003509",
                "source_role": "DART_ORIGINAL",
                "text": "(2) 산업의 성장성 AI 수요가 증가하며 메모리 시장 성장이 이어지고 있습니다.",
            },
            {
                "unit_id": "U_COMPETITION",
                "source_id": "20260814003509",
                "source_role": "DART_ORIGINAL",
                "text": "원가 경쟁력 중심에서 제품 가치 중심으로 경쟁의 패러다임이 전환되고 있습니다.",
            },
            {
                "unit_id": "U_RISK",
                "source_id": "20260814003509",
                "source_role": "DART_ORIGINAL",
                "text": "위험관리의 목적과 정책에 따라 신용위험을 지속적으로 관리하고 있습니다.",
            },
        ],
        "source_assignment": {
            "dart_documents": [
                {
                    "source_id": "20260814003509",
                    "available_at": "2026-08-14T23:59:59+09:00",
                }
            ]
        },
    }

    cards, ledger = FundamentalResearchService._context_evidence(pack, cutoff, "000660")

    assert [item["evidence_type"] for item in cards] == [
        "INDUSTRY_OUTLOOK",
        "COMPETITIVE_DYNAMICS",
        "RISK_MANAGEMENT",
    ]
    assert all(item["direction"] == "NEUTRAL" for item in cards)
    assert len(ledger) == 3


def test_current_context_evidence_falls_back_to_selected_dart_excerpt() -> None:
    cutoff = datetime.fromisoformat("2026-08-18T23:59:59+09:00")
    pack = {
        "excerpts": [
            {
                "unit_id": "U_OTHER",
                "source_id": "20260814000001",
                "source_role": "DART_ORIGINAL",
                "text": "회사는 고객에게 공급하는 장비의 생산 일정을 지속적으로 관리하고 있습니다.",
            }
        ]
    }

    cards, _ = FundamentalResearchService._context_evidence(pack, cutoff, "000001")

    assert len(cards) == 1
    assert cards[0]["evidence_type"] == "DISCLOSURE_CONTEXT"
    assert cards[0]["direction"] == "NEUTRAL"


def test_current_report_repository_honors_as_of_and_rejects_bad_ticker(
    tmp_path: Path,
) -> None:
    report_dir = tmp_path / "research-reports" / "2026-08-18" / "reports" / "000001"
    _write_json(
        report_dir / "report.json",
        {
            "ticker": "000001",
            "cutoff": "2026-08-18T23:59:59+09:00",
            "status": "NO_PERIODIC_PIT_FILING",
        },
    )
    repository = ResearchArtifactRepository(tmp_path)

    assert repository.load_current_report("000001")["status"] == "NO_PERIODIC_PIT_FILING"
    try:
        repository.load_current_report("../000001")
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("path-like ticker must be rejected")
