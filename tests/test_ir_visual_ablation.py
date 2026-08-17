from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from moatrader.evidence.models import (
    AtomicEvidenceExtraction,
    AtomicMoatRole,
    EvidenceDirection,
    EvidenceType,
)
from scripts.audit_ir_visual_ablation import (
    CoverageJudgment,
    PageClaimExtraction,
    _candidate_relations,
    _candidate_source_text,
    _deterministic_minimum_judgment,
    _extractor_user,
    _gold_route,
    _lane_report,
    _merge_page_extractions,
    _preferred_minimum_candidate_index,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_visual_gold_has_frozen_role_and_subtype_for_every_claim() -> None:
    root = Path(__file__).resolve().parents[1]
    gold = json.loads((root / "docs" / "ir-visual-coverage-gold-v1.json").read_text(encoding="utf-8"))

    assert gold["schema_version"] == "ir-visual-coverage-gold/3"
    assert len(gold["claims"]) == 30
    assert all(
        claim.get("gold_role") and claim.get("gold_subtype") and claim.get("gold_rationale")
        for claim in gold["claims"]
    )
    assert Counter(claim["gold_role"] for claim in gold["claims"]) == {
        "NONE": 23,
        "OUTCOME": 5,
        "COUNTER": 1,
        "MECHANISM": 1,
    }
    adjudicated = [claim for claim in gold["claims"] if claim.get("adjudication_class")]
    assert Counter(claim["adjudication_class"] for claim in adjudicated) == {
        "A": 7,
        "B": 1,
        "C": 1,
    }


def test_candidate_source_text_is_deterministic_and_preserves_relations() -> None:
    candidate = {
        "claim": "Issuer share reached 75%.",
        "axis_legend": ["2025", "%"],
        "series_identity": ["Issuer=blue"],
        "numeric_anchors": ["75%"],
        "trend_relations": ["share increased"],
    }

    assert _candidate_source_text(candidate) == (
        "Issuer share reached 75%.\n"
        "Axis/legend: 2025; %\n"
        "Series identity: Issuer=blue\n"
        "Numeric anchors: 75%\n"
        "Trend relations: share increased"
    )


def test_anchor_slot_precedes_easier_supporting_numeric_claim() -> None:
    parsed = {
        "claims": [
            {
                "claim": "Condition has 98% CVS distribution.",
                "source_kind": "CHART",
                "axis_legend": ["CVS distribution", "%"],
                "series_identity": ["Condition"],
                "numeric_anchors": ["98%"],
                "trend_relations": [],
                "observations": [],
            },
            {
                "claim": "HK inno.N states that Condition maintains number-one market share.",
                "source_kind": "TEXT",
                "axis_legend": [],
                "series_identity": ["HK inno.N", "Condition"],
                "numeric_anchors": ["#1"],
                "trend_relations": ["maintains number-one share"],
                "observations": [],
            },
        ],
        "anchor_candidates": [
            {
                "anchor_type": "MARKET_SHARE",
                "component": "HK inno.N's Condition product maintains number-one market share.",
                "source_claim_indices": [1],
                "issuer_link_explicit": True,
                "evidence_basis": ["explicit maintained number-one rank"],
            }
        ],
    }

    candidates = _candidate_relations(parsed, issuer_name="HK inno.N")

    assert candidates[0]["candidate_origin"] == "ANCHOR_SLOT"
    assert candidates[0]["anchor_type"] == "MARKET_SHARE"
    assert "number-one market share" in candidates[0]["claim"]
    assert candidates[-1]["claim"] == "HK inno.N states that Condition maintains number-one market share."


def test_deterministic_relation_builder_computes_margin_sequence_without_llm_inference() -> None:
    observations = [
        ("2020 Q3", 24.7),
        ("2020 Q4", 12.1),
        ("2021 Q1", 30.5),
        ("2021 Q2", 11.8),
        ("2021 Q3", 28.9),
    ]
    parsed = {
        "claims": [
            {
                "claim": "The table reports five quarterly operating margins.",
                "source_kind": "TABLE",
                "axis_legend": [period for period, _value in observations] + ["%"],
                "series_identity": ["Operating margin"],
                "numeric_anchors": [f"{value}%" for _period, value in observations],
                "trend_relations": [],
                "observations": [
                    {
                        "metric": "Operating margin",
                        "series": "Eugene Technology",
                        "period": period,
                        "value": value,
                        "unit": "%",
                    }
                    for period, value in observations
                ],
            }
        ],
        "anchor_candidates": [],
    }

    candidates = _candidate_relations(parsed, issuer_name="Eugene Technology")
    derived = next(row for row in candidates if row["candidate_origin"] == "DETERMINISTIC_RELATION")

    assert derived["anchor_type"] == "MARGIN_STABILITY"
    assert "range is 18.7 percentage points" in derived["claim"]
    assert "largest adjacent change is 18.7 percentage points" in derived["claim"]
    assert "direction reverses 3 times" in derived["claim"]
    assert len(derived["observations"]) == 5


def test_deterministic_relation_builder_supports_non_percent_time_series() -> None:
    parsed = {
        "claims": [
            {
                "claim": "Cold-chain revenue by year.",
                "source_kind": "CHART",
                "observations": [
                    {
                        "metric": "Revenue",
                        "series": "Cold chain",
                        "period": period,
                        "value": value,
                        "unit": "KRW 100m",
                    }
                    for period, value in [
                        ("FY2020", 317.7),
                        ("FY2021", 320.8),
                        ("FY2022", 140.1),
                        ("FY2023", 403.9),
                        ("FY2024", 459.3),
                    ]
                ],
            }
        ],
        "anchor_candidates": [],
    }

    candidates = _candidate_relations(parsed, issuer_name="Issuer")
    derived = next(row for row in candidates if row["candidate_origin"] == "DETERMINISTIC_RELATION")

    assert derived["anchor_type"] is None
    assert "minimum of 140.1KRW 100m in 2022" in derived["claim"]
    assert "maximum of 459.3KRW 100m in 2024" in derived["claim"]
    assert derived["axis_legend"][-1] == "KRW 100m"
    assert len(derived["numeric_anchors"]) == 5


def test_extractor_user_keeps_ocr_bounding_boxes_in_vision_context() -> None:
    item = {
        "issuer_id": "ISSUER",
        "issuer_name": "Issuer",
        "ticker": "000001",
        "source_document_id": "DOC",
        "page": 4,
        "page_text": "parser text",
        "ocr_text": "[OCR bbox=10.0,20.0,30.0,40.0 confidence=0.990] Phase II",
    }

    control = _extractor_user(item, include_ocr=False)
    vision = _extractor_user(item, include_ocr=True)

    assert "PAGE OCR" not in control
    assert "bbox=10.0,20.0,30.0,40.0" in vision


def test_deterministic_process_builder_joins_explicit_ecml_comparison_sides() -> None:
    parsed = {
        "claims": [
            {"claim": "MPL 면역증강제 기술 비교에 따라 EcML과 GLA가 비교된다.", "source_kind": "DIAGRAM"},
            {"claim": "EcML은 직접 생산대장균에서 생산된다.", "source_kind": "DIAGRAM"},
            {"claim": "EcML은 GLA의 단점을 극복했다.", "source_kind": "DIAGRAM"},
            {
                "claim": "GLA는 30단계 이상의 유기합성 및 정제 단계가 필요하고 높은 제조비용이 수반된다.",
                "source_kind": "DIAGRAM",
                "numeric_anchors": ["30단계 이상"],
            },
        ],
        "anchor_candidates": [],
    }

    candidates = _candidate_relations(parsed, issuer_name="EuBiologics")
    derived = next(
        row
        for row in candidates
        if row["candidate_origin"] == "DETERMINISTIC_RELATION"
        and row["anchor_type"] == "COST_ADVANTAGE"
    )

    assert "EuBiologics" in derived["claim"]
    assert "more than 30" in derived["claim"]
    assert "high manufacturing cost" in derived["claim"]
    assert derived["source_claim_indices"] == [0, 1, 2, 3]


def test_complementary_extraction_passes_union_claims_and_remap_anchor_indices() -> None:
    inventory = {
        "claims": [
            {
                "claim": "The page compares EcML with GLA.",
                "source_kind": "DIAGRAM",
                "series_identity": ["EcML", "GLA"],
            },
            {
                "claim": "EcML is produced directly in E. coli.",
                "source_kind": "DIAGRAM",
                "series_identity": ["EcML"],
            },
        ],
        "anchor_candidates": [],
    }
    anchor_audit = {
        "claims": [
            {
                "claim": "GLA requires more than 30 synthesis and purification steps.",
                "source_kind": "DIAGRAM",
                "series_identity": ["GLA"],
                "numeric_anchors": ["more than 30 steps"],
            },
            {
                "claim": "EcML has lower manufacturing cost than GLA.",
                "source_kind": "DIAGRAM",
                "series_identity": ["EcML", "GLA"],
            },
        ],
        "anchor_candidates": [
            {
                "anchor_type": "COST_ADVANTAGE",
                "component": "EuBiologics' EcML has lower manufacturing cost than GLA.",
                "source_claim_indices": [0, 1],
                "issuer_link_explicit": True,
                "evidence_basis": ["30-plus steps versus lower manufacturing cost"],
            }
        ],
    }

    merged = _merge_page_extractions(
        [("inventory", inventory), ("anchor_audit", anchor_audit)]
    )

    assert len(merged.claims) == 4
    assert len(merged.anchor_candidates) == 1
    assert merged.anchor_candidates[0].source_claim_indices == [2, 3]


def test_complementary_extraction_passes_merge_duplicate_observations() -> None:
    repeated_claim = {
        "claim": "The table reports quarterly operating margins.",
        "source_kind": "TABLE",
        "observations": [
            {
                "metric": "Operating margin",
                "series": "Issuer",
                "period": "2021 Q1",
                "value": 30.5,
                "unit": "%",
            }
        ],
    }
    second = json.loads(json.dumps(repeated_claim))
    second["observations"].append(
        {
            "metric": "Operating margin",
            "series": "Issuer",
            "period": "2021 Q2",
            "value": 11.8,
            "unit": "%",
        }
    )

    merged = _merge_page_extractions(
        [
            ("inventory", {"claims": [repeated_claim], "anchor_candidates": []}),
            ("numeric_series", {"claims": [second], "anchor_candidates": []}),
        ]
    )

    assert len(merged.claims) == 1
    assert [value.period for value in merged.claims[0].observations] == ["2021 Q1", "2021 Q2"]


def test_minimum_candidate_reconciliation_uses_full_operating_margin_sequence() -> None:
    candidates = [
        {
            "claim": "Issuer Gross % has five observations: 53%, 48.4%, 49.2%, 46%, 48.7%.",
            "candidate_origin": "DETERMINISTIC_RELATION",
            "anchor_type": "MARGIN_STABILITY",
        },
        {
            "claim": "Issuer OP % has five observations: 24.7%, 12.1%, 30.5%, 11.8%, 28.9%.",
            "candidate_origin": "DETERMINISTIC_RELATION",
            "anchor_type": "MARGIN_STABILITY",
        },
        {
            "claim": "Issuer Net % has five observations: 15.8%, -0.4%, 26.3%, 7.4%, 24.5%.",
            "candidate_origin": "DETERMINISTIC_RELATION",
            "anchor_type": "MARGIN_STABILITY",
        },
    ]
    claim = {
        "gold_subtype": "MARGIN_STABILITY",
        "score_bearing_component": (
            "Issuer operating margin moved 24.7%, 12.1%, 30.5%, 11.8%, and 28.9%."
        ),
    }

    assert _preferred_minimum_candidate_index(candidates, claim) == 1


def test_minimum_candidate_reconciliation_prefers_numeric_market_share_anchor() -> None:
    candidates = [
        {
            "claim": "Global Tax Free is the domestic number-one operator.",
            "candidate_origin": "ANCHOR_SLOT",
            "anchor_type": "MARKET_SHARE",
        },
        {
            "claim": "Global Tax Free has 70-75% domestic tax-refund market share.",
            "candidate_origin": "ANCHOR_SLOT",
            "anchor_type": "MARKET_SHARE",
        },
    ]
    claim = {
        "gold_subtype": "MARKET_SHARE",
        "score_bearing_component": (
            "Global Tax Free is domestic number one with 70-75% market share."
        ),
    }

    assert _preferred_minimum_candidate_index(candidates, claim) == 1


def test_typed_anchor_judgment_requires_all_target_numbers() -> None:
    claim = {
        "gold_subtype": "MARKET_SHARE",
        "score_bearing_component": "Issuer is domestic number one with 70-75% market share.",
        "requirements": ["series_identity", "numeric_recovery", "trend_relation"],
    }
    candidates = [
        {
            "claim": "Issuer has 70-75% domestic market share.",
            "candidate_origin": "ANCHOR_SLOT",
            "anchor_type": "MARKET_SHARE",
            "series_identity": ["Issuer"],
            "numeric_anchors": ["70-75%"],
            "trend_relations": ["domestic leader"],
        }
    ]

    judgment = _deterministic_minimum_judgment(candidates, claim)

    assert judgment is not None
    assert judgment.minimum_component_recovered is True
    assert judgment.best_claim_index == 0
    candidates[0]["claim"] = "Issuer is a domestic leader."
    assert _deterministic_minimum_judgment(candidates, claim) is None


def test_sparse_anchor_filter_rejects_inferred_slots_and_binds_issuer_name() -> None:
    parsed = {
        "claims": [
            {
                "claim": "GTF has 70-75% tax-refund market share.",
                "source_kind": "CHART",
                "numeric_anchors": ["70-75%"],
            }
        ],
        "anchor_candidates": [
            {
                "anchor_type": "MARKET_SHARE",
                "component": "GTF 70-75% tax-refund market share",
                "source_claim_indices": [0],
                "issuer_link_explicit": True,
            },
            {
                "anchor_type": "SWITCHING_COST",
                "component": "Market dominance implies switching costs",
                "source_claim_indices": [0],
                "issuer_link_explicit": True,
            },
        ],
    }

    candidates = _candidate_relations(parsed, issuer_name="Global Tax Free")
    anchors = [row for row in candidates if row["candidate_origin"] == "ANCHOR_SLOT"]

    assert len(anchors) == 1
    assert anchors[0]["anchor_type"] == "MARKET_SHARE"
    assert anchors[0]["claim"].startswith("Global Tax Free:")


def test_anchor_indices_are_zero_based_and_fail_closed() -> None:
    with pytest.raises(ValueError, match="valid zero-based claim indices"):
        PageClaimExtraction.model_validate(
            {
                "claims": [
                    {
                        "claim": "Issuer share is 75%.",
                        "source_kind": "CHART",
                    }
                ],
                "anchor_candidates": [
                    {
                        "anchor_type": "MARKET_SHARE",
                        "component": "Issuer share is 75%.",
                        "source_claim_indices": [1],
                        "issuer_link_explicit": True,
                    }
                ],
            }
        )


def test_visual_extractor_boss_gold_freezes_seven_A_and_two_BC_targets() -> None:
    root = Path(__file__).resolve().parents[1]
    gold = json.loads(
        (root / "docs" / "ir-visual-extractor-boss-v1.json").read_text(encoding="utf-8")
    )

    assert gold["schema_version"] == "ir-visual-extractor-boss/1"
    assert gold["methodology"]["boss_fight"] is True
    assert Counter(claim["adjudication_class"] for claim in gold["claims"]) == {
        "A": 7,
        "B": 1,
        "C": 1,
    }
    assert all(
        bool(claim.get("score_bearing_component")) == (claim["adjudication_class"] == "A")
        for claim in gold["claims"]
    )


def test_gold_route_separates_none_outcome_and_counter() -> None:
    assert _gold_route({"gold_role": "NONE", "gold_subtype": "OTHER"}) == (
        "NONE",
        False,
        "OTHER",
        "NEUTRAL",
    )
    assert _gold_route({"gold_role": "OUTCOME", "gold_subtype": "MARKET_SHARE"}) == (
        "OUTCOME",
        True,
        "MARKET_SHARE",
        "MOAT_POSITIVE",
    )
    assert _gold_route({"gold_role": "COUNTER", "gold_subtype": "MARGIN_STABILITY"}) == (
        "COUNTER",
        True,
        "MARGIN_STABILITY",
        "MOAT_NEGATIVE",
    )


def test_lane_report_does_not_credit_an_unrecovered_none_claim(tmp_path: Path) -> None:
    gold = {
        "claims": [
            {
                "claim_id": "C_SHARE",
                "ticker": "000001",
                "page": 1,
                "requirements": ["series_identity"],
                "gold_role": "OUTCOME",
                "gold_subtype": "MARKET_SHARE",
            },
            {
                "claim_id": "C_NONE",
                "ticker": "000001",
                "page": 2,
                "requirements": ["series_identity"],
                "gold_role": "NONE",
                "gold_subtype": "OTHER",
            },
        ]
    }
    _write(
        tmp_path / "checkpoints" / "judgment" / "vision" / "C_SHARE.json",
        {
            "status": "SUCCESS",
            "parsed": CoverageJudgment(
                best_claim_index=0,
                atomic_claim=True,
                series_identity=True,
                reason="explicit share",
            ).model_dump(mode="json"),
        },
    )
    _write(
        tmp_path / "checkpoints" / "judgment" / "vision" / "C_NONE.json",
        {
            "status": "SUCCESS",
            "parsed": CoverageJudgment(reason="not recovered").model_dump(mode="json"),
        },
    )
    vote = AtomicEvidenceExtraction(
        is_investment_relevant=True,
        moat_role=AtomicMoatRole.OUTCOME,
        evidence_type=EvidenceType.MARKET_SHARE,
        direction=EvidenceDirection.MOAT_POSITIVE,
        fact="issuer share",
    )
    for index in range(1, 7):
        _write(
            tmp_path
            / "checkpoints"
            / "classification"
            / "vision"
            / "C_SHARE"
            / f"vote-{index:02d}.json",
            {"status": "SUCCESS", "parsed": vote.model_dump(mode="json", by_alias=True)},
        )

    result = _lane_report(lane="vision", gold=gold, output=tmp_path, votes=6)
    metrics = result["metrics"]

    assert metrics["atomic_graphical_claim_recall"] == {
        "numerator": 1,
        "denominator": 2,
        "rate": 0.5,
    }
    assert metrics["gold_role_agreement_on_recovered_claims"]["rate"] == 1.0
    assert metrics["gold_role_agreement_on_all_claims"]["rate"] == 0.5
    assert metrics["score_bearing_gold_route_recall"]["rate"] == 1.0
    assert metrics["non_score_bearing_rejection_on_all_claims"]["rate"] == 0.0
