from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from moatrader.evidence.models import (
    AtomicEvidenceExtraction,
    AtomicMoatRole,
    EvidenceDirection,
    EvidenceType,
)
from scripts.audit_ir_visual_ablation import (
    CoverageJudgment,
    _candidate_source_text,
    _gold_route,
    _lane_report,
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
