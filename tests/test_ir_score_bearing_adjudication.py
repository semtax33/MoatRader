from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from scripts.audit_ir_score_bearing_adjudication import (
    _expected_classifier_route,
    _manual_signature,
    _strict_majority,
)


def test_score_bearing_adjudication_gold_is_complete_and_source_grounded() -> None:
    root = Path(__file__).resolve().parents[1]
    gold = json.loads(
        (root / "docs" / "ir-score-bearing-adjudication-v1.json").read_text(encoding="utf-8")
    )

    assert gold["schema_version"] == "ir-score-bearing-adjudication/1"
    assert gold["selection_uses_return_data"] is False
    assert len(gold["claims"]) == 9
    assert Counter(claim["adjudication_class"] for claim in gold["claims"]) == {
        "A": 7,
        "B": 1,
        "C": 1,
    }
    assert all((root / claim["source_pdf"]).is_file() for claim in gold["claims"])
    assert all((root / claim["rendered_page"]).is_file() for claim in gold["claims"])
    assert all(claim["canonical_classifier_text"] for claim in gold["claims"])
    assert all(
        claim["score_bearing_route"] and claim["score_bearing_component"]
        if claim["adjudication_class"] == "A"
        else claim["score_bearing_route"] is None and claim["score_bearing_component"] is None
        for claim in gold["claims"]
    )


def test_manual_and_classifier_routes_fail_closed_for_B_and_C() -> None:
    root = Path(__file__).resolve().parents[1]
    claims = json.loads(
        (root / "docs" / "ir-score-bearing-adjudication-v1.json").read_text(encoding="utf-8")
    )["claims"]

    redcap = next(claim for claim in claims if claim["claim_id"] == "VC_038390_03_OPERATING_CHANGE")
    avaco = next(claim for claim in claims if claim["claim_id"] == "VC_083930_09_SPUTTER_INSTALLED_BASE")
    gtf = next(claim for claim in claims if claim["claim_id"] == "VC_204620_06_TAX_REFUND_SHARE")

    assert _manual_signature(redcap) == ("C", "NONE", "OTHER")
    assert _expected_classifier_route(redcap) == ("NONE", False, "OTHER", "NEUTRAL")
    assert _manual_signature(avaco) == ("B", "NONE", "OTHER")
    assert _expected_classifier_route(avaco) == ("NONE", False, "OTHER", "NEUTRAL")
    assert _manual_signature(gtf) == ("A", "OUTCOME", "MARKET_SHARE")
    assert _expected_classifier_route(gtf) == (
        "OUTCOME",
        True,
        "MARKET_SHARE",
        "MOAT_POSITIVE",
    )


def test_strict_majority_rejects_a_three_way_split() -> None:
    assert _strict_majority([("A",), ("A",), ("B",)]) == ("A",)
    assert _strict_majority([("A",), ("B",), ("C",)]) is None
