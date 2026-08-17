from __future__ import annotations

from pydantic import Field

from moatrader.canonical.models import ContractModel


EVIDENCE_SENSOR_VERSION = "evidence-sensor/1"
IR_VISUAL_EXTRACTOR_CONTRACT_VERSION = "ir-page-claim-extractor/6"

# Frozen development-gold regression boundaries. These are sensor checks, not
# investment-return tuning targets.
FROZEN_BOSS_GATES = {
    "score_bearing_minimum_component_recall": 1.00,
    "score_bearing_gold_route_recall": 1.00,
    "non_score_bearing_rejection": 1.00,
    "route_repeatability": 0.90,
    "maximum_score_route_conflict": 0.00,
}

FROZEN_FULL_GATES = {
    "full_semantic_preservation": 0.80,
    "atomic_graphical_claim_recall": 0.90,
    "gold_role_agreement_on_recovered_claims": 0.95,
    "score_bearing_gold_route_recall": 0.70,
    "route_repeatability": 0.90,
}


class ExtractionSetReproducibility(ContractModel):
    run_a_claim_count: int = Field(ge=0)
    run_b_claim_count: int = Field(ge=0)
    intersection_count: int = Field(ge=0)
    union_count: int = Field(ge=0)
    extraction_set_jaccard: float | None = Field(default=None, ge=0, le=1)
    score_bearing_gold_count: int = Field(ge=0)
    score_bearing_present_in_both: int = Field(ge=0)
    score_bearing_present_in_either: int = Field(ge=0)
    score_bearing_presence_repeat_rate: float | None = Field(default=None, ge=0, le=1)
    score_bearing_presence_agreement_rate: float | None = Field(default=None, ge=0, le=1)


def extraction_set_reproducibility(
    run_a_claims: set[str],
    run_b_claims: set[str],
    *,
    score_bearing_presence_a: dict[str, bool],
    score_bearing_presence_b: dict[str, bool],
) -> ExtractionSetReproducibility:
    gold_ids = sorted(set(score_bearing_presence_a) | set(score_bearing_presence_b))
    intersection = run_a_claims & run_b_claims
    union = run_a_claims | run_b_claims
    both = sum(
        bool(score_bearing_presence_a.get(item)) and bool(score_bearing_presence_b.get(item))
        for item in gold_ids
    )
    either = sum(
        bool(score_bearing_presence_a.get(item)) or bool(score_bearing_presence_b.get(item))
        for item in gold_ids
    )
    agreement = sum(
        bool(score_bearing_presence_a.get(item)) == bool(score_bearing_presence_b.get(item))
        for item in gold_ids
    )
    return ExtractionSetReproducibility(
        run_a_claim_count=len(run_a_claims),
        run_b_claim_count=len(run_b_claims),
        intersection_count=len(intersection),
        union_count=len(union),
        extraction_set_jaccard=len(intersection) / len(union) if union else None,
        score_bearing_gold_count=len(gold_ids),
        score_bearing_present_in_both=both,
        score_bearing_present_in_either=either,
        score_bearing_presence_repeat_rate=both / either if either else None,
        score_bearing_presence_agreement_rate=agreement / len(gold_ids) if gold_ids else None,
    )
