from __future__ import annotations

import re
from difflib import SequenceMatcher

from moatrader.canonical.models import SourceType, StatementType
from moatrader.evidence.models import (
    EvidenceCard,
    EvidenceCluster,
    EvidenceDirection,
    EvidenceRelation,
    EvidenceRelationType,
)


_NUMBER_RE = re.compile(r"[-+]?\d[\d,.]*(?:%|개월|년|월|일)?")


def calibrate_card_reliability(card: EvidenceCard) -> EvidenceCard:
    caps = {
        StatementType.DISCLOSED_FACT: 0.95,
        StatementType.DERIVED_METRIC: 0.90,
        StatementType.MANAGEMENT_CLAIM: 0.60,
        StatementType.ANALYST_INTERPRETATION: 0.65,
        StatementType.INDUSTRY_INTERPRETATION: 0.65,
        StatementType.FORECAST: 0.50,
    }
    cap = caps[card.statement_type]
    if card.source_type == SourceType.IR and card.statement_type in {
        StatementType.MANAGEMENT_CLAIM,
        StatementType.FORECAST,
    }:
        cap -= 0.05
    return card.model_copy(update={"reliability": min(card.reliability, round(cap, 2))})


def build_evidence_relations(
    cards: list[EvidenceCard],
    *,
    duplicate_threshold: float = 0.92,
    contradiction_threshold: float = 0.72,
    update_threshold: float = 0.82,
    support_threshold: float = 0.55,
    weakens_threshold: float = 0.45,
) -> list[EvidenceRelation]:
    relations: list[EvidenceRelation] = []
    normalized = [re.sub(r"\s+", " ", card.fact).strip().casefold() for card in cards]
    for index, card in enumerate(cards):
        for prior_index in range(index):
            prior = cards[prior_index]
            if card.evidence_type != prior.evidence_type:
                continue
            similarity = SequenceMatcher(None, normalized[index], normalized[prior_index]).ratio()
            numbers = _NUMBER_RE.findall(normalized[index])
            prior_numbers = _NUMBER_RE.findall(normalized[prior_index])
            relation = None
            if card.direction == prior.direction and similarity >= duplicate_threshold and numbers == prior_numbers:
                relation = EvidenceRelationType.DUPLICATES
            elif card.direction == prior.direction and similarity >= update_threshold and numbers != prior_numbers:
                relation = EvidenceRelationType.UPDATES
            elif (
                card.direction != prior.direction
                and EvidenceDirection.NEUTRAL not in {card.direction, prior.direction}
                and similarity >= contradiction_threshold
            ):
                relation = EvidenceRelationType.CONTRADICTS
            elif (
                card.direction != prior.direction
                and EvidenceDirection.NEUTRAL not in {card.direction, prior.direction}
                and similarity >= weakens_threshold
            ):
                relation = EvidenceRelationType.WEAKENS
            elif card.direction == prior.direction and similarity >= support_threshold:
                relation = EvidenceRelationType.SUPPORTS
            if relation is not None:
                relations.append(
                    EvidenceRelation(
                        from_evidence_id=card.evidence_id,
                        to_evidence_id=prior.evidence_id,
                        relation=relation,
                    )
                )
                break
    return relations


def cluster_duplicate_evidence(
    cards: list[EvidenceCard],
    relations: list[EvidenceRelation],
) -> list[EvidenceCluster]:
    """Return one deterministic representative plus all duplicate supporting IDs."""
    card_by_id = {card.evidence_id: card for card in cards}
    order = {card.evidence_id: index for index, card in enumerate(cards)}
    if len(card_by_id) != len(cards):
        raise ValueError("evidence IDs must be unique before clustering")
    parent = {evidence_id: evidence_id for evidence_id in card_by_id}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for relation in relations:
        if relation.relation != EvidenceRelationType.DUPLICATES:
            continue
        if relation.from_evidence_id not in card_by_id or relation.to_evidence_id not in card_by_id:
            raise ValueError("duplicate relation references missing evidence")
        union(relation.from_evidence_id, relation.to_evidence_id)

    groups: dict[str, list[EvidenceCard]] = {}
    for card in cards:
        groups.setdefault(find(card.evidence_id), []).append(card)
    clusters: list[EvidenceCluster] = []
    for group in groups.values():
        canonical = max(
            group,
            key=lambda card: (
                card.reliability * card.strength,
                card.reliability,
                card.strength,
                -order[card.evidence_id],
            ),
        )
        supporters = [card.evidence_id for card in group if card.evidence_id != canonical.evidence_id]
        clusters.append(
            EvidenceCluster(
                canonical_evidence_id=canonical.evidence_id,
                supporting_evidence_ids=supporters,
            )
        )
    return sorted(clusters, key=lambda cluster: order[cluster.canonical_evidence_id])
