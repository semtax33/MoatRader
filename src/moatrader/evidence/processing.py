from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher

from moatrader.canonical.ids import stable_id
from moatrader.canonical.models import SourceType, StatementType
from moatrader.evidence.models import (
    DcfLink,
    CanonicalClaimSignature,
    ClaimCluster,
    EconomicScope,
    EvidenceCard,
    EvidenceCluster,
    EvidenceDirection,
    EvidenceRelation,
    EvidenceRelationType,
    EvidenceType,
    ForwardDriverCard,
    ForwardDriverType,
)
from moatrader.semantic.chunker import SemanticChunk


_NUMBER_RE = re.compile(r"[-+]?\d[\d,.]*(?:%|개월|년|배|원)?")
_DEMAND_RE = re.compile(
    r"시장\s*(?:규모|성장)|수요|환자.{0,12}(?:증가|성장)|환자\s*수|방문객|CAGR|TAM|market\s+(?:size|growth|demand)",
    re.IGNORECASE,
)
_SHARE_RE = re.compile(r"점유율|시장\s*점유|market\s*share|share\s+of\s+market", re.IGNORECASE)
_RECURRING_CATEGORY_RE = re.compile(
    r"재시술|반복\s*시술|시술\s*주기|\d+\s*[~-]\s*\d+\s*개월|repeat(?:ed)?\s+treatment",
    re.IGNORECASE,
)
_RETENTION_RE = re.compile(
    r"고객\s*(?:유지|이탈률|재구매율)|갱신률|renewal|retention|churn|switching|전환\s*비용",
    re.IGNORECASE,
)


def calibrate_card_reliability(card: EvidenceCard) -> EvidenceCard:
    source_text = " ".join(filter(None, [card.raw_quote, card.fact]))
    if card.statement_type == StatementType.DISCLOSED_FACT and re.search(
        r"전망|계획|예상|기대|목표|추정|will|expect|plan|target|forecast",
        source_text,
        re.IGNORECASE,
    ):
        card = card.model_copy(update={"statement_type": StatementType.MANAGEMENT_CLAIM})
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
    if not card.raw_quote:
        cap = min(cap, 0.20)
    if card.economic_scope not in {EconomicScope.COMPANY, EconomicScope.SEGMENT}:
        cap = min(cap, 0.50)
    if (
        card.direction == EvidenceDirection.MOAT_POSITIVE
        and card.evidence_type in {
            EvidenceType.SWITCHING_COST,
            EvidenceType.NETWORK_EFFECT,
            EvidenceType.COST_ADVANTAGE,
            EvidenceType.INTANGIBLE_ASSET,
            EvidenceType.SCALE_ADVANTAGE,
            EvidenceType.REGULATORY_BARRIER,
        }
        and not card.mechanism
    ):
        cap = min(cap, 0.30)
    # Reliability is a deterministic policy value, not model
    # self-confidence. Using the model-proposed value made identical evidence
    # affect the public score differently between executions.
    return card.model_copy(update={"reliability": round(max(0.0, cap), 2)})


def grounded_evidence_id(card: EvidenceCard, chunk: SemanticChunk) -> str:
    """Build an ID from the cited source span, never from model labels/text."""

    quote = " ".join((card.raw_quote or "").split())
    atomic_key = chunk.metadata.get("atomic_evidence_key")
    if atomic_key:
        # Node/paragraph order is presentation metadata.  The source document,
        # atomic source text and quote are sufficient for a stable audit ID.
        return stable_id("E", chunk.document_id, atomic_key, quote)
    return stable_id("E", chunk.document_id, *sorted(set(card.node_ids)), quote)


def _canonical_slot(value: str | None, fallback: str) -> str:
    text = unicodedata.normalize("NFKC", value or fallback).casefold()
    text = re.sub(r"[^0-9a-z가-힣一-龥%><=]+", "_", text, flags=re.IGNORECASE)
    return text.strip("_") or fallback


def _metric_value_bucket(card: EvidenceCard) -> str | None:
    if not card.metrics:
        return None
    value = str(card.metrics[0].value).replace(",", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if not match:
        return _canonical_slot(value, "unspecified")
    numeric = abs(float(match.group()))
    if "%" in value:
        if numeric < 5:
            return "pct_lt_5"
        if numeric < 20:
            return "pct_5_20"
        if numeric < 50:
            return "pct_20_50"
        return "pct_ge_50"
    if numeric < 1:
        return "lt_1"
    if numeric < 3:
        return "1_3"
    if numeric < 10:
        return "3_10"
    return "ge_10"


def assign_canonical_claim_identity(card: EvidenceCard, *, issuer_id: str | None) -> EvidenceCard:
    proposed = card.claim_signature
    signature = CanonicalClaimSignature(
        moat_source=card.evidence_type,
        subject=_canonical_slot(
            proposed.subject if proposed else card.segment or card.company_scope,
            "company",
        ),
        predicate=_canonical_slot(
            proposed.predicate if proposed else card.fact,
            "unspecified",
        ),
        direction=card.direction,
        horizon=_canonical_slot(
            proposed.horizon if proposed else card.forecast_horizon or card.period,
            "unspecified",
        ).upper(),
        metric=(
            _canonical_slot(proposed.metric, "unspecified")
            if proposed and proposed.metric
            else (_canonical_slot(card.metrics[0].name, "unspecified") if card.metrics else None)
        ),
        value_bucket=(
            _canonical_slot(proposed.value_bucket, "unspecified")
            if proposed and proposed.value_bucket
            else _metric_value_bucket(card)
        ),
    )
    claim_id = stable_id(
        "CL",
        issuer_id or "UNKNOWN_ISSUER",
        signature.moat_source.value,
        signature.subject,
        signature.predicate,
        signature.direction.value,
        signature.horizon,
        signature.metric or "",
        signature.value_bucket or "",
    )
    return card.model_copy(update={"claim_signature": signature, "claim_id": claim_id})


def build_canonical_claim_set(
    cards: list[EvidenceCard],
    *,
    issuer_id: str | None,
) -> tuple[list[EvidenceCard], list[ClaimCluster]]:
    """Reduce evidence to a commutative, associative and idempotent set."""

    grouped: dict[str, list[EvidenceCard]] = {}
    for source in cards:
        card = assign_canonical_claim_identity(source, issuer_id=issuer_id)
        assert card.claim_id is not None
        grouped.setdefault(card.claim_id, []).append(card)
    canonical_cards: list[EvidenceCard] = []
    clusters: list[ClaimCluster] = []
    for claim_id in sorted(grouped):
        group_by_evidence_id = {card.evidence_id: card for card in grouped[claim_id]}
        group = list(group_by_evidence_id.values())
        canonical = min(group, key=lambda card: (-card.reliability, card.evidence_id))
        supporters = sorted(card.evidence_id for card in group if card.evidence_id != canonical.evidence_id)
        canonical_cards.append(canonical)
        clusters.append(
            ClaimCluster(
                claim_id=claim_id,
                canonical_evidence_id=canonical.evidence_id,
                supporting_evidence_ids=supporters,
            )
        )
    return canonical_cards, clusters


def normalize_card_semantics(card: EvidenceCard) -> EvidenceCard:
    """Correct category-demand claims and promote grounded DCF drivers."""

    text = " ".join([card.fact, *card.mechanism])
    update: dict[str, object] = {}
    links = list(card.dcf_links)
    if card.evidence_type == EvidenceType.MARKET_SHARE and _DEMAND_RE.search(text) and not _SHARE_RE.search(text):
        update.update(
            evidence_type=EvidenceType.MARKET_DEMAND,
            economic_scope=EconomicScope.INDUSTRY,
            direction=EvidenceDirection.NEUTRAL,
            forward_driver_type=ForwardDriverType.MARKET_GROWTH,
        )
        links.append(DcfLink.REVENUE)
    if (
        card.evidence_type == EvidenceType.CUSTOMER_RETENTION
        and _RECURRING_CATEGORY_RE.search(text)
        and not _RETENTION_RE.search(text)
    ):
        update.update(
            evidence_type=EvidenceType.CATEGORY_RECURRING_DEMAND,
            economic_scope=EconomicScope.PRODUCT_CATEGORY,
            direction=EvidenceDirection.NEUTRAL,
            forward_driver_type=ForwardDriverType.VOLUME,
        )
        links.append(DcfLink.REVENUE)
    if re.search(r"가동률|capacity\s+utili[sz]ation", text, re.IGNORECASE):
        update.setdefault("forward_driver_type", ForwardDriverType.UTILIZATION)
        links.extend([DcfLink.REVENUE, DcfLink.CAPEX])
    elif re.search(r"생산능력|CAPA|production\s+capacity", text, re.IGNORECASE):
        update.setdefault("forward_driver_type", ForwardDriverType.CAPACITY)
        links.extend([DcfLink.REVENUE, DcfLink.CAPEX])
    if re.search(r"수출\s*(?:비중|믹스)|export\s+mix", text, re.IGNORECASE):
        update.setdefault("forward_driver_type", ForwardDriverType.EXPORT_MIX)
        links.extend([DcfLink.REVENUE, DcfLink.EBIT_MARGIN])
    if re.search(r"ASP|평균\s*판매\s*가격|판매단가", text, re.IGNORECASE):
        update.setdefault("forward_driver_type", ForwardDriverType.ASP)
        links.extend([DcfLink.REVENUE, DcfLink.EBIT_MARGIN])
    if re.search(r"원재료|raw\s+material|input\s+cost", text, re.IGNORECASE):
        update.setdefault("forward_driver_type", ForwardDriverType.RAW_MATERIAL_COST)
        links.append(DcfLink.EBIT_MARGIN)
    if update.get("forward_driver_type") is not None or card.forward_driver_type is not None:
        update["dcf_links"] = list(dict.fromkeys(links))
    return card.model_copy(update=update) if update else card


def build_forward_driver_cards(cards: list[EvidenceCard]) -> list[ForwardDriverCard]:
    result: list[ForwardDriverCard] = []
    for card in cards:
        if card.forward_driver_type is None or not card.dcf_links:
            continue
        result.append(
            ForwardDriverCard(
                driver_id=stable_id("FD", card.evidence_id, card.forward_driver_type.value),
                source_evidence_id=card.evidence_id,
                source_chunk_id=card.source_chunk_id,
                node_ids=card.node_ids,
                driver_type=card.forward_driver_type,
                evidence=card.fact,
                implication=card.mechanism,
                dcf_links=card.dcf_links,
                statement_type=card.statement_type,
                economic_scope=card.economic_scope,
                segment=card.segment,
                period=card.period,
                forecast_horizon=card.forecast_horizon,
                reliability=card.reliability,
            )
        )
    return result


def build_evidence_relations(
    cards: list[EvidenceCard],
    *,
    duplicate_threshold: float = 0.92,
    contradiction_threshold: float = 0.72,
    update_threshold: float = 0.82,
    support_threshold: float = 0.55,
    weakens_threshold: float = 0.45,
) -> list[EvidenceRelation]:
    cards = sorted(cards, key=lambda card: card.evidence_id)
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
        canonical = min(
            group,
            key=lambda card: (
                -card.reliability,
                card.evidence_id,
            ),
        )
        supporters = sorted(card.evidence_id for card in group if card.evidence_id != canonical.evidence_id)
        clusters.append(
            EvidenceCluster(
                canonical_evidence_id=canonical.evidence_id,
                supporting_evidence_ids=supporters,
            )
        )
    return sorted(clusters, key=lambda cluster: cluster.canonical_evidence_id)
