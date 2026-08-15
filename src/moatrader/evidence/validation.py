from __future__ import annotations

import re
from decimal import Decimal

from moatrader.canonical.models import CanonicalDocumentBundle
from moatrader.evidence.models import (
    Durability,
    EconomicScope,
    EvidenceBatchExtractionResult,
    EvidenceCard,
    EvidenceDirection,
    EvidenceExtractionResult,
    MoatScore,
    STRUCTURAL_MOAT_TYPES,
)
from moatrader.semantic.chunker import SemanticChunk


def _number_tokens(value: str) -> set[str]:
    """Return normalized numeric tokens so commas/decimal zeroes compare."""
    result: set[str] = set()
    for token in re.findall(r"(?<![A-Za-z])[-+]?\d[\d,.]*(?:%|배|개|명|원|년|월|일)?", value):
        suffix_match = re.search(r"(%|배|개|명|원|년|월|일)$", token)
        suffix = suffix_match.group(1) if suffix_match else ""
        number = token[: -len(suffix)] if suffix else token
        candidate = number.replace(",", "")
        try:
            normalized = format(Decimal(candidate), "f")
            if "." in normalized:
                normalized = normalized.rstrip("0").rstrip(".")
            if normalized in {"", "-", "+"}:
                normalized = "0"
            result.add(normalized + suffix)
            # A source may express a date/amount with or without its suffix.
            result.add(normalized)
        except Exception:
            result.add(token)
    return result


def _normalize_raw_quote(card: object, chunk: SemanticChunk) -> bool:
    raw_quote = getattr(card, "raw_quote", None)
    if not raw_quote:
        return False
    if raw_quote in chunk.markdown:
        return True
    tokens = str(raw_quote).split()
    match = re.search(r"\s+".join(re.escape(token) for token in tokens), chunk.markdown) if tokens else None
    if match:
        card.raw_quote = match.group(0)
        return True
    return False


def validate_evidence_result(
    result: EvidenceExtractionResult,
    chunk: SemanticChunk,
    bundle: CanonicalDocumentBundle,
    *,
    discard_invalid_cards: bool = False,
) -> list[str]:
    """Return validation errors; no LLM-generated claim is accepted silently."""
    errors: list[str] = []
    if result.chunk_id != chunk.chunk_id:
        errors.append("result chunk_id does not match the supplied chunk")
    known_nodes = bundle.ast.node_index()
    chunk_nodes = set(chunk.node_ids)
    source_numbers = _number_tokens(chunk.markdown)
    grounded_cards = []
    for card in result.cards:
        card_errors: list[str] = []
        quote_is_grounded = _normalize_raw_quote(card, chunk)
        missing = set(card.node_ids) - set(known_nodes)
        outside = set(card.node_ids) - chunk_nodes
        if missing:
            card_errors.append(f"{card.evidence_id}: unknown node IDs: {sorted(missing)}")
        if outside:
            card_errors.append(f"{card.evidence_id}: node IDs outside cited chunk: {sorted(outside)}")
        if missing or outside:
            if not discard_invalid_cards:
                errors.extend(card_errors)
            continue
        if not quote_is_grounded:
            card_errors.append(
                f"{card.evidence_id}: raw_quote is required and must be a verbatim chunk substring"
            )
            if not discard_invalid_cards:
                errors.extend(card_errors)
            continue
        claim_numbers = _number_tokens(" ".join([card.fact, *card.mechanism]))
        unsupported_numbers = claim_numbers - source_numbers
        if unsupported_numbers:
            card_errors.append(
                f"{card.evidence_id}: claim contains numbers absent from cited chunk: "
                f"{sorted(unsupported_numbers)}"
            )
            if not discard_invalid_cards:
                errors.extend(card_errors)
            continue
        grounded_metrics = []
        for metric in card.metrics:
            if isinstance(metric.value, Decimal):
                forms = {str(metric.value), f"{metric.value:f}", f"{metric.value:,}"}
                if not any(_number_tokens(form) & source_numbers for form in forms):
                    # Preserve the grounded qualitative claim, but never keep
                    # a numeric metric that is absent from the cited chunk.
                    continue
            grounded_metrics.append(metric)
        card.metrics = grounded_metrics
        grounded_cards.append(card)
    result.cards = grounded_cards
    return errors


def validate_evidence_batch_result(
    result: EvidenceBatchExtractionResult,
    chunks: list[SemanticChunk],
    bundles: dict[str, CanonicalDocumentBundle],
    *,
    discard_invalid_cards: bool = False,
) -> list[str]:
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    errors: list[str] = []
    grounded_cards = []
    for card in result.cards:
        chunk = chunk_by_id.get(card.source_chunk_id)
        if chunk is None:
            # Provider models occasionally return a node ID or invent a chunk
            # ID.  The claim cannot be grounded, so discard it instead of
            # failing an otherwise usable batch after repeated retries.
            continue
        bundle = bundles.get(chunk.document_id)
        if bundle is None:
            errors.append(f"{card.evidence_id}: no bundle for document {chunk.document_id}")
            continue
        single = EvidenceExtractionResult(chunk_id=chunk.chunk_id, cards=[card])
        card_errors = validate_evidence_result(
            single,
            chunk,
            bundle,
            discard_invalid_cards=discard_invalid_cards,
        )
        errors.extend(card_errors)
        if single.cards:
            grounded_cards.extend(single.cards)
    result.cards = grounded_cards
    return errors


def validate_moat_score(score: MoatScore, evidence: list[EvidenceCard]) -> list[str]:
    """Validate semantic score/evidence invariants before accepting a score."""
    card_by_id = {card.evidence_id: card for card in evidence}
    errors: list[str] = []
    mechanism_types: set[object] = set()
    for mechanism in score.mechanisms:
        if mechanism.evidence_type not in STRUCTURAL_MOAT_TYPES:
            errors.append(f"{mechanism.evidence_type.value}: not a structural moat mechanism")
        if mechanism.evidence_type in mechanism_types:
            errors.append(f"{mechanism.evidence_type.value}: duplicate mechanism")
        mechanism_types.add(mechanism.evidence_type)
        for evidence_id in mechanism.evidence_ids:
            card = card_by_id.get(evidence_id)
            if card is None:
                errors.append(f"{mechanism.evidence_type.value}: unknown evidence ID {evidence_id}")
                continue
            if card.direction != EvidenceDirection.MOAT_POSITIVE:
                errors.append(
                    f"{mechanism.evidence_type.value}: {evidence_id} direction is {card.direction.value}, "
                    "expected MOAT_POSITIVE"
                )
            if card.evidence_type != mechanism.evidence_type:
                errors.append(
                    f"{mechanism.evidence_type.value}: {evidence_id} type is {card.evidence_type.value}"
                )
            if card.economic_scope not in {EconomicScope.COMPANY, EconomicScope.SEGMENT}:
                errors.append(
                    f"{mechanism.evidence_type.value}: {evidence_id} scope is {card.economic_scope.value}"
                )
    for evidence_id in score.counterevidence_ids:
        card = card_by_id.get(evidence_id)
        if card is None:
            errors.append(f"counterevidence cites unknown evidence ID {evidence_id}")
        elif card.direction != EvidenceDirection.MOAT_NEGATIVE:
            errors.append(
                f"counterevidence {evidence_id} direction is {card.direction.value}, expected MOAT_NEGATIVE"
            )
    available_negative = any(card.direction == EvidenceDirection.MOAT_NEGATIVE for card in evidence)
    if score.mechanisms and available_negative and not score.counterevidence_ids:
        errors.append("positive moat assessment must cite available counterevidence")
    if score.economic_moat_score > 0 and not score.mechanisms:
        errors.append("positive moat score requires at least one validated mechanism")
    return errors


def derive_moat_score(score: MoatScore, evidence: list[EvidenceCard]) -> MoatScore:
    """Conservatively recompute the public score from grounded evidence.

    The LLM proposes mechanism-level strengths and durability.  Code enforces
    evidence-quality caps, counterevidence penalties, and durability ceilings.
    """
    validation_errors = validate_moat_score(score, evidence)
    if validation_errors:
        raise ValueError("cannot derive MOAT score from invalid evidence: " + "; ".join(validation_errors))
    card_by_id = {card.evidence_id: card for card in evidence}
    adjusted_mechanisms = []
    for mechanism in score.mechanisms:
        cards = [card_by_id[item] for item in mechanism.evidence_ids]
        qualities = sorted(
            (
                card.strength
                * card.reliability
                * (1.0 if card.economic_scope == EconomicScope.COMPANY else 0.9)
                for card in cards
            ),
            reverse=True,
        )
        independent_sources = len({card.source_chunk_id for card in cards})
        corroboration_bonus = min(0.10, max(0, independent_sources - 1) * 0.05)
        evidence_cap = min(10.0, 10.0 * min(1.0, qualities[0] + corroboration_bonus))
        adjusted_mechanisms.append(
            mechanism.model_copy(update={"score": round(min(mechanism.score, evidence_cap), 2)})
        )

    strengths = sorted((item.score for item in adjusted_mechanisms), reverse=True)
    if not strengths:
        base_score = 0.0
    elif len(strengths) == 1:
        base_score = strengths[0]
    else:
        base_score = 0.8 * strengths[0] + 0.2 * strengths[1]

    negative_qualities = sorted(
        (
            card_by_id[item].strength * card_by_id[item].reliability
            for item in score.counterevidence_ids
        ),
        reverse=True,
    )
    counter_penalty = min(3.0, sum(negative_qualities[:3]))
    durability_cap = {
        Durability.LOW: 3.0,
        Durability.MEDIUM: 5.0,
        Durability.MEDIUM_HIGH: 8.0,
        Durability.HIGH: 10.0,
    }[score.durability]
    derived = round(max(0.0, min(durability_cap, base_score - counter_penalty)), 2)
    mean_reliability = (
        sum(card_by_id[item].reliability for mechanism in adjusted_mechanisms for item in mechanism.evidence_ids)
        / sum(len(mechanism.evidence_ids) for mechanism in adjusted_mechanisms)
        if adjusted_mechanisms
        else 0.0
    )
    confidence = round(min(score.model_confidence, mean_reliability), 2)
    caveats = list(score.caveats)
    if not adjusted_mechanisms:
        caveats.append("No validated company-specific structural moat mechanism.")
    return score.model_copy(
        update={
            "economic_moat_score": derived,
            "mechanisms": adjusted_mechanisms,
            "model_confidence": confidence,
            "llm_proposed_score": score.economic_moat_score,
            "caveats": list(dict.fromkeys(caveats)),
        }
    )
