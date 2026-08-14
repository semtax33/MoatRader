from __future__ import annotations

import re
from decimal import Decimal

from moatrader.canonical.models import CanonicalDocumentBundle
from moatrader.evidence.models import EvidenceBatchExtractionResult, EvidenceExtractionResult, MoatScore
from moatrader.semantic.chunker import SemanticChunk


def _number_tokens(value: str) -> set[str]:
    return set(re.findall(r"(?<![A-Za-z])[-+]?\d[\d,.]*", value))


def _normalize_raw_quote(card: object, chunk: SemanticChunk) -> None:
    raw_quote = getattr(card, "raw_quote", None)
    if not raw_quote or raw_quote in chunk.markdown:
        return
    tokens = str(raw_quote).split()
    match = re.search(r"\s+".join(re.escape(token) for token in tokens), chunk.markdown) if tokens else None
    if match:
        card.raw_quote = match.group(0)
        return
    card.raw_quote = None
    card.reliability = min(float(getattr(card, "reliability", 0.5)), 0.4)


def validate_evidence_result(
    result: EvidenceExtractionResult,
    chunk: SemanticChunk,
    bundle: CanonicalDocumentBundle,
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
        _normalize_raw_quote(card, chunk)
        missing = set(card.node_ids) - set(known_nodes)
        outside = set(card.node_ids) - chunk_nodes
        if missing or outside:
            continue
        if card.raw_quote and card.raw_quote not in chunk.markdown:
            errors.append(f"{card.evidence_id}: raw_quote is not a verbatim chunk substring")
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
        card_errors = validate_evidence_result(single, chunk, bundle)
        errors.extend(card_errors)
        if single.cards:
            grounded_cards.extend(single.cards)
    result.cards = grounded_cards
    return errors


def validate_moat_score(score: MoatScore, evidence_ids: set[str]) -> list[str]:
    cited = {item for mechanism in score.mechanisms for item in mechanism.evidence_ids}
    cited.update(score.counterevidence_ids)
    missing = cited - evidence_ids
    return [f"moat score cites unknown evidence IDs: {sorted(missing)}"] if missing else []
