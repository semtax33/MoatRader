from __future__ import annotations

import re
from collections import defaultdict
from datetime import date
from decimal import Decimal

from moatrader.canonical.models import CanonicalDocumentBundle
from moatrader.evidence.models import (
    ContextualMoatAssessment,
    Durability,
    CoverageMetrics,
    EconomicScope,
    EvidenceBatchExtractionResult,
    EvidenceCard,
    EvidenceDirection,
    EvidenceExtractionResult,
    EvidenceType,
    MoatMechanismScore,
    MoatOutcomeScore,
    MoatAuditStatus,
    MoatScore,
    OUTCOME_CORROBORATION_TYPES,
    ReconciledCounterevidence,
    ReconciledMechanismStrength,
    ReconciledMoatAssessment,
    ReconciledOutcomeStrength,
    ReconciliationDecision,
    STRUCTURAL_MOAT_TYPES,
)
from moatrader.semantic.chunker import SemanticChunk


MOAT_COUNTEREVIDENCE_TYPES = STRUCTURAL_MOAT_TYPES | OUTCOME_CORROBORATION_TYPES | {
    EvidenceType.COMPETITIVE_THREAT,
    EvidenceType.CUSTOMER_CONCENTRATION,
    EvidenceType.SUBSTITUTION_RISK,
    EvidenceType.TECHNOLOGY_RISK,
    EvidenceType.CAPITAL_INTENSITY,
}


def _is_moat_counterevidence(card: EvidenceCard) -> bool:
    return (
        card.direction == EvidenceDirection.MOAT_NEGATIVE
        and card.evidence_type in MOAT_COUNTEREVIDENCE_TYPES
        and card.economic_scope in {EconomicScope.COMPANY, EconomicScope.SEGMENT}
    )


def _number_tokens(value: str) -> set[str]:
    """Return normalized numeric tokens so commas/decimal zeroes compare."""
    result: set[str] = set()
    suffix_pattern = r"%|배|개|명|원|년|월|일"

    def add(number: str, suffix: str = "") -> None:
        # The source token regex intentionally accepts decimal separators, but
        # that also captures sentence/table punctuation after a number (for
        # example ``2013.12.``).  Strip only trailing punctuation so an exact
        # grounded decimal/date can compare with a claim ending at the number.
        candidate = number.rstrip(".,").replace(",", "")
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
            # Multi-dot dates such as ``2020.09.03`` are not Decimal values.
            # Keep their canonical punctuation-stripped form so a sentence
            # comma or period cannot create a false ungrounded-number error.
            result.add(candidate + suffix)
            result.add(candidate)

    # Treat ISO/dotted dates as one token before scanning ordinary numbers.
    # Otherwise the hyphenated month/day fragments (for example ``-27`` in
    # ``2025-10-27``) look like invented signed values in an otherwise fully
    # grounded claim.
    masked = list(value)
    full_date_pattern = re.compile(
        r"(?P<year>\d{4})[.-](?P<month>\d{1,2})[.-](?P<day>\d{1,2})"
        r"(?=$|[^\d]|20\d{2}[.-]|[1-9]\.)"
    )
    for match in full_date_pattern.finditer(value):
        year = match.group("year")
        month = int(match.group("month"))
        day = int(match.group("day"))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        canonical = f"{year}-{month:02d}-{day:02d}"
        result.add(canonical)
        result.add(year)
        for index in range(match.start(), match.end()):
            masked[index] = " "

    # A source line may concatenate adjacent dates without whitespace, and a
    # rendered table may concatenate a date with the next numbered heading:
    # ``2025.10.172023.04.06`` and ``2027-08-193. 기타``.  Full dates above
    # deliberately accept those two bounded continuations.  Parse remaining
    # year-month forms only after the full-date spans have been removed.
    partially_masked = "".join(masked)
    partial_date_pattern = re.compile(
        r"(?<!\d)(?P<year>\d{4})[.-](?P<month>\d{1,2})(?=$|[^\d])"
    )
    for match in partial_date_pattern.finditer(partially_masked):
        year = match.group("year")
        month = int(match.group("month"))
        if not 1 <= month <= 12:
            continue
        result.add(f"{year}-{month:02d}")
        result.add(year)
        for index in range(match.start(), match.end()):
            masked[index] = " "
    remaining = "".join(masked)

    # DART issue/series labels compact repeated suffixes, for example
    # ``제43-1,2회`` means issues ``43-1`` and ``43-2``.  Treat the comma as
    # an enumerator here; the generic amount parser would otherwise collapse
    # ``1,2`` into the unrelated number ``12``.
    enumerated_label_pattern = re.compile(
        r"(?<!\d)(?P<series>\d+)-(?P<variants>\d+(?:,\d+)+)(?=$|\D)"
    )
    enumerated_masked = list(remaining)
    for match in enumerated_label_pattern.finditer(remaining):
        add(match.group("series"))
        for variant in match.group("variants").split(","):
            add(variant)
        for index in range(match.start(), match.end()):
            enumerated_masked[index] = " "
    remaining = "".join(enumerated_masked)

    # DART table linearization can concatenate an amount and the following
    # year, e.g. ``$34,000,0002020년``.  Recover only this narrow, comma-grouped
    # boundary instead of accepting arbitrary four-digit substrings in IDs or
    # unseparated amounts.
    for match in re.finditer(
        r"\d{1,3}(?:,\d{3})+(?P<year>20\d{2})(?=\D|$)",
        remaining,
    ):
        add(match.group("year"))

    # A year immediately followed by the Korean year suffix remains grounded
    # even when it is joined to an ASCII section label such as ``GOLF2016년``.
    for match in re.finditer(r"(?P<year>\d{4})(?=\s*년)", remaining):
        add(match.group("year"), "년")

    number_pattern = re.compile(
        rf"(?<![A-Za-z0-9])(?P<number>[-+]?\d[\d,]*(?:\.\d+)?)(?:\s*(?P<suffix>{suffix_pattern}))?"
    )
    for match in number_pattern.finditer(remaining):
        add(match.group("number"), match.group("suffix") or "")
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
        elif not _is_moat_counterevidence(card):
            errors.append(
                f"counterevidence {evidence_id} is outside the company/segment structural risk rubric"
            )
    available_negative = any(_is_moat_counterevidence(card) for card in evidence)
    if score.mechanisms and available_negative and not score.counterevidence_ids:
        errors.append("positive moat assessment must cite available counterevidence")
    if score.economic_moat_score > 0 and not score.mechanisms:
        errors.append("positive moat score requires at least one validated mechanism")
    if score.scoring_method == "DUAL_LANE_CONTEXTUAL_STRENGTH_REDUCER_V1":
        if score.evidence_confidence != score.model_confidence:
            errors.append("dual-lane evidence_confidence must match compatibility model_confidence")
        if score.economic_moat_score > 0 and not score.context_chunk_ids:
            errors.append("dual-lane positive score requires contextual chunk citations")
        if score.economic_moat_score > 0 and not score.atomic_evidence_ids:
            errors.append("dual-lane positive score requires atomic evidence IDs")
        if score.audit_status == MoatAuditStatus.FAIL and score.economic_moat_score > 0:
            errors.append("failed reconciliation cannot publish a positive score")
        for outcome in score.outcome_strengths:
            if outcome.evidence_type not in OUTCOME_CORROBORATION_TYPES:
                errors.append(
                    f"outcome {outcome.evidence_type.value} is outside the outcome rubric"
                )
            for evidence_id in outcome.evidence_ids:
                card = card_by_id.get(evidence_id)
                if card is None:
                    errors.append(
                        f"outcome {outcome.evidence_type.value}: unknown evidence ID {evidence_id}"
                    )
                elif (
                    card.direction != EvidenceDirection.MOAT_POSITIVE
                    or card.evidence_type != outcome.evidence_type
                ):
                    errors.append(
                        f"outcome {outcome.evidence_type.value}: invalid atomic evidence {evidence_id}"
                    )
    return errors


def derive_moat_score(
    score: MoatScore | None,
    evidence: list[EvidenceCard],
    *,
    issuer_id: str | None = None,
    as_of: date | None = None,
    document_coverage: CoverageMetrics | None = None,
) -> MoatScore:
    """Legacy atomic-only reducer retained for audit metamorphic algebra.

    Input is first reduced to a canonical claim set, so permutations,
    partitions and duplicate evidence cannot change this audit result. It is
    not the production economic-strength score; the runner uses
    ``derive_audited_moat_score`` after contextual reconciliation.
    """
    from moatrader.evidence.processing import build_canonical_claim_set

    effective_issuer_id = issuer_id or (score.issuer_id if score else None)
    evidence, _claim_clusters = build_canonical_claim_set(
        evidence,
        issuer_id=effective_issuer_id,
    )
    proposed_score = score.economic_moat_score if score is not None else None
    if score is None:
        if as_of is None:
            raise ValueError("as_of is required for deterministic MOAT scoring")
        score = MoatScore(
            issuer_id=effective_issuer_id,
            as_of=as_of,
            economic_moat_score=0.0,
            durability=Durability.LOW,
            model_confidence=0.0,
            document_coverage=document_coverage or CoverageMetrics(),
        )
    else:
        validation_errors = validate_moat_score(score, evidence)
        if validation_errors:
            raise ValueError("cannot derive MOAT score from invalid evidence: " + "; ".join(validation_errors))
    grouped: dict[EvidenceType, list[EvidenceCard]] = defaultdict(list)
    for card in evidence:
        if (
            card.direction == EvidenceDirection.MOAT_POSITIVE
            and card.evidence_type in STRUCTURAL_MOAT_TYPES
            and card.economic_scope in {EconomicScope.COMPANY, EconomicScope.SEGMENT}
        ):
            grouped[card.evidence_type].append(card)

    adjusted_mechanisms: list[MoatMechanismScore] = []
    source_counts: dict[EvidenceType, int] = {}
    for evidence_type in sorted(grouped, key=lambda item: item.value):
        cards = sorted(grouped[evidence_type], key=lambda item: item.evidence_id)
        qualities = sorted((_deterministic_card_quality(card) for card in cards), reverse=True)
        independent_sources = len({card.source_chunk_id for card in cards})
        source_counts[evidence_type] = independent_sources
        corroboration_bonus = min(0.10, max(0, independent_sources - 1) * 0.05)
        mechanism_score = round(10.0 * min(1.0, qualities[0] + corroboration_bonus), 2)
        adjusted_mechanisms.append(
            MoatMechanismScore(
                evidence_type=evidence_type,
                score=mechanism_score,
                evidence_ids=[card.evidence_id for card in cards],
                rationale=(
                    f"Deterministic aggregation of {len(cards)} grounded "
                    f"{evidence_type.value} evidence unit(s)."
                ),
            )
        )

    strengths = sorted((item.score for item in adjusted_mechanisms), reverse=True)
    if not strengths:
        base_score = 0.0
    elif len(strengths) == 1:
        base_score = strengths[0]
    else:
        base_score = 0.8 * strengths[0] + 0.2 * strengths[1]

    counter_cards = sorted(
        (
            card
            for card in evidence
            if _is_moat_counterevidence(card)
        ),
        key=lambda card: (-_deterministic_card_quality(card), card.evidence_id),
    )
    selected_counter_cards = counter_cards[:3]
    counter_penalty = min(
        3.0,
        sum(_deterministic_card_quality(card) for card in selected_counter_cards),
    )
    top_mechanism = max(adjusted_mechanisms, key=lambda item: item.score, default=None)
    top_sources = source_counts.get(top_mechanism.evidence_type, 0) if top_mechanism else 0
    if top_mechanism is None or top_mechanism.score < 4.0:
        durability = Durability.LOW
    elif top_mechanism.score >= 8.5 and top_sources >= 3:
        durability = Durability.HIGH
    elif top_mechanism.score >= 6.0 and top_sources >= 2:
        durability = Durability.MEDIUM_HIGH
    else:
        durability = Durability.MEDIUM
    durability_cap = {
        Durability.LOW: 3.0,
        Durability.MEDIUM: 5.0,
        Durability.MEDIUM_HIGH: 8.0,
        Durability.HIGH: 10.0,
    }[durability]
    derived = round(max(0.0, min(durability_cap, base_score - counter_penalty)), 2)
    mechanism_cards = [card for cards in grouped.values() for card in cards]
    confidence = (
        round(sum(_deterministic_card_quality(card) for card in mechanism_cards) / len(mechanism_cards), 2)
        if mechanism_cards
        else 0.0
    )
    caveats = ["Public MOAT score, durability, and counterevidence penalty are deterministic."]
    if not adjusted_mechanisms:
        caveats.append("No validated company-specific structural moat mechanism.")
    if selected_counter_cards:
        caveats.append(
            f"Applied {len(selected_counter_cards)} grounded structural counterevidence unit(s)."
        )
    return score.model_copy(
        update={
            "economic_moat_score": derived,
            "mechanisms": adjusted_mechanisms,
            "counterevidence_ids": [card.evidence_id for card in selected_counter_cards],
            "canonical_claim_ids": sorted(
                card.claim_id for card in evidence if card.claim_id is not None
            ),
            "durability": durability,
            "model_confidence": confidence,
            "llm_proposed_score": proposed_score,
            "caveats": caveats,
        }
    )


def _deterministic_card_quality(card: EvidenceCard) -> float:
    scope_factor = 1.0 if card.economic_scope == EconomicScope.COMPANY else 0.9
    return round(card.reliability * scope_factor, 4)


def validate_contextual_moat_assessment(
    assessment: ContextualMoatAssessment,
    chunks: list[SemanticChunk],
) -> list[str]:
    """Validate broad-context attributes directly against canonical chunks."""

    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    errors: list[str] = []
    seen: dict[str, set[EvidenceType]] = defaultdict(set)
    groups = (
        ("mechanism", assessment.mechanisms),
        ("outcome", assessment.outcome_confirmation),
        ("counterevidence", assessment.counterevidence),
    )
    for category, items in groups:
        for item in items:
            evidence_type = item.evidence_type
            if evidence_type in seen[category]:
                errors.append(f"{category}: duplicate {evidence_type.value}")
            seen[category].add(evidence_type)
            if category == "mechanism":
                if evidence_type not in STRUCTURAL_MOAT_TYPES:
                    errors.append(f"mechanism: {evidence_type.value} is not structural")
                if item.economic_scope not in {
                    EconomicScope.COMPANY,
                    EconomicScope.SEGMENT,
                }:
                    errors.append(
                        f"mechanism: {evidence_type.value} scope is "
                        f"{item.economic_scope.value}"
                    )
            elif category == "outcome" and evidence_type not in OUTCOME_CORROBORATION_TYPES:
                errors.append(f"outcome: {evidence_type.value} is not an outcome type")
            elif category == "counterevidence" and evidence_type not in MOAT_COUNTEREVIDENCE_TYPES:
                errors.append(
                    f"counterevidence: {evidence_type.value} is outside the risk rubric"
                )

            cited_chunks: list[SemanticChunk] = []
            for citation in item.citations:
                chunk = chunk_by_id.get(citation.chunk_id)
                if chunk is None:
                    errors.append(
                        f"{category} {evidence_type.value}: unknown chunk {citation.chunk_id}"
                    )
                    continue
                cited_chunks.append(chunk)
                outside_nodes = set(citation.node_ids) - set(chunk.node_ids)
                if outside_nodes:
                    errors.append(
                        f"{category} {evidence_type.value}: node IDs outside "
                        f"{citation.chunk_id}: {sorted(outside_nodes)}"
                    )
                if not _quote_in_text(citation.raw_quote, chunk.markdown):
                    errors.append(
                        f"{category} {evidence_type.value}: raw quote is not in "
                        f"{citation.chunk_id}"
                    )
            source_numbers = set().union(
                *(_number_tokens(chunk.markdown) for chunk in cited_chunks)
            ) if cited_chunks else set()
            unsupported = _number_tokens(item.rationale) - source_numbers
            if unsupported:
                errors.append(
                    f"{category} {evidence_type.value}: rationale contains unsupported "
                    f"numbers {sorted(unsupported)}"
                )
    if assessment.durability_bucket > 0 and not assessment.mechanisms:
        errors.append("positive durability requires at least one contextual mechanism")
    return errors


def _quote_in_text(quote: str, text: str) -> bool:
    if quote in text:
        return True
    tokens = quote.split()
    return bool(tokens and re.search(r"\s+".join(re.escape(token) for token in tokens), text))


def reconcile_context_and_claims(
    assessment: ContextualMoatAssessment,
    evidence: list[EvidenceCard],
    *,
    contextual_chunks: list[SemanticChunk],
    atomic_units: list[SemanticChunk],
) -> ReconciledMoatAssessment:
    """Gate contextual strength through grounded context and atomic claims."""

    validation_errors = validate_contextual_moat_assessment(
        assessment,
        contextual_chunks,
    )
    if validation_errors:
        raise ValueError(
            "invalid contextual MOAT assessment: " + "; ".join(validation_errors)
        )
    context_by_id = {chunk.chunk_id: chunk for chunk in contextual_chunks}
    origins_by_atomic_id = {
        unit.chunk_id: set(unit.metadata.get("origin_chunk_ids") or [])
        for unit in atomic_units
    }

    def citation_chunk_ids(item: object) -> list[str]:
        return sorted({citation.chunk_id for citation in item.citations})

    def matches_context(card: EvidenceCard, chunk_ids: set[str]) -> bool:
        origins = origins_by_atomic_id.get(card.source_chunk_id, set())
        if origins & chunk_ids:
            return True
        return bool(
            card.raw_quote
            and any(
                chunk_id in context_by_id
                and _quote_in_text(card.raw_quote, context_by_id[chunk_id].markdown)
                for chunk_id in chunk_ids
            )
        )

    decisions: list[ReconciliationDecision] = []
    mechanisms: list[ReconciledMechanismStrength] = []
    outcomes: list[ReconciledOutcomeStrength] = []
    counters: list[ReconciledCounterevidence] = []

    for item in assessment.mechanisms:
        chunk_ids = set(citation_chunk_ids(item))
        matching = sorted(
            (
                card
                for card in evidence
                if card.direction == EvidenceDirection.MOAT_POSITIVE
                and card.evidence_type == item.evidence_type
                and card.economic_scope in {EconomicScope.COMPANY, EconomicScope.SEGMENT}
                and matches_context(card, chunk_ids)
            ),
            key=lambda card: card.evidence_id,
        )
        accepted = bool(matching)
        decisions.append(
            ReconciliationDecision(
                category="MECHANISM",
                evidence_type=item.evidence_type,
                accepted=accepted,
                reason=(
                    "context citation matched a positive atomic/canonical claim"
                    if accepted
                    else "no positive atomic/canonical claim matched the cited context"
                ),
                context_chunk_ids=sorted(chunk_ids),
                atomic_evidence_ids=[card.evidence_id for card in matching],
            )
        )
        if accepted:
            mechanisms.append(
                ReconciledMechanismStrength(
                    evidence_type=item.evidence_type,
                    strength_bucket=item.strength_bucket,
                    scope_materiality_bucket=item.scope_materiality_bucket,
                    economic_scope=item.economic_scope,
                    context_chunk_ids=sorted(chunk_ids),
                    atomic_evidence_ids=[card.evidence_id for card in matching],
                    rationale=item.rationale,
                )
            )

    for item in assessment.outcome_confirmation:
        chunk_ids = set(citation_chunk_ids(item))
        matching = sorted(
            (
                card
                for card in evidence
                if card.direction == EvidenceDirection.MOAT_POSITIVE
                and card.evidence_type == item.evidence_type
                and matches_context(card, chunk_ids)
            ),
            key=lambda card: card.evidence_id,
        )
        accepted = bool(matching)
        decisions.append(
            ReconciliationDecision(
                category="OUTCOME",
                evidence_type=item.evidence_type,
                accepted=accepted,
                reason=(
                    "context outcome matched a positive atomic/canonical claim"
                    if accepted
                    else "no positive atomic/canonical outcome matched the cited context"
                ),
                context_chunk_ids=sorted(chunk_ids),
                atomic_evidence_ids=[card.evidence_id for card in matching],
            )
        )
        if accepted:
            outcomes.append(
                ReconciledOutcomeStrength(
                    evidence_type=item.evidence_type,
                    strength_bucket=item.strength_bucket,
                    persistence_bucket=item.persistence_bucket,
                    context_chunk_ids=sorted(chunk_ids),
                    atomic_evidence_ids=[card.evidence_id for card in matching],
                    rationale=item.rationale,
                )
            )

    represented_counter_ids: set[str] = set()
    for item in assessment.counterevidence:
        chunk_ids = set(citation_chunk_ids(item))
        matching = sorted(
            (
                card
                for card in evidence
                if _is_moat_counterevidence(card)
                and card.evidence_type == item.evidence_type
                and matches_context(card, chunk_ids)
            ),
            key=lambda card: card.evidence_id,
        )
        represented_counter_ids.update(card.evidence_id for card in matching)
        # A context counter is already directly grounded by validated quotes;
        # an atomic match strengthens audit confidence but is not allowed to
        # erase adverse source evidence.
        counters.append(
            ReconciledCounterevidence(
                evidence_type=item.evidence_type,
                severity_bucket=item.severity_bucket,
                context_chunk_ids=sorted(chunk_ids),
                atomic_evidence_ids=[card.evidence_id for card in matching],
                rationale=item.rationale,
            )
        )
        decisions.append(
            ReconciliationDecision(
                category="COUNTEREVIDENCE",
                evidence_type=item.evidence_type,
                accepted=True,
                reason=(
                    "grounded context counterevidence; atomic match present"
                    if matching
                    else "grounded context counterevidence retained conservatively"
                ),
                context_chunk_ids=sorted(chunk_ids),
                atomic_evidence_ids=[card.evidence_id for card in matching],
            )
        )

    # Atomic negative claims are never top-k pruned or discarded because the
    # contextual analyst omitted them.
    for card in sorted(
        (card for card in evidence if _is_moat_counterevidence(card)),
        key=lambda card: card.evidence_id,
    ):
        if card.evidence_id in represented_counter_ids:
            continue
        origin_ids = sorted(origins_by_atomic_id.get(card.source_chunk_id, set()))
        counters.append(
            ReconciledCounterevidence(
                evidence_type=card.evidence_type,
                severity_bucket=1,
                context_chunk_ids=origin_ids,
                atomic_evidence_ids=[card.evidence_id],
                rationale="Validated atomic counterevidence retained independently of context scoring.",
            )
        )

    positive_decisions = [
        decision
        for decision in decisions
        if decision.category in {"MECHANISM", "OUTCOME"}
    ]
    match_rate = (
        sum(decision.accepted for decision in positive_decisions) / len(positive_decisions)
        if positive_decisions
        else 1.0
    )
    audit_status = (
        MoatAuditStatus.PASS
        if match_rate >= 0.75
        else MoatAuditStatus.PARTIAL
    )
    if assessment.mechanisms and not mechanisms:
        audit_status = MoatAuditStatus.FAIL
    all_atomic_ids = sorted(
        {
            evidence_id
            for item in [*mechanisms, *outcomes, *counters]
            for evidence_id in item.atomic_evidence_ids
        }
    )
    all_context_ids = sorted(
        {
            chunk_id
            for item in [*mechanisms, *outcomes, *counters]
            for chunk_id in item.context_chunk_ids
        }
    )
    all_context_document_ids = sorted(
        {
            context_by_id[chunk_id].document_id
            for chunk_id in all_context_ids
            if chunk_id in context_by_id
        }
    )
    return ReconciledMoatAssessment(
        evidence_sufficiency=assessment.evidence_sufficiency,
        durability_bucket=assessment.durability_bucket,
        mechanisms=mechanisms,
        outcomes=outcomes,
        counterevidence=counters,
        decisions=decisions,
        context_chunk_ids=all_context_ids,
        context_document_ids=all_context_document_ids,
        atomic_evidence_ids=all_atomic_ids,
        audit_status=audit_status,
    )


def derive_audited_moat_score(
    reconciled: ReconciledMoatAssessment,
    evidence: list[EvidenceCard],
    *,
    issuer_id: str | None,
    as_of: date,
    document_coverage: CoverageMetrics,
) -> MoatScore:
    """Score economic strength and evidence confidence on separate axes."""

    from moatrader.evidence.processing import build_canonical_claim_set

    original_evidence = list(evidence)
    canonical_evidence, _clusters = build_canonical_claim_set(
        original_evidence,
        issuer_id=issuer_id,
    )
    card_by_id = {card.evidence_id: card for card in original_evidence}
    mechanisms: list[MoatMechanismScore] = []
    effective_buckets: list[float] = []
    for item in reconciled.mechanisms:
        effective_bucket = min(item.strength_bucket, item.scope_materiality_bucket)
        effective_buckets.append(float(effective_bucket))
        mechanisms.append(
            MoatMechanismScore(
                evidence_type=item.evidence_type,
                score=round(effective_bucket / 4 * 10, 2),
                evidence_ids=item.atomic_evidence_ids,
                rationale=item.rationale,
                strength_bucket=item.strength_bucket,
                scope_materiality_bucket=item.scope_materiality_bucket,
                context_chunk_ids=item.context_chunk_ids,
            )
        )

    outcome_scores: list[MoatOutcomeScore] = []
    for item in reconciled.outcomes:
        component = round((item.strength_bucket + item.persistence_bucket) / 4, 2)
        outcome_scores.append(
            MoatOutcomeScore(
                evidence_type=item.evidence_type,
                score=component,
                strength_bucket=item.strength_bucket,
                persistence_bucket=item.persistence_bucket,
                evidence_ids=item.atomic_evidence_ids,
                context_chunk_ids=item.context_chunk_ids,
                rationale=item.rationale,
            )
        )

    if effective_buckets:
        ranked_buckets = sorted(effective_buckets, reverse=True)
        mechanism_component = min(
            4.0,
            ranked_buckets[0]
            + (0.25 * ranked_buckets[1] if len(ranked_buckets) > 1 else 0.0),
        )
        outcome_component = max((item.score for item in outcome_scores), default=0.0)
        durability_component = reconciled.durability_bucket / 2
        gross_eight = mechanism_component + outcome_component + durability_component
        counter_component = max(
            (item.severity_bucket / 2 for item in reconciled.counterevidence),
            default=0.0,
        )
        economic_score = round(max(0.0, gross_eight - counter_component) / 8 * 10, 2)
    else:
        economic_score = 0.0

    durability = {
        0: Durability.LOW,
        1: Durability.LOW,
        2: Durability.MEDIUM,
        3: Durability.MEDIUM_HIGH,
        4: Durability.HIGH,
    }[reconciled.durability_bucket]

    positive_decisions = [
        decision
        for decision in reconciled.decisions
        if decision.category in {"MECHANISM", "OUTCOME"}
    ]
    match_rate = (
        sum(decision.accepted for decision in positive_decisions) / len(positive_decisions)
        if positive_decisions
        else 1.0
    )
    matched_cards = [
        card_by_id[evidence_id]
        for evidence_id in reconciled.atomic_evidence_ids
        if evidence_id in card_by_id
    ]
    reliability = (
        sum(card.reliability for card in matched_cards) / len(matched_cards)
        if matched_cards
        else (1.0 if not positive_decisions else 0.0)
    )
    source_diversity = min(1.0, len(reconciled.context_document_ids) / 3)
    coverage = document_coverage.moat_evidence_coverage
    if coverage is None:
        coverage = document_coverage.section_retention or 0.0
    confidence = round(
        0.30 * match_rate
        + 0.25 * reliability
        + 0.15 * source_diversity
        + 0.20 * coverage
        + 0.10 * (reconciled.evidence_sufficiency / 4),
        2,
    )
    audit_status = reconciled.audit_status
    if audit_status == MoatAuditStatus.PASS and confidence < 0.60:
        audit_status = MoatAuditStatus.PARTIAL

    counter_ids = sorted(
        {
            evidence_id
            for item in reconciled.counterevidence
            for evidence_id in item.atomic_evidence_ids
        }
    )
    counter_chunk_ids = sorted(
        {
            chunk_id
            for item in reconciled.counterevidence
            for chunk_id in item.context_chunk_ids
        }
    )
    caveats = [
        "Economic strength uses contextual buckets; evidence confidence is computed separately.",
        "dual-lane-strength-reducer/1 weights are a research hypothesis requiring holdout/backtest validation.",
    ]
    if reconciled.audit_status == MoatAuditStatus.FAIL:
        caveats.append("Contextual mechanisms failed the atomic/canonical reconciliation gate.")
    if counter_ids or counter_chunk_ids:
        caveats.append("Grounded counterevidence was retained independently of positive retrieval.")
    return MoatScore(
        issuer_id=issuer_id,
        as_of=as_of,
        economic_moat_score=economic_score,
        mechanisms=mechanisms,
        outcome_strengths=outcome_scores,
        counterevidence_ids=counter_ids,
        counterevidence_context_chunk_ids=counter_chunk_ids,
        canonical_claim_ids=sorted(
            card.claim_id for card in canonical_evidence if card.claim_id is not None
        ),
        context_chunk_ids=reconciled.context_chunk_ids,
        context_document_ids=reconciled.context_document_ids,
        atomic_evidence_ids=reconciled.atomic_evidence_ids,
        durability=durability,
        model_confidence=confidence,
        evidence_confidence=confidence,
        document_coverage=document_coverage,
        audit_status=audit_status,
        scoring_method="DUAL_LANE_CONTEXTUAL_STRENGTH_REDUCER_V1",
        caveats=caveats,
    )
