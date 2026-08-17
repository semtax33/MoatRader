from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Callable

from moatrader.canonical.ids import stable_id
from moatrader.canonical.models import SectionRole, SourceRef, SourceType
from moatrader.evidence.atomic import (
    atomic_unit_set_sha256,
    build_atomic_evidence_units,
    select_atomic_evidence_units,
    select_context_cited_atomic_units,
    split_atomic_evidence_text,
)
from moatrader.evidence.models import (
    AtomicEvidenceJudgment,
    ContextualMoatAssessment,
    EvidenceCard,
    EvidenceDirection,
    STRUCTURAL_MOAT_TYPES,
)
from moatrader.context.moat_strength import MoatStrengthContext
from moatrader.evidence.processing import atomic_judgment_to_card, build_canonical_claim_set
from moatrader.evidence.validation import derive_moat_score
from moatrader.semantic.chunker import HeuristicTokenCounter, SemanticChunk


METAMORPHIC_SCHEMA_VERSION = "moatrader-moat-metamorphic/1"


def _is_reducer_input(card: EvidenceCard) -> bool:
    """Mirror the production reducer input contract exactly.

    All negative claims remain in the audited canonical claim set even when
    their scope prevents them from affecting the public score.  The reducer
    itself applies that scoring-scope rule.
    """

    return card.direction == EvidenceDirection.MOAT_NEGATIVE or (
        card.direction == EvidenceDirection.MOAT_POSITIVE
        and card.evidence_type in STRUCTURAL_MOAT_TYPES
    )


def _read_jsonl(path: Path, model: type[SemanticChunk] | type[EvidenceCard]) -> list[object]:
    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line
    ]


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _rebuild_markdown(chunk: SemanticChunk, *, reverse: bool, decorated: bool = False) -> str:
    units = split_atomic_evidence_text(chunk.markdown)
    if reverse:
        units = list(reversed(units))
    if decorated:
        return "# Presentation-only heading\n\n" + "\n\n".join(
            f"-   {'   '.join(unit.split())}" for unit in units
        )
    return "\n\n".join(units)


def _map_markdown(
    chunks: list[SemanticChunk],
    mapper: Callable[[SemanticChunk], str],
) -> list[SemanticChunk]:
    counter = HeuristicTokenCounter()
    return [
        chunk.model_copy(
            update={
                "markdown": (markdown := mapper(chunk)),
                "token_count": counter.count(markdown),
            }
        )
        for chunk in chunks
    ]


def _variants(chunks: list[SemanticChunk]) -> dict[str, list[SemanticChunk]]:
    sentence_shuffle = _map_markdown(chunks, lambda chunk: _rebuild_markdown(chunk, reverse=True))
    paragraph_shuffle = list(reversed(sentence_shuffle))
    duplicate = [
        *chunks,
        *[
            chunk.model_copy(update={"chunk_id": stable_id("META_DUP", chunk.chunk_id)})
            for chunk in chunks
        ],
    ]
    whitespace_heading = _map_markdown(
        chunks,
        lambda chunk: _rebuild_markdown(chunk, reverse=False, decorated=True),
    )
    boilerplate = SemanticChunk(
        chunk_id="META_BOILERPLATE",
        document_id=chunks[0].document_id if chunks else "META",
        section_path=["Administrative boilerplate"],
        section_role=SectionRole.OTHER,
        node_ids=["META_BOILERPLATE_NODE"],
        chunk_type="text",
        markdown="This administrative notice contains no company competitive claim.",
        token_count=12,
        source_refs=list(chunks[0].source_refs) if chunks else [],
        metadata={"metamorphic": "irrelevant_boilerplate"},
    )
    summary = SemanticChunk(
        chunk_id="META_GENERATED_SUMMARY",
        document_id="META_GENERATED_SUMMARY",
        section_path=["Generated summary"],
        section_role=SectionRole.BUSINESS,
        node_ids=["META_GENERATED_SUMMARY_NODE"],
        chunk_type="generated_summary",
        markdown="The company has a powerful customer lock-in and an exceptional moat.",
        token_count=14,
        source_refs=[
            SourceRef(
                source_type=SourceType.GENERATED_SUMMARY,
                document_id="META_GENERATED_SUMMARY",
            )
        ],
        metadata={"generated_summary": True},
    )
    return {
        "sentence_shuffle": sentence_shuffle,
        "paragraph_shuffle": paragraph_shuffle,
        "duplicate_evidence": duplicate,
        "summary_injection": [summary, *chunks],
        "whitespace_heading_change": whitespace_heading,
        "irrelevant_boilerplate_injection": [*chunks, boilerplate],
        "node_order_change": list(reversed(chunks)),
    }


def audit_company_metamorphs(
    company_directory: str | Path,
    *,
    issuer_id: str | None,
    maximum_atomic_units: int | None,
    maximum_ir_atomic_units: int | None = None,
    preserve_ir_document_coverage: bool = False,
) -> dict[str, object]:
    directory = Path(company_directory)
    chunks = list(_read_jsonl(directory / "chunks.jsonl", SemanticChunk))
    selected = list(_read_jsonl(directory / "atomic-evidence-units.jsonl", SemanticChunk))
    selection_manifest = json.loads(
        (directory / "evidence-chunk-selection.json").read_text(encoding="utf-8-sig")
    )
    source_partitioned = selection_manifest.get("method") == "dual_lane_citation_audit/1" and (
        "baseline_base_atomic_unit_set_sha256" in selection_manifest
    )
    contextual_assessment = ContextualMoatAssessment.model_validate_json(
        (directory / "contextual-moat-assessment.json").read_text(encoding="utf-8-sig")
    )
    strength_context = MoatStrengthContext.model_validate_json(
        (directory / "moat-strength-context.json").read_text(encoding="utf-8-sig")
    )
    chunk_id_by_ref = {
        reference.ref_id: reference.chunk_id
        for reference in strength_context.references
    }
    raw_quote_by_ref = {
        reference.ref_id: reference.raw_quote
        for reference in strength_context.references
    }
    claim_cards = list(_read_jsonl(directory / "canonical-claim-set.jsonl", EvidenceCard))
    all_cards = list(_read_jsonl(directory / "evidence.jsonl", EvidenceCard))
    current_cards = list(_read_jsonl(directory / "current-evidence.jsonl", EvidenceCard))
    current_ids = {card.evidence_id for card in current_cards}
    carried_cards = [card for card in all_cards if card.evidence_id not in current_ids]
    judgments = {
        path.stem: AtomicEvidenceJudgment.model_validate_json(
            path.read_text(encoding="utf-8-sig")
        )
        for path in (directory / "atomic-judgment-by-key").glob("*.json")
    }
    baseline_score = json.loads((directory / "moat-score.json").read_text(encoding="utf-8-sig"))
    scoring_cards = [card for card in claim_cards if _is_reducer_input(card)]
    as_of = baseline_score["as_of"]
    selected_keys = {str(unit.metadata["atomic_evidence_key"]) for unit in selected}
    failures: list[str] = []
    results: dict[str, object] = {}

    # Algebraic reducer audit: set union must be commutative, associative and
    # idempotent.  Duplicate/reordered lists intentionally feed the reducer.
    reducer_inputs = {
        "original": scoring_cards,
        "reversed": list(reversed(scoring_cards)),
        "duplicated": [*scoring_cards, *scoring_cards],
        "partitioned": [*scoring_cards[::2], *scoring_cards[1::2]],
    }
    reducer_outputs = {
        name: derive_moat_score(
            None,
            cards,
            issuer_id=issuer_id,
            as_of=date.fromisoformat(as_of),
        )
        for name, cards in reducer_inputs.items()
    }
    atomic_baseline = reducer_outputs["original"]
    baseline_claim_ids = set(atomic_baseline.canonical_claim_ids)
    baseline_evidence_ids = {
        evidence_id
        for mechanism in atomic_baseline.mechanisms
        for evidence_id in mechanism.evidence_ids
    } | set(atomic_baseline.counterevidence_ids)
    expected_score = atomic_baseline.economic_moat_score
    for name, score in reducer_outputs.items():
        if score.economic_moat_score != expected_score:
            failures.append(f"atomic audit reducer {name} score delta is not zero")
        if set(score.canonical_claim_ids) != baseline_claim_ids:
            failures.append(f"reducer {name} claim set changed")

    for name, transformed_chunks in _variants(chunks).items():
        all_transformed_units = build_atomic_evidence_units(
            transformed_chunks,
            issuer_id=issuer_id,
        )
        if source_partitioned:
            transformed_base_units = [
                unit
                for unit in all_transformed_units
                if not any(ref.source_type == SourceType.IR for ref in unit.source_refs)
            ]
            transformed_ir_units = [
                unit
                for unit in all_transformed_units
                if any(ref.source_type == SourceType.IR for ref in unit.source_refs)
            ]
            baseline_transformed_units = [
                *select_atomic_evidence_units(
                    transformed_base_units,
                    maximum_atomic_units,
                ),
                *select_atomic_evidence_units(
                    transformed_ir_units,
                    maximum_ir_atomic_units,
                    preserve_document_coverage=preserve_ir_document_coverage,
                ),
            ]
        else:
            baseline_transformed_units = select_atomic_evidence_units(
                all_transformed_units,
                maximum_atomic_units,
            )
        cited_transformed_units = select_context_cited_atomic_units(
            all_transformed_units,
            contextual_assessment,
            chunk_id_by_ref=chunk_id_by_ref,
            raw_quote_by_ref=raw_quote_by_ref,
        )
        transformed_units = sorted(
            {
                unit.chunk_id: unit
                for unit in [*baseline_transformed_units, *cited_transformed_units]
            }.values(),
            key=lambda unit: str(unit.metadata["atomic_evidence_key"]),
        )
        transformed_keys = {
            str(unit.metadata["atomic_evidence_key"]) for unit in transformed_units
        }
        key_jaccard = _jaccard(selected_keys, transformed_keys)
        missing_judgments = sorted(transformed_keys - set(judgments))
        transformed_current = [
            atomic_judgment_to_card(judgments[str(unit.metadata["atomic_evidence_key"])], unit, issuer_id=issuer_id)
            for unit in transformed_units
            if str(unit.metadata["atomic_evidence_key"]) in judgments
            and judgments[str(unit.metadata["atomic_evidence_key"])].is_investment_relevant
        ]
        transformed_claims, _ = build_canonical_claim_set(
            [*carried_cards, *transformed_current],
            issuer_id=issuer_id,
        )
        transformed_scoring = [
            card for card in transformed_claims if _is_reducer_input(card)
        ]
        transformed_score = derive_moat_score(
            None,
            transformed_scoring,
            issuer_id=issuer_id,
            as_of=date.fromisoformat(as_of),
        )
        transformed_evidence_ids = {
            evidence_id
            for mechanism in transformed_score.mechanisms
            for evidence_id in mechanism.evidence_ids
        } | set(transformed_score.counterevidence_ids)
        evidence_jaccard = _jaccard(baseline_evidence_ids, transformed_evidence_ids)
        claim_jaccard = _jaccard(
            baseline_claim_ids,
            set(transformed_score.canonical_claim_ids),
        )
        score_delta = abs(transformed_score.economic_moat_score - expected_score)
        passed = (
            not missing_judgments
            and key_jaccard == 1.0
            and evidence_jaccard == 1.0
            and claim_jaccard == 1.0
            and score_delta == 0.0
        )
        if not passed:
            failures.append(
                f"{name}: atomic/evidence set changed (Jaccard={key_jaccard:.3f})"
            )
        results[name] = {
            "passed": passed,
            "atomic_key_jaccard": key_jaccard,
            "evidence_jaccard": evidence_jaccard,
            "claim_jaccard": claim_jaccard,
            "score_delta": score_delta,
            "missing_judgment_keys": missing_judgments,
            "atomic_unit_set_sha256": atomic_unit_set_sha256(transformed_units),
        }

    return {
        "schema_version": METAMORPHIC_SCHEMA_VERSION,
        "passed": not failures,
        "failures": failures,
        "baseline_atomic_unit_set_sha256": atomic_unit_set_sha256(selected),
        "baseline_claim_ids": sorted(baseline_claim_ids),
        "published_contextual_score": float(baseline_score["economic_moat_score"]),
        "published_scoring_method": baseline_score.get("scoring_method"),
        "note": "metamorphic algebra applies to the atomic audit lane, not contextual strength",
        "reducer_algebra": {
            name: {
                "score": score.economic_moat_score,
                "claim_ids": score.canonical_claim_ids,
            }
            for name, score in reducer_outputs.items()
        },
        "transformations": results,
    }
