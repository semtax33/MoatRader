from datetime import datetime

from moatrader.canonical.models import SourceType, StatementType
from moatrader.evidence.models import (
    AtomicMoatRole,
    EvidenceCard,
    EvidenceDirection,
    EvidenceType,
)
from moatrader.valuation import ProbabilitySupport
from scripts.build_expectation_gap_research_inputs import (
    probability_support,
    source_references,
)


CUTOFF = datetime.fromisoformat("2026-08-31T23:59:59.999999+09:00")


def _card(
    evidence_id: str,
    *,
    source_type: SourceType,
    direction: EvidenceDirection,
) -> EvidenceCard:
    return EvidenceCard(
        evidence_id=evidence_id,
        source_chunk_id=f"C_{evidence_id}",
        node_ids=[f"N_{evidence_id}"],
        moat_role=(
            AtomicMoatRole.COUNTER
            if direction == EvidenceDirection.MOAT_NEGATIVE
            else AtomicMoatRole.MECHANISM
        ),
        evidence_type=EvidenceType.COST_ADVANTAGE,
        statement_type=StatementType.DISCLOSED_FACT,
        fact="Unit cost advantage evidence",
        direction=direction,
        source_type=source_type,
        reliability=0.9,
    )


def test_probability_support_preserves_mixed_external_evidence() -> None:
    probable, mapped = probability_support(
        [
            _card(
                "SUPPORT",
                source_type=SourceType.IR,
                direction=EvidenceDirection.MOAT_POSITIVE,
            ),
            _card(
                "COUNTER",
                source_type=SourceType.INDUSTRY,
                direction=EvidenceDirection.MOAT_NEGATIVE,
            ),
        ],
        issuer_id="ISSUER",
        cutoff=CUTOFF,
    )

    assert probable == ProbabilitySupport.MIXED
    assert {item.source_type for item in mapped} == {SourceType.IR, SourceType.INDUSTRY}


def test_source_references_use_chunk_available_at_and_deduplicate_document() -> None:
    chunks = [
        {
            "chunk_id": "C_SUPPORT",
            "metadata": {"available_at": "2026-08-15T09:00:00+09:00"},
            "source_refs": [
                {"source_type": "IR", "document_id": "IR-1"},
                {"source_type": "IR", "document_id": "IR-1"},
            ],
        }
    ]

    references = source_references(
        chunks=chunks,
        evidence_chunk_ids={"C_SUPPORT"},
        cutoff=CUTOFF,
    )

    assert len(references) == 1
    assert references[0].document_id == "IR-1"
    assert references[0].available_at.isoformat() == "2026-08-15T09:00:00+09:00"
