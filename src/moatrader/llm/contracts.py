from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any

from pydantic import Field

from moatrader.canonical.models import ContractModel
from moatrader.context.pack import CompanyEvidencePack
from moatrader.evidence.models import (
    CompanyDossier,
    EvidenceBatchExtractionResult,
    EvidenceCard,
    EvidenceExtractionResult,
    MoatScore,
    SectionSummary,
)
from moatrader.semantic.chunker import SemanticChunk


class LLMTask(StrEnum):
    LOCAL_EVIDENCE_EXTRACTION = "LOCAL_EVIDENCE_EXTRACTION"
    SECTION_SUMMARY = "SECTION_SUMMARY"
    FINAL_MOAT_SCORING = "FINAL_MOAT_SCORING"


class LLMRequest(ContractModel):
    task: LLMTask
    system: str
    user: str
    response_schema: dict[str, Any]
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    input_sha256: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def _hash_input(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\n\n{user}".encode("utf-8")).hexdigest()


def build_evidence_request(chunk: SemanticChunk) -> LLMRequest:
    system = """You extract local economic evidence from financial disclosures.
Do not score the company or infer facts not explicitly supported by this chunk.
Do not use outside knowledge. Every card must cite source_chunk_id and node_ids from the input.
Preserve periods, units, segments, uncertainty, and whether text is a disclosed fact or management claim.
Return an empty cards list when there is no investment-relevant evidence."""
    user = f"""Extract evidence cards from this canonical semantic chunk.

Chunk ID: {chunk.chunk_id}
Allowed node IDs: {json.dumps(chunk.node_ids, ensure_ascii=False)}
Section: {' > '.join(chunk.section_path) or '(root)'}

--- BEGIN CANONICAL CHUNK ---
{chunk.markdown}
--- END CANONICAL CHUNK ---"""
    return LLMRequest(
        task=LLMTask.LOCAL_EVIDENCE_EXTRACTION,
        system=system,
        user=user,
        response_schema=EvidenceExtractionResult.model_json_schema(),
        input_sha256=_hash_input(system, user),
        metadata={
            "chunk_id": chunk.chunk_id,
            "node_ids": chunk.node_ids,
            "source_type": chunk.source_refs[0].source_type.value if chunk.source_refs else "OTHER",
        },
    )


def build_evidence_batch_request(chunks: list[SemanticChunk]) -> LLMRequest:
    if not chunks:
        raise ValueError("evidence batch requires at least one chunk")
    system = """You extract local economic evidence from financial disclosures.
Do not score the company or infer facts not explicitly supported by the supplied chunks.
Do not use outside knowledge. Every card must cite exactly one supplied source_chunk_id and only node_ids allowed for that chunk.
Preserve periods, units, segments, uncertainty, and whether text is a disclosed fact or management claim.
Return an empty cards list when there is no investment-relevant evidence."""
    blocks: list[str] = []
    for chunk in chunks:
        blocks.append(
            f"""## Chunk {chunk.chunk_id}
Allowed node IDs: {json.dumps(chunk.node_ids, ensure_ascii=False)}
Section: {' > '.join(chunk.section_path) or '(root)'}

--- BEGIN CANONICAL CHUNK ---
{chunk.markdown}
--- END CANONICAL CHUNK ---"""
        )
    user = "Extract evidence cards from this bounded canonical chunk batch.\n\n" + "\n\n".join(blocks)
    return LLMRequest(
        task=LLMTask.LOCAL_EVIDENCE_EXTRACTION,
        system=system,
        user=user,
        response_schema=EvidenceBatchExtractionResult.model_json_schema(),
        input_sha256=_hash_input(system, user),
        metadata={
            "chunk_ids": [chunk.chunk_id for chunk in chunks],
            "node_ids_by_chunk": {chunk.chunk_id: chunk.node_ids for chunk in chunks},
        },
    )


def build_section_summary_request(section_path: list[str], cards: list[EvidenceCard]) -> LLMRequest:
    if not cards:
        raise ValueError("section summary requires validated evidence cards")
    system = """You summarize one financial-document section using only supplied evidence cards.
Every positive/negative conclusion must cite an existing evidence_id. Every mechanism, KPI, and uncertainty claim
must also carry one or more evidence_ids from the supplied allowlist. Do not create new facts or evidence.
Preserve counterevidence, uncertainties, mechanism chains, and KPIs that would falsify the claims."""
    cards_json = json.dumps(
        [card.model_dump(mode="json", exclude_none=True) for card in cards],
        ensure_ascii=False,
        indent=2,
    )
    user = f"""Section: {' > '.join(section_path)}
Allowed evidence IDs: {json.dumps([card.evidence_id for card in cards], ensure_ascii=False)}

```json
{cards_json}
```"""
    return LLMRequest(
        task=LLMTask.SECTION_SUMMARY,
        system=system,
        user=user,
        response_schema=SectionSummary.model_json_schema(),
        input_sha256=_hash_input(system, user),
        metadata={"section_path": section_path, "evidence_ids": [card.evidence_id for card in cards]},
    )


def build_moat_request(dossier: CompanyDossier, financial_markdown: str) -> LLMRequest:
    system = """You are the final economic-moat scorer.
Use only the supplied dossier and financial snapshot. Every mechanism and counterevidence must cite an existing evidence_id.
Separate document coverage from model confidence. Do not perform a DCF or invent financial figures.
Score durability and strength after weighing both positive evidence and counterevidence."""
    evidence_json = json.dumps(
        [card.model_dump(mode="json", exclude_none=True) for card in dossier.evidence],
        ensure_ascii=False,
        indent=2,
    )
    summaries_json = json.dumps(
        [summary.model_dump(mode="json", exclude_none=True) for summary in dossier.section_summaries],
        ensure_ascii=False,
        indent=2,
    )
    user = f"""# COMPANY EVIDENCE DOSSIER

Company: {dossier.issuer_name}
As of: {dossier.as_of.isoformat()}

## Business Summary
{dossier.business_summary or 'Not provided'}

## Financial Summary
{dossier.financial_summary or 'Not provided'}

## Structured Financial Snapshot
{financial_markdown}

## Section Summaries
```json
{summaries_json}
```

## Evidence Cards (authoritative IDs)
```json
{evidence_json}
```"""
    return LLMRequest(
        task=LLMTask.FINAL_MOAT_SCORING,
        system=system,
        user=user,
        response_schema=MoatScore.model_json_schema(),
        input_sha256=_hash_input(system, user),
        metadata={"issuer_id": dossier.issuer_id, "as_of": dossier.as_of.isoformat()},
    )


def build_moat_pack_request(dossier: CompanyDossier, pack: CompanyEvidencePack) -> LLMRequest:
    system = """You are the final economic-moat scorer.
Use only the supplied three-layer evidence pack. Every mechanism and counterevidence must cite an evidence_id present in the pack.
Separate document coverage from model confidence. Do not perform DCF arithmetic or invent financial figures.
Weigh positive evidence, counterevidence, reliability, durability, and segment scope before scoring."""
    user = pack.markdown
    return LLMRequest(
        task=LLMTask.FINAL_MOAT_SCORING,
        system=system,
        user=user,
        response_schema=MoatScore.model_json_schema(),
        input_sha256=_hash_input(system, user),
        metadata={
            "issuer_id": dossier.issuer_id,
            "as_of": dossier.as_of.isoformat(),
            "evidence_ids": pack.evidence_ids,
            "raw_chunk_ids": pack.raw_chunk_ids,
        },
    )
