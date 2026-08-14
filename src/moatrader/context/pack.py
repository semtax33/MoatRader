from __future__ import annotations

from pydantic import Field

from moatrader.canonical.models import ContractModel
from moatrader.evidence.models import CompanyDossier, EvidenceDirection
from moatrader.financial.snapshot import FinancialSnapshot
from moatrader.semantic.chunker import HeuristicTokenCounter, SemanticChunk, TokenCounter


class CompanyEvidencePack(ContractModel):
    issuer_name: str
    evidence_ids: list[str] = Field(default_factory=list)
    raw_chunk_ids: list[str] = Field(default_factory=list)
    markdown: str
    token_count: int = Field(ge=1)


class EvidencePackBuilder:
    """Build the L1 summary / L2 cards / L3 raw-evidence context contract."""

    def __init__(self, token_counter: TokenCounter | None = None) -> None:
        self.tokens = token_counter or HeuristicTokenCounter()

    def build(
        self,
        dossier: CompanyDossier,
        financial_snapshot: FinancialSnapshot,
        raw_chunks: list[SemanticChunk],
    ) -> CompanyEvidencePack:
        positive = [card for card in dossier.evidence if card.direction == EvidenceDirection.MOAT_POSITIVE]
        negative = [card for card in dossier.evidence if card.direction == EvidenceDirection.MOAT_NEGATIVE]
        lines = [
            "# COMPANY EVIDENCE PACK",
            "",
            "## 0. Metadata",
            "",
            f"- Company: {dossier.issuer_name}",
            f"- Ticker: {dossier.ticker or 'Unknown'}",
            f"- Evidence Cutoff: {dossier.as_of.isoformat()}",
            f"- Source Documents: {', '.join(dossier.source_document_ids)}",
            "",
            "# L1. Structured Summary",
            "",
            "## Business",
            "",
            dossier.business_summary or "_Not supplied._",
            "",
            "## Financial Economics",
            "",
            dossier.financial_summary or "_Not supplied._",
            "",
            financial_snapshot.to_markdown(),
            "",
            "# L2. Evidence Cards",
            "",
            "## Positive Moat Evidence",
            "",
        ]
        lines.extend(self._render_card(card) for card in positive)
        lines.extend(["", "## Counterevidence", ""])
        lines.extend(self._render_card(card) for card in negative)
        neutral = [
            card
            for card in dossier.evidence
            if card.direction not in {EvidenceDirection.MOAT_POSITIVE, EvidenceDirection.MOAT_NEGATIVE}
        ]
        if neutral:
            lines.extend(["", "## Neutral / Contextual Evidence", ""])
            lines.extend(self._render_card(card) for card in neutral)
        lines.extend(["", "# L3. Raw Evidence Appendix", ""])
        for chunk in raw_chunks:
            lines.extend(
                [
                    f"## [{chunk.chunk_id}] {' > '.join(chunk.section_path) or '(root)'}",
                    "",
                    f"- Node IDs: {', '.join(chunk.node_ids)}",
                    "",
                    chunk.markdown,
                    "",
                ]
            )
        markdown = "\n".join(lines).strip() + "\n"
        return CompanyEvidencePack(
            issuer_name=dossier.issuer_name,
            evidence_ids=[card.evidence_id for card in dossier.evidence],
            raw_chunk_ids=[chunk.chunk_id for chunk in raw_chunks],
            markdown=markdown,
            token_count=self.tokens.count(markdown),
        )

    @staticmethod
    def _render_card(card) -> str:
        mechanisms = " → ".join(card.mechanism) if card.mechanism else "Not specified"
        return "\n".join(
            [
                f"### [{card.evidence_id}] {card.evidence_type.value}",
                "",
                f"- Fact: {card.fact}",
                f"- Direction: {card.direction.value}",
                f"- Statement Type: {card.statement_type.value}",
                f"- Strength: {card.strength:.2f}",
                f"- Reliability: {card.reliability:.2f}",
                f"- Mechanism: {mechanisms}",
                f"- Source Chunk: {card.source_chunk_id}",
                f"- Node IDs: {', '.join(card.node_ids)}",
                "",
            ]
        )

