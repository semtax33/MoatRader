from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

from moatrader.evidence.models import (
    EconomicScope,
    EvidenceCard,
    EvidenceDirection,
    EvidenceRelation,
    EvidenceRelationType,
    EvidenceType,
    STRUCTURAL_MOAT_TYPES,
)
from moatrader.semantic.chunker import SemanticChunk


LEDGER_SCHEMA_VERSION = "moatrader-evidence-ledger/1"
_PERSISTENT_COUNTER_TYPES = STRUCTURAL_MOAT_TYPES | {
    EvidenceType.COMPETITIVE_THREAT,
    EvidenceType.CUSTOMER_CONCENTRATION,
    EvidenceType.SUBSTITUTION_RISK,
    EvidenceType.TECHNOLOGY_RISK,
    EvidenceType.CAPITAL_INTENSITY,
}


@dataclass(frozen=True)
class EvidenceLedgerMerge:
    cards: list[EvidenceCard]
    current_evidence_ids: set[str]
    active_source_document_ids: list[str]
    carried_evidence_count: int


class EvidenceLedgerStore:
    """PIT-safe structural evidence history for one fresh experiment."""

    def __init__(self, root: str | Path, *, experiment_id: str) -> None:
        self.root = Path(root).resolve() / experiment_id
        self.root.mkdir(parents=True, exist_ok=True)
        self._guard = Lock()
        self._ticker_locks: dict[str, Lock] = {}

    def merge(
        self,
        ticker: str,
        *,
        as_of: datetime,
        current_cards: list[EvidenceCard],
        chunks: list[SemanticChunk],
        document_available_at: dict[str, datetime],
    ) -> EvidenceLedgerMerge:
        current_by_id = {card.evidence_id: card for card in current_cards}
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        with self._lock(ticker):
            payload = self._load(ticker)
            records = {record["evidence_id"]: record for record in payload["records"]}
            for card in current_cards:
                if not self._persistent(card):
                    continue
                chunk = chunk_by_id[card.source_chunk_id]
                source_document_id = chunk.document_id
                available_at = document_available_at[source_document_id]
                existing = records.get(card.evidence_id)
                if existing is None:
                    records[card.evidence_id] = {
                        "evidence_id": card.evidence_id,
                        "card": card.model_dump(mode="json", exclude_none=True),
                        "source_document_id": source_document_id,
                        "source_available_at": available_at.isoformat(),
                        "valid_from": as_of.isoformat(),
                        "valid_to": None,
                        "first_seen_at": as_of.isoformat(),
                        "last_confirmed_at": as_of.isoformat(),
                        "superseded_by": None,
                        "retracted_by": None,
                    }
                else:
                    existing["card"] = card.model_dump(mode="json", exclude_none=True)
                    existing["last_confirmed_at"] = as_of.isoformat()
            payload["records"] = sorted(records.values(), key=lambda item: item["evidence_id"])
            self._write(ticker, payload)

            active_records = [record for record in payload["records"] if self._active(record, as_of)]
            carried = [
                EvidenceCard.model_validate(record["card"])
                for record in active_records
                if record["evidence_id"] not in current_by_id
            ]
            carried.sort(key=lambda card: card.evidence_id)
            cards = [*carried, *current_cards]
            return EvidenceLedgerMerge(
                cards=cards,
                current_evidence_ids=set(current_by_id),
                active_source_document_ids=sorted(
                    {str(record["source_document_id"]) for record in active_records}
                ),
                carried_evidence_count=len(carried),
            )

    def apply_relations(
        self,
        ticker: str,
        *,
        as_of: datetime,
        current_evidence_ids: set[str],
        relations: list[EvidenceRelation],
    ) -> set[str]:
        invalidated: set[str] = set()
        with self._lock(ticker):
            payload = self._load(ticker)
            records = {record["evidence_id"]: record for record in payload["records"]}
            for relation in relations:
                if (
                    relation.from_evidence_id not in current_evidence_ids
                    or relation.to_evidence_id in current_evidence_ids
                    or relation.to_evidence_id not in records
                ):
                    continue
                target = records[relation.to_evidence_id]
                if relation.relation in {
                    EvidenceRelationType.DUPLICATES,
                    EvidenceRelationType.UPDATES,
                }:
                    target["valid_to"] = as_of.isoformat()
                    target["superseded_by"] = relation.from_evidence_id
                    invalidated.add(relation.to_evidence_id)
                elif relation.relation == EvidenceRelationType.CONTRADICTS:
                    target["valid_to"] = as_of.isoformat()
                    target["retracted_by"] = relation.from_evidence_id
                    invalidated.add(relation.to_evidence_id)
            if invalidated:
                payload["records"] = sorted(records.values(), key=lambda item: item["evidence_id"])
                self._write(ticker, payload)
        return invalidated

    def records(self, ticker: str) -> list[dict[str, Any]]:
        with self._lock(ticker):
            return list(self._load(ticker)["records"])

    def active_source_document_ids(self, ticker: str, *, as_of: datetime) -> list[str]:
        with self._lock(ticker):
            return sorted(
                {
                    str(record["source_document_id"])
                    for record in self._load(ticker)["records"]
                    if self._active(record, as_of)
                }
            )

    @staticmethod
    def _persistent(card: EvidenceCard) -> bool:
        if card.economic_scope not in {EconomicScope.COMPANY, EconomicScope.SEGMENT}:
            return False
        if card.direction == EvidenceDirection.MOAT_POSITIVE:
            return card.evidence_type in STRUCTURAL_MOAT_TYPES
        if card.direction == EvidenceDirection.MOAT_NEGATIVE:
            return card.evidence_type in _PERSISTENT_COUNTER_TYPES
        return False

    @staticmethod
    def _active(record: dict[str, Any], as_of: datetime) -> bool:
        valid_from = datetime.fromisoformat(str(record["valid_from"]))
        valid_to_text = record.get("valid_to")
        valid_to = datetime.fromisoformat(str(valid_to_text)) if valid_to_text else None
        source_available = datetime.fromisoformat(str(record["source_available_at"]))
        return source_available <= as_of and valid_from <= as_of and (valid_to is None or valid_to > as_of)

    def _load(self, ticker: str) -> dict[str, Any]:
        path = self._path(ticker)
        if not path.is_file():
            return {"schema_version": LEDGER_SCHEMA_VERSION, "ticker": ticker, "records": []}
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != LEDGER_SCHEMA_VERSION or payload.get("ticker") != ticker:
            raise ValueError(f"invalid evidence ledger contract: {path}")
        return payload

    def _write(self, ticker: str, payload: dict[str, Any]) -> None:
        path = self._path(ticker)
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)

    def _path(self, ticker: str) -> Path:
        return self.root / f"{ticker}.json"

    def _lock(self, ticker: str) -> Lock:
        with self._guard:
            return self._ticker_locks.setdefault(ticker, Lock())
