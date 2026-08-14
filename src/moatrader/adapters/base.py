from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

from moatrader.canonical.models import CanonicalDocumentBundle, DocumentMetadata, SourceType


@dataclass(slots=True)
class RawDocument:
    content: bytes
    uri: str | None = None
    fetched_at: datetime | None = None
    media_type: str | None = None
    hints: dict[str, Any] = field(default_factory=dict)


ParsedT = TypeVar("ParsedT")


class SourceAdapter(ABC, Generic[ParsedT]):
    source_type: SourceType

    @abstractmethod
    def detect(self, source: RawDocument) -> bool:
        """Return true only when this adapter can interpret the source."""

    @abstractmethod
    def extract_metadata(self, source: RawDocument) -> DocumentMetadata:
        """Create source-neutral metadata, including a PIT-safe available_at."""

    @abstractmethod
    def parse_structure(self, source: RawDocument) -> ParsedT:
        """Parse into a source-specific intermediate representation."""

    @abstractmethod
    def convert(self, source: RawDocument) -> CanonicalDocumentBundle:
        """Cross the canonical boundary and return a validated bundle."""


class AdapterRegistry:
    def __init__(self, adapters: list[SourceAdapter[Any]] | None = None) -> None:
        self._adapters = list(adapters or [])

    def register(self, adapter: SourceAdapter[Any]) -> None:
        self._adapters.append(adapter)

    def select(self, source: RawDocument) -> SourceAdapter[Any]:
        matches = [adapter for adapter in self._adapters if adapter.detect(source)]
        if not matches:
            raise ValueError("no source adapter recognized the document")
        if len(matches) > 1:
            names = ", ".join(type(adapter).__name__ for adapter in matches)
            raise ValueError(f"ambiguous source; matching adapters: {names}")
        return matches[0]

    def convert(self, source: RawDocument) -> CanonicalDocumentBundle:
        return self.select(source).convert(source)

