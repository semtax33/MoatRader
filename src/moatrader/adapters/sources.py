from __future__ import annotations

from zoneinfo import ZoneInfo

from moatrader.adapters.base import RawDocument
from moatrader.adapters.html import BaseHtmlFinancialAdapter
from moatrader.canonical.models import SourceType


def _hint_source(source: RawDocument) -> str:
    return str(source.hints.get("source_type", "")).upper().replace(" ", "_")


class DartHtmlAdapter(BaseHtmlFinancialAdapter):
    source_type = SourceType.DART
    default_zone = ZoneInfo("Asia/Seoul")

    def detect(self, source: RawDocument) -> bool:
        hinted = _hint_source(source)
        return hinted == "DART" or (not hinted and "rcept_no" in source.hints)


class EdgarHtmlAdapter(BaseHtmlFinancialAdapter):
    source_type = SourceType.SEC_EDGAR
    default_zone = ZoneInfo("America/New_York")

    def detect(self, source: RawDocument) -> bool:
        hinted = _hint_source(source)
        return hinted in {"SEC", "SEC_EDGAR", "EDGAR"} or (
            not hinted and "accession_number" in source.hints
        )


class IrHtmlAdapter(BaseHtmlFinancialAdapter):
    source_type = SourceType.IR
    default_zone = ZoneInfo("UTC")

    def detect(self, source: RawDocument) -> bool:
        return _hint_source(source) in {"IR", "INVESTOR_RELATIONS"}

