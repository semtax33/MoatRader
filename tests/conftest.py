from __future__ import annotations

from moatrader.adapters import DartHtmlAdapter, RawDocument


def build_dart_bundle(html: str, **overrides):
    hints = {
        "source_type": "DART",
        "rcept_no": "20250515000123",
        "corp_code": "00126380",
        "issuer_name": "테스트전자",
        "stock_code": "005930",
        "report_name": "사업보고서",
        "available_at": "2025-05-15T09:01:02+09:00",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
        "language": "ko",
    }
    hints.update(overrides)
    return DartHtmlAdapter().convert(
        RawDocument(content=html.encode("utf-8"), uri="https://dart.example/report.html", hints=hints)
    )

