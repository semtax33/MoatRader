from __future__ import annotations

from datetime import datetime

import fitz

from moatrader.adapters import (
    IrHtmlAdapter,
    IrPdfAdapter,
    PdfOcrBlock,
    PdfOcrResult,
    RawDocument,
)
from moatrader.canonical.models import (
    DocumentType,
    FigureNode,
    SectionNode,
    SourceType,
    TableNode,
)
from moatrader.pipeline import default_registry


def _source(content: bytes) -> RawDocument:
    return RawDocument(
        content=content,
        uri="https://kind.krx.co.kr/example.pdf",
        media_type="application/pdf",
        hints={
            "source_type": "IR",
            "source_document_id": "KINDIR_1_1",
            "issuer_id": "00126380",
            "issuer_name": "샘플",
            "ticker": "005930",
            "title": "2025 IR Book",
            "published_at": "2025-08-29T00:00:00+09:00",
            "available_at": "2025-08-30T00:00:00+09:00",
            "availability_precision": "DAY",
            "availability_source": "fixture",
        },
        fetched_at=datetime.fromisoformat("2025-08-30T00:00:00+09:00"),
    )


def _digital_pdf() -> bytes:
    document = fitz.open()
    page = document.new_page(width=595, height=842)
    page.insert_text((50, 65), "Business Overview", fontsize=22)
    page.insert_text((50, 105), "Long-term customer contracts and recurring revenue.", fontsize=11)
    page.insert_text((390, 135), "Unit: KRW", fontsize=8)
    xs = [50, 240, 370, 510]
    ys = [150, 185, 220, 255]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    values = [
        ["Metric", "2024", "2025"],
        ["Revenue", "1,000", "1,200"],
        ["Margin", "10", "12"],
    ]
    for row, cells in enumerate(values):
        for column, value in enumerate(cells):
            page.insert_text((xs[column] + 6, ys[row] + 23), value, fontsize=9)
    return document.tobytes()


def _image_pdf() -> bytes:
    image_document = fitz.open()
    image_page = image_document.new_page(width=400, height=400)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 300, 300), False)
    pixmap.clear_with(220)
    image_page.insert_image(image_page.rect, stream=pixmap.tobytes("png"))
    return image_document.tobytes()


def test_ir_pdf_adapter_preserves_page_bbox_table_and_management_claim_semantics() -> None:
    source = _source(_digital_pdf())
    bundle = IrPdfAdapter().convert(source)
    nodes = list(bundle.ast.walk())
    tables = [node for node in nodes if isinstance(node, TableNode)]

    assert bundle.metadata.source_type == SourceType.IR
    assert bundle.metadata.document_type == DocumentType.IR_PRESENTATION
    assert bundle.metadata.available_at.isoformat() == "2025-08-30T00:00:00+09:00"
    assert bundle.metadata.source_specific["statement_type"] == "MANAGEMENT_CLAIM"
    assert any(isinstance(node, SectionNode) for node in nodes)
    assert len(tables) == 1
    assert tables[0].source_refs[0].page == 1
    assert tables[0].source_refs[0].bbox is not None
    assert tables[0].rows[1].cells[1].numeric_value == 1000
    assert tables[0].rows[1].cells[1].source_ref is not None
    assert tables[0].unit is not None
    assert tables[0].unit.currency == "KRW"
    assert bundle.quality.numeric_retention == 1.0
    assert bundle.facts == []


def test_ir_pdf_adapter_fails_closed_when_ocr_is_required() -> None:
    bundle = IrPdfAdapter().convert(_source(_image_pdf()))

    assert any("OCR_REQUIRED" in warning for warning in bundle.quality.warnings)
    assert any(isinstance(node, FigureNode) for node in bundle.ast.walk())


class FakeOcrAdapter:
    name = "fake-korean-ocr/1"

    def extract_page(self, _page: object, *, dpi: int) -> PdfOcrResult:
        return PdfOcrResult(
            blocks=(
                PdfOcrBlock(
                    text="수주잔고 1,234억원",
                    bbox=(40.0, 50.0, 250.0, 80.0),
                    confidence=0.99,
                ),
            ),
            dpi=dpi,
            mean_confidence=0.99,
            engine=self.name,
        )


def test_ir_pdf_adapter_recovers_ocr_text_with_page_bbox_provenance() -> None:
    bundle = IrPdfAdapter(ocr_adapter=FakeOcrAdapter()).convert(_source(_image_pdf()))
    nodes = list(bundle.ast.walk())
    recovered = [node for node in nodes if "수주잔고" in node.raw_text]

    assert len(recovered) == 1
    assert recovered[0].source_refs[0].page == 1
    assert recovered[0].source_refs[0].bbox is not None
    assert recovered[0].attributes["text_engine"] == "fake-korean-ocr/1"
    assert recovered[0].attributes["ocr_confidence"] == 0.99
    assert bundle.metadata.source_specific["ocr"]["applied_page_count"] == 1
    assert bundle.metadata.source_specific["ocr"]["failed_page_count"] == 0
    assert not any("no reviewed OCR adapter" in warning for warning in bundle.quality.warnings)


def test_ir_pdf_text_retention_counts_section_titles() -> None:
    bundle = IrPdfAdapter().convert(_source(_digital_pdf()))

    assert bundle.quality.text_retention == 1.0


def test_pdf_and_html_ir_detection_is_unambiguous() -> None:
    source = _source(_digital_pdf())

    assert IrPdfAdapter().detect(source)
    assert not IrHtmlAdapter().detect(source)
    assert type(default_registry().select(source)).__name__ == "IrPdfAdapter"
