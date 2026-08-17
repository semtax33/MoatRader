from moatrader.adapters.base import AdapterRegistry, RawDocument, SourceAdapter
from moatrader.adapters.ocr import PaddlePdfOcrAdapter, PdfOcrAdapter, PdfOcrBlock, PdfOcrResult
from moatrader.adapters.pdf import (
    IrPdfAdapter,
    enrich_ir_table_semantics,
    enrich_pdf_table_semantics,
)
from moatrader.adapters.sources import DartHtmlAdapter, EdgarHtmlAdapter, IrHtmlAdapter
from moatrader.adapters.synalyst_pdf import IndustryPdfAdapter

__all__ = [
    "AdapterRegistry",
    "RawDocument",
    "SourceAdapter",
    "DartHtmlAdapter",
    "EdgarHtmlAdapter",
    "IrHtmlAdapter",
    "IrPdfAdapter",
    "IndustryPdfAdapter",
    "enrich_ir_table_semantics",
    "enrich_pdf_table_semantics",
    "PaddlePdfOcrAdapter",
    "PdfOcrAdapter",
    "PdfOcrBlock",
    "PdfOcrResult",
]
