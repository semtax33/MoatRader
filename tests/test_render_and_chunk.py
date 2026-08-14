from __future__ import annotations

from moatrader.render import CanonicalMarkdownRenderer
from moatrader.semantic import HeuristicTokenCounter, SemanticChunker

from conftest import build_dart_bundle
from test_html_adapter import DART_TABLE_HTML


def test_structured_markdown_has_table_semantics_and_provenance():
    bundle = build_dart_bundle(DART_TABLE_HTML)
    markdown = CanonicalMarkdownRenderer().render_document(bundle)
    assert "# CANONICAL FINANCIAL DOCUMENT" in markdown
    assert "Unit: 백만원 (KRW_MILLION)" in markdown
    assert "Section: II. 사업의 내용 > 1. 주요 제품 및 서비스" in markdown
    assert "2025년 > 매출" in markdown
    assert "전자부품 | 카메라모듈 | (700)" in markdown
    assert "Footnote 주1): 주1) 연결기준입니다." in markdown
    assert "DART:20250515000123 @" in markdown


def test_semantic_chunks_keep_section_path_and_do_not_split_rows():
    rows = "".join(f"<tr><td>전자부품</td><td>제품{i}</td><td>{i * 100}</td></tr>" for i in range(1, 30))
    html = f"""<html><body><h1>II. 사업의 내용</h1><h2>1. 주요 제품</h2>
    <p>(단위: 백만원)</p><table><tr><th>사업부</th><th>제품</th><th>매출</th></tr>{rows}</table></body></html>"""
    bundle = build_dart_bundle(html)
    chunker = SemanticChunker(target_tokens=100, max_tokens=180, token_counter=HeuristicTokenCounter())
    chunks = chunker.chunk(bundle)
    table_chunks = [chunk for chunk in chunks if chunk.chunk_type == "table_slice"]
    assert len(table_chunks) > 1
    assert all(chunk.section_path == ["II. 사업의 내용", "1. 주요 제품"] for chunk in table_chunks)
    assert all("| 사업부 | 제품 | 매출 |" in chunk.markdown for chunk in table_chunks)
    covered_rows = sum(chunk.metadata["row_end"] - chunk.metadata["row_start"] + 1 for chunk in table_chunks)
    assert covered_rows == 29
    assert len({chunk.node_ids[0] for chunk in table_chunks}) == 1

