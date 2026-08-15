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
    assert "Structured facts: 0/0" in markdown


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


def test_wide_table_rows_are_bounded_with_column_slices() -> None:
    headers = "".join(f"<th>지표{i}</th>" for i in range(1, 41))
    values = "".join(f"<td>{i * 100:,}</td>" for i in range(1, 41))
    bundle = build_dart_bundle(
        f"""<html><body><h1>III. 재무제표</h1><table>
        <tr><th>구분</th>{headers}</tr><tr><td>당기</td>{values}</tr>
        </table></body></html>"""
    )
    chunker = SemanticChunker(
        target_tokens=100,
        max_tokens=180,
        token_counter=HeuristicTokenCounter(),
    )
    chunks = chunker.chunk(bundle)
    table_chunks = [chunk for chunk in chunks if chunk.chunk_type == "table_column_slice"]

    assert len(table_chunks) > 1
    assert all(chunk.token_count <= 180 for chunk in table_chunks)
    assert all(chunk.metadata["column_indices"][0] == 0 for chunk in table_chunks)
    covered = {
        column
        for chunk in table_chunks
        for column in chunk.metadata["column_indices"]
    }
    assert covered == set(range(41))


def test_long_single_cell_table_is_split_without_oversized_chunks() -> None:
    disclosure = "\n".join(
        f"회계정책 문단 {index}: 중요한 정책 설명과 적용 기준을 상세히 기술합니다."
        for index in range(120)
    )
    bundle = build_dart_bundle(
        f"<html><body><h1>III. 재무제표 주석</h1><table><tr><td>{disclosure}</td></tr></table></body></html>"
    )
    chunks = SemanticChunker(
        target_tokens=100,
        max_tokens=180,
        token_counter=HeuristicTokenCounter(),
    ).chunk(bundle)
    fragments = [chunk for chunk in chunks if chunk.chunk_type == "table_cell_fragment"]

    assert len(fragments) > 1
    assert all(chunk.token_count <= 180 for chunk in fragments)
    ranges = [
        (chunk.metadata["cell_fragment_start"], chunk.metadata["cell_fragment_end"])
        for chunk in fragments
    ]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(disclosure)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))


def test_long_cell_inside_wide_table_is_split_with_key_column_context() -> None:
    disclosure = "\n".join(
        f"conversion condition {index}: adjustment and exercise terms remain applicable."
        for index in range(300)
    )
    bundle = build_dart_bundle(
        f"""<html><body><h1>Preferred shares</h1><table>
        <tr><th>Category</th><th>Subcategory</th><th>Terms</th><th>Amount</th></tr>
        <tr><td>Share terms</td><td>Conversion</td><td>{disclosure}</td><td>1,000</td></tr>
        </table></body></html>"""
    )
    chunks = SemanticChunker(
        target_tokens=100,
        max_tokens=180,
        token_counter=HeuristicTokenCounter(),
    ).chunk(bundle)
    fragments = [
        chunk
        for chunk in chunks
        if chunk.chunk_type == "table_cell_fragment" and chunk.metadata.get("cell_column") == 2
    ]

    assert len(fragments) > 1
    assert all(chunk.token_count <= 180 for chunk in chunks)
    assert all("Share terms" in chunk.markdown for chunk in fragments)
    ranges = [
        (chunk.metadata["cell_fragment_start"], chunk.metadata["cell_fragment_end"])
        for chunk in fragments
    ]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(disclosure)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))


def test_long_paragraph_is_split_without_oversized_chunks() -> None:
    disclosure = " ".join(
        f"산업 특성과 진입장벽에 관한 근거 문장 {index}." for index in range(500)
    )
    bundle = build_dart_bundle(
        f"<html><body><h1>II. 사업의 내용</h1><p>{disclosure}</p></body></html>"
    )
    chunks = SemanticChunker(
        target_tokens=100,
        max_tokens=180,
        token_counter=HeuristicTokenCounter(),
    ).chunk(bundle)
    fragments = [chunk for chunk in chunks if chunk.chunk_type == "paragraph_fragment"]

    assert len(fragments) > 1
    assert all(chunk.token_count <= 180 for chunk in fragments)
    ranges = [
        (chunk.metadata["text_fragment_start"], chunk.metadata["text_fragment_end"])
        for chunk in fragments
    ]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(disclosure)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))


def test_attached_footnote_is_chunked_once_instead_of_repeated_per_table_slice() -> None:
    headers = "".join(f"<th>Column {index}</th>" for index in range(1, 31))
    values = "".join(f"<td>{index * 100:,}</td>" for index in range(1, 31))
    footnote_body = " ".join(f"material disclosure sentence {index}." for index in range(300))
    bundle = build_dart_bundle(
        f"""<html><body><h1>Financial statements</h1><table>
        <tr><th>Category</th>{headers}</tr><tr><td>Current</td>{values}</tr>
        </table><p>주1) {footnote_body}</p></body></html>"""
    )
    chunks = SemanticChunker(
        target_tokens=100,
        max_tokens=180,
        token_counter=HeuristicTokenCounter(),
    ).chunk(bundle)
    table_chunks = [chunk for chunk in chunks if chunk.chunk_type == "table_column_slice"]
    note_chunks = [chunk for chunk in chunks if chunk.chunk_type == "note_fragment"]

    assert len(table_chunks) > 1
    assert note_chunks
    assert all(chunk.token_count <= 180 for chunk in chunks)
    assert all("material disclosure sentence" not in chunk.markdown for chunk in table_chunks)
    ranges = [
        (chunk.metadata["text_fragment_start"], chunk.metadata["text_fragment_end"])
        for chunk in note_chunks
    ]
    assert ranges[0][0] == 0
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert sum(end - start for start, end in ranges) == ranges[-1][1]
