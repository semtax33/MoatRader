from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any, Protocol

from pydantic import Field

from moatrader.canonical.ids import stable_id
from moatrader.canonical.models import (
    CanonicalDocumentBundle,
    CanonicalNode,
    ContractModel,
    ListNode,
    SectionNode,
    SectionRole,
    SourceRef,
    TableNode,
    TableRow,
)
from moatrader.render.markdown import CanonicalMarkdownRenderer


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class HeuristicTokenCounter:
    """Conservative fallback; inject a model tokenizer in production."""

    _piece_re = re.compile(r"[가-힣]|[一-龥]|[A-Za-z0-9_]+|[^\s]", re.UNICODE)

    def count(self, text: str) -> int:
        pieces = self._piece_re.findall(text)
        return max(1, math.ceil(sum(1 if len(piece) == 1 else max(1, len(piece) / 4) for piece in pieces)))


class SemanticChunk(ContractModel):
    chunk_id: str
    document_id: str
    section_path: list[str] = Field(default_factory=list)
    section_role: SectionRole | None = None
    node_ids: list[str] = Field(min_length=1)
    chunk_type: str
    markdown: str
    token_count: int = Field(ge=1)
    source_refs: list[SourceRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class _Atom:
    def __init__(
        self,
        *,
        node_ids: list[str],
        section_path: list[str],
        section_role: SectionRole | None,
        chunk_type: str,
        markdown: str,
        source_refs: list[SourceRef],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.node_ids = node_ids
        self.section_path = section_path
        self.section_role = section_role
        self.chunk_type = chunk_type
        self.markdown = markdown
        self.source_refs = source_refs
        self.metadata = metadata or {}


class SemanticChunker:
    def __init__(
        self,
        *,
        target_tokens: int = 1_500,
        max_tokens: int = 2_500,
        token_counter: TokenCounter | None = None,
        renderer: CanonicalMarkdownRenderer | None = None,
    ) -> None:
        if target_tokens <= 0 or max_tokens < target_tokens:
            raise ValueError("require 0 < target_tokens <= max_tokens")
        self.target_tokens = target_tokens
        self.max_tokens = max_tokens
        self.tokens = token_counter or HeuristicTokenCounter()
        self.renderer = renderer or CanonicalMarkdownRenderer()

    def chunk(self, bundle: CanonicalDocumentBundle) -> list[SemanticChunk]:
        atoms = list(self._atoms(bundle))
        chunks: list[SemanticChunk] = []
        current: list[_Atom] = []
        current_tokens = 0
        current_path: list[str] | None = None

        def flush() -> None:
            nonlocal current, current_tokens, current_path
            if not current:
                return
            markdown = "\n\n".join(atom.markdown for atom in current)
            node_ids = list(dict.fromkeys(node_id for atom in current for node_id in atom.node_ids))
            refs = list({ref.model_dump_json(): ref for atom in current for ref in atom.source_refs}.values())
            metadata: dict[str, Any] = {}
            if len(current) == 1:
                metadata.update(current[0].metadata)
            chunk_type = current[0].chunk_type if len({atom.chunk_type for atom in current}) == 1 else "mixed"
            chunks.append(
                SemanticChunk(
                    chunk_id=stable_id("C", bundle.ast.document_id, node_ids, len(chunks), markdown),
                    document_id=bundle.ast.document_id,
                    section_path=current[0].section_path,
                    section_role=current[0].section_role,
                    node_ids=node_ids,
                    chunk_type=chunk_type,
                    markdown=markdown,
                    token_count=self.tokens.count(markdown),
                    source_refs=refs,
                    metadata=metadata,
                )
            )
            current = []
            current_tokens = 0
            current_path = None

        for atom in atoms:
            atom_tokens = self.tokens.count(atom.markdown)
            path_changed = current_path is not None and current_path != atom.section_path
            would_overflow = current and current_tokens + atom_tokens > self.max_tokens
            target_reached = current and current_tokens >= self.target_tokens
            if path_changed or would_overflow or target_reached:
                flush()
            current.append(atom)
            current_tokens += atom_tokens
            current_path = atom.section_path
            if atom_tokens >= self.max_tokens:
                flush()
        flush()
        return chunks

    def _atoms(self, bundle: CanonicalDocumentBundle) -> Iterable[_Atom]:
        role_by_path: dict[tuple[str, ...], SectionRole | None] = {}

        def descend(nodes: Iterable[CanonicalNode], inherited_role: SectionRole | None = None) -> Iterable[_Atom]:
            for node in nodes:
                if isinstance(node, SectionNode):
                    role = node.role or inherited_role
                    role_by_path[tuple(node.section_path)] = role
                    yield from descend(node.children, role)
                    continue
                role = role_by_path.get(tuple(node.section_path), inherited_role)
                if isinstance(node, TableNode):
                    yield from self._table_atoms(node, role)
                    continue
                markdown = self.renderer.render_node(node)
                if not markdown:
                    continue
                atom = _Atom(
                    node_ids=[node.node_id],
                    section_path=node.section_path,
                    section_role=role,
                    chunk_type="list" if isinstance(node, ListNode) else node.kind,
                    markdown=markdown,
                    source_refs=node.source_refs,
                )
                yield from self._bounded_text_atom(atom)

        yield from descend(bundle.ast.children)

    def _bounded_text_atom(self, atom: _Atom) -> Iterable[_Atom]:
        if self.tokens.count(atom.markdown) <= self.max_tokens:
            yield atom
            return
        prefix = ""
        content = atom.markdown
        if content.startswith("<!--") and "-->\n" in content:
            marker_end = content.index("-->\n") + len("-->\n")
            prefix, content = content[:marker_end], content[marker_end:]
        start = 0
        while start < len(content):
            low, high = start + 1, len(content)
            best = start
            while low <= high:
                end = (low + high) // 2
                rendered = prefix + content[start:end]
                if self.tokens.count(rendered) <= self.max_tokens:
                    best = end
                    low = end + 1
                else:
                    high = end - 1
            if best == start:
                yield atom
                return
            if best < len(content):
                line_break = content.rfind("\n", start + 1, best)
                word_break = content.rfind(" ", start + 1, best)
                boundary = max(line_break, word_break)
                if boundary > start + (best - start) // 2:
                    best = boundary + 1
            yield _Atom(
                node_ids=atom.node_ids,
                section_path=atom.section_path,
                section_role=atom.section_role,
                chunk_type=f"{atom.chunk_type}_fragment",
                markdown=prefix + content[start:best],
                source_refs=atom.source_refs,
                metadata={"text_fragment_start": start, "text_fragment_end": best},
            )
            start = best

    def _table_atoms(self, table: TableNode, role: SectionRole | None) -> Iterable[_Atom]:
        # Notes remain first-class AST nodes and are chunked independently. Repeating
        # every attached footnote in every table slice can make otherwise tiny slices
        # exceed the token budget and duplicates evidence in the LLM context.
        full = self.renderer.render_table(table, include_footnotes=False)
        if self.tokens.count(full) <= self.max_tokens:
            yield _Atom(
                node_ids=[table.node_id],
                section_path=table.section_path,
                section_role=role,
                chunk_type="table",
                markdown=full,
                source_refs=table.source_refs,
            )
            return
        body = table.rows[table.header_row_count :]
        group: list[TableRow] = []
        for row in body:
            candidate = [*group, row]
            rendered = self.renderer.render_table(table, rows=candidate, include_footnotes=False)
            if group and self.tokens.count(rendered) > self.max_tokens:
                yield from self._bounded_table_slices(table, role, group)
                group = [row]
            else:
                group = candidate
        if group:
            yield from self._bounded_table_slices(table, role, group)
        elif not body:
            yield _Atom(
                node_ids=[table.node_id],
                section_path=table.section_path,
                section_role=role,
                chunk_type="table",
                markdown=full,
                source_refs=table.source_refs,
            )

    def _bounded_table_slices(
        self,
        table: TableNode,
        role: SectionRole | None,
        rows: list[TableRow],
    ) -> Iterable[_Atom]:
        rendered = self.renderer.render_table(table, rows=rows, include_footnotes=False)
        if self.tokens.count(rendered) <= self.max_tokens:
            yield self._table_slice(table, role, rows)
            return
        if len(rows) > 1:
            for row in rows:
                yield from self._bounded_table_slices(table, role, [row])
            return

        width = len(rows[0].cells)
        if width <= 1:
            yield from self._single_cell_slices(table, role, rows[0])
            return
        key_columns = [0]
        current = list(key_columns)
        for column in range(1, width):
            single_column_text = self.renderer.render_table(
                table,
                rows=rows,
                columns=[*key_columns, column],
                include_footnotes=False,
            )
            if self.tokens.count(single_column_text) > self.max_tokens:
                if len(current) > len(key_columns):
                    yield self._table_slice(table, role, rows, columns=current)
                yield from self._long_table_cell_slices(
                    table,
                    role,
                    rows[0],
                    key_columns=key_columns,
                    target_column=column,
                )
                current = list(key_columns)
                continue
            candidate = [*current, column]
            candidate_text = self.renderer.render_table(
                table,
                rows=rows,
                columns=candidate,
                include_footnotes=False,
            )
            if len(current) > len(key_columns) and self.tokens.count(candidate_text) > self.max_tokens:
                yield self._table_slice(table, role, rows, columns=current)
                current = [*key_columns, column]
            else:
                current = candidate
        if current:
            if len(current) > len(key_columns):
                yield self._table_slice(table, role, rows, columns=current)

    def _long_table_cell_slices(
        self,
        table: TableNode,
        role: SectionRole | None,
        row: TableRow,
        *,
        key_columns: list[int],
        target_column: int,
    ) -> Iterable[_Atom]:
        text = row.cells[target_column].normalized_text
        start = 0
        while start < len(text):
            low, high = start + 1, len(text)
            best = start
            while low <= high:
                end = (low + high) // 2
                fragment_cell = row.cells[target_column].model_copy(
                    update={"raw_text": text[start:end], "normalized_text": text[start:end]}
                )
                fragment_cells = list(row.cells)
                fragment_cells[target_column] = fragment_cell
                fragment_row = row.model_copy(update={"cells": fragment_cells})
                rendered = self.renderer.render_table(
                    table,
                    rows=[fragment_row],
                    columns=[*key_columns, target_column],
                    include_footnotes=False,
                )
                if self.tokens.count(rendered) <= self.max_tokens:
                    best = end
                    low = end + 1
                else:
                    high = end - 1
            if best == start:
                yield self._table_slice(
                    table,
                    role,
                    [row],
                    columns=[*key_columns, target_column],
                )
                return
            if best < len(text):
                line_break = text.rfind("\n", start + 1, best)
                word_break = text.rfind(" ", start + 1, best)
                boundary = max(line_break, word_break)
                if boundary > start + (best - start) // 2:
                    best = boundary + 1
            fragment_cell = row.cells[target_column].model_copy(
                update={"raw_text": text[start:best], "normalized_text": text[start:best]}
            )
            fragment_cells = list(row.cells)
            fragment_cells[target_column] = fragment_cell
            fragment_row = row.model_copy(update={"cells": fragment_cells})
            yield self._table_slice(
                table,
                role,
                [fragment_row],
                columns=[*key_columns, target_column],
                chunk_type="table_cell_fragment",
                metadata_extra={
                    "cell_column": target_column,
                    "cell_fragment_start": start,
                    "cell_fragment_end": best,
                },
            )
            start = best

    def _single_cell_slices(
        self,
        table: TableNode,
        role: SectionRole | None,
        row: TableRow,
    ) -> Iterable[_Atom]:
        text = row.cells[0].normalized_text
        start = 0
        while start < len(text):
            low, high = start + 1, len(text)
            best = start
            while low <= high:
                end = (low + high) // 2
                fragment_cell = row.cells[0].model_copy(
                    update={"raw_text": text[start:end], "normalized_text": text[start:end]}
                )
                fragment_row = row.model_copy(update={"cells": [fragment_cell]})
                rendered = self.renderer.render_table(
                    table,
                    rows=[fragment_row],
                    columns=[0],
                    include_footnotes=False,
                )
                if self.tokens.count(rendered) <= self.max_tokens:
                    best = end
                    low = end + 1
                else:
                    high = end - 1
            if best == start:
                yield self._table_slice(table, role, [row])
                return
            if best < len(text):
                line_break = text.rfind("\n", start + 1, best)
                word_break = text.rfind(" ", start + 1, best)
                boundary = max(line_break, word_break)
                if boundary > start + (best - start) // 2:
                    best = boundary + 1
            fragment_cell = row.cells[0].model_copy(
                update={"raw_text": text[start:best], "normalized_text": text[start:best]}
            )
            fragment_row = row.model_copy(update={"cells": [fragment_cell]})
            yield self._table_slice(
                table,
                role,
                [fragment_row],
                columns=[0],
                chunk_type="table_cell_fragment",
                metadata_extra={"cell_fragment_start": start, "cell_fragment_end": best},
            )
            start = best

    def _table_slice(
        self,
        table: TableNode,
        role: SectionRole | None,
        rows: list[TableRow],
        *,
        columns: list[int] | None = None,
        chunk_type: str | None = None,
        metadata_extra: dict[str, Any] | None = None,
    ) -> _Atom:
        metadata: dict[str, Any] = {
            "table_id": table.node_id,
            "row_start": rows[0].index,
            "row_end": rows[-1].index,
        }
        if columns is not None:
            metadata["column_indices"] = columns
        if metadata_extra:
            metadata.update(metadata_extra)
        return _Atom(
            node_ids=[table.node_id],
            section_path=table.section_path,
            section_role=role,
            chunk_type=chunk_type or ("table_column_slice" if columns is not None else "table_slice"),
            markdown=self.renderer.render_table(
                table,
                rows=rows,
                columns=columns,
                include_footnotes=False,
            ),
            source_refs=table.source_refs,
            metadata=metadata,
        )
