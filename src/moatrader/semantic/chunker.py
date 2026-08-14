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
                yield _Atom(
                    node_ids=[node.node_id],
                    section_path=node.section_path,
                    section_role=role,
                    chunk_type="list" if isinstance(node, ListNode) else node.kind,
                    markdown=markdown,
                    source_refs=node.source_refs,
                )

        yield from descend(bundle.ast.children)

    def _table_atoms(self, table: TableNode, role: SectionRole | None) -> Iterable[_Atom]:
        full = self.renderer.render_table(table)
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
            rendered = self.renderer.render_table(table, rows=candidate)
            if group and self.tokens.count(rendered) > self.max_tokens:
                yield self._table_slice(table, role, group)
                group = [row]
            else:
                group = candidate
        if group:
            yield self._table_slice(table, role, group)
        elif not body:
            yield _Atom(
                node_ids=[table.node_id],
                section_path=table.section_path,
                section_role=role,
                chunk_type="table",
                markdown=full,
                source_refs=table.source_refs,
            )

    def _table_slice(self, table: TableNode, role: SectionRole | None, rows: list[TableRow]) -> _Atom:
        return _Atom(
            node_ids=[table.node_id],
            section_path=table.section_path,
            section_role=role,
            chunk_type="table_slice",
            markdown=self.renderer.render_table(table, rows=rows),
            source_refs=table.source_refs,
            metadata={"table_id": table.node_id, "row_start": rows[0].index, "row_end": rows[-1].index},
        )
