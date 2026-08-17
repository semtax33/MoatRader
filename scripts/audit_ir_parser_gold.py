from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import fitz

from moatrader.canonical.models import CanonicalDocumentBundle, TableNode
from moatrader.quality import assess_parser_quality


SCHEMA_VERSION = "moatrader-ir-parser-gold-audit/1"
_NUMBER = re.compile(r"^\(?[-+]?\d[\d,]*(?:\.\d+)?\)?%?$")


def _numeric_text(value: str) -> str | None:
    text = value.strip().replace("−", "-")
    return text if _NUMBER.fullmatch(text) else None


def _pick_documents(
    bundles: list[tuple[Path, CanonicalDocumentBundle]],
    sample_size: int,
    seed: str,
) -> list[tuple[Path, CanonicalDocumentBundle]]:
    by_ticker: dict[str, list[tuple[Path, CanonicalDocumentBundle]]] = {}
    for item in bundles:
        ticker = item[1].metadata.ticker or ""
        by_ticker.setdefault(ticker, []).append(item)
    picked: list[tuple[Path, CanonicalDocumentBundle]] = []
    for index, ticker in enumerate(sorted(by_ticker)):
        choices = sorted(
            by_ticker[ticker], key=lambda item: item[1].metadata.available_at
        )
        picked.append(choices[0] if index % 2 == 0 else choices[-1])
        if len(picked) == sample_size:
            return picked
    remaining = [item for item in bundles if item not in picked]
    remaining.sort(
        key=lambda item: hashlib.sha256(
            f"{seed}:{item[1].metadata.source_document_id}".encode()
        ).hexdigest()
    )
    return [*picked, *remaining[: max(0, sample_size - len(picked))]]


def _table_metrics(page: fitz.Page, table: TableNode) -> dict[str, Any]:
    numeric_cells = [
        cell
        for row in table.rows
        for cell in row.cells
        if cell.numeric_value is not None
    ]
    provenance_complete = 0
    coordinate_matches = 0
    for cell in numeric_cells:
        reference = cell.source_ref
        if reference is None or reference.page is None or reference.bbox is None:
            continue
        provenance_complete += 1
        bbox = reference.bbox
        words = page.get_text(
            "words",
            clip=fitz.Rect(bbox.x0, bbox.y0, bbox.x1, bbox.y1),
            sort=True,
        )
        raw_numeric = _numeric_text(cell.raw_text)
        if raw_numeric is not None and any(
            _numeric_text(str(word[4])) == raw_numeric for word in words
        ):
            coordinate_matches += 1
    return {
        "numeric_cell_count": len(numeric_cells),
        "numeric_provenance_complete_count": provenance_complete,
        "numeric_coordinate_match_count": coordinate_matches,
        "numeric_provenance_completeness": (
            provenance_complete / len(numeric_cells) if numeric_cells else None
        ),
        "numeric_coordinate_match_rate": (
            coordinate_matches / len(numeric_cells) if numeric_cells else None
        ),
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.experiment_root).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    render_dir = output / "rendered"
    render_dir.mkdir(parents=True, exist_ok=True)
    bundles: list[tuple[Path, CanonicalDocumentBundle]] = []
    for path in (root / "parsed").rglob("bundle.json"):
        bundle = CanonicalDocumentBundle.model_validate_json(
            path.read_text(encoding="utf-8-sig")
        )
        bundles.append((path, bundle))
    selected = _pick_documents(bundles, args.sample_size, args.seed)
    rows: list[dict[str, Any]] = []
    for bundle_path, bundle in selected:
        document_id = bundle.metadata.source_document_id
        pdfs = list((root / "bronze" / "kind-ir" / document_id).rglob("document.pdf"))
        if len(pdfs) != 1:
            raise ValueError(f"{document_id}: expected exactly one source PDF")
        tables = [node for node in bundle.ast.walk() if isinstance(node, TableNode)]
        table = max(
            tables,
            key=lambda node: sum(
                cell.numeric_value is not None
                for row in node.rows
                for cell in row.cells
            ),
            default=None,
        )
        if table is None or not table.source_refs or table.source_refs[0].page is None:
            rows.append(
                {
                    "ticker": bundle.metadata.ticker,
                    "source_document_id": document_id,
                    "listed_on": bundle.metadata.source_specific.get("listed_on"),
                    "status": "NO_NUMERIC_TABLE",
                }
            )
            continue
        page_number = int(table.source_refs[0].page)
        document = fitz.open(pdfs[0])
        page = document[page_number - 1]
        metrics = _table_metrics(page, table)
        render_path = render_dir / (
            f"{bundle.metadata.ticker}-{bundle.metadata.source_specific.get('listed_on')}-"
            f"page-{page_number:02d}.png"
        )
        page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False).save(render_path)
        quality = assess_parser_quality(bundle)
        rows.append(
            {
                "ticker": bundle.metadata.ticker,
                "source_document_id": document_id,
                "listed_on": bundle.metadata.source_specific.get("listed_on"),
                "parser_version": bundle.metadata.parser_version,
                "page": page_number,
                "table_node_id": table.node_id,
                "table_strategy": table.attributes.get("table_extraction_strategy"),
                "row_count": len(table.rows),
                "column_count": max((len(row.cells) for row in table.rows), default=0),
                "quality_gate_passed": quality.passed,
                "quality_failures": quality.failures,
                "render_path": str(render_path),
                "status": "PASS" if quality.passed else "FAIL",
                **metrics,
            }
        )
    numeric_total = sum(int(row.get("numeric_cell_count", 0)) for row in rows)
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_root": str(root),
        "selection_uses_return_data": False,
        "sample_size": len(rows),
        "quality_gate_pass_count": sum(
            row.get("quality_gate_passed") is True for row in rows
        ),
        "numeric_cell_count": numeric_total,
        "numeric_provenance_completeness": (
            sum(int(row.get("numeric_provenance_complete_count", 0)) for row in rows)
            / numeric_total
            if numeric_total
            else None
        ),
        "numeric_coordinate_match_rate": (
            sum(int(row.get("numeric_coordinate_match_count", 0)) for row in rows)
            / numeric_total
            if numeric_total
            else None
        ),
        "rows": rows,
    }
    (output / "parser-gold-audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# IR Parser Gold Audit",
        "",
        f"- Sample: {len(rows)} documents",
        f"- Quality gate pass: {result['quality_gate_pass_count']}/{len(rows)}",
        f"- Numeric provenance completeness: {result['numeric_provenance_completeness']}",
        f"- Numeric coordinate match rate: {result['numeric_coordinate_match_rate']}",
        "- Return data used for selection: no",
        "",
        "| Ticker | Listed | Page | Grid | Numeric | Coordinate match | Quality |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.get('ticker')} | {row.get('listed_on')} | {row.get('page', '')} | "
            f"{row.get('row_count', '')}x{row.get('column_count', '')} | "
            f"{row.get('numeric_cell_count', '')} | "
            f"{row.get('numeric_coordinate_match_rate', '')} | {row['status']} |"
        )
    (output / "parser-gold-audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit page-grounded IR PDF table quality")
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", default="ir-parser-gold-v1")
    return parser


if __name__ == "__main__":
    print(json.dumps(audit(build_parser().parse_args()), ensure_ascii=False, indent=2))
