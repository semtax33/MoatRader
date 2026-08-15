from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from statistics import fmean
from typing import Any

from moatrader.adapters import DartHtmlAdapter, RawDocument
from moatrader.financial import FinancialSnapshotBuilder
from moatrader.quality import assess_parser_quality
from moatrader.semantic import SemanticChunker


KEY_CONCEPTS = ("REVENUE", "EBIT", "NET_INCOME", "CFO", "CAPEX", "CASH")
CSV_FIELDS = (
    "source_document_id",
    "issuer_id",
    "issuer_name",
    "ticker",
    "report_name",
    "report_date",
    "parser_version",
    "is_amendment",
    "reported_as_amendment",
    "amendment_link_status",
    "amends_document_id",
    "is_periodic_report",
    "quality_passed",
    "text_retention",
    "raw_table_count",
    "ast_table_count",
    "numeric_retention",
    "raw_numeric_cell_count",
    "numeric_cell_count",
    "structured_fact_retention",
    "raw_structured_fact_count",
    "structured_fact_count",
    "snapshot_concepts",
    "snapshot_point_count",
    "chunk_count",
    "max_chunk_tokens",
    "oversized_chunk_count",
    "warnings",
    "failures",
    "status",
    "error",
)


def _load_document(bronze_root: Path, latest_path: Path) -> tuple[Path, dict[str, Any]]:
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    metadata_path = bronze_root / latest["metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_path = bronze_root / metadata["storage"]["primary_path"]
    return source_path, metadata


def _audit_one(task: tuple[str, str]) -> dict[str, Any]:
    bronze_root = Path(task[0])
    latest_path = Path(task[1])
    source_document_id = latest_path.parent.name
    try:
        source_path, metadata = _load_document(bronze_root, latest_path)
        bundle = DartHtmlAdapter().convert(
            RawDocument(
                content=source_path.read_bytes(),
                uri=metadata.get("primary_document_url"),
                hints=metadata,
            )
        )
        assessment = assess_parser_quality(bundle)
        snapshot = FinancialSnapshotBuilder().build(
            [bundle],
            as_of=bundle.metadata.available_at,
        )
        chunks = SemanticChunker().chunk(bundle)
        concepts = sorted(snapshot.series_index())
        reported_as_amendment = bool(
            bundle.metadata.source_specific.get("reported_as_amendment", False)
        )
        normalized_report_name = str(
            bundle.metadata.source_specific.get("normalized_report_name")
            or metadata.get("report_name")
            or ""
        ).strip()
        is_periodic_report = bool(
            re.match(r"^(?:사업보고서|반기보고서|분기보고서)\s*\(", normalized_report_name)
        )
        return {
            "source_document_id": source_document_id,
            "issuer_id": metadata.get("issuer_id") or "",
            "issuer_name": metadata.get("issuer_name") or "",
            "ticker": metadata.get("ticker") or "",
            "report_name": metadata.get("report_name") or "",
            "report_date": metadata.get("report_date") or "",
            "parser_version": bundle.metadata.parser_version,
            "is_amendment": bundle.metadata.is_amendment,
            "reported_as_amendment": reported_as_amendment,
            "amendment_link_status": bundle.metadata.source_specific.get("amendment_link_status") or "",
            "amends_document_id": bundle.metadata.amends_document_id or "",
            "is_periodic_report": is_periodic_report,
            "quality_passed": assessment.passed,
            "text_retention": bundle.quality.text_retention,
            "raw_table_count": bundle.quality.raw_table_count,
            "ast_table_count": bundle.quality.ast_table_count,
            "numeric_retention": bundle.quality.numeric_retention,
            "raw_numeric_cell_count": bundle.quality.raw_numeric_cell_count,
            "numeric_cell_count": bundle.quality.numeric_cell_count,
            "structured_fact_retention": bundle.quality.structured_fact_retention,
            "raw_structured_fact_count": bundle.quality.raw_structured_fact_count,
            "structured_fact_count": bundle.quality.structured_fact_count,
            "snapshot_concepts": "|".join(concepts),
            "snapshot_point_count": sum(len(series.points) for series in snapshot.series),
            "chunk_count": len(chunks),
            "max_chunk_tokens": max((chunk.token_count for chunk in chunks), default=0),
            "oversized_chunk_count": sum(chunk.token_count > 2_500 for chunk in chunks),
            "warnings": " | ".join(bundle.quality.warnings),
            "failures": " | ".join(assessment.failures),
            "status": "OK" if assessment.passed else "QUALITY_FAILED",
            "error": "",
        }
    except Exception as exc:  # pragma: no cover - exercised against external corpora
        return {
            **{field: "" for field in CSV_FIELDS},
            "source_document_id": source_document_id,
            "quality_passed": False,
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _ratio_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) not in {None, ""}]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["status"] != "ERROR"]
    reported_amendments = [row for row in successful if row.get("reported_as_amendment")]
    non_amendments = [
        row
        for row in successful
        if not row.get("is_amendment") and not row.get("reported_as_amendment")
    ]
    periodic_non_amendments = [row for row in non_amendments if row.get("is_periodic_report")]
    summary: dict[str, Any] = {
        "document_count": len(rows),
        "successful_count": len(successful),
        "quality_passed_count": sum(row["status"] == "OK" for row in rows),
        "quality_failed_count": sum(row["status"] == "QUALITY_FAILED" for row in rows),
        "error_count": sum(row["status"] == "ERROR" for row in rows),
        "linked_amendment_count": sum(bool(row.get("is_amendment")) for row in successful),
        "reported_amendment_count": len(reported_amendments),
        "unresolved_reported_amendment_count": sum(
            bool(row.get("reported_as_amendment")) and not bool(row.get("is_amendment"))
            for row in successful
        ),
        "empty_snapshot_count": sum(not row.get("snapshot_concepts") for row in successful),
        "empty_snapshot_non_amendment_count": sum(
            not row.get("snapshot_concepts") for row in non_amendments
        ),
        "empty_snapshot_periodic_non_amendment_count": sum(
            not row.get("snapshot_concepts") for row in periodic_non_amendments
        ),
        "oversized_chunk_document_count": sum(
            int(row.get("oversized_chunk_count") or 0) > 0 for row in successful
        ),
        "max_chunk_tokens": max(
            (int(row.get("max_chunk_tokens") or 0) for row in successful),
            default=0,
        ),
        "missing_key_concepts": {
            concept: sum(
                concept not in str(row.get("snapshot_concepts") or "").split("|")
                for row in successful
            )
            for concept in KEY_CONCEPTS
        },
        "missing_key_concepts_non_amendments": {
            concept: sum(
                concept not in str(row.get("snapshot_concepts") or "").split("|")
                for row in non_amendments
            )
            for concept in KEY_CONCEPTS
        },
        "missing_key_concepts_periodic_non_amendments": {
            concept: sum(
                concept not in str(row.get("snapshot_concepts") or "").split("|")
                for row in periodic_non_amendments
            )
            for concept in KEY_CONCEPTS
        },
    }
    for key in (
        "text_retention",
        "numeric_retention",
        "structured_fact_retention",
    ):
        values = _ratio_values(successful, key)
        summary[key] = {
            "minimum": min(values) if values else None,
            "average": fmean(values) if values else None,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reparse DART Bronze documents and emit parser/snapshot/chunk coverage artifacts."
    )
    parser.add_argument("--bronze-root", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--rcept-no", action="append")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--fail-on-quality", action="store_true")
    args = parser.parse_args()
    if args.max_documents is not None and args.max_documents <= 0:
        parser.error("--max-documents must be positive")
    if args.workers <= 0:
        parser.error("--workers must be positive")

    bronze_root = args.bronze_root.resolve()
    latest_paths = sorted((bronze_root / "dart").glob("*/latest.json"))
    if args.rcept_no:
        selected = set(args.rcept_no)
        latest_paths = [path for path in latest_paths if path.parent.name in selected]
    if args.max_documents is not None:
        latest_paths = latest_paths[: args.max_documents]
    if not latest_paths:
        raise RuntimeError("no DART latest.json files matched the audit selection")

    tasks = [(str(bronze_root), str(path)) for path in latest_paths]
    if args.workers == 1:
        rows = [_audit_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(_audit_one, tasks))

    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "parser-coverage.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows)
    (output / "parser-coverage-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.fail_on_quality and (
        summary["quality_failed_count"] or summary["error_count"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
