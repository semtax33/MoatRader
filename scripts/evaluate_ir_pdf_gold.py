from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import fitz

from moatrader.adapters import PaddlePdfOcrAdapter


def _normalized(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", value).casefold()


def _contains(blocks: list[str], expected: str) -> bool:
    needle = _normalized(expected)
    joined = _normalized(" ".join(blocks))
    return bool(needle) and needle in joined


def _ratio(hits: int, total: int) -> float:
    return hits / total if total else 1.0


def evaluate(
    *,
    fixture_path: Path,
    bronze_root: Path,
    output: Path,
    device: str,
    cpu_threads: int,
    dpi: int,
) -> dict[str, Any]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8-sig"))
    ocr = PaddlePdfOcrAdapter(device=device, cpu_threads=cpu_threads)
    counters = {
        "text_hits": 0,
        "text_total": 0,
        "number_hits": 0,
        "number_total": 0,
        "association_hits": 0,
        "association_total": 0,
        "row_hits": 0,
        "row_total": 0,
        "ordered_hits": 0,
        "ordered_total": 0,
        "provenance_hits": 0,
        "provenance_total": 0,
        "severe_table_corruptions": 0,
    }
    rows: list[dict[str, Any]] = []
    for item in fixture["documents"]:
        candidates = list((bronze_root / item["document_id"]).rglob("document.pdf"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"{item['document_id']}: expected one document.pdf, found {len(candidates)}"
            )
        with fitz.open(candidates[0]) as document:
            page = document[int(item["page"]) - 1]
            result = ocr.extract_page(page, dpi=dpi)
            blocks = [block.text for block in result.blocks if block.confidence >= 0.55]
            text_hits = sum(_contains(blocks, value) for value in item["text"])
            number_hits = sum(_contains(blocks, value) for value in item["numbers"])
            association_hits = sum(
                _contains(blocks, number) and _contains(blocks, label)
                for number, label in item["associations"]
            )
            row_hits = sum(
                all(_contains(blocks, value) for value in expected_row)
                for expected_row in item["rows"]
            )
            order_pairs = list(zip(result.blocks, result.blocks[1:]))
            ordered_hits = sum(
                (left.bbox[1], left.bbox[0]) <= (right.bbox[1], right.bbox[0])
                for left, right in order_pairs
            )
            provenance_hits = sum(
                0 <= block.bbox[0] <= block.bbox[2] <= page.rect.width
                and 0 <= block.bbox[1] <= block.bbox[3] <= page.rect.height
                for block in result.blocks
            )
            severe = int(bool(item["rows"]) and row_hits < len(item["rows"]))
            counters["text_hits"] += text_hits
            counters["text_total"] += len(item["text"])
            counters["number_hits"] += number_hits
            counters["number_total"] += len(item["numbers"])
            counters["association_hits"] += association_hits
            counters["association_total"] += len(item["associations"])
            counters["row_hits"] += row_hits
            counters["row_total"] += len(item["rows"])
            counters["ordered_hits"] += ordered_hits
            counters["ordered_total"] += len(order_pairs)
            counters["provenance_hits"] += provenance_hits
            counters["provenance_total"] += len(result.blocks)
            counters["severe_table_corruptions"] += severe
            rows.append(
                {
                    "ticker": item["ticker"],
                    "document_id": item["document_id"],
                    "page": item["page"],
                    "ocr_block_count": len(result.blocks),
                    "ocr_mean_confidence": result.mean_confidence,
                    "text_recall": _ratio(text_hits, len(item["text"])),
                    "numeric_recall": _ratio(number_hits, len(item["numbers"])),
                    "association_accuracy": _ratio(
                        association_hits, len(item["associations"])
                    ),
                    "row_reconstruction_accuracy": _ratio(row_hits, len(item["rows"])),
                    "severe_table_corruption": bool(severe),
                }
            )
    metrics = {
        "text_snippet_recall": _ratio(counters["text_hits"], counters["text_total"]),
        "numeric_token_recall": _ratio(counters["number_hits"], counters["number_total"]),
        "numeric_label_association_accuracy": _ratio(
            counters["association_hits"], counters["association_total"]
        ),
        "table_row_reconstruction_accuracy": _ratio(
            counters["row_hits"], counters["row_total"]
        ),
        "reading_order_accuracy": _ratio(
            counters["ordered_hits"], counters["ordered_total"]
        ),
        "page_bbox_provenance_accuracy": _ratio(
            counters["provenance_hits"], counters["provenance_total"]
        ),
        "severe_table_corruptions": counters["severe_table_corruptions"],
    }
    thresholds = fixture["thresholds"]
    failures = [
        key
        for key, threshold in thresholds.items()
        if (
            metrics["severe_table_corruptions"] > threshold
            if key == "maximum_severe_table_corruptions"
            else metrics[key] < threshold
        )
    ]
    report = {
        "schema_version": "ir-pdf-gold-evaluation/1",
        "fixture": str(fixture_path.resolve()),
        "document_count": len(rows),
        "metrics": metrics,
        "thresholds": thresholds,
        "passed": not failures,
        "failures": failures,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate representative IR PDF OCR gold pages")
    parser.add_argument("--fixture", default="tests/fixtures/ir-pdf-gold-v1.json")
    parser.add_argument("--bronze-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=200)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    report = evaluate(
        fixture_path=Path(args.fixture),
        bronze_root=Path(args.bronze_root),
        output=Path(args.output),
        device=args.device,
        cpu_threads=args.cpu_threads,
        dpi=args.dpi,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 2)
