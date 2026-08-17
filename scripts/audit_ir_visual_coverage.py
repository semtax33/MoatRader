#!/usr/bin/env python3
"""Audit semantic chart/figure coverage on the frozen parser-gold-v04 IR set.

The gold file contains a page inventory and manually adjudicated claim-level
semantics.  This script validates those annotations against the immutable
parsed bundles and scoring artifacts, then emits machine-readable and Markdown
reports.  It deliberately does not call a vision model or modify the parser.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


DIMENSIONS = ("axis_legend", "series_identity", "numeric_recovery", "trend_relation")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _walk_nodes(node: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(node, dict):
        return
    if "kind" in node:
        yield node
    for child in node.get("children", []) or []:
        yield from _walk_nodes(child)


def _node_pages(node: dict[str, Any]) -> set[int]:
    return {
        int(ref["page"])
        for ref in node.get("source_refs", []) or []
        if ref.get("page") is not None
    }


def _is_semantic_figure(node: dict[str, Any]) -> bool:
    if node.get("kind") != "figure":
        return False
    attributes = node.get("attributes") or {}
    return bool(
        node.get("caption")
        or attributes.get("chart_type")
        or attributes.get("legend")
        or attributes.get("series")
        or attributes.get("axis")
    )


def _pages_for_atomic_units(
    units: list[dict[str, Any]], source_document_id: str
) -> dict[int, set[str]]:
    result: dict[int, set[str]] = defaultdict(set)
    for unit in units:
        if unit.get("document_id") != source_document_id:
            continue
        for ref in unit.get("source_refs", []) or []:
            page = ref.get("page")
            if page is not None:
                result[int(page)].add(str(unit.get("chunk_id")))
    return result


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": _rate(numerator, denominator),
    }


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def run_audit(repo_root: Path, gold_path: Path, output_dir: Path) -> dict[str, Any]:
    experiment_root = (
        repo_root
        / "data-lake"
        / "experiments"
        / "source-ablation-20250831-longitudinal-v3"
    )
    parser_gold_path = experiment_root / "evaluation" / "parser-gold-v04" / "parser-gold-audit.json"
    gold = _read_json(gold_path)
    parser_gold = _read_json(parser_gold_path)

    documents = {row["ticker"]: row for row in gold["documents"]}
    claims_by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in gold["claims"]:
        ticker = claim["ticker"]
        if ticker not in documents:
            raise ValueError(f"claim {claim['claim_id']} references unknown ticker {ticker}")
        if claim["page"] not in documents[ticker]["meaningful_visual_pages"]:
            raise ValueError(f"claim {claim['claim_id']} page is absent from visual inventory")
        missing_observed = {
            "atomic_claim",
            "accepted_evidence",
            *DIMENSIONS,
        } - set(claim["observed"])
        if missing_observed:
            raise ValueError(f"claim {claim['claim_id']} lacks observed keys: {sorted(missing_observed)}")
        claims_by_ticker[ticker].append(claim)

    document_rows: list[dict[str, Any]] = []
    all_visual_pages = 0
    raw_figure_hits = 0
    semantic_figure_hits = 0
    for ticker, doc in documents.items():
        source_document_id = doc["source_document_id"]
        bundle_path = experiment_root / "parsed" / ticker / source_document_id / "bundle.json"
        bundle = _read_json(bundle_path)
        nodes = list(_walk_nodes(bundle["ast"]))
        raw_figure_pages = {
            page
            for node in nodes
            if node.get("kind") == "figure"
            for page in _node_pages(node)
        }
        semantic_figure_pages = {
            page
            for node in nodes
            if _is_semantic_figure(node)
            for page in _node_pages(node)
        }
        visual_pages = set(doc["meaningful_visual_pages"])

        company_dir = (
            experiment_root
            / "live-runs-v3"
            / doc["run_id"]
            / "companies"
            / ticker
        )
        units = _read_jsonl(company_dir / "atomic-evidence-units.jsonl")
        evidence = _read_jsonl(company_dir / "evidence.jsonl")
        unit_pages = _pages_for_atomic_units(units, source_document_id)
        accepted_by_chunk = Counter(str(row.get("source_chunk_id")) for row in evidence)

        raw_overlap = visual_pages & raw_figure_pages
        semantic_overlap = visual_pages & semantic_figure_pages
        all_visual_pages += len(visual_pages)
        raw_figure_hits += len(raw_overlap)
        semantic_figure_hits += len(semantic_overlap)

        claim_rows: list[dict[str, Any]] = []
        for claim in claims_by_ticker[ticker]:
            page_chunks = unit_pages.get(int(claim["page"]), set())
            claim_rows.append(
                {
                    **claim,
                    "page_atomic_unit_count": len(page_chunks),
                    "page_accepted_evidence_count": sum(
                        accepted_by_chunk[chunk_id] for chunk_id in page_chunks
                    ),
                }
            )

        document_rows.append(
            {
                "ticker": ticker,
                "source_document_id": source_document_id,
                "bundle_path": str(bundle_path.resolve()),
                "run_id": doc["run_id"],
                "meaningful_visual_page_count": len(visual_pages),
                "meaningful_visual_pages": sorted(visual_pages),
                "raw_figure_node_count": sum(1 for node in nodes if node.get("kind") == "figure"),
                "raw_figure_node_pages": sorted(raw_figure_pages),
                "raw_figure_page_hits": sorted(raw_overlap),
                "semantic_figure_page_hits": sorted(semantic_overlap),
                "claim_count": len(claim_rows),
                "claims": claim_rows,
            }
        )

    claims = gold["claims"]
    dimension_metrics: dict[str, dict[str, Any]] = {}
    for dimension in DIMENSIONS:
        required = [claim for claim in claims if dimension in claim["requirements"]]
        passed = [claim for claim in required if claim["observed"][dimension]]
        dimension_metrics[dimension] = _metric(len(passed), len(required))

    parser_complete = [
        claim
        for claim in claims
        if all(claim["observed"][dimension] for dimension in claim["requirements"])
    ]
    atomic_claims = [claim for claim in claims if claim["observed"]["atomic_claim"]]
    accepted_claims = [claim for claim in claims if claim["observed"]["accepted_evidence"]]

    metrics = {
        "document_count": len(documents),
        "meaningful_visual_page_count": all_visual_pages,
        "raw_figure_node_page_recall": _metric(raw_figure_hits, all_visual_pages),
        "semantic_figure_page_recall": _metric(semantic_figure_hits, all_visual_pages),
        "claim_count": len(claims),
        "claim_dimension_recall": dimension_metrics,
        "all_required_parser_semantics_recall": _metric(len(parser_complete), len(claims)),
        "atomic_claim_recall": _metric(len(atomic_claims), len(claims)),
        "accepted_moat_evidence_recall": _metric(len(accepted_claims), len(claims)),
    }

    result = {
        "schema_version": "moatrader-ir-visual-coverage-audit/1",
        "audit_id": gold["audit_id"],
        "selection_uses_return_data": False,
        "parser_gold_baseline": {
            "sample_size": parser_gold["sample_size"],
            "quality_gate_pass_count": parser_gold["quality_gate_pass_count"],
            "numeric_cell_count": parser_gold["numeric_cell_count"],
            "numeric_provenance_completeness": parser_gold["numeric_provenance_completeness"],
            "numeric_coordinate_match_rate": parser_gold["numeric_coordinate_match_rate"],
        },
        "methodology": gold["methodology"],
        "metrics": metrics,
        "decision": {
            "visual_information_loss_material": semantic_figure_hits < all_visual_pages,
            "add_visual_extraction_experiment": True,
            "ship_vision_to_all_pdfs_now": False,
            "recommended_next_gate": (
                "Run a frozen 30-claim text/table-only versus text/table-plus-vision ablation; "
                "require materially higher series/trend and accepted-evidence recall without "
                "lower repeatability."
            ),
            "classifier_instability_still_independent": True,
        },
        "documents": document_rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ir-visual-coverage-audit.json"
    md_path = output_dir / "ir-visual-coverage-audit.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(result), encoding="utf-8")
    return result


def _render_markdown(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    baseline = result["parser_gold_baseline"]
    lines = [
        "# IR Visual Coverage Audit v1",
        "",
        "## Verdict",
        "",
        "Visual information loss is material. The existing parser-gold result remains valid for tables, numbers, coordinates, and provenance, but it does not validate chart/figure semantics.",
        "",
        "A visual extraction lane is justified as a bounded experiment on this frozen gold set. It should not yet be enabled for every PDF. Semantic classifier repeatability remains a separate unresolved problem.",
        "",
        "## Frozen baseline",
        "",
        f"- Parser quality gates: {baseline['quality_gate_pass_count']}/{baseline['sample_size']}",
        f"- Numeric cells: {baseline['numeric_cell_count']}",
        f"- Numeric provenance completeness: {_percent(baseline['numeric_provenance_completeness'])}",
        f"- Numeric coordinate match: {_percent(baseline['numeric_coordinate_match_rate'])}",
        "",
        "## Visual coverage metrics",
        "",
        "| Metric | Result |",
        "| --- | ---: |",
    ]

    metric_rows = [
        ("Meaning-bearing visual pages", None, metrics["meaningful_visual_page_count"]),
        ("Raw `figure` node page overlap", metrics["raw_figure_node_page_recall"], None),
        ("Semantic figure recall", metrics["semantic_figure_page_recall"], None),
        ("Axis/legend recall", metrics["claim_dimension_recall"]["axis_legend"], None),
        ("Series identity recall", metrics["claim_dimension_recall"]["series_identity"], None),
        ("Numeric anchor recovery", metrics["claim_dimension_recall"]["numeric_recovery"], None),
        ("Trend/relation recall", metrics["claim_dimension_recall"]["trend_relation"], None),
        ("All required parser semantics", metrics["all_required_parser_semantics_recall"], None),
        ("Atomic graphical-claim recall", metrics["atomic_claim_recall"], None),
        ("Accepted MOAT-evidence recall", metrics["accepted_moat_evidence_recall"], None),
    ]
    for label, metric, scalar in metric_rows:
        if metric is None:
            value = str(scalar)
        else:
            value = f"{metric['numerator']}/{metric['denominator']} ({_percent(metric['rate'])})"
        lines.append(f"| {label} | {value} |")

    lines.extend(
        [
            "",
            "`figure` node overlap is reported only as a diagnostic. Those nodes are OCR wrappers with no caption, chart type, legend, axis, or series metadata, so they do not satisfy semantic figure detection.",
            "",
            "## Per-document inventory",
            "",
            "| Ticker | Visual pages | Raw figure-page hits | Semantic hits | Gold claims |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["documents"]:
        lines.append(
            "| {ticker} | {pages} | {raw} | {semantic} | {claims} |".format(
                ticker=row["ticker"],
                pages=row["meaningful_visual_page_count"],
                raw=len(row["raw_figure_page_hits"]),
                semantic=len(row["semantic_figure_page_hits"]),
                claims=row["claim_count"],
            )
        )

    lines.extend(
        [
            "",
            "## Claim audit",
            "",
            "Legend: `Y` = explicitly preserved, `N` = absent/flattened, `-` = not required for the claim.",
            "",
            "| Claim | Page | Axis/legend | Series | Numeric | Trend/relation | Atomic | Accepted |",
            "| --- | ---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for row in result["documents"]:
        for claim in row["claims"]:
            def mark(key: str) -> str:
                if key in DIMENSIONS and key not in claim["requirements"]:
                    return "-"
                return "Y" if claim["observed"][key] else "N"

            lines.append(
                f"| {claim['claim_id']} | {claim['page']} | {mark('axis_legend')} | "
                f"{mark('series_identity')} | {mark('numeric_recovery')} | "
                f"{mark('trend_relation')} | {mark('atomic_claim')} | "
                f"{mark('accepted_evidence')} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "1. Text-layer extraction often recovers labels and numeric anchors, so the problem is not simple OCR failure.",
            "2. Series identity, spatial pairing, and graphical trend relations are frequently flattened. Raw coordinates make later reconstruction possible but do not place those relations in the current LLM input.",
            "3. Downstream selection is an additional bottleneck: only one of the 30 strict graphical claims appears as an atomic claim, and none survives as accepted MOAT evidence.",
            "4. Therefore the IR hypothesis is still alive, but a vision lane alone cannot fix semantic-classifier repeatability.",
            "",
            "## Next experiment",
            "",
            "Run the same frozen 30 claims through two treatments: current text/table parser versus current parser plus page-level visual extraction. Score the same dimensions and accepted-evidence recall, then repeat the classification run to ensure visual enrichment does not reduce reproducibility.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("docs/ir-visual-coverage-gold-v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data-lake/experiments/source-ablation-20250831-longitudinal-v3/"
            "evaluation/ir-visual-coverage-v1"
        ),
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    gold_path = args.gold if args.gold.is_absolute() else repo_root / args.gold
    output_dir = args.output_dir if args.output_dir.is_absolute() else repo_root / args.output_dir
    result = run_audit(repo_root, gold_path, output_dir)
    metrics = result["metrics"]
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"wrote: {output_dir / 'ir-visual-coverage-audit.json'}")
    print(f"wrote: {output_dir / 'ir-visual-coverage-audit.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
