#!/usr/bin/env python3
"""Re-adjudicate the nine IR score-bearing claims and retest the classifier.

The audit is deliberately return-data-free.  Three independent image votes
check the manual A/B/C re-adjudication, then six independent production
classifier votes test the frozen, source-grounded canonical claim text.  Every
call is checkpointed, so failures can be resumed without restarting.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from moatrader.canonical.ids import stable_id
from moatrader.canonical.models import SourceRef, SourceType
from moatrader.evidence.models import AtomicEvidenceExtraction
from moatrader.evidence.processing import (
    atomic_routing_signature,
    build_atomic_classification_consensus,
    normalize_atomic_extraction,
)
from moatrader.llm.contracts import build_atomic_evidence_request
from moatrader.semantic.chunker import HeuristicTokenCounter, SemanticChunk

try:
    from scripts.audit_ir_visual_ablation import (
        _call_structured,
        _checkpointed_call,
        _read_json,
        _run_tasks,
        _sha256_file,
        _sha256_text,
        _write_json,
    )
except ModuleNotFoundError:  # Running this file directly adds scripts/ to sys.path.
    from audit_ir_visual_ablation import (  # type: ignore[no-redef]
        _call_structured,
        _checkpointed_call,
        _read_json,
        _run_tasks,
        _sha256_file,
        _sha256_text,
        _write_json,
    )


ADJUDICATION_PROMPT_VERSION = "ir-score-bearing-adjudicator/1"
REPORT_SCHEMA_VERSION = "moatrader-ir-score-bearing-adjudication/1"


class ScoreBearingAdjudication(BaseModel):
    category: Literal["A", "B", "C"]
    role: Literal["MECHANISM", "OUTCOME", "COUNTER", "NONE"]
    subtype: Literal[
        "MARKET_SHARE",
        "CUSTOMER_RETENTION",
        "MARGIN_STABILITY",
        "COST_ADVANTAGE",
        "OTHER",
    ]
    score_bearing_component: str | None = Field(default=None, max_length=800)
    observable_anchors: list[str] = Field(default_factory=list, max_length=8)
    rationale: str = Field(min_length=1, max_length=1_000)


def _adjudicator_system() -> str:
    return """Independently adjudicate one claim from exactly one issuer IR page. Use only the supplied page image and claim text. Do not use outside knowledge, prior labels, classifier output, valuation, or return data.
Choose one category:
- A: genuinely score-bearing because an issuer-linked observable anchor is explicit.
- B: MOAT-relevant context, but no observable score-bearing anchor.
- C: insufficient information for a MOAT classification. Ordinary performance movement belongs here unless it directly erodes an allowed MOAT anchor.
Observable anchors are fail-closed:
- MARKET_SHARE: issuer-specific numeric share, explicit issuer/product leader or rank, or peer-relative share. Sales, distribution, growth, capacity, or a large contract alone do not count.
- CUSTOMER_RETENTION: renewal, repeat purchase/order by the same customer, retention/churn, contract renewal, or installed-base reuse/service dependence. Cumulative orders, customer references, and trust claims alone do not count.
- MARGIN_STABILITY: comparable margin or profitability behavior across at least three periods, or explicit multi-period/consecutive profitability. One-period profit or a single year-over-year change does not count.
- COST_ADVANTAGE: direct issuer-process comparison linking lower cost, fewer steps, less time, or higher yield to a named alternative. Technology alone does not count.
- COUNTER: explicit deterioration or instability of an allowed MOAT mechanism/outcome. Ordinary revenue or profit decline alone does not count.
For A, return the exact role and subtype plus the smallest score-bearing component. For B or C, return role=NONE, subtype=OTHER, and score_bearing_component=null. Prefer B/C over inferred precision."""


def _validate_adjudication(value: ScoreBearingAdjudication) -> ScoreBearingAdjudication:
    if value.category == "A":
        if value.role == "NONE" or value.subtype == "OTHER" or not value.score_bearing_component:
            raise ValueError("category A requires a score-bearing role, subtype, and component")
    elif value.role != "NONE" or value.subtype != "OTHER" or value.score_bearing_component is not None:
        raise ValueError("categories B/C require NONE/OTHER and no score-bearing component")
    return value


def _manual_signature(claim: dict[str, Any]) -> tuple[str, str, str]:
    route = claim.get("score_bearing_route")
    if claim["adjudication_class"] != "A" or not route:
        return (str(claim["adjudication_class"]), "NONE", "OTHER")
    return ("A", str(route["role"]), str(route["subtype"]))


def _expected_classifier_route(claim: dict[str, Any]) -> tuple[str, bool, str, str]:
    route = claim.get("score_bearing_route")
    if claim["adjudication_class"] != "A" or not route:
        return ("NONE", False, "OTHER", "NEUTRAL")
    role = str(route["role"])
    direction = "MOAT_NEGATIVE" if role == "COUNTER" else "MOAT_POSITIVE"
    return (role, True, str(route["subtype"]), direction)


def _strict_majority(values: list[tuple[str, ...]]) -> tuple[str, ...] | None:
    if not values:
        return None
    counts = Counter(values)
    winner, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
    return winner if count >= len(values) // 2 + 1 else None


def run_adjudication_votes(
    *,
    repo_root: Path,
    gold: dict[str, Any],
    output: Path,
    model: str,
    effort: str,
    votes: int,
    max_workers: int,
) -> list[dict[str, Any]]:
    system = _adjudicator_system()
    tasks: list[tuple[dict[str, Any], int, Path, Path]] = []
    for claim, vote in itertools.product(gold["claims"], range(1, votes + 1)):
        image_path = repo_root / claim["rendered_page"]
        path = output / "checkpoints" / "adjudication" / claim["claim_id"] / f"vote-{vote:02d}.json"
        tasks.append((claim, vote, image_path, path))

    def execute(task: tuple[dict[str, Any], int, Path, Path]) -> dict[str, Any]:
        claim, vote, image_path, path = task
        user = f"""Issuer ID: {claim['ticker']}
Issuer name: {claim['issuer_name']}
PDF page: {claim['page']}
Candidate claim: {claim['canonical_classifier_text']}

Inspect the attached source page and adjudicate the candidate claim."""
        identity = {
            "stage": "independent-adjudication",
            "prompt_version": ADJUDICATION_PROMPT_VERSION,
            "model": model,
            "effort": effort,
            "claim_id": claim["claim_id"],
            "vote": vote,
            "system_sha256": _sha256_text(system),
            "user_sha256": _sha256_text(user),
            "image_sha256": _sha256_file(image_path),
        }

        def call() -> dict[str, Any]:
            result = _call_structured(
                model=model,
                effort=effort,
                system=system,
                user=user,
                response_model=ScoreBearingAdjudication,
                image_path=image_path,
                max_output_tokens=1_800,
            )
            parsed = _validate_adjudication(ScoreBearingAdjudication.model_validate(result["parsed"]))
            result["parsed"] = parsed.model_dump(mode="json")
            return result

        payload = _checkpointed_call(path, identity, call)
        return {"claim_id": claim["claim_id"], "vote": vote, **payload}

    return _run_tasks(tasks, execute, max_workers=max_workers)


def _classification_chunk(claim: dict[str, Any]) -> SemanticChunk:
    source_text = str(claim["canonical_classifier_text"])
    atomic_key = stable_id("AEK", "ir-score-bearing-adjudication-v1", claim["claim_id"], source_text)
    node_id = stable_id("N", "ir-score-bearing-adjudication-v1", claim["claim_id"])
    return SemanticChunk(
        chunk_id=stable_id("C", atomic_key),
        document_id=f"IR-ADJ-{claim['ticker']}",
        section_path=["IR score-bearing adjudication", f"page {claim['page']}"],
        node_ids=[node_id],
        chunk_type="atomic_evidence",
        markdown=source_text,
        token_count=HeuristicTokenCounter().count(source_text),
        source_refs=[
            SourceRef(
                source_type=SourceType.IR,
                document_id=f"IR-ADJ-{claim['ticker']}",
                page=int(claim["page"]),
            )
        ],
        metadata={"atomic_evidence_key": atomic_key, "adjudication_claim_id": claim["claim_id"]},
    )


def run_classifier_votes(
    *,
    gold: dict[str, Any],
    output: Path,
    model: str,
    effort: str,
    votes: int,
    max_workers: int,
) -> list[dict[str, Any]]:
    tasks: list[tuple[dict[str, Any], int, SemanticChunk, Path]] = []
    for claim, vote in itertools.product(gold["claims"], range(1, votes + 1)):
        chunk = _classification_chunk(claim)
        path = output / "checkpoints" / "classifier" / claim["claim_id"] / f"vote-{vote:02d}.json"
        tasks.append((claim, vote, chunk, path))

    def execute(task: tuple[dict[str, Any], int, SemanticChunk, Path]) -> dict[str, Any]:
        claim, vote, chunk, path = task
        request = build_atomic_evidence_request(
            chunk,
            issuer_id=str(claim["ticker"]),
            issuer_name=str(claim["issuer_name"]),
            classification_vote=vote,
        )
        identity = {
            "stage": "production-classifier",
            "prompt_version": request.metadata["prompt_version"],
            "rubric_version": request.metadata["rubric_version"],
            "model": model,
            "effort": effort,
            "claim_id": claim["claim_id"],
            "vote": vote,
            "input_sha256": request.input_sha256,
        }

        def call() -> dict[str, Any]:
            result = _call_structured(
                model=model,
                effort=effort,
                system=request.system,
                user=request.user,
                response_model=AtomicEvidenceExtraction,
                max_output_tokens=2_000,
            )
            extraction = AtomicEvidenceExtraction.model_validate(result["parsed"])
            result["raw_parsed"] = extraction.model_dump(mode="json", by_alias=True)
            normalized, repair_actions = normalize_atomic_extraction(
                extraction,
                source_text=chunk.markdown,
            )
            result["parsed"] = normalized.model_dump(mode="json", by_alias=True)
            result["repair_actions"] = repair_actions
            return result

        payload = _checkpointed_call(path, identity, call)
        return {"claim_id": claim["claim_id"], "vote": vote, **payload}

    return _run_tasks(tasks, execute, max_workers=max_workers)


def _load_successful(path: Path, count: int, model: type[BaseModel]) -> list[BaseModel]:
    values: list[BaseModel] = []
    for vote in range(1, count + 1):
        checkpoint = path / f"vote-{vote:02d}.json"
        if not checkpoint.exists():
            continue
        payload = _read_json(checkpoint)
        if payload.get("status") == "SUCCESS":
            values.append(model.model_validate(payload["parsed"]))
    return values


def _metric(numerator: int, denominator: int) -> dict[str, Any]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator if denominator else None,
    }


def _usage(output: Path) -> dict[str, int]:
    totals = {"successful_calls": 0, "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    for path in (output / "checkpoints").rglob("*.json"):
        payload = _read_json(path)
        if payload.get("status") != "SUCCESS":
            continue
        totals["successful_calls"] += 1
        usage = payload.get("usage") or {}
        for key in ("input_tokens", "output_tokens", "cached_input_tokens"):
            totals[key] += int(usage.get(key, 0) or 0)
    return totals


def build_report(
    *,
    gold: dict[str, Any],
    output: Path,
    adjudication_votes: int,
    classifier_votes: int,
    adjudicator_model: str,
    classifier_model: str,
    pre_change_report: Path | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    adjudication_category_hits = 0
    adjudication_route_hits = 0
    adjudication_a_count = 0
    classifier_a_hits = 0
    classifier_a_count = 0
    classifier_none_hits = 0
    classifier_none_count = 0
    repeat_hits = 0
    repeat_count = 0
    conflict_hits = 0
    conflict_count = 0

    for claim in gold["claims"]:
        adjudications = _load_successful(
            output / "checkpoints" / "adjudication" / claim["claim_id"],
            adjudication_votes,
            ScoreBearingAdjudication,
        )
        adjudication_signatures = [
            (value.category, value.role, value.subtype) for value in adjudications  # type: ignore[attr-defined]
        ]
        adjudication_consensus = _strict_majority(adjudication_signatures)
        manual_signature = _manual_signature(claim)
        adjudication_category_hits += int(
            adjudication_consensus is not None and adjudication_consensus[0] == manual_signature[0]
        )
        if manual_signature[0] == "A":
            adjudication_a_count += 1
            adjudication_route_hits += int(adjudication_consensus == manual_signature)

        classifier_values = [
            value
            for value in _load_successful(
                output / "checkpoints" / "classifier" / claim["claim_id"],
                classifier_votes,
                AtomicEvidenceExtraction,
            )
            if isinstance(value, AtomicEvidenceExtraction)
        ]
        consensus_route: tuple[str, bool, str, str] | None = None
        group_a_route: tuple[str, bool, str, str] | None = None
        group_b_route: tuple[str, bool, str, str] | None = None
        if len(classifier_values) == classifier_votes:
            selected, _ = build_atomic_classification_consensus(
                classifier_values,
                source_text=str(claim["canonical_classifier_text"]),
            )
            group_a, _ = build_atomic_classification_consensus(
                classifier_values[: classifier_votes // 2],
                source_text=str(claim["canonical_classifier_text"]),
            )
            group_b, _ = build_atomic_classification_consensus(
                classifier_values[classifier_votes // 2 :],
                source_text=str(claim["canonical_classifier_text"]),
            )
            consensus_route = atomic_routing_signature(selected)
            group_a_route = atomic_routing_signature(group_a)
            group_b_route = atomic_routing_signature(group_b)
            repeat_count += 1
            repeat_hits += int(group_a_route == group_b_route)
            if group_a_route[0] != "NONE" or group_b_route[0] != "NONE":
                conflict_count += 1
                conflict_hits += int(group_a_route != group_b_route)

        expected_route = _expected_classifier_route(claim)
        if claim["adjudication_class"] == "A":
            classifier_a_count += 1
            classifier_a_hits += int(consensus_route == expected_route)
        else:
            classifier_none_count += 1
            classifier_none_hits += int(consensus_route == expected_route)
        rows.append(
            {
                "claim_id": claim["claim_id"],
                "manual_signature": list(manual_signature),
                "adjudication_vote_count": len(adjudications),
                "adjudication_consensus": list(adjudication_consensus) if adjudication_consensus else None,
                "classifier_vote_count": len(classifier_values),
                "expected_classifier_route": list(expected_route),
                "classifier_consensus_route": list(consensus_route) if consensus_route else None,
                "group_a_route": list(group_a_route) if group_a_route else None,
                "group_b_route": list(group_b_route) if group_b_route else None,
            }
        )

    metrics = {
        "independent_adjudication_category_agreement": _metric(adjudication_category_hits, len(gold["claims"])),
        "independent_adjudication_A_route_agreement": _metric(adjudication_route_hits, adjudication_a_count),
        "classifier_true_score_bearing_route_recall": _metric(classifier_a_hits, classifier_a_count),
        "classifier_BC_rejection": _metric(classifier_none_hits, classifier_none_count),
        "classifier_independent_three_vote_repeatability": _metric(repeat_hits, repeat_count),
        "classifier_score_bearing_route_conflict": _metric(conflict_hits, conflict_count),
    }
    pre_change = _read_json(pre_change_report) if pre_change_report and pre_change_report.exists() else None
    pre_change_metric = None
    if pre_change:
        pre_change_metric = pre_change["lanes"]["vision"]["metrics"]["score_bearing_gold_route_recall"]
    gates = {
        "independent_adjudication_category_agreement_at_least_0_80": float(metrics["independent_adjudication_category_agreement"]["rate"] or 0) >= 0.80,
        "independent_adjudication_A_route_agreement_at_least_0_80": float(metrics["independent_adjudication_A_route_agreement"]["rate"] or 0) >= 0.80,
        "classifier_true_score_bearing_recall_at_least_0_85": float(metrics["classifier_true_score_bearing_route_recall"]["rate"] or 0) >= 0.85,
        "classifier_BC_rejection_is_1_00": metrics["classifier_BC_rejection"]["rate"] == 1.0,
        "classifier_repeatability_at_least_0_90": float(metrics["classifier_independent_three_vote_repeatability"]["rate"] or 0) >= 0.90,
        "classifier_score_bearing_conflict_below_0_10": float(metrics["classifier_score_bearing_route_conflict"]["rate"] or 0) < 0.10,
    }
    failures = [
        str(path.resolve())
        for path in (output / "checkpoints").rglob("*.json")
        if _read_json(path).get("status") == "ERROR"
    ]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_uses_return_data": False,
        "manual_gold_schema_version": gold["schema_version"],
        "models": {"independent_adjudicator": adjudicator_model, "production_classifier": classifier_model},
        "votes": {"adjudication": adjudication_votes, "classifier": classifier_votes},
        "manual_counts": gold["summary"],
        "pre_change_vision_score_bearing_route_recall": pre_change_metric,
        "metrics": metrics,
        "success_gates": gates,
        "passed": all(gates.values()) and not failures,
        "claims": rows,
        "usage_current_checkpoints": _usage(output),
        "failures": failures,
    }
    _write_json(output / "ir-score-bearing-adjudication.json", report)
    (output / "ir-score-bearing-adjudication.md").write_text(_render_markdown(report), encoding="utf-8")
    return report


def _render_markdown(report: dict[str, Any]) -> str:
    def metric(value: dict[str, Any] | None) -> str:
        if not value or value.get("rate") is None:
            return "n/a"
        return f"{value['numerator']}/{value['denominator']} ({value['rate']:.1%})"

    lines = [
        "# IR Score-Bearing Adjudication",
        "",
        "## Verdict",
        "",
        "PASS" if report["passed"] else "FAIL",
        "",
        "## Metrics",
        "",
        f"- Manual A/B/C counts: A={report['manual_counts']['A_true_score_bearing']}, B={report['manual_counts']['B_relevant_non_score_bearing']}, C={report['manual_counts']['C_insufficient_information']}",
        f"- Pre-change vision classifier score-bearing route recall: {metric(report['pre_change_vision_score_bearing_route_recall'])}",
    ]
    lines.extend(f"- {name}: {metric(value)}" for name, value in report["metrics"].items())
    lines.extend(["", "## Gates", ""])
    lines.extend(f"- {'PASS' if passed else 'FAIL'}: `{name}`" for name, passed in report["success_gates"].items())
    lines.extend(
        [
            "",
            "## Claims",
            "",
            "| Claim | Manual | Independent adjudication | Classifier expected | Classifier consensus | Repeat |",
            "| --- | --- | --- | --- | --- | :---: |",
        ]
    )
    for row in report["claims"]:
        repeat = row["group_a_route"] == row["group_b_route"] if row["group_a_route"] else False
        lines.append(
            f"| {row['claim_id']} | {row['manual_signature']} | {row['adjudication_consensus'] or '-'} | "
            f"{row['expected_classifier_route']} | {row['classifier_consensus_route'] or '-'} | {'Y' if repeat else 'N'} |"
        )
    usage = report["usage_current_checkpoints"]
    lines.extend(
        [
            "",
            "## API usage",
            "",
            f"- Successful calls: {usage['successful_calls']}",
            f"- Input tokens: {usage['input_tokens']}",
            f"- Cached input tokens: {usage['cached_input_tokens']}",
            f"- Output tokens: {usage['output_tokens']}",
            f"- Failed checkpoints: {len(report['failures'])}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--gold", type=Path, default=Path("docs/ir-score-bearing-adjudication-v1.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data-lake/experiments/source-ablation-20250831-longitudinal-v3/"
            "evaluation/ir-score-bearing-adjudication-v1"
        ),
    )
    parser.add_argument(
        "--pre-change-report",
        type=Path,
        default=Path(
            "data-lake/experiments/source-ablation-20250831-longitudinal-v3/"
            "evaluation/ir-visual-ablation-v1-pre-anchor-corrected-gold/ir-visual-ablation.json"
        ),
    )
    parser.add_argument("--adjudicator-model", default="gpt-5.6-luna")
    parser.add_argument("--classifier-model", default="gpt-5.6-luna")
    parser.add_argument("--adjudicator-effort", choices=("none", "low", "medium", "high"), default="medium")
    parser.add_argument("--classifier-effort", choices=("none", "low", "medium", "high"), default="medium")
    parser.add_argument("--adjudication-votes", type=int, choices=(3, 5), default=3)
    parser.add_argument("--classifier-votes", type=int, choices=(6, 8, 10), default=6)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--stage", choices=("all", "adjudicate", "classify", "report"), default="all")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    gold_path = args.gold if args.gold.is_absolute() else repo_root / args.gold
    output = args.output if args.output.is_absolute() else repo_root / args.output
    pre_change = args.pre_change_report if args.pre_change_report.is_absolute() else repo_root / args.pre_change_report
    output.mkdir(parents=True, exist_ok=True)
    gold = _read_json(gold_path)

    if args.stage in {"all", "adjudicate"}:
        rows = run_adjudication_votes(
            repo_root=repo_root,
            gold=gold,
            output=output,
            model=args.adjudicator_model,
            effort=args.adjudicator_effort,
            votes=args.adjudication_votes,
            max_workers=args.max_workers,
        )
        print(f"adjudication checkpoints: {sum(row.get('status') == 'SUCCESS' for row in rows)}/{len(rows)} success")
    if args.stage in {"all", "classify"}:
        rows = run_classifier_votes(
            gold=gold,
            output=output,
            model=args.classifier_model,
            effort=args.classifier_effort,
            votes=args.classifier_votes,
            max_workers=args.max_workers,
        )
        print(f"classifier checkpoints: {sum(row.get('status') == 'SUCCESS' for row in rows)}/{len(rows)} success")
    if args.stage in {"all", "report"}:
        report = build_report(
            gold=gold,
            output=output,
            adjudication_votes=args.adjudication_votes,
            classifier_votes=args.classifier_votes,
            adjudicator_model=args.adjudicator_model,
            classifier_model=args.classifier_model,
            pre_change_report=pre_change,
        )
        print(json.dumps({"metrics": report["metrics"], "success_gates": report["success_gates"], "passed": report["passed"], "failures": len(report["failures"])}, ensure_ascii=False, indent=2))
        print(f"wrote: {output / 'ir-score-bearing-adjudication.json'}")
        print(f"wrote: {output / 'ir-score-bearing-adjudication.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
