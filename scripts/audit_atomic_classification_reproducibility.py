from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from moatrader.evidence.models import (
    AtomicEvidenceExtraction,
    AtomicMoatRole,
    EconomicScope,
    EvidenceDirection,
    EvidenceType,
    OUTCOME_CORROBORATION_TYPES,
    STRUCTURAL_MOAT_TYPES,
)
from moatrader.evidence.processing import (
    atomic_classification_signature,
    atomic_routing_signature,
    build_atomic_classification_consensus,
    normalize_atomic_extraction,
)
from moatrader.llm import OpenAIResponsesTransport, build_atomic_evidence_request
from moatrader.semantic.chunker import SemanticChunk


COUNTER_TYPES = STRUCTURAL_MOAT_TYPES | OUTCOME_CORROBORATION_TYPES | {
    EvidenceType.COMPETITIVE_THREAT,
    EvidenceType.CUSTOMER_CONCENTRATION,
    EvidenceType.SUBSTITUTION_RISK,
    EvidenceType.TECHNOLOGY_RISK,
    EvidenceType.CAPITAL_INTENSITY,
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _company_directories(root: Path) -> dict[str, Path]:
    direct = root / "companies"
    paths = direct.glob("*") if direct.is_dir() else root.glob("*dart-plus-ir-*/companies/*")
    return {path.name: path for path in paths if path.is_dir()}


def _judgments(company: Path) -> dict[str, dict[str, Any]]:
    return {
        path.stem: _read_json(path)
        for path in (company / "atomic-judgment-by-key").glob("*.json")
    }


def _units(company: Path) -> dict[str, SemanticChunk]:
    result: dict[str, SemanticChunk] = {}
    path = company / "atomic-evidence-units.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        chunk = SemanticChunk.model_validate_json(line)
        result[str(chunk.metadata["atomic_evidence_key"])] = chunk
    return result


def _issuer_name(company: Path) -> str | None:
    for filename in ("result.json", "dossier.json"):
        path = company / filename
        if not path.is_file():
            continue
        value = _read_json(path).get("issuer_name")
        if value:
            return str(value)
    return None


def _legacy_role(value: dict[str, Any]) -> AtomicMoatRole:
    if not value.get("is_investment_relevant"):
        return AtomicMoatRole.NONE
    if value.get("economic_scope") not in {
        EconomicScope.COMPANY.value,
        EconomicScope.SEGMENT.value,
    }:
        return AtomicMoatRole.NONE
    evidence_type = EvidenceType(value.get("evidence_type", EvidenceType.OTHER.value))
    direction = EvidenceDirection(value.get("direction", EvidenceDirection.NEUTRAL.value))
    if evidence_type in STRUCTURAL_MOAT_TYPES and direction == EvidenceDirection.MOAT_POSITIVE:
        return AtomicMoatRole.MECHANISM
    if evidence_type in OUTCOME_CORROBORATION_TYPES and direction == EvidenceDirection.MOAT_POSITIVE:
        return AtomicMoatRole.OUTCOME
    if evidence_type in COUNTER_TYPES and direction == EvidenceDirection.MOAT_NEGATIVE:
        return AtomicMoatRole.COUNTER
    return AtomicMoatRole.NONE


def _legacy_signature(value: dict[str, Any]) -> tuple[str, bool, str, str, str]:
    return (
        _legacy_role(value).value,
        bool(value.get("is_investment_relevant")),
        str(value.get("evidence_type", EvidenceType.OTHER.value)),
        str(value.get("direction", EvidenceDirection.NEUTRAL.value)),
        str(value.get("economic_scope", EconomicScope.COMPANY.value)),
    )


def _legacy_routing_signature(value: dict[str, Any]) -> tuple[str, bool, str, str]:
    return _legacy_signature(value)[:4]


def select_frozen_cases(
    baseline_root: Path,
    repeat_root: Path,
    *,
    sample_size: int,
) -> list[dict[str, Any]]:
    """Select deterministic, score-relevant baseline disagreements.

    The sample is company-diverse and deliberately enriched for IR and role
    transitions.  It is frozen before any new model call.
    """

    baseline = _company_directories(baseline_root)
    repeat = _company_directories(repeat_root)
    candidates: list[dict[str, Any]] = []
    for ticker in sorted(baseline.keys() & repeat.keys()):
        left = _judgments(baseline[ticker])
        right = _judgments(repeat[ticker])
        units = _units(baseline[ticker])
        for atomic_key in sorted(left.keys() & right.keys() & units.keys()):
            left_signature = _legacy_signature(left[atomic_key])
            right_signature = _legacy_signature(right[atomic_key])
            left_role, right_role = left_signature[0], right_signature[0]
            if left_signature == right_signature:
                continue
            if left_role == right_role == AtomicMoatRole.NONE.value:
                continue
            chunk = units[atomic_key]
            source_type = str(chunk.metadata.get("source_type") or "OTHER")
            candidates.append(
                {
                    "ticker": ticker,
                    "issuer_name": _issuer_name(baseline[ticker]),
                    "atomic_evidence_key": atomic_key,
                    "source_type": source_type,
                    "document_id": chunk.document_id,
                    "baseline_signature": list(left_signature),
                    "repeat_signature": list(right_signature),
                    "role_transition": left_role != right_role,
                    "chunk": chunk.model_dump(mode="json"),
                }
            )
    candidates.sort(
        key=lambda row: (
            row["ticker"] != "206650",
            not row["role_transition"],
            row["source_type"] != "IR",
            row["ticker"],
            row["atomic_evidence_key"],
        )
    )
    selected: list[dict[str, Any]] = []
    per_company: Counter[str] = Counter()
    # First pass maximizes company coverage; second pass fills remaining slots.
    for maximum_per_company in (1, 2, sample_size):
        for row in candidates:
            if len(selected) >= sample_size:
                break
            identity = (row["ticker"], row["atomic_evidence_key"])
            if any((item["ticker"], item["atomic_evidence_key"]) == identity for item in selected):
                continue
            if per_company[row["ticker"]] >= maximum_per_company:
                continue
            selected.append(row)
            per_company[row["ticker"]] += 1
        if len(selected) >= sample_size:
            break
    if len(selected) < sample_size:
        raise ValueError(
            f"only {len(selected)} score-relevant baseline disagreements are available; "
            f"requested {sample_size}"
        )
    return selected


def _pairwise_rate(signatures: list[tuple[Any, ...]]) -> float:
    pairs = list(combinations(signatures, 2))
    return statistics.mean(left == right for left, right in pairs) if pairs else 1.0


def _three_vote_diagnostics(
    votes: list[AtomicEvidenceExtraction],
    *,
    source_text: str,
) -> dict[str, Any]:
    full, _full_audit = build_atomic_classification_consensus(
        votes,
        source_text=source_text,
    )
    full_route = atomic_routing_signature(full)
    routes: list[tuple[str, bool, str, str]] = []
    for indexes in combinations(range(len(votes)), 3):
        selected, _audit = build_atomic_classification_consensus(
            [votes[index] for index in indexes],
            source_text=source_text,
        )
        routes.append(atomic_routing_signature(selected))
    route_counts = Counter(routes)
    route_match_count = sum(route == full_route for route in routes)
    moat_conflict_count = sum(
        route != full_route
        and (
            route[0] != AtomicMoatRole.NONE.value
            or full_route[0] != AtomicMoatRole.NONE.value
        )
        for route in routes
    )
    return {
        "combination_count": len(routes),
        "full_vote_route": list(full_route),
        "route_match_to_full_rate": route_match_count / len(routes),
        "any_route_mismatch_rate": 1 - (route_match_count / len(routes)),
        "moat_route_conflict_rate": moat_conflict_count / len(routes),
        "route_distribution": [
            {"route": list(route), "count": count}
            for route, count in sorted(route_counts.items())
        ],
    }


def evaluate_case(
    case: dict[str, Any],
    votes: list[AtomicEvidenceExtraction],
) -> dict[str, Any]:
    if len(votes) < 2 or len(votes) % 2:
        raise ValueError("evaluation requires an even vote count split into two independent groups")
    split = len(votes) // 2
    chunk = SemanticChunk.model_validate(case["chunk"])
    signatures = [atomic_classification_signature(vote) for vote in votes]
    routing_signatures = [atomic_routing_signature(vote) for vote in votes]
    first, first_audit = build_atomic_classification_consensus(
        votes[:split], source_text=chunk.markdown
    )
    second, second_audit = build_atomic_classification_consensus(
        votes[split:], source_text=chunk.markdown
    )
    first_signature = atomic_classification_signature(first)
    second_signature = atomic_classification_signature(second)
    first_route = atomic_routing_signature(first)
    second_route = atomic_routing_signature(second)
    counts = Counter(signatures)
    route_counts = Counter(routing_signatures)
    modal_count = max(counts.values())
    route_modal_count = max(route_counts.values())
    three_vote = _three_vote_diagnostics(votes, source_text=chunk.markdown)
    return {
        **{key: value for key, value in case.items() if key != "chunk"},
        "source_text": chunk.markdown,
        "vote_count": len(votes),
        "raw_signatures": [list(signature) for signature in signatures],
        "raw_routing_signatures": [list(signature) for signature in routing_signatures],
        "raw_modal_agreement_rate": modal_count / len(votes),
        "raw_pairwise_agreement_rate": _pairwise_rate(signatures),
        "raw_route_modal_agreement_rate": route_modal_count / len(votes),
        "raw_route_pairwise_agreement_rate": _pairwise_rate(routing_signatures),
        "group_a_consensus": first_audit,
        "group_b_consensus": second_audit,
        "group_a_signature": list(first_signature),
        "group_b_signature": list(second_signature),
        "group_a_route": list(first_route),
        "group_b_route": list(second_route),
        "consensus_role_match": first_signature[0] == second_signature[0],
        "consensus_route_signature_match": first_route == second_route,
        "consensus_signature_match": first_signature == second_signature,
        "consensus_scope_match": first_signature[4] == second_signature[4],
        "score_route_conflict": (
            first_route != second_route
            and (
                first_route[0] != AtomicMoatRole.NONE.value
                or second_route[0] != AtomicMoatRole.NONE.value
            )
        ),
        "production_three_vote": three_vote,
    }


def summarize(cases: list[dict[str, Any]], *, requested_votes: int) -> dict[str, Any]:
    baseline_exact = statistics.mean(
        row["baseline_signature"] == row["repeat_signature"] for row in cases
    )
    baseline_route_exact = statistics.mean(
        row["baseline_signature"][:4] == row["repeat_signature"][:4] for row in cases
    )
    role_match = statistics.mean(row["consensus_role_match"] for row in cases)
    route_match = statistics.mean(row["consensus_route_signature_match"] for row in cases)
    signature_match = statistics.mean(row["consensus_signature_match"] for row in cases)
    scope_match = statistics.mean(row["consensus_scope_match"] for row in cases)
    conflict_rate = statistics.mean(row["score_route_conflict"] for row in cases)
    modal_agreement = statistics.mean(row["raw_modal_agreement_rate"] for row in cases)
    pairwise_agreement = statistics.mean(row["raw_pairwise_agreement_rate"] for row in cases)
    route_modal_agreement = statistics.mean(
        row["raw_route_modal_agreement_rate"] for row in cases
    )
    route_pairwise_agreement = statistics.mean(
        row["raw_route_pairwise_agreement_rate"] for row in cases
    )
    three_vote_route_match = statistics.mean(
        row["production_three_vote"]["route_match_to_full_rate"] for row in cases
    )
    three_vote_any_mismatch = statistics.mean(
        row["production_three_vote"]["any_route_mismatch_rate"] for row in cases
    )
    three_vote_moat_conflict = statistics.mean(
        row["production_three_vote"]["moat_route_conflict_rate"] for row in cases
    )
    complete = all(row["vote_count"] == requested_votes for row in cases)
    criteria = {
        "all_frozen_cases_have_requested_votes": complete,
        "mean_raw_route_modal_agreement_at_least_0_60": route_modal_agreement >= 0.60,
        "independent_consensus_role_match_at_least_0_90": role_match >= 0.90,
        "independent_consensus_route_match_at_least_0_90": route_match >= 0.90,
        "independent_consensus_scope_match_at_least_0_80": scope_match >= 0.80,
        "score_route_conflict_at_most_0_10": conflict_rate <= 0.10,
        "consensus_route_improves_baseline": route_match > baseline_route_exact,
    }
    production_criteria = {
        "three_vote_route_match_to_full_at_least_0_95": three_vote_route_match >= 0.95,
        "three_vote_moat_route_conflict_at_most_0_01": three_vote_moat_conflict <= 0.01,
    }
    return {
        "schema_version": "atomic-classification-reproducibility/1",
        "sample_count": len(cases),
        "votes_per_case": requested_votes,
        "baseline_exact_signature_rate": baseline_exact,
        "baseline_exact_route_rate": baseline_route_exact,
        "mean_raw_modal_agreement_rate": modal_agreement,
        "mean_raw_pairwise_agreement_rate": pairwise_agreement,
        "mean_raw_route_modal_agreement_rate": route_modal_agreement,
        "mean_raw_route_pairwise_agreement_rate": route_pairwise_agreement,
        "independent_consensus_role_match_rate": role_match,
        "independent_consensus_route_match_rate": route_match,
        "independent_consensus_signature_match_rate": signature_match,
        "independent_consensus_scope_match_rate": scope_match,
        "score_route_conflict_rate": conflict_rate,
        "production_three_vote_route_match_to_full_rate": three_vote_route_match,
        "production_three_vote_any_route_mismatch_rate": three_vote_any_mismatch,
        "production_three_vote_moat_route_conflict_rate": three_vote_moat_conflict,
        "pre_registered_criteria": criteria,
        "production_three_vote_diagnostic_criteria": production_criteria,
        "classifier_stability_supported": all(criteria.values()),
        "production_three_vote_supported": all(production_criteria.values()),
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Atomic Evidence Classification Reproducibility",
        "",
        f"- frozen cases: {summary['sample_count']}",
        f"- independent votes per case: {summary['votes_per_case']}",
        f"- baseline exact signature rate: {summary['baseline_exact_signature_rate']:.3f}",
        f"- baseline exact economic-route rate: {summary['baseline_exact_route_rate']:.3f}",
        f"- raw modal agreement: {summary['mean_raw_modal_agreement_rate']:.3f}",
        f"- raw pairwise agreement: {summary['mean_raw_pairwise_agreement_rate']:.3f}",
        f"- raw economic-route modal agreement: {summary['mean_raw_route_modal_agreement_rate']:.3f}",
        f"- raw economic-route pairwise agreement: {summary['mean_raw_route_pairwise_agreement_rate']:.3f}",
        f"- independent consensus role match: {summary['independent_consensus_role_match_rate']:.3f}",
        f"- independent consensus economic-route match: {summary['independent_consensus_route_match_rate']:.3f}",
        f"- independent consensus signature match: {summary['independent_consensus_signature_match_rate']:.3f}",
        f"- independent consensus scope match: {summary['independent_consensus_scope_match_rate']:.3f}",
        f"- score-route conflict rate: {summary['score_route_conflict_rate']:.3f}",
        f"- production 3-vote route match to full consensus: {summary['production_three_vote_route_match_to_full_rate']:.3f}",
        f"- production 3-vote any-route mismatch: {summary['production_three_vote_any_route_mismatch_rate']:.3f}",
        f"- production 3-vote MOAT-route conflict: {summary['production_three_vote_moat_route_conflict_rate']:.3f}",
        "",
        "## Pre-registered criteria",
        "",
        *[
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in summary["pre_registered_criteria"].items()
        ],
        "",
        "## Production 3-vote diagnostics",
        "",
        *[
            f"- {name}: {'PASS' if passed else 'FAIL'}"
            for name, passed in summary["production_three_vote_diagnostic_criteria"].items()
        ],
        "",
        "## Verdict",
        "",
        (
            "Classifier stability is supported on this frozen small sample."
            if summary["classifier_stability_supported"]
            else "Classifier stability is not supported on this frozen small sample."
        ),
        "",
        "## Cases",
        "",
        "| Ticker | Source | Atomic key | Raw route modal | Consensus role | Economic route | Full signature | Route conflict |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["cases"]:
        lines.append(
            f"| {row['ticker']} | {row['source_type']} | {row['atomic_evidence_key']} | "
            f"{row['raw_route_modal_agreement_rate']:.3f} | "
            f"{row['consensus_role_match']} | {row['consensus_route_signature_match']} | "
            f"{row['consensus_signature_match']} | "
            f"{row['score_route_conflict']} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    vote_source = Path(args.vote_source).resolve() if args.vote_source else None
    sample_path = output / "frozen-sample.json"
    if sample_path.is_file():
        frozen_cases = _read_json(sample_path)["cases"]
    else:
        frozen_cases = select_frozen_cases(
            Path(args.baseline_root).resolve(),
            Path(args.repeat_root).resolve(),
            sample_size=args.sample_size,
        )
        _write_json(
            sample_path,
            {
                "schema_version": "atomic-classification-frozen-sample/1",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "baseline_root": str(Path(args.baseline_root).resolve()),
                "repeat_root": str(Path(args.repeat_root).resolve()),
                "selection": "company-diverse-score-relevant-baseline-disagreements/1",
                "cases": frozen_cases,
            },
        )
    if args.freeze_only:
        return {"frozen_sample": str(sample_path), "sample_count": len(frozen_cases)}

    transport = OpenAIResponsesTransport(
        summary_model="gpt-5-nano",
        moat_model=args.model,
        atomic_reasoning_effort=args.reasoning_effort,
        max_output_tokens=2_000,
        max_retries=args.api_retries,
        timeout_seconds=args.api_timeout,
    )
    evaluated: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    total_usage = {"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0}
    for case in frozen_cases:
        chunk = SemanticChunk.model_validate(case["chunk"])
        vote_values: list[AtomicEvidenceExtraction] = []
        for vote_index in range(1, args.votes + 1):
            checkpoint = (
                output
                / "votes"
                / case["ticker"]
                / case["atomic_evidence_key"]
                / f"vote-{vote_index:02d}.json"
            )
            source_checkpoint = (
                vote_source
                / "votes"
                / case["ticker"]
                / case["atomic_evidence_key"]
                / f"vote-{vote_index:02d}.json"
                if vote_source is not None
                else None
            )
            try:
                replay_checkpoint = (
                    checkpoint
                    if checkpoint.is_file()
                    else source_checkpoint
                    if source_checkpoint is not None and source_checkpoint.is_file()
                    else None
                )
                if replay_checkpoint is not None:
                    record = _read_json(replay_checkpoint)
                    extraction = AtomicEvidenceExtraction.model_validate(record["normalized_output"])
                else:
                    request = build_atomic_evidence_request(
                        chunk,
                        issuer_id=case["ticker"],
                        issuer_name=case.get("issuer_name"),
                        classification_vote=vote_index,
                    )
                    result = transport.execute(request, AtomicEvidenceExtraction)
                    extraction, repair_actions = normalize_atomic_extraction(result.parsed)
                    usage = result.usage.model_dump(mode="json")
                    for field in total_usage:
                        total_usage[field] += int(usage.get(field, 0))
                    _write_json(
                        checkpoint,
                        {
                            "schema_version": "atomic-classification-independent-vote/1",
                            "ticker": case["ticker"],
                            "atomic_evidence_key": case["atomic_evidence_key"],
                            "vote": vote_index,
                            "model": result.model,
                            "response_id": result.response_id,
                            "usage": usage,
                            "repair_actions": repair_actions,
                            "normalized_output": extraction.model_dump(mode="json"),
                        },
                    )
                vote_values.append(extraction)
            except Exception as exc:  # Continue; a rerun resumes only missing votes.
                failures.append(
                    {
                        "ticker": case["ticker"],
                        "atomic_evidence_key": case["atomic_evidence_key"],
                        "vote": vote_index,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
        if len(vote_values) == args.votes:
            evaluated.append(evaluate_case(case, vote_values))
    if len(evaluated) != len(frozen_cases):
        _write_json(output / "failures.json", {"failures": failures})
        raise RuntimeError(
            f"only {len(evaluated)}/{len(frozen_cases)} cases have all votes; "
            "rerun the same command to resume missing votes"
        )
    summary = summarize(evaluated, requested_votes=args.votes)
    report = {
        "schema_version": "atomic-classification-reproducibility-report/1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "return_data_used": False,
        "vote_source": str(vote_source) if vote_source is not None else None,
        "frozen_sample_path": str(sample_path),
        "usage_this_invocation": total_usage,
        "summary": summary,
        "cases": evaluated,
        "failures": failures,
    }
    _write_json(output / "atomic-classification-reproducibility.json", report)
    (output / "atomic-classification-reproducibility.md").write_text(
        _markdown(report), encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze ambiguous atomic evidence and measure independent classification consensus."
    )
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--repeat-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--vote-source",
        help="optional completed vote directory to re-evaluate without new API calls",
    )
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--votes", type=int, choices=[6, 8, 10], default=10)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh", "max"],
        default="medium",
    )
    parser.add_argument("--api-retries", type=int, default=4)
    parser.add_argument("--api-timeout", type=float, default=180.0)
    parser.add_argument("--freeze-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    result = run(parse_args())
    if "summary" not in result:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 0 if result["summary"]["classifier_stability_supported"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
