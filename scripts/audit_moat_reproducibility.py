from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from moatrader.runner.models import CompanyRunStatus, UniverseRunResult
try:
    from scripts.merge_kr_signal_panel import spearman
except ModuleNotFoundError:  # Direct `python scripts\...py` execution.
    from merge_kr_signal_panel import spearman


def _scored_companies(result: UniverseRunResult) -> dict[str, object]:
    return {
        company.ticker: company
        for company in result.companies
        if company.status == CompanyRunStatus.COMPLETE and company.moat_score is not None
    }


def _evidence_ids(company: object) -> set[str]:
    score = company.moat_score
    return {
        evidence_id
        for mechanism in score.mechanisms
        for evidence_id in mechanism.evidence_ids
    } | set(score.counterevidence_ids)


def _claim_ids(company: object) -> set[str]:
    return set(company.moat_score.canonical_claim_ids)


def _context_ids(company: object) -> set[str]:
    return set(company.moat_score.context_chunk_ids)


def _strength_signature(company: object) -> tuple[object, ...]:
    score = company.moat_score
    mechanisms = tuple(
        sorted(
            (
                item.evidence_type.value,
                item.strength_bucket,
                item.scope_materiality_bucket,
            )
            for item in score.mechanisms
        )
    )
    outcomes = tuple(
        sorted(
            (
                item.evidence_type.value,
                item.strength_bucket,
                item.persistence_bucket,
            )
            for item in score.outcome_strengths
        )
    )
    return (
        mechanisms,
        outcomes,
        score.durability.value,
        score.audit_status.value,
        score.scoring_method,
    )


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def compare_runs(
    baseline: UniverseRunResult,
    candidate: UniverseRunResult,
    *,
    minimum_score_spearman: float = 0.90,
    minimum_mean_evidence_jaccard: float = 0.50,
    minimum_mean_claim_jaccard: float = 0.50,
    minimum_mean_context_jaccard: float = 0.50,
    minimum_strength_attribute_match: float = 0.90,
    maximum_median_score_delta: float = 0.50,
    maximum_company_score_delta: float = 2.0,
    require_same_universe: bool = True,
) -> dict[str, object]:
    before = _scored_companies(baseline)
    after = _scored_companies(candidate)
    missing = sorted(set(before) - set(after))
    added = sorted(set(after) - set(before))
    common = sorted(set(before) & set(after))
    if not common:
        raise ValueError("the two runs have no common completed MOAT scores")

    before_scores = [float(before[ticker].moat_score.economic_moat_score) for ticker in common]
    after_scores = [float(after[ticker].moat_score.economic_moat_score) for ticker in common]
    deltas = [abs(left - right) for left, right in zip(before_scores, after_scores, strict=True)]
    rank_correlation = spearman(before_scores, after_scores)
    if rank_correlation is None and all(delta == 0 for delta in deltas):
        rank_correlation = 1.0
    jaccards = [
        _jaccard(_evidence_ids(before[ticker]), _evidence_ids(after[ticker])) for ticker in common
    ]
    claim_jaccards = [
        _jaccard(_claim_ids(before[ticker]), _claim_ids(after[ticker])) for ticker in common
    ]
    context_jaccards = [
        _jaccard(_context_ids(before[ticker]), _context_ids(after[ticker])) for ticker in common
    ]
    strength_attribute_matches = [
        _strength_signature(before[ticker]) == _strength_signature(after[ticker])
        for ticker in common
    ]
    failures: list[str] = []
    if require_same_universe and (missing or added):
        failures.append("completed scored universe differs")
    if rank_correlation is None or rank_correlation < minimum_score_spearman:
        failures.append(
            f"score Spearman {rank_correlation!r} is below {minimum_score_spearman:.3f}"
        )
    median_delta = statistics.median(deltas)
    largest_delta = max(deltas)
    mean_jaccard = statistics.mean(jaccards)
    mean_claim_jaccard = statistics.mean(claim_jaccards)
    mean_context_jaccard = statistics.mean(context_jaccards)
    strength_attribute_match_rate = statistics.mean(strength_attribute_matches)
    if median_delta > maximum_median_score_delta:
        failures.append(
            f"median score delta {median_delta:.3f} exceeds {maximum_median_score_delta:.3f}"
        )
    if largest_delta > maximum_company_score_delta:
        failures.append(
            f"maximum company score delta {largest_delta:.3f} exceeds {maximum_company_score_delta:.3f}"
        )
    if mean_jaccard < minimum_mean_evidence_jaccard:
        failures.append(
            f"mean evidence Jaccard {mean_jaccard:.3f} is below {minimum_mean_evidence_jaccard:.3f}"
        )
    if mean_claim_jaccard < minimum_mean_claim_jaccard:
        failures.append(
            f"mean claim Jaccard {mean_claim_jaccard:.3f} is below {minimum_mean_claim_jaccard:.3f}"
        )
    if mean_context_jaccard < minimum_mean_context_jaccard:
        failures.append(
            f"mean context Jaccard {mean_context_jaccard:.3f} is below "
            f"{minimum_mean_context_jaccard:.3f}"
        )
    if strength_attribute_match_rate < minimum_strength_attribute_match:
        failures.append(
            f"strength attribute match {strength_attribute_match_rate:.3f} is below "
            f"{minimum_strength_attribute_match:.3f}"
        )
    details = [
        {
            "ticker": ticker,
            "baseline_score": before_scores[index],
            "candidate_score": after_scores[index],
            "absolute_score_delta": deltas[index],
            "evidence_jaccard": jaccards[index],
            "claim_jaccard": claim_jaccards[index],
            "context_jaccard": context_jaccards[index],
            "strength_attributes_equal": strength_attribute_matches[index],
        }
        for index, ticker in enumerate(common)
    ]
    return {
        "schema_version": "moatrader-moat-reproducibility/3",
        "baseline_run_id": baseline.run_id,
        "candidate_run_id": candidate.run_id,
        "common_company_count": len(common),
        "missing_tickers": missing,
        "added_tickers": added,
        "score_spearman": rank_correlation,
        "median_absolute_score_delta": median_delta,
        "maximum_absolute_score_delta": largest_delta,
        "mean_evidence_jaccard": mean_jaccard,
        "mean_claim_jaccard": mean_claim_jaccard,
        "mean_context_jaccard": mean_context_jaccard,
        "strength_attribute_match_rate": strength_attribute_match_rate,
        "passed": not failures,
        "failures": failures,
        "companies": details,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate repeat or input-reordering runs for MOAT score/evidence invariance."
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-score-spearman", type=float, default=0.90)
    parser.add_argument("--minimum-mean-evidence-jaccard", type=float, default=0.50)
    parser.add_argument("--minimum-mean-claim-jaccard", type=float, default=0.50)
    parser.add_argument("--minimum-mean-context-jaccard", type=float, default=0.50)
    parser.add_argument("--minimum-strength-attribute-match", type=float, default=0.90)
    parser.add_argument("--maximum-median-score-delta", type=float, default=0.50)
    parser.add_argument("--maximum-company-score-delta", type=float, default=2.0)
    parser.add_argument("--allow-universe-mismatch", action="store_true")
    args = parser.parse_args()

    baseline = UniverseRunResult.model_validate_json(args.baseline.read_text(encoding="utf-8-sig"))
    candidate = UniverseRunResult.model_validate_json(args.candidate.read_text(encoding="utf-8-sig"))
    report = compare_runs(
        baseline,
        candidate,
        minimum_score_spearman=args.minimum_score_spearman,
        minimum_mean_evidence_jaccard=args.minimum_mean_evidence_jaccard,
        minimum_mean_claim_jaccard=args.minimum_mean_claim_jaccard,
        minimum_mean_context_jaccard=args.minimum_mean_context_jaccard,
        minimum_strength_attribute_match=args.minimum_strength_attribute_match,
        maximum_median_score_delta=args.maximum_median_score_delta,
        maximum_company_score_delta=args.maximum_company_score_delta,
        require_same_universe=not args.allow_universe_mismatch,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise RuntimeError("MOAT reproducibility gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
