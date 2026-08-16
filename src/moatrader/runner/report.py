from __future__ import annotations

import csv
from decimal import Decimal
from io import StringIO

from moatrader.runner.models import UniverseRunResult
from moatrader.screening import CandidateInput, RankedCandidate, SelectorConfig, ValueMoatRanker


def rank_run_result(
    result: UniverseRunResult,
    config: SelectorConfig | None = None,
) -> list[RankedCandidate]:
    candidates: list[CandidateInput] = []
    for company in result.companies:
        if (
            not company.moat_score
            or not company.dcf
            or company.dcf.fair_value_per_share <= 0
            or not company.dcf.screening_eligible
            or company.current_price is None
            or company.price_as_of is None
            or company.valuation_as_of is None
        ):
            continue
        coverage = company.moat_score.document_coverage
        coverage_scalar = Decimal(str(coverage.moat_evidence_coverage or 0))
        candidates.append(
            CandidateInput(
                issuer_id=company.issuer_id or company.ticker,
                ticker=company.ticker,
                current_price=company.current_price,
                dcf_fair_value=company.dcf.fair_value_per_share,
                moat_score=Decimal(str(company.moat_score.economic_moat_score)),
                model_confidence=Decimal(
                    str(
                        company.moat_score.evidence_confidence
                        if company.moat_score.evidence_confidence is not None
                        else company.moat_score.model_confidence
                    )
                ),
                document_coverage=coverage_scalar,
                valuation_as_of=company.valuation_as_of,
                price_as_of=company.price_as_of,
            )
        )
    return ValueMoatRanker(config).rank(candidates)


def results_csv(result: UniverseRunResult) -> str:
    stream = StringIO(newline="")
    fieldnames = [
        "ticker",
        "issuer_name",
        "status",
        "moat_score",
        "durability",
        "audit_status",
        "scoring_method",
        "evidence_confidence",
        "model_confidence",
        "mechanism_strengths",
        "outcome_strengths",
        "document_coverage",
        "dcf_fair_value",
        "current_price",
        "price_to_dcf",
        "evidence_count",
        "chunk_count",
        "selected_chunk_count",
        "strength_context_chunk_count",
        "input_tokens",
        "output_tokens",
        "error",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for company in result.companies:
        score = company.moat_score
        coverage = score.document_coverage.moat_evidence_coverage if score else None
        fair_value = company.dcf.fair_value_per_share if company.dcf else None
        writer.writerow(
            {
                "ticker": company.ticker,
                "issuer_name": company.issuer_name,
                "status": company.status.value,
                "moat_score": score.economic_moat_score if score else None,
                "durability": score.durability.value if score else None,
                "audit_status": score.audit_status.value if score else None,
                "scoring_method": score.scoring_method if score else None,
                "evidence_confidence": score.evidence_confidence if score else None,
                "model_confidence": score.model_confidence if score else None,
                "mechanism_strengths": (
                    ";".join(
                        f"{item.evidence_type.value}:{item.strength_bucket}"
                        for item in score.mechanisms
                    )
                    if score
                    else None
                ),
                "outcome_strengths": (
                    ";".join(
                        f"{item.evidence_type.value}:{item.strength_bucket}/{item.persistence_bucket}"
                        for item in score.outcome_strengths
                    )
                    if score
                    else None
                ),
                "document_coverage": coverage,
                "dcf_fair_value": fair_value,
                "current_price": company.current_price,
                "price_to_dcf": company.current_price / fair_value if company.current_price and fair_value else None,
                "evidence_count": company.evidence_count,
                "chunk_count": company.chunk_count,
                "selected_chunk_count": company.selected_chunk_count,
                "strength_context_chunk_count": company.strength_context_chunk_count,
                "input_tokens": company.llm_usage.input_tokens,
                "output_tokens": company.llm_usage.output_tokens,
                "error": company.error,
            }
        )
    return stream.getvalue()


def ranking_csv(result: UniverseRunResult) -> str:
    stream = StringIO(newline="")
    fieldnames = [
        "rank",
        "ticker",
        "moat_score",
        "price_to_dcf",
        "margin_of_safety",
        "quality_value_score",
        "moat_percentile",
        "value_percentile",
        "valuation_as_of",
        "price_as_of",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for rank, candidate in enumerate(result.ranking, start=1):
        writer.writerow(
            {
                "rank": rank,
                "ticker": candidate.ticker,
                "moat_score": candidate.moat_score,
                "price_to_dcf": candidate.price_to_dcf,
                "margin_of_safety": candidate.margin_of_safety,
                "quality_value_score": candidate.quality_value_score,
                "moat_percentile": candidate.moat_percentile,
                "value_percentile": candidate.value_percentile,
                "valuation_as_of": candidate.valuation_as_of.isoformat(),
                "price_as_of": candidate.price_as_of.isoformat(),
            }
        )
    return stream.getvalue()
