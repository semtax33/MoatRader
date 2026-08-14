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
            or company.current_price is None
            or company.price_as_of is None
            or company.valuation_as_of is None
        ):
            continue
        coverage = company.moat_score.document_coverage
        values = [
            value
            for value in (
                coverage.token_retention,
                coverage.evidence_retention,
                coverage.char_retention,
                coverage.section_retention,
                coverage.table_retention,
                coverage.numeric_retention,
            )
            if value is not None
        ]
        coverage_scalar = Decimal(str(min(values))) if values else Decimal(0)
        candidates.append(
            CandidateInput(
                issuer_id=company.issuer_id or company.ticker,
                ticker=company.ticker,
                current_price=company.current_price,
                dcf_fair_value=company.dcf.fair_value_per_share,
                moat_score=Decimal(str(company.moat_score.economic_moat_score)),
                model_confidence=Decimal(str(company.moat_score.model_confidence)),
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
        "model_confidence",
        "document_coverage",
        "dcf_fair_value",
        "current_price",
        "price_to_dcf",
        "evidence_count",
        "chunk_count",
        "selected_chunk_count",
        "input_tokens",
        "output_tokens",
        "error",
    ]
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for company in result.companies:
        score = company.moat_score
        coverage_values = []
        if score:
            coverage_values = [
                value
                for value in (
                    score.document_coverage.token_retention,
                    score.document_coverage.evidence_retention,
                    score.document_coverage.char_retention,
                    score.document_coverage.section_retention,
                    score.document_coverage.table_retention,
                    score.document_coverage.numeric_retention,
                )
                if value is not None
            ]
        coverage = min(coverage_values) if coverage_values else None
        fair_value = company.dcf.fair_value_per_share if company.dcf else None
        writer.writerow(
            {
                "ticker": company.ticker,
                "issuer_name": company.issuer_name,
                "status": company.status.value,
                "moat_score": score.economic_moat_score if score else None,
                "durability": score.durability.value if score else None,
                "model_confidence": score.model_confidence if score else None,
                "document_coverage": coverage,
                "dcf_fair_value": fair_value,
                "current_price": company.current_price,
                "price_to_dcf": company.current_price / fair_value if company.current_price and fair_value else None,
                "evidence_count": company.evidence_count,
                "chunk_count": company.chunk_count,
                "selected_chunk_count": company.selected_chunk_count,
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
                "valuation_as_of": candidate.valuation_as_of.isoformat(),
                "price_as_of": candidate.price_as_of.isoformat(),
            }
        )
    return stream.getvalue()
