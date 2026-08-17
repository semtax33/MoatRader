from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal
from io import StringIO
from pathlib import Path

from moatrader.runner.models import CompanyRunStatus, UniverseRunResult
from moatrader.runner.report import rank_run_result, ranking_csv, results_csv
from moatrader.runstore import RunStore


def _coverage_values(company: object) -> tuple[float | None, float | None]:
    score = getattr(company, "moat_score", None)
    if score is None:
        return None, None
    coverage = score.document_coverage
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
    return (min(values) if values else None), coverage.evidence_retention


def _detail_rows(result: UniverseRunResult) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for company in result.companies:
        score = company.moat_score
        fair_value = company.dcf.fair_value_per_share if company.dcf else None
        price = company.current_price
        coverage_min, evidence_coverage = _coverage_values(company)
        ratio = price / fair_value if price is not None and fair_value and fair_value > 0 else None
        margin = Decimal(1) - ratio if ratio is not None else None
        evidence_confidence = (
            score.evidence_confidence
            if score is not None and score.evidence_confidence is not None
            else (score.model_confidence if score is not None else None)
        )
        composite = None
        if (
            score is not None
            and score.score_eligible
            and margin is not None
            and coverage_min is not None
        ):
            composite = (
                Decimal(str(score.economic_moat_score))
                / Decimal(10)
                * max(Decimal(0), margin)
                * Decimal(str(evidence_confidence))
                * Decimal(str(coverage_min))
            )
        rows.append(
            {
                "ticker": company.ticker,
                "issuer_name": company.issuer_name,
                "status": company.status.value,
                "moat_score": score.economic_moat_score if score else None,
                "rank_refinement_status": (
                    score.rank_refinement_status.value
                    if score and score.rank_refinement_status
                    else None
                ),
                "rank_mechanism_component": (
                    score.rank_refinement.mechanism_component
                    if score and score.rank_refinement
                    else None
                ),
                "rank_outcome_component": (
                    score.rank_refinement.outcome_component
                    if score and score.rank_refinement
                    else None
                ),
                "rank_durability_component": (
                    score.rank_refinement.durability_component
                    if score and score.rank_refinement
                    else None
                ),
                "rank_counter_component": (
                    score.rank_refinement.counter_component
                    if score and score.rank_refinement
                    else None
                ),
                "durability": score.durability.value if score else None,
                "audit_status": score.audit_status.value if score else None,
                "score_eligible": score.score_eligible if score else None,
                "eligibility_status": score.eligibility_status.value if score else None,
                "evidence_confidence": evidence_confidence,
                "model_confidence": score.model_confidence if score else None,
                "document_coverage_min": coverage_min,
                "evidence_coverage": evidence_coverage,
                "dcf_fair_value": fair_value,
                "current_price": price,
                "price_to_dcf": ratio,
                "margin_of_safety": margin,
                "quality_value_score": composite,
                "evidence_count": company.evidence_count,
                "error": company.error,
            }
        )
    return rows


def _csv(rows: list[dict[str, object]], *, add_rank: bool = False) -> str:
    if not rows:
        return ""
    stream = StringIO(newline="")
    fields = list(rows[0])
    if add_rank:
        fields = ["rank", *fields]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for rank, row in enumerate(rows, start=1):
        writer.writerow({"rank": rank, **row} if add_rank else row)
    return stream.getvalue()


def _markdown(
    result: UniverseRunResult,
    detail: list[dict[str, object]],
    shortlist: list[dict[str, object]],
) -> str:
    complete = sum(company.status == CompanyRunStatus.COMPLETE for company in result.companies)
    failed = sum(company.status == CompanyRunStatus.FAILED for company in result.companies)
    positive_dcf = sum(row["price_to_dcf"] is not None for row in detail)
    lines = [
        "# MOAT + DCF 유니버스 결과",
        "",
        f"- 기준 시점: {result.as_of.isoformat()}",
        f"- 대상: {len(result.companies)}개",
        f"- 완료: {complete}개",
        f"- 실패/보류: {failed}개",
        f"- 양(+)의 DCF 공정가치 산출: {positive_dcf}개",
        f"- 기본 엄격 스크린 통과: {len(result.ranking)}개",
        f"- 리서치 후보: {len(shortlist)}개",
        "",
        "## 리서치 후보 상위 20개",
        "",
        "리서치 후보 기준은 MOAT 5점 이상, 안전마진 20% 이상, 모델 신뢰도 0.5 이상입니다. 문서 커버리지는 순위 점수에 반영되지만 이 후보 단계에서 강제 탈락시키지는 않습니다.",
        "",
        "| 순위 | 종목 | 기업 | MOAT | 신뢰도 | 현재가 | DCF | Price/DCF | 안전마진 | 종합점수 |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(shortlist[:20], start=1):
        lines.append(
            "| {rank} | {ticker} | {name} | {moat:.1f} | {confidence:.2f} | {price:,.0f} | "
            "{dcf:,.0f} | {ratio:.3f} | {margin:.1%} | {composite:.5f} |".format(
                rank=rank,
                ticker=row["ticker"],
                name=row["issuer_name"] or "",
                moat=float(row["moat_score"]),
                confidence=float(row["model_confidence"]),
                price=float(row["current_price"]),
                dcf=float(row["dcf_fair_value"]),
                ratio=float(row["price_to_dcf"]),
                margin=float(row["margin_of_safety"]),
                composite=float(row["quality_value_score"] or 0),
            )
        )
    lines.extend(
        [
            "",
            "## 해석 주의",
            "",
            "이 순위는 공시 근거 기반의 정량·정성 스크리닝 결과이며 투자 권유가 아닙니다. DCF는 가정 변화에 민감하므로 후보 종목은 원문 공시, 산업 구조, 자본배분, 민감도 분석을 추가 확인해야 합니다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge completed MOAT universe-run shards.")
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run-id", default="combined")
    args = parser.parse_args()

    shard_results = [
        UniverseRunResult.model_validate_json((path / "run-result.json").read_text(encoding="utf-8-sig"))
        for path in args.shards
    ]
    by_ticker = {}
    for shard in shard_results:
        for company in shard.companies:
            by_ticker[company.ticker] = company
    companies = sorted(by_ticker.values(), key=lambda company: company.ticker)
    combined = UniverseRunResult(
        run_id=args.run_id,
        as_of=shard_results[0].as_of,
        started_at=min(item.started_at for item in shard_results),
        completed_at=datetime.now(timezone.utc),
        companies=companies,
    )
    combined = combined.model_copy(update={"ranking": rank_run_result(combined)})

    detail = _detail_rows(combined)
    rankable = [row for row in detail if row["quality_value_score"] is not None]
    rankable.sort(
        key=lambda row: (Decimal(str(row["quality_value_score"])), -Decimal(str(row["price_to_dcf"]))),
        reverse=True,
    )
    shortlist = [
        row
        for row in rankable
        if Decimal(str(row["moat_score"])) >= Decimal(5)
        and Decimal(str(row["margin_of_safety"])) >= Decimal("0.20")
        and Decimal(str(row["evidence_confidence"])) >= Decimal("0.50")
        and row["audit_status"] != "FAIL"
    ]

    store = RunStore(args.output)
    store.write_json(store.root / "run-result.json", combined)
    store.write_text(store.root / "results.csv", results_csv(combined))
    store.write_text(store.root / "strict-ranking.csv", ranking_csv(combined))
    store.write_text(store.root / "full-positive-dcf-ranking.csv", _csv(rankable, add_rank=True))
    store.write_text(store.root / "research-shortlist.csv", _csv(shortlist, add_rank=True))
    store.write_text(store.root / "summary.md", _markdown(combined, detail, shortlist))


if __name__ == "__main__":
    main()
