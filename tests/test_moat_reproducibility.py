from __future__ import annotations

from datetime import datetime

from moatrader.evidence.models import CoverageMetrics, Durability, EvidenceType, MoatMechanismScore, MoatScore
from moatrader.runner.models import CompanyRunResult, CompanyRunStatus, UniverseRunResult
from scripts.audit_moat_reproducibility import compare_runs


NOW = datetime.fromisoformat("2025-01-01T00:00:00+00:00")


def _run(run_id: str, scores: dict[str, float]) -> UniverseRunResult:
    companies = []
    for ticker, score in scores.items():
        companies.append(
            CompanyRunResult(
                ticker=ticker,
                status=CompanyRunStatus.COMPLETE,
                run_signature=f"sig-{ticker}",
                artifact_directory=f"/tmp/{ticker}",
                moat_score=MoatScore(
                    as_of=NOW.date(),
                    economic_moat_score=score,
                    mechanisms=[
                        MoatMechanismScore(
                            evidence_type=EvidenceType.SWITCHING_COST,
                            score=score,
                            evidence_ids=[f"E-{ticker}"],
                            rationale="test",
                        )
                    ],
                    durability=Durability.HIGH,
                    model_confidence=0.8,
                    document_coverage=CoverageMetrics(moat_evidence_coverage=1),
                ),
            )
        )
    return UniverseRunResult(
        run_id=run_id,
        as_of=NOW,
        started_at=NOW,
        completed_at=NOW,
        companies=companies,
    )


def test_reproducibility_gate_passes_reordered_equal_results() -> None:
    report = compare_runs(
        _run("a", {"AAA": 4, "BBB": 6, "CCC": 8}),
        _run("b", {"CCC": 8, "AAA": 4, "BBB": 6}),
    )

    assert report["passed"] is True
    assert report["score_spearman"] == 1
    assert report["mean_evidence_jaccard"] == 1


def test_reproducibility_gate_fails_rank_reversal() -> None:
    report = compare_runs(
        _run("a", {"AAA": 4, "BBB": 6, "CCC": 8}),
        _run("b", {"AAA": 8, "BBB": 6, "CCC": 4}),
    )

    assert report["passed"] is False
    assert any("Spearman" in failure for failure in report["failures"])
