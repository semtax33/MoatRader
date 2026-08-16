from __future__ import annotations

from scripts.materialize_moat_rank_shadow import (
    CANDIDATE_PUBLIC,
    CANDIDATE_RAW,
    CANDIDATE_STABLE,
    distribution_metrics,
    repeat_candidate_metrics,
)


def _row(
    ticker: str,
    public: float,
    raw: float | None,
    *,
    eligible: bool = True,
    mechanism: float = 2.0,
) -> dict[str, object]:
    return {
        "date": "2025-08-31",
        "ticker": ticker,
        "score_eligible": eligible,
        "economic_moat_score": public,
        "raw_ordinal_shadow_score": raw,
        "rank_refinement_status": "STABLE_COMPONENTS" if eligible else "INELIGIBLE",
        "rank_mechanism_component": mechanism if eligible else None,
        "rank_outcome_component": 0.0 if eligible else None,
        "rank_durability_component": 1.0 if eligible else None,
        "rank_counter_component": 1.0 if eligible else None,
    }


def test_shadow_distribution_reports_resolution_and_ineligible_null_semantics() -> None:
    report = distribution_metrics(
        [
            _row("A", 3.12, 2.8),
            _row("B", 3.12, 3.1),
            _row("C", 3.12, 3.4),
            _row("D", 0, None, eligible=False),
        ]
    )

    date = report["dates"][0]["candidates"]
    assert report["ineligible_candidates_are_null"] is True
    assert date[CANDIDATE_PUBLIC]["distinct_rank_count"] == 1
    assert date[CANDIDATE_RAW]["distinct_rank_count"] == 3
    assert date[CANDIDATE_STABLE]["distinct_rank_count"] == 1
    assert date[CANDIDATE_PUBLIC]["max_single_rank_share"] == 1
    assert date[CANDIDATE_RAW]["max_single_rank_share"] == 1 / 3


def test_shadow_repeat_metrics_exposes_rank_instability() -> None:
    baseline = [_row("A", 3.12, 1), _row("B", 3.12, 2), _row("C", 3.12, 3)]
    repeat = [_row("A", 3.12, 3), _row("B", 3.12, 2), _row("C", 3.12, 1)]

    report = repeat_candidate_metrics(baseline, repeat)

    assert report["common_eligible_count"] == 3
    assert report["candidates"][CANDIDATE_RAW]["rank_key_spearman"] == -1
    assert report["candidates"][CANDIDATE_STABLE]["rank_key_spearman"] == 1
    assert report["candidates"][CANDIDATE_STABLE]["pairwise_order_agreement"] == 1
