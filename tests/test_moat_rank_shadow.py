from __future__ import annotations

from scripts.materialize_moat_rank_shadow import distribution_metrics, repeat_metrics


def _row(
    ticker: str,
    public: float,
    rank: float | None,
    *,
    eligible: bool = True,
) -> dict[str, object]:
    return {
        "date": "2025-08-31",
        "ticker": ticker,
        "score_eligible": eligible,
        "economic_moat_score": public,
        "economic_moat_rank_score": rank,
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

    date = report["dates"][0]
    assert report["ineligible_rank_is_none"] is True
    assert date["public_distinct_score_count"] == 1
    assert date["rank_distinct_score_count"] == 3
    assert date["public_max_single_score_share"] == 1
    assert date["rank_max_single_score_share"] == 1 / 3


def test_shadow_repeat_metrics_exposes_rank_instability() -> None:
    baseline = [_row("A", 3.12, 1), _row("B", 3.12, 2), _row("C", 3.12, 3)]
    repeat = [_row("A", 3.12, 3), _row("B", 3.12, 2), _row("C", 3.12, 1)]

    report = repeat_metrics(baseline, repeat)

    assert report["common_eligible_count"] == 3
    assert report["rank_score_spearman"] == -1
    assert report["max_absolute_delta"] == 2
