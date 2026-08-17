from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.compare_old_current_moat_economic import build_exact_panel, compare_panel


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_compare_panel_uses_same_names_returns_and_equal_tails() -> None:
    panel = [
        {
            "ticker": str(index).zfill(6),
            "market": "KOSPI",
            "size_bucket": "LARGE",
            "old_holistic_score": float(index),
            "current_public_score": float(11 - index),
            "forward_return": float(index) / 100,
        }
        for index in range(1, 11)
    ]

    first = compare_panel(
        panel,
        tie_simulations=100,
        bootstrap_samples=100,
        permutation_samples=100,
        seed="fixed-test-seed",
    )
    second = compare_panel(
        panel,
        tie_simulations=100,
        bootstrap_samples=100,
        permutation_samples=100,
        seed="fixed-test-seed",
    )

    assert first == second
    assert first["metrics"]["OLD_HOLISTIC"]["rank_ic"] == pytest.approx(1)
    assert first["metrics"]["CURRENT_PUBLIC"]["rank_ic"] == pytest.approx(-1)
    assert first["metrics"]["OLD_HOLISTIC"]["equal_count_q5_minus_q1"]["top_count"] == 2
    assert first["metrics"]["CURRENT_PUBLIC"]["equal_count_q5_minus_q1"]["bottom_count"] == 2
    assert first["direct_comparison"]["old_minus_current_rank_ic"] == pytest.approx(2)


def test_build_exact_panel_cross_checks_all_sources(tmp_path: Path) -> None:
    tickers = [str(value).zfill(6) for value in range(1, 7)]
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps({"economic": {"tickers": tickers}}),
        encoding="utf-8",
    )
    metadata = tmp_path / "metadata.csv"
    old_scores = tmp_path / "old.csv"
    current = tmp_path / "current.csv"
    prices = tmp_path / "prices.csv"
    reference = tmp_path / "reference.csv"
    _write_csv(
        metadata,
        [
            {
                "ticker": ticker,
                "company_name": f"Company {ticker}",
                "market": "KOSPI",
                "size_bucket": "LARGE",
                "old_score_signal_date": 10 * index,
            }
            for index, ticker in enumerate(tickers, start=1)
        ],
    )
    _write_csv(
        old_scores,
        [
            {
                "stock_code": ticker,
                "as_of": "2025-08-31",
                "economic_moat_score_100": 10 * index,
            }
            for index, ticker in enumerate(tickers, start=1)
        ],
    )
    _write_csv(
        current,
        [
            {
                "date": "2025-08-31",
                "ticker": ticker,
                "score_eligible": index != 6,
                "eligibility_status": "VALID_MOAT" if index != 6 else "BRIDGE_FAIL",
                "economic_moat_score": index / 2,
            }
            for index, ticker in enumerate(tickers, start=1)
        ],
    )
    _write_csv(
        prices,
        [
            {
                "ticker": ticker,
                "signal_date": "2025-08-31",
                "signal_price": 100,
                "return_date": "2025-11-30",
                "return_price": 100 + index,
            }
            for index, ticker in enumerate(tickers, start=1)
        ],
    )
    _write_csv(
        reference,
        [
            {
                "ticker": ticker,
                "score_eligible": index != 6,
                "old_score": 10 * index,
                "moat_score": index / 2,
                "forward_return": index / 100,
            }
            for index, ticker in enumerate(tickers, start=1)
        ],
    )

    panel, audit = build_exact_panel(
        protocol_path=protocol,
        sample_metadata_path=metadata,
        old_scores_path=old_scores,
        current_shadow_path=current,
        forward_prices_path=prices,
        reference_panel_path=reference,
        signal_date="2025-08-31",
        expected_eligible_count=5,
    )

    assert len(panel) == 5
    assert audit["protocol_cohort_count"] == 6
    assert audit["excluded"] == [
        {"ticker": "000006", "eligibility_status": "BRIDGE_FAIL"}
    ]
    assert audit["source_value_checks"] == {
        "old_source_matches_metadata_and_reference": True,
        "current_shadow_matches_reference": True,
        "price_derived_returns_match_reference": True,
        "eligibility_matches_reference": True,
    }


def test_build_exact_panel_rejects_reference_drift(tmp_path: Path) -> None:
    ticker = "000001"
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"economic": {"tickers": [ticker]}}), encoding="utf-8")
    files = {
        "metadata": [
            {
                "ticker": ticker,
                "company_name": "Company",
                "market": "KOSPI",
                "size_bucket": "LARGE",
                "old_score_signal_date": 50,
            }
        ],
        "old": [{"stock_code": ticker, "as_of": "2025-08-31", "economic_moat_score_100": 50}],
        "current": [
            {
                "date": "2025-08-31",
                "ticker": ticker,
                "score_eligible": True,
                "eligibility_status": "VALID_MOAT",
                "economic_moat_score": 4,
            }
        ],
        "prices": [
            {
                "ticker": ticker,
                "signal_date": "2025-08-31",
                "signal_price": 100,
                "return_date": "2025-11-30",
                "return_price": 110,
            }
        ],
        "reference": [
            {
                "ticker": ticker,
                "score_eligible": True,
                "old_score": 50,
                "moat_score": 3,
                "forward_return": 0.1,
            }
        ],
    }
    paths: dict[str, Path] = {}
    for name, rows in files.items():
        paths[name] = tmp_path / f"{name}.csv"
        _write_csv(paths[name], rows)

    with pytest.raises(ValueError, match="current source-to-reference mismatch"):
        build_exact_panel(
            protocol_path=protocol,
            sample_metadata_path=paths["metadata"],
            old_scores_path=paths["old"],
            current_shadow_path=paths["current"],
            forward_prices_path=paths["prices"],
            reference_panel_path=paths["reference"],
            signal_date="2025-08-31",
            expected_eligible_count=1,
        )
