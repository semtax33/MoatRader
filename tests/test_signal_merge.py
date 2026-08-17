from __future__ import annotations

import sys
import json
from decimal import Decimal
from pathlib import Path

import pytest

from scripts.merge_kr_signal_panel import (
    MOAT_RANK_STRATEGY,
    MOAT_RANK_VALIDATION_SCHEMA_VERSION,
    load_validated_moat_rank_report,
    main,
    rank_key_percentiles,
    spearman,
    stable_moat_rank_keys,
)
from moatrader.preflight import PREFLIGHT_SCHEMA_VERSION, ticker_set_sha256
from moatrader.runner.engine import RUNNER_VERSION


def test_spearman_handles_ties_and_detects_unstable_scores() -> None:
    assert spearman([0, 1, 1, 2], [0, 1, 1, 2]) == 1.0
    assert spearman([0, 1, 2], [2, 1, 0]) == -1.0


def test_signal_rank_key_is_public_first_and_refines_exact_ties() -> None:
    rows = [
        {
            "moat_score": "3.12",
            "rank_refinement_status": "STABLE_COMPONENTS",
            "rank_mechanism_component": "4",
            "rank_outcome_component": "0",
            "rank_durability_component": "1",
            "rank_counter_component": "1",
        },
        {
            "moat_score": "3.75",
            "rank_refinement_status": "STABLE_COMPONENTS",
            "rank_mechanism_component": "0",
            "rank_outcome_component": "0",
            "rank_durability_component": "0",
            "rank_counter_component": "2",
        },
        {
            "moat_score": "3.12",
            "rank_refinement_status": "STABLE_COMPONENTS",
            "rank_mechanism_component": "2",
            "rank_outcome_component": "0",
            "rank_durability_component": "1",
            "rank_counter_component": "1",
        },
    ]

    keys = stable_moat_rank_keys(rows)
    percentiles = rank_key_percentiles(keys)

    assert keys[1] > keys[0] > keys[2]
    assert percentiles[1] > percentiles[0] > percentiles[2]


def test_signal_rank_key_preserves_whole_bucket_tie_when_incomplete() -> None:
    rows = [
        {
            "moat_score": "3.12",
            "rank_refinement_status": "STABLE_COMPONENTS",
            "rank_mechanism_component": "4",
            "rank_outcome_component": "0",
            "rank_durability_component": "1",
            "rank_counter_component": "1",
        },
        {
            "moat_score": "3.12",
            "rank_refinement_status": "PUBLIC_SCORE_ONLY",
            "rank_mechanism_component": "",
            "rank_outcome_component": "",
            "rank_durability_component": "",
            "rank_counter_component": "",
        },
    ]

    assert stable_moat_rank_keys(rows) == [
        (Decimal("3.12"),),
        (Decimal("3.12"),),
    ]


def test_rank_validation_report_approves_only_stable_public_first_strategy(
    tmp_path: Path,
) -> None:
    path = tmp_path / "rank-validation.json"
    payload = {
        "schema_version": MOAT_RANK_VALIDATION_SCHEMA_VERSION,
        "runner_version": RUNNER_VERSION,
        "candidate_selection": {"selected": MOAT_RANK_STRATEGY},
        "production_gate": {"passed": True},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_validated_moat_rank_report(path) == payload

    payload["candidate_selection"] = {"selected": "PUBLIC_PLUS_RAW_WITHIN_BUCKET"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="stable public-first RankKey"):
        load_validated_moat_rank_report(path)


def test_large_panel_merge_is_blocked_without_preflight(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "universe.csv").write_text(
        "stock_code,name\n" + "".join(f"{index:06d},Company {index}\n" for index in range(6)),
        encoding="utf-8",
    )
    (inputs / "dates.csv").write_text("as_of\n2025-08-31\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_kr_signal_panel.py",
            "--workspace",
            str(tmp_path),
            "--run",
            f"2025-08-31={tmp_path / 'missing.json'}",
            "--output",
            str(tmp_path / "signals.csv"),
        ],
    )

    with pytest.raises(RuntimeError, match="requires a passed preflight report"):
        main()


def test_large_panel_merge_is_blocked_when_rank_stability_is_unvalidated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    inputs = tmp_path / "inputs"
    diagnostics = tmp_path / "diagnostics"
    inputs.mkdir()
    diagnostics.mkdir()
    tickers = [f"{index:06d}" for index in range(6)]
    (inputs / "universe.csv").write_text(
        "stock_code,name\n"
        + "".join(f"{ticker},Company {ticker}\n" for ticker in tickers),
        encoding="utf-8",
    )
    (inputs / "dates.csv").write_text("as_of\n2025-08-31\n", encoding="utf-8")
    (diagnostics / "moat-preflight.json").write_text(
        json.dumps(
            {
                "schema_version": PREFLIGHT_SCHEMA_VERSION,
                "passed": True,
                "runner_version": RUNNER_VERSION,
                "approved_universe_tickers_sha256": ticker_set_sha256(tickers),
                "dates": ["2025-08-31"],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_kr_signal_panel.py",
            "--workspace",
            str(tmp_path),
            "--run",
            f"2025-08-31={tmp_path / 'missing.json'}",
            "--output",
            str(tmp_path / "signals.csv"),
        ],
    )

    with pytest.raises(RuntimeError, match="rank validation report"):
        main()
