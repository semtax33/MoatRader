from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from moatrader.evidence.models import (
    CoverageMetrics,
    Durability,
    EvidenceType,
    MoatMechanismScore,
    MoatScore,
)
from moatrader.preflight import validate_preflight_approval
from moatrader.runner.engine import RUNNER_VERSION
from moatrader.runner.models import (
    CompanyRunResult,
    CompanyRunStatus,
    UniverseRunConfig,
    UniverseRunResult,
)
from scripts.approve_moat_preflight import main as approve_main
from scripts.setup_kr_signal_backtest import select_preflight_sample


def _score(ticker: str, as_of: str) -> MoatScore:
    public_score = float(int(ticker[-1]) + 3)
    return MoatScore(
        as_of=date.fromisoformat(as_of),
        economic_moat_score=public_score,
        mechanisms=[
            MoatMechanismScore(
                evidence_type=EvidenceType.SWITCHING_COST,
                score=public_score,
                evidence_ids=[f"E-{ticker}"],
                rationale="deterministic fixture",
            )
        ],
        durability=Durability.MEDIUM,
        model_confidence=0.8,
        document_coverage=CoverageMetrics(moat_evidence_coverage=1),
        llm_proposed_score=7,
    )


def _write_run(
    root: Path,
    *,
    run_id: str,
    as_of: str,
    tickers: list[str],
    replayed: bool,
) -> Path:
    directory = root / run_id
    companies = []
    for ticker in tickers:
        artifact = directory / "companies" / ticker
        artifact.mkdir(parents=True)
        if replayed:
            (artifact / "llm-calls.jsonl").write_text(
                json.dumps(
                    {
                        "task": "LOCAL_EVIDENCE_EXTRACTION",
                        "model": "fixture",
                        "replayed": True,
                        "replay_cache_key": f"cache-{as_of}-{ticker}",
                        "normalized_output_sha256": f"output-{as_of}-{ticker}",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        metamorphic_names = [
            "sentence_shuffle",
            "paragraph_shuffle",
            "duplicate_evidence",
            "summary_injection",
            "whitespace_heading_change",
            "irrelevant_boilerplate_injection",
            "node_order_change",
        ]
        (artifact / "metamorphic-audit.json").write_text(
            json.dumps(
                {
                    "schema_version": "moatrader-moat-metamorphic/1",
                    "passed": True,
                    "failures": [],
                    "transformations": {
                        name: {
                            "passed": True,
                            "atomic_key_jaccard": 1.0,
                            "evidence_jaccard": 1.0,
                            "claim_jaccard": 1.0,
                            "score_delta": 0.0,
                        }
                        for name in metamorphic_names
                    },
                }
            ),
            encoding="utf-8",
        )
        companies.append(
            CompanyRunResult(
                ticker=ticker,
                status=CompanyRunStatus.COMPLETE,
                run_signature=f"signature-{as_of}-{ticker}",
                source_document_ids=[f"D-{as_of}-{ticker}"],
                moat_score=_score(ticker, as_of),
                artifact_directory=str(artifact),
                runner_version=RUNNER_VERSION,
            )
        )
    result = UniverseRunResult(
        run_id=run_id,
        as_of=datetime.fromisoformat(f"{as_of}T23:59:59+09:00"),
        started_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        companies=companies,
    )
    path = directory / "run-result.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    config = UniverseRunConfig(
        run_id=run_id,
        as_of=result.as_of,
        experiment_id="experiment-a",
        llm_replay_cache_directory=str(root / "replay"),
    )
    (directory / "run-config.json").write_text(
        json.dumps({**config.model_dump(mode="json"), "tickers": tickers}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def test_sample_selection_is_deterministic_and_stratified() -> None:
    rows = [
        {"stock_code": f"{index:06d}", "market": market, "size_bucket": size, "selection_seed": "42"}
        for index, (market, size) in enumerate(
            [
                ("KOSPI", "LARGE"),
                ("KOSPI", "MID"),
                ("KOSDAQ", "SMALL"),
                ("KOSDAQ", "MID"),
                ("KOSPI", "SMALL"),
                ("KOSDAQ", "LARGE"),
            ],
            start=1,
        )
    ]

    first = select_preflight_sample(rows, sample_size=5, explicit_tickers=[])
    second = select_preflight_sample(rows, sample_size=5, explicit_tickers=[])

    assert first == second
    assert len(first) == 5
    assert select_preflight_sample(rows, sample_size=5, explicit_tickers=["1", "2", "3"]) == [
        "000001",
        "000002",
        "000003",
    ]


def test_preflight_approval_requires_replayed_repeat_runs_and_approves_full_universe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    inputs = workspace / "inputs"
    inputs.mkdir(parents=True)
    dates = ["2025-08-31", "2025-11-30", "2026-02-28", "2026-05-31"]
    sample = ["000001", "000002", "000003"]
    universe = [f"{index:06d}" for index in range(1, 7)]
    (inputs / "universe.csv").write_text(
        "stock_code,name\n" + "".join(f"{ticker},Company {ticker}\n" for ticker in universe),
        encoding="utf-8",
    )
    (inputs / "dates.csv").write_text(
        "as_of\n" + "".join(f"{value}\n" for value in dates),
        encoding="utf-8",
    )
    (workspace / "workspace-manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "moatrader-kr-signal-backtest/2",
                "experiment_id": "experiment-a",
                "dates": dates,
                "preflight_sample_tickers": sample,
            }
        ),
        encoding="utf-8",
    )
    baseline_paths = {
        value: _write_run(
            workspace / "runs",
            run_id=f"baseline-{value}",
            as_of=value,
            tickers=sample,
            replayed=False,
        )
        for value in dates
    }
    candidate_paths = {
        value: _write_run(
            workspace / "runs",
            run_id=f"candidate-{value}",
            as_of=value,
            tickers=sample,
            replayed=True,
        )
        for value in dates
    }
    output = workspace / "diagnostics" / "moat-preflight.json"
    argv = ["approve_moat_preflight.py", "--workspace", str(workspace), "--output", str(output)]
    for value in dates:
        argv.extend(["--baseline", f"{value}={baseline_paths[value]}"])
        argv.extend(["--candidate", f"{value}={candidate_paths[value]}"])
    monkeypatch.setattr(sys, "argv", argv)

    assert approve_main() == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["passed"] is True
    assert report["candidate_replay_calls"] == 12
    assert report["candidate_replay_hits"] == 12

    full_config = UniverseRunConfig(
        run_id="full",
        as_of=datetime.fromisoformat("2025-08-31T23:59:59+09:00"),
        experiment_id="experiment-a",
        llm_replay_cache_directory=str(workspace / "replay"),
    )
    validate_preflight_approval(
        output,
        universe_tickers=universe,
        as_of_date="2025-08-31",
        config=full_config,
        runner_version=RUNNER_VERSION,
    )

    changed = full_config.model_copy(update={"moat_reasoning_effort": "high"})
    with pytest.raises(ValueError, match="execution contract differs"):
        validate_preflight_approval(
            output,
            universe_tickers=universe,
            as_of_date="2025-08-31",
            config=changed,
            runner_version=RUNNER_VERSION,
        )
