from __future__ import annotations

import json
from pathlib import Path

from scripts.run_multi_period_economic_holdout import _batch_status, _command


def test_batch_status_requires_every_expected_company(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run-result.json").write_text(
        json.dumps(
            {
                "companies": [
                    {"ticker": "000001", "status": "COMPLETE"},
                    {"ticker": "000002", "status": "FAILED"},
                ]
            }
        ),
        encoding="utf-8",
    )

    status = _batch_status(run_dir, {"000001", "000002", "000003"})

    assert status["complete"] is False
    assert status["failed"] == ["000002", "000003"]
    assert status["missing"] == ["000003"]


def test_command_pins_models_and_only_resumes_existing_run(tmp_path: Path) -> None:
    command = _command(
        manifest=tmp_path / "batch.csv",
        output=tmp_path / "runs",
        run_id="batch-1",
        date="2025-11-30",
        replay_cache=tmp_path / "replay",
        experiment_id="experiment",
        resume=True,
        workers=2,
    )

    assert command[command.index("--summary-model") + 1] == "gpt-5-nano"
    assert command[command.index("--moat-model") + 1] == "gpt-5.6-luna"
    assert command[command.index("--workers") + 1] == "2"
    assert command[-1] == "--resume"
