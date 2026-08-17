from __future__ import annotations

import json
from pathlib import Path

from moatrader.cli import _identifier_values, _preflight_universe_tickers, main


ROOT = Path(__file__).resolve().parents[1]


def test_cli_can_dry_run_one_selected_ticker(tmp_path: Path, capsys: object) -> None:
    exit_code = main(
        [
            "analyze",
            "run",
            "--universe",
            str(ROOT / "examples" / "universe.csv"),
            "--ticker",
            "SAMPLE",
            "--as-of",
            "2025-05-16T00:00:00+09:00",
            "--output",
            str(tmp_path),
            "--run-id",
            "cli-dry",
            "--dry-run",
        ]
    )

    assert exit_code == 0
    run_dir = tmp_path / "cli-dry"
    assert (run_dir / "run-result.json").is_file()
    run_config = json.loads((run_dir / "run-config.json").read_text(encoding="utf-8"))
    assert run_config["summary_model"] == "gpt-5-nano"
    assert run_config["moat_model"] == "gpt-5.6-luna"
    assert main(["moat", "status", "--run-dir", str(run_dir)]) == 0
    assert main(["screen", "rank", "--run-dir", str(run_dir)]) == 0
    assert (run_dir / "ranking.csv").is_file()


def test_cli_status_reads_in_progress_checkpoints(tmp_path: Path) -> None:
    run_dir = tmp_path / "in-progress"
    checkpoint = run_dir / "companies" / "AAA" / "checkpoint.json"
    checkpoint.parent.mkdir(parents=True)
    (run_dir / "run-config.json").write_text(
        '{"run_id":"in-progress","as_of":"2025-01-01T00:00:00+00:00","tickers":["AAA","BBB"]}',
        encoding="utf-8",
    )
    checkpoint.write_text('{"stage":"SUMMARIZING"}', encoding="utf-8")

    assert main(["moat", "status", "--run-dir", str(run_dir)]) == 0


def test_collect_commands_fail_cleanly_when_required_credentials_are_missing(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    monkeypatch.delenv("DART_API_KEY", raising=False)  # type: ignore[attr-defined]
    dart_code = main(
        [
            "collect",
            "dart",
            "--from",
            "2025-01-01",
            "--to",
            "2025-01-31",
            "--stock-code",
            "005930",
            "--output",
            str(tmp_path),
        ]
    )
    assert dart_code == 2

    monkeypatch.delenv("SEC_USER_AGENT", raising=False)  # type: ignore[attr-defined]
    sec_code = main(
        [
            "collect",
            "sec",
            "--from",
            "2025-01-01",
            "--to",
            "2025-01-31",
            "--ticker",
            "AAPL",
            "--output",
            str(tmp_path),
        ]
    )
    assert sec_code == 2
    assert "DART_API_KEY" in capsys.readouterr().err  # type: ignore[attr-defined]


def test_identifier_files_support_large_comma_or_newline_universes(tmp_path: Path) -> None:
    path = tmp_path / "tickers.txt"
    path.write_text("AAPL, MSFT\n# comment\nNVDA # leader\nAAPL\n", encoding="utf-8")

    assert _identifier_values(["GOOG"], [str(path)]) == ["GOOG", "AAPL", "MSFT", "NVDA"]


def test_preflight_uses_original_workspace_universe_when_a_date_has_no_pit_document(
    tmp_path: Path,
) -> None:
    workspace_manifest = tmp_path / "workspace-manifest.json"
    workspace_manifest.write_text("{}", encoding="utf-8")
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "universe.csv").write_text(
        "stock_code,name\n005930,Samsung\n094800,No PIT filing\n",
        encoding="utf-8",
    )

    assert _preflight_universe_tickers(workspace_manifest, ["005930"]) == [
        "005930",
        "094800",
    ]


def test_large_manifest_is_blocked_without_preflight_report(tmp_path: Path, capsys: object) -> None:
    source = ROOT / "examples" / "sample-dart.html"
    metadata = ROOT / "examples" / "sample-dart-metadata.json"
    universe = tmp_path / "large-universe.csv"
    universe.write_text(
        "ticker,source,input,metadata,issuer_name\n"
        + "".join(
            f"T{index:03d},DART,{source},{metadata},Company {index}\n"
            for index in range(6)
        ),
        encoding="utf-8",
    )

    code = main(
        [
            "moat",
            "run",
            "--universe",
            str(universe),
            "--as-of",
            "2025-05-16T00:00:00+09:00",
            "--output",
            str(tmp_path / "runs"),
            "--run-id",
            "blocked-large",
            "--dry-run",
        ]
    )

    assert code == 2
    assert "requires a passed --preflight-report" in capsys.readouterr().err  # type: ignore[attr-defined]
