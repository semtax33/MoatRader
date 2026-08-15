from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.merge_kr_signal_panel import main, spearman


def test_spearman_handles_ties_and_detects_unstable_scores() -> None:
    assert spearman([0, 1, 1, 2], [0, 1, 1, 2]) == 1.0
    assert spearman([0, 1, 2], [2, 1, 0]) == -1.0


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
