from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from moatrader.quality import ParserQualityAssessment
from moatrader.runstore import RunStore


def test_runstore_serializes_nested_pydantic_models_as_objects(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    path = tmp_path / "models.json"
    store.write_json(
        path,
        [ParserQualityAssessment(source_document_id="DOC1", passed=True)],
    )

    assert store.read_json(path) == [
        {
            "source_document_id": "DOC1",
            "passed": True,
            "failures": [],
            "warnings": [],
        }
    ]


def test_runstore_retries_a_transient_windows_replace_lock(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    path = tmp_path / "audit.jsonl"
    original_replace = Path.replace
    attempts = 0

    def flaky_replace(source: Path, target: Path) -> Path:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(5, "transient lock", str(target))
        return original_replace(source, target)

    with patch.object(Path, "replace", flaky_replace), patch("moatrader.runstore.time.sleep"):
        store.write_text(path, "ok\n")

    assert attempts == 3
    assert path.read_text(encoding="utf-8") == "ok\n"
