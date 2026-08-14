from __future__ import annotations

from pathlib import Path

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
