from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from moatrader.canonical.models import CanonicalDocumentBundle, DocumentMetadata, SourceType

from conftest import build_dart_bundle


def test_bundle_round_trip_and_json_schema_preserve_discriminated_nodes():
    bundle = build_dart_bundle("<html><body><h1>II. 사업의 내용</h1><p>문단입니다.</p></body></html>")
    restored = CanonicalDocumentBundle.model_validate_json(bundle.to_json())
    assert restored == bundle
    schema = CanonicalDocumentBundle.model_json_schema()
    assert schema["title"] == "CanonicalDocumentBundle"
    assert "SectionNode" in schema["$defs"]
    assert "StructuredFact" in schema["$defs"]


def test_available_at_is_timezone_aware_and_controls_point_in_time_view():
    bundle = build_dart_bundle("<html><body><p>공시 내용</p></body></html>")
    assert bundle.as_of(datetime(2025, 5, 15, 0, 0, tzinfo=timezone.utc)) is None
    assert bundle.as_of(datetime(2025, 5, 15, 1, 0, tzinfo=timezone.utc)) is bundle
    with pytest.raises(ValueError, match="cutoff must be timezone-aware"):
        bundle.as_of(datetime(2025, 5, 15))


def test_metadata_rejects_naive_available_at():
    valid = build_dart_bundle("<html><body><p>x</p></body></html>").metadata.model_dump()
    valid["available_at"] = datetime(2025, 1, 1)
    with pytest.raises(ValidationError, match="timezone-aware"):
        DocumentMetadata.model_validate(valid)


def test_source_specific_identifiers_do_not_leak_into_top_level_contract():
    bundle = build_dart_bundle("<html><body><p>x</p></body></html>")
    assert bundle.metadata.source_type == SourceType.DART
    assert bundle.metadata.source_specific["rcept_no"] == "20250515000123"
    assert "rcept_no" not in DocumentMetadata.model_fields

