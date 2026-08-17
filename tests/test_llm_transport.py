from __future__ import annotations

from types import SimpleNamespace

from moatrader.evidence.models import ContextualMoatAssessment, EvidenceExtractionResult
from moatrader.evidence.models import MoatScore
from moatrader.llm import LLMRequest, LLMTask, OpenAIResponsesTransport
from moatrader.llm.contracts import _prompt_cache_key
from moatrader.llm.transport import _canonical_json_object, _openai_compatible_schema


def test_openai_schema_removes_unsupported_lookaround_but_keeps_safe_patterns() -> None:
    schema = {
        "properties": {
            "decimal": {"type": "string", "pattern": r"^(?!^[-+.]*$)[+-]?\d+$"},
            "ticker": {"type": "string", "pattern": r"^\d{6}$"},
        },
        "$defs": [{"pattern": r"(?<=prefix)value"}],
    }

    compatible = _openai_compatible_schema(schema)

    assert "pattern" not in compatible["properties"]["decimal"]
    assert compatible["properties"]["ticker"]["pattern"] == r"^\d{6}$"
    assert "pattern" not in compatible["$defs"][0]
    assert schema["properties"]["decimal"]["pattern"]  # input is not mutated


def test_canonical_json_object_recursively_sorts_keys() -> None:
    value = {"z": [{"b": 2, "a": 1}], "a": {"d": 4, "c": 3}}

    assert list(_canonical_json_object(value)) == ["a", "z"]
    assert list(_canonical_json_object(value)["a"]) == ["c", "d"]
    assert list(_canonical_json_object(value)["z"][0]) == ["a", "b"]


def test_openai_transport_defers_client_initialization_for_offline_replay() -> None:
    transport = OpenAIResponsesTransport()

    assert transport.client is None


def test_prompt_cache_key_is_stable_and_rate_partitioned() -> None:
    first = _prompt_cache_key(
        "atomic-v4", static_prefix="stable rubric", routing_identity="EVIDENCE-1"
    )
    repeated = _prompt_cache_key(
        "atomic-v4", static_prefix="stable rubric", routing_identity="EVIDENCE-1"
    )
    routed = {
        _prompt_cache_key(
            "atomic-v4",
            static_prefix="stable rubric",
            routing_identity=f"EVIDENCE-{index}",
        )
        for index in range(128)
    }

    assert first == repeated
    assert 1 < len(routed) <= 32
    assert all(key.startswith("moatrader:atomic-v4:") for key in routed)


def test_openai_transport_uses_responses_create_and_usage() -> None:
    captured: list[dict[str, object]] = []

    class FakeResponses:
        def create(self, **kwargs: object) -> object:
            captured.append(kwargs)
            return SimpleNamespace(
                id="resp_fixture",
                model=f"{kwargs['model']}-fixture",
                output_text='{"chunk_id":"C1","cards":[]}',
                usage=SimpleNamespace(
                    input_tokens=10,
                    output_tokens=3,
                    input_tokens_details=SimpleNamespace(cached_tokens=4, cache_write_tokens=2),
                ),
            )

    client = SimpleNamespace(responses=FakeResponses())
    request = LLMRequest(
        task=LLMTask.LOCAL_EVIDENCE_EXTRACTION,
        system="system",
        user="user",
        response_schema=EvidenceExtractionResult.model_json_schema(),
        input_sha256="0" * 64,
        prompt_cache_key="moatrader:test",
        prompt_cache_breakpoint=True,
    )
    transport = OpenAIResponsesTransport(client=client)

    result = transport.execute(request, EvidenceExtractionResult)

    assert captured[0]["model"] == "gpt-5.6-luna"
    assert captured[0]["text"]["format"]["type"] == "json_schema"  # type: ignore[index]
    assert captured[0]["text"]["verbosity"] == "low"  # type: ignore[index]
    assert captured[0]["reasoning"] == {"effort": "medium"}
    assert captured[0]["max_output_tokens"] == 2_000
    assert captured[0]["prompt_cache_key"] == "moatrader:test"
    assert captured[0]["prompt_cache_options"] == {"mode": "explicit", "ttl": "30m"}
    assert captured[0]["input"][0]["content"][0]["prompt_cache_breakpoint"] == {  # type: ignore[index]
        "mode": "explicit"
    }
    assert captured[0]["input"][1]["content"][0]["text"] == "user"  # type: ignore[index]
    assert captured[0]["store"] is False
    assert result.response_id == "resp_fixture"
    assert result.model == "gpt-5.6-luna-fixture"
    assert result.usage.input_tokens == 10
    assert result.usage.cached_input_tokens == 4
    assert result.usage.cache_write_tokens == 2

    moat_request = request.model_copy(update={"task": LLMTask.FINAL_MOAT_SCORING})
    moat_result = transport.execute(moat_request, EvidenceExtractionResult)

    assert captured[1]["model"] == "gpt-5.6-luna"
    assert captured[1]["reasoning"] == {"effort": "medium"}
    assert captured[1]["max_output_tokens"] == 4_000
    assert moat_result.model == "gpt-5.6-luna-fixture"


def test_openai_transport_legacy_model_alias_routes_every_task_to_one_model() -> None:
    captured: list[str] = []

    class FakeResponses:
        def create(self, **kwargs: object) -> object:
            captured.append(str(kwargs["model"]))
            return SimpleNamespace(
                id="resp_fixture",
                model=kwargs["model"],
                output_text='{"chunk_id":"C1","cards":[]}',
                usage=None,
            )

    request = LLMRequest(
        task=LLMTask.SECTION_SUMMARY,
        system="system",
        user="user",
        response_schema=EvidenceExtractionResult.model_json_schema(),
        input_sha256="0" * 64,
    )
    transport = OpenAIResponsesTransport(client=SimpleNamespace(responses=FakeResponses()), model="fixture-model")

    transport.execute(request, EvidenceExtractionResult)

    assert captured == ["fixture-model"]


def test_contextual_strength_routes_to_luna_with_full_quality_budget() -> None:
    captured: list[dict[str, object]] = []

    class FakeResponses:
        def create(self, **kwargs: object) -> object:
            captured.append(kwargs)
            return SimpleNamespace(
                id="resp_strength",
                model=kwargs["model"],
                output_text='{"evidence_sufficiency":0,"mechanisms":[],"outcome_confirmation":[],"counterevidence":[],"llm_proposed_score":null}',
                usage=None,
            )

    request = LLMRequest(
        task=LLMTask.CONTEXTUAL_MOAT_STRENGTH,
        system="strength rubric",
        user="broad context",
        response_schema=ContextualMoatAssessment.model_json_schema(),
        input_sha256="1" * 64,
    )
    result = OpenAIResponsesTransport(
        client=SimpleNamespace(responses=FakeResponses())
    ).execute(request, ContextualMoatAssessment)

    assert captured[0]["model"] == "gpt-5.6-luna"
    assert captured[0]["reasoning"] == {"effort": "medium"}
    assert captured[0]["max_output_tokens"] == 8_000
    assert result.parsed.mechanisms == []


def test_openai_transport_recovers_literal_control_character_in_json_string() -> None:
    class FakeResponses:
        def create(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                id="resp_fixture",
                model="fixture",
                output_text='{"chunk_id":"C1","cards":[],"ignored":"line\nbreak"}',
                usage=None,
            )

    # Use a permissive tiny fixture model to isolate tolerant JSON decoding.
    from pydantic import BaseModel, ConfigDict

    class Fixture(BaseModel):
        model_config = ConfigDict(extra="ignore")
        chunk_id: str
        cards: list[object]

    request = LLMRequest(
        task=LLMTask.LOCAL_EVIDENCE_EXTRACTION,
        system="system",
        user="user",
        response_schema=Fixture.model_json_schema(),
        input_sha256="0" * 64,
    )
    result = OpenAIResponsesTransport(client=SimpleNamespace(responses=FakeResponses())).execute(request, Fixture)

    assert result.parsed.chunk_id == "C1"


def test_openai_transport_restores_single_chunk_grounding_fields() -> None:
    class FakeResponses:
        def create(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                id="resp_fixture",
                model="fixture",
                output_text='{"cards":[]}',
                usage=None,
            )

    request = LLMRequest(
        task=LLMTask.LOCAL_EVIDENCE_EXTRACTION,
        system="system",
        user="user",
        response_schema=EvidenceExtractionResult.model_json_schema(),
        input_sha256="0" * 64,
        metadata={"chunk_id": "C1", "node_ids": ["N1"], "source_type": "DART"},
    )
    result = OpenAIResponsesTransport(client=SimpleNamespace(responses=FakeResponses())).execute(
        request, EvidenceExtractionResult
    )

    assert result.parsed.chunk_id == "C1"


def test_openai_transport_overrides_wrong_single_chunk_id() -> None:
    request = LLMRequest(
        task=LLMTask.LOCAL_EVIDENCE_EXTRACTION,
        system="system",
        user="user",
        response_schema=EvidenceExtractionResult.model_json_schema(),
        input_sha256="0" * 64,
        metadata={"chunk_id": "C1", "node_ids": ["N1"], "source_type": "DART"},
    )
    payload = {
        "chunk_id": "provider-invented-id",
        "cards": [
            {
                "evidence_id": "E1",
                "source_chunk_id": "N1",
                "node_ids": ["N1"],
                "fact": "Grounded fact",
            }
        ],
    }

    normalized = OpenAIResponsesTransport._normalize_grounding_fields(
        request, EvidenceExtractionResult, payload
    )
    result = EvidenceExtractionResult.model_validate(normalized)

    assert result.chunk_id == "C1"
    assert result.cards[0].source_chunk_id == "C1"


def test_openai_transport_flattens_list_wrapped_evidence_batch() -> None:
    from moatrader.evidence.models import EvidenceBatchExtractionResult

    request = LLMRequest(
        task=LLMTask.LOCAL_EVIDENCE_EXTRACTION,
        system="system",
        user="user",
        response_schema=EvidenceBatchExtractionResult.model_json_schema(),
        input_sha256="0" * 64,
        metadata={"node_ids_by_chunk": {"C1": ["N1"]}},
    )
    payload = [
        {
            "cards": [
                {
                    "evidence_id": "E1",
                    "source_chunk_id": "C1",
                    "node_ids": ["N1"],
                    "fact": "Grounded fact",
                }
            ]
        },
        {"error": "irrelevant provider note"},
    ]

    normalized = OpenAIResponsesTransport._normalize_grounding_fields(
        request, EvidenceBatchExtractionResult, payload
    )
    result = EvidenceBatchExtractionResult.model_validate(normalized)

    assert len(result.cards) == 1
    assert result.cards[0].source_chunk_id == "C1"


def test_openai_transport_drops_an_ungrounded_batch_card() -> None:
    from moatrader.evidence.models import EvidenceBatchExtractionResult

    request = LLMRequest(
        task=LLMTask.LOCAL_EVIDENCE_EXTRACTION,
        system="system",
        user="user",
        response_schema=EvidenceBatchExtractionResult.model_json_schema(),
        input_sha256="0" * 64,
        metadata={"node_ids_by_chunk": {"C1": ["N1"]}},
    )
    payload = {
        "cards": [
            {
                "evidence_id": "E1",
                "node_ids": ["UNKNOWN_NODE"],
                "fact": "Cannot be grounded to a requested chunk.",
            }
        ]
    }

    normalized = OpenAIResponsesTransport._normalize_grounding_fields(
        request, EvidenceBatchExtractionResult, payload
    )
    result = EvidenceBatchExtractionResult.model_validate(normalized)

    assert result.cards == []


def test_openai_transport_treats_blank_evidence_payload_as_no_cards() -> None:
    from moatrader.evidence.models import EvidenceBatchExtractionResult

    request = LLMRequest(
        task=LLMTask.LOCAL_EVIDENCE_EXTRACTION,
        system="system",
        user="user",
        response_schema=EvidenceBatchExtractionResult.model_json_schema(),
        input_sha256="0" * 64,
        metadata={"node_ids_by_chunk": {"C1": ["N1"]}},
    )

    normalized = OpenAIResponsesTransport._normalize_grounding_fields(
        request, EvidenceBatchExtractionResult, ""
    )
    result = EvidenceBatchExtractionResult.model_validate(normalized)

    assert result.cards == []


def test_openai_transport_ignores_trailing_json_after_first_structured_object() -> None:
    class FakeResponses:
        def create(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                id="resp_fixture",
                model="fixture",
                output_text='{"chunk_id":"C1","cards":[]} {"debug":true}',
                usage=None,
            )

    request = LLMRequest(
        task=LLMTask.LOCAL_EVIDENCE_EXTRACTION,
        system="system",
        user="user",
        response_schema=EvidenceExtractionResult.model_json_schema(),
        input_sha256="0" * 64,
        metadata={"chunk_id": "C1", "node_ids": ["N1"], "source_type": "DART"},
    )
    result = OpenAIResponsesTransport(client=SimpleNamespace(responses=FakeResponses())).execute(
        request, EvidenceExtractionResult
    )

    assert result.parsed.cards == []


def test_openai_transport_repairs_missing_comma_between_fields() -> None:
    class FakeResponses:
        def create(self, **_kwargs: object) -> object:
            return SimpleNamespace(
                id="resp_fixture",
                model="fixture",
                output_text='{"chunk_id":"C1" "cards":[]}',
                usage=None,
            )

    request = LLMRequest(
        task=LLMTask.LOCAL_EVIDENCE_EXTRACTION,
        system="system",
        user="user",
        response_schema=EvidenceExtractionResult.model_json_schema(),
        input_sha256="0" * 64,
        metadata={"chunk_id": "C1", "node_ids": ["N1"], "source_type": "DART"},
    )
    result = OpenAIResponsesTransport(client=SimpleNamespace(responses=FakeResponses())).execute(
        request, EvidenceExtractionResult
    )

    assert result.parsed.chunk_id == "C1"


def test_openai_transport_supplies_conservative_missing_moat_metadata() -> None:
    payload = {
        "economic_moat_score": 4,
        "mechanisms": [
            {
                "evidence_type": "SWITCHING_COST",
                "score": 4,
                "evidence_ids": ["E1"],
                "rationale": "Grounded rationale",
            }
        ],
    }
    request = LLMRequest(
        task=LLMTask.FINAL_MOAT_SCORING,
        system="system",
        user="user",
        response_schema=MoatScore.model_json_schema(),
        input_sha256="0" * 64,
        metadata={"issuer_id": "059100", "as_of": "2025-08-01"},
    )

    normalized = OpenAIResponsesTransport._normalize_grounding_fields(request, MoatScore, payload)
    result = MoatScore.model_validate(normalized)

    assert result.issuer_id == "059100"
    assert result.durability.value == "MEDIUM"
    assert result.model_confidence == 0.5
    assert result.document_coverage.model_dump(exclude_none=True) == {}


def test_openai_transport_merges_duplicate_moat_mechanism_types() -> None:
    payload = {
        "economic_moat_score": 6,
        "mechanisms": [
            {
                "evidence_type": "INTANGIBLE_ASSET",
                "score": 4,
                "evidence_ids": ["E1"],
                "rationale": "First claim",
            },
            {
                "evidence_type": "INTANGIBLE_ASSET",
                "score": 6,
                "evidence_ids": ["E2", "E1"],
                "rationale": "Stronger claim",
            },
        ],
    }
    request = LLMRequest(
        task=LLMTask.FINAL_MOAT_SCORING,
        system="system",
        user="user",
        response_schema=MoatScore.model_json_schema(),
        input_sha256="0" * 64,
        metadata={"issuer_id": "058730", "as_of": "2025-08-01"},
    )

    normalized = OpenAIResponsesTransport._normalize_grounding_fields(request, MoatScore, payload)
    result = MoatScore.model_validate(normalized)

    assert len(result.mechanisms) == 1
    assert result.mechanisms[0].score == 6
    assert result.mechanisms[0].evidence_ids == ["E1", "E2"]
    assert result.mechanisms[0].rationale == "Stronger claim"


def test_openai_transport_converts_an_empty_moat_score_to_zero() -> None:
    payload = {"economic_moat_score": "", "mechanisms": []}
    request = LLMRequest(
        task=LLMTask.FINAL_MOAT_SCORING,
        system="system",
        user="user",
        response_schema=MoatScore.model_json_schema(),
        input_sha256="0" * 64,
        metadata={"issuer_id": "009180", "as_of": "2025-08-01"},
    )

    normalized = OpenAIResponsesTransport._normalize_grounding_fields(request, MoatScore, payload)
    result = MoatScore.model_validate(normalized)

    assert result.economic_moat_score == 0
    assert result.durability.value == "LOW"


def test_openai_transport_zeroes_a_positive_score_without_cited_mechanisms() -> None:
    payload = {"economic_moat_score": 6, "mechanisms": [], "durability": "HIGH"}
    request = LLMRequest(
        task=LLMTask.FINAL_MOAT_SCORING,
        system="system",
        user="user",
        response_schema=MoatScore.model_json_schema(),
        input_sha256="0" * 64,
        metadata={"issuer_id": "014970", "as_of": "2025-08-01"},
    )

    normalized = OpenAIResponsesTransport._normalize_grounding_fields(request, MoatScore, payload)
    result = MoatScore.model_validate(normalized)

    assert result.economic_moat_score == 0
    assert result.durability.value == "LOW"
