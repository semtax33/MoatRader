from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from typing import Any, Generic, Protocol, TypeVar

from pydantic import BaseModel, Field

from moatrader.canonical.models import ContractModel
from moatrader.llm.contracts import LLMRequest, LLMTask


ResponseT = TypeVar("ResponseT", bound=BaseModel)


_UNSUPPORTED_OPENAI_REGEX_TOKENS = ("(?=", "(?!", "(?<=", "(?<!")


def _openai_compatible_schema(value: Any) -> Any:
    """Drop provider-unsupported regex assertions from a strict JSON schema.

    Pydantic's Decimal schema uses a negative lookahead. OpenAI Structured
    Outputs accepts ``pattern`` but rejects lookaround syntax. The response is
    still validated against the original Pydantic model after decoding, so
    removing only that provider-incompatible presentation constraint does not
    weaken local semantic validation.
    """

    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if (
                key == "pattern"
                and isinstance(item, str)
                and any(token in item for token in _UNSUPPORTED_OPENAI_REGEX_TOKENS)
            ):
                continue
            result[key] = _openai_compatible_schema(item)
        return result
    if isinstance(value, list):
        return [_openai_compatible_schema(item) for item in value]
    return value


def _canonical_json_object(value: Any) -> Any:
    """Return a recursively key-sorted JSON value for byte-stable API input."""

    if isinstance(value, dict):
        return {key: _canonical_json_object(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_json_object(item) for item in value]
    return value


class TransportUsage(ContractModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_input_tokens: int = Field(default=0, ge=0)
    cache_write_tokens: int = Field(default=0, ge=0)


class TransportResult(ContractModel, Generic[ResponseT]):
    parsed: ResponseT
    provider: str
    model: str
    response_id: str | None = None
    usage: TransportUsage = Field(default_factory=TransportUsage)
    raw_output_text: str | None = None


class LLMTransport(Protocol):
    def execute(self, request: LLMRequest, response_model: type[ResponseT]) -> TransportResult[ResponseT]: ...


class OpenAIResponsesTransport:
    """OpenAI Responses API transport with typed Pydantic Structured Outputs."""

    def __init__(
        self,
        *,
        summary_model: str = "gpt-5-nano",
        moat_model: str = "gpt-5.6-luna",
        summary_reasoning_effort: str = "low",
        atomic_reasoning_effort: str = "medium",
        moat_reasoning_effort: str = "medium",
        max_output_tokens: int = 8_000,
        max_retries: int = 4,
        timeout_seconds: float = 180.0,
        client: Any | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        # ``model`` and ``reasoning_effort`` remain compatibility aliases for
        # callers that intentionally want one model/effort for every task.
        if model is not None:
            summary_model = model
            moat_model = model
        if reasoning_effort is not None:
            summary_reasoning_effort = reasoning_effort
            atomic_reasoning_effort = reasoning_effort
            moat_reasoning_effort = reasoning_effort
        self.summary_model = summary_model
        self.moat_model = moat_model
        self.summary_reasoning_effort = summary_reasoning_effort
        self.atomic_reasoning_effort = atomic_reasoning_effort
        self.moat_reasoning_effort = moat_reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.client = client

    def _client(self) -> Any:
        """Initialize the SDK only when a cache miss needs a live request.

        A fully populated replay cache is intentionally usable offline and
        without exposing an API credential to a reproducibility run.
        """

        if self.client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError('OpenAI support is optional; install with: pip install -e ".[llm]"') from exc
            try:
                self.client = OpenAI(timeout=self.timeout_seconds, max_retries=0)
            except Exception as exc:
                raise RuntimeError(
                    "failed to initialize OpenAI client; verify OPENAI_API_KEY and client configuration"
                ) from exc
        return self.client

    def execute(self, request: LLMRequest, response_model: type[ResponseT]) -> TransportResult[ResponseT]:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                request_model = self._model_for(request.task)
                try:
                    from openai.lib._pydantic import to_strict_json_schema
                except ImportError as exc:
                    raise RuntimeError("installed OpenAI SDK cannot build a strict response schema") from exc
                static_content: dict[str, Any] = {
                    "type": "input_text",
                    "text": request.system,
                }
                if request_model.startswith("gpt-5.6") and request.prompt_cache_breakpoint:
                    static_content["prompt_cache_breakpoint"] = {"mode": "explicit"}
                create_kwargs: dict[str, Any] = dict(
                    model=request_model,
                    input=[
                        {"role": "system", "content": [static_content]},
                        {
                            "role": "user",
                            "content": [{"type": "input_text", "text": request.user}],
                        },
                    ],
                    text={
                        "verbosity": "low",
                        "format": {
                            "type": "json_schema",
                            "name": response_model.__name__,
                            "strict": True,
                            "schema": _canonical_json_object(
                                _openai_compatible_schema(
                                    to_strict_json_schema(response_model)
                                )
                            ),
                        }
                    },
                    reasoning={"effort": self._effort_for(request.task)},
                    max_output_tokens=self._max_output_tokens_for(request.task),
                    store=False,
                )
                if request.prompt_cache_key:
                    create_kwargs["prompt_cache_key"] = request.prompt_cache_key
                if request_model.startswith("gpt-5.6"):
                    # Explicit-only mode prevents one-off company/source
                    # suffixes from becoming paid cache writes. Prefixes below
                    # GPT-5.6's 1,024-token minimum simply remain uncached.
                    cache_options: dict[str, str] = {"mode": "explicit"}
                    if request.prompt_cache_breakpoint:
                        cache_options["ttl"] = request.prompt_cache_ttl
                    create_kwargs["prompt_cache_options"] = cache_options
                response = self._client().responses.create(**create_kwargs)
                output_text = str(getattr(response, "output_text", "") or "")
                if not output_text:
                    refusal = self._refusal_text(response)
                    raise RuntimeError(f"model returned no parsed output{': ' + refusal if refusal else ''}")
                # Some nano responses contain a literal newline/control
                # character inside an otherwise valid JSON string. Python's
                # non-strict decoder recovers that representation; Pydantic
                # still performs the full semantic schema validation after.
                candidate = output_text.strip()
                if candidate.startswith("```"):
                    candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
                    candidate = re.sub(r"\s*```$", "", candidate)
                payload = self._decode_json_object(candidate)
                parsed = response_model.model_validate(
                    self._normalize_grounding_fields(request, response_model, payload)
                )
                usage = getattr(response, "usage", None)
                input_details = getattr(usage, "input_tokens_details", None) if usage else None
                return TransportResult[ResponseT](
                    parsed=parsed,
                    provider="openai",
                    model=str(getattr(response, "model", None) or request_model),
                    response_id=getattr(response, "id", None),
                    raw_output_text=output_text,
                    usage=TransportUsage(
                        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
                        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
                        cached_input_tokens=int(getattr(input_details, "cached_tokens", 0) or 0),
                        cache_write_tokens=int(getattr(input_details, "cache_write_tokens", 0) or 0),
                    ),
                )
            except Exception as exc:  # SDK exceptions are optional at import time.
                last_error = exc
                if attempt >= self.max_retries or not self._retryable(exc):
                    break
                time.sleep(min(30.0, 1.5 * (2**attempt)))
        assert last_error is not None
        raise RuntimeError(f"OpenAI request failed after {self.max_retries + 1} attempt(s): {last_error}") from last_error

    def _model_for(self, task: LLMTask) -> str:
        if task in {
            LLMTask.LOCAL_EVIDENCE_EXTRACTION,
            LLMTask.CONTEXTUAL_MOAT_STRENGTH,
            LLMTask.IR_INCREMENTAL_ASSESSMENT,
            LLMTask.CANDIDATE_ATOMIC_AUDIT,
            LLMTask.FINAL_MOAT_SCORING,
        }:
            return self.moat_model
        return self.summary_model

    @staticmethod
    def _normalize_grounding_fields(
        request: LLMRequest,
        response_model: type[ResponseT],
        payload: Any,
    ) -> Any:
        if (
            isinstance(payload, str)
            and not payload.strip()
            and response_model.__name__ in {"EvidenceExtractionResult", "EvidenceBatchExtractionResult"}
        ):
            payload = {"cards": []}
        if isinstance(payload, list) and response_model.__name__ in {
            "EvidenceExtractionResult",
            "EvidenceBatchExtractionResult",
        }:
            cards = []
            for item in payload:
                if not isinstance(item, dict):
                    continue
                nested = item.get("cards")
                if isinstance(nested, list):
                    cards.extend(nested)
                elif item.get("fact"):
                    cards.append(item)
            payload = {"cards": cards}
        if not isinstance(payload, dict):
            return payload
        if response_model.__name__ == "MoatScore":
            payload.setdefault("issuer_id", request.metadata.get("issuer_id"))
            payload.setdefault("as_of", request.metadata.get("as_of"))
            mechanisms = payload.get("mechanisms")
            if isinstance(mechanisms, list):
                deduplicated: list[Any] = []
                by_type: dict[str, dict[str, Any]] = {}
                for mechanism in mechanisms:
                    if not isinstance(mechanism, dict) or not mechanism.get("evidence_type"):
                        deduplicated.append(mechanism)
                        continue
                    key = str(mechanism["evidence_type"]).strip().upper()
                    current = by_type.get(key)
                    if current is None:
                        current = dict(mechanism)
                        current["evidence_ids"] = list(dict.fromkeys(current.get("evidence_ids") or []))
                        by_type[key] = current
                        deduplicated.append(current)
                        continue
                    current["evidence_ids"] = list(
                        dict.fromkeys([*(current.get("evidence_ids") or []), *(mechanism.get("evidence_ids") or [])])
                    )
                    try:
                        stronger = float(mechanism.get("score", 0)) > float(current.get("score", 0))
                    except (TypeError, ValueError):
                        stronger = False
                    if stronger:
                        current["score"] = mechanism.get("score")
                        if mechanism.get("rationale"):
                            current["rationale"] = mechanism["rationale"]
                payload["mechanisms"] = deduplicated
            for alias in ("moat_score", "overall_score", "score"):
                if "economic_moat_score" not in payload and alias in payload:
                    payload["economic_moat_score"] = payload[alias]
            score = payload.get("economic_moat_score", 0)
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                numeric_score = 0.0
                payload["economic_moat_score"] = numeric_score
            if numeric_score > 0 and not payload.get("mechanisms"):
                numeric_score = 0.0
                payload["economic_moat_score"] = numeric_score
                payload["durability"] = "LOW"
            if not payload.get("durability"):
                payload["durability"] = (
                    "HIGH"
                    if numeric_score >= 8
                    else "MEDIUM_HIGH"
                    if numeric_score >= 6
                    else "MEDIUM"
                    if numeric_score >= 3
                    else "LOW"
                )
            elif isinstance(payload["durability"], str):
                payload["durability"] = payload["durability"].strip().upper().replace("-", "_").replace(" ", "_")
            for alias in ("confidence", "confidence_score"):
                if "model_confidence" not in payload and alias in payload:
                    payload["model_confidence"] = payload[alias]
            confidence = payload.setdefault("model_confidence", 0.5)
            try:
                if float(confidence) > 1:
                    payload["model_confidence"] = float(confidence) / 100
            except (TypeError, ValueError):
                payload["model_confidence"] = 0.5
            coverage = payload.get("document_coverage", payload.get("coverage", {}))
            payload["document_coverage"] = coverage if isinstance(coverage, dict) else {}
            return payload
        if response_model.__name__ not in {
            "EvidenceExtractionResult",
            "EvidenceBatchExtractionResult",
        }:
            return payload
        cards = payload.get("cards")
        if not isinstance(cards, list):
            return payload
        single_chunk_id = request.metadata.get("chunk_id")
        if response_model.__name__ == "EvidenceExtractionResult" and isinstance(single_chunk_id, str):
            payload["chunk_id"] = single_chunk_id
        nodes_by_chunk = request.metadata.get("node_ids_by_chunk")
        aliases = {
            "evidenceid": "evidence_id",
            "sourcechunkid": "source_chunk_id",
            "nodeids": "node_ids",
            "evidencetype": "evidence_type",
            "statementtype": "statement_type",
            "sourcetype": "source_type",
            "companyscope": "company_scope",
            "rawquote": "raw_quote",
        }
        normalized_cards = []
        for index, card in enumerate(cards):
            if not isinstance(card, dict):
                continue
            for supplied_key in list(card):
                canonical = aliases.get(re.sub(r"[^a-z0-9]", "", supplied_key.casefold()))
                if canonical and canonical not in card:
                    card[canonical] = card[supplied_key]
                if canonical and supplied_key != canonical:
                    card.pop(supplied_key, None)
            if isinstance(single_chunk_id, str):
                # A single-chunk request has only one valid source. Override a
                # provider-invented/node ID deterministically.
                card["source_chunk_id"] = single_chunk_id
                card["source_type"] = request.metadata.get("source_type", "OTHER")
            elif isinstance(nodes_by_chunk, dict) and card.get("source_chunk_id") not in nodes_by_chunk:
                cited_nodes = set(card.get("node_ids") or [])
                matches = [
                    chunk_id
                    for chunk_id, node_ids in nodes_by_chunk.items()
                    if cited_nodes and cited_nodes <= set(node_ids or [])
                ]
                if len(matches) == 1:
                    card["source_chunk_id"] = matches[0]
            card.setdefault("evidence_id", f"pending-{index}")
            source_chunk_id = card.get("source_chunk_id")
            has_grounded_source = isinstance(source_chunk_id, str) and (
                not isinstance(nodes_by_chunk, dict) or source_chunk_id in nodes_by_chunk
            )
            if (
                card.get("fact")
                and isinstance(card.get("node_ids"), list)
                and card["node_ids"]
                and (isinstance(single_chunk_id, str) or has_grounded_source)
            ):
                normalized_cards.append(card)
        payload["cards"] = normalized_cards
        return payload

    @staticmethod
    def _decode_json_object(candidate: str) -> Any:
        decoder = json.JSONDecoder(strict=False)
        try:
            payload, _end = decoder.raw_decode(candidate)
            return payload
        except json.JSONDecodeError:
            repaired = re.sub(
                r'(?<=[}\]"0-9])\s*(?="(?:[^"\\]|\\.)*"\s*:)',
                ",",
                candidate,
            )
            repaired = re.sub(r"}\s*{", "},{", repaired)
            try:
                payload, _end = decoder.raw_decode(repaired)
                return payload
            except json.JSONDecodeError:
                from json_repair import repair_json

                return repair_json(candidate, return_objects=True)

    def _effort_for(self, task: LLMTask) -> str:
        if task == LLMTask.LOCAL_EVIDENCE_EXTRACTION:
            return self.atomic_reasoning_effort
        if task in {
            LLMTask.CONTEXTUAL_MOAT_STRENGTH,
            LLMTask.CANDIDATE_ATOMIC_AUDIT,
            LLMTask.FINAL_MOAT_SCORING,
        }:
            return self.moat_reasoning_effort
        return self.summary_reasoning_effort

    def _max_output_tokens_for(self, task: LLMTask) -> int:
        # Atomic classification is one source unit and should never consume a
        # company-level answer budget. Caps prevent malformed verbose outputs
        # while the configured global maximum remains a compatibility ceiling.
        if task == LLMTask.LOCAL_EVIDENCE_EXTRACTION:
            return min(self.max_output_tokens, 2_000)
        if task in {
            LLMTask.CONTEXTUAL_MOAT_STRENGTH,
            LLMTask.IR_INCREMENTAL_ASSESSMENT,
        }:
            return min(self.max_output_tokens, 8_000)
        if task == LLMTask.CANDIDATE_ATOMIC_AUDIT:
            return min(self.max_output_tokens, 2_000)
        if task == LLMTask.SECTION_SUMMARY:
            return min(self.max_output_tokens, 3_000)
        return min(self.max_output_tokens, 4_000)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        if isinstance(exc, (TypeError, ValueError)):
            return False
        status = getattr(exc, "status_code", None)
        if status is None:
            return True
        try:
            code = int(status)
        except (TypeError, ValueError):
            return False
        return code in {408, 409, 429} or code >= 500

    @staticmethod
    def _refusal_text(response: Any) -> str | None:
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                refusal = getattr(content, "refusal", None)
                if refusal:
                    return str(refusal)
        return None


class FunctionTransport:
    """Deterministic injectable transport for tests, replays, and local evals."""

    def __init__(self, handler: Callable[[LLMRequest, type[BaseModel]], BaseModel], model: str = "fixture") -> None:
        self.handler = handler
        self.model = model

    def execute(self, request: LLMRequest, response_model: type[ResponseT]) -> TransportResult[ResponseT]:
        value = self.handler(request, response_model)
        parsed = response_model.model_validate(value)
        raw_output_text = parsed.model_dump_json(exclude_none=True)
        return TransportResult[ResponseT](
            parsed=parsed,
            provider="function",
            model=self.model,
            raw_output_text=raw_output_text,
        )
