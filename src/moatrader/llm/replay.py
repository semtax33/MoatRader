from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel

from moatrader.llm.contracts import LLMRequest, LLMTask
from moatrader.llm.transport import TransportResult, TransportUsage


ResponseT = TypeVar("ResponseT", bound=BaseModel)
REPLAY_SCHEMA_VERSION = "moatrader-llm-replay/2"
NORMALIZATION_VERSION = "openai-structured-grounding/1"


class LLMReplayCache:
    """Experiment-scoped, content-addressed cache for validated LLM responses.

    A fresh experiment gets a fresh namespace. Within that namespace an exact
    request contract is evaluated once and replayed everywhere else, including
    later PIT dates that contain the same disclosure chunk. The cache key
    includes the requested model, reasoning effort, response schema, and engine
    version so implementation changes cannot silently reuse stale annotations.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        experiment_id: str,
        summary_model: str,
        moat_model: str,
        summary_reasoning_effort: str,
        moat_reasoning_effort: str,
        engine_version: str,
        lock_timeout_seconds: float = 600.0,
    ) -> None:
        self.root = Path(root).resolve() / experiment_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.experiment_id = experiment_id
        self.summary_model = summary_model
        self.moat_model = moat_model
        self.summary_reasoning_effort = summary_reasoning_effort
        self.moat_reasoning_effort = moat_reasoning_effort
        self.engine_version = engine_version
        self.lock_timeout_seconds = lock_timeout_seconds
        self._locks_guard = Lock()
        self._locks: dict[str, Lock] = {}

    def identity(self, request: LLMRequest, response_model: type[ResponseT]) -> tuple[str, dict[str, Any]]:
        schema = json.dumps(
            response_model.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        atomic_key = (
            str(request.metadata.get("atomic_evidence_key"))
            if request.task == LLMTask.LOCAL_EVIDENCE_EXTRACTION
            and request.metadata.get("atomic_evidence_key")
            else None
        )
        payload = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
            "engine_version": self.engine_version,
            "experiment_id": self.experiment_id,
            "task": request.task.value,
            "identity_mode": "ATOMIC_EVIDENCE" if atomic_key else "FULL_PROMPT",
            "input_sha256": atomic_key or request.input_sha256,
            "prompt_version": request.metadata.get("prompt_version"),
            "rubric_version": request.metadata.get("rubric_version"),
            "model": self._model_for(request.task),
            "reasoning_effort": self._effort_for(request.task),
            "temperature": request.temperature,
            "response_model": response_model.__name__,
            "response_schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest(), payload

    def load(
        self,
        request: LLMRequest,
        response_model: type[ResponseT],
    ) -> tuple[str, TransportResult[ResponseT] | None]:
        key, identity = self.identity(request, response_model)
        path = self._record_path(request.task, key)
        if not path.is_file():
            return key, None
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("identity") != identity or record.get("cache_key") != key:
                return key, None
            parsed = response_model.model_validate(record["normalized_output"])
            return key, TransportResult[ResponseT](
                parsed=parsed,
                provider=str(record.get("provider") or "replay"),
                model=str(record["model"]),
                response_id=record.get("response_id"),
                raw_output_text=record.get("raw_output_text"),
                usage=TransportUsage(),
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return key, None

    def store(
        self,
        request: LLMRequest,
        response_model: type[ResponseT],
        result: TransportResult[ResponseT],
    ) -> str:
        key, identity = self.identity(request, response_model)
        path = self._record_path(request.task, key)
        record = {
            "schema_version": REPLAY_SCHEMA_VERSION,
            "cache_key": key,
            "identity": identity,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": result.provider,
            "model": result.model,
            "response_id": result.response_id,
            "raw_output_text": result.raw_output_text,
            "normalized_output": result.parsed.model_dump(mode="json", exclude_none=True),
        }
        content = json.dumps(record, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
        return key

    def discard(self, task: LLMTask, cache_key: str) -> None:
        path = self._record_path(task, cache_key)
        if path.is_file():
            path.unlink()

    @contextmanager
    def locked(self, request: LLMRequest, response_model: type[ResponseT]) -> Iterator[str]:
        cache_key, _ = self.identity(request, response_model)
        with self._locks_guard:
            local_lock = self._locks.setdefault(cache_key, Lock())
        with local_lock:
            lock_path = self._record_path(request.task, cache_key).with_suffix(".lock")
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            started = time.monotonic()
            while True:
                try:
                    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                    os.close(descriptor)
                    break
                except FileExistsError:
                    if time.monotonic() - started > self.lock_timeout_seconds:
                        raise TimeoutError(f"timed out waiting for LLM replay lock: {lock_path}")
                    time.sleep(0.1)
            try:
                yield cache_key
            finally:
                lock_path.unlink(missing_ok=True)

    def _record_path(self, task: LLMTask, cache_key: str) -> Path:
        return self.root / task.value.lower() / cache_key[:2] / f"{cache_key}.json"

    def _model_for(self, task: LLMTask) -> str:
        if task in {LLMTask.LOCAL_EVIDENCE_EXTRACTION, LLMTask.FINAL_MOAT_SCORING}:
            return self.moat_model
        return self.summary_model

    def _effort_for(self, task: LLMTask) -> str:
        if task in {LLMTask.LOCAL_EVIDENCE_EXTRACTION, LLMTask.FINAL_MOAT_SCORING}:
            return self.moat_reasoning_effort
        return self.summary_reasoning_effort
