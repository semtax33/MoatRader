from __future__ import annotations

import os
import time

from moatrader.evidence.models import AtomicEvidenceExtraction
from moatrader.llm.contracts import build_atomic_evidence_request
from moatrader.llm.replay import LLMReplayCache
from moatrader.evidence.atomic import build_atomic_evidence_units
from moatrader.semantic.chunker import SemanticChunk


def test_replay_cache_reclaims_a_timed_out_orphan_lock(tmp_path) -> None:
    source = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown="Customer qualification takes 18 months.",
        token_count=5,
    )
    unit = build_atomic_evidence_units([source], issuer_id="ISSUER-1")[0]
    request = build_atomic_evidence_request(unit, issuer_id="ISSUER-1")
    cache = LLMReplayCache(
        tmp_path,
        experiment_id="experiment",
        summary_model="gpt-5-nano",
        moat_model="gpt-5.6-luna",
        summary_reasoning_effort="low",
        atomic_reasoning_effort="low",
        moat_reasoning_effort="medium",
        engine_version="0.8.0",
        lock_timeout_seconds=0.01,
    )
    key, _ = cache.identity(request, AtomicEvidenceExtraction)
    lock_path = cache._record_path(request.task, key).with_suffix(".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"")
    old = time.time() - 1.0
    os.utime(lock_path, (old, old))

    with cache.locked(request, AtomicEvidenceExtraction) as locked_key:
        assert locked_key == key
        assert lock_path.is_file()

    assert not lock_path.exists()
    assert not lock_path.with_name(f"{lock_path.name}.reclaim").exists()
