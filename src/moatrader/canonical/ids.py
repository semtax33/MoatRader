from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any


_SPACE_RE = re.compile(r"[ \t\r\f\v]+")


def normalize_text(value: str) -> str:
    """Normalize layout noise without destroying intentional line boundaries."""
    lines = [_SPACE_RE.sub(" ", line).strip() for line in value.replace("\u00a0", " ").split("\n")]
    compact: list[str] = []
    for line in lines:
        if line or (compact and compact[-1]):
            compact.append(line)
    return "\n".join(compact).strip()


def content_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def stable_id(prefix: str, *parts: Any, length: int = 20) -> str:
    """Return a reproducible ID from canonical JSON-encoded identity parts."""
    encoded = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def canonical_mapping_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

