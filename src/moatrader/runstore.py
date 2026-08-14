from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class RunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def company_dir(self, ticker: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,31}", ticker):
            raise ValueError(f"unsafe ticker for artifact directory: {ticker!r}")
        path = self.root / "companies" / ticker
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_text(self, path: Path, value: str) -> None:
        self._atomic_write(path, value.encode("utf-8"))

    def write_json(self, path: Path, value: BaseModel | dict[str, Any] | list[Any]) -> None:
        if isinstance(value, BaseModel):
            text = value.model_dump_json(indent=2, exclude_none=True)
        else:
            text = json.dumps(value, ensure_ascii=False, indent=2, default=self._json_default)
        self.write_text(path, text + ("" if text.endswith("\n") else "\n"))

    def write_jsonl(self, path: Path, values: list[BaseModel]) -> None:
        text = "\n".join(value.model_dump_json(exclude_none=True) for value in values)
        self.write_text(path, text + ("\n" if text else ""))

    @staticmethod
    def read_json(path: Path) -> Any:
        return json.loads(path.read_text(encoding="utf-8-sig"))

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json", exclude_none=True)
        return str(value)

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
