from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from moatrader.canonical.models import SourceType
from moatrader.ingestion.models import (
    COLLECTOR_VERSION,
    CollectedFiling,
    CollectionAction,
    FilingDescriptor,
)


_DOCUMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def safe_relative_path(value: str) -> Path:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe artifact path: {value!r}")
    if any(":" in part for part in path.parts):
        raise ValueError(f"unsafe artifact path: {value!r}")
    return Path(*path.parts)


def source_namespace(source_type: SourceType) -> str:
    if source_type == SourceType.DART:
        return "dart"
    if source_type == SourceType.SEC_EDGAR:
        return "sec-edgar"
    if source_type == SourceType.IR:
        return "kind-ir"
    if source_type == SourceType.INDUSTRY:
        return "hankyung-industry"
    raise ValueError(f"Bronze API collector does not support source type {source_type.value}")


class BronzeFilingStore:
    """Immutable, content-addressed Bronze storage with an atomic latest pointer."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def current(self, source_type: SourceType, source_document_id: str) -> CollectedFiling | None:
        directory = self._filing_directory(source_type, source_document_id)
        latest_path = directory / "latest.json"
        if not latest_path.is_file():
            return None
        latest = json.loads(latest_path.read_text(encoding="utf-8-sig"))
        metadata_path = self.root / safe_relative_path(str(latest["metadata_path"]))
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Bronze latest pointer references missing metadata: {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        return self._collected_from_metadata(metadata, CollectionAction.UNCHANGED)

    def save(
        self,
        descriptor: FilingDescriptor,
        *,
        files: dict[str, bytes],
        primary_path: str,
        downloaded_at: datetime | None = None,
    ) -> CollectedFiling:
        downloaded_at = downloaded_at or datetime.now(timezone.utc)
        if downloaded_at.tzinfo is None or downloaded_at.utcoffset() is None:
            raise ValueError("downloaded_at must be timezone-aware")
        normalized_files = {str(safe_relative_path(name)): content for name, content in files.items()}
        normalized_primary = str(safe_relative_path(primary_path))
        if normalized_primary not in normalized_files:
            raise ValueError(f"primary_path is not present in files: {primary_path}")
        if len(normalized_files) != len(files):
            raise ValueError("artifact paths collide after normalization")

        raw_sha256 = hashlib.sha256(normalized_files[normalized_primary]).hexdigest()
        version_sha256 = self._version_hash(normalized_files)
        filing_dir = self._filing_directory(descriptor.source_type, descriptor.source_document_id)
        versions_dir = filing_dir / "versions"
        version_dir = versions_dir / version_sha256
        latest_path = filing_dir / "latest.json"
        previous = self.current(descriptor.source_type, descriptor.source_document_id)
        if version_dir.is_dir():
            metadata_path = version_dir / "metadata.json"
            if not metadata_path.is_file():
                raise FileNotFoundError(f"incomplete immutable Bronze version: {version_dir}")
            self._write_latest(latest_path, version_dir, metadata_path, raw_sha256, version_sha256)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            return self._collected_from_metadata(metadata, CollectionAction.UNCHANGED)

        versions_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".collect-", dir=versions_dir))
        try:
            file_hashes: dict[str, str] = {}
            for relative, content in normalized_files.items():
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                self._atomic_write(target, content)
                file_hashes[relative] = hashlib.sha256(content).hexdigest()

            metadata = descriptor.adapter_metadata()
            metadata.update(
                {
                    "raw_sha256": raw_sha256,
                    "version_sha256": version_sha256,
                    "downloaded_at": downloaded_at.isoformat(),
                    "collector_version": COLLECTOR_VERSION,
                    "supersedes_version_sha256": previous.version_sha256 if previous else None,
                    "storage": {
                        "primary_path": self._relative(version_dir / normalized_primary),
                        "metadata_path": self._relative(version_dir / "metadata.json"),
                        "version_directory": self._relative(version_dir),
                        "files": file_hashes,
                    },
                }
            )
            metadata = {key: value for key, value in metadata.items() if value is not None}
            self._write_json(temporary / "metadata.json", metadata)
            checksum_lines = "".join(f"{digest}  {name.replace(os.sep, '/')}\n" for name, digest in sorted(file_hashes.items()))
            self._atomic_write(temporary / "sha256.txt", checksum_lines.encode("utf-8"))
            try:
                temporary.replace(version_dir)
            except FileExistsError:
                shutil.rmtree(temporary)
            self._write_latest(latest_path, version_dir, version_dir / "metadata.json", raw_sha256, version_sha256)
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary)
            raise

        action = CollectionAction.REVISED if previous else CollectionAction.DOWNLOADED
        saved_metadata = json.loads((version_dir / "metadata.json").read_text(encoding="utf-8-sig"))
        return self._collected_from_metadata(saved_metadata, action)

    def iter_current(self, sources: set[SourceType] | None = None) -> list[CollectedFiling]:
        selected = sources or {SourceType.DART, SourceType.SEC_EDGAR, SourceType.IR}
        filings: list[CollectedFiling] = []
        for source_type in sorted(selected, key=lambda value: value.value):
            source_dir = self.root / source_namespace(source_type)
            if not source_dir.is_dir():
                continue
            for filing_dir in sorted(path for path in source_dir.iterdir() if path.is_dir()):
                current = self.current(source_type, filing_dir.name)
                if current is not None:
                    current.verify_files()
                    filings.append(current)
        return filings

    def _filing_directory(self, source_type: SourceType, source_document_id: str) -> Path:
        if not _DOCUMENT_ID.fullmatch(source_document_id):
            raise ValueError(f"unsafe source document ID: {source_document_id!r}")
        return self.root / source_namespace(source_type) / source_document_id

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.root).as_posix()

    def _write_latest(
        self,
        path: Path,
        version_dir: Path,
        metadata_path: Path,
        raw_sha256: str,
        version_sha256: str,
    ) -> None:
        self._write_json(
            path,
            {
                "version_sha256": version_sha256,
                "raw_sha256": raw_sha256,
                "version_directory": self._relative(version_dir),
                "metadata_path": self._relative(metadata_path),
            },
        )

    def _collected_from_metadata(
        self,
        metadata: dict[str, object],
        action: CollectionAction,
    ) -> CollectedFiling:
        storage = dict(metadata["storage"])  # type: ignore[arg-type]
        ticker = str(metadata.get("ticker") or metadata.get("stock_code") or "")
        if not ticker:
            prefix = "CIK" if metadata["source_type"] == SourceType.SEC_EDGAR.value else "DART"
            ticker = prefix + str(metadata["issuer_id"])
        return CollectedFiling(
            source_type=SourceType(str(metadata["source_type"])),
            source_document_id=str(metadata["source_document_id"]),
            issuer_id=str(metadata["issuer_id"]),
            issuer_name=str(metadata["issuer_name"]) if metadata.get("issuer_name") else None,
            ticker=ticker,
            action=action,
            raw_sha256=str(metadata["raw_sha256"]),
            version_sha256=str(metadata["version_sha256"]),
            downloaded_at=datetime.fromisoformat(str(metadata["downloaded_at"]).replace("Z", "+00:00")),
            input_path=str((self.root / safe_relative_path(str(storage["primary_path"]))).resolve()),
            metadata_path=str((self.root / safe_relative_path(str(storage["metadata_path"]))).resolve()),
            version_directory=str((self.root / safe_relative_path(str(storage["version_directory"]))).resolve()),
        )

    @staticmethod
    def _version_hash(files: dict[str, bytes]) -> str:
        digest = hashlib.sha256()
        for name, content in sorted(files.items()):
            digest.update(name.replace(os.sep, "/").encode("utf-8"))
            digest.update(b"\0")
            digest.update(hashlib.sha256(content).digest())
        return digest.hexdigest()

    @staticmethod
    def _write_json(path: Path, value: dict[str, object]) -> None:
        text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
        BronzeFilingStore._atomic_write(path, text.encode("utf-8"))

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_bytes(content)
        temporary.replace(path)
