from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pydantic import Field

from moatrader.canonical.models import ContractModel
from moatrader.experiments.contract import compute_contract_sha256


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath((str(path), str(parent))) == str(parent)
    except ValueError:
        return False


def ensure_new_experiment_output(
    output: Path,
    *,
    required_label: str,
    protected_roots: list[Path],
) -> Path:
    """Require a new labeled experiment path outside protected artifact trees."""

    resolved = output.resolve()
    if not required_label.strip() or required_label.casefold() not in resolved.name.casefold():
        raise ValueError(f"experiment output directory name must contain {required_label!r}")
    for root in protected_roots:
        protected = root.resolve()
        if _is_within(resolved, protected) or _is_within(protected, resolved):
            raise ValueError(f"experiment output overlaps protected path: {protected}")
    if resolved.exists():
        raise FileExistsError(f"experiment output must be new and must not overwrite data: {resolved}")
    return resolved


def snapshot_protected_files(
    *,
    repository_root: Path,
    contract_path: Path,
    stability_directories: list[Path],
) -> dict[str, str]:
    """Hash every v6 contract/source/artifact file without changing it."""

    root = repository_root.resolve()
    contract_file = contract_path.resolve()
    payload = json.loads(contract_file.read_text(encoding="utf-8-sig"))
    files: set[Path] = {contract_file}
    files.update(
        root / relative for relative in payload.get("frozen_source_sha256", {})
    )
    files.update(path.resolve() for directory in stability_directories for path in directory.rglob("*") if path.is_file())
    files.update(path.resolve() for path in contract_file.parent.rglob("*") if path.is_file())
    snapshot: dict[str, str] = {}
    for path in sorted(files, key=lambda item: item.as_posix()):
        if not path.is_file():
            relative = path.relative_to(root).as_posix() if _is_within(path, root) else str(path)
            snapshot[relative] = "MISSING"
            continue
        relative = path.relative_to(root).as_posix() if _is_within(path, root) else str(path)
        snapshot[relative] = sha256_file(path)
    return snapshot


class IntegrityMismatch(ContractModel):
    path: str
    expected_sha256: str = Field(pattern=r"^(?:[0-9a-f]{64}|MISSING)$")
    actual_sha256: str = Field(pattern=r"^(?:[0-9a-f]{64}|MISSING)$")


class V6IntegrityReport(ContractModel):
    schema_version: str = "v6-sacred-holdout-integrity/1"
    contract_path: str
    contract_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_payload_hash_valid: bool
    frozen_source_mismatches: list[IntegrityMismatch]
    engineering_artifact_mismatches: list[IntegrityMismatch]
    current_file_sha256: dict[str, str]

    @property
    def pristine_against_contract(self) -> bool:
        return (
            self.contract_payload_hash_valid
            and not self.frozen_source_mismatches
            and not self.engineering_artifact_mismatches
        )


def audit_v6_integrity(
    *,
    repository_root: Path,
    contract_path: Path,
    stability_directory: Path,
    stability_directories_for_snapshot: list[Path] | None = None,
) -> V6IntegrityReport:
    root = repository_root.resolve()
    contract_file = contract_path.resolve()
    payload = json.loads(contract_file.read_text(encoding="utf-8-sig"))
    expected_contract_hash = str(payload.get("contract_sha256", ""))
    actual_contract_hash = compute_contract_sha256(payload)

    source_mismatches: list[IntegrityMismatch] = []
    for relative, expected in sorted(payload.get("frozen_source_sha256", {}).items()):
        path = root / relative
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != expected:
            source_mismatches.append(
                IntegrityMismatch(path=relative, expected_sha256=expected, actual_sha256=actual)
            )

    artifact_mismatches: list[IntegrityMismatch] = []
    stability = stability_directory.resolve()
    for name, expected in sorted(payload.get("engineering_stability_sha256", {}).items()):
        path = stability / name
        actual = sha256_file(path) if path.is_file() else "MISSING"
        if actual != expected:
            artifact_mismatches.append(
                IntegrityMismatch(
                    path=path.relative_to(root).as_posix() if _is_within(path, root) else str(path),
                    expected_sha256=expected,
                    actual_sha256=actual,
                )
            )

    snapshot_dirs = stability_directories_for_snapshot or [stability]
    return V6IntegrityReport(
        contract_path=(
            contract_file.relative_to(root).as_posix()
            if _is_within(contract_file, root)
            else str(contract_file)
        ),
        contract_file_sha256=sha256_file(contract_file),
        contract_payload_sha256=actual_contract_hash,
        contract_payload_hash_valid=actual_contract_hash == expected_contract_hash,
        frozen_source_mismatches=source_mismatches,
        engineering_artifact_mismatches=artifact_mismatches,
        current_file_sha256=snapshot_protected_files(
            repository_root=root,
            contract_path=contract_file,
            stability_directories=snapshot_dirs,
        ),
    )
