from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(record: dict[str, Any]) -> str | None:
    path = Path(str(record["path"]))
    stat = path.stat()
    if stat.st_size != int(record["byte_count"]):
        return f"byte_count mismatch: {path}"
    if stat.st_mtime_ns != int(record["modified_time_ns"]):
        return f"modified_time_ns mismatch: {path}"
    if _sha256(path) != str(record["raw_sha256"]):
        return f"sha256 mismatch: {path}"
    return None


def verify(manifest_path: Path, *, workers: int) -> int:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = list(payload["records"])
    if workers < 1:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for index, error in enumerate(executor.map(_verify, records), start=1):
            if error is not None:
                raise ValueError(error)
            if index % 5_000 == 0 or index == len(records):
                print(f"verified: {index}/{len(records)}", flush=True)
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only parallel verification of a historical source-integrity manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    count = verify(args.manifest, workers=args.workers)
    print(f"PASS_NO_SOURCE_MUTATION {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
