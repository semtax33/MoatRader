from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

from moatrader.financial.historical_xbrl import parse_dart_ifrs_archive


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reparse v7 OpenDART XBRL metrics using each filing's actual fiscal period end."
    )
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if "v7" not in root.as_posix().casefold():
        raise ValueError("metric repair is restricted to a v7 path")
    changed: list[dict[str, object]] = []
    skipped = 0
    for metadata_path in sorted(root.rglob("metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        xbrl_path = metadata_path.parent / "financial-statements-xbrl.zip"
        if not xbrl_path.is_file():
            skipped += 1
            continue
        expected = metadata.get("xbrl_archive_sha256")
        actual = _sha256(xbrl_path)
        if expected != actual:
            raise ValueError(f"v7 XBRL hash mismatch: {xbrl_path}")
        period_end = date.fromisoformat(metadata["fiscal_period_end"])
        old_coverage = (metadata.get("metrics") or {}).get("metric_coverage_count")
        metrics = parse_dart_ifrs_archive(
            xbrl_path.read_bytes(),
            fiscal_year=period_end.year,
            period_end=period_end,
        )
        metadata["metrics"] = metrics.model_dump(mode="json")
        metadata["metric_parser"] = "v7-actual-fiscal-period-end/1"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if old_coverage != metrics.metric_coverage_count:
            changed.append(
                {
                    "rcept_no": metadata["rcept_no"],
                    "fiscal_period_end": period_end.isoformat(),
                    "old_coverage": old_coverage,
                    "new_coverage": metrics.metric_coverage_count,
                }
            )
    report = {
        "schema_version": "v7-opendart-metric-repair/1",
        "metadata_count": len(list(root.rglob("metadata.json"))),
        "xbrl_missing_count": skipped,
        "coverage_changed_count": len(changed),
        "coverage_changes": changed,
    }
    (root / "metric-repair-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(root / "metric-repair-report.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
