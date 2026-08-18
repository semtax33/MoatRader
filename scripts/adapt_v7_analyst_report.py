from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from moatrader.evidence.research_reports import ResearchReportBundle


def _aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("as-of must include a UTC offset")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quarantine analyst market opinion and emit v7 intrinsic driver evidence."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--as-of", type=_aware_datetime, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"v7 analyst adapter output already exists: {output}")
    bundle = ResearchReportBundle.model_validate_json(
        args.input.resolve().read_text(encoding="utf-8-sig")
    )
    intrinsic = bundle.intrinsic_view(as_of=args.as_of)
    drivers = intrinsic.to_valuation_driver_bundle()
    payload = {
        "schema_version": "v7-analyst-intrinsic-adapter/1",
        "intrinsic_research": intrinsic.model_dump(mode="json"),
        "valuation_driver_evidence": drivers.model_dump(mode="json"),
        "market_opinion_item_count_quarantined": len(bundle.market_opinion),
        "price_leakage_detected": False,
        "return_data_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
