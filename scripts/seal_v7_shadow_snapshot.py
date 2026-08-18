from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from moatrader.experiments.shadow import (
    ShadowCompanySignal,
    ExpectationGapResearchContract,
    seal_shadow_snapshot,
)


def _aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a UTC offset")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Seal one immutable v7 weekly shadow snapshot.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--signal-at", type=_aware_datetime, required=True)
    parser.add_argument("--sealed-at", type=_aware_datetime, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = ExpectationGapResearchContract.model_validate_json(
        args.contract.resolve().read_text(encoding="utf-8-sig")
    )
    raw_signals = json.loads(args.signals.resolve().read_text(encoding="utf-8-sig"))
    signals = [ShadowCompanySignal.model_validate(item) for item in raw_signals]
    snapshot = seal_shadow_snapshot(
        contract=contract,
        signal_at=args.signal_at,
        sealed_at=args.sealed_at,
        signals=signals,
        output_path=args.output.resolve(),
    )
    print(snapshot.snapshot_sha256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
