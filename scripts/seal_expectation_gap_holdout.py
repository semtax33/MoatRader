from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from moatrader.expectations import (
    FrozenRiskOverlayPolicy,
    HoldoutSignal,
    build_holdout_candidates,
    verify_and_normalize_holdout_ranks,
)
from moatrader.experiments import (
    FrozenExpectationGapContract,
    compute_contract_sha256,
    verify_frozen_sources,
)


SEOUL = ZoneInfo("Asia/Seoul")
FORBIDDEN_FIELDS = {"forward_return", "return", "future_price", "price_end", "performance"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Seal one prospective Expectation GAP holdout date before returns exist.")
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True, help="JSON array of HoldoutSignal objects")
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"holdout seal output already exists: {output}")
    raw_contract = json.loads(args.contract.resolve().read_text(encoding="utf-8-sig"))
    if compute_contract_sha256(raw_contract) != raw_contract.get("contract_sha256"):
        raise ValueError("frozen contract content hash is invalid")
    contract = FrozenExpectationGapContract.model_validate(raw_contract)
    verify_frozen_sources(contract, repository_root=args.repository_root.resolve())
    if args.as_of not in contract.holdout_dates:
        raise ValueError(f"date {args.as_of} is not preregistered in the frozen holdout")
    if datetime.now(tz=SEOUL).date() < args.as_of:
        raise ValueError("cannot seal a holdout date before its signal date")
    if sha256(args.universe.resolve()) != contract.universe_sha256:
        raise ValueError("holdout universe differs from the frozen universe")
    raw_signals = json.loads(args.signals.resolve().read_text(encoding="utf-8-sig"))
    for row in raw_signals:
        overlap = FORBIDDEN_FIELDS & {str(key).lower() for key in row}
        if overlap:
            raise ValueError(f"return-like fields are forbidden before signal seal: {sorted(overlap)}")
    signals = [HoldoutSignal.model_validate(row) for row in raw_signals]
    if len(signals) != contract.universe_count:
        raise ValueError(f"expected {contract.universe_count} signals, got {len(signals)}")
    if len({item.ticker for item in signals}) != len(signals):
        raise ValueError("holdout signals contain duplicate tickers")
    with args.universe.resolve().open("r", encoding="utf-8-sig", newline="") as stream:
        universe_tickers = {
            str(row.get("stock_code") or row.get("ticker") or "").strip().zfill(6)
            for row in csv.DictReader(stream)
        }
    signal_tickers = {item.ticker for item in signals}
    if signal_tickers != universe_tickers:
        missing = sorted(universe_tickers - signal_tickers)
        extra = sorted(signal_tickers - universe_tickers)
        raise ValueError(f"holdout signal universe mismatch; missing={missing}, extra={extra}")
    if any(item.signal_date != args.as_of for item in signals):
        raise ValueError("every holdout signal must match --as-of")
    signals = verify_and_normalize_holdout_ranks(signals)
    policy = FrozenRiskOverlayPolicy.model_validate(contract.risk_policy)
    candidates = [build_holdout_candidates(item, policy=policy) for item in signals]
    output.mkdir(parents=True)
    normalized_signals = [item.model_dump(mode="json") for item in sorted(signals, key=lambda item: item.ticker)]
    signal_path = output / "sealed-signals.json"
    signal_path.write_text(
        json.dumps(normalized_signals, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    candidate_path = output / "candidates.csv"
    fields = [
        "signal_date", "ticker", "cheap_rank", "candidate_a_eligible",
        "candidate_b_eligible", "candidate_c_eligible", "candidate_c_position_multiplier",
        "risk_decision", "risk_reason_codes",
    ]
    with candidate_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in sorted(candidates, key=lambda value: value.ticker):
            row = item.model_dump(mode="json")
            row["risk_reason_codes"] = ";".join(row["risk_reason_codes"])
            writer.writerow(row)
    embargo_until = args.as_of + timedelta(days=contract.forward_return_calendar_days)
    seal = {
        "schema_version": "expectation-gap-holdout-seal/1",
        "signal_date": args.as_of.isoformat(),
        "sealed_at": datetime.now(tz=SEOUL).isoformat(),
        "contract_sha256": contract.contract_sha256,
        "universe_sha256": contract.universe_sha256,
        "sealed_signals_sha256": sha256(signal_path),
        "candidates_sha256": sha256(candidate_path),
        "signal_count": len(signals),
        "candidate_a_count": sum(item.candidate_a_eligible for item in candidates),
        "candidate_b_count": sum(item.candidate_b_eligible for item in candidates),
        "candidate_c_count": sum(item.candidate_c_eligible for item in candidates),
        "returns_embargo_until": embargo_until.isoformat(),
        "return_data_accessed": False,
    }
    (output / "seal.json").write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output / "seal.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
