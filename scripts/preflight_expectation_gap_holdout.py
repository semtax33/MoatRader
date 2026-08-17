from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from moatrader.expectations import (
    HoldoutResearchInput,
    HoldoutSignal,
    verify_and_normalize_holdout_ranks,
)
from moatrader.experiments import (
    FrozenExpectationGapContract,
    compute_contract_sha256,
    verify_frozen_sources,
)
from moatrader.financial.pit_sector import load_pit_sector_csv, resolve_pit_sector


SEOUL = ZoneInfo("Asia/Seoul")
RETURN_LIKE_NAMES = {
    "return",
    "returns",
    "forward_return",
    "forward_returns",
    "future_price",
    "price_end",
    "performance",
    "evaluation",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def normalized_ticker(value: Any) -> str:
    return str(value or "").strip().zfill(6)


def return_like(path: Path) -> bool:
    normalized = path.stem.lower().replace("-", "_").replace(".", "_")
    tokens = {
        item.lower()
        for item in normalized.split("_")
        if item
    }
    return bool(
        tokens & RETURN_LIKE_NAMES
        or any(marker in normalized for marker in RETURN_LIKE_NAMES)
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Return-blind readiness audit for one preregistered holdout date."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--pit-sector-map", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--valuation-signals", type=Path, required=True)
    parser.add_argument("--research-inputs", type=Path, required=True)
    parser.add_argument("--built-signals", type=Path, required=True)
    parser.add_argument("--sealed-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    checks: list[dict[str, str]] = []

    def record(name: str, status: str, detail: str) -> None:
        checks.append({"name": name, "status": status, "detail": detail})

    raw_contract = json.loads(args.contract.resolve().read_text(encoding="utf-8-sig"))
    if compute_contract_sha256(raw_contract) != raw_contract.get("contract_sha256"):
        record("contract_content", "FAIL", "contract SHA-256 does not match content")
        contract = None
    else:
        contract = FrozenExpectationGapContract.model_validate(raw_contract)
        record("contract_content", "PASS", str(contract.contract_sha256))
        try:
            verify_frozen_sources(contract, repository_root=args.repository_root.resolve())
            record("frozen_sources", "PASS", f"{len(contract.frozen_source_sha256)} sources")
        except ValueError as exc:
            record("frozen_sources", "FAIL", str(exc))

    today = datetime.now(tz=SEOUL).date()
    if contract is not None and args.as_of not in contract.holdout_dates:
        record("preregistered_date", "FAIL", f"{args.as_of} is not in the contract")
    else:
        record("preregistered_date", "PASS", args.as_of.isoformat())
    record(
        "signal_date_reached",
        "PASS" if today >= args.as_of else "WAIT",
        f"today={today}; signal_date={args.as_of}",
    )

    universe_path = args.universe.resolve()
    universe_rows = read_csv(universe_path)
    universe_tickers = {
        normalized_ticker(row.get("stock_code") or row.get("ticker"))
        for row in universe_rows
    }
    if contract is None:
        record("frozen_universe", "FAIL", "contract unavailable")
    elif sha256(universe_path) != contract.universe_sha256:
        record("frozen_universe", "FAIL", "universe hash differs from contract")
    elif len(universe_rows) != contract.universe_count or len(universe_tickers) != contract.universe_count:
        record("frozen_universe", "FAIL", "universe count or ticker uniqueness differs")
    else:
        record("frozen_universe", "PASS", f"{len(universe_tickers)} tickers")

    required_paths = {
        "pit_sector_map": args.pit_sector_map.resolve(),
        "routing": args.routing.resolve(),
        "valuation_signals": args.valuation_signals.resolve(),
        "research_inputs": args.research_inputs.resolve(),
        "built_signals": args.built_signals.resolve(),
    }
    due_status = "FAIL" if today >= args.as_of else "WAIT"
    for name, path in required_paths.items():
        if return_like(path):
            record(name, "FAIL", f"return-like input path is forbidden before seal: {path}")
        elif not path.is_file():
            record(name, due_status, f"missing: {path}")
        else:
            record(name, "PASS", str(path))

    pit_path = required_paths["pit_sector_map"]
    if pit_path.is_file():
        cutoff = datetime.combine(args.as_of, time.max, tzinfo=SEOUL)
        records = load_pit_sector_csv(pit_path)
        covered = {
            code
            for code in universe_tickers
            if resolve_pit_sector(records, ticker=code, as_of=cutoff) is not None
        }
        if covered != universe_tickers:
            record(
                "pit_sector_coverage",
                "FAIL",
                f"covered={len(covered)}/{len(universe_tickers)}",
            )
        else:
            record("pit_sector_coverage", "PASS", f"{len(covered)} PIT sectors")

    for name in ("routing", "valuation_signals"):
        path = required_paths[name]
        if path.is_file():
            rows = [row for row in read_csv(path) if str(row.get("date")) == args.as_of.isoformat()]
            tickers = {normalized_ticker(row.get("ticker")) for row in rows}
            if len(rows) != len(universe_tickers) or tickers != universe_tickers:
                record(f"{name}_panel", "FAIL", f"rows={len(rows)}; unique_tickers={len(tickers)}")
            else:
                record(f"{name}_panel", "PASS", f"{len(rows)} rows")

    research_path = required_paths["research_inputs"]
    if research_path.is_file():
        try:
            research = [
                HoldoutResearchInput.model_validate(item)
                for item in json.loads(research_path.read_text(encoding="utf-8-sig"))
            ]
            tickers = [item.ticker for item in research]
            if (
                len(tickers) != len(universe_tickers)
                or set(tickers) != universe_tickers
                or len(set(tickers)) != len(tickers)
            ):
                raise ValueError(
                    f"rows={len(tickers)}; unique_tickers={len(set(tickers))}"
                )
            record("research_panel", "PASS", f"{len(tickers)} validated rows")
        except Exception as exc:
            record("research_panel", "FAIL", f"{type(exc).__name__}: {exc}")
        manifest_path = research_path.with_suffix(".manifest.json")
        if not manifest_path.is_file():
            record("research_manifest", "FAIL", f"missing: {manifest_path}")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            valid_manifest = (
                manifest.get("return_data_accessed") is False
                and manifest.get("row_count") == len(universe_tickers)
                and manifest.get("unique_ticker_count") == len(universe_tickers)
                and manifest.get("research_inputs_sha256") == sha256(research_path)
            )
            record(
                "research_manifest",
                "PASS" if valid_manifest else "FAIL",
                str(manifest_path),
            )

    built_path = required_paths["built_signals"]
    if built_path.is_file():
        try:
            signals = [
                HoldoutSignal.model_validate(item)
                for item in json.loads(built_path.read_text(encoding="utf-8-sig"))
            ]
            if {item.ticker for item in signals} != universe_tickers or len(signals) != len(universe_tickers):
                raise ValueError("built signals differ from frozen universe")
            if any(item.signal_date != args.as_of for item in signals):
                raise ValueError("built signal date differs from requested date")
            verify_and_normalize_holdout_ranks(signals)
            record("built_signal_contract", "PASS", f"{len(signals)} validated signals")
        except Exception as exc:
            record("built_signal_contract", "FAIL", f"{type(exc).__name__}: {exc}")

    seal_path = args.sealed_output.resolve() / "seal.json"
    if seal_path.is_file():
        seal = json.loads(seal_path.read_text(encoding="utf-8-sig"))
        valid = (
            contract is not None
            and seal.get("contract_sha256") == contract.contract_sha256
            and seal.get("signal_date") == args.as_of.isoformat()
            and seal.get("return_data_accessed") is False
        )
        record("signal_seal", "PASS" if valid else "FAIL", str(seal_path))
    else:
        record("signal_seal", due_status, f"missing: {seal_path}")

    status = (
        "FAIL"
        if any(item["status"] == "FAIL" for item in checks)
        else "WAIT"
        if any(item["status"] == "WAIT" for item in checks)
        else "READY"
    )
    report = {
        "schema_version": "expectation-gap-holdout-preflight/1",
        "as_of": args.as_of.isoformat(),
        "checked_at": datetime.now(tz=SEOUL).isoformat(),
        "return_data_accessed": False,
        "status": status,
        "checks": checks,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if status == "READY" else 2 if status == "WAIT" else 1


if __name__ == "__main__":
    sys.exit(main())
