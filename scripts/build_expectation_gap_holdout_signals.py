from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from moatrader.expectations import (
    AlphaSignal,
    AlphaSignalStatus,
    CheapSignal,
    HoldoutSignal,
    HoldoutResearchInput,
    assign_method_archetype_percentiles,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def ticker(value: Any) -> str:
    return str(value or "").strip().zfill(6)


def number(value: Any) -> Decimal | None:
    if value in (None, ""):
        return None
    result = Decimal(str(value))
    return result if result.is_finite() else None


def sector_snapshot_date(evidence_ref: str) -> date:
    parts = evidence_ref.split(":")
    if len(parts) < 3 or parts[0:2] != ["KRX", "MDCSTAT03901"]:
        raise ValueError(f"unsupported PIT sector evidence reference: {evidence_ref}")
    return date.fromisoformat(parts[2])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Assemble return-blind HoldoutSignal JSON from routed valuation and research inputs."
    )
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--valuation-signals", type=Path, required=True)
    parser.add_argument(
        "--research-inputs",
        type=Path,
        required=True,
        help="JSON array with risk, confirmation, and PIT source_references per ticker",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"holdout signal output already exists: {output}")

    universe_rows = read_csv(args.universe.resolve())
    if len(universe_rows) != 150:
        raise ValueError(f"frozen holdout requires 150 universe rows, got {len(universe_rows)}")
    universe = {ticker(row.get("stock_code") or row.get("ticker")): row for row in universe_rows}
    routing = {
        ticker(row.get("ticker")): row
        for row in read_csv(args.routing.resolve())
        if str(row.get("date")) == args.as_of.isoformat()
    }
    valuation = {
        ticker(row.get("ticker")): row
        for row in read_csv(args.valuation_signals.resolve())
        if str(row.get("date")) == args.as_of.isoformat()
    }
    raw_research = json.loads(args.research_inputs.resolve().read_text(encoding="utf-8-sig"))
    research_items = [HoldoutResearchInput.model_validate(row) for row in raw_research]
    if len({item.ticker for item in research_items}) != len(research_items):
        raise ValueError("research inputs contain duplicate tickers")
    research = {item.ticker: item for item in research_items}
    expected = set(universe)
    for name, values in (("routing", routing), ("valuation", valuation), ("research", research)):
        if set(values) != expected:
            raise ValueError(f"{name} ticker set differs from the frozen universe")

    cheap_signals: list[CheapSignal] = []
    ordered = sorted(expected)
    for code in ordered:
        row = valuation[code]
        status = AlphaSignalStatus(str(row.get("alpha_status") or "MODEL_NOT_APPLICABLE"))
        market_price = number(row.get("market_price"))
        fair_value = number(row.get("primary_fair_value_per_share"))
        raw_gap = number(row.get("raw_value_gap") or row.get("raw_expectation_gap"))
        if status == AlphaSignalStatus.VALID:
            if market_price is None or fair_value is None:
                raise ValueError(f"VALID Cheap input is incomplete for {code}")
            cheap = CheapSignal.from_values(
                valuation_method=str(row["method"]),
                economic_archetype=str(row["economic_archetype"]),
                market_price=market_price,
                primary_fair_value_per_share=fair_value,
            )
            if raw_gap is None or abs(cheap.raw_value_gap - raw_gap) > Decimal("0.00000001"):
                raise ValueError(f"raw expectation gap drift for {code}")
        else:
            complete = market_price is not None and fair_value is not None and raw_gap is not None
            cheap = CheapSignal(
                valuation_method=str(row["method"]),
                economic_archetype=str(row["economic_archetype"]),
                market_price=market_price if complete else None,
                primary_fair_value_per_share=fair_value if complete else None,
                raw_value_gap=raw_gap if complete else None,
                status=status,
                rank_eligible=False,
            )
        cheap_signals.append(cheap)
    cheap_signals = assign_method_archetype_percentiles(cheap_signals)

    signals: list[HoldoutSignal] = []
    for code, cheap in zip(ordered, cheap_signals, strict=True):
        route = routing[code]
        info = research[code]
        sector_ref = str(route["sector_evidence_ref"])
        signals.append(
            HoldoutSignal(
                signal_date=args.as_of,
                ticker=code,
                issuer_name=str(universe[code].get("name") or route.get("issuer_name") or code),
                sector=str(route["sector"]),
                sector_snapshot_date=sector_snapshot_date(sector_ref),
                sector_evidence_ref=sector_ref,
                alpha=AlphaSignal(cheap=cheap),
                risk=info.risk,
                confirmation=info.confirmation,
                route_profile_sha256=str(route["profile_sha256"]),
                source_references=info.source_references,
                legacy_composite_diagnostic=info.legacy_composite_diagnostic,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in signals],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
