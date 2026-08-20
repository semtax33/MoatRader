from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from moatrader.valuation import (
    RoutedValuationInput,
    ScenarioAnnualObservation,
    ScenarioDcfBuildInput,
    ScenarioDcfBuilder,
    ScenarioDcfEngine,
    ValuationMethod,
)

try:
    from scripts.build_normalized_fcff_inputs import (
        decimal,
        file_sha256,
        latest_snapshot_values,
        read_json,
    )
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from build_normalized_fcff_inputs import (  # type: ignore[no-redef]
        decimal,
        file_sha256,
        latest_snapshot_values,
        read_json,
    )


BUILD_REPORT_VERSION = "scenario-dcf-input-build/1"
BIOTECH_TERMS = (
    "제약",
    "생물공학",
    "건강관리",
    "Pharmaceuticals",
    "Biotechnology",
    "Health Care",
    "HealthCare",
)
CYCLICAL_TERMS = (
    "철강",
    "화학",
    "조선",
    "해운",
    "자동차",
    "건설",
    "기계",
    "Steel",
    "Chemicals",
    "Shipbuilding",
    "Transportation Equipment",
    "Construction",
    "Machinery",
)


def _history(dcf_input: dict[str, Any]) -> list[ScenarioAnnualObservation]:
    sources = {
        int(item["business_year"]): item
        for item in dcf_input.get("annual_sources") or []
    }
    observations: list[ScenarioAnnualObservation] = []
    for item in (dcf_input.get("annual_history") or [])[-5:]:
        year = int(item["year"])
        metrics = item.get("metrics") or {}
        revenue = decimal(metrics.get("revenue"), field=f"revenue:{year}")
        ebit = decimal(metrics.get("ebit"), field=f"ebit:{year}")
        source = sources.get(year, {})
        source_ref = (
            f"OPENDART:{source.get('receipt_no', 'UNKNOWN')}:{year}:"
            f"SHA256:{source.get('payload_sha256', 'UNKNOWN')}"
        )
        observations.append(
            ScenarioAnnualObservation(
                fiscal_year=year,
                revenue=revenue,
                ebit=ebit,
                source_refs=[source_ref],
            )
        )
    return observations


def build_scenario_input(
    *,
    ticker: str,
    as_of: str,
    dcf_path: Path,
    snapshot_path: Path,
) -> RoutedValuationInput:
    dcf_input = read_json(dcf_path)
    snapshot = read_json(snapshot_path)
    observations = _history(dcf_input)
    metrics = dcf_input.get("metrics") or {}
    legacy = dcf_input.get("assumptions") or {}
    base_revenue = decimal(metrics.get("revenue"), field="base_revenue")
    base_ebit = decimal(metrics.get("ebit"), field="base_ebit")
    values = latest_snapshot_values(snapshot)
    try:
        invested_capital = (
            values["TOTAL_EQUITY"] + values["TOTAL_DEBT"] - values["CASH"]
        )
    except KeyError as exc:
        raise ValueError(f"snapshot lacks invested-capital component: {exc.args[0]}") from exc
    if invested_capital <= 0:
        raise ValueError("base invested capital must be positive")
    history_refs = [ref for item in observations for ref in item.source_refs]
    source_refs = list(
        dict.fromkeys(
            [
                f"PIT_DCF_INPUT:SHA256:{file_sha256(dcf_path)}",
                f"PIT_FINANCIAL_SNAPSHOT:SHA256:{file_sha256(snapshot_path)}",
                *history_refs,
                "POLICY:scenario-dcf-policy/1",
            ]
        )
    )
    source = ScenarioDcfBuildInput(
        issuer_id=str(snapshot.get("issuer_id") or ticker),
        as_of=as_of,
        observations=observations,
        base_period=str(legacy.get("base_period") or dcf_input.get("as_of") or as_of),
        base_revenue=base_revenue,
        base_ebit=base_ebit,
        base_invested_capital=invested_capital,
        tax_rate=decimal(legacy.get("tax_rate", "0.24"), field="tax_rate"),
        wacc=decimal(legacy.get("wacc"), field="wacc"),
        stable_growth=decimal(
            legacy.get("terminal_growth", "0.02"), field="stable_growth"
        ),
        net_debt=decimal(legacy.get("net_debt", "0"), field="net_debt"),
        diluted_shares=decimal(legacy.get("diluted_shares"), field="diluted_shares"),
        recovery_evidence=[
            "PIT_RECOVERY_PATH:CURRENT_EBIT_MARGIN_ABOVE_LATEST_ANNUAL"
        ],
        provenance=source_refs,
    )
    assumptions = ScenarioDcfBuilder().build(source)
    ScenarioDcfEngine().value(assumptions)
    return RoutedValuationInput(
        issuer_id=source.issuer_id,
        as_of=as_of,
        method=ValuationMethod.SCENARIO_DCF,
        assumptions=assumptions.model_dump(mode="json"),
        source_refs=source_refs,
    )


def read_routing(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output must be new and empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in read_routing(args.routing)
        if row.get("primary_method") == ValuationMethod.SCENARIO_DCF.value
    ]
    generated: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for row in rows:
        as_of = row["date"]
        ticker = row["ticker"].zfill(6)
        sector = str(row.get("sector") or "").replace(" ", "")
        if any(term.replace(" ", "") in sector for term in BIOTECH_TERMS):
            skipped.append(
                {
                    "date": as_of,
                    "ticker": ticker,
                    "reason": "PIPELINE_ADJUDICATION_REQUIRED",
                }
            )
            continue
        if any(term.replace(" ", "") in sector for term in CYCLICAL_TERMS):
            skipped.append(
                {
                    "date": as_of,
                    "ticker": ticker,
                    "reason": "STRUCTURAL_CYCLICAL_REQUIRES_NORMALIZED_FCFF",
                }
            )
            continue
        dcf_path = args.base_root / "date-inputs" / as_of / "dcf-inputs" / f"{ticker}.json"
        snapshot_path = (
            args.base_root
            / "runs"
            / f"kr-signal-{as_of}"
            / "companies"
            / ticker
            / "financial-snapshot.json"
        )
        try:
            envelope = build_scenario_input(
                ticker=ticker,
                as_of=as_of,
                dcf_path=dcf_path,
                snapshot_path=snapshot_path,
            )
            output_path = args.output / as_of / f"{ticker}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    envelope.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            generated.append({"date": as_of, "ticker": ticker})
        except Exception as exc:
            skipped.append(
                {
                    "date": as_of,
                    "ticker": ticker,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    report: dict[str, Any] = {
        "schema_version": BUILD_REPORT_VERSION,
        "routing_sha256": file_sha256(args.routing),
        "base_root": str(args.base_root),
        "llm_call_count": 0,
        "routed_count": len(rows),
        "generated_count": len(generated),
        "skipped_count": len(skipped),
        "generated": generated,
        "skipped": skipped,
    }
    (args.output / "_build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic Scenario DCF inputs from PIT operating history."
    )
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("routing", "base_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["skipped_count"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
