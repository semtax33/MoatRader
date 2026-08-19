from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal as D
from pathlib import Path

from moatrader.valuation import (
    ApvAssumptions,
    ApvCase,
    EconomicArchetype,
    NavAsset,
    NavAssumptions,
    PipelineAsset,
    RimAssumptions,
    RimScenarioSet,
    RnpvScenarioSet,
    RoutedValuationInput,
    ValuationMethod,
    ValuationProfile,
)
from moatrader.valuation.biotech_rnpv import BiotechRnpvAssumptions
from scripts.audit_expanded_valuation_signals import run


DATE = "2026-05-31"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8")


def _point(concept: str, value: str) -> dict[str, object]:
    return {
        "concept": concept,
        "points": [{"period": "2025", "period_basis": "FY", "value": value}],
    }


def _rim(roe: str) -> RimAssumptions:
    return RimAssumptions(
        book_equity=D("1000"),
        roe_path=[D(roe)] * 5,
        cost_of_equity=D("0.10"),
        payout_ratio=D("0.40"),
        terminal_roe=D(roe),
        terminal_growth=D("0.03"),
        diluted_shares=D("10"),
        assumption_confidence=D("0.8"),
        provenance=["PIT:RIM"],
    )


def _rnpv(probability: str) -> BiotechRnpvAssumptions:
    return BiotechRnpvAssumptions(
        assets=[
            PipelineAsset(
                name="Drug A",
                years_to_launch=3,
                probability_of_approval=D(probability),
                launch_value=D("1000"),
                remaining_development_costs=[D("20")] * 3,
                evidence_ids=["PIT:TRIAL"],
            )
        ],
        discount_rate=D("0.12"),
        net_cash=D("100"),
        diluted_shares=D("10"),
    )


def _apv_case(fcff: str, shield: str) -> ApvCase:
    return ApvCase(
        unlevered_fcff=[D(fcff)] * 5,
        terminal_cash_flow=D(fcff),
        terminal_growth=D("0.02"),
        unlevered_cost_of_capital=D("0.10"),
        tax_shields=[D(shield)] * 5,
        tax_shield_discount_rate=D("0.06"),
    )


def _input(ticker: str, method: ValuationMethod, assumptions: object) -> dict[str, object]:
    return RoutedValuationInput(
        issuer_id=ticker,
        as_of=DATE,
        method=method,
        assumptions=assumptions.model_dump(mode="json"),
        source_refs=[f"PIT:FIXTURE:{ticker}"],
    ).model_dump(mode="json")


def test_audit_executes_rim_nav_rnpv_and_emits_unified_route_audit(tmp_path: Path) -> None:
    dates = tmp_path / "dates.csv"
    universe = tmp_path / "universe.csv"
    sectors = tmp_path / "sectors.csv"
    base = tmp_path / "base"
    valuation_inputs = tmp_path / "valuation-inputs"
    valuation_profiles = tmp_path / "valuation-profiles"
    output = tmp_path / "output"
    output_b = tmp_path / "output-b"
    _write_csv(dates, [{"date": DATE}])
    _write_csv(
        universe,
        [
            {"stock_code": "000001", "name": "은행", "size_bucket": "LARGE", "listed_shares": 10, "finance_hint": 1, "holding_hint": 0},
            {"stock_code": "000002", "name": "리츠", "size_bucket": "MID", "listed_shares": 10, "finance_hint": 0, "holding_hint": 0},
            {"stock_code": "000003", "name": "바이오", "size_bucket": "SMALL", "listed_shares": 10, "finance_hint": 0, "holding_hint": 0},
            {"stock_code": "000004", "name": "레버리지", "size_bucket": "MID", "listed_shares": 10, "finance_hint": 0, "holding_hint": 0},
        ],
    )
    _write_csv(
        sectors,
        [
            {"ticker": "000001", "sector": "은행", "industry_code": "BANK"},
            {"ticker": "000002", "sector": "리츠", "industry_code": "REIT"},
            {"ticker": "000003", "sector": "생물공학", "industry_code": "BIO"},
            {"ticker": "000004", "sector": "산업재", "industry_code": "IND"},
        ],
    )
    snapshots = {
        "000001": [_point("TOTAL_EQUITY", "1000"), _point("NET_INCOME", "120")],
        "000002": [_point("TOTAL_ASSETS", "1200"), _point("TOTAL_DEBT", "300"), _point("CASH", "100")],
        "000003": [_point("REVENUE", "1"), _point("EBIT", "-5"), _point("TOTAL_ASSETS", "1000")],
        "000004": [_point("REVENUE", "1000"), _point("EBIT", "100")],
    }
    for ticker, series in snapshots.items():
        _write_json(
            base / "runs" / f"kr-signal-{DATE}" / "companies" / ticker / "financial-snapshot.json",
            {"issuer_id": ticker, "series": series},
        )
    _write_csv(
        base / "date-inputs" / DATE / "universe-manifest.csv",
        [
            {"ticker": "000001", "current_price": "100"},
            {"ticker": "000002", "current_price": "50"},
            {"ticker": "000003", "current_price": "10"},
            {"ticker": "000004", "current_price": "100"},
        ],
    )
    method_inputs = {
        "000001": _input(
            "000001",
            ValuationMethod.RIM,
            RimScenarioSet(downside=_rim("0.08"), base=_rim("0.12"), upside=_rim("0.16")),
        ),
        "000002": _input(
            "000002",
            ValuationMethod.NAV,
            NavAssumptions(
                assets=[NavAsset(name="Property", base_value=D("1000"), evidence_ids=["PIT:NAV"])],
                cash=D("100"),
                debt=D("300"),
                diluted_shares=D("10"),
                assumption_confidence=D("0.8"),
                provenance=["PIT:NAV"],
            ),
        ),
        "000003": _input(
            "000003",
            ValuationMethod.RNPV,
            RnpvScenarioSet(
                downside=_rnpv("0.2"), base=_rnpv("0.5"), upside=_rnpv("0.8")
            ),
        ),
        "000004": _input(
            "000004",
            ValuationMethod.APV,
            ApvAssumptions(
                downside=_apv_case("70", "5"),
                base=_apv_case("100", "10"),
                upside=_apv_case("130", "15"),
                debt=D("300"),
                cash=D("50"),
                diluted_shares=D("10"),
                assumption_confidence=D("0.8"),
                provenance=["PIT:APV"],
            ),
        ),
    }
    for ticker, payload in method_inputs.items():
        _write_json(valuation_inputs / DATE / f"{ticker}.json", payload)
    _write_json(
        valuation_profiles / DATE / "000004.json",
        ValuationProfile(
            issuer_id="000004",
            as_of=DATE,
            sector="산업재",
            industry="레버리지",
            economic_archetype=EconomicArchetype.LEVERAGE_DRIVEN,
            leverage_path_material=True,
            available_data=[
                "unlevered_cashflows",
                "debt_schedule",
                "tax_shields",
                "diluted_shares",
            ],
            provenance=["PIT:DEBT_SCHEDULE:000004"],
        ).model_dump(mode="json"),
    )

    args = argparse.Namespace(
            dates=dates,
            universe=universe,
            base_root=base,
            output=output,
            valuation_input_root=valuation_inputs,
            valuation_profile_root=valuation_profiles,
            pit_sector_map=None,
            development_sector_map=sectors,
            mode="development",
            expected_date_count=1,
            expected_universe_count=4,
    )
    coverage = run(args)
    args.output = output_b
    coverage_b = run(args)

    assert coverage["valuation_generated_count"] == 4
    assert coverage["rank_eligible_count"] == 4
    assert coverage["actual_engine_counts"] == {
        "ApvEngine": 1,
        "CommonRimEngine": 1,
        "CommonRnpvEngine": 1,
        "NavEngine": 1,
    }
    assert coverage["fallback_fcff_count"] == 0
    assert coverage["llm_call_count"] == 0
    assert coverage_b == coverage
    assert all(row["rank_eligible_count"] == 1 for row in coverage["method_audit"].values())

    with (output / "signals.csv").open(encoding="utf-8-sig", newline="") as stream:
        signals = list(csv.DictReader(stream))
    assert {row["method"] for row in signals} == {"APV", "RIM", "NAV", "RNPV"}
    assert {row["actual_engine"] for row in signals} == {
        "ApvEngine",
        "CommonRimEngine",
        "CommonRnpvEngine",
        "NavEngine",
    }
    assert {row["unified_value_score"] for row in signals} == {"50.0"}
    assert all("::" in row["reference_class"] for row in signals)

    with (output / "routing.csv").open(encoding="utf-8-sig", newline="") as stream:
        routing = {row["ticker"]: row for row in csv.DictReader(stream)}
    assert routing["000004"]["profile_input_source"] == "EXPLICIT_PIT_VALUATION_PROFILE"

    contract = json.loads((output / "audit-contract.json").read_text(encoding="utf-8"))
    assert contract["cross_method_fallback_forbidden"] is True
    assert contract["llm_calls_for_routing_or_valuation"] == 0
    for name in ("routing.csv", "signals.csv", "coverage.json", "audit-contract.json"):
        assert (output / name).read_bytes() == (output_b / name).read_bytes()
