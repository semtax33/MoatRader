from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import date, datetime, time as datetime_time
from decimal import Decimal
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from moatrader.expectations import CheapSignal, assign_method_archetype_percentiles
from moatrader.financial.dcf import DcfAssumptions
from moatrader.financial.pit_sector import PitSectorRecord, load_pit_sector_csv, resolve_pit_sector
from moatrader.valuation import (
    CommonRimEngine,
    EconomicArchetype,
    ExecutionStatus,
    LegacyFcffCommonEngine,
    LegacyFcffScenarioSet,
    PreparedValuationInput,
    RimAssumptions,
    RimScenarioSet,
    RoutedValuationExecutor,
    ValuationMethod,
    ValuationProfile,
    ValuationProfileRouter,
    stress_legacy_fcff,
)


SEOUL = ZoneInfo("Asia/Seoul")
SCHEMA_VERSION = "expanded-valuation-signal-audit/2"
FORBIDDEN_INPUT_NAMES = {"returns.csv", "forward-returns.csv", "evaluation.json"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_payload(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def number(value: Any) -> Decimal | None:
    try:
        result = Decimal(str(value))
    except Exception:
        return None
    return result if result.is_finite() else None


def truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def ticker(value: Any) -> str:
    return str(value or "").strip().zfill(6)


def as_of_datetime(value: str) -> datetime:
    return datetime.combine(date.fromisoformat(value), datetime_time(23, 59, 59), tzinfo=SEOUL)


def _series(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for series in snapshot.get("series", []):
        points = series.get("points") or []
        if points:
            result[str(series["concept"])] = max(points, key=lambda item: str(item.get("period") or ""))
    return result


def _annualized(point: dict[str, Any] | None) -> Decimal | None:
    if not point:
        return None
    value = number(point.get("value"))
    if value is None:
        return None
    basis = str(point.get("period_basis") or "").upper()
    multiplier = Decimal(4) if basis == "Q1" else Decimal(2) if basis in {"H1", "Q2"} else Decimal(4) / Decimal(3) if basis == "Q3" else Decimal(1)
    return value * multiplier


def _current_sector_map(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    return {ticker(row.get("ticker")): row for row in read_csv(path)}


def _sector_for(
    *,
    code: str,
    as_of: datetime,
    pit_records: list[PitSectorRecord],
    development: dict[str, dict[str, str]],
    mode: str,
) -> tuple[str, str, bool, str]:
    record = resolve_pit_sector(pit_records, ticker=code, as_of=as_of)
    if record is not None:
        return record.sector, record.industry_code or "", True, record.evidence_ref
    if mode == "pit_strict":
        raise ValueError(f"PIT-strict audit lacks sector classification for {code} at {as_of.date()}")
    fallback = development.get(code, {})
    return (
        str(fallback.get("sector") or "UNKNOWN"),
        str(fallback.get("industry_code") or ""),
        False,
        "NON_PIT_DEVELOPMENT_CURRENT_CLASSIFICATION",
    )


def _profile(
    *,
    code: str,
    as_of: str,
    universe: dict[str, str],
    sector: str,
    sector_evidence: str,
    snapshot: dict[str, Any],
    dcf_input: dict[str, Any] | None,
    prepared_input: PreparedValuationInput | None = None,
) -> tuple[ValuationProfile, dict[str, Decimal | None]]:
    series = _series(snapshot)
    revenue = _annualized(series.get("REVENUE"))
    ebit = _annualized(series.get("EBIT"))
    net_income = _annualized(series.get("NET_INCOME"))
    equity = number((series.get("TOTAL_EQUITY") or {}).get("value"))
    assets = number((series.get("TOTAL_ASSETS") or {}).get("value"))
    debt = number((series.get("TOTAL_DEBT") or {}).get("value"))
    cash = number((series.get("CASH") or {}).get("value"))
    shares = number(universe.get("listed_shares"))
    metrics = (dcf_input or {}).get("metrics", {})
    if revenue is None:
        revenue = number(metrics.get("revenue"))
    if ebit is None:
        ebit = number(metrics.get("ebit"))
    available: set[str] = set()
    if revenue is not None:
        available.add("revenue")
    if ebit is not None:
        available.add("ebit")
    if shares is not None and shares > 0:
        available.add("diluted_shares")
    if equity is not None and equity > 0:
        available.add("book_equity")
    if net_income is not None:
        available.add("net_income")
    if equity is not None and debt is not None and cash is not None and equity + debt - cash > 0:
        available.add("invested_capital")
    history = (dcf_input or {}).get("annual_history") or []
    if len(history) >= 3:
        available.update({"revenue_history", "margin_history"})
    if dcf_input is not None:
        available.update({"scenario_assumptions", "valuation_assumptions"})
    if prepared_input is not None:
        available.update(prepared_input.available_data)

    normalized_sector = sector.replace(" ", "")
    finance = truthy(universe.get("finance_hint")) or any(
        term in normalized_sector
        for term in ("은행", "증권", "보험", "Banks", "Securities", "Insurance")
    )
    holding = truthy(universe.get("holding_hint"))
    biotech_prior = any(
        term in normalized_sector
        for term in (
            "제약",
            "생물공학",
            "건강관리",
            "Pharmaceuticals",
            "Biotechnology",
            "HealthCare",
        )
    ) or "바이오" in str(universe.get("name") or "")
    pre_revenue_like = (
        biotech_prior
        and ebit is not None
        and ebit < 0
        and revenue is not None
        and assets is not None
        and assets > 0
        and revenue / assets < Decimal("0.02")
    )
    asset_primary = any(
        term in normalized_sector
        for term in ("리츠", "부동산", "광업", "석유와가스", "REIT", "RealEstate", "Mining", "Oil&Gas")
    )
    cyclical = any(
        term in normalized_sector
        for term in (
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
            "TransportationEquipment",
            "Construction",
            "Machinery",
        )
    )
    if finance:
        archetype = EconomicArchetype.FINANCIAL_INTERMEDIARY
    elif holding:
        archetype = EconomicArchetype.MULTI_BUSINESS
    elif pre_revenue_like:
        archetype = EconomicArchetype.PRE_REVENUE_BIOTECH
    elif asset_primary:
        archetype = EconomicArchetype.ASSET_BACKED
    elif ebit is not None and ebit < 0:
        archetype = EconomicArchetype.LOSS_MAKING_GROWTH
    elif cyclical:
        archetype = EconomicArchetype.CYCLICAL_OPERATING
    else:
        archetype = EconomicArchetype.GENERAL_OPERATING
    profile = ValuationProfile(
        issuer_id=str(snapshot.get("issuer_id") or code),
        as_of=date.fromisoformat(as_of),
        sector=sector or "UNKNOWN",
        industry=str(universe.get("name") or "UNKNOWN"),
        economic_archetype=archetype,
        is_financial_intermediary=finance,
        is_reit=asset_primary and any(term in normalized_sector for term in ("리츠", "REIT", "RealEstate")),
        is_resource_company=asset_primary and any(
            term in normalized_sector for term in ("광업", "석유와가스", "Mining", "Oil&Gas")
        ),
        revenue_positive=revenue > 0 if revenue is not None else None,
        ebit_positive=ebit > 0 if ebit is not None else None,
        fcf_positive=None,
        pipeline_assets_material=pre_revenue_like,
        multi_segment=holding,
        segment_heterogeneity_material=holding,
        asset_value_primary=asset_primary,
        materially_cyclical=cyclical,
        available_data=sorted(available),
        provenance=[
            f"PIT:financial-snapshot:{as_of}:{code}",
            sector_evidence,
            "PIT:universe-listed-shares:2025-08-01",
            *(prepared_input.envelope.source_refs if prepared_input else []),
        ],
    )
    return profile, {
        "revenue": revenue,
        "ebit": ebit,
        "net_income": net_income,
        "book_equity": equity,
        "shares": shares,
    }


def _rim_assumptions(
    *,
    book_equity: Decimal,
    net_income: Decimal,
    shares: Decimal,
    size_bucket: str,
    scenario_shift: Decimal,
    provenance: list[str],
) -> RimAssumptions:
    cost = Decimal("0.09") if size_bucket == "LARGE" else Decimal("0.10") if size_bucket == "MID" else Decimal("0.12")
    roe = net_income / book_equity + scenario_shift
    if abs(roe) > Decimal("0.50"):
        raise ValueError("RIM_ROE_OUTSIDE_ENGINEERING_BOUND")
    terminal_roe = cost + (roe - cost) * Decimal("0.25")
    path = [roe + (terminal_roe - roe) * Decimal(year) / Decimal(5) for year in range(1, 6)]
    return RimAssumptions(
        book_equity=book_equity,
        roe_path=path,
        cost_of_equity=cost,
        payout_ratio=Decimal("0.40"),
        terminal_roe=terminal_roe,
        terminal_growth=Decimal("0.02"),
        diluted_shares=shares,
        assumption_confidence=Decimal("0.50"),
        provenance=provenance + ["POLICY:RIM_ENGINEERING_V1"],
    )


def _valuation_for(
    *,
    route,
    metrics: dict[str, Decimal | None],
    universe: dict[str, str],
    dcf_input: dict[str, Any] | None,
    provenance: list[str],
    prepared_input: PreparedValuationInput | None,
    input_error: str,
):
    method = route.primary_method
    if input_error:
        return (
            None,
            input_error,
            "",
            "INVALID_EXPLICIT_METHOD_INPUT",
            ExecutionStatus.INVALID_METHOD_INPUT.value,
        )
    if prepared_input is not None:
        execution = RoutedValuationExecutor().execute(route, prepared_input)
        return (
            execution.valuation,
            ";".join(execution.reason_codes),
            execution.actual_engine or "",
            "EXPLICIT_ROUTED_INPUT",
            execution.status.value,
        )
    if method == ValuationMethod.RIM:
        book = metrics["book_equity"]
        income = metrics["net_income"]
        shares = metrics["shares"]
        if book is None or income is None or shares is None:
            return None, "MISSING_RIM_INPUT", "", "DERIVED_PIT_FINANCIALS", ExecutionStatus.MISSING_METHOD_INPUT.value
        try:
            scenarios = [
                _rim_assumptions(
                    book_equity=book,
                    net_income=income,
                    shares=shares,
                    size_bucket=str(universe.get("size_bucket") or "SMALL"),
                    scenario_shift=shift,
                    provenance=provenance,
                )
                for shift in (Decimal("-0.03"), Decimal(0), Decimal("0.03"))
            ]
            return CommonRimEngine().value(
                RimScenarioSet(downside=scenarios[0], base=scenarios[1], upside=scenarios[2])
            ), "", "CommonRimEngine", "DERIVED_PIT_FINANCIALS", ExecutionStatus.VALUED.value
        except Exception as exc:
            return None, f"RIM_ERROR:{type(exc).__name__}:{exc}", "CommonRimEngine", "DERIVED_PIT_FINANCIALS", ExecutionStatus.VALUATION_ERROR.value
    if method == ValuationMethod.ECONOMIC_FCFF:
        if dcf_input is None:
            return None, "MISSING_FCFF_INPUT", "", "LEGACY_PIT_TTM", ExecutionStatus.MISSING_METHOD_INPUT.value
        try:
            base = DcfAssumptions.model_validate(dcf_input["assumptions"])
            return LegacyFcffCommonEngine().value(
                LegacyFcffScenarioSet(
                    downside=stress_legacy_fcff(base, direction=-1),
                    base=base,
                    upside=stress_legacy_fcff(base, direction=1),
                    method=method,
                    provenance=provenance + ["ADAPTER:LEGACY_PIT_TTM"],
                )
            ), "", "LegacyFcffCommonEngine", "LEGACY_PIT_TTM", ExecutionStatus.VALUED.value
        except Exception as exc:
            return None, f"FCFF_ERROR:{type(exc).__name__}:{exc}", "LegacyFcffCommonEngine", "LEGACY_PIT_TTM", ExecutionStatus.VALUATION_ERROR.value
    return None, f"METHOD_INPUT_NOT_AVAILABLE:{method.value}", "", "NO_FALLBACK", ExecutionStatus.MISSING_METHOD_INPUT.value


def run(args: argparse.Namespace) -> dict[str, Any]:
    for path in (
        args.dates,
        args.universe,
        args.base_root,
        args.output,
        getattr(args, "valuation_input_root", None),
        getattr(args, "valuation_profile_root", None),
    ):
        if path is None:
            continue
        if path.name.lower() in FORBIDDEN_INPUT_NAMES:
            raise ValueError(f"return/evaluation input is forbidden before contract freeze: {path}")
    dates = [str(next(iter(row.values()))).strip() for row in read_csv(args.dates)]
    universe_rows = read_csv(args.universe)
    if len(dates) != args.expected_date_count or len(universe_rows) != args.expected_universe_count:
        raise ValueError(
            f"expected {args.expected_date_count} dates x {args.expected_universe_count} stocks, "
            f"got {len(dates)} x {len(universe_rows)}"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"audit output must be new and empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    pit_records = load_pit_sector_csv(args.pit_sector_map) if args.pit_sector_map else []
    development = _current_sector_map(args.development_sector_map)
    universe = {ticker(row.get("stock_code")): row for row in universe_rows}
    router = ValuationProfileRouter()
    routing_rows: list[dict[str, Any]] = []
    signal_objects: dict[str, list[tuple[int, CheapSignal]]] = {}
    signals: list[dict[str, Any]] = []
    for as_of in dates:
        cutoff = as_of_datetime(as_of)
        for code, universe_row in sorted(universe.items()):
            sector, industry_code, pit_eligible, sector_evidence = _sector_for(
                code=code,
                as_of=cutoff,
                pit_records=pit_records,
                development=development,
                mode=args.mode,
            )
            snapshot_path = args.base_root / "runs" / f"kr-signal-{as_of}" / "companies" / code / "financial-snapshot.json"
            dcf_path = args.base_root / "date-inputs" / as_of / "dcf-inputs" / f"{code}.json"
            valuation_input_root = getattr(args, "valuation_input_root", None)
            valuation_input_path = (
                valuation_input_root / as_of / f"{code}.json"
                if valuation_input_root
                else args.base_root / "date-inputs" / as_of / "valuation-inputs" / f"{code}.json"
            )
            valuation_profile_root = getattr(args, "valuation_profile_root", None)
            valuation_profile_path = (
                valuation_profile_root / as_of / f"{code}.json"
                if valuation_profile_root
                else args.base_root / "date-inputs" / as_of / "valuation-profiles" / f"{code}.json"
            )
            manifest_path = args.base_root / "date-inputs" / as_of / "universe-manifest.csv"
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8-sig")) if snapshot_path.is_file() else {"issuer_id": code, "series": []}
            dcf_input = json.loads(dcf_path.read_text(encoding="utf-8-sig")) if dcf_path.is_file() else None
            prepared_input = None
            input_error = ""
            if valuation_input_path.is_file():
                try:
                    prepared_input = RoutedValuationExecutor.prepare(
                        json.loads(valuation_input_path.read_text(encoding="utf-8-sig"))
                    )
                except Exception as exc:
                    input_error = f"INVALID_METHOD_INPUT:{type(exc).__name__}:{exc}"
            heuristic_profile, metrics = _profile(
                code=code,
                as_of=as_of,
                universe=universe_row,
                sector=sector,
                sector_evidence=sector_evidence,
                snapshot=snapshot,
                dcf_input=dcf_input,
                prepared_input=prepared_input,
            )
            profile = heuristic_profile
            profile_input_source = "DETERMINISTIC_SNAPSHOT_HEURISTIC"
            if valuation_profile_path.is_file():
                try:
                    explicit_profile = ValuationProfile.model_validate(
                        json.loads(valuation_profile_path.read_text(encoding="utf-8-sig"))
                    )
                except Exception as exc:
                    raise ValueError(
                        f"INVALID_ROUTING_PROFILE:{as_of}:{code}:{type(exc).__name__}:{exc}"
                    ) from exc
                if explicit_profile.issuer_id != heuristic_profile.issuer_id:
                    raise ValueError(f"ROUTING_PROFILE_ISSUER_MISMATCH:{as_of}:{code}")
                if explicit_profile.as_of != date.fromisoformat(as_of):
                    raise ValueError(f"ROUTING_PROFILE_AS_OF_MISMATCH:{as_of}:{code}")
                profile_payload = explicit_profile.model_dump()
                profile_payload["available_data"] = sorted(
                    set(explicit_profile.available_data)
                    | set(prepared_input.available_data if prepared_input else [])
                )
                profile_payload["provenance"] = list(
                    dict.fromkeys(
                        explicit_profile.provenance
                        + (prepared_input.envelope.source_refs if prepared_input else [])
                    )
                )
                profile = ValuationProfile.model_validate(profile_payload)
                profile_input_source = "EXPLICIT_PIT_VALUATION_PROFILE"
            route = router.route(profile)
            valuation, valuation_reason, actual_engine, valuation_input_source, execution_status = _valuation_for(
                route=route,
                metrics=metrics,
                universe=universe_row,
                dcf_input=dcf_input,
                provenance=profile.provenance,
                prepared_input=prepared_input,
                input_error=input_error,
            ) if input_error or route.applicability.status.value == "ELIGIBLE" else (
                None,
                ";".join(route.applicability.reason_codes),
                "",
                "ROUTE_NOT_APPLICABLE",
                ExecutionStatus.ROUTE_NOT_APPLICABLE.value,
            )
            price = None
            if manifest_path.is_file():
                for item in read_csv(manifest_path):
                    if ticker(item.get("ticker")) == code:
                        price = number(item.get("current_price"))
                        break
            cheap = None
            if valuation is not None and price is not None and price > 0:
                try:
                    cheap = CheapSignal.from_valuation(
                        valuation=valuation,
                        economic_archetype=profile.economic_archetype.value,
                        market_price=price,
                    )
                except Exception as exc:
                    valuation_reason = f"CHEAP_ERROR:{type(exc).__name__}:{exc}"
            index = len(signals)
            signals.append(
                {
                    "date": as_of,
                    "ticker": code,
                    "method": route.primary_method.value,
                    "actual_engine": actual_engine,
                    "economic_archetype": profile.economic_archetype.value,
                    "sector": sector,
                    "sector_pit_eligible": int(pit_eligible),
                    "market_price": str(cheap.market_price) if cheap and cheap.market_price is not None else "",
                    "primary_fair_value_per_share": (
                        str(cheap.primary_fair_value_per_share)
                        if cheap and cheap.primary_fair_value_per_share is not None
                        else ""
                    ),
                    "raw_expectation_gap": str(cheap.raw_expectation_gap) if cheap else "",
                    "method_percentile": "",
                    "method_archetype_percentile": "",
                    "unified_value_score": "",
                    "reference_class": cheap.reference_class if cheap else f"{route.primary_method.value}::{profile.economic_archetype.value}",
                    "rank_eligible": int(bool(cheap and cheap.rank_eligible)),
                    "alpha_status": cheap.status.value if cheap else "MODEL_NOT_APPLICABLE",
                    "downside_value_per_share": (
                        str(valuation.downside_value_per_share) if valuation else ""
                    ),
                    "base_value_per_share": (
                        str(valuation.base_value_per_share) if valuation else ""
                    ),
                    "upside_value_per_share": (
                        str(valuation.upside_value_per_share) if valuation else ""
                    ),
                    "assumption_confidence": (
                        str(valuation.assumption_confidence)
                        if valuation and valuation.assumption_confidence is not None
                        else ""
                    ),
                    "valuation_warning_count": len(valuation.warnings) if valuation else "",
                    "trust_reason_codes": ";".join(cheap.trust_reason_codes) if cheap else "",
                    "possible_pass": int(bool(cheap and cheap.rank_eligible)),
                    "valuation_reason": valuation_reason,
                    "valuation_input_source": valuation_input_source,
                }
            )
            if cheap is not None:
                signal_objects.setdefault(as_of, []).append((index, cheap))
            routing_rows.append(
                {
                    "date": as_of,
                    "ticker": code,
                    "issuer_name": universe_row.get("name", ""),
                    "sector": sector,
                    "industry_code": industry_code,
                    "sector_pit_eligible": int(pit_eligible),
                    "sector_evidence_ref": sector_evidence,
                    "economic_archetype": profile.economic_archetype.value,
                    "primary_method": route.primary_method.value,
                    "secondary_method": route.secondary_method.value if route.secondary_method else "",
                    "applicability_status": route.applicability.status.value,
                    "missing_fields": ";".join(route.applicability.missing_fields),
                    "profile_sha256": route.profile_sha256,
                    "profile_input_source": profile_input_source,
                    "valuation_generated": int(valuation is not None),
                    "execution_status": execution_status,
                    "actual_engine": actual_engine,
                    "valuation_input_source": valuation_input_source,
                    "valuation_input_refs": ";".join(
                        prepared_input.envelope.source_refs if prepared_input else []
                    ),
                    "valuation_reason": valuation_reason,
                }
            )
    for pairs in signal_objects.values():
        normalized = assign_method_archetype_percentiles([item[1] for item in pairs])
        for (index, _), cheap in zip(pairs, normalized, strict=True):
            signals[index]["method_percentile"] = cheap.method_percentile
            signals[index]["method_archetype_percentile"] = cheap.method_archetype_percentile
            signals[index]["unified_value_score"] = cheap.unified_value_score
            signals[index]["reference_class"] = cheap.reference_class
    routing_rows.sort(key=lambda row: (row["date"], row["ticker"]))
    signals.sort(key=lambda row: (row["date"], row["ticker"]))
    write_csv(
        args.output / "routing.csv",
        routing_rows,
        [
            "date", "ticker", "issuer_name", "sector", "industry_code",
            "sector_pit_eligible", "sector_evidence_ref", "economic_archetype", "primary_method", "secondary_method",
            "applicability_status", "missing_fields", "profile_sha256", "profile_input_source",
            "valuation_generated",
            "execution_status", "actual_engine", "valuation_input_source", "valuation_input_refs",
            "valuation_reason",
        ],
    )
    write_csv(
        args.output / "signals.csv",
        signals,
        [
            "date", "ticker", "method", "actual_engine", "economic_archetype", "sector",
            "sector_pit_eligible", "market_price", "primary_fair_value_per_share",
            "raw_expectation_gap", "method_percentile", "method_archetype_percentile",
            "unified_value_score", "reference_class",
            "rank_eligible", "alpha_status", "downside_value_per_share",
            "base_value_per_share", "upside_value_per_share", "assumption_confidence",
            "valuation_warning_count", "trust_reason_codes", "possible_pass", "valuation_reason",
            "valuation_input_source",
        ],
    )
    method_audit: dict[str, dict[str, int | float]] = {}
    for method in sorted({row["primary_method"] for row in routing_rows}):
        routed = [row for row in routing_rows if row["primary_method"] == method]
        method_signals = [row for row in signals if row["method"] == method]
        generated = sum(int(row["valuation_generated"]) for row in routed)
        trusted = sum(int(row["rank_eligible"]) for row in method_signals)
        method_audit[method] = {
            "routed_count": len(routed),
            "route_share": len(routed) / len(routing_rows) if routing_rows else 0.0,
            "valuation_generated_count": generated,
            "rank_eligible_count": trusted,
            "trusted_route_share": trusted / len(routed) if routed else 0.0,
        }
    coverage = {
        "schema_version": SCHEMA_VERSION,
        "row_count": len(routing_rows),
        "return_data_accessed": False,
        "mode": args.mode,
        "method_counts": dict(sorted(Counter(row["primary_method"] for row in routing_rows).items())),
        "applicability_counts": dict(sorted(Counter(row["applicability_status"] for row in routing_rows).items())),
        "valuation_generated_count": sum(row["valuation_generated"] for row in routing_rows),
        "rank_eligible_count": sum(row["rank_eligible"] for row in signals),
        "method_audit": method_audit,
        "actual_engine_counts": dict(
            sorted(
                Counter(
                    row["actual_engine"]
                    for row in routing_rows
                    if row["actual_engine"] and row["valuation_generated"]
                ).items()
            )
        ),
        "engine_attempt_counts": dict(
            sorted(Counter(row["actual_engine"] for row in routing_rows if row["actual_engine"]).items())
        ),
        "fallback_fcff_count": sum(
            row["valuation_input_source"] == "FALLBACK_FCFF" for row in routing_rows
        ),
        "llm_call_count": 0,
        "pit_sector_count": sum(row["sector_pit_eligible"] for row in routing_rows),
        "routing_payload_sha256": sha256_payload(routing_rows),
        "signal_payload_sha256": sha256_payload(signals),
    }
    write_json(args.output / "coverage.json", coverage)
    write_json(
        args.output / "audit-contract.json",
        {
            "schema_version": SCHEMA_VERSION,
            "dates_sha256": sha256_file(args.dates),
            "universe_sha256": sha256_file(args.universe),
            "pit_sector_map_sha256": sha256_file(args.pit_sector_map) if args.pit_sector_map else None,
            "development_sector_map_sha256": sha256_file(args.development_sector_map) if args.development_sector_map else None,
            "return_inputs_forbidden": True,
            "price_used_only_after_route": True,
            "route_selected_before_valuation": True,
            "cross_method_fallback_forbidden": True,
            "llm_calls_for_routing_or_valuation": 0,
            "expected_date_count": args.expected_date_count,
            "expected_universe_count": args.expected_universe_count,
        },
    )
    return coverage


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit expanded valuation routing and Cheap stability without returns.")
    parser.add_argument("--dates", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--valuation-input-root",
        type=Path,
        help="Optional root containing <date>/<ticker>.json routed valuation inputs.",
    )
    parser.add_argument(
        "--valuation-profile-root",
        type=Path,
        help="Optional root containing price-free <date>/<ticker>.json ValuationProfile files.",
    )
    parser.add_argument("--pit-sector-map", type=Path)
    parser.add_argument("--development-sector-map", type=Path)
    parser.add_argument("--mode", choices=("development", "pit_strict"), default="development")
    parser.add_argument("--expected-date-count", type=int, default=4)
    parser.add_argument("--expected-universe-count", type=int, default=150)
    args = parser.parse_args()
    for name in (
        "dates",
        "universe",
        "base_root",
        "output",
        "valuation_input_root",
        "valuation_profile_root",
        "pit_sector_map",
        "development_sector_map",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    coverage = run(args)
    print(json.dumps(coverage, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
