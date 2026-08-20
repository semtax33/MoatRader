from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from moatrader.valuation import (
    NormalizationContract,
    NormalizedAnnualObservation,
    NormalizedFcffBuildInput,
    NormalizedFcffBuilder,
    NormalizedFcffEngine,
    RoutedValuationInput,
    ValuationMethod,
    infer_cycle_phase,
)

try:
    from scripts.prepare_kr_dcf_manifest import annual_metrics
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from prepare_kr_dcf_manifest import annual_metrics  # type: ignore[no-redef]


BUILD_REPORT_VERSION = "normalized-fcff-input-build/1"


@dataclass(frozen=True)
class AnnualHistoryResult:
    observations: list[NormalizedAnnualObservation]
    excluded_fiscal_years: list[int]
    source_refs: list[str]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def decimal(value: Any, *, field: str) -> Decimal:
    try:
        result = Decimal(str(value))
    except Exception as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if not result.is_finite():
        raise ValueError(f"{field} must be finite")
    return result


def latest_snapshot_values(snapshot: dict[str, Any]) -> dict[str, Decimal]:
    result: dict[str, Decimal] = {}
    for item in snapshot.get("series") or []:
        concept = str(item.get("concept") or "")
        points = item.get("points") or []
        if not concept or not points:
            continue
        point = max(points, key=lambda value: str(value.get("period") or ""))
        try:
            result[concept] = decimal(point.get("value"), field=concept)
        except ValueError:
            continue
    return result


def _payload_path(
    *, dcf_path: Path, ticker: str, source: dict[str, Any]
) -> Path:
    declared = Path(str(source.get("payload_path") or ""))
    if declared.is_file():
        return declared
    filename = (
        f"{int(source['business_year'])}-{source['report_code']}-"
        f"{source['fs_div']}.json"
    )
    reconstructed = dcf_path.parent.parent / "source" / "financials" / ticker / filename
    if reconstructed.is_file():
        return reconstructed
    raise FileNotFoundError(f"annual source payload not found: {filename}")


def _latest_consecutive_run(
    observations: dict[int, NormalizedAnnualObservation],
    *,
    minimum_observations: int,
    window_years: int,
) -> list[NormalizedAnnualObservation]:
    years = sorted(observations)
    runs: list[list[int]] = []
    for year in years:
        if not runs or year != runs[-1][-1] + 1:
            runs.append([year])
        else:
            runs[-1].append(year)
    eligible = [run[-window_years:] for run in runs if len(run) >= minimum_observations]
    if not eligible:
        raise ValueError(
            f"no consecutive annual history with {minimum_observations}+ observations"
        )
    selected = max(eligible, key=lambda run: (run[-1], len(run)))
    return [observations[year] for year in selected]


def reconstruct_annual_history(
    dcf_input: dict[str, Any],
    *,
    dcf_path: Path,
    ticker: str,
    minimum_observations: int = 5,
    window_years: int = 7,
) -> AnnualHistoryResult:
    sources = sorted(
        dcf_input.get("annual_sources") or [],
        key=lambda item: (str(item.get("available_at") or ""), int(item["business_year"])),
    )
    if not sources:
        raise ValueError("annual_sources are missing")
    latest_year = max(int(item["business_year"]) for item in sources)
    first_year = latest_year - window_years + 1
    by_year: dict[int, NormalizedAnnualObservation] = {}
    used_refs: list[str] = []
    for source in sources:
        path = _payload_path(dcf_path=dcf_path, ticker=ticker, source=source)
        actual_hash = file_sha256(path)
        declared_hash = str(source.get("payload_sha256") or "")
        if declared_hash and actual_hash != declared_hash:
            raise ValueError(f"annual source hash mismatch: {path.name}")
        payload = read_json(path)
        if str(payload.get("status")) != "000" or not payload.get("list"):
            continue
        business_year = int(source["business_year"])
        for observation_year, amount_field in (
            (business_year - 2, "bfefrmtrm_amount"),
            (business_year - 1, "frmtrm_amount"),
            (business_year, "thstrm_amount"),
        ):
            if not first_year <= observation_year <= latest_year:
                continue
            metrics = annual_metrics(payload["list"], fields=(amount_field,))
            revenue = metrics.get("revenue")
            ebit = metrics.get("ebit")
            if revenue is None or revenue <= 0 or ebit is None:
                continue
            ref = (
                f"OPENDART:{source.get('receipt_no')}:{observation_year}:"
                f"{amount_field}:SHA256:{actual_hash}"
            )
            # Sources are filing-time sorted; later-filed comparatives supersede
            # older values and preserve PIT restatements.
            by_year[observation_year] = NormalizedAnnualObservation(
                fiscal_year=observation_year,
                revenue=revenue,
                ebit=ebit,
                source_refs=[ref],
            )
    selected = _latest_consecutive_run(
        by_year,
        minimum_observations=minimum_observations,
        window_years=window_years,
    )
    selected_years = {item.fiscal_year for item in selected}
    desired_years = set(range(latest_year - window_years + 1, latest_year + 1))
    for item in selected:
        used_refs.extend(item.source_refs)
    return AnnualHistoryResult(
        observations=selected,
        excluded_fiscal_years=sorted(desired_years - selected_years),
        source_refs=list(dict.fromkeys(used_refs)),
    )


def build_normalized_input(
    *,
    ticker: str,
    as_of: str,
    dcf_path: Path,
    snapshot_path: Path,
) -> RoutedValuationInput:
    dcf_input = read_json(dcf_path)
    snapshot = read_json(snapshot_path)
    history = reconstruct_annual_history(
        dcf_input,
        dcf_path=dcf_path,
        ticker=ticker,
    )
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
    phase = infer_cycle_phase(
        history.observations,
        current_ebit_margin=base_ebit / base_revenue,
    )
    contract = NormalizationContract(
        window_years=7,
        minimum_observations=5,
        included_fiscal_years=[item.fiscal_year for item in history.observations],
        excluded_fiscal_years=history.excluded_fiscal_years,
        cycle_phase=phase,
    )
    source_refs = list(
        dict.fromkeys(
            [
                f"PIT_DCF_INPUT:SHA256:{file_sha256(dcf_path)}",
                f"PIT_FINANCIAL_SNAPSHOT:SHA256:{file_sha256(snapshot_path)}",
                *history.source_refs,
                f"POLICY:{contract.contract_version}",
            ]
        )
    )
    build_source = NormalizedFcffBuildInput(
        issuer_id=str(snapshot.get("issuer_id") or ticker),
        as_of=as_of,
        observations=history.observations,
        normalization=contract,
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
        provenance=source_refs,
    )
    assumptions = NormalizedFcffBuilder().build(build_source)
    # Fail before persistence when real engine cannot produce an ordered value.
    NormalizedFcffEngine().value(assumptions)
    return RoutedValuationInput(
        issuer_id=build_source.issuer_id,
        as_of=as_of,
        method=ValuationMethod.NORMALIZED_FCFF,
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
        if row.get("primary_method") == ValuationMethod.NORMALIZED_FCFF.value
    ]
    generated: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    for row in rows:
        as_of = row["date"]
        ticker = row["ticker"].zfill(6)
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
            envelope = build_normalized_input(
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
        description="Build deterministic routed Normalized FCFF inputs from PIT OpenDART caches."
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
