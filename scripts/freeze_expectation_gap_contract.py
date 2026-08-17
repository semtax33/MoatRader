from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

from moatrader.expectations import FrozenRiskOverlayPolicy
from moatrader.experiments import FrozenExpectationGapContract, compute_contract_sha256
from moatrader.valuation import ROUTER_CONTRACT_VERSION, ValuationMethod


FROZEN_SOURCES = (
    "src/moatrader/canonical/models.py",
    "src/moatrader/business/drivers.py",
    "src/moatrader/evidence/models.py",
    "src/moatrader/expectations/__init__.py",
    "src/moatrader/expectations/alpha.py",
    "src/moatrader/expectations/holdout.py",
    "src/moatrader/expectations/risk.py",
    "src/moatrader/expectations/scoring.py",
    "src/moatrader/experiments/contract.py",
    "src/moatrader/ingestion/krx.py",
    "src/moatrader/financial/dcf.py",
    "src/moatrader/valuation/base.py",
    "src/moatrader/valuation/__init__.py",
    "src/moatrader/valuation/assumptions.py",
    "src/moatrader/valuation/biotech_rnpv.py",
    "src/moatrader/valuation/economic_dcf.py",
    "src/moatrader/valuation/profile.py",
    "src/moatrader/valuation/router.py",
    "src/moatrader/valuation/rim.py",
    "src/moatrader/valuation/common_engines.py",
    "src/moatrader/valuation/scenario_dcf.py",
    "src/moatrader/valuation/apv.py",
    "src/moatrader/valuation/nav.py",
    "src/moatrader/valuation/sotp.py",
    "src/moatrader/valuation/three_p.py",
    "src/moatrader/valuation/legacy_fcff_adapter.py",
    "src/moatrader/financial/pit_sector.py",
    "scripts/audit_expanded_valuation_signals.py",
    "scripts/build_expectation_gap_holdout_signals.py",
    "scripts/build_expectation_gap_research_inputs.py",
    "scripts/collect_krx_pit_sectors.py",
    "scripts/seal_expectation_gap_holdout.py",
    "scripts/evaluate_frozen_expectation_gap_holdout.py",
    "scripts/preflight_expectation_gap_holdout.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the return-blind Expectation GAP holdout contract.")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--development-dates", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--stability-a", type=Path, required=True)
    parser.add_argument("--stability-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-on", type=date.fromisoformat, required=True)
    parser.add_argument("--holdout-dates", required=True, help="comma-separated ISO dates")
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"frozen contract already exists: {output}")
    comparable = ("routing.csv", "signals.csv", "coverage.json", "audit-contract.json")
    stability_hashes: dict[str, str] = {}
    for name in comparable:
        left = args.stability_a.resolve() / name
        right = args.stability_b.resolve() / name
        left_hash = sha256(left)
        right_hash = sha256(right)
        if left_hash != right_hash:
            raise ValueError(f"engineering stability mismatch for {name}")
        stability_hashes[name] = left_hash
    coverage = read_json(args.stability_a.resolve() / "coverage.json")
    if coverage.get("row_count") != 600 or coverage.get("pit_sector_count") != 600:
        raise ValueError("freeze requires a complete 600-row PIT-sector engineering audit")
    if coverage.get("return_data_accessed") is not False:
        raise ValueError("freeze requires a return-free engineering audit")
    with args.development_dates.resolve().open("r", encoding="utf-8-sig", newline="") as stream:
        development = [date.fromisoformat(str(next(iter(row.values()))).strip()) for row in csv.DictReader(stream)]
    holdout = [date.fromisoformat(item.strip()) for item in args.holdout_dates.split(",") if item.strip()]
    if len(holdout) != 4 or len(set(holdout)) != 4:
        raise ValueError("production holdout requires four unique dates")
    if any(item <= args.frozen_on for item in holdout):
        raise ValueError("prospective holdout dates must be after the freeze date")
    with args.universe.resolve().open("r", encoding="utf-8-sig", newline="") as stream:
        universe_count = sum(1 for _ in csv.DictReader(stream))
    payload: dict[str, object] = {
        "schema_version": "expectation-gap-production-candidate/1",
        "frozen_on": args.frozen_on.isoformat(),
        "development_dates": [item.isoformat() for item in development],
        "holdout_dates": [item.isoformat() for item in holdout],
        "universe_sha256": sha256(args.universe.resolve()),
        "universe_count": universe_count,
        "valuation_methods": [
            method.value
            for method in (
                ValuationMethod.ECONOMIC_FCFF,
                ValuationMethod.NORMALIZED_FCFF,
                ValuationMethod.RIM,
                ValuationMethod.RNPV,
                ValuationMethod.SCENARIO_DCF,
                ValuationMethod.APV,
                ValuationMethod.NAV,
                ValuationMethod.SOTP,
            )
        ],
        "router_contract_version": ROUTER_CONTRACT_VERSION,
        "cheap_definition": "primary_method_fair_value_per_share / PIT_market_price - 1",
        "percentile_cohort": "signal_date x valuation_method x economic_archetype",
        "risk_policy": FrozenRiskOverlayPolicy().model_dump(mode="json"),
        "legacy_composite_role": "BENCHMARK_DIAGNOSTIC_ONLY_NOT_RANK_ELIGIBLE",
        "improving_role": "THESIS_CONFIRMATION_DIAGNOSTIC_NOT_WEIGHTED",
        "sector_policy": "OFFICIAL_KRX_MDCSTAT03901_SNAPSHOT_AVAILABLE_BY_SIGNAL_DATE_NO_CURRENT_FALLBACK",
        "source_cutoff_policy": "ALL_FINANCIAL_AND_EVIDENCE_AVAILABLE_AT_MUST_BE_LTE_SIGNAL_AS_OF",
        "signal_seal_required": True,
        "return_inputs_forbidden_before_signal_seal": True,
        "forward_return_calendar_days": 77,
        "maximum_sector_neutral_ic_sacrifice": 0.05,
        "minimum_worst_decile_improvement": 0.03,
        "minimum_downside_capture_improvement": 0.10,
        "frozen_source_sha256": {relative: sha256(root / relative) for relative in FROZEN_SOURCES},
        "engineering_stability_sha256": stability_hashes,
        "engineering_return_data_accessed": False,
    }
    payload["contract_sha256"] = compute_contract_sha256(payload)
    contract = FrozenExpectationGapContract.model_validate(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(contract.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    dates_path = output.with_name("holdout-dates.csv")
    dates_path.write_text("as_of\n" + "".join(f"{item.isoformat()}\n" for item in holdout), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
