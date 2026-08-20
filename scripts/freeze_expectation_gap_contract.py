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
    "src/moatrader/valuation/execution.py",
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


ENGINEERING_AUDIT_SCHEMA_VERSION = "expanded-valuation-signal-audit/4"
REQUIRED_ARCHITECTURE_METHODS = {
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
}


def validate_engineering_coverage(coverage: dict[str, object]) -> None:
    if coverage.get("schema_version") != ENGINEERING_AUDIT_SCHEMA_VERSION:
        raise ValueError("freeze requires routed valuation audit schema v4")
    if coverage.get("row_count") != 600 or coverage.get("pit_sector_count") != 600:
        raise ValueError("freeze requires a complete 600-row PIT-sector engineering audit")
    if coverage.get("return_data_accessed") is not False:
        raise ValueError("freeze requires a return-free engineering audit")
    if coverage.get("fallback_fcff_count") != 0:
        raise ValueError("freeze forbids fallback FCFF valuations")
    if coverage.get("llm_call_count") != 0:
        raise ValueError("routing and valuation freeze requires zero LLM calls")
    if coverage.get("route_actual_engine_match_rate") != 1.0:
        raise ValueError("freeze requires 100% route-to-actual-engine match")
    if coverage.get("architecture_gate_pass") is not True:
        raise ValueError("freeze requires a passing architecture calibration gate")
    if float(coverage.get("route_stability", 0.0)) < 0.90:
        raise ValueError("freeze requires route stability of at least 90%")
    normalization = coverage.get("normalization_policy")
    if not isinstance(normalization, dict):
        raise ValueError("freeze requires a normalization policy manifest")
    if normalization.get("min_reference_class_size") != 20:
        raise ValueError("freeze requires minimum reference class N=20")
    if normalization.get("parent_class_fallback") is not False:
        raise ValueError("freeze forbids parent reference-class fallback")
    method_audit = coverage.get("method_audit")
    if not isinstance(method_audit, dict) or not method_audit:
        raise ValueError("freeze requires per-method route/generation/trust audit")
    execution_gaps: list[str] = []
    rank_gaps: list[str] = []
    routed_total = 0
    generated_total = 0
    trusted_total = 0
    for method, raw in method_audit.items():
        if not isinstance(raw, dict):
            dead_routes.append(str(method))
            continue
        routed = int(raw.get("routed_count", 0))
        eligible = int(raw.get("eligible_route_count", 0))
        generated = int(raw.get("valuation_generated_count", 0))
        trusted = int(raw.get("rank_eligible_count", 0))
        if not 0 <= trusted <= generated <= eligible <= routed:
            raise ValueError(f"invalid route coverage ordering for {method}")
        routed_total += routed
        generated_total += generated
        trusted_total += trusted
        if generated != eligible:
            execution_gaps.append(str(method))
        if int(raw.get("max_reference_class_size", 0)) >= 20 and trusted == 0:
            rank_gaps.append(str(method))
    if execution_gaps:
        raise ValueError(
            "freeze forbids eligible route execution gaps: "
            + ",".join(sorted(execution_gaps))
        )
    if rank_gaps:
        raise ValueError(
            "freeze requires trusted values for reference classes at N=20: "
            + ",".join(sorted(rank_gaps))
        )
    if set(method_audit) != REQUIRED_ARCHITECTURE_METHODS:
        raise ValueError("freeze requires all eight routed valuation methods")
    if routed_total != int(coverage["row_count"]):
        raise ValueError("per-method routed counts must sum to audit row_count")
    if generated_total != int(coverage.get("valuation_generated_count", -1)):
        raise ValueError("per-method generated counts must match valuation_generated_count")
    if trusted_total != int(coverage.get("rank_eligible_count", -1)):
        raise ValueError("per-method trusted counts must match rank_eligible_count")
    engine_counts = coverage.get("actual_engine_counts")
    if not isinstance(engine_counts, dict) or len(engine_counts) < 3:
        raise ValueError("freeze requires at least three distinct executed valuation engines")
    total = sum(int(value) for value in engine_counts.values())
    if total != generated_total:
        raise ValueError("actual engine counts must match valuation_generated_count")
    dominant = max((int(value) for value in engine_counts.values()), default=0)
    if total <= 0 or dominant / total > 0.90:
        raise ValueError("freeze forbids a single valuation engine exceeding 90% of generated values")


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
    validate_engineering_coverage(coverage)
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
