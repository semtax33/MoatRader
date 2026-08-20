from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from moatrader.expectations import UnifiedValueNormalizationPolicy, ValuationTrustPolicy
from moatrader.valuation import (
    ASSUMPTION_POLICY_VERSION,
    ROUTED_VALUATION_INPUT_VERSION,
    ROUTER_CONTRACT_VERSION,
)
try:
    from scripts.audit_expanded_valuation_signals import (
        BENCHMARK_POLICY_VERSION,
        run as run_architecture_audit,
    )
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from audit_expanded_valuation_signals import (  # type: ignore[no-redef]
        BENCHMARK_POLICY_VERSION,
        run as run_architecture_audit,
    )


EXPERIMENT_SCHEMA_VERSION = "unified-value-architecture-calibration/4"
COMPARABLE_ARTIFACTS = (
    "routing.csv",
    "signals.csv",
    "coverage.json",
    "audit-contract.json",
)
FROZEN_CODE = (
    "src/moatrader/expectations/alpha.py",
    "src/moatrader/valuation/base.py",
    "src/moatrader/valuation/profile.py",
    "src/moatrader/valuation/router.py",
    "src/moatrader/valuation/execution.py",
    "src/moatrader/valuation/common_engines.py",
    "src/moatrader/valuation/legacy_fcff_adapter.py",
    "src/moatrader/valuation/rim.py",
    "src/moatrader/valuation/biotech_rnpv.py",
    "src/moatrader/valuation/nav.py",
    "src/moatrader/valuation/sotp.py",
    "src/moatrader/valuation/apv.py",
    "src/moatrader/valuation/scenario_dcf.py",
    "src/moatrader/valuation/normalized_fcff.py",
    "src/moatrader/valuation/rnpv.py",
    "scripts/audit_expanded_valuation_signals.py",
    "scripts/analyze_universal_value_factor_test.py",
    "scripts/build_normalized_fcff_inputs.py",
    "scripts/build_scenario_dcf_inputs.py",
    "scripts/build_rnpv_inputs.py",
    "scripts/build_rim_inputs.py",
    "scripts/run_unified_value_architecture_calibration.py",
)
BASE_SOURCE_PATTERNS = (
    "runs/kr-signal-*/companies/*/financial-snapshot.json",
    "date-inputs/*/dcf-inputs/*.json",
    "date-inputs/*/valuation-inputs/*.json",
    "date-inputs/*/valuation-profiles/*.json",
    "date-inputs/*/universe-manifest.csv",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_payload(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def frozen_policy(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repository_root
    trust = ValuationTrustPolicy()
    normalization = UnifiedValueNormalizationPolicy()
    input_hashes = {
        "dates": sha256_file(args.dates),
        "universe": sha256_file(args.universe),
        "pit_sector_map": sha256_file(args.pit_sector_map),
        "base_source_tree": tree_hash(args.base_root, BASE_SOURCE_PATTERNS),
    }
    if args.valuation_input_root:
        input_hashes["valuation_input_tree"] = tree_hash(args.valuation_input_root)
    if args.valuation_profile_root:
        input_hashes["valuation_profile_tree"] = tree_hash(args.valuation_profile_root)
    code_hashes = {relative: sha256_file(root / relative) for relative in FROZEN_CODE}
    policy: dict[str, Any] = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_name": "Routing Architecture Calibration / Sanity Test",
        "analysis_grade": "RETURN_BLIND_ARCHITECTURE_CALIBRATION_NOT_ALPHA_VALIDATION",
        "frozen_at": args.frozen_at.isoformat(),
        "frozen_before_execution": True,
        "commit_sha": git_commit(root),
        "router_policy_version": ROUTER_CONTRACT_VERSION,
        "valuation_input_schema_version": ROUTED_VALUATION_INPUT_VERSION,
        "assumption_policy_version": ASSUMPTION_POLICY_VERSION,
        "trust_policy": trust.model_dump(mode="json"),
        "normalization_policy": normalization.model_dump(mode="json"),
        "benchmark_policy_version": BENCHMARK_POLICY_VERSION,
        "benchmark_policy": {
            "ECONOMIC_FCFF": "MATCHED_SAMPLE_PER_PBR",
            "NORMALIZED_FCFF": "MATCHED_SAMPLE_PER_PBR",
            "SCENARIO_DCF": "MATCHED_SAMPLE_PER_PBR",
            "APV": "MATCHED_SAMPLE_FCFF_AND_PER_PBR",
            "RIM": "MATCHED_SAMPLE_PBR_AND_PBR_ROE",
            "RNPV": "MATCHED_SAMPLE_PSR_PBR_BASELINE",
            "SOTP": "MATCHED_SAMPLE_PBR_NAV_PROXY",
            "NAV": "MATCHED_SAMPLE_PBR_NAV_PROXY",
        },
        "matched_sample_keys": ["signal_date", "issuer_id", "investable_universe"],
        "value_gap_field": "raw_value_gap",
        "value_gap_semantics": "SUPPORTED_INTRINSIC_VALUE_OVER_PRICE_MINUS_ONE",
        "expectation_gap_semantics": "SEPARATE_REVERSE_MODEL_DRIVER_DISTANCE_EXPERIMENT",
        "route_trigger_change_after_return_access_forbidden": True,
        "assumption_change_after_return_access_forbidden": True,
        "parent_reference_class_fallback": normalization.parent_class_fallback,
        "reference_class_hierarchy": list(normalization.reference_class_hierarchy),
        "broad_value_role": "COMPARISON_BASELINE_ONLY_NOT_PRIMARY_RANK",
        "primary_ranking_policy_changed": False,
        "return_inputs_forbidden": True,
        "llm_calls_for_routing_valuation_normalization": 0,
        "base_source_patterns": list(BASE_SOURCE_PATTERNS),
        "input_sha256": input_hashes,
        "source_snapshot_hash": sha256_payload(input_hashes),
        "frozen_code_sha256": code_hashes,
        "frozen_code_snapshot_hash": sha256_payload(code_hashes),
        "assumption_generation_rules": {
            "ECONOMIC_FCFF": "EXPLICIT_TYPED_INPUT; AUDITED_LEGACY_PIT_TTM_COMPATIBILITY_ENGINE_ALLOWED",
            "NORMALIZED_FCFF": "EXPLICIT_NORMALIZED_TYPED_INPUT_ONLY",
            "SCENARIO_DCF": "EXPLICIT_THREE_CASE_INPUT; BASE_ONLY_RANK; FIXED_WEIGHTS_DIAGNOSTIC_ONLY",
            "APV": "EXPLICIT_UNLEVERED_CF_DEBT_TAX_SHIELD_INPUT_ONLY",
            "RIM": "EXPLICIT_PIT_TTM_INPUT_OR_FROZEN_RIM_POLICY_V1",
            "RNPV": "EXPLICIT_ROLE_SEPARATED_PIPELINE_EVIDENCE_AND_FROZEN_POS",
            "SOTP": "ACTUAL_SUBMODEL_EXECUTION_WITH_BASIS_OWNERSHIP_NET_DEBT_NCI_SCOPES",
            "NAV": "EXPLICIT_ASSET_LIABILITY_TYPED_INPUT_ONLY",
        },
    }
    policy["policy_sha256"] = sha256_payload(policy)
    return policy


def tree_hash(root: Path, patterns: tuple[str, ...] = ("**/*.json",)) -> str:
    paths = {
        path
        for pattern in patterns
        for path in root.glob(pattern)
        if path.is_file()
    }
    rows = [
        {"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(paths)
    ]
    return sha256_payload(rows)


def render_report(result: dict[str, Any]) -> str:
    coverage = result["coverage"]
    lines = [
        "# Unified Value Routing Architecture Calibration",
        "",
        f"- 판정: **{result['verdict']}**",
        "- 등급: return-blind architecture calibration; alpha 성과검증 아님",
        f"- A/B byte-identical: `{str(result['repeatability_pass']).lower()}`",
        f"- LLM calls: `{coverage['llm_call_count']}`",
        f"- Fallback FCFF: `{coverage['fallback_fcff_count']}`",
        f"- Route→actual engine match: `{coverage['route_actual_engine_match_rate']:.1%}`",
        f"- Route stability: `{coverage['route_stability']:.1%}`",
        f"- Trust-gate pass: `{coverage['trust_gate_pass_count']}/{coverage['valuation_generated_count']}`",
        f"- Rank-eligible score: `{coverage['rank_eligible_count']}/{coverage['row_count']}`",
        "- Normalization: `method+archetype → method → model family`, minimum N=20",
        "- Broad Value/PER+PBR role: comparison baseline only; primary ranking policy unchanged",
        f"- Historical calibration: `{result['historical_calibration_status']}`",
        "",
        "## Route audit",
        "",
        "| Route | Routed | Eligible | Valued | Ranked | Exec/Eligible | Trusted/Valued | Ranked/Routed | Engine match |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, item in coverage["method_audit"].items():
        lines.append(
            f"| {method} | {item['routed_count']} | {item['eligible_route_count']} "
            f"| {item['valuation_generated_count']} | {item['rank_eligible_count']} "
            f"| {item['execution_rate']:.1%} | {item['trusted_generated_share']:.1%} "
            f"| {item['score_coverage']:.1%} | {item['actual_engine_match_rate']:.1%} |"
        )
    lines.extend(
        [
            "",
            "## Trust and normalization reasons",
            "",
            f"- Pre-normalization status: `{coverage['pre_normalization_status_counts']}`",
            f"- Final alpha status: `{coverage['alpha_status_counts']}`",
            f"- Trust/status reasons: `{coverage['trust_reason_counts_by_status']}`",
            f"- Normalization levels: `{coverage['normalization_level_counts']}`",
            "",
            "## Gate failures",
            "",
            *[f"- `{reason}`" for reason in coverage["architecture_gate_failures"]],
            "",
            "이 검증은 수익률을 읽지 않았습니다. PASS는 이후 matched-sample 성과 보정을 진행할 수 있다는 뜻일 뿐 alpha를 증명하지 않습니다.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"calibration output already exists: {args.output}")
    args.output.mkdir(parents=True)
    policy = frozen_policy(args)
    policy_path = args.output / "preregistered-policy.json"
    write_json(policy_path, policy)
    normalization_policy = UnifiedValueNormalizationPolicy.model_validate(
        policy["normalization_policy"]
    )
    trust_policy = ValuationTrustPolicy.model_validate(policy["trust_policy"])

    coverages: list[dict[str, Any]] = []
    for label in ("run-a", "run-b"):
        audit_args = argparse.Namespace(
            dates=args.dates,
            universe=args.universe,
            base_root=args.base_root,
            output=args.output / label,
            valuation_input_root=args.valuation_input_root,
            valuation_profile_root=args.valuation_profile_root,
            pit_sector_map=args.pit_sector_map,
            development_sector_map=None,
            mode="pit_strict",
            expected_date_count=args.expected_date_count,
            expected_universe_count=args.expected_universe_count,
            normalization_policy=normalization_policy,
            trust_policy=trust_policy,
        )
        coverages.append(run_architecture_audit(audit_args))

    artifact_hashes: dict[str, str] = {}
    repeatability = True
    for name in COMPARABLE_ARTIFACTS:
        left = args.output / "run-a" / name
        right = args.output / "run-b" / name
        left_hash = sha256_file(left)
        right_hash = sha256_file(right)
        repeatability = repeatability and left_hash == right_hash
        artifact_hashes[name] = left_hash
    repeatability = repeatability and coverages[0] == coverages[1]
    architecture_pass = bool(coverages[0]["architecture_gate_pass"] and repeatability)
    result = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "policy_sha256": policy["policy_sha256"],
        "repeatability_pass": repeatability,
        "artifact_sha256": artifact_hashes,
        "coverage": coverages[0],
        "verdict": "PASS" if architecture_pass else "FAIL",
        "historical_calibration_status": (
            "READY_FOR_EX_POST_MATCHED_SAMPLE_CALIBRATION"
            if architecture_pass
            else "NOT_RUN_ARCHITECTURE_GATE_FAILED"
        ),
        "return_data_accessed": False,
    }
    write_json(args.output / "FINAL-RESULT.json", result)
    (args.output / "FINAL-REPORT.md").write_text(
        render_report(result), encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze and run return-blind Unified Value architecture calibration twice."
    )
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--dates", type=Path, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--pit-sector-map", type=Path, required=True)
    parser.add_argument("--valuation-input-root", type=Path)
    parser.add_argument("--valuation-profile-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-at", type=datetime.fromisoformat, required=True)
    parser.add_argument("--expected-date-count", type=int, default=4)
    parser.add_argument("--expected-universe-count", type=int, default=150)
    args = parser.parse_args()
    if args.frozen_at.tzinfo is None or args.frozen_at.utcoffset() is None:
        parser.error("--frozen-at must include a timezone offset")
    for name in (
        "repository_root",
        "dates",
        "universe",
        "base_root",
        "pit_sector_map",
        "valuation_input_root",
        "valuation_profile_root",
        "output",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.resolve())
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
