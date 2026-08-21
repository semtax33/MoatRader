from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from moatrader.expectations.future_eri import EvidenceScoreBand, evidence_score_band
from moatrader.expectations.historical_evidence import (
    HistoricalEvidenceFeatureRowV1,
    seal_historical_evidence_features,
    sha256_file,
)


SEOUL = ZoneInfo("Asia/Seoul")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[HistoricalEvidenceFeatureRowV1]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _read_features(path: Path) -> list[HistoricalEvidenceFeatureRowV1]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            HistoricalEvidenceFeatureRowV1.model_validate_json(line)
            for line in handle
            if line.strip()
        ]


def audit_v1r_feasibility(
    *,
    feature_build: Path,
    contract_freeze_manifest: Path,
    parser_validation_manifest: Path,
    output: Path,
    v2_feature_coverage: Path | None = None,
    allow_dry_run_contract: bool = False,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    feature_path = feature_build / "features-v1r-pre-feasibility.jsonl"
    coverage_path = feature_build / "v1r-feature-coverage-report.json"
    source_effect_path = feature_build / "v1r-source-effect-report.json"
    upstream_status_path = feature_build / "stage-status.json"
    for path in (
        feature_path,
        coverage_path,
        source_effect_path,
        upstream_status_path,
        contract_freeze_manifest,
        parser_validation_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    upstream = json.loads(upstream_status_path.read_text(encoding="utf-8"))
    if upstream.get("status") != "V1R_COMPLETE_CASE_FEATURES_BUILT_AWAITING_FEASIBILITY_AUDIT":
        raise ValueError("V1R feature build is incomplete")
    if upstream.get("outcome_vault_opened", False) or upstream.get(
        "return_data_opened", False
    ) or upstream.get("value_data_opened", False):
        raise ValueError("V1R feature build is contaminated by downstream data")
    expected_artifacts = {
        "features": feature_path,
        "coverage": coverage_path,
        "source_effect": source_effect_path,
    }
    for name, path in expected_artifacts.items():
        if upstream.get("artifact_hashes", {}).get(name) != sha256_file(path):
            raise ValueError(f"V1R feature artifact changed before feasibility audit: {name}")
    contract = json.loads(contract_freeze_manifest.read_text(encoding="utf-8"))
    allowed_contract_status = {"V1R_PREOUTCOME_CONTRACT_FROZEN"}
    if allow_dry_run_contract:
        allowed_contract_status.add("V1R_PREOUTCOME_CONTRACT_DRY_RUN")
    if contract.get("status") not in allowed_contract_status:
        raise ValueError("V1R feasibility requires the frozen V1R contract")
    if contract.get("dry_run_only", True) and not allow_dry_run_contract:
        raise ValueError("production V1R feasibility cannot use a dry-run contract")
    if upstream.get("input_hashes", {}).get("contract_freeze") != sha256_file(
        contract_freeze_manifest
    ):
        raise ValueError("V1R feature build used a different contract freeze")
    parser = json.loads(parser_validation_manifest.read_text(encoding="utf-8"))
    if parser.get("status") != "V1R_LOCKED_TEST_PASSED" or not parser.get(
        "source_stratum_gate_passed", False
    ):
        raise ValueError("V1R LOCKED parser gate has not passed")
    if upstream.get("input_hashes", {}).get("parser_validation") != sha256_file(
        parser_validation_manifest
    ):
        raise ValueError("V1R feature build used a different parser validation")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    source_effect = json.loads(source_effect_path.read_text(encoding="utf-8"))
    if any(
        payload.get("outcome_vault_opened", False)
        or payload.get("outcomes_opened", False)
        or payload.get("return_data_opened", False)
        or payload.get("returns_opened", False)
        or payload.get("value_data_opened", False)
        for payload in (coverage, source_effect, parser, contract)
    ):
        raise ValueError("V1R feasibility inputs are contaminated")

    rows = _read_features(feature_path)
    if len({row.observation_id for row in rows}) != len(rows):
        raise ValueError("V1R feature observation IDs must be unique")
    bands = Counter(
        evidence_score_band(row.evidence.evidence_f_score).value
        for row in rows
        if row.evidence.evidence_f_score is not None
    )
    band_counts = {band.value: bands[band.value] for band in EvidenceScoreBand}
    if band_counts != coverage.get("five_band_counts"):
        raise ValueError("V1R reported band counts do not match sealed feature rows")
    if len(rows) != coverage.get("six_axis_complete_features"):
        raise ValueError("V1R complete-case count does not match feature rows")
    minimum_rows_per_band = 20
    bands_passed = all(value >= minimum_rows_per_band for value in band_counts.values())
    feasibility_passed = bool(rows) and bands_passed

    v2_reference: dict[str, Any] | None = None
    if v2_feature_coverage is not None:
        if not v2_feature_coverage.is_file():
            raise FileNotFoundError(v2_feature_coverage)
        v2 = json.loads(v2_feature_coverage.read_text(encoding="utf-8"))
        if v2.get("outcomes_opened", False) or v2.get("returns_opened", False):
            raise ValueError("V2 reference contains downstream data")
        v2_reference = {
            "sha256": sha256_file(v2_feature_coverage),
            "pair_count": v2.get("pair_count"),
            "grounded_axis_count_histogram": v2.get("grounded_axis_count_histogram"),
            "source_type_distribution": v2.get("source_type_distribution"),
        }

    output.mkdir(parents=True, exist_ok=True)
    feasibility_report = {
        "schema_version": "moatrader-historical-v1r-feasibility-report/1",
        "status": "PASS" if feasibility_passed else "FAIL_COMPLETE_CASE_COVERAGE",
        "contract_tag": "future-eri-v1r-three-section-preoutcome",
        "same_hypothesis_as_v1": True,
        "source_corrected_replication": True,
        "total_filing_pairs": coverage["total_filing_pairs"],
        "axis_complete_rate": coverage["axis_complete_rate"],
        "six_axis_complete_count": len(rows),
        "six_axis_complete_coverage": coverage["complete_case_coverage"],
        "unique_issuers": coverage["unique_issuers"],
        "unique_signal_months": coverage["unique_signal_months"],
        "five_band_counts": band_counts,
        "minimum_rows_per_band": minimum_rows_per_band,
        "all_five_bands_at_least_20": bands_passed,
        "source_effect_report_sha256": sha256_file(source_effect_path),
        "original_v1_reference": source_effect.get("original_v1_feature_coverage"),
        "v2_feature_only_reference": v2_reference,
        "research_interpretation": (
            "V1R_FEASIBLE_SOURCE_INGESTION_INCOMPLETENESS_MAY_EXPLAIN_V1_COLLAPSE"
            if feasibility_passed
            else "SOURCE_OMISSION_NOT_SUFFICIENT_COMPLETE_CASE_DESIGN_REMAINS_INFEASIBLE"
        ),
        "outcomes_opened": False,
        "returns_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    report_path = output / "v1r-feasibility-report.json"
    _write_json(report_path, feasibility_report)
    seal_path: Path | None = None
    sealed_feature_path: Path | None = None
    if rows:
        sealed_feature_path = output / "features-pre-outcome.jsonl"
        _write_jsonl(sealed_feature_path, sorted(rows, key=lambda row: row.observation_id))
        seal = seal_historical_evidence_features(rows, sealed_at=datetime.now(SEOUL))
        seal_path = output / "feature-seal.json"
        _write_json(seal_path, seal.model_dump(mode="json"))
    manifest = {
        "schema_version": "moatrader-historical-v1r-preoutcome-manifest/1",
        "status": (
            "V1R_PREOUTCOME_FEASIBILITY_SEALED"
            if feasibility_passed
            else "V1R_FEASIBILITY_TOMBSTONED_COMPLETE_CASE_COLLAPSE"
        ),
        "contract_tag": "future-eri-v1r-three-section-preoutcome",
        "original_v1_tag": contract["original_v1_tag"],
        "original_v1_tag_commit": contract["original_v1_tag_commit"],
        "original_v1_tag_preserved": True,
        "git_commit": contract["git_commit"],
        "feature_policy_sha256": contract["feature_policy_sha256"],
        "band_policy_sha256": contract["band_policy_sha256"],
        "source_policy_sha256": contract["source_policy_sha256"],
        "parser_prompt_sha256": contract["parser_prompt_sha256"],
        "locked_test_sha256": sha256_file(parser_validation_manifest),
        "minimum_rows_per_band": minimum_rows_per_band,
        "five_band_counts": band_counts,
        "feasibility_report_sha256": sha256_file(report_path),
        "feature_dataset_sha256": (
            sha256_file(sealed_feature_path) if sealed_feature_path is not None else None
        ),
        "feature_seal_sha256": sha256_file(seal_path) if seal_path is not None else None,
        "feature_contract_modification_forbidden": True,
        "outcome_stage_authorized": feasibility_passed,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
    }
    manifest_path = output / "pre-outcome-manifest.json"
    _write_json(manifest_path, manifest)
    stage = {
        "schema_version": "moatrader-historical-v1r-feasibility-stage/1",
        "status": (
            "V1R_FEASIBILITY_PASSED_ERI_MECHANISM_ELIGIBLE"
            if feasibility_passed
            else "V1R_FEASIBILITY_FAILED_COMPLETE_CASE_COVERAGE_COLLAPSE"
        ),
        "feature_dataset_sealed": bool(rows),
        "six_axis_complete_features": len(rows),
        "all_five_bands_at_least_20": bands_passed,
        "outcome_stage_authorized": feasibility_passed,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "feasibility_report_sha256": sha256_file(report_path),
        "pre_outcome_manifest_sha256": sha256_file(manifest_path),
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "stage-status.json", stage)
    return stage


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the outcome-blind V1R complete-case feasibility gate."
    )
    parser.add_argument("--feature-build", type=Path, required=True)
    parser.add_argument("--contract-freeze-manifest", type=Path, required=True)
    parser.add_argument("--parser-validation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--v2-feature-coverage", type=Path)
    parser.add_argument("--allow-dry-run-contract", action="store_true")
    args = parser.parse_args()
    result = audit_v1r_feasibility(
        feature_build=args.feature_build,
        contract_freeze_manifest=args.contract_freeze_manifest,
        parser_validation_manifest=args.parser_validation_manifest,
        output=args.output,
        v2_feature_coverage=args.v2_feature_coverage,
        allow_dry_run_contract=args.allow_dry_run_contract,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
