from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import Field

from moatrader.canonical.models import ContractModel
from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisPairClassification,
    HistoricalFilingPair,
    PairedAxisPacket,
    packet_id,
    sha256_file,
    validate_classification_grounding,
)
from moatrader.expectations.historical_evidence_v2 import (
    AxisApplicabilityV2,
    HistoricalSparseEvidenceFeatureRowV2,
    SparseAxisAvailabilityV2,
    SparseAxisEvidenceV2,
    build_sparse_feature_row_v2,
    merge_axis_evidence_v2,
    qualitative_axis_evidence,
    sparse_feature_coverage_report,
)


class AxisApplicabilityDecisionInputV2(ContractModel):
    schema_version: str = "moatrader-axis-applicability-decision-v2/1"
    pair_id: str = Field(pattern=r"^PAIR_[0-9a-f]{24}$")
    axis: OperatingEvidenceAxis
    applicability: AxisApplicabilityV2
    rule_id: str = Field(min_length=1)
    outcome_data_accessed: bool = False
    return_data_accessed: bool = False


class DeterministicAxisEvidenceInputV2(ContractModel):
    schema_version: str = "moatrader-deterministic-axis-evidence-input-v2/1"
    pair_id: str = Field(pattern=r"^PAIR_[0-9a-f]{24}$")
    evidence: SparseAxisEvidenceV2
    outcome_data_accessed: bool = False
    return_data_accessed: bool = False


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def _packet_groups(path: Path) -> Iterator[list[PairedAxisPacket]]:
    group: list[PairedAxisPacket] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                group.append(PairedAxisPacket.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"invalid blinded packet at line {number}: {exc}") from exc
            if len(group) == len(OperatingEvidenceAxis):
                if {item.axis for item in group} != set(OperatingEvidenceAxis):
                    raise ValueError("blinded packet group does not contain exactly six axes")
                yield group
                group = []
    if group:
        raise ValueError("blinded packet input has a trailing incomplete pair group")


def _validation_status(path: Path | None, *, coverage_only_unvalidated: bool) -> dict[str, Any]:
    if path is None:
        if not coverage_only_unvalidated:
            raise ValueError(
                "V2 parser validation manifest is required unless coverage-only mode is explicit"
            )
        return {
            "status": "COVERAGE_ONLY_UNVALIDATED",
            "directional_strata_gate_passed": False,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "V2_LOCKED_TESTS_PASSED":
        raise ValueError("V2 parser validation requires passing Natural and Balanced LOCKED tests")
    if not payload.get("natural_frequency_gate_passed", False):
        raise ValueError("V2 parser validation lacks the Natural-frequency LOCKED gate")
    if not payload.get("directional_strata_gate_passed", False):
        raise ValueError("V2 parser validation lacks balanced directional strata coverage")
    if payload.get("outcome_vault_opened", False) or payload.get("return_data_opened", False):
        raise ValueError("V2 parser validation manifest is contaminated by downstream data")
    return payload


def _measurement_contract(
    *,
    contract_freeze_manifest: Path | None,
    deterministic_pit_manifest: Path | None,
    deterministic_evidence_input: Path | None,
    applicability_input: Path | None,
    validation: dict[str, Any],
    coverage_only_unvalidated: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if contract_freeze_manifest is None or deterministic_pit_manifest is None:
        if coverage_only_unvalidated:
            return ({"status": "COVERAGE_ONLY_UNFROZEN_CONTRACT"}, {})
        raise ValueError(
            "production V2 features require the frozen measurement contract and deterministic PIT manifest"
        )
    for path in (contract_freeze_manifest, deterministic_pit_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = json.loads(contract_freeze_manifest.read_text(encoding="utf-8"))
    pit = json.loads(deterministic_pit_manifest.read_text(encoding="utf-8"))
    if contract.get("status") != "V2_PRE_OUTCOME_CONTRACT_FROZEN":
        raise ValueError("V2 measurement contract is not frozen")
    if contract.get("dry_run_only", True) or contract.get("worktree_dirty", True):
        raise ValueError("production V2 features require a clean committed contract freeze")
    if pit.get("status") != "DETERMINISTIC_PIT_EVIDENCE_COMPLETE_OUTCOME_BLIND":
        raise ValueError("deterministic PIT stage is incomplete")
    if deterministic_evidence_input is None or applicability_input is None:
        raise ValueError("frozen V2 feature build requires deterministic evidence and applicability")
    if pit.get("deterministic_evidence_sha256") != sha256_file(deterministic_evidence_input):
        raise ValueError("deterministic evidence changed after PIT stage")
    if pit.get("applicability_sha256") != sha256_file(applicability_input):
        raise ValueError("applicability decisions changed after PIT stage")
    if pit.get("rules_contract_sha256") != contract.get("applicability_policy_sha256"):
        raise ValueError("PIT rules do not match the frozen applicability policy")
    if pit.get("evidence_priority") != contract.get("evidence_priority"):
        raise ValueError("PIT evidence priority does not match the frozen contract")
    if validation.get("parser_freeze_sha256") != contract.get("parser_freeze_sha256"):
        raise ValueError("dual LOCKED validation does not match the frozen parser contract")
    workspace = Path(__file__).resolve().parents[1]
    current_code_paths = {
        "feature_contract": workspace / "src" / "moatrader" / "expectations" / "historical_evidence_v2.py",
        "deterministic_builder": workspace / "scripts" / "build_historical_deterministic_pit_evidence_v2.py",
        "semantic_selector": workspace / "scripts" / "prepare_historical_semantic_packets_v2.py",
        "sparse_builder": workspace / "scripts" / "build_historical_sparse_features_v2.py",
        "calibrator": workspace / "scripts" / "calibrate_historical_sparse_features_v2.py",
        "locked_evaluator": workspace / "scripts" / "evaluate_historical_evidence_parser_v2.py",
        "locked_set_preparer": workspace / "scripts" / "prepare_historical_locked_sets_v2.py",
        "abstention_audit": workspace / "scripts" / "audit_historical_evidence_abstentions_v2.py",
    }
    frozen_code_hashes = contract.get("code_sha256", {})
    for name, path in current_code_paths.items():
        if frozen_code_hashes.get(name) != sha256_file(path):
            raise ValueError(f"V2 measurement code changed after contract freeze: {name}")
    if any(
        payload.get("outcome_vault_opened", False) or payload.get("return_data_opened", False)
        for payload in (contract, pit)
    ):
        raise ValueError("V2 measurement inputs are contaminated by downstream data")
    return contract, pit


def _source_contract(input_build: Path, *, allow_test_input_without_source_audit: bool) -> dict[str, Any]:
    source_audit_path = input_build / "source-audit.json"
    build_manifest_path = input_build / "build-manifest.json"
    before_path = input_build / "private" / "source-integrity-before.json"
    after_path = input_build / "private" / "source-integrity-after.json"
    required = (source_audit_path, build_manifest_path, before_path, after_path)
    if not all(path.is_file() for path in required):
        if allow_test_input_without_source_audit:
            return {"status": "TEST_INPUT_SOURCE_AUDIT_BYPASS", "verified": False}
        raise FileNotFoundError(
            "V2 production input requires the Arcana + MoatRader source audit and before/after integrity manifests"
        )
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    if not source_audit.get("both_source_systems_used", False):
        raise ValueError("V2 source build must use both Arcana HTML and MoatRader originals")
    if source_audit.get("arcana_regular_filing_count", 0) < 1:
        raise ValueError("V2 source audit has no Arcana regular filings")
    if source_audit.get("moatrader_regular_original_filing_count", 0) < 1:
        raise ValueError("V2 source audit has no MoatRader original regular filings")
    if not source_audit.get("all_arcana_sections_discovered", False):
        raise ValueError(
            "V2 source audit must discover Arcana business-info, finance-comment, "
            "and finance-statement HTML"
        )
    if not source_audit.get("all_arcana_sections_read_for_pairs", False):
        raise ValueError(
            "V2 source extraction must read all three Arcana DART HTML sections"
        )
    if not source_audit.get("all_arcana_sections_contributed_to_packets", False):
        raise ValueError(
            "V2 source packets must contain evidence excerpts contributed by all three "
            "Arcana DART HTML sections"
        )
    section_audit = source_audit.get("arcana_section_audit", {}).get("sections", {})
    required_arcana_sections = ("business-info", "finance-comment", "finance-statement")
    if any(
        section_audit.get(section, {}).get("attached_source_variant_count", 0) < 1
        for section in required_arcana_sections
    ):
        raise ValueError("V2 source audit has an Arcana DART section with no attached source")
    if source_audit.get("source_files_modified", True):
        raise ValueError("source audit reports modified original files")
    if build_manifest.get("source_files_modified", True):
        raise ValueError("source build manifest reports modified original files")
    if before.get("mutation_policy") != "ARCANA_AND_MOATRADER_SOURCE_FILES_READ_ONLY":
        raise ValueError("source integrity manifest lacks the read-only mutation policy")
    if after.get("verification_status") != "PASS_NO_SOURCE_MUTATION":
        raise ValueError("source integrity verification did not pass")
    if before.get("records") != after.get("records"):
        raise ValueError("before/after original-source integrity records differ")
    if len(before.get("records", [])) != source_audit.get("source_integrity_record_count"):
        raise ValueError("source integrity record count does not match the source audit")
    artifacts = build_manifest.get("artifacts", {})
    artifact_paths = {
        "source-audit.json": source_audit_path,
        "private/filing-pairs.jsonl": input_build / "private" / "filing-pairs.jsonl",
        "llm/blinded-packets.jsonl": input_build / "llm" / "blinded-packets.jsonl",
        "private/source-integrity-before.json": before_path,
        "private/source-integrity-after.json": after_path,
    }
    for name, path in artifact_paths.items():
        if artifacts.get(name) != sha256_file(path):
            raise ValueError(f"source build artifact changed after its build manifest: {name}")
    return {
        "status": "ARCANA_AND_MOATRADER_ORIGINALS_VERIFIED_READ_ONLY",
        "verified": True,
        "regular_pair_count": source_audit.get("regular_pair_count"),
        "arcana_regular_filing_count": source_audit.get("arcana_regular_filing_count"),
        "moatrader_regular_original_filing_count": source_audit.get(
            "moatrader_regular_original_filing_count"
        ),
        "arcana_section_audit": source_audit.get("arcana_section_audit"),
        "pair_source_extraction_by_origin": source_audit.get(
            "pair_source_extraction_by_origin"
        ),
        "all_arcana_sections_read_for_pairs": True,
        "all_arcana_sections_contributed_to_packets": True,
        "source_integrity_record_count": source_audit.get("source_integrity_record_count"),
        "source_audit_sha256": sha256_file(source_audit_path),
        "build_manifest_sha256": sha256_file(build_manifest_path),
        "source_integrity_before_sha256": sha256_file(before_path),
        "source_integrity_after_sha256": sha256_file(after_path),
    }


def build_sparse_features(
    *,
    input_build: Path,
    classification_build: Path,
    output: Path,
    parser_validation_manifest: Path | None = None,
    applicability_input: Path | None = None,
    deterministic_evidence_input: Path | None = None,
    contract_freeze_manifest: Path | None = None,
    deterministic_pit_manifest: Path | None = None,
    coverage_only_unvalidated: bool = False,
    allow_test_input_without_source_audit: bool = False,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    pair_path = input_build / "private" / "filing-pairs.jsonl"
    packet_path = input_build / "llm" / "blinded-packets.jsonl"
    classification_path = classification_build / "classifications.jsonl"
    classification_stage_path = classification_build / "stage-status.json"
    for path in (pair_path, packet_path, classification_path, classification_stage_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    validation = _validation_status(
        parser_validation_manifest,
        coverage_only_unvalidated=coverage_only_unvalidated,
    )
    source_contract = _source_contract(
        input_build,
        allow_test_input_without_source_audit=allow_test_input_without_source_audit,
    )
    measurement_contract, pit_stage = _measurement_contract(
        contract_freeze_manifest=contract_freeze_manifest,
        deterministic_pit_manifest=deterministic_pit_manifest,
        deterministic_evidence_input=deterministic_evidence_input,
        applicability_input=applicability_input,
        validation=validation,
        coverage_only_unvalidated=coverage_only_unvalidated,
    )
    classification_stage = json.loads(classification_stage_path.read_text(encoding="utf-8"))
    if classification_stage.get("status") != "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE":
        raise ValueError("classification build is incomplete")
    if parser_validation_manifest is not None:
        for key in ("parser_version", "prompt_sha256", "requested_model"):
            if classification_stage.get(key) != validation.get(key):
                raise ValueError(f"classification build does not match validated {key}")

    classifications_list = _read_jsonl(classification_path, AxisPairClassification)
    classifications = {item.packet_id: item for item in classifications_list}
    if len(classifications) != len(classifications_list):
        raise ValueError("V2 classifications must have unique packet IDs")

    applicability_rows = (
        _read_jsonl(applicability_input, AxisApplicabilityDecisionInputV2)
        if applicability_input is not None
        else []
    )
    applicability = {(item.pair_id, item.axis): item for item in applicability_rows}
    if len(applicability) != len(applicability_rows):
        raise ValueError("V2 applicability decisions must be unique by pair and axis")
    if any(item.outcome_data_accessed or item.return_data_accessed for item in applicability_rows):
        raise ValueError("V2 applicability decisions must be outcome and return blind")

    deterministic_rows = (
        _read_jsonl(deterministic_evidence_input, DeterministicAxisEvidenceInputV2)
        if deterministic_evidence_input is not None
        else []
    )
    deterministic = {(item.pair_id, item.evidence.axis): item.evidence for item in deterministic_rows}
    if len(deterministic) != len(deterministic_rows):
        raise ValueError("deterministic V2 evidence must be unique by pair and axis")
    if any(item.outcome_data_accessed or item.return_data_accessed for item in deterministic_rows):
        raise ValueError("deterministic V2 evidence must be outcome and return blind")

    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "sparse-features-all-pairs.jsonl"
    features: list[HistoricalSparseEvidenceFeatureRowV2] = []
    used_classification_ids: set[str] = set()
    used_applicability_keys: set[tuple[str, OperatingEvidenceAxis]] = set()
    used_deterministic_keys: set[tuple[str, OperatingEvidenceAxis]] = set()
    deterministic_priority_axes = {
        OperatingEvidenceAxis.MARGIN,
        OperatingEvidenceAxis.INVENTORY_MISMATCH,
        OperatingEvidenceAxis.BACKLOG,
        OperatingEvidenceAxis.CAPACITY_CAPEX,
    }
    pair_count = 0
    conflict_counts: Counter[str] = Counter()
    with pair_path.open("r", encoding="utf-8") as pair_handle, feature_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as feature_handle:
        pair_lines = (line for line in pair_handle if line.strip())
        groups = _packet_groups(packet_path)
        for pair_line, packets in zip(pair_lines, groups, strict=True):
            pair = HistoricalFilingPair.model_validate_json(pair_line)
            packet_by_axis = {item.axis: item for item in packets}
            expected_ids = {packet_id(pair.pair_id, axis) for axis in OperatingEvidenceAxis}
            if {item.packet_id for item in packets} != expected_ids:
                raise ValueError(f"packet group does not match filing pair: {pair.pair_id}")
            evidence_rows: list[SparseAxisEvidenceV2] = []
            for axis in OperatingEvidenceAxis:
                packet = packet_by_axis[axis]
                classification = classifications.get(packet.packet_id)
                if classification is not None:
                    validate_classification_grounding(classification, packet)
                    used_classification_ids.add(packet.packet_id)
                decision = applicability.get((pair.pair_id, axis))
                if applicability_input is not None and decision is None:
                    raise ValueError(
                        f"applicability input is missing pair/axis: {pair.pair_id}/{axis.value}"
                    )
                if decision is not None:
                    used_applicability_keys.add((pair.pair_id, axis))
                deterministic_item = deterministic.get((pair.pair_id, axis))
                if (
                    decision is not None
                    and deterministic_item is not None
                    and decision.applicability != deterministic_item.applicability
                ):
                    raise ValueError(
                        "PIT applicability disagrees with deterministic evidence: "
                        f"{pair.pair_id}/{axis.value}"
                    )
                if deterministic_evidence_input is not None and axis in deterministic_priority_axes:
                    if deterministic_item is None:
                        raise ValueError(
                            "deterministic PIT input is missing a priority axis: "
                            f"{pair.pair_id}/{axis.value}"
                        )
                if deterministic_item is not None:
                    used_deterministic_keys.add((pair.pair_id, axis))
                qualitative = qualitative_axis_evidence(
                    classification=classification,
                    packet=packet,
                    pair=pair,
                    applicability=(
                        decision.applicability
                        if decision is not None
                        else AxisApplicabilityV2.APPLICABLE
                    ),
                    applicability_rule_id=(
                        decision.rule_id
                        if decision is not None
                        else "UNIVERSAL_APPLICABLE_CALIBRATION_ONLY_V2"
                    ),
                )
                if (
                    deterministic_item is not None
                    and deterministic_item.availability == SparseAxisAvailabilityV2.GROUNDED
                    and qualitative.availability == SparseAxisAvailabilityV2.GROUNDED
                ):
                    same_direction = deterministic_item.direction == qualitative.direction
                    conflict_counts[
                        f"{axis.value}|{'AGREE' if same_direction else 'CONFLICT'}"
                    ] += 1
                evidence_rows.append(
                    merge_axis_evidence_v2(
                        qualitative,
                        deterministic_item,
                    )
                )
            feature = build_sparse_feature_row_v2(pair=pair, axis_evidence=evidence_rows)
            features.append(feature)
            feature_handle.write(feature.model_dump_json() + "\n")
            pair_count += 1
    unused = set(classifications) - used_classification_ids
    if unused:
        raise ValueError(f"classification build contains packets outside the filing-pair universe: {sorted(unused)[:5]}")
    unused_applicability = set(applicability) - used_applicability_keys
    if unused_applicability:
        raise ValueError("applicability input contains rows outside the filing-pair universe")
    unused_deterministic = set(deterministic) - used_deterministic_keys
    if unused_deterministic:
        raise ValueError("deterministic input contains rows outside the filing-pair universe")
    if source_contract.get("verified") and source_contract.get("regular_pair_count") != pair_count:
        raise ValueError("V2 all-pair row count does not match the verified source audit")
    if measurement_contract.get("status") == "V2_PRE_OUTCOME_CONTRACT_FROZEN":
        if measurement_contract.get("source_audit_sha256") != source_contract.get(
            "source_audit_sha256"
        ):
            raise ValueError("feature source audit does not match the frozen V2 contract")
        if pit_stage.get("pair_count") != pair_count:
            raise ValueError("deterministic PIT did not cover the full filing-pair universe")

    coverage = sparse_feature_coverage_report(features)
    coverage.update(
        classification_packet_count=len(classifications),
        applicability_contract_complete=applicability_input is not None,
        deterministic_axis_evidence_count=len(deterministic),
        deterministic_pit_priority_applied=deterministic_evidence_input is not None,
        parser_directional_validation_passed=bool(
            validation.get("directional_strata_gate_passed", False)
        ),
    )
    _write_json(output / "sparse-feature-coverage-report.json", coverage)
    priority_report = {
        "schema_version": "moatrader-evidence-priority-conflict-report-v2/1",
        "priority": ["DETERMINISTIC_NUMERIC", "STRUCTURED_TABLE", "LLM_NARRATIVE"],
        "score_averaging_used": False,
        "conflict_counts": dict(sorted(conflict_counts.items())),
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    priority_report_path = output / "evidence-priority-conflicts.json"
    _write_json(priority_report_path, priority_report)
    status = {
        "schema_version": "moatrader-historical-sparse-feature-stage-v2/1",
        "status": "SPARSE_FEATURES_BUILT_AWAITING_OUTCOME_BLIND_CALIBRATION",
        "pair_count": pair_count,
        "classification_packet_count": len(classifications),
        "applicability_contract_complete": applicability_input is not None,
        "parser_directional_validation_passed": bool(
            validation.get("directional_strata_gate_passed", False)
        ),
        "deterministic_pit_priority_applied": deterministic_evidence_input is not None,
        "feature_dataset_sealed": False,
        "outcome_stage_authorized": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "missing_is_neutral": False,
        "primary_feature": "SIGNED_BREADTH_WITH_SEPARATE_NOBS_AND_COVERAGE",
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
        "source_contract": source_contract,
        "measurement_contract_tag": measurement_contract.get("contract_tag"),
        "measurement_contract_frozen": measurement_contract.get("status")
        == "V2_PRE_OUTCOME_CONTRACT_FROZEN",
        "input_hashes": {
            "filing_pairs": sha256_file(pair_path),
            "blinded_packets": sha256_file(packet_path),
            "classifications": sha256_file(classification_path),
            "parser_validation_manifest": (
                sha256_file(parser_validation_manifest)
                if parser_validation_manifest is not None
                else None
            ),
            "applicability": (
                sha256_file(applicability_input) if applicability_input is not None else None
            ),
            "deterministic_evidence": (
                sha256_file(deterministic_evidence_input)
                if deterministic_evidence_input is not None
                else None
            ),
            "contract_freeze_manifest": (
                sha256_file(contract_freeze_manifest)
                if contract_freeze_manifest is not None
                else None
            ),
            "deterministic_pit_manifest": (
                sha256_file(deterministic_pit_manifest)
                if deterministic_pit_manifest is not None
                else None
            ),
            "evidence_priority_conflicts": sha256_file(priority_report_path),
        },
    }
    _write_json(output / "stage-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build outcome-blind V2 sparse evidence features across every filing pair; "
            "missing axes remain NA and deterministic PIT evidence has priority."
        )
    )
    parser.add_argument("--input-build", type=Path, required=True)
    parser.add_argument("--classification-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parser-validation-manifest", type=Path)
    parser.add_argument("--applicability-input", type=Path)
    parser.add_argument("--deterministic-evidence-input", type=Path)
    parser.add_argument("--contract-freeze-manifest", type=Path)
    parser.add_argument("--deterministic-pit-manifest", type=Path)
    parser.add_argument("--coverage-only-unvalidated", action="store_true")
    parser.add_argument(
        "--allow-test-input-without-source-audit",
        action="store_true",
        help="Fixture-only bypass; production V2 builds must not use this option.",
    )
    args = parser.parse_args()
    result = build_sparse_features(
        input_build=args.input_build,
        classification_build=args.classification_build,
        output=args.output,
        parser_validation_manifest=args.parser_validation_manifest,
        applicability_input=args.applicability_input,
        deterministic_evidence_input=args.deterministic_evidence_input,
        contract_freeze_manifest=args.contract_freeze_manifest,
        deterministic_pit_manifest=args.deterministic_pit_manifest,
        coverage_only_unvalidated=args.coverage_only_unvalidated,
        allow_test_input_without_source_audit=args.allow_test_input_without_source_audit,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
