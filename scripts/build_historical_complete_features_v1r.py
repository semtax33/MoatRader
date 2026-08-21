from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator

from moatrader.canonical.models import StatementType
from moatrader.expectations.future_eri import (
    EvidenceObservation,
    EvidenceScoreBand,
    OperatingEvidenceAxis,
    evidence_score_band,
)
from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    AxisPairClassification,
    HistoricalEvidenceFeatureRowV1,
    HistoricalFilingPair,
    PairedAxisPacket,
    build_historical_evidence_feature_row,
    packet_id,
    sha256_file,
    validate_classification_grounding,
)


ARCANA_ORIGINS = {
    "ARCANA_BUSINESS_HTML",
    "ARCANA_FINANCE_COMMENT_HTML",
    "ARCANA_FINANCE_STATEMENT_HTML",
}


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
            payload = row.model_dump(mode="json") if hasattr(row, "model_dump") else row
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def _packet_groups(path: Path) -> Iterator[list[PairedAxisPacket]]:
    group: list[PairedAxisPacket] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            group.append(PairedAxisPacket.model_validate_json(line))
            if len(group) == len(OperatingEvidenceAxis):
                if {row.axis for row in group} != set(OperatingEvidenceAxis):
                    raise ValueError("V1R packet group must contain exactly six axes")
                yield group
                group = []
    if group:
        raise ValueError("V1R packet input has a trailing incomplete group")


def _evidence_id(pair_id_value: str, axis: OperatingEvidenceAxis, side: str) -> str:
    digest = hashlib.sha256(
        f"{pair_id_value}|{axis.value}|{side}".encode("utf-8")
    ).hexdigest()
    return f"EVID_{digest[:24]}"


def _observation(
    *,
    pair: HistoricalFilingPair,
    classification: AxisPairClassification,
    private: dict[str, Any],
    side: str,
) -> EvidenceObservation:
    if classification.status != AxisClassificationStatus.COMPLETE:
        raise ValueError("cannot build V1R observation from an abstention")
    previous = side == "previous"
    source_id = (
        classification.previous_source_id if previous else classification.current_source_id
    )
    source_span = (
        classification.previous_source_span if previous else classification.current_source_span
    )
    state = classification.previous_state if previous else classification.current_state
    assert source_id is not None and source_span is not None and state is not None
    source = private["sources"][source_id]
    if source["side"] != side:
        raise ValueError("V1R classification source side mismatch")
    filing = pair.previous if previous else pair.current
    return EvidenceObservation(
        observation_id=_evidence_id(pair.pair_id, classification.axis, side.upper()),
        issuer_id=pair.ticker,
        fiscal_period=filing.fiscal_period_end.isoformat(),
        axis=classification.axis,
        state=state,
        source_document_id=filing.rcept_no,
        source_span=source_span,
        source_published_at=filing.published_at,
        available_at=filing.available_at,
        signal_timestamp=filing.signal_timestamp,
        statement_type=StatementType.DISCLOSED_FACT,
        classification_rule_id="BLINDED_PAIRED_AXIS_LLM_V1R_SOURCE_CORRECTED",
        materiality_rule_id="QUALITATIVE_AXIS_STATE_EQUAL_UNIT_V1",
        confidence=Decimal(str(classification.confidence)),
        materiality=Decimal(1),
    )


def _source_contract(input_build: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    source_audit_path = input_build / "source-audit.json"
    build_manifest_path = input_build / "build-manifest.json"
    before_path = input_build / "private" / "source-integrity-before.json"
    after_path = input_build / "private" / "source-integrity-after.json"
    for path in (source_audit_path, build_manifest_path, before_path, after_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    build_manifest = json.loads(build_manifest_path.read_text(encoding="utf-8"))
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    if source_audit.get("research_variant") != "V1R":
        raise ValueError("V1R features require a V1R source build")
    for key in (
        "all_arcana_sections_discovered",
        "all_arcana_sections_read_for_pairs",
        "all_arcana_sections_contributed_to_packets",
        "both_source_systems_used",
    ):
        if not source_audit.get(key, False):
            raise ValueError(f"V1R source build did not pass {key}")
    if source_audit.get("source_files_modified", True) or build_manifest.get(
        "source_files_modified", True
    ):
        raise ValueError("V1R source build reports original-source mutation")
    if after.get("verification_status") != "PASS_NO_SOURCE_MUTATION":
        raise ValueError("V1R source integrity did not pass")
    if before.get("records") != after.get("records"):
        raise ValueError("V1R before/after source-integrity records differ")
    artifacts = build_manifest.get("artifacts", {})
    for name, path in {
        "source-audit.json": source_audit_path,
        "private/filing-pairs.jsonl": input_build / "private" / "filing-pairs.jsonl",
        "private/pair-source-map.jsonl": input_build / "private" / "pair-source-map.jsonl",
        "llm/blinded-packets.jsonl": input_build / "llm" / "blinded-packets.jsonl",
        "private/source-integrity-before.json": before_path,
        "private/source-integrity-after.json": after_path,
    }.items():
        if artifacts.get(name) != sha256_file(path):
            raise ValueError(f"V1R source artifact changed after build: {name}")
    return source_audit, build_manifest


def _pair_overlap_category(pair: HistoricalFilingPair) -> str:
    side_moatrader = [
        any(variant.origin.value == "MOATRADER_OPENDART_ARCHIVE" for variant in filing.source_variants)
        for filing in (pair.previous, pair.current)
    ]
    if all(side_moatrader):
        return "BOTH_PERIODS_ARCANA_MOATRADER_OVERLAP"
    if any(side_moatrader):
        return "ONE_PERIOD_ARCANA_MOATRADER_OVERLAP"
    return "ARCANA_ONLY_BOTH_PERIODS"


def build_v1r_features(
    *,
    input_build: Path,
    classification_build: Path,
    parser_validation_manifest: Path,
    contract_freeze_manifest: Path,
    output: Path,
    original_v1_feature_coverage: Path | None = None,
    allow_dry_run_contract: bool = False,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    pair_path = input_build / "private" / "filing-pairs.jsonl"
    private_path = input_build / "private" / "pair-source-map.jsonl"
    packet_path = input_build / "llm" / "blinded-packets.jsonl"
    classification_path = classification_build / "classifications.jsonl"
    classification_status_path = classification_build / "stage-status.json"
    for path in (
        pair_path,
        private_path,
        packet_path,
        classification_path,
        classification_status_path,
        parser_validation_manifest,
        contract_freeze_manifest,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_audit, _ = _source_contract(input_build)
    parser_validation = json.loads(parser_validation_manifest.read_text(encoding="utf-8"))
    if parser_validation.get("status") != "V1R_LOCKED_TEST_PASSED" or not parser_validation.get(
        "source_stratum_gate_passed", False
    ):
        raise ValueError("V1R source-stratified LOCKED parser gate has not passed")
    if (
        parser_validation.get("outcome_vault_opened", False)
        or parser_validation.get("return_data_opened", False)
        or parser_validation.get("value_data_opened", False)
    ):
        raise ValueError("V1R parser validation is contaminated")
    contract = json.loads(contract_freeze_manifest.read_text(encoding="utf-8"))
    allowed_contract_status = {"V1R_PREOUTCOME_CONTRACT_FROZEN"}
    if allow_dry_run_contract:
        allowed_contract_status.add("V1R_PREOUTCOME_CONTRACT_DRY_RUN")
    if contract.get("status") not in allowed_contract_status:
        raise ValueError("V1R pre-outcome contract is not frozen")
    if contract.get("dry_run_only", True) and not allow_dry_run_contract:
        raise ValueError("production V1R feature build cannot use a dry-run contract")
    if (
        contract.get("outcome_vault_opened", False)
        or contract.get("return_data_opened", False)
        or contract.get("value_data_opened", False)
    ):
        raise ValueError("V1R pre-outcome contract is contaminated")
    if contract.get("source_audit_sha256") != sha256_file(input_build / "source-audit.json"):
        raise ValueError("V1R source build does not match the frozen contract")
    if contract.get("parser_freeze_sha256") != parser_validation.get("parser_freeze_sha256"):
        raise ValueError("V1R parser validation and measurement contract use different freezes")
    workspace = Path(__file__).resolve().parents[1]
    current_code_paths = {
        "source_builder": workspace / "scripts" / "build_historical_future_eri_evidence.py",
        "historical_contract": workspace
        / "src"
        / "moatrader"
        / "expectations"
        / "historical_evidence.py",
        "locked_preparer": workspace / "scripts" / "prepare_historical_v1r_locked_set.py",
        "locked_evaluator": workspace / "scripts" / "evaluate_historical_evidence_parser_v1r.py",
        "feature_builder": workspace / "scripts" / "build_historical_complete_features_v1r.py",
        "feasibility_auditor": workspace / "scripts" / "audit_historical_v1r_feasibility.py",
    }
    for name, path in current_code_paths.items():
        if contract.get("code_sha256", {}).get(name) != sha256_file(path):
            raise ValueError(f"V1R measurement code changed after freeze: {name}")
    classification_stage = json.loads(
        classification_status_path.read_text(encoding="utf-8")
    )
    if classification_stage.get("status") != "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE":
        raise ValueError("V1R full classifications are incomplete")
    if (
        classification_stage.get("outcome_vault_opened", False)
        or classification_stage.get("return_data_opened", False)
        or classification_stage.get("value_data_opened", False)
    ):
        raise ValueError("V1R full classifications are contaminated by downstream data")
    if classification_stage.get("input_blinded_packet_sha256") != sha256_file(packet_path):
        raise ValueError("V1R full classification input hash mismatch")
    for key in ("parser_version", "prompt_sha256", "requested_model"):
        if classification_stage.get(key) != parser_validation.get(key):
            raise ValueError(f"V1R full classifications do not match LOCKED {key}")

    classification_rows = _read_jsonl(classification_path, AxisPairClassification)
    classifications = {row.packet_id: row for row in classification_rows}
    if len(classifications) != len(classification_rows):
        raise ValueError("V1R classifications must have unique packet IDs")
    features: list[HistoricalEvidenceFeatureRowV1] = []
    exclusions: list[dict[str, Any]] = []
    axis_status: dict[str, Counter[str]] = defaultdict(Counter)
    axis_delta: dict[str, Counter[str]] = defaultdict(Counter)
    source_grounding: dict[str, Counter[str]] = defaultdict(Counter)
    source_side_patterns: Counter[str] = Counter()
    complete_by_overlap: Counter[str] = Counter()
    total_by_overlap: Counter[str] = Counter()
    used_ids: set[str] = set()
    pair_count = 0
    feature_output = output / "features-v1r-pre-feasibility.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    private_lines = (
        json.loads(line)
        for line in private_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    with pair_path.open("r", encoding="utf-8") as pair_handle, feature_output.open(
        "w", encoding="utf-8", newline="\n"
    ) as feature_handle:
        pair_lines = (line for line in pair_handle if line.strip())
        for pair_line, packets, private in zip(
            pair_lines, _packet_groups(packet_path), private_lines, strict=True
        ):
            pair = HistoricalFilingPair.model_validate_json(pair_line)
            pair_count += 1
            overlap = _pair_overlap_category(pair)
            total_by_overlap[overlap] += 1
            by_axis = {packet.axis: packet for packet in packets}
            if {packet.packet_id for packet in packets} != {
                packet_id(pair.pair_id, axis) for axis in OperatingEvidenceAxis
            }:
                raise ValueError(f"V1R packets do not match pair: {pair.pair_id}")
            previous: list[EvidenceObservation] = []
            current: list[EvidenceObservation] = []
            missing: list[dict[str, str]] = []
            for axis in OperatingEvidenceAxis:
                packet = by_axis[axis]
                classification = classifications.get(packet.packet_id)
                if classification is None:
                    raise ValueError(f"V1R classification missing: {packet.packet_id}")
                used_ids.add(packet.packet_id)
                validate_classification_grounding(classification, packet)
                axis_status[axis.value][classification.status.value] += 1
                if classification.status != AxisClassificationStatus.COMPLETE:
                    missing.append({"axis": axis.value, "reason": classification.status.value})
                    continue
                previous_observation = _observation(
                    pair=pair,
                    classification=classification,
                    private=private,
                    side="previous",
                )
                current_observation = _observation(
                    pair=pair,
                    classification=classification,
                    private=private,
                    side="current",
                )
                previous.append(previous_observation)
                current.append(current_observation)
                assert classification.delta is not None
                axis_delta[axis.value][str(classification.delta)] += 1
                assert classification.previous_source_id is not None
                assert classification.current_source_id is not None
                previous_origin = str(
                    private["sources"][classification.previous_source_id]["origin"]
                )
                current_origin = str(
                    private["sources"][classification.current_source_id]["origin"]
                )
                source_grounding[axis.value][previous_origin] += 1
                source_grounding[axis.value][current_origin] += 1
                source_side_patterns[
                    "SAME_SOURCE_ORIGIN" if previous_origin == current_origin else "CROSS_SOURCE_ORIGIN"
                ] += 1
            if missing:
                exclusions.append(
                    {"pair_id": pair.pair_id, "overlap_category": overlap, "reasons": missing}
                )
                continue
            feature = build_historical_evidence_feature_row(
                pair=pair,
                previous_observations=previous,
                current_observations=current,
                coverage_sector=str(private.get("coverage_sector") or "UNMAPPED"),
            )
            features.append(feature)
            complete_by_overlap[overlap] += 1
            feature_handle.write(feature.model_dump_json() + "\n")
    if set(classifications) != used_ids:
        raise ValueError("V1R classification build contains packets outside the source universe")
    if pair_count != source_audit.get("regular_pair_count"):
        raise ValueError("V1R feature pair count does not match source audit")

    _write_jsonl(output / "complete-case-exclusions.jsonl", exclusions)
    bands: Counter[str] = Counter()
    issuers: Counter[str] = Counter()
    months: Counter[str] = Counter()
    years: Counter[str] = Counter()
    for feature in features:
        assert feature.evidence.evidence_f_score is not None
        bands[evidence_score_band(feature.evidence.evidence_f_score).value] += 1
        issuers[feature.issuer_id] += 1
        months[feature.signal_timestamp.strftime("%Y-%m")] += 1
        years[str(feature.signal_timestamp.year)] += 1
    band_counts = {band.value: bands[band.value] for band in EvidenceScoreBand}
    coverage = {
        "schema_version": "moatrader-historical-feature-coverage-v1r/1",
        "research_variant": "V1R_SOURCE_CORRECTED_REPLICATION",
        "feature_rule": "SIX_AXIS_COMPLETE_CASE_F_SCORE_SUM_UNCHANGED_FROM_V1",
        "total_filing_pairs": pair_count,
        "six_axis_complete_features": len(features),
        "complete_case_coverage": len(features) / pair_count if pair_count else 0.0,
        "unique_issuers": len(issuers),
        "unique_signal_months": len(months),
        "by_signal_year": dict(sorted(years.items())),
        "axis_complete_rate": {
            axis.value: (
                axis_status[axis.value][AxisClassificationStatus.COMPLETE.value] / pair_count
                if pair_count
                else 0.0
            )
            for axis in OperatingEvidenceAxis
        },
        "axis_classification_status_distribution": {
            axis.value: {
                status.value: axis_status[axis.value][status.value]
                for status in AxisClassificationStatus
            }
            for axis in OperatingEvidenceAxis
        },
        "axis_delta_distribution": {
            axis.value: {
                "-1": axis_delta[axis.value]["-1"],
                "0": axis_delta[axis.value]["0"],
                "+1": axis_delta[axis.value]["1"],
            }
            for axis in OperatingEvidenceAxis
        },
        "five_band_counts": band_counts,
        "minimum_rows_per_band": 20,
        "all_five_bands_at_least_20": all(value >= 20 for value in band_counts.values()),
        "outcomes_opened": False,
        "returns_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    coverage_path = output / "v1r-feature-coverage-report.json"
    _write_json(coverage_path, coverage)

    original_reference: dict[str, Any] | None = None
    if original_v1_feature_coverage is not None:
        if not original_v1_feature_coverage.is_file():
            raise FileNotFoundError(original_v1_feature_coverage)
        original_payload = json.loads(original_v1_feature_coverage.read_text(encoding="utf-8"))
        if original_payload.get("outcomes_opened", False) or original_payload.get(
            "returns_opened", False
        ):
            raise ValueError("V1 reference coverage is contaminated by outcomes")
        original_reference = {
            "sha256": sha256_file(original_v1_feature_coverage),
            "six_axis_complete_features": original_payload.get("six_axis_complete_features"),
            "unique_issuers": original_payload.get("unique_issuers"),
            "unique_signal_months": original_payload.get("unique_signal_months"),
            "five_band_counts": original_payload.get("feature_band_counts"),
        }
    source_effect = {
        "schema_version": "moatrader-v1r-source-effect-report/1",
        "research_design": {
            "A_V1": "ORIGINAL_SOURCE_INCOMPLETE_COMPLETE_CASE",
            "B_V1R": "THREE_SECTION_SOURCE_CORRECTED_COMPLETE_CASE",
            "C_V2": "THREE_SECTION_SOURCE_CORRECTED_SPARSE_BREADTH",
            "A_TO_B": "SOURCE_COVERAGE_EFFECT_ONLY",
            "B_TO_C": "FEATURE_CONTRACT_EFFECT_ONLY",
        },
        "source_build_effect_audit": source_audit.get("source_effect_audit"),
        "classification_grounding_by_axis_and_origin": {
            axis.value: dict(sorted(source_grounding[axis.value].items()))
            for axis in OperatingEvidenceAxis
        },
        "classification_source_side_patterns": dict(sorted(source_side_patterns.items())),
        "complete_case_by_source_overlap": {
            category: {
                "total_pairs": total_by_overlap[category],
                "six_axis_complete": complete_by_overlap[category],
                "coverage": (
                    complete_by_overlap[category] / total_by_overlap[category]
                    if total_by_overlap[category]
                    else 0.0
                ),
            }
            for category in sorted(total_by_overlap)
        },
        "original_v1_feature_coverage": original_reference,
        "v1r_feature_coverage_sha256": sha256_file(coverage_path),
        "outcomes_opened": False,
        "returns_opened": False,
        "value_data_opened": False,
    }
    source_effect_path = output / "v1r-source-effect-report.json"
    _write_json(source_effect_path, source_effect)
    status = {
        "schema_version": "moatrader-historical-complete-feature-stage-v1r/1",
        "status": "V1R_COMPLETE_CASE_FEATURES_BUILT_AWAITING_FEASIBILITY_AUDIT",
        "contract_tag": "future-eri-v1r-three-section-preoutcome",
        "pair_count": pair_count,
        "six_axis_complete_features": len(features),
        "feature_dataset_sealed": False,
        "outcome_stage_authorized": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
        "input_hashes": {
            "source_audit": sha256_file(input_build / "source-audit.json"),
            "filing_pairs": sha256_file(pair_path),
            "blinded_packets": sha256_file(packet_path),
            "classifications": sha256_file(classification_path),
            "parser_validation": sha256_file(parser_validation_manifest),
            "contract_freeze": sha256_file(contract_freeze_manifest),
        },
        "artifact_hashes": {
            "features": sha256_file(feature_output),
            "coverage": sha256_file(coverage_path),
            "source_effect": sha256_file(source_effect_path),
            "exclusions": sha256_file(output / "complete-case-exclusions.jsonl"),
        },
    }
    _write_json(output / "stage-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build V1R six-axis complete-case FScore features without opening outcomes."
    )
    parser.add_argument("--input-build", type=Path, required=True)
    parser.add_argument("--classification-build", type=Path, required=True)
    parser.add_argument("--parser-validation-manifest", type=Path, required=True)
    parser.add_argument("--contract-freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-v1-feature-coverage", type=Path)
    parser.add_argument("--allow-dry-run-contract", action="store_true")
    args = parser.parse_args()
    result = build_v1r_features(
        input_build=args.input_build,
        classification_build=args.classification_build,
        parser_validation_manifest=args.parser_validation_manifest,
        contract_freeze_manifest=args.contract_freeze_manifest,
        output=args.output,
        original_v1_feature_coverage=args.original_v1_feature_coverage,
        allow_dry_run_contract=args.allow_dry_run_contract,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
