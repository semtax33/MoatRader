from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from moatrader.backtest.universe_corrected import (
    FINANCE_HINT_RE,
    HOLDING_HINT_RE,
    classify_security,
)
from moatrader.expectations.future_eri import (
    EvidenceIndexFeatureDatasetSealV2,
    EvidenceIndexFutureEriFeatureRowV2,
    FutureEriOutcomeInputV1,
    build_evidence_index_future_eri_label_v2,
)
from moatrader.expectations.historical_evidence import sha256_file
from scripts.prepare_historical_evidence_index_eri_inputs_v2 import (
    _filtered_parquet,
    _git_state,
    _row_map,
    _size_maps,
)


STAGES = (
    ("evidence_eligible", "Evidence eligible"),
    ("price_pit_available", "Price/PIT available"),
    ("reverse_valuation_available", "Reverse valuation available"),
    ("t63_snapshot_available", "t+63 snapshot available"),
    ("eri_decomposition_valid", "ERI decomposition valid"),
    ("final_common", "Final common"),
)

PRECHECK_REASON_PRIORITY = (
    "NO_EXACT_SIGNAL_OPEN_PRICE",
    "NON_COMMON_SECURITY",
    "FCFF_INCOMPARABLE_ARCHETYPE",
    "NO_SIGNAL_SIZE_BUCKET",
    "INVALID_SIGNAL_MARKET_INPUT",
    "NON_POSITIVE_SIGNAL_OPEN",
    "NON_POSITIVE_SIGNAL_LISTED_SHARES",
    "FEWER_THAN_TWO_VALID_PIT_ANNUALS",
)

T63_REASON_PRIORITY = (
    "MISSING_EXACT_T_PLUS_63_SESSION",
    "OUTCOME_WINDOW_INCOMPLETE",
    "SIGNAL_SESSION_ABSENT_FROM_CALENDAR",
    "MISSING_OUTCOME_ELIGIBILITY_INVENTORY",
    "TARGET_SESSION_NOT_EXACT_T_PLUS_63",
    "MISSING_TARGET_PRICE_METADATA",
    "MISSING_T_PLUS_63_PIT_FINANCIALS",
    "MISSING_EQUITY_BRIDGE_INPUTS",
    "MISSING_WACC_SOURCE",
)

BAND_ORDER = {
    "STRONG_BEAR": 0,
    "BEAR": 1,
    "NEUTRAL": 2,
    "BULL": 3,
    "STRONG_BULL": 4,
}


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    raw = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(raw, list):
        raise ValueError(f"record collection required: {path}")
    return [dict(item) for item in raw]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _index_records(
    records: Sequence[dict[str, Any]], *, source: str
) -> dict[str, dict[str, Any]]:
    indexed = {str(row["observation_id"]): row for row in records}
    if len(indexed) != len(records):
        raise ValueError(f"duplicate observation_id in {source}")
    return indexed


def _primary_reason(reasons: Sequence[str], priority: Sequence[str]) -> str:
    unique = set(reasons)
    for reason in priority:
        if reason in unique:
            return reason
    if not unique:
        raise ValueError("at least one exclusion reason is required")
    return sorted(unique)[0]


def _reverse_model_status(reason: str) -> str:
    prefix = "REVERSE_DCF_ERROR:"
    if not reason.startswith(prefix):
        raise ValueError(f"not a reverse DCF reason: {reason}")
    detail = reason[len(prefix) :]
    if detail.startswith("ValueError:"):
        detail = detail[len("ValueError:") :]
    aliases = {
        "base valuation must have positive equity value": (
            "REVERSE_DCF_BASE_VALUATION_NON_POSITIVE_EQUITY"
        ),
        "NO_POSITIVE_TURBO_DRIVER": "REVERSE_DCF_NO_POSITIVE_TURBO_DRIVER",
        "REVERSE_DCF_CENSORED_HIGH": "REVERSE_DCF_CENSORED_HIGH",
        "REVERSE_DCF_CENSORED_LOW": "REVERSE_DCF_CENSORED_LOW",
        "REVERSE_DCF_NO_ELIGIBLE_VALUATION": "REVERSE_DCF_NO_ELIGIBLE_VALUATION",
    }
    if detail in aliases:
        return aliases[detail]
    normalized = re.sub(r"[^A-Z0-9]+", "_", detail.upper()).strip("_")
    return f"REVERSE_DCF_ERROR_{normalized or 'UNKNOWN'}"


def _load_sector_map(path: Path) -> dict[str, str]:
    tables = pd.read_html(path, encoding="euc-kr")
    if not tables or tables[0].shape[1] < 4:
        raise ValueError("sector map must contain at least four KRX KIND columns")
    frame = tables[0]
    codes = (
        frame.iloc[:, 2]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(6)
    )
    sectors = frame.iloc[:, 3].astype(str).str.strip()
    return {
        code: sector
        for code, sector in zip(codes, sectors, strict=False)
        if re.fullmatch(r"[0-9A-Z]{6}", code) and sector and sector.casefold() != "nan"
    }


def _market_dimensions(
    *, pre_stage: dict[str, Any], full_rows: dict[str, dict[str, Any]]
) -> tuple[
    dict[str, str],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    sources = pre_stage.get("marcap_sources", {})
    if not isinstance(sources, dict) or not sources:
        raise ValueError("sealed pre-outcome stage has no MARCAP source map")
    paths: list[Path] = []
    hashes: dict[str, str] = {}
    for year, source in sorted(sources.items()):
        path = Path(str(source["path"]))
        expected = str(source["sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"sealed MARCAP source changed for {year}: {path}")
        paths.append(path)
        hashes[f"marcap_{year}"] = actual
    signal_dates = {
        date.fromisoformat(str(row["signal_timestamp"])[:10]) for row in full_rows.values()
    }
    frame = _filtered_parquet(
        paths,
        dates=signal_dates,
        columns=["Date", "Code", "Name", "Open", "Stocks", "Marcap", "MarketId"],
    )
    points = _row_map(frame)
    sizes = _size_maps(frame)
    size_by_id: dict[str, str] = {}
    point_by_id: dict[str, dict[str, Any]] = {}
    for observation_id, row in full_rows.items():
        signal_date = date.fromisoformat(str(row["signal_timestamp"])[:10])
        ticker = str(row["issuer_id"]).zfill(6)
        point = points.get((signal_date, ticker))
        if point is not None:
            point_by_id[observation_id] = point
        size_by_id[observation_id] = sizes.get(signal_date, {}).get(
            ticker, "UNKNOWN_SIGNAL_SIZE"
        )
    return size_by_id, point_by_id, hashes


def _route(
    *, reasons: Sequence[str], point: dict[str, Any] | None
) -> tuple[str, str]:
    reason_set = set(reasons)
    if "NON_COMMON_SECURITY" in reason_set:
        name = str((point or {}).get("Name") or "")
        return "INELIGIBLE_SECURITY", classify_security(name) if name else "UNKNOWN"
    if "FCFF_INCOMPARABLE_ARCHETYPE" in reason_set:
        return "FCFF_INCOMPARABLE_ARCHETYPE", "COMMON"
    if point is None:
        return "UNKNOWN_NO_SIGNAL_MARKET_ROW", "UNKNOWN"
    name = str(point.get("Name") or "")
    security = classify_security(name)
    if security != "COMMON":
        return "INELIGIBLE_SECURITY", security
    if FINANCE_HINT_RE.search(name) or HOLDING_HINT_RE.search(name):
        return "FCFF_INCOMPARABLE_ARCHETYPE", security
    return "FCFF", security


def _expectation_size(row: dict[str, Any]) -> str | None:
    values = (
        row.get("frozen_expectation_assumptions", {})
        .get("assumption_sources", {})
        .get("wacc", [])
    )
    prefix = "FROZEN_WACC_POLICY:MARKET_CAP_TERCILE:"
    for value in values:
        text = str(value)
        if text.startswith(prefix):
            return text[len(prefix) :]
    return None


def _coverage_rows(
    ledger: Sequence[dict[str, Any]], *, dimension: str
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ledger:
        value = row.get(dimension)
        grouped[str(value) if value not in {None, ""} else "UNKNOWN"].append(row)
    total_baseline = len(ledger)
    total_final = sum(bool(row["final_common"]) for row in ledger)
    result: list[dict[str, Any]] = []
    for value, rows in grouped.items():
        counts = {stage: sum(bool(row[stage]) for row in rows) for stage, _ in STAGES}
        issuers = {
            stage: len({row["issuer_id"] for row in rows if row[stage]})
            for stage, _ in STAGES
        }
        baseline = counts["evidence_eligible"]
        final = counts["final_common"]
        baseline_share = baseline / total_baseline if total_baseline else None
        final_share = final / total_final if total_final else None
        result.append(
            {
                "schema_version": "moatrader-eri-eligibility-coverage-dimension-v2/1",
                "dimension": dimension,
                "dimension_value": value,
                "evidence_eligible_count": baseline,
                "price_pit_available_count": counts["price_pit_available"],
                "reverse_valuation_available_count": counts[
                    "reverse_valuation_available"
                ],
                "t63_snapshot_available_count": counts["t63_snapshot_available"],
                "eri_decomposition_valid_count": counts["eri_decomposition_valid"],
                "final_common_count": final,
                "lost_before_price_pit_count": baseline
                - counts["price_pit_available"],
                "lost_at_reverse_valuation_count": counts["price_pit_available"]
                - counts["reverse_valuation_available"],
                "lost_at_t63_snapshot_count": counts["reverse_valuation_available"]
                - counts["t63_snapshot_available"],
                "lost_at_eri_decomposition_count": counts["t63_snapshot_available"]
                - counts["eri_decomposition_valid"],
                "final_retention_from_evidence": final / baseline if baseline else None,
                "evidence_eligible_issuer_count": issuers["evidence_eligible"],
                "price_pit_available_issuer_count": issuers["price_pit_available"],
                "reverse_valuation_available_issuer_count": issuers[
                    "reverse_valuation_available"
                ],
                "t63_snapshot_available_issuer_count": issuers[
                    "t63_snapshot_available"
                ],
                "eri_decomposition_valid_issuer_count": issuers[
                    "eri_decomposition_valid"
                ],
                "final_common_issuer_count": issuers["final_common"],
                "baseline_observation_share": baseline_share,
                "final_observation_share": final_share,
                "final_minus_baseline_share": (
                    final_share - baseline_share
                    if baseline_share is not None and final_share is not None
                    else None
                ),
            }
        )
    if dimension in {"full_evidence_band", "core_evidence_band"}:
        return sorted(
            result,
            key=lambda row: (
                BAND_ORDER.get(str(row["dimension_value"]), 999),
                str(row["dimension_value"]),
            ),
        )
    return sorted(
        result,
        key=lambda row: (-int(row["evidence_eligible_count"]), str(row["dimension_value"])),
    )


def _issuer_concentration(
    ledger: Sequence[dict[str, Any]], *, stage: str
) -> dict[str, Any]:
    counts = Counter(str(row["issuer_id"]) for row in ledger if row[stage])
    total = sum(counts.values())
    shares = sorted((value / total for value in counts.values()), reverse=True) if total else []
    top_ten = sum(shares[:10])
    top_decile_count = max(1, math.ceil(len(shares) * 0.10)) if shares else 0
    return {
        "stage": stage,
        "observation_count": total,
        "issuer_count": len(counts),
        "issuer_hhi": sum(value * value for value in shares),
        "largest_issuer_share": shares[0] if shares else None,
        "top_10_issuer_share": top_ten,
        "top_issuer_decile_observation_share": (
            sum(shares[:top_decile_count]) if shares else None
        ),
    }


def _distribution_shift(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "maximum_absolute_share_shift": None,
            "maximum_shift_category": None,
        }
    selected = max(rows, key=lambda row: abs(float(row["final_minus_baseline_share"])))
    return {
        "maximum_absolute_share_shift": abs(
            float(selected["final_minus_baseline_share"])
        ),
        "maximum_shift_category": selected["dimension_value"],
        "baseline_share": selected["baseline_observation_share"],
        "final_share": selected["final_observation_share"],
        "signed_shift": selected["final_minus_baseline_share"],
    }


def _stage_bridge(ledger: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous_count: int | None = None
    for key, label in STAGES:
        selected = [row for row in ledger if row[key]]
        count = len(selected)
        result.append(
            {
                "stage": key,
                "stage_label": label,
                "observation_count": count,
                "issuer_count": len({row["issuer_id"] for row in selected}),
                "loss_from_previous_stage": (
                    0 if previous_count is None else previous_count - count
                ),
                "retention_from_previous_stage": (
                    1.0
                    if previous_count is None
                    else count / previous_count
                    if previous_count
                    else None
                ),
                "retention_from_evidence_eligible": count / len(ledger) if ledger else None,
            }
        )
        previous_count = count
    return result


def _exclusion_stage_summary(
    ledger: Sequence[dict[str, Any]],
    *,
    stage: str,
    before_key: str,
    after_key: str,
) -> dict[str, Any]:
    selected = [row for row in ledger if row[before_key] and not row[after_key]]
    primary = Counter(str(row["primary_exclusion_reason"]) for row in selected)
    overlapping: Counter[str] = Counter()
    for row in selected:
        overlapping.update(str(value) for value in row["all_exclusion_reasons"])
    return {
        "stage": stage,
        "eligible_before_count": sum(bool(row[before_key]) for row in ledger),
        "retained_after_count": sum(bool(row[after_key]) for row in ledger),
        "lost_count": len(selected),
        "primary_reason_counts": dict(sorted(primary.items())),
        "overlapping_reason_counts": dict(sorted(overlapping.items())),
        "primary_reason_policy": "DETERMINISTIC_UPSTREAM_PRIORITY; OVERLAPPING_COUNTS_RETAINED",
    }


def _load_label_validation_reasons(
    *,
    feature_by_id: dict[str, dict[str, Any]],
    old_outcome_by_id: dict[str, dict[str, Any]],
    invalid_ids: set[str],
    feature_seal_path: Path,
    trading_sessions_path: Path,
) -> tuple[dict[str, str], dict[str, str]]:
    seal = EvidenceIndexFeatureDatasetSealV2.model_validate_json(
        feature_seal_path.read_text(encoding="utf-8")
    )
    sessions = [
        date.fromisoformat(str(value))
        for value in json.loads(trading_sessions_path.read_text(encoding="utf-8"))
    ]
    reasons: dict[str, str] = {}
    details: dict[str, str] = {}
    for observation_id in sorted(invalid_ids):
        feature = EvidenceIndexFutureEriFeatureRowV2.model_validate(
            feature_by_id[observation_id]
        )
        outcome = FutureEriOutcomeInputV1.model_validate(
            old_outcome_by_id[observation_id]
        )
        try:
            build_evidence_index_future_eri_label_v2(
                feature=feature,
                outcome=outcome,
                feature_seal=seal,
                trading_sessions=sessions,
            )
        except ValueError as exc:
            detail = str(exc)
            details[observation_id] = detail
            reasons[observation_id] = (
                "INVALID_ERI_ENTERPRISE_VALUE"
                if detail == "actual and counterfactual enterprise values must be positive"
                else "ERI_LABEL_VALIDATION_ERROR:ValueError"
            )
        else:
            raise ValueError(
                f"label-safe exclusion unexpectedly validates: {observation_id}"
            )
    return reasons, details


def _input_paths(
    *,
    full_index_build: Path,
    core_index_build: Path,
    pre_outcome_build: Path,
    feature_build: Path,
    outcome_vault_build: Path,
    label_safe_outcome_build: Path,
    eri_build: Path,
    sector_map: Path,
) -> dict[str, Path]:
    return {
        "full_rows": full_index_build / "full-evidence-index-eligible-nobs2.jsonl",
        "full_seal": full_index_build / "full-evidence-index-seal.json",
        "full_stage": full_index_build / "stage-status.json",
        "core_rows": core_index_build / "deterministic-core-index-eligible-nobs2.jsonl",
        "core_manifest": core_index_build / "pre-outcome-index-manifest.json",
        "core_stage": core_index_build / "stage-status.json",
        "expectations": pre_outcome_build / "expectations-pre-outcome.jsonl",
        "expectation_exclusions": pre_outcome_build / "expectation-exclusions.json",
        "eligibility_inventory": pre_outcome_build
        / "outcome-eligibility-inventory.jsonl",
        "trading_sessions": pre_outcome_build / "trading-sessions.json",
        "pre_outcome_seal": pre_outcome_build / "pre-outcome-input-seal.json",
        "pre_outcome_stage": pre_outcome_build / "stage-status.json",
        "feature_rows": feature_build
        / "features-with-frozen-expectations-pre-outcome.jsonl",
        "feature_exclusions": feature_build / "outcome-eligibility-exclusions.json",
        "feature_seal": feature_build / "feature-seal-pre-outcome.json",
        "feature_manifest": feature_build / "build-manifest.json",
        "feature_stage": feature_build / "stage-status.json",
        "old_outcomes": outcome_vault_build / "future-eri-outcomes.jsonl",
        "old_outcome_stage": outcome_vault_build / "stage-status.json",
        "safe_outcomes": label_safe_outcome_build / "future-eri-outcomes.jsonl",
        "safe_outcome_stage": label_safe_outcome_build / "stage-status.json",
        "eri_labels": eri_build / "future-eri-labels.jsonl",
        "eri_report": eri_build / "dual-evidence-index-eri-report.json",
        "eri_manifest": eri_build / "build-manifest.json",
        "eri_stage": eri_build / "stage-status.json",
        "sector_map": sector_map,
    }


def audit_eri_eligibility_bridge_v2(
    *,
    workspace: Path,
    full_index_build: Path,
    core_index_build: Path,
    pre_outcome_build: Path,
    feature_build: Path,
    outcome_vault_build: Path,
    label_safe_outcome_build: Path,
    eri_build: Path,
    sector_map: Path,
    output: Path,
    audit_as_of: str,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    date.fromisoformat(audit_as_of)
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production ERI eligibility audit requires a clean worktree")

    paths = _input_paths(
        full_index_build=full_index_build,
        core_index_build=core_index_build,
        pre_outcome_build=pre_outcome_build,
        feature_build=feature_build,
        outcome_vault_build=outcome_vault_build,
        label_safe_outcome_build=label_safe_outcome_build,
        eri_build=eri_build,
        sector_map=sector_map,
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"eligibility audit inputs are missing: {missing}")
    input_hashes = {role: sha256_file(path) for role, path in paths.items()}

    full_stage = _read_json(paths["full_stage"])
    core_stage = _read_json(paths["core_stage"])
    pre_stage = _read_json(paths["pre_outcome_stage"])
    feature_stage = _read_json(paths["feature_stage"])
    old_outcome_stage = _read_json(paths["old_outcome_stage"])
    safe_outcome_stage = _read_json(paths["safe_outcome_stage"])
    eri_stage = _read_json(paths["eri_stage"])
    eri_report = _read_json(paths["eri_report"])
    core_manifest = _read_json(paths["core_manifest"])
    feature_manifest = _read_json(paths["feature_manifest"])
    eri_manifest = _read_json(paths["eri_manifest"])
    primary = eri_report.get("primary_full", {})
    if not (
        full_stage.get("status") == "V2_FULL_EVIDENCE_INDEX_SEALED_OUTCOMES_CLOSED"
        and core_stage.get("outcome_vault_opened") is False
        and pre_stage.get("status")
        == "ERI_PRE_OUTCOME_INPUTS_PREPARED_OUTCOMES_CLOSED"
        and feature_stage.get("status") == "ERI_FEATURE_PANEL_SEALED_OUTCOMES_CLOSED"
        and old_outcome_stage.get("status")
        == "ERI_OUTCOME_VAULT_MATERIALIZED_AFTER_FEATURE_PANEL_SEAL"
        and safe_outcome_stage.get("status")
        == "ERI_OUTCOME_VAULT_MATERIALIZED_AFTER_FEATURE_PANEL_SEAL"
        and eri_stage.get("status")
        == "FULL_PRIMARY_MECHANISM_REJECTED_OR_INCONCLUSIVE"
        and primary.get("mechanism_gate_passed") is False
        and int(primary.get("adjacent_median_nondecreasing_count", -1)) < 4
        and eri_stage.get("future_eri_used_as_ranking") is False
        and eri_stage.get("return_data_opened") is False
    ):
        raise ValueError("V2 is not in the expected sealed failed-promotion state")
    provenance = full_stage.get("source_provenance_gate", {})
    if not all(
        provenance.get(field) is True
        for field in (
            "arcana_business_info_read",
            "arcana_finance_comment_read",
            "arcana_finance_statement_read",
            "moatrader_original_regular_filings_read",
            "all_expected_source_paths_verified",
        )
    ) or not (
        provenance.get("source_files_modified") is False
        and int(provenance.get("source_hash_mismatch_count", -1)) == 0
    ):
        raise ValueError("sealed Full Index source-provenance gate is incomplete")
    sealed_hash_checks = {
        "full_rows": full_stage.get("artifact_hashes", {}).get(
            "full_evidence_index_eligible_nobs2"
        )
        == input_hashes["full_rows"],
        "full_seal": full_stage.get("full_evidence_index_seal_sha256")
        == input_hashes["full_seal"],
        "core_rows": core_manifest.get("artifact_hashes", {}).get(
            "deterministic_core_index_eligible_nobs2"
        )
        == input_hashes["core_rows"],
        "core_manifest": core_stage.get("pre_outcome_index_manifest_sha256")
        == input_hashes["core_manifest"],
        "pre_expectations": pre_stage.get("artifact_hashes", {}).get(
            "expectations_pre_outcome"
        )
        == input_hashes["expectations"],
        "pre_expectation_exclusions": pre_stage.get("artifact_hashes", {}).get(
            "expectation_exclusions"
        )
        == input_hashes["expectation_exclusions"],
        "pre_inventory": pre_stage.get("artifact_hashes", {}).get(
            "outcome_eligibility_inventory"
        )
        == input_hashes["eligibility_inventory"],
        "pre_sessions": pre_stage.get("artifact_hashes", {}).get("trading_sessions")
        == input_hashes["trading_sessions"],
        "pre_seal": pre_stage.get("pre_outcome_input_seal_sha256")
        == input_hashes["pre_outcome_seal"],
        "feature_expectations": feature_manifest.get("expectation_input_sha256")
        == input_hashes["expectations"],
        "feature_inventory": feature_manifest.get("eligibility_inventory_sha256")
        == input_hashes["eligibility_inventory"],
        "feature_rows": feature_manifest.get("feature_artifact_sha256")
        == input_hashes["feature_rows"],
        "feature_exclusions": feature_manifest.get("eligibility_exclusions_sha256")
        == input_hashes["feature_exclusions"],
        "feature_seal": feature_manifest.get("feature_seal_sha256")
        == input_hashes["feature_seal"],
        "feature_sessions": feature_manifest.get("trading_sessions_sha256")
        == input_hashes["trading_sessions"],
        "initial_outcomes": old_outcome_stage.get("outcome_input_sha256")
        == input_hashes["old_outcomes"],
        "label_safe_outcomes": safe_outcome_stage.get("outcome_input_sha256")
        == input_hashes["safe_outcomes"],
        "eri_features": eri_manifest.get("feature_input_sha256")
        == input_hashes["feature_rows"],
        "eri_feature_seal": eri_manifest.get("feature_seal_pre_outcome_sha256")
        == input_hashes["feature_seal"],
        "eri_outcomes": eri_manifest.get("outcome_input_sha256")
        == input_hashes["safe_outcomes"],
        "eri_labels": eri_manifest.get("future_eri_labels_sha256")
        == input_hashes["eri_labels"],
        "eri_report": eri_manifest.get("dual_mechanism_report_sha256")
        == input_hashes["eri_report"],
        "eri_sessions": eri_manifest.get("trading_sessions_sha256")
        == input_hashes["trading_sessions"],
        "eri_stage": eri_manifest.get("stage_status_sha256")
        == input_hashes["eri_stage"],
    }
    failed_hash_checks = sorted(
        key for key, passed in sealed_hash_checks.items() if not passed
    )
    if failed_hash_checks:
        raise ValueError(f"sealed V2 artifact hash mismatch: {failed_hash_checks}")

    full_all = _index_records(_read_records(paths["full_rows"]), source="Full Index")
    core_all = _index_records(_read_records(paths["core_rows"]), source="Core Index")
    common_ids = set(full_all) & set(core_all)
    if common_ids != set(core_all) or len(common_ids) != int(
        pre_stage["common_full_core_index_count"]
    ):
        raise ValueError("Full/Core common Evidence-eligible panel changed")
    full_rows = {key: full_all[key] for key in common_ids}
    core_rows = {key: core_all[key] for key in common_ids}

    expectation_by_id = _index_records(
        _read_records(paths["expectations"]), source="expectations"
    )
    expectation_exclusion_by_id = _index_records(
        _read_records(paths["expectation_exclusions"]),
        source="expectation exclusions",
    )
    if (
        set(expectation_by_id) & set(expectation_exclusion_by_id)
        or set(expectation_by_id) | set(expectation_exclusion_by_id) != common_ids
        or len(expectation_by_id) != int(pre_stage["expectation_count"])
    ):
        raise ValueError("expectation successes/exclusions do not partition the common panel")

    precheck_by_id: dict[str, list[str]] = {}
    reverse_failure_by_id: dict[str, str] = {}
    for observation_id, row in expectation_exclusion_by_id.items():
        reasons = sorted({str(value) for value in row.get("reasons", [])})
        reverse = [reason for reason in reasons if reason.startswith("REVERSE_DCF_ERROR:")]
        precheck = [reason for reason in reasons if not reason.startswith("REVERSE_DCF_ERROR:")]
        if bool(reverse) == bool(precheck) or len(reverse) > 1:
            raise ValueError(
                f"expectation exclusion is not one sequential stage: {observation_id}"
            )
        if reverse:
            reverse_failure_by_id[observation_id] = reverse[0]
        else:
            precheck_by_id[observation_id] = precheck
    price_pit_ids = set(expectation_by_id) | set(reverse_failure_by_id)
    reverse_ids = set(expectation_by_id)

    feature_by_id = _index_records(
        _read_records(paths["feature_rows"]), source="feature panel"
    )
    feature_exclusion_by_id = _index_records(
        _read_records(paths["feature_exclusions"]), source="feature exclusions"
    )
    if (
        set(feature_by_id) & set(feature_exclusion_by_id)
        or set(feature_by_id) | set(feature_exclusion_by_id) != common_ids
        or not set(feature_by_id) <= reverse_ids
        or len(feature_by_id) != int(feature_manifest["feature_count"])
    ):
        raise ValueError("t+63 feature panel/exclusions do not partition the common panel")
    t63_ids = set(feature_by_id)

    old_outcome_by_id = _index_records(
        _read_records(paths["old_outcomes"]), source="initial outcome vault"
    )
    safe_outcome_by_id = _index_records(
        _read_records(paths["safe_outcomes"]), source="label-safe outcome vault"
    )
    label_by_id = _index_records(_read_records(paths["eri_labels"]), source="ERI labels")
    old_outcome_ids = set(old_outcome_by_id)
    safe_ids = set(safe_outcome_by_id)
    label_ids = set(label_by_id)
    if (
        not safe_ids <= old_outcome_ids <= t63_ids
        or label_ids != safe_ids
        or len(old_outcome_ids) != int(old_outcome_stage["outcome_count"])
        or len(safe_ids) != int(safe_outcome_stage["outcome_count"])
        or len(label_ids) != int(eri_stage["label_count"])
    ):
        raise ValueError("outcome vault and final ERI label ID sets are inconsistent")

    outcome_reason_by_id = {
        observation_id: "INVALID_REALIZED_NPAT_MARGIN"
        for observation_id in t63_ids - old_outcome_ids
    }
    validation_reasons, validation_details = _load_label_validation_reasons(
        feature_by_id=feature_by_id,
        old_outcome_by_id=old_outcome_by_id,
        invalid_ids=old_outcome_ids - safe_ids,
        feature_seal_path=paths["feature_seal"],
        trading_sessions_path=paths["trading_sessions"],
    )
    outcome_reason_by_id.update(validation_reasons)
    if Counter(outcome_reason_by_id.values()) != Counter(
        safe_outcome_stage.get("outcome_exclusion_counts", {})
    ):
        raise ValueError("reconstructed per-ID outcome exclusions do not match the seal")

    size_by_id, point_by_id, marcap_hashes = _market_dimensions(
        pre_stage=pre_stage, full_rows=full_rows
    )
    for observation_id, expectation in expectation_by_id.items():
        expected_size = _expectation_size(expectation)
        if expected_size is None or expected_size != size_by_id[observation_id]:
            raise ValueError(f"signal size reconstruction mismatch: {observation_id}")
    sector_by_ticker = _load_sector_map(sector_map)

    ledger: list[dict[str, Any]] = []
    for observation_id in sorted(common_ids):
        full = full_rows[observation_id]
        core = core_rows[observation_id]
        ticker = str(full["issuer_id"]).zfill(6)
        signal_timestamp = str(full["signal_timestamp"])
        pre_reasons = precheck_by_id.get(observation_id, [])
        route, security_type = _route(
            reasons=pre_reasons,
            point=point_by_id.get(observation_id),
        )
        all_reasons: list[str] = []
        primary_stage: str | None = None
        primary_reason: str | None = None
        detail: str | None = None
        if observation_id in precheck_by_id:
            all_reasons = precheck_by_id[observation_id]
            primary_stage = "EVIDENCE_TO_PRICE_PIT"
            primary_reason = _primary_reason(all_reasons, PRECHECK_REASON_PRIORITY)
            model_status = f"PRE_REVERSE_NOT_RUN:{primary_reason}"
        elif observation_id in reverse_failure_by_id:
            all_reasons = [reverse_failure_by_id[observation_id]]
            primary_stage = "PRICE_PIT_TO_REVERSE_VALUATION"
            primary_reason = all_reasons[0]
            model_status = _reverse_model_status(primary_reason)
        elif observation_id not in t63_ids:
            reasons = [
                str(value)
                for value in feature_exclusion_by_id[observation_id].get("reasons", [])
                if str(value) not in {"MISSING_T_REVERSE_DCF", "INVALID_T_REVERSE_DCF"}
            ]
            if not reasons:
                raise ValueError(f"reverse-success t+63 exclusion has no reason: {observation_id}")
            all_reasons = sorted(set(reasons))
            primary_stage = "REVERSE_VALUATION_TO_T63_SNAPSHOT"
            primary_reason = _primary_reason(all_reasons, T63_REASON_PRIORITY)
            model_status = f"REVERSE_DCF_SOLVED:{primary_reason}"
        elif observation_id not in safe_ids:
            primary_stage = "T63_SNAPSHOT_TO_ERI_DECOMPOSITION"
            primary_reason = outcome_reason_by_id[observation_id]
            all_reasons = [primary_reason]
            detail = validation_details.get(observation_id)
            model_status = f"ERI_DECOMPOSITION_INVALID:{primary_reason}"
        else:
            model_status = "ERI_LABEL_VALID"

        expectation = expectation_by_id.get(observation_id)
        driver = (
            str(expectation.get("reverse_dcf_provenance", {}).get("selected_driver"))
            if expectation is not None
            else None
        )
        if driver == "None":
            driver = None
        ledger.append(
            {
                "schema_version": "moatrader-eri-eligibility-ledger-row-v2/1",
                "observation_id": observation_id,
                "pair_id": str(full["pair_id"]),
                "issuer_id": ticker,
                "signal_timestamp": signal_timestamp,
                "signal_year": int(signal_timestamp[:4]),
                "sector": sector_by_ticker.get(ticker, "UNKNOWN_CURRENT_SECTOR"),
                "sector_basis": "CURRENT_2026_KRX_KIND_NON_PIT_DESCRIPTIVE_ONLY",
                "signal_size_bucket": size_by_id[observation_id],
                "valuation_route": route,
                "security_type": security_type,
                "reverse_dcf_driver": driver,
                "model_status": model_status,
                "full_evidence_index": str(full["full_evidence_index"]),
                "full_evidence_band": str(full["band"]),
                "full_nobs": int(full["nobs"]),
                "core_evidence_index": str(core["core_evidence_index"]),
                "core_evidence_band": str(core["band"]),
                "core_nobs": int(core["nobs"]),
                "evidence_source_mode": (
                    "SEMANTIC_INCLUDED"
                    if int(full.get("semantic_grounded_axis_count", 0)) > 0
                    else "DETERMINISTIC_ONLY"
                ),
                "evidence_eligible": True,
                "price_pit_available": observation_id in price_pit_ids,
                "reverse_valuation_available": observation_id in reverse_ids,
                "t63_snapshot_available": observation_id in t63_ids,
                "eri_decomposition_valid": observation_id in safe_ids,
                "final_common": observation_id in label_ids,
                "primary_exclusion_stage": primary_stage,
                "primary_exclusion_reason": primary_reason,
                "all_exclusion_reasons": all_reasons,
                "exclusion_detail": detail,
            }
        )

    bridge = _stage_bridge(ledger)
    expected_counts = {
        "evidence_eligible": int(pre_stage["common_full_core_index_count"]),
        "price_pit_available": len(price_pit_ids),
        "reverse_valuation_available": int(pre_stage["expectation_count"]),
        "t63_snapshot_available": int(pre_stage["potential_label_eligible_count"]),
        "eri_decomposition_valid": int(safe_outcome_stage["outcome_count"]),
        "final_common": int(eri_stage["label_count"]),
    }
    if {row["stage"]: row["observation_count"] for row in bridge} != expected_counts:
        raise ValueError("reconstructed eligibility bridge does not match sealed stage counts")

    dimension_fields = {
        "route": "valuation_route",
        "reverse_driver": "reverse_dcf_driver",
        "sector": "sector",
        "size": "signal_size_bucket",
        "year": "signal_year",
        "issuer": "issuer_id",
        "model_status": "model_status",
        "full_band": "full_evidence_band",
        "core_band": "core_evidence_band",
        "evidence_source_mode": "evidence_source_mode",
    }
    coverage = {
        name: _coverage_rows(ledger, dimension=field)
        for name, field in dimension_fields.items()
    }

    exclusion_summary = {
        "schema_version": "moatrader-eri-eligibility-exclusion-summary-v2/1",
        "primary_reason_definition": (
            "One deterministic upstream reason per observation for an additive waterfall; "
            "all contemporaneous reasons are retained separately."
        ),
        "stages": [
            _exclusion_stage_summary(
                ledger,
                stage="EVIDENCE_TO_PRICE_PIT",
                before_key="evidence_eligible",
                after_key="price_pit_available",
            ),
            _exclusion_stage_summary(
                ledger,
                stage="PRICE_PIT_TO_REVERSE_VALUATION",
                before_key="price_pit_available",
                after_key="reverse_valuation_available",
            ),
            _exclusion_stage_summary(
                ledger,
                stage="REVERSE_VALUATION_TO_T63_SNAPSHOT",
                before_key="reverse_valuation_available",
                after_key="t63_snapshot_available",
            ),
            _exclusion_stage_summary(
                ledger,
                stage="T63_SNAPSHOT_TO_ERI_DECOMPOSITION",
                before_key="t63_snapshot_available",
                after_key="eri_decomposition_valid",
            ),
            _exclusion_stage_summary(
                ledger,
                stage="ERI_DECOMPOSITION_TO_FINAL_COMMON",
                before_key="eri_decomposition_valid",
                after_key="final_common",
            ),
        ],
    }

    baseline_count = expected_counts["evidence_eligible"]
    final_count = expected_counts["final_common"]
    baseline_issuers = bridge[0]["issuer_count"]
    final_issuers = bridge[-1]["issuer_count"]
    full_band_rows = coverage["full_band"]
    strong_bull = next(
        row for row in full_band_rows if row["dimension_value"] == "STRONG_BULL"
    )
    shift_names = (
        "route",
        "sector",
        "size",
        "year",
        "full_band",
        "evidence_source_mode",
    )
    selection_bias = {
        "schema_version": "moatrader-eri-eligibility-selection-bias-v2/1",
        "status": (
            "SEVERE_SELECTION_COMPRESSION"
            if final_count / baseline_count < 0.10 or final_issuers / baseline_issuers < 0.20
            else "MODERATE_OR_LOW_SELECTION_COMPRESSION"
        ),
        "baseline_observation_count": baseline_count,
        "final_observation_count": final_count,
        "observation_retention": final_count / baseline_count,
        "baseline_issuer_count": baseline_issuers,
        "final_issuer_count": final_issuers,
        "issuer_retention": final_issuers / baseline_issuers,
        "issuer_concentration": {
            "baseline": _issuer_concentration(ledger, stage="evidence_eligible"),
            "final": _issuer_concentration(ledger, stage="final_common"),
        },
        "distribution_shifts": {
            name: _distribution_shift(coverage[name]) for name in shift_names
        },
        "strong_bull_selection": {
            "baseline_count": strong_bull["evidence_eligible_count"],
            "final_count": strong_bull["final_common_count"],
            "retention": strong_bull["final_retention_from_evidence"],
            "baseline_share": strong_bull["baseline_observation_share"],
            "final_share": strong_bull["final_observation_share"],
            "final_minus_baseline_share": strong_bull[
                "final_minus_baseline_share"
            ],
            "retention_relative_to_overall": (
                strong_bull["final_retention_from_evidence"]
                / (final_count / baseline_count)
            ),
        },
        "sector_limitation": (
            "Sector is a current 2026 KRX KIND classification used only for descriptive "
            "selection auditing; it is not PIT and did not affect eligibility."
        ),
    }

    termination_seal = {
        "schema_version": "moatrader-evidence-index-v2-termination-seal/1",
        "audit_as_of": audit_as_of,
        "status": "V2_PROMOTION_FAILED_AND_SEALED",
        "failure_reason": "FAILED_PREREGISTERED_MEDIAN_FIVE_BAND_MONOTONIC_GATE",
        "primary_endpoint": eri_stage["primary_endpoint"],
        "primary_mechanism_gate_passed": False,
        "adjacent_median_nondecreasing_count": primary[
            "adjacent_median_nondecreasing_count"
        ],
        "required_adjacent_median_nondecreasing_count": 4,
        "monthly_ic_mean_diagnostic_only": primary["monthly_ic_hac"]["mean"],
        "monthly_ic_hac_t_diagnostic_only": primary["monthly_ic_hac"][
            "t_statistic"
        ],
        "mean_reclassified_as_primary": False,
        "v2_thresholds_changed": False,
        "v2_index_formula_changed": False,
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "return_data_opened_by_audit": False,
        "eri_stage_sha256": input_hashes["eri_stage"],
        "eri_report_sha256": input_hashes["eri_report"],
    }

    output.mkdir(parents=True, exist_ok=True)
    ledger_path = output / "observation-eligibility-ledger.jsonl"
    bridge_path = output / "eligibility-bridge.json"
    exclusion_path = output / "exclusion-reason-summary.json"
    selection_path = output / "selection-bias-summary.json"
    termination_path = output / "v2-termination-seal.json"
    _write_jsonl(ledger_path, ledger)
    _write_json(
        bridge_path,
        {
            "schema_version": "moatrader-eri-eligibility-bridge-v2/1",
            "status": "AUDITED_FROM_SEALED_V2_IDS",
            "stages": bridge,
            "stage_definitions": {
                "evidence_eligible": "Common Full/Core Nobs>=2 Evidence Index observations",
                "price_pit_available": (
                    "Exact signal open/common FCFF-comparable security/size bucket/two valid "
                    "PIT annual financial histories/positive market inputs"
                ),
                "reverse_valuation_available": "Frozen one-driver reverse DCF solved",
                "t63_snapshot_available": (
                    "Exact t+63 session plus target price/PIT realized financial/equity bridge/WACC metadata"
                ),
                "eri_decomposition_valid": (
                    "Realized margin and positive counterfactual equity/enterprise ERI bridge validate"
                ),
                "final_common": "Exact intersection of label-safe outcomes and final ERI labels",
            },
            "return_data_opened": False,
            "v2_signal_or_threshold_changed": False,
        },
    )
    _write_json(exclusion_path, exclusion_summary)
    _write_json(selection_path, selection_bias)
    _write_json(termination_path, termination_seal)

    coverage_paths: dict[str, Path] = {}
    for name, rows in coverage.items():
        path = output / f"coverage-by-{name.replace('_', '-')}.jsonl"
        _write_jsonl(path, rows)
        coverage_paths[name] = path

    after_input_hashes = {role: sha256_file(path) for role, path in paths.items()}
    after_marcap_hashes = {
        key: sha256_file(Path(str(pre_stage["marcap_sources"][key.split("_", 1)[1]]["path"])))
        for key in marcap_hashes
    }
    if input_hashes != after_input_hashes or marcap_hashes != after_marcap_hashes:
        raise RuntimeError("an authoritative source changed during eligibility auditing")

    status = {
        "schema_version": "moatrader-eri-eligibility-bridge-stage-v2/1",
        "status": "V2_TERMINATED_ERI_ELIGIBILITY_BRIDGE_AUDITED",
        "v2_promotion_status": "FAILED_AND_SEALED",
        **{f"{key}_count": value for key, value in expected_counts.items()},
        "evidence_eligible_issuer_count": baseline_issuers,
        "final_common_issuer_count": final_issuers,
        "route_sector_size_year_issuer_model_status_audited": True,
        "observation_level_exclusion_ledger_written": True,
        "sealed_source_provenance_reused_and_verified": True,
        "source_files_modified": False,
        "return_data_opened": False,
        "v2_thresholds_changed": False,
        "v2_index_formula_changed": False,
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
    }
    stage_path = output / "stage-status.json"
    _write_json(stage_path, status)
    output_paths = {
        "ledger": ledger_path,
        "bridge": bridge_path,
        "exclusion_summary": exclusion_path,
        "selection_bias": selection_path,
        "termination_seal": termination_path,
        "stage": stage_path,
        **{f"coverage_{name}": path for name, path in coverage_paths.items()},
    }
    _write_json(
        output / "audit-manifest.json",
        {
            **status,
            "audit_as_of": audit_as_of,
            "git_commit": commit,
            "worktree_dirty": False,
            "input_paths": {role: str(path.resolve()) for role, path in paths.items()},
            "input_hashes": input_hashes,
            "marcap_input_hashes": marcap_hashes,
            "output_hashes": {
                role: sha256_file(path) for role, path in output_paths.items()
            },
            "source_integrity_verification_status": "PASS_NO_SOURCE_MUTATION",
            "sector_basis": "CURRENT_2026_KRX_KIND_NON_PIT_DESCRIPTIVE_ONLY",
        },
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit sealed V2 ERI eligibility from 37,014 Evidence rows to final labels."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--full-index-build", type=Path, required=True)
    parser.add_argument("--core-index-build", type=Path, required=True)
    parser.add_argument("--pre-outcome-build", type=Path, required=True)
    parser.add_argument("--feature-build", type=Path, required=True)
    parser.add_argument("--outcome-vault-build", type=Path, required=True)
    parser.add_argument("--label-safe-outcome-build", type=Path, required=True)
    parser.add_argument("--eri-build", type=Path, required=True)
    parser.add_argument("--sector-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-as-of", default=date.today().isoformat())
    args = parser.parse_args()
    result = audit_eri_eligibility_bridge_v2(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
