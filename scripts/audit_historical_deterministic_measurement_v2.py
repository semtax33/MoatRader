from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from statistics import fmean, pvariance
from typing import Any

from moatrader.adapters.html import decode_html_document
from moatrader.expectations.future_eri import EvidenceState, OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import sha256_file
from moatrader.expectations.historical_evidence_v2 import (
    AxisSignedScoreRoleV2,
    PITApplicabilityRulesV2,
    SparseAxisAvailabilityV2,
)
from scripts.build_historical_deterministic_pit_evidence_v2 import (
    LastGroundedDeterministicBasisInputV2,
    PITOperatingPairInputV2,
)
from scripts.build_historical_sparse_features_v2 import DeterministicAxisEvidenceInputV2


D = Decimal
PRIMARY_DETERMINISTIC_AXES = (
    OperatingEvidenceAxis.MARGIN,
    OperatingEvidenceAxis.INVENTORY_MISMATCH,
    OperatingEvidenceAxis.BACKLOG,
)
AXIS_METRICS = {
    OperatingEvidenceAxis.MARGIN: ("revenue", "operating_profit"),
    OperatingEvidenceAxis.INVENTORY_MISMATCH: ("inventory", "revenue"),
    OperatingEvidenceAxis.BACKLOG: ("backlog",),
    OperatingEvidenceAxis.CAPACITY_CAPEX: ("capex", "revenue", "ppe", "assets"),
}
REPORT_CODE_NAMES = {
    "11011": "ANNUAL",
    "11012": "HALF_YEAR",
    "11013": "Q1",
    "11014": "Q3",
}
_CONSOLIDATED_RE = re.compile(r"연결\s*(?:재무제표|재무상태표|포괄손익계산서)", re.I)
_SEPARATE_RE = re.compile(r"(?:별도\s*)?재무제표|재무상태표", re.I)


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _classification_map(
    path: Path,
) -> dict[tuple[str, OperatingEvidenceAxis], DeterministicAxisEvidenceInputV2]:
    rows = _read_jsonl(path, DeterministicAxisEvidenceInputV2)
    result = {(row.pair_id, row.evidence.axis): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"deterministic evidence keys must be unique: {path}")
    return result


def _breadth_report(
    *,
    pairs: list[PITOperatingPairInputV2],
    evidence: dict[tuple[str, OperatingEvidenceAxis], DeterministicAxisEvidenceInputV2],
) -> dict[str, Any]:
    nobs = Counter({str(index): 0 for index in range(4)})
    breadth = Counter()
    breadth_values: list[float] = []
    co_observation = Counter()
    margin_inventory = 0
    for pair in pairs:
        directions: list[int] = []
        grounded_axes: list[str] = []
        for axis in PRIMARY_DETERMINISTIC_AXES:
            row = evidence[(pair.pair_id, axis)].evidence
            if row.availability != SparseAxisAvailabilityV2.GROUNDED:
                continue
            if row.signed_score_role != AxisSignedScoreRoleV2.PRIMARY_SIGNED_SCORE:
                raise ValueError(f"primary deterministic axis has non-primary score role: {axis}")
            assert row.direction is not None
            directions.append(row.direction.value)
            grounded_axes.append(axis.value)
        nobs[str(len(directions))] += 1
        if {
            OperatingEvidenceAxis.MARGIN.value,
            OperatingEvidenceAxis.INVENTORY_MISMATCH.value,
        }.issubset(grounded_axes):
            margin_inventory += 1
        for left_index, left in enumerate(grounded_axes):
            for right in grounded_axes[left_index + 1 :]:
                co_observation["|".join(sorted((left, right)))] += 1
        if directions:
            value = sum(directions) / len(directions)
            breadth[f"{value:.12g}"] += 1
            breadth_values.append(value)
    return {
        "pair_count": len(pairs),
        "nobs_histogram": dict(sorted(nobs.items())),
        "nobs_at_least_2": sum(value for key, value in nobs.items() if int(key) >= 2),
        "nobs_at_least_2_rate": (
            sum(value for key, value in nobs.items() if int(key) >= 2) / len(pairs)
        ),
        "margin_inventory_both_grounded": margin_inventory,
        "co_observation_counts": dict(sorted(co_observation.items())),
        "signed_breadth_distribution": dict(
            sorted(breadth.items(), key=lambda item: float(item[0]))
        ),
        "signed_breadth_observed_count": len(breadth_values),
        "signed_breadth_unique_count": len(set(breadth_values)),
        "signed_breadth_mean": fmean(breadth_values) if breadth_values else None,
        "signed_breadth_population_variance": (
            pvariance(breadth_values) if len(breadth_values) > 1 else None
        ),
        "signed_breadth_population_std": (
            math.sqrt(pvariance(breadth_values)) if len(breadth_values) > 1 else None
        ),
        "capex_included": False,
    }


def _direction(value: Decimal, tolerance: Decimal) -> EvidenceState:
    # Inventory mismatch is economically inverted: slower inventory than revenue is bullish.
    raw = (value > tolerance) - (value < -tolerance)
    return EvidenceState(-raw)


def _inventory_audit(
    *,
    pairs: list[PITOperatingPairInputV2],
    evidence: dict[tuple[str, OperatingEvidenceAxis], DeterministicAxisEvidenceInputV2],
    last_grounded: dict[
        tuple[str, OperatingEvidenceAxis], LastGroundedDeterministicBasisInputV2
    ],
    rules: PITApplicabilityRulesV2,
    sample_size: int,
    seed: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    population: list[dict[str, Any]] = []
    direction_counts = Counter()
    mismatch_count = 0
    pair_lookup = {pair.pair_id: pair for pair in pairs}
    for (pair_id, axis), wrapped in evidence.items():
        if axis != OperatingEvidenceAxis.INVENTORY_MISMATCH:
            continue
        row = wrapped.evidence
        if row.availability != SparseAxisAvailabilityV2.GROUNDED:
            continue
        pair = pair_lookup[pair_id]
        basis = last_grounded.get((pair_id, axis))
        previous = basis.previous if basis is not None else pair.previous
        current = pair.current
        if previous.inventory in (None, D(0)) or previous.revenue in (None, D(0)):
            raise ValueError(f"grounded inventory row has invalid previous denominator: {pair_id}")
        if current.inventory is None or current.revenue is None:
            raise ValueError(f"grounded inventory row lacks current evidence: {pair_id}")
        inventory_growth = current.inventory / previous.inventory - D(1)
        revenue_growth = current.revenue / previous.revenue - D(1)
        mismatch = inventory_growth - revenue_growth
        expected = _direction(mismatch, rules.inventory_mismatch_neutral_tolerance)
        assert row.direction is not None
        matches = expected == row.direction
        mismatch_count += int(not matches)
        direction_counts[str(row.direction.value)] += 1
        population.append(
            {
                "pair_id": pair_id,
                "issuer_id": pair.current.issuer_id,
                "previous_fiscal_period_end": previous.fiscal_period_end.isoformat(),
                "current_fiscal_period_end": current.fiscal_period_end.isoformat(),
                "previous_inventory": str(previous.inventory),
                "current_inventory": str(current.inventory),
                "previous_revenue": str(previous.revenue),
                "current_revenue": str(current.revenue),
                "inventory_growth": str(inventory_growth),
                "revenue_growth": str(revenue_growth),
                "raw_inventory_mismatch": str(mismatch),
                "assigned_direction": row.direction.value,
                "expected_economic_polarity": expected.value,
                "polarity_matches": matches,
                "economic_interpretation": (
                    "BULLISH_INVENTORY_GROWS_SLOWER_THAN_REVENUE"
                    if expected == EvidenceState.IMPROVING
                    else "BEARISH_INVENTORY_GROWS_FASTER_THAN_REVENUE"
                    if expected == EvidenceState.WEAKENING
                    else "NEUTRAL_WITHIN_FROZEN_TOLERANCE"
                ),
                "previous_evidence_basis": row.previous_evidence_basis.value,
                "prior_age_days": row.prior_age_days,
            }
        )
    if mismatch_count:
        raise ValueError(f"inventory polarity mismatch detected: {mismatch_count}")
    strata = sorted(direction_counts, key=int)
    base, remainder = divmod(sample_size, len(strata))
    selected: list[dict[str, Any]] = []
    for index, direction in enumerate(strata):
        target = base + (1 if index < remainder else 0)
        rows = [row for row in population if str(row["assigned_direction"]) == direction]
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{seed}|{row['pair_id']}".encode("utf-8")
            ).hexdigest()
        )
        if len(rows) < target:
            raise ValueError(f"not enough inventory rows for direction stratum {direction}")
        selected.extend(rows[:target])
    selected.sort(key=lambda row: (int(row["assigned_direction"]), row["pair_id"]))
    return (
        {
            "population_count": len(population),
            "direction_distribution": dict(sorted(direction_counts.items(), key=lambda x: int(x[0]))),
            "polarity_mismatch_count": mismatch_count,
            "sample_count": len(selected),
            "sample_direction_distribution": dict(
                sorted(
                    Counter(str(row["assigned_direction"]) for row in selected).items(),
                    key=lambda item: int(item[0]),
                )
            ),
            "positive_direction_definition": (
                "INVENTORY_GROWTH_MINUS_REVENUE_GROWTH_BELOW_NEGATIVE_TOLERANCE"
            ),
            "negative_direction_definition": (
                "INVENTORY_GROWTH_MINUS_REVENUE_GROWTH_ABOVE_POSITIVE_TOLERANCE"
            ),
            "neutral_tolerance": str(rules.inventory_mismatch_neutral_tolerance),
            "sample_selection": "DETERMINISTIC_HASH_WITHIN_ASSIGNED_DIRECTION",
        },
        selected,
    )


def _origin_pattern(source_map: dict[str, Any], side: str) -> str:
    origins = {
        payload["origin"]
        for payload in source_map["sources"].values()
        if payload["side"] == side
    }
    arcana = {
        "ARCANA_BUSINESS_HTML",
        "ARCANA_FINANCE_COMMENT_HTML",
        "ARCANA_FINANCE_STATEMENT_HTML",
    }
    has_original = "MOATRADER_OPENDART_ARCHIVE" in origins
    if arcana.issubset(origins):
        return "ARCANA_ALL_3_PLUS_ORIGINAL" if has_original else "ARCANA_ALL_3"
    if origins & arcana:
        return "ARCANA_PARTIAL_PLUS_ORIGINAL" if has_original else "ARCANA_PARTIAL"
    if has_original:
        return "MOATRADER_ORIGINAL_ONLY"
    return "NO_SUPPORTED_SOURCE"


def _financial_statement_path(source_map: dict[str, Any], side: str) -> str | None:
    return next(
        (
            payload["path"]
            for payload in source_map["sources"].values()
            if payload["side"] == side
            and payload["origin"] == "ARCANA_FINANCE_STATEMENT_HTML"
        ),
        None,
    )


def _fs_scope(path: str | None, cache: dict[str, str]) -> str:
    if path is None:
        return "NO_ARCANA_FINANCE_STATEMENT"
    if path in cache:
        return cache[path]
    raw = Path(path).read_bytes()
    document, _ = decode_html_document(raw)
    has_consolidated = bool(_CONSOLIDATED_RE.search(document))
    has_statement = bool(_SEPARATE_RE.search(document))
    value = (
        "CONSOLIDATED_PRESENT"
        if has_consolidated
        else "SEPARATE_OR_GENERIC_ONLY"
        if has_statement
        else "UNKNOWN_STATEMENT_SCOPE"
    )
    cache[path] = value
    return value


def _pair_metadata(
    *,
    filing_pair_input: Path,
    pair_source_map_input: Path,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    scope_cache: dict[str, str] = {}
    with filing_pair_input.open("r", encoding="utf-8") as filings, pair_source_map_input.open(
        "r", encoding="utf-8"
    ) as sources:
        for filing_line, source_line in zip(filings, sources, strict=True):
            pair = json.loads(filing_line)
            source_map = json.loads(source_line)
            previous = pair["previous"]
            current = pair["current"]
            previous_scope = _fs_scope(
                _financial_statement_path(source_map, "previous"), scope_cache
            )
            current_scope = _fs_scope(
                _financial_statement_path(source_map, "current"), scope_cache
            )
            result[pair["pair_id"]] = {
                "year": str(current["fiscal_period_end"])[:4],
                "issuer": pair["ticker"],
                "filing_type": REPORT_CODE_NAMES.get(
                    str(current.get("report_code") or ""),
                    str(current.get("report_code") or "UNKNOWN"),
                ),
                "fs_type": f"{previous_scope}->{current_scope}",
                "source_format": (
                    f"{_origin_pattern(source_map, 'previous')}->"
                    f"{_origin_pattern(source_map, 'current')}"
                ),
            }
    return result


def _missing_side(pair: PITOperatingPairInputV2, axis: OperatingEvidenceAxis) -> str:
    metrics = AXIS_METRICS[axis]
    if axis == OperatingEvidenceAxis.CAPACITY_CAPEX:
        previous_ready = (
            pair.previous.capex is not None and pair.previous.revenue not in (None, D(0))
        ) or (pair.previous.ppe is not None and pair.previous.assets not in (None, D(0)))
        current_ready = (
            pair.current.capex is not None and pair.current.revenue not in (None, D(0))
        ) or (pair.current.ppe is not None and pair.current.assets not in (None, D(0)))
    else:
        previous_ready = all(getattr(pair.previous, metric) is not None for metric in metrics)
        current_ready = all(getattr(pair.current, metric) is not None for metric in metrics)
        if axis in {OperatingEvidenceAxis.INVENTORY_MISMATCH, OperatingEvidenceAxis.BACKLOG}:
            first = metrics[0]
            previous_ready = previous_ready and getattr(pair.previous, first) != D(0)
        if axis in {OperatingEvidenceAxis.MARGIN, OperatingEvidenceAxis.INVENTORY_MISMATCH}:
            previous_ready = previous_ready and pair.previous.revenue != D(0)
            if axis == OperatingEvidenceAxis.MARGIN:
                current_ready = current_ready and pair.current.revenue != D(0)
    if previous_ready and current_ready:
        return "BOTH_PRESENT_OTHER_EXTRACTION_CONSTRAINT"
    if not previous_ready and not current_ready:
        return "BOTH_SIDES_MISSING"
    return "PREVIOUS_MISSING" if not previous_ready else "CURRENT_MISSING"


def _dimension_rows(
    denominator: Counter[str],
    failures: Counter[str],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    values = [
        {
            "group": group,
            "pair_count": denominator[group],
            "failure_count": count,
            "failure_rate": count / denominator[group],
        }
        for group, count in failures.items()
    ]
    values.sort(key=lambda row: (-row["failure_count"], -row["failure_rate"], row["group"]))
    return values[:limit] if limit is not None else values


def _failure_concentration(
    *,
    pairs: list[PITOperatingPairInputV2],
    evidence: dict[tuple[str, OperatingEvidenceAxis], DeterministicAxisEvidenceInputV2],
    metadata: dict[str, dict[str, str]],
) -> dict[str, Any]:
    pair_lookup = {pair.pair_id: pair for pair in pairs}
    dimensions = ("year", "issuer", "filing_type", "fs_type", "source_format")
    denominators = {
        dimension: Counter(row[dimension] for row in metadata.values())
        for dimension in dimensions
    }
    result: dict[str, Any] = {}
    for axis in (
        OperatingEvidenceAxis.MARGIN,
        OperatingEvidenceAxis.INVENTORY_MISMATCH,
        OperatingEvidenceAxis.BACKLOG,
        OperatingEvidenceAxis.CAPACITY_CAPEX,
    ):
        failures = [
            pair_id
            for (pair_id, row_axis), wrapped in evidence.items()
            if row_axis == axis
            and wrapped.evidence.abstention_reason is not None
            and wrapped.evidence.abstention_reason.value == "TABLE_EXTRACTION_FAIL"
        ]
        by_dimension: dict[str, list[dict[str, Any]]] = {}
        for dimension in dimensions:
            counts = Counter(metadata[pair_id][dimension] for pair_id in failures)
            by_dimension[dimension] = _dimension_rows(
                denominators[dimension],
                counts,
                limit=50 if dimension == "issuer" else None,
            )
        result[axis.value] = {
            "failure_count": len(failures),
            "failure_rate": len(failures) / len(pairs),
            "missing_side_distribution": dict(
                sorted(Counter(_missing_side(pair_lookup[pair_id], axis) for pair_id in failures).items())
            ),
            "by_dimension": by_dimension,
        }
    return result


def audit_deterministic_measurement(
    *,
    pit_pair_input: Path,
    rules_input: Path,
    baseline_evidence_input: Path,
    evidence_input: Path,
    filing_pair_input: Path,
    pair_source_map_input: Path,
    output: Path,
    last_grounded_input: Path | None = None,
    inventory_sample_size: int = 100,
    seed: str = "MOATRADER_V2_INVENTORY_POLARITY_AUDIT_20260821",
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    required = (
        pit_pair_input,
        rules_input,
        baseline_evidence_input,
        evidence_input,
        filing_pair_input,
        pair_source_map_input,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if inventory_sample_size < 3:
        raise ValueError("inventory polarity sample must contain all three directions")
    pairs = _read_jsonl(pit_pair_input, PITOperatingPairInputV2)
    rules = PITApplicabilityRulesV2.model_validate_json(
        rules_input.read_text(encoding="utf-8")
    )
    baseline = _classification_map(baseline_evidence_input)
    final = _classification_map(evidence_input)
    expected_keys = {
        (pair.pair_id, axis)
        for pair in pairs
        for axis in (
            OperatingEvidenceAxis.MARGIN,
            OperatingEvidenceAxis.INVENTORY_MISMATCH,
            OperatingEvidenceAxis.BACKLOG,
            OperatingEvidenceAxis.CAPACITY_CAPEX,
        )
    }
    if set(baseline) != expected_keys or set(final) != expected_keys:
        raise ValueError("deterministic evidence does not exactly cover four axes for every pair")
    last_rows = (
        _read_jsonl(last_grounded_input, LastGroundedDeterministicBasisInputV2)
        if last_grounded_input is not None
        else []
    )
    last_grounded = {(row.pair_id, row.axis): row for row in last_rows}
    if len(last_grounded) != len(last_rows):
        raise ValueError("last-grounded audit input has duplicate pair-axis keys")

    inventory, sample = _inventory_audit(
        pairs=pairs,
        evidence=final,
        last_grounded=last_grounded,
        rules=rules,
        sample_size=inventory_sample_size,
        seed=seed,
    )
    metadata = _pair_metadata(
        filing_pair_input=filing_pair_input,
        pair_source_map_input=pair_source_map_input,
    )
    if set(metadata) != {pair.pair_id for pair in pairs}:
        raise ValueError("filing/source metadata does not match the PIT pair universe")
    baseline_breadth = _breadth_report(pairs=pairs, evidence=baseline)
    final_breadth = _breadth_report(pairs=pairs, evidence=final)
    report = {
        "schema_version": "moatrader-deterministic-measurement-audit-v2/1",
        "status": "DETERMINISTIC_MEASUREMENT_AUDITED_OUTCOME_BLIND",
        "pair_count": len(pairs),
        "inventory_polarity": inventory,
        "baseline_primary_deterministic_features": baseline_breadth,
        "selection_2_primary_deterministic_features": final_breadth,
        "selection_2_nobs_at_least_2_increment": (
            final_breadth["nobs_at_least_2"] - baseline_breadth["nobs_at_least_2"]
        ),
        "baseline_table_extraction_failure_concentration": _failure_concentration(
            pairs=pairs,
            evidence=baseline,
            metadata=metadata,
        ),
        "selection_2_table_extraction_failure_concentration": _failure_concentration(
            pairs=pairs,
            evidence=final,
            metadata=metadata,
        ),
        "fs_type_definition": (
            "ARCANA finance-statement document marker presence; consolidated marker takes priority"
        ),
        "source_format_definition": (
            "per-side availability pattern of Arcana three HTML sections and MoatRader original ZIP"
        ),
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
        "input_hashes": {
            "pit_pairs": sha256_file(pit_pair_input),
            "rules": sha256_file(rules_input),
            "baseline_evidence": sha256_file(baseline_evidence_input),
            "selection_2_evidence": sha256_file(evidence_input),
            "filing_pairs": sha256_file(filing_pair_input),
            "pair_source_map": sha256_file(pair_source_map_input),
            "last_grounded": (
                sha256_file(last_grounded_input) if last_grounded_input is not None else None
            ),
        },
        "source_files_modified": False,
        "source_write_operations": 0,
    }
    output.mkdir(parents=True, exist_ok=True)
    sample_path = output / "inventory-polarity-audit-sample-100.csv"
    with sample_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample[0]))
        writer.writeheader()
        writer.writerows(sample)
    report["inventory_sample_sha256"] = sha256_file(sample_path)
    report_path = output / "deterministic-measurement-audit.json"
    _write_json(report_path, report)
    stage = {
        "schema_version": "moatrader-deterministic-measurement-audit-stage-v2/1",
        "status": report["status"],
        "pair_count": len(pairs),
        "inventory_sample_count": len(sample),
        "inventory_polarity_mismatch_count": inventory["polarity_mismatch_count"],
        "baseline_nobs_at_least_2": baseline_breadth["nobs_at_least_2"],
        "selection_2_nobs_at_least_2": final_breadth["nobs_at_least_2"],
        "audit_report_sha256": sha256_file(report_path),
        "inventory_sample_sha256": sha256_file(sample_path),
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
    }
    _write_json(output / "stage-status.json", stage)
    return stage


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit V2 inventory polarity, deterministic-only Nobs/SignedBreadth, and "
            "table-extraction-failure concentration without opening outcomes."
        )
    )
    parser.add_argument("--pit-pair-input", type=Path, required=True)
    parser.add_argument("--rules-input", type=Path, required=True)
    parser.add_argument("--baseline-evidence-input", type=Path, required=True)
    parser.add_argument("--evidence-input", type=Path, required=True)
    parser.add_argument("--filing-pair-input", type=Path, required=True)
    parser.add_argument("--pair-source-map-input", type=Path, required=True)
    parser.add_argument("--last-grounded-input", type=Path)
    parser.add_argument("--inventory-sample-size", type=int, default=100)
    parser.add_argument("--seed", default="MOATRADER_V2_INVENTORY_POLARITY_AUDIT_20260821")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit_deterministic_measurement(
        pit_pair_input=args.pit_pair_input,
        rules_input=args.rules_input,
        baseline_evidence_input=args.baseline_evidence_input,
        evidence_input=args.evidence_input,
        filing_pair_input=args.filing_pair_input,
        pair_source_map_input=args.pair_source_map_input,
        output=args.output,
        last_grounded_input=args.last_grounded_input,
        inventory_sample_size=args.inventory_sample_size,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
