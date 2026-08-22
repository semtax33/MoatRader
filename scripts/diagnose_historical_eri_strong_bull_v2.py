from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from moatrader.expectations.future_eri import (
    EvidenceIndexFeatureDatasetSealV2,
    EvidenceIndexFutureEriFeatureRowV2,
    FutureEriLabelV1,
    FutureEriOutcomeInputV1,
    model_sha256,
    roll_forward_evidence_index_expectations_v2,
)
from moatrader.expectations.historical_evidence import sha256_file
from moatrader.valuation.economic_dcf import EconomicDcfEngine
from scripts.prepare_historical_evidence_index_eri_inputs_v2 import _git_state


D = Decimal
BAND_ORDER = {
    "STRONG_BEAR": 0,
    "BEAR": 1,
    "NEUTRAL": 2,
    "BULL": 3,
    "STRONG_BULL": 4,
}
COMPONENT_FIELDS = (
    "realization_component",
    "discount_rate_component",
    "expectation_revision_component",
    "total_log_price_bridge",
    "market_total_log_price_bridge",
    "enterprise_realization_component",
    "enterprise_discount_rate_component",
    "enterprise_expectation_revision_component",
    "enterprise_total_log_bridge",
    "capital_structure_bridge_effect",
    "signal_reverse_fit_log_gap",
    "wacc_change",
    "realized_minus_signal_nopat_margin",
    "log_realized_to_signal_revenue",
)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
TOLERANCE = D("1e-22")


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []
    raw = json.loads(text) if text.lstrip().startswith("[") else [
        json.loads(line) for line in text.splitlines() if line.strip()
    ]
    if not isinstance(raw, list):
        raise ValueError(f"record collection required: {path}")
    return [dict(row) for row in raw]


def _index(
    records: Sequence[dict[str, Any]], *, source: str
) -> dict[str, dict[str, Any]]:
    result = {str(row["observation_id"]): row for row in records}
    if len(result) != len(records):
        raise ValueError(f"duplicate observation_id in {source}")
    return result


def _finite(value: Decimal, *, field: str) -> Decimal:
    if not value.is_finite():
        raise ValueError(f"non-finite decomposition value: {field}")
    return value


def decompose_eri_bridge_v2(
    *,
    feature: EvidenceIndexFutureEriFeatureRowV2,
    outcome: FutureEriOutcomeInputV1,
    label: FutureEriLabelV1,
    engine: EconomicDcfEngine | None = None,
) -> dict[str, Decimal]:
    """Build an exact log bridge without changing the frozen V2 ERI label.

    Realization re-anchors the base financials, net debt, shares, and elapsed CAP at
    the signal WACC. Discount rate then changes only WACC to the target policy WACC.
    Expectation revision is the already-sealed Future ERI residual. Their sum equals
    the target price divided by the signal fitted value, not a new ranking signal.
    """

    if not (
        feature.observation_id == outcome.observation_id == label.observation_id
    ):
        raise ValueError("feature/outcome/label observation IDs must match")
    dcf = engine or EconomicDcfEngine()
    frozen = feature.frozen_expectation_assumptions
    rolled = roll_forward_evidence_index_expectations_v2(feature, outcome)
    realization_only = rolled.model_copy(update={"wacc": frozen.wacc})
    signal = dcf.value(frozen)
    realization = dcf.value(realization_only)
    counterfactual = dcf.value(rolled)
    values = {
        "signal_value_per_share": signal.fair_value_per_share,
        "realization_value_per_share": realization.fair_value_per_share,
        "counterfactual_value_per_share": counterfactual.fair_value_per_share,
        "signal_enterprise_value": signal.enterprise_value,
        "realization_enterprise_value": realization.enterprise_value,
        "counterfactual_enterprise_value": counterfactual.enterprise_value,
    }
    non_positive = [name for name, value in values.items() if value <= 0]
    if non_positive:
        raise ValueError(f"decomposition requires positive fitted values: {non_positive}")
    if abs(counterfactual.fair_value_per_share - label.counterfactual_value_per_share) > (
        TOLERANCE
    ):
        raise ValueError("diagnostic counterfactual does not reproduce the sealed ERI label")

    actual_enterprise = (
        outcome.actual_market_price * outcome.realized_state.diluted_shares
        + outcome.realized_state.net_debt
    )
    if actual_enterprise <= 0:
        raise ValueError("diagnostic actual enterprise value must be positive")
    realization_component = (
        realization.fair_value_per_share / signal.fair_value_per_share
    ).ln()
    discount_component = (
        counterfactual.fair_value_per_share / realization.fair_value_per_share
    ).ln()
    expectation_component = (
        outcome.actual_market_price / counterfactual.fair_value_per_share
    ).ln()
    signal_market_price = feature.expectation_state.market_price
    total_bridge = (outcome.actual_market_price / signal.fair_value_per_share).ln()
    signal_fit_gap = (signal.fair_value_per_share / signal_market_price).ln()
    market_total_bridge = (outcome.actual_market_price / signal_market_price).ln()
    enterprise_realization = (
        realization.enterprise_value / signal.enterprise_value
    ).ln()
    enterprise_discount = (
        counterfactual.enterprise_value / realization.enterprise_value
    ).ln()
    enterprise_expectation = (
        actual_enterprise / counterfactual.enterprise_value
    ).ln()
    enterprise_total = (actual_enterprise / signal.enterprise_value).ln()
    if abs(
        realization_component + discount_component + expectation_component - total_bridge
    ) > TOLERANCE:
        raise ValueError("equity decomposition is not exactly additive")
    if abs(signal_fit_gap + total_bridge - market_total_bridge) > TOLERANCE:
        raise ValueError("signal-fit adjusted market-price bridge is not exactly additive")
    if abs(
        enterprise_realization
        + enterprise_discount
        + enterprise_expectation
        - enterprise_total
    ) > TOLERANCE:
        raise ValueError("enterprise decomposition is not exactly additive")
    if abs(expectation_component - label.future_eri) > TOLERANCE:
        raise ValueError("expectation-revision component changed the sealed Future ERI")
    if abs(enterprise_expectation - label.enterprise_future_eri) > TOLERANCE:
        raise ValueError("enterprise expectation revision changed the sealed Future ERI")
    return {
        **{name: _finite(value, field=name) for name, value in values.items()},
        "actual_enterprise_value": actual_enterprise,
        "realization_component": realization_component,
        "discount_rate_component": discount_component,
        "expectation_revision_component": expectation_component,
        "total_log_price_bridge": total_bridge,
        "market_total_log_price_bridge": market_total_bridge,
        "enterprise_realization_component": enterprise_realization,
        "enterprise_discount_rate_component": enterprise_discount,
        "enterprise_expectation_revision_component": enterprise_expectation,
        "enterprise_total_log_bridge": enterprise_total,
        "capital_structure_bridge_effect": label.capital_structure_bridge_effect,
        "signal_reverse_fit_log_gap": signal_fit_gap,
        "wacc_change": rolled.wacc - frozen.wacc,
        "realized_minus_signal_nopat_margin": (
            outcome.realized_state.base_nopat_margin - frozen.base_nopat_margin
        ),
        "log_realized_to_signal_revenue": (
            outcome.realized_state.base_revenue / frozen.base_revenue
        ).ln(),
    }


def _statistics(records: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    values = np.asarray([float(row[field]) for row in records], dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError(f"finite non-empty values required for {field}")
    quantiles = np.quantile(values, QUANTILES, method="linear")
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "p10": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p50": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p90": float(quantiles[4]),
        "negative_share": float(np.mean(values < -1e-12)),
        "positive_share": float(np.mean(values > 1e-12)),
        "nonzero_absolute_gt_1e_12_share": float(np.mean(np.abs(values) > 1e-12)),
        "absolute_gt_0_01_share": float(np.mean(np.abs(values) > 0.01)),
        "absolute_gt_0_05_share": float(np.mean(np.abs(values) > 0.05)),
    }


def _group_summaries(
    records: Sequence[dict[str, Any]], *, dimensions: Sequence[str]
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        key = tuple(
            str(row.get(field)) if row.get(field) not in {None, ""} else "UNKNOWN"
            for field in dimensions
        )
        grouped[key].append(row)
    result: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        result.append(
            {
                "schema_version": "moatrader-eri-nonlinearity-group-diagnostic-v2/1",
                "dimensions": dict(zip(dimensions, key, strict=True)),
                "observation_count": len(rows),
                "issuer_count": len({str(row["issuer_id"]) for row in rows}),
                "small_cell_less_than_20": len(rows) < 20,
                "negative_expectation_revision_share": sum(
                    float(row["expectation_revision_component"]) < 0 for row in rows
                )
                / len(rows),
                "component_statistics": {
                    field: _statistics(rows, field) for field in COMPONENT_FIELDS
                },
            }
        )

    def order(row: dict[str, Any]) -> tuple[Any, ...]:
        values = row["dimensions"]
        return tuple(
            (
                BAND_ORDER.get(values[field], 999)
                if field == "full_evidence_band"
                else values[field]
            )
            for field in dimensions
        )

    return sorted(result, key=order)


def _median(summary: dict[str, Any], field: str) -> float:
    return float(summary["component_statistics"][field]["p50"])


def _strong_bull_interpretation(
    *,
    band_summaries: Sequence[dict[str, Any]],
    coverage_selection: dict[str, Any],
) -> dict[str, Any]:
    by_band = {
        str(row["dimensions"]["full_evidence_band"]): row for row in band_summaries
    }
    required = set(BAND_ORDER)
    if set(by_band) != required:
        raise ValueError("five Full Evidence bands are required for Strong Bull diagnosis")
    strong_bull = by_band["STRONG_BULL"]
    bull = by_band["BULL"]
    strong_bear = by_band["STRONG_BEAR"]
    fields = (
        "realization_component",
        "discount_rate_component",
        "expectation_revision_component",
        "total_log_price_bridge",
        "market_total_log_price_bridge",
        "enterprise_realization_component",
        "enterprise_discount_rate_component",
        "enterprise_expectation_revision_component",
        "capital_structure_bridge_effect",
    )
    versus_bull = {
        field: _median(strong_bull, field) - _median(bull, field) for field in fields
    }
    versus_strong_bear = {
        field: _median(strong_bull, field) - _median(strong_bear, field)
        for field in fields
    }
    realization_delta = versus_bull["realization_component"]
    discount_delta = versus_bull["discount_rate_component"]
    revision_delta = versus_bull["expectation_revision_component"]
    median_tolerance = 1e-12
    if revision_delta >= 0:
        classification = "NO_STRONG_BULL_REVISION_BREAK_RELATIVE_TO_BULL"
    elif (
        abs(realization_delta) <= median_tolerance
        and abs(discount_delta) <= median_tolerance
    ):
        classification = (
            "WEAKER_EXPECTATION_REVISION_WITH_NO_MEDIAN_REALIZATION_OR_WACC_SHIFT"
        )
    elif realization_delta > 0 and discount_delta >= 0:
        classification = (
            "POSITIVE_REALIZATION_BUT_WEAKER_RESIDUAL_EXPECTATION_REVISION_PREPRICING_CANDIDATE"
        )
    elif realization_delta > 0 and discount_delta < 0:
        classification = (
            "POSITIVE_REALIZATION_OFFSET_BY_DISCOUNT_RATE_AND_WEAKER_EXPECTATION_REVISION"
        )
    elif realization_delta < -median_tolerance:
        classification = "WEAKER_REALIZATION_AND_WEAKER_EXPECTATION_REVISION"
    else:
        classification = "WEAKER_EXPECTATION_REVISION_RESIDUAL"
    coverage = coverage_selection["strong_bull_selection"]
    return {
        "classification": classification,
        "strong_bull_observation_count": strong_bull["observation_count"],
        "strong_bull_negative_expectation_revision_share": strong_bull[
            "negative_expectation_revision_share"
        ],
        "strong_bull_medians": {
            field: _median(strong_bull, field) for field in fields
        },
        "bull_medians": {field: _median(bull, field) for field in fields},
        "strong_bull_minus_bull_median": versus_bull,
        "strong_bull_minus_strong_bear_median": versus_strong_bear,
        "coverage_selection": coverage,
        "strong_bull_overrepresented_in_final_panel": (
            float(coverage["final_minus_baseline_share"]) > 0
        ),
        "selection_inference": (
            "Strong Bull is underrepresented in the final ERI panel, so its median break is "
            "not caused by simple Strong Bull over-sampling. Coverage compression still limits "
            "generalization beyond the 279 final issuers."
        ),
        "causal_claim_allowed": False,
        "v2_retuning_allowed": False,
    }


def _measurement_quality(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    def coverage(field: str, threshold: float = 1e-12) -> dict[str, Any]:
        values = np.asarray([float(row[field]) for row in rows], dtype=float)
        count = int(np.sum(np.abs(values) > threshold))
        return {
            "nonzero_count": count,
            "nonzero_share": count / len(values),
            "median": float(np.median(values)),
        }

    fit = np.asarray(
        [abs(float(row["signal_reverse_fit_log_gap"])) for row in rows], dtype=float
    )
    by_driver: dict[str, Any] = {}
    for driver in sorted({str(row["reverse_dcf_driver"]) for row in rows}):
        selected = [row for row in rows if str(row["reverse_dcf_driver"]) == driver]
        values = np.asarray(
            [abs(float(row["signal_reverse_fit_log_gap"])) for row in selected],
            dtype=float,
        )
        by_driver[driver] = {
            "observation_count": len(selected),
            "median_absolute_log_gap": float(np.median(values)),
            "absolute_log_gap_gt_0_01_share": float(np.mean(values > 0.01)),
            "absolute_log_gap_gt_0_05_share": float(np.mean(values > 0.05)),
            "maximum_absolute_log_gap": float(np.max(values)),
        }
    reported_zero = [
        row
        for row in rows
        if abs(float(row["reverse_solution_reported_relative_price_error"])) <= 1e-12
    ]
    reported_zero_but_material = [
        row
        for row in reported_zero
        if abs(float(row["signal_reverse_fit_log_gap"])) > 0.01
    ]
    cap = [row for row in rows if str(row["reverse_dcf_driver"]) == "CAP"]
    cap_fractional_to_integer = [
        row
        for row in cap
        if abs(
            float(row["reverse_solution_implied"])
            - float(row["frozen_driver_value"])
        )
        > 1e-12
    ]
    return {
        "status": (
            "MATERIAL_SIGNAL_REVERSE_FIT_MISMATCH"
            if float(np.mean(fit > 0.01)) > 0.10
            else "LOW_SIGNAL_REVERSE_FIT_MISMATCH"
        ),
        "signal_reverse_fit": {
            "observation_count": len(rows),
            "median_absolute_log_gap": float(np.median(fit)),
            "absolute_log_gap_gt_0_01_count": int(np.sum(fit > 0.01)),
            "absolute_log_gap_gt_0_01_share": float(np.mean(fit > 0.01)),
            "absolute_log_gap_gt_0_05_count": int(np.sum(fit > 0.05)),
            "absolute_log_gap_gt_0_05_share": float(np.mean(fit > 0.05)),
            "maximum_absolute_log_gap": float(np.max(fit)),
            "by_reverse_dcf_driver": by_driver,
            "diagnostic_interpretation": (
                "The three-component bridge starts from fitted signal value, not signal "
                "market price. The signal-fit residual is therefore reported as a fourth "
                "measurement component and is not silently assigned to realization."
            ),
        },
        "reverse_solver_provenance_consistency": {
            "status": (
                "FAIL_REPORTED_ZERO_ERROR_NOT_REPRODUCED_BY_FROZEN_ASSUMPTIONS"
                if reported_zero_but_material
                else "PASS"
            ),
            "reported_zero_relative_price_error_count": len(reported_zero),
            "reported_zero_but_actual_abs_log_gap_gt_0_01_count": len(
                reported_zero_but_material
            ),
            "reported_zero_but_actual_abs_log_gap_gt_0_01_share": (
                len(reported_zero_but_material) / len(reported_zero)
                if reported_zero
                else None
            ),
            "cap_driver_count": len(cap),
            "cap_fractional_implied_to_integer_frozen_count": len(
                cap_fractional_to_integer
            ),
            "diagnostic_root_cause": (
                "The sealed solution provenance records modeled_price equal to market price "
                "and relative_price_error=0, but direct revaluation of the sealed frozen "
                "assumptions does not reproduce that price. This is consistent with a "
                "nonlinear grid interpolation being recorded without post-freeze revaluation. "
                "Every CAP solution also carries a fractional implied year into an integer "
                "frozen CAP value. This is measurement diagnosis, not a V2 repair."
            ),
        },
        "t63_measurement_change_coverage": {
            "realization_component": coverage("realization_component"),
            "discount_rate_component": coverage("discount_rate_component"),
            "wacc_change": coverage("wacc_change"),
            "base_nopat_margin_change": coverage(
                "realized_minus_signal_nopat_margin"
            ),
            "base_revenue_change": coverage("log_realized_to_signal_revenue"),
        },
        "business_realization_median_attribution_status": (
            "NOT_IDENTIFIED_AT_T63_WITH_ANNUAL_PIT_SNAPSHOTS"
            if coverage("log_realized_to_signal_revenue")["nonzero_count"] == 0
            else "PARTIALLY_IDENTIFIED"
        ),
    }


def _conditional_strong_bull_deltas(
    summaries: Sequence[dict[str, Any]],
    *,
    group_dimension: str,
    minimum_each_band: int,
) -> dict[str, Any]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in summaries:
        dimensions = row["dimensions"]
        grouped[str(dimensions[group_dimension])][
            str(dimensions["full_evidence_band"])
        ] = row
    comparisons: list[dict[str, Any]] = []
    for group, bands in sorted(grouped.items()):
        bull = bands.get("BULL")
        strong_bull = bands.get("STRONG_BULL")
        if bull is None or strong_bull is None:
            continue
        bull_n = int(bull["observation_count"])
        strong_bull_n = int(strong_bull["observation_count"])
        if min(bull_n, strong_bull_n) < minimum_each_band:
            continue
        bull_median = _median(bull, "expectation_revision_component")
        strong_bull_median = _median(
            strong_bull, "expectation_revision_component"
        )
        comparisons.append(
            {
                "group": group,
                "bull_count": bull_n,
                "strong_bull_count": strong_bull_n,
                "bull_median_expectation_revision": bull_median,
                "strong_bull_median_expectation_revision": strong_bull_median,
                "strong_bull_minus_bull_median": strong_bull_median - bull_median,
            }
        )
    return {
        "group_dimension": group_dimension,
        "minimum_each_band": minimum_each_band,
        "comparable_group_count": len(comparisons),
        "negative_delta_group_count": sum(
            row["strong_bull_minus_bull_median"] < 0 for row in comparisons
        ),
        "positive_or_zero_delta_group_count": sum(
            row["strong_bull_minus_bull_median"] >= 0 for row in comparisons
        ),
        "comparisons": comparisons,
    }


def _robustness_diagnostics(
    *,
    rows: Sequence[dict[str, Any]],
    by_nobs: Sequence[dict[str, Any]],
    by_source: Sequence[dict[str, Any]],
    by_driver: Sequence[dict[str, Any]],
    by_sector: Sequence[dict[str, Any]],
    by_month: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    band_counts = defaultdict(Counter)
    for row in rows:
        band_counts[str(row["full_evidence_band"])][str(row["nobs_bucket"])] += 1
    strong_bull_nobs2_share = (
        band_counts["STRONG_BULL"]["NOBS_2"]
        / sum(band_counts["STRONG_BULL"].values())
    )
    bull_nobs2_share = band_counts["BULL"]["NOBS_2"] / sum(
        band_counts["BULL"].values()
    )
    source_counts = defaultdict(Counter)
    for row in rows:
        source_counts[str(row["full_evidence_band"])][
            str(row["evidence_source_mode"])
        ] += 1
    nobs = _conditional_strong_bull_deltas(
        by_nobs, group_dimension="nobs_bucket", minimum_each_band=20
    )
    source = _conditional_strong_bull_deltas(
        by_source, group_dimension="evidence_source_mode", minimum_each_band=20
    )
    driver = _conditional_strong_bull_deltas(
        by_driver, group_dimension="reverse_dcf_driver", minimum_each_band=20
    )
    sector = _conditional_strong_bull_deltas(
        by_sector, group_dimension="sector", minimum_each_band=5
    )
    month = _conditional_strong_bull_deltas(
        by_month, group_dimension="signal_month", minimum_each_band=5
    )
    nobs_by_group = {row["group"]: row for row in nobs["comparisons"]}
    status = (
        "AGGREGATE_BREAK_NOT_ROBUST_WITHIN_NOBS_STRATA_COMPOSITION_SENSITIVE"
        if nobs_by_group.get("NOBS_3_PLUS", {}).get(
            "strong_bull_minus_bull_median", -1
        )
        >= 0
        else "AGGREGATE_BREAK_PERSISTS_ACROSS_NOBS_STRATA"
    )
    return {
        "status": status,
        "nobs_composition": {
            "strong_bull_nobs2_count": band_counts["STRONG_BULL"]["NOBS_2"],
            "strong_bull_nobs2_share": strong_bull_nobs2_share,
            "bull_nobs2_count": band_counts["BULL"]["NOBS_2"],
            "bull_nobs2_share": bull_nobs2_share,
            "strong_bull_minus_bull_nobs2_share": strong_bull_nobs2_share
            - bull_nobs2_share,
        },
        "evidence_source_composition": {
            "strong_bull": dict(sorted(source_counts["STRONG_BULL"].items())),
            "bull": dict(sorted(source_counts["BULL"].items())),
        },
        "conditional_strong_bull_minus_bull": {
            "nobs": nobs,
            "evidence_source_mode": source,
            "reverse_dcf_driver": driver,
            "sector_minimum_5_each_band": sector,
            "signal_month_minimum_5_each_band": month,
        },
        "diagnostic_interpretation": (
            "The aggregate Strong Bull median break is highly composition-sensitive: "
            "Strong Bull is dominated by Nobs=2 deterministic-only rows, while the "
            "Nobs>=3 stratum does not reproduce the break. Driver, month, and sector "
            "conditional signs are mixed. This does not authorize a V2 threshold change."
        ),
    }


def _input_paths(
    *,
    eri_build: Path,
    outcome_build: Path,
    coverage_audit: Path,
    pre_outcome_build: Path,
) -> dict[str, Path]:
    return {
        "features": eri_build / "features-with-frozen-expectations-pre-outcome.jsonl",
        "feature_seal": eri_build / "feature-seal-pre-outcome.json",
        "labels": eri_build / "future-eri-labels.jsonl",
        "eri_report": eri_build / "dual-evidence-index-eri-report.json",
        "eri_manifest": eri_build / "build-manifest.json",
        "eri_stage": eri_build / "stage-status.json",
        "outcomes": outcome_build / "future-eri-outcomes.jsonl",
        "outcome_stage": outcome_build / "stage-status.json",
        "coverage_ledger": coverage_audit / "observation-eligibility-ledger.jsonl",
        "coverage_selection": coverage_audit / "selection-bias-summary.json",
        "coverage_manifest": coverage_audit / "audit-manifest.json",
        "coverage_stage": coverage_audit / "stage-status.json",
        "v2_termination_seal": coverage_audit / "v2-termination-seal.json",
        "reverse_expectations": pre_outcome_build / "expectations-pre-outcome.jsonl",
        "pre_outcome_stage": pre_outcome_build / "stage-status.json",
        "pre_outcome_seal": pre_outcome_build / "pre-outcome-input-seal.json",
    }


def diagnose_strong_bull_nonlinearity_v2(
    *,
    workspace: Path,
    eri_build: Path,
    outcome_build: Path,
    coverage_audit: Path,
    pre_outcome_build: Path,
    output: Path,
    audit_as_of: str,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    date.fromisoformat(audit_as_of)
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production Strong Bull diagnostic requires a clean worktree")
    paths = _input_paths(
        eri_build=eri_build,
        outcome_build=outcome_build,
        coverage_audit=coverage_audit,
        pre_outcome_build=pre_outcome_build,
    )
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Strong Bull diagnostic inputs are missing: {missing}")
    input_hashes = {role: sha256_file(path) for role, path in paths.items()}
    eri_stage = _read_json(paths["eri_stage"])
    eri_manifest = _read_json(paths["eri_manifest"])
    eri_report = _read_json(paths["eri_report"])
    outcome_stage = _read_json(paths["outcome_stage"])
    coverage_stage = _read_json(paths["coverage_stage"])
    coverage_manifest = _read_json(paths["coverage_manifest"])
    termination = _read_json(paths["v2_termination_seal"])
    pre_outcome_stage = _read_json(paths["pre_outcome_stage"])
    primary = eri_report.get("primary_full", {})
    if not (
        eri_stage.get("status") == "FULL_PRIMARY_MECHANISM_REJECTED_OR_INCONCLUSIVE"
        and primary.get("mechanism_gate_passed") is False
        and int(primary.get("adjacent_median_nondecreasing_count", -1)) == 3
        and coverage_stage.get("status")
        == "V2_TERMINATED_ERI_ELIGIBILITY_BRIDGE_AUDITED"
        and termination.get("status") == "V2_PROMOTION_FAILED_AND_SEALED"
        and termination.get("v2_thresholds_changed") is False
        and termination.get("v2_index_formula_changed") is False
        and coverage_manifest.get("source_integrity_verification_status")
        == "PASS_NO_SOURCE_MUTATION"
        and outcome_stage.get("outcome_count") == eri_stage.get("label_count") == 1640
    ):
        raise ValueError("V2 is not sealed and eligible for post-failure diagnosis")
    sealed_hash_checks = {
        "features": eri_manifest.get("feature_input_sha256") == input_hashes["features"],
        "feature_seal": eri_manifest.get("feature_seal_pre_outcome_sha256")
        == input_hashes["feature_seal"],
        "labels": eri_manifest.get("future_eri_labels_sha256")
        == input_hashes["labels"],
        "report": eri_manifest.get("dual_mechanism_report_sha256")
        == input_hashes["eri_report"],
        "eri_stage": eri_manifest.get("stage_status_sha256")
        == input_hashes["eri_stage"],
        "outcomes": eri_manifest.get("outcome_input_sha256")
        == input_hashes["outcomes"],
        "coverage_ledger": coverage_manifest.get("output_hashes", {}).get("ledger")
        == input_hashes["coverage_ledger"],
        "coverage_selection": coverage_manifest.get("output_hashes", {}).get(
            "selection_bias"
        )
        == input_hashes["coverage_selection"],
        "coverage_stage": coverage_manifest.get("output_hashes", {}).get("stage")
        == input_hashes["coverage_stage"],
        "termination": coverage_manifest.get("output_hashes", {}).get(
            "termination_seal"
        )
        == input_hashes["v2_termination_seal"],
        "reverse_expectations": pre_outcome_stage.get("artifact_hashes", {}).get(
            "expectations_pre_outcome"
        )
        == input_hashes["reverse_expectations"],
        "pre_outcome_seal": pre_outcome_stage.get("pre_outcome_input_seal_sha256")
        == input_hashes["pre_outcome_seal"],
    }
    failures = sorted(key for key, value in sealed_hash_checks.items() if not value)
    if failures:
        raise ValueError(f"sealed diagnostic input hash mismatch: {failures}")

    feature_records = _read_records(paths["features"])
    outcome_records = _read_records(paths["outcomes"])
    label_records = _read_records(paths["labels"])
    ledger_records = _read_records(paths["coverage_ledger"])
    expectation_records = _read_records(paths["reverse_expectations"])
    feature_by_id = _index(feature_records, source="ERI features")
    outcome_by_id = _index(outcome_records, source="label-safe outcomes")
    label_by_id = _index(label_records, source="ERI labels")
    ledger_by_id = _index(ledger_records, source="eligibility ledger")
    expectation_by_id = _index(expectation_records, source="reverse expectations")
    final_ids = set(label_by_id)
    if not (
        final_ids == set(outcome_by_id)
        and final_ids <= set(feature_by_id)
        and final_ids <= set(ledger_by_id)
        and final_ids <= set(expectation_by_id)
        and len(final_ids) == 1640
        and all(ledger_by_id[key]["final_common"] is True for key in final_ids)
    ):
        raise ValueError("diagnostic inputs do not share the sealed final-common IDs")
    seal = EvidenceIndexFeatureDatasetSealV2.model_validate_json(
        paths["feature_seal"].read_text(encoding="utf-8")
    )
    dcf = EconomicDcfEngine()
    rows: list[dict[str, Any]] = []
    maximum_additive_error = D(0)
    maximum_signal_fit_gap = D(0)
    for observation_id in sorted(final_ids):
        feature = EvidenceIndexFutureEriFeatureRowV2.model_validate(
            feature_by_id[observation_id]
        )
        if seal.feature_row_sha256[observation_id] != model_sha256(feature):
            raise ValueError(f"feature row changed after sealing: {observation_id}")
        outcome = FutureEriOutcomeInputV1.model_validate(outcome_by_id[observation_id])
        label = FutureEriLabelV1.model_validate(label_by_id[observation_id])
        components = decompose_eri_bridge_v2(
            feature=feature,
            outcome=outcome,
            label=label,
            engine=dcf,
        )
        additive_error = abs(
            components["realization_component"]
            + components["discount_rate_component"]
            + components["expectation_revision_component"]
            - components["total_log_price_bridge"]
        )
        maximum_additive_error = max(maximum_additive_error, additive_error)
        maximum_signal_fit_gap = max(
            maximum_signal_fit_gap, abs(components["signal_reverse_fit_log_gap"])
        )
        dimension = ledger_by_id[observation_id]
        reverse = expectation_by_id[observation_id]["reverse_dcf_provenance"]
        solution = reverse["solution"]
        driver = str(reverse["selected_driver"])
        frozen_driver_value = {
            "GROWTH": feature.frozen_expectation_assumptions.revenue_growth,
            "MARGIN": feature.frozen_expectation_assumptions.target_nopat_margin,
            "ROIIC": feature.frozen_expectation_assumptions.roiic,
            "CAP": D(feature.frozen_expectation_assumptions.competitive_advantage_period_years),
        }[driver]
        if driver != str(dimension["reverse_dcf_driver"]):
            raise ValueError(f"reverse-driver provenance mismatch: {observation_id}")
        rows.append(
            {
                "schema_version": "moatrader-eri-nonlinearity-decomposition-row-v2/1",
                "observation_id": observation_id,
                "issuer_id": feature.issuer_id,
                "signal_timestamp": feature.signal_timestamp.isoformat(),
                "signal_month": feature.signal_timestamp.strftime("%Y-%m"),
                "full_evidence_index": str(feature.full_evidence_index),
                "full_evidence_band": str(dimension["full_evidence_band"]),
                "full_nobs": feature.full_nobs,
                "nobs_bucket": "NOBS_2" if feature.full_nobs == 2 else "NOBS_3_PLUS",
                "evidence_source_mode": str(dimension["evidence_source_mode"]),
                "valuation_route": str(dimension["valuation_route"]),
                "reverse_dcf_driver": dimension.get("reverse_dcf_driver"),
                "reverse_solution_reported_modeled_price": str(
                    solution["modeled_price"]
                ),
                "reverse_solution_reported_relative_price_error": str(
                    solution["relative_price_error"]
                ),
                "reverse_solution_implied": str(solution["implied"]),
                "frozen_driver_value": str(frozen_driver_value),
                "sector": str(dimension["sector"]),
                "sector_basis": str(dimension["sector_basis"]),
                "signal_size_bucket": str(dimension["signal_size_bucket"]),
                **{key: str(value) for key, value in components.items()},
                "decomposition_additive_error": str(additive_error),
                "future_eri_used_as_signal": False,
                "future_eri_used_as_ranking": False,
                "v2_threshold_changed": False,
            }
        )

    band = _group_summaries(rows, dimensions=("full_evidence_band",))
    by_nobs = _group_summaries(
        rows, dimensions=("nobs_bucket", "full_evidence_band")
    )
    by_source = _group_summaries(
        rows, dimensions=("evidence_source_mode", "full_evidence_band")
    )
    by_route = _group_summaries(
        rows, dimensions=("valuation_route", "full_evidence_band")
    )
    by_driver = _group_summaries(
        rows, dimensions=("reverse_dcf_driver", "full_evidence_band")
    )
    by_sector = _group_summaries(rows, dimensions=("sector", "full_evidence_band"))
    by_month = _group_summaries(
        rows, dimensions=("signal_month", "full_evidence_band")
    )
    coverage_selection = _read_json(paths["coverage_selection"])
    interpretation = _strong_bull_interpretation(
        band_summaries=band,
        coverage_selection=coverage_selection,
    )
    measurement_quality = _measurement_quality(rows)
    robustness = _robustness_diagnostics(
        rows=rows,
        by_nobs=by_nobs,
        by_source=by_source,
        by_driver=by_driver,
        by_sector=by_sector,
        by_month=by_month,
    )
    summary = {
        "schema_version": "moatrader-eri-strong-bull-nonlinearity-diagnostic-v2/1",
        "status": "V2_POST_FAILURE_DIAGNOSTIC_COMPLETE_NO_RETUNING",
        "audit_as_of": audit_as_of,
        "observation_count": len(rows),
        "issuer_count": len({row["issuer_id"] for row in rows}),
        "decomposition_identity": (
            "log(P_t63 / V_t_fitted) = realization_at_signal_wacc + "
            "target_wacc_effect + sealed_future_eri_expectation_revision"
        ),
        "market_price_bridge_identity": (
            "log(P_t63 / P_t_market) = signal_reverse_fit_log_gap + "
            "realization_at_signal_wacc + target_wacc_effect + "
            "sealed_future_eri_expectation_revision"
        ),
        "future_eri_definition": (
            "The sealed V2 Future ERI is the expectation-revision residual, not the sum. "
            "The additive sum is a diagnostic bridge across the already-opened ERI price levels."
        ),
        "realization_definition": (
            "Re-anchor base revenue, NOPAT margin, invested capital, net debt, shares, and "
            "elapsed CAP while holding WACC at its signal value."
        ),
        "discount_rate_definition": (
            "Change only WACC from its signal policy value to the sealed target policy value."
        ),
        "expectation_revision_definition": (
            "Sealed log(actual target price / frozen-expectation counterfactual value)."
        ),
        "maximum_additive_identity_error": str(maximum_additive_error),
        "maximum_absolute_signal_reverse_fit_log_gap": str(maximum_signal_fit_gap),
        "measurement_quality": measurement_quality,
        "robustness_diagnostics": robustness,
        "strong_bull_diagnosis": interpretation,
        "external_return_dataset_opened": False,
        "existing_eri_price_levels_used_for_diagnostic_bridge": True,
        "v2_thresholds_changed": False,
        "v2_index_formula_changed": False,
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "causal_claim_allowed": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    output_files: dict[str, Path] = {
        "decomposition_rows": output / "eri-decomposition-observations.jsonl",
        "band_summary": output / "band-component-summary.jsonl",
        "nobs_summary": output / "diagnostics-by-nobs.jsonl",
        "source_summary": output / "diagnostics-by-evidence-source-mode.jsonl",
        "route_summary": output / "diagnostics-by-route.jsonl",
        "driver_summary": output / "diagnostics-by-reverse-driver.jsonl",
        "sector_summary": output / "diagnostics-by-sector.jsonl",
        "month_summary": output / "diagnostics-by-signal-month.jsonl",
        "summary": output / "strong-bull-diagnostic-summary.json",
    }
    _write_jsonl(output_files["decomposition_rows"], rows)
    _write_jsonl(output_files["band_summary"], band)
    _write_jsonl(output_files["nobs_summary"], by_nobs)
    _write_jsonl(output_files["source_summary"], by_source)
    _write_jsonl(output_files["route_summary"], by_route)
    _write_jsonl(output_files["driver_summary"], by_driver)
    _write_jsonl(output_files["sector_summary"], by_sector)
    _write_jsonl(output_files["month_summary"], by_month)
    _write_json(output_files["summary"], summary)

    after_hashes = {role: sha256_file(path) for role, path in paths.items()}
    if input_hashes != after_hashes:
        raise RuntimeError("a sealed V2 source changed during Strong Bull diagnosis")
    status = {
        "schema_version": "moatrader-eri-nonlinearity-diagnostic-stage-v2/1",
        "status": "V2_STRONG_BULL_NONLINEARITY_DIAGNOSED_NO_RETUNING",
        "v2_promotion_status": "FAILED_AND_SEALED",
        "observation_count": len(rows),
        "issuer_count": len({row["issuer_id"] for row in rows}),
        "five_band_component_quantiles_written": True,
        "nobs_semantic_route_sector_month_diagnostics_written": True,
        "external_return_dataset_opened": False,
        "v2_thresholds_changed": False,
        "v2_index_formula_changed": False,
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "source_files_modified": False,
    }
    stage_path = output / "stage-status.json"
    _write_json(stage_path, status)
    output_files["stage"] = stage_path
    _write_json(
        output / "build-manifest.json",
        {
            **status,
            "audit_as_of": audit_as_of,
            "git_commit": commit,
            "worktree_dirty": False,
            "input_paths": {role: str(path.resolve()) for role, path in paths.items()},
            "input_hashes": input_hashes,
            "output_hashes": {
                role: sha256_file(path) for role, path in output_files.items()
            },
            "source_integrity_verification_status": "PASS_NO_SOURCE_MUTATION",
        },
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose the sealed V2 Strong Bull median break without retuning."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--eri-build", type=Path, required=True)
    parser.add_argument("--outcome-build", type=Path, required=True)
    parser.add_argument("--coverage-audit", type=Path, required=True)
    parser.add_argument("--pre-outcome-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-as-of", default=date.today().isoformat())
    args = parser.parse_args()
    result = diagnose_strong_bull_nonlinearity_v2(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
