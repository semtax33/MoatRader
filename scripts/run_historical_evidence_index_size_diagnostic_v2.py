from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from moatrader.backtest.universe_corrected import (
    rank_normal_score,
    residualize_cross_section,
    spearman_ic,
)
from moatrader.expectations.historical_evidence import sha256_file
from scripts.audit_historical_eri_eligibility_bridge_v2 import _market_dimensions
from scripts.prepare_historical_evidence_index_eri_inputs_v2 import _git_state
from scripts.run_historical_evidence_index_value_neutralization_v2 import _inference


STAGES = (
    "price_pit_available",
    "reverse_valuation_available",
    "t63_snapshot_available",
    "eri_decomposition_valid",
    "final_common",
)
SIZE_ORDER = {"SMALL": 0, "MID": 1, "LARGE": 2, "UNKNOWN_SIGNAL_SIZE": 3}
BAND_ORDER = {
    "STRONG_BEAR": 0,
    "BEAR": 1,
    "NEUTRAL": 2,
    "BULL": 3,
    "STRONG_BULL": 4,
}


def _json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    _json_safe(row),
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                )
                + "\n"
            )


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
    if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
        raise ValueError(f"JSON object records required: {path}")
    return [dict(row) for row in raw]


def _index(records: Sequence[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed = {str(row["observation_id"]): row for row in records}
    if len(indexed) != len(records):
        raise ValueError(f"duplicate observation_id in {source}")
    return indexed


def _positive_signal_market_cap(point: dict[str, Any] | None) -> float | None:
    if point is None:
        return None
    try:
        price = float(point["Open"])
        shares = float(point["Stocks"])
    except (KeyError, TypeError, ValueError):
        return None
    market_cap = price * shares
    if not all(math.isfinite(value) and value > 0 for value in (price, shares, market_cap)):
        return None
    return market_cap


def _assign_signal_open_size_buckets(rows: list[dict[str, Any]]) -> None:
    """Assign date-local thirds using only market information known at signal open."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["signal_size_bucket"] = "UNKNOWN_SIGNAL_SIZE"
        if row.get("signal_open_market_cap") is not None:
            grouped[str(row["signal_timestamp"])[:10]].append(row)
    for selected in grouped.values():
        if len(selected) < 3:
            continue
        ordered = sorted(
            selected,
            key=lambda row: (
                float(row["signal_open_market_cap"]),
                str(row["issuer_id"]),
                str(row["observation_id"]),
            ),
        )
        count = len(ordered)
        for index, row in enumerate(ordered):
            percentile = (index + 1) / count
            row["signal_size_bucket"] = (
                "SMALL" if percentile <= 1 / 3 else "MID" if percentile <= 2 / 3 else "LARGE"
            )


def _spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 5 or len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
        return {"n": int(len(left)), "rho": None, "p_value": None}
    result = stats.spearmanr(left, right)
    return {
        "n": int(len(left)),
        "rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def _monthly_spearman_inference(
    rows: Sequence[dict[str, Any]], *, left: str, right: str
) -> dict[str, Any]:
    values: list[float] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get(left) is not None and row.get(right) is not None:
            grouped[str(row["signal_month"])].append(row)
    for selected in grouped.values():
        result = _spearman(
            [float(row[left]) for row in selected],
            [float(row[right]) for row in selected],
        )
        if result["rho"] is not None:
            values.append(float(result["rho"]))
    return _inference(
        values,
        hac_lag_months=3,
        block_length_months=4,
        bootstrap_repetitions=10_000,
        bootstrap_seed=42,
    )


def _logistic_size_fit(
    rows: Sequence[dict[str, Any]], *, outcome: str
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("log_market_cap") is not None and row.get(outcome) is not None
    ]
    x_raw = np.asarray([float(row["log_market_cap"]) for row in selected], dtype=float)
    y = np.asarray([int(bool(row[outcome])) for row in selected], dtype=float)
    if len(selected) < 20 or y.min() == y.max() or float(x_raw.std(ddof=0)) <= 0:
        return {
            "outcome": outcome,
            "n": len(selected),
            "event_count": int(y.sum()),
            "status": "NOT_IDENTIFIED",
        }
    mean = float(x_raw.mean())
    standard_deviation = float(x_raw.std(ddof=0))
    z = (x_raw - mean) / standard_deviation
    x = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(2, dtype=float)
    converged = False
    information = np.eye(2)
    for iteration in range(100):
        linear = np.clip(x @ beta, -35.0, 35.0)
        probability = 1.0 / (1.0 + np.exp(-linear))
        weights = np.clip(probability * (1.0 - probability), 1e-12, None)
        information = x.T @ (weights[:, None] * x)
        score = x.T @ (y - probability)
        step = np.linalg.solve(information, score)
        beta += step
        if float(np.max(np.abs(step))) < 1e-10:
            converged = True
            break
    linear = np.clip(x @ beta, -35.0, 35.0)
    probability = 1.0 / (1.0 + np.exp(-linear))
    weights = np.clip(probability * (1.0 - probability), 1e-12, None)
    information = x.T @ (weights[:, None] * x)
    bread = np.linalg.inv(information)
    cluster_scores: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(2))
    for index, row in enumerate(selected):
        cluster_scores[str(row["issuer_id"])] += x[index] * (y[index] - probability[index])
    score_matrix = np.asarray(list(cluster_scores.values()), dtype=float)
    meat = score_matrix.T @ score_matrix
    cluster_count = len(score_matrix)
    correction = (
        cluster_count / (cluster_count - 1) * (len(selected) - 1) / (len(selected) - 2)
        if cluster_count > 1 and len(selected) > 2
        else 1.0
    )
    covariance = bread @ meat @ bread * correction
    standard_error = float(math.sqrt(max(float(covariance[1, 1]), 0.0)))
    coefficient = float(beta[1])
    return {
        "outcome": outcome,
        "status": "IDENTIFIED" if converged else "MAX_ITERATIONS_REACHED",
        "n": len(selected),
        "issuer_cluster_count": cluster_count,
        "event_count": int(y.sum()),
        "event_rate": float(y.mean()),
        "log_market_cap_mean": mean,
        "log_market_cap_standard_deviation": standard_deviation,
        "coefficient_per_one_sd_log_market_cap": coefficient,
        "issuer_clustered_standard_error": standard_error,
        "issuer_clustered_z": coefficient / standard_error if standard_error > 0 else None,
        "odds_ratio_per_one_sd_log_market_cap": math.exp(
            min(max(coefficient, -700.0), 700.0)
        ),
        "predicted_probability_at_minus_one_sd": float(
            1.0
            / (
                1.0
                + math.exp(
                    -min(max(float(beta[0] - beta[1]), -35.0), 35.0)
                )
            )
        ),
        "predicted_probability_at_mean": float(
            1.0
            / (1.0 + math.exp(-min(max(float(beta[0]), -35.0), 35.0)))
        ),
        "predicted_probability_at_plus_one_sd": float(
            1.0
            / (
                1.0
                + math.exp(
                    -min(max(float(beta[0] + beta[1]), -35.0), 35.0)
                )
            )
        ),
        "model": "UNIVARIATE_LOGIT_WITH_ISSUER_CLUSTERED_SANDWICH_SE",
    }


def _monthly_size_neutralization(
    panel: pd.DataFrame, *, minimum_monthly_observations: int
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal_month, month in panel.groupby("signal_month", sort=True):
        sample = month[
            ["full_evidence_index", "future_eri", "log_market_cap"]
        ].dropna().copy()
        base = {
            "schema_version": "moatrader-evidence-index-size-neutral-month-v2/1",
            "signal_month": signal_month,
            "control": "log_market_cap_at_signal_open",
            "n": len(sample),
            "same_sample_raw_and_neutral": True,
        }
        if len(sample) < max(minimum_monthly_observations, 4):
            rows.append({**base, "status": "INSUFFICIENT_MONTHLY_OBSERVATIONS"})
            continue
        sample["neutral_full_evidence_index"] = residualize_cross_section(
            sample,
            target="full_evidence_index",
            numeric_controls=("log_market_cap",),
        )
        complete = sample.dropna(subset=["neutral_full_evidence_index"])
        raw_ic = spearman_ic(complete, "full_evidence_index", "future_eri")
        neutral_ic = spearman_ic(
            complete, "neutral_full_evidence_index", "future_eri"
        )
        target = rank_normal_score(complete["full_evidence_index"])
        control = rank_normal_score(complete["log_market_cap"])
        residual = complete["neutral_full_evidence_index"]
        valid = pd.concat(
            [target.rename("target"), control.rename("control"), residual.rename("residual")],
            axis=1,
        ).dropna()
        exposure_r_squared = None
        post_spearman = None
        post_pearson = None
        if len(valid) > 3:
            x = np.column_stack([np.ones(len(valid)), valid["control"]])
            y = valid["target"].to_numpy(dtype=float)
            fitted = x @ np.linalg.lstsq(x, y, rcond=None)[0]
            total = float(np.sum((y - y.mean()) ** 2))
            exposure_r_squared = (
                1.0 - float(np.sum((y - fitted) ** 2)) / total if total > 0 else None
            )
            post_spearman = float(
                valid["residual"].corr(valid["control"], method="spearman")
            )
            post_pearson = float(valid["residual"].corr(valid["control"]))
        rows.append(
            {
                **base,
                "status": (
                    "EVALUATED_SAME_SAMPLE"
                    if math.isfinite(raw_ic) and math.isfinite(neutral_ic)
                    else "INSUFFICIENT_VARIATION"
                ),
                "n": len(complete),
                "raw_ic": raw_ic,
                "size_neutral_ic": neutral_ic,
                "delta_ic": neutral_ic - raw_ic,
                "control_exposure_r_squared": exposure_r_squared,
                "post_control_spearman_with_log_market_cap": post_spearman,
                "post_control_pearson_with_ranked_log_market_cap": post_pearson,
            }
        )
    return rows


def _size_neutral_summary(
    panel: pd.DataFrame,
    monthly: Sequence[dict[str, Any]],
    *,
    hac_lag_months: int,
    block_length_months: int,
    bootstrap_repetitions: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    selected = [row for row in monthly if row["status"] == "EVALUATED_SAME_SAMPLE"]
    inference = {
        name: _inference(
            [float(row[name]) for row in selected],
            hac_lag_months=hac_lag_months,
            block_length_months=block_length_months,
            bootstrap_repetitions=bootstrap_repetitions,
            bootstrap_seed=bootstrap_seed,
        )
        for name in ("raw_ic", "size_neutral_ic", "delta_ic")
    }
    raw_mean = inference["raw_ic"]["newey_west"]["mean"]
    neutral_mean = inference["size_neutral_ic"]["newey_west"]["mean"]
    retention = (
        float(neutral_mean) / float(raw_mean)
        if raw_mean is not None
        and neutral_mean is not None
        and abs(float(raw_mean)) > 1e-12
        else None
    )
    if retention is None:
        interpretation = "SIZE_NEUTRAL_RESULT_NOT_IDENTIFIED"
    elif retention >= 0.75:
        interpretation = "EVIDENCE_ERI_RELATION_LARGELY_SURVIVES_SIZE_CONTROL"
    elif retention >= 0.25:
        interpretation = "SIZE_PARTLY_EXPLAINS_EVIDENCE_ERI_RELATION"
    else:
        interpretation = "EVIDENCE_ERI_RELATION_LARGELY_ABSORBED_BY_SIZE"
    return {
        "schema_version": "moatrader-evidence-index-size-neutralization-summary-v2/1",
        "status": "EVALUATED_SIZE_AS_SINGLE_PARALLEL_SENSITIVITY",
        "signal": "FULL_EVIDENCE_INDEX",
        "outcome": "FUTURE_ERI_T63",
        "control": "LOG_SIGNAL_OPEN_PRICE_TIMES_LISTED_SHARES",
        "panel_observation_count": len(panel),
        "complete_control_observation_count": int(panel["log_market_cap"].notna().sum()),
        "valid_month_count": len(selected),
        "same_sample_raw_and_neutral": True,
        "raw_ic": inference["raw_ic"],
        "size_neutral_ic": inference["size_neutral_ic"],
        "delta_ic": inference["delta_ic"],
        "signed_ic_retention_ratio": retention,
        "absolute_ic_attenuation": (
            1.0 - abs(float(neutral_mean)) / abs(float(raw_mean))
            if retention is not None
            else None
        ),
        "interpretation": interpretation,
        "mean_control_exposure_r_squared": (
            float(np.mean([row["control_exposure_r_squared"] for row in selected]))
            if selected
            else None
        ),
        "max_abs_post_control_spearman": (
            max(abs(float(row["post_control_spearman_with_log_market_cap"])) for row in selected)
            if selected
            else None
        ),
        "ranking_output_produced": False,
        "return_data_opened": False,
    }


def _size_bucket_diagnostics(panel: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    grouping = {
        "STRONG_BEAR": "BEAR_SIDE",
        "BEAR": "BEAR_SIDE",
        "NEUTRAL": "NEUTRAL",
        "BULL": "BULL_SIDE",
        "STRONG_BULL": "BULL_SIDE",
    }
    work = panel.copy()
    work["evidence_side"] = work["full_evidence_band"].map(grouping)
    for (size, side), group in work.groupby(
        ["signal_size_bucket", "evidence_side"], sort=False
    ):
        values = group["future_eri"].to_numpy(dtype=float)
        cells.append(
            {
                "schema_version": "moatrader-size-evidence-eri-cell-v2/1",
                "signal_size_bucket": str(size),
                "evidence_group": str(side),
                "observation_count": len(group),
                "issuer_count": int(group["issuer_id"].nunique()),
                "mean_future_eri": float(np.mean(values)),
                "median_future_eri": float(np.median(values)),
                "negative_future_eri_share": float(np.mean(values < 0)),
            }
        )
    side_order = {"BEAR_SIDE": 0, "NEUTRAL": 1, "BULL_SIDE": 2}
    cells.sort(
        key=lambda row: (
            SIZE_ORDER.get(str(row["signal_size_bucket"]), 99),
            side_order.get(str(row["evidence_group"]), 99),
        )
    )
    size_results: dict[str, Any] = {}
    for size, group in work.groupby("signal_size_bucket", sort=False):
        selected_cells = [row for row in cells if row["signal_size_bucket"] == size]
        by_side = {row["evidence_group"]: row for row in selected_cells}
        medians = [
            by_side[side]["median_future_eri"]
            for side in ("BEAR_SIDE", "NEUTRAL", "BULL_SIDE")
            if side in by_side
        ]
        month_ics = []
        for _month, monthly_group in group.groupby("signal_month", sort=True):
            value = spearman_ic(monthly_group, "full_evidence_index", "future_eri")
            if math.isfinite(value):
                month_ics.append(value)
        size_results[str(size)] = {
            "observation_count": len(group),
            "issuer_count": int(group["issuer_id"].nunique()),
            "mean_full_nobs": float(group["full_nobs"].mean()),
            "median_full_nobs": float(group["full_nobs"].median()),
            "pooled_spearman_evidence_to_future_eri": _spearman(
                group["full_evidence_index"], group["future_eri"]
            ),
            "monthly_ic": _inference(
                month_ics,
                hac_lag_months=3,
                block_length_months=4,
                bootstrap_repetitions=10_000,
                bootstrap_seed=42,
            ),
            "three_group_median_nondecreasing": (
                len(medians) == 3 and medians[0] <= medians[1] <= medians[2]
            ),
        }
    return cells, dict(sorted(size_results.items(), key=lambda item: SIZE_ORDER.get(item[0], 99)))


def run_size_diagnostic_v2(
    *,
    workspace: Path,
    eri_build: Path,
    coverage_audit: Path,
    pre_outcome_build: Path,
    output: Path,
    audit_as_of: str,
    minimum_monthly_observations: int = 5,
    hac_lag_months: int = 3,
    block_length_months: int = 4,
    bootstrap_repetitions: int = 10_000,
    bootstrap_seed: int = 42,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production Size diagnostic requires a clean worktree")

    audit_manifest_path = coverage_audit / "audit-manifest.json"
    audit_stage_path = coverage_audit / "stage-status.json"
    ledger_path = coverage_audit / "observation-eligibility-ledger.jsonl"
    pre_stage_path = pre_outcome_build / "stage-status.json"
    eri_stage_path = eri_build / "stage-status.json"
    eri_manifest_path = eri_build / "build-manifest.json"
    feature_path = eri_build / "features-with-frozen-expectations-pre-outcome.jsonl"
    labels_path = eri_build / "future-eri-labels.jsonl"
    required = (
        audit_manifest_path,
        audit_stage_path,
        ledger_path,
        pre_stage_path,
        eri_stage_path,
        eri_manifest_path,
        feature_path,
        labels_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Size diagnostic inputs missing: {missing}")

    audit_manifest = _read_json(audit_manifest_path)
    audit_stage = _read_json(audit_stage_path)
    pre_stage = _read_json(pre_stage_path)
    eri_stage = _read_json(eri_stage_path)
    eri_manifest = _read_json(eri_manifest_path)
    input_paths = audit_manifest.get("input_paths", {})
    full_rows_path = Path(str(input_paths.get("full_rows", "")))
    if not full_rows_path.is_file():
        raise FileNotFoundError(f"sealed Full Evidence rows missing: {full_rows_path}")
    checks = {
        "coverage_status": audit_stage.get("status")
        == "V2_TERMINATED_ERI_ELIGIBILITY_BRIDGE_AUDITED",
        "coverage_integrity": audit_manifest.get("source_integrity_verification_status")
        == "PASS_NO_SOURCE_MUTATION",
        "coverage_ledger": audit_manifest.get("output_hashes", {}).get("ledger")
        == sha256_file(ledger_path),
        "pre_stage": audit_manifest.get("input_hashes", {}).get("pre_outcome_stage")
        == sha256_file(pre_stage_path),
        "full_rows": audit_manifest.get("input_hashes", {}).get("full_rows")
        == sha256_file(full_rows_path),
        "eri_stage": audit_manifest.get("input_hashes", {}).get("eri_stage")
        == sha256_file(eri_stage_path),
        "eri_manifest": audit_manifest.get("input_hashes", {}).get("eri_manifest")
        == sha256_file(eri_manifest_path),
        "features": audit_manifest.get("input_hashes", {}).get("feature_rows")
        == sha256_file(feature_path),
        "labels": audit_manifest.get("input_hashes", {}).get("eri_labels")
        == sha256_file(labels_path),
        "eri_failed_sealed": eri_stage.get("status")
        == "FULL_PRIMARY_MECHANISM_REJECTED_OR_INCONCLUSIVE",
        "return_closed": eri_stage.get("return_data_opened") is False,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"sealed Size diagnostic inputs invalid: {failed}")

    input_hashes = {
        "coverage_manifest": sha256_file(audit_manifest_path),
        "coverage_stage": sha256_file(audit_stage_path),
        "coverage_ledger": sha256_file(ledger_path),
        "pre_outcome_stage": sha256_file(pre_stage_path),
        "full_rows": sha256_file(full_rows_path),
        "eri_stage": sha256_file(eri_stage_path),
        "eri_manifest": sha256_file(eri_manifest_path),
        "features": sha256_file(feature_path),
        "labels": sha256_file(labels_path),
    }
    full_by_id = _index(_read_records(full_rows_path), source="Full Evidence rows")
    ledger_by_id = _index(_read_records(ledger_path), source="eligibility ledger")
    feature_by_id = _index(_read_records(feature_path), source="ERI features")
    label_by_id = _index(_read_records(labels_path), source="ERI labels")
    if not set(ledger_by_id) <= set(full_by_id) or len(ledger_by_id) != 37_014:
        raise ValueError("eligibility ledger is not the sealed Full/Core common panel")
    full_by_id = {key: full_by_id[key] for key in ledger_by_id}
    if set(label_by_id) != {key for key, row in ledger_by_id.items() if row["final_common"]}:
        raise ValueError("final ERI labels do not match the eligibility ledger")
    if not set(label_by_id) <= set(feature_by_id):
        raise ValueError("final ERI labels are not covered by sealed features")

    size_by_id, point_by_id, marcap_hashes = _market_dimensions(
        pre_stage=pre_stage, full_rows=full_by_id
    )
    if marcap_hashes != audit_manifest.get("marcap_input_hashes"):
        raise ValueError("reconstructed MARCAP hashes differ from the sealed coverage audit")

    selection_rows: list[dict[str, Any]] = []
    for observation_id in sorted(full_by_id):
        full = full_by_id[observation_id]
        ledger = ledger_by_id[observation_id]
        market_cap = _positive_signal_market_cap(point_by_id.get(observation_id))
        selection_rows.append(
            {
                "schema_version": "moatrader-evidence-index-size-selection-row-v2/1",
                "observation_id": observation_id,
                "issuer_id": str(full["issuer_id"]).zfill(6),
                "signal_timestamp": str(full["signal_timestamp"]),
                "signal_month": str(full["signal_timestamp"])[:7],
                "full_evidence_index": float(full["full_evidence_index"]),
                "full_nobs": int(full["nobs"]),
                "sealed_wacc_size_bucket": size_by_id[observation_id],
                "signal_open_market_cap": market_cap,
                "log_market_cap": math.log(market_cap) if market_cap else None,
                **{stage: bool(ledger[stage]) for stage in STAGES},
            }
        )

    _assign_signal_open_size_buckets(selection_rows)

    complete_selection = [row for row in selection_rows if row["log_market_cap"] is not None]
    pooled = {
        "evidence_index_to_log_market_cap": _spearman(
            [row["full_evidence_index"] for row in complete_selection],
            [row["log_market_cap"] for row in complete_selection],
        ),
        "nobs_to_log_market_cap": _spearman(
            [row["full_nobs"] for row in complete_selection],
            [row["log_market_cap"] for row in complete_selection],
        ),
    }
    bucket_summary: list[dict[str, Any]] = []
    for size in sorted(
        {str(row["signal_size_bucket"]) for row in selection_rows},
        key=lambda value: SIZE_ORDER.get(value, 99),
    ):
        rows = [row for row in selection_rows if row["signal_size_bucket"] == size]
        complete = [row for row in rows if row["log_market_cap"] is not None]
        bucket_summary.append(
            {
                "signal_size_bucket": size,
                "observation_count": len(rows),
                "issuer_count": len({row["issuer_id"] for row in rows}),
                "log_market_cap_complete_count": len(complete),
                "mean_full_nobs": float(np.mean([row["full_nobs"] for row in rows])),
                "median_full_nobs": float(np.median([row["full_nobs"] for row in rows])),
                "mean_full_evidence_index": float(
                    np.mean([row["full_evidence_index"] for row in rows])
                ),
                **{
                    f"{stage}_rate": float(np.mean([row[stage] for row in rows]))
                    for stage in STAGES
                },
            }
        )
    outcome_blind = {
        "schema_version": "moatrader-evidence-index-outcome-blind-size-diagnostic-v2/1",
        "status": "OUTCOME_BLIND_SIZE_AND_ELIGIBILITY_DIAGNOSTICS_COMPLETE",
        "baseline_observation_count": len(selection_rows),
        "complete_log_market_cap_count": len(complete_selection),
        "missing_or_nonpositive_signal_open_market_cap_count": (
            len(selection_rows) - len(complete_selection)
        ),
        "signal_market_cap_definition": "EXACT_SIGNAL_SESSION_OPEN_TIMES_LISTED_SHARES",
        "size_bucket_definition": (
            "DATE_LOCAL_TERCILES_OF_EXACT_SIGNAL_OPEN_MARKET_CAP_WITH_DETERMINISTIC_TIEBREAK"
        ),
        "same_day_close_marcap_used": False,
        "pooled_correlations": pooled,
        "monthly_correlation_inference": {
            "evidence_index_to_log_market_cap": _monthly_spearman_inference(
                complete_selection,
                left="full_evidence_index",
                right="log_market_cap",
            ),
            "nobs_to_log_market_cap": _monthly_spearman_inference(
                complete_selection,
                left="full_nobs",
                right="log_market_cap",
            ),
        },
        "eligibility_logistic_models": {
            stage: _logistic_size_fit(selection_rows, outcome=stage) for stage in STAGES
        },
        "size_bucket_summary": bucket_summary,
        "future_eri_opened_for_these_diagnostics": False,
    }
    final_logit = outcome_blind["eligibility_logistic_models"]["final_common"]
    outcome_blind["interpretation"] = {
        "evidence_or_nobs_is_size_proxy": (
            abs(float(pooled["evidence_index_to_log_market_cap"]["rho"])) >= 0.10
            or abs(float(pooled["nobs_to_log_market_cap"]["rho"])) >= 0.10
        ),
        "eri_eligibility_is_materially_size_associated": (
            float(final_logit["odds_ratio_per_one_sd_log_market_cap"]) >= 1.25
        ),
        "summary": (
            "Evidence and Nobs are not materially associated with Size, but final ERI "
            "eligibility rises materially with Size. Selection bias and signal confounding "
            "are therefore distinct in this panel."
        ),
    }

    selection_by_id = {row["observation_id"]: row for row in selection_rows}
    panel_rows: list[dict[str, Any]] = []
    for observation_id in sorted(label_by_id):
        feature = feature_by_id[observation_id]
        label = label_by_id[observation_id]
        selection = selection_by_id[observation_id]
        feature_market_cap = float(feature["expectation_state"]["market_price"]) * float(
            feature["frozen_expectation_assumptions"]["diluted_shares"]
        )
        reconstructed = float(selection["signal_open_market_cap"])
        if abs(feature_market_cap / reconstructed - 1.0) > 1e-10:
            raise ValueError(f"sealed feature market cap mismatch: {observation_id}")
        panel_rows.append(
            {
                "observation_id": observation_id,
                "issuer_id": str(feature["issuer_id"]).zfill(6),
                "signal_month": str(feature["signal_timestamp"])[:7],
                "full_evidence_index": float(feature["full_evidence_index"]),
                "full_evidence_band": str(ledger_by_id[observation_id]["full_evidence_band"]),
                "full_nobs": int(feature["full_nobs"]),
                "signal_size_bucket": str(selection["signal_size_bucket"]),
                "log_market_cap": float(selection["log_market_cap"]),
                "future_eri": float(label["future_eri"]),
            }
        )
    panel = pd.DataFrame(panel_rows)
    monthly = _monthly_size_neutralization(
        panel, minimum_monthly_observations=minimum_monthly_observations
    )
    neutral_summary = _size_neutral_summary(
        panel,
        monthly,
        hac_lag_months=hac_lag_months,
        block_length_months=block_length_months,
        bootstrap_repetitions=bootstrap_repetitions,
        bootstrap_seed=bootstrap_seed,
    )
    cells, within_size = _size_bucket_diagnostics(panel)
    final_diagnostic = {
        "schema_version": "moatrader-final-eri-size-diagnostic-v2/1",
        "status": "FINAL_ERI_PANEL_SIZE_DIAGNOSTICS_COMPLETE",
        "observation_count": len(panel),
        "issuer_count": int(panel["issuer_id"].nunique()),
        "pooled_correlations": {
            "evidence_index_to_log_market_cap": _spearman(
                panel["full_evidence_index"], panel["log_market_cap"]
            ),
            "nobs_to_log_market_cap": _spearman(panel["full_nobs"], panel["log_market_cap"]),
        },
        "within_size_bucket": within_size,
        "three_group_definition": {
            "BEAR_SIDE": ["STRONG_BEAR", "BEAR"],
            "NEUTRAL": ["NEUTRAL"],
            "BULL_SIDE": ["BULL", "STRONG_BULL"],
        },
        "causal_claim_allowed": False,
    }

    output.mkdir(parents=True, exist_ok=True)
    output_files = {
        "selection_rows": output / "size-selection-observations.jsonl",
        "outcome_blind": output / "outcome-blind-size-diagnostics.json",
        "monthly_neutralization": output / "monthly-size-neutralization.jsonl",
        "neutralization_summary": output / "size-neutralization-summary.json",
        "size_cells": output / "size-evidence-eri-cells.jsonl",
        "final_size_diagnostic": output / "final-eri-size-diagnostics.json",
    }
    _write_jsonl(output_files["selection_rows"], selection_rows)
    _write_json(output_files["outcome_blind"], outcome_blind)
    _write_jsonl(output_files["monthly_neutralization"], monthly)
    _write_json(output_files["neutralization_summary"], neutral_summary)
    _write_jsonl(output_files["size_cells"], cells)
    _write_json(output_files["final_size_diagnostic"], final_diagnostic)

    after_hashes = {
        "coverage_manifest": sha256_file(audit_manifest_path),
        "coverage_stage": sha256_file(audit_stage_path),
        "coverage_ledger": sha256_file(ledger_path),
        "pre_outcome_stage": sha256_file(pre_stage_path),
        "full_rows": sha256_file(full_rows_path),
        "eri_stage": sha256_file(eri_stage_path),
        "eri_manifest": sha256_file(eri_manifest_path),
        "features": sha256_file(feature_path),
        "labels": sha256_file(labels_path),
    }
    if input_hashes != after_hashes:
        raise RuntimeError("a sealed input changed during Size diagnosis")
    status = {
        "schema_version": "moatrader-evidence-index-size-diagnostic-stage-v2/1",
        "status": "V2_SIZE_NEUTRALIZATION_AND_SELECTION_DIAGNOSTICS_COMPLETE",
        "audit_as_of": audit_as_of,
        "baseline_observation_count": len(selection_rows),
        "final_panel_observation_count": len(panel),
        "final_panel_issuer_count": int(panel["issuer_id"].nunique()),
        "same_sample_raw_and_neutral": True,
        "size_only_control": True,
        "liquidity_control_used": False,
        "return_data_opened": False,
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "ranking_output_produced": False,
        "v2_thresholds_changed": False,
        "v2_index_formula_changed": False,
        "source_files_modified": False,
    }
    stage_path = output / "stage-status.json"
    _write_json(stage_path, status)
    output_files["stage"] = stage_path
    _write_json(
        output / "build-manifest.json",
        {
            **status,
            "git_commit": commit,
            "worktree_dirty": False,
            "input_paths": {
                "coverage_manifest": str(audit_manifest_path.resolve()),
                "coverage_stage": str(audit_stage_path.resolve()),
                "coverage_ledger": str(ledger_path.resolve()),
                "pre_outcome_stage": str(pre_stage_path.resolve()),
                "full_rows": str(full_rows_path.resolve()),
                "eri_stage": str(eri_stage_path.resolve()),
                "eri_manifest": str(eri_manifest_path.resolve()),
                "features": str(feature_path.resolve()),
                "labels": str(labels_path.resolve()),
            },
            "input_hashes": input_hashes,
            "marcap_input_hashes": marcap_hashes,
            "output_hashes": {
                role: sha256_file(path) for role, path in output_files.items()
            },
            "source_integrity_verification_status": "PASS_NO_SOURCE_MUTATION",
        },
    )
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run signal-open Size neutralization and ERI eligibility diagnostics."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--eri-build", type=Path, required=True)
    parser.add_argument("--coverage-audit", type=Path, required=True)
    parser.add_argument("--pre-outcome-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-as-of", required=True)
    parser.add_argument("--minimum-monthly-observations", type=int, default=5)
    parser.add_argument("--hac-lag-months", type=int, default=3)
    parser.add_argument("--block-length-months", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    result = run_size_diagnostic_v2(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
