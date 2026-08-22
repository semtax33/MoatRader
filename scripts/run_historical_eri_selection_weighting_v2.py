from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from moatrader.expectations.historical_evidence import sha256_file
from scripts.prepare_historical_evidence_index_eri_inputs_v2 import (
    PITEconomicAnnualSnapshotV2,
    _git_state,
    _timeline,
    _valid_financial,
)
from scripts.run_historical_evidence_index_value_neutralization_v2 import _inference


NUMERIC_FEATURES = (
    "log_market_cap",
    "full_nobs",
    "coverage",
    "applicable_axis_count",
    "unavailable_axis_count",
    "semantic_grounded_axis_count",
    "deterministic_core_grounded_axis_count",
    "pit_snapshot_count",
    "pit_valid_history_count",
    "latest_pit_metric_count",
    "pit_history_year_span",
)
CATEGORICAL_FEATURES = (
    "signal_year",
    "signal_size_bucket",
    "sector",
    "listing_age_proxy",
    "financial_archetype",
    "security_type",
    "evidence_source_mode",
    "annual_source_mode",
    "signal_market_data_available",
)
BAND_ORDER = {
    "STRONG_BEAR": 0,
    "BEAR": 1,
    "NEUTRAL": 2,
    "BULL": 3,
    "STRONG_BULL": 4,
}
WEIGHT_CAP = 20.0
PROPENSITY_FLOOR = 0.005


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


def _read_snapshots(path: Path) -> list[PITEconomicAnnualSnapshotV2]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            PITEconomicAnnualSnapshotV2.model_validate_json(line)
            for line in handle
            if line.strip()
        ]


def _source_mode(snapshot: PITEconomicAnnualSnapshotV2 | None) -> str:
    if snapshot is None:
        return "NO_PIT_ANNUAL"
    paths = [value.casefold() for value in snapshot.verified_source_hashes]
    arcana = any("\\arcana\\" in value or "/arcana/" in value for value in paths)
    moatrader = any("\\moatrader\\" in value or "/moatrader/" in value for value in paths)
    if arcana and moatrader:
        return "ARCANA_AND_MOATRADER"
    if arcana:
        return "ARCANA_ONLY"
    if moatrader:
        return "MOATRADER_ONLY"
    return "OTHER_SOURCE"


def _first_seen_dates(
    pre_stage: dict[str, Any], *, tickers: set[str]
) -> tuple[dict[str, pd.Timestamp], dict[str, str], pd.Timestamp]:
    first_seen: dict[str, pd.Timestamp] = {}
    hashes: dict[str, str] = {}
    provider_start: pd.Timestamp | None = None
    for year, source in sorted(pre_stage["marcap_sources"].items()):
        path = Path(str(source["path"]))
        actual = sha256_file(path)
        if actual != str(source["sha256"]):
            raise ValueError(f"sealed MARCAP source changed for {year}")
        hashes[f"marcap_{year}"] = actual
        frame = pd.read_parquet(
            path,
            columns=["Date", "Code"],
            filters=[("Code", "in", sorted(tickers))],
        )
        if frame.empty:
            continue
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame["Code"] = frame["Code"].astype(str).str.zfill(6)
        minimum = frame["Date"].min()
        provider_start = minimum if provider_start is None else min(provider_start, minimum)
        for ticker, value in frame.groupby("Code")["Date"].min().items():
            previous = first_seen.get(str(ticker))
            first_seen[str(ticker)] = value if previous is None else min(previous, value)
    if provider_start is None:
        raise ValueError("MARCAP provider has no rows for the evidence universe")
    return first_seen, hashes, provider_start


def _listing_age_proxy(
    *, signal: pd.Timestamp, first_seen: pd.Timestamp | None, provider_start: pd.Timestamp
) -> str:
    if first_seen is None:
        return "NO_PROVIDER_HISTORY"
    days = int((signal.normalize().tz_localize(None) - first_seen.normalize()).days)
    if days >= 365:
        return "OBSERVED_GE_365D"
    if first_seen.normalize() <= provider_start.normalize():
        return "LEFT_CENSORED_LT_365D"
    return "OBSERVED_LT_365D"


def _propensity_pipeline() -> Pipeline:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transform = ColumnTransformer(
        [
            ("numeric", numeric, list(NUMERIC_FEATURES)),
            ("categorical", categorical, list(CATEGORICAL_FEATURES)),
        ]
    )
    model = LogisticRegression(
        C=1.0,
        solver="liblinear",
        max_iter=2_000,
        random_state=42,
    )
    return Pipeline([("features", transform), ("model", model)])


def cross_fitted_propensity(
    frame: pd.DataFrame, *, folds: int = 5
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    required = {*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "issuer_id", "final_common"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"propensity frame fields missing: {missing}")
    groups = frame["issuer_id"].astype(str).to_numpy()
    target = frame["final_common"].astype(int).to_numpy()
    probabilities = np.full(len(frame), np.nan, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=folds)
    feature_frame = frame[[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES]].copy()
    for fold, (train, test) in enumerate(splitter.split(feature_frame, target, groups), start=1):
        if len(np.unique(target[train])) != 2:
            raise ValueError(f"propensity fold {fold} training set has one class")
        pipeline = _propensity_pipeline()
        pipeline.fit(feature_frame.iloc[train], target[train])
        predicted = pipeline.predict_proba(feature_frame.iloc[test])[:, 1]
        probabilities[test] = predicted
        fold_rows.append(
            {
                "fold": fold,
                "train_observation_count": len(train),
                "test_observation_count": len(test),
                "train_issuer_count": len(set(groups[train])),
                "test_issuer_count": len(set(groups[test])),
                "train_event_count": int(target[train].sum()),
                "test_event_count": int(target[test].sum()),
            }
        )
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("cross-fitted propensity must be finite and inside [0,1]")
    return probabilities, fold_rows


def _weighted_quantile(
    values: Sequence[float], weights: Sequence[float], quantile: float
) -> float:
    value_array = np.asarray(values, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    valid = np.isfinite(value_array) & np.isfinite(weight_array) & (weight_array > 0)
    value_array = value_array[valid]
    weight_array = weight_array[valid]
    if not len(value_array):
        return float("nan")
    order = np.argsort(value_array, kind="stable")
    cumulative = np.cumsum(weight_array[order])
    cutoff = quantile * float(cumulative[-1])
    return float(value_array[order][np.searchsorted(cumulative, cutoff, side="left")])


def weighted_spearman(
    x: Sequence[float], y: Sequence[float], weights: Sequence[float]
) -> float:
    left = np.asarray(x, dtype=float)
    right = np.asarray(y, dtype=float)
    weight = np.asarray(weights, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right) & np.isfinite(weight) & (weight > 0)
    left = left[valid]
    right = right[valid]
    weight = weight[valid]
    if len(left) < 5 or len(np.unique(left)) < 2 or len(np.unique(right)) < 2:
        return float("nan")
    left_rank = stats.rankdata(left, method="average")
    right_rank = stats.rankdata(right, method="average")
    total = float(weight.sum())
    left_mean = float(weight @ left_rank / total)
    right_mean = float(weight @ right_rank / total)
    covariance = float(weight @ ((left_rank - left_mean) * (right_rank - right_mean)) / total)
    left_variance = float(weight @ ((left_rank - left_mean) ** 2) / total)
    right_variance = float(weight @ ((right_rank - right_mean) ** 2) / total)
    denominator = math.sqrt(left_variance * right_variance)
    return covariance / denominator if denominator > 0 else float("nan")


def _effective_sample_size(weights: Sequence[float]) -> float:
    values = np.asarray(weights, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    return float(values.sum() ** 2 / (values @ values)) if len(values) else 0.0


def _band_summary(frame: pd.DataFrame, *, weight: str | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band, group in frame.groupby("full_evidence_band", sort=False):
        values = group["future_eri"].to_numpy(dtype=float)
        weights = (
            group[weight].to_numpy(dtype=float)
            if weight is not None
            else np.ones(len(group), dtype=float)
        )
        rows.append(
            {
                "band": str(band),
                "observation_count": len(group),
                "issuer_count": int(group["issuer_id"].nunique()),
                "weight_sum": float(weights.sum()),
                "effective_sample_size": _effective_sample_size(weights),
                "mean_future_eri": float(np.average(values, weights=weights)),
                "median_future_eri": _weighted_quantile(values, weights, 0.5),
                "negative_future_eri_share": float(
                    np.average((values < 0).astype(float), weights=weights)
                ),
            }
        )
    return sorted(rows, key=lambda row: BAND_ORDER.get(row["band"], 99))


def _adjacent_nondecreasing(rows: Sequence[dict[str, Any]]) -> int:
    values = [float(row["median_future_eri"]) for row in rows]
    return sum(left <= right for left, right in zip(values, values[1:]))


def _monthly_results(frame: pd.DataFrame, *, weight: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for month, group in frame.groupby("signal_month", sort=True):
        raw = weighted_spearman(
            group["full_evidence_index"],
            group["future_eri"],
            np.ones(len(group)),
        )
        weighted = weighted_spearman(
            group["full_evidence_index"], group["future_eri"], group[weight]
        )
        rows.append(
            {
                "schema_version": "moatrader-eri-selection-weighted-month-v2/1",
                "signal_month": str(month),
                "observation_count": len(group),
                "issuer_count": int(group["issuer_id"].nunique()),
                "effective_sample_size": _effective_sample_size(group[weight]),
                "raw_ic": raw,
                "selection_weighted_ic": weighted,
                "delta_ic": weighted - raw,
            }
        )
    return rows


def _calibration_rows(target: np.ndarray, propensity: np.ndarray) -> list[dict[str, Any]]:
    frame = pd.DataFrame({"target": target, "propensity": propensity})
    frame["decile"] = pd.qcut(
        frame["propensity"].rank(method="first"), 10, labels=False
    )
    return [
        {
            "decile": int(decile) + 1,
            "observation_count": len(group),
            "mean_predicted_probability": float(group["propensity"].mean()),
            "actual_eligibility_rate": float(group["target"].mean()),
        }
        for decile, group in frame.groupby("decile", sort=True)
    ]


def run_selection_weighting_v2(
    *,
    workspace: Path,
    eri_build: Path,
    coverage_audit: Path,
    size_diagnostic: Path,
    pre_outcome_build: Path,
    output: Path,
    audit_as_of: str,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production selection weighting requires a clean worktree")

    coverage_manifest_path = coverage_audit / "audit-manifest.json"
    coverage_stage_path = coverage_audit / "stage-status.json"
    ledger_path = coverage_audit / "observation-eligibility-ledger.jsonl"
    size_manifest_path = size_diagnostic / "build-manifest.json"
    size_stage_path = size_diagnostic / "stage-status.json"
    size_rows_path = size_diagnostic / "size-selection-observations.jsonl"
    pre_stage_path = pre_outcome_build / "stage-status.json"
    snapshots_path = pre_outcome_build / "private-pit-annual-financial-snapshots.jsonl"
    labels_path = eri_build / "future-eri-labels.jsonl"
    required = (
        coverage_manifest_path,
        coverage_stage_path,
        ledger_path,
        size_manifest_path,
        size_stage_path,
        size_rows_path,
        pre_stage_path,
        snapshots_path,
        labels_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"selection diagnostic inputs missing: {missing}")

    coverage_manifest = _read_json(coverage_manifest_path)
    coverage_stage = _read_json(coverage_stage_path)
    size_manifest = _read_json(size_manifest_path)
    size_stage = _read_json(size_stage_path)
    pre_stage = _read_json(pre_stage_path)
    full_rows_path = Path(str(coverage_manifest["input_paths"]["full_rows"]))
    checks = {
        "coverage_status": coverage_stage.get("status")
        == "V2_TERMINATED_ERI_ELIGIBILITY_BRIDGE_AUDITED",
        "coverage_ledger": coverage_manifest.get("output_hashes", {}).get("ledger")
        == sha256_file(ledger_path),
        "size_status": size_stage.get("status")
        == "V2_SIZE_NEUTRALIZATION_AND_SELECTION_DIAGNOSTICS_COMPLETE",
        "size_rows": size_manifest.get("output_hashes", {}).get("selection_rows")
        == sha256_file(size_rows_path),
        "snapshots": pre_stage.get("artifact_hashes", {}).get("annual_snapshots")
        == sha256_file(snapshots_path),
        "full_rows": coverage_manifest.get("input_hashes", {}).get("full_rows")
        == sha256_file(full_rows_path),
        "labels": coverage_manifest.get("input_hashes", {}).get("eri_labels")
        == sha256_file(labels_path),
        "sources_immutable": coverage_manifest.get("source_integrity_verification_status")
        == "PASS_NO_SOURCE_MUTATION",
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"sealed selection diagnostic inputs invalid: {failed}")

    input_hashes = {
        "coverage_manifest": sha256_file(coverage_manifest_path),
        "coverage_stage": sha256_file(coverage_stage_path),
        "ledger": sha256_file(ledger_path),
        "size_manifest": sha256_file(size_manifest_path),
        "size_stage": sha256_file(size_stage_path),
        "size_rows": sha256_file(size_rows_path),
        "pre_stage": sha256_file(pre_stage_path),
        "snapshots": sha256_file(snapshots_path),
        "full_rows": sha256_file(full_rows_path),
        "labels": sha256_file(labels_path),
    }
    ledger_by_id = _index(_read_records(ledger_path), source="eligibility ledger")
    size_by_id = _index(_read_records(size_rows_path), source="Size selection rows")
    full_all = _index(_read_records(full_rows_path), source="Full Evidence rows")
    full_by_id = {key: full_all[key] for key in ledger_by_id}
    if set(size_by_id) != set(ledger_by_id) or len(ledger_by_id) != 37_014:
        raise ValueError("selection inputs do not share the 37,014-row baseline")

    snapshots = _read_snapshots(snapshots_path)
    snapshots_by_ticker: dict[str, list[PITEconomicAnnualSnapshotV2]] = defaultdict(list)
    for snapshot in snapshots:
        snapshots_by_ticker[snapshot.issuer_id].append(snapshot)
    tickers = {str(row["issuer_id"]).zfill(6) for row in ledger_by_id.values()}
    first_seen, marcap_hashes, provider_start = _first_seen_dates(
        pre_stage, tickers=tickers
    )

    propensity_rows: list[dict[str, Any]] = []
    for observation_id in sorted(ledger_by_id):
        ledger = ledger_by_id[observation_id]
        size = size_by_id[observation_id]
        full = full_by_id[observation_id]
        signal = pd.Timestamp(ledger["signal_timestamp"])
        ticker = str(ledger["issuer_id"]).zfill(6)
        timeline = _timeline(snapshots_by_ticker.get(ticker, []), ticker, signal.to_pydatetime())
        valid_history = [item for item in timeline if _valid_financial(item)]
        latest = timeline[-1] if timeline else None
        metric_names = (
            "revenue",
            "operating_profit",
            "total_assets",
            "total_equity",
            "cash",
            "debt",
        )
        latest_metric_count = sum(
            latest is not None and getattr(latest, field) is not None for field in metric_names
        )
        year_span = (
            valid_history[-1].fiscal_year - valid_history[0].fiscal_year
            if len(valid_history) >= 2
            else 0
        )
        propensity_rows.append(
            {
                "schema_version": "moatrader-eri-selection-propensity-row-v2/1",
                "observation_id": observation_id,
                "issuer_id": ticker,
                "signal_timestamp": str(ledger["signal_timestamp"]),
                "signal_month": str(ledger["signal_timestamp"])[:7],
                "signal_year": str(ledger["signal_year"]),
                "final_common": bool(ledger["final_common"]),
                "log_market_cap": size.get("log_market_cap"),
                "signal_size_bucket": str(size["signal_size_bucket"]),
                "sector": str(ledger["sector"]),
                "listing_age_proxy": _listing_age_proxy(
                    signal=signal,
                    first_seen=first_seen.get(ticker),
                    provider_start=provider_start,
                ),
                "financial_archetype": str(ledger["valuation_route"]),
                "security_type": str(ledger["security_type"]),
                "evidence_source_mode": str(ledger["evidence_source_mode"]),
                "annual_source_mode": _source_mode(latest),
                "signal_market_data_available": (
                    "AVAILABLE" if size.get("log_market_cap") is not None else "UNAVAILABLE"
                ),
                "full_nobs": int(full["nobs"]),
                "coverage": float(full["coverage"]),
                "applicable_axis_count": int(full["applicable_axis_count"]),
                "unavailable_axis_count": int(full["unavailable_axis_count"]),
                "semantic_grounded_axis_count": int(full["semantic_grounded_axis_count"]),
                "deterministic_core_grounded_axis_count": int(
                    full["deterministic_core_grounded_axis_count"]
                ),
                "pit_snapshot_count": len(timeline),
                "pit_valid_history_count": len(valid_history),
                "latest_pit_metric_count": latest_metric_count,
                "pit_history_year_span": year_span,
            }
        )

    frame = pd.DataFrame(propensity_rows)
    propensity, fold_rows = cross_fitted_propensity(frame)
    event_rate = float(frame["final_common"].mean())
    clipped = np.clip(propensity, PROPENSITY_FLOOR, 1.0 - PROPENSITY_FLOOR)
    stabilized = event_rate / clipped
    capped = np.minimum(stabilized, WEIGHT_CAP)
    eligible_mask = frame["final_common"].to_numpy(dtype=bool)
    eligible_mean = float(capped[eligible_mask].mean())
    normalized = capped / eligible_mean
    for index, row in enumerate(propensity_rows):
        row["cross_fitted_propensity"] = float(propensity[index])
        row["clipped_propensity"] = float(clipped[index])
        row["stabilized_inverse_probability_weight"] = float(stabilized[index])
        row["analysis_weight_capped_20_normalized_eligible_mean_1"] = float(
            normalized[index]
        )
        row["outcome_values_opened_when_propensity_was_fit"] = False
    prohibited = ("future_eri", "actual_market_price", "counterfactual_value")
    if any(
        any(fragment in key.casefold() for fragment in prohibited)
        for row in propensity_rows
        for key in row
    ):
        raise ValueError("propensity rows contain ERI outcome fields")

    output.mkdir(parents=True, exist_ok=True)
    propensity_path = output / "selection-propensity-pre-eri.jsonl"
    _write_jsonl(propensity_path, propensity_rows)
    propensity_hash = sha256_file(propensity_path)
    seal = {
        "schema_version": "moatrader-eri-selection-propensity-seal-v2/1",
        "status": "SELECTION_PROPENSITY_FIT_AND_SEALED_BEFORE_ERI_VALUES_OPENED",
        "baseline_observation_count": len(frame),
        "eligible_observation_count": int(eligible_mask.sum()),
        "issuer_count": int(frame["issuer_id"].nunique()),
        "cross_fit": "ISSUER_GROUP_KFOLD_5",
        "numeric_features": list(NUMERIC_FEATURES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "sector_basis": "CURRENT_2026_KRX_KIND_NON_PIT_SENSITIVITY_ONLY",
        "listing_age_basis": "MARCAP_PROVIDER_FIRST_SEEN_LEFT_CENSOR_AWARE_PROXY",
        "financial_statement_type_available": False,
        "financial_statement_proxy": "ANNUAL_SOURCE_MODE_PLUS_METRIC_COMPLETENESS",
        "propensity_floor": PROPENSITY_FLOOR,
        "weight_cap": WEIGHT_CAP,
        "propensity_rows_sha256": propensity_hash,
        "future_eri_values_opened": False,
        "return_data_opened": False,
        "v2_primary_retest": False,
    }
    seal_path = output / "selection-propensity-seal.json"
    _write_json(seal_path, seal)

    # Future ERI values are opened only after the selection model and weights are sealed.
    label_by_id = _index(_read_records(labels_path), source="sealed ERI labels")
    eligible_ids = {row["observation_id"] for row in propensity_rows if row["final_common"]}
    if set(label_by_id) != eligible_ids or len(label_by_id) != 1_640:
        raise ValueError("sealed ERI labels do not match selected propensity rows")
    analysis_rows: list[dict[str, Any]] = []
    propensity_by_id = {row["observation_id"]: row for row in propensity_rows}
    for observation_id in sorted(label_by_id):
        source = propensity_by_id[observation_id]
        ledger = ledger_by_id[observation_id]
        analysis_rows.append(
            {
                "observation_id": observation_id,
                "issuer_id": source["issuer_id"],
                "signal_month": source["signal_month"],
                "full_evidence_index": float(ledger["full_evidence_index"]),
                "full_evidence_band": str(ledger["full_evidence_band"]),
                "future_eri": float(label_by_id[observation_id]["future_eri"]),
                "propensity": source["cross_fitted_propensity"],
                "analysis_weight": source[
                    "analysis_weight_capped_20_normalized_eligible_mean_1"
                ],
            }
        )
    analysis = pd.DataFrame(analysis_rows)
    monthly = _monthly_results(analysis, weight="analysis_weight")
    inference = {
        name: _inference(
            [float(row[name]) for row in monthly if math.isfinite(float(row[name]))],
            hac_lag_months=3,
            block_length_months=4,
            bootstrap_repetitions=10_000,
            bootstrap_seed=42,
        )
        for name in ("raw_ic", "selection_weighted_ic", "delta_ic")
    }
    raw_bands = _band_summary(analysis, weight=None)
    weighted_bands = _band_summary(analysis, weight="analysis_weight")
    selected_probabilities = propensity[eligible_mask]
    selected_weights = normalized[eligible_mask]
    summary = {
        "schema_version": "moatrader-eri-selection-weighting-diagnostic-v2/1",
        "status": "V2_SELECTION_WEIGHTING_SENSITIVITY_COMPLETE_NOT_PRIMARY_RETEST",
        "role": "SELECTION_SENSITIVITY_DIAGNOSTIC_ONLY",
        "propensity_sealed_before_future_eri_values_opened": True,
        "panel_observation_count": len(analysis),
        "panel_issuer_count": int(analysis["issuer_id"].nunique()),
        "propensity_model": {
            "cross_fit_folds": fold_rows,
            "oof_auc": float(roc_auc_score(eligible_mask.astype(int), propensity)),
            "oof_brier_score": float(
                brier_score_loss(eligible_mask.astype(int), propensity)
            ),
            "calibration_deciles": _calibration_rows(
                eligible_mask.astype(int), propensity
            ),
        },
        "weight_diagnostics": {
            "selected_propensity_min": float(selected_probabilities.min()),
            "selected_propensity_p10": float(np.quantile(selected_probabilities, 0.10)),
            "selected_propensity_median": float(np.median(selected_probabilities)),
            "selected_propensity_p90": float(np.quantile(selected_probabilities, 0.90)),
            "selected_propensity_max": float(selected_probabilities.max()),
            "selected_weight_min": float(selected_weights.min()),
            "selected_weight_median": float(np.median(selected_weights)),
            "selected_weight_p90": float(np.quantile(selected_weights, 0.90)),
            "selected_weight_p99": float(np.quantile(selected_weights, 0.99)),
            "selected_weight_max": float(selected_weights.max()),
            "weight_cap_hit_count": int(np.sum(stabilized[eligible_mask] > WEIGHT_CAP)),
            "unweighted_effective_sample_size": len(analysis),
            "weighted_effective_sample_size": _effective_sample_size(selected_weights),
        },
        "monthly_ic": inference,
        "unweighted_five_band": raw_bands,
        "selection_weighted_five_band": weighted_bands,
        "unweighted_adjacent_median_nondecreasing_count": _adjacent_nondecreasing(
            raw_bands
        ),
        "selection_weighted_adjacent_median_nondecreasing_count": _adjacent_nondecreasing(
            weighted_bands
        ),
        "selection_weighting_changes_v2_gate": False,
        "future_eri_used_as_signal": False,
        "future_eri_used_as_ranking": False,
        "ranking_output_produced": False,
        "return_data_opened": False,
        "causal_claim_allowed": False,
    }
    monthly_path = output / "monthly-selection-weighted-ic.jsonl"
    summary_path = output / "selection-weighting-summary.json"
    _write_jsonl(monthly_path, monthly)
    _write_json(summary_path, summary)

    after_hashes = {
        "coverage_manifest": sha256_file(coverage_manifest_path),
        "coverage_stage": sha256_file(coverage_stage_path),
        "ledger": sha256_file(ledger_path),
        "size_manifest": sha256_file(size_manifest_path),
        "size_stage": sha256_file(size_stage_path),
        "size_rows": sha256_file(size_rows_path),
        "pre_stage": sha256_file(pre_stage_path),
        "snapshots": sha256_file(snapshots_path),
        "full_rows": sha256_file(full_rows_path),
        "labels": sha256_file(labels_path),
    }
    if input_hashes != after_hashes:
        raise RuntimeError("a sealed input changed during selection weighting")
    status = {
        "schema_version": "moatrader-eri-selection-weighting-stage-v2/1",
        "status": "V2_SELECTION_WEIGHTING_DIAGNOSTIC_COMPLETE_NO_RETUNING",
        "audit_as_of": audit_as_of,
        "baseline_observation_count": len(frame),
        "eligible_observation_count": len(analysis),
        "propensity_sealed_before_future_eri_values_opened": True,
        "v2_primary_retest": False,
        "v2_thresholds_changed": False,
        "v2_index_formula_changed": False,
        "return_data_opened": False,
        "source_files_modified": False,
    }
    stage_path = output / "stage-status.json"
    _write_json(stage_path, status)
    output_files = {
        "propensity_rows": propensity_path,
        "propensity_seal": seal_path,
        "monthly": monthly_path,
        "summary": summary_path,
        "stage": stage_path,
    }
    _write_json(
        output / "build-manifest.json",
        {
            **status,
            "git_commit": commit,
            "worktree_dirty": False,
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
        description="Run outcome-blind ERI selection propensity weighting sensitivity."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--eri-build", type=Path, required=True)
    parser.add_argument("--coverage-audit", type=Path, required=True)
    parser.add_argument("--size-diagnostic", type=Path, required=True)
    parser.add_argument("--pre-outcome-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-as-of", required=True)
    args = parser.parse_args()
    result = run_selection_weighting_v2(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
