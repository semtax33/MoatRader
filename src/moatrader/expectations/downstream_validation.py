from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.expectations.eri_validation import HacStatisticV1, _hac_mean, _spearman


VALUE_METRICS = ("PBR", "PER", "P_FCF", "PSR", "PCR", "EV_EBITDA", "RPR")


class AnalystRevisionObservationV1(ContractModel):
    observation_id: str = Field(min_length=1)
    issuer_id: str = Field(min_length=1)
    signal_timestamp: datetime
    evidence_f_score: int = Field(ge=-6, le=6)
    horizon_days: Literal[5, 15, 30]
    consensus_revision: float
    source_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def aware_signal(self) -> "AnalystRevisionObservationV1":
        if self.signal_timestamp.tzinfo is None or self.signal_timestamp.utcoffset() is None:
            raise ValueError("signal_timestamp must be timezone-aware")
        return self


class FundamentalValidationObservationV1(ContractModel):
    observation_id: str = Field(min_length=1)
    issuer_id: str = Field(min_length=1)
    signal_timestamp: datetime
    evidence_f_score: int = Field(ge=-6, le=6)
    future_revenue_growth: float | None = None
    future_ebit_growth: float | None = None
    future_margin_change: float | None = None
    source_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def at_least_one_fundamental(self) -> "FundamentalValidationObservationV1":
        if self.signal_timestamp.tzinfo is None or self.signal_timestamp.utcoffset() is None:
            raise ValueError("signal_timestamp must be timezone-aware")
        if all(
            value is None
            for value in (
                self.future_revenue_growth,
                self.future_ebit_growth,
                self.future_margin_change,
            )
        ):
            raise ValueError("at least one future fundamental metric is required")
        return self


class ReturnNeutralizationObservationV1(ContractModel):
    observation_id: str = Field(min_length=1)
    issuer_id: str = Field(min_length=1)
    signal_timestamp: datetime
    predicted_revision_score: int = Field(ge=-6, le=6)
    future_return_63d: float
    value_metrics: dict[str, float | None]
    momentum: float | None = None
    analyst_revision_at_signal: float | None = None
    return_source_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def fixed_signal_and_value_taxonomy(self) -> "ReturnNeutralizationObservationV1":
        if self.signal_timestamp.tzinfo is None or self.signal_timestamp.utcoffset() is None:
            raise ValueError("signal_timestamp must be timezone-aware")
        unknown = set(self.value_metrics) - set(VALUE_METRICS)
        if unknown:
            raise ValueError(f"unsupported value metrics: {sorted(unknown)}")
        return self


def _require_mechanism_gate(mechanism_gate_passed: bool) -> None:
    if not mechanism_gate_passed:
        raise PermissionError("downstream stage is blocked until the ERI mechanism gate passes")


def _scalar_link(
    scores: Sequence[int],
    values: Sequence[float],
    months: Sequence[str],
) -> dict[str, Any]:
    overall = _spearman(scores, values)
    monthly: list[float] = []
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, month in enumerate(months):
        grouped[month].append(index)
    for indices in grouped.values():
        value = _spearman([scores[index] for index in indices], [values[index] for index in indices])
        if value is not None:
            monthly.append(value)
    hac = _hac_mean(monthly, lag=3)
    return {
        "observation_count": len(values),
        "spearman": overall,
        "monthly_ic_hac": hac.model_dump(mode="json"),
    }


def evaluate_analyst_revision(
    rows: Sequence[AnalystRevisionObservationV1],
    *,
    mechanism_gate_passed: bool,
) -> dict[str, Any]:
    _require_mechanism_gate(mechanism_gate_passed)
    result: dict[str, Any] = {}
    for horizon in (5, 15, 30):
        values = [item for item in rows if item.horizon_days == horizon]
        result[f"D_PLUS_{horizon}"] = _scalar_link(
            [item.evidence_f_score for item in values],
            [item.consensus_revision for item in values],
            [item.signal_timestamp.strftime("%Y-%m") for item in values],
        )
    return {
        "schema_version": "moatrader-analyst-revision-validation-v1/1",
        "status": "EVALUATED_AFTER_ERI_MECHANISM_GATE",
        "by_horizon": result,
        "future_return_opened": False,
        "future_eri_used_as_signal": False,
    }


def evaluate_future_fundamentals(
    rows: Sequence[FundamentalValidationObservationV1],
    *,
    mechanism_gate_passed: bool,
) -> dict[str, Any]:
    _require_mechanism_gate(mechanism_gate_passed)
    metrics: dict[str, Any] = {}
    for field in ("future_revenue_growth", "future_ebit_growth", "future_margin_change"):
        values = [item for item in rows if getattr(item, field) is not None]
        metrics[field] = _scalar_link(
            [item.evidence_f_score for item in values],
            [float(getattr(item, field)) for item in values],
            [item.signal_timestamp.strftime("%Y-%m") for item in values],
        )
    return {
        "schema_version": "moatrader-future-fundamental-validation-v1/1",
        "status": "EVALUATED_AFTER_ERI_MECHANISM_GATE",
        "metrics": metrics,
        "future_return_opened": False,
        "future_eri_used_as_signal": False,
    }


def _rank_percentiles(values: np.ndarray, months: Sequence[str]) -> np.ndarray:
    output = np.full(len(values), np.nan)
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, month in enumerate(months):
        if math.isfinite(values[index]):
            grouped[month].append(index)
    for indices in grouped.values():
        ordered = sorted(indices, key=lambda index: values[index])
        cursor = 0
        while cursor < len(ordered):
            end = cursor + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
                end += 1
            rank = (cursor + end - 1) / 2
            percentile = rank / max(len(ordered) - 1, 1)
            for index in ordered[cursor:end]:
                output[index] = percentile
            cursor = end
    return output


def _cluster_meat(
    x: np.ndarray,
    residual: np.ndarray,
    groups: Sequence[str],
) -> tuple[np.ndarray, int]:
    scores: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(x.shape[1]))
    for row, error, group in zip(x, residual, groups, strict=True):
        scores[group] += row * error
    meat = np.zeros((x.shape[1], x.shape[1]))
    for score in scores.values():
        meat += np.outer(score, score)
    return meat, len(scores)


def _regression_spec(
    rows: Sequence[ReturnNeutralizationObservationV1],
    *,
    controls: Sequence[str],
    uncontrolled_coefficient: float | None,
) -> dict[str, Any]:
    months_all = [item.signal_timestamp.strftime("%Y-%m") for item in rows]
    control_values: dict[str, np.ndarray] = {}
    for control in controls:
        if control in VALUE_METRICS:
            raw = np.asarray(
                [
                    float(item.value_metrics.get(control))
                    if item.value_metrics.get(control) is not None
                    else np.nan
                    for item in rows
                ]
            )
        elif control == "MOMENTUM":
            raw = np.asarray(
                [float(item.momentum) if item.momentum is not None else np.nan for item in rows]
            )
        elif control == "ANALYST_REVISION":
            raw = np.asarray(
                [
                    float(item.analyst_revision_at_signal)
                    if item.analyst_revision_at_signal is not None
                    else np.nan
                    for item in rows
                ]
            )
        else:
            raise ValueError(f"unknown neutralization control: {control}")
        control_values[control] = _rank_percentiles(raw, months_all)
    include = [
        index
        for index in range(len(rows))
        if all(math.isfinite(control_values[control][index]) for control in controls)
    ]
    if len(include) < max(3, len(controls) + 2):
        return {
            "controls": list(controls),
            "status": "INSUFFICIENT_COMPLETE_CASES",
            "observation_count": len(include),
            "predicted_revision_coefficient": None,
            "coefficient_retention_vs_uncontrolled": None,
            "signal_value_neutralization_r_squared": None,
        }
    selected = [rows[index] for index in include]
    months = [months_all[index] for index in include]
    unique_months = sorted(set(months))
    month_columns = [
        np.asarray([float(month == value) for month in months]) for value in unique_months[1:]
    ]
    score = np.asarray([item.predicted_revision_score for item in selected], dtype=float)
    y = np.asarray([item.future_return_63d for item in selected], dtype=float)
    columns = [np.ones(len(selected)), score]
    columns.extend(control_values[control][include] for control in controls)
    columns.extend(month_columns)
    x = np.column_stack(columns)
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    issuer_meat, issuer_count = _cluster_meat(
        x,
        residual,
        [item.issuer_id for item in selected],
    )
    time_meat, month_count = _cluster_meat(x, residual, months)
    cell_meat, _ = _cluster_meat(
        x,
        residual,
        [f"{item.issuer_id}|{month}" for item, month in zip(selected, months, strict=True)],
    )
    covariance = inverse @ (issuer_meat + time_meat - cell_meat) @ inverse
    variance = max(float(covariance[1, 1]), 0.0)
    standard_error = math.sqrt(variance)

    if controls:
        z_columns = [np.ones(len(selected))]
        z_columns.extend(control_values[control][include] for control in controls)
        z_columns.extend(month_columns)
        z = np.column_stack(z_columns)
        fitted_score = z @ np.linalg.pinv(z.T @ z) @ z.T @ score
        total = float(np.dot(score - np.mean(score), score - np.mean(score)))
        unexplained = float(np.dot(score - fitted_score, score - fitted_score))
        neutral_r_squared = 1.0 - unexplained / total if total > 0 else None
    else:
        neutral_r_squared = 0.0
    coefficient = float(beta[1])
    return {
        "controls": list(controls),
        "status": "OK",
        "observation_count": len(selected),
        "issuer_cluster_count": issuer_count,
        "calendar_cluster_count": month_count,
        "predicted_revision_coefficient": coefficient,
        "standard_error": standard_error,
        "t_statistic": coefficient / standard_error if standard_error > 0 else None,
        "coefficient_retention_vs_uncontrolled": (
            coefficient / uncontrolled_coefficient
            if uncontrolled_coefficient not in (None, 0.0)
            else None
        ),
        "signal_value_neutralization_r_squared": neutral_r_squared,
    }


def evaluate_return_value_neutralization(
    rows: Sequence[ReturnNeutralizationObservationV1],
    *,
    mechanism_gate_passed: bool,
) -> dict[str, Any]:
    _require_mechanism_gate(mechanism_gate_passed)
    if not rows:
        raise ValueError("return neutralization requires observations")
    uncontrolled = _regression_spec(rows, controls=(), uncontrolled_coefficient=None)
    base_coefficient = uncontrolled["predicted_revision_coefficient"]
    specifications: dict[str, Any] = {
        "UNCONTROLLED": uncontrolled,
        "ALL_VALUE_METRICS_JOINT": _regression_spec(
            rows,
            controls=VALUE_METRICS,
            uncontrolled_coefficient=base_coefficient,
        ),
    }
    for metric in VALUE_METRICS:
        specifications[f"VALUE_{metric}"] = _regression_spec(
            rows,
            controls=(metric,),
            uncontrolled_coefficient=base_coefficient,
        )
    specifications["PER_PBR_COMPARATOR_ONLY"] = _regression_spec(
        rows,
        controls=("PER", "PBR"),
        uncontrolled_coefficient=base_coefficient,
    )
    specifications["VALUE_MOMENTUM_REVISION_FULL"] = _regression_spec(
        rows,
        controls=(*VALUE_METRICS, "MOMENTUM", "ANALYST_REVISION"),
        uncontrolled_coefficient=base_coefficient,
    )
    return {
        "schema_version": "moatrader-return-value-neutralization-v1/1",
        "status": "EVALUATED_AFTER_ERI_MECHANISM_GATE",
        "primary_neutralization_spec": "ALL_VALUE_METRICS_JOINT",
        "value_metrics": list(VALUE_METRICS),
        "specifications": specifications,
        "signal_rank_policy": "F_SCORE_ONLY_NO_VALUE_PRIMARY_RANKING",
        "per_pbr_primary_ranking": False,
        "per_pbr_role": "COMPARATOR_CONTROL_ONLY",
        "actual_future_eri_used_as_signal": False,
        "return_data_accessed": True,
        "covariance": "ISSUER_X_CALENDAR_TWO_WAY_CLUSTER",
    }
