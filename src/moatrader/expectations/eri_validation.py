from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Literal, Sequence

import numpy as np
from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.expectations.future_eri import (
    EriMechanismObservationV1,
    EriMonotonicityPolicyV1,
    evaluate_future_eri_monotonicity,
)


class ClusteredEriObservationV1(ContractModel):
    observation_id: str = Field(min_length=1)
    issuer_id: str = Field(min_length=1)
    signal_timestamp: datetime
    evidence_f_score: int = Field(ge=-6, le=6)
    future_eri: Decimal

    @model_validator(mode="after")
    def aware_signal(self) -> "ClusteredEriObservationV1":
        if self.signal_timestamp.tzinfo is None or self.signal_timestamp.utcoffset() is None:
            raise ValueError("signal_timestamp must be timezone-aware")
        return self


class RegressionSlopeV1(ContractModel):
    slope: float | None = None
    standard_error: float | None = None
    t_statistic: float | None = None
    observation_count: int = Field(ge=0)
    issuer_cluster_count: int = Field(ge=0)
    calendar_cluster_count: int = Field(ge=0)
    covariance: Literal["ISSUER_X_CALENDAR_TWO_WAY_CLUSTER"] = (
        "ISSUER_X_CALENDAR_TWO_WAY_CLUSTER"
    )


class HacStatisticV1(ContractModel):
    mean: float | None = None
    standard_error: float | None = None
    t_statistic: float | None = None
    period_count: int = Field(ge=0)
    lag_months: int = Field(ge=0)


class ClusteredEriMechanismReportV1(ContractModel):
    schema_version: str = "moatrader-clustered-eri-mechanism-v1/1"
    observation_count: int = Field(ge=0)
    fixed_monotonicity_report: dict[str, Any]
    monthly_ic_hac: HacStatisticV1
    monthly_outer_spread_hac: HacStatisticV1
    panel_score_slope: RegressionSlopeV1
    mechanism_gate_passed: bool
    primary_endpoint: Literal["FIXED_FIVE_BAND_MONOTONICITY"] = "FIXED_FIVE_BAND_MONOTONICITY"
    secondary_statistics_only: bool = True
    overlapping_63_session_horizon_addressed: bool = True
    outcome_metric: Literal["FROZEN_EQUITY_ERI_V1", "ENTERPRISE_ERI_DIAGNOSTIC"]
    return_data_accessed: Literal[False] = False
    primary_ranking_policy: Literal["NONE_MECHANISM_ONLY"] = "NONE_MECHANISM_ONLY"


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2:
        return None
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    x_rank = _average_ranks(x_values)
    y_rank = _average_ranks(y_values)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _hac_mean(values: Sequence[float], *, lag: int) -> HacStatisticV1:
    clean = np.asarray([item for item in values if math.isfinite(item)], dtype=float)
    count = len(clean)
    if count == 0:
        return HacStatisticV1(period_count=0, lag_months=lag)
    mean = float(np.mean(clean))
    if count == 1:
        return HacStatisticV1(mean=mean, period_count=1, lag_months=lag)
    residual = clean - mean
    gamma_zero = float(np.dot(residual, residual) / count)
    long_run_variance = gamma_zero
    effective_lag = min(lag, count - 1)
    for offset in range(1, effective_lag + 1):
        gamma = float(np.dot(residual[offset:], residual[:-offset]) / count)
        weight = 1.0 - offset / (effective_lag + 1.0)
        long_run_variance += 2.0 * weight * gamma
    variance_of_mean = max(long_run_variance / count, 0.0)
    standard_error = math.sqrt(variance_of_mean)
    return HacStatisticV1(
        mean=mean,
        standard_error=standard_error,
        t_statistic=(mean / standard_error if standard_error > 0 else None),
        period_count=count,
        lag_months=lag,
    )


def _cluster_meat(
    x: np.ndarray,
    residual: np.ndarray,
    groups: Sequence[str],
) -> tuple[np.ndarray, int]:
    accumulators: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(x.shape[1]))
    for row, error, group in zip(x, residual, groups, strict=True):
        accumulators[group] += row * error
    meat = np.zeros((x.shape[1], x.shape[1]))
    for score in accumulators.values():
        meat += np.outer(score, score)
    return meat, len(accumulators)


def _two_way_cluster_slope(
    rows: Sequence[ClusteredEriObservationV1],
) -> RegressionSlopeV1:
    count = len(rows)
    issuer_count = len({item.issuer_id for item in rows})
    months = [item.signal_timestamp.strftime("%Y-%m") for item in rows]
    month_count = len(set(months))
    if count < 3:
        return RegressionSlopeV1(
            observation_count=count,
            issuer_cluster_count=issuer_count,
            calendar_cluster_count=month_count,
        )
    score = np.asarray([item.evidence_f_score for item in rows], dtype=float)
    y = np.asarray([float(item.future_eri) for item in rows], dtype=float)
    x = np.column_stack([np.ones(count), score])
    xtx_inverse = np.linalg.pinv(x.T @ x)
    beta = xtx_inverse @ x.T @ y
    residual = y - x @ beta
    issuer_meat, issuer_clusters = _cluster_meat(
        x,
        residual,
        [item.issuer_id for item in rows],
    )
    time_meat, time_clusters = _cluster_meat(x, residual, months)
    intersection_meat, _ = _cluster_meat(
        x,
        residual,
        [f"{item.issuer_id}|{month}" for item, month in zip(rows, months, strict=True)],
    )
    covariance = xtx_inverse @ (issuer_meat + time_meat - intersection_meat) @ xtx_inverse
    slope_variance = max(float(covariance[1, 1]), 0.0)
    standard_error = math.sqrt(slope_variance)
    return RegressionSlopeV1(
        slope=float(beta[1]),
        standard_error=standard_error,
        t_statistic=(float(beta[1]) / standard_error if standard_error > 0 else None),
        observation_count=count,
        issuer_cluster_count=issuer_clusters,
        calendar_cluster_count=time_clusters,
    )


def evaluate_clustered_eri_mechanism(
    rows: Sequence[ClusteredEriObservationV1],
    *,
    minimum_observations_per_band: int = 20,
    hac_lag_months: int = 3,
    outcome_metric: Literal[
        "FROZEN_EQUITY_ERI_V1", "ENTERPRISE_ERI_DIAGNOSTIC"
    ] = "FROZEN_EQUITY_ERI_V1",
) -> ClusteredEriMechanismReportV1:
    if not rows:
        raise ValueError("clustered ERI mechanism evaluation requires observations")
    ids = [item.observation_id for item in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("clustered ERI observation IDs must be unique")
    fixed = evaluate_future_eri_monotonicity(
        [
            EriMechanismObservationV1(
                observation_id=item.observation_id,
                signal_timestamp=item.signal_timestamp,
                evidence_f_score=item.evidence_f_score,
                future_eri=item.future_eri,
            )
            for item in rows
        ],
        policy=EriMonotonicityPolicyV1(
            minimum_observations_per_band=minimum_observations_per_band
        ),
    )
    by_month: dict[str, list[ClusteredEriObservationV1]] = defaultdict(list)
    for item in rows:
        by_month[item.signal_timestamp.strftime("%Y-%m")].append(item)
    monthly_ic: list[float] = []
    monthly_spread: list[float] = []
    for month_rows in by_month.values():
        ic = _spearman(
            [item.evidence_f_score for item in month_rows],
            [float(item.future_eri) for item in month_rows],
        )
        if ic is not None:
            monthly_ic.append(ic)
        bullish = [float(item.future_eri) for item in month_rows if item.evidence_f_score >= 1]
        bearish = [float(item.future_eri) for item in month_rows if item.evidence_f_score <= -1]
        if bullish and bearish:
            monthly_spread.append(float(np.mean(bullish) - np.mean(bearish)))
    return ClusteredEriMechanismReportV1(
        observation_count=len(rows),
        fixed_monotonicity_report=fixed.model_dump(mode="json"),
        monthly_ic_hac=_hac_mean(monthly_ic, lag=hac_lag_months),
        monthly_outer_spread_hac=_hac_mean(monthly_spread, lag=hac_lag_months),
        panel_score_slope=_two_way_cluster_slope(rows),
        mechanism_gate_passed=fixed.mechanism_gate_passed,
        outcome_metric=outcome_metric,
    )
