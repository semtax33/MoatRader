from __future__ import annotations

import argparse
import bisect
import json
import math
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from moatrader.backtest.universe_corrected import (
    rank_normal_score,
    residualize_cross_section,
    spearman_ic,
)
from moatrader.expectations.historical_evidence import sha256_file
from scripts.prepare_historical_evidence_index_eri_inputs_v2 import (
    PITEconomicAnnualSnapshotV2,
    _git_state,
    _load_sessions,
    _timeline,
)
from scripts.run_historical_evidence_index_short_momentum_v2 import (
    WINDOW_DEFINITIONS_V2,
    _load_return_maps,
    historical_momentum_window,
)
from scripts.run_historical_evidence_index_value_neutralization_v2 import (
    _inference,
    _read_json,
    _read_records,
    _write_json,
    _write_jsonl,
)


D = Decimal
BAND_ORDER_V3 = ("STRONG_BEAR", "BEAR", "NEUTRAL", "BULL", "STRONG_BULL")
VALUE_FIELDS_V3 = (
    "value_btm",
    "value_sales_yield",
    "value_operating_income_yield",
    "value_ebit_ev_yield",
    "value_assets_yield",
)
NUMERIC_CONTROL_TESTS_V3: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("raw", ()),
    ("size", ("log_market_cap",)),
    ("momentum_1m", ("momentum_1m",)),
    ("momentum_3_1", ("momentum_3_1",)),
    ("momentum_6_1", ("momentum_6_1",)),
    ("momentum_12_1", ("momentum_12_1",)),
    (
        "momentum_joint",
        ("momentum_1m", "momentum_3_1", "momentum_6_1", "momentum_12_1"),
    ),
    ("value_core_composite", ("value_core_composite",)),
    ("growth", ("growth_revenue_yoy",)),
    ("quality", ("quality_operating_roa_minus_leverage",)),
    (
        "momentum_growth_quality_joint",
        (
            "momentum_1m",
            "momentum_3_1",
            "momentum_6_1",
            "momentum_12_1",
            "growth_revenue_yoy",
            "quality_operating_roa_minus_leverage",
        ),
    ),
    (
        "all_numeric_joint",
        (
            "log_market_cap",
            "momentum_1m",
            "momentum_3_1",
            "momentum_6_1",
            "momentum_12_1",
            "value_core_composite",
            "growth_revenue_yoy",
            "quality_operating_roa_minus_leverage",
        ),
    ),
)


def _index(records: Sequence[dict[str, Any]], *, source: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        observation_id = str(record.get("observation_id", "")).strip()
        if not observation_id or observation_id in indexed:
            raise ValueError(f"{source} observation IDs must be nonblank and unique")
        indexed[observation_id] = record
    return indexed


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_yield(numerator: object, denominator: object) -> float | None:
    top = _finite(numerator)
    bottom = _finite(denominator)
    if top is None or bottom is None or top <= 0 or bottom <= 0:
        return None
    result = top / bottom
    return result if math.isfinite(result) else None


def _read_snapshots(path: Path) -> list[PITEconomicAnnualSnapshotV2]:
    with path.open("r", encoding="utf-8") as handle:
        return [
            PITEconomicAnnualSnapshotV2.model_validate_json(line)
            for line in handle
            if line.strip()
        ]


def _verify_hashes(hashes: dict[str, str], *, label: str) -> None:
    changed = [
        path for path, expected in hashes.items() if sha256_file(Path(path)) != expected
    ]
    if changed:
        raise RuntimeError(f"{label} source changed: {changed[:3]}")


def _contract_payload(
    contract_path: Path,
    *,
    full_rows_path: Path,
    ledger_path: Path,
    size_rows_path: Path,
    snapshot_path: Path,
    snapshot_stage_path: Path,
    discovery_summary_path: Path,
    marcap_files: Sequence[Path],
) -> dict[str, Any]:
    contract = _read_json(contract_path)
    frozen = contract.get("frozen_inputs") or {}
    expected_marcap = frozen.get("marcap_source_sha256_by_year") or {}
    actual_marcap = {
        path.stem.rsplit("-", 1)[-1]: sha256_file(path) for path in marcap_files
    }
    checks = {
        "status": contract.get("status")
        == "CONFIRMATORY_PROTOCOL_FROZEN_AFTER_NARROW_RETURN_DISCOVERY_BEFORE_BROAD_RETURN_PANEL",
        "baseline": frozen.get("baseline_observation_count") == 37_014,
        "full": frozen.get("full_evidence_rows_sha256") == sha256_file(full_rows_path),
        "ledger": frozen.get("eligibility_ledger_sha256") == sha256_file(ledger_path),
        "size": frozen.get("size_selection_rows_sha256") == sha256_file(size_rows_path),
        "snapshots": frozen.get("pit_annual_snapshots_sha256")
        == sha256_file(snapshot_path),
        "snapshot_stage": frozen.get("pit_snapshot_stage_sha256")
        == sha256_file(snapshot_stage_path),
        "discovery": frozen.get("narrow_panel_return_discovery_summary_sha256")
        == sha256_file(discovery_summary_path),
        "marcap": expected_marcap == actual_marcap,
        "open_anchor": contract.get("primary_outcome", {}).get("signal_price")
        == "EXACT_SIGNAL_SESSION_OPEN",
        "per_pbr_priority": contract.get("factor_controls", {}).get("per_pbr_priority")
        is False,
        "source_read_only": contract.get("source_and_role_guards", {}).get(
            "source_files_read_only"
        )
        is True,
        "no_ranking": contract.get("broadness_and_interpretation_gate", {}).get(
            "ranking_or_portfolio_output_allowed"
        )
        is False,
    }
    failed = sorted(key for key, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"invalid V3 broad-return contract or frozen source: {failed}")
    return contract


def _used_snapshot_hashes(
    snapshots: Iterable[PITEconomicAnnualSnapshotV2],
) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for snapshot in snapshots:
        for path, expected in snapshot.verified_source_hashes.items():
            previous = hashes.setdefault(str(Path(path).resolve()), str(expected))
            if previous != str(expected):
                raise ValueError(f"conflicting source hashes for {path}")
    return hashes


def _value_and_fundamental_controls(
    timeline: Sequence[PITEconomicAnnualSnapshotV2], market_cap: float | None
) -> tuple[dict[str, float | None], list[PITEconomicAnnualSnapshotV2]]:
    current = timeline[-1] if timeline else None
    used = [current] if current is not None else []
    values = {field: None for field in VALUE_FIELDS_V3}
    growth: float | None = None
    quality: float | None = None
    if current is not None and market_cap is not None and market_cap > 0:
        values["value_btm"] = _positive_yield(current.total_equity, market_cap)
        values["value_sales_yield"] = _positive_yield(current.revenue, market_cap)
        values["value_operating_income_yield"] = _positive_yield(
            current.operating_profit, market_cap
        )
        values["value_assets_yield"] = _positive_yield(current.total_assets, market_cap)
        enterprise_value = None
        if current.debt is not None and current.cash is not None:
            enterprise_value = D(str(market_cap)) + current.debt - current.cash
        values["value_ebit_ev_yield"] = _positive_yield(
            current.operating_profit, enterprise_value
        )
    if current is not None:
        if (
            current.operating_profit is not None
            and current.debt is not None
            and current.total_assets is not None
            and current.total_assets > 0
        ):
            quality = float(
                (current.operating_profit - current.debt) / current.total_assets
            )
        revenue_history = [
            item for item in timeline if item.revenue is not None and item.revenue > 0
        ]
        if len(revenue_history) >= 2:
            current_revenue, prior_revenue = revenue_history[-1], revenue_history[-2]
            growth = float(current_revenue.revenue / prior_revenue.revenue - D(1))
            used.extend([current_revenue, prior_revenue])
    return {
        **values,
        "growth_revenue_yoy": growth,
        "quality_operating_roa_minus_leverage": quality,
    }, list({item.rcept_no: item for item in used}.values())


def _add_monthly_value_composite(frame: pd.DataFrame) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    for _, month in frame.groupby("signal_month", sort=True):
        scores = pd.DataFrame(
            {field: rank_normal_score(month[field]) for field in VALUE_FIELDS_V3},
            index=month.index,
        )
        available = month[list(VALUE_FIELDS_V3)].notna().sum(axis=1)
        composite = scores.mean(axis=1, skipna=True).where(available >= 3)
        result.loc[month.index] = composite
    return result


def prepare_controls_pre_return(
    *,
    workspace: Path,
    contract_path: Path,
    full_rows_path: Path,
    ledger_path: Path,
    size_rows_path: Path,
    snapshot_path: Path,
    snapshot_stage_path: Path,
    discovery_summary_path: Path,
    marcap_files: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production V3 control preparation requires a clean worktree")
    contract = _contract_payload(
        contract_path,
        full_rows_path=full_rows_path,
        ledger_path=ledger_path,
        size_rows_path=size_rows_path,
        snapshot_path=snapshot_path,
        snapshot_stage_path=snapshot_stage_path,
        discovery_summary_path=discovery_summary_path,
        marcap_files=marcap_files,
    )
    full_all = _index(_read_records(full_rows_path), source="Full Evidence")
    ledger = _index(_read_records(ledger_path), source="eligibility ledger")
    size = _index(_read_records(size_rows_path), source="Size rows")
    if len(ledger) != 37_014 or set(size) != set(ledger) or not set(ledger).issubset(full_all):
        raise ValueError("V3 inputs do not share the exact 37,014-row baseline")
    full = {key: full_all[key] for key in ledger}
    snapshots = _read_snapshots(snapshot_path)
    snapshots_by_ticker: dict[str, list[PITEconomicAnnualSnapshotV2]] = defaultdict(list)
    for snapshot in snapshots:
        snapshots_by_ticker[snapshot.issuer_id].append(snapshot)
    tickers = {str(row["issuer_id"]).zfill(6) for row in ledger.values()}
    sessions = _load_sessions(marcap_files)
    returns_by_ticker, marcap_hashes = _load_return_maps(marcap_files, tickers)
    rows: list[dict[str, Any]] = []
    used_snapshots: list[PITEconomicAnnualSnapshotV2] = []
    for observation_id in sorted(ledger):
        evidence = full[observation_id]
        ledger_row = ledger[observation_id]
        size_row = size[observation_id]
        signal = pd.Timestamp(evidence["signal_timestamp"])
        if signal != pd.Timestamp(ledger_row["signal_timestamp"]) or signal != pd.Timestamp(
            size_row["signal_timestamp"]
        ):
            raise ValueError(f"signal timestamp mismatch: {observation_id}")
        ticker = str(evidence["issuer_id"]).zfill(6)
        if ticker != str(ledger_row["issuer_id"]).zfill(6):
            raise ValueError(f"issuer mismatch: {observation_id}")
        ticker_returns = returns_by_ticker.get(ticker, {})
        momentum_values: dict[str, float | None] = {}
        momentum_diagnostics: dict[str, dict[str, Any]] = {}
        for field, definition in WINDOW_DEFINITIONS_V2.items():
            calculated = historical_momentum_window(
                ticker_returns,
                sessions,
                signal_date=signal.date(),
                lookback_sessions=definition["lookback_sessions"],
                skip_most_recent_sessions=definition["skip_most_recent_sessions"],
                minimum_return_observations=definition["minimum_return_observations"],
            )
            momentum_values[field] = calculated.pop("value")
            momentum_diagnostics[field] = calculated
        market_cap = _finite(size_row.get("signal_open_market_cap"))
        visible = _timeline(
            snapshots_by_ticker.get(ticker, []), ticker, signal.to_pydatetime()
        )
        fundamental_values, used = _value_and_fundamental_controls(visible, market_cap)
        used_snapshots.extend(used)
        rows.append(
            {
                "schema_version": "moatrader-broad-return-control-row-v3/1",
                "observation_id": observation_id,
                "issuer_id": ticker,
                "signal_timestamp": signal.isoformat(),
                "signal_month": signal.strftime("%Y-%m"),
                "full_evidence_index": float(evidence["full_evidence_index"]),
                "full_evidence_band": str(evidence["band"]),
                "full_nobs": int(evidence["nobs"]),
                "coverage": float(evidence["coverage"]),
                "final_eri_1640": bool(ledger_row["final_common"]),
                "log_market_cap": _finite(size_row.get("log_market_cap")),
                "signal_open_market_cap": market_cap,
                "signal_size_bucket": str(size_row["signal_size_bucket"]),
                "security_type": str(ledger_row["security_type"]),
                "sector": str(ledger_row["sector"]),
                "sector_basis": str(ledger_row["sector_basis"]),
                "control_available_at": signal.isoformat(),
                "momentum_window_diagnostics": momentum_diagnostics,
                **momentum_values,
                **fundamental_values,
            }
        )
    frame = pd.DataFrame(rows)
    frame["value_core_composite"] = _add_monthly_value_composite(frame)
    rows = frame.where(pd.notna(frame), None).to_dict(orient="records")
    coverage_fields = (
        "log_market_cap",
        *WINDOW_DEFINITIONS_V2,
        *VALUE_FIELDS_V3,
        "value_core_composite",
        "growth_revenue_yoy",
        "quality_operating_roa_minus_leverage",
    )
    coverage = {
        field: int(pd.to_numeric(frame[field], errors="coerce").notna().sum())
        for field in coverage_fields
    }
    original_hashes = _used_snapshot_hashes(used_snapshots)
    _verify_hashes(original_hashes, label="original periodic filing")
    _verify_hashes(marcap_hashes, label="MARCAP")
    output.mkdir(parents=True, exist_ok=True)
    controls_path = output / "broad-controls-pre-return.jsonl"
    _write_jsonl(controls_path, rows)
    seal = {
        "schema_version": "moatrader-broad-return-control-seal-v3/1",
        "status": "V3_BROAD_CONTROLS_SEALED_RETURNS_CLOSED",
        "git_commit": commit,
        "worktree_dirty": False,
        "contract_sha256": sha256_file(contract_path),
        "script_sha256": sha256_file(Path(__file__)),
        "baseline_observation_count": len(rows),
        "baseline_issuer_count": int(frame["issuer_id"].nunique()),
        "baseline_signal_month_count": int(frame["signal_month"].nunique()),
        "final_eri_flag_count": int(frame["final_eri_1640"].sum()),
        "security_type_counts": {
            str(key): int(value)
            for key, value in frame["security_type"].value_counts().sort_index().items()
        },
        "control_coverage_counts": coverage,
        "value_metric_fields": list(VALUE_FIELDS_V3),
        "value_neutralizer_priority": "NONE_PARALLEL_SENSITIVITY",
        "per_pbr_primary_ranking": False,
        "sector_basis": "CURRENT_2026_KRX_KIND_NON_PIT_SENSITIVITY_ONLY",
        "momentum_window_definitions": WINDOW_DEFINITIONS_V2,
        "controls_sha256": sha256_file(controls_path),
        "marcap_source_hashes": marcap_hashes,
        "verified_original_periodic_filing_count": len(original_hashes),
        "original_periodic_filing_hashes_verified": True,
        "forward_returns_opened": False,
        "future_eri_values_opened": False,
        "return_used_as_signal_or_ranking": False,
        "source_files_modified": False,
        "outcome_stage_authorized": True,
    }
    seal_path = output / "broad-controls-seal.json"
    _write_json(seal_path, seal)
    status = {**seal, "control_seal_sha256": sha256_file(seal_path)}
    _write_json(output / "stage-status.json", status)
    return status


def _filtered_prices(
    files: Sequence[Path], *, dates: set[date], tickers: set[str]
) -> tuple[dict[tuple[date, str], dict[str, Any]], dict[str, str]]:
    selected_dates: dict[int, list[pd.Timestamp]] = defaultdict(list)
    for value in dates:
        selected_dates[value.year].append(pd.Timestamp(value))
    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    for path in files:
        hashes[str(path.resolve())] = sha256_file(path)
        year = int(path.stem.rsplit("-", 1)[-1])
        if year not in selected_dates:
            continue
        frame = pd.read_parquet(
            path,
            columns=["Date", "Code", "Open", "Close", "ChangesRatio"],
            filters=[
                ("Date", "in", sorted(selected_dates[year])),
                ("Code", "in", sorted(tickers)),
            ],
        )
        if frame.empty:
            continue
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.date
        frame["Code"] = frame["Code"].astype(str).str.zfill(6)
        pieces.append(frame)
    rows: dict[tuple[date, str], dict[str, Any]] = {}
    if pieces:
        combined = pd.concat(pieces, ignore_index=True).sort_values(["Date", "Code"])
        for row in combined.to_dict(orient="records"):
            rows.setdefault((row["Date"], str(row["Code"])), row)
    return rows, hashes


def open_to_close_forward_return(
    price_rows: dict[tuple[date, str], dict[str, Any]],
    *,
    ticker: str,
    holding_sessions: Sequence[date],
    minimum_following_returns: int,
) -> dict[str, Any]:
    if not holding_sessions:
        return {"status": "NO_EXACT_HORIZON"}
    signal_session = holding_sessions[0]
    target_session = holding_sessions[-1]
    signal = price_rows.get((signal_session, ticker))
    target = price_rows.get((target_session, ticker))
    if signal is None or _finite(signal.get("Open")) is None or _finite(signal.get("Close")) is None:
        return {"status": "NO_EXACT_SIGNAL_OPEN_CLOSE"}
    if target is None or _finite(target.get("Close")) is None:
        return {"status": "NO_EXACT_TARGET_CLOSE"}
    signal_open = float(signal["Open"])
    signal_close = float(signal["Close"])
    target_close = float(target["Close"])
    if signal_open <= 0 or signal_close <= 0 or target_close <= 0:
        return {"status": "NON_POSITIVE_SIGNAL_OR_TARGET_PRICE"}
    following_values = [
        value
        for session in holding_sessions[1:]
        if (row := price_rows.get((session, ticker))) is not None
        and (value := _finite(row.get("ChangesRatio"))) is not None
    ]
    if len(following_values) < minimum_following_returns:
        return {
            "status": "INSUFFICIENT_FOLLOWING_RETURN_COVERAGE",
            "observed_following_return_count": len(following_values),
        }
    gross = signal_close / signal_open
    for value in following_values:
        gross *= 1.0 + value / 100.0
    next_open_return: float | None = None
    if len(holding_sessions) >= 2:
        next_row = price_rows.get((holding_sessions[1], ticker))
        if next_row is not None:
            next_open = _finite(next_row.get("Open"))
            next_close = _finite(next_row.get("Close"))
            if next_open is not None and next_close is not None and next_open > 0 and next_close > 0:
                next_gross = next_close / next_open
                for session in holding_sessions[2:]:
                    row = price_rows.get((session, ticker))
                    value = _finite(row.get("ChangesRatio")) if row is not None else None
                    if value is not None:
                        next_gross *= 1.0 + value / 100.0
                next_open_return = next_gross - 1.0
    result = gross - 1.0
    if not math.isfinite(result):
        return {"status": "NON_FINITE_FORWARD_RETURN"}
    return {
        "status": "RETURN_ELIGIBLE",
        "forward_return_63_open_to_close": result,
        "forward_return_63_direct_close_over_open": target_close / signal_open - 1.0,
        "forward_return_63_next_open_sensitivity": next_open_return,
        "signal_session": signal_session.isoformat(),
        "target_session": target_session.isoformat(),
        "signal_open": signal_open,
        "signal_close": signal_close,
        "target_close": target_close,
        "observed_following_return_count": len(following_values),
        "expected_following_return_count": len(holding_sessions) - 1,
    }


def materialize_forward_returns(
    *,
    workspace: Path,
    contract_path: Path,
    full_rows_path: Path,
    ledger_path: Path,
    size_rows_path: Path,
    snapshot_path: Path,
    snapshot_stage_path: Path,
    discovery_summary_path: Path,
    marcap_files: Sequence[Path],
    control_build: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production V3 return materialization requires a clean worktree")
    contract = _contract_payload(
        contract_path,
        full_rows_path=full_rows_path,
        ledger_path=ledger_path,
        size_rows_path=size_rows_path,
        snapshot_path=snapshot_path,
        snapshot_stage_path=snapshot_stage_path,
        discovery_summary_path=discovery_summary_path,
        marcap_files=marcap_files,
    )
    controls_path = control_build / "broad-controls-pre-return.jsonl"
    control_seal_path = control_build / "broad-controls-seal.json"
    control_seal = _read_json(control_seal_path)
    if not (
        control_seal.get("status") == "V3_BROAD_CONTROLS_SEALED_RETURNS_CLOSED"
        and control_seal.get("baseline_observation_count") == 37_014
        and control_seal.get("controls_sha256") == sha256_file(controls_path)
        and control_seal.get("forward_returns_opened") is False
        and control_seal.get("outcome_stage_authorized") is True
    ):
        raise ValueError("V3 control seal does not authorize opening returns")
    controls = _read_records(controls_path)
    if len(controls) != 37_014:
        raise ValueError("sealed broad controls changed row count")
    sessions = _load_sessions(marcap_files)
    horizon = int(contract["primary_outcome"]["horizon_common_trading_sessions"])
    requested_dates: set[date] = set()
    holding_by_id: dict[str, list[date]] = {}
    for row in controls:
        signal_date = pd.Timestamp(row["signal_timestamp"]).date()
        location = bisect.bisect_left(sessions, signal_date)
        holding = list(sessions[location : location + horizon])
        if not holding or holding[0] != signal_date or len(holding) < horizon:
            holding = []
        holding_by_id[str(row["observation_id"])] = holding
        requested_dates.update(holding)
    tickers = {str(row["issuer_id"]).zfill(6) for row in controls}
    price_rows, marcap_hashes = _filtered_prices(
        marcap_files, dates=requested_dates, tickers=tickers
    )
    minimum = int(contract["primary_outcome"]["minimum_observed_following_session_returns"])
    results: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()
    for row in controls:
        observation_id = str(row["observation_id"])
        calculated = open_to_close_forward_return(
            price_rows,
            ticker=str(row["issuer_id"]).zfill(6),
            holding_sessions=holding_by_id[observation_id],
            minimum_following_returns=minimum,
        )
        exclusions[str(calculated["status"])] += 1
        results.append(
            {
                "schema_version": "moatrader-broad-forward-return-row-v3/1",
                "observation_id": observation_id,
                "issuer_id": str(row["issuer_id"]).zfill(6),
                "signal_timestamp": str(row["signal_timestamp"]),
                **calculated,
            }
        )
    eligible = [row for row in results if row["status"] == "RETURN_ELIGIBLE"]
    eligible_issuers = len({row["issuer_id"] for row in eligible})
    eligible_months = len({str(row["signal_timestamp"])[:7] for row in eligible})
    gate = contract["broadness_and_interpretation_gate"]
    broadness_checks = {
        "minimum_primary_return_observations": len(eligible)
        >= int(gate["minimum_primary_return_observations"]),
        "minimum_primary_return_issuers": eligible_issuers
        >= int(gate["minimum_primary_return_issuers"]),
        "minimum_valid_signal_months": eligible_months
        >= int(gate["minimum_valid_signal_months"]),
    }
    _verify_hashes(marcap_hashes, label="MARCAP")
    output.mkdir(parents=True, exist_ok=True)
    returns_path = output / "broad-forward-returns-63.jsonl"
    _write_jsonl(returns_path, results)
    seal = {
        "schema_version": "moatrader-broad-forward-return-seal-v3/1",
        "status": (
            "V3_BROAD_FORWARD_RETURNS_SEALED"
            if all(broadness_checks.values())
            else "V3_BROAD_FORWARD_RETURNS_SEALED_BROADNESS_GATE_FAILED"
        ),
        "git_commit": commit,
        "worktree_dirty": False,
        "contract_sha256": sha256_file(contract_path),
        "script_sha256": sha256_file(Path(__file__)),
        "control_seal_sha256": sha256_file(control_seal_path),
        "controls_sha256": sha256_file(controls_path),
        "returns_sha256": sha256_file(returns_path),
        "baseline_observation_count": len(results),
        "return_eligible_observation_count": len(eligible),
        "return_eligible_issuer_count": eligible_issuers,
        "return_eligible_signal_month_count": eligible_months,
        "return_status_counts": dict(sorted(exclusions.items())),
        "broadness_checks": broadness_checks,
        "broadness_gate_passed": all(broadness_checks.values()),
        "return_definition": contract["primary_outcome"],
        "marcap_source_hashes": marcap_hashes,
        "forward_returns_opened": True,
        "future_eri_values_opened": False,
        "return_used_as_signal_or_ranking": False,
        "ranking_output_produced": False,
        "source_files_modified": False,
        "evaluation_stage_authorized": all(broadness_checks.values()),
    }
    seal_path = output / "broad-forward-returns-seal.json"
    _write_json(seal_path, seal)
    status = {**seal, "return_seal_sha256": sha256_file(seal_path)}
    _write_json(output / "stage-status.json", status)
    return status


def _prepare_analysis_panel(
    controls: Sequence[dict[str, Any]], returns: Sequence[dict[str, Any]]
) -> pd.DataFrame:
    control_by_id = _index(controls, source="broad controls")
    return_by_id = _index(returns, source="broad returns")
    if set(control_by_id) != set(return_by_id):
        raise ValueError("broad control and return panels differ")
    rows: list[dict[str, Any]] = []
    numeric = {
        "full_evidence_index",
        "log_market_cap",
        *WINDOW_DEFINITIONS_V2,
        *VALUE_FIELDS_V3,
        "value_core_composite",
        "growth_revenue_yoy",
        "quality_operating_roa_minus_leverage",
        "forward_return_63_open_to_close",
        "forward_return_63_direct_close_over_open",
        "forward_return_63_next_open_sensitivity",
    }
    for observation_id in sorted(control_by_id):
        control = control_by_id[observation_id]
        outcome = return_by_id[observation_id]
        if str(control["signal_timestamp"]) != str(outcome["signal_timestamp"]):
            raise ValueError(f"control/return timestamp mismatch: {observation_id}")
        row = {**control, **outcome}
        for field in numeric:
            row[field] = _finite(row.get(field))
        rows.append(row)
    return pd.DataFrame(rows)


def _monthly_neutralization(
    panel: pd.DataFrame, *, minimum_n: int
) -> list[dict[str, Any]]:
    tests = list(NUMERIC_CONTROL_TESTS_V3) + [
        (f"value_parallel:{field}", (field,)) for field in VALUE_FIELDS_V3
    ]
    rows: list[dict[str, Any]] = []
    for signal_month, month in panel.groupby("signal_month", sort=True):
        for test, controls in tests:
            categorical = ("sector",) if test == "all_numeric_plus_sector_sensitivity" else ()
            columns = [
                "full_evidence_index",
                "forward_return_63_open_to_close",
                *controls,
                *categorical,
            ]
            sample = month[columns].dropna().copy()
            base = {
                "schema_version": "moatrader-broad-return-monthly-neutralization-v3/1",
                "signal_month": str(signal_month),
                "test": test,
                "numeric_controls": list(controls),
                "categorical_controls": list(categorical),
                "n": len(sample),
                "same_sample_raw_and_neutral": True,
            }
            if len(sample) < max(minimum_n, len(controls) + 3):
                rows.append({**base, "status": "INSUFFICIENT_MONTHLY_OBSERVATIONS"})
                continue
            raw_ic = spearman_ic(
                sample, "full_evidence_index", "forward_return_63_open_to_close"
            )
            if controls or categorical:
                sample["neutral_evidence"] = residualize_cross_section(
                    sample,
                    target="full_evidence_index",
                    numeric_controls=controls,
                    categorical_controls=categorical,
                )
            else:
                sample["neutral_evidence"] = rank_normal_score(
                    sample["full_evidence_index"]
                )
            sample = sample.dropna(subset=["neutral_evidence"])
            neutral_ic = spearman_ic(
                sample, "neutral_evidence", "forward_return_63_open_to_close"
            )
            post = [
                abs(
                    float(
                        sample["neutral_evidence"].corr(sample[field], method="spearman")
                    )
                )
                for field in controls
            ]
            rows.append(
                {
                    **base,
                    "status": "EVALUATED_SAME_SAMPLE",
                    "n": len(sample),
                    "raw_ic": raw_ic,
                    "neutral_ic": neutral_ic,
                    "delta_ic": neutral_ic - raw_ic,
                    "max_abs_post_numeric_control_spearman": max(post) if post else 0.0,
                }
            )
    # Current-sector classification is deliberately a separate non-PIT sensitivity.
    for signal_month, month in panel.groupby("signal_month", sort=True):
        controls = NUMERIC_CONTROL_TESTS_V3[-1][1]
        columns = [
            "full_evidence_index",
            "forward_return_63_open_to_close",
            *controls,
            "sector",
        ]
        sample = month[columns].dropna().copy()
        base = {
            "schema_version": "moatrader-broad-return-monthly-neutralization-v3/1",
            "signal_month": str(signal_month),
            "test": "all_numeric_plus_sector_sensitivity",
            "numeric_controls": list(controls),
            "categorical_controls": ["sector"],
            "n": len(sample),
            "same_sample_raw_and_neutral": True,
        }
        if len(sample) < max(minimum_n, len(controls) + 3):
            rows.append({**base, "status": "INSUFFICIENT_MONTHLY_OBSERVATIONS"})
            continue
        sample["neutral_evidence"] = residualize_cross_section(
            sample,
            target="full_evidence_index",
            numeric_controls=controls,
            categorical_controls=("sector",),
        )
        sample = sample.dropna(subset=["neutral_evidence"])
        raw_ic = spearman_ic(
            sample, "full_evidence_index", "forward_return_63_open_to_close"
        )
        neutral_ic = spearman_ic(
            sample, "neutral_evidence", "forward_return_63_open_to_close"
        )
        rows.append(
            {
                **base,
                "status": "EVALUATED_SAME_SAMPLE",
                "n": len(sample),
                "raw_ic": raw_ic,
                "neutral_ic": neutral_ic,
                "delta_ic": neutral_ic - raw_ic,
                "max_abs_post_numeric_control_spearman": max(
                    abs(float(sample["neutral_evidence"].corr(sample[field], method="spearman")))
                    for field in controls
                ),
            }
        )
    return rows


def _test_summary(
    panel: pd.DataFrame, monthly: Sequence[dict[str, Any]], contract: dict[str, Any]
) -> dict[str, Any]:
    tests = sorted({str(row["test"]) for row in monthly})
    result: dict[str, Any] = {}
    inference_config = contract["statistics"]
    for test in tests:
        selected = [
            row
            for row in monthly
            if row["test"] == test and row["status"] == "EVALUATED_SAME_SAMPLE"
        ]
        inference = {
            field: _inference(
                [float(row[field]) for row in selected],
                hac_lag_months=int(inference_config["newey_west_lag_months"]),
                block_length_months=int(inference_config["moving_block_length_months"]),
                bootstrap_repetitions=int(inference_config["bootstrap_repetitions"]),
                bootstrap_seed=int(inference_config["bootstrap_seed"]),
            )
            for field in ("raw_ic", "neutral_ic", "delta_ic")
        }
        raw = inference["raw_ic"]["newey_west"]["mean"]
        neutral = inference["neutral_ic"]["newey_west"]["mean"]
        retention = (
            float(neutral) / float(raw)
            if raw is not None
            and neutral is not None
            and math.isfinite(float(raw))
            and math.isfinite(float(neutral))
            and abs(float(raw)) > 1e-12
            else None
        )
        sample_n = sum(int(row["n"]) for row in selected)
        result[test] = {
            "valid_month_count": len(selected),
            "monthly_complete_case_observation_sum": sample_n,
            "same_sample_raw_and_neutral": True,
            **inference,
            "signed_ic_retention_ratio": retention,
        }
    return result


def _raw_outcome_summary(
    panel: pd.DataFrame, *, outcome: str, contract: dict[str, Any]
) -> dict[str, Any]:
    monthly: list[float] = []
    minimum = int(contract["factor_controls"]["minimum_monthly_observations"])
    for _, month in panel.groupby("signal_month", sort=True):
        sample = month[["full_evidence_index", outcome]].dropna()
        if len(sample) >= minimum:
            monthly.append(spearman_ic(sample, "full_evidence_index", outcome))
    return {
        "outcome": outcome,
        "available_observation_count": int(panel[outcome].notna().sum()),
        "valid_month_count": len(monthly),
        "ic": _inference(
            monthly,
            hac_lag_months=int(contract["statistics"]["newey_west_lag_months"]),
            block_length_months=int(contract["statistics"]["moving_block_length_months"]),
            bootstrap_repetitions=int(contract["statistics"]["bootstrap_repetitions"]),
            bootstrap_seed=int(contract["statistics"]["bootstrap_seed"]),
        ),
    }


def _five_band(panel: pd.DataFrame, *, outcome: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for band in BAND_ORDER_V3:
        group = panel.loc[panel["full_evidence_band"] == band, outcome].dropna()
        rows.append(
            {
                "band": band,
                "n": len(group),
                "mean_forward_return": float(group.mean()) if len(group) else None,
                "median_forward_return": float(group.median()) if len(group) else None,
                "positive_return_share": float((group > 0).mean()) if len(group) else None,
            }
        )
    return rows


def _selection_comparison(panel: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    eligible = panel.dropna(subset=["forward_return_63_open_to_close"])
    final_months = set(eligible.loc[eligible["final_eri_1640"], "signal_month"])
    groups = {
        "BROAD_RETURN_ELIGIBLE": eligible,
        "BROAD_MATCHED_TO_FINAL_ERI_SIGNAL_MONTHS": eligible[
            eligible["signal_month"].isin(final_months)
        ],
        "FINAL_ERI_1640_INTERSECTION": eligible[eligible["final_eri_1640"]],
        "COMMON_SECURITY_RETURN_ELIGIBLE_SENSITIVITY": eligible[
            eligible["security_type"] == "COMMON"
        ],
    }
    result: dict[str, Any] = {
        "diagnostic_timing": {
            "BROAD_RETURN_ELIGIBLE": "PREREGISTERED",
            "FINAL_ERI_1640_INTERSECTION": "PREREGISTERED",
            "BROAD_MATCHED_TO_FINAL_ERI_SIGNAL_MONTHS": (
                "POST_PRIMARY_RESULT_SELECTION_DECOMPOSITION"
            ),
            "COMMON_SECURITY_RETURN_ELIGIBLE_SENSITIVITY": (
                "POST_PRIMARY_RESULT_SECURITY_TYPE_SENSITIVITY"
            ),
        }
    }
    for name, group in groups.items():
        raw = _raw_outcome_summary(
            group, outcome="forward_return_63_open_to_close", contract=contract
        )
        bands = _five_band(group, outcome="forward_return_63_open_to_close")
        result[name] = {
            "observation_count": len(group),
            "issuer_count": int(group["issuer_id"].nunique()),
            "signal_month_count": int(group["signal_month"].nunique()),
            "mean_log_market_cap": float(group["log_market_cap"].mean()),
            "mean_nobs": float(group["full_nobs"].mean()),
            "factor_coverage_share": {
                field: float(group[field].notna().mean())
                for field in (
                    "log_market_cap",
                    *WINDOW_DEFINITIONS_V2,
                    "value_core_composite",
                    "growth_revenue_yoy",
                    "quality_operating_roa_minus_leverage",
                )
            },
            "raw_forward_return_ic": raw["ic"],
            "five_band": bands,
        }
    broad_ic = float(
        result["BROAD_RETURN_ELIGIBLE"]["raw_forward_return_ic"]["newey_west"]["mean"]
    )
    matched_ic = float(
        result["BROAD_MATCHED_TO_FINAL_ERI_SIGNAL_MONTHS"]["raw_forward_return_ic"][
            "newey_west"
        ]["mean"]
    )
    final_ic = float(
        result["FINAL_ERI_1640_INTERSECTION"]["raw_forward_return_ic"]["newey_west"][
            "mean"
        ]
    )
    result["selection_decomposition_post_primary"] = {
        "broad_raw_ic": broad_ic,
        "same_month_broad_raw_ic": matched_ic,
        "final_eri_intersection_raw_ic": final_ic,
        "calendar_composition_increment": matched_ic - broad_ic,
        "within_matched_month_final_eri_selection_increment": final_ic - matched_ic,
        "total_final_eri_minus_broad_increment": final_ic - broad_ic,
        "broad_retention_vs_final_eri": broad_ic / final_ic,
        "same_month_broad_retention_vs_final_eri": matched_ic / final_ic,
        "alpha_claim_allowed": False,
    }
    return result


def evaluate_broad_return(
    *,
    workspace: Path,
    contract_path: Path,
    full_rows_path: Path,
    ledger_path: Path,
    size_rows_path: Path,
    snapshot_path: Path,
    snapshot_stage_path: Path,
    discovery_summary_path: Path,
    marcap_files: Sequence[Path],
    control_build: Path,
    return_build: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production V3 broad-return evaluation requires a clean worktree")
    contract = _contract_payload(
        contract_path,
        full_rows_path=full_rows_path,
        ledger_path=ledger_path,
        size_rows_path=size_rows_path,
        snapshot_path=snapshot_path,
        snapshot_stage_path=snapshot_stage_path,
        discovery_summary_path=discovery_summary_path,
        marcap_files=marcap_files,
    )
    controls_path = control_build / "broad-controls-pre-return.jsonl"
    control_seal_path = control_build / "broad-controls-seal.json"
    returns_path = return_build / "broad-forward-returns-63.jsonl"
    return_seal_path = return_build / "broad-forward-returns-seal.json"
    control_seal = _read_json(control_seal_path)
    return_seal = _read_json(return_seal_path)
    if not (
        control_seal.get("controls_sha256") == sha256_file(controls_path)
        and control_seal.get("forward_returns_opened") is False
        and return_seal.get("returns_sha256") == sha256_file(returns_path)
        and return_seal.get("control_seal_sha256") == sha256_file(control_seal_path)
        and return_seal.get("broadness_gate_passed") is True
        and return_seal.get("evaluation_stage_authorized") is True
    ):
        raise ValueError("sealed V3 inputs do not authorize return evaluation")
    panel = _prepare_analysis_panel(_read_records(controls_path), _read_records(returns_path))
    eligible = panel.dropna(subset=["forward_return_63_open_to_close"]).copy()
    minimum = int(contract["factor_controls"]["minimum_monthly_observations"])
    monthly = _monthly_neutralization(eligible, minimum_n=minimum)
    tests = _test_summary(eligible, monthly, contract)
    raw_sensitivities = {
        outcome: _raw_outcome_summary(eligible, outcome=outcome, contract=contract)
        for outcome in (
            "forward_return_63_open_to_close",
            "forward_return_63_direct_close_over_open",
            "forward_return_63_next_open_sensitivity",
        )
    }
    bands = _five_band(eligible, outcome="forward_return_63_open_to_close")
    by_band = {row["band"]: row for row in bands}
    bull_bear_spread = (
        float(by_band["STRONG_BULL"]["median_forward_return"])
        - float(by_band["STRONG_BEAR"]["median_forward_return"])
        if by_band["STRONG_BULL"]["median_forward_return"] is not None
        and by_band["STRONG_BEAR"]["median_forward_return"] is not None
        else None
    )
    selection = _selection_comparison(panel, contract)
    raw_test = tests["raw"]
    gate_config = contract["broadness_and_interpretation_gate"]
    confirmation_checks = {
        "broadness_gate": return_seal["broadness_gate_passed"] is True,
        "minimum_valid_signal_months": raw_test["valid_month_count"]
        >= int(gate_config["minimum_valid_signal_months"]),
        "primary_bootstrap_ci_low_positive": float(
            raw_test["neutral_ic"]["moving_block_bootstrap"]["ci_low"]
        )
        > float(gate_config["supportive_historical_confirmation_requires_primary_bootstrap_ci_low_above"]),
    }
    confirmation_status = "SUPPORTIVE_HISTORICAL_CONFIRMATION" if all(
        confirmation_checks.values()
    ) else "HISTORICAL_CONFIRMATION_FAILED_OR_INCONCLUSIVE"
    output.mkdir(parents=True, exist_ok=True)
    monthly_path = output / "monthly-broad-return-neutralization.jsonl"
    summary_path = output / "broad-return-validation-summary.json"
    selection_path = output / "broad-vs-final-eri-selection-comparison.json"
    _write_jsonl(monthly_path, monthly)
    summary = {
        "schema_version": "moatrader-broad-return-validation-summary-v3/1",
        "status": confirmation_status,
        "independence_classification": contract["independence_classification"],
        "panel_observation_count": len(panel),
        "return_eligible_observation_count": len(eligible),
        "return_eligible_issuer_count": int(eligible["issuer_id"].nunique()),
        "return_eligible_signal_month_count": int(eligible["signal_month"].nunique()),
        "tests": tests,
        "raw_outcome_sensitivities": raw_sensitivities,
        "five_band": bands,
        "strong_bull_minus_strong_bear_median_forward_return": bull_bear_spread,
        "confirmation_checks": confirmation_checks,
        "alpha_claim_allowed": False,
        "ranking_output_produced": False,
        "per_pbr_primary_ranking": False,
        "sector_is_non_pit_sensitivity_only": True,
        "post_primary_selection_diagnostics_added": True,
    }
    _write_json(summary_path, summary)
    _write_json(selection_path, selection)
    manifest = {
        "schema_version": "moatrader-broad-return-validation-manifest-v3/1",
        "status": confirmation_status,
        "git_commit": commit,
        "worktree_dirty": False,
        "script_sha256": sha256_file(Path(__file__)),
        "contract_sha256": sha256_file(contract_path),
        "control_seal_sha256": sha256_file(control_seal_path),
        "controls_sha256": sha256_file(controls_path),
        "return_seal_sha256": sha256_file(return_seal_path),
        "returns_sha256": sha256_file(returns_path),
        "output_hashes": {
            "monthly": sha256_file(monthly_path),
            "summary": sha256_file(summary_path),
            "selection": sha256_file(selection_path),
        },
        "source_files_modified": False,
        "source_integrity_verification_status": "PASS_NO_SOURCE_MUTATION",
        "future_eri_used_as_signal_or_ranking": False,
        "return_used_as_signal_or_ranking": False,
        "historical_result_is_independent_oos": False,
        "alpha_claim_allowed": False,
        "ranking_output_produced": False,
    }
    manifest_path = output / "build-manifest.json"
    _write_json(manifest_path, manifest)
    stage = {
        **manifest,
        "manifest_sha256": sha256_file(manifest_path),
        "forward_returns_opened": True,
    }
    _write_json(output / "stage-status.json", stage)
    return stage


def _common_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--contract-path", type=Path, required=True)
    parser.add_argument("--full-rows-path", type=Path, required=True)
    parser.add_argument("--ledger-path", type=Path, required=True)
    parser.add_argument("--size-rows-path", type=Path, required=True)
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument("--snapshot-stage-path", type=Path, required=True)
    parser.add_argument("--discovery-summary-path", type=Path, required=True)
    parser.add_argument("--marcap-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run frozen V3 broad-universe forward-return validation."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = _common_parser()
    subparsers.add_parser("prepare-controls", parents=[common], add_help=True)
    materialize = subparsers.add_parser(
        "materialize-returns", parents=[common], add_help=True
    )
    materialize.add_argument("--control-build", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate", parents=[common], add_help=True)
    evaluate.add_argument("--control-build", type=Path, required=True)
    evaluate.add_argument("--return-build", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "prepare-controls":
        result = prepare_controls_pre_return(**args)
    elif command == "materialize-returns":
        result = materialize_forward_returns(**args)
    else:
        result = evaluate_broad_return(**args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
