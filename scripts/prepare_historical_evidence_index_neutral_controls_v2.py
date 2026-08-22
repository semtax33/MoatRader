from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from moatrader.backtest.universe_corrected import trailing_momentum
from moatrader.expectations.historical_evidence import sha256_file
from scripts.prepare_historical_evidence_index_eri_inputs_v2 import (
    PITEconomicAnnualSnapshotV2,
    _git_state,
    _load_sessions,
    _read_json,
    _read_jsonl,
    _timeline,
    _valid_financial,
)
from scripts.run_historical_evidence_index_value_neutralization_v2 import (
    NEUTRALIZER_PRIORITY_V2,
    VALUE_METRIC_FIELDS_V2,
)
from scripts.run_v7_1_value_neutral_sensitivity import extract_value_fundamentals


D = Decimal
SEOUL = ZoneInfo("Asia/Seoul")
FACTOR_CONTROL_FIELDS_V2 = (
    "factor_momentum_12_1",
    "factor_revenue_growth_yoy",
    "factor_quality_roa_cfo_leverage",
    "factor_analyst_forward_eps_yield",
    "factor_analyst_eps_revision_30d",
)
FACTOR_DEFINITIONS_V2 = {
    "factor_momentum_12_1": "compounded daily return from signal-365d through signal-30d",
    "factor_revenue_growth_yoy": "latest PIT annual revenue / prior PIT annual revenue - 1",
    "factor_quality_roa_cfo_leverage": "(operating income + CFO - debt) / total assets",
    "factor_analyst_forward_eps_yield": (
        "median latest-per-broker positive EPS for nearest nonpast fiscal year / signal price"
    ),
    "factor_analyst_eps_revision_30d": (
        "same-target-year analyst EPS consensus at signal / consensus at signal-30d - 1"
    ),
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
    return [dict(item) for item in raw]


def _finite_or_none(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _positive_yield(numerator: object, denominator: object) -> float | None:
    top = _finite_or_none(numerator)
    bottom = _finite_or_none(denominator)
    if top is None or bottom is None or top <= 0 or bottom <= 0:
        return None
    result = top / bottom
    return result if math.isfinite(result) else None


def _parse_timestamp(raw: object) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    for pattern in ("%Y%m%d%H%M%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            parsed = datetime.strptime(text, pattern)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=SEOUL)
    return None


def _parse_decimal(raw: object) -> Decimal | None:
    text = str(raw or "").replace(",", "").strip()
    if not text:
        return None
    try:
        value = D(text)
    except Exception:
        return None
    return value if value.is_finite() else None


def _load_hankyung_records(
    root: Path, tickers: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], Counter[str]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    hashes: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for path in sorted(root.glob("*.json")):
        ticker = path.name[:6]
        if ticker not in tickers or len(path.name) < 8 or path.name[6] != "_":
            continue
        counts["RELEVANT_TICKER_REPORT_FILE"] += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            counts["INVALID_JSON"] += 1
            continue
        registered = _parse_timestamp(payload.get("REGISTER_DATE"))
        updated = _parse_timestamp(payload.get("UPDATE_DATE")) or registered
        eps = _parse_decimal(payload.get("STOCK_PRE_EPS"))
        settlement = str(payload.get("STOCK_SETTLEMENT_DAY") or "").strip()
        if not (registered and updated and eps is not None and eps > 0):
            counts["MISSING_PIT_EPS_FIELDS"] += 1
            continue
        if len(settlement) < 4 or not settlement[:4].isdigit():
            counts["INVALID_SETTLEMENT_PERIOD"] += 1
            continue
        available_at = max(registered, updated)
        source_hash = sha256_file(path)
        hashes[str(path.resolve())] = source_hash
        rows[ticker].append(
            {
                "available_at": available_at,
                "registered_at": registered,
                "updated_at": updated,
                "forecast_year": int(settlement[:4]),
                "eps": eps,
                "broker": str(
                    payload.get("PUBLISH_CODE") or payload.get("OFFICE_NAME") or path.name
                ),
                "source_id": f"HANKYUNG:{source_hash}:{payload.get('REPORT_IDX', path.stem)}",
            }
        )
        counts["VALID_PIT_EPS_REPORT"] += 1
    for values in rows.values():
        values.sort(key=lambda item: (item["available_at"], item["source_id"]))
    return rows, hashes, counts


def analyst_eps_consensus(
    records: Sequence[dict[str, Any]],
    *,
    cutoff: datetime,
    forecast_year: int | None = None,
    lookback_days: int = 180,
    minimum_brokers: int = 2,
) -> dict[str, Any] | None:
    lower = cutoff - timedelta(days=lookback_days)
    visible = [
        item
        for item in records
        if lower <= item["available_at"] <= cutoff and item["forecast_year"] >= cutoff.year
    ]
    if forecast_year is None:
        years = sorted({int(item["forecast_year"]) for item in visible})
        if not years:
            return None
        forecast_year = years[0]
    visible = [item for item in visible if int(item["forecast_year"]) == forecast_year]
    latest_by_broker: dict[str, dict[str, Any]] = {}
    for item in visible:
        broker = str(item["broker"])
        previous = latest_by_broker.get(broker)
        if previous is None or item["available_at"] > previous["available_at"]:
            latest_by_broker[broker] = item
    if len(latest_by_broker) < minimum_brokers:
        return None
    selected = list(latest_by_broker.values())
    return {
        "forecast_year": forecast_year,
        "eps": D(str(statistics.median(float(item["eps"]) for item in selected))),
        "broker_count": len(selected),
        "available_at": max(item["available_at"] for item in selected),
        "source_ids": sorted(str(item["source_id"]) for item in selected),
    }


def _load_momentum_panel(
    files: Sequence[Path], tickers: set[str]
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    pieces: list[pd.DataFrame] = []
    hashes: dict[str, str] = {}
    selected = sorted(tickers)
    for path in files:
        hashes[str(path.resolve())] = sha256_file(path)
        frame = pd.read_parquet(
            path,
            columns=["Date", "Code", "ChangesRatio"],
            filters=[("Code", "in", selected)],
        )
        if frame.empty:
            continue
        frame["Date"] = pd.to_datetime(frame["Date"])
        frame["Code"] = frame["Code"].astype(str).str.zfill(6)
        pieces.append(frame)
    combined = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    return (
        {ticker: group.sort_values("Date") for ticker, group in combined.groupby("Code")}
        if not combined.empty
        else {},
        hashes,
    )


def _normalized_fundamentals(
    root: Path, snapshot: PITEconomicAnnualSnapshotV2
) -> tuple[dict[str, float | None], str | None, str | None]:
    path = root / f"kr_normalized_{snapshot.issuer_id}_{snapshot.fiscal_year}.12.csv"
    if not path.is_file():
        return {}, None, None
    fundamentals = extract_value_fundamentals(pd.read_csv(path, low_memory=False))
    source_hash = sha256_file(path)
    return fundamentals, str(path.resolve()), source_hash


def _value_metrics(
    *,
    snapshot: PITEconomicAnnualSnapshotV2,
    supplemental: dict[str, float | None],
    market_cap: Decimal,
) -> dict[str, float | None]:
    revenue = snapshot.revenue
    ebit = snapshot.operating_profit
    assets = snapshot.total_assets
    equity = snapshot.total_equity
    cash = snapshot.cash
    debt = snapshot.debt
    net_income = supplemental.get("fund_net_income")
    cfo = supplemental.get("fund_cfo")
    capex = supplemental.get("fund_capex")
    dna = supplemental.get("fund_dna")
    gross_profit = supplemental.get("fund_gross_profit")
    rnd = supplemental.get("fund_rnd")
    retained = supplemental.get("fund_retained_earnings")
    current_assets = supplemental.get("fund_current_assets")
    liabilities = supplemental.get("fund_total_liabilities")
    enterprise_value = (
        market_cap + debt - cash if debt is not None and cash is not None else None
    )
    fcf = (
        D(str(cfo)) - D(str(capex)) if cfo is not None and capex is not None else None
    )
    ebitda = (
        ebit + D(str(dna)) if ebit is not None and dna is not None else None
    )
    ncav = (
        D(str(current_assets)) - D(str(liabilities))
        if current_assets is not None and liabilities is not None
        else None
    )
    return {
        "value_btm": _positive_yield(equity, market_cap),
        "value_earnings_yield": _positive_yield(net_income, market_cap),
        "value_fcf_yield": _positive_yield(fcf, market_cap),
        "value_sales_yield": _positive_yield(revenue, market_cap),
        "value_cfo_yield": _positive_yield(cfo, market_cap),
        "value_ebitda_ev_yield": _positive_yield(ebitda, enterprise_value),
        "value_ebit_ev_yield": _positive_yield(ebit, enterprise_value),
        "value_operating_income_yield": _positive_yield(ebit, market_cap),
        "value_gross_profit_yield": _positive_yield(gross_profit, market_cap),
        "value_rnd_yield": _positive_yield(rnd, market_cap),
        "value_retained_earnings_yield": _positive_yield(retained, market_cap),
        "value_assets_yield": _positive_yield(assets, market_cap),
        "value_ncav_yield": _positive_yield(ncav, market_cap),
    }


def prepare_neutral_controls_v2(
    *,
    workspace: Path,
    eri_build: Path,
    pre_outcome_build: Path,
    arcana_snapshot_root: Path,
    analyst_root: Path,
    marcap_files: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production neutral-control preparation requires a clean worktree")
    eri_stage_path = eri_build / "stage-status.json"
    eri_manifest_path = eri_build / "build-manifest.json"
    feature_path = eri_build / "features-with-frozen-expectations-pre-outcome.jsonl"
    labels_path = eri_build / "future-eri-labels.jsonl"
    eri_stage = _read_json(eri_stage_path)
    eri_manifest = _read_json(eri_manifest_path)
    if not (
        eri_stage.get("status")
        in {"FULL_PRIMARY_MECHANISM_PASSED", "FULL_PRIMARY_MECHANISM_REJECTED_OR_INCONCLUSIVE"}
        and eri_stage.get("value_neutralization_stage_authorized") is True
        and eri_stage.get("return_data_opened") is False
        and eri_stage.get("future_eri_used_as_signal") is False
        and eri_stage.get("future_eri_used_as_ranking") is False
        and eri_manifest.get("future_eri_labels_sha256") == sha256_file(labels_path)
        and eri_manifest.get("feature_input_sha256") == sha256_file(feature_path)
    ):
        raise ValueError("ERI gate does not authorize neutral-control preparation")
    pre_stage = _read_json(pre_outcome_build / "stage-status.json")
    snapshot_path = pre_outcome_build / "private-pit-annual-financial-snapshots.jsonl"
    if pre_stage.get("artifact_hashes", {}).get("annual_snapshots") != sha256_file(snapshot_path):
        raise ValueError("sealed PIT annual snapshots changed")

    features = {item["observation_id"]: item for item in _read_records(feature_path)}
    label_records = _read_records(labels_path)
    label_ids = {str(item["observation_id"]) for item in label_records}
    if len(label_ids) != len(label_records):
        raise ValueError("ERI labels must have unique observation IDs")
    snapshots = _read_jsonl(snapshot_path, PITEconomicAnnualSnapshotV2)
    by_ticker: dict[str, list[PITEconomicAnnualSnapshotV2]] = defaultdict(list)
    for snapshot in snapshots:
        by_ticker[snapshot.issuer_id].append(snapshot)
    tickers = {str(features[key]["issuer_id"]).zfill(6) for key in label_ids}
    momentum, marcap_hashes = _load_momentum_panel(marcap_files, tickers)
    analyst, analyst_hashes, analyst_counts = _load_hankyung_records(analyst_root, tickers)
    _load_sessions(marcap_files)  # validates the common trading calendar without opening returns

    normalized_hashes: dict[str, str] = {}
    original_hashes: dict[str, str] = {}
    normalized_cache: dict[
        tuple[str, int], tuple[dict[str, float | None], str | None, str | None]
    ] = {}
    value_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    coverage: Counter[str] = Counter()
    for observation_id in sorted(label_ids):
        feature = features[observation_id]
        ticker = str(feature["issuer_id"]).zfill(6)
        signal = datetime.fromisoformat(str(feature["signal_timestamp"]))
        history = [
            item
            for item in _timeline(by_ticker.get(ticker, []), ticker, signal)
            if _valid_financial(item)
        ]
        if len(history) < 2:
            raise ValueError(f"sealed ERI feature lost its PIT financial history: {observation_id}")
        current, prior = history[-1], history[-2]
        normalized_key = (current.issuer_id, current.fiscal_year)
        if normalized_key not in normalized_cache:
            normalized_cache[normalized_key] = _normalized_fundamentals(
                arcana_snapshot_root, current
            )
        supplemental, normalized_path, normalized_hash = normalized_cache[normalized_key]
        if normalized_path and normalized_hash:
            normalized_hashes[normalized_path] = normalized_hash
        original_hashes.update(current.verified_source_hashes)
        expectation = feature["expectation_state"]
        assumptions = feature["frozen_expectation_assumptions"]
        price = D(str(expectation["market_price"]))
        shares = D(str(assumptions["diluted_shares"]))
        market_cap = price * shares
        values = _value_metrics(
            snapshot=current,
            supplemental=supplemental,
            market_cap=market_cap,
        )
        annual_source_ids = sorted(
            {
                source_id
                for source_ids in current.metric_source_ids.values()
                for source_id in source_ids
            }
        )
        value_source_ids = [str(expectation["market_price_source_id"]), *annual_source_ids]
        if normalized_hash:
            value_source_ids.append(
                f"ARCANA_NORMALIZED:{normalized_hash}:{current.issuer_id}:{current.fiscal_year}FY"
            )
        value_rows.append(
            {
                "schema_version": "moatrader-evidence-index-value-control-row-v2/1",
                "observation_id": observation_id,
                "issuer_id": ticker,
                "signal_timestamp": signal.isoformat(),
                "value_available_at": signal.isoformat(),
                "value_source_ids": sorted(set(value_source_ids)),
                **values,
            }
        )
        for field, value in values.items():
            coverage[f"VALUE:{field}"] += int(value is not None)

        revenue_growth = (
            float(current.revenue / prior.revenue - D(1))
            if current.revenue is not None and prior.revenue is not None and prior.revenue > 0
            else None
        )
        cfo = supplemental.get("fund_cfo")
        quality = (
            float(
                (current.operating_profit + D(str(cfo)) - current.debt)
                / current.total_assets
            )
            if cfo is not None
            and current.operating_profit is not None
            and current.debt is not None
            and current.total_assets is not None
            and current.total_assets > 0
            else None
        )
        ticker_momentum = momentum.get(ticker)
        momentum_value = (
            trailing_momentum(ticker_momentum, as_of=signal.date())
            if ticker_momentum is not None
            else float("nan")
        )
        momentum_value = momentum_value if math.isfinite(momentum_value) else None
        current_consensus = analyst_eps_consensus(analyst.get(ticker, []), cutoff=signal)
        prior_consensus = (
            analyst_eps_consensus(
                analyst.get(ticker, []),
                cutoff=signal - timedelta(days=30),
                forecast_year=int(current_consensus["forecast_year"]),
            )
            if current_consensus
            else None
        )
        eps_yield = (
            float(current_consensus["eps"] / price)
            if current_consensus and price > 0
            else None
        )
        eps_revision = (
            float(current_consensus["eps"] / prior_consensus["eps"] - D(1))
            if current_consensus
            and prior_consensus
            and prior_consensus["eps"] > 0
            else None
        )
        factor_source_ids = sorted(
            set(
                value_source_ids
                + ([f"MARCAP_12_1:{ticker}:{signal.date().isoformat()}"] if momentum_value is not None else [])
                + (list(current_consensus["source_ids"]) if current_consensus else [])
                + (list(prior_consensus["source_ids"]) if prior_consensus else [])
            )
        )
        factor_values = {
            "factor_momentum_12_1": momentum_value,
            "factor_revenue_growth_yoy": revenue_growth,
            "factor_quality_roa_cfo_leverage": quality,
            "factor_analyst_forward_eps_yield": eps_yield,
            "factor_analyst_eps_revision_30d": eps_revision,
        }
        factor_rows.append(
            {
                "schema_version": "moatrader-evidence-index-factor-control-row-v2/1",
                "observation_id": observation_id,
                "issuer_id": ticker,
                "signal_timestamp": signal.isoformat(),
                "factor_available_at": signal.isoformat(),
                "factor_source_ids": factor_source_ids,
                "analyst_forecast_year": (
                    int(current_consensus["forecast_year"]) if current_consensus else None
                ),
                "analyst_broker_count": (
                    int(current_consensus["broker_count"]) if current_consensus else 0
                ),
                **factor_values,
            }
        )
        for field, value in factor_values.items():
            coverage[f"FACTOR:{field}"] += int(value is not None)

    for path, expected in {
        **original_hashes,
        **normalized_hashes,
        **analyst_hashes,
        **marcap_hashes,
    }.items():
        if sha256_file(Path(path)) != expected:
            raise RuntimeError(f"neutral-control source changed during construction: {path}")
    output.mkdir(parents=True, exist_ok=True)
    value_path = output / "value-controls.jsonl"
    factor_path = output / "factor-controls.jsonl"
    _write_jsonl(value_path, value_rows)
    _write_jsonl(factor_path, factor_rows)
    common = {
        "git_commit": commit,
        "worktree_dirty": False,
        "eri_stage_status_sha256": sha256_file(eri_stage_path),
        "eri_build_manifest_sha256": sha256_file(eri_manifest_path),
        "feature_input_sha256": sha256_file(feature_path),
        "future_eri_labels_sha256": sha256_file(labels_path),
        "point_in_time_at_signal_verified": True,
        "return_data_opened": False,
        "source_files_read_only": True,
        "source_files_modified": False,
        "source_integrity_verification_status": "PASS_NO_SOURCE_MUTATION",
        "source_hashes": {
            "pre_outcome_stage": sha256_file(pre_outcome_build / "stage-status.json"),
            "annual_snapshot": sha256_file(snapshot_path),
            "original_regular_filing_count": len(original_hashes),
            "normalized_snapshot_count": len(normalized_hashes),
            "analyst_report_count": len(analyst_hashes),
            "marcap_file_count": len(marcap_hashes),
        },
    }
    value_manifest = {
        "schema_version": "moatrader-evidence-index-value-controls-v2/1",
        "status": "V2_VALUE_CONTROLS_PREPARED_AFTER_ERI_GATE",
        "value_input_sha256": sha256_file(value_path),
        "value_available_no_later_than_signal_verified": True,
        "future_eri_used_to_construct_value_controls": False,
        "metric_fields": list(VALUE_METRIC_FIELDS_V2),
        "metric_orientation": {field: "HIGHER_IS_CHEAPER" for field in VALUE_METRIC_FIELDS_V2},
        "neutralizer_priority": NEUTRALIZER_PRIORITY_V2,
        "per_pbr_joint_primary": False,
        "per_pbr_primary_ranking": False,
        "ranking_policy": "NO_VALUE_BASED_RANKING",
        **common,
    }
    factor_manifest = {
        "schema_version": "moatrader-evidence-index-factor-controls-v2/1",
        "status": "V2_FACTOR_CONTROLS_PREPARED_AFTER_ERI_GATE",
        "factor_input_sha256": sha256_file(factor_path),
        "factor_available_no_later_than_signal_verified": True,
        "future_eri_used_to_construct_factor_controls": False,
        "control_fields": list(FACTOR_CONTROL_FIELDS_V2),
        "control_definitions": FACTOR_DEFINITIONS_V2,
        "analyst_pit_policy": (
            "USE_MAX_REGISTER_UPDATE_TIMESTAMP_NO_LATER_THAN_SIGNAL; "
            "LATEST_PER_BROKER; 180D LOOKBACK; MINIMUM_TWO_BROKERS"
        ),
        "ranking_policy": "NO_FACTOR_BASED_RANKING",
        **common,
    }
    value_manifest_path = output / "value-controls-manifest.json"
    factor_manifest_path = output / "factor-controls-manifest.json"
    _write_json(value_manifest_path, value_manifest)
    _write_json(factor_manifest_path, factor_manifest)
    status = {
        "schema_version": "moatrader-neutral-control-preparation-stage-v2/1",
        "status": "V2_VALUE_AND_FACTOR_CONTROLS_PREPARED",
        "observation_count": len(label_ids),
        "coverage_counts": dict(sorted(coverage.items())),
        "analyst_source_counts": dict(sorted(analyst_counts.items())),
        "value_input_sha256": sha256_file(value_path),
        "factor_input_sha256": sha256_file(factor_path),
        "value_manifest_sha256": sha256_file(value_manifest_path),
        "factor_manifest_sha256": sha256_file(factor_manifest_path),
        "return_data_opened": False,
        "source_files_modified": False,
        "future_eri_used_as_ranking": False,
        "per_pbr_primary_ranking": False,
    }
    _write_json(output / "stage-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare PIT Value, Momentum, Growth, Quality, and analyst EPS controls."
    )
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--eri-build", type=Path, required=True)
    parser.add_argument("--pre-outcome-build", type=Path, required=True)
    parser.add_argument("--arcana-snapshot-root", type=Path, required=True)
    parser.add_argument("--analyst-root", type=Path, required=True)
    parser.add_argument("--marcap-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_neutral_controls_v2(**vars(args))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
