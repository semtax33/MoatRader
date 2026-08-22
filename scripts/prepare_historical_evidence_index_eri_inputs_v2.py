from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
from pydantic import Field, model_validator

from moatrader.backtest.universe_corrected import (
    FINANCE_HINT_RE,
    HOLDING_HINT_RE,
    assign_size_bucket,
    classify_security,
)
from moatrader.canonical.models import ContractModel
from moatrader.expectations.driver_signals import (
    ImpliedSolutionStatus,
    WACC_BY_SIZE,
    implied_driver_solution,
    supported_driver_estimate,
)
from moatrader.expectations.future_eri import (
    CurrentExpectationStateV1,
    EvidenceIndexFeatureDatasetSealV2,
    FutureEriOutcomeInputV1,
    RealizedFcffStateV1,
    target_trading_session,
)
from moatrader.expectations.historical_evidence import (
    canonical_payload_sha256,
    sha256_file,
)
from moatrader.expectations.historical_evidence_v2 import (
    DeterministicCoreIndexRowV2,
    FullEvidenceIndexRowV2,
)
from moatrader.expectations.revision import (
    assumptions_with_driver,
    driver_sensitivities,
    turbo_driver,
)
from moatrader.valuation.assumptions import EconomicDcfAssumptions
from scripts.audit_historical_future_eri_outcome_eligibility import (
    OutcomeEligibilityInventoryRowV1,
)
from scripts.prepare_historical_deterministic_pit_inputs_v2 import (
    FilingSource,
    FilingTask,
    _decoded_source,
    _label,
    _parse_tree,
    _row_values,
    _tag,
    _text,
    _UNIT_MULTIPLIER,
    _UNIT_RE,
    _read_universe,
)


D = Decimal
SEOUL = ZoneInfo("Asia/Seoul")
TAX_RATE = D("0.24")
_EQUITY_LABELS = {"자본총계", "총자본"}
_CASH_LABELS = {"현금및현금성자산", "현금및현금성자산및단기금융상품"}
_DEBT_LABELS = {
    "단기차입금",
    "유동성장기차입금",
    "유동성장기부채",
    "유동성사채",
    "장기차입금",
    "비유동차입금",
    "사채",
    "유동리스부채",
    "비유동리스부채",
}
_REVENUE_LABELS = {"수익매출액", "매출액", "매출액계", "매출", "영업수익", "수익"}
_OPERATING_PROFIT_LABELS = {"영업이익", "영업이익손실", "영업손실", "영업손익"}
_ASSET_LABELS = {"자산총계"}


class PITEconomicAnnualSnapshotV2(ContractModel):
    schema_version: str = "moatrader-pit-economic-annual-snapshot-v2/1"
    issuer_id: str = Field(pattern=r"^[0-9]{6}$")
    rcept_no: str = Field(min_length=14)
    fiscal_year: int = Field(ge=1900, le=2200)
    fiscal_period_end: date
    available_at: datetime
    revenue: Decimal | None = None
    operating_profit: Decimal | None = None
    total_assets: Decimal | None = None
    total_equity: Decimal | None = None
    cash: Decimal | None = None
    debt: Decimal | None = None
    metric_source_ids: dict[str, list[str]] = Field(default_factory=dict)
    verified_source_hashes: dict[str, str] = Field(default_factory=dict)
    extraction_errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def point_in_time_snapshot(self) -> "PITEconomicAnnualSnapshotV2":
        if self.available_at.tzinfo is None or self.available_at.utcoffset() is None:
            raise ValueError("annual PIT snapshot available_at must be timezone-aware")
        if self.fiscal_period_end.month != 12 or self.fiscal_period_end.year != self.fiscal_year:
            raise ValueError("annual PIT snapshot requires a December fiscal period")
        for name in ("revenue", "total_assets", "total_equity", "cash", "debt"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")
        return self


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


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _read_jsonl(path: Path, model: type[ContractModel]) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def _git_state(workspace: Path) -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return commit, dirty


def _git_blob_sha256(workspace: Path, *, commit: str, repository_path: str) -> str:
    content = subprocess.run(
        ["git", "show", f"{commit}:{repository_path}"],
        cwd=workspace,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(content).hexdigest()


def _current_value(values: list[Decimal]) -> Decimal | None:
    return values[0] if values else None


def extract_pit_economic_metrics_from_html(document: str) -> dict[str, Decimal | None]:
    """Extract annual FCFF bridge fields from one immutable regular filing."""

    root = _parse_tree(document)
    current_unit = D(1)
    income_candidates: list[tuple[int, Decimal, Decimal]] = []
    balance_candidates: list[
        tuple[int, Decimal | None, Decimal | None, Decimal | None, Decimal | None]
    ] = []
    for order, element in enumerate(root.iter()):
        tag = _tag(element)
        if tag in {"p", "span", "td", "th"}:
            match = _UNIT_RE.search(_text(element))
            if match:
                current_unit = _UNIT_MULTIPLIER[match.group(1)]
        if tag != "table":
            continue
        table_unit = current_unit
        match = _UNIT_RE.search(_text(element))
        if match:
            table_unit = _UNIT_MULTIPLIER[match.group(1)]
        rows = _row_values(element)
        if not rows:
            continue
        row_index = {_label(name): values for name, values in rows}
        revenue = next(
            (_current_value(values) for name, values in row_index.items() if name in _REVENUE_LABELS),
            None,
        )
        operating_profit = next(
            (
                _current_value(values)
                for name, values in row_index.items()
                if name in _OPERATING_PROFIT_LABELS
            ),
            None,
        )
        if revenue is not None and operating_profit is not None:
            income_candidates.append(
                (order, revenue * table_unit, operating_profit * table_unit)
            )
        assets = next(
            (_current_value(values) for name, values in row_index.items() if name in _ASSET_LABELS),
            None,
        )
        equity = next(
            (_current_value(values) for name, values in row_index.items() if name in _EQUITY_LABELS),
            None,
        )
        if assets is None or equity is None:
            continue
        cash = next(
            (_current_value(values) for name, values in row_index.items() if name in _CASH_LABELS),
            None,
        )
        debt_values = [
            value
            for name, values in row_index.items()
            if name in _DEBT_LABELS
            for value in [_current_value(values)]
            if value is not None
        ]
        balance_candidates.append(
            (
                order,
                assets * table_unit,
                equity * table_unit,
                cash * table_unit if cash is not None else None,
                sum(debt_values, D(0)) * table_unit if debt_values else None,
            )
        )
    income = min(income_candidates, default=None, key=lambda item: item[0])
    balance = min(balance_candidates, default=None, key=lambda item: item[0])
    return {
        "revenue": income[1] if income else None,
        "operating_profit": income[2] if income else None,
        "total_assets": balance[1] if balance else None,
        "total_equity": balance[2] if balance else None,
        "cash": balance[3] if balance else None,
        "debt": balance[4] if balance else None,
    }


def _annual_sources(task: FilingTask) -> tuple[FilingSource, ...]:
    return tuple(
        source
        for source in (
            task.finance_statement,
            task.moatrader_original,
            task.finance_comment,
            task.business_info,
        )
        if source is not None
    )


def _extract_annual_task(task: FilingTask) -> dict[str, Any]:
    metrics: dict[str, Decimal | None] = {
        "revenue": None,
        "operating_profit": None,
        "total_assets": None,
        "total_equity": None,
        "cash": None,
        "debt": None,
    }
    metric_source_ids: dict[str, list[str]] = {}
    verified: dict[str, str] = {}
    errors: list[str] = []
    for source in _annual_sources(task):
        try:
            decoded, raw_hash = _decoded_source(source)
            verified[source.path] = raw_hash
            extracted = extract_pit_economic_metrics_from_html(decoded)
        except Exception as exc:  # every failure is retained in the private audit row
            errors.append(f"{source.origin}:{type(exc).__name__}:{exc}")
            continue
        for name in metrics:
            value = extracted[name]
            if metrics[name] is None and value is not None:
                if name != "operating_profit" and value < 0:
                    errors.append(f"{source.origin}:NEGATIVE_ACCOUNT_VALUE_REJECTED:{name}")
                    continue
                metrics[name] = value
                metric_source_ids[name] = [source.source_id]
    period_end = date.fromisoformat(task.fiscal_period_end)
    snapshot = PITEconomicAnnualSnapshotV2(
        issuer_id=task.ticker,
        rcept_no=task.rcept_no,
        fiscal_year=period_end.year,
        fiscal_period_end=period_end,
        available_at=datetime.fromisoformat(task.available_at),
        metric_source_ids=metric_source_ids,
        verified_source_hashes=verified,
        extraction_errors=errors,
        **metrics,
    )
    return snapshot.model_dump(mode="json")


def _load_common_index_rows(
    full_index_build: Path, core_index_build: Path
) -> tuple[dict[str, FullEvidenceIndexRowV2], dict[str, DeterministicCoreIndexRowV2]]:
    full = {
        row.observation_id: row
        for row in _read_jsonl(
            full_index_build / "full-evidence-index-eligible-nobs2.jsonl",
            FullEvidenceIndexRowV2,
        )
    }
    core = {
        row.observation_id: row
        for row in _read_jsonl(
            core_index_build / "deterministic-core-index-eligible-nobs2.jsonl",
            DeterministicCoreIndexRowV2,
        )
    }
    common = set(full) & set(core)
    return ({key: full[key] for key in common}, {key: core[key] for key in common})


def _parquet_source_map(files: Sequence[Path]) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for path in files:
        year = int(path.stem.rsplit("-", 1)[-1])
        result[year] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    return result


def _load_sessions(files: Sequence[Path]) -> list[date]:
    values: set[date] = set()
    for path in files:
        frame = pd.read_parquet(path, columns=["Date", "MarketId"])
        dates = pd.to_datetime(frame.loc[frame["MarketId"].isin(["STK", "KSQ"]), "Date"])
        values.update(item.date() for item in dates)
    return sorted(values)


def _filtered_parquet(
    files: Sequence[Path], *, dates: set[date], columns: Sequence[str]
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    by_year: dict[int, list[pd.Timestamp]] = defaultdict(list)
    for value in sorted(dates):
        by_year[value.year].append(pd.Timestamp(value))
    for path in files:
        year = int(path.stem.rsplit("-", 1)[-1])
        selected = by_year.get(year)
        if not selected:
            continue
        frame = pd.read_parquet(
            path,
            columns=list(columns),
            filters=[("Date", "in", selected)],
        )
        if frame.empty:
            continue
        frame["Date"] = pd.to_datetime(frame["Date"]).dt.date
        frame["Code"] = frame["Code"].astype(str).str.zfill(6)
        pieces.append(frame)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=columns)


def _row_map(frame: pd.DataFrame) -> dict[tuple[date, str], dict[str, Any]]:
    result: dict[tuple[date, str], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = (row["Date"], str(row["Code"]).zfill(6))
        result.setdefault(key, row)
    return result


def _size_maps(frame: pd.DataFrame) -> dict[date, dict[str, str]]:
    result: dict[date, dict[str, str]] = {}
    if frame.empty or "Marcap" not in frame:
        return result
    for value, group in frame.groupby("Date", sort=True):
        valid = group[
            group["MarketId"].isin(["STK", "KSQ"])
            & pd.to_numeric(group["Marcap"], errors="coerce").gt(0)
        ].copy()
        if len(valid) < 3:
            continue
        valid["market_cap"] = pd.to_numeric(valid["Marcap"], errors="coerce")
        valid = assign_size_bucket(valid)
        result[value] = {
            str(row.Code).zfill(6): str(row.size_bucket)
            for row in valid[["Code", "size_bucket"]].itertuples(index=False)
        }
    return result


def _timeline(
    snapshots: Sequence[PITEconomicAnnualSnapshotV2], ticker: str, cutoff: datetime
) -> list[PITEconomicAnnualSnapshotV2]:
    visible = [
        item
        for item in snapshots
        if item.issuer_id == ticker
        and item.available_at <= cutoff
        and item.fiscal_period_end <= cutoff.date()
    ]
    latest: dict[int, PITEconomicAnnualSnapshotV2] = {}
    for item in sorted(visible, key=lambda row: (row.available_at, row.rcept_no)):
        latest[item.fiscal_year] = item
    return [latest[year] for year in sorted(latest)]


def _valid_financial(item: PITEconomicAnnualSnapshotV2) -> bool:
    return bool(
        item.revenue is not None
        and item.revenue > 0
        and item.operating_profit is not None
        and item.total_equity is not None
        and item.total_equity > 0
        and item.cash is not None
        and item.debt is not None
        and item.total_equity + item.debt - item.cash > 0
    )


def _history_payload(
    rows: Sequence[PITEconomicAnnualSnapshotV2],
) -> list[tuple[int, dict[str, Decimal | None]]]:
    return [
        (
            item.fiscal_year,
            {
                "revenue": item.revenue,
                "ebit": item.operating_profit,
                "cash": item.cash,
                "debt": item.debt,
                "total_equity": item.total_equity,
            },
        )
        for item in rows
    ]


def _market_source_id(
    sources: dict[int, dict[str, str]], value: date, ticker: str, field: str
) -> str:
    source = sources[value.year]
    return f"MARCAP:{source['sha256']}:{value.isoformat()}:{ticker}:{field}"


def _exact_assumption_sources(
    assumptions: EconomicDcfAssumptions,
    history: Sequence[PITEconomicAnnualSnapshotV2],
    *,
    share_source_id: str,
    size_bucket: str,
) -> EconomicDcfAssumptions:
    latest = history[-1]
    annual_ids = sorted(
        {
            source_id
            for item in history
            for source_ids in item.metric_source_ids.values()
            for source_id in source_ids
        }
    )
    latest_ids = sorted(
        {source_id for source_ids in latest.metric_source_ids.values() for source_id in source_ids}
    )
    sources = dict(assumptions.assumption_sources)
    for field in ("base_revenue", "base_nopat_margin", "base_invested_capital", "net_debt"):
        sources[field] = latest_ids
    for field in ("revenue_growth", "target_nopat_margin", "roiic", "competitive_advantage_period_years"):
        sources[field] = annual_ids
    sources["diluted_shares"] = [share_source_id]
    sources["wacc"] = [f"FROZEN_WACC_POLICY:MARKET_CAP_TERCILE:{size_bucket}"]
    return assumptions.model_copy(update={"assumption_sources": sources})


def prepare_pre_outcome_inputs(
    *,
    workspace: Path,
    full_index_build: Path,
    core_index_build: Path,
    filing_pair_input: Path,
    pair_source_map_input: Path,
    marcap_files: Sequence[Path],
    marcap_commit: str,
    output: Path,
    workers: int = 16,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production ERI pre-outcome input preparation requires a clean worktree")
    full_stage = _read_json(full_index_build / "stage-status.json")
    full_seal_path = full_index_build / "full-evidence-index-seal.json"
    core_manifest_path = core_index_build / "pre-outcome-index-manifest.json"
    if not (
        full_stage.get("status") == "V2_FULL_EVIDENCE_INDEX_SEALED_OUTCOMES_CLOSED"
        and full_stage.get("outcome_stage_authorized") is True
        and full_stage.get("outcome_vault_opened") is False
        and full_stage.get("return_data_opened") is False
        and full_stage.get("value_data_opened") is False
        and full_stage.get("full_evidence_index_seal_sha256") == sha256_file(full_seal_path)
    ):
        raise ValueError("Full Index is not sealed with outcomes closed")
    if not marcap_files or workers < 1:
        raise ValueError("marcap files and positive workers are required")
    full_rows, core_rows = _load_common_index_rows(full_index_build, core_index_build)
    common_ids = sorted(set(full_rows) & set(core_rows))
    if not common_ids:
        raise ValueError("Full/Core eligible panels do not overlap")

    _pairs, tasks = _read_universe(filing_pair_input, pair_source_map_input)
    annual_tasks = [
        task for task in tasks if date.fromisoformat(task.fiscal_period_end).month == 12
    ]
    if workers == 1:
        raw_snapshots = [_extract_annual_task(task) for task in annual_tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            raw_snapshots = list(executor.map(_extract_annual_task, annual_tasks, chunksize=8))
    snapshots = [PITEconomicAnnualSnapshotV2.model_validate(row) for row in raw_snapshots]
    snapshots_by_ticker: dict[str, list[PITEconomicAnnualSnapshotV2]] = defaultdict(list)
    for item in snapshots:
        snapshots_by_ticker[item.issuer_id].append(item)

    source_hash_mismatches: list[str] = []
    verified_paths: dict[str, str] = {}
    for item in snapshots:
        verified_paths.update(item.verified_source_hashes)
    for raw_path, expected in verified_paths.items():
        if sha256_file(Path(raw_path)) != expected:
            source_hash_mismatches.append(raw_path)
    if source_hash_mismatches:
        raise ValueError(f"original filing sources changed during ERI extraction: {source_hash_mismatches[:5]}")

    sessions = _load_sessions(marcap_files)
    signal_dates = {full_rows[key].signal_timestamp.date() for key in common_ids}
    targets_by_id: dict[str, date] = {}
    for key in common_ids:
        try:
            targets_by_id[key] = target_trading_session(
                full_rows[key].signal_timestamp.date(), sessions, horizon=63
            )
        except ValueError:
            continue
    target_dates = set(targets_by_id.values())
    signal_frame = _filtered_parquet(
        marcap_files,
        dates=signal_dates,
        columns=["Date", "Code", "Name", "Open", "Stocks", "Marcap", "MarketId"],
    )
    target_metadata = _filtered_parquet(
        marcap_files,
        dates=target_dates,
        columns=["Date", "Code"],
    )
    signal_prices = _row_map(signal_frame)
    target_presence = set(_row_map(target_metadata))
    signal_sizes = _size_maps(signal_frame)
    marcap_sources = _parquet_source_map(marcap_files)

    expectations: list[dict[str, Any]] = []
    expectation_exclusions: list[dict[str, Any]] = []
    expectation_reason_counts: Counter[str] = Counter()
    for observation_id in common_ids:
        full = full_rows[observation_id]
        ticker = full.issuer_id
        signal_date = full.signal_timestamp.date()
        reasons: list[str] = []
        point = signal_prices.get((signal_date, ticker))
        if point is None:
            reasons.append("NO_EXACT_SIGNAL_OPEN_PRICE")
        else:
            name = str(point.get("Name") or "")
            if classify_security(name) != "COMMON":
                reasons.append("NON_COMMON_SECURITY")
            if FINANCE_HINT_RE.search(name) or HOLDING_HINT_RE.search(name):
                reasons.append("FCFF_INCOMPARABLE_ARCHETYPE")
        size_bucket = signal_sizes.get(signal_date, {}).get(ticker)
        if size_bucket is None:
            reasons.append("NO_SIGNAL_SIZE_BUCKET")
        history = [
            item
            for item in _timeline(
                snapshots_by_ticker.get(ticker, []), ticker, full.signal_timestamp
            )
            if _valid_financial(item)
        ]
        if len(history) < 2:
            reasons.append("FEWER_THAN_TWO_VALID_PIT_ANNUALS")
        try:
            open_price = D(str(point["Open"])) if point is not None else D(0)
            shares = D(str(point["Stocks"])) if point is not None else D(0)
            if open_price <= 0:
                reasons.append("NON_POSITIVE_SIGNAL_OPEN")
            if shares <= 0:
                reasons.append("NON_POSITIVE_SIGNAL_LISTED_SHARES")
        except Exception:
            open_price = D(0)
            shares = D(0)
            reasons.append("INVALID_SIGNAL_MARKET_INPUT")
        if reasons:
            for reason in set(reasons):
                expectation_reason_counts[reason] += 1
            expectation_exclusions.append(
                {"observation_id": observation_id, "reasons": sorted(set(reasons))}
            )
            continue
        assert point is not None and size_bucket is not None
        try:
            estimate = supported_driver_estimate(
                _history_payload(history),
                size_bucket=size_bucket,
                diluted_shares=shares,
            )
            base = estimate.assumptions()
            selected_driver = turbo_driver(driver_sensitivities(base))
            if selected_driver is None:
                raise ValueError("NO_POSITIVE_TURBO_DRIVER")
            solution = implied_driver_solution(
                base=base,
                current_price=open_price,
                driver=selected_driver,
            )
            if solution.status != ImpliedSolutionStatus.SOLVED or solution.implied is None:
                raise ValueError(f"REVERSE_DCF_{solution.status.value}")
            frozen = assumptions_with_driver(base, selected_driver, solution.implied)
            share_source_id = _market_source_id(
                marcap_sources, signal_date, ticker, "OPEN_AND_LISTED_SHARES"
            )
            frozen = _exact_assumption_sources(
                frozen,
                history,
                share_source_id=share_source_id,
                size_bucket=size_bucket,
            )
            reverse_payload = {
                "method": "TURBO_DRIVER_ONE_DIMENSIONAL_REVERSE_DCF_V2",
                "observation_id": observation_id,
                "market_price": str(open_price),
                "market_price_source_id": share_source_id,
                "supported_assumptions": base.model_dump(mode="json"),
                "selected_driver": selected_driver.value,
                "solution": solution.model_dump(mode="json"),
                "annual_source_ids": sorted(
                    {
                        source_id
                        for item in history
                        for source_ids in item.metric_source_ids.values()
                        for source_id in source_ids
                    }
                ),
            }
            state = CurrentExpectationStateV1(
                issuer_id=ticker,
                signal_timestamp=full.signal_timestamp,
                market_price=open_price,
                market_price_at=full.signal_timestamp,
                market_price_source_id=share_source_id,
                implied_growth=frozen.revenue_growth,
                implied_margin=frozen.target_nopat_margin,
                implied_roiic=frozen.roiic,
                implied_cap_years=D(frozen.competitive_advantage_period_years),
                reverse_dcf_method="TURBO_DRIVER_ONE_DIMENSIONAL_REVERSE_DCF_V2",
                reverse_dcf_input_sha256=canonical_payload_sha256(reverse_payload),
            )
            expectations.append(
                {
                    "observation_id": observation_id,
                    "expectation_state": state.model_dump(mode="json"),
                    "frozen_expectation_assumptions": frozen.model_dump(mode="json"),
                    "reverse_dcf_provenance": reverse_payload,
                }
            )
        except Exception as exc:
            reason = f"REVERSE_DCF_ERROR:{type(exc).__name__}:{exc}"
            expectation_reason_counts[reason] += 1
            expectation_exclusions.append(
                {"observation_id": observation_id, "reasons": [reason]}
            )

    inventory_rows: list[OutcomeEligibilityInventoryRowV1] = []
    inventory_counts: Counter[str] = Counter()
    for observation_id in common_ids:
        full = full_rows[observation_id]
        ticker = full.issuer_id
        target = targets_by_id.get(observation_id)
        if target is None:
            inventory_counts["NO_EXACT_T63_SESSION"] += 1
            continue
        price_present = (target, ticker) in target_presence
        target_price_at = datetime.combine(target, time(15, 30), tzinfo=SEOUL) if price_present else None
        financials = [
            item
            for item in _timeline(
                snapshots_by_ticker.get(ticker, []),
                ticker,
                target_price_at or datetime.combine(target, time.max, tzinfo=SEOUL),
            )
            if _valid_financial(item)
        ]
        realized = financials[-1] if financials else None
        target_source_id = (
            _market_source_id(marcap_sources, target, ticker, "CLOSE") if price_present else None
        )
        share_source_id = (
            _market_source_id(marcap_sources, target, ticker, "LISTED_SHARES")
            if price_present
            else None
        )
        source_ids = (
            sorted(
                {
                    source_id
                    for values in realized.metric_source_ids.values()
                    for source_id in values
                }
            )
            if realized is not None
            else []
        )
        row = OutcomeEligibilityInventoryRowV1(
            observation_id=observation_id,
            target_session=target,
            target_price_at=target_price_at,
            target_price_source_id=target_source_id,
            realized_financials_available_at=(realized.available_at if realized else None),
            realized_financial_source_ids=source_ids,
            net_debt_source_id=(source_ids[0] if realized and source_ids else None),
            diluted_shares_source_id=share_source_id,
            wacc_source_id=(
                "FROZEN_WACC_POLICY:TARGET_MARKET_CAP_TERCILE" if price_present else None
            ),
        )
        inventory_rows.append(row)
        inventory_counts["TARGET_PRICE_METADATA"] += int(price_present)
        inventory_counts["TARGET_PIT_FINANCIALS"] += int(realized is not None)
        inventory_counts["COMPLETE_METADATA"] += int(
            bool(
                target_source_id
                and share_source_id
                and realized is not None
                and source_ids
            )
        )

    output.mkdir(parents=True, exist_ok=True)
    snapshot_path = output / "private-pit-annual-financial-snapshots.jsonl"
    expectation_path = output / "expectations-pre-outcome.jsonl"
    inventory_path = output / "outcome-eligibility-inventory.jsonl"
    sessions_path = output / "trading-sessions.json"
    exclusion_path = output / "expectation-exclusions.json"
    _write_jsonl(snapshot_path, sorted(snapshots, key=lambda row: (row.issuer_id, row.available_at)))
    _write_jsonl(expectation_path, sorted(expectations, key=lambda row: row["observation_id"]))
    _write_jsonl(inventory_path, sorted(inventory_rows, key=lambda row: row.observation_id))
    _write_json(sessions_path, [item.isoformat() for item in sessions])
    _write_json(exclusion_path, expectation_exclusions)
    expectation_ids = {str(item["observation_id"]) for item in expectations}
    complete_inventory_ids = {
        item.observation_id
        for item in inventory_rows
        if item.target_price_at
        and item.target_price_source_id
        and item.realized_financials_available_at
        and item.realized_financial_source_ids
        and item.net_debt_source_id
        and item.diluted_shares_source_id
        and item.wacc_source_id
    }
    artifacts = {
        "annual_snapshots": sha256_file(snapshot_path),
        "expectations_pre_outcome": sha256_file(expectation_path),
        "outcome_eligibility_inventory": sha256_file(inventory_path),
        "trading_sessions": sha256_file(sessions_path),
        "expectation_exclusions": sha256_file(exclusion_path),
    }
    seal = {
        "schema_version": "moatrader-eri-pre-outcome-input-seal-v2/1",
        "status": "ERI_PRE_OUTCOME_INPUTS_PREPARED_OUTCOMES_CLOSED",
        "git_commit": commit,
        "worktree_dirty": False,
        "script_sha256": sha256_file(Path(__file__)),
        "full_index_seal_sha256": sha256_file(full_seal_path),
        "core_pre_outcome_manifest_sha256": sha256_file(core_manifest_path),
        "filing_pair_sha256": sha256_file(filing_pair_input),
        "pair_source_map_sha256": sha256_file(pair_source_map_input),
        "marcap_provider_commit": marcap_commit,
        "marcap_sources": marcap_sources,
        "common_full_core_index_count": len(common_ids),
        "annual_snapshot_count": len(snapshots),
        "expectation_count": len(expectations),
        "inventory_count": len(inventory_rows),
        "potential_label_eligible_count": len(expectation_ids & complete_inventory_ids),
        "expectation_exclusion_reason_counts": dict(sorted(expectation_reason_counts.items())),
        "inventory_counts": dict(sorted(inventory_counts.items())),
        "reverse_dcf_method": "TURBO_DRIVER_ONE_DIMENSIONAL_REVERSE_DCF_V2",
        "exact_horizon_trading_sessions": 63,
        "original_source_files_modified": False,
        "source_hash_mismatch_count": 0,
        "verified_original_source_count": len(verified_paths),
        "artifact_hashes": artifacts,
        "outcome_stage_authorized": bool(expectation_ids & complete_inventory_ids),
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    seal_path = output / "pre-outcome-input-seal.json"
    _write_json(seal_path, seal)
    status = {**seal, "pre_outcome_input_seal_sha256": sha256_file(seal_path)}
    _write_json(output / "stage-status.json", status)
    return status


def materialize_outcome_vault(
    *,
    workspace: Path,
    pre_outcome_build: Path,
    feature_panel_build: Path,
    marcap_files: Sequence[Path],
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    commit, dirty = _git_state(workspace)
    if dirty:
        raise ValueError("production ERI outcome materialization requires a clean worktree")
    stage = _read_json(pre_outcome_build / "stage-status.json")
    seal_path = pre_outcome_build / "pre-outcome-input-seal.json"
    if not (
        stage.get("status") == "ERI_PRE_OUTCOME_INPUTS_PREPARED_OUTCOMES_CLOSED"
        and stage.get("outcome_stage_authorized") is True
        and stage.get("outcome_vault_opened") is False
        and stage.get("return_data_opened") is False
        and stage.get("value_data_opened") is False
        and stage.get("pre_outcome_input_seal_sha256") == sha256_file(seal_path)
    ):
        raise ValueError("pre-outcome ERI input seal is missing or unauthorized")
    script_hash = sha256_file(Path(__file__))
    sealed_commit = str(stage.get("git_commit") or "")
    if stage.get("script_sha256") != _git_blob_sha256(
        workspace,
        commit=sealed_commit,
        repository_path="scripts/prepare_historical_evidence_index_eri_inputs_v2.py",
    ):
        raise ValueError("sealed ERI pre-outcome builder does not match its recorded git blob")
    feature_stage_path = feature_panel_build / "stage-status.json"
    feature_seal_path = feature_panel_build / "feature-seal-pre-outcome.json"
    feature_path = feature_panel_build / "features-with-frozen-expectations-pre-outcome.jsonl"
    feature_stage = _read_json(feature_stage_path)
    feature_seal = EvidenceIndexFeatureDatasetSealV2.model_validate(
        _read_json(feature_seal_path)
    )
    if not (
        feature_stage.get("status") == "ERI_FEATURE_PANEL_SEALED_OUTCOMES_CLOSED"
        and feature_stage.get("feature_panel_sealed") is True
        and feature_stage.get("outcome_stage_authorized") is True
        and feature_stage.get("outcome_vault_opened") is False
        and feature_stage.get("return_data_opened") is False
        and feature_stage.get("value_data_opened") is False
        and feature_stage.get("pre_outcome_input_seal_sha256") == sha256_file(seal_path)
        and feature_stage.get("feature_seal_sha256") == sha256_file(feature_seal_path)
        and feature_stage.get("feature_artifact_sha256") == sha256_file(feature_path)
        and feature_stage.get("feature_dataset_sha256") == feature_seal.feature_dataset_sha256
        and feature_stage.get("script_sha256")
        == _git_blob_sha256(
            workspace,
            commit=str(feature_stage.get("git_commit") or ""),
            repository_path="scripts/seal_historical_evidence_index_eri_feature_panel_v2.py",
        )
    ):
        raise ValueError("ERI feature panel is not sealed with outcomes closed")
    feature_ids = set(feature_seal.observation_ids)
    snapshot_path = pre_outcome_build / "private-pit-annual-financial-snapshots.jsonl"
    inventory_path = pre_outcome_build / "outcome-eligibility-inventory.jsonl"
    if stage.get("artifact_hashes", {}).get("annual_snapshots") != sha256_file(snapshot_path):
        raise ValueError("annual PIT financial snapshots changed after seal")
    if stage.get("artifact_hashes", {}).get("outcome_eligibility_inventory") != sha256_file(
        inventory_path
    ):
        raise ValueError("outcome eligibility inventory changed after seal")
    snapshots = _read_jsonl(snapshot_path, PITEconomicAnnualSnapshotV2)
    snapshots_by_ticker: dict[str, list[PITEconomicAnnualSnapshotV2]] = defaultdict(list)
    for item in snapshots:
        snapshots_by_ticker[item.issuer_id].append(item)
    inventory = _read_jsonl(inventory_path, OutcomeEligibilityInventoryRowV1)
    target_dates = {item.target_session for item in inventory if item.target_price_at is not None}
    target_frame = _filtered_parquet(
        marcap_files,
        dates=target_dates,
        columns=["Date", "Code", "Name", "Close", "Stocks", "Marcap", "MarketId"],
    )
    target_rows = _row_map(target_frame)
    target_sizes = _size_maps(target_frame)
    marcap_sources = _parquet_source_map(marcap_files)
    outcomes: list[FutureEriOutcomeInputV1] = []
    exclusion_counts: Counter[str] = Counter()
    for item in inventory:
        if item.observation_id not in feature_ids:
            continue
        if not (
            item.target_price_at
            and item.target_price_source_id
            and item.realized_financials_available_at
            and item.realized_financial_source_ids
            and item.net_debt_source_id
            and item.diluted_shares_source_id
            and item.wacc_source_id
        ):
            exclusion_counts["INCOMPLETE_ELIGIBILITY_METADATA"] += 1
            continue
        ticker = item.target_price_source_id.split(":")[-2]
        point = target_rows.get((item.target_session, ticker))
        if point is None:
            exclusion_counts["MISSING_TARGET_MARKET_ROW"] += 1
            continue
        size_bucket = target_sizes.get(item.target_session, {}).get(ticker)
        if size_bucket is None:
            exclusion_counts["MISSING_TARGET_SIZE_BUCKET"] += 1
            continue
        timeline = [
            row
            for row in _timeline(
                snapshots_by_ticker.get(ticker, []), ticker, item.target_price_at
            )
            if _valid_financial(row)
        ]
        if not timeline:
            exclusion_counts["MISSING_REALIZED_PIT_ANNUAL"] += 1
            continue
        realized = timeline[-1]
        realized_source_ids = sorted(
            {
                source_id
                for values in realized.metric_source_ids.values()
                for source_id in values
            }
        )
        if (
            realized.available_at != item.realized_financials_available_at
            or realized_source_ids != item.realized_financial_source_ids
        ):
            exclusion_counts["REALIZED_FINANCIAL_METADATA_MISMATCH"] += 1
            continue
        close = D(str(point["Close"]))
        shares = D(str(point["Stocks"]))
        if close <= 0 or shares <= 0:
            exclusion_counts["NON_POSITIVE_TARGET_MARKET_INPUT"] += 1
            continue
        assert realized.revenue is not None
        assert realized.operating_profit is not None
        assert realized.total_equity is not None
        assert realized.cash is not None
        assert realized.debt is not None
        capital = realized.total_equity + realized.debt - realized.cash
        nopat_margin = realized.operating_profit * (D(1) - TAX_RATE) / realized.revenue
        if not D(-1) < nopat_margin < D(1):
            exclusion_counts["INVALID_REALIZED_NPAT_MARGIN"] += 1
            continue
        state = RealizedFcffStateV1(
            available_at=realized.available_at,
            base_period=f"{realized.fiscal_year}FY",
            base_revenue=realized.revenue,
            base_nopat_margin=nopat_margin,
            base_invested_capital=capital,
            net_debt=realized.debt - realized.cash,
            diluted_shares=shares,
            wacc=WACC_BY_SIZE[size_bucket],
            wacc_source_id=f"FROZEN_WACC_POLICY:MARKET_CAP_TERCILE:{size_bucket}",
            source_document_ids=sorted(
                {
                    source_id
                    for values in realized.metric_source_ids.values()
                    for source_id in values
                }
            ),
        )
        target_source_id = _market_source_id(
            marcap_sources, item.target_session, ticker, "CLOSE"
        )
        share_source_id = _market_source_id(
            marcap_sources, item.target_session, ticker, "LISTED_SHARES"
        )
        if (
            target_source_id != item.target_price_source_id
            or share_source_id != item.diluted_shares_source_id
        ):
            exclusion_counts["TARGET_MARKET_SOURCE_ID_MISMATCH"] += 1
            continue
        outcomes.append(
            FutureEriOutcomeInputV1(
                observation_id=item.observation_id,
                target_session=item.target_session,
                target_price_at=item.target_price_at,
                actual_market_price=close,
                target_price_source_id=target_source_id,
                realized_state=state,
            )
        )
    output.mkdir(parents=True, exist_ok=True)
    outcome_path = output / "future-eri-outcomes.jsonl"
    _write_jsonl(outcome_path, sorted(outcomes, key=lambda row: row.observation_id))
    status = {
        "schema_version": "moatrader-eri-outcome-vault-stage-v2/1",
        "status": "ERI_OUTCOME_VAULT_MATERIALIZED_AFTER_FEATURE_PANEL_SEAL",
        "git_commit": commit,
        "worktree_dirty": False,
        "script_sha256": script_hash,
        "pre_outcome_input_seal_sha256": sha256_file(seal_path),
        "feature_panel_stage_sha256": sha256_file(feature_stage_path),
        "feature_seal_sha256": sha256_file(feature_seal_path),
        "feature_dataset_sha256": feature_seal.feature_dataset_sha256,
        "full_index_seal_sha256": stage["full_index_seal_sha256"],
        "outcome_count": len(outcomes),
        "outcome_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "outcome_input_sha256": sha256_file(outcome_path),
        "outcome_vault_opened": True,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "stage-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare exact-source PIT reverse-DCF/eligibility inputs, then materialize "
            "the t+63 ERI outcome vault in a separately authorized stage."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser("pre-outcome")
    pre.add_argument("--workspace", type=Path, default=Path.cwd())
    pre.add_argument("--full-index-build", type=Path, required=True)
    pre.add_argument("--core-index-build", type=Path, required=True)
    pre.add_argument("--filing-pair-input", type=Path, required=True)
    pre.add_argument("--pair-source-map-input", type=Path, required=True)
    pre.add_argument("--marcap-files", type=Path, nargs="+", required=True)
    pre.add_argument("--marcap-commit", required=True)
    pre.add_argument("--output", type=Path, required=True)
    pre.add_argument("--workers", type=int, default=16)
    outcome = subparsers.add_parser("outcome")
    outcome.add_argument("--workspace", type=Path, default=Path.cwd())
    outcome.add_argument("--pre-outcome-build", type=Path, required=True)
    outcome.add_argument("--feature-panel-build", type=Path, required=True)
    outcome.add_argument("--marcap-files", type=Path, nargs="+", required=True)
    outcome.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    values = vars(args)
    command = values.pop("command")
    result = (
        prepare_pre_outcome_inputs(**values)
        if command == "pre-outcome"
        else materialize_outcome_vault(**values)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
