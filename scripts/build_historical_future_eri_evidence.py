from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import date, datetime
from itertools import groupby
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from moatrader.expectations.future_eri import EvidenceObservation, OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisPairClassification,
    HistoricalFilingPair,
    HistoricalRegularFiling,
    PairedAxisPacket,
    build_blinded_packets,
    build_regular_filing_pairs,
    build_source_integrity_manifest,
    discover_arcana_regular_sources,
    discover_moatrader_original_sources,
    merge_historical_sources,
    sha256_file,
    source_variant_axis_windows,
    validate_packet_anonymization,
    verify_source_integrity,
)


SEOUL = ZoneInfo("Asia/Seoul")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCANA_ROOT = PROJECT_ROOT.parent / "Arcana" / "data-lake"
DEFAULT_MOATRADER_DATA_LAKE = PROJECT_ROOT / "data-lake"
DEFAULT_CALENDAR_ROOT = (
    DEFAULT_MOATRADER_DATA_LAKE
    / "experiments"
    / "historical-validation-v7-2020-2025"
    / "prices"
    / "source"
)
DEFAULT_2026_CALENDAR = (
    DEFAULT_MOATRADER_DATA_LAKE
    / "backtests"
    / "kr-all-research-20260818-v1"
    / "inputs"
    / "marcap"
    / "source"
    / "marcap-2026.parquet"
)
DEFAULT_SECTOR_MAP = (
    DEFAULT_MOATRADER_DATA_LAKE
    / "experiments"
    / "expectation-gap-ablation-20260818-v1"
    / "source-map"
    / "sectors.csv"
)


FROZEN_FEATURE_CONTRACT_V1: dict[str, Any] = {
    "schema_version": "moatrader-historical-future-eri-feature-v1/1",
    "research_status": "PSEUDO_OOS_CALIBRATION_EVIDENCE",
    "live_shadow_start": "2026-08-20",
    "source_scope": {
        "included": [
            "Arcana data-lake raw business-info HTML",
            "MoatRader data-lake original OpenDART archives",
        ],
        "regular_reports_only": ["사업보고서", "반기보고서", "분기보고서"],
        "amendment_policy": "EXCLUDE_AND_REPORT_SEPARATELY",
        "mutation_policy": "SOURCE_FILES_READ_ONLY_SHA256_BEFORE_AND_AFTER",
    },
    "feature": {
        "axes": [item.value for item in OperatingEvidenceAxis],
        "state_space": [-1, 0, 1],
        "primary_score": "equal-weight sum of six comparable-period state deltas",
        "missing_axis_policy": "EXCLUDE_NO_IMPUTATION",
        "llm_role": "PAIRED_FACT_CLASSIFICATION_ONLY",
        "contamination_controls": [
            "issuer removed",
            "ticker removed",
            "receipt number removed",
            "dates removed",
            "future context excluded",
            "market data excluded",
        ],
        "required_grounding": "verbatim previous/current source spans",
    },
    "feature_bands": {
        "VERY_BEARISH": [-6, -3],
        "BEARISH": [-2, -1],
        "NEUTRAL": [0, 0],
        "BULLISH": [1, 2],
        "VERY_BULLISH": [3, 6],
    },
    "outcome_vault": "CLOSED_UNTIL_FEATURE_QUALITY_AND_SEAL_GATES_PASS",
    "returns": "PROHIBITED_AT_FEATURE_STAGE",
    "primary_ranking_policy": "NONE_MECHANISM_ONLY",
    "per_pbr_role": "NOT_USED_AT_FEATURE_OR_ERI_MECHANISM_STAGE",
}
FROZEN_FEATURE_CONTRACT_V1R: dict[str, Any] = {
    **FROZEN_FEATURE_CONTRACT_V1,
    "schema_version": "moatrader-historical-future-eri-feature-v1r/1",
    "research_status": "SOURCE_CORRECTED_V1_REPLICATION_PREOUTCOME",
    "replication_of": "future-eri-v1-preoutcome",
    "source_change_only": True,
    "source_scope": {
        **FROZEN_FEATURE_CONTRACT_V1["source_scope"],
        "included": [
            "Arcana data-lake raw business-info HTML",
            "Arcana data-lake raw finance-comment HTML",
            "Arcana data-lake raw finance-statement HTML",
            "MoatRader data-lake original OpenDART archives",
        ],
    },
    "intended_freeze_tag": "future-eri-v1r-three-section-preoutcome",
}
# Backward-compatible import name. New builds default to V1R explicitly below.
FROZEN_FEATURE_CONTRACT = FROZEN_FEATURE_CONTRACT_V1R


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def _git_optional(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def frozen_manifest(
    *,
    created_at: datetime,
    research_variant: Literal["V1", "V1R"],
    contract: dict[str, Any],
) -> dict[str, Any]:
    tag = (
        "future-eri-v1-preoutcome"
        if research_variant == "V1"
        else "future-eri-v1r-three-section-preoutcome"
    )
    tag_commit = _git_optional("rev-list", "-n", "1", tag)
    tag_timestamp = (
        _git(
            "for-each-ref",
            "--format=%(creatordate:iso-strict)",
            f"refs/tags/{tag}",
        )
        if tag_commit
        else ""
    )
    builder_commit = _git("rev-parse", "HEAD")
    schemas = {
        "EvidenceObservation": EvidenceObservation.model_json_schema(),
        "HistoricalRegularFiling": HistoricalRegularFiling.model_json_schema(),
        "HistoricalFilingPair": HistoricalFilingPair.model_json_schema(),
        "PairedAxisPacket": PairedAxisPacket.model_json_schema(),
        "AxisPairClassification": AxisPairClassification.model_json_schema(),
    }
    return {
        "schema_version": "moatrader-future-eri-source-build-freeze-manifest/2",
        "research_variant": research_variant,
        "freeze_tag": tag,
        "freeze_tag_exists": bool(tag_commit),
        "freeze_tag_commit": tag_commit or None,
        "freeze_timestamp": tag_timestamp or None,
        "original_v1_tag_preserved": bool(
            _git("rev-list", "-n", "1", "future-eri-v1-preoutcome")
        ),
        "builder_git_commit": builder_commit,
        "builder_execution_timestamp": created_at.isoformat(),
        "feature_schema_hash": _canonical_hash(schemas),
        "feature_policy_hash": _canonical_hash(contract["feature"]),
        "eri_policy_hash": _canonical_hash(
            {
                "route": "FCFF",
                "horizon_trading_days": 63,
                "label": "log(actual_market_price_t_plus_63/counterfactual_fcff_value_t_plus_63)",
                "outcome_vault": contract["outcome_vault"],
            }
        ),
        "band_policy_hash": _canonical_hash(contract["feature_bands"]),
        "valuation_engine_version": (
            f"moatrader-future-eri-{research_variant.lower()}@{tag_commit or builder_commit}"
        ),
        "signal_timestamp_policy": "same session only before close; otherwise next trading session",
        "primary_ranking_policy": "NONE_MECHANISM_ONLY",
        "per_pbr_role": "NOT_USED",
    }


def read_trading_sessions(
    source: Path,
    *,
    begin_year: int,
    end_year: int,
) -> tuple[list[date], dict[str, Any]]:
    if source.is_dir():
        paths = [source / f"marcap-{year}.parquet" for year in range(begin_year, end_year + 2)]
        paths = [path for path in paths if path.is_file()]
        if (
            source.resolve() == DEFAULT_CALENDAR_ROOT.resolve()
            and end_year >= 2025
            and DEFAULT_2026_CALENDAR.is_file()
            and DEFAULT_2026_CALENDAR not in paths
        ):
            paths.append(DEFAULT_2026_CALENDAR)
    else:
        paths = [source]
    if not paths:
        raise FileNotFoundError(f"no trading-calendar inputs found: {source}")
    values: set[date] = set()
    sources: list[dict[str, Any]] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            frame = pd.read_parquet(path, columns=["Date"])
            series = frame["Date"]
            columns_read = ["Date"]
        elif suffix == ".csv":
            frame = pd.read_csv(path, usecols=["Date"])
            series = frame["Date"]
            columns_read = ["Date"]
        else:
            text = path.read_text(encoding="utf-8-sig")
            raw = json.loads(text) if text.lstrip().startswith("[") else text.splitlines()
            series = pd.Series(raw)
            columns_read = ["date_only"]
        for value in pd.to_datetime(series, errors="coerce").dropna():
            values.add(value.date())
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "columns_read": columns_read,
                "price_columns_read": False,
                "return_columns_read": False,
            }
        )
    sessions = sorted(values)
    if not sessions:
        raise ValueError("trading calendar contains no valid sessions")
    return sessions, {
        "session_count": len(sessions),
        "first_session": sessions[0].isoformat(),
        "last_session": sessions[-1].isoformat(),
        "sources": sources,
        "outcome_data_opened": False,
        "return_data_opened": False,
    }


def read_sector_map(
    path: Path | None,
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    if path is None or not path.is_file():
        return {}, {}, {
            "status": "NOT_AVAILABLE",
            "role": "COVERAGE_DIAGNOSTIC_ONLY_NOT_A_FEATURE",
        }
    mapping: dict[str, str] = {}
    issuer_names: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ticker = str(row.get("ticker") or row.get("stock_code") or "").zfill(6)
            sector = str(row.get("sector") or row.get("broad_sector") or "").strip()
            issuer_name = str(row.get("issuer_name") or row.get("corp_name") or "").strip()
            if len(ticker) == 6 and sector:
                mapping[ticker] = sector
            if len(ticker) == 6 and issuer_name:
                issuer_names[ticker] = issuer_name
    return mapping, issuer_names, {
        "status": "AVAILABLE",
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "role": "CURRENT_NON_PIT_COVERAGE_DIAGNOSTIC_ONLY_NOT_A_FEATURE",
        "mapped_ticker_count": len(mapping),
        "issuer_name_mask_count": len(issuer_names),
    }


def _gold_rows(
    candidates: dict[OperatingEvidenceAxis, list[PairedAxisPacket]],
) -> Iterable[dict[str, Any]]:
    for axis in OperatingEvidenceAxis:
        for packet in sorted(candidates[axis], key=lambda item: item.packet_id):
            yield {
                "packet_id": packet.packet_id,
                "axis": axis.value,
                "previous_excerpts_json": json.dumps(
                    [item.model_dump(mode="json") for item in packet.previous_excerpts],
                    ensure_ascii=False,
                ),
                "current_excerpts_json": json.dumps(
                    [item.model_dump(mode="json") for item in packet.current_excerpts],
                    ensure_ascii=False,
                ),
                "human_status": "",
                "human_previous_state": "",
                "human_current_state": "",
                "human_previous_source_id": "",
                "human_current_source_id": "",
                "human_previous_source_span": "",
                "human_current_source_span": "",
                "reviewer": "",
                "review_notes": "",
            }


def run(
    *,
    arcana_metadata: Path,
    arcana_business_html: Path,
    arcana_finance_comment_html: Path | None = None,
    arcana_finance_statement_html: Path | None = None,
    moatrader_data_lake: Path,
    trading_calendar: Path,
    output: Path,
    begin_year: int = 2020,
    end_year: int = 2025,
    sector_map_path: Path | None = None,
    tickers: set[str] | None = None,
    max_pairs: int | None = None,
    gold_per_axis: int = 40,
    extraction_workers: int = 1,
    research_variant: Literal["V1", "V1R"] = "V1R",
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if begin_year > end_year:
        raise ValueError("begin_year cannot exceed end_year")
    if max_pairs is not None and max_pairs < 1:
        raise ValueError("max_pairs must be positive")
    if extraction_workers < 1:
        raise ValueError("extraction_workers must be positive")
    contract = (
        FROZEN_FEATURE_CONTRACT_V1
        if research_variant == "V1"
        else FROZEN_FEATURE_CONTRACT_V1R
    )
    selected_arcana_sections = (
        ("business-info",)
        if research_variant == "V1"
        else ("business-info", "finance-comment", "finance-statement")
    )
    created_at = datetime.now(SEOUL)

    sessions, calendar_audit = read_trading_sessions(
        trading_calendar,
        begin_year=begin_year,
        end_year=end_year,
    )
    print(f"calendar: {len(sessions)} date-only sessions", flush=True)
    sectors, issuer_names, sector_audit = read_sector_map(sector_map_path)
    arcana, amendments, arcana_section_audit = discover_arcana_regular_sources(
        metadata_path=arcana_metadata,
        business_html_root=arcana_business_html,
        finance_comment_html_root=arcana_finance_comment_html,
        finance_statement_html_root=arcana_finance_statement_html,
        trading_sessions=sessions,
        begin_year=begin_year,
        end_year=end_year,
        tickers=tickers,
        included_sections=selected_arcana_sections,
    )
    print(
        f"arcana: regular={len(arcana)} amendments={len(amendments)} "
        f"all_three={arcana_section_audit['filing_count_with_all_three_sections']} "
        "source hashes complete",
        flush=True,
    )
    moatrader, moatrader_audit = discover_moatrader_original_sources(
        data_lake_root=moatrader_data_lake,
        trading_sessions=sessions,
        begin_year=begin_year,
        end_year=end_year,
        tickers=tickers,
    )
    print(
        f"moatrader: regular_original={len(moatrader)} metadata_scanned="
        f"{moatrader_audit['metadata_with_original_archive']}",
        flush=True,
    )
    merged, merge_audit = merge_historical_sources(arcana, moatrader)
    integrity = build_source_integrity_manifest(merged, created_at=created_at)
    pairs = build_regular_filing_pairs(merged)
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    print(
        f"merge: filings={len(merged)} pairs={len(pairs)} integrity_records={len(integrity.records)}",
        flush=True,
    )

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "frozen-contract.json", contract)
    _write_json(
        output / "freeze-manifest.json",
        frozen_manifest(
            created_at=created_at,
            research_variant=research_variant,
            contract=contract,
        ),
    )
    _write_json(output / "input-calendar-audit.json", calendar_audit)
    _write_json(output / "sector-map-audit.json", sector_audit)
    _write_json(
        output / "private" / "source-integrity-before.json",
        integrity.model_dump(mode="json"),
    )
    _write_jsonl(
        output / "private" / "regular-filings.jsonl",
        (item.model_dump(mode="json") for item in merged),
    )
    _write_jsonl(
        output / "private" / "filing-pairs.jsonl",
        (item.model_dump(mode="json") for item in pairs),
    )
    _write_json(
        output / "amendments-excluded.json",
        {
            "policy": "EXCLUDE_FROM_REGULAR_PAIRS_AND_REPORT_SEPARATELY",
            "count": len(amendments),
            "rows": amendments,
        },
    )

    axis_packet_count: Counter[str] = Counter()
    axis_both_count: Counter[str] = Counter()
    missing_axis_count: Counter[str] = Counter()
    complete_year: Counter[str] = Counter()
    complete_sector: Counter[str] = Counter()
    complete_pairs = 0
    extraction_read_count: Counter[str] = Counter()
    extraction_empty_candidate_count: Counter[str] = Counter()
    extraction_axis_window_count: dict[str, Counter[str]] = {}
    packet_excerpt_count: dict[str, Counter[str]] = {}
    two_period_candidate_by_origin: dict[str, Counter[str]] = {}
    packet_source_pattern_count: Counter[str] = Counter()
    incremental_finance_candidate_by_axis: dict[str, Counter[str]] = {
        "ARCANA_FINANCE_COMMENT_HTML": Counter(),
        "ARCANA_FINANCE_STATEMENT_HTML": Counter(),
    }
    gold: dict[OperatingEvidenceAxis, list[PairedAxisPacket]] = {
        axis: [] for axis in OperatingEvidenceAxis
    }
    private_path = output / "private" / "pair-source-map.jsonl"
    packet_path = output / "llm" / "blinded-packets.jsonl"
    private_path.parent.mkdir(parents=True, exist_ok=True)
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    pair_index = 0
    executor_type = ProcessPoolExecutor if extraction_workers > 1 else ThreadPoolExecutor
    with (
        executor_type(max_workers=extraction_workers) as extraction_pool,
        private_path.open("w", encoding="utf-8", newline="\n") as private_handle,
        packet_path.open("w", encoding="utf-8", newline="\n") as packet_handle,
    ):
        for ticker, pair_group in groupby(pairs, key=lambda item: item.ticker):
            ticker_pairs = list(pair_group)
            variants = {
                variant.raw_sha256: variant
                for pair in ticker_pairs
                for filing in (pair.previous, pair.current)
                for variant in filing.source_variants
            }
            window_cache = dict(
                extraction_pool.map(source_variant_axis_windows, variants.values())
            )
            for raw_sha256, variant in variants.items():
                origin = variant.origin.value
                windows = window_cache[raw_sha256]
                extraction_read_count[origin] += 1
                extraction_empty_candidate_count[origin] += int(
                    not any(windows[axis] for axis in OperatingEvidenceAxis)
                )
                origin_axis = extraction_axis_window_count.setdefault(origin, Counter())
                for axis in OperatingEvidenceAxis:
                    origin_axis[axis.value] += len(windows[axis])
            for pair in ticker_pairs:
                pair_index += 1
                protected_names = tuple(
                    name
                    for name in (issuer_names.get(ticker, ""),)
                    if name and name not in {pair.previous.issuer_name, pair.current.issuer_name}
                )
                packets, private = build_blinded_packets(
                    pair,
                    window_cache=window_cache,
                    brand_terms=protected_names,
                )
                private["coverage_sector"] = sectors.get(ticker, "UNMAPPED")
                private_handle.write(json.dumps(private, ensure_ascii=False, sort_keys=True) + "\n")
                pair_complete = True
                for packet in packets:
                    validate_packet_anonymization(
                        packet,
                        issuer_name=private["issuer_name"],
                        ticker=private["ticker"],
                        rcept_numbers=[private["previous_rcept_no"], private["current_rcept_no"]],
                        brand_terms=protected_names,
                    )
                    packet_handle.write(
                        json.dumps(packet.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
                        + "\n"
                    )
                    for excerpt in (*packet.previous_excerpts, *packet.current_excerpts):
                        origin = str(private["sources"][excerpt.source_id]["origin"])
                        packet_excerpt_count.setdefault(origin, Counter())[packet.axis.value] += 1
                    previous_origins = {
                        str(private["sources"][excerpt.source_id]["origin"])
                        for excerpt in packet.previous_excerpts
                    }
                    current_origins = {
                        str(private["sources"][excerpt.source_id]["origin"])
                        for excerpt in packet.current_excerpts
                    }
                    both_origins = previous_origins & current_origins
                    for origin in both_origins:
                        two_period_candidate_by_origin.setdefault(origin, Counter())[packet.axis.value] += 1
                    arcana_origins = (previous_origins | current_origins) & {
                        "ARCANA_BUSINESS_HTML",
                        "ARCANA_FINANCE_COMMENT_HTML",
                        "ARCANA_FINANCE_STATEMENT_HTML",
                    }
                    packet_source_pattern_count[
                        "MULTI_SECTION_EVIDENCE"
                        if len(arcana_origins) >= 2
                        else (
                            next(iter(arcana_origins))
                            if arcana_origins
                            else "NO_ARCANA_AXIS_CANDIDATE"
                        )
                    ] += 1
                    legacy_two_period = bool(
                        {
                            "ARCANA_BUSINESS_HTML",
                            "MOATRADER_OPENDART_ARCHIVE",
                        }
                        & both_origins
                    )
                    if not legacy_two_period:
                        for origin in incremental_finance_candidate_by_axis:
                            if origin in both_origins:
                                incremental_finance_candidate_by_axis[origin][
                                    packet.axis.value
                                ] += 1
                    axis_packet_count[packet.axis.value] += 1
                    both = bool(packet.previous_excerpts) and bool(packet.current_excerpts)
                    axis_both_count[packet.axis.value] += int(both)
                    if not both:
                        pair_complete = False
                        missing_axis_count[packet.axis.value] += 1
                    elif len(gold[packet.axis]) < gold_per_axis:
                        gold[packet.axis].append(packet)
                        gold[packet.axis].sort(key=lambda item: item.packet_id)
                        del gold[packet.axis][gold_per_axis:]
                if pair_complete:
                    complete_pairs += 1
                    complete_year[str(pair.current.fiscal_period_end.year)] += 1
                    complete_sector[sectors.get(ticker, "UNMAPPED")] += 1
                if pair_index % 1000 == 0:
                    print(
                        f"packets: pairs={pair_index}/{len(pairs)} complete_candidates={complete_pairs}",
                        flush=True,
                    )

    coverage = {
        "schema_version": "moatrader-historical-prelabel-coverage-v1/1",
        "total_filing_pairs": len(pairs),
        "six_axis_candidate_complete": complete_pairs,
        "candidate_coverage": complete_pairs / len(pairs) if pairs else 0.0,
        "unique_issuers": len({pair.ticker for pair in pairs}),
        "unique_signal_months": len(
            {pair.current.signal_timestamp.strftime("%Y-%m") for pair in pairs}
        ),
        "by_axis": {
            axis.value: {
                "packet_count": axis_packet_count[axis.value],
                "both_periods_have_candidate_spans": axis_both_count[axis.value],
                "missing_candidate_pair_count": missing_axis_count[axis.value],
            }
            for axis in OperatingEvidenceAxis
        },
        "complete_by_fiscal_year": dict(sorted(complete_year.items())),
        "complete_by_sector": dict(sorted(complete_sector.items())),
        "sector_role": "COVERAGE_DIAGNOSTIC_ONLY_NOT_A_FEATURE",
        "outcomes_opened": False,
        "returns_opened": False,
        "feature_labels_complete": False,
    }
    _write_json(output / "coverage" / "prelabel-packet-coverage.json", coverage)

    gold_path = output / "quality" / "human-gold-template.csv"
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    gold_rows = list(_gold_rows(gold))
    fieldnames = list(gold_rows[0]) if gold_rows else ["packet_id", "axis"]
    with gold_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(gold_rows)

    filing_origin_patterns: Counter[str] = Counter()
    for filing in merged:
        origins = {variant.origin.value for variant in filing.source_variants}
        has_moatrader = "MOATRADER_OPENDART_ARCHIVE" in origins
        arcana_count = len(
            origins
            & {
                "ARCANA_BUSINESS_HTML",
                "ARCANA_FINANCE_COMMENT_HTML",
                "ARCANA_FINANCE_STATEMENT_HTML",
            }
        )
        filing_origin_patterns[
            "ARCANA_MOATRADER_OVERLAP" if has_moatrader and arcana_count else (
                "ARCANA_ONLY" if arcana_count else "MOATRADER_ONLY"
            )
        ] += 1
        filing_origin_patterns["FINANCE_COMMENT_ATTACHED"] += int(
            "ARCANA_FINANCE_COMMENT_HTML" in origins
        )
        filing_origin_patterns["FINANCE_STATEMENT_ATTACHED"] += int(
            "ARCANA_FINANCE_STATEMENT_HTML" in origins
        )
        filing_origin_patterns["ALL_THREE_ARCANA_ATTACHED"] += int(arcana_count == 3)
    pair_overlap_patterns: Counter[str] = Counter()
    for pair in pairs:
        sides = [
            {variant.origin.value for variant in filing.source_variants}
            for filing in (pair.previous, pair.current)
        ]
        side_has_moatrader = ["MOATRADER_OPENDART_ARCHIVE" in origins for origins in sides]
        pair_overlap_patterns["BOTH_PERIODS_MOATRADER_OVERLAP"] += int(
            all(side_has_moatrader)
        )
        pair_overlap_patterns["EITHER_PERIOD_MOATRADER_OVERLAP"] += int(
            any(side_has_moatrader)
        )
        pair_overlap_patterns["ARCANA_ONLY_BOTH_PERIODS"] += int(
            not any(side_has_moatrader)
        )

    source_audit = {
        "schema_version": "moatrader-historical-source-audit-v1/2",
        "research_variant": research_variant,
        "source_contract_tag": (
            "future-eri-v1-preoutcome"
            if research_variant == "V1"
            else "future-eri-v1r-three-section-preoutcome"
        ),
        "same_feature_rule_as_v1": True,
        "arcana_section_selection": list(selected_arcana_sections),
        "arcana_regular_filing_count": len(arcana),
        "arcana_amendment_count": len(amendments),
        "arcana_section_audit": arcana_section_audit,
        "pair_source_extraction_by_origin": {
            origin: {
                "source_files_read": extraction_read_count[origin],
                "source_files_without_any_axis_candidate": extraction_empty_candidate_count[
                    origin
                ],
                "axis_keyword_windows": {
                    axis.value: extraction_axis_window_count.get(origin, Counter())[axis.value]
                    for axis in OperatingEvidenceAxis
                },
                "selected_packet_excerpt_count": sum(
                    packet_excerpt_count.get(origin, Counter()).values()
                ),
                "selected_packet_excerpts_by_axis": {
                    axis.value: packet_excerpt_count.get(origin, Counter())[axis.value]
                    for axis in OperatingEvidenceAxis
                },
            }
            for origin in sorted(set(extraction_read_count) | set(packet_excerpt_count))
        },
        "all_arcana_sections_discovered": research_variant == "V1R" and all(
            arcana_section_audit["sections"].get(section, {}).get(
                "discovered_nonempty_count", 0
            )
            > 0
            for section in ("business-info", "finance-comment", "finance-statement")
        ),
        "all_arcana_sections_read_for_pairs": research_variant == "V1R" and all(
            extraction_read_count[origin] > 0
            for origin in (
                "ARCANA_BUSINESS_HTML",
                "ARCANA_FINANCE_COMMENT_HTML",
                "ARCANA_FINANCE_STATEMENT_HTML",
            )
        ),
        "all_arcana_sections_contributed_to_packets": research_variant == "V1R" and all(
            sum(packet_excerpt_count.get(origin, Counter()).values()) > 0
            for origin in (
                "ARCANA_BUSINESS_HTML",
                "ARCANA_FINANCE_COMMENT_HTML",
                "ARCANA_FINANCE_STATEMENT_HTML",
            )
        ),
        "source_effect_audit": {
            "interpretation": {
                "A_TO_B": "SOURCE_COVERAGE_EFFECT_ONLY",
                "B_TO_C": "FEATURE_CONTRACT_EFFECT_ONLY",
            },
            "filing_origin_patterns": dict(sorted(filing_origin_patterns.items())),
            "pair_overlap_patterns": dict(sorted(pair_overlap_patterns.items())),
            "packet_source_pattern_counts": dict(
                sorted(packet_source_pattern_count.items())
            ),
            "two_period_candidate_packets_by_origin_and_axis": {
                origin: {
                    axis.value: two_period_candidate_by_origin.get(origin, Counter())[
                        axis.value
                    ]
                    for axis in OperatingEvidenceAxis
                }
                for origin in sorted(two_period_candidate_by_origin)
            },
            "incremental_two_period_candidates_without_business_or_moatrader_baseline": {
                origin: {
                    axis.value: incremental_finance_candidate_by_axis[origin][axis.value]
                    for axis in OperatingEvidenceAxis
                }
                for origin in sorted(incremental_finance_candidate_by_axis)
            },
            "outcomes_opened": False,
            "returns_opened": False,
            "value_data_opened": False,
        },
        "moatrader_regular_original_filing_count": len(moatrader),
        "moatrader_original_audit": moatrader_audit,
        "merge_audit": merge_audit,
        "regular_pair_count": len(pairs),
        "both_source_systems_used": bool(arcana) and bool(moatrader),
        "source_integrity_record_count": len(integrity.records),
        "source_files_modified": False,
    }
    _write_json(output / "source-audit.json", source_audit)
    _write_json(
        output / "stage-status.json",
        {
            "stage": "HISTORICAL_EVIDENCE_PACKET_BUILD",
            "research_variant": research_variant,
            "status": "AWAITING_LLM_AND_HUMAN_LABEL_QUALITY",
            "feature_dataset_sealed": False,
            "outcome_vault_opened": False,
            "return_data_opened": False,
            "next_gate": "HUMAN_GOLD_AGREEMENT_AND_SIX_AXIS_COMPLETE_FEATURE_COVERAGE",
            "primary_ranking_policy": "NONE_MECHANISM_ONLY",
            "per_pbr_role": "NOT_USED",
        },
    )

    verify_source_integrity(integrity)
    print("source-integrity-after: PASS_NO_SOURCE_MUTATION", flush=True)
    _write_json(
        output / "private" / "source-integrity-after.json",
        {
            **integrity.model_dump(mode="json"),
            "verified_at": datetime.now(SEOUL).isoformat(),
            "verification_status": "PASS_NO_SOURCE_MUTATION",
        },
    )
    artifacts = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "build-manifest.json"
    }
    _write_json(
        output / "build-manifest.json",
        {
            "schema_version": "moatrader-historical-evidence-build-manifest-v1/1",
            "research_variant": research_variant,
            "artifacts": artifacts,
            "credentials_persisted": False,
            "source_files_modified": False,
            "outcome_data_opened": False,
            "return_data_opened": False,
        },
    )
    return {
        **source_audit,
        "six_axis_candidate_complete": complete_pairs,
        "candidate_coverage": coverage["candidate_coverage"],
        "human_gold_template_count": len(gold_rows),
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "output": str(output.resolve()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build read-only, blinded historical filing pairs before any ERI outcome access."
    )
    parser.add_argument(
        "--arcana-metadata",
        type=Path,
        default=DEFAULT_ARCANA_ROOT / "silver" / "dart" / "kr_report_metadata.csv",
    )
    parser.add_argument(
        "--arcana-business-html",
        type=Path,
        default=DEFAULT_ARCANA_ROOT / "bronze" / "dart" / "business-info",
    )
    parser.add_argument(
        "--arcana-finance-comment-html",
        type=Path,
        default=DEFAULT_ARCANA_ROOT / "bronze" / "dart" / "finance-comment",
    )
    parser.add_argument(
        "--arcana-finance-statement-html",
        type=Path,
        default=DEFAULT_ARCANA_ROOT / "bronze" / "dart" / "finance-statement",
    )
    parser.add_argument("--moatrader-data-lake", type=Path, default=DEFAULT_MOATRADER_DATA_LAKE)
    parser.add_argument("--trading-calendar", type=Path, default=DEFAULT_CALENDAR_ROOT)
    parser.add_argument("--sector-map", type=Path, default=DEFAULT_SECTOR_MAP)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--begin-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--ticker", action="append", default=[])
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--gold-per-axis", type=int, default=40)
    parser.add_argument("--extraction-workers", type=int, default=8)
    parser.add_argument("--research-variant", choices=["V1", "V1R"], default="V1R")
    args = parser.parse_args()
    tickers = {str(item).zfill(6) for item in args.ticker} or None
    result = run(
        arcana_metadata=args.arcana_metadata,
        arcana_business_html=args.arcana_business_html,
        arcana_finance_comment_html=args.arcana_finance_comment_html,
        arcana_finance_statement_html=args.arcana_finance_statement_html,
        moatrader_data_lake=args.moatrader_data_lake,
        trading_calendar=args.trading_calendar,
        output=args.output,
        begin_year=args.begin_year,
        end_year=args.end_year,
        sector_map_path=args.sector_map,
        tickers=tickers,
        max_pairs=args.max_pairs,
        gold_per_axis=args.gold_per_axis,
        extraction_workers=args.extraction_workers,
        research_variant=args.research_variant,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
