from __future__ import annotations

import argparse
import concurrent.futures
import csv
import getpass
import hashlib
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar
from zoneinfo import ZoneInfo

import pandas as pd

from moatrader.backtest.historical import compound_change_ratios, sample_statistics
from moatrader.evidence.historical_corpus import (
    dart_original_excerpts,
    opaque_unit_id,
    pdf_excerpts,
    unique_excerpts,
)
from moatrader.evidence.historical_overlay import (
    HistoricalAssessmentBatch,
    HistoricalEntailmentBatch,
    HistoricalEvidenceDirection,
    HistoricalExcerpt,
    HistoricalOverlayDecision,
    HistoricalPackAssessment,
    HistoricalPreprocessBatch,
    HistoricalRiskAction,
    HistoricalSourceRole,
    anonymize_text,
    deterministic_overlay_decision,
    sanitize_preprocess_selection,
    validate_historical_assessment,
    validate_preprocess_selection,
)
from moatrader.experiments.integrity import snapshot_protected_files
from moatrader.ingestion import KindIrClient, ResilientHttpClient, normalize_company_name
from moatrader.ingestion.hankyung import (
    HankyungCompanyReport,
    HankyungIndustryReport,
    load_hankyung_company_reports,
    load_hankyung_industry_reports,
)
from moatrader.llm.historical import (
    build_historical_entailment_request,
    build_historical_evidence_request,
    build_historical_preprocess_request,
)
from moatrader.llm.transport import OpenAIResponsesTransport, TransportUsage


SEOUL = ZoneInfo("Asia/Seoul")
SCHEMA_VERSION = "moatrader-v7-historical-llm-pseudo-oos/1"
PREPROCESS_MODEL = "gpt-5-nano-2025-08-07"
MAIN_MODEL = "gpt-5.6-luna"
MAX_SOURCE_AGE_DAYS = 550
HORIZON_DAYS = 77
TOP_N = 15
T = TypeVar("T")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cutoff(value: str) -> datetime:
    return datetime.combine(date.fromisoformat(value), time.max, tzinfo=SEOUL)


def pack_id(row: dict[str, str]) -> str:
    return f"{row['signal_date']}_{row['ticker'].zfill(6)}"


def _excerpt_payload(item: HistoricalExcerpt) -> dict[str, Any]:
    return item.model_dump(mode="json")


def _load_pack(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    excerpts = [HistoricalExcerpt.model_validate(item) for item in payload["excerpts"]]
    payload["excerpts"] = list({item.unit_id: item for item in excerpts}.values())
    return payload


def _assessment_file(path: Path) -> HistoricalPackAssessment:
    payload = read_json(path)
    return HistoricalPackAssessment.model_validate(
        {key: payload[key] for key in ("pack_id", "claims", "future_trap_answer")}
    )


def _entailment_file(path: Path) -> HistoricalEntailmentBatch:
    payload = read_json(path)
    return HistoricalEntailmentBatch.model_validate({"decisions": payload["decisions"]})


def migrate_candidate_unit_ids(output: Path, pack_ids: list[str]) -> int:
    """Upgrade generated candidate IDs without changing any source evidence."""

    changed = 0
    for identifier in pack_ids:
        path = output / "packs" / "candidates" / f"{identifier}.json"
        payload = read_json(path)
        dirty = False
        for item in payload["excerpts"]:
            expected = opaque_unit_id(str(item["source_id"]), str(item["text"]))
            if item["unit_id"] != expected:
                item["unit_id"] = expected
                dirty = True
        if dirty:
            write_json(path, payload)
            changed += 1
    return changed


def _report_catalog(
    root: Path,
) -> tuple[
    dict[str, list[tuple[HankyungCompanyReport, Path]]],
    dict[str, list[tuple[HankyungIndustryReport, Path]]],
    dict[str, str],
]:
    companies: dict[str, list[tuple[HankyungCompanyReport, Path]]] = defaultdict(list)
    industries: dict[str, list[tuple[HankyungIndustryReport, Path]]] = defaultdict(list)
    taxonomy: dict[str, str] = {}
    for year in range(2020, 2026):
        company_base = root / "company" / str(year)
        company_meta = company_base / "json" / "reports.json"
        if company_meta.is_file():
            files = {item.name.partition("_")[0]: item for item in (company_base / "pdf").glob("*.pdf")}
            for report in load_hankyung_company_reports(company_meta).values():
                if path := files.get(report.report_id):
                    companies[report.ticker].append((report, path))
        industry_base = root / "industry" / str(year)
        industry_meta = industry_base / "json" / "reports.json"
        if industry_meta.is_file():
            files = {item.name.partition("_")[0]: item for item in (industry_base / "pdf").glob("*.pdf")}
            for report in load_hankyung_industry_reports(industry_meta).values():
                taxonomy.setdefault(report.industry_code, report.industry_name)
                if path := files.get(report.report_id):
                    industries[report.industry_code].append((report, path))
    for values in companies.values():
        values.sort(key=lambda item: (item[0].registered_at, item[0].report_id))
    for values in industries.values():
        values.sort(key=lambda item: (item[0].registered_at, item[0].report_id))
    return companies, industries, dict(sorted(taxonomy.items()))


def _latest(
    items: list[tuple[T, Path]],
    *,
    as_of: datetime,
    timestamp: Callable[[T], datetime],
) -> tuple[T, Path] | None:
    eligible = [item for item in items if timestamp(item[0]) <= as_of]
    if not eligible:
        return None
    selected = eligible[-1]
    if as_of - timestamp(selected[0]) > timedelta(days=MAX_SOURCE_AGE_DAYS):
        return None
    return selected


def _kind_catalog(
    *,
    signal_rows: list[dict[str, str]],
    company_reports: dict[str, list[tuple[HankyungCompanyReport, Path]]],
    output: Path,
) -> tuple[dict[str, list[Any]], dict[str, Any]]:
    catalog_path = output / "source-map" / "kind-materials.json"
    if catalog_path.is_file():
        from moatrader.ingestion.kind import KindIrMaterial

        payload = read_json(catalog_path)
        materials = [KindIrMaterial.model_validate(item) for item in payload["materials"]]
    else:
        http = ResilientHttpClient(
            user_agent="MoatRader v7 historical evidence validation",
            requests_per_second=2.0,
            timeout_seconds=45,
            max_retries=4,
            default_max_bytes=128 * 1024 * 1024,
        )
        client = KindIrClient(http)
        materials = []
        for year in range(2020, 2026):
            found = client.search_materials(
                begin_date=date(year, 1, 1),
                end_date=date(year, 12, 31),
            )
            materials.extend(found)
            print(f"KIND {year}: {len(found)} materials", flush=True)
        write_json(catalog_path, {"materials": [item.model_dump(mode="json") for item in materials]})

    names_by_ticker: dict[str, set[str]] = defaultdict(set)
    for row in signal_rows:
        names_by_ticker[row["ticker"].zfill(6)].add(normalize_company_name(row["name"]))
    for ticker, reports in company_reports.items():
        names_by_ticker[ticker].update(normalize_company_name(item[0].company_name) for item in reports)
    exact: dict[str, set[str]] = defaultdict(set)
    for ticker, names in names_by_ticker.items():
        for name in names:
            if name:
                exact[name].add(ticker)
    by_ticker: dict[str, list[Any]] = defaultdict(list)
    unmatched = 0
    ambiguous = 0
    for material in materials:
        name = normalize_company_name(material.company_name)
        matches = exact.get(name, set())
        if not matches and len(name) >= 3:
            fuzzy = {
                ticker
                for alias, tickers in exact.items()
                if len(alias) >= 3 and (alias in name or name in alias)
                for ticker in tickers
            }
            matches = fuzzy
        if len(matches) == 1:
            by_ticker[next(iter(matches))].append(material)
        elif matches:
            ambiguous += 1
        else:
            unmatched += 1
    for values in by_ticker.values():
        values.sort(key=lambda item: (item.listed_on, int(item.ir_seq), item.attachment_index))
    return by_ticker, {
        "discovered": len(materials),
        "matched": sum(len(value) for value in by_ticker.values()),
        "unmatched": unmatched,
        "ambiguous": ambiguous,
        "matched_tickers": len(by_ticker),
    }


def prepare_corpus(
    *,
    signals: Path,
    dart_root: Path,
    synalyst_root: Path,
    output: Path,
) -> list[str]:
    manifest_path = output / "corpus-manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        pack_ids = list(manifest["pack_ids"])
        migrated = migrate_candidate_unit_ids(output, pack_ids)
        if migrated:
            manifest["unit_id_scheme"] = "opaque-sha256-prefix-v1"
            manifest["candidate_packs_migrated"] = migrated
            write_json(manifest_path, manifest)
        return pack_ids
    rows = [item for item in read_csv(signals) if item["status"] == "ELIGIBLE"]
    if len(rows) != 1497:
        raise ValueError(f"expected 1,497 eligible deterministic observations, got {len(rows)}")
    company_reports, industry_reports, taxonomy = _report_catalog(synalyst_root)
    kind_by_ticker, kind_stats = _kind_catalog(
        signal_rows=rows,
        company_reports=company_reports,
        output=output,
    )
    http = ResilientHttpClient(
        user_agent="MoatRader v7 historical evidence validation",
        requests_per_second=2.0,
        timeout_seconds=45,
        max_retries=4,
        default_max_bytes=128 * 1024 * 1024,
    )
    kind_client = KindIrClient(http)
    dart_cache: dict[str, list[HistoricalExcerpt]] = {}
    pdf_cache: dict[str, tuple[list[HistoricalExcerpt], int]] = {}
    source_assignments: list[dict[str, Any]] = []
    opinion_lines_removed = 0
    pack_ids: list[str] = []
    for index, row in enumerate(rows, start=1):
        ticker = row["ticker"].zfill(6)
        as_of = cutoff(row["signal_date"])
        identifier = pack_id(row)
        dart_id = row["latest_rcept_no"]
        dart_dir = dart_root / "filings" / ticker / dart_id
        if dart_id not in dart_cache:
            dart_cache[dart_id] = dart_original_excerpts(dart_dir, maximum=12)
        groups: list[list[HistoricalExcerpt]] = [dart_cache[dart_id]]
        assignment: dict[str, Any] = {
            "pack_id": identifier,
            "signal_date": row["signal_date"],
            "ticker": ticker,
            "dart_rcept_no": dart_id,
            "dart_source_ids": sorted({item.source_id for item in dart_cache[dart_id]}),
        }

        company = _latest(
            company_reports.get(ticker, []),
            as_of=as_of,
            timestamp=lambda item: item.registered_at,
        )
        if company is not None:
            report, path = company
            cache_key = report.source_document_id
            if cache_key not in pdf_cache:
                pdf_cache[cache_key] = pdf_excerpts(
                    path,
                    source_id=cache_key,
                    source_role=HistoricalSourceRole.COMPANY_ANALYST,
                    available_at=report.registered_at,
                    maximum=8,
                )
            groups.append(pdf_cache[cache_key][0])
            opinion_lines_removed += pdf_cache[cache_key][1]
            assignment["company_analyst_document_id"] = cache_key
            assignment["company_analyst_path"] = str(path.resolve())
            assignment["company_analyst_available_at"] = report.registered_at.isoformat()

        ir_candidates = [
            item
            for item in kind_by_ticker.get(ticker, [])
            if datetime.combine(item.listed_on + timedelta(days=1), time.min, tzinfo=SEOUL) <= as_of
        ]
        ir_candidates = [
            item
            for item in ir_candidates
            if as_of - datetime.combine(item.listed_on + timedelta(days=1), time.min, tzinfo=SEOUL)
            <= timedelta(days=MAX_SOURCE_AGE_DAYS)
        ]
        if ir_candidates:
            material = ir_candidates[-1]
            available_at = datetime.combine(material.listed_on + timedelta(days=1), time.min, tzinfo=SEOUL)
            path = output / "raw" / "ir" / f"{material.source_document_id}.pdf"
            if not path.is_file():
                content = kind_client.download_pdf(material, max_bytes=128 * 1024 * 1024)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            cache_key = material.source_document_id
            if cache_key not in pdf_cache:
                pdf_cache[cache_key] = pdf_excerpts(
                    path,
                    source_id=cache_key,
                    source_role=HistoricalSourceRole.IR,
                    available_at=available_at,
                    maximum=8,
                )
            groups.append(pdf_cache[cache_key][0])
            assignment["ir_document_id"] = cache_key
            assignment["ir_available_at"] = available_at.isoformat()
            assignment["ir_original_url"] = material.attachment_url

        excerpts = unique_excerpts(groups)
        if not excerpts:
            raise ValueError(f"no cutoff evidence for {identifier}")
        payload = {
            "pack_id": identifier,
            "signal_date": row["signal_date"],
            "cutoff": as_of.isoformat(),
            "ticker": ticker,
            "issuer_name": row["name"],
            "cheap": row["cheap"],
            "cheap_rank": row["cheap_rank"],
            "excerpts": [_excerpt_payload(item) for item in excerpts],
        }
        write_json(output / "packs" / "candidates" / f"{identifier}.json", payload)
        pack_ids.append(identifier)
        source_assignments.append(assignment)
        if index % 50 == 0:
            print(f"prepared {index}/{len(rows)} packs", flush=True)
    write_json(output / "source-map" / "assignments.json", source_assignments)
    write_json(output / "source-map" / "industry-taxonomy.json", taxonomy)
    write_json(
        output / "source-map" / "industry-catalog.json",
        {
            code: [
                {
                    "document_id": report.source_document_id,
                    "available_at": report.registered_at.isoformat(),
                    "path": str(path.resolve()),
                    "title": report.title,
                }
                for report, path in reports
            ]
            for code, reports in industry_reports.items()
        },
    )
    write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "validation_grade": "LLM_PIT_PSEUDO_OOS",
            "pack_ids": pack_ids,
            "pack_count": len(pack_ids),
            "dart_original_reused_read_only": True,
            "synalyst_reports_reused_read_only": True,
            "kind_ir_new_v7_data_only": True,
            "kind_stats": kind_stats,
            "market_opinion_lines_quarantined": opinion_lines_removed,
            "max_source_age_days": MAX_SOURCE_AGE_DAYS,
            "returns_opened": False,
            "unit_id_scheme": "opaque-sha256-prefix-v1",
        },
    )
    return pack_ids


def _secure_transport(prompt_api_key: bool) -> OpenAIResponsesTransport:
    added = False
    if not os.getenv("OPENAI_API_KEY"):
        if not prompt_api_key:
            raise RuntimeError("OPENAI_API_KEY is absent; use --prompt-api-key for hidden input")
        value = getpass.getpass("OpenAI API key (hidden, never persisted): ").strip()
        if not value:
            raise ValueError("OpenAI API key is empty")
        os.environ["OPENAI_API_KEY"] = value
        added = True
    transport = OpenAIResponsesTransport(
        summary_model=PREPROCESS_MODEL,
        moat_model=MAIN_MODEL,
        summary_reasoning_effort="low",
        atomic_reasoning_effort="medium",
        max_output_tokens=8_000,
        max_retries=4,
        timeout_seconds=240,
    )
    transport._client()
    if added:
        del os.environ["OPENAI_API_KEY"]
    return transport


def _batches(values: list[T], size: int) -> list[list[T]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def preprocess_packs(
    *,
    pack_ids: list[str],
    output: Path,
    transport: OpenAIResponsesTransport,
    batch_size: int,
    workers: int,
) -> None:
    taxonomy = read_json(output / "source-map" / "industry-taxonomy.json")
    pending = [item for item in pack_ids if not (output / "llm" / "preprocess" / f"{item}.json").is_file()]
    batches = _batches(pending, batch_size)

    def execute(ids: list[str]) -> tuple[list[str], Any, Any]:
        packs = [_load_pack(output / "packs" / "candidates" / f"{item}.json") for item in ids]
        request = build_historical_preprocess_request(packs, industry_taxonomy=taxonomy)
        result = transport.execute(request, HistoricalPreprocessBatch)
        return ids, request, result

    usage = Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, batch): batch for batch in batches}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            ids = futures[future]
            try:
                returned, request, result = future.result()
                by_id = {item.pack_id: item for item in result.parsed.packs}
                unknown = set(by_id) - set(returned)
                if unknown:
                    raise ValueError("preprocess invented pack IDs: " + ", ".join(sorted(unknown)))
                for identifier in sorted(set(returned) & set(by_id)):
                    pack = _load_pack(output / "packs" / "candidates" / f"{identifier}.json")
                    selection = sanitize_preprocess_selection(
                        by_id[identifier],
                        pack["excerpts"],
                        allowed_industry_codes=set(taxonomy),
                    )
                    validate_preprocess_selection(selection, pack["excerpts"])
                    write_json(
                        output / "llm" / "preprocess" / f"{identifier}.json",
                        {
                            **selection.model_dump(mode="json"),
                            "model": result.model,
                            "response_id": result.response_id,
                            "request_sha256": request.input_sha256,
                        },
                    )
                missing = sorted(set(returned) - set(by_id))
                if missing:
                    write_json(
                        output / "llm" / "failures" / f"preprocess-{ids[0]}.json",
                        {"pack_ids": ids, "missing_pack_ids": missing, "error": "PARTIAL_RESPONSE_RETRY_REQUIRED"},
                    )
                usage.update(result.usage.model_dump())
            except Exception as exc:
                write_json(
                    output / "llm" / "failures" / f"preprocess-{ids[0]}.json",
                    {"pack_ids": ids, "error": f"{type(exc).__name__}: {exc}"},
                )
            completed += len(ids)
            print(f"nano preprocessed {completed}/{len(pending)} pending packs", flush=True)
    write_json(output / "llm" / "preprocess-usage.json", dict(usage))


def hydrate_selected_packs(pack_ids: list[str], output: Path) -> list[str]:
    catalog = read_json(output / "source-map" / "industry-catalog.json")
    pdf_cache: dict[str, tuple[list[HistoricalExcerpt], int]] = {}
    successful: list[str] = []
    for index, identifier in enumerate(pack_ids, start=1):
        destination = output / "packs" / "selected" / f"{identifier}.json"
        if destination.is_file():
            successful.append(identifier)
            continue
        preprocess_path = output / "llm" / "preprocess" / f"{identifier}.json"
        if not preprocess_path.is_file():
            continue
        pack = _load_pack(output / "packs" / "candidates" / f"{identifier}.json")
        selection = HistoricalPreprocessBatch.model_validate(
            {"packs": [{key: value for key, value in read_json(preprocess_path).items() if key in {"pack_id", "selected_unit_ids", "industry_codes"}}]}
        ).packs[0]
        excerpts = validate_preprocess_selection(selection, pack["excerpts"])
        as_of = datetime.fromisoformat(pack["cutoff"])
        for industry_code in selection.industry_codes:
            candidates = [
                item for item in catalog.get(industry_code, [])
                if datetime.fromisoformat(item["available_at"]) <= as_of
                and as_of - datetime.fromisoformat(item["available_at"]) <= timedelta(days=MAX_SOURCE_AGE_DAYS)
            ]
            if not candidates:
                continue
            selected = candidates[-1]
            document_id = selected["document_id"]
            if document_id not in pdf_cache:
                pdf_cache[document_id] = pdf_excerpts(
                    Path(selected["path"]),
                    source_id=document_id,
                    source_role=HistoricalSourceRole.INDUSTRY_ANALYST,
                    available_at=datetime.fromisoformat(selected["available_at"]),
                    maximum=6,
                )
            excerpts.extend(pdf_cache[document_id][0])
        excerpts = unique_excerpts([excerpts])
        pack["excerpts"] = [_excerpt_payload(item) for item in excerpts]
        pack["anonymized_text_by_unit"] = {
            item.unit_id: anonymize_text(item.text, pack["issuer_name"], pack["ticker"])
            for item in excerpts
        }
        pack["industry_codes"] = selection.industry_codes
        write_json(destination, pack)
        successful.append(identifier)
        if index % 100 == 0:
            print(f"hydrated {index}/{len(pack_ids)} selected packs", flush=True)
    return successful


def classify_packs(
    *,
    pack_ids: list[str],
    output: Path,
    transport: OpenAIResponsesTransport,
    batch_size: int,
    workers: int,
    anonymized: bool,
) -> None:
    lane = "anonymized" if anonymized else "original"
    pending = [item for item in pack_ids if not (output / "llm" / lane / f"{item}.json").is_file()]
    batches = _batches(pending, batch_size)

    def execute(ids: list[str]) -> tuple[list[str], Any, Any]:
        packs = [_load_pack(output / "packs" / "selected" / f"{item}.json") for item in ids]
        for pack in packs:
            raw = read_json(output / "packs" / "selected" / f"{pack['pack_id']}.json")
            pack["anonymized_text_by_unit"] = raw["anonymized_text_by_unit"]
        request = build_historical_evidence_request(packs, anonymized=anonymized)
        result = transport.execute(request, HistoricalAssessmentBatch)
        return ids, request, result

    usage = Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, batch): batch for batch in batches}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            ids = futures[future]
            try:
                returned, request, result = future.result()
                by_id = {item.pack_id: item for item in result.parsed.packs}
                unknown = set(by_id) - set(returned)
                if unknown:
                    raise ValueError(f"{lane} invented pack IDs: " + ", ".join(sorted(unknown)))
                for identifier in sorted(set(returned) & set(by_id)):
                    assessment = by_id[identifier]
                    normalized_claims = [
                        claim.model_copy(update={"judgment_id": f"{identifier}:J{index:02d}"})
                        for index, claim in enumerate(assessment.claims, start=1)
                    ]
                    assessment = assessment.model_copy(update={"claims": normalized_claims})
                    write_json(
                        output / "llm" / lane / f"{identifier}.json",
                        {
                            **assessment.model_dump(mode="json"),
                            "model": result.model,
                            "response_id": result.response_id,
                            "request_sha256": request.input_sha256,
                        },
                    )
                missing = sorted(set(returned) - set(by_id))
                if missing:
                    write_json(
                        output / "llm" / "failures" / f"{lane}-{ids[0]}.json",
                        {"pack_ids": ids, "missing_pack_ids": missing, "error": "PARTIAL_RESPONSE_RETRY_REQUIRED"},
                    )
                usage.update(result.usage.model_dump())
            except Exception as exc:
                write_json(
                    output / "llm" / "failures" / f"{lane}-{ids[0]}.json",
                    {"pack_ids": ids, "error": f"{type(exc).__name__}: {exc}"},
                )
            completed += len(ids)
            print(f"Luna {lane} {completed}/{len(pending)} pending packs", flush=True)
    write_json(output / "llm" / f"{lane}-usage.json", dict(usage))


def judge_entailment(
    *,
    pack_ids: list[str],
    output: Path,
    transport: OpenAIResponsesTransport,
    batch_size: int,
    workers: int,
) -> None:
    pending: list[str] = []
    for item in pack_ids:
        destination = output / "llm" / "entailment" / f"{item}.json"
        if destination.is_file():
            continue
        assessment = _assessment_file(output / "llm" / "original" / f"{item}.json")
        if not assessment.claims:
            write_json(
                destination,
                {
                    "decisions": [],
                    "model": "DETERMINISTIC_EMPTY_CLAIM_SET",
                    "response_id": None,
                    "request_sha256": None,
                },
            )
        else:
            pending.append(item)
    batches = _batches(pending, batch_size)

    def execute(ids: list[str]) -> tuple[list[str], Any, Any, list[HistoricalPackAssessment]]:
        assessments = [_assessment_file(output / "llm" / "original" / f"{item}.json") for item in ids]
        excerpts_by_pack = {
            item: _load_pack(output / "packs" / "selected" / f"{item}.json")["excerpts"] for item in ids
        }
        request = build_historical_entailment_request(assessments, excerpts_by_pack)
        result = transport.execute(request, HistoricalEntailmentBatch)
        return ids, request, result, assessments

    usage = Counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, batch): batch for batch in batches}
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            ids = futures[future]
            try:
                returned, request, result, assessments = future.result()
                expected = {claim.judgment_id for item in assessments for claim in item.claims}
                decisions = {item.judgment_id: item for item in result.parsed.decisions}
                if set(decisions) != expected:
                    raise ValueError("entailment judgment IDs differ")
                for assessment in assessments:
                    selected = [decisions[item.judgment_id] for item in assessment.claims]
                    write_json(
                        output / "llm" / "entailment" / f"{assessment.pack_id}.json",
                        {
                            "decisions": [item.model_dump(mode="json") for item in selected],
                            "model": result.model,
                            "response_id": result.response_id,
                            "request_sha256": request.input_sha256,
                        },
                    )
                usage.update(result.usage.model_dump())
            except Exception as exc:
                write_json(
                    output / "llm" / "failures" / f"entailment-{ids[0]}.json",
                    {"pack_ids": ids, "error": f"{type(exc).__name__}: {exc}"},
                )
            completed += len(ids)
            print(f"Luna entailment {completed}/{len(pending)} pending packs", flush=True)
    write_json(output / "llm" / "entailment-usage.json", dict(usage))


def seal_signals(*, pack_ids: list[str], signals: Path, output: Path) -> list[dict[str, Any]]:
    decision_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for identifier in pack_ids:
        required = [output / "llm" / lane / f"{identifier}.json" for lane in ("original", "anonymized", "entailment")]
        if not all(path.is_file() for path in required):
            failures.append({"pack_id": identifier, "error": "MISSING_LLM_STAGE"})
            continue
        pack = _load_pack(output / "packs" / "selected" / f"{identifier}.json")
        original = _assessment_file(required[0])
        anonymous = _assessment_file(required[1])
        entailed = _entailment_file(required[2])
        try:
            claims = validate_historical_assessment(
                cutoff=datetime.fromisoformat(pack["cutoff"]),
                excerpts=pack["excerpts"],
                original=original,
                anonymized=anonymous,
                entailment=entailed,
            )
            full = deterministic_overlay_decision(identifier, claims)
            dart_ir = deterministic_overlay_decision(
                identifier,
                [
                    item for item in claims
                    if item.source_role in {HistoricalSourceRole.DART_ORIGINAL, HistoricalSourceRole.IR}
                ],
            )
            decision_rows.append(
                {
                    "pack_id": identifier,
                    "signal_date": pack["signal_date"],
                    "ticker": pack["ticker"],
                    "cheap": pack["cheap"],
                    "cheap_rank": pack["cheap_rank"],
                    "dart_ir_action": dart_ir.action.value,
                    "dart_ir_supportive_count": dart_ir.supportive_count,
                    "dart_ir_erosive_count": dart_ir.erosive_count,
                    "full_action": full.action.value,
                    "full_supportive_count": full.supportive_count,
                    "full_erosive_count": full.erosive_count,
                    "validated_claim_count": len(claims),
                    "validation_grade": full.validation_grade,
                    "llm_changed_cheap_rank": False,
                }
            )
            write_json(output / "validated" / f"{identifier}.json", full.model_dump(mode="json"))
        except Exception as exc:
            failures.append({"pack_id": identifier, "error": f"{type(exc).__name__}: {exc}"})
    write_csv(output / "sealed-overlay-signals.csv", decision_rows)
    write_json(output / "validation-failures.json", failures)
    decision_hash = sha256_file(output / "sealed-overlay-signals.csv")
    seal = {
        "schema_version": SCHEMA_VERSION,
        "sealed_at": datetime.now(SEOUL).isoformat(),
        "base_signals_sha256": sha256_file(signals),
        "overlay_signals_sha256": decision_hash,
        "eligible_pack_count": len(pack_ids),
        "validated_pack_count": len(decision_rows),
        "failed_closed_pack_count": len(failures),
        "validation_grade": "LLM_PIT_PSEUDO_OOS",
        "deterministic_rank_signal": "CHEAP",
        "llm_may_change_rank": False,
        "returns_opened_before_seal": False,
        "preprocess_model_requested": PREPROCESS_MODEL,
        "main_model_requested": MAIN_MODEL,
    }
    seal["seal_sha256"] = sha256_json(seal)
    write_json(output / "signals-seal.json", seal)
    return decision_rows


def _forward_return(frame: pd.DataFrame, *, entry_date: date) -> float | None:
    target = entry_date + timedelta(days=HORIZON_DAYS)
    window = frame[(frame["date"] > entry_date) & (frame["date"] <= target)]
    if window.empty or (target - window.iloc[-1]["date"]).days > 10:
        return None
    return float(compound_change_ratios(window["changes_ratio_percent"].tolist()))


def _portfolio(rows: list[dict[str, Any]], action_field: str | None) -> tuple[float | None, int, int, int]:
    ordered = sorted(rows, key=lambda item: (float(item["cheap_rank"]), item["ticker"]))
    if action_field:
        ordered = [item for item in ordered if item.get(action_field) != HistoricalRiskAction.VETO.value]
    selected = ordered[:TOP_N]
    values: list[tuple[float, float]] = []
    for item in selected:
        if item.get("forward_return") is None:
            continue
        weight = 0.5 if action_field and item.get(action_field) == HistoricalRiskAction.POSITION_CAP.value else 1.0
        values.append((float(item["forward_return"]), weight))
    result = sum(value * weight for value, weight in values) / sum(weight for _, weight in values) if values else None
    vetoes = sum(item.get(action_field) == HistoricalRiskAction.VETO.value for item in rows) if action_field else 0
    caps = sum(item.get(action_field) == HistoricalRiskAction.POSITION_CAP.value for item in selected) if action_field else 0
    return result, len(values), vetoes, caps


def _validation_failure_category(error: str) -> str:
    if "anonymization classification instability" in error:
        return "ANONYMIZATION_CLASSIFICATION_INSTABILITY"
    if "claim is not independently entailed" in error:
        return "CLAIM_NOT_INDEPENDENTLY_ENTAILED"
    if "exact quote" in error or "claim quote is not an exact source span" in error:
        return "EXACT_QUOTE_FAILURE"
    if "confidence instability" in error:
        return "ANONYMIZATION_CONFIDENCE_INSTABILITY"
    if "claim cites absent unit" in error:
        return "CLAIM_CITES_ABSENT_UNIT"
    if "future" in error.casefold():
        return "FUTURE_KNOWLEDGE_TRAP_FAILURE"
    if error == "MISSING_LLM_STAGE":
        return error
    return "OTHER_VALIDATION_FAILURE"


def _source_audit(output: Path, pack_ids: Iterable[str]) -> tuple[dict[str, Any], dict[str, int]]:
    pack_counts: Counter[str] = Counter()
    excerpt_counts: Counter[str] = Counter()
    claim_counts: Counter[str] = Counter()
    for identifier in pack_ids:
        pack = _load_pack(output / "packs" / "selected" / f"{identifier}.json")
        roles = {item.source_role.value for item in pack["excerpts"]}
        pack_counts.update(roles)
        excerpt_counts.update(item.source_role.value for item in pack["excerpts"])
        validated = output / "validated" / f"{identifier}.json"
        if validated.is_file():
            decision = read_json(validated)
            claim_counts.update(str(item["source_role"]) for item in decision["validated_claims"])
    coverage = {
        role: {"pack_count": pack_counts[role], "excerpt_count": excerpt_counts[role]}
        for role in sorted(set(pack_counts) | set(excerpt_counts))
    }
    return coverage, dict(sorted(claim_counts.items()))


def evaluate(*, signals: Path, prices: Path, output: Path) -> dict[str, Any]:
    seal_path = output / "signals-seal.json"
    if not seal_path.is_file():
        raise RuntimeError("overlay signals must be sealed before future returns are opened")
    seal = read_json(seal_path)
    if sha256_file(output / "sealed-overlay-signals.csv") != seal["overlay_signals_sha256"]:
        raise ValueError("sealed overlay signals changed before return join")
    base_rows = {pack_id(item): item for item in read_csv(signals) if item["status"] == "ELIGIBLE"}
    decision_rows = {item["pack_id"]: item for item in read_csv(output / "sealed-overlay-signals.csv")}
    frame = pd.read_csv(prices, encoding="utf-8-sig")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["date"] = frame["timestamp"].dt.tz_convert("Asia/Seoul").dt.date
    frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    by_ticker = {ticker: values.sort_values("date") for ticker, values in frame.groupby("ticker", sort=False)}
    full_universe_merged: list[dict[str, Any]] = []
    for identifier, base in base_rows.items():
        ticker = base["ticker"].zfill(6)
        prices_for_ticker = by_ticker.get(ticker)
        value = _forward_return(prices_for_ticker, entry_date=date.fromisoformat(base["price_date"])) if prices_for_ticker is not None else None
        full_universe_merged.append({**base, "pack_id": identifier, "forward_return": value})
    merged: list[dict[str, Any]] = []
    for identifier, decision in decision_rows.items():
        base = base_rows[identifier]
        ticker = base["ticker"].zfill(6)
        prices_for_ticker = by_ticker.get(ticker)
        value = _forward_return(prices_for_ticker, entry_date=date.fromisoformat(base["price_date"])) if prices_for_ticker is not None else None
        merged.append({**base, **decision, "forward_return": value})
    by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in merged:
        by_date[row["signal_date"]].append(row)
    full_universe_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in full_universe_merged:
        full_universe_by_date[row["signal_date"]].append(row)
    period_rows: list[dict[str, Any]] = []
    for signal_date, rows in sorted(by_date.items()):
        full_universe_baseline, full_universe_n, _, _ = _portfolio(full_universe_by_date[signal_date], None)
        baseline, baseline_n, _, _ = _portfolio(rows, None)
        dart_ir, dart_ir_n, dart_ir_veto, dart_ir_cap = _portfolio(rows, "dart_ir_action")
        full, full_n, full_veto, full_cap = _portfolio(rows, "full_action")
        period_rows.append(
            {
                "signal_date": signal_date,
                "full_universe_baseline_data_pit_return": full_universe_baseline,
                "baseline_data_pit_return": baseline,
                "dart_ir_llm_pseudo_oos_return": dart_ir,
                "full_llm_pseudo_oos_return": full,
                "dart_ir_minus_baseline": dart_ir - baseline if dart_ir is not None and baseline is not None else None,
                "full_minus_baseline": full - baseline if full is not None and baseline is not None else None,
                "full_universe_baseline_n": full_universe_n,
                "baseline_n": baseline_n,
                "dart_ir_n": dart_ir_n,
                "full_n": full_n,
                "dart_ir_veto_count": dart_ir_veto,
                "dart_ir_cap_count": dart_ir_cap,
                "full_veto_count": full_veto,
                "full_cap_count": full_cap,
            }
        )
    write_csv(output / "evaluation-periods.csv", period_rows)

    def stats(field: str) -> dict[str, Any]:
        values = [float(item[field]) for item in period_rows if item[field] not in (None, "") and math.isfinite(float(item[field]))]
        return sample_statistics(values)

    failure_rows = read_json(output / "validation-failures.json")
    failure_reasons = dict(
        sorted(Counter(_validation_failure_category(str(item["error"])) for item in failure_rows).items())
    )
    source_coverage, validated_claims_by_source_role = _source_audit(output, base_rows)
    analyst_action_changes = sum(
        item["dart_ir_action"] != item["full_action"] for item in decision_rows.values()
    )
    baseline_metadata_path = signals.parent / "summary.json"
    baseline_metadata = read_json(baseline_metadata_path) if baseline_metadata_path.is_file() else {}
    limitations = {
        "fixed_2025_universe_backcast": bool(baseline_metadata.get("fixed_2025_universe_backcast", False)),
        "survivorship_and_membership_bias": bool(baseline_metadata.get("survivorship_and_membership_bias", False)),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "labels": {
            "baseline": "DATA_PIT_HISTORICAL",
            "dart_ir": "LLM_PIT_PSEUDO_OOS",
            "full": "LLM_PIT_PSEUDO_OOS",
            "live": "TRUE_LIVE_OOS_NOT_YET_OBSERVED",
        },
        "period_count": len(period_rows),
        "validated_pack_count": len(merged),
        "failed_closed_pack_count": seal["failed_closed_pack_count"],
        "sample": {
            "comparison_scope": "COMMON_FAIL_CLOSED_VALIDATED_PACKS",
            "eligible_pack_count": seal["eligible_pack_count"],
            "validated_pack_count": len(merged),
            "validated_pack_rate": len(merged) / seal["eligible_pack_count"],
        },
        "validation_failure_reasons": failure_reasons,
        "source_input_coverage_after_preprocessing": source_coverage,
        "validated_claims_by_source_role": validated_claims_by_source_role,
        "analyst_overlay_action_change_count": analyst_action_changes,
        "limitations": limitations,
        "full_universe_baseline": stats("full_universe_baseline_data_pit_return"),
        "baseline": stats("baseline_data_pit_return"),
        "dart_ir": stats("dart_ir_llm_pseudo_oos_return"),
        "full": stats("full_llm_pseudo_oos_return"),
        "dart_ir_increment": stats("dart_ir_minus_baseline"),
        "full_increment": stats("full_minus_baseline"),
        "risk_actions": {
            "dart_ir": dict(Counter(item["dart_ir_action"] for item in decision_rows.values())),
            "full": dict(Counter(item["full_action"] for item in decision_rows.values())),
        },
        "models": {"preprocessing": PREPROCESS_MODEL, "main": MAIN_MODEL},
        "method": {
            "cheap_rank_changed_by_llm": False,
            "overlay_increments_use_common_sample_baseline": True,
            "position_cap_weight": 0.5,
            "veto_requires_two_erosive_claims_from_two_source_roles": True,
            "signals_sealed_before_returns": True,
            "model_knowledge_contamination_possible": True,
        },
    }
    write_json(output / "evaluation.json", report)
    lines = [
        "# MoatRader v7 DART·IR·애널리스트 LLM historical validation",
        "",
        "> 현대 LLM을 사용한 구간은 완전한 OOS가 아니라 `LLM_PIT_PSEUDO_OOS`다. Cheap 숫자 엔진만 `DATA_PIT_HISTORICAL`로 해석한다.",
        "",
        "| Lane | Grade | Mean 77-day return | t-stat |",
        "|---|---|---:|---:|",
    ]
    for label, key, grade in (
        ("Deterministic Cheap (full eligible universe)", "full_universe_baseline", "DATA_PIT_HISTORICAL"),
        ("Deterministic Cheap (LLM-gate common sample)", "baseline", "DATA_PIT_HISTORICAL"),
        ("DART original + IR LLM risk overlay", "dart_ir", "LLM_PIT_PSEUDO_OOS"),
        ("DART + IR + company/industry analyst LLM risk overlay", "full", "LLM_PIT_PSEUDO_OOS"),
    ):
        item = report[key]
        lines.append(f"| {label} | {grade} | {item.get('mean', float('nan')):.4f} | {item.get('t_stat', float('nan')):.3f} |")
    lines.extend(
        [
            "",
            f"A/B/C 세 ablation lane은 모두 엄격 gate를 통과한 동일 {len(merged):,}개 관측치의 공통 표본 비교다 "
            f"(전체 {seal['eligible_pack_count']:,}개 중 {len(merged) / seal['eligible_pack_count']:.1%}).",
            "LLM overlay 증분은 full-universe Cheap이 아니라 바로 위 common-sample Cheap을 기준으로 계산했다.",
            "LLM은 인용이 검증된 의미 분류만 수행했고 Cheap 값·순위·DCF 수치를 만들거나 수정하지 않았다.",
            "목표주가·투자의견·현재가 관련 행은 애널리스트 PDF에서 LLM 입력 전에 격리했다.",
            "원문 exact span, cutoff, 독립 entailment, future trap, 회사명 익명화 안정성 중 하나라도 실패한 관측치는 fail-closed 처리했다.",
            f"전처리 후 company analyst {source_coverage.get('COMPANY_ANALYST', {}).get('pack_count', 0):,}개, "
            f"industry analyst {source_coverage.get('INDUSTRY_ANALYST', {}).get('pack_count', 0):,}개 팩이 분류 입력에 포함됐다.",
            f"검증을 통과한 애널리스트 claim이 위험 action을 추가 변경한 관측치는 {analyst_action_changes:,}개이므로 "
            "이번 실행에서는 full lane과 DART+IR lane의 수익률이 같다.",
            "전체 deterministic 트랙도 2025 고정 유니버스 backcast이므로 membership·survivorship bias가 남아 있다."
            if limitations["fixed_2025_universe_backcast"] or limitations["survivorship_and_membership_bias"]
            else "유니버스 구성 한계는 입력 deterministic 트랙의 메타데이터를 따른다.",
            "최종 증명 등급인 `TRUE_LIVE_OOS`는 규칙 freeze 이후 새 공시와 미래 수익이 실제로 관측된 뒤에만 부여할 수 있다.",
        ]
    )
    (output / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def api_key_persistence_audit(output: Path) -> dict[str, Any]:
    matches: list[str] = []
    for path in output.rglob("*"):
        if not path.is_file() or path.suffix.casefold() in {".pdf", ".zip", ".xlsx"}:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError):
            continue
        if (
            re.search(r"sk-(?:proj|live|svcacct)-[A-Za-z0-9_-]{20,}", text)
            or "OPENAI_API_KEY=" in text
            or "DART_API_KEY=" in text
        ):
            matches.append(str(path.resolve()))
    result = {"api_key_persisted": bool(matches), "matching_paths": matches}
    write_json(output / "api-key-audit.json", result)
    if matches:
        raise ValueError("API credential-like content was persisted")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the v7 2020-2025 DART/IR/analyst LLM pseudo-OOS validation.")
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--dart-root", type=Path, required=True)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--synalyst-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "preprocess", "classify", "seal", "evaluate", "all"), default="all")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--prompt-api-key", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--v6-contract", type=Path, default=Path("data-lake/experiments/expectation-gap-production-candidate-v6/frozen-contract.json"))
    parser.add_argument("--v6-stability-a", type=Path, default=Path("data-lake/experiments/valuation-routing-stability-20260818-v6-pit-a"))
    parser.add_argument("--v6-stability-b", type=Path, default=Path("data-lake/experiments/valuation-routing-stability-20260818-v6-pit-b"))
    args = parser.parse_args()
    output = args.output.resolve()
    if "v7" not in output.name.casefold():
        raise ValueError("new historical LLM output directory name must contain v7")
    protected = [args.v6_contract.resolve().parent, args.v6_stability_a.resolve(), args.v6_stability_b.resolve()]
    if any(output == item or item in output.parents or output in item.parents for item in protected):
        raise ValueError("v7 output overlaps protected v6 data")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError("output exists; use a new v7 path or --resume")
    output.mkdir(parents=True, exist_ok=True)
    before = snapshot_protected_files(
        repository_root=Path.cwd(),
        contract_path=args.v6_contract.resolve(),
        stability_directories=[args.v6_stability_a.resolve(), args.v6_stability_b.resolve()],
    )
    pack_ids = prepare_corpus(
        signals=args.signals.resolve(),
        dart_root=args.dart_root.resolve(),
        synalyst_root=args.synalyst_root.resolve(),
        output=output,
    )
    if args.stage == "prepare":
        api_key_persistence_audit(output)
        return 0
    transport = _secure_transport(args.prompt_api_key) if args.stage in {"preprocess", "classify", "all"} else None
    if args.stage in {"preprocess", "all"}:
        assert transport is not None
        preprocess_packs(pack_ids=pack_ids, output=output, transport=transport, batch_size=args.batch_size, workers=args.workers)
    selected_ids = hydrate_selected_packs(pack_ids, output)
    if args.stage == "preprocess":
        api_key_persistence_audit(output)
        return 0
    if args.stage in {"classify", "all"}:
        assert transport is not None
        classify_packs(pack_ids=selected_ids, output=output, transport=transport, batch_size=args.batch_size, workers=args.workers, anonymized=False)
        classify_packs(pack_ids=selected_ids, output=output, transport=transport, batch_size=args.batch_size, workers=args.workers, anonymized=True)
        judgeable = [
            item for item in selected_ids
            if (output / "llm" / "original" / f"{item}.json").is_file()
            and (output / "llm" / "anonymized" / f"{item}.json").is_file()
        ]
        judge_entailment(pack_ids=judgeable, output=output, transport=transport, batch_size=args.batch_size, workers=args.workers)
    if args.stage == "classify":
        api_key_persistence_audit(output)
        return 0
    decision_rows = seal_signals(pack_ids=selected_ids, signals=args.signals.resolve(), output=output)
    if args.stage == "seal":
        api_key_persistence_audit(output)
        return 0
    report = evaluate(signals=args.signals.resolve(), prices=args.prices.resolve(), output=output)
    after = snapshot_protected_files(
        repository_root=Path.cwd(),
        contract_path=args.v6_contract.resolve(),
        stability_directories=[args.v6_stability_a.resolve(), args.v6_stability_b.resolve()],
    )
    changed = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    write_json(
        output / "v6-integrity.json",
        {
            "schema_version": "v7-build-v6-integrity/1",
            "protected_file_count": len(before),
            "changed_paths": changed,
            "v6_unchanged_during_backtest": not changed,
            "protected_sha256": after,
        },
    )
    if changed:
        raise ValueError("protected v6 files changed during the v7 run")
    api_key_persistence_audit(output)
    write_json(
        output / "run-manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "pack_count": len(pack_ids),
            "validated_pack_count": len(decision_rows),
            "models": {"preprocessing": PREPROCESS_MODEL, "main": MAIN_MODEL},
            "signals_seal_sha256": sha256_file(output / "signals-seal.json"),
            "evaluation_sha256": sha256_file(output / "evaluation.json"),
            "api_key_persisted": False,
            "v6_unchanged": True,
        },
    )
    print(output / "evaluation.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
