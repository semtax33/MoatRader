from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from lxml import etree, html

from moatrader.evidence.historical_corpus import (
    excerpts_from_text,
    pdf_excerpts,
    unique_excerpts,
)
from moatrader.evidence.historical_overlay import (
    HistoricalAssessmentBatch,
    HistoricalEntailmentBatch,
    HistoricalEntailment,
    HistoricalExcerpt,
    HistoricalPackAssessment,
    HistoricalPreprocessBatch,
    HistoricalSourceRole,
    HistoricalTrapAnswer,
    anonymize_text,
    deterministic_overlay_decision,
    validate_historical_assessment,
)
from moatrader.financial.dcf import DcfAssumptions, DcfEngine
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
from moatrader.llm.transport import OpenAIResponsesTransport


SEOUL = ZoneInfo("Asia/Seoul")
SCHEMA_VERSION = "kr-all-current-reports/1"
PREPROCESS_MODEL = "gpt-5-nano"
MAIN_MODEL = "gpt-5.6-luna"
MAX_SOURCE_AGE_DAYS = 550
MAX_DART_EXCERPTS_PER_DOCUMENT = 12
MAX_COMPANY_ANALYST_EXCERPTS = 8
MAX_IR_EXCERPTS = 8
MAX_INDUSTRY_EXCERPTS = 6

# Stable readable labels are used because the archived Hankyung metadata has
# replacement-character mojibake in several Korean name fields.  Codes remain
# the immutable provider identifiers and unknown codes remain available.
INDUSTRY_CODE_LABELS = {
    "001": "거시경제·시장전략",
    "005": "음식료·담배",
    "006": "섬유·의류 OEM",
    "008": "화학",
    "011": "철강·금속",
    "012": "기계·로봇·방산·항공",
    "013": "전자부품·디스플레이",
    "014": "제지·포장",
    "015": "글로벌 자동차·전기차",
    "016": "유통·소매",
    "017": "전력·유틸리티",
    "018": "건설·부동산",
    "020": "통신",
    "021": "금융지주·종합금융",
    "022": "은행",
    "024": "증권",
    "025": "보험",
    "026": "건설자재·인테리어",
    "027": "소비재·이커머스",
    "029": "운송·물류·항공",
    "031": "미디어·광고",
    "037": "게임·엔터테인먼트",
    "041": "중국 소비·시장",
    "042": "IT 산업",
    "043": "IT 하드웨어·소비자전자",
    "056": "소비재",
    "058": "뷰티·화장품 유통",
    "065": "정유·석유가스",
    "066": "제약·바이오·헬스케어",
    "068": "원자재·귀금속",
    "070": "로봇·자동화",
    "072": "전자소재·부품",
    "074": "미용의료·헬스케어",
    "075": "자동차·모빌리티",
    "077": "화장품",
    "152": "조선·해양",
    "153": "인터넷·소프트웨어·게임",
    "154": "에너지·전력 인프라",
    "155": "정보보안",
    "156": "콘텐츠·플랫폼",
    "157": "화장품 제조",
    "158": "자동차·자동차부품",
    "159": "반도체",
    "160": "반도체·IT 장비",
    "184": "시장 테마·종목 전략",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _metadata_datetime(metadata: dict[str, Any], key: str = "available_at") -> datetime:
    value = str(metadata.get(key) or "").strip().replace("Z", "+00:00")
    if not value:
        raise ValueError(f"metadata is missing {key}")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"metadata {key} must be timezone-aware")
    return parsed


def _dart_document_text(path: Path) -> str:
    """Extract stable human-visible blocks while retaining the original file."""

    content = path.read_bytes()
    parser = etree.XMLParser(recover=True, huge_tree=True, resolve_entities=False, no_network=True)
    try:
        root = etree.fromstring(content, parser=parser)
    except etree.XMLSyntaxError:
        root = html.fromstring(content)
    blocks: list[str] = []
    # Keep leaf-ish layout blocks only. Walking DIV/SECTION and every TD would
    # repeatedly traverse the same large subtree and turn multi-megabyte DART
    # filings into near-quadratic work. TR retains a whole table row once.
    block_tags = {"P", "TITLE", "TR", "LI", "PRE"}
    for node in root.iter():
        tag = str(node.tag).rsplit("}", 1)[-1].upper() if isinstance(node.tag, str) else ""
        if tag not in block_tags:
            continue
        text = " ".join(" ".join(node.itertext()).split())
        if len(text) >= 40:
            blocks.append(text)
    if not blocks:
        text = " ".join(" ".join(root.itertext()).split())
        if text:
            blocks.append(text)
    # Nested DART tags duplicate paragraphs. Exact normalized text is retained
    # once so the citation span remains deterministic.
    return "\n\n".join(dict.fromkeys(blocks))


def _excerpt_payload(item: HistoricalExcerpt) -> dict[str, Any]:
    return item.model_dump(mode="json")


def _load_pack(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    payload["excerpts"] = [HistoricalExcerpt.model_validate(item) for item in payload["excerpts"]]
    return payload


def _report_catalog(
    root: Path,
    *,
    cutoff: datetime,
) -> tuple[
    dict[str, list[tuple[HankyungCompanyReport, Path]]],
    dict[str, list[tuple[HankyungIndustryReport, Path]]],
    dict[str, str],
]:
    companies: dict[str, list[tuple[HankyungCompanyReport, Path]]] = defaultdict(list)
    industries: dict[str, list[tuple[HankyungIndustryReport, Path]]] = defaultdict(list)
    taxonomy: dict[str, str] = {}
    begin_year = (cutoff - timedelta(days=MAX_SOURCE_AGE_DAYS)).year
    for year in range(begin_year, cutoff.year + 1):
        company_base = root / "company" / str(year)
        company_metadata = company_base / "json" / "reports.json"
        if company_metadata.is_file():
            files = {item.name.partition("_")[0]: item for item in (company_base / "pdf").glob("*.pdf")}
            for report in load_hankyung_company_reports(company_metadata).values():
                path = files.get(report.report_id)
                if path is not None and report.registered_at <= cutoff:
                    companies[report.ticker].append((report, path))
        industry_base = root / "industry" / str(year)
        industry_metadata = industry_base / "json" / "reports.json"
        if industry_metadata.is_file():
            files = {item.name.partition("_")[0]: item for item in (industry_base / "pdf").glob("*.pdf")}
            for report in load_hankyung_industry_reports(industry_metadata).values():
                path = files.get(report.report_id)
                if path is None or report.registered_at > cutoff:
                    continue
                industries[report.industry_code].append((report, path))
                taxonomy[report.industry_code] = INDUSTRY_CODE_LABELS.get(
                    report.industry_code,
                    f"한경 산업분류 {report.industry_code}",
                )
    for values in companies.values():
        values.sort(key=lambda item: (item[0].registered_at, item[0].report_id))
    for values in industries.values():
        values.sort(key=lambda item: (item[0].registered_at, item[0].report_id))
    return companies, industries, dict(sorted(taxonomy.items()))


def _latest(
    values: list[tuple[Any, Path]],
    *,
    cutoff: datetime,
    timestamp: Any,
) -> tuple[Any, Path] | None:
    eligible = [item for item in values if timestamp(item[0]) <= cutoff]
    if not eligible:
        return None
    selected = eligible[-1]
    if cutoff - timestamp(selected[0]) > timedelta(days=MAX_SOURCE_AGE_DAYS):
        return None
    return selected


def _ir_catalog(path: Path | None, *, cutoff: datetime) -> dict[str, list[dict[str, str]]]:
    result: dict[str, list[dict[str, str]]] = defaultdict(list)
    if path is None or not path.is_file():
        return result
    base = path.resolve().parent
    for row in read_csv(path):
        if str(row.get("source") or "").strip().upper() != "IR":
            continue
        metadata_path = Path(row["metadata"])
        metadata_path = metadata_path.resolve() if metadata_path.is_absolute() else (base / metadata_path).resolve()
        input_path = Path(row["input"])
        input_path = input_path.resolve() if input_path.is_absolute() else (base / input_path).resolve()
        row = {**row, "metadata": str(metadata_path), "input": str(input_path)}
        metadata = read_json(metadata_path)
        available_at = _metadata_datetime(metadata)
        if available_at <= cutoff and cutoff - available_at <= timedelta(days=MAX_SOURCE_AGE_DAYS):
            result[str(row["ticker"]).zfill(6)].append(row)
    for rows in result.values():
        rows.sort(key=lambda row: _metadata_datetime(read_json(Path(row["metadata"]).resolve())))
    return result


def prepare(
    *,
    universe_path: Path,
    dart_manifest_path: Path,
    ir_manifest_path: Path | None,
    synalyst_hankyung_root: Path,
    output: Path,
    cutoff: datetime,
) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    universe = read_csv(universe_path)
    universe_by_ticker = {str(row["stock_code"]).zfill(6): row for row in universe}
    dart_rows_by_security: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(dart_manifest_path):
        dart_rows_by_security[str(row["ticker"]).zfill(6)].append(row)
    company_reports, industry_reports, taxonomy = _report_catalog(
        synalyst_hankyung_root,
        cutoff=cutoff,
    )
    ir_by_ticker = _ir_catalog(ir_manifest_path, cutoff=cutoff)

    industry_catalog = {
        code: [
            {
                "document_id": report.source_document_id,
                "available_at": report.registered_at.isoformat(),
                "path": str(path.resolve()),
                "title": report.title,
                "raw_sha256": file_sha256(path),
            }
            for report, path in reports
        ]
        for code, reports in industry_reports.items()
    }
    write_json(output / "source-map" / "industry-taxonomy.json", taxonomy)
    write_json(output / "source-map" / "industry-catalog.json", industry_catalog)

    filing_groups: dict[str, dict[str, Any]] = {}
    security_map: list[dict[str, Any]] = []
    for ticker, source in universe_by_ticker.items():
        rows = dart_rows_by_security.get(ticker, [])
        filing_ticker = str(rows[0].get("filing_ticker") or ticker).zfill(6) if rows else ticker
        security_map.append(
            {
                "ticker": ticker,
                "name": source.get("name", ""),
                "market": source.get("market", ""),
                "security_type": source.get("security_type", ""),
                "filing_ticker": filing_ticker if rows else "",
                "pack_id": f"KR-{cutoff.date().isoformat()}-{filing_ticker}" if rows else "",
                "status": "PIT_DOCUMENTS_AVAILABLE" if rows else "NO_PERIODIC_PIT_FILING",
            }
        )
        if not rows:
            continue
        group = filing_groups.setdefault(
            filing_ticker,
            {
                "rows": rows,
                "issuer_name": str(rows[0].get("issuer_name") or source.get("name") or ""),
                "issuer_id": str(rows[0].get("issuer_id") or ""),
                "securities": [],
            },
        )
        group["securities"].append(ticker)

    pack_ids: list[str] = []
    source_assignments: list[dict[str, Any]] = []
    opinion_lines_removed = 0
    for index, (filing_ticker, group) in enumerate(sorted(filing_groups.items()), start=1):
        pack_id = f"KR-{cutoff.date().isoformat()}-{filing_ticker}"
        destination = output / "packs" / "candidates" / f"{pack_id}.json"
        if destination.is_file():
            existing = read_json(destination)
            source_assignments.append(
                existing.get(
                    "source_assignment",
                    {
                        "pack_id": pack_id,
                        "ticker": filing_ticker,
                        "security_tickers": sorted(group["securities"]),
                        "status": "REUSED_PREPARED_WITHOUT_LEGACY_SOURCE_ASSIGNMENT",
                    },
                )
            )
            pack_ids.append(pack_id)
            continue
        excerpts: list[HistoricalExcerpt] = []
        assignment: dict[str, Any] = {
            "pack_id": pack_id,
            "ticker": filing_ticker,
            "security_tickers": sorted(group["securities"]),
            "dart_documents": [],
        }
        seen_dart: set[str] = set()
        for row in group["rows"]:
            input_path = Path(row["input"]).resolve()
            metadata_path = Path(row["metadata"]).resolve()
            metadata = read_json(metadata_path)
            source_id = str(metadata.get("source_document_id") or input_path.stem)
            if source_id in seen_dart:
                continue
            seen_dart.add(source_id)
            available_at = _metadata_datetime(metadata)
            if available_at > cutoff:
                continue
            found, _ = excerpts_from_text(
                source_id=source_id,
                source_role=HistoricalSourceRole.DART_ORIGINAL,
                available_at=available_at,
                text=_dart_document_text(input_path),
                maximum=MAX_DART_EXCERPTS_PER_DOCUMENT,
            )
            excerpts.extend(found)
            assignment["dart_documents"].append(
                {
                    "source_id": source_id,
                    "available_at": available_at.isoformat(),
                    "input": str(input_path),
                    "metadata": str(metadata_path),
                    "raw_sha256": file_sha256(input_path),
                }
            )

        company = _latest(
            company_reports.get(filing_ticker, []),
            cutoff=cutoff,
            timestamp=lambda item: item.registered_at,
        )
        if company is not None:
            report, path = company
            found, removed = pdf_excerpts(
                path,
                source_id=report.source_document_id,
                source_role=HistoricalSourceRole.COMPANY_ANALYST,
                available_at=report.registered_at,
                maximum=MAX_COMPANY_ANALYST_EXCERPTS,
            )
            excerpts.extend(found)
            opinion_lines_removed += removed
            assignment["company_analyst"] = {
                "source_id": report.source_document_id,
                "available_at": report.registered_at.isoformat(),
                "path": str(path.resolve()),
                "raw_sha256": file_sha256(path),
                "market_opinion_lines_removed": removed,
            }

        ir_rows = ir_by_ticker.get(filing_ticker, [])
        if ir_rows:
            row = ir_rows[-1]
            path = Path(row["input"]).resolve()
            metadata_path = Path(row["metadata"]).resolve()
            metadata = read_json(metadata_path)
            available_at = _metadata_datetime(metadata)
            source_id = str(metadata.get("source_document_id") or path.stem)
            found, _ = pdf_excerpts(
                path,
                source_id=source_id,
                source_role=HistoricalSourceRole.IR,
                available_at=available_at,
                maximum=MAX_IR_EXCERPTS,
            )
            excerpts.extend(found)
            assignment["ir"] = {
                "source_id": source_id,
                "available_at": available_at.isoformat(),
                "path": str(path),
                "metadata": str(metadata_path),
                "raw_sha256": file_sha256(path),
            }

        excerpts = unique_excerpts([excerpts])
        if not excerpts:
            assignment["status"] = "NO_EXTRACTABLE_CUTOFF_EVIDENCE"
            source_assignments.append(assignment)
            continue
        assignment["status"] = "PREPARED"
        assignment["candidate_excerpt_count"] = len(excerpts)
        pack = {
            "schema_version": SCHEMA_VERSION,
            "pack_id": pack_id,
            "signal_date": cutoff.date().isoformat(),
            "cutoff": cutoff.isoformat(),
            "ticker": filing_ticker,
            "issuer_id": group["issuer_id"],
            "issuer_name": group["issuer_name"],
            "security_tickers": sorted(group["securities"]),
            # Compatibility-only fields used by the historical stage helpers;
            # current reports do not expose a Cheap rank or let LLM alter one.
            "cheap": None,
            "cheap_rank": None,
            "excerpts": [_excerpt_payload(item) for item in excerpts],
            "source_assignment": assignment,
        }
        write_json(destination, pack)
        source_assignments.append(assignment)
        pack_ids.append(pack_id)
        if index % 100 == 0:
            print(f"prepared {index}/{len(filing_groups)} issuer packs", flush=True)

    write_json(output / "source-map" / "assignments.json", source_assignments)
    write_json(output / "security-map.json", security_map)
    write_json(
        output / "corpus-manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "cutoff": cutoff.isoformat(),
            "universe_count": len(universe),
            "issuer_pack_count": len(pack_ids),
            "pack_ids": pack_ids,
            "market_opinion_lines_removed": opinion_lines_removed,
            "source_max_age_days": MAX_SOURCE_AGE_DAYS,
            "models": {"preprocess": PREPROCESS_MODEL, "main": MAIN_MODEL},
            "ranking_policy": "NO_LLM_RANK_SIGNAL",
        },
    )
    return pack_ids


def _batches(values: list[str], size: int) -> list[list[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def deterministic_current_units(
    excerpts: list[HistoricalExcerpt],
    *,
    maximum: int = 16,
) -> list[HistoricalExcerpt]:
    """Freeze a DART-first source-balanced evidence set before main-model use."""

    role_caps = {
        HistoricalSourceRole.DART_ORIGINAL: 12,
        HistoricalSourceRole.IR: 2,
        HistoricalSourceRole.COMPANY_ANALYST: 2,
    }
    selected: list[HistoricalExcerpt] = []
    selected_ids: set[str] = set()
    for role, cap in role_caps.items():
        for item in (value for value in excerpts if value.source_role == role):
            if sum(value.source_role == role for value in selected) >= cap:
                break
            selected.append(item)
            selected_ids.add(item.unit_id)
    for item in excerpts:
        if len(selected) >= maximum:
            break
        if item.unit_id not in selected_ids:
            selected.append(item)
            selected_ids.add(item.unit_id)
    return selected[:maximum]


def deterministic_industry_codes(
    excerpts: list[HistoricalExcerpt],
    *,
    allowed_codes: set[str],
) -> list[str]:
    """Map only explicit cutoff text to provider codes; no issuer prior is used."""

    text = "\n".join(item.text for item in excerpts).casefold()
    rules: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("159", ("반도체", "메모리", "wafer", "semiconductor")),
        ("066", ("제약", "바이오", "의약품", "신약", "헬스케어")),
        ("152", ("조선", "선박", "해양플랜트")),
        ("158", ("자동차", "차량", "모빌리티", "전기차")),
        ("153", ("소프트웨어", "인터넷", "게임", "플랫폼", "클라우드")),
        ("022", ("은행", "예대마진", "여신", "수신")),
        ("024", ("증권", "브로커리지", "investment banking")),
        ("025", ("보험", "보험료", "손해율")),
        ("005", ("식품", "음료", "담배")),
        ("011", ("철강", "비철금속", "제련")),
        ("008", ("화학", "석유화학", "합성수지")),
        ("018", ("건설", "주택사업", "부동산개발")),
        ("020", ("통신", "이동전화", "5g")),
        ("029", ("운송", "물류", "항공운송", "해운")),
        ("157", ("화장품", "코스메틱")),
        ("012", ("기계", "로봇", "방산", "항공우주")),
        ("013", ("전자부품", "디스플레이", "인쇄회로", "pcb")),
        ("154", ("에너지", "발전", "전력망", "원유", "천연가스")),
        ("016", ("유통", "백화점", "소매", "편의점")),
        ("037", ("엔터테인먼트", "방송", "음원", "콘텐츠")),
    )
    scored: list[tuple[int, int, str]] = []
    for priority, (code, terms) in enumerate(rules):
        if code not in allowed_codes:
            continue
        hits = sum(text.count(term.casefold()) for term in terms)
        if hits:
            scored.append((hits, -priority, code))
    return [item[2] for item in sorted(scored, reverse=True)[:2]]


def _transport() -> OpenAIResponsesTransport:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("OPENAI_API_KEY is required for the llm stage")
    return OpenAIResponsesTransport(
        summary_model=PREPROCESS_MODEL,
        moat_model=MAIN_MODEL,
        summary_reasoning_effort="low",
        atomic_reasoning_effort="medium",
        moat_reasoning_effort="medium",
        max_output_tokens=8_000,
        max_retries=4,
        timeout_seconds=240,
    )


def _execute_batches(
    *,
    ids: list[str],
    batch_size: int,
    workers: int,
    execute: Any,
) -> Iterable[tuple[list[str], Any | None, Exception | None]]:
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(execute, batch): batch for batch in _batches(ids, batch_size)}
        for future in concurrent.futures.as_completed(futures):
            batch = futures[future]
            try:
                yield batch, future.result(), None
            except Exception as exc:
                yield batch, None, exc


def preprocess(*, pack_ids: list[str], output: Path, transport: OpenAIResponsesTransport, batch_size: int, workers: int) -> None:
    from moatrader.evidence.historical_overlay import sanitize_preprocess_selection

    taxonomy = read_json(output / "source-map" / "industry-taxonomy.json")
    pending = [item for item in pack_ids if not (output / "llm" / "preprocess" / f"{item}.json").is_file()]
    usage: Counter[str] = Counter()

    def execute(ids: list[str]) -> tuple[Any, Any]:
        packs = [_load_pack(output / "packs" / "candidates" / f"{item}.json") for item in ids]
        request = build_historical_preprocess_request(packs, industry_taxonomy=taxonomy)
        return request, transport.execute(request, HistoricalPreprocessBatch)

    completed = 0
    for ids, execution, execution_error in _execute_batches(
        ids=pending, batch_size=batch_size, workers=workers, execute=execute
    ):
        if execution_error is not None:
            write_json(
                output / "llm" / "failures" / f"preprocess-{ids[0]}.json",
                {"pack_ids": ids, "error": f"{type(execution_error).__name__}: {execution_error}"},
            )
            completed += len(ids)
            print(f"nano preprocessed {completed}/{len(pending)} pending packs", flush=True)
            continue
        assert execution is not None
        request, result = execution
        try:
            by_id = {item.pack_id: item for item in result.parsed.packs}
            if set(by_id) - set(ids):
                raise ValueError("preprocess invented pack IDs")
            for identifier in ids:
                selection = by_id.get(identifier)
                if selection is None:
                    continue
                pack = _load_pack(output / "packs" / "candidates" / f"{identifier}.json")
                selection = sanitize_preprocess_selection(
                    selection,
                    pack["excerpts"],
                    allowed_industry_codes=set(taxonomy),
                )
                write_json(
                    output / "llm" / "preprocess" / f"{identifier}.json",
                    {
                        **selection.model_dump(mode="json"),
                        "model": result.model,
                        "response_id": result.response_id,
                        "request_sha256": request.input_sha256,
                    },
                )
            missing = sorted(set(ids) - set(by_id))
            if missing:
                write_json(
                    output / "llm" / "failures" / f"preprocess-partial-{ids[0]}.json",
                    {
                        "pack_ids": ids,
                        "missing_pack_ids": missing,
                        "error": "PARTIAL_RESPONSE_RETRY_REQUIRED",
                    },
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


def hydrate(pack_ids: list[str], output: Path) -> list[str]:
    from moatrader.evidence.historical_overlay import validate_preprocess_selection

    catalog = read_json(output / "source-map" / "industry-catalog.json")
    successful: list[str] = []
    pdf_cache: dict[str, list[HistoricalExcerpt]] = {}
    for identifier in pack_ids:
        destination = output / "packs" / "selected" / f"{identifier}.json"
        if destination.is_file():
            successful.append(identifier)
            continue
        selection_path = output / "llm" / "preprocess" / f"{identifier}.json"
        if not selection_path.is_file():
            continue
        pack = _load_pack(output / "packs" / "candidates" / f"{identifier}.json")
        raw = read_json(selection_path)
        selection = HistoricalPreprocessBatch.model_validate(
            {"packs": [{key: raw[key] for key in ("pack_id", "selected_unit_ids", "industry_codes")} ]}
        ).packs[0]
        model_selected = validate_preprocess_selection(selection, pack["excerpts"])
        base_selected = deterministic_current_units(pack["excerpts"])
        excerpts = list(base_selected)
        deterministic_codes = deterministic_industry_codes(
            pack["excerpts"],
            allowed_codes=set(catalog),
        )
        cutoff = datetime.fromisoformat(pack["cutoff"])
        for code in deterministic_codes:
            candidates = [
                item for item in catalog.get(code, [])
                if datetime.fromisoformat(item["available_at"]) <= cutoff
                and cutoff - datetime.fromisoformat(item["available_at"]) <= timedelta(days=MAX_SOURCE_AGE_DAYS)
            ]
            if not candidates:
                continue
            selected = candidates[-1]
            source_id = selected["document_id"]
            if source_id not in pdf_cache:
                found, _ = pdf_excerpts(
                    Path(selected["path"]),
                    source_id=source_id,
                    source_role=HistoricalSourceRole.INDUSTRY_ANALYST,
                    available_at=datetime.fromisoformat(selected["available_at"]),
                    maximum=MAX_INDUSTRY_EXCERPTS,
                )
                pdf_cache[source_id] = found
            excerpts.extend(pdf_cache[source_id])
        excerpts = unique_excerpts([excerpts])
        pack["excerpts"] = [_excerpt_payload(item) for item in excerpts]
        pack["industry_codes"] = deterministic_codes
        pack["preprocess_audit"] = {
            "schema_version": "current-preprocess-audit/1",
            "model": PREPROCESS_MODEL,
            "model_selected_unit_ids": [item.unit_id for item in model_selected],
            "deterministic_selected_unit_ids": [item.unit_id for item in base_selected],
            "hydrated_industry_unit_ids": [
                item.unit_id
                for item in excerpts
                if item.source_role == HistoricalSourceRole.INDUSTRY_ANALYST
            ],
            "model_suggested_industry_codes": selection.industry_codes,
            "deterministic_industry_codes": deterministic_codes,
            "model_unit_selection_controls_main_evidence": False,
            "model_industry_suggestion_controls_industry_evidence": False,
            "selection_policy": "DART_FIRST_SOURCE_BALANCED_AND_EXPLICIT_KEYWORD_ROUTING",
        }
        pack["anonymized_text_by_unit"] = {
            item.unit_id: anonymize_text(item.text, pack["issuer_name"], pack["ticker"])
            for item in excerpts
        }
        write_json(destination, pack)
        successful.append(identifier)
    return successful


def classify(
    *,
    pack_ids: list[str],
    output: Path,
    transport: OpenAIResponsesTransport,
    batch_size: int,
    workers: int,
    anonymized: bool,
    vote: int | None = None,
) -> None:
    lane = "anonymized" if anonymized else "original"
    lane_directory = (
        output / "llm" / f"{lane}-votes" / f"vote-{vote:02d}"
        if vote is not None
        else output / "llm" / lane
    )
    lane_label = f"{lane}-vote-{vote:02d}" if vote is not None else lane
    pending = [item for item in pack_ids if not (lane_directory / f"{item}.json").is_file()]
    usage: Counter[str] = Counter()

    def execute(ids: list[str]) -> tuple[Any, Any]:
        packs = [_load_pack(output / "packs" / "selected" / f"{item}.json") for item in ids]
        for pack in packs:
            pack["anonymized_text_by_unit"] = read_json(
                output / "packs" / "selected" / f"{pack['pack_id']}.json"
            )["anonymized_text_by_unit"]
        request = build_historical_evidence_request(packs, anonymized=anonymized)
        return request, transport.execute(request, HistoricalAssessmentBatch)

    completed = 0
    for ids, execution, execution_error in _execute_batches(
        ids=pending, batch_size=batch_size, workers=workers, execute=execute
    ):
        if execution_error is not None:
            write_json(
                output / "llm" / "failures" / f"{lane_label}-{ids[0]}.json",
                {"pack_ids": ids, "error": f"{type(execution_error).__name__}: {execution_error}"},
            )
            completed += len(ids)
            print(f"Luna {lane_label} {completed}/{len(pending)} pending packs", flush=True)
            continue
        assert execution is not None
        request, result = execution
        try:
            by_id = {item.pack_id: item for item in result.parsed.packs}
            if set(by_id) - set(ids):
                raise ValueError(f"{lane} invented pack IDs")
            for identifier in ids:
                assessment = by_id.get(identifier)
                if assessment is None:
                    continue
                assessment = assessment.model_copy(
                    update={
                        "claims": [
                            claim.model_copy(update={"judgment_id": f"{identifier}:J{index:02d}"})
                            for index, claim in enumerate(assessment.claims, start=1)
                        ]
                    }
                )
                write_json(
                    lane_directory / f"{identifier}.json",
                    {
                        **assessment.model_dump(mode="json"),
                        "model": result.model,
                        "response_id": result.response_id,
                        "request_sha256": request.input_sha256,
                    },
                )
            missing = sorted(set(ids) - set(by_id))
            if missing:
                write_json(
                    output / "llm" / "failures" / f"{lane_label}-partial-{ids[0]}.json",
                    {
                        "pack_ids": ids,
                        "missing_pack_ids": missing,
                        "error": "PARTIAL_RESPONSE_RETRY_REQUIRED",
                    },
                )
            usage.update(result.usage.model_dump())
        except Exception as exc:
            write_json(
                output / "llm" / "failures" / f"{lane_label}-{ids[0]}.json",
                {"pack_ids": ids, "error": f"{type(exc).__name__}: {exc}"},
            )
        completed += len(ids)
        print(f"Luna {lane_label} {completed}/{len(pending)} pending packs", flush=True)
    usage_name = f"{lane}-vote-{vote:02d}-usage.json" if vote is not None else f"{lane}-usage.json"
    write_json(output / "llm" / usage_name, dict(usage))


def _assessment(path: Path) -> HistoricalPackAssessment:
    payload = read_json(path)
    return HistoricalPackAssessment.model_validate(
        {key: payload[key] for key in ("pack_id", "claims", "future_trap_answer")}
    )


def consolidate_classification_votes(
    *,
    pack_ids: list[str],
    output: Path,
    anonymized: bool,
    votes: int = 2,
) -> None:
    lane = "anonymized" if anonymized else "original"
    for identifier in pack_ids:
        destination = output / "llm" / lane / f"{identifier}.json"
        if destination.is_file():
            continue
        paths = [
            output / "llm" / f"{lane}-votes" / f"vote-{vote:02d}" / f"{identifier}.json"
            for vote in range(1, votes + 1)
        ]
        if not all(path.is_file() for path in paths):
            continue
        assessments = [_assessment(path) for path in paths]
        future_trap_passed = all(
            item.future_trap_answer == HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE
            for item in assessments
        )
        stable_keys: set[tuple[str, str, str, str]] = set()
        if future_trap_passed:
            claim_maps = [
                {
                    (
                        claim.axis.value,
                        claim.direction.value,
                        claim.unit_id,
                        claim.exact_quote,
                    ): claim
                    for claim in assessment.claims
                }
                for assessment in assessments
            ]
            stable_keys = set(claim_maps[0])
            for mapping in claim_maps[1:]:
                stable_keys &= set(mapping)
            stable_claims = []
            for index, key in enumerate(sorted(stable_keys), start=1):
                candidates = [mapping[key] for mapping in claim_maps]
                chosen = min(candidates, key=lambda item: item.confidence)
                stable_claims.append(
                    chosen.model_copy(
                        update={
                            "judgment_id": f"{identifier}:J{index:02d}",
                            "confidence": min(item.confidence for item in candidates),
                        }
                    )
                )
            trap_answer = HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE
        else:
            stable_claims = []
            trap_answer = HistoricalTrapAnswer.CLAIMED_FUTURE_KNOWLEDGE
        consensus = HistoricalPackAssessment(
            pack_id=identifier,
            claims=stable_claims,
            future_trap_answer=trap_answer,
        )
        write_json(
            destination,
            {
                **consensus.model_dump(mode="json"),
                "model": MAIN_MODEL,
                "classification_consensus": {
                    "schema_version": "current-classification-consensus/1",
                    "vote_count": votes,
                    "policy": "EXACT_INTERSECTION_AXIS_DIRECTION_UNIT_QUOTE",
                    "vote_claim_counts": [len(item.claims) for item in assessments],
                    "stable_claim_count": len(stable_keys),
                    "future_trap_all_votes_passed": future_trap_passed,
                },
            },
        )


def validate_current_assessment(
    *,
    cutoff: datetime,
    excerpts: list[HistoricalExcerpt],
    original: HistoricalPackAssessment,
    anonymized: HistoricalPackAssessment,
    entailment: HistoricalEntailmentBatch,
    issuer_name: str = "",
    ticker: str = "",
    maximum_confidence_delta: float = 0.15,
) -> tuple[list[Any], dict[str, Any]]:
    """Fail closed per unstable claim while preserving the contamination audit."""

    if original.pack_id != anonymized.pack_id:
        raise ValueError("original/anonymized pack IDs differ")
    expected_trap = HistoricalTrapAnswer.UNKNOWN_FROM_CUTOFF_EVIDENCE
    if original.future_trap_answer != expected_trap:
        raise ValueError("original future-knowledge trap failed")
    if anonymized.future_trap_answer != expected_trap:
        raise ValueError("anonymized future-knowledge trap failed")

    original_by_signature: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
    anonymized_by_signature: dict[tuple[str, str, str, str], list[Any]] = defaultdict(list)
    for claim in original.claims:
        anonymous_quote = (
            anonymize_text(claim.exact_quote, issuer_name, ticker)
            if issuer_name or ticker
            else claim.exact_quote
        )
        original_by_signature[
            (claim.axis.value, claim.direction.value, claim.unit_id, anonymous_quote)
        ].append(claim)
    for claim in anonymized.claims:
        anonymized_by_signature[
            (claim.axis.value, claim.direction.value, claim.unit_id, claim.exact_quote)
        ].append(claim)
    stable_pairs: list[tuple[Any, Any]] = []
    for signature in sorted(set(original_by_signature) & set(anonymized_by_signature)):
        left = sorted(original_by_signature[signature], key=lambda item: item.confidence)
        right = sorted(anonymized_by_signature[signature], key=lambda item: item.confidence)
        for original_claim, anonymized_claim in zip(left, right):
            if abs(original_claim.confidence - anonymized_claim.confidence) <= maximum_confidence_delta:
                stable_pairs.append((original_claim, anonymized_claim))
    decisions = {item.judgment_id: item.verdict for item in entailment.decisions}
    if len(decisions) != len(entailment.decisions):
        raise ValueError("entailment decision IDs must be unique")
    excerpts_by_id = {item.unit_id: item for item in excerpts}
    grounded_pairs: list[tuple[Any, Any]] = []
    discarded_entailment = 0
    discarded_grounding = 0
    for original_claim, anonymized_claim in stable_pairs:
        if original_claim.direction.value in {"UNKNOWN", "MIXED"}:
            discarded_grounding += 1
            continue
        if decisions.get(original_claim.judgment_id) != HistoricalEntailment.ENTAILED:
            discarded_entailment += 1
            continue
        excerpt = excerpts_by_id.get(original_claim.unit_id)
        if (
            excerpt is None
            or excerpt.available_at > cutoff
            or excerpt.text.find(original_claim.exact_quote) < 0
        ):
            discarded_grounding += 1
            continue
        grounded_pairs.append((original_claim, anonymized_claim))
    stable_original = [item[0] for item in stable_pairs]
    stable_anonymized = [item[1] for item in stable_pairs]
    filtered_original = original.model_copy(update={"claims": [item[0] for item in grounded_pairs]})
    filtered_anonymized = anonymized.model_copy(update={"claims": [item[1] for item in grounded_pairs]})
    claims = validate_historical_assessment(
        cutoff=cutoff,
        excerpts=excerpts,
        original=filtered_original,
        anonymized=filtered_anonymized,
        entailment=entailment,
        maximum_confidence_delta=maximum_confidence_delta,
    )
    instability = len(stable_original) != len(original.claims) or len(stable_anonymized) != len(
        anonymized.claims
    )
    return claims, {
        "schema_version": "current-anonymization-audit/1",
        "original_claim_count": len(original.claims),
        "anonymized_claim_count": len(anonymized.claims),
        "stable_claim_count": len(stable_original),
        "discarded_original_claim_count": len(original.claims) - len(stable_original),
        "discarded_anonymized_claim_count": len(anonymized.claims) - len(stable_anonymized),
        "discarded_entailment_claim_count": discarded_entailment,
        "discarded_grounding_claim_count": discarded_grounding,
        "validated_claim_count": len(claims),
        "anonymization_instability_detected": instability,
        "unstable_claims_fail_closed": True,
        "unentailed_claims_fail_closed": True,
        "ungrounded_claims_fail_closed": True,
        "maximum_confidence_delta": maximum_confidence_delta,
    }


def entail(
    *,
    pack_ids: list[str],
    output: Path,
    transport: OpenAIResponsesTransport,
    batch_size: int,
    workers: int,
) -> None:
    pending: list[str] = []
    for identifier in pack_ids:
        destination = output / "llm" / "entailment" / f"{identifier}.json"
        if destination.is_file():
            continue
        assessment = _assessment(output / "llm" / "original" / f"{identifier}.json")
        if assessment.claims:
            pending.append(identifier)
        else:
            write_json(destination, {"decisions": [], "model": "DETERMINISTIC_EMPTY_CLAIM_SET"})
    usage: Counter[str] = Counter()

    def execute(ids: list[str]) -> tuple[Any, Any, list[HistoricalPackAssessment]]:
        assessments = [_assessment(output / "llm" / "original" / f"{item}.json") for item in ids]
        excerpts = {
            item: _load_pack(output / "packs" / "selected" / f"{item}.json")["excerpts"]
            for item in ids
        }
        request = build_historical_entailment_request(assessments, excerpts)
        return request, transport.execute(request, HistoricalEntailmentBatch), assessments

    completed = 0
    for ids, execution, execution_error in _execute_batches(
        ids=pending, batch_size=batch_size, workers=workers, execute=execute
    ):
        if execution_error is not None:
            write_json(
                output / "llm" / "failures" / f"entailment-{ids[0]}.json",
                {"pack_ids": ids, "error": f"{type(execution_error).__name__}: {execution_error}"},
            )
            completed += len(ids)
            print(f"Luna entailment {completed}/{len(pending)} pending packs", flush=True)
            continue
        assert execution is not None
        request, result, assessments = execution
        try:
            expected = {claim.judgment_id for item in assessments for claim in item.claims}
            decisions = {item.judgment_id: item for item in result.parsed.decisions}
            if set(decisions) != expected:
                raise ValueError("entailment judgment IDs differ")
            for assessment in assessments:
                write_json(
                    output / "llm" / "entailment" / f"{assessment.pack_id}.json",
                    {
                        "decisions": [
                            decisions[claim.judgment_id].model_dump(mode="json")
                            for claim in assessment.claims
                        ],
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


def run_llm(*, output: Path, batch_size: int, workers: int) -> None:
    manifest = read_json(output / "corpus-manifest.json")
    pack_ids = list(manifest["pack_ids"])
    transport = _transport()
    for attempt in range(1, 6):
        preprocess(
            pack_ids=pack_ids,
            output=output,
            transport=transport,
            batch_size=batch_size if attempt == 1 else 1,
            workers=workers,
        )
        missing = [
            item
            for item in pack_ids
            if not (output / "llm" / "preprocess" / f"{item}.json").is_file()
        ]
        if not missing:
            break
        print(f"preprocess retry {attempt}: missing={len(missing)}", flush=True)
    if missing:
        raise RuntimeError(f"preprocess remained incomplete after retries: {len(missing)} packs")
    selected = hydrate(pack_ids, output)
    if set(selected) != set(pack_ids):
        raise RuntimeError(
            f"selected-pack hydration incomplete: expected={len(pack_ids)} actual={len(selected)}"
        )
    classification_votes = 2
    for anonymized in (False, True):
        lane = "anonymized" if anonymized else "original"
        for vote in range(1, classification_votes + 1):
            vote_directory = output / "llm" / f"{lane}-votes" / f"vote-{vote:02d}"
            for attempt in range(1, 6):
                classify(
                    pack_ids=selected,
                    output=output,
                    transport=transport,
                    batch_size=batch_size if attempt == 1 else 1,
                    workers=workers,
                    anonymized=anonymized,
                    vote=vote,
                )
                missing = [
                    item
                    for item in selected
                    if not (vote_directory / f"{item}.json").is_file()
                ]
                if not missing:
                    break
                print(
                    f"{lane} vote {vote} retry {attempt}: missing={len(missing)}",
                    flush=True,
                )
            if missing:
                raise RuntimeError(
                    f"{lane} vote {vote} remained incomplete after retries: {len(missing)} packs"
                )
        consolidate_classification_votes(
            pack_ids=selected,
            output=output,
            anonymized=anonymized,
            votes=classification_votes,
        )
        missing = [
            item
            for item in selected
            if not (output / "llm" / lane / f"{item}.json").is_file()
        ]
        if missing:
            raise RuntimeError(f"{lane} consensus incomplete: {len(missing)} packs")
    for attempt in range(1, 6):
        entail(
            pack_ids=selected,
            output=output,
            transport=transport,
            batch_size=batch_size if attempt == 1 else 1,
            workers=workers,
        )
        missing = [
            item
            for item in selected
            if not (output / "llm" / "entailment" / f"{item}.json").is_file()
        ]
        if not missing:
            break
        print(f"entailment retry {attempt}: missing={len(missing)}", flush=True)
    if missing:
        raise RuntimeError(f"entailment remained incomplete after retries: {len(missing)} packs")


def _dcf_payload(row: dict[str, str] | None) -> dict[str, Any]:
    if row is None or not str(row.get("dcf_assumptions") or "").strip():
        return {"status": "NOT_APPLICABLE_OR_UNAVAILABLE"}
    path = Path(row["dcf_assumptions"]).resolve()
    try:
        assumptions = DcfAssumptions.model_validate_json(path.read_text(encoding="utf-8-sig"))
        valuation = DcfEngine().value(assumptions)
        current_price = Decimal(str(row["current_price"]))
        fair_value = valuation.fair_value_per_share
        return {
            "status": "READY" if valuation.screening_eligible else "CALCULATED_NOT_SCREENING_ELIGIBLE",
            "calculation": "DETERMINISTIC_PYTHON",
            "llm_used_for_numbers": False,
            "assumptions_path": str(path),
            "assumptions_sha256": file_sha256(path),
            "current_price": str(current_price),
            "price_as_of": row.get("price_as_of"),
            "fair_value_per_share": str(fair_value),
            "price_to_dcf": str(current_price / fair_value) if fair_value > 0 else None,
            "screening_eligible": valuation.screening_eligible,
            "screening_exclusion_reasons": valuation.screening_exclusion_reasons,
            "assumption_confidence": str(valuation.assumption_confidence),
            "terminal_value_share": str(valuation.terminal_value_share),
            "provenance_warnings": valuation.provenance_warnings,
        }
    except Exception as exc:
        return {"status": "CALCULATION_FAILED_CLOSED", "error": f"{type(exc).__name__}: {exc}"}


def _markdown(report: dict[str, Any]) -> str:
    evidence = report["evidence_overlay"]
    valuation = report["valuation"]
    lines = [
        f"# {report['name']} ({report['ticker']})",
        "",
        f"- 기준 시각: {report['cutoff']}",
        f"- 시장: {report['market']}",
        f"- 상태: {report['status']}",
        f"- 검증 등급: {evidence.get('validation_grade', 'DATA_ONLY')}",
        "",
        "## 데이터·모델 역할",
        "",
        "DART 원문이 1차 회사 증거다. IR은 경영진 주장, 기업 애널리스트 보고서는 외부 해석, 산업 보고서는 reference-class 맥락으로만 사용한다. LLM은 숫자·확률·적정가를 만들지 않는다.",
        "",
        "## 결정론적 가치평가",
        "",
        f"- 상태: {valuation.get('status')}",
    ]
    if valuation.get("fair_value_per_share") is not None:
        lines.extend(
            [
                f"- 기준 가격: {valuation.get('current_price')} ({valuation.get('price_as_of')})",
                f"- FCFF 참고가치/주: {valuation.get('fair_value_per_share')}",
                f"- Price/DCF: {valuation.get('price_to_dcf')}",
                f"- 스크리닝 적격: {valuation.get('screening_eligible')}",
            ]
        )
    lines.extend(
        [
            "",
            "## LLM 증거 오버레이",
            "",
            f"- 위험 조치: {evidence.get('action', 'FAIL_CLOSED')}",
            f"- 지지 증거: {evidence.get('supportive_count', 0)}",
            f"- 훼손 증거: {evidence.get('erosive_count', 0)}",
            f"- 검증 통과 claim: {len(evidence.get('validated_claims', []))}",
            "",
        ]
    )
    for claim in evidence.get("validated_claims", []):
        lines.extend(
            [
                f"### {claim['axis']} · {claim['direction']}",
                "",
                claim["claim"],
                "",
                f"> {claim['exact_quote']}",
                "",
                f"출처: {claim['source_role']} / {claim['source_id']} / available_at={claim['available_at']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 해석 제한",
            "",
            "이 보고서는 2026-08-18 PIT historical/current validation 산출물이다. 현대 LLM의 사전학습 지식 오염 가능성 때문에 model-time true OOS가 아니며, 모든 LLM claim은 cutoff 이전 exact quote와 독립 entailment 검증을 통과해야 한다.",
            "",
        ]
    )
    return "\n".join(lines)


def seal(
    *,
    output: Path,
    universe_path: Path,
    dart_manifest_path: Path,
    dcf_audit_path: Path,
    exclusions_path: Path,
    cutoff: datetime,
) -> None:
    universe = {str(row["stock_code"]).zfill(6): row for row in read_csv(universe_path)}
    security_map = {row["ticker"]: row for row in read_json(output / "security-map.json")}
    manifest_rows = read_csv(dart_manifest_path)
    first_manifest = {str(row["ticker"]).zfill(6): row for row in manifest_rows}
    dcf_audit = {str(row["stock_code"]).zfill(6): row for row in read_csv(dcf_audit_path)} if dcf_audit_path.is_file() else {}
    exclusion_rows = read_csv(exclusions_path) if exclusions_path.is_file() else []
    exclusions: dict[str, list[str]] = defaultdict(list)
    for row in exclusion_rows:
        exclusions[str(row["stock_code"]).zfill(6)].append(row["reason"])

    validated_by_pack: dict[str, dict[str, Any]] = {}
    validation_failures: list[dict[str, str]] = []
    for pack_id in read_json(output / "corpus-manifest.json")["pack_ids"]:
        required = [output / "llm" / lane / f"{pack_id}.json" for lane in ("original", "anonymized", "entailment")]
        if not all(path.is_file() for path in required):
            validation_failures.append({"pack_id": pack_id, "error": "MISSING_LLM_STAGE"})
            continue
        pack = _load_pack(output / "packs" / "selected" / f"{pack_id}.json")
        original = _assessment(required[0])
        anonymous = _assessment(required[1])
        entailed = HistoricalEntailmentBatch.model_validate(
            {"decisions": read_json(required[2]).get("decisions", [])}
        )
        try:
            claims, anonymization_audit = validate_current_assessment(
                cutoff=cutoff,
                excerpts=pack["excerpts"],
                original=original,
                anonymized=anonymous,
                entailment=entailed,
                issuer_name=str(pack["issuer_name"]),
                ticker=str(pack["ticker"]),
            )
            decision = deterministic_overlay_decision(pack_id, claims)
            payload = decision.model_dump(mode="json")
            payload["anonymization_audit"] = anonymization_audit
            validated_by_pack[pack_id] = payload
            write_json(output / "validated" / f"{pack_id}.json", payload)
        except Exception as exc:
            validation_failures.append({"pack_id": pack_id, "error": f"{type(exc).__name__}: {exc}"})
    write_json(output / "validation-failures.json", validation_failures)

    catalog_rows: list[dict[str, Any]] = []
    source_role_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for ticker, source in sorted(universe.items()):
        mapping = security_map[ticker]
        pack_id = str(mapping.get("pack_id") or "")
        overlay = validated_by_pack.get(pack_id)
        if overlay is None:
            overlay = {
                "pack_id": pack_id or None,
                "action": "FAIL_CLOSED",
                "validated_claims": [],
                "supportive_count": 0,
                "erosive_count": 0,
                "source_role_count": 0,
                "validation_grade": "DATA_ONLY_NO_VALIDATED_LLM_OVERLAY",
                "llm_changed_cheap_rank": False,
            }
        for claim in overlay.get("validated_claims", []):
            source_role_counts.update([claim["source_role"]])
        row = first_manifest.get(ticker)
        if row is not None and ticker in dcf_audit:
            row = {**row, **dcf_audit[ticker]}
        valuation = _dcf_payload(row)
        status = (
            "COMPLETE"
            if mapping["status"] == "PIT_DOCUMENTS_AVAILABLE" and overlay["action"] != "FAIL_CLOSED"
            else "COMPLETE_DATA_ONLY"
            if mapping["status"] == "PIT_DOCUMENTS_AVAILABLE"
            else "NO_PERIODIC_PIT_FILING"
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "ticker": ticker,
            "name": source.get("name", ""),
            "market": source.get("market", ""),
            "security_type": source.get("security_type", ""),
            "cutoff": cutoff.isoformat(),
            "price_as_of": source.get("price_as_of"),
            "price_source": source.get("price_source"),
            "status": status,
            "filing_ticker": mapping.get("filing_ticker") or None,
            "source_status": mapping["status"],
            "source_exclusions": exclusions.get(ticker, []),
            "valuation": valuation,
            "evidence_overlay": overlay,
            "model_contract": {
                "preprocess_model": PREPROCESS_MODEL,
                "main_model": MAIN_MODEL,
                "data_pit": True,
                "model_time_true_oos": False,
                "future_knowledge_trap_required": True,
                "anonymized_stability_required": True,
                "independent_entailment_required": True,
                "llm_may_generate_valuation_numbers": False,
                "llm_may_change_rank": False,
            },
        }
        report_dir = output / "reports" / ticker
        write_json(report_dir / "report.json", report)
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.md").write_text(_markdown(report), encoding="utf-8")
        status_counts.update([status])
        catalog_rows.append(
            {
                "ticker": ticker,
                "name": source.get("name", ""),
                "market": source.get("market", ""),
                "security_type": source.get("security_type", ""),
                "status": status,
                "filing_ticker": mapping.get("filing_ticker") or "",
                "overlay_action": overlay["action"],
                "validated_claim_count": len(overlay.get("validated_claims", [])),
                "valuation_status": valuation["status"],
                "report_json": str((report_dir / "report.json").resolve()),
                "report_markdown": str((report_dir / "report.md").resolve()),
            }
        )
    write_csv(
        output / "report-catalog.csv",
        catalog_rows,
        (
            "ticker", "name", "market", "security_type", "status", "filing_ticker",
            "overlay_action", "validated_claim_count", "valuation_status",
            "report_json", "report_markdown",
        ),
    )
    coverage = {
        "schema_version": SCHEMA_VERSION,
        "cutoff": cutoff.isoformat(),
        "universe_count": len(universe),
        "report_count": len(catalog_rows),
        "unique_ticker_count": len({row["ticker"] for row in catalog_rows}),
        "status_counts": dict(sorted(status_counts.items())),
        "validated_source_role_claim_counts": dict(sorted(source_role_counts.items())),
        "llm_validation_failure_count": len(validation_failures),
        "all_universe_securities_have_report": len(catalog_rows) == len(universe),
        "models": {"preprocess": PREPROCESS_MODEL, "main": MAIN_MODEL},
    }
    write_json(output / "coverage.json", coverage)
    print(json.dumps(coverage, ensure_ascii=False, indent=2))


def parse_cutoff(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("--cutoff must include a timezone offset")
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build all-Korean-security PIT reports with DART-first LLM evidence overlays."
    )
    parser.add_argument("--stage", choices=("prepare", "llm", "seal", "all"), required=True)
    parser.add_argument("--universe", required=True, type=Path)
    parser.add_argument("--dart-manifest", required=True, type=Path)
    parser.add_argument("--ir-manifest", type=Path)
    parser.add_argument("--dcf-audit", type=Path)
    parser.add_argument("--exclusions", type=Path)
    parser.add_argument("--synalyst-hankyung-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cutoff", required=True, type=parse_cutoff)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 6:
        raise ValueError("--batch-size must be between 1 and 6")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    output = args.output.resolve()
    if args.stage in {"prepare", "all"}:
        prepare(
            universe_path=args.universe.resolve(),
            dart_manifest_path=args.dart_manifest.resolve(),
            ir_manifest_path=args.ir_manifest.resolve() if args.ir_manifest else None,
            synalyst_hankyung_root=args.synalyst_hankyung_root.resolve(),
            output=output,
            cutoff=args.cutoff,
        )
    if args.stage in {"llm", "all"}:
        run_llm(output=output, batch_size=args.batch_size, workers=args.workers)
    if args.stage in {"seal", "all"}:
        if args.dcf_audit is None or args.exclusions is None:
            raise ValueError("seal requires --dcf-audit and --exclusions")
        seal(
            output=output,
            universe_path=args.universe.resolve(),
            dart_manifest_path=args.dart_manifest.resolve(),
            dcf_audit_path=args.dcf_audit.resolve(),
            exclusions_path=args.exclusions.resolve(),
            cutoff=args.cutoff,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
