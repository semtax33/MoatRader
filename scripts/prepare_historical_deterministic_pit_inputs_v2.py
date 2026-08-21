from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from lxml import etree, html

from moatrader.adapters.html import decode_html_document
from moatrader.expectations.historical_evidence import sha256_file
from moatrader.expectations.historical_evidence_v2 import (
    PITApplicabilityRulesV2,
    PITOperatingSnapshotV2,
)
from scripts.build_historical_deterministic_pit_evidence_v2 import PITOperatingPairInputV2


D = Decimal
_XML_DECLARATION_RE = re.compile(r"^\s*<\?xml[^>]*\?>", re.I)
_SPACE_RE = re.compile(r"\s+")
_NON_LABEL_RE = re.compile(r"[\s\u3000ㆍ·:：.,_\-/]")
_NUMBER_RE = re.compile(r"^\(?-?\d[\d,]*(?:\.\d+)?\)?$")
_UNIT_RE = re.compile(r"단위\s*[:：]?\s*(백만원|천원|억원|원)", re.I)
_CAPACITY_RE = re.compile(r"생산능력|생산설비|설비능력|증설|공장|CAPA|capacity", re.I)
_BACKLOG_RE = re.compile(r"수주잔고|계약잔액|기말수주잔고|잔여계약", re.I)
_TOTAL_RE = re.compile(r"합계|총계")
_REVENUE_RE = re.compile(r"^(?:수익\(?매출액\)?|매출액|매출|영업수익|수익)$")
_OPERATING_PROFIT_RE = re.compile(r"^(?:영업이익(?:\(?손실\)?)?|영업손실)$")
_INVENTORY_RE = re.compile(r"^(?:재고자산|유동재고자산)$")
_ASSETS_RE = re.compile(r"^자산총계$")
_PPE_RE = re.compile(r"^유형자산$")
_CAPEX_PPE_RE = re.compile(r"^(?:유형자산의취득|유형자산취득)$")
_CAPEX_INTANGIBLE_RE = re.compile(r"^(?:무형자산의취득|무형자산취득)$")
_CAPEX_CIP_RE = re.compile(r"^(?:건설중인자산의취득|건설중인자산취득)$")

_UNIT_MULTIPLIER = {
    "원": D(1),
    "천원": D(1_000),
    "백만원": D(1_000_000),
    "억원": D(100_000_000),
}


@dataclass(frozen=True, slots=True)
class FilingSource:
    source_id: str
    origin: str
    path: str
    raw_sha256: str


@dataclass(frozen=True, slots=True)
class FilingTask:
    ticker: str
    rcept_no: str
    fiscal_period_end: str
    available_at: str
    finance_statement: FilingSource | None
    finance_comment: FilingSource | None
    business_info: FilingSource | None
    moatrader_original: FilingSource | None


@dataclass(frozen=True, slots=True)
class PairRef:
    pair_id: str
    ticker: str
    previous_key: tuple[str, str, str]
    current_key: tuple[str, str, str]


def _tag(element: etree._Element) -> str:
    if not isinstance(element.tag, str):
        return ""
    return element.tag.rsplit("}", 1)[-1].split(":")[-1].lower()


def _text(element: etree._Element) -> str:
    return _SPACE_RE.sub(" ", " ".join(element.itertext()).replace("\xa0", " ")).strip()


def _label(value: str) -> str:
    return _NON_LABEL_RE.sub("", value)


def _number(value: str) -> Decimal | None:
    normalized = value.strip().replace("\xa0", "").replace(" ", "")
    if not _NUMBER_RE.fullmatch(normalized):
        return None
    negative = normalized.startswith("(") and normalized.endswith(")")
    normalized = normalized.strip("()").replace(",", "")
    try:
        result = D(normalized)
    except InvalidOperation:
        return None
    return -result if negative else result


def _row_values(table: etree._Element) -> list[tuple[str, list[Decimal]]]:
    result: list[tuple[str, list[Decimal]]] = []
    for row in table.xpath(".//*[local-name()='tr' or local-name()='TR']"):
        cells = row.xpath("./*[local-name()='td' or local-name()='TD' or local-name()='th' or local-name()='TH']")
        if not cells:
            continue
        values = [_text(cell) for cell in cells]
        numbers = [item for item in (_number(value) for value in values[1:]) if item is not None]
        result.append((values[0], numbers))
    return result


def _current_flow_value(values: list[Decimal], *, fiscal_period_end: date) -> Decimal | None:
    if not values:
        return None
    if fiscal_period_end.month in {3, 6, 9} and len(values) >= 4:
        return values[1]
    return values[0]


def _parse_tree(document: str) -> etree._Element:
    return html.document_fromstring(_XML_DECLARATION_RE.sub("", document, count=1))


def extract_pit_metrics_from_html(
    document: str,
    *,
    fiscal_period_end: date,
) -> dict[str, Decimal | bool | None]:
    """Extract current-period accounting values without LLM arithmetic or outcome data."""

    root = _parse_tree(document)
    current_unit = D(1)
    income_candidates: list[tuple[int, Decimal, Decimal]] = []
    balance_candidates: list[tuple[int, Decimal | None, Decimal, Decimal | None]] = []
    capex_candidates: list[tuple[int, Decimal]] = []
    backlog_values: list[Decimal] = []
    backlog_disclosed = False
    capacity_disclosed = bool(_CAPACITY_RE.search(_text(root)))

    for order, element in enumerate(root.iter()):
        tag = _tag(element)
        if tag in {"p", "span", "td", "th"}:
            match = _UNIT_RE.search(_text(element))
            if match:
                current_unit = _UNIT_MULTIPLIER[match.group(1)]
        if tag != "table":
            continue
        table_unit = current_unit
        table_unit_match = _UNIT_RE.search(_text(element))
        if table_unit_match:
            table_unit = _UNIT_MULTIPLIER[table_unit_match.group(1)]
        rows = _row_values(element)
        if not rows:
            continue
        row_index = {_label(name): values for name, values in rows}

        revenue: Decimal | None = None
        operating_profit: Decimal | None = None
        inventory: Decimal | None = None
        assets: Decimal | None = None
        ppe: Decimal | None = None
        for name, values in row_index.items():
            if revenue is None and _REVENUE_RE.fullmatch(name):
                revenue = _current_flow_value(values, fiscal_period_end=fiscal_period_end)
            if operating_profit is None and _OPERATING_PROFIT_RE.fullmatch(name):
                operating_profit = _current_flow_value(values, fiscal_period_end=fiscal_period_end)
            if inventory is None and _INVENTORY_RE.fullmatch(name) and values:
                inventory = values[0]
            if assets is None and _ASSETS_RE.fullmatch(name) and values:
                assets = values[0]
            if ppe is None and _PPE_RE.fullmatch(name) and values:
                ppe = values[0]
        if revenue is not None and operating_profit is not None:
            income_candidates.append(
                (order, revenue * table_unit, operating_profit * table_unit)
            )
        if assets is not None:
            balance_candidates.append(
                (
                    order,
                    inventory * table_unit if inventory is not None else None,
                    assets * table_unit,
                    ppe * table_unit if ppe is not None else None,
                )
            )

        ppe_capex: Decimal | None = None
        intangible_capex: Decimal | None = None
        cip_capex: Decimal | None = None
        for name, values in row_index.items():
            value = _current_flow_value(values, fiscal_period_end=fiscal_period_end)
            if value is None:
                continue
            if _CAPEX_PPE_RE.fullmatch(name):
                ppe_capex = abs(value)
            elif _CAPEX_INTANGIBLE_RE.fullmatch(name):
                intangible_capex = abs(value)
            elif _CAPEX_CIP_RE.fullmatch(name):
                cip_capex = abs(value)
        if ppe_capex is not None or intangible_capex is not None or cip_capex is not None:
            capex = (ppe_capex if ppe_capex is not None else (cip_capex or D(0))) + (
                intangible_capex or D(0)
            )
            capex_candidates.append((order, capex * table_unit))
            capacity_disclosed = True

        header_text = " ".join(
            _text(cell)
            for cell in element.xpath(".//*[local-name()='th' or local-name()='TH']")
        )
        if _BACKLOG_RE.search(header_text):
            backlog_disclosed = True
            total_rows = [
                values[-1]
                for name, values in rows
                if values and _TOTAL_RE.search(name)
            ]
            if total_rows:
                backlog_values.append(abs(total_rows[-1]) * table_unit)
            else:
                detail_values = [
                    values[-1]
                    for name, values in rows
                    if values and not _BACKLOG_RE.search(name)
                ]
                if detail_values:
                    backlog_values.append(
                        sum((abs(value) for value in detail_values), D(0)) * table_unit
                    )

    income = min(income_candidates, default=None, key=lambda item: item[0])
    balance = min(balance_candidates, default=None, key=lambda item: item[0])
    capex = min(capex_candidates, default=None, key=lambda item: item[0])
    return {
        "revenue": income[1] if income else None,
        "operating_profit": income[2] if income else None,
        "inventory": balance[1] if balance else None,
        "assets": balance[2] if balance else None,
        "ppe": balance[3] if balance else None,
        "capex": capex[1] if capex else None,
        "backlog": sum(backlog_values, D(0)) if backlog_values else None,
        "backlog_disclosed": backlog_disclosed,
        "capacity_disclosed": capacity_disclosed,
    }


def _source_bytes(source: FilingSource) -> tuple[bytes, str]:
    raw = Path(source.path).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != source.raw_sha256:
        raise ValueError(f"source hash mismatch: {source.path}")
    if source.origin != "MOATRADER_OPENDART_ARCHIVE":
        return raw, actual
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        candidates = [
            item
            for item in archive.infolist()
            if Path(item.filename).suffix.lower() in {".xml", ".html", ".htm"}
        ]
        if not candidates:
            raise ValueError(f"OpenDART archive contains no HTML/XML member: {source.path}")
        member = max(candidates, key=lambda item: item.file_size)
        return archive.read(member), actual


def _decoded_source(source: FilingSource) -> tuple[str, str]:
    content, raw_hash = _source_bytes(source)
    decoded, _ = decode_html_document(content)
    return decoded, raw_hash


def _extract_task(task: FilingTask) -> dict[str, Any]:
    period_end = date.fromisoformat(task.fiscal_period_end)
    metrics: dict[str, Decimal | bool | None] = {
        "revenue": None,
        "operating_profit": None,
        "inventory": None,
        "assets": None,
        "ppe": None,
        "capex": None,
        "backlog": None,
        "backlog_disclosed": False,
        "capacity_disclosed": False,
    }
    source_ids: dict[str, list[str]] = {}
    origins: dict[str, str] = {}
    verified_hashes: dict[str, str] = {}
    errors: list[str] = []

    decoded_cache: dict[str, str] = {}
    parsed_by_path: dict[str, dict[str, Decimal | bool | None] | None] = {}

    def parsed(source: FilingSource) -> dict[str, Decimal | bool | None] | None:
        if source.path in parsed_by_path:
            return parsed_by_path[source.path]
        try:
            if source.path not in decoded_cache:
                decoded_cache[source.path], verified_hashes[source.path] = _decoded_source(source)
            result = extract_pit_metrics_from_html(
                decoded_cache[source.path], fiscal_period_end=period_end
            )
            parsed_by_path[source.path] = result
            return result
        except (OSError, ValueError, zipfile.BadZipFile, etree.ParserError) as exc:
            errors.append(f"{source.origin}:{type(exc).__name__}:{exc}")
            parsed_by_path[source.path] = None
            return None

    available_sources = tuple(
        source
        for source in (
            task.finance_statement,
            task.finance_comment,
            task.business_info,
            task.moatrader_original,
        )
        if source is not None
    )
    extracted_by_source = {
        source.path: parsed(source)
        for source in available_sources
    }

    finance_priority = tuple(
        source
        for source in (
            task.finance_statement,
            task.finance_comment,
            task.moatrader_original,
            task.business_info,
        )
        if source is not None
    )
    business_priority = tuple(
        source
        for source in (
            task.business_info,
            task.finance_comment,
            task.moatrader_original,
            task.finance_statement,
        )
        if source is not None
    )

    for source in available_sources:
        extracted = extracted_by_source[source.path]
        if extracted is None:
            continue
        metrics["backlog_disclosed"] = bool(metrics["backlog_disclosed"]) or bool(
            extracted["backlog_disclosed"]
        )
        metrics["capacity_disclosed"] = bool(metrics["capacity_disclosed"]) or bool(
            extracted["capacity_disclosed"]
        )

    for metric in ("revenue", "operating_profit", "inventory", "assets", "ppe", "capex"):
        for source in finance_priority:
            extracted = extracted_by_source[source.path]
            if extracted is None or extracted[metric] is None:
                continue
            value = extracted[metric]
            if (
                metric in {"revenue", "inventory", "assets", "ppe", "capex"}
                and isinstance(value, Decimal)
                and value < 0
            ):
                errors.append(
                    f"{source.origin}:NEGATIVE_ACCOUNT_VALUE_REJECTED:{metric}"
                )
                continue
            metrics[metric] = value
            source_ids[metric] = [source.source_id]
            origins[metric] = source.origin
            break

    for source in business_priority:
        extracted = extracted_by_source[source.path]
        if extracted is None or extracted["backlog"] is None:
            continue
        value = extracted["backlog"]
        if isinstance(value, Decimal) and value < 0:
            errors.append(f"{source.origin}:NEGATIVE_ACCOUNT_VALUE_REJECTED:backlog")
            continue
        metrics["backlog"] = value
        source_ids["backlog"] = [source.source_id]
        origins["backlog"] = source.origin
        break

    snapshot = PITOperatingSnapshotV2(
        issuer_id=task.ticker,
        fiscal_period_end=period_end,
        available_at=datetime.fromisoformat(task.available_at),
        source_ids=source_ids,
        **metrics,
    )
    return {
        "key": (task.ticker, task.rcept_no, task.fiscal_period_end),
        "snapshot": snapshot.model_dump(mode="json"),
        "origins": origins,
        "verified_hashes": verified_hashes,
        "errors": errors,
    }


def _filing_source(payload: dict[str, Any]) -> FilingSource:
    raw_sha256 = str(payload["raw_sha256"])
    return FilingSource(
        source_id=f"PIT_SRC_{raw_sha256[:20]}",
        origin=str(payload["origin"]),
        path=str(payload["path"]),
        raw_sha256=raw_sha256,
    )


def _task_for_side(pair: dict[str, Any], source_map: dict[str, Any], side: str) -> FilingTask:
    filing = dict(pair[side])
    by_origin: dict[str, FilingSource] = {}
    for raw in dict(source_map["sources"]).values():
        payload = dict(raw)
        if payload.get("side") == side:
            by_origin[str(payload["origin"])] = _filing_source(payload)
    return FilingTask(
        ticker=str(filing["ticker"]),
        rcept_no=str(filing["rcept_no"]),
        fiscal_period_end=str(filing["fiscal_period_end"]),
        available_at=str(filing["available_at"]),
        finance_statement=by_origin.get("ARCANA_FINANCE_STATEMENT_HTML"),
        finance_comment=by_origin.get("ARCANA_FINANCE_COMMENT_HTML"),
        business_info=by_origin.get("ARCANA_BUSINESS_HTML"),
        moatrader_original=by_origin.get("MOATRADER_OPENDART_ARCHIVE"),
    )


def _read_universe(
    filing_pair_input: Path,
    pair_source_map_input: Path,
) -> tuple[list[PairRef], list[FilingTask]]:
    pairs: list[PairRef] = []
    tasks: dict[tuple[str, str, str], FilingTask] = {}
    with filing_pair_input.open("r", encoding="utf-8") as pair_handle, pair_source_map_input.open(
        "r", encoding="utf-8"
    ) as source_handle:
        pair_lines = (line for line in pair_handle if line.strip())
        source_lines = (line for line in source_handle if line.strip())
        for pair_line, source_line in zip(pair_lines, source_lines, strict=True):
            pair = json.loads(pair_line)
            source_map = json.loads(source_line)
            if pair["pair_id"] != source_map["pair_id"]:
                raise ValueError("filing-pair and source-map order mismatch")
            previous = _task_for_side(pair, source_map, "previous")
            current = _task_for_side(pair, source_map, "current")
            previous_key = (previous.ticker, previous.rcept_no, previous.fiscal_period_end)
            current_key = (current.ticker, current.rcept_no, current.fiscal_period_end)
            for key, task in ((previous_key, previous), (current_key, current)):
                existing = tasks.setdefault(key, task)
                if existing != task:
                    raise ValueError(f"inconsistent source mapping for filing {key}")
            pairs.append(
                PairRef(
                    pair_id=str(pair["pair_id"]),
                    ticker=str(pair["ticker"]),
                    previous_key=previous_key,
                    current_key=current_key,
                )
            )
    return pairs, sorted(tasks.values(), key=lambda item: (item.ticker, item.fiscal_period_end))


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def prepare_pit_inputs(
    *,
    filing_pair_input: Path,
    pair_source_map_input: Path,
    output: Path,
    workers: int = 8,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    for path in (filing_pair_input, pair_source_map_input):
        if not path.is_file():
            raise FileNotFoundError(path)
    if workers < 1:
        raise ValueError("workers must be positive")

    pairs, tasks = _read_universe(filing_pair_input, pair_source_map_input)
    if workers == 1:
        extracted = [_extract_task(task) for task in tasks]
    else:
        extracted = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for completed, row in enumerate(
                executor.map(_extract_task, tasks, chunksize=8),
                start=1,
            ):
                extracted.append(row)
                if completed % 1_000 == 0 or completed == len(tasks):
                    print(f"extracted filings: {completed}/{len(tasks)}", flush=True)
    by_key = {tuple(row["key"]): PITOperatingSnapshotV2.model_validate(row["snapshot"]) for row in extracted}
    if len(by_key) != len(tasks):
        raise ValueError("filing extraction keys must be unique")

    output.mkdir(parents=True, exist_ok=True)
    pair_output = output / "pit-operating-pairs.jsonl"
    with pair_output.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            row = PITOperatingPairInputV2(
                pair_id=pair.pair_id,
                previous=by_key[pair.previous_key],
                current=by_key[pair.current_key],
            )
            handle.write(row.model_dump_json() + "\n")
    rules = PITApplicabilityRulesV2()
    rules_output = output / "pit-applicability-rules-v2.json"
    _write_json(rules_output, rules.model_dump(mode="json"))

    metric_coverage = Counter()
    origin_coverage = Counter()
    error_types = Counter()
    verified_paths: dict[str, str] = {}
    verified_origin_paths: dict[str, set[str]] = {}
    expected_source_paths: dict[str, set[str]] = {}
    task_by_key = {
        (task.ticker, task.rcept_no, task.fiscal_period_end): task for task in tasks
    }
    for task in tasks:
        for source in (
            task.finance_statement,
            task.finance_comment,
            task.business_info,
            task.moatrader_original,
        ):
            if source is not None:
                expected_source_paths.setdefault(source.origin, set()).add(source.path)
    audit_rows: list[dict[str, Any]] = []
    for row in extracted:
        snapshot = PITOperatingSnapshotV2.model_validate(row["snapshot"])
        for metric in ("revenue", "operating_profit", "inventory", "assets", "backlog", "capex", "ppe"):
            if getattr(snapshot, metric) is not None:
                metric_coverage[metric] += 1
        for metric, origin in dict(row["origins"]).items():
            origin_coverage[f"{metric}|{origin}"] += 1
        for error in row["errors"]:
            parts = str(error).split(":", 2)
            error_types[parts[1] if len(parts) > 1 else parts[0]] += 1
        for path, raw_hash in dict(row["verified_hashes"]).items():
            previous_hash = verified_paths.setdefault(path, raw_hash)
            if previous_hash != raw_hash:
                raise ValueError(f"source hash changed during extraction: {path}")
        task = task_by_key[tuple(row["key"])]
        for source in (
            task.finance_statement,
            task.finance_comment,
            task.business_info,
            task.moatrader_original,
        ):
            if source is not None and source.path in row["verified_hashes"]:
                verified_origin_paths.setdefault(source.origin, set()).add(source.path)
        audit_rows.append(
            {
                "ticker": snapshot.issuer_id,
                "fiscal_period_end": snapshot.fiscal_period_end.isoformat(),
                "available_at": snapshot.available_at.isoformat(),
                "metric_presence": {
                    metric: getattr(snapshot, metric) is not None
                    for metric in ("revenue", "operating_profit", "inventory", "assets", "backlog", "capex", "ppe")
                },
                "origins": row["origins"],
                "errors": row["errors"],
            }
        )
    audit_path = output / "filing-extraction-audit.jsonl"
    _write_jsonl(audit_path, audit_rows)
    report = {
        "schema_version": "moatrader-historical-pit-input-preparation-v2/1",
        "status": "PIT_INPUTS_PREPARED_OUTCOME_BLIND",
        "pair_count": len(pairs),
        "unique_filing_count": len(tasks),
        "metric_filing_coverage": dict(sorted(metric_coverage.items())),
        "metric_origin_coverage": dict(sorted(origin_coverage.items())),
        "extraction_error_distribution": dict(sorted(error_types.items())),
        "verified_source_path_count": len(verified_paths),
        "expected_source_path_count_by_origin": {
            origin: len(paths) for origin, paths in sorted(expected_source_paths.items())
        },
        "verified_source_path_count_by_origin": {
            origin: len(paths) for origin, paths in sorted(verified_origin_paths.items())
        },
        "all_available_source_variants_read": all(
            verified_origin_paths.get(origin, set()) == paths
            for origin, paths in expected_source_paths.items()
        ),
        "source_hash_mismatch_count": 0,
        "source_write_operations": 0,
        "source_files_modified": False,
        "arcana_finance_statement_used": any(task.finance_statement for task in tasks),
        "arcana_finance_comment_used": any(task.finance_comment for task in tasks),
        "arcana_business_info_used": any(task.business_info for task in tasks),
        "moatrader_original_used": any(task.moatrader_original for task in tasks),
        "capex_direction_contract": "RAW_INVESTMENT_DIRECTION_ONLY",
        "capex_in_primary_signed_score": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "per_pbr_role": "NOT_USED",
        "input_hashes": {
            "filing_pairs": sha256_file(filing_pair_input),
            "pair_source_map": sha256_file(pair_source_map_input),
        },
        "output_hashes": {
            "pit_operating_pairs": sha256_file(pair_output),
            "pit_applicability_rules": sha256_file(rules_output),
            "filing_extraction_audit": sha256_file(audit_path),
        },
    }
    _write_json(output / "stage-status.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare outcome-blind deterministic PIT inputs from Arcana DART HTML and "
            "MoatRader OpenDART originals without modifying either source lake."
        )
    )
    parser.add_argument("--filing-pair-input", type=Path, required=True)
    parser.add_argument("--pair-source-map-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = prepare_pit_inputs(
        filing_pair_input=args.filing_pair_input,
        pair_source_map_input=args.pair_source_map_input,
        output=args.output,
        workers=args.workers,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
