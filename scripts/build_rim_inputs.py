from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Any

from moatrader.valuation import (
    CommonRimEngine,
    RIM_POLICY_VERSION,
    RimBuildInput,
    RimBuilder,
    RoutedValuationInput,
    ValuationMethod,
)

try:
    from scripts.build_normalized_fcff_inputs import file_sha256, read_json
except ModuleNotFoundError:  # direct ``python scripts/<file>.py`` execution
    from build_normalized_fcff_inputs import file_sha256, read_json  # type: ignore[no-redef]


BUILD_REPORT_VERSION = "rim-input-build/1"
REFERENCE_CLASS_MINIMUM = 20
_TE_RE = re.compile(r"<TE\b([^>]*)>(.*?)</TE>", re.IGNORECASE | re.DOTALL)
_ATTR_RE = re.compile(r'(\w+)="([^"]*)"')
_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class DartDocument:
    input_path: Path
    metadata_path: Path
    source_id: str
    issuer_id: str
    report_name: str
    period_end: str
    available_at: date
    raw_sha256: str


@dataclass(frozen=True)
class RimAccountingValues:
    book_equity: Decimal
    prior_fy_net_income: Decimal
    current_ytd_net_income: Decimal
    prior_ytd_net_income: Decimal
    accounting_scope: str


def _decimal_text(text: str, *, negated: bool, scale: int) -> Decimal:
    cleaned = html.unescape(_TAG_RE.sub("", text)).strip().replace(",", "")
    if not cleaned or cleaned in {"-", "—"}:
        raise ValueError("empty DART numeric cell")
    parenthesized = cleaned.startswith("(") and cleaned.endswith(")")
    if parenthesized:
        cleaned = cleaned[1:-1]
    value = Decimal(cleaned)
    if parenthesized:
        value = -abs(value)
    elif negated:
        value = -abs(value)
    return value * (Decimal(10) ** scale)


def _cells(path: Path) -> list[tuple[str, str, Decimal]]:
    text = path.read_text(encoding="utf-8-sig", errors="strict")
    cells: list[tuple[str, str, Decimal]] = []
    for match in _TE_RE.finditer(text):
        attrs = {key.upper(): value for key, value in _ATTR_RE.findall(match.group(1))}
        account = attrs.get("ACODE", "")
        context = attrs.get("ACONTEXT", "")
        decimal_attr = attrs.get("ADECIMAL", "")
        if not account or not context or not re.fullmatch(r"-?\d+", decimal_attr):
            continue
        scale = max(0, -int(decimal_attr))
        try:
            value = _decimal_text(
                match.group(2),
                negated=attrs.get("ANEGATED") == "Y",
                scale=scale,
            )
        except Exception:
            continue
        cells.append((account, context, value))
    return cells


def _context_value(
    cells: list[tuple[str, str, Decimal]],
    *,
    account: str,
    period_pattern: str,
) -> Decimal:
    candidates: list[tuple[tuple[int, int], Decimal]] = []
    context_re = re.compile(period_pattern)
    for actual_account, context, value in cells:
        if actual_account != account or not context_re.match(context):
            continue
        if "_ifrs-full_ConsolidatedMember_" in context:
            dimension_penalty = 2
        elif context.endswith("_ifrs-full_ConsolidatedMember"):
            dimension_penalty = 0
        elif "_ifrs-full_SeparateMember" in context:
            dimension_penalty = 3
        else:
            dimension_penalty = 1
        candidates.append(((dimension_penalty, len(context)), value))
    if not candidates:
        raise ValueError(f"DART value missing: {account}:{period_pattern}")
    candidates.sort(key=lambda item: item[0])
    best_score = candidates[0][0]
    best_values = {value for score, value in candidates if score == best_score}
    if len(best_values) != 1:
        raise ValueError(f"ambiguous DART value: {account}:{period_pattern}")
    return next(iter(best_values))


def extract_rim_accounting_values(
    *, annual_path: Path, interim_path: Path
) -> RimAccountingValues:
    annual_cells = _cells(annual_path)
    interim_cells = _cells(interim_path)
    bases = (
        (
            "OWNERS_OF_PARENT",
            "ifrs-full_EquityAttributableToOwnersOfParent",
            "ifrs-full_ProfitLossAttributableToOwnersOfParent",
        ),
        ("TOTAL", "ifrs-full_Equity", "ifrs-full_ProfitLoss"),
    )
    errors: list[str] = []
    for scope, equity_account, income_account in bases:
        try:
            return RimAccountingValues(
                book_equity=_context_value(
                    interim_cells,
                    account=equity_account,
                    period_pattern=r"^CFY\d+eHYA(?:_|$)",
                ),
                prior_fy_net_income=_context_value(
                    annual_cells,
                    account=income_account,
                    period_pattern=r"^CFY\d+dFY(?:_|$)",
                ),
                current_ytd_net_income=_context_value(
                    interim_cells,
                    account=income_account,
                    period_pattern=r"^CFY\d+dHYA(?:_|$)",
                ),
                prior_ytd_net_income=_context_value(
                    interim_cells,
                    account=income_account,
                    period_pattern=r"^PFY\d+dHYA(?:_|$)",
                ),
                accounting_scope=scope,
            )
        except ValueError as exc:
            errors.append(f"{scope}:{exc}")
    raise ValueError(";".join(errors))


def _load_documents(assignment: dict[str, Any], *, cutoff: date) -> list[DartDocument]:
    documents: list[DartDocument] = []
    for item in assignment.get("dart_documents") or []:
        metadata_path = Path(str(item["metadata"]))
        input_path = Path(str(item["input"]))
        metadata = read_json(metadata_path)
        available_at = date.fromisoformat(str(metadata["available_at"])[:10])
        if available_at > cutoff:
            continue
        actual_hash = file_sha256(input_path)
        declared_hash = str(item.get("raw_sha256") or metadata.get("raw_sha256") or "")
        if declared_hash and actual_hash != declared_hash:
            raise ValueError(f"DART hash mismatch: {item['source_id']}")
        documents.append(
            DartDocument(
                input_path=input_path,
                metadata_path=metadata_path,
                source_id=str(item["source_id"]),
                issuer_id=str(metadata["issuer_id"]),
                report_name=str(metadata.get("report_name") or metadata.get("title") or ""),
                period_end=str(metadata["period_end"]),
                available_at=available_at,
                raw_sha256=actual_hash,
            )
        )
    return documents


def _latest_document(documents: list[DartDocument], *, period_month: int) -> DartDocument:
    matches = [
        item
        for item in documents
        if date.fromisoformat(item.period_end).month == period_month
    ]
    if not matches:
        raise ValueError(f"missing DART report for period month {period_month}")
    return max(matches, key=lambda item: (item.period_end, item.available_at, item.source_id))


def build_rim_input(
    *,
    universe_row: dict[str, str],
    assignment: dict[str, Any],
    as_of: str,
) -> RoutedValuationInput:
    cutoff = date.fromisoformat(as_of)
    documents = _load_documents(assignment, cutoff=cutoff)
    annual = _latest_document(documents, period_month=12)
    interim = _latest_document(documents, period_month=6)
    values = extract_rim_accounting_values(
        annual_path=annual.input_path,
        interim_path=interim.input_path,
    )
    if annual.issuer_id != interim.issuer_id:
        raise ValueError("annual/interim issuer mismatch")
    evidence_refs = [
        f"DART:{annual.source_id}:SHA256:{annual.raw_sha256}",
        f"DART:{interim.source_id}:SHA256:{interim.raw_sha256}",
    ]
    source = RimBuildInput(
        issuer_id=annual.issuer_id,
        as_of=as_of,
        book_equity=values.book_equity,
        prior_fy_net_income=values.prior_fy_net_income,
        current_ytd_net_income=values.current_ytd_net_income,
        prior_ytd_net_income=values.prior_ytd_net_income,
        diluted_shares=Decimal(universe_row["listed_shares"]),
        size_bucket=str(universe_row.get("size_bucket") or "SMALL").upper(),
        evidence_available_at={
            evidence_refs[0]: annual.available_at,
            evidence_refs[1]: interim.available_at,
        },
        provenance=evidence_refs
        + [
            f"ACCOUNTING_SCOPE:{values.accounting_scope}",
            "TTM:PRIOR_FY_PLUS_CURRENT_YTD_MINUS_PRIOR_YTD",
            f"POLICY:{RIM_POLICY_VERSION}",
        ],
    )
    assumptions = RimBuilder().build(source)
    CommonRimEngine().value(assumptions)
    return RoutedValuationInput(
        issuer_id=source.issuer_id,
        as_of=as_of,
        method=ValuationMethod.RIM,
        assumptions=assumptions.model_dump(mode="json"),
        source_refs=source.provenance,
    )


def _read_universe(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output must be new and empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in _read_universe(args.universe)
        if str(row.get("finance_hint", "")).lower() == "true"
        and row.get("security_type") == "COMMON"
    ]
    assignments = {
        str(item["ticker"]).zfill(6): item for item in read_json(args.assignments)
    }
    generated: list[dict[str, str]] = []
    skipped: list[dict[str, str]] = []
    issuer_ids: set[str] = set()
    for row in rows:
        ticker = str(row["stock_code"]).zfill(6)
        try:
            assignment = assignments[ticker]
            envelope = build_rim_input(
                universe_row=row,
                assignment=assignment,
                as_of=args.as_of,
            )
            if envelope.issuer_id in issuer_ids:
                raise ValueError("duplicate issuer share class")
            issuer_ids.add(envelope.issuer_id)
            output_path = args.output / args.as_of / f"{ticker}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(
                    envelope.model_dump(mode="json"),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            generated.append({"date": args.as_of, "ticker": ticker, "issuer_id": envelope.issuer_id})
        except Exception as exc:
            skipped.append(
                {
                    "date": args.as_of,
                    "ticker": ticker,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    report: dict[str, Any] = {
        "schema_version": BUILD_REPORT_VERSION,
        "as_of": args.as_of,
        "universe_sha256": file_sha256(args.universe),
        "assignments_sha256": file_sha256(args.assignments),
        "llm_call_count": 0,
        "finance_common_candidate_count": len(rows),
        "generated_count": len(generated),
        "unique_issuer_count": len(issuer_ids),
        "skipped_count": len(skipped),
        "reference_class_minimum": REFERENCE_CLASS_MINIMUM,
        "score_reference_class_eligible": len(issuer_ids) >= REFERENCE_CLASS_MINIMUM,
        "generated": generated,
        "skipped": skipped,
    }
    (args.output / "_build-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build deterministic PIT RIM inputs from cached DART originals."
    )
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("universe", "assignments", "output"):
        setattr(args, name, getattr(args, name).resolve())
    report = run(args)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["score_reference_class_eligible"] else 2


if __name__ == "__main__":
    sys.exit(main())
