from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from moatrader.business.drivers import ValuationDriver, ValuationDriverMapper, ValuationEvidenceRole
from moatrader.canonical.models import SourceType
from moatrader.evidence.models import EvidenceCard
from moatrader.expectations import (
    ConfirmationStatus,
    HoldoutResearchInput,
    HoldoutSignal,
    HoldoutSourceReference,
    RiskProfile,
    ThesisConfirmation,
    ThreePValidity,
    ValuationFragilityDiagnostics,
)
from moatrader.valuation import CheckStatus, PlausibilityStatus, ProbabilitySupport


SEOUL = ZoneInfo("Asia/Seoul")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def ticker(value: Any) -> str:
    return str(value or "").strip().zfill(6)


def number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def percentiles(values: list[float]) -> list[float]:
    if len(values) == 1:
        return [50.0]
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + end - 1) / 2.0
        for offset in range(position, end):
            result[order[offset]] = 100.0 * rank / (len(values) - 1)
        position = end
    return result


def probability_support(cards: list[EvidenceCard], *, issuer_id: str, cutoff: datetime) -> tuple[ProbabilitySupport, list[Any]]:
    bundle = ValuationDriverMapper().map_cards(
        issuer_id=issuer_id,
        as_of=cutoff,
        cards=cards,
    )
    statuses: list[ProbabilitySupport] = []
    grouped = bundle.by_driver()
    for driver in ValuationDriver:
        items = grouped[driver]
        support = any(item.role == ValuationEvidenceRole.SUPPORT for item in items)
        counter = any(item.role == ValuationEvidenceRole.COUNTER for item in items)
        if counter and not support:
            statuses.append(ProbabilitySupport.CONTRADICTED)
        elif counter and support:
            statuses.append(ProbabilitySupport.MIXED)
        elif support:
            statuses.append(ProbabilitySupport.SUPPORTED)
        else:
            statuses.append(ProbabilitySupport.WEAK)
    if ProbabilitySupport.CONTRADICTED in statuses:
        aggregate = ProbabilitySupport.CONTRADICTED
    elif ProbabilitySupport.MIXED in statuses or (
        ProbabilitySupport.SUPPORTED in statuses and ProbabilitySupport.WEAK in statuses
    ):
        aggregate = ProbabilitySupport.MIXED
    elif statuses and all(item == ProbabilitySupport.SUPPORTED for item in statuses):
        aggregate = ProbabilitySupport.SUPPORTED
    else:
        aggregate = ProbabilitySupport.WEAK
    return aggregate, bundle.evidence


def source_references(
    *,
    chunks: list[dict[str, Any]],
    evidence_chunk_ids: set[str],
    cutoff: datetime,
) -> list[HoldoutSourceReference]:
    by_key: dict[tuple[str, str], HoldoutSourceReference] = {}
    for chunk in chunks:
        if str(chunk.get("chunk_id")) not in evidence_chunk_ids:
            continue
        available_at = (chunk.get("metadata") or {}).get("available_at")
        if not available_at:
            raise ValueError(f"chunk {chunk.get('chunk_id')} lacks available_at")
        available = datetime.fromisoformat(str(available_at))
        if available > cutoff:
            raise ValueError(f"chunk {chunk.get('chunk_id')} is after the PIT cutoff")
        for source in chunk.get("source_refs") or []:
            source_type = SourceType(str(source["source_type"]))
            if source_type == SourceType.GENERATED_SUMMARY:
                continue
            document_id = str(source["document_id"])
            key = (source_type.value, document_id)
            by_key.setdefault(
                key,
                HoldoutSourceReference(
                    document_id=document_id,
                    source_type=source_type,
                    available_at=available,
                ),
            )
    if not by_key:
        raise ValueError("no source-grounded evidence references were found")
    return [by_key[key] for key in sorted(by_key)]


def improving_percentiles(
    rows: dict[str, dict[str, str]],
    prior_path: Path | None,
) -> dict[str, float]:
    if prior_path is None:
        return {}
    prior = {
        item.ticker: item
        for item in (
            HoldoutSignal.model_validate(raw)
            for raw in json.loads(prior_path.read_text(encoding="utf-8-sig"))
        )
    }
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for code, row in rows.items():
        current = number(row.get("base_value_per_share"))
        previous = prior.get(code)
        previous_value = (
            number(previous.alpha.cheap.primary_fair_value_per_share)
            if previous is not None
            else None
        )
        if current is None or previous_value is None or previous_value <= 0:
            continue
        grouped[(str(row["method"]), str(row["economic_archetype"]))].append(
            (code, current / previous_value - 1.0)
        )
    result: dict[str, float] = {}
    for items in grouped.values():
        ranks = percentiles([item[1] for item in items])
        for (code, _), rank in zip(items, ranks, strict=True):
            result[code] = rank
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build 150 return-blind risk/confirmation inputs from company evidence outputs."
    )
    parser.add_argument("--as-of", type=date.fromisoformat, required=True)
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--company-root", type=Path, required=True)
    parser.add_argument("--valuation-signals", type=Path, required=True)
    parser.add_argument("--prior-built-signals", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"research input output already exists: {output}")
    cutoff = datetime.combine(args.as_of, time.max, tzinfo=SEOUL)
    universe_rows = read_csv(args.universe.resolve())
    if len(universe_rows) != 150:
        raise ValueError(f"expected 150 universe rows, got {len(universe_rows)}")
    universe = {ticker(row.get("stock_code") or row.get("ticker")): row for row in universe_rows}
    valuation_rows = {
        ticker(row.get("ticker")): row
        for row in read_csv(args.valuation_signals.resolve())
        if str(row.get("date")) == args.as_of.isoformat()
    }
    if set(valuation_rows) != set(universe):
        raise ValueError("valuation signal ticker set differs from the fixed universe")
    improving = improving_percentiles(
        valuation_rows,
        args.prior_built_signals.resolve() if args.prior_built_signals else None,
    )

    result: list[HoldoutResearchInput] = []
    for code in sorted(universe):
        company = args.company_root.resolve() / code
        dossier_path = company / "dossier.json"
        evidence_path = company / "evidence.jsonl"
        chunks_path = company / "chunks.jsonl"
        if not dossier_path.is_file() or not evidence_path.is_file() or not chunks_path.is_file():
            universe_available = datetime.combine(
                date.fromisoformat(str(universe[code]["as_of"])),
                time.max,
                tzinfo=SEOUL,
            )
            confirmation = (
                ThesisConfirmation(
                    improving=improving[code],
                    status=ConfirmationStatus.AVAILABLE,
                )
                if code in improving
                else ThesisConfirmation(
                    improving=None,
                    status=ConfirmationStatus.INSUFFICIENT_EVIDENCE,
                )
            )
            result.append(
                HoldoutResearchInput(
                    ticker=code,
                    risk=RiskProfile(
                        fragility_score=None,
                        three_p=ThreePValidity(
                            possible=CheckStatus.FAIL,
                            plausible=PlausibilityStatus.UNKNOWN,
                            probable=ProbabilitySupport.WEAK,
                            hard_gate_pass=False,
                            review_required=True,
                        ),
                        industry_counterevidence_count=None,
                        industry_range_widener_count=None,
                        industry_evidence_available=False,
                        reason_codes=[
                            "COMPANY_EVIDENCE_OUTPUT_MISSING",
                            "VALUATION_POSSIBLE_FAIL",
                            "PLAUSIBILITY_REFERENCE_CLASS_MISSING",
                            "INDUSTRY_CONTEXT_MISSING",
                        ],
                    ),
                    confirmation=confirmation,
                    source_references=[
                        HoldoutSourceReference(
                            document_id=f"UNIVERSE_SNAPSHOT:{code}",
                            source_type=SourceType.OTHER,
                            available_at=universe_available,
                        )
                    ],
                )
            )
            continue
        dossier = json.loads(dossier_path.read_text(encoding="utf-8-sig"))
        if datetime.fromisoformat(str(dossier["as_of"])) != cutoff:
            raise ValueError(f"dossier cutoff differs for {code}")
        cards = [EvidenceCard.model_validate(item) for item in read_jsonl(evidence_path)]
        probable, mapped = probability_support(
            cards,
            issuer_id=str(dossier["issuer_id"]),
            cutoff=cutoff,
        )
        row = valuation_rows[code]
        possible = CheckStatus.PASS if str(row.get("possible_pass")) == "1" else CheckStatus.FAIL
        three_p = ThreePValidity(
            possible=possible,
            plausible=PlausibilityStatus.UNKNOWN,
            probable=probable,
            hard_gate_pass=possible != CheckStatus.FAIL,
            review_required=True,
        )
        downside = number(row.get("downside_value_per_share"))
        base = number(row.get("base_value_per_share"))
        upside = number(row.get("upside_value_per_share"))
        fragility = None
        if downside is not None and base is not None and upside is not None:
            fragility = ValuationFragilityDiagnostics(
                downside_value_per_share=downside,
                base_value_per_share=base,
                upside_value_per_share=upside,
                assumption_confidence=number(row.get("assumption_confidence")),
                warning_count=int(number(row.get("valuation_warning_count")) or 0),
            ).score()
        industry = [item for item in mapped if item.source_type == SourceType.INDUSTRY]
        industry_counter = sum(item.role == ValuationEvidenceRole.COUNTER for item in industry)
        industry_wideners = sum(
            item.role == ValuationEvidenceRole.RANGE_WIDENER or item.range_widening_required
            for item in industry
        )
        reason_codes = ["PLAUSIBILITY_REFERENCE_CLASS_MISSING"]
        if possible == CheckStatus.FAIL:
            reason_codes.append("VALUATION_POSSIBLE_FAIL")
        if not industry:
            reason_codes.append("INDUSTRY_CONTEXT_MISSING")
        references = source_references(
            chunks=read_jsonl(chunks_path),
            evidence_chunk_ids={item.source_chunk_id for item in cards},
            cutoff=cutoff,
        )
        confirmation = (
            ThesisConfirmation(
                improving=improving[code],
                status=ConfirmationStatus.AVAILABLE,
            )
            if code in improving
            else ThesisConfirmation(
                improving=None,
                status=ConfirmationStatus.INSUFFICIENT_EVIDENCE,
            )
        )
        result.append(
            HoldoutResearchInput(
                ticker=code,
                risk=RiskProfile(
                    fragility_score=fragility,
                    three_p=three_p,
                    industry_counterevidence_count=(industry_counter if industry else None),
                    industry_range_widener_count=(industry_wideners if industry else None),
                    industry_evidence_available=bool(industry),
                    reason_codes=reason_codes,
                ),
                confirmation=confirmation,
                source_references=references,
            )
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in result],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "expectation-gap-holdout-research-inputs/1",
        "as_of": args.as_of.isoformat(),
        "row_count": len(result),
        "unique_ticker_count": len({item.ticker for item in result}),
        "possible_counts": dict(
            sorted(Counter(item.risk.three_p.possible.value for item in result).items())
        ),
        "probable_counts": dict(
            sorted(Counter(item.risk.three_p.probable.value for item in result).items())
        ),
        "fragility_present_count": sum(
            item.risk.fragility_score is not None for item in result
        ),
        "industry_evidence_present_count": sum(
            item.risk.industry_evidence_available for item in result
        ),
        "source_type_counts": dict(
            sorted(
                Counter(
                    reference.source_type.value
                    for item in result
                    for reference in item.source_references
                ).items()
            )
        ),
        "return_data_accessed": False,
        "research_inputs_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
