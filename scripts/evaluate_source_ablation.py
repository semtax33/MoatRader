from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from moatrader.runner.models import CompanyRunStatus, UniverseRunResult


SCHEMA_VERSION = "moatrader-source-ablation-evaluation/3"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_run(path: Path) -> UniverseRunResult:
    candidate = path / "run-result.json" if path.is_dir() else path
    return UniverseRunResult.model_validate_json(candidate.read_text(encoding="utf-8-sig"))


def _index(result: UniverseRunResult, label: str) -> dict[str, Any]:
    indexed = {company.ticker: company for company in result.companies}
    if len(indexed) != len(result.companies):
        raise ValueError(f"{label} contains duplicate tickers")
    incomplete = [
        company.ticker
        for company in result.companies
        if company.status != CompanyRunStatus.COMPLETE or company.moat_score is None
    ]
    if incomplete:
        raise ValueError(f"{label} has incomplete companies: {incomplete}")
    return indexed


def _rank(values: list[float]) -> list[float]:
    result = [0.0] * len(values)
    order = sorted(range(len(values)), key=lambda index: values[index])
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x, y = _rank(left), _rank(right)
    x_mean, y_mean = statistics.mean(x), statistics.mean(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y, strict=True))
    denominator = math.sqrt(
        sum((a - x_mean) ** 2 for a in x) * sum((b - y_mean) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _artifact(company: Any) -> Path:
    path = Path(company.artifact_directory)
    if not path.is_dir():
        raise FileNotFoundError(f"company artifact directory not found: {path}")
    return path


def _source_maps(directory: Path) -> tuple[dict[str, set[str]], dict[str, set[str]], dict[str, set[str]]]:
    context = _read_json(directory / "moat-strength-context.json")
    ref_sources = {
        reference["ref_id"]: set(reference.get("source_types") or [])
        for reference in context.get("references") or []
    }
    chunks = _read_jsonl(directory / "chunks.jsonl")
    chunk_sources = {
        chunk["chunk_id"]: {
            reference["source_type"] for reference in chunk.get("source_refs") or []
        }
        for chunk in chunks
    }
    evidence = _read_jsonl(directory / "evidence.jsonl")
    evidence_sources = {
        item["evidence_id"]: {str(item.get("source_type") or "OTHER")}
        for item in evidence
    }
    return ref_sources, chunk_sources, evidence_sources


def _item_sources(
    item: dict[str, Any],
    ref_sources: dict[str, set[str]],
    chunk_sources: dict[str, set[str]],
    evidence_sources: dict[str, set[str]],
) -> set[str]:
    sources: set[str] = set()
    for reference_id in item.get("reference_ids") or []:
        sources.update(ref_sources.get(reference_id, set()))
    for chunk_id in item.get("context_chunk_ids") or []:
        sources.update(chunk_sources.get(chunk_id, set()))
    for evidence_id in item.get("atomic_evidence_ids") or item.get("evidence_ids") or []:
        sources.update(evidence_sources.get(evidence_id, set()))
    return sources


def _company_metrics(company: Any) -> dict[str, Any]:
    directory = _artifact(company)
    assessment = _read_json(directory / "contextual-moat-assessment.json")
    reconciliation = _read_json(directory / "moat-reconciliation.json")
    score = _read_json(directory / "moat-score.json")
    quality = _read_json(directory / "quality-gate.json")
    evidence = _read_jsonl(directory / "evidence.jsonl")
    ref_sources, chunk_sources, evidence_sources = _source_maps(directory)

    contextual_items = [
        *(assessment.get("mechanisms") or []),
        *(assessment.get("outcome_confirmation") or []),
        *(assessment.get("counterevidence") or []),
    ]
    accepted_items = [
        *(reconciliation.get("mechanisms") or []),
        *(reconciliation.get("outcomes") or []),
        *(reconciliation.get("counterevidence") or []),
    ]
    decisions = [
        decision
        for decision in reconciliation.get("decisions") or []
        if decision.get("category") in {"MECHANISM", "OUTCOME"}
    ]
    rejected = [decision for decision in decisions if decision.get("accepted") is not True]
    contextual_ir = sum(
        "IR" in _item_sources(item, ref_sources, chunk_sources, evidence_sources)
        for item in contextual_items
    )
    accepted_ir = sum(
        "IR" in _item_sources(item, ref_sources, chunk_sources, evidence_sources)
        for item in accepted_items
    )
    ir_quality = [item for item in quality if str(item.get("source_document_id", "")).startswith("KINDIR_")]
    calls = _read_jsonl(directory / "llm-calls.jsonl")
    treatment = (
        _read_json(directory / "ir-treatment-audit.json")
        if (directory / "ir-treatment-audit.json").is_file()
        else {}
    )
    selection = _read_json(directory / "evidence-chunk-selection.json")
    return {
        "ticker": company.ticker,
        "issuer_name": company.issuer_name,
        "score": float(score["economic_moat_score"]),
        "score_eligible": bool(score["score_eligible"]),
        "evidence_sufficiency": int(assessment["evidence_sufficiency"]),
        "contextual_mechanism_count": len(assessment.get("mechanisms") or []),
        "contextual_outcome_count": len(assessment.get("outcome_confirmation") or []),
        "contextual_persistent_outcome_count": sum(
            int(item.get("persistence_bucket") or 0) >= 2
            for item in assessment.get("outcome_confirmation") or []
        ),
        "contextual_counter_count": len(assessment.get("counterevidence") or []),
        "accepted_mechanism_count": len(reconciliation.get("mechanisms") or []),
        "accepted_outcome_count": len(reconciliation.get("outcomes") or []),
        "accepted_counter_count": len(reconciliation.get("counterevidence") or []),
        "bridge_candidate_count": len(decisions),
        "bridge_failure_count": len(rejected),
        "bridge_fail_rate": len(rejected) / len(decisions) if decisions else None,
        "contextual_ir_item_count": contextual_ir,
        "accepted_ir_item_count": accepted_ir,
        "ir_atomic_card_count": sum(item.get("source_type") == "IR" for item in evidence),
        "ir_context_reference_count": sum(
            "IR" in sources for sources in ref_sources.values()
        ),
        "ir_quality_rejection_count": sum(item.get("passed") is not True for item in ir_quality),
        "ir_quality_warning_count": sum(len(item.get("warnings") or []) for item in ir_quality),
        "mechanism_types": sorted(item["evidence_type"] for item in reconciliation.get("mechanisms") or []),
        "outcome_types": sorted(item["evidence_type"] for item in reconciliation.get("outcomes") or []),
        "counter_types": sorted(item["evidence_type"] for item in reconciliation.get("counterevidence") or []),
        "audit_status": score["audit_status"],
        "input_tokens": int(company.llm_usage.input_tokens),
        "output_tokens": int(company.llm_usage.output_tokens),
        "cached_input_tokens": int(company.llm_usage.cached_input_tokens),
        "replayed_call_count": sum(call.get("replayed") is True for call in calls),
        "call_count": len(calls),
        "source_document_ids": list(company.source_document_ids),
        "treatment_compliant": bool(treatment.get("treatment_compliant", False)),
        "ir_available": bool(treatment.get("IR_AVAILABLE", False)),
        "ir_usable": bool(treatment.get("IR_USABLE", False)),
        "ir_document_count": int(treatment.get("ir_document_count", 0)),
        "usable_ir_document_count": int(
            treatment.get("usable_ir_document_count", 0)
        ),
        "available_ir_years": list(treatment.get("available_ir_years") or []),
        "usable_ir_years": list(treatment.get("usable_ir_years") or []),
        "accepted_ir_years": list(treatment.get("accepted_ir_years") or []),
        "longitudinal_ir_mode": bool(
            treatment.get("longitudinal_ir_mode", False)
        ),
        "longitudinal_ir_usable": bool(
            treatment.get("longitudinal_ir_usable", False)
        ),
        "longitudinal_treatment_compliant": bool(
            treatment.get("longitudinal_treatment_compliant", False)
        ),
        "treatment_accepted_ir_item_count": int(
            treatment.get("accepted_ir_item_count", 0)
        ),
        "frozen_base_input_sha256": treatment.get("frozen_base_input_sha256"),
        "base_atomic_unit_set_sha256": selection.get(
            "baseline_base_atomic_unit_set_sha256"
        ),
        "noncompliant_score_invariant_passed": bool(
            treatment.get("noncompliant_score_invariant_passed", False)
        ),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bridge_candidates = sum(row["bridge_candidate_count"] for row in rows)
    bridge_failures = sum(row["bridge_failure_count"] for row in rows)
    return {
        "company_count": len(rows),
        "score": _summary([row["score"] for row in rows]),
        "distinct_score_count": len({row["score"] for row in rows}),
        "score_eligible_rate": statistics.mean(row["score_eligible"] for row in rows),
        "evidence_sufficiency": _summary([row["evidence_sufficiency"] for row in rows]),
        "contextual_mechanism_count": _summary([row["contextual_mechanism_count"] for row in rows]),
        "contextual_outcome_count": _summary([row["contextual_outcome_count"] for row in rows]),
        "contextual_persistent_outcome_count": _summary([row["contextual_persistent_outcome_count"] for row in rows]),
        "contextual_counter_count": _summary([row["contextual_counter_count"] for row in rows]),
        "accepted_mechanism_count": _summary([row["accepted_mechanism_count"] for row in rows]),
        "accepted_outcome_count": _summary([row["accepted_outcome_count"] for row in rows]),
        "accepted_counter_count": _summary([row["accepted_counter_count"] for row in rows]),
        "bridge_fail_rate": bridge_failures / bridge_candidates if bridge_candidates else None,
        "bridge_failures": bridge_failures,
        "bridge_candidates": bridge_candidates,
        "contextual_ir_item_count": sum(row["contextual_ir_item_count"] for row in rows),
        "accepted_ir_item_count": sum(row["accepted_ir_item_count"] for row in rows),
        "ir_atomic_card_count": sum(row["ir_atomic_card_count"] for row in rows),
        "ir_context_reference_count": sum(row["ir_context_reference_count"] for row in rows),
        "ir_quality_rejected_company_count": sum(row["ir_quality_rejection_count"] > 0 for row in rows),
        "ir_document_count": _summary(
            [row["ir_document_count"] for row in rows]
        ),
        "usable_ir_document_count": _summary(
            [row["usable_ir_document_count"] for row in rows]
        ),
        "available_ir_year_count": _summary(
            [len(row["available_ir_years"]) for row in rows]
        ),
        "usable_ir_year_count": _summary(
            [len(row["usable_ir_years"]) for row in rows]
        ),
        "accepted_ir_year_count": _summary(
            [len(row["accepted_ir_years"]) for row in rows]
        ),
        "longitudinal_treatment_compliant_company_count": sum(
            row["longitudinal_treatment_compliant"] for row in rows
        ),
        "audit_status": dict(Counter(row["audit_status"] for row in rows)),
        "usage": {
            "input_tokens": sum(row["input_tokens"] for row in rows),
            "output_tokens": sum(row["output_tokens"] for row in rows),
            "cached_input_tokens": sum(row["cached_input_tokens"] for row in rows),
            "calls": sum(row["call_count"] for row in rows),
            "replayed_calls": sum(row["replayed_call_count"] for row in rows),
        },
    }


def _repeatability(
    baseline: dict[str, Any],
    repeated: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if repeated is None:
        return None
    common = sorted(set(baseline) & set(repeated))
    before = [float(baseline[ticker].moat_score.economic_moat_score) for ticker in common]
    after = [float(repeated[ticker].moat_score.economic_moat_score) for ticker in common]
    return {
        "company_count": len(common),
        "score_spearman": _spearman(before, after),
        "exact_score_match_rate": statistics.mean(a == b for a, b in zip(before, after, strict=True)) if common else None,
        "maximum_absolute_score_delta": max((abs(a - b) for a, b in zip(before, after, strict=True)), default=None),
    }


def _treatment_effect_repeatability(
    dart: dict[str, Any],
    ir: dict[str, Any],
    dart_repeat: dict[str, Any] | None,
    ir_repeat: dict[str, Any] | None,
    *,
    tickers: set[str] | None = None,
) -> dict[str, Any] | None:
    if dart_repeat is None or ir_repeat is None:
        return None
    common = sorted(set(dart) & set(ir) & set(dart_repeat) & set(ir_repeat))
    if tickers is not None:
        common = [ticker for ticker in common if ticker in tickers]
    original = [
        float(ir[ticker].moat_score.economic_moat_score)
        - float(dart[ticker].moat_score.economic_moat_score)
        for ticker in common
    ]
    repeated = [
        float(ir_repeat[ticker].moat_score.economic_moat_score)
        - float(dart_repeat[ticker].moat_score.economic_moat_score)
        for ticker in common
    ]

    def sign(value: float) -> int:
        return 1 if value > 0 else (-1 if value < 0 else 0)

    return {
        "company_count": len(common),
        "treatment_delta_spearman": _spearman(original, repeated),
        "exact_treatment_delta_match_rate": (
            statistics.mean(a == b for a, b in zip(original, repeated, strict=True))
            if common
            else None
        ),
        "treatment_direction_match_rate": (
            statistics.mean(sign(a) == sign(b) for a, b in zip(original, repeated, strict=True))
            if common
            else None
        ),
        "maximum_absolute_treatment_delta_difference": max(
            (abs(a - b) for a, b in zip(original, repeated, strict=True)),
            default=None,
        ),
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dart_result = _load_run(Path(args.dart_only).resolve())
    ir_result = _load_run(Path(args.dart_plus_ir).resolve())
    dart = _index(dart_result, "DART-only")
    ir = _index(ir_result, "DART+IR")
    if set(dart) != set(ir):
        raise ValueError("source-ablation lanes use different company sets")
    if dart_result.as_of != ir_result.as_of:
        raise ValueError("source-ablation lanes use different as_of timestamps")

    dart_repeat = _index(_load_run(Path(args.dart_only_repeat).resolve()), "DART-only repeat") if args.dart_only_repeat else None
    ir_repeat = _index(_load_run(Path(args.dart_plus_ir_repeat).resolve()), "DART+IR repeat") if args.dart_plus_ir_repeat else None
    dart_rows = {ticker: _company_metrics(dart[ticker]) for ticker in sorted(dart)}
    ir_rows = {ticker: _company_metrics(ir[ticker]) for ticker in sorted(ir)}
    repeat_ir_rows = (
        {ticker: _company_metrics(ir_repeat[ticker]) for ticker in sorted(ir_repeat)}
        if ir_repeat is not None
        else None
    )
    paired: list[dict[str, Any]] = []
    for ticker in sorted(dart):
        left, right = dart_rows[ticker], ir_rows[ticker]
        left_dart_ids = {value for value in left["source_document_ids"] if not value.startswith("KINDIR_")}
        right_dart_ids = {value for value in right["source_document_ids"] if not value.startswith("KINDIR_")}
        ir_ids = [value for value in right["source_document_ids"] if value.startswith("KINDIR_")]
        if left_dart_ids != right_dart_ids:
            raise ValueError(f"ticker {ticker}: DART source document set changed")
        if len(ir_ids) < args.minimum_ir_documents_per_company:
            raise ValueError(
                f"ticker {ticker}: only {len(ir_ids)} IR documents; "
                f"minimum={args.minimum_ir_documents_per_company}"
            )
        if (
            args.maximum_ir_documents_per_company is not None
            and len(ir_ids) > args.maximum_ir_documents_per_company
        ):
            raise ValueError(
                f"ticker {ticker}: {len(ir_ids)} IR documents; "
                f"maximum={args.maximum_ir_documents_per_company}"
            )
        paired.append(
            {
                "ticker": ticker,
                "issuer_name": left["issuer_name"],
                "dart_score": left["score"],
                "dart_plus_ir_score": right["score"],
                "score_delta": right["score"] - left["score"],
                "dart_evidence_sufficiency": left["evidence_sufficiency"],
                "dart_plus_ir_evidence_sufficiency": right["evidence_sufficiency"],
                "evidence_sufficiency_delta": right["evidence_sufficiency"] - left["evidence_sufficiency"],
                "contextual_mechanism_delta": right["contextual_mechanism_count"] - left["contextual_mechanism_count"],
                "contextual_outcome_delta": right["contextual_outcome_count"] - left["contextual_outcome_count"],
                "persistent_outcome_delta": right["contextual_persistent_outcome_count"] - left["contextual_persistent_outcome_count"],
                "counterevidence_delta": right["contextual_counter_count"] - left["contextual_counter_count"],
                "accepted_mechanism_delta": right["accepted_mechanism_count"] - left["accepted_mechanism_count"],
                "accepted_outcome_delta": right["accepted_outcome_count"] - left["accepted_outcome_count"],
                "accepted_counter_delta": right["accepted_counter_count"] - left["accepted_counter_count"],
                "bridge_fail_rate_delta": (
                    right["bridge_fail_rate"] - left["bridge_fail_rate"]
                    if right["bridge_fail_rate"] is not None and left["bridge_fail_rate"] is not None
                    else None
                ),
                "ir_atomic_card_count": right["ir_atomic_card_count"],
                "ir_context_reference_count": right["ir_context_reference_count"],
                "contextual_ir_item_count": right["contextual_ir_item_count"],
                "accepted_ir_item_count": right["accepted_ir_item_count"],
                "ir_document_count": right["ir_document_count"],
                "usable_ir_document_count": right["usable_ir_document_count"],
                "available_ir_year_count": len(right["available_ir_years"]),
                "usable_ir_year_count": len(right["usable_ir_years"]),
                "accepted_ir_year_count": len(right["accepted_ir_years"]),
                "accepted_ir_years": right["accepted_ir_years"],
                "longitudinal_ir_mode": right["longitudinal_ir_mode"],
                "longitudinal_ir_usable": right["longitudinal_ir_usable"],
                "longitudinal_treatment_compliant": right[
                    "longitudinal_treatment_compliant"
                ],
                "ir_quality_rejected": right["ir_quality_rejection_count"] > 0,
                "ir_available": right["ir_available"],
                "ir_usable": right["ir_usable"],
                "treatment_compliant": right["treatment_compliant"],
                "treatment_accepted_ir_item_count": right[
                    "treatment_accepted_ir_item_count"
                ],
                "dart_base_input_identical": (
                    left["frozen_base_input_sha256"]
                    == right["frozen_base_input_sha256"]
                ),
                "dart_atomic_selection_identical": (
                    left["base_atomic_unit_set_sha256"]
                    == right["base_atomic_unit_set_sha256"]
                ),
                "noncompliant_score_invariant_passed": right[
                    "noncompliant_score_invariant_passed"
                ],
                "added_mechanism_types": sorted(set(right["mechanism_types"]) - set(left["mechanism_types"])),
                "removed_mechanism_types": sorted(set(left["mechanism_types"]) - set(right["mechanism_types"])),
                "added_outcome_types": sorted(set(right["outcome_types"]) - set(left["outcome_types"])),
                "removed_outcome_types": sorted(set(left["outcome_types"]) - set(right["outcome_types"])),
            }
        )

    score_left = [row["dart_score"] for row in paired]
    score_right = [row["dart_plus_ir_score"] for row in paired]
    score_delta = [row["score_delta"] for row in paired]
    contract_failures = [
        row["ticker"]
        for row in paired
        if not row["dart_base_input_identical"]
        or not row["dart_atomic_selection_identical"]
        or not row["noncompliant_score_invariant_passed"]
    ]
    if contract_failures:
        raise ValueError(
            "paired source-ablation contract failed for tickers: "
            + ", ".join(contract_failures)
        )
    treatment_compliance_repeatability = None
    compliant_in_both: set[str] | None = None
    if repeat_ir_rows is not None:
        common_repeat = sorted(set(ir_rows) & set(repeat_ir_rows))
        main_compliant = {
            ticker for ticker in common_repeat if ir_rows[ticker]["treatment_compliant"]
        }
        repeat_compliant = {
            ticker
            for ticker in common_repeat
            if repeat_ir_rows[ticker]["treatment_compliant"]
        }
        compliant_in_both = main_compliant & repeat_compliant
        compliant_union = main_compliant | repeat_compliant
        treatment_compliance_repeatability = {
            "company_count": len(common_repeat),
            "main_compliant_count": len(main_compliant),
            "repeat_compliant_count": len(repeat_compliant),
            "compliant_in_both_count": len(compliant_in_both),
            "compliance_agreement_rate": (
                statistics.mean(
                    ir_rows[ticker]["treatment_compliant"]
                    == repeat_ir_rows[ticker]["treatment_compliant"]
                    for ticker in common_repeat
                )
                if common_repeat
                else None
            ),
            "compliant_set_jaccard": (
                len(compliant_in_both) / len(compliant_union)
                if compliant_union
                else None
            ),
            "compliant_in_both_tickers": sorted(compliant_in_both),
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "as_of": dart_result.as_of.isoformat(),
        "company_count": len(paired),
        "return_data_used": False,
        "assignment_contract": {
            "minimum_ir_documents_per_company": (
                args.minimum_ir_documents_per_company
            ),
            "maximum_ir_documents_per_company": (
                args.maximum_ir_documents_per_company
            ),
        },
        "treatment_contract_verified": True,
        "dart_only": _aggregate(list(dart_rows.values())),
        "dart_plus_ir": _aggregate(list(ir_rows.values())),
        "paired": {
            "score_spearman": _spearman(score_left, score_right),
            "score_delta": _summary(score_delta),
            "score_increase_count": sum(value > 0 for value in score_delta),
            "score_decrease_count": sum(value < 0 for value in score_delta),
            "score_unchanged_count": sum(value == 0 for value in score_delta),
            "evidence_sufficiency_increase_count": sum(row["evidence_sufficiency_delta"] > 0 for row in paired),
            "mechanism_coverage_increase_count": sum(row["contextual_mechanism_delta"] > 0 for row in paired),
            "outcome_coverage_increase_count": sum(row["contextual_outcome_delta"] > 0 for row in paired),
            "persistence_coverage_increase_count": sum(row["persistent_outcome_delta"] > 0 for row in paired),
            "counterevidence_increase_count": sum(row["counterevidence_delta"] > 0 for row in paired),
            "accepted_ir_item_company_count": sum(row["accepted_ir_item_count"] > 0 for row in paired),
            "treatment_compliant_company_count": sum(
                row["treatment_compliant"] for row in paired
            ),
            "ir_usable_company_count": sum(row["ir_usable"] for row in paired),
            "longitudinal_ir_usable_company_count": sum(
                row["longitudinal_ir_usable"] for row in paired
            ),
            "longitudinal_treatment_compliant_company_count": sum(
                row["longitudinal_treatment_compliant"] for row in paired
            ),
            "multi_year_accepted_ir_company_count": sum(
                row["accepted_ir_year_count"] >= 2 for row in paired
            ),
            "dart_base_input_identical_count": sum(
                row["dart_base_input_identical"] for row in paired
            ),
            "dart_atomic_selection_identical_count": sum(
                row["dart_atomic_selection_identical"] for row in paired
            ),
            "noncompliant_score_change_count": sum(
                not row["treatment_compliant"] and row["score_delta"] != 0
                for row in paired
            ),
            "ir_atomic_card_company_count": sum(row["ir_atomic_card_count"] > 0 for row in paired),
            "ir_context_reference_company_count": sum(row["ir_context_reference_count"] > 0 for row in paired),
            "no_ir_exposure_score_change_count": sum(
                row["ir_context_reference_count"] == 0
                and row["ir_atomic_card_count"] == 0
                and row["score_delta"] != 0
                for row in paired
            ),
        },
        "repeatability": {
            "dart_only": _repeatability(dart, dart_repeat),
            "dart_plus_ir": _repeatability(ir, ir_repeat),
            "treatment_compliance": treatment_compliance_repeatability,
            "treatment_effect": _treatment_effect_repeatability(
                dart,
                ir,
                dart_repeat,
                ir_repeat,
            ),
            "treatment_effect_compliant_in_both": _treatment_effect_repeatability(
                dart,
                ir,
                dart_repeat,
                ir_repeat,
                tickers=compliant_in_both,
            ),
        },
        "rows": paired,
    }
    output = Path(args.output).resolve()
    _write_json(output / "source-ablation-report.json", report)
    list_fields = {
        "accepted_ir_years",
        "added_mechanism_types",
        "removed_mechanism_types",
        "added_outcome_types",
        "removed_outcome_types",
    }
    fields = [key for key in paired[0] if key not in list_fields]
    csv_rows = [
        {
            **row,
            "accepted_ir_years": ";".join(
                str(year) for year in row["accepted_ir_years"]
            ),
            "added_mechanism_types": ";".join(row["added_mechanism_types"]),
            "removed_mechanism_types": ";".join(row["removed_mechanism_types"]),
            "added_outcome_types": ";".join(row["added_outcome_types"]),
            "removed_outcome_types": ";".join(row["removed_outcome_types"]),
        }
        for row in paired
    ]
    _write_csv(
        output / "source-ablation-paired.csv",
        csv_rows,
        [
            *fields,
            "accepted_ir_years",
            "added_mechanism_types",
            "removed_mechanism_types",
            "added_outcome_types",
            "removed_outcome_types",
        ],
    )
    _write_markdown(output / "source-ablation-report.md", report)
    return report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    left, right, paired = report["dart_only"], report["dart_plus_ir"], report["paired"]
    assignment = report["assignment_contract"]
    minimum_ir = assignment["minimum_ir_documents_per_company"]
    maximum_ir = assignment["maximum_ir_documents_per_company"]
    assignment_text = (
        f"회사별 PIT-가용 KIND IR PDF {minimum_ir}~{maximum_ir}건"
        if maximum_ir is not None and maximum_ir != minimum_ir
        else f"회사별 PIT-가용 KIND IR PDF {minimum_ir}건"
    )
    lines = [
        "# DART-only vs DART+IR Source Ablation",
        "",
        f"- PIT 기준: {report['as_of']}",
        f"- 동일 표본: {report['company_count']}개사",
        "- 선택에 미래수익률 사용: 아니오",
        f"- Assignment: {assignment_text} 추가",
        "- Treatment-compliant: accepted IR 근거가 score-producing claim까지 살아남은 경우만 인정",
        "- 비채택/품질탈락 IR: frozen DART score를 그대로 유지",
        "",
        "## 핵심 결과",
        "",
        "| 지표 | DART-only | DART+IR |",
        "|---|---:|---:|",
        f"| 평균 evidence sufficiency | {left['evidence_sufficiency']['mean']:.2f} | {right['evidence_sufficiency']['mean']:.2f} |",
        f"| 평균 contextual mechanisms | {left['contextual_mechanism_count']['mean']:.2f} | {right['contextual_mechanism_count']['mean']:.2f} |",
        f"| 평균 contextual outcomes | {left['contextual_outcome_count']['mean']:.2f} | {right['contextual_outcome_count']['mean']:.2f} |",
        f"| 평균 persistent outcomes | {left['contextual_persistent_outcome_count']['mean']:.2f} | {right['contextual_persistent_outcome_count']['mean']:.2f} |",
        f"| 평균 counterevidence | {left['contextual_counter_count']['mean']:.2f} | {right['contextual_counter_count']['mean']:.2f} |",
        f"| Bridge fail rate | {_pct(left['bridge_fail_rate'])} | {_pct(right['bridge_fail_rate'])} |",
        f"| 평균 MOAT score | {left['score']['mean']:.3f} | {right['score']['mean']:.3f} |",
        f"| Score eligible rate | {_pct(left['score_eligible_rate'])} | {_pct(right['score_eligible_rate'])} |",
        "",
        "## Paired 변화",
        "",
        f"- Score rank Spearman: {paired['score_spearman']}",
        f"- Score 상승/하락/동일: {paired['score_increase_count']}/{paired['score_decrease_count']}/{paired['score_unchanged_count']}",
        f"- Evidence sufficiency 증가: {paired['evidence_sufficiency_increase_count']}개사",
        f"- Mechanism/outcome/persistence coverage 증가: {paired['mechanism_coverage_increase_count']}/{paired['outcome_coverage_increase_count']}/{paired['persistence_coverage_increase_count']}개사",
        f"- Counterevidence 증가: {paired['counterevidence_increase_count']}개사",
        f"- 최종 accepted item에 IR 근거가 남은 회사: {paired['accepted_ir_item_company_count']}개사",
        f"- Treatment-compliant 회사: {paired['treatment_compliant_company_count']}개사",
        f"- IR usable 회사: {paired['ir_usable_company_count']}개사",
        f"- Longitudinal IR usable 회사: {paired['longitudinal_ir_usable_company_count']}개사",
        f"- Longitudinal treatment-compliant 회사: {paired['longitudinal_treatment_compliant_company_count']}개사",
        f"- 2개 이상 연도의 accepted IR 근거가 남은 회사: {paired['multi_year_accepted_ir_company_count']}개사",
        f"- DART base 입력 byte-identical: {paired['dart_base_input_identical_count']}/{report['company_count']}",
        f"- DART atomic selection 동일: {paired['dart_atomic_selection_identical_count']}/{report['company_count']}",
        f"- Treatment 미채택인데 점수가 변한 회사: {paired['noncompliant_score_change_count']}개사",
        f"- IR atomic card가 생성된 회사: {paired['ir_atomic_card_company_count']}개사",
        f"- Broad context에 IR reference가 포함된 회사: {paired['ir_context_reference_company_count']}개사",
        f"- IR 노출 없이 점수가 변한 회사: {paired['no_ir_exposure_score_change_count']}개사",
        "",
        "## 해석 제한",
        "",
        "- 이 단계는 source adequacy 실험이며 forward return/IC/Q5-Q1을 계산하지 않았다.",
        f"- IR parser quality gate 미통과 회사: {right['ir_quality_rejected_company_count']}개사. 원시 경고와 coverage는 결과에 보존했다.",
        "- IR은 management claim으로 유지했으며 DART disclosed fact와 합쳐 확정 fact로 승격하지 않았다.",
        "",
    ]
    repeat = report["repeatability"]
    if repeat["dart_only"] or repeat["dart_plus_ir"]:
        lines.extend(["## Repeatability pilot", ""])
        for label in ("dart_only", "dart_plus_ir"):
            item = repeat[label]
            if item:
                lines.append(
                    f"- {label}: n={item['company_count']}, Spearman={item['score_spearman']}, exact={_pct(item['exact_score_match_rate'])}, max |Δ|={item['maximum_absolute_score_delta']}"
                )
        treatment = repeat.get("treatment_effect")
        if treatment:
            lines.append(
                "- intention-to-treat delta (전체 표본 진단용): "
                f"n={treatment['company_count']}, "
                f"delta Spearman={treatment['treatment_delta_spearman']}, "
                f"exact={_pct(treatment['exact_treatment_delta_match_rate'])}, "
                f"direction={_pct(treatment['treatment_direction_match_rate'])}, "
                "max |delta difference|="
                f"{treatment['maximum_absolute_treatment_delta_difference']}"
            )
        compliance = repeat.get("treatment_compliance")
        if compliance:
            lines.append(
                "- treatment compliance: "
                f"main/repeat/both={compliance['main_compliant_count']}/"
                f"{compliance['repeat_compliant_count']}/"
                f"{compliance['compliant_in_both_count']}, "
                f"agreement={_pct(compliance['compliance_agreement_rate'])}, "
                f"Jaccard={_pct(compliance['compliant_set_jaccard'])}"
            )
        compliant_treatment = repeat.get("treatment_effect_compliant_in_both")
        if compliant_treatment:
            lines.append(
                "- accepted-IR treatment effect (양 실행 모두 compliant): "
                f"n={compliant_treatment['company_count']}, "
                f"delta Spearman={compliant_treatment['treatment_delta_spearman']}, "
                f"exact={_pct(compliant_treatment['exact_treatment_delta_match_rate'])}, "
                f"direction={_pct(compliant_treatment['treatment_direction_match_rate'])}, "
                "max |delta difference|="
                f"{compliant_treatment['maximum_absolute_treatment_delta_difference']}"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a paired DART-only vs DART+IR source ablation")
    parser.add_argument("--dart-only", required=True)
    parser.add_argument("--dart-plus-ir", required=True)
    parser.add_argument("--dart-only-repeat")
    parser.add_argument("--dart-plus-ir-repeat")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--minimum-ir-documents-per-company",
        type=int,
        default=1,
        help="Minimum assigned PIT-available IR documents required per company.",
    )
    parser.add_argument(
        "--maximum-ir-documents-per-company",
        type=int,
        default=1,
        help=(
            "Maximum assigned IR documents allowed per company. Set to 5 for "
            "the longitudinal annual-snapshot experiment."
        ),
    )
    return parser


if __name__ == "__main__":
    result = evaluate(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
