from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "moatrader-ir-ablation-mode-comparison/1"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rate(count: int | float | None, total: int) -> float | None:
    if count is None or total <= 0:
        return None
    return float(count) / total


def _metric(report: dict[str, Any]) -> dict[str, Any]:
    n = int(report["company_count"])
    paired = report["paired"]
    repeat = report.get("repeatability") or {}
    treatment = repeat.get("treatment_effect_compliant_in_both") or {}
    ir_repeat = repeat.get("dart_plus_ir") or {}
    multi_year_count = int(paired.get("multi_year_accepted_ir_company_count", 0))
    longitudinal_compliant = int(
        paired.get(
            "longitudinal_treatment_compliant_company_count",
            paired.get("treatment_compliant_company_count", 0),
        )
    )
    return {
        "company_count": n,
        "ir_usable_company_rate": _rate(
            paired.get("ir_usable_company_count"), n
        ),
        "treatment_compliant_company_rate": _rate(
            paired.get("treatment_compliant_company_count"), n
        ),
        "longitudinal_treatment_compliant_company_rate": _rate(
            longitudinal_compliant, n
        ),
        "multi_year_accepted_ir_company_rate": _rate(multi_year_count, n),
        "evidence_sufficiency_increase_rate": _rate(
            paired.get("evidence_sufficiency_increase_count"), n
        ),
        "mechanism_coverage_increase_rate": _rate(
            paired.get("mechanism_coverage_increase_count"), n
        ),
        "outcome_coverage_increase_rate": _rate(
            paired.get("outcome_coverage_increase_count"), n
        ),
        "persistence_coverage_increase_rate": _rate(
            paired.get("persistence_coverage_increase_count"), n
        ),
        "counterevidence_increase_rate": _rate(
            paired.get("counterevidence_increase_count"), n
        ),
        "score_change_rate": _rate(
            int(paired.get("score_increase_count", 0))
            + int(paired.get("score_decrease_count", 0)),
            n,
        ),
        "noncompliant_score_change_count": int(
            paired.get("noncompliant_score_change_count", 0)
        ),
        "dart_plus_ir_repeat_spearman": ir_repeat.get("score_spearman"),
        "dart_plus_ir_repeat_exact_rate": ir_repeat.get(
            "exact_score_match_rate"
        ),
        "accepted_ir_delta_repeat_company_count": treatment.get("company_count"),
        "accepted_ir_delta_repeat_spearman": treatment.get(
            "treatment_delta_spearman"
        ),
        "accepted_ir_delta_repeat_exact_rate": treatment.get(
            "exact_treatment_delta_match_rate"
        ),
        "accepted_ir_delta_repeat_direction_rate": treatment.get(
            "treatment_direction_match_rate"
        ),
    }


def _greater(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and left > right


def _not_worse(left: float | None, right: float | None, tolerance: float) -> bool:
    return left is not None and right is not None and left >= right - tolerance


def compare(single: dict[str, Any], longitudinal: dict[str, Any]) -> dict[str, Any]:
    single_metrics = _metric(single)
    longitudinal_metrics = _metric(longitudinal)
    criteria = {
        "multi_year_evidence_reached_scoring": (
            (longitudinal_metrics["multi_year_accepted_ir_company_rate"] or 0.0)
            > 0.0
        ),
        "persistence_coverage_rate_improved": _greater(
            longitudinal_metrics["persistence_coverage_increase_rate"],
            single_metrics["persistence_coverage_increase_rate"],
        ),
        "accepted_ir_delta_spearman_improved": _greater(
            longitudinal_metrics["accepted_ir_delta_repeat_spearman"],
            single_metrics["accepted_ir_delta_repeat_spearman"],
        ),
        "accepted_ir_delta_exact_not_worse": _not_worse(
            longitudinal_metrics["accepted_ir_delta_repeat_exact_rate"],
            single_metrics["accepted_ir_delta_repeat_exact_rate"],
            0.05,
        ),
        "accepted_ir_delta_direction_not_worse": _not_worse(
            longitudinal_metrics["accepted_ir_delta_repeat_direction_rate"],
            single_metrics["accepted_ir_delta_repeat_direction_rate"],
            0.05,
        ),
        "dart_plus_ir_repeat_spearman_not_worse": _not_worse(
            longitudinal_metrics["dart_plus_ir_repeat_spearman"],
            single_metrics["dart_plus_ir_repeat_spearman"],
            0.05,
        ),
        "noncompliant_score_invariant": (
            longitudinal_metrics["noncompliant_score_change_count"] == 0
        ),
    }
    required = (
        "multi_year_evidence_reached_scoring",
        "persistence_coverage_rate_improved",
        "accepted_ir_delta_spearman_improved",
        "accepted_ir_delta_exact_not_worse",
        "accepted_ir_delta_direction_not_worse",
        "dart_plus_ir_repeat_spearman_not_worse",
        "noncompliant_score_invariant",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "return_data_used": False,
        "comparison_basis": (
            "company-count-normalized source-adequacy and independent-repeat metrics"
        ),
        "single_ir": single_metrics,
        "longitudinal_ir": longitudinal_metrics,
        "pre_registered_diagnostic_criteria": criteria,
        "sensor_connection_supported": all(criteria[key] for key in required),
        "interpretation_limit": (
            "This verdict only evaluates the IR evidence lane. It does not establish "
            "MOAT alpha or authorize forward-return analysis."
        ),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_markdown(path: Path, result: dict[str, Any]) -> None:
    single = result["single_ir"]
    longitudinal = result["longitudinal_ir"]
    fields = (
        "company_count",
        "ir_usable_company_rate",
        "treatment_compliant_company_rate",
        "longitudinal_treatment_compliant_company_rate",
        "multi_year_accepted_ir_company_rate",
        "mechanism_coverage_increase_rate",
        "outcome_coverage_increase_rate",
        "persistence_coverage_increase_rate",
        "counterevidence_increase_rate",
        "score_change_rate",
        "dart_plus_ir_repeat_spearman",
        "dart_plus_ir_repeat_exact_rate",
        "accepted_ir_delta_repeat_company_count",
        "accepted_ir_delta_repeat_spearman",
        "accepted_ir_delta_repeat_exact_rate",
        "accepted_ir_delta_repeat_direction_rate",
    )
    lines = [
        "# Single IR vs Longitudinal IR",
        "",
        "- Forward-return data used: no",
        "- Comparison: source adequacy and independent repeat only",
        "",
        "| Metric | Single IR | Longitudinal IR |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {field} | {_fmt(single[field])} | {_fmt(longitudinal[field])} |"
        for field in fields
    )
    lines.extend(["", "## Pre-registered diagnostic criteria", ""])
    lines.extend(
        f"- {name}: {'PASS' if passed else 'FAIL'}"
        for name, passed in result["pre_registered_diagnostic_criteria"].items()
    )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            (
                "IR sensor connection supported. A separate forward-return experiment "
                "may now be considered."
                if result["sensor_connection_supported"]
                else "IR sensor connection not supported under the frozen criteria. "
                "Do not proceed to forward-return analysis on this evidence."
            ),
            "",
            result["interpretation_limit"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    result = compare(
        _read(Path(args.single_report).resolve()),
        _read(Path(args.longitudinal_report).resolve()),
    )
    (output / "ir-ablation-mode-comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(output / "ir-ablation-mode-comparison.md", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare single-snapshot and longitudinal IR ablations"
    )
    parser.add_argument("--single-report", required=True)
    parser.add_argument("--longitudinal-report", required=True)
    parser.add_argument("--output", required=True)
    return parser


if __name__ == "__main__":
    print(json.dumps(main(build_parser().parse_args()), ensure_ascii=False, indent=2))
