from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from moatrader.context import ContextEvidenceReference
import moatrader.evidence.validation as moat_validation
from moatrader.evidence.models import (
    ContextualMoatAssessment,
    MoatScore,
    ReconciledMoatAssessment,
)
from moatrader.evidence.validation import (
    derive_audited_moat_rank_score,
    normalize_contextual_moat_rank_assessment,
)
from moatrader.runner.engine import RUNNER_VERSION

try:
    from scripts.merge_kr_signal_panel import spearman
    from scripts.evaluate_signal_panel import (
        nonoverlapping_quantile_spread,
        signal_tie_diagnostics,
        winsorize,
    )
except ModuleNotFoundError:  # Direct ``python scripts\...py`` execution.
    from merge_kr_signal_panel import spearman
    from evaluate_signal_panel import (
        nonoverlapping_quantile_spread,
        signal_tie_diagnostics,
        winsorize,
    )


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def materialize_company(company_dir: Path) -> dict[str, object]:
    raw = ContextualMoatAssessment.model_validate(
        _load_json(company_dir / "contextual-moat-assessment-raw.json")
    )
    context = dict(_load_json(company_dir / "moat-strength-context.json"))
    references = [
        ContextEvidenceReference.model_validate(item)
        for item in context.get("references", [])
    ]
    rank_input, repair = normalize_contextual_moat_rank_assessment(raw, references)
    reconciled = ReconciledMoatAssessment.model_validate(
        _load_json(company_dir / "moat-reconciliation.json")
    )
    public = MoatScore.model_validate(_load_json(company_dir / "moat-score.json"))
    rank_score = derive_audited_moat_rank_score(
        rank_input,
        reconciled,
        score_eligible=public.score_eligible,
    )
    return {
        "date": public.as_of.isoformat(),
        "ticker": company_dir.name,
        "score_eligible": public.score_eligible,
        "eligibility_status": public.eligibility_status.value,
        "audit_status": public.audit_status.value,
        "economic_moat_score": public.economic_moat_score,
        "economic_moat_rank_score": rank_score,
        "accepted_mechanism_count": len(reconciled.mechanisms),
        "accepted_outcome_count": len(reconciled.outcomes),
        "counterevidence_count": len(reconciled.counterevidence),
        "rank_repair_action_count": repair["action_count"],
        "company_directory": str(company_dir.resolve()),
    }


def materialize_root(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for score_path in sorted(root.rglob("moat-score.json")):
        company_dir = score_path.parent
        required = (
            "contextual-moat-assessment-raw.json",
            "moat-strength-context.json",
            "moat-reconciliation.json",
        )
        if not all((company_dir / name).is_file() for name in required):
            continue
        rows.append(materialize_company(company_dir))
    return rows


def distribution_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    eligible = [
        row
        for row in rows
        if row["score_eligible"] and row["economic_moat_rank_score"] is not None
    ]
    by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in eligible:
        by_date[str(row["date"])].append(row)
    dates = []
    for date, items in sorted(by_date.items()):
        public = [float(row["economic_moat_score"]) for row in items]
        rank = [float(row["economic_moat_rank_score"]) for row in items]
        public_counts = Counter(public)
        rank_counts = Counter(rank)
        dates.append(
            {
                "date": date,
                "eligible_count": len(items),
                "public_distinct_score_count": len(public_counts),
                "rank_distinct_score_count": len(rank_counts),
                "public_max_single_score_share": max(public_counts.values()) / len(items),
                "rank_max_single_score_share": max(rank_counts.values()) / len(items),
                "public_to_rank_spearman": spearman(public, rank),
            }
        )
    return {
        "row_count": len(rows),
        "eligible_count": len(eligible),
        "ineligible_rank_is_none": all(
            row["economic_moat_rank_score"] is None
            for row in rows
            if not row["score_eligible"]
        ),
        "dates": dates,
    }


def repeat_metrics(
    baseline: list[dict[str, object]],
    repeat: list[dict[str, object]],
) -> dict[str, object]:
    left = {
        (str(row["date"]), str(row["ticker"])): row
        for row in baseline
        if row["economic_moat_rank_score"] is not None
    }
    right = {
        (str(row["date"]), str(row["ticker"])): row
        for row in repeat
        if row["economic_moat_rank_score"] is not None
    }
    common = sorted(set(left) & set(right))
    left_scores = [float(left[key]["economic_moat_rank_score"]) for key in common]
    right_scores = [float(right[key]["economic_moat_rank_score"]) for key in common]
    deltas = [abs(a - b) for a, b in zip(left_scores, right_scores, strict=True)]
    return {
        "common_eligible_count": len(common),
        "rank_score_spearman": spearman(left_scores, right_scores),
        "median_absolute_delta": statistics.median(deltas) if deltas else None,
        "max_absolute_delta": max(deltas) if deltas else None,
    }


def economic_metrics(
    shadow_rows: list[dict[str, object]],
    economic_panel: Path,
) -> dict[str, object]:
    with economic_panel.open("r", encoding="utf-8-sig", newline="") as stream:
        realized = {row["ticker"].zfill(6): row for row in csv.DictReader(stream)}
    matched = [
        (row, realized[str(row["ticker"]).zfill(6)])
        for row in shadow_rows
        if row["score_eligible"]
        and row["economic_moat_rank_score"] is not None
        and str(row["ticker"]).zfill(6) in realized
        and realized[str(row["ticker"]).zfill(6)].get("forward_return", "") != ""
    ]
    tickers = [str(row["ticker"]).zfill(6) for row, _ in matched]
    public = [float(row["economic_moat_score"]) for row, _ in matched]
    rank = [float(row["economic_moat_rank_score"]) for row, _ in matched]
    returns = [float(realized_row["forward_return"]) for _, realized_row in matched]
    clipped = winsorize(returns)
    public_spread, public_top, public_bottom = nonoverlapping_quantile_spread(
        public, clipped, tickers, seed="economic-public"
    )
    rank_spread, rank_top, rank_bottom = nonoverlapping_quantile_spread(
        rank, clipped, tickers, seed="economic-rank"
    )
    return {
        "matched_eligible_count": len(matched),
        "public_rank_ic": spearman(public, clipped),
        "raw_rank_score_ic": spearman(rank, clipped),
        "public_q5_minus_q1_tie_randomized": public_spread,
        "raw_rank_q5_minus_q1_tie_randomized": rank_spread,
        "public_top_bottom_counts": [public_top, public_bottom],
        "rank_top_bottom_counts": [rank_top, rank_bottom],
        "public_ties": signal_tie_diagnostics(public),
        "rank_ties": signal_tie_diagnostics(rank),
    }


def old_holistic_metrics(
    shadow_rows: list[dict[str, object]],
    old_scores_path: Path,
) -> dict[str, object]:
    with old_scores_path.open("r", encoding="utf-8-sig", newline="") as stream:
        old = {
            ((row.get("as_of") or "").strip(), (row.get("stock_code") or "").zfill(6)):
            float(row["economic_moat_score_100"])
            for row in csv.DictReader(stream)
            if row.get("as_of") and row.get("stock_code") and row.get("economic_moat_score_100")
        }
    by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in shadow_rows:
        if row["score_eligible"] and row["economic_moat_rank_score"] is not None:
            by_date[str(row["date"])].append(row)
    dates = []
    for date, rows in sorted(by_date.items()):
        pairs = [
            (row, old[(date, str(row["ticker"]).zfill(6))])
            for row in rows
            if (date, str(row["ticker"]).zfill(6)) in old
        ]
        rank_rho = spearman(
            [float(row["economic_moat_rank_score"]) for row, _ in pairs],
            [old_score for _, old_score in pairs],
        )
        public_rho = spearman(
            [float(row["economic_moat_score"]) for row, _ in pairs],
            [old_score for _, old_score in pairs],
        )
        dates.append(
            {
                "date": date,
                "matched_count": len(pairs),
                "public_to_old_spearman": public_rho,
                "rank_to_old_spearman": rank_rho,
            }
        )
    present = [row["rank_to_old_spearman"] for row in dates if row["rank_to_old_spearman"] is not None]
    return {
        "mean_rank_to_old_spearman": statistics.mean(present) if present else None,
        "positive_rank_to_old_dates": sum(value > 0 for value in present),
        "dates": dates,
    }
def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["date", "ticker"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute rank-only MOAT scores from saved raw artifacts without API calls."
    )
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--economic-panel",
        type=Path,
        help="optional ticker/forward_return panel evaluated against the final root",
    )
    parser.add_argument(
        "--old-scores",
        type=Path,
        help="optional legacy holistic score CSV used as a bridge diagnostic",
    )
    args = parser.parse_args()

    roots = [root.resolve() for root in args.root]
    panels = {str(root): materialize_root(root) for root in roots}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, object] = {}
    for index, root in enumerate(roots, start=1):
        rows = panels[str(root)]
        _write_csv(output / f"root-{index}-rank-shadow.csv", rows)
        summaries[str(root)] = distribution_metrics(rows)
    report: dict[str, object] = {
        "schema_version": "moatrader-moat-rank-shadow/1",
        "runner_version": RUNNER_VERSION,
        "rank_reducer_contract_sha256": hashlib.sha256(
            "\n\n".join(
                inspect.getsource(function)
                for function in (
                    moat_validation._normalize_contextual_moat_assessment,
                    normalize_contextual_moat_rank_assessment,
                    moat_validation._economic_moat_strength_score,
                    derive_audited_moat_rank_score,
                )
            ).encode("utf-8")
        ).hexdigest(),
        "api_calls": 0,
        "roots": summaries,
    }
    repeat_passed = False
    if len(roots) >= 2:
        report["repeat"] = repeat_metrics(
            panels[str(roots[0])], panels[str(roots[1])]
        )
        repeat_passed = (
            report["repeat"]["rank_score_spearman"] is not None
            and report["repeat"]["rank_score_spearman"] >= 0.90
        )
    if args.economic_panel is not None:
        report["economic_diagnostic"] = economic_metrics(
            panels[str(roots[-1])], args.economic_panel.resolve()
        )
    old_passed = False
    if args.old_scores is not None:
        report["old_holistic_diagnostic"] = old_holistic_metrics(
            panels[str(roots[0])], args.old_scores.resolve()
        )
        old_summary = report["old_holistic_diagnostic"]
        old_passed = bool(
            old_summary["mean_rank_to_old_spearman"] is not None
            and old_summary["mean_rank_to_old_spearman"] >= 0.50
            and old_summary["positive_rank_to_old_dates"] >= 3
        )
    baseline_dates = list(summaries[str(roots[0])]["dates"])
    resolution_passed = bool(baseline_dates) and all(
        row["rank_distinct_score_count"] >= row["public_distinct_score_count"]
        and row["rank_max_single_score_share"] <= row["public_max_single_score_share"]
        for row in baseline_dates
    ) and any(
        row["rank_distinct_score_count"] > row["public_distinct_score_count"]
        for row in baseline_dates
    )
    report["production_gate"] = {
        "minimum_repeat_rank_spearman": 0.90,
        "minimum_mean_old_holistic_spearman": 0.50,
        "minimum_positive_old_holistic_dates": 3,
        "repeat_rank_passed": repeat_passed,
        "old_holistic_bridge_passed": old_passed,
        "resolution_and_tie_share_passed": resolution_passed,
        "passed": repeat_passed and old_passed and resolution_passed,
    }
    (output / "rank-shadow-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
