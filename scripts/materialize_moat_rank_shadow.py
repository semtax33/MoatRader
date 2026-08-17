from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import moatrader.evidence.validation as moat_validation
from moatrader.context import ContextEvidenceReference
from moatrader.evidence.models import (
    ContextualMoatAssessment,
    MoatScore,
    ReconciledMoatAssessment,
)
from moatrader.evidence.validation import (
    derive_rank_refinement,
    derive_raw_ordinal_shadow_score,
    repair_contextual_moat_structure,
)
from moatrader.runner.engine import RUNNER_VERSION

try:
    from scripts.evaluate_signal_panel import (
        nonoverlapping_quantile_spread,
        signal_tie_diagnostics,
        winsorize,
    )
    from scripts.merge_kr_signal_panel import spearman
except ModuleNotFoundError:  # Direct ``python scripts\...py`` execution.
    from evaluate_signal_panel import (
        nonoverlapping_quantile_spread,
        signal_tie_diagnostics,
        winsorize,
    )
    from merge_kr_signal_panel import spearman


RankKey = tuple[float, ...]
CANDIDATE_PUBLIC = "PUBLIC_ONLY"
CANDIDATE_RAW = "PUBLIC_PLUS_RAW_WITHIN_BUCKET"
CANDIDATE_STABLE = "PUBLIC_PLUS_STABLE_COMPONENT_KEY"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _rank_key(row: dict[str, object], candidate: str) -> RankKey:
    public = float(row["economic_moat_score"])
    if candidate == CANDIDATE_PUBLIC:
        return (public,)
    if candidate == CANDIDATE_RAW:
        raw = row["raw_ordinal_shadow_score"]
        return (public,) if raw is None else (public, float(raw))
    if candidate == CANDIDATE_STABLE:
        if row["rank_refinement_status"] != "STABLE_COMPONENTS":
            return (public,)
        return (
            public,
            float(row["rank_mechanism_component"]),
            float(row["rank_outcome_component"]),
            float(row["rank_durability_component"]),
            -float(row["rank_counter_component"]),
        )
    raise ValueError(f"unknown rank candidate: {candidate}")


def _rank_keys(rows: list[dict[str, object]], candidate: str) -> list[RankKey]:
    if candidate != CANDIDATE_STABLE:
        return [_rank_key(row, candidate) for row in rows]
    complete_by_public: dict[float, bool] = {}
    for row in rows:
        public = float(row["economic_moat_score"])
        complete_by_public[public] = (
            complete_by_public.get(public, True)
            and row["rank_refinement_status"] == "STABLE_COMPONENTS"
        )
    return [
        _rank_key(row, candidate)
        if complete_by_public[float(row["economic_moat_score"])]
        else (float(row["economic_moat_score"]),)
        for row in rows
    ]


def _average_ranks(values: list[RankKey]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2
        for position in range(cursor, end):
            result[ordered[position]] = rank
        cursor = end
    return result


def materialize_company(company_dir: Path) -> dict[str, object]:
    raw = ContextualMoatAssessment.model_validate(
        _load_json(company_dir / "contextual-moat-assessment-raw.json")
    )
    context = dict(_load_json(company_dir / "moat-strength-context.json"))
    references = [
        ContextEvidenceReference.model_validate(item)
        for item in context.get("references", [])
    ]
    structurally_repaired, repair = repair_contextual_moat_structure(raw, references)
    reconciled = ReconciledMoatAssessment.model_validate(
        _load_json(company_dir / "moat-reconciliation.json")
    )
    public = MoatScore.model_validate(_load_json(company_dir / "moat-score.json"))
    raw_shadow = derive_raw_ordinal_shadow_score(
        structurally_repaired,
        reconciled,
        score_eligible=public.score_eligible,
    )
    refinement, refinement_status = derive_rank_refinement(
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
        "raw_ordinal_shadow_score": raw_shadow,
        "rank_refinement_status": refinement_status.value,
        "rank_mechanism_component": (
            refinement.mechanism_component if refinement else None
        ),
        "rank_outcome_component": refinement.outcome_component if refinement else None,
        "rank_durability_component": (
            refinement.durability_component if refinement else None
        ),
        "rank_counter_component": refinement.counter_component if refinement else None,
        "accepted_mechanism_count": len(reconciled.mechanisms),
        "accepted_outcome_count": len(reconciled.outcomes),
        "counterevidence_count": len(reconciled.counterevidence),
        "structural_repair_action_count": repair["action_count"],
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
        if all((company_dir / name).is_file() for name in required):
            rows.append(materialize_company(company_dir))
    return rows


def _key_distribution(keys: list[RankKey]) -> dict[str, object]:
    counts = Counter(keys)
    return {
        "distinct_rank_count": len(counts),
        "max_single_rank_share": max(counts.values()) / len(keys) if keys else None,
    }


def distribution_metrics(rows: list[dict[str, object]]) -> dict[str, object]:
    eligible = [row for row in rows if row["score_eligible"]]
    by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in eligible:
        by_date[str(row["date"])].append(row)
    dates = []
    for date, items in sorted(by_date.items()):
        candidates = {
            candidate: _key_distribution(
                _rank_keys(items, candidate)
            )
            for candidate in (
                CANDIDATE_PUBLIC,
                CANDIDATE_RAW,
                CANDIDATE_STABLE,
            )
        }
        dates.append(
            {
                "date": date,
                "eligible_count": len(items),
                "candidates": candidates,
            }
        )
    return {
        "row_count": len(rows),
        "eligible_count": len(eligible),
        "ineligible_candidates_are_null": all(
            row["raw_ordinal_shadow_score"] is None
            and row["rank_refinement_status"] == "INELIGIBLE"
            for row in rows
            if not row["score_eligible"]
        ),
        "dates": dates,
    }


def _pairwise_order_agreement(left: list[RankKey], right: list[RankKey]) -> float:
    agreements = []
    for first in range(len(left)):
        for second in range(first + 1, len(left)):
            left_sign = (left[first] > left[second]) - (left[first] < left[second])
            right_sign = (right[first] > right[second]) - (right[first] < right[second])
            agreements.append(left_sign == right_sign)
    return statistics.mean(agreements) if agreements else 1.0


def repeat_candidate_metrics(
    baseline: list[dict[str, object]],
    repeat: list[dict[str, object]],
) -> dict[str, object]:
    left = {
        (str(row["date"]), str(row["ticker"])): row
        for row in baseline
        if row["score_eligible"]
    }
    right = {
        (str(row["date"]), str(row["ticker"])): row
        for row in repeat
        if row["score_eligible"]
    }
    common = sorted(set(left) & set(right))
    candidates = {}
    for candidate in (CANDIDATE_PUBLIC, CANDIDATE_RAW, CANDIDATE_STABLE):
        left_rows = [left[key] for key in common]
        right_rows = [right[key] for key in common]
        left_keys = _rank_keys(left_rows, candidate)
        right_keys = _rank_keys(right_rows, candidate)
        left_ranks = _average_ranks(left_keys)
        right_ranks = _average_ranks(right_keys)
        deltas = [
            abs(a - b) for a, b in zip(left_ranks, right_ranks, strict=True)
        ]
        rho = spearman(left_keys, right_keys)
        if rho is None and left_keys == right_keys:
            rho = 1.0
        candidates[candidate] = {
            "rank_key_spearman": rho,
            "pairwise_order_agreement": _pairwise_order_agreement(
                left_keys, right_keys
            ),
            "median_absolute_rank_delta": (
                statistics.median(deltas) if deltas else None
            ),
            "max_absolute_rank_delta": max(deltas) if deltas else None,
        }
    return {"common_eligible_count": len(common), "candidates": candidates}


def old_holistic_metrics(
    shadow_rows: list[dict[str, object]],
    old_scores_path: Path,
    *,
    candidate: str,
) -> dict[str, object]:
    with old_scores_path.open("r", encoding="utf-8-sig", newline="") as stream:
        old = {
            ((row.get("as_of") or "").strip(), (row.get("stock_code") or "").zfill(6)):
            float(row["economic_moat_score_100"])
            for row in csv.DictReader(stream)
            if row.get("as_of")
            and row.get("stock_code")
            and row.get("economic_moat_score_100")
        }
    by_date: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in shadow_rows:
        if row["score_eligible"]:
            by_date[str(row["date"])].append(row)
    dates = []
    for date, rows in sorted(by_date.items()):
        matched = [
            row
            for row in rows
            if (date, str(row["ticker"]).zfill(6)) in old
        ]
        keys = _rank_keys(matched, candidate)
        rho = spearman(
            keys,
            [old[(date, str(row["ticker"]).zfill(6))] for row in matched],
        )
        dates.append({"date": date, "matched_count": len(matched), "spearman": rho})
    present = [row["spearman"] for row in dates if row["spearman"] is not None]
    return {
        "candidate": candidate,
        "mean_spearman": statistics.mean(present) if present else None,
        "positive_dates": sum(value > 0 for value in present),
        "dates": dates,
    }


def economic_metrics(
    shadow_rows: list[dict[str, object]],
    economic_panel: Path,
    *,
    selected_candidate: str,
) -> dict[str, object]:
    with economic_panel.open("r", encoding="utf-8-sig", newline="") as stream:
        realized = {row["ticker"].zfill(6): row for row in csv.DictReader(stream)}
    matched = [
        (row, realized[str(row["ticker"]).zfill(6)])
        for row in shadow_rows
        if row["score_eligible"]
        and str(row["ticker"]).zfill(6) in realized
        and realized[str(row["ticker"]).zfill(6)].get("forward_return", "") != ""
    ]
    tickers = [str(row["ticker"]).zfill(6) for row, _ in matched]
    returns = [float(realized_row["forward_return"]) for _, realized_row in matched]
    clipped = winsorize(returns)

    def metrics(candidate: str) -> dict[str, object]:
        keys = _rank_keys([row for row, _ in matched], candidate)
        signals = _average_ranks(keys)
        spread, top, bottom = nonoverlapping_quantile_spread(
            signals, clipped, tickers, seed=f"economic-{candidate}"
        )
        return {
            "rank_ic": spearman(signals, clipped),
            "q5_minus_q1_tie_randomized": spread,
            "top_bottom_counts": [top, bottom],
            "ties": signal_tie_diagnostics(signals),
        }

    return {
        "selection_used_forward_returns": False,
        "matched_eligible_count": len(matched),
        "selected_candidate": selected_candidate,
        "selected_candidate_metrics": metrics(selected_candidate),
        "raw_shadow_post_selection_diagnostic": metrics(CANDIDATE_RAW),
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
        description="Compare public-first rank refinements from saved artifacts without API calls."
    )
    parser.add_argument("--root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--old-scores", type=Path)
    parser.add_argument("--economic-panel", type=Path)
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

    repeat = (
        repeat_candidate_metrics(panels[str(roots[0])], panels[str(roots[1])])
        if len(roots) >= 2
        else {"common_eligible_count": 0, "candidates": {}}
    )
    candidate_metrics = repeat["candidates"]
    raw_repeat_passed = bool(
        candidate_metrics.get(CANDIDATE_RAW, {}).get("rank_key_spearman", -math.inf)
        >= 0.90
        and candidate_metrics.get(CANDIDATE_RAW, {}).get(
            "pairwise_order_agreement", -math.inf
        )
        >= 0.90
    )
    stable_repeat_passed = bool(
        candidate_metrics.get(CANDIDATE_STABLE, {}).get(
            "rank_key_spearman", -math.inf
        )
        >= 0.90
        and candidate_metrics.get(CANDIDATE_STABLE, {}).get(
            "pairwise_order_agreement", -math.inf
        )
        >= 0.90
    )
    if raw_repeat_passed:
        selected_candidate = CANDIDATE_RAW
        selection_reason = "raw within-public-bucket refinement passed repeat gates"
    elif stable_repeat_passed:
        selected_candidate = CANDIDATE_STABLE
        selection_reason = (
            "raw refinement failed; stable calibrated components preserve exact ties when they add no distinction"
        )
    else:
        selected_candidate = CANDIDATE_PUBLIC
        selection_reason = "no refinement passed; retain exact public-score ties"

    old_diagnostic = None
    old_passed = False
    if args.old_scores is not None:
        old_diagnostic = old_holistic_metrics(
            panels[str(roots[0])],
            args.old_scores.resolve(),
            candidate=selected_candidate,
        )
        old_passed = bool(
            old_diagnostic["mean_spearman"] is not None
            and old_diagnostic["mean_spearman"] >= 0.50
            and old_diagnostic["positive_dates"] >= 3
        )

    baseline_dates = list(summaries[str(roots[0])]["dates"])
    resolution_nonworse = bool(baseline_dates) and all(
        row["candidates"][selected_candidate]["distinct_rank_count"]
        >= row["candidates"][CANDIDATE_PUBLIC]["distinct_rank_count"]
        and row["candidates"][selected_candidate]["max_single_rank_share"]
        <= row["candidates"][CANDIDATE_PUBLIC]["max_single_rank_share"]
        for row in baseline_dates
    )
    selected_repeat_passed = bool(
        candidate_metrics.get(selected_candidate, {}).get("rank_key_spearman", -math.inf)
        >= 0.90
        and candidate_metrics.get(selected_candidate, {}).get(
            "pairwise_order_agreement", -math.inf
        )
        >= 0.90
    )

    report: dict[str, object] = {
        "schema_version": "moatrader-moat-rank-shadow/2",
        "runner_version": RUNNER_VERSION,
        "rank_reducer_contract_sha256": hashlib.sha256(
            "\n\n".join(
                inspect.getsource(function)
                for function in (
                    repair_contextual_moat_structure,
                    moat_validation.calibrate_contextual_moat_ordinals,
                    derive_raw_ordinal_shadow_score,
                    derive_rank_refinement,
                )
            ).encode("utf-8")
        ).hexdigest(),
        "api_calls": 0,
        "candidate_selection_used_forward_returns": False,
        "roots": summaries,
        "repeat": repeat,
        "candidate_selection": {
            "selected": selected_candidate,
            "reason": selection_reason,
            "raw_repeat_passed": raw_repeat_passed,
            "stable_component_repeat_passed": stable_repeat_passed,
        },
        "old_holistic_diagnostic": old_diagnostic,
        "production_gate": {
            "minimum_rank_key_spearman": 0.90,
            "minimum_pairwise_order_agreement": 0.90,
            "minimum_mean_old_holistic_spearman": 0.50,
            "minimum_positive_old_holistic_dates": 3,
            "selected_rank_key_repeat_passed": selected_repeat_passed,
            "old_holistic_bridge_passed": old_passed,
            "resolution_and_tie_share_nonworse": resolution_nonworse,
            "passed": selected_repeat_passed and old_passed and resolution_nonworse,
        },
    }
    if args.economic_panel is not None:
        report["economic_post_selection_diagnostic"] = economic_metrics(
            panels[str(roots[-1])],
            args.economic_panel.resolve(),
            selected_candidate=selected_candidate,
        )
    (output / "rank-shadow-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
