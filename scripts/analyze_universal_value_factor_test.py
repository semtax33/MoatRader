from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.run_v7_1_value_neutral_sensitivity import (
    CORE_VALUE_COLUMNS,
    extract_value_fundamentals,
    rank_normal_score,
)


SCHEMA_VERSION = "universal-value-factor-readiness/2"
VALUE_COLUMNS = (
    "value_btm",
    "value_earnings_yield",
    "value_fcf_yield",
    "value_sales_yield",
    "value_cfo_yield",
    "value_ebitda_ev_yield",
    "value_ebit_ev_yield",
    "value_operating_income_yield",
    "value_gross_profit_yield",
    "value_rnd_yield",
    "value_retained_earnings_yield",
    "value_assets_yield",
    "value_ncav_yield",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def positive_yield(top: pd.Series, bottom: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(top, errors="coerce")
    denominator = pd.to_numeric(bottom, errors="coerce")
    result = numerator / denominator
    return result.where((numerator > 0) & (denominator > 0) & np.isfinite(result))


def fundamentals(universe: pd.DataFrame, root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in universe["stock_code"].astype(str).str.zfill(6):
        path = root / f"kr_normalized_{ticker}_2025.12.csv"
        values: dict[str, Any] = {}
        if path.is_file():
            values = extract_value_fundamentals(pd.read_csv(path, low_memory=False))
        rows.append({"ticker": ticker, "fund_snapshot": str(path) if path.is_file() else "", **values})
    return pd.DataFrame(rows)


def add_value_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    market_cap = pd.to_numeric(result["market_cap"], errors="coerce")
    result["value_btm"] = positive_yield(result["fund_total_equity"], market_cap)
    result["value_earnings_yield"] = positive_yield(result["fund_net_income"], market_cap)
    result["fund_fcf"] = pd.to_numeric(result["fund_cfo"], errors="coerce") - pd.to_numeric(
        result["fund_capex"], errors="coerce"
    )
    result["value_fcf_yield"] = positive_yield(result["fund_fcf"], market_cap)
    result["value_sales_yield"] = positive_yield(result["fund_revenue"], market_cap)
    result["value_cfo_yield"] = positive_yield(result["fund_cfo"], market_cap)
    debt = pd.to_numeric(result["fund_debt"], errors="coerce").fillna(0.0)
    cash = pd.to_numeric(result["fund_cash"], errors="coerce")
    result["fund_enterprise_value"] = market_cap + debt - cash
    result["fund_ebitda"] = pd.to_numeric(result["fund_ebit"], errors="coerce") + pd.to_numeric(
        result["fund_dna"], errors="coerce"
    )
    result["value_ebitda_ev_yield"] = positive_yield(
        result["fund_ebitda"], result["fund_enterprise_value"]
    )
    result["value_ebit_ev_yield"] = positive_yield(
        result["fund_ebit"], result["fund_enterprise_value"]
    )
    result["value_operating_income_yield"] = positive_yield(result["fund_ebit"], market_cap)
    result["value_gross_profit_yield"] = positive_yield(result["fund_gross_profit"], market_cap)
    result["value_rnd_yield"] = positive_yield(result["fund_rnd"], market_cap)
    result["value_retained_earnings_yield"] = positive_yield(
        result["fund_retained_earnings"], market_cap
    )
    result["value_assets_yield"] = positive_yield(result["fund_total_assets"], market_cap)
    result["fund_ncav"] = pd.to_numeric(result["fund_current_assets"], errors="coerce") - pd.to_numeric(
        result["fund_total_liabilities"], errors="coerce"
    )
    result["value_ncav_yield"] = positive_yield(result["fund_ncav"], market_cap)
    ranks = pd.DataFrame({column: rank_normal_score(result[column]) for column in VALUE_COLUMNS})
    result["broad_value_core"] = ranks[list(CORE_VALUE_COLUMNS)].mean(axis=1).where(
        ranks[list(CORE_VALUE_COLUMNS)].notna().all(axis=1)
    )
    expanded_count = ranks.notna().sum(axis=1)
    result["broad_value_expanded"] = ranks.mean(axis=1).where(expanded_count >= 6)
    result["broad_value_metric_count"] = expanded_count
    return result


def coverage_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = (*VALUE_COLUMNS, "broad_value_core", "broad_value_expanded")
    return [
        {
            "metric": column,
            "valid_count": int(frame[column].notna().sum()),
            "coverage": float(frame[column].notna().mean()),
        }
        for column in columns
    ]


def correlation_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    measures = (*VALUE_COLUMNS, "broad_value_core", "broad_value_expanded")
    for method, group in frame.groupby("method", dropna=False):
        raw = pd.to_numeric(group["raw_value_gap"], errors="coerce")
        for measure in measures:
            value = pd.to_numeric(group[measure], errors="coerce")
            valid = raw.notna() & value.notna()
            n = int(valid.sum())
            rows.append(
                {
                    "method": method,
                    "comparison": measure,
                    "n": n,
                    "spearman": (
                        float(raw[valid].corr(value[valid], method="spearman")) if n >= 10 else None
                    ),
                    "interpretation": "METHOD_SPECIFIC_RAW_GAP_DIAGNOSTIC_NOT_CROSS_METHOD_FACTOR",
                }
            )
    return rows


def _tokens(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    raw = str(value)
    if raw.startswith("["):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            pass
        else:
            return [str(token) for token in decoded if str(token)]
    return [token for token in raw.split(";") if token]


def diagnostic_reason_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for category, column in (
        ("STATUS_REASON", "trust_reason_codes"),
        ("VALUATION_WARNING", "valuation_warning_codes"),
        ("DISCLOSURE_NOT_TRUST_FAILURE", "valuation_disclosures"),
    ):
        if column not in frame:
            continue
        exploded = frame[["alpha_status", "method", column]].copy()
        exploded["reason_code"] = exploded[column].map(_tokens)
        exploded = exploded.explode("reason_code")
        exploded = exploded.loc[exploded["reason_code"].notna() & exploded["reason_code"].ne("")]
        if exploded.empty:
            continue
        grouped = (
            exploded.groupby(["alpha_status", "method", "reason_code"], dropna=False)
            .size()
            .rename("count")
            .reset_index()
        )
        for item in grouped.to_dict("records"):
            rows.append({"category": category, **item})
    return sorted(
        rows,
        key=lambda item: (
            str(item["category"]),
            str(item["alpha_status"]),
            -int(item["count"]),
            str(item["reason_code"]),
        ),
    )


def normalization_class_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    ranked = frame.loc[
        pd.to_numeric(frame["rank_eligible"], errors="coerce").fillna(0).astype(bool)
    ].copy()
    if ranked.empty:
        return []
    rows: list[dict[str, Any]] = []
    for reference_class, group in ranked.groupby("reference_class", dropna=False):
        methods = sorted(group["method"].astype(str).unique())
        levels = sorted(group["normalization_level"].dropna().astype(str).unique())
        rows.append(
            {
                "reference_class": str(reference_class),
                "normalization_level": ";".join(levels),
                "reference_distribution_size": int(
                    pd.to_numeric(group["reference_class_size"], errors="coerce").max()
                ),
                "rank_eligible_count": len(group),
                "method_count": len(methods),
                "methods": ";".join(methods),
                "fallback_count": int(
                    pd.to_numeric(
                        group.get("normalization_fallback_used", 0), errors="coerce"
                    ).fillna(0).sum()
                ),
            }
        )
    return sorted(rows, key=lambda item: (-int(item["rank_eligible_count"]), item["reference_class"]))


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["| Metric | N | Coverage |", "|---|---:|---:|"]
    for row in rows:
        lines.append(f"| {row['metric']} | {row['valid_count']} | {row['coverage']:.1%} |")
    return lines


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output must be new and empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    universe = pd.read_csv(args.universe, dtype={"stock_code": str})
    signals = pd.read_csv(args.signals, dtype={"ticker": str})
    routing = pd.read_csv(args.routing, dtype={"ticker": str})
    if len(universe) != 150 or len(signals) != 150 or len(routing) != 150:
        raise ValueError(
            f"150-row invariant failed: universe={len(universe)} signals={len(signals)} routing={len(routing)}"
        )
    universe["stock_code"] = universe["stock_code"].str.zfill(6)
    signals["ticker"] = signals["ticker"].str.zfill(6)
    routing["ticker"] = routing["ticker"].str.zfill(6)
    if set(universe["stock_code"]) != set(signals["ticker"]) or set(universe["stock_code"]) != set(routing["ticker"]):
        raise ValueError("universe, signals, and routing ticker sets differ")
    frame = universe.merge(
        fundamentals(universe, args.annual_snapshot_root),
        left_on="stock_code",
        right_on="ticker",
        validate="one_to_one",
    ).merge(signals, on="ticker", validate="one_to_one", suffixes=("", "_signal"))
    frame = add_value_metrics(frame)
    coverage = coverage_rows(frame)
    correlations = correlation_rows(frame)
    reasons = diagnostic_reason_rows(frame)
    normalization_classes = normalization_class_rows(frame)
    broad_correlation_summary = [
        row
        for row in correlations
        if row["comparison"] in {"broad_value_core", "broad_value_expanded"}
        and row["n"] >= 10
    ]
    architecture = read_json(args.architecture_result)
    audit = architecture["coverage"]
    status_counts = {str(key): int(value) for key, value in frame["alpha_status"].value_counts().items()}
    pre_status_counts = {
        str(key): int(value)
        for key, value in frame.get("pre_normalization_status", frame["alpha_status"]).value_counts().items()
    }
    route_counts = {str(key): int(value) for key, value in frame["method"].value_counts().items()}
    engine_counts = {str(key): int(value) for key, value in frame["actual_engine"].dropna().value_counts().items()}
    ranked_count = int(pd.to_numeric(frame["rank_eligible"], errors="coerce").fillna(0).astype(bool).sum())
    trust_gate_pass_count = int(
        pd.to_numeric(frame.get("trust_gate_pass", 0), errors="coerce").fillna(0).astype(bool).sum()
    )
    generated_count = int(frame["actual_engine"].notna().sum())
    broad_overlap = int(
        (pd.to_numeric(frame["unified_value_score"], errors="coerce").notna() & frame["broad_value_expanded"].notna()).sum()
    )
    unified_score = pd.to_numeric(frame["unified_value_score"], errors="coerce")
    universal_broad_correlations: list[dict[str, Any]] = []
    for measure in ("broad_value_core", "broad_value_expanded"):
        broad = pd.to_numeric(frame[measure], errors="coerce")
        valid = unified_score.notna() & broad.notna()
        universal_broad_correlations.append(
            {
                "comparison": measure,
                "n": int(valid.sum()),
                "spearman": (
                    float(unified_score[valid].corr(broad[valid], method="spearman"))
                    if int(valid.sum()) >= 10
                    else None
                ),
                "interpretation": "CROSS_METHOD_NORMALIZED_SCORE_DIAGNOSTIC_NOT_ALPHA_TEST",
            }
        )
    gate_failures: list[str] = []
    if generated_count / 150 < 0.70:
        gate_failures.append("VALUATION_GENERATION_COVERAGE_LT_70_PERCENT")
    if ranked_count / 150 < 0.60:
        gate_failures.append("RANK_ELIGIBLE_COVERAGE_LT_60_PERCENT")
    if len(normalization_classes) < 3:
        gate_failures.append("FEWER_THAN_3_SCORE_BEARING_REFERENCE_CLASSES")
    if broad_overlap < 30:
        gate_failures.append("BROAD_VALUE_DISTINCTNESS_OVERLAP_LT_30")
    gate_failures.append("NO_FORWARD_RETURN_FOR_2026_08_19_SIGNAL")
    verdict = "NOT_READY_AS_UNIVERSAL_VALUE_FACTOR"
    result = {
        "schema_version": SCHEMA_VERSION,
        "signal_date": "2026-08-19",
        "architecture_verdict": architecture["verdict"],
        "factor_verdict": verdict,
        "universe_count": 150,
        "valuation_generated_count": generated_count,
        "valuation_generated_coverage": generated_count / 150,
        "trust_gate_pass_count": trust_gate_pass_count,
        "trust_gate_pass_coverage": trust_gate_pass_count / 150,
        "rank_eligible_count": ranked_count,
        "rank_eligible_coverage": ranked_count / 150,
        "broad_value_overlap_count": broad_overlap,
        "route_counts": route_counts,
        "actual_engine_counts": engine_counts,
        "alpha_status_counts": status_counts,
        "pre_normalization_status_counts": pre_status_counts,
        "failure_reason_breakdown": [
            row for row in reasons if row["category"] != "DISCLOSURE_NOT_TRUST_FAILURE"
        ],
        "normalization_classes": normalization_classes,
        "broad_value_coverage": coverage,
        "method_broad_value_correlations": broad_correlation_summary,
        "universal_score_broad_value_correlations": universal_broad_correlations,
        "factor_gate_failures": gate_failures,
        "llm_call_count": int(audit["llm_call_count"]),
        "fallback_fcff_count": int(audit["fallback_fcff_count"]),
        "return_data_accessed_for_2026_08_19": False,
        "historical_broad_value_evidence_scope": "LEGACY_HISTORICAL_FCFF_CHEAP_ONLY_NOT_CURRENT_UNIVERSAL_VALUE",
        "broad_value_role": "COMPARISON_BASELINE_ONLY_NOT_PRIMARY_RANK",
        "primary_ranking_policy_changed": False,
    }
    frame.to_csv(args.output / "RESULTS-150.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(coverage).to_csv(args.output / "BROAD-VALUE-COVERAGE.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(correlations).to_csv(
        args.output / "METHOD-BROAD-VALUE-CORRELATIONS.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(reasons).to_csv(
        args.output / "FAILURE-REASON-BREAKDOWN.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(normalization_classes).to_csv(
        args.output / "NORMALIZATION-CLASS-COVERAGE.csv", index=False, encoding="utf-8-sig"
    )
    (args.output / "FACTOR-READINESS.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Universal Value 150-stock factor-readiness test",
        "",
        f"- Architecture: **{architecture['verdict']}**",
        f"- Universal Value Factor: **{verdict}**",
        "- Signal date: `2026-08-19`",
        f"- Actual valuations: `{generated_count}/150` ({generated_count / 150:.1%})",
        f"- Trust-gate pass before normalization: `{trust_gate_pass_count}/150` ({trust_gate_pass_count / 150:.1%})",
        f"- Rank-eligible Universal Value scores: `{ranked_count}/150` ({ranked_count / 150:.1%})",
        f"- Broad Value distinctness overlap: `{broad_overlap}`",
        f"- LLM/fallback: `{audit['llm_call_count']}` / `{audit['fallback_fcff_count']}`",
        "",
        "## 판정",
        "",
        "Trust threshold를 낮추지 않고 disclosure와 valuation-specific warning을 분리했습니다. 정규화는 method+archetype → method → model family의 return-blind hierarchy를 사용하며 model family 밖으로 fallback하지 않습니다. 아직 coverage와 reference-class 다양성 gate를 통과하지 못했으므로 Universal Value Factor로 사용할 수 없습니다. 8월 19일 forward return도 아직 없어 alpha 가능성은 검정되지 않았습니다.",
        "",
        "PER+PBR은 비교 baseline일 뿐 우선 랭킹으로 전환하지 않았고 기존 primary ranking policy도 변경하지 않았습니다.",
        "",
        "Gate failures:",
        "",
        *[f"- `{item}`" for item in gate_failures],
        "",
        "## Failure reason breakdown",
        "",
        "| Status | Method | Layer | Reason | N |",
        "|---|---|---|---|---:|",
        *[
            f"| {row['alpha_status']} | {row['method']} | {row['category']} | {row['reason_code']} | {row['count']} |"
            for row in reasons
            if row["category"] != "DISCLOSURE_NOT_TRUST_FAILURE"
        ],
        "",
        "## Score-bearing normalization classes",
        "",
        "| Reference class | Level | Distribution N | Ranked | Methods | Fallback rows |",
        "|---|---|---:|---:|---|---:|",
        *[
            f"| {row['reference_class']} | {row['normalization_level']} | {row['reference_distribution_size']} | {row['rank_eligible_count']} | {row['methods']} | {row['fallback_count']} |"
            for row in normalization_classes
        ],
        "",
        "## Normalized Universal score vs Broad Value",
        "",
        "| Broad measure | N | Spearman |",
        "|---|---:|---:|",
        *[
            f"| {row['comparison']} | {row['n']} | {row['spearman']:.3f} |"
            for row in universal_broad_correlations
            if row["spearman"] is not None
        ],
        "",
        "## Broad Value coverage",
        "",
        *markdown_table(coverage),
        "",
        "## Method-specific raw gap vs Broad Value",
        "",
        "서로 다른 valuation method의 raw gap은 직접 합치지 않습니다. 아래 값은 각 method 내부 진단이며 factor correlation이 아닙니다.",
        "",
        "| Method | Broad measure | N | Spearman |",
        "|---|---|---:|---:|",
        *[
            f"| {row['method']} | {row['comparison']} | {row['n']} | {row['spearman']:.3f} |"
            for row in broad_correlation_summary
        ],
        "",
        "## Historical context",
        "",
        "기존 2020-03~2025-09 사후 분석에서 legacy FCFF Cheap은 PER, PBR, P/FCF, PSR, PCR, EV/EBITDA, EV/EBIT, operating-income/P, gross-profit/P, R&D/P, retained-earnings/P, assets/P, NCAV/P 및 core composite 통제 후 유의한 잔존 IC가 없었습니다. 이는 현재 multi-model Universal Value의 성과검정이 아니라 과거 one-size-fits-all FCFF가 Broad Value 노출이었다는 근거입니다.",
        "",
        "`RESULTS-150.csv`에는 150종목 route, actual engine, valuation 상태와 모든 Broad Value 지표가 들어 있습니다. `FAILURE-REASON-BREAKDOWN.csv`와 `NORMALIZATION-CLASS-COVERAGE.csv`가 실패 원인과 fallback 경로를 고정합니다.",
    ]
    (args.output / "FINAL-REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--universe", type=Path, required=True)
    parser.add_argument("--signals", type=Path, required=True)
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--architecture-result", type=Path, required=True)
    parser.add_argument("--annual-snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in (
        "universe",
        "signals",
        "routing",
        "architecture_result",
        "annual_snapshot_root",
        "output",
    ):
        setattr(args, name, getattr(args, name).resolve())
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
