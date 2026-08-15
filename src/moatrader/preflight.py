from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from moatrader.runner.models import UniverseRunConfig
from moatrader.evidence.atomic import ATOMIC_RUBRIC_VERSION, ATOMIC_SEGMENTATION_VERSION


PREFLIGHT_SCHEMA_VERSION = "moatrader-moat-preflight/3"
EXECUTION_CONTRACT_FIELDS = (
    "summary_model",
    "moat_model",
    "summary_reasoning_effort",
    "moat_reasoning_effort",
    "context_tokens",
    "prompt_reserve_tokens",
    "max_output_tokens",
    "minimum_text_retention",
    "minimum_numeric_retention",
    "minimum_structured_fact_retention",
    "require_table_count_match",
    "require_financial_table_semantics",
    "allow_low_quality",
    "maximum_price_age_days",
    "maximum_atomic_evidence_units",
    "consolidate_section_summaries",
    "validation_attempts",
    "experiment_id",
)


def ticker_set_sha256(tickers: Iterable[str]) -> str:
    normalized = "\n".join(sorted({str(ticker).strip() for ticker in tickers if str(ticker).strip()})) + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def execution_contract(config: UniverseRunConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    contract = {field: payload.get(field) for field in EXECUTION_CONTRACT_FIELDS}
    contract["llm_replay_enabled"] = bool(config.llm_replay_cache_directory)
    contract["evidence_ledger_enabled"] = bool(config.evidence_ledger_directory)
    contract["atomic_segmentation_version"] = ATOMIC_SEGMENTATION_VERSION
    contract["atomic_rubric_version"] = ATOMIC_RUBRIC_VERSION
    contract["scoring_reducer_version"] = "canonical-claim-reducer/1"
    contract["generated_summary_in_judge_context"] = False
    contract["section_summary_generator"] = "deterministic-python"
    contract["compact_factor_pack_version"] = "compact-factor-pack/1"
    contract["compression_invariance_gate_required"] = True
    contract["metamorphic_gate_required"] = True
    contract["moat_model_snapshot_policy"] = "EXACT_ID_NO_LATEST_ALIAS"
    return contract


def contract_sha256(contract: dict[str, Any]) -> str:
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def find_workspace_manifest(*paths: str | Path) -> Path | None:
    checked: set[Path] = set()
    for value in paths:
        path = Path(value).resolve()
        current = path if path.is_dir() else path.parent
        for parent in (current, *current.parents):
            if parent in checked:
                continue
            checked.add(parent)
            candidate = parent / "workspace-manifest.json"
            if candidate.is_file():
                return candidate
    return None


def validate_preflight_sample_selection(
    selected_tickers: Iterable[str],
    *,
    workspace_manifest: Path | None,
) -> None:
    selected = sorted(set(selected_tickers))
    if not 3 <= len(selected) <= 5:
        raise ValueError("--preflight-sample requires exactly 3 to 5 unique tickers")
    if workspace_manifest is None:
        return
    payload = json.loads(workspace_manifest.read_text(encoding="utf-8-sig"))
    expected = sorted(payload.get("preflight_sample_tickers") or [])
    if expected and selected != expected:
        raise ValueError(
            "preflight sample differs from workspace-manifest.json: "
            f"expected={expected}, actual={selected}"
        )


def validate_preflight_approval(
    path: str | Path,
    *,
    universe_tickers: Iterable[str],
    as_of_date: str,
    config: UniverseRunConfig,
    runner_version: str,
) -> dict[str, Any]:
    approval_path = Path(path).resolve()
    if not approval_path.is_file():
        raise FileNotFoundError(f"preflight report not found: {approval_path}")
    report = json.loads(approval_path.read_text(encoding="utf-8-sig"))
    failures: list[str] = []
    if report.get("schema_version") != PREFLIGHT_SCHEMA_VERSION:
        failures.append("unsupported preflight schema")
    if report.get("passed") is not True:
        failures.append("preflight report did not pass")
    if report.get("runner_version") != runner_version:
        failures.append("runner version differs from preflight")
    expected_ticker_hash = ticker_set_sha256(universe_tickers)
    if report.get("approved_universe_tickers_sha256") != expected_ticker_hash:
        failures.append("universe ticker set differs from preflight")
    if as_of_date not in set(report.get("dates") or []):
        failures.append(f"as-of date {as_of_date} was not preflighted")
    sample = report.get("sample_tickers") or []
    if not 3 <= len(set(sample)) <= 5:
        failures.append("preflight sample size is outside 3..5")
    expected_contract = execution_contract(config)
    if report.get("execution_contract_sha256") != contract_sha256(expected_contract):
        failures.append("execution contract differs from preflight")
    if failures:
        raise ValueError("full-universe preflight approval rejected: " + "; ".join(failures))
    return report
