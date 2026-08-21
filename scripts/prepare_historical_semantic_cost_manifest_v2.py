from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Sequence

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import PairedAxisPacket, sha256_file
from scripts.classify_historical_future_eri_evidence import (
    GROUNDING_VALIDATION_ATTEMPTS,
    ParserProfile,
    SemanticExecutionScope,
    parser_spec,
)


D = Decimal
SEMANTIC_AXES = {OperatingEvidenceAxis.DEMAND, OperatingEvidenceAxis.PRICE_MIX}
TOKEN_KEYS = (
    "input_tokens",
    "cached_input_tokens",
    "cache_write_tokens",
    "output_tokens",
)
OFFICIAL_PRICING_URL = "https://developers.openai.com/api/docs/models/gpt-5.6-luna"
DEFAULT_USD_PER_MILLION = {
    "uncached_input": D("0.20"),
    "cached_input": D("0.02"),
    "cache_write": D("0.25"),
    "output": D("1.20"),
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _money(value: Decimal) -> str:
    return str(value.quantize(D("0.000001"), rounding=ROUND_HALF_UP))


def _expected_token_counts(
    usage: dict[str, int],
    *,
    pilot_packets: int,
    full_packets: int,
) -> dict[str, int]:
    if pilot_packets < 1 or full_packets < 1:
        raise ValueError("pilot and full packet counts must be positive")
    return {
        key: int(
            (D(usage[key]) * D(full_packets) / D(pilot_packets)).to_integral_value(
                rounding=ROUND_HALF_UP
            )
        )
        for key in TOKEN_KEYS
    }


def _cost_components(
    tokens: dict[str, int],
    rates: dict[str, Decimal],
) -> dict[str, str]:
    uncached = (
        tokens["input_tokens"]
        - tokens["cached_input_tokens"]
        - tokens["cache_write_tokens"]
    )
    if uncached < 0:
        raise ValueError("cached and cache-write tokens exceed total input tokens")
    parts = {
        "uncached_input": D(uncached) * rates["uncached_input"] / D(1_000_000),
        "cached_input": D(tokens["cached_input_tokens"])
        * rates["cached_input"]
        / D(1_000_000),
        "cache_write": D(tokens["cache_write_tokens"])
        * rates["cache_write"]
        / D(1_000_000),
        "output": D(tokens["output_tokens"]) * rates["output"] / D(1_000_000),
    }
    parts["total"] = sum(parts.values(), D(0))
    return {key: _money(value) for key, value in parts.items()}


def prepare_cost_manifest(
    *,
    semantic_packet_input: Path,
    semantic_selection_manifest: Path,
    pilot_stage_manifests: Sequence[Path],
    output: Path,
    model: str = "gpt-5.6-luna",
    pricing_checked_date: str = "2026-08-21",
    prepared_at: datetime | None = None,
    rates: dict[str, Decimal] | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"output must be new: {output}")
    if model != "gpt-5.6-luna":
        raise ValueError("default price contract is valid only for gpt-5.6-luna")
    selection = _read_json(semantic_selection_manifest)
    packet_sha = sha256_file(semantic_packet_input)
    if selection.get("output_packet_sha256") != packet_sha:
        raise ValueError("semantic selection manifest does not seal packet input")
    if selection.get("outcome_vault_opened") is not False:
        raise ValueError("semantic selection must remain outcome blind")
    if selection.get("return_data_opened") is not False:
        raise ValueError("semantic selection must keep return data closed")

    packet_count = 0
    axis_counts: Counter[str] = Counter()
    with semantic_packet_input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            packet = PairedAxisPacket.model_validate_json(line)
            if packet.axis not in SEMANTIC_AXES:
                raise ValueError("full semantic run may contain only Demand and PriceMix")
            packet_count += 1
            axis_counts[packet.axis.value] += 1
    if packet_count != int(selection.get("selected_packet_count", -1)):
        raise ValueError("semantic packet count does not match selection manifest")
    if packet_count < 1:
        raise ValueError("semantic packet input is empty")

    combined_usage = {key: 0 for key in TOKEN_KEYS}
    pilot_packets = 0
    pilot_sources: list[dict[str, Any]] = []
    spec = parser_spec(ParserProfile.DEMAND_PRICE_MIX_V2)
    for path in pilot_stage_manifests:
        stage = _read_json(path)
        if stage.get("status") != "CLASSIFICATION_COMPLETE_AWAITING_HUMAN_GOLD_GATE":
            raise ValueError(f"pilot stage is not complete: {path}")
        for key, expected in (
            ("parser_profile", spec.profile.value),
            ("parser_version", spec.parser_version),
            ("prompt_sha256", spec.prompt_sha256),
            ("requested_model", model),
            (
                "semantic_execution_scope",
                SemanticExecutionScope.PILOT_OR_LOCKED_VALIDATION.value,
            ),
            ("full_historical_execution_authorized", False),
        ):
            if stage.get(key) != expected:
                raise ValueError(
                    f"pilot stage does not match frozen semantic V2 {key}: {path}"
                )
        if any(
            stage.get(key) is not False
            for key in (
                "outcome_vault_opened",
                "return_data_opened",
                "value_data_opened",
                "credentials_persisted",
            )
        ):
            raise ValueError(f"pilot stage violated closed-data contract: {path}")
        if stage.get("per_pbr_role", "NOT_USED") != "NOT_USED":
            raise ValueError(f"pilot stage used PER/PBR before Full Index seal: {path}")
        count = int(stage.get("classification_count", -1))
        if count < 1 or count != int(stage.get("packet_count", -2)):
            raise ValueError(f"pilot classification count mismatch: {path}")
        usage = stage.get("usage", {})
        for key in TOKEN_KEYS:
            value = int(usage.get(key, -1))
            if value < 0:
                raise ValueError(f"pilot usage missing {key}: {path}")
            combined_usage[key] += value
        pilot_packets += count
        pilot_sources.append(
            {
                "path": str(path),
                "sha256": sha256_file(path),
                "packet_count": count,
                "parser_profile": stage.get("parser_profile"),
                "parser_version": stage.get("parser_version"),
                "prompt_sha256": stage.get("prompt_sha256"),
                "requested_model": stage.get("requested_model"),
                "semantic_execution_scope": stage.get("semantic_execution_scope"),
            }
        )
    if len(pilot_sources) < 2:
        raise ValueError("cost estimate requires both natural and balanced pilot stages")

    expected = _expected_token_counts(
        combined_usage,
        pilot_packets=pilot_packets,
        full_packets=packet_count,
    )
    active_rates = rates or DEFAULT_USD_PER_MILLION
    base_cost = _cost_components(expected, active_rates)
    conservative_tokens = {
        key: int((D(value) * D("1.20")).to_integral_value(rounding=ROUND_HALF_UP))
        for key, value in expected.items()
    }
    conservative_cost = _cost_components(conservative_tokens, active_rates)
    created = prepared_at or datetime.now(timezone.utc)
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("prepared_at must be timezone-aware")
    manifest: dict[str, Any] = {
        "schema_version": "moatrader-historical-semantic-cost-manifest-v2/1",
        "status": "FULL_SEMANTIC_RUN_COST_PRESPECIFIED_NO_EXTERNAL_CALL",
        "prepared_at": created.isoformat(),
        "exact_expected_api_calls_without_retries": packet_count,
        "maximum_api_calls_with_grounding_retries": (
            packet_count * GROUNDING_VALIDATION_ATTEMPTS
        ),
        "exact_packet_count": packet_count,
        "axis_counts": dict(sorted(axis_counts.items())),
        "model": model,
        "reasoning_effort": "medium",
        "max_output_tokens_per_call": 2_000,
        "parser_profile": spec.profile.value,
        "parser_version": spec.parser_version,
        "prompt_sha256": spec.prompt_sha256,
        "prompt_cache_key": spec.prompt_cache_key,
        "token_estimation": {
            "method": "ACTUAL_DUAL_LOCKED_PILOT_MEAN_SCALED_BY_EXACT_PACKET_COUNT",
            "pilot_packet_count": pilot_packets,
            "pilot_usage": combined_usage,
            "expected_tokens": expected,
            "conservative_20pct_tokens": conservative_tokens,
            "pilot_prompt_differs_from_frozen_full_prompt": False,
            "pilot_contract_matches_frozen_full_prompt": True,
            "estimate_not_research_result": True,
        },
        "pricing": {
            "currency": "USD",
            "unit": "PER_1M_TOKENS",
            "official_pricing_url": OFFICIAL_PRICING_URL,
            "pricing_checked_date": pricing_checked_date,
            "rates": {key: str(value) for key, value in active_rates.items()},
            "expected_cost": base_cost,
            "conservative_20pct_cost": conservative_cost,
        },
        "inputs": {
            "semantic_packet_input": str(semantic_packet_input),
            "semantic_packet_sha256": packet_sha,
            "semantic_selection_manifest": str(semantic_selection_manifest),
            "semantic_selection_manifest_sha256": sha256_file(
                semantic_selection_manifest
            ),
            "pilot_stage_manifests": pilot_sources,
        },
        "api_calls_executed": False,
        "credentials_persisted": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze expected calls, tokens, and cost before a full semantic LLM run."
    )
    parser.add_argument("--semantic-packet-input", type=Path, required=True)
    parser.add_argument("--semantic-selection-manifest", type=Path, required=True)
    parser.add_argument(
        "--pilot-stage-manifest",
        type=Path,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--pricing-checked-date", default="2026-08-21")
    args = parser.parse_args()
    result = prepare_cost_manifest(
        semantic_packet_input=args.semantic_packet_input,
        semantic_selection_manifest=args.semantic_selection_manifest,
        pilot_stage_manifests=args.pilot_stage_manifest,
        output=args.output,
        model=args.model,
        pricing_checked_date=args.pricing_checked_date,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
