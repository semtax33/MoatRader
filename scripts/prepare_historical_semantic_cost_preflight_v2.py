from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.prepare_historical_semantic_cost_manifest_v2 import (
    prepare_prelock_cost_preflight,
)


def _parse_role_paths(values: list[str], *, option: str) -> dict[str, Path]:
    stages: dict[str, Path] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role or not raw_path:
            raise ValueError(f"{option} must use ROLE=PATH")
        if role in stages:
            raise ValueError(f"duplicate {option} role: {role}")
        stages[role] = Path(raw_path)
    return stages


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate full semantic-run tokens and cost before dual LOCKED gates; "
            "this artifact never authorizes the full run."
        )
    )
    parser.add_argument("--semantic-packet-input", type=Path, required=True)
    parser.add_argument("--semantic-selection-manifest", type=Path, required=True)
    parser.add_argument(
        "--observed-stage",
        action="append",
        required=True,
        help="Observed exact-prompt stage as ROLE=PATH; provide at least two.",
    )
    parser.add_argument(
        "--passed-gate",
        action="append",
        default=[],
        help=(
            "Optional passed retest evaluation as ROLE=PATH; accepted roles are "
            "NATURAL_RETEST_1 and BALANCED_RETEST_1."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--pricing-checked-date", default="2026-08-22")
    args = parser.parse_args()
    result = prepare_prelock_cost_preflight(
        semantic_packet_input=args.semantic_packet_input,
        semantic_selection_manifest=args.semantic_selection_manifest,
        observed_stage_manifests=_parse_role_paths(
            args.observed_stage, option="--observed-stage"
        ),
        passed_gate_manifests=_parse_role_paths(
            args.passed_gate, option="--passed-gate"
        ),
        output=args.output,
        model=args.model,
        pricing_checked_date=args.pricing_checked_date,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
