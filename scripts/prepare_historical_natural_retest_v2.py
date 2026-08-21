from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from moatrader.expectations.historical_evidence import PairedAxisPacket, sha256_file
from scripts.classify_historical_future_eri_evidence import (
    ParserProfile,
    parser_spec,
)
from scripts.prepare_historical_locked_sets_v2 import (
    SEMANTIC_AXES,
    _blank_gold_rows,
    _iter_packets,
    _packet_ids,
    _selection_key,
    _write_gold,
    _write_json,
    _write_jsonl,
)


RETEST_SPLIT = "V2_NATURAL_LOCKED_RETEST_1"
RETEST_CONTRACT = "V2_NATURAL_FREQUENCY_LOCKED_RETEST_1"
RETEST_SEED = "MOATRADER_V2_NATURAL_RETEST_1_20260822"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must contain a JSON object: {path}")
    return payload


def _require_downstream_closed(payload: dict[str, Any], description: str) -> None:
    for key in ("outcome_vault_opened", "return_data_opened", "value_data_opened"):
        if payload.get(key, False):
            raise ValueError(f"{description} opened forbidden downstream data: {key}")
    if payload.get("per_pbr_role", "NOT_USED") != "NOT_USED":
        raise ValueError(f"{description} used PER/PBR before the Full Index seal")


def _require_frozen_parser_contract(payload: dict[str, Any]) -> None:
    spec = parser_spec(ParserProfile.DEMAND_PRICE_MIX_V2)
    expected = {
        "parser_profile": spec.profile.value,
        "parser_version": spec.parser_version,
        "prompt_sha256": spec.prompt_sha256,
        "requested_model": "gpt-5.6-luna",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"parser freeze does not match semantic V2 {key}")


def prepare_natural_retest_candidates(
    *,
    packet_input: Path,
    prior_v1_inputs: Sequence[Path],
    dev_inputs: Sequence[Path],
    prior_v2_locked_inputs: Sequence[Path],
    failed_natural_evaluation_manifest: Path,
    failed_natural_consumption_record: Path,
    parser_freeze_manifest: Path,
    output: Path,
    per_axis: int = 40,
    seed: str = RETEST_SEED,
) -> dict[str, Any]:
    """Prepare an outcome-blind Natural retest without reusing post-test disagreements.

    The first Natural LOCKED test remains consumed. This function only creates a new,
    disjoint packet set. It does not read classifications, HUMAN re-review decisions,
    outcomes, returns, or value data.
    """

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if per_axis < 20:
        raise ValueError("Natural retest requires at least 20 packets per semantic axis")
    if not prior_v1_inputs:
        raise ValueError("explicit prior V1 inputs are required")
    if not dev_inputs:
        raise ValueError("explicit semantic DEV inputs are required")
    if len(prior_v2_locked_inputs) < 2:
        raise ValueError("both prior V2 Natural and Balanced LOCKED inputs are required")
    if not packet_input.is_file():
        raise FileNotFoundError(packet_input)

    freeze = _read_json(parser_freeze_manifest)
    if freeze.get("schema_version") != "moatrader-historical-evidence-parser-freeze-v2/2":
        raise ValueError("Natural retest requires the original semantic V2 parser freeze")
    if freeze.get("status") != "V2_PARSER_FROZEN_AWAITING_DUAL_INDEPENDENT_LOCKED_TESTS":
        raise ValueError("semantic V2 parser is not frozen")
    _require_frozen_parser_contract(freeze)
    _require_downstream_closed(freeze, "parser freeze")

    evaluation = _read_json(failed_natural_evaluation_manifest)
    if evaluation.get("status") != "V2_NATURAL_EVIDENCE_PARSER_NOT_VALIDATED":
        raise ValueError("Natural retest requires a failed first Natural LOCKED evaluation")
    if evaluation.get("locked_kind") != "NATURAL" or evaluation.get("gate_passed") is not False:
        raise ValueError("supplied evaluation is not the failed Natural LOCKED gate")
    if evaluation.get("parser_freeze_sha256") != sha256_file(parser_freeze_manifest):
        raise ValueError("failed Natural evaluation does not reference the supplied freeze")
    _require_downstream_closed(evaluation, "failed Natural evaluation")

    consumption = _read_json(failed_natural_consumption_record)
    if consumption.get("status") != "COMPLETED_SINGLE_USE":
        raise ValueError("first Natural LOCKED test has not been consumed")
    if consumption.get("locked_kind") != "NATURAL" or consumption.get("gate_passed") is not False:
        raise ValueError("Natural retest requires a consumed failing Natural test")
    if consumption.get("parser_freeze_sha256") != sha256_file(parser_freeze_manifest):
        raise ValueError("Natural consumption record does not reference the supplied freeze")
    if consumption.get("locked_packet_sha256") != freeze.get(
        "natural_locked_packet_sha256"
    ):
        raise ValueError("Natural consumption packet hash does not match the parser freeze")
    _require_downstream_closed(consumption, "Natural consumption record")

    prior_v2_hashes = {sha256_file(path) for path in prior_v2_locked_inputs}
    required_v2_hashes = {
        str(freeze.get("natural_locked_packet_sha256") or ""),
        str(freeze.get("balanced_locked_packet_sha256") or ""),
    }
    if "" in required_v2_hashes or not required_v2_hashes.issubset(prior_v2_hashes):
        raise ValueError(
            "prior V2 inputs must explicitly include the frozen Natural and Balanced sets"
        )

    prior_v1_ids = set().union(*(_packet_ids(path) for path in prior_v1_inputs))
    dev_ids = set().union(*(_packet_ids(path) for path in dev_inputs))
    prior_v2_ids = set().union(*(_packet_ids(path) for path in prior_v2_locked_inputs))
    excluded = prior_v1_ids | dev_ids | prior_v2_ids

    heaps: dict[Any, list[tuple[int, str, PairedAxisPacket]]] = {
        axis: [] for axis in SEMANTIC_AXES
    }
    total_packet_count = 0
    nonsemantic_packet_count = 0
    excluded_packet_count = 0
    eligible_axis_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for packet in _iter_packets(packet_input):
        total_packet_count += 1
        if packet.packet_id in seen_ids:
            raise ValueError(f"packet IDs must be unique: {packet.packet_id}")
        seen_ids.add(packet.packet_id)
        if packet.axis not in SEMANTIC_AXES:
            nonsemantic_packet_count += 1
            continue
        if packet.packet_id in excluded:
            excluded_packet_count += 1
            continue
        eligible_axis_counts[packet.axis.value] += 1
        key = int(_selection_key(packet, seed), 16)
        item = (-key, packet.packet_id, packet)
        heap = heaps[packet.axis]
        if len(heap) < per_axis:
            heapq.heappush(heap, item)
        elif key < -heap[0][0]:
            heapq.heapreplace(heap, item)

    selected: list[PairedAxisPacket] = []
    axis_counts: dict[str, int] = {}
    for axis in SEMANTIC_AXES:
        rows = sorted(
            (item[2] for item in heaps[axis]),
            key=lambda packet: _selection_key(packet, seed),
        )
        if len(rows) != per_axis:
            raise ValueError(f"not enough independent {axis.value} packets for retest")
        selected.extend(rows)
        axis_counts[axis.value] = len(rows)
    selected.sort(key=lambda packet: (packet.axis.value, packet.packet_id))
    selected_ids = {packet.packet_id for packet in selected}
    if len(selected_ids) != len(selected) or selected_ids & excluded:
        raise AssertionError("Natural retest independence invariant failed")

    output.mkdir(parents=True, exist_ok=True)
    packet_output = output / "natural-retest-packets.jsonl"
    template_output = output / "natural-retest-human-gold-template.csv"
    _write_jsonl(packet_output, selected)
    _write_gold(
        template_output,
        _blank_gold_rows(selected, split=RETEST_SPLIT, contract=RETEST_CONTRACT),
    )
    manifest = {
        "schema_version": "moatrader-v2-natural-retest-candidate-preparation/1",
        "status": "V2_NATURAL_RETEST_1_PREPARED_OUTCOME_BLIND",
        "retest_number": 1,
        "selection_seed": seed,
        "selection_memory_policy": "STREAMING_SMALLEST_HASH_KEYS_PER_AXIS_NO_DIRECTION_CUES",
        "selection_used_parser_classifications": False,
        "selection_used_post_test_disagreement_rows": False,
        "first_natural_test_remains_consumed": True,
        "first_natural_result_superseded": False,
        "new_independent_human_review_required": True,
        "semantic_parser_axes": [axis.value for axis in SEMANTIC_AXES],
        "gold_split": RETEST_SPLIT,
        "gold_contract_version": RETEST_CONTRACT,
        "source_packet_count": total_packet_count,
        "nonsemantic_packets_ignored": nonsemantic_packet_count,
        "excluded_packet_count": excluded_packet_count,
        "eligible_semantic_axis_counts": dict(sorted(eligible_axis_counts.items())),
        "per_axis": per_axis,
        "axis_counts": axis_counts,
        "packet_count": len(selected),
        "source_packet_sha256": sha256_file(packet_input),
        "prior_v1_input_sha256": [sha256_file(path) for path in prior_v1_inputs],
        "dev_input_sha256": [sha256_file(path) for path in dev_inputs],
        "prior_v2_locked_input_sha256": [
            sha256_file(path) for path in prior_v2_locked_inputs
        ],
        "failed_natural_evaluation_manifest_sha256": sha256_file(
            failed_natural_evaluation_manifest
        ),
        "failed_natural_consumption_record_sha256": sha256_file(
            failed_natural_consumption_record
        ),
        "parser_freeze_sha256": sha256_file(parser_freeze_manifest),
        "natural_retest_packet_sha256": sha256_file(packet_output),
        "natural_retest_human_gold_template_sha256": sha256_file(template_output),
        "prior_v1_packet_id_count": len(prior_v1_ids),
        "dev_packet_id_count": len(dev_ids),
        "prior_v2_locked_packet_id_count": len(prior_v2_ids),
        "all_exclusion_overlaps": 0,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "natural-retest-preparation-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an independent, outcome-blind Natural LOCKED retest after a "
            "consumed failing first test."
        )
    )
    parser.add_argument("--packet-input", type=Path, required=True)
    parser.add_argument("--prior-v1-input", type=Path, action="append", required=True)
    parser.add_argument("--dev-input", type=Path, action="append", required=True)
    parser.add_argument(
        "--prior-v2-locked-input", type=Path, action="append", required=True
    )
    parser.add_argument(
        "--failed-natural-evaluation-manifest", type=Path, required=True
    )
    parser.add_argument(
        "--failed-natural-consumption-record", type=Path, required=True
    )
    parser.add_argument("--parser-freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-axis", type=int, default=40)
    parser.add_argument("--seed", default=RETEST_SEED)
    args = parser.parse_args()
    result = prepare_natural_retest_candidates(
        packet_input=args.packet_input,
        prior_v1_inputs=args.prior_v1_input,
        dev_inputs=args.dev_input,
        prior_v2_locked_inputs=args.prior_v2_locked_input,
        failed_natural_evaluation_manifest=args.failed_natural_evaluation_manifest,
        failed_natural_consumption_record=args.failed_natural_consumption_record,
        parser_freeze_manifest=args.parser_freeze_manifest,
        output=args.output,
        per_axis=args.per_axis,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
