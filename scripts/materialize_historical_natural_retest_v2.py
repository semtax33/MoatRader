from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    PairedAxisPacket,
    sha256_file,
)
from scripts.classify_historical_future_eri_evidence import (
    ParserProfile,
    parser_spec,
)
from scripts.materialize_historical_human_gold_v2 import _source_for_anchor
from scripts.prepare_historical_locked_sets_v2 import GOLD_FIELDS
from scripts.prepare_historical_natural_retest_v2 import (
    RETEST_CONTRACT,
    RETEST_SPLIT,
)


SEOUL = ZoneInfo("Asia/Seoul")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)
        + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _read_packets(path: Path) -> list[PairedAxisPacket]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [
            PairedAxisPacket.model_validate_json(line)
            for line in handle
            if line.strip()
        ]
    if len({row.packet_id for row in rows}) != len(rows):
        raise ValueError("Natural retest packet IDs must be unique")
    return rows


def _require_downstream_closed(payload: dict[str, Any], description: str) -> None:
    for key in ("outcome_vault_opened", "return_data_opened", "value_data_opened"):
        if payload.get(key, False):
            raise ValueError(f"{description} opened forbidden downstream data: {key}")
    if payload.get("per_pbr_role", "NOT_USED") != "NOT_USED":
        raise ValueError(f"{description} used PER/PBR before the Full Index seal")


def _reject_model_fields(payload: dict[str, Any], description: str) -> None:
    forbidden_prefixes = (
        "machine_",
        "model_",
        "predicted_",
        "parser_",
        "classification_",
        "selection_hint",
    )
    present = [
        str(key)
        for key, value in payload.items()
        if value not in (None, "", [], {})
        and str(key).casefold().startswith(forbidden_prefixes)
    ]
    if present:
        raise ValueError(f"{description} contains forbidden model-derived fields: {present}")


def materialize_natural_retest_human_gold(
    *, candidate_build: Path, review_decisions: Path, output: Path
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    manifest_path = candidate_build / "natural-retest-preparation-manifest.json"
    packet_path = candidate_build / "natural-retest-packets.jsonl"
    template_path = candidate_build / "natural-retest-human-gold-template.csv"
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "V2_NATURAL_RETEST_1_PREPARED_OUTCOME_BLIND":
        raise ValueError("Natural retest candidates are not outcome-blind and ready")
    _require_downstream_closed(manifest, "Natural retest candidate manifest")
    if manifest.get("selection_used_parser_classifications") is not False or manifest.get(
        "selection_used_post_test_disagreement_rows"
    ) is not False:
        raise ValueError("Natural retest selection was contaminated by prior test results")
    if manifest.get("first_natural_test_remains_consumed") is not True:
        raise ValueError("first Natural LOCKED test must remain consumed")
    if manifest.get("natural_retest_packet_sha256") != sha256_file(packet_path):
        raise ValueError("Natural retest packet input changed after preparation")
    if manifest.get("natural_retest_human_gold_template_sha256") != sha256_file(
        template_path
    ):
        raise ValueError("Natural retest HUMAN template changed after preparation")

    packets = _read_packets(packet_path)
    packet_lookup = {row.packet_id: row for row in packets}
    payload = _read_json(review_decisions)
    _reject_model_fields(payload, "review decision payload")
    if payload.get("reviewer") != "HUMAN":
        raise ValueError("Natural retest review decisions must be tagged exactly HUMAN")
    reviewer_name = str(payload.get("human_reviewer_name") or "").strip()
    if not reviewer_name:
        raise ValueError("Natural retest requires the actual HUMAN reviewer name")
    if payload.get("attestation") != "YES":
        raise ValueError("Natural retest HUMAN attestation must be exactly YES")
    review_date = str(payload.get("review_date") or "").strip()
    if not review_date:
        raise ValueError("Natural retest requires a review date")
    _require_downstream_closed(payload, "Natural retest review decisions")
    decisions = payload.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("Natural retest review decisions must contain a decisions list")

    rows: list[dict[str, str]] = []
    status_counts: dict[str, int] = {}
    seen: set[str] = set()
    for number, raw in enumerate(decisions, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"review decision item {number} is not an object")
        _reject_model_fields(raw, f"review decision item {number}")
        packet_id = str(raw.get("packet_id") or "").strip()
        if packet_id in seen:
            raise ValueError(f"duplicate Natural retest decision: {packet_id}")
        packet = packet_lookup.get(packet_id)
        if packet is None:
            raise ValueError(f"review decision is outside Natural retest 1: {packet_id}")
        if raw.get("axis") not in (None, "", packet.axis.value):
            raise ValueError(f"review decision axis mismatch for {packet_id}")
        if raw.get("contract_self_check") != "YES":
            raise ValueError(
                f"contract_self_check must be exactly YES for {packet_id}"
            )
        seen.add(packet_id)
        status = AxisClassificationStatus(str(raw.get("status") or "").strip())
        status_counts[status.value] = status_counts.get(status.value, 0) + 1
        notes = str(raw.get("review_notes") or "").strip()
        if not notes:
            raise ValueError(f"review_notes are required for {packet_id}")
        row = {
            "packet_id": packet_id,
            "axis": packet.axis.value,
            "human_status": status.value,
            "human_previous_state": "",
            "human_current_state": "",
            "human_previous_source_id": "",
            "human_current_source_id": "",
            "human_previous_source_span": "",
            "human_current_source_span": "",
            "gold_split": RETEST_SPLIT,
            "gold_contract_version": RETEST_CONTRACT,
            "reviewer": "HUMAN",
            "review_notes": notes,
        }
        if status == AxisClassificationStatus.COMPLETE:
            try:
                previous_state = int(raw["previous_state"])
                current_state = int(raw["current_state"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"COMPLETE states are required for {packet_id}") from exc
            if previous_state not in (-1, 0, 1) or current_state not in (-1, 0, 1):
                raise ValueError(f"invalid HUMAN state for {packet_id}")
            previous_anchor = str(raw.get("previous_anchor") or "")
            current_anchor = str(raw.get("current_anchor") or "")
            if not previous_anchor or not current_anchor:
                raise ValueError(f"COMPLETE exact anchors are required for {packet_id}")
            previous_source_id, previous_span = _source_for_anchor(
                packet.previous_excerpts, previous_anchor
            )
            current_source_id, current_span = _source_for_anchor(
                packet.current_excerpts, current_anchor
            )
            row.update(
                human_previous_state=str(previous_state),
                human_current_state=str(current_state),
                human_previous_source_id=previous_source_id,
                human_current_source_id=current_source_id,
                human_previous_source_span=previous_span,
                human_current_source_span=current_span,
            )
        elif any(
            raw.get(key) not in (None, "")
            for key in (
                "previous_state",
                "current_state",
                "previous_anchor",
                "current_anchor",
            )
        ):
            raise ValueError(
                f"non-COMPLETE decision must leave state and anchors blank: {packet_id}"
            )
        rows.append(row)

    if seen != set(packet_lookup):
        missing = sorted(set(packet_lookup) - seen)
        extra = sorted(seen - set(packet_lookup))
        raise ValueError(
            f"HUMAN decisions must exactly cover Natural retest 1; "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    output.mkdir(parents=True, exist_ok=True)
    gold_path = output / "natural-retest-human-gold.csv"
    with gold_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GOLD_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    result = {
        "schema_version": "moatrader-v2-natural-retest-human-gold-materialization/2",
        "status": "V2_NATURAL_RETEST_1_HUMAN_GOLD_MATERIALIZED_OUTCOME_BLIND",
        "reviewer": "HUMAN",
        "human_reviewer_name": reviewer_name,
        "attestation": "YES",
        "review_date": review_date,
        "review_decision_count": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "gold_split": RETEST_SPLIT,
        "gold_contract_version": RETEST_CONTRACT,
        "review_decisions_sha256": sha256_file(review_decisions),
        "candidate_preparation_manifest_sha256": sha256_file(manifest_path),
        "natural_retest_packet_sha256": sha256_file(packet_path),
        "natural_retest_human_gold_sha256": sha256_file(gold_path),
        "source_spans_materialized_from_human_anchors": True,
        "model_fields_accepted": False,
        "contract_self_check_required": True,
        "first_natural_test_remains_consumed": True,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "natural-retest-human-gold-materialization-manifest.json", result)
    return result


def freeze_natural_retest_measurement(
    *,
    parser_freeze_manifest: Path,
    candidate_build: Path,
    human_gold_build: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Natural retest measurement freeze already exists: {output}")
    candidate_manifest_path = candidate_build / "natural-retest-preparation-manifest.json"
    packet_path = candidate_build / "natural-retest-packets.jsonl"
    materialization_path = (
        human_gold_build / "natural-retest-human-gold-materialization-manifest.json"
    )
    human_gold_path = human_gold_build / "natural-retest-human-gold.csv"
    freeze = _read_json(parser_freeze_manifest)
    candidate = _read_json(candidate_manifest_path)
    materialization = _read_json(materialization_path)
    for payload, description in (
        (freeze, "root parser freeze"),
        (candidate, "Natural retest candidate manifest"),
        (materialization, "Natural retest HUMAN materialization"),
    ):
        _require_downstream_closed(payload, description)
    if freeze.get("schema_version") != "moatrader-historical-evidence-parser-freeze-v2/2":
        raise ValueError("Natural retest must inherit the original semantic V2 freeze")
    if freeze.get("status") != "V2_PARSER_FROZEN_AWAITING_DUAL_INDEPENDENT_LOCKED_TESTS":
        raise ValueError("root semantic V2 parser freeze is invalid")
    spec = parser_spec(ParserProfile.DEMAND_PRICE_MIX_V2)
    expected = {
        "parser_profile": spec.profile.value,
        "parser_version": spec.parser_version,
        "prompt_sha256": spec.prompt_sha256,
        "requested_model": "gpt-5.6-luna",
    }
    for key, value in expected.items():
        if freeze.get(key) != value:
            raise ValueError(f"root parser freeze changed semantic V2 {key}")
    if candidate.get("status") != "V2_NATURAL_RETEST_1_PREPARED_OUTCOME_BLIND":
        raise ValueError("Natural retest candidate preparation is invalid")
    if candidate.get("natural_retest_packet_sha256") != sha256_file(packet_path):
        raise ValueError("Natural retest packets changed before freeze")
    if candidate.get("parser_freeze_sha256") != sha256_file(parser_freeze_manifest):
        raise ValueError("Natural retest candidates do not inherit the supplied root freeze")
    if materialization.get("status") != (
        "V2_NATURAL_RETEST_1_HUMAN_GOLD_MATERIALIZED_OUTCOME_BLIND"
    ):
        raise ValueError("Natural retest HUMAN gold is not materialized")
    if materialization.get("candidate_preparation_manifest_sha256") != sha256_file(
        candidate_manifest_path
    ):
        raise ValueError("Natural retest HUMAN gold does not match candidate preparation")
    if materialization.get("natural_retest_packet_sha256") != sha256_file(packet_path):
        raise ValueError("Natural retest HUMAN gold references changed packets")
    if materialization.get("natural_retest_human_gold_sha256") != sha256_file(
        human_gold_path
    ):
        raise ValueError("Natural retest HUMAN gold changed after materialization")

    result = {
        "schema_version": "moatrader-historical-evidence-parser-retest-freeze-v2/1",
        "status": "V2_NATURAL_RETEST_1_FROZEN_AWAITING_SINGLE_USE_TEST",
        "frozen_at": datetime.now(SEOUL).isoformat(),
        **expected,
        "semantic_parser_root_freeze_sha256": sha256_file(parser_freeze_manifest),
        "candidate_preparation_manifest_sha256": sha256_file(
            candidate_manifest_path
        ),
        "human_gold_materialization_manifest_sha256": sha256_file(
            materialization_path
        ),
        "natural_retest_packet_sha256": sha256_file(packet_path),
        "human_gold_sha256": sha256_file(human_gold_path),
        "gold_split": RETEST_SPLIT,
        "gold_contract_version": RETEST_CONTRACT,
        "retest_number": 1,
        "first_natural_test_remains_consumed": True,
        "first_natural_result_superseded": False,
        "locked_sets_disjoint": True,
        "v1_locked_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize and freeze independent Natural LOCKED Retest 1 HUMAN gold."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--candidate-build", type=Path, required=True)
    materialize.add_argument("--review-decisions", type=Path, required=True)
    materialize.add_argument("--output", type=Path, required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--parser-freeze-manifest", type=Path, required=True)
    freeze.add_argument("--candidate-build", type=Path, required=True)
    freeze.add_argument("--human-gold-build", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "materialize":
        result = materialize_natural_retest_human_gold(
            candidate_build=args.candidate_build,
            review_decisions=args.review_decisions,
            output=args.output,
        )
    elif args.command == "freeze":
        result = freeze_natural_retest_measurement(
            parser_freeze_manifest=args.parser_freeze_manifest,
            candidate_build=args.candidate_build,
            human_gold_build=args.human_gold_build,
            output=args.output,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
