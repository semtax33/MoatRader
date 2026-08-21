from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisPairClassification,
    PairedAxisPacket,
    sha256_file,
)
from scripts.classify_historical_future_eri_evidence import (
    ParserProfile,
    parser_spec,
)
from scripts.prepare_historical_locked_sets_v2 import (
    SEMANTIC_AXES,
    _blank_gold_rows,
    _classification,
    _directional_review_hint,
    _iter_packets,
    _packet_ids,
    _selection_key,
    _stratum,
    _write_gold,
    _write_json,
    _write_jsonl,
)


CANDIDATE_SPLIT = "V2_BALANCED_RETEST_1_CANDIDATE_REVIEW"
CANDIDATE_CONTRACT = "V2_DIRECTIONAL_BALANCED_RETEST_1_CANDIDATE_POOL"
RETEST_SPLIT = "V2_BALANCED_LOCKED_RETEST_1"
RETEST_CONTRACT = "V2_DIRECTIONAL_BALANCED_LOCKED_RETEST_1"
RETEST_SEED = "MOATRADER_V2_BALANCED_RETEST_1_20260822"

_PROSPECTIVE_RE = re.compile(
    r"전망|예상|계획|기대|목표|가능성|향후|예측|추정|것으로\s*(?:보|판단)|예정",
    flags=re.I,
)
_REALIZED_RE = re.compile(
    r"전년|전기|당기|금년|상반기|하반기|분기|누계|대비|실적|기록|달성|"
    r"증감|변동|추이|현재|최근|말\s*기준",
    flags=re.I,
)
_POSITIVE_RE = re.compile(
    r"증가|성장|확대|상승|호조|회복|개선|급증|인상|상향|고부가|프리미엄",
    flags=re.I,
)
_NEGATIVE_RE = re.compile(
    r"감소|하락|축소|둔화|부진|침체|급감|악화|인하|하향|저가",
    flags=re.I,
)
_STABLE_RE = re.compile(
    r"유지|보합|정체|동일|변동\s*(?:이\s*)?없|안정",
    flags=re.I,
)
_SUBJECTS: dict[OperatingEvidenceAxis, tuple[tuple[str, re.Pattern[str]], ...]] = {
    OperatingEvidenceAxis.DEMAND: (
        ("DEMAND", re.compile(r"수요", flags=re.I)),
        (
            "VOLUME",
            re.compile(
                r"판매량|출하량|주문량|수주량|물동량|트래픽|Traffic|객수",
                flags=re.I,
            ),
        ),
        (
            "ADOPTION",
            re.compile(r"이용자|사용자|가입자|도입|채택|가동률", flags=re.I),
        ),
    ),
    OperatingEvidenceAxis.PRICE_MIX: (
        (
            "PRICE",
            re.compile(r"평균\s*판매가격|판매가격|판매단가|판가|\bASP\b", flags=re.I),
        ),
        (
            "MIX",
            re.compile(
                r"제품\s*(?:믹스|mix)|고부가(?:가치)?\s*제품\s*비중|"
                r"프리미엄\s*제품\s*비중",
                flags=re.I,
            ),
        ),
    ),
}


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


def _selection_key_from_id(packet_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{packet_id}".encode("utf-8")).hexdigest()


def _side_realized_direction(
    packet: PairedAxisPacket, *, previous: bool
) -> tuple[int | str | None, frozenset[str]]:
    excerpts = packet.previous_excerpts if previous else packet.current_excerpts
    states: set[int] = set()
    families: set[str] = set()
    for excerpt in excerpts:
        text = re.sub(r"\s+", " ", excerpt.text)
        for family, subject_re in _SUBJECTS[packet.axis]:
            for match in subject_re.finditer(text):
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 160)
                window = text[start:end]
                realized = bool(_REALIZED_RE.search(window))
                if _PROSPECTIVE_RE.search(window) and not realized:
                    continue
                positive = bool(_POSITIVE_RE.search(window))
                negative = bool(_NEGATIVE_RE.search(window))
                stable = bool(_STABLE_RE.search(window))
                if not realized:
                    compact = re.sub(r"\s+", "", window)
                    subject = re.sub(r"\s+", "", match.group(0))
                    if not re.search(
                        re.escape(subject)
                        + r".{0,45}(?:증가|감소|상승|하락|인상|인하|유지|동일|보합|정체)",
                        compact,
                        flags=re.I,
                    ):
                        continue
                if positive and not negative:
                    states.add(1)
                    families.add(family)
                elif negative and not positive:
                    states.add(-1)
                    families.add(family)
                elif stable and not positive and not negative:
                    states.add(0)
                    families.add(family)
    if len(states) == 1:
        return next(iter(states)), frozenset(families)
    if len(states) > 1:
        return "AMBIGUOUS", frozenset(families)
    return None, frozenset()


def _candidate_hint(packet: PairedAxisPacket) -> str:
    previous, previous_families = _side_realized_direction(packet, previous=True)
    current, current_families = _side_realized_direction(packet, previous=False)
    if isinstance(previous, int) and isinstance(current, int):
        if not previous_families.intersection(current_families):
            return "AMBIGUOUS"
        delta = current - previous
        if delta < 0:
            return "COMPLETE_NEGATIVE"
        if delta > 0:
            return "COMPLETE_POSITIVE"
        return "COMPLETE_NEUTRAL"
    if previous == "AMBIGUOUS" or current == "AMBIGUOUS":
        return "AMBIGUOUS"
    return "INSUFFICIENT_EVIDENCE"


def _read_prior_human_strata(
    *, human_gold: Path, materialization_manifest: Path
) -> tuple[dict[tuple[OperatingEvidenceAxis, str], list[str]], set[str]]:
    materialization = _read_json(materialization_manifest)
    if materialization.get("status") != (
        "V2_HUMAN_REVIEW_DECISIONS_MATERIALIZED_OUTCOME_BLIND"
    ):
        raise ValueError("prior HUMAN gold is not an outcome-blind materialization")
    if materialization.get("reviewer") != "HUMAN":
        raise ValueError("prior HUMAN gold lacks HUMAN authority")
    if materialization.get("adjudicated_human_gold_sha256") != sha256_file(human_gold):
        raise ValueError("prior HUMAN gold changed after materialization")
    _require_downstream_closed(materialization, "prior HUMAN materialization")

    buckets: dict[tuple[OperatingEvidenceAxis, str], list[str]] = defaultdict(list)
    all_ids: set[str] = set()
    with human_gold.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        forbidden = [
            name
            for name in (reader.fieldnames or [])
            if name.casefold().startswith(
                ("machine_", "model_", "predicted_", "parser_", "classification_")
            )
        ]
        if forbidden:
            raise ValueError(f"prior HUMAN gold contains model-derived fields: {forbidden}")
        for number, raw in enumerate(reader, start=2):
            if str(raw.get("reviewer") or "").strip() != "HUMAN":
                raise ValueError(f"prior HUMAN gold reviewer is not HUMAN at row {number}")
            label: AxisPairClassification = _classification(dict(raw))
            if label.packet_id in all_ids:
                raise ValueError(f"duplicate prior HUMAN packet ID: {label.packet_id}")
            all_ids.add(label.packet_id)
            buckets[(label.axis, _stratum(label))].append(label.packet_id)
    return buckets, all_ids


def prepare_balanced_retest_candidates(
    *,
    packet_input: Path,
    prior_v1_inputs: Sequence[Path],
    dev_inputs: Sequence[Path],
    prior_v2_locked_inputs: Sequence[Path],
    prior_human_gold: Path,
    prior_human_gold_materialization_manifest: Path,
    failed_balanced_evaluation_manifest: Path,
    failed_balanced_consumption_record: Path,
    parser_freeze_manifest: Path,
    output: Path,
    directional_candidates_per_axis_stratum: int = 25,
    nondirectional_candidates_per_axis_stratum: int = 10,
    seed: str = RETEST_SEED,
) -> dict[str, Any]:
    """Prepare an independent balanced-retest review pool without model outputs."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if directional_candidates_per_axis_stratum < 10:
        raise ValueError("Balanced retest needs at least ten directional candidates per bucket")
    if nondirectional_candidates_per_axis_stratum < 5:
        raise ValueError("Balanced retest needs at least five non-direction candidates per bucket")
    if not prior_v1_inputs or not dev_inputs:
        raise ValueError("explicit prior V1 and semantic DEV inputs are required")
    if len(prior_v2_locked_inputs) < 2:
        raise ValueError("both prior V2 Natural and Balanced LOCKED inputs are required")
    for path in (
        packet_input,
        prior_human_gold,
        prior_human_gold_materialization_manifest,
        failed_balanced_evaluation_manifest,
        failed_balanced_consumption_record,
        parser_freeze_manifest,
        *prior_v1_inputs,
        *dev_inputs,
        *prior_v2_locked_inputs,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    freeze = _read_json(parser_freeze_manifest)
    if freeze.get("schema_version") != "moatrader-historical-evidence-parser-freeze-v2/2":
        raise ValueError("Balanced retest requires the original semantic V2 parser freeze")
    if freeze.get("status") != "V2_PARSER_FROZEN_AWAITING_DUAL_INDEPENDENT_LOCKED_TESTS":
        raise ValueError("semantic V2 parser is not frozen")
    _require_frozen_parser_contract(freeze)
    _require_downstream_closed(freeze, "parser freeze")

    evaluation = _read_json(failed_balanced_evaluation_manifest)
    if evaluation.get("status") != "V2_BALANCED_EVIDENCE_PARSER_NOT_VALIDATED":
        raise ValueError("Balanced retest requires a failed first Balanced evaluation")
    if evaluation.get("locked_kind") != "BALANCED" or evaluation.get("gate_passed") is not False:
        raise ValueError("supplied evaluation is not the failed Balanced LOCKED gate")
    if evaluation.get("parser_freeze_sha256") != sha256_file(parser_freeze_manifest):
        raise ValueError("failed Balanced evaluation does not reference the supplied freeze")
    if evaluation.get("input_blinded_packet_sha256") != freeze.get(
        "balanced_locked_packet_sha256"
    ):
        raise ValueError("failed Balanced evaluation packet lineage is invalid")
    _require_downstream_closed(evaluation, "failed Balanced evaluation")

    consumption = _read_json(failed_balanced_consumption_record)
    if consumption.get("status") != "COMPLETED_SINGLE_USE":
        raise ValueError("first Balanced LOCKED test has not been consumed")
    if consumption.get("locked_kind") != "BALANCED" or consumption.get("gate_passed") is not False:
        raise ValueError("Balanced retest requires a consumed failing Balanced test")
    if consumption.get("parser_freeze_sha256") != sha256_file(parser_freeze_manifest):
        raise ValueError("Balanced consumption record does not reference the supplied freeze")
    if consumption.get("locked_packet_sha256") != freeze.get(
        "balanced_locked_packet_sha256"
    ):
        raise ValueError("Balanced consumption packet hash does not match the parser freeze")
    _require_downstream_closed(consumption, "Balanced consumption record")

    prior_v2_hashes = {sha256_file(path) for path in prior_v2_locked_inputs}
    required_v2_hashes = {
        str(freeze.get("natural_locked_packet_sha256") or ""),
        str(freeze.get("balanced_locked_packet_sha256") or ""),
    }
    if "" in required_v2_hashes or not required_v2_hashes.issubset(prior_v2_hashes):
        raise ValueError("prior V2 inputs must include the frozen Natural and Balanced sets")

    prior_human_buckets, all_prior_human_ids = _read_prior_human_strata(
        human_gold=prior_human_gold,
        materialization_manifest=prior_human_gold_materialization_manifest,
    )
    prior_v1_ids = set().union(*(_packet_ids(path) for path in prior_v1_inputs))
    dev_ids = set().union(*(_packet_ids(path) for path in dev_inputs))
    prior_v2_ids = set().union(*(_packet_ids(path) for path in prior_v2_locked_inputs))
    used_ids = prior_v1_ids | dev_ids | prior_v2_ids

    nondirectional_targets: dict[tuple[OperatingEvidenceAxis, str], set[str]] = {}
    fresh_targets: dict[tuple[OperatingEvidenceAxis, str], int] = {}
    for axis in SEMANTIC_AXES:
        for stratum in ("COMPLETE_NEUTRAL", "INSUFFICIENT_EVIDENCE", "AMBIGUOUS"):
            ordered = sorted(
                (
                    packet_id
                    for packet_id in prior_human_buckets[(axis, stratum)]
                    if packet_id not in used_ids
                ),
                key=lambda packet_id: _selection_key_from_id(packet_id, seed),
            )
            chosen = ordered[:nondirectional_candidates_per_axis_stratum]
            nondirectional_targets[(axis, stratum)] = set(chosen)
            fresh_targets[(axis, stratum)] = (
                nondirectional_candidates_per_axis_stratum - len(chosen)
            )
        for stratum in ("COMPLETE_NEGATIVE", "COMPLETE_POSITIVE"):
            fresh_targets[(axis, stratum)] = directional_candidates_per_axis_stratum
    selected_prior_human_ids = set().union(*nondirectional_targets.values())

    fresh_heaps: dict[
        tuple[OperatingEvidenceAxis, str], list[tuple[int, str, PairedAxisPacket]]
    ] = {
        (axis, stratum): []
        for axis in SEMANTIC_AXES
        for stratum in (
            "COMPLETE_NEGATIVE",
            "COMPLETE_NEUTRAL",
            "COMPLETE_POSITIVE",
            "INSUFFICIENT_EVIDENCE",
            "AMBIGUOUS",
        )
    }
    selected_prior_packets: dict[str, PairedAxisPacket] = {}
    fresh_hint_population: Counter[str] = Counter()
    source_packet_count = 0
    nonsemantic_packet_count = 0
    excluded_used_count = 0
    excluded_prior_human_count = 0
    seen_ids: set[str] = set()
    for packet in _iter_packets(packet_input):
        source_packet_count += 1
        if packet.packet_id in seen_ids:
            raise ValueError(f"packet IDs must be unique: {packet.packet_id}")
        seen_ids.add(packet.packet_id)
        if packet.axis not in SEMANTIC_AXES:
            nonsemantic_packet_count += 1
            continue
        if packet.packet_id in selected_prior_human_ids:
            selected_prior_packets[packet.packet_id] = packet
            continue
        if packet.packet_id in used_ids:
            excluded_used_count += 1
            continue
        if packet.packet_id in all_prior_human_ids:
            excluded_prior_human_count += 1
            continue
        refined_hint = _candidate_hint(packet)
        hint = refined_hint
        priority = 0
        if refined_hint not in {"COMPLETE_NEGATIVE", "COMPLETE_POSITIVE"}:
            broad_hint = _directional_review_hint(packet)
            if broad_hint in {"COMPLETE_NEGATIVE", "COMPLETE_POSITIVE"}:
                hint = broad_hint
                priority = 1
        bucket_key = (packet.axis, hint)
        target_count = fresh_targets[bucket_key]
        fresh_hint_population[f"{packet.axis.value}/{hint}"] += 1
        if target_count == 0:
            continue
        key = int(_selection_key(packet, seed), 16)
        rank_key = priority * (1 << 256) + key
        item = (-rank_key, packet.packet_id, packet)
        heap = fresh_heaps[bucket_key]
        if len(heap) < target_count:
            heapq.heappush(heap, item)
        elif rank_key < -heap[0][0]:
            heapq.heapreplace(heap, item)

    if set(selected_prior_packets) != selected_prior_human_ids:
        missing = sorted(selected_prior_human_ids - set(selected_prior_packets))
        raise ValueError(f"prior HUMAN candidate packets are missing from source: {missing[:5]}")

    selected: list[PairedAxisPacket] = list(selected_prior_packets.values())
    selection_hints: dict[str, dict[str, Any]] = {}
    for (axis, stratum), ids in nondirectional_targets.items():
        for packet_id in ids:
            selection_hints[packet_id] = {
                "selection_hint": stratum,
                "selection_source": "PRIOR_HUMAN_STRATUM_FOR_SAMPLING_ONLY",
            }
    for axis in SEMANTIC_AXES:
        for stratum in (
            "COMPLETE_NEGATIVE",
            "COMPLETE_NEUTRAL",
            "COMPLETE_POSITIVE",
            "INSUFFICIENT_EVIDENCE",
            "AMBIGUOUS",
        ):
            rows = sorted(
                (item[2] for item in fresh_heaps[(axis, stratum)]),
                key=lambda packet: _selection_key(packet, seed),
            )
            target_count = fresh_targets[(axis, stratum)]
            prior_count = len(nondirectional_targets.get((axis, stratum), set()))
            minimum_total = 10 if stratum in {
                "COMPLETE_NEGATIVE",
                "COMPLETE_POSITIVE",
            } else 5
            if len(rows) + prior_count < minimum_total:
                raise ValueError(
                    f"insufficient independent {axis.value}/{stratum} fresh candidates: "
                    f"fresh={len(rows)} prior_human_sampling={prior_count} "
                    f"minimum_total={minimum_total} requested_fresh={target_count}"
                )
            selected.extend(rows)
            for packet in rows:
                refined_hint = _candidate_hint(packet)
                selection_hints[packet.packet_id] = {
                    "selection_hint": stratum,
                    "selection_source": (
                        "OUTCOME_BLIND_REALIZED_TEXT_CUE_NOT_GOLD"
                        if refined_hint == stratum
                        else "OUTCOME_BLIND_BROAD_TEXT_CUE_FALLBACK_NOT_GOLD"
                    ),
                }
    selected.sort(key=lambda packet: (packet.axis.value, packet.packet_id))
    selected_ids = {packet.packet_id for packet in selected}
    if len(selected_ids) != len(selected) or selected_ids & used_ids:
        raise AssertionError("Balanced retest candidate independence invariant failed")

    output.mkdir(parents=True, exist_ok=True)
    packet_output = output / "balanced-retest-candidate-packets.jsonl"
    template_output = output / "balanced-retest-human-gold-template.csv"
    hint_output = output / "balanced-retest-selection-hints.jsonl"
    _write_jsonl(packet_output, selected)
    _write_gold(
        template_output,
        _blank_gold_rows(selected, split=CANDIDATE_SPLIT, contract=CANDIDATE_CONTRACT),
    )
    with hint_output.open("w", encoding="utf-8", newline="\n") as handle:
        for packet in selected:
            hint = selection_hints[packet.packet_id]
            handle.write(
                json.dumps(
                    {
                        "packet_id": packet.packet_id,
                        "axis": packet.axis.value,
                        **hint,
                        "gold_label": False,
                        "human_review_required": True,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    hint_counts = Counter(
        f"{packet.axis.value}/{selection_hints[packet.packet_id]['selection_hint']}"
        for packet in selected
    )
    manifest = {
        "schema_version": "moatrader-v2-balanced-retest-candidate-preparation/1",
        "status": "V2_BALANCED_RETEST_1_CANDIDATES_PREPARED_OUTCOME_BLIND",
        "retest_number": 1,
        "selection_seed": seed,
        "selection_policy": (
            "FRESH_REALIZED_TEXT_DIRECTION_CUES_THEN_BROAD_TEXT_CUE_FALLBACK_PLUS_"
            "UNUSED_PRIOR_HUMAN_NONDIRECTION_STRATA_FRESH_REVIEW_REQUIRED"
        ),
        "selection_used_parser_classifications": False,
        "selection_used_post_test_disagreement_rows": False,
        "selection_used_prior_human_strata_only": True,
        "prior_human_labels_accepted_as_retest_gold": False,
        "fresh_independent_human_review_required": True,
        "selection_hints_exposed_to_reviewer": False,
        "first_balanced_test_remains_consumed": True,
        "first_balanced_result_superseded": False,
        "semantic_parser_axes": [axis.value for axis in SEMANTIC_AXES],
        "candidate_gold_split": CANDIDATE_SPLIT,
        "candidate_gold_contract_version": CANDIDATE_CONTRACT,
        "final_gold_split": RETEST_SPLIT,
        "final_gold_contract_version": RETEST_CONTRACT,
        "source_packet_count": source_packet_count,
        "nonsemantic_packets_ignored": nonsemantic_packet_count,
        "excluded_used_packet_count": excluded_used_count,
        "excluded_prior_human_packet_count": excluded_prior_human_count,
        "fresh_candidate_hint_population_counts": dict(
            sorted(fresh_hint_population.items())
        ),
        "directional_candidates_per_axis_stratum": directional_candidates_per_axis_stratum,
        "nondirectional_candidates_per_axis_stratum": (
            nondirectional_candidates_per_axis_stratum
        ),
        "prior_human_sampling_counts": {
            f"{axis.value}/{stratum}": len(ids)
            for (axis, stratum), ids in sorted(
                nondirectional_targets.items(),
                key=lambda item: (item[0][0].value, item[0][1]),
            )
        },
        "selection_hint_counts": dict(sorted(hint_counts.items())),
        "candidate_packet_count": len(selected),
        "source_packet_sha256": sha256_file(packet_input),
        "prior_v1_input_sha256": [sha256_file(path) for path in prior_v1_inputs],
        "dev_input_sha256": [sha256_file(path) for path in dev_inputs],
        "prior_v2_locked_input_sha256": [
            sha256_file(path) for path in prior_v2_locked_inputs
        ],
        "prior_human_gold_sha256": sha256_file(prior_human_gold),
        "prior_human_gold_materialization_manifest_sha256": sha256_file(
            prior_human_gold_materialization_manifest
        ),
        "failed_balanced_evaluation_manifest_sha256": sha256_file(
            failed_balanced_evaluation_manifest
        ),
        "failed_balanced_consumption_record_sha256": sha256_file(
            failed_balanced_consumption_record
        ),
        "parser_freeze_sha256": sha256_file(parser_freeze_manifest),
        "balanced_retest_candidate_packet_sha256": sha256_file(packet_output),
        "balanced_retest_human_gold_template_sha256": sha256_file(template_output),
        "balanced_retest_selection_hint_sha256": sha256_file(hint_output),
        "prior_v1_packet_id_count": len(prior_v1_ids),
        "dev_packet_id_count": len(dev_ids),
        "prior_v2_locked_packet_id_count": len(prior_v2_ids),
        "all_used_exclusion_overlaps": 0,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "balanced-retest-preparation-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare an independent, outcome-blind Balanced LOCKED retest review "
            "pool after a consumed failing first test."
        )
    )
    parser.add_argument("--packet-input", type=Path, required=True)
    parser.add_argument("--prior-v1-input", type=Path, action="append", required=True)
    parser.add_argument("--dev-input", type=Path, action="append", required=True)
    parser.add_argument("--prior-v2-locked-input", type=Path, action="append", required=True)
    parser.add_argument("--prior-human-gold", type=Path, required=True)
    parser.add_argument(
        "--prior-human-gold-materialization-manifest", type=Path, required=True
    )
    parser.add_argument("--failed-balanced-evaluation-manifest", type=Path, required=True)
    parser.add_argument("--failed-balanced-consumption-record", type=Path, required=True)
    parser.add_argument("--parser-freeze-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directional-candidates-per-axis-stratum", type=int, default=25)
    parser.add_argument("--nondirectional-candidates-per-axis-stratum", type=int, default=10)
    parser.add_argument("--seed", default=RETEST_SEED)
    args = parser.parse_args()
    result = prepare_balanced_retest_candidates(
        packet_input=args.packet_input,
        prior_v1_inputs=args.prior_v1_input,
        dev_inputs=args.dev_input,
        prior_v2_locked_inputs=args.prior_v2_locked_input,
        prior_human_gold=args.prior_human_gold,
        prior_human_gold_materialization_manifest=(
            args.prior_human_gold_materialization_manifest
        ),
        failed_balanced_evaluation_manifest=args.failed_balanced_evaluation_manifest,
        failed_balanced_consumption_record=args.failed_balanced_consumption_record,
        parser_freeze_manifest=args.parser_freeze_manifest,
        output=args.output,
        directional_candidates_per_axis_stratum=(
            args.directional_candidates_per_axis_stratum
        ),
        nondirectional_candidates_per_axis_stratum=(
            args.nondirectional_candidates_per_axis_stratum
        ),
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
