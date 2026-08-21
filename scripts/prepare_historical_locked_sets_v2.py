from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    AxisClassificationStatus,
    AxisPairClassification,
    PairedAxisPacket,
    sha256_file,
    validate_classification_grounding,
)


SEMANTIC_AXES = (
    OperatingEvidenceAxis.DEMAND,
    OperatingEvidenceAxis.PRICE_MIX,
)
STRATA = (
    "COMPLETE_NEGATIVE",
    "COMPLETE_NEUTRAL",
    "COMPLETE_POSITIVE",
    "INSUFFICIENT_EVIDENCE",
    "AMBIGUOUS",
)
GOLD_FIELDS = (
    "packet_id",
    "axis",
    "human_status",
    "human_previous_state",
    "human_current_state",
    "human_previous_source_id",
    "human_current_source_id",
    "human_previous_source_span",
    "human_current_source_span",
    "gold_split",
    "gold_contract_version",
    "reviewer",
    "review_notes",
)
HUMAN_EVIDENCE_FIELDS = (
    "human_previous_state",
    "human_current_state",
    "human_previous_source_id",
    "human_current_source_id",
    "human_previous_source_span",
    "human_current_source_span",
)
HUMAN_REVIEW_FIELDS = (
    "human_status",
    *HUMAN_EVIDENCE_FIELDS,
    "reviewer",
    "review_notes",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[PairedAxisPacket]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _read_packets(path: Path) -> list[PairedAxisPacket]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [PairedAxisPacket.model_validate_json(line) for line in handle if line.strip()]
    if len({row.packet_id for row in rows}) != len(rows):
        raise ValueError(f"packet IDs must be unique: {path}")
    return rows


def _iter_packets(path: Path) -> Iterable[PairedAxisPacket]:
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield PairedAxisPacket.model_validate_json(line)
            except ValueError as exc:
                raise ValueError(f"invalid packet at line {number}: {exc}") from exc


def _packet_ids(path: Path) -> set[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return {
                str(row.get("packet_id") or "").strip()
                for row in csv.DictReader(handle)
                if str(row.get("packet_id") or "").strip()
            }
    result: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            packet_id = str(payload.get("packet_id") or "").strip()
            if packet_id:
                result.add(packet_id)
    return result


def _selection_key(packet: PairedAxisPacket, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{packet.packet_id}".encode("utf-8")).hexdigest()


_DIRECTION_POSITIVE_RE = re.compile(
    r"증가|성장|확대|상승|호조|회복|개선|급증|인상|상향",
    flags=re.I,
)
_DIRECTION_NEGATIVE_RE = re.compile(
    r"감소|하락|축소|둔화|부진|침체|급감|악화|인하|하향",
    flags=re.I,
)
_DIRECTION_STABLE_RE = re.compile(
    r"유지|보합|정체|동일|변동\s*(?:이\s*)?없",
    flags=re.I,
)
_DEMAND_SUBJECT_RE = re.compile(
    r"수요|매출(?:액)?|판매량|판매실적|출하량|고객\s*주문|주문량",
    flags=re.I,
)
_PRICE_MIX_SUBJECT_RE = re.compile(
    r"평균\s*판매가격|판매가격|판가|\bASP\b|제품\s*(?:믹스|mix)|"
    r"고부가(?:가치)?(?:\s*제품)?|프리미엄(?:\s*제품)?",
    flags=re.I,
)
_STRICT_DIRECTION_POSITIVE_RE = re.compile(
    r"증가|성장|확대|상승|호조|회복|개선|급증|인상|상향|고부가|프리미엄|고가",
    flags=re.I,
)
_STRICT_DIRECTION_NEGATIVE_RE = re.compile(
    r"감소|하락|축소|둔화|부진|침체|급감|악화|인하|하향|저가",
    flags=re.I,
)
_STRICT_DIRECTION_STABLE_RE = re.compile(
    r"유지|보합|정체|동일|변동\s*(?:이\s*)?없|안정",
    flags=re.I,
)
_STRICT_DEMAND_SUBJECT_RE = re.compile(
    r"수요|판매량|판매실적|출하량|주문량|고객\s*주문|트래픽|객수",
    flags=re.I,
)
_STRICT_PRICE_MIX_SUBJECT_RE = re.compile(
    r"평균\s*판매가격|판매가격|판가|\bASP\b|제품\s*(?:믹스|mix)|"
    r"고부가(?:가치)?\s*제품|프리미엄\s*제품",
    flags=re.I,
)


def _side_direction_hint(packet: PairedAxisPacket, *, previous: bool) -> int | str | None:
    """Produce an outcome-blind review hint, never a gold label."""

    excerpts = packet.previous_excerpts if previous else packet.current_excerpts
    subject_re = (
        _DEMAND_SUBJECT_RE
        if packet.axis == OperatingEvidenceAxis.DEMAND
        else _PRICE_MIX_SUBJECT_RE
    )
    observed_axis_language = False
    states: set[int] = set()
    for excerpt in excerpts:
        text = re.sub(r"\s+", " ", excerpt.text)
        matches = list(subject_re.finditer(text))
        if not matches:
            continue
        observed_axis_language = True
        for match in matches:
            start = max(0, match.start() - 180)
            end = min(len(text), match.end() + 180)
            window = text[start:end]
            positive = bool(_DIRECTION_POSITIVE_RE.search(window))
            negative = bool(_DIRECTION_NEGATIVE_RE.search(window))
            stable = bool(_DIRECTION_STABLE_RE.search(window))
            if positive and not negative:
                states.add(1)
            elif negative and not positive:
                states.add(-1)
            elif stable and not positive and not negative:
                states.add(0)
    if len(states) == 1:
        return next(iter(states))
    if len(states) > 1:
        return "AMBIGUOUS"
    if observed_axis_language:
        return "AXIS_LANGUAGE_ONLY"
    return None


def _directional_review_hint(packet: PairedAxisPacket) -> str:
    """Route candidate review only; a HUMAN must still adjudicate every row."""

    previous = _side_direction_hint(packet, previous=True)
    current = _side_direction_hint(packet, previous=False)
    if previous == "AMBIGUOUS" or current == "AMBIGUOUS":
        return "AMBIGUOUS"
    if previous == "AXIS_LANGUAGE_ONLY" and current == "AXIS_LANGUAGE_ONLY":
        return "AMBIGUOUS"
    if not isinstance(previous, int) or not isinstance(current, int):
        return "INSUFFICIENT_EVIDENCE"
    return {
        -1: "COMPLETE_NEGATIVE",
        0: "COMPLETE_NEUTRAL",
        1: "COMPLETE_POSITIVE",
    }[max(-1, min(1, current - previous))]


def _directional_review_priority(packet: PairedAxisPacket, hint: str) -> int:
    """Prefer genuinely conflicting cues over mere axis-language for AMBIGUOUS review."""

    if hint != "AMBIGUOUS":
        return 0
    previous = _side_direction_hint(packet, previous=True)
    current = _side_direction_hint(packet, previous=False)
    return 0 if "AMBIGUOUS" in (previous, current) else 1


def _strict_side_direction_hint(
    packet: PairedAxisPacket, *, previous: bool
) -> int | str | None:
    """Route a directional supplement with narrow nearby cues, never gold."""

    excerpts = packet.previous_excerpts if previous else packet.current_excerpts
    subject_re = (
        _STRICT_DEMAND_SUBJECT_RE
        if packet.axis == OperatingEvidenceAxis.DEMAND
        else _STRICT_PRICE_MIX_SUBJECT_RE
    )
    observed_axis_language = False
    states: set[int] = set()
    for excerpt in excerpts:
        text = re.sub(r"\s+", " ", excerpt.text)
        for match in subject_re.finditer(text):
            observed_axis_language = True
            window = text[max(0, match.start() - 60) : min(len(text), match.end() + 100)]
            positive = bool(_STRICT_DIRECTION_POSITIVE_RE.search(window))
            negative = bool(_STRICT_DIRECTION_NEGATIVE_RE.search(window))
            stable = bool(_STRICT_DIRECTION_STABLE_RE.search(window))
            if positive and not negative:
                states.add(1)
            elif negative and not positive:
                states.add(-1)
            elif stable and not positive and not negative:
                states.add(0)
    if len(states) == 1:
        return next(iter(states))
    if len(states) > 1:
        return "AMBIGUOUS"
    if observed_axis_language:
        return "AXIS_LANGUAGE_ONLY"
    return None


def _strict_directional_review_hint(
    packet: PairedAxisPacket,
) -> tuple[str, int | str | None, int | str | None]:
    previous = _strict_side_direction_hint(packet, previous=True)
    current = _strict_side_direction_hint(packet, previous=False)
    if isinstance(previous, int) and isinstance(current, int):
        delta = current - previous
        if delta < 0:
            return "COMPLETE_NEGATIVE", previous, current
        if delta > 0:
            return "COMPLETE_POSITIVE", previous, current
        return "COMPLETE_NEUTRAL", previous, current
    if previous == "AMBIGUOUS" or current == "AMBIGUOUS":
        return "AMBIGUOUS", previous, current
    return "INSUFFICIENT_EVIDENCE", previous, current


def _blank_gold_rows(
    packets: Sequence[PairedAxisPacket], *, split: str, contract: str
) -> list[dict[str, str]]:
    return [
        {
            "packet_id": packet.packet_id,
            "axis": packet.axis.value,
            "human_status": "",
            "human_previous_state": "",
            "human_current_state": "",
            "human_previous_source_id": "",
            "human_current_source_id": "",
            "human_previous_source_span": "",
            "human_current_source_span": "",
            "gold_split": split,
            "gold_contract_version": contract,
            "reviewer": "",
            "review_notes": "",
        }
        for packet in packets
    ]


def _write_gold(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GOLD_FIELDS), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def prepare_locked_candidates(
    *,
    packet_input: Path,
    prior_v1_inputs: Sequence[Path],
    dev_inputs: Sequence[Path],
    output: Path,
    natural_per_axis: int = 40,
    balanced_candidates_per_axis: int = 250,
    directional_candidate_stratification: bool = False,
    seed: str = "MOATRADER_V2_LOCKED_20260821",
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if not prior_v1_inputs:
        raise ValueError("explicit prior V1 inputs are required to prove no V1 row reuse")
    if not dev_inputs:
        raise ValueError("explicit DEV inputs are required to keep LOCKED rows independent")
    if natural_per_axis < 1 or balanced_candidates_per_axis < 5:
        raise ValueError("LOCKED candidate sample sizes are too small")
    prior_ids = set().union(*(_packet_ids(path) for path in prior_v1_inputs))
    dev_ids = set().union(*(_packet_ids(path) for path in dev_inputs))
    excluded = prior_ids | dev_ids
    general_keep_per_axis = natural_per_axis + balanced_candidates_per_axis
    general_heaps: dict[OperatingEvidenceAxis, list[tuple[int, str, PairedAxisPacket]]] = {
        axis: [] for axis in SEMANTIC_AXES
    }
    cue_keep_per_stratum = natural_per_axis + balanced_candidates_per_axis
    cue_heaps: dict[
        tuple[OperatingEvidenceAxis, str], list[tuple[int, str, PairedAxisPacket]]
    ] = {
        (axis, stratum): [] for axis in SEMANTIC_AXES for stratum in STRATA
    }
    cue_population_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    total_packet_count = 0
    nonsemantic_packet_count = 0
    excluded_packet_count = 0
    eligible_axis_counts: Counter[str] = Counter()
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
        heap = general_heaps[packet.axis]
        if len(heap) < general_keep_per_axis:
            heapq.heappush(heap, item)
        elif key < -heap[0][0]:
            heapq.heapreplace(heap, item)
        if directional_candidate_stratification:
            cue = _directional_review_hint(packet)
            cue_population_counts[f"{packet.axis.value}/{cue}"] += 1
            cue_heap = cue_heaps[(packet.axis, cue)]
            cue_rank_key = (
                _directional_review_priority(packet, cue) * (1 << 256) + key
            )
            cue_item = (-cue_rank_key, packet.packet_id, packet)
            if len(cue_heap) < cue_keep_per_stratum:
                heapq.heappush(cue_heap, cue_item)
            elif cue_rank_key < -cue_heap[0][0]:
                heapq.heapreplace(cue_heap, cue_item)

    natural: list[PairedAxisPacket] = []
    balanced_candidates: list[PairedAxisPacket] = []
    selected_hints: dict[str, str] = {}
    for axis in SEMANTIC_AXES:
        general_ordered = sorted(
            (item[2] for item in general_heaps[axis]),
            key=lambda row: _selection_key(row, seed),
        )
        if len(general_ordered) < natural_per_axis + balanced_candidates_per_axis:
            raise ValueError(
                f"not enough independent {axis.value} packets for Natural and Balanced pools"
            )
        axis_natural = general_ordered[:natural_per_axis]
        natural.extend(axis_natural)
        natural_axis_ids = {row.packet_id for row in axis_natural}
        axis_candidates: list[PairedAxisPacket] = []
        if directional_candidate_stratification:
            base, remainder = divmod(balanced_candidates_per_axis, len(STRATA))
            targets = {
                stratum: base + (1 if index < remainder else 0)
                for index, stratum in enumerate(STRATA)
            }
            for stratum in STRATA:
                cue_ordered = sorted(
                    (item[2] for item in cue_heaps[(axis, stratum)]),
                    key=lambda row: _selection_key(row, seed),
                )
                chosen = [
                    row
                    for row in cue_ordered
                    if row.packet_id not in natural_axis_ids
                    and row.packet_id not in selected_hints
                ][: targets[stratum]]
                axis_candidates.extend(chosen)
                selected_hints.update({row.packet_id: stratum for row in chosen})
        selected_axis_ids = {row.packet_id for row in axis_candidates}
        if len(axis_candidates) < balanced_candidates_per_axis:
            fallback = [
                row
                for row in general_ordered
                if row.packet_id not in natural_axis_ids
                and row.packet_id not in selected_axis_ids
            ][: balanced_candidates_per_axis - len(axis_candidates)]
            axis_candidates.extend(fallback)
            selected_hints.update(
                {row.packet_id: _directional_review_hint(row) for row in fallback}
            )
        balanced_candidates.extend(axis_candidates)
    natural.sort(key=lambda row: (row.axis.value, row.packet_id))
    balanced_candidates.sort(key=lambda row: (row.axis.value, row.packet_id))
    natural_ids = {row.packet_id for row in natural}
    balanced_ids = {row.packet_id for row in balanced_candidates}
    if natural_ids & balanced_ids or (natural_ids | balanced_ids) & excluded:
        raise AssertionError("LOCKED selection independence invariant failed")

    natural_path = output / "natural-locked-packets.jsonl"
    balanced_path = output / "balanced-candidate-packets.jsonl"
    _write_jsonl(natural_path, natural)
    _write_jsonl(balanced_path, balanced_candidates)
    _write_gold(
        output / "natural-human-gold-template.csv",
        _blank_gold_rows(
            natural,
            split="V2_NATURAL_LOCKED_TEST",
            contract="V2_NATURAL_FREQUENCY_LOCKED",
        ),
    )
    hint_path = output / "balanced-candidate-selection-hints.jsonl"
    with hint_path.open("w", encoding="utf-8", newline="\n") as handle:
        for packet in balanced_candidates:
            handle.write(
                json.dumps(
                    {
                        "packet_id": packet.packet_id,
                        "axis": packet.axis.value,
                        "selection_hint": selected_hints.get(
                            packet.packet_id, _directional_review_hint(packet)
                        ),
                        "gold_label": False,
                        "human_review_required": True,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    _write_gold(
        output / "balanced-candidate-human-gold-template.csv",
        _blank_gold_rows(
            balanced_candidates,
            split="V2_BALANCED_CANDIDATE_REVIEW",
            contract="V2_DIRECTIONAL_BALANCED_CANDIDATE_POOL",
        ),
    )
    manifest = {
        "schema_version": "moatrader-v2-locked-candidate-preparation/1",
        "status": "V2_INDEPENDENT_LOCKED_CANDIDATES_PREPARED_OUTCOME_BLIND",
        "selection_seed": seed,
        "semantic_parser_axes": [axis.value for axis in SEMANTIC_AXES],
        "source_packet_count": total_packet_count,
        "nonsemantic_packets_ignored": nonsemantic_packet_count,
        "excluded_prior_or_dev_packet_count": excluded_packet_count,
        "eligible_semantic_axis_counts": dict(sorted(eligible_axis_counts.items())),
        "selection_memory_policy": (
            "STREAMING_DIRECTION_CUE_STRATIFIED_HASH_KEYS_PER_AXIS"
            if directional_candidate_stratification
            else "STREAMING_SMALLEST_HASH_KEYS_PER_AXIS"
        ),
        "directional_candidate_stratification": directional_candidate_stratification,
        "directional_cue_population_counts": dict(sorted(cue_population_counts.items())),
        "natural_per_axis": natural_per_axis,
        "balanced_candidates_per_axis": balanced_candidates_per_axis,
        "source_packet_sha256": sha256_file(packet_input),
        "prior_v1_input_sha256": [sha256_file(path) for path in prior_v1_inputs],
        "dev_input_sha256": [sha256_file(path) for path in dev_inputs],
        "prior_v1_packet_id_count": len(prior_ids),
        "dev_packet_id_count": len(dev_ids),
        "natural_locked_packet_sha256": sha256_file(natural_path),
        "balanced_candidate_packet_sha256": sha256_file(balanced_path),
        "balanced_candidate_selection_hint_sha256": sha256_file(hint_path),
        "locked_sets_disjoint": True,
        "v1_locked_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "candidate-preparation-manifest.json", manifest)
    return manifest


def prepare_supplemental_candidates(
    *,
    packet_input: Path,
    base_candidate_build: Path,
    prior_v1_inputs: Sequence[Path],
    dev_inputs: Sequence[Path],
    output: Path,
    minimum_per_axis_direction: int = 5,
    seed: str = "MOATRADER_V2_DIRECTIONAL_SUPPLEMENT_20260822",
) -> dict[str, Any]:
    """Select new outcome-blind direction-cue packets after a HUMAN stratum shortfall."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if minimum_per_axis_direction < 5:
        raise ValueError("supplement must preserve the five-case directional minimum")
    if not prior_v1_inputs:
        raise ValueError("explicit prior V1 inputs are required for a supplement")
    if not dev_inputs:
        raise ValueError("explicit DEV inputs are required for a supplement")
    base_manifest_path = base_candidate_build / "candidate-preparation-manifest.json"
    base_natural_path = base_candidate_build / "natural-locked-packets.jsonl"
    base_balanced_path = base_candidate_build / "balanced-candidate-packets.jsonl"
    for path in (
        packet_input,
        base_manifest_path,
        base_natural_path,
        base_balanced_path,
        *prior_v1_inputs,
        *dev_inputs,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    if base.get("status") != "V2_INDEPENDENT_LOCKED_CANDIDATES_PREPARED_OUTCOME_BLIND":
        raise ValueError("base V2 LOCKED candidate preparation has not passed")
    if base.get("v1_locked_rows_reused", True):
        raise ValueError("base V2 candidates reuse V1 LOCKED rows")
    if base.get("outcome_vault_opened", True) or base.get("return_data_opened", True):
        raise ValueError("base V2 candidates are contaminated by downstream data")
    expected_hashes = {
        "source_packet_sha256": sha256_file(packet_input),
        "natural_locked_packet_sha256": sha256_file(base_natural_path),
        "balanced_candidate_packet_sha256": sha256_file(base_balanced_path),
    }
    for key, value in expected_hashes.items():
        if base.get(key) != value:
            raise ValueError(f"base candidate artifact hash mismatch: {key}")

    exclusion_paths = (
        base_natural_path,
        base_balanced_path,
        *prior_v1_inputs,
        *dev_inputs,
    )
    excluded = set().union(*(_packet_ids(path) for path in exclusion_paths))
    selected: list[tuple[PairedAxisPacket, str, int, int]] = []
    seen_ids: set[str] = set()
    for packet in _iter_packets(packet_input):
        if packet.packet_id in seen_ids:
            raise ValueError(f"packet IDs must be unique: {packet.packet_id}")
        seen_ids.add(packet.packet_id)
        if packet.axis not in SEMANTIC_AXES or packet.packet_id in excluded:
            continue
        hint, previous, current = _strict_directional_review_hint(packet)
        if hint not in {"COMPLETE_NEGATIVE", "COMPLETE_POSITIVE"}:
            continue
        assert isinstance(previous, int) and isinstance(current, int)
        selected.append((packet, hint, previous, current))
    selected.sort(
        key=lambda item: (
            item[0].axis.value,
            item[1],
            _selection_key(item[0], seed),
        )
    )
    counts = Counter(f"{packet.axis.value}/{hint}" for packet, hint, _, _ in selected)
    required = tuple(
        f"{axis.value}/{direction}"
        for axis in SEMANTIC_AXES
        for direction in ("COMPLETE_NEGATIVE", "COMPLETE_POSITIVE")
    )
    for key in required:
        if counts[key] < minimum_per_axis_direction:
            raise ValueError(
                f"insufficient strict supplemental candidates: {key}={counts[key]}"
            )
    selected_packets = [packet for packet, _, _, _ in selected]
    output.mkdir(parents=True, exist_ok=True)
    packet_path = output / "supplemental-candidate-packets.jsonl"
    hint_path = output / "supplemental-selection-hints.jsonl"
    _write_jsonl(packet_path, selected_packets)
    with hint_path.open("w", encoding="utf-8", newline="\n") as handle:
        for packet, hint, previous, current in selected:
            handle.write(
                json.dumps(
                    {
                        "packet_id": packet.packet_id,
                        "axis": packet.axis.value,
                        "selection_hint": hint,
                        "previous_text_hint": previous,
                        "current_text_hint": current,
                        "gold_label": False,
                        "human_review_required": True,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    manifest = {
        "schema_version": "moatrader-v2-locked-supplemental-candidate-preparation/1",
        "status": "V2_INDEPENDENT_DIRECTIONAL_SUPPLEMENT_PREPARED_OUTCOME_BLIND",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "selection_seed": seed,
        "selection_policy": "STRICT_NEARBY_REALIZED_DIRECTION_CUES_ALL_MATCHES",
        "selection_hint_is_gold_label": False,
        "human_review_required": True,
        "minimum_per_axis_direction": minimum_per_axis_direction,
        "source_packet_sha256": sha256_file(packet_input),
        "base_candidate_manifest_sha256": sha256_file(base_manifest_path),
        "exclusion_hashes": {
            str(path): sha256_file(path) for path in exclusion_paths
        },
        "excluded_packet_id_count": len(excluded),
        "supplemental_packet_count": len(selected_packets),
        "supplemental_directional_cue_counts": dict(sorted(counts.items())),
        "supplemental_packet_sha256": sha256_file(packet_path),
        "selection_hint_sha256": sha256_file(hint_path),
        "semantic_parser_axes": [axis.value for axis in SEMANTIC_AXES],
        "locked_sets_disjoint": True,
        "v1_locked_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "supplemental-candidate-preparation-manifest.json", manifest)
    return manifest


def extend_locked_candidates(
    *,
    base_candidate_build: Path,
    supplemental_candidate_build: Path,
    output: Path,
) -> dict[str, Any]:
    """Combine a frozen base pool with a disjoint review supplement."""

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    base_manifest_path = base_candidate_build / "candidate-preparation-manifest.json"
    natural_path = base_candidate_build / "natural-locked-packets.jsonl"
    balanced_path = base_candidate_build / "balanced-candidate-packets.jsonl"
    base_hint_path = base_candidate_build / "balanced-candidate-selection-hints.jsonl"
    supplement_manifest_path = (
        supplemental_candidate_build / "supplemental-candidate-preparation-manifest.json"
    )
    supplement_path = supplemental_candidate_build / "supplemental-candidate-packets.jsonl"
    supplement_hint_path = supplemental_candidate_build / "supplemental-selection-hints.jsonl"
    for path in (
        base_manifest_path,
        natural_path,
        balanced_path,
        base_hint_path,
        supplement_manifest_path,
        supplement_path,
        supplement_hint_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    base = json.loads(base_manifest_path.read_text(encoding="utf-8"))
    supplement = json.loads(supplement_manifest_path.read_text(encoding="utf-8"))
    if base.get("status") != "V2_INDEPENDENT_LOCKED_CANDIDATES_PREPARED_OUTCOME_BLIND":
        raise ValueError("base V2 LOCKED candidate preparation has not passed")
    if supplement.get("status") != (
        "V2_INDEPENDENT_DIRECTIONAL_SUPPLEMENT_PREPARED_OUTCOME_BLIND"
    ):
        raise ValueError("supplemental V2 candidate preparation has not passed")
    if supplement.get("base_candidate_manifest_sha256") != sha256_file(base_manifest_path):
        raise ValueError("supplement does not descend from the supplied base candidate build")
    if supplement.get("source_packet_sha256") != base.get("source_packet_sha256"):
        raise ValueError("base and supplement do not share one semantic packet source")
    for key, path in (
        ("natural_locked_packet_sha256", natural_path),
        ("balanced_candidate_packet_sha256", balanced_path),
        ("balanced_candidate_selection_hint_sha256", base_hint_path),
    ):
        if base.get(key) != sha256_file(path):
            raise ValueError(f"base candidate artifact hash mismatch: {key}")
    if supplement.get("supplemental_packet_sha256") != sha256_file(supplement_path):
        raise ValueError("supplemental packet hash mismatch")
    if supplement.get("selection_hint_sha256") != sha256_file(supplement_hint_path):
        raise ValueError("supplemental hint hash mismatch")
    for manifest in (base, supplement):
        if manifest.get("v1_locked_rows_reused", True):
            raise ValueError("V1 LOCKED rows cannot enter the combined V2 candidate build")
        if manifest.get("outcome_vault_opened", True) or manifest.get(
            "return_data_opened", True
        ):
            raise ValueError("candidate build is contaminated by downstream data")
    if supplement.get("selection_hint_is_gold_label", True):
        raise ValueError("supplemental selection hints cannot be accepted as gold")
    if not supplement.get("human_review_required", False):
        raise ValueError("supplemental candidates must require HUMAN review")

    natural = _read_packets(natural_path)
    balanced = _read_packets(balanced_path)
    supplemental = _read_packets(supplement_path)
    natural_ids = {packet.packet_id for packet in natural}
    balanced_ids = {packet.packet_id for packet in balanced}
    supplemental_ids = {packet.packet_id for packet in supplemental}
    if (
        natural_ids & balanced_ids
        or natural_ids & supplemental_ids
        or balanced_ids & supplemental_ids
    ):
        raise ValueError("base Natural, base Balanced, and supplement must be disjoint")
    if any(packet.axis not in SEMANTIC_AXES for packet in supplemental):
        raise ValueError("supplement contains a non-semantic-parser axis")
    combined_balanced = sorted(
        (*balanced, *supplemental), key=lambda row: (row.axis.value, row.packet_id)
    )

    def read_hints(path: Path) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = dict(json.loads(line))
                packet_id = str(value.get("packet_id") or "").strip()
                if not packet_id or packet_id in result:
                    raise ValueError(f"invalid or duplicate selection hint at {path}:{number}")
                if value.get("gold_label") is not False or value.get(
                    "human_review_required"
                ) is not True:
                    raise ValueError("selection hints must remain non-gold HUMAN-review routes")
                result[packet_id] = value
        return result

    hints = read_hints(base_hint_path)
    supplement_hints = read_hints(supplement_hint_path)
    if set(hints) != balanced_ids or set(supplement_hints) != supplemental_ids:
        raise ValueError("selection hints must exactly cover their candidate packet inputs")
    hints.update(supplement_hints)

    output.mkdir(parents=True, exist_ok=True)
    output_natural = output / "natural-locked-packets.jsonl"
    output_balanced = output / "balanced-candidate-packets.jsonl"
    output_hints = output / "balanced-candidate-selection-hints.jsonl"
    _write_jsonl(output_natural, natural)
    _write_jsonl(output_balanced, combined_balanced)
    with output_hints.open("w", encoding="utf-8", newline="\n") as handle:
        for packet in combined_balanced:
            handle.write(
                json.dumps(hints[packet.packet_id], ensure_ascii=False, sort_keys=True)
                + "\n"
            )
    _write_gold(
        output / "natural-human-gold-template.csv",
        _blank_gold_rows(
            natural,
            split="V2_NATURAL_LOCKED_TEST",
            contract="V2_NATURAL_FREQUENCY_LOCKED",
        ),
    )
    _write_gold(
        output / "balanced-candidate-human-gold-template.csv",
        _blank_gold_rows(
            combined_balanced,
            split="V2_BALANCED_CANDIDATE_REVIEW",
            contract="V2_DIRECTIONAL_BALANCED_CANDIDATE_POOL",
        ),
    )
    axis_counts = Counter(packet.axis.value for packet in combined_balanced)
    manifest = {
        "schema_version": "moatrader-v2-locked-candidate-preparation/2",
        "status": "V2_INDEPENDENT_LOCKED_CANDIDATES_PREPARED_OUTCOME_BLIND",
        "candidate_extension": True,
        "source_packet_count": base.get("source_packet_count"),
        "source_packet_sha256": base["source_packet_sha256"],
        "base_candidate_preparation_manifest_sha256": sha256_file(
            base_manifest_path
        ),
        "supplemental_candidate_preparation_manifest_sha256": sha256_file(
            supplement_manifest_path
        ),
        "base_balanced_candidate_count": len(balanced),
        "supplemental_candidate_count": len(supplemental),
        "balanced_candidate_count": len(combined_balanced),
        "balanced_candidate_axis_counts": dict(sorted(axis_counts.items())),
        "natural_locked_packet_sha256": sha256_file(output_natural),
        "balanced_candidate_packet_sha256": sha256_file(output_balanced),
        "balanced_candidate_selection_hint_sha256": sha256_file(output_hints),
        "semantic_parser_axes": [axis.value for axis in SEMANTIC_AXES],
        "selection_hint_is_gold_label": False,
        "human_review_required": True,
        "locked_sets_disjoint": True,
        "v1_locked_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "candidate-preparation-manifest.json", manifest)
    return manifest


def _classification(row: dict[str, str]) -> AxisPairClassification:
    status = AxisClassificationStatus(str(row.get("human_status") or "").strip())
    if status != AxisClassificationStatus.COMPLETE and any(
        str(row.get(name) or "").strip() for name in HUMAN_EVIDENCE_FIELDS
    ):
        raise ValueError(
            "INSUFFICIENT_EVIDENCE and AMBIGUOUS rows must leave every state/source field blank"
        )
    payload: dict[str, Any] = {
        "packet_id": str(row.get("packet_id") or "").strip(),
        "axis": str(row.get("axis") or "").strip(),
        "status": status,
        "confidence": 1.0,
    }
    if status == AxisClassificationStatus.COMPLETE:
        payload.update(
            previous_state=int(str(row.get("human_previous_state") or "").strip()),
            current_state=int(str(row.get("human_current_state") or "").strip()),
            previous_source_id=str(row.get("human_previous_source_id") or "").strip(),
            current_source_id=str(row.get("human_current_source_id") or "").strip(),
            previous_source_span=str(row.get("human_previous_source_span") or "").strip(),
            current_source_span=str(row.get("human_current_source_span") or "").strip(),
        )
    return AxisPairClassification.model_validate(payload)


def _review_row_is_blank(row: dict[str, str]) -> bool:
    return not any(str(row.get(field) or "").strip() for field in HUMAN_REVIEW_FIELDS)


def _stratum(value: AxisPairClassification) -> str:
    if value.status == AxisClassificationStatus.INSUFFICIENT_EVIDENCE:
        return "INSUFFICIENT_EVIDENCE"
    if value.status == AxisClassificationStatus.AMBIGUOUS:
        return "AMBIGUOUS"
    return {
        -1: "COMPLETE_NEGATIVE",
        0: "COMPLETE_NEUTRAL",
        1: "COMPLETE_POSITIVE",
    }[value.delta]


def finalize_locked_sets(
    *,
    candidate_build: Path,
    adjudicated_human_gold: Path,
    output: Path,
    minimum_per_axis_stratum: int = 5,
    seed: str = "MOATRADER_V2_BALANCED_20260821",
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if minimum_per_axis_stratum < 5:
        raise ValueError("Balanced LOCKED requires at least five cases per axis/stratum")
    candidate_manifest_path = candidate_build / "candidate-preparation-manifest.json"
    natural_path = candidate_build / "natural-locked-packets.jsonl"
    balanced_candidate_path = candidate_build / "balanced-candidate-packets.jsonl"
    for path in (
        candidate_manifest_path,
        natural_path,
        balanced_candidate_path,
        adjudicated_human_gold,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    candidate_manifest = json.loads(candidate_manifest_path.read_text(encoding="utf-8"))
    if candidate_manifest.get("status") != (
        "V2_INDEPENDENT_LOCKED_CANDIDATES_PREPARED_OUTCOME_BLIND"
    ):
        raise ValueError("V2 LOCKED candidate preparation has not passed")
    if candidate_manifest.get("natural_locked_packet_sha256") != sha256_file(natural_path):
        raise ValueError("Natural LOCKED packets changed after candidate preparation")
    if candidate_manifest.get("balanced_candidate_packet_sha256") != sha256_file(
        balanced_candidate_path
    ):
        raise ValueError("Balanced candidate packets changed after candidate preparation")

    natural = _read_packets(natural_path)
    candidates = _read_packets(balanced_candidate_path)
    packet_lookup = {packet.packet_id: packet for packet in (*natural, *candidates)}
    human_rows: dict[str, dict[str, str]] = {}
    human_labels: dict[str, AxisPairClassification] = {}
    with adjudicated_human_gold.open("r", encoding="utf-8-sig", newline="") as handle:
        for number, raw in enumerate(csv.DictReader(handle), start=2):
            packet_id = str(raw.get("packet_id") or "").strip()
            if packet_id not in packet_lookup:
                continue
            if _review_row_is_blank(raw):
                continue
            if packet_id in human_rows:
                raise ValueError(f"duplicate human gold packet ID at row {number}: {packet_id}")
            row = dict(raw)
            if str(row.get("reviewer") or "").strip() != "HUMAN":
                raise ValueError(
                    f"V2 gold reviewer must be tagged exactly HUMAN at row {number}"
                )
            label = _classification(row)
            validate_classification_grounding(label, packet_lookup[packet_id])
            human_rows[packet_id] = row
            human_labels[packet_id] = label
    natural_ids = {packet.packet_id for packet in natural}
    if not natural_ids.issubset(human_rows):
        raise ValueError("adjudicated human gold must cover every Natural LOCKED packet")

    buckets: dict[tuple[OperatingEvidenceAxis, str], list[PairedAxisPacket]] = defaultdict(list)
    for packet in candidates:
        label = human_labels.get(packet.packet_id)
        if label is not None:
            buckets[(packet.axis, _stratum(label))].append(packet)
    balanced: list[PairedAxisPacket] = []
    stratum_counts: dict[str, dict[str, int]] = {}
    for axis in SEMANTIC_AXES:
        stratum_counts[axis.value] = {}
        for stratum in STRATA:
            rows = sorted(
                buckets[(axis, stratum)], key=lambda row: _selection_key(row, seed)
            )
            if len(rows) < minimum_per_axis_stratum:
                raise ValueError(
                    f"insufficient human-adjudicated {axis.value}/{stratum} Balanced cases"
                )
            chosen = rows[:minimum_per_axis_stratum]
            balanced.extend(chosen)
            stratum_counts[axis.value][stratum] = len(chosen)
    balanced.sort(key=lambda row: (row.axis.value, row.packet_id))
    if natural_ids & {packet.packet_id for packet in balanced}:
        raise AssertionError("Natural and Balanced LOCKED sets overlap")

    output.mkdir(parents=True, exist_ok=True)
    output_natural = output / "natural-locked-packets.jsonl"
    output_balanced = output / "balanced-locked-packets.jsonl"
    _write_jsonl(output_natural, natural)
    _write_jsonl(output_balanced, balanced)
    gold_rows: list[dict[str, str]] = []
    for packet, split, contract in (
        *(
            (item, "V2_NATURAL_LOCKED_TEST", "V2_NATURAL_FREQUENCY_LOCKED")
            for item in natural
        ),
        *(
            (item, "V2_BALANCED_LOCKED_TEST", "V2_DIRECTIONAL_BALANCED_LOCKED")
            for item in balanced
        ),
    ):
        row = {field: str(human_rows[packet.packet_id].get(field) or "") for field in GOLD_FIELDS}
        row["packet_id"] = packet.packet_id
        row["axis"] = packet.axis.value
        row["gold_split"] = split
        row["gold_contract_version"] = contract
        gold_rows.append(row)
    gold_path = output / "v2-locked-human-gold.csv"
    _write_gold(gold_path, gold_rows)
    manifest = {
        "schema_version": "moatrader-v2-locked-set-preparation/1",
        "status": "V2_DUAL_INDEPENDENT_LOCKED_SETS_PREPARED_OUTCOME_BLIND",
        "candidate_preparation_manifest_sha256": sha256_file(candidate_manifest_path),
        "adjudicated_human_gold_sha256": sha256_file(adjudicated_human_gold),
        "natural_locked_packet_sha256": sha256_file(output_natural),
        "balanced_locked_packet_sha256": sha256_file(output_balanced),
        "human_gold_sha256": sha256_file(gold_path),
        "natural_packet_count": len(natural),
        "balanced_packet_count": len(balanced),
        "balanced_stratum_counts": stratum_counts,
        "minimum_per_axis_stratum": minimum_per_axis_stratum,
        "semantic_parser_axes": [axis.value for axis in SEMANTIC_AXES],
        "gold_label_authority": "HUMAN",
        "model_manual_labels_accepted_as_human": False,
        "locked_sets_disjoint": True,
        "v1_locked_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "locked-set-preparation-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare new, independent Natural and directional-balanced V2 LOCKED sets."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-candidates")
    prepare.add_argument("--packet-input", type=Path, required=True)
    prepare.add_argument("--prior-v1-input", type=Path, action="append", required=True)
    prepare.add_argument("--dev-input", type=Path, action="append", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--natural-per-axis", type=int, default=40)
    prepare.add_argument("--balanced-candidates-per-axis", type=int, default=250)
    prepare.add_argument("--directional-candidate-stratification", action="store_true")
    prepare.add_argument("--seed", default="MOATRADER_V2_LOCKED_20260821")
    supplement = subparsers.add_parser("prepare-supplement")
    supplement.add_argument("--packet-input", type=Path, required=True)
    supplement.add_argument("--base-candidate-build", type=Path, required=True)
    supplement.add_argument("--prior-v1-input", type=Path, action="append", required=True)
    supplement.add_argument("--dev-input", type=Path, action="append", required=True)
    supplement.add_argument("--output", type=Path, required=True)
    supplement.add_argument("--minimum-per-axis-direction", type=int, default=5)
    supplement.add_argument(
        "--seed", default="MOATRADER_V2_DIRECTIONAL_SUPPLEMENT_20260822"
    )
    extend = subparsers.add_parser("extend-candidates")
    extend.add_argument("--base-candidate-build", type=Path, required=True)
    extend.add_argument("--supplemental-candidate-build", type=Path, required=True)
    extend.add_argument("--output", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--candidate-build", type=Path, required=True)
    finalize.add_argument("--adjudicated-human-gold", type=Path, required=True)
    finalize.add_argument("--output", type=Path, required=True)
    finalize.add_argument("--minimum-per-axis-stratum", type=int, default=5)
    finalize.add_argument("--seed", default="MOATRADER_V2_BALANCED_20260821")
    args = parser.parse_args()
    if args.command == "prepare-candidates":
        result = prepare_locked_candidates(
            packet_input=args.packet_input,
            prior_v1_inputs=args.prior_v1_input,
            dev_inputs=args.dev_input,
            output=args.output,
            natural_per_axis=args.natural_per_axis,
            balanced_candidates_per_axis=args.balanced_candidates_per_axis,
            directional_candidate_stratification=args.directional_candidate_stratification,
            seed=args.seed,
        )
    elif args.command == "prepare-supplement":
        result = prepare_supplemental_candidates(
            packet_input=args.packet_input,
            base_candidate_build=args.base_candidate_build,
            prior_v1_inputs=args.prior_v1_input,
            dev_inputs=args.dev_input,
            output=args.output,
            minimum_per_axis_direction=args.minimum_per_axis_direction,
            seed=args.seed,
        )
    elif args.command == "extend-candidates":
        result = extend_locked_candidates(
            base_candidate_build=args.base_candidate_build,
            supplemental_candidate_build=args.supplemental_candidate_build,
            output=args.output,
        )
    elif args.command == "finalize":
        result = finalize_locked_sets(
            candidate_build=args.candidate_build,
            adjudicated_human_gold=args.adjudicated_human_gold,
            output=args.output,
            minimum_per_axis_stratum=args.minimum_per_axis_stratum,
            seed=args.seed,
        )
    else:  # pragma: no cover - argparse guarantees a known subcommand
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
