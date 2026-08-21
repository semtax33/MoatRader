from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import (
    BlindedExcerpt,
    PairedAxisPacket,
    sha256_file,
)


class V1RSourceStratum(StrEnum):
    BUSINESS_INFO = "BUSINESS_INFO_EVIDENCE"
    FINANCE_COMMENT = "FINANCE_COMMENT_EVIDENCE"
    FINANCE_STATEMENT = "FINANCE_STATEMENT_EVIDENCE"
    MULTI_SOURCE_MOATRADER_OVERLAP = "MULTI_SOURCE_MOATRADER_OVERLAP_EVIDENCE"


STRATUM_ORIGIN = {
    V1RSourceStratum.BUSINESS_INFO: "ARCANA_BUSINESS_HTML",
    V1RSourceStratum.FINANCE_COMMENT: "ARCANA_FINANCE_COMMENT_HTML",
    V1RSourceStratum.FINANCE_STATEMENT: "ARCANA_FINANCE_STATEMENT_HTML",
}
ARCANA_ORIGINS = frozenset(STRATUM_ORIGIN.values())
MOATRADER_ORIGIN = "MOATRADER_OPENDART_ARCHIVE"
GOLD_FIELDS = (
    "packet_id",
    "base_packet_id",
    "axis",
    "source_stratum",
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


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            if hasattr(row, "model_dump_json"):
                handle.write(row.model_dump_json() + "\n")
            else:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _packet_groups(path: Path) -> Iterator[list[PairedAxisPacket]]:
    group: list[PairedAxisPacket] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            group.append(PairedAxisPacket.model_validate_json(line))
            if len(group) == len(OperatingEvidenceAxis):
                if {row.axis for row in group} != set(OperatingEvidenceAxis):
                    raise ValueError("V1R source packet group must contain exactly six axes")
                yield group
                group = []
    if group:
        raise ValueError("V1R source packet input has a trailing incomplete group")


def _external_packet_ids(path: Path) -> set[str]:
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


def _derived_packet_id(base_packet_id: str, stratum: V1RSourceStratum) -> str:
    digest = hashlib.sha256(
        f"V1R_THREE_SECTION_LOCKED|{base_packet_id}|{stratum.value}".encode("utf-8")
    ).hexdigest()
    return f"PKT_{digest[:24]}"


def _stable_key(packet_id: str, seed: str) -> str:
    return hashlib.sha256(f"{seed}|{packet_id}".encode("utf-8")).hexdigest()


def _origins(
    excerpts: Sequence[BlindedExcerpt], source_map: dict[str, dict[str, Any]]
) -> set[str]:
    return {str(source_map[row.source_id]["origin"]) for row in excerpts}


def _filter_origin(
    excerpts: Sequence[BlindedExcerpt],
    source_map: dict[str, dict[str, Any]],
    allowed: set[str],
) -> list[BlindedExcerpt]:
    return [row for row in excerpts if str(source_map[row.source_id]["origin"]) in allowed]


def _derived_packet(
    packet: PairedAxisPacket,
    *,
    stratum: V1RSourceStratum,
    source_map: dict[str, dict[str, Any]],
) -> PairedAxisPacket | None:
    if stratum == V1RSourceStratum.MULTI_SOURCE_MOATRADER_OVERLAP:
        previous_origins = _origins(packet.previous_excerpts, source_map)
        current_origins = _origins(packet.current_excerpts, source_map)
        if (
            MOATRADER_ORIGIN not in previous_origins
            or MOATRADER_ORIGIN not in current_origins
            or not (previous_origins & ARCANA_ORIGINS)
            or not (current_origins & ARCANA_ORIGINS)
        ):
            return None
        allowed = {*ARCANA_ORIGINS, MOATRADER_ORIGIN}
    else:
        allowed = {STRATUM_ORIGIN[stratum]}
    previous = _filter_origin(packet.previous_excerpts, source_map, allowed)
    current = _filter_origin(packet.current_excerpts, source_map, allowed)
    if not previous or not current:
        return None
    payload = packet.model_dump(mode="python")
    payload.update(
        packet_id=_derived_packet_id(packet.packet_id, stratum),
        previous_excerpts=previous,
        current_excerpts=current,
    )
    return PairedAxisPacket.model_validate(payload)


def prepare_v1r_locked_set(
    *,
    input_build: Path,
    prior_v1_inputs: Sequence[Path],
    dev_inputs: Sequence[Path],
    output: Path,
    minimum_per_axis_source_stratum: int = 5,
    seed: str = "MOATRADER_V1R_THREE_SECTION_LOCKED_20260821",
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    if not prior_v1_inputs:
        raise ValueError("V1R preparation requires explicit prior V1 packet inputs")
    if not dev_inputs:
        raise ValueError("V1R preparation requires explicit DEV packet inputs")
    if minimum_per_axis_source_stratum < 1:
        raise ValueError("minimum source-stratum cases must be positive")
    packet_path = input_build / "llm" / "blinded-packets.jsonl"
    private_path = input_build / "private" / "pair-source-map.jsonl"
    source_audit_path = input_build / "source-audit.json"
    for path in (
        packet_path,
        private_path,
        source_audit_path,
        *prior_v1_inputs,
        *dev_inputs,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if source_audit.get("research_variant") != "V1R":
        raise ValueError("V1R LOCKED preparation requires a V1R three-section source build")
    for key in (
        "all_arcana_sections_discovered",
        "all_arcana_sections_read_for_pairs",
        "all_arcana_sections_contributed_to_packets",
    ):
        if not source_audit.get(key, False):
            raise ValueError(f"V1R source build did not pass {key}")
    excluded_v1 = set().union(*(_external_packet_ids(path) for path in prior_v1_inputs))
    excluded_dev = set().union(*(_external_packet_ids(path) for path in dev_inputs))
    excluded = excluded_v1 | excluded_dev

    candidates: dict[
        tuple[OperatingEvidenceAxis, V1RSourceStratum],
        list[tuple[PairedAxisPacket, str, dict[str, str]]],
    ] = defaultdict(list)
    private_lines = (
        json.loads(line)
        for line in private_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    pair_count = 0
    for packets, private in zip(_packet_groups(packet_path), private_lines, strict=True):
        pair_count += 1
        source_map = dict(private["sources"])
        for packet in packets:
            if packet.packet_id in excluded:
                continue
            for stratum in V1RSourceStratum:
                derived = _derived_packet(packet, stratum=stratum, source_map=source_map)
                if derived is not None:
                    selected_source_ids = {
                        excerpt.source_id
                        for excerpt in (*derived.previous_excerpts, *derived.current_excerpts)
                    }
                    source_origins = {
                        source_id: str(source_map[source_id]["origin"])
                        for source_id in sorted(selected_source_ids)
                    }
                    candidates[(packet.axis, stratum)].append(
                        (derived, packet.packet_id, source_origins)
                    )

    selected: list[
        tuple[PairedAxisPacket, str, V1RSourceStratum, dict[str, str]]
    ] = []
    candidate_counts: dict[str, dict[str, int]] = {}
    for axis in OperatingEvidenceAxis:
        candidate_counts[axis.value] = {}
        for stratum in V1RSourceStratum:
            rows = sorted(
                candidates[(axis, stratum)],
                key=lambda item: _stable_key(item[0].packet_id, seed),
            )
            candidate_counts[axis.value][stratum.value] = len(rows)
            if len(rows) < minimum_per_axis_source_stratum:
                raise ValueError(
                    f"insufficient V1R {axis.value}/{stratum.value} candidates: {len(rows)}"
                )
            selected.extend(
                (packet, base_packet_id, stratum, source_origins)
                for packet, base_packet_id, source_origins in rows[
                    :minimum_per_axis_source_stratum
                ]
            )
    selected.sort(key=lambda row: (row[0].axis.value, row[2].value, row[0].packet_id))
    derived_ids = {row[0].packet_id for row in selected}
    base_ids = {row[1] for row in selected}
    if len(derived_ids) != len(selected):
        raise AssertionError("V1R derived LOCKED packet IDs are not unique")
    if derived_ids & excluded or base_ids & excluded:
        raise AssertionError("V1R LOCKED selection reused V1 or DEV rows")

    output.mkdir(parents=True, exist_ok=True)
    packet_output = output / "v1r-locked-packets.jsonl"
    strata_output = output / "v1r-locked-source-strata.jsonl"
    _write_jsonl(packet_output, (row[0] for row in selected))
    _write_jsonl(
        strata_output,
        (
            {
                "packet_id": packet.packet_id,
                "base_packet_id": base_packet_id,
                "axis": packet.axis.value,
                "source_stratum": stratum.value,
                "source_origins": source_origins,
            }
            for packet, base_packet_id, stratum, source_origins in selected
        ),
    )
    gold_path = output / "v1r-locked-human-gold-template.csv"
    with gold_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(GOLD_FIELDS))
        writer.writeheader()
        for packet, base_packet_id, stratum, _source_origins in selected:
            writer.writerow(
                {
                    "packet_id": packet.packet_id,
                    "base_packet_id": base_packet_id,
                    "axis": packet.axis.value,
                    "source_stratum": stratum.value,
                    "human_status": "",
                    "human_previous_state": "",
                    "human_current_state": "",
                    "human_previous_source_id": "",
                    "human_current_source_id": "",
                    "human_previous_source_span": "",
                    "human_current_source_span": "",
                    "gold_split": "V1R_LOCKED_TEST",
                    "gold_contract_version": "V1R_THREE_SECTION_SOURCE_STRATIFIED_LOCKED",
                    "reviewer": "",
                    "review_notes": "",
                }
            )
    manifest = {
        "schema_version": "moatrader-v1r-locked-set-preparation/1",
        "status": "V1R_SOURCE_STRATIFIED_LOCKED_SET_PREPARED_OUTCOME_BLIND",
        "contract_tag": "future-eri-v1r-three-section-preoutcome",
        "source_strata": [item.value for item in V1RSourceStratum],
        "minimum_per_axis_source_stratum": minimum_per_axis_source_stratum,
        "candidate_counts": candidate_counts,
        "selected_packet_count": len(selected),
        "source_pair_count": pair_count,
        "selection_seed": seed,
        "source_audit_sha256": sha256_file(source_audit_path),
        "source_packet_sha256": sha256_file(packet_path),
        "private_source_map_sha256": sha256_file(private_path),
        "prior_v1_input_sha256": [sha256_file(path) for path in prior_v1_inputs],
        "dev_input_sha256": [sha256_file(path) for path in dev_inputs],
        "prior_v1_packet_id_count": len(excluded_v1),
        "dev_packet_id_count": len(excluded_dev),
        "locked_packet_sha256": sha256_file(packet_output),
        "source_strata_sha256": sha256_file(strata_output),
        "human_gold_template_sha256": sha256_file(gold_path),
        "derived_packet_ids": True,
        "v1_locked_rows_reused": False,
        "dev_rows_reused": False,
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "locked-set-preparation-manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare a new V1R LOCKED set stratified by three Arcana sections."
    )
    parser.add_argument("--input-build", type=Path, required=True)
    parser.add_argument("--prior-v1-input", type=Path, action="append", required=True)
    parser.add_argument("--dev-input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-per-axis-source-stratum", type=int, default=5)
    parser.add_argument("--seed", default="MOATRADER_V1R_THREE_SECTION_LOCKED_20260821")
    args = parser.parse_args()
    result = prepare_v1r_locked_set(
        input_build=args.input_build,
        prior_v1_inputs=args.prior_v1_input,
        dev_inputs=args.dev_input,
        output=args.output,
        minimum_per_axis_source_stratum=args.minimum_per_axis_source_stratum,
        seed=args.seed,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
