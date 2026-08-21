from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable

from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import sha256_file
from moatrader.expectations.historical_evidence_v2 import (
    PITApplicabilityRulesV2,
    PITOperatingSnapshotV2,
    SparseAxisAvailabilityV2,
    build_deterministic_pit_axis_evidence,
)
from scripts.build_historical_deterministic_pit_evidence_v2 import (
    LastGroundedDeterministicBasisInputV2,
    PITOperatingPairInputV2,
)


D = Decimal
DETERMINISTIC_AXES = (
    OperatingEvidenceAxis.MARGIN,
    OperatingEvidenceAxis.INVENTORY_MISMATCH,
    OperatingEvidenceAxis.BACKLOG,
    OperatingEvidenceAxis.CAPACITY_CAPEX,
)


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[LastGroundedDeterministicBasisInputV2]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _current_has_axis_inputs(
    snapshot: PITOperatingSnapshotV2,
    axis: OperatingEvidenceAxis,
) -> bool:
    if axis == OperatingEvidenceAxis.MARGIN:
        return snapshot.revenue not in (None, D(0)) and snapshot.operating_profit is not None
    if axis == OperatingEvidenceAxis.INVENTORY_MISMATCH:
        return snapshot.inventory is not None and snapshot.revenue is not None
    if axis == OperatingEvidenceAxis.BACKLOG:
        return snapshot.backlog is not None
    if axis == OperatingEvidenceAxis.CAPACITY_CAPEX:
        capex_ready = snapshot.capex is not None and snapshot.revenue not in (None, D(0))
        ppe_ready = snapshot.ppe is not None and snapshot.assets not in (None, D(0))
        return capex_ready or ppe_ready
    raise ValueError(f"unsupported deterministic axis: {axis.value}")


def _grounded_with_previous(
    *,
    previous: PITOperatingSnapshotV2,
    current: PITOperatingSnapshotV2,
    axis: OperatingEvidenceAxis,
    rules: PITApplicabilityRulesV2,
):
    return build_deterministic_pit_axis_evidence(
        previous=previous,
        current=current,
        rules=rules,
    )[axis]


def prepare_last_grounded_inputs(
    *,
    pit_pair_input: Path,
    rules_input: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    for path in (pit_pair_input, rules_input):
        if not path.is_file():
            raise FileNotFoundError(path)
    pairs = _read_jsonl(pit_pair_input, PITOperatingPairInputV2)
    rules = PITApplicabilityRulesV2.model_validate_json(
        rules_input.read_text(encoding="utf-8")
    )
    if rules.last_grounded_staleness_days != 450:
        raise ValueError("selection 2 requires the frozen 450-day staleness limit")

    snapshots_by_key: dict[tuple[str, object, object], PITOperatingSnapshotV2] = {}
    for pair in pairs:
        for snapshot in (pair.previous, pair.current):
            key = (
                snapshot.issuer_id,
                snapshot.fiscal_period_end,
                snapshot.available_at,
            )
            existing = snapshots_by_key.setdefault(key, snapshot)
            if existing != snapshot:
                raise ValueError(f"inconsistent PIT snapshot identity: {key}")
    timelines: dict[str, list[PITOperatingSnapshotV2]] = defaultdict(list)
    for snapshot in snapshots_by_key.values():
        timelines[snapshot.issuer_id].append(snapshot)
    for values in timelines.values():
        values.sort(key=lambda item: (item.fiscal_period_end, item.available_at))

    rows: list[LastGroundedDeterministicBasisInputV2] = []
    immediate_na = Counter()
    recovered = Counter()
    unrecovered = Counter()
    age_bands = Counter()
    for pair in pairs:
        immediate = build_deterministic_pit_axis_evidence(
            previous=pair.previous,
            current=pair.current,
            rules=rules,
        )
        history = [
            item
            for item in timelines[pair.current.issuer_id]
            if item.fiscal_period_end < pair.current.fiscal_period_end
            and item.available_at < pair.current.available_at
        ]
        history.sort(key=lambda item: (item.fiscal_period_end, item.available_at), reverse=True)
        for axis in DETERMINISTIC_AXES:
            if immediate[axis].availability != SparseAxisAvailabilityV2.NA:
                continue
            immediate_na[axis.value] += 1
            if not _current_has_axis_inputs(pair.current, axis):
                unrecovered[f"{axis.value}|CURRENT_EVIDENCE_MISSING"] += 1
                continue
            selected: PITOperatingSnapshotV2 | None = None
            selected_age: int | None = None
            stale_groundable = False
            for previous in history:
                age = (pair.current.available_at.date() - previous.available_at.date()).days
                replacement = _grounded_with_previous(
                    previous=previous,
                    current=pair.current,
                    axis=axis,
                    rules=rules,
                )
                if replacement.availability != SparseAxisAvailabilityV2.GROUNDED:
                    continue
                if age > rules.last_grounded_staleness_days:
                    stale_groundable = True
                    continue
                selected = previous
                selected_age = age
                break
            if selected is None or selected_age is None:
                reason = "STALE_PRIOR_ONLY" if stale_groundable else "NO_GROUNDABLE_PRIOR"
                unrecovered[f"{axis.value}|{reason}"] += 1
                continue
            replacement = _grounded_with_previous(
                previous=selected,
                current=pair.current,
                axis=axis,
                rules=rules,
            )
            rows.append(
                LastGroundedDeterministicBasisInputV2(
                    pair_id=pair.pair_id,
                    issuer_id=pair.current.issuer_id,
                    axis=axis,
                    previous=selected,
                    current_fiscal_period_end=pair.current.fiscal_period_end,
                    current_available_at=pair.current.available_at,
                    prior_age_days=selected_age,
                    staleness_limit_days=rules.last_grounded_staleness_days,
                    applicability_rule_id=(
                        f"{replacement.applicability_rule_id}__"
                        "LAST_GROUNDED_PREVIOUS_BASIS_450D"
                    ),
                )
            )
            recovered[axis.value] += 1
            if selected_age <= 120:
                age_bands["000_120"] += 1
            elif selected_age <= 240:
                age_bands["121_240"] += 1
            elif selected_age <= 365:
                age_bands["241_365"] += 1
            else:
                age_bands["366_450"] += 1

    if len({(row.pair_id, row.axis) for row in rows}) != len(rows):
        raise ValueError("last-grounded basis rows must be unique by pair and axis")
    rows.sort(key=lambda item: (item.pair_id, item.axis.value))
    output.mkdir(parents=True, exist_ok=True)
    input_path = output / "last-grounded-deterministic-bases.jsonl"
    _write_jsonl(input_path, rows)
    report = {
        "schema_version": "moatrader-last-grounded-input-preparation-v2/1",
        "status": "LAST_GROUNDED_PREVIOUS_BASES_PREPARED_OUTCOME_BLIND",
        "pair_count": len(pairs),
        "unique_snapshot_count": len(snapshots_by_key),
        "staleness_limit_days": rules.last_grounded_staleness_days,
        "selection_policy": "LATEST_GROUNDABLE_PREVIOUS_FILING_WITHIN_450D",
        "current_evidence_carried_forward": False,
        "immediate_na_count_by_axis": dict(sorted(immediate_na.items())),
        "recovered_count": len(rows),
        "recovered_count_by_axis": dict(sorted(recovered.items())),
        "unrecovered_reason_distribution": dict(sorted(unrecovered.items())),
        "prior_age_band_distribution": dict(sorted(age_bands.items())),
        "pit_pair_input_sha256": sha256_file(pit_pair_input),
        "rules_input_sha256": sha256_file(rules_input),
        "last_grounded_input_sha256": sha256_file(input_path),
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "value_data_opened": False,
        "source_files_modified": False,
        "source_write_operations": 0,
    }
    _write_json(output / "stage-status.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare outcome-blind 450-day last-grounded previous comparison bases "
            "without carrying forward missing current evidence."
        )
    )
    parser.add_argument("--pit-pair-input", type=Path, required=True)
    parser.add_argument("--rules-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_last_grounded_inputs(
        pit_pair_input=args.pit_pair_input,
        rules_input=args.rules_input,
        output=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
