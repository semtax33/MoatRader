from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel
from moatrader.expectations.future_eri import OperatingEvidenceAxis
from moatrader.expectations.historical_evidence import canonical_payload_sha256, sha256_file
from moatrader.expectations.historical_evidence_v2 import (
    AxisApplicabilityV2,
    PITApplicabilityRulesV2,
    PITOperatingSnapshotV2,
    PreviousEvidenceBasisV2,
    SparseAxisAvailabilityV2,
    SparseAxisEvidenceV2,
    build_deterministic_pit_axis_evidence,
)
from scripts.build_historical_sparse_features_v2 import (
    AxisApplicabilityDecisionInputV2,
    DeterministicAxisEvidenceInputV2,
)


class PITOperatingPairInputV2(ContractModel):
    schema_version: str = "moatrader-pit-operating-pair-input-v2/1"
    pair_id: str = Field(pattern=r"^PAIR_[0-9a-f]{24}$")
    previous: PITOperatingSnapshotV2
    current: PITOperatingSnapshotV2
    outcome_data_accessed: bool = False
    return_data_accessed: bool = False

    @model_validator(mode="after")
    def issuer_and_blindness(self) -> "PITOperatingPairInputV2":
        if self.previous.issuer_id != self.current.issuer_id:
            raise ValueError("PIT operating pair issuer mismatch")
        if self.outcome_data_accessed or self.return_data_accessed:
            raise ValueError("PIT operating pair must be outcome and return blind")
        return self


class LastGroundedDeterministicBasisInputV2(ContractModel):
    schema_version: str = "moatrader-last-grounded-deterministic-basis-v2/1"
    pair_id: str = Field(pattern=r"^PAIR_[0-9a-f]{24}$")
    issuer_id: str = Field(pattern=r"^[0-9]{6}$")
    axis: OperatingEvidenceAxis
    previous: PITOperatingSnapshotV2
    current_fiscal_period_end: date
    current_available_at: datetime
    prior_age_days: int = Field(ge=0, le=450)
    staleness_limit_days: Literal[450] = 450
    applicability_rule_id: str = Field(min_length=1)
    selection_policy: Literal[
        "LATEST_GROUNDABLE_PREVIOUS_FILING_WITHIN_450D"
    ] = "LATEST_GROUNDABLE_PREVIOUS_FILING_WITHIN_450D"
    current_evidence_carried_forward: Literal[False] = False
    outcome_data_accessed: Literal[False] = False
    return_data_accessed: Literal[False] = False

    @model_validator(mode="after")
    def valid_previous_basis(self) -> "LastGroundedDeterministicBasisInputV2":
        if self.axis not in {
            OperatingEvidenceAxis.MARGIN,
            OperatingEvidenceAxis.INVENTORY_MISMATCH,
            OperatingEvidenceAxis.BACKLOG,
            OperatingEvidenceAxis.CAPACITY_CAPEX,
        }:
            raise ValueError("last-grounded deterministic basis supports only four numeric axes")
        if self.previous.issuer_id != self.issuer_id:
            raise ValueError("last-grounded previous basis issuer mismatch")
        if self.current_available_at.tzinfo is None or self.current_available_at.utcoffset() is None:
            raise ValueError("last-grounded current_available_at must be timezone-aware")
        if self.previous.fiscal_period_end >= self.current_fiscal_period_end:
            raise ValueError("last-grounded previous basis must precede the current fiscal period")
        if self.previous.available_at >= self.current_available_at:
            raise ValueError("last-grounded previous basis was not PIT-available before current")
        expected_age = (
            self.current_available_at.date() - self.previous.available_at.date()
        ).days
        if self.prior_age_days != expected_age:
            raise ValueError("last-grounded prior_age_days does not match availability dates")
        if self.prior_age_days > self.staleness_limit_days:
            raise ValueError("last-grounded previous basis exceeds the 450-day limit")
        return self


def _read_jsonl(path: Path, model: type[Any]) -> list[Any]:
    with path.open("r", encoding="utf-8") as handle:
        return [model.model_validate_json(line) for line in handle if line.strip()]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[ContractModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def build_pit_evidence(
    *,
    pit_pair_input: Path,
    rules_input: Path,
    output: Path,
    last_grounded_input: Path | None = None,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be new or empty: {output}")
    for path in (pit_pair_input, rules_input):
        if not path.is_file():
            raise FileNotFoundError(path)
    rules = PITApplicabilityRulesV2.model_validate_json(rules_input.read_text(encoding="utf-8"))
    pairs = _read_jsonl(pit_pair_input, PITOperatingPairInputV2)
    if len({row.pair_id for row in pairs}) != len(pairs):
        raise ValueError("PIT operating pair IDs must be unique")
    last_rows = (
        _read_jsonl(last_grounded_input, LastGroundedDeterministicBasisInputV2)
        if last_grounded_input is not None
        else []
    )
    last_by_key = {(row.pair_id, row.axis): row for row in last_rows}
    if len(last_by_key) != len(last_rows):
        raise ValueError("last-grounded inputs must be unique by pair and axis")

    deterministic_rows: list[DeterministicAxisEvidenceInputV2] = []
    applicability_rows: list[AxisApplicabilityDecisionInputV2] = []
    used_last: set[tuple[str, OperatingEvidenceAxis]] = set()
    for pair in pairs:
        pit = build_deterministic_pit_axis_evidence(
            previous=pair.previous,
            current=pair.current,
            rules=rules,
        )
        for axis in OperatingEvidenceAxis:
            candidate: SparseAxisEvidenceV2 | None = pit.get(axis)
            last = last_by_key.get((pair.pair_id, axis))
            if last is not None and candidate is not None and (
                candidate.availability == SparseAxisAvailabilityV2.NA
            ):
                if last.issuer_id != pair.current.issuer_id:
                    raise ValueError("last-grounded basis does not match pair issuer")
                if last.current_fiscal_period_end != pair.current.fiscal_period_end:
                    raise ValueError("last-grounded basis current period anchor mismatch")
                if last.current_available_at != pair.current.available_at:
                    raise ValueError("last-grounded basis current availability anchor mismatch")
                if last.staleness_limit_days != rules.last_grounded_staleness_days:
                    raise ValueError("last-grounded basis does not match the frozen staleness rule")
                replacement = build_deterministic_pit_axis_evidence(
                    previous=last.previous,
                    current=pair.current,
                    rules=rules,
                )[axis]
                if replacement.availability != SparseAxisAvailabilityV2.GROUNDED:
                    raise ValueError(
                        f"last-grounded basis cannot ground {pair.pair_id}/{axis.value}"
                    )
                candidate = replacement.model_copy(
                    update={
                        "previous_evidence_basis": (
                            PreviousEvidenceBasisV2.LAST_GROUNDED_WITHIN_STALENESS
                        ),
                        "prior_age_days": last.prior_age_days,
                        "staleness_limit_days": last.staleness_limit_days,
                        "applicability_rule_id": last.applicability_rule_id,
                    }
                )
                used_last.add((pair.pair_id, axis))
            if candidate is not None:
                deterministic_rows.append(
                    DeterministicAxisEvidenceInputV2(pair_id=pair.pair_id, evidence=candidate)
                )
                applicability = candidate.applicability
                rule_id = candidate.applicability_rule_id
            else:
                applicability = AxisApplicabilityV2.APPLICABLE
                rule_id = "PIT_UNIVERSAL_QUALITATIVE_AXIS_APPLICABILITY_V2"
            applicability_rows.append(
                AxisApplicabilityDecisionInputV2(
                    pair_id=pair.pair_id,
                    axis=axis,
                    applicability=applicability,
                    rule_id=rule_id,
                )
            )
    unused_last = set(last_by_key) - used_last
    if unused_last:
        raise ValueError(
            "last-grounded input was outside the pair universe or superseded by grounded PIT: "
            f"{sorted((pair_id, axis.value) for pair_id, axis in unused_last)[:5]}"
        )

    output.mkdir(parents=True, exist_ok=True)
    deterministic_path = output / "deterministic-axis-evidence.jsonl"
    applicability_path = output / "axis-applicability.jsonl"
    _write_jsonl(deterministic_path, deterministic_rows)
    _write_jsonl(applicability_path, applicability_rows)
    by_axis: dict[str, object] = {}
    for axis in (
        OperatingEvidenceAxis.MARGIN,
        OperatingEvidenceAxis.INVENTORY_MISMATCH,
        OperatingEvidenceAxis.BACKLOG,
        OperatingEvidenceAxis.CAPACITY_CAPEX,
    ):
        values = [row.evidence for row in deterministic_rows if row.evidence.axis == axis]
        directions = Counter(
            item.direction.value for item in values if item.direction is not None
        )
        reasons = Counter(
            item.abstention_reason.value
            for item in values
            if item.abstention_reason is not None
        )
        provenance = Counter(
            item.provenance.value for item in values if item.provenance is not None
        )
        previous_basis = Counter(
            item.previous_evidence_basis.value
            for item in values
            if item.previous_evidence_basis is not None
        )
        metrics = Counter(
            item.deterministic_metric_name
            for item in values
            if item.deterministic_metric_name is not None
        )
        signed_score_roles = Counter(item.signed_score_role.value for item in values)
        by_axis[axis.value] = {
            "applicable": sum(item.applicability == AxisApplicabilityV2.APPLICABLE for item in values),
            "grounded": sum(item.availability == SparseAxisAvailabilityV2.GROUNDED for item in values),
            "na": sum(item.availability == SparseAxisAvailabilityV2.NA for item in values),
            "not_applicable": sum(
                item.availability == SparseAxisAvailabilityV2.NOT_APPLICABLE for item in values
            ),
            "-1": directions[-1],
            "0": directions[0],
            "+1": directions[1],
            "stale_previous_state": reasons["STALE_PRIOR_STATE"],
            "deterministic_extraction_failure": reasons["TABLE_EXTRACTION_FAIL"],
            "reason_distribution": dict(sorted(reasons.items())),
            "source_type_distribution": dict(sorted(provenance.items())),
            "previous_evidence_basis_distribution": dict(sorted(previous_basis.items())),
            "deterministic_metric_distribution": dict(sorted(metrics.items())),
            "signed_score_role_distribution": dict(sorted(signed_score_roles.items())),
            "primary_signed_score_included": axis != OperatingEvidenceAxis.CAPACITY_CAPEX,
        }
    report = {
        "schema_version": "moatrader-historical-deterministic-pit-coverage-v2/1",
        "pair_count": len(pairs),
        "by_axis": by_axis,
        "evidence_priority": [
            "DETERMINISTIC_NUMERIC",
            "STRUCTURED_TABLE",
            "LLM_NARRATIVE",
        ],
        "score_averaging_across_source_types": False,
        "capacity_signed_score_policy": "RAW_DIRECTION_ONLY_NOT_IN_PRIMARY_SIGNED_SCORE",
        "outcome_vault_opened": False,
        "return_data_opened": False,
    }
    report_path = output / "deterministic-pit-coverage-report.json"
    _write_json(report_path, report)
    status = {
        "schema_version": "moatrader-historical-deterministic-pit-stage-v2/1",
        "status": "DETERMINISTIC_PIT_EVIDENCE_COMPLETE_OUTCOME_BLIND",
        "pair_count": len(pairs),
        "deterministic_or_last_grounded_row_count": len(deterministic_rows),
        "applicability_row_count": len(applicability_rows),
        "last_grounded_row_count": len(used_last),
        "last_grounded_row_count_by_axis": dict(
            sorted(Counter(axis.value for _, axis in used_last).items())
        ),
        "last_grounded_staleness_days": rules.last_grounded_staleness_days,
        "last_grounded_role": "PREVIOUS_COMPARISON_BASIS_ONLY_NEVER_CURRENT_EVIDENCE",
        "current_evidence_carry_forward": False,
        "evidence_priority": [
            "DETERMINISTIC_NUMERIC",
            "STRUCTURED_TABLE",
            "LLM_NARRATIVE",
        ],
        "deterministic_priority_axes": [
            OperatingEvidenceAxis.MARGIN.value,
            OperatingEvidenceAxis.INVENTORY_MISMATCH.value,
            OperatingEvidenceAxis.BACKLOG.value,
            OperatingEvidenceAxis.CAPACITY_CAPEX.value,
        ],
        "qualitative_primary_axes": [
            OperatingEvidenceAxis.DEMAND.value,
            OperatingEvidenceAxis.PRICE_MIX.value,
        ],
        "rules_sha256": sha256_file(rules_input),
        "rules_contract_sha256": canonical_payload_sha256(rules.model_dump(mode="json")),
        "pit_pair_input_sha256": sha256_file(pit_pair_input),
        "last_grounded_input_sha256": (
            sha256_file(last_grounded_input) if last_grounded_input is not None else None
        ),
        "deterministic_evidence_sha256": sha256_file(deterministic_path),
        "applicability_sha256": sha256_file(applicability_path),
        "coverage_report_sha256": sha256_file(report_path),
        "outcome_vault_opened": False,
        "return_data_opened": False,
        "per_pbr_role": "NOT_USED",
    }
    _write_json(output / "stage-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build outcome-blind deterministic PIT evidence and six-axis applicability; "
            "last-grounded state is allowed only inside the frozen staleness window."
        )
    )
    parser.add_argument("--pit-pair-input", type=Path, required=True)
    parser.add_argument("--rules-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--last-grounded-input", type=Path)
    args = parser.parse_args()
    result = build_pit_evidence(
        pit_pair_input=args.pit_pair_input,
        rules_input=args.rules_input,
        output=args.output,
        last_grounded_input=args.last_grounded_input,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
