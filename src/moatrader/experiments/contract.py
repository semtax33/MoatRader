from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class FrozenExpectationGapContract(ContractModel):
    schema_version: str = "expectation-gap-production-candidate/1"
    frozen_on: date
    development_dates: list[date] = Field(min_length=1)
    holdout_dates: list[date] = Field(min_length=1)
    universe_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    universe_count: int = Field(gt=0)
    valuation_methods: list[str] = Field(min_length=1)
    router_contract_version: str
    cheap_definition: str
    percentile_cohort: str
    risk_policy: dict[str, object]
    legacy_composite_role: str
    improving_role: str
    sector_policy: str
    source_cutoff_policy: str
    signal_seal_required: bool
    return_inputs_forbidden_before_signal_seal: bool
    forward_return_calendar_days: int = Field(gt=0)
    maximum_sector_neutral_ic_sacrifice: float = Field(ge=0)
    minimum_worst_decile_improvement: float = Field(ge=0)
    minimum_downside_capture_improvement: float = Field(ge=0)
    frozen_source_sha256: dict[str, str]
    engineering_stability_sha256: dict[str, str]
    engineering_return_data_accessed: bool
    contract_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def no_development_holdout_overlap(self) -> "FrozenExpectationGapContract":
        if set(self.development_dates) & set(self.holdout_dates):
            raise ValueError("holdout dates must not overlap development dates")
        if not self.signal_seal_required or not self.return_inputs_forbidden_before_signal_seal:
            raise ValueError("holdout contract must seal signals before return access")
        if self.engineering_return_data_accessed:
            raise ValueError("engineering stability must not access return data")
        return self


def compute_contract_sha256(payload: dict[str, object]) -> str:
    canonical = dict(payload)
    canonical.pop("contract_sha256", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def verify_frozen_sources(
    contract: FrozenExpectationGapContract,
    *,
    repository_root: Path,
) -> None:
    mismatches: list[str] = []
    for relative, expected in contract.frozen_source_sha256.items():
        path = repository_root / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"
        if actual != expected:
            mismatches.append(f"{relative}: expected {expected}, got {actual}")
    if mismatches:
        raise ValueError("frozen source verification failed: " + "; ".join(mismatches))
