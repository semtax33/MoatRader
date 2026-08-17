from __future__ import annotations

import math
from enum import StrEnum

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class ExpectationScoreStatus(StrEnum):
    VALID = "VALID"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    HIGH_FRAGILITY = "HIGH_FRAGILITY"
    MODEL_NOT_APPLICABLE = "MODEL_NOT_APPLICABLE"


class ThreeAxisPercentiles(ContractModel):
    expectation_gap: float = Field(ge=0, le=100)
    probable_mos: float = Field(ge=0, le=100)
    plausible_mos: float = Field(ge=0, le=100)
    probable_value_revision: float | None = Field(default=None, ge=0, le=100)
    plausible_value_revision: float | None = Field(default=None, ge=0, le=100)
    driver_breadth: float | None = Field(default=None, ge=0, le=100)
    evidence_revision: float | None = Field(default=None, ge=0, le=100)


class FragilityComponents(ContractModel):
    wacc_sensitivity: float = Field(ge=0, le=100)
    terminal_growth_sensitivity: float = Field(ge=0, le=100)
    scenario_dispersion: float = Field(ge=0, le=100)
    terminal_value_share: float = Field(ge=0, le=100)
    single_driver_dependence: float = Field(ge=0, le=100)
    evidence_weakness: float = Field(ge=0, le=100)

    def score(self) -> float:
        return (
            0.20 * self.wacc_sensitivity
            + 0.15 * self.terminal_growth_sensitivity
            + 0.20 * self.scenario_dispersion
            + 0.15 * self.terminal_value_share
            + 0.15 * self.single_driver_dependence
            + 0.15 * self.evidence_weakness
        )


class ExpectationThreeAxisScore(ContractModel):
    """Legacy weighted composite retained for benchmark diagnostics only."""

    cheap: float = Field(ge=0, le=100)
    improving: float | None = Field(default=None, ge=0, le=100)
    non_fragile: float = Field(ge=0, le=100)
    composite: float | None = Field(default=None, ge=0, le=100)
    status: ExpectationScoreStatus
    rank_eligible: bool = False
    diagnostic_only: bool = True

    @model_validator(mode="after")
    def consistent_status(self) -> "ExpectationThreeAxisScore":
        if not self.diagnostic_only:
            raise ValueError("weighted composite is permanently diagnostic-only")
        if self.rank_eligible:
            raise ValueError("weighted composite cannot be used as a production rank")
        return self


def weighted_geometric_score(cheap: float, improving: float, non_fragile: float) -> float:
    """Frozen, return-blind 40/35/25 expectation score."""

    if any(not 0 <= value <= 100 for value in (cheap, improving, non_fragile)):
        raise ValueError("axis scores must be inside [0, 100]")
    if 0 in (cheap, improving, non_fragile):
        return 0.0
    return 100.0 * (cheap / 100.0) ** 0.40 * (improving / 100.0) ** 0.35 * (
        non_fragile / 100.0
    ) ** 0.25


def build_three_axis_score(
    percentiles: ThreeAxisPercentiles,
    fragility: FragilityComponents,
    *,
    model_applicable: bool = True,
    non_fragile_gate: float = 30.0,
) -> ExpectationThreeAxisScore:
    cheap = (
        0.50 * percentiles.expectation_gap
        + 0.30 * percentiles.probable_mos
        + 0.20 * percentiles.plausible_mos
    )
    non_fragile = max(0.0, min(100.0, 100.0 - fragility.score()))
    improving_fields = (
        percentiles.probable_value_revision,
        percentiles.plausible_value_revision,
        percentiles.driver_breadth,
        percentiles.evidence_revision,
    )
    if not model_applicable:
        return ExpectationThreeAxisScore(
            cheap=cheap,
            non_fragile=non_fragile,
            status=ExpectationScoreStatus.MODEL_NOT_APPLICABLE,
        )
    if any(value is None or not math.isfinite(value) for value in improving_fields):
        return ExpectationThreeAxisScore(
            cheap=cheap,
            non_fragile=non_fragile,
            status=ExpectationScoreStatus.INSUFFICIENT_EVIDENCE,
        )
    probable_revision, plausible_revision, breadth, evidence_revision = (
        float(value) for value in improving_fields if value is not None
    )
    improving = (
        0.35 * probable_revision
        + 0.25 * plausible_revision
        + 0.25 * breadth
        + 0.15 * evidence_revision
    )
    composite = weighted_geometric_score(cheap, improving, non_fragile)
    if non_fragile < non_fragile_gate:
        return ExpectationThreeAxisScore(
            cheap=cheap,
            improving=improving,
            non_fragile=non_fragile,
            composite=composite,
            status=ExpectationScoreStatus.HIGH_FRAGILITY,
        )
    return ExpectationThreeAxisScore(
        cheap=cheap,
        improving=improving,
        non_fragile=non_fragile,
        composite=composite,
        status=ExpectationScoreStatus.VALID,
    )


def average_tie_percentiles(values: list[float]) -> list[float]:
    """Return 0..100 percentile ranks with average ranks for exact ties."""

    if not values:
        return []
    if any(not math.isfinite(value) for value in values):
        raise ValueError("percentile input must be finite")
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[position]]:
            end += 1
        average_rank = (position + end - 1) / 2.0
        percentile = 50.0 if len(values) == 1 else 100.0 * average_rank / (len(values) - 1)
        for offset in range(position, end):
            ranks[ordered[offset]] = percentile
        position = end
    return ranks
