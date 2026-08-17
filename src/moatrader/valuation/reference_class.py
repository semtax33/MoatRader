from __future__ import annotations

from datetime import datetime
from decimal import Decimal

from pydantic import Field, model_validator

from moatrader.canonical.models import ContractModel


class DecimalRange(ContractModel):
    low: Decimal
    high: Decimal

    @model_validator(mode="after")
    def ordered(self) -> "DecimalRange":
        if self.high < self.low:
            raise ValueError("range high must be greater than or equal to low")
        return self

    def contains(self, value: Decimal) -> bool:
        return self.low <= value <= self.high


class IntegerRange(ContractModel):
    low: int
    high: int

    @model_validator(mode="after")
    def ordered(self) -> "IntegerRange":
        if self.high < self.low:
            raise ValueError("range high must be greater than or equal to low")
        return self

    def contains(self, value: int) -> bool:
        return self.low <= value <= self.high


class PlausibilityReferenceClass(ContractModel):
    name: str = Field(min_length=1)
    as_of: datetime
    source_refs: list[str] = Field(min_length=1)
    revenue_growth: DecimalRange | None = None
    nopat_margin: DecimalRange | None = None
    roiic: DecimalRange | None = None
    sales_to_capital: DecimalRange | None = None
    cap_years: IntegerRange | None = None
    stable_growth: DecimalRange | None = None

    @model_validator(mode="after")
    def pit_safe(self) -> "PlausibilityReferenceClass":
        if self.as_of.tzinfo is None or self.as_of.utcoffset() is None:
            raise ValueError("reference-class as_of must be timezone-aware")
        return self
