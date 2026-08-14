from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal

from pydantic import Field

from moatrader.canonical.models import (
    CanonicalDocumentBundle,
    ConsolidationScope,
    ContractModel,
    StructuredFact,
)


CANONICAL_CONCEPT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("REVENUE", re.compile(r"revenue|sales", re.I)),
    ("EBIT", re.compile(r"operatingincome|operatingprofit|profitlossfromoperatingactivities|영업이익", re.I)),
    ("NET_INCOME", re.compile(r"netincome|profitloss(?!fromoperating)|당기순이익", re.I)),
    ("CFO", re.compile(r"netcash.*operating|cashflowsfromusedinoperating|영업활동.*현금", re.I)),
    ("CAPEX", re.compile(r"paymentstoacquire.*property|purchaseofproperty|capitalexpenditure|유형자산.*취득", re.I)),
    ("CASH", re.compile(r"cashandcashequivalents|현금및현금성", re.I)),
    ("TOTAL_DEBT", re.compile(r"longtermdebt|shorttermborrowings|차입금", re.I)),
    ("DILUTED_SHARES", re.compile(r"weightedaveragenumberofdilutedshares|희석.*주식", re.I)),
]


def canonicalize_concept(fact: StructuredFact) -> str | None:
    if fact.canonical_concept:
        return fact.canonical_concept.upper()
    local = fact.concept.rsplit(":", 1)[-1].rsplit("_", 1)[-1]
    compact = re.sub(r"[^0-9A-Za-z가-힣]", "", local)
    for canonical, pattern in CANONICAL_CONCEPT_PATTERNS:
        if pattern.search(compact):
            return canonical
    return None


def _period_date(fact: StructuredFact) -> date | None:
    return fact.period.instant or fact.period.end


class FinancialPoint(ContractModel):
    period: date
    value: Decimal
    unit: str | None = None
    source_fact_ids: list[str] = Field(min_length=1)
    available_at: datetime


class FinancialSeries(ContractModel):
    concept: str
    points: list[FinancialPoint] = Field(default_factory=list)


class DerivedMetric(ContractModel):
    name: str
    period: date | None = None
    value: Decimal
    unit: str
    derived_from_fact_ids: list[str] = Field(min_length=1)


class FinancialSnapshot(ContractModel):
    as_of: datetime
    issuer_id: str | None = None
    issuer_name: str | None = None
    series: list[FinancialSeries] = Field(default_factory=list)
    derived_metrics: list[DerivedMetric] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def series_index(self) -> dict[str, FinancialSeries]:
        return {series.concept: series for series in self.series}

    def to_markdown(self) -> str:
        series_index = self.series_index()
        periods = sorted({point.period for series in self.series for point in series.points})
        if not periods:
            return "## Financial Snapshot\n\n_No canonical numeric facts available._"
        point_index = {
            (series.concept, point.period): point
            for series in self.series
            for point in series.points
        }
        lines = [
            "## Financial Snapshot",
            "",
            f"- As of: {self.as_of.isoformat()}",
            "- Values come from StructuredFact; calculations are deterministic.",
            "",
            "| Metric | " + " | ".join(period.isoformat() for period in periods) + " |",
            "|---|" + "---:|" * len(periods),
        ]
        for concept in sorted(series_index):
            values = []
            for period in periods:
                point = point_index.get((concept, period))
                values.append(f"{point.value:,}" if point else "")
            lines.append(f"| {concept} | " + " | ".join(values) + " |")
        if self.derived_metrics:
            lines.extend(["", "### Derived Metrics", "", "| Metric | Period | Value | Unit | Sources |", "|---|---|---:|---|---|"])
            for metric in self.derived_metrics:
                lines.append(
                    f"| {metric.name} | {metric.period.isoformat() if metric.period else 'multi-period'} | "
                    f"{metric.value:,} | {metric.unit} | {', '.join(metric.derived_from_fact_ids)} |"
                )
        if self.warnings:
            lines.extend(["", "### Data Warnings", ""])
            lines.extend(f"- {warning}" for warning in self.warnings)
        return "\n".join(lines)


class FinancialSnapshotBuilder:
    """Build point-in-time series and derived metrics without asking an LLM to do arithmetic."""

    def build(
        self,
        bundles: Iterable[CanonicalDocumentBundle],
        *,
        as_of: datetime,
    ) -> FinancialSnapshot:
        if as_of.tzinfo is None or as_of.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        visible = [bundle for bundle in bundles if bundle.metadata.available_at <= as_of]
        if not visible:
            return FinancialSnapshot(as_of=as_of, warnings=["no document was available at the cutoff"])
        issuer_id = visible[0].metadata.issuer_id
        issuer_name = visible[0].metadata.issuer_name
        candidates: dict[tuple[str, date], list[StructuredFact]] = defaultdict(list)
        for bundle in visible:
            for fact in bundle.facts:
                if fact.numeric_value is None or fact.available_at > as_of:
                    continue
                concept = canonicalize_concept(fact)
                period = _period_date(fact)
                if concept and period:
                    candidates[(concept, period)].append(fact)

        chosen: dict[tuple[str, date], StructuredFact] = {}
        warnings: list[str] = []
        for key, facts in candidates.items():
            facts.sort(
                key=lambda fact: (
                    fact.scope == ConsolidationScope.CONSOLIDATED,
                    not fact.dimensions,
                    fact.is_restated,
                    fact.available_at,
                ),
                reverse=True,
            )
            chosen[key] = facts[0]
            if len({fact.numeric_value for fact in facts}) > 1:
                warnings.append(f"conflicting values for {key[0]} at {key[1]}; selected latest preferred context")

        grouped: dict[str, list[FinancialPoint]] = defaultdict(list)
        for (concept, period), fact in chosen.items():
            unit = fact.unit.canonical if fact.unit and fact.unit.canonical else (fact.unit.raw if fact.unit else None)
            grouped[concept].append(
                FinancialPoint(
                    period=period,
                    value=fact.numeric_value or Decimal(0),
                    unit=unit,
                    source_fact_ids=[fact.fact_id],
                    available_at=fact.available_at,
                )
            )
        series = [
            FinancialSeries(concept=concept, points=sorted(points, key=lambda item: item.period))
            for concept, points in sorted(grouped.items())
        ]
        derived = self._derive(series)
        return FinancialSnapshot(
            as_of=as_of,
            issuer_id=issuer_id,
            issuer_name=issuer_name,
            series=series,
            derived_metrics=derived,
            warnings=warnings,
        )

    def _derive(self, series: list[FinancialSeries]) -> list[DerivedMetric]:
        index = {
            (item.concept, point.period): point
            for item in series
            for point in item.points
        }
        periods = sorted({period for _concept, period in index})
        result: list[DerivedMetric] = []
        for period in periods:
            revenue = index.get(("REVENUE", period))
            ebit = index.get(("EBIT", period))
            cfo = index.get(("CFO", period))
            capex = index.get(("CAPEX", period))
            if revenue and ebit and revenue.value:
                result.append(
                    DerivedMetric(
                        name="EBIT_MARGIN",
                        period=period,
                        value=ebit.value / revenue.value,
                        unit="RATIO",
                        derived_from_fact_ids=[*revenue.source_fact_ids, *ebit.source_fact_ids],
                    )
                )
            if cfo and capex:
                fcf = cfo.value - abs(capex.value)
                source_ids = [*cfo.source_fact_ids, *capex.source_fact_ids]
                result.append(
                    DerivedMetric(name="FCF", period=period, value=fcf, unit=cfo.unit or "UNKNOWN", derived_from_fact_ids=source_ids)
                )
                if revenue and revenue.value:
                    result.append(
                        DerivedMetric(
                            name="FCF_MARGIN",
                            period=period,
                            value=fcf / revenue.value,
                            unit="RATIO",
                            derived_from_fact_ids=[*source_ids, *revenue.source_fact_ids],
                        )
                    )
        revenues = next((item for item in series if item.concept == "REVENUE"), None)
        if revenues and len(revenues.points) >= 2:
            first, last = revenues.points[0], revenues.points[-1]
            years = max(1.0, (last.period - first.period).days / 365.25)
            if first.value > 0 and last.value > 0:
                cagr = (last.value / first.value) ** (Decimal(1) / Decimal(str(years))) - Decimal(1)
                result.append(
                    DerivedMetric(
                        name="REVENUE_CAGR",
                        value=cagr,
                        unit="RATIO",
                        derived_from_fact_ids=[*first.source_fact_ids, *last.source_fact_ids],
                    )
                )
        return result

