from __future__ import annotations

import re
from calendar import monthrange
from collections import defaultdict
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal

from pydantic import Field

from moatrader.canonical.ids import stable_id
from moatrader.canonical.models import (
    CanonicalDocumentBundle,
    ConsolidationScope,
    ContractModel,
    StructuredFact,
    TableNode,
    UnitSpec,
)


CANONICAL_CONCEPT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "REVENUE",
        re.compile(
            r"^(?:revenue|revenues|salesrevenue|sales|"
            r"revenuefromcontractwithcustomer.*)$",
            re.I,
        ),
    ),
    (
        "EBIT",
        re.compile(
            r"^(?:operatingincome(?:loss)?|operatingprofit(?:loss)?|"
            r"profitlossfromoperatingactivities|영업이익)$",
            re.I,
        ),
    ),
    (
        "NET_INCOME",
        re.compile(
            r"^(?:netincome(?:loss)?|profitloss|profitlossfortheperiod|"
            r"netprofitloss|당기순이익|분기순이익|반기순이익)$",
            re.I,
        ),
    ),
    (
        "CFO",
        re.compile(
            r"^(?:netcashprovidedbyusedinoperatingactivities(?:continuingoperations)?|"
            r"cashflowsfromusedinoperatingactivities(?:continuingoperations)?|영업활동현금흐름)$",
            re.I,
        ),
    ),
    (
        "CAPEX",
        re.compile(
            r"^(?:paymentstoacquirepropertyplantandequipment|"
            r"purchaseofpropertyplantandequipment(?:classifiedasinvestingactivities)?|"
            r"capitalexpenditures?|유형자산의취득)$",
            re.I,
        ),
    ),
    (
        "CASH",
        re.compile(
            r"^(?:cashandcashequivalents|cashandcashequivalentsatendofperiodcf|"
            r"현금및현금성자산|기말현금및현금성자산)$",
            re.I,
        ),
    ),
    (
        "TOTAL_DEBT",
        re.compile(
            r"^(?:totalborrowings|borrowings|borrowingsanddebentures|총차입금)$",
            re.I,
        ),
    ),
    (
        "SHORT_TERM_DEBT",
        re.compile(
            r"^(?:shorttermborrowings|currentborrowings|currentportionoflongtermdebt|"
            r"currentportionofdebentures|단기차입금|유동성장기차입금|유동성사채)$",
            re.I,
        ),
    ),
    (
        "LONG_TERM_DEBT",
        re.compile(r"^(?:longtermdebt|longtermborrowings|debentures|장기차입금|사채)$", re.I),
    ),
    (
        "LEASE_LIABILITIES",
        re.compile(r"^(?:current|noncurrent)?leaseliabilit(?:y|ies)|리스부채$", re.I),
    ),
    (
        "CONVERTIBLE_LIABILITIES",
        re.compile(r"^(?:convertiblebonds?|convertibledebts?|전환사채)$", re.I),
    ),
    (
        "PREFERRED_LIABILITIES",
        re.compile(r"^(?:redeemablepreferredshares?|preferredshareliabilit(?:y|ies)|상환전환우선주)$", re.I),
    ),
    (
        "DEPRECIATION_AMORTIZATION",
        re.compile(
            r"^(?:depreciation(?:andamorti[sz]ation)?|amorti[sz]ation|"
            r"depreciationandamortisationexpense|감가상각비|무형자산상각비)$",
            re.I,
        ),
    ),
    (
        "R_AND_D_EXPENSE",
        re.compile(r"^(?:researchanddevelopmentexpense|researchdevelopmentexpense|연구개발비)$", re.I),
    ),
    (
        "RECEIVABLES",
        re.compile(r"^(?:tradeandothercurrentreceivables|tradereceivables|accountsreceivable|매출채권)$", re.I),
    ),
    ("INVENTORY", re.compile(r"^(?:inventories|inventory|재고자산)$", re.I)),
    (
        "PAYABLES",
        re.compile(r"^(?:tradeandothercurrentpayables|tradepayables|accountspayable|매입채무)$", re.I),
    ),
    ("TOTAL_ASSETS", re.compile(r"^(?:assets|totalassets|자산총계)$", re.I)),
    ("TOTAL_LIABILITIES", re.compile(r"^(?:liabilities|totalliabilities|부채총계)$", re.I)),
    ("TOTAL_EQUITY", re.compile(r"^(?:equity|totalequity|자본총계)$", re.I)),
    (
        "PPE",
        re.compile(r"^(?:propertyplantandequipment|propertyplantandequipmentnet|유형자산)$", re.I),
    ),
    (
        "DILUTED_SHARES",
        re.compile(
            r"^(?:weightedaveragenumberofdilutedshares(?:outstanding)?|희석.*주식)$",
            re.I,
        ),
    ),
]

_CAPEX_AGGREGATE_LABEL_RE = re.compile(
    r"^(?:\uc720\ud615\uc790\uc0b0\uc758?\ucde8\ub4dd|\uc720\ud615\uc790\uc0b0\uc758?\uc99d\uac00)(?:\uc8fc\d+)?$"
)
_CAPEX_COMPONENT_CONCEPT_RE = re.compile(
    r"^(?:purchaseof|paymentstoacquire)(?:"
    r"land|buildings?|machinery|structures?|vehicles?|"
    r"officeequipment|computerequipment|fixturesandfittings|"
    r"otherpropertyplantandequipment|constructioninprogress"
    r")$",
    re.I,
)
_CAPEX_COMPONENT_LABEL_RE = re.compile(
    r"^(?:"
    r"\ud1a0\uc9c0|\uac74\ubb3c|\uae30\uacc4\uc7a5\uce58|\uad6c\ucd95\ubb3c|\ucc28\ub7c9\uc6b4\ubc18\uad6c|"
    r"\ube44\ud488|\uc0ac\ubb34\uc6a9\ube44\ud488|\uacf5\uad6c\uc640\uae30\uad6c|\uc2dc\uc124\uc7a5\uce58|"
    r"\uae30\ud0c0\uc720\ud615\uc790\uc0b0|\uac74\uc124\uc911\uc778\uc790\uc0b0"
    r")\uc758?(?:\ucde8\ub4dd|\uc99d\uac00)(?:\uc8fc\d+)?$"
)


def _compact_identifier(value: str | None) -> str:
    return re.sub(r"[^0-9A-Za-z\uac00-\ud7a3]", "", value or "")


def _local_concept(concept: str) -> str:
    local = concept.rsplit(":", 1)[-1]
    return local.split("_", 1)[1] if "_" in local else local


def canonicalize_concept(fact: StructuredFact) -> str | None:
    if fact.canonical_concept:
        return fact.canonical_concept.upper()
    compact = _compact_identifier(_local_concept(fact.concept))
    for canonical, pattern in CANONICAL_CONCEPT_PATTERNS:
        if pattern.search(compact):
            return canonical
    if _CAPEX_AGGREGATE_LABEL_RE.fullmatch(_compact_identifier(fact.label)):
        return "CAPEX"
    return None


def _capex_component_identity(fact: StructuredFact) -> str | None:
    local = _compact_identifier(_local_concept(fact.concept))
    if _CAPEX_COMPONENT_CONCEPT_RE.fullmatch(local):
        return local.lower()
    label = _compact_identifier(fact.label)
    if _CAPEX_COMPONENT_LABEL_RE.fullmatch(label):
        return f"label:{label}"
    return None


def _period_date(fact: StructuredFact) -> date | None:
    return fact.period.instant or fact.period.end


def _fact_period_basis(fact: StructuredFact) -> str | None:
    if fact.period.fiscal_period:
        return fact.period.fiscal_period.upper()
    if fact.period.start and fact.period.end:
        days = (fact.period.end - fact.period.start).days
        if days >= 300:
            return "FY"
        if days <= 100:
            return "Q"
        if days <= 200:
            return "H1"
        return "Q3"
    return "INSTANT" if fact.period.instant else None


def _fact_preference(fact: StructuredFact) -> tuple[int, bool, bool, int, bool, datetime]:
    return (
        {
            ConsolidationScope.CONSOLIDATED: 2,
            ConsolidationScope.SEPARATE: 1,
        }.get(fact.scope, 0),
        not fact.dimensions,
        fact.unit is not None and fact.unit.scale == Decimal("1"),
        (
            (fact.period.end - fact.period.start).days
            if fact.period.start and fact.period.end
            else 0
        ),
        fact.is_restated,
        fact.available_at,
    )


def _normalized_value_and_unit(fact: StructuredFact) -> tuple[Decimal, str | None]:
    unit = fact.unit.canonical if fact.unit and fact.unit.canonical else (fact.unit.raw if fact.unit else None)
    value = fact.numeric_value or Decimal(0)
    if fact.unit and fact.unit.scale != Decimal("1") and fact.unit.currency:
        value *= fact.unit.scale
        unit = fact.unit.currency
    return value, unit


_TABLE_REVENUE_HEADER_RE = re.compile(r"(?:매출(?:액)?|revenue|sales)", re.I)
_TABLE_REVENUE_RATIO_RE = re.compile(r"매출(?:액)?\s*(?:대비|비율)|(?:revenue|sales)\s*(?:ratio|percent)", re.I)
_TABLE_SALES_CONTEXT_RE = re.compile(r"주요\s*제품|매출\s*현황|매출\s*및\s*수주|sales\s+by", re.I)
_TABLE_AMOUNT_HEADER_RE = re.compile(r"(?:^|[>\s])(?:금액|매출액|amount)$", re.I)
_TABLE_TOTAL_LABEL_RE = re.compile(r"^(?:합계|총계|계|total)$", re.I)
_SUMMARY_FINANCIAL_ROW_MAP = {
    "매출액": "REVENUE",
    "영업이익": "EBIT",
    "영업이익손실": "EBIT",
    "당기순이익": "NET_INCOME",
    "재고자산": "INVENTORY",
    "유형자산": "PPE",
    "자산총계": "TOTAL_ASSETS",
    "부채총계": "TOTAL_LIABILITIES",
    "자본총계": "TOTAL_EQUITY",
}


def _summary_financial_concept(label: str) -> str | None:
    compact = re.sub(r"[^0-9A-Za-z가-힣]", "", label)
    return _SUMMARY_FINANCIAL_ROW_MAP.get(compact)


def _table_column_period(table: TableNode, header_text: str) -> date | None:
    explicit_dates = [
        date(int(year), int(month), int(day))
        for year, month, day in re.findall(
            r"(?<!\d)(20\d{2})\s*(?:년|[.\-/])\s*(\d{1,2})\s*(?:월|[.\-/])\s*(\d{1,2})",
            header_text,
        )
    ]
    if explicit_dates:
        return max(explicit_dates)
    year_months = [
        (int(year), int(month))
        for year, month in re.findall(
            r"(?<!\d)(20\d{2})\s*년\s*(\d{1,2})\s*월(?:말)?",
            header_text,
        )
    ]
    if year_months:
        year, month = max(year_months)
        return date(year, month, monthrange(year, month)[1])
    years = [int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", header_text)]
    year = max(years) if years else (table.period.fiscal_year if table.period else None)
    if year is None:
        return None
    text = header_text.upper()
    if re.search(r"(?:1\s*분기|Q1|1Q)", text):
        return date(year, 3, 31)
    if re.search(r"(?:반기|H1)", text):
        return date(year, 6, 30)
    if re.search(r"(?:3\s*분기|Q3|3Q)", text):
        return date(year, 9, 30)
    return date(year, 12, 31)


def _table_column_basis(header_text: str) -> str:
    text = header_text.upper()
    if re.search(r"(?:1\s*분기|Q1|1Q)", text):
        return "Q1"
    if re.search(r"(?:반기|H1)", text):
        return "H1"
    if re.search(r"(?:3\s*분기|Q3|3Q)", text):
        return "Q3"
    return "FY"


def _table_value(value: Decimal, table: TableNode) -> tuple[Decimal, str | None]:
    return _value_with_unit(value, table.unit)


def _value_with_unit(value: Decimal, unit: UnitSpec | None) -> tuple[Decimal, str | None]:
    if unit and unit.currency:
        return value * unit.scale, unit.currency
    return value, unit.canonical if unit and unit.canonical else (unit.raw if unit else None)


class FinancialPoint(ContractModel):
    period: date
    period_basis: str | None = None
    value: Decimal
    unit: str | None = None
    source_fact_ids: list[str] = Field(min_length=1)
    available_at: datetime


class FinancialSeries(ContractModel):
    concept: str
    points: list[FinancialPoint] = Field(default_factory=list)


class FinancialBreakdownSeries(ContractModel):
    concept: str
    dimension: str
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
    breakdowns: list[FinancialBreakdownSeries] = Field(default_factory=list)
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
        if self.breakdowns:
            lines.extend(
                [
                    "",
                    "### Segment / Dimension Breakdowns",
                    "",
                    "| Metric | Dimension | Period | Value | Unit | Sources |",
                    "|---|---|---|---:|---|---|",
                ]
            )
            for breakdown in self.breakdowns:
                for point in breakdown.points:
                    lines.append(
                        f"| {breakdown.concept} | {breakdown.dimension} | {point.period.isoformat()} | "
                        f"{point.value:,} | {point.unit or 'UNKNOWN'} | {', '.join(point.source_fact_ids)} |"
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
        breakdown_candidates: dict[tuple[str, str, date], list[StructuredFact]] = defaultdict(list)
        capex_components: dict[date, list[tuple[str, StructuredFact]]] = defaultdict(list)
        for bundle in visible:
            for fact in bundle.facts:
                if fact.numeric_value is None or fact.available_at > as_of:
                    continue
                concept = canonicalize_concept(fact)
                period = _period_date(fact)
                if concept and period:
                    candidates[(concept, period)].append(fact)
                    material_dimensions = [
                        item
                        for item in fact.dimensions
                        if not item.member.endswith("ReportedAmountMember")
                    ]
                    if material_dimensions and concept in {"REVENUE", "EBIT", "CAPEX"}:
                        dimension = " | ".join(
                            f"{item.axis}={item.member}" for item in material_dimensions
                        )
                        breakdown_candidates[(concept, dimension, period)].append(fact)
                elif period:
                    component_identity = _capex_component_identity(fact)
                    if component_identity:
                        capex_components[period].append((component_identity, fact))

        chosen: dict[tuple[str, date], StructuredFact] = {}
        warnings: list[str] = []
        for key, facts in candidates.items():
            facts.sort(key=_fact_preference, reverse=True)
            chosen[key] = facts[0]
            if len({fact.numeric_value for fact in facts}) > 1:
                warnings.append(f"conflicting values for {key[0]} at {key[1]}; selected latest preferred context")

        grouped: dict[str, list[FinancialPoint]] = defaultdict(list)
        for (concept, period), fact in chosen.items():
            value, unit = _normalized_value_and_unit(fact)
            grouped[concept].append(
                FinancialPoint(
                    period=period,
                    period_basis=_fact_period_basis(fact),
                    value=value,
                    unit=unit,
                    source_fact_ids=[fact.fact_id],
                    available_at=fact.available_at,
                )
            )
        self._add_component_capex_fallback(
            grouped,
            chosen,
            capex_components,
            warnings,
        )
        self._add_summary_financial_table_fallback(visible, grouped, warnings)
        table_breakdowns = self._add_table_revenue_fallback(
            visible,
            grouped,
            warnings,
        )
        self._add_total_debt_fallback(grouped, warnings)
        series = [
            FinancialSeries(concept=concept, points=sorted(points, key=lambda item: item.period))
            for concept, points in sorted(grouped.items())
        ]
        breakdowns: list[FinancialBreakdownSeries] = []
        for (concept, dimension, period), facts in sorted(breakdown_candidates.items()):
            facts.sort(key=_fact_preference, reverse=True)
            fact = facts[0]
            value, unit = _normalized_value_and_unit(fact)
            breakdowns.append(
                FinancialBreakdownSeries(
                    concept=concept,
                    dimension=dimension,
                    points=[
                        FinancialPoint(
                            period=period,
                            period_basis=_fact_period_basis(fact),
                            value=value,
                            unit=unit,
                            source_fact_ids=[fact.fact_id],
                            available_at=fact.available_at,
                        )
                    ],
                )
            )
        breakdowns.extend(table_breakdowns)
        derived = self._derive(series)
        return FinancialSnapshot(
            as_of=as_of,
            issuer_id=issuer_id,
            issuer_name=issuer_name,
            series=series,
            breakdowns=breakdowns,
            derived_metrics=derived,
            warnings=warnings,
        )

    @staticmethod
    def _add_summary_financial_table_fallback(
        bundles: list[CanonicalDocumentBundle],
        grouped: dict[str, list[FinancialPoint]],
        warnings: list[str],
    ) -> None:
        """Fill audited comparative summary rows while preferring consolidated tables."""

        for bundle in bundles:
            unit_by_section: dict[tuple[str, ...], UnitSpec] = {}
            candidates: dict[str, list[tuple[Decimal, TableNode, list[tuple[str, date, str, Decimal, str | None, str]]]]] = defaultdict(list)
            for node in bundle.ast.walk():
                if not isinstance(node, TableNode) or "요약재무정보" not in " ".join(node.section_path):
                    continue
                section_key = tuple(node.section_path)
                if node.unit and node.unit.currency:
                    unit_by_section[section_key] = node.unit
                effective_unit = node.unit if node.unit and node.unit.currency else unit_by_section.get(section_key)
                if effective_unit is None:
                    continue
                values: list[tuple[str, date, str, Decimal, str | None, str]] = []
                families: set[str] = set()
                start_row = node.header_row_count or 1
                for row in node.rows[start_row:]:
                    if not row.cells:
                        continue
                    concept = _summary_financial_concept(row.cells[0].normalized_text)
                    if concept is None:
                        continue
                    families.add("IS" if concept in {"REVENUE", "EBIT", "NET_INCOME"} else "BS")
                    for header in node.column_headers[1:]:
                        if header.col >= len(row.cells):
                            continue
                        cell = row.cells[header.col]
                        if cell.numeric_value is None:
                            continue
                        header_text = " ".join(header.path)
                        period = _table_column_period(node, header_text)
                        if period is None:
                            continue
                        value, unit = _value_with_unit(cell.numeric_value, effective_unit)
                        values.append(
                            (
                                concept,
                                period,
                                _table_column_basis(header_text) if concept in {"REVENUE", "EBIT", "NET_INCOME"} else "INSTANT",
                                value,
                                unit,
                                stable_id("TF", node.node_id, row.index, header.col, concept),
                            )
                        )
                if not values or len(families) != 1:
                    continue
                comparisons: list[Decimal] = []
                existing = {
                    (concept, point.period): point
                    for concept, points in grouped.items()
                    for point in points
                }
                for concept, period, _basis, value, unit, _source in values:
                    point = existing.get((concept, period))
                    if point and point.unit == unit:
                        comparisons.append(abs(value - point.value) / max(abs(point.value), Decimal(1)))
                if not comparisons:
                    continue
                score = sum(comparisons, Decimal(0)) / Decimal(len(comparisons))
                candidates[next(iter(families))].append((score, node, values))

            for family, family_candidates in candidates.items():
                score, node, values = min(family_candidates, key=lambda item: (item[0], item[1].order))
                if score > Decimal("0.05"):
                    continue
                existing_keys = {
                    (concept, point.period)
                    for concept, points in grouped.items()
                    for point in points
                }
                added = 0
                for concept, period, basis, value, unit, source_id in values:
                    if (concept, period) in existing_keys:
                        continue
                    grouped[concept].append(
                        FinancialPoint(
                            period=period,
                            period_basis=basis,
                            value=value,
                            unit=unit,
                            source_fact_ids=[source_id],
                            available_at=bundle.metadata.available_at,
                        )
                    )
                    existing_keys.add((concept, period))
                    added += 1
                if added:
                    warnings.append(
                        f"{family} comparatives were inferred deterministically from audited summary table "
                        f"{node.node_id} (consolidated-match error {score:.4f})"
                    )

    @staticmethod
    def _add_table_revenue_fallback(
        bundles: list[CanonicalDocumentBundle],
        grouped: dict[str, list[FinancialPoint]],
        warnings: list[str],
    ) -> list[FinancialBreakdownSeries]:
        """Promote explicit revenue columns when a filing lacks XBRL/ACODE facts."""

        existing = {point.period for point in grouped.get("REVENUE", [])}
        breakdowns: list[FinancialBreakdownSeries] = []
        for bundle in bundles:
            for node in bundle.ast.walk():
                if not isinstance(node, TableNode):
                    continue
                if node.unit is None or node.unit.currency is None:
                    continue
                table_context = " ".join([node.caption or "", *node.section_path])
                sales_table = bool(_TABLE_SALES_CONTEXT_RE.search(table_context))
                for header in node.column_headers:
                    header_text = " ".join(header.path)
                    revenue_column = bool(_TABLE_REVENUE_HEADER_RE.search(header_text)) or (
                        sales_table and bool(_TABLE_AMOUNT_HEADER_RE.search(header_text))
                    )
                    if not revenue_column or _TABLE_REVENUE_RATIO_RE.search(header_text):
                        continue
                    period = _table_column_period(node, header_text)
                    if period is None:
                        continue
                    if period > bundle.metadata.available_at.date():
                        reporting = bundle.metadata.reporting_period
                        reported_period = reporting.instant if reporting else None
                        reported_period = reported_period or (reporting.end if reporting else None)
                        if reported_period is None or reported_period > bundle.metadata.available_at.date():
                            continue
                        period = reported_period
                    start_row = node.header_row_count or 1
                    values: list[tuple[str, Decimal, str]] = []
                    for row in node.rows[start_row:]:
                        if header.col >= len(row.cells):
                            continue
                        cell = row.cells[header.col]
                        if cell.numeric_value is None:
                            continue
                        labels = [
                            item.normalized_text
                            for item in row.cells[: header.col]
                            if item.normalized_text and item.numeric_value is None
                        ]
                        label = " / ".join(dict.fromkeys(labels)) or f"row-{row.index}"
                        source_id = stable_id("TF", node.node_id, row.index, header.col, label)
                        values.append((label, cell.numeric_value, source_id))
                    if not values:
                        continue
                    total_rows = [
                        item
                        for item in values
                        if any(
                            _TABLE_TOTAL_LABEL_RE.fullmatch(part.replace(" ", ""))
                            for part in item[0].split("/")
                        )
                    ]
                    selected = total_rows[-1:] if total_rows else values
                    normalized = [_table_value(value, node) for _label, value, _source in selected]
                    units = {unit for _value, unit in normalized}
                    if len(units) != 1:
                        continue
                    if period not in existing:
                        grouped["REVENUE"].append(
                            FinancialPoint(
                                period=period,
                                period_basis=_table_column_basis(header_text),
                                value=sum((value for value, _unit in normalized), Decimal(0)),
                                unit=next(iter(units)),
                                source_fact_ids=[source for _label, _value, source in selected],
                                available_at=bundle.metadata.available_at,
                            )
                        )
                        existing.add(period)
                        warnings.append(
                            f"REVENUE at {period} was inferred deterministically from table {node.node_id}; "
                            "no canonical XBRL fact was available"
                        )
                    for label, value, source_id in values:
                        normalized_value, unit = _table_value(value, node)
                        breakdowns.append(
                            FinancialBreakdownSeries(
                                concept="REVENUE",
                                dimension=label,
                                points=[
                                    FinancialPoint(
                                        period=period,
                                        period_basis=_table_column_basis(header_text),
                                        value=normalized_value,
                                        unit=unit,
                                        source_fact_ids=[source_id],
                                        available_at=bundle.metadata.available_at,
                                    )
                                ],
                            )
                        )
        return breakdowns

    @staticmethod
    def _add_total_debt_fallback(
        grouped: dict[str, list[FinancialPoint]],
        warnings: list[str],
    ) -> None:
        existing_periods = {point.period for point in grouped.get("TOTAL_DEBT", [])}
        component_names = ["SHORT_TERM_DEBT", "LONG_TERM_DEBT", "LEASE_LIABILITIES", "CONVERTIBLE_LIABILITIES"]
        by_period: dict[date, list[FinancialPoint]] = defaultdict(list)
        for name in component_names:
            for point in grouped.get(name, []):
                by_period[point.period].append(point)
        for period, points in by_period.items():
            if period in existing_periods:
                continue
            units = {point.unit for point in points}
            if len(units) != 1:
                warnings.append(f"debt components at {period} use incompatible units; total debt was not derived")
                continue
            grouped["TOTAL_DEBT"].append(
                FinancialPoint(
                    period=period,
                    period_basis=points[0].period_basis,
                    value=sum((abs(point.value) for point in points), Decimal(0)),
                    unit=next(iter(units)),
                    source_fact_ids=[source for point in points for source in point.source_fact_ids],
                    available_at=max(point.available_at for point in points),
                )
            )

    def _add_component_capex_fallback(
        self,
        grouped: dict[str, list[FinancialPoint]],
        chosen: dict[tuple[str, date], StructuredFact],
        capex_components: dict[date, list[tuple[str, StructuredFact]]],
        warnings: list[str],
    ) -> None:
        for period, components in capex_components.items():
            if ("CAPEX", period) in chosen:
                continue
            preferred_scope = max(
                (fact for _identity, fact in components),
                key=_fact_preference,
            ).scope
            by_identity: dict[str, list[StructuredFact]] = defaultdict(list)
            for identity, fact in components:
                if fact.scope == preferred_scope:
                    by_identity[identity].append(fact)
            selected: list[StructuredFact] = []
            for facts in by_identity.values():
                facts.sort(key=_fact_preference, reverse=True)
                selected.append(facts[0])
            if not selected:
                continue
            normalized = [_normalized_value_and_unit(fact) for fact in selected]
            units = {unit for _value, unit in normalized}
            if len(units) != 1:
                warnings.append(
                    f"CAPEX components at {period} use incompatible units; fallback was skipped"
                )
                continue
            grouped["CAPEX"].append(
                FinancialPoint(
                    period=period,
                    period_basis=_fact_period_basis(selected[0]),
                    value=-sum((abs(value) for value, _unit in normalized), Decimal(0)),
                    unit=next(iter(units)),
                    source_fact_ids=[fact.fact_id for fact in selected],
                    available_at=max(fact.available_at for fact in selected),
                )
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
            cash = index.get(("CASH", period))
            debt = index.get(("TOTAL_DEBT", period))
            receivables = index.get(("RECEIVABLES", period))
            inventory = index.get(("INVENTORY", period))
            payables = index.get(("PAYABLES", period))
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
            if cash and debt and cash.unit == debt.unit:
                result.append(
                    DerivedMetric(
                        name="NET_DEBT",
                        period=period,
                        value=debt.value - cash.value,
                        unit=debt.unit or "UNKNOWN",
                        derived_from_fact_ids=[*debt.source_fact_ids, *cash.source_fact_ids],
                    )
                )
            if receivables and inventory and payables and len({receivables.unit, inventory.unit, payables.unit}) == 1:
                result.append(
                    DerivedMetric(
                        name="OPERATING_NWC",
                        period=period,
                        value=receivables.value + inventory.value - payables.value,
                        unit=receivables.unit or "UNKNOWN",
                        derived_from_fact_ids=[
                            *receivables.source_fact_ids,
                            *inventory.source_fact_ids,
                            *payables.source_fact_ids,
                        ],
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
        annual_revenues = [point for point in revenues.points if point.period_basis == "FY"] if revenues else []
        if len(annual_revenues) >= 2:
            first, last = annual_revenues[0], annual_revenues[-1]
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
