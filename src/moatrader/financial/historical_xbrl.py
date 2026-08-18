from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation

from lxml import etree
from pydantic import Field

from moatrader.canonical.models import ContractModel


_REVENUE = re.compile(r"^(?:Revenue|SalesRevenue|RevenueFromContractsWithCustomers)$", re.I)
_EBIT = re.compile(r"(?:OperatingIncomeLoss|ProfitLossFromOperatingActivities)", re.I)
_CAPEX = re.compile(r"PurchaseOf(?:PropertyPlantAndEquipment|IntangibleAssets)", re.I)
_DEPRECIATION = re.compile(r"(?:DepreciationAndAmortisation|Depreciation|Amortisation)(?:Expense)?$", re.I)
_CASH = re.compile(r"CashAndCashEquivalents$", re.I)
_DEBT_AGGREGATE = re.compile(r"^(?:Borrowings|LoansAndBorrowings)$", re.I)
_DEBT_COMPONENT = re.compile(
    r"(?:ShortTermBorrowings|LongTermBorrowings|CurrentPortionOfLongTermBorrowings|Bonds|Debentures|LeaseLiabilities)$",
    re.I,
)
_RECEIVABLES = re.compile(r"(?:TradeAndOtherCurrentReceivables|TradeReceivables)$", re.I)
_INVENTORY = re.compile(r"Inventories$", re.I)
_PAYABLES = re.compile(r"(?:TradeAndOtherCurrentPayables|TradePayables)$", re.I)


@dataclass(frozen=True)
class ContextInfo:
    context_id: str
    start: date | None
    end: date | None
    instant: date | None
    members: tuple[str, ...]

    @property
    def is_consolidated(self) -> bool:
        return any(member.endswith("ConsolidatedMember") for member in self.members)

    @property
    def is_separate(self) -> bool:
        return any(member.endswith("SeparateMember") for member in self.members)


@dataclass(frozen=True)
class NumericFact:
    concept: str
    value: Decimal
    context: ContextInfo


class AnnualFinancialMetrics(ContractModel):
    fiscal_year: int
    revenue: Decimal | None = None
    ebit: Decimal | None = None
    capex: Decimal | None = None
    depreciation: Decimal | None = None
    cash: Decimal | None = None
    debt: Decimal | None = None
    nwc: Decimal | None = None
    extracted_fact_count: int = Field(ge=0)
    metric_coverage_count: int = Field(ge=0, le=7)
    instance_member: str

    def as_assumption_metrics(self) -> dict[str, Decimal | None]:
        return {
            "revenue": self.revenue,
            "ebit": self.ebit,
            "capex": self.capex,
            "depreciation": self.depreciation,
            "cash": self.cash,
            "debt": self.debt,
            "nwc": self.nwc,
        }


def _local_name(tag: str) -> str:
    return etree.QName(tag).localname if tag.startswith("{") else tag.split(":")[-1]


def _parse_date(text: str | None) -> date | None:
    return date.fromisoformat(text[:10]) if text else None


def _contexts(root: etree._Element) -> dict[str, ContextInfo]:
    result: dict[str, ContextInfo] = {}
    for element in root.xpath("//*[local-name()='context']"):
        context_id = element.get("id")
        if not context_id:
            continue
        start = element.xpath("string(.//*[local-name()='startDate'][1])") or None
        end = element.xpath("string(.//*[local-name()='endDate'][1])") or None
        instant = element.xpath("string(.//*[local-name()='instant'][1])") or None
        members = tuple(
            " ".join(value.split())
            for value in element.xpath(".//*[local-name()='explicitMember']/text()")
        )
        result[context_id] = ContextInfo(
            context_id=context_id,
            start=_parse_date(start),
            end=_parse_date(end),
            instant=_parse_date(instant),
            members=members,
        )
    return result


def _numeric_facts(root: etree._Element, contexts: dict[str, ContextInfo]) -> list[NumericFact]:
    facts: list[NumericFact] = []
    for element in root.iter():
        context_ref = element.get("contextRef") or element.get("contextref")
        context = contexts.get(context_ref or "")
        if context is None or len(element):
            continue
        raw = "".join(element.itertext()).replace(",", "").strip()
        if not raw or raw in {"-", "—"}:
            continue
        try:
            value = Decimal(raw)
        except InvalidOperation:
            continue
        facts.append(NumericFact(concept=_local_name(element.tag), value=value, context=context))
    return facts


def _context_score(context: ContextInfo, *, period_end: date, flow: bool) -> int | None:
    if flow:
        if context.start is None or context.end != period_end:
            return None
        duration_days = (context.end - context.start).days
        if not 350 <= duration_days <= 370:
            return None
    elif context.instant != period_end:
        return None
    score = 0
    if context.is_consolidated:
        score += 100
    if context.is_separate:
        score -= 100
    unrelated = [
        member
        for member in context.members
        if not member.endswith("ConsolidatedMember") and not member.endswith("SeparateMember")
    ]
    score -= 10 * len(unrelated)
    if not context.members:
        score += 25
    return score


def _values(
    facts: list[NumericFact],
    pattern: re.Pattern[str],
    *,
    period_end: date,
    flow: bool,
) -> list[Decimal]:
    candidates: list[tuple[int, Decimal]] = []
    for fact in facts:
        if not pattern.search(fact.concept):
            continue
        score = _context_score(fact.context, period_end=period_end, flow=flow)
        if score is not None:
            candidates.append((score, fact.value))
    if not candidates:
        return []
    best = max(score for score, _value in candidates)
    return [value for score, value in candidates if score == best]


def _max_abs(values: list[Decimal]) -> Decimal | None:
    return max(values, key=lambda value: abs(value)) if values else None


def _sum_unique_abs(values: list[Decimal]) -> Decimal | None:
    unique = {abs(value) for value in values if value != 0}
    return sum(unique, Decimal(0)) if unique else None


def parse_dart_ifrs_archive(
    archive: bytes,
    *,
    fiscal_year: int,
    period_end: date | None = None,
) -> AnnualFinancialMetrics:
    target_period_end = period_end or date(fiscal_year, 12, 31)
    with zipfile.ZipFile(io.BytesIO(archive)) as source:
        instances = [info for info in source.infolist() if info.filename.lower().endswith(".xbrl")]
        if not instances:
            raise ValueError("DART IFRS archive has no XBRL instance")
        instance = max(instances, key=lambda item: item.file_size)
        content = source.read(instance)
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False, huge_tree=True)
    root = etree.fromstring(content, parser=parser)
    contexts = _contexts(root)
    facts = _numeric_facts(root, contexts)

    revenue = _max_abs(_values(facts, _REVENUE, period_end=target_period_end, flow=True))
    ebit = _max_abs(_values(facts, _EBIT, period_end=target_period_end, flow=True))
    capex = _sum_unique_abs(_values(facts, _CAPEX, period_end=target_period_end, flow=True))
    depreciation_values = _values(facts, _DEPRECIATION, period_end=target_period_end, flow=True)
    depreciation = _max_abs(depreciation_values)
    cash = _max_abs(_values(facts, _CASH, period_end=target_period_end, flow=False))
    aggregate_debt = _max_abs(
        _values(facts, _DEBT_AGGREGATE, period_end=target_period_end, flow=False)
    )
    component_debt = _sum_unique_abs(
        _values(facts, _DEBT_COMPONENT, period_end=target_period_end, flow=False)
    )
    debt = aggregate_debt if aggregate_debt is not None else component_debt
    receivables = _max_abs(_values(facts, _RECEIVABLES, period_end=target_period_end, flow=False))
    inventory = _max_abs(_values(facts, _INVENTORY, period_end=target_period_end, flow=False))
    payables = _max_abs(_values(facts, _PAYABLES, period_end=target_period_end, flow=False))
    nwc = None
    if any(value is not None for value in (receivables, inventory, payables)):
        nwc = (receivables or Decimal(0)) + (inventory or Decimal(0)) - (payables or Decimal(0))
    metrics = (revenue, ebit, capex, depreciation, cash, debt, nwc)
    return AnnualFinancialMetrics(
        fiscal_year=fiscal_year,
        revenue=revenue,
        ebit=ebit,
        capex=capex,
        depreciation=depreciation,
        cash=cash,
        debt=debt,
        nwc=nwc,
        extracted_fact_count=len(facts),
        metric_coverage_count=sum(value is not None for value in metrics),
        instance_member=instance.filename,
    )
