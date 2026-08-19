from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Iterable

from moatrader.api.models import (
    Assumption,
    CompanyProfile,
    DecisionSupport,
    DriverRange,
    EconomicValueScore,
    EvidenceItem,
    EvidenceSource,
    ImpliedDriver,
    ImpliedPoint,
    IndustryAnalysis,
    IndustryForce,
    MarketExpectations,
    Metric,
    MixItem,
    MoatAnalysis,
    MoatAxis,
    ModelRoute,
    PriceExplanation,
    ReportMeta,
    ResearchReport,
    SensitivityAnalysis,
    SensitivityDriver,
    ThesisAnalysis,
    ThesisChange,
    ThesisMonitorItem,
    ValuationAnalysis,
    ValuationScenario,
    ValueLink,
    VersionInfo,
)
from moatrader.api.repository import (
    ResearchArtifactNotFoundError,
    ResearchArtifactRepository,
    ResearchArtifacts,
)
from moatrader.financial.dcf import DcfAssumptions, DcfEngine


SCHEMA_VERSION = "fundamental-research/1.0"
CALCULATION_VERSION = "research-assembly/1.0"
PCT = Decimal("100")


MOAT_AXES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("switching_cost", "전환비용", ("SWITCHING_COST",)),
    ("network_effect", "네트워크 효과", ("NETWORK_EFFECT",)),
    ("brand_intangible", "브랜드·무형자산", ("INTANGIBLE_ASSET", "BRAND")),
    ("pricing_power", "가격 결정력", ("PRICING_POWER",)),
    ("scale", "규모·설치 기반", ("SCALE_ADVANTAGE",)),
    ("cost_advantage", "비용 우위", ("COST_ADVANTAGE",)),
)


@dataclass(frozen=True)
class SurfacePoint:
    growth: float
    margin: float
    roiic: float
    cap_years: int
    modeled_price: float
    relative_error: float


def _decimal(value: Any, default: str = "0") -> Decimal:
    try:
        return Decimal(str(value))
    except (ArithmeticError, TypeError, ValueError):
        return Decimal(default)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _pct(value: Any) -> float:
    return round(_float(value) * 100, 2)


def _clamp(value: Decimal, low: Decimal, high: Decimal) -> Decimal:
    return min(max(value, low), high)


def _confidence_label(value: float) -> str:
    if value >= 0.75:
        return "HIGH"
    if value >= 0.5:
        return "MEDIUM"
    return "LOW"


def _fragility_label(terminal_share: float, scenario_width: float) -> str:
    if terminal_share >= 0.75 or scenario_width >= 0.9:
        return "HIGH"
    if terminal_share >= 0.6 or scenario_width >= 0.5:
        return "MEDIUM"
    return "LOW"


def _clean_dimension(value: str) -> str:
    cleaned = value.strip().removeprefix("-").strip()
    replacements = {
        "SearchPlatform": "서치플랫폼",
        "Commerce": "커머스",
        "Fintech": "핀테크",
        "Contents": "콘텐츠",
        "Enterprise": "엔터프라이즈",
        "US": "미국",
        "JP": "일본",
        "OtherCountries": "기타 해외",
        "CountryOfDomicile": "국내",
    }
    for token, label in replacements.items():
        if token in cleaned:
            return label
    return cleaned.replace("MemberOfProductsAndServicesMemberOfDisclosureOfOperatingSegmentsTableOfMember", "")


class FundamentalResearchService:
    def __init__(self, repository: ResearchArtifactRepository) -> None:
        self.repository = repository
        self.dcf_engine = DcfEngine()

    def get_report(self, ticker: str, *, as_of=None) -> ResearchReport:
        artifacts = self.repository.load(ticker, as_of=as_of)
        assumptions = DcfAssumptions.model_validate(artifacts.dcf_assumptions)
        base_value = self.dcf_engine.value(assumptions)
        current_price = _float(artifacts.result.get("current_price"))
        scenarios = self._scenarios(assumptions, current_price, base_value.assumption_confidence)
        surface = self._reverse_surface(assumptions, current_price)
        sensitivity = self._sensitivity(assumptions, base_value.fair_value_per_share, scenarios)
        evidence = self._evidence(artifacts)
        evidence_by_id = {item.id: item for item in evidence}
        company = self._company(artifacts)
        moat = self._moat(artifacts, evidence_by_id)
        industry = self._industry(company, moat, evidence)
        economic_value = self._economic_value(
            artifacts,
            current_price=current_price,
            base_fair_value=_float(base_value.fair_value_per_share),
            terminal_share=_float(base_value.terminal_value_share),
            scenario_width=(scenarios[-1].high - scenarios[0].low) / max(current_price, 1),
        )
        valuation = self._valuation(
            artifacts,
            assumptions,
            scenarios,
            economic_value,
            current_price,
            _float(base_value.fair_value_per_share),
        )
        price_explanation = self._price_explanation(
            company, moat, valuation, surface, sensitivity, evidence
        )
        thesis = self._thesis(
            artifacts, company, moat, valuation, sensitivity, evidence
        )
        quality_warnings = list(artifacts.dcf_assumptions.get("provenance_warnings", []))
        if len(self.repository.latest_results()) < 20:
            quality_warnings.append("Economic Value percentile uses a small available-report reference sample.")
        return ResearchReport(
            meta=self._meta(artifacts, quality_warnings),
            company=company,
            industry=industry,
            moat=moat,
            valuation=valuation,
            market_expectations=surface,
            sensitivity=sensitivity,
            price_explanation=price_explanation,
            evidence=evidence,
            thesis=thesis,
            decision_support=self._decision_support(
                moat, valuation, sensitivity, thesis, evidence
            ),
            versions=VersionInfo(
                runner=str(artifacts.result.get("runner_version", "unknown")),
                model=str(artifacts.run_manifest.get("model", "unknown")),
                prompt=str(artifacts.run_manifest.get("prompt_version", "unknown")),
                parser=str(artifacts.run_manifest.get("parser_version", "unknown")),
                calculation=CALCULATION_VERSION,
            ),
        )

    @staticmethod
    def _meta(artifacts: ResearchArtifacts, warnings: list[str]) -> ReportMeta:
        result = artifacts.result
        manifest = artifacts.run_manifest
        coverage = _float(
            artifacts.moat_score.get("document_coverage", {}).get("moat_evidence_coverage")
        )
        data_grade = "RESEARCH" if coverage >= 0.7 else "LIMITED" if coverage > 0 else "INSUFFICIENT"
        return ReportMeta(
            schema_version=SCHEMA_VERSION,
            report_id=f"{result['ticker']}:{result['run_signature'][:16]}",
            generated_at=datetime.now(UTC),
            as_of=datetime.fromisoformat(result["valuation_as_of"]),
            evidence_cutoff=datetime.fromisoformat(
                manifest.get("evidence_cutoff", result["valuation_as_of"])
            ),
            price_as_of=datetime.fromisoformat(result["price_as_of"]),
            data_grade=data_grade,
            source_document_count=len(result.get("source_document_ids", [])),
            evidence_count=int(result.get("evidence_count", 0)),
            warnings=warnings,
        )

    @staticmethod
    def _scenario_input(
        base: DcfAssumptions,
        *,
        growth_scale: Decimal,
        margin_delta: Decimal,
        wacc_delta: Decimal,
    ) -> DcfAssumptions:
        payload = base.model_dump(mode="python")
        payload["revenue_growth"] = [
            _clamp(item * growth_scale, Decimal("-0.25"), Decimal("0.60"))
            for item in base.revenue_growth
        ]
        payload["ebit_margin"] = [
            _clamp(item + margin_delta, Decimal("-0.50"), Decimal("0.80"))
            for item in base.ebit_margin
        ]
        payload["wacc"] = _clamp(base.wacc + wacc_delta, Decimal("0.03"), Decimal("0.35"))
        return DcfAssumptions.model_validate(payload)

    def _scenarios(
        self,
        assumptions: DcfAssumptions,
        current_price: float,
        confidence: Decimal,
    ) -> list[ValuationScenario]:
        definitions = (
            ("bear", "Bear", Decimal("0.70"), Decimal("-0.020"), Decimal("0.010")),
            ("base", "Base", Decimal("1.00"), Decimal("0.000"), Decimal("0.000")),
            ("bull", "Bull", Decimal("1.20"), Decimal("0.015"), Decimal("-0.0075")),
        )
        band = Decimal("0.04") + (Decimal(1) - confidence) * Decimal("0.08")
        scenarios: list[ValuationScenario] = []
        for scenario_id, label, growth, margin, wacc in definitions:
            varied = self._scenario_input(
                assumptions,
                growth_scale=growth,
                margin_delta=margin,
                wacc_delta=wacc,
            )
            value = self.dcf_engine.value(varied).fair_value_per_share
            scenarios.append(
                ValuationScenario(
                    id=scenario_id,
                    label=label,
                    low=_float(value * (Decimal(1) - band)),
                    central=_float(value),
                    high=_float(value * (Decimal(1) + band)),
                    upside_pct=round((_float(value) / max(current_price, 1) - 1) * 100, 1),
                    assumptions=[
                        f"성장 경로 × {growth}",
                        f"영업마진 {float(margin) * 100:+.1f}%p",
                        f"WACC {float(wacc) * 100:+.2f}%p",
                    ],
                )
            )
        return scenarios

    @staticmethod
    def _resize_forecast(
        base: DcfAssumptions,
        horizon: int,
        growth_scale: Decimal,
        margin_delta: Decimal,
    ) -> tuple[list[Decimal], list[Decimal]]:
        growth = [
            _clamp(item * growth_scale, Decimal("-0.25"), Decimal("0.75"))
            for item in base.revenue_growth[:horizon]
        ]
        margins = [
            _clamp(item + margin_delta, Decimal("-0.50"), Decimal("0.80"))
            for item in base.ebit_margin[:horizon]
        ]
        while len(growth) < horizon:
            remaining = horizon - len(growth)
            last_growth = growth[-1] if growth else base.terminal_growth
            step = (last_growth - base.terminal_growth) / Decimal(remaining + 1)
            growth.append(max(base.terminal_growth, last_growth - step))
            margins.append(margins[-1] if margins else base.ebit_margin[-1] + margin_delta)
        return growth, margins

    @staticmethod
    def _roiic_proxy(
        growth: Decimal,
        margin: Decimal,
        tax_rate: Decimal,
        depreciation: Decimal,
        capex: Decimal,
        nwc: Decimal,
    ) -> float:
        incremental_investment = max(
            capex - depreciation + nwc * max(growth, Decimal(0)), Decimal("0.0001")
        )
        incremental_nopat = max(growth, Decimal(0)) * max(
            margin * (Decimal(1) - tax_rate), Decimal(0)
        )
        return _float(incremental_nopat / incremental_investment)

    def _reverse_surface(
        self, assumptions: DcfAssumptions, current_price: float
    ) -> MarketExpectations:
        points: list[SurfacePoint] = []
        growth_scales = ("0.50", "0.75", "1.00", "1.25", "1.50")
        margin_deltas = ("-0.030", "-0.015", "0", "0.015", "0.030", "0.045")
        efficiencies = ("0.75", "1.00", "1.25", "1.50")
        cap_years = (3, 5, 7, 9, 12)
        for scale_raw in growth_scales:
            for margin_raw in margin_deltas:
                for efficiency_raw in efficiencies:
                    for cap in cap_years:
                        scale = Decimal(scale_raw)
                        margin_delta = Decimal(margin_raw)
                        efficiency = Decimal(efficiency_raw)
                        growth, margins = self._resize_forecast(
                            assumptions, cap, scale, margin_delta
                        )
                        payload = assumptions.model_dump(mode="python")
                        payload["revenue_growth"] = growth
                        payload["ebit_margin"] = margins
                        spread = max(
                            assumptions.capex_pct_revenue
                            - assumptions.depreciation_pct_revenue,
                            Decimal("0.001"),
                        )
                        payload["capex_pct_revenue"] = (
                            assumptions.depreciation_pct_revenue + spread / efficiency
                        )
                        varied = DcfAssumptions.model_validate(payload)
                        valuation = self.dcf_engine.value(varied)
                        modeled = _float(valuation.fair_value_per_share)
                        if modeled <= 0:
                            continue
                        revenue_multiple = math.prod(1 + _float(item) for item in growth)
                        cagr = revenue_multiple ** (1 / len(growth)) - 1
                        roiic = self._roiic_proxy(
                            growth[-1],
                            margins[-1],
                            assumptions.tax_rate,
                            assumptions.depreciation_pct_revenue,
                            varied.capex_pct_revenue,
                            assumptions.nwc_pct_revenue,
                        )
                        points.append(
                            SurfacePoint(
                                growth=cagr,
                                margin=_float(margins[-1]),
                                roiic=roiic,
                                cap_years=cap,
                                modeled_price=modeled,
                                relative_error=(modeled / max(current_price, 1)) - 1,
                            )
                        )
        points.sort(key=lambda item: (abs(item.relative_error), item.cap_years))
        solutions = [item for item in points if abs(item.relative_error) <= 0.05]
        representatives = solutions or points[:18]
        display_points = representatives[:8]

        if not representatives:
            return MarketExpectations(
                status="UNAVAILABLE",
                method="Joint reverse-DCF surface (legacy FCFF bridge)",
                solution_count=0,
                evaluated_point_count=0,
                tolerance_pct=5.0,
                drivers=[],
                representative_points=[],
                headline=(
                    "현재 FCFF 탐색 범위에서는 양의 주주가치를 만드는 가격 내재 기대 조합을 "
                    "식별하지 못했습니다."
                ),
                identification_caveat=(
                    "모든 탐색 조합의 영업가치가 순부채를 넘지 못해 Reverse DCF 범위를 "
                    "표시하지 않습니다. 이는 보고서 오류나 0원 목표가가 아니라, 현재 가정과 "
                    "탐색 범위로는 시장가격을 설명할 수 없다는 진단입니다."
                ),
            )

        def bounds(attr: str, multiplier: float = 1.0) -> DriverRange:
            values = [getattr(item, attr) * multiplier for item in representatives]
            return DriverRange(low=round(min(values), 2), high=round(max(values), 2))

        base_growth = math.prod(1 + _float(item) for item in assumptions.revenue_growth) ** (
            1 / len(assumptions.revenue_growth)
        ) - 1
        base_roiic = self._roiic_proxy(
            assumptions.revenue_growth[-1],
            assumptions.ebit_margin[-1],
            assumptions.tax_rate,
            assumptions.depreciation_pct_revenue,
            assumptions.capex_pct_revenue,
            assumptions.nwc_pct_revenue,
        )
        growth_range = bounds("growth", 100)
        margin_range = bounds("margin", 100)
        roiic_range = bounds("roiic", 100)
        cap_range = bounds("cap_years")
        headline = (
            f"현재 가격은 매출 CAGR {growth_range.low:.1f}~{growth_range.high:.1f}%, "
            f"정상 마진 {margin_range.low:.1f}~{margin_range.high:.1f}% 조합과 맞닿아 있습니다."
        )
        return MarketExpectations(
            status="AVAILABLE",
            method="Joint reverse-DCF surface (legacy FCFF bridge)",
            solution_count=len(solutions),
            evaluated_point_count=len(points),
            tolerance_pct=5.0,
            drivers=[
                ImpliedDriver(
                    id="growth",
                    label="Sales Growth",
                    unit="% CAGR",
                    implied=growth_range,
                    base_case=round(base_growth * 100, 2),
                    interpretation="시장가격과 양립하는 명시기간 매출 성장 조합입니다.",
                ),
                ImpliedDriver(
                    id="margin",
                    label="Operating Margin",
                    unit="%",
                    implied=margin_range,
                    base_case=round(_float(assumptions.ebit_margin[-1]) * 100, 2),
                    interpretation="시장가격이 허용하는 정상 영업마진 조합입니다.",
                ),
                ImpliedDriver(
                    id="roiic",
                    label="ROIIC",
                    unit="% proxy",
                    implied=roiic_range,
                    base_case=round(base_roiic * 100, 2),
                    interpretation="증분 NOPAT과 순투자율로 계산한 재투자 효율 대용치입니다.",
                ),
                ImpliedDriver(
                    id="cap",
                    label="CAP",
                    unit="years",
                    implied=cap_range,
                    base_case=float(len(assumptions.revenue_growth)),
                    interpretation="초과성과가 명시적으로 유지·fade되는 기간 대용치입니다.",
                ),
            ],
            representative_points=[
                ImpliedPoint(
                    growth_pct=round(item.growth * 100, 2),
                    margin_pct=round(item.margin * 100, 2),
                    roiic_pct=round(item.roiic * 100, 2),
                    cap_years=item.cap_years,
                    modeled_price=round(item.modeled_price, 2),
                    relative_error_pct=round(item.relative_error * 100, 2),
                )
                for item in display_points
            ],
            headline=headline,
            identification_caveat=(
                "Reverse DCF의 네 변수는 공동으로 가격을 설명하므로 하나의 정답으로 식별되지 않습니다. "
                "표시 범위는 5% 가격 허용오차 내의 복수 조합이며 ROIIC와 CAP는 현재 FCFF 자료에 맞춘 대용치입니다."
            ),
        )

    def _sensitivity(
        self,
        assumptions: DcfAssumptions,
        base_value: Decimal,
        scenarios: list[ValuationScenario],
    ) -> SensitivityAnalysis:
        variants: list[tuple[str, str, str, DcfAssumptions]] = []
        payload = assumptions.model_dump(mode="python")
        payload["revenue_growth"] = [item + Decimal("0.01") for item in assumptions.revenue_growth]
        variants.append(("growth", "매출 성장", "+1.0%p", DcfAssumptions.model_validate(payload)))
        payload = assumptions.model_dump(mode="python")
        payload["ebit_margin"] = [item + Decimal("0.01") for item in assumptions.ebit_margin]
        variants.append(("margin", "영업마진", "+1.0%p", DcfAssumptions.model_validate(payload)))
        payload = assumptions.model_dump(mode="python")
        spread = max(
            assumptions.capex_pct_revenue - assumptions.depreciation_pct_revenue,
            Decimal("0.001"),
        )
        payload["capex_pct_revenue"] = assumptions.depreciation_pct_revenue + spread / Decimal("1.10")
        variants.append(("roiic", "재투자 효율", "+10%", DcfAssumptions.model_validate(payload)))
        growth, margins = self._resize_forecast(
            assumptions, len(assumptions.revenue_growth) + 1, Decimal(1), Decimal(0)
        )
        payload = assumptions.model_dump(mode="python")
        payload["revenue_growth"] = growth
        payload["ebit_margin"] = margins
        variants.append(("cap", "CAP", "+1년", DcfAssumptions.model_validate(payload)))
        drivers: list[SensitivityDriver] = []
        for driver_id, label, change, varied in variants:
            varied_value = self.dcf_engine.value(varied).fair_value_per_share
            impact = _float((varied_value / base_value - Decimal(1)) * PCT)
            drivers.append(
                SensitivityDriver(
                    id=driver_id,
                    label=label,
                    assumed_change=change,
                    value_impact_pct=round(impact, 2),
                    tone="positive" if impact > 0 else "negative" if impact < 0 else "neutral",
                )
            )
        primary = max(drivers, key=lambda item: abs(item.value_impact_pct))
        width = (scenarios[-1].high - scenarios[0].low) / max(scenarios[1].central, 1)
        fragility = "HIGH" if width >= 0.8 else "MEDIUM" if width >= 0.45 else "LOW"
        return SensitivityAnalysis(
            primary_driver_id=primary.id,
            primary_driver_label=primary.label,
            drivers=drivers,
            turbo_trigger=(
                f"{primary.label} 가정이 검증 범위의 상단으로 이동하는지가 가장 큰 재평가 촉매입니다. "
                f"표준 충격에서 가치 영향은 {primary.value_impact_pct:+.1f}%입니다."
            ),
            fragility=fragility,
        )

    @staticmethod
    def _latest_metric(snapshot: dict[str, Any], name: str) -> dict[str, Any] | None:
        rows = [item for item in snapshot.get("derived_metrics", []) if item.get("name") == name]
        return sorted(rows, key=lambda item: item.get("period") or "")[-1] if rows else None

    @staticmethod
    def _mix(snapshot: dict[str, Any]) -> list[MixItem]:
        revenue_series = next(
            (item for item in snapshot.get("series", []) if item.get("concept") == "REVENUE"),
            None,
        )
        annual = [
            item for item in (revenue_series or {}).get("points", []) if item.get("period_basis") == "FY"
        ]
        if not annual:
            return []
        latest_period = max(item["period"] for item in annual)
        total = max(
            (_float(item["value"]) for item in annual if item["period"] == latest_period),
            default=0,
        )
        rows = []
        for item in snapshot.get("breakdowns", []):
            if item.get("concept") != "REVENUE":
                continue
            dimension = str(item.get("dimension", ""))
            if not dimension.strip().startswith("-"):
                continue
            points = [
                point
                for point in item.get("points", [])
                if point.get("period") == latest_period and point.get("period_basis") == "FY"
            ]
            if not points:
                continue
            value = max(_float(point.get("value")) for point in points)
            if value <= 0:
                continue
            rows.append(
                MixItem(
                    name=_clean_dimension(dimension),
                    value=value,
                    share_pct=round(value / total * 100, 1) if total else None,
                    unit="KRW",
                    period=latest_period,
                )
            )
        unique: dict[str, MixItem] = {}
        for row in rows:
            if row.name not in unique or row.value > unique[row.name].value:
                unique[row.name] = row
        return sorted(unique.values(), key=lambda item: item.value, reverse=True)[:8]

    @staticmethod
    def _geography(snapshot: dict[str, Any]) -> list[MixItem]:
        candidates: list[MixItem] = []
        for item in snapshot.get("breakdowns", []):
            dimension = str(item.get("dimension", ""))
            if item.get("concept") != "REVENUE" or "GeographicalAreasAxis" not in dimension:
                continue
            annual = [point for point in item.get("points", []) if point.get("period_basis") == "FY"]
            if not annual:
                continue
            point = sorted(annual, key=lambda row: row.get("period", ""))[-1]
            candidates.append(
                MixItem(
                    name=_clean_dimension(dimension),
                    value=_float(point.get("value")),
                    unit=str(point.get("unit", "KRW")),
                    period=str(point.get("period")),
                )
            )
        if not candidates:
            return []
        latest = max(item.period for item in candidates)
        selected = [item for item in candidates if item.period == latest and item.value > 0]
        total = sum(item.value for item in selected)
        return [
            item.model_copy(update={"share_pct": round(item.value / total * 100, 1) if total else None})
            for item in sorted(selected, key=lambda row: row.value, reverse=True)[:6]
        ]

    def _company(self, artifacts: ResearchArtifacts) -> CompanyProfile:
        snapshot = artifacts.financial_snapshot
        mix = self._mix(snapshot)
        geography = self._geography(snapshot)
        top_names = [item.name for item in mix[:3]]
        business_model = (
            f"{', '.join(top_names)}를 중심으로 매출을 만드는 다각화 영업모델입니다."
            if top_names
            else "공시 재무와 사업 근거를 결합해 현금흐름을 창출하는 영업모델을 분석합니다."
        )
        cagr = self._latest_metric(snapshot, "REVENUE_CAGR")
        margin = self._latest_metric(snapshot, "EBIT_MARGIN")
        fcf_margin = self._latest_metric(snapshot, "FCF_MARGIN")
        metrics = [
            Metric(
                id="revenue_cagr",
                label="매출 CAGR",
                value=_pct(cagr.get("value")) if cagr else None,
                unit="%",
                period=cagr.get("period") if cagr else None,
                trend="positive" if cagr and _float(cagr.get("value")) > 0 else "neutral",
            ),
            Metric(
                id="ebit_margin",
                label="영업이익률",
                value=_pct(margin.get("value")) if margin else None,
                unit="%",
                period=margin.get("period") if margin else None,
                trend="positive" if margin and _float(margin.get("value")) >= 0.1 else "neutral",
            ),
            Metric(
                id="fcf_margin",
                label="FCF 마진",
                value=_pct(fcf_margin.get("value")) if fcf_margin else None,
                unit="%",
                period=fcf_margin.get("period") if fcf_margin else None,
                trend="warning" if fcf_margin and _float(fcf_margin.get("value")) < 0.05 else "neutral",
            ),
        ]
        digital_tokens = {"서치플랫폼", "커머스", "핀테크", "콘텐츠", "엔터프라이즈"}
        industry = (
            "디지털 플랫폼·인터넷 서비스"
            if digital_tokens.intersection(top_names)
            else "다각화 영업기업"
        )
        return CompanyProfile(
            ticker=str(artifacts.result["ticker"]),
            issuer_id=artifacts.result.get("issuer_id"),
            issuer_name=str(artifacts.result["issuer_name"]),
            business_summary=(
                f"{artifacts.result['issuer_name']}의 공시 원문, 재무 세그먼트와 해자 근거를 "
                "사업→경쟁우위→가치의 흐름으로 연결한 시점고정 분석입니다."
            ),
            business_model=business_model,
            industry_label=industry,
            revenue_mix=mix,
            geography=geography,
            key_metrics=metrics,
        )

    @staticmethod
    def _source_index(artifacts: ResearchArtifacts) -> dict[str, dict[str, Any]]:
        fallback_at = artifacts.run_manifest.get(
            "evidence_cutoff", artifacts.result["valuation_as_of"]
        )
        records = artifacts.evidence_ledger.get("records", [])
        return {
            str(record.get("evidence_id")): {
                "document_id": str(record.get("source_document_id", "UNKNOWN")),
                "available_at": record.get("source_available_at") or fallback_at,
            }
            for record in records
        }

    def _evidence(self, artifacts: ResearchArtifacts) -> list[EvidenceItem]:
        source_index = self._source_index(artifacts)
        moat = artifacts.moat_score
        selected_ids: list[str] = []
        for mechanism in moat.get("mechanisms", []):
            selected_ids.extend(mechanism.get("evidence_ids", [])[:2])
        selected_ids.extend(moat.get("counterevidence_ids", [])[:5])
        selected_ids = list(dict.fromkeys(selected_ids))[:16]
        cards = {str(item.get("evidence_id")): item for item in artifacts.dossier.get("evidence", [])}
        output: list[EvidenceItem] = []
        for evidence_id in selected_ids:
            card = cards.get(evidence_id)
            if not card:
                continue
            source = source_index.get(
                evidence_id,
                {
                    "document_id": "UNKNOWN",
                    "available_at": artifacts.run_manifest.get(
                        "evidence_cutoff", artifacts.result["valuation_as_of"]
                    ),
                },
            )
            document_id = source["document_id"]
            source_type = str(card.get("source_type", "UNKNOWN"))
            url = (
                f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={document_id}"
                if source_type == "DART" and document_id != "UNKNOWN"
                else ""
            )
            raw_direction = str(card.get("direction", ""))
            direction = "positive" if "POSITIVE" in raw_direction else "negative" if "NEGATIVE" in raw_direction else "neutral"
            output.append(
                EvidenceItem(
                    id=evidence_id,
                    direction=direction,
                    evidence_type=str(card.get("evidence_type", "OTHER")),
                    fact=str(card.get("fact", "")),
                    exact_quote=str(card.get("raw_quote", "")),
                    mechanism=[str(item) for item in card.get("mechanism", [])],
                    strength=_float(card.get("strength"), 0.5),
                    reliability=_float(card.get("reliability"), 0.5),
                    period=card.get("period"),
                    linked_drivers=[str(item) for item in card.get("dcf_links", [])],
                    source=EvidenceSource(
                        source_type=source_type,
                        document_id=document_id,
                        title=f"DART 원문 공시 {document_id}" if source_type == "DART" else f"{source_type} 원문",
                        available_at=datetime.fromisoformat(str(source["available_at"])),
                        url=url,
                    ),
                )
            )
        return output

    @staticmethod
    def _moat(
        artifacts: ResearchArtifacts, evidence_by_id: dict[str, EvidenceItem]
    ) -> MoatAnalysis:
        score_data = artifacts.moat_score
        mechanism_scores: dict[str, dict[str, Any]] = {
            str(item.get("evidence_type")): item for item in score_data.get("mechanisms", [])
        }
        negative_by_type: dict[str, list[str]] = {}
        for evidence_id in score_data.get("counterevidence_ids", []):
            item = evidence_by_id.get(evidence_id)
            if item:
                negative_by_type.setdefault(item.evidence_type, []).append(evidence_id)
        axes: list[MoatAxis] = []
        for axis_id, label, types in MOAT_AXES:
            matches = [mechanism_scores[item] for item in types if item in mechanism_scores]
            evidence_ids = list(
                dict.fromkeys(
                    evidence_id
                    for match in matches
                    for evidence_id in match.get("evidence_ids", [])
                )
            )[:4]
            counters = list(
                dict.fromkeys(
                    evidence_id
                    for evidence_type in types
                    for evidence_id in negative_by_type.get(evidence_type, [])
                )
            )[:3]
            axis_score = max((_float(item.get("score")) for item in matches), default=0.0)
            status = "MIXED" if evidence_ids and counters else "SUPPORTED" if evidence_ids else "NOT_OBSERVED"
            explanation = (
                f"{len(evidence_ids)}개의 검증된 원문 근거가 {label} 메커니즘을 지지합니다."
                if evidence_ids
                else f"현재 cutoff 원문만으로는 {label}을 확인하지 못했습니다."
            )
            axes.append(
                MoatAxis(
                    id=axis_id,
                    label=label,
                    status=status,
                    score=round(axis_score, 2) if evidence_ids else None,
                    explanation=explanation,
                    evidence_ids=evidence_ids,
                    counterevidence_ids=counters,
                )
            )
        score = _float(score_data.get("economic_moat_score"))
        rating = "WIDE" if score >= 7.5 else "NARROW" if score >= 5 else "NONE"
        primary = [axis.label for axis in axes if axis.status != "NOT_OBSERVED"][:3]
        coverage = _float(score_data.get("document_coverage", {}).get("moat_evidence_coverage"))
        return MoatAnalysis(
            score=score,
            rating=rating,
            durability=str(score_data.get("durability", "UNKNOWN")),
            confidence=_float(score_data.get("model_confidence")),
            evidence_coverage=coverage,
            primary_sources=primary,
            summary=(
                f"핵심 해자 원천은 {', '.join(primary)}입니다. "
                f"공시 기반 내구성 평가는 {score_data.get('durability', 'UNKNOWN')}이며, "
                "확인되지 않은 축은 점수로 추정하지 않았습니다."
            ),
            axes=axes,
        )

    @staticmethod
    def _industry(
        company: CompanyProfile, moat: MoatAnalysis, evidence: list[EvidenceItem]
    ) -> IndustryAnalysis:
        revenue_growth = next((item for item in company.key_metrics if item.id == "revenue_cagr"), None)
        regulatory = [item.id for item in evidence if item.evidence_type == "REGULATORY_BARRIER" and item.direction == "negative"]
        competition = [item.id for item in evidence if item.direction == "negative" and item.id not in regulatory]
        growth_value = revenue_growth.value if revenue_growth and revenue_growth.value is not None else 0
        forces = [
            IndustryForce(
                id="growth",
                label="성장",
                status="확장" if growth_value > 5 else "안정",
                tone="positive" if growth_value > 5 else "neutral",
                description=f"공시 재무에서 관측된 매출 CAGR은 {growth_value:.1f}%입니다.",
            ),
            IndustryForce(
                id="competition",
                label="경쟁",
                status="주의" if competition else "근거 제한",
                tone="warning" if competition else "neutral",
                description="경쟁우위는 플랫폼 이용·무형자산·규모의 유지 여부에 좌우됩니다.",
                evidence_ids=competition,
            ),
            IndustryForce(
                id="cycle",
                label="사이클",
                status="사업 믹스 분산",
                tone="neutral",
                description=f"{len(company.revenue_mix)}개 주요 매출 축의 조합이 단일 사업 변동을 일부 분산합니다.",
            ),
            IndustryForce(
                id="regulation",
                label="규제",
                status="모니터 필요" if regulatory else "중립",
                tone="negative" if regulatory else "neutral",
                description=f"원문에서 {len(regulatory)}개의 규제·소송 관련 반대 근거가 확인됐습니다.",
                evidence_ids=regulatory,
            ),
        ]
        primary_moat = moat.primary_sources[0] if moat.primary_sources else "검증된 경쟁우위"
        return IndustryAnalysis(
            structure_summary=(
                f"{company.industry_label}의 기업가치는 사용자·거래·콘텐츠가 연결되는 정도와 "
                "수익화 효율, 규제 비용의 균형에 의해 결정됩니다."
            ),
            forces=forces,
            value_driver_chain=[
                ValueLink(stage="industry", title="산업 구조", description=company.industry_label),
                ValueLink(stage="advantage", title="경쟁우위", description=primary_moat),
                ValueLink(stage="driver", title="가치 동인", description="성장·마진·재투자 효율·CAP"),
                ValueLink(stage="valuation", title="가치평가", description="FCFF 시나리오와 Reverse DCF"),
            ],
        )

    def _economic_value(
        self,
        artifacts: ResearchArtifacts,
        *,
        current_price: float,
        base_fair_value: float,
        terminal_share: float,
        scenario_width: float,
    ) -> EconomicValueScore:
        gaps: list[float] = []
        for result in self.repository.latest_results():
            price = _float(result.get("current_price"))
            fair = _float((result.get("dcf") or {}).get("fair_value_per_share"))
            if price > 0 and fair > 0:
                gaps.append(fair / price - 1)
        own_gap = base_fair_value / max(current_price, 1) - 1
        percentile = None
        if gaps:
            rank = sum(value < own_gap for value in gaps) + 0.5 * sum(value == own_gap for value in gaps)
            percentile = round(rank / len(gaps) * 100, 1)
        confidence_value = (
            _float((artifacts.result.get("dcf") or {}).get("assumption_confidence"))
            + _float(artifacts.moat_score.get("model_confidence"))
        ) / 2
        fragility = _fragility_label(terminal_share, scenario_width)
        label = (
            "상대적으로 저평가" if percentile is not None and percentile >= 70 else
            "중립 범위" if percentile is not None and percentile >= 30 else
            "상대적으로 높은 기대 반영" if percentile is not None else "표본 부족"
        )
        return EconomicValueScore(
            percentile=percentile,
            label=label,
            reference_class="동일 시점 MoatRader 완료 보고서",
            sample_size=len(gaps),
            confidence=_confidence_label(confidence_value),
            fragility=fragility,
            coverage=_float(
                artifacts.moat_score.get("document_coverage", {}).get("moat_evidence_coverage")
            ),
            caveat=(
                "이 percentile은 독립 알파가 아니라 모델별 가치격차를 현재 사용 가능한 완료 표본 안에서 정규화한 진단치입니다."
            ),
        )

    @staticmethod
    def _valuation(
        artifacts: ResearchArtifacts,
        assumptions: DcfAssumptions,
        scenarios: list[ValuationScenario],
        score: EconomicValueScore,
        current_price: float,
        fair_value: float,
    ) -> ValuationAnalysis:
        labels = {
            "revenue_growth": "매출 성장 경로",
            "ebit_margin": "영업이익률 경로",
            "wacc": "WACC",
            "terminal_growth": "영구성장률",
            "capex_pct_revenue": "매출 대비 CAPEX",
            "nwc_pct_revenue": "매출 대비 운전자본",
        }
        assumptions_output: list[Assumption] = []
        raw = assumptions.model_dump(mode="python")
        for key in labels:
            value = raw[key]
            if isinstance(value, list):
                shown = " → ".join(f"{_pct(item):.1f}%" for item in value)
            elif key in {"wacc", "terminal_growth", "capex_pct_revenue", "nwc_pct_revenue"}:
                shown = f"{_pct(value):.2f}%"
            else:
                shown = str(value)
            assumptions_output.append(
                Assumption(
                    id=key,
                    label=labels[key],
                    value=shown,
                    source_type=str(artifacts.dcf_assumptions.get("assumption_types", {}).get(key, "UNSPECIFIED")),
                    sources=[str(item) for item in artifacts.dcf_assumptions.get("assumption_sources", {}).get(key, [])],
                )
            )
        return ValuationAnalysis(
            route=ModelRoute(
                primary_model="FCFF",
                base_period=assumptions.base_period,
                rationale="현재 완료 산출물은 비금융 영업기업용 FCFF와 명시적 재투자 가정을 사용합니다.",
                cross_checks=["Reverse DCF", "Economic Value percentile", "Scenario sensitivity"],
            ),
            currency="KRW",
            current_price=current_price,
            base_fair_value=fair_value,
            base_value_gap_pct=round((fair_value / max(current_price, 1) - 1) * 100, 1),
            scenarios=scenarios,
            economic_value=score,
            assumptions=assumptions_output,
        )

    @staticmethod
    def _price_explanation(
        company: CompanyProfile,
        moat: MoatAnalysis,
        valuation: ValuationAnalysis,
        surface: MarketExpectations,
        sensitivity: SensitivityAnalysis,
        evidence: list[EvidenceItem],
    ) -> PriceExplanation:
        gap = valuation.base_value_gap_pct
        direction = "낮은" if gap > 0 else "높은"
        concern = next((item.fact for item in evidence if item.direction == "negative"), "경쟁우위 지속성과 정상 수익성에 대한 불확실성")
        primary_moat = moat.primary_sources[0] if moat.primary_sources else "경쟁우위"
        return PriceExplanation(
            headline=f"현재 가격은 FCFF 기준가치보다 {abs(gap):.1f}% {direction} 수준입니다.",
            summary=(
                f"단순 멀티플보다 중요한 쟁점은 {surface.headline} "
                f"{company.issuer_name}의 {primary_moat}가 이 가정을 지지하는지가 가격 해석의 핵심입니다."
            ),
            core_question=f"{sensitivity.primary_driver_label} 개선이 실제 현금흐름으로 이어질 수 있는가?",
            market_concern=concern,
            rerating_condition=sensitivity.turbo_trigger,
        )

    def _thesis(
        self,
        artifacts: ResearchArtifacts,
        company: CompanyProfile,
        moat: MoatAnalysis,
        valuation: ValuationAnalysis,
        sensitivity: SensitivityAnalysis,
        evidence: list[EvidenceItem],
    ) -> ThesisAnalysis:
        positive = [item for item in evidence if item.direction == "positive"]
        negative = [item for item in evidence if item.direction == "negative"]
        margin = next((item for item in company.key_metrics if item.id == "ebit_margin"), None)
        valuation_status = "INTACT" if valuation.base_value_gap_pct > 10 else "WATCH"
        monitor = [
            ThesisMonitorItem(
                id="margin",
                label="Margin thesis",
                status="WATCH" if margin and margin.value is not None and margin.value < 10 else "INTACT",
                tone="warning" if margin and margin.value is not None and margin.value < 10 else "positive",
                detail=f"최근 관측 영업이익률 {margin.value:.1f}%" if margin and margin.value is not None else "가용 마진 근거 없음",
            ),
            ThesisMonitorItem(
                id="moat",
                label="Moat evidence",
                status="INTACT" if moat.rating in {"WIDE", "NARROW"} else "WATCH",
                tone="positive" if moat.rating in {"WIDE", "NARROW"} else "warning",
                detail=f"공시 근거 점수 {moat.score:.2f}/10, 내구성 {moat.durability}",
                evidence_ids=[item.id for item in positive[:3]],
            ),
            ThesisMonitorItem(
                id="risk",
                label="Industry / regulation",
                status="WEAKENING" if len(negative) >= 3 else "WATCH",
                tone="negative" if len(negative) >= 3 else "warning",
                detail=f"구조적 반대 근거 {len(negative)}개를 계속 추적해야 합니다.",
                evidence_ids=[item.id for item in negative[:3]],
            ),
            ThesisMonitorItem(
                id="valuation",
                label="Valuation",
                status=valuation_status,
                tone="positive" if valuation_status == "INTACT" else "warning",
                detail=f"Base value gap {valuation.base_value_gap_pct:+.1f}%",
            ),
        ]
        changes: list[ThesisChange] = []
        try:
            previous = self.repository.previous(
                str(artifacts.result["ticker"]), artifacts.valuation_at
            )
        except ResearchArtifactNotFoundError:
            # Previous-period comparison is optional. A historical run can have a
            # COMPLETE result while predating one of the API's required artifacts;
            # that must not make the current, otherwise valid report unavailable.
            previous = None
        if previous:
            previous_score = _float(previous.moat_score.get("economic_moat_score"))
            current_score = moat.score
            changes.append(
                ThesisChange(
                    label="해자 점수",
                    previous=f"{previous_score:.2f}",
                    current=f"{current_score:.2f}",
                    tone="positive" if current_score > previous_score else "negative" if current_score < previous_score else "neutral",
                )
            )
            previous_margin = self._latest_metric(previous.financial_snapshot, "EBIT_MARGIN")
            if previous_margin and margin and margin.value is not None:
                old = _pct(previous_margin.get("value"))
                changes.append(
                    ThesisChange(
                        label="영업이익률",
                        previous=f"{old:.1f}%",
                        current=f"{margin.value:.1f}%",
                        tone="positive" if margin.value > old else "negative" if margin.value < old else "neutral",
                    )
                )
        return ThesisAnalysis(
            core_thesis=(
                f"{company.issuer_name}의 {', '.join(moat.primary_sources[:2]) or '검증된 경쟁우위'}가 "
                f"{sensitivity.primary_driver_label}을 방어하면 현재 가격과 기준가치의 간극이 축소될 수 있습니다."
            ),
            supporting_evidence_ids=[item.id for item in positive[:5]],
            breakers=[item.fact for item in negative[:4]],
            breaker_evidence_ids=[item.id for item in negative[:4]],
            monitor=monitor,
            changes_since_previous=changes,
        )

    @staticmethod
    def _decision_support(
        moat: MoatAnalysis,
        valuation: ValuationAnalysis,
        sensitivity: SensitivityAnalysis,
        thesis: ThesisAnalysis,
        evidence: list[EvidenceItem],
    ) -> DecisionSupport:
        negatives = len([item for item in evidence if item.direction == "negative"])
        if valuation.base_value_gap_pct > 10 and moat.rating in {"WIDE", "NARROW"} and negatives < 3:
            diagnosis = "싼 가격과 훼손되지 않은 경쟁우위가 함께 관측됩니다. 다만 가치함정 감소 효과는 별도 검증 전입니다."
        elif valuation.base_value_gap_pct > 10:
            diagnosis = "가격 할인은 있으나 반대 근거가 적지 않아 Value Trap 가능성을 배제할 수 없습니다."
        else:
            diagnosis = "현재 가격에는 기준 시나리오 이상의 기대가 반영돼 안전마진이 제한적입니다."
        return DecisionSupport(
            value_trap_diagnosis=diagnosis,
            payoff_profile=(
                f"Bear {valuation.scenarios[0].low:,.0f}원부터 Bull {valuation.scenarios[-1].high:,.0f}원까지의 "
                f"비대칭 범위이며 가정 취약도는 {valuation.economic_value.fragility}입니다."
            ),
            what_to_watch_next=[
                sensitivity.primary_driver_label,
                "신규 공시의 해자 침식 근거",
                "영업마진과 FCF 전환",
                *[item.label for item in thesis.monitor if item.status in {"WATCH", "WEAKENING"}][:2],
            ],
            use_boundary=(
                "이 응답은 종목 추천이나 독립 알파 신호가 아니라, 결정론적 가치평가와 cutoff 원문 근거를 연결한 진단 도구입니다."
            ),
            disclaimer="투자 판단과 손익의 책임은 사용자에게 있으며, 가정과 원문을 직접 검토해야 합니다.",
        )
