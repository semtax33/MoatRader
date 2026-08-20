from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
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
    SegmentTrend,
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
DETAIL_VALUATION_STATUSES = {"READY", "CALCULATED_NOT_SCREENING_ELIGIBLE"}


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
        try:
            artifacts = self.repository.load(ticker, as_of=as_of)
        except ResearchArtifactNotFoundError:
            artifacts = self._current_artifacts(ticker, as_of=as_of)
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

    def catalog_results(self) -> list[dict[str, Any]]:
        """Return detail-capable legacy and all-security reports once per ticker."""
        selected: dict[str, dict[str, Any]] = {}
        for result in self.repository.latest_results():
            ticker = str(result["ticker"]).upper()
            try:
                self.repository.load(ticker)
            except ResearchArtifactNotFoundError:
                continue
            selected[ticker] = result
        for report in self.repository.latest_current_report_catalog():
            ticker = str(report.get("ticker") or "").upper()
            if (
                not ticker
                or ticker in selected
                or report.get("status") != "COMPLETE"
                or report.get("valuation_status") not in DETAIL_VALUATION_STATUSES
            ):
                continue
            selected[ticker] = {
                "ticker": ticker,
                "issuer_name": report.get("name") or ticker,
                "valuation_as_of": report.get("valuation_as_of"),
                "status": report.get("status"),
                "current_price": None,
                "moat_score": None,
            }
        return [selected[ticker] for ticker in sorted(selected)]

    @staticmethod
    def _current_run_root(report_path: Path) -> Path:
        for parent in report_path.parents:
            if parent.name == "research-reports":
                return parent.parent
        raise ResearchArtifactNotFoundError(
            f"current report is outside a research-reports directory: {report_path}"
        )

    @staticmethod
    def _current_input_path(
        run_root: Path,
        cutoff: datetime,
        kind: str,
        ticker: str,
        configured: Any = None,
    ) -> Path:
        local = run_root / "date-inputs" / cutoff.date().isoformat() / kind / f"{ticker}.json"
        if local.is_file():
            return local
        if configured:
            configured_path = Path(str(configured))
            if configured_path.is_file():
                return configured_path
        raise ResearchArtifactNotFoundError(
            f"required current-report input is missing: {kind}/{ticker}.json"
        )

    @staticmethod
    def _current_selected_pack(
        run_root: Path,
        cutoff: datetime,
        ticker: str,
    ) -> dict[str, Any]:
        path = (
            run_root
            / "research-reports"
            / cutoff.date().isoformat()
            / "packs"
            / "selected"
            / f"KR-{cutoff.date().isoformat()}-{ticker}.json"
        )
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    @staticmethod
    def _segment_trends(
        selected_pack: dict[str, Any],
        cutoff: datetime,
    ) -> list[dict[str, Any]]:
        trends: dict[str, dict[str, Any]] = {}
        for excerpt in selected_pack.get("excerpts", []):
            if excerpt.get("source_role") != "DART_ORIGINAL":
                continue
            text = re.sub(r"\s+", " ", str(excerpt.get("text") or "")).strip()
            if "부문별로" not in text:
                continue
            clause = text.split("부문별로", 1)[1]
            for fragment in re.split(r"[,;]", clause):
                match = re.search(
                    r"(?P<name>.+?)(?:이|가|은|는)\s*"
                    r"(?P<value>\d+(?:\.\d+)?)%\s*(?P<move>증가|감소)",
                    fragment,
                )
                if not match:
                    continue
                name = re.sub(r"^.*?대비\s*", "", match.group("name")).strip(" -·")
                if not name or len(name) > 40:
                    continue
                magnitude = float(match.group("value"))
                move = match.group("move")
                trends[name] = {
                    "name": name,
                    "change_pct": magnitude if move == "증가" else -magnitude,
                    "direction": "positive" if move == "증가" else "negative",
                    "period": cutoff.date().isoformat(),
                    "metric_label": "전년 동기 대비 매출",
                    "source_document_id": str(excerpt.get("source_id") or "UNKNOWN"),
                }
        return list(trends.values())[:8]

    @staticmethod
    def _context_evidence(
        selected_pack: dict[str, Any],
        cutoff: datetime,
        ticker: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        definitions = (
            (
                "SEGMENT_TREND",
                lambda text: "부문별로" in text and "매출" in text,
                "공시된 사업부문별 매출 변화",
            ),
            (
                "FINANCIAL_RESULT",
                lambda text: "매출액은" in text and "영업이익" in text,
                "공시된 최근 매출과 영업이익 변화",
            ),
            (
                "RISK_EXPOSURE",
                lambda text: "위험에 노출" in text,
                "공시에 명시된 주요 재무위험 노출",
            ),
            (
                "BUSINESS_MODEL",
                lambda text: "수익은 주로" in text or "주요 원재료" in text,
                "공시에 명시된 사업·원가 구조",
            ),
            (
                "INDUSTRY_OUTLOOK",
                lambda text: any(
                    token in text
                    for token in (
                        "산업의 성장성",
                        "시장 성장",
                        "시장 규모",
                        "시장 수요",
                        "수요가 증가",
                        "수요가 감소",
                        "시황이 악화",
                        "시황이 개선",
                    )
                ),
                "공시에 제시된 산업 수요와 시장 환경",
            ),
            (
                "COMPETITIVE_DYNAMICS",
                lambda text: any(
                    token in text
                    for token in (
                        "핵심 경쟁 요소",
                        "경쟁의 패러다임",
                        "중요한 사업 경쟁력",
                        "기술 경쟁력",
                        "원가 경쟁력",
                        "시장점유율",
                        "진입장벽",
                    )
                ),
                "공시에 설명된 산업 경쟁요소와 기술 변화",
            ),
            (
                "RISK_MANAGEMENT",
                lambda text: (
                    "위험관리의 목적" in text[:180]
                    or "위험회피전략" in text[:180]
                    or "환율변동위험" in text[:180]
                    or "시장위험" in text[:180]
                    or "신용위험" in text[:180]
                ),
                "공시에 설명된 주요 위험과 관리 정책",
            ),
            (
                "PRODUCT_PORTFOLIO",
                lambda text: any(
                    token in text
                    for token in (
                        "주요 제품",
                        "제품 및 서비스",
                        "제품과 서비스",
                    )
                ),
                "공시에 설명된 주요 제품과 수요처",
            ),
        )
        available_at = {
            str(item.get("source_id")): item.get("available_at")
            for item in (selected_pack.get("source_assignment") or {}).get(
                "dart_documents", []
            )
        }
        selected: list[tuple[str, str, dict[str, Any], str]] = []
        selected_units: set[str] = set()
        excerpts = sorted(
            (
                item
                for item in selected_pack.get("excerpts", [])
                if item.get("source_role") == "DART_ORIGINAL"
            ),
            key=lambda item: str(
                available_at.get(str(item.get("source_id")))
                or item.get("available_at")
                or ""
            ),
            reverse=True,
        )
        for evidence_type, predicate, fact in definitions:
            match = next(
                (
                    (item, re.sub(r"\s+", " ", str(item.get("text") or "")).strip())
                    for item in excerpts
                    if str(item.get("unit_id") or "") not in selected_units
                    if predicate(re.sub(r"\s+", " ", str(item.get("text") or "")))
                ),
                None,
            )
            if match:
                item, quote = match
                selected.append((evidence_type, fact, item, quote[:700]))
                selected_units.add(str(item.get("unit_id") or ""))
            if len(selected) >= 6:
                break

        # Every detail-capable current report has a PIT-selected DART pack.  If
        # none of the named categories match, surface the most business-relevant
        # excerpt as neutral disclosure context instead of presenting an empty
        # ledger.  This is deliberately not promoted to moat support.
        if not selected and excerpts:
            relevance_tokens = (
                "산업",
                "시장",
                "제품",
                "고객",
                "수요",
                "경쟁",
                "생산",
                "기술",
                "매출",
                "원재료",
                "위험",
            )
            boilerplate_tokens = (
                "회계정책",
                "리스기간",
                "퇴직급여",
                "법인세",
                "공정가치로 측정",
            )

            def relevance(item: dict[str, Any]) -> tuple[int, int]:
                text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
                score = sum(2 for token in relevance_tokens if token in text)
                score -= sum(3 for token in boilerplate_tokens if token in text)
                return score, min(len(text), 2000)

            item = max(excerpts, key=relevance)
            quote = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
            if quote:
                selected.append(
                    (
                        "DISCLOSURE_CONTEXT",
                        "PIT 선별 공시의 사업·재무 문맥",
                        item,
                        quote[:700],
                    )
                )
        cards: list[dict[str, Any]] = []
        ledger: list[dict[str, Any]] = []
        for index, (evidence_type, fact, item, quote) in enumerate(selected, start=1):
            evidence_id = f"{ticker}:CTX{index:02d}"
            source_id = str(item.get("source_id") or "UNKNOWN")
            cards.append(
                {
                    "evidence_id": evidence_id,
                    "evidence_type": evidence_type,
                    "fact": fact,
                    "mechanism": ["DIRECT_DISCLOSURE_CONTEXT"],
                    "direction": "NEUTRAL",
                    "strength": 1.0,
                    "reliability": 1.0,
                    "source_type": "DART",
                    "raw_quote": quote,
                    "period": cutoff.date().isoformat(),
                    "dcf_links": [],
                }
            )
            ledger.append(
                {
                    "evidence_id": evidence_id,
                    "source_document_id": source_id,
                    "source_available_at": available_at.get(source_id)
                    or cutoff.isoformat(),
                }
            )
        return cards, ledger

    @classmethod
    def _current_financial_snapshot(
        cls,
        dcf_input: dict[str, Any],
        selected_pack: dict[str, Any],
        cutoff: datetime,
    ) -> dict[str, Any]:
        annual = sorted(
            dcf_input.get("annual_history", []),
            key=lambda item: int(item.get("year") or 0),
        )
        revenue_points = [
            {
                "period": f"{int(item['year'])}-12-31",
                "period_basis": "FY",
                "value": item.get("metrics", {}).get("revenue"),
                "unit": "KRW",
            }
            for item in annual
            if item.get("year") and item.get("metrics", {}).get("revenue") is not None
        ]
        derived: list[dict[str, Any]] = []
        if len(revenue_points) >= 2:
            first = _float(revenue_points[0]["value"])
            last = _float(revenue_points[-1]["value"])
            years = int(annual[-1]["year"]) - int(annual[0]["year"])
            if first > 0 and last > 0 and years > 0:
                derived.append(
                    {
                        "name": "REVENUE_CAGR",
                        "period": revenue_points[-1]["period"],
                        "value": (last / first) ** (1 / years) - 1,
                        "unit": "RATIO",
                    }
                )
        metrics = dcf_input.get("metrics") or {}
        revenue = _float(metrics.get("revenue"))
        ebit = _float(metrics.get("ebit"))
        if revenue > 0:
            derived.append(
                {
                    "name": "EBIT_MARGIN",
                    "period": str((dcf_input.get("pit") or {}).get("latest_report_period") or "PIT"),
                    "value": ebit / revenue,
                    "unit": "RATIO",
                }
            )
        return {
            "series": [{"concept": "REVENUE", "points": revenue_points}],
            "breakdowns": [],
            "derived_metrics": derived,
            "segment_trends": cls._segment_trends(selected_pack, cutoff),
            "current_report_adapter": True,
        }

    def _current_artifacts(self, ticker: str, *, as_of=None) -> ResearchArtifacts:
        report_path, report = self.repository.load_current_report_entry(ticker, as_of=as_of)
        valuation = report.get("valuation") or {}
        if (
            report.get("status") != "COMPLETE"
            or valuation.get("status") not in DETAIL_VALUATION_STATUSES
        ):
            reasons = ", ".join(str(item) for item in report.get("source_exclusions", []))
            reason_suffix = f": {reasons}" if reasons else ""
            raise ValueError(
                f"{ticker} 보고서는 존재하지만 현재 상세 가치평가 형식을 적용할 수 없습니다 "
                f"({valuation.get('status', 'UNAVAILABLE')}{reason_suffix})"
            )

        normalized = str(report.get("ticker") or ticker).upper()
        cutoff = datetime.fromisoformat(str(report["cutoff"]))
        run_root = self._current_run_root(report_path)
        assumptions_path = self._current_input_path(
            run_root,
            cutoff,
            "assumptions",
            normalized,
            valuation.get("assumptions_path"),
        )
        dcf_input_path = self._current_input_path(
            run_root,
            cutoff,
            "dcf-inputs",
            normalized,
        )
        assumptions = self.repository._read(assumptions_path)
        dcf_input = self.repository._read(dcf_input_path)
        selected_pack = self._current_selected_pack(run_root, cutoff, normalized)
        context_cards, context_ledger = self._context_evidence(
            selected_pack,
            cutoff,
            normalized,
        )
        overlay = report.get("evidence_overlay") or {}
        claims = overlay.get("validated_claims") or []
        supportive = [item for item in claims if item.get("direction") == "SUPPORTIVE"]
        evidence_ids = [
            str(item.get("judgment_id") or f"{normalized}:E{index:03d}")
            for index, item in enumerate(claims, start=1)
        ]
        claim_pairs = list(zip(evidence_ids, claims))
        mechanisms: list[dict[str, Any]] = []
        for axis in sorted({str(item.get("axis") or "OTHER_MOAT") for item in supportive}):
            matching = [
                (evidence_id, item)
                for evidence_id, item in claim_pairs
                if item.get("direction") == "SUPPORTIVE"
                and str(item.get("axis") or "OTHER_MOAT") == axis
            ]
            mechanisms.append(
                {
                    "evidence_type": axis,
                    "score": round(
                        sum(_float(item.get("confidence"), 0.5) for _, item in matching)
                        / max(len(matching), 1)
                        * 10,
                        2,
                    ),
                    "evidence_ids": [evidence_id for evidence_id, _ in matching],
                }
            )
        erosive_ids = [
            evidence_id
            for evidence_id, item in claim_pairs
            if item.get("direction") == "EROSIVE"
        ]
        original_count = int(
            (overlay.get("anonymization_audit") or {}).get("original_claim_count")
            or len(claims)
        )
        coverage = len(claims) / original_count if original_count else 0.0
        source_ids = list(
            dict.fromkeys(
                [str(item.get("source_id")) for item in claims if item.get("source_id")]
                + [
                    str(item.get("source_document_id"))
                    for item in context_ledger
                    if item.get("source_document_id")
                ]
            )
        )
        signature = hashlib.sha256(
            json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        result = {
            "ticker": normalized,
            "issuer_name": str(report.get("name") or normalized),
            "status": "COMPLETE",
            "run_signature": signature,
            "source_document_ids": source_ids,
            "evidence_count": len(claims) + len(context_cards),
            "moat_score": {},
            "dcf": {
                "fair_value_per_share": valuation.get("fair_value_per_share"),
                "assumption_confidence": valuation.get("assumption_confidence"),
                "terminal_value_share": valuation.get("terminal_value_share"),
            },
            "current_price": valuation.get("current_price"),
            "price_as_of": report.get("price_as_of"),
            "valuation_as_of": report.get("cutoff"),
            "runner_version": "kr-all-current-report/1",
            "valuation_method": "FCFF DCF",
            "valuation_priority": "DCF",
            "expectation_gap_status": "NOT_AVAILABLE_FOR_REPORT_CUTOFF",
        }
        dossier = {
            "evidence": (
                [
                    {
                        "evidence_id": evidence_id,
                        "evidence_type": str(item.get("axis") or "OTHER_MOAT"),
                        "fact": str(item.get("claim") or ""),
                        "mechanism": [str(item.get("axis") or "OTHER_MOAT")],
                        "direction": (
                            "MOAT_POSITIVE"
                            if item.get("direction") == "SUPPORTIVE"
                            else "MOAT_NEGATIVE"
                            if item.get("direction") == "EROSIVE"
                            else "NEUTRAL"
                        ),
                        "strength": _float(item.get("confidence"), 0.5),
                        "reliability": _float(item.get("confidence"), 0.5),
                        "source_type": "DART",
                        "raw_quote": str(item.get("exact_quote") or ""),
                        "period": cutoff.date().isoformat(),
                        "dcf_links": [],
                    }
                    for evidence_id, item in claim_pairs
                ]
                + context_cards
            )
        }
        moat_score = {
            "score_status": "UNSCORED_CURRENT_OVERLAY",
            "economic_moat_score": 0,
            "durability": "UNSCORED",
            "model_confidence": max(
                (_float(item.get("confidence")) for item in claims),
                default=0.0,
            ),
            "document_coverage": {"moat_evidence_coverage": min(coverage, 1.0)},
            "mechanisms": mechanisms,
            "counterevidence_ids": erosive_ids,
            "context_evidence_ids": [
                str(item["evidence_id"]) for item in context_cards
            ],
        }
        manifest = {
            "run_id": str(overlay.get("pack_id") or signature[:16]),
            "model": str((report.get("model_contract") or {}).get("main_model") or "unknown"),
            "created_at": report.get("cutoff"),
            "evidence_cutoff": report.get("cutoff"),
            "prompt_version": "kr-all-current-report/1",
            "parser_version": "kr-all-current-report/1",
        }
        ledger = {
            "records": (
                [
                    {
                        "evidence_id": evidence_id,
                        "source_document_id": item.get("source_id"),
                        "source_available_at": item.get("available_at")
                        or report.get("cutoff"),
                    }
                    for evidence_id, item in claim_pairs
                ]
                + context_ledger
            )
        }
        return ResearchArtifacts(
            directory=report_path.parent,
            result=result,
            dossier=dossier,
            moat_score=moat_score,
            dcf_assumptions=assumptions,
            financial_snapshot=self._current_financial_snapshot(
                dcf_input,
                selected_pack,
                cutoff,
            ),
            run_manifest=manifest,
            evidence_ledger=ledger,
        )

    @staticmethod
    def _meta(artifacts: ResearchArtifacts, warnings: list[str]) -> ReportMeta:
        result = artifacts.result
        manifest = artifacts.run_manifest
        coverage = _float(
            artifacts.moat_score.get("document_coverage", {}).get("moat_evidence_coverage")
        )
        if coverage >= 0.7:
            data_grade = "RESEARCH"
        elif coverage > 0 or (
            artifacts.financial_snapshot.get("current_report_adapter")
            and int(result.get("evidence_count") or 0) > 0
        ):
            data_grade = "LIMITED"
        else:
            data_grade = "INSUFFICIENT"
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
        segment_trends = [
            SegmentTrend.model_validate(item)
            for item in snapshot.get("segment_trends", [])
        ]
        geography = self._geography(snapshot)
        top_names = [item.name for item in mix[:3]]
        business_model = (
            f"{', '.join(top_names)}를 중심으로 매출을 만드는 다각화 영업모델입니다."
            if top_names
            else (
                f"{', '.join(item.name for item in segment_trends[:4])} 등 "
                "공시에 식별된 사업부문을 통해 매출을 창출합니다."
            )
            if segment_trends
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
        segment_names = {item.name for item in segment_trends}
        industry = (
            "디지털 플랫폼·인터넷 서비스"
            if digital_tokens.intersection(top_names)
            else "반도체·디스플레이·소비자전자"
            if {"DX 부문", "DS 부문", "SDC", "Harman"}.intersection(segment_names)
            else "다각화 영업기업"
        )
        if snapshot.get("current_report_adapter"):
            business_summary = (
                f"{artifacts.result['issuer_name']}의 시점고정 공시 재무, 사업부문 변화와 "
                "FCFF DCF를 연결한 분석입니다. 검증되지 않은 해자 주장은 별도로 추정하지 않습니다."
            )
        else:
            business_summary = (
                f"{artifacts.result['issuer_name']}의 공시 원문, 재무 세그먼트와 해자 근거를 "
                "사업→경쟁우위→가치의 흐름으로 연결한 시점고정 분석입니다."
            )
        return CompanyProfile(
            ticker=str(artifacts.result["ticker"]),
            issuer_id=artifacts.result.get("issuer_id"),
            issuer_name=str(artifacts.result["issuer_name"]),
            business_summary=business_summary,
            business_model=business_model,
            industry_label=industry,
            revenue_mix=mix,
            segment_trends=segment_trends,
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
        selected_ids.extend(moat.get("context_evidence_ids", [])[:6])
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
        rating = (
            "INSUFFICIENT"
            if score_data.get("score_status") == "UNSCORED_CURRENT_OVERLAY"
            else "WIDE"
            if score >= 7.5
            else "NARROW"
            if score >= 5
            else "NONE"
        )
        primary = [axis.label for axis in axes if axis.status != "NOT_OBSERVED"][:3]
        coverage = _float(score_data.get("document_coverage", {}).get("moat_evidence_coverage"))
        if primary:
            summary = (
                f"핵심 해자 원천은 {', '.join(primary)}입니다. "
                f"공시 기반 내구성 평가는 {score_data.get('durability', 'UNKNOWN')}이며, "
                "확인되지 않은 축은 점수로 추정하지 않았습니다."
            )
        else:
            summary = (
                "현재 검증된 해자 원천이 없습니다. 직접 공시 맥락은 제공하되, "
                "검증되지 않은 전환비용·네트워크 효과·가격 결정력은 점수로 추정하지 않았습니다."
            )
        return MoatAnalysis(
            score=score,
            rating=rating,
            durability=str(score_data.get("durability", "UNKNOWN")),
            confidence=_float(score_data.get("model_confidence")),
            evidence_coverage=coverage,
            primary_sources=primary,
            summary=summary,
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
                description=(
                    "검증된 경쟁우위 근거가 있는지와 그 지속성을 원문 기준으로 추적합니다."
                ),
                evidence_ids=competition,
            ),
            IndustryForce(
                id="cycle",
                label="사이클",
                status="사업 믹스 분산",
                tone="neutral",
                description=(
                    f"{len(company.revenue_mix) or len(company.segment_trends)}개 공시 사업 축을 "
                    "함께 추적해 단일 사업 변동의 영향을 확인합니다."
                ),
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
        primary_moat = moat.primary_sources[0] if moat.primary_sources else "경쟁우위 검증 필요"
        if company.industry_label == "디지털 플랫폼·인터넷 서비스":
            structure_summary = (
                "디지털 플랫폼·인터넷 서비스의 기업가치는 사용자·거래·콘텐츠 연결과 "
                "수익화 효율, 규제 비용의 균형에 좌우됩니다."
            )
        elif company.segment_trends:
            structure_summary = (
                f"{company.industry_label}의 기업가치는 "
                f"{', '.join(item.name for item in company.segment_trends[:4])}의 "
                "수요, 가격, 원가와 투자 효율의 조합에 좌우됩니다."
            )
        else:
            structure_summary = (
                f"{company.industry_label}의 기업가치는 매출 성장, 정상 마진, "
                "재투자 효율과 자본비용의 균형에 좌우됩니다."
            )
        return IndustryAnalysis(
            structure_summary=structure_summary,
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
        is_current_adapter = bool(
            artifacts.financial_snapshot.get("current_report_adapter")
        )
        if not is_current_adapter:
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
            "상대적으로 높은 기대 반영" if percentile is not None else
            "동일 cutoff 비교표본 미연결" if is_current_adapter else "표본 부족"
        )
        return EconomicValueScore(
            percentile=percentile,
            label=label,
            reference_class=(
                "동일 cutoff 전 종목 DCF 비교표본"
                if is_current_adapter
                else "동일 시점 MoatRader 완료 보고서"
            ),
            sample_size=len(gaps),
            confidence=_confidence_label(confidence_value),
            fragility=fragility,
            coverage=_float(
                artifacts.moat_score.get("document_coverage", {}).get("moat_evidence_coverage")
            ),
            caveat=(
                "동일 cutoff 전 종목 DCF 분포가 API 비교표본으로 연결되지 않아 percentile을 "
                "표시하지 않습니다. 적정가는 위 FCFF DCF 절대가치 범위를 우선 확인하십시오."
                if is_current_adapter
                else "이 percentile은 독립 알파가 아니라 모델별 가치격차를 현재 사용 가능한 "
                "완료 표본 안에서 정규화한 진단치입니다."
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
        valuation_method = str(artifacts.result.get("valuation_method") or "FCFF")
        if artifacts.result.get("valuation_priority") == "DCF":
            rationale = (
                "동일 cutoff의 Expectation GAP 적정가 산출물이 없어 결정론적 FCFF DCF를 "
                "1순위 적정가로 사용합니다. Reverse DCF는 적정가가 아니라 현재 가격의 "
                "내재 기대를 보여주는 보조 진단입니다."
            )
        else:
            rationale = (
                "현재 완료 산출물은 비금융 영업기업용 FCFF와 명시적 재투자 가정을 사용합니다."
            )
        return ValuationAnalysis(
            route=ModelRoute(
                primary_model=valuation_method,
                base_period=assumptions.base_period,
                rationale=rationale,
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
        concern = next(
            (item.fact for item in evidence if item.direction == "negative"),
            next(
                (item.fact for item in evidence if item.evidence_type == "RISK_EXPOSURE"),
                "경쟁우위 지속성과 정상 수익성에 대한 불확실성",
            ),
        )
        method = valuation.route.primary_model
        if moat.primary_sources:
            summary = (
                f"{surface.headline} {company.issuer_name}의 "
                f"{moat.primary_sources[0]} 근거가 이 가정을 지지하는지가 핵심입니다."
            )
        else:
            summary = (
                f"{surface.headline} 현재 검증된 해자 지지 근거가 없으므로, "
                f"{sensitivity.primary_driver_label}과 현금흐름 전환을 후속 공시에서 확인해야 합니다."
            )
        return PriceExplanation(
            headline=(
                f"현재 가격은 {method} 1순위 적정가보다 {abs(gap):.1f}% {direction} 수준입니다."
            ),
            summary=summary,
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
        neutral = [item for item in evidence if item.direction == "neutral"]
        risk_context = [
            item
            for item in neutral
            if item.evidence_type in {"RISK_EXPOSURE", "RISK_MANAGEMENT"}
        ]
        assumption_context = [item for item in neutral if item not in risk_context]
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
                detail=(
                    f"구조적 반대 근거 {len(negative)}개를 계속 추적해야 합니다."
                    if negative
                    else "검증된 반대 근거가 없지만, 이는 위험 부재를 뜻하지 않습니다."
                ),
                evidence_ids=[item.id for item in negative[:3]],
            ),
            ThesisMonitorItem(
                id="valuation",
                label="Valuation",
                status=valuation_status,
                tone="positive" if valuation_status == "INTACT" else "warning",
                detail=(
                    f"{valuation.route.primary_model} 적정가 대비 "
                    f"{valuation.base_value_gap_pct:+.1f}%"
                ),
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
        if positive:
            core_thesis = (
                f"{company.issuer_name}의 {', '.join(moat.primary_sources[:2])} 근거가 "
                f"{sensitivity.primary_driver_label}을 방어하면 현재 가격과 "
                f"{valuation.route.primary_model} 적정가의 간극이 축소될 수 있습니다."
            )
        else:
            core_thesis = (
                f"검증된 해자 지지 근거가 아직 없습니다. {company.issuer_name}의 "
                f"{valuation.route.primary_model} 적정가가 성립하려면 "
                f"{sensitivity.primary_driver_label} 가정과 현금흐름 전환을 후속 공시에서 확인해야 합니다."
            )
        return ThesisAnalysis(
            core_thesis=core_thesis,
            supporting_evidence_ids=[item.id for item in positive[:5]],
            context_evidence_ids=[item.id for item in assumption_context[:4]],
            breakers=[item.fact for item in negative[:4]],
            breaker_evidence_ids=[item.id for item in negative[:4]],
            risk_context_evidence_ids=[item.id for item in risk_context[:3]],
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
                f"이 응답은 종목 추천이나 독립 알파 신호가 아니라, "
                f"{valuation.route.primary_model} 적정가와 cutoff 원문 근거를 연결한 진단 도구입니다."
            ),
            disclaimer="투자 판단과 손익의 책임은 사용자에게 있으며, 가정과 원문을 직접 검토해야 합니다.",
        )
