from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from moatrader.canonical.ids import stable_id
from moatrader.canonical.models import SourceType, StatementType
from moatrader.evidence.models import (
    AtomicEvidenceExtraction,
    AtomicEvidenceJudgment,
    AtomicMoatRole,
    DcfLink,
    CanonicalClaimSignature,
    ClaimCluster,
    CitedSummaryClaim,
    EconomicScope,
    EvidenceCard,
    EvidenceCluster,
    EvidenceDirection,
    EvidenceMetric,
    EvidenceRelation,
    EvidenceRelationType,
    EvidenceType,
    ForwardDriverCard,
    ForwardDriverType,
    OUTCOME_CORROBORATION_TYPES,
    SectionSummary,
    STRUCTURAL_MOAT_TYPES,
)
from moatrader.semantic.chunker import SemanticChunk


_NUMBER_RE = re.compile(r"[-+]?\d[\d,.]*(?:%|개월|년|배|원)?")
_DEMAND_RE = re.compile(
    r"시장\s*(?:규모|성장)|수요|환자.{0,12}(?:증가|성장)|환자\s*수|방문객|CAGR|TAM|market\s+(?:size|growth|demand)",
    re.IGNORECASE,
)
_SHARE_RE = re.compile(r"점유율|시장\s*점유|market\s*share|share\s+of\s+market", re.IGNORECASE)
_RECURRING_CATEGORY_RE = re.compile(
    r"재시술|반복\s*시술|시술\s*주기|\d+\s*[~-]\s*\d+\s*개월|repeat(?:ed)?\s+treatment",
    re.IGNORECASE,
)
_RETENTION_RE = re.compile(
    r"고객\s*(?:유지|이탈률|재구매율)|갱신률|renewal|retention|churn|switching|전환\s*비용",
    re.IGNORECASE,
)
_MARKET_SHARE_ANCHOR_RE = re.compile(
    r"(?:시장\s*)?점유율|시장\s*점유|M\s*/\s*S|market\s*share|share\s+of\s+(?:the\s+)?market|"
    r"(?:국내|글로벌|시장|업계|제품|category|market)\s*(?:내\s*)?(?:#\s*)?(?:1위|leader)|"
    r"#\s*1|number[-\s]?one|market\s+leader|"
    r"(?:\d+(?:\.\d+)?\s*%).{0,50}(?:시장|market)|(?:시장|market).{0,50}(?:\d+(?:\.\d+)?\s*%)",
    re.IGNORECASE,
)
_CUSTOMER_RETENTION_ANCHOR_RE = re.compile(
    r"고객\s*(?:유지|이탈)|유지율|이탈률|재구매|재수주|반복\s*(?:구매|주문|수주)|"
    r"재계약|계약\s*갱신|갱신률|renewal|retention|churn|reorder|"
    r"repeat(?:ed)?\s+(?:purchase|order)|same\s+customer|"
    r"installed\s+base.{0,50}(?:reuse|service|maintenance|upgrade)|"
    r"설치\s*(?:기반|대수).{0,50}(?:재사용|서비스|유지보수|교체|업그레이드)",
    re.IGNORECASE,
)
_MARGIN_OR_PROFITABILITY_ANCHOR_RE = re.compile(
    r"마진|이익률|수익성|수익\s*실현|흑자|"
    r"gross\s*(?:margin|%)|operating\s*(?:margin|%)|net\s*(?:margin|%)|"
    r"profitability|profitable",
    re.IGNORECASE,
)
_EXPLICIT_MULTIPERIOD_RE = re.compile(
    r"(?:[3-9]|[1-9]\d+)\s*(?:개\s*)?(?:년|분기)|"
    r"(?:three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d+)[-\s]+(?:years?|quarters?)|"
    r"연속|consecutive",
    re.IGNORECASE,
)
_PERIOD_TOKEN_RE = re.compile(
    r"(?:19|20)\d{2}(?:[.\-/\s]?(?:[1-4]Q|Q[1-4]|[1-4]\s*분기))?|[‘'’]?\d{2}[.]?[1-4]Q",
    re.IGNORECASE,
)
_COST_PROCESS_ANCHOR_RE = re.compile(
    r"비용|원가|공정|단계|시간|수율|생산성|cost|unit\s+cost|steps?|process|time|yield|productivity",
    re.IGNORECASE,
)
_DIRECT_COMPARISON_RE = re.compile(
    r"대비|비교|절감|감축|줄(?:임|여)|낮은|적은|단축|피하|극복|"
    r"versus|vs\.?|compared\s+(?:with|to)|avoid|reduce|lower|less|fewer|higher\s+yield|alternative",
    re.IGNORECASE,
)
_COUNTER_EROSION_RE = re.compile(
    r"불안정|변동|등락|악화|침식|훼손|축소|약화|"
    r"volatil|instabil|unstable|deterior|erosion|erod|weaken|alternat|"
    r"direction\s+revers|reverses?\s+(?:its\s+)?direction|"
    r"declin(?:e|ed|ing).{0,30}(?:share|margin|retention|renewal)",
    re.IGNORECASE,
)
_FORWARD_LANGUAGE_RE = re.compile(
    r"전망|계획|예상|기대|목표|가이던스|추정|will|expect|plan|target|forecast|guidance",
    re.IGNORECASE,
)
_LLM_NUMERIC_FRAGMENT_RE = re.compile(r"\d[\d,.:/%+\-~–—?]*")

_MOAT_COUNTER_TYPES: frozenset[EvidenceType] = (
    STRUCTURAL_MOAT_TYPES
    | OUTCOME_CORROBORATION_TYPES
    | {
        EvidenceType.COMPETITIVE_THREAT,
        EvidenceType.CUSTOMER_CONCENTRATION,
        EvidenceType.SUBSTITUTION_RISK,
        EvidenceType.TECHNOLOGY_RISK,
        EvidenceType.CAPITAL_INTENSITY,
    }
)
_NON_MOAT_RELEVANT_TYPES: frozenset[EvidenceType] = frozenset(
    {
        EvidenceType.MARKET_DEMAND,
        EvidenceType.CATEGORY_RECURRING_DEMAND,
        EvidenceType.CAPACITY_UTILIZATION,
        EvidenceType.EXPORT_MIX,
        EvidenceType.OPERATING_DRIVER,
    }
)
_CANONICAL_MECHANISM_PHRASE: dict[EvidenceType, str] = {
    EvidenceType.SWITCHING_COST: "observable switching friction",
    EvidenceType.NETWORK_EFFECT: "observable network-effect barrier",
    EvidenceType.COST_ADVANTAGE: "observable persistent cost barrier",
    EvidenceType.INTANGIBLE_ASSET: "observable protected intangible barrier",
    EvidenceType.SCALE_ADVANTAGE: "observable scale-based entry barrier",
    EvidenceType.REGULATORY_BARRIER: "observable regulatory entry barrier",
}


def atomic_moat_role(extraction: AtomicEvidenceExtraction) -> AtomicMoatRole:
    """Return a role only when role, subtype, direction, and scope agree.

    Legacy callers may omit ``role``; those values are inferred from the old
    type/direction contract.  New model output must explicitly select a
    mutually compatible role.  Invalid combinations fail closed to ``NONE``.
    """

    if not extraction.is_investment_relevant:
        return AtomicMoatRole.NONE
    if extraction.economic_scope not in {EconomicScope.COMPANY, EconomicScope.SEGMENT}:
        return AtomicMoatRole.NONE
    explicit = extraction.moat_role
    inferred = AtomicMoatRole.NONE
    if (
        extraction.evidence_type in STRUCTURAL_MOAT_TYPES
        and extraction.direction == EvidenceDirection.MOAT_POSITIVE
    ):
        inferred = AtomicMoatRole.MECHANISM
    elif (
        extraction.evidence_type in OUTCOME_CORROBORATION_TYPES
        and extraction.direction == EvidenceDirection.MOAT_POSITIVE
    ):
        inferred = AtomicMoatRole.OUTCOME
    elif (
        extraction.evidence_type in _MOAT_COUNTER_TYPES
        and extraction.direction == EvidenceDirection.MOAT_NEGATIVE
    ):
        inferred = AtomicMoatRole.COUNTER
    if explicit is None:
        return inferred
    return explicit if explicit == inferred else AtomicMoatRole.NONE


def observable_atomic_anchor_violation(
    extraction: AtomicEvidenceExtraction,
    source_text: str,
) -> str | None:
    """Return a fail-closed reason when selected MOAT routes lack an observable anchor.

    The LLM chooses the semantic route, but Python enforces the narrow claims
    that repeatedly caused false precision in IR pages.  This is deliberately
    not a keyword classifier: it can reject an unsupported route, but it never
    promotes a ``NONE`` vote into score-bearing evidence.
    """

    role = atomic_moat_role(extraction)
    if role == AtomicMoatRole.NONE:
        return None
    text = unicodedata.normalize("NFKC", source_text or "")
    evidence_type = extraction.evidence_type
    if evidence_type == EvidenceType.MARKET_SHARE and not _MARKET_SHARE_ANCHOR_RE.search(text):
        return "MARKET_SHARE_REQUIRES_ISSUER_SHARE_OR_RANK"
    if (
        evidence_type == EvidenceType.CUSTOMER_RETENTION
        and not _CUSTOMER_RETENTION_ANCHOR_RE.search(text)
    ):
        return "CUSTOMER_RETENTION_REQUIRES_DIRECT_RETENTION_BEHAVIOR"
    if evidence_type == EvidenceType.MARGIN_STABILITY:
        periods = {match.group(0).casefold() for match in _PERIOD_TOKEN_RE.finditer(text)}
        multi_period = bool(_EXPLICIT_MULTIPERIOD_RE.search(text)) or len(periods) >= 3
        if not (_MARGIN_OR_PROFITABILITY_ANCHOR_RE.search(text) and multi_period):
            return "MARGIN_STABILITY_REQUIRES_MULTI_PERIOD_PROFITABILITY"
    if evidence_type == EvidenceType.COST_ADVANTAGE and not (
        _COST_PROCESS_ANCHOR_RE.search(text) and _DIRECT_COMPARISON_RE.search(text)
    ):
        return "COST_ADVANTAGE_REQUIRES_DIRECT_PROCESS_COMPARISON"
    if role == AtomicMoatRole.COUNTER and not _COUNTER_EROSION_RE.search(text):
        return "COUNTER_REQUIRES_EXPLICIT_MOAT_ANCHOR_EROSION"
    return None


def _canonical_atomic_fact(source_text: str) -> str:
    fact = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", source_text)).strip()
    if not fact:
        return "No stable investment-relevant evidence"
    return fact[:1_200].rstrip()


def atomic_classification_signature(
    extraction: AtomicEvidenceExtraction,
) -> tuple[str, bool, str, str, str]:
    """Fields that can change MOAT routing or score eligibility."""

    return (
        atomic_moat_role(extraction).value,
        extraction.is_investment_relevant,
        extraction.evidence_type.value,
        extraction.direction.value,
        extraction.economic_scope.value,
    )


def atomic_routing_signature(
    extraction: AtomicEvidenceExtraction,
) -> tuple[str, bool, str, str]:
    """Economic route fields; scope is calibrated after route consensus."""

    signature = atomic_classification_signature(extraction)
    return signature[:4]


def build_atomic_classification_consensus(
    votes: list[AtomicEvidenceExtraction],
    *,
    source_text: str,
) -> tuple[AtomicEvidenceExtraction, dict[str, object]]:
    """Reduce independent atomic votes with a strict majority.

    No majority means no score-bearing evidence.  Free-form model wording is
    never selected by the vote: the final fact and claim identity are derived
    from the frozen source plus the winning closed-set label.
    """

    if not votes:
        raise ValueError("atomic classification consensus requires votes")
    signatures = [atomic_classification_signature(vote) for vote in votes]
    routing_signatures = [atomic_routing_signature(vote) for vote in votes]
    route_counts = Counter(routing_signatures)
    winning_route, winning_count = min(
        route_counts.items(), key=lambda item: (-item[1], item[0])
    )
    quorum = len(votes) // 2 + 1
    agreed = winning_count >= quorum
    if agreed:
        role_value, relevant, type_value, direction_value = winning_route
        winning_votes = [
            vote
            for vote, route in zip(votes, routing_signatures, strict=True)
            if route == winning_route
        ]
        scope_counts = Counter(vote.economic_scope for vote in winning_votes)
        scope, _scope_count = min(
            scope_counts.items(),
            key=lambda item: (-item[1], item[0].value),
        )
        role = AtomicMoatRole(role_value)
        evidence_type = EvidenceType(type_value)
        direction = EvidenceDirection(direction_value)
        fact = (
            _canonical_atomic_fact(source_text)
            if relevant
            else "No stable investment-relevant evidence"
        )
        selected = AtomicEvidenceExtraction(
            is_investment_relevant=relevant,
            moat_role=role,
            evidence_type=evidence_type,
            direction=direction,
            fact=fact,
            mechanism=(
                [_CANONICAL_MECHANISM_PHRASE[evidence_type]]
                if role == AtomicMoatRole.MECHANISM
                else []
            ),
            economic_scope=scope,
            claim_subject=(
                "company"
                if scope == EconomicScope.COMPANY
                else "segment"
                if scope == EconomicScope.SEGMENT
                else scope.value.casefold()
            ),
            claim_predicate=evidence_type.value.casefold(),
        )
        status = "CONSENSUS"
    else:
        scope_counts = Counter()
        selected = AtomicEvidenceExtraction(
            is_investment_relevant=False,
            moat_role=AtomicMoatRole.NONE,
            evidence_type=EvidenceType.OTHER,
            direction=EvidenceDirection.NEUTRAL,
            fact="No stable investment-relevant evidence",
            economic_scope=EconomicScope.COMPANY,
        )
        status = "NO_CONSENSUS_FAIL_CLOSED"
    diagnostics: dict[str, object] = {
        "schema_version": "atomic-classification-consensus/1",
        "vote_count": len(votes),
        "quorum": quorum,
        "status": status,
        "winning_vote_count": winning_count,
        "agreement_rate": winning_count / len(votes),
        "winning_route": list(winning_route) if agreed else None,
        "winning_signature": (
            list(atomic_classification_signature(selected)) if agreed else None
        ),
        "vote_signatures": [list(signature) for signature in signatures],
        "vote_routing_signatures": [list(signature) for signature in routing_signatures],
        "signature_counts": [
            {"signature": list(signature), "count": count}
            for signature, count in sorted(Counter(signatures).items())
        ],
        "routing_signature_counts": [
            {"signature": list(signature), "count": count}
            for signature, count in sorted(route_counts.items())
        ],
        "winning_scope_counts": [
            {"scope": scope.value, "count": count}
            for scope, count in sorted(scope_counts.items(), key=lambda item: item[0].value)
        ],
    }
    return selected, diagnostics


def normalize_atomic_extraction(
    extraction: AtomicEvidenceExtraction,
    *,
    source_text: str | None = None,
) -> tuple[AtomicEvidenceExtraction, list[str]]:
    """Keep free LLM prose qualitative; Python owns source numbers/periods.

    Atomic raw quotes, metrics and periods are hydrated from the canonical
    source after this step, so removing digits from model-authored prose loses
    no auditable numeric evidence and prevents one malformed suffix from
    failing an entire company.
    """

    actions: list[str] = []

    def qualitative(value: str | None, field: str) -> str | None:
        if value is None:
            return None
        cleaned = _LLM_NUMERIC_FRAGMENT_RE.sub("", value)
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:-~")
        if cleaned != value:
            actions.append(f"REMOVE_NUMERIC_PROSE:{field}")
        return cleaned or None

    fact = qualitative(extraction.fact, "fact") or "Grounded canonical source evidence."
    mechanisms = [
        cleaned
        for index, value in enumerate(extraction.mechanism)
        if (cleaned := qualitative(value, f"mechanism[{index}]") or "")
    ]
    role = atomic_moat_role(extraction)
    relevant = extraction.is_investment_relevant
    evidence_type = extraction.evidence_type
    direction = extraction.direction
    scope = extraction.economic_scope
    if source_text is not None:
        anchor_violation = observable_atomic_anchor_violation(extraction, source_text)
        if anchor_violation is not None:
            relevant = False
            actions.append(f"FAIL_CLOSED_OBSERVABLE_ANCHOR:{anchor_violation}")
    if not relevant:
        role = AtomicMoatRole.NONE
        evidence_type = EvidenceType.OTHER
        direction = EvidenceDirection.NEUTRAL
        scope = EconomicScope.COMPANY
        mechanisms = []
        actions.append("CANONICALIZE_IRRELEVANT_LABEL")
    elif role == AtomicMoatRole.NONE:
        if evidence_type not in _NON_MOAT_RELEVANT_TYPES:
            relevant = False
            evidence_type = EvidenceType.OTHER
            scope = EconomicScope.COMPANY
            actions.append("FAIL_CLOSED_INVALID_OR_NONE_MOAT_ROLE")
        else:
            actions.append("PRESERVE_EXPLICIT_NON_MOAT_DCF_DRIVER")
        direction = EvidenceDirection.NEUTRAL
        mechanisms = []
    else:
        expected_direction = (
            EvidenceDirection.MOAT_NEGATIVE
            if role == AtomicMoatRole.COUNTER
            else EvidenceDirection.MOAT_POSITIVE
        )
        if direction != expected_direction:
            actions.append(f"CANONICALIZE_DIRECTION:{direction.value}->{expected_direction.value}")
            direction = expected_direction
        if role != AtomicMoatRole.MECHANISM:
            mechanisms = []
    normalized = extraction.model_copy(
        update={
            "is_investment_relevant": relevant,
            "moat_role": role if relevant else AtomicMoatRole.NONE,
            "evidence_type": evidence_type,
            "direction": direction,
            "economic_scope": scope,
            "fact": fact if relevant else "No investment-relevant evidence",
            "mechanism": mechanisms,
            "claim_subject": qualitative(extraction.claim_subject, "subject") or "company",
            "claim_predicate": qualitative(extraction.claim_predicate, "predicate")
            or "unspecified",
            # Source-derived period/horizon is added by
            # atomic_extraction_to_judgment below.
            "claim_horizon": qualitative(extraction.claim_horizon, "horizon"),
            "claim_metric": qualitative(extraction.claim_metric, "metric"),
        }
    )
    return normalized, actions
_ATOMIC_METRIC_RE = re.compile(
    r"(?<![0-9A-Za-z])(?P<value>[-+]?\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>%p|pp|%|억원|백만원|천원|원|달러|USD|KRW|개월|년|배)?",
    re.IGNORECASE,
)
_ATOMIC_PERIOD_RE = re.compile(
    r"(?:20\d{2}\s*년(?:\s*\d{1,2}\s*월)?|"
    r"20\d{2}(?:[./-]\d{1,2}(?:[./-]\d{1,2})?)?|"
    r"(?:향후|최근)\s*\d+\s*(?:년|개월)|"
    r"\d+\s*[~-]\s*\d+\s*(?:년|개월))",
    re.IGNORECASE,
)
_FORWARD_DRIVER_RULES: tuple[tuple[ForwardDriverType, re.Pattern[str], tuple[DcfLink, ...]], ...] = (
    (ForwardDriverType.UTILIZATION, re.compile(r"가동률|capacity\s+utili[sz]ation", re.I), (DcfLink.REVENUE, DcfLink.CAPEX)),
    (ForwardDriverType.CAPACITY, re.compile(r"생산능력|CAPA|production\s+capacity", re.I), (DcfLink.REVENUE, DcfLink.CAPEX)),
    (ForwardDriverType.ASP, re.compile(r"ASP|평균\s*판매\s*가격|판매단가", re.I), (DcfLink.REVENUE, DcfLink.EBIT_MARGIN)),
    (ForwardDriverType.EXPORT_MIX, re.compile(r"수출\s*(?:비중|믹스)|export\s+mix", re.I), (DcfLink.REVENUE, DcfLink.EBIT_MARGIN)),
    (ForwardDriverType.PRODUCT_MIX, re.compile(r"제품\s*믹스|product\s+mix", re.I), (DcfLink.REVENUE, DcfLink.EBIT_MARGIN)),
    (ForwardDriverType.RAW_MATERIAL_COST, re.compile(r"원재료|raw\s+material|input\s+cost", re.I), (DcfLink.EBIT_MARGIN,)),
    (ForwardDriverType.WORKING_CAPITAL, re.compile(r"운전\s*자본|working\s+capital", re.I), (DcfLink.NWC,)),
    (ForwardDriverType.CAPEX, re.compile(r"CAPEX|설비\s*투자|자본적\s*지출", re.I), (DcfLink.CAPEX, DcfLink.DEPRECIATION)),
    (ForwardDriverType.MARGIN, re.compile(r"마진|이익률|margin", re.I), (DcfLink.EBIT_MARGIN,)),
    (ForwardDriverType.VOLUME, re.compile(r"판매량|출하량|생산량|volume|shipments?", re.I), (DcfLink.REVENUE,)),
    (ForwardDriverType.MARKET_GROWTH, _DEMAND_RE, (DcfLink.REVENUE,)),
)


def calibrate_card_reliability(card: EvidenceCard) -> EvidenceCard:
    source_text = " ".join(filter(None, [card.raw_quote, card.fact]))
    if card.statement_type == StatementType.DISCLOSED_FACT and re.search(
        r"전망|계획|예상|기대|목표|추정|will|expect|plan|target|forecast",
        source_text,
        re.IGNORECASE,
    ):
        card = card.model_copy(update={"statement_type": StatementType.MANAGEMENT_CLAIM})
    caps = {
        StatementType.DISCLOSED_FACT: 0.95,
        StatementType.DERIVED_METRIC: 0.90,
        StatementType.MANAGEMENT_CLAIM: 0.60,
        StatementType.ANALYST_INTERPRETATION: 0.65,
        StatementType.INDUSTRY_INTERPRETATION: 0.65,
        StatementType.FORECAST: 0.50,
    }
    cap = caps[card.statement_type]
    if card.source_type == SourceType.IR and card.statement_type in {
        StatementType.MANAGEMENT_CLAIM,
        StatementType.FORECAST,
    }:
        cap -= 0.05
    if not card.raw_quote:
        cap = min(cap, 0.20)
    if card.economic_scope not in {EconomicScope.COMPANY, EconomicScope.SEGMENT}:
        cap = min(cap, 0.50)
    if (
        card.direction == EvidenceDirection.MOAT_POSITIVE
        and card.evidence_type in {
            EvidenceType.SWITCHING_COST,
            EvidenceType.NETWORK_EFFECT,
            EvidenceType.COST_ADVANTAGE,
            EvidenceType.INTANGIBLE_ASSET,
            EvidenceType.SCALE_ADVANTAGE,
            EvidenceType.REGULATORY_BARRIER,
        }
        and not card.mechanism
    ):
        cap = min(cap, 0.30)
    # Reliability is a deterministic policy value, not model
    # self-confidence. Using the model-proposed value made identical evidence
    # affect the public score differently between executions.
    return card.model_copy(update={"reliability": round(max(0.0, cap), 2)})


def atomic_extraction_to_judgment(
    extraction: AtomicEvidenceExtraction,
    chunk: SemanticChunk,
) -> AtomicEvidenceJudgment:
    """Deterministically enrich the minimal LLM classification from source text."""

    text = chunk.markdown
    source_type = chunk.source_refs[0].source_type if chunk.source_refs else SourceType.OTHER
    if _FORWARD_LANGUAGE_RE.search(text) and source_type in {
        SourceType.ANALYST,
        SourceType.INDUSTRY,
    }:
        statement_type = StatementType.FORECAST
    elif source_type == SourceType.ANALYST:
        statement_type = StatementType.ANALYST_INTERPRETATION
    elif source_type == SourceType.INDUSTRY:
        statement_type = StatementType.INDUSTRY_INTERPRETATION
    elif _FORWARD_LANGUAGE_RE.search(text):
        statement_type = (
            StatementType.MANAGEMENT_CLAIM
            if source_type in {SourceType.DART, SourceType.SEC_EDGAR, SourceType.IR}
            else StatementType.FORECAST
        )
    else:
        statement_type = StatementType.DISCLOSED_FACT

    metrics: list[EvidenceMetric] = []
    for match in _ATOMIC_METRIC_RE.finditer(text):
        value = match.group("value")
        unit = match.group("unit")
        try:
            calendar_year = int(value.replace(",", ""))
        except ValueError:
            calendar_year = -1
        if unit in {None, "년"} and 1900 <= calendar_year <= 2100:
            continue
        metrics.append(
            EvidenceMetric(
                name=f"source_numeric_{len(metrics) + 1}",
                value=value,
                unit=unit,
            )
        )
        if len(metrics) >= 8:
            break
    periods = list(dict.fromkeys(match.group(0).strip() for match in _ATOMIC_PERIOD_RE.finditer(text)))
    period = "; ".join(periods[:4]) or None
    forward_driver_type = None
    dcf_links: list[DcfLink] = []
    for driver, pattern, links in _FORWARD_DRIVER_RULES:
        if pattern.search(text):
            forward_driver_type = driver
            dcf_links = list(links)
            break
    claim = CanonicalClaimSignature(
        moat_source=extraction.evidence_type,
        subject=extraction.claim_subject or "company",
        predicate=extraction.claim_predicate or "unspecified",
        direction=extraction.direction,
        horizon=(extraction.claim_horizon or period or "UNSPECIFIED"),
        metric=extraction.claim_metric,
    )
    return AtomicEvidenceJudgment(
        is_investment_relevant=extraction.is_investment_relevant,
        moat_role=atomic_moat_role(extraction),
        evidence_type=extraction.evidence_type,
        statement_type=statement_type,
        fact=extraction.fact,
        mechanism=extraction.mechanism,
        direction=extraction.direction,
        # Strength is not a model opinion and is not used in public scoring.
        strength=0.5,
        economic_scope=extraction.economic_scope,
        segment=extraction.segment,
        metrics=metrics,
        unit=next((metric.unit for metric in metrics if metric.unit), None),
        period=period,
        forward_driver_type=forward_driver_type,
        dcf_links=dcf_links,
        forecast_horizon=extraction.claim_horizon,
        claim_signature=claim,
    )


def grounded_evidence_id(card: EvidenceCard, chunk: SemanticChunk) -> str:
    """Build an ID from the cited source span, never from model labels/text."""

    quote = " ".join((card.raw_quote or "").split())
    atomic_key = chunk.metadata.get("atomic_evidence_key")
    if atomic_key:
        # Node/paragraph order is presentation metadata.  The source document,
        # atomic source text and quote are sufficient for a stable audit ID.
        return stable_id("E", chunk.document_id, atomic_key, quote)
    return stable_id("E", chunk.document_id, *sorted(set(card.node_ids)), quote)


def atomic_judgment_to_card(
    judgment: AtomicEvidenceJudgment,
    chunk: SemanticChunk,
    *,
    issuer_id: str | None,
) -> EvidenceCard:
    if not judgment.is_investment_relevant:
        raise ValueError("irrelevant atomic judgment cannot become an evidence card")
    source_type = chunk.source_refs[0].source_type if chunk.source_refs else SourceType.OTHER
    provisional = EvidenceCard(
        evidence_id="PENDING",
        source_chunk_id=chunk.chunk_id,
        node_ids=sorted(set(chunk.node_ids)),
        moat_role=judgment.moat_role,
        evidence_type=judgment.evidence_type,
        statement_type=judgment.statement_type,
        fact=judgment.fact,
        mechanism=judgment.mechanism,
        direction=judgment.direction,
        strength=judgment.strength,
        source_type=source_type,
        economic_scope=judgment.economic_scope,
        segment=judgment.segment,
        metrics=judgment.metrics,
        unit=judgment.unit,
        period=judgment.period,
        raw_quote=chunk.markdown,
        forward_driver_type=judgment.forward_driver_type,
        dcf_links=judgment.dcf_links,
        forecast_horizon=judgment.forecast_horizon,
        atomic_evidence_key=str(chunk.metadata["atomic_evidence_key"]),
        claim_signature=judgment.claim_signature,
    )
    normalized = normalize_card_semantics(provisional)
    evidence_id = grounded_evidence_id(normalized, chunk)
    calibrated = calibrate_card_reliability(
        normalized.model_copy(update={"evidence_id": evidence_id})
    )
    return assign_canonical_claim_identity(calibrated, issuer_id=issuer_id)


def _canonical_slot(value: str | None, fallback: str) -> str:
    text = unicodedata.normalize("NFKC", value or fallback).casefold()
    text = re.sub(r"[^0-9a-z가-힣一-龥%><=]+", "_", text, flags=re.IGNORECASE)
    return text.strip("_") or fallback


def _metric_value_bucket(card: EvidenceCard) -> str | None:
    if not card.metrics:
        return None
    value = str(card.metrics[0].value).replace(",", "").strip()
    match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
    if not match:
        return _canonical_slot(value, "unspecified")
    numeric = abs(float(match.group()))
    if "%" in value:
        if numeric < 5:
            return "pct_lt_5"
        if numeric < 20:
            return "pct_5_20"
        if numeric < 50:
            return "pct_20_50"
        return "pct_ge_50"
    if numeric < 1:
        return "lt_1"
    if numeric < 3:
        return "1_3"
    if numeric < 10:
        return "3_10"
    return "ge_10"


def assign_canonical_claim_identity(card: EvidenceCard, *, issuer_id: str | None) -> EvidenceCard:
    proposed = card.claim_signature
    signature = CanonicalClaimSignature(
        moat_source=card.evidence_type,
        subject=_canonical_slot(
            proposed.subject if proposed else card.segment or card.company_scope,
            "company",
        ),
        predicate=_canonical_slot(
            proposed.predicate if proposed else card.fact,
            "unspecified",
        ),
        direction=card.direction,
        horizon=_canonical_slot(
            proposed.horizon if proposed else card.forecast_horizon or card.period,
            "unspecified",
        ).upper(),
        metric=(
            _canonical_slot(proposed.metric, "unspecified")
            if proposed and proposed.metric
            else (
                None
                if proposed is not None
                else (_canonical_slot(card.metrics[0].name, "unspecified") if card.metrics else None)
            )
        ),
        value_bucket=(
            _canonical_slot(proposed.value_bucket, "unspecified")
            if proposed and proposed.value_bucket
            else (
                _metric_value_bucket(card)
                if proposed is None or proposed.metric is not None
                else None
            )
        ),
    )
    claim_id = stable_id(
        "CL",
        issuer_id or "UNKNOWN_ISSUER",
        signature.moat_source.value,
        signature.subject,
        signature.predicate,
        signature.direction.value,
        signature.horizon,
        signature.metric or "",
        signature.value_bucket or "",
    )
    return card.model_copy(update={"claim_signature": signature, "claim_id": claim_id})


def build_canonical_claim_set(
    cards: list[EvidenceCard],
    *,
    issuer_id: str | None,
) -> tuple[list[EvidenceCard], list[ClaimCluster]]:
    """Reduce evidence to a commutative, associative and idempotent set."""

    grouped: dict[str, list[EvidenceCard]] = {}
    for source in cards:
        card = assign_canonical_claim_identity(source, issuer_id=issuer_id)
        assert card.claim_id is not None
        grouped.setdefault(card.claim_id, []).append(card)
    canonical_cards: list[EvidenceCard] = []
    clusters: list[ClaimCluster] = []
    for claim_id in sorted(grouped):
        group_by_evidence_id = {card.evidence_id: card for card in grouped[claim_id]}
        group = list(group_by_evidence_id.values())
        canonical = min(group, key=lambda card: (-card.reliability, card.evidence_id))
        supporters = sorted(card.evidence_id for card in group if card.evidence_id != canonical.evidence_id)
        canonical_cards.append(canonical)
        clusters.append(
            ClaimCluster(
                claim_id=claim_id,
                canonical_evidence_id=canonical.evidence_id,
                supporting_evidence_ids=supporters,
            )
        )
    return canonical_cards, clusters


def normalize_card_semantics(card: EvidenceCard) -> EvidenceCard:
    """Correct category-demand claims and promote grounded DCF drivers."""

    text = " ".join([card.fact, *card.mechanism])
    update: dict[str, object] = {}
    links = list(card.dcf_links)
    if card.evidence_type == EvidenceType.MARKET_SHARE and _DEMAND_RE.search(text) and not _SHARE_RE.search(text):
        update.update(
            evidence_type=EvidenceType.MARKET_DEMAND,
            economic_scope=EconomicScope.INDUSTRY,
            direction=EvidenceDirection.NEUTRAL,
            forward_driver_type=ForwardDriverType.MARKET_GROWTH,
        )
        links.append(DcfLink.REVENUE)
    if (
        card.evidence_type == EvidenceType.CUSTOMER_RETENTION
        and _RECURRING_CATEGORY_RE.search(text)
        and not _RETENTION_RE.search(text)
    ):
        update.update(
            evidence_type=EvidenceType.CATEGORY_RECURRING_DEMAND,
            economic_scope=EconomicScope.PRODUCT_CATEGORY,
            direction=EvidenceDirection.NEUTRAL,
            forward_driver_type=ForwardDriverType.VOLUME,
        )
        links.append(DcfLink.REVENUE)
    if re.search(r"가동률|capacity\s+utili[sz]ation", text, re.IGNORECASE):
        update.setdefault("forward_driver_type", ForwardDriverType.UTILIZATION)
        links.extend([DcfLink.REVENUE, DcfLink.CAPEX])
    elif re.search(r"생산능력|CAPA|production\s+capacity", text, re.IGNORECASE):
        update.setdefault("forward_driver_type", ForwardDriverType.CAPACITY)
        links.extend([DcfLink.REVENUE, DcfLink.CAPEX])
    if re.search(r"수출\s*(?:비중|믹스)|export\s+mix", text, re.IGNORECASE):
        update.setdefault("forward_driver_type", ForwardDriverType.EXPORT_MIX)
        links.extend([DcfLink.REVENUE, DcfLink.EBIT_MARGIN])
    if re.search(r"ASP|평균\s*판매\s*가격|판매단가", text, re.IGNORECASE):
        update.setdefault("forward_driver_type", ForwardDriverType.ASP)
        links.extend([DcfLink.REVENUE, DcfLink.EBIT_MARGIN])
    if re.search(r"원재료|raw\s+material|input\s+cost", text, re.IGNORECASE):
        update.setdefault("forward_driver_type", ForwardDriverType.RAW_MATERIAL_COST)
        links.append(DcfLink.EBIT_MARGIN)
    if update.get("forward_driver_type") is not None or card.forward_driver_type is not None:
        update["dcf_links"] = list(dict.fromkeys(links))
    return card.model_copy(update=update) if update else card


def build_forward_driver_cards(cards: list[EvidenceCard]) -> list[ForwardDriverCard]:
    result: list[ForwardDriverCard] = []
    for card in cards:
        if card.forward_driver_type is None or not card.dcf_links:
            continue
        result.append(
            ForwardDriverCard(
                driver_id=stable_id("FD", card.evidence_id, card.forward_driver_type.value),
                source_evidence_id=card.evidence_id,
                source_chunk_id=card.source_chunk_id,
                node_ids=card.node_ids,
                driver_type=card.forward_driver_type,
                evidence=card.fact,
                implication=card.mechanism,
                dcf_links=card.dcf_links,
                statement_type=card.statement_type,
                economic_scope=card.economic_scope,
                segment=card.segment,
                period=card.period,
                forecast_horizon=card.forecast_horizon,
                reliability=card.reliability,
            )
        )
    return result


def build_evidence_preserving_summaries(
    cards: list[EvidenceCard],
    *,
    section_path_by_chunk: dict[str, list[str]],
    consolidate: bool,
) -> list[SectionSummary]:
    """Create display summaries without a second generative interpretation.

    Every line is an existing canonical fact attached to one Evidence ID.
    Positive, counter, and KPI/context lanes remain separate, so compression
    cannot erase adverse evidence or count a quote and its summary twice.
    """

    grouped: dict[tuple[str, ...], list[EvidenceCard]] = defaultdict(list)
    if consolidate and cards:
        grouped[("Selected MOAT evidence",)] = list(cards)
    else:
        for card in cards:
            grouped[tuple(section_path_by_chunk.get(card.source_chunk_id, []))].append(card)
    summaries: list[SectionSummary] = []
    for path, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda item: (item.claim_id or "", item.evidence_id))
        positive = [item for item in ordered if item.direction == EvidenceDirection.MOAT_POSITIVE]
        negative = [item for item in ordered if item.direction == EvidenceDirection.MOAT_NEGATIVE]
        mechanisms = [
            CitedSummaryClaim(text=item.fact, evidence_ids=[item.evidence_id])
            for item in positive
            if item.evidence_type in STRUCTURAL_MOAT_TYPES
        ]
        kpis = [
            CitedSummaryClaim(text=item.fact, evidence_ids=[item.evidence_id])
            for item in ordered
            if item.direction != EvidenceDirection.MOAT_NEGATIVE
            and not (
                item.direction == EvidenceDirection.MOAT_POSITIVE
                and item.evidence_type in STRUCTURAL_MOAT_TYPES
            )
        ]
        uncertainties = [
            CitedSummaryClaim(text=item.fact, evidence_ids=[item.evidence_id])
            for item in negative
        ]
        summaries.append(
            SectionSummary(
                section_path=list(path),
                positive_evidence_ids=[item.evidence_id for item in positive],
                negative_evidence_ids=[item.evidence_id for item in negative],
                key_mechanisms=mechanisms,
                key_kpis=kpis,
                uncertainties=uncertainties,
            )
        )
    return summaries


def build_evidence_relations(
    cards: list[EvidenceCard],
    *,
    duplicate_threshold: float = 0.92,
    contradiction_threshold: float = 0.72,
    update_threshold: float = 0.82,
    support_threshold: float = 0.55,
    weakens_threshold: float = 0.45,
) -> list[EvidenceRelation]:
    cards = sorted(cards, key=lambda card: card.evidence_id)
    relations: list[EvidenceRelation] = []
    normalized = [re.sub(r"\s+", " ", card.fact).strip().casefold() for card in cards]
    for index, card in enumerate(cards):
        for prior_index in range(index):
            prior = cards[prior_index]
            if card.evidence_type != prior.evidence_type:
                continue
            similarity = SequenceMatcher(None, normalized[index], normalized[prior_index]).ratio()
            numbers = _NUMBER_RE.findall(normalized[index])
            prior_numbers = _NUMBER_RE.findall(normalized[prior_index])
            relation = None
            if card.direction == prior.direction and similarity >= duplicate_threshold and numbers == prior_numbers:
                relation = EvidenceRelationType.DUPLICATES
            elif card.direction == prior.direction and similarity >= update_threshold and numbers != prior_numbers:
                relation = EvidenceRelationType.UPDATES
            elif (
                card.direction != prior.direction
                and EvidenceDirection.NEUTRAL not in {card.direction, prior.direction}
                and similarity >= contradiction_threshold
            ):
                relation = EvidenceRelationType.CONTRADICTS
            elif (
                card.direction != prior.direction
                and EvidenceDirection.NEUTRAL not in {card.direction, prior.direction}
                and similarity >= weakens_threshold
            ):
                relation = EvidenceRelationType.WEAKENS
            elif card.direction == prior.direction and similarity >= support_threshold:
                relation = EvidenceRelationType.SUPPORTS
            if relation is not None:
                relations.append(
                    EvidenceRelation(
                        from_evidence_id=card.evidence_id,
                        to_evidence_id=prior.evidence_id,
                        relation=relation,
                    )
                )
                break
    return relations


def cluster_duplicate_evidence(
    cards: list[EvidenceCard],
    relations: list[EvidenceRelation],
) -> list[EvidenceCluster]:
    """Return one deterministic representative plus all duplicate supporting IDs."""
    card_by_id = {card.evidence_id: card for card in cards}
    if len(card_by_id) != len(cards):
        raise ValueError("evidence IDs must be unique before clustering")
    parent = {evidence_id: evidence_id for evidence_id in card_by_id}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for relation in relations:
        if relation.relation != EvidenceRelationType.DUPLICATES:
            continue
        if relation.from_evidence_id not in card_by_id or relation.to_evidence_id not in card_by_id:
            raise ValueError("duplicate relation references missing evidence")
        union(relation.from_evidence_id, relation.to_evidence_id)

    groups: dict[str, list[EvidenceCard]] = {}
    for card in cards:
        groups.setdefault(find(card.evidence_id), []).append(card)
    clusters: list[EvidenceCluster] = []
    for group in groups.values():
        canonical = min(
            group,
            key=lambda card: (
                -card.reliability,
                card.evidence_id,
            ),
        )
        supporters = sorted(card.evidence_id for card in group if card.evidence_id != canonical.evidence_id)
        clusters.append(
            EvidenceCluster(
                canonical_evidence_id=canonical.evidence_id,
                supporting_evidence_ids=supporters,
            )
        )
    return sorted(clusters, key=lambda cluster: cluster.canonical_evidence_id)
