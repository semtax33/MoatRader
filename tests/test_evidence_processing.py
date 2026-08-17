from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from moatrader.canonical.models import SourceType, StatementType
from moatrader.evidence.models import (
    AtomicEvidenceExtraction,
    AtomicMoatRole,
    DcfLink,
    EconomicScope,
    EvidenceBatchExtractionResult,
    EvidenceCard,
    EvidenceDirection,
    EvidenceExtractionResult,
    EvidenceMetric,
    EvidenceRelationType,
    EvidenceType,
    ForwardDriverType,
)
from moatrader.evidence.ledger import EvidenceLedgerStore
from moatrader.evidence.processing import (
    atomic_moat_role,
    build_atomic_classification_consensus,
    build_forward_driver_cards,
    build_evidence_relations,
    calibrate_card_reliability,
    cluster_duplicate_evidence,
    grounded_evidence_id,
    normalize_card_semantics,
    normalize_atomic_extraction,
)
from moatrader.evidence.validation import validate_evidence_batch_result, validate_evidence_result
from moatrader.semantic.chunker import SemanticChunk


def _card(evidence_id: str, fact: str, direction: EvidenceDirection) -> EvidenceCard:
    return EvidenceCard(
        evidence_id=evidence_id,
        source_chunk_id="C1",
        node_ids=["N1"],
        evidence_type=EvidenceType.SWITCHING_COST,
        statement_type=StatementType.MANAGEMENT_CLAIM,
        fact=fact,
        raw_quote=fact,
        direction=direction,
        strength=0.7,
        source_type=SourceType.IR,
        reliability=0.95,
    )


def test_reliability_is_capped_by_statement_and_source() -> None:
    card = _card("E1", "Customer qualification takes 18 months.", EvidenceDirection.MOAT_POSITIVE)
    card.mechanism = ["qualification creates switching friction"]
    assert calibrate_card_reliability(card).reliability == 0.55


def test_grounded_evidence_id_ignores_model_wording_and_classification() -> None:
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown="Customer qualification takes 18 months.",
        token_count=5,
    )
    first = _card("pending", "First paraphrase.", EvidenceDirection.MOAT_POSITIVE)
    first.raw_quote = "Customer qualification takes 18 months."
    second = first.model_copy(
        update={
            "fact": "Different paraphrase.",
            "direction": EvidenceDirection.NEUTRAL,
            "evidence_type": EvidenceType.OPERATING_DRIVER,
        }
    )

    assert grounded_evidence_id(first, chunk) == grounded_evidence_id(second, chunk)


def _atomic_vote(
    role: AtomicMoatRole,
    evidence_type: EvidenceType,
    direction: EvidenceDirection,
    *,
    scope: EconomicScope = EconomicScope.COMPANY,
) -> AtomicEvidenceExtraction:
    return AtomicEvidenceExtraction(
        is_investment_relevant=True,
        moat_role=role,
        evidence_type=evidence_type,
        direction=direction,
        economic_scope=scope,
        fact="model wording that must not become consensus identity",
        mechanism=["model phrase"],
    )


def test_atomic_consensus_uses_strict_majority_and_source_owned_text() -> None:
    mechanism = _atomic_vote(
        AtomicMoatRole.MECHANISM,
        EvidenceType.COST_ADVANTAGE,
        EvidenceDirection.MOAT_POSITIVE,
    )
    outcome = _atomic_vote(
        AtomicMoatRole.OUTCOME,
        EvidenceType.MARKET_SHARE,
        EvidenceDirection.MOAT_POSITIVE,
    )

    selected, audit = build_atomic_classification_consensus(
        [mechanism, outcome, mechanism],
        source_text="생산 수율이 10배 개선되어 제조비용이 낮아졌다.",
    )

    assert audit["status"] == "CONSENSUS"
    assert audit["winning_vote_count"] == 2
    assert selected.moat_role == AtomicMoatRole.MECHANISM
    assert selected.evidence_type == EvidenceType.COST_ADVANTAGE
    assert selected.fact == "생산 수율이 10배 개선되어 제조비용이 낮아졌다."
    assert selected.claim_predicate == "cost_advantage"
    assert selected.mechanism == ["observable persistent cost barrier"]


def test_atomic_consensus_fails_closed_without_exact_label_majority() -> None:
    votes = [
        _atomic_vote(
            AtomicMoatRole.MECHANISM,
            EvidenceType.COST_ADVANTAGE,
            EvidenceDirection.MOAT_POSITIVE,
        ),
        _atomic_vote(
            AtomicMoatRole.OUTCOME,
            EvidenceType.MARKET_SHARE,
            EvidenceDirection.MOAT_POSITIVE,
        ),
        _atomic_vote(
            AtomicMoatRole.COUNTER,
            EvidenceType.COMPETITIVE_THREAT,
            EvidenceDirection.MOAT_NEGATIVE,
        ),
    ]

    selected, audit = build_atomic_classification_consensus(votes, source_text="ambiguous")

    assert audit["status"] == "NO_CONSENSUS_FAIL_CLOSED"
    assert selected.is_investment_relevant is False
    assert selected.moat_role == AtomicMoatRole.NONE
    assert selected.evidence_type == EvidenceType.OTHER


def test_atomic_consensus_votes_on_route_before_scope() -> None:
    company = _atomic_vote(
        AtomicMoatRole.OUTCOME,
        EvidenceType.CUSTOMER_RETENTION,
        EvidenceDirection.MOAT_POSITIVE,
        scope=EconomicScope.COMPANY,
    )
    segment = _atomic_vote(
        AtomicMoatRole.OUTCOME,
        EvidenceType.CUSTOMER_RETENTION,
        EvidenceDirection.MOAT_POSITIVE,
        scope=EconomicScope.SEGMENT,
    )
    irrelevant = AtomicEvidenceExtraction()

    selected, audit = build_atomic_classification_consensus(
        [company, company, segment, segment, irrelevant],
        source_text="장기 렌탈 재계약률은 90% 이상이다.",
    )

    assert audit["status"] == "CONSENSUS"
    assert audit["winning_vote_count"] == 4
    assert audit["winning_route"] == [
        "OUTCOME",
        True,
        "CUSTOMER_RETENTION",
        "MOAT_POSITIVE",
    ]
    assert selected.moat_role == AtomicMoatRole.OUTCOME
    assert selected.evidence_type == EvidenceType.CUSTOMER_RETENTION
    assert selected.economic_scope == EconomicScope.COMPANY


def test_atomic_role_requires_compatible_type_direction_and_company_scope() -> None:
    invalid_type = _atomic_vote(
        AtomicMoatRole.MECHANISM,
        EvidenceType.MARKET_SHARE,
        EvidenceDirection.MOAT_POSITIVE,
    )
    category_scope = _atomic_vote(
        AtomicMoatRole.MECHANISM,
        EvidenceType.COST_ADVANTAGE,
        EvidenceDirection.MOAT_POSITIVE,
        scope=EconomicScope.PRODUCT_CATEGORY,
    )

    assert atomic_moat_role(invalid_type) == AtomicMoatRole.NONE
    assert atomic_moat_role(category_scope) == AtomicMoatRole.NONE
    normalized, actions = normalize_atomic_extraction(invalid_type)
    assert normalized.is_investment_relevant is False
    assert normalized.evidence_type == EvidenceType.OTHER
    assert "FAIL_CLOSED_INVALID_OR_NONE_MOAT_ROLE" in actions


def test_atomic_none_role_preserves_explicit_forward_dcf_driver() -> None:
    driver = AtomicEvidenceExtraction(
        is_investment_relevant=True,
        moat_role=AtomicMoatRole.NONE,
        evidence_type=EvidenceType.CAPACITY_UTILIZATION,
        direction=EvidenceDirection.NEUTRAL,
        economic_scope=EconomicScope.INDUSTRY,
        fact="Industry utilization is expected to recover next year.",
    )

    normalized, actions = normalize_atomic_extraction(driver)

    assert normalized.is_investment_relevant is True
    assert normalized.moat_role == AtomicMoatRole.NONE
    assert normalized.evidence_type == EvidenceType.CAPACITY_UTILIZATION
    assert normalized.direction == EvidenceDirection.NEUTRAL
    assert normalized.economic_scope == EconomicScope.INDUSTRY
    assert "PRESERVE_EXPLICIT_NON_MOAT_DCF_DRIVER" in actions


def test_observable_anchor_rejects_cumulative_orders_as_customer_retention() -> None:
    vote = _atomic_vote(
        AtomicMoatRole.OUTCOME,
        EvidenceType.CUSTOMER_RETENTION,
        EvidenceDirection.MOAT_POSITIVE,
    )

    normalized, actions = normalize_atomic_extraction(
        vote,
        source_text=(
            "AVACO shows cumulative sputter orders rising 74.9 times and describes "
            "a long trust relationship with a global display customer."
        ),
    )

    assert normalized.is_investment_relevant is False
    assert normalized.moat_role == AtomicMoatRole.NONE
    assert normalized.evidence_type == EvidenceType.OTHER
    assert any("CUSTOMER_RETENTION_REQUIRES_DIRECT_RETENTION_BEHAVIOR" in action for action in actions)


def test_observable_anchor_rejects_ordinary_profit_decline_as_counter() -> None:
    vote = _atomic_vote(
        AtomicMoatRole.COUNTER,
        EvidenceType.MARGIN_STABILITY,
        EvidenceDirection.MOAT_NEGATIVE,
    )

    normalized, actions = normalize_atomic_extraction(
        vote,
        source_text="2021 Q3 revenue fell 7% and operating profit fell 33% year over year.",
    )

    assert normalized.is_investment_relevant is False
    assert normalized.evidence_type == EvidenceType.OTHER
    assert any("MARGIN_STABILITY_REQUIRES_MULTI_PERIOD_PROFITABILITY" in action for action in actions)


def test_observable_anchors_preserve_true_share_margin_cost_and_retention_routes() -> None:
    cases = [
        (
            _atomic_vote(
                AtomicMoatRole.OUTCOME,
                EvidenceType.MARKET_SHARE,
                EvidenceDirection.MOAT_POSITIVE,
            ),
            "Global Tax Free is the domestic number-one operator with 70-75% market share.",
        ),
        (
            _atomic_vote(
                AtomicMoatRole.OUTCOME,
                EvidenceType.CUSTOMER_RETENTION,
                EvidenceDirection.MOAT_POSITIVE,
            ),
            "The same customers renewed 92% of expiring contracts during 2025.",
        ),
        (
            _atomic_vote(
                AtomicMoatRole.COUNTER,
                EvidenceType.MARGIN_STABILITY,
                EvidenceDirection.MOAT_NEGATIVE,
            ),
            "Operating margin was 24.7%, 12.1%, 30.5%, 11.8%, and 28.9% over five-quarter volatility.",
        ),
        (
            _atomic_vote(
                AtomicMoatRole.COUNTER,
                EvidenceType.MARGIN_STABILITY,
                EvidenceDirection.MOAT_NEGATIVE,
            ),
            (
                "Operating margin alternates across five consecutive quarters: 24.7%, 12.1%, "
                "30.5%, 11.8%, and 28.9%; direction reverses three times."
            ),
        ),
        (
            _atomic_vote(
                AtomicMoatRole.MECHANISM,
                EvidenceType.COST_ADVANTAGE,
                EvidenceDirection.MOAT_POSITIVE,
            ),
            "EcML reduces production time and manufacturing cost versus GLA's 30-plus synthesis steps.",
        ),
        (
            _atomic_vote(
                AtomicMoatRole.OUTCOME,
                EvidenceType.MARGIN_STABILITY,
                EvidenceDirection.MOAT_POSITIVE,
            ),
            "BIO-FD&C remained profitable for eleven consecutive years from 2011 through 2023.",
        ),
    ]

    for vote, source_text in cases:
        normalized, actions = normalize_atomic_extraction(vote, source_text=source_text)
        assert normalized.is_investment_relevant is True
        assert normalized.moat_role == vote.moat_role
        assert normalized.evidence_type == vote.evidence_type
        assert not any(action.startswith("FAIL_CLOSED_OBSERVABLE_ANCHOR") for action in actions)


def test_evidence_ledger_carries_omitted_structural_evidence_without_future_leak(tmp_path: Path) -> None:
    ledger = EvidenceLedgerStore(tmp_path, experiment_id="exp")
    first_date = datetime.fromisoformat("2025-08-31T23:59:59+09:00")
    later_date = datetime.fromisoformat("2025-11-30T23:59:59+09:00")
    earlier_date = datetime.fromisoformat("2025-05-31T23:59:59+09:00")
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown="Customer qualification takes 18 months.",
        token_count=5,
    )
    card = _card("E1", "Customer qualification takes 18 months.", EvidenceDirection.MOAT_POSITIVE)
    card.mechanism = ["qualification creates switching friction"]

    ledger.merge(
        "000001",
        as_of=first_date,
        current_cards=[card],
        chunks=[chunk],
        document_available_at={"D1": first_date},
    )
    carried = ledger.merge(
        "000001",
        as_of=later_date,
        current_cards=[],
        chunks=[],
        document_available_at={},
    )
    historical = ledger.merge(
        "000001",
        as_of=earlier_date,
        current_cards=[],
        chunks=[],
        document_available_at={},
    )

    assert [item.evidence_id for item in carried.cards] == ["E1"]
    assert carried.carried_evidence_count == 1
    assert historical.cards == []


def test_relations_preserve_duplicates_updates_and_contradictions() -> None:
    relations = build_evidence_relations(
        [
            _card("E1", "Customer qualification takes 18 months.", EvidenceDirection.MOAT_POSITIVE),
            _card("E2", "Customer qualification takes 18 months.", EvidenceDirection.MOAT_POSITIVE),
            _card("E3", "Customer qualification takes 24 months.", EvidenceDirection.MOAT_POSITIVE),
            _card("E4", "Customer qualification takes 18 months.", EvidenceDirection.MOAT_NEGATIVE),
        ],
        update_threshold=0.70,
    )

    assert [relation.relation for relation in relations] == [
        EvidenceRelationType.DUPLICATES,
        EvidenceRelationType.UPDATES,
        EvidenceRelationType.CONTRADICTS,
    ]


def test_relations_preserve_supports_and_weakens() -> None:
    relations = build_evidence_relations(
        [
            _card("E1", "Qualification creates customer friction.", EvidenceDirection.MOAT_POSITIVE),
            _card("E2", "Integration creates customer friction.", EvidenceDirection.MOAT_POSITIVE),
            _card("E3", "Dual sourcing reduces customer friction.", EvidenceDirection.MOAT_NEGATIVE),
        ],
        duplicate_threshold=1.1,
        update_threshold=1.1,
        contradiction_threshold=1.1,
        support_threshold=0.0,
        weakens_threshold=0.0,
    )

    assert [relation.relation for relation in relations] == [
        EvidenceRelationType.SUPPORTS,
        EvidenceRelationType.WEAKENS,
    ]


def test_duplicate_clusters_choose_one_canonical_and_keep_supporters() -> None:
    cards = [
        _card("E1", "Customer qualification takes 18 months.", EvidenceDirection.MOAT_POSITIVE),
        _card("E2", "Customer qualification takes 18 months.", EvidenceDirection.MOAT_POSITIVE),
    ]
    relations = build_evidence_relations(cards)

    clusters = cluster_duplicate_evidence(cards, relations)

    assert len(clusters) == 1
    assert clusters[0].canonical_evidence_id == "E1"
    assert clusters[0].supporting_evidence_ids == ["E2"]


def test_batch_validation_drops_cards_with_unknown_chunk_ids() -> None:
    result = EvidenceBatchExtractionResult(
        cards=[_card("E1", "Customer qualification takes 18 months.", EvidenceDirection.MOAT_POSITIVE)]
    )

    errors = validate_evidence_batch_result(result, [], {})

    assert errors == []
    assert result.cards == []


def test_validation_drops_only_an_ungrounded_numeric_metric() -> None:
    card = _card("E1", "Customer qualification is lengthy.", EvidenceDirection.MOAT_POSITIVE)
    card.metrics = [EvidenceMetric(name="qualification_months", value=18)]
    result = EvidenceExtractionResult(chunk_id="C1", cards=[card])
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown="Customer qualification is lengthy.",
        token_count=5,
    )
    bundle = SimpleNamespace(ast=SimpleNamespace(node_index=lambda: {"N1": object()}))

    errors = validate_evidence_result(result, chunk, bundle)

    assert errors == []
    assert result.cards[0].metrics == []


def test_validation_accepts_grounded_decimal_followed_by_source_punctuation() -> None:
    card = _card("E1", "The contract date is 2013.12", EvidenceDirection.NEUTRAL)
    card.raw_quote = "Contract date: 2013.12."
    result = EvidenceExtractionResult(chunk_id="C1", cards=[card])
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown="Contract date: 2013.12.",
        token_count=4,
    )
    bundle = SimpleNamespace(ast=SimpleNamespace(node_index=lambda: {"N1": object()}))

    errors = validate_evidence_result(result, chunk, bundle)

    assert errors == []
    assert [item.evidence_id for item in result.cards] == ["E1"]


def test_validation_accepts_grounded_multi_dot_dates_with_sentence_punctuation() -> None:
    card = _card(
        "E1",
        "The license runs from 2020.09.03, through 2040.09.03.",
        EvidenceDirection.NEUTRAL,
    )
    card.raw_quote = "License period: 2020.09.03 through 2040.09.03."
    result = EvidenceExtractionResult(chunk_id="C1", cards=[card])
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown="License period: 2020.09.03 through 2040.09.03.",
        token_count=6,
    )
    bundle = SimpleNamespace(ast=SimpleNamespace(node_index=lambda: {"N1": object()}))

    errors = validate_evidence_result(result, chunk, bundle)

    assert errors == []
    assert [item.evidence_id for item in result.cards] == ["E1"]


def test_validation_recovers_a_year_joined_to_an_ascii_section_label() -> None:
    card = _card(
        "E1",
        "The GOLF business launched LPGA golfwear in August 2016.",
        EvidenceDirection.NEUTRAL,
    )
    card.raw_quote = "GOLF2016년 8월에 LPGA 골프웨어를 런칭하였습니다."
    result = EvidenceExtractionResult(chunk_id="C1", cards=[card])
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown="(4) GOLF2016년 8월에 LPGA 골프웨어를 런칭하였습니다.",
        token_count=6,
    )
    bundle = SimpleNamespace(ast=SimpleNamespace(node_index=lambda: {"N1": object()}))

    errors = validate_evidence_result(result, chunk, bundle)

    assert errors == []
    assert [item.evidence_id for item in result.cards] == ["E1"]


def test_validation_accepts_spaced_percent_and_iso_date_range() -> None:
    source = "Contract value is 3.8 % of sales for 2025-03-17~2025-10-27."
    card = _card(
        "E1",
        "The contract is 3.8% of sales and runs from 2025-03-17 to 2025-10-27.",
        EvidenceDirection.NEUTRAL,
    )
    card.raw_quote = source
    result = EvidenceExtractionResult(chunk_id="C1", cards=[card])
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown=source,
        token_count=8,
    )
    bundle = SimpleNamespace(ast=SimpleNamespace(node_index=lambda: {"N1": object()}))

    errors = validate_evidence_result(result, chunk, bundle)

    assert errors == []
    assert [item.evidence_id for item in result.cards] == ["E1"]


def test_validation_does_not_treat_iso_date_day_as_a_signed_value() -> None:
    source = "The contract ends on 2025-10-27."
    card = _card("E1", "The contract declined by -27 units.", EvidenceDirection.NEUTRAL)
    card.raw_quote = source
    result = EvidenceExtractionResult(chunk_id="C1", cards=[card])
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown=source,
        token_count=6,
    )
    bundle = SimpleNamespace(ast=SimpleNamespace(node_index=lambda: {"N1": object()}))

    errors = validate_evidence_result(result, chunk, bundle)

    assert any("-27" in error for error in errors)
    assert result.cards == []


def test_validation_recovers_year_joined_to_comma_grouped_amount() -> None:
    source = "Milestone amount $34,000,0002020년 5월과 11월에 수익을 인식했습니다."
    card = _card(
        "E1",
        "The milestone revenue was recognized in May and November 2020.",
        EvidenceDirection.NEUTRAL,
    )
    card.raw_quote = source
    result = EvidenceExtractionResult(chunk_id="C1", cards=[card])
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown=source,
        token_count=7,
    )
    bundle = SimpleNamespace(ast=SimpleNamespace(node_index=lambda: {"N1": object()}))

    errors = validate_evidence_result(result, chunk, bundle)

    assert errors == []
    assert [item.evidence_id for item in result.cards] == ["E1"]


def test_validation_recovers_dates_joined_to_adjacent_dates_and_section_number() -> None:
    source = (
        "2025.10.172023.04.062020.07.14 | "
        "The contract was extended to 2027-08-193. Other matters"
    )
    card = _card(
        "E1",
        "The disclosure was amended on 2023.04.06 and the contract ends 2027-08-19.",
        EvidenceDirection.NEUTRAL,
    )
    card.raw_quote = source
    result = EvidenceExtractionResult(chunk_id="C1", cards=[card])
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown=source,
        token_count=9,
    )
    bundle = SimpleNamespace(ast=SimpleNamespace(node_index=lambda: {"N1": object()}))

    errors = validate_evidence_result(result, chunk, bundle)

    assert errors == []
    assert [item.evidence_id for item in result.cards] == ["E1"]


def test_validation_expands_compact_bond_issue_enumeration() -> None:
    source = (
        "Debt ratios must remain below 550% (issue 42-2) and "
        "600% (issue 43-1,2)."
    )
    card = _card(
        "E1",
        "Debt ratios must remain below 550% for issue 42-2 and 600% for "
        "issues 43-1 and 43-2.",
        EvidenceDirection.NEUTRAL,
    )
    card.raw_quote = source
    result = EvidenceExtractionResult(chunk_id="C1", cards=[card])
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown=source,
        token_count=14,
    )
    bundle = SimpleNamespace(ast=SimpleNamespace(node_index=lambda: {"N1": object()}))

    errors = validate_evidence_result(result, chunk, bundle)

    assert errors == []
    assert [item.evidence_id for item in result.cards] == ["E1"]


def test_validation_rejects_a_card_without_verbatim_grounding() -> None:
    card = _card("E1", "Paraphrased claim.", EvidenceDirection.MOAT_POSITIVE)
    card.raw_quote = "Text that is not present"
    result = EvidenceExtractionResult(chunk_id="C1", cards=[card])
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown="Canonical source text.",
        token_count=3,
    )
    bundle = SimpleNamespace(ast=SimpleNamespace(node_index=lambda: {"N1": object()}))

    errors = validate_evidence_result(result, chunk, bundle)

    assert any("raw_quote is required" in error for error in errors)
    assert result.cards == []


def test_validation_can_conservatively_discard_an_ungrounded_card() -> None:
    card = _card("E1", "Paraphrased claim.", EvidenceDirection.MOAT_POSITIVE)
    card.raw_quote = "Text that is not present"
    result = EvidenceExtractionResult(chunk_id="C1", cards=[card])
    chunk = SemanticChunk(
        chunk_id="C1",
        document_id="D1",
        node_ids=["N1"],
        chunk_type="paragraph",
        markdown="Canonical source text.",
        token_count=3,
    )
    bundle = SimpleNamespace(ast=SimpleNamespace(node_index=lambda: {"N1": object()}))

    errors = validate_evidence_result(
        result,
        chunk,
        bundle,
        discard_invalid_cards=True,
    )

    assert errors == []
    assert result.cards == []


def test_evidence_metric_ignores_repaired_json_fragment_keys() -> None:
    metric = EvidenceMetric.model_validate(
        {"name": "revenue", "value": 8780, "unit": "KRW mn", '8780,"unit': "KRW mn"}
    )

    assert metric.name == "revenue"
    assert metric.value == 8780


def test_evidence_metric_unwraps_provider_scalar_objects() -> None:
    metric = EvidenceMetric.model_validate(
        {"name": "fiscal_year", "value": {"type": "number", "value": 2024}}
    )

    assert metric.value == 2024


def test_forward_driver_type_unwraps_provider_containers() -> None:
    payload = _card("E1", "Capacity will increase.", EvidenceDirection.NEUTRAL).model_dump(mode="json")

    for wrapped in (["CAPACITY"], {"ForwardDriverType": "CAPACITY"}):
        payload["forward_driver_type"] = wrapped
        card = EvidenceCard.model_validate(payload)

        assert card.forward_driver_type == ForwardDriverType.CAPACITY


def test_empty_card_confidence_fields_use_neutral_defaults() -> None:
    payload = _card("E1", "Grounded fact.", EvidenceDirection.NEUTRAL).model_dump(mode="json")
    payload["strength"] = ""
    payload["reliability"] = ""

    card = EvidenceCard.model_validate(payload)

    assert card.strength == 0.5
    assert card.reliability == 0.5


def test_card_confidence_normalizes_common_alternate_scales() -> None:
    payload = _card("E1", "Grounded fact.", EvidenceDirection.NEUTRAL).model_dump(mode="json")
    payload["strength"] = 4
    payload["reliability"] = 80

    card = EvidenceCard.model_validate(payload)

    assert card.strength == 0.8
    assert card.reliability == 0.8


def test_numeric_evidence_period_is_normalized_to_string() -> None:
    payload = _card("E1", "Grounded fact.", EvidenceDirection.NEUTRAL).model_dump(mode="json")
    payload["period"] = 2025

    card = EvidenceCard.model_validate(payload)

    assert card.period == "2025"


def test_malformed_metric_string_fragments_are_dropped() -> None:
    payload = _card("E1", "Grounded fact.", EvidenceDirection.NEUTRAL).model_dump(mode="json")
    payload["metrics"] = [
        {"name": "revenue", "value": 100, "unit": "KRW mn"},
        {"value": 30049, "unit": "KRW mn"},
        ',\n{',
        'evidence_id":"EVID-3',
    ]

    card = EvidenceCard.model_validate(payload)

    assert len(card.metrics) == 1
    assert card.metrics[0].name == "revenue"

    payload["metrics"] = ""
    assert EvidenceCard.model_validate(payload).metrics == []


def test_evidence_type_and_dcf_links_normalize_provider_containers() -> None:
    payload = _card("E1", "Grounded fact.", EvidenceDirection.NEUTRAL).model_dump(mode="json")
    payload["evidence_type"] = ["FORECAST", "DERIVED_METRIC"]
    payload["dcf_links"] = {"items": [{"name": "REVENUE"}, {"type": "OTHER"}]}

    card = EvidenceCard.model_validate(payload)

    assert card.evidence_type == EvidenceType.OTHER
    assert card.dcf_links == [DcfLink.REVENUE]


def test_empty_mechanism_values_are_normalized_to_empty_list() -> None:
    payload = _card("E1", "Grounded fact.", EvidenceDirection.NEUTRAL).model_dump(mode="json")
    payload["mechanism"] = {}

    card = EvidenceCard.model_validate(payload)

    assert card.mechanism == []

    payload["mechanism"] = [{"text": "Customer workflow integration creates friction."}]

    card = EvidenceCard.model_validate(payload)

    assert card.mechanism == ["Customer workflow integration creates friction."]

    payload["mechanism"] = ""

    card = EvidenceCard.model_validate(payload)

    assert card.mechanism == []


def test_market_demand_is_not_kept_as_company_market_share() -> None:
    card = _card("E1", "외국인 피부과 환자가 전년 대비 증가했다.", EvidenceDirection.MOAT_POSITIVE)
    card.evidence_type = EvidenceType.MARKET_SHARE

    normalized = normalize_card_semantics(card)

    assert normalized.evidence_type == EvidenceType.MARKET_DEMAND
    assert normalized.economic_scope == EconomicScope.INDUSTRY
    assert normalized.direction == EvidenceDirection.NEUTRAL
    assert normalized.forward_driver_type == ForwardDriverType.MARKET_GROWTH
    assert normalized.dcf_links == [DcfLink.REVENUE]


def test_recurring_category_treatment_is_not_customer_retention() -> None:
    card = _card("E1", "스킨부스터는 3~6개월마다 재시술이 필요하다.", EvidenceDirection.MOAT_POSITIVE)
    card.evidence_type = EvidenceType.CUSTOMER_RETENTION

    normalized = normalize_card_semantics(card)

    assert normalized.evidence_type == EvidenceType.CATEGORY_RECURRING_DEMAND
    assert normalized.economic_scope == EconomicScope.PRODUCT_CATEGORY
    assert normalized.direction == EvidenceDirection.NEUTRAL
    assert normalized.forward_driver_type == ForwardDriverType.VOLUME


def test_utilization_evidence_is_promoted_to_forward_driver_card() -> None:
    card = _card("E1", "2공장 가동률은 61%에서 110%로 상승했다.", EvidenceDirection.NEUTRAL)
    normalized = normalize_card_semantics(card)

    drivers = build_forward_driver_cards([normalized])

    assert normalized.forward_driver_type == ForwardDriverType.UTILIZATION
    assert normalized.dcf_links == [DcfLink.REVENUE, DcfLink.CAPEX]
    assert len(drivers) == 1
    assert drivers[0].source_evidence_id == "E1"
