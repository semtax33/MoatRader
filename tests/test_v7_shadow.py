from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from pathlib import Path

import pytest
from pydantic import ValidationError

from moatrader.canonical.models import SourceType
from moatrader.valuation.base import ValuationMethod
from moatrader.experiments.shadow import (
    ShadowCompanySignal,
    ShadowRankStatus,
    ShadowSourceReference,
    ExpectationGapResearchContract,
    seal_shadow_snapshot,
)


SEOUL = timezone(timedelta(hours=9))
SIGNAL_AT = datetime(2026, 8, 18, 9, tzinfo=SEOUL)


def _contract() -> ExpectationGapResearchContract:
    return ExpectationGapResearchContract.create(
        created_at=SIGNAL_AT,
        parent_v6_contract_file_sha256="a" * 64,
        parent_v6_contract_payload_sha256="b" * 64,
        parent_engineering_input_sha256={"routing.csv": "c" * 64},
        expected_universe_count=1,
        scheduled_signal_at=[SIGNAL_AT, SIGNAL_AT + timedelta(days=7)],
    )


def _signal(*, source_at: datetime = SIGNAL_AT) -> ShadowCompanySignal:
    return ShadowCompanySignal(
        ticker="000001",
        signal_at=SIGNAL_AT,
        valuation_method=ValuationMethod.ECONOMIC_FCFF,
        economic_archetype="CAP_COMPOUNDER",
        status=ShadowRankStatus.VALID,
        market_price=D("100"),
        fair_value_per_share=D("125"),
        expectation_gap=D("0.25"),
        rank_percentile=D("90"),
        source_references=[
            ShadowSourceReference(
                document_id="DART:1",
                source_type=SourceType.DART,
                available_at=source_at,
                source_sha256="d" * 64,
            )
        ],
    )


def test_shadow_contract_and_snapshot_are_hashed_and_immutable(tmp_path: Path) -> None:
    contract = _contract()
    output = tmp_path / "snapshot.json"
    snapshot = seal_shadow_snapshot(
        contract=contract,
        signal_at=SIGNAL_AT,
        sealed_at=SIGNAL_AT + timedelta(minutes=5),
        signals=[_signal()],
        output_path=output,
    )

    assert snapshot.return_data_accessed is False
    assert json.loads(output.read_text(encoding="utf-8"))["snapshot_sha256"] == snapshot.snapshot_sha256
    with pytest.raises(FileExistsError, match="immutable"):
        seal_shadow_snapshot(
            contract=contract,
            signal_at=SIGNAL_AT,
            sealed_at=SIGNAL_AT + timedelta(minutes=6),
            signals=[_signal()],
            output_path=output,
        )


def test_shadow_snapshot_rejects_future_source() -> None:
    with pytest.raises(ValueError, match="future-source leakage"):
        seal_shadow_snapshot(
            contract=_contract(),
            signal_at=SIGNAL_AT,
            sealed_at=SIGNAL_AT + timedelta(minutes=5),
            signals=[_signal(source_at=SIGNAL_AT + timedelta(seconds=1))],
            output_path=Path("unused-v7-shadow.json"),
        )


def test_shadow_signal_contract_rejects_return_field() -> None:
    payload = _signal().model_dump(mode="json")
    payload["forward_return_77d"] = "0.10"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ShadowCompanySignal.model_validate(payload)
